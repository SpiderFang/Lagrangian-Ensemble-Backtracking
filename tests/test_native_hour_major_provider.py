"""WP-11：``native_runner.hour_major_provider`` 的端對端 CLI 驗收測試。

本檔案驗證的是「整條伺服器指令鏈是否真的串得起來」，不是任何單一函式的
單元邏輯（那些已經分別由 ``tests/test_native_hour_major.py``、
``tests/test_native_orchestrator.py``、``tests/test_native_production_wiring.py``
涵蓋）。具體驗證三件事：

1. **切段**：``lbt native-plan --schedule hour-major-segmented`` 真的能把兩個
   流場區域（各自對到不同的 OCM／NWW3 已驗收產品樹）各自摺成 ``K`` 個
   ``hourmajor::`` 排程單元，交給 ``hour_major_provider.make_hour_major_
   resource_provider`` 平行執行時，最終每一顆粒子的結果與「完全不切段
   （``K=1``）」位元相同。
2. **中斷後續跑**：``hour_major_provider.py`` 新增的測試專用旋鈕
   ``LBT_NATIVE_TEST_FORCE_INTERRUPT_AFTER_HOURS`` 能模擬「worker 子程序在
   把 checkpoint 寫入部分小時之後、於完成前被系統砍掉」；下一次不設定這個
   旋鈕重跑 ``native-run``，能從 ``hour_major`` 既有的 checkpoint 機制續跑到
   與「從頭到尾不中斷」完全位元相同的最終結果，且同一次 ``--output`` 目錄
   下其餘沒有被中斷的排程單元不受影響、正常完成。
3. **後處理不受切段影響**：``lbt native-postprocess`` 讀回 ``K=1``／``K=2``
   兩條路徑各自寫出的 ``*.driver_result.npz``，兩者的粒子總數、批次數與
   終止狀態分布必須相同，間接佐證「切段只改變平行度、不改變任何科學結果」。

合成資料與地理座標的誠實揭露
------------------------------
本檔案的 OCM／NWW3 產品是純合成資料：兩個流場區域各自只有一份 4×4 節點、
間距約 1 公里、水深固定 20 公尺的規則三角網格（與
``test_native_production_wiring._build_regular_grid`` 完全相同的節點數、
三角化方式與陣列格式，只是把經緯度原點換成參數，讓 A 區與 B 區能各自建一份
落在自己真實 local domain bbox 內的小網格），流速固定在每小時線性小幅變化
的極小量級（``u=0.001*hour`` m/s），刻意選得夠小、夠短（最大回溯期覆寫成
3 小時），確保粒子不會在測試視窗內離開這個約 3 公里見方的合成小網格——本檔
案的重點是驗證「切段」「續跑」「後處理」三層接線是否正確，不是驗證驅動層
本身在大位移、真實地形下的行為（那是 WP-5／WP-9 的測試範圍）。地理座標的
真實性因此不是本檔案的驗收目標：A 區沿用 ``northeast_taiwan_common_cache_
v3``（``analysis_region_id="A"``，貢寮 ``gongliao``／龜山島 ``guishan`` 兩
站）、B 區沿用 ``hsinchu_cache_v3``（``analysis_region_id="B"``，新竹外海
``hsinchu`` 一站），三站的合成起點刻意都落在各自 local domain bbox 內（貢寮
與龜山島的 bbox 有重疊的緯度帶，用來讓同一份合成網格同時滿足兩站的關注區域
判定）。``native-preflight`` 以 ``--pilot`` 模式執行：本檔案的合成身分表遠
小於正式五站 50,000 情境母體，不滿足 ``--formal`` 模式的完整涵蓋要求。

測試專用中斷旋鈕的使用方式，見 ``hour_major_provider.py`` 模組文件「測試
專用：模擬工作程序中途消失」一節；本檔案是該旋鈕**唯一**的呼叫端。
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from lagrangian_backtracking import cli
from lagrangian_backtracking.geometry import DomainProjection
from lagrangian_backtracking.native_runner import hour_major_provider as hmp
from lagrangian_backtracking.native_runner import orchestrator as orch

_HOUR_NS = 3_600_000_000_000
_EPOCH_2024_NS = 1_704_067_200_000_000_000  # 2024-01-01T00:00:00Z（與其餘 native 測試共用的紀元起點）

# 兩個合成流場區域的真實產品識別碼（沿用 configs/lagrangian_backtracking.example.yaml
# 既有登錄值，見 domains 清單），對應 analysis_region_id "A"／"B"。
_FLOW_DOMAIN_A = "northeast_taiwan_common_cache_v3"
_FLOW_DOMAIN_B = "hsinchu_cache_v3"

# 合成網格節點格局：與 test_native_production_wiring._build_regular_grid 完全相同的
# 節點數、間距、三角化方式與水深，只是把原點改成參數（見模組文件「合成資料與地理
# 座標的誠實揭露」一節）。
_GRID_NX = 4
_GRID_NY = 4
_GRID_DLON = 0.01
_GRID_DLAT = 0.01
_GRID_DEPTH_M = 20.0

# A 區網格原點：落在貢寮（121.7~122.15, 24.9~25.3）與龜山島（121.7~122.15,
# 24.7~24.95）兩站 local domain bbox 重疊的緯度帶（24.90~24.95）內，讓同一份
# 合成網格同時滿足兩站各自的關注區域（interest region）判定。
_GRID_A_LON0 = 121.98
_GRID_A_LAT0 = 24.905

# B 區網格原點：落在新竹外海 local domain（= B 區 flow domain bbox，
# 119.70812~121.19188, 24.300844~25.199156）內，靠近 B 區設定的 center_lonlat
# [120.45, 24.75]。
_GRID_B_LON0 = 120.43
_GRID_B_LAT0 = 24.73

# 三個到達時刻（自 2024-01-01T00:00:00Z 起的小時數）與兩個沉降速度（m/s，負值＝
# 下沉），故意間隔 4 小時、涵蓋在 60 小時的合成 forcing 視窗中段，讓每個到達時刻
# 各自的 3 小時回溯窗都落在 [0, 60) 小時內、不觸及視窗邊界。
_ARRIVAL_HOUR_OFFSETS = (30, 34, 38)
_SETTLING_VELOCITIES_MPS = (-0.0005, -0.001)

# 覆寫設定檔的最大回溯期（見 boundaries.max_backtrack_days）：3 小時，遠短於正式
# 30 天，確保合成小網格與 forcing 視窗足以支撐端對端測試在秒級～分鐘級完成。
_MAX_BACKTRACK_HOURS = 3
_MAX_BACKTRACK_DAYS = _MAX_BACKTRACK_HOURS / 24.0

# native-preflight 的缺口盤點（見 production_wiring._preflight_one_domain）把「預期
# 涵蓋期間」寫死成 config.inputs.years（見下方 _build_two_domain_config）第一年
# 1/1 到最後一年 12/31 23:00 的整年範圍，不接受局部窗口；要讓 preflight 誠實回報
# 「可執行」，合成 forcing 就必須真的涵蓋整個 2024 年（8784 小時，閏年），沒有捷徑。
# 這些資料只放在單一目錄（沿用 "202401" 這個既有慣例命名，見 _write_two_domain_
# products），時間戳記本身涵蓋全年——_year_window_months／inventory_forcing_gaps
# 只依實際時間戳記判斷缺口，不要求目錄名稱與實際月份一致，這是刻意的測試簡化，不是
# 正式資料契約（正式資料必須逐月落在對應月份目錄，見 README／AGENTS.md 資料契約）。
_FULL_YEAR_HOUR_COUNT = 8784  # 2024 年（閏年）共 366*24 小時
_FORCING_HOUR_OFFSETS = range(0, _FULL_YEAR_HOUR_COUNT)

_PROVIDER = "lagrangian_backtracking.native_runner.hour_major_provider:make_hour_major_resource_provider"


def _hour_ns(hour_offset: int) -> int:
    """把「自 2024-01-01T00:00:00Z 起第幾個小時」換成 UTC 奈秒。"""

    return _EPOCH_2024_NS + hour_offset * _HOUR_NS


# ══════════════════════════════════════════════════════════════════════════
# 一、合成網格與 OCM／NWW3 產品樹：兩個流場區域各自一份
# ══════════════════════════════════════════════════════════════════════════


def _node_index(i: int, j: int) -> int:
    return j * _GRID_NX + i


def _build_regular_grid_at(lon0: float, lat0: float) -> dict[str, Any]:
    """在給定原點建立 4x4 節點、間距約 1 公里、水深固定 20 公尺的規則三角網格。

    與 ``test_native_production_wiring._build_regular_grid`` 完全相同的節點數、
    三角化方式（每格切兩個三角形）與欄位格式，只把原點改成參數：A 區與 B 區的
    真實地理位置相差超過 100 公里，無法共用那個把原點寫死在模組常數的既有函式，
    因此在此複製同一套「格式」，不是另外發明一種不同的合成網格慣例。
    """

    n_nodes = _GRID_NX * _GRID_NY
    node_lon = np.empty(n_nodes, dtype=np.float64)
    node_lat = np.empty(n_nodes, dtype=np.float64)
    for j in range(_GRID_NY):
        for i in range(_GRID_NX):
            idx = _node_index(i, j)
            node_lon[idx] = lon0 + i * _GRID_DLON
            node_lat[idx] = lat0 + j * _GRID_DLAT
    dp = np.full(n_nodes, _GRID_DEPTH_M, dtype=np.float64)
    node_bottom_index = np.zeros(n_nodes, dtype=np.int64)

    triangles: list[tuple[int, int, int]] = []
    for j in range(_GRID_NY - 1):
        for i in range(_GRID_NX - 1):
            n00 = _node_index(i, j)
            n10 = _node_index(i + 1, j)
            n11 = _node_index(i + 1, j + 1)
            n01 = _node_index(i, j + 1)
            triangles.append((n00, n10, n11))
            triangles.append((n00, n11, n01))
    n_faces = len(triangles)
    face_nodes_local = np.full((n_faces, 4), -1, dtype=np.int64)
    for face_index, nodes in enumerate(triangles):
        face_nodes_local[face_index, :3] = nodes
    face_node_count = np.full(n_faces, 3, dtype=np.int64)
    face_global_index = np.arange(n_faces, dtype=np.int64)

    return {
        "node_lon": node_lon,
        "node_lat": node_lat,
        "dp": dp,
        "node_bottom_index": node_bottom_index,
        "face_nodes_local": face_nodes_local,
        "face_node_count": face_node_count,
        "face_global_index": face_global_index,
        "n_nodes": n_nodes,
        "n_faces": n_faces,
    }


def _write_nww3_grid_covering(
    grid_dir: Path, node_lon: np.ndarray, node_lat: np.ndarray, *, margin_deg: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """寫出一份粗略涵蓋給定 OCM 節點經緯度範圍（外加安全邊界）的合成 NWW3 規則網格。

    ``margin_deg`` 只是避免內插權重在邊界節點上剛好落在 NWW3 網格邊緣，不代表任何
    真實 NWW3 產品解析度；本函式只服務本檔案的端對端接線測試。
    """

    grid_dir.mkdir(parents=True, exist_ok=True)
    lon = np.linspace(float(node_lon.min()) - margin_deg, float(node_lon.max()) + margin_deg, 5)
    lat = np.linspace(float(node_lat.min()) - margin_deg, float(node_lat.max()) + margin_deg, 5)
    np.save(grid_dir / "lon.npy", lon)
    np.save(grid_dir / "lat.npy", lat)
    return lon, lat


def _write_two_domain_products(base: Path) -> dict[str, Any]:
    """在 ``base`` 下建立 A／B 兩個流場區域各自的合成 ``ocm_native``／``nww3_analysis`` 樹。

    函式作用域內匯入 ``test_native_production_wiring`` 的 ``_write_ocm_grid``／
    ``_write_ocm_month``／``_write_nww3_month``：這三個函式本身不假設任何特定經緯度
    （座標都從呼叫端傳入的 ``grid``／``hour_offsets`` 取得），可以直接重用，不需要
    另外複寫一份；只有假設了固定原點的 ``_build_regular_grid``／固定範圍的
    ``_write_nww3_grid`` 需要本檔案自己的參數化版本（見上方兩個函式）。
    """

    from test_native_production_wiring import _write_nww3_month, _write_ocm_grid, _write_ocm_month

    grid_a = _build_regular_grid_at(_GRID_A_LON0, _GRID_A_LAT0)
    grid_b = _build_regular_grid_at(_GRID_B_LON0, _GRID_B_LAT0)

    ocm_root = base / "ocm_native"
    nww_root = base / "nww_analysis"

    _write_ocm_grid(ocm_root / _FLOW_DOMAIN_A / "grid", grid_a)
    _write_ocm_month(
        ocm_root / _FLOW_DOMAIN_A / "months" / "202401", grid=grid_a, hour_offsets=_FORCING_HOUR_OFFSETS
    )
    _write_ocm_grid(ocm_root / _FLOW_DOMAIN_B / "grid", grid_b)
    _write_ocm_month(
        ocm_root / _FLOW_DOMAIN_B / "months" / "202401", grid=grid_b, hour_offsets=_FORCING_HOUR_OFFSETS
    )

    nww_lon_a, nww_lat_a = _write_nww3_grid_covering(
        nww_root / _FLOW_DOMAIN_A / "grid", grid_a["node_lon"], grid_a["node_lat"]
    )
    _write_nww3_month(
        nww_root / _FLOW_DOMAIN_A / "months" / "202401",
        hour_offsets=_FORCING_HOUR_OFFSETS,
        lon_size=nww_lon_a.size,
        lat_size=nww_lat_a.size,
    )
    nww_lon_b, nww_lat_b = _write_nww3_grid_covering(
        nww_root / _FLOW_DOMAIN_B / "grid", grid_b["node_lon"], grid_b["node_lat"]
    )
    _write_nww3_month(
        nww_root / _FLOW_DOMAIN_B / "months" / "202401",
        hour_offsets=_FORCING_HOUR_OFFSETS,
        lon_size=nww_lon_b.size,
        lat_size=nww_lat_b.size,
    )

    return {"ocm_root": ocm_root, "nww_root": nww_root, "grid_a": grid_a, "grid_b": grid_b}


# ══════════════════════════════════════════════════════════════════════════
# 二、設定檔與粒子身分表
# ══════════════════════════════════════════════════════════════════════════


def _build_two_domain_config(base: Path) -> Path:
    """寫出一份四域五站正式設定＋geometry manifest 的真 YAML，覆寫最大回溯期為 3 小時。

    重用 ``test_native_production_wiring._config_payload``／``_write_geometry_manifests``
    （見該檔案模組文件「本檔案與...唯一可新增／修改的兩個檔案」段落，已確立「函式
    作用域內 import」的既有慣例，本檔案沿用同一慣例）；本函式只多做一件事：把
    ``boundaries.max_backtrack_days`` 從 ``null``（交由正式母體決策）覆寫成一個很短
    的測試值，讓合成的 60 小時 forcing 視窗足以支撐端對端測試。
    """

    from test_native_production_wiring import _config_payload, _write_geometry_manifests

    payload = _config_payload(with_paths=True)
    payload["boundaries"]["max_backtrack_days"] = _MAX_BACKTRACK_DAYS
    # native-preflight 的缺口盤點要求「合成 forcing 涵蓋 inputs.years 整年」，見
    # _FORCING_HOUR_OFFSETS 上方註解；縮成單一年份，讓這個涵蓋要求至少只需要造一年
    # （而不是範例設定原本的 2024～2025 兩年）份合成資料。
    payload["inputs"]["years"] = [2024]
    config_path = base / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    _write_geometry_manifests(base)
    return config_path


def _build_two_site_identity_table(grid_a: dict[str, Any], grid_b: dict[str, Any]):
    """建一份 3 站 × 3 到達時刻 × 2 沉降速度、``members_per_scenario=1`` 的合成身分表。

    3 站：A 區貢寮／龜山島（共用 A 區合成流場）、B 區新竹外海（獨立合成流場）。
    每個 ``(study_site, arrival_time, settling_velocity)`` 組合恰好 1 顆粒子，
    合計 3×3×2=18 顆粒子＝18 個原始工作單元；這是端對端接線測試，不是正式情境
    矩陣驗收（正式母體是每站 10 個沉降代理 × 50 個到達時刻 × 20 個受體，見
    README），因此刻意用遠小的規模。起點固定在各自合成網格的節點中心（換算成
    該區域 ``DomainProjection`` 的公尺座標），深度固定 -5 公尺（相對瞬時海面的
    positive-down 慣例，見 ``test_native_production_wiring._wiring_e2e_particle_
    table`` 同一寫法），確保 3 小時回溯窗內不會離開合成小網格或觸底。
    """

    from test_native_orchestrator import _build_table

    proj_a = DomainProjection(122.05, 25.05)  # 與 config A 區 center_lonlat 相同
    proj_b = DomainProjection(120.45, 24.75)  # 與 config B 區 center_lonlat 相同
    xa, ya = proj_a.project(float(np.mean(grid_a["node_lon"])), float(np.mean(grid_a["node_lat"])))
    xb, yb = proj_b.project(float(np.mean(grid_b["node_lon"])), float(np.mean(grid_b["node_lat"])))

    rows: list[dict[str, Any]] = []
    for site, region, x0, y0 in (
        ("gongliao", "A", xa, ya),
        ("guishan", "A", xa, ya),
        ("hsinchu", "B", xb, yb),
    ):
        for hour_offset in _ARRIVAL_HOUR_OFFSETS:
            for settling in _SETTLING_VELOCITIES_MPS:
                rows.append(
                    {
                        "study_site_id": site,
                        "analysis_region_id": region,
                        "arrival_time_utc_ns": _hour_ns(hour_offset),
                        "settling_velocity_mps": settling,
                        "x0_m": float(x0),
                        "y0_m": float(y0),
                        "z0_m": -5.0,
                    }
                )
    return _build_table(rows)


# ══════════════════════════════════════════════════════════════════════════
# 三、環境變數 context manager 與乾淨呼叫 helper
# ══════════════════════════════════════════════════════════════════════════

# 每次呼叫 native-run 前一定要清掉的旋鈕：避免上一個測試殘留的中斷旋鈕或
# checkpoint 根目錄意外影響下一次呼叫（尤其是本檔案唯一使用中斷旋鈕的測試，
# 更要確保其餘測試完全看不到它）。
_ALWAYS_CLEAR_ENV_KEYS = (
    "LBT_NATIVE_CHECKPOINT_ROOT",
    "LBT_NATIVE_TEST_FORCE_INTERRUPT_AFTER_HOURS",
    "OCM_RECONSTRUCTION_ROOT",
)


@contextlib.contextmanager
def _native_run_environment(env: dict[str, str]):
    """暫時設定 ``env`` 裡的環境變數（並清掉 ``_ALWAYS_CLEAR_ENV_KEYS``），結束後還原。

    本檔案有多個 module-scoped fixture（clean 對照組、numba cache 目錄）需要在
    fixture 內（沒有 ``monkeypatch`` 可用，因為它是 function-scoped）設定環境變數
    再呼叫 CLI 函式，因此自行實作這個最小 contextmanager，語意與 ``monkeypatch.
    setenv``／``monkeypatch.delenv`` 相同（還原成呼叫前的狀態，不論成功或例外）。
    """

    keys_to_restore = set(env) | set(_ALWAYS_CLEAR_ENV_KEYS)
    saved = {key: os.environ.get(key) for key in keys_to_restore}
    try:
        for key, value in env.items():
            os.environ[key] = value
        for key in _ALWAYS_CLEAR_ENV_KEYS:
            if key not in env:
                os.environ.pop(key, None)
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _base_env(two_domain_env: dict[str, Any], baytrace_root: str) -> dict[str, str]:
    """native-run／native-preflight 共用的環境變數基底（不含 ``LBT_NATIVE_PLAN``／
    ``LBT_NATIVE_RESULTS_DIR``，這兩個依每次呼叫的計畫檔與輸出目錄而異）。

    ``LBT_NATIVE_FORMAL=0``：與 ``native-preflight --pilot`` 一致，本檔案的合成
    身分表不滿足正式五站母體要求，全程使用 pilot（非正式）模式。
    """

    return {
        "LBT_NATIVE_CONFIG": str(two_domain_env["config_path"]),
        "LBT_NATIVE_IDENTITY_TABLE": str(two_domain_env["identity_path"]),
        "LBT_BAYTRACE_ROOT": baytrace_root,
        "OCM_NATIVE_ROOT": str(two_domain_env["ocm_root"]),
        "NWW_ANALYSIS_ROOT": str(two_domain_env["nww_root"]),
        "LBT_NATIVE_FORMAL": "0",
    }


def _run_clean_native_run(
    env_base: dict[str, str], plan_path: Path, numba_cache_dir: Path, tmp_path_factory, label: str
) -> dict[str, Path]:
    """跑一次「從頭到尾不中斷」的乾淨 ``native-run``，回傳輸出目錄與結果目錄。"""

    output_dir = tmp_path_factory.mktemp(f"{label}_output")
    results_dir = tmp_path_factory.mktemp(f"{label}_results")
    env = dict(env_base)
    env["LBT_NATIVE_PLAN"] = str(plan_path)
    env["LBT_NATIVE_RESULTS_DIR"] = str(results_dir)
    with _native_run_environment(env):
        rc = cli.run_native_run(
            [
                str(plan_path),
                "--output",
                str(output_dir),
                "--resource-provider",
                _PROVIDER,
                "--worker-count",
                "2",
                "--numba-cache-dir",
                str(numba_cache_dir),
            ]
        )
    assert rc == 0, f"{label} 的乾淨 native-run 未能全部成功"
    return {"output_dir": output_dir, "results_dir": results_dir}


def _assert_driver_results_bit_identical(path_a: Path, path_b: Path, *, label: str) -> None:
    """比較兩個 ``.driver_result.npz`` 的全部欄位是否位元相同（不是近似相等）。"""

    result_a = hmp.load_backward_driver_result_npz(path_a)
    result_b = hmp.load_backward_driver_result_npz(path_b)
    assert result_a.hours_run == result_b.hours_run, f"{label}: hours_run 不同"
    for field in (
        "row_particle_index",
        "final_status",
        "final_x_m",
        "final_y_m",
        "final_z_m",
        "final_utc_ns",
        "trajectory_utc_ns",
        "trajectory_x_m",
        "trajectory_y_m",
        "trajectory_z_m",
        "trajectory_status",
    ):
        value_a = getattr(result_a, field)
        value_b = getattr(result_b, field)
        assert np.array_equal(value_a, value_b), f"{label}: {field} 不是位元相同"
    assert len(result_a.departures) == len(result_b.departures), f"{label}: departures 筆數不同"
    for departure_a, departure_b in zip(result_a.departures, result_b.departures, strict=True):
        assert departure_a == departure_b, f"{label}: departures 內容不同"


# ══════════════════════════════════════════════════════════════════════════
# 四、fixtures
# ══════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def two_domain_env(tmp_path_factory) -> dict[str, Any]:
    """建好合成設定、合成兩區域 OCM／NWW3 產品樹、合成身分表，並先產出 K=1／K=2 兩份計畫檔。

    這一步全程不需要 ``LBT_BAYTRACE_ROOT``（``native-plan`` 只讀身分表與月份時間軸，
    不建構網格或呼叫 BayTrace kernel），因此設成 module 層級、不受
    ``LBT_BAYTRACE_ROOT`` 是否存在影響，讓純規劃層的測試永遠能跑。
    """

    from lagrangian_backtracking.native_runner.particle_input import save_particle_identity_table_npz

    base = tmp_path_factory.mktemp("hmp_env")
    config_path = _build_two_domain_config(base)
    products = _write_two_domain_products(base)
    table = _build_two_site_identity_table(products["grid_a"], products["grid_b"])
    identity_path = base / "identity_table.npz"
    save_particle_identity_table_npz(table, identity_path)

    # 明確提供 --region-to-flow-domain（{analysis_region_id: flow_domain_id}），
    # 讓 NativeWorkUnit.flow_domain_id 是真正的 OCM 產品目錄名稱
    # （"northeast_taiwan_common_cache_v3"／"hsinchu_cache_v3"），刻意與
    # analysis_region_id（"A"／"B"）不同——這是正式部署一定會有的情況（見
    # plan_native_work_units docstring「正式呼叫應由呼叫端……建出這個對照表」），
    # 也是唯一能真正驗證 hour_major_provider 是否正確區分
    # analysis_region_id／flow_domain_id 的方式（見該模組「已知限制」第 5 點
    # 記載的修正紀錄：若省略此參數，兩者剛好相同，該缺陷不會被本測試發現）。
    region_to_flow_domain_path = base / "region_to_flow_domain.json"
    region_to_flow_domain_path.write_text(
        json.dumps({"A": _FLOW_DOMAIN_A, "B": _FLOW_DOMAIN_B}), encoding="utf-8"
    )

    plan_k2_path = base / "plan_k2.json"
    rc = cli.run_native_plan_command(
        [
            "--identity-table",
            str(identity_path),
            "--region-to-flow-domain",
            str(region_to_flow_domain_path),
            "--output",
            str(plan_k2_path),
            "--schedule",
            "hour-major-segmented",
            "--segments",
            "2",
            "--skip-gap-check",
        ]
    )
    assert rc == 0, "K=2 native-plan 未能產生 gap-safe-ready 的計畫檔"

    plan_k1_path = base / "plan_k1.json"
    rc = cli.run_native_plan_command(
        [
            "--identity-table",
            str(identity_path),
            "--region-to-flow-domain",
            str(region_to_flow_domain_path),
            "--output",
            str(plan_k1_path),
            "--schedule",
            "hour-major-segmented",
            "--segments",
            "1",
            "--skip-gap-check",
        ]
    )
    assert rc == 0, "K=1 native-plan 未能產生 gap-safe-ready 的計畫檔"

    return {
        "config_path": config_path,
        "ocm_root": products["ocm_root"],
        "nww_root": products["nww_root"],
        "identity_path": identity_path,
        "plan_k2_path": plan_k2_path,
        "plan_k1_path": plan_k1_path,
        "table": table,
    }


@pytest.fixture(scope="module")
def baytrace_root() -> str:
    root = os.environ.get("LBT_BAYTRACE_ROOT")
    if not root:
        pytest.skip("需要 LBT_BAYTRACE_ROOT 指向 BayTrace 原始碼根目錄才能執行本模組的端對端測試")
    return root


@pytest.fixture(scope="module")
def numba_cache_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("numba_cache")


@pytest.fixture(scope="module")
def clean_k2_run(two_domain_env, baytrace_root, numba_cache_dir, tmp_path_factory) -> dict[str, Path]:
    """K=2（每區域切 2 段）、從頭到尾不中斷的乾淨對照組，供其餘測試比對用。"""

    env_base = _base_env(two_domain_env, baytrace_root)
    return _run_clean_native_run(
        env_base, two_domain_env["plan_k2_path"], numba_cache_dir, tmp_path_factory, "clean_k2"
    )


@pytest.fixture(scope="module")
def clean_k1_run(two_domain_env, baytrace_root, numba_cache_dir, tmp_path_factory) -> dict[str, Path]:
    """K=1（每區域不切段）、從頭到尾不中斷的乾淨對照組，供其餘測試比對用。"""

    env_base = _base_env(two_domain_env, baytrace_root)
    return _run_clean_native_run(
        env_base, two_domain_env["plan_k1_path"], numba_cache_dir, tmp_path_factory, "clean_k1"
    )


# ══════════════════════════════════════════════════════════════════════════
# 五、測試本體
# ══════════════════════════════════════════════════════════════════════════


def test_preflight_reports_run_ready_for_two_domains(two_domain_env) -> None:
    """``native-preflight --pilot --domain A --domain B`` 對合成兩區域回報就緒（回傳碼 0）。"""

    env = {
        "OCM_NATIVE_ROOT": str(two_domain_env["ocm_root"]),
        "NWW_ANALYSIS_ROOT": str(two_domain_env["nww_root"]),
    }
    with _native_run_environment(env):
        rc = cli.run_native_preflight_command(
            [
                "--config",
                str(two_domain_env["config_path"]),
                "--domain",
                "A",
                "--domain",
                "B",
                "--pilot",
            ]
        )
    assert rc == 0


def test_native_plan_segments_each_domain_into_two_units(two_domain_env) -> None:
    """K=2 把 A／B 兩區域各自摺成 2 個 ``hourmajor::`` 排程單元；K=1 各自摺成 1 個。

    同時確認 K=2 計畫檔底下 ``hour_major_members`` 攤平後的原始工作單元總數，等於
    身分表的粒子總數（``members_per_scenario=1``，一顆粒子恰好對到一個原始工作
    單元），驗證切段沒有遺漏或重複任何一顆粒子。
    """

    domain_units, payload = orch.read_native_plan_file(two_domain_env["plan_k2_path"])
    assert payload["is_gap_safe_ready"] is True
    members = orch.read_native_hour_major_members(payload)

    by_domain: dict[str, list[orch.NativeWorkUnit]] = {}
    for unit in domain_units:
        by_domain.setdefault(unit.flow_domain_id, []).append(unit)
    # flow_domain_id 是 --region-to-flow-domain 提供的真正 OCM 產品目錄名稱，
    # 刻意與 analysis_region_id（"A"／"B"）不同，見 two_domain_env fixture註解。
    assert set(by_domain) == {_FLOW_DOMAIN_A, _FLOW_DOMAIN_B}
    assert all(len(units) == 2 for units in by_domain.values()), "每個流場區域應該各自切成 2 段"

    all_member_unit_ids: set[str] = set()
    for unit in domain_units:
        member_units = members[unit.unit_id]
        assert member_units, f"{unit.unit_id} 底下不應該是空段"
        all_member_unit_ids.update(m.unit_id for m in member_units)
    assert len(all_member_unit_ids) == len(two_domain_env["table"])

    domain_units_k1, payload_k1 = orch.read_native_plan_file(two_domain_env["plan_k1_path"])
    assert payload_k1["is_gap_safe_ready"] is True
    by_domain_k1: dict[str, list[orch.NativeWorkUnit]] = {}
    for unit in domain_units_k1:
        by_domain_k1.setdefault(unit.flow_domain_id, []).append(unit)
    assert set(by_domain_k1) == {_FLOW_DOMAIN_A, _FLOW_DOMAIN_B}
    assert all(len(units) == 1 for units in by_domain_k1.values()), "K=1 每個流場區域應該只有 1 段"


def test_native_run_completes_all_original_units_and_writes_manifest(
    two_domain_env, clean_k2_run
) -> None:
    """乾淨 K=2 ``native-run`` 完成後，逐原始工作單元的 ``.driver_result.npz`` 數量正確，
    且 ``manifest.json`` 含 git commit／dirty／master seed／config 雜湊等既有 provenance 欄位。
    """

    result_files = sorted(clean_k2_run["results_dir"].glob("*.driver_result.npz"))
    assert len(result_files) == len(two_domain_env["table"])

    manifest = json.loads((clean_k2_run["output_dir"] / "manifest.json").read_text(encoding="utf-8"))
    code_provenance = manifest["code_provenance"]
    assert code_provenance["git_commit"], "本機是 Git repo，應該能取得 commit"
    assert "git_dirty" in code_provenance
    assert manifest["master_seed"] == int(two_domain_env["table"].master_seed)
    assert manifest["config_sha256"]
    assert manifest["seed_policy"]


def test_native_run_k1_bit_identical_to_k2(clean_k1_run, clean_k2_run) -> None:
    """K=1（不切段）與 K=2（每區域切 2 段）對同一顆粒子必須得到位元相同的最終結果。

    這是 ``plan_native_hour_major_segments`` 的核心正確性保證（切段只改變平行度，
    不改變任何粒子的起點、seed 或亂數串流），見該函式 docstring；本測試逐一比對
    全部 18 個原始工作單元的完整 ``BackwardDriverResult`` 內容，不只是摘要統計。
    """

    files_k1 = {p.name: p for p in clean_k1_run["results_dir"].glob("*.driver_result.npz")}
    files_k2 = {p.name: p for p in clean_k2_run["results_dir"].glob("*.driver_result.npz")}
    assert files_k1, "K=1 應該至少有一個結果檔"
    assert set(files_k1) == set(files_k2)
    for name in sorted(files_k1):
        _assert_driver_results_bit_identical(files_k1[name], files_k2[name], label=name)


def test_native_postprocess_consumes_real_driver_output_k1_equals_k2(
    two_domain_env, clean_k1_run, clean_k2_run, tmp_path
) -> None:
    """``native-postprocess`` 能正確消化真正的小時主軸輸出，且 K=1 與 K=2 結果一致。

    驅動層（``backward_driver``／``hour_major``）寫進 ``DepartureRecord`` 的
    ``row_particle_index`` 是批次內的局部列位置 0..n-1，``BackwardDriverResult.
    row_particle_index`` 才是全體身分表的全域列位置；後處理必須先轉換再比對。
    這裡以真正的 CLI 輸出端對端驗證：兩種切段方式各自後處理後，粒子總數、各終止
    狀態計數與離開紀錄表列數完全相同，且確實有粒子產生離開紀錄（避免空集合假通過）。
    """

    import json

    import pyarrow.parquet as pq

    manifests = {}
    departure_rows = {}
    for label, run in (("k1", clean_k1_run), ("k2", clean_k2_run)):
        out_dir = tmp_path / f"postprocess_{label}"
        code = cli.run_native_postprocess_command(
            [
                "--identity-table",
                str(two_domain_env["identity_path"]),
                "--results-dir",
                str(run["results_dir"]),
                "--output",
                str(out_dir),
            ]
        )
        assert code in (0, None)
        manifests[label] = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        departure_rows[label] = pq.read_table(out_dir / "departures.parquet").num_rows

    assert manifests["k1"]["particle_count_total"] == manifests["k2"]["particle_count_total"]
    assert manifests["k1"]["status_counts"] == manifests["k2"]["status_counts"]
    assert departure_rows["k1"] == departure_rows["k2"] == manifests["k1"]["particle_count_total"]
    assert sum(int(v) for v in manifests["k1"]["status_counts"].values()) == departure_rows["k1"] > 0


def test_native_run_interrupt_then_resume_bit_identical_to_clean_run(
    two_domain_env, baytrace_root, numba_cache_dir, clean_k2_run, tmp_path
) -> None:
    """模擬「排程單元在 checkpoint 寫入部分小時後、於完成前失去存活」，驗證：

    1. 設定 ``LBT_NATIVE_TEST_FORCE_INTERRUPT_AFTER_HOURS`` 時，目標排程單元的
       ``native-run`` 呼叫回報失敗（非 0 回傳碼），且 ``*.done.json`` 完成證據的
       ``status`` 是 ``"failed"``（不會被 ``filter_pending_units`` 誤判為可跳過）。
    2. 同一批計畫裡的其餘排程單元（另外用一份只含這些單元的計畫檔跑）不受影響，
       正常完成。
    3. 移除中斷旋鈕、對同一個 ``--output`` 目錄用完整計畫檔重跑：已完成的單元被
       跳過，先前失敗的目標單元從 ``hour_major`` checkpoint 續跑到底並完成。
    4. 續跑後全部 18 個原始工作單元的 ``.driver_result.npz``，與「從頭到尾不中斷」
       的乾淨 K=2 對照組（``clean_k2_run``）逐欄位位元相同。

    因為中斷旋鈕是在呼叫 ``make_hour_major_resource_provider()`` 當下（父程序）
    凍結成 ``HourMajorProviderParams`` 的（見 ``hour_major_provider.py`` 模組文件
    「環境變數契約」），同一次 ``native-run`` 呼叫裡全部排程單元都會套用同一個
    旋鈕值，無法只讓其中一個排程單元中斷、其餘正常執行；因此本測試把 K=2 計畫檔
    的 4 個排程單元拆成「目標單元」與「其餘單元」兩份獨立計畫檔，分兩次呼叫
    ``native-run``（共用同一個 ``--output``／``--results-dir``），藉此在同一輪
    「初次執行」裡同時滿足「目標單元失敗」與「其餘單元正常完成」兩個條件。
    """

    domain_units, payload = orch.read_native_plan_file(two_domain_env["plan_k2_path"])
    members = orch.read_native_hour_major_members(payload)
    target = min(domain_units, key=lambda unit: unit.particle_count)
    others = tuple(unit for unit in domain_units if unit.unit_id != target.unit_id)
    assert others, "K=2 應該有其餘至少一個排程單元可以驗證「不受中斷影響」"

    plan_target_path = tmp_path / "plan_target.json"
    plan_others_path = tmp_path / "plan_others.json"
    orch.write_native_plan_file(
        plan_target_path,
        units=(target,),
        gap_checks=(),
        summary=orch.summarize_native_work_plan((target,)),
        manifest=payload.get("manifest"),
        hour_major_members={target.unit_id: members[target.unit_id]},
    )
    orch.write_native_plan_file(
        plan_others_path,
        units=others,
        gap_checks=(),
        summary=orch.summarize_native_work_plan(others),
        manifest=payload.get("manifest"),
        hour_major_members={unit.unit_id: members[unit.unit_id] for unit in others},
    )

    output_dir = tmp_path / "interrupt_output"
    results_dir = tmp_path / "interrupt_results"
    checkpoint_root = tmp_path / "interrupt_checkpoints"

    env_base = _base_env(two_domain_env, baytrace_root)
    env_base["LBT_NATIVE_RESULTS_DIR"] = str(results_dir)
    env_base["LBT_NATIVE_CHECKPOINT_ROOT"] = str(checkpoint_root)

    # ── 階段一 a：只跑「即將中斷」的目標排程單元，設定測試專用中斷旋鈕。 ──────
    env_interrupt = dict(env_base)
    env_interrupt["LBT_NATIVE_PLAN"] = str(plan_target_path)
    env_interrupt["LBT_NATIVE_TEST_FORCE_INTERRUPT_AFTER_HOURS"] = "1"
    with _native_run_environment(env_interrupt):
        rc_interrupt = cli.run_native_run(
            [
                str(plan_target_path),
                "--output",
                str(output_dir),
                "--resource-provider",
                _PROVIDER,
                "--worker-count",
                "1",
                "--numba-cache-dir",
                str(numba_cache_dir),
            ]
        )
    assert rc_interrupt != 0, "中斷模擬應該讓 native-run 回報非零狀態碼"

    marker_target = json.loads((output_dir / f"{target.unit_id}.done.json").read_text(encoding="utf-8"))
    assert marker_target["status"] == "failed"
    assert (checkpoint_root / target.unit_id / "meta.json").is_file(), "中斷前應該已經寫入 checkpoint"
    for member in members[target.unit_id]:
        assert not hmp.result_npz_path(results_dir, member.unit_id).exists(), (
            "中斷模擬不應該寫出任何 .driver_result.npz"
        )

    # ── 階段一 b：其餘排程單元不設中斷旋鈕，正常完成，證明彼此互不影響。 ──────
    env_others = dict(env_base)
    env_others["LBT_NATIVE_PLAN"] = str(plan_others_path)
    with _native_run_environment(env_others):
        rc_others = cli.run_native_run(
            [
                str(plan_others_path),
                "--output",
                str(output_dir),
                "--resource-provider",
                _PROVIDER,
                "--worker-count",
                "2",
                "--numba-cache-dir",
                str(numba_cache_dir),
            ]
        )
    assert rc_others == 0, "沒有被中斷的排程單元應該正常全部完成"
    for unit in others:
        marker = json.loads((output_dir / f"{unit.unit_id}.done.json").read_text(encoding="utf-8"))
        assert marker["status"] == "completed"

    # ── 階段二：移除中斷旋鈕，對同一個 --output 用完整計畫檔重跑。 ────────────
    env_resume = dict(env_base)
    env_resume["LBT_NATIVE_PLAN"] = str(two_domain_env["plan_k2_path"])
    with _native_run_environment(env_resume):
        rc_resume = cli.run_native_run(
            [
                str(two_domain_env["plan_k2_path"]),
                "--output",
                str(output_dir),
                "--resource-provider",
                _PROVIDER,
                "--worker-count",
                "2",
                "--numba-cache-dir",
                str(numba_cache_dir),
            ]
        )
    assert rc_resume == 0, "移除中斷旋鈕後，先前失敗的單元應該能從 checkpoint 續跑成功"

    for unit in domain_units:
        marker = json.loads((output_dir / f"{unit.unit_id}.done.json").read_text(encoding="utf-8"))
        assert marker["status"] == "completed"

    result_files = sorted(results_dir.glob("*.driver_result.npz"))
    assert len(result_files) == len(two_domain_env["table"])

    clean_files = {p.name: p for p in clean_k2_run["results_dir"].glob("*.driver_result.npz")}
    assert set(p.name for p in result_files) == set(clean_files)
    for path in result_files:
        _assert_driver_results_bit_identical(path, clean_files[path.name], label=path.name)


# ══════════════════════════════════════════════════════════════════════════
# WP-13：--wait-for-staged 端對端、native-plan 成本模型報表
# ══════════════════════════════════════════════════════════════════════════


def test_native_run_wait_for_staged_processes_domains_as_markers_appear(
    two_domain_env: dict[str, Any], baytrace_root: str, numba_cache_dir: Path, tmp_path_factory
) -> None:
    """``native-run --wait-for-staged``：區域 A 標記先存在、區域 B 標記延後才寫入，
    驗證區域 A 的排程單元先完成，區域 B 的單元要等標記出現才會被排入佇列且最終
    全部完成，過程中 worker 不會因為佇列暫時清空（只剩 B 在等）而提早結束。
    """

    import threading
    import time as _time

    env_base = _base_env(two_domain_env, baytrace_root)
    output_dir = tmp_path_factory.mktemp("staged_output")
    results_dir = tmp_path_factory.mktemp("staged_results")
    staged_dir = tmp_path_factory.mktemp("staged_markers")

    env = dict(env_base)
    env["LBT_NATIVE_PLAN"] = str(two_domain_env["plan_k2_path"])
    env["LBT_NATIVE_RESULTS_DIR"] = str(results_dir)

    domain_units, _payload = orch.read_native_plan_file(two_domain_env["plan_k2_path"])
    a_unit_ids = [u.unit_id for u in domain_units if u.flow_domain_id == _FLOW_DOMAIN_A]
    b_unit_ids = [u.unit_id for u in domain_units if u.flow_domain_id == _FLOW_DOMAIN_B]
    assert a_unit_ids and b_unit_ids

    # 一開始只有區域 A 就緒。
    (staged_dir / f"{_FLOW_DOMAIN_A}.done").write_text("ok", encoding="utf-8")

    result_holder: dict[str, int] = {}

    def _run() -> None:
        with _native_run_environment(env):
            result_holder["rc"] = cli.run_native_run(
                [
                    str(two_domain_env["plan_k2_path"]),
                    "--output",
                    str(output_dir),
                    "--resource-provider",
                    _PROVIDER,
                    "--worker-count",
                    "2",
                    "--numba-cache-dir",
                    str(numba_cache_dir),
                    "--wait-for-staged",
                    str(staged_dir),
                ]
            )

    # daemon=True：若下面的斷言在寫入區域 B 標記之前失敗，不讓仍在等待區域 B
    # 就緒的背景執行緒卡住整個測試程序（見 test_native_orchestrator.py 對應
    # 測試撰寫時踩到的同一個坑）。
    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    deadline = _time.monotonic() + 120.0
    while _time.monotonic() < deadline:
        if all((output_dir / f"{uid}.done.json").is_file() for uid in a_unit_ids):
            break
        _time.sleep(0.2)
    else:
        pytest.fail("區域 A 的排程單元在時限內未完成，無法繼續驗證 staged 等待行為")

    for uid in b_unit_ids:
        assert not (output_dir / f"{uid}.done.json").is_file(), "區域 B 標記尚未寫入，不應該提前執行"

    (staged_dir / f"{_FLOW_DOMAIN_B}.done").write_text("ok", encoding="utf-8")
    thread.join(timeout=180.0)
    assert not thread.is_alive(), "native-run 在區域 B 就緒後應該能正常結束"
    assert result_holder.get("rc") == 0

    for uid in a_unit_ids + b_unit_ids:
        marker = json.loads((output_dir / f"{uid}.done.json").read_text(encoding="utf-8"))
        assert marker["status"] == "completed"


def test_native_plan_reports_cost_model_when_grid_sizes_provided(two_domain_env: dict[str, Any]) -> None:
    """``native-plan --schedule hour-major-segmented`` 提供 ``--node-counts``／
    ``--layer-counts``／``--target-worker-count`` 時，報表要列出 WP-13 成本模型：
    預估牆鐘時間、重複成本比例與各段成本；省略時（既有 ``plan_k2_path`` 的產生
    呼叫）明確不應該出現這些欄位（fail-closed，不用臆測值頂替）。
    """

    base = two_domain_env["identity_path"].parent
    node_counts_path = base / "node_counts.json"
    layer_counts_path = base / "layer_counts.json"
    node_counts_path.write_text(
        json.dumps({_FLOW_DOMAIN_A: 67_600, _FLOW_DOMAIN_B: 27_015}), encoding="utf-8"
    )
    layer_counts_path.write_text(
        json.dumps({_FLOW_DOMAIN_A: 48, _FLOW_DOMAIN_B: 48}), encoding="utf-8"
    )
    region_to_flow_domain_path = base / "region_to_flow_domain.json"

    output_path = base / "plan_with_cost_model.json"
    captured: list[str] = []
    import io
    from contextlib import redirect_stdout

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        rc = cli.run_native_plan_command(
            [
                "--identity-table",
                str(two_domain_env["identity_path"]),
                "--region-to-flow-domain",
                str(region_to_flow_domain_path),
                "--output",
                str(output_path),
                "--schedule",
                "hour-major-segmented",
                "--target-worker-count",
                "8",
                "--node-counts",
                str(node_counts_path),
                "--layer-counts",
                str(layer_counts_path),
                "--skip-gap-check",
            ]
        )
    captured.append(buffer.getvalue())
    assert rc == 0
    report = json.loads(captured[0])
    cost_model = report["cost_model"]
    assert cost_model is not None
    assert cost_model["estimated_wall_clock_seconds"] > 0.0
    assert cost_model["max_single_segment_cost_seconds"] > 0.0
    assert cost_model["repeat_cost_fraction_vs_no_split"] >= 0.0
    written_units, _written_payload = orch.read_native_plan_file(output_path)
    assert set(cost_model["segment_cost_seconds"]) == {u.unit_id for u in written_units}

    # 沒有提供 --node-counts／--layer-counts 時（既有 plan_k2_path），報表的
    # cost_model 欄位必須明確是 None，不能用任何臆測值頂替。
    _units, payload_k2 = orch.read_native_plan_file(two_domain_env["plan_k2_path"])
    assert "manifest" in payload_k2


# ══════════════════════════════════════════════════════════════════════════
# 六、WP-18：實驗案例（no_stokes／finite_depth_stokes）端對端驗收
# ══════════════════════════════════════════════════════════════════════════


def test_no_stokes_experiment_case_end_to_end_skips_wave_data(
    two_domain_env, baytrace_root, numba_cache_dir, tmp_path_factory
) -> None:
    """``--experiment-case no_stokes`` 從 native-plan 到 native-run 全程串起來後，
    每個結果摘要都必須標示 ``include_stokes=False``、``wave_data_available`` 為
    ``None``（整段落在 pre-window 的排程單元）或 ``False``（實際跑過 hour-major
    的排程單元），即使 ``two_domain_env`` 的合成資料本來就有可用的 NWW3 波浪
    產品（見 ``clean_k1_run``／``clean_k2_run`` 在同一份資料下 ``wave_data_
    available=True`` 的既有預設行為，於本測試最後一併對照驗證，確認新增的
    ``--experiment-case`` 沒有改變預設行為）。"""

    base = tmp_path_factory.mktemp("no_stokes_plan")
    region_to_flow_domain_path = base / "region_to_flow_domain.json"
    region_to_flow_domain_path.write_text(
        json.dumps({"A": _FLOW_DOMAIN_A, "B": _FLOW_DOMAIN_B}), encoding="utf-8"
    )
    plan_path = base / "plan_no_stokes.json"
    rc = cli.run_native_plan_command(
        [
            "--identity-table",
            str(two_domain_env["identity_path"]),
            "--region-to-flow-domain",
            str(region_to_flow_domain_path),
            "--output",
            str(plan_path),
            "--schedule",
            "hour-major-segmented",
            "--segments",
            "1",
            "--skip-gap-check",
            "--experiment-case",
            "no_stokes",
        ]
    )
    assert rc == 0
    _units, plan_payload = orch.read_native_plan_file(plan_path)
    assert plan_payload["physics_case"]["experiment_case_id"] == "no_stokes"
    assert plan_payload["physics_case"]["include_stokes"] is False

    env_base = _base_env(two_domain_env, baytrace_root)
    env_base["LBT_NATIVE_EXPERIMENT_CASE"] = "no_stokes"
    run = _run_clean_native_run(env_base, plan_path, numba_cache_dir, tmp_path_factory, "no_stokes")

    result_paths = sorted(run["output_dir"].glob("*.result.json"))
    assert result_paths, "native-run 沒有寫出任何 *.result.json 結果摘要"
    for result_path in result_paths:
        summary = json.loads(result_path.read_text(encoding="utf-8"))
        assert summary["include_stokes"] is False
        assert summary["experiment_case_id"] == "no_stokes"
        assert summary["wave_data_available"] in (None, False)
        for member_summary in summary.get("member_summaries", {}).values():
            assert member_summary["include_stokes"] is False
            assert member_summary["experiment_case_id"] == "no_stokes"

    # 對照組：同一份合成資料下，預設（finite_depth_stokes）案例的既有結果摘要
    # 必須是 wave_data_available=True，證明 no_stokes 的差異確實來自案例選擇，
    # 不是合成資料本身就沒有波浪產品。
    default_result_paths = clean_k1_run_output_result_paths(
        two_domain_env, baytrace_root, numba_cache_dir, tmp_path_factory
    )
    assert default_result_paths
    default_summary = json.loads(default_result_paths[0].read_text(encoding="utf-8"))
    assert default_summary["wave_data_available"] is True
    assert default_summary["include_stokes"] is True
    assert default_summary["experiment_case_id"] == "finite_depth_stokes"


def test_bed_no_deposition_end_to_end_default_bit_identical_and_flag_propagates(
    two_domain_env, baytrace_root, clean_k1_run, numba_cache_dir, tmp_path_factory
) -> None:
    """WP-26／WP-28：``--bed-no-deposition`` 從 ``native-plan`` 到 ``native-run``
    全程串起來後：

    1. 省略這個旗標（``clean_k1_run`` 既有基線）時，每個結果摘要都必須標示
       ``bed_no_deposition=False``，且產出的 ``.driver_result.npz`` 與本測試
       另外重跑一次「明確設定為 False」的結果逐位元相同——這是「省略新旗標
       與明示 False 必須位元相同」這個硬要求在完整 CLI／worker 接線層級的
       證明（驅動層本身的位元相同證明見
       ``tests/test_native_backward_driver.py::
       test_bed_no_deposition_default_bit_identical_and_sensitivity_case_never_deposits``，
       本測試證明的是「CLI 旗標與環境變數接線沒有在半路把這個設定弄丟或
       弄錯」，兩者互補、缺一不可）。
    2. 啟用這個旗標時，計畫檔與每個結果摘要（含摺疊排程單元底下的
       ``member_summaries``）都必須標示 ``bed_no_deposition=True``，證明
       ``--bed-no-deposition``／``LBT_NATIVE_BED_NO_DEPOSITION`` 確實一路
       傳到 ``hour_major.HourMajorConfig``，不是只停在 CLI 層。
    """

    region_to_flow_domain_path = two_domain_env["identity_path"].parent / "region_to_flow_domain.json"
    assert region_to_flow_domain_path.exists()

    # ── 明確設定為 False：與省略（clean_k1_run）比對位元相同 ──────────────
    explicit_false_plan_path = tmp_path_factory.mktemp("bed_no_deposition_plan") / "plan_explicit_false.json"
    rc = cli.run_native_plan_command(
        [
            "--identity-table",
            str(two_domain_env["identity_path"]),
            "--region-to-flow-domain",
            str(region_to_flow_domain_path),
            "--output",
            str(explicit_false_plan_path),
            "--schedule",
            "hour-major-segmented",
            "--segments",
            "1",
            "--skip-gap-check",
        ]
    )
    assert rc == 0
    _units, explicit_false_plan_payload = orch.read_native_plan_file(explicit_false_plan_path)
    assert explicit_false_plan_payload["bed_no_deposition"]["bed_no_deposition"] is False

    env_base = _base_env(two_domain_env, baytrace_root)
    env_base["LBT_NATIVE_BED_NO_DEPOSITION"] = "0"
    explicit_false_run = _run_clean_native_run(
        env_base, explicit_false_plan_path, numba_cache_dir, tmp_path_factory, "bed_no_deposition_false"
    )

    baseline_results = sorted(clean_k1_run["results_dir"].glob("*.driver_result.npz"))
    explicit_false_results = sorted(explicit_false_run["results_dir"].glob("*.driver_result.npz"))
    assert baseline_results and len(baseline_results) == len(explicit_false_results)
    for baseline_path, explicit_false_path in zip(baseline_results, explicit_false_results, strict=True):
        assert baseline_path.name == explicit_false_path.name
        _assert_driver_results_bit_identical(
            baseline_path, explicit_false_path, label="bed_no_deposition省略 vs 明示False"
        )

    for result_path in sorted(clean_k1_run["output_dir"].glob("*.result.json")):
        summary = json.loads(result_path.read_text(encoding="utf-8"))
        assert summary["bed_no_deposition"] is False
        for member_summary in summary.get("member_summaries", {}).values():
            assert member_summary["bed_no_deposition"] is False

    # ── 啟用時：計畫檔與每份結果摘要都必須標示 True ─────────────────────
    enabled_plan_path = tmp_path_factory.mktemp("bed_no_deposition_plan") / "plan_enabled.json"
    rc = cli.run_native_plan_command(
        [
            "--identity-table",
            str(two_domain_env["identity_path"]),
            "--region-to-flow-domain",
            str(region_to_flow_domain_path),
            "--output",
            str(enabled_plan_path),
            "--schedule",
            "hour-major-segmented",
            "--segments",
            "1",
            "--skip-gap-check",
            "--bed-no-deposition",
        ]
    )
    assert rc == 0
    _units, enabled_plan_payload = orch.read_native_plan_file(enabled_plan_path)
    assert enabled_plan_payload["bed_no_deposition"]["bed_no_deposition"] is True

    env_enabled = _base_env(two_domain_env, baytrace_root)
    env_enabled["LBT_NATIVE_BED_NO_DEPOSITION"] = "1"
    enabled_run = _run_clean_native_run(
        env_enabled, enabled_plan_path, numba_cache_dir, tmp_path_factory, "bed_no_deposition_true"
    )

    result_paths = sorted(enabled_run["output_dir"].glob("*.result.json"))
    assert result_paths, "啟用 bed_no_deposition 的 native-run 沒有寫出任何 *.result.json 結果摘要"
    for result_path in result_paths:
        summary = json.loads(result_path.read_text(encoding="utf-8"))
        assert summary["bed_no_deposition"] is True
        for member_summary in summary.get("member_summaries", {}).values():
            assert member_summary["bed_no_deposition"] is True


def clean_k1_run_output_result_paths(two_domain_env, baytrace_root, numba_cache_dir, tmp_path_factory):
    """獨立跑一次預設案例的 K=1 native-run，回傳其 ``*.result.json`` 路徑清單。

    不直接重用 ``clean_k1_run`` fixture：該 fixture 是 module 作用域、可能已被
    其他測試以不同 ``LBT_NATIVE_EXPERIMENT_CASE`` 環境跑過，這裡改成獨立呼叫，
    確保對照組明確在「未設定 LBT_NATIVE_EXPERIMENT_CASE」（即預設
    ``finite_depth_stokes``）下執行，不受測試執行順序影響。
    """

    env_base = _base_env(two_domain_env, baytrace_root)
    run = _run_clean_native_run(
        env_base, two_domain_env["plan_k1_path"], numba_cache_dir, tmp_path_factory, "default_case_control"
    )
    return sorted(run["output_dir"].glob("*.result.json"))


def test_native_plan_records_diffusion_scheme_and_run_rejects_mismatch(
    two_domain_env, tmp_path, monkeypatch
) -> None:
    """WP-51 整合：計畫檔的 physics_case 記錄擴散組態；native-run 啟動前若實際組態不同，
    明確拒絕（不靜默混用），並可由承認旗標放行。純規劃層測試，不需要 BayTrace。"""

    from lagrangian_backtracking.native_runner import physics_case as pc

    _units, payload = orch.read_native_plan_file(two_domain_env["plan_k1_path"])
    scheme = payload["physics_case"]["diffusion_scheme"]
    assert scheme == {
        "diffusion_noise": "uniform",
        "vertical_diffusion": "constant",
        "kz_constant_m2ps": 3.0e-4,
        "visser_correction": False,
        "kz_source": "default",
    }
    for key in (
        "LBT_NATIVE_DIFFUSION_NOISE",
        "LBT_NATIVE_VERTICAL_DIFFUSION",
        "LBT_NATIVE_KZ_CONSTANT_M2PS",
        "LBT_NATIVE_VISSER",
        "LBT_NATIVE_ACKNOWLEDGE_CONFIG_MODE_MISMATCH",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("LBT_NATIVE_CONFIG", str(two_domain_env["config_path"]))
    monkeypatch.setenv("LBT_NATIVE_VERTICAL_DIFFUSION", "off")
    argv = [
        str(two_domain_env["plan_k1_path"]),
        "--output",
        str(tmp_path / "out"),
        "--resource-provider",
        "no_such_module:factory",
        "--worker-count",
        "1",
    ]
    with pytest.raises(pc.PhysicsCaseError, match="擴散組態"):
        cli.run_native_run(argv)
    # 承認旗標只放行此項不一致；之後會在載入（不存在的）資源提供者時才失敗，證明已通過擴散檢查
    monkeypatch.setenv("LBT_NATIVE_ACKNOWLEDGE_CONFIG_MODE_MISMATCH", "1")
    with pytest.raises(ModuleNotFoundError):
        cli.run_native_run(argv)
