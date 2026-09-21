"""WP-12：本機一次性複製（「local staging」）目錄結構的端對端驗證。

背景（見 ``AGENTS.md``、任務指示 WP-12）：正式伺服器 ``.203`` 讀 NAS 頻寬只有
約 11.7 MB/s，已驗收的 ``ocm_native``／``nww3_analysis`` 產品正被一次性複製到
本機，且**只複製逆向運算需要的變數**——OCM 月份副本刻意不含
``diffusivity.npy``（以及 ``temp``／``salt``／``water_density``／``dahv``／
``wind_speed`` 等本模組本來就不讀的變數），NWW3 月份副本只保留必要陣列；
``reconstruction/`` 之下 ``reconstruction-index.json``／domain manifest 原封
不動複製，因此仍會列出「本機並未實際複製」的較早月份（例如 2024-01～
2024-09），但 ``months/<YYYYMM>/`` 底下只有真正落在 2024-10～2025-12 這個
本機複製範圍內的補值月份。

本檔案驗證兩層：

1. ``NativeForcingBridgeReader``／``HourlyFieldPair`` 的 ``load_diffusivity``
   開關本身（不透過 ``production_wiring``，直接對 ``forcing_bridge`` 造合成
   月份，成本最低、最貼近單元測試）。
2. ``production_wiring`` 串起完整「本機一次性複製」目錄結構後的端對端行為：
   ``build_domain_resources``／``preflight_native_inputs``／
   ``run_backward_batch`` 是否都能在「沒有 diffusivity.npy、且重建索引列出
   未複製月份」的情況下正常運作。

沿用既有慣例（見 ``tests/test_native_production_wiring.py`` 模組文件）：凡是
已經在其他測試檔案證明可行的 fixture（合成網格、正式設定、``ocm_native``／
``nww3_analysis`` 產品樹寫入方式），一律在函式作用域內 ``import`` 重用，不
重新發明一份寫法可能不一致的合成資料產生器。本檔案唯二自行新增的邏輯，是
「省略 diffusivity.npy」與「合成一份帶 checksum 的重建 patch＋未複製月份索引」
這兩件本檔案獨有的需求。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# 重用 test_native_production_wiring.py 已證明可行、且是本工作包指定可修改
# 檔案之一的正式設定與 ocm_native/nww3_analysis 合成 fixture；wiring_config
# 是 pytest fixture，直接以原名匯入即可被本檔案的測試函式當成 fixture 使用
# （既有慣例見 tests/test_pilot_config.py 的 runtime_fixture 匯入寫法）。
from test_native_production_wiring import _DP_M as _DP_M
from test_native_production_wiring import _FLOW_DOMAIN_ID as _FLOW_DOMAIN_ID
from test_native_production_wiring import _NVRT as _NVRT
from test_native_production_wiring import _build_regular_grid as _build_regular_grid
from test_native_production_wiring import _hour_ns as _hour_ns
from test_native_production_wiring import _require_baytrace_root as _require_baytrace_root
from test_native_production_wiring import _write_nww3_grid as _write_nww3_grid
from test_native_production_wiring import _write_nww3_month as _write_nww3_month
from test_native_production_wiring import _write_ocm_grid as _write_ocm_grid
from test_native_production_wiring import wiring_config as wiring_config

from lagrangian_backtracking.forcing import OCMReconstructionPatchMonth
from lagrangian_backtracking.geometry import DomainProjection
from lagrangian_backtracking.mesh import NativeMesh
from lagrangian_backtracking.native_runner import forcing_bridge as fb
from lagrangian_backtracking.native_runner import production_wiring as pw

_DOMAIN_A_PROJECTION = DomainProjection(122.05, 25.05)  # 與範例設定 A 區 center_lonlat 一致


# ══════════════════════════════════════════════════════════════════════════
# 一、forcing_bridge 層級：load_diffusivity 開關本身（不需要 mesh／production_wiring）
# ══════════════════════════════════════════════════════════════════════════


def _write_local_staged_month(
    root: Path,
    month_id: str,
    *,
    time_hours: list[int],
    node_count: int = 3,
    layer_count: int = 2,
    face_count: int = 1,
    include_diffusivity: bool = False,
    diffusivity_value: float = 1.0e-3,
) -> Path:
    """造一個「本機一次性複製」風格的合成 OCM 月份：預設不含 ``diffusivity.npy``。

    刻意不重用 ``tests/test_native_forcing_bridge.py`` 的 ``_write_month``
    完整簽章（該函式的預設值與本檔案「省略 diffusivity」的核心測試意圖沒有
    直接關係），改成一個只保留本檔案需要的最小欄位子集的獨立寫法，行為與
    背景描述的本機月份副本一致：只有 ``hvel``／``vertical_velocity``／
    ``zcor``／``elev``／``wetdry_elem``／``time_utc_ns``，``diffusivity.npy``
    只在 ``include_diffusivity=True``（供「即使檔案存在、關閉開關時仍必須
    忽略」測試使用）時才寫入。
    """

    month_dir = root / month_id
    month_dir.mkdir(parents=True, exist_ok=True)
    times = np.asarray([_hour_ns(h) for h in time_hours], dtype=np.int64)
    time_count = times.size

    u = np.zeros((time_count, node_count, layer_count), dtype=np.float64)
    v = np.zeros((time_count, node_count, layer_count), dtype=np.float64)
    for t in range(time_count):
        u[t] = 0.01 * (t + 1)
        v[t] = -0.01 * (t + 1)
    hvel = np.stack((u, v), axis=-1)
    vertical_velocity = np.zeros((time_count, node_count, layer_count), dtype=np.float64)
    zcor = -5.0 * np.arange(layer_count)[None, None, :].astype(np.float64)
    zcor = np.broadcast_to(zcor, (time_count, node_count, layer_count)).copy()
    elev = np.zeros((time_count, node_count), dtype=np.float64)
    wetdry_elem = np.zeros((time_count, face_count), dtype=np.float64)

    np.save(month_dir / "time_utc_ns.npy", times)
    np.save(month_dir / "hvel.npy", hvel)
    np.save(month_dir / "vertical_velocity.npy", vertical_velocity)
    np.save(month_dir / "zcor.npy", zcor)
    np.save(month_dir / "elev.npy", elev)
    np.save(month_dir / "wetdry_elem.npy", wetdry_elem)
    if include_diffusivity:
        diffusivity = np.full((time_count, node_count, layer_count), diffusivity_value, dtype=np.float64)
        np.save(month_dir / "diffusivity.npy", diffusivity)
    return month_dir


def test_default_mode_still_requires_diffusivity_npy(tmp_path: Path) -> None:
    """驗收 1：預設 ``load_diffusivity=True`` 行為必須與既有完全相同——缺檔報錯。"""

    month_dir = _write_local_staged_month(tmp_path, "202401", time_hours=[0, 1, 2])
    assert not (month_dir / "diffusivity.npy").is_file()

    with pytest.raises(fb.ForcingBridgeError, match="diffusivity"):
        fb.NativeForcingBridgeReader({"202401": month_dir}, ibf=1)


def test_load_diffusivity_false_succeeds_without_diffusivity_npy(tmp_path: Path) -> None:
    """關閉開關時，即使完全沒有 diffusivity.npy 也能正常建立與 resolve。"""

    month_dir = _write_local_staged_month(tmp_path, "202401", time_hours=[0, 1, 2, 3])
    reader = fb.NativeForcingBridgeReader({"202401": month_dir}, ibf=1, load_diffusivity=False)

    pair = reader.resolve(_hour_ns(1))

    assert pair.diffusivity_field1 is None
    assert pair.diffusivity_field2 is None
    assert np.all(np.isfinite(pair.u_field1))
    assert not np.any(pair.missing_mask1)
    assert not np.any(pair.missing_mask2)
    # 不因為缺值遮罩不再檢查 diffusivity 而放行真正的（hvel/zcor/w）缺值。
    pair.raise_if_unresolved_gap()


def test_load_diffusivity_false_ignores_diffusivity_even_if_file_exists(tmp_path: Path) -> None:
    """即使 diffusivity.npy 意外存在且整個是 NaN，``load_diffusivity=False`` 時
    仍必須完全忽略它——既不讀取也不讓它污染缺值遮罩。

    這是驗收 1「缺值遮罩不再納入 diffusivity 的有限性檢查」最直接的反例測試：
    若本模組不慎仍讀了這個檔案，``missing_mask`` 會被這個全 NaN 陣列污染成
    全部為 ``True``，``raise_if_unresolved_gap`` 就會誤報缺值。
    """

    month_dir = _write_local_staged_month(
        tmp_path,
        "202401",
        time_hours=[0, 1],
        node_count=3,
        layer_count=2,
        include_diffusivity=True,
        diffusivity_value=float("nan"),
    )
    assert (month_dir / "diffusivity.npy").is_file()

    reader = fb.NativeForcingBridgeReader({"202401": month_dir}, ibf=1, load_diffusivity=False)
    pair = reader.resolve(_hour_ns(0))

    assert pair.diffusivity_field1 is None
    assert not np.any(pair.missing_mask1)
    pair.raise_if_unresolved_gap()


def _minimal_local_staged_patch(
    month_id: str, *, missing_hour: int, node_count: int, layer_count: int, face_count: int, u_value: float
) -> OCMReconstructionPatchMonth:
    """一個補值月份的最小合成 patch；補值仍保留完整欄位（含 diffusivity）。

    背景明確指出本機一次性複製的補值月份是「完整檔案，含 diffusivity」，與
    OCM 觀測月份刻意省略 diffusivity 不對稱；本函式據此保留 diffusivity 欄位，
    驗證 ``load_diffusivity=False`` 時即使 patch 本身仍帶著這個欄位，合併
    邏輯也不會因為觀測月份沒有 diffusivity 可比對形狀而出錯（見
    ``forcing_bridge._merge_month_with_patch`` 的「補值 patch 的 diffusivity
    列也忽略」）。
    """

    time_utc_ns = np.asarray([_hour_ns(missing_hour)], dtype=np.int64)
    u = np.full((1, node_count, layer_count), u_value, dtype=np.float64)
    v = -u
    hvel = np.stack((u, v), axis=-1)
    vertical_velocity = np.zeros((1, node_count, layer_count), dtype=np.float64)
    zcor = -5.0 * np.arange(layer_count)[None, None, :].astype(np.float64)
    zcor = np.broadcast_to(zcor, (1, node_count, layer_count)).copy()
    elev = np.zeros((1, node_count), dtype=np.float64)
    wetdry_elem = np.zeros((1, face_count), dtype=np.float64)
    diffusivity = np.full((1, node_count, layer_count), 1.0e-4, dtype=np.float64)
    return OCMReconstructionPatchMonth(
        month_id=month_id,
        time_utc_ns=time_utc_ns,
        hvel=hvel,
        vertical_velocity=vertical_velocity,
        zcor=zcor,
        elev=elev,
        wetdry_elem=wetdry_elem,
        diffusivity=diffusivity,
        origin_code=np.array([1], dtype=np.int64),
        quality_flags=np.array([1], dtype=np.int64),
        metadata={"method": "short_linear_v1", "gap_id": "local-staging-test-gap"},
    )


def test_load_diffusivity_false_resolves_gap_via_patch_with_full_diffusivity(tmp_path: Path) -> None:
    """驗收 2 的核心情境（forcing_bridge 層級）：OCM 觀測月份無 diffusivity、
    但補值 patch 仍帶完整 diffusivity 時，缺口仍可被補值正確解析。
    """

    month_dir = _write_local_staged_month(
        tmp_path, "202401", time_hours=[0, 1, 2, 4], node_count=3, layer_count=2, face_count=1
    )
    patch = _minimal_local_staged_patch(
        "202401", missing_hour=3, node_count=3, layer_count=2, face_count=1, u_value=999.0
    )

    reader = fb.NativeForcingBridgeReader(
        {"202401": month_dir}, ibf=1, patches={"202401": patch}, load_diffusivity=False
    )
    pair = reader.resolve(_hour_ns(3))

    assert pair.time_origin1 == "reconstructed" or pair.time_origin2 == "reconstructed"
    assert pair.diffusivity_field1 is None and pair.diffusivity_field2 is None
    reconstructed_u = pair.u_field1 if pair.time_origin1 == "reconstructed" else pair.u_field2
    np.testing.assert_allclose(reconstructed_u, 999.0)


# ══════════════════════════════════════════════════════════════════════════
# 二、production_wiring 層級：完整「本機一次性複製」目錄結構端對端驗證
# ══════════════════════════════════════════════════════════════════════════

_GAP_HOUR = 24
_PATCH_U_VALUE = 777.0
_UNCOPIED_MONTH_ID = "202401"  # 索引／manifest 仍列出，但本機未複製


def _write_local_staged_ocm_month(
    month_dir: Path, *, grid: dict[str, Any], hour_offsets: list[int]
) -> None:
    """本機一次性複製版本的 OCM 月份：無 diffusivity.npy，且刻意跳過 ``_GAP_HOUR``。

    速度量級沿用 ``test_native_production_wiring._write_ocm_month`` 的保守
    設計理由（避免端到端測試的合成粒子在小網格上位移過大），只是額外拿掉
    diffusivity 這個欄位。
    """

    month_dir.mkdir(parents=True, exist_ok=True)
    times = np.asarray([_hour_ns(h) for h in hour_offsets], dtype=np.int64)
    time_count = times.size
    n_nodes = grid["n_nodes"]
    n_faces = grid["n_faces"]

    hour_values = np.asarray(hour_offsets, dtype=np.float64)
    u = np.broadcast_to(hour_values[:, None, None] * 0.001, (time_count, n_nodes, _NVRT)).copy()
    v = -u
    hvel = np.stack((u, v), axis=-1)
    vertical_velocity = np.zeros((time_count, n_nodes, _NVRT), dtype=np.float64)
    zcor = np.zeros((time_count, n_nodes, _NVRT), dtype=np.float64)
    zcor[:, :, 0] = -_DP_M
    zcor[:, :, 1] = 0.0
    elev = np.zeros((time_count, n_nodes), dtype=np.float64)
    wetdry_elem = np.zeros((time_count, n_faces), dtype=np.float64)

    np.save(month_dir / "time_utc_ns.npy", times)
    np.save(month_dir / "hvel.npy", hvel)
    np.save(month_dir / "vertical_velocity.npy", vertical_velocity)
    np.save(month_dir / "zcor.npy", zcor)
    np.save(month_dir / "elev.npy", elev)
    np.save(month_dir / "wetdry_elem.npy", wetdry_elem)
    # 刻意不寫 diffusivity.npy——本機一次性複製只保留逆向運算需要的變數。


def _write_reconstruction_tree(
    reconstruction_root: Path,
    *,
    flow_domain_id: str,
    mesh: NativeMesh,
    grid: dict[str, Any],
    patch_month_id: str,
    patch_hour_offset: int,
    patch_u_value: float,
    uncopied_month_id: str,
) -> None:
    """合成一份帶合法 checksum 的重建 patch 樹，並在 manifest 內額外列出一個
    「本機並未複製」的月份（只出現在 manifest 的 ``months`` 索引裡，``months/``
    底下沒有對應資料夾）。

    dataclass／checksum 驗證邏輯完全對照
    ``tests/test_forcing_window.py::_write_reconstruction_fixture``（唯一已
    證明可行、會被 ``forcing_window._ReconstructionPatchStore`` 接受的合成
    寫法），此處獨立重寫一份而非 import 該私有函式，因為那個函式綁死在
    ``test_forcing_window.py`` 自己的合成網格與月份資料上，且該檔案不在本
    工作包可修改清單內。
    """

    patch_root = reconstruction_root
    domain_root = patch_root / flow_domain_id
    month_root = domain_root / "months" / patch_month_id
    month_root.mkdir(parents=True, exist_ok=True)

    n_nodes = grid["n_nodes"]
    n_faces = grid["n_faces"]
    time_utc_ns = np.asarray([_hour_ns(patch_hour_offset)], dtype=np.int64)
    u = np.full((1, n_nodes, _NVRT), patch_u_value, dtype=np.float64)
    v = np.full((1, n_nodes, _NVRT), -patch_u_value, dtype=np.float64)
    hvel = np.stack((u, v), axis=-1)
    vertical_velocity = np.zeros((1, n_nodes, _NVRT), dtype=np.float64)
    zcor = np.zeros((1, n_nodes, _NVRT), dtype=np.float64)
    zcor[:, :, 0] = -_DP_M
    zcor[:, :, 1] = 0.0
    elev = np.zeros((1, n_nodes), dtype=np.float64)
    wetdry_elem = np.zeros((1, n_faces), dtype=np.float64)
    diffusivity = np.full((1, n_nodes, _NVRT), 1.0e-4, dtype=np.float64)
    origin_code = np.asarray([1], dtype=np.uint8)
    quality_flags = np.asarray([1], dtype=np.uint16)

    arrays: dict[str, np.ndarray] = {
        "time_utc_ns": time_utc_ns,
        "hvel": hvel,
        "vertical_velocity": vertical_velocity,
        "zcor": zcor,
        "elev": elev,
        "wetdry_elem": wetdry_elem,
        "diffusivity": diffusivity,
        "origin_code": origin_code,
        "quality_flags": quality_flags,
    }
    array_sha256: dict[str, str] = {}
    for name, value in arrays.items():
        path = month_root / f"{name}.npy"
        np.save(path, value)
        array_sha256[name] = hashlib.sha256(path.read_bytes()).hexdigest()

    metadata = {
        "schema_version": "ocm_reconstruction_patch_v1",
        "flow_domain_id": flow_domain_id,
        "month_id": patch_month_id,
        "method": "synthetic_local_staging_gap_v1",
        "gap_ids": ["synthetic-local-staging-gap"],
        "arrays": {name: {"sha256": sha} for name, sha in array_sha256.items()},
    }
    metadata_bytes = json.dumps(
        metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    metadata_path = month_root / "metadata.json"
    metadata_path.write_bytes(metadata_bytes)
    (month_root / "metadata.json.sha256").write_text(
        hashlib.sha256(metadata_bytes).hexdigest() + "\n", encoding="ascii"
    )

    from lagrangian_backtracking import forcing_window as forcing_window_module

    manifest = {
        "schema_version": "ocm_reconstruction_patch_v1",
        "flow_domain_id": flow_domain_id,
        "mesh_fingerprint": forcing_window_module._mesh_fingerprint(mesh),
        "source_fingerprint": "synthetic-local-staging-source-v1",
        "months": [
            {
                "month_id": patch_month_id,
                "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
                "files": array_sha256,
            },
            {
                # 索引／manifest 原封不動複製，仍列出本機並未複製的較早月份
                # （見模組文件）；checksum 刻意留假值——正常路徑下這個月份
                # 永遠不會被查詢到（build_domain_resources 只依「本機
                # ocm_native 實際存在的月份」查詢 patch，見
                # production_wiring._load_reconstruction_patches），因此
                # 假 checksum 不會被驗證到，也不應該被驗證到。
                "month_id": uncopied_month_id,
                "metadata_sha256": "0" * 64,
                "files": {name: "0" * 64 for name in arrays},
            },
        ],
    }
    manifest_bytes = json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    manifest_path = domain_root / "reconstruction-manifest.json"
    manifest_path.write_bytes(manifest_bytes)
    (domain_root / "reconstruction-manifest.json.sha256").write_text(
        hashlib.sha256(manifest_bytes).hexdigest() + "\n", encoding="ascii"
    )

    index_path = patch_root / "reconstruction-index.json"
    index_bytes = json.dumps(
        {"flow_domains": [flow_domain_id], "months_not_staged_locally": [uncopied_month_id]},
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    index_path.write_bytes(index_bytes)
    (patch_root / "reconstruction-index.json.sha256").write_text(
        hashlib.sha256(index_bytes).hexdigest() + "\n", encoding="ascii"
    )


@pytest.fixture
def local_staging_roots(tmp_path: Path) -> dict[str, Any]:
    """完整合成一份「本機一次性複製」目錄結構：ocm_native／nww3_analysis／
    reconstruction 三個 root，涵蓋 2024-10 一個月（48 小時，含一個已由補值
    patch 登錄的缺口），流場區域固定用範例設定 A 區現行 ``flow_domain_id``。
    """

    grid = _build_regular_grid()
    ocm_root = tmp_path / "ocm_native"
    nww_root = tmp_path / "nww_analysis"
    reconstruction_root = tmp_path / "reconstruction"

    _write_ocm_grid(ocm_root / _FLOW_DOMAIN_ID / "grid", grid)
    hour_offsets = [h for h in range(48) if h != _GAP_HOUR]
    _write_local_staged_ocm_month(
        ocm_root / _FLOW_DOMAIN_ID / "months" / "202410", grid=grid, hour_offsets=hour_offsets
    )

    nww_lon, nww_lat = _write_nww3_grid(nww_root / _FLOW_DOMAIN_ID / "grid")
    _write_nww3_month(
        nww_root / _FLOW_DOMAIN_ID / "months" / "202410",
        hour_offsets=range(48),
        lon_size=nww_lon.size,
        lat_size=nww_lat.size,
    )

    mesh = NativeMesh.from_directory(
        ocm_root / _FLOW_DOMAIN_ID / "grid", projection=_DOMAIN_A_PROJECTION
    )
    _write_reconstruction_tree(
        reconstruction_root,
        flow_domain_id=_FLOW_DOMAIN_ID,
        mesh=mesh,
        grid=grid,
        patch_month_id="202410",
        patch_hour_offset=_GAP_HOUR,
        patch_u_value=_PATCH_U_VALUE,
        uncopied_month_id=_UNCOPIED_MONTH_ID,
    )

    return {
        "ocm_root": ocm_root,
        "nww_root": nww_root,
        "reconstruction_root": reconstruction_root,
        "grid": grid,
        "mesh": mesh,
    }


def _local_staging_env(local_staging_roots: dict[str, Any], baytrace_root: str) -> dict[str, str]:
    return {
        "OCM_NATIVE_ROOT": str(local_staging_roots["ocm_root"]),
        "NWW_ANALYSIS_ROOT": str(local_staging_roots["nww_root"]),
        "OCM_RECONSTRUCTION_ROOT": str(local_staging_roots["reconstruction_root"]),
        "LBT_BAYTRACE_ROOT": baytrace_root,
    }


def test_build_domain_resources_succeeds_without_diffusivity_and_with_uncopied_patch_months(
    wiring_config, local_staging_roots
) -> None:
    """驗收 2：``build_domain_resources`` 在本機一次性複製目錄結構下能正常建出資源。

    涵蓋：(1) 不因缺 diffusivity.npy 失敗；(2) 不因 reconstruction manifest
    列出本機未複製的月份而失敗（該月份根本不會被查詢）；(3) 只把本機實際
    存在月份範圍內的 patch 記進 provenance。
    """

    baytrace_root = _require_baytrace_root()
    _config, config_path = wiring_config
    env = _local_staging_env(local_staging_roots, baytrace_root)

    resources = pw.build_domain_resources(config_path, "A", env=env, formal=True)

    assert resources.provenance["ocm_diffusivity_used"] is False
    assert resources.provenance["ocm_reconstruction_attempted"] is True
    assert resources.provenance["ocm_reconstruction_patch_months_used"] == ("202410",)
    assert _UNCOPIED_MONTH_ID not in resources.provenance["ocm_reconstruction_patch_months_used"]


def test_forcing_resolver_correct_values_and_gap_filled_by_patch(
    wiring_config, local_staging_roots
) -> None:
    """驗收 2：解析器讀得到正確數值，且缺口小時的值確實來自補值 patch。"""

    baytrace_root = _require_baytrace_root()
    _config, config_path = wiring_config
    env = _local_staging_env(local_staging_roots, baytrace_root)
    grid = local_staging_roots["grid"]

    resources = pw.build_domain_resources(config_path, "A", env=env, formal=True)
    forcing_resolve = resources.forcing_resolver_factory()

    # 正常（非缺口）小時：數值必須與寫入的合成資料一致（u = 0.001 * hour）。
    normal_pair = forcing_resolve(_hour_ns(10))
    assert normal_pair.u_field1.shape == (_NVRT, grid["n_nodes"])
    np.testing.assert_allclose(np.abs(normal_pair.u_field1), 0.001 * 10, atol=1e-9)
    assert normal_pair.diffusivity_field1 is None  # load_diffusivity=False 全程生效

    # 缺口小時：field1／field2 其中之一必須是補值 patch 的獨特數值。
    gap_pair = forcing_resolve(_hour_ns(_GAP_HOUR))
    reconstructed_u = (
        gap_pair.u_field1 if gap_pair.time_origin1 == "reconstructed" else gap_pair.u_field2
    )
    np.testing.assert_allclose(np.abs(reconstructed_u), _PATCH_U_VALUE)


def test_wave_resolver_works_with_minimal_nww3_arrays(wiring_config, local_staging_roots) -> None:
    """驗收 2：NWW3 只有必要陣列時，波浪解析器仍可正常使用。"""

    baytrace_root = _require_baytrace_root()
    _config, config_path = wiring_config
    env = _local_staging_env(local_staging_roots, baytrace_root)

    resources = pw.build_domain_resources(config_path, "A", env=env, formal=True)
    assert resources.wave_resolver_factory is not None
    wave_resolve = resources.wave_resolver_factory()
    wave = wave_resolve(_hour_ns(10))

    assert np.all(wave.valid_mask_wave)
    np.testing.assert_allclose(wave.significant_wave_height_m, 1.2)


def test_preflight_reports_ready_without_diffusivity(wiring_config, local_staging_roots) -> None:
    """驗收 2：``preflight_native_inputs`` 不得因缺 diffusivity.npy 判定未就緒。

    這裡刻意不斷言 ``is_run_ready`` 為真（整年份完整度不是本測試重點、要造
    出全年覆蓋的合成資料成本過高且與本工作包主題無關），改成直接斷言：
    （1）節點數／hvel 抽樣、wetdry 檢查都正常通過；（2）沒有任何問題訊息
    提到 diffusivity——這才是 WP-12 真正要求「不得因缺 diffusivity.npy 判定
    未就緒」的可驗證核心。
    """

    _config, config_path = wiring_config
    env = {
        "OCM_NATIVE_ROOT": str(local_staging_roots["ocm_root"]),
        "NWW_ANALYSIS_ROOT": str(local_staging_roots["nww_root"]),
        "OCM_RECONSTRUCTION_ROOT": str(local_staging_roots["reconstruction_root"]),
    }

    report = pw.preflight_native_inputs(config_path, env=env, domain_ids=["A"], formal=True)

    domain_report = report.domain_reports[0]
    grid = local_staging_roots["grid"]
    assert domain_report.node_count_grid == grid["n_nodes"]
    assert domain_report.node_count_hvel_sample == grid["n_nodes"]
    assert domain_report.wetdry_elem_values_ok is True
    assert domain_report.forcing_gap_report is not None
    assert not any("diffusivity" in problem for problem in domain_report.problems)


def test_preflight_reports_not_ready_when_required_file_missing(
    wiring_config, local_staging_roots
) -> None:
    """故意刪掉一個必要檔（非 diffusivity）時，preflight 必須明確指出缺哪個月份哪個檔。"""

    _config, config_path = wiring_config
    zcor_path = local_staging_roots["ocm_root"] / _FLOW_DOMAIN_ID / "months" / "202410" / "zcor.npy"
    assert zcor_path.is_file()
    zcor_path.unlink()

    env = {
        "OCM_NATIVE_ROOT": str(local_staging_roots["ocm_root"]),
        "NWW_ANALYSIS_ROOT": str(local_staging_roots["nww_root"]),
        "OCM_RECONSTRUCTION_ROOT": str(local_staging_roots["reconstruction_root"]),
    }

    report = pw.preflight_native_inputs(config_path, env=env, domain_ids=["A"], formal=True)

    domain_report = report.domain_reports[0]
    assert not domain_report.is_run_ready
    assert any(
        "202410" in problem and "zcor.npy" in problem for problem in domain_report.problems
    ), domain_report.problems


# ══════════════════════════════════════════════════════════════════════════
# 三、端到端：以 build_domain_resources 接出的資源直接驅動 run_backward_batch
# ══════════════════════════════════════════════════════════════════════════


class _LocalStagingSoa:
    """欄位相容的最小 ``BayTraceParticleSoA`` 替身，寫法比照
    ``tests/test_native_production_wiring.py`` 的 ``_WiringE2ESoa``。
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, row_particle_index: np.ndarray) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.z0 = z
        self.row_particle_index = row_particle_index


def test_run_backward_batch_completes_hours_via_local_staging_resources(
    wiring_config, local_staging_roots
) -> None:
    """驗收 2 最後一項：整條「本機一次性複製」接線可以真正驅動 BayTrace 核心跑數小時。

    到達時刻與回溯窗刻意避開 ``_GAP_HOUR``（本測試的重點是「整條接線可以
    跑」，缺口補值的正確性已由 ``test_forcing_resolver_correct_values_and_
    gap_filled_by_patch`` 另外驗證，這裡不重複）。
    """

    from lagrangian_backtracking.native_runner import backward_driver as bd

    baytrace_root = _require_baytrace_root()
    _config, config_path = wiring_config
    env = _local_staging_env(local_staging_roots, baytrace_root)
    grid = local_staging_roots["grid"]

    resources = pw.build_domain_resources(config_path, "A", env=env, formal=True)

    center_lon = float(np.mean(grid["node_lon"]))
    center_lat = float(np.mean(grid["node_lat"]))
    center_x, center_y = _DOMAIN_A_PROJECTION.project(center_lon, center_lat)
    row_particle_index = np.arange(3, dtype=np.int64)
    soa = _LocalStagingSoa(
        x=np.asarray([center_x - 30.0, center_x, center_x + 30.0], dtype=np.float64),
        y=np.full(3, center_y, dtype=np.float64),
        z=np.full(3, -5.0, dtype=np.float64),
        row_particle_index=row_particle_index,
    )
    config = bd.BackwardDriverConfig(
        arrival_time_utc_ns=_hour_ns(44),
        settling_velocity_mps=-0.0005,
        max_backtrack_seconds=3.0 * 3600.0,
        enable_diffusion=False,
        master_seed=20260921,
    )
    seed_hex_values = [format(int(row), "032x") for row in row_particle_index]
    wave_resolver = (
        resources.wave_resolver_factory() if resources.wave_resolver_factory is not None else None
    )

    result = bd.run_backward_batch(
        resources.driver_grid,
        bd.load_baytrace_modules(baytrace_root),
        soa,
        seed_hex_values=seed_hex_values,
        config=config,
        forcing_resolver=resources.forcing_resolver_factory(),
        interest_region_polygon=resources.interest_regions["gongliao"],
        wave_field_resolver=wave_resolver,
    )

    assert result.hours_run >= 1
    assert result.final_status.shape == (3,)
