"""WP-8：``native_runner.production_wiring``（正式接線）的驗收測試。

涵蓋（對應工作包驗收要求逐項）：

1. ``build_domain_resources``：合成 ``ocm_native``／``nww3_analysis`` 產品目錄 ＋
   合成設定，接出正確的網格、關注區域與解析器；解析器回傳的時間範圍與數值正確。
2. ``make_domain_resources_factory``：回傳值可以被 pickle，且能在真正以
   ``multiprocessing`` ``spawn`` 啟動的子程序中呼叫並取得資源。
3. ``build_formal_identity_table``：在正式（``formal=True``）五站 50,000 情境
   母體下，``members_per_scenario=1`` 與 ``=10`` 都能得到正確數量與正確身分／
   初始位置。
4. ``preflight_native_inputs``：正確回報缺目錄、節點數不一致、缺口影響到達
   時刻回溯窗。
5. 端到端：接出的資源可以直接餵給 ``native_runner.orchestrator.run_native_plan``
   ＋真正的 ``run_backward_batch`` 跑數小時，不需要另外手刻網格或驅動層。

除步驟 5（需要真正呼叫 BayTrace 原版 njit 核心）外，其餘測試不需要
``LBT_BAYTRACE_ROOT``；需要的測試比照既有慣例在缺少此環境變數時直接跳過。

本檔案與 ``native_runner/production_wiring.py`` 是本工作包唯一可新增／修改的
兩個檔案，因此本檔案刻意**不**修改 ``tests/test_manifests.py``：凡是需要沿用
該檔案既有、已證明可用的合成 fixture（``_receptor_payload``／``_arrival_payload``／
``_initial_condition_payload``／``_geometry_payloads``／``_write_json``）之處一律
用 ``from test_manifests import ...`` 在函式作用域內匯入（既有
``tests/test_completed_subset_preview.py``／``tests/test_native_coast.py`` 已使用
這個慣例）；凡是需要「同時取得可寫回磁碟的原始 payload dict、又要建出
``ProjectConfig``」之處（``_config_payload`` 之下），才在本檔案內獨立複寫一份
與 ``test_manifests._config`` 相同的 payload 突變邏輯，因為那個函式只回傳
``ProjectConfig``，沒有暴露原始 dict 讓本檔案另外寫成 YAML 檔案。
"""

from __future__ import annotations

import functools
import json
import multiprocessing
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from lagrangian_backtracking.config import ProjectConfig
from lagrangian_backtracking.geometry import DomainProjection
from lagrangian_backtracking.native_runner import orchestrator as orch
from lagrangian_backtracking.native_runner import production_wiring as pw

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = ROOT / "configs" / "lagrangian_backtracking.example.yaml"

_HOUR_NS = 3_600_000_000_000
_EPOCH_2024_NS = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1_000_000_000)
_FLOW_DOMAIN_ID = "northeast_taiwan_common_cache_v3"  # 貢寮／龜山島（A 區）現行上游 ID


def _hour_ns(hour_offset: int) -> int:
    """把「自 2024-01-01T00:00:00Z 起第幾個小時」換成 UTC 奈秒。"""

    return _EPOCH_2024_NS + hour_offset * _HOUR_NS


def _require_baytrace_root() -> str:
    root = os.environ.get("LBT_BAYTRACE_ROOT")
    if not root:
        pytest.skip("需要 LBT_BAYTRACE_ROOT 指向 BayTrace 原始碼根目錄才能執行本測試")
    return root


# ══════════════════════════════════════════════════════════════════════════
# 一、設定 fixture：獨立複寫 test_manifests._config 的 payload 突變邏輯
# ══════════════════════════════════════════════════════════════════════════


def _config_payload(*, with_paths: bool) -> dict[str, Any]:
    """回傳與 ``test_manifests._config`` 相同突變規則的原始 payload dict。

    本工作包不得修改 ``tests/test_manifests.py``，而該檔案的 ``_config`` 只回傳
    已驗證的 ``ProjectConfig``、不暴露原始 dict；本函式需要原始 dict 才能另外
    寫成一份真正可被 ``load_config`` 讀回的 YAML 檔案（``make_domain_resources_factory``
    的子程序測試、端到端測試都需要真實檔案路徑），因此在此獨立重寫一份相同的
    突變規則，而不是嘗試從 ``ProjectConfig`` 反推回 dict。
    """

    payload = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    for field in (
        "horizontal_receptor_coordinates",
        "horizontal_receptor_source_manifest_sha256",
        "horizontal_receptor_selection_policy",
        "horizontal_receptor_coordinate_tolerance_m",
        "receptor_priority_polygon",
        "receptor_priority_selection",
    ):
        for site in payload["study_sites"]:
            site.pop(field, None)
    payload["design_version"] = "design_baseline_v3_non_rising_a_v3_local20_20260909"
    payload.pop("arrival_time_selection", None)
    payload["scenarios"].pop("bed_residence_time", None)
    payload["inputs"].pop("backtrack_support_days", None)
    if with_paths:
        payload["scenarios"]["material_manifest"] = "material.json"
        payload["scenarios"]["receptor_manifest"] = "receptor.json"
        payload["scenarios"]["arrival_time_manifest"] = "arrival.json"
        payload["scenarios"]["receptor_arrival_initial_condition_manifest"] = (
            "initial_conditions.json"
        )
        payload["geometry"]["domain_manifest"] = "domain.json"
        payload["geometry"]["local_domain_manifest"] = "local.json"
        payload["geometry"]["open_boundary_manifest"] = "open.json"
    else:
        payload["scenarios"]["receptor_arrival_initial_condition_manifest"] = None
    return payload


def _write_geometry_manifests(tmp_path: Path) -> None:
    from test_manifests import _geometry_payloads, _write_json

    domain_payload, local_payload, open_payload = _geometry_payloads()
    _write_json(tmp_path / "domain.json", domain_payload)
    _write_json(tmp_path / "local.json", local_payload)
    _write_json(tmp_path / "open.json", open_payload)


@pytest.fixture
def wiring_config(tmp_path: Path) -> tuple[ProjectConfig, Path]:
    """四域五站的正式設定＋geometry manifest＋一份可被 ``load_config`` 讀回的真 YAML。

    刻意不寫 material／receptor／arrival／initial_conditions manifest：
    ``build_domain_resources`` 只呼叫 ``load_boundary_geometries``，不呼叫
    ``load_scenario_inputs``，因此不需要這些檔案存在。
    """

    payload = _config_payload(with_paths=True)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    _write_geometry_manifests(tmp_path)
    config = ProjectConfig.model_validate(payload)
    return config, config_path


# ══════════════════════════════════════════════════════════════════════════
# 二、合成 ocm_native／nww3_analysis 產品樹：只造 A 區（貢寮／龜山島）
# ══════════════════════════════════════════════════════════════════════════

_NX = 4
_NY = 4
_DLON = 0.01
_DLAT = 0.01
_LON0 = 121.98  # 落在貢寮 local domain（121.7~122.15, 24.9~25.3）與 A 區 bbox 內
_LAT0 = 25.02
_DP_M = 20.0
_NVRT = 2


def _node_index(i: int, j: int) -> int:
    return j * _NX + i


def _build_regular_grid() -> dict[str, Any]:
    """建立 4x4 節點、間距約 1 公里、水深固定 20 公尺的規則三角網格。

    三角化方式（每格切兩個三角形）與 ``tests/test_native_backward_driver.py``
    的 ``_build_mesh`` 完全相同寫法，避免自創一種可能引入邊鄰接慣例錯誤的切法。
    """

    n_nodes = _NX * _NY
    node_lon = np.empty(n_nodes, dtype=np.float64)
    node_lat = np.empty(n_nodes, dtype=np.float64)
    for j in range(_NY):
        for i in range(_NX):
            idx = _node_index(i, j)
            node_lon[idx] = _LON0 + i * _DLON
            node_lat[idx] = _LAT0 + j * _DLAT
    dp = np.full(n_nodes, _DP_M, dtype=np.float64)
    node_bottom_index = np.zeros(n_nodes, dtype=np.int64)

    triangles: list[tuple[int, int, int]] = []
    for j in range(_NY - 1):
        for i in range(_NX - 1):
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


def _write_ocm_grid(grid_dir: Path, grid: dict[str, Any]) -> None:
    grid_dir.mkdir(parents=True, exist_ok=True)
    np.save(grid_dir / "source_lon.npy", grid["node_lon"])
    np.save(grid_dir / "source_lat.npy", grid["node_lat"])
    np.save(grid_dir / "source_depth_m.npy", grid["dp"])
    np.save(grid_dir / "source_node_bottom_index.npy", grid["node_bottom_index"])
    np.save(grid_dir / "source_face_nodes_local.npy", grid["face_nodes_local"])
    np.save(grid_dir / "source_face_node_count.npy", grid["face_node_count"])
    np.save(grid_dir / "source_face_global_index.npy", grid["face_global_index"])


def _write_ocm_month(
    month_dir: Path,
    *,
    grid: dict[str, Any],
    hour_offsets: range,
    node_count: int | None = None,
) -> None:
    """寫出一個最小可用的合成 ``ocm_native`` 月份。

    流速在空間上均勻、隨小時線性小幅變化（``u=0.001*hour``），刻意取很小的
    係數：端到端測試的合成網格只有約 3 公里見方，若用一般 forcing 測試常見的
    量級（例如 0.1 m/s 起跳且隨小時線性增大），4 小時回溯的位移會遠遠超出
    這個小網格範圍，讓粒子在測試一開始就離開網格；本模組的重點是驗證「接線
    是否正確」，不是驗證驅動層本身在大位移下的行為（那是 WP-5 的測試範圍），
    因此刻意選一個不會讓合成粒子離開合成小網格的保守量級。

    ``node_count`` 只在「刻意製造節點數與網格不一致」的 preflight 測試中會被
    覆寫成與 ``grid`` 不同的值。
    """

    month_dir.mkdir(parents=True, exist_ok=True)
    times = np.asarray([_hour_ns(h) for h in hour_offsets], dtype=np.int64)
    time_count = times.size
    n_nodes = node_count if node_count is not None else grid["n_nodes"]
    n_faces = grid["n_faces"]

    hour_values = np.asarray(list(hour_offsets), dtype=np.float64)
    u = np.broadcast_to(hour_values[:, None, None] * 0.001, (time_count, n_nodes, _NVRT)).copy()
    v = -u
    hvel = np.stack((u, v), axis=-1)
    vertical_velocity = np.zeros((time_count, n_nodes, _NVRT), dtype=np.float64)
    zcor = np.zeros((time_count, n_nodes, _NVRT), dtype=np.float64)
    zcor[:, :, 0] = -_DP_M
    zcor[:, :, 1] = 0.0
    elev = np.zeros((time_count, n_nodes), dtype=np.float64)
    wetdry_elem = np.zeros((time_count, n_faces), dtype=np.float64)
    diffusivity = np.full((time_count, n_nodes, _NVRT), 1.0e-4, dtype=np.float64)

    np.save(month_dir / "time_utc_ns.npy", times)
    np.save(month_dir / "hvel.npy", hvel)
    np.save(month_dir / "vertical_velocity.npy", vertical_velocity)
    np.save(month_dir / "zcor.npy", zcor)
    np.save(month_dir / "elev.npy", elev)
    np.save(month_dir / "wetdry_elem.npy", wetdry_elem)
    np.save(month_dir / "diffusivity.npy", diffusivity)


def _write_nww3_grid(grid_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """寫出涵蓋合成 OCM 網格整個經緯度範圍的粗略 NWW3 規則網格。"""

    grid_dir.mkdir(parents=True, exist_ok=True)
    lon = np.linspace(121.9, 122.1, 5)
    lat = np.linspace(24.95, 25.15, 5)
    np.save(grid_dir / "lon.npy", lon)
    np.save(grid_dir / "lat.npy", lat)
    return lon, lat


def _write_nww3_month(month_dir: Path, *, hour_offsets: range, lon_size: int, lat_size: int) -> None:
    month_dir.mkdir(parents=True, exist_ok=True)
    times = np.asarray([_hour_ns(h) for h in hour_offsets], dtype=np.int64)
    shape = (times.size, lat_size, lon_size)
    np.save(month_dir / "time_utc_ns.npy", times)
    np.save(month_dir / "significant_wave_height.npy", np.full(shape, 1.2, dtype=np.float64))
    np.save(month_dir / "peak_frequency.npy", np.full(shape, 0.1, dtype=np.float64))
    np.save(month_dir / "peak_direction_raw_deg.npy", np.full(shape, 45.0, dtype=np.float64))
    np.save(month_dir / "valid_mask_wave.npy", np.ones(shape, dtype=bool))
    np.save(month_dir / "qc_flags.npy", np.zeros(shape, dtype=np.uint16))


@pytest.fixture
def ocm_and_nww_roots(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    """僅涵蓋 2024-01-01 起 48 小時、流場區域 ``northeast_taiwan_common_cache_v3``
    的合成 ``ocm_native``／``nww3_analysis`` 產品樹。

    刻意只有 48 小時（不到兩天）：``build_domain_resources`` 只要求「至少一個
    月份」，不要求完整涵蓋；完整度、缺口一律留給 ``preflight_native_inputs``
    檢查（見 ``production_wiring`` 模組 docstring「月份選擇政策」一節），因此
    這個小 fixture 同時可以：(1) 驗證 ``build_domain_resources`` 在「有資料但
    不完整」時仍能正常接線；(2) 讓 ``preflight_native_inputs`` 的缺口偵測測試
    有真正的缺口可以偵測，不需要另外造一份「刻意不完整」的 fixture。
    """

    grid = _build_regular_grid()
    ocm_root = tmp_path / "ocm_native"
    nww_root = tmp_path / "nww_analysis"
    _write_ocm_grid(ocm_root / _FLOW_DOMAIN_ID / "grid", grid)
    _write_ocm_month(ocm_root / _FLOW_DOMAIN_ID / "months" / "202401", grid=grid, hour_offsets=range(48))
    nww_lon, nww_lat = _write_nww3_grid(nww_root / _FLOW_DOMAIN_ID / "grid")
    _write_nww3_month(
        nww_root / _FLOW_DOMAIN_ID / "months" / "202401",
        hour_offsets=range(48),
        lon_size=nww_lon.size,
        lat_size=nww_lat.size,
    )
    return ocm_root, nww_root, grid


# ══════════════════════════════════════════════════════════════════════════
# 三、build_domain_resources
# ══════════════════════════════════════════════════════════════════════════


def test_build_domain_resources_basic_correctness(wiring_config, ocm_and_nww_roots) -> None:
    baytrace_root = _require_baytrace_root()
    config, config_path = wiring_config
    ocm_root, nww_root, grid = ocm_and_nww_roots
    env = {
        "OCM_NATIVE_ROOT": str(ocm_root),
        "NWW_ANALYSIS_ROOT": str(nww_root),
        "LBT_BAYTRACE_ROOT": baytrace_root,
    }

    resources = pw.build_domain_resources(config_path, "A", env=env, formal=True)

    assert resources.domain_id == "A"
    assert resources.mesh.node_xy.shape == (grid["n_nodes"], 2)
    assert set(resources.interest_regions) == {"gongliao", "guishan"}
    for polygon in resources.interest_regions.values():
        assert polygon.is_valid
        assert polygon.area > 0.0

    lo_ns, hi_ns = resources.available_utc_range_ns
    assert lo_ns == _hour_ns(0)
    assert hi_ns == _hour_ns(47)

    forcing_resolve = resources.forcing_resolver_factory()
    pair = forcing_resolve(_hour_ns(24))
    assert pair.u_field1.shape == (_NVRT, grid["n_nodes"])
    assert np.all(np.isfinite(pair.u_field1))
    assert np.all(np.isfinite(pair.v_field2))
    assert pair.ibf == -1
    assert pair.month_id1 == "202401"
    assert pair.month_id2 == "202401"

    assert resources.wave_resolver_factory is not None
    wave_resolve = resources.wave_resolver_factory()
    wave = wave_resolve(_hour_ns(24))
    assert np.all(wave.valid_mask_wave)
    np.testing.assert_allclose(wave.significant_wave_height_m, 1.2)

    provenance = resources.provenance
    assert provenance["domain_id"] == "A"
    assert provenance["flow_domain_id"] == _FLOW_DOMAIN_ID
    assert provenance["ocm_months_used"] == ("202401",)
    assert provenance["nww_months_used"] == ("202401",)
    assert provenance["wave_data_available"] is True
    assert provenance["ocm_reconstruction_attempted"] is False
    assert provenance["ocm_reconstruction_root"] is None
    assert len(provenance["mesh_sha256"]) == 64
    assert "own_local_domain" in provenance["interest_region_definition"]
    assert provenance["config_sha256"] == config.config_hash()


def test_build_domain_resources_missing_ocm_root_env_raises(wiring_config) -> None:
    _config, config_path = wiring_config
    with pytest.raises(pw.WiringError, match="OCM_NATIVE_ROOT"):
        pw.build_domain_resources(config_path, "A", env={}, formal=True)


def test_build_domain_resources_rejects_blank_domain_id(wiring_config) -> None:
    _config, config_path = wiring_config
    with pytest.raises(pw.WiringError):
        pw.build_domain_resources(config_path, "  ", env={}, formal=True)


def test_build_domain_resources_without_wave_data(
    wiring_config, ocm_and_nww_roots, tmp_path: Path
) -> None:
    """NWW3 root 環境變數有值，但目錄底下沒有任何內容時，波浪解析器必須是 None。"""

    baytrace_root = _require_baytrace_root()
    _config, config_path = wiring_config
    ocm_root, _nww_root, _grid = ocm_and_nww_roots
    empty_nww_root = tmp_path / "empty_nww_analysis"
    env = {
        "OCM_NATIVE_ROOT": str(ocm_root),
        "NWW_ANALYSIS_ROOT": str(empty_nww_root),
        "LBT_BAYTRACE_ROOT": baytrace_root,
    }

    resources = pw.build_domain_resources(config_path, "A", env=env, formal=True)

    assert resources.wave_resolver_factory is None
    assert resources.provenance["wave_data_available"] is False
    assert resources.provenance["nww_months_used"] == ()


# ══════════════════════════════════════════════════════════════════════════
# 四、make_domain_resources_factory：pickle 往返與真正 spawn 子程序呼叫
# ══════════════════════════════════════════════════════════════════════════


def test_make_domain_resources_factory_rejects_project_config_instance(wiring_config) -> None:
    config, _config_path = wiring_config
    with pytest.raises(pw.WiringError):
        pw.make_domain_resources_factory(config)


def _call_factory_and_summarize(factory: pw.DomainResourcesFactory, domain_id: str) -> dict[str, Any]:
    """在（可能是子程序的）呼叫端呼叫一次工廠，只回傳可 pickle 的摘要。

    刻意不直接回傳整個 ``DomainResources``：其 ``forcing_resolver_factory``／
    ``wave_resolver_factory`` 是 ``build_domain_resources`` 內部定義的閉包，
    無法被 pickle 送回父程序（這正是 ``resources.DomainResources`` 模組文件
    所說「每個工作程序各自呼叫一次取得自己的解析器」的原因——資源本來就設計
    成不跨程序傳遞）；本函式必須是模組層級函式，才能被 ``ProcessPoolExecutor``
    以 spawn context 呼叫。
    """

    resources = factory(domain_id)
    return {
        "domain_id": resources.domain_id,
        "node_count": int(resources.mesh.node_xy.shape[0]),
        "wave_data_available": resources.wave_resolver_factory is not None,
        "available_utc_range_ns": resources.available_utc_range_ns,
    }


def test_make_domain_resources_factory_pickle_roundtrip_and_subprocess_call(
    wiring_config, ocm_and_nww_roots
) -> None:
    baytrace_root = _require_baytrace_root()
    _config, config_path = wiring_config
    ocm_root, nww_root, grid = ocm_and_nww_roots
    env = {
        "OCM_NATIVE_ROOT": str(ocm_root),
        "NWW_ANALYSIS_ROOT": str(nww_root),
        "LBT_BAYTRACE_ROOT": baytrace_root,
    }

    factory = pw.make_domain_resources_factory(config_path, env=env, formal=True)
    assert isinstance(factory, functools.partial)

    restored = pickle.loads(pickle.dumps(factory))
    assert isinstance(restored, functools.partial)

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
        summary = pool.submit(_call_factory_and_summarize, restored, "A").result(timeout=180)

    assert summary["domain_id"] == "A"
    assert summary["node_count"] == grid["n_nodes"]
    assert summary["wave_data_available"] is True
    assert summary["available_utc_range_ns"] == (_hour_ns(0), _hour_ns(47))


# ══════════════════════════════════════════════════════════════════════════
# 五、build_formal_identity_table：formal=True、完整五站 50,000/500,000 情境
# ══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def formal_identity_config(tmp_path: Path) -> tuple[ProjectConfig, Path]:
    """建立可直接餵給 ``build_formal_identity_table(formal=True)`` 的完整正式母體。

    直接沿用 ``test_manifests.py`` 已證明可行的 fixture 組合（見該檔案
    ``test_full_scenario_inputs_are_deterministic_and_hash_bound``）：預設的
    ``_receptor_payload()``／``_arrival_payload()``／``_initial_condition_payload()``
    恰好就是完整五站 50,000 情境、5,000 筆 receptor×arrival 初始條件母體，因此
    這裡可以直接用 ``formal=True`` 測試，不需要另外用 ``formal=False`` 迴避
    ``validate_baseline_coverage``。
    """

    from test_manifests import (
        _arrival_payload,
        _initial_condition_payload,
        _material_payload,
        _receptor_payload,
        _write_json,
    )

    payload = _config_payload(with_paths=True)
    payload["scenarios"]["master_seed"] = 424242
    payload["execution"]["shard_scenario_count"] = 2_500
    config = ProjectConfig.model_validate(payload)

    config_path = tmp_path / "config.yaml"
    config_path.write_text("placeholder: true\n", encoding="utf-8")
    _write_json(tmp_path / "material.json", _material_payload())
    receptor_payload = _receptor_payload()
    arrival_payload = _arrival_payload()
    _write_json(tmp_path / "receptor.json", receptor_payload)
    _write_json(tmp_path / "arrival.json", arrival_payload)
    _write_json(
        tmp_path / "initial_conditions.json",
        _initial_condition_payload(config, receptors=receptor_payload, arrivals=arrival_payload),
    )
    return config, config_path


def test_build_formal_identity_table_members_per_scenario_one(formal_identity_config) -> None:
    config, config_path = formal_identity_config

    table = pw.build_formal_identity_table(
        config, members_per_scenario=1, config_path=config_path, formal=True
    )

    assert len(table) == 50_000
    assert set(np.unique(table.analysis_region_id)) == {"A", "B", "C", "D"}
    assert set(np.unique(table.study_site_id)) == {
        "gongliao",
        "guishan",
        "hsinchu",
        "nanwan",
        "lienchiang",
    }

    # 針對一顆已知粒子（貢寮第一個受體 × 第一個到達時刻）逐項核對水平／垂向初始位置，
    # 驗證 resolve_initial_position 真的把受體水平模板與 receptor×arrival 正式初始
    # 條件的實際深度正確合併，而不是張冠李戴或用預設值頂替。
    mask = (table.receptor_id == "gongliao_r00") & (table.arrival_time_id == "gongliao_t00")
    rows = np.flatnonzero(mask)
    assert rows.size == 10  # 10 種材料，M=1 時每個 (receptor,arrival,material) 恰一顆粒子

    domain_a = next(item for item in config.domains if item.analysis_region_id == "A")
    projection = DomainProjection(*domain_a.center_lonlat)
    expected_x, expected_y = projection.project(121.0, 24.0)  # gongliao_r00 的 lon/lat
    # _initial_condition_payload 的公式：vertical_id="z0"→fraction=0.10，
    # eta=1.0+(0%5)*0.01=1.0，bed=-20.0-(0%2)*0.1=-20.0，
    # z=bed+(eta-bed)*fraction=-20.0+21.0*0.10=-17.9。
    expected_z = -17.9
    for row in rows:
        assert table.x0_m[row] == pytest.approx(float(expected_x))
        assert table.y0_m[row] == pytest.approx(float(expected_y))
        assert table.z0_m[row] == pytest.approx(expected_z)
        assert table.study_site_id[row] == "gongliao"
        assert table.analysis_region_id[row] == "A"


def test_build_formal_identity_table_members_per_scenario_ten(formal_identity_config) -> None:
    config, config_path = formal_identity_config

    table = pw.build_formal_identity_table(
        config, members_per_scenario=10, config_path=config_path, formal=True
    )

    assert len(table) == 500_000
    # 每個基礎情境恰有 10 個獨立系集成員，member_id 應涵蓋 0..9。
    assert set(np.unique(table.member_id).tolist()) == set(range(10))


def test_build_formal_identity_table_rejects_missing_master_seed(tmp_path: Path) -> None:
    """master_seed 尚未核定（沿用範例設定的 null）時必須 fail-closed，且不必等
    manifest 讀完才發現——見 production_wiring.py 對這個檢查順序的說明。
    """

    payload = _config_payload(with_paths=True)
    assert payload["scenarios"]["master_seed"] is None
    config = ProjectConfig.model_validate(payload)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("placeholder: true\n", encoding="utf-8")

    with pytest.raises(pw.WiringError, match="master_seed"):
        pw.build_formal_identity_table(config, members_per_scenario=1, config_path=config_path, formal=True)


# ══════════════════════════════════════════════════════════════════════════
# 六、preflight_native_inputs
# ══════════════════════════════════════════════════════════════════════════


def test_preflight_reports_missing_directories(wiring_config) -> None:
    _config, config_path = wiring_config

    report = pw.preflight_native_inputs(config_path, env={}, domain_ids=["A"], formal=True)

    assert len(report.domain_reports) == 1
    domain_report = report.domain_reports[0]
    assert not domain_report.is_run_ready
    assert not report.is_run_ready
    assert any("OCM_NATIVE_ROOT" in problem for problem in domain_report.problems)
    assert any("NWW_ANALYSIS_ROOT" in problem for problem in domain_report.problems)
    assert domain_report.ocm_months_found == ()
    assert domain_report.nww_months_found == ()
    # describe() 必須至少能正常組字串，不因為欄位是 None 而拋例外。
    assert "尚不可執行" in domain_report.describe()


def test_preflight_reports_node_count_mismatch(
    wiring_config, ocm_and_nww_roots, tmp_path: Path
) -> None:
    _config, config_path = wiring_config
    _ocm_root, nww_root, grid = ocm_and_nww_roots

    broken_ocm_root = tmp_path / "broken_ocm_native"
    _write_ocm_grid(broken_ocm_root / _FLOW_DOMAIN_ID / "grid", grid)
    _write_ocm_month(
        broken_ocm_root / _FLOW_DOMAIN_ID / "months" / "202401",
        grid=grid,
        hour_offsets=range(6),
        node_count=grid["n_nodes"] - 1,  # 刻意比網格少一個節點
    )
    env = {"OCM_NATIVE_ROOT": str(broken_ocm_root), "NWW_ANALYSIS_ROOT": str(nww_root)}

    report = pw.preflight_native_inputs(config_path, env=env, domain_ids=["A"], formal=True)

    domain_report = report.domain_reports[0]
    assert not domain_report.is_run_ready
    assert domain_report.node_count_grid == grid["n_nodes"]
    assert domain_report.node_count_hvel_sample == grid["n_nodes"] - 1
    assert any("節點" in problem for problem in domain_report.problems)


def test_preflight_reports_gap_affecting_arrival_window(tmp_path: Path, ocm_and_nww_roots) -> None:
    """限制 ``inputs.years=[2024]`` 讓「預期涵蓋整個 2024 年」但實際只有 48 小時
    資料的落差變成明確缺口；到達時刻恰好落在資料涵蓋最開頭（30 天回溯窗必然
    往前跨出涵蓋範圍），驗證缺口盤點與到達窗安全性檢查都確實接到既有函式。
    """

    from test_manifests import _arrival_payload, _material_payload, _receptor_payload, _write_json

    payload = _config_payload(with_paths=True)
    payload["scenarios"]["receptor_arrival_initial_condition_manifest"] = None
    payload["inputs"]["years"] = [2024]
    config = ProjectConfig.model_validate(payload)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("placeholder: true\n", encoding="utf-8")
    _write_json(tmp_path / "material.json", _material_payload())
    _write_json(tmp_path / "receptor.json", _receptor_payload(per_site=1))
    _write_json(tmp_path / "arrival.json", _arrival_payload(per_site=1))

    ocm_root, nww_root, _grid = ocm_and_nww_roots
    env = {"OCM_NATIVE_ROOT": str(ocm_root), "NWW_ANALYSIS_ROOT": str(nww_root)}

    report = pw.preflight_native_inputs(
        config, env=env, config_path=config_path, domain_ids=["A"], formal=True
    )

    domain_report = report.domain_reports[0]
    assert domain_report.forcing_gap_report is not None
    assert not domain_report.forcing_gap_report.is_run_ready
    assert domain_report.forcing_gap_report.unregistered_hour_count > 0
    # gongliao／guishan 各一個到達時刻（2024-01-01T00:00Z），30 天回溯窗必然
    # 往前跨出「預期涵蓋起點＝2024-01-01T00:00Z」，因此兩者都應判定不安全。
    assert domain_report.arrival_total_count == 2
    assert domain_report.arrival_unsafe_count == 2
    assert not domain_report.is_run_ready
    assert "缺口" in domain_report.describe()


# ══════════════════════════════════════════════════════════════════════════
# 七、端到端：build_domain_resources 接出的資源直接餵給 run_native_plan
# ══════════════════════════════════════════════════════════════════════════

_WIRING_E2E_CONFIG_PATH_ENV = "LBT_WIRING_E2E_CONFIG_PATH"
_WIRING_E2E_DOMAIN_ID = "A"
_WIRING_E2E_SITE_ID = "gongliao"
_WIRING_E2E_ARRIVAL_UTC_NS = _hour_ns(40)
_WIRING_E2E_SETTLING_MPS = -0.0005
_WIRING_E2E_MAX_BACKTRACK_SECONDS = 4.0 * 3600.0


def _wiring_e2e_particle_table():
    """端到端測試專用、與正式情境矩陣完全無關的極小合成身分表（三顆粒子）。

    起點固定在合成網格中心附近（僅相差數十公尺），確保 4 小時、極小流速下的
    位移量不會讓粒子離開這個約 3 公里見方的合成小網格。
    """

    from test_native_orchestrator import _build_table

    grid = _build_regular_grid()
    center_lon = float(np.mean(grid["node_lon"]))
    center_lat = float(np.mean(grid["node_lat"]))
    projection = DomainProjection(122.05, 25.05)  # 與 config 的 A 區 center_lonlat 相同
    center_x, center_y = projection.project(center_lon, center_lat)
    rows = [
        {
            "study_site_id": _WIRING_E2E_SITE_ID,
            "analysis_region_id": _WIRING_E2E_DOMAIN_ID,
            "arrival_time_utc_ns": _WIRING_E2E_ARRIVAL_UTC_NS,
            "settling_velocity_mps": _WIRING_E2E_SETTLING_MPS,
            "x0_m": float(center_x) + offset,
            "y0_m": float(center_y),
            "z0_m": -5.0,
        }
        for offset in (-30.0, 0.0, 30.0)
    ]
    return _build_table(rows)


def _wiring_e2e_build_worker_resources(worker_id: int) -> dict[str, Any]:
    """``run_native_plan`` 每個工作程序呼叫一次；直接呼叫本工作包的
    ``build_domain_resources``，不手刻網格或驅動層。

    模組層級函式（而非閉包）：spawn 子程序需要能夠重新 import 這個名稱，見
    ``tests/test_native_orchestrator.py`` 模組文件同一段說明。
    """

    from lagrangian_backtracking.native_runner import backward_driver as bd

    config_path = os.environ[_WIRING_E2E_CONFIG_PATH_ENV]
    baytrace_root = os.environ["LBT_BAYTRACE_ROOT"]
    resources = pw.build_domain_resources(config_path, _WIRING_E2E_DOMAIN_ID, env=os.environ, formal=True)
    handles = bd.load_baytrace_modules(baytrace_root)
    wave_resolver = resources.wave_resolver_factory() if resources.wave_resolver_factory is not None else None
    return {
        "handles": handles,
        "driver_grid": resources.driver_grid,
        "forcing_resolver": resources.forcing_resolver_factory(),
        "wave_resolver": wave_resolver,
        "interest_region_polygon": resources.interest_regions[_WIRING_E2E_SITE_ID],
    }


class _WiringE2ESoa:
    """欄位相容的最小 ``BayTraceParticleSoA`` 替身，寫法比照
    ``tests/test_native_backward_driver.py`` 的 ``_SyntheticSoa``。
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, row_particle_index: np.ndarray) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.z0 = z
        self.row_particle_index = row_particle_index


def _wiring_e2e_run_unit(resources: dict[str, Any], spec: orch.NativeWorkUnitSpec) -> dict[str, Any]:
    from lagrangian_backtracking.native_runner import backward_driver as bd

    table = _wiring_e2e_particle_table()
    indices = spec.particle_row_indices
    soa = _WiringE2ESoa(
        x=table.x0_m[indices].copy(),
        y=table.y0_m[indices].copy(),
        z=table.z0_m[indices].copy(),
        row_particle_index=indices,
    )
    config = bd.BackwardDriverConfig(
        arrival_time_utc_ns=int(spec.arrival_time_utc_ns),
        settling_velocity_mps=float(spec.settling_velocity_mps),
        max_backtrack_seconds=_WIRING_E2E_MAX_BACKTRACK_SECONDS,
        enable_diffusion=False,
        master_seed=20260921,
    )
    seed_hex_values = [format(int(row), "032x") for row in indices]
    result = bd.run_backward_batch(
        resources["driver_grid"],
        resources["handles"],
        soa,
        seed_hex_values=seed_hex_values,
        config=config,
        forcing_resolver=resources["forcing_resolver"],
        interest_region_polygon=resources["interest_region_polygon"],
        wave_field_resolver=resources["wave_resolver"],
    )
    return {"particle_count": int(indices.size), "hours_run": int(result.hours_run)}


def test_end_to_end_backward_batch_via_build_domain_resources(
    wiring_config, ocm_and_nww_roots, monkeypatch, tmp_path: Path
) -> None:
    """接出的 ``DomainResources`` 直接餵給 ``run_native_plan``，跑數小時逆向追蹤。

    這是本工作包背景要求的「端到端」驗收：從正式設定＋合成已驗收產品出發，
    不手刻網格或驅動層，確認能透過 ``build_domain_resources`` 接出的資源，
    驅動真正的 ``run_backward_batch`` 完成數小時的逆向追蹤。
    """

    baytrace_root = _require_baytrace_root()
    _config, config_path = wiring_config
    ocm_root, nww_root, _grid = ocm_and_nww_roots

    monkeypatch.setenv(_WIRING_E2E_CONFIG_PATH_ENV, str(config_path))
    monkeypatch.setenv("OCM_NATIVE_ROOT", str(ocm_root))
    monkeypatch.setenv("NWW_ANALYSIS_ROOT", str(nww_root))
    monkeypatch.setenv("LBT_BAYTRACE_ROOT", baytrace_root)
    monkeypatch.delenv("OCM_RECONSTRUCTION_ROOT", raising=False)

    table = _wiring_e2e_particle_table()
    units = orch.plan_native_work_units(
        table, region_to_flow_domain={_WIRING_E2E_DOMAIN_ID: _FLOW_DOMAIN_ID}
    )
    assert len(units) == 1  # 單一站點、單一到達時刻、單一沉降速度 → 一個工作單元

    summary = orch.run_native_plan(
        units,
        worker_count=1,
        build_worker_resources=_wiring_e2e_build_worker_resources,
        run_unit=_wiring_e2e_run_unit,
        output_dir=tmp_path,
    )

    assert summary.completed_unit_count == 1
    assert summary.failed_unit_count == 0, summary.failed_unit_ids

    payload = json.loads((tmp_path / f"{units[0].unit_id}.result.json").read_text(encoding="utf-8"))
    assert payload["particle_count"] == 3
    assert payload["hours_run"] >= 1
