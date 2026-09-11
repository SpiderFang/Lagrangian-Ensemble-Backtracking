"""SERVER v3 input derivation 的 synthetic fixture 與 immutable binding 測試。

測試資料只用小型規則網格與每月前八天的逐時片段，目的是驗證 CLI、schema、時間軸、
受體／到達時次配對、dynamic initial-condition 與 hash closure 的工程連接，不是 OCM 或
NWW3 的科學成果。正式資料仍須在 SERVER 以真實四域產品重新執行同一 CLI 並保留輸入
manifest、source fingerprint、QC 與資源紀錄。
"""

from __future__ import annotations

import json
import shutil
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml
from shapely.geometry import box, shape

import lagrangian_backtracking.input_derivation as input_derivation_module
from lagrangian_backtracking.cli import main
from lagrangian_backtracking.config import ProjectConfig, StudySiteConfig, resolve_flow_domain_id
from lagrangian_backtracking.geometry import DomainProjection
from lagrangian_backtracking.input_derivation import (
    ARTIFACT_FILENAMES,
    InputDerivationError,
    build_input_derivatives,
    create_release_config,
    read_canonical_json,
    validate_input_derivatives,
    validate_release_config,
    write_canonical_json,
)
from lagrangian_backtracking.mesh import NativeMesh
from lagrangian_backtracking.receptors import (
    prepare_horizontal_receptor_candidates,
    select_horizontal_receptors_from_pool,
)
from lagrangian_backtracking.scenarios import ArrivalTime, stable_identifier

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = ROOT / "configs" / "lagrangian_backtracking.example.yaml"


def _rectangular_grid(
    *, lon_min: float, lon_max: float, lat_min: float, lat_max: float, nx: int, ny: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """建立連通的四邊形 source mesh，所有座標仍以 WGS84 經緯度交換。

    回傳的 ``face_nodes`` 使用逆時針四邊形節點順序；NativeMesh 會再依公尺制對角線
    規則拆成三角形。A 區網格刻意跨越兩個 study site，測試同一 flow domain 的共享
    forcing 與兩組獨立 receptor；B-D 則各使用小型中心網格。
    """

    lon_axis = np.linspace(lon_min, lon_max, nx)
    lat_axis = np.linspace(lat_min, lat_max, ny)
    lon_grid, lat_grid = np.meshgrid(lon_axis, lat_axis, indexing="xy")
    node_lon = lon_grid.ravel()
    node_lat = lat_grid.ravel()
    faces: list[list[int]] = []
    for row in range(ny - 1):
        for column in range(nx - 1):
            lower_left = row * nx + column
            lower_right = lower_left + 1
            upper_right = (row + 1) * nx + column + 1
            upper_left = (row + 1) * nx + column
            faces.append([lower_left, lower_right, upper_right, upper_left])
    return node_lon, node_lat, np.asarray(faces, dtype=np.int64)


def _array_contract(shape: tuple[int, ...], dtype: np.dtype[Any]) -> dict[str, object]:
    """把 synthetic NPY header 的 shape/dtype 寫成 preflight 需要的 JSON contract。"""

    return {"shape": list(shape), "dtype": np.dtype(dtype).name}


def _write_ocm_domain(
    root: Path,
    *,
    flow_domain_id: str,
    months: list[str],
    node_lon: np.ndarray,
    node_lat: np.ndarray,
    faces: np.ndarray,
) -> None:
    """寫入一個符合 OCM schema 3 最小讀取契約的 synthetic native domain。"""

    domain_root = root / flow_domain_id
    grid = domain_root / "grid"
    grid.mkdir(parents=True)
    (domain_root / "months").mkdir()
    node_count = int(node_lon.size)
    face_count = int(faces.shape[0])
    np.save(grid / "source_lon.npy", node_lon.astype(np.float64))
    np.save(grid / "source_lat.npy", node_lat.astype(np.float64))
    np.save(grid / "source_face_nodes_local.npy", faces)
    np.save(grid / "source_face_node_count.npy", np.full(face_count, 4, dtype=np.int64))
    np.save(grid / "source_depth_m.npy", np.full(node_count, 20.0, dtype=np.float64))
    np.save(grid / "source_node_bottom_index.npy", np.full(node_count, 3, dtype=np.int64))
    np.save(grid / "source_face_global_index.npy", np.arange(face_count, dtype=np.int64) + 10_000)
    (grid / "metadata.json").write_text(
        json.dumps(
            {
                "cache_schema_version": "3.0.0",
                "domain": {"domain_id": flow_domain_id},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    layers = np.asarray([-20.0, -13.333333, -6.666667, 0.0], dtype=np.float32)
    for month in months:
        year, number = int(month[:4]), int(month[4:])
        start_ns = int(datetime(year, number, 1, tzinfo=UTC).timestamp() * 1_000_000_000)
        time_ns = start_ns + np.arange(8 * 24, dtype=np.int64) * 3_600_000_000_000
        time_count = int(time_ns.size)
        month_dir = domain_root / "months" / month
        month_dir.mkdir()
        shape_hvel = (time_count, node_count, 4, 2)
        shape_node_layer = (time_count, node_count, 4)
        shape_node = (time_count, node_count)
        shape_face = (time_count, face_count)
        hvel = np.zeros(shape_hvel, dtype=np.float32)
        hvel[..., 0] = 0.1
        arrays = {
            "time_utc_ns.npy": time_ns,
            "hvel.npy": hvel,
            "vertical_velocity.npy": np.zeros(shape_node_layer, dtype=np.float32),
            "zcor.npy": np.broadcast_to(layers, shape_node_layer).copy(),
            "elev.npy": np.zeros(shape_node, dtype=np.float32),
            "wetdry_elem.npy": np.zeros(shape_face, dtype=np.int8),
            "diffusivity.npy": np.zeros(shape_node_layer, dtype=np.float32),
        }
        for name, array in arrays.items():
            np.save(month_dir / name, array)
        (month_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "cache_schema_version": "3.0.0",
                    "domain": {"domain_id": flow_domain_id},
                    "month": month,
                    "status": "ready",
                    "cache_kind": "standard_month",
                    "arrays": {
                        name: _array_contract(tuple(array.shape), array.dtype)
                        for name, array in arrays.items()
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )


def _write_nww_domain(
    root: Path,
    *,
    flow_domain_id: str,
    months: list[str],
    center_lon: float,
    center_lat: float,
    regular_axes: bool = False,
) -> None:
    """寫入符合 NWW3 analysis schema 1 的最小 2×2 analysis grid。

    ``regular_axes`` 可切換成 runtime 支援的一維嚴格遞增 lon／lat 軸，或刻意建立二維
    座標網格供 fail-closed 測試；正式 NWW gate 只允許前者。這些檔案只供純工程測試，
    不代表 accepted NWW3 科學產品。
    """

    domain_root = root / flow_domain_id
    grid = domain_root / "grid"
    grid.mkdir(parents=True)
    (domain_root / "months").mkdir()
    lon_axis = np.asarray([center_lon - 0.5, center_lon + 0.5], dtype=np.float64)
    lat_axis = np.asarray([center_lat - 0.5, center_lat + 0.5], dtype=np.float64)
    if regular_axes:
        # 一維軸由 production helper 展開成四角座標；static mask 仍以二維 spatial shape
        # 保存，讓測試能明確區分四角靜態支撐與單一 nearest dynamic mask。
        grid_arrays = {
            "lon.npy": lon_axis,
            "lat.npy": lat_axis,
            "mask_static.npy": np.ones((2, 2), dtype=np.uint8),
        }
    else:
        lon, lat = np.meshgrid(lon_axis, lat_axis, indexing="xy")
        grid_arrays = {
            "lon.npy": lon,
            "lat.npy": lat,
            "mask_static.npy": np.ones((2, 2), dtype=np.uint8),
        }
    for name, array in grid_arrays.items():
        np.save(grid / name, array)
    (grid / "metadata.json").write_text(
        json.dumps(
            {"schema_version": "1.2.0", "flow_domain_id": flow_domain_id},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    for month in months:
        year, number = int(month[:4]), int(month[4:])
        start_ns = int(datetime(year, number, 1, tzinfo=UTC).timestamp() * 1_000_000_000)
        time_ns = start_ns + np.arange(8 * 24, dtype=np.int64) * 3_600_000_000_000
        time_count = int(time_ns.size)
        month_dir = domain_root / "months" / month
        month_dir.mkdir()
        arrays = {
            "time_utc_ns.npy": time_ns,
            "significant_wave_height.npy": np.ones((time_count, 2, 2), dtype=np.float32),
            "peak_frequency.npy": np.full((time_count, 2, 2), 0.1, dtype=np.float32),
            "peak_direction_raw_deg.npy": np.full((time_count, 2, 2), 90.0, dtype=np.float32),
            "valid_mask_wave.npy": np.ones((time_count, 2, 2), dtype=np.uint8),
            "qc_flags.npy": np.zeros((time_count, 2, 2), dtype=np.int16),
        }
        for name, array in arrays.items():
            np.save(month_dir / name, array)
        (month_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.2.0",
                    "flow_domain_id": flow_domain_id,
                    "month": month,
                    "status": "ready",
                    "cache_kind": "ocm_analysis_grid_resample_from_nww3_native",
                    "arrays": {
                        name: _array_contract(tuple(array.shape), array.dtype)
                        for name, array in arrays.items()
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )


def _write_surface_domain(
    root: Path,
    *,
    flow_domain_id: str,
    months: list[str],
    center_lon: float,
    center_lat: float,
) -> None:
    """寫入供 arrival selector 使用的 OCM schema 3 surface cache fixture。

    這個 fixture 明確提供 surface 的規則格網、靜態海陸遮罩與逐時 u/v、surface/eta、
    valid mask、QC 及 UTC 軸。它與 native mesh 分離，測試可因此確認 arrival scalar
    不是從 native 全域 hvel 推導；dynamic initial-condition 仍會使用 native 的必要
    zcor/elev/wetdry 切片。
    """

    domain_root = root / flow_domain_id
    grid = domain_root / "grid"
    (domain_root / "months").mkdir(parents=True)
    grid.mkdir(parents=True)
    lon_axis = np.asarray([center_lon - 0.5, center_lon + 0.5], dtype=np.float64)
    lat_axis = np.asarray([center_lat - 0.5, center_lat + 0.5], dtype=np.float64)
    lon, lat = np.meshgrid(lon_axis, lat_axis, indexing="xy")
    grid_arrays = {
        "lon.npy": lon,
        "lat.npy": lat,
        "mask_static.npy": np.ones((2, 2), dtype=np.uint8),
    }
    for name, array in grid_arrays.items():
        np.save(grid / name, array)
    (grid / "metadata.json").write_text(
        json.dumps(
            {
                "cache_schema_version": "3.0.0",
                "domain": {"domain_id": flow_domain_id},
                "source_provenance": {"fixture": "surface"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    for month in months:
        year, number = int(month[:4]), int(month[4:])
        start_ns = int(datetime(year, number, 1, tzinfo=UTC).timestamp() * 1_000_000_000)
        time_ns = start_ns + np.arange(8 * 24, dtype=np.int64) * 3_600_000_000_000
        time_count = int(time_ns.size)
        shape_grid = (time_count, 2, 2)
        arrays = {
            "time_utc_ns.npy": time_ns,
            "u_surface_mps.npy": np.full(shape_grid, 0.1, dtype=np.float32),
            "v_surface_mps.npy": np.zeros(shape_grid, dtype=np.float32),
            "surface_z.npy": np.zeros(shape_grid, dtype=np.float32),
            "eta_m.npy": np.zeros(shape_grid, dtype=np.float32),
            "valid_mask_surface.npy": np.ones(shape_grid, dtype=np.uint8),
            "qc_flags.npy": np.zeros(shape_grid, dtype=np.int16),
        }
        month_dir = domain_root / "months" / month
        month_dir.mkdir()
        for name, array in arrays.items():
            np.save(month_dir / name, array)
        (month_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "cache_schema_version": "3.0.0",
                    "domain": {"domain_id": flow_domain_id},
                    "month": month,
                    "status": "ready",
                    "cache_kind": "standard_month",
                    "source_provenance": {"fixture": "surface"},
                    "arrays": {
                        name: _array_contract(tuple(array.shape), array.dtype)
                        for name, array in arrays.items()
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )


@pytest.fixture()
def synthetic_input_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    """建立兩年、四域、每月八天逐時的 synthetic source 與可載入 config。"""

    payload = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    # 這組 coarse 8×10/2×2 mesh 只驗證既有 schema 連接，不能冒充新 policy 所要求的
    # 三套 forcing 共同 margin 證據；明示 legacy policy 以保留 margin=0 的 synthetic
    # fixture 語意與舊 geometry／scenario identity。
    payload["design_version"] = "design_baseline_v2_non_rising_oca_proxy"
    payload["domains"][0]["formal_domain_policy"] = "expanded_domain_v1"
    payload["domains"][0]["expanded_domain_candidate_id"] = (
        "northeast_taiwan_common_cache_v4_lbt_south_expanded"
    )
    payload["domains"][0]["expanded_bbox_lon_lat"] = [121.306315, 122.793685, 24.480000, 25.499156]
    payload["domains"][0]["formal_release_flow_domain_id"] = None
    payload["domains"][0]["formal_release_domain_status"] = "expanded_domain_generation_required"
    # 將 synthetic mesh 的 face scale 設為零 margin；正式資料仍由 config 登錄的兩格
    # margin gate 決定。這個測試只隔離 receptor／dynamic 的 schema 連接，不放寬 production
    # 預設值本身。
    for site in payload["study_sites"]:
        if site["analysis_region_id"] == "A":
            site["formal_release_flow_domain_id"] = None
            site["local_domain_baseline_radius_m"] = 25000
            site["local_domain_sensitivity_radii_m"] = [20000, 35000]
            site["minimum_flow_domain_margin_local_grid_scales"] = 0
            site["radius_35000_requires_expanded_flow_domain"] = True
        if site["study_site_id"] == "houwan":
            # 多數 synthetic 測試驗證舊版核心／全域 selector 的相容路徑；紅框契約
            # 另由專用測試以明示 candidate regions 驗證，避免低解析度 fixture 用
            # 不覆蓋西岸紅框的網格錯誤代表正式資料不足。
            site.pop("receptor_candidate_regions", None)
            site.pop("receptor_candidate_selection", None)
            site.pop("receptor_candidate_regions_provenance", None)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    months = [f"{year}{month:02d}" for year in (2024, 2025) for month in range(1, 13)]
    ocm_root = tmp_path / "ocm_native"
    surface_root = tmp_path / "ocm_surface"
    nww_root = tmp_path / "nww_analysis"
    domain_specs = {
        "A": ("northeast_taiwan_common_cache_v3", (121.70, 122.20, 24.55, 25.30), (8, 10)),
        # B 區設定的 12.5 km receptor core 只縮小受體候選，不縮小 flow/local polygon；
        # 測試網格因此提高面數，讓核心內仍有至少五個 persistent-wet face 可供正式
        # 5×4 receptor contract 選點。這是 synthetic fixture 的解析度調整，不代表
        # SERVER accepted mesh 的實際水平解析度。
        "B": ("hsinchu_cache_v3", (120.25, 120.65, 24.55, 24.95), (7, 7)),
        # C 區舊版相容測試會在 fixture 上另行明示 12.5 km receptor core；紅框專用測試
        # 使用真實候選 polygon／配額契約，避免以低解析度 fixture 偽造 production
        # 紅框的「候選不足」資料限制。
        "C": ("houwan_nmmba_cache_v3", (120.70, 121.10, 21.80, 22.20), (7, 7)),
        # D 區現在也明示 12.5 km core；提高 synthetic 網格解析度，避免 coarse face
        # 尺度讓核心內只剩一個 persistent-wet face，這是 fixture 解析度限制而非正式
        # receptor 配額的放寬。
        "D": ("lienchiang_common_cache_v3", (119.75, 120.15, 26.00, 26.40), (7, 7)),
    }
    for domain in payload["domains"]:
        region = domain["analysis_region_id"]
        flow_id, bbox, shape = domain_specs[region]
        node_lon, node_lat, faces = _rectangular_grid(
            lon_min=bbox[0], lon_max=bbox[1], lat_min=bbox[2], lat_max=bbox[3], nx=shape[0], ny=shape[1]
        )
        _write_ocm_domain(
            ocm_root,
            flow_domain_id=flow_id,
            months=months,
            node_lon=node_lon,
            node_lat=node_lat,
            faces=faces,
        )
        _write_surface_domain(
            surface_root,
            flow_domain_id=flow_id,
            months=months,
            center_lon=float(domain["center_lonlat"][0]),
            center_lat=float(domain["center_lonlat"][1]),
        )
        _write_nww_domain(
            nww_root,
            flow_domain_id=flow_id,
            months=months,
            center_lon=float(domain["center_lonlat"][0]),
            center_lat=float(domain["center_lonlat"][1]),
            regular_axes=True,
        )
    return config_path, ocm_root, surface_root, nww_root, tmp_path


def test_canonical_json_binding_is_immutable_and_detects_tamper(tmp_path: Path) -> None:
    """canonical JSON 的 raw/canonical hash 與 sidecar 應拒絕改寫與內容竄改。"""

    path = tmp_path / "component.json"
    write_canonical_json(path, {"z": 2, "a": 1})
    payload, fingerprint = read_canonical_json(path)
    assert payload == {"a": 1, "z": 2}
    assert len(fingerprint["sha256"]) == 64
    with pytest.raises(FileExistsError):
        write_canonical_json(path, {"another": "payload"})
    path.write_text(path.read_text(encoding="utf-8").replace('"z": 2', '"z": 3'), encoding="utf-8")
    with pytest.raises(ValueError, match="binding"):
        read_canonical_json(path)


def test_nww_cache_kind_allowlist_accepts_current_and_rejects_unknown() -> None:
    """NWW allowlist 應保留舊名稱、接受目前 SERVER kind，並拒絕未登錄名稱。"""

    payload = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    config = ProjectConfig.model_validate(payload)
    current_kind = "ocm_spatial_grid_resample_from_complete_available_nww3_archive_native_time_v1"
    accepted_kinds = config.inputs.nww_contract["accepted_cache_kinds"]
    assert "ocm_analysis_grid_resample_from_nww3_native" in accepted_kinds
    assert current_kind in accepted_kinds

    metadata = {
        "schema_version": "1.2.0",
        "flow_domain_id": "northeast_taiwan_common_cache_v3",
        "month": "202401",
        "status": "ready",
    }
    for cache_kind in accepted_kinds:
        candidate = {**metadata, "cache_kind": cache_kind}
        input_derivation_module._validate_month_metadata(
            "nww3_analysis",
            "northeast_taiwan_common_cache_v3",
            "202401",
            candidate,
            config,
        )
    with pytest.raises(InputDerivationError, match="cache_kind"):
        input_derivation_module._validate_month_metadata(
            "nww3_analysis",
            "northeast_taiwan_common_cache_v3",
            "202401",
            {**metadata, "cache_kind": "unregistered_nww_cache_kind"},
            config,
        )


def test_nww_arrival_selector_uses_runtime_exact_hour_four_corner_support(
    tmp_path: Path,
) -> None:
    """arrival selector 必須重現 runtime 的 exact-hour 四角 mask、物理與雙線性 gate。"""

    payload = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    config = ProjectConfig.model_validate(payload)
    flow_domain_id = "nww_selector_test_domain"
    nww_root = tmp_path / "nww"
    _write_nww_domain(
        nww_root,
        flow_domain_id=flow_domain_id,
        months=["202401"],
        center_lon=0.0,
        center_lat=0.0,
        regular_axes=True,
    )
    product = input_derivation_module._load_product(
        product="nww3_analysis",
        root=nww_root,
        root_token="NWW_ANALYSIS_ROOT",
        flow_domain_id=flow_domain_id,
        months=["202401"],
        config=config,
    )
    month_dir = nww_root / flow_domain_id / "months" / "202401"
    wave_height = np.load(month_dir / "significant_wave_height.npy", mmap_mode="r+")
    frequency = np.load(month_dir / "peak_frequency.npy", mmap_mode="r+")
    direction = np.load(month_dir / "peak_direction_raw_deg.npy", mmap_mode="r+")
    wave_mask = np.load(month_dir / "valid_mask_wave.npy", mmap_mode="r+")
    wave_height.reshape(wave_height.shape[0], -1)[:] = np.asarray([1.0, 3.0, 5.0, 7.0])
    frequency[...] = 0.1
    direction[...] = 90.0
    wave_mask[...] = True
    wave_height.flush()
    frequency.flush()
    direction.flush()
    wave_mask.flush()

    # 0.0, 0.0 位於四角中心，runtime 的 x/y 權重各為 0.5；四角 Hs 的雙線性結果是
    # (1+3+5+7)/4=4.0，而不是任何一個 nearest scalar。
    series, available = input_derivation_module._nww_series_for_location(
        product,
        lon=0.0,
        lat=0.0,
    )
    assert all(available.values())
    assert all(value == pytest.approx(4.0) for value in series.values())

    # 只要四角任一 UTC dynamic mask 為 False，不能忽略該角或搜尋其他「最近有效」格點。
    wave_mask[...] = True
    wave_mask.reshape(wave_mask.shape[0], -1)[:, 0] = False
    wave_mask.flush()
    series, available = input_derivation_module._nww_series_for_location(
        product,
        lon=0.0,
        lat=0.0,
    )
    assert not any(available.values())
    assert all(np.isnan(value) for value in series.values())

    wave_mask[...] = True
    wave_mask.flush()
    wave_height.reshape(wave_height.shape[0], -1)[:, 0] = np.nan
    wave_height.flush()
    _, hs_nonfinite = input_derivation_module._nww_series_for_location(
        product,
        lon=0.0,
        lat=0.0,
    )
    assert not any(hs_nonfinite.values())

    wave_height.reshape(wave_height.shape[0], -1)[:] = -1.0
    wave_height.flush()
    _, hs_negative = input_derivation_module._nww_series_for_location(
        product,
        lon=0.0,
        lat=0.0,
    )
    assert not any(hs_negative.values())

    wave_height.reshape(wave_height.shape[0], -1)[:] = 1.0
    wave_height.flush()
    frequency[...] = 0.1
    frequency.reshape(frequency.shape[0], -1)[:, 0] = np.nan
    frequency.flush()
    _, fp_nonfinite = input_derivation_module._nww_series_for_location(
        product,
        lon=0.0,
        lat=0.0,
    )
    assert not any(fp_nonfinite.values())

    frequency[...] = 0.0
    frequency.flush()
    _, fp_invalid = input_derivation_module._nww_series_for_location(
        product,
        lon=0.0,
        lat=0.0,
    )
    assert not any(fp_invalid.values())

    frequency[...] = 0.1
    frequency.flush()
    direction[...] = 90.0
    direction.reshape(direction.shape[0], -1)[:, 0] = np.nan
    direction.flush()
    _, direction_nonfinite = input_derivation_module._nww_series_for_location(
        product,
        lon=0.0,
        lat=0.0,
    )
    assert not any(direction_nonfinite.values())

    direction.reshape(direction.shape[0], -1)[:] = np.asarray([0.0, 90.0, 180.0, 270.0])
    direction.flush()
    _, direction_degenerate = input_derivation_module._nww_series_for_location(
        product,
        lon=0.0,
        lat=0.0,
    )
    assert not any(direction_degenerate.values())

    direction[...] = 90.0
    direction.flush()
    static_mask = np.load(nww_root / flow_domain_id / "grid" / "mask_static.npy", mmap_mode="r+")
    static_mask[...] = True
    static_mask.ravel()[0] = False
    static_mask.flush()
    _, static_invalid = input_derivation_module._nww_series_for_location(
        product,
        lon=0.0,
        lat=0.0,
    )
    assert not any(static_invalid.values())

    static_mask[...] = True
    static_mask.flush()
    _, outside = input_derivation_module._nww_series_for_location(
        product,
        lon=2.0,
        lat=2.0,
    )
    assert not any(outside.values())


def test_nww_series_rejects_two_dimensional_runtime_unsupported_grid(tmp_path: Path) -> None:
    """NWW 二維座標網格不得被 input gate 默許成 runtime 可重現的規則格網。"""

    payload = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    config = ProjectConfig.model_validate(payload)
    flow_domain_id = "nww_two_dimensional_grid_test_domain"
    nww_root = tmp_path / "nww"
    _write_nww_domain(
        nww_root,
        flow_domain_id=flow_domain_id,
        months=["202401"],
        center_lon=0.0,
        center_lat=0.0,
        regular_axes=False,
    )
    product = input_derivation_module._load_product(
        product="nww3_analysis",
        root=nww_root,
        root_token="NWW_ANALYSIS_ROOT",
        flow_domain_id=flow_domain_id,
        months=["202401"],
        config=config,
    )
    with pytest.raises(InputDerivationError, match="一維"):
        input_derivation_module._nww_series_for_location(product, lon=0.0, lat=0.0)


def test_nww_exact_hour_sample_indexes_only_requested_rows_and_four_corners(
    tmp_path: Path,
) -> None:
    """NWW 向量化 helper 不得先切出整月網格，只能索引 requested rows×4 corners。"""

    payload = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    config = ProjectConfig.model_validate(payload)
    flow_domain_id = "nww_four_corner_index_test_domain"
    nww_root = tmp_path / "nww"
    _write_nww_domain(
        nww_root,
        flow_domain_id=flow_domain_id,
        months=["202401"],
        center_lon=0.0,
        center_lat=0.0,
        regular_axes=True,
    )
    product = input_derivation_module._load_product(
        product="nww3_analysis",
        root=nww_root,
        root_token="NWW_ANALYSIS_ROOT",
        flow_domain_id=flow_domain_id,
        months=["202401"],
        config=config,
    )
    cache = input_derivation_module._load_nww_runtime_cache(product)
    support = input_derivation_module._nww_runtime_spatial_support(
        cache,
        lon=0.0,
        lat=0.0,
    )
    assert support is not None
    corner_y, corner_x, spatial_weights, static_valid = support

    class IndexedArray:
        """攔截 NPY 索引，拒絕任何沒有同時指定三個 broadcast 維度的取法。"""

        def __init__(self, values: np.ndarray) -> None:
            self.values = values
            self.index_shapes: list[tuple[tuple[int, ...], ...]] = []

        def __getitem__(self, key: object) -> np.ndarray:
            """只放行 (rows, corner_y, corner_x) 的三維 advanced indexing。"""

            if not isinstance(key, tuple) or len(key) != 3:
                raise AssertionError("NWW helper 不得先以 rows materialize 整月陣列")
            shapes = tuple(np.shape(item) for item in key)
            self.index_shapes.append(shapes)
            assert shapes == ((2, 1), (1, 4), (1, 4))
            return self.values[key]

    source_month = cache.months[0]
    tracked = [
        IndexedArray(np.asarray(source_month.significant_wave_height)),
        IndexedArray(np.asarray(source_month.peak_frequency)),
        IndexedArray(np.asarray(source_month.peak_direction_raw_deg)),
        IndexedArray(np.asarray(source_month.valid_mask_wave)),
    ]
    tracked_month = input_derivation_module._NWWRuntimeMonth(
        source=source_month.source,
        significant_wave_height=tracked[0],
        peak_frequency=tracked[1],
        peak_direction_raw_deg=tracked[2],
        valid_mask_wave=tracked[3],
    )
    sampled_hs, sampled_valid = input_derivation_module._nww_exact_hour_sample_rows(
        tracked_month,
        np.asarray([0, 1], dtype=np.int64),
        corner_y=corner_y,
        corner_x=corner_x,
        spatial_weights=spatial_weights,
        static_valid=static_valid,
    )
    assert sampled_hs.shape == (2,)
    assert sampled_valid.shape == (2,)
    assert all(item.index_shapes == [((2, 1), (1, 4), (1, 4))] for item in tracked)


def _metric_location_test_cache() -> input_derivation_module._NWWRuntimeCache:
    """建立只含座標軸與 static mask 的 NWW cache，供 metric policy 純函式測試使用。"""

    axis = np.asarray([0.0, 0.1, 0.2, 0.3], dtype=np.float64)
    return input_derivation_module._NWWRuntimeCache(
        lon_axis=axis.copy(),
        lat_axis=axis.copy(),
        static_mask=np.ones((axis.size, axis.size), dtype=bool),
        months=(),
    )


def _metric_location_test_arrivals(site_id: str, *, start: int = 0) -> list[ArrivalTime]:
    """建立 50 筆最小 arrival records，讓測試只驗證 selector 參數與 location binding。"""

    return [
        ArrivalTime(
            arrival_time_id=f"arr-{site_id}-{index}",
            study_site_id=site_id,
            time_utc_ns=start + index,
            year=2024,
            season="DJF",
            tide_class="spring_proxy",
            phase_or_event=f"phase-{index}",
            metadata={},
        )
        for index in range(50)
    ]


def test_nww_metric_location_anchor_first_preserves_runtime_cell_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """anchor 完整支援時不得 snap，且每筆 arrival 都保存 anchor 的 runtime cell binding。"""

    cache = _metric_location_test_cache()
    selector_calls: list[dict[str, object]] = []

    def fake_series(
        product: object,
        *,
        lon: float,
        lat: float,
        cache: object,
    ) -> tuple[dict[int, float], dict[int, bool]]:
        """提供全 UTC 有效的 fake NWW series，不讀取任何月份檔案。"""

        del product, lon, lat, cache
        return ({index: 1.0 for index in range(50)}, {index: True for index in range(50)})

    def fake_selector(**kwargs: object) -> list[ArrivalTime]:
        """記錄 strict 旗標並回傳恰好 50 筆，模擬 anchor 完整 selector 成功。"""

        selector_calls.append(kwargs)
        return _metric_location_test_arrivals("anchor")

    monkeypatch.setattr(input_derivation_module, "_nww_series_for_location", fake_series)
    monkeypatch.setattr(input_derivation_module, "_select_arrivals_for_site", fake_selector)
    selected, context = input_derivation_module._select_arrivals_with_nww_metric_location(
        site_id="anchor",
        anchor_lon=0.05,
        anchor_lat=0.05,
        local_polygon_lonlat=input_derivation_module.Polygon(
            [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
        ),
        projection=input_derivation_module.DomainProjection(0.15, 0.15),
        product=object(),
        nww_product=object(),
        nww_cache=cache,
        ocm_elevation={index: 0.0 for index in range(50)},
        ocm_speed={index: 1.0 for index in range(50)},
        max_backtrack_days=7.0,
        design_version="metric-test",
        expected_axis=np.arange(50, dtype=np.int64),
        strict=True,
    )

    binding = context.metric_binding
    assert binding.location_kind == "anchor"
    assert binding.anchor_distance_m == 0.0
    assert (binding.cell_x0, binding.cell_x1, binding.cell_y0, binding.cell_y1) == (0, 1, 0, 1)
    assert binding.policy_id == input_derivation_module.NWW_METRIC_LOCATION_POLICY_ID
    assert binding.maximum_snap_distance_m == pytest.approx(
        2.0 * binding.representative_grid_scale_m
    )
    assert len(selected) == 50
    assert all(item.metadata["metric_location_kind"] == "anchor" for item in selected)
    assert all(item.metadata["metric_location_cell_x0"] == 0 for item in selected)
    assert [call["strict"] for call in selector_calls] == [True]


def test_nww_metric_location_snaps_to_nearest_supported_cell_center_deterministically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """anchor dynamic NWW 失敗時，應依距離再 y0/x0 選 local cell center 並完整 strict 50 筆。"""

    cache = _metric_location_test_cache()
    anchor = (0.10, 0.12)
    selector_locations: list[tuple[float, float, bool]] = []
    series_locations: list[tuple[float, float]] = []

    def fake_series(
        product: object,
        *,
        lon: float,
        lat: float,
        cache: object,
    ) -> tuple[dict[int, float], dict[int, bool]]:
        """提供每個位置相同的有限 NWW 序列，將測試焦點放在 location policy。"""

        del product, cache
        series_locations.append((float(lon), float(lat)))
        return ({index: 1.0 for index in range(50)}, {index: True for index in range(50)})

    def fake_selector(**kwargs: object) -> list[ArrivalTime]:
        """只拒絕 anchor，模擬 anchor dynamic series 不支援而 cell center 可用。"""

        # helper 將位置封裝在 fake series 之外；以 monkeypatch 的 call 順序與 strict 旗標
        # 驗證 anchor 先試，第二次則視為第一個排序後 cell center 成功。
        call_number = len(selector_locations)
        selector_locations.append((float(call_number), float(call_number), bool(kwargs["strict"])))
        if call_number == 0:
            raise ValueError("anchor dynamic NWW unsupported")
        return _metric_location_test_arrivals("cell")

    monkeypatch.setattr(input_derivation_module, "_nww_series_for_location", fake_series)
    monkeypatch.setattr(input_derivation_module, "_select_arrivals_for_site", fake_selector)
    local_polygon = input_derivation_module.Polygon(
        [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
    )
    projection = input_derivation_module.DomainProjection(0.15, 0.15)
    candidates = input_derivation_module._nww_metric_location_candidates(
        cache,
        projection=projection,
        anchor_lon=anchor[0],
        anchor_lat=anchor[1],
        local_polygon_lonlat=local_polygon,
        representative_grid_scale_m=input_derivation_module._nww_representative_grid_scale_m(
            cache,
            projection=projection,
            anchor_lon=anchor[0],
            anchor_lat=anchor[1],
        ),
    )
    selected, context = input_derivation_module._select_arrivals_with_nww_metric_location(
        site_id="cell",
        anchor_lon=anchor[0],
        anchor_lat=anchor[1],
        local_polygon_lonlat=local_polygon,
        projection=projection,
        product=object(),
        nww_product=object(),
        nww_cache=cache,
        ocm_elevation={index: 0.0 for index in range(50)},
        ocm_speed={index: 1.0 for index in range(50)},
        max_backtrack_days=7.0,
        design_version="metric-test",
        expected_axis=np.arange(50, dtype=np.int64),
        strict=True,
    )

    assert candidates == tuple(
        sorted(candidates, key=lambda item: (item.distance_m, item.cell_y0, item.cell_x0))
    )
    assert context.metric_binding.location_kind == "cell_center"
    assert context.metric_binding.lon == pytest.approx(0.15)
    assert context.metric_binding.lat == pytest.approx(0.15)
    assert (context.metric_binding.cell_x0, context.metric_binding.cell_y0) == (1, 1)
    assert (
        context.metric_binding.anchor_distance_m
        <= context.metric_binding.maximum_snap_distance_m
    )
    assert len(selected) == 50
    assert all(item.metadata["metric_location_kind"] == "cell_center" for item in selected)
    assert series_locations[0] == anchor
    assert series_locations[1] == (candidates[0].lon, candidates[0].lat)
    assert selector_locations[0][2] is True
    assert all(item[2] is True for item in selector_locations)


def test_nww_metric_location_all_candidates_fail_closed_without_synthetic_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """anchor 與所有兩格尺度 cell center 都失敗時，必須 fail closed 且不呼叫非 strict fallback。"""

    cache = _metric_location_test_cache()
    strict_calls: list[bool] = []

    def fake_series(
        product: object,
        *,
        lon: float,
        lat: float,
        cache: object,
    ) -> tuple[dict[int, float], dict[int, bool]]:
        """提供有限 NWW 序列，讓每次失敗都明確來自 strict selector。"""

        del product, lon, lat, cache
        return ({index: 1.0 for index in range(50)}, {index: True for index in range(50)})

    def always_fail_selector(**kwargs: object) -> list[ArrivalTime]:
        """記錄 selector strict 模式後拒絕所有位置，禁止 synthetic fallback 混入。"""

        strict_calls.append(bool(kwargs["strict"]))
        raise ValueError("strict selector has fewer than 50 arrivals")

    monkeypatch.setattr(input_derivation_module, "_nww_series_for_location", fake_series)
    monkeypatch.setattr(input_derivation_module, "_select_arrivals_for_site", always_fail_selector)
    with pytest.raises(InputDerivationError, match=r"strict 48\+2 selector"):
        input_derivation_module._select_arrivals_with_nww_metric_location(
            site_id="all-fail",
            anchor_lon=0.10,
            anchor_lat=0.12,
            local_polygon_lonlat=input_derivation_module.Polygon(
                [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
            ),
            projection=input_derivation_module.DomainProjection(0.15, 0.15),
            product=object(),
            nww_product=object(),
            nww_cache=cache,
            ocm_elevation={index: 0.0 for index in range(50)},
            ocm_speed={index: 1.0 for index in range(50)},
            max_backtrack_days=7.0,
            design_version="metric-test",
            expected_axis=np.arange(50, dtype=np.int64),
            strict=True,
        )
    assert strict_calls
    assert set(strict_calls) == {True}


def test_nww_metric_location_candidates_respect_local_polygon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cell-center candidates 不得因距 anchor 近就越過 study-site local polygon。"""

    del monkeypatch
    cache = _metric_location_test_cache()
    projection = input_derivation_module.DomainProjection(0.15, 0.15)
    candidates = input_derivation_module._nww_metric_location_candidates(
        cache,
        projection=projection,
        anchor_lon=0.10,
        anchor_lat=0.12,
        local_polygon_lonlat=input_derivation_module.Polygon(
            [(0.09, 0.09), (0.11, 0.09), (0.11, 0.11), (0.09, 0.11)]
        ),
        representative_grid_scale_m=input_derivation_module._nww_representative_grid_scale_m(
            cache,
            projection=projection,
            anchor_lon=0.10,
            anchor_lat=0.12,
        ),
    )
    assert candidates == ()


def test_paired_a_clone_uses_guishan_metrics_and_rejects_shared_utc_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 區 clone 應使用 guishan 自己指標／binding，且 shared UTC 任一缺支援即 fail closed。"""

    del monkeypatch
    canonical = np.arange(300, dtype=np.int64) * 3_600_000_000_000
    source_arrivals = [
        ArrivalTime(
            arrival_time_id=f"gongliao-{index}",
            study_site_id="gongliao",
            time_utc_ns=int(canonical[170 + index]),
            year=2024,
            season="DJF",
            tide_class="spring_proxy",
            phase_or_event=f"phase-{index}",
            metadata={
                "elevation_m": -10.0,
                "elevation_derivative_mps": 99.0,
                "tidal_strength_proxy_m": 88.0,
                "significant_wave_height_m": 77.0,
                "current_speed_mps": 66.0,
                **{
                    "metric_location_kind": "anchor",
                    "metric_location_policy_id": "gongliao-policy",
                },
            },
        )
        for index in range(50)
    ]
    binding = input_derivation_module.NWWMetricLocationBinding(
        policy_id=input_derivation_module.NWW_METRIC_LOCATION_POLICY_ID,
        location_kind="cell_center",
        lon=0.15,
        lat=0.15,
        anchor_distance_m=100.0,
        representative_grid_scale_m=100.0,
        maximum_snap_distance_m=200.0,
        cell_x0=1,
        cell_x1=2,
        cell_y0=1,
        cell_y1=2,
    )
    context = input_derivation_module._SiteArrivalSelectionContext(
        elevation={int(value): 1000.0 + float(index) for index, value in enumerate(canonical)},
        speed={int(value): 2000.0 + float(index) for index, value in enumerate(canonical)},
        nww_series={int(value): 3000.0 + float(index) for index, value in enumerate(canonical)},
        nww_valid={int(value): True for value in canonical},
        metric_binding=binding,
    )
    product = SimpleNamespace(canonical=SimpleNamespace(time_utc_ns=canonical))
    cloned = input_derivation_module._clone_paired_a_arrivals(
        source_arrivals=source_arrivals,
        guishan_context=context,
        guishan_product=product,
        expected_axis=canonical,
        max_backtrack_days=7.0,
        design_version="paired-test",
        strict=True,
    )

    assert [item.time_utc_ns for item in cloned] == [item.time_utc_ns for item in source_arrivals]
    assert all(item.study_site_id == "guishan" for item in cloned)
    assert all(item.metadata["elevation_m"] != -10.0 for item in cloned)
    assert cloned[0].metadata["significant_wave_height_m"] == pytest.approx(3170.0)
    assert cloned[0].metadata["current_speed_mps"] == pytest.approx(2170.0)
    assert cloned[0].metadata["metric_location_kind"] == "cell_center"
    assert cloned[0].metadata["metric_location_policy_id"] == binding.policy_id
    assert cloned[0].metadata["shared_A_forcing_utc"] == "true"
    assert cloned[0].metadata["shared_A_forcing_reference_site"] == "gongliao"
    assert cloned[0].metadata["tide_phase_label_source"] == "gongliao_paired_design_inherited"
    assert all("elevation_derivative_mps" not in item.metadata for item in cloned)
    assert all("tidal_strength_proxy_m" not in item.metadata for item in cloned)

    unsupported_time = int(source_arrivals[3].time_utc_ns)
    invalid_context = input_derivation_module._SiteArrivalSelectionContext(
        elevation=context.elevation,
        speed=context.speed,
        nww_series=context.nww_series,
        nww_valid={**context.nww_valid, unsupported_time: False},
        metric_binding=binding,
    )
    with pytest.raises(InputDerivationError, match="paired UTC"):
        input_derivation_module._clone_paired_a_arrivals(
            source_arrivals=source_arrivals,
            guishan_context=invalid_context,
            guishan_product=product,
            expected_axis=canonical,
            max_backtrack_days=7.0,
            design_version="paired-test",
            strict=True,
        )


def test_receptor_nww_reselection_prepares_one_pool_per_site_and_caches_support(
    synthetic_input_fixture: tuple[Path, Path, Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NWW 淘汰後應由同一本站 pool deterministic 重選，成功 face 不重做 support I/O。"""

    config_path, ocm_root, surface_root, nww_root, root = synthetic_input_fixture
    original_prepare = input_derivation_module.prepare_horizontal_receptor_candidates
    original_select = input_derivation_module.select_horizontal_receptors_from_pool
    original_exact_hour = input_derivation_module._nww_exact_hour_samples
    original_load_nww_cache = input_derivation_module._load_nww_runtime_cache
    prepare_calls: list[str] = []
    nww_cache_loads: list[str] = []
    pools: dict[str, object] = {}
    selector_calls: dict[str, list[tuple[int, ...]]] = {}
    face_positions: dict[tuple[float, float], tuple[str, int]] = {}
    bad_positions: dict[tuple[float, float], tuple[str, int]] = {}
    support_calls: dict[tuple[str, int], int] = {}

    def prepare_spy(**kwargs: object) -> object:
        """記錄每個站點的 pool 建立次數，再交回正式一次性 geometry 掃描結果。"""

        pool = original_prepare(**kwargs)  # type: ignore[arg-type]
        site_id = str(kwargs["study_site_id"])
        prepare_calls.append(site_id)
        pools[site_id] = pool
        return pool

    def select_spy(pool: object, *, count: int = 5, excluded_face_indices: object = ()) -> list[object]:
        """保留正式 pool selector 結果，並記錄每輪排除集合後的 face 順序。"""

        selected = original_select(  # type: ignore[arg-type]
            pool,
            count=count,
            excluded_face_indices=excluded_face_indices,  # type: ignore[arg-type]
        )
        site_id = str(pool.study_site_id)  # type: ignore[attr-defined]
        face_list = tuple(item.source_face_local_index for item in selected)
        selector_calls.setdefault(site_id, []).append(face_list)
        for item in selected:
            face_positions[(float(item.lon), float(item.lat))] = (
                site_id,
                int(item.source_face_local_index),
            )
        if len(selector_calls[site_id]) == 1:
            first = selected[0]
            bad_positions[(float(first.lon), float(first.lat))] = (
                site_id,
                int(first.source_face_local_index),
            )
        return selected  # type: ignore[return-value]

    def exact_hour_spy(
        cache: object,
        *,
        lon: float,
        lat: float,
        requested_times: object = None,
    ) -> tuple[dict[int, float], dict[int, bool]]:
        """只讓每站第一輪第一個 face 失敗，arrival 全軸仍使用正式 exact-hour helper。"""

        if requested_times is None:
            return original_exact_hour(cache, lon=lon, lat=lat)  # type: ignore[arg-type]
        times = tuple(int(value) for value in requested_times)  # type: ignore[union-attr]
        position = (float(lon), float(lat))
        face = face_positions.get(position)
        if face is not None:
            support_calls[face] = support_calls.get(face, 0) + 1
        supported = position not in bad_positions
        return (
            {value: 1.0 if supported else float("nan") for value in times},
            {value: supported for value in times},
        )

    def load_nww_cache_spy(product: object) -> object:
        """記錄每個 analysis region 的 NWW cache 建立，確認 arrival／receptor 共用。"""

        nww_cache_loads.append(str(product.flow_domain_id))  # type: ignore[attr-defined]
        return original_load_nww_cache(product)  # type: ignore[arg-type]

    monkeypatch.setattr(input_derivation_module, "prepare_horizontal_receptor_candidates", prepare_spy)
    monkeypatch.setattr(input_derivation_module, "select_horizontal_receptors_from_pool", select_spy)
    monkeypatch.setattr(input_derivation_module, "_nww_exact_hour_samples", exact_hour_spy)
    monkeypatch.setattr(input_derivation_module, "_load_nww_runtime_cache", load_nww_cache_spy)
    input_derivation_module.build_input_derivatives(
        config_path=config_path,
        destination=root / "derived-inputs-nww-reselect",
        ocm_native_root=ocm_root,
        ocm_surface_root=surface_root,
        nww_analysis_root=nww_root,
        formal=False,
    )

    expected_sites = {"gongliao", "guishan", "hsinchu", "houwan", "lienchiang"}
    assert sorted(nww_cache_loads) == [
        "houwan_nmmba_cache_v3",
        "hsinchu_cache_v3",
        "lienchiang_common_cache_v3",
        "northeast_taiwan_common_cache_v3",
    ]
    assert set(prepare_calls) == expected_sites
    assert len(prepare_calls) == len(expected_sites)
    for site_id in sorted(expected_sites):
        calls = selector_calls[site_id]
        assert len(calls) == 2
        first_face = calls[0][0]
        assert first_face not in calls[1]
        expected = original_select(  # type: ignore[arg-type]
            pools[site_id],
            count=5,
            excluded_face_indices={first_face},
        )
        assert calls[1] == tuple(item.source_face_local_index for item in expected)
        shared_faces = set(calls[0]).intersection(calls[1])
        # 核心圓可能讓 deterministic maximin 在 blacklist 後改選完全不同的五面，
        # 因此不再把「兩輪必須重疊」當成幾何前提；只要同一 face 重返選集，support
        # gate 就必須由 cache 保證最多執行一次。未重疊的兩輪也各自只能執行一次。
        observed_faces = set(calls[0]).union(calls[1])
        assert all(support_calls[(site_id, face)] == 1 for face in observed_faces)
        assert all(support_calls[(site_id, face)] == 1 for face in shared_faces)


def test_receptor_nww_common_candidates_below_five_fail_closed(
    synthetic_input_fixture: tuple[Path, Path, Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """若所有 horizontal face 都缺 NWW runtime 支撐，不能放寬 gate 或產生少於五面。"""

    config_path, ocm_root, surface_root, nww_root, root = synthetic_input_fixture
    original_exact_hour = input_derivation_module._nww_exact_hour_samples

    def unsupported_exact_hour(
        cache: object,
        *,
        lon: float,
        lat: float,
        requested_times: object = None,
    ) -> tuple[dict[int, float], dict[int, bool]]:
        """保留 arrival scalar 的正式結果，只讓 receptor 的所有 requested UTC 失敗。"""

        if requested_times is None:
            return original_exact_hour(cache, lon=lon, lat=lat)  # type: ignore[arg-type]
        times = tuple(int(value) for value in requested_times)  # type: ignore[union-attr]
        return (
            {value: float("nan") for value in times},
            {value: False for value in times},
        )

    monkeypatch.setattr(input_derivation_module, "_nww_exact_hour_samples", unsupported_exact_hour)
    with pytest.raises(InputDerivationError, match="候選不足"):
        input_derivation_module.build_input_derivatives(
            config_path=config_path,
            destination=root / "derived-inputs-nww-shortage",
            ocm_native_root=ocm_root,
            ocm_surface_root=surface_root,
            nww_analysis_root=nww_root,
            formal=False,
        )


def test_face_vertical_support_rejects_steep_node_without_target_bracket() -> None:
    """陡峭海床的淺節點若無法包住代表 target，不能被 face 中位數掩蓋。"""

    # 深度與 zcor 以 positive-up 公尺表示；兩個淺節點的最低 z 只有 -10 m，
    # 另兩個深節點到 -50 m，代表 bed=-30 m 的 near-bed target 對淺節點沒有 below。
    zcor = np.asarray(
        [
            [-10.0, -6.666667, -3.333333, 0.0],
            [-10.0, -6.666667, -3.333333, 0.0],
            [-50.0, -33.333333, -16.666667, 0.0],
            [-50.0, -33.333333, -16.666667, 0.0],
        ],
        dtype=np.float64,
    )
    with pytest.raises(InputDerivationError, match="node=.*near_bed"):
        input_derivation_module._build_face_vertical_support(
            zcor_node_layer=zcor,
            node_elev_m=np.zeros(4, dtype=np.float64),
            node_depth_m=np.asarray([10.0, 10.0, 50.0, 50.0], dtype=np.float64),
            vertical_ids=("near_bed",),
        )


def test_face_vertical_support_rejects_any_node_missing_one_side() -> None:
    """只要任一 node 缺少 target 的上側或下側有限 zcor，就必須 fail closed。"""

    zcor = np.asarray(
        [
            [-20.0, -13.333333, -6.666667, 0.0],
            [-20.0, -13.333333, -6.666667, 0.0],
            [-20.0, -13.333333, -6.666667, -3.0],
            [-20.0, -13.333333, -6.666667, 0.0],
        ],
        dtype=np.float64,
    )
    with pytest.raises(InputDerivationError, match="node=2.*upper_water_column"):
        input_derivation_module._build_face_vertical_support(
            zcor_node_layer=zcor,
            node_elev_m=np.zeros(4, dtype=np.float64),
            node_depth_m=np.full(4, 20.0, dtype=np.float64),
            vertical_ids=("upper_water_column",),
        )


def test_face_vertical_support_accepts_common_four_node_column() -> None:
    """所有 node 都有共同雙側支撐時，helper 應保留既有四個 vertical target。"""

    zcor = np.asarray(
        [
            [-20.0, -13.333333, -6.666667, 0.0],
            [-20.0, -13.333333, -6.666667, 0.0],
            [-20.0, -13.333333, -6.666667, 0.0],
            [-20.0, -13.333333, -6.666667, 0.0],
        ],
        dtype=np.float64,
    )
    support = input_derivation_module._build_face_vertical_support(
        zcor_node_layer=zcor,
        node_elev_m=np.zeros(4, dtype=np.float64),
        node_depth_m=np.full(4, 20.0, dtype=np.float64),
        vertical_ids=input_derivation_module._VERTICAL_TARGET_IDS,
    )
    assert [target.vertical_id for target in support.targets] == [
        "upper_water_column",
        "mid_upper_water_column",
        "mid_lower_water_column",
        "near_bed",
    ]
    assert support.bed_z_m_positive_up == -20.0
    assert support.eta_z_m_positive_up == 0.0
    assert all(upper > lower for _vertical_id, lower, upper in support.brackets)


@pytest.mark.parametrize(
    ("bad_field", "bad_value", "expected_message"),
    [
        ("eta", np.nan, "eta"),
        ("depth", np.nan, "source depth"),
        ("depth", 0.0, "source depth"),
        ("depth", -1.0, "source depth"),
    ],
)
def test_face_vertical_support_rejects_invalid_node_surface_or_depth(
    bad_field: str,
    bad_value: float,
    expected_message: str,
) -> None:
    """正式 runtime 不得忽略單一 node 的缺失 eta 或非物理 source depth。"""

    zcor = np.asarray(
        [
            [-20.0, -13.333333, -6.666667, 0.0],
            [-20.0, -13.333333, -6.666667, 0.0],
            [-20.0, -13.333333, -6.666667, 0.0],
            [-20.0, -13.333333, -6.666667, 0.0],
        ],
        dtype=np.float64,
    )
    node_eta = np.zeros(4, dtype=np.float64)
    node_depth = np.full(4, 20.0, dtype=np.float64)
    if bad_field == "eta":
        node_eta[2] = bad_value
    else:
        node_depth[2] = bad_value

    with pytest.raises(InputDerivationError, match=expected_message):
        input_derivation_module._build_face_vertical_support(
            zcor_node_layer=zcor,
            node_elev_m=node_eta,
            node_depth_m=node_depth,
            vertical_ids=("upper_water_column",),
        )


def test_inputs_build_validate_and_release_config(
    synthetic_input_fixture: tuple[Path, Path, Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """完整 synthetic orchestration 應產出固定 counts、tokenized source 與 generated config。"""

    config_path, ocm_root, surface_root, nww_root, root = synthetic_input_fixture
    # 把 native hvel payload 改成非有限值；arrival scalar 若仍依賴 native hvel，這個
    # build 便應失敗。測試只保留 NPY header/size 不變，正好驗證 arrival 已改由獨立的
    # OCM surface cache 提供 u/v，而 native hvel 只保留 accepted-product 結構契約與
    # dynamic／geometry 所需欄位邊界。
    for hvel_path in sorted(ocm_root.glob("*/months/*/hvel.npy")):
        hvel = np.load(hvel_path, mmap_mode="r+")
        hvel[...] = np.nan
        hvel.flush()
    artifact_directory = root / "derived-inputs"
    result = build_input_derivatives(
        config_path=config_path,
        destination=artifact_directory,
        ocm_native_root=ocm_root,
        ocm_surface_root=surface_root,
        nww_analysis_root=nww_root,
        formal=False,
    )
    assert set(result.paths) == set(ARTIFACT_FILENAMES)
    forcing, _ = read_canonical_json(result.paths["forcing_inventory"])
    assert forcing["status"] == "generated"
    assert len(forcing["products"]) == 12
    arrival_payload, _ = read_canonical_json(result.paths["arrival"])
    assert arrival_payload["selection_method_id"] == input_derivation_module.ARRIVAL_SELECTION_METHOD_ID
    assert arrival_payload["provenance"]["method_id"] == input_derivation_module.ARRIVAL_SELECTION_METHOD_ID
    assert all(
        {
            "metric_location_policy_id",
            "metric_location_kind",
            "metric_location_lon",
            "metric_location_lat",
            "metric_location_anchor_distance_m",
            "metric_location_representative_grid_scale_m",
            "metric_location_maximum_snap_distance_m",
            "metric_location_cell_x0",
            "metric_location_cell_x1",
            "metric_location_cell_y0",
            "metric_location_cell_y1",
        }.issubset(record["metadata"])
        for record in arrival_payload["records"]
    )
    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert isinstance(config_payload, dict)
    base_bbox_by_region = {
        domain["analysis_region_id"]: domain["bbox_lon_lat"] for domain in config_payload["domains"]
    }
    assert all(
        product["grid_metadata"]["bbox_lon_lat"] == base_bbox_by_region[product["analysis_region_id"]]
        for product in forcing["products"]
    )
    assert all(
        record["path"].startswith("$OCM_NATIVE_ROOT/")
        for product in forcing["products"]
        if product["product"] == "ocm_native"
        for record in product["files"]
    )
    assert all(
        record["path"].startswith("$NWW_ANALYSIS_ROOT/")
        for product in forcing["products"]
        if product["product"] == "nww3_analysis"
        for record in product["files"]
    )
    assert all(
        record["path"].startswith("$OCM_SURFACE_ROOT/")
        for product in forcing["products"]
        if product["product"] == "ocm_surface"
        for record in product["files"]
    )
    surface_products = [item for item in forcing["products"] if item["product"] == "ocm_surface"]
    assert len(surface_products) == 4
    assert all(
        item["metadata_provenance"]["grid"]["source_provenance"] == {"fixture": "surface"}
        for item in surface_products
    )
    validation = validate_input_derivatives(
        artifact_directory,
        config_path=config_path,
        ocm_native_root=ocm_root,
        ocm_surface_root=surface_root,
        nww_analysis_root=nww_root,
        formal=False,
    )
    assert validation["valid"] is True, validation
    assert validation["summary"]["receptor_count"] == 100
    assert validation["summary"]["arrival_count"] == 250
    assert validation["summary"]["dynamic_initial_condition_count"] == 5_000
    # 以同一份已建立的 artifact 驗證相對目錄入口；config 與三個 accepted root 仍明示
    # 為絕對路徑，故這裡專門測試 artifact component path 的 lexical 絕對化，而不新增
    # 第二套 expensive fixture 或重建任何 synthetic product。
    monkeypatch.chdir(root)
    relative_validation = validate_input_derivatives(
        Path("derived-inputs"),
        config_path=config_path,
        ocm_native_root=ocm_root,
        ocm_surface_root=surface_root,
        nww_analysis_root=nww_root,
        formal=False,
    )
    assert relative_validation["valid"] is True, relative_validation
    assert relative_validation == validation
    assert (
        main(
            [
                "inputs-validate",
                "--input-directory",
                str(artifact_directory),
                "--config",
                str(config_path),
                "--ocm-native-root",
                str(ocm_root),
                "--ocm-surface-root",
                str(surface_root),
                "--nww-analysis-root",
                str(nww_root),
            ]
        )
        == 0
    )

    release_path = root / "release-config.yaml"
    created = create_release_config(
        config_template_path=config_path,
        input_directory=artifact_directory,
        output_path=release_path,
        formal=True,
    )
    assert created["config_status"] == "generated"
    release_validation = validate_release_config(release_path, input_directory=artifact_directory)
    assert release_validation["valid"] is True, release_validation
    assert validate_release_config(release_path)["valid"] is True


def test_hsinchu_explicit_pilot_window_is_deterministic_and_keeps_full_counts(
    synthetic_input_fixture: tuple[Path, Path, Path, Path, Path],
) -> None:
    """明示新竹 pilot 視窗替換一筆 arrival，仍保留 250/5,000 與可驗證 artifact。"""

    config_path, ocm_root, surface_root, nww_root, root = synthetic_input_fixture
    artifact_directory = root / "derived-inputs-hsinchu-pilot"
    result = build_input_derivatives(
        config_path=config_path,
        destination=artifact_directory,
        ocm_native_root=ocm_root,
        ocm_surface_root=surface_root,
        nww_analysis_root=nww_root,
        formal=False,
        pilot_arrival_utc={"hsinchu": "2024-01-02T01:00:00Z"},
    )
    arrival_payload, _ = read_canonical_json(result.paths["arrival"])
    hsinchu_arrivals = [
        row for row in arrival_payload["records"] if row["study_site_id"] == "hsinchu"
    ]
    explicit = [
        row
        for row in hsinchu_arrivals
        if row["metadata"].get("pilot_replacement_policy_id")
        == input_derivation_module.PILOT_EXPLICIT_WINDOW_POLICY_ID
    ]
    assert len(hsinchu_arrivals) == 50
    assert len(explicit) == 1
    explicit_row = explicit[0]
    expected_time_ns = int(datetime(2024, 1, 2, 1, tzinfo=UTC).timestamp() * 1_000_000_000)
    expected_id = stable_identifier(
        "arr",
        [
            "hsinchu",
            "2024-01-02T01:00:00Z",
            input_derivation_module.PILOT_EXPLICIT_WINDOW_POLICY_ID,
            ProjectConfig.model_validate(yaml.safe_load(config_path.read_text(encoding="utf-8"))).design_version,
        ],
    )
    assert explicit_row["arrival_time_id"] == expected_id
    assert explicit_row["time_utc_ns"] == expected_time_ns
    assert explicit_row["tide_class"] == input_derivation_module.PILOT_EXPLICIT_TIDE_CLASS
    assert explicit_row["phase_or_event"] == input_derivation_module.PILOT_EXPLICIT_PHASE_OR_EVENT
    metadata = explicit_row["metadata"]
    assert metadata["explicit_pilot_window"] == (
        "2024-01-01T01:00:00Z/2024-01-02T01:00:00Z/inclusive_1h"
    )
    assert metadata["explicit_pilot_window_expected_step_count"] == 25
    assert metadata["pilot_replaced_arrival_time_id"] != explicit_row["arrival_time_id"]
    assert metadata["pilot_replacement_arrival_time_id"] == explicit_row["arrival_time_id"]

    gap_payload, _ = read_canonical_json(result.paths["ocm_gap_safe_arrival_horizon"])
    pilot_gap = next(row for row in gap_payload["records"] if row["arrival_time_id"] == expected_id)
    assert pilot_gap["horizon_start_utc"] == "2024-01-01T01:00:00Z"
    assert pilot_gap["horizon_end_utc"] == "2024-01-02T01:00:00Z"
    assert pilot_gap["max_backtrack_days"] == pytest.approx(1.0)
    assert pilot_gap["expected_step_count"] == 25
    assert pilot_gap["supported_step_count"] == 25
    assert pilot_gap["crossed_gap"] is False
    assert pilot_gap["missing_utc"] == []

    dynamic_payload, _ = read_canonical_json(result.paths["initial_condition"])
    pilot_pairs = [
        row for row in dynamic_payload["records"] if row["arrival_time_id"] == expected_id
    ]
    assert len(dynamic_payload["records"]) == 5_000
    assert len(pilot_pairs) == 20
    assert arrival_payload["selection_method_id"] == (
        input_derivation_module.PILOT_ARRIVAL_SELECTION_METHOD_ID
    )
    assert arrival_payload["provenance"]["pilot_selection_scope"]["selection_scope"] == "hsinchu_only"

    index_payload, _ = read_canonical_json(artifact_directory / "artifact_index.json")
    assert index_payload["source_bindings"]["pilot_selection_scope"]["arrival_time_id"] == expected_id
    validation = validate_input_derivatives(
        artifact_directory,
        config_path=config_path,
        ocm_native_root=ocm_root,
        ocm_surface_root=surface_root,
        nww_analysis_root=nww_root,
        formal=False,
    )
    assert validation["valid"] is True, validation

    release_path = root / "hsinchu-pilot-release-config.yaml"
    created = create_release_config(
        config_template_path=config_path,
        input_directory=artifact_directory,
        output_path=release_path,
        formal=True,
    )
    assert created["config_status"] == "generated"
    assert any("formal_pilot_explicit_window_not_48_plus_2" in item for item in created["blockers"])
    assert validate_release_config(release_path, input_directory=artifact_directory)["valid"] is True


@pytest.mark.parametrize("missing_product", ["ocm_surface", "nww3_analysis"])
def test_hsinchu_explicit_pilot_rejects_middle_hour_product_support(
    synthetic_input_fixture: tuple[Path, Path, Path, Path, Path],
    missing_product: str,
) -> None:
    """視窗中間任一 OCM surface 或 NWW exact-hour 支援缺失時不得發布 artifact。"""

    config_path, ocm_root, surface_root, nww_root, root = synthetic_input_fixture
    middle_index = 13  # 2024-01-01T13:00:00Z，刻意不是 arrival endpoint。
    if missing_product == "ocm_surface":
        path = surface_root / "hsinchu_cache_v3" / "months" / "202401" / "valid_mask_surface.npy"
        values = np.load(path, mmap_mode="r+")
        values[middle_index, ...] = 0
        values.flush()
    else:
        path = nww_root / "hsinchu_cache_v3" / "months" / "202401" / "valid_mask_wave.npy"
        values = np.load(path, mmap_mode="r+")
        values[middle_index, ...] = 0
        values.flush()

    destination = root / f"derived-inputs-pilot-missing-{missing_product}"
    with pytest.raises(InputDerivationError, match="explicit_24h_window|explicit_pilot_window"):
        build_input_derivatives(
            config_path=config_path,
            destination=destination,
            ocm_native_root=ocm_root,
            ocm_surface_root=surface_root,
            nww_analysis_root=nww_root,
            formal=False,
            pilot_arrival_utc={"hsinchu": "2024-01-02T01:00:00Z"},
        )
    assert not destination.exists()
    assert not list(root.glob(f".{destination.name}.partial-*"))


def test_hsinchu_explicit_pilot_formal_rejects_before_destination_write(
    synthetic_input_fixture: tuple[Path, Path, Path, Path, Path],
) -> None:
    """formal=True 搭配 pilot 入口必須在讀取 source／寫入 destination 前拒絕。"""

    config_path, ocm_root, surface_root, nww_root, root = synthetic_input_fixture
    destination = root / "derived-inputs-formal-pilot-rejected"
    with pytest.raises(InputDerivationError, match="禁止 pilot-only"):
        build_input_derivatives(
            config_path=config_path,
            destination=destination,
            ocm_native_root=ocm_root,
            ocm_surface_root=surface_root,
            nww_analysis_root=nww_root,
            formal=True,
            pilot_arrival_utc={"hsinchu": "2024-01-02T01:00:00Z"},
        )
    assert not destination.exists()
    assert not list(root.glob(f".{destination.name}.partial-*"))


def test_hsinchu_explicit_pilot_does_not_register_ocm_missing_00z() -> None:
    """未登錄的 Jan 1 00Z 入口應直接拒絕，不讓 NWW 00Z 反向擴張 OCM 視窗。"""

    with pytest.raises(InputDerivationError, match="只登錄 2024-01-02T01:00:00Z"):
        input_derivation_module._parse_explicit_pilot_arrivals(
            {"hsinchu": "2024-01-01T00:00:00Z"}
        )


@pytest.mark.parametrize("site_id", ["hsinchu", "houwan", "lienchiang"])
def test_explicit_pilot_registry_accepts_each_bcd_single_site(site_id: str) -> None:
    """版本化 registry 允許 B/C/D 各自單站，且固定相同 24 h／25 nodes 視窗。"""

    parsed = input_derivation_module._parse_explicit_pilot_arrivals(
        {site_id: "2024-01-02T01:00:00Z"}
    )
    assert tuple(parsed) == (site_id,)
    explicit = parsed[site_id]
    assert explicit.selection_scope == f"{site_id}_only"
    assert input_derivation_module._explicit_pilot_window_times(explicit).size == 25


def test_explicit_pilot_registry_requires_a_exact_pair() -> None:
    """A 區 pilot 不能只替換一站或與 B/C/D 混合。"""

    with pytest.raises(InputDerivationError, match="exact pair"):
        input_derivation_module._parse_explicit_pilot_arrivals(
            {"gongliao": "2024-01-02T01:00:00Z"}
        )
    with pytest.raises(InputDerivationError, match="exact pair"):
        input_derivation_module._parse_explicit_pilot_arrivals(
            {
                "gongliao": "2024-01-02T01:00:00Z",
                "guishan": "2024-01-02T01:00:00Z",
                "hsinchu": "2024-01-02T01:00:00Z",
            }
        )


def test_explicit_pilot_summary_preserves_a_pair_ids() -> None:
    """A pair summary 必須列出兩站 arrival IDs，讓兩站 gap-safe override 各自綁定。"""

    parsed = input_derivation_module._parse_explicit_pilot_arrivals(
        {
            "gongliao": "2024-01-02T01:00:00Z",
            "guishan": "2024-01-02T01:00:00Z",
        }
    )
    arrivals = []
    for site_id, explicit in parsed.items():
        replacement_id = f"replacement-{site_id}"
        arrivals.append(
            ArrivalTime(
                arrival_time_id=replacement_id,
                study_site_id=site_id,
                time_utc_ns=explicit.time_utc_ns,
                year=2024,
                season="DJF",
                tide_class=input_derivation_module.PILOT_EXPLICIT_TIDE_CLASS,
                phase_or_event=input_derivation_module.PILOT_EXPLICIT_PHASE_OR_EVENT,
                metadata={
                    "pilot_replacement_policy_id": input_derivation_module.PILOT_EXPLICIT_WINDOW_POLICY_ID,
                    "pilot_selection_scope": explicit.selection_scope,
                    "pilot_replaced_arrival_time_id": f"original-{site_id}",
                    "pilot_replacement_arrival_time_id": replacement_id,
                    "explicit_pilot_window_start_utc": "2024-01-01T01:00:00Z",
                    "explicit_pilot_window_end_utc": explicit.time_utc,
                    "explicit_pilot_window_expected_step_count": 25,
                },
            )
        )
    summary = input_derivation_module._pilot_selection_summary(arrivals)
    assert summary is not None
    assert summary["site_ids"] == ["gongliao", "guishan"]
    assert summary["arrival_time_ids"] == ["replacement-gongliao", "replacement-guishan"]
    assert summary["selection_scope"] == "A_pair_only"


@pytest.mark.parametrize("missing_product", ["OCM surface", "NWW3 analysis"])
def test_explicit_pilot_rejects_middle_hour_missing_from_product_axis(missing_product: str) -> None:
    """三產品軸只要少一個視窗中間 exact-hour，就在 spatial sampling 前 fail closed。"""

    explicit = input_derivation_module._parse_explicit_pilot_arrivals(
        {"hsinchu": "2024-01-02T01:00:00Z"}
    )["hsinchu"]
    window = input_derivation_module._explicit_pilot_window_times(explicit)
    missing_index = 12
    full_axis = SimpleNamespace(canonical=SimpleNamespace(time_utc_ns=window))
    missing_axis = SimpleNamespace(
        canonical=SimpleNamespace(time_utc_ns=np.delete(window, missing_index))
    )
    surface_product = missing_axis if missing_product == "OCM surface" else full_axis
    nww_product = missing_axis if missing_product == "NWW3 analysis" else full_axis
    with pytest.raises(InputDerivationError, match=f"{missing_product} exact UTC 視窗缺少逐時節點"):
        input_derivation_module._validate_explicit_pilot_arrival_support(
            explicit=explicit,
            product=full_axis,
            surface_product=surface_product,
            nww_product=nww_product,
            nww_cache=object(),
            context=SimpleNamespace(),
            expected_axis=window,
        )


def test_hsinchu_receptor_core_limits_positions_without_shrinking_local_domain(
    synthetic_input_fixture: tuple[Path, Path, Path, Path, Path],
) -> None:
    """新竹五個水平位置受 12.5 km 核心限制，但 B 區 local 仍精確等於 flow。"""

    config_path, ocm_root, surface_root, nww_root, root = synthetic_input_fixture
    result = build_input_derivatives(
        config_path=config_path,
        destination=root / "derived-inputs-hsinchu-core",
        ocm_native_root=ocm_root,
        ocm_surface_root=surface_root,
        nww_analysis_root=nww_root,
        formal=False,
    )
    receptor_payload, _ = read_canonical_json(result.paths["receptor"])
    domain_payload, _ = read_canonical_json(result.paths["domain_geometry"])
    local_payload, _ = read_canonical_json(result.paths["local_geometry"])

    hsinchu_records = [
        record for record in receptor_payload["records"] if record["study_site_id"] == "hsinchu"
    ]
    horizontal_positions = {
        (float(record["lon"]), float(record["lat"])) for record in hsinchu_records
    }
    assert len(hsinchu_records) == 20
    assert len(horizontal_positions) == 5

    # 距離必須用與 runtime 相同的 B 區 AEQD 公尺投影計算；不能以 lon/lat 差換算公里。
    projection = DomainProjection(120.45, 24.75)
    anchor_x, anchor_y = projection.project(120.45, 24.75)
    distances_m = []
    for lon, lat in horizontal_positions:
        x_m, y_m = projection.project(lon, lat)
        distances_m.append(float(np.hypot(x_m - anchor_x, y_m - anchor_y)))
    assert max(distances_m) <= 12_500.0 + 1.0

    domain_by_region = {record["analysis_region_id"]: record for record in domain_payload["records"]}
    local_by_site = {record["study_site_id"]: record for record in local_payload["records"]}
    hsinchu_domain = domain_by_region["B"]
    hsinchu_local = local_by_site["hsinchu"]
    assert hsinchu_domain["flow_domain_id"] == "hsinchu_cache_v3"
    assert hsinchu_local["flow_domain_id"] == "hsinchu_cache_v3"
    assert hsinchu_local["local_equals_flow"] is True
    assert shape(hsinchu_local["geometry"]).equals(shape(hsinchu_domain["geometry"]))


def test_receptor_candidate_without_core_preserves_legacy_selection() -> None:
    """未明示 core 的站點應把原 local polygon 原樣交給既有 maximin selector。"""

    projection = DomainProjection(120.45, 24.75)
    # 以 AEQD 公尺方框反投影建立 synthetic WGS84 local polygon；實際候選仍會再投影回
    # 同一座標系，故測試可直接比較有／無 core 兩條 candidate 建立路徑。
    local_polygon_lonlat = projection.unproject_geometry(box(-6_000.0, -6_000.0, 6_000.0, 6_000.0))
    site_without_core = StudySiteConfig(
        study_site_id="no_core",
        study_site_name_zh="無核心測試站",
        analysis_region_id="B",
        flow_domain_id="hsinchu_cache_v3",
    )
    candidate_metric = input_derivation_module._receptor_candidate_polygon_metric(
        site=site_without_core,
        projection=projection,
        local_polygon_lonlat=local_polygon_lonlat,
    )
    legacy_candidate_metric = projection.project_geometry(local_polygon_lonlat)
    assert candidate_metric.equals_exact(legacy_candidate_metric, tolerance=1e-6)

    # 用五個固定公尺位置建立最小 NativeMesh，確認 candidate polygon 未改變時，
    # persistent-wet pool 與 deterministic maximin 的回傳受體也完全相同。
    centers = np.asarray([[-4_000.0, 0.0], [-2_000.0, 0.0], [0.0, 0.0], [2_000.0, 0.0], [4_000.0, 0.0]])
    node_xy = np.asarray(
        [
            point
            for center in centers
            for point in (
                center + np.asarray([-150.0, -150.0]),
                center + np.asarray([150.0, -150.0]),
                center + np.asarray([0.0, 150.0]),
            )
        ],
        dtype=np.float64,
    )
    face_nodes = np.asarray(
        [[index, index + 1, index + 2, -1] for index in range(0, node_xy.shape[0], 3)],
        dtype=np.int64,
    )
    node_lon, node_lat = projection.unproject(node_xy[:, 0], node_xy[:, 1])
    mesh = NativeMesh(
        node_lon=node_lon,
        node_lat=node_lat,
        node_xy=node_xy,
        source_depth_m=np.full(node_xy.shape[0], 20.0),
        source_node_bottom_index=np.zeros(node_xy.shape[0], dtype=np.int64),
        face_nodes_local=face_nodes,
        face_node_count=np.full(5, 3, dtype=np.int64),
        source_face_global_index=np.arange(5, dtype=np.int64) + 100,
    )
    wetdry = np.zeros((50, 5), dtype=np.float64)
    actual_pool = prepare_horizontal_receptor_candidates(
        study_site_id="no_core",
        mesh=mesh,
        candidate_polygon_metric=candidate_metric,
        anchor_xy=(0.0, 0.0),
        wetdry_at_arrivals=wetdry,
    )
    legacy_pool = prepare_horizontal_receptor_candidates(
        study_site_id="no_core",
        mesh=mesh,
        candidate_polygon_metric=legacy_candidate_metric,
        anchor_xy=(0.0, 0.0),
        wetdry_at_arrivals=wetdry,
    )
    assert select_horizontal_receptors_from_pool(actual_pool) == select_horizontal_receptors_from_pool(
        legacy_pool
    )


def test_houwan_candidate_regions_keep_registered_geometry_and_2plus3_quota() -> None:
    """C 區候選 helper 應保留兩個 GeoJSON 子區，並對錯誤配額 fail closed。"""

    config = input_derivation_module.load_config(EXAMPLE_CONFIG)
    houwan = next(site for site in config.study_sites if site.study_site_id == "houwan")
    specs = input_derivation_module._candidate_region_specs(houwan)
    assert [spec.region_id for spec in specs] == ["c_west_coast", "c_south_tip"]
    assert [spec.allocation_count for spec in specs] == [2, 3]
    assert all(spec.geometry_lonlat.is_valid and spec.geometry_lonlat.area > 0.0 for spec in specs)

    regions = deepcopy(houwan.receptor_candidate_regions)
    assert regions is not None
    regions[0]["allocation_count"] = 1
    invalid_houwan = houwan.model_copy(update={"receptor_candidate_regions": regions})
    with pytest.raises(InputDerivationError, match="配額總數"):
        input_derivation_module._candidate_region_specs(invalid_houwan)


def test_source_inventory_uses_structural_fingerprint_for_large_npy(
    synthetic_input_fixture: tuple[Path, Path, Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """required NPY 不得經過逐 byte SHA-256；metadata/time 軸仍須保留實際內容 hash。"""

    config_path, ocm_root, surface_root, nww_root, root = synthetic_input_fixture
    observed_npy_hashes: list[Path] = []
    original_hash = input_derivation_module._sha256_file

    def spy_hash(path: str | Path) -> str:
        """記錄 builder 交給內容 hash helper 的 NPY，確認只有時間軸進入小檔案路徑。"""

        candidate = Path(path)
        if candidate.suffix == ".npy":
            observed_npy_hashes.append(candidate)
        return original_hash(path)

    monkeypatch.setattr(input_derivation_module, "_sha256_file", spy_hash)
    artifact_directory = root / "derived-inputs-structural"
    build_input_derivatives(
        config_path=config_path,
        destination=artifact_directory,
        ocm_native_root=ocm_root,
        ocm_surface_root=surface_root,
        nww_analysis_root=nww_root,
        formal=False,
    )
    forcing, _ = read_canonical_json(artifact_directory / ARTIFACT_FILENAMES["forcing_inventory"])
    assert observed_npy_hashes
    assert all(path.name == "time_utc_ns.npy" for path in observed_npy_hashes)
    assert all(
        "sha256" not in record
        for product in forcing["products"]
        for record in product["files"]
        if record["file_kind"] in {"grid_npy", "month_npy"}
    )
    assert all(
        record["fingerprint_kind"] == "npy_header_structural"
        and set(record["npy_header"]) == {"shape", "dtype"}
        for product in forcing["products"]
        for record in product["files"]
        if record["file_kind"] in {"grid_npy", "month_npy"}
    )


def test_source_inventory_rejects_size_header_metadata_and_time_tamper(
    synthetic_input_fixture: tuple[Path, Path, Path, Path, Path],
) -> None:
    """validator 應分別拒絕 large NPY 的 size/header、metadata bytes 與 time payload 竄改。"""

    config_path, ocm_root, surface_root, nww_root, root = synthetic_input_fixture
    artifact_directory = root / "derived-inputs-source-tamper"
    build_input_derivatives(
        config_path=config_path,
        destination=artifact_directory,
        ocm_native_root=ocm_root,
        ocm_surface_root=surface_root,
        nww_analysis_root=nww_root,
        formal=False,
    )
    forcing, _ = read_canonical_json(artifact_directory / ARTIFACT_FILENAMES["forcing_inventory"])

    def source_path(product_name: str, file_kind: str, product_root: Path) -> Path:
        """依 tokenized inventory record 定位 fixture source，不依任意檔名猜測月份。"""

        for product in forcing["products"]:
            if product["product"] != product_name:
                continue
            for record in product["files"]:
                if record["file_kind"] == file_kind:
                    prefix = f"${product['root_token']}/"
                    assert record["path"].startswith(prefix)
                    return product_root / record["path"][len(prefix) :]
        raise AssertionError(f"找不到 {product_name}/{file_kind} source record")

    def validate_with_roots() -> dict[str, Any]:
        """以實際三產品 root 重算 inventory structural binding。"""

        return validate_input_derivatives(
            artifact_directory,
            config_path=config_path,
            ocm_native_root=ocm_root,
            ocm_surface_root=surface_root,
            nww_analysis_root=nww_root,
            formal=False,
        )

    hvel_path = source_path("ocm_native", "month_npy", ocm_root)
    original_hvel = hvel_path.read_bytes()
    # endian descriptor 同長度替換，保留檔案 size 並讓 np.load 仍可讀；validator 必須
    # 依 structural dtype 重算，而不能只依 size 判定未變更。
    tampered_hvel = original_hvel.replace(b"'descr': '<f4'", b"'descr': '>f4'", 1)
    assert tampered_hvel != original_hvel
    hvel_path.write_bytes(tampered_hvel)
    result = validate_with_roots()
    assert result["valid"] is False
    assert any("source_npy_header_mismatch" in error for error in result["errors"])
    hvel_path.write_bytes(original_hvel)

    metadata_path = source_path("ocm_native", "month_metadata", ocm_root)
    original_metadata = metadata_path.read_text(encoding="utf-8")
    metadata_path.write_text(
        original_metadata.replace('"status": "ready"', '"status": "eager"'),
        encoding="utf-8",
    )
    result = validate_with_roots()
    assert result["valid"] is False
    assert any("source_metadata_hash_mismatch" in error for error in result["errors"])
    metadata_path.write_text(original_metadata, encoding="utf-8")

    time_path = source_path("ocm_native", "time_axis", ocm_root)
    time_values = np.load(time_path, mmap_mode="r+")
    original_time = int(time_values[0])
    time_values[0] = original_time + 3_600_000_000_000
    time_values.flush()
    result = validate_with_roots()
    assert result["valid"] is False
    assert any("source_time_hash_mismatch" in error for error in result["errors"])
    time_values[0] = original_time
    time_values.flush()

    # 最後再確認檔案大小變化也會 fail closed；這是 structural fingerprint 對截斷／附加
    # bytes 的基本保障，與保留相同 size 的 header tamper 是兩條獨立的檢查路徑。
    hvel_path.write_bytes(original_hvel + b"x")
    result = validate_with_roots()
    assert result["valid"] is False
    assert any("source_size_mismatch" in error for error in result["errors"])


def test_release_config_binds_expanded_a_inventory_id_round_trip(
    synthetic_input_fixture: tuple[Path, Path, Path, Path, Path],
) -> None:
    """當 A artifact 使用明示 expanded candidate 時，release 與 runtime resolver 必須一致。"""

    config_path, ocm_root, surface_root, nww_root, root = synthetic_input_fixture
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    candidate_id = "northeast_taiwan_common_cache_v4_synthetic_expanded"
    base_id = "northeast_taiwan_common_cache_v3"
    for domain in payload["domains"]:
        if domain["analysis_region_id"] == "A":
            domain["expanded_domain_candidate_id"] = candidate_id
    a_template_domain = next(domain for domain in payload["domains"] if domain["analysis_region_id"] == "A")
    expanded_bbox = list(a_template_domain["expanded_bbox_lon_lat"])
    expanded_config = root / "expanded-config.yaml"
    expanded_config.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")

    # 三套 accepted product 必須以同一實際 ID 綁定；複製 synthetic domain 後同步更新
    # 小型 metadata 的 domain ID，模擬 SERVER 產生 expanded A cache 而不覆寫 base v3。
    hidden_base_directories: list[tuple[Path, Path]] = []
    for product_root in (ocm_root, surface_root, nww_root):
        source = product_root / base_id
        target = product_root / candidate_id
        shutil.copytree(source, target)
        metadata_paths = [
            target / "grid" / "metadata.json",
            *sorted((target / "months").glob("*/metadata.json")),
        ]
        for metadata_path in metadata_paths:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if "domain" in metadata:
                metadata["domain"]["domain_id"] = candidate_id
            if "flow_domain_id" in metadata:
                metadata["flow_domain_id"] = candidate_id
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
        if product_root == ocm_root:
            # candidate mesh 的實際南界也必須落在明示 expanded bbox；只替換 metadata
            # domain ID 會掩蓋 v3 mesh 仍停在 24.55°N 的錯配，不能作為 expanded fixture。
            latitude_path = target / "grid" / "source_lat.npy"
            latitude = np.load(latitude_path, mmap_mode="r+")
            latitude[latitude == np.min(latitude)] = expanded_bbox[2]
            latitude.flush()
            assert np.isclose(float(np.min(latitude)), expanded_bbox[2], rtol=0.0, atol=1e-12)
        # builder 的 deterministic resolver 會優先使用 base；暫時把原 base 移到同一
        # fixture 的隔離備份，讓這個測試明確模擬「實際 accepted source 只有 expanded
        # candidate」，並在 finally 還原，不刪除任何 fixture 資料。
        backup = root / f"{product_root.name}-base-backup"
        shutil.move(str(source), str(backup))
        hidden_base_directories.append((source, backup))

    try:
        artifact_directory = root / "derived-inputs-expanded-a"
        build_input_derivatives(
            config_path=expanded_config,
            destination=artifact_directory,
            ocm_native_root=ocm_root,
            ocm_surface_root=surface_root,
            nww_analysis_root=nww_root,
            formal=False,
        )
        release_path = root / "expanded-release-config.yaml"
        created = create_release_config(
            config_template_path=expanded_config,
            input_directory=artifact_directory,
            output_path=release_path,
            formal=True,
        )
        assert created["config_status"] == "generated"
        release_payload = yaml.safe_load(release_path.read_text(encoding="utf-8"))
        assert isinstance(release_payload, dict)
        a_domain = next(item for item in release_payload["domains"] if item["analysis_region_id"] == "A")
        assert a_domain["formal_release_flow_domain_id"] == candidate_id
        assert a_domain["bbox_lon_lat"] == expanded_bbox
        assert {
            site["formal_release_flow_domain_id"]
            for site in release_payload["study_sites"]
            if site["analysis_region_id"] == "A"
        } == {candidate_id}
        release_config = input_derivation_module.load_config(release_path)
        assert resolve_flow_domain_id(release_config, "A", formal=True) == candidate_id
        forcing, _ = read_canonical_json(artifact_directory / ARTIFACT_FILENAMES["forcing_inventory"])
        a_products = [item for item in forcing["products"] if item["analysis_region_id"] == "A"]
        assert {item["product"] for item in a_products} == {
            "ocm_native",
            "ocm_surface",
            "nww3_analysis",
        }
        assert all(item["grid_metadata"]["bbox_lon_lat"] == expanded_bbox for item in a_products)
        domain_component, _ = read_canonical_json(artifact_directory / ARTIFACT_FILENAMES["domain_geometry"])
        domain_record = next(row for row in domain_component["records"] if row["analysis_region_id"] == "A")
        assert np.allclose(
            shape(domain_record["geometry"]).bounds,
            (expanded_bbox[0], expanded_bbox[2], expanded_bbox[1], expanded_bbox[3]),
            rtol=0.0,
            atol=1e-12,
        )
        open_component, _ = read_canonical_json(artifact_directory / ARTIFACT_FILENAMES["open_boundary"])
        open_record = next(
            row
            for row in open_component["records"]
            if row["owner_kind"] == "flow_domain" and row["analysis_region_id"] == "A"
        )
        assert np.allclose(
            shape(open_record["geometry"]).bounds,
            (expanded_bbox[0], expanded_bbox[2], expanded_bbox[1], expanded_bbox[3]),
            rtol=0.0,
            atol=1e-12,
        )
        for kind in ("domain_geometry", "local_geometry", "initial_condition"):
            component, _ = read_canonical_json(artifact_directory / ARTIFACT_FILENAMES[kind])
            rows = component["records"]
            assert all(
                row["flow_domain_id"] == candidate_id for row in rows if row.get("analysis_region_id") == "A"
            )
        assert validate_release_config(release_path, input_directory=artifact_directory)["valid"] is True
    finally:
        for source, backup in hidden_base_directories:
            shutil.move(str(backup), str(source))


def test_expanded_candidate_without_registered_bbox_fails_closed() -> None:
    """expanded candidate 若沒有明示 expanded bbox，release binding 必須拒絕猜測。"""

    payload = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    a_domain = next(domain for domain in payload["domains"] if domain["analysis_region_id"] == "A")
    # 這是 legacy expanded resolver 的回歸測試；新 v3 policy 不得含 candidate 欄位，
    # 因此 fixture 先明示舊 policy 與候選，再移除 bbox 觸發原本的 fail-closed gate。
    payload["design_version"] = "design_baseline_v2_non_rising_oca_proxy"
    a_domain["formal_domain_policy"] = "expanded_domain_v1"
    candidate_id = "northeast_taiwan_common_cache_v4_lbt_south_expanded"
    a_domain["expanded_domain_candidate_id"] = candidate_id
    a_domain["formal_release_flow_domain_id"] = candidate_id
    a_domain["formal_release_domain_status"] = "approved"
    a_domain["expanded_bbox_lon_lat"] = [121.306315, 122.793685, 24.480000, 25.499156]
    for site in payload["study_sites"]:
        if site["analysis_region_id"] == "A":
            site["formal_release_flow_domain_id"] = candidate_id
    a_domain.pop("expanded_bbox_lon_lat", None)
    inventory_flow_ids = {
        "A": candidate_id,
        "B": "hsinchu_cache_v3",
        "C": "houwan_nmmba_cache_v3",
        "D": "lienchiang_common_cache_v3",
    }
    with pytest.raises(InputDerivationError, match="expanded_bbox_lon_lat"):
        input_derivation_module._bind_inventory_flow_domains(
            payload,
            inventory_flow_ids=inventory_flow_ids,
        )


def test_v3_policy_binds_exact_source_and_preserves_pending_status() -> None:
    """新 policy 的三套 inventory source 必須 exact v3，binding 不得 mint approved status。"""

    payload = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    result = input_derivation_module._bind_inventory_flow_domains(
        payload,
        inventory_flow_ids={
            "A": "northeast_taiwan_common_cache_v3",
            "B": "hsinchu_cache_v3",
            "C": "houwan_nmmba_cache_v3",
            "D": "lienchiang_common_cache_v3",
        },
    )
    a_domain = next(domain for domain in result["domains"] if domain["analysis_region_id"] == "A")
    assert a_domain["formal_release_flow_domain_id"] == "northeast_taiwan_common_cache_v3"
    assert a_domain["bbox_lon_lat"] == [121.306315, 122.793685, 24.600844, 25.499156]
    assert a_domain["formal_release_domain_status"] == "pending_common_support"
    assert a_domain["formal_release_domain_status"] != "approved_source_bound_by_inventory"
    assert {
        site["formal_release_flow_domain_id"]
        for site in result["study_sites"]
        if site["analysis_region_id"] == "A"
    } == {"northeast_taiwan_common_cache_v3"}


def test_v3_policy_resolver_does_not_fallback_to_v4(tmp_path: Path) -> None:
    """A 區 v3 缺失時即使同根目錄有 v4，也只能回報 v3 待載入，不得 fallback。"""

    config = input_derivation_module.load_config(EXAMPLE_CONFIG)
    root = tmp_path / "ocm"
    (root / "northeast_taiwan_common_cache_v4_lbt_south_expanded").mkdir(parents=True)
    resolved = input_derivation_module._resolve_flow_product_id(config, "A", root, formal=False)
    assert resolved == "northeast_taiwan_common_cache_v3"
    with pytest.raises(InputDerivationError, match="exact northeast_taiwan_common_cache_v3"):
        input_derivation_module._authoritative_flow_domain_bbox_lon_lat(
            config.domains[0], "northeast_taiwan_common_cache_v4_lbt_south_expanded"
        )


def _geometry_policy_test_products(
    tmp_path: Path, config: ProjectConfig
) -> dict[str, input_derivation_module._ProductData]:
    """建立只含 A 區靜態 mesh 的 geometry 測試產品容器。

    ``_geometry_payloads`` 對 B-D 只需要產品的識別資料；A 區則需要一個涵蓋固定 v3
    bbox 的原生 face mesh，才能實際測試完整 local circle 的 pre-clip 檢查。這裡不建立
    月份或 forcing time axis，因此不會把 geometry 單元測試誤當成 source coverage 驗收。
    """

    products: dict[str, input_derivation_module._ProductData] = {}
    for domain in config.domains:
        product_root = tmp_path / domain.analysis_region_id
        grid_dir = product_root / domain.flow_domain_id / "grid"
        if domain.analysis_region_id == "A":
            node_lon, node_lat, faces = _rectangular_grid(
                lon_min=domain.bbox_lon_lat[0],
                lon_max=domain.bbox_lon_lat[1],
                lat_min=domain.bbox_lon_lat[2],
                lat_max=domain.bbox_lon_lat[3],
                nx=8,
                ny=10,
            )
            _write_ocm_domain(
                product_root,
                flow_domain_id=domain.flow_domain_id,
                months=[],
                node_lon=node_lon,
                node_lat=node_lat,
                faces=faces,
            )
        products[domain.analysis_region_id] = input_derivation_module._ProductData(
            product="ocm_native",
            flow_domain_id=domain.flow_domain_id,
            root=product_root,
            root_token=f"TEST_{domain.analysis_region_id}",
            grid_dir=grid_dir,
            grid_metadata={"cache_schema_version": "3.0.0"},
            months=(),
            canonical=SimpleNamespace(),
            required_arrays=(),
            source_file_records=(),
        )
    return products


def test_v3_geometry_identity_and_preclip_fail_closed(tmp_path: Path) -> None:
    """v3 geometry 必須帶 20 km identity，且越界 local 不得先 clip 後掩蓋。"""

    config = input_derivation_module.load_config(EXAMPLE_CONFIG)
    products = _geometry_policy_test_products(tmp_path, config)
    source_hashes = {"geometry-test": "a" * 64}
    domain_payload, local_payload, open_payload, _, _ = input_derivation_module._geometry_payloads(
        config=config,
        products_by_region=products,
        source_hashes=source_hashes,
        strict=False,
    )

    a_domain = next(row for row in domain_payload["records"] if row["analysis_region_id"] == "A")
    a_locals = [row for row in local_payload["records"] if row["analysis_region_id"] == "A"]
    a_open = [
        row
        for row in open_payload["records"]
        if row["analysis_region_id"] == "A" and row["owner_kind"] == "local_domain"
    ]
    assert a_domain["source_geometry_id"].endswith("_base_bbox_v3_local20km_20260909_v1")
    assert {row["study_site_id"] for row in a_locals} == {"gongliao", "guishan"}
    assert all("_local_domain_v3_local20km_20260909_v1" in row["source_geometry_id"] for row in a_locals)
    assert all(
        "_local_exterior_open_boundary_v3_local20km_20260909_v1" in row["source_geometry_id"]
        for row in a_open
    )
    assert domain_payload["provenance"]["method_id"] == "server_v3_domain_bbox_geometry_v1"
    assert local_payload["provenance"]["method_id"].endswith("v3_local20km_20260909_v1")
    assert open_payload["provenance"]["method_id"].endswith("v3_local20km_20260909_v1")

    drifted_sites = [
        site.model_copy(update={"local_domain_baseline_radius_m": 50_000.0})
        if site.study_site_id == "gongliao"
        else site
        for site in config.study_sites
    ]
    drifted_config = config.model_copy(deep=True, update={"study_sites": drifted_sites})
    with pytest.raises(InputDerivationError, match="不得以 clip 隱藏越界"):
        input_derivation_module._geometry_payloads(
            config=drifted_config,
            products_by_region=products,
            source_hashes=source_hashes,
            strict=False,
        )


def test_input_validator_rejects_artifact_closure_tamper(
    synthetic_input_fixture: tuple[Path, Path, Path, Path, Path],
) -> None:
    """artifact index 或 component sidecar 被改寫時，validator 必須 fail closed。"""

    config_path, ocm_root, surface_root, nww_root, root = synthetic_input_fixture
    artifact_directory = root / "derived-inputs"
    build_input_derivatives(
        config_path=config_path,
        destination=artifact_directory,
        ocm_native_root=ocm_root,
        ocm_surface_root=surface_root,
        nww_analysis_root=nww_root,
        formal=False,
    )
    index = artifact_directory / "artifact_index.json"
    original = index.read_text(encoding="utf-8")
    index.write_text(original.replace('"status": "immutable"', '"status": "tampered"'), encoding="utf-8")
    result = validate_input_derivatives(artifact_directory, config_path=config_path, formal=False)
    assert result["valid"] is False
    assert any("artifact_index" in error or "closure" in error for error in result["errors"])


def test_artifact_prepublish_cross_reference_failure_removes_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """partial cross-reference validator 失敗時不得留下 final 或 hidden partial。"""

    config_path = tmp_path / "config.yaml"
    config_path.write_text("config_status: generated\n", encoding="utf-8")
    native_root = tmp_path / "ocm_native"
    surface_root = tmp_path / "ocm_surface"
    nww_root = tmp_path / "nww_analysis"
    for root in (native_root, surface_root, nww_root):
        root.mkdir()
    calls: dict[str, Any] = {}

    def reject_partial(directory: Path, **kwargs: Any) -> dict[str, Any]:
        """模擬 component cross-reference 錯配，並記錄 writer 傳入的正式參數。"""

        calls["directory"] = directory
        calls.update(kwargs)
        return {
            "valid": False,
            "errors": ["component_cross_reference_invalid:fixture"],
            "warnings": [],
            "summary": {},
        }

    monkeypatch.setattr(input_derivation_module, "validate_input_derivatives", reject_partial)
    destination = tmp_path / "derived-inputs"
    with pytest.raises(InputDerivationError, match="partial publish"):
        input_derivation_module._write_artifact_directory(
            destination,
            {"material": {"fixture": True}},
            config_path=config_path,
            formal=False,
            ocm_native_root=native_root,
            ocm_surface_root=surface_root,
            nww_analysis_root=nww_root,
        )

    assert not destination.exists()
    assert not list(tmp_path.glob(f".{destination.name}.partial-*"))
    assert calls["directory"].name.startswith(f".{destination.name}.partial-")
    assert calls["formal"] is False
    assert calls["config_path"] == config_path
    assert calls["ocm_native_root"] == native_root
    assert calls["ocm_surface_root"] == surface_root
    assert calls["nww_analysis_root"] == nww_root
