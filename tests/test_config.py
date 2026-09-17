"""設定 schema、五站計數與正式發布閘門測試。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from lagrangian_backtracking.config import (
    CURRENT_DESIGN_VERSION,
    FORMAL_DOMAIN_POLICY_V3_LOCAL20KM_20260909_V1,
    GUISHAN_SOFT_PRIORITY_POLICY_ID,
    GUISHAN_SOFT_PRIORITY_POLYGON_LON_LAT,
    HSINCHU_FIXED_HORIZONTAL_RECEPTOR_COORDINATES_LON_LAT,
    HSINCHU_FIXED_HORIZONTAL_RECEPTOR_POLICY_ID,
    HSINCHU_FIXED_HORIZONTAL_RECEPTOR_SOURCE_SHA256,
    LEGACY_DESIGN_VERSION_V2,
    ProjectConfig,
    load_config,
    resolve_flow_domain_id,
)
from lagrangian_backtracking.scenarios import BASELINE_BEHAVIORS

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = ROOT / "configs" / "lagrangian_backtracking.example.yaml"


def _payload() -> dict:
    """讀取範例 YAML mapping，供單一契約破壞測試使用。"""

    value = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _legacy_example_payload() -> dict:
    """由正式範例明確建立未啟用隨機沉底與共同支援窗的舊相容 fixture。"""

    payload = deepcopy(_payload())
    # 現行 C 站點已改為南灣；legacy fixture 必須明確還原舊後灣 site，才能測試
    # 舊 artifact 的相容載入，而不是把 current site 名稱誤當成歷史契約。
    nanwan = next(site for site in payload["study_sites"] if site["study_site_id"] == "nanwan")
    nanwan["study_site_id"] = "houwan"
    nanwan["study_site_name_zh"] = "後灣海生館"
    nanwan["anchor_lonlat"] = [120.893355, 22.0]
    payload["domains"][2]["analysis_region_name_zh"] = "後灣海域"
    # 這個 fixture 必須重現歷史 v3 YAML 的位元組語意，才能讓既有 canonical hash
    # 測試繼續保護舊 artifact。紅框候選只在這個 legacy fixture 出現；現行南灣
    # 正式設定刻意不再載入這些 2+3 子區。
    nanwan.update(
        {
            "receptor_candidate_selection": {
                "policy_id": "houwan_red_frame_two_subregions_anchor_first_maximin_2plus3_v1",
                "require_each_region": True,
                "total_horizontal_count": 5,
                "distance_coordinate_system": "local_azimuthal_equidistant_m",
                "boundary_policy": "approved_C_flow_local_intersection_plus_persistent_wet_margin",
            },
            "receptor_candidate_regions_provenance": {
                "source_artifact": "horizontal_overview.png",
                "source_sha256": "0c28a19f9707f5d4717799b4936bb99195a868b76aec1322c21c725192132beb",
                "image_width_px": 1236,
                "image_height_px": 832,
                "plot_bbox_pixels_xy": [65, 46, 1180, 789],
                "plot_bbox_lon_lat": [120.16671, 121.62, 21.550844, 22.449156],
                "digitization_method": "manual_red_frame_vertex_trace_v1",
            },
            "receptor_candidate_regions": [
                {
                    "region_id": "c_west_coast",
                    "name_zh": "屏東西南沿岸紅框",
                    "allocation_count": 2,
                    "coordinate_reference": "EPSG:4326",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [120.67242885, 22.13359822],
                                [120.62941668, 22.12876208],
                                [120.60465210, 22.09128204],
                                [120.58770791, 22.03203939],
                                [120.58119091, 21.97158771],
                                [120.58249431, 21.93652573],
                                [120.59683170, 21.91718120],
                                [120.64636087, 21.90025472],
                                [120.66591186, 21.93652573],
                                [120.68024925, 21.98246901],
                                [120.70240703, 22.02720326],
                                [120.70501383, 22.05742910],
                                [120.69328324, 22.09490914],
                                [120.67242885, 22.13359822],
                            ]
                        ],
                    },
                },
                {
                    "region_id": "c_south_tip",
                    "name_zh": "恆春半島南端近岸紅框",
                    "allocation_count": 3,
                    "coordinate_reference": "EPSG:4326",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [120.71153083, 21.94619800],
                                [120.77018378, 21.95345220],
                                [120.82883674, 21.95345220],
                                [120.86663532, 21.93773477],
                                [120.89531010, 21.91597216],
                                [120.91616448, 21.88816439],
                                [120.92007468, 21.86035661],
                                [120.87706251, 21.84584821],
                                [120.83796054, 21.83617594],
                                [120.77409398, 21.83496691],
                                [120.72065462, 21.84463918],
                                [120.68546284, 21.86640178],
                                [120.67633905, 21.89300052],
                                [120.67894585, 21.92080830],
                                [120.71153083, 21.94619800],
                            ]
                        ],
                    },
                },
            ],
        }
    )
    # 舊 A anchor 也是 frozen v3 hash 的一部分；現行正式核對圖才改成高精度 anchor。
    payload["study_sites"][0]["anchor_lonlat"] = [121.92807, 25.11245]
    for site in payload["study_sites"]:
        for field in (
            "horizontal_receptor_coordinates",
            "horizontal_receptor_source_manifest_sha256",
            "horizontal_receptor_selection_policy",
            "horizontal_receptor_coordinate_tolerance_m",
            "receptor_priority_polygon",
            "receptor_priority_selection",
        ):
            site.pop(field, None)
    # 範例 YAML 現在屬於 gap-censored 新版；legacy fixture 必須明確移除新增的
    # time-axis policy，才能驗證「未宣告新欄位的舊 hash 不漂移」，而不是把新版
    # 語意誤當成舊設定的一部分。
    time_axis_contract = payload["inputs"].get("time_axis_contract")
    if isinstance(time_axis_contract, dict):
        for field in ("gap_policy", "stop_at_first_gap", "denominator_policy"):
            time_axis_contract.pop(field, None)
    payload["scenarios"].pop("bed_residence_time", None)
    payload["inputs"].pop("backtrack_support_days", None)
    payload["design_version"] = "design_baseline_v3_non_rising_a_v3_local20_20260909"
    payload["inputs"]["available_data_contract"]["population_id"] = "available_2024_2025_v1"
    payload["arrival_time_selection"].pop("policy", None)
    payload["arrival_time_selection"].pop("observation_years", None)
    payload["arrival_time_selection"].pop("replicates", None)
    payload["arrival_time_selection"]["core_design"] = (
        "two_years_by_four_seasons_by_spring_neap_by_three_tidal_phase_proxies"
    )
    return payload


def _bed_residence_block() -> dict:
    """建立完整的新沉底時間區塊，讓 schema 測試只修改一個契約欄位。"""

    return {
        "backtrack_mode": "fixed_calendar_window",
        "supported_backtrack_modes": [
            "fixed_calendar_window",
            "full_horizon_from_deposition",
        ],
        "maximum_age_days": 90,
        "sample_count_per_site": 50,
        "sampling_policy": "discrete_hourly_stratified_uniform_v1",
        "sampling_seed": 20260916,
        "shared_age_offsets_across_sites": True,
        "availability_conditioning_policy": (
            "reject_unavailable_deposition_hour_within_stratum_v1"
        ),
        "pre_window_policy": "record_pre_window_deposition_without_transport",
        "runtime_horizon_support_days": 90,
    }


def _legacy_hash_payload() -> dict:
    """由現行範例還原未含 policy 的舊設定快照，避免測試依賴 git 指令。"""

    payload = _legacy_example_payload()
    payload["config_status"] = "design_baseline_example"
    payload["design_version"] = "design_baseline_v2_non_rising_oca_proxy"
    a_domain = payload["domains"][0]
    a_domain.pop("formal_domain_policy", None)
    a_domain.pop("runtime_spatial_support_policy", None)
    a_domain["formal_release_flow_domain_id"] = None
    a_domain["formal_release_domain_status"] = "expanded_domain_generation_required"
    # 這兩個 extra 是舊 v3 YAML 曾宣告、但沒有經驗證的共同 forcing margin；legacy hash
    # snapshot 必須保留歷史位元組語意，與新 policy 明確移除此宣告分開測試。
    a_domain["minimum_common_forcing_margin_grid_cells"] = 2
    a_domain["margin_required_for_forcings"] = ["ocm_native", "ocm_surface", "nww3_analysis"]
    a_domain["expanded_domain_candidate_id"] = "northeast_taiwan_common_cache_v4_lbt_south_expanded"
    a_domain["expanded_bbox_lon_lat"] = [121.306315, 122.793685, 24.480000, 25.499156]
    a_domain["expanded_south_boundary_at_or_south_of_deg"] = 24.48
    a_domain["radius_25000_formal_requires_expanded_domain"] = True
    a_domain["radius_35000_formal_requires_expanded_domain"] = True
    for site in payload["study_sites"]:
        if site["analysis_region_id"] == "A":
            site["formal_release_flow_domain_id"] = None
            site["local_domain_baseline_radius_m"] = 25_000
            site["local_domain_sensitivity_radii_m"] = [20_000, 35_000]
            site["radius_35000_requires_expanded_flow_domain"] = True
        if site["study_site_id"] in {"houwan", "lienchiang"}:
            # 這兩個欄位是本期 v3/試跑回寫；舊 v2 snapshot 本來沒有它們，移除後
            # 測試才是在比對原始 v2 canonical hash，而不是把新站點設定誤算進 legacy。
            site.pop("anchor_lonlat", None)
            site.pop("receptor_core_radius_m", None)
        for field in (
            "horizontal_receptor_coordinates",
            "horizontal_receptor_source_manifest_sha256",
            "horizontal_receptor_selection_policy",
            "horizontal_receptor_coordinate_tolerance_m",
            "receptor_priority_polygon",
            "receptor_priority_selection",
        ):
            site.pop(field, None)
        site.pop("receptor_candidate_regions", None)
        site.pop("receptor_candidate_selection", None)
        site.pop("receptor_candidate_regions_provenance", None)
    payload["boundaries"].pop("sensitivity_case_region_exclusions", None)
    return payload


def test_example_config_has_fixed_scientific_counts() -> None:
    """正式母體範例保留固定研究計數並啟用 180/90/50 沉底契約。"""

    config = load_config(EXAMPLE_CONFIG)
    assert len(config.domains) == 4
    assert len(config.study_sites) == 5
    assert config.scenarios.expected_receptor_count == 100
    assert config.scenarios.scenario_count == 50_000
    assert config.scenarios.seed_policy == "sha256_v1_pcg64dxsm"
    assert config.execution.checkpoint_interval_sweeps is None
    assert config.execution.active_chunk_size is None
    assert config.execution.max_resident_forcing_months == 2
    bed = config.normalized_payload()["scenarios"]["bed_residence_time"]
    assert bed["backtrack_mode"] == "fixed_calendar_window"
    assert bed["supported_backtrack_modes"] == [
        "fixed_calendar_window",
        "full_horizon_from_deposition",
    ]
    assert bed["maximum_age_days"] == 90
    assert bed["sample_count_per_site"] == 50
    assert bed["sampling_seed"] == 20260916
    assert bed["runtime_horizon_support_days"] is None
    assert config.inputs.backtrack_support_days == 180
    assert config.arrival_time_selection is not None
    assert config.arrival_time_selection.policy == "observation_year_stratified_48_plus_2_v1"
    assert config.arrival_time_selection.observation_years == [2025]
    assert config.arrival_time_selection.replicates == 2


def test_new_observation_years_must_be_nonempty_forcing_subset() -> None:
    """新版 observation 年份不可脫離 2024–2025 forcing 聯集或變成空集合。"""

    for years in ([], [2023], [2024, 2025, 2025]):
        payload = _payload()
        payload["arrival_time_selection"]["observation_years"] = years
        with pytest.raises(ValueError, match="observation_years"):
            ProjectConfig.model_validate(payload)


def test_receptor_candidate_domain_policy_is_versioned_and_core_aware() -> None:
    """設定檔應明示 core∩local 與未明示 core fallback 的候選區政策。"""

    payload = _payload()
    assert payload["scenarios"]["other_site_receptor_candidate_domain"] == (
        "site_explicit_core_intersect_local_else_local_or_flow_v1"
    )


def test_example_material_table_matches_runtime_baseline() -> None:
    """YAML 與程式內建表不可各自維護不同的材質、形狀、條件或速度。"""

    payload = _payload()
    assert payload["design_version"] == CURRENT_DESIGN_VERSION
    assert payload["physics"]["settling"]["material_classes"] == [asdict(item) for item in BASELINE_BEHAVIORS]


def test_example_registers_current_c_nanwan_and_d_receptor_anchors() -> None:
    """現行範例固定南灣／D anchor，且 C 不得殘留後灣 2+3 候選。"""

    config = load_config(EXAMPLE_CONFIG)
    sites = {site.study_site_id: site for site in config.study_sites}
    assert config.design_version == CURRENT_DESIGN_VERSION
    assert set(sites) == {"gongliao", "guishan", "hsinchu", "nanwan", "lienchiang"}
    assert sites["nanwan"].anchor_lonlat == (120.763161, 21.946577)
    assert sites["nanwan"].receptor_core_radius_m == 12_500
    assert sites["nanwan"].receptor_candidate_regions is None
    assert sites["nanwan"].receptor_candidate_selection is None
    assert sites["nanwan"].receptor_candidate_regions_provenance is None
    assert sites["lienchiang"].anchor_lonlat == (119.95, 26.2)
    assert sites["lienchiang"].receptor_core_radius_m == 12_500
    assert (
        tuple(sites["hsinchu"].horizontal_receptor_coordinates)
        == HSINCHU_FIXED_HORIZONTAL_RECEPTOR_COORDINATES_LON_LAT
    )
    assert sites["hsinchu"].horizontal_receptor_source_manifest_sha256 == (
        HSINCHU_FIXED_HORIZONTAL_RECEPTOR_SOURCE_SHA256
    )
    assert sites["hsinchu"].horizontal_receptor_selection_policy == (
        HSINCHU_FIXED_HORIZONTAL_RECEPTOR_POLICY_ID
    )
    assert tuple(sites["guishan"].receptor_priority_polygon or ()) == (
        GUISHAN_SOFT_PRIORITY_POLYGON_LON_LAT
    )
    assert sites["guishan"].receptor_priority_selection == {
        "policy_id": GUISHAN_SOFT_PRIORITY_POLICY_ID,
        "fallback": "same_core_pool",
        "total_horizontal_count": 5,
        "coordinate_reference": "EPSG:4326",
    }


def test_current_design_rejects_legacy_houwan_site() -> None:
    """current formal design 不得以 houwan 取代南灣，即使 forcing ID 相同。"""

    payload = _payload()
    nanwan = next(site for site in payload["study_sites"] if site["study_site_id"] == "nanwan")
    nanwan["study_site_id"] = "houwan"
    nanwan["study_site_name_zh"] = "後灣海生館"
    with pytest.raises(ValueError, match="study_site_id"):
        ProjectConfig.model_validate(payload)


def test_current_design_rejects_shifted_flow_domain_bbox() -> None:
    """current 四區 bbox 必須沿用 OCM-SVD-Analysis，不能採用平移南灣候選框。"""

    payload = _payload()
    payload["domains"][2]["bbox_lon_lat"] = [
        120.036516,
        121.489806,
        21.497421,
        22.395733,
    ]
    with pytest.raises(ValueError, match="bbox 必須 exact"):
        ProjectConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("horizontal_receptor_coordinates", [[120.0, 24.0]] * 4),
        ("horizontal_receptor_source_manifest_sha256", "0" * 64),
        ("horizontal_receptor_selection_policy", "other_policy"),
    ],
)
def test_current_hsinchu_fixed_receptor_contract_is_immutable(field: str, value: object) -> None:
    """B 區固定受體清單、來源 hash 與 policy 任一被改寫都必須 fail closed。"""

    payload = _payload()
    hsinchu = next(site for site in payload["study_sites"] if site["study_site_id"] == "hsinchu")
    hsinchu[field] = value
    with pytest.raises(ValueError, match="B 區|hsinchu|固定水平受體"):
        ProjectConfig.model_validate(payload)


def test_design_version_and_a_policy_binding_is_bidirectional() -> None:
    """v3 design 缺 A policy 或 legacy design 借用 v3 policy 都必須 fail closed。"""

    payload = _payload()
    assert payload["domains"][0]["formal_domain_policy"] == FORMAL_DOMAIN_POLICY_V3_LOCAL20KM_20260909_V1
    missing_policy = deepcopy(payload)
    missing_policy["domains"][0].pop("formal_domain_policy")
    with pytest.raises(ValueError, match="必須明示 A 區 formal_domain_policy"):
        ProjectConfig.model_validate(missing_policy)

    legacy_with_v3 = deepcopy(payload)
    legacy_with_v3["design_version"] = LEGACY_DESIGN_VERSION_V2
    with pytest.raises(ValueError, match="只能搭配"):
        ProjectConfig.model_validate(legacy_with_v3)


@pytest.mark.parametrize("invalid_velocity", [0.0, 0.001])
def test_config_rejects_zero_or_rising_material_velocity(invalid_velocity: float) -> None:
    """正式設定不得藉修改 YAML 重新加入中性懸浮或上浮物性速度。"""

    payload = _payload()
    payload["physics"]["settling"]["material_classes"][0]["settling_velocity_mps"] = invalid_velocity
    with pytest.raises(ValueError, match="嚴格小於 0"):
        ProjectConfig.model_validate(payload)


def test_config_hash_is_independent_of_mapping_order() -> None:
    """canonical hash 不應因 YAML key 順序改變，避免無科學差異卻產生新 run。"""

    first = ProjectConfig.model_validate(_payload())
    reordered = dict(reversed(list(_payload().items())))
    second = ProjectConfig.model_validate(reordered)
    assert first.config_hash() == second.config_hash()


def test_bed_residence_config_is_typed_complete_and_enters_hash() -> None:
    """明示沉底模型時完整保留 mode、抽樣設定與 runtime 支援窗於 canonical payload。"""

    payload = _payload()
    payload["scenarios"]["bed_residence_time"] = _bed_residence_block()
    config = ProjectConfig.model_validate(payload)
    normalized = config.normalized_payload()["scenarios"]["bed_residence_time"]
    assert normalized["backtrack_mode"] == "fixed_calendar_window"
    assert set(normalized["supported_backtrack_modes"]) == {
        "fixed_calendar_window",
        "full_horizon_from_deposition",
    }
    assert normalized["maximum_age_days"] == 90
    assert normalized["sample_count_per_site"] == config.scenarios.expected_arrival_time_count_per_site
    assert normalized["runtime_horizon_support_days"] == 90
    assert config.inputs.backtrack_support_days == 180
    assert config.config_hash() != ProjectConfig.model_validate(_legacy_example_payload()).config_hash()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("backtrack_mode", "1"),
        ("supported_backtrack_modes", ["fixed_calendar_window"]),
        ("maximum_age_days", 89),
        ("sample_count_per_site", 49),
        ("sampling_policy", "unregistered_policy"),
        ("sampling_seed", True),
        ("sampling_seed", -1),
        ("shared_age_offsets_across_sites", False),
        ("pre_window_policy", "drop_member"),
        ("runtime_horizon_support_days", 0),
    ],
)
def test_bed_residence_config_rejects_unregistered_contract_values(
    field: str, value: object
) -> None:
    """新區塊逐欄 fail closed，避免錯誤值落入 extra 欄位或靜默轉型。"""

    payload = _payload()
    block = _bed_residence_block()
    block[field] = value
    payload["scenarios"]["bed_residence_time"] = block
    with pytest.raises(ValueError):
        ProjectConfig.model_validate(payload)


def test_omitted_backtrack_support_keeps_legacy_hash() -> None:
    """明確移除新欄位的舊 fixture 維持歷史 hash 與 normalized payload 語意。"""

    config = ProjectConfig.model_validate(_legacy_example_payload())
    assert config.config_hash() == "c38ba2c7b21ab7249b517120f4d5964b87747eb6680c410d03090cc66301b893"
    assert "backtrack_support_days" not in config.normalized_payload()["inputs"]
    assert "bed_residence_time" not in config.normalized_payload()["scenarios"]


def test_omitted_ocm_interpolation_backend_keeps_numpy_and_current_hash() -> None:
    """舊 YAML 省略 OCM backend 時走 NumPy，且不得因預設欄位改變 canonical hash。"""

    config = ProjectConfig.model_validate(_legacy_example_payload())
    assert config.execution.ocm_interpolation_backend == "numpy_v1"
    assert "ocm_interpolation_backend" not in config.normalized_payload()["execution"]
    assert config.config_hash() == "c38ba2c7b21ab7249b517120f4d5964b87747eb6680c410d03090cc66301b893"


def test_omitted_physics_kernel_backend_keeps_numpy_and_current_hash() -> None:
    """舊 YAML 未帶 CPU 核心選項時仍用 NumPy，且 config hash 維持凍結值。"""

    config = ProjectConfig.model_validate(_legacy_example_payload())
    assert config.execution.physics_kernel_backend == "numpy_v1"
    assert "physics_kernel_backend" not in config.normalized_payload()["execution"]
    assert config.config_hash() == "c38ba2c7b21ab7249b517120f4d5964b87747eb6680c410d03090cc66301b893"


@pytest.mark.parametrize("backend", ["numpy_v1", "numba_cpu_v1"])
def test_explicit_physics_kernel_backend_is_versioned_and_hashed(backend: str) -> None:
    """明示 CPU 數值後端時必須在 canonical 設定與 hash 中留下版本 token。"""

    payload = _payload()
    payload["execution"]["physics_kernel_backend"] = backend
    config = ProjectConfig.model_validate(payload)
    assert config.execution.physics_kernel_backend == backend
    assert config.normalized_payload()["execution"]["physics_kernel_backend"] == backend
    assert config.config_hash() != ProjectConfig.model_validate(_payload()).config_hash()


@pytest.mark.parametrize("backend", [None, "numpy", "numba", "numba_cpu_v2", True, 1])
def test_invalid_physics_kernel_backend_is_rejected(backend: object) -> None:
    """未知或非字串 CPU 核心後端不得靜默退回 NumPy。"""

    payload = _payload()
    payload["execution"]["physics_kernel_backend"] = backend
    with pytest.raises((TypeError, ValueError)):
        ProjectConfig.model_validate(payload)


@pytest.mark.parametrize("backend", ["numpy_v1", "numba_ocm_v1"])
def test_explicit_ocm_interpolation_backend_is_versioned_and_hashed(backend: str) -> None:
    """明示的 OCM backend 應保留在 normalized config，讓 run binding 可追溯。"""

    payload = _payload()
    payload["execution"]["ocm_interpolation_backend"] = backend
    config = ProjectConfig.model_validate(payload)
    assert config.execution.ocm_interpolation_backend == backend
    assert config.normalized_payload()["execution"]["ocm_interpolation_backend"] == backend
    assert config.config_hash() != ProjectConfig.model_validate(_payload()).config_hash()


@pytest.mark.parametrize(
    "backend",
    [None, "numpy", "numba", "numba_ocm_v2", True, 1],
)
def test_invalid_ocm_interpolation_backend_is_rejected(backend: object) -> None:
    """未知、空值與非字串 backend 不得由 extra 或型別轉換靜默接受。"""

    payload = _payload()
    payload["execution"]["ocm_interpolation_backend"] = backend
    with pytest.raises((TypeError, ValueError)):
        ProjectConfig.model_validate(payload)


def test_explicit_backtrack_support_is_strict_and_bounds_requested_horizon() -> None:
    """明示母體支援窗只接受正整日，且 requested 不得超出母體。"""

    payload = _payload()
    payload["inputs"]["backtrack_support_days"] = 30
    payload["boundaries"].update({"max_backtrack_days": 7.0, "maximum_step_count": 10_000})
    config = ProjectConfig.model_validate(payload)
    assert config.inputs.backtrack_support_days == 30
    assert config.effective_backtrack_support_days == 30

    for invalid in (True, 0, -1, 30.0, "30", float("nan"), float("inf")):
        candidate = deepcopy(payload)
        candidate["inputs"]["backtrack_support_days"] = invalid
        with pytest.raises((TypeError, ValueError)):
            ProjectConfig.model_validate(candidate)

    over = deepcopy(payload)
    over["boundaries"]["max_backtrack_days"] = 31.0
    with pytest.raises(ValueError, match="不得超過"):
        ProjectConfig.model_validate(over)


def test_explicit_null_backtrack_support_only_allows_preparation_config() -> None:
    """明示 null 表示母體尚未定案；有 requested horizon 時必須 fail closed。"""

    payload = _payload()
    payload["inputs"]["backtrack_support_days"] = None
    payload["boundaries"].update({"max_backtrack_days": None, "maximum_step_count": None})
    assert ProjectConfig.model_validate(payload).inputs.backtrack_support_days is None
    requested = deepcopy(payload)
    requested["boundaries"]["max_backtrack_days"] = 7.0
    with pytest.raises(ValueError, match="尚未定案"):
        ProjectConfig.model_validate(requested)


def test_legacy_config_hash_preserves_omitted_policy_semantics() -> None:
    """未含新 policy 欄位的既有設定必須保留舊 canonical hash 與 payload 語意。"""

    config = ProjectConfig.model_validate(_legacy_hash_payload())
    assert config.config_hash() == (
        "d7151792e1432b919ce46a8f2a40571f7c78977a500d2d2f472cccdbc921d711"
    )
    assert config.domains[0].formal_domain_policy == "expanded_domain_v1"
    assert "formal_domain_policy" not in config.domains[0].model_fields_set
    assert "formal_domain_policy" not in config.normalized_payload()["domains"][0]


def test_omitted_optional_boundary_controls_preserve_legacy_payload_shape() -> None:
    """舊 YAML 未明示的新 BoundaryConfig 欄位不得以 null 寫入 normalized hash。"""

    payload = _legacy_hash_payload()
    payload["boundaries"].pop("stop_at_forcing_start", None)
    payload["boundaries"].pop("stop_at_data_gap", None)
    config = ProjectConfig.model_validate(payload)
    normalized_boundaries = config.normalized_payload()["boundaries"]
    assert "stop_at_forcing_start" not in normalized_boundaries
    assert "stop_at_data_gap" not in normalized_boundaries


def test_rejects_region_a_site_merging() -> None:
    """A 區必須同時保留貢寮與龜山島兩個獨立 study_site_id。"""

    payload = deepcopy(_payload())
    payload["study_sites"] = [site for site in payload["study_sites"] if site["study_site_id"] != "guishan"]
    payload["study_area"]["expected_study_site_count"] = 4
    with pytest.raises(ValueError, match="A 區必須恰含"):
        ProjectConfig.model_validate(payload)


def test_rejects_reduced_scenario_count() -> None:
    """不得以修改設定把全案完整交叉靜默降回 10,000 或每站 1,000。"""

    payload = _payload()
    payload["scenarios"]["scenario_count"] = 10_000
    with pytest.raises(ValueError, match="情境契約"):
        ProjectConfig.model_validate(payload)


def test_example_is_intentionally_blocked_for_formal_release() -> None:
    """新 v3 design example 尚缺衍生 manifest，正式模式必須 fail closed。"""

    with pytest.raises(ValueError, match="approved reconstruction 或 gap-safe"):
        load_config(EXAMPLE_CONFIG, formal_release=True)


def test_formal_time_gate_accepts_either_ocm_support_manifest_name() -> None:
    """重建未過門檻時可採 gap-safe baseline，設定 gate 不得強迫偽造重建成功。"""

    payload = _payload()
    payload["inputs"]["ocm_gap_safe_arrival_manifest"] = "manifests/ocm-gap-safe.json"
    config = ProjectConfig.model_validate(payload)
    with pytest.raises(ValueError) as exc_info:
        config.assert_formal_release_ready()
    assert "approved reconstruction 或 gap-safe" not in str(exc_info.value)


def test_yaml_member_field_is_not_silently_ignored() -> None:
    """schema 欄位必須直接對應 YAML 的 members_per_scenario，避免正式 M 永遠讀成 None。"""

    payload = _payload()
    payload["scenarios"]["members_per_scenario"] = 8
    config = ProjectConfig.model_validate(payload)
    assert config.scenarios.members_per_scenario == 8


def test_legacy_checkpoint_interval_field_is_rejected_locally() -> None:
    """舊 output-step 欄位不得因 StrictModel extra=allow 而靜默進入設定。"""

    payload = _payload()
    payload["execution"]["checkpoint_interval_output_steps"] = 3
    with pytest.raises(ValueError, match="checkpoint_interval_sweeps"):
        ProjectConfig.model_validate(payload)


def test_flow_domain_resolver_selects_formal_release_id_only_in_formal_mode() -> None:
    """legacy expanded policy 的 A 區 formal 使用 expanded ID；pilot 維持 base。"""

    payload = _payload()
    # expanded_domain_v1 是 v2 legacy scope；v3 design 必須和本期 v3 policy 雙向綁定，
    # 不可藉改 policy 將現行 v3 config 降回舊來源。
    payload["design_version"] = "design_baseline_v2_non_rising_oca_proxy"
    payload["arrival_time_selection"].pop("policy", None)
    payload["arrival_time_selection"].pop("observation_years", None)
    payload["arrival_time_selection"].pop("replicates", None)
    payload["domains"][0]["formal_domain_policy"] = "expanded_domain_v1"
    payload["domains"][0].pop("runtime_spatial_support_policy", None)
    payload["domains"][0]["formal_release_flow_domain_id"] = (
        "northeast_taiwan_common_cache_v4_lbt_south_expanded"
    )
    payload["domains"][0]["formal_release_domain_status"] = "approved"
    payload["domains"][0]["expanded_domain_candidate_id"] = (
        "northeast_taiwan_common_cache_v4_lbt_south_expanded"
    )
    payload["domains"][0]["expanded_bbox_lon_lat"] = [121.306315, 122.793685, 24.480000, 25.499156]
    for site in payload["study_sites"]:
        if site["analysis_region_id"] == "A":
            site["formal_release_flow_domain_id"] = (
                "northeast_taiwan_common_cache_v4_lbt_south_expanded"
            )
    config = ProjectConfig.model_validate(payload)
    assert resolve_flow_domain_id(config, "A") == "northeast_taiwan_common_cache_v3"
    assert (
        resolve_flow_domain_id(config, "A", formal=True)
        == "northeast_taiwan_common_cache_v4_lbt_south_expanded"
    )
    assert resolve_flow_domain_id(config, "B", formal=True) == "hsinchu_cache_v3"


def test_v3_local20_policy_records_no_expansion_runtime_stage_contract() -> None:
    """A 區固定 v3/20 km 與逐 stage policy，且不宣稱已量測共同 forcing margin。"""

    config = load_config(EXAMPLE_CONFIG)
    assert config.domains[0].formal_domain_policy == "v3_local20km_20260909_v1"
    assert config.domains[0].runtime_spatial_support_policy == (
        "runtime_stage_fail_closed_no_expansion_v1"
    )
    assert config.domains[0].formal_release_domain_status == "no_expansion_runtime_stage_fail_closed"
    assert resolve_flow_domain_id(config, "A", formal=True) == "northeast_taiwan_common_cache_v3"
    assert config.boundaries.stop_at_data_gap is True
    assert config.boundaries.stop_at_forcing_start is True
    assert config.boundaries.flow_domain_open_boundary == "stop_at_first_crossing"
    assert "minimum_common_forcing_margin_grid_cells" not in (config.domains[0].model_extra or {})
    assert all(
        site.model_extra["minimum_flow_domain_margin_local_grid_scales"] == 2
        for site in config.study_sites
        if site.analysis_region_id == "A"
    )


def test_v3_formal_config_gate_no_longer_has_unconditional_margin_blocker() -> None:
    """完整設定可通過 config-only gate；這不替代 accepted-product 與 manifest 驗證。"""

    payload = _payload()
    payload["config_status"] = "approved"
    payload["inputs"].update(
        {
            "ocm_gap_safe_arrival_manifest": "manifests/fake-ocm-gap-safe.json",
            "nww_full_hourly_analysis_manifest": "manifests/fake-nww-hourly.json",
        }
    )
    payload["scenarios"].update(
        {
            "members_per_scenario": 64,
            "master_seed": 20260909,
            "receptor_manifest": "manifests/fake-receptor.json",
            "arrival_time_manifest": "manifests/fake-arrival.json",
            "material_manifest": "manifests/fake-material.json",
            "receptor_arrival_initial_condition_manifest": "manifests/fake-initial.json",
        }
    )
    payload["geometry"].update(
        {
            "domain_manifest": "manifests/fake-domain.json",
            "local_domain_manifest": "manifests/fake-local.json",
            "open_boundary_manifest": "manifests/fake-open.json",
            "receptor_manifest": "manifests/fake-receptor-geometry.json",
        }
    )
    payload["physics"]["settling"]["material_manifest"] = "manifests/fake-material.json"
    payload["physics"]["horizontal_diffusion"]["constant_kh_m2ps"] = 1.0
    payload["physics"]["vertical_diffusion"]["constant_kz_m2ps"] = 0.001
    payload["forcing"]["ocm"]["wetdry_semantics_decision_status"] = "approved"
    payload["integration"].update(
        {
            "output_interval_seconds": 3600.0,
            "dt_min_seconds": 30.0,
            "dt_max_seconds": 300.0,
        }
    )
    payload["boundaries"].update({"max_backtrack_days": 7.0, "maximum_step_count": 20_160})
    payload["execution"].update({"shard_scenario_count": 1_000, "checkpoint_interval_sweeps": 10})

    config = ProjectConfig.model_validate(payload)
    config.assert_formal_release_ready()


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (
            ("domains", 0, "formal_release_flow_domain_id"),
            "northeast_taiwan_common_cache_v4",
            "formal flow domain",
        ),
        (("domains", 0, "flow_domain_id"), "northeast_taiwan_common_cache_v4", "exact v3"),
        (("domains", 0, "bbox_lon_lat"), [121.30, 122.79, 24.60, 25.50], "bbox"),
        (("study_sites", 0, "local_domain_baseline_radius_m"), 35_000, "local radius"),
        (("study_sites", 1, "receptor_core_radius_m"), 20_000, "receptor core"),
        (("study_sites", 0, "local_domain_sensitivity_radii_m"), [35_000], "sensitivity"),
        (("study_sites", 0, "radius_35000_requires_expanded_flow_domain"), True, "expanded radius mandatory"),
        (("domains", 0, "minimum_common_forcing_margin_grid_cells"), 2, "未量測的共同 forcing margin"),
        (
            ("domains", 0, "expanded_domain_candidate_id"),
            "northeast_taiwan_common_cache_v4",
            "expanded candidate",
        ),
    ],
)
def test_v3_local20_policy_rejects_scope_drift(
    path: tuple[str, int, str], value: object, message: str
) -> None:
    """v3 policy 的來源、半徑、sensitivity 與 margin 不能退回未核准的研究設計。"""

    payload = _payload()
    section, index, field = path
    payload[section][index][field] = value
    if section == "domains" and field == "flow_domain_id":
        # 先同步兩站的 region/base ID，讓測試到達 v3 的 exact ID policy，而非被更早的
        # 通用 region-to-domain 一致性檢查攔截；這不是放寬規則，只隔離目標 gate。
        for site in payload["study_sites"]:
            if site["analysis_region_id"] == "A":
                site["flow_domain_id"] = value
    with pytest.raises(ValueError, match=message):
        ProjectConfig.model_validate(payload)


def test_v3_local20_policy_rejects_unknown_policy() -> None:
    """未知 policy 不得由 extra=allow 靜默變成 legacy 或 v3。"""

    payload = _payload()
    payload["domains"][0]["formal_domain_policy"] = "v3_local20km_unregistered"
    with pytest.raises(ValueError, match="formal_domain_policy"):
        ProjectConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("domains", "runtime_spatial_support_policy", None, "runtime_spatial_support_policy"),
        ("boundaries", "stop_at_data_gap", False, "stop_at_data_gap=true"),
        ("boundaries", "stop_at_forcing_start", False, "stop_at_forcing_start=true"),
        (
            "boundaries",
            "flow_domain_open_boundary",
            "record_and_continue",
            "flow_domain_open_boundary=stop_at_first_crossing",
        ),
    ],
)
def test_v3_runtime_spatial_support_policy_rejects_missing_or_relaxed_controls(
    section: str, field: str, value: object, message: str
) -> None:
    """逐 stage 契約缺漏或任一既有停止控制被放寬時，必須在 config 層拒絕。"""

    payload = _payload()
    if section == "domains":
        payload["domains"][0][field] = value
    else:
        payload["boundaries"][field] = value
    with pytest.raises(ValueError, match=message):
        ProjectConfig.model_validate(payload)


@pytest.mark.parametrize("missing_field", ["anchor_lonlat", "receptor_core_radius_m"])
def test_receptor_core_anchor_and_radius_must_be_declared_as_a_pair(missing_field: str) -> None:
    """核心 anchor 與公尺半徑缺一時，config-check 應在讀取 inputs 前 fail closed。"""

    payload = _payload()
    hsinchu = next(site for site in payload["study_sites"] if site["study_site_id"] == "hsinchu")
    hsinchu[missing_field] = None
    with pytest.raises(ValueError, match="必須同時明示 anchor_lonlat 與 receptor_core_radius_m"):
        ProjectConfig.model_validate(payload)


@pytest.mark.parametrize("invalid_radius", [0.0, -1.0, float("inf"), float("nan")])
def test_receptor_core_radius_must_be_finite_and_positive(invalid_radius: float) -> None:
    """核心半徑是 AEQD 公尺距離，零、負值與非有限值不得進入幾何 intersection。"""

    payload = _payload()
    hsinchu = next(site for site in payload["study_sites"] if site["study_site_id"] == "hsinchu")
    hsinchu["receptor_core_radius_m"] = invalid_radius
    with pytest.raises(ValueError, match="receptor_core_radius_m 必須是有限正數"):
        ProjectConfig.model_validate(payload)


def test_formal_release_requires_dynamic_initial_condition_manifest_path() -> None:
    """正式設定即使已有 receptor／arrival path，也不可省略 dynamic pair manifest。"""

    payload = _payload()
    payload["scenarios"]["receptor_arrival_initial_condition_manifest"] = None
    config = ProjectConfig.model_validate(payload)
    with pytest.raises(ValueError, match="receptor_arrival_initial_condition_manifest"):
        config.assert_formal_release_ready()
