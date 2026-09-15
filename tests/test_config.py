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


def _legacy_hash_payload() -> dict:
    """由現行範例還原未含 policy 的舊設定快照，避免測試依賴 git 指令。"""

    payload = deepcopy(_payload())
    payload["config_status"] = "design_baseline_example"
    payload["design_version"] = "design_baseline_v2_non_rising_oca_proxy"
    a_domain = payload["domains"][0]
    a_domain.pop("formal_domain_policy", None)
    a_domain["formal_release_flow_domain_id"] = None
    a_domain["formal_release_domain_status"] = "expanded_domain_generation_required"
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
        site.pop("receptor_candidate_regions", None)
        site.pop("receptor_candidate_selection", None)
        site.pop("receptor_candidate_regions_provenance", None)
    payload["boundaries"].pop("sensitivity_case_region_exclusions", None)
    return payload


def test_example_config_has_fixed_scientific_counts() -> None:
    """範例設定必須保留 4 domains、5 sites 與全案 50,000 基礎情境。"""

    config = load_config(EXAMPLE_CONFIG)
    assert len(config.domains) == 4
    assert len(config.study_sites) == 5
    assert config.scenarios.expected_receptor_count == 100
    assert config.scenarios.scenario_count == 50_000
    assert config.scenarios.seed_policy == "sha256_v1_pcg64dxsm"
    assert config.execution.checkpoint_interval_sweeps is None
    assert config.execution.active_chunk_size is None
    assert config.execution.max_resident_forcing_months == 2


def test_receptor_candidate_domain_policy_is_versioned_and_core_aware() -> None:
    """設定檔應明示 core∩local 與未明示 core fallback 的候選區政策。"""

    payload = _payload()
    assert payload["scenarios"]["other_site_receptor_candidate_domain"] == (
        "site_explicit_core_intersect_local_else_local_or_flow_v1"
    )


def test_example_material_table_matches_runtime_baseline() -> None:
    """YAML 與程式內建表不可各自維護不同的材質、形狀、條件或速度。"""

    payload = _payload()
    assert payload["design_version"] == "design_baseline_v3_non_rising_a_v3_local20_20260909"
    assert payload["physics"]["settling"]["material_classes"] == [asdict(item) for item in BASELINE_BEHAVIORS]


def test_example_registers_c_and_d_receptor_anchors_and_c_red_frame_quota() -> None:
    """目前試跑回寫的 C／D 受體 anchor、核心與 C 區 2+3 配額必須進入設定。"""

    config = load_config(EXAMPLE_CONFIG)
    sites = {site.study_site_id: site for site in config.study_sites}
    assert config.design_version == CURRENT_DESIGN_VERSION
    assert sites["houwan"].anchor_lonlat == (120.893355, 22.0)
    assert sites["houwan"].receptor_core_radius_m == 12_500
    assert sites["lienchiang"].anchor_lonlat == (119.95, 26.2)
    assert sites["lienchiang"].receptor_core_radius_m == 12_500
    assert sites["houwan"].receptor_candidate_selection["total_horizontal_count"] == 5
    assert [item["allocation_count"] for item in sites["houwan"].receptor_candidate_regions] == [2, 3]


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


def test_omitted_backtrack_support_keeps_current_example_hash() -> None:
    """未宣告新支援欄位時，現行範例 hash 必須維持主專案已凍結值。"""

    config = ProjectConfig.model_validate(_payload())
    assert config.config_hash() == "163ee4f9f113a567206a28354e69e783db2da4eedc6ea618276893d4c1554214"
    assert "backtrack_support_days" not in config.normalized_payload()["inputs"]


def test_omitted_ocm_interpolation_backend_keeps_numpy_and_current_hash() -> None:
    """舊 YAML 省略 OCM backend 時走 NumPy，且不得因預設欄位改變 canonical hash。"""

    config = ProjectConfig.model_validate(_payload())
    assert config.execution.ocm_interpolation_backend == "numpy_v1"
    assert "ocm_interpolation_backend" not in config.normalized_payload()["execution"]
    assert config.config_hash() == "163ee4f9f113a567206a28354e69e783db2da4eedc6ea618276893d4c1554214"


def test_omitted_physics_kernel_backend_keeps_numpy_and_current_hash() -> None:
    """舊 YAML 未帶 CPU 核心選項時仍用 NumPy，且 config hash 維持凍結值。"""

    config = ProjectConfig.model_validate(_payload())
    assert config.execution.physics_kernel_backend == "numpy_v1"
    assert "physics_kernel_backend" not in config.normalized_payload()["execution"]
    assert config.config_hash() == "163ee4f9f113a567206a28354e69e783db2da4eedc6ea618276893d4c1554214"


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
    payload["domains"][0]["formal_domain_policy"] = "expanded_domain_v1"
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


def test_v3_local20_policy_loads_but_formal_release_stays_blocked() -> None:
    """A 區核准的 v3/20 km scope 可載入，但共同 forcing 邊界未驗證時必須阻擋 formal。"""

    config = load_config(EXAMPLE_CONFIG)
    assert config.domains[0].formal_domain_policy == "v3_local20km_20260909_v1"
    assert resolve_flow_domain_id(config, "A", formal=True) == "northeast_taiwan_common_cache_v3"
    with pytest.raises(ValueError, match="v3/20km共同forcing邊界支援尚待實際驗證"):
        config.assert_formal_release_ready()


def test_v3_formal_gate_ignores_fake_approved_evidence() -> None:
    """新 policy 即使填滿 approved／path 欄位，也不得偽造共同 forcing 邊界證據。"""

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
    with pytest.raises(ValueError, match="v3/20km共同forcing邊界支援尚待實際驗證"):
        config.assert_formal_release_ready()


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (
            ("domains", 0, "formal_release_flow_domain_id"),
            "northeast_taiwan_common_cache_v4",
            "formal flow domain",
        ),
        (("study_sites", 0, "local_domain_baseline_radius_m"), 35_000, "local radius"),
        (("study_sites", 1, "receptor_core_radius_m"), 20_000, "receptor core"),
        (("study_sites", 0, "local_domain_sensitivity_radii_m"), [35_000], "sensitivity"),
        (("study_sites", 0, "radius_35000_requires_expanded_flow_domain"), True, "expanded radius mandatory"),
        (("domains", 0, "minimum_common_forcing_margin_grid_cells"), 0, "margin"),
    ],
)
def test_v3_local20_policy_rejects_scope_drift(
    path: tuple[str, int, str], value: object, message: str
) -> None:
    """v3 policy 的來源、半徑、sensitivity 與 margin 不能退回未核准的研究設計。"""

    payload = _payload()
    section, index, field = path
    payload[section][index][field] = value
    with pytest.raises(ValueError, match=message):
        ProjectConfig.model_validate(payload)


def test_v3_local20_policy_rejects_unknown_policy() -> None:
    """未知 policy 不得由 extra=allow 靜默變成 legacy 或 v3。"""

    payload = _payload()
    payload["domains"][0]["formal_domain_policy"] = "v3_local20km_unregistered"
    with pytest.raises(ValueError, match="formal_domain_policy"):
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
