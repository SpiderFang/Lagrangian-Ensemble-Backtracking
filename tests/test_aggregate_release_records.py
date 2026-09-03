"""聚合 release record 資料契約與情境 join 的完整測試。

本檔只測試 ``aggregate_release_records`` 對已建立資料列的邊界驗證，不讀取 raw
NetCDF，也不改動 production。測試 fixture 使用公尺制位置、速度與垂向值；經緯度
僅代表 WGS84 交換欄位。動態初始條件代表單一 receptor×arrival pair 的 OCM-derived
資料，整組 ``None`` 則代表 pilot bundle 沒有任何可用動態條件，而不是把未知狀態
轉成零值。
"""

from __future__ import annotations

import math
from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pytest

from lagrangian_backtracking.aggregate_release_records import (
    AggregateShardBinding,
    ScenarioStratum,
    scenario_inputs_to_strata,
)
from lagrangian_backtracking.manifests import ScenarioInputs
from lagrangian_backtracking.scenarios import (
    ArrivalTime,
    Behavior,
    Receptor,
    ReceptorArrivalInitialCondition,
    Scenario,
)

_DESIGN_VERSION = "design-v1"
_WETDRY_SEMANTICS_ID = "schism_wetdry_elem_0_wet_1_dry"
_DYNAMIC_FLOAT_FIELDS = (
    "initial_z_m_positive_up",
    "initial_eta_m_positive_up",
    "initial_bed_z_m_positive_up",
    "initial_water_column_height_m",
    "initial_height_above_bed_m",
    "initial_zcor_lower_m_positive_up",
    "initial_zcor_upper_m_positive_up",
    "initial_vertical_bracket_alpha",
)
_SCIENTIFIC_FLOAT_FIELDS = (
    "settling_velocity_mps",
    "receptor_lon_deg",
    "receptor_lat_deg",
    "receptor_template_z_m_positive_up",
    *_DYNAMIC_FLOAT_FIELDS,
)
_REQUIRED_TEXT_FIELDS = (
    "scenario_id",
    "study_site_id",
    "analysis_region_id",
    "material_id",
    "material_category_zh",
    "material_family_zh",
    "representative_shape_zh",
    "behavior_class",
    "applicability_condition_zh",
    "calibration_status",
    "evidence_grade",
    "receptor_id",
    "vertical_id",
    "arrival_time_id",
    "season",
    "tide_class",
    "phase_or_event",
    "design_version",
)
_OPTIONAL_DYNAMIC_TEXT_FIELDS = (
    "initial_wetdry_semantics_id",
    "initial_ocm_month_yyyymm",
    "initial_ocm_time_origin",
)


def _valid_binding() -> AggregateShardBinding:
    """建立含句點 slug 的最小合法 shard 綁定，供各測試以 ``replace`` 竄改。

    scenario index 是半開區間 ``[start, stop)``；三種 count 分別代表粒子、觀測與
    事件資料列數，且 observation/event 必須至少各有一筆對應每個粒子的基礎資料。
    路徑刻意使用 POSIX 相對路徑，hash 則使用 64 位小寫 SHA-256 測試值。
    """

    return AggregateShardBinding(
        shard_id="release.v1",
        scenario_start_index=0,
        scenario_stop_index=2,
        output_relative_path="shards/release.v1/trajectory.parquet",
        trajectory_manifest_sha256="a" * 64,
        particle_count=2,
        observation_count=4,
        event_count=3,
    )


def _valid_stratum_kwargs(*, dynamic: bool) -> dict[str, object]:
    """建立 ScenarioStratum 的完整合法欄位字典。

    static 欄位對應 Behavior、Receptor、ArrivalTime 與 Scenario 的 join 結果；dynamic
    欄位在 ``dynamic=False`` 時必須全部為 ``None``，在 ``dynamic=True`` 時則完整描述
    OCM 的實際深度、海面高程、海床、垂向 bracket、來源 face、濕乾旗標及時間 provenance。
    ``z`` 位於 ``[-5, -3]`` 公尺 bracket 內，水柱與離床高度均以非負公尺保存。
    """

    values: dict[str, object] = {
        "scenario_id": "scenario-001",
        "study_site_id": "site-a",
        "analysis_region_id": "region-a",
        "material_id": "material-a",
        "material_category_zh": "塑膠",
        "material_family_zh": "聚乙烯",
        "representative_shape_zh": "片狀",
        "behavior_class": "sinking",
        "settling_velocity_mps": -0.5,
        "applicability_condition_zh": "已吸水且完全沉沒",
        "calibration_status": "provisional",
        "evidence_grade": "C",
        "receptor_id": "receptor-a",
        "receptor_lon_deg": 121.5,
        "receptor_lat_deg": 22.5,
        "receptor_template_z_m_positive_up": -5.0,
        "vertical_id": "z2",
        "arrival_time_id": "arrival-001",
        "arrival_time_utc_ns": 1_735_689_600_000_000_000,
        "arrival_year": 2025,
        "season": "JJA",
        "tide_class": "spring_proxy",
        "phase_or_event": "fastest_rising",
        "design_version": "design-v1",
        "initial_z_m_positive_up": None,
        "initial_eta_m_positive_up": None,
        "initial_bed_z_m_positive_up": None,
        "initial_water_column_height_m": None,
        "initial_height_above_bed_m": None,
        "initial_zcor_lower_m_positive_up": None,
        "initial_zcor_upper_m_positive_up": None,
        "initial_vertical_bracket_alpha": None,
        "initial_source_face_local_index": None,
        "initial_source_face_global_index": None,
        "initial_wetdry_elem_value": None,
        "initial_wetdry_semantics_id": None,
        "initial_ocm_month_yyyymm": None,
        "initial_ocm_source_time_index": None,
        "initial_ocm_time_origin": None,
    }
    if dynamic:
        values.update(
            {
                "initial_z_m_positive_up": -4.0,
                "initial_eta_m_positive_up": 0.5,
                "initial_bed_z_m_positive_up": -10.0,
                "initial_water_column_height_m": 10.5,
                "initial_height_above_bed_m": 6.0,
                "initial_zcor_lower_m_positive_up": -5.0,
                "initial_zcor_upper_m_positive_up": -3.0,
                "initial_vertical_bracket_alpha": 0.5,
                "initial_source_face_local_index": 10,
                "initial_source_face_global_index": 100,
                "initial_wetdry_elem_value": 0,
                "initial_wetdry_semantics_id": "schism_wetdry_elem_0_wet_1_dry",
                "initial_ocm_month_yyyymm": "202501",
                "initial_ocm_source_time_index": 4,
                "initial_ocm_time_origin": "2025-01-01T00:00:00Z",
            }
        )
    return values


def test_valid_aggregate_shard_binding_accepts_dot_slug() -> None:
    """合法綁定應接受含句點的 slug，並保留所有已驗證欄位。"""

    binding = _valid_binding()

    assert binding.shard_id == "release.v1"
    assert binding.output_relative_path == "shards/release.v1/trajectory.parquet"
    assert binding.scenario_start_index == 0
    assert binding.scenario_stop_index == 2
    assert binding.particle_count == 2
    assert binding.observation_count == 4
    assert binding.event_count == 3


def test_valid_scenario_stratum_accepts_pilot_none_and_complete_dynamic() -> None:
    """pilot 可保留整組動態缺值，完整 dynamic row 則應宣告具備初始條件。"""

    pilot = ScenarioStratum(**_valid_stratum_kwargs(dynamic=False))
    formal = ScenarioStratum(**_valid_stratum_kwargs(dynamic=True))

    assert pilot.has_dynamic_initial_condition is False
    assert pilot.initial_z_m_positive_up is None
    assert formal.has_dynamic_initial_condition is True
    assert formal.initial_z_m_positive_up == -4.0
    assert formal.initial_zcor_lower_m_positive_up == -5.0
    assert formal.initial_zcor_upper_m_positive_up == -3.0


def _scenario_inputs_fixture() -> ScenarioInputs:
    """建立兩筆真實 component record 與反向 scenario 順序的最小 join fixture。

    兩個 scenario 共用站點與分析區域，但各自使用不同的 Behavior、Receptor、ArrivalTime
    及 receptor×arrival 動態條件。輸出情境刻意排列為 ``scenario-002``、``scenario-001``，
    用來確認聚合前轉換器保留 immutable scenario 順序，而不依 component 或識別碼排序。
    ``initial_conditions`` 只保存 pair 維度，不因 material 數量重複展開。
    """

    behavior_a = Behavior(
        material_id="material-a",
        oca_category_zh="塑膠",
        material_family_zh="聚乙烯",
        representative_shape_zh="片狀",
        settling_velocity_mps=-0.5,
        behavior_class="sinking",
        applicability_condition_zh="已吸水且完全沉沒",
        calibration_status="provisional",
        evidence_grade="C",
    )
    behavior_b = Behavior(
        material_id="material-b",
        oca_category_zh="纖維",
        material_family_zh="尼龍",
        representative_shape_zh="網片",
        settling_velocity_mps=-0.25,
        behavior_class="sinking",
        applicability_condition_zh="已進水且完全沉沒",
        calibration_status="screening",
        evidence_grade="D",
    )
    receptor_a = Receptor(
        receptor_id="receptor-a",
        study_site_id="site-a",
        analysis_region_id="region-a",
        lon=121.5,
        lat=22.5,
        z_m_positive_up=-5.0,
        vertical_id="z2",
        metadata={"candidate_rank": 1},
    )
    receptor_b = Receptor(
        receptor_id="receptor-b",
        study_site_id="site-a",
        analysis_region_id="region-a",
        lon=121.6,
        lat=22.6,
        z_m_positive_up=-6.0,
        vertical_id="z3",
        metadata={"candidate_rank": 2},
    )
    arrival_a = ArrivalTime(
        arrival_time_id="arrival-001",
        study_site_id="site-a",
        time_utc_ns=1_735_689_600_000_000_000,
        year=2025,
        season="JJA",
        tide_class="spring_proxy",
        phase_or_event="fastest_rising",
        metadata={"selection_rank": 0},
    )
    arrival_b = ArrivalTime(
        arrival_time_id="arrival-002",
        study_site_id="site-a",
        time_utc_ns=1_735_693_200_000_000_000,
        year=2025,
        season="DJF",
        tide_class="neap_proxy",
        phase_or_event="fastest_falling",
        metadata={"selection_rank": 1},
    )
    pair_a = ReceptorArrivalInitialCondition(
        receptor_id="receptor-a",
        arrival_time_id="arrival-001",
        study_site_id="site-a",
        analysis_region_id="region-a",
        flow_domain_id="domain-a",
        time_utc_ns=arrival_a.time_utc_ns,
        vertical_id="z2",
        z_m_positive_up=-4.0,
        eta_m_positive_up=0.5,
        bed_z_m_positive_up=-10.0,
        water_column_height_m=10.5,
        height_above_bed_m=6.0,
        zcor_lower_m_positive_up=-5.0,
        zcor_upper_m_positive_up=-3.0,
        vertical_bracket_alpha=0.5,
        source_face_local_index=10,
        source_face_global_index=100,
        wetdry_elem_value=0,
        wetdry_semantics_id=_WETDRY_SEMANTICS_ID,
        ocm_month_yyyymm="202501",
        ocm_source_time_index=4,
        ocm_time_origin="2025-01-01T00:00:00Z",
    )
    pair_b = ReceptorArrivalInitialCondition(
        receptor_id="receptor-b",
        arrival_time_id="arrival-002",
        study_site_id="site-a",
        analysis_region_id="region-a",
        flow_domain_id="domain-a",
        time_utc_ns=arrival_b.time_utc_ns,
        vertical_id="z3",
        z_m_positive_up=-6.0,
        eta_m_positive_up=0.25,
        bed_z_m_positive_up=-12.0,
        water_column_height_m=12.25,
        height_above_bed_m=6.25,
        zcor_lower_m_positive_up=-7.0,
        zcor_upper_m_positive_up=-5.0,
        vertical_bracket_alpha=0.5,
        source_face_local_index=11,
        source_face_global_index=101,
        wetdry_elem_value=0,
        wetdry_semantics_id=_WETDRY_SEMANTICS_ID,
        ocm_month_yyyymm="202501",
        ocm_source_time_index=5,
        ocm_time_origin="2025-01-01T01:00:00Z",
    )
    scenario_a = Scenario(
        scenario_id="scenario-001",
        study_site_id="site-a",
        analysis_region_id="region-a",
        material_id="material-a",
        receptor_id="receptor-a",
        arrival_time_id="arrival-001",
        settling_velocity_mps=behavior_a.settling_velocity_mps,
        arrival_time_utc_ns=arrival_a.time_utc_ns,
        design_version=_DESIGN_VERSION,
    )
    scenario_b = Scenario(
        scenario_id="scenario-002",
        study_site_id="site-a",
        analysis_region_id="region-a",
        material_id="material-b",
        receptor_id="receptor-b",
        arrival_time_id="arrival-002",
        settling_velocity_mps=behavior_b.settling_velocity_mps,
        arrival_time_utc_ns=arrival_b.time_utc_ns,
        design_version=_DESIGN_VERSION,
    )
    return ScenarioInputs(
        materials=(behavior_a, behavior_b),
        receptors=(receptor_a, receptor_b),
        arrival_times=(arrival_a, arrival_b),
        scenarios=(scenario_b, scenario_a),
        file_sha256={},
        canonical_component_hashes={},
        design_version=_DESIGN_VERSION,
        initial_conditions=(pair_b, pair_a),
    )


@pytest.fixture
def scenario_inputs() -> ScenarioInputs:
    """每個測試取得獨立的真實 ScenarioInputs dataclass fixture。"""

    return _scenario_inputs_fixture()


def _without_dynamic_conditions(inputs: ScenarioInputs) -> ScenarioInputs:
    """移除整份 pair manifest，並清空 derived pair index 以維持 ScenarioInputs 契約。"""

    return replace(inputs, initial_conditions=(), initial_conditions_by_pair={})


def _with_scenarios(inputs: ScenarioInputs, scenarios: tuple[Scenario, ...]) -> ScenarioInputs:
    """以指定 scenario tuple 取代 immutable fixture，保留其他 component 與 pair 資料。"""

    return replace(inputs, scenarios=scenarios)


def _with_initial_conditions(
    inputs: ScenarioInputs,
    initial_conditions: tuple[ReceptorArrivalInitialCondition, ...],
) -> ScenarioInputs:
    """取代 pair manifest 並讓 ScenarioInputs 重新建立 pair index，避免測試繞過資料契約。"""

    return replace(
        inputs,
        initial_conditions=initial_conditions,
        initial_conditions_by_pair={},
    )


def _expected_stratum_fields(
    scenario: Scenario,
    behavior: Behavior,
    receptor: Receptor,
    arrival: ArrivalTime,
    initial: ReceptorArrivalInitialCondition,
) -> dict[str, Any]:
    """建立完整欄位預期，逐一對應五種來源 record 的資料語意。

    這個 expected mapping 不從 ``ScenarioStratum`` 反向取值，避免測試只驗證欄位存在；
    它明確核對 material metadata、受體定位、arrival 分層、scenario join 欄位與 OCM
    dynamic provenance 是否各自來自正確來源。所有 scientific numeric fixture 已是原生
    Python 數值，故預期值也符合 converter 的 canonical float 結果。
    """

    return {
        "scenario_id": scenario.scenario_id,
        "study_site_id": scenario.study_site_id,
        "analysis_region_id": scenario.analysis_region_id,
        "material_id": behavior.material_id,
        "material_category_zh": behavior.oca_category_zh,
        "material_family_zh": behavior.material_family_zh,
        "representative_shape_zh": behavior.representative_shape_zh,
        "behavior_class": behavior.behavior_class,
        "settling_velocity_mps": behavior.settling_velocity_mps,
        "applicability_condition_zh": behavior.applicability_condition_zh,
        "calibration_status": behavior.calibration_status,
        "evidence_grade": behavior.evidence_grade,
        "receptor_id": receptor.receptor_id,
        "receptor_lon_deg": receptor.lon,
        "receptor_lat_deg": receptor.lat,
        "receptor_template_z_m_positive_up": receptor.z_m_positive_up,
        "vertical_id": receptor.vertical_id,
        "arrival_time_id": arrival.arrival_time_id,
        "arrival_time_utc_ns": arrival.time_utc_ns,
        "arrival_year": arrival.year,
        "season": arrival.season,
        "tide_class": arrival.tide_class,
        "phase_or_event": arrival.phase_or_event,
        "design_version": _DESIGN_VERSION,
        "initial_z_m_positive_up": initial.z_m_positive_up,
        "initial_eta_m_positive_up": initial.eta_m_positive_up,
        "initial_bed_z_m_positive_up": initial.bed_z_m_positive_up,
        "initial_water_column_height_m": initial.water_column_height_m,
        "initial_height_above_bed_m": initial.height_above_bed_m,
        "initial_zcor_lower_m_positive_up": initial.zcor_lower_m_positive_up,
        "initial_zcor_upper_m_positive_up": initial.zcor_upper_m_positive_up,
        "initial_vertical_bracket_alpha": initial.vertical_bracket_alpha,
        "initial_source_face_local_index": initial.source_face_local_index,
        "initial_source_face_global_index": initial.source_face_global_index,
        "initial_wetdry_elem_value": initial.wetdry_elem_value,
        "initial_wetdry_semantics_id": initial.wetdry_semantics_id,
        "initial_ocm_month_yyyymm": initial.ocm_month_yyyymm,
        "initial_ocm_source_time_index": initial.ocm_source_time_index,
        "initial_ocm_time_origin": initial.ocm_time_origin,
    }


def test_scenario_inputs_to_strata_preserves_order_and_maps_all_fields(
    scenario_inputs: ScenarioInputs,
) -> None:
    """formal 轉換應保留 scenario 順序，且完整映射 Behavior／受體／arrival／pair 欄位。"""

    strata = scenario_inputs_to_strata(scenario_inputs, formal=True)
    behaviors = {item.material_id: item for item in scenario_inputs.materials}
    receptors = {item.receptor_id: item for item in scenario_inputs.receptors}
    arrivals = {item.arrival_time_id: item for item in scenario_inputs.arrival_times}
    pairs = {
        (item.receptor_id, item.arrival_time_id): item
        for item in scenario_inputs.initial_conditions
    }

    assert tuple(item.scenario_id for item in strata) == tuple(
        item.scenario_id for item in scenario_inputs.scenarios
    )
    assert all(item.has_dynamic_initial_condition for item in strata)
    for stratum, scenario in zip(strata, scenario_inputs.scenarios, strict=True):
        behavior = behaviors[scenario.material_id]
        receptor = receptors[scenario.receptor_id]
        arrival = arrivals[scenario.arrival_time_id]
        initial = pairs[(scenario.receptor_id, scenario.arrival_time_id)]
        assert asdict(stratum) == _expected_stratum_fields(
            scenario,
            behavior,
            receptor,
            arrival,
            initial,
        )


def test_scenario_inputs_to_strata_pilot_without_dynamic_conditions(
    scenario_inputs: ScenarioInputs,
) -> None:
    """pilot 僅在整份 initial_conditions 為空時允許所有輸出列的 dynamic 欄位為 None。"""

    pilot_inputs = _without_dynamic_conditions(scenario_inputs)
    strata = scenario_inputs_to_strata(pilot_inputs, formal=False)

    assert len(strata) == len(pilot_inputs.scenarios)
    assert all(not item.has_dynamic_initial_condition for item in strata)
    assert all(
        getattr(stratum, field_name) is None
        for stratum in strata
        for field_name in _DYNAMIC_FLOAT_FIELDS
        + (
            "initial_source_face_local_index",
            "initial_source_face_global_index",
            "initial_wetdry_elem_value",
            *_OPTIONAL_DYNAMIC_TEXT_FIELDS,
            "initial_ocm_source_time_index",
        )
    )


def test_scenario_stratum_canonicalizes_integer_scientific_values_to_float() -> None:
    """原生 int 可作科學數值輸入，但資料列邊界必須保存為 Python float。"""

    payload = _valid_stratum_kwargs(dynamic=True)
    payload.update(
        {
            "settling_velocity_mps": -1,
            "receptor_lon_deg": 121,
            "receptor_lat_deg": 22,
            "receptor_template_z_m_positive_up": -5,
            "initial_z_m_positive_up": -4,
            "initial_eta_m_positive_up": 0,
            "initial_bed_z_m_positive_up": -10,
            "initial_water_column_height_m": 10,
            "initial_height_above_bed_m": 6,
            "initial_zcor_lower_m_positive_up": -5,
            "initial_zcor_upper_m_positive_up": -3,
            "initial_vertical_bracket_alpha": 1,
        }
    )
    stratum = ScenarioStratum(**payload)

    assert all(type(getattr(stratum, field_name)) is float for field_name in _SCIENTIFIC_FLOAT_FIELDS)


@pytest.mark.parametrize(
    "field_name",
    _SCIENTIFIC_FLOAT_FIELDS,
)
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_scenario_stratum_rejects_non_finite_scientific_values(
    field_name: str,
    value: float,
) -> None:
    """所有科學浮點欄位都不得把 NaN 或無限值帶入 release record。"""

    payload = _valid_stratum_kwargs(dynamic=True)
    payload[field_name] = value

    with pytest.raises(ValueError):
        ScenarioStratum(**payload)


@pytest.mark.parametrize("settling_velocity", [0.0, 1.0])
def test_scenario_stratum_rejects_non_negative_settling_velocity(
    settling_velocity: float,
) -> None:
    """沉降速度契約只接受嚴格負值，零與正值均不能表示非上浮材料。"""

    payload = _valid_stratum_kwargs(dynamic=False)
    payload["settling_velocity_mps"] = settling_velocity

    with pytest.raises(ValueError):
        ScenarioStratum(**payload)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("receptor_lon_deg", -180.1),
        ("receptor_lon_deg", 180.1),
        ("receptor_lat_deg", -90.1),
        ("receptor_lat_deg", 90.1),
    ],
)
def test_scenario_stratum_rejects_out_of_range_wgs84_coordinates(
    field_name: str,
    value: float,
) -> None:
    """WGS84 經緯度只允許其標準閉區間，避免交換欄位帶入非法定位。"""

    payload = _valid_stratum_kwargs(dynamic=False)
    payload[field_name] = value

    with pytest.raises(ValueError):
        ScenarioStratum(**payload)


def test_scenario_stratum_rejects_partial_dynamic_condition() -> None:
    """動態條件不可只填部分欄位，避免把缺值誤讀成可用的初始狀態。"""

    absent_payload = _valid_stratum_kwargs(dynamic=False)
    absent_payload["initial_z_m_positive_up"] = -4.0
    present_payload = _valid_stratum_kwargs(dynamic=True)
    present_payload["initial_eta_m_positive_up"] = None

    with pytest.raises(ValueError):
        ScenarioStratum(**absent_payload)
    with pytest.raises(ValueError):
        ScenarioStratum(**present_payload)


@pytest.mark.parametrize(
    "field_name",
    ["initial_water_column_height_m", "initial_height_above_bed_m"],
)
def test_scenario_stratum_rejects_negative_water_or_height_above_bed(field_name: str) -> None:
    """水柱高度與離床高度是幾何長度，不可使用負值掩蓋 OCM 或網格錯誤。"""

    payload = _valid_stratum_kwargs(dynamic=True)
    payload[field_name] = -0.1

    with pytest.raises(ValueError):
        ScenarioStratum(**payload)


@pytest.mark.parametrize(
    ("lower", "initial", "upper"),
    [
        (-5.0, -6.0, -3.0),
        (-2.0, -4.0, -3.0),
        (-5.0, -4.0, -6.0),
    ],
)
def test_scenario_stratum_rejects_initial_z_outside_vertical_bracket(
    lower: float,
    initial: float,
    upper: float,
) -> None:
    """實際初始深度必須落在 OCM 垂向 bracket 的兩端點之間。"""

    payload = _valid_stratum_kwargs(dynamic=True)
    payload["initial_zcor_lower_m_positive_up"] = lower
    payload["initial_z_m_positive_up"] = initial
    payload["initial_zcor_upper_m_positive_up"] = upper

    with pytest.raises(ValueError):
        ScenarioStratum(**payload)


@pytest.mark.parametrize("alpha", [-0.1, 1.1])
def test_scenario_stratum_rejects_vertical_bracket_alpha_outside_unit_interval(
    alpha: float,
) -> None:
    """垂向線性內插權重必須位於 [0, 1]，不能產生 bracket 外的外插。"""

    payload = _valid_stratum_kwargs(dynamic=True)
    payload["initial_vertical_bracket_alpha"] = alpha

    with pytest.raises(ValueError):
        ScenarioStratum(**payload)


@pytest.mark.parametrize(
    "field_name",
    [
        "initial_source_face_local_index",
        "initial_source_face_global_index",
        "initial_ocm_source_time_index",
    ],
)
def test_scenario_stratum_rejects_negative_dynamic_indices(field_name: str) -> None:
    """來源 face 與 OCM 時間索引是陣列定位值，只能是非負原生整數。"""

    payload = _valid_stratum_kwargs(dynamic=True)
    payload[field_name] = -1

    with pytest.raises(ValueError):
        ScenarioStratum(**payload)


def test_scenario_stratum_rejects_nonzero_wetdry_value() -> None:
    """正式可用的 wet/dry 語意只接受 0（濕元素），乾元素不可偽裝成有效初始條件。"""

    payload = _valid_stratum_kwargs(dynamic=True)
    payload["initial_wetdry_elem_value"] = 1

    with pytest.raises(ValueError):
        ScenarioStratum(**payload)


@pytest.mark.parametrize("month", ["202500", "202513", "2025-01", "abc123"])
def test_scenario_stratum_rejects_invalid_ocm_month(month: str) -> None:
    """OCM 月份 provenance 必須是存在月份的 YYYYMM，避免跨月來源無法重建。"""

    payload = _valid_stratum_kwargs(dynamic=True)
    payload["initial_ocm_month_yyyymm"] = month

    with pytest.raises(ValueError):
        ScenarioStratum(**payload)


@pytest.mark.parametrize("field_name", _REQUIRED_TEXT_FIELDS)
def test_scenario_stratum_rejects_whitespace_required_text(field_name: str) -> None:
    """識別、分類與版本文字不可含首尾空白，避免分組 key 出現不可見差異。"""

    payload = _valid_stratum_kwargs(dynamic=False)
    payload[field_name] = f" {payload[field_name]}"

    with pytest.raises(ValueError):
        ScenarioStratum(**payload)


@pytest.mark.parametrize("field_name", _OPTIONAL_DYNAMIC_TEXT_FIELDS)
def test_scenario_stratum_rejects_whitespace_optional_dynamic_text(field_name: str) -> None:
    """dynamic 已存在時，其 optional text provenance 也不能以空白冒充缺值。"""

    payload = _valid_stratum_kwargs(dynamic=True)
    payload[field_name] = " value "

    with pytest.raises(ValueError):
        ScenarioStratum(**payload)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("arrival_time_utc_ns", True),
        ("arrival_year", np.int64(2025)),
        ("initial_source_face_local_index", np.int64(10)),
        ("initial_ocm_source_time_index", range(1)),
    ],
)
def test_scenario_stratum_rejects_non_native_integer_values(
    field_name: str,
    value: object,
) -> None:
    """時間與索引欄位只接受原生 int，拒絕 bool、NumPy scalar 與 range 容器。"""

    payload = _valid_stratum_kwargs(dynamic=True)
    payload[field_name] = value

    with pytest.raises(ValueError):
        ScenarioStratum(**payload)


@pytest.mark.parametrize(
    "shard_id",
    ["a" * 129, "-starts-with-symbol", "contains space", "contains/slash", ""],
)
def test_aggregate_shard_binding_rejects_invalid_slug(shard_id: str) -> None:
    """shard slug 必須短於 129 字元、以英數字起始，且只含契約允許字元。"""

    with pytest.raises(ValueError):
        replace(_valid_binding(), shard_id=shard_id)


@pytest.mark.parametrize(
    "path",
    [
        "/absolute/path/trajectory.parquet",
        "../trajectory.parquet",
        "a/../trajectory.parquet",
        r"a\\trajectory.parquet",
        "a//trajectory.parquet",
        "a/./trajectory.parquet",
    ],
)
def test_aggregate_shard_binding_rejects_unsafe_output_path(path: str) -> None:
    """輸出路徑必須是沒有父層跳脫、空層、dot segment 或反斜線的相對 POSIX 路徑。"""

    with pytest.raises(ValueError):
        replace(_valid_binding(), output_relative_path=path)


@pytest.mark.parametrize(
    "digest",
    ["A" * 64, "a" * 63, "a" * 65, "g" * 64, ""],
)
def test_aggregate_shard_binding_rejects_invalid_sha256(digest: str) -> None:
    """trajectory manifest hash 必須是 64 位小寫十六進位 SHA-256。"""

    with pytest.raises(ValueError):
        replace(_valid_binding(), trajectory_manifest_sha256=digest)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("scenario_start_index", True),
        ("scenario_stop_index", np.int64(2)),
        ("particle_count", np.int64(2)),
        ("observation_count", range(4)),
        ("event_count", False),
    ],
)
def test_aggregate_shard_binding_rejects_non_native_integer_values(
    field_name: str,
    value: object,
) -> None:
    """索引與計數只接受原生 int，拒絕 bool、NumPy 整數及 range 容器。"""

    with pytest.raises(ValueError):
        replace(_valid_binding(), **{field_name: value})


@pytest.mark.parametrize(
    "changes",
    [
        {"scenario_start_index": -1},
        {"scenario_stop_index": 0},
        {"scenario_start_index": 2},
        {"particle_count": 0},
        {"observation_count": 1},
        {"event_count": 1},
    ],
)
def test_aggregate_shard_binding_rejects_invalid_index_and_count_relations(
    changes: dict[str, int],
) -> None:
    """半開 scenario 範圍與 particle／observation／event 最小數量關係必須同時成立。"""

    with pytest.raises(ValueError):
        replace(_valid_binding(), **changes)


@pytest.mark.parametrize(
    "component_name",
    ["materials", "receptors", "arrival_times"],
)
def test_scenario_inputs_to_strata_rejects_duplicate_component_ids(
    scenario_inputs: ScenarioInputs,
    component_name: str,
) -> None:
    """component index 不可讓重複識別碼靜默覆蓋早先 record。"""

    records = getattr(scenario_inputs, component_name)
    tampered = replace(
        scenario_inputs,
        **{component_name: records + (records[0],)},
    )

    with pytest.raises(ValueError):
        scenario_inputs_to_strata(tampered, formal=False)


def test_scenario_inputs_to_strata_rejects_duplicate_scenario_id(
    scenario_inputs: ScenarioInputs,
) -> None:
    """scenario_id 是輸出列的唯一識別，重複時不可依最後一筆覆寫。"""

    tampered = _with_scenarios(
        scenario_inputs,
        scenario_inputs.scenarios + (scenario_inputs.scenarios[0],),
    )

    with pytest.raises(ValueError):
        scenario_inputs_to_strata(tampered, formal=False)


def test_scenario_inputs_to_strata_rejects_duplicate_receptor_arrival_pair(
    scenario_inputs: ScenarioInputs,
) -> None:
    """OCM dynamic record 必須以 receptor×arrival pair 唯一，不能保留歧義來源。"""

    tampered = _with_initial_conditions(
        scenario_inputs,
        scenario_inputs.initial_conditions + (scenario_inputs.initial_conditions[0],),
    )

    with pytest.raises(ValueError):
        scenario_inputs_to_strata(tampered, formal=False)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("material_id", "missing-material"),
        ("receptor_id", "missing-receptor"),
        ("arrival_time_id", "missing-arrival"),
    ],
)
def test_scenario_inputs_to_strata_rejects_missing_scenario_join(
    scenario_inputs: ScenarioInputs,
    field_name: str,
    value: str,
) -> None:
    """scenario 指向不存在的 material、receptor 或 arrival 時必須 fail fast。"""

    first = replace(scenario_inputs.scenarios[0], **{field_name: value})
    tampered = _with_scenarios(scenario_inputs, (first, *scenario_inputs.scenarios[1:]))

    with pytest.raises(ValueError):
        scenario_inputs_to_strata(tampered, formal=False)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("receptor_id", "missing-receptor"),
        ("arrival_time_id", "missing-arrival"),
    ],
)
def test_scenario_inputs_to_strata_rejects_missing_dynamic_join(
    scenario_inputs: ScenarioInputs,
    field_name: str,
    value: str,
) -> None:
    """dynamic record 指向不存在的 receptor 或 arrival 時不可被其他 pair 替代。"""

    first = replace(scenario_inputs.initial_conditions[0], **{field_name: value})
    tampered = _with_initial_conditions(
        scenario_inputs,
        (first, *scenario_inputs.initial_conditions[1:]),
    )

    with pytest.raises(ValueError):
        scenario_inputs_to_strata(tampered, formal=False)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("study_site_id", "site-other"),
        ("analysis_region_id", "region-other"),
        ("arrival_time_utc_ns", 1),
        ("settling_velocity_mps", -0.125),
        ("design_version", "design-other"),
    ],
)
def test_scenario_inputs_to_strata_rejects_scenario_join_mismatch(
    scenario_inputs: ScenarioInputs,
    field_name: str,
    value: object,
) -> None:
    """scenario 與 receptor／arrival／Behavior／inputs design 的欄位必須完全相等。"""

    first = replace(scenario_inputs.scenarios[0], **{field_name: value})
    tampered = _with_scenarios(scenario_inputs, (first, *scenario_inputs.scenarios[1:]))

    with pytest.raises(ValueError):
        scenario_inputs_to_strata(tampered, formal=False)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("study_site_id", "site-other"),
        ("analysis_region_id", "region-other"),
        ("time_utc_ns", 1),
        ("vertical_id", "vertical-other"),
    ],
)
def test_scenario_inputs_to_strata_rejects_dynamic_join_mismatch(
    scenario_inputs: ScenarioInputs,
    field_name: str,
    value: object,
) -> None:
    """pair manifest 的站點、區域、UTC 奈秒與垂向設定必須對應 joined records。"""

    first = replace(scenario_inputs.initial_conditions[0], **{field_name: value})
    tampered = _with_initial_conditions(
        scenario_inputs,
        (first, *scenario_inputs.initial_conditions[1:]),
    )

    with pytest.raises(ValueError):
        scenario_inputs_to_strata(tampered, formal=False)


def test_scenario_inputs_to_strata_rejects_formal_missing_dynamic(
    scenario_inputs: ScenarioInputs,
) -> None:
    """formal 模式不得以整組 None 的 pilot 輸入產生正式 release strata。"""

    tampered = _without_dynamic_conditions(scenario_inputs)

    with pytest.raises(ValueError):
        scenario_inputs_to_strata(tampered, formal=True)


def test_scenario_inputs_to_strata_rejects_pilot_partial_pair(
    scenario_inputs: ScenarioInputs,
) -> None:
    """pilot 一旦存在任何 dynamic pair，就必須覆蓋每個 scenario，不可只覆蓋子集。"""

    tampered = _with_initial_conditions(
        scenario_inputs,
        (scenario_inputs.initial_conditions[0],),
    )

    with pytest.raises(ValueError):
        scenario_inputs_to_strata(tampered, formal=False)
