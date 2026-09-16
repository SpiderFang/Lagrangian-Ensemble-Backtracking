"""沉底漁業用具材質統計 reducer 的純計算契約測試。

本檔只建立記憶體中的 synthetic ``Scenario``、``ParticleResult``、``Observation`` 與
``BoundaryEvent``，驗證 site×material 分組、有效成員分母、海床接觸／沉積 member 去重、
零分母、輸入驗證與一次 iterable。測試資料不是 OCM／NWW3 正式科學成果，也不把主管
「特別關注沉底漁業用具」解讀成出現率或來源先驗。
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from types import MappingProxyType

import pytest

from lagrangian_backtracking.engine import Observation, ParticleResult
from lagrangian_backtracking.models import BoundaryEvent, EventType, ParticleState, ParticleStatus
from lagrangian_backtracking.report_material_statistics import (
    FISHING_GEAR_MATERIAL_ID,
    MaterialStatistics,
    MaterialStatisticsAccumulator,
    MaterialStatisticsProduct,
    build_material_statistics,
)
from lagrangian_backtracking.scenarios import (
    BASELINE_BEHAVIORS,
    Scenario,
    validate_non_rising_behaviors,
)


def _scenario(
    site_id: str,
    material_id: str,
    *,
    label: str,
    arrival_time_utc_ns: int = 100,
) -> Scenario:
    """建立一筆只供 reducer 使用的 site×material scenario strata。"""

    return Scenario(
        scenario_id=f"scenario-{site_id}-{material_id}-{label}",
        study_site_id=site_id,
        analysis_region_id=f"region-{site_id}",
        material_id=material_id,
        receptor_id=f"receptor-{site_id}-{label}",
        arrival_time_id=f"arrival-{site_id}-{label}",
        settling_velocity_mps=-0.002,
        arrival_time_utc_ns=arrival_time_utc_ns,
        design_version="design_baseline_v2_non_rising_oca_proxy",
    )


def _event(
    scenario: Scenario,
    *,
    member_id: int,
    particle_id: str,
    event_type: EventType,
    time_utc_ns: int,
) -> BoundaryEvent:
    """建立帶有完整 identity 的 synthetic 海床事件。"""

    return BoundaryEvent(
        particle_id=particle_id,
        scenario_id=scenario.scenario_id,
        member_id=member_id,
        study_site_id=scenario.study_site_id,
        analysis_region_id=scenario.analysis_region_id,
        receptor_id=scenario.receptor_id,
        event_type=event_type,
        time_utc_ns=time_utc_ns,
        x_m=1.0,
        y_m=2.0,
        z_m=-3.0,
        fraction=0.5,
    )


def _result(
    scenario: Scenario,
    *,
    member_id: int,
    status: ParticleStatus = ParticleStatus.MAX_AGE,
    event_types: tuple[EventType, ...] = (),
    event_times: tuple[int, ...] | None = None,
) -> ParticleResult:
    """建立可供 material reducer 驗證的兩筆 backward observation 結果。"""

    particle_id = f"particle-{scenario.scenario_id}-{member_id}"
    if event_times is None:
        event_times = tuple(90 - index * 10 for index in range(len(event_types)))
    events = tuple(
        _event(
            scenario,
            member_id=member_id,
            particle_id=particle_id,
            event_type=event_type,
            time_utc_ns=time_utc_ns,
        )
        for event_type, time_utc_ns in zip(event_types, event_times, strict=True)
    )
    observations = [
        Observation(particle_id, 100, 0.0, 0.0, 0.0, -1.0, ParticleStatus.ACTIVE),
        Observation(particle_id, 70, 3.0, 1.0, 2.0, -3.0, status),
    ]
    final_state = ParticleState(
        particle_id=particle_id,
        scenario_id=scenario.scenario_id,
        member_id=member_id,
        study_site_id=scenario.study_site_id,
        analysis_region_id=scenario.analysis_region_id,
        receptor_id=scenario.receptor_id,
        x_m=1.0,
        y_m=2.0,
        z_m=-3.0,
        time_utc_ns=70,
        age_seconds=3.0,
        status=status,
    )
    return ParticleResult(
        final_state=final_state,
        observations=observations,
        events=list(events),
        step_count=1,
        minimum_clamp_count=0,
    )


def _scenarios() -> dict[str, Scenario]:
    """建立兩站兩材質的最小分層索引。"""

    values = (
        _scenario("site-a", FISHING_GEAR_MATERIAL_ID, label="gear"),
        _scenario("site-a", "material-paper", label="paper"),
        _scenario("site-b", FISHING_GEAR_MATERIAL_ID, label="gear"),
        _scenario("site-b", "material-paper", label="paper"),
    )
    return {scenario.scenario_id: scenario for scenario in values}


def _find_scenario(
    scenarios: dict[str, Scenario],
    *,
    site_id: str,
    material_id: str,
) -> Scenario:
    """依 site×material 找到 synthetic scenario，避免測試重複實作 join 條件。"""

    return next(
        scenario
        for scenario in scenarios.values()
        if scenario.study_site_id == site_id and scenario.material_id == material_id
    )


def test_builds_multi_site_material_member_counts_and_ratios() -> None:
    """多站多材質輸出應以有效 member 分母計算首次接觸與沉積比例。"""

    scenarios = _scenarios()
    gear_a = _find_scenario(
        scenarios,
        site_id="site-a",
        material_id=FISHING_GEAR_MATERIAL_ID,
    )
    paper_a = _find_scenario(
        scenarios,
        site_id="site-a",
        material_id="material-paper",
    )
    gear_b = _find_scenario(
        scenarios,
        site_id="site-b",
        material_id=FISHING_GEAR_MATERIAL_ID,
    )
    results = (
        _result(
            gear_a,
            member_id=0,
            event_types=(EventType.BED_CONTACT,),
        ),
        _result(
            gear_a,
            member_id=1,
            status=ParticleStatus.DEPOSITED,
            event_types=(EventType.DEPOSITED,),
        ),
        _result(
            paper_a,
            member_id=0,
        ),
        _result(
            gear_b,
            member_id=0,
            status=ParticleStatus.DEPOSITED,
            event_types=(EventType.BED_CONTACT, EventType.DEPOSITED),
        ),
    )

    product = build_material_statistics(
        (result for result in results),
        scenarios_by_id=scenarios,
        site_ids=("site-a", "site-b"),
        material_ids=(FISHING_GEAR_MATERIAL_ID, "material-paper"),
    )

    gear_a = product[("site-a", FISHING_GEAR_MATERIAL_ID)]
    assert gear_a.valid_member_denominator == 2
    assert gear_a.first_bed_contact_member_count == 2
    assert gear_a.first_bed_contact_fraction == 1.0
    assert gear_a.deposited_member_count == 1
    assert gear_a.deposited_fraction == 0.5
    gear_b = product.by_site_material[("site-b", FISHING_GEAR_MATERIAL_ID)]
    assert gear_b.first_bed_contact_count == gear_b.deposited_count == 1
    assert gear_b.first_bed_contact_ratio == gear_b.deposited_ratio == 1.0


def test_repeated_bed_events_count_one_member_and_are_not_required() -> None:
    """同一 member 多次 BED_CONTACT 只算一次，DEPOSITED 單事件即可作為首次接觸。"""

    scenario = _scenario("site-a", FISHING_GEAR_MATERIAL_ID, label="repeated")
    product = build_material_statistics(
        (
            _result(
                scenario,
                member_id=0,
                event_types=(EventType.BED_CONTACT, EventType.BED_CONTACT, EventType.DEPOSITED),
                event_times=(80, 90, 70),
            ),
        ),
        scenarios_by_id={scenario.scenario_id: scenario},
    )
    row = product[('site-a', FISHING_GEAR_MATERIAL_ID)]
    assert row.valid_member_denominator == 1
    assert row.first_bed_contact_member_count == 1
    assert row.deposited_member_count == 1
    # 沒有 repeated-contact event 的 member 也能正常建立有效 zero-contact row。
    no_event = build_material_statistics(
        (_result(scenario, member_id=1),),
        scenarios_by_id={scenario.scenario_id: scenario},
    )[('site-a', FISHING_GEAR_MATERIAL_ID)]
    assert no_event.valid_member_denominator == 1
    assert no_event.first_bed_contact_member_count == 0


def test_invalid_members_are_excluded_and_empty_pairs_keep_none_ratios() -> None:
    """資料／數值失敗與 pre-window 均不進有效分母，未有事件 pair 保留零分母。"""

    scenarios = _scenarios()
    gear_a = next(
        scenario
        for scenario in scenarios.values()
        if scenario.study_site_id == "site-a" and scenario.material_id == FISHING_GEAR_MATERIAL_ID
    )
    product = build_material_statistics(
        (
            _result(
                gear_a,
                member_id=0,
                status=ParticleStatus.DATA_GAP,
                event_types=(EventType.BED_CONTACT,),
            ),
            _result(
                gear_a,
                member_id=1,
                status=ParticleStatus.NUMERICAL_FAILURE,
                event_types=(EventType.BED_CONTACT,),
            ),
            _result(
                gear_a,
                member_id=2,
                status=ParticleStatus.PRE_WINDOW_DEPOSITION,
                event_types=(EventType.BED_CONTACT,),
            ),
        ),
        scenarios_by_id=scenarios,
        site_ids=("site-a", "site-b"),
        material_ids=(FISHING_GEAR_MATERIAL_ID, "material-paper"),
    )
    row = product[('site-a', FISHING_GEAR_MATERIAL_ID)]
    assert row.valid_member_denominator == 0
    assert row.first_bed_contact_member_count == 0
    assert row.deposited_member_count == 0
    assert row.first_bed_contact_fraction is None
    assert row.deposited_fraction is None
    assert len(product) == 4

    empty_product = build_material_statistics(
        (),
        scenarios_by_id=scenarios,
        site_ids=("site-a", "site-b"),
        material_ids=(FISHING_GEAR_MATERIAL_ID, "material-paper"),
    )
    assert all(record.valid_member_denominator == 0 for record in empty_product.records)
    assert all(record.first_bed_contact_fraction is None for record in empty_product.records)


def test_accumulator_is_one_pass_and_final_product_is_immutable() -> None:
    """reducer 不重跑 iterable，完成後 mapping、records 與 dataclass 都不可寫。"""

    scenario = _scenario("site-a", FISHING_GEAR_MATERIAL_ID, label="stream")
    calls = 0

    def one_pass_results():
        nonlocal calls
        calls += 1
        if calls > 1:
            raise AssertionError("results 不得被第二次遍歷")
        yield _result(scenario, member_id=0, event_types=(EventType.BED_CONTACT,))

    accumulator = MaterialStatisticsAccumulator(scenarios_by_id={scenario.scenario_id: scenario})
    accumulator.add_many(one_pass_results())
    product = accumulator.finalize()
    assert calls == 1
    assert isinstance(product.statistics, MappingProxyType)
    assert isinstance(product, MaterialStatisticsProduct)
    assert isinstance(product.records, tuple)
    with pytest.raises(FrozenInstanceError):
        product.records = ()  # type: ignore[misc]
    with pytest.raises(TypeError):
        product.statistics[('site-a', FISHING_GEAR_MATERIAL_ID)] = None  # type: ignore[index]
    with pytest.raises(ValueError, match="已關閉"):
        accumulator.add(_result(scenario, member_id=1))


def test_duplicate_member_identity_is_rejected_before_denominator_inflation() -> None:
    """同一 scenario×member 的重複輸入必須 fail-closed，不可悄悄增加有效分母。"""

    scenario = _scenario("site-a", FISHING_GEAR_MATERIAL_ID, label="duplicate")
    accumulator = MaterialStatisticsAccumulator(scenarios_by_id={scenario.scenario_id: scenario})
    result = _result(scenario, member_id=0)
    accumulator.add(result)
    with pytest.raises(ValueError, match="不可重複輸入"):
        accumulator.add(result)
    with pytest.raises(ValueError, match="已關閉"):
        accumulator.finalize()


@pytest.mark.parametrize(
    "bad_result",
    [
        object(),
        _result(
            _scenario("site-a", FISHING_GEAR_MATERIAL_ID, label="bad-event"),
            member_id=0,
        ),
    ],
)
def test_rejects_wrong_result_or_unknown_scenario(bad_result: object) -> None:
    """輸入型別與未登錄 scenario 必須在統計前被拒絕。"""

    known = _scenario("site-a", FISHING_GEAR_MATERIAL_ID, label="known")
    accumulator = MaterialStatisticsAccumulator(scenarios_by_id={known.scenario_id: known})
    with pytest.raises((TypeError, ValueError)):
        accumulator.add(bad_result)  # type: ignore[arg-type]


def test_rejects_boundary_event_fraction_outside_unit_interval() -> None:
    """事件交點比例超出步首／步末範圍時不可進入材質統計。"""

    scenario = _scenario("site-a", FISHING_GEAR_MATERIAL_ID, label="fraction")
    result = _result(scenario, member_id=0, event_types=(EventType.BED_CONTACT,))
    result.events[0] = replace(result.events[0], fraction=1.1)
    accumulator = MaterialStatisticsAccumulator(scenarios_by_id={scenario.scenario_id: scenario})
    with pytest.raises(ValueError, match="fraction"):
        accumulator.add(result)


def test_product_direct_validation_rejects_inconsistent_ratio() -> None:
    """直接建構 immutable product 時仍需驗證比例不可與 raw count 分離。"""

    with pytest.raises(ValueError, match="精確等於"):
        MaterialStatistics(
            study_site_id="site-a",
            material_id=FISHING_GEAR_MATERIAL_ID,
            valid_member_denominator=2,
            first_bed_contact_member_count=1,
            first_bed_contact_fraction=0.0,
            deposited_member_count=0,
            deposited_fraction=0.0,
        )

    with pytest.raises(ValueError, match="子集合"):
        MaterialStatistics(
            study_site_id="site-a",
            material_id=FISHING_GEAR_MATERIAL_ID,
            valid_member_denominator=2,
            first_bed_contact_member_count=0,
            first_bed_contact_fraction=0.0,
            deposited_member_count=1,
            deposited_fraction=0.5,
        )


def test_baseline_material_contract_keeps_ten_negative_speed_proxies() -> None:
    """報告統計修正不得改變十類基線、速度格點或漁具代理速度。"""

    validate_non_rising_behaviors(BASELINE_BEHAVIORS)
    assert len(BASELINE_BEHAVIORS) == 10
    assert tuple(item.settling_velocity_mps for item in BASELINE_BEHAVIORS) == (
        -0.0001,
        -0.0002,
        -0.0005,
        -0.001,
        -0.002,
        -0.005,
        -0.010,
        -0.020,
        -0.050,
        -0.100,
    )
    fishing_gear = next(
        item for item in BASELINE_BEHAVIORS if item.material_id == FISHING_GEAR_MATERIAL_ID
    )
    assert fishing_gear.settling_velocity_mps == -0.002
    assert all(item.settling_velocity_mps < 0.0 for item in BASELINE_BEHAVIORS)
