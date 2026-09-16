"""驗證正式有效成員分母政策與事件 numerator 的一致性。

這裡刻意建立一條最終狀態為 ``DATA_GAP`` 的結果，但在失敗前已發生
local 語意邊界事件、海床接觸與其他站點 enter；獨立 valid control 再覆蓋 outer
語意。測試確認這些事件仍先通過
完整的 semantic、幾何、弧長與旅行年齡驗證，卻不會進入條件式來源足跡的 numerator；
只有 failure grid 與 outcome 保留該失敗成員。另一個獨立 valid fixture 確認同一套
事件仍會被正常計數；固定日曆研究窗前已沉底的 pre-window member 則只保留 outcome，
不進有效分母或任何來源 numerator。所有資料都是公尺制與 UTC 奈秒的人工工程 fixture，不代表
真實 OCM／NWW3 科學成果，也不代表絕對來源機率。
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from test_event_aggregation_core import (
    _SHARED_SEGMENT_ID,
    _aggregate,
    _boundary_event,
    _particle_result,
    _scenario,
)

from lagrangian_backtracking.engine import Observation, ParticleResult
from lagrangian_backtracking.event_aggregation import (
    BoundaryAggregateKey,
    CrossSiteAggregateKey,
    ReceptorAggregateKey,
    SourceReceptorAggregateKey,
)
from lagrangian_backtracking.models import EventType, ParticleState, ParticleStatus


def _invalid_result_with_prior_events():
    """建立含四類來源事件、但最終以 DATA_GAP 結束的完整合法結果。

    事件分別提供 local semantic、BED_CONTACT、other-site enter 及 DATA_GAP
    terminal。現有事件契約把 ``FLOW_DOMAIN_OPEN_EXIT`` 定義為 terminal，不能在
    同一結果中再放置 DATA_GAP terminal；outer semantic 由獨立 valid control
    fixture 覆蓋。所有事件時間皆落在 final state 與 arrival 之間，因此聚合器必須
    真正完成所有事件欄位驗證，不能因結果 invalid 就提早跳過。
    """

    scenario = _scenario()
    particle_id = "particle-invalid-with-prior-events"
    local_event = _boundary_event(
        scenario,
        particle_id,
        EventType.LOCAL_DOMAIN_FIRST_EXIT,
        5.0,
        x_m=0.25,
        y_m=0.25,
        boundary_segment_id=_SHARED_SEGMENT_ID,
        boundary_s_m=0.5,
    )
    bed_event = _boundary_event(
        scenario,
        particle_id,
        EventType.BED_CONTACT,
        10.0,
        x_m=0.25,
        y_m=0.25,
    )
    cross_event = _boundary_event(
        scenario,
        particle_id,
        EventType.OTHER_SITE_LOCAL_DOMAIN_ENTER,
        15.0,
        x_m=0.25,
        y_m=0.25,
        related_study_site_id="site-b",
    )
    failure_event = _boundary_event(
        scenario,
        particle_id,
        EventType.DATA_GAP,
        20.0,
        x_m=2.0,
        y_m=2.0,
    )
    return scenario, _particle_result(
        scenario,
        particle_id,
        ParticleStatus.DATA_GAP,
        final_age_seconds=20.0,
        final_x_m=2.0,
        final_y_m=2.0,
        events=(local_event, bed_event, cross_event, failure_event),
    )


def test_invalid_member_keeps_failure_diagnostic_but_excludes_all_source_numerators() -> None:
    """最終 DATA_GAP 成員的來源 numerator 應全為零，failure/outcome 應為一。

    valid denominator policy 是依最終狀態定義的成員母體，而不是依事件發生時間
    定義。即使失敗前已穿越 local、接觸海床或進入其他站點，這些事件都不能
    與 valid denominator 混合；DATA_GAP failure grid 則是例外診斷，必須精確保留。
    """

    scenario, invalid_result = _invalid_result_with_prior_events()
    aggregate = _aggregate((invalid_result,), (scenario,))

    counts = aggregate.site_grid_counts["site-a"]
    assert aggregate.total_member_count_by_site == {"site-a": 1, "site-b": 0}
    assert aggregate.valid_member_denominator_by_site == {"site-a": 0, "site-b": 0}
    assert aggregate.valid_member_denominator_by_receptor == {
        ReceptorAggregateKey("site-a", "receptor-a"): 0,
    }
    assert aggregate.outcome_count_by_site["site-a"][ParticleStatus.DATA_GAP.value] == 1
    assert int(counts.data_gap_failure_count.sum()) == 1
    assert int(counts.numerical_failure_count.sum()) == 0

    # 失敗前的 local semantic 與海床接觸都已驗證，但不可成為來源 numerator。
    assert int(counts.local_first_exit_count.sum()) == 0
    assert int(counts.outer_first_exit_count.sum()) == 0
    assert int(counts.bed_first_contact_count.sum()) == 0
    assert int(counts.bed_repeated_contact_count.sum()) == 0
    for boundary_kind in ("local", "outer"):
        boundary_key = BoundaryAggregateKey(
            "site-a",
            boundary_kind,
            _SHARED_SEGMENT_ID,
        )
        source_key = SourceReceptorAggregateKey(
            "site-a",
            "receptor-a",
            boundary_kind,
            _SHARED_SEGMENT_ID,
        )
        assert int(aggregate.boundary_arclength_raw_count[boundary_key].sum()) == 0
        assert int(aggregate.boundary_travel_age_histogram[boundary_key].sum()) == 0
        assert aggregate.source_receptor_raw_count[source_key] == 0
        assert int(aggregate.source_receptor_travel_age_histogram[source_key].sum()) == 0
    assert all(
        count == 0
        for count in aggregate.cross_site_unique_member_count.values()
    )


def test_valid_member_with_same_event_categories_still_counts_all_numerators() -> None:
    """合法 MAX_AGE 成員仍可同時貢獻 local、outer、bed 與跨站 numerator。"""

    scenario = _scenario()
    particle_id = "particle-valid-control"
    flow_event = _boundary_event(
        scenario,
        particle_id,
        EventType.FLOW_DOMAIN_OPEN_EXIT,
        20.0,
        x_m=2.0,
        y_m=2.0,
        boundary_segment_id=_SHARED_SEGMENT_ID,
        boundary_s_m=2.5,
        attributes={"also_local_domain_first_exit": True},
    )
    bed_event = _boundary_event(
        scenario,
        particle_id,
        EventType.BED_CONTACT,
        10.0,
        x_m=0.25,
        y_m=0.25,
    )
    cross_event = _boundary_event(
        scenario,
        particle_id,
        EventType.OTHER_SITE_LOCAL_DOMAIN_ENTER,
        15.0,
        x_m=0.25,
        y_m=0.25,
        related_study_site_id="site-b",
    )
    result = _particle_result(
        scenario,
        particle_id,
        ParticleStatus.FLOW_DOMAIN_EXIT,
        final_age_seconds=20.0,
        final_x_m=2.0,
        final_y_m=2.0,
        events=(flow_event, bed_event, cross_event),
    )
    aggregate = _aggregate((result,), (scenario,))

    counts = aggregate.site_grid_counts["site-a"]
    assert aggregate.valid_member_denominator_by_site["site-a"] == 1
    assert int(counts.local_first_exit_count.sum()) == 1
    assert int(counts.outer_first_exit_count.sum()) == 1
    assert int(counts.bed_first_contact_count.sum()) == 1
    assert int(counts.bed_repeated_contact_count.sum()) == 0
    assert aggregate.cross_site_unique_member_count[
        CrossSiteAggregateKey("site-a", "site-b")
    ] == 1


def test_pre_window_member_keeps_outcome_but_has_no_valid_denominator_or_source_numerator() -> None:
    """單筆 age=0 pre-window terminal 結果保留 outcome，且不進任何有效來源統計。"""

    scenario = _scenario()
    particle_id = "particle-pre-window"
    initial = ParticleState(
        particle_id=particle_id,
        scenario_id=scenario.scenario_id,
        member_id=0,
        study_site_id=scenario.study_site_id,
        analysis_region_id=scenario.analysis_region_id,
        receptor_id=scenario.receptor_id,
        x_m=0.25,
        y_m=0.25,
        z_m=-1.0,
        time_utc_ns=scenario.arrival_time_utc_ns,
        age_seconds=0.0,
        status=ParticleStatus.PRE_WINDOW_DEPOSITION,
    )
    terminal = replace(
        _boundary_event(
            scenario,
            particle_id,
            EventType.PRE_WINDOW_DEPOSITION,
            0.0,
            x_m=initial.x_m,
            y_m=initial.y_m,
        ),
        fraction=0.0,
    )
    result = ParticleResult(
        final_state=initial,
        observations=[
            Observation(
                particle_id,
                scenario.arrival_time_utc_ns,
                0.0,
                initial.x_m,
                initial.y_m,
                initial.z_m,
                ParticleStatus.PRE_WINDOW_DEPOSITION,
            )
        ],
        events=[terminal],
        step_count=0,
        minimum_clamp_count=0,
    )

    aggregate = _aggregate((result,), (scenario,))

    counts = aggregate.site_grid_counts["site-a"]
    assert aggregate.total_member_count_by_site == {"site-a": 1, "site-b": 0}
    assert aggregate.valid_member_denominator_by_site == {"site-a": 0, "site-b": 0}
    assert aggregate.valid_member_denominator_by_receptor[
        ReceptorAggregateKey("site-a", "receptor-a")
    ] == 0
    assert aggregate.outcome_count_by_site["site-a"][
        ParticleStatus.PRE_WINDOW_DEPOSITION.value
    ] == 1
    assert int(counts.data_gap_failure_count.sum()) == 0
    assert int(counts.numerical_failure_count.sum()) == 0
    assert int(counts.local_first_exit_count.sum()) == 0
    assert int(counts.outer_first_exit_count.sum()) == 0
    assert int(counts.bed_first_contact_count.sum()) == 0
    assert int(counts.bed_repeated_contact_count.sum()) == 0
    assert all(
        int(values.sum()) == 0
        for values in aggregate.boundary_arclength_raw_count.values()
    )
    assert all(count == 0 for count in aggregate.source_receptor_raw_count.values())
    assert all(
        int(values.sum()) == 0
        for values in aggregate.source_receptor_travel_age_histogram.values()
    )
    assert all(count == 0 for count in aggregate.cross_site_unique_member_count.values())


def test_invalid_member_event_is_still_rejected_when_semantic_geometry_is_bad() -> None:
    """invalid 成員仍須拒絕越界 semantic 弧長，不能用 early continue 繞過驗證。"""

    scenario, invalid_result = _invalid_result_with_prior_events()
    bad_flow = replace(
        invalid_result.events[0],
        boundary_s_m=99.0,
    )
    bad_result = replace(
        invalid_result,
        events=[bad_flow, *invalid_result.events[1:]],
    )

    with pytest.raises(ValueError, match="boundary_s_m"):
        _aggregate((bad_result,), (scenario,))
