"""事件聚合核心六個 happy-path 的最小契約測試。

本檔使用兩個 2×2 公尺站點、同一條跨站共享邊界段，以及短尾弧長 bin，驗證
初始化、邊界事件、海床接觸、失敗狀態與跨站事件的公開聚合行為。所有座標均是
公尺制人工 fixture，UTC 時間以奈秒表示；這些資料只用來固定資料結構與分箱
邊界，不代表正式海洋資料或任何絕對來源機率。
"""

from __future__ import annotations

import numpy as np
import pytest

from lagrangian_backtracking.aggregate_spec import (
    AggregateSpec,
    SiteBoundarySegments,
    SiteGridSpec,
    SiteMetricCRSSpec,
)
from lagrangian_backtracking.engine import Observation, ParticleResult
from lagrangian_backtracking.event_aggregation import (
    BoundaryAggregateKey,
    CrossSiteAggregateKey,
    ReceptorAggregateKey,
    SourceReceptorAggregateKey,
    aggregate_result_events,
    initialize_event_aggregate,
)
from lagrangian_backtracking.models import (
    BoundaryEvent,
    EventType,
    ParticleState,
    ParticleStatus,
)
from lagrangian_backtracking.scenarios import Scenario

_ARRIVAL_TIME_UTC_NS = 1_000_000_000_000
_AGE_BIN_EDGES_SECONDS = np.array([0.0, 10.0, 20.0], dtype=np.float64)
_SHARED_SEGMENT_ID = "shared-segment"


def _aggregate_spec() -> AggregateSpec:
    """建立兩站最小彙整規格，固定公尺制網格、投影與共享邊界資料契約。

    兩站的矩形各為 2×2 公尺，cell 尺寸為 1 公尺，因此事件網格應為
    ``(y_cell, x_cell) = (2, 2)``。兩站的 local 與 outer 都引用同一條長 2.5
    公尺的邊界段；以 1 公尺分箱後會得到 ``[0, 1, 2, 2.5]``，用來覆蓋最後
    短尾 s-bin 與跨站／跨角色 key 分離。age 軸明確使用 0、10、20 秒。
    """

    site_grids = {
        "site-a": SiteGridSpec(
            x_min_m=0.0,
            x_max_m=2.0,
            y_min_m=0.0,
            y_max_m=2.0,
        ),
        "site-b": SiteGridSpec(
            x_min_m=10.0,
            x_max_m=12.0,
            y_min_m=10.0,
            y_max_m=12.0,
        ),
    }
    site_metric_crs = {
        "site-a": SiteMetricCRSSpec(
            projection_method="azimuthal_equidistant_wgs84",
            center_lon_deg=121.0,
            center_lat_deg=25.0,
            linear_unit="m",
            axis_order="x_east_y_north",
        ),
        "site-b": SiteMetricCRSSpec(
            projection_method="azimuthal_equidistant_wgs84",
            center_lon_deg=121.1,
            center_lat_deg=25.1,
            linear_unit="m",
            axis_order="x_east_y_north",
        ),
    }
    site_boundary_segments = {
        site_id: SiteBoundarySegments(
            local_segment_ids=(_SHARED_SEGMENT_ID,),
            outer_segment_ids=(_SHARED_SEGMENT_ID,),
        )
        for site_id in site_grids
    }
    return AggregateSpec(
        schema_version="1.0.0",
        run_id="event-aggregation-core-fixture",
        grid_cell_size_m=1.0,
        site_grids=site_grids,
        site_metric_crs=site_metric_crs,
        boundary_bin_size_m=1.0,
        boundary_segment_lengths_m={_SHARED_SEGMENT_ID: 2.5},
        site_boundary_segment_ids=site_boundary_segments,
        kde_bandwidths_m=(1.0, 2.0, 3.0),
        hdr_levels=(0.5, 0.75, 0.9),
        age_bin_edges_seconds=(0.0, 10.0, 20.0),
        bootstrap_replicates=1,
        bootstrap_confidence_level=0.95,
        bootstrap_seed=0,
        denominator_policy="exclude_data_gap_and_numerical_failure_v1",
        source_sha256="0" * 64,
        canonical_sha256="1" * 64,
    )


def _scenario(
    scenario_id: str = "scenario-a",
    *,
    site_id: str = "site-a",
    receptor_id: str = "receptor-a",
) -> Scenario:
    """建立與粒子結果 identity 對齊的最小 Scenario。

    Scenario 只保存聚合器需要核對的站點、受體、情境與到達 UTC；速度與材料
    欄位仍填入合法的研究設計值，但本檔不執行平流計算。不同站點使用不同
    ``scenario_id``，以便測試站點／受體分母與跨站拓撲。
    """

    return Scenario(
        scenario_id=scenario_id,
        study_site_id=site_id,
        analysis_region_id="region-fixture",
        material_id="material-fixture",
        receptor_id=receptor_id,
        arrival_time_id="arrival-fixture",
        settling_velocity_mps=-0.001,
        arrival_time_utc_ns=_ARRIVAL_TIME_UTC_NS,
        design_version="event-aggregation-test-v1",
    )


def _observation(
    scenario: Scenario,
    particle_id: str,
    age_seconds: float,
    *,
    x_m: float,
    y_m: float,
    status: ParticleStatus,
) -> Observation:
    """依 Scenario 到達時刻建立一筆合法的逆向觀測。

    回溯年齡以秒表示，UTC 則由到達時間減去相同秒數換算為奈秒；這使 fixture
    同時滿足聚合器要求的「年齡嚴格增加」與「UTC 嚴格倒退」，並保留最後觀測
    與 ParticleState 完全一致的終止狀態。
    """

    time_utc_ns = scenario.arrival_time_utc_ns - int(round(age_seconds * 1.0e9))
    return Observation(
        particle_id=particle_id,
        time_utc_ns=time_utc_ns,
        age_seconds=age_seconds,
        x_m=x_m,
        y_m=y_m,
        z_m=-1.0,
        status=status,
    )


def _boundary_event(
    scenario: Scenario,
    particle_id: str,
    event_type: EventType,
    age_seconds: float,
    *,
    x_m: float,
    y_m: float,
    boundary_segment_id: str | None = None,
    boundary_s_m: float | None = None,
    related_study_site_id: str | None = None,
    attributes: dict[str, bool | float | int | str] | None = None,
) -> BoundaryEvent:
    """建立與粒子 identity、到達時間及事件旅行年齡一致的 BoundaryEvent。

    ``age_seconds`` 是由事件 UTC 與 Scenario 到達 UTC 推導的回溯年齡，因此可
    精確測試 age bin；邊界事件額外提供共享段與弧長，其他站點事件則提供
    ``related_study_site_id``。未參與該情境的可選欄位保持 ``None``，避免把
    非必要 metadata 混入各測試的核心語意。
    """

    time_utc_ns = scenario.arrival_time_utc_ns - int(round(age_seconds * 1.0e9))
    return BoundaryEvent(
        particle_id=particle_id,
        scenario_id=scenario.scenario_id,
        member_id=0,
        study_site_id=scenario.study_site_id,
        analysis_region_id=scenario.analysis_region_id,
        receptor_id=scenario.receptor_id,
        event_type=event_type,
        time_utc_ns=time_utc_ns,
        x_m=x_m,
        y_m=y_m,
        z_m=-1.0,
        fraction=1.0,
        related_study_site_id=related_study_site_id,
        boundary_segment_id=boundary_segment_id,
        boundary_s_m=boundary_s_m,
        attributes={} if attributes is None else attributes,
    )


def _particle_result(
    scenario: Scenario,
    particle_id: str,
    final_status: ParticleStatus,
    *,
    final_age_seconds: float,
    final_x_m: float,
    final_y_m: float,
    events: tuple[BoundaryEvent, ...],
) -> ParticleResult:
    """把最小初始／終止觀測與事件包裝為可被公開聚合入口驗證的結果。

    聚合器要求至少一筆從 age 0 開始的 observation，且最後 observation 必須
    與 final_state 完全相同；因此 helper 固定產生兩筆觀測，並以同一個 final
    status 建立 terminal state。所有 fixture 使用 member 0，事件 identity
    也由此一致，讓測試專注在事件彙整而非引擎執行細節。
    """

    observations = [
        _observation(
            scenario,
            particle_id,
            0.0,
            x_m=final_x_m,
            y_m=final_y_m,
            status=ParticleStatus.ACTIVE,
        ),
        _observation(
            scenario,
            particle_id,
            final_age_seconds,
            x_m=final_x_m,
            y_m=final_y_m,
            status=final_status,
        ),
    ]
    final_observation = observations[-1]
    final_state = ParticleState(
        particle_id=particle_id,
        scenario_id=scenario.scenario_id,
        member_id=0,
        study_site_id=scenario.study_site_id,
        analysis_region_id=scenario.analysis_region_id,
        receptor_id=scenario.receptor_id,
        x_m=final_x_m,
        y_m=final_y_m,
        z_m=final_observation.z_m,
        time_utc_ns=final_observation.time_utc_ns,
        age_seconds=final_age_seconds,
        status=final_status,
    )
    return ParticleResult(
        final_state=final_state,
        observations=observations,
        events=list(events),
        step_count=0,
        minimum_clamp_count=0,
    )


def _aggregate(
    results: tuple[ParticleResult, ...],
    scenarios: tuple[Scenario, ...],
):
    """以同一份兩站規格呼叫公開事件聚合入口，集中固定 age 軸契約。"""

    return aggregate_result_events(
        results,
        scenarios_by_id={scenario.scenario_id: scenario for scenario in scenarios},
        spec=_aggregate_spec(),
        age_bin_edges_seconds=_AGE_BIN_EDGES_SECONDS,
    )


def test_initialize_event_aggregate_builds_readonly_two_site_topology() -> None:
    """初始化應建立 2×2 網格、0/10/20 age 軸及分離的共享邊界角色 key。"""

    aggregate = initialize_event_aggregate(
        _aggregate_spec(),
        age_bin_edges_seconds=_AGE_BIN_EDGES_SECONDS,
    )

    assert np.array_equal(aggregate.age_bin_edges_seconds, [0.0, 10.0, 20.0])
    assert aggregate.age_bin_edges_seconds.flags.writeable is False
    shapes_by_site = {
        site_id: counts.local_first_exit_count.shape
        for site_id, counts in aggregate.site_grid_counts.items()
    }
    assert shapes_by_site == {
        "site-a": (2, 2),
        "site-b": (2, 2),
    }

    expected_boundary_keys = {
        BoundaryAggregateKey(site_id, kind, _SHARED_SEGMENT_ID)
        for site_id in ("site-a", "site-b")
        for kind in ("local", "outer")
    }
    assert set(aggregate.boundary_bin_edges_m) == expected_boundary_keys
    assert len(aggregate.boundary_bin_edges_m) == 4
    assert np.array_equal(
        aggregate.boundary_bin_edges_m[
            BoundaryAggregateKey("site-a", "local", _SHARED_SEGMENT_ID)
        ],
        [0.0, 1.0, 2.0, 2.5],
    )
    assert aggregate.boundary_arclength_raw_count[
        BoundaryAggregateKey("site-a", "local", _SHARED_SEGMENT_ID)
    ].shape == (3,)
    assert aggregate.boundary_travel_age_histogram[
        BoundaryAggregateKey("site-a", "outer", _SHARED_SEGMENT_ID)
    ].shape == (3, 2)
    assert aggregate.site_grid_counts["site-a"].local_first_exit_count.flags.writeable is False
    assert aggregate.boundary_arclength_raw_count[
        BoundaryAggregateKey("site-a", "local", _SHARED_SEGMENT_ID)
    ].flags.writeable is False
    with pytest.raises(ValueError):
        aggregate.age_bin_edges_seconds[0] = 99.0
    with pytest.raises(TypeError):
        aggregate.boundary_bin_edges_m[
            BoundaryAggregateKey("site-a", "local", _SHARED_SEGMENT_ID)
        ] = np.array(
            [0.0, 1.0],
            dtype=np.float64,
        )


def test_flow_domain_open_exit_counts_local_outer_and_final_bins() -> None:
    """FLOW_DOMAIN_OPEN_EXIT 同時具 local 語意時，兩角色與兩類直方圖都應加一。"""

    scenario = _scenario()
    particle_id = "particle-flow"
    event = _boundary_event(
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
    aggregate = _aggregate(
        (
            _particle_result(
                scenario,
                particle_id,
                ParticleStatus.FLOW_DOMAIN_EXIT,
                final_age_seconds=20.0,
                final_x_m=2.0,
                final_y_m=2.0,
                events=(event,),
            ),
        ),
        (scenario,),
    )

    local_grid = aggregate.site_grid_counts["site-a"].local_first_exit_count
    outer_grid = aggregate.site_grid_counts["site-a"].outer_first_exit_count
    assert local_grid[1, 1] == 1
    assert outer_grid[1, 1] == 1
    assert int(local_grid.sum()) == 1
    assert int(outer_grid.sum()) == 1
    for boundary_kind in ("local", "outer"):
        boundary_key = BoundaryAggregateKey("site-a", boundary_kind, _SHARED_SEGMENT_ID)
        source_key = SourceReceptorAggregateKey(
            "site-a",
            "receptor-a",
            boundary_kind,
            _SHARED_SEGMENT_ID,
        )
        assert aggregate.boundary_arclength_raw_count[boundary_key].tolist() == [0, 0, 1]
        assert aggregate.boundary_travel_age_histogram[boundary_key].tolist() == [[0, 0], [0, 0], [0, 1]]
        assert aggregate.source_receptor_raw_count[source_key] == 1
        assert aggregate.source_receptor_travel_age_histogram[source_key].tolist() == [0, 1]


def test_local_first_exit_then_max_age_counts_valid_member() -> None:
    """先有 local 首次離開、後以 MAX_AGE 終止時，local 計數與 valid 分母都應保留。"""

    scenario = _scenario()
    particle_id = "particle-local-max-age"
    local_event = _boundary_event(
        scenario,
        particle_id,
        EventType.LOCAL_DOMAIN_FIRST_EXIT,
        10.0,
        x_m=0.25,
        y_m=0.25,
        boundary_segment_id=_SHARED_SEGMENT_ID,
        boundary_s_m=0.0,
    )
    max_age_event = _boundary_event(
        scenario,
        particle_id,
        EventType.MAX_AGE,
        20.0,
        x_m=0.25,
        y_m=0.25,
    )
    aggregate = _aggregate(
        (
            _particle_result(
                scenario,
                particle_id,
                ParticleStatus.MAX_AGE,
                final_age_seconds=20.0,
                final_x_m=0.25,
                final_y_m=0.25,
                events=(local_event, max_age_event),
            ),
        ),
        (scenario,),
    )

    assert aggregate.site_grid_counts["site-a"].local_first_exit_count.tolist() == [[1, 0], [0, 0]]
    assert aggregate.total_member_count_by_site["site-a"] == 1
    assert aggregate.valid_member_denominator_by_site["site-a"] == 1
    assert aggregate.outcome_count_by_site["site-a"][ParticleStatus.MAX_AGE.value] == 1
    assert aggregate.valid_member_denominator_by_receptor[
        ReceptorAggregateKey("site-a", "receptor-a")
    ] == 1


def test_bed_contact_and_deposited_use_smallest_backtrack_age_as_first() -> None:
    """BED_CONTACT 與 DEPOSITED 應依最小回溯年齡排序，而非依輸入事件順序分 first。"""

    scenario = _scenario()
    particle_id = "particle-bed"
    deposited_event = _boundary_event(
        scenario,
        particle_id,
        EventType.DEPOSITED,
        15.0,
        x_m=1.75,
        y_m=1.75,
    )
    earlier_bed_event = _boundary_event(
        scenario,
        particle_id,
        EventType.BED_CONTACT,
        5.0,
        x_m=0.25,
        y_m=0.25,
    )
    aggregate = _aggregate(
        (
            _particle_result(
                scenario,
                particle_id,
                ParticleStatus.DEPOSITED,
                final_age_seconds=15.0,
                final_x_m=1.75,
                final_y_m=1.75,
                events=(deposited_event, earlier_bed_event),
            ),
        ),
        (scenario,),
    )

    assert aggregate.site_grid_counts["site-a"].bed_first_contact_count.tolist() == [[1, 0], [0, 0]]
    assert aggregate.site_grid_counts["site-a"].bed_repeated_contact_count.tolist() == [[0, 0], [0, 1]]


def test_failure_outcomes_update_total_failure_grid_and_valid_denominators() -> None:
    """DATA_GAP、NUMERICAL_FAILURE 與 MAX_AGE 應分別保留 outcome、失敗格網與有效分母。"""

    scenario_a = _scenario()
    scenario_b = _scenario("scenario-b", site_id="site-b", receptor_id="receptor-b")
    gap_id = "particle-gap"
    numerical_id = "particle-numerical"
    max_age_id = "particle-max-age"
    gap_event = _boundary_event(
        scenario_a,
        gap_id,
        EventType.DATA_GAP,
        5.0,
        x_m=2.0,
        y_m=2.0,
    )
    numerical_event = _boundary_event(
        scenario_a,
        numerical_id,
        EventType.NUMERICAL_FAILURE,
        6.0,
        x_m=0.25,
        y_m=0.25,
    )
    max_age_event = _boundary_event(
        scenario_b,
        max_age_id,
        EventType.MAX_AGE,
        20.0,
        x_m=12.0,
        y_m=12.0,
    )
    aggregate = _aggregate(
        (
            _particle_result(
                scenario_a,
                gap_id,
                ParticleStatus.DATA_GAP,
                final_age_seconds=5.0,
                final_x_m=2.0,
                final_y_m=2.0,
                events=(gap_event,),
            ),
            _particle_result(
                scenario_a,
                numerical_id,
                ParticleStatus.NUMERICAL_FAILURE,
                final_age_seconds=6.0,
                final_x_m=0.25,
                final_y_m=0.25,
                events=(numerical_event,),
            ),
            _particle_result(
                scenario_b,
                max_age_id,
                ParticleStatus.MAX_AGE,
                final_age_seconds=20.0,
                final_x_m=12.0,
                final_y_m=12.0,
                events=(max_age_event,),
            ),
        ),
        (scenario_a, scenario_b),
    )

    assert aggregate.input_particle_count == 3
    assert aggregate.total_member_count_by_site == {"site-a": 2, "site-b": 1}
    assert aggregate.valid_member_denominator_by_site == {"site-a": 0, "site-b": 1}
    assert aggregate.outcome_count_by_site["site-a"][ParticleStatus.DATA_GAP.value] == 1
    assert aggregate.outcome_count_by_site["site-a"][ParticleStatus.NUMERICAL_FAILURE.value] == 1
    assert aggregate.outcome_count_by_site["site-b"][ParticleStatus.MAX_AGE.value] == 1
    assert aggregate.site_grid_counts["site-a"].data_gap_failure_count.tolist() == [[0, 0], [0, 1]]
    assert aggregate.site_grid_counts["site-a"].numerical_failure_count.tolist() == [[1, 0], [0, 0]]
    assert aggregate.valid_member_denominator_by_receptor == {
        ReceptorAggregateKey("site-a", "receptor-a"): 0,
        ReceptorAggregateKey("site-b", "receptor-b"): 1,
    }


def test_repeated_other_site_enter_is_unique_exit_is_ignored_and_pairs_exist() -> None:
    """重複其他站點 enter 只計一次、exit 不計數，且兩個有序跨站 key 都保留。"""

    scenario = _scenario()
    particle_id = "particle-cross-site"
    first_enter = _boundary_event(
        scenario,
        particle_id,
        EventType.OTHER_SITE_LOCAL_DOMAIN_ENTER,
        3.0,
        x_m=0.25,
        y_m=0.25,
        related_study_site_id="site-b",
    )
    repeated_enter = _boundary_event(
        scenario,
        particle_id,
        EventType.OTHER_SITE_LOCAL_DOMAIN_ENTER,
        4.0,
        x_m=1.25,
        y_m=1.25,
        related_study_site_id="site-b",
    )
    exit_event = _boundary_event(
        scenario,
        particle_id,
        EventType.OTHER_SITE_LOCAL_DOMAIN_EXIT,
        5.0,
        x_m=1.75,
        y_m=1.75,
        related_study_site_id="site-b",
    )
    terminal_event = _boundary_event(
        scenario,
        particle_id,
        EventType.MAX_AGE,
        10.0,
        x_m=1.75,
        y_m=1.75,
    )
    aggregate = _aggregate(
        (
            _particle_result(
                scenario,
                particle_id,
                ParticleStatus.MAX_AGE,
                final_age_seconds=10.0,
                final_x_m=1.75,
                final_y_m=1.75,
                events=(first_enter, repeated_enter, exit_event, terminal_event),
            ),
        ),
        (scenario,),
    )

    expected_pairs = {
        CrossSiteAggregateKey("site-a", "site-b"),
        CrossSiteAggregateKey("site-b", "site-a"),
    }
    assert set(aggregate.cross_site_unique_member_count) == expected_pairs
    assert aggregate.cross_site_unique_member_count[CrossSiteAggregateKey("site-a", "site-b")] == 1
    assert aggregate.cross_site_unique_member_count[CrossSiteAggregateKey("site-b", "site-a")] == 0
