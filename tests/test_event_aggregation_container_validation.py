"""事件聚合容器跨欄位守恆關係的 fail-closed 測試。

本檔先以核心測試模組提供的 fixture/helper 建立一個合法的
``EventAggregateChunk``，再透過 ``dataclasses.replace`` 搭配 mapping 與 NumPy
陣列的防禦性複製，逐一破壞一項關係。測試中的座標是公尺制、旅行年齡是秒、
事件時間是 UTC 奈秒整數；各種計數代表粒子或事件數，不代表物理零值，也不把
條件式來源足跡解讀為絕對來源機率。每個案例都必須在容器封存時回傳
``ValueError``，避免不一致的分母、網格或直方圖進入正式結果。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest
from test_event_aggregation_core import (
    _aggregate,
    _boundary_event,
    _particle_result,
    _scenario,
)

from lagrangian_backtracking.event_aggregation import (
    BoundaryAggregateKey,
    CrossSiteAggregateKey,
    EventAggregateChunk,
    SiteEventGridCounts,
    SourceReceptorAggregateKey,
)
from lagrangian_backtracking.models import EventType, ParticleStatus

_SHARED_SEGMENT_ID = "shared-segment"


def _legal_aggregate() -> EventAggregateChunk:
    """建立含站內事件、失敗結果與跨站 enter 的合法聚合容器。

    site-a 有一筆 local 首次離開、一筆同時具 local/outer 語意的 flow exit、
    一筆 DATA_GAP 與一筆 NUMERICAL_FAILURE；site-b 有一筆 local 結果。如此
    ``input_particle_count`` 等於兩站 total，valid 分母排除兩類失敗，local/outer
    網格與 boundary raw count、source-receptor age histogram 與跨站 unique count
    都具有非零資料，任何單一 tamper 都能被明確辨識。兩站的來源受體與邊界段
    仍沿用核心測試的 2×2 公尺 fixture，不代表正式海洋資料。
    """

    scenario_a = _scenario()
    scenario_b = _scenario("scenario-b", site_id="site-b", receptor_id="receptor-b")

    local_id = "particle-local-cross"
    local_event = _boundary_event(
        scenario_a,
        local_id,
        EventType.LOCAL_DOMAIN_FIRST_EXIT,
        5.0,
        x_m=0.25,
        y_m=0.25,
        boundary_segment_id=_SHARED_SEGMENT_ID,
        boundary_s_m=0.5,
    )
    cross_event = _boundary_event(
        scenario_a,
        local_id,
        EventType.OTHER_SITE_LOCAL_DOMAIN_ENTER,
        3.0,
        x_m=0.25,
        y_m=0.25,
        related_study_site_id="site-b",
    )
    local_terminal = _boundary_event(
        scenario_a,
        local_id,
        EventType.MAX_AGE,
        10.0,
        x_m=0.25,
        y_m=0.25,
    )

    flow_id = "particle-flow"
    flow_event = _boundary_event(
        scenario_a,
        flow_id,
        EventType.FLOW_DOMAIN_OPEN_EXIT,
        20.0,
        x_m=2.0,
        y_m=2.0,
        boundary_segment_id=_SHARED_SEGMENT_ID,
        boundary_s_m=2.5,
        attributes={"also_local_domain_first_exit": True},
    )

    gap_id = "particle-gap"
    gap_event = _boundary_event(
        scenario_a,
        gap_id,
        EventType.DATA_GAP,
        5.0,
        x_m=2.0,
        y_m=2.0,
    )

    numerical_id = "particle-numerical"
    numerical_event = _boundary_event(
        scenario_a,
        numerical_id,
        EventType.NUMERICAL_FAILURE,
        6.0,
        x_m=0.25,
        y_m=0.25,
    )

    site_b_id = "particle-site-b"
    site_b_event = _boundary_event(
        scenario_b,
        site_b_id,
        EventType.LOCAL_DOMAIN_FIRST_EXIT,
        10.0,
        x_m=10.25,
        y_m=10.25,
        boundary_segment_id=_SHARED_SEGMENT_ID,
        boundary_s_m=1.5,
    )
    site_b_terminal = _boundary_event(
        scenario_b,
        site_b_id,
        EventType.MAX_AGE,
        20.0,
        x_m=10.25,
        y_m=10.25,
    )

    results = (
        _particle_result(
            scenario_a,
            local_id,
            ParticleStatus.MAX_AGE,
            final_age_seconds=10.0,
            final_x_m=0.25,
            final_y_m=0.25,
            events=(local_event, cross_event, local_terminal),
        ),
        _particle_result(
            scenario_a,
            flow_id,
            ParticleStatus.FLOW_DOMAIN_EXIT,
            final_age_seconds=20.0,
            final_x_m=2.0,
            final_y_m=2.0,
            events=(flow_event,),
        ),
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
            site_b_id,
            ParticleStatus.MAX_AGE,
            final_age_seconds=20.0,
            final_x_m=10.25,
            final_y_m=10.25,
            events=(site_b_event, site_b_terminal),
        ),
    )
    return _aggregate(results, (scenario_a, scenario_b))


def _replace_outcomes(
    aggregate: EventAggregateChunk,
    site_id: str,
    outcomes: dict[str, int],
) -> EventAggregateChunk:
    """複製站點 outcome mapping，保留其他欄位與原容器不變。"""

    outcome_by_site = dict(aggregate.outcome_count_by_site)
    outcome_by_site[site_id] = dict(outcomes)
    return replace(aggregate, outcome_count_by_site=outcome_by_site)


def _replace_site_grid(
    aggregate: EventAggregateChunk,
    site_id: str,
    counts: SiteEventGridCounts,
) -> EventAggregateChunk:
    """複製站點網格 mapping，讓 tamper 只影響指定站點。"""

    site_grid_counts = dict(aggregate.site_grid_counts)
    site_grid_counts[site_id] = counts
    return replace(aggregate, site_grid_counts=site_grid_counts)


def _legal_case_aggregate() -> EventAggregateChunk:
    """提供每個 parametrized case 一份獨立的合法基準容器。"""

    return _legal_aggregate()


def _tamper_site_set(aggregate: EventAggregateChunk) -> EventAggregateChunk:
    outcomes = dict(aggregate.outcome_count_by_site)
    outcomes["unknown-site"] = {}
    return replace(aggregate, outcome_count_by_site=outcomes)


def _tamper_input_total(aggregate: EventAggregateChunk) -> EventAggregateChunk:
    return replace(
        aggregate,
        input_particle_count=aggregate.input_particle_count + 1,
    )


def _tamper_valid_gt_total(aggregate: EventAggregateChunk) -> EventAggregateChunk:
    valid_by_site = dict(aggregate.valid_member_denominator_by_site)
    valid_by_site["site-a"] = aggregate.total_member_count_by_site["site-a"] + 1
    return replace(aggregate, valid_member_denominator_by_site=valid_by_site)


def _tamper_active_outcome(aggregate: EventAggregateChunk) -> EventAggregateChunk:
    outcomes = dict(aggregate.outcome_count_by_site["site-a"])
    outcomes[ParticleStatus.ACTIVE.value] = 0
    return _replace_outcomes(aggregate, "site-a", outcomes)


def _tamper_outcome_total(aggregate: EventAggregateChunk) -> EventAggregateChunk:
    outcomes = dict(aggregate.outcome_count_by_site["site-a"])
    outcomes[ParticleStatus.MAX_AGE.value] += 1
    return _replace_outcomes(aggregate, "site-a", outcomes)


def _tamper_receptor_valid_sum(aggregate: EventAggregateChunk) -> EventAggregateChunk:
    valid_by_receptor = dict(aggregate.valid_member_denominator_by_receptor)
    receptor_key = next(
        key for key in valid_by_receptor if key.study_site_id == "site-a"
    )
    valid_by_receptor[receptor_key] += 1
    return replace(
        aggregate,
        valid_member_denominator_by_receptor=valid_by_receptor,
    )


def _tamper_boundary_raw_vs_travel(
    aggregate: EventAggregateChunk,
) -> EventAggregateChunk:
    boundary_key = BoundaryAggregateKey("site-a", "local", _SHARED_SEGMENT_ID)
    travel = {
        key: np.array(value, dtype=np.int64, copy=True)
        for key, value in aggregate.boundary_travel_age_histogram.items()
    }
    travel[boundary_key][0, 0] += 1
    return replace(aggregate, boundary_travel_age_histogram=travel)


def _tamper_source_raw_vs_age(
    aggregate: EventAggregateChunk,
) -> EventAggregateChunk:
    source_key = SourceReceptorAggregateKey(
        "site-a",
        "receptor-a",
        "local",
        _SHARED_SEGMENT_ID,
    )
    travel = {
        key: np.array(value, dtype=np.int64, copy=True)
        for key, value in aggregate.source_receptor_travel_age_histogram.items()
    }
    travel[source_key][0] += 1
    return replace(aggregate, source_receptor_travel_age_histogram=travel)


def _tamper_source_total_vs_boundary(
    aggregate: EventAggregateChunk,
) -> EventAggregateChunk:
    """同步增加 outer source raw 與 age histogram，使其只破壞 boundary 守恆。

    選擇原本只有一筆的 outer 來源列，讓增加後仍不超過受體 valid denominator；
    如此本案例繼續直接測試既有 source-to-boundary 守恆，而不會先命中新政策的
    source numerator 上限。
    """

    source_key = SourceReceptorAggregateKey(
        "site-a",
        "receptor-a",
        "outer",
        _SHARED_SEGMENT_ID,
    )
    source_raw = dict(aggregate.source_receptor_raw_count)
    source_raw[source_key] += 1
    source_travel = {
        key: np.array(value, dtype=np.int64, copy=True)
        for key, value in aggregate.source_receptor_travel_age_histogram.items()
    }
    source_travel[source_key][0] += 1
    return replace(
        aggregate,
        source_receptor_raw_count=source_raw,
        source_receptor_travel_age_histogram=source_travel,
    )


def _tamper_local_grid_vs_boundary(
    aggregate: EventAggregateChunk,
) -> EventAggregateChunk:
    """增加 outer 網格一格，讓空間投影總量大於 boundary raw count。

    原本 outer numerator 只有一筆，增加後仍不超過 site valid denominator；因此
    這個案例專門保留既有 grid-to-boundary 守恆測試，新的 local／outer 上限則由
    獨立 tamper 案例覆蓋。
    """

    counts = aggregate.site_grid_counts["site-a"]
    outer_grid = np.array(counts.outer_first_exit_count, dtype=np.int64, copy=True)
    outer_grid[0, 0] += 1
    bad_counts = replace(counts, outer_first_exit_count=outer_grid)
    return _replace_site_grid(aggregate, "site-a", bad_counts)


def _tamper_local_grid_above_valid_denominator(
    aggregate: EventAggregateChunk,
) -> EventAggregateChunk:
    """只把 local grid 推到有效成員分母以上，直接測試 numerator gate。

    這個 tamper 不同步修改 boundary raw；新 relationship gate 應在既有投影守恆
    之前辨識「單站 local numerator 不可能超過同站 valid member 母體」的錯誤。
    失敗 grid 不參與這個 gate，因為它是 outcome 對應的例外診斷欄位。
    """

    counts = aggregate.site_grid_counts["site-a"]
    local_grid = np.array(counts.local_first_exit_count, dtype=np.int64, copy=True)
    local_grid[0, 0] += 1
    bad_counts = replace(counts, local_first_exit_count=local_grid)
    return _replace_site_grid(aggregate, "site-a", bad_counts)


def _tamper_outer_grid_above_valid_denominator(
    aggregate: EventAggregateChunk,
) -> EventAggregateChunk:
    """只把 outer grid 推到有效成員分母以上，直接測試 outer numerator gate。"""

    counts = aggregate.site_grid_counts["site-a"]
    outer_grid = np.array(counts.outer_first_exit_count, dtype=np.int64, copy=True)
    outer_grid[0, 0] += aggregate.valid_member_denominator_by_site["site-a"]
    bad_counts = replace(counts, outer_first_exit_count=outer_grid)
    return _replace_site_grid(aggregate, "site-a", bad_counts)


def _tamper_failure_grid_vs_outcome(
    aggregate: EventAggregateChunk,
) -> EventAggregateChunk:
    """增加 DATA_GAP 失敗網格一格，使其不再等於 DATA_GAP outcome。"""

    counts = aggregate.site_grid_counts["site-a"]
    failure_grid = np.array(
        counts.data_gap_failure_count,
        dtype=np.int64,
        copy=True,
    )
    failure_grid[0, 0] += 1
    bad_counts = replace(counts, data_gap_failure_count=failure_grid)
    return _replace_site_grid(aggregate, "site-a", bad_counts)


def _tamper_cross_count_gt_source(
    aggregate: EventAggregateChunk,
) -> EventAggregateChunk:
    """把有序跨站 unique count 增至超過來源站點 valid denominator。"""

    cross_key = CrossSiteAggregateKey("site-a", "site-b")
    cross_counts = dict(aggregate.cross_site_unique_member_count)
    cross_counts[cross_key] = (
        aggregate.valid_member_denominator_by_site["site-a"] + 1
    )
    return replace(aggregate, cross_site_unique_member_count=cross_counts)


def _tamper_source_raw_above_receptor_valid(
    aggregate: EventAggregateChunk,
) -> EventAggregateChunk:
    """把單一 source-receptor raw 推到其受體有效分母以上。

    正式 gate 會先跨同一 site、receptor 與 boundary kind 的 segment 合計；核心
    fixture 只有一個 segment，因此這裡以單 key 直接覆蓋同一條規則的最小案例。
    raw 與 age histogram 仍保持相等，讓測試只命中新 denominator 關係而非形狀或
    raw/hist 守恆檢查。
    """

    source_key = SourceReceptorAggregateKey(
        "site-a",
        "receptor-a",
        "local",
        _SHARED_SEGMENT_ID,
    )
    receptor_key = next(
        key
        for key in aggregate.valid_member_denominator_by_receptor
        if key.study_site_id == "site-a" and key.receptor_id == "receptor-a"
    )
    target = aggregate.valid_member_denominator_by_receptor[receptor_key] + 1
    source_raw = dict(aggregate.source_receptor_raw_count)
    source_raw[source_key] = target
    source_travel = {
        key: np.array(value, dtype=np.int64, copy=True)
        for key, value in aggregate.source_receptor_travel_age_histogram.items()
    }
    source_travel[source_key][:] = 0
    source_travel[source_key][0] = target
    return replace(
        aggregate,
        source_receptor_raw_count=source_raw,
        source_receptor_travel_age_histogram=source_travel,
    )


def _tamper_bed_first_above_valid_denominator(
    aggregate: EventAggregateChunk,
) -> EventAggregateChunk:
    """把站點 bed first contact 合計推到有效成員分母以上。"""

    counts = aggregate.site_grid_counts["site-a"]
    bed_first = np.array(counts.bed_first_contact_count, dtype=np.int64, copy=True)
    bed_first[0, 0] = aggregate.valid_member_denominator_by_site["site-a"] + 1
    bad_counts = replace(counts, bed_first_contact_count=bed_first)
    return _replace_site_grid(aggregate, "site-a", bad_counts)


def _tamper_unknown_site_key(
    aggregate: EventAggregateChunk,
) -> EventAggregateChunk:
    """加入引用未登錄站點的 cross-site key。"""

    cross_counts = dict(aggregate.cross_site_unique_member_count)
    cross_counts[CrossSiteAggregateKey("unknown-site", "site-a")] = 0
    return replace(aggregate, cross_site_unique_member_count=cross_counts)


_TAMPER_CASES: tuple[
    tuple[str, Callable[[EventAggregateChunk], EventAggregateChunk], str], ...
] = (
    (
        "site set mismatch",
        _tamper_site_set,
        "site set",
    ),
    (
        "input count differs from site total",
        _tamper_input_total,
        "input_particle_count.*total_member_count_by_site",
    ),
    (
        "valid denominator exceeds total",
        _tamper_valid_gt_total,
        "valid denominator.*total member count",
    ),
    (
        "ACTIVE outcome key",
        _tamper_active_outcome,
        "outcome key.*ACTIVE",
    ),
    (
        "outcome total mismatch",
        _tamper_outcome_total,
        "outcome count 合計.*total member count",
    ),
    (
        "receptor valid sum mismatch",
        _tamper_receptor_valid_sum,
        "receptor valid 合計.*site valid denominator",
    ),
    (
        "boundary raw differs from travel histogram",
        _tamper_boundary_raw_vs_travel,
        "raw array 合計.*histogram 合計",
    ),
    (
        "source raw differs from age histogram",
        _tamper_source_raw_vs_age,
        "raw scalar.*age histogram 合計",
    ),
    (
        "source total differs from boundary raw",
        _tamper_source_total_vs_boundary,
        "所有 receptor source raw.*boundary raw",
    ),
    (
        "outer grid differs from boundary raw",
        _tamper_local_grid_vs_boundary,
        "outer grid 合計.*boundary raw",
    ),
    (
        "local grid exceeds valid denominator",
        _tamper_local_grid_above_valid_denominator,
        "local grid 合計不可大於.*valid denominator",
    ),
    (
        "outer grid exceeds valid denominator",
        _tamper_outer_grid_above_valid_denominator,
        "outer grid 合計不可大於.*valid denominator",
    ),
    (
        "source raw exceeds receptor valid denominator",
        _tamper_source_raw_above_receptor_valid,
        "source-receptor.*raw 跨 segment 合計不可大於.*valid denominator",
    ),
    (
        "bed first exceeds valid denominator",
        _tamper_bed_first_above_valid_denominator,
        "bed_first_contact_count 合計不可大於.*valid denominator",
    ),
    (
        "failure grid differs from failure outcome",
        _tamper_failure_grid_vs_outcome,
        "data_gap_failure grid 合計.*DATA_GAP outcome",
    ),
    (
        "cross count exceeds source valid denominator",
        _tamper_cross_count_gt_source,
        "cross-site key.*source site valid denominator",
    ),
    (
        "key references unknown site",
        _tamper_unknown_site_key,
        "cross-site key.*未知 site",
    ),
)


@pytest.mark.parametrize(
    ("case_name", "tamper", "error_pattern"),
    _TAMPER_CASES,
    ids=[case[0] for case in _TAMPER_CASES],
)
def test_event_aggregate_container_rejects_core_relationship_tampering(
    case_name: str,
    tamper: Callable[[EventAggregateChunk], EventAggregateChunk],
    error_pattern: str,
) -> None:
    """每個核心 tamper 都必須因明確跨欄位關係失效而拒絕封存。"""

    del case_name
    with pytest.raises(ValueError, match=error_pattern):
        tamper(_legal_case_aggregate())
