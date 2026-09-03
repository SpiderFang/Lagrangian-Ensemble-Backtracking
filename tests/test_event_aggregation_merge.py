"""事件聚合 chunk merge 的 happy-path 測試。

本檔直接重用 ``test_event_aggregation_core`` 的最小兩站 fixture 與公開事件
聚合 API，驗證單一 chunk 複製、相同拓撲的分片相加，以及後續受體聯集與排序。
測試中的網格軸固定為公尺制 ``(y_cell, x_cell)``，邊界旅行直方圖軸固定為
``(s_bin, age_bin)``，其中 ``s`` 以公尺、回溯年齡以秒表示。多 chunk merge
假設上游 run manifest 已證明各 shard 的粒子集合互斥且不重複；merge 本身不
保存 particle ID，不能自行偵測重複輸入。

所有計數只代表條件式來源足跡或相對來源權重的原始累加量，不代表絕對來源
機率或因果歸因。這些人工 fixture 用來固定資料契約與數值守恆，不代表正式
海洋資料。
"""

from __future__ import annotations

import numpy as np
from test_event_aggregation_core import (
    _AGE_BIN_EDGES_SECONDS,
    _SHARED_SEGMENT_ID,
    _aggregate,
    _boundary_event,
    _particle_result,
    _scenario,
)

from lagrangian_backtracking.event_aggregation import (
    ReceptorAggregateKey,
    SourceReceptorAggregateKey,
    merge_event_aggregate_chunks,
)
from lagrangian_backtracking.models import EventType, ParticleStatus

_GRID_FIELDS = (
    "local_first_exit_count",
    "outer_first_exit_count",
    "bed_first_contact_count",
    "bed_repeated_contact_count",
    "data_gap_failure_count",
    "numerical_failure_count",
)


def _assert_chunk_values_equal(actual, expected) -> None:
    """逐欄比較兩個 chunk 的標量、網格、邊界與 source-receptor 數值。

    比較 helper 將網格明確解讀為 ``(y_cell, x_cell)``，將邊界旅行直方圖
    明確解讀為 ``(s_bin, age_bin)``；這能避免只比較少數非零欄位而漏掉零拓撲、
    失敗網格或其餘受體分母。mapping 的 key 與數值都必須相同，陣列則逐元素
    比較，以驗證條件式來源足跡的整體資料結構沒有因 merge 而改變。
    """

    assert actual.input_particle_count == expected.input_particle_count
    assert dict(actual.source_receptor_raw_count) == dict(
        expected.source_receptor_raw_count
    )
    assert dict(actual.cross_site_unique_member_count) == dict(
        expected.cross_site_unique_member_count
    )
    assert {
        site_id: dict(outcomes)
        for site_id, outcomes in actual.outcome_count_by_site.items()
    } == {
        site_id: dict(outcomes)
        for site_id, outcomes in expected.outcome_count_by_site.items()
    }
    assert dict(actual.valid_member_denominator_by_site) == dict(
        expected.valid_member_denominator_by_site
    )
    assert dict(actual.total_member_count_by_site) == dict(
        expected.total_member_count_by_site
    )
    assert dict(actual.valid_member_denominator_by_receptor) == dict(
        expected.valid_member_denominator_by_receptor
    )

    np.testing.assert_array_equal(
        actual.age_bin_edges_seconds,
        expected.age_bin_edges_seconds,
    )
    for field in (
        "boundary_bin_edges_m",
        "boundary_arclength_raw_count",
        "boundary_travel_age_histogram",
        "source_receptor_travel_age_histogram",
    ):
        actual_mapping = getattr(actual, field)
        expected_mapping = getattr(expected, field)
        assert set(actual_mapping) == set(expected_mapping)
        for key in actual_mapping:
            np.testing.assert_array_equal(
                actual_mapping[key],
                expected_mapping[key],
            )

    assert set(actual.site_grid_counts) == set(expected.site_grid_counts)
    for site_id in actual.site_grid_counts:
        actual_grid = actual.site_grid_counts[site_id]
        expected_grid = expected.site_grid_counts[site_id]
        for field in _GRID_FIELDS:
            np.testing.assert_array_equal(
                getattr(actual_grid, field),
                getattr(expected_grid, field),
            )


def _assert_no_array_alias(actual, original) -> None:
    """確認 merge 結果的每個陣列都不與輸入 chunk 共用記憶體。

    ``EventAggregateChunk`` 對外是唯讀資料，但唯讀 view 仍可能指向 caller 的
    原始 buffer；正式 release 若保留這種 alias，呼叫端後續的 buffer 生命週期
    或低階修改就可能改變已封存的來源足跡。因此此 helper 覆蓋 age 軸、邊界
    格線與兩類直方圖、source-receptor age 軸，以及六類站點網格。
    """

    assert actual.age_bin_edges_seconds is not original.age_bin_edges_seconds
    assert not np.shares_memory(
        actual.age_bin_edges_seconds,
        original.age_bin_edges_seconds,
    )
    for field in (
        "boundary_bin_edges_m",
        "boundary_arclength_raw_count",
        "boundary_travel_age_histogram",
        "source_receptor_travel_age_histogram",
    ):
        actual_mapping = getattr(actual, field)
        original_mapping = getattr(original, field)
        for key in original_mapping:
            assert actual_mapping[key] is not original_mapping[key]
            assert not np.shares_memory(
                actual_mapping[key],
                original_mapping[key],
            )
    for site_id in original.site_grid_counts:
        actual_grid = actual.site_grid_counts[site_id]
        original_grid = original.site_grid_counts[site_id]
        assert actual_grid is not original_grid
        for field in _GRID_FIELDS:
            assert getattr(actual_grid, field) is not getattr(original_grid, field)
            assert not np.shares_memory(
                getattr(actual_grid, field),
                getattr(original_grid, field),
            )


def _assert_chunk_is_exact_sum(merged, left, right) -> None:
    """驗證兩個互斥 chunk 的所有計數欄位逐值相加且軸線未變。

    此 helper 對每一個固定公尺制網格、邊界 ``s`` bin、旅行年齡秒 bin、事件
    outcome、站點／受體分母與跨站計數逐項檢查。source-receptor 與 receptor
    mapping 允許以聯集為拓撲；缺少的 key 按零處理，對應正式 shard 可能只
    含部分受體或部分來源分類的情境。
    """

    assert merged.input_particle_count == (
        left.input_particle_count + right.input_particle_count
    )
    np.testing.assert_array_equal(
        merged.age_bin_edges_seconds,
        left.age_bin_edges_seconds,
    )
    np.testing.assert_array_equal(
        merged.age_bin_edges_seconds,
        right.age_bin_edges_seconds,
    )

    for field in (
        "boundary_bin_edges_m",
        "boundary_arclength_raw_count",
        "boundary_travel_age_histogram",
    ):
        merged_mapping = getattr(merged, field)
        left_mapping = getattr(left, field)
        right_mapping = getattr(right, field)
        assert set(left_mapping) == set(right_mapping) == set(merged_mapping)
        for key in merged_mapping:
            if field == "boundary_bin_edges_m":
                np.testing.assert_array_equal(
                    merged_mapping[key],
                    left_mapping[key],
                )
                np.testing.assert_array_equal(
                    merged_mapping[key],
                    right_mapping[key],
                )
            else:
                np.testing.assert_array_equal(
                    merged_mapping[key],
                    left_mapping[key] + right_mapping[key],
                )

    for site_id in merged.site_grid_counts:
        merged_grid = merged.site_grid_counts[site_id]
        left_grid = left.site_grid_counts[site_id]
        right_grid = right.site_grid_counts[site_id]
        for field in _GRID_FIELDS:
            np.testing.assert_array_equal(
                getattr(merged_grid, field),
                getattr(left_grid, field) + getattr(right_grid, field),
            )

    source_keys = set(left.source_receptor_raw_count) | set(
        right.source_receptor_raw_count
    )
    assert set(merged.source_receptor_raw_count) == source_keys
    assert set(merged.source_receptor_travel_age_histogram) == source_keys
    for key in source_keys:
        assert merged.source_receptor_raw_count[key] == (
            left.source_receptor_raw_count.get(key, 0)
            + right.source_receptor_raw_count.get(key, 0)
        )
        left_hist = left.source_receptor_travel_age_histogram.get(key)
        right_hist = right.source_receptor_travel_age_histogram.get(key)
        if left_hist is None:
            expected_hist = right_hist
        elif right_hist is None:
            expected_hist = left_hist
        else:
            expected_hist = left_hist + right_hist
        np.testing.assert_array_equal(
            merged.source_receptor_travel_age_histogram[key],
            expected_hist,
        )

    assert set(merged.cross_site_unique_member_count) == (
        set(left.cross_site_unique_member_count)
        | set(right.cross_site_unique_member_count)
    )
    for key in merged.cross_site_unique_member_count:
        assert merged.cross_site_unique_member_count[key] == (
            left.cross_site_unique_member_count.get(key, 0)
            + right.cross_site_unique_member_count.get(key, 0)
        )

    assert set(merged.outcome_count_by_site) == (
        set(left.outcome_count_by_site) | set(right.outcome_count_by_site)
    )
    for site_id in merged.outcome_count_by_site:
        statuses = set(left.outcome_count_by_site.get(site_id, {})) | set(
            right.outcome_count_by_site.get(site_id, {})
        )
        assert set(merged.outcome_count_by_site[site_id]) == statuses
        for status in statuses:
            assert merged.outcome_count_by_site[site_id][status] == (
                left.outcome_count_by_site.get(site_id, {}).get(status, 0)
                + right.outcome_count_by_site.get(site_id, {}).get(status, 0)
            )

    for field in (
        "valid_member_denominator_by_site",
        "total_member_count_by_site",
        "valid_member_denominator_by_receptor",
    ):
        merged_mapping = getattr(merged, field)
        left_mapping = getattr(left, field)
        right_mapping = getattr(right, field)
        assert set(merged_mapping) == set(left_mapping) | set(right_mapping)
        for key in merged_mapping:
            assert merged_mapping[key] == (
                left_mapping.get(key, 0) + right_mapping.get(key, 0)
            )


def test_merge_single_chunk_preserves_values_without_double_count_or_aliasing() -> None:
    """單一 chunk merge 應保持所有原值、只計一次，且完全切斷陣列記憶體 alias。

    這裡以 local 首次離開加 MAX_AGE terminal 組成一個合法結果，讓至少一個
    邊界與 source-receptor age histogram 非零；其餘站點與事件類別仍保留完整
    零拓撲。預期 merge 不會因重新封存而 double count，也不會把原始公尺制網格、
    ``s`` 弧長或秒制 age 軸 buffer 帶入輸出。
    """

    scenario = _scenario()
    particle_id = "particle-single-merge"
    local_event = _boundary_event(
        scenario,
        particle_id,
        EventType.LOCAL_DOMAIN_FIRST_EXIT,
        5.0,
        x_m=0.25,
        y_m=0.25,
        boundary_segment_id=_SHARED_SEGMENT_ID,
        boundary_s_m=0.0,
    )
    terminal_event = _boundary_event(
        scenario,
        particle_id,
        EventType.MAX_AGE,
        20.0,
        x_m=0.25,
        y_m=0.25,
    )
    result = _particle_result(
        scenario,
        particle_id,
        ParticleStatus.MAX_AGE,
        final_age_seconds=20.0,
        final_x_m=0.25,
        final_y_m=0.25,
        events=(local_event, terminal_event),
    )
    chunk = _aggregate((result,), (scenario,))

    merged = merge_event_aggregate_chunks((chunk,))

    _assert_chunk_values_equal(merged, chunk)
    assert merged is not chunk
    _assert_no_array_alias(merged, chunk)


def test_merge_two_same_topology_chunks_adds_all_counts_exactly() -> None:
    """兩個相同拓撲且粒子互斥的 chunk 應對所有計數欄位精確相加。

    第一個粒子包含 local 邊界事件與一次跨站 enter，第二個粒子以同一個
    ``FLOW_DOMAIN_OPEN_EXIT`` 同時承擔 local／outer 語意；兩者共享相同的
    site、邊界、cross-site 與 outcome 拓撲，但事件落在不同類別或 bin。測試
    因而同時覆蓋 input、站點網格、邊界 raw／旅行 age histogram、source-receptor、
    outcome、分母與跨站計數的逐值守恆。
    """

    scenario = _scenario()
    first_particle_id = "particle-sum-left"
    first_local_event = _boundary_event(
        scenario,
        first_particle_id,
        EventType.LOCAL_DOMAIN_FIRST_EXIT,
        5.0,
        x_m=0.25,
        y_m=0.25,
        boundary_segment_id=_SHARED_SEGMENT_ID,
        boundary_s_m=0.0,
    )
    first_cross_event = _boundary_event(
        scenario,
        first_particle_id,
        EventType.OTHER_SITE_LOCAL_DOMAIN_ENTER,
        10.0,
        x_m=0.25,
        y_m=0.25,
        related_study_site_id="site-b",
    )
    first_terminal_event = _boundary_event(
        scenario,
        first_particle_id,
        EventType.MAX_AGE,
        20.0,
        x_m=0.25,
        y_m=0.25,
    )
    first_result = _particle_result(
        scenario,
        first_particle_id,
        ParticleStatus.MAX_AGE,
        final_age_seconds=20.0,
        final_x_m=0.25,
        final_y_m=0.25,
        events=(first_local_event, first_cross_event, first_terminal_event),
    )

    second_particle_id = "particle-sum-right"
    second_flow_event = _boundary_event(
        scenario,
        second_particle_id,
        EventType.FLOW_DOMAIN_OPEN_EXIT,
        20.0,
        x_m=2.0,
        y_m=2.0,
        boundary_segment_id=_SHARED_SEGMENT_ID,
        boundary_s_m=2.5,
        attributes={"also_local_domain_first_exit": True},
    )
    second_result = _particle_result(
        scenario,
        second_particle_id,
        ParticleStatus.FLOW_DOMAIN_EXIT,
        final_age_seconds=20.0,
        final_x_m=2.0,
        final_y_m=2.0,
        events=(second_flow_event,),
    )

    left = _aggregate((first_result,), (scenario,))
    right = _aggregate((second_result,), (scenario,))
    merged = merge_event_aggregate_chunks((left, right))

    _assert_chunk_is_exact_sum(merged, left, right)
    expected = _aggregate((first_result, second_result), (scenario,))
    _assert_chunk_values_equal(merged, expected)


def _single_receptor_chunk(
    receptor_id: str,
    particle_id: str,
    event_age_seconds: float,
):
    """建立只含一個受體的 chunk，供聯集與輸入順序測試重用。

    兩個呼叫使用相同 site 與 AggregateSpec，但各自只有一個受體／粒子；local
    crossing 的 ``s`` 以公尺表示，事件 travel age 以秒表示，故每個 chunk 只
    會填入自己的 source-receptor key 與受體有效分母。兩個 chunk 若由正式
    shard manifest 證明粒子互斥，即可代表可安全相加的條件式來源足跡分片。
    """

    scenario = _scenario(
        scenario_id=f"scenario-{receptor_id}",
        receptor_id=receptor_id,
    )
    local_event = _boundary_event(
        scenario,
        particle_id,
        EventType.LOCAL_DOMAIN_FIRST_EXIT,
        event_age_seconds,
        x_m=0.25,
        y_m=0.25,
        boundary_segment_id=_SHARED_SEGMENT_ID,
        boundary_s_m=0.0,
    )
    terminal_event = _boundary_event(
        scenario,
        particle_id,
        EventType.MAX_AGE,
        20.0,
        x_m=0.25,
        y_m=0.25,
    )
    result = _particle_result(
        scenario,
        particle_id,
        ParticleStatus.MAX_AGE,
        final_age_seconds=20.0,
        final_x_m=0.25,
        final_y_m=0.25,
        events=(local_event, terminal_event),
    )
    return _aggregate((result,), (scenario,))


def test_merge_two_receptor_chunks_unions_source_and_receptor_keys() -> None:
    """不同受體的 chunk merge 應取 key 聯集，並保留各自 source／分母計數。

    每個 chunk 只有一個 site-a 受體與一個互斥粒子，但兩者共享完整的站點、
    邊界與跨站零拓撲。受體 A 的 local event 落在第一個秒制 age bin，受體 B
    落在第二個 bin；預期 source-receptor raw／旅行 age histogram 與受體有效
    分母各自只增加一次，不能因 merge 只沿用第一個 chunk 的 key 而遺失資料。
    """

    receptor_a = _single_receptor_chunk(
        "receptor-a",
        "particle-receptor-a",
        5.0,
    )
    receptor_b = _single_receptor_chunk(
        "receptor-b",
        "particle-receptor-b",
        15.0,
    )

    merged = merge_event_aggregate_chunks((receptor_a, receptor_b))

    expected_source_keys = {
        SourceReceptorAggregateKey(
            "site-a",
            receptor_id,
            boundary_kind,
            _SHARED_SEGMENT_ID,
        )
        for receptor_id in ("receptor-a", "receptor-b")
        for boundary_kind in ("local", "outer")
    }
    assert set(merged.source_receptor_raw_count) == expected_source_keys
    assert set(merged.source_receptor_travel_age_histogram) == expected_source_keys
    for receptor_id, expected_histogram in (
        ("receptor-a", [1, 0]),
        ("receptor-b", [0, 1]),
    ):
        local_key = SourceReceptorAggregateKey(
            "site-a",
            receptor_id,
            "local",
            _SHARED_SEGMENT_ID,
        )
        outer_key = SourceReceptorAggregateKey(
            "site-a",
            receptor_id,
            "outer",
            _SHARED_SEGMENT_ID,
        )
        assert merged.source_receptor_raw_count[local_key] == 1
        assert merged.source_receptor_raw_count[outer_key] == 0
        np.testing.assert_array_equal(
            merged.source_receptor_travel_age_histogram[local_key],
            expected_histogram,
        )
        np.testing.assert_array_equal(
            merged.source_receptor_travel_age_histogram[outer_key],
            [0, 0],
        )

    assert merged.valid_member_denominator_by_receptor == {
        ReceptorAggregateKey("site-a", "receptor-a"): 1,
        ReceptorAggregateKey("site-a", "receptor-b"): 1,
    }
    assert merged.valid_member_denominator_by_site == {"site-a": 2, "site-b": 0}
    assert merged.total_member_count_by_site == {"site-a": 2, "site-b": 0}
    np.testing.assert_array_equal(
        merged.age_bin_edges_seconds,
        _AGE_BIN_EDGES_SECONDS,
    )


def test_merge_reversed_chunk_order_is_value_and_key_order_deterministic() -> None:
    """顛倒 chunk 順序後所有數值與 source／receptor key iteration 順序都相同。

    這裡刻意先建立 receptor-z、再建立 receptor-a，使兩次 merge 的第一個
    chunk 與 mapping 插入順序不同。公開 merge 必須對 source-receptor 與受體
    分母 key 依 site、受體、邊界分類及 segment 的資料契約排序；因此結果既要
    保持公尺制網格、弧長／秒制 histogram 與所有 scalar 數值相同，也要讓下游
    serialisation、checksum 與條件式來源足跡報表得到穩定 key 順序。
    """

    first = _single_receptor_chunk(
        "receptor-z",
        "particle-order-z",
        5.0,
    )
    second = _single_receptor_chunk(
        "receptor-a",
        "particle-order-a",
        15.0,
    )

    forward = merge_event_aggregate_chunks((first, second))
    reverse = merge_event_aggregate_chunks((second, first))

    _assert_chunk_values_equal(reverse, forward)
    assert list(forward.source_receptor_raw_count) == list(
        reverse.source_receptor_raw_count
    )
    assert list(forward.source_receptor_travel_age_histogram) == list(
        reverse.source_receptor_travel_age_histogram
    )
    assert list(forward.valid_member_denominator_by_receptor) == list(
        reverse.valid_member_denominator_by_receptor
    )

    expected_source_order = sorted(
        forward.source_receptor_raw_count,
        key=lambda key: (
            key.study_site_id,
            key.receptor_id,
            key.boundary_kind,
            key.boundary_segment_id,
        ),
    )
    expected_receptor_order = sorted(
        forward.valid_member_denominator_by_receptor,
        key=lambda key: (key.study_site_id, key.receptor_id),
    )
    assert list(forward.source_receptor_raw_count) == expected_source_order
    assert list(forward.source_receptor_travel_age_histogram) == expected_source_order
    assert list(forward.valid_member_denominator_by_receptor) == expected_receptor_order
