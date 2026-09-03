"""事件聚合串流 accumulator 的資料契約與生命週期測試。

本檔驗證 ``EventAggregateAccumulator`` 在逐個 trajectory chunk 到達時立即
累加，而不是把所有大型公尺制網格留在歷史清單中。測試使用既有的合成事件
fixture：站點網格軸是 ``(y_cell, x_cell)``，邊界弧長與 grid 座標以公尺（m）
表示，旅行年齡 bin 以秒（s）表示。這些 fixture 只用來驗證資料結構、守恆、
overflow 與記憶體生命週期；它們不是實際 OCM／NWW 資料，也不是科學成果。
"""

from __future__ import annotations

import gc
import weakref
from dataclasses import replace

import numpy as np
import pytest
from test_event_aggregation_merge import (
    _assert_chunk_values_equal,
    _single_receptor_chunk,
)
from test_event_aggregation_merge_validation import _legal_int64_max_chunk

import lagrangian_backtracking.event_aggregation as event_aggregation
from lagrangian_backtracking.event_aggregation import (
    EventAggregateAccumulator,
    merge_event_aggregate_chunks,
)


def test_accumulator_does_not_allocate_grid_until_first_add() -> None:
    """新 reducer 不應在尚未知道輸入 topology 時配置網格或 histogram。"""

    accumulator = EventAggregateAccumulator()

    assert accumulator.chunk_count == 0
    assert accumulator._site_grid_arrays is None
    with pytest.raises(ValueError, match="chunks"):
        accumulator.finalize()
    assert accumulator.chunk_count == 0


def test_accumulator_matches_streaming_merge_and_unions_dynamic_keys() -> None:
    """多個 chunk 應精確相加，並以固定排序保留動態 source／receptor 聯集。"""

    first = _single_receptor_chunk("receptor-z", "particle-z", 5.0)
    second = _single_receptor_chunk("receptor-a", "particle-a", 15.0)

    accumulator = EventAggregateAccumulator()
    accumulator.add(first)
    accumulator.add(second)
    streamed = accumulator.finalize()
    expected = merge_event_aggregate_chunks((first, second))

    _assert_chunk_values_equal(streamed, expected)
    assert accumulator.chunk_count == 2
    assert list(streamed.source_receptor_raw_count) == sorted(
        streamed.source_receptor_raw_count,
        key=lambda key: (
            key.study_site_id,
            key.receptor_id,
            key.boundary_kind,
            key.boundary_segment_id,
        ),
    )
    assert list(streamed.valid_member_denominator_by_receptor) == sorted(
        streamed.valid_member_denominator_by_receptor,
        key=lambda key: (key.study_site_id, key.receptor_id),
    )


def test_accumulator_key_order_is_deterministic_when_chunk_order_is_reversed() -> None:
    """反轉 chunk 到達順序不應改變聯集 key 的序列化順序或數值。"""

    first = _single_receptor_chunk("receptor-z", "particle-z", 5.0)
    second = _single_receptor_chunk("receptor-a", "particle-a", 15.0)

    forward = EventAggregateAccumulator()
    forward.add(first)
    forward.add(second)
    forward_result = forward.finalize()

    reverse = EventAggregateAccumulator()
    reverse.add(second)
    reverse.add(first)
    reverse_result = reverse.finalize()

    _assert_chunk_values_equal(reverse_result, forward_result)
    assert list(reverse_result.source_receptor_raw_count) == list(
        forward_result.source_receptor_raw_count
    )
    assert list(reverse_result.valid_member_denominator_by_receptor) == list(
        forward_result.valid_member_denominator_by_receptor
    )


def test_merge_consumes_iterator_once_and_does_not_retain_old_chunk_arrays() -> None:
    """merge 僅建立一次 iterator，且 generator 的舊 chunk 可在讀取後釋放。"""

    state = {"iter_calls": 0, "next_calls": 0}
    array_refs: list[weakref.ReferenceType[np.ndarray]] = []

    class OnePassIterator:
        """只允許一次 ``iter`` 的小型 iterator，避免測試意外重播輸入。"""

        def __iter__(self):
            state["iter_calls"] += 1
            if state["iter_calls"] > 1:
                raise AssertionError("merge 不應重建或重播輸入 iterator")
            return self

        def __next__(self):
            state["next_calls"] += 1
            if state["next_calls"] == 1:
                chunk = _single_receptor_chunk(
                    "receptor-stream",
                    "particle-stream",
                    5.0,
                )
                array_refs.append(
                    weakref.ref(chunk.site_grid_counts["site-a"].local_first_exit_count)
                )
                return chunk
            raise StopIteration

    merged = merge_event_aggregate_chunks(OnePassIterator())
    gc.collect()

    assert state == {"iter_calls": 1, "next_calls": 2}
    assert array_refs[0]() is None
    assert merged.input_particle_count == 1


def test_accumulator_closes_after_finalize_and_rejects_empty_finalize() -> None:
    """零 chunk、重複 finalize 與 finalize 後 add 都必須固定拒絕。"""

    empty = EventAggregateAccumulator()
    with pytest.raises(ValueError, match="chunks"):
        empty.finalize()

    accumulator = EventAggregateAccumulator()
    chunk = _single_receptor_chunk("receptor-life", "particle-life", 5.0)
    accumulator.add(chunk)
    accumulator.finalize()

    with pytest.raises(ValueError, match="封閉"):
        accumulator.finalize()
    with pytest.raises(ValueError, match="封閉"):
        accumulator.add(chunk)


def test_second_chunk_topology_failure_closes_without_partial_accumulation() -> None:
    """第二個 chunk 的固定 age 軸錯誤不得寫入部分數值，且 reducer 要 fail closed。"""

    first = _single_receptor_chunk("receptor-safe", "particle-safe", 5.0)
    second = _single_receptor_chunk("receptor-safe", "particle-safe-2", 5.0)
    second = replace(
        second,
        age_bin_edges_seconds=np.array([0.0, 10.0, 30.0], dtype=np.float64),
    )

    accumulator = EventAggregateAccumulator()
    accumulator.add(first)
    before_grid = {
        field: np.array(
            accumulator._site_grid_arrays["site-a"][field],
            copy=True,
        )
        for field in accumulator._site_grid_arrays["site-a"]
    }
    before_source_keys = set(accumulator._source_raw)
    before_input_count = accumulator._input_particle_count

    with pytest.raises(ValueError, match=r"chunks\[1\].*age_bin_edges_seconds"):
        accumulator.add(second)

    assert accumulator.chunk_count == 1
    assert set(accumulator._source_raw) == before_source_keys
    assert accumulator._input_particle_count == before_input_count
    for field, expected in before_grid.items():
        np.testing.assert_array_equal(
            accumulator._site_grid_arrays["site-a"][field],
            expected,
        )
    with pytest.raises(ValueError, match="封閉"):
        accumulator.add(first)
    with pytest.raises(ValueError, match="封閉"):
        accumulator.finalize()


def test_accumulator_rejects_int64_overflow_before_mutating_state() -> None:
    """固定寬度陣列溢位必須在寫入前拒絕，避免繞回負值或半加總結果。"""

    first = _legal_int64_max_chunk()
    second = _legal_int64_max_chunk()
    accumulator = EventAggregateAccumulator()
    accumulator.add(first)
    before = np.array(
        accumulator._site_grid_arrays["site-a"]["local_first_exit_count"],
        copy=True,
    )

    with pytest.raises(RuntimeError, match="int64"):
        accumulator.add(second)

    np.testing.assert_array_equal(
        accumulator._site_grid_arrays["site-a"]["local_first_exit_count"],
        before,
    )
    assert accumulator.chunk_count == 1
    with pytest.raises(ValueError, match="封閉"):
        accumulator.finalize()


def test_accumulator_result_has_no_array_alias_with_input_or_private_state() -> None:
    """finalize 結果必須切斷輸入 chunk 與 accumulator mutable arrays 的 alias。"""

    source = _single_receptor_chunk("receptor-copy", "particle-copy", 5.0)
    accumulator = EventAggregateAccumulator()
    accumulator.add(source)
    private_age_edges = accumulator._age_edges
    private_mappings = {
        "boundary_bin_edges_m": accumulator._boundary_edges,
        "boundary_arclength_raw_count": accumulator._boundary_raw,
        "boundary_travel_age_histogram": accumulator._boundary_travel,
        "source_receptor_travel_age_histogram": accumulator._source_travel,
    }
    private_grid_arrays = accumulator._site_grid_arrays
    result = accumulator.finalize()

    assert result is not source
    assert not np.shares_memory(result.age_bin_edges_seconds, source.age_bin_edges_seconds)
    assert not np.shares_memory(
        result.age_bin_edges_seconds,
        private_age_edges,
    )
    assert accumulator._age_edges is None
    assert accumulator._site_grid_arrays is None
    assert accumulator._boundary_edges is None
    assert accumulator._boundary_raw is None
    assert accumulator._boundary_travel is None
    assert accumulator._source_raw is None
    assert accumulator._source_travel is None
    assert accumulator._cross_site is None
    assert accumulator._outcomes is None
    assert accumulator._valid_by_site is None
    assert accumulator._total_by_site is None
    assert accumulator._valid_by_receptor is None
    for field in (
        "boundary_bin_edges_m",
        "boundary_arclength_raw_count",
        "boundary_travel_age_histogram",
        "source_receptor_travel_age_histogram",
    ):
        result_mapping = getattr(result, field)
        source_mapping = getattr(source, field)
        for key in source_mapping:
            assert not np.shares_memory(result_mapping[key], source_mapping[key])
            assert not np.shares_memory(
                result_mapping[key],
                private_mappings[field][key],
            )
    for site_id, counts in source.site_grid_counts.items():
        for field in (
            "local_first_exit_count",
            "outer_first_exit_count",
            "bed_first_contact_count",
            "bed_repeated_contact_count",
            "data_gap_failure_count",
            "numerical_failure_count",
        ):
            assert not np.shares_memory(
                getattr(result.site_grid_counts[site_id], field),
                getattr(counts, field),
            )
            assert not np.shares_memory(
                getattr(result.site_grid_counts[site_id], field),
                private_grid_arrays[site_id][field],
            )


def test_finalize_releases_private_arrays_after_defensive_result_copy() -> None:
    """finalize 後大型 private arrays 應可回收，結果仍獨立保存完整資料。"""

    source = _single_receptor_chunk("receptor-release", "particle-release", 5.0)
    accumulator = EventAggregateAccumulator()
    accumulator.add(source)
    private_array_refs = [
        weakref.ref(accumulator._age_edges),
        weakref.ref(
            accumulator._site_grid_arrays["site-a"]["local_first_exit_count"]
        ),
        weakref.ref(next(iter(accumulator._boundary_edges.values()))),
        weakref.ref(next(iter(accumulator._boundary_travel.values()))),
        weakref.ref(next(iter(accumulator._source_travel.values()))),
    ]

    result = accumulator.finalize()
    gc.collect()

    assert result.input_particle_count == source.input_particle_count
    assert all(reference() is None for reference in private_array_refs)
    assert accumulator._age_edges is None
    assert accumulator._site_grid_arrays is None
    assert accumulator._boundary_edges is None
    assert accumulator._boundary_raw is None
    assert accumulator._boundary_travel is None
    assert accumulator._source_travel is None


def test_accumulator_commit_does_not_call_legacy_safe_add(monkeypatch) -> None:
    """向量化 commit 應避開舊逐格 helper，且結果仍與既有 merge 完全相同。"""

    first = _single_receptor_chunk("receptor-vector-z", "particle-vector-z", 5.0)
    second = _single_receptor_chunk("receptor-vector-a", "particle-vector-a", 15.0)
    expected = merge_event_aggregate_chunks((first, second))

    accumulator = EventAggregateAccumulator()
    accumulator.add(first)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("accumulator commit 不應呼叫逐格 safe-add helper")

    monkeypatch.setattr(
        event_aggregation,
        "_safe_add_int64_arrays",
        fail_if_called,
    )
    accumulator.add(second)

    _assert_chunk_values_equal(accumulator.finalize(), expected)


def test_merge_wraps_iterator_errors_but_preserves_add_errors() -> None:
    """iterator 建立／next 錯誤包成 ValueError，chunk 驗證錯誤則保留位置。"""

    class BrokenIterable:
        def __iter__(self):
            raise RuntimeError("cannot create iterator")

    class BrokenIterator:
        def __iter__(self):
            return self

        def __next__(self):
            raise RuntimeError("cannot read next chunk")

    with pytest.raises(ValueError, match="可迭代"):
        merge_event_aggregate_chunks(BrokenIterable())
    with pytest.raises(ValueError, match="iterator"):
        merge_event_aggregate_chunks(BrokenIterator())
    with pytest.raises(ValueError, match=r"chunks\[0\]"):
        merge_event_aggregate_chunks((object(),))
