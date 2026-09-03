"""固定記憶體 pathway chunk reducer 的公開契約測試。

本檔重用 ``test_streaming_aggregation`` 的小型 synthetic fixture；資料只包含公尺制
x/y 格線、秒數 age 邊界、``(y, x)`` 計數／停留陣列與 ``(y, x, age)`` 首次進入
直方圖。測試目的，是驗證逐 trajectory shard 累加時的記憶體、dtype、溢位、時間守恆
與生命週期契約，不是重建任何 OCM 或 NWW 科學資料，也不代表正式海洋計算成果。
"""

from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest
import test_streaming_aggregation as aggregation_fixture

import lagrangian_backtracking.streaming_aggregation as aggregation_module
from lagrangian_backtracking.streaming_aggregation import (
    StreamingPathwayAccumulator,
    StreamingPathwayAggregate,
    merge_streaming_pathway_aggregates,
)


def _chunk(**overrides: object) -> StreamingPathwayAggregate:
    """建立獨立且時間守恆的 synthetic chunk，避免測試之間共享可變陣列。"""

    return aggregation_fixture._aggregate_fixture(**overrides)


def _exact_chunks() -> tuple[StreamingPathwayAggregate, StreamingPathwayAggregate]:
    """建立兩個同 topology、可精確預期逐項總和的 synthetic shard。"""

    first = _chunk(
        unique_particle_count=np.array([[1, 2]], dtype=np.int64),
        residence_time_seconds=np.array([[2.0, 1.0]], dtype=np.float64),
        first_passage_age_histogram=np.array([[[1, 0, 0], [0, 2, 0]]], dtype=np.int64),
        input_particle_count=2,
        input_interval_seconds=3.0,
        allocated_interval_seconds=3.0,
    )
    second = _chunk(
        unique_particle_count=np.array([[2, 3]], dtype=np.int64),
        residence_time_seconds=np.array([[4.0, 5.0]], dtype=np.float64),
        first_passage_age_histogram=np.array([[[0, 2, 0], [0, 0, 3]]], dtype=np.int64),
        input_particle_count=3,
        input_interval_seconds=9.0,
        allocated_interval_seconds=9.0,
    )
    return first, second


def test_accumulator_adds_chunks_exactly_and_exposes_readonly_count() -> None:
    """逐 chunk 加入後，三組產品、粒子數與兩種秒數都須得到精確總和。"""

    first, second = _exact_chunks()
    accumulator = StreamingPathwayAccumulator()

    assert accumulator.chunk_count == 0
    with pytest.raises(AttributeError):
        accumulator.chunk_count = 99  # type: ignore[misc]

    accumulator.add(first)
    assert accumulator.chunk_count == 1
    accumulator.add(second)
    merged = accumulator.finalize()

    assert accumulator.chunk_count == 2
    assert np.array_equal(merged.unique_particle_count, np.array([[3, 5]], dtype=np.int64))
    assert np.array_equal(
        merged.residence_time_seconds,
        np.array([[6.0, 6.0]], dtype=np.float64),
    )
    assert np.array_equal(
        merged.first_passage_age_histogram,
        np.array([[[1, 2, 0], [0, 2, 3]]], dtype=np.int64),
    )
    assert type(merged.input_particle_count) is int
    assert merged.input_particle_count == 5
    assert merged.input_interval_seconds == 12.0
    assert merged.allocated_interval_seconds == 12.0
    assert all(
        not array.flags.writeable
        for array in (
            merged.x_edges_m,
            merged.y_edges_m,
            merged.age_bin_edges_seconds,
            merged.unique_particle_count,
            merged.residence_time_seconds,
            merged.first_passage_age_histogram,
        )
    )


def test_accumulator_does_not_retain_old_chunk_arrays_after_add() -> None:
    """加入後釋放舊 chunk 時，reducer 不得因保存輸入 ndarray reference 而延長其生命。"""

    chunk = _chunk()
    old_residence = chunk.residence_time_seconds
    old_residence_reference = weakref.ref(old_residence)
    accumulator = StreamingPathwayAccumulator()

    accumulator.add(chunk)
    assert accumulator._residence_total is not old_residence
    assert not np.shares_memory(accumulator._residence_total, old_residence)

    del old_residence
    del chunk
    gc.collect()
    assert old_residence_reference() is None


def test_merge_consumes_one_shot_generator_and_allows_old_chunk_release() -> None:
    """merge 必須逐項消費 one-shot generator，且在取得下一 chunk 前不保存前一 chunk。"""

    def chunks():
        first = _chunk()
        first_array_reference = weakref.ref(first.residence_time_seconds)
        yield first
        # 生成器本身不再需要第一份 aggregate；第三次 next 時，merge 的 chunk 區域也
        # 已經改指向第二份，因此若實作曾以 tuple/list 保存全部輸入，此 assertion 會失敗。
        del first
        yield _chunk()
        gc.collect()
        assert first_array_reference() is None

    merged = merge_streaming_pathway_aggregates(chunks())

    assert merged.input_particle_count == 2
    assert merged.input_interval_seconds == 20.0
    assert np.array_equal(merged.unique_particle_count, np.array([[2, 2]], dtype=np.int64))


def test_merge_interleaves_iterator_next_with_accumulator_add(monkeypatch: pytest.MonkeyPatch) -> None:
    """事件順序必須是 next、add 交錯，藉此固定 merge 不得先 materialize 全部輸入。"""

    first, second = _exact_chunks()
    events: list[str] = []

    class OneShotIterator:
        """只提供逐次 next 的 iterator，沒有可供 tuple/list 取長度的額外容器。"""

        def __init__(self) -> None:
            self._index = 0

        def __iter__(self) -> OneShotIterator:
            events.append("iter")
            return self

        def __next__(self) -> StreamingPathwayAggregate:
            if self._index == 0:
                events.append("next-0")
                self._index += 1
                return first
            if self._index == 1:
                events.append("next-1")
                self._index += 1
                return second
            events.append("stop")
            raise StopIteration

    original_add = aggregation_module.StreamingPathwayAccumulator.add

    def recording_add(
        accumulator: StreamingPathwayAccumulator,
        chunk: StreamingPathwayAggregate,
    ) -> None:
        events.append("add")
        original_add(accumulator, chunk)

    monkeypatch.setattr(aggregation_module.StreamingPathwayAccumulator, "add", recording_add)
    merge_streaming_pathway_aggregates(OneShotIterator())

    assert events == ["iter", "next-0", "add", "next-1", "add", "stop"]


def test_accumulator_rejects_zero_chunk_and_stays_closed() -> None:
    """零 chunk finalize 是固定錯誤，之後 add 與再次 finalize 都不得重新開啟 reducer。"""

    accumulator = StreamingPathwayAccumulator()

    with pytest.raises(ValueError, match="零 chunk"):
        accumulator.finalize()
    with pytest.raises(ValueError, match="已關閉"):
        accumulator.add(_chunk())
    with pytest.raises(ValueError, match="已關閉"):
        accumulator.finalize()
    assert accumulator.chunk_count == 0


def test_accumulator_rejects_finalize_twice_and_add_after_finalize() -> None:
    """成功 finalize 後 reducer 必須封閉，不能再產生第二份產品或接受新 chunk。"""

    accumulator = StreamingPathwayAccumulator()
    accumulator.add(_chunk())
    result = accumulator.finalize()

    assert result.input_interval_seconds == 10.0
    with pytest.raises(ValueError, match="已關閉"):
        accumulator.finalize()
    with pytest.raises(ValueError, match="已關閉"):
        accumulator.add(_chunk())


@pytest.mark.parametrize("failure", ["edge", "shape"], ids=["edge-mismatch", "shape-mismatch"])
def test_second_chunk_edge_or_shape_failure_closes_without_partial_add(failure: str) -> None:
    """第二 chunk 的 topology 失敗後不得部分寫入，且 reducer 必須 fail-closed。"""

    first = _chunk()
    if failure == "edge":
        bad = _chunk(x_edges_m=np.array([0.0, 2.0, 4.0], dtype=np.float64))
    else:
        bad = _chunk()
        object.__setattr__(bad, "unique_particle_count", np.zeros((2, 1), dtype=np.int64))
    valid_after_failure = _chunk()
    accumulator = StreamingPathwayAccumulator()
    accumulator.add(first)
    before_unique = accumulator._unique_total.copy()
    before_residence = accumulator._residence_total.copy()
    before_histogram = accumulator._histogram_total.copy()

    with pytest.raises(ValueError):
        accumulator.add(bad)
    assert accumulator.chunk_count == 1
    assert np.array_equal(accumulator._unique_total, before_unique)
    assert np.array_equal(accumulator._residence_total, before_residence)
    assert np.array_equal(accumulator._histogram_total, before_histogram)
    with pytest.raises(ValueError, match="已關閉"):
        accumulator.add(valid_after_failure)


def test_int64_count_overflow_is_atomic_and_closes_reducer() -> None:
    """任一 int64 計數加總溢位都必須在寫入前失敗，避免 unique 或 histogram 半更新。"""

    maximum = aggregation_fixture._INT64_MAX_FOR_TEST
    first_histogram = np.zeros((1, 2, 3), dtype=np.int64)
    first_histogram[0, 0, 0] = maximum
    first = _chunk(
        unique_particle_count=np.array([[maximum, 0]], dtype=np.int64),
        first_passage_age_histogram=first_histogram,
        input_particle_count=maximum,
    )
    second_histogram = np.zeros((1, 2, 3), dtype=np.int64)
    second_histogram[0, 0, 0] = 1
    second = _chunk(
        unique_particle_count=np.array([[1, 0]], dtype=np.int64),
        first_passage_age_histogram=second_histogram,
        input_particle_count=1,
    )
    accumulator = StreamingPathwayAccumulator()
    accumulator.add(first)
    before_unique = accumulator._unique_total.copy()
    before_histogram = accumulator._histogram_total.copy()

    with pytest.raises(RuntimeError, match="int64 上限"):
        accumulator.add(second)
    assert np.array_equal(accumulator._unique_total, before_unique)
    assert np.array_equal(accumulator._histogram_total, before_histogram)
    with pytest.raises(ValueError, match="已關閉"):
        accumulator.add(_chunk())


def test_residence_float64_overflow_is_atomic_and_closes_reducer() -> None:
    """合法極大秒數相加成 inf 時，候選陣列須先被拒絕且既有狀態不可改變。"""

    maximum = np.finfo(np.float64).max
    chunk = _chunk(
        residence_time_seconds=np.array([[maximum, 0.0]], dtype=np.float64),
        input_interval_seconds=maximum,
        allocated_interval_seconds=maximum,
    )
    accumulator = StreamingPathwayAccumulator()
    accumulator.add(chunk)
    before_residence = accumulator._residence_total.copy()

    with pytest.raises(RuntimeError, match="有限 float64"):
        accumulator.add(chunk)
    assert np.array_equal(accumulator._residence_total, before_residence)
    with pytest.raises(ValueError, match="已關閉"):
        accumulator.add(_chunk())


def test_finalize_output_does_not_alias_input_chunk_or_mutable_accumulator() -> None:
    """finalize 回傳的唯讀產品不得與輸入 chunk 或 reducer 內部 mutable arrays 共用記憶體。"""

    chunk = _chunk()
    source_arrays = (
        chunk.x_edges_m,
        chunk.y_edges_m,
        chunk.age_bin_edges_seconds,
        chunk.unique_particle_count,
        chunk.residence_time_seconds,
        chunk.first_passage_age_histogram,
    )
    accumulator = StreamingPathwayAccumulator()
    accumulator.add(chunk)
    accumulator_arrays = (
        accumulator._x_edges,
        accumulator._y_edges,
        accumulator._age_edges,
        accumulator._unique_total,
        accumulator._residence_total,
        accumulator._histogram_total,
    )
    result = accumulator.finalize()
    result_arrays = (
        result.x_edges_m,
        result.y_edges_m,
        result.age_bin_edges_seconds,
        result.unique_particle_count,
        result.residence_time_seconds,
        result.first_passage_age_histogram,
    )

    assert all(
        not np.shares_memory(result_array, source_array)
        for result_array, source_array in zip(result_arrays, source_arrays, strict=True)
    )
    assert all(
        not np.shares_memory(result_array, accumulator_array)
        for result_array, accumulator_array in zip(result_arrays, accumulator_arrays, strict=True)
    )


def test_finalize_rechecks_merged_time_and_residence_conservation() -> None:
    """finalize 會重新檢查合併 scalar 與 residence 總和，失敗後仍維持封閉狀態。"""

    accumulator = StreamingPathwayAccumulator()
    accumulator.add(_chunk())
    accumulator._allocated_interval_seconds = 11.0

    with pytest.raises(RuntimeError, match="未守恆"):
        accumulator.finalize()
    with pytest.raises(ValueError, match="已關閉"):
        accumulator.finalize()


def test_merge_iterator_creation_and_next_failures_are_value_errors() -> None:
    """iter() 或 next() 的輸入迭代器故障須轉成 merge 的 ValueError，而非洩漏實作例外。"""

    class BrokenIterable:
        def __iter__(self):
            raise RuntimeError("synthetic iterator construction failure")

    class BrokenIterator:
        def __iter__(self):
            return self

        def __next__(self):
            raise RuntimeError("synthetic iterator next failure")

    with pytest.raises(ValueError, match="可迭代"):
        merge_streaming_pathway_aggregates(BrokenIterable())
    with pytest.raises(ValueError, match="iterator"):
        merge_streaming_pathway_aggregates(BrokenIterator())
