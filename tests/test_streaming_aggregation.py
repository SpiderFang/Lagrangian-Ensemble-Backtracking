"""串流路徑、停留時間與首次通過年齡分布的公開 API 契約測試。

本檔只使用串流聚合模組的公開資料類別與函式，以及引擎公開的
ParticleResult／Observation 資料類別；所有軌跡均為公尺、秒與世界協調時間（UTC）
奈秒組成的小型人工 fixture，不代表任何正式海洋資料。測試特別固定
二維陣列的軸順序為（y cell、x cell），三維首次通過 histogram 的軸順序為
（y cell、x cell、age bin），以防止未來以矩陣轉置或錯誤分箱掩蓋物理計算錯誤。
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from typing import Any

import numpy as np
import pytest

from lagrangian_backtracking.engine import Observation, ParticleResult
from lagrangian_backtracking.models import ParticleState, ParticleStatus
from lagrangian_backtracking.streaming_aggregation import (
    StreamingPathwayAggregate,
    first_passage_quantiles,
    merge_streaming_pathway_aggregates,
    stream_pathway_first_passage,
)

_BASE_TIME_UTC_NS = 10_000_000_000
_INT64_MAX_FOR_TEST = int(np.iinfo(np.int64).max)


def _observation(
    particle_id: str,
    age_seconds: float,
    x_m: float,
    y_m: float,
    *,
    z_m: float = -1.0,
    status: ParticleStatus = ParticleStatus.ACTIVE,
    time_utc_ns: int | None = None,
) -> Observation:
    """建立一筆小型逆向軌跡觀測，並以年齡推導嚴格遞減的 backward UTC。

    年齡是由受體向過去累積的秒數；正常 fixture 以固定的 UTC 基準減去年齡，
    使相鄰觀測同時符合 age 嚴格遞增與 backward UTC 嚴格遞減。非法數值測試
    可以明確傳入 time_utc_ns，避免在建立非有限年齡時先被 fixture 計算擋住。
    """

    if time_utc_ns is None:
        if np.isfinite(age_seconds):
            time_utc_ns = _BASE_TIME_UTC_NS - int(round(age_seconds * 1_000_000.0))
        else:
            time_utc_ns = _BASE_TIME_UTC_NS
    return Observation(
        particle_id=particle_id,
        time_utc_ns=time_utc_ns,
        age_seconds=age_seconds,
        x_m=x_m,
        y_m=y_m,
        z_m=z_m,
        status=status,
    )


def _result(
    particle_id: str,
    observations: list[Observation],
    *,
    final_status: ParticleStatus | None = None,
    state_particle_id: str | None = None,
) -> ParticleResult:
    """把 observation 序列包成與末筆資料對齊的最小 ParticleResult。

    final_state 保存聚合器需要核對的粒子身分、末端位置、UTC、年齡與停止狀態；
    events、step_count 與 minimum_clamp_count 對本組純軌跡聚合測試沒有作用，因此
    使用空事件與零計數，避免把邊界事件或引擎執行細節混入測試焦點。
    """

    if observations:
        last = observations[-1]
    else:
        last = _observation(
            particle_id,
            0.0,
            0.25,
            0.25,
            status=ParticleStatus.ACTIVE,
        )
    status = last.status if final_status is None else final_status
    state = ParticleState(
        particle_id=particle_id if state_particle_id is None else state_particle_id,
        scenario_id="scenario-fixture",
        member_id=0,
        study_site_id="site-fixture",
        analysis_region_id="region-fixture",
        receptor_id="receptor-fixture",
        x_m=last.x_m,
        y_m=last.y_m,
        z_m=last.z_m,
        time_utc_ns=last.time_utc_ns,
        age_seconds=last.age_seconds,
        status=status,
    )
    return ParticleResult(
        final_state=state,
        observations=list(observations),
        events=[],
        step_count=0,
        minimum_clamp_count=0,
    )


def _edges() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """回傳每次呼叫都獨立的 2×1 公尺格網與秒數 age bins。"""

    return (
        np.array([0.0, 1.0, 2.0], dtype=np.float64),
        np.array([0.0, 1.0], dtype=np.float64),
        np.array([0.0, 5.0, 10.0, 15.0], dtype=np.float64),
    )


def _stream(
    results: Any,
    *,
    x_edges_m: np.ndarray | None = None,
    y_edges_m: np.ndarray | None = None,
    age_bin_edges_seconds: np.ndarray | None = None,
) -> StreamingPathwayAggregate:
    """以合法預設格線呼叫公開串流聚合入口，讓各測試只替換一個條件。"""

    default_x, default_y, default_age = _edges()
    return stream_pathway_first_passage(
        results,
        x_edges_m=default_x if x_edges_m is None else x_edges_m,
        y_edges_m=default_y if y_edges_m is None else y_edges_m,
        age_bin_edges_seconds=default_age if age_bin_edges_seconds is None else age_bin_edges_seconds,
    )


def _aggregate_fixture(
    *,
    x_edges_m: object | None = None,
    y_edges_m: object | None = None,
    age_bin_edges_seconds: object | None = None,
    unique_particle_count: object | None = None,
    residence_time_seconds: object | None = None,
    first_passage_age_histogram: object | None = None,
    input_particle_count: object = 1,
    input_interval_seconds: object = 10.0,
    allocated_interval_seconds: object = 10.0,
) -> StreamingPathwayAggregate:
    """建立一份形狀固定且時間守恆的直接聚合 fixture，供 merge 契約測試重用。

    預設格網含一列 y cell、兩欄 x cell 與三個 age bin；residence 的元素單位是秒，
    計數陣列軸則分別固定為（y、x）與（y、x、age），且每格 age 計數總和等於 unique
    粒子數。呼叫端可只替換一個欄位，讓非法 dtype、shape、軸順序或 scalar 測試不必
    複製其他合法資料。
    """

    default_x, default_y, default_age = _edges()
    return StreamingPathwayAggregate(
        x_edges_m=default_x if x_edges_m is None else x_edges_m,
        y_edges_m=default_y if y_edges_m is None else y_edges_m,
        age_bin_edges_seconds=default_age if age_bin_edges_seconds is None else age_bin_edges_seconds,
        unique_particle_count=(
            np.array([[1, 1]], dtype=np.int64)
            if unique_particle_count is None
            else unique_particle_count
        ),
        residence_time_seconds=(
            np.array([[6.0, 4.0]], dtype=np.float64)
            if residence_time_seconds is None
            else residence_time_seconds
        ),
        first_passage_age_histogram=(
            np.array([[[1, 0, 0], [0, 1, 0]]], dtype=np.int64)
            if first_passage_age_histogram is None
            else first_passage_age_histogram
        ),
        input_particle_count=input_particle_count,
        input_interval_seconds=input_interval_seconds,
        allocated_interval_seconds=allocated_interval_seconds,
    )


def _straight_result(particle_id: str = "p-straight") -> ParticleResult:
    """建立從左 cell 直線跨入右 cell 的十秒軌跡。"""

    return _result(
        particle_id,
        [
            _observation(particle_id, 0.0, 0.25, 0.5),
            _observation(
                particle_id,
                10.0,
                1.75,
                0.5,
                status=ParticleStatus.MAX_AGE,
            ),
        ],
    )


def test_straight_crossing_splits_residence_and_first_passage_bins() -> None:
    """直線十秒跨越兩個 x cell 時，停留秒數與首次通過年齡須各分配五秒。"""

    aggregate = _stream([_straight_result()])

    assert aggregate.input_particle_count == 1
    assert np.array_equal(aggregate.unique_particle_count, np.array([[1, 1]], dtype=np.int64))
    assert np.allclose(aggregate.residence_time_seconds, np.array([[5.0, 5.0]]))
    assert np.array_equal(
        aggregate.first_passage_age_histogram,
        np.array([[[1, 0, 0], [0, 1, 0]]], dtype=np.int64),
    )
    assert aggregate.input_interval_seconds == pytest.approx(10.0)
    assert aggregate.allocated_interval_seconds == pytest.approx(10.0)
    assert aggregate.first_passage_age_histogram.dtype == np.int64


def test_repeated_backtracking_visits_are_unique_once_but_residence_accumulates() -> None:
    """同一粒子往返並再次停留時，unique 只計一次而 residence 與最早 passage 分開保存。"""

    result = _result(
        "p-loop",
        [
            _observation("p-loop", 0.0, 0.25, 0.5),
            _observation("p-loop", 2.0, 1.75, 0.5),
            _observation("p-loop", 4.0, 0.25, 0.5),
            _observation("p-loop", 6.0, 0.25, 0.5, status=ParticleStatus.MAX_AGE),
        ],
    )
    aggregate = _stream(
        [result],
        age_bin_edges_seconds=np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
    )

    assert np.array_equal(aggregate.unique_particle_count, np.array([[1, 1]], dtype=np.int64))
    assert np.allclose(aggregate.residence_time_seconds, np.array([[4.0, 2.0]]))
    assert np.array_equal(
        aggregate.first_passage_age_histogram,
        np.array([[[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0]]], dtype=np.int64),
    )
    assert aggregate.input_interval_seconds == pytest.approx(6.0)
    assert aggregate.allocated_interval_seconds == pytest.approx(6.0)


def test_single_observation_zero_displacement_and_shard_addition_are_associative() -> None:
    """兩粒子含單 observation 與零位移 interval 時，逐 shard 相加須等於整批聚合。"""

    x_edges = np.array([0.0, 1.0, 2.0])
    y_edges = np.array([0.0, 1.0, 2.0])
    age_edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    single = _result(
        "p-single",
        [_observation("p-single", 0.0, 0.25, 0.25, status=ParticleStatus.MAX_AGE)],
    )
    zero_displacement = _result(
        "p-zero",
        [
            _observation("p-zero", 0.0, 1.25, 1.25),
            _observation(
                "p-zero",
                3.0,
                1.25,
                1.25,
                status=ParticleStatus.MAX_AGE,
            ),
        ],
    )

    shard_a = _stream(
        (item for item in [single]),
        x_edges_m=x_edges,
        y_edges_m=y_edges,
        age_bin_edges_seconds=age_edges,
    )
    shard_b = _stream(
        [zero_displacement],
        x_edges_m=x_edges,
        y_edges_m=y_edges,
        age_bin_edges_seconds=age_edges,
    )
    full = _stream(
        [single, zero_displacement],
        x_edges_m=x_edges,
        y_edges_m=y_edges,
        age_bin_edges_seconds=age_edges,
    )

    assert np.array_equal(
        full.unique_particle_count,
        shard_a.unique_particle_count + shard_b.unique_particle_count,
    )
    assert np.allclose(
        full.residence_time_seconds,
        shard_a.residence_time_seconds + shard_b.residence_time_seconds,
    )
    assert np.array_equal(
        full.first_passage_age_histogram,
        shard_a.first_passage_age_histogram + shard_b.first_passage_age_histogram,
    )
    assert full.input_particle_count == shard_a.input_particle_count + shard_b.input_particle_count == 2
    assert full.input_interval_seconds == pytest.approx(
        shard_a.input_interval_seconds + shard_b.input_interval_seconds
    )
    assert full.allocated_interval_seconds == pytest.approx(
        shard_a.allocated_interval_seconds + shard_b.allocated_interval_seconds
    )
    assert np.array_equal(full.unique_particle_count, np.array([[1, 0], [0, 1]], dtype=np.int64))
    assert np.allclose(full.residence_time_seconds, np.array([[0.0, 0.0], [0.0, 3.0]]))


def test_zero_displacement_uses_y_then_x_axis_order_off_diagonal() -> None:
    """非對角零位移區段須寫入（iy, ix），不可把 x/y cell 索引轉置。

    粒子固定在第二個 x cell、第一個 y cell，兩秒區段因此只應累積到陣列位置
    （0, 1）。此案例刻意避開對角 cell，讓 residence、unique 與首次通過 histogram
    任一處誤用（ix, iy）時都會在轉置位置留下可觀察的錯誤。
    """

    result = _result(
        "p-zero-off-diagonal",
        [
            _observation("p-zero-off-diagonal", 0.0, 1.25, 0.25),
            _observation(
                "p-zero-off-diagonal",
                2.0,
                1.25,
                0.25,
                status=ParticleStatus.MAX_AGE,
            ),
        ],
    )
    aggregate = _stream(
        [result],
        x_edges_m=np.array([0.0, 1.0, 2.0]),
        y_edges_m=np.array([0.0, 1.0, 2.0]),
        age_bin_edges_seconds=np.array([0.0, 1.0, 2.0, 3.0]),
    )

    assert np.array_equal(
        aggregate.unique_particle_count,
        np.array([[0, 1], [0, 0]], dtype=np.int64),
    )
    assert np.allclose(
        aggregate.residence_time_seconds,
        np.array([[0.0, 2.0], [0.0, 0.0]]),
    )
    assert np.array_equal(
        aggregate.first_passage_age_histogram,
        np.array(
            [
                [[0, 0, 0], [1, 0, 0]],
                [[0, 0, 0], [0, 0, 0]],
            ],
            dtype=np.int64,
        ),
    )
    assert aggregate.input_interval_seconds == pytest.approx(2.0)
    assert aggregate.allocated_interval_seconds == pytest.approx(2.0)


def test_non_square_grid_and_closed_upper_right_boundary_use_metric_cells() -> None:
    """非正方形格網依各自 x/y 寬度切段，最右與最上邊界歸入最後 cell。"""

    x_edges = np.array([0.0, 2.0, 5.0])
    y_edges = np.array([0.0, 4.0, 10.0])
    age_edges = np.array([0.0, 3.0, 6.0, 9.0, 12.0, 15.0])
    result = _result(
        "p-rectangle",
        [
            _observation("p-rectangle", 0.0, 0.5, 2.0),
            _observation(
                "p-rectangle",
                12.0,
                5.0,
                10.0,
                status=ParticleStatus.MAX_AGE,
            ),
        ],
    )

    aggregate = _stream(
        [result],
        x_edges_m=x_edges,
        y_edges_m=y_edges,
        age_bin_edges_seconds=age_edges,
    )

    assert np.array_equal(
        aggregate.unique_particle_count,
        np.array([[1, 0], [1, 1]], dtype=np.int64),
    )
    assert np.allclose(
        aggregate.residence_time_seconds,
        np.array([[3.0, 0.0], [1.0, 8.0]]),
    )
    assert np.array_equal(
        aggregate.first_passage_age_histogram,
        np.array(
            [
                [[1, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                [[0, 1, 0, 0, 0], [0, 1, 0, 0, 0]],
            ],
            dtype=np.int64,
        ),
    )
    assert aggregate.input_interval_seconds == pytest.approx(12.0)
    assert aggregate.allocated_interval_seconds == pytest.approx(12.0)


def test_aggregate_is_frozen_slotted_and_all_arrays_are_read_only_defensive_copies() -> None:
    """聚合結果須為 frozen/slots，且 caller 修改輸入或輸出陣列都不能污染產品。"""

    x_edges, y_edges, age_edges = _edges()
    original_x = x_edges.copy()
    original_y = y_edges.copy()
    original_age = age_edges.copy()
    aggregate = _stream(
        [_straight_result()],
        x_edges_m=x_edges,
        y_edges_m=y_edges,
        age_bin_edges_seconds=age_edges,
    )

    x_edges[:] = -99.0
    y_edges[:] = -99.0
    age_edges[:] = -99.0
    assert np.array_equal(aggregate.x_edges_m, original_x)
    assert np.array_equal(aggregate.y_edges_m, original_y)
    assert np.array_equal(aggregate.age_bin_edges_seconds, original_age)
    assert not np.shares_memory(aggregate.x_edges_m, x_edges)
    assert not np.shares_memory(aggregate.y_edges_m, y_edges)
    assert not np.shares_memory(aggregate.age_bin_edges_seconds, age_edges)

    arrays = (
        aggregate.x_edges_m,
        aggregate.y_edges_m,
        aggregate.age_bin_edges_seconds,
        aggregate.unique_particle_count,
        aggregate.residence_time_seconds,
        aggregate.first_passage_age_histogram,
    )
    assert all(not array.flags.writeable for array in arrays)
    for array in arrays:
        with pytest.raises(ValueError):
            array.flat[0] = 123

    assert not hasattr(aggregate, "__dict__")
    with pytest.raises(FrozenInstanceError):
        aggregate.input_particle_count = 99


def test_merge_two_valid_chunks_exactly_sums_all_products_and_scalars() -> None:
    """兩個同格網 shard 合併時，三種陣列與 Python scalar 都必須逐項得到精確總和。"""

    first = _aggregate_fixture(
        unique_particle_count=np.array([[1, 2]], dtype=np.int64),
        residence_time_seconds=np.array([[2.0, 1.0]], dtype=np.float64),
        first_passage_age_histogram=np.array([[[1, 0, 0], [0, 2, 0]]], dtype=np.int64),
        input_particle_count=2,
        input_interval_seconds=3.0,
        allocated_interval_seconds=3.0,
    )
    second = _aggregate_fixture(
        unique_particle_count=np.array([[2, 3]], dtype=np.int64),
        residence_time_seconds=np.array([[4.0, 5.0]], dtype=np.float64),
        first_passage_age_histogram=np.array([[[0, 2, 0], [0, 0, 3]]], dtype=np.int64),
        input_particle_count=3,
        input_interval_seconds=9.0,
        allocated_interval_seconds=9.0,
    )

    merged = merge_streaming_pathway_aggregates(chunk for chunk in (first, second))

    assert np.array_equal(merged.unique_particle_count, [[3, 5]])
    assert np.array_equal(
        merged.first_passage_age_histogram,
        np.array([[[1, 2, 0], [0, 2, 3]]], dtype=np.int64),
    )
    assert np.array_equal(merged.residence_time_seconds, [[6.0, 6.0]])
    assert type(merged.input_particle_count) is int
    assert merged.input_particle_count == 5
    assert merged.input_interval_seconds == 12.0
    assert merged.allocated_interval_seconds == 12.0


def test_merge_single_chunk_defensively_copies_and_does_not_modify_input() -> None:
    """單 shard 合併仍須產生獨立唯讀結果，且不能改寫輸入聚合或其原始建構陣列。"""

    source_unique = np.array([[1, 1]], dtype=np.int64)
    source_residence = np.array([[6.0, 4.0]], dtype=np.float64)
    source_histogram = np.array([[[1, 0, 0], [0, 1, 0]]], dtype=np.int64)
    source_edges = _edges()
    chunk = StreamingPathwayAggregate(
        x_edges_m=source_edges[0],
        y_edges_m=source_edges[1],
        age_bin_edges_seconds=source_edges[2],
        unique_particle_count=source_unique,
        residence_time_seconds=source_residence,
        first_passage_age_histogram=source_histogram,
        input_particle_count=1,
        input_interval_seconds=10.0,
        allocated_interval_seconds=10.0,
    )
    merged = merge_streaming_pathway_aggregates([chunk])

    expected_arrays = (
        chunk.x_edges_m.copy(),
        chunk.y_edges_m.copy(),
        chunk.age_bin_edges_seconds.copy(),
        chunk.unique_particle_count.copy(),
        chunk.residence_time_seconds.copy(),
        chunk.first_passage_age_histogram.copy(),
    )
    source_unique[:] = 99
    source_residence[:] = 99.0
    source_histogram[:] = 99
    for actual, expected in zip(
        (
            chunk.x_edges_m,
            chunk.y_edges_m,
            chunk.age_bin_edges_seconds,
            chunk.unique_particle_count,
            chunk.residence_time_seconds,
            chunk.first_passage_age_histogram,
        ),
        expected_arrays,
        strict=True,
    ):
        assert np.array_equal(actual, expected)
    assert np.array_equal(merged.unique_particle_count, expected_arrays[3])
    assert np.array_equal(merged.residence_time_seconds, expected_arrays[4])
    assert np.array_equal(merged.first_passage_age_histogram, expected_arrays[5])
    assert all(
        not np.shares_memory(actual, original)
        for actual, original in zip(
            (
                merged.x_edges_m,
                merged.y_edges_m,
                merged.age_bin_edges_seconds,
                merged.unique_particle_count,
                merged.residence_time_seconds,
                merged.first_passage_age_histogram,
            ),
            (
                chunk.x_edges_m,
                chunk.y_edges_m,
                chunk.age_bin_edges_seconds,
                chunk.unique_particle_count,
                chunk.residence_time_seconds,
                chunk.first_passage_age_histogram,
            ),
            strict=True,
        )
    )
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


@pytest.mark.parametrize(
    "chunks",
    [(), [], None, [object()], ["not-an-aggregate"]],
    ids=["empty-tuple", "empty-list", "not-materializable", "wrong-object", "wrong-string"],
)
def test_merge_rejects_empty_or_wrong_chunk_inputs(chunks: Any) -> None:
    """merge 只接受非空且每一項都是 StreamingPathwayAggregate 的可 materialize 輸入。"""

    with pytest.raises(ValueError):
        merge_streaming_pathway_aggregates(chunks)


@pytest.mark.parametrize("axis", ["x", "y", "age"])
def test_merge_rejects_different_grid_or_age_axis_definitions(axis: str) -> None:
    """不同 x、y 或 age 邊界即使陣列形狀相同，也不得跨 shard 混合。"""

    first = _aggregate_fixture()
    changed_edges: dict[str, np.ndarray] = {
        "x": np.array([0.0, 2.0, 4.0]),
        "y": np.array([0.0, 2.0]),
        "age": np.array([0.0, 4.0, 10.0, 15.0]),
    }
    second = _aggregate_fixture(
        x_edges_m=changed_edges["x"] if axis == "x" else None,
        y_edges_m=changed_edges["y"] if axis == "y" else None,
        age_bin_edges_seconds=changed_edges["age"] if axis == "age" else None,
    )

    with pytest.raises(ValueError, match="不完全相同"):
        merge_streaming_pathway_aggregates([first, second])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("unique_particle_count", np.zeros((2, 1), dtype=np.int64)),
        ("residence_time_seconds", np.zeros((1, 1), dtype=np.float64)),
        ("first_passage_age_histogram", np.zeros((1, 2), dtype=np.int64)),
        ("first_passage_age_histogram", np.zeros((1, 2, 4), dtype=np.int64)),
    ],
    ids=["unique-axis-swapped", "residence-shape", "hist-missing-age-axis", "hist-age-axis-long"],
)
def test_aggregate_rejects_shape_and_age_axis_mismatches(field: str, value: np.ndarray) -> None:
    """直接建構時若 y/x 或 age 軸不符合格線推導形狀，必須立即拒絕。"""

    with pytest.raises(ValueError):
        replace(_aggregate_fixture(), **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("unique_particle_count", np.array([[-1, 0]], dtype=np.int64)),
        ("unique_particle_count", np.array([[1.0, 0.0]], dtype=np.float64)),
        ("unique_particle_count", np.array([[True, False]], dtype=bool)),
        (
            "unique_particle_count",
            np.array([[_INT64_MAX_FOR_TEST + 1, 0]], dtype=np.uint64),
        ),
        ("first_passage_age_histogram", np.array([[[-1, 0, 0], [0, 0, 0]]], dtype=np.int64)),
        (
            "first_passage_age_histogram",
            np.ones((1, 2, 3), dtype=np.float64),
        ),
        (
            "first_passage_age_histogram",
            np.full((1, 2, 3), _INT64_MAX_FOR_TEST + 1, dtype=np.uint64),
        ),
    ],
    ids=[
        "unique-negative",
        "unique-float",
        "unique-bool",
        "unique-uint-overflow",
        "hist-negative",
        "hist-float",
        "hist-uint-overflow",
    ],
)
def test_aggregate_rejects_invalid_count_dtype_or_value(field: str, value: np.ndarray) -> None:
    """unique 與 histogram 只接受非負且可安全轉成 int64 的整數計數。"""

    with pytest.raises(ValueError):
        replace(_aggregate_fixture(), **{field: value})


@pytest.mark.parametrize(
    "residence",
    [
        np.array([[np.nan, 0.0]], dtype=np.float64),
        np.array([[np.inf, 0.0]], dtype=np.float64),
        np.array([[-1.0, 11.0]], dtype=np.float64),
    ],
    ids=["nan", "infinity", "negative"],
)
def test_aggregate_rejects_nonfinite_or_negative_residence(residence: np.ndarray) -> None:
    """停留時間是秒數，任何非有限或負值都不能進入聚合產品。"""

    with pytest.raises(ValueError):
        replace(_aggregate_fixture(), residence_time_seconds=residence)


def test_merge_rejects_residence_float64_overflow() -> None:
    """兩個各自合法的極大有限 residence shard 相加溢位時必須明確失敗。"""

    maximum = np.finfo(np.float64).max
    chunk = _aggregate_fixture(
        residence_time_seconds=np.array([[maximum, 0.0]], dtype=np.float64),
        input_interval_seconds=maximum,
        allocated_interval_seconds=maximum,
    )

    with pytest.raises(RuntimeError, match="有限 float64"):
        merge_streaming_pathway_aggregates([chunk, chunk])


def test_merge_rejects_int64_count_sum_overflow() -> None:
    """兩份各自合法的 int64 計數在逐元素合併超過上限時不得 NumPy 繞回。"""

    maximum = _INT64_MAX_FOR_TEST
    first_histogram = np.zeros((1, 2, 3), dtype=np.int64)
    first_histogram[0, 0, 0] = maximum
    second_histogram = np.zeros((1, 2, 3), dtype=np.int64)
    second_histogram[0, 0, 0] = 1
    first = _aggregate_fixture(
        unique_particle_count=np.array([[maximum, 0]], dtype=np.int64),
        first_passage_age_histogram=first_histogram,
        input_particle_count=maximum,
    )
    second = _aggregate_fixture(
        unique_particle_count=np.array([[1, 0]], dtype=np.int64),
        first_passage_age_histogram=second_histogram,
        input_particle_count=1,
    )

    with pytest.raises(RuntimeError, match="int64 上限"):
        merge_streaming_pathway_aggregates([first, second])


def test_aggregate_rejects_first_passage_and_unique_count_mismatch() -> None:
    """每格首次進入 age histogram 總數若不等於 unique 粒子數，必須立即拒絕。"""

    mismatched_histogram = np.array([[[1, 0, 0], [0, 0, 0]]], dtype=np.int64)

    with pytest.raises(ValueError, match="精確等於 unique_particle_count"):
        replace(
            _aggregate_fixture(),
            first_passage_age_histogram=mismatched_histogram,
        )


def test_aggregate_rejects_unique_count_above_input_particle_denominator() -> None:
    """任一格 unique 粒子數不得超過該 shard 的輸入粒子分母。"""

    unique_count = np.array([[2, 1]], dtype=np.int64)
    histogram = np.array([[[2, 0, 0], [0, 1, 0]]], dtype=np.int64)

    with pytest.raises(ValueError, match="不可大於 input_particle_count"):
        replace(
            _aggregate_fixture(),
            unique_particle_count=unique_count,
            first_passage_age_histogram=histogram,
            input_particle_count=1,
        )


def test_aggregate_rejects_python_int_age_axis_sum_above_int64() -> None:
    """age bins 各自合法但 Python int 總和超過 int64 時，不得在 NumPy 中繞回。"""

    maximum = _INT64_MAX_FOR_TEST
    histogram = np.zeros((1, 2, 3), dtype=np.int64)
    histogram[0, 0, 0] = maximum
    histogram[0, 0, 1] = 1

    with pytest.raises(ValueError, match="沿 age 軸合計超過 int64 上限"):
        replace(
            _aggregate_fixture(),
            unique_particle_count=np.array([[maximum, 0]], dtype=np.int64),
            first_passage_age_histogram=histogram,
            input_particle_count=maximum,
        )


@pytest.mark.parametrize(
    "value",
    [True, np.int64(1), 1.0, -1],
    ids=["bool", "numpy-int", "float", "negative"],
)
def test_aggregate_requires_nonnegative_python_int_particle_count(value: Any) -> None:
    """粒子數是分母，必須保存為非負真正 Python int，不能混入 bool 或 NumPy scalar。"""

    with pytest.raises(ValueError):
        replace(_aggregate_fixture(), input_particle_count=value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("input_interval_seconds", True),
        ("input_interval_seconds", np.array([10.0])),
        ("input_interval_seconds", np.nan),
        ("input_interval_seconds", np.inf),
        ("input_interval_seconds", -1.0),
        ("allocated_interval_seconds", "10.0"),
        ("allocated_interval_seconds", np.array([10.0])),
        ("allocated_interval_seconds", np.nan),
        ("allocated_interval_seconds", np.inf),
        ("allocated_interval_seconds", -1.0),
    ],
    ids=[
        "input-bool",
        "input-array",
        "input-nan",
        "input-infinity",
        "input-negative",
        "allocated-string",
        "allocated-array",
        "allocated-nan",
        "allocated-infinity",
        "allocated-negative",
    ],
)
def test_aggregate_rejects_invalid_interval_scalars(field: str, value: Any) -> None:
    """輸入與分配 interval 只接受有限非負秒數 scalar，拒絕陣列與非數值型別。"""

    with pytest.raises(ValueError):
        replace(_aggregate_fixture(), **{field: value})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_interval_seconds": 9.0},
        {"allocated_interval_seconds": 9.0},
        {"residence_time_seconds": np.array([[9.0, 0.0]], dtype=np.float64)},
    ],
    ids=["input-versus-allocated", "allocated-versus-input", "residence-versus-allocated"],
)
def test_aggregate_rejects_interval_or_residence_time_nonconservation(kwargs: dict[str, Any]) -> None:
    """直接建構與 merge 都不能以不守恆的秒數建立聚合結果。"""

    with pytest.raises(RuntimeError, match="守恆"):
        replace(_aggregate_fixture(), **kwargs)
def test_quantiles_use_bin_midpoints_preserve_requested_order_and_mark_empty_cells_nan() -> None:
    """首次通過 histogram 的 q25/q50/q75 以 bin midpoint 近似，空 cell 回傳 NaN。"""

    hist = np.array(
        [
            [[1, 1, 2, 0], [0, 0, 0, 0]],
            [[0, 2, 0, 2], [0, 0, 1, 3]],
        ],
        dtype=np.int64,
    )
    age_edges = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    quantiles = first_passage_quantiles(
        hist,
        age_bin_edges_seconds=age_edges,
        quantiles=(0.75, 0.25, 0.5),
    )

    assert list(quantiles) == [0.75, 0.25, 0.5]
    assert np.allclose(
        quantiles[0.25],
        np.array([[5.0, np.nan], [15.0, 25.0]]),
        equal_nan=True,
    )
    assert np.allclose(
        quantiles[0.5],
        np.array([[15.0, np.nan], [15.0, 35.0]]),
        equal_nan=True,
    )
    assert np.allclose(
        quantiles[0.75],
        np.array([[25.0, np.nan], [35.0, 35.0]]),
        equal_nan=True,
    )
    assert all(not array.flags.writeable for array in quantiles.values())
    assert all(not np.shares_memory(array, hist) for array in quantiles.values())
    hist[:] = 99
    age_edges[:] = -1.0
    assert quantiles[0.25][0, 0] == pytest.approx(5.0)


@pytest.mark.parametrize(
    "bad_results",
    [
        [],
        [_result("p-empty", [])],
        [
            _straight_result("p-duplicate"),
            _straight_result("p-duplicate"),
        ],
    ],
    ids=["empty-results", "empty-observations", "duplicate-particle"],
)
def test_rejects_empty_or_duplicate_particle_inputs(bad_results: list[ParticleResult]) -> None:
    """空結果、空 observation 與同一 call 重複 particle_id 均不得靜默聚合。"""

    with pytest.raises(ValueError):
        _stream(bad_results)


def test_rejects_observation_and_final_state_particle_identity_mismatch() -> None:
    """觀測列與 final state 的 particle identity 不一致時必須立即拒絕。"""

    observation_mismatch = _result(
        "p-state",
        [
            _observation("p-other", 0.0, 0.25, 0.5, status=ParticleStatus.MAX_AGE),
        ],
    )
    final_state_mismatch = _result(
        "p-observation",
        [
            _observation("p-observation", 0.0, 0.25, 0.5, status=ParticleStatus.MAX_AGE),
        ],
        state_particle_id="p-other",
    )

    with pytest.raises(ValueError):
        _stream([observation_mismatch])
    with pytest.raises(ValueError):
        _stream([final_state_mismatch])


@pytest.mark.parametrize(
    "ages",
    [(0.0, 0.0), (1.0, 0.0), (-1.0, 0.0)],
    ids=["equal-age", "decreasing-age", "negative-age"],
)
def test_rejects_non_increasing_or_negative_observation_age(ages: tuple[float, float]) -> None:
    """觀測 age 必須有限、非負且嚴格遞增，不能把重複或負年齡當成零秒區段。"""

    result = _result(
        "p-bad-age",
        [
            _observation("p-bad-age", ages[0], 0.25, 0.5, time_utc_ns=100),
            _observation(
                "p-bad-age",
                ages[1],
                0.75,
                0.5,
                status=ParticleStatus.MAX_AGE,
                time_utc_ns=90,
            ),
        ],
    )

    with pytest.raises(ValueError):
        _stream([result])


@pytest.mark.parametrize(
    "times",
    [(100, 100), (100, 101)],
    ids=["equal-backward-utc", "increasing-backward-utc"],
)
def test_rejects_non_decreasing_backward_utc(times: tuple[int, int]) -> None:
    """逆向軌跡的 UTC 奈秒必須嚴格遞減，禁止同時刻或向未來回跳。"""

    result = _result(
        "p-bad-time",
        [
            _observation("p-bad-time", 0.0, 0.25, 0.5, time_utc_ns=times[0]),
            _observation(
                "p-bad-time",
                1.0,
                0.75,
                0.5,
                status=ParticleStatus.MAX_AGE,
                time_utc_ns=times[1],
            ),
        ],
    )

    with pytest.raises(ValueError):
        _stream([result])


def test_rejects_final_status_mismatch() -> None:
    """最後 observation 的 status 必須與 ParticleResult.final_state.status 完全相同。"""

    result = _result(
        "p-bad-status",
        [
            _observation("p-bad-status", 0.0, 0.25, 0.5),
            _observation(
                "p-bad-status",
                1.0,
                0.75,
                0.5,
                status=ParticleStatus.MAX_AGE,
            ),
        ],
        final_status=ParticleStatus.FLOW_DOMAIN_EXIT,
    )

    with pytest.raises(ValueError):
        _stream([result])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("x_m", np.nan),
        ("x_m", np.inf),
        ("y_m", np.nan),
        ("y_m", -np.inf),
        ("z_m", np.nan),
        ("z_m", np.inf),
        ("age_seconds", np.nan),
        ("age_seconds", np.inf),
    ],
    ids=[
        "x-nan",
        "x-infinity",
        "y-nan",
        "y-negative-infinity",
        "z-nan",
        "z-infinity",
        "age-nan",
        "age-infinity",
    ],
)
def test_rejects_nonfinite_xyz_or_age(field: str, value: float) -> None:
    """x/y/z 公尺與 age 秒只接受有限數值，缺值不可轉成格網邊界或零值。"""

    observations = [
        _observation("p-nonfinite", 0.0, 0.25, 0.5, time_utc_ns=100),
        _observation(
            "p-nonfinite",
            1.0,
            0.75,
            0.5,
            status=ParticleStatus.MAX_AGE,
            time_utc_ns=90,
        ),
    ]
    observations[0] = replace(observations[0], **{field: value})

    with pytest.raises(ValueError):
        _stream([_result("p-nonfinite", observations)])


@pytest.mark.parametrize(
    ("x_m", "y_m"),
    [
        (-1.0e-9, 0.5),
        (2.0 + 1.0e-9, 0.5),
        (0.5, -1.0e-9),
        (0.5, 1.0 + 1.0e-9),
    ],
    ids=["left-outside", "right-outside", "bottom-outside", "top-outside"],
)
def test_rejects_observation_outside_closed_grid_domain(x_m: float, y_m: float) -> None:
    """所有觀測都必須位於閉矩形公尺格網內，域外位置不可被裁到邊界 cell。"""

    result = _result(
        "p-outside",
        [
            _observation(
                "p-outside",
                0.0,
                x_m,
                y_m,
                status=ParticleStatus.MAX_AGE,
            ),
        ],
    )

    with pytest.raises(ValueError):
        _stream([result])


def test_rejects_observation_age_beyond_last_age_edge() -> None:
    """觀測 age 超過最後一條 age edge 時必須失敗，不得把資料 clip 到最後 bin。"""

    result = _result(
        "p-age-overflow",
        [
            _observation("p-age-overflow", 0.0, 0.25, 0.5),
            _observation(
                "p-age-overflow",
                10.1,
                0.75,
                0.5,
                status=ParticleStatus.MAX_AGE,
            ),
        ],
    )

    with pytest.raises(ValueError):
        _stream(
            [result],
            age_bin_edges_seconds=np.array([0.0, 5.0, 10.0]),
        )


@pytest.mark.parametrize(
    "axis",
    ["x", "y"],
)
@pytest.mark.parametrize(
    "invalid_edges",
    [
        np.array(0.0),
        np.array([0.0]),
        np.array([[0.0, 1.0]]),
        np.array([0.0, np.nan, 2.0]),
        np.array([0.0, np.inf, 2.0]),
        np.array([0.0, 2.0, 1.0]),
        np.array([0.0, 1.0, 1.0]),
        np.array(["zero", "one"]),
    ],
    ids=[
        "scalar",
        "too-short",
        "two-dimensional",
        "nan",
        "infinity",
        "decreasing",
        "duplicate",
        "nonnumeric",
    ],
)
def test_rejects_invalid_x_or_y_edges(axis: str, invalid_edges: np.ndarray) -> None:
    """x/y 邊界必須是至少兩點、有限、嚴格遞增的一維公尺格線。"""

    kwargs: dict[str, Any] = {"x_edges_m": _edges()[0], "y_edges_m": _edges()[1]}
    kwargs["x_edges_m" if axis == "x" else "y_edges_m"] = invalid_edges

    with pytest.raises(ValueError):
        _stream([_straight_result()], **kwargs)


@pytest.mark.parametrize(
    "invalid_edges",
    [
        np.array(0.0),
        np.array([0.0]),
        np.array([[0.0, 1.0]]),
        np.array([0.0, np.nan, 2.0]),
        np.array([0.0, np.inf, 2.0]),
        np.array([0.0, 2.0, 1.0]),
        np.array([0.0, 1.0, 1.0]),
        np.array(["zero", "one"]),
        np.array([1.0, 2.0]),
    ],
    ids=[
        "scalar",
        "too-short",
        "two-dimensional",
        "nan",
        "infinity",
        "decreasing",
        "duplicate",
        "nonnumeric",
        "first-edge-not-zero",
    ],
)
def test_rejects_invalid_age_edges(invalid_edges: np.ndarray) -> None:
    """age 邊界必須是有限嚴格遞增的一維秒數格線，且第一點必須精確為零。"""

    with pytest.raises(ValueError):
        _stream([_straight_result()], age_bin_edges_seconds=invalid_edges)


@pytest.mark.parametrize(
    "invalid_histogram",
    [
        np.ones((1, 2, 3), dtype=np.float64),
        np.ones((1, 2, 3), dtype=bool),
        np.ones((1, 2), dtype=np.int64),
        np.ones((1, 2, 3, 1), dtype=np.int64),
        np.array([[[1, -1, 0], [0, 0, 0]]], dtype=np.int64),
        np.ones((1, 2, 2), dtype=np.int64),
        np.array([[["one", "two", "three"], ["0", "0", "0"]]]),
    ],
    ids=[
        "float-dtype",
        "bool-dtype",
        "two-dimensional",
        "four-dimensional",
        "negative-count",
        "age-axis-mismatch",
        "string-dtype",
    ],
)
def test_rejects_invalid_first_passage_histogram(invalid_histogram: np.ndarray) -> None:
    """quantile histogram 必須是與 age edges 對應的三維非負整數陣列。"""

    with pytest.raises(ValueError):
        first_passage_quantiles(
            invalid_histogram,
            age_bin_edges_seconds=np.array([0.0, 1.0, 2.0, 3.0]),
        )


@pytest.mark.parametrize(
    "invalid_quantiles",
    [
        (),
        (0.25, 0.25),
        (0.0,),
        (1.0,),
        (-0.1,),
        (1.1,),
        (np.nan,),
        (np.inf,),
        (True,),
        ("0.5",),
    ],
    ids=[
        "empty",
        "duplicate",
        "zero",
        "one",
        "negative",
        "greater-than-one",
        "nan",
        "infinity",
        "bool",
        "string",
    ],
)
def test_rejects_invalid_quantile_requests(invalid_quantiles: Any) -> None:
    """quantiles 必須非空、唯一、有限且嚴格介於零與一，並拒絕布林與字串。"""

    hist = np.zeros((1, 2, 3), dtype=np.int64)
    with pytest.raises(ValueError):
        first_passage_quantiles(
            hist,
            age_bin_edges_seconds=np.array([0.0, 1.0, 2.0, 3.0]),
            quantiles=invalid_quantiles,
        )


def test_quantile_output_arrays_remain_independent_and_read_only() -> None:
    """quantile 結果必須複製輸入語意、保持 caller 指定順序，且每個陣列不可寫入。"""

    hist = np.array([[[1, 0, 0], [0, 1, 0]]], dtype=np.int64)
    age_edges = np.array([0.0, 10.0, 20.0, 30.0])
    output = first_passage_quantiles(
        hist,
        age_bin_edges_seconds=age_edges,
        quantiles=(0.5, 0.25),
    )

    assert list(output) == [0.5, 0.25]
    assert all(not value.flags.writeable for value in output.values())
    assert all(not np.shares_memory(value, hist) for value in output.values())
    hist[:] = 0
    age_edges[:] = -10.0
    assert output[0.5][0, 0] == pytest.approx(5.0)
    for value in output.values():
        with pytest.raises(ValueError):
            value.flat[0] = 999
