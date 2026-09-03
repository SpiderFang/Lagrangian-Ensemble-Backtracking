"""事件聚合塊的大計數與防禦性複製契約測試。

本檔只驗證 ``EventAggregateChunk`` 的兩項封存保證。第一項從核心測試的合法
兩站零拓撲建立可寫輸入，確認建構完成後，呼叫端再修改原始陣列與 mapping
不會改變已封存結果，且結果內的陣列確實是唯讀副本。第二項把兩個固定寬度
``int64`` 陣列元素各設為 ``np.iinfo(np.int64).max``，讓同一類 raw、網格與
旅行年齡直方圖的 Python 整數總和達到兩倍上限；單一元素仍在 ``int64`` 可表達
範圍內，因此測試能分辨「元素驗證合法」與「以 ``np.int64`` 累加時可能溢位」
這兩件事。這些計數只代表條件式來源足跡或相對來源權重的原始累加量，不是
絕對來源機率或因果歸因。
"""

from __future__ import annotations

import numpy as np
import pytest
from test_event_aggregation_core import (
    _AGE_BIN_EDGES_SECONDS,
    _SHARED_SEGMENT_ID,
    _aggregate_spec,
)

from lagrangian_backtracking.event_aggregation import (
    BoundaryAggregateKey,
    CrossSiteAggregateKey,
    EventAggregateChunk,
    ReceptorAggregateKey,
    SiteEventGridCounts,
    SourceReceptorAggregateKey,
    initialize_event_aggregate,
)
from lagrangian_backtracking.models import ParticleStatus

_GRID_FIELDS = (
    "local_first_exit_count",
    "outer_first_exit_count",
    "bed_first_contact_count",
    "bed_repeated_contact_count",
    "data_gap_failure_count",
    "numerical_failure_count",
)


def _mutable_inputs_from_core() -> dict[str, object]:
    """從核心 fixture 複製完整合法零拓撲，並將外層輸入改為可寫容器。

    核心 fixture 已固定兩站 2×2 公尺網格、local/outer 邊界 key、兩個年齡
    bin 與所有關聯欄位。這裡只複製其資料，不重新發明拓撲；計數陣列、邊界
    格線、年齡格線及各層 mapping 都由測試保留可寫輸入，供建構後的修改檢查
    使用。``SiteEventGridCounts`` 本身會先複製其六個網格陣列，符合公開型別
    的既有資料契約。
    """

    core = initialize_event_aggregate(
        _aggregate_spec(),
        age_bin_edges_seconds=_AGE_BIN_EDGES_SECONDS,
    )
    site_grid_counts = {
        site_id: SiteEventGridCounts(
            **{
                field: np.array(getattr(counts, field), copy=True)
                for field in _GRID_FIELDS
            }
        )
        for site_id, counts in core.site_grid_counts.items()
    }
    return {
        "site_grid_counts": site_grid_counts,
        "boundary_bin_edges_m": {
            key: np.array(value, copy=True)
            for key, value in core.boundary_bin_edges_m.items()
        },
        "boundary_arclength_raw_count": {
            key: np.array(value, copy=True)
            for key, value in core.boundary_arclength_raw_count.items()
        },
        "boundary_travel_age_histogram": {
            key: np.array(value, copy=True)
            for key, value in core.boundary_travel_age_histogram.items()
        },
        "age_bin_edges_seconds": np.array(
            core.age_bin_edges_seconds,
            copy=True,
        ),
        "source_receptor_raw_count": dict(core.source_receptor_raw_count),
        "source_receptor_travel_age_histogram": {
            key: np.array(value, copy=True)
            for key, value in core.source_receptor_travel_age_histogram.items()
        },
        "cross_site_unique_member_count": dict(
            core.cross_site_unique_member_count
        ),
        "outcome_count_by_site": {
            site_id: dict(outcomes)
            for site_id, outcomes in core.outcome_count_by_site.items()
        },
        "valid_member_denominator_by_site": dict(
            core.valid_member_denominator_by_site
        ),
        "total_member_count_by_site": dict(core.total_member_count_by_site),
        "valid_member_denominator_by_receptor": dict(
            core.valid_member_denominator_by_receptor
        ),
        "input_particle_count": core.input_particle_count,
    }


def _python_int_sum(array: np.ndarray) -> int:
    """逐元素轉成 Python ``int`` 後加總，避免測試本身先發生固定寬度溢位。"""

    return sum(int(value) for value in array.flat)


def _assert_chunk_arrays_readonly(chunk: EventAggregateChunk) -> None:
    """確認聚合塊內所有計數、格線與年齡軸陣列均已關閉寫入權限。"""

    for counts in chunk.site_grid_counts.values():
        for field in _GRID_FIELDS:
            assert getattr(counts, field).flags.writeable is False
    for mapping in (
        chunk.boundary_bin_edges_m,
        chunk.boundary_arclength_raw_count,
        chunk.boundary_travel_age_histogram,
        chunk.source_receptor_travel_age_histogram,
    ):
        for array in mapping.values():
            assert array.flags.writeable is False
    assert chunk.age_bin_edges_seconds.flags.writeable is False


def test_event_aggregate_chunk_defensively_copies_core_aggregate_inputs() -> None:
    """建構後修改可寫輸入，不得改變結果；所有輸出陣列與 mapping 必須唯讀。

    聚合結果會被寫入正式 release，不能讓仍被上游累加器持有的陣列或巢狀
    mapping 透過共享記憶體悄悄改寫。測試先以核心 fixture 的完整零拓撲直接
    建構 ``EventAggregateChunk``，再分別修改邊界格線、raw/hist 陣列、年齡軸、
    外層 key mapping 與 outcome 內層 mapping；結果若仍保持原值，才表示
    defensive-copy 與唯讀封存同時成立。
    """

    inputs = _mutable_inputs_from_core()
    local_key = BoundaryAggregateKey(
        "site-a",
        "local",
        _SHARED_SEGMENT_ID,
    )
    source_key = SourceReceptorAggregateKey(
        "site-a",
        "receptor-a",
        "local",
        _SHARED_SEGMENT_ID,
    )
    cross_key = CrossSiteAggregateKey("site-a", "site-b")
    receptor_key = ReceptorAggregateKey("site-a", "receptor-a")

    chunk = EventAggregateChunk(**inputs)

    boundary_edges = inputs["boundary_bin_edges_m"]
    boundary_raw = inputs["boundary_arclength_raw_count"]
    boundary_hist = inputs["boundary_travel_age_histogram"]
    age_edges = inputs["age_bin_edges_seconds"]
    site_grid_counts = inputs["site_grid_counts"]
    source_raw = inputs["source_receptor_raw_count"]
    source_hist = inputs["source_receptor_travel_age_histogram"]
    cross_counts = inputs["cross_site_unique_member_count"]
    outcomes = inputs["outcome_count_by_site"]
    valid_by_site = inputs["valid_member_denominator_by_site"]
    total_by_site = inputs["total_member_count_by_site"]
    valid_by_receptor = inputs["valid_member_denominator_by_receptor"]

    boundary_edges[local_key][1] = 99.0
    boundary_raw[local_key][0] = 7
    boundary_hist[local_key][0, 0] = 7
    age_edges[1] = 99.0
    site_grid_counts["site-a"] = site_grid_counts["site-b"]
    source_raw[source_key] = 1
    source_hist[source_key] = np.array([1, 0], dtype=np.int64)
    cross_counts[cross_key] = 1
    outcomes["site-a"][ParticleStatus.MAX_AGE.value] = 1
    valid_by_site["site-a"] = 1
    total_by_site["site-a"] = 1
    valid_by_receptor[receptor_key] = 1

    assert np.array_equal(
        chunk.boundary_bin_edges_m[local_key],
        [0.0, 1.0, 2.0, 2.5],
    )
    assert np.array_equal(
        chunk.boundary_arclength_raw_count[local_key],
        [0, 0, 0],
    )
    assert np.array_equal(
        chunk.boundary_travel_age_histogram[local_key],
        np.zeros((3, 2), dtype=np.int64),
    )
    assert np.array_equal(chunk.age_bin_edges_seconds, [0.0, 10.0, 20.0])
    assert "site-a" in chunk.site_grid_counts
    assert source_key not in chunk.source_receptor_raw_count
    assert source_key not in chunk.source_receptor_travel_age_histogram
    assert cross_key not in chunk.cross_site_unique_member_count
    assert chunk.outcome_count_by_site["site-a"] == {}
    assert chunk.valid_member_denominator_by_site["site-a"] == 0
    assert chunk.total_member_count_by_site["site-a"] == 0
    assert receptor_key not in chunk.valid_member_denominator_by_receptor
    _assert_chunk_arrays_readonly(chunk)
    with pytest.raises(TypeError):
        chunk.boundary_bin_edges_m[local_key] = np.array(
            [0.0, 1.0],
            dtype=np.float64,
        )
    with pytest.raises(TypeError):
        chunk.outcome_count_by_site["site-a"][ParticleStatus.MAX_AGE.value] = 1
    with pytest.raises(ValueError):
        chunk.age_bin_edges_seconds[0] = 1.0


def test_event_aggregate_chunk_accepts_python_int_totals_above_int64() -> None:
    """兩個 int64 上限元素的 Python 總和為兩倍上限時，合法聚合仍可封存。

    local 網格、local 邊界弧長 raw 與其 ``(s_bin, age_bin)`` 直方圖各放入兩個
    不同 cell/bin 的 ``int64`` 最大值；source-receptor raw 使用可容納兩倍值的
    Python ``int``，其 age histogram 則放入兩個單獨的最大值。site total、valid、
    outcome、receptor 分母及 input count 均同步設為同一個兩倍總量，其他 site、
    local/outer 邊界 key 維持核心 fixture 的完整零拓撲。建構成功表示關聯驗證
    使用 Python 任意精度累加，而不是讓 ``np.int64`` 累加器在兩倍上限處繞回。
    這些值仍是條件式來源足跡的計數載體，不代表絕對來源機率。
    """

    inputs = _mutable_inputs_from_core()
    max_int64 = int(np.iinfo(np.int64).max)
    total = 2 * max_int64
    local_key = BoundaryAggregateKey(
        "site-a",
        "local",
        _SHARED_SEGMENT_ID,
    )
    source_key = SourceReceptorAggregateKey(
        "site-a",
        "receptor-a",
        "local",
        _SHARED_SEGMENT_ID,
    )
    receptor_key = ReceptorAggregateKey("site-a", "receptor-a")

    grid_arrays = {
        field: np.array(
            getattr(inputs["site_grid_counts"]["site-a"], field),
            copy=True,
        )
        for field in _GRID_FIELDS
    }
    grid_arrays["local_first_exit_count"][0, 0] = max_int64
    grid_arrays["local_first_exit_count"][1, 1] = max_int64
    inputs["site_grid_counts"]["site-a"] = SiteEventGridCounts(
        **grid_arrays
    )

    raw = inputs["boundary_arclength_raw_count"][local_key]
    raw[0] = max_int64
    raw[1] = max_int64
    travel_hist = inputs["boundary_travel_age_histogram"][local_key]
    travel_hist[0, 0] = max_int64
    travel_hist[1, 1] = max_int64

    inputs["source_receptor_raw_count"] = {source_key: total}
    inputs["source_receptor_travel_age_histogram"] = {
        source_key: np.array([max_int64, max_int64], dtype=np.int64)
    }
    inputs["outcome_count_by_site"]["site-a"] = {
        ParticleStatus.MAX_AGE.value: total
    }
    inputs["valid_member_denominator_by_site"]["site-a"] = total
    inputs["total_member_count_by_site"]["site-a"] = total
    inputs["valid_member_denominator_by_receptor"][receptor_key] = total
    inputs["input_particle_count"] = total

    chunk = EventAggregateChunk(**inputs)

    assert _python_int_sum(
        chunk.site_grid_counts["site-a"].local_first_exit_count
    ) == total
    assert _python_int_sum(chunk.boundary_arclength_raw_count[local_key]) == total
    assert _python_int_sum(
        chunk.boundary_travel_age_histogram[local_key]
    ) == total
    assert _python_int_sum(
        chunk.source_receptor_travel_age_histogram[source_key]
    ) == total
    assert chunk.source_receptor_raw_count[source_key] == total
    assert type(chunk.source_receptor_raw_count[source_key]) is int
    assert chunk.total_member_count_by_site["site-a"] == total
    assert chunk.valid_member_denominator_by_site["site-a"] == total
    assert chunk.outcome_count_by_site["site-a"][ParticleStatus.MAX_AGE.value] == total
    assert chunk.valid_member_denominator_by_receptor[receptor_key] == total
    assert chunk.input_particle_count == total
    assert set(chunk.boundary_bin_edges_m) == {
        BoundaryAggregateKey(site_id, kind, _SHARED_SEGMENT_ID)
        for site_id in ("site-a", "site-b")
        for kind in ("local", "outer")
    }
    assert _python_int_sum(
        chunk.boundary_arclength_raw_count[
            BoundaryAggregateKey("site-a", "outer", _SHARED_SEGMENT_ID)
        ]
    ) == 0
    assert _python_int_sum(
        chunk.site_grid_counts["site-a"].outer_first_exit_count
    ) == 0
    _assert_chunk_arrays_readonly(chunk)
