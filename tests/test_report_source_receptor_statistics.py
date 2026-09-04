"""來源段—受體報告統計的 immutable 與條件式比例契約測試。

測試資料是公尺／秒制的 synthetic source-receptor count 與 travel-age histogram。
同一受體／邊界分類下的 event share 使用所有 segment 的 raw count 作分母，而條件式
通過比例使用有效 receptor member 分母；兩者不能互相替代，也不把零分母補成零比例。
"""

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import numpy as np
import pytest

from lagrangian_backtracking.event_aggregation import (
    ReceptorAggregateKey,
    SourceReceptorAggregateKey,
)
from lagrangian_backtracking.report_ratio_statistics import EstimateStatus
from lagrangian_backtracking.report_source_receptor_statistics import (
    SourceReceptorStatistic,
    SourceReceptorStatistics,
    build_source_receptor_statistics,
)

_EDGES = np.array([0.0, 10.0, 30.0, 60.0], dtype=np.float64)
_QUANTILES = (0.25, 0.5, 0.75)


def _key(segment: str, *, receptor: str = "receptor-a", kind: str = "local") -> SourceReceptorAggregateKey:
    """建立固定站點、受體、邊界分類下的來源段 key。"""

    return SourceReceptorAggregateKey("site-a", receptor, kind, segment)


def test_source_receptor_keeps_conditional_ratio_event_share_and_age_midpoint_quantiles() -> None:
    """同一受體的兩個 segment 應保存兩種不同分母的比例與秒制 age 分位數。"""

    first = _key("segment-1")
    second = _key("segment-2")
    product = build_source_receptor_statistics(
        {first: 3, second: 1},
        {
            first: np.array([1, 1, 1], dtype=np.int64),
            second: np.array([0, 1, 0], dtype=np.int64),
        },
        {
            ReceptorAggregateKey("site-a", "receptor-a"): 5,
        },
        age_bin_edges_seconds=_EDGES,
        quantiles=_QUANTILES,
        minimum_count=2,
    )

    assert type(product) is SourceReceptorStatistics
    assert [record.key for record in product.records] == [first, second]
    row = product[first]
    assert type(row) is SourceReceptorStatistic
    assert row.raw_numerator == 3
    assert row.valid_receptor_denominator == 5
    assert row.event_share_denominator == 4
    assert row.conditional_ratio.ratio == 3 / 5
    assert row.event_share_ratio.ratio == 3 / 4
    assert row.conditional_ratio.status is EstimateStatus.available
    assert row.event_share_ratio.status is EstimateStatus.available
    assert row.travel_age.raw_count.tolist() == [1, 1, 1]
    assert row.travel_age.cell_count == 3
    assert row.travel_age.quantile_values_seconds[0.25] == 5.0
    assert row.travel_age.quantile_values_seconds[0.5] == 20.0
    assert row.travel_age.quantile_values_seconds[0.75] == 45.0
    assert np.array_equal(row.travel_age.bin_widths_seconds, [10.0, 20.0, 30.0])

    # 第二段的 raw 只有一筆，低樣本狀態仍保存原始比例與 age 中點近似。
    assert product[second].conditional_ratio.status is EstimateStatus.low_count
    assert product[second].event_share_ratio.status is EstimateStatus.low_count
    assert product[second].travel_age.status is EstimateStatus.low_count
    assert product[second].travel_age.quantile_values_seconds[0.5] == 20.0


def test_zero_source_receptor_keeps_explicit_unavailable_age_and_share_state() -> None:
    """零 raw count 的來源段要保留 zero_denominator，而非製造零 event share。"""

    empty = _key("segment-empty", receptor="receptor-empty", kind="outer")
    product = build_source_receptor_statistics(
        {empty: 0},
        {empty: np.zeros(3, dtype=np.int64)},
        {
            ReceptorAggregateKey("site-a", "receptor-empty"): 0,
        },
        age_bin_edges_seconds=_EDGES,
        quantiles=_QUANTILES,
    )

    row = product[empty]
    assert row.conditional_ratio.ratio is None
    assert row.event_share_ratio.ratio is None
    assert row.conditional_ratio.status is EstimateStatus.zero_denominator
    assert row.event_share_ratio.status is EstimateStatus.zero_denominator
    assert row.travel_age.status is EstimateStatus.zero_denominator
    assert all(value is None for value in row.travel_age.quantile_values_seconds.values())


def test_source_receptor_products_are_readonly_and_mapping_keys_are_exact() -> None:
    """來源 histogram、age 軸與 nested mapping 都必須是 defensive readonly snapshot。"""

    key = _key("segment-1")
    histogram = np.array([1, 0, 0], dtype=np.int64)
    product = build_source_receptor_statistics(
        {key: 1},
        {key: histogram},
        {ReceptorAggregateKey("site-a", "receptor-a"): 1},
        age_bin_edges_seconds=_EDGES,
        quantiles=_QUANTILES,
    )
    histogram[0] = 99
    row = product[key]
    assert row.travel_age.raw_count[0] == 1
    assert isinstance(product.by_key, MappingProxyType)
    assert isinstance(row.travel_age.quantile_values_seconds, MappingProxyType)
    assert all(
        not array.flags.writeable
        for array in (
            row.travel_age.raw_count,
            row.travel_age.age_bin_edges_seconds,
            row.travel_age.bin_midpoints_seconds,
            row.travel_age.bin_widths_seconds,
        )
    )
    with pytest.raises(ValueError):
        row.travel_age.raw_count[0] = 2
    with pytest.raises(FrozenInstanceError):
        row.raw_numerator = 2  # type: ignore[misc]
    with pytest.raises(TypeError):
        product.by_key[key] = row  # type: ignore[index]


@pytest.mark.parametrize(
    "bad_input",
    [
        "raw-count-and-travel-histogram-mismatch",
        "travel-age-bin-shape-mismatch",
        "source-count-exceeds-receptor-denominator",
    ],
)
def test_source_receptor_rejects_inconsistent_raw_inputs(bad_input: str) -> None:
    """raw count、histogram 合計與 receptor 分母不一致時不得交給 renderer 猜測。"""

    key = _key("segment-1")
    raw = {key: 1}
    hist = {key: np.array([1, 0, 0], dtype=np.int64)}
    denominator = {ReceptorAggregateKey("site-a", "receptor-a"): 1}
    if bad_input == "raw-count-and-travel-histogram-mismatch":
        raw[key] = 2
    elif bad_input == "travel-age-bin-shape-mismatch":
        hist[key] = np.array([1, 0], dtype=np.int64)
    else:
        denominator[ReceptorAggregateKey("site-a", "receptor-a")] = 0

    with pytest.raises((TypeError, ValueError)):
        build_source_receptor_statistics(
            raw,
            hist,
            denominator,
            age_bin_edges_seconds=_EDGES,
            quantiles=_QUANTILES,
        )
