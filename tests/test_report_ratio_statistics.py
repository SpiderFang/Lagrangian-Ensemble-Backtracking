"""報告比例與首次通過年齡分箱統計的資料契約測試。

本檔只使用小型人工 int64 histogram 與計數，不讀取 OCM／NWW3 或 trajectory
檔案。測試固定 histogram 軸順序為 ``(y_cell, x_cell, age_bin)``，並檢查報告
產品是否保留 raw count、正確區分零分母／低樣本、拒絕非原生 scalar 與 overflow，
以及對外暴露的陣列與 mapping 是否真正 defensive readonly。
"""

from dataclasses import FrozenInstanceError, is_dataclass
from types import MappingProxyType

import numpy as np
import pytest

import lagrangian_backtracking.report_ratio_statistics as ratio_statistics
from lagrangian_backtracking.report_ratio_statistics import (
    CountRatio,
    EstimateStatus,
    HistogramQuantileProduct,
    build_count_ratio,
    build_histogram_quantiles,
)

_INT64_MAX = int(np.iinfo(np.int64).max)


def test_estimate_status_is_string_enum_with_three_semantic_states() -> None:
    """狀態必須是字串 Enum，且三個值分別表示可用、零分母與低樣本。"""

    assert issubclass(EstimateStatus, str)
    assert list(EstimateStatus) == [
        EstimateStatus.available,
        EstimateStatus.zero_denominator,
        EstimateStatus.low_count,
    ]
    assert [status.value for status in EstimateStatus] == [
        "available",
        "zero_denominator",
        "low_count",
    ]
    assert EstimateStatus.AVAILABLE is EstimateStatus.available


@pytest.mark.parametrize(
    ("numerator", "denominator", "minimum_count", "status", "ratio"),
    [
        (2, 4, 2, EstimateStatus.available, 0.5),
        (0, 0, 2, EstimateStatus.zero_denominator, None),
        (1, 1, 2, EstimateStatus.low_count, 1.0),
        (0, 4, 2, EstimateStatus.low_count, 0.0),
    ],
)
def test_build_count_ratio_preserves_counts_and_distinguishes_states(
    numerator: int,
    denominator: int,
    minimum_count: int,
    status: EstimateStatus,
    ratio: float | None,
) -> None:
    """比例值與零分母／低樣本狀態不可互相混淆，低樣本仍保留精確比例。"""

    product = build_count_ratio(
        numerator,
        denominator,
        minimum_count=minimum_count,
    )

    assert isinstance(product, CountRatio)
    assert product.numerator == numerator
    assert product.denominator == denominator
    assert product.raw_numerator == numerator
    assert product.raw_denominator == denominator
    assert product.minimum_count == minimum_count
    assert product.status is status
    assert product.ratio == ratio
    assert product.fraction == ratio


def test_count_ratio_uses_numerator_for_low_count_with_large_denominator() -> None:
    """分母很大但分子為一時，仍須依分子門檻標 low_count 並保存比例。"""

    product = build_count_ratio(1, _INT64_MAX, minimum_count=2)

    assert product.status is EstimateStatus.low_count
    assert product.ratio == 1 / _INT64_MAX
    assert type(product.ratio) is float


def test_count_ratio_is_frozen_and_uses_native_output_scalars() -> None:
    """CountRatio 必須是 frozen dataclass，且可用比例是原生 Python scalar。"""

    product = build_count_ratio(1, 4, minimum_count=1)

    assert is_dataclass(product)
    assert product.__dataclass_params__.frozen is True
    assert type(product.numerator) is int
    assert type(product.denominator) is int
    assert type(product.minimum_count) is int
    assert type(product.ratio) is float
    assert type(product.status) is EstimateStatus
    with pytest.raises(FrozenInstanceError):
        product.ratio = 0.75  # type: ignore[misc]


@pytest.mark.parametrize(
    ("numerator", "denominator", "minimum_count"),
    [
        (np.int64(1), 2, 1),
        (1, np.int64(2), 1),
        (1, 2, np.int64(1)),
        (True, 1, 1),
        (1, 1, True),
        (-1, 1, 1),
        (2, 1, 1),
    ],
)
def test_count_ratio_rejects_non_native_scalars_and_invalid_count_relation(
    numerator: object,
    denominator: object,
    minimum_count: object,
) -> None:
    """比例 builder 不可讓 bool、NumPy scalar、負數或超分母計數偷換契約。"""

    with pytest.raises((TypeError, ValueError)):
        build_count_ratio(  # type: ignore[arg-type]
            numerator,
            denominator,
            minimum_count=minimum_count,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("numerator", "denominator"),
    [(_INT64_MAX + 1, _INT64_MAX + 1), (0, _INT64_MAX + 1)],
)
def test_count_ratio_rejects_signed_int64_overflow(
    numerator: int,
    denominator: int,
) -> None:
    """超過 signed int64 的 Python 大整數必須在比例計算前失敗。"""

    with pytest.raises(ValueError, match="int64"):
        build_count_ratio(numerator, denominator)


def _fixture_histogram() -> tuple[np.ndarray, np.ndarray]:
    """建立含 low、zero 與兩個 available cell 的小型首次通過年齡 histogram。"""

    histogram = np.array(
        [
            [[1, 0, 0], [0, 0, 0]],
            [[1, 1, 0], [0, 2, 0]],
        ],
        dtype=np.int64,
    )
    edges = np.array([0.0, 10.0, 20.0, 30.0], dtype=np.float64)
    return histogram, edges


def test_build_histogram_quantiles_reuses_midpoint_quantiles_and_states() -> None:
    """分位數沿用串流核心的 bin 選擇，並以中點秒數輸出逐格狀態。"""

    histogram, edges = _fixture_histogram()
    product = build_histogram_quantiles(
        histogram,
        age_bin_edges_seconds=edges,
        quantiles=(0.75, 0.25, 0.5),
        minimum_count=2,
    )

    assert isinstance(product, HistogramQuantileProduct)
    assert product.quantiles == (0.75, 0.25, 0.5)
    assert np.array_equal(product.raw_count, histogram)
    assert np.array_equal(product.cell_count, np.array([[1, 0], [2, 2]], dtype=np.int64))
    assert np.array_equal(
        product.status,
        np.array(
            [
                [EstimateStatus.low_count, EstimateStatus.zero_denominator],
                [EstimateStatus.available, EstimateStatus.available],
            ],
            dtype=object,
        ),
    )
    assert np.array_equal(product.bin_midpoints_seconds, np.array([5.0, 15.0, 25.0]))
    assert np.array_equal(product.bin_widths_seconds, np.array([10.0, 10.0, 10.0]))
    assert np.allclose(
        product.quantile_values_seconds[0.25],
        np.array([[5.0, np.nan], [5.0, 15.0]]),
        equal_nan=True,
    )
    assert np.allclose(
        product.quantile_values_seconds[0.75],
        np.array([[5.0, np.nan], [15.0, 15.0]]),
        equal_nan=True,
    )


def test_histogram_product_is_frozen_and_defensively_readonly() -> None:
    """產品陣列與 quantile mapping 不共享輸入，也不能由 caller 改寫。"""

    histogram, edges = _fixture_histogram()
    product = build_histogram_quantiles(
        histogram,
        age_bin_edges_seconds=edges,
        quantiles=(0.5,),
        minimum_count=1,
    )

    histogram[1, 0, 0] = 99
    edges[1] = 999.0
    assert product.raw_count[1, 0, 0] == 1
    assert product.age_bin_edges_seconds[1] == 10.0
    assert isinstance(product.quantile_values_seconds, MappingProxyType)
    arrays = (
        product.raw_count,
        product.cell_count,
        product.age_bin_edges_seconds,
        product.bin_midpoints_seconds,
        product.bin_widths_seconds,
        product.status,
        product.quantile_values_seconds[0.5],
    )
    assert all(array.flags.writeable is False for array in arrays)
    assert all(not np.shares_memory(array, histogram) for array in arrays if array.dtype != object)
    with pytest.raises(FrozenInstanceError):
        product.minimum_count = 3  # type: ignore[misc]
    with pytest.raises(TypeError):
        product.quantile_values_seconds[0.5] = product.quantile_values_seconds[0.5]  # type: ignore[index]
    with pytest.raises(ValueError):
        product.raw_count[0, 0, 0] = 3


def test_histogram_total_count_rejects_element_and_sum_overflow() -> None:
    """單一 bin 與多個合法 bin 的總數都不可繞回 signed int64。"""

    element_overflow = np.array([[[np.uint64(_INT64_MAX) + np.uint64(1)]]], dtype=np.uint64)
    with pytest.raises(ValueError, match="int64"):
        build_histogram_quantiles(
            element_overflow,
            age_bin_edges_seconds=np.array([0.0, 1.0]),
        )

    sum_overflow = np.array(
        [[[np.uint64(_INT64_MAX), np.uint64(_INT64_MAX)]]],
        dtype=np.uint64,
    )
    with pytest.raises(ValueError, match="總數"):
        build_histogram_quantiles(
            sum_overflow,
            age_bin_edges_seconds=np.array([0.0, 1.0, 2.0]),
        )


def test_histogram_uses_nonuniform_bin_midpoints_as_approximation() -> None:
    """非均勻 age bins 應以實際邊界中點，而非固定平均寬度，近似分位數。"""

    product = build_histogram_quantiles(
        np.array([[[0, 1]]], dtype=np.int64),
        age_bin_edges_seconds=np.array([0.0, 10.0, 30.0]),
        quantiles=(0.5,),
    )

    assert product.quantile_values_seconds[0.5][0, 0] == pytest.approx(20.0)
    assert np.array_equal(product.bin_midpoints_seconds, np.array([5.0, 20.0]))
    assert np.array_equal(product.bin_widths_seconds, np.array([10.0, 20.0]))


@pytest.mark.parametrize(
    "bad_quantiles",
    [
        (0.5, 0.5),
        (0.0,),
        (1.0,),
        (np.float64(0.5),),
        (True,),
        ("0.5",),
    ],
)
def test_histogram_rejects_invalid_quantile_scalars(bad_quantiles: object) -> None:
    """分位數必須是唯一、開區間內的原生 Python float。"""

    with pytest.raises((TypeError, ValueError)):
        build_histogram_quantiles(
            np.array([[[1]]], dtype=np.int64),
            age_bin_edges_seconds=np.array([0.0, 10.0]),
            quantiles=bad_quantiles,  # type: ignore[arg-type]
        )


def test_histogram_builder_delegates_quantile_selection_to_streaming_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """報告 wrapper 必須呼叫既有 streaming quantile 核心，而不是複製選 bin 演算法。"""

    histogram = np.array([[[1, 0]]], dtype=np.int64)
    edges = np.array([0.0, 10.0, 20.0])
    called: dict[str, object] = {}
    original = ratio_statistics.first_passage_quantiles

    def spy(
        input_histogram: np.ndarray,
        *,
        age_bin_edges_seconds: np.ndarray,
        quantiles: tuple[float, ...],
    ) -> dict[float, np.ndarray]:
        """記錄 wrapper 傳入的 canonical inputs，再交回正式核心結果。"""

        called["histogram"] = input_histogram
        called["edges"] = age_bin_edges_seconds
        called["quantiles"] = quantiles
        return original(
            input_histogram,
            age_bin_edges_seconds=age_bin_edges_seconds,
            quantiles=quantiles,
        )

    monkeypatch.setattr(ratio_statistics, "first_passage_quantiles", spy)
    build_histogram_quantiles(
        histogram,
        age_bin_edges_seconds=edges,
        quantiles=(0.5,),
    )

    assert called["quantiles"] == (0.5,)
    assert isinstance(called["histogram"], np.ndarray)
    assert np.array_equal(called["histogram"], histogram)
