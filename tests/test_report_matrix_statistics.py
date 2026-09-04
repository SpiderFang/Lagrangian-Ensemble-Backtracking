"""報告層停止結果與方向性跨站矩陣的資料契約測試。

測試只使用記憶體中的小型計數 mapping，並把它們視為已由 aggregate release 驗證過
的輸入。測試特別固定 source→target 的矩陣方向、有效成員分母、cross-site event
內部 share 與 diagonal 不適用遮罩，避免 renderer 以零值猜測缺資料或把兩種比例混用。
"""

from dataclasses import FrozenInstanceError, is_dataclass
from types import MappingProxyType

import numpy as np
import pytest

from lagrangian_backtracking.event_aggregation import CrossSiteAggregateKey
from lagrangian_backtracking.models import ParticleStatus
from lagrangian_backtracking.report_matrix_statistics import (
    ConnectivityStatistics,
    OutcomeStatistics,
    build_connectivity_statistics,
    build_outcome_statistics,
)
from lagrangian_backtracking.report_ratio_statistics import EstimateStatus

_SITES = ("site-a", "site-b", "site-c")
_OUTCOME_KEYS = tuple(
    status.value for status in ParticleStatus if status is not ParticleStatus.ACTIVE
)


def _outcomes(*, maximum: int, data_gap: int, numerical: int, total: int) -> dict[str, int]:
    """建立完整的非 ACTIVE 停止狀態拓撲，未指定狀態維持明確的零計數。"""

    counts = {key: 0 for key in _OUTCOME_KEYS}
    counts[ParticleStatus.MAX_AGE.value] = maximum
    counts[ParticleStatus.DATA_GAP.value] = data_gap
    counts[ParticleStatus.NUMERICAL_FAILURE.value] = numerical
    assert sum(counts.values()) == total
    return counts


def _cross_counts() -> dict[CrossSiteAggregateKey, int]:
    """建立三站完整有向 pair 拓撲；對角線不以 key 表示。"""

    return {
        CrossSiteAggregateKey(source, target): count
        for source, target, count in (
            ("site-a", "site-b", 2),
            ("site-a", "site-c", 1),
            ("site-b", "site-a", 1),
            ("site-b", "site-c", 0),
            ("site-c", "site-a", 0),
            ("site-c", "site-b", 0),
        )
    }


def test_outcome_statistics_preserve_raw_counts_ratios_and_failure_exposure() -> None:
    """停止狀態比例以 total 分母計算，失敗曝光另列且不進 valid 分母。"""

    product = build_outcome_statistics(
        {
            "site-a": _outcomes(maximum=2, data_gap=1, numerical=0, total=3),
            "site-b": _outcomes(maximum=2, data_gap=0, numerical=0, total=2),
        },
        {"site-a": 3, "site-b": 2},
        valid_member_denominator_by_site={"site-a": 2, "site-b": 2},
        minimum_count=2,
    )

    assert type(product) is OutcomeStatistics
    assert product.site_ids == ("site-a", "site-b")
    assert product.outcome_count_by_site["site-a"][ParticleStatus.MAX_AGE.value] == 2
    assert product.total_member_denominator_by_site["site-a"] == 3
    assert product.valid_member_denominator_by_site["site-a"] == 2

    max_age = product.outcome_ratios_by_site["site-a"][ParticleStatus.MAX_AGE.value]
    assert max_age.raw_numerator == 2
    assert max_age.raw_denominator == 3
    assert max_age.ratio == 2 / 3
    assert max_age.status is EstimateStatus.available

    gap = product.data_gap_exposure_by_site["site-a"]
    assert gap.raw_numerator == 1
    assert gap.raw_denominator == 3
    assert gap.ratio == 1 / 3
    assert gap.status is EstimateStatus.low_count
    assert product.failure_exposure_by_site["site-a"][ParticleStatus.NUMERICAL_FAILURE.value].ratio == 0.0


def test_connectivity_keeps_direction_and_separates_visit_fraction_from_event_share() -> None:
    """source→target 計數不可轉置，逐列 visit fraction 與列內 event share 必須分開。"""

    product = build_connectivity_statistics(
        _cross_counts(),
        {"site-a": 4, "site-b": 2, "site-c": 0},
        site_ids=_SITES,
        minimum_count=2,
    )

    assert type(product) is ConnectivityStatistics
    assert product.source_site_ids == _SITES
    assert product.target_site_ids == _SITES
    assert product.raw_matrix.tolist() == [[0, 2, 1], [1, 0, 0], [0, 0, 0]]
    assert product.row_valid_member_denominator.tolist() == [4, 2, 0]
    assert np.allclose(
        product.visit_fraction[:2, :],
        [[np.nan, 0.5, 0.25], [0.5, np.nan, 0.0]],
        equal_nan=True,
    )
    assert np.allclose(
        product.cross_site_event_share[:2, :],
        [[np.nan, 2 / 3, 1 / 3], [1.0, np.nan, 0.0]],
        equal_nan=True,
    )
    assert np.isnan(product.visit_fraction[2]).all()
    assert np.isnan(product.cross_site_event_share[2]).all()
    assert np.array_equal(
        product.diagonal_not_applicable_mask,
        np.eye(3, dtype=bool),
    )
    assert product.visit_fraction_status[0, 2] is EstimateStatus.low_count
    assert product.cross_site_event_share_status[1, 2] is EstimateStatus.low_count
    assert product.visit_fraction_status[2, 0] is EstimateStatus.zero_denominator


def test_connectivity_exposes_only_explicitly_declared_a_zone_two_by_two_diagnostic() -> None:
    """A 區 2×2 診斷必須依 caller 登錄的兩站順序切片，不能從矩陣猜站。"""

    product = build_connectivity_statistics(
        _cross_counts(),
        {"site-a": 4, "site-b": 2, "site-c": 1},
        site_ids=_SITES,
        a_zone_site_ids=("site-b", "site-a"),
    )

    assert product.a_zone_site_ids == ("site-b", "site-a")
    assert product.a_zone_2x2_raw_matrix.tolist() == [[0, 1], [2, 0]]
    assert np.array_equal(
        product.a_zone_2x2_diagonal_not_applicable_mask,
        np.eye(2, dtype=bool),
    )
    assert np.allclose(
        product.a_zone_2x2_visit_fraction,
        [[np.nan, 1 / 2], [2 / 4, np.nan]],
        equal_nan=True,
    )
    assert all(
        not array.flags.writeable
        for array in (
            product.a_zone_2x2_raw_matrix,
            product.a_zone_2x2_visit_fraction,
            product.a_zone_2x2_event_share,
            product.a_zone_2x2_diagonal_not_applicable_mask,
            product.a_zone_row_valid_member_denominator,
        )
    )


def test_matrix_products_are_frozen_defensive_and_keep_zero_denominator_unavailable() -> None:
    """矩陣陣列與 mapping 必須隔離輸入；零分母只能是 NaN 加明確狀態。"""

    counts = _cross_counts()
    denominators = {"site-a": 4, "site-b": 2, "site-c": 0}
    product = build_connectivity_statistics(counts, denominators, site_ids=_SITES)

    assert is_dataclass(product)
    assert product.__dataclass_params__.frozen is True
    assert isinstance(product.valid_member_denominator_by_source_site, MappingProxyType)
    assert all(not array.flags.writeable for array in (
        product.raw_matrix,
        product.row_valid_member_denominator,
        product.row_event_denominator,
        product.visit_fraction,
        product.cross_site_event_share,
        product.diagonal_not_applicable_mask,
    ))
    counts[CrossSiteAggregateKey("site-a", "site-b")] = 99
    denominators["site-a"] = 99
    assert product.raw_matrix[0, 1] == 2
    assert product.row_valid_member_denominator[0] == 4
    with pytest.raises(ValueError):
        product.raw_matrix[0, 1] = 99
    with pytest.raises(FrozenInstanceError):
        product.raw_matrix = np.zeros((3, 3), dtype=np.int64)  # type: ignore[misc]


def test_matrix_builders_reject_count_overflow_or_inconsistent_failure_denominator() -> None:
    """分子超過列分母、停止守恆或失敗未排除時都必須 fail closed。"""

    bad_cross = _cross_counts()
    bad_cross[CrossSiteAggregateKey("site-a", "site-b")] = 5
    with pytest.raises(ValueError, match="numerator"):
        build_connectivity_statistics(
            bad_cross,
            {"site-a": 4, "site-b": 2, "site-c": 0},
            site_ids=_SITES,
        )

    bad_outcomes = _outcomes(maximum=2, data_gap=1, numerical=0, total=3)
    bad_outcomes[ParticleStatus.MAX_AGE.value] = 1
    with pytest.raises(ValueError, match="合計"):
        build_outcome_statistics(
            {"site-a": bad_outcomes},
            {"site-a": 3},
            valid_member_denominator_by_site={"site-a": 2},
        )

    with pytest.raises(ValueError, match="DATA_GAP"):
        build_outcome_statistics(
            {"site-a": _outcomes(maximum=2, data_gap=1, numerical=0, total=3)},
            {"site-a": 3},
            valid_member_denominator_by_site={"site-a": 3},
        )
