"""``report_bootstrap`` exact categorical member bootstrap 的契約測試。

測試資料都是記憶體中的 synthetic member code；每個整數代表一個固定 sample unit，
``None`` 代表保留在 member denominator 但沒有 categorical numerator。這些測試驗證
seed provenance、一次串流、exact conditional-binomial 權重守恆、輸入閘門與 immutable
typed result，不把 synthetic 結果解讀成 OCM/NWW 科學成果或絕對來源機率。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from lagrangian_backtracking.report_bootstrap import (
    BOOTSTRAP_METHOD_ID,
    BOOTSTRAP_QUANTILE_METHOD,
    CategoricalBootstrapIntervals,
    bootstrap_categorical_intervals,
    derive_bootstrap_group_seed,
)


def test_public_contract_and_canonical_group_seed() -> None:
    """方法常數、公開名稱與 canonical JSON digest 必須固定且可重建。"""

    assert CategoricalBootstrapIntervals.__module__ == "lagrangian_backtracking.report_bootstrap"
    expected_public = {
        "BOOTSTRAP_METHOD_ID",
        "BOOTSTRAP_QUANTILE_METHOD",
        "CategoricalBootstrapIntervals",
        "derive_bootstrap_group_seed",
        "bootstrap_categorical_intervals",
    }
    from lagrangian_backtracking import report_bootstrap

    assert set(report_bootstrap.__all__) == expected_public
    assert BOOTSTRAP_METHOD_ID == "sequential_conditional_binomial_multinomial_v1"
    assert BOOTSTRAP_QUANTILE_METHOD == "linear"

    seed, digest = derive_bootstrap_group_seed(2**128 - 1, "receptor/a|boundary:7")
    canonical = json.dumps(
        {
            "method": BOOTSTRAP_METHOD_ID,
            "base_seed": str(2**128 - 1),
            "group": "receptor/a|boundary:7",
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    expected_digest_bytes = hashlib.sha256(canonical).digest()
    assert digest == expected_digest_bytes.hex()
    assert seed == int.from_bytes(expected_digest_bytes[:16], "big")


def test_seed_is_deterministic_and_group_specific() -> None:
    """相同 method/base/group 重現相同 seed，改 group 則改變 seed 與完整 digest。"""

    first = derive_bootstrap_group_seed(123, "group-a")
    second = derive_bootstrap_group_seed(123, "group-a")
    changed_group = derive_bootstrap_group_seed(123, "group-b")
    changed_seed = derive_bootstrap_group_seed(124, "group-a")

    assert first == second
    assert first != changed_group
    assert first != changed_seed
    assert len(first[1]) == 64
    assert first[1] == first[1].lower()


def test_one_shot_iterable_is_consumed_once_and_result_is_reproducible() -> None:
    """演示輸入只被迭代一次，且不需要 materialize member code。"""

    class OneShotCodes:
        """只允許一次 ``iter`` 呼叫的 synthetic iterable。"""

        def __init__(self, values: list[int | None]) -> None:
            self._values = values
            self.iter_calls = 0

        def __iter__(self):
            self.iter_calls += 1
            if self.iter_calls != 1:
                raise AssertionError("codes 被重複迭代")
            yield from self._values

    values = [0, None, 1, 0, None]
    one_shot = OneShotCodes(values)
    result = bootstrap_categorical_intervals(
        one_shot,
        member_count=len(values),
        category_count=2,
        replicates=31,
        confidence=0.9,
        base_seed=17,
        group_key="one-shot",
    )
    repeated = bootstrap_categorical_intervals(
        iter(values),
        member_count=len(values),
        category_count=2,
        replicates=31,
        confidence=0.9,
        base_seed=17,
        group_key="one-shot",
    )

    assert one_shot.iter_calls == 1
    assert np.array_equal(result.raw_counts, [2, 1])
    assert np.array_equal(result.raw_counts, repeated.raw_counts)
    assert np.array_equal(result.lower, repeated.lower)
    assert np.array_equal(result.median, repeated.median)
    assert np.array_equal(result.upper, repeated.upper)
    assert result.digest == repeated.digest


def test_all_one_category_proves_per_replicate_weight_conservation() -> None:
    """所有 member 都屬同類時，每個 replicate 的類別總 weight 必須恰為 N。"""

    result = bootstrap_categorical_intervals(
        (0 for _ in range(7)),
        member_count=7,
        category_count=3,
        replicates=101,
        confidence=0.95,
        base_seed=0,
        group_key="all-one-category",
    )

    assert np.array_equal(result.raw_counts, [7, 0, 0])
    assert np.array_equal(result.lower, [1.0, 0.0, 0.0])
    assert np.array_equal(result.median, [1.0, 0.0, 0.0])
    assert np.array_equal(result.upper, [1.0, 0.0, 0.0])


def test_none_stays_in_denominator_and_all_none_is_zero() -> None:
    """``None`` 不增加 numerator，但仍讓 bootstrap 分母維持完整 member_count。"""

    mixed = bootstrap_categorical_intervals(
        [0, None, 1, None],
        member_count=4,
        category_count=2,
        replicates=53,
        confidence=0.9,
        base_seed=3,
        group_key="missing-members",
    )
    all_none = bootstrap_categorical_intervals(
        [None, None, None],
        member_count=3,
        category_count=2,
        replicates=7,
        confidence=0.8,
        base_seed=3,
        group_key="all-none",
    )

    assert np.array_equal(mixed.raw_counts, [1, 1])
    assert np.all(mixed.lower >= 0.0)
    assert np.all(mixed.upper <= 1.0)
    assert np.array_equal(all_none.raw_counts, [0, 0])
    assert np.array_equal(all_none.lower, [0.0, 0.0])
    assert np.array_equal(all_none.median, [0.0, 0.0])
    assert np.array_equal(all_none.upper, [0.0, 0.0])


def test_category_zero_has_binomial_three_one_third_quantiles() -> None:
    """固定 seed 下以寬鬆 tolerance 檢查 ``N=3`` 的 Binomial(3, 1/3) 分布。"""

    result = bootstrap_categorical_intervals(
        [0, 1, 2],
        member_count=3,
        category_count=3,
        replicates=30_000,
        confidence=0.9,
        base_seed=2025,
        group_key="binomial-check",
    )

    # X/N 的理論 5%、50%、95% 分位數分別落在 0、1/3、2/3；抽樣誤差以
    # 寬鬆 tolerance 接受固定 PCG64DXSM stream 的有限 replicate 波動。
    assert result.lower[0] == pytest.approx(0.0, abs=0.02)
    assert result.median[0] == pytest.approx(1.0 / 3.0, abs=0.02)
    assert result.upper[0] == pytest.approx(2.0 / 3.0, abs=0.02)


@pytest.mark.parametrize(
    ("member_count", "category_count", "replicates", "confidence"),
    [
        (0, 2, 3, 0.95),
        (2, 0, 3, 0.95),
        (2, 2, 1, 0.95),
        (2, 2, 3, 0.0),
        (2, 2, 3, 1.0),
        (2, 2, 3, float("nan")),
    ],
)
def test_rejects_invalid_scalar_parameters(
    member_count: int,
    category_count: int,
    replicates: int,
    confidence: float,
) -> None:
    """零值、邊界值與非有限 confidence 都必須在消費 iterable 前 fail closed。"""

    with pytest.raises((TypeError, ValueError)):
        bootstrap_categorical_intervals(
            [0, 1],
            member_count=member_count,
            category_count=category_count,
            replicates=replicates,
            confidence=confidence,
            base_seed=0,
            group_key="invalid-parameter",
        )


@pytest.mark.parametrize("bad_seed", [-1, 2**128, True, np.int64(1)])
def test_rejects_invalid_base_seed(bad_seed: object) -> None:
    """base seed 嚴格遵守 128 位元無號原生 int 契約。"""

    with pytest.raises((TypeError, ValueError)):
        derive_bootstrap_group_seed(bad_seed, "seed-test")  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_group", ["", " leading", "trailing ", "不可用", "line\nbreak"])
def test_rejects_unsafe_group_key(bad_group: str) -> None:
    """空白、控制字元與非 ASCII group key 不得進入 seed canonicalization。"""

    with pytest.raises(ValueError):
        derive_bootstrap_group_seed(0, bad_group)


@pytest.mark.parametrize("bad_code", [True, 1.0, np.int64(1), -1, 2])
def test_rejects_non_native_or_out_of_range_code(bad_code: object) -> None:
    """categorical code 必須是原生 int 且嚴格落在固定類別軸內。"""

    with pytest.raises((TypeError, ValueError)):
        bootstrap_categorical_intervals(
            [0, bad_code],  # type: ignore[list-item]
            member_count=2,
            category_count=2,
            replicates=5,
            confidence=0.9,
            base_seed=0,
            group_key="invalid-code",
        )


@pytest.mark.parametrize("codes", [[0], [0, 1, 0]])
def test_rejects_short_or_long_iterable(codes: list[int]) -> None:
    """輸入少於或多於 N 筆都不能以靜默裁切方式產生結果。"""

    with pytest.raises(ValueError, match="成員"):
        bootstrap_categorical_intervals(
            codes,
            member_count=2,
            category_count=2,
            replicates=5,
            confidence=0.9,
            base_seed=0,
            group_key="wrong-length",
        )


def test_result_arrays_are_defensive_and_read_only() -> None:
    """結果不共享 constructor 輸入陣列，且所有 exposed arrays 都不可寫。"""

    raw = np.array([2, 1], dtype=np.int64)
    lower = np.array([0.1, 0.2], dtype=np.float64)
    median = np.array([0.3, 0.4], dtype=np.float64)
    upper = np.array([0.5, 0.6], dtype=np.float64)
    result = CategoricalBootstrapIntervals(
        method=BOOTSTRAP_METHOD_ID,
        group="immutable",
        digest="a" * 64,
        member_count=3,
        category_count=2,
        replicates=5,
        confidence=0.9,
        raw_counts=raw,
        lower=lower,
        median=median,
        upper=upper,
        quantile_method=BOOTSTRAP_QUANTILE_METHOD,
    )

    raw[0] = 99
    lower[0] = 99.0
    assert result.raw_counts[0] == 2
    assert result.lower[0] == 0.1
    for array in (result.raw_counts, result.lower, result.median, result.upper):
        assert array.flags.writeable is False
        with pytest.raises(ValueError):
            array[0] = array[0]
    with pytest.raises(FrozenInstanceError):
        result.confidence = 0.8  # type: ignore[misc]


@pytest.mark.parametrize(
    "overrides",
    [
        {"raw_counts": np.array([3, 1], dtype=np.int64)},
        {"lower": np.array([0.4, 0.2])},
        {"digest": "A" * 64},
        {"quantile_method": "midpoint"},
    ],
)
def test_result_cross_field_validation_rejects_inconsistent_product(overrides: dict[str, object]) -> None:
    """typed product 不能接受 raw 超分母、非單調區間或非固定 provenance 欄位。"""

    fields: dict[str, object] = {
        "method": BOOTSTRAP_METHOD_ID,
        "group": "cross-field",
        "digest": "b" * 64,
        "member_count": 3,
        "category_count": 2,
        "replicates": 5,
        "confidence": 0.9,
        "raw_counts": np.array([2, 1], dtype=np.int64),
        "lower": np.array([0.1, 0.2], dtype=np.float64),
        "median": np.array([0.3, 0.4], dtype=np.float64),
        "upper": np.array([0.5, 0.6], dtype=np.float64),
        "quantile_method": BOOTSTRAP_QUANTILE_METHOD,
    }
    fields.update(overrides)

    with pytest.raises((TypeError, ValueError)):
        CategoricalBootstrapIntervals(**fields)  # type: ignore[arg-type]
