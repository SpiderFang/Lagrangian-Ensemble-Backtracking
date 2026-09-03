"""報告層路徑格網統計的 immutable、分母與低樣本遮罩契約測試。

測試資料是記憶體中的 synthetic ``StreamingPathwayAggregate``，用來驗證報告層的
shape、單位軸、分箱分位數、有效成員分母與 defensive-copy 行為；它不代表 OCM/NWW3
正式科學成果，也不把條件式來源足跡解讀成絕對來源機率。
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from lagrangian_backtracking.report_pathway_statistics import (
    PathwayGridStatistics,
    build_pathway_grid_statistics,
)
from lagrangian_backtracking.streaming_aggregation import StreamingPathwayAggregate


def _pathway(*, input_particle_count: int = 3) -> StreamingPathwayAggregate:
    """建立一列三格的公尺／秒制 pathway fixture。

    三個 cell 的 unique count 為 0、2、3；每個 age histogram 沿 age 軸的總和與
    unique count 相同，residence 秒數總和為 6 秒，符合既有 aggregate 的守恆契約。
    """

    return StreamingPathwayAggregate(
        x_edges_m=np.array([0.0, 10.0, 20.0, 30.0]),
        y_edges_m=np.array([0.0, 5.0]),
        age_bin_edges_seconds=np.array([0.0, 10.0, 20.0, 30.0]),
        unique_particle_count=np.array([[0, 2, 3]], dtype=np.int64),
        residence_time_seconds=np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
        first_passage_age_histogram=np.array(
            [[[0, 0, 0], [1, 1, 0], [0, 1, 2]]],
            dtype=np.int64,
        ),
        input_particle_count=input_particle_count,
        input_interval_seconds=6.0,
        allocated_interval_seconds=6.0,
    )


def _zero_pathway() -> StreamingPathwayAggregate:
    """建立沒有有效成員的零分母 pathway，保留合法的零秒 aggregate。"""

    return StreamingPathwayAggregate(
        x_edges_m=np.array([0.0, 10.0]),
        y_edges_m=np.array([0.0, 5.0]),
        age_bin_edges_seconds=np.array([0.0, 10.0]),
        unique_particle_count=np.zeros((1, 1), dtype=np.int64),
        residence_time_seconds=np.zeros((1, 1), dtype=np.float64),
        first_passage_age_histogram=np.zeros((1, 1, 1), dtype=np.int64),
        input_particle_count=0,
        input_interval_seconds=0.0,
        allocated_interval_seconds=0.0,
    )


def test_builds_numerator_fractions_quantiles_and_low_sample_mask() -> None:
    """輸出應保留訪格 numerator、分母比例、age midpoint quantiles 與低樣本判定。"""

    result = build_pathway_grid_statistics(
        _pathway(),
        valid_member_denominator=3,
        quantiles=(0.75, 0.25, 0.5),
        low_sample_min_member_count=2,
    )

    assert type(result) is PathwayGridStatistics
    assert result.valid_member_denominator == 3
    assert result.low_sample_min_member_count == 2
    assert "visit_count" not in PathwayGridStatistics.__dataclass_fields__
    assert np.array_equal(result.visit_numerator, [[0, 2, 3]])
    assert result.visit_count is result.visit_numerator
    assert np.allclose(result.visit_fraction, [[0.0, 2 / 3, 1.0]])
    assert np.array_equal(result.low_sample_mask, [[True, False, False]])
    assert list(result.first_passage_quantiles) == [0.75, 0.25, 0.5]
    assert np.allclose(
        result.first_passage_quantiles[0.25],
        [[np.nan, 5.0, 15.0]],
        equal_nan=True,
    )
    assert np.allclose(
        result.first_passage_quantiles[0.5],
        [[np.nan, 5.0, 25.0]],
        equal_nan=True,
    )
    assert np.allclose(
        result.first_passage_quantiles[0.75],
        [[np.nan, 15.0, 25.0]],
        equal_nan=True,
    )


def test_zero_denominator_keeps_fraction_nan() -> None:
    """零有效成員不是零訪問率，比例格網應全部保留 NaN。"""

    result = build_pathway_grid_statistics(
        _zero_pathway(),
        valid_member_denominator=0,
        quantiles=(0.5,),
        low_sample_min_member_count=1,
    )

    assert np.isnan(result.visit_fraction).all()
    assert np.array_equal(result.visit_numerator, [[0]])
    assert np.array_equal(result.low_sample_mask, [[True]])
    assert np.isnan(result.first_passage_quantiles[0.5]).all()


def test_rejects_denominator_mismatch_and_visit_numerator_overflow() -> None:
    """有效分母必須綁定 aggregate 輸入粒子數，且每格 numerator 不可超過分母。"""

    with pytest.raises(ValueError, match="input_particle_count"):
        build_pathway_grid_statistics(
            _pathway(),
            valid_member_denominator=2,
            quantiles=(0.5,),
            low_sample_min_member_count=1,
        )

    # 既有 aggregate constructor 本身也保證 unique count 不超過 input count；這裡刻意
    # 以 object.__new__ 建立只供本層 gate 測試的破壞性 exact instance，確認報告層不會
    # 因未來底層驗證放寬而讓 numerator 靜默超過有效分母。
    invalid = object.__new__(StreamingPathwayAggregate)
    object.__setattr__(invalid, "x_edges_m", np.array([0.0, 10.0]))
    object.__setattr__(invalid, "y_edges_m", np.array([0.0, 5.0]))
    object.__setattr__(invalid, "age_bin_edges_seconds", np.array([0.0, 10.0]))
    object.__setattr__(invalid, "unique_particle_count", np.array([[2]], dtype=np.int64))
    object.__setattr__(invalid, "residence_time_seconds", np.array([[0.0]], dtype=np.float64))
    object.__setattr__(invalid, "first_passage_age_histogram", np.array([[[2]]], dtype=np.int64))
    object.__setattr__(invalid, "input_particle_count", 1)
    object.__setattr__(invalid, "input_interval_seconds", 0.0)
    object.__setattr__(invalid, "allocated_interval_seconds", 0.0)
    with pytest.raises(ValueError, match="visit_numerator"):
        build_pathway_grid_statistics(
            invalid,
            valid_member_denominator=1,
            quantiles=(0.5,),
            low_sample_min_member_count=1,
        )


def test_arrays_edges_and_quantile_mapping_are_defensive_readonly() -> None:
    """輸出陣列與 mapping 不得回寫，且不得和 pathway 的輸入陣列共享記憶體。"""

    pathway = _pathway()
    result = build_pathway_grid_statistics(
        pathway,
        valid_member_denominator=3,
        quantiles=(0.5,),
        low_sample_min_member_count=1,
    )

    arrays = (
        result.x_edges_m,
        result.y_edges_m,
        result.age_bin_edges_seconds,
        result.visit_numerator,
        result.visit_fraction,
        result.residence_time_seconds,
        result.low_sample_mask,
        result.first_passage_quantiles[0.5],
    )
    assert all(not array.flags.writeable for array in arrays)
    assert not np.shares_memory(result.x_edges_m, pathway.x_edges_m)
    assert not np.shares_memory(result.y_edges_m, pathway.y_edges_m)
    assert not np.shares_memory(result.age_bin_edges_seconds, pathway.age_bin_edges_seconds)
    assert not np.shares_memory(result.visit_numerator, pathway.unique_particle_count)
    assert not np.shares_memory(result.residence_time_seconds, pathway.residence_time_seconds)
    assert result.input_interval_seconds == 6.0
    assert result.allocated_interval_seconds == 6.0

    with pytest.raises(ValueError):
        result.residence_time_seconds[0, 0] = 99.0
    with pytest.raises(ValueError):
        result.x_edges_m[0] = -1.0
    with pytest.raises(TypeError):
        result.first_passage_quantiles[0.5] = np.zeros((1, 3))  # type: ignore[index]


def test_post_init_rejects_tampered_low_sample_mask() -> None:
    """低樣本遮罩必須由正式 numerator 與封存門檻精確重建，不接受外部竄改。"""

    result = build_pathway_grid_statistics(
        _pathway(),
        valid_member_denominator=3,
        quantiles=(0.5,),
        low_sample_min_member_count=2,
    )
    tampered_mask = result.low_sample_mask.copy()
    tampered_mask[0, 0] = False
    with pytest.raises(ValueError, match="low_sample_mask"):
        replace(result, low_sample_mask=tampered_mask)


@pytest.mark.parametrize(
    ("cell", "tampered_value"),
    [
        ((0, 1), np.nan),
        ((0, 0), 5.0),
        ((0, 1), 12.5),
    ],
    ids=["visited-cell-to-nan", "empty-cell-to-finite", "visited-cell-to-non-midpoint"],
)
def test_post_init_rejects_tampered_first_passage_quantiles(
    cell: tuple[int, int],
    tampered_value: float,
) -> None:
    """首次進入分位數必須同時符合訪格 numerator 與 canonical age midpoint 契約。"""

    result = build_pathway_grid_statistics(
        _pathway(),
        valid_member_denominator=3,
        quantiles=(0.5,),
        low_sample_min_member_count=1,
    )
    tampered_quantiles = dict(result.first_passage_quantiles)
    tampered_grid = tampered_quantiles[0.5].copy()
    tampered_grid[cell] = tampered_value
    tampered_quantiles[0.5] = tampered_grid

    with pytest.raises(ValueError, match="first_passage_quantiles"):
        replace(result, first_passage_quantiles=tampered_quantiles)


def test_time_provenance_is_preserved_and_conserves_residence() -> None:
    """輸入／分配秒數應原樣封存，且分配範圍與 residence 必須符合時間契約。"""

    result = build_pathway_grid_statistics(
        _pathway(),
        valid_member_denominator=3,
        quantiles=(0.5,),
        low_sample_min_member_count=1,
    )

    # 小於既有絕對容許值的差異仍屬底層 tolerance 內，且 allocated 不超過 input。
    tolerant = replace(
        result,
        input_interval_seconds=6.0 + 5.0e-9,
        allocated_interval_seconds=6.0 + 5.0e-9,
    )
    assert tolerant.input_interval_seconds == pytest.approx(6.0 + 5.0e-9)
    assert tolerant.allocated_interval_seconds == pytest.approx(6.0 + 5.0e-9)

    with pytest.raises(RuntimeError, match="allocated_interval_seconds 不可大於"):
        replace(result, allocated_interval_seconds=7.0)
    with pytest.raises(RuntimeError, match="總和與"):
        replace(
            result,
            input_interval_seconds=6.1,
            allocated_interval_seconds=6.1,
        )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("input_interval_seconds", 6, TypeError),
        ("input_interval_seconds", True, TypeError),
        ("input_interval_seconds", np.float64(6.0), TypeError),
        ("input_interval_seconds", np.nan, ValueError),
        ("input_interval_seconds", np.inf, ValueError),
        ("input_interval_seconds", -1.0, ValueError),
        ("allocated_interval_seconds", 6, TypeError),
        ("allocated_interval_seconds", False, TypeError),
        ("allocated_interval_seconds", np.float64(6.0), TypeError),
        ("allocated_interval_seconds", np.nan, ValueError),
        ("allocated_interval_seconds", np.inf, ValueError),
        ("allocated_interval_seconds", -1.0, ValueError),
    ],
)
def test_time_provenance_type_gates(field: str, value: object, error: type[Exception]) -> None:
    """時間 provenance 只接受原生有限非負 float，不接受 bool 或 NumPy scalar。"""

    result = build_pathway_grid_statistics(
        _pathway(),
        valid_member_denominator=3,
        quantiles=(0.5,),
        low_sample_min_member_count=1,
    )
    with pytest.raises(error):
        replace(result, **{field: value})


def test_dataclass_is_frozen_and_quantile_validation_is_delegated() -> None:
    """產品欄位不可重新綁定，quantile 仍遵守既有 helper 的唯一與範圍規則。"""

    result = build_pathway_grid_statistics(
        _pathway(),
        valid_member_denominator=3,
        quantiles=[0.5],
        low_sample_min_member_count=1,
    )
    with pytest.raises(FrozenInstanceError):
        result.valid_member_denominator = 9  # type: ignore[misc]

    for quantiles in [(), (0.5, 0.5), (0.0,), (1.0,), (np.nan,)]:
        with pytest.raises(ValueError):
            build_pathway_grid_statistics(
                _pathway(),
                valid_member_denominator=3,
                quantiles=quantiles,
                low_sample_min_member_count=1,
            )


@pytest.mark.parametrize(
    ("name", "value", "error"),
    [
        ("pathway", object(), TypeError),
        ("valid_member_denominator", True, TypeError),
        ("valid_member_denominator", np.int64(3), TypeError),
        ("valid_member_denominator", -1, ValueError),
        ("quantiles", {"q": 0.5}, TypeError),
        ("quantiles", (value for value in (0.5,)), TypeError),
        ("low_sample_min_member_count", True, TypeError),
        ("low_sample_min_member_count", np.int64(1), TypeError),
        ("low_sample_min_member_count", 1.0, TypeError),
        ("low_sample_min_member_count", 0, ValueError),
    ],
)
def test_parameter_type_and_value_gates(name: str, value: object, error: type[Exception]) -> None:
    """builder 不應以隱式轉型放寬 exact pathway、scalar 與 sequence 契約。"""

    kwargs: dict[str, object] = {
        "pathway": _pathway(),
        "valid_member_denominator": 3,
        "quantiles": (0.5,),
        "low_sample_min_member_count": 1,
    }
    kwargs[name] = value
    with pytest.raises(error):
        build_pathway_grid_statistics(**kwargs)  # type: ignore[arg-type]
