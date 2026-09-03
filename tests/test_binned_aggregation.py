"""格網化高斯核密度產品的質量守恆、決定性與輸入契約測試。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import lagrangian_backtracking.aggregation as aggregation_module
from lagrangian_backtracking.aggregation import BinnedKDEGrid, binned_gaussian_kde_2d


def _valid_arguments() -> dict[str, Any]:
    """建立 2×2 正計數格網，供各類 invalid 測試只替換單一契約欄位。"""

    return {
        "raw_count": np.array([[1, 2], [3, 4]], dtype=np.int64),
        "x_edges_m": np.array([0.0, 1.0, 2.0]),
        "y_edges_m": np.array([0.0, 1.0, 2.0]),
        "bandwidth_m": 1.0,
        "hdr_levels": (0.50, 0.75, 0.90),
    }


def test_binned_kde_preserves_probability_mass_and_area_density() -> None:
    """每格機率總和應為一，面積密度乘回矩形 cell 面積後須還原相同質量。"""

    x_edges = np.array([0.0, 2.0, 4.0])
    y_edges = np.array([0.0, 5.0, 10.0])
    grid = binned_gaussian_kde_2d(
        np.array([[1, 0], [2, 3]], dtype=np.int64),
        x_edges_m=x_edges,
        y_edges_m=y_edges,
        bandwidth_m=2.0,
    )
    cell_area_m2 = np.diff(y_edges)[:, None] * np.diff(x_edges)[None, :]

    assert isinstance(grid, BinnedKDEGrid)
    assert np.isclose(np.sum(grid.cell_probability), 1.0, rtol=0.0, atol=1.0e-12)
    assert np.allclose(grid.density_per_m2 * cell_area_m2, grid.cell_probability)
    assert np.isclose(np.sum(grid.density_per_m2 * cell_area_m2), 1.0)
    assert np.all(grid.cell_probability >= 0.0)
    assert np.all(grid.density_per_m2 >= 0.0)


def test_binned_kde_defensively_copies_edges_and_counts_and_keeps_raw_total() -> None:
    """caller 後續修改輸入時，不得改變已建立產品的格線、原始計數與交點總數。"""

    counts = np.array([[1, 2], [3, 4]], dtype=np.int32)
    x_edges = np.array([0.0, 2.0, 4.0])
    y_edges = np.array([0.0, 3.0, 6.0])
    expected_counts = counts.astype(np.int64)
    expected_x_edges = x_edges.copy()
    expected_y_edges = y_edges.copy()

    grid = binned_gaussian_kde_2d(
        counts,
        x_edges_m=x_edges,
        y_edges_m=y_edges,
        bandwidth_m=1.5,
    )
    counts[:] = 99
    x_edges[:] = -1.0
    y_edges[:] = -2.0

    assert grid.raw_point_count == 10
    assert grid.raw_count.dtype == np.int64
    assert np.array_equal(grid.raw_count, expected_counts)
    assert np.array_equal(grid.x_edges_m, expected_x_edges)
    assert np.array_equal(grid.y_edges_m, expected_y_edges)
    assert not np.shares_memory(grid.raw_count, counts)
    assert not np.shares_memory(grid.x_edges_m, x_edges)
    assert not np.shares_memory(grid.y_edges_m, y_edges)


def test_binned_kde_hdr_is_deterministic_row_major_and_nested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """相同機率平手時固定採 row-major 順序，重跑結果一致且較高 HDR 包含較低 HDR。"""

    def identity_filter(values: np.ndarray, **_: Any) -> np.ndarray:
        """保留均勻計數，使測試只觀察穩定排序與 HDR 選格規則。"""

        return values.copy()

    monkeypatch.setattr(aggregation_module, "gaussian_filter", identity_filter)
    arguments = {
        "raw_count": np.ones((2, 2), dtype=np.int64),
        "x_edges_m": np.array([0.0, 1.0, 2.0]),
        "y_edges_m": np.array([0.0, 1.0, 2.0]),
        "bandwidth_m": 1.0,
        "hdr_levels": (0.25, 0.50, 0.75),
    }

    first = binned_gaussian_kde_2d(**arguments)
    second = binned_gaussian_kde_2d(**arguments)

    assert np.array_equal(first.hdr_masks[0.25], [[True, False], [False, False]])
    assert np.array_equal(first.hdr_masks[0.50], [[True, True], [False, False]])
    assert np.array_equal(first.hdr_masks[0.75], [[True, True], [True, False]])
    assert np.all(first.hdr_masks[0.25] <= first.hdr_masks[0.50])
    assert np.all(first.hdr_masks[0.50] <= first.hdr_masks[0.75])
    for level in arguments["hdr_levels"]:
        assert np.array_equal(first.hdr_masks[level], second.hdr_masks[level])


def test_binned_kde_calls_gaussian_filter_with_metric_axis_sigma(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """矩形格網須依 y、x 格距換算 sigma，並固定使用零值 constant 邊界。"""

    captured: dict[str, Any] = {}

    def capture_filter(
        values: np.ndarray,
        *,
        sigma: tuple[float, float],
        mode: str,
        cval: float,
    ) -> np.ndarray:
        """記錄傳給 SciPy 的參數，回傳正計數以繼續完成正規化。"""

        captured.update(values=values.copy(), sigma=sigma, mode=mode, cval=cval)
        return values.copy()

    monkeypatch.setattr(aggregation_module, "gaussian_filter", capture_filter)
    counts = np.array([[1, 2], [3, 4]], dtype=np.int64)

    binned_gaussian_kde_2d(
        counts,
        x_edges_m=np.array([0.0, 2.0, 4.0]),
        y_edges_m=np.array([0.0, 5.0, 10.0]),
        bandwidth_m=10.0,
        hdr_levels=(0.50,),
    )

    assert captured["values"].dtype == np.float64
    assert np.array_equal(captured["values"], counts)
    assert np.allclose(captured["sigma"], (2.0, 5.0))
    assert captured["mode"] == "constant"
    assert captured["cval"] == 0.0


@pytest.mark.parametrize(
    "invalid_counts",
    [
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([[True, False], [False, True]]),
        np.array([[1, -1], [2, 3]], dtype=np.int64),
        np.zeros((2, 2), dtype=np.int64),
        np.array([[1, 2], [3, np.iinfo(np.int64).max + 1]], dtype=np.uint64),
    ],
    ids=["float", "bool", "negative", "zero-total", "uint64-overflow"],
)
def test_binned_kde_rejects_invalid_counts(invalid_counts: np.ndarray) -> None:
    """原始計數只接受可安全轉成 int64、非負且總數大於零的整數矩陣。"""

    arguments = _valid_arguments()
    arguments["raw_count"] = invalid_counts
    with pytest.raises(ValueError):
        binned_gaussian_kde_2d(**arguments)


@pytest.mark.parametrize(
    "invalid_shape",
    [
        np.array(1, dtype=np.int64),
        np.array([1, 2], dtype=np.int64),
        np.ones((1, 2), dtype=np.int64),
        np.ones((2, 1), dtype=np.int64),
        np.ones((2, 2, 1), dtype=np.int64),
    ],
    ids=["scalar", "one-dimensional", "wrong-y", "wrong-x", "three-dimensional"],
)
def test_binned_kde_rejects_invalid_count_shape(invalid_shape: np.ndarray) -> None:
    """計數矩陣維度與形狀必須精確對應 y/x cell 數。"""

    arguments = _valid_arguments()
    arguments["raw_count"] = invalid_shape
    with pytest.raises(ValueError):
        binned_gaussian_kde_2d(**arguments)


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize(
    "invalid_edges",
    [
        np.array([0.0]),
        np.array([[0.0, 1.0, 2.0]]),
        np.array([0.0, np.nan, 2.0]),
        np.array([0.0, 2.0, 1.0]),
        np.array([0.0, 1.0, 1.0]),
        np.array([0.0, 1.0, 3.0]),
        np.array(["zero", "one", "two"]),
    ],
    ids=["too-short", "not-1d", "nonfinite", "descending", "duplicate", "nonuniform", "nonnumeric"],
)
def test_binned_kde_rejects_invalid_edges(axis: str, invalid_edges: np.ndarray) -> None:
    """x/y 邊界皆須是可轉數值、有限、嚴格遞增且等距的一維格線。"""

    arguments = _valid_arguments()
    arguments[f"{axis}_edges_m"] = invalid_edges
    with pytest.raises(ValueError):
        binned_gaussian_kde_2d(**arguments)


@pytest.mark.parametrize(
    "invalid_bandwidth",
    [0.0, -1.0, np.nan, np.inf, True, "1.0", [1.0], 1.0 + 0.0j],
    ids=["zero", "negative", "nan", "infinity", "bool", "string", "array", "complex"],
)
def test_binned_kde_rejects_invalid_bandwidth(invalid_bandwidth: Any) -> None:
    """物理頻寬只接受有限且大於零的實數 scalar。"""

    arguments = _valid_arguments()
    arguments["bandwidth_m"] = invalid_bandwidth
    with pytest.raises(ValueError):
        binned_gaussian_kde_2d(**arguments)


@pytest.mark.parametrize(
    "invalid_levels",
    [
        (),
        (0.50, 0.50),
        (0.0,),
        (1.0,),
        (-0.1,),
        (np.nan,),
        (np.inf,),
        (True,),
        ("0.5",),
        ([0.5],),
        None,
    ],
    ids=[
        "empty",
        "duplicate",
        "zero",
        "one",
        "negative",
        "nan",
        "infinity",
        "bool",
        "string",
        "nested",
        "none",
    ],
)
def test_binned_kde_rejects_invalid_hdr_levels(invalid_levels: Any) -> None:
    """HDR 門檻必須是非空、唯一、有限且嚴格位於零與一之間的 scalar 序列。"""

    arguments = _valid_arguments()
    arguments["hdr_levels"] = invalid_levels
    with pytest.raises(ValueError):
        binned_gaussian_kde_2d(**arguments)
