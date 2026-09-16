"""報告層 KDE／HDR 頻寬敏感度產品的 synthetic engineering evidence。

本測試只建立記憶體中的小型合成格網，不讀取 OCM schema 3、NWW3 schema 1 或任何
SERVER 資料，也不產生科學成果。測試中的 `(y_cell, x_cell)` raw count、x/y 公尺
邊界、KDE 頻寬與 HDR 遮罩只用來驗證資料契約、低樣本政策、數值守恆與 defensive
readonly 行為；通過這些測試不代表真實海流、波浪或條件式來源足跡已被驗證。
"""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

import lagrangian_backtracking.aggregation as aggregation_module
import lagrangian_backtracking.report_kde_statistics as kde_module
from lagrangian_backtracking.aggregate_spec import (
    AGGREGATE_SPEC_SCHEMA_VERSION,
    AggregateSpec,
    SiteBoundarySegments,
    SiteGridSpec,
    SiteMetricCRSSpec,
)
from lagrangian_backtracking.aggregation import BinnedKDEGrid
from lagrangian_backtracking.report_kde_statistics import (
    KDEBandwidthLayer,
    KDEEstimateStatus,
    KDESensitivityProduct,
    build_kde_sensitivity_product,
)
from lagrangian_backtracking.report_spec import (
    REPORT_SPEC_SCHEMA_VERSION,
    ReportSpec,
)

_SITE_ID = "synthetic-site"
_SOURCE_HASH = "a" * 64
_CANONICAL_HASH = "b" * 64
_BANDWIDTHS = (1.0, 2.0, 3.0)
_HDR_LEVELS = (0.5, 0.75, 0.9)


@pytest.fixture
def aggregate_spec() -> AggregateSpec:
    """建立 3×4 公尺制 synthetic site grid 與固定三頻寬 aggregate spec。"""

    return AggregateSpec(
        schema_version=AGGREGATE_SPEC_SCHEMA_VERSION,
        run_id="synthetic-kde-run",
        grid_cell_size_m=1,
        site_grids={
            _SITE_ID: SiteGridSpec(
                x_min_m=0,
                x_max_m=4,
                y_min_m=0,
                y_max_m=3,
            )
        },
        site_metric_crs={
            _SITE_ID: SiteMetricCRSSpec(
                projection_method="azimuthal_equidistant_wgs84",
                center_lon_deg=121.0,
                center_lat_deg=24.0,
                linear_unit="m",
                axis_order="x_east_y_north",
            )
        },
        boundary_bin_size_m=1,
        boundary_segment_lengths_m={"synthetic-segment": 4},
        site_boundary_segment_ids={
            _SITE_ID: SiteBoundarySegments(
                local_segment_ids=("synthetic-segment",),
                outer_segment_ids=("synthetic-segment",),
            )
        },
        kde_bandwidths_m=_BANDWIDTHS,
        hdr_levels=_HDR_LEVELS,
        age_bin_edges_seconds=(0, 10),
        bootstrap_replicates=10,
        bootstrap_confidence_level=0.95,
        bootstrap_seed=0,
        denominator_policy="exclude_data_gap_numerical_failure_and_pre_window_deposition_v1",
        source_sha256=_SOURCE_HASH,
        canonical_sha256=_CANONICAL_HASH,
    )


@pytest.fixture
def report_spec(aggregate_spec: AggregateSpec) -> ReportSpec:
    """建立 primary=2 m、最低 raw count=4 的 synthetic report spec。"""

    return ReportSpec(
        schema_version=REPORT_SPEC_SCHEMA_VERSION,
        run_id=aggregate_spec.run_id,
        aggregate_spec_canonical_sha256=aggregate_spec.canonical_sha256,
        primary_kde_bandwidth_m=2,
        minimum_kde_raw_count=4,
        low_sample_min_member_count=1,
        vertical_depth_bin_edges_m=(0.0, 5.0, 10.0),
        representative_trajectory_count_per_site=8,
        representative_selection_policy="stable_hash_core_season_tide_v1",
        representative_selection_seed=0,
        travel_age_quantiles=(0.05, 0.25, 0.5, 0.75, 0.95),
        pathway_first_passage_quantiles=(0.25, 0.5, 0.75),
        figure_formats=("png", "svg", "pdf"),
        raster_dpi=300,
        renderer_style_version="academic_zh_tw_v1",
        language="zh-TW",
        source_sha256=_SOURCE_HASH,
        canonical_sha256=_CANONICAL_HASH,
    )


def _raw_count() -> np.ndarray:
    """回傳具有多個非零 `(y, x)` cell 的 synthetic 原始交點計數。"""

    return np.array(
        [
            [1, 0, 2, 0],
            [0, 3, 0, 0],
            [1, 0, 0, 4],
        ],
        dtype=np.int64,
    )


def _canonical_mask(probability: np.ndarray, level: float) -> np.ndarray:
    """以既有 binned core 的 stable C-order 規則重建一個 synthetic HDR mask。"""

    flat_probability = probability.ravel(order="C")
    descending_order = np.argsort(-flat_probability, kind="stable")
    cumulative_probability = np.cumsum(
        flat_probability[descending_order],
        dtype=np.float64,
    )
    selected_count = int(np.searchsorted(cumulative_probability, level, side="left")) + 1
    selected_count = min(selected_count, descending_order.size)
    mask = np.zeros(flat_probability.size, dtype=np.bool_)
    mask[descending_order[:selected_count]] = True
    return mask.reshape(probability.shape)


def test_available_calls_all_bandwidths_and_binds_primary(
    aggregate_spec: AggregateSpec,
    report_spec: ReportSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """達到門檻時三個 aggregate bandwidth 都必須呼叫既有 core，primary 不可漂移。"""

    calls: list[dict[str, Any]] = []
    real_core = aggregation_module.binned_gaussian_kde_2d

    def spy(raw_count: np.ndarray, **kwargs: Any) -> BinnedKDEGrid:
        """記錄 wrapper 傳入的 raw count、頻寬與 HDR 設定後執行真實 core。"""

        calls.append({"raw_count": raw_count.copy(), **kwargs})
        return real_core(raw_count, **kwargs)

    monkeypatch.setattr(kde_module, "binned_gaussian_kde_2d", spy)
    product = build_kde_sensitivity_product(
        _raw_count(),
        site_id=_SITE_ID,
        aggregate_spec=aggregate_spec,
        report_spec=report_spec,
    )

    assert [call["bandwidth_m"] for call in calls] == list(_BANDWIDTHS)
    assert all(call["hdr_levels"] == _HDR_LEVELS for call in calls)
    assert all(np.array_equal(call["raw_count"], _raw_count()) for call in calls)
    assert product.primary_bandwidth_m == 2.0
    assert product.bandwidths_m == _BANDWIDTHS
    assert tuple(product.layers) == _BANDWIDTHS
    assert all(layer.status is KDEEstimateStatus.available for layer in product.layers.values())
    assert all(layer.grid is not None for layer in product.layers.values())


@pytest.mark.parametrize(
    ("counts", "minimum", "expected_status"),
    [
        (np.zeros((3, 4), dtype=np.int64), 1, KDEEstimateStatus.zero_raw_count),
        (
            np.pad(np.array([[1]], dtype=np.int64), ((0, 2), (0, 3))),
            2,
            KDEEstimateStatus.below_minimum_raw_count,
        ),
    ],
    ids=["zero-raw-count", "below-minimum-raw-count"],
)
def test_zero_and_low_sample_never_call_core(
    aggregate_spec: AggregateSpec,
    report_spec: ReportSpec,
    counts: np.ndarray,
    minimum: int,
    expected_status: KDEEstimateStatus,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """零樣本與低樣本只保留 raw count，三個 layer 都不得建立 KDE grid。"""

    def fail_if_called(*_: Any, **__: Any) -> BinnedKDEGrid:
        """任何呼叫都表示低樣本閘門失效。"""

        raise AssertionError("unavailable KDE status must not call binned core")

    monkeypatch.setattr(kde_module, "binned_gaussian_kde_2d", fail_if_called)
    product = build_kde_sensitivity_product(
        counts,
        site_id=_SITE_ID,
        aggregate_spec=aggregate_spec,
        report_spec=replace(report_spec, minimum_kde_raw_count=minimum),
    )

    assert product.raw_point_count == int(counts.sum())
    assert np.array_equal(product.raw_count, counts)
    assert all(layer.status is expected_status for layer in product.layers.values())
    assert all(layer.grid is None for layer in product.layers.values())


def test_available_grid_preserves_probability_density_and_nested_hdr(
    aggregate_spec: AggregateSpec,
    report_spec: ReportSpec,
) -> None:
    """available layer 應具有公尺面積一致的 density、總和為一且 HDR 由低至高巢狀。"""

    product = build_kde_sensitivity_product(
        _raw_count(),
        site_id=_SITE_ID,
        aggregate_spec=aggregate_spec,
        report_spec=report_spec,
    )
    grid = product.layers[product.primary_bandwidth_m].grid
    assert grid is not None
    cell_area_m2 = np.diff(product.y_edges_m)[:, None] * np.diff(product.x_edges_m)[None, :]
    assert product.x_edges_m.dtype == np.float64
    assert product.y_edges_m.dtype == np.float64
    assert np.array_equal(product.x_edges_m, np.arange(5, dtype=np.float64))
    assert np.array_equal(product.y_edges_m, np.arange(4, dtype=np.float64))
    assert np.isclose(np.sum(grid.cell_probability), 1.0, rtol=0.0, atol=1.0e-12)
    assert np.allclose(grid.density_per_m2 * cell_area_m2, grid.cell_probability)
    assert tuple(grid.hdr_masks) == _HDR_LEVELS
    assert all(mask.dtype == np.bool_ for mask in grid.hdr_masks.values())
    assert np.all(grid.hdr_masks[0.5] <= grid.hdr_masks[0.75])
    assert np.all(grid.hdr_masks[0.75] <= grid.hdr_masks[0.9])


@pytest.mark.parametrize(
    "invalid_counts",
    [
        np.array([[1.0] * 4] * 3),
        np.array([[True] * 4] * 3),
        np.array([[1, -1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int64),
        np.array([[1, 2, 3]], dtype=np.int64),
        np.array(
            [[np.iinfo(np.int64).max + 1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            dtype=np.uint64,
        ),
    ],
    ids=["float", "bool", "negative", "shape", "uint64-overflow"],
)
def test_raw_count_contract_rejects_invalid_dtype_shape_and_range(
    aggregate_spec: AggregateSpec,
    report_spec: ReportSpec,
    invalid_counts: np.ndarray,
) -> None:
    """raw count 必須是精確 shape、非 bool 的非負整數且每格可表示為 int64。"""

    with pytest.raises((TypeError, ValueError)):
        build_kde_sensitivity_product(
            invalid_counts,
            site_id=_SITE_ID,
            aggregate_spec=aggregate_spec,
            report_spec=report_spec,
        )


def test_uint64_cells_sum_with_python_int_without_fixed_width_overflow(
    aggregate_spec: AggregateSpec,
    report_spec: ReportSpec,
) -> None:
    """多個合法 int64 cell 相加可超過 int64，但 raw_point_count 不得繞回。"""

    max_int64 = np.iinfo(np.int64).max
    counts = np.zeros((3, 4), dtype=np.uint64)
    counts[0, 0] = np.uint64(max_int64)
    counts[0, 1] = np.uint64(max_int64)
    product = build_kde_sensitivity_product(
        counts,
        site_id=_SITE_ID,
        aggregate_spec=aggregate_spec,
        report_spec=replace(report_spec, minimum_kde_raw_count=2 * int(max_int64) + 1),
    )

    assert product.raw_point_count == 2 * int(max_int64)
    assert product.raw_count.dtype == np.int64
    assert all(layer.status is KDEEstimateStatus.below_minimum_raw_count for layer in product.layers.values())


def test_spec_site_and_primary_binding_is_fail_closed(
    aggregate_spec: AggregateSpec,
    report_spec: ReportSpec,
) -> None:
    """錯誤 site、run/hash 或未登錄 primary 都不得進入 KDE 計算。"""

    with pytest.raises(ValueError):
        build_kde_sensitivity_product(
            _raw_count(),
            site_id="other-site",
            aggregate_spec=aggregate_spec,
            report_spec=report_spec,
        )
    with pytest.raises(ValueError):
        build_kde_sensitivity_product(
            _raw_count(),
            site_id=_SITE_ID,
            aggregate_spec=aggregate_spec,
            report_spec=replace(report_spec, aggregate_spec_canonical_sha256="c" * 64),
        )
    with pytest.raises(ValueError):
        build_kde_sensitivity_product(
            _raw_count(),
            site_id=_SITE_ID,
            aggregate_spec=aggregate_spec,
            report_spec=replace(report_spec, primary_kde_bandwidth_m=4),
        )


def test_product_and_each_layer_are_defensive_readonly_and_not_shared(
    aggregate_spec: AggregateSpec,
    report_spec: ReportSpec,
) -> None:
    """product、三層 grid 與 HDR mapping 都必須獨立複製並設為唯讀。"""

    counts = _raw_count()
    product = build_kde_sensitivity_product(
        counts,
        site_id=_SITE_ID,
        aggregate_spec=aggregate_spec,
        report_spec=report_spec,
    )
    counts[0, 0] = 99
    assert product.raw_count[0, 0] == 1
    assert product.raw_count.flags.writeable is False
    assert product.x_edges_m.flags.writeable is False
    assert product.y_edges_m.flags.writeable is False
    assert not np.shares_memory(product.raw_count, counts)

    grids = [layer.grid for layer in product.layers.values()]
    assert all(grid is not None for grid in grids)
    concrete_grids = [grid for grid in grids if grid is not None]
    for grid in concrete_grids:
        assert grid.raw_count.flags.writeable is False
        assert grid.x_edges_m.flags.writeable is False
        assert grid.y_edges_m.flags.writeable is False
        assert grid.density_per_m2.flags.writeable is False
        assert grid.cell_probability.flags.writeable is False
        assert isinstance(grid.hdr_masks, MappingProxyType)
        assert all(mask.flags.writeable is False for mask in grid.hdr_masks.values())
        assert not np.shares_memory(grid.raw_count, product.raw_count)
        assert not np.shares_memory(grid.x_edges_m, product.x_edges_m)
    for left_index, left in enumerate(concrete_grids):
        for right in concrete_grids[left_index + 1 :]:
            assert left is not right
            assert not np.shares_memory(left.raw_count, right.raw_count)
            assert not np.shares_memory(left.cell_probability, right.cell_probability)
            assert not np.shares_memory(left.hdr_masks[0.5], right.hdr_masks[0.5])

    with pytest.raises(ValueError):
        product.raw_count[0, 0] = 1
    with pytest.raises(TypeError):
        product.layers[1.0].grid.hdr_masks[0.5] = np.ones((3, 4), dtype=bool)  # type: ignore[union-attr]


def test_direct_constructor_rejects_tampered_cross_fields(
    aggregate_spec: AggregateSpec,
    report_spec: ReportSpec,
) -> None:
    """直接建構或 replace 的 raw total、primary、mapping order 與 core payload tamper 都拒絕。"""

    product = build_kde_sensitivity_product(
        _raw_count(),
        site_id=_SITE_ID,
        aggregate_spec=aggregate_spec,
        report_spec=report_spec,
    )
    with pytest.raises(ValueError):
        replace(product, raw_point_count=product.raw_point_count + 1)
    with pytest.raises(ValueError):
        replace(product, primary_bandwidth_m=9.0)
    with pytest.raises(ValueError):
        replace(product, layers=dict(reversed(tuple(product.layers.items()))))
    with pytest.raises(ValueError):
        replace(product, minimum_raw_count=product.raw_point_count + 1)

    grid = product.layers[1.0].grid
    assert grid is not None
    bad_probability = np.array(grid.cell_probability, copy=True)
    bad_probability[0, 0] += 0.01
    bad_grid = BinnedKDEGrid(
        x_edges_m=np.array(grid.x_edges_m, copy=True),
        y_edges_m=np.array(grid.y_edges_m, copy=True),
        raw_count=np.array(grid.raw_count, copy=True),
        density_per_m2=np.array(grid.density_per_m2, copy=True),
        cell_probability=bad_probability,
        hdr_masks={key: np.array(mask, copy=True) for key, mask in grid.hdr_masks.items()},
        bandwidth_m=grid.bandwidth_m,
        raw_point_count=grid.raw_point_count,
    )
    with pytest.raises(ValueError):
        KDEBandwidthLayer(
            bandwidth_m=1.0,
            status=KDEEstimateStatus.available,
            raw_point_count=product.raw_point_count,
            grid=bad_grid,
        )


def test_hdr_tamper_with_nested_same_count_masks_is_rejected(
    aggregate_spec: AggregateSpec,
    report_spec: ReportSpec,
) -> None:
    """交換次高與最低 probability 後，仍 nested 的錯誤選格必須被 exact gate 拒絕。"""

    product = build_kde_sensitivity_product(
        _raw_count(),
        site_id=_SITE_ID,
        aggregate_spec=aggregate_spec,
        report_spec=report_spec,
    )
    original_grid = product.layers[1.0].grid
    assert original_grid is not None

    # 只交換 probability 的次高與最低 cell；總和、shape、每個原 mask 的 true-count
    # 與 mask 間 nested 關係都保持不變。density 同步重算，讓測試專門命中 HDR exact
    # selection gate，而不是較早的 density=probability/area gate。
    probability = np.array(original_grid.cell_probability, copy=True)
    flat_probability = probability.ravel(order="C")
    descending_order = np.argsort(-flat_probability, kind="stable")
    second_highest = int(descending_order[1])
    lowest = int(descending_order[-1])
    flat_probability[second_highest], flat_probability[lowest] = (
        flat_probability[lowest],
        flat_probability[second_highest],
    )
    swapped_probability = flat_probability.reshape(probability.shape)
    cell_area_m2 = np.diff(original_grid.y_edges_m)[:, None] * np.diff(original_grid.x_edges_m)[None, :]
    swapped_density = swapped_probability / cell_area_m2
    unchanged_masks = {level: np.array(mask, copy=True) for level, mask in original_grid.hdr_masks.items()}
    assert any(
        not np.array_equal(
            unchanged_masks[level],
            _canonical_mask(swapped_probability, level),
        )
        for level in _HDR_LEVELS
    )
    assert all(
        int(unchanged_masks[level].sum()) == int(original_grid.hdr_masks[level].sum())
        for level in _HDR_LEVELS
    )
    assert np.all(unchanged_masks[0.5] <= unchanged_masks[0.75])
    assert np.all(unchanged_masks[0.75] <= unchanged_masks[0.9])

    bad_grid = BinnedKDEGrid(
        x_edges_m=np.array(original_grid.x_edges_m, copy=True),
        y_edges_m=np.array(original_grid.y_edges_m, copy=True),
        raw_count=np.array(original_grid.raw_count, copy=True),
        density_per_m2=swapped_density,
        cell_probability=swapped_probability,
        hdr_masks=unchanged_masks,
        bandwidth_m=1.0,
        raw_point_count=original_grid.raw_point_count,
    )
    with pytest.raises(ValueError):
        KDEBandwidthLayer(
            bandwidth_m=1.0,
            status=KDEEstimateStatus.available,
            raw_point_count=product.raw_point_count,
            grid=bad_grid,
        )

    # 模擬 caller 直接竄改 frozen layer 內的 shallow core payload；product constructor
    # 也必須重新 snapshot 並拒絕，而不能只信任 layer 已經是 dataclass。
    forged_layer = object.__new__(KDEBandwidthLayer)
    object.__setattr__(forged_layer, "bandwidth_m", 1.0)
    object.__setattr__(forged_layer, "status", KDEEstimateStatus.available)
    object.__setattr__(forged_layer, "raw_point_count", product.raw_point_count)
    object.__setattr__(forged_layer, "grid", bad_grid)
    tampered_layers = dict(product.layers)
    tampered_layers[1.0] = forged_layer
    with pytest.raises(ValueError):
        KDESensitivityProduct(
            site_id=product.site_id,
            x_edges_m=product.x_edges_m,
            y_edges_m=product.y_edges_m,
            raw_count=product.raw_count,
            raw_point_count=product.raw_point_count,
            bandwidths_m=product.bandwidths_m,
            hdr_levels=product.hdr_levels,
            primary_bandwidth_m=product.primary_bandwidth_m,
            minimum_raw_count=product.minimum_raw_count,
            layers=tampered_layers,
        )


def test_core_runtime_error_is_propagated_without_rewriting(
    aggregate_spec: AggregateSpec,
    report_spec: ReportSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """既有 core 的 RuntimeError 必須原樣回傳，不能被 wrapper 吞掉或改成 unavailable。"""

    sentinel = RuntimeError("synthetic core numerical failure")

    def raise_sentinel(*_: Any, **__: Any) -> BinnedKDEGrid:
        """模擬 core 的數值失敗。"""

        raise sentinel

    monkeypatch.setattr(kde_module, "binned_gaussian_kde_2d", raise_sentinel)
    with pytest.raises(RuntimeError) as caught:
        build_kde_sensitivity_product(
            _raw_count(),
            site_id=_SITE_ID,
            aggregate_spec=aggregate_spec,
            report_spec=report_spec,
        )
    assert caught.value is sentinel
