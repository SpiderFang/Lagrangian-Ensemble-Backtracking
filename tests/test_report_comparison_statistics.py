"""F11/T05 比較統計純計算產品的最小完整驗收測試。"""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from test_report_statistics import (
    _integration_aggregate_spec,
    _integration_event_aggregate,
    _integration_scenario,
    _integration_shard,
)

from lagrangian_backtracking.aggregate_release_payload import AggregateReleasePayload
from lagrangian_backtracking.event_aggregation import BoundaryAggregateKey, CrossSiteAggregateKey
from lagrangian_backtracking.report_comparison_statistics import (
    ComparisonParameterDifference,
    ComparisonRatioDifference,
    ReportComparisonStatistics,
    build_report_comparison_statistics,
)
from lagrangian_backtracking.report_material_statistics import (
    MaterialStatistics,
    MaterialStatisticsProduct,
)
from lagrangian_backtracking.report_ratio_statistics import EstimateStatus
from lagrangian_backtracking.report_statistics import ReportStatistics, build_report_statistics


def _report(
    *,
    run_id: str,
    case_id: str,
    event_override: object | None = None,
    material: MaterialStatisticsProduct | None = None,
) -> ReportStatistics:
    """建立兩站、完整路徑格網／核密度估計（KDE）的合成報告統計集合。

    沿用既有統計測試的完整來源聚合資料，只替換執行批次識別碼、實驗案例或明示的
    事件原始計數，讓輸入通過實際的來源／規格核對。material 參數只供拒絕案例使用：
    材質產品缺少可核對的來源，帶來源的建立入口必須拋出 ValueError，不回傳產品。
    合成數值只驗證差值方向、來源限制與缺值語意，不代表真實海洋資料成果。
    """

    source_spec = _integration_aggregate_spec()
    aggregate_spec = replace(
        source_spec,
        run_id=run_id,
        canonical_sha256=(run_id[0] * 64),
    )
    strata = tuple(
        _integration_scenario(
            site_id=site_id,
            receptor_id=f"receptor-{site_id[-1]}",
            index=index,
        )
        for index, site_id in enumerate(("site-a", "site-b"))
    )
    event = _integration_event_aggregate()
    if event_override is not None:
        event = event_override
    payload = AggregateReleasePayload(
        schema_version=payload_schema_version(),
        run_id=run_id,
        run_kind="synthetic",
        experiment_case_id=case_id,
        members_per_scenario=2,
        config_hash="c" * 64,
        checkpoint_input_binding_hash="d" * 64,
        source_run_plan_sha256="e" * 64,
        source_run_progress_sha256="f" * 64,
        source_normalized_config_sha256="0" * 64,
        source_input_inventory_sha256="1" * 64,
        aggregate_spec=aggregate_spec,
        shard_bindings=tuple(_integration_shard(index=index) for index in range(2)),
        scenario_strata=strata,
        event_aggregate=event,
        pathway_by_site={
            site_id: _integration_pathway_for_test()
            for site_id in ("site-a", "site-b")
        },
    )
    report_spec = report_spec_for_test(payload)
    return build_report_statistics(
        payload,
        report_spec=report_spec,
        material_statistics=material,
    )


def payload_schema_version() -> str:
    """回傳既有 aggregate release schema 版本，避免在測試重複硬編碼。"""

    from lagrangian_backtracking.aggregate_release_payload import (
        AGGREGATE_RELEASE_SCHEMA_VERSION,
    )

    return AGGREGATE_RELEASE_SCHEMA_VERSION


def report_spec_for_test(payload: AggregateReleasePayload):
    """依既有 integration report policy 建立與新 run exact binding 的 ReportSpec。"""

    from lagrangian_backtracking.report_spec import REPORT_SPEC_SCHEMA_VERSION, ReportSpec

    return ReportSpec(
        schema_version=REPORT_SPEC_SCHEMA_VERSION,
        run_id=payload.run_id,
        aggregate_spec_canonical_sha256=payload.aggregate_spec.canonical_sha256,
        primary_kde_bandwidth_m=2.0,
        minimum_kde_raw_count=1,
        low_sample_min_member_count=2,
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
        source_sha256="2" * 64,
        canonical_sha256="3" * 64,
    )


def _integration_pathway_for_test():
    """以既有兩秒輸入區間 fixture 建立 pathway aggregate。"""

    from test_report_statistics import _integration_pathway

    return _integration_pathway()


def _changed_event() -> object:
    """建立只改變 outcome 與 A→B raw event 的 comparison event。

    total member denominator 不變；每站把 MAX_AGE 的一名成員改為 DEPOSITED，
    同時移除 A→B 的一名跨站成員。這讓測試可檢查 ratio 與方向性 visit-fraction
    差值，而不改變 scenario、geometry 或時間軸。
    """

    event = _integration_event_aggregate()
    outcome_by_site = {}
    for site_id, outcomes in event.outcome_count_by_site.items():
        changed = dict(outcomes)
        changed["max_age"] = 1
        changed["deposited"] = 1
        outcome_by_site[site_id] = changed
    return replace(
        event,
        outcome_count_by_site=outcome_by_site,
        cross_site_unique_member_count={
            CrossSiteAggregateKey("site-a", "site-b"): 0,
            CrossSiteAggregateKey("site-b", "site-a"): 0,
        },
    )


def _shifted_kde_event() -> object:
    """把 local first-exit raw cell 從 x=0 移到 x=1，形成可解析 HDR 差異。"""

    event = _integration_event_aggregate()
    site_grid_counts = {}
    for site_id, counts in event.site_grid_counts.items():
        site_grid_counts[site_id] = replace(
            counts,
            local_first_exit_count=np.array([[0, 1]], dtype=np.int64),
        )
    return replace(event, site_grid_counts=site_grid_counts)


def test_comparison_products_keep_direction_and_raw_ratio_status() -> None:
    """驗證 outcome、connectivity、source-receptor 與 pathway 差值方向。"""

    baseline = _report(run_id="baseline-run", case_id="baseline")
    comparison = _report(
        run_id="comparison-run",
        case_id="changed-case",
        event_override=_changed_event(),
    )
    result = build_report_comparison_statistics(
        baseline,
        comparison,
        ComparisonParameterDifference(
            name="settling_velocity_mps",
            baseline_value=-0.001,
            comparison_value=-0.002,
            unit="m/s",
        ),
    )

    assert type(result) is ReportComparisonStatistics
    assert result.baseline_run_id == "baseline-run"
    assert result.comparison_run_id == "comparison-run"
    assert result.parameter_difference.delta == pytest.approx(-0.001)

    outcome_difference = result.outcome_ratio_differences_by_site["site-a"]["max_age"]
    assert outcome_difference.baseline_numerator == 2
    assert outcome_difference.baseline_denominator == 2
    assert outcome_difference.comparison_numerator == 1
    assert outcome_difference.comparison_ratio == pytest.approx(0.5)
    assert outcome_difference.difference == pytest.approx(-0.5)

    # connectivity matrix 軸是 source×target；A→B 的變化應在 [0, 1]，不能被
    # 排序、轉置或改以 target row 解讀。
    assert result.connectivity_visit_fraction_difference[0, 1] == pytest.approx(-0.5)
    assert result.connectivity_visit_fraction_difference[1, 0] == pytest.approx(0.0)
    assert result.connectivity_valid_member_denominator_by_source_site["site-a"] == (2, 2)

    source_key = next(iter(result.source_receptor_conditional_ratio_differences_by_key))
    source_difference = result.source_receptor_conditional_ratio_differences_by_key[source_key]
    assert source_difference.difference == pytest.approx(0.0)
    age_difference = result.source_receptor_travel_age_median_differences_by_key[source_key]
    assert age_difference.difference == pytest.approx(0.0)

    np.testing.assert_array_equal(
        result.pathway_visit_fraction_difference_by_site["site-a"],
        np.zeros((1, 2), dtype=np.float64),
    )
    np.testing.assert_array_equal(
        result.pathway_mean_residence_seconds_difference_by_site["site-a"],
        np.zeros((1, 2), dtype=np.float64),
    )


def test_zero_value_ratio_keeps_low_count_and_unavailable_stays_none() -> None:
    """raw numerator=0 且分母=2 仍是可追溯的 low_count，不得誤標 zero denominator。"""

    baseline = _report(run_id="baseline-none", case_id="baseline")
    event = _integration_event_aggregate()
    source_key = next(iter(event.source_receptor_raw_count))
    source_raw = dict(event.source_receptor_raw_count)
    source_raw[source_key] = 0
    source_age = dict(event.source_receptor_travel_age_histogram)
    source_age[source_key] = np.array([0, 0], dtype=np.int64)
    boundary_key = BoundaryAggregateKey(
        study_site_id=source_key.study_site_id,
        boundary_kind=source_key.boundary_kind,
        boundary_segment_id=source_key.boundary_segment_id,
    )
    boundary_raw = dict(event.boundary_arclength_raw_count)
    boundary_raw[boundary_key] = np.array([0, 0], dtype=np.int64)
    boundary_age = dict(event.boundary_travel_age_histogram)
    boundary_age[boundary_key] = np.zeros((2, 2), dtype=np.int64)
    site_grid_counts = dict(event.site_grid_counts)
    site_grid_counts[source_key.study_site_id] = replace(
        site_grid_counts[source_key.study_site_id],
        local_first_exit_count=np.zeros((1, 2), dtype=np.int64),
    )
    comparison_event = replace(
        event,
        boundary_arclength_raw_count=boundary_raw,
        boundary_travel_age_histogram=boundary_age,
        site_grid_counts=site_grid_counts,
        source_receptor_raw_count=source_raw,
        source_receptor_travel_age_histogram=source_age,
    )
    comparison = _report(
        run_id="comparison-none",
        case_id="changed-case",
        event_override=comparison_event,
    )
    result = build_report_comparison_statistics(
        baseline,
        comparison,
        ComparisonParameterDifference("diff", 1.0, 2.0, "m"),
    )
    ratio = result.source_receptor_conditional_ratio_differences_by_key[source_key]
    assert ratio.difference == pytest.approx(-0.5)
    assert ratio.comparison_ratio == pytest.approx(0.0)
    assert ratio.comparison_status is EstimateStatus.low_count
    age = result.source_receptor_travel_age_median_differences_by_key[source_key]
    assert age.difference is None
    assert age.comparison_value is None
    assert age.comparison_status is EstimateStatus.zero_denominator

    # 直接鎖住不可估計 contract：zero denominator 的 None 差值不能被任何實數替代。
    unavailable = ComparisonRatioDifference(
        baseline_numerator=0,
        baseline_denominator=0,
        baseline_ratio=None,
        baseline_status=EstimateStatus.zero_denominator,
        comparison_numerator=0,
        comparison_denominator=0,
        comparison_ratio=None,
        comparison_status=EstimateStatus.zero_denominator,
        difference=None,
    )
    assert unavailable.difference is None


def test_hdr_overlap_uses_each_primary_level_without_recomputing_grid() -> None:
    """兩張 local KDE 支援集合平移後，三個 HDR level 的 Jaccard 仍可追溯。"""

    baseline = _report(run_id="baseline-hdr", case_id="baseline")
    comparison = _report(
        run_id="comparison-hdr",
        case_id="changed-case",
        event_override=_shifted_kde_event(),
    )
    result = build_report_comparison_statistics(
        baseline,
        comparison,
        ComparisonParameterDifference("diff", 1.0, 2.0, "m"),
    )
    overlaps = result.kde_primary_hdr_jaccard_by_site["site-a"]
    assert tuple(overlaps) == (0.5, 0.75, 0.9)
    assert dict(overlaps) == {
        0.5: pytest.approx(0.0),
        0.75: pytest.approx(1.0),
        0.9: pytest.approx(1.0),
    }
    assert result.kde_primary_hdr_status_by_site["site-a"][0.5].available is True


def test_all_semantic_axis_mismatches_fail_closed() -> None:
    """逐項確認 run、report policy、geometry、scenario、arrival、材料與 axes 不可混比。"""

    baseline = _report(run_id="baseline-axis", case_id="baseline")
    parameter = ComparisonParameterDifference("diff", 1.0, 2.0, "m")

    with pytest.raises(ValueError, match="run_id"):
        build_report_comparison_statistics(baseline, baseline, parameter)

    comparison = _report(run_id="comparison-axis", case_id="changed")
    mismatched_spec = replace(
        comparison.report_spec,
        low_sample_min_member_count=3,
    )
    mismatched_policy = build_report_statistics(
        comparison.aggregate_payload,
        report_spec=mismatched_spec,
    )
    with pytest.raises(ValueError, match="ReportSpec|binding"):
        build_report_comparison_statistics(baseline, mismatched_policy, parameter)

    altered_aggregate = replace(
        comparison.aggregate_payload.aggregate_spec,
        kde_bandwidths_m=(1.0, 2.0, 4.0),
        canonical_sha256="d" * 64,
    )
    altered_payload = replace(
        comparison.aggregate_payload,
        aggregate_spec=altered_aggregate,
    )
    altered_spec = replace(
        comparison.report_spec,
        aggregate_spec_canonical_sha256=altered_aggregate.canonical_sha256,
    )
    altered_comparison = build_report_statistics(
        altered_payload,
        report_spec=altered_spec,
    )
    with pytest.raises(ValueError):
        build_report_comparison_statistics(
            baseline,
            altered_comparison,
            parameter,
        )

    altered_stratum = replace(
        comparison.aggregate_payload.scenario_strata[0],
        receptor_id="different-receptor",
    )
    # 既有 AggregateReleasePayload 先封閉 event topology；不合法 receptor join
    # 在進入比較函式前就必須拒絕，不能靠比較層猜測或補建缺列。
    with pytest.raises(ValueError, match="source[_-]receptor"):
        replace(
            comparison.aggregate_payload,
            scenario_strata=(altered_stratum, *comparison.aggregate_payload.scenario_strata[1:]),
        )


def test_material_requires_pure_products_and_cannot_enter_source_bound_comparison() -> None:
    """材質保留無來源產品用法，但不得藉比較入口取得未核對的執行批次身分。

    同一份合成材質產品在帶來源入口須被拒絕；無材質的合法比較仍正常回傳，且兩種
    材質差值及其組合索引明確為空。純產品介面可保存原材質物件，但放在比較任一側
    都必須因缺少來源聚合資料而失敗，不能替它補造來源或零差值。
    """

    records = tuple(
        MaterialStatistics(
            study_site_id=site_id,
            material_id="material-a",
            valid_member_denominator=2,
            first_bed_contact_member_count=1,
            first_bed_contact_fraction=0.5,
            deposited_member_count=0,
            deposited_fraction=0.0,
        )
        for site_id in ("site-a", "site-b")
    )
    material = MaterialStatisticsProduct(
        records=records,
        site_ids=("site-a", "site-b"),
        material_ids=("material-a",),
    )
    # 材質本身型別與數值合法，拒絕原因必須是來源不可核對，而非材質資料損壞。
    with pytest.raises(ValueError, match="payload-bound material_statistics"):
        _report(run_id="baseline-material", case_id="baseline", material=material)

    baseline = _report(run_id="baseline-no-material", case_id="baseline")
    comparison = _report(run_id="comparison-no-material", case_id="changed")
    parameter = ComparisonParameterDifference("diff", 1.0, 2.0, "m")
    result = build_report_comparison_statistics(baseline, comparison, parameter)
    assert result.baseline_run_id == baseline.run_id
    assert result.comparison_run_id == comparison.run_id
    assert dict(result.material_bed_contact_ratio_differences_by_site_material) == {}
    assert dict(result.material_deposited_ratio_differences_by_site_material) == {}
    assert dict(result.material_ratio_differences_by_site_material) == {}

    # 只組合已建立的核心與材質產品，不附加來源或報告規格，保留原純產品用途。
    pure = build_report_statistics(
        outcome_statistics=baseline.outcome,
        connectivity_statistics=baseline.connectivity,
        source_receptor_statistics=baseline.source_receptor,
        material_statistics=material,
    )
    assert pure.material_statistics is material
    assert pure.products["material"] is material
    assert pure.run_id is None
    assert pure.aggregate_payload is None
    assert pure.report_spec is None
    for left, right in ((pure, comparison), (baseline, pure)):
        with pytest.raises(ValueError, match="必須帶有 exact AggregateReleasePayload"):
            build_report_comparison_statistics(left, right, parameter)
    assert pure.run_id is None


def test_defensive_immutability_and_exact_inputs() -> None:
    """輸出 array/mapping 不可回寫，且非 exact ReportStatistics 直接拒絕。"""

    baseline = _report(run_id="baseline-immutable", case_id="baseline")
    comparison = _report(run_id="comparison-immutable", case_id="changed")
    result = build_report_comparison_statistics(
        baseline,
        comparison,
        ComparisonParameterDifference("diff", 1.0, 2.0, "m"),
    )
    with pytest.raises(ValueError):
        result.connectivity_visit_fraction_difference[0, 1] = 1.0
    with pytest.raises(TypeError):
        result.outcome_ratio_differences_by_site["new"] = {}  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        result.site_ids = ("other",)  # type: ignore[misc]
    with pytest.raises(TypeError):
        build_report_comparison_statistics(
            object(),  # type: ignore[arg-type]
            comparison,
            ComparisonParameterDifference("diff", 1.0, 2.0, "m"),
        )
