"""環境完整性第一增量的串流統計契約測試。

本檔只使用小型 synthetic ``ParticleResult``；位置與環境上下界均為公尺，``z_m``
採海面向上為正，時間與情境識別碼只用來驗證 member identity。測試通過只代表
reducer 的分母、缺值狀態、垂向分箱與記憶體邊界符合程式契約，不代表真實 OCM／NWW3
資料或正式來源足跡已完成。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import FrozenInstanceError, replace
from types import MappingProxyType

import numpy as np
import pytest

from lagrangian_backtracking.aggregate_release_records import ScenarioStratum
from lagrangian_backtracking.aggregate_spec import (
    AggregateSpec,
    SiteBoundarySegments,
    SiteGridSpec,
    SiteMetricCRSSpec,
)
from lagrangian_backtracking.engine import EnvironmentSampleStatus, Observation, ParticleResult
from lagrangian_backtracking.models import ParticleState, ParticleStatus
from lagrangian_backtracking.report_spec import REPORT_SPEC_SCHEMA_VERSION, ReportSpec
from lagrangian_backtracking.report_trajectory_identity import (
    CORE_SEASONS,
    CORE_TIDE_CLASSES,
)
from lagrangian_backtracking.report_trajectory_stream import (
    EnvironmentCompletenessAccumulator,
    EnvironmentCompletenessStatistics,
    MaterialStatisticsProduct,
    RepresentativeSelection,
    TrajectoryReportAccumulator,
    TrajectoryStreamStatistics,
    build_trajectory_stream_statistics,
)

_HASH = "a" * 64
_OTHER_HASH = "b" * 64


def _aggregate_spec(*, site_ids: tuple[str, ...] = ("site-a",)) -> AggregateSpec:
    """建立固定 20×20 公尺格網與 0–10 秒年齡軸的 synthetic AggregateSpec。"""

    site_grids = {
        site_id: SiteGridSpec(x_min_m=0, x_max_m=20, y_min_m=0, y_max_m=20)
        for site_id in site_ids
    }
    site_metric_crs = {
        site_id: SiteMetricCRSSpec(
            projection_method="azimuthal_equidistant_wgs84",
            center_lon_deg=121.0,
            center_lat_deg=25.0,
            linear_unit="m",
            axis_order="x_east_y_north",
        )
        for site_id in site_ids
    }
    site_boundary_segments = {
        site_id: SiteBoundarySegments(
            local_segment_ids=("segment-a",),
            outer_segment_ids=("segment-a",),
        )
        for site_id in site_ids
    }
    return AggregateSpec(
        schema_version="1.0.0",
        run_id="synthetic-stream-run",
        grid_cell_size_m=10,
        site_grids=site_grids,
        site_metric_crs=site_metric_crs,
        boundary_bin_size_m=10,
        boundary_segment_lengths_m={"segment-a": 20},
        site_boundary_segment_ids=site_boundary_segments,
        kde_bandwidths_m=(50, 100, 200),
        hdr_levels=(0.5, 0.75, 0.9),
        age_bin_edges_seconds=(0, 5, 10),
        bootstrap_replicates=1,
        bootstrap_confidence_level=0.95,
        bootstrap_seed=0,
        denominator_policy="exclude_data_gap_numerical_failure_and_pre_window_deposition_v1",
        source_sha256=_HASH,
        canonical_sha256=_HASH,
    )


def _stratum(
    scenario_id: str,
    *,
    season: str,
    tide_class: str,
    site_id: str = "site-a",
    material_id: str = "material-a",
    arrival_time_id: str | None = None,
    arrival_time_utc_ns: int = 100,
) -> ScenarioStratum:
    """建立帶完整 material／arrival／season／tide identity 的 synthetic scenario 列。"""

    return ScenarioStratum(
        scenario_id=scenario_id,
        study_site_id=site_id,
        analysis_region_id="region-a",
        material_id=material_id,
        material_category_zh="合成材料",
        material_family_zh="測試材料",
        representative_shape_zh="球狀",
        behavior_class="neutral",
        settling_velocity_mps=-0.001,
        applicability_condition_zh="僅供串流統計測試",
        calibration_status="未校準",
        evidence_grade="synthetic",
        receptor_id="receptor-a",
        receptor_lon_deg=121.0,
        receptor_lat_deg=25.0,
        receptor_template_z_m_positive_up=-1.0,
        vertical_id="surface",
        arrival_time_id=arrival_time_id or f"arrival-{scenario_id}",
        arrival_time_utc_ns=arrival_time_utc_ns,
        arrival_year=2024,
        season=season,
        tide_class=tide_class,
        phase_or_event="synthetic-event",
        design_version="synthetic-stream-v1",
        initial_z_m_positive_up=None,
        initial_eta_m_positive_up=None,
        initial_bed_z_m_positive_up=None,
        initial_water_column_height_m=None,
        initial_height_above_bed_m=None,
        initial_zcor_lower_m_positive_up=None,
        initial_zcor_upper_m_positive_up=None,
        initial_vertical_bracket_alpha=None,
        initial_source_face_local_index=None,
        initial_source_face_global_index=None,
        initial_wetdry_elem_value=None,
        initial_wetdry_semantics_id=None,
        initial_ocm_month_yyyymm=None,
        initial_ocm_source_time_index=None,
        initial_ocm_time_origin=None,
    )


def _trajectory_result(
    scenario_id: str,
    *,
    member_id: int,
    status: ParticleStatus = ParticleStatus.MAX_AGE,
    site_id: str = "site-a",
) -> ParticleResult:
    """建立兩筆可繪製 observation 的 synthetic 終止結果。"""

    particle_id = f"particle-{scenario_id}-{member_id}"
    observations = [
        _observation(
            particle_id,
            0,
            z_m=-1.0,
            status=ParticleStatus.ACTIVE,
        ),
        _observation(
            particle_id,
            1,
            z_m=-1.0,
            status=status,
        ),
    ]
    return _result(
        site_id=site_id,
        scenario_id=scenario_id,
        member_id=member_id,
        particle_id=particle_id,
        status=status,
        observations=observations,
    )


def _core_strata_and_results(
    *,
    site_id: str = "site-a",
    material_id: str = "material-a",
) -> tuple[tuple[ScenarioStratum, ...], tuple[ParticleResult, ...]]:
    """建立每個核心 season×tide 都恰有一筆候選的最小完整 fixture。"""

    strata: list[ScenarioStratum] = []
    results: list[ParticleResult] = []
    member_id = 0
    for season in CORE_SEASONS:
        for tide_class in CORE_TIDE_CLASSES:
            scenario_id = f"scenario-core-{member_id}"
            strata.append(
                _stratum(
                    scenario_id,
                    season=season,
                    tide_class=tide_class,
                    site_id=site_id,
                    material_id=material_id,
                )
            )
            results.append(
                _trajectory_result(
                    scenario_id,
                    member_id=member_id,
                    site_id=site_id,
                )
            )
            member_id += 1
    return tuple(strata), tuple(results)


def _report_spec(*, edges: tuple[float, ...] = (0.0, 5.0, 10.0)) -> ReportSpec:
    """建立具固定垂向邊界的最小 exact ``ReportSpec`` synthetic fixture。"""

    return ReportSpec(
        schema_version=REPORT_SPEC_SCHEMA_VERSION,
        run_id="synthetic-stream-run",
        aggregate_spec_canonical_sha256=_HASH,
        primary_kde_bandwidth_m=100,
        minimum_kde_raw_count=1,
        low_sample_min_member_count=1,
        vertical_depth_bin_edges_m=edges,
        representative_trajectory_count_per_site=8,
        representative_selection_policy="stable_hash_core_season_tide_v1",
        representative_selection_seed=0,
        travel_age_quantiles=(0.05, 0.25, 0.5, 0.75, 0.95),
        pathway_first_passage_quantiles=(0.25, 0.5, 0.75),
        figure_formats=("png", "svg", "pdf"),
        raster_dpi=300,
        renderer_style_version="academic_zh_tw_v1",
        language="zh-TW",
        source_sha256=_HASH,
        canonical_sha256=_OTHER_HASH,
    )


def _observation(
    particle_id: str,
    index: int,
    *,
    z_m: float,
    status: ParticleStatus,
    sample_status: EnvironmentSampleStatus = EnvironmentSampleStatus.NOT_SAMPLED,
    eta_m: float | None = None,
    bed_z_m: float | None = None,
    qc_flags: int | None = None,
) -> Observation:
    """建立一筆含明確環境缺值語意的 synthetic observation。"""

    if sample_status is EnvironmentSampleStatus.VALID:
        return Observation(
            particle_id,
            100 - index,
            float(index),
            float(index),
            0.0,
            z_m,
            status,
            sample_status,
            eta_m,
            bed_z_m,
            "202401",
            0,
        )
    if sample_status is EnvironmentSampleStatus.INVALID:
        return Observation(
            particle_id,
            100 - index,
            float(index),
            float(index),
            0.0,
            z_m,
            status,
            sample_status,
            None,
            None,
            None,
            7 if qc_flags is None else qc_flags,
        )
    return Observation(
        particle_id,
        100 - index,
        float(index),
        float(index),
        0.0,
        z_m,
        status,
    )


def _result(
    *,
    site_id: str = "site-a",
    scenario_id: str = "scenario-a",
    member_id: int = 0,
    particle_id: str | None = None,
    status: ParticleStatus = ParticleStatus.MAX_AGE,
    observations: list[Observation] | None = None,
) -> ParticleResult:
    """把 synthetic observation 序列包成 exact ``ParticleResult``。"""

    resolved_particle_id = particle_id or f"particle-{scenario_id}-{member_id}"
    return ParticleResult(
        final_state=ParticleState(
            particle_id=resolved_particle_id,
            scenario_id=scenario_id,
            member_id=member_id,
            study_site_id=site_id,
            analysis_region_id="region-a",
            receptor_id="receptor-a",
            x_m=0.0,
            y_m=0.0,
            z_m=-1.0,
            time_utc_ns=90,
            age_seconds=10.0,
            status=status,
        ),
        observations=([] if observations is None else observations),
        events=[],
        step_count=1,
        minimum_clamp_count=0,
    )


def test_valid_and_failure_members_are_separated_and_terminal_not_sampled_is_counted() -> None:
    """有效 member 不混入資料／數值失敗；raw mapping 保留所有具名排除類別。"""

    valid_particle = "particle-valid"
    valid = _result(
        member_id=0,
        particle_id=valid_particle,
        observations=[
            _observation(
                valid_particle,
                0,
                z_m=-1.0,
                status=ParticleStatus.ACTIVE,
                sample_status=EnvironmentSampleStatus.INVALID,
            ),
            _observation(
                valid_particle,
                1,
                z_m=-5.0,
                status=ParticleStatus.ACTIVE,
                sample_status=EnvironmentSampleStatus.VALID,
                eta_m=0.0,
                bed_z_m=-10.0,
            ),
            _observation(
                valid_particle,
                2,
                z_m=-2.0,
                status=ParticleStatus.MAX_AGE,
                sample_status=EnvironmentSampleStatus.NOT_SAMPLED,
            ),
        ],
    )
    gap_particle = "particle-gap"
    gap = _result(
        scenario_id="scenario-gap",
        member_id=1,
        particle_id=gap_particle,
        status=ParticleStatus.DATA_GAP,
        observations=[
            _observation(
                gap_particle,
                0,
                z_m=-1.0,
                status=ParticleStatus.DATA_GAP,
                sample_status=EnvironmentSampleStatus.INVALID,
            )
        ],
    )
    numerical_particle = "particle-numerical"
    numerical = _result(
        scenario_id="scenario-numerical",
        member_id=2,
        particle_id=numerical_particle,
        status=ParticleStatus.NUMERICAL_FAILURE,
        observations=[],
    )
    accumulator = EnvironmentCompletenessAccumulator(_report_spec(), site_ids=["site-a"])
    accumulator.add_result(valid)
    accumulator.add_result(gap)
    accumulator.add_result(numerical)

    product = accumulator.finalize()
    row = product.statistics_by_site["site-a"]
    assert isinstance(product, EnvironmentCompletenessStatistics)
    assert row.total_member_count == 3
    assert row.valid_member_count == 1
    assert row.data_gap_member_count == 1
    assert row.numerical_failure_member_count == 1
    assert row.observation_count == 3
    assert row.valid_observation_count == 1
    assert row.not_sampled_observation_count == 1
    assert row.invalid_observation_count == 1
    assert row.terminal_not_sampled_count == 1
    assert row.member_status_counts == {
        "valid": 1,
        "data_gap": 1,
        "numerical_failure": 1,
        "pre_window_deposition": 0,
    }
    assert row.observation_counts_by_member_status["data_gap"]["invalid"] == 1


def test_depth_and_height_bins_use_positive_down_edges_with_explicit_underflow_overflow() -> None:
    """精確落在 edge 的值進入右側箱；超出設計範圍則各自保留 under/overflow。"""

    particle_id = "particle-boundaries"
    observations = [
        _observation(
            particle_id,
            0,
            z_m=0.0,
            status=ParticleStatus.ACTIVE,
            sample_status=EnvironmentSampleStatus.VALID,
            eta_m=0.0,
            bed_z_m=-10.0,
        ),
        _observation(
            particle_id,
            1,
            z_m=-5.0,
            status=ParticleStatus.ACTIVE,
            sample_status=EnvironmentSampleStatus.VALID,
            eta_m=0.0,
            bed_z_m=-10.0,
        ),
        _observation(
            particle_id,
            2,
            z_m=-10.0,
            status=ParticleStatus.ACTIVE,
            sample_status=EnvironmentSampleStatus.VALID,
            eta_m=0.0,
            bed_z_m=-10.0,
        ),
        _observation(
            particle_id,
            3,
            z_m=5.0e-7,
            status=ParticleStatus.ACTIVE,
            sample_status=EnvironmentSampleStatus.VALID,
            eta_m=0.0,
            bed_z_m=-10.0,
        ),
            _observation(
                particle_id,
                4,
                z_m=-10.0000005,
                status=ParticleStatus.ACTIVE,
                sample_status=EnvironmentSampleStatus.VALID,
                eta_m=0.0,
                bed_z_m=-10.0,
            ),
        _observation(
            particle_id,
            5,
            z_m=0.0,
            status=ParticleStatus.MAX_AGE,
            sample_status=EnvironmentSampleStatus.VALID,
            eta_m=0.0,
            bed_z_m=-11.0,
        ),
    ]
    accumulator = EnvironmentCompletenessAccumulator(_report_spec(), ["site-a"])
    accumulator.add_result(_result(particle_id=particle_id, observations=observations))
    row = accumulator.finalize()["site-a"]

    np.testing.assert_array_equal(row.depth_bin_counts, np.array([2, 2], dtype=np.int64))
    assert row.depth_underflow_count == 1
    assert row.depth_overflow_count == 1
    np.testing.assert_array_equal(
        row.height_above_bed_bin_counts,
        np.array([1, 2], dtype=np.int64),
    )
    assert row.height_above_bed_underflow_count == 1
    assert row.height_above_bed_overflow_count == 2


def test_final_product_is_defensive_and_finalize_closes_accumulator() -> None:
    """產品的 nested mapping／陣列不可寫，且 finalize 後不得再吸收新結果。"""

    source_site_ids = ["site-a"]
    accumulator = EnvironmentCompletenessAccumulator(_report_spec(), source_site_ids)
    source_site_ids.append("site-b")
    accumulator.add_result(_result())
    product = accumulator.finalize()
    row = product.statistics_by_site["site-a"]

    assert accumulator.site_ids == ("site-a",)
    assert isinstance(product.statistics_by_site, MappingProxyType)
    assert isinstance(row.environment_sample_status_counts, MappingProxyType)
    assert row.depth_bin_counts.flags.writeable is False
    with pytest.raises(TypeError):
        product.statistics_by_site["site-b"] = row  # type: ignore[index]
    with pytest.raises(ValueError):
        row.depth_bin_counts[0] = 9
    with pytest.raises(FrozenInstanceError):
        row.valid_member_count = 99  # type: ignore[misc]
    with pytest.raises(RuntimeError, match="finalize"):
        accumulator.add_result(_result(scenario_id="scenario-late", member_id=1))
    assert accumulator.finalize() is product


def test_accumulator_keeps_only_counters_and_identity_sets_not_result_payloads() -> None:
    """大量 observation 只增加計數，不在 reducer 狀態保留結果或觀測序列。"""

    particle_id = "particle-stream"
    observations = [
        _observation(particle_id, index, z_m=-1.0, status=ParticleStatus.ACTIVE)
        for index in range(100)
    ]
    observations[-1] = _observation(
        particle_id,
        99,
        z_m=-1.0,
        status=ParticleStatus.MAX_AGE,
    )
    result = _result(particle_id=particle_id, observations=observations)
    accumulator = EnvironmentCompletenessAccumulator(_report_spec(), ["site-a"])
    accumulator.add_result(result)

    assert accumulator.member_count == 1
    assert not hasattr(accumulator, "_results")
    assert not hasattr(accumulator, "_observations")
    result.observations.clear()
    assert accumulator.finalize()["site-a"].observation_count == 100


@pytest.mark.parametrize("duplicate_kind", ["member", "particle"])
def test_duplicate_scenario_member_or_particle_fails_closed(duplicate_kind: str) -> None:
    """重複 logical member 或 particle identity 不得增加計數，且失敗後不可 finalize。"""

    first = _result(
        scenario_id="scenario-duplicate",
        member_id=0,
        particle_id="particle-first",
    )
    second = (
        _result(
            scenario_id="scenario-duplicate",
            member_id=0,
            particle_id="particle-second",
        )
        if duplicate_kind == "member"
        else _result(
            scenario_id="scenario-other",
            member_id=1,
            particle_id="particle-first",
        )
    )
    accumulator = EnvironmentCompletenessAccumulator(_report_spec(), ["site-a"])
    accumulator.add_result(first)
    with pytest.raises(ValueError, match="重複"):
        accumulator.add_result(second)
    with pytest.raises(ValueError, match="已關閉"):
        accumulator.finalize()


def test_stream_builder_consumes_one_shot_shard_and_returns_three_exact_products() -> None:
    """builder 只遍歷一次 shard，且三個 reducer 與 pathway 使用同一批有效 member。"""

    strata, results = _core_strata_and_results()
    calls = 0

    def one_shot_shard() -> Iterable[ParticleResult]:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise AssertionError("one-shot shard 不可被第二次迭代")
        yield from results

    product = build_trajectory_stream_statistics(
        one_shot_shard(),
        _aggregate_spec(),
        _report_spec(),
        strata,
    )

    assert calls == 1
    assert type(product) is TrajectoryStreamStatistics
    assert type(product.environment_completeness) is EnvironmentCompletenessStatistics
    assert type(product.material_statistics) is MaterialStatisticsProduct
    assert type(product.representative_selection) is RepresentativeSelection
    assert tuple(product.pathway_by_site) == ("site-a",)
    assert product.environment["site-a"].valid_member_count == 8
    assert product.material[("site-a", "material-a")].valid_member_count == 8
    assert product.pathway_by_site["site-a"].valid_member_denominator == 8
    assert sum(product.selection.eligible_count_by_site_stratum["site-a"].values()) == 8


def test_scenario_metadata_binds_site_region_receptor_and_selector_strata() -> None:
    """結果的 join identity 不符 stratum 時，parent 必須在任何 product 前 fail closed。"""

    strata, results = _core_strata_and_results()
    bad_state = replace(results[0].final_state, analysis_region_id="wrong-region")
    bad_result = replace(results[0], final_state=bad_state)
    accumulator = TrajectoryReportAccumulator(
        _aggregate_spec(),
        _report_spec(),
        strata,
    )

    with pytest.raises(ValueError, match="identity"):
        accumulator.add_result(bad_result)
    assert accumulator.state == "failed"
    with pytest.raises(ValueError, match="已關閉"):
        accumulator.finalize()


def test_failed_members_early_observations_are_excluded_from_pathway() -> None:
    """DATA_GAP member 的早期 observation 可供環境稽核，但不得進 pathway 分子或分母。"""

    strata, results = _core_strata_and_results()
    failed_stratum = _stratum(
        "scenario-gap",
        season="DJF",
        tide_class="spring_proxy",
    )
    failed_result = _trajectory_result(
        "scenario-gap",
        member_id=100,
        status=ParticleStatus.DATA_GAP,
    )
    accumulator = TrajectoryReportAccumulator(
        _aggregate_spec(),
        _report_spec(),
        (*strata, failed_stratum),
    )
    accumulator.add_shard((*results, failed_result))
    product = accumulator.finalize()

    environment_row = product.environment["site-a"]
    pathway = product.pathway_by_site["site-a"]
    assert environment_row.total_member_count == 9
    assert environment_row.valid_member_count == 8
    assert environment_row.data_gap_member_count == 1
    assert pathway.valid_member_denominator == 8
    assert int(pathway.visit_numerator.sum()) == 8
    assert int(pathway.visit_numerator.sum()) == 8
    assert product.selection.excluded_invalid_member_count_by_site["site-a"] == 1


def test_event_arrival_is_counted_by_environment_but_excluded_from_selector() -> None:
    """事件到達結果仍屬有效 pathway member，但必須排除於代表軌跡核心候選。"""

    strata, results = _core_strata_and_results()
    event_stratum = _stratum(
        "scenario-event",
        season="DJF",
        tide_class="event",
        arrival_time_id="event-arrival",
    )
    event_result = _trajectory_result("scenario-event", member_id=101)
    accumulator = TrajectoryReportAccumulator(
        _aggregate_spec(),
        _report_spec(),
        (*strata, event_stratum),
    )
    accumulator.add_shard((*results, event_result))
    product = accumulator.finalize()

    assert product.environment["site-a"].valid_member_count == 9
    assert product.pathway_by_site["site-a"].valid_member_denominator == 9
    assert product.selection.excluded_event_arrival_count_by_site["site-a"] == 1
    assert sum(product.selection.eligible_count_by_site_stratum["site-a"].values()) == 8


def test_two_shards_merge_pathway_chunks_without_retaining_old_shard_results() -> None:
    """跨 shard 合併時，pre-window 獨立計數且不進有效 pathway 分母。"""

    strata, results = _core_strata_and_results()
    pre_window_stratum = _stratum(
        "scenario-pre-window",
        season="DJF",
        tide_class="spring_proxy",
    )
    pre_window_particle_id = "particle-scenario-pre-window-108"
    pre_window_result = _result(
        scenario_id="scenario-pre-window",
        member_id=108,
        particle_id=pre_window_particle_id,
        status=ParticleStatus.PRE_WINDOW_DEPOSITION,
        observations=[
            _observation(
                pre_window_particle_id,
                0,
                z_m=-1.0,
                status=ParticleStatus.PRE_WINDOW_DEPOSITION,
            )
        ],
    )
    pre_window_result = replace(
        pre_window_result,
        final_state=replace(
            pre_window_result.final_state,
            time_utc_ns=100,
            age_seconds=0.0,
        ),
        step_count=0,
    )
    accumulator = TrajectoryReportAccumulator(
        _aggregate_spec(),
        _report_spec(),
        (*strata, pre_window_stratum),
    )
    accumulator.add_shard(iter(results[:4]))
    assert accumulator.pathway_chunk_count_by_site["site-a"] == 1
    accumulator.add_shard(iter((*results[4:], pre_window_result)))
    assert accumulator.pathway_chunk_count_by_site["site-a"] == 2

    product = accumulator.finalize()
    environment_row = product.environment["site-a"]
    assert environment_row.total_member_count == 9
    assert environment_row.valid_member_count == 8
    assert environment_row.data_gap_member_count == 0
    assert environment_row.numerical_failure_member_count == 0
    assert environment_row.pre_window_deposition_member_count == 1
    assert environment_row.member_status_counts == {
        "valid": 8,
        ParticleStatus.DATA_GAP.value: 0,
        ParticleStatus.NUMERICAL_FAILURE.value: 0,
        ParticleStatus.PRE_WINDOW_DEPOSITION.value: 1,
    }
    assert environment_row.observation_counts_by_member_status[
        ParticleStatus.PRE_WINDOW_DEPOSITION.value
    ][EnvironmentSampleStatus.NOT_SAMPLED.value] == 1
    assert environment_row.terminal_not_sampled_count_by_member_status[
        ParticleStatus.PRE_WINDOW_DEPOSITION.value
    ] == 0
    assert product.material[("site-a", "material-a")].valid_member_count == 8
    assert product.selection.excluded_invalid_member_count_by_site["site-a"] == 1
    assert product.pathway_by_site["site-a"].valid_member_denominator == 8
    assert int(product.pathway_by_site["site-a"].visit_numerator.sum()) == 8


def test_eight_season_tide_capacity_gate_cannot_be_degraded() -> None:
    """八個核心季節×潮況 strata 少一層時，不得自動補值或 pooled finalize。"""

    strata, results = _core_strata_and_results()
    accumulator = TrajectoryReportAccumulator(
        _aggregate_spec(),
        _report_spec(),
        strata,
    )
    accumulator.add_shard(results[:-1])

    with pytest.raises(ValueError, match="候選不足"):
        accumulator.finalize()
    assert accumulator.state == "failed"
    with pytest.raises(ValueError, match="已關閉"):
        accumulator.add_shard((results[-1],))


def test_zero_valid_site_fails_closed_before_pathway_product() -> None:
    """所有 member 都是 DATA_GAP 時，selector 與 pathway 均不可產生空白替代產品。"""

    strata, results = _core_strata_and_results()
    failed_results = tuple(
        _trajectory_result(
            result.final_state.scenario_id,
            member_id=result.final_state.member_id,
            status=ParticleStatus.DATA_GAP,
        )
        for result in results
    )
    accumulator = TrajectoryReportAccumulator(
        _aggregate_spec(),
        _report_spec(),
        strata,
    )
    accumulator.add_shard(failed_results)

    with pytest.raises(ValueError):
        accumulator.finalize()
    assert accumulator.state == "failed"


def test_parent_defensively_snapshots_topology_axes_and_product_arrays() -> None:
    """caller 修改 topology 後不影響 parent；組合產品的 mapping、dataclass、陣列皆唯讀。"""

    source_site_ids = ["site-a"]
    source_material_ids = ["material-a", "material-empty"]
    strata, results = _core_strata_and_results()
    accumulator = TrajectoryReportAccumulator(
        _aggregate_spec(),
        _report_spec(),
        strata,
        site_ids=source_site_ids,
        material_ids=source_material_ids,
    )
    source_site_ids.append("site-b")
    source_material_ids.append("material-mutated")
    accumulator.add_shard(iter(results))
    product = accumulator.finalize()

    assert accumulator.site_ids == ("site-a",)
    assert accumulator.material_ids == ("material-a", "material-empty")
    np.testing.assert_array_equal(
        product.pathway_by_site["site-a"].x_edges_m,
        np.array([0.0, 10.0, 20.0]),
    )
    np.testing.assert_array_equal(
        product.pathway_by_site["site-a"].y_edges_m,
        np.array([0.0, 10.0, 20.0]),
    )
    np.testing.assert_array_equal(
        product.pathway_by_site["site-a"].age_bin_edges_seconds,
        np.array([0.0, 5.0, 10.0]),
    )
    assert np.issubdtype(product.pathway_by_site["site-a"].visit_numerator.dtype, np.integer)
    assert product.pathway_by_site["site-a"].visit_numerator.flags.writeable is False
    assert isinstance(product.pathway_by_site, MappingProxyType)
    assert isinstance(product.material.statistics_by_site_material, MappingProxyType)
    with pytest.raises(TypeError):
        product.pathway_by_site["site-b"] = product.pathway_by_site["site-a"]  # type: ignore[index]
    with pytest.raises(ValueError):
        product.pathway_by_site["site-a"].visit_numerator[0, 0] = 9
    with pytest.raises(FrozenInstanceError):
        product.pathway_by_site = {}  # type: ignore[misc]
    assert accumulator.finalize() is product


def test_parent_duplicate_identity_fails_closed_before_finalize() -> None:
    """同一 logical member 重複進入 parent 時，不得以 pathway chunk 掩蓋 identity 錯誤。"""

    strata, results = _core_strata_and_results()
    accumulator = TrajectoryReportAccumulator(
        _aggregate_spec(),
        _report_spec(),
        strata,
    )
    with pytest.raises(ValueError, match="重複"):
        accumulator.add_shard((results[0], results[0]))
    assert accumulator.state == "failed"
    with pytest.raises(ValueError, match="已關閉"):
        accumulator.finalize()
