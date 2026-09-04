"""renderer-facing 統一報告統計 facade 的組合與 immutable 邊界測試。"""

from dataclasses import FrozenInstanceError, replace
from types import MappingProxyType

import numpy as np
import pytest

import lagrangian_backtracking.report_statistics as report_statistics_module
from lagrangian_backtracking.aggregate_release_payload import (
    AGGREGATE_RELEASE_SCHEMA_VERSION,
    AggregateReleasePayload,
)
from lagrangian_backtracking.aggregate_release_records import (
    AggregateShardBinding,
    ScenarioStratum,
)
from lagrangian_backtracking.aggregate_spec import (
    AGGREGATE_SPEC_SCHEMA_VERSION,
    AggregateSpec,
    SiteBoundarySegments,
    SiteGridSpec,
    SiteMetricCRSSpec,
)
from lagrangian_backtracking.event_aggregation import (
    BoundaryAggregateKey,
    CrossSiteAggregateKey,
    EventAggregateChunk,
    ReceptorAggregateKey,
    SiteEventGridCounts,
    SourceReceptorAggregateKey,
)
from lagrangian_backtracking.models import ParticleStatus
from lagrangian_backtracking.report_material_statistics import build_material_statistics
from lagrangian_backtracking.report_matrix_statistics import (
    build_connectivity_statistics,
    build_outcome_statistics,
)
from lagrangian_backtracking.report_source_receptor_statistics import (
    build_source_receptor_statistics,
)
from lagrangian_backtracking.report_spec import REPORT_SPEC_SCHEMA_VERSION, ReportSpec
from lagrangian_backtracking.report_statistics import (
    ReportStatistics,
    build_report_statistics,
)
from lagrangian_backtracking.streaming_aggregation import StreamingPathwayAggregate

_INTEGRATION_SITE_IDS = ("site-a", "site-b")
_INTEGRATION_RECEPTOR_BY_SITE = {
    "site-a": "receptor-a",
    "site-b": "receptor-b",
}
_INTEGRATION_SEGMENT_ID = "segment-1"
_INTEGRATION_AGE_EDGES_SECONDS = (0.0, 10.0, 20.0)
_INTEGRATION_MEMBERS_PER_SCENARIO = 2


def _integration_outcomes() -> dict[str, int]:
    """建立完整停止狀態列，讓兩名合成成員都以最大旅行年齡停止。

    ``AggregateReleasePayload`` 要求每站保留所有非 ``ACTIVE`` 狀態，即使某一狀態
    的 raw count 為零也不可刪列。此 fixture 刻意沒有資料缺口或數值失敗；零值因此
    只代表該停止類別在本小型樣本中沒有成員，並不把缺值或不可用狀態補成物理零。
    """

    return {
        status.value: (
            _INTEGRATION_MEMBERS_PER_SCENARIO
            if status is ParticleStatus.MAX_AGE
            else 0
        )
        for status in ParticleStatus
        if status is not ParticleStatus.ACTIVE
    }


def _integration_aggregate_spec() -> AggregateSpec:
    """建立兩站共用的 1×2 公尺制格網與 0–20 秒旅行年齡規格。

    每站網格的陣列軸固定為 ``(y_cell, x_cell)=(1, 2)``；local 與 outer 角色共用
    同一條 2 m 邊界段，但仍以 ``boundary_kind`` 分開計數。兩站都使用相同的格網
    尺度只為縮小 fixture，站點識別與投影中心仍各自獨立。canonical hash 是 payload
    與 ``ReportSpec`` 的 exact binding 鎖點，本測試不重新計算或放寬其比對。
    """

    site_grids = {
        site_id: SiteGridSpec(
            x_min_m=0.0,
            x_max_m=2.0,
            y_min_m=0.0,
            y_max_m=1.0,
        )
        for site_id in _INTEGRATION_SITE_IDS
    }
    site_metric_crs = {
        site_id: SiteMetricCRSSpec(
            projection_method="azimuthal_equidistant_wgs84",
            center_lon_deg=121.0 + index,
            center_lat_deg=25.0,
            linear_unit="m",
            axis_order="x_east_y_north",
        )
        for index, site_id in enumerate(_INTEGRATION_SITE_IDS)
    }
    site_boundary_segments = {
        site_id: SiteBoundarySegments(
            local_segment_ids=(_INTEGRATION_SEGMENT_ID,),
            outer_segment_ids=(_INTEGRATION_SEGMENT_ID,),
        )
        for site_id in _INTEGRATION_SITE_IDS
    }
    return AggregateSpec(
        schema_version=AGGREGATE_SPEC_SCHEMA_VERSION,
        run_id="report-statistics-integration",
        grid_cell_size_m=1.0,
        site_grids=site_grids,
        site_metric_crs=site_metric_crs,
        boundary_bin_size_m=1.0,
        boundary_segment_lengths_m={_INTEGRATION_SEGMENT_ID: 2.0},
        site_boundary_segment_ids=site_boundary_segments,
        kde_bandwidths_m=(1.0, 2.0, 3.0),
        hdr_levels=(0.5, 0.75, 0.9),
        age_bin_edges_seconds=_INTEGRATION_AGE_EDGES_SECONDS,
        bootstrap_replicates=1,
        bootstrap_confidence_level=0.95,
        bootstrap_seed=0,
        denominator_policy="exclude_data_gap_and_numerical_failure_v1",
        source_sha256="a" * 64,
        canonical_sha256="b" * 64,
    )


def _integration_scenario(
    *,
    site_id: str,
    receptor_id: str,
    index: int,
) -> ScenarioStratum:
    """建立一站一受體的 synthetic scenario，不以零冒充動態初始條件缺值。

    這些資料列只支援跨產品 join 與分母守恆；``initial_*`` 全部為 ``None``，表示
    synthetic payload 沒有正式 OCM 動態初始條件。經緯度僅作資料交換，統計格網仍
    使用 ``AggregateSpec`` 的公尺制 x/y 軸。每站各一列 scenario、每列兩名成員，
    因而各站有效分母與總分母都應精確為 2。
    """

    return ScenarioStratum(
        scenario_id=f"scenario-{index}",
        study_site_id=site_id,
        analysis_region_id=f"region-{index}",
        material_id="material-a",
        material_category_zh="合成材料",
        material_family_zh="測試材料",
        representative_shape_zh="球狀",
        behavior_class="neutral",
        settling_velocity_mps=-0.001,
        applicability_condition_zh="僅供統計整合測試",
        calibration_status="未校準",
        evidence_grade="synthetic",
        receptor_id=receptor_id,
        receptor_lon_deg=121.0 + index,
        receptor_lat_deg=25.0,
        receptor_template_z_m_positive_up=-1.0,
        vertical_id="surface",
        arrival_time_id=f"arrival-{index}",
        arrival_time_utc_ns=1_700_000_000_000_000_000 + index,
        arrival_year=2023,
        season="winter",
        tide_class="flood",
        phase_or_event="fixture-event",
        design_version="report-statistics-integration-v1",
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


def _integration_shard(*, index: int) -> AggregateShardBinding:
    """建立恰好覆蓋一筆 scenario 的 shard 綁定與兩名成員計數。

    scenario 半開區間 ``[index, index + 1)`` 依 payload 原始 tuple 順序相接；粒子、
    observation 與 event 基礎列都設為 2，讓 shard 計數與一列 scenario 乘上
    ``members_per_scenario=2`` 精確守恆。SHA-256 只作 immutable provenance fixture。
    """

    return AggregateShardBinding(
        shard_id=f"shard-{index}",
        scenario_start_index=index,
        scenario_stop_index=index + 1,
        output_relative_path=f"trajectories/shard-{index}.parquet",
        trajectory_manifest_sha256=str(index + 1) * 64,
        particle_count=_INTEGRATION_MEMBERS_PER_SCENARIO,
        observation_count=_INTEGRATION_MEMBERS_PER_SCENARIO,
        event_count=_INTEGRATION_MEMBERS_PER_SCENARIO,
    )


def _integration_event_aggregate() -> EventAggregateChunk:
    """建立兩站完整零拓撲、非零來源事件與 A→B 方向性連通計數。

    每站 local 與 outer 各有一筆事件，對應 raw boundary、來源段—受體 raw count
    及旅行年齡第一個 age bin 都精確為 1；同一 ``boundary_kind`` 的 event share
    分母因此為 1，而受體有效成員分母為 2。跨站矩陣只讓 A→B 為 1、B→A 為 0，
    用來辨識 source row 與 target column 的方向。六類格網均採 ``(y, x)=(1, 2)``，
    failure grid 保留全零但其停止狀態也為零，不會把不可用資料混入有效分母。
    """

    site_grid_counts: dict[str, SiteEventGridCounts] = {}
    boundary_edges: dict[BoundaryAggregateKey, np.ndarray] = {}
    boundary_raw: dict[BoundaryAggregateKey, np.ndarray] = {}
    boundary_age: dict[BoundaryAggregateKey, np.ndarray] = {}
    source_raw: dict[SourceReceptorAggregateKey, int] = {}
    source_age: dict[SourceReceptorAggregateKey, np.ndarray] = {}
    receptor_denominators: dict[ReceptorAggregateKey, int] = {}

    for site_id in _INTEGRATION_SITE_IDS:
        zero_grid = np.zeros((1, 2), dtype=np.int64)
        one_event_grid = np.array([[1, 0]], dtype=np.int64)
        site_grid_counts[site_id] = SiteEventGridCounts(
            local_first_exit_count=one_event_grid,
            outer_first_exit_count=one_event_grid,
            bed_first_contact_count=zero_grid,
            bed_repeated_contact_count=zero_grid,
            data_gap_failure_count=zero_grid,
            numerical_failure_count=zero_grid,
        )
        receptor_id = _INTEGRATION_RECEPTOR_BY_SITE[site_id]
        receptor_denominators[
            ReceptorAggregateKey(site_id, receptor_id)
        ] = _INTEGRATION_MEMBERS_PER_SCENARIO
        for boundary_kind in ("local", "outer"):
            boundary_key = BoundaryAggregateKey(
                study_site_id=site_id,
                boundary_kind=boundary_kind,
                boundary_segment_id=_INTEGRATION_SEGMENT_ID,
            )
            source_key = SourceReceptorAggregateKey(
                study_site_id=site_id,
                receptor_id=receptor_id,
                boundary_kind=boundary_kind,
                boundary_segment_id=_INTEGRATION_SEGMENT_ID,
            )
            boundary_edges[boundary_key] = np.array(
                [0.0, 1.0, 2.0],
                dtype=np.float64,
            )
            boundary_raw[boundary_key] = np.array([1, 0], dtype=np.int64)
            boundary_age[boundary_key] = np.array(
                [[1, 0], [0, 0]],
                dtype=np.int64,
            )
            source_raw[source_key] = 1
            source_age[source_key] = np.array([1, 0], dtype=np.int64)

    return EventAggregateChunk(
        site_grid_counts=site_grid_counts,
        boundary_bin_edges_m=boundary_edges,
        boundary_arclength_raw_count=boundary_raw,
        boundary_travel_age_histogram=boundary_age,
        age_bin_edges_seconds=np.array(
            _INTEGRATION_AGE_EDGES_SECONDS,
            dtype=np.float64,
        ),
        source_receptor_raw_count=source_raw,
        source_receptor_travel_age_histogram=source_age,
        cross_site_unique_member_count={
            CrossSiteAggregateKey("site-a", "site-b"): 1,
            CrossSiteAggregateKey("site-b", "site-a"): 0,
        },
        outcome_count_by_site={
            site_id: _integration_outcomes() for site_id in _INTEGRATION_SITE_IDS
        },
        valid_member_denominator_by_site={
            site_id: _INTEGRATION_MEMBERS_PER_SCENARIO
            for site_id in _INTEGRATION_SITE_IDS
        },
        total_member_count_by_site={
            site_id: _INTEGRATION_MEMBERS_PER_SCENARIO
            for site_id in _INTEGRATION_SITE_IDS
        },
        valid_member_denominator_by_receptor=receptor_denominators,
        input_particle_count=(
            len(_INTEGRATION_SITE_IDS) * _INTEGRATION_MEMBERS_PER_SCENARIO
        ),
    )


def _integration_pathway() -> StreamingPathwayAggregate:
    """建立 1×2 路徑網格，保持秒數守恆與 ``(y, x, age)`` 軸順序。

    兩個 cell 各由一名不同成員首次進入；first-passage age 分別落在 0–10 秒與
    10–20 秒。停留時間合計 2 秒，等於配置的輸入區間。這個小型產品使完整 facade
    可走過既有 pathway builder，同時不從軌跡檔重新推導或補建任何格點。
    """

    return StreamingPathwayAggregate(
        x_edges_m=np.array([0.0, 1.0, 2.0], dtype=np.float64),
        y_edges_m=np.array([0.0, 1.0], dtype=np.float64),
        age_bin_edges_seconds=np.array(
            _INTEGRATION_AGE_EDGES_SECONDS,
            dtype=np.float64,
        ),
        unique_particle_count=np.array([[1, 1]], dtype=np.int64),
        residence_time_seconds=np.array([[1.0, 1.0]], dtype=np.float64),
        first_passage_age_histogram=np.array(
            [[[1, 0], [0, 1]]],
            dtype=np.int64,
        ),
        input_particle_count=_INTEGRATION_MEMBERS_PER_SCENARIO,
        input_interval_seconds=2.0,
        allocated_interval_seconds=2.0,
    )


@pytest.fixture
def exact_payload_and_report_spec() -> tuple[AggregateReleasePayload, ReportSpec]:
    """建立供四個 public builder 共用的 exact payload 與 exact ReportSpec。

    fixture 直接實例化 production immutable records，不使用普通 mapping 假裝 payload。
    ``ReportSpec`` 的 aggregate canonical SHA-256 精確引用同一份 ``AggregateSpec``；
    旅行年齡 quantile、低樣本門檻與 KDE 門檻也由 facade binding 讀取。KDE 門檻設為
    3，而每站 local raw count 只有 1，因此完整 facade 會保留明確 unavailable layer，
    不會把不足樣本的密度補成零。
    """

    aggregate_spec = _integration_aggregate_spec()
    scenario_strata = tuple(
        _integration_scenario(
            site_id=site_id,
            receptor_id=_INTEGRATION_RECEPTOR_BY_SITE[site_id],
            index=index,
        )
        for index, site_id in enumerate(_INTEGRATION_SITE_IDS)
    )
    payload = AggregateReleasePayload(
        schema_version=AGGREGATE_RELEASE_SCHEMA_VERSION,
        run_id=aggregate_spec.run_id,
        run_kind="synthetic",
        experiment_case_id="report-statistics-case",
        members_per_scenario=_INTEGRATION_MEMBERS_PER_SCENARIO,
        config_hash="c" * 64,
        checkpoint_input_binding_hash="d" * 64,
        source_run_plan_sha256="e" * 64,
        source_run_progress_sha256="f" * 64,
        source_normalized_config_sha256="0" * 64,
        source_input_inventory_sha256="1" * 64,
        aggregate_spec=aggregate_spec,
        shard_bindings=tuple(
            _integration_shard(index=index)
            for index in range(len(_INTEGRATION_SITE_IDS))
        ),
        scenario_strata=scenario_strata,
        event_aggregate=_integration_event_aggregate(),
        pathway_by_site={
            site_id: _integration_pathway() for site_id in _INTEGRATION_SITE_IDS
        },
    )
    report_spec = ReportSpec(
        schema_version=REPORT_SPEC_SCHEMA_VERSION,
        run_id=payload.run_id,
        aggregate_spec_canonical_sha256=payload.aggregate_spec.canonical_sha256,
        primary_kde_bandwidth_m=2.0,
        minimum_kde_raw_count=3,
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
    return payload, report_spec


def test_exact_payload_builders_preserve_counts_axes_binding_and_facade_identity(
    exact_payload_and_report_spec: tuple[AggregateReleasePayload, ReportSpec],
) -> None:
    """同一份 exact payload 經四個 builder 後須保留軸、分母、規格與物件 identity。

    此測試不是以拆散的 mapping 模擬輸入，而是讓真實 ``AggregateReleasePayload``
    依序通過停止結果、方向性連通、來源段—受體與統一 facade。斷言同時鎖住 raw
    numerator／denominator、source→target site axis、共同旅行年齡邊界，以及
    facade 對 payload、``ReportSpec`` 與其 renderer-facing 子產品的 identity。
    """

    payload, report_spec = exact_payload_and_report_spec
    outcome = build_outcome_statistics(payload)
    connectivity = build_connectivity_statistics(payload)
    source_receptor = build_source_receptor_statistics(
        payload,
        report_spec=report_spec,
    )
    facade = build_report_statistics(payload, report_spec=report_spec)

    # AggregateSpec 的 insertion order 是兩個矩陣產品共用的 canonical site axis；
    # A→B 必須位於 [0, 1]，不能因排序、轉置或目標／來源名稱混淆而落到 [1, 0]。
    assert outcome.site_ids == _INTEGRATION_SITE_IDS
    assert connectivity.source_site_ids == _INTEGRATION_SITE_IDS
    assert connectivity.target_site_ids == _INTEGRATION_SITE_IDS
    np.testing.assert_array_equal(
        connectivity.raw_matrix,
        np.array([[0, 1], [0, 0]], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        connectivity.row_valid_member_denominator,
        np.array([2, 2], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        connectivity.row_event_denominator,
        np.array([1, 0], dtype=np.int64),
    )

    max_age = outcome.outcome_ratios_by_site["site-a"][ParticleStatus.MAX_AGE.value]
    assert outcome.outcome_count_by_site["site-a"][ParticleStatus.MAX_AGE.value] == 2
    assert outcome.total_member_denominator_by_site["site-a"] == 2
    assert outcome.valid_member_denominator_by_site["site-a"] == 2
    assert max_age.raw_numerator == 2
    assert max_age.raw_denominator == 2

    local_source_key = SourceReceptorAggregateKey(
        "site-a",
        "receptor-a",
        "local",
        _INTEGRATION_SEGMENT_ID,
    )
    local_source = source_receptor[local_source_key]
    assert source_receptor.site_ids == _INTEGRATION_SITE_IDS
    assert source_receptor.minimum_count == report_spec.low_sample_min_member_count
    assert source_receptor.quantiles == report_spec.travel_age_quantiles
    assert local_source.raw_numerator == 1
    assert local_source.valid_receptor_denominator == 2
    assert local_source.conditional_ratio.raw_numerator == 1
    assert local_source.conditional_ratio.raw_denominator == 2
    assert local_source.event_share_denominator == 1
    assert local_source.event_share_ratio.raw_numerator == 1
    assert local_source.event_share_ratio.raw_denominator == 1
    np.testing.assert_array_equal(
        source_receptor.age_bin_edges_seconds,
        np.array(_INTEGRATION_AGE_EDGES_SECONDS, dtype=np.float64),
    )
    np.testing.assert_array_equal(
        local_source.travel_age.raw_count,
        np.array([1, 0], dtype=np.int64),
    )

    # facade 必須引用原 exact payload／spec，且 products index 與命名屬性指向同一
    # immutable 子產品；renderer 因而不需要、也不能自行重算比例或替換 age 軸。
    assert facade.aggregate_payload is payload
    assert facade.report_spec is report_spec
    assert facade.run_id == payload.run_id
    assert facade.products["outcome"] is facade.outcome_statistics
    assert facade.products["connectivity"] is facade.connectivity_statistics
    assert facade.products["source_receptor"] is facade.source_receptor_statistics
    assert facade.outcome is facade.outcome_statistics
    assert facade.connectivity is facade.connectivity_statistics
    assert facade.source_receptor is facade.source_receptor_statistics
    assert facade.outcome_statistics.site_ids == outcome.site_ids
    np.testing.assert_array_equal(
        facade.connectivity_statistics.raw_matrix,
        connectivity.raw_matrix,
    )
    np.testing.assert_array_equal(
        facade.source_receptor_statistics.age_bin_edges_seconds,
        source_receptor.age_bin_edges_seconds,
    )


def test_exact_payload_report_spec_hash_mismatch_fails_closed(
    exact_payload_and_report_spec: tuple[AggregateReleasePayload, ReportSpec],
) -> None:
    """ReportSpec 綁到另一份 aggregate canonical hash 時兩個公開入口都必須拒絕。

    錯誤規格仍是欄位型別完整的 exact ``ReportSpec``，只替換其 aggregate canonical
    SHA-256。來源段 builder 與統一 facade 都應在任何統計產品建立前 fail closed，
    不得因 run ID、age 軸或 KDE 帶寬恰好相同就接受來自另一份 aggregate 的規格。
    """

    payload, report_spec = exact_payload_and_report_spec
    mismatched_report_spec = replace(
        report_spec,
        aggregate_spec_canonical_sha256="4" * 64,
    )

    with pytest.raises(ValueError, match="aggregate canonical hash"):
        build_source_receptor_statistics(
            payload,
            report_spec=mismatched_report_spec,
        )
    with pytest.raises(ValueError, match="aggregate canonical hash"):
        build_report_statistics(payload, report_spec=mismatched_report_spec)


def _products() -> tuple[object, object, object]:
    """建立三個已完成驗證的 typed product，供 facade 只做組合測試。"""

    outcomes = {
        status.value: 0
        for status in ParticleStatus
        if status is not ParticleStatus.ACTIVE
    }
    outcomes[ParticleStatus.MAX_AGE.value] = 1
    outcome = build_outcome_statistics(
        {"site-a": outcomes},
        {"site-a": 1},
        valid_member_denominator_by_site={"site-a": 1},
    )
    connectivity = build_connectivity_statistics(
        {},
        {"site-a": 1},
        site_ids=("site-a",),
    )
    source_key = SourceReceptorAggregateKey("site-a", "receptor-a", "local", "segment-1")
    source = build_source_receptor_statistics(
        {source_key: 1},
        {source_key: np.array([1, 0], dtype=np.int64)},
        {ReceptorAggregateKey("site-a", "receptor-a"): 1},
        age_bin_edges_seconds=np.array([0.0, 10.0, 20.0]),
        quantiles=(0.5,),
    )
    return outcome, connectivity, source


def test_facade_combines_exact_typed_products_for_renderer() -> None:
    """facade 要保留三類產品的 identity，不在 renderer 端重算任何比例。"""

    outcome, connectivity, source = _products()
    facade = build_report_statistics(
        outcome_statistics=outcome,
        connectivity_statistics=connectivity,
        source_receptor_statistics=source,
    )

    assert type(facade) is ReportStatistics
    assert facade.outcome_statistics is outcome
    assert facade.connectivity_statistics is connectivity
    assert facade.source_receptor_statistics is source
    assert facade.outcome is outcome
    assert facade.connectivity is connectivity
    assert facade.source_receptor is source
    assert isinstance(facade.products, MappingProxyType)
    assert facade.products["outcome"] is outcome
    assert facade.products["connectivity"] is connectivity
    assert facade.products["source_receptor"] is source
    with pytest.raises(FrozenInstanceError):
        facade.outcome_statistics = None  # type: ignore[misc]


def test_facade_rejects_non_exact_products_and_does_not_accept_renderer_parameters() -> None:
    """facade 只接受 immutable typed products，錯誤型別與臨時 quantile 必須拒絕。"""

    outcome, connectivity, source = _products()
    with pytest.raises((TypeError, ValueError)):
        build_report_statistics(
            outcome_statistics=object(),  # type: ignore[arg-type]
            connectivity_statistics=connectivity,
            source_receptor_statistics=source,
        )
    with pytest.raises((TypeError, ValueError)):
        build_report_statistics(
            outcome_statistics=outcome,
            connectivity_statistics=connectivity,
            source_receptor_statistics=source,
            quantiles=(0.5,),  # type: ignore[call-arg]
        )


def _bound_core(payload: AggregateReleasePayload, spec: ReportSpec) -> dict[str, object]:
    """從同一合成來源建立三類核心產品，明示沿用報告規格的政策。"""

    return {
        "outcome_statistics": build_outcome_statistics(
            payload, minimum_count=spec.low_sample_min_member_count,
        ),
        "connectivity_statistics": build_connectivity_statistics(
            payload, minimum_count=spec.low_sample_min_member_count,
        ),
        "source_receptor_statistics": build_source_receptor_statistics(payload, report_spec=spec),
    }


@pytest.mark.parametrize("entry", ("constructor", "builder"))
@pytest.mark.parametrize("changed", (
    "outcome_counts", "outcome_failure_grid_axes", "connectivity_counts",
    "connectivity_denominator", "source_counts", "source_denominator",
    "source_histogram", "source_age_axis",
))
def test_payload_bound_core_rejects_other_counts_denominators_and_axes(
    exact_payload_and_report_spec: tuple[AggregateReleasePayload, ReportSpec],
    entry: str,
    changed: str,
) -> None:
    """型別與站點合法但來源不同的資料不得保留原執行批次身分。

    每個替代產品先通過自身建立函式／建構子，故拒絕原因必須是跨產品來源核對。
    停止產品建立函式原有的互斥參數檢查仍保留；另外兩類則進入來源聚合資料組合路徑。
    測試只使用合成計數，不代表真實 OCM／NWW 科學驗證。
    """

    payload, spec = exact_payload_and_report_spec
    core = _bound_core(payload, spec)
    event = payload.event_aggregate
    if changed.startswith("outcome"):
        product_name = "outcome_statistics"
        if changed == "outcome_counts":
            counts = {site: dict(rows) for site, rows in event.outcome_count_by_site.items()}
            counts["site-a"][ParticleStatus.MAX_AGE.value] -= 1
            counts["site-a"][ParticleStatus.FLOW_DOMAIN_EXIT.value] += 1
            altered = replace(core[product_name], outcome_count_by_site=counts)
        else:
            # 失敗數仍為零，但改成另一張格網；只比總數會漏掉此類來源錯配。
            grids = dict(core[product_name].data_gap_failure_grid_by_site)
            grids["site-a"] = np.zeros((2, 1), dtype=np.int64)
            altered = replace(core[product_name], data_gap_failure_grid_by_site=grids)
    elif changed.startswith("connectivity"):
        product_name = "connectivity_statistics"
        counts = dict(event.cross_site_unique_member_count)
        denominators = dict(event.valid_member_denominator_by_site)
        if changed == "connectivity_counts":
            counts[CrossSiteAggregateKey("site-a", "site-b")] = 0
        else:
            denominators["site-a"] += 1
        altered = build_connectivity_statistics(
            counts, denominators, site_ids=_INTEGRATION_SITE_IDS,
            minimum_count=spec.low_sample_min_member_count,
        )
    else:
        product_name = "source_receptor_statistics"
        counts = dict(event.source_receptor_raw_count)
        histograms = dict(event.source_receptor_travel_age_histogram)
        denominators = dict(event.valid_member_denominator_by_receptor)
        edges = event.age_bin_edges_seconds.copy()
        key = SourceReceptorAggregateKey("site-a", "receptor-a", "local", _INTEGRATION_SEGMENT_ID)
        if changed == "source_counts":
            counts[key] = 0
            histograms[key] = np.array([0, 0], dtype=np.int64)
        elif changed == "source_denominator":
            denominators[ReceptorAggregateKey("site-a", "receptor-a")] += 1
        elif changed == "source_histogram":
            histograms[key] = np.array([0, 1], dtype=np.int64)
        else:
            edges *= 2.0
        altered = build_source_receptor_statistics(
            counts, histograms, denominators, age_bin_edges_seconds=edges,
            quantiles=spec.travel_age_quantiles, minimum_count=spec.low_sample_min_member_count,
        )
    core[product_name] = altered
    with pytest.raises(ValueError, match=product_name if entry == "constructor" else "outcome|payload"):
        if entry == "constructor":
            ReportStatistics(**core, aggregate_payload=payload, report_spec=spec)
        else:
            build_report_statistics(payload, report_spec=spec, **{product_name: altered})


@pytest.mark.parametrize("entry", ("constructor", "builder"))
def test_payload_bound_accepts_matching_core_and_pathway(
    exact_payload_and_report_spec: tuple[AggregateReleasePayload, ReportSpec], entry: str,
) -> None:
    """原來源聚合資料衍生的完整產品可重用，逐欄核對後仍保存同一來源與物件。"""

    payload, spec = exact_payload_and_report_spec
    original = build_report_statistics(payload, report_spec=spec)
    core = _bound_core(payload, spec)
    optional = {"pathway_statistics_by_site": original.pathway_by_site, "kde_statistics_by_site": {}}
    if entry == "constructor":
        result = ReportStatistics(**core, aggregate_payload=payload, report_spec=spec, **optional)
    else:
        result = build_report_statistics(
            payload, report_spec=spec, connectivity_statistics=core["connectivity_statistics"],
            source_receptor_statistics=core["source_receptor_statistics"], **optional,
        )
    assert result.run_id == payload.run_id
    assert result.connectivity is core["connectivity_statistics"]
    assert result.source_receptor is core["source_receptor_statistics"]
    assert result.pathway_by_site["site-a"] is original.pathway_by_site["site-a"]


def test_constructor_without_report_spec_still_checks_payload_counts(
    exact_payload_and_report_spec: tuple[AggregateReleasePayload, ReportSpec],
) -> None:
    """省略報告規格不能跳過來源核對；只是不宣稱綁定特定報告統計政策。"""

    payload, spec = exact_payload_and_report_spec
    core = _bound_core(payload, spec)
    assert ReportStatistics(**core, aggregate_payload=payload).run_id == payload.run_id
    counts = {site: dict(rows) for site, rows in payload.event_aggregate.outcome_count_by_site.items()}
    counts["site-a"][ParticleStatus.MAX_AGE.value] -= 1
    counts["site-a"][ParticleStatus.FLOW_DOMAIN_EXIT.value] += 1
    core["outcome_statistics"] = replace(core["outcome_statistics"], outcome_count_by_site=counts)
    with pytest.raises(ValueError, match="outcome_statistics.*payload"):
        ReportStatistics(**core, aggregate_payload=payload)


@pytest.mark.parametrize("entry", ("constructor", "builder"))
def test_matching_source_cannot_override_report_minimum_count(
    exact_payload_and_report_spec: tuple[AggregateReleasePayload, ReportSpec], entry: str,
) -> None:
    """原始數值吻合仍須服從明示規格，避免來源修正削弱既有低樣本政策。"""

    payload, spec = exact_payload_and_report_spec
    core = _bound_core(payload, spec)
    core["connectivity_statistics"] = build_connectivity_statistics(
        payload, minimum_count=spec.low_sample_min_member_count + 1,
    )
    with pytest.raises(ValueError, match="minimum_count.*report_spec"):
        if entry == "constructor":
            ReportStatistics(**core, aggregate_payload=payload, report_spec=spec)
        else:
            build_report_statistics(
                payload, report_spec=spec, connectivity_statistics=core["connectivity_statistics"],
            )


@pytest.mark.parametrize("entry", ("constructor", "builder"))
def test_payload_bound_rejects_pathway_residence_override(
    exact_payload_and_report_spec: tuple[AggregateReleasePayload, ReportSpec], entry: str,
) -> None:
    """停留時間總和、站點與分母皆相同，逐格秒數不同仍須拒絕。"""

    payload, spec = exact_payload_and_report_spec
    original = build_report_statistics(payload, report_spec=spec)
    pathway = dict(original.pathway_by_site)
    pathway["site-a"] = replace(
        pathway["site-a"], residence_time_seconds=np.array([[0.5, 1.5]], dtype=np.float64),
    )
    with pytest.raises(ValueError, match="pathway_statistics_by_site.*payload"):
        if entry == "constructor":
            ReportStatistics(
                **_bound_core(payload, spec), aggregate_payload=payload, report_spec=spec,
                pathway_statistics_by_site=pathway,
            )
        else:
            build_report_statistics(payload, report_spec=spec, pathway_by_site=pathway)


@pytest.mark.parametrize("entry", ("constructor", "builder"))
@pytest.mark.parametrize("optional_name", ("kde_statistics_by_site", "material_statistics"))
def test_unverifiable_overrides_rejected_when_bound_but_pure_products_retained(
    exact_payload_and_report_spec: tuple[AggregateReleasePayload, ReportSpec],
    entry: str, optional_name: str,
) -> None:
    """材料／外部核密度估計（KDE）保留純產品用法，不藉來源資料冒用執行批次身分。"""

    payload, spec = exact_payload_and_report_spec
    original = build_report_statistics(payload, report_spec=spec)
    material = build_material_statistics((), scenario_strata=payload.scenario_strata)
    optional = {optional_name: original.kde_by_site if optional_name.startswith("kde") else material}
    core = _bound_core(payload, spec)
    with pytest.raises(ValueError, match="payload-bound"):
        if entry == "constructor":
            ReportStatistics(**core, aggregate_payload=payload, report_spec=spec, **optional)
        else:
            build_report_statistics(payload, report_spec=spec, **optional)
    pure = ReportStatistics(**core, **optional) if entry == "constructor" else build_report_statistics(
        **core, **optional,
    )
    assert pure.run_id is None
    assert pure.aggregate_payload is None
    assert getattr(pure, optional_name) == optional[optional_name]


@pytest.mark.parametrize("entry", ("constructor", "builder"))
def test_payload_kde_is_built_once_per_site_after_source_validation(
    exact_payload_and_report_spec: tuple[AggregateReleasePayload, ReportSpec],
    monkeypatch: pytest.MonkeyPatch, entry: str,
) -> None:
    """帶來源的入口不為核對重算核密度估計（KDE）；每站只執行一次原建立函式。"""

    payload, spec = exact_payload_and_report_spec
    spec = replace(spec, minimum_kde_raw_count=1)
    calls = []
    real_builder = report_statistics_module.build_kde_sensitivity_product

    def counted_builder(*args, **kwargs):
        """只記錄站點後呼叫實作，避免以假產品繞過來源及數值契約。"""
        calls.append(kwargs["site_id"])
        return real_builder(*args, **kwargs)

    monkeypatch.setattr(report_statistics_module, "build_kde_sensitivity_product", counted_builder)
    if entry == "constructor":
        result = ReportStatistics(
            **_bound_core(payload, spec), aggregate_payload=payload,
            report_spec=spec, kde_statistics_by_site=None,
        )
    else:
        result = build_report_statistics(payload, report_spec=spec)
    assert calls == list(_INTEGRATION_SITE_IDS)
    assert set(result.kde_by_site) == set(_INTEGRATION_SITE_IDS)
    assert all(
        layer.grid is not None
        for product in result.kde_by_site.values()
        for layer in product.layers.values()
    )

    # 同樣要求自動核密度估計，但核心原始計數不符時必須在任何昂貴平滑之前拒絕。
    altered = build_connectivity_statistics(
        {key: 0 for key in payload.event_aggregate.cross_site_unique_member_count},
        payload.event_aggregate.valid_member_denominator_by_site,
        site_ids=_INTEGRATION_SITE_IDS, minimum_count=spec.low_sample_min_member_count,
    )
    with pytest.raises(ValueError, match="connectivity_statistics.*payload"):
        build_report_statistics(payload, report_spec=spec, connectivity_statistics=altered)
    assert calls == list(_INTEGRATION_SITE_IDS)
