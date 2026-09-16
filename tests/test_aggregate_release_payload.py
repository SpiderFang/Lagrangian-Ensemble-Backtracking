"""Aggregate release payload 的最小合法合成資料契約測試。

本檔建立一個自包含的單站人工 fixture，並以重新建構底層 immutable 物件的方式
測試 AggregateReleasePayload 的跨產品防竄改契約。fixture 的水平網格採公尺制，
形狀固定為 ``(y_cell, x_cell) = (1, 2)``；旅行年齡邊界採秒制 ``[0, 1, 2]``，
因此事件與路徑產品都必須使用兩個 age bins。所有計數只用來驗證資料容器的守恆
關係，不代表絕對來源機率或因果歸因。
"""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pytest

from lagrangian_backtracking.aggregate_release_payload import (
    AGGREGATE_RELEASE_SCHEMA_VERSION,
    AggregateReleasePayload,
)
from lagrangian_backtracking.aggregate_release_records import (
    AggregateShardBinding,
    ScenarioStratum,
)
from lagrangian_backtracking.aggregate_spec import (
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
from lagrangian_backtracking.streaming_aggregation import StreamingPathwayAggregate

_RUN_ID = "aggregate-payload-fixture"
_EXPERIMENT_CASE_ID = "synthetic-case"
_SITE_ID = "site-a"
_SECOND_SITE_ID = "site-b"
_RECEPTOR_ID = "receptor-a"
_SECOND_RECEPTOR_ID = "receptor-b"
_SEGMENT_ID = "segment-a"
_ZERO_SEGMENT_ID = "segment-zero"
_AGE_BIN_EDGES_SECONDS = (0.0, 1.0, 2.0)
_MEMBERS_PER_SCENARIO = 2


def _outcome_counts(max_age_count: int) -> dict[str, int]:
    """建立完整的非 ``ACTIVE`` 終止狀態 mapping，保留零值拓撲列。

    ``EventAggregateChunk`` 的 outcome mapping 不只保存目前非零的終止原因；每站
    都必須列出 ``ParticleStatus`` 中除 ``ACTIVE`` 外的完整集合。零計數列代表一個
    已登錄但本批沒有成員落入的狀態，是可追溯拓撲而非可省略資料，因此 fixture
    只讓 ``MAX_AGE`` 承擔原本的成員數，其餘終止狀態固定為 0。此 helper 不改變
    成員總數，回傳的新 dict 由每次呼叫獨立建立，供底層 immutable constructor 封存。
    """

    return {
        status.value: max_age_count if status == ParticleStatus.MAX_AGE else 0
        for status in ParticleStatus
        if status != ParticleStatus.ACTIVE
    }


def _aggregate_spec() -> AggregateSpec:
    """建立一站、1×2 格網的完整公尺制 AggregateSpec。

    x 軸範圍為 0–2 m、y 軸範圍為 0–1 m，搭配 1 m cell 形成一列兩欄；
    local 與 outer 刻意共用同一條 2 m 邊界段，但在事件產品中仍以不同
    ``boundary_kind`` 保留兩種統計角色；local 額外列出一條沒有事件的
    ``segment-zero``，用來驗證零計數 boundary key 仍是 spec 要求的拓撲列。age 軸
    與事件／路徑 fixture 完全共用，以便 AggregateReleasePayload 驗證跨產品的
    精確邊界一致性。
    """

    return AggregateSpec(
        schema_version="1.0.0",
        run_id=_RUN_ID,
        grid_cell_size_m=1.0,
        site_grids={
            _SITE_ID: SiteGridSpec(
                x_min_m=0.0,
                x_max_m=2.0,
                y_min_m=0.0,
                y_max_m=1.0,
            )
        },
        site_metric_crs={
            _SITE_ID: SiteMetricCRSSpec(
                projection_method="azimuthal_equidistant_wgs84",
                center_lon_deg=121.0,
                center_lat_deg=25.0,
                linear_unit="m",
                axis_order="x_east_y_north",
            )
        },
        boundary_bin_size_m=1.0,
        boundary_segment_lengths_m={
            _SEGMENT_ID: 2.0,
            _ZERO_SEGMENT_ID: 2.0,
        },
        site_boundary_segment_ids={
            _SITE_ID: SiteBoundarySegments(
                local_segment_ids=(_SEGMENT_ID, _ZERO_SEGMENT_ID),
                outer_segment_ids=(_SEGMENT_ID,),
            )
        },
        kde_bandwidths_m=(1.0, 2.0, 3.0),
        hdr_levels=(0.5, 0.75, 0.9),
        age_bin_edges_seconds=_AGE_BIN_EDGES_SECONDS,
        bootstrap_replicates=1,
        bootstrap_confidence_level=0.95,
        bootstrap_seed=0,
        denominator_policy="exclude_data_gap_numerical_failure_and_pre_window_deposition_v1",
        source_sha256="a" * 64,
        canonical_sha256="b" * 64,
    )


def _scenario_stratum() -> ScenarioStratum:
    """建立一筆沒有動態初始條件的合成 ScenarioStratum。

    本列的 ``initial_*`` 欄位全部為 ``None``，表示這個 synthetic bundle 不攜帶
    OCM-derived 動態初始條件；這符合非 formal payload 的缺值政策。其餘欄位仍
    明確填入可驗證的站點、受體、材料、到達時間與公尺／秒物理量，避免 fixture
    依賴任何 production builder 或未公開的測試 helper。
    """

    return ScenarioStratum(
        scenario_id="scenario-a",
        study_site_id=_SITE_ID,
        analysis_region_id="region-a",
        material_id="material-a",
        material_category_zh="合成材料",
        material_family_zh="測試材料",
        representative_shape_zh="球狀",
        behavior_class="neutral",
        settling_velocity_mps=-0.001,
        applicability_condition_zh="僅供合成測試",
        calibration_status="未校準",
        evidence_grade="synthetic",
        receptor_id=_RECEPTOR_ID,
        receptor_lon_deg=121.0,
        receptor_lat_deg=25.0,
        receptor_template_z_m_positive_up=-1.0,
        vertical_id="surface",
        arrival_time_id="arrival-a",
        arrival_time_utc_ns=1_700_000_000_000_000_000,
        arrival_year=2023,
        season="winter",
        tide_class="flood",
        phase_or_event="fixture-event",
        design_version="aggregate-payload-test-v1",
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


def _shard_binding() -> AggregateShardBinding:
    """建立覆蓋唯一 scenario 的 ``[0, 1)`` shard，粒子數為 ``M=2``。

    ``observation_count`` 與 ``event_count`` 各至少包含兩個粒子的基礎資料列，
    因而符合 shard 自身的下限；payload 層則再核對此 shard 的粒子總數與
    ``scenario_strata × members_per_scenario`` 完全相等。
    """

    return AggregateShardBinding(
        shard_id="shard-0",
        scenario_start_index=0,
        scenario_stop_index=1,
        output_relative_path="trajectories/shard-0.parquet",
        trajectory_manifest_sha256="c" * 64,
        particle_count=_MEMBERS_PER_SCENARIO,
        observation_count=_MEMBERS_PER_SCENARIO,
        event_count=_MEMBERS_PER_SCENARIO,
    )


def _event_aggregate() -> EventAggregateChunk:
    """建立與單站 2 粒子分母一致的完整 EventAggregateChunk。

    六個站點網格欄位都是 ``(1, 2)``。local 與 outer 各有一筆首次離開事件，另有
    一條 local ``segment-zero`` 的 boundary 與 source-receptor 零列；三組 boundary
    mapping、兩組 source-receptor mapping 仍逐 key 對齊，且 raw／histogram 守恆。
    兩個成員都以 ``MAX_AGE`` 結束，所以 outcome、valid denominator、receptor
    denominator 與 input particle count 都是 2；其餘非 ``ACTIVE`` outcome 雖為 0，
    仍必須保留，因為零列是拓撲而非可省略資料。cross-site mapping 保持精確空值，
    因為本 fixture 只有一個 study site。
    """

    local_boundary_key = BoundaryAggregateKey(
        study_site_id=_SITE_ID,
        boundary_kind="local",
        boundary_segment_id=_SEGMENT_ID,
    )
    outer_boundary_key = BoundaryAggregateKey(
        study_site_id=_SITE_ID,
        boundary_kind="outer",
        boundary_segment_id=_SEGMENT_ID,
    )
    zero_boundary_key = BoundaryAggregateKey(
        study_site_id=_SITE_ID,
        boundary_kind="local",
        boundary_segment_id=_ZERO_SEGMENT_ID,
    )
    local_source_key = SourceReceptorAggregateKey(
        study_site_id=_SITE_ID,
        receptor_id=_RECEPTOR_ID,
        boundary_kind="local",
        boundary_segment_id=_SEGMENT_ID,
    )
    outer_source_key = SourceReceptorAggregateKey(
        study_site_id=_SITE_ID,
        receptor_id=_RECEPTOR_ID,
        boundary_kind="outer",
        boundary_segment_id=_SEGMENT_ID,
    )
    zero_source_key = SourceReceptorAggregateKey(
        study_site_id=_SITE_ID,
        receptor_id=_RECEPTOR_ID,
        boundary_kind="local",
        boundary_segment_id=_ZERO_SEGMENT_ID,
    )
    receptor_key = ReceptorAggregateKey(
        study_site_id=_SITE_ID,
        receptor_id=_RECEPTOR_ID,
    )

    local_grid = np.array([[1, 0]], dtype=np.int64)
    outer_grid = np.array([[1, 0]], dtype=np.int64)
    zero_grid = np.zeros((1, 2), dtype=np.int64)
    site_grid_counts = SiteEventGridCounts(
        local_first_exit_count=local_grid,
        outer_first_exit_count=outer_grid,
        bed_first_contact_count=zero_grid,
        bed_repeated_contact_count=zero_grid,
        data_gap_failure_count=zero_grid,
        numerical_failure_count=zero_grid,
    )

    boundary_edges = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    boundary_raw_count = np.array([1, 0], dtype=np.int64)
    boundary_age_histogram = np.array(
        [[1, 0], [0, 0]],
        dtype=np.int64,
    )
    zero_boundary_raw_count = np.zeros(2, dtype=np.int64)
    zero_boundary_age_histogram = np.zeros((2, 2), dtype=np.int64)
    source_age_histogram = np.array([1, 0], dtype=np.int64)
    zero_source_age_histogram = np.zeros(2, dtype=np.int64)

    return EventAggregateChunk(
        site_grid_counts={_SITE_ID: site_grid_counts},
        boundary_bin_edges_m={
            local_boundary_key: boundary_edges,
            outer_boundary_key: boundary_edges,
            zero_boundary_key: boundary_edges,
        },
        boundary_arclength_raw_count={
            local_boundary_key: boundary_raw_count,
            outer_boundary_key: boundary_raw_count,
            zero_boundary_key: zero_boundary_raw_count,
        },
        boundary_travel_age_histogram={
            local_boundary_key: boundary_age_histogram,
            outer_boundary_key: boundary_age_histogram,
            zero_boundary_key: zero_boundary_age_histogram,
        },
        age_bin_edges_seconds=np.array(_AGE_BIN_EDGES_SECONDS, dtype=np.float64),
        source_receptor_raw_count={
            local_source_key: 1,
            outer_source_key: 1,
            zero_source_key: 0,
        },
        source_receptor_travel_age_histogram={
            local_source_key: source_age_histogram,
            outer_source_key: source_age_histogram,
            zero_source_key: zero_source_age_histogram,
        },
        cross_site_unique_member_count={},
        outcome_count_by_site={_SITE_ID: _outcome_counts(_MEMBERS_PER_SCENARIO)},
        valid_member_denominator_by_site={_SITE_ID: _MEMBERS_PER_SCENARIO},
        total_member_count_by_site={_SITE_ID: _MEMBERS_PER_SCENARIO},
        valid_member_denominator_by_receptor={receptor_key: _MEMBERS_PER_SCENARIO},
        input_particle_count=_MEMBERS_PER_SCENARIO,
    )


def _pathway_aggregate() -> StreamingPathwayAggregate:
    """建立與 spec 對齊的 1×2 pathway 聚合及兩個 age bins。

    每個 x cell 各由一個不同粒子首次進入，因此 unique count 與首次通過
    histogram 都各為 1；兩個 cell 的 residence 秒數合計 2 s，與輸入及配置
    的 interval 完全守恆。陣列軸固定為 ``(y_cell, x_cell)``，不得以轉置掩蓋
    網格定義錯誤。
    """

    return StreamingPathwayAggregate(
        x_edges_m=np.array([0.0, 1.0, 2.0], dtype=np.float64),
        y_edges_m=np.array([0.0, 1.0], dtype=np.float64),
        age_bin_edges_seconds=np.array(_AGE_BIN_EDGES_SECONDS, dtype=np.float64),
        unique_particle_count=np.array([[1, 1]], dtype=np.int64),
        residence_time_seconds=np.array([[1.0, 1.0]], dtype=np.float64),
        first_passage_age_histogram=np.array(
            [[[1, 0], [0, 1]]],
            dtype=np.int64,
        ),
        input_particle_count=_MEMBERS_PER_SCENARIO,
        input_interval_seconds=2.0,
        allocated_interval_seconds=2.0,
    )


def _valid_payload_kwargs(
    *,
    pathway_by_site: dict[str, StreamingPathwayAggregate] | None = None,
) -> dict[str, object]:
    """回傳每次呼叫都重新建立的合法 payload 欄位字典。

    mapping 參數刻意保留 caller 傳入的普通 dict，讓合法測試能在建構後修改
    原 mapping，確認 AggregateReleasePayload 只保存防禦性 snapshot，而不暴露
    外部後續修改的 alias。其他底層 immutable 類別則各自負責其陣列與 mapping
    的封存，因此此 helper 不使用任何 production 私有函式。
    """

    pathway = _pathway_aggregate()
    return {
        "schema_version": AGGREGATE_RELEASE_SCHEMA_VERSION,
        "run_id": _RUN_ID,
        "run_kind": "synthetic",
        "experiment_case_id": _EXPERIMENT_CASE_ID,
        "members_per_scenario": _MEMBERS_PER_SCENARIO,
        "config_hash": "d" * 64,
        "checkpoint_input_binding_hash": "e" * 64,
        "source_run_plan_sha256": "f" * 64,
        "source_run_progress_sha256": "0" * 64,
        "source_normalized_config_sha256": "1" * 64,
        "source_input_inventory_sha256": "2" * 64,
        "aggregate_spec": _aggregate_spec(),
        "shard_bindings": [_shard_binding()],
        "scenario_strata": [_scenario_stratum()],
        "event_aggregate": _event_aggregate(),
        "pathway_by_site": {
            _SITE_ID: pathway
        }
        if pathway_by_site is None
        else pathway_by_site,
    }


def test_valid_synthetic_payload_and_defensive_mapping_snapshot() -> None:
    """合法 synthetic payload 應建立成功，且隔離 caller 的 pathway mapping。"""

    pathway = _pathway_aggregate()
    pathway_by_site = {_SITE_ID: pathway}
    payload = AggregateReleasePayload(**_valid_payload_kwargs(pathway_by_site=pathway_by_site))

    assert payload.run_kind == "synthetic"
    assert payload.run_id == _RUN_ID
    assert payload.members_per_scenario == _MEMBERS_PER_SCENARIO
    assert payload.shard_bindings[0].particle_count == _MEMBERS_PER_SCENARIO
    assert payload.scenario_strata[0].scenario_id == "scenario-a"
    assert payload.event_aggregate.input_particle_count == _MEMBERS_PER_SCENARIO
    assert payload.pathway_by_site[_SITE_ID] is pathway
    assert isinstance(payload.pathway_by_site, MappingProxyType)
    assert payload.event_aggregate.cross_site_unique_member_count == {}

    # payload 建構完成後清空 caller 的普通 dict；若 production 直接保存 alias，
    # payload 將失去必需站點而違反自身拓撲，這正是 defensive mapping snapshot
    # 要防止的變更通道。
    pathway_by_site.clear()
    assert set(payload.pathway_by_site) == {_SITE_ID}
    assert payload.pathway_by_site[_SITE_ID] is pathway


def test_formal_payload_without_dynamic_initial_condition_is_rejected() -> None:
    """正式 payload 不得把 synthetic fixture 的整組缺值初始條件當成合法資料。"""

    kwargs = _valid_payload_kwargs()
    kwargs["run_kind"] = "formal"

    # ScenarioStratum 本身仍是合法的「整組 None」非正式資料；這裡只切換 release
    # 模式，確認拒絕責任落在 payload 的跨產品 formal 規則，而不是較底層 record 建構子。
    with pytest.raises(ValueError, match="動態初始條件"):
        AggregateReleasePayload(**kwargs)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (
        ("schema_version", "0.0.0"),
        ("run_kind", "unregistered"),
        # 使用仍符合 slug 的值，讓錯誤落在 payload 與 aggregate_spec.run_id 的跨產品
        # 精確比對，而不是被識別碼的底層格式檢查提前攔截。
        ("run_id", "tampered-run"),
        # 3 仍是正確的原生 int；它只會使既有 shard 粒子數不再等於 span×M。
        ("members_per_scenario", 3),
    ),
)
def test_invalid_payload_metadata_is_rejected(
    field_name: str,
    invalid_value: object,
) -> None:
    """schema、模式、run 識別與成員倍率的非法跨產品 metadata 都應被拒絕。"""

    kwargs = _valid_payload_kwargs()
    kwargs[field_name] = invalid_value

    with pytest.raises(ValueError):
        AggregateReleasePayload(**kwargs)


def test_aggregate_spec_run_id_mismatch_is_rejected() -> None:
    """aggregate_spec 的 run_id 改變後，不得繼續與 payload 其他產品拼接。"""

    kwargs = _valid_payload_kwargs()
    kwargs["aggregate_spec"] = replace(
        kwargs["aggregate_spec"],
        run_id="tampered-run",
    )

    with pytest.raises(ValueError, match="run_id"):
        AggregateReleasePayload(**kwargs)


@pytest.mark.parametrize(
    "field_name",
    (
        "config_hash",
        "checkpoint_input_binding_hash",
        "source_run_plan_sha256",
        "source_run_progress_sha256",
        "source_normalized_config_sha256",
        "source_input_inventory_sha256",
    ),
)
def test_invalid_provenance_sha256_is_rejected(field_name: str) -> None:
    """六個 provenance SHA-256 任一非法時，payload 都不得建立。"""

    kwargs = _valid_payload_kwargs()
    # 這個值不是 64 碼小寫十六進位摘要；其他欄位保持 fixture 合法，將失敗責任
    # 明確集中在 AggregateReleasePayload 的 provenance 欄位檢查。
    kwargs[field_name] = "not-a-sha256"

    with pytest.raises(ValueError, match="SHA-256"):
        AggregateReleasePayload(**kwargs)


@pytest.mark.parametrize(
    ("tampered_start", "tampered_stop"),
    (
        # 第二份 shard 從 2 開始，與第一份的 stop=1 之間留下 [1, 2) 缺口。
        (2, 3),
        # 第二份 shard 又回到 0，與第一份 [0, 1) 重疊且未向前銜接。
        (0, 1),
    ),
)
def test_shard_ranges_must_be_contiguous_and_non_overlapping(
    tampered_start: int,
    tampered_stop: int,
) -> None:
    """shard 的 scenario 半開區間必須依原始順序無缺口、無重疊地銜接。"""

    first_shard = _shard_binding()
    second_shard = replace(
        first_shard,
        shard_id="shard-1",
        scenario_start_index=tampered_start,
        scenario_stop_index=tampered_stop,
        output_relative_path="trajectories/shard-1.parquet",
    )
    kwargs = _valid_payload_kwargs()
    kwargs["shard_bindings"] = [first_shard, second_shard]

    # 兩筆 AggregateShardBinding 各自都符合底層格式；這裡驗證的是 payload 才知道的
    # 原始 scenario 順序關係，而不是重複測試 shard record 的 start/stop 基本限制。
    with pytest.raises(ValueError, match="scenario range"):
        AggregateReleasePayload(**kwargs)


def test_shard_particle_count_must_equal_range_times_members() -> None:
    """合法 shard record 若粒子數不等於 scenario span×M，payload 應拒絕。"""

    kwargs = _valid_payload_kwargs()
    # particle_count=1 仍滿足 AggregateShardBinding 的 observation/event 下限；只有
    # payload 將 [0, 1) 與 M=2 連接後，才能判定應為 2 而非 1。
    kwargs["shard_bindings"] = [
        replace(_shard_binding(), particle_count=1),
    ]

    with pytest.raises(ValueError, match="particle_count"):
        AggregateReleasePayload(**kwargs)


def test_duplicate_scenario_id_is_rejected() -> None:
    """scenario_strata 不得以重複 scenario_id 讓兩列資料被誤合併。"""

    stratum = _scenario_stratum()
    kwargs = _valid_payload_kwargs()
    kwargs["scenario_strata"] = [
        stratum,
        replace(stratum, scenario_id=stratum.scenario_id),
    ]

    with pytest.raises(ValueError, match="scenario_id"):
        AggregateReleasePayload(**kwargs)


@pytest.mark.parametrize("site_set_case", ("missing", "extra"))
def test_pathway_site_set_must_match_spec(site_set_case: str) -> None:
    """pathway site index 缺站或多站時，跨產品 site set 必須被拒絕。"""

    pathway = _pathway_aggregate()
    if site_set_case == "missing":
        # 保留非空 mapping，讓 pathway value 自身仍是合法物件；site-a 缺少而 site-b
        # 是未登錄站點，錯誤因此落在 payload 的 site set join，而非空 mapping 檢查。
        pathway_by_site = {_SECOND_SITE_ID: pathway}
    else:
        pathway_by_site = {_SITE_ID: pathway, _SECOND_SITE_ID: pathway}

    kwargs = _valid_payload_kwargs(pathway_by_site=pathway_by_site)

    with pytest.raises(ValueError, match="site set"):
        AggregateReleasePayload(**kwargs)


def test_event_grid_shape_must_match_spec() -> None:
    """事件六類網格即使彼此同形，也不得使用與 spec 不同的 (y, x) shape。"""

    wrong_shape_counts = SiteEventGridCounts(
        local_first_exit_count=np.array([[1]], dtype=np.int64),
        outer_first_exit_count=np.array([[1]], dtype=np.int64),
        bed_first_contact_count=np.zeros((1, 1), dtype=np.int64),
        bed_repeated_contact_count=np.zeros((1, 1), dtype=np.int64),
        data_gap_failure_count=np.zeros((1, 1), dtype=np.int64),
        numerical_failure_count=np.zeros((1, 1), dtype=np.int64),
    )
    event = replace(
        _event_aggregate(),
        site_grid_counts={_SITE_ID: wrong_shape_counts},
    )
    kwargs = _valid_payload_kwargs()
    kwargs["event_aggregate"] = event

    with pytest.raises(ValueError, match="shape"):
        AggregateReleasePayload(**kwargs)


@pytest.mark.parametrize("product", ("event", "pathway"))
def test_age_edges_must_match_spec(product: str) -> None:
    """事件與 pathway 的 travel-age 秒數邊界都必須精確共用 spec 定義。"""

    tampered_age_edges = np.array([0.0, 1.0, 3.0], dtype=np.float64)
    if product == "event":
        kwargs = _valid_payload_kwargs()
        kwargs["event_aggregate"] = replace(
            _event_aggregate(),
            age_bin_edges_seconds=tampered_age_edges,
        )
    else:
        pathway = replace(
            _pathway_aggregate(),
            age_bin_edges_seconds=tampered_age_edges,
        )
        kwargs = _valid_payload_kwargs(pathway_by_site={_SITE_ID: pathway})

    # 邊界長度保持兩個 age bins，因此 event/pathway 各自的 constructor 仍然合法；
    # 只有 payload 讀取 spec 與兩個產品的共同時間軸後，才可拒絕這個跨產品差異。
    with pytest.raises(ValueError, match="age"):
        AggregateReleasePayload(**kwargs)


def test_global_particle_count_must_match_strata_times_members() -> None:
    """事件 aggregate 的全域輸入粒子數不得偏離 strata×M。"""

    kwargs = _valid_payload_kwargs()
    kwargs["event_aggregate"] = _event_aggregate_with_member_count(1)

    with pytest.raises(ValueError, match="input_particle_count"):
        AggregateReleasePayload(**kwargs)


def test_site_particle_count_must_match_site_strata_times_members() -> None:
    """各站 total member count 必須等於該站 strata 列數×M，而非只看全域合計。"""

    kwargs = _two_site_payload_kwargs()
    event = kwargs["event_aggregate"]
    assert isinstance(event, EventAggregateChunk)
    kwargs["event_aggregate"] = replace(
        event,
        # A 站少一個成員、B 站多一個成員，仍保持全域 input=4 與底層事件資料關聯
        # 合法；payload 才能精確指出 A 站的 site-level 分母與 strata 不一致。
        outcome_count_by_site={
            _SITE_ID: _outcome_counts(1),
            _SECOND_SITE_ID: _outcome_counts(3),
        },
        valid_member_denominator_by_site={
            _SITE_ID: 1,
            _SECOND_SITE_ID: _MEMBERS_PER_SCENARIO,
        },
        total_member_count_by_site={
            _SITE_ID: 1,
            _SECOND_SITE_ID: 3,
        },
        valid_member_denominator_by_receptor={
            ReceptorAggregateKey(
                study_site_id=_SITE_ID,
                receptor_id=_RECEPTOR_ID,
            ): 1,
            ReceptorAggregateKey(
                study_site_id=_SECOND_SITE_ID,
                receptor_id=_SECOND_RECEPTOR_ID,
            ): _MEMBERS_PER_SCENARIO,
        },
    )

    with pytest.raises(ValueError, match="total_member_count_by_site"):
        AggregateReleasePayload(**kwargs)


def test_pathway_input_particle_count_must_match_site_total() -> None:
    """pathway 的輸入粒子分母必須與同站 strata 粒子數一致。"""

    pathway = replace(_pathway_aggregate(), input_particle_count=1)
    kwargs = _valid_payload_kwargs(pathway_by_site={_SITE_ID: pathway})

    # unique count 每格都是 1，因此 input=1 仍通過 pathway 自身的合法性；跨產品
    # 驗證才會發現該站由一列 scenario×M 應有 2 個輸入粒子。
    with pytest.raises(ValueError, match="input_particle_count"):
        AggregateReleasePayload(**kwargs)


def test_receptor_denominator_key_set_must_match_strata() -> None:
    """受體有效分母的 (site, receptor) key set 不得缺漏或換成未登錄受體。"""

    event = _event_aggregate()
    unexpected_key = ReceptorAggregateKey(
        study_site_id=_SITE_ID,
        receptor_id="receptor-extra",
    )
    tampered_event = replace(
        event,
        # 這個替代 key 仍指向既有站點，且 count=2 仍滿足 EventAggregateChunk 的
        # 站點分母守恆；只有 payload 能以 strata 的 receptor join key 判定它不相符。
        valid_member_denominator_by_receptor={unexpected_key: _MEMBERS_PER_SCENARIO},
    )
    kwargs = _valid_payload_kwargs()
    kwargs["event_aggregate"] = tampered_event

    with pytest.raises(ValueError, match="key set"):
        AggregateReleasePayload(**kwargs)


def test_missing_boundary_zero_key_is_rejected_by_payload_topology() -> None:
    """移除零計數 boundary 拓撲列時，錯誤必須落在 payload 的 spec join。

    被移除的 ``segment-zero`` raw count 本來就是 0，因此同步移除它在三個
    boundary mapping 的列，以及對應 receptor 的 source-receptor 零列，不會改變
    任一事件 raw／travel 合計、grid 守恆或成員分母。重建後的
    ``EventAggregateChunk`` 因而仍是底層合法物件；只有 AggregateReleasePayload
    把事件 key 集合與 AggregateSpec 對照時，才能發現「零列是拓撲，不是可省略資料」。
    測試全程使用 ``dataclasses.replace`` 與普通 mapping 重建，不以低階屬性寫入
    繞過 immutable constructor。
    """

    event = _event_aggregate()
    missing_boundary_key = BoundaryAggregateKey(
        study_site_id=_SITE_ID,
        boundary_kind="local",
        boundary_segment_id=_ZERO_SEGMENT_ID,
    )
    missing_source_key = SourceReceptorAggregateKey(
        study_site_id=_SITE_ID,
        receptor_id=_RECEPTOR_ID,
        boundary_kind="local",
        boundary_segment_id=_ZERO_SEGMENT_ID,
    )

    tampered_event = replace(
        event,
        boundary_bin_edges_m={
            key: value
            for key, value in event.boundary_bin_edges_m.items()
            if key != missing_boundary_key
        },
        boundary_arclength_raw_count={
            key: value
            for key, value in event.boundary_arclength_raw_count.items()
            if key != missing_boundary_key
        },
        boundary_travel_age_histogram={
            key: value
            for key, value in event.boundary_travel_age_histogram.items()
            if key != missing_boundary_key
        },
        source_receptor_raw_count={
            key: value
            for key, value in event.source_receptor_raw_count.items()
            if key != missing_source_key
        },
        source_receptor_travel_age_histogram={
            key: value
            for key, value in event.source_receptor_travel_age_histogram.items()
            if key != missing_source_key
        },
    )
    assert isinstance(tampered_event, EventAggregateChunk)
    assert missing_boundary_key not in tampered_event.boundary_bin_edges_m
    assert missing_source_key not in tampered_event.source_receptor_raw_count

    kwargs = _valid_payload_kwargs()
    kwargs["event_aggregate"] = tampered_event

    with pytest.raises(ValueError, match="boundary"):
        AggregateReleasePayload(**kwargs)


def test_boundary_edges_must_match_canonical_spec_exactly() -> None:
    """邊界格線即使 shape 與 raw 守恆合法，也不得偏離 spec canonical edges。

    這裡只把既有非零 local boundary 的一個內部分箱位置改成另一組嚴格遞增的
    公尺制格線；raw count 仍是兩個 s-bin、travel histogram 仍是
    ``(s_bin, age_bin)=(2, 2)``，且兩者合計都仍為 1。因此底層
    ``EventAggregateChunk`` 可以合法封存，錯誤只能由 payload 將 edges 與 spec
    重新建立的 canonical 格線做 exact comparison 時被發現，而非由 shape 或守恆
    檢查提前攔截。
    """

    event = _event_aggregate()
    boundary_key = BoundaryAggregateKey(
        study_site_id=_SITE_ID,
        boundary_kind="local",
        boundary_segment_id=_SEGMENT_ID,
    )
    tampered_edges = dict(event.boundary_bin_edges_m)
    tampered_edges[boundary_key] = np.array([0.0, 0.5, 2.0], dtype=np.float64)
    tampered_event = replace(
        event,
        boundary_bin_edges_m=tampered_edges,
    )
    assert isinstance(tampered_event, EventAggregateChunk)
    assert tampered_event.boundary_arclength_raw_count[boundary_key].shape == (2,)
    assert tampered_event.boundary_travel_age_histogram[boundary_key].shape == (2, 2)

    kwargs = _valid_payload_kwargs()
    kwargs["event_aggregate"] = tampered_event

    with pytest.raises(ValueError, match="canonical spec"):
        AggregateReleasePayload(**kwargs)


def test_missing_source_receptor_zero_key_is_rejected_by_payload_topology() -> None:
    """移除 source-receptor 零列時，錯誤必須落在完整 cross product 拓撲。

    ``segment-zero`` 的 boundary raw count 是 0，所以刪除其對應受體 source key
    不改變任一 raw／travel 合計；raw 與 age histogram 兩個 source mapping 會同步
    移除同一 key，讓 ``EventAggregateChunk`` 自身仍通過 key 對齊與守恆驗證。這個
    測試特別保留 boundary 零列，藉此區分「boundary topology 缺列」與「每個
    scenario×receptor×boundary 的 source-receptor topology 缺列」。零列是拓撲，
    不是可省略資料。
    """

    event = _event_aggregate()
    missing_source_key = SourceReceptorAggregateKey(
        study_site_id=_SITE_ID,
        receptor_id=_RECEPTOR_ID,
        boundary_kind="local",
        boundary_segment_id=_ZERO_SEGMENT_ID,
    )
    tampered_event = replace(
        event,
        source_receptor_raw_count={
            key: value
            for key, value in event.source_receptor_raw_count.items()
            if key != missing_source_key
        },
        source_receptor_travel_age_histogram={
            key: value
            for key, value in event.source_receptor_travel_age_histogram.items()
            if key != missing_source_key
        },
    )
    assert isinstance(tampered_event, EventAggregateChunk)
    assert missing_source_key not in tampered_event.source_receptor_raw_count
    assert any(
        key.boundary_segment_id == _ZERO_SEGMENT_ID
        for key in tampered_event.boundary_bin_edges_m
    )

    kwargs = _valid_payload_kwargs()
    kwargs["event_aggregate"] = tampered_event

    with pytest.raises(ValueError, match="cross product"):
        AggregateReleasePayload(**kwargs)


def test_missing_ordered_cross_site_pair_is_rejected_by_payload_topology() -> None:
    """兩站 event 缺少任一方向的零計數 pair 時，payload 應拒絕缺列。

    兩站 fixture 先保留 A→B 與 B→A 兩個 ordered distinct pair，且兩列都是 0；
    這些零值是固定跨站拓撲，不是可由下游省略的空摘要。本測試只刪除 A→B，其他
    事件 boundary、source-receptor、outcome 與所有粒子守恆均不變，並以
    ``dataclasses.replace`` 重建底層合法 chunk，確保錯誤責任落在 payload 的跨站
    exact key set，而不是 EventAggregateChunk 的基本型別檢查。
    """

    kwargs = _two_site_payload_kwargs()
    event = kwargs["event_aggregate"]
    assert isinstance(event, EventAggregateChunk)
    missing_pair = CrossSiteAggregateKey(
        source_study_site_id=_SITE_ID,
        target_study_site_id=_SECOND_SITE_ID,
    )
    cross_site_counts = {
        key: value
        for key, value in event.cross_site_unique_member_count.items()
        if key != missing_pair
    }
    tampered_event = replace(
        event,
        cross_site_unique_member_count=cross_site_counts,
    )
    assert isinstance(tampered_event, EventAggregateChunk)
    assert missing_pair not in tampered_event.cross_site_unique_member_count
    assert CrossSiteAggregateKey(
        source_study_site_id=_SECOND_SITE_ID,
        target_study_site_id=_SITE_ID,
    ) in tampered_event.cross_site_unique_member_count

    kwargs["event_aggregate"] = tampered_event

    with pytest.raises(ValueError, match="ordered distinct"):
        AggregateReleasePayload(**kwargs)


def test_missing_zero_non_active_outcome_key_is_rejected_by_payload_topology() -> None:
    """移除零計數非 ACTIVE outcome 列時，payload 應拒絕不完整狀態拓撲。

    ``FLOW_DOMAIN_EXIT`` 在合法單站 fixture 中的計數是 0；刪除它不改變 outcome
    合計 2，也不改變 valid／total member denominator。底層事件容器仍能以合法的
    mapping 與守恆封存，但 payload 必須依 ``ParticleStatus`` 的完整非 ACTIVE
    value 集合辨識缺列，因為零列是拓撲而非可省略資料。
    """

    event = _event_aggregate()
    missing_status = ParticleStatus.FLOW_DOMAIN_EXIT.value
    assert event.outcome_count_by_site[_SITE_ID][missing_status] == 0
    outcomes_by_site = {
        site_id: dict(outcomes)
        for site_id, outcomes in event.outcome_count_by_site.items()
    }
    del outcomes_by_site[_SITE_ID][missing_status]
    tampered_event = replace(
        event,
        outcome_count_by_site=outcomes_by_site,
    )
    assert isinstance(tampered_event, EventAggregateChunk)
    assert missing_status not in tampered_event.outcome_count_by_site[_SITE_ID]

    kwargs = _valid_payload_kwargs()
    kwargs["event_aggregate"] = tampered_event

    with pytest.raises(ValueError, match="ParticleStatus"):
        AggregateReleasePayload(**kwargs)


def _event_aggregate_with_member_count(member_count: int) -> EventAggregateChunk:
    """重建底層仍守恆、但全域成員數不同的事件 aggregate tamper fixture。

    單站事件容器的 ``input_particle_count`` 必須等於站點 total、outcome 與受體
    分母的關聯合計，因此不能只改一個 scalar 後直接交給 constructor。這個 helper
    同步重建其必要的關聯欄位，保留邊界事件與網格 shape 不變，把測試焦點限定在
    AggregateReleasePayload 對 strata×M 的全域粒子數核對。
    """

    event = _event_aggregate()
    receptor_key = ReceptorAggregateKey(
        study_site_id=_SITE_ID,
        receptor_id=_RECEPTOR_ID,
    )
    return replace(
        event,
        outcome_count_by_site={_SITE_ID: _outcome_counts(member_count)},
        valid_member_denominator_by_site={_SITE_ID: member_count},
        total_member_count_by_site={_SITE_ID: member_count},
        valid_member_denominator_by_receptor={receptor_key: member_count},
        input_particle_count=member_count,
    )


def _two_site_payload_kwargs() -> dict[str, object]:
    """建立兩站、兩列 scenario 的合法 synthetic payload 供 site-level tamper 使用。

    這個輔助 fixture 沿用單站 fixture 的公尺格網、事件守恆與 pathway 陣列，只新增
    第二站的 spec／scenario／shard／事件 key。兩站各一列 scenario、各有 M=2 個成員，
    因而全域粒子數為 4；這讓 site 分母被竄改時仍可維持全域總數，避免測試只命中
    global count 分支。所有新增 record 都透過公開 constructor 或 dataclasses.replace
    建立，不直接改寫 frozen dataclass 的底層欄位。
    """

    base_spec = _aggregate_spec()
    spec = replace(
        base_spec,
        site_grids={
            **base_spec.site_grids,
            _SECOND_SITE_ID: SiteGridSpec(
                x_min_m=0.0,
                x_max_m=2.0,
                y_min_m=0.0,
                y_max_m=1.0,
            ),
        },
        site_metric_crs={
            **base_spec.site_metric_crs,
            _SECOND_SITE_ID: SiteMetricCRSSpec(
                projection_method="azimuthal_equidistant_wgs84",
                center_lon_deg=122.0,
                center_lat_deg=25.0,
                linear_unit="m",
                axis_order="x_east_y_north",
            ),
        },
        site_boundary_segment_ids={
            **base_spec.site_boundary_segment_ids,
            _SECOND_SITE_ID: SiteBoundarySegments(
                local_segment_ids=(_SEGMENT_ID, _ZERO_SEGMENT_ID),
                outer_segment_ids=(_SEGMENT_ID,),
            ),
        },
    )

    first_stratum = _scenario_stratum()
    second_stratum = replace(
        first_stratum,
        scenario_id="scenario-b",
        study_site_id=_SECOND_SITE_ID,
        receptor_id=_SECOND_RECEPTOR_ID,
        arrival_time_id="arrival-b",
        arrival_time_utc_ns=1_700_000_000_000_000_001,
    )

    first_shard = _shard_binding()
    second_shard = replace(
        first_shard,
        shard_id="shard-1",
        scenario_start_index=1,
        scenario_stop_index=2,
        output_relative_path="trajectories/shard-1.parquet",
    )

    base_event = _event_aggregate()
    local_a = BoundaryAggregateKey(
        study_site_id=_SITE_ID,
        boundary_kind="local",
        boundary_segment_id=_SEGMENT_ID,
    )
    outer_a = BoundaryAggregateKey(
        study_site_id=_SITE_ID,
        boundary_kind="outer",
        boundary_segment_id=_SEGMENT_ID,
    )
    zero_a = BoundaryAggregateKey(
        study_site_id=_SITE_ID,
        boundary_kind="local",
        boundary_segment_id=_ZERO_SEGMENT_ID,
    )
    local_b = replace(local_a, study_site_id=_SECOND_SITE_ID)
    outer_b = replace(outer_a, study_site_id=_SECOND_SITE_ID)
    zero_b = replace(zero_a, study_site_id=_SECOND_SITE_ID)
    source_a = SourceReceptorAggregateKey(
        study_site_id=_SITE_ID,
        receptor_id=_RECEPTOR_ID,
        boundary_kind="local",
        boundary_segment_id=_SEGMENT_ID,
    )
    outer_source_a = SourceReceptorAggregateKey(
        study_site_id=_SITE_ID,
        receptor_id=_RECEPTOR_ID,
        boundary_kind="outer",
        boundary_segment_id=_SEGMENT_ID,
    )
    zero_source_a = SourceReceptorAggregateKey(
        study_site_id=_SITE_ID,
        receptor_id=_RECEPTOR_ID,
        boundary_kind="local",
        boundary_segment_id=_ZERO_SEGMENT_ID,
    )
    source_b = replace(
        source_a,
        study_site_id=_SECOND_SITE_ID,
        receptor_id=_SECOND_RECEPTOR_ID,
    )
    outer_source_b = replace(
        outer_source_a,
        study_site_id=_SECOND_SITE_ID,
        receptor_id=_SECOND_RECEPTOR_ID,
    )
    zero_source_b = replace(
        zero_source_a,
        study_site_id=_SECOND_SITE_ID,
        receptor_id=_SECOND_RECEPTOR_ID,
    )
    receptor_a = ReceptorAggregateKey(
        study_site_id=_SITE_ID,
        receptor_id=_RECEPTOR_ID,
    )
    receptor_b = replace(receptor_a, study_site_id=_SECOND_SITE_ID, receptor_id=_SECOND_RECEPTOR_ID)
    site_grid = base_event.site_grid_counts[_SITE_ID]

    event = replace(
        base_event,
        site_grid_counts={_SITE_ID: site_grid, _SECOND_SITE_ID: site_grid},
        boundary_bin_edges_m={
            local_a: base_event.boundary_bin_edges_m[local_a],
            outer_a: base_event.boundary_bin_edges_m[outer_a],
            zero_a: base_event.boundary_bin_edges_m[zero_a],
            local_b: base_event.boundary_bin_edges_m[local_a],
            outer_b: base_event.boundary_bin_edges_m[outer_a],
            zero_b: base_event.boundary_bin_edges_m[zero_a],
        },
        boundary_arclength_raw_count={
            local_a: base_event.boundary_arclength_raw_count[local_a],
            outer_a: base_event.boundary_arclength_raw_count[outer_a],
            zero_a: base_event.boundary_arclength_raw_count[zero_a],
            local_b: base_event.boundary_arclength_raw_count[local_a],
            outer_b: base_event.boundary_arclength_raw_count[outer_a],
            zero_b: base_event.boundary_arclength_raw_count[zero_a],
        },
        boundary_travel_age_histogram={
            local_a: base_event.boundary_travel_age_histogram[local_a],
            outer_a: base_event.boundary_travel_age_histogram[outer_a],
            zero_a: base_event.boundary_travel_age_histogram[zero_a],
            local_b: base_event.boundary_travel_age_histogram[local_a],
            outer_b: base_event.boundary_travel_age_histogram[outer_a],
            zero_b: base_event.boundary_travel_age_histogram[zero_a],
        },
        source_receptor_raw_count={
            source_a: base_event.source_receptor_raw_count[source_a],
            outer_source_a: base_event.source_receptor_raw_count[outer_source_a],
            zero_source_a: base_event.source_receptor_raw_count[zero_source_a],
            source_b: base_event.source_receptor_raw_count[source_a],
            outer_source_b: base_event.source_receptor_raw_count[outer_source_a],
            zero_source_b: base_event.source_receptor_raw_count[zero_source_a],
        },
        source_receptor_travel_age_histogram={
            source_a: base_event.source_receptor_travel_age_histogram[source_a],
            outer_source_a: base_event.source_receptor_travel_age_histogram[outer_source_a],
            zero_source_a: base_event.source_receptor_travel_age_histogram[zero_source_a],
            source_b: base_event.source_receptor_travel_age_histogram[source_a],
            outer_source_b: base_event.source_receptor_travel_age_histogram[outer_source_a],
            zero_source_b: base_event.source_receptor_travel_age_histogram[zero_source_a],
        },
        cross_site_unique_member_count={
            CrossSiteAggregateKey(
                source_study_site_id=_SITE_ID,
                target_study_site_id=_SECOND_SITE_ID,
            ): 0,
            CrossSiteAggregateKey(
                source_study_site_id=_SECOND_SITE_ID,
                target_study_site_id=_SITE_ID,
            ): 0,
        },
        outcome_count_by_site={
            _SITE_ID: _outcome_counts(_MEMBERS_PER_SCENARIO),
            _SECOND_SITE_ID: _outcome_counts(_MEMBERS_PER_SCENARIO),
        },
        valid_member_denominator_by_site={
            _SITE_ID: _MEMBERS_PER_SCENARIO,
            _SECOND_SITE_ID: _MEMBERS_PER_SCENARIO,
        },
        total_member_count_by_site={
            _SITE_ID: _MEMBERS_PER_SCENARIO,
            _SECOND_SITE_ID: _MEMBERS_PER_SCENARIO,
        },
        valid_member_denominator_by_receptor={
            receptor_a: _MEMBERS_PER_SCENARIO,
            receptor_b: _MEMBERS_PER_SCENARIO,
        },
        input_particle_count=2 * _MEMBERS_PER_SCENARIO,
    )

    pathway = _pathway_aggregate()
    kwargs = _valid_payload_kwargs(
        pathway_by_site={_SITE_ID: pathway, _SECOND_SITE_ID: pathway},
    )
    kwargs.update(
        aggregate_spec=spec,
        shard_bindings=[first_shard, second_shard],
        scenario_strata=[first_stratum, second_stratum],
        event_aggregate=event,
    )
    return kwargs
