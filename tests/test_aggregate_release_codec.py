"""Aggregate release 公開編碼器的固定拓撲、數值與防竄改契約測試。

本測試檔重用 payload 測試中的單站 fixture，驗證公開 encoder 產出的九張表與十八個
NumPy 陣列不只形狀正確，還必須符合固定欄位、列值、dtype、offset 與 C-order 資料
契約。測試資料是公尺制單站 synthetic payload；其中的相對來源權重與事件計數僅代表
條件式來源足跡載體，不代表絕對來源機率或因果歸因。
"""

from __future__ import annotations

import numpy as np
import pytest
import test_aggregate_release_payload as payload_fixture

from lagrangian_backtracking.aggregate_release_codec import (
    AggregateReleaseMetadata,
    encode_aggregate_release_payload,
    metadata_from_payload,
)
from lagrangian_backtracking.aggregate_release_layout import (
    AGGREGATE_RELEASE_ARRAY_FILES,
    AGGREGATE_RELEASE_TABLE_FILES,
)
from lagrangian_backtracking.aggregate_release_payload import AggregateReleasePayload

_EXPECTED_TABLE_CONTRACTS: dict[
    str,
    tuple[tuple[str, ...], tuple[dict[str, object], ...]],
] = {
    "shard_bindings.parquet": (
        (
            "shard_index",
            "shard_id",
            "scenario_start_index",
            "scenario_stop_index",
            "output_relative_path",
            "trajectory_manifest_sha256",
            "particle_count",
            "observation_count",
            "event_count",
        ),
        (
            {
                "shard_index": 0,
                "shard_id": "shard-0",
                "scenario_start_index": 0,
                "scenario_stop_index": 1,
                "output_relative_path": "trajectories/shard-0.parquet",
                "trajectory_manifest_sha256": "c" * 64,
                "particle_count": 2,
                "observation_count": 2,
                "event_count": 2,
            },
        ),
    ),
    "scenario_strata.parquet": (
        (
            "scenario_index",
            "scenario_id",
            "study_site_id",
            "analysis_region_id",
            "material_id",
            "material_category_zh",
            "material_family_zh",
            "representative_shape_zh",
            "behavior_class",
            "settling_velocity_mps",
            "applicability_condition_zh",
            "calibration_status",
            "evidence_grade",
            "receptor_id",
            "receptor_lon_deg",
            "receptor_lat_deg",
            "receptor_template_z_m_positive_up",
            "vertical_id",
            "arrival_time_id",
            "arrival_time_utc_ns",
            "arrival_year",
            "season",
            "tide_class",
            "phase_or_event",
            "design_version",
            "initial_z_m_positive_up",
            "initial_eta_m_positive_up",
            "initial_bed_z_m_positive_up",
            "initial_water_column_height_m",
            "initial_height_above_bed_m",
            "initial_zcor_lower_m_positive_up",
            "initial_zcor_upper_m_positive_up",
            "initial_vertical_bracket_alpha",
            "initial_source_face_local_index",
            "initial_source_face_global_index",
            "initial_wetdry_elem_value",
            "initial_wetdry_semantics_id",
            "initial_ocm_month_yyyymm",
            "initial_ocm_source_time_index",
            "initial_ocm_time_origin",
        ),
        (
            {
                "scenario_index": 0,
                "scenario_id": "scenario-a",
                "study_site_id": "site-a",
                "analysis_region_id": "region-a",
                "material_id": "material-a",
                "material_category_zh": "合成材料",
                "material_family_zh": "測試材料",
                "representative_shape_zh": "球狀",
                "behavior_class": "neutral",
                "settling_velocity_mps": -0.001,
                "applicability_condition_zh": "僅供合成測試",
                "calibration_status": "未校準",
                "evidence_grade": "synthetic",
                "receptor_id": "receptor-a",
                "receptor_lon_deg": 121.0,
                "receptor_lat_deg": 25.0,
                "receptor_template_z_m_positive_up": -1.0,
                "vertical_id": "surface",
                "arrival_time_id": "arrival-a",
                "arrival_time_utc_ns": 1_700_000_000_000_000_000,
                "arrival_year": 2023,
                "season": "winter",
                "tide_class": "flood",
                "phase_or_event": "fixture-event",
                "design_version": "aggregate-payload-test-v1",
                "initial_z_m_positive_up": None,
                "initial_eta_m_positive_up": None,
                "initial_bed_z_m_positive_up": None,
                "initial_water_column_height_m": None,
                "initial_height_above_bed_m": None,
                "initial_zcor_lower_m_positive_up": None,
                "initial_zcor_upper_m_positive_up": None,
                "initial_vertical_bracket_alpha": None,
                "initial_source_face_local_index": None,
                "initial_source_face_global_index": None,
                "initial_wetdry_elem_value": None,
                "initial_wetdry_semantics_id": None,
                "initial_ocm_month_yyyymm": None,
                "initial_ocm_source_time_index": None,
                "initial_ocm_time_origin": None,
            },
        ),
    ),
    "site_index.parquet": (
        (
            "site_index",
            "study_site_id",
            "analysis_region_id",
            "x_min_m",
            "x_max_m",
            "y_min_m",
            "y_max_m",
            "x_cell_count",
            "y_cell_count",
            "cell_start_offset",
            "cell_stop_offset",
            "projection_method",
            "center_lon_deg",
            "center_lat_deg",
            "linear_unit",
            "axis_order",
            "scenario_count",
            "total_member_count",
            "valid_member_denominator",
            "pathway_input_particle_count",
            "pathway_input_interval_seconds",
            "pathway_allocated_interval_seconds",
        ),
        (
            {
                "site_index": 0,
                "study_site_id": "site-a",
                "analysis_region_id": "region-a",
                "x_min_m": 0.0,
                "x_max_m": 2.0,
                "y_min_m": 0.0,
                "y_max_m": 1.0,
                "x_cell_count": 2,
                "y_cell_count": 1,
                "cell_start_offset": 0,
                "cell_stop_offset": 2,
                "projection_method": "azimuthal_equidistant_wgs84",
                "center_lon_deg": 121.0,
                "center_lat_deg": 25.0,
                "linear_unit": "m",
                "axis_order": "x_east_y_north",
                "scenario_count": 1,
                "total_member_count": 2,
                "valid_member_denominator": 2,
                "pathway_input_particle_count": 2,
                "pathway_input_interval_seconds": 2.0,
                "pathway_allocated_interval_seconds": 2.0,
            },
        ),
    ),
    "boundary_index.parquet": (
        (
            "boundary_index",
            "study_site_id",
            "boundary_kind",
            "boundary_segment_id",
            "segment_length_m",
            "edge_start_offset",
            "edge_stop_offset",
            "bin_start_offset",
            "bin_stop_offset",
        ),
        (
            {
                "boundary_index": 0,
                "study_site_id": "site-a",
                "boundary_kind": "local",
                "boundary_segment_id": "segment-a",
                "segment_length_m": 2.0,
                "edge_start_offset": 0,
                "edge_stop_offset": 3,
                "bin_start_offset": 0,
                "bin_stop_offset": 2,
            },
            {
                "boundary_index": 1,
                "study_site_id": "site-a",
                "boundary_kind": "local",
                "boundary_segment_id": "segment-zero",
                "segment_length_m": 2.0,
                "edge_start_offset": 3,
                "edge_stop_offset": 6,
                "bin_start_offset": 2,
                "bin_stop_offset": 4,
            },
            {
                "boundary_index": 2,
                "study_site_id": "site-a",
                "boundary_kind": "outer",
                "boundary_segment_id": "segment-a",
                "segment_length_m": 2.0,
                "edge_start_offset": 6,
                "edge_stop_offset": 9,
                "bin_start_offset": 4,
                "bin_stop_offset": 6,
            },
        ),
    ),
    "source_receptor_index.parquet": (
        (
            "source_receptor_index",
            "study_site_id",
            "receptor_id",
            "boundary_kind",
            "boundary_segment_id",
        ),
        (
            {
                "source_receptor_index": 0,
                "study_site_id": "site-a",
                "receptor_id": "receptor-a",
                "boundary_kind": "local",
                "boundary_segment_id": "segment-a",
            },
            {
                "source_receptor_index": 1,
                "study_site_id": "site-a",
                "receptor_id": "receptor-a",
                "boundary_kind": "local",
                "boundary_segment_id": "segment-zero",
            },
            {
                "source_receptor_index": 2,
                "study_site_id": "site-a",
                "receptor_id": "receptor-a",
                "boundary_kind": "outer",
                "boundary_segment_id": "segment-a",
            },
        ),
    ),
    "cross_site_counts.parquet": (
        (
            "source_study_site_id",
            "target_study_site_id",
            "unique_member_count",
        ),
        (),
    ),
    "outcome_counts.parquet": (
        ("study_site_id", "outcome", "count"),
        (
            {"study_site_id": "site-a", "outcome": "coast_contact", "count": 0},
            {"study_site_id": "site-a", "outcome": "data_gap", "count": 0},
            {"study_site_id": "site-a", "outcome": "deposited", "count": 0},
            {"study_site_id": "site-a", "outcome": "flow_domain_open_exit", "count": 0},
            {"study_site_id": "site-a", "outcome": "forcing_start", "count": 0},
            {"study_site_id": "site-a", "outcome": "max_age", "count": 2},
            {"study_site_id": "site-a", "outcome": "numerical_failure", "count": 0},
            {"study_site_id": "site-a", "outcome": "surface_regime_exit", "count": 0},
        ),
    ),
    "site_denominators.parquet": (
        ("study_site_id", "valid_member_denominator", "total_member_count"),
        (
            {
                "study_site_id": "site-a",
                "valid_member_denominator": 2,
                "total_member_count": 2,
            },
        ),
    ),
    "receptor_denominators.parquet": (
        ("study_site_id", "receptor_id", "valid_member_denominator"),
        (
            {
                "study_site_id": "site-a",
                "receptor_id": "receptor-a",
                "valid_member_denominator": 2,
            },
        ),
    ),
}


_EXPECTED_ARRAYS = {
    "age_bin_edges_seconds.npy": np.array([0.0, 1.0, 2.0], dtype=np.float64),
    "site_cell_offsets.npy": np.array([0, 2], dtype=np.int64),
    "local_first_exit_count.npy": np.array([1, 0], dtype=np.int64),
    "outer_first_exit_count.npy": np.array([1, 0], dtype=np.int64),
    "bed_first_contact_count.npy": np.array([0, 0], dtype=np.int64),
    "bed_repeated_contact_count.npy": np.array([0, 0], dtype=np.int64),
    "data_gap_failure_count.npy": np.array([0, 0], dtype=np.int64),
    "numerical_failure_count.npy": np.array([0, 0], dtype=np.int64),
    "pathway_unique_particle_count.npy": np.array([1, 1], dtype=np.int64),
    "pathway_residence_time_seconds.npy": np.array([1.0, 1.0], dtype=np.float64),
    "pathway_first_passage_age_histogram.npy": np.array([1, 0, 0, 1], dtype=np.int64),
    "boundary_edge_offsets.npy": np.array([0, 3, 6, 9], dtype=np.int64),
    "boundary_bin_edges_m.npy": np.array(
        [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        dtype=np.float64,
    ),
    "boundary_bin_offsets.npy": np.array([0, 2, 4, 6], dtype=np.int64),
    "boundary_arclength_raw_count.npy": np.array([1, 0, 0, 0, 1, 0], dtype=np.int64),
    "boundary_travel_age_histogram.npy": np.array(
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        dtype=np.int64,
    ),
    "source_receptor_raw_count.npy": np.array([1, 0, 1], dtype=np.int64),
    "source_receptor_travel_age_histogram.npy": np.array(
        [1, 0, 0, 0, 1, 0],
        dtype=np.int64,
    ),
}


def _single_site_payload() -> AggregateReleasePayload:
    """建立既有測試 fixture 的單站 payload，避免測試複製 production 建構流程。

    fixture 的格網軸是公尺制的 ``(y_cell, x_cell)=(1, 2)``，年齡軸是秒制的
    ``[0, 1, 2]``；這些輸入會在 encoder 中展平，測試再以固定 canonical 值核對。
    """

    return AggregateReleasePayload(**payload_fixture._valid_payload_kwargs())


def _expected_metadata_document() -> dict[str, object]:
    """建立單站 metadata 的固定 19 欄快照，供結構與錯誤輸入測試共用。

    這些值來自測試用 synthetic payload：八個 SHA-256 欄位是輸入與 AggregateSpec
    provenance 的識別摘要，僅用來追蹤資料來源與設定版本；七個 count 欄位則描述
    成員數與固定索引表的拓撲列數。兩者都是可重現性與資料契約 metadata，不是事件
    發生量、有效分母、條件式來源足跡或相對來源權重等科學結果，因此測試必須核對
    原值與欄位順序，不能把它們當作可由其他欄位推導的統計量。
    """

    return {
        "schema_version": "1.0.0",
        "run_id": "aggregate-payload-fixture",
        "run_kind": "synthetic",
        "experiment_case_id": "synthetic-case",
        "members_per_scenario": 2,
        "config_hash": "d" * 64,
        "checkpoint_input_binding_hash": "e" * 64,
        "source_run_plan_sha256": "f" * 64,
        "source_run_progress_sha256": "0" * 64,
        "source_normalized_config_sha256": "1" * 64,
        "source_input_inventory_sha256": "2" * 64,
        "aggregate_spec_source_sha256": "a" * 64,
        "aggregate_spec_canonical_sha256": "b" * 64,
        "input_particle_count": 2,
        "shard_row_count": 1,
        "scenario_row_count": 1,
        "site_row_count": 1,
        "boundary_row_count": 3,
        "source_receptor_row_count": 3,
    }


def _assert_exact_table(
    encoded_tables: object,
    *,
    file_name: str,
    expected_columns: tuple[str, ...],
    expected_rows: tuple[dict[str, object], ...],
) -> None:
    """逐列核對表格欄位順序與完整值，避免只用列數或 key set 掩蓋資料錯置。

    ``EncodedAggregateProducts`` 的表列是唯讀 mapping；轉成普通 dict 只用於比對，不會
    改寫產品。空的 ``cross_site_counts.parquet`` 仍須核對為精確空 tuple；其三欄 schema
    由 encoder 的 row contract 固定，但單站沒有 row 可供容器保存欄位 metadata。
    """

    assert isinstance(encoded_tables, dict) or hasattr(encoded_tables, "__getitem__")
    rows = encoded_tables[file_name]  # type: ignore[index]
    assert len(rows) == len(expected_rows)
    for row, expected in zip(rows, expected_rows, strict=True):
        assert tuple(row) == expected_columns
        assert dict(row) == expected
    if not expected_rows:
        assert rows == ()


def test_single_site_encoder_emits_exact_tables_and_arrays() -> None:
    """驗證單站 fixture 的九表、十八陣列、欄位、列值、offset 與記憶體布局。

    表格列值包含零事件拓撲與非零事件，陣列則包含所有展平後的 offset 與 histogram
    位置；逐元素 exact comparison 可抓出排序、軸轉置、錯誤 bin 位置或 silent cast，
    這些錯誤不會被單看 shape 或集合的斷言發現。
    """

    encoded = encode_aggregate_release_payload(_single_site_payload())

    assert frozenset(encoded.tables) == AGGREGATE_RELEASE_TABLE_FILES
    assert len(encoded.tables) == 9
    assert frozenset(encoded.arrays) == AGGREGATE_RELEASE_ARRAY_FILES
    assert len(encoded.arrays) == 18

    for file_name, (expected_columns, expected_rows) in _EXPECTED_TABLE_CONTRACTS.items():
        _assert_exact_table(
            encoded.tables,
            file_name=file_name,
            expected_columns=expected_columns,
            expected_rows=expected_rows,
        )

    for file_name, expected in _EXPECTED_ARRAYS.items():
        actual = encoded.arrays[file_name]
        assert actual.dtype == expected.dtype
        assert actual.flags.c_contiguous
        assert not actual.flags.writeable
        np.testing.assert_array_equal(actual, expected)


def test_encoding_same_payload_twice_is_exact_and_does_not_share_buffers() -> None:
    """同一 payload 編碼兩次應得到相同內容，且輸出不可互相共享可寫記憶體。

    encoder 會重新驗證並重建 nested records，之後 layout container 也會再做 defensive
    snapshot。這裡逐表逐列與逐陣列比對，並檢查 object identity、memory sharing 及唯讀
    旗標，防止第一次結果被第二次呼叫或 caller 的後續寫入影響。
    """

    payload = _single_site_payload()
    first = encode_aggregate_release_payload(payload)
    second = encode_aggregate_release_payload(payload)

    assert frozenset(first.tables) == frozenset(second.tables)
    assert frozenset(first.arrays) == frozenset(second.arrays)
    for file_name in sorted(first.tables):
        first_rows = first.tables[file_name]
        second_rows = second.tables[file_name]
        assert len(first_rows) == len(second_rows)
        # 單站 cross-site 表依法是空 tuple；空 tuple 可能由 Python 共用 singleton，
        # 因此只核對其精確空拓撲，不把不可變 singleton 誤當成可寫 alias。
        if first_rows:
            assert first_rows is not second_rows
        for first_row, second_row in zip(first_rows, second_rows, strict=True):
            assert first_row is not second_row
            assert tuple(first_row) == tuple(second_row)
            assert dict(first_row) == dict(second_row)

    for file_name in sorted(first.arrays):
        first_array = first.arrays[file_name]
        second_array = second.arrays[file_name]
        assert first_array is not second_array
        assert first_array.dtype == second_array.dtype
        assert first_array.flags.c_contiguous
        assert second_array.flags.c_contiguous
        assert not first_array.flags.writeable
        assert not second_array.flags.writeable
        assert not np.shares_memory(first_array, second_array)
        np.testing.assert_array_equal(first_array, second_array)


def test_encoder_fails_closed_after_nested_frozen_shard_binding_tamper() -> None:
    """深層竄改 frozen ``AggregateShardBinding`` 後，公開 encoder 必須重新驗證並拒絕。

    兩個 range 欄位被低階改成另一個仍可單獨建構的半開區間，但它不再從 scenario 0
    連續覆蓋唯一 scenario。測試刻意使用 ``object.__setattr__`` 模擬不可由一般 API
    取得的破壞入口，確認公開 encoder 不會直接信任已封存的 nested record。
    """

    payload = _single_site_payload()
    shard = payload.shard_bindings[0]
    object.__setattr__(shard, "scenario_start_index", 1)
    object.__setattr__(shard, "scenario_stop_index", 2)

    with pytest.raises(ValueError, match="scenario range"):
        encode_aggregate_release_payload(payload)


def test_encoder_fails_closed_after_nested_frozen_site_grid_tamper() -> None:
    """深層竄改 frozen ``SiteEventGridCounts`` 後，公開 encoder 必須拒絕負事件計數。

    事件格網的每個元素代表指定站點與 ``(y_cell, x_cell)`` 儲存格的非負計數；把
    local first-exit 的第一格改成負值會破壞缺值／失敗狀態不可用零值或負值替代的
    資料契約。低階寫入後仍須由 encoder 的深層重建流程 fail closed。
    """

    payload = _single_site_payload()
    grid_counts = payload.event_aggregate.site_grid_counts["site-a"]
    object.__setattr__(
        grid_counts,
        "local_first_exit_count",
        np.array([[-1, 0]], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="負值|負"):
        encode_aggregate_release_payload(payload)


def test_encoder_rejects_signed_int64_overflow_in_scenario_arrival_time() -> None:
    """scenario arrival UTC 奈秒超出 signed int64 時，公開 encoder 必須拒絕而非繞回。

    ``ScenarioStratum`` 的原生 Python int 可暫時容納任意精度；但 release 表格欄位固定
    為 signed int64，且 arrival time 允許 epoch 前的負值。這裡從 payload 建構完成後
    直接竄改成 ``int64.max + 1``，專門驗證 encoder 的序列化邊界檢查，而非只測底層
    record constructor 的型別檢查。
    """

    payload = _single_site_payload()
    scenario = payload.scenario_strata[0]
    overflow_value = int(np.iinfo(np.int64).max) + 1
    object.__setattr__(scenario, "arrival_time_utc_ns", overflow_value)

    with pytest.raises(ValueError, match="arrival_time_utc_ns|int64"):
        encode_aggregate_release_payload(payload)


_METADATA_COUNT_FIELDS = (
    "members_per_scenario",
    "input_particle_count",
    "shard_row_count",
    "scenario_row_count",
    "site_row_count",
    "boundary_row_count",
    "source_receptor_row_count",
)

_METADATA_DIGEST_FIELDS = (
    "config_hash",
    "checkpoint_input_binding_hash",
    "source_run_plan_sha256",
    "source_run_progress_sha256",
    "source_normalized_config_sha256",
    "source_input_inventory_sha256",
    "aggregate_spec_source_sha256",
    "aggregate_spec_canonical_sha256",
)

_METADATA_INVALID_COUNT_VALUES = (
    # count 是資料拓撲與 provenance 的原生整數邊界；這四種值都不可藉由隱式轉型通過。
    pytest.param(0, id="zero"),
    pytest.param(True, id="bool"),
    pytest.param(np.int64(1), id="numpy-int64"),
    pytest.param(
        int(np.iinfo(np.int64).max) + 1,
        id="signed-int64-overflow",
    ),
)


def test_metadata_from_payload_has_exact_fields_and_round_trips() -> None:
    """驗證 metadata 的 19 欄固定順序、exact 值、round-trip 與 dict 隔離性。

    provenance 欄位是 run 設定、輸入 inventory 與 AggregateSpec 的來源摘要；count
    欄位是成員數及固定索引表的列數。兩者都只是可追溯性與資料拓撲 metadata，不是
    事件發生量、有效分母、條件式來源足跡或相對來源權重等科學結果，因此這裡同時
    核對欄位順序與完整值，不能只檢查 key set 或列數。``to_dict`` 的每次結果也必須
    是獨立的普通 dict，避免呼叫端改寫快照時污染 metadata 本身。
    """

    metadata = metadata_from_payload(_single_site_payload())
    expected = _expected_metadata_document()
    document = metadata.to_dict()

    assert type(document) is dict
    assert len(document) == 19
    assert tuple(document) == tuple(expected)
    assert document == expected

    round_tripped = AggregateReleaseMetadata.from_dict(document)
    assert round_tripped == metadata
    assert round_tripped.to_dict() == expected

    first_snapshot = metadata.to_dict()
    second_snapshot = metadata.to_dict()
    assert type(first_snapshot) is dict
    assert type(second_snapshot) is dict
    assert first_snapshot is not second_snapshot
    first_snapshot["run_id"] = "caller-mutated"
    assert second_snapshot == expected
    assert metadata.to_dict() == expected


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param("missing", id="missing-one-key"),
        pytest.param("extra", id="extra-one-key"),
    ],
)
def test_metadata_from_dict_requires_exact_key_set(mutation: str) -> None:
    """驗證 from_dict 缺一欄或多一欄都必須拒絕，不能靜默忽略 metadata 變更。

    以同一份 19 欄合法基準字典逐次只改動一個 key：缺少欄位代表 provenance、拓撲
    或計數契約不完整，額外欄位則代表尚未登錄的 schema 擴充；兩種情況都不能被當成
    已驗收的 release metadata。這些欄位仍是資料來源與固定列結構的描述，不是科學結果。
    """

    document = _expected_metadata_document()
    if mutation == "missing":
        document.pop("source_receptor_row_count")
        expected_message = "缺少 key"
    else:
        document["unregistered_key"] = "not-allowed"
        expected_message = "未知 key"

    with pytest.raises(ValueError, match=expected_message):
        AggregateReleaseMetadata.from_dict(document)


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        pytest.param("schema_version", "9.9.9", id="unsupported-schema"),
        pytest.param("run_id", "../unsafe-run", id="unsafe-run-slug"),
        pytest.param(
            "experiment_case_id",
            "unsafe/experiment",
            id="unsafe-experiment-slug",
        ),
        pytest.param("run_kind", "unknown", id="unknown-run-kind"),
    ],
)
def test_metadata_from_dict_rejects_invalid_schema_identity_and_run_kind(
    field_name: str,
    bad_value: object,
) -> None:
    """驗證 schema、識別 slug 與 run kind 的錯誤值均採 fail-closed 政策。

    每個案例都從完整合法文件複製，且只替換一個識別或模式欄位。run／experiment
    slug 會進入路徑與 join key，schema 與 run kind 則決定欄位語意；拒絕它們可避免
    不同 provenance 被誤合併。這些 metadata 驗證不會把任何欄位解讀成科學結果。
    """

    document = _expected_metadata_document()
    document[field_name] = bad_value

    with pytest.raises(ValueError, match=field_name):
        AggregateReleaseMetadata.from_dict(document)


@pytest.mark.parametrize("digest_field", _METADATA_DIGEST_FIELDS)
@pytest.mark.parametrize(
    "bad_digest",
    [
        pytest.param("not-a-sha256", id="malformed-sha"),
        pytest.param("A" * 64, id="uppercase-sha"),
    ],
)
def test_metadata_from_dict_rejects_malformed_or_uppercase_digest(
    digest_field: str,
    bad_digest: str,
) -> None:
    """驗證每個 provenance SHA-256 欄位都拒絕錯誤格式與大寫摘要。

    八個 digest 是輸入設定、run provenance 與 AggregateSpec 來源關係的鎖點；只要
    一欄格式不符合小寫 64 碼 SHA-256，就不能把文件當成同一份可追溯 metadata。測試
    每次只改一個 digest，並確認錯誤指出該欄位；摘要本身不代表事件或來源科學推論。
    """

    document = _expected_metadata_document()
    document[digest_field] = bad_digest

    with pytest.raises(ValueError, match=digest_field):
        AggregateReleaseMetadata.from_dict(document)


@pytest.mark.parametrize("count_field", _METADATA_COUNT_FIELDS)
@pytest.mark.parametrize("bad_value", _METADATA_INVALID_COUNT_VALUES)
def test_metadata_from_dict_rejects_invalid_count_values(
    count_field: str,
    bad_value: object,
) -> None:
    """驗證每個 count 欄位拒絕零、bool、NumPy int64 與 signed int64 溢位。

    ``members_per_scenario`` 與 ``input_particle_count`` 描述已驗證的成員拓撲，其餘
    row count 描述固定索引表的實際列數；它們都是工程 metadata，不是事件量、有效
    分母或條件式來源足跡。每個案例都重新建立完整基準 dict，逐次只改一個欄位，
    確認型別與 signed int64 邊界都以封閉方式驗證，而不靠 ``int()`` 隱式轉型。
    """

    document = _expected_metadata_document()
    document[count_field] = bad_value

    with pytest.raises(ValueError, match=count_field):
        AggregateReleaseMetadata.from_dict(document)


def test_metadata_from_payload_revalidates_nested_shard_range_tamper() -> None:
    """深層竄改 nested shard 的 scenario 半開區間後，metadata builder 必須拒絕。

    ``scenario_start_index`` 與 ``scenario_stop_index`` 描述 shard 覆蓋的基礎 scenario
    範圍，不是可由 row count 猜測的科學量。低階寫入一個單獨看似合法、但不再從 0
    連續覆蓋唯一 scenario 的範圍後，metadata_from_payload 仍必須重建 shard 與最外層
    payload；拒絕可證明 metadata 不會繞過 nested provenance 與拓撲驗證。
    """

    payload = _single_site_payload()
    shard = payload.shard_bindings[0]
    object.__setattr__(shard, "scenario_start_index", 1)
    object.__setattr__(shard, "scenario_stop_index", 2)

    with pytest.raises(ValueError, match="scenario range"):
        metadata_from_payload(payload)


def test_metadata_from_payload_revalidates_nested_spec_hash_tamper() -> None:
    """深層竄改 nested AggregateSpec hash 後，metadata builder 必須拒絕不合規摘要。

    AggregateSpec 的 canonical hash 是規格來源 provenance；這裡以低階方式改成大寫
    摘要，模擬 frozen nested object 被竄改。metadata_from_payload 必須重新建構 spec，
    重新套用 SHA-256 格式契約，而非直接信任 caller 留在 payload 內的物件；摘要驗證
    只維護來源可追溯性，不代表任何科學結果已被驗證。
    """

    payload = _single_site_payload()
    object.__setattr__(payload.aggregate_spec, "canonical_sha256", "A" * 64)

    with pytest.raises(ValueError, match="canonical_sha256"):
        metadata_from_payload(payload)
