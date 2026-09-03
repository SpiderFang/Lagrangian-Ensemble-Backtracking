"""Aggregate release 落檔 schema 與 I/O 邊界的資料契約測試。

production 目前只提供九張 Parquet 表的明示 Apache Arrow schema，尚未提供
``_write_encoded_products``／``_read_encoded_products``。因此本檔先固定欄名順序、
Arrow 型別與 nullability，並特別驗證單站時即使 cross-site 表為零列，仍保有三欄
schema。待 I/O helper 落地後，再由同一檔案加入 27 個產品檔、contract、checksum、
NumPy little-endian 格式、round-trip 與惡意檔案拒絕測試；本輪不預先猜測其介面。
"""

from __future__ import annotations

import pyarrow as pa

from lagrangian_backtracking.aggregate_release import TABLE_SCHEMAS
from lagrangian_backtracking.aggregate_release_layout import AGGREGATE_RELEASE_TABLE_FILES

_EXPECTED_TABLE_SCHEMAS = {
    "shard_bindings.parquet": pa.schema(
        [
            pa.field("shard_index", pa.int64(), nullable=False),
            pa.field("shard_id", pa.string(), nullable=False),
            pa.field("scenario_start_index", pa.int64(), nullable=False),
            pa.field("scenario_stop_index", pa.int64(), nullable=False),
            pa.field("output_relative_path", pa.string(), nullable=False),
            pa.field("trajectory_manifest_sha256", pa.string(), nullable=False),
            pa.field("particle_count", pa.int64(), nullable=False),
            pa.field("observation_count", pa.int64(), nullable=False),
            pa.field("event_count", pa.int64(), nullable=False),
        ]
    ),
    "scenario_strata.parquet": pa.schema(
        [
            pa.field("scenario_index", pa.int64(), nullable=False),
            pa.field("scenario_id", pa.string(), nullable=False),
            pa.field("study_site_id", pa.string(), nullable=False),
            pa.field("analysis_region_id", pa.string(), nullable=False),
            pa.field("material_id", pa.string(), nullable=False),
            pa.field("material_category_zh", pa.string(), nullable=False),
            pa.field("material_family_zh", pa.string(), nullable=False),
            pa.field("representative_shape_zh", pa.string(), nullable=False),
            pa.field("behavior_class", pa.string(), nullable=False),
            pa.field("settling_velocity_mps", pa.float64(), nullable=False),
            pa.field("applicability_condition_zh", pa.string(), nullable=False),
            pa.field("calibration_status", pa.string(), nullable=False),
            pa.field("evidence_grade", pa.string(), nullable=False),
            pa.field("receptor_id", pa.string(), nullable=False),
            pa.field("receptor_lon_deg", pa.float64(), nullable=False),
            pa.field("receptor_lat_deg", pa.float64(), nullable=False),
            pa.field("receptor_template_z_m_positive_up", pa.float64(), nullable=False),
            pa.field("vertical_id", pa.string(), nullable=False),
            pa.field("arrival_time_id", pa.string(), nullable=False),
            pa.field("arrival_time_utc_ns", pa.int64(), nullable=False),
            pa.field("arrival_year", pa.int64(), nullable=False),
            pa.field("season", pa.string(), nullable=False),
            pa.field("tide_class", pa.string(), nullable=False),
            pa.field("phase_or_event", pa.string(), nullable=False),
            pa.field("design_version", pa.string(), nullable=False),
            pa.field("initial_z_m_positive_up", pa.float64(), nullable=True),
            pa.field("initial_eta_m_positive_up", pa.float64(), nullable=True),
            pa.field("initial_bed_z_m_positive_up", pa.float64(), nullable=True),
            pa.field("initial_water_column_height_m", pa.float64(), nullable=True),
            pa.field("initial_height_above_bed_m", pa.float64(), nullable=True),
            pa.field("initial_zcor_lower_m_positive_up", pa.float64(), nullable=True),
            pa.field("initial_zcor_upper_m_positive_up", pa.float64(), nullable=True),
            pa.field("initial_vertical_bracket_alpha", pa.float64(), nullable=True),
            pa.field("initial_source_face_local_index", pa.int64(), nullable=True),
            pa.field("initial_source_face_global_index", pa.int64(), nullable=True),
            pa.field("initial_wetdry_elem_value", pa.int64(), nullable=True),
            pa.field("initial_wetdry_semantics_id", pa.string(), nullable=True),
            pa.field("initial_ocm_month_yyyymm", pa.string(), nullable=True),
            pa.field("initial_ocm_source_time_index", pa.int64(), nullable=True),
            pa.field("initial_ocm_time_origin", pa.string(), nullable=True),
        ]
    ),
    "site_index.parquet": pa.schema(
        [
            pa.field("site_index", pa.int64(), nullable=False),
            pa.field("study_site_id", pa.string(), nullable=False),
            pa.field("analysis_region_id", pa.string(), nullable=False),
            pa.field("x_min_m", pa.float64(), nullable=False),
            pa.field("x_max_m", pa.float64(), nullable=False),
            pa.field("y_min_m", pa.float64(), nullable=False),
            pa.field("y_max_m", pa.float64(), nullable=False),
            pa.field("x_cell_count", pa.int64(), nullable=False),
            pa.field("y_cell_count", pa.int64(), nullable=False),
            pa.field("cell_start_offset", pa.int64(), nullable=False),
            pa.field("cell_stop_offset", pa.int64(), nullable=False),
            pa.field("projection_method", pa.string(), nullable=False),
            pa.field("center_lon_deg", pa.float64(), nullable=False),
            pa.field("center_lat_deg", pa.float64(), nullable=False),
            pa.field("linear_unit", pa.string(), nullable=False),
            pa.field("axis_order", pa.string(), nullable=False),
            pa.field("scenario_count", pa.int64(), nullable=False),
            pa.field("total_member_count", pa.int64(), nullable=False),
            pa.field("valid_member_denominator", pa.int64(), nullable=False),
            pa.field("pathway_input_particle_count", pa.int64(), nullable=False),
            pa.field("pathway_input_interval_seconds", pa.float64(), nullable=False),
            pa.field("pathway_allocated_interval_seconds", pa.float64(), nullable=False),
        ]
    ),
    "boundary_index.parquet": pa.schema(
        [
            pa.field("boundary_index", pa.int64(), nullable=False),
            pa.field("study_site_id", pa.string(), nullable=False),
            pa.field("boundary_kind", pa.string(), nullable=False),
            pa.field("boundary_segment_id", pa.string(), nullable=False),
            pa.field("segment_length_m", pa.float64(), nullable=False),
            pa.field("edge_start_offset", pa.int64(), nullable=False),
            pa.field("edge_stop_offset", pa.int64(), nullable=False),
            pa.field("bin_start_offset", pa.int64(), nullable=False),
            pa.field("bin_stop_offset", pa.int64(), nullable=False),
        ]
    ),
    "source_receptor_index.parquet": pa.schema(
        [
            pa.field("source_receptor_index", pa.int64(), nullable=False),
            pa.field("study_site_id", pa.string(), nullable=False),
            pa.field("receptor_id", pa.string(), nullable=False),
            pa.field("boundary_kind", pa.string(), nullable=False),
            pa.field("boundary_segment_id", pa.string(), nullable=False),
        ]
    ),
    "cross_site_counts.parquet": pa.schema(
        [
            pa.field("source_study_site_id", pa.string(), nullable=False),
            pa.field("target_study_site_id", pa.string(), nullable=False),
            pa.field("unique_member_count", pa.int64(), nullable=False),
        ]
    ),
    "outcome_counts.parquet": pa.schema(
        [
            pa.field("study_site_id", pa.string(), nullable=False),
            pa.field("outcome", pa.string(), nullable=False),
            pa.field("count", pa.int64(), nullable=False),
        ]
    ),
    "site_denominators.parquet": pa.schema(
        [
            pa.field("study_site_id", pa.string(), nullable=False),
            pa.field("valid_member_denominator", pa.int64(), nullable=False),
            pa.field("total_member_count", pa.int64(), nullable=False),
        ]
    ),
    "receptor_denominators.parquet": pa.schema(
        [
            pa.field("study_site_id", pa.string(), nullable=False),
            pa.field("receptor_id", pa.string(), nullable=False),
            pa.field("valid_member_denominator", pa.int64(), nullable=False),
        ]
    ),
}


_EXPECTED_NULLABLE_INITIAL_FIELDS = (
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
)


def test_table_schemas_have_exact_file_set_fields_types_and_nullability() -> None:
    """九張表必須逐表符合 exact 欄位順序、Arrow type 與 nullability。

    欄位順序會同時影響 Parquet physical schema、manifest field contract 與 decoder 的
    typed join；因此測試直接比較完整 ``pa.Schema`` 並啟用 metadata 比對，不只看欄名
    set。整數計數與 offset 固定為 int64，物理 scalar 為 float64，識別與描述為 string。
    """

    assert frozenset(TABLE_SCHEMAS) == AGGREGATE_RELEASE_TABLE_FILES
    assert frozenset(TABLE_SCHEMAS) == frozenset(_EXPECTED_TABLE_SCHEMAS)
    assert len(TABLE_SCHEMAS) == 9

    for file_name, expected_schema in _EXPECTED_TABLE_SCHEMAS.items():
        actual_schema = TABLE_SCHEMAS[file_name]
        assert tuple(actual_schema.names) == tuple(expected_schema.names)
        assert actual_schema.equals(expected_schema, check_metadata=True)
        assert tuple(
            (field.name, field.type, field.nullable) for field in actual_schema
        ) == tuple(
            (field.name, field.type, field.nullable) for field in expected_schema
        )


def test_only_fifteen_scenario_initial_fields_are_nullable() -> None:
    """ScenarioStratum 只能讓十五個 ``initial_*`` 欄位為 null，其他欄位皆必填。

    這十五欄代表可整組缺值的動態初始條件；null 不得擴散到 run／scenario join key、
    受體座標或材料欄位，也不可被解讀成零值、乾點、時間缺口或數值失敗。欄位順序亦
    固定，避免 writer 與 decoder 對整組有值／整組缺值政策產生不同欄位集合。
    """

    scenario_schema = TABLE_SCHEMAS["scenario_strata.parquet"]
    nullable_fields = tuple(field.name for field in scenario_schema if field.nullable)
    initial_fields = tuple(
        field.name for field in scenario_schema if field.name.startswith("initial_")
    )

    assert nullable_fields == _EXPECTED_NULLABLE_INITIAL_FIELDS
    assert initial_fields == _EXPECTED_NULLABLE_INITIAL_FIELDS
    assert len(nullable_fields) == 15
    assert all(
        field.nullable == (field.name in _EXPECTED_NULLABLE_INITIAL_FIELDS)
        for field in scenario_schema
    )


def test_empty_cross_site_table_retains_exact_three_column_schema() -> None:
    """單站 release 的 cross-site 表即使零列，仍必須保留三個 non-nullable 欄位。

    零列表示沒有跨站 ordered pair，不代表 schema 不存在。以明示 schema 建立空 Arrow
    table 可驗證後續 Parquet writer 能輸出具固定欄位的零列檔案，而非無欄 table；三欄
    依序保存來源站、目標站與不重複成員計數。
    """

    expected_schema = _EXPECTED_TABLE_SCHEMAS["cross_site_counts.parquet"]
    empty_table = pa.Table.from_pylist([], schema=TABLE_SCHEMAS["cross_site_counts.parquet"])

    assert empty_table.num_rows == 0
    assert empty_table.num_columns == 3
    assert tuple(empty_table.column_names) == (
        "source_study_site_id",
        "target_study_site_id",
        "unique_member_count",
    )
    assert empty_table.schema.equals(expected_schema, check_metadata=True)
    assert all(not field.nullable for field in empty_table.schema)
