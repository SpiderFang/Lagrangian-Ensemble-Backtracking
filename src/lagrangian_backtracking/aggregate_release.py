"""Aggregate release 的固定拓撲、實體 schema 與完整公開讀取驗證層。

本模組保存 schema 1 release 的固定名稱集合與 Arrow 表格 schema，提供九表／十八陣列
的內部實體 writer/reader，以及完整公開 release reader／validator。驗證流程鎖定 manifest
與三十二個普通檔案，再於讀取任何來源語意、Parquet 或 NPY 前驗證全部大小與 SHA-256；
manifest 不得提供任意路徑。模組不執行聚合、粒子平流、統計推估，也不讀取原始
OCM／NWW forcing。

五個 JSON 保存已驗證 source run／AggregateSpec 的來源快照；九張 Parquet 表保存
shard、scenario、站點／邊界拓撲、來源—受體關聯與計數／分母；十八個 NumPy 陣列保存
秒制 age 軸、公尺制格網／邊界、停留秒數及非負計數。來源文件語意驗證會把 plan、
progress、normalized config、input inventory 與完整 AggregateSpec 綁回 manifest metadata；
公開 reader 只在 topology、manifest、checksum、來源文件、產品 contract、codec decoder
與 run shard binding 全部通過後回傳 typed payload；公開 validator 則只回傳固定
JSON-safe 摘要，不暴露路徑、Arrow table 或 NumPy 陣列。所有本機 synthetic smoke test
只代表工程驗證，不是真實 OCM／NWW 科學成果；真實結果仍必須在具備正式 OCM schema 3
與 NWW3 schema 1 已驗收產品的 SERVER 上執行與驗證。

表格中的經緯度僅是 WGS84 資料交換與圖面定位，所有格線、邊界長度、深度與路徑時間
仍依資料契約分別使用公尺或秒。``ScenarioStratum`` 的十五個 ``initial_*`` 欄位
是唯一允許 Arrow null 的欄位群，且其上游 record 仍要求整組有值或整組缺值；null 不
代表零值，也不把乾點、缺時、域外或數值失敗折疊成單一數值。這些產品最多支持
「條件式來源足跡」或「相對來源權重」的保存，不能單獨宣稱絕對來源機率、因果歸因或
觀測驗證結果。
"""

from __future__ import annotations

import json
import math
import os
import shutil
import stat
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Final, NoReturn
from uuid import uuid4

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .aggregate_release_codec import (
    _BOUNDARY_INDEX_COLUMNS,
    _CROSS_SITE_COUNT_COLUMNS,
    _OUTCOME_COUNT_COLUMNS,
    _RECEPTOR_DENOMINATOR_COLUMNS,
    _SCENARIO_STRATUM_COLUMNS,
    _SHARD_BINDING_COLUMNS,
    _SITE_DENOMINATOR_COLUMNS,
    _SITE_INDEX_COLUMNS,
    _SOURCE_RECEPTOR_INDEX_COLUMNS,
    AggregateReleaseMetadata,
    decode_aggregate_release_payload,
    encode_aggregate_release_payload,
    metadata_from_payload,
)
from .aggregate_release_layout import (
    AGGREGATE_RELEASE_ARRAY_FILES,
    AGGREGATE_RELEASE_TABLE_FILES,
    EncodedAggregateProducts,
)
from .aggregate_release_payload import (
    AGGREGATE_RELEASE_SCHEMA_VERSION,
    AggregateReleasePayload,
)
from .aggregate_release_records import AggregateShardBinding
from .aggregate_spec import AggregateSpec, load_aggregate_spec
from .run_control import (
    _cross_check_plan_progress,
    load_run_plan,
    load_run_progress,
    validate_run_plan_document,
    validate_run_progress_document,
)
from .run_locking import RunLockBusyError, acquire_run_lock
from .run_validation import validate_run

__all__ = [
    "TABLE_SCHEMAS",
    "read_aggregate_release",
    "validate_aggregate_release",
    "write_aggregate_release",
]

_MANIFEST_FILE_NAME: Final[str] = "aggregate_manifest.json"
_FINAL_DIRECTORY_SUFFIX: Final[str] = ".aggregate-v1"

# 來源快照以固定順序驗證與摘要；這些名稱只可在 release 根目錄直接組合，不能由
# manifest 注入其他相對路徑。順序同時固定 checksum 與第一個失敗 stage 的重現性。
_SOURCE_FILE_ORDER: Final[tuple[str, ...]] = (
    "aggregate_spec.json",
    "source_run_plan.json",
    "source_run_progress.json",
    "source_normalized_config.json",
    "source_input_inventory.json",
)

# 這五個檔名是本模組 writer 從 prepared snapshot 寫出的固定來源快照，也是 validator
# strict parse、雜湊與 provenance 綁定時唯一允許讀取的 JSON 名稱。它們固定放在 release
# 根目錄，不接受 manifest 或 caller 注入任意相對路徑，也不回讀 raw OCM／NWW forcing。
_SOURCE_JSON_FILES: Final[frozenset[str]] = frozenset(
    {
        "aggregate_spec.json",
        "source_run_plan.json",
        "source_run_progress.json",
        "source_normalized_config.json",
        "source_input_inventory.json",
    }
)

# release payload 是 manifest 之外的完整資料檔案集合。manifest 本身刻意不加入此集合，
# 讓後續 validator 能先以固定名稱枚舉五個 JSON、九張表與十八個陣列，再另外處理唯一
# 的 aggregate_manifest.json；任何額外檔案／缺檔政策由上層 caller 執行。
_RELEASE_PAYLOAD_FILES: Final[frozenset[str]] = (
    _SOURCE_JSON_FILES | AGGREGATE_RELEASE_TABLE_FILES | AGGREGATE_RELEASE_ARRAY_FILES
)

# I/O helper 以固定順序處理檔案，避免依 frozenset 的非定序迭代產生難以重現的錯誤
# 訊息或 contract 順序。這個順序不改變表格資料列的科學排序；列排序已由 codec 在
# 建立 EncodedAggregateProducts 前決定。
_TABLE_FILE_ORDER: Final[tuple[str, ...]] = (
    "shard_bindings.parquet",
    "scenario_strata.parquet",
    "site_index.parquet",
    "boundary_index.parquet",
    "source_receptor_index.parquet",
    "cross_site_counts.parquet",
    "outcome_counts.parquet",
    "site_denominators.parquet",
    "receptor_denominators.parquet",
)
_ARRAY_FILE_ORDER: Final[tuple[str, ...]] = (
    "age_bin_edges_seconds.npy",
    "site_cell_offsets.npy",
    "local_first_exit_count.npy",
    "outer_first_exit_count.npy",
    "bed_first_contact_count.npy",
    "bed_repeated_contact_count.npy",
    "data_gap_failure_count.npy",
    "numerical_failure_count.npy",
    "pathway_unique_particle_count.npy",
    "pathway_residence_time_seconds.npy",
    "pathway_first_passage_age_histogram.npy",
    "boundary_edge_offsets.npy",
    "boundary_bin_edges_m.npy",
    "boundary_bin_offsets.npy",
    "boundary_arclength_raw_count.npy",
    "boundary_travel_age_histogram.npy",
    "source_receptor_raw_count.npy",
    "source_receptor_travel_age_histogram.npy",
)

# NPY 檔名和 dtype 是一對一的容器契約。只有 age 邊界、停留秒數及 boundary bin 公尺
# 邊界是 float64；三類 histogram 都是 int64 count，其他整數檔則保存非負計數或 offset。
# reader 會以 dtype.str 精確比對，不能讀到後再用轉型、裁切或填零偽裝成合法 release。
_ARRAY_DTYPES: Final[Mapping[str, str]] = {
    "age_bin_edges_seconds.npy": "<f8",
    "site_cell_offsets.npy": "<i8",
    "local_first_exit_count.npy": "<i8",
    "outer_first_exit_count.npy": "<i8",
    "bed_first_contact_count.npy": "<i8",
    "bed_repeated_contact_count.npy": "<i8",
    "data_gap_failure_count.npy": "<i8",
    "numerical_failure_count.npy": "<i8",
    "pathway_unique_particle_count.npy": "<i8",
    "pathway_residence_time_seconds.npy": "<f8",
    "pathway_first_passage_age_histogram.npy": "<i8",
    "boundary_edge_offsets.npy": "<i8",
    "boundary_bin_edges_m.npy": "<f8",
    "boundary_bin_offsets.npy": "<i8",
    "boundary_arclength_raw_count.npy": "<i8",
    "boundary_travel_age_histogram.npy": "<i8",
    "source_receptor_raw_count.npy": "<i8",
    "source_receptor_travel_age_histogram.npy": "<i8",
}
_ARRAY_DTYPES = MappingProxyType(dict(_ARRAY_DTYPES))

# 只有三種真正代表連續量的陣列使用 float64：age 邊界是秒、pathway residence 是秒、
# boundary bin 邊界是公尺。其餘十五種陣列（包含 first-passage／travel-age histogram）
# 都是非負計數或索引，必須保存為 little-endian int64；histogram 的 age 軸位置不會
# 因為其數值是計數而改變，仍由上游 codec 的 shape／offset 契約解釋。
_FLOAT_ARRAY_FILES: Final[frozenset[str]] = frozenset(
    {
        "age_bin_edges_seconds.npy",
        "pathway_residence_time_seconds.npy",
        "boundary_bin_edges_m.npy",
    }
)
_INT_ARRAY_FILES: Final[frozenset[str]] = frozenset(_ARRAY_FILE_ORDER) - _FLOAT_ARRAY_FILES

_INT64_MIN: Final[int] = int(np.iinfo(np.int64).min)
_INT64_MAX: Final[int] = int(np.iinfo(np.int64).max)
_FLOAT_SCHEMA: Final[pa.DataType] = pa.float64()
_INT_SCHEMA: Final[pa.DataType] = pa.int64()
_STRING_SCHEMA: Final[pa.DataType] = pa.string()

# 九張表的欄位 tuple 是 schema 1 的資料契約：欄位名稱與順序同時影響 Parquet physical
# layout、manifest field contract 與 decoder 的 typed join。這裡保留 codec 的 internal
# 欄位常數作 import-time 交叉斷言；若 codec 的正式欄位被誤改，module import 應立即
# 失敗，而不是讓 I/O 層以錯位欄位產生看似可讀但語意錯誤的產品。
TABLE_SCHEMAS: Mapping[str, pa.Schema] = {
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
TABLE_SCHEMAS = MappingProxyType(dict(TABLE_SCHEMAS))


# 這些是 import-time 的 schema 邊界檢查，不是 runtime validator：它們只確認本檔明示
# 的九張表仍與 codec 正式欄位契約完全同名、同順序、同檔案集合。資料列 scalar、null
# 值、Parquet bytes、array shape 與跨表拓撲會由後續 I/O helper 逐項驗證。
assert frozenset(TABLE_SCHEMAS) == AGGREGATE_RELEASE_TABLE_FILES
assert tuple(TABLE_SCHEMAS["shard_bindings.parquet"].names) == _SHARD_BINDING_COLUMNS
assert tuple(TABLE_SCHEMAS["scenario_strata.parquet"].names) == _SCENARIO_STRATUM_COLUMNS
assert tuple(TABLE_SCHEMAS["site_index.parquet"].names) == _SITE_INDEX_COLUMNS
assert tuple(TABLE_SCHEMAS["boundary_index.parquet"].names) == _BOUNDARY_INDEX_COLUMNS
assert tuple(TABLE_SCHEMAS["source_receptor_index.parquet"].names) == _SOURCE_RECEPTOR_INDEX_COLUMNS
assert tuple(TABLE_SCHEMAS["cross_site_counts.parquet"].names) == _CROSS_SITE_COUNT_COLUMNS
assert tuple(TABLE_SCHEMAS["outcome_counts.parquet"].names) == _OUTCOME_COUNT_COLUMNS
assert tuple(TABLE_SCHEMAS["site_denominators.parquet"].names) == _SITE_DENOMINATOR_COLUMNS
assert tuple(TABLE_SCHEMAS["receptor_denominators.parquet"].names) == _RECEPTOR_DENOMINATOR_COLUMNS

# 這些 module import-time 斷言把固定檔案清單、欄位 schema 與 NPY 型別表綁成同一個
# compile-time-like 邊界。它們不是對外 validator，也不讀磁碟；若開發者日後只修改
# 某一處而漏改另一處，import 階段就應失敗，禁止產生部分相容的 release。
assert tuple(TABLE_SCHEMAS) == _TABLE_FILE_ORDER
assert frozenset(_TABLE_FILE_ORDER) == AGGREGATE_RELEASE_TABLE_FILES
assert frozenset(_ARRAY_FILE_ORDER) == AGGREGATE_RELEASE_ARRAY_FILES
assert tuple(_ARRAY_DTYPES) == _ARRAY_FILE_ORDER
assert frozenset(
    {
        "aggregate_spec.json",
        "source_run_plan.json",
        "source_run_progress.json",
        "source_normalized_config.json",
        "source_input_inventory.json",
    }
) == _SOURCE_JSON_FILES
assert frozenset(
    filename for filename, dtype in _ARRAY_DTYPES.items() if dtype == "<f8"
) == _FLOAT_ARRAY_FILES
assert frozenset(
    filename for filename, dtype in _ARRAY_DTYPES.items() if dtype == "<i8"
) == _INT_ARRAY_FILES
assert len(_SOURCE_JSON_FILES) == 5
assert len(AGGREGATE_RELEASE_TABLE_FILES) == 9
assert len(AGGREGATE_RELEASE_ARRAY_FILES) == 18
assert len(_FLOAT_ARRAY_FILES) == 3
assert len(_INT_ARRAY_FILES) == 15
assert len(_RELEASE_PAYLOAD_FILES) == 32


def _sha256_file(path: Path) -> str:
    """以串流方式計算檔案 SHA-256，避免大型 Parquet／NPY 整檔載入記憶體。

    ``path`` 由 writer 以固定 release 檔名建立；函式只讀取 bytes 並回傳 64 碼小寫
    digest，不解析資料內容，也不改變檔案位置、權限或時間。1 MiB 分塊可限制記憶體
    峰值，且 SHA-256 的結果與一次讀入全部 bytes 完全相同。
    """

    digest = sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_encoded_products(
    directory: str | Path,
    products: EncodedAggregateProducts,
) -> dict[str, dict[str, object]]:
    """把已編碼九表與十八陣列寫入既有普通目錄並回傳檔案 contracts。

    入口目錄可為 ``Path`` 或可由 ``Path`` 解析的路徑值，但必須已存在、是普通目錄且
    本身不可為符號連結。函式在任何寫入前會完整掃描全部 27 個固定目標；任一目標已
    存在（包含 broken symlink）即拒絕整次操作。掃描後的實際開檔仍使用 ``xb`` exclusive
    create，避免競態期間出現同名檔案時覆寫既有資料。

    ``products`` 必須是 exact ``EncodedAggregateProducts``，並再透過其公開 constructor
    建立防禦性快照。九張表逐列要求欄位 iteration order 與 ``TABLE_SCHEMAS`` 完全一致，
    每欄以明示 Arrow type 建立 array；non-nullable 欄位不可含 null，空的單站 cross-site
    表仍以三欄 schema 寫出。Parquet 固定使用 Zstandard 壓縮、不建 dictionary，並保存
    statistics。十八個陣列只接受一維、C-contiguous、native int64／float64；計數、索引、
    秒數與公尺值皆須非負，浮點值另須有限。實體 NPY 一律防禦性複製成 little-endian
    ``<i8``／``<f8``，且禁止 pickle。

    回傳 mapping 恰有 27 個檔名。共同 contract 保存 ``kind``、檔案 byte 數與 SHA-256；
    Parquet 另保存 row count 與欄位名稱／Arrow type／nullability，NPY 另保存 little-endian
    dtype、shape list 與 element count。函式不建立 manifest、不讀回產品，也不清理 caller
    目錄；若寫入中途發生 I/O 錯誤，已成功建立的檔案會原樣保留供上層診斷或清理。

    Raises:
        ValueError: 目錄、產品型別、表格 schema 或陣列 dtype／shape／數值不符契約。
        FileExistsError: 任一固定目標在掃描時已存在，或 exclusive-create 遇到競態檔案。
    """

    try:
        output_directory = Path(directory)
    except (TypeError, ValueError) as error:
        raise ValueError("directory 必須可轉換成 Path") from error
    if output_directory.is_symlink() or not output_directory.is_dir():
        raise ValueError("directory 必須是已存在的普通非 symlink 目錄")

    # 先建立全部固定目標並完整掃描，不能在發現後段衝突前已寫入前段檔案。is_symlink
    # 另行檢查 broken symlink，因為這類目標的 exists() 會回傳 False，但仍不可覆寫。
    target_paths = {
        file_name: output_directory / file_name
        for file_name in (*_TABLE_FILE_ORDER, *_ARRAY_FILE_ORDER)
    }
    conflicts = tuple(
        file_name
        for file_name, target in target_paths.items()
        if target.exists() or target.is_symlink()
    )
    if conflicts:
        raise FileExistsError(f"aggregate release 固定目標已存在：{conflicts!r}")

    if type(products) is not EncodedAggregateProducts:
        raise ValueError("products 必須是 exact EncodedAggregateProducts 實例")
    # frozen dataclass 與唯讀 mapping 仍可能被 object.__setattr__ 低階竄改；重跑公開
    # constructor 可重新核對固定 key set、table snapshots 與安全數值陣列，不直接信任
    # caller 持有的 reference。
    validated_products = EncodedAggregateProducts(
        tables=products.tables,
        arrays=products.arrays,
    )

    # 所有 Arrow table 先在記憶體建好並完成 schema/null 驗證，避免資料契約錯誤造成只寫
    # 出前幾張表。真正 I/O 錯誤仍依函式契約保留已寫檔案，不做隱式 rollback。
    prepared_tables: dict[str, pa.Table] = {}
    for file_name in _TABLE_FILE_ORDER:
        schema = TABLE_SCHEMAS[file_name]
        rows = validated_products.tables[file_name]
        expected_columns = tuple(schema.names)
        for row_index, row in enumerate(rows):
            if tuple(row.keys()) != expected_columns:
                raise ValueError(
                    f"{file_name}[{row_index}] 欄位名稱與順序必須精確等於 explicit schema"
                )

        arrow_columns: list[pa.Array] = []
        for field in schema:
            values: list[object] = []
            for row_index, row in enumerate(rows):
                value = row[field.name]
                if value is None:
                    if not field.nullable:
                        raise ValueError(
                            f"{file_name}[{row_index}].{field.name} 是 non-nullable，不可為 None"
                        )
                    values.append(None)
                    continue

                # Arrow 可把 bool、NumPy scalar 或其他可轉型物件靜默 cast 成 schema
                # dtype；writer 必須在呼叫 pa.array 前先守住 release table 的原生 Python
                # scalar 邊界，否則不同 adapter 可能把同一份錯型資料序列化成不同結果。
                if field.type == _STRING_SCHEMA:
                    if type(value) is not str or not value or value != value.strip():
                        raise ValueError(
                            f"{file_name}[{row_index}].{field.name} "
                            "必須是非空且無首尾空白的原生 str"
                        )
                elif field.type == _INT_SCHEMA:
                    if type(value) is not int:
                        raise ValueError(
                            f"{file_name}[{row_index}].{field.name} 必須是原生 int"
                        )
                    if not _INT64_MIN <= value <= _INT64_MAX:
                        raise ValueError(
                            f"{file_name}[{row_index}].{field.name} 必須位於 signed int64 範圍"
                        )
                    is_nonnegative_field = (
                        field.name.endswith(("_index", "_count", "_offset"))
                        or field.name in {"count", "valid_member_denominator"}
                    )
                    if is_nonnegative_field and value < 0:
                        raise ValueError(
                            f"{file_name}[{row_index}].{field.name} 不可為負"
                        )
                elif field.type == _FLOAT_SCHEMA:
                    if type(value) is not float or not math.isfinite(value):
                        raise ValueError(
                            f"{file_name}[{row_index}].{field.name} 必須是有限原生 float"
                        )
                else:
                    raise ValueError(
                        f"{file_name}[{row_index}].{field.name} 使用未登錄的 Arrow type"
                    )
                values.append(value)

            try:
                column = pa.array(values, type=field.type)
            except (pa.ArrowException, TypeError, ValueError, OverflowError) as error:
                raise ValueError(
                    f"{file_name}.{field.name} 無法依 explicit Arrow type 建立欄位"
                ) from error
            if not field.nullable and column.null_count != 0:
                raise ValueError(f"{file_name}.{field.name} 是 non-nullable，不可含 null")
            arrow_columns.append(column)

        table = pa.Table.from_arrays(arrow_columns, schema=schema)
        if not table.schema.equals(schema, check_metadata=True):
            raise ValueError(f"{file_name} 的 Arrow table schema 與 explicit schema 不一致")
        prepared_tables[file_name] = table

    # 先驗證原生記憶體產品，再複製成固定 little-endian 儲存 dtype。這個轉換只改變 byte
    # order，不允許把浮點近似成整數、把負值 clip 成零，或把多維資料 reshape 成一維。
    prepared_arrays: dict[str, np.ndarray] = {}
    for file_name in _ARRAY_FILE_ORDER:
        values = validated_products.arrays[file_name]
        storage_dtype = _ARRAY_DTYPES[file_name]
        expected_native_dtype = (
            np.dtype(np.float64)
            if file_name in _FLOAT_ARRAY_FILES
            else np.dtype(np.int64)
        )
        if not isinstance(values, np.ndarray) or values.dtype != expected_native_dtype:
            raise ValueError(
                f"{file_name} 必須是 exact native {expected_native_dtype.name} np.ndarray"
            )
        if values.ndim != 1:
            raise ValueError(f"{file_name} 必須是一維陣列")
        if not values.flags.c_contiguous:
            raise ValueError(f"{file_name} 必須是 C-contiguous")
        if file_name in _FLOAT_ARRAY_FILES:
            if not bool(np.all(np.isfinite(values))):
                raise ValueError(f"{file_name} 的 float64 值必須全部有限")
            if bool(np.any(values < 0.0)):
                raise ValueError(f"{file_name} 的 float64 值不可為負")
            if storage_dtype != "<f8":
                raise ValueError(f"{file_name} 與 float64 儲存契約不一致")
        else:
            if bool(np.any(values < 0)):
                raise ValueError(f"{file_name} 的 int64 值不可為負")
            if storage_dtype != "<i8":
                raise ValueError(f"{file_name} 與 int64 儲存契約不一致")

        stored = np.array(
            values,
            dtype=np.dtype(storage_dtype),
            order="C",
            copy=True,
        )
        if stored.dtype.str != storage_dtype or stored.ndim != 1 or not stored.flags.c_contiguous:
            raise ValueError(f"{file_name} 無法建立固定 little-endian 一維儲存副本")
        prepared_arrays[file_name] = stored

    contracts: dict[str, dict[str, object]] = {}
    for file_name in _TABLE_FILE_ORDER:
        target = target_paths[file_name]
        table = prepared_tables[file_name]
        # Python binary stream 使用 xb 排除掃描後的同名檔競態；pyarrow 直接寫入此 stream，
        # 不會以 path API 的覆寫語意重新開啟目標。
        with target.open("xb") as stream:
            pq.write_table(
                table,
                stream,
                compression="zstd",
                use_dictionary=False,
                write_statistics=True,
            )
        contracts[file_name] = {
            "kind": "parquet",
            "size_bytes": target.stat().st_size,
            "sha256": _sha256_file(target),
            "row_count": table.num_rows,
            "fields": [
                {
                    "name": field.name,
                    "type": str(field.type),
                    "nullable": field.nullable,
                }
                for field in table.schema
            ],
        }

    for file_name in _ARRAY_FILE_ORDER:
        target = target_paths[file_name]
        values = prepared_arrays[file_name]
        with target.open("xb") as stream:
            np.save(stream, values, allow_pickle=False)
        contracts[file_name] = {
            "kind": "npy",
            "size_bytes": target.stat().st_size,
            "sha256": _sha256_file(target),
            "dtype": values.dtype.str,
            "shape": [int(length) for length in values.shape],
            "element_count": int(values.size),
        }

    if len(contracts) != len(_TABLE_FILE_ORDER) + len(_ARRAY_FILE_ORDER):
        raise RuntimeError("writer 必須恰好建立 27 份產品 contract")
    return contracts


def _require_exact_parquet_schema(
    actual_schema: pa.Schema,
    *,
    expected_schema: pa.Schema,
    file_name: str,
) -> None:
    """驗證已開啟 Parquet 的 Arrow schema 完全符合固定 release 契約。

    欄位名稱、順序、logical type 與 nullability 都必須逐欄完全相同；schema 與 field
    metadata 一律禁止，避免 pandas metadata、任意 adapter 標記或未登錄 extension
    改變同一張表的語意。``file_name`` 只接受本模組固定檔名，錯誤訊息不包含 directory
    或絕對路徑。此 helper 不讀 row、不依 manifest 猜 schema，也不執行跨表 join。
    """

    if actual_schema.metadata is not None:
        raise ValueError(f"{file_name} 的 Arrow schema 不可含 metadata")
    if tuple(actual_schema.names) != tuple(expected_schema.names):
        raise ValueError(f"{file_name} 的欄位名稱與順序不符合固定 schema")
    if len(actual_schema) != len(expected_schema):
        raise ValueError(f"{file_name} 的欄位數量不符合固定 schema")

    for actual_field, expected_field in zip(
        actual_schema,
        expected_schema,
        strict=True,
    ):
        if actual_field.metadata is not None:
            raise ValueError(f"{file_name}.{actual_field.name} 不可含 field metadata")
        if (
            actual_field.name != expected_field.name
            or actual_field.type != expected_field.type
            or actual_field.nullable != expected_field.nullable
        ):
            raise ValueError(
                f"{file_name}.{expected_field.name} 的 type 或 nullability 不符合固定 schema"
            )
    if not actual_schema.equals(expected_schema, check_metadata=True):
        raise ValueError(f"{file_name} 的 Arrow schema 不符合固定 exact contract")


def _validate_loaded_table_scalar(
    value: object,
    *,
    field: pa.Field,
    file_name: str,
    row_index: int,
) -> object:
    """驗證 Parquet row 的原生 Python scalar 與欄位 nullability。

    Arrow 的 ``to_pylist`` 可能把值 materialize 成 Python scalar；reader 仍明確要求
    string／int64／float64 分別成為 exact ``str``／``int``／``float``，拒絕 bool、
    NumPy scalar 或其他可轉型代理。只有 schema 中十五個 scenario ``initial_*`` 欄位
    可為 ``None``；整組有值／整組缺值政策留給 codec decoder 的公開 record constructor
    驗證。計數、index 與 offset 必須非負，浮點位置、深度、公尺或秒 scalar 必須有限，
    但合法的負座標、向上為正深度與沉降速度不在本 I/O helper 改寫或 clip。
    """

    label = f"{file_name}[{row_index}].{field.name}"
    if value is None:
        if not field.nullable:
            raise ValueError(f"{label} 是 non-nullable，不可為 None")
        return None

    if field.type == _STRING_SCHEMA:
        if type(value) is not str or not value or value != value.strip():
            raise ValueError(f"{label} 必須是非空且無首尾空白的原生 str")
    elif field.type == _INT_SCHEMA:
        if type(value) is not int:
            raise ValueError(f"{label} 必須是原生 int")
        if not _INT64_MIN <= value <= _INT64_MAX:
            raise ValueError(f"{label} 必須位於 signed int64 範圍")
        is_nonnegative_field = (
            field.name.endswith(("_index", "_count", "_offset"))
            or field.name in {"count", "valid_member_denominator"}
        )
        if is_nonnegative_field and value < 0:
            raise ValueError(f"{label} 不可為負")
    elif field.type == _FLOAT_SCHEMA:
        if type(value) is not float or not math.isfinite(value):
            raise ValueError(f"{label} 必須是有限原生 float")
    else:
        raise ValueError(f"{label} 使用未登錄的 Arrow type")
    return value


def _read_encoded_products(directory: str | Path) -> EncodedAggregateProducts:
    """由既有 release 目錄讀取固定九表與十八個 NPY 記憶體產品。

    ``directory`` 必須已存在、是普通目錄且本身不可為 symbolic link。reader 只依
    ``_TABLE_FILE_ORDER`` 與 ``_ARRAY_FILE_ORDER`` 逐一組合固定檔名；每個目標都必須
    存在、是普通檔案且不可為 symbolic link，完全不接受 manifest 提供任意 path，亦不
    讀取五個 source JSON。所有 Arrow、NumPy 與檔案 I/O 失敗都轉成固定 ``ValueError``，
    訊息只含固定產品名稱，不暴露 release 的絕對路徑。

    Parquet 先以 ``pq.ParquetFile`` 檢查欄名、順序、logical type、nullability 與無 schema
    metadata，再以 ``pq.read_table``／``to_pylist`` materialize rows，逐值驗證 exact native
    Python scalar。單站 release 的 ``cross_site_counts.parquet`` 可得到精確空 tuple；其他
    空表由最終 ``EncodedAggregateProducts`` 固定拓撲 constructor 拒絕。NPY 使用
    ``np.load(..., allow_pickle=False)`` 從普通 binary stream 讀取，不使用 memory map；
    dtype.str 必須逐檔精確為 little-endian ``<i8``／``<f8``，陣列必須一維且 C-contiguous。
    int64 計數／offset 不可為負，float64 秒數與公尺值必須有限且非負；reader 不呼叫
    ``astype``、不 reshape、不 clip，也不以零值或最近值修補內容。

    回傳值只完成固定檔案、實體 schema、scalar 與 NPY storage-level 驗證，最後交由
    ``EncodedAggregateProducts`` 建立防禦性唯讀 snapshot。site／boundary offsets、
    C-order histogram shape、零列拓撲與跨產品 join 的科學語意，必須在後續呼叫 codec
    decoder 時驗證；本 helper 不猜測或提前重建。產品仍只描述條件式來源足跡或相對
    來源權重，不能單獨解讀為絕對來源機率或因果歸因。

    Args:
        directory: 直接包含固定九張 Parquet 與十八個 NPY 的既有普通目錄。

    Returns:
        已封存九表與十八陣列的 ``EncodedAggregateProducts``。

    Raises:
        ValueError: 目錄／固定檔案節點不安全、Parquet schema／row scalar 不符，NPY
            dtype／shape／連續性／數值不符，或底層 Arrow／NumPy／I/O 無法安全讀取。
    """

    try:
        input_directory = Path(directory)
    except (TypeError, ValueError) as error:
        raise ValueError("directory 必須可轉換成 Path") from error
    try:
        directory_is_symlink = input_directory.is_symlink()
        directory_is_directory = input_directory.is_dir()
    except OSError as error:
        raise ValueError("directory 無法依固定產品 reader 契約驗證") from error
    if directory_is_symlink or not directory_is_directory:
        raise ValueError("directory 必須是已存在的普通非 symlink 目錄")

    target_paths = {
        file_name: input_directory / file_name
        for file_name in (*_TABLE_FILE_ORDER, *_ARRAY_FILE_ORDER)
    }
    for file_name in (*_TABLE_FILE_ORDER, *_ARRAY_FILE_ORDER):
        target = target_paths[file_name]
        try:
            if target.is_symlink():
                raise ValueError(f"{file_name} 不可為 symbolic link")
            if not target.exists():
                raise ValueError(f"{file_name} 固定產品缺失")
            if not target.is_file():
                raise ValueError(f"{file_name} 必須是普通檔案")
        except OSError as error:
            raise ValueError(f"{file_name} 無法驗證固定檔案節點") from error

    tables: dict[str, tuple[dict[str, object], ...]] = {}
    for file_name in _TABLE_FILE_ORDER:
        target = target_paths[file_name]
        expected_schema = TABLE_SCHEMAS[file_name]
        try:
            parquet_file = pq.ParquetFile(target)
            physical_schema = parquet_file.schema_arrow
        except (pa.ArrowException, OSError, TypeError, ValueError) as error:
            raise ValueError(f"{file_name} 無法讀取固定 Parquet schema") from error
        _require_exact_parquet_schema(
            physical_schema,
            expected_schema=expected_schema,
            file_name=file_name,
        )

        try:
            table = pq.read_table(target)
        except (pa.ArrowException, OSError, TypeError, ValueError) as error:
            raise ValueError(f"{file_name} 無法讀取固定 Parquet table") from error
        _require_exact_parquet_schema(
            table.schema,
            expected_schema=expected_schema,
            file_name=file_name,
        )
        try:
            loaded_rows = table.to_pylist()
        except (pa.ArrowException, TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{file_name} 無法轉成固定 Python row") from error

        validated_rows: list[dict[str, object]] = []
        expected_columns = tuple(expected_schema.names)
        for row_index, row in enumerate(loaded_rows):
            if type(row) is not dict or tuple(row.keys()) != expected_columns:
                raise ValueError(
                    f"{file_name}[{row_index}] 欄位名稱與順序必須精確等於固定 schema"
                )
            validated_row: dict[str, object] = {}
            for field in expected_schema:
                validated_row[field.name] = _validate_loaded_table_scalar(
                    row[field.name],
                    field=field,
                    file_name=file_name,
                    row_index=row_index,
                )
            validated_rows.append(validated_row)
        if table.num_rows != len(validated_rows):
            raise ValueError(f"{file_name} 的 Arrow row count 與 materialized rows 不一致")
        tables[file_name] = tuple(validated_rows)

    arrays: dict[str, np.ndarray] = {}
    for file_name in _ARRAY_FILE_ORDER:
        target = target_paths[file_name]
        try:
            with target.open("rb") as stream:
                loaded = np.load(stream, allow_pickle=False)
        except (EOFError, OSError, TypeError, ValueError) as error:
            raise ValueError(f"{file_name} 無法讀取固定 NPY 產品") from error
        if type(loaded) is not np.ndarray:
            raise ValueError(f"{file_name} 必須是單一 NPY ndarray，不接受壓縮 archive")

        expected_dtype = _ARRAY_DTYPES[file_name]
        if loaded.dtype.str != expected_dtype:
            raise ValueError(f"{file_name} dtype.str 必須精確為 {expected_dtype}")
        if loaded.ndim != 1:
            raise ValueError(f"{file_name} 必須是一維陣列")
        if not loaded.flags.c_contiguous:
            raise ValueError(f"{file_name} 必須是 C-contiguous")
        if file_name in _INT_ARRAY_FILES:
            if bool(np.any(loaded < 0)):
                raise ValueError(f"{file_name} 的 int64 值不可為負")
        elif file_name in _FLOAT_ARRAY_FILES:
            if not bool(np.all(np.isfinite(loaded))):
                raise ValueError(f"{file_name} 的 float64 值必須全部有限")
            if bool(np.any(loaded < 0.0)):
                raise ValueError(f"{file_name} 的 float64 值不可為負")
        else:
            raise ValueError(f"{file_name} 缺少固定 NPY 數值型別契約")
        arrays[file_name] = loaded

    return EncodedAggregateProducts(tables=tables, arrays=arrays)


class _ReleaseValidationError(ValueError):
    """表示 release 基礎驗證失敗，只攜帶固定 stage code。

    ``stage`` 只允許已登錄的 topology、manifest、name、checksum、source、products
    或 decoder；例外文字不串接 caller 路徑、作業系統訊息或第三方細節，讓公開
    validator 能安全轉成 JSON 報告。
    """

    _STAGES: Final[frozenset[str]] = frozenset(
        {
            "topology",
            "manifest",
            "name",
            "checksum",
            "source",
            "products",
            "decoder",
        }
    )

    def __init__(self, stage: str) -> None:
        if stage not in self._STAGES:
            raise ValueError("release validation stage 未登錄")
        self.stage = stage
        super().__init__(stage)


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    """拒絕任一層 JSON object 的 duplicate key，避免後值靜默覆蓋前值。"""

    document: dict[str, object] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError("JSON object 含重複欄位")
        document[key] = value
    return document


def _reject_json_constant(_token: str) -> NoReturn:
    """拒絕 Python JSON decoder 額外接受的 NaN 與 Infinity token。"""

    raise ValueError("JSON 不允許非有限常數")


def _reject_nonfinite_json_numbers(value: object) -> None:
    """遞迴拒絕 ``1e999`` 等解析後才成為無限大的 JSON number。"""

    if type(value) is float and not math.isfinite(value):
        raise ValueError("JSON number 必須有限")
    if type(value) is dict:
        for nested in value.values():
            _reject_nonfinite_json_numbers(nested)
    elif type(value) is list:
        for nested in value:
            _reject_nonfinite_json_numbers(nested)


def _strict_json_object_from_bytes(raw_bytes: bytes) -> dict[str, object]:
    """把嚴格 UTF-8 JSON bytes 解析成根節點普通 dict。

    parser 不接受編碼替代、duplicate key、NaN／Infinity，亦會遞迴拒絕極大指數解析
    成的非有限 float。根節點只接受 JSON object，不把 list、scalar 或可轉型 mapping
    當成文件；此邊界將由 manifest 使用，後續 source semantic slice 也應重用同一實作。
    """

    if type(raw_bytes) is not bytes:
        raise ValueError("JSON 輸入必須是 bytes")
    document = json.loads(
        raw_bytes.decode("utf-8", errors="strict"),
        object_pairs_hook=_reject_duplicate_json_keys,
        parse_constant=_reject_json_constant,
    )
    _reject_nonfinite_json_numbers(document)
    if type(document) is not dict:
        raise ValueError("JSON root 必須是 object")
    return document


def _canonical_json_bytes(document: object) -> bytes:
    """依 schema 1 規則產生 compact canonical UTF-8 JSON 與唯一尾端換行。"""

    return (
        json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _open_regular_file_descriptor(path: Path) -> int:
    """以不跟隨 symbolic link 的方式開啟普通檔。

    topology 已先以 ``lstat`` 鎖定節點類型，實際讀取仍在平台可用時加入
    ``O_NOFOLLOW`` 並以 ``fstat`` 再檢查，避免檢查後被換成目錄、FIFO 或連結。
    """

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("固定節點不是普通檔")
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _read_regular_file_bytes(path: Path) -> bytes:
    """從普通檔 descriptor 讀取 exact bytes，不跟隨連結也不修改來源。"""

    descriptor = _open_regular_file_descriptor(path)
    chunks: list[bytes] = []
    try:
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def _regular_file_size_and_sha256(path: Path) -> tuple[int, str]:
    """串流計算普通檔大小與 SHA-256，不解析 JSON、Parquet 或 NPY 內容。"""

    descriptor = _open_regular_file_descriptor(path)
    digest = sha256()
    size_bytes = 0
    try:
        while chunk := os.read(descriptor, 1024 * 1024):
            size_bytes += len(chunk)
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return size_bytes, digest.hexdigest()


def _require_nonnegative_native_int(value: object, *, label: str) -> int:
    """要求 contract 整數為非 bool 的非負原生 Python ``int``。"""

    if type(value) is not int or value < 0:
        raise ValueError(f"{label} 必須是非負原生 int")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    """要求 digest 精確為 64 碼小寫十六進位 SHA-256。"""

    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} 必須是 64 碼小寫 SHA-256")
    return value


def _expected_field_contracts(file_name: str) -> list[dict[str, object]]:
    """由固定 Arrow schema 建立有序 manifest field contracts。"""

    return [
        {
            "name": field.name,
            "type": str(field.type),
            "nullable": field.nullable,
        }
        for field in TABLE_SCHEMAS[file_name]
    ]


def _validate_manifest_file_contracts(
    files: object,
) -> dict[str, dict[str, object]]:
    """驗證 manifest 的三十二份 exact file contracts。

    五份 JSON、九張 Parquet 與十八個 NPY 的檔名及 contract key set 皆固定。共同
    size/hash 拒絕 bool、負值與大小寫錯誤摘要；Parquet fields 必須是 ordered
    ``list[dict]`` 且逐欄等於 ``TABLE_SCHEMAS``；NPY dtype 必須等於
    ``_ARRAY_DTYPES``，shape 只能是一維原生非負 int list，並與 element count 相等。
    此函式只驗宣告，不開啟 manifest 指定的任何 path。
    """

    if type(files) is not dict or frozenset(files) != _RELEASE_PAYLOAD_FILES:
        raise ValueError("manifest files 必須恰好包含固定三十二檔")

    validated: dict[str, dict[str, object]] = {}
    for file_name in (*_SOURCE_FILE_ORDER, *_TABLE_FILE_ORDER, *_ARRAY_FILE_ORDER):
        contract = files[file_name]
        if type(contract) is not dict:
            raise ValueError("manifest file contract 必須是普通 object")

        if file_name in _SOURCE_JSON_FILES:
            expected_keys = {"kind", "size_bytes", "sha256"}
            expected_kind = "json"
        elif file_name in AGGREGATE_RELEASE_TABLE_FILES:
            expected_keys = {"kind", "size_bytes", "sha256", "row_count", "fields"}
            expected_kind = "parquet"
        else:
            expected_keys = {
                "kind",
                "size_bytes",
                "sha256",
                "dtype",
                "shape",
                "element_count",
            }
            expected_kind = "npy"

        if set(contract) != expected_keys:
            raise ValueError("manifest file contract key set 錯誤")
        if type(contract["kind"]) is not str or contract["kind"] != expected_kind:
            raise ValueError("manifest file kind 錯誤")
        _require_nonnegative_native_int(
            contract["size_bytes"],
            label=f"{file_name}.size_bytes",
        )
        _require_sha256(contract["sha256"], label=f"{file_name}.sha256")

        if file_name in AGGREGATE_RELEASE_TABLE_FILES:
            _require_nonnegative_native_int(
                contract["row_count"],
                label=f"{file_name}.row_count",
            )
            fields = contract["fields"]
            if type(fields) is not list:
                raise ValueError("Parquet fields 必須是 ordered list")
            for field in fields:
                if type(field) is not dict or set(field) != {
                    "name",
                    "type",
                    "nullable",
                }:
                    raise ValueError("Parquet field contract key set 錯誤")
                if (
                    type(field["name"]) is not str
                    or type(field["type"]) is not str
                    or type(field["nullable"]) is not bool
                ):
                    raise ValueError("Parquet field contract scalar 型別錯誤")
            if fields != _expected_field_contracts(file_name):
                raise ValueError("Parquet fields 與固定 schema 不一致")

        if file_name in AGGREGATE_RELEASE_ARRAY_FILES:
            if (
                type(contract["dtype"]) is not str
                or contract["dtype"] != _ARRAY_DTYPES[file_name]
            ):
                raise ValueError("NPY dtype contract 錯誤")
            shape = contract["shape"]
            if type(shape) is not list or len(shape) != 1:
                raise ValueError("NPY shape 必須是精確一維 list")
            shape_length = _require_nonnegative_native_int(
                shape[0],
                label=f"{file_name}.shape[0]",
            )
            element_count = _require_nonnegative_native_int(
                contract["element_count"],
                label=f"{file_name}.element_count",
            )
            if element_count != shape_length:
                raise ValueError("一維 NPY shape 與 element_count 不一致")

        validated[file_name] = dict(contract)
    return validated


def _validate_manifest_bytes(
    raw_bytes: bytes,
) -> tuple[AggregateReleaseMetadata, dict[str, dict[str, object]]]:
    """驗證 canonical manifest root、metadata 與三十二份 file contracts。

    root 只接受 ``schema_version``、``metadata``、``files`` 三個 key；
    schema 必須精確為 release schema 1，metadata 必須由
    ``AggregateReleaseMetadata.from_dict`` 深層重建，且其 schema 與 root 相同。
    manifest bytes 必須是 compact、排序鍵名、禁止 NaN 的 canonical JSON 加單一換行。
    """

    document = _strict_json_object_from_bytes(raw_bytes)
    if set(document) != {"schema_version", "metadata", "files"}:
        raise ValueError("manifest root key set 錯誤")
    if raw_bytes != _canonical_json_bytes(document):
        raise ValueError("manifest bytes 不是 canonical JSON")
    schema_version = document["schema_version"]
    if (
        type(schema_version) is not str
        or schema_version != AGGREGATE_RELEASE_SCHEMA_VERSION
    ):
        raise ValueError("manifest schema_version 不受支援")
    metadata_document = document["metadata"]
    if type(metadata_document) is not dict:
        raise ValueError("manifest metadata 必須是普通 object")
    metadata = AggregateReleaseMetadata.from_dict(metadata_document)
    if metadata.schema_version != schema_version:
        raise ValueError("manifest 與 metadata schema_version 不一致")
    contracts = _validate_manifest_file_contracts(document["files"])
    return metadata, contracts


def _validate_release_topology_and_checksums(
    directory: str | Path,
    *,
    require_final_name: bool,
) -> tuple[
    AggregateReleaseMetadata,
    dict[str, dict[str, object]],
    dict[str, bytes],
]:
    """驗證 release 固定 topology、manifest 與全部 payload checksums。

    根目錄必須是既有普通非 symbolic-link 目錄，且恰有一份 manifest 加三十二個固定
    普通非連結檔；任何額外、缺少、目錄、FIFO 或 broken symbolic link 都在讀 manifest
    前以 ``topology`` 拒絕。manifest 僅能宣告固定名稱，不可注入 path；若
    ``require_final_name`` 為真，basename 還必須精確為
    ``<metadata.run_id>.aggregate-v1``。

    驗證順序刻意先完成 manifest exact contract，再依固定五 JSON、九 Parquet、十八
    NPY 順序核對全部 size 與 SHA-256。只有三十二檔全部通過 checksum，才重新以安全
    普通檔 reader 讀取五份 source JSON exact bytes 並回傳；本 slice 不解析其 run/spec
    語意，也不開啟 Parquet、NPY 或 decoder。錯誤只會是
    ``_ReleaseValidationError`` 的 topology、manifest、name、checksum 固定 stage，
    不含 caller 絕對路徑或底層例外文字。

    Args:
        directory: release 根目錄。
        require_final_name: 是否強制 final basename；後續 writer 可對自有 partial 傳入
            ``False``，但不會因此放寬節點、manifest 或 checksum 契約。

    Returns:
        已驗證 metadata、三十二份普通 contract dict，以及固定五份 source JSON bytes。
    """

    try:
        if type(require_final_name) is not bool:
            raise ValueError("require_final_name 必須是 bool")
        root = Path(directory)
        root_stat = os.lstat(root)
        if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
            raise ValueError("release root 必須是普通非 symlink 目錄")

        expected_names = {_MANIFEST_FILE_NAME, *_RELEASE_PAYLOAD_FILES}
        with os.scandir(root) as entries:
            actual_names = {entry.name for entry in entries}
        if actual_names != expected_names or len(actual_names) != 33:
            raise ValueError("release root 固定節點集合錯誤")
        fixed_order = (
            _MANIFEST_FILE_NAME,
            *_SOURCE_FILE_ORDER,
            *_TABLE_FILE_ORDER,
            *_ARRAY_FILE_ORDER,
        )
        for file_name in fixed_order:
            node_stat = os.lstat(root / file_name)
            if stat.S_ISLNK(node_stat.st_mode) or not stat.S_ISREG(node_stat.st_mode):
                raise ValueError("release 固定節點必須是普通非 symlink 檔案")
    except Exception:
        raise _ReleaseValidationError("topology") from None

    try:
        manifest_bytes = _read_regular_file_bytes(root / _MANIFEST_FILE_NAME)
        metadata, contracts = _validate_manifest_bytes(manifest_bytes)
    except Exception:
        raise _ReleaseValidationError("manifest") from None

    if require_final_name:
        try:
            expected_name = f"{metadata.run_id}{_FINAL_DIRECTORY_SUFFIX}"
            if root.name != expected_name:
                raise ValueError("release final basename 錯誤")
        except Exception:
            raise _ReleaseValidationError("name") from None

    try:
        for file_name in (*_SOURCE_FILE_ORDER, *_TABLE_FILE_ORDER, *_ARRAY_FILE_ORDER):
            size_bytes, digest = _regular_file_size_and_sha256(root / file_name)
            contract = contracts[file_name]
            if (
                size_bytes != contract["size_bytes"]
                or digest != contract["sha256"]
            ):
                raise ValueError("release payload checksum 不一致")

        # source bytes 必須在全部三十二檔 checksum 通過後才開啟。此處只保存 exact bytes，
        # 缺值、公尺／秒欄位與 run/spec provenance 由下一個 semantic slice 處理。
        source_json_bytes = {
            file_name: _read_regular_file_bytes(root / file_name)
            for file_name in _SOURCE_FILE_ORDER
        }
    except Exception:
        raise _ReleaseValidationError("checksum") from None

    return metadata, contracts, source_json_bytes


@dataclass(frozen=True, slots=True)
class _SourceSnapshot:
    """封存 release 來源文件的深層快照與 provenance 摘要。

    ``aggregate_spec`` 是由 release 根目錄的完整 ``aggregate_spec.json`` 重建出的
    不可變規格；其餘四個 JSON 則保存已通過 run-control schema 的普通 Python 文件。
    plan／progress 內含巢狀 shard、file contract 與 metrics mapping，normalized config
    與 input inventory 也可能含巢狀 object／array，因此建立這個 private snapshot 時
    會再次深拷貝，而不保存呼叫端或 parser 暫存物件的可變 alias。``source_sha256_by_file``
    只記錄五份 release source bytes 的 SHA-256，供後續 decoder 或 writer 綁定；它們
    是位元組 provenance，不代表 OCM／NWW 的物理正確性，也不會把本機 synthetic
    fixture 提升成真實科學成果。

    這個類別只供本模組的 source semantic validator 使用，不是公開 API；frozen
    dataclass 只能保護欄位重新賦值，故 ``__post_init__`` 仍必須把所有 Mapping 深層
    複製後才封存。巢狀資料不在此轉型、補預設值或改寫缺值語意。
    """

    aggregate_spec: AggregateSpec
    plan: dict[str, object]
    progress: dict[str, object]
    normalized_config: dict[str, object]
    input_inventory: dict[str, object]
    source_sha256_by_file: Mapping[str, str]

    def __post_init__(self) -> None:
        """建立不與來源 parser 或呼叫端共享的 private defensive snapshot。"""

        if type(self.aggregate_spec) is not AggregateSpec:
            raise ValueError("aggregate_spec 必須是 exact AggregateSpec 實例")

        for field_name in (
            "plan",
            "progress",
            "normalized_config",
            "input_inventory",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, Mapping):
                raise ValueError(f"{field_name} 必須是 mapping")
            try:
                copied = deepcopy(dict(value))
            except Exception as error:
                raise ValueError(f"{field_name} 無法建立深層 snapshot") from error
            object.__setattr__(self, field_name, copied)

        if not isinstance(self.source_sha256_by_file, Mapping):
            raise ValueError("source_sha256_by_file 必須是 mapping")
        try:
            copied_digests = dict(self.source_sha256_by_file)
        except Exception as error:
            raise ValueError("source_sha256_by_file 無法建立 snapshot") from error
        if set(copied_digests) != set(_SOURCE_FILE_ORDER):
            raise ValueError("source_sha256_by_file 必須恰好包含五份 source JSON")
        for file_name in _SOURCE_FILE_ORDER:
            _require_sha256(
                copied_digests[file_name],
                label=f"source_sha256_by_file[{file_name}]",
            )
        object.__setattr__(
            self,
            "source_sha256_by_file",
            MappingProxyType(copied_digests),
        )


@dataclass(frozen=True, slots=True)
class _PreparedAggregateRelease:
    """封存 writer 前段已完成來源綁定的 aggregate release 記憶體快照。

    ``metadata`` 與 ``products`` 是由同一份 exact ``AggregateReleasePayload`` 重新驗證
    後產生的 typed snapshot；``source_json_bytes`` 則依固定五檔順序保存
    ``aggregate_spec.json``、run plan、run progress、normalized config 與 input inventory
    的原始 bytes。這個容器不保存任何 ``Path``、SERVER 絕對位置或 caller mapping 的
    alias，因而可安全交給後續 writer 階段建立 release 目錄。

    本類別只表示「source JSON／plan／progress／spec／payload 基本 binding 已通過」。它
    刻意不保存或驗證目前 trajectory shard 的 ``manifest.json``；那個輸出檔案的當下
    checksum 與 particle／observation／event 計數，必須由後續獨立的 shard-output binding
    helper 在發布前處理。即使此容器在本機 synthetic fixture 中建立成功，也只代表工程
    資料契約成立，不代表已取得正式 OCM schema 3／NWW3 schema 1，也不代表產生科學成果。
    """

    metadata: AggregateReleaseMetadata
    products: EncodedAggregateProducts
    source_json_bytes: Mapping[str, bytes]

    def __post_init__(self) -> None:
        """驗證 exact typed inputs，並建立 metadata／產品／bytes 的防禦性快照。"""

        if type(self.metadata) is not AggregateReleaseMetadata:
            raise ValueError("metadata 必須是 exact AggregateReleaseMetadata 實例")
        if type(self.products) is not EncodedAggregateProducts:
            raise ValueError("products 必須是 exact EncodedAggregateProducts 實例")

        # metadata 本身是 frozen dataclass，但低階 object.__setattr__ 仍可能竄改欄位；
        # 以公開 constructor 從 JSON-safe dict 重建，讓準備容器不直接信任 caller reference。
        metadata_snapshot = AggregateReleaseMetadata.from_dict(self.metadata.to_dict())
        # products 的表列與 ndarray 也可能被低階方式替換；公開容器 constructor 會重新
        # 複製列、鎖定 mapping 與 ndarray buffer，避免 writer 後續看到外部可變 alias。
        products_snapshot = EncodedAggregateProducts(
            tables=self.products.tables,
            arrays=self.products.arrays,
        )

        if not isinstance(self.source_json_bytes, Mapping):
            raise ValueError("source_json_bytes 必須是 mapping")
        try:
            raw_source_bytes = dict(self.source_json_bytes)
        except Exception as error:
            raise ValueError("source_json_bytes 無法建立防禦性 snapshot") from error
        if set(raw_source_bytes) != set(_SOURCE_FILE_ORDER):
            raise ValueError("source_json_bytes 必須恰好包含固定五份 source JSON")

        # bytes 是不可變值，但仍要求 exact 原生 bytes，拒絕 bytearray、memoryview 或自訂
        # bytes subclass；這樣後續 writer 複製的確是固定內容，不會因 adapter 轉型改變。
        source_snapshot: dict[str, bytes] = {}
        for file_name in _SOURCE_FILE_ORDER:
            raw_bytes = raw_source_bytes[file_name]
            if type(raw_bytes) is not bytes:
                raise ValueError(
                    f"source_json_bytes[{file_name}] 必須是 exact 原生 bytes"
                )
            source_snapshot[file_name] = bytes(raw_bytes)

        object.__setattr__(self, "metadata", metadata_snapshot)
        object.__setattr__(self, "products", products_snapshot)
        object.__setattr__(
            self,
            "source_json_bytes",
            MappingProxyType(source_snapshot),
        )


def _validate_current_trajectory_manifests(
    *,
    source_run_root: str | Path,
    prepared: _PreparedAggregateRelease,
) -> None:
    """把 prepared 的 shard rows 綁定到目前 run 的 trajectory manifest。

    這是 aggregate release source preparation 與未來 publish 之間的唯讀 output gate；它
    不取得 lock、不寫入檔案、不建立 manifest，也不會修改 ``run_progress.json``。呼叫端
    必須先在外層持有 run gate exclusive lock，因為本函式會把已封存的 prepared plan／progress
    與當下 ``shards/<shard_id>/manifest.json`` 的 bytes 放在同一個 publish 前檢查邊界內。
    ``prepared`` 已由前一階段以 ``validate_run(require_complete=True)`` 建立，但此處仍以
    exact source bytes 重新 strict parse、驗證 plan/progress document 與 cross-check，並
    重建 ``AggregateShardBinding``；這不是用另一套邏輯取代既有 run validator，而是把
    release 產品列序、輸出 token、manifest SHA-256 與三個 trajectory count 封閉綁在一起。

    plan／progress 的 scenario range 是半開區間 ``[start, stop)``，``shard_index`` 是
    aggregate table 原始列序，並非重新排序後的 shard 順序。trajectory manifest 的三個
    count 必須是非 bool 的原生 Python ``int``；``particle_count`` 必須大於零，而
    observation／event count 的 ``>= particle_count`` 限制沿用
    ``AggregateShardBinding`` record constructor 已建立的資料契約。manifest bytes 只以
    現有 no-follow ordinary-file reader 讀取並雜湊；不追隨 manifest 提供的任何路徑，也不
    讀取 trajectory payload 檔案。

    所有解析、schema、cross-check、lstat、普通檔案讀取、SHA-256、row 或 count 失敗都
    統一成固定 ``ValueError("aggregate release trajectory 綁定失敗")``，不讓絕對路徑、
    作業系統錯誤或第三方 parser 文字穿透。成功只代表目前輸出與 synthetic／正式 run
    的工程資料契約一致；本機 synthetic fixture 的通過仍不是真實 OCM schema 3／NWW3
    schema 1 科學成果，也不代表絕對來源機率、因果歸因或觀測驗證。

    Args:
        source_run_root: source run workspace 根目錄；固定要求普通非 symbolic-link 目錄。
        prepared: 前一階段建立的 exact ``_PreparedAggregateRelease`` snapshot。此 helper
            不接受 payload、任意 mapping 或外部 path 來替代 prepared 的 plan／progress／rows。

    Raises:
        ValueError: 任一 trajectory output binding 不符合固定契約；訊息永遠是固定文字，
            且不帶入 source run 的部署路徑。
    """

    try:
        if type(prepared) is not _PreparedAggregateRelease:
            raise ValueError("prepared 型別不符")

        # frozen dataclass 只能防止一般重新賦值；以同一個 private constructor 重建一次，
        # 讓低階 object.__setattr__ 竄改或 caller 仍持有的巢狀 mapping／ndarray alias 不會
        # 直接進入 output binding。這裡不重新編碼產品，因為 prepared 已保存 exact rows。
        prepared_snapshot = _PreparedAggregateRelease(
            metadata=prepared.metadata,
            products=prepared.products,
            source_json_bytes=prepared.source_json_bytes,
        )

        # source bytes 已在 preparation 階段由 no-follow reader 封存；此處只從記憶體 strict
        # parse，不回讀 run_plan／run_progress。duplicate key、NaN／Infinity、1e999 與非
        # object root 都會在同一個固定失敗邊界被拒絕。
        source_documents = {
            file_name: _strict_json_object_from_bytes(
                prepared_snapshot.source_json_bytes[file_name]
            )
            for file_name in (
                "source_run_plan.json",
                "source_run_progress.json",
            )
        }
        plan = validate_run_plan_document(source_documents["source_run_plan.json"])
        progress = validate_run_progress_document(
            source_documents["source_run_progress.json"]
        )
        _cross_check_plan_progress(plan, progress)
        if plan["output_root"] != "shards":
            raise ValueError("run plan output_root 不符合固定 shards root")
        if progress["run_lifecycle"] != "COMPLETE":
            raise ValueError("run progress 必須是 COMPLETE")

        plan_shards = plan["shards"]
        progress_shards = progress["shards"]
        table_rows = prepared_snapshot.products.tables["shard_bindings.parquet"]
        expected_shard_count = plan["shard_count"]
        if len(plan_shards) != expected_shard_count or len(table_rows) != expected_shard_count:
            raise ValueError("shard count 與 prepared table rows 不一致")

        # 每一層目錄都以 lstat 驗證，不能使用 is_dir() 讓 symlink 被跟隨；這個檢查只
        # 固定 source root、shards root 與各 shard output directory，且不掃描或修改其他節點。
        root = Path(source_run_root)

        def require_regular_directory(path: Path) -> None:
            """以 lstat 要求指定節點是普通非 symlink 目錄。"""

            node_status = os.lstat(path)
            if stat.S_ISLNK(node_status.st_mode) or not stat.S_ISDIR(node_status.st_mode):
                raise ValueError("trajectory output directory node 不安全")

        require_regular_directory(root)
        shards_root = root / "shards"
        require_regular_directory(shards_root)

        # encoded shard table 的 rows 已由 codec 固定欄位與順序；這裡再次明示列序和
        # AggregateShardBinding 欄位，避免 output binding 依 shard_id 排序而錯綁另一個
        # scenario 半開區間。AggregateShardBinding constructor 也會重跑路徑、hash、原生
        # integer 與 count invariant，故不以 int()/str() 修補低階竄改。
        expected_row_keys = set(_SHARD_BINDING_COLUMNS)
        for expected_index, (plan_row, table_row) in enumerate(
            zip(plan_shards, table_rows, strict=True)
        ):
            if (
                not isinstance(table_row, Mapping)
                or set(table_row) != expected_row_keys
                or tuple(table_row) != _SHARD_BINDING_COLUMNS
            ):
                raise ValueError("shard binding row schema/order 不一致")
            if type(table_row["shard_index"]) is not int or table_row["shard_index"] != expected_index:
                raise ValueError("shard binding shard_index 不連續")

            binding = AggregateShardBinding(
                shard_id=table_row["shard_id"],
                scenario_start_index=table_row["scenario_start_index"],
                scenario_stop_index=table_row["scenario_stop_index"],
                output_relative_path=table_row["output_relative_path"],
                trajectory_manifest_sha256=table_row["trajectory_manifest_sha256"],
                particle_count=table_row["particle_count"],
                observation_count=table_row["observation_count"],
                event_count=table_row["event_count"],
            )
            shard_id = plan_row["shard_id"]
            progress_row = progress_shards[shard_id]
            expected_output_token = f"shards/{shard_id}"
            if (
                binding.shard_id != shard_id
                or binding.scenario_start_index != plan_row["scenario_start_index"]
                or binding.scenario_stop_index != plan_row["scenario_stop_index"]
                or binding.particle_count != plan_row["particle_count"]
                or binding.output_relative_path != expected_output_token
                or progress_row["lifecycle"] != "COMPLETE"
                or progress_row["output_relative_path"] != expected_output_token
            ):
                raise ValueError("shard binding 與 plan/progress output token 不一致")

            output_directory = shards_root / shard_id
            require_regular_directory(output_directory)
            # _read_regular_file_bytes 以 O_NOFOLLOW 加 fstat 開啟普通檔；不採用 manifest
            # 內的 files mapping，也不讀取 particle table、events 或任何 trajectory array。
            manifest_bytes = _read_regular_file_bytes(output_directory / "manifest.json")
            if sha256(manifest_bytes).hexdigest() != binding.trajectory_manifest_sha256:
                raise ValueError("trajectory manifest SHA-256 不一致")
            manifest = _strict_json_object_from_bytes(manifest_bytes)

            manifest_counts = {}
            for field_name in ("particle_count", "observation_count", "event_count"):
                manifest_value = _require_nonnegative_native_int(
                    manifest.get(field_name),
                    label=f"manifest.{field_name}",
                )
                manifest_counts[field_name] = manifest_value
            if (
                manifest_counts["particle_count"] <= 0
                or manifest_counts["particle_count"] != binding.particle_count
                or manifest_counts["observation_count"] != binding.observation_count
                or manifest_counts["event_count"] != binding.event_count
            ):
                raise ValueError("trajectory manifest count 與 shard binding 不一致")
    except Exception:
        # 固定訊息是此 helper 唯一的公開失敗邊界；特別避免 Path、lstat、JSON、SHA-256
        # 或檔案 reader 的例外帶出 SERVER 絕對位置與環境細節。
        raise ValueError("aggregate release trajectory 綁定失敗") from None


def _write_prepared_release_directory(
    directory: str | Path,
    prepared: _PreparedAggregateRelease,
) -> dict[str, dict[str, object]]:
    """把 prepared snapshot 寫成未發布 partial release 的固定三十三檔拓撲。

    ``directory`` 必須由外層 caller 預先建立，且在本函式開始時是既有、完全空白的普通
    非 symbolic-link 目錄。本 helper 不建立目錄、不取得 run lock、不選 destination、
    不呼叫 ``os.replace``、不發布 final 名稱，也不在失敗時清理任何節點；只有知道該
    partial path 確實由自己建立的未來 public writer，才有權決定是否移除失敗現場。

    函式先以 ``_PreparedAggregateRelease`` constructor 重建 metadata、九表、十八陣列與
    五份 source exact bytes，隔離 caller 的低階 alias。二十七份產品交由既有
    ``_write_encoded_products`` 依固定 Arrow／NumPy 契約以 exclusive create 寫入；其中
    travel age 與停留時間以秒、公尺格線與 boundary 距離以公尺、計數與 offset 以非負
    int64 保存。每個產品完成後都以 no-follow readonly descriptor 執行 ``fsync``，不能
    只因 Python stream 已關閉就假設資料已進入穩定儲存。

    五份 source JSON 依 ``_SOURCE_FILE_ORDER`` 直接複製 prepared 保存的原生 bytes，使用
    ``xb`` 排除覆寫與 symbolic-link 目標競態，逐檔 ``flush``／``fsync`` 後建立
    ``kind=json``、byte size 與 SHA-256 contract；不重新格式化 JSON，也不把缺值、乾點、
    資料缺口或 provenance 改寫成其他值。三十二份 contract 依固定來源、表格、陣列順序
    合併後，manifest root 只含 schema version、``metadata.to_dict()`` 與 files，並以
    ``_canonical_json_bytes`` 在所有 payload 完成後最後 exclusive-create、同步。接著只
    ``fsync`` partial 目錄本身，不同步 parent，因為本階段尚未發布 directory entry。

    任何平台或檔案系統回報的真實 ``fsync`` 失敗都必須使整個 helper fail closed，不能
    吞掉或假裝 durability 已成立。同步完成後再要求 exact 三十三個普通非連結檔案，並
    只呼叫 ``_validate_release_topology_and_checksums(..., require_final_name=False)`` 重驗
    foundation；本階段不執行 source semantic、產品 reader、codec decoder 或 full
    inspection。成功回傳的是與磁碟 manifest 脫鉤的三十二份普通 contract 防禦性副本。

    即使本機 synthetic fixture 能完整寫入與重驗，也只代表未發布 release 的工程 I/O、
    公尺／秒單位與完整性契約成立，不是真實 OCM schema 3／NWW3 schema 1 科學成果，亦
    不能據此宣稱絕對來源機率、因果歸因或觀測驗證。

    Args:
        directory: caller 已新建、既有且完全空白的 partial release 普通目錄。
        prepared: exact ``_PreparedAggregateRelease``；其 source bytes 不會重新序列化。

    Returns:
        依固定三十二檔順序建立的普通 ``dict`` contracts 防禦性副本。

    Raises:
        ValueError: 任一目錄、prepared、exclusive write、durability、manifest 或 foundation
            驗證失敗；訊息固定為 ``aggregate release partial 寫入失敗`` 且不含 path。
    """

    try:
        if type(prepared) is not _PreparedAggregateRelease:
            raise ValueError("prepared 型別不符")

        # 先重建 private snapshot，讓 products 的列 mapping／NumPy buffer、metadata 與
        # source bytes 都不直接沿用 caller reference；本階段不重算聚合或 trajectory。
        prepared_snapshot = _PreparedAggregateRelease(
            metadata=prepared.metadata,
            products=prepared.products,
            source_json_bytes=prepared.source_json_bytes,
        )

        output_directory = Path(directory)
        directory_status = os.lstat(output_directory)
        if stat.S_ISLNK(directory_status.st_mode) or not stat.S_ISDIR(
            directory_status.st_mode
        ):
            raise ValueError("partial directory 必須是普通非 symlink 目錄")
        with os.scandir(output_directory) as entries:
            if next(entries, None) is not None:
                raise ValueError("partial directory 必須完全空白")

        def fsync_regular_file(path: Path) -> None:
            """以 no-follow readonly descriptor 同步一份既有普通檔。"""

            descriptor = _open_regular_file_descriptor(path)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

        def write_exclusive_durable_bytes(path: Path, raw_bytes: bytes) -> None:
            """以 ``xb`` 寫 exact bytes，確認普通檔後 flush 並同步 descriptor。"""

            if type(raw_bytes) is not bytes:
                raise ValueError("durable bytes 必須是 exact 原生 bytes")
            # O_CREAT|O_EXCL 的 xb 語意不會追隨已存在的 final-component symlink；fstat
            # 再確認實際 descriptor 是普通檔，避免特殊節點進入 partial release。
            with path.open("xb") as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise ValueError("exclusive target 必須是普通檔")
                written = stream.write(raw_bytes)
                if written != len(raw_bytes):
                    raise OSError("exclusive durable write 未完成")
                stream.flush()
                os.fsync(stream.fileno())

        # 產品 writer 已使用固定名稱與 xb，不建立 manifest。它回傳後，逐一以固定順序
        # no-follow 開啟並 fsync；任一同步失敗都保留現場、由外層 future writer 處理。
        product_contracts = _write_encoded_products(
            output_directory,
            prepared_snapshot.products,
        )
        if set(product_contracts) != set((*_TABLE_FILE_ORDER, *_ARRAY_FILE_ORDER)):
            raise ValueError("產品 contracts 固定集合不一致")
        for file_name in (*_TABLE_FILE_ORDER, *_ARRAY_FILE_ORDER):
            fsync_regular_file(output_directory / file_name)

        source_contracts: dict[str, dict[str, object]] = {}
        for file_name in _SOURCE_FILE_ORDER:
            source_bytes = prepared_snapshot.source_json_bytes[file_name]
            write_exclusive_durable_bytes(output_directory / file_name, source_bytes)
            source_contracts[file_name] = {
                "kind": "json",
                "size_bytes": len(source_bytes),
                "sha256": sha256(source_bytes).hexdigest(),
            }

        # 固定順序與 exact key set 在建立 manifest 前明示；深拷貝 product contract 可隔離
        # Arrow fields／NPY shape 的巢狀 list，避免後續 manifest 與 helper 暫存 mapping 共用。
        unordered_contracts: dict[str, dict[str, object]] = {
            **source_contracts,
            **product_contracts,
        }
        fixed_file_order = (*_SOURCE_FILE_ORDER, *_TABLE_FILE_ORDER, *_ARRAY_FILE_ORDER)
        if set(unordered_contracts) != set(fixed_file_order) or len(unordered_contracts) != 32:
            raise ValueError("aggregate release 必須恰有三十二份 payload contract")
        contracts = {
            file_name: deepcopy(unordered_contracts[file_name])
            for file_name in fixed_file_order
        }

        manifest_document = {
            "schema_version": prepared_snapshot.metadata.schema_version,
            "metadata": prepared_snapshot.metadata.to_dict(),
            "files": deepcopy(contracts),
        }
        manifest_bytes = _canonical_json_bytes(manifest_document)
        # manifest 必須在三十二份 payload 全部同步後才建立；其 fsync 成功仍不足以發布，
        # 還要同步 partial 目錄 entry，且刻意不碰尚未 os.replace 的 parent directory。
        write_exclusive_durable_bytes(
            output_directory / _MANIFEST_FILE_NAME,
            manifest_bytes,
        )

        directory_flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            directory_flags |= os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            directory_flags |= os.O_NOFOLLOW
        directory_descriptor = os.open(output_directory, directory_flags)
        try:
            if not stat.S_ISDIR(os.fstat(directory_descriptor).st_mode):
                raise ValueError("partial directory descriptor 型別不符")
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)

        # foundation 會再次驗 exact 33-node topology、canonical manifest 與全部 32 檔
        # size/hash；require_final_name=False 只放寬 partial basename，不放寬任何內容契約。
        validated_metadata, validated_contracts, validated_source_bytes = (
            _validate_release_topology_and_checksums(
                output_directory,
                require_final_name=False,
            )
        )
        if (
            validated_metadata != prepared_snapshot.metadata
            or validated_contracts != contracts
            or dict(validated_source_bytes) != dict(prepared_snapshot.source_json_bytes)
        ):
            raise ValueError("partial foundation 回讀結果不一致")
        return {
            file_name: deepcopy(validated_contracts[file_name])
            for file_name in fixed_file_order
        }
    except Exception:
        # 本 helper 絕不清理：即使已留下部分檔案，也只固定化錯誤，不刪除 caller 可能
        # 擁有的 directory 或診斷現場；public writer 後續必須依 ownership 決定清理範圍。
        raise ValueError("aggregate release partial 寫入失敗") from None


def _prepare_aggregate_release_source(
    *,
    source_run_root: str | Path,
    aggregate_spec_path: str | Path,
    payload: AggregateReleasePayload,
    checkpoint_root: str | Path | None = None,
) -> _PreparedAggregateRelease:
    """準備 aggregate writer 所需的來源快照與基本跨文件 binding。

    本 helper 是 writer 的來源準備階段，不建立目錄、不寫檔、不建立 manifest、不執行
    ``os.replace``，也不取得 lock。正式 writer 的外層必須先持有 source run 的
    ``run_gate`` exclusive lock，確保本函式先呼叫 ``validate_run``、再讀 exact bytes、
    再重建 typed snapshot 的期間，immutable plan、progress、輸入檔與 trajectory output
    不會被另一個 worker 變更。這裡只保存五份 source JSON bytes，不保存任何 Path 或
    SERVER 絕對位置。

    執行順序先以 ``metadata_from_payload`` 與 ``encode_aggregate_release_payload``
    重新驗證 exact payload，隔離 caller 對 frozen／巢狀物件的低階竄改；接著以既有
    ``validate_run(require_complete=True)`` 作為完整 run gate，再用
    ``load_run_plan``／``load_run_progress`` 與 document validators 重驗 immutable plan
    和 progress。五份來源檔案都透過本模組的 no-follow ordinary-file reader 取得 exact
    bytes；AggregateSpec 的原始 bytes hash、canonical hash、完整 ``to_dict`` 與 run ID，
    以及四份 run source bytes 的 SHA-256，均必須和 payload／metadata 一致。

    plan 的 run identity、run kind、experiment case、members-per-scenario、config hash、
    checkpoint input binding、particle／scenario／shard counts，及 normalized config／input
    inventory 的 size/hash contract 也必須逐欄相符。normalized config 另以 compact、
    sorted、UTF-8 JSON（不含尾端換行）重算 config hash；input inventory 只在此階段要求
    strict JSON object，不臆測正式 OCM／NWW inventory 欄位。payload 的 shard bindings
    只與 plan 原順序及 COMPLETE progress 的 ID、半開 scenario range、particle count、
    output token 基本核對；本函式不另讀、不 hash、不解析目前 shard 的 trajectory
    ``manifest.json``，該責任由後續獨立的 shard-output binding helper 處理。既有
    ``validate_run`` 本身若執行其完整唯讀 run gate，仍屬既有 validator 行為，不在此
    重複或宣稱已完成發布前的 output-manifest binding。

    所有來源、型別、檔案系統、JSON、hash、plan/progress 或 payload binding 失敗，均由
    最外層轉成固定 ``ValueError("aggregate release source 準備失敗")``，不攜帶任何
    absolute path、SERVER 路徑、作業系統訊息或第三方例外。成功回傳的資料仍只是
    條件式來源足跡／相對來源權重發布的工程輸入，不是絕對來源機率、因果歸因或真實
    OCM／NWW 科學成果。

    Args:
        source_run_root: 已存在的 source run workspace 普通非 symlink 目錄。
        aggregate_spec_path: caller 明示的 AggregateSpec 普通非 symlink JSON 檔案；其
            exact bytes 會以固定 ``aggregate_spec.json`` source 名稱保存。
        payload: 待發布的 exact ``AggregateReleasePayload``；函式會重新 encode／建 metadata。
        checkpoint_root: optional external checkpoint root，原樣傳給唯讀 ``validate_run``，
            不會保存於回傳容器。

    Returns:
        已完成 source JSON／plan／progress／spec／payload 基本 binding 的 private snapshot。

    Raises:
        ValueError: 任一準備或 binding 檢查失敗；對外訊息固定且不含路徑。
    """

    try:
        # 兩個公開 codec 入口各自重建 payload；metadata 與產品不直接信任 caller 保留的
        # frozen reference，也不以零值、最近值或轉型方式修補缺少的事件／路徑資料。
        if type(payload) is not AggregateReleasePayload:
            raise ValueError("payload 必須是 exact AggregateReleasePayload 實例")
        metadata = metadata_from_payload(payload)
        products = encode_aggregate_release_payload(payload)
        if type(metadata) is not AggregateReleaseMetadata:
            raise ValueError("metadata_from_payload 回傳型別不符")
        if type(products) is not EncodedAggregateProducts:
            raise ValueError("encode_aggregate_release_payload 回傳型別不符")

        # source run root 與 spec path 先以 lstat 驗證；實際 bytes 讀取稍後仍會使用
        # O_NOFOLLOW/fstat 的 helper，避免只依賴一次 path-level exists/is_file 檢查。
        root = Path(source_run_root)
        root_status = os.lstat(root)
        if stat.S_ISLNK(root_status.st_mode) or not stat.S_ISDIR(root_status.st_mode):
            raise ValueError("source run root 必須是普通非 symlink 目錄")
        spec_path = Path(aggregate_spec_path)
        spec_status = os.lstat(spec_path)
        if stat.S_ISLNK(spec_status.st_mode) or not stat.S_ISREG(spec_status.st_mode):
            raise ValueError("aggregate spec 必須是普通非 symlink 檔案")

        # 這是既有 run 的唯讀完整性 gate；不 reconcile、不更新 latest、不修改 progress，
        # external checkpoint root 只在本次驗證使用，絕不進入 Prepared snapshot。
        run_validation = validate_run(
            root,
            require_complete=True,
            checkpoint_root=checkpoint_root,
        )
        if not isinstance(run_validation, Mapping) or run_validation.get("valid") is not True:
            raise ValueError("source run 未通過 require_complete validation")

        # 先依既有 loader 重驗 plan/progress，再以 exact bytes 的 strict parser 重驗一次；
        # 前者保留 run-control 的正式 schema 邏輯，後者拒絕 duplicate key、NaN、Infinity
        # 與 1e999，且確保 Prepared 保存的 bytes 與 validator 看到的是同一份文件。
        loaded_plan = load_run_plan(root)
        loaded_progress = load_run_progress(root)
        _cross_check_plan_progress(loaded_plan, loaded_progress)

        source_json_bytes: dict[str, bytes] = {
            "aggregate_spec.json": _read_regular_file_bytes(spec_path),
            "source_run_plan.json": _read_regular_file_bytes(root / "run_plan.json"),
            "source_run_progress.json": _read_regular_file_bytes(root / "run_progress.json"),
            "source_normalized_config.json": _read_regular_file_bytes(
                root / "normalized_config.json"
            ),
            "source_input_inventory.json": _read_regular_file_bytes(
                root / "input_inventory.json"
            ),
        }
        source_documents = {
            file_name: _strict_json_object_from_bytes(source_json_bytes[file_name])
            for file_name in _SOURCE_FILE_ORDER
        }
        plan = validate_run_plan_document(source_documents["source_run_plan.json"])
        progress = validate_run_progress_document(
            source_documents["source_run_progress.json"]
        )
        _cross_check_plan_progress(plan, progress)
        if plan != loaded_plan or progress != loaded_progress:
            raise ValueError("loader 與 exact source bytes 不一致")

        if progress["run_lifecycle"] != "COMPLETE":
            raise ValueError("run progress 必須是 COMPLETE")
        progress_rows = progress["shards"]
        if not isinstance(progress_rows, dict) or not progress_rows:
            raise ValueError("progress.shards 必須是非空 object")
        for plan_row in plan["shards"]:
            progress_row = progress_rows[plan_row["shard_id"]]
            if progress_row["lifecycle"] != "COMPLETE":
                raise ValueError("所有 progress shard 都必須是 COMPLETE")

        # AggregateSpec loader 會依自己的嚴格 schema 重建物件；其再次讀取仍受外層
        # run_gate 保護，而 exact source bytes hash 可偵測讀取期間的 spec 替換。
        aggregate_spec = load_aggregate_spec(spec_path)
        spec_bytes = source_json_bytes["aggregate_spec.json"]
        spec_source_sha256 = sha256(spec_bytes).hexdigest()
        if aggregate_spec.source_sha256 != spec_source_sha256:
            raise ValueError("AggregateSpec source hash 與 exact bytes 不一致")
        if aggregate_spec.source_sha256 != metadata.aggregate_spec_source_sha256:
            raise ValueError("AggregateSpec source hash 與 metadata 不一致")
        if aggregate_spec.canonical_sha256 != metadata.aggregate_spec_canonical_sha256:
            raise ValueError("AggregateSpec canonical hash 與 metadata 不一致")
        if aggregate_spec.run_id != payload.run_id:
            raise ValueError("AggregateSpec run_id 與 payload 不一致")
        if aggregate_spec.to_dict() != payload.aggregate_spec.to_dict():
            raise ValueError("AggregateSpec to_dict 與 payload 不一致")

        # 四份 run source bytes 的 digest 是 payload 建立時保存的 provenance；metadata 是
        # 同一份 payload 重新推導的獨立 typed view，兩者都要與目前 exact bytes 對上。
        source_hash_bindings = (
            (
                "source_run_plan.json",
                payload.source_run_plan_sha256,
                metadata.source_run_plan_sha256,
            ),
            (
                "source_run_progress.json",
                payload.source_run_progress_sha256,
                metadata.source_run_progress_sha256,
            ),
            (
                "source_normalized_config.json",
                payload.source_normalized_config_sha256,
                metadata.source_normalized_config_sha256,
            ),
            (
                "source_input_inventory.json",
                payload.source_input_inventory_sha256,
                metadata.source_input_inventory_sha256,
            ),
        )
        for file_name, payload_digest, metadata_digest in source_hash_bindings:
            actual_digest = sha256(source_json_bytes[file_name]).hexdigest()
            if actual_digest != payload_digest or actual_digest != metadata_digest:
                raise ValueError("source JSON hash 與 payload/metadata 不一致")

        # plan 的 identity、設定與計數直接對照 payload 與 metadata；particle count 來自
        # event aggregate 的完整輸入粒子數，scenario／shard count 由 canonical products
        # metadata 保存，不能從 output manifest 或 event count 猜測替代。
        plan_identity_bindings = (
            ("run_id", payload.run_id, metadata.run_id),
            ("run_kind", payload.run_kind, metadata.run_kind),
            ("experiment_case_id", payload.experiment_case_id, metadata.experiment_case_id),
            ("members_per_scenario", payload.members_per_scenario, metadata.members_per_scenario),
            ("config_hash", payload.config_hash, metadata.config_hash),
            (
                "checkpoint_input_binding_hash",
                payload.checkpoint_input_binding_hash,
                metadata.checkpoint_input_binding_hash,
            ),
            (
                "particle_count",
                payload.event_aggregate.input_particle_count,
                metadata.input_particle_count,
            ),
            ("scenario_count", len(payload.scenario_strata), metadata.scenario_row_count),
            ("shard_count", len(payload.shard_bindings), metadata.shard_row_count),
        )
        for field_name, payload_value, metadata_value in plan_identity_bindings:
            if plan[field_name] != payload_value or plan[field_name] != metadata_value:
                raise ValueError(f"run plan {field_name} binding 不一致")

        # plan files contract 保存的是 source run 中 immutable input 的原始 bytes；release
        # 會把它們改用 source_ 前綴保存，但 size/hash 必須完全相同，不能以 canonical
        # config hash 或 inventory 內容摘要取代 raw bytes provenance。
        plan_files = plan["files"]
        file_bindings = (
            ("normalized_config.json", "source_normalized_config.json"),
            ("input_inventory.json", "source_input_inventory.json"),
        )
        for plan_file_name, source_file_name in file_bindings:
            plan_contract = plan_files[plan_file_name]
            raw_bytes = source_json_bytes[source_file_name]
            if (
                plan_contract["size_bytes"] != len(raw_bytes)
                or plan_contract["sha256"] != sha256(raw_bytes).hexdigest()
            ):
                raise ValueError("run plan input file contract 與 exact source bytes 不一致")

        # config hash 的資料契約是 JSON 語意 canonical bytes，不附加 release／run JSON 常用
        # 的尾端換行；strict parser 已先拒絕 duplicate key、非有限值與非 object root。
        normalized_config = source_documents["source_normalized_config.json"]
        canonical_config_bytes = json.dumps(
            normalized_config,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        canonical_config_hash = sha256(canonical_config_bytes).hexdigest()
        if (
            canonical_config_hash != plan["config_hash"]
            or canonical_config_hash != payload.config_hash
            or canonical_config_hash != metadata.config_hash
        ):
            raise ValueError("normalized config canonical hash 不一致")

        # input inventory 已由 strict parser 確認為普通 object；此階段只保存 exact bytes
        # 與 plan file contract，不把尚未驗收的 OCM/NWW 欄位自行解讀成科學 provenance。
        if type(source_documents["source_input_inventory.json"]) is not dict:
            raise ValueError("input inventory 必須是 strict JSON object")

        # 只核對 payload shard binding 的基本 identity/range/output token。這裡不開啟
        # shards/<id>/manifest.json，也不重新計算 trajectory count；後續獨立 output-binding
        # 階段必須在 release publish 前補上當下 manifest 的 safe hash 與三個 count gate。
        if len(payload.shard_bindings) != len(plan["shards"]):
            raise ValueError("payload shard binding count 與 run plan 不一致")
        for payload_binding, plan_row in zip(
            payload.shard_bindings,
            plan["shards"],
            strict=True,
        ):
            progress_row = progress_rows[plan_row["shard_id"]]
            if (
                payload_binding.shard_id != plan_row["shard_id"]
                or payload_binding.scenario_start_index
                != plan_row["scenario_start_index"]
                or payload_binding.scenario_stop_index
                != plan_row["scenario_stop_index"]
                or payload_binding.particle_count != plan_row["particle_count"]
                or payload_binding.output_relative_path
                != progress_row["output_relative_path"]
            ):
                raise ValueError("payload shard binding 與 plan/progress 不一致")

        return _PreparedAggregateRelease(
            metadata=metadata,
            products=products,
            source_json_bytes=source_json_bytes,
        )
    except Exception:
        # source preparation 的所有錯誤都必須固定化；尤其不能讓 Path、SERVER root、
        # OSError、JSON parser 或 third-party codec 例外文字穿透到未來公開 writer CLI。
        raise ValueError("aggregate release source 準備失敗") from None


def _validate_source_documents(
    directory: str | Path,
    *,
    metadata: AggregateReleaseMetadata,
    contracts: Mapping[str, Mapping[str, object]],
    source_json_bytes: Mapping[str, bytes],
) -> _SourceSnapshot:
    """驗證五份 source JSON 並建立與 release metadata 綁定的深層快照。

    入口只接受 foundation 已完成 topology、manifest 與全部 payload checksum 的結果。
    為封閉 foundation checksum 後重新讀 source bytes 所留下的競態窗口，本 helper
    先逐份對記憶體中的 exact bytes 計算大小與 SHA-256，再和 manifest contract 精確
    比對；任何 source JSON 尚未通過這個 re-hash gate 前都不進入語意解析。五份文件
    接著共用嚴格 UTF-8／duplicate-key／有限數值／object-root parser；其中
    ``aggregate_spec.json`` 額外以 ``load_aggregate_spec`` 從固定 release root 重建
    完整 ``AggregateSpec``，並要求第二次讀取的 source hash 與 foundation 已讀 bytes
    完全相同，避免只接受競態期間被替換的另一份 spec。

    ``run_plan`` 與 ``run_progress`` 使用 run-control 的既有文件驗證與 cross-check，
    再要求整個 run 及每個 shard 都是 ``COMPLETE``，且每個完成 shard 都有 output token、
    沒有 error 或 failure token。metadata 的四個 source digest、兩個 AggregateSpec
    digest、run identity、run kind、experiment case、members、設定／checkpoint binding
    與粒子／情境／shard counts 都必須逐欄等於 source 文件；plan 另須把其 immutable
    normalized-config／input-inventory file contract 綁到 release 內兩份 source bytes。
    normalized config 的 config hash 使用 compact sorted UTF-8 JSON 計算，刻意不加入
    manifest canonical JSON 的尾端換行。input inventory 只要求 strict object 與 plan
    file binding，不假設或捏造任何正式 inventory 欄位。

    所有失敗（包含底層 parser、檔案讀取、mapping、hash 或 unexpected exception）都
    統一轉成只帶固定 stage code 的 ``_ReleaseValidationError("source")``；不把 path、
    作業系統文字或第三方例外內容帶出，也不在失敗後繼續解析或產生部分快照。這個
    helper 的成功只代表 release source provenance 與工程資料契約一致，並不代表
    本機合成資料已產生真實 OCM／NWW 科學結果。

    Args:
        directory: foundation 已驗證的 release 根目錄；只用於載入固定 spec 檔名。
        metadata: 已由 manifest foundation 建立的 typed release metadata。
        contracts: manifest foundation 驗證後的三十二份 file contracts。
        source_json_bytes: foundation 在全部 checksum 通過後讀取的五份 exact bytes。

    Returns:
        保存 AggregateSpec、四份已驗證 source 文件與五份 source digest 的 private
        defensive snapshot。

    Raises:
        _ReleaseValidationError: 任一來源位元組、語意、provenance 或 cross-document
            binding 不符合固定契約時，stage 永遠是 ``"source"``。
    """

    try:
        if type(metadata) is not AggregateReleaseMetadata:
            raise ValueError("metadata 型別不符")
        if not isinstance(contracts, Mapping) or not isinstance(source_json_bytes, Mapping):
            raise ValueError("contracts/source_json_bytes 必須是 mapping")
        if set(source_json_bytes) != set(_SOURCE_FILE_ORDER):
            raise ValueError("source_json_bytes 必須恰好包含五份 source JSON")

        # 先重算 foundation 已取得的五份 exact bytes；這裡不能直接信任 manifest
        # contract 或 foundation 回傳 mapping，因為 source bytes 可能在兩階段之間被
        # 替換，且 semantic parser 不應在 checksum gate 之前觀察其內容。
        source_digests: dict[str, str] = {}
        for file_name in _SOURCE_FILE_ORDER:
            raw_bytes = source_json_bytes[file_name]
            if type(raw_bytes) is not bytes:
                raise ValueError("source bytes 型別不符")
            contract = contracts[file_name]
            digest = sha256(raw_bytes).hexdigest()
            if (
                len(raw_bytes) != contract["size_bytes"]
                or digest != contract["sha256"]
            ):
                raise ValueError("source bytes 與 manifest contract 不一致")
            source_digests[file_name] = digest

        # 五份來源都必須經過同一個 strict JSON boundary。這裡不將 inventory 的內容
        # 解讀成特定正式欄位；strict object 只保證其 bytes 可安全保存與綁定。
        source_documents = {
            file_name: _strict_json_object_from_bytes(source_json_bytes[file_name])
            for file_name in _SOURCE_FILE_ORDER
        }

        # foundation 已用 checksum 讀過 aggregate_spec；此處依固定檔名重新透過正式
        # loader 建立完整 AggregateSpec，並以原先 exact bytes 的 hash 封閉兩次讀取間
        # 的 TOCTOU 窗口。不能用 strict parser 的普通 dict 取代完整 spec constructor，
        # 因為公尺制格網、邊界段、秒制 age 軸與 canonical hash 都在 loader 內驗證。
        root = Path(directory)
        aggregate_spec = load_aggregate_spec(root / "aggregate_spec.json")
        if type(aggregate_spec) is not AggregateSpec:
            raise ValueError("aggregate_spec loader 回傳型別不符")
        if aggregate_spec.source_sha256 != source_digests["aggregate_spec.json"]:
            raise ValueError("aggregate_spec source hash 與 exact bytes 不一致")
        if aggregate_spec.source_sha256 != metadata.aggregate_spec_source_sha256:
            raise ValueError("aggregate_spec source hash 與 metadata 不一致")
        if aggregate_spec.canonical_sha256 != metadata.aggregate_spec_canonical_sha256:
            raise ValueError("aggregate_spec canonical hash 與 metadata 不一致")
        if aggregate_spec.run_id != metadata.run_id:
            raise ValueError("aggregate_spec.run_id 與 metadata 不一致")

        # metadata 的四個 source digest 分別綁定 release 保存的 plan、progress、設定
        # 與 inventory exact bytes；這些 digest 不由內容重新推導替代，也不接受近似值。
        metadata_source_fields = (
            ("source_run_plan_sha256", "source_run_plan.json"),
            ("source_run_progress_sha256", "source_run_progress.json"),
            ("source_normalized_config_sha256", "source_normalized_config.json"),
            ("source_input_inventory_sha256", "source_input_inventory.json"),
        )
        for metadata_field, file_name in metadata_source_fields:
            if getattr(metadata, metadata_field) != source_digests[file_name]:
                raise ValueError("metadata source hash 與 exact source bytes 不一致")

        plan = validate_run_plan_document(source_documents["source_run_plan.json"])
        progress = validate_run_progress_document(
            source_documents["source_run_progress.json"]
        )
        _cross_check_plan_progress(plan, progress)

        # release 只能發布已完成的 immutable run；COMPLETE shard 的 output token 是
        # 後續 decoder 綁定 trajectory shard 的唯一相對路徑，failure/error 必須保持
        # None，不能把 FAILED、PAUSED 或缺 output 的狀態包裝成完成成果。
        if progress["run_lifecycle"] != "COMPLETE":
            raise ValueError("run progress 必須是 COMPLETE")
        progress_shards = progress["shards"]
        if not isinstance(progress_shards, dict) or not progress_shards:
            raise ValueError("progress.shards 必須是非空 object")
        for shard_id, progress_row in progress_shards.items():
            if (
                not isinstance(progress_row, dict)
                or progress_row["lifecycle"] != "COMPLETE"
                or progress_row["output_relative_path"] is None
                or progress_row["error_code"] is not None
                or progress_row["failure_relative_path"] is not None
            ):
                raise ValueError(f"progress shard {shard_id!r} 未完成")

        # 這些欄位是 release metadata 和 immutable run plan 的 identity／計數 binding。
        # particle_count 是 scenario_count×M 的 plan 值；release 的 scenario row count
        # 與 shard row count 則必須各自精確對應固定產品拓撲，不能以 event count 代替。
        plan_metadata_bindings = (
            ("run_id", metadata.run_id),
            ("run_kind", metadata.run_kind),
            ("experiment_case_id", metadata.experiment_case_id),
            ("members_per_scenario", metadata.members_per_scenario),
            ("config_hash", metadata.config_hash),
            ("checkpoint_input_binding_hash", metadata.checkpoint_input_binding_hash),
            ("particle_count", metadata.input_particle_count),
            ("scenario_count", metadata.scenario_row_count),
            ("shard_count", metadata.shard_row_count),
        )
        for field_name, expected_value in plan_metadata_bindings:
            if plan[field_name] != expected_value:
                raise ValueError("run plan 與 metadata binding 不一致")

        # run plan 的 file contract 描述 workspace 中的 normalized_config／input_inventory
        # 實體檔案；release 以 source_ 前綴保存其 exact bytes，所以兩者必須逐一 size/hash
        # 相等。此處不要求 inventory 有特定欄位，避免 synthetic 與正式 inventory schema
        # 被錯誤地混為同一層契約。
        plan_files = plan["files"]
        if not isinstance(plan_files, dict):
            raise ValueError("run plan files 必須是 object")
        plan_file_bindings = (
            ("normalized_config.json", "source_normalized_config.json"),
            ("input_inventory.json", "source_input_inventory.json"),
        )
        for plan_file_name, source_file_name in plan_file_bindings:
            plan_contract = plan_files[plan_file_name]
            if (
                plan_contract["size_bytes"] != len(source_json_bytes[source_file_name])
                or plan_contract["sha256"] != source_digests[source_file_name]
            ):
                raise ValueError("run plan file contract 與 release source 不一致")

        # config_hash 是 normalized config JSON 語意的 compact canonical hash；與 manifest
        # canonical bytes 不同，這裡刻意不附加尾端 newline，確保格式化差異不會被誤認成
        # 同一設定，也不把 aggregate manifest 的外層欄位混入 run config provenance。
        normalized_config = source_documents["source_normalized_config.json"]
        canonical_config_bytes = json.dumps(
            normalized_config,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        canonical_config_hash = sha256(canonical_config_bytes).hexdigest()
        if (
            canonical_config_hash != plan["config_hash"]
            or canonical_config_hash != metadata.config_hash
        ):
            raise ValueError("normalized config canonical hash 不一致")

        return _SourceSnapshot(
            aggregate_spec=aggregate_spec,
            plan=plan,
            progress=progress,
            normalized_config=source_documents["source_normalized_config.json"],
            input_inventory=source_documents["source_input_inventory.json"],
            source_sha256_by_file=source_digests,
        )
    except _ReleaseValidationError:
        raise
    except Exception:
        # 來源語意層對外只允許固定 stage code；不能把 path、JSON parser、底層 OS 或
        # 自訂 Mapping 的錯誤文字帶出，也不能在部分驗證成功後繼續建造 snapshot。
        raise _ReleaseValidationError("source") from None


def _validate_product_contracts(
    root: str | Path,
    contracts: Mapping[str, Mapping[str, object]],
) -> EncodedAggregateProducts:
    """讀取固定產品並逐檔核對 manifest 宣告的數量、型別與形狀。

    這個 helper 必須位於 topology、manifest 與三十二檔 checksum gate 之後；它只呼叫
    既有的 ``_read_encoded_products`` 取得九張 Parquet 表與十八個 NumPy 陣列，再將
    實際記憶體產品和 manifest contract 逐檔精確比對。Parquet 的 ``row_count`` 是表格
    列數，NPY 的 ``dtype``、一維 ``shape`` 與 ``element_count`` 是容器契約；它們不
    代表事件物理量，也不以轉型、reshape、補零或最近值修補不一致內容。

    ``root`` 只應是 release 根目錄；本 helper 不接受 manifest 注入的任意檔名。所有
    reader、contract、mapping、NumPy 或未預期錯誤都收斂成固定 ``products`` stage，
    因而不把作業系統訊息、第三方例外或 SERVER／本機絕對路徑暴露給上層。成功只表示
    實體產品與其 manifest 宣告一致，不代表本機 synthetic 測試已產生真實 OCM／NWW
    科學成果。

    Args:
        root: 已通過固定 topology 與 checksum 驗證的 release 根目錄。
        contracts: foundation 解析出的三十二份 manifest file contracts。

    Returns:
        已由既有產品 reader 建立、並通過所有 manifest product contracts 的
        ``EncodedAggregateProducts``。

    Raises:
        _ReleaseValidationError: 任一產品讀取或 contract 驗證失敗，stage 固定為
            ``"products"``。
    """

    try:
        products = _read_encoded_products(root)

        # 表格的列數必須直接與 manifest 宣告相等；不能把缺列當成零事件列，也不能以
        # 其他表格的數量推導替代值。九張表的欄位與 scalar 已由既有 reader 驗證。
        for file_name in _TABLE_FILE_ORDER:
            contract = contracts[file_name]
            if len(products.tables[file_name]) != contract["row_count"]:
                raise ValueError("Parquet row_count 與 manifest contract 不一致")

        # 陣列的 dtype.str、shape 與元素數是三個獨立的儲存邊界，全部必須精確相等。
        # int(array.size) 明確把 NumPy scalar 轉成 Python int 後再比較，但不改變陣列內容。
        for file_name in _ARRAY_FILE_ORDER:
            contract = contracts[file_name]
            values = products.arrays[file_name]
            if values.dtype.str != contract["dtype"]:
                raise ValueError("NPY dtype 與 manifest contract 不一致")
            if list(values.shape) != contract["shape"]:
                raise ValueError("NPY shape 與 manifest contract 不一致")
            if int(values.size) != contract["element_count"]:
                raise ValueError("NPY element_count 與 manifest contract 不一致")
        return products
    except Exception:
        # 對外只保留固定 stage；不可讓底層 reader 的 path、Arrow／NumPy 文字或自訂
        # mapping 例外穿透，避免 validator 契約因環境差異而洩漏敏感部署資訊。
        raise _ReleaseValidationError("products") from None


def _decode_and_bind_release(
    metadata: AggregateReleaseMetadata,
    source_snapshot: _SourceSnapshot,
    products: EncodedAggregateProducts,
) -> AggregateReleasePayload:
    """解碼產品並將 shard／scenario 拓撲綁回原始 run plan 與完成進度。

    decoder 本身負責所有九表、十八陣列、AggregateSpec、payload constructor 與
    encode-back exact equality；本 helper 不重複另一套 codec 邏輯，只在 decoder 成功後
    以 source snapshot 的 immutable plan 原始順序逐列核對四個 shard identity／range／
    particle 欄位，並核對同一 shard 的 ``COMPLETE`` progress output token。plan 的
    scenario table 並未封存在 aggregate release，因此這裡只要求 decoded
    ``scenario_strata`` 數量精確等於 plan 的 ``scenario_count``；scenario ID 的唯一性、
    shard 半開區間連續性及其可覆蓋範圍，交由既有 decoder／payload constructor 已證明的
    契約，不捏造 release 沒有的 scenario table 來做額外比對。

    所有 decoder、plan／progress mapping、zip strict 或 cross-binding 錯誤都轉成固定
    ``decoder`` stage，不將路徑或第三方例外文字帶出。輸出仍只是可供工程流程保存的
    條件式來源足跡／相對來源權重 payload；本機 synthetic input 的通過不等於真實
    OCM／NWW 科學結果或觀測驗證。

    Args:
        metadata: 已通過 manifest foundation 的 typed release metadata。
        source_snapshot: 已通過 source semantic validation 的 plan、progress 與
            AggregateSpec 快照。
        products: 已通過固定 product contracts 的九表／十八陣列產品。

    Returns:
        通過 codec decoder 與 run-level shard/scenario binding 的
        ``AggregateReleasePayload``。

    Raises:
        _ReleaseValidationError: 任一解碼或跨文件 binding 失敗，stage 固定為
            ``"decoder"``。
    """

    try:
        decoded = decode_aggregate_release_payload(
            metadata,
            source_snapshot.aggregate_spec,
            products,
        )

        plan = source_snapshot.plan
        plan_shards = plan["shards"]
        expected_shard_count = plan["shard_count"]
        if len(plan_shards) != expected_shard_count:
            raise ValueError("plan shard count 與 shard rows 不一致")
        if len(decoded.shard_bindings) != expected_shard_count:
            raise ValueError("decoded shard binding count 與 plan 不一致")

        progress_shards = source_snapshot.progress["shards"]
        # plan_shards 是 run controller 發布的 immutable 原始順序；不可依 shard_id 排序，
        # 因為 scenario range 與 execution group 的語意就是由此順序決定。strict=True
        # 讓任何數量漂移都在此邊界明確失敗。
        for plan_row, decoded_shard in zip(
            plan_shards,
            decoded.shard_bindings,
            strict=True,
        ):
            shard_id = plan_row["shard_id"]
            progress_row = progress_shards[shard_id]
            if progress_row["lifecycle"] != "COMPLETE":
                raise ValueError("shard progress 必須是 COMPLETE")
            if (
                decoded_shard.shard_id != shard_id
                or decoded_shard.scenario_start_index
                != plan_row["scenario_start_index"]
                or decoded_shard.scenario_stop_index
                != plan_row["scenario_stop_index"]
                or decoded_shard.particle_count != plan_row["particle_count"]
                or decoded_shard.output_relative_path
                != progress_row["output_relative_path"]
            ):
                raise ValueError("decoded shard binding 與 plan/progress 不一致")

        if len(decoded.scenario_strata) != plan["scenario_count"]:
            raise ValueError("decoded scenario strata count 與 plan 不一致")
        return decoded
    except Exception:
        # decoder 與跨 binding 共用同一個公開 stage；避免把 plan key error、路徑或底層
        # codec 例外傳到公開 validator，也避免錯誤發生後繼續回傳部分 payload。
        raise _ReleaseValidationError("decoder") from None


# 公開 validator 的 summary 是固定十二欄純量契約。欄位順序用於產生可重現的普通 dict，
# 但驗證輸入時只要求 exact key set，再依這個 tuple 重排；如此不會把 JSON object 的輸入
# 順序誤當成科學語意。前四欄是文字識別，其餘八欄是非 bool 的正原生 Python 整數。
_INSPECTION_SUMMARY_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "run_id",
    "run_kind",
    "experiment_case_id",
    "members_per_scenario",
    "input_particle_count",
    "shard_row_count",
    "scenario_row_count",
    "site_row_count",
    "boundary_row_count",
    "source_receptor_row_count",
    "payload_file_count",
)
_INSPECTION_SUMMARY_TEXT_FIELDS: Final[frozenset[str]] = frozenset(
    _INSPECTION_SUMMARY_FIELDS[:4]
)


@dataclass(frozen=True, slots=True)
class _AggregateReleaseInspection:
    """封存完整解碼 payload 與不含實作物件的固定公開摘要。

    ``payload`` 必須是 decoder 新建的 exact ``AggregateReleasePayload``；本容器保存同一
    reference，不再複製、重算或轉換其中公尺制格網、秒制時間軸、計數與缺值狀態。
    ``summary`` 只接受固定十二欄，識別欄必須是原生 ``str``，計數欄必須是排除 bool、
    NumPy scalar 與 Arrow scalar 的正原生 Python ``int``，且 ``payload_file_count`` 固定
    為 32。驗證後依固定欄位順序重建普通 dict，再以 ``MappingProxyType`` 封存，因此
    summary 不可能攜帶 ``Path``、NumPy 陣列／scalar、Arrow 物件或其他不可 JSON
    序列化的 implementation detail。

    這個 private inspection 只表示 release 工程契約與來源綁定通過；即使 payload 的
    ``run_kind`` 是 ``synthetic``，也不代表本機已讀取正式 OCM／NWW 或產生可解讀的
    科學成果。公開 API 只會回傳 payload 或 summary 的普通 dict 副本，不直接暴露此容器。
    """

    payload: AggregateReleasePayload
    summary: Mapping[str, object]

    def __post_init__(self) -> None:
        """驗證 exact payload 與十二欄 JSON-safe scalar 後建立唯讀摘要快照。"""

        if type(self.payload) is not AggregateReleasePayload:
            raise ValueError("payload 必須是 exact AggregateReleasePayload 實例")
        if not isinstance(self.summary, Mapping):
            raise ValueError("summary 必須是 mapping")
        try:
            summary_copy = dict(self.summary)
        except Exception as error:
            raise ValueError("summary 無法建立普通 dict 快照") from error
        if set(summary_copy) != set(_INSPECTION_SUMMARY_FIELDS):
            raise ValueError("summary 必須恰好包含固定十二欄")

        # 依固定欄位順序取值，同時以 exact type 排除 Path、bool、NumPy／Arrow scalar
        # 及其他具有隱式字串或整數轉型能力的物件。這裡不呼叫 str() 或 int() 修補輸入。
        ordered_summary: dict[str, object] = {}
        for field_name in _INSPECTION_SUMMARY_FIELDS:
            value = summary_copy[field_name]
            if field_name in _INSPECTION_SUMMARY_TEXT_FIELDS:
                if type(value) is not str:
                    raise ValueError("summary 文字欄位必須是原生 str")
            elif type(value) is not int or value <= 0:
                raise ValueError("summary 計數欄位必須是正原生 int")
            ordered_summary[field_name] = value
        if ordered_summary["payload_file_count"] != 32:
            raise ValueError("summary.payload_file_count 必須固定為 32")

        object.__setattr__(self, "summary", MappingProxyType(ordered_summary))


def _inspect_aggregate_release(
    path: str | Path,
    require_final_name: bool = True,
) -> _AggregateReleaseInspection:
    """依固定 stage 順序完整驗證並解碼一份 aggregate release。

    驗證順序不可調換：先由 foundation 鎖定 33 個普通節點、canonical manifest、final
    basename（若啟用）與三十二檔 checksum；再驗證 source run／AggregateSpec 文件；接著
    讀取九張 Parquet 與十八個 NPY 並核對 manifest product contracts；最後才執行 codec
    decoder 及 plan/progress shard binding。函式不攔截既有
    ``_ReleaseValidationError``，因此 topology、manifest、name、checksum、source、
    products 與 decoder 的第一個失敗 stage 能原樣傳給公開 validator，也不會失敗後
    繼續讀取後續產品。

    成功後的 summary 只從 typed ``AggregateReleaseMetadata`` 建立十二個原生字串／整數，
    不從 payload 陣列反推計數，也不放入 caller path、NumPy、Arrow 或 source 文件。
    ``require_final_name`` 的 exact bool 邊界由 foundation 驗證；公開 reader／validator
    一律使用 ``True``，保留 ``False`` 只供後續 writer 驗證其自有 partial 目錄。完整工程
    驗證仍不等同正式 OCM／NWW 科學正確性或觀測驗證。

    Args:
        path: 直接包含 manifest 與三十二份 payload 檔案的 release 根目錄。
        require_final_name: 是否要求 basename 精確為 ``<run_id>.aggregate-v1``。

    Returns:
        保存 exact decoded payload 與固定 JSON-safe 摘要的 private inspection。

    Raises:
        _ReleaseValidationError: 任一既有驗證 stage 失敗；函式不改寫或吞掉 stage。
        ValueError: inspection summary 的內部固定契約意外不一致。
    """

    metadata, contracts, source_json_bytes = _validate_release_topology_and_checksums(
        path,
        require_final_name=require_final_name,
    )
    source_snapshot = _validate_source_documents(
        path,
        metadata=metadata,
        contracts=contracts,
        source_json_bytes=source_json_bytes,
    )
    products = _validate_product_contracts(path, contracts)
    payload = _decode_and_bind_release(metadata, source_snapshot, products)

    # metadata 已由 AggregateReleaseMetadata constructor 驗證為原生 scalar。這裡明示
    # 十二欄，不以 metadata.to_dict() 加減欄位，避免 provenance digest 或未來新增欄位
    # 意外擴張公開 validator 的穩定 summary 契約。
    summary: dict[str, object] = {
        "schema_version": metadata.schema_version,
        "run_id": metadata.run_id,
        "run_kind": metadata.run_kind,
        "experiment_case_id": metadata.experiment_case_id,
        "members_per_scenario": metadata.members_per_scenario,
        "input_particle_count": metadata.input_particle_count,
        "shard_row_count": metadata.shard_row_count,
        "scenario_row_count": metadata.scenario_row_count,
        "site_row_count": metadata.site_row_count,
        "boundary_row_count": metadata.boundary_row_count,
        "source_receptor_row_count": metadata.source_receptor_row_count,
        "payload_file_count": 32,
    }
    return _AggregateReleaseInspection(payload=payload, summary=summary)


def validate_aggregate_release(path: str | Path) -> dict[str, object]:
    """完整驗證 final aggregate release 並回傳固定 JSON-safe 報告。

    成功結果精確包含 ``valid=True``、空 ``errors`` 與十二欄 summary；summary 只含原生
    ``str``／``int``，不含 decoded payload、Path、NumPy、Arrow 或 source 文件。已登錄
    release 錯誤只回傳第一個固定 stage code；任何未預期例外則收斂為 ``decoder``，避免
    擴張公開錯誤契約或洩漏本機／SERVER 路徑與第三方訊息。此 API 一律要求 final basename，
    不接受 partial 目錄冒充已發布 release。

    ``valid=True`` 只證明檔案、來源 provenance、產品與 codec 工程契約一致；若使用本機
    synthetic fixture，結果不是正式 OCM schema 3／NWW3 schema 1 科學成果，也不建立
    絕對來源機率、因果歸因或觀測驗證。

    Args:
        path: 待驗證的 final aggregate release 根目錄。

    Returns:
        精確含 ``valid``、``errors``、``summary`` 三個 key 的普通 JSON-safe dict。
    """

    try:
        inspection = _inspect_aggregate_release(path, require_final_name=True)
    except _ReleaseValidationError as error:
        return {"valid": False, "errors": [error.stage], "summary": {}}
    except Exception:
        # 未預期錯誤不建立新的 internal stage，也不回傳例外文字或 path；固定歸入
        # decoder 代表完整 typed payload 尚未安全建立。
        return {"valid": False, "errors": ["decoder"], "summary": {}}
    return {
        "valid": True,
        "errors": [],
        "summary": dict(inspection.summary),
    }


def read_aggregate_release(path: str | Path) -> AggregateReleasePayload:
    """完整驗證並讀取 final aggregate release 的 exact decoded payload。

    本入口和公開 validator 使用完全相同的 final-name inspection；只有 topology、manifest、
    全部 checksum、source semantic、產品 contract、codec round-trip 與 run shard binding
    全數通過才回傳 ``AggregateReleasePayload``。任何已登錄或未預期失敗都統一轉成不帶
    cause、stage、第三方文字或 caller path 的固定 ``ValueError``，讓 CLI 可提供穩定且
    不洩漏部署資訊的錯誤界面。

    payload 中格網／邊界距離以公尺、travel age／停留時間以秒，缺值、乾點、域外、資料
    缺口與數值失敗仍由各自欄位保存，不以零值替代。成功讀取本機 synthetic release 只
    是工程驗證，不是真實 OCM／NWW 科學成果，也不可單獨解讀為絕對來源機率或因果歸因。

    Args:
        path: 待讀取的 final aggregate release 根目錄。

    Returns:
        通過完整驗證的 exact ``AggregateReleasePayload``。

    Raises:
        ValueError: 任一驗證或解碼失敗，訊息固定為 ``aggregate release 驗證失敗``。
    """

    try:
        return _inspect_aggregate_release(path, require_final_name=True).payload
    except Exception:
        raise ValueError("aggregate release 驗證失敗") from None


def write_aggregate_release(
    *,
    source_run_root: str | Path,
    aggregate_spec_path: str | Path,
    payload: AggregateReleasePayload,
    destination: str | Path | None = None,
    checkpoint_root: str | Path | None = None,
) -> Path:
    """在 source run exclusive gate 內原子發布一份 aggregate release。

    這是正式 CLI 未來使用的公開寫入邊界。source run 根目錄、``locks`` 與既有
    ``locks/run_gate.lock`` 必須先通過不跟隨 symbolic link 的節點檢查；取得非阻塞
    exclusive run gate 後，整個流程才會讀取 source plan／progress、驗證完整 run、檢查
    trajectory manifest、建立 partial、寫入產品、完整 inspection、重驗 live trajectory、
    原子 rename 及同步 final parent。如此同一個 run 的官方 writer 不會在兩次 output
    binding 之間互相覆蓋或讀到另一個 worker 正在更新的 shard。

    writer 先由 ``_prepare_aggregate_release_source`` 產生不含絕對路徑的 prepared snapshot，
    並要求 source root basename 精確等於 prepared 的 ``run_id``。``destination=None`` 時，
    final 位置固定是 source run resolved parent 下的 ``<run_id>.aggregate-v1``；明示
    destination 時，必須同時通過 lexical absolute-normalized exact equality、resolved
    parent equality、固定 basename、parent 非 symbolic link 與 final absent 檢查。既有
    regular file、directory、symbolic link 或 broken symbolic link 都不可覆寫。

    partial 只在 final parent 以 ``mkdir(mode=0o700, exist_ok=False)`` 建立唯一名稱
    ``.<run_id>.aggregate-v1.partial-<uuid>``，並記錄本次建立目錄的 device/inode。所有
    失敗清理只能在 rename 前針對這個已驗證 identity 的 owned partial；不掃描、不刪除
    其他 partial，也永不修改 source run。partial 內容包含 9 張 Parquet、18 個 NumPy
    陣列、5 份 source exact JSON bytes 與最後建立的 canonical aggregate manifest；格網與
    邊界距離使用公尺，age／停留時間使用秒，計數與 offset 依固定 release schema 保存。

    partial 完成後會以 full inspection 重跑 topology、manifest、checksum、source semantic、
    product contract 與 decoder 驗證，並以 ``metadata_from_payload`` 比較 inspection
    payload 的 metadata；不能直接對含 NumPy array 的 payload 使用 ``==``。在
    ``os.replace`` 前會再次從同一 source run 讀取並 hash trajectory manifest，立即重查
    final absent 及 partial device/inode；rename 成功後 ownership 轉移，任何 parent
    ``fsync`` 失敗都不刪除 final，而固定回報「已發布但 parent durability 未確認」。
    parent 尚未被 rename 修改前不會 fsync；本函式不把任何絕對 SERVER path 寫入 release
    manifest。

    contention 的 ``RunLockBusyError`` 會原樣傳出，讓 CLI 能區分「另一個 writer 正在
    執行」與一般資料／I/O 失敗。rename 前其他失敗一律固定為
    ``ValueError("aggregate release 寫入失敗")``，不攜帶 path、cause、OS 或第三方錯誤；
    真正的 OCM／NWW 科學資料仍須由 SERVER 的正式產品輸入，因而本機 synthetic release
    通過只代表工程資料契約，不是科學成果、絕對來源機率、因果歸因或觀測驗證。

    Args:
        source_run_root: 已完成且受 run gate 保護的 source run workspace 普通目錄。
        aggregate_spec_path: caller 明示的 AggregateSpec JSON；只在 lock 內由 source
            preparation 讀取並封存 exact bytes。
        payload: 已完成聚合的 exact ``AggregateReleasePayload``，包含公尺／秒單位與
            條件式來源足跡／相對來源權重的工程資料。
        destination: 可選 final path；必須與 source run parent 下的固定 final 完全相同。
        checkpoint_root: 可選外部 checkpoint root，只傳給唯讀 source run validator，不會
            寫入 manifest 或 release path。

    Returns:
        已完成原子 rename 且 parent durability 已確認的 final release ``Path``。

    Raises:
        RunLockBusyError: source run gate 被其他程序持有，原樣傳出。
        ValueError: rename 前任一資料、路徑、I/O、partial 或驗證失敗，固定訊息為
            ``aggregate release 寫入失敗``。
        RuntimeError: rename 已成功但 final parent ``fsync`` 未成功，固定訊息為
            ``aggregate release 已發布但 parent durability 未確認``。
    """

    partial_path: Path | None = None
    partial_identity: tuple[int, int] | None = None
    final_path: Path | None = None
    published = False

    def require_directory_node(path: Path) -> os.stat_result:
        """以 lstat 要求既有節點是普通非 symbolic-link 目錄。"""

        node_status = os.lstat(path)
        if stat.S_ISLNK(node_status.st_mode) or not stat.S_ISDIR(node_status.st_mode):
            raise ValueError("directory node 不符合 release contract")
        return node_status

    def require_absent_node(path: Path) -> None:
        """以 lstat 拒絕既有普通節點與 broken symbolic link。"""

        try:
            os.lstat(path)
        except FileNotFoundError:
            return
        raise ValueError("release final target 必須不存在")

    def require_same_directory_identity(
        path: Path,
        expected_identity: tuple[int, int],
    ) -> None:
        """確認目錄仍是原本的普通非連結 device/inode。"""

        node_status = require_directory_node(path)
        if (node_status.st_dev, node_status.st_ino) != expected_identity:
            raise ValueError("directory identity 已改變")

    def cleanup_owned_partial() -> None:
        """只在 rename 前清理本次建立且 identity 未變的 partial。"""

        if partial_path is None or partial_identity is None or published:
            return
        try:
            node_status = os.lstat(partial_path)
            if (
                stat.S_ISLNK(node_status.st_mode)
                or not stat.S_ISDIR(node_status.st_mode)
                or (node_status.st_dev, node_status.st_ino) != partial_identity
            ):
                return
            shutil.rmtree(partial_path)
        except Exception:
            # 原始固定錯誤比 cleanup 例外更重要；若 identity、權限或 FS 狀態已不明，
            # 寧可保留現場，也不冒險刪除非本次 writer 擁有的節點。
            return

    def fsync_directory(path: Path, expected_identity: tuple[int, int]) -> None:
        """以 no-follow directory descriptor 同步指定 parent，錯誤不得被吞掉。"""

        flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        try:
            node_status = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(node_status.st_mode)
                or (node_status.st_dev, node_status.st_ino) != expected_identity
            ):
                raise OSError("release parent identity 不一致")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    try:
        # 先把 source root 正規化成 lexical absolute path，並只以 lstat 檢查 root 與
        # parent 的最終節點；祖先路徑可能是作業系統別名（例如 macOS 的 /var），不應
        # 因 lexical path 與 resolve 結果不同而被誤判。resolved parent 只供明示
        # destination 做實體 parent 相等性檢查，不能反過來限制 source 的 lexical path。
        source_root_input = Path(source_run_root)
        source_root = Path(
            os.path.abspath(os.path.normpath(os.fspath(source_root_input)))
        )
        require_directory_node(source_root)

        source_parent = source_root.parent
        source_parent_status = require_directory_node(source_parent)
        source_parent_resolved = source_parent.resolve(strict=True)
        source_parent_identity = (
            source_parent_status.st_dev,
            source_parent_status.st_ino,
        )

        locks_root = source_root / "locks"
        require_directory_node(locks_root)
        lock_path = locks_root / "run_gate.lock"
        lock_status = os.lstat(lock_path)
        if stat.S_ISLNK(lock_status.st_mode) or not stat.S_ISREG(lock_status.st_mode):
            raise ValueError("run_gate.lock 必須是普通非 symlink 檔案")

        # 從這個 context 起，所有 source plan/progress、trajectory output、partial/final
        # 讀寫與 parent durability 都由同一個 non-blocking exclusive run gate 包住。
        with acquire_run_lock(lock_path, mode="exclusive", blocking=False):
            prepared = _prepare_aggregate_release_source(
                source_run_root=source_root,
                aggregate_spec_path=aggregate_spec_path,
                payload=payload,
                checkpoint_root=checkpoint_root,
            )
            run_id = prepared.metadata.run_id
            if source_root.name != run_id:
                raise ValueError("source root basename 與 run_id 不一致")
            final_name = f"{run_id}{_FINAL_DIRECTORY_SUFFIX}"
            expected_final = source_parent / final_name

            if destination is None:
                final_path = expected_final
            else:
                destination_input = Path(destination)
                if ".." in destination_input.parts:
                    raise ValueError("destination 不可含 ..")
                destination_lexical = Path(
                    os.path.abspath(os.path.normpath(os.fspath(destination_input)))
                )
                if destination_lexical != expected_final:
                    raise ValueError("destination lexical path 不符合固定 final")
                if (
                    destination_lexical.name != final_name
                    or destination_lexical.parent != source_parent
                ):
                    raise ValueError("destination basename/parent 不符合固定 final")
                destination_parent_status = require_directory_node(
                    destination_lexical.parent
                )
                if destination_lexical.parent.resolve(strict=True) != source_parent_resolved:
                    raise ValueError("destination resolved parent 不符合 source parent")
                if (
                    destination_parent_status.st_dev,
                    destination_parent_status.st_ino,
                ) != source_parent_identity:
                    raise ValueError("destination parent identity 不一致")
                final_path = destination_lexical

            if final_path is None:
                raise ValueError("final path 尚未建立")
            require_absent_node(final_path)
            require_same_directory_identity(source_parent, source_parent_identity)

            partial_path = source_parent / f".{final_name}.partial-{uuid4().hex}"
            partial_path.mkdir(mode=0o700, exist_ok=False)
            partial_status = os.lstat(partial_path)
            if stat.S_ISLNK(partial_status.st_mode) or not stat.S_ISDIR(
                partial_status.st_mode
            ):
                raise ValueError("partial 必須是普通非 symlink 目錄")
            partial_identity = (partial_status.st_dev, partial_status.st_ino)

            _validate_current_trajectory_manifests(
                source_run_root=source_root,
                prepared=prepared,
            )
            _write_prepared_release_directory(partial_path, prepared)
            inspection = _inspect_aggregate_release(
                partial_path,
                require_final_name=False,
            )
            if metadata_from_payload(inspection.payload) != prepared.metadata:
                raise ValueError("inspection metadata 與 prepared 不一致")

            # 這是 rename 前最後一次 live trajectory binding；後面只做 identity／target
            # lstat 與原子 rename，不再讀 source JSON、產品或其他未鎖定路徑。
            _validate_current_trajectory_manifests(
                source_run_root=source_root,
                prepared=prepared,
            )
            require_same_directory_identity(source_parent, source_parent_identity)
            require_absent_node(final_path)
            require_same_directory_identity(partial_path, partial_identity)

            os.replace(partial_path, final_path)
            published = True
            fsync_directory(source_parent, source_parent_identity)

        return final_path
    except RunLockBusyError:
        # lock contention 是可預期的控制流；保留原始例外型別與訊息，讓 caller 能重試或
        # 排程，而不是把「忙碌」誤判成資料損壞。此例外不會留下本次 partial。
        raise
    except Exception:
        if published:
            # os.replace 已經把 partial ownership 轉交給 final；無論 parent fsync 的
            # 平台錯誤內容為何，都不能刪除 final 或假裝 durability 已確認。
            raise RuntimeError(
                "aggregate release 已發布但 parent durability 未確認"
            ) from None
        cleanup_owned_partial()
        raise ValueError("aggregate release 寫入失敗") from None
