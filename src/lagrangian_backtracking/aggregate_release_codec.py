"""Aggregate release 的純記憶體固定拓撲 codec。

本模組把已完成跨產品驗證的 ``AggregateReleasePayload`` 轉成九張表與十八個
NumPy 陣列，並由同一份 canonical 產品建立 manifest 與 decoder 之間使用的
``AggregateReleaseMetadata``。公開 decoder 接受 exact typed metadata、
``AggregateSpec`` 與固定產品，驗證欄位、排序、offset、shape、拓撲及 round-trip
一致性後，重建完整 ``AggregateReleasePayload``。本模組不讀寫 Parquet／NPY、不建立
或解析 release manifest，也不替缺失的站點、邊界、受體或計數猜值。所有空間長度採
公尺、路徑與 travel-age 時間採秒；格網一律按 ``(y_cell, x_cell)`` 的 C-order 展平，
直方圖的 age-bin 軸為最後一軸。

編解碼產品只是條件式來源足跡或相對來源權重的可發布載體；成功 round-trip 不代表
資料已通過觀測驗證，也不能據此宣稱絕對來源機率或因果歸因。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Final

import numpy as np

from .aggregate_release_layout import (
    AGGREGATE_RELEASE_ARRAY_FILES,
    AGGREGATE_RELEASE_TABLE_FILES,
    EncodedAggregateProducts,
)
from .aggregate_release_payload import (
    AGGREGATE_RELEASE_SCHEMA_VERSION,
    AggregateReleasePayload,
)
from .aggregate_release_payload import _require_safe_slug as _payload_require_safe_slug
from .aggregate_release_payload import _require_sha256 as _payload_require_sha256
from .aggregate_release_records import AggregateShardBinding, ScenarioStratum
from .aggregate_spec import (
    AggregateSpec,
    SiteBoundarySegments,
    SiteGridSpec,
    SiteMetricCRSSpec,
)
from .event_aggregation import (
    BoundaryAggregateKey,
    CrossSiteAggregateKey,
    EventAggregateChunk,
    ReceptorAggregateKey,
    SiteEventGridCounts,
    SourceReceptorAggregateKey,
    initialize_event_aggregate,
)
from .models import ParticleStatus
from .streaming_aggregation import StreamingPathwayAggregate

__all__ = [
    "AggregateReleaseMetadata",
    "decode_aggregate_release_payload",
    "encode_aggregate_release_payload",
    "metadata_from_payload",
]


_INT64_MIN = int(np.iinfo(np.int64).min)
_INT64_MAX = int(np.iinfo(np.int64).max)

# 六個事件格網與輸出檔名是一對一固定契約；tuple 順序同時固定 encoder 的處理次序，
# 但每站資料仍一律依 site_index 的站點字典序串接。
_EVENT_GRID_PRODUCTS = (
    ("local_first_exit_count.npy", "local_first_exit_count"),
    ("outer_first_exit_count.npy", "outer_first_exit_count"),
    ("bed_first_contact_count.npy", "bed_first_contact_count"),
    ("bed_repeated_contact_count.npy", "bed_repeated_contact_count"),
    ("data_gap_failure_count.npy", "data_gap_failure_count"),
    ("numerical_failure_count.npy", "numerical_failure_count"),
)

# 每張表的欄位順序是 release schema 的一部分，不能依賴 mapping 插入順序或
# Parquet reader 的實作細節。decoder 會先逐列核對這些 tuple，再把 row 轉回公開
# dataclass／key constructor；因此額外欄位、缺欄位與錯誤欄位順序都會在資料進入
# 科學容器前 fail closed。所有距離／格線欄位仍是公尺，時間 scalar 是秒，計數與
# offset 是可寫入 signed int64 的非負原生 Python int。
_SHARD_BINDING_COLUMNS: Final[tuple[str, ...]] = (
    "shard_index",
    "shard_id",
    "scenario_start_index",
    "scenario_stop_index",
    "output_relative_path",
    "trajectory_manifest_sha256",
    "particle_count",
    "observation_count",
    "event_count",
)
_SCENARIO_STRATUM_COLUMNS: Final[tuple[str, ...]] = (
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
)
_SITE_INDEX_COLUMNS: Final[tuple[str, ...]] = (
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
)
_BOUNDARY_INDEX_COLUMNS: Final[tuple[str, ...]] = (
    "boundary_index",
    "study_site_id",
    "boundary_kind",
    "boundary_segment_id",
    "segment_length_m",
    "edge_start_offset",
    "edge_stop_offset",
    "bin_start_offset",
    "bin_stop_offset",
)
_SOURCE_RECEPTOR_INDEX_COLUMNS: Final[tuple[str, ...]] = (
    "source_receptor_index",
    "study_site_id",
    "receptor_id",
    "boundary_kind",
    "boundary_segment_id",
)
_CROSS_SITE_COUNT_COLUMNS: Final[tuple[str, ...]] = (
    "source_study_site_id",
    "target_study_site_id",
    "unique_member_count",
)
_OUTCOME_COUNT_COLUMNS: Final[tuple[str, ...]] = (
    "study_site_id",
    "outcome",
    "count",
)
_SITE_DENOMINATOR_COLUMNS: Final[tuple[str, ...]] = (
    "study_site_id",
    "valid_member_denominator",
    "total_member_count",
)
_RECEPTOR_DENOMINATOR_COLUMNS: Final[tuple[str, ...]] = (
    "study_site_id",
    "receptor_id",
    "valid_member_denominator",
)

_SHARD_BINDING_FIELDS: Final[tuple[str, ...]] = _SHARD_BINDING_COLUMNS[1:]
_SCENARIO_STRATUM_FIELDS: Final[tuple[str, ...]] = _SCENARIO_STRATUM_COLUMNS[1:]
_SCENARIO_DYNAMIC_OPTIONAL_FIELDS: Final[tuple[str, ...]] = (
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


def _require_positive_int64_count(value: object, *, label: str) -> int:
    """驗證 metadata 計數是可寫入 signed int64 的正原生 Python 整數。

    metadata 會成為 manifest 與後續 decoder 之間的型別邊界；成員數與列數若被
    ``bool``、NumPy scalar、浮點數或可轉型物件冒充，JSON／Parquet 邊界可能產生
    不同的型別或溢位語意。因此這裡刻意使用精確型別檢查，不做 ``int(value)``
    或其他寬鬆轉換；零也不接受，因為本 release payload 的固定拓撲要求這些
    成員與五張核心索引表都實際存在。
    """

    if type(value) is not int or not 0 < value <= _INT64_MAX:
        raise ValueError(
            f"{label} 必須是 1 到 signed int64 上限內的原生 Python int"
        )
    return value


@dataclass(frozen=True, slots=True)
class AggregateReleaseMetadata:
    """manifest 與 decoder 之間的不可變聚合 release metadata 邊界。

    前半欄位保存 release 的 schema、run／experiment 識別與輸入 provenance：
    ``config_hash`` 與 ``checkpoint_input_binding_hash`` 綁定執行設定和 checkpoint
    輸入，四個 ``source_*_sha256`` 綁定建立 payload 時所依據的 source JSON，兩個
    ``aggregate_spec_*_sha256`` 則直接來自重建後 ``AggregateSpec`` 的原始檔案與
    canonical 內容摘要。這些摘要只描述位元組或 canonical JSON 的來源關係，
    不會替代 writer 的檔案 checksum，也不會替 decoder 猜測缺少的來源檔案。

    ``input_particle_count`` 是事件 aggregate 已驗證的全域輸入粒子數；其餘五個
    ``*_row_count`` 分別是 shard binding、scenario stratum、site、boundary 與
    source-receptor 五張固定索引表的實際列數。列數是產品拓撲與可追溯性的工程
    metadata，不是事件數、有效分母、條件式來源足跡或相對來源權重；零值、缺列、
    資料缺口、乾點與數值失敗仍須由各自產品欄位保存，不能用這些欄位代替。

    所有 count 都必須是可無損表示為 signed ``int64`` 的正原生 Python ``int``，
    所有 digest 都必須是 64 碼小寫 SHA-256。這個類別只驗證與封存 metadata，
    不讀寫 manifest、不執行 decoder、不重算科學量；成功建立它仍只能支持
    「條件式來源足跡」或「相對來源權重」的發布，不能宣稱絕對來源機率或因果
    歸因。
    """

    schema_version: str
    run_id: str
    run_kind: str
    experiment_case_id: str
    members_per_scenario: int
    config_hash: str
    checkpoint_input_binding_hash: str
    source_run_plan_sha256: str
    source_run_progress_sha256: str
    source_normalized_config_sha256: str
    source_input_inventory_sha256: str
    aggregate_spec_source_sha256: str
    aggregate_spec_canonical_sha256: str
    input_particle_count: int
    shard_row_count: int
    scenario_row_count: int
    site_row_count: int
    boundary_row_count: int
    source_receptor_row_count: int

    def __post_init__(self) -> None:
        """以與 payload 相同的識別規則完成 metadata 的 fail-closed 驗證。

        schema version 必須精確等於目前 release 版本；run ID 與 experiment case
        ID 直接重用 payload 的安全 ASCII slug 驗證；run kind 只接受已登錄的三種
        release 類型。所有 provenance digest 都逐欄核對小寫 SHA-256，所有粒子／列
        計數都拒絕 bool、NumPy scalar、零與超出 signed int64 的值。此方法不從其他
        欄位推導或修補值，並把驗證後的原值寫回 frozen instance，確保 constructor
        與 ``to_dict`` 看到的是同一份型別與內容。
        """

        if type(self.schema_version) is not str:
            raise ValueError("schema_version 必須是原生 str")
        if self.schema_version != AGGREGATE_RELEASE_SCHEMA_VERSION:
            raise ValueError("schema_version 不受支援")

        run_id = _payload_require_safe_slug(self.run_id, label="run_id")
        experiment_case_id = _payload_require_safe_slug(
            self.experiment_case_id,
            label="experiment_case_id",
        )
        if type(self.run_kind) is not str or self.run_kind not in {
            "synthetic",
            "pilot",
            "formal",
        }:
            raise ValueError("run_kind 只允許 synthetic、pilot 或 formal")

        digest_fields = (
            "config_hash",
            "checkpoint_input_binding_hash",
            "source_run_plan_sha256",
            "source_run_progress_sha256",
            "source_normalized_config_sha256",
            "source_input_inventory_sha256",
            "aggregate_spec_source_sha256",
            "aggregate_spec_canonical_sha256",
        )
        digests = {
            field_name: _payload_require_sha256(
                getattr(self, field_name),
                label=field_name,
            )
            for field_name in digest_fields
        }

        count_fields = (
            "members_per_scenario",
            "input_particle_count",
            "shard_row_count",
            "scenario_row_count",
            "site_row_count",
            "boundary_row_count",
            "source_receptor_row_count",
        )
        counts = {
            field_name: _require_positive_int64_count(
                getattr(self, field_name),
                label=field_name,
            )
            for field_name in count_fields
        }

        # 所有 helper 都只回傳通過驗證的原值；這裡不做字串化、數值轉型或缺值猜測，
        # 讓 frozen dataclass 的公開欄位與輸入 payload 保持一一對應的資料契約。
        object.__setattr__(self, "schema_version", self.schema_version)
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "experiment_case_id", experiment_case_id)
        for field_name, digest in digests.items():
            object.__setattr__(self, field_name, digest)
        for field_name, count in counts.items():
            object.__setattr__(self, field_name, count)

    def to_dict(self) -> dict[str, object]:
        """依固定欄位順序回傳只含原生字串與整數的 JSON-safe 普通 dict。

        dict 的插入順序刻意與 metadata schema 及 dataclass 欄位順序完全一致，
        使 manifest writer 後續可採同一個公開欄位契約序列化；本方法不加入衍生欄位、
        不排序鍵名、不把整數或 digest 轉成字串，也不回傳 dataclass、mapping proxy
        或 NumPy scalar。此結果仍只是 metadata 快照，不代表資料已完成科學驗證。
        """

        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "run_kind": self.run_kind,
            "experiment_case_id": self.experiment_case_id,
            "members_per_scenario": self.members_per_scenario,
            "config_hash": self.config_hash,
            "checkpoint_input_binding_hash": self.checkpoint_input_binding_hash,
            "source_run_plan_sha256": self.source_run_plan_sha256,
            "source_run_progress_sha256": self.source_run_progress_sha256,
            "source_normalized_config_sha256": self.source_normalized_config_sha256,
            "source_input_inventory_sha256": self.source_input_inventory_sha256,
            "aggregate_spec_source_sha256": self.aggregate_spec_source_sha256,
            "aggregate_spec_canonical_sha256": self.aggregate_spec_canonical_sha256,
            "input_particle_count": self.input_particle_count,
            "shard_row_count": self.shard_row_count,
            "scenario_row_count": self.scenario_row_count,
            "site_row_count": self.site_row_count,
            "boundary_row_count": self.boundary_row_count,
            "source_receptor_row_count": self.source_receptor_row_count,
        }

    @classmethod
    def from_dict(cls, document: Mapping[str, object]) -> AggregateReleaseMetadata:
        """由 exact-key mapping 重建 metadata，並把每個原值直接交給 constructor。

        輸入必須是 mapping，且欄位集合不得有任何額外或缺少的 key；先完成這個
        結構檢查，再依固定欄位順序取值。方法不套用 ``str``、``int``、NumPy
        scalar 或預設值轉換，因此 JSON／外部 adapter 若給出錯誤型別，會由
        ``__post_init__`` 立即拒絕，而不會被寬鬆轉型掩蓋。
        """

        if not isinstance(document, Mapping):
            raise ValueError("metadata document 必須是 mapping")

        field_names = tuple(field.name for field in fields(cls))
        expected_keys = frozenset(field_names)
        try:
            actual_keys = frozenset(document)
        except Exception as error:
            raise ValueError("metadata document 無法建立 exact key set") from error
        if actual_keys != expected_keys:
            unknown_keys = tuple(key for key in document if key not in expected_keys)
            missing_keys = tuple(key for key in field_names if key not in actual_keys)
            raise ValueError(
                "metadata document 必須恰好包含固定欄位；"
                f"未知 key={unknown_keys!r}；缺少 key={missing_keys!r}"
            )

        try:
            values = {field_name: document[field_name] for field_name in field_names}
        except Exception as error:
            raise ValueError("metadata document 欄位無法讀取") from error
        return cls(**values)


def _require_int64_count(value: object, *, label: str) -> int:
    """驗證 Python 計數可無損寫入非負 ``int64``，明確排除 bool 與 NumPy scalar。"""

    if type(value) is not int or not 0 <= value <= _INT64_MAX:
        raise ValueError(f"{label} 必須是 0 到 int64 上限內的原生 Python int")
    return value


def _require_int64_scalar(value: object, *, label: str) -> int:
    """驗證一般 Arrow 整數欄位是 native Python int 且位於 signed int64 範圍。

    世界協調時間奈秒（``arrival_time_utc_ns``）可合法位於 Unix epoch 之前，因此不能
    套用非負 count 規則；年份與 wet/dry 值等其他非計數整數也走此 signed 邊界。bool
    與 NumPy scalar 明確拒絕，避免 writer 階段才發生型別推斷或溢位差異。
    """

    if type(value) is not int or not _INT64_MIN <= value <= _INT64_MAX:
        raise ValueError(f"{label} 必須是 signed int64 範圍內的原生 Python int")
    return value


def _checked_count_product(left: object, right: object, *, label: str) -> int:
    """以 Python 任意精度整數相乘後檢查 int64 上限，避免 NumPy 乘法先溢位。"""

    left_count = _require_int64_count(left, label=f"{label} 左運算元")
    right_count = _require_int64_count(right, label=f"{label} 右運算元")
    return _require_int64_count(left_count * right_count, label=label)


def _append_nonempty_offset(
    offsets: list[int],
    segment_length: object,
    *,
    label: str,
) -> tuple[int, int]:
    """把非空片段加入累積 offset，回傳其 ``[start, stop)`` 半開範圍。

    起點沿用前一片段終點，終點以 Python int 加法取得後才檢查 int64 上限。片段長度
    必須大於零，因此 offset 嚴格遞增；codec 不允許空片段、倒退、截短或 padding。
    """

    length = _require_int64_count(segment_length, label=f"{label} 長度")
    if length == 0:
        raise ValueError(f"{label} 不得是空片段")
    start = offsets[-1]
    stop = _require_int64_count(start + length, label=f"{label} stop offset")
    if stop <= start:
        raise ValueError(f"{label} offset 必須嚴格遞增")
    offsets.append(stop)
    return start, stop


def _require_nonnegative_finite_float(value: object, *, label: str) -> float:
    """驗證秒數或非負公尺值是有限 Python scalar，並 canonical 成 float。"""

    if type(value) not in (int, float):
        raise ValueError(f"{label} 必須是有限非負的原生 Python int 或 float")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{label} 無法轉成有限 float") from error
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} 必須是有限非負值")
    return number


def _require_finite_float(value: object, *, label: str) -> float:
    """驗證座標或物理 scalar 可有限表示為 Python float，不改變其科學語意。"""

    if type(value) not in (int, float):
        raise ValueError(f"{label} 必須是有限的原生 Python int 或 float")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{label} 無法轉成有限 float") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} 必須是有限值")
    return number


def _scalar_row(row: dict[str, object], *, label: str) -> dict[str, object]:
    """確認表格列只含欄名字串與可直接序列化的 Python scalar。

    codec 不把 list、tuple、mapping、NumPy scalar 或 ndarray 藏進 Parquet 列。欄名屬於
    index、offset、count 或 denominator 時，還會重用非負 int64 檢查，避免後續 writer
    才因 fixed-width 轉型產生 silent overflow。
    """

    copied: dict[str, object] = {}
    for field_name, value in row.items():
        if type(field_name) is not str:
            raise ValueError(f"{label} 的欄名必須是原生 str")
        if value is None:
            copied[field_name] = None
            continue
        if type(value) not in (str, int, float, bool):
            raise ValueError(f"{label}.{field_name} 只能是 None/str/int/float/bool scalar")
        if type(value) is float and not math.isfinite(value):
            raise ValueError(f"{label}.{field_name} 的 float 必須有限")
        if type(value) is int:
            if (
                field_name.endswith(("_index", "_offset", "_count"))
                or field_name in {"count", "valid_member_denominator"}
            ):
                value = _require_int64_count(value, label=f"{label}.{field_name}")
            else:
                # arrival_time_utc_ns、arrival_year 與其他固定 Arrow int64 欄位允許
                # signed 值；全部在 codec 階段檢查，不能延後到 writer 才發現溢位。
                value = _require_int64_scalar(value, label=f"{label}.{field_name}")
        copied[field_name] = value
    return copied


def _indexed_dataclass_row(
    value: object,
    *,
    index_name: str,
    index: int,
    label: str,
) -> dict[str, object]:
    """在 dataclass 全部正式欄位之前加入穩定 index，並拒絕任何巢狀列值。"""

    try:
        record_values = asdict(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} 必須是可展開的 dataclass record") from error
    row = {index_name: _require_int64_count(index, label=index_name), **record_values}
    return _scalar_row(row, label=label)


def _int64_array(
    value: object,
    *,
    label: str,
    expected_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    """安全複製非負整數 ndarray；unsigned 超界在轉型前即拒絕。"""

    if not isinstance(value, np.ndarray) or value.dtype.kind not in "iu":
        raise ValueError(f"{label} 必須是非負整數 np.ndarray")
    if expected_shape is not None and value.shape != expected_shape:
        raise ValueError(f"{label} shape 必須精確為 {expected_shape}")
    if value.dtype.kind == "i" and bool(np.any(value < 0)):
        raise ValueError(f"{label} 不可含負值")
    if value.dtype.kind == "u" and bool(np.any(value > _INT64_MAX)):
        raise ValueError(f"{label} 不可超過 int64 上限")
    return np.array(value, dtype=np.int64, order="C", copy=True)


def _float64_array(
    value: object,
    *,
    label: str,
    expected_shape: tuple[int, ...] | None = None,
    nonnegative: bool = False,
) -> np.ndarray:
    """安全複製有限數值 ndarray 成 C-contiguous float64，拒絕非有限轉型結果。"""

    if not isinstance(value, np.ndarray) or value.dtype.kind not in "iuf":
        raise ValueError(f"{label} 必須是 integer 或 float np.ndarray")
    if expected_shape is not None and value.shape != expected_shape:
        raise ValueError(f"{label} shape 必須精確為 {expected_shape}")
    if not bool(np.all(np.isfinite(value))):
        raise ValueError(f"{label} 必須全部有限")
    if nonnegative and bool(np.any(value < 0)):
        raise ValueError(f"{label} 不可含負值")
    with np.errstate(over="ignore", invalid="ignore"):
        copied = np.array(value, dtype=np.float64, order="C", copy=True)
    if not bool(np.all(np.isfinite(copied))):
        raise ValueError(f"{label} 無法安全轉成有限 float64")
    return copied


def _count_array(values: list[int], *, label: str) -> np.ndarray:
    """逐一檢查 Python 計數後建立 int64 陣列，禁止 ``np.asarray`` 靜默溢位。"""

    checked = [
        _require_int64_count(value, label=f"{label}[{index}]")
        for index, value in enumerate(values)
    ]
    return np.array(checked, dtype=np.int64)


def _concatenate_1d(
    parts: list[np.ndarray],
    *,
    expected_length: int,
    label: str,
) -> np.ndarray:
    """依既定列順序串接一維片段，並核對最終 offset 所宣告的軸長。"""

    if not parts:
        raise ValueError(f"{label} 缺少可串接片段")
    result = np.concatenate(parts, axis=0)
    checked_length = _require_int64_count(expected_length, label=f"{label} 預期長度")
    if result.ndim != 1 or result.size != checked_length:
        raise ValueError(f"{label} 串接長度與最終 offset 不符")
    return result


def _all_dataclass_field_values(
    value: object,
    *,
    expected_type: type[object],
    label: str,
) -> dict[str, object]:
    """讀取指定公開 dataclass 的全部正式欄位，不遞迴轉成巢狀 dict。

    exact-type 檢查可避免 subclass 覆寫欄位存取或攜帶未受資料契約約束的狀態；回傳值
    只供對應公開 constructor 的 keyword arguments 使用。任何欄位存取或 constructor
    錯誤都原樣向外傳遞，不吞例外、不轉型，也不修補遭竄改的值。
    """

    if type(value) is not expected_type:
        raise ValueError(f"{label} 必須是 exact {expected_type.__name__} 實例")
    return {field.name: getattr(value, field.name) for field in fields(expected_type)}


def _revalidate_shard_binding(
    value: AggregateShardBinding,
    *,
    label: str,
) -> AggregateShardBinding:
    """以全部欄位重建 shard record，重跑安全相對路徑、SHA-256 與計數契約。"""

    return AggregateShardBinding(
        **_all_dataclass_field_values(
            value,
            expected_type=AggregateShardBinding,
            label=label,
        )
    )


def _revalidate_scenario_stratum(
    value: ScenarioStratum,
    *,
    label: str,
) -> ScenarioStratum:
    """以全部欄位重建 scenario 列，重跑座標、時間與動態初始條件整組政策。"""

    return ScenarioStratum(
        **_all_dataclass_field_values(
            value,
            expected_type=ScenarioStratum,
            label=label,
        )
    )


def _revalidate_aggregate_spec(value: AggregateSpec) -> AggregateSpec:
    """重建 AggregateSpec 及三類站點子規格，再重跑公尺制與拓撲契約。

    ``SiteGridSpec`` 重新驗證矩形公尺邊界；``SiteMetricCRSSpec`` 重新驗證投影中心、
    公尺單位與軸序；``SiteBoundarySegments`` 重新驗證 local／outer segment tuple。
    其餘 mapping、tuple、數值與 SHA 欄位原樣交由 ``AggregateSpec`` 公開 constructor
    防禦性複製及驗證，不從 aggregate array 反推或修補任何規格值。
    """

    values = _all_dataclass_field_values(
        value,
        expected_type=AggregateSpec,
        label="aggregate_spec",
    )
    values["site_grids"] = {
        site_id: SiteGridSpec(
            **_all_dataclass_field_values(
                grid,
                expected_type=SiteGridSpec,
                label=f"aggregate_spec.site_grids[{site_id!r}]",
            )
        )
        for site_id, grid in value.site_grids.items()
    }
    values["site_metric_crs"] = {
        site_id: SiteMetricCRSSpec(
            **_all_dataclass_field_values(
                metric_crs,
                expected_type=SiteMetricCRSSpec,
                label=f"aggregate_spec.site_metric_crs[{site_id!r}]",
            )
        )
        for site_id, metric_crs in value.site_metric_crs.items()
    }
    values["site_boundary_segment_ids"] = {
        site_id: SiteBoundarySegments(
            **_all_dataclass_field_values(
                segments,
                expected_type=SiteBoundarySegments,
                label=f"aggregate_spec.site_boundary_segment_ids[{site_id!r}]",
            )
        )
        for site_id, segments in value.site_boundary_segment_ids.items()
    }
    return AggregateSpec(**values)


def _revalidate_site_grid_counts(
    value: SiteEventGridCounts,
    *,
    label: str,
) -> SiteEventGridCounts:
    """以六個事件陣列重建站點格網，重跑 int64、非負、二維與同 shape 驗證。"""

    return SiteEventGridCounts(
        **_all_dataclass_field_values(
            value,
            expected_type=SiteEventGridCounts,
            label=label,
        )
    )


def _revalidate_boundary_key(value: object, *, label: str) -> BoundaryAggregateKey:
    """以三個公開欄位重建 boundary key。"""

    return BoundaryAggregateKey(
        **_all_dataclass_field_values(
            value,
            expected_type=BoundaryAggregateKey,
            label=label,
        )
    )


def _revalidate_source_key(value: object, *, label: str) -> SourceReceptorAggregateKey:
    """以四個公開欄位重建 source-receptor key。"""

    return SourceReceptorAggregateKey(
        **_all_dataclass_field_values(
            value,
            expected_type=SourceReceptorAggregateKey,
            label=label,
        )
    )


def _revalidate_cross_key(value: object, *, label: str) -> CrossSiteAggregateKey:
    """以來源與目標站點欄位重建 cross-site key。"""

    return CrossSiteAggregateKey(
        **_all_dataclass_field_values(
            value,
            expected_type=CrossSiteAggregateKey,
            label=label,
        )
    )


def _revalidate_receptor_key(value: object, *, label: str) -> ReceptorAggregateKey:
    """以站點與受體欄位重建 denominator key。"""

    return ReceptorAggregateKey(
        **_all_dataclass_field_values(
            value,
            expected_type=ReceptorAggregateKey,
            label=label,
        )
    )


def _revalidate_keyed_mapping(
    value: object,
    *,
    key_kind: str,
    label: str,
) -> dict[object, object]:
    """逐 key 呼叫對應公開 constructor，value 原樣交給外層 aggregate constructor。

    scalar 與 ndarray value 不在此 helper 複製或轉型；``EventAggregateChunk`` 會依欄位
    語意完成非負、有限、shape 與防禦性複製。重建後若兩個原 key 碰撞，必須拒絕而非
    讓 dict 靜默覆蓋，避免竄改資料列被當成同一拓撲。
    """

    if not isinstance(value, Mapping):
        raise ValueError(f"{label} 必須是 mapping")
    rebuilt: dict[object, object] = {}
    for index, (key, item) in enumerate(value.items()):
        key_label = f"{label} key[{index}]"
        if key_kind == "boundary":
            rebuilt_key = _revalidate_boundary_key(key, label=key_label)
        elif key_kind == "source":
            rebuilt_key = _revalidate_source_key(key, label=key_label)
        elif key_kind == "cross":
            rebuilt_key = _revalidate_cross_key(key, label=key_label)
        elif key_kind == "receptor":
            rebuilt_key = _revalidate_receptor_key(key, label=key_label)
        else:
            raise RuntimeError(f"未知的 event key kind：{key_kind}")
        if rebuilt_key in rebuilt:
            raise ValueError(f"{label} 的 key 重建後不得重複")
        rebuilt[rebuilt_key] = item
    return rebuilt


def _revalidate_event_aggregate(value: EventAggregateChunk) -> EventAggregateChunk:
    """深層重建 event container、六格網與四類 dataclass key。

    所有 mapping value 均保持原值，交由 ``SiteEventGridCounts`` 或
    ``EventAggregateChunk`` 公開 constructor 重新複製並驗證；此流程不使用 event
    模組的私有 helper，不吞 constructor 例外，也不以零值補齊缺少的 key／array。
    """

    values = _all_dataclass_field_values(
        value,
        expected_type=EventAggregateChunk,
        label="event_aggregate",
    )
    if not isinstance(value.site_grid_counts, Mapping):
        raise ValueError("event_aggregate.site_grid_counts 必須是 mapping")
    values["site_grid_counts"] = {
        site_id: _revalidate_site_grid_counts(
            grid_counts,
            label=f"event_aggregate.site_grid_counts[{site_id!r}]",
        )
        for site_id, grid_counts in value.site_grid_counts.items()
    }
    values["boundary_bin_edges_m"] = _revalidate_keyed_mapping(
        value.boundary_bin_edges_m,
        key_kind="boundary",
        label="event_aggregate.boundary_bin_edges_m",
    )
    values["boundary_arclength_raw_count"] = _revalidate_keyed_mapping(
        value.boundary_arclength_raw_count,
        key_kind="boundary",
        label="event_aggregate.boundary_arclength_raw_count",
    )
    values["boundary_travel_age_histogram"] = _revalidate_keyed_mapping(
        value.boundary_travel_age_histogram,
        key_kind="boundary",
        label="event_aggregate.boundary_travel_age_histogram",
    )
    values["source_receptor_raw_count"] = _revalidate_keyed_mapping(
        value.source_receptor_raw_count,
        key_kind="source",
        label="event_aggregate.source_receptor_raw_count",
    )
    values["source_receptor_travel_age_histogram"] = _revalidate_keyed_mapping(
        value.source_receptor_travel_age_histogram,
        key_kind="source",
        label="event_aggregate.source_receptor_travel_age_histogram",
    )
    values["cross_site_unique_member_count"] = _revalidate_keyed_mapping(
        value.cross_site_unique_member_count,
        key_kind="cross",
        label="event_aggregate.cross_site_unique_member_count",
    )
    values["valid_member_denominator_by_receptor"] = _revalidate_keyed_mapping(
        value.valid_member_denominator_by_receptor,
        key_kind="receptor",
        label="event_aggregate.valid_member_denominator_by_receptor",
    )
    return EventAggregateChunk(**values)


def _revalidate_pathway(
    value: StreamingPathwayAggregate,
    *,
    label: str,
) -> StreamingPathwayAggregate:
    """以全部欄位重建 pathway，重跑公尺格線、秒數、shape 與守恆驗證。"""

    return StreamingPathwayAggregate(
        **_all_dataclass_field_values(
            value,
            expected_type=StreamingPathwayAggregate,
            label=label,
        )
    )


def _revalidate_payload(payload: AggregateReleasePayload) -> AggregateReleasePayload:
    """深層重建全部公開容器後，再重建最外層 payload。

    重建範圍涵蓋 shard／scenario records、AggregateSpec 站點子規格、event 的站點格網
    與 boundary／source／cross／receptor keys，以及每站 StreamingPathwayAggregate。
    最後的 ``AggregateReleasePayload`` constructor 使用這些新物件重跑跨產品站點、粒子
    分母、age 軸與 shape 契約，防止任何內外層 frozen dataclass 被低階賦值後直接編碼。
    """

    values = _all_dataclass_field_values(
        payload,
        expected_type=AggregateReleasePayload,
        label="payload",
    )
    values["aggregate_spec"] = _revalidate_aggregate_spec(payload.aggregate_spec)
    values["shard_bindings"] = tuple(
        _revalidate_shard_binding(shard, label=f"shard_bindings[{index}]")
        for index, shard in enumerate(payload.shard_bindings)
    )
    values["scenario_strata"] = tuple(
        _revalidate_scenario_stratum(stratum, label=f"scenario_strata[{index}]")
        for index, stratum in enumerate(payload.scenario_strata)
    )
    values["event_aggregate"] = _revalidate_event_aggregate(payload.event_aggregate)
    if not isinstance(payload.pathway_by_site, Mapping):
        raise ValueError("payload.pathway_by_site 必須是 mapping")
    values["pathway_by_site"] = {
        site_id: _revalidate_pathway(
            pathway,
            label=f"pathway_by_site[{site_id!r}]",
        )
        for site_id, pathway in payload.pathway_by_site.items()
    }
    return AggregateReleasePayload(**values)


def encode_aggregate_release_payload(
    payload: AggregateReleasePayload,
) -> EncodedAggregateProducts:
    """將完整 payload 編成固定九表與十八陣列的純記憶體產品。

    函式起始先重建 payload，確保 frozen dataclass 若曾被 ``object.__setattr__`` 竄改，
    仍須重新通過 constructor。資料列排序、半開 offset、C-order 展平與公尺／秒單位均
    依 release codec 契約固定；本函式不做任何檔案 I/O、manifest 寫入或缺值補建。
    """

    validated = _revalidate_payload(payload)
    return _encode_validated_payload(validated)


def _encode_validated_payload(
    payload: AggregateReleasePayload,
) -> EncodedAggregateProducts:
    """組裝固定拓撲；各 helper 只接受已通過 payload constructor 的資料。"""

    shard_rows = tuple(
        _indexed_dataclass_row(
            shard,
            index_name="shard_index",
            index=index,
            label=f"shard_bindings[{index}]",
        )
        for index, shard in enumerate(payload.shard_bindings)
    )
    scenario_rows = tuple(
        _indexed_dataclass_row(
            stratum,
            index_name="scenario_index",
            index=index,
            label=f"scenario_strata[{index}]",
        )
        for index, stratum in enumerate(payload.scenario_strata)
    )

    site_rows, site_arrays, site_ids, age_bin_count = _encode_site_products(payload)
    boundary_rows, boundary_arrays = _encode_boundary_products(
        payload,
        site_ids=site_ids,
        age_bin_count=age_bin_count,
    )
    source_rows, source_arrays = _encode_source_receptor_products(
        payload,
        site_ids=site_ids,
        age_bin_count=age_bin_count,
    )
    cross_rows, outcome_rows, site_denominator_rows, receptor_denominator_rows = (
        _encode_count_tables(payload, site_ids=site_ids)
    )

    tables = {
        "shard_bindings.parquet": shard_rows,
        "scenario_strata.parquet": scenario_rows,
        "site_index.parquet": site_rows,
        "boundary_index.parquet": boundary_rows,
        "source_receptor_index.parquet": source_rows,
        "cross_site_counts.parquet": cross_rows,
        "outcome_counts.parquet": outcome_rows,
        "site_denominators.parquet": site_denominator_rows,
        "receptor_denominators.parquet": receptor_denominator_rows,
    }
    arrays = {**site_arrays, **boundary_arrays, **source_arrays}

    # 固定集合比對是 codec 內部的完整性斷言；EncodedAggregateProducts 還會再次執行
    # 相同 exact-key 與安全數值封存，形成組裝與容器兩層獨立防線。
    if frozenset(tables) != AGGREGATE_RELEASE_TABLE_FILES:
        raise RuntimeError("encoder 產生的 table topology 與固定契約不符")
    if frozenset(arrays) != AGGREGATE_RELEASE_ARRAY_FILES:
        raise RuntimeError("encoder 產生的 array topology 與固定契約不符")
    return EncodedAggregateProducts(tables=tables, arrays=arrays)


def metadata_from_payload(
    payload: AggregateReleasePayload,
) -> AggregateReleaseMetadata:
    """由深層重建後的 payload 與 canonical 表格建立 release metadata。

    函式先以 ``_revalidate_payload`` 重新建構 spec、shard、scenario、event、pathway
    及其巢狀 key／陣列，再以 ``_encode_validated_payload`` 取得唯一固定列序的
    canonical 產品；因此 metadata 的 row count 不是從 caller 的可變 mapping 猜測，
    也不會繞過 payload 或 encoder 的跨產品驗證。``input_particle_count`` 直接取自
    重建後 event，五個 row count 直接取自 shard、scenario、site、boundary 與
    source-receptor 五張固定表的實際 ``len``；其餘 provenance 與 run 欄位直接取自
    重建後 payload，AggregateSpec 的 source／canonical hash 直接取自重建後 spec。

    本函式只建立記憶體內 typed metadata，不讀取或建立 writer、manifest、Parquet
    或 NPY。row count 表示固定拓撲的列數，不代表事件發生量、有效分母或科學上的
    絕對來源機率；缺值、資料缺口、乾點與數值失敗仍由 payload 的原始欄位分開保存。
    """

    validated = _revalidate_payload(payload)
    canonical_products = _encode_validated_payload(validated)
    event = validated.event_aggregate
    spec = validated.aggregate_spec
    tables = canonical_products.tables

    return AggregateReleaseMetadata(
        schema_version=validated.schema_version,
        run_id=validated.run_id,
        run_kind=validated.run_kind,
        experiment_case_id=validated.experiment_case_id,
        members_per_scenario=validated.members_per_scenario,
        config_hash=validated.config_hash,
        checkpoint_input_binding_hash=validated.checkpoint_input_binding_hash,
        source_run_plan_sha256=validated.source_run_plan_sha256,
        source_run_progress_sha256=validated.source_run_progress_sha256,
        source_normalized_config_sha256=validated.source_normalized_config_sha256,
        source_input_inventory_sha256=validated.source_input_inventory_sha256,
        aggregate_spec_source_sha256=spec.source_sha256,
        aggregate_spec_canonical_sha256=spec.canonical_sha256,
        input_particle_count=event.input_particle_count,
        shard_row_count=len(tables["shard_bindings.parquet"]),
        scenario_row_count=len(tables["scenario_strata.parquet"]),
        site_row_count=len(tables["site_index.parquet"]),
        boundary_row_count=len(tables["boundary_index.parquet"]),
        source_receptor_row_count=len(tables["source_receptor_index.parquet"]),
    )


def decode_aggregate_release_payload(
    metadata: AggregateReleaseMetadata,
    aggregate_spec: AggregateSpec,
    products: EncodedAggregateProducts,
) -> AggregateReleasePayload:
    """由固定表格與陣列 fail-closed 還原完整的 aggregate release payload。

    ``metadata``、``aggregate_spec`` 與 ``products`` 是 manifest validator 與記憶體
    codec 之間的 typed boundary；三者必須是 exact 公開型別，且本函式會讀取所有正式
    欄位後重新呼叫公開 constructor，不能因 frozen dataclass 或唯讀容器而直接信任
    現成 reference。表格列的欄位集合、欄位順序、原生 Python scalar 型別、排序、
    index、offset 與零值拓撲都逐項檢查；格網軸固定為公尺制的 ``(y_cell, x_cell)``，
    pathway 與事件的 age 軸固定為秒，邊界 travel histogram 軸為
    ``(s_bin, age_bin)``，source-receptor histogram 則只有 age 軸。

    還原只依 ``AggregateSpec`` 重建 pathway x/y 公尺格線與 canonical boundary edges，
    不從陣列猜測、轉置、clip、padding、最近值或零值補建資料。``None`` 只允許出現
    在 ``ScenarioStratum`` 的整組動態初始條件欄位；缺值、乾點、域外、時間缺口與
    數值失敗仍由各自固定欄位保存。完成最外層 payload 第二層 constructor 後，會再以
    ``metadata_from_payload`` 與重新 encode 的逐列／逐 array exact equality 作最後
    round-trip gate；任何不一致都以 ``ValueError`` fail closed。輸出仍只是條件式
    來源足跡或相對來源權重的資料載體，並不構成絕對來源機率或因果歸因。

    Args:
        metadata: 已由 ``AggregateReleaseMetadata`` constructor 驗證的固定 manifest
            metadata；其中 source／canonical spec SHA 與 run_id 必須和 spec 一致。
        aggregate_spec: 已驗證的 ``AggregateSpec``，其格線與 boundary 拓撲是解碼時
            唯一允許的 canonical 科學規格來源。
        products: 已由 ``EncodedAggregateProducts`` 封存的九張表與十八個陣列；本函式
            仍會逐欄與逐值檢查，不把容器層的 snapshot 視為語意驗證。

    Returns:
        通過完整跨產品與 round-trip 驗證的新建 ``AggregateReleasePayload``。

    Raises:
        ValueError: 任一型別、欄位、scalar、排序、拓撲、shape、offset、dtype、值、
            provenance 或 round-trip equality 不符合固定 release 契約。
    """

    try:
        return _decode_aggregate_release_payload_impl(
            metadata=metadata,
            aggregate_spec=aggregate_spec,
            products=products,
        )
    except ValueError:
        raise
    except Exception as error:
        # decoder 對外只暴露資料契約的 ValueError；底層 mapping、NumPy reshape 或
        # constructor 若因遭竄改而丟出其他例外，不把 Python implementation detail
        # 洩漏成不穩定的 reader API。原始例外保留為 cause 供除錯，但 user-visible
        # 訊息固定且不含本機路徑、manifest 或任何 I/O 資訊。
        raise ValueError("aggregate release payload 解碼失敗；資料不符合固定 codec 契約") from error


def _encode_site_products(
    payload: AggregateReleasePayload,
) -> tuple[
    tuple[dict[str, object], ...],
    dict[str, np.ndarray],
    tuple[str, ...],
    int,
]:
    """依站點字典序建立 site_index、cell offsets 與九個站點陣列。

    六類事件與 pathway unique/residence 以 ``(y_cell, x_cell)`` C-order 展平成
    ``[cell_start_offset, cell_stop_offset)``。first-passage 原 shape 是
    ``(y_cell, x_cell, age_bin)``，同樣以 C-order 展平，因此每站實際片段是上述
    cell offset 乘以 ``age_bin_count``。所有時間值以秒保存；不從 residence 陣列
    反推或覆寫 pathway 的 input／allocated interval scalar。
    """

    spec = payload.aggregate_spec
    event = payload.event_aggregate
    age_edges = _float64_array(
        event.age_bin_edges_seconds,
        label="event_aggregate.age_bin_edges_seconds",
        nonnegative=True,
    )
    if (
        age_edges.ndim != 1
        or age_edges.size < 2
        or age_edges[0] != 0.0
        or not bool(np.all(np.diff(age_edges) > 0.0))
    ):
        raise ValueError("age_bin_edges_seconds 必須是一維、從 0 秒開始且嚴格遞增")
    age_bin_count = _require_int64_count(
        int(age_edges.size) - 1,
        label="age_bin_count",
    )

    site_ids = tuple(sorted(spec.site_grids))
    if not site_ids:
        raise ValueError("site_index 不得為空")

    analysis_regions_by_site = {site_id: set() for site_id in site_ids}
    scenario_count_by_site = {site_id: 0 for site_id in site_ids}
    for stratum in payload.scenario_strata:
        site_id = stratum.study_site_id
        if site_id not in analysis_regions_by_site:
            raise ValueError(f"scenario site {site_id!r} 不存在於 site_index 拓撲")
        analysis_regions_by_site[site_id].add(stratum.analysis_region_id)
        next_count = scenario_count_by_site[site_id] + 1
        scenario_count_by_site[site_id] = _require_int64_count(
            next_count,
            label=f"site {site_id!r} scenario_count",
        )

    event_parts: dict[str, list[np.ndarray]] = {
        file_name: [] for file_name, _ in _EVENT_GRID_PRODUCTS
    }
    pathway_unique_parts: list[np.ndarray] = []
    pathway_residence_parts: list[np.ndarray] = []
    pathway_first_passage_parts: list[np.ndarray] = []
    site_offsets = [0]
    rows: list[dict[str, object]] = []

    for site_index, site_id in enumerate(site_ids):
        region_ids = analysis_regions_by_site[site_id]
        if len(region_ids) != 1:
            raise ValueError(
                f"site {site_id!r} 的 scenario strata 必須恰有一個 analysis_region_id"
            )
        analysis_region_id = next(iter(region_ids))

        grid_counts = event.site_grid_counts[site_id]
        reference = _int64_array(
            grid_counts.local_first_exit_count,
            label=f"event.site_grid_counts[{site_id!r}].local_first_exit_count",
        )
        if reference.ndim != 2:
            raise ValueError(f"site {site_id!r} 的事件格網必須是 (y_cell, x_cell) 二維陣列")
        y_cell_count = _require_int64_count(
            int(reference.shape[0]),
            label=f"site {site_id!r} y_cell_count",
        )
        x_cell_count = _require_int64_count(
            int(reference.shape[1]),
            label=f"site {site_id!r} x_cell_count",
        )
        cell_count = _checked_count_product(
            y_cell_count,
            x_cell_count,
            label=f"site {site_id!r} cell_count",
        )
        cell_start, cell_stop = _append_nonempty_offset(
            site_offsets,
            cell_count,
            label=f"site {site_id!r} cell segment",
        )
        expected_cell_shape = (y_cell_count, x_cell_count)

        for file_name, field_name in _EVENT_GRID_PRODUCTS:
            values = _int64_array(
                getattr(grid_counts, field_name),
                label=f"event.site_grid_counts[{site_id!r}].{field_name}",
                expected_shape=expected_cell_shape,
            )
            event_parts[file_name].append(values.reshape(-1, order="C"))

        pathway = payload.pathway_by_site[site_id]
        pathway_age_edges = _float64_array(
            pathway.age_bin_edges_seconds,
            label=f"pathway_by_site[{site_id!r}].age_bin_edges_seconds",
            expected_shape=age_edges.shape,
            nonnegative=True,
        )
        if not np.array_equal(pathway_age_edges, age_edges):
            raise ValueError(f"site {site_id!r} 的 pathway age edges 與 release age 軸不一致")

        pathway_unique = _int64_array(
            pathway.unique_particle_count,
            label=f"pathway_by_site[{site_id!r}].unique_particle_count",
            expected_shape=expected_cell_shape,
        )
        pathway_residence = _float64_array(
            pathway.residence_time_seconds,
            label=f"pathway_by_site[{site_id!r}].residence_time_seconds",
            expected_shape=expected_cell_shape,
            nonnegative=True,
        )
        expected_histogram_shape = (*expected_cell_shape, age_bin_count)
        pathway_first_passage = _int64_array(
            pathway.first_passage_age_histogram,
            label=f"pathway_by_site[{site_id!r}].first_passage_age_histogram",
            expected_shape=expected_histogram_shape,
        )
        pathway_unique_parts.append(pathway_unique.reshape(-1, order="C"))
        pathway_residence_parts.append(pathway_residence.reshape(-1, order="C"))
        pathway_first_passage_parts.append(pathway_first_passage.reshape(-1, order="C"))

        scenario_count = _require_int64_count(
            scenario_count_by_site[site_id],
            label=f"site {site_id!r} scenario_count",
        )
        expected_member_count = _checked_count_product(
            scenario_count,
            payload.members_per_scenario,
            label=f"site {site_id!r} scenario_count×members_per_scenario",
        )
        total_member_count = _require_int64_count(
            event.total_member_count_by_site[site_id],
            label=f"site {site_id!r} total_member_count",
        )
        pathway_particle_count = _require_int64_count(
            pathway.input_particle_count,
            label=f"site {site_id!r} pathway_input_particle_count",
        )
        if total_member_count != expected_member_count or pathway_particle_count != expected_member_count:
            raise ValueError(
                f"site {site_id!r} 的 scenario、event 與 pathway 粒子數必須精確一致"
            )
        valid_denominator = _require_int64_count(
            event.valid_member_denominator_by_site[site_id],
            label=f"site {site_id!r} valid_member_denominator",
        )
        if valid_denominator > total_member_count:
            raise ValueError(f"site {site_id!r} 的有效分母不可超過總成員數")

        grid = spec.site_grids[site_id]
        metric_crs = spec.site_metric_crs[site_id]
        row = {
            "site_index": _require_int64_count(site_index, label="site_index"),
            "study_site_id": site_id,
            "analysis_region_id": analysis_region_id,
            "x_min_m": _require_finite_float(grid.x_min_m, label=f"site {site_id!r} x_min_m"),
            "x_max_m": _require_finite_float(grid.x_max_m, label=f"site {site_id!r} x_max_m"),
            "y_min_m": _require_finite_float(grid.y_min_m, label=f"site {site_id!r} y_min_m"),
            "y_max_m": _require_finite_float(grid.y_max_m, label=f"site {site_id!r} y_max_m"),
            "x_cell_count": x_cell_count,
            "y_cell_count": y_cell_count,
            "cell_start_offset": cell_start,
            "cell_stop_offset": cell_stop,
            "projection_method": metric_crs.projection_method,
            "center_lon_deg": _require_finite_float(
                metric_crs.center_lon_deg,
                label=f"site {site_id!r} center_lon_deg",
            ),
            "center_lat_deg": _require_finite_float(
                metric_crs.center_lat_deg,
                label=f"site {site_id!r} center_lat_deg",
            ),
            "linear_unit": metric_crs.linear_unit,
            "axis_order": metric_crs.axis_order,
            "scenario_count": scenario_count,
            "total_member_count": total_member_count,
            "valid_member_denominator": valid_denominator,
            "pathway_input_particle_count": pathway_particle_count,
            "pathway_input_interval_seconds": _require_nonnegative_finite_float(
                pathway.input_interval_seconds,
                label=f"site {site_id!r} pathway_input_interval_seconds",
            ),
            "pathway_allocated_interval_seconds": _require_nonnegative_finite_float(
                pathway.allocated_interval_seconds,
                label=f"site {site_id!r} pathway_allocated_interval_seconds",
            ),
        }
        rows.append(_scalar_row(row, label=f"site_index[{site_index}]"))

    total_cells = site_offsets[-1]
    total_cell_age_values = _checked_count_product(
        total_cells,
        age_bin_count,
        label="pathway_first_passage_age_histogram 總長度",
    )
    arrays = {
        "age_bin_edges_seconds.npy": age_edges,
        "site_cell_offsets.npy": _count_array(site_offsets, label="site_cell_offsets"),
        **{
            file_name: _concatenate_1d(
                parts,
                expected_length=total_cells,
                label=file_name,
            )
            for file_name, parts in event_parts.items()
        },
        "pathway_unique_particle_count.npy": _concatenate_1d(
            pathway_unique_parts,
            expected_length=total_cells,
            label="pathway_unique_particle_count.npy",
        ),
        "pathway_residence_time_seconds.npy": _concatenate_1d(
            pathway_residence_parts,
            expected_length=total_cells,
            label="pathway_residence_time_seconds.npy",
        ),
        "pathway_first_passage_age_histogram.npy": _concatenate_1d(
            pathway_first_passage_parts,
            expected_length=total_cell_age_values,
            label="pathway_first_passage_age_histogram.npy",
        ),
    }
    return tuple(rows), arrays, site_ids, age_bin_count


def _encode_boundary_products(
    payload: AggregateReleasePayload,
    *,
    site_ids: tuple[str, ...],
    age_bin_count: int,
) -> tuple[tuple[dict[str, object], ...], dict[str, np.ndarray]]:
    """依 ``(site, kind, segment)`` 建立邊界索引與公尺／秒陣列。

    edge 與 s-bin 各自使用長度為 ``boundary_count+1`` 的半開 offset。邊界 edges 以
    公尺串接且保留每段端點，不跨段去重；raw count 依 s-bin 串接，travel-age 則把
    ``(s_bin, age_bin)`` 以 C-order 展平成一維。``segment_length_m`` 只取自
    ``AggregateSpec``，不得以最後一個 edge 猜測或覆寫。
    """

    event = payload.event_aggregate
    edge_keys = set(event.boundary_bin_edges_m)
    if set(event.boundary_arclength_raw_count) != edge_keys:
        raise ValueError("boundary raw count key set 必須與 boundary edges 完全一致")
    if set(event.boundary_travel_age_histogram) != edge_keys:
        raise ValueError("boundary travel-age key set 必須與 boundary edges 完全一致")
    boundary_keys = tuple(
        sorted(
            edge_keys,
            key=lambda key: (
                key.study_site_id,
                key.boundary_kind,
                key.boundary_segment_id,
            ),
        )
    )
    if not boundary_keys:
        raise ValueError("boundary_index 不得為空")

    site_set = set(site_ids)
    edge_offsets = [0]
    bin_offsets = [0]
    edge_parts: list[np.ndarray] = []
    raw_parts: list[np.ndarray] = []
    travel_parts: list[np.ndarray] = []
    rows: list[dict[str, object]] = []

    for boundary_index, key in enumerate(boundary_keys):
        if key.study_site_id not in site_set:
            raise ValueError(f"boundary site {key.study_site_id!r} 不存在於 site_index")
        try:
            segment_length_value = payload.aggregate_spec.boundary_segment_lengths_m[
                key.boundary_segment_id
            ]
        except KeyError as error:
            raise ValueError(
                f"boundary segment {key.boundary_segment_id!r} 缺少 AggregateSpec 長度"
            ) from error
        segment_length_m = _require_nonnegative_finite_float(
            segment_length_value,
            label=f"boundary {key.boundary_segment_id!r} segment_length_m",
        )
        if segment_length_m <= 0.0:
            raise ValueError(f"boundary {key.boundary_segment_id!r} 長度必須大於 0 公尺")

        edges = _float64_array(
            event.boundary_bin_edges_m[key],
            label=f"boundary_bin_edges_m[{key!r}]",
            nonnegative=True,
        )
        if edges.ndim != 1 or edges.size < 2 or not bool(np.all(np.diff(edges) > 0.0)):
            raise ValueError(f"boundary {key!r} edges 必須是一維且至少形成一個嚴格遞增 bin")
        edge_count = _require_int64_count(
            int(edges.size),
            label=f"boundary {key!r} edge_count",
        )
        bin_count = _require_int64_count(
            edge_count - 1,
            label=f"boundary {key!r} bin_count",
        )
        edge_start, edge_stop = _append_nonempty_offset(
            edge_offsets,
            edge_count,
            label=f"boundary {key!r} edge segment",
        )
        bin_start, bin_stop = _append_nonempty_offset(
            bin_offsets,
            bin_count,
            label=f"boundary {key!r} bin segment",
        )

        raw = _int64_array(
            event.boundary_arclength_raw_count[key],
            label=f"boundary_arclength_raw_count[{key!r}]",
            expected_shape=(bin_count,),
        )
        travel = _int64_array(
            event.boundary_travel_age_histogram[key],
            label=f"boundary_travel_age_histogram[{key!r}]",
            expected_shape=(bin_count, age_bin_count),
        )
        edge_parts.append(edges.reshape(-1, order="C"))
        raw_parts.append(raw.reshape(-1, order="C"))
        travel_parts.append(travel.reshape(-1, order="C"))

        row = {
            "boundary_index": _require_int64_count(boundary_index, label="boundary_index"),
            "study_site_id": key.study_site_id,
            "boundary_kind": key.boundary_kind,
            "boundary_segment_id": key.boundary_segment_id,
            "segment_length_m": segment_length_m,
            "edge_start_offset": edge_start,
            "edge_stop_offset": edge_stop,
            "bin_start_offset": bin_start,
            "bin_stop_offset": bin_stop,
        }
        rows.append(_scalar_row(row, label=f"boundary_index[{boundary_index}]"))

    total_edges = edge_offsets[-1]
    total_bins = bin_offsets[-1]
    total_travel_values = _checked_count_product(
        total_bins,
        age_bin_count,
        label="boundary_travel_age_histogram 總長度",
    )
    arrays = {
        "boundary_edge_offsets.npy": _count_array(
            edge_offsets,
            label="boundary_edge_offsets",
        ),
        "boundary_bin_edges_m.npy": _concatenate_1d(
            edge_parts,
            expected_length=total_edges,
            label="boundary_bin_edges_m.npy",
        ),
        "boundary_bin_offsets.npy": _count_array(
            bin_offsets,
            label="boundary_bin_offsets",
        ),
        "boundary_arclength_raw_count.npy": _concatenate_1d(
            raw_parts,
            expected_length=total_bins,
            label="boundary_arclength_raw_count.npy",
        ),
        "boundary_travel_age_histogram.npy": _concatenate_1d(
            travel_parts,
            expected_length=total_travel_values,
            label="boundary_travel_age_histogram.npy",
        ),
    }
    return tuple(rows), arrays


def _encode_source_receptor_products(
    payload: AggregateReleasePayload,
    *,
    site_ids: tuple[str, ...],
    age_bin_count: int,
) -> tuple[tuple[dict[str, object], ...], dict[str, np.ndarray]]:
    """依 ``(site, receptor, kind, segment)`` 建立來源—受體列與一維計數陣列。

    第 i 列的 raw count 位於一維陣列索引 i；travel-age 使用緊鄰的固定長度
    age-bin 片段 ``[i*age_bin_count, (i+1)*age_bin_count)``。codec 不因零事件刪列，
    也不為缺少的 key 或 histogram 補零。
    """

    event = payload.event_aggregate
    raw_keys = set(event.source_receptor_raw_count)
    if set(event.source_receptor_travel_age_histogram) != raw_keys:
        raise ValueError("source-receptor raw 與 travel-age key set 必須完全一致")
    source_keys = tuple(
        sorted(
            raw_keys,
            key=lambda key: (
                key.study_site_id,
                key.receptor_id,
                key.boundary_kind,
                key.boundary_segment_id,
            ),
        )
    )
    if not source_keys:
        raise ValueError("source_receptor_index 不得為空；零事件仍須保留拓撲列")

    site_set = set(site_ids)
    rows: list[dict[str, object]] = []
    raw_counts: list[int] = []
    travel_parts: list[np.ndarray] = []
    for source_index, key in enumerate(source_keys):
        if key.study_site_id not in site_set:
            raise ValueError(f"source-receptor site {key.study_site_id!r} 不存在於 site_index")
        row = {
            "source_receptor_index": _require_int64_count(
                source_index,
                label="source_receptor_index",
            ),
            "study_site_id": key.study_site_id,
            "receptor_id": key.receptor_id,
            "boundary_kind": key.boundary_kind,
            "boundary_segment_id": key.boundary_segment_id,
        }
        rows.append(_scalar_row(row, label=f"source_receptor_index[{source_index}]"))
        raw_counts.append(
            _require_int64_count(
                event.source_receptor_raw_count[key],
                label=f"source_receptor_raw_count[{key!r}]",
            )
        )
        travel = _int64_array(
            event.source_receptor_travel_age_histogram[key],
            label=f"source_receptor_travel_age_histogram[{key!r}]",
            expected_shape=(age_bin_count,),
        )
        travel_parts.append(travel.reshape(-1, order="C"))

    source_count = _require_int64_count(len(source_keys), label="source_receptor_count")
    total_travel_values = _checked_count_product(
        source_count,
        age_bin_count,
        label="source_receptor_travel_age_histogram 總長度",
    )
    arrays = {
        "source_receptor_raw_count.npy": _count_array(
            raw_counts,
            label="source_receptor_raw_count",
        ),
        "source_receptor_travel_age_histogram.npy": _concatenate_1d(
            travel_parts,
            expected_length=total_travel_values,
            label="source_receptor_travel_age_histogram.npy",
        ),
    }
    if arrays["source_receptor_raw_count.npy"].size != source_count:
        raise ValueError("source_receptor_raw_count 長度與 index 列數不符")
    return tuple(rows), arrays


def _encode_count_tables(
    payload: AggregateReleasePayload,
    *,
    site_ids: tuple[str, ...],
) -> tuple[
    tuple[dict[str, object], ...],
    tuple[dict[str, object], ...],
    tuple[dict[str, object], ...],
    tuple[dict[str, object], ...],
]:
    """建立 deterministic cross-site、outcome 與兩層 denominator 表。

    所有 count／denominator 在進入列之前先以 Python int 檢查非負 int64 範圍。單站
    release 的 cross-site 表可為零列；多站 release 若完全沒有 cross-site 拓撲則拒絕，
    不能把缺列解讀成零事件。outcome 即使 count 為零也必須保留明示狀態列。
    """

    event = payload.event_aggregate
    site_set = set(site_ids)

    cross_keys = tuple(
        sorted(
            event.cross_site_unique_member_count,
            key=lambda key: (
                key.source_study_site_id,
                key.target_study_site_id,
            ),
        )
    )
    if len(site_ids) > 1 and not cross_keys:
        raise ValueError("多站 release 的 cross_site_counts 不得缺少拓撲列")
    cross_rows: list[dict[str, object]] = []
    for index, key in enumerate(cross_keys):
        if (
            key.source_study_site_id not in site_set
            or key.target_study_site_id not in site_set
        ):
            raise ValueError(f"cross_site_counts[{index}] 引用未知站點")
        cross_rows.append(
            _scalar_row(
                {
                    "source_study_site_id": key.source_study_site_id,
                    "target_study_site_id": key.target_study_site_id,
                    "unique_member_count": _require_int64_count(
                        event.cross_site_unique_member_count[key],
                        label=f"cross_site_unique_member_count[{key!r}]",
                    ),
                },
                label=f"cross_site_counts[{index}]",
            )
        )

    outcome_rows: list[dict[str, object]] = []
    for site_id in site_ids:
        counts_by_outcome = event.outcome_count_by_site[site_id]
        if not counts_by_outcome:
            raise ValueError(f"site {site_id!r} 的 outcome topology 不得為空")
        for outcome in sorted(counts_by_outcome):
            if type(outcome) is not str or not outcome:
                raise ValueError(f"site {site_id!r} 的 outcome 必須是非空原生 str")
            outcome_rows.append(
                _scalar_row(
                    {
                        "study_site_id": site_id,
                        "outcome": outcome,
                        "count": _require_int64_count(
                            counts_by_outcome[outcome],
                            label=f"outcome_count_by_site[{site_id!r}][{outcome!r}]",
                        ),
                    },
                    label=f"outcome_counts[{len(outcome_rows)}]",
                )
            )

    site_denominator_rows = tuple(
        _scalar_row(
            {
                "study_site_id": site_id,
                "valid_member_denominator": _require_int64_count(
                    event.valid_member_denominator_by_site[site_id],
                    label=f"valid_member_denominator_by_site[{site_id!r}]",
                ),
                "total_member_count": _require_int64_count(
                    event.total_member_count_by_site[site_id],
                    label=f"total_member_count_by_site[{site_id!r}]",
                ),
            },
            label=f"site_denominators[{index}]",
        )
        for index, site_id in enumerate(site_ids)
    )

    receptor_keys = tuple(
        sorted(
            event.valid_member_denominator_by_receptor,
            key=lambda key: (key.study_site_id, key.receptor_id),
        )
    )
    if not receptor_keys:
        raise ValueError("receptor_denominators 不得為空")
    receptor_rows = tuple(
        _scalar_row(
            {
                "study_site_id": key.study_site_id,
                "receptor_id": key.receptor_id,
                "valid_member_denominator": _require_int64_count(
                    event.valid_member_denominator_by_receptor[key],
                    label=f"valid_member_denominator_by_receptor[{key!r}]",
                ),
            },
            label=f"receptor_denominators[{index}]",
        )
        for index, key in enumerate(receptor_keys)
    )
    for index, key in enumerate(receptor_keys):
        if key.study_site_id not in site_set:
            raise ValueError(f"receptor_denominators[{index}] 引用未知站點")

    return (
        tuple(cross_rows),
        tuple(outcome_rows),
        site_denominator_rows,
        receptor_rows,
    )


def _revalidate_metadata(value: object) -> AggregateReleaseMetadata:
    """以 metadata 全部公開欄位重新建構 immutable typed boundary。

    ``AggregateReleaseMetadata`` 雖然是 frozen dataclass，仍可能被低階方式竄改欄位；
    decoder 不能只讀取其中幾個 hash 或 count。這裡逐一取出所有正式欄位並重新呼叫
    公開 constructor，讓 schema、run 識別、八個 digest 與七個正計數再次套用同一份
    fail-closed 契約，而且不把任何值轉成字串或 NumPy scalar。
    """

    values = _all_dataclass_field_values(
        value,
        expected_type=AggregateReleaseMetadata,
        label="metadata",
    )
    return AggregateReleaseMetadata(**values)


def _revalidate_products(value: object) -> EncodedAggregateProducts:
    """以 ``EncodedAggregateProducts`` 公開 constructor 深層封存解碼輸入。

    表列會重新複製成固定順序的 mapping snapshot，陣列會重新 canonical 成唯讀
    ``int64``／``float64`` buffer。這一層只處理容器完整性；逐表欄位、排序、offset、
    shape 與跨產品 join 仍由 decoder 後續明確驗證，不能把 frozen container 當成語意
    驗證的替代品。
    """

    values = _all_dataclass_field_values(
        value,
        expected_type=EncodedAggregateProducts,
        label="products",
    )
    return EncodedAggregateProducts(**values)


# Parquet reader 交給 codec 的是 Python row mapping；decoder 先按下列固定型別群組
# 檢查原生 scalar，再把欄位交給對應的公開 record／key constructor。這些集合不是
# writer schema 的推測結果，而是本模組明示的 release schema 1 contract。
_TABLE_TEXT_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "shard_id",
        "output_relative_path",
        "trajectory_manifest_sha256",
        "scenario_id",
        "study_site_id",
        "analysis_region_id",
        "material_id",
        "material_category_zh",
        "material_family_zh",
        "representative_shape_zh",
        "behavior_class",
        "applicability_condition_zh",
        "calibration_status",
        "evidence_grade",
        "receptor_id",
        "vertical_id",
        "arrival_time_id",
        "season",
        "tide_class",
        "phase_or_event",
        "design_version",
        "initial_wetdry_semantics_id",
        "initial_ocm_month_yyyymm",
        "initial_ocm_time_origin",
        "projection_method",
        "linear_unit",
        "axis_order",
        "boundary_kind",
        "boundary_segment_id",
        "source_study_site_id",
        "target_study_site_id",
        "outcome",
    }
)
_TABLE_FLOAT_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "settling_velocity_mps",
        "receptor_lon_deg",
        "receptor_lat_deg",
        "receptor_template_z_m_positive_up",
        "initial_z_m_positive_up",
        "initial_eta_m_positive_up",
        "initial_bed_z_m_positive_up",
        "initial_water_column_height_m",
        "initial_height_above_bed_m",
        "initial_zcor_lower_m_positive_up",
        "initial_zcor_upper_m_positive_up",
        "initial_vertical_bracket_alpha",
        "x_min_m",
        "x_max_m",
        "y_min_m",
        "y_max_m",
        "center_lon_deg",
        "center_lat_deg",
        "pathway_input_interval_seconds",
        "pathway_allocated_interval_seconds",
        "segment_length_m",
    }
)
_TABLE_INT_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "shard_index",
        "scenario_start_index",
        "scenario_stop_index",
        "particle_count",
        "observation_count",
        "event_count",
        "scenario_index",
        "arrival_time_utc_ns",
        "arrival_year",
        "initial_source_face_local_index",
        "initial_source_face_global_index",
        "initial_wetdry_elem_value",
        "initial_ocm_source_time_index",
        "x_cell_count",
        "y_cell_count",
        "site_index",
        "cell_start_offset",
        "cell_stop_offset",
        "scenario_count",
        "total_member_count",
        "valid_member_denominator",
        "pathway_input_particle_count",
        "boundary_index",
        "edge_start_offset",
        "edge_stop_offset",
        "bin_start_offset",
        "bin_stop_offset",
        "source_receptor_index",
        "unique_member_count",
        "count",
    }
)
_TABLE_OPTIONAL_COLUMNS: Final[frozenset[str]] = frozenset(
    _SCENARIO_DYNAMIC_OPTIONAL_FIELDS
)
_TABLE_OPTIONAL_FLOAT_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "initial_z_m_positive_up",
        "initial_eta_m_positive_up",
        "initial_bed_z_m_positive_up",
        "initial_water_column_height_m",
        "initial_height_above_bed_m",
        "initial_zcor_lower_m_positive_up",
        "initial_zcor_upper_m_positive_up",
        "initial_vertical_bracket_alpha",
    }
)
_TABLE_OPTIONAL_INT_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "initial_source_face_local_index",
        "initial_source_face_global_index",
        "initial_wetdry_elem_value",
        "initial_ocm_source_time_index",
    }
)
_TABLE_OPTIONAL_TEXT_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "initial_wetdry_semantics_id",
        "initial_ocm_month_yyyymm",
        "initial_ocm_time_origin",
    }
)
_TABLE_NONNEGATIVE_INT_COLUMNS: Final[frozenset[str]] = frozenset(
    _TABLE_INT_COLUMNS
    - {"arrival_time_utc_ns", "arrival_year"}
)
_TABLE_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "shard_bindings.parquet": _SHARD_BINDING_COLUMNS,
    "scenario_strata.parquet": _SCENARIO_STRATUM_COLUMNS,
    "site_index.parquet": _SITE_INDEX_COLUMNS,
    "boundary_index.parquet": _BOUNDARY_INDEX_COLUMNS,
    "source_receptor_index.parquet": _SOURCE_RECEPTOR_INDEX_COLUMNS,
    "cross_site_counts.parquet": _CROSS_SITE_COUNT_COLUMNS,
    "outcome_counts.parquet": _OUTCOME_COUNT_COLUMNS,
    "site_denominators.parquet": _SITE_DENOMINATOR_COLUMNS,
    "receptor_denominators.parquet": _RECEPTOR_DENOMINATOR_COLUMNS,
}


def _grid_cell_counts_from_spec(
    aggregate_spec: AggregateSpec,
    site_id: str,
) -> tuple[int, int]:
    """由 AggregateSpec 的公尺制範圍重建 ``(x_cell_count, y_cell_count)``。

    spec constructor 已驗證每一軸可由整數個公尺 cell 覆蓋；decoder 仍在此明確使用
    同一個 canonical round 語意取得形狀，不讀取 encoded pathway shape 來反推或裁切
    規格。回傳的 x、y 順序只用於建立產品 shape，實際事件與路徑陣列仍固定為
    ``(y_cell, x_cell)``，不代表經緯度座標或任何海域有效遮罩。
    """

    try:
        grid = aggregate_spec.site_grids[site_id]
        x_count = int(round((grid.x_max_m - grid.x_min_m) / aggregate_spec.grid_cell_size_m))
        y_count = int(round((grid.y_max_m - grid.y_min_m) / aggregate_spec.grid_cell_size_m))
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"site {site_id!r} 無法由 AggregateSpec 建立整數 cell count") from error
    if x_count <= 0 or y_count <= 0:
        raise ValueError(f"site {site_id!r} 的 x/y cell count 必須大於 0")
    return _require_int64_count(x_count, label=f"site {site_id!r} x_cell_count"), _require_int64_count(
        y_count,
        label=f"site {site_id!r} y_cell_count",
    )


def _rebuild_metric_edges(
    minimum_m: int | float,
    maximum_m: int | float,
    cell_size_m: int | float,
    cell_count: int,
) -> np.ndarray:
    """依 spec 的公尺制 min/max、cell size 與 cell 數建立 canonical x/y edges。

    中間 edge 使用固定的 float64 公尺運算，最後一點明確寫成 spec 的 max；這是
    AggregateReleasePayload 與 StreamingPathwayAggregate 共用的格線重建規則。decoder
    不使用 encoded pathway edges（產品並未保存它們），也不以近似值、padding 或
    transposition 修補格網；無法形成有限的一維格線時直接 fail closed。
    """

    checked_count = _require_int64_count(cell_count, label="metric grid cell_count")
    try:
        edges = np.asarray(minimum_m, dtype=np.float64) + (
            np.arange(checked_count + 1, dtype=np.float64)
            * np.asarray(cell_size_m, dtype=np.float64)
        )
        edges[-1] = np.asarray(maximum_m, dtype=np.float64)
    except (TypeError, ValueError, OverflowError, MemoryError) as error:
        raise ValueError("公尺制 pathway 格線無法由 AggregateSpec 安全重建") from error
    if edges.ndim != 1 or edges.size != checked_count + 1:
        raise ValueError("公尺制 pathway 格線 shape 不符合 AggregateSpec cell count")
    if not bool(np.all(np.isfinite(edges))) or bool(np.any(edges[1:] <= edges[:-1])):
        raise ValueError("公尺制 pathway 格線必須有限且嚴格遞增")
    return np.array(edges, dtype=np.float64, order="C", copy=True)


def _validate_fixed_table_rows(
    products: EncodedAggregateProducts,
    *,
    file_name: str,
    expected_columns: tuple[str, ...],
    expected_row_count: int | None,
    allow_empty: bool,
) -> tuple[Mapping[str, object], ...]:
    """逐列驗證固定欄位集合、iteration order 與原生 scalar 型別。

    ``EncodedAggregateProducts`` 已隔離外層 mapping，但 row mapping 仍可能是被低階
    改寫後的資料。這裡要求 ``tuple(row.keys())`` 與 schema tuple 完全一致，而不是只
    比較 set；因此欄位遺漏、未知欄位與欄位順序錯位都會被拒絕。Parquet 的整數欄位
    必須是原生 Python ``int``，浮點欄位必須是原生 Python ``float``，文字欄位必須是
    原生 Python ``str``；唯一可為 ``None`` 的值是 ScenarioStratum 動態初始條件，且
    後續仍會由公開 constructor 驗證全有／全無政策。此 helper 不做 trim、cast、clip、
    padding 或缺值補零。
    """

    try:
        rows = products.tables[file_name]
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"{file_name} 缺少固定 table 產品") from error
    if type(rows) is not tuple:
        raise ValueError(f"{file_name} 的 row container 必須是 tuple")
    if not rows and not allow_empty:
        raise ValueError(f"{file_name} 不得為空")
    if expected_row_count is not None and len(rows) != expected_row_count:
        raise ValueError(
            f"{file_name} row count 必須精確為 {expected_row_count}"
        )

    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"{file_name}[{row_index}] 必須是 mapping")
        try:
            actual_columns = tuple(row.keys())
        except Exception as error:
            raise ValueError(f"{file_name}[{row_index}] 欄位順序無法讀取") from error
        if actual_columns != expected_columns:
            raise ValueError(
                f"{file_name}[{row_index}] 必須恰好使用固定欄位與順序："
                f"expected={expected_columns!r} actual={actual_columns!r}"
            )

        for column_name in expected_columns:
            try:
                value = row[column_name]
            except Exception as error:
                raise ValueError(
                    f"{file_name}[{row_index}].{column_name} 無法讀取"
                ) from error

            if column_name in _TABLE_OPTIONAL_COLUMNS:
                if value is None:
                    continue
                if column_name in _TABLE_OPTIONAL_FLOAT_COLUMNS:
                    valid_type = type(value) is float
                elif column_name in _TABLE_OPTIONAL_INT_COLUMNS:
                    valid_type = type(value) is int
                else:
                    valid_type = type(value) is str
                if not valid_type:
                    raise ValueError(
                        f"{file_name}[{row_index}].{column_name} 型別不符"
                    )
                if type(value) is float and not math.isfinite(value):
                    raise ValueError(
                        f"{file_name}[{row_index}].{column_name} 必須是有限 float"
                    )
                if type(value) is str and (not value or value != value.strip()):
                    raise ValueError(
                        f"{file_name}[{row_index}].{column_name} 必須是非空原生 str"
                    )
                if type(value) is int:
                    _require_int64_count(
                        value,
                        label=f"{file_name}[{row_index}].{column_name}",
                    )
                continue

            if value is None:
                raise ValueError(
                    f"{file_name}[{row_index}].{column_name} 不允許缺值"
                )
            if column_name in _TABLE_TEXT_COLUMNS:
                if type(value) is not str or not value or value != value.strip():
                    raise ValueError(
                        f"{file_name}[{row_index}].{column_name} 必須是非空原生 str"
                    )
            elif column_name in _TABLE_FLOAT_COLUMNS:
                if type(value) is not float or not math.isfinite(value):
                    raise ValueError(
                        f"{file_name}[{row_index}].{column_name} 必須是有限原生 float"
                    )
            elif column_name in _TABLE_INT_COLUMNS:
                if type(value) is not int:
                    raise ValueError(
                        f"{file_name}[{row_index}].{column_name} 必須是原生 int"
                    )
                if column_name in _TABLE_NONNEGATIVE_INT_COLUMNS:
                    _require_int64_count(
                        value,
                        label=f"{file_name}[{row_index}].{column_name}",
                    )
                else:
                    _require_int64_scalar(
                        value,
                        label=f"{file_name}[{row_index}].{column_name}",
                    )
            else:
                raise ValueError(
                    f"{file_name}[{row_index}].{column_name} 不在固定欄位型別契約"
                )
    return rows


def _require_exact_int64_array(
    value: object,
    *,
    file_name: str,
    expected_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    """要求產品陣列已是 exact ``int64``、非負且符合指定 shape。"""

    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(np.int64):
        raise ValueError(f"{file_name} 必須是 exact int64 np.ndarray")
    if expected_shape is not None and value.shape != expected_shape:
        raise ValueError(f"{file_name} shape 必須精確為 {expected_shape}")
    if bool(np.any(value < 0)):
        raise ValueError(f"{file_name} 不可含負值")
    return value


def _require_exact_float64_array(
    value: object,
    *,
    file_name: str,
    expected_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    """要求產品陣列已是 exact ``float64``、有限且符合指定 shape。"""

    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(np.float64):
        raise ValueError(f"{file_name} 必須是 exact float64 np.ndarray")
    if expected_shape is not None and value.shape != expected_shape:
        raise ValueError(f"{file_name} shape 必須精確為 {expected_shape}")
    if not bool(np.all(np.isfinite(value))):
        raise ValueError(f"{file_name} 必須全部有限")
    return value


def _decode_offsets(
    value: object,
    *,
    file_name: str,
    expected_length: int,
) -> tuple[int, ...]:
    """驗證一維 int64 offset 從零開始並嚴格遞增，回傳 Python int tuple。

    offset 代表串接陣列中的索引位置，不是公尺或秒；轉成 Python ``int`` 只在已確認
    input dtype 是 int64 後進行，因此後續所有乘法／加法都能先以任意精度運算，再由
    ``_checked_count_product`` 或 count helper 檢查 signed int64 邊界。任何空片段、倒退、
    越界或無法對應的尾端都不在此層修補。
    """

    checked_length = _require_int64_count(expected_length, label=f"{file_name} expected length")
    offsets = _require_exact_int64_array(
        value,
        file_name=file_name,
        expected_shape=(checked_length,),
    )
    values = tuple(int(item) for item in offsets)
    if not values or values[0] != 0:
        raise ValueError(f"{file_name} 必須從 0 開始")
    if any(left >= right for left, right in zip(values, values[1:], strict=False)):
        raise ValueError(f"{file_name} 必須嚴格遞增，不得有空片段或倒退")
    return values


def _require_exact_float_scalar(
    value: object,
    expected: object,
    *,
    label: str,
) -> None:
    """核對 table 的 Python float 與 spec scalar 的 float64 值完全相同。

    表格 writer 會把 spec 的合法整數／浮點座標 canonical 成 Python ``float``；這裡
    不使用近似容差，並額外比較 float64 bytes，讓負零等 bit-level 差異不會被當成另一
    個可接受的 grid bound、投影中心或秒數 scalar。"""

    if type(value) is not float:
        raise ValueError(f"{label} 必須是原生 float")
    try:
        expected_float = float(expected)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} 的 spec scalar 無法 canonical") from error
    if not math.isfinite(expected_float) or not math.isfinite(value):
        raise ValueError(f"{label} 必須是有限 float")
    value_bytes = np.asarray(value, dtype=np.float64).tobytes()
    expected_bytes = np.asarray(expected_float, dtype=np.float64).tobytes()
    if value != expected_float or value_bytes != expected_bytes:
        raise ValueError(f"{label} 必須精確等於 AggregateSpec")


def _decode_shard_and_scenario_rows(
    products: EncodedAggregateProducts,
    metadata: AggregateReleaseMetadata,
) -> tuple[
    tuple[AggregateShardBinding, ...],
    tuple[ScenarioStratum, ...],
    dict[str, tuple[ScenarioStratum, ...]],
    set[tuple[str, str]],
]:
    """依表格原始列序重建 shard／scenario records 與站點 join index。

    ``shard_index`` 與 ``scenario_index`` 只是 release row 的連續索引，真正的
    shard scenario range 與 scenario 內容仍由各自公開 constructor 保存；decoder
    不把 index 欄位塞回 record。scenario 的動態初始條件只有整組 ``None`` 或整組
    有值兩種狀態，不能因 table nullable 欄位而接受部分缺值。回傳的 site／receptor
    index 只供後續拓撲核對，不會由缺少的列補出任何事件或來源資料。
    """

    shard_rows = _validate_fixed_table_rows(
        products,
        file_name="shard_bindings.parquet",
        expected_columns=_SHARD_BINDING_COLUMNS,
        expected_row_count=metadata.shard_row_count,
        allow_empty=False,
    )
    scenario_rows = _validate_fixed_table_rows(
        products,
        file_name="scenario_strata.parquet",
        expected_columns=_SCENARIO_STRATUM_COLUMNS,
        expected_row_count=metadata.scenario_row_count,
        allow_empty=False,
    )

    shards: list[AggregateShardBinding] = []
    shard_ids: set[str] = set()
    for row_index, row in enumerate(shard_rows):
        if row["shard_index"] != row_index:
            raise ValueError("shard_index 必須依 row 順序連續為 0..N-1")
        shard_id = row["shard_id"]
        if shard_id in shard_ids:
            raise ValueError("shard_bindings.parquet 的 shard_id 不得重複")
        shard_ids.add(shard_id)  # type: ignore[arg-type]
        record_values = {
            field_name: row[field_name] for field_name in _SHARD_BINDING_FIELDS
        }
        shards.append(AggregateShardBinding(**record_values))  # type: ignore[arg-type]

    strata: list[ScenarioStratum] = []
    scenario_ids: set[str] = set()
    strata_by_site_lists: dict[str, list[ScenarioStratum]] = {}
    scenario_receptor_keys: set[tuple[str, str]] = set()
    for row_index, row in enumerate(scenario_rows):
        if row["scenario_index"] != row_index:
            raise ValueError("scenario_index 必須依 row 順序連續為 0..N-1")
        scenario_id = row["scenario_id"]
        if scenario_id in scenario_ids:
            raise ValueError("scenario_strata.parquet 的 scenario_id 不得重複")
        scenario_ids.add(scenario_id)  # type: ignore[arg-type]
        record_values = {
            field_name: row[field_name] for field_name in _SCENARIO_STRATUM_FIELDS
        }
        stratum = ScenarioStratum(**record_values)  # type: ignore[arg-type]
        strata.append(stratum)
        strata_by_site_lists.setdefault(stratum.study_site_id, []).append(stratum)
        scenario_receptor_keys.add((stratum.study_site_id, stratum.receptor_id))

    strata_by_site = {
        site_id: tuple(site_strata)
        for site_id, site_strata in strata_by_site_lists.items()
    }
    return tuple(shards), tuple(strata), strata_by_site, scenario_receptor_keys


def _expected_boundary_key_set(
    aggregate_spec: AggregateSpec,
) -> set[BoundaryAggregateKey]:
    """由 spec 的每站 local／outer segment 引用建立完整 boundary topology。

    這個集合是規格宣告的資料拓撲，不是依現有計數列反推的結果；同一 segment 在
    不同站點或不同 boundary kind 出現時必須形成不同公開 key。建構器已驗證 segment
    ID 與長度表關係，因此 decoder 只建立 key，不替缺少的邊界列補零。
    """

    expected: set[BoundaryAggregateKey] = set()
    for site_id, segment_groups in aggregate_spec.site_boundary_segment_ids.items():
        for boundary_kind, segment_ids in (
            ("local", segment_groups.local_segment_ids),
            ("outer", segment_groups.outer_segment_ids),
        ):
            for segment_id in segment_ids:
                expected.add(
                    BoundaryAggregateKey(
                        study_site_id=site_id,
                        boundary_kind=boundary_kind,
                        boundary_segment_id=segment_id,
                    )
                )
    if not expected:
        raise ValueError("AggregateSpec 的 boundary topology 不得為空")
    return expected


def _expected_source_key_set(
    scenario_receptor_keys: set[tuple[str, str]],
    boundary_keys: set[BoundaryAggregateKey],
) -> set[SourceReceptorAggregateKey]:
    """建立 scenario receptor 與站點 boundary 的完整 source-receptor 笛卡兒積。"""

    expected: set[SourceReceptorAggregateKey] = set()
    for site_id, receptor_id in scenario_receptor_keys:
        for boundary_key in boundary_keys:
            if boundary_key.study_site_id != site_id:
                continue
            expected.add(
                SourceReceptorAggregateKey(
                    study_site_id=site_id,
                    receptor_id=receptor_id,
                    boundary_kind=boundary_key.boundary_kind,
                    boundary_segment_id=boundary_key.boundary_segment_id,
                )
            )
    if not expected:
        raise ValueError("scenario 與 boundary 不得形成空的 source-receptor topology")
    return expected


def _decode_age_axis(
    aggregate_spec: AggregateSpec,
    products: EncodedAggregateProducts,
) -> tuple[np.ndarray, int]:
    """驗證唯一秒制 age 軸並回傳 float64 副本與 Python age-bin count。

    age edges 由 encoded product 提供，但合法性與 canonical 值由 AggregateSpec 鎖定；
    這裡要求一維、至少兩點、從精確 0 秒開始且嚴格遞增。所有後續 histogram 長度
    乘法都使用回傳的 Python ``int``，避免 NumPy 固定寬度運算在極大 offset 或 bin
    數下先繞回後才被錯誤地當成合法 shape。
    """

    values = _require_exact_float64_array(
        products.arrays["age_bin_edges_seconds.npy"],
        file_name="age_bin_edges_seconds.npy",
    )
    if values.ndim != 1 or values.size < 2:
        raise ValueError("age_bin_edges_seconds.npy 必須是一維且至少有兩個 edge")
    if values[0] != 0.0:
        raise ValueError("age_bin_edges_seconds.npy 必須從 0 秒開始")
    if bool(np.any(values[1:] <= values[:-1])):
        raise ValueError("age_bin_edges_seconds.npy 必須嚴格遞增")

    expected = np.array(
        aggregate_spec.age_bin_edges_seconds,
        dtype=np.float64,
        order="C",
        copy=True,
    )
    if not np.array_equal(values, expected):
        raise ValueError("age_bin_edges_seconds.npy 必須精確等於 AggregateSpec age 軸")
    age_bin_count = _require_int64_count(
        int(values.size) - 1,
        label="age_bin_count",
    )
    if age_bin_count <= 0:
        raise ValueError("age_bin_count 必須大於 0")
    return np.array(values, dtype=np.float64, order="C", copy=True), age_bin_count


def _decode_site_products(
    aggregate_spec: AggregateSpec,
    metadata: AggregateReleaseMetadata,
    products: EncodedAggregateProducts,
    *,
    strata_by_site: Mapping[str, tuple[ScenarioStratum, ...]],
    age_edges: np.ndarray,
    age_bin_count: int,
) -> tuple[
    tuple[str, ...],
    dict[str, Mapping[str, object]],
    dict[str, SiteEventGridCounts],
    dict[str, StreamingPathwayAggregate],
]:
    """還原 site index、事件格網、pathway 與 site-cell 串接片段。

    site row 依 ``study_site_id`` 字典序描述每站公尺制矩形與 metric CRS；六類事件
    格網、pathway unique／residence 都切取同一個 ``[cell_start, cell_stop)`` C-order
    片段，軸固定為 ``(y_cell, x_cell)``。first-passage 使用同一站的 cell offset
    乘以 age-bin 數，軸固定為 ``(y_cell, x_cell, age_bin)``。x/y edges 不從產品陣列
    反推，而是只能由 spec 的 min/max/cell size canonical 重建；row 中的 input／
    allocated interval 以秒原樣交給 pathway constructor，不以 residence array 覆蓋。
    """

    site_rows = _validate_fixed_table_rows(
        products,
        file_name="site_index.parquet",
        expected_columns=_SITE_INDEX_COLUMNS,
        expected_row_count=metadata.site_row_count,
        allow_empty=False,
    )
    spec_site_ids = tuple(sorted(aggregate_spec.site_grids))
    if len(site_rows) != len(spec_site_ids):
        raise ValueError("site_index row count 必須精確等於 AggregateSpec site count")
    site_count = _require_int64_count(len(spec_site_ids), label="site_count")
    site_offsets = _decode_offsets(
        products.arrays["site_cell_offsets.npy"],
        file_name="site_cell_offsets.npy",
        expected_length=site_count + 1,
    )
    total_cells = site_offsets[-1]
    if total_cells <= 0:
        raise ValueError("site_cell_offsets.npy 的每站 cell 片段不得為空")

    event_arrays: dict[str, np.ndarray] = {}
    for file_name, _field_name in _EVENT_GRID_PRODUCTS:
        event_arrays[file_name] = _require_exact_int64_array(
            products.arrays[file_name],
            file_name=file_name,
            expected_shape=(total_cells,),
        )
    pathway_unique = _require_exact_int64_array(
        products.arrays["pathway_unique_particle_count.npy"],
        file_name="pathway_unique_particle_count.npy",
        expected_shape=(total_cells,),
    )
    pathway_residence = _require_exact_float64_array(
        products.arrays["pathway_residence_time_seconds.npy"],
        file_name="pathway_residence_time_seconds.npy",
        expected_shape=(total_cells,),
    )
    total_first_passage_values = _checked_count_product(
        total_cells,
        age_bin_count,
        label="pathway_first_passage_age_histogram 總長度",
    )
    pathway_first_passage = _require_exact_int64_array(
        products.arrays["pathway_first_passage_age_histogram.npy"],
        file_name="pathway_first_passage_age_histogram.npy",
        expected_shape=(total_first_passage_values,),
    )

    actual_site_ids: list[str] = []
    site_rows_by_id: dict[str, Mapping[str, object]] = {}
    site_grid_counts: dict[str, SiteEventGridCounts] = {}
    pathway_by_site: dict[str, StreamingPathwayAggregate] = {}

    for site_index, row in enumerate(site_rows):
        if row["site_index"] != site_index:
            raise ValueError("site_index 必須依 row 順序連續為 0..N-1")
        site_id = row["study_site_id"]
        if type(site_id) is not str:
            raise ValueError("site_index.study_site_id 必須是原生 str")
        if actual_site_ids and site_id <= actual_site_ids[-1]:
            raise ValueError("site_index 必須依 study_site_id 字典序且不得重複")
        if site_id not in aggregate_spec.site_grids:
            raise ValueError(f"site_index 引用 AggregateSpec 未知站點：{site_id!r}")
        if site_id in site_rows_by_id:
            raise ValueError("site_index 的 study_site_id 不得重複")
        actual_site_ids.append(site_id)

        site_strata = strata_by_site.get(site_id, ())
        if not site_strata:
            raise ValueError(f"site {site_id!r} 必須至少有一列 scenario_strata")
        analysis_regions = {stratum.analysis_region_id for stratum in site_strata}
        if len(analysis_regions) != 1:
            raise ValueError(
                f"site {site_id!r} 的 scenario_strata 必須恰有一個 analysis_region_id"
            )
        if row["analysis_region_id"] != next(iter(analysis_regions)):
            raise ValueError(f"site {site_id!r} 的 analysis_region_id 與 strata 不一致")

        grid = aggregate_spec.site_grids[site_id]
        for field_name in ("x_min_m", "x_max_m", "y_min_m", "y_max_m"):
            _require_exact_float_scalar(
                row[field_name],
                getattr(grid, field_name),
                label=f"site_index[{site_index}].{field_name}",
            )
        metric_crs = aggregate_spec.site_metric_crs[site_id]
        for field_name in ("projection_method", "linear_unit", "axis_order"):
            if row[field_name] != getattr(metric_crs, field_name):
                raise ValueError(
                    f"site_index[{site_index}].{field_name} 必須精確等於 AggregateSpec"
                )
        for field_name in ("center_lon_deg", "center_lat_deg"):
            _require_exact_float_scalar(
                row[field_name],
                getattr(metric_crs, field_name),
                label=f"site_index[{site_index}].{field_name}",
            )

        x_cell_count, y_cell_count = _grid_cell_counts_from_spec(
            aggregate_spec,
            site_id,
        )
        if row["x_cell_count"] != x_cell_count or row["y_cell_count"] != y_cell_count:
            raise ValueError(f"site {site_id!r} 的 x/y cell count 與 AggregateSpec 不一致")
        cell_count = _checked_count_product(
            x_cell_count,
            y_cell_count,
            label=f"site {site_id!r} cell_count",
        )

        cell_start = row["cell_start_offset"]
        cell_stop = row["cell_stop_offset"]
        if cell_start != site_offsets[site_index] or cell_stop != site_offsets[site_index + 1]:
            raise ValueError(f"site {site_id!r} 的 row cell offset 與 site_cell_offsets 不一致")
        if cell_stop <= cell_start or cell_stop - cell_start != cell_count:
            raise ValueError(f"site {site_id!r} 的 cell offset 長度與 spec 不一致")

        scenario_count = len(site_strata)
        expected_member_count = _checked_count_product(
            scenario_count,
            metadata.members_per_scenario,
            label=f"site {site_id!r} scenario_count×members_per_scenario",
        )
        if row["scenario_count"] != scenario_count:
            raise ValueError(f"site {site_id!r} 的 scenario_count 與 strata 不一致")
        for field_name in ("total_member_count", "pathway_input_particle_count"):
            if row[field_name] != expected_member_count:
                raise ValueError(
                    f"site {site_id!r} 的 {field_name} 必須等於 scenario_count×members_per_scenario"
                )
        if row["valid_member_denominator"] > row["total_member_count"]:
            raise ValueError(f"site {site_id!r} 的有效分母不可超過總成員數")
        _require_nonnegative_finite_float(
            row["pathway_input_interval_seconds"],
            label=f"site_index[{site_index}].pathway_input_interval_seconds",
        )
        _require_nonnegative_finite_float(
            row["pathway_allocated_interval_seconds"],
            label=f"site_index[{site_index}].pathway_allocated_interval_seconds",
        )

        expected_shape = (y_cell_count, x_cell_count)
        grid_values: dict[str, np.ndarray] = {}
        for file_name, field_name in _EVENT_GRID_PRODUCTS:
            grid_values[field_name] = np.array(
                event_arrays[file_name][cell_start:cell_stop],
                dtype=np.int64,
                order="C",
                copy=True,
            ).reshape(expected_shape, order="C")
        site_grid_counts[site_id] = SiteEventGridCounts(**grid_values)

        first_start = _checked_count_product(
            cell_start,
            age_bin_count,
            label=f"site {site_id!r} first-passage start offset",
        )
        first_stop = _checked_count_product(
            cell_stop,
            age_bin_count,
            label=f"site {site_id!r} first-passage stop offset",
        )
        expected_histogram_shape = (y_cell_count, x_cell_count, age_bin_count)
        first_passage_values = np.array(
            pathway_first_passage[first_start:first_stop],
            dtype=np.int64,
            order="C",
            copy=True,
        ).reshape(expected_histogram_shape, order="C")
        unique_values = np.array(
            pathway_unique[cell_start:cell_stop],
            dtype=np.int64,
            order="C",
            copy=True,
        ).reshape(expected_shape, order="C")
        residence_values = np.array(
            pathway_residence[cell_start:cell_stop],
            dtype=np.float64,
            order="C",
            copy=True,
        ).reshape(expected_shape, order="C")

        x_edges = _rebuild_metric_edges(
            grid.x_min_m,
            grid.x_max_m,
            aggregate_spec.grid_cell_size_m,
            x_cell_count,
        )
        y_edges = _rebuild_metric_edges(
            grid.y_min_m,
            grid.y_max_m,
            aggregate_spec.grid_cell_size_m,
            y_cell_count,
        )
        pathway_by_site[site_id] = StreamingPathwayAggregate(
            x_edges_m=x_edges,
            y_edges_m=y_edges,
            age_bin_edges_seconds=np.array(age_edges, dtype=np.float64, copy=True),
            unique_particle_count=unique_values,
            residence_time_seconds=residence_values,
            first_passage_age_histogram=first_passage_values,
            input_particle_count=row["pathway_input_particle_count"],
            input_interval_seconds=row["pathway_input_interval_seconds"],
            allocated_interval_seconds=row["pathway_allocated_interval_seconds"],
        )
        site_rows_by_id[site_id] = row

    if tuple(actual_site_ids) != spec_site_ids:
        raise ValueError("site_index 的 study_site_id set 必須精確等於 AggregateSpec")
    if site_offsets[-1] != total_cells:
        raise ValueError("site_cell_offsets 最後一值不符合串接 cell 陣列長度")
    return tuple(actual_site_ids), site_rows_by_id, site_grid_counts, pathway_by_site


def _decode_boundary_products(
    aggregate_spec: AggregateSpec,
    metadata: AggregateReleaseMetadata,
    products: EncodedAggregateProducts,
    *,
    site_ids: tuple[str, ...],
    age_bin_count: int,
    age_edges: np.ndarray,
) -> tuple[
    dict[BoundaryAggregateKey, np.ndarray],
    dict[BoundaryAggregateKey, np.ndarray],
    dict[BoundaryAggregateKey, np.ndarray],
    set[BoundaryAggregateKey],
]:
    """依 boundary index 還原邊界 edges、弧長 raw 與 ``(s_bin, age_bin)`` histogram。

    rows 必須依 ``(study_site_id, boundary_kind, boundary_segment_id)`` 排序且 index
    連續；兩套一維 offset 分別切分公尺制 edge 與 s-bin raw count，travel array 則以
    Python int 先計算 ``bin_offset × age_bin_count`` 後按 C-order reshape。每段 edges
    會與公開 ``initialize_event_aggregate(spec, age)`` 重新建立的 canonical 公尺格線
    逐值 exact 比對；segment length 只接受 spec 明示長度，不能由最後 edge 近似推導。
    """

    rows = _validate_fixed_table_rows(
        products,
        file_name="boundary_index.parquet",
        expected_columns=_BOUNDARY_INDEX_COLUMNS,
        expected_row_count=metadata.boundary_row_count,
        allow_empty=False,
    )
    expected_boundary_keys = _expected_boundary_key_set(aggregate_spec)
    if len(rows) != len(expected_boundary_keys):
        raise ValueError("boundary_index row count 必須精確等於 spec boundary topology")
    if set(site_ids) != set(aggregate_spec.site_grids):
        raise ValueError("site_ids 與 AggregateSpec site set 不一致")

    edge_offsets = _decode_offsets(
        products.arrays["boundary_edge_offsets.npy"],
        file_name="boundary_edge_offsets.npy",
        expected_length=len(rows) + 1,
    )
    bin_offsets = _decode_offsets(
        products.arrays["boundary_bin_offsets.npy"],
        file_name="boundary_bin_offsets.npy",
        expected_length=len(rows) + 1,
    )
    edge_values = _require_exact_float64_array(
        products.arrays["boundary_bin_edges_m.npy"],
        file_name="boundary_bin_edges_m.npy",
        expected_shape=(edge_offsets[-1],),
    )
    raw_values = _require_exact_int64_array(
        products.arrays["boundary_arclength_raw_count.npy"],
        file_name="boundary_arclength_raw_count.npy",
        expected_shape=(bin_offsets[-1],),
    )
    total_travel_values = _checked_count_product(
        bin_offsets[-1],
        age_bin_count,
        label="boundary_travel_age_histogram 總長度",
    )
    travel_values = _require_exact_int64_array(
        products.arrays["boundary_travel_age_histogram.npy"],
        file_name="boundary_travel_age_histogram.npy",
        expected_shape=(total_travel_values,),
    )

    # 這是唯一允許的 boundary edge 來源：它同時固定每個 segment 的 bin 數、尾端
    # 是否有短 bin，以及最後一點的 float64 canonical 值；decoder 不使用產品 edge
    # 反推 segment_length 或修補不一致的資料。
    canonical_event = initialize_event_aggregate(
        aggregate_spec,
        age_bin_edges_seconds=np.array(age_edges, dtype=np.float64, order="C", copy=True),
    )
    actual_keys: set[BoundaryAggregateKey] = set()
    previous_sort_key: tuple[str, str, str] | None = None
    boundary_edges: dict[BoundaryAggregateKey, np.ndarray] = {}
    boundary_raw: dict[BoundaryAggregateKey, np.ndarray] = {}
    boundary_travel: dict[BoundaryAggregateKey, np.ndarray] = {}

    for boundary_index, row in enumerate(rows):
        if row["boundary_index"] != boundary_index:
            raise ValueError("boundary_index 必須依 row 順序連續為 0..N-1")
        key = BoundaryAggregateKey(
            study_site_id=row["study_site_id"],
            boundary_kind=row["boundary_kind"],
            boundary_segment_id=row["boundary_segment_id"],
        )
        sort_key = (
            key.study_site_id,
            key.boundary_kind,
            key.boundary_segment_id,
        )
        if previous_sort_key is not None and sort_key <= previous_sort_key:
            raise ValueError("boundary_index 必須依 site、kind、segment 嚴格排序且不得重複")
        previous_sort_key = sort_key
        if key in actual_keys:
            raise ValueError("boundary_index 的 key 不得重複")
        if key not in expected_boundary_keys:
            raise ValueError(f"boundary_index 引用未知 boundary key：{key!r}")
        if key.study_site_id not in site_ids:
            raise ValueError(f"boundary_index 引用未知站點：{key.study_site_id!r}")
        actual_keys.add(key)

        try:
            expected_segment_length = aggregate_spec.boundary_segment_lengths_m[
                key.boundary_segment_id
            ]
            expected_edges = canonical_event.boundary_bin_edges_m[key]
        except (KeyError, TypeError) as error:
            raise ValueError(f"boundary key 缺少 AggregateSpec canonical 定義：{key!r}") from error
        _require_exact_float_scalar(
            row["segment_length_m"],
            expected_segment_length,
            label=f"boundary_index[{boundary_index}].segment_length_m",
        )

        edge_start = row["edge_start_offset"]
        edge_stop = row["edge_stop_offset"]
        bin_start = row["bin_start_offset"]
        bin_stop = row["bin_stop_offset"]
        if (
            edge_start != edge_offsets[boundary_index]
            or edge_stop != edge_offsets[boundary_index + 1]
            or bin_start != bin_offsets[boundary_index]
            or bin_stop != bin_offsets[boundary_index + 1]
        ):
            raise ValueError(f"boundary_index[{boundary_index}] 的 row offsets 不一致")
        edge_count = edge_stop - edge_start
        bin_count = bin_stop - bin_start
        if edge_count < 2 or bin_count < 1 or edge_count != bin_count + 1:
            raise ValueError(f"boundary_index[{boundary_index}] 的 edge/bin 長度不一致")
        if expected_edges.size != edge_count:
            raise ValueError(f"boundary {key!r} 的 edge 數與 canonical spec 不一致")

        edges = np.array(
            edge_values[edge_start:edge_stop],
            dtype=np.float64,
            order="C",
            copy=True,
        )
        if not np.array_equal(edges, expected_edges):
            raise ValueError(f"boundary {key!r} 的 edges 必須精確等於 canonical spec edges")
        raw = np.array(
            raw_values[bin_start:bin_stop],
            dtype=np.int64,
            order="C",
            copy=True,
        )
        travel_start = _checked_count_product(
            bin_start,
            age_bin_count,
            label=f"boundary {key!r} travel start offset",
        )
        travel_stop = _checked_count_product(
            bin_stop,
            age_bin_count,
            label=f"boundary {key!r} travel stop offset",
        )
        travel = np.array(
            travel_values[travel_start:travel_stop],
            dtype=np.int64,
            order="C",
            copy=True,
        ).reshape((bin_count, age_bin_count), order="C")
        boundary_edges[key] = edges
        boundary_raw[key] = raw
        boundary_travel[key] = travel

    if actual_keys != expected_boundary_keys:
        raise ValueError("boundary_index 的 key set 必須精確等於 AggregateSpec topology")
    if edge_offsets[-1] != edge_values.size or bin_offsets[-1] != raw_values.size:
        raise ValueError("boundary offsets 尾端與串接陣列長度不一致")
    if total_travel_values != travel_values.size:
        raise ValueError("boundary travel offset 尾端與串接陣列長度不一致")
    return boundary_edges, boundary_raw, boundary_travel, actual_keys


def _decode_source_receptor_products(
    aggregate_spec: AggregateSpec,
    metadata: AggregateReleaseMetadata,
    products: EncodedAggregateProducts,
    *,
    site_ids: tuple[str, ...],
    scenario_receptor_keys: set[tuple[str, str]],
    boundary_keys: set[BoundaryAggregateKey],
    age_bin_count: int,
) -> tuple[
    dict[SourceReceptorAggregateKey, int],
    dict[SourceReceptorAggregateKey, np.ndarray],
]:
    """依 source-receptor index 還原 raw count 與一維秒制 age histogram。

    列序固定為 ``(study_site_id, receptor_id, boundary_kind, boundary_segment_id)``；
    第 i 列只可使用 raw array 的第 i 個值，以及 travel array 中由
    ``i × age_bin_count`` 計算出的固定連續片段。source key 必須是 scenario receptor
    與 AggregateSpec boundary topology 的完整笛卡兒積；零計數列也要保留，不能以缺列
    代替零值或以未知站點／segment 猜測 join 關係。
    """

    rows = _validate_fixed_table_rows(
        products,
        file_name="source_receptor_index.parquet",
        expected_columns=_SOURCE_RECEPTOR_INDEX_COLUMNS,
        expected_row_count=metadata.source_receptor_row_count,
        allow_empty=False,
    )
    expected_keys = _expected_source_key_set(scenario_receptor_keys, boundary_keys)
    if len(rows) != len(expected_keys):
        raise ValueError("source_receptor_index row count 必須精確等於 source topology")
    if set(site_ids) != set(aggregate_spec.site_grids):
        raise ValueError("source-receptor site set 與 AggregateSpec 不一致")

    raw_values = _require_exact_int64_array(
        products.arrays["source_receptor_raw_count.npy"],
        file_name="source_receptor_raw_count.npy",
        expected_shape=(len(rows),),
    )
    total_travel_values = _checked_count_product(
        len(rows),
        age_bin_count,
        label="source_receptor_travel_age_histogram 總長度",
    )
    travel_values = _require_exact_int64_array(
        products.arrays["source_receptor_travel_age_histogram.npy"],
        file_name="source_receptor_travel_age_histogram.npy",
        expected_shape=(total_travel_values,),
    )

    actual_keys: set[SourceReceptorAggregateKey] = set()
    previous_sort_key: tuple[str, str, str, str] | None = None
    raw_by_key: dict[SourceReceptorAggregateKey, int] = {}
    travel_by_key: dict[SourceReceptorAggregateKey, np.ndarray] = {}
    for source_index, row in enumerate(rows):
        if row["source_receptor_index"] != source_index:
            raise ValueError("source_receptor_index 必須依 row 順序連續為 0..N-1")
        key = SourceReceptorAggregateKey(
            study_site_id=row["study_site_id"],
            receptor_id=row["receptor_id"],
            boundary_kind=row["boundary_kind"],
            boundary_segment_id=row["boundary_segment_id"],
        )
        sort_key = (
            key.study_site_id,
            key.receptor_id,
            key.boundary_kind,
            key.boundary_segment_id,
        )
        if previous_sort_key is not None and sort_key <= previous_sort_key:
            raise ValueError("source_receptor_index 必須依 site、receptor、kind、segment 排序")
        previous_sort_key = sort_key
        if key in actual_keys:
            raise ValueError("source_receptor_index 的 key 不得重複")
        if key not in expected_keys:
            raise ValueError(f"source_receptor_index 引用未知 source key：{key!r}")
        if key.study_site_id not in site_ids:
            raise ValueError(f"source_receptor_index 引用未知站點：{key.study_site_id!r}")
        actual_keys.add(key)

        raw_by_key[key] = int(raw_values[source_index])
        travel_start = _checked_count_product(
            source_index,
            age_bin_count,
            label=f"source-receptor {key!r} travel start offset",
        )
        travel_stop = _checked_count_product(
            source_index + 1,
            age_bin_count,
            label=f"source-receptor {key!r} travel stop offset",
        )
        travel_by_key[key] = np.array(
            travel_values[travel_start:travel_stop],
            dtype=np.int64,
            order="C",
            copy=True,
        ).reshape((age_bin_count,), order="C")

    if actual_keys != expected_keys:
        raise ValueError("source_receptor_index 的 key set 必須精確等於 source topology")
    if raw_values.size != len(rows) or travel_values.size != total_travel_values:
        raise ValueError("source-receptor 陣列長度與 row count 不一致")
    return raw_by_key, travel_by_key


def _decode_count_tables(
    aggregate_spec: AggregateSpec,
    products: EncodedAggregateProducts,
    *,
    site_ids: tuple[str, ...],
    scenario_receptor_keys: set[tuple[str, str]],
    site_rows_by_id: Mapping[str, Mapping[str, object]],
) -> tuple[
    dict[CrossSiteAggregateKey, int],
    dict[str, dict[str, int]],
    dict[str, int],
    dict[str, int],
    dict[ReceptorAggregateKey, int],
]:
    """還原 cross/outcome/site/receptor 四張計數與分母表。

    四表都使用 module 內固定欄位與 sort order；cross-site、outcome 與 receptor
    topology 即使 count 為零也必須保留完整列。site denominator 的三欄再逐欄核對
    ``site_index``，避免兩個產品各自合法但分母語意分裂。typed key 會透過公開
    ``CrossSiteAggregateKey`` 與 ``ReceptorAggregateKey`` constructor 建立，未知或
    重複 join key 一律拒絕；這些 count 仍只是條件式來源足跡／相對來源權重的統計
    載體，不是絕對來源機率。
    """

    cross_rows = _validate_fixed_table_rows(
        products,
        file_name="cross_site_counts.parquet",
        expected_columns=_CROSS_SITE_COUNT_COLUMNS,
        expected_row_count=None,
        allow_empty=True,
    )
    outcome_rows = _validate_fixed_table_rows(
        products,
        file_name="outcome_counts.parquet",
        expected_columns=_OUTCOME_COUNT_COLUMNS,
        expected_row_count=None,
        allow_empty=False,
    )
    site_denominator_rows = _validate_fixed_table_rows(
        products,
        file_name="site_denominators.parquet",
        expected_columns=_SITE_DENOMINATOR_COLUMNS,
        expected_row_count=None,
        allow_empty=False,
    )
    receptor_rows = _validate_fixed_table_rows(
        products,
        file_name="receptor_denominators.parquet",
        expected_columns=_RECEPTOR_DENOMINATOR_COLUMNS,
        expected_row_count=None,
        allow_empty=False,
    )

    expected_cross_keys = {
        CrossSiteAggregateKey(
            source_study_site_id=source_site_id,
            target_study_site_id=target_site_id,
        )
        for source_site_id in site_ids
        for target_site_id in site_ids
        if source_site_id != target_site_id
    }
    if len(cross_rows) != len(expected_cross_keys):
        raise ValueError("cross_site_counts row topology 不完整或含有額外列")
    cross_counts: dict[CrossSiteAggregateKey, int] = {}
    actual_cross_keys: set[CrossSiteAggregateKey] = set()
    previous_cross_sort_key: tuple[str, str] | None = None
    for _row_index, row in enumerate(cross_rows):
        key = CrossSiteAggregateKey(
            source_study_site_id=row["source_study_site_id"],
            target_study_site_id=row["target_study_site_id"],
        )
        sort_key = (key.source_study_site_id, key.target_study_site_id)
        if previous_cross_sort_key is not None and sort_key <= previous_cross_sort_key:
            raise ValueError("cross_site_counts 必須依 source、target 嚴格排序且不得重複")
        previous_cross_sort_key = sort_key
        if key in actual_cross_keys:
            raise ValueError("cross_site_counts 的 key 不得重複")
        if key not in expected_cross_keys:
            raise ValueError(f"cross_site_counts 引用未知站點 pair：{key!r}")
        actual_cross_keys.add(key)
        cross_counts[key] = row["unique_member_count"]  # type: ignore[assignment]
    if actual_cross_keys != expected_cross_keys:
        raise ValueError("cross_site_counts 的 key set 必須精確等於完整 ordered site pairs")

    expected_outcomes = {
        status.value for status in ParticleStatus if status != ParticleStatus.ACTIVE
    }
    expected_outcome_keys = {
        (site_id, outcome)
        for site_id in site_ids
        for outcome in expected_outcomes
    }
    if len(outcome_rows) != len(expected_outcome_keys):
        raise ValueError("outcome_counts 必須保留每站每個非 ACTIVE 狀態的完整零列")
    outcomes_by_site = {site_id: {} for site_id in site_ids}
    actual_outcome_keys: set[tuple[str, str]] = set()
    previous_outcome_sort_key: tuple[str, str] | None = None
    for row_index, row in enumerate(outcome_rows):
        site_id = row["study_site_id"]
        outcome = row["outcome"]
        if site_id not in site_ids or outcome not in expected_outcomes:
            raise ValueError(f"outcome_counts[{row_index}] 引用未知 site 或 outcome")
        sort_key = (site_id, outcome)  # type: ignore[assignment]
        if previous_outcome_sort_key is not None and sort_key <= previous_outcome_sort_key:
            raise ValueError("outcome_counts 必須依 study_site_id、outcome 嚴格排序")
        previous_outcome_sort_key = sort_key
        outcome_key = (site_id, outcome)  # type: ignore[assignment]
        if outcome_key in actual_outcome_keys:
            raise ValueError("outcome_counts 的 key 不得重複")
        actual_outcome_keys.add(outcome_key)
        outcomes_by_site[site_id][outcome] = row["count"]  # type: ignore[index,assignment]
    if actual_outcome_keys != expected_outcome_keys:
        raise ValueError("outcome_counts 的 key set 必須精確等於非 ACTIVE 狀態拓撲")

    if len(site_denominator_rows) != len(site_ids):
        raise ValueError("site_denominators row count 必須精確等於 site count")
    valid_by_site: dict[str, int] = {}
    total_by_site: dict[str, int] = {}
    actual_site_denominator_ids: list[str] = []
    for row_index, row in enumerate(site_denominator_rows):
        site_id = row["study_site_id"]
        if site_id not in site_ids:
            raise ValueError(f"site_denominators[{row_index}] 引用未知站點")
        if actual_site_denominator_ids and site_id <= actual_site_denominator_ids[-1]:
            raise ValueError("site_denominators 必須依 study_site_id 嚴格排序且不得重複")
        if site_id in valid_by_site:
            raise ValueError("site_denominators 的 study_site_id 不得重複")
        actual_site_denominator_ids.append(site_id)
        valid_count = row["valid_member_denominator"]
        total_count = row["total_member_count"]
        site_row = site_rows_by_id[site_id]
        if (
            valid_count != site_row["valid_member_denominator"]
            or total_count != site_row["total_member_count"]
        ):
            raise ValueError(f"site_denominators[{row_index}] 必須逐欄等於 site_index")
        if valid_count > total_count:
            raise ValueError(f"site_denominators[{row_index}] 的有效分母不可超過總成員數")
        valid_by_site[site_id] = valid_count  # type: ignore[assignment]
        total_by_site[site_id] = total_count  # type: ignore[assignment]
    if tuple(actual_site_denominator_ids) != site_ids:
        raise ValueError("site_denominators 的 site set／排序必須精確等於 site_index")

    expected_receptor_keys = {
        ReceptorAggregateKey(study_site_id=site_id, receptor_id=receptor_id)
        for site_id, receptor_id in scenario_receptor_keys
    }
    if len(receptor_rows) != len(expected_receptor_keys):
        raise ValueError("receptor_denominators row count 必須精確等於 scenario receptor set")
    receptor_counts: dict[ReceptorAggregateKey, int] = {}
    actual_receptor_keys: set[ReceptorAggregateKey] = set()
    previous_receptor_sort_key: tuple[str, str] | None = None
    for _row_index, row in enumerate(receptor_rows):
        key = ReceptorAggregateKey(
            study_site_id=row["study_site_id"],
            receptor_id=row["receptor_id"],
        )
        sort_key = (key.study_site_id, key.receptor_id)
        if previous_receptor_sort_key is not None and sort_key <= previous_receptor_sort_key:
            raise ValueError("receptor_denominators 必須依 site、receptor 嚴格排序")
        previous_receptor_sort_key = sort_key
        if key in actual_receptor_keys:
            raise ValueError("receptor_denominators 的 key 不得重複")
        if key not in expected_receptor_keys:
            raise ValueError(f"receptor_denominators 引用未知 site/receptor：{key!r}")
        actual_receptor_keys.add(key)
        receptor_counts[key] = row["valid_member_denominator"]  # type: ignore[assignment]
    if actual_receptor_keys != expected_receptor_keys:
        raise ValueError("receptor_denominators 的 key set 必須精確等於 scenario receptor set")

    return cross_counts, outcomes_by_site, valid_by_site, total_by_site, receptor_counts


def _products_are_exactly_equal(
    left: EncodedAggregateProducts,
    right: EncodedAggregateProducts,
) -> bool:
    """比較兩份 canonical products 的表格列序、scalar、dtype、shape 與 bytes。

    round-trip gate 不能只比較 row count 或 NumPy shape；欄位順序、零列拓撲、C-order
    flatten 位置與浮點／整數的每一個值都屬 release contract。浮點 scalar／array
    另外比較 float64 bytes，以保留 signed zero 等 exact value 差異；此比較只讀取
    immutable snapshots，不修改任一產品。
    """

    if frozenset(left.tables) != frozenset(right.tables):
        return False
    for file_name in sorted(left.tables):
        left_rows = left.tables[file_name]
        right_rows = right.tables[file_name]
        if len(left_rows) != len(right_rows):
            return False
        for left_row, right_row in zip(left_rows, right_rows, strict=True):
            if tuple(left_row.keys()) != tuple(right_row.keys()):
                return False
            for column_name in left_row:
                left_value = left_row[column_name]
                right_value = right_row[column_name]
                if type(left_value) is not type(right_value):
                    return False
                if type(left_value) is float:
                    if (
                        left_value != right_value
                        or np.asarray(left_value, dtype=np.float64).tobytes()
                        != np.asarray(right_value, dtype=np.float64).tobytes()
                    ):
                        return False
                elif left_value != right_value:
                    return False

    if frozenset(left.arrays) != frozenset(right.arrays):
        return False
    for file_name in sorted(left.arrays):
        left_array = left.arrays[file_name]
        right_array = right.arrays[file_name]
        if left_array.dtype != right_array.dtype or left_array.shape != right_array.shape:
            return False
        if not np.array_equal(left_array, right_array):
            return False
        if left_array.tobytes(order="C") != right_array.tobytes(order="C"):
            return False
    return True


def _decode_aggregate_release_payload_impl(
    *,
    metadata: AggregateReleaseMetadata,
    aggregate_spec: AggregateSpec,
    products: EncodedAggregateProducts,
) -> AggregateReleasePayload:
    """執行 decoder 的固定還原順序；對外 wrapper 負責統一非 ValueError 例外。"""

    if type(metadata) is not AggregateReleaseMetadata:
        raise ValueError("metadata 必須是 exact AggregateReleaseMetadata 實例")
    if type(aggregate_spec) is not AggregateSpec:
        raise ValueError("aggregate_spec 必須是 exact AggregateSpec 實例")
    if type(products) is not EncodedAggregateProducts:
        raise ValueError("products 必須是 exact EncodedAggregateProducts 實例")

    # 三個公開輸入都可能由 object.__setattr__ 竄改；先用全部正式欄位重建，再開始
    # 讀表與切陣列。spec hash 與 run_id 是 typed metadata 對科學規格的 provenance
    # 鎖點，兩者不一致時不能讓不同 run 的資料拼成同一個 release。
    validated_metadata = _revalidate_metadata(metadata)
    validated_spec = _revalidate_aggregate_spec(aggregate_spec)
    validated_products = _revalidate_products(products)
    if validated_metadata.run_id != validated_spec.run_id:
        raise ValueError("metadata.run_id 必須精確等於 aggregate_spec.run_id")
    if (
        validated_metadata.aggregate_spec_source_sha256
        != validated_spec.source_sha256
    ):
        raise ValueError("metadata.aggregate_spec_source_sha256 與 AggregateSpec 不一致")
    if (
        validated_metadata.aggregate_spec_canonical_sha256
        != validated_spec.canonical_sha256
    ):
        raise ValueError(
            "metadata.aggregate_spec_canonical_sha256 與 AggregateSpec 不一致"
        )

    shards, strata, strata_by_site, scenario_receptor_keys = (
        _decode_shard_and_scenario_rows(validated_products, validated_metadata)
    )
    age_edges, age_bin_count = _decode_age_axis(validated_spec, validated_products)

    expected_total_particle_count = _checked_count_product(
        len(strata),
        validated_metadata.members_per_scenario,
        label="scenario_count×members_per_scenario",
    )
    if expected_total_particle_count != validated_metadata.input_particle_count:
        raise ValueError("metadata.input_particle_count 必須等於 scenario_count×members_per_scenario")

    # shard record 的 range 是原始 scenario tuple 的半開區間；逐 shard 檢查連續性與
    # particle_count，並以 Python 任意精度先計算 span×M，避免 signed int64 靜默繞回。
    shard_cursor = 0
    shard_particle_total = 0
    for shard_index, shard in enumerate(shards):
        if shard.scenario_start_index != shard_cursor:
            raise ValueError("shard_bindings 的 scenario range 必須從 0 連續銜接")
        span = shard.scenario_stop_index - shard.scenario_start_index
        expected_shard_particles = _checked_count_product(
            span,
            validated_metadata.members_per_scenario,
            label=f"shard_bindings[{shard_index}] scenario span×members_per_scenario",
        )
        if shard.particle_count != expected_shard_particles:
            raise ValueError("shard particle_count 必須等於 scenario span×members_per_scenario")
        shard_particle_total = _require_int64_count(
            shard_particle_total + shard.particle_count,
            label="shard particle_count 總和",
        )
        shard_cursor = shard.scenario_stop_index
    if shard_cursor != len(strata) or shard_particle_total != expected_total_particle_count:
        raise ValueError("shard ranges 與 scenario_strata 必須完整且無缺口覆蓋")

    site_ids, site_rows_by_id, site_grid_counts, pathway_by_site = _decode_site_products(
        validated_spec,
        validated_metadata,
        validated_products,
        strata_by_site=strata_by_site,
        age_edges=age_edges,
        age_bin_count=age_bin_count,
    )
    (
        boundary_edges,
        boundary_raw,
        boundary_travel,
        boundary_keys,
    ) = _decode_boundary_products(
        validated_spec,
        validated_metadata,
        validated_products,
        site_ids=site_ids,
        age_bin_count=age_bin_count,
        age_edges=age_edges,
    )
    source_raw, source_travel = _decode_source_receptor_products(
        validated_spec,
        validated_metadata,
        validated_products,
        site_ids=site_ids,
        scenario_receptor_keys=scenario_receptor_keys,
        boundary_keys=boundary_keys,
        age_bin_count=age_bin_count,
    )
    (
        cross_counts,
        outcomes_by_site,
        valid_by_site,
        total_by_site,
        receptor_counts,
    ) = _decode_count_tables(
        validated_spec,
        validated_products,
        site_ids=site_ids,
        scenario_receptor_keys=scenario_receptor_keys,
        site_rows_by_id=site_rows_by_id,
    )

    # 所有固定產品都已逐表驗證後，才交給公開 EventAggregateChunk constructor。此處
    # 明確使用 metadata 的全域 input_particle_count；不從 site denominator、outcome
    # 或任何陣列總和反推粒子數，以維持 metadata 與 payload 的 provenance 邊界。
    event = EventAggregateChunk(
        site_grid_counts=site_grid_counts,
        boundary_bin_edges_m=boundary_edges,
        boundary_arclength_raw_count=boundary_raw,
        boundary_travel_age_histogram=boundary_travel,
        age_bin_edges_seconds=np.array(age_edges, dtype=np.float64, order="C", copy=True),
        source_receptor_raw_count=source_raw,
        source_receptor_travel_age_histogram=source_travel,
        cross_site_unique_member_count=cross_counts,
        outcome_count_by_site=outcomes_by_site,
        valid_member_denominator_by_site=valid_by_site,
        total_member_count_by_site=total_by_site,
        valid_member_denominator_by_receptor=receptor_counts,
        input_particle_count=validated_metadata.input_particle_count,
    )

    decoded = AggregateReleasePayload(
        schema_version=validated_metadata.schema_version,
        run_id=validated_metadata.run_id,
        run_kind=validated_metadata.run_kind,
        experiment_case_id=validated_metadata.experiment_case_id,
        members_per_scenario=validated_metadata.members_per_scenario,
        config_hash=validated_metadata.config_hash,
        checkpoint_input_binding_hash=validated_metadata.checkpoint_input_binding_hash,
        source_run_plan_sha256=validated_metadata.source_run_plan_sha256,
        source_run_progress_sha256=validated_metadata.source_run_progress_sha256,
        source_normalized_config_sha256=validated_metadata.source_normalized_config_sha256,
        source_input_inventory_sha256=validated_metadata.source_input_inventory_sha256,
        aggregate_spec=validated_spec,
        shard_bindings=shards,
        scenario_strata=strata,
        event_aggregate=event,
        pathway_by_site=pathway_by_site,
    )
    # 最外層 payload 再做一次完整深層重建，讓 decoder 的所有輸出仍符合 encode 與
    # metadata_from_payload 對 frozen object 的防竄改政策，而非只依賴本輪局部變數。
    decoded = _revalidate_payload(decoded)
    decoded_metadata = metadata_from_payload(decoded)
    if decoded_metadata != validated_metadata:
        raise ValueError("metadata_from_payload(decoded) 必須精確等於 validated metadata")

    reencoded = _encode_validated_payload(decoded)
    if not _products_are_exactly_equal(reencoded, validated_products):
        raise ValueError("decoder round-trip products 必須逐表逐陣列 exact 相等")
    return decoded
