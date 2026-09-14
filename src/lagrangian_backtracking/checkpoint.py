"""安全保存與恢復 reference／production 粒子中途狀態。

schema 1 只保存 ``ParticleState``，本模組保留其既有 ``write_checkpoint``／
``load_checkpoint`` API。execution checkpoint 的新 writer 固定發布 schema ``3.0.0``，
以不可變歷史 segment 與小型 current-state compact 分離寫入；每次 checkpoint 只追加
本次新增的 observation／event rows，不重寫既有歷史。loader 會沿每一代
``checkpoint.json`` SHA-256 chain 還原完整 execution；該 manifest 同時綁定 compact、
history segment、RNG 與 provenance。schema ``2.0.0``／``2.1.0``／``2.2.0`` 僅作舊檔工程相容讀取。
為了讓舊 run 可持續使用，
``load_execution_checkpoint`` 仍接受三個 2.x 版本，但不會把舊目錄原地升級。
舊有 2.2 payload 的讀取契約如下：schema ``2.2.0`` 保存完整 ``ParticleExecutionState``、
每條粒子的 PCG64DXSM generator state、triangle hint
與固定 RunUnit identity；schema ``2.0.0``／``2.1.0`` 僅作舊檔工程相容讀取。2.2 observation
在既有環境樣本欄位之外，保存同一個輸出觀測點的總速度、OCM current、Stokes 水平速度、
向上為正的沉降速度、速度樣本狀態與獨立速度品質旗標；這些欄位必須由 engine 的明示
資料來源提供，loader 不會從時間、位置或位移猜測。每個舊 schema 使用固定欄位集合，
不會因目前 ``Observation`` dataclass 新增欄位而被靜默改寫；舊檔缺少的速度資料維持
``NOT_SAMPLED``／``None``，不自動升格為新證據。forcing、mesh、邊界與其他外部大型資料
不序列化，必須由同一個 request factory 在 restore 時重建。所有 JSON 禁止 NaN／無限值，
資料檔案以大小與 SHA-256 驗證，目標目錄以 partial directory 完成後原子更名且不允許
覆寫既有 checkpoint。schema ``2.0.0`` 缺少環境欄位，不能作正式垂向證據；schema
``2.0.0``／``2.1.0`` 均沒有速度欄位，不能作速度證據。``2.1.0`` 既有的環境欄位仍可供
環境資料讀取，缺少速度不表示環境欄位失效。checkpoint 本身是續跑工程產品，不是正式
trajectory artifact；本機 synthetic 測試也不是 OCM／NWW3 科學成果。
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields, replace
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .engine import EnvironmentSampleStatus, Observation, ParticleExecutionState
from .models import (
    BoundaryEvent,
    EventType,
    ParticleState,
    ParticleStatus,
    VelocitySampleStatus,
)
from .outputs import sha256_file

_SCHEMA30_VERSION = "3.0.0"
CHECKPOINT_SCHEMA_VERSION = _SCHEMA30_VERSION


@dataclass(frozen=True, slots=True)
class CheckpointBinding:
    """判定中途計算狀態是否可安全續跑的固定識別欄位。

    ``random_stream_id`` 是可選的共同亂數流命名空間。舊 checkpoint 不含此欄位時，
    loader 會以 ``None`` 還原並維持原有六欄 binding；paired run 的 writer 則保存非空
    stream ID，讓物理案例身分不同但 RNG 命名空間相同的 run 不能誤用彼此的 checkpoint。
    """

    config_hash: str
    input_inventory_hash: str
    experiment_case_id: str
    shard_id: str
    seed_policy: str
    code_commit: str
    random_stream_id: str | None = None

    def __post_init__(self) -> None:
        """拒絕空白 stream ID，避免 checkpoint binding 以空值代表不同語意。"""

        if self.random_stream_id is not None and (
            type(self.random_stream_id) is not str or not self.random_stream_id.strip()
        ):
            raise ValueError("random_stream_id 必須是非空白字串或 None")


@dataclass(slots=True)
class ExecutionCheckpoint:
    """schema 2.x／3.0 execution checkpoint 的記憶體表示，供 loader 與 ``ProductionBatch`` 共用。

    ``executions``、``rng_states``、``triangle_hints`` 與 ``run_unit_identities`` 的順序
    都是固定 particle order；任何一項重新排序都會在讀取或 restore 時拒絕。``binding``
    在純記憶體 snapshot 可為 ``None``，但寫入檔案時必須是 ``CheckpointBinding``。
    """

    binding: CheckpointBinding | None
    sequence: int
    run_unit_identities: list[dict[str, Any]]
    executions: list[ParticleExecutionState]
    rng_states: list[dict[str, Any]]
    triangle_hints: list[int]
    schema_version: str = _SCHEMA30_VERSION
    observation_cursors: list[int] = field(default_factory=list)
    event_cursors: list[int] = field(default_factory=list)

    @property
    def particle_order(self) -> tuple[str, ...]:
        """依 checkpoint 固定順序回傳 particle_id，供 restore 與稽核報告使用。"""

        return tuple(identity["particle_id"] for identity in self.run_unit_identities)


_SCHEMA20_VERSION = "2.0.0"
_SCHEMA21_VERSION = "2.1.0"
_SCHEMA22_VERSION = "2.2.0"
_WRITER_SCHEMA_VERSION = _SCHEMA30_VERSION
_SUPPORTED_EXECUTION_SCHEMA_VERSIONS = frozenset(
    {_SCHEMA20_VERSION, _SCHEMA21_VERSION, _SCHEMA22_VERSION, _SCHEMA30_VERSION}
)
_SCHEMA2_DATA_FILES = frozenset({"execution_state.json", "rng_states.json"})
_SCHEMA3_DATA_FILES = frozenset({"compact_state.json", "history_segment.json"})
_SCHEMA3_GENERATION_NAME_RE = re.compile(r"checkpoint-[0-9]{8}\Z")
_BINDING_FIELDS = (
    "config_hash",
    "input_inventory_hash",
    "experiment_case_id",
    "shard_id",
    "seed_policy",
    "code_commit",
)
_BINDING_FIELDS_WITH_RANDOM = _BINDING_FIELDS + ("random_stream_id",)
_PARTICLE_FIELDS = tuple(field.name for field in fields(ParticleState))
_LEGACY_OBSERVATION_FIELDS = (
    "particle_id",
    "time_utc_ns",
    "age_seconds",
    "x_m",
    "y_m",
    "z_m",
    "status",
)
_ENVIRONMENT_OBSERVATION_FIELDS = (
    "environment_sample_status",
    "eta_m",
    "bed_z_m",
    "forcing_month_id",
    "environment_qc_flags",
)
_VELOCITY_OBSERVATION_FIELDS = (
    "velocity_sample_status",
    "total_u_mps",
    "total_v_mps",
    "total_w_mps",
    "ocm_u_mps",
    "ocm_v_mps",
    "ocm_w_mps",
    "stokes_u_mps",
    "stokes_v_mps",
    "settling_w_mps",
    "velocity_qc_flags",
)
_SCHEMA21_OBSERVATION_FIELDS = _LEGACY_OBSERVATION_FIELDS + _ENVIRONMENT_OBSERVATION_FIELDS
_SCHEMA22_OBSERVATION_FIELDS = _SCHEMA21_OBSERVATION_FIELDS + _VELOCITY_OBSERVATION_FIELDS
_EVENT_FIELDS = tuple(field.name for field in fields(BoundaryEvent))
_YYYYMM_PATTERN = re.compile(r"^[0-9]{6}$")


def _json_safe(value: Any) -> Any:
    """把 Enum、NumPy scalar／array 與巢狀容器轉為可禁止 NaN 的 JSON 原生型別。"""

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: object, *, sort_keys: bool = False) -> None:
    """以 UTF-8、禁止 NaN／無限值的方式寫 JSON；資料順序由呼叫端保留。"""

    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            _json_safe(payload),
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=sort_keys,
            allow_nan=False,
        )
        handle.write("\n")


def _binding_payload(binding: CheckpointBinding) -> dict[str, Any]:
    """建立 checkpoint metadata 的 binding object，並保留舊六欄格式。

    ``dataclasses.asdict`` 會把新增的 optional 欄位寫成 ``null``，造成未啟用配對的舊
    run 也改變 metadata bytes 與欄位集合。這個 helper 只有在 stream ID 明示時才加入
    第七欄，讓既有 schema 2 checkpoint 可由新版 loader 讀取，且 paired checkpoint 的
    binding 又能直接保存並驗證其亂數命名空間。
    """

    payload = {
        field_name: getattr(binding, field_name)
        for field_name in _BINDING_FIELDS
    }
    if binding.random_stream_id is not None:
        payload["random_stream_id"] = binding.random_stream_id
    return payload


def _reject_json_constant(value: str) -> None:
    """拒絕 Python JSON parser 預設容許的 NaN、Infinity 與 -Infinity。"""

    raise ValueError(f"checkpoint JSON 不允許非有限值：{value}")


def _read_json(path: Path) -> Any:
    """讀取 JSON 並遞迴拒絕非有限浮點數，避免損壞檔案悄悄進入計算。"""

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle, parse_constant=_reject_json_constant)
    _assert_finite_json(payload)
    return payload


def _assert_finite_json(value: Any) -> None:
    """檢查 parser 可能由極大指數產生的 inf，以及所有巢狀 JSON 值。"""

    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("checkpoint JSON 含有非有限浮點數")
    if isinstance(value, dict):
        for item in value.values():
            _assert_finite_json(item)
    elif isinstance(value, list):
        for item in value:
            _assert_finite_json(item)


def _require_exact_keys(value: Any, expected: Sequence[str], *, label: str) -> dict[str, Any]:
    """要求 JSON object 欄位完整且不含未知欄位，防止 schema 漂移被靜默忽略。"""

    if not isinstance(value, dict) or set(value) != set(expected):
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(f"{label} 欄位不符：actual={actual}, expected={sorted(expected)}")
    return value


def _nonnegative_int(value: Any, *, label: str) -> int:
    """解析不接受 bool 的非負整數欄位。"""

    if type(value) is not int or value < 0:
        raise ValueError(f"{label} 必須是非負整數")
    return value


def _positive_int(value: Any, *, label: str) -> int:
    """解析嚴格大於零的整數欄位，供 generation 與粒子計數使用。"""

    normalized = _nonnegative_int(value, label=label)
    if normalized < 1:
        raise ValueError(f"{label} 必須是正整數")
    return normalized


def _integer(value: Any, *, label: str) -> int:
    """解析只接受 JSON 整數的欄位，避免 bool 或浮點數冒充時間／識別碼。"""

    if type(value) is not int:
        raise ValueError(f"{label} 必須是整數，且不可為 bool")
    return value


def _nonempty_string(value: Any, *, label: str) -> str:
    """解析不可為空的文字身分欄位，避免空字串遮蔽粒子或事件歸屬。"""

    if type(value) is not str or not value:
        raise ValueError(f"{label} 必須是非空字串")
    return value


def _optional_string(value: Any, *, label: str) -> str | None:
    """解析可省略的文字識別欄位；有值時仍不可是空字串或其他 primitive。"""

    if value is None:
        return None
    return _nonempty_string(value, label=label)


def _optional_integer(value: Any, *, label: str) -> int | None:
    """解析可省略的整數識別欄位，保留資料來源可能使用的整數 sentinel。"""

    if value is None:
        return None
    return _integer(value, label=label)


def _environment_sample_status(value: Any, *, label: str) -> EnvironmentSampleStatus:
    """嚴格解析 observation 的環境樣本狀態列舉。

    checkpoint JSON 只保存 ``EnvironmentSampleStatus.value`` 字串；這裡不接受列舉名稱、
    整數或其他可轉型物件，讓「尚未取樣」、「有效」與「無效」三種資料來源狀態不會被
    讀取端模糊合併。實際四欄之間的相依限制仍交由 engine 的 ``Observation`` constructor
    做最後驗證，避免 checkpoint 與執行期的資料契約分叉。
    """

    if type(value) is not str:
        raise ValueError(f"{label} 必須是 EnvironmentSampleStatus 字串")
    try:
        return EnvironmentSampleStatus(value)
    except ValueError as error:
        raise ValueError(f"{label} 含有未知 EnvironmentSampleStatus：{value!r}") from error


def _velocity_sample_status(value: Any, *, label: str) -> VelocitySampleStatus:
    """嚴格解析 observation 的速度樣本狀態列舉。

    checkpoint JSON 保存的是固定的狀態值字串；這裡拒絕列舉名稱、整數、bool 與其他
    可轉型物件。``NOT_SAMPLED`` 代表該 observation 沒有同點速度取樣，不是速度為零；
    ``TOTAL_ONLY``、``COMPLETE`` 與各種無效狀態則由 engine 的 Observation constructor
    再驗證其欄位和獨立速度品質旗標，避免舊檔或手動修改的資料被讀成完整分項。
    """

    if type(value) is not str:
        raise ValueError(f"{label} 必須是 VelocitySampleStatus 字串")
    try:
        return VelocitySampleStatus(value)
    except ValueError as error:
        raise ValueError(f"{label} 含有未知 VelocitySampleStatus：{value!r}") from error


def _optional_forcing_month(value: Any, *, label: str) -> str | None:
    """解析可省略的 forcing 月份識別，要求真實存在的 ``YYYYMM``。

    月份只是資料來源識別，不可從 observation UTC 時間推導；因此 checkpoint 只接受
    ASCII 六位數、年份 0001--9999 且月份 01--12 的原生字串，所有其他型別與日期都拒絕。
    """

    if value is None:
        return None
    if type(value) is not str or _YYYYMM_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} 必須是合法 YYYYMM")
    year = int(value[:4])
    month = int(value[4:])
    if not 1 <= year <= 9999 or not 1 <= month <= 12:
        raise ValueError(f"{label} 必須是合法 YYYYMM")
    return value


def _finite_float(value: Any, *, label: str) -> float:
    """解析有限數值欄位，避免 bool、文字或非有限值進入位置、年齡與步長。"""

    if type(value) not in (int, float):
        raise ValueError(f"{label} 必須是有限數值")
    try:
        normalized = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{label} 必須是有限數值") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{label} 必須是有限數值")
    return normalized


def _nonnegative_finite_float(value: Any, *, label: str) -> float:
    """解析非負且有限的秒數欄位；年齡不能用負值表示未來方向。"""

    normalized = _finite_float(value, label=label)
    if normalized < 0.0:
        raise ValueError(f"{label} 不可為負")
    return normalized


def _optional_finite_float(value: Any, *, label: str) -> float | None:
    """解析可省略的有限數值欄位；有值時不接受文字、bool 或非有限數值。"""

    if value is None:
        return None
    return _finite_float(value, label=label)


def _boolean(value: Any, *, label: str) -> bool:
    """解析真正的 JSON boolean，避免整數 0／1 被誤當成狀態旗標。"""

    if type(value) is not bool:
        raise ValueError(f"{label} 必須是 boolean")
    return value


def _particle_status(value: Any, *, label: str) -> ParticleStatus:
    """解析已登錄的粒子狀態字串，拒絕未知名稱與非文字 primitive。"""

    if type(value) is not str:
        raise ValueError(f"{label} 必須是 ParticleStatus 字串")
    try:
        return ParticleStatus(value)
    except ValueError as error:
        raise ValueError(f"{label} 含有未知 ParticleStatus：{value!r}") from error


def _event_type(value: Any, *, label: str) -> EventType:
    """解析已登錄的事件名稱，確保事件表不會含有無法聚合的未知類別。"""

    if type(value) is not str:
        raise ValueError(f"{label} 必須是 EventType 字串")
    try:
        return EventType(value)
    except ValueError as error:
        raise ValueError(f"{label} 含有未知 EventType：{value!r}") from error


def _serialize_particle_state(state: ParticleState) -> dict[str, Any]:
    """把 ParticleState 的既有欄位寫成可稽核 JSON，狀態保留字串值。"""

    row = asdict(state)
    row["status"] = state.status.value
    return _json_safe(row)


def _deserialize_particle_state(value: Any, *, label: str) -> ParticleState:
    """嚴格還原 ParticleState，先驗證欄位語意再呼叫 dataclass constructor。

    粒子身分欄位不可為空；member 與 UTC 奈秒時間必須是真正整數；位置與回溯年齡是
    公尺／秒的有限數值，年齡不得為負；``own_local_exit_recorded`` 必須是真正布林值。
    這些檢查不能交給 dataclass，因為 dataclass 不會在執行期阻止文字、bool 或 NaN。
    """

    row = _require_exact_keys(value, _PARTICLE_FIELDS, label=label)
    return ParticleState(
        particle_id=_nonempty_string(row["particle_id"], label=f"{label}.particle_id"),
        scenario_id=_nonempty_string(row["scenario_id"], label=f"{label}.scenario_id"),
        member_id=_nonnegative_int(row["member_id"], label=f"{label}.member_id"),
        study_site_id=_nonempty_string(row["study_site_id"], label=f"{label}.study_site_id"),
        analysis_region_id=_nonempty_string(
            row["analysis_region_id"], label=f"{label}.analysis_region_id"
        ),
        receptor_id=_nonempty_string(row["receptor_id"], label=f"{label}.receptor_id"),
        x_m=_finite_float(row["x_m"], label=f"{label}.x_m"),
        y_m=_finite_float(row["y_m"], label=f"{label}.y_m"),
        z_m=_finite_float(row["z_m"], label=f"{label}.z_m"),
        time_utc_ns=_integer(row["time_utc_ns"], label=f"{label}.time_utc_ns"),
        age_seconds=_nonnegative_finite_float(row["age_seconds"], label=f"{label}.age_seconds"),
        status=_particle_status(row["status"], label=f"{label}.status"),
        own_local_exit_recorded=_boolean(
            row["own_local_exit_recorded"], label=f"{label}.own_local_exit_recorded"
        ),
    )


def _serialize_observation(
    observation: Observation,
    *,
    schema_version: str = _WRITER_SCHEMA_VERSION,
) -> dict[str, Any]:
    """把 Observation 序列化成 v3 segment／compact 使用的固定 JSON object。

    2.2 與 3.0 以明確固定的 23 欄保存舊有位置／狀態、環境 context 與速度紀錄；不使用目前
    ``Observation`` dataclass 的反射欄位集合，避免日後新增執行期欄位時改變檔案拓撲。
    列舉使用穩定的 ``value`` 而不是 Python 名稱；``None`` 是唯一的 JSON 缺值表示，
    絕不把 NaN／無限值當作缺值。函式刻意拒絕以 schema 2.0／2.1 寫檔，因為兩者僅是
    loader 的舊檔工程相容格式，不能讓新 writer 產生沒有新速度紀錄的 checkpoint。
    """

    if schema_version not in {_SCHEMA22_VERSION, _SCHEMA30_VERSION}:
        raise ValueError("execution checkpoint writer 不提供 schema 2.0／2.1 downgrade")
    row = {name: getattr(observation, name) for name in _SCHEMA22_OBSERVATION_FIELDS}
    row["status"] = observation.status.value
    row["environment_sample_status"] = observation.environment_sample_status.value
    row["velocity_sample_status"] = observation.velocity_sample_status.value
    return _json_safe(row)


def _deserialize_observation(
    value: Any,
    *,
    label: str,
    schema_version: str,
) -> Observation:
    """依 execution schema 嚴格還原 Observation 並交由 constructor 驗證跨欄位契約。

    schema 2.0 的 observation 只允許原有七欄；schema 2.1 只允許七欄加五個環境欄位；
    schema 2.2／3.0 才允許再加固定的 11 個速度欄位。舊版本讀取後新增欄位一律保持 engine
    預設的 ``NOT_SAMPLED`` 與 ``None``，不從時間、位置、月份、深度或總速度推測。2.1／
    2.2 的有限數值、合法 ``YYYYMM``、狀態列舉與原生整數品質旗標先經本模組的 JSON
    邊界檢查，再交給 engine constructor 做 status／垂向範圍／速度總和的 cross-field
    驗證。這條責任界線確保舊檔可重啟，但不會冒充 F03／F09 正式證據。
    """

    if schema_version == _SCHEMA20_VERSION:
        expected_fields = _LEGACY_OBSERVATION_FIELDS
    elif schema_version == _SCHEMA21_VERSION:
        expected_fields = _SCHEMA21_OBSERVATION_FIELDS
    elif schema_version in {_SCHEMA22_VERSION, _SCHEMA30_VERSION}:
        expected_fields = _SCHEMA22_OBSERVATION_FIELDS
    else:
        raise ValueError(f"checkpoint schema 不支援：{schema_version!r}")
    row = _require_exact_keys(value, expected_fields, label=label)
    kwargs: dict[str, Any] = {
        "particle_id": _nonempty_string(row["particle_id"], label=f"{label}.particle_id"),
        "time_utc_ns": _integer(row["time_utc_ns"], label=f"{label}.time_utc_ns"),
        "age_seconds": _nonnegative_finite_float(
            row["age_seconds"], label=f"{label}.age_seconds"
        ),
        "x_m": _finite_float(row["x_m"], label=f"{label}.x_m"),
        "y_m": _finite_float(row["y_m"], label=f"{label}.y_m"),
        "z_m": _finite_float(row["z_m"], label=f"{label}.z_m"),
        "status": _particle_status(row["status"], label=f"{label}.status"),
    }
    if schema_version in {_SCHEMA21_VERSION, _SCHEMA22_VERSION, _SCHEMA30_VERSION}:
        kwargs.update(
            environment_sample_status=_environment_sample_status(
                row["environment_sample_status"],
                label=f"{label}.environment_sample_status",
            ),
            eta_m=_optional_finite_float(row["eta_m"], label=f"{label}.eta_m"),
            bed_z_m=_optional_finite_float(row["bed_z_m"], label=f"{label}.bed_z_m"),
            forcing_month_id=_optional_forcing_month(
                row["forcing_month_id"], label=f"{label}.forcing_month_id"
            ),
            environment_qc_flags=_optional_integer(
                row["environment_qc_flags"], label=f"{label}.environment_qc_flags"
            ),
        )
    if schema_version in {_SCHEMA22_VERSION, _SCHEMA30_VERSION}:
        kwargs.update(
            velocity_sample_status=_velocity_sample_status(
                row["velocity_sample_status"],
                label=f"{label}.velocity_sample_status",
            ),
            total_u_mps=_optional_finite_float(
                row["total_u_mps"], label=f"{label}.total_u_mps"
            ),
            total_v_mps=_optional_finite_float(
                row["total_v_mps"], label=f"{label}.total_v_mps"
            ),
            total_w_mps=_optional_finite_float(
                row["total_w_mps"], label=f"{label}.total_w_mps"
            ),
            ocm_u_mps=_optional_finite_float(row["ocm_u_mps"], label=f"{label}.ocm_u_mps"),
            ocm_v_mps=_optional_finite_float(row["ocm_v_mps"], label=f"{label}.ocm_v_mps"),
            ocm_w_mps=_optional_finite_float(row["ocm_w_mps"], label=f"{label}.ocm_w_mps"),
            stokes_u_mps=_optional_finite_float(
                row["stokes_u_mps"], label=f"{label}.stokes_u_mps"
            ),
            stokes_v_mps=_optional_finite_float(
                row["stokes_v_mps"], label=f"{label}.stokes_v_mps"
            ),
            settling_w_mps=_optional_finite_float(
                row["settling_w_mps"], label=f"{label}.settling_w_mps"
            ),
            velocity_qc_flags=_optional_integer(
                row["velocity_qc_flags"], label=f"{label}.velocity_qc_flags"
            ),
        )
    return Observation(**kwargs)


def _validate_attributes(value: Any, *, label: str) -> dict[str, bool | float | int | str]:
    """驗證事件 attributes 的基本型別並保留 JSON object 插入順序。

    attributes 是事件的補充稽核欄位，不是可任意巢狀的科學資料結構；因此只允許
    非空文字 key 與真正的 bool、int、float、str primitive。使用 ``type`` 而不是寬鬆的
    ``isinstance``，可避免 bool 被當成 int，或自訂數值物件在序列化時產生不穩定結果。
    """

    # schema 3 的 BoundaryEvent.attributes 由輕量 tracker 包裝；它仍是受限 mapping，
    # 序列化時轉回普通 dict 以維持既有 JSON 契約，不讓追蹤器類別名稱滲入檔案格式。
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} 必須是 object")
    result: dict[str, bool | float | int | str] = {}
    for key, item in value.items():
        if type(key) is not str or type(item) not in (bool, int, float, str):
            raise ValueError(f"{label} 只能含 string key 與 bool/int/float/string value")
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError(f"{label} 不可含非有限浮點數")
        result[key] = item
    return result


def _serialize_event(event: BoundaryEvent) -> dict[str, Any]:
    """把 BoundaryEvent 轉為完整 JSON，attributes 不排序以保留其原有順序。"""

    row = asdict(event)
    row["event_type"] = event.event_type.value
    row["attributes"] = _validate_attributes(event.attributes, label="event.attributes")
    return _json_safe(row)


def _deserialize_event(value: Any, *, label: str) -> BoundaryEvent:
    """嚴格還原事件型別、座標、optional identity 與 attributes。

    event 的位置與弧長使用公尺制有限數值，``fraction`` 是本步交點比例且必須落在
    [0, 1]；時間與整數 ID 欄位不接受 bool 或浮點數。可選文字 ID 只能是 ``None`` 或
    非空字串，可選數值欄位只能是 ``None`` 或有限數值。事件 attributes 維持原有 primitive
    型別與插入順序，讓 boundary event round-trip 不因 constructor 的寬鬆行為而失真。
    """

    row = _require_exact_keys(value, _EVENT_FIELDS, label=label)
    fraction = _finite_float(row["fraction"], label=f"{label}.fraction")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"{label}.fraction 必須介於 0 與 1 之間")
    return BoundaryEvent(
        particle_id=_nonempty_string(row["particle_id"], label=f"{label}.particle_id"),
        scenario_id=_nonempty_string(row["scenario_id"], label=f"{label}.scenario_id"),
        member_id=_nonnegative_int(row["member_id"], label=f"{label}.member_id"),
        study_site_id=_nonempty_string(row["study_site_id"], label=f"{label}.study_site_id"),
        analysis_region_id=_nonempty_string(
            row["analysis_region_id"], label=f"{label}.analysis_region_id"
        ),
        receptor_id=_nonempty_string(row["receptor_id"], label=f"{label}.receptor_id"),
        event_type=_event_type(row["event_type"], label=f"{label}.event_type"),
        time_utc_ns=_integer(row["time_utc_ns"], label=f"{label}.time_utc_ns"),
        x_m=_finite_float(row["x_m"], label=f"{label}.x_m"),
        y_m=_finite_float(row["y_m"], label=f"{label}.y_m"),
        z_m=_finite_float(row["z_m"], label=f"{label}.z_m"),
        fraction=fraction,
        related_study_site_id=_optional_string(
            row["related_study_site_id"], label=f"{label}.related_study_site_id"
        ),
        boundary_segment_id=_optional_string(
            row["boundary_segment_id"], label=f"{label}.boundary_segment_id"
        ),
        boundary_s_m=_optional_finite_float(row["boundary_s_m"], label=f"{label}.boundary_s_m"),
        source_face_id=_optional_integer(row["source_face_id"], label=f"{label}.source_face_id"),
        triangle_id=_optional_integer(row["triangle_id"], label=f"{label}.triangle_id"),
        forcing_month_id=_optional_string(
            row["forcing_month_id"], label=f"{label}.forcing_month_id"
        ),
        attributes=_validate_attributes(row["attributes"], label=f"{label}.attributes"),
    )


def _state_identity(state: ParticleState) -> tuple[Any, ...]:
    """提取運行期間不變、與 RunUnit 必須一致的固定 identity 欄位。"""

    return (
        state.particle_id,
        state.scenario_id,
        state.member_id,
        state.study_site_id,
        state.analysis_region_id,
        state.receptor_id,
    )


def _run_unit_identity(unit: Any) -> dict[str, Any]:
    """提取 RunUnit 的固定欄位，避免 checkpoint 只用 particle_id 掩蓋錯誤 scenario。"""

    scenario = unit.scenario
    return {
        "particle_id": unit.particle_id,
        "scenario_id": scenario.scenario_id,
        "member_id": int(unit.member_id),
        "study_site_id": scenario.study_site_id,
        "analysis_region_id": scenario.analysis_region_id,
        "receptor_id": scenario.receptor_id,
        "arrival_time_utc_ns": int(scenario.arrival_time_utc_ns),
        "experiment_case_id": unit.experiment_case_id,
        "seed": int(unit.seed),
    }


def _run_identity_tuple(identity: dict[str, Any]) -> tuple[Any, ...]:
    """把已驗證的 JSON identity 轉為可比較 tuple。"""

    return (
        identity["particle_id"],
        identity["scenario_id"],
        identity["member_id"],
        identity["study_site_id"],
        identity["analysis_region_id"],
        identity["receptor_id"],
        identity["arrival_time_utc_ns"],
        identity["experiment_case_id"],
        identity["seed"],
    )


def _validate_run_identity(value: Any, *, label: str) -> dict[str, Any]:
    """驗證固定 RunUnit identity 欄位與 primitive 型別。"""

    expected = (
        "particle_id",
        "scenario_id",
        "member_id",
        "study_site_id",
        "analysis_region_id",
        "receptor_id",
        "arrival_time_utc_ns",
        "experiment_case_id",
        "seed",
    )
    identity = _require_exact_keys(value, expected, label=label)
    for name in (
        "particle_id",
        "scenario_id",
        "study_site_id",
        "analysis_region_id",
        "receptor_id",
        "experiment_case_id",
    ):
        identity[name] = _nonempty_string(identity[name], label=f"{label}.{name}")
    identity["member_id"] = _nonnegative_int(identity["member_id"], label=f"{label}.member_id")
    # UTC 奈秒可合法落在 Unix epoch 之前，所以 arrival_time 只要求整數，不額外限制符號。
    identity["arrival_time_utc_ns"] = _integer(
        identity["arrival_time_utc_ns"], label=f"{label}.arrival_time_utc_ns"
    )
    identity["seed"] = _nonnegative_int(identity["seed"], label=f"{label}.seed")
    return identity


def _strict_writer_run_unit_identity(unit: Any, *, label: str) -> dict[str, Any]:
    """建立並嚴格驗證 writer 使用的 canonical RunUnit identity。

    ``_run_unit_identity`` 會把 member、到達 UTC 與 seed 轉成 Python ``int``，原本的
    便利轉換可能把 ``True``、``1.0`` 或數字文字誤藏成合法 identity。正式 v3 writer
    必須在建立 compact／metadata 前沿用 loader 的欄位契約，因此先對 RunUnit 原始欄位
    做不接受 bool 的整數檢查，再把提取結果交給 ``_validate_run_identity``。回傳值是
    後續 state、observation、event 與 metadata 共用的 canonical dict；這裡只驗證固定
    identity，不改變 RunUnit 或 execution 物件。
    """

    scenario = unit.scenario
    # 先驗原始來源，避免 _run_unit_identity 的 int(...) 將錯誤 primitive 靜默轉型；
    # arrival UTC 可為負值（代表 Unix epoch 之前），所以只要求原生整數而不限制符號。
    _nonnegative_int(unit.member_id, label=f"{label}.member_id")
    _integer(scenario.arrival_time_utc_ns, label=f"{label}.arrival_time_utc_ns")
    _nonnegative_int(unit.seed, label=f"{label}.seed")
    return _validate_run_identity(_run_unit_identity(unit), label=label)


def _normalize_triangle_hint(value: Any, *, label: str) -> int:
    """將 checkpoint 的三角形提示固定為 ``-1`` 或非負 int64 可表示的整數。"""

    if value is None:
        return -1
    if isinstance(value, bool) or not isinstance(value, int) or value < -1:
        raise ValueError(f"{label} 必須是 -1 或非負整數")
    if value > np.iinfo(np.int64).max:
        raise ValueError(f"{label} 超出 int64 範圍")
    return value


def _serialize_execution(
    execution: ParticleExecutionState,
    *,
    triangle_hint: int,
    schema_version: str = _WRITER_SCHEMA_VERSION,
) -> dict[str, Any]:
    """完整序列化一條 execution，不遺漏觀測、事件、步數或輸出游標。

    execution 的資料拓撲在 2.0／2.1／2.2 間保持不變，差異只在 observation 欄位；v3
    writer 不把整條 execution 放進 compact，而是另由呼叫端挑出 immutable history rows。
    舊版 fixture helper 會明示傳入 2.2，讀取端則依 metadata 版本選擇固定欄位集合。
    """

    if not execution.observations:
        raise ValueError("execution observations 不可為空")
    # execution 是可重啟游標；寫入前也要拒絕 bool 或文字等 dataclass 不會自行攔截的值，
    # 否則 writer 可能把 True 靜默轉成 1，造成 restore 後的狀態與原執行不再相同。
    _nonnegative_int(execution.step_count, label="execution.step_count")
    _nonnegative_int(execution.minimum_clamp_count, label="execution.minimum_clamp_count")
    next_output_age_seconds = _finite_float(
        execution.next_output_age_seconds, label="execution.next_output_age_seconds"
    )
    if next_output_age_seconds <= 0:
        raise ValueError("execution next_output_age_seconds 必須是有限正數")
    return {
        "state": _serialize_particle_state(execution.state),
        "observations": [
            _serialize_observation(item, schema_version=schema_version)
            for item in execution.observations
        ],
        "events": [_serialize_event(item) for item in execution.events],
        "step_count": execution.step_count,
        "minimum_clamp_count": execution.minimum_clamp_count,
        "next_output_age_seconds": next_output_age_seconds,
        "triangle_hint": triangle_hint,
    }


def _validate_observation_sequence(
    observations: Sequence[Observation],
    state: ParticleState,
    *,
    label: str,
) -> None:
    """驗證回溯觀測的方向與目前 state 關係，不額外假設輸出間隔必定整除步長。

    reference engine 的觀測可能因自適應步長而不落在完全等距的時間格點；因此這裡只
    驗證不可違反的方向性：年齡不可倒退、UTC 時間不可往未來增加，且目前 state 不可比
    最後觀測更年輕或更晚。這能抓出篡改後的序列／future observation，同時不拒絕合法的
    非整數輸出間隔與中途 checkpoint。
    """

    previous: Observation | None = None
    for index, observation in enumerate(observations):
        if previous is not None:
            if observation.age_seconds + 1.0e-12 < previous.age_seconds:
                raise ValueError(f"{label}[{index}] age_seconds 不得遞減")
            if observation.time_utc_ns > previous.time_utc_ns:
                raise ValueError(f"{label}[{index}] time_utc_ns 不得往未來增加")
        previous = observation
    last = observations[-1]
    if state.age_seconds + 1.0e-9 < last.age_seconds:
        raise ValueError(f"{label} 最後 age_seconds 超過目前 state")
    if state.time_utc_ns > last.time_utc_ns:
        raise ValueError(f"{label} time_utc_ns 晚於目前 state")


def _observation_core(value: Observation) -> tuple[object, ...]:
    """提取 engine 同點替換契約固定的 particle identity 與 UTC 時間。

    engine 的 ``_append_or_replace_observation`` 以 particle、time 與「絕對誤差不超過
    1e-12 秒」的 age 判斷是否更新最後一筆 observation；同點的 status、context 及邊界
    修正後的 x/y/z 都可能合法變更。checkpoint continuation 必須遵循這個既有契約，否則
    會把合法的 surface recovery 誤判成歷史篡改。較早 stable row 仍由完整 dataclass
    equality 驗證。
    """

    return (
        value.particle_id,
        value.time_utc_ns,
    )


def _require_observation_core_match(
    expected: Observation,
    actual: Observation,
    *,
    label: str,
) -> None:
    """核對跨 generation observation 核心；status/context 更新仍由 engine 契約允許。

    同一輸出點的 engine 更新可改變生命週期 status、環境／速度 context，以及邊界修正後
    的位置；particle identity、UTC time 必須相同，age 則遵循 engine 使用的絕對容差
    ``1e-12`` 秒。超過此範圍代表呼叫端換成另一筆 observation；這種替換若只依 cursor
    切片會在 restore 時靜默遺失，因此直接以 engine 的同點 key 比對拒絕。
    """

    if _observation_core(expected) != _observation_core(actual) or not bool(
        np.isclose(expected.age_seconds, actual.age_seconds, rtol=0.0, atol=1.0e-12)
    ):
        raise ValueError(f"{label} observation core 不一致")


def _validate_adjacent_observation_engine_keys(
    observations: Sequence[Observation],
    *,
    start: int,
    label: str,
) -> None:
    """檢查本代新增觀測與 pending 邊界不得留下 engine 同點重複列。

    引擎的 ``_append_or_replace_observation`` 對同一粒子只保留最後一筆相同 UTC／age
    的觀測，age 使用絕對容差 ``1e-12`` 秒。若外部呼叫端以 ``append``、``extend``、
    ``insert``、空 slice insertion、``+=`` 或 ``*=`` 繞過 history tracker，可能把原本
    應被替換的 pending row 變成兩筆相鄰資料；writer 若只依 cursor 切片，這個假列會
    在 restore 後形成零長度時間區段。因此這裡在發布前以同一 engine key 做 fail-closed
    檢查。

    ``start`` 是上一代已發布的 observation cursor。為維持分段 checkpoint 的線性寫入
    成本，只檢查 ``start`` 位置與其後的新列，並保留前一列作為邊界比較；不重掃已發布
    的更早 history。範圍包含最後一筆 pending observation，因為同點假列也可能只出現在
    stable segment 與 pending 的交界。schema 2 migration 的 ``start=0`` 會一次檢查完整
    舊 history，這是遷移 root 的一次性驗證成本。
    """

    first_pair = max(1, start)
    for index in range(first_pair, len(observations)):
        previous = observations[index - 1]
        current = observations[index]
        if _observation_core(previous) == _observation_core(current) and bool(
            np.isclose(previous.age_seconds, current.age_seconds, rtol=0.0, atol=1.0e-12)
        ):
            raise ValueError(
                f"{label}[{index - 1}:{index + 1}] 相鄰 observation engine key 重複"
            )


def _serialize_and_validate_writer_state(
    state: ParticleState,
    *,
    expected_identity: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    """以 loader 同一套 primitive／cross-field 規則預驗 compact current state。

    ``ParticleState`` 是執行期 dataclass，並不會自動拒絕 bool 充當整數、浮點充當 UTC
    奈秒或錯誤的 own-local-exit 型別；若 writer 只呼叫 ``asdict``，可能先發布一個
    loader 必定拒絕的 generation。這裡把即將寫入的 JSON 先送回嚴格 decoder，並核對
    反序列化後的固定 identity。回傳同一份已驗證的 payload，供 compact 重用，避免在
    建立 partial 後才發現 semantic mismatch。
    """

    payload = _serialize_particle_state(state)
    validated = _deserialize_particle_state(payload, label=label)
    expected = (
        expected_identity["particle_id"],
        expected_identity["scenario_id"],
        expected_identity["member_id"],
        expected_identity["study_site_id"],
        expected_identity["analysis_region_id"],
        expected_identity["receptor_id"],
    )
    if _state_identity(validated) != expected:
        raise ValueError(f"{label} identity 與 RunUnit identity 不一致")
    return payload


def _serialize_and_validate_writer_observation(
    observation: Observation,
    *,
    expected_particle_id: str,
    label: str,
) -> dict[str, Any]:
    """預驗即將寫入的 observation，確保 writer 與 loader 的欄位語意一致。

    驗證範圍只包含本代新增 rows 與最後 pending row；已發布的更早 prefix 已在上一代
    chain 驗證，避免每次 checkpoint 重新掃描完整 history。除了 Observation constructor
    所涵蓋的環境／速度欄位，也明確核對 particle_id，防止新列被錯誤粒子使用而直到
    restore 才失敗。回傳的 payload 供 segment 或 compact 直接寫入。
    """

    payload = _serialize_observation(observation, schema_version=_SCHEMA30_VERSION)
    validated = _deserialize_observation(
        payload,
        label=label,
        schema_version=_SCHEMA30_VERSION,
    )
    if validated.particle_id != expected_particle_id:
        raise ValueError(f"{label} particle_id 與 RunUnit identity 不一致")
    return payload


def _serialize_and_validate_writer_event(
    event: BoundaryEvent,
    *,
    expected_identity: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    """預驗本代新增 event 的 JSON primitive、fraction 與完整 particle identity。

    事件欄位在執行期由 frozen dataclass 承載，但 constructor 不負責所有 JSON 邊界型別；
    例如 fraction 超過 1 或 UTC 使用 bool 都可能在 writer 直接被序列化。先以 loader 的
    strict decoder 還原一次，再比對六個固定 identity，才能保證 generation 一旦發布就
    可立即由同一 loader 讀回。只處理 event cursor 之後的增量列，維持線性 checkpoint 成本。
    """

    payload = _serialize_event(event)
    validated = _deserialize_event(payload, label=label)
    if (
        validated.particle_id != expected_identity["particle_id"]
        or validated.scenario_id != expected_identity["scenario_id"]
        or validated.member_id != expected_identity["member_id"]
        or validated.study_site_id != expected_identity["study_site_id"]
        or validated.analysis_region_id != expected_identity["analysis_region_id"]
        or validated.receptor_id != expected_identity["receptor_id"]
    ):
        raise ValueError(f"{label} identity 與 RunUnit identity 不一致")
    return payload


def _validate_legacy_execution_prefix(
    current: ParticleExecutionState,
    legacy: ParticleExecutionState,
    *,
    label: str,
) -> None:
    """確認 schema 2 migration 的 current history 仍以 legacy execution 為前綴。

    legacy payload 會被完整載入一次，因此遷移 root 可以逐筆驗證，而不必把舊目錄改寫。
    最後一筆 observation 允許 engine 既有的同點 status/context 更新；其 particle、UTC
    與 age（絕對容差 1e-12 秒）核心欄位，以及所有更早 observation、event row，則必須
    逐欄相同。current 只能追加資料，不能刪除或重排 legacy history。
    """

    if len(current.observations) < len(legacy.observations):
        raise ValueError(f"{label} observations 不再以 legacy 為前綴")
    if len(current.events) < len(legacy.events):
        raise ValueError(f"{label} events 不再以 legacy 為前綴")
    for index, (expected, actual) in enumerate(
        zip(legacy.observations, current.observations, strict=False)
    ):
        if index == len(legacy.observations) - 1:
            _require_observation_core_match(
                expected,
                actual,
                label=f"{label}.observations[{index}]",
            )
        elif expected != actual:
            raise ValueError(f"{label}.observations[{index}] stable row 不一致")
    for index, (expected, actual) in enumerate(
        zip(legacy.events, current.events, strict=False)
    ):
        if expected != actual:
            raise ValueError(f"{label}.events[{index}] stable row 不一致")


def _validate_terminal_continuation(
    execution: ParticleExecutionState,
    previous_record: Mapping[str, Any],
    *,
    rng_state: Mapping[str, Any],
    triangle_hint: int,
    label: str,
) -> None:
    """凍結上一代已終止粒子的 current compact 與 history 邊界。

    粒子一旦進入非 ``ACTIVE`` 狀態，engine 不應再替它產生任何步進、事件或觀測；
    公開 writer 也必須把這項生命週期契約當成資料完整性閘門。這裡只比較上一代 compact
    保存的 O(P) 輕量欄位：完整 ``ParticleState``、步數與最小步長夾制計數、下一個輸出
    年齡、history cursors、pending observation、RNG state 與三角形提示。既有 history
    tracker 會另外拒絕已發布 prefix 的內容修改，因而不需要為每個 generation 重讀整條
    歷史來建立平方級驗證成本。任何新增或替換 terminal particle 的資料都直接拒絕，避免
    產生 loader 可讀但執行語意已分歧的後代 checkpoint。
    """

    previous_state = _deserialize_particle_state(
        previous_record["state"], label=f"{label}.previous.state"
    )
    if previous_state.status is ParticleStatus.ACTIVE:
        return

    if _serialize_particle_state(execution.state) != previous_record["state"]:
        raise ValueError(f"{label} terminal state 不可在後代 generation 改變")
    previous_step_count = _nonnegative_int(
        previous_record["step_count"], label=f"{label}.previous.step_count"
    )
    previous_minimum_clamp_count = _nonnegative_int(
        previous_record["minimum_clamp_count"], label=f"{label}.previous.minimum_clamp_count"
    )
    previous_next_output_age_seconds = _finite_float(
        previous_record["next_output_age_seconds"],
        label=f"{label}.previous.next_output_age_seconds",
    )
    previous_observation_cursor = _nonnegative_int(
        previous_record["observation_cursor"], label=f"{label}.previous.observation_cursor"
    )
    previous_event_cursor = _nonnegative_int(
        previous_record["event_cursor"], label=f"{label}.previous.event_cursor"
    )
    if execution.step_count != previous_step_count:
        raise ValueError(f"{label} terminal step_count 不可在後代 generation 改變")
    if execution.minimum_clamp_count != previous_minimum_clamp_count:
        raise ValueError(f"{label} terminal minimum_clamp_count 不可在後代 generation 改變")
    if execution.next_output_age_seconds != previous_next_output_age_seconds:
        raise ValueError(f"{label} terminal next_output_age_seconds 不可在後代 generation 改變")
    if len(execution.observations) != previous_observation_cursor + 1:
        raise ValueError(f"{label} terminal observation history 不可新增或刪除")
    if len(execution.events) != previous_event_cursor:
        raise ValueError(f"{label} terminal event history 不可新增或刪除")

    # 即使 pending row 後續只用序列化結果比較，也先完整反序列化一次，確保上一代
    # compact 的 observation 欄位仍符合 engine 的 schema 與有限值契約。
    _deserialize_observation(
        previous_record["pending_observation"],
        label=f"{label}.previous.pending_observation",
        schema_version=_SCHEMA30_VERSION,
    )
    if _serialize_observation(
        execution.observations[-1], schema_version=_SCHEMA30_VERSION
    ) != previous_record["pending_observation"]:
        raise ValueError(f"{label} terminal pending observation 不可在後代 generation 改變")
    if dict(rng_state) != previous_record["rng_state"]:
        raise ValueError(f"{label} terminal RNG state 不可在後代 generation 改變")
    if triangle_hint != _normalize_triangle_hint(
        previous_record["triangle_hint"], label=f"{label}.previous.triangle_hint"
    ):
        raise ValueError(f"{label} terminal triangle hint 不可在後代 generation 改變")


def _legacy_terminal_compact_record(
    execution: ParticleExecutionState,
    *,
    rng_state: Mapping[str, Any],
    triangle_hint: int,
    label: str,
) -> dict[str, Any] | None:
    """把舊 schema 的 terminal execution 轉成 terminal 凍結檢查所需的輕量 record。

    schema 2 沒有 v3 compact 檔，卻仍保存完整的 current state、history、RNG 與三角形
    提示。遷移 root 若只驗證 history 前綴，呼叫端仍可能在寫入 v3 前偷偷改掉已終止粒子
    的位置、計數器或亂數狀態，造成可讀但不可重現的續跑結果。因此這裡只把舊 execution
    已有的 terminal 欄位映射成與 v3 compact 相同的檢查介面；active 粒子回傳 ``None``，
    讓遷移流程維持既有的可前進行為。這個 record 只在記憶體內使用，不會改寫舊檔。
    """

    if execution.state.status is ParticleStatus.ACTIVE:
        return None
    if not execution.observations:
        raise ValueError(f"{label} terminal execution observations 不可為空")
    return {
        "state": _serialize_particle_state(execution.state),
        "step_count": execution.step_count,
        "minimum_clamp_count": execution.minimum_clamp_count,
        "next_output_age_seconds": execution.next_output_age_seconds,
        "observation_cursor": len(execution.observations) - 1,
        "event_cursor": len(execution.events),
        "pending_observation": _serialize_observation(
            execution.observations[-1], schema_version=_SCHEMA30_VERSION
        ),
        "rng_state": deepcopy(dict(rng_state)),
        "triangle_hint": triangle_hint,
    }


def _deserialize_execution(
    value: Any,
    *,
    label: str,
    schema_version: str,
) -> tuple[ParticleExecutionState, int]:
    """依 schema 版本完整還原 execution 並拒絕未知狀態、事件或數值。

    schema 版本在 metadata gate 通過後一路傳入 observation decoder；不能先用最新欄位
    猜測資料格式，否則 2.0 舊檔可能被錯誤地視為「缺欄」或把未知 2.1 欄位靜默忽略。
    execution 的 checksum、identity、序列方向與事件驗證則維持既有語意。
    """

    expected = (
        "state",
        "observations",
        "events",
        "step_count",
        "minimum_clamp_count",
        "next_output_age_seconds",
        "triangle_hint",
    )
    row = _require_exact_keys(value, expected, label=label)
    state = _deserialize_particle_state(row["state"], label=f"{label}.state")
    if not isinstance(row["observations"], list) or not row["observations"]:
        raise ValueError(f"{label}.observations 必須是非空陣列")
    if not isinstance(row["events"], list):
        raise ValueError(f"{label}.events 必須是陣列")
    observations = [
        _deserialize_observation(
            item,
            label=f"{label}.observations[{index}]",
            schema_version=schema_version,
        )
        for index, item in enumerate(row["observations"])
    ]
    events = [
        _deserialize_event(item, label=f"{label}.events[{index}]")
        for index, item in enumerate(row["events"])
    ]
    execution = ParticleExecutionState(
        state=state,
        observations=observations,
        events=events,
        step_count=_nonnegative_int(row["step_count"], label=f"{label}.step_count"),
        minimum_clamp_count=_nonnegative_int(
            row["minimum_clamp_count"], label=f"{label}.minimum_clamp_count"
        ),
        next_output_age_seconds=_finite_float(
            row["next_output_age_seconds"], label=f"{label}.next_output_age_seconds"
        ),
    )
    if execution.next_output_age_seconds <= 0:
        raise ValueError(f"{label}.next_output_age_seconds 必須為正")
    particle_id = execution.state.particle_id
    if any(item.particle_id != particle_id for item in execution.observations):
        raise ValueError(f"{label}.observations particle_id 與 state 不一致")
    _validate_observation_sequence(
        execution.observations,
        execution.state,
        label=f"{label}.observations",
    )
    if any(
        (
            item.particle_id != particle_id
            or item.scenario_id != execution.state.scenario_id
            or item.member_id != execution.state.member_id
            or item.study_site_id != execution.state.study_site_id
            or item.analysis_region_id != execution.state.analysis_region_id
            or item.receptor_id != execution.state.receptor_id
        )
        for item in execution.events
    ):
        raise ValueError(f"{label}.events identity 與 state 不一致")
    return execution, _normalize_triangle_hint(row["triangle_hint"], label=f"{label}.triangle_hint")


def _copy_rng_state(rng: np.random.Generator) -> dict[str, Any]:
    """複製並驗證每粒子 PCG64DXSM state，拒絕其他 generator 造成不可重現。"""

    if not isinstance(rng, np.random.Generator) or rng.bit_generator.__class__.__name__ != "PCG64DXSM":
        raise ValueError("execution checkpoint 只接受 PCG64DXSM Generator")
    state = _json_safe(deepcopy(rng.bit_generator.state))
    _validate_rng_state(state, label="rng_state")
    return state


def _validate_rng_state(value: Any, *, label: str) -> dict[str, Any]:
    """以 NumPy 真正設定 bit_generator 的方式驗證 JSON RNG state 結構。"""

    if type(value) is not dict or value.get("bit_generator") != "PCG64DXSM":
        raise ValueError(f"{label} 必須是 PCG64DXSM state")
    try:
        generator = np.random.Generator(np.random.PCG64DXSM())
        generator.bit_generator.state = deepcopy(value)
    except (AttributeError, KeyError, OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{label} 結構無法恢復") from error
    return value


def build_execution_checkpoint(
    *,
    binding: CheckpointBinding | None,
    sequence: int,
    run_units: Sequence[Any],
    executions: Sequence[ParticleExecutionState],
    rngs: Sequence[np.random.Generator],
    triangle_hints: Sequence[int | None],
) -> ExecutionCheckpoint:
    """建立 execution snapshot，供記憶體檢查及 schema 3.0 磁碟 writer 共用。

    ``run_units``、``executions``、``rngs`` 與 ``triangle_hints`` 必須逐項同序；state 的
    particle／scenario／member／站點／受體／到達時間會立即和 RunUnit 核對。這個函式只
    複製 checkpoint 資料，不序列化 request factory 產生的 forcing 與 geometry。
    """

    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
        raise ValueError("checkpoint sequence 必須是非負整數")
    if binding is not None and not isinstance(binding, CheckpointBinding):
        raise TypeError("binding 必須是 CheckpointBinding 或 None")
    lengths = (len(run_units), len(executions), len(rngs), len(triangle_hints))
    if not lengths[0] or len(set(lengths)) != 1:
        raise ValueError("execution checkpoint 不允許空資料且所有欄位長度必須一致")
    identities = [_run_unit_identity(unit) for unit in run_units]
    identity_tuples = [_run_identity_tuple(identity) for identity in identities]
    if len(set(identity_tuples)) != len(identity_tuples):
        raise ValueError("execution checkpoint RunUnit identity 必須唯一")
    normalized_hints: list[int] = []
    observation_cursors: list[int] = []
    event_cursors: list[int] = []
    copied_executions: list[ParticleExecutionState] = []
    copied_rng_states: list[dict[str, Any]] = []
    for index, (identity, execution, rng, triangle_hint) in enumerate(
        zip(identities, executions, rngs, triangle_hints, strict=True)
    ):
        if not isinstance(execution, ParticleExecutionState):
            raise TypeError(f"execution[{index}] 必須是 ParticleExecutionState")
        if _state_identity(execution.state) != (
            identity["particle_id"],
            identity["scenario_id"],
            identity["member_id"],
            identity["study_site_id"],
            identity["analysis_region_id"],
            identity["receptor_id"],
        ):
            raise ValueError(f"execution[{index}] 與 RunUnit identity 不一致")
        normalized_hint = _normalize_triangle_hint(triangle_hint, label=f"triangle_hints[{index}]")
        _serialize_execution(execution, triangle_hint=normalized_hint)
        normalized_hints.append(normalized_hint)
        observation_cursors.append(len(execution.observations))
        event_cursors.append(len(execution.events))
        copied_executions.append(deepcopy(execution))
        copied_rng_states.append(_copy_rng_state(rng))
    return ExecutionCheckpoint(
        binding=binding,
        sequence=sequence,
        run_unit_identities=deepcopy(identities),
        executions=copied_executions,
        rng_states=copied_rng_states,
        triangle_hints=normalized_hints,
        schema_version=_WRITER_SCHEMA_VERSION,
        observation_cursors=observation_cursors,
        event_cursors=event_cursors,
    )


def _write_execution_checkpoint_schema22(
    destination: str | Path,
    *,
    binding: CheckpointBinding,
    run_units: Sequence[Any],
    executions: Sequence[ParticleExecutionState],
    rngs: Sequence[np.random.Generator],
    triangle_hints: Sequence[int | None],
    sequence: int,
) -> Path:
    """以相容用途原子寫入 schema 2.2.0 execution checkpoint。

    目標已存在時拒絕覆寫；寫入期間使用同父目錄的 ``.partial-*``，只有 execution JSON、
    RNG JSON、大小與 SHA-256 manifest 都成功建立後才以 ``os.replace`` 發布。空 shard、
    duplicate identity、非 PCG64DXSM、未知狀態或不一致欄位會在建立 partial 前拒絕。新
    writer 沒有 downgrade 參數；schema 2.0／2.1 僅由 loader 相容讀取，不能由新寫入流程
    製造。這個 fixture helper 固定寫出 2.2 的 11 個速度欄位，即使 observation 是 ``NOT_SAMPLED``
    也會保存其明示的狀態與 ``None`` 缺值，而不會將舊資料偽裝成速度證據。
    此函式只保留給需要建立舊版 fixture 的內部相容路徑；正式 ``ProductionBatch`` writer
    使用下面的 ``write_execution_checkpoint``，固定發布 schema 3.0.0，避免每次 checkpoint
    重寫完整 observation／event history。
    """

    if not isinstance(binding, CheckpointBinding):
        raise TypeError("binding 必須是 CheckpointBinding")
    snapshot = build_execution_checkpoint(
        binding=binding,
        sequence=sequence,
        run_units=run_units,
        executions=executions,
        rngs=rngs,
        triangle_hints=triangle_hints,
    )
    target = Path(destination)
    if target.exists():
        raise FileExistsError(f"不可覆寫 checkpoint：{target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.parent / f".{target.name}.partial-{uuid4().hex}"
    try:
        partial.mkdir()
        execution_records = []
        observation_count = 0
        event_count = 0
        for identity, execution, triangle_hint in zip(
            snapshot.run_unit_identities,
            snapshot.executions,
            snapshot.triangle_hints,
            strict=True,
        ):
            execution_payload = _serialize_execution(
                execution,
                triangle_hint=triangle_hint,
                # 這個私有 helper 只供建立舊版 fixture；正式 writer 由下方
                # ``write_execution_checkpoint`` 固定發布 schema 3.0.0。
                schema_version=_SCHEMA22_VERSION,
            )
            execution_records.append({"identity": identity, "execution": execution_payload})
            observation_count += len(execution.observations)
            event_count += len(execution.events)
        execution_path = partial / "execution_state.json"
        rng_path = partial / "rng_states.json"
        _write_json(execution_path, {"records": execution_records})
        _write_json(
            rng_path,
            {
                "records": [
                    {"particle_id": identity["particle_id"], "rng_state": rng_state}
                    for identity, rng_state in zip(
                        snapshot.run_unit_identities, snapshot.rng_states, strict=True
                    )
                ]
            },
        )
        files = {
            path.name: {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in (execution_path, rng_path)
        }
        metadata = {
            "schema_version": _SCHEMA22_VERSION,
            "sequence": snapshot.sequence,
            "particle_count": len(snapshot.executions),
            "observation_count": observation_count,
            "event_count": event_count,
            "binding": _binding_payload(binding),
            "particle_order": [identity["particle_id"] for identity in snapshot.run_unit_identities],
            "files": files,
        }
        _write_json(partial / "checkpoint.json", metadata, sort_keys=True)
        os.replace(partial, target)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return target


def _binding_from_payload(value: Any, *, label: str = "binding") -> CheckpointBinding:
    """嚴格解析 JSON binding，讓 2.x 與 3.0 loader 共用同一個 identity gate。"""

    if not isinstance(value, dict):
        raise ValueError(f"{label} 必須是 object")
    keys = set(value)
    if keys == set(_BINDING_FIELDS):
        row = _require_exact_keys(value, _BINDING_FIELDS, label=label)
    elif keys == set(_BINDING_FIELDS_WITH_RANDOM):
        row = _require_exact_keys(value, _BINDING_FIELDS_WITH_RANDOM, label=label)
        if type(row["random_stream_id"]) is not str or not row["random_stream_id"].strip():
            raise ValueError(f"{label}.random_stream_id 必須是非空白字串")
    else:
        raise ValueError(f"{label} 欄位集合不符")
    try:
        return CheckpointBinding(**row)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} 欄位無效") from error


def _sha256_or_none(value: Any, *, label: str) -> str | None:
    """解析 segment chain 使用的可空 SHA-256，避免空字串偽裝 chain root。"""

    if value is None:
        return None
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} 必須是小寫 SHA-256 或 None")
    return value


class _CheckpointTrackedAttributes(dict[str, Any]):
    """追蹤已發布 BoundaryEvent.attributes 的巢狀 dict 修改。"""

    __slots__ = ("_checkpoint_owner", "_checkpoint_stable")

    def __init__(
        self,
        value: Mapping[str, Any] | None = None,
        *,
        owner: _CheckpointHistoryList | None = None,
        stable: bool = False,
    ) -> None:
        """建立 attributes 追蹤器；stable 只用於上一代已發布 event。"""

        super().__init__({} if value is None else value)
        self._checkpoint_owner = owner
        self._checkpoint_stable = stable

    def _mark_if_stable(self) -> None:
        """巢狀欄位改動若屬已發布 event，標記 owner history dirty。"""

        if self._checkpoint_stable and self._checkpoint_owner is not None:
            self._checkpoint_owner._checkpoint_history_dirty = True

    def __setitem__(self, key: str, value: Any) -> None:
        """追蹤 attributes 單欄修改。"""

        self._mark_if_stable()
        super().__setitem__(key, value)

    def __delitem__(self, key: str) -> None:
        """追蹤 attributes 單欄刪除。"""

        self._mark_if_stable()
        super().__delitem__(key)

    def clear(self) -> None:
        """追蹤 attributes 清除。"""

        self._mark_if_stable()
        super().clear()

    _MISSING = object()

    def pop(self, key: str, default: Any = _MISSING) -> Any:
        """追蹤 attributes pop。"""

        if key in self:
            self._mark_if_stable()
        if default is self._MISSING:
            return super().pop(key)
        return super().pop(key, default)

    def popitem(self) -> tuple[str, Any]:
        """追蹤 attributes popitem。"""

        self._mark_if_stable()
        return super().popitem()

    def setdefault(self, key: str, default: Any = None) -> Any:
        """追蹤 attributes setdefault。"""

        if key not in self:
            self._mark_if_stable()
        return super().setdefault(key, default)

    def update(self, *args: Any, **kwargs: Any) -> None:
        """追蹤 attributes update。"""

        if args or kwargs:
            self._mark_if_stable()
        super().update(*args, **kwargs)

    def __ior__(self, value: Mapping[str, Any]) -> _CheckpointTrackedAttributes:
        """追蹤 ``attributes |= mapping``；dict 內建實作不一定呼叫 update hook。"""

        self._mark_if_stable()
        super().__ior__(value)
        return self

    def mark_checkpoint_stable(self, stable: bool) -> None:
        """設定此 attributes 是否屬於已發布 event。"""

        self._checkpoint_stable = bool(stable)

    def __deepcopy__(self, memo: dict[int, Any]) -> dict[str, Any]:
        """deepcopy 時輸出無 owner 的普通 dict，避免複製追蹤器循環參照。"""

        copied = {deepcopy(key, memo): deepcopy(value, memo) for key, value in self.items()}
        memo[id(self)] = copied
        return copied


class _CheckpointHistoryList(list[Any]):
    """追蹤已發布歷史前綴，拒絕把 immutable row 靜默換成另一筆資料。

    ``ParticleExecutionState`` 的 observations/events 在 engine 內仍是可變 list：最新
    observation 可能因同一時間／位置的 status 或 context 更新而被替換，事件則只會追加。
    schema 3 writer 若只依 cursor 切片，呼叫端把較早的 stable row 改掉時會悄悄忽略該改動，
    造成記憶體中的 execution 與 restore 後的 chain 不一致。這個 list subclass 記住上次
    發布後的 immutable prefix 長度；對該前綴的替換、刪除、插入或排序會標記 dirty，writer
    在下一次發布前 fail-closed。prefix 之後的 observation pending/context 更新與 event
    append 仍符合 engine 契約。這項追蹤只保存一個整數與旗標，不複製歷史資料。
    """

    __slots__ = ("_checkpoint_immutable_prefix_length", "_checkpoint_history_dirty")

    def __init__(self, iterable: Sequence[Any] = (), *, immutable_prefix_length: int = 0) -> None:
        """建立可追蹤 list；prefix 長度代表最近一次成功發布的穩定 row 數。"""

        super().__init__()
        self._checkpoint_immutable_prefix_length = 0
        self._checkpoint_history_dirty = False
        self.extend(iterable)
        self.mark_checkpoint_prefix(immutable_prefix_length)

    def _mark_if_prefix_touched(self, index: int) -> None:
        """若索引落在 immutable prefix 內，記錄不可接受的歷史修改。"""

        normalized = index if index >= 0 else len(self) + index
        if normalized < self._checkpoint_immutable_prefix_length:
            self._checkpoint_history_dirty = True

    def __setitem__(self, index: int | slice, value: Any) -> None:
        """追蹤單項或區段替換；pending row 以後的修改仍可由 engine 使用。"""

        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            touched = list(range(start, stop, step))
            # list 的 slice assignment 接受任意 iterable；先 materialize，除了讓 tuple、
            # generator 等呼叫形式可建立 event tracker，也能判斷這次是否真的插入了列。
            value = list(value)
            # 空 slice assignment 是插入操作；只要插入點落在 immutable prefix 中，
            # 且該位置後仍有既有列，即使原本沒有被替換的 index，也會改變後續 cursor，
            # 必須標記 dirty。插入點等於 len(self) 是合法的尾端追加，不應誤標記。
            insertion_point = start if step > 0 else stop
            if (
                any(item < self._checkpoint_immutable_prefix_length for item in touched)
                or insertion_point < self._checkpoint_immutable_prefix_length
                or (
                    value
                    and index.step is None
                    and start == stop
                    and insertion_point == self._checkpoint_immutable_prefix_length
                    and insertion_point < len(self)
                )
            ):
                self._checkpoint_history_dirty = True
            if index.step is None:
                value = [
                    self._wrap_event(
                        item,
                        stable=start < self._checkpoint_immutable_prefix_length,
                    )
                    for item in value
                ]
            else:
                value = [
                    self._wrap_event(
                        item,
                        stable=position < self._checkpoint_immutable_prefix_length,
                    )
                    for position, item in zip(touched, value, strict=True)
                ]
        else:
            self._mark_if_prefix_touched(index)
            value = self._wrap_event(
                value,
                stable=(index if index >= 0 else len(self) + index)
                < self._checkpoint_immutable_prefix_length,
            )
        super().__setitem__(index, value)

    def __delitem__(self, index: int | slice) -> None:
        """追蹤刪除，避免 stable prefix 被縮短或重排。"""

        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if any(
                item < self._checkpoint_immutable_prefix_length
                for item in range(start, stop, step)
            ):
                self._checkpoint_history_dirty = True
        else:
            self._mark_if_prefix_touched(index)
        super().__delitem__(index)

    def _wrap_event(self, value: Any, *, stable: bool) -> Any:
        """將 BoundaryEvent 的巢狀 attributes 綁回此 history tracker。"""

        if not isinstance(value, BoundaryEvent):
            return value
        if (
            isinstance(value.attributes, _CheckpointTrackedAttributes)
            and value.attributes._checkpoint_owner is self
        ):
            # 同一 BoundaryEvent 可能被呼叫端 append／insert 到另一個位置；直接共用
            # owner tracker 並把 stable 降成 False 會讓原本已發布 row 失去 dirty
            # 追蹤。只有 stable 狀態相同才能安全重用，否則複製 event 與 attributes。
            if value.attributes._checkpoint_stable == stable:
                return value
            return replace(
                value,
                attributes=_CheckpointTrackedAttributes(
                    value.attributes,
                    owner=self,
                    stable=stable,
                ),
            )
        return replace(
            value,
            attributes=_CheckpointTrackedAttributes(value.attributes, owner=self, stable=stable),
        )

    def append(self, value: Any) -> None:
        """追加 event 時建立未發布 attributes tracker；observation 則原樣追加。"""

        super().append(self._wrap_event(value, stable=False))

    def extend(self, values: Sequence[Any]) -> None:
        """逐項追加並追蹤新 event 的 attributes。"""

        for value in values:
            self.append(value)

    def insert(self, index: int, value: Any) -> None:
        """追蹤插入；插入 stable prefix 會改變既有 row cursor，必須拒絕。"""

        # list.insert 會把超出範圍的 index 夾到 [0, len]；使用實際插入位置判定，
        # 避免把「插入 prefix 尾端且 prefix == len」誤判成修改 immutable rows。
        normalized = min(max(index if index >= 0 else len(self) + index, 0), len(self))
        if normalized < self._checkpoint_immutable_prefix_length or (
            normalized == self._checkpoint_immutable_prefix_length and normalized < len(self)
        ):
            self._checkpoint_history_dirty = True
        super().insert(
            index,
            self._wrap_event(value, stable=normalized < self._checkpoint_immutable_prefix_length),
        )

    def clear(self) -> None:
        """清空 history 會觸碰任何既有 prefix，故先標記 dirty。"""

        if self._checkpoint_immutable_prefix_length:
            self._checkpoint_history_dirty = True
        super().clear()

    def pop(self, index: int = -1) -> Any:
        """追蹤移除 history row。"""

        # 只有實際成功移除 row 才標記 dirty；若 index 無效，list.pop 原本應拋出
        # IndexError，不能把「沒有任何資料變更」誤記成永久不可續跑。
        normalized = index if index >= 0 else len(self) + index
        if 0 <= normalized < len(self):
            self._mark_if_prefix_touched(index)
        return super().pop(index)

    def remove(self, value: Any) -> None:
        """追蹤依值移除 history row；位置未知時以 dirty 保守處理。"""

        try:
            index = self.index(value)
        except ValueError:
            # 與普通 list.remove 相同，找不到資料時不應改變 tracker 狀態。
            raise
        if index < self._checkpoint_immutable_prefix_length:
            self._checkpoint_history_dirty = True
        super().remove(value)

    def reverse(self) -> None:
        """反轉 history 會改變 stable row 順序，直接標記 dirty。"""

        if self:
            self._checkpoint_history_dirty = True
        super().reverse()

    def sort(self, *args: Any, **kwargs: Any) -> None:
        """排序 history 會改變 stable row 順序，直接標記 dirty。"""

        if self:
            self._checkpoint_history_dirty = True
        super().sort(*args, **kwargs)

    def __imul__(self, value: int) -> _CheckpointHistoryList:
        """追蹤重複或清空 list 的 in-place 操作。"""

        if self and self._checkpoint_immutable_prefix_length and value != 1:
            self._checkpoint_history_dirty = True
        if value == 1:
            return self
        return super().__imul__(value)

    def __iadd__(self, values: Sequence[Any]) -> _CheckpointHistoryList:
        """追蹤 in-place 追加。"""

        self.extend(values)
        return self

    @property
    def checkpoint_immutable_prefix_length(self) -> int:
        """回傳最近一次發布時承諾不可變的 row 數。"""

        return self._checkpoint_immutable_prefix_length

    @property
    def checkpoint_history_dirty(self) -> bool:
        """回傳 stable prefix 是否曾被呼叫端修改。"""

        return self._checkpoint_history_dirty

    def mark_checkpoint_prefix(self, length: int) -> None:
        """在成功發布後設定新的 stable prefix，清除本次已驗證的 dirty 標記。"""

        if type(length) is not int or length < 0 or length > len(self):
            raise ValueError("checkpoint immutable prefix 長度無效")
        previous_length = self._checkpoint_immutable_prefix_length
        if length < previous_length:
            raise ValueError("checkpoint immutable prefix 不可回退")
        self._checkpoint_immutable_prefix_length = length
        self._checkpoint_history_dirty = False
        # 觀測列是 immutable dataclass，不含可變 attributes；不必在每次 checkpoint 重掃
        # 完整歷史。事件的 attributes 需要轉成 stable tracker，但只處理這次新發布的
        # prefix 區間，讓 cadence 次數增加時 CPU 成本仍與新增 event row 數近似成正比。
        for index in range(previous_length, length):
            value = self[index]
            if isinstance(value, BoundaryEvent) and isinstance(
                value.attributes, _CheckpointTrackedAttributes
            ):
                value.attributes.mark_checkpoint_stable(index < length)
            elif isinstance(value, BoundaryEvent):
                # 這個分支只會在外部以底層 list API 躲過一般 hook 時出現；發布前仍把
                # event attributes 補上 tracker，避免之後的巢狀修改被靜默忽略。
                super().__setitem__(
                    index,
                    replace(
                        value,
                        attributes=_CheckpointTrackedAttributes(
                            value.attributes,
                            owner=self,
                            stable=True,
                        ),
                    ),
                )

    def __deepcopy__(self, memo: dict[int, Any]) -> _CheckpointHistoryList:
        """建立獨立追蹤 list，避免 snapshot 複製回原 owner。"""

        copied = _CheckpointHistoryList()
        memo[id(self)] = copied
        for value in self:
            copied.append(deepcopy(value, memo))
        # 先維持 copied 的 prefix=0，再逐步標記到原 prefix；若先把 prefix 設成目標值，
        # ``mark_checkpoint_prefix`` 只會掃描新增區間的設計就不會走訪既有 BoundaryEvent，
        # 造成 deep-copy 後的 stable attributes 仍被誤標為可變，巢狀修改便可能繞過 dirty
        # gate。最後恢復原 dirty 旗標，保留 snapshot 建立前已觀測到的篡改證據。
        original_prefix = self._checkpoint_immutable_prefix_length
        original_dirty = self._checkpoint_history_dirty
        copied.mark_checkpoint_prefix(original_prefix)
        copied._checkpoint_history_dirty = original_dirty
        return copied


def _track_execution_history(
    execution: ParticleExecutionState,
    *,
    observation_prefix_length: int = 0,
    event_prefix_length: int = 0,
) -> None:
    """把 execution 的兩條歷史 list 置於 prefix 追蹤狀態，供 writer/resume 共用。

    初次建立 root 時兩個 prefix 都是零；從已發布 schema 3 restore 時，呼叫端傳入上一代
    compact cursor 對應的 stable／event prefix。若外部以整個 plain list 取代既有追蹤物件，
    caller 可先以本函式建立新物件，但 schema 3 continuation 仍會由 writer 的型別 gate
    拒絕，避免把未追蹤的替換歷史當成合法延續。
    """

    for attribute, prefix_length, label in (
        ("observations", observation_prefix_length, "observations"),
        ("events", event_prefix_length, "events"),
    ):
        current = getattr(execution, attribute)
        if isinstance(current, _CheckpointHistoryList):
            if current.checkpoint_immutable_prefix_length != prefix_length:
                raise ValueError(f"execution {label} checkpoint prefix 長度不一致")
            continue
        setattr(
            execution,
            attribute,
            _CheckpointHistoryList(current, immutable_prefix_length=prefix_length),
        )


def _reset_checkpoint_history_tracking(execution: ParticleExecutionState) -> None:
    """把 execution 的歷史重新視為新 chain root，保留列內容但清除舊 owner 狀態。

    獨立 root writer 可以合法接收從另一份已載入 checkpoint deep-copy 出來的 execution；
    這時原 list 可能仍帶著前一條 chain 的 immutable prefix 或 dirty 旗標。root 沒有前代
    可供核對，應將目前所有 rows 當成新 chain 的輸入，而不是誤套用舊 cursor。migration
    root 也先以 legacy payload 驗證 prefix，再使用相同的重置，讓新 chain 自己管理 nested
    event attributes。schema 3 continuation 不呼叫本函式，因為那條路徑必須保留原 tracker
    並嚴格核對前代 cursor。
    """

    for attribute in ("observations", "events"):
        current = getattr(execution, attribute)
        if not isinstance(current, _CheckpointHistoryList):
            continue
        if current.checkpoint_immutable_prefix_length == 0 and not current.checkpoint_history_dirty:
            continue
        setattr(execution, attribute, _CheckpointHistoryList(current))


def _verify_schema3_files(root: Path, metadata: dict[str, Any]) -> None:
    """驗證 schema 3 generation 的固定檔案拓撲與兩份 payload checksum。

    schema 3 把 immutable history segment 與 compact current state 分成兩個普通檔案。
    兩者都在同一個 partial directory 完成後才發布；loader 先確認目錄和 checksum，再
    解析資料，避免將殘留 partial、symlink 或截斷檔當成可續跑狀態。
    """

    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"schema 3 checkpoint 根目錄必須是普通目錄：{root}")
    expected_names = _SCHEMA3_DATA_FILES | {"checkpoint.json"}
    entries = tuple(root.iterdir())
    if {entry.name for entry in entries} != expected_names:
        raise ValueError("schema 3 checkpoint 目錄含有遺失或未知檔案")
    for entry in entries:
        if entry.is_symlink() or not entry.is_file():
            raise ValueError(f"schema 3 checkpoint 只允許固定普通檔案：{entry.name}")
    files = metadata.get("files")
    if not isinstance(files, dict) or set(files) != _SCHEMA3_DATA_FILES:
        raise ValueError("schema 3 checkpoint files manifest 不完整或含未知檔案")
    for filename in sorted(_SCHEMA3_DATA_FILES):
        contract = files[filename]
        if not isinstance(contract, dict) or set(contract) != {"size_bytes", "sha256"}:
            raise ValueError(f"schema 3 {filename} checksum contract 不完整")
        size = contract["size_bytes"]
        digest = contract["sha256"]
        if type(size) is not int or size < 0:
            raise ValueError(f"schema 3 {filename}.size_bytes 無效")
        if type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"schema 3 {filename}.sha256 無效")
        path = root / filename
        if path.stat().st_size != size or sha256_file(path) != digest:
            raise ValueError(f"schema 3 {filename} checksum 或 size 不符")


def _read_schema3_generation_header(
    root: Path,
    *,
    expected_binding: CheckpointBinding,
    expected_run_units: Sequence[Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """讀取 schema 3 generation 的 compact 與 segment metadata，不建立歷史 dataclass。

    writer 在建立下一代 segment 時只需要上一代 compact cursor 與 checkpoint.json SHA；這個
    helper 仍會讀取並 JSON parse 該代完整 history payload，檢查 checksum、固定欄位、cursor
    與 row count，但不把每一列反序列化成 ``Observation``／``BoundaryEvent``。完整 hash
    chain 與 rows 仍由 ``_load_schema3_execution_checkpoint`` 在 resume／validation 時逐代
    還原與驗證，避免 checkpoint cadence 因重建 dataclass 引入平方級 CPU 成本。
    """

    metadata = _read_json(root / "checkpoint.json")
    expected_metadata_keys = (
        "schema_version",
        "sequence",
        "particle_count",
        "observation_count",
        "event_count",
        "binding",
        "particle_order",
        "files",
        "segment",
        "chain_root_sequence",
        "legacy_source",
    )
    metadata = _require_exact_keys(metadata, expected_metadata_keys, label="schema 3 metadata")
    if metadata["schema_version"] != _SCHEMA30_VERSION:
        raise ValueError("schema 3 generation schema_version 不符")
    sequence = _nonnegative_int(metadata["sequence"], label="schema 3 sequence")
    if sequence < 1:
        raise ValueError("schema 3 sequence 必須從 1 開始")
    chain_root_sequence = _positive_int(
        metadata["chain_root_sequence"], label="schema 3 chain_root_sequence"
    )
    if chain_root_sequence > sequence:
        raise ValueError("schema 3 chain_root_sequence 不可晚於 sequence")
    legacy_source = metadata["legacy_source"]
    if legacy_source is not None:
        legacy_source = _require_exact_keys(
            legacy_source,
            (
                "schema_version",
                "sequence",
                "relative_directory",
                "checkpoint_json_sha256",
            ),
            label="schema 3 legacy source",
        )
        if legacy_source["schema_version"] not in {
            _SCHEMA20_VERSION,
            _SCHEMA21_VERSION,
            _SCHEMA22_VERSION,
        }:
            raise ValueError("schema 3 legacy source schema 不支援")
        # 舊 schema execution checkpoint 的工程序號歷來允許 0；遷移時仍須保留其原值，
        # 例如 checkpoint-00000000 → v3 checkpoint-00000001，不能因 v3 generation 從 1
        # 開始而把 legacy source 誤判成非法。
        legacy_sequence = _nonnegative_int(
            legacy_source["sequence"], label="schema 3 legacy source sequence"
        )
        relative_directory = legacy_source["relative_directory"]
        if (
            type(relative_directory) is not str
            or _SCHEMA3_GENERATION_NAME_RE.fullmatch(relative_directory) is None
        ):
            raise ValueError("schema 3 legacy source relative_directory 不安全")
        if int(relative_directory.removeprefix("checkpoint-")) != legacy_sequence:
            raise ValueError("schema 3 legacy source sequence 與 relative_directory 不一致")
        source_sha = _sha256_or_none(
            legacy_source["checkpoint_json_sha256"],
            label="schema 3 legacy source checkpoint_json_sha256",
        )
        if source_sha is None:
            raise ValueError("schema 3 legacy source checkpoint_json_sha256 不可為 None")
        source_root = root.parent / relative_directory
        if source_root == root or source_root.is_symlink() or not source_root.is_dir():
            raise ValueError("schema 3 legacy source 必須位於同一 parent 的固定目錄")
        source_metadata_path = source_root / "checkpoint.json"
        source_metadata = _read_json(source_metadata_path)
        if source_metadata.get("schema_version") != legacy_source["schema_version"]:
            raise ValueError("schema 3 legacy source schema_version 已改變")
        if (
            _nonnegative_int(
                source_metadata.get("sequence"),
                label="schema 3 legacy source metadata sequence",
            )
            != legacy_sequence
        ):
            raise ValueError("schema 3 legacy source sequence 已改變")
        if sha256_file(source_metadata_path) != source_sha:
            raise ValueError("schema 3 legacy source checkpoint.json SHA-256 不一致")
        _verify_schema2_files(source_root, source_metadata)
        legacy_source["sequence"] = legacy_sequence
    particle_count = _positive_int(metadata["particle_count"], label="schema 3 particle_count")
    observation_count = _nonnegative_int(metadata["observation_count"], label="schema 3 observation_count")
    event_count = _nonnegative_int(metadata["event_count"], label="schema 3 event_count")
    actual_binding = _binding_from_payload(metadata["binding"], label="schema 3 binding")
    if actual_binding != expected_binding:
        raise ValueError(f"checkpoint binding 不相容：actual={actual_binding}, expected={expected_binding}")
    particle_order = metadata["particle_order"]
    if (
        not isinstance(particle_order, list)
        or len(particle_order) != particle_count
        or any(type(item) is not str or not item for item in particle_order)
        or len(set(particle_order)) != particle_count
    ):
        raise ValueError("schema 3 particle_order 無效或含 duplicate")
    segment_meta = _require_exact_keys(
        metadata["segment"],
        ("sequence", "previous_checkpoint_json_sha256", "record_count"),
        label="schema 3 segment metadata",
    )
    segment_sequence = _nonnegative_int(
        segment_meta["sequence"], label="schema 3 segment sequence"
    )
    if segment_sequence != sequence:
        raise ValueError("schema 3 segment sequence 與 generation 不一致")
    previous_checkpoint_json_sha256 = _sha256_or_none(
        segment_meta["previous_checkpoint_json_sha256"],
        label="schema 3 previous_checkpoint_json_sha256",
    )
    if previous_checkpoint_json_sha256 is None:
        if legacy_source is None:
            if sequence != 1 or chain_root_sequence != sequence:
                raise ValueError("schema 3 無 legacy source 的 root 必須是 sequence 1")
        elif legacy_source["sequence"] != sequence - 1 or chain_root_sequence != sequence:
            raise ValueError("schema 3 legacy source 必須緊接 root generation")
    elif legacy_source is not None:
        raise ValueError("schema 3 continuation 不得帶 legacy source")
    segment_record_count = _positive_int(
        segment_meta["record_count"], label="schema 3 segment record_count"
    )
    if segment_record_count != particle_count:
        raise ValueError("schema 3 segment record_count 不一致")
    _verify_schema3_files(root, metadata)

    compact = _read_json(root / "compact_state.json")
    compact = _require_exact_keys(
        compact,
        ("schema_version", "sequence", "binding", "particle_order", "records"),
        label="schema 3 compact state",
    )
    compact_sequence = _nonnegative_int(compact["sequence"], label="schema 3 compact sequence")
    if compact["schema_version"] != _SCHEMA30_VERSION or compact_sequence != sequence:
        raise ValueError("schema 3 compact schema/sequence 不一致")
    compact_binding = _binding_from_payload(compact["binding"], label="schema 3 compact binding")
    if compact_binding != expected_binding:
        raise ValueError("schema 3 compact binding 不一致")
    if compact["particle_order"] != particle_order:
        raise ValueError("schema 3 compact particle_order 不一致")
    compact_records = compact["records"]
    if not isinstance(compact_records, list) or len(compact_records) != particle_count:
        raise ValueError("schema 3 compact records 數量不一致")
    identities: list[dict[str, Any]] = []
    for index, record in enumerate(compact_records):
        row = _require_exact_keys(
            record,
            (
                "identity",
                "state",
                "step_count",
                "minimum_clamp_count",
                "next_output_age_seconds",
                "observation_cursor",
                "event_cursor",
                "rng_state",
                "triangle_hint",
                "pending_observation",
            ),
            label=f"schema 3 compact.records[{index}]",
        )
        identity = _validate_run_identity(row["identity"], label=f"compact.records[{index}].identity")
        if identity["particle_id"] != particle_order[index]:
            raise ValueError("schema 3 compact particle order 與 identity 不一致")
        state = _deserialize_particle_state(row["state"], label=f"compact.records[{index}].state")
        if _state_identity(state) != (
            identity["particle_id"],
            identity["scenario_id"],
            identity["member_id"],
            identity["study_site_id"],
            identity["analysis_region_id"],
            identity["receptor_id"],
        ):
            raise ValueError("schema 3 compact state identity 不一致")
        _nonnegative_int(row["step_count"], label=f"compact.records[{index}].step_count")
        _nonnegative_int(
            row["minimum_clamp_count"], label=f"compact.records[{index}].minimum_clamp_count"
        )
        next_age = _finite_float(
            row["next_output_age_seconds"], label=f"compact.records[{index}].next_output_age_seconds"
        )
        if next_age <= 0:
            raise ValueError("schema 3 compact next_output_age_seconds 必須為正")
        observation_cursor = _nonnegative_int(
            row["observation_cursor"],
            label=f"compact.records[{index}].observation_cursor",
        )
        event_cursor = _nonnegative_int(
            row["event_cursor"], label=f"compact.records[{index}].event_cursor"
        )
        pending_observation = _deserialize_observation(
            row["pending_observation"],
            label=f"compact.records[{index}].pending_observation",
            schema_version=_SCHEMA30_VERSION,
        )
        if pending_observation.particle_id != identity["particle_id"]:
            raise ValueError("schema 3 compact pending observation identity 不一致")
        # 粒子可以在兩個輸出觀測點之間多走數個數值步，因此最後一筆觀測不一定
        # 與 compact 的 current state 落在同一時間。只要求它仍位於目前 state 之前，
        # 並保留完整 loader 對歷史順序的進一步檢查。
        if (
            state.age_seconds + 1.0e-9 < pending_observation.age_seconds
            or state.time_utc_ns > pending_observation.time_utc_ns
        ):
            raise ValueError("schema 3 compact pending observation 晚於 current state")
        _validate_rng_state(row["rng_state"], label=f"compact.records[{index}].rng_state")
        _normalize_triangle_hint(row["triangle_hint"], label=f"compact.records[{index}].triangle_hint")
        # 後續 header cursor 比對使用已驗證的 Python int，避免 JSON bool 因為
        # ``True == 1`` 被錯誤視為合法 cursor。
        row["observation_cursor"] = observation_cursor
        row["event_cursor"] = event_cursor
        identities.append(identity)
    if len({_run_identity_tuple(identity) for identity in identities}) != particle_count:
        raise ValueError("schema 3 compact 含 duplicate RunUnit identity")
    compact_observation_count = sum(
        _nonnegative_int(
            record["observation_cursor"],
            label=f"compact.records[{index}].observation_cursor",
        )
        + 1
        for index, record in enumerate(compact_records)
    )
    compact_event_count = sum(
        _nonnegative_int(record["event_cursor"], label=f"compact.records[{index}].event_cursor")
        for index, record in enumerate(compact_records)
    )
    if compact_observation_count != observation_count or compact_event_count != event_count:
        raise ValueError("schema 3 metadata count 與 compact cursor 不一致")
    if expected_run_units is not None:
        expected_identities = [_run_unit_identity(unit) for unit in expected_run_units]
        if expected_identities != identities:
            raise ValueError("checkpoint RunUnit identity/order 與目前 shard 不一致")

    segment = _read_json(root / "history_segment.json")
    segment = _require_exact_keys(
        segment,
        (
            "schema_version",
            "sequence",
            "binding",
            "particle_order",
            "previous_checkpoint_json_sha256",
            "records",
        ),
        label="schema 3 history segment",
    )
    segment_sequence = _nonnegative_int(segment["sequence"], label="schema 3 history sequence")
    if segment["schema_version"] != _SCHEMA30_VERSION or segment_sequence != sequence:
        raise ValueError("schema 3 history segment schema/sequence 不一致")
    segment_binding = _binding_from_payload(segment["binding"], label="schema 3 segment binding")
    if segment_binding != expected_binding:
        raise ValueError("schema 3 history segment binding 不一致")
    if segment["particle_order"] != particle_order:
        raise ValueError("schema 3 segment particle_order 不一致")
    if _sha256_or_none(
        segment["previous_checkpoint_json_sha256"],
        label="schema 3 segment previous checkpoint hash",
    ) != previous_checkpoint_json_sha256:
        raise ValueError("schema 3 segment previous checkpoint hash 與 metadata 不一致")
    segment_records = segment["records"]
    if not isinstance(segment_records, list) or len(segment_records) != particle_count:
        raise ValueError("schema 3 history segment records 數量不一致")
    for index, record in enumerate(segment_records):
        row = _require_exact_keys(
            record,
            (
                "identity",
                "observation_start_cursor",
                "observation_end_cursor",
                "event_start_cursor",
                "event_end_cursor",
                "observation_row_count",
                "event_row_count",
                "observations",
                "events",
            ),
            label=f"schema 3 history segment.records[{index}]",
        )
        identity = _validate_run_identity(
            row["identity"], label=f"schema 3 segment.records[{index}].identity"
        )
        if identity != identities[index]:
            raise ValueError("schema 3 segment identity/order 不一致")
        observation_start = _nonnegative_int(
            row["observation_start_cursor"],
            label=f"schema 3 segment.records[{index}].observation_start_cursor",
        )
        observation_end = _nonnegative_int(
            row["observation_end_cursor"],
            label=f"schema 3 segment.records[{index}].observation_end_cursor",
        )
        event_start = _nonnegative_int(
            row["event_start_cursor"],
            label=f"schema 3 segment.records[{index}].event_start_cursor",
        )
        event_end = _nonnegative_int(
            row["event_end_cursor"],
            label=f"schema 3 segment.records[{index}].event_end_cursor",
        )
        observations = row["observations"]
        events = row["events"]
        if not isinstance(observations, list) or not isinstance(events, list):
            raise ValueError("schema 3 segment observations/events 必須是陣列")
        observation_row_count = _nonnegative_int(
            row["observation_row_count"],
            label=f"schema 3 segment.records[{index}].observation_row_count",
        )
        event_row_count = _nonnegative_int(
            row["event_row_count"],
            label=f"schema 3 segment.records[{index}].event_row_count",
        )
        if (
            observation_end - observation_start != len(observations)
            or event_end - event_start != len(events)
            or observation_row_count != len(observations)
            or event_row_count != len(events)
        ):
            raise ValueError("schema 3 segment cursor/row count 不一致")
        compact_record = compact_records[index]
        if (
            observation_end
            != _nonnegative_int(
                compact_record["observation_cursor"],
                label=f"schema 3 compact.records[{index}].observation_cursor",
            )
            or event_end
            != _nonnegative_int(
                compact_record["event_cursor"],
                label=f"schema 3 compact.records[{index}].event_cursor",
            )
        ):
            raise ValueError("schema 3 segment end cursor 與 compact cursor 不一致")
    return metadata, compact, segment


def _load_schema3_execution_checkpoint(
    root: Path,
    *,
    expected_binding: CheckpointBinding,
    expected_run_units: Sequence[Any] | None = None,
) -> ExecutionCheckpoint:
    """沿 checkpoint.json SHA-256 chain 還原一份 schema 3 execution checkpoint。

    每一代只保存新增 rows，故還原時必須從 chain root 依粒子固定順序逐段套用 cursor。
    任一代缺失、跳號、hash 不相符、粒子重排、cursor 不連續或 identity/binding 改變都會
    fail-closed；不以較舊 generation 掩蓋損壞歷史。compact state 則提供最後的 current
    state、step counters、輸出游標、亂數 state 與 triangle hint。
    """

    metadata, compact, segment = _read_schema3_generation_header(
        root,
        expected_binding=expected_binding,
        expected_run_units=expected_run_units,
    )
    target_sequence = int(metadata["sequence"])
    chain: list[tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]] = [
        (root, metadata, compact, segment)
    ]
    current = (root, metadata, compact, segment)
    while True:
        previous_checkpoint_json_sha256 = _sha256_or_none(
            current[3]["previous_checkpoint_json_sha256"],
            label="schema 3 previous checkpoint.json hash",
        )
        if previous_checkpoint_json_sha256 is None:
            if int(current[1]["chain_root_sequence"]) != int(current[1]["sequence"]):
                raise ValueError("schema 3 hash chain root metadata 不一致")
            break
        sequence = int(current[1]["sequence"])
        previous_path = root.parent / f"checkpoint-{sequence - 1:08d}"
        if sequence <= 1 or not previous_path.is_dir() or previous_path.is_symlink():
            raise ValueError("schema 3 segment chain 缺少前一代 checkpoint")
        previous_metadata_path = previous_path / "checkpoint.json"
        previous_raw = _read_json(previous_metadata_path)
        if previous_raw.get("schema_version") != _SCHEMA30_VERSION:
            raise ValueError("schema 3 segment chain 不可跨越非 schema 3 generation")
        previous = _read_schema3_generation_header(
            previous_path,
            expected_binding=expected_binding,
            expected_run_units=expected_run_units,
        )
        previous_checkpoint_json_sha256_actual = sha256_file(previous_metadata_path)
        if previous_checkpoint_json_sha256_actual != previous_checkpoint_json_sha256:
            raise ValueError("schema 3 previous checkpoint.json SHA-256 不一致")
        chain.append((previous_path, *previous))
        current = (previous_path, *previous)
    chain.reverse()

    identities = [record["identity"] for record in compact["records"]]
    particle_order = list(metadata["particle_order"])
    observation_rows: list[list[Observation]] = [[] for _ in identities]
    event_rows: list[list[BoundaryEvent]] = [[] for _ in identities]
    chain_root_sequence = int(chain[0][1]["chain_root_sequence"])
    if chain_root_sequence != int(chain[0][1]["sequence"]):
        raise ValueError("schema 3 hash chain root sequence 不一致")
    previous_sequence = chain_root_sequence - 1
    for segment_path, segment_metadata, segment_compact, segment_payload in chain:
        del segment_path, segment_compact
        sequence = int(segment_metadata["sequence"])
        if sequence != previous_sequence + 1:
            raise ValueError("schema 3 segment sequence 跳號或重排")
        if int(segment_metadata["chain_root_sequence"]) != chain_root_sequence:
            raise ValueError("schema 3 chain_root_sequence 在 generation 間改變")
        previous_sequence = sequence
        if segment_payload["particle_order"] != particle_order:
            raise ValueError("schema 3 segment particle order 重排")
        records = segment_payload["records"]
        for index, record in enumerate(records):
            row = _require_exact_keys(
                record,
                (
                    "identity",
                    "observation_start_cursor",
                    "observation_end_cursor",
                    "event_start_cursor",
                    "event_end_cursor",
                    "observation_row_count",
                    "event_row_count",
                    "observations",
                    "events",
                ),
                label=f"schema 3 segment[{sequence}].records[{index}]",
            )
            identity = _validate_run_identity(
                row["identity"], label=f"segment[{sequence}].records[{index}].identity"
            )
            if identity != identities[index] or identity["particle_id"] != particle_order[index]:
                raise ValueError("schema 3 segment identity/order 不一致")
            observation_start = _nonnegative_int(
                row["observation_start_cursor"],
                label=f"segment[{sequence}].records[{index}].observation_start_cursor",
            )
            observation_end = _nonnegative_int(
                row["observation_end_cursor"],
                label=f"segment[{sequence}].records[{index}].observation_end_cursor",
            )
            event_start = _nonnegative_int(
                row["event_start_cursor"], label=f"segment[{sequence}].records[{index}].event_start_cursor"
            )
            event_end = _nonnegative_int(
                row["event_end_cursor"], label=f"segment[{sequence}].records[{index}].event_end_cursor"
            )
            observations = row["observations"]
            events = row["events"]
            if not isinstance(observations, list) or not isinstance(events, list):
                raise ValueError("schema 3 segment observations/events 必須是陣列")
            if (
                observation_end - observation_start != len(observations)
                or event_end - event_start != len(events)
                or row["observation_row_count"] != len(observations)
                or row["event_row_count"] != len(events)
            ):
                raise ValueError("schema 3 segment cursor/row count 不一致")
            if observation_start != len(observation_rows[index]) or event_start != len(event_rows[index]):
                raise ValueError("schema 3 segment cursor 不連續、重複或缺失")
            observation_rows[index].extend(
                _deserialize_observation(
                    value,
                    label=f"segment[{sequence}].records[{index}].observations[{row_index}]",
                    schema_version=_SCHEMA30_VERSION,
                )
                for row_index, value in enumerate(observations)
            )
            event_rows[index].extend(
                _deserialize_event(
                    value,
                    label=f"segment[{sequence}].records[{index}].events[{row_index}]",
                )
                for row_index, value in enumerate(events)
            )

    executions: list[ParticleExecutionState] = []
    rng_states: list[dict[str, Any]] = []
    hints: list[int] = []
    total_observations = 0
    total_events = 0
    for index, record in enumerate(compact["records"]):
        state = _deserialize_particle_state(record["state"], label=f"compact.records[{index}].state")
        if _state_identity(state) != (
            identities[index]["particle_id"],
            identities[index]["scenario_id"],
            identities[index]["member_id"],
            identities[index]["study_site_id"],
            identities[index]["analysis_region_id"],
            identities[index]["receptor_id"],
        ):
            raise ValueError("schema 3 compact state identity 不一致")
        observation_cursor = _nonnegative_int(
            record["observation_cursor"], label=f"compact.records[{index}].observation_cursor"
        )
        event_cursor = _nonnegative_int(
            record["event_cursor"], label=f"compact.records[{index}].event_cursor"
        )
        if observation_cursor != len(observation_rows[index]) or event_cursor != len(event_rows[index]):
            raise ValueError("schema 3 compact cursor 與 segment history 不一致")
        pending_observation = _deserialize_observation(
            record["pending_observation"],
            label=f"compact.records[{index}].pending_observation",
            schema_version=_SCHEMA30_VERSION,
        )
        if pending_observation.particle_id != state.particle_id:
            raise ValueError("schema 3 pending observation identity 與 state 不一致")
        if (
            state.age_seconds + 1.0e-9 < pending_observation.age_seconds
            or state.time_utc_ns > pending_observation.time_utc_ns
        ):
            raise ValueError("schema 3 pending observation 晚於 current state")
        complete_observations = [*observation_rows[index], pending_observation]
        execution = ParticleExecutionState(
            state=state,
            observations=complete_observations,
            events=event_rows[index],
            step_count=_nonnegative_int(record["step_count"], label=f"compact.records[{index}].step_count"),
            minimum_clamp_count=_nonnegative_int(
                record["minimum_clamp_count"], label=f"compact.records[{index}].minimum_clamp_count"
            ),
            next_output_age_seconds=_finite_float(
                record["next_output_age_seconds"],
                label=f"compact.records[{index}].next_output_age_seconds",
            ),
        )
        if execution.next_output_age_seconds <= 0:
            raise ValueError("schema 3 next_output_age_seconds 必須為正")
        if any(item.particle_id != state.particle_id for item in execution.observations):
            raise ValueError("schema 3 observation identity 與 state 不一致")
        if any(
            item.particle_id != state.particle_id
            or item.scenario_id != state.scenario_id
            or item.member_id != state.member_id
            or item.study_site_id != state.study_site_id
            or item.analysis_region_id != state.analysis_region_id
            or item.receptor_id != state.receptor_id
            for item in execution.events
        ):
            raise ValueError("schema 3 event identity 與 state 不一致")
        _validate_observation_sequence(execution.observations, state, label=f"schema 3 observations[{index}]")
        # 公開 loader 也直接回傳可續寫的 tracker；不能只依賴 ProductionBatch 私下補包裝，
        # 否則呼叫端以 load_execution_checkpoint→advance→write continuation 時，stable
        # prefix 會退化成普通 list，無法保證歷史列未被替換。
        _track_execution_history(
            execution,
            observation_prefix_length=observation_cursor,
            event_prefix_length=event_cursor,
        )
        executions.append(execution)
        rng_states.append(
            _validate_rng_state(record["rng_state"], label=f"compact.records[{index}].rng_state")
        )
        hints.append(
            _normalize_triangle_hint(
                record["triangle_hint"], label=f"compact.records[{index}].triangle_hint"
            )
        )
        total_observations += observation_cursor + 1
        total_events += event_cursor
    if total_observations != metadata["observation_count"] or total_events != metadata["event_count"]:
        raise ValueError("schema 3 observation/event count 不符")
    return ExecutionCheckpoint(
        binding=expected_binding,
        sequence=target_sequence,
        run_unit_identities=deepcopy(identities),
        executions=executions,
        rng_states=rng_states,
        triangle_hints=hints,
        schema_version=_SCHEMA30_VERSION,
        observation_cursors=[len(item.observations) for item in executions],
        event_cursors=[len(item.events) for item in executions],
    )


def _prepare_schema3_writer_inputs(
    *,
    run_units: Sequence[Any],
    executions: Sequence[ParticleExecutionState],
    rngs: Sequence[np.random.Generator],
    triangle_hints: Sequence[int | None],
) -> tuple[
    list[dict[str, Any]],
    list[ParticleExecutionState],
    list[dict[str, Any]],
    list[int],
]:
    """驗證 v3 writer 的固定 identity 並只保留目前 execution 的輕量參照。

    舊的 ``build_execution_checkpoint`` 為了提供可獨立保存的記憶體 snapshot，會 deep-copy
    每粒子的完整 observation／event history。正式 checkpoint 若在數萬步中每個 cadence
    都重複複製這些歷史，雖然檔案 segment 已改成線性，writer 仍會在記憶體與 CPU 產生
    隱性的 O(n²) 成本。因此磁碟 writer 不把完整 history 複製到另一份 snapshot；它只在
    下面序列化本代新增 rows、最後一筆 pending observation、目前 state、RNG 與 triangle
    hint。呼叫端在完整 sweep 邊界同步呼叫 writer，故這些參照在一次寫入期間不會被 engine
    併發修改；公開 ``snapshot`` API 仍維持獨立 deep-copy 語意。

    回傳的 execution list 只含原物件參照，供 writer 依上一代 cursor 切片；writer 會在
    切片後以固定 serializer 驗證要發布的 rows，並由 loader 在 resume 時完整重建與再驗證。
    """

    lengths = (len(run_units), len(executions), len(rngs), len(triangle_hints))
    if not lengths[0] or len(set(lengths)) != 1:
        raise ValueError("execution checkpoint 不允許空資料且所有欄位長度必須一致")
    # writer 必須先依 loader 同一 strict schema 驗證 RunUnit 原始 identity，才能保證
    # metadata／compact 一旦發布就可立即被 resume loader 還原；validator 回傳的 canonical
    # dict 會在本函式後續所有 state、history 與 particle order 檢查中共用。
    identities = [
        _strict_writer_run_unit_identity(unit, label=f"run_units[{index}].identity")
        for index, unit in enumerate(run_units)
    ]
    identity_tuples = [_run_identity_tuple(identity) for identity in identities]
    if len(set(identity_tuples)) != len(identity_tuples):
        raise ValueError("execution checkpoint RunUnit identity 必須唯一")
    particle_ids = [identity["particle_id"] for identity in identities]
    if len(set(particle_ids)) != len(particle_ids):
        raise ValueError("execution checkpoint particle_id 必須唯一")

    normalized_hints: list[int] = []
    execution_rows: list[ParticleExecutionState] = []
    rng_states: list[dict[str, Any]] = []
    for index, (identity, execution, rng, triangle_hint) in enumerate(
        zip(identities, executions, rngs, triangle_hints, strict=True)
    ):
        if not isinstance(execution, ParticleExecutionState):
            raise TypeError(f"execution[{index}] 必須是 ParticleExecutionState")
        # 初次 root writer 可能接到 engine 建立的普通 list；先把它轉成輕量追蹤器，讓
        # 後續 publish 後能辨識 stable prefix 是否被替換。這裡只保存長度與 dirty 旗標，
        # 不複製歷史 row，也不改變 observations/events 的資料內容或順序。
        for attribute in ("observations", "events"):
            current_history = getattr(execution, attribute)
            if not isinstance(current_history, _CheckpointHistoryList):
                setattr(execution, attribute, _CheckpointHistoryList(current_history))
        if _state_identity(execution.state) != (
            identity["particle_id"],
            identity["scenario_id"],
            identity["member_id"],
            identity["study_site_id"],
            identity["analysis_region_id"],
            identity["receptor_id"],
        ):
            raise ValueError(f"execution[{index}] 與 RunUnit identity 不一致")
        if not execution.observations:
            raise ValueError(f"execution[{index}] observations 不可為空")
        _nonnegative_int(execution.step_count, label=f"execution[{index}].step_count")
        _nonnegative_int(
            execution.minimum_clamp_count,
            label=f"execution[{index}].minimum_clamp_count",
        )
        next_age = _finite_float(
            execution.next_output_age_seconds,
            label=f"execution[{index}].next_output_age_seconds",
        )
        if next_age <= 0:
            raise ValueError(f"execution[{index}].next_output_age_seconds 必須為正")
        normalized_hint = _normalize_triangle_hint(
            triangle_hint,
            label=f"triangle_hints[{index}]",
        )
        # 只檢查最後一筆與 current state 的不可逆方向關係；歷史前綴已由上一代
        # published chain 驗證，新增區間會在 writer 的切片後再檢查完整順序。
        _validate_observation_sequence(
            execution.observations[-1:],
            execution.state,
            label=f"execution[{index}].observations",
        )
        normalized_hints.append(normalized_hint)
        execution_rows.append(execution)
        rng_states.append(_copy_rng_state(rng))
    return identities, execution_rows, rng_states, normalized_hints


def write_execution_checkpoint(
    destination: str | Path,
    *,
    binding: CheckpointBinding,
    run_units: Sequence[Any],
    executions: Sequence[ParticleExecutionState],
    rngs: Sequence[np.random.Generator],
    triangle_hints: Sequence[int | None],
    sequence: int,
    previous_checkpoint: str | Path | None = None,
) -> Path:
    """以 schema 3.0.0 原子追加 checkpoint generation。

    ``previous_checkpoint`` 指向同一 shard 的上一個已發布 generation。新 generation 的
    compact file 只保存最新 ParticleState／step counter／輸出游標／RNG／triangle hint；
    history segment 只保存各粒子自上一代 cursor 之後新增的 observation 與 event。因而
    每代寫入量與新增 row 數近似成正比，不會隨累積歷史重寫完整 execution JSON。第一代
    沒有前代時 cursor 從零開始；若 caller 由舊 schema 2.x 起始，會把該 checkpoint 當作
    一次性 chain root 讀入並追加完整歷史，舊目錄本身不會被修改。

    寫入順序是 partial directory → checksum manifest → atomic rename；目標已存在、前代
    binding／RunUnit 順序不符、history cursor 回退、相鄰 observation engine key 重複，或
    輸出資料在完整 sweep/macro boundary 外呼叫，均由 caller 或本函式拒絕。RunUnit identity
    會在建立 partial 前沿用 loader 的 strict 欄位驗證，並要求所有粒子的 ``particle_id``
    全域唯一；state、觀測與事件也會預驗同一套 semantic decoder，確保成功發布的 generation
    可立即還原。此 API 不刪除任何 generation，避免 retention 破壞 checkpoint.json hash chain。
    """

    if not isinstance(binding, CheckpointBinding):
        raise TypeError("binding 必須是 CheckpointBinding")
    # 在初始化 root tracker 前記住原始型別；v3 continuation 若整條替換成普通 list，
    # 後續必須拒絕，避免新 tracker 把未驗證的 stable prefix 偽裝成合法延續。
    had_untracked_history = any(
        not isinstance(getattr(execution, attribute), _CheckpointHistoryList)
        for execution in executions
        for attribute in ("observations", "events")
    )
    identities, execution_rows, rng_states, normalized_hints = _prepare_schema3_writer_inputs(
        run_units=run_units,
        executions=executions,
        rngs=rngs,
        triangle_hints=triangle_hints,
    )
    if type(sequence) is not int or sequence < 1:
        raise ValueError("schema 3 sequence 必須是從 1 開始的整數")
    target = Path(destination)
    expected_target_name = f"checkpoint-{sequence:08d}"
    if target.name != expected_target_name:
        raise ValueError("schema 3 target 名稱必須是 checkpoint-{sequence:08d}")
    if target.exists():
        raise FileExistsError(f"不可覆寫 checkpoint：{target}")
    previous_schema: str | None = None
    previous_checkpoint_json_sha256: str | None = None
    legacy_source: dict[str, Any] | None = None
    if previous_checkpoint is not None:
        previous_root = Path(previous_checkpoint)
        expected_previous_root = target.parent / f"checkpoint-{sequence - 1:08d}"
        if previous_root.parent != target.parent or previous_root.name != expected_previous_root.name:
            raise ValueError(
                "schema 3 previous_checkpoint 必須位於 target 同一 parent，且名稱必須是上一代 checkpoint"
            )
        previous_metadata = _read_json(previous_root / "checkpoint.json")
        previous_schema = previous_metadata.get("schema_version")
        if previous_schema == _SCHEMA30_VERSION:
            if had_untracked_history:
                raise ValueError("schema 3 continuation 的 observations/events 必須保留追蹤器")
            previous_metadata, previous_compact, _ = _read_schema3_generation_header(
                previous_root,
                expected_binding=binding,
                expected_run_units=run_units,
            )
            previous_sequence = _nonnegative_int(
                previous_metadata["sequence"], label="previous sequence"
            )
            if sequence != previous_sequence + 1:
                raise ValueError("schema 3 sequence 必須緊接上一代 generation")
            previous_checkpoint_json_sha256 = sha256_file(previous_root / "checkpoint.json")
            previous_records = previous_compact["records"]
            # 既有 v3 continuation 必須沿用同一條 chain 的 root sequence。這個欄位
            # 代表 v3 歷史鏈真正開始的 generation；若前一代是從舊 schema 遷移而來，
            # root 可能大於 1，不能因為目前 generation 已經往後推進就重新算成 1 或
            # current sequence，否則 loader 會把合法的遷移鏈誤判為 root 改變。
            chain_root_sequence = _positive_int(
                previous_metadata["chain_root_sequence"],
                label="previous chain_root_sequence",
            )
        elif previous_schema in {
            _SCHEMA20_VERSION,
            _SCHEMA21_VERSION,
            _SCHEMA22_VERSION,
        }:
            # 舊 schema 沒有 segment cursor；完整舊 execution 只在這個 migration root 讀一次。
            previous_loaded = load_execution_checkpoint(
                previous_root,
                expected_binding=binding,
                expected_run_units=run_units,
            )
            for index, (current, legacy) in enumerate(
                zip(execution_rows, previous_loaded.executions, strict=True)
            ):
                _validate_legacy_execution_prefix(
                    current,
                    legacy,
                    label=f"schema 3 migration execution[{index}]",
                )
                # schema 2 沒有 compact record，但舊 execution 已保存 terminal 粒子的
                # 完整 current／cursor／pending／RNG／hint。遷移時沿用同一個 O(P) 凍結
                # 閘門，避免只驗 history 前綴而讓已終止粒子在 v3 root 靜默分歧。
                legacy_record = _legacy_terminal_compact_record(
                    legacy,
                    rng_state=previous_loaded.rng_states[index],
                    triangle_hint=previous_loaded.triangle_hints[index],
                    label=f"schema 3 migration execution[{index}]",
                )
                if legacy_record is not None:
                    _validate_terminal_continuation(
                        current,
                        legacy_record,
                        rng_state=rng_states[index],
                        triangle_hint=normalized_hints[index],
                        label=f"schema 3 migration execution[{index}]",
                    )
            previous_sequence = previous_loaded.sequence
            if sequence != previous_sequence + 1:
                raise ValueError("schema 3 migration sequence 必須緊接舊 checkpoint")
            previous_checkpoint_json_sha256 = None
            legacy_source = {
                "schema_version": previous_schema,
                "sequence": previous_sequence,
                "relative_directory": previous_root.name,
                "checkpoint_json_sha256": sha256_file(previous_root / "checkpoint.json"),
            }
            # 舊 schema 沒有 v3 segment，因此本次發布的 generation 是新 hash chain
            # 的 root。root sequence 必須保留本代實際 sequence，讓遷移後的下一代／
            # 再下一代都能繼承同一值並通過完整鏈驗證。
            chain_root_sequence = sequence
            previous_records = [
                {
                    "identity": identity,
                    # 舊 schema 的完整 execution 已包含所有 rows；轉成 v3 時只需把
                    # 穩定列寫入新 chain，最後一筆保留在 compact pending observation。
                    "observation_cursor": 0,
                    # 舊目錄不是 v3 hash chain 的前一代，因此舊事件沒有在任何已發布
                    # segment 中出現；遷移 root 必須從 cursor 0 重新寫入全部事件，否則
                    # loader 會得到少於舊 execution 的事件歷史，且 resume 後事件順序會
                    # 靜默缺列。這不會改寫舊檔，只決定一次性 v3 migration segment。
                    "event_cursor": 0,
                }
                for identity, execution in zip(
                    previous_loaded.run_unit_identities,
                    previous_loaded.executions,
                    strict=True,
                )
            ]
        else:
            raise ValueError("previous_checkpoint schema 不支援")
        previous_order = [record["identity"]["particle_id"] for record in previous_records]
        if previous_order != [identity["particle_id"] for identity in identities]:
            raise ValueError("previous checkpoint particle order 不一致")
    else:
        previous_sequence = 0
        previous_checkpoint_json_sha256 = None
        # 沒有前代時建立全新的 v3 chain，固定由 sequence 1 作為 root。正式 controller
        # 會依序發布 generation；此欄位不能因為後續寫入而重新計算。
        chain_root_sequence = 1
        previous_records = [
            {"identity": identity, "observation_cursor": 0, "event_cursor": 0}
            for identity in identities
        ]
    if previous_schema != _SCHEMA30_VERSION:
        # root／legacy migration 沒有可沿用的 v3 cursor；若 execution 來自另一份 v3
        # snapshot，其 tracker prefix 只屬於來源 chain，不能拿來限制新 chain 的完整歷史。
        # legacy prefix 已在上方先完成逐欄核對，這裡只重建 owner 與從零開始的追蹤狀態。
        for execution in execution_rows:
            _reset_checkpoint_history_tracking(execution)
    if previous_sequence != sequence - 1:
        raise ValueError("schema 3 generation sequence 不連續")

    segment_records: list[dict[str, Any]] = []
    compact_records: list[dict[str, Any]] = []
    observation_count = 0
    event_count = 0
    for index, (identity, execution, previous_record) in enumerate(
        zip(identities, execution_rows, previous_records, strict=True)
    ):
        if identity != previous_record["identity"]:
            raise ValueError(f"schema 3 previous/current identity 不一致：index={index}")
        observation_start = _nonnegative_int(
            previous_record["observation_cursor"], label=f"previous observation cursor[{index}]"
        )
        event_start = _nonnegative_int(
            previous_record["event_cursor"], label=f"previous event cursor[{index}]"
        )
        if observation_start > len(execution.observations) or event_start > len(execution.events):
            raise ValueError("schema 3 history cursor 回退")
        if previous_schema == _SCHEMA30_VERSION:
            _validate_terminal_continuation(
                execution,
                previous_record,
                rng_state=rng_states[index],
                triangle_hint=normalized_hints[index],
                label=f"schema 3 execution[{index}]",
            )
        if execution.observations.checkpoint_history_dirty or execution.events.checkpoint_history_dirty:
            raise ValueError("schema 3 stable observation/event prefix 已被修改")
        if (
            execution.observations.checkpoint_immutable_prefix_length != observation_start
            or execution.events.checkpoint_immutable_prefix_length != event_start
        ):
            raise ValueError("schema 3 stable observation/event prefix tracking 不一致")
        if not execution.observations:
            raise ValueError("schema 3 execution observations 不可為空")
        # engine 允許在下一個取樣前以同一時間／位置更新最後一筆 observation 的
        # context/status；因此最後一筆先放在 compact，等下一筆出現後才成為 immutable
        # segment row，避免 checkpoint 重新寫入同一歷史列。
        stable_observation_end = len(execution.observations) - 1
        if observation_start > stable_observation_end:
            raise ValueError("schema 3 history cursor 超過穩定 observation 數")
        if previous_schema == _SCHEMA30_VERSION:
            previous_pending = _deserialize_observation(
                previous_record["pending_observation"],
                label=f"previous compact.records[{index}].pending_observation",
                schema_version=_SCHEMA30_VERSION,
            )
            _require_observation_core_match(
                previous_pending,
                execution.observations[observation_start],
                label=f"schema 3 execution[{index}] pending boundary",
            )
        # engine 對同一 particle／UTC／age 只保留一筆最後 observation；這項增量 gate
        # 封住 append、extend、空 slice insertion、insert、+= 與 *= 等一般 list API，避免
        # 它們在 stable prefix gate 未觸發時製造相鄰假列。包含最後 pending row，才能抓到
        # stable segment 與 pending 交界的重複；start 以前的歷史由上一代 loader 驗證。
        _validate_adjacent_observation_engine_keys(
            execution.observations,
            start=observation_start,
            label=f"execution[{index}].observations",
        )
        observations = execution.observations[observation_start:stable_observation_end]
        events = execution.events[event_start:]
        _validate_observation_sequence(
            execution.observations[observation_start:],
            execution.state,
            label=f"execution[{index}].observations",
        )
        serialized_state = _serialize_and_validate_writer_state(
            execution.state,
            expected_identity=identity,
            label=f"schema 3 compact.records[{index}].state",
        )
        observation_tail = execution.observations[observation_start:]
        serialized_observation_tail = [
            _serialize_and_validate_writer_observation(
                observation,
                expected_particle_id=identity["particle_id"],
                label=f"schema 3 execution[{index}].observations[{observation_start + row_index}]",
            )
            for row_index, observation in enumerate(observation_tail)
        ]
        serialized_observations = serialized_observation_tail[:-1]
        serialized_pending_observation = serialized_observation_tail[-1]
        serialized_events = [
            _serialize_and_validate_writer_event(
                event,
                expected_identity=identity,
                label=f"schema 3 execution[{index}].events[{event_start + event_index}]",
            )
            for event_index, event in enumerate(events)
        ]
        segment_records.append(
            {
                "identity": deepcopy(identity),
                "observation_start_cursor": observation_start,
                "observation_end_cursor": stable_observation_end,
                "event_start_cursor": event_start,
                "event_end_cursor": len(execution.events),
                "observation_row_count": len(observations),
                "event_row_count": len(events),
                "observations": serialized_observations,
                "events": serialized_events,
            }
        )
        compact_records.append(
            {
                "identity": deepcopy(identity),
                "state": serialized_state,
                "step_count": execution.step_count,
                "minimum_clamp_count": execution.minimum_clamp_count,
                "next_output_age_seconds": execution.next_output_age_seconds,
                "observation_cursor": len(execution.observations) - 1,
                "event_cursor": len(execution.events),
                "pending_observation": serialized_pending_observation,
                "rng_state": rng_states[index],
                "triangle_hint": normalized_hints[index],
            }
        )
        observation_count += len(execution.observations)
        event_count += len(execution.events)
    segment_payload = {
        "schema_version": _SCHEMA30_VERSION,
        "sequence": sequence,
        "binding": _binding_payload(binding),
        "particle_order": [identity["particle_id"] for identity in identities],
        "previous_checkpoint_json_sha256": previous_checkpoint_json_sha256,
        "records": segment_records,
    }
    compact_payload = {
        "schema_version": _SCHEMA30_VERSION,
        "sequence": sequence,
        "binding": _binding_payload(binding),
        "particle_order": [identity["particle_id"] for identity in identities],
        "records": compact_records,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.parent / f".{target.name}.partial-{uuid4().hex}"
    try:
        partial.mkdir()
        compact_path = partial / "compact_state.json"
        segment_path = partial / "history_segment.json"
        _write_json(compact_path, compact_payload)
        _write_json(segment_path, segment_payload)
        files = {
            compact_path.name: {
                "size_bytes": int(compact_path.stat().st_size),
                "sha256": sha256_file(compact_path),
            },
            segment_path.name: {
                "size_bytes": int(segment_path.stat().st_size),
                "sha256": sha256_file(segment_path),
            },
        }
        metadata = {
            "schema_version": _SCHEMA30_VERSION,
            "sequence": sequence,
            "particle_count": len(execution_rows),
            "observation_count": observation_count,
            "event_count": event_count,
            "binding": _binding_payload(binding),
            "particle_order": [identity["particle_id"] for identity in identities],
            "files": files,
            # 一般 v3 chain 從 sequence 1 開始；由舊 schema 2.x resume 的第一個 v3
            # generation 以自身作 chain root，明示遷移邊界而不改寫舊目錄。
            "chain_root_sequence": chain_root_sequence,
            "legacy_source": legacy_source,
            "segment": {
                "sequence": sequence,
                "previous_checkpoint_json_sha256": previous_checkpoint_json_sha256,
                "record_count": len(segment_records),
            },
        }
        _write_json(partial / "checkpoint.json", metadata, sort_keys=True)
        os.replace(partial, target)
        for execution in execution_rows:
            execution.observations.mark_checkpoint_prefix(len(execution.observations) - 1)
            execution.events.mark_checkpoint_prefix(len(execution.events))
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return target


def _verify_schema2_files(root: Path, metadata: dict[str, Any]) -> None:
    """驗證 execution schema 2.x 的固定目錄拓撲、實體檔案與 data file checksum。

    checkpoint 只允許一個 metadata 檔與兩個 payload 檔。未知子目錄、額外隱藏檔、把
    payload 換成 symlink，或把預期檔名替換成目錄，都會在讀取 JSON payload 前拒絕；這
    可避免 loader 跟隨未受 manifest 保護的外部路徑，也避免 partial／暫存內容被誤當正式
    checkpoint。manifest 只對兩個 payload 保存大小與 SHA-256，metadata 本身仍由 JSON
    schema 欄位檢查保護。
    """

    if root.is_symlink():
        raise ValueError(f"schema 2 checkpoint 根目錄不允許 symlink：{root}")
    files = metadata.get("files")
    if not isinstance(files, dict) or set(files) != set(_SCHEMA2_DATA_FILES):
        raise ValueError("schema 2 checkpoint files manifest 不完整或含未知檔案")
    expected_names = set(_SCHEMA2_DATA_FILES) | {"checkpoint.json"}
    entries = tuple(root.iterdir())
    actual_names = {entry.name for entry in entries}
    if actual_names != expected_names:
        raise ValueError("schema 2 checkpoint 目錄含有遺失或未知檔案")
    for entry in entries:
        if entry.is_symlink():
            raise ValueError(f"schema 2 checkpoint 不允許 symlink：{entry.name}")
        if not entry.is_file():
            raise ValueError(f"schema 2 checkpoint 只允許固定檔案：{entry.name}")
    for filename in _SCHEMA2_DATA_FILES:
        path = root / filename
        contract = files[filename]
        if not isinstance(contract, dict) or set(contract) != {"size_bytes", "sha256"}:
            raise ValueError(f"{filename} checksum contract 不完整")
        expected_size = contract["size_bytes"]
        expected_sha = contract["sha256"]
        if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size < 0:
            raise ValueError(f"{filename} size_bytes 無效")
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise ValueError(f"{filename} sha256 無效")
        if path.stat().st_size != expected_size or sha256_file(path) != expected_sha:
            raise ValueError(f"{filename} checksum 或 size 不符")


def load_execution_checkpoint(
    path: str | Path,
    *,
    expected_binding: CheckpointBinding,
    expected_run_units: Sequence[Any] | None = None,
) -> ExecutionCheckpoint:
    """嚴格讀取 schema 3.0 與舊 schema 2.0／2.1／2.2 checkpoint。

    schema 3 會先沿 checkpoint.json hash chain 還原完整歷史；2.0 只接受舊七
    欄並將環境與速度 context 設為 ``NOT_SAMPLED``／``None``；2.1 只接受舊環境欄位；
    2.2 才接受完整的速度 11 欄，三個版本都不容許 unknown／missing。之後仍會拒絕
    binding、checksum、檔案／觀測／事件／粒子數、duplicate identity、未知 status、未知
    JSON 欄位及 RNG state 不符；若提供 ``expected_run_units``，其完整 identity 與 checkpoint
    particle order 必須逐項相同。forcing 與 geometry 不從檔案讀取；2.0／2.1 只具工程重啟
    相容性，不可作為 F03／F09 正式垂向證據。
    """

    if not isinstance(expected_binding, CheckpointBinding):
        raise TypeError("expected_binding 必須是 CheckpointBinding")
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint 目錄不存在：{root}")
    # 先讀版本欄位再分派，避免把 schema 3 的 compact／segment 拓撲誤判為 schema 2
    # 的 execution_state／rng_states。舊 schema 只走下方既有相容 parser，完全不改寫原檔。
    initial_metadata = _read_json(root / "checkpoint.json")
    if initial_metadata.get("schema_version") == _SCHEMA30_VERSION:
        return _load_schema3_execution_checkpoint(
            root,
            expected_binding=expected_binding,
            expected_run_units=expected_run_units,
        )
    metadata = _read_json(root / "checkpoint.json")
    expected_metadata_keys = (
        "schema_version",
        "sequence",
        "particle_count",
        "observation_count",
        "event_count",
        "binding",
        "particle_order",
        "files",
    )
    metadata = _require_exact_keys(metadata, expected_metadata_keys, label="checkpoint metadata")
    schema_version = metadata["schema_version"]
    if type(schema_version) is not str or schema_version not in _SUPPORTED_EXECUTION_SCHEMA_VERSIONS:
        raise ValueError(f"checkpoint schema 不支援：{schema_version!r}")
    sequence = _nonnegative_int(metadata["sequence"], label="sequence")
    particle_count = _nonnegative_int(metadata["particle_count"], label="particle_count")
    observation_count = _nonnegative_int(metadata["observation_count"], label="observation_count")
    event_count = _nonnegative_int(metadata["event_count"], label="event_count")
    if particle_count < 1:
        raise ValueError("schema 2 checkpoint 不允許空 shard")
    binding_value = metadata["binding"]
    if not isinstance(binding_value, dict):
        raise ValueError("binding 必須是 object")
    binding_keys = set(binding_value)
    if binding_keys == set(_BINDING_FIELDS):
        # 舊 checkpoint 沒有 stream 欄位；dataclass default None 保留既有 binding 語意。
        binding_row = _require_exact_keys(binding_value, _BINDING_FIELDS, label="binding")
    elif binding_keys == set(_BINDING_FIELDS_WITH_RANDOM):
        binding_row = _require_exact_keys(
            binding_value,
            _BINDING_FIELDS_WITH_RANDOM,
            label="binding",
        )
        if type(binding_row["random_stream_id"]) is not str or not binding_row[
            "random_stream_id"
        ].strip():
            raise ValueError("binding.random_stream_id 必須是非空白字串")
    else:
        raise ValueError("binding 欄位集合不符")
    try:
        actual_binding = CheckpointBinding(**binding_row)
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint binding 欄位無效") from error
    if actual_binding != expected_binding:
        raise ValueError(f"checkpoint binding 不相容：actual={actual_binding}, expected={expected_binding}")
    _verify_schema2_files(root, metadata)

    execution_payload = _read_json(root / "execution_state.json")
    rng_payload = _read_json(root / "rng_states.json")
    execution_payload = _require_exact_keys(execution_payload, ("records",), label="execution_state")
    rng_payload = _require_exact_keys(rng_payload, ("records",), label="rng_states")
    execution_records = execution_payload["records"]
    rng_records = rng_payload["records"]
    if not isinstance(execution_records, list) or not isinstance(rng_records, list):
        raise ValueError("execution/rng records 必須是陣列")
    if len(execution_records) != particle_count or len(rng_records) != particle_count:
        raise ValueError("checkpoint particle count 不符")
    particle_order = metadata["particle_order"]
    if (
        not isinstance(particle_order, list)
        or len(particle_order) != particle_count
        or any(not isinstance(item, str) for item in particle_order)
        or len(set(particle_order)) != particle_count
    ):
        raise ValueError("checkpoint particle_order 無效或含 duplicate")

    identities: list[dict[str, Any]] = []
    executions: list[ParticleExecutionState] = []
    hints: list[int] = []
    total_observations = 0
    total_events = 0
    for index, record in enumerate(execution_records):
        row = _require_exact_keys(record, ("identity", "execution"), label=f"execution_records[{index}]")
        identity = _validate_run_identity(row["identity"], label=f"execution_records[{index}].identity")
        if identity["particle_id"] != particle_order[index]:
            raise ValueError("checkpoint particle_order 與 execution record 順序不一致")
        execution, hint = _deserialize_execution(
            row["execution"],
            label=f"execution_records[{index}].execution",
            schema_version=schema_version,
        )
        if _state_identity(execution.state) != (
            identity["particle_id"],
            identity["scenario_id"],
            identity["member_id"],
            identity["study_site_id"],
            identity["analysis_region_id"],
            identity["receptor_id"],
        ):
            raise ValueError(f"execution record 與 state identity 不一致：index={index}")
        identities.append(identity)
        executions.append(execution)
        hints.append(hint)
        total_observations += len(execution.observations)
        total_events += len(execution.events)
    identity_tuples = [_run_identity_tuple(identity) for identity in identities]
    if len(set(identity_tuples)) != len(identity_tuples):
        raise ValueError("checkpoint 含有 duplicate RunUnit identity")
    if total_observations != observation_count or total_events != event_count:
        raise ValueError("checkpoint observation/event count 不符")

    rng_states: list[dict[str, Any]] = []
    for index, record in enumerate(rng_records):
        row = _require_exact_keys(record, ("particle_id", "rng_state"), label=f"rng_records[{index}]")
        if row["particle_id"] != identities[index]["particle_id"]:
            raise ValueError("checkpoint RNG particle order 不一致")
        rng_states.append(_validate_rng_state(row["rng_state"], label=f"rng_records[{index}].rng_state"))

    if expected_run_units is not None:
        expected_identities = [_run_unit_identity(unit) for unit in expected_run_units]
        if expected_identities != identities:
            raise ValueError("checkpoint RunUnit identity/order 與目前 shard 不一致")
    return ExecutionCheckpoint(
        binding=actual_binding,
        sequence=sequence,
        run_unit_identities=identities,
        executions=executions,
        rng_states=rng_states,
        triangle_hints=hints,
        schema_version=schema_version,
        observation_cursors=[len(execution.observations) for execution in executions],
        event_cursors=[len(execution.events) for execution in executions],
    )


def inspect_execution_checkpoint(
    path: str | Path,
    *,
    expected_binding: CheckpointBinding,
    expected_run_units: Sequence[Any] | None = None,
) -> tuple[int, int, int]:
    """只讀取 generation counter，避免掃描 v3 時建立完整歷史 dataclass。

    回傳 ``(sequence, sweeps_lower_bound, particle_steps)``。schema 3 仍會讀取並 JSON
    parse metadata、compact 與當代 history segment，並驗證 payload checksum、欄位拓撲、
    cursor 與 row count；它只不建立每條 observation／event 的 ``Observation``／
    ``BoundaryEvent`` 物件。最高 generation 的完整 chain validation 或真正 resume 才會
    反序列化並逐欄還原全部歷史。schema 2.x 沒有 compact state，因此沿用完整 loader，
    保留舊檔的嚴格驗證語意。此函式不能取代 resume validation；它的用途是避免掃描數百個
    generation 時對同一段歷史反覆建立 dataclass，不能保證冷 NFS cache 下完全不讀歷史。
    """

    root = Path(path)
    initial_metadata = _read_json(root / "checkpoint.json")
    if initial_metadata.get("schema_version") == _SCHEMA30_VERSION:
        metadata, compact, _ = _read_schema3_generation_header(
            root,
            expected_binding=expected_binding,
            expected_run_units=expected_run_units,
        )
        step_counts = [
            _nonnegative_int(
                record["step_count"], label=f"compact.records[{index}].step_count"
            )
            for index, record in enumerate(compact["records"])
        ]
        return int(metadata["sequence"]), max(step_counts, default=0), sum(step_counts)

    loaded = load_execution_checkpoint(
        root,
        expected_binding=expected_binding,
        expected_run_units=expected_run_units,
    )
    step_counts = [int(execution.step_count) for execution in loaded.executions]
    return loaded.sequence, max(step_counts, default=0), sum(step_counts)


def write_checkpoint(
    destination: str | Path,
    *,
    binding: CheckpointBinding,
    states: Sequence[ParticleState],
    sequence: int,
) -> Path:
    """以 schema 1 格式保存粒子狀態；此 API 保留給既有呼叫端。"""

    if sequence < 0 or not states:
        raise ValueError("checkpoint sequence 必須非負且 states 不可空")
    if len({state.particle_id for state in states}) != len(states):
        raise ValueError("checkpoint particle_id 必須唯一")
    target = Path(destination)
    if target.exists():
        raise FileExistsError(f"不可覆寫 checkpoint：{target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.parent / f".{target.name}.partial-{uuid4().hex}"
    try:
        partial.mkdir()
        rows = []
        for state in states:
            row = asdict(state)
            row["status"] = state.status.value
            rows.append(row)
        state_path = partial / "particle_states.parquet"
        pq.write_table(pa.Table.from_pylist(rows), state_path)
        metadata = {
            "schema_version": "1.0.0",
            "sequence": sequence,
            "particle_count": len(states),
            "binding": _binding_payload(binding),
            "files": {
                state_path.name: {
                    "size_bytes": state_path.stat().st_size,
                    "sha256": sha256_file(state_path),
                }
            },
        }
        with (partial / "checkpoint.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(partial, target)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return target


def load_checkpoint(
    path: str | Path,
    *,
    expected_binding: CheckpointBinding,
) -> tuple[list[ParticleState], int]:
    """以 schema 1 API 讀回粒子狀態，並保留原有 binding/checksum/count 檢查。"""

    root = Path(path)
    metadata = json.loads((root / "checkpoint.json").read_text(encoding="utf-8"))
    actual_binding = CheckpointBinding(**metadata["binding"])
    if actual_binding != expected_binding:
        raise ValueError(f"checkpoint binding 不相容：actual={actual_binding}, expected={expected_binding}")
    state_path = root / "particle_states.parquet"
    contract = metadata["files"][state_path.name]
    if state_path.stat().st_size != contract["size_bytes"] or sha256_file(state_path) != contract["sha256"]:
        raise ValueError("checkpoint state checksum 或 size 不符")
    rows = pq.read_table(state_path).to_pylist()
    if len(rows) != metadata["particle_count"]:
        raise ValueError("checkpoint particle row count 不符")
    states = [ParticleState(**{**row, "status": ParticleStatus(row["status"])}) for row in rows]
    if len({state.particle_id for state in states}) != len(states):
        raise ValueError("checkpoint 含重複 particle_id")
    return states, int(metadata["sequence"])


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointBinding",
    "ExecutionCheckpoint",
    "build_execution_checkpoint",
    "inspect_execution_checkpoint",
    "load_checkpoint",
    "load_execution_checkpoint",
    "write_checkpoint",
    "write_execution_checkpoint",
]
