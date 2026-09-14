"""安全保存與恢復 reference／production 粒子中途狀態。

schema 1 只保存 ``ParticleState``，本模組保留其既有 ``write_checkpoint``／
``load_checkpoint`` API。execution checkpoint 的 writer 固定發布 schema ``2.2.0``，
保存完整 ``ParticleExecutionState``、每條粒子的 PCG64DXSM generator state、triangle hint
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
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
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
    """schema 2.0／2.1／2.2 execution checkpoint 的記憶體表示，供 loader 與 ``ProductionBatch`` 共用。

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

    @property
    def particle_order(self) -> tuple[str, ...]:
        """依 checkpoint 固定順序回傳 particle_id，供 restore 與稽核報告使用。"""

        return tuple(identity["particle_id"] for identity in self.run_unit_identities)


_SCHEMA20_VERSION = "2.0.0"
_SCHEMA21_VERSION = "2.1.0"
_SCHEMA22_VERSION = "2.2.0"
_WRITER_SCHEMA_VERSION = _SCHEMA22_VERSION
_SUPPORTED_EXECUTION_SCHEMA_VERSIONS = frozenset(
    {_SCHEMA20_VERSION, _SCHEMA21_VERSION, _SCHEMA22_VERSION}
)
_SCHEMA2_DATA_FILES = frozenset({"execution_state.json", "rng_states.json"})
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
    """把 Observation 序列化成 writer 固定發布的 2.2 JSON object。

    2.2 以明確固定的 23 欄保存舊有位置／狀態、環境 context 與速度紀錄；不使用目前
    ``Observation`` dataclass 的反射欄位集合，避免日後新增執行期欄位時改變檔案拓撲。
    列舉使用穩定的 ``value`` 而不是 Python 名稱；``None`` 是唯一的 JSON 缺值表示，
    絕不把 NaN／無限值當作缺值。函式刻意拒絕以 schema 2.0／2.1 寫檔，因為兩者僅是
    loader 的舊檔工程相容格式，不能讓新 writer 產生沒有新速度紀錄的 checkpoint。
    """

    if schema_version != _WRITER_SCHEMA_VERSION:
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
    schema 2.2 才允許再加固定的 11 個速度欄位。舊版本讀取後新增欄位一律保持 engine
    預設的 ``NOT_SAMPLED`` 與 ``None``，不從時間、位置、月份、深度或總速度推測。2.1／
    2.2 的有限數值、合法 ``YYYYMM``、狀態列舉與原生整數品質旗標先經本模組的 JSON
    邊界檢查，再交給 engine constructor 做 status／垂向範圍／速度總和的 cross-field
    驗證。這條責任界線確保舊檔可重啟，但不會冒充 F03／F09 正式證據。
    """

    if schema_version == _SCHEMA20_VERSION:
        expected_fields = _LEGACY_OBSERVATION_FIELDS
    elif schema_version == _SCHEMA21_VERSION:
        expected_fields = _SCHEMA21_OBSERVATION_FIELDS
    elif schema_version == _SCHEMA22_VERSION:
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
    if schema_version in {_SCHEMA21_VERSION, _SCHEMA22_VERSION}:
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
    if schema_version == _SCHEMA22_VERSION:
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

    if type(value) is not dict:
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

    execution 的資料拓撲在 2.0／2.1／2.2 間保持不變，差異只在 observation 欄位；
    writer 明示傳入 2.2，並由 observation serializer 保證永不產生沒有新速度欄位的舊
    payload。讀取端則依 metadata 版本選擇對應的固定 observation 欄位集合。
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
    """建立 execution snapshot，供記憶體檢查及 schema 2.2 磁碟 writer 共用。

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
        copied_executions.append(deepcopy(execution))
        copied_rng_states.append(_copy_rng_state(rng))
    return ExecutionCheckpoint(
        binding=binding,
        sequence=sequence,
        run_unit_identities=deepcopy(identities),
        executions=copied_executions,
        rng_states=copied_rng_states,
        triangle_hints=normalized_hints,
    )


def write_execution_checkpoint(
    destination: str | Path,
    *,
    binding: CheckpointBinding,
    run_units: Sequence[Any],
    executions: Sequence[ParticleExecutionState],
    rngs: Sequence[np.random.Generator],
    triangle_hints: Sequence[int | None],
    sequence: int,
) -> Path:
    """以 schema 2.2.0 原子寫入 execution checkpoint，保存完整軌跡與 RNG continuation。

    目標已存在時拒絕覆寫；寫入期間使用同父目錄的 ``.partial-*``，只有 execution JSON、
    RNG JSON、大小與 SHA-256 manifest 都成功建立後才以 ``os.replace`` 發布。空 shard、
    duplicate identity、非 PCG64DXSM、未知狀態或不一致欄位會在建立 partial 前拒絕。新
    writer 沒有 downgrade 參數；schema 2.0／2.1 僅由 loader 相容讀取，不能由新寫入流程
    製造。新 writer 固定寫出 2.2 的 11 個速度欄位，即使 observation 是 ``NOT_SAMPLED``
    也會保存其明示的狀態與 ``None`` 缺值，而不會將舊資料偽裝成速度證據。
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
    partial.mkdir()
    try:
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
                schema_version=_WRITER_SCHEMA_VERSION,
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
            "schema_version": _WRITER_SCHEMA_VERSION,
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
    except Exception:
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
    """嚴格讀取 schema 2.0／2.1／2.2 checkpoint，並可核對同一 shard 的固定 RunUnit 順序。

    loader 先以 metadata 的精確 schema version 決定 observation 欄位契約：2.0 只接受舊七
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
    )


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
    partial.mkdir()
    try:
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
    except Exception:
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
    "CheckpointBinding",
    "ExecutionCheckpoint",
    "build_execution_checkpoint",
    "load_checkpoint",
    "load_execution_checkpoint",
    "write_checkpoint",
    "write_execution_checkpoint",
]
