"""安全寫出軌跡、事件與檢查資料，避免留下半套結果。

所有粒子的觀測點放在連續的一維陣列；``trajectory_offsets`` 記錄每條軌跡在陣列中的起點
和終點，所以不需要難以處理的巢狀陣列。每個結果分片先寫到暫存目錄，確認檔案大小、筆數
和 SHA-256 檔案指紋都正確後，才一次改成正式名稱。已有同名結果時一律拒絕覆寫。
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .engine import EnvironmentSampleStatus, Observation, ParticleResult
from .models import BoundaryEvent, EventType, ParticleState, ParticleStatus

# trajectory shard 的 writer 只發布此版本；1.0.0 僅保留為唯讀相容格式，不能再由 writer
# 產生。版本值寫入 manifest，讓下游可以在讀取第一個 observation 前決定固定 payload 拓撲。
TRAJECTORY_SHARD_SCHEMA_VERSION = "2.0.0"
_LEGACY_TRAJECTORY_SHARD_SCHEMA_VERSION = "1.0.0"

# 這九個檔案是 legacy 1.0.0 的固定 payload。status_code.npy 在兩個版本都仍然是粒子
# 生命週期狀態字串；環境取樣狀態使用 v2 額外的 environment_sample_status_code.npy，
# 不可將兩者混成同一個欄位。
_BASE_SHARD_PAYLOAD_FILES = frozenset(
    {
        "particle_table.parquet",
        "events.parquet",
        "trajectory_offsets.npy",
        "time_utc_ns.npy",
        "age_seconds.npy",
        "x_m.npy",
        "y_m.npy",
        "z_m.npy",
        "status_code.npy",
    }
)
_ENVIRONMENT_SHARD_PAYLOAD_FILES = frozenset(
    {
        "environment_sample_status_code.npy",
        "eta_m.npy",
        "bed_z_m.npy",
        "forcing_month_yyyymm.npy",
        "environment_qc_flags.npy",
    }
)
_SCHEMA_PAYLOAD_FILES = {
    _LEGACY_TRAJECTORY_SHARD_SCHEMA_VERSION: _BASE_SHARD_PAYLOAD_FILES,
    TRAJECTORY_SHARD_SCHEMA_VERSION: _BASE_SHARD_PAYLOAD_FILES | _ENVIRONMENT_SHARD_PAYLOAD_FILES,
}
# 保留 union 供內部錯誤分類使用；實際 validator 會先從 manifest schema 選擇上面的精確集合。
_SHARD_PAYLOAD_FILES = _BASE_SHARD_PAYLOAD_FILES | _ENVIRONMENT_SHARD_PAYLOAD_FILES
_SHARD_FILES = _SHARD_PAYLOAD_FILES | {"manifest.json"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_YYYYMM_RE = re.compile(r"^[0-9]{6}$")
_ENVIRONMENT_STATUS_TO_CODE = {
    EnvironmentSampleStatus.NOT_SAMPLED: 0,
    EnvironmentSampleStatus.VALID: 1,
    EnvironmentSampleStatus.INVALID: 2,
}
_ENVIRONMENT_CODE_TO_STATUS = {
    code: status for status, code in _ENVIRONMENT_STATUS_TO_CODE.items()
}
_ENVIRONMENT_GEOMETRY_TOLERANCE_M = 1.0e-6
_TRACKED_RUN_METADATA_FIELDS = frozenset(
    {
        "run_id",
        "run_kind",
        "config_hash",
        "input_inventory_sha256",
        "checkpoint_input_binding_hash",
        "component_canonical_hashes",
        "geometry_canonical_hashes",
        "code_commit",
        "deployment_tree_sha256",
        "uv_lock_sha256",
        "dirty_flag",
        "seed_policy",
        "shard_id",
        "experiment_case_id",
        "resource_usage",
    }
)
_RESOURCE_USAGE_FIELDS = frozenset(
    {
        "wall_seconds",
        "process_cpu_seconds",
        "max_rss_bytes",
        "output_bytes",
        "checkpoint_bytes",
        "particle_steps",
    }
)


def _assert_finite_json(value: Any) -> None:
    """遞迴拒絕 JSON 中的 NaN／Infinity 與由極大指數解析出的 infinity。

    Python 的標準 JSON parser 預設接受非標準 ``NaN`` 常數，且可能把 ``1e400`` 解析成
    infinity；若把這類 manifest 原樣放入 validator 回傳值，結果便不再是可攜的 JSON。
    因此在任何 metadata/checksum 判定前先拒絕，讓損壞輸入固定回傳 ``valid=false``。
    """

    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("manifest 含非有限浮點數")
    if isinstance(value, dict):
        for item in value.values():
            _assert_finite_json(item)
    elif isinstance(value, list):
        for item in value:
            _assert_finite_json(item)


def _validate_hash_mapping(value: Any, *, label: str, add: Any) -> None:
    """驗證 component／geometry mapping 的 key 與 SHA-256，不拋出結構例外。"""

    if not isinstance(value, dict) or not value:
        add(f"run_metadata.{label}: not_nonempty_object")
        return
    for key, digest in value.items():
        if type(key) is not str or not key.strip():
            add(f"run_metadata.{label}: invalid_key")
        if type(digest) is not str or _SHA256_RE.fullmatch(digest) is None:
            add(f"run_metadata.{label}: invalid_sha256")


def _validate_tracked_run_metadata(metadata: Mapping[str, Any], *, require_formal: bool, add: Any) -> None:
    """驗證 formal/pilot trajectory shard 的可追溯 metadata 與工程量測型別。

    formal 與 pilot 都必須保存同一組欄位；差異只在 Git 判定能力：formal 必須是 40 位
    commit 且 ``dirty_flag=false``，pilot 可明示 ``None``，但不可省略欄位。所有時間、CPU
    與 bytes 都是非負工程量測，不參與條件式來源足跡的科學值。
    """

    missing = sorted(_TRACKED_RUN_METADATA_FIELDS - set(metadata))
    if missing:
        label = "missing_formal_fields" if require_formal else "missing_tracked_fields"
        add(f"run_metadata: {label}=" + ",".join(missing))
        return
    run_kind = metadata.get("run_kind")
    if type(run_kind) is not str or run_kind not in {"formal", "pilot"}:
        add("run_metadata.run_kind: expected_formal_or_pilot")
    if require_formal and run_kind != "formal":
        add("run_metadata.run_kind: formal_required")
    for key in ("run_id", "seed_policy", "shard_id", "experiment_case_id"):
        value = metadata.get(key)
        if type(value) is not str or not value.strip() or "/" in value or "\\" in value:
            add(f"run_metadata.{key}: invalid_string")
    for key in (
        "config_hash",
        "input_inventory_sha256",
        "checkpoint_input_binding_hash",
        "deployment_tree_sha256",
        "uv_lock_sha256",
    ):
        value = metadata.get(key)
        if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
            add(f"run_metadata.{key}: invalid_sha256")
    _validate_hash_mapping(metadata.get("component_canonical_hashes"), label="component", add=add)
    _validate_hash_mapping(metadata.get("geometry_canonical_hashes"), label="geometry", add=add)
    commit = metadata.get("code_commit")
    dirty = metadata.get("dirty_flag")
    if run_kind == "formal" or require_formal:
        if type(commit) is not str or _COMMIT_RE.fullmatch(commit) is None:
            add("run_metadata.code_commit: formal_requires_40hex")
        if dirty is not False:
            add("run_metadata.dirty_flag: formal_requires_false")
    else:
        if commit is not None and (type(commit) is not str or _COMMIT_RE.fullmatch(commit) is None):
            add("run_metadata.code_commit: pilot_invalid_nullable_commit")
        if not isinstance(dirty, (bool, type(None))):
            add("run_metadata.dirty_flag: pilot_invalid_nullable_flag")
    resource_usage = metadata.get("resource_usage")
    if not isinstance(resource_usage, dict):
        add("run_metadata.resource_usage: not_object")
        return
    missing_resources = sorted(_RESOURCE_USAGE_FIELDS - set(resource_usage))
    if missing_resources:
        add("run_metadata.resource_usage: missing=" + ",".join(missing_resources))
        return
    for key in ("wall_seconds", "process_cpu_seconds"):
        value = resource_usage.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            add(f"run_metadata.resource_usage.{key}: invalid_number")
        elif not math.isfinite(float(value)) or float(value) < 0:
            add(f"run_metadata.resource_usage.{key}: nonfinite_or_negative")
    for key in ("max_rss_bytes", "output_bytes", "checkpoint_bytes", "particle_steps"):
        value = resource_usage.get(key)
        if type(value) is not int or value < 0:
            add(f"run_metadata.resource_usage.{key}: invalid_nonnegative_integer")


def sha256_file(path: str | Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """分段讀取檔案並計算 SHA-256 指紋，不把大型資料一次放進記憶體。"""

    digest = sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    """遞迴把 Enum 轉 value，使 dataclass 事件可交給 Arrow/JSON。"""

    if hasattr(value, "value"):
        return value.value
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    return value


def _write_json(path: Path, payload: object) -> None:
    """以 UTF-8、排序 key 與禁止 NaN 的格式寫可稽核 JSON。"""

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _forcing_month_to_yyyymm_int(value: object, *, label: str) -> int:
    """把合法 UTC ``YYYYMM`` 月份轉為 v2 的 ``int32`` 整數欄位。

    月份不是可由檔名猜測的時間標籤，而是該 observation 實際取樣 forcing 的 provenance。
    ``000101`` 這類合法的一年一月仍須保留六位數語意；寫入整數後由 reader 以六位補零
    還原。故此函式只接受 ASCII 六位數、年份 0001--9999 及月份 01--12。
    """

    if type(value) is not str or _YYYYMM_RE.fullmatch(value) is None:
        raise ValueError(f"{label} 必須是合法 YYYYMM")
    year = int(value[:4])
    month = int(value[4:])
    if not 1 <= year <= 9999 or not 1 <= month <= 12:
        raise ValueError(f"{label} 必須是合法 YYYYMM")
    return year * 100 + month


def _finite_environment_value(value: object, *, label: str, allow_none: bool) -> float | None:
    """整理環境高度為原生有限浮點數，並明確處理 optional 缺值。

    ``eta_m`` 與 ``bed_z_m`` 是海面向上為正的公尺制垂向座標。這裡不接受 bool、NumPy
    scalar、字串或 infinity；optional 欄位只有 Python ``None`` 才會在 NPY 中寫成 NaN
    sentinel，避免把任意可轉換值悄悄變成科學資料。
    """

    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{label} 不可為缺值")
    if type(value) not in (int, float):
        raise TypeError(f"{label} 必須是原生有限數值")
    try:
        normalized = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{label} 必須是有限數值") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{label} 不可為非有限數值")
    return normalized


def _encode_environment_observation(
    observation: Observation,
) -> tuple[int, float, float, int, int]:
    """將單筆 Observation 的環境 context 編碼成 v2 五個 NPY scalar。

    回傳順序固定為 ``status_code``、``eta_m``、``bed_z_m``、``forcing_month_yyyymm``
    與 ``environment_qc_flags``。狀態碼是缺值語意唯一來源：code 0 會強制所有環境
    payload 為 NaN/0，code 1 只接受完整且幾何相容的有效樣本，code 2 允許上下界或月份
    缺失但必須有非零品質旗標。此 writer-level gate 不取代 Observation constructor，
    而是防止被不當 ``object.__setattr__`` 或錯誤型別物件污染的輸出繞過資料契約。
    """

    status = observation.environment_sample_status
    if type(status) is not EnvironmentSampleStatus or status not in _ENVIRONMENT_STATUS_TO_CODE:
        raise TypeError("未知 environment_sample_status")

    if status is EnvironmentSampleStatus.NOT_SAMPLED:
        if any(
            value is not None
            for value in (
                observation.eta_m,
                observation.bed_z_m,
                observation.forcing_month_id,
                observation.environment_qc_flags,
            )
        ):
            raise ValueError("not_sampled 的環境 context 必須全部是 None")
        return 0, math.nan, math.nan, 0, 0

    if status is EnvironmentSampleStatus.VALID:
        eta_m = _finite_environment_value(observation.eta_m, label="eta_m", allow_none=False)
        bed_z_m = _finite_environment_value(observation.bed_z_m, label="bed_z_m", allow_none=False)
        month = _forcing_month_to_yyyymm_int(observation.forcing_month_id, label="forcing_month_id")
        qc = observation.environment_qc_flags
        if type(qc) is not int or qc != 0:
            raise ValueError("valid 的 environment_qc_flags 必須是原生整數 0")
        z_m = _finite_environment_value(observation.z_m, label="z_m", allow_none=False)
        assert eta_m is not None and bed_z_m is not None and z_m is not None
        if (
            bed_z_m > z_m + _ENVIRONMENT_GEOMETRY_TOLERANCE_M
            or z_m > eta_m + _ENVIRONMENT_GEOMETRY_TOLERANCE_M
            or bed_z_m > eta_m + _ENVIRONMENT_GEOMETRY_TOLERANCE_M
        ):
            raise ValueError("valid context 的 bed_z_m、z_m、eta_m 垂向範圍不相容")
        return 1, eta_m, bed_z_m, month, 0

    # INVALID 的上下界／月份是「可有但不保證完整」的診斷 context；唯一必須存在的是
    # 非零品質旗標。None 會分別進入 NaN 與 0 sentinel，reader 會依 code 2 還原為 None。
    qc = observation.environment_qc_flags
    if type(qc) is not int or not 1 <= qc <= np.iinfo(np.uint32).max:
        raise ValueError("invalid 的 environment_qc_flags 必須介於 1 與 uint32 最大值")
    eta_m = _finite_environment_value(observation.eta_m, label="eta_m", allow_none=True)
    bed_z_m = _finite_environment_value(observation.bed_z_m, label="bed_z_m", allow_none=True)
    month = (
        0
        if observation.forcing_month_id is None
        else _forcing_month_to_yyyymm_int(observation.forcing_month_id, label="forcing_month_id")
    )
    return 2, math.nan if eta_m is None else eta_m, math.nan if bed_z_m is None else bed_z_m, month, qc


def write_trajectory_shard(
    destination: str | Path,
    results: Sequence[ParticleResult],
    *,
    run_metadata: dict[str, Any],
) -> Path:
    """安全發布一個已完成的粒子結果分片。

    ``run_metadata`` 至少要寫下設定與輸入資料的指紋、程式版本、是否有未提交修改、亂數
    種子規則和此分片涵蓋的情境範圍。函式不替呼叫端猜這些資訊。空分片沒有可用分母，
    不能產生可解釋的比例，因此直接拒絕。
    """

    if not results:
        raise ValueError("trajectory shard 不可為空")
    target = Path(destination)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"不可覆寫既有 shard：{target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.parent / f".{target.name}.partial-{uuid4().hex}"
    partial.mkdir()
    try:
        particle_rows: list[dict[str, Any]] = []
        event_rows: list[dict[str, Any]] = []
        offsets = [0]
        time_values: list[int] = []
        age_values: list[float] = []
        x_values: list[float] = []
        y_values: list[float] = []
        z_values: list[float] = []
        status_values: list[str] = []
        environment_status_values: list[int] = []
        eta_values: list[float] = []
        bed_values: list[float] = []
        forcing_month_values: list[int] = []
        environment_qc_values: list[int] = []
        for result in results:
            state = result.final_state
            particle_rows.append(
                {
                    "particle_id": state.particle_id,
                    "scenario_id": state.scenario_id,
                    "member_id": state.member_id,
                    "study_site_id": state.study_site_id,
                    "analysis_region_id": state.analysis_region_id,
                    "receptor_id": state.receptor_id,
                    "final_status": state.status.value,
                    "step_count": result.step_count,
                    "minimum_clamp_count": result.minimum_clamp_count,
                }
            )
            for observation in result.observations:
                time_values.append(observation.time_utc_ns)
                age_values.append(observation.age_seconds)
                x_values.append(observation.x_m)
                y_values.append(observation.y_m)
                z_values.append(observation.z_m)
                status_values.append(observation.status.value)
                (
                    environment_status,
                    eta_m,
                    bed_z_m,
                    forcing_month_yyyymm,
                    environment_qc,
                ) = _encode_environment_observation(observation)
                environment_status_values.append(environment_status)
                eta_values.append(eta_m)
                bed_values.append(bed_z_m)
                forcing_month_values.append(forcing_month_yyyymm)
                environment_qc_values.append(environment_qc)
            offsets.append(len(time_values))
            for event in result.events:
                row = {key: _json_safe(value) for key, value in asdict(event).items()}
                row["attributes_json"] = json.dumps(row.pop("attributes"), ensure_ascii=False, sort_keys=True)
                event_rows.append(row)

        pq.write_table(pa.Table.from_pylist(particle_rows), partial / "particle_table.parquet")
        event_table = (
            pa.Table.from_pylist(event_rows)
            if event_rows
            else pa.table({"particle_id": pa.array([], pa.string())})
        )
        pq.write_table(event_table, partial / "events.parquet")
        arrays = {
            "trajectory_offsets.npy": np.asarray(offsets, dtype=np.int64),
            "time_utc_ns.npy": np.asarray(time_values, dtype=np.int64),
            "age_seconds.npy": np.asarray(age_values, dtype=np.float64),
            "x_m.npy": np.asarray(x_values, dtype=np.float64),
            "y_m.npy": np.asarray(y_values, dtype=np.float64),
            "z_m.npy": np.asarray(z_values, dtype=np.float64),
            "status_code.npy": np.asarray(status_values, dtype="U32"),
            "environment_sample_status_code.npy": np.asarray(environment_status_values, dtype=np.uint8),
            "eta_m.npy": np.asarray(eta_values, dtype=np.float64),
            "bed_z_m.npy": np.asarray(bed_values, dtype=np.float64),
            "forcing_month_yyyymm.npy": np.asarray(forcing_month_values, dtype=np.int32),
            "environment_qc_flags.npy": np.asarray(environment_qc_values, dtype=np.uint32),
        }
        for filename, values in arrays.items():
            np.save(partial / filename, values, allow_pickle=False)
        files = sorted(path for path in partial.iterdir() if path.is_file())
        manifest = {
            "schema_version": TRAJECTORY_SHARD_SCHEMA_VERSION,
            "particle_count": len(results),
            "observation_count": len(time_values),
            "event_count": len(event_rows),
            "run_metadata": run_metadata,
            "files": {
                path.name: {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in files
            },
        }
        _write_json(partial / "manifest.json", manifest)
        os.replace(partial, target)
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return target


def validate_trajectory_shard(
    path: str | Path,
    *,
    require_formal_metadata: bool = False,
    strict_run_metadata: bool = False,
    expected_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """以不拋出結構性例外的方式重新確認一個 trajectory shard。

    驗證器第一個資料邊界永遠是 shard 根目錄下固定名稱的 ``manifest.json``；讀出並確認
    schema 後，才依 1.0.0 或 2.0.0 選擇 hard-coded payload 拓撲，絕不追隨 manifest
    提供的任意路徑。兩個版本都保留原九個軌跡／事件檔案；2.0.0 另須有五個環境 context
    陣列。輸入不存在、JSON 損壞、固定檔案被替換成目錄或 symlink、checksum 失敗及
    Parquet/NumPy 內容錯誤都轉成可機器讀取的錯誤，不讓上層 reconcile 因 ``KeyError``
    中斷。``expected_metadata`` 用於 run controller reconcile 的 binding 比對；值會以
    JSON 語意逐欄比較。``strict_run_metadata`` 用於 tracked formal/pilot run；
    ``require_formal_metadata`` 另要求 commit/dirty 的 formal gate，並隱含 strict mode。
    合成 smoke 可維持較寬鬆 metadata，但固定 payload、checksum、物理順序與 v2 環境狀態
    gate 不放寬。v1 是工程相容格式，不含垂向環境 context，不能作為 F03/F09 的正式垂向
    證據。
    """

    root = Path(path)
    errors: list[str] = []
    manifest: dict[str, Any] | None = None

    def add(error: str) -> None:
        """避免同一個結構問題在多個下游檢查重複堆疊。"""

        if error not in errors:
            errors.append(error)

    try:
        # 先驗證根目錄與固定 manifest。這個順序是安全邊界：尚未知道 schema 前，不讀取
        # 任何 payload，也不接受 manifest 內可能含有的任意檔名。
        if root.is_symlink():
            add("root: symlink")
        elif not root.exists():
            add("root: missing")
        elif not root.is_dir():
            add("root: not_directory")
        if errors:
            return {"valid": False, "errors": errors, "manifest": None}

        manifest_path = root / "manifest.json"
        if manifest_path.is_symlink():
            add("manifest.json: symlink")
        elif not manifest_path.is_file():
            add("manifest.json: missing_or_not_file")
        else:
            descriptor: int | None = None
            try:
                # O_NOFOLLOW 防止 check-then-open 期間把固定 manifest 替換成 symlink；
                # manifest 只讀取 UTF-8 JSON，payload 路徑仍完全不由其內容決定。
                flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
                descriptor = os.open(manifest_path, flags)
                with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
                    descriptor = None

                    def reject_constant(token: str) -> None:
                        """拒絕 JSON 規格外的 NaN 與正負 Infinity 常數。"""

                        raise ValueError(f"manifest 含非有限常數：{token}")

                    def reject_duplicate_manifest_keys(
                        pairs: list[tuple[str, Any]],
                    ) -> dict[str, Any]:
                        """拒絕 manifest 任一層級的重複 JSON key，避免靜默覆寫契約。"""

                        result: dict[str, Any] = {}
                        for key, value in pairs:
                            if key in result:
                                raise ValueError("manifest 含重複 key")
                            result[key] = value
                        return result

                    loaded = json.load(
                        handle,
                        object_pairs_hook=reject_duplicate_manifest_keys,
                        parse_constant=reject_constant,
                    )
                _assert_finite_json(loaded)
                if not isinstance(loaded, dict):
                    add("manifest.json: root_not_object")
                else:
                    manifest = loaded
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
                add(f"manifest.json: invalid_json:{type(exc).__name__}")
            finally:
                if descriptor is not None:
                    os.close(descriptor)

        if manifest is None:
            return {"valid": False, "errors": errors, "manifest": None}

        required_manifest = {
            "schema_version",
            "particle_count",
            "observation_count",
            "event_count",
            "run_metadata",
            "files",
        }
        missing_manifest = sorted(required_manifest - set(manifest))
        if missing_manifest:
            add("manifest: missing_keys=" + ",".join(missing_manifest))
        unknown_manifest = sorted(set(manifest) - required_manifest)
        if unknown_manifest:
            add("manifest: unknown_keys=" + ",".join(unknown_manifest))

        schema_version = manifest.get("schema_version")
        if type(schema_version) is not str or schema_version not in _SCHEMA_PAYLOAD_FILES:
            add("manifest.schema_version: unsupported")
            return {"valid": False, "errors": errors, "manifest": manifest}
        expected_payload_files = _SCHEMA_PAYLOAD_FILES[schema_version]

        for name in ("particle_count", "observation_count", "event_count"):
            value = manifest.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                add(f"manifest.{name}: invalid_nonnegative_integer")
        run_metadata = manifest.get("run_metadata")
        if not isinstance(run_metadata, dict):
            add("run_metadata: not_object")
            run_metadata = {}
        if strict_run_metadata or require_formal_metadata:
            _validate_tracked_run_metadata(
                run_metadata,
                require_formal=require_formal_metadata,
                add=add,
            )
        if expected_metadata is not None:
            for key, expected in expected_metadata.items():
                if key not in run_metadata:
                    add(f"run_metadata: missing_expected_field={key}")
                elif run_metadata[key] != expected:
                    add(f"run_metadata: expected_mismatch={key}")

        # schema 決定後才掃描實際目錄；每個版本都要求根目錄只有固定 payload 加 manifest。
        try:
            entries = tuple(root.iterdir())
        except OSError as exc:
            add(f"root: unreadable:{type(exc).__name__}")
            entries = ()
        actual_names = {entry.name for entry in entries}
        expected_root_files = expected_payload_files | {"manifest.json"}
        for entry in entries:
            if entry.is_symlink():
                add(f"{entry.name}: symlink")
            elif not entry.is_file():
                add(f"{entry.name}: unexpected_directory_or_special_file")
        for name in sorted(actual_names - expected_root_files):
            add(f"{name}: unknown_file_or_directory")
        for name in sorted(expected_root_files - actual_names):
            add(f"{name}: missing")

        files = manifest.get("files")
        if not isinstance(files, dict):
            add("files: not_object")
            files = {}
        elif set(files) != expected_payload_files:
            add("files: fixed_payload_set_mismatch")
        for filename, contract in files.items():
            # 只檢查 manifest key 是否屬於目前版本的固定集合；即使 key 惡意指向外部，
            # 後續 checksum 與內容讀取也只會使用 expected_payload_files 的 literal 名稱。
            if (
                not isinstance(filename, str)
                or Path(filename).name != filename
                or filename not in expected_payload_files
            ):
                add(f"files: unsafe_path={filename!r}")
                continue
            if not isinstance(contract, dict) or set(contract) != {"size_bytes", "sha256"}:
                add(f"{filename}: checksum_contract_invalid")
                continue
            expected_size = contract.get("size_bytes")
            expected_sha = contract.get("sha256")
            if (
                isinstance(expected_size, bool)
                or not isinstance(expected_size, int)
                or expected_size < 0
                or not isinstance(expected_sha, str)
                or _SHA256_RE.fullmatch(expected_sha) is None
            ):
                add(f"{filename}: checksum_contract_invalid")
                continue
            file_path = root / filename
            if file_path.is_symlink():
                add(f"{filename}: symlink")
            elif not file_path.is_file():
                add(f"{filename}: missing_or_not_file")
            else:
                try:
                    if file_path.stat().st_size != expected_size:
                        add(f"{filename}: size")
                    elif sha256_file(file_path) != expected_sha:
                        add(f"{filename}: checksum")
                except OSError as exc:
                    add(f"{filename}: unreadable:{type(exc).__name__}")

        # 有固定檔案才進入內容層；任何單一 reader 失敗都保留其餘錯誤，讓 CLI 能一次回報
        # 可修復的 shard 問題，而不是只露出第一個例外。
        def load_array(filename: str) -> np.ndarray | None:
            """以固定 literal 檔名載入不允許 pickle 的 NumPy payload。"""

            file_path = root / filename
            if file_path.is_symlink() or not file_path.is_file():
                return None
            try:
                values = np.load(file_path, mmap_mode="r", allow_pickle=False)
                if not isinstance(values, np.ndarray):
                    add(f"{filename}: not_ndarray")
                    return None
                return values
            except (OSError, ValueError, EOFError, TypeError) as exc:
                add(f"{filename}: invalid_npy:{type(exc).__name__}")
                return None

        arrays: dict[str, np.ndarray | None] = {
            filename: load_array(filename)
            for filename in (
                "trajectory_offsets.npy",
                "time_utc_ns.npy",
                "age_seconds.npy",
                "x_m.npy",
                "y_m.npy",
                "z_m.npy",
                "status_code.npy",
            )
        }
        if schema_version == TRAJECTORY_SHARD_SCHEMA_VERSION:
            arrays.update(
                {
                    filename: load_array(filename)
                    for filename in _ENVIRONMENT_SHARD_PAYLOAD_FILES
                }
            )

        expected_dtypes = {
            "trajectory_offsets.npy": np.dtype(np.int64),
            "time_utc_ns.npy": np.dtype(np.int64),
            "age_seconds.npy": np.dtype(np.float64),
            "x_m.npy": np.dtype(np.float64),
            "y_m.npy": np.dtype(np.float64),
            "z_m.npy": np.dtype(np.float64),
            "status_code.npy": np.dtype("U32"),
        }
        if schema_version == TRAJECTORY_SHARD_SCHEMA_VERSION:
            expected_dtypes.update(
                {
                    "environment_sample_status_code.npy": np.dtype(np.uint8),
                    "eta_m.npy": np.dtype(np.float64),
                    "bed_z_m.npy": np.dtype(np.float64),
                    "forcing_month_yyyymm.npy": np.dtype(np.int32),
                    "environment_qc_flags.npy": np.dtype(np.uint32),
                }
            )
        for filename, expected_dtype in expected_dtypes.items():
            values = arrays.get(filename)
            if values is not None:
                if values.dtype != expected_dtype:
                    add(f"{filename}: dtype")
                if values.ndim != 1:
                    add(f"{filename}: dimension")

        particle_table = None
        particle_path = root / "particle_table.parquet"
        if particle_path.is_file() and not particle_path.is_symlink():
            try:
                particle_table = pq.read_table(particle_path)
            except (OSError, ValueError, pa.ArrowException) as exc:
                add(f"particle_table.parquet: invalid_parquet:{type(exc).__name__}")
        event_path = root / "events.parquet"
        if event_path.is_file() and not event_path.is_symlink():
            try:
                event_rows = pq.read_metadata(event_path).num_rows
                if event_rows != manifest.get("event_count"):
                    add("events: row_count")
            except (OSError, ValueError, pa.ArrowException) as exc:
                add(f"events.parquet: invalid_parquet:{type(exc).__name__}")

        particle_count = manifest.get("particle_count")
        observation_count = manifest.get("observation_count")
        offsets = arrays.get("trajectory_offsets.npy")
        observations = arrays.get("time_utc_ns.npy")
        base_shape = observations.shape if observations is not None and observations.ndim == 1 else None
        if base_shape is not None:
            for filename in (
                "trajectory_offsets.npy",
                "age_seconds.npy",
                "x_m.npy",
                "y_m.npy",
                "z_m.npy",
                "status_code.npy",
            ):
                values = arrays.get(filename)
                if values is not None and values.shape != base_shape and filename != "trajectory_offsets.npy":
                    # offsets 的合法 shape 是 particle_count+1，不應與 observation array 同長；
                    # 其餘五個 base observation array 才必須逐筆同 shape。
                    add(f"{filename}: observation_shape")
            if observations.size != observation_count:
                add("time_utc_ns: observation_count")
        if (
            offsets is not None
            and observations is not None
            and offsets.ndim == 1
            and observations.ndim == 1
            and isinstance(particle_count, int)
            and not isinstance(particle_count, bool)
            and isinstance(observation_count, int)
            and not isinstance(observation_count, bool)
        ):
            if (
                offsets.shape != (particle_count + 1,)
                or offsets.size == 0
                or int(offsets[0]) != 0
                or int(offsets[-1]) != observations.size
            ):
                add("trajectory_offsets: csr_contract")
            elif np.any(np.diff(offsets) <= 0):
                add("trajectory_offsets: empty_or_nonmonotonic")
        if (
            particle_table is not None
            and isinstance(particle_count, int)
            and not isinstance(particle_count, bool)
        ):
            if particle_table.num_rows != particle_count:
                add("particle_table: row_count")
            try:
                particle_rows = particle_table.to_pylist()
                particle_ids = [row["particle_id"] for row in particle_rows]
                if len(set(particle_ids)) != len(particle_ids):
                    add("particle_table: duplicate_particle_id")
            except (KeyError, TypeError) as exc:
                add(f"particle_table: missing_identity_column:{type(exc).__name__}")
                particle_rows = []
        else:
            particle_rows = []

        for filename in ("age_seconds.npy", "x_m.npy", "y_m.npy", "z_m.npy"):
            values = arrays.get(filename)
            if (
                values is not None
                and values.dtype == np.dtype(np.float64)
                and values.ndim == 1
                and not np.all(np.isfinite(values))
            ):
                add(f"{filename}: nonfinite")
        status_values = arrays.get("status_code.npy")
        known_statuses = {status.value for status in ParticleStatus}
        if (
            status_values is not None
            and status_values.dtype == np.dtype("U32")
            and status_values.ndim == 1
            and any(str(value) not in known_statuses for value in status_values)
        ):
            add("status_code: unknown")

        # v2 五個環境欄位的狀態碼是缺值語意唯一來源；NaN 只作 optional payload sentinel，
        # 不允許以 NaN／0 自行推論 valid 或 invalid。逐筆檢查也把 z_m 的公尺制幾何關係
        # 綁回同一 observation，避免錯掛另一個時間點的海面／海床資料。
        if schema_version == TRAJECTORY_SHARD_SCHEMA_VERSION:
            environment_code = arrays.get("environment_sample_status_code.npy")
            eta_values = arrays.get("eta_m.npy")
            bed_values = arrays.get("bed_z_m.npy")
            month_values = arrays.get("forcing_month_yyyymm.npy")
            qc_values = arrays.get("environment_qc_flags.npy")
            if base_shape is not None:
                for filename, values in (
                    ("environment_sample_status_code.npy", environment_code),
                    ("eta_m.npy", eta_values),
                    ("bed_z_m.npy", bed_values),
                    ("forcing_month_yyyymm.npy", month_values),
                    ("environment_qc_flags.npy", qc_values),
                ):
                    if values is not None and values.shape != base_shape:
                        add(f"{filename}: observation_shape")
            environment_ready = (
                environment_code is not None
                and eta_values is not None
                and bed_values is not None
                and month_values is not None
                and qc_values is not None
                and observations is not None
                and arrays.get("z_m.npy") is not None
                and all(
                    values.shape == observations.shape
                    for values in (
                        environment_code,
                        eta_values,
                        bed_values,
                        month_values,
                        qc_values,
                    )
                )
                and all(
                    values.ndim == 1
                    and values.dtype == expected_dtypes[filename]
                    for filename, values in (
                        ("environment_sample_status_code.npy", environment_code),
                        ("eta_m.npy", eta_values),
                        ("bed_z_m.npy", bed_values),
                        ("forcing_month_yyyymm.npy", month_values),
                        ("environment_qc_flags.npy", qc_values),
                    )
                )
            )
            if environment_ready:
                assert environment_code is not None
                assert eta_values is not None
                assert bed_values is not None
                assert month_values is not None
                assert qc_values is not None
                assert observations is not None
                z_values = arrays["z_m.npy"]
                assert z_values is not None
                if not np.all(np.isin(environment_code, np.array([0, 1, 2], dtype=np.uint8))):
                    add("environment_sample_status_code: unknown")
                for index, code_value in enumerate(environment_code):
                    code = int(code_value)
                    eta = float(eta_values[index])
                    bed = float(bed_values[index])
                    month = int(month_values[index])
                    qc = int(qc_values[index])
                    eta_is_nan = math.isnan(eta)
                    bed_is_nan = math.isnan(bed)
                    eta_is_finite = math.isfinite(eta)
                    bed_is_finite = math.isfinite(bed)
                    month_is_valid = (
                        1 <= month // 100 <= 9999
                        and 1 <= month % 100 <= 12
                        and month >= 100
                    )
                    if code == 0:
                        if not eta_is_nan or not bed_is_nan or month != 0 or qc != 0:
                            add(f"environment[{index}]: not_sampled_contract")
                    elif code == 1:
                        if (
                            not eta_is_finite
                            or not bed_is_finite
                            or not month_is_valid
                            or qc != 0
                            or bed > float(z_values[index]) + _ENVIRONMENT_GEOMETRY_TOLERANCE_M
                            or float(z_values[index]) > eta + _ENVIRONMENT_GEOMETRY_TOLERANCE_M
                            or bed > eta + _ENVIRONMENT_GEOMETRY_TOLERANCE_M
                        ):
                            add(f"environment[{index}]: valid_contract")
                    elif code == 2 and (
                        not (eta_is_nan or eta_is_finite)
                        or not (bed_is_nan or bed_is_finite)
                        or not (month == 0 or month_is_valid)
                        or qc == 0
                    ):
                        add(f"environment[{index}]: invalid_contract")
        if (
            offsets is not None
            and observations is not None
            and offsets.ndim == 1
            and observations.ndim == 1
            and offsets.shape == (len(particle_rows) + 1,)
            and arrays.get("age_seconds.npy") is not None
            and arrays.get("status_code.npy") is not None
            and arrays["age_seconds.npy"].ndim == 1
            and arrays["status_code.npy"].ndim == 1
        ):
            age_values = arrays["age_seconds.npy"]
            status_values = arrays["status_code.npy"]
            for index, row in enumerate(particle_rows):
                try:
                    start = int(offsets[index])
                    stop = int(offsets[index + 1])
                    if start < 0 or stop <= start or stop > observations.size:
                        add(f"particle[{index}]: offset_contract")
                        continue
                    age = age_values[start:stop]
                    utc = observations[start:stop]
                    if age.size == 0 or not np.isclose(age[0], 0.0) or np.any(np.diff(age) <= 0):
                        add(f"particle[{index}]: age_contract")
                    if utc.size == 0 or np.any(np.diff(utc) >= 0):
                        add(f"particle[{index}]: backward_time_contract")
                    if str(status_values[stop - 1]) != row.get("final_status"):
                        add(f"particle[{index}]: final_status")
                except (IndexError, KeyError, TypeError, ValueError) as exc:
                    add(f"particle[{index}]: content_error:{type(exc).__name__}")
        return {"valid": not errors, "errors": errors, "manifest": manifest}
    except (OSError, ValueError, TypeError, KeyError, pa.ArrowException) as exc:
        # validator 是稽核 API；即使遇到未預期的壞 payload，也必須把錯誤轉成 JSON-safe 結果。
        add(f"validator_exception:{type(exc).__name__}")
        return {"valid": False, "errors": errors, "manifest": manifest}


def _reject_duplicate_attribute_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """保留事件屬性 JSON 的鍵唯一性，避免標準 parser 靜默覆蓋前一個值。

    ``attributes_json`` 是 writer 將 ``BoundaryEvent.attributes`` 展平後保存的 JSON
    字串。Python 的一般 JSON 解析器遇到重複鍵時會保留最後一個值，這會讓結果無法判斷
    原始事件究竟包含哪個屬性；因此 reader 使用 ``object_pairs_hook`` 在解析階段拒絕
    重複鍵。這個 hook 也會作用於巢狀物件，即使巢狀物件之後會因不是允許的純量而被拒絕。
    """

    parsed: dict[str, Any] = {}
    for key, value in pairs:
        if key in parsed:
            raise ValueError("attributes_json 含重複鍵")
        parsed[key] = value
    return parsed


def _reject_nonfinite_attribute_constant(token: str) -> None:
    """拒絕 JSON 規格外的 NaN、Infinity 與 -Infinity 常數。"""

    raise ValueError(f"attributes_json 含非有限常數：{token}")


def _read_event_attributes(value: Any) -> dict[str, bool | float | int | str]:
    """嚴格解析事件屬性，僅接受由布林、有限浮點數、整數與字串組成的物件。

    事件屬性是診斷補充資料，不應透過任意 JSON 結構或特殊浮點值攜帶未定義語意。
    根物件以外的陣列、巢狀物件與 ``null`` 都不在 schema 1.0.0 的允許集合內；這樣
    讀回後的 ``BoundaryEvent.attributes`` 才能直接符合模型宣告的 scalar mapping，
    並避免把缺值誤當成有效科學量。
    """

    if not isinstance(value, str):
        raise ValueError("attributes_json 必須是字串")
    parsed = json.loads(
        value,
        object_pairs_hook=_reject_duplicate_attribute_keys,
        parse_constant=_reject_nonfinite_attribute_constant,
    )
    if not isinstance(parsed, dict):
        raise ValueError("attributes_json 必須是 JSON object")
    for key, item in parsed.items():
        if type(key) is not str:
            raise ValueError("attributes_json 的鍵必須是字串")
        if type(item) is bool or type(item) is int or type(item) is str:
            continue
        if type(item) is float and math.isfinite(item):
            continue
        # ``json.loads`` 會把極大指數轉成 infinity；這個分支同時攔截該情況以及
        # null、陣列與巢狀物件，避免只依賴 parse_constant 而留下非有限數值。
        raise ValueError("attributes_json 含不允許的值")
    return parsed


def _read_text_field(row: Mapping[str, Any], field: str) -> str:
    """從 Parquet row 取出必填字串欄位，拒絕 null 與隱式數值轉字串。"""

    value = row[field]
    if isinstance(value, np.str_):
        value = str(value)
    if not isinstance(value, str):
        raise ValueError(f"欄位 {field} 必須是字串")
    return value


def _read_integer_field(row: Mapping[str, Any], field: str, *, nonnegative: bool = False) -> int:
    """從 Parquet row 取出整數欄位，拒絕布林、浮點與負的計數值。"""

    value = row[field]
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"欄位 {field} 必須是整數")
    converted = int(value)
    if nonnegative and converted < 0:
        raise ValueError(f"欄位 {field} 不可為負")
    return converted


def _read_finite_float_field(row: Mapping[str, Any], field: str) -> float:
    """從 Parquet row 取出有限公尺／秒／比例等浮點欄位。"""

    value = row[field]
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"欄位 {field} 必須是數值")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"欄位 {field} 不可為非有限數值")
    return converted


def _read_optional_text_field(row: Mapping[str, Any], field: str) -> str | None:
    """從 Parquet row 取出可為 null 的字串欄位。"""

    value = row[field]
    if value is None:
        return None
    return _read_text_field(row, field)


def _read_optional_integer_field(row: Mapping[str, Any], field: str) -> int | None:
    """從 Parquet row 取出可為 null 的整數欄位。"""

    if row[field] is None:
        return None
    return _read_integer_field(row, field)


def _read_optional_float_field(row: Mapping[str, Any], field: str) -> float | None:
    """從 Parquet row 取出可為 null 且必須有限的浮點欄位。"""

    if row[field] is None:
        return None
    return _read_finite_float_field(row, field)


def _read_status_value(value: Any) -> ParticleStatus:
    """把固定狀態字串轉成 ``ParticleStatus``，不接受未知狀態。"""

    if isinstance(value, np.str_):
        value = str(value)
    if not isinstance(value, str):
        raise ValueError("狀態欄位必須是字串")
    try:
        return ParticleStatus(value)
    except ValueError as exc:
        raise ValueError("狀態欄位不是已定義的 ParticleStatus") from exc


def _read_event_type(value: Any) -> EventType:
    """把事件名稱轉成固定的 ``EventType``，拒絕未登錄事件。"""

    if isinstance(value, np.str_):
        value = str(value)
    if not isinstance(value, str):
        raise ValueError("事件類型欄位必須是字串")
    try:
        return EventType(value)
    except ValueError as exc:
        raise ValueError("事件類型欄位不是已定義的 EventType") from exc


def _load_shard_array(root: Path, filename: str) -> np.ndarray:
    """以固定檔名、唯讀 mmap 與 ``allow_pickle=False`` 載入一個 NumPy 陣列。

    reader 不採用 manifest 提供的路徑，因為 schema 1.0.0／2.0.0 的檔案拓撲已固定；
    即使 manifest 被竄改，也只能讀取 shard 根目錄下預先定義的檔名。陣列仍以 mmap
    開啟，讓大型軌跡不必先完整複製到記憶體；回傳的結果物件只保存逐筆 Python 值。
    """

    file_path = root / filename
    if file_path.is_symlink() or not file_path.is_file():
        raise ValueError("固定 NumPy payload 不存在或不是一般檔案")
    values = np.load(file_path, mmap_mode="r", allow_pickle=False)
    if not isinstance(values, np.ndarray):
        raise ValueError("NumPy payload 必須是單一 ndarray")
    return values


def read_trajectory_shard(
    path: str | Path,
    *,
    require_formal_metadata: bool = False,
    strict_run_metadata: bool = False,
    expected_metadata: Mapping[str, Any] | None = None,
) -> tuple[ParticleResult, ...]:
    """安全讀回 1.0.0 legacy 或 2.0.0 trajectory shard。

    函式第一個資料操作一定是呼叫 ``validate_trajectory_shard``，並完整轉送 formal
    metadata、strict metadata 與 expected metadata 三個驗證選項；驗證結果不是明確的
    ``True`` 時立即停止。通過後才從固定檔名讀取 Parquet 與 NumPy payload，依粒子表的
    原始列順序及 ``trajectory_offsets`` 的原始區段建立 ``ParticleResult`` tuple，不做
    排序、不以事件時間重新排列，也不追隨 manifest 內的任意路徑。

    粒子的 final state 由 particle table 的 identity、final status、step 計數與最後一筆
    observation 的公尺制位置、UTC 奈秒時間、回溯年齡及狀態重建；兩版都沒有保存
    ``own_local_exit_recorded``，因此讀回時使用 ``ParticleState`` 的既有預設值 ``False``。
    legacy 1.0.0 沒有環境 context，reader 明示還原成 ``NOT_SAMPLED`` 與 ``None``，這只
    是工程相容行為，不能供 F03/F09 正式垂向證據。2.0.0 則依 status code 還原
    ``EnvironmentSampleStatus``、海面／海床公尺值、UTC ``YYYYMM`` 與品質旗標，再由
    ``Observation`` constructor 重驗。事件的 ``attributes_json`` 必須是無重複鍵的 JSON
    object，值只能是布林、有限浮點數、整數或字串。空事件表只會含 writer 目前產生的
    ``particle_id`` 空欄，讀回時直接得到空事件清單，不要求不存在的事件欄位。

    任何驗證、檔案讀取、欄位缺失、型別不符、狀態／事件名稱未知或 JSON 屬性不合法都會
    以不含輸入路徑的 ``ValueError`` fail closed；這避免把伺服器絕對路徑洩漏到上層日志。
    """

    # 這裡刻意不先建立 Path 或讀取 manifest；驗證器必須是 reader 的第一個資料邊界，
    # 且三個選項要原封不動傳遞，讓 formal caller 不會因 reader 而降低既有 gate。
    try:
        validation = validate_trajectory_shard(
            path,
            require_formal_metadata=require_formal_metadata,
            strict_run_metadata=strict_run_metadata,
            expected_metadata=expected_metadata,
        )
    except Exception:
        raise ValueError("trajectory shard 驗證失敗") from None
    if not isinstance(validation, Mapping) or validation.get("valid") is not True:
        raise ValueError("trajectory shard 驗證失敗")
    manifest = validation.get("manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("trajectory shard 驗證失敗")
    schema_version = manifest.get("schema_version")
    if schema_version not in _SCHEMA_PAYLOAD_FILES:
        raise ValueError("trajectory shard 驗證失敗")

    try:
        root = Path(path)
        offsets = _load_shard_array(root, "trajectory_offsets.npy")
        time_utc_ns = _load_shard_array(root, "time_utc_ns.npy")
        age_seconds = _load_shard_array(root, "age_seconds.npy")
        x_m = _load_shard_array(root, "x_m.npy")
        y_m = _load_shard_array(root, "y_m.npy")
        z_m = _load_shard_array(root, "z_m.npy")
        status_code = _load_shard_array(root, "status_code.npy")
        environment_status_code = None
        eta_m = None
        bed_z_m = None
        forcing_month_yyyymm = None
        environment_qc_flags = None
        if schema_version == TRAJECTORY_SHARD_SCHEMA_VERSION:
            environment_status_code = _load_shard_array(root, "environment_sample_status_code.npy")
            eta_m = _load_shard_array(root, "eta_m.npy")
            bed_z_m = _load_shard_array(root, "bed_z_m.npy")
            forcing_month_yyyymm = _load_shard_array(root, "forcing_month_yyyymm.npy")
            environment_qc_flags = _load_shard_array(root, "environment_qc_flags.npy")

        if not np.issubdtype(offsets.dtype, np.integer) or offsets.ndim != 1:
            raise ValueError("trajectory_offsets 型別或維度不符")
        if not np.issubdtype(time_utc_ns.dtype, np.integer) or time_utc_ns.ndim != 1:
            raise ValueError("time_utc_ns 型別或維度不符")
        for values in (age_seconds, x_m, y_m, z_m):
            if not np.issubdtype(values.dtype, np.number) or values.ndim != 1:
                raise ValueError("數值 observation 陣列型別或維度不符")
        if status_code.ndim != 1:
            raise ValueError("status_code 維度不符")
        if schema_version == TRAJECTORY_SHARD_SCHEMA_VERSION and any(
            values is None or values.ndim != 1
            for values in (
                environment_status_code,
                eta_m,
                bed_z_m,
                forcing_month_yyyymm,
                environment_qc_flags,
            )
        ):
            raise ValueError("environment context 維度不符")

        expected_reader_dtypes = [
            (offsets, np.dtype(np.int64)),
            (time_utc_ns, np.dtype(np.int64)),
            (age_seconds, np.dtype(np.float64)),
            (x_m, np.dtype(np.float64)),
            (y_m, np.dtype(np.float64)),
            (z_m, np.dtype(np.float64)),
            (status_code, np.dtype("U32")),
        ]
        if schema_version == TRAJECTORY_SHARD_SCHEMA_VERSION:
            assert environment_status_code is not None
            assert eta_m is not None
            assert bed_z_m is not None
            assert forcing_month_yyyymm is not None
            assert environment_qc_flags is not None
            expected_reader_dtypes.extend(
                [
                    (environment_status_code, np.dtype(np.uint8)),
                    (eta_m, np.dtype(np.float64)),
                    (bed_z_m, np.dtype(np.float64)),
                    (forcing_month_yyyymm, np.dtype(np.int32)),
                    (environment_qc_flags, np.dtype(np.uint32)),
                ]
            )
        for values, expected_dtype in expected_reader_dtypes:
            if values.dtype != expected_dtype:
                raise ValueError("trajectory shard payload dtype 不符")

        particle_rows = pq.read_table(root / "particle_table.parquet").to_pylist()
        if offsets.shape != (len(particle_rows) + 1,):
            raise ValueError("trajectory_offsets 與 particle table 筆數不一致")
        if any(values.shape != time_utc_ns.shape for values in (age_seconds, x_m, y_m, z_m, status_code)):
            raise ValueError("observation 陣列長度不一致")
        if schema_version == TRAJECTORY_SHARD_SCHEMA_VERSION and any(
            values.shape != time_utc_ns.shape
            for values in (
                environment_status_code,
                eta_m,
                bed_z_m,
                forcing_month_yyyymm,
                environment_qc_flags,
            )
        ):
            raise ValueError("environment context 長度不一致")
        if offsets.size == 0 or int(offsets[0]) != 0 or int(offsets[-1]) != time_utc_ns.size:
            raise ValueError("trajectory_offsets 邊界不符")
        if np.any(np.diff(offsets) <= 0):
            raise ValueError("trajectory_offsets 含空區段或非遞增值")

        particle_payloads: list[
            tuple[ParticleState, list[Observation], int, int, tuple[str, int, str, str, str]]
        ] = []
        particle_index: dict[str, int] = {}
        for index, row in enumerate(particle_rows):
            particle_id = _read_text_field(row, "particle_id")
            if particle_id in particle_index:
                raise ValueError("particle_id 重複")
            scenario_id = _read_text_field(row, "scenario_id")
            member_id = _read_integer_field(row, "member_id")
            study_site_id = _read_text_field(row, "study_site_id")
            analysis_region_id = _read_text_field(row, "analysis_region_id")
            receptor_id = _read_text_field(row, "receptor_id")
            final_status = _read_status_value(row["final_status"])
            step_count = _read_integer_field(row, "step_count", nonnegative=True)
            minimum_clamp_count = _read_integer_field(row, "minimum_clamp_count", nonnegative=True)

            start = int(offsets[index])
            stop = int(offsets[index + 1])
            if start < 0 or stop <= start or stop > time_utc_ns.size:
                raise ValueError("particle observation offset 超出範圍")
            observations: list[Observation] = []
            for position in range(start, stop):
                observation_status = _read_status_value(status_code[position])
                if schema_version == _LEGACY_TRAJECTORY_SHARD_SCHEMA_VERSION:
                    environment_kwargs = {
                        "environment_sample_status": EnvironmentSampleStatus.NOT_SAMPLED,
                        "eta_m": None,
                        "bed_z_m": None,
                        "forcing_month_id": None,
                        "environment_qc_flags": None,
                    }
                else:
                    assert environment_status_code is not None
                    assert eta_m is not None
                    assert bed_z_m is not None
                    assert forcing_month_yyyymm is not None
                    assert environment_qc_flags is not None
                    environment_code = int(environment_status_code[position])
                    environment_status = _ENVIRONMENT_CODE_TO_STATUS.get(environment_code)
                    if environment_status is None:
                        raise ValueError("未知 environment_sample_status code")
                    month_value = int(forcing_month_yyyymm[position])
                    month_id = None if month_value == 0 else f"{month_value:06d}"
                    eta_value = float(eta_m[position])
                    bed_value = float(bed_z_m[position])
                    environment_kwargs = {
                        "environment_sample_status": environment_status,
                        "eta_m": None if math.isnan(eta_value) else eta_value,
                        "bed_z_m": None if math.isnan(bed_value) else bed_value,
                        "forcing_month_id": month_id,
                        # code 0 的 NPY qc sentinel 是 0，但 Observation 的語意缺值仍須
                        # 還原成 None；否則會把「尚未取樣」誤建成 valid 的 qc=0。
                        "environment_qc_flags": (
                            None
                            if environment_status is EnvironmentSampleStatus.NOT_SAMPLED
                            else int(environment_qc_flags[position])
                        ),
                    }
                observations.append(
                    Observation(
                        particle_id=particle_id,
                        time_utc_ns=int(time_utc_ns[position]),
                        age_seconds=float(age_seconds[position]),
                        x_m=float(x_m[position]),
                        y_m=float(y_m[position]),
                        z_m=float(z_m[position]),
                        status=observation_status,
                        **environment_kwargs,
                    )
                )
            last_observation = observations[-1]
            if last_observation.status != final_status:
                raise ValueError("final_status 與最後 observation 狀態不一致")
            final_state = ParticleState(
                particle_id=particle_id,
                scenario_id=scenario_id,
                member_id=member_id,
                study_site_id=study_site_id,
                analysis_region_id=analysis_region_id,
                receptor_id=receptor_id,
                x_m=last_observation.x_m,
                y_m=last_observation.y_m,
                z_m=last_observation.z_m,
                time_utc_ns=last_observation.time_utc_ns,
                age_seconds=last_observation.age_seconds,
                status=final_status,
            )
            particle_index[particle_id] = index
            particle_payloads.append(
                (
                    final_state,
                    observations,
                    step_count,
                    minimum_clamp_count,
                    (scenario_id, member_id, study_site_id, analysis_region_id, receptor_id),
                )
            )

        events_by_particle: list[list[BoundaryEvent]] = [[] for _ in particle_payloads]
        event_table = pq.read_table(root / "events.parquet")
        if event_table.num_rows:
            for row in event_table.to_pylist():
                particle_id = _read_text_field(row, "particle_id")
                particle_position = particle_index.get(particle_id)
                if particle_position is None:
                    raise ValueError("事件 particle_id 不存在")
                expected_identity = particle_payloads[particle_position][4]
                event_identity = (
                    _read_text_field(row, "scenario_id"),
                    _read_integer_field(row, "member_id"),
                    _read_text_field(row, "study_site_id"),
                    _read_text_field(row, "analysis_region_id"),
                    _read_text_field(row, "receptor_id"),
                )
                if event_identity != expected_identity:
                    raise ValueError("事件 identity 與 particle table 不一致")
                events_by_particle[particle_position].append(
                    BoundaryEvent(
                        particle_id=particle_id,
                        scenario_id=event_identity[0],
                        member_id=event_identity[1],
                        study_site_id=event_identity[2],
                        analysis_region_id=event_identity[3],
                        receptor_id=event_identity[4],
                        event_type=_read_event_type(row["event_type"]),
                        time_utc_ns=_read_integer_field(row, "time_utc_ns"),
                        x_m=_read_finite_float_field(row, "x_m"),
                        y_m=_read_finite_float_field(row, "y_m"),
                        z_m=_read_finite_float_field(row, "z_m"),
                        fraction=_read_finite_float_field(row, "fraction"),
                        related_study_site_id=_read_optional_text_field(row, "related_study_site_id"),
                        boundary_segment_id=_read_optional_text_field(row, "boundary_segment_id"),
                        boundary_s_m=_read_optional_float_field(row, "boundary_s_m"),
                        source_face_id=_read_optional_integer_field(row, "source_face_id"),
                        triangle_id=_read_optional_integer_field(row, "triangle_id"),
                        forcing_month_id=_read_optional_text_field(row, "forcing_month_id"),
                        attributes=_read_event_attributes(row["attributes_json"]),
                    )
                )

        return tuple(
            ParticleResult(
                final_state=payload[0],
                observations=payload[1],
                events=events_by_particle[index],
                step_count=payload[2],
                minimum_clamp_count=payload[3],
            )
            for index, payload in enumerate(particle_payloads)
        )
    except Exception:
        # Arrow、NumPy、JSON 與型別例外的原文可能包含絕對檔案路徑；public reader 對外
        # 只提供固定且不洩漏部署位置的錯誤，詳細原因由先行 validator 負責報告。
        raise ValueError("trajectory shard 讀取失敗") from None


def temporary_output_directory(prefix: str = "lbt-") -> Path:
    """建立明示位於系統 temporary root 的測試／smoke 目錄。"""

    return Path(tempfile.mkdtemp(prefix=prefix))
