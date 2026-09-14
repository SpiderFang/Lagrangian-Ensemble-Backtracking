"""正式 run 的 immutable plan、mutable progress 與 CPU shard controller。

本模組把「要執行什麼」與「目前執行到哪裡」分成兩個不同檔案：schema 2.1
``run_plan.json`` 發布
後不可修改，保存設定、輸入、程式部署、情境排序與分片範圍；``run_progress.json`` 以
revision 原子替換，保存可恢復的生命週期與 checkpoint/output token。execution checkpoint
由 schema 3 writer 保存 immutable history segment 與 compact state；schema 2.x 僅作
loader 的唯讀遷移來源。這裡負責 run-level binding、原子發布、CPU/NumPy ProductionBatch
orchestration、Unix lock topology 及 reconcile。

正式 controller 不會讀取 raw NetCDF，也不會把 forcing、geometry 或 request factory
序列化；caller 必須以相同的 manifest/config 建立 request。這個邊界可讓 SERVER 的大型
陣列由 ``ForcingWindowManager`` 管理，而 run plan 只保存可稽核的 hash 與相對路徑 token。
execution checkpoint 由 schema 3.0／3.1 的 immutable history segment 與 compact current state
保存；controller 仍會在每次 checkpoint 以共享 ``progress.lock`` 原子發布 run-level 進度，
因此 schema 3 只改善歷史 payload 的重複寫入，不代表 NFS progress lock 競爭自動消失。
schema 2.x 只在 loader 中維持舊目錄的唯讀相容性。
"""

from __future__ import annotations

import json
import math
import os
import re
import resource
import shutil
import stat
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal
from uuid import uuid4

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .checkpoint import CheckpointBinding, inspect_execution_checkpoint, load_execution_checkpoint
from .outputs import sha256_file, validate_trajectory_shard, write_trajectory_shard
from .pilot_selection import (
    build_full_scenario_selection,
    scenario_ids_sha256,
    validate_scenario_selection_binding_shape,
)
from .production import ProductionBatch
from .provenance import CodeProvenance
from .run_locking import acquire_run_lock
from .runner import (
    SCENARIO_ORDERING_POLICY,
    ReferenceParticleRequest,
    RunUnit,
    ScenarioShard,
    _scenario_shard_id,
    iter_run_units,
    plan_scenario_shards,
    scenario_execution_sort_key,
)
from .scenarios import Scenario, stable_identifier, validate_random_stream_id

# schema 1 的 scenario_id lexical ordering 沒有明確的流場 locality 契約；schema 2.1 因此
# 只接受含 ordering policy、execution group、固定 lock topology 與 scenario selection 的新
# workspace。schema 2.0 仍可唯讀相容，但沒有 selection 時只能解讀為完整 full coverage，
# 不猜測舊 workspace 的 pilot 子集或排序後 resume。schema 2.2 僅供明示共同亂數流的
# 新 workspace 使用；未指定 stream 時仍發布 2.1，讓舊 plan 與 seed table 的 bytes／欄位
# 契約保持不變。
RUN_PLAN_SCHEMA_VERSION = "2.1.0"
RUN_PLAN_PAIRED_SCHEMA_VERSION = "2.2.0"
RUN_PLAN_LEGACY_SCHEMA_VERSION = "2.0.0"
RUN_PROGRESS_SCHEMA_VERSION = "1.0.0"
_LIFECYCLES = frozenset({"PLANNED", "RUNNING", "PAUSED", "COMPLETE", "FAILED"})
_RUN_KINDS = frozenset({"formal", "pilot", "synthetic"})
_SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_CHECKPOINT_RE = re.compile(r"^checkpoint-([0-9]{8})$")
# execution checkpoint 的 scanner 必須知道 schema 的單向遷移邊界。2.x 是既有唯讀
# 格式；3.x 是新 writer 的 immutable segment 格式。沒有 retention 時，這裡也要求
# generation 序號完整連續，避免 reconcile 從缺代或較高偽 root 靜默採認錯誤狀態。
_CHECKPOINT_SCHEMA2_VERSIONS = frozenset({"2.0.0", "2.1.0", "2.2.0"})
_CHECKPOINT_SCHEMA3_VERSIONS = frozenset({"3.0.0", "3.1.0"})
_CHECKPOINT_SCHEMA30_VERSION = "3.0.0"
_CHECKPOINT_SCHEMA3_VERSION = "3.1.0"
_CHECKPOINT_SCHEMA_VERSIONS = _CHECKPOINT_SCHEMA2_VERSIONS | _CHECKPOINT_SCHEMA3_VERSIONS
# candidate 狀態需區分「確定沒有已發布代」、「暫時無法讀取」與「已通過採認」；NFS
# 讀取錯誤不能和確定的 publish 前失敗共用同一個 False，否則合法 orphan 會被誤標 FAILED。
_CheckpointAdoptionStatus = Literal["adopted", "indeterminate", "not_adopted"]
_FAILURE_RE = re.compile(r"^failure-[0-9a-f]{32}\.json$")
_RUN_FILE_NAMES = frozenset(
    {"normalized_config.json", "input_inventory.json", "scenario_table.parquet", "seed_table.parquet"}
)
_LOCK_ROOT = "locks"
_LOCK_FIXED_NAMES = frozenset({"run_gate.lock", "progress.lock"})
_METRIC_REQUIRED_KEYS = frozenset(
    {
        "wall_seconds",
        "process_cpu_seconds",
        "max_rss_bytes",
        "output_bytes",
        "checkpoint_bytes",
        "particle_steps",
    }
)
# forcing cache 的四個計數器（counter）代表本次 process 內的累計事件數；manager_count 與
# resident_bytes 是某一時間點的狀態量（gauge）。controller 會在每個分片（shard）的單次
# 執行（invocation）開始時取起始讀值（baseline），再把 counter 轉成該 invocation 的
# 增量，避免共用 manager 的累計值被重複相加。
_FORCING_CACHE_COUNTER_KEYS = ("loads", "hits", "misses", "evictions")
_FORCING_CACHE_GAUGE_KEYS = ("manager_count", "resident_bytes")
_FORCING_CACHE_KEYS = _FORCING_CACHE_COUNTER_KEYS + _FORCING_CACHE_GAUGE_KEYS
_FORCING_CACHE_STATS_SEMANTICS_KEY = "forcing_cache_stats_semantics"
_FORCING_CACHE_STATS_SEMANTICS_V1 = "invocation_delta_v1"
_FORCING_CACHE_STATS_LEGACY = "legacy_unknown"
_FORCING_CACHE_STATS_STATUS_KEY = "forcing_cache_stats_status"
_FORCING_CACHE_STATS_STATUS_UNAVAILABLE = "unavailable_due_to_reporter_error"
_SCENARIO_COLUMNS = (
    "scenario_id",
    "study_site_id",
    "analysis_region_id",
    "material_id",
    "receptor_id",
    "arrival_time_id",
    "settling_velocity_mps",
    "arrival_time_utc_ns",
    "design_version",
)
_SEED_COLUMNS = (
    "scenario_id",
    "experiment_case_id",
    "member_id",
    "particle_id",
    "seed_128_hex",
)
_SEED_COLUMNS_PAIRED = _SEED_COLUMNS + ("random_stream_id",)


@dataclass(frozen=True, slots=True)
class RunWorkspace:
    """已原子發布的 run 目錄位置與 immutable plan 摘要。

    ``path`` 只供目前程序定位檔案，不會寫進 run JSON；``plan`` 是 read-only mapping 的
    in-memory view。呼叫端可把此物件傳給 ``RunController``，也可直接使用 ``Path``
    （controller 同樣接受 Path），以兼容批次腳本與測試 fixture。
    """

    path: Path
    plan: Mapping[str, Any]

    def __fspath__(self) -> str:
        """讓 workspace 可交給接受 ``os.PathLike`` 的標準函式。"""

        return str(self.path)

    def __truediv__(self, component: str) -> Path:
        """提供 ``workspace / "run_plan.json"`` 的 Path-like 使用方式。"""

        return self.path / component


@dataclass(frozen=True, slots=True)
class RunExecutionSummary:
    """一次 shard controller 呼叫的 machine-readable 摘要。"""

    run_id: str
    shard_id: str
    lifecycle: str
    scenario_count: int
    particle_count: int
    sweeps_completed: int
    particle_steps: int
    output_relative_path: str | None
    checkpoint_relative_path: str | None


@dataclass(frozen=True, slots=True)
class _CheckpointSelection:
    """通過 schema 3.x／舊 schema 2.x 與 run binding 驗證的最高 checkpoint generation。

    ``particle_steps`` 是所有 execution ``step_count`` 的精確總和。若 generation 已寫完、
    但程序在更新 ``latest.json`` 前中斷，schema 2 沒有 run-level sweep counter；此時只能
    以所有粒子 ``step_count`` 最大值作 sweep 下界，並以
    ``execution_step_count_lower_bound`` 明示。controller 的下一個 checkpoint cadence 只從
    此 generation 再前進固定 interval，不依這個下界取模，因此不會改變科學狀態或 RNG。
    """

    path: Path
    sequence: int
    sweeps_completed: int
    particle_steps: int
    sweeps_source: str
    pointer_state: str
    # 這兩個值由已通過掃描的 generation 目錄普通檔案 st_size 重建；logical 是所有已發布
    # generation 的累計寫入量，active 是目前已發布 generation 加 latest pointer 的邏輯
    # 檔案長度加總。不把 active 當成 lifetime counter，避免 orphan reconcile 後統計低估。
    logical_bytes: int = 0
    active_bytes: int = 0


def _canonical_bytes(value: Any) -> bytes:
    """以禁止 NaN 的 canonical JSON 固定 run hash 的欄位順序與型別。"""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_hash(value: Any) -> str:
    """回傳 JSON 語意的 SHA-256。"""

    return sha256(_canonical_bytes(value)).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    """以 UTF-8、固定 key 順序、禁止 NaN 寫入 JSON。"""

    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _atomic_json(path: Path, value: Any) -> None:
    """在同一 parent 以 temporary file + ``os.replace`` 原子更新 mutable JSON。"""

    if path.is_symlink():
        raise ValueError(f"JSON 目標不允許 symlink：{path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.partial-{uuid4().hex}"
    try:
        _write_json(temporary, value)
        os.replace(temporary, path)
    except BaseException:
        # cleanup 也可能遇到 NFS 暫時 I/O 錯誤；只能 best-effort 嘗試，不能讓 unlink
        # 的新例外覆蓋原本的 KeyboardInterrupt／發布錯誤。若暫存檔仍殘留，後續 scanner
        # 會維持 fail-closed，操作人員須保存證據後依 runbook 處理，不能把它當成功資料。
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise


def _checkpoint_storage_metrics(parent: Path) -> tuple[int, int]:
    """由 checkpoint parent 的已發布普通檔案重建 logical 與 active 位元組數。

    schema 3 不刪除仍被 chain 參照的 generation，因此所有固定 generation payload 的普通檔案
    ``st_size`` 可直接相加作為 lifetime logical bytes-written 的可證據值；active logical
    file bytes 再加上 ``latest.json`` pointer。這不是 NFS 實際配置空間，不包含目錄 block、
    block rounding、metadata、replication、snapshot 或 partial／暫存檔，不能取代 ``du``、
    ``df`` 與 SERVER 儲存閘門。此 helper 只在掃描／reconcile 或沒有既有 gauge 的維護路徑
    使用，不經浮點運算，也不把未知項目靜默算入結果。
    """

    logical_bytes = 0
    active_bytes = 0
    if not parent.exists():
        return 0, 0
    for entry in parent.iterdir():
        if entry.name == "latest.json":
            if entry.is_file() and not entry.is_symlink():
                active_bytes += int(entry.stat().st_size)
            continue
        if _CHECKPOINT_RE.fullmatch(entry.name) is None or not entry.is_dir():
            continue
        generation_bytes = sum(
            int(path.stat().st_size)
            for path in entry.iterdir()
            if path.is_file() and not path.is_symlink()
        )
        logical_bytes += generation_bytes
        active_bytes += generation_bytes
    return int(logical_bytes), int(active_bytes)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    """嚴格讀取 JSON object，拒絕 symlink、缺檔與非有限 JSON 常數。"""

    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} 必須是存在的普通檔案")

    def reject_constant(value: str) -> None:
        """讓 ``NaN``／Infinity 不進入 run metadata。"""

        raise ValueError(f"{label} 含非有限 JSON 常數：{value}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle, parse_constant=reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} JSON 無法讀取") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} root 必須是 object")
    _assert_json_safe(value, label=label)
    return value


def _assert_json_safe(value: Any, *, label: str) -> None:
    """遞迴拒絕非有限浮點數，包含 JSON parser 由極大指數產生的 infinity。"""

    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} 含非有限浮點數")
    if isinstance(value, dict):
        for item in value.values():
            _assert_json_safe(item, label=label)
    elif isinstance(value, list):
        for item in value:
            _assert_json_safe(item, label=label)


def _safe_slug(value: str, *, label: str) -> str:
    """驗證 run/shard identifier 只能是單一安全路徑元件。"""

    if not isinstance(value, str) or _SLUG_RE.fullmatch(value) is None or value in {".", ".."}:
        raise ValueError(f"{label} 必須是安全 slug，不可含 traversal 或 path separator")
    if "/" in value or "\\" in value:
        raise ValueError(f"{label} 不可含 path separator")
    return value


def _nonempty_string(value: Any, *, label: str) -> str:
    """驗證 JSON 欄位為去除前後空白後仍有內容的字串。"""

    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} 必須是非空字串")
    return value


def _nonnegative_int(value: Any, *, label: str) -> int:
    """驗證非負整數並拒絕 Python 中可冒充整數的布林值。"""

    if type(value) is not int or value < 0:
        raise ValueError(f"{label} 必須是非負整數，且不可為 bool")
    return value


def _positive_int(value: Any, *, label: str) -> int:
    """驗證正整數，供 count、interval 與 chunk 契約共用。"""

    result = _nonnegative_int(value, label=label)
    if result < 1:
        raise ValueError(f"{label} 必須是正整數")
    return result


def _utc_timestamp(value: Any, *, label: str) -> str:
    """驗證以 ``Z`` 結尾的世界協調時間（UTC）ISO-8601 timestamp。"""

    text = _nonempty_string(value, label=label)
    if not text.endswith("Z"):
        raise ValueError(f"{label} 必須以 Z 明示 UTC")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{label} 不是合法 ISO-8601 UTC timestamp") from exc
    if parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError(f"{label} 必須是 UTC")
    return text


def _relative_token(value: Any, *, label: str) -> str:
    """驗證只含安全相對元件的 portable path token，不接觸檔案系統。"""

    token = _nonempty_string(value, label=label)
    path = Path(token)
    if path.is_absolute() or "\\" in token or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} 必須是無 traversal 的相對 path token")
    return token


def _checkpoint_token(value: Any, *, run_id: str, shard_id: str, sequence: int, label: str) -> str:
    """驗證 progress checkpoint token 與 run、shard、sequence 完全一致。"""

    token = _relative_token(value, label=label)
    expected = f"{run_id}/{shard_id}/checkpoint-{sequence:08d}"
    if token != expected:
        raise ValueError(f"{label} 與 run/shard/sequence 不一致")
    return token


def _validate_metrics(value: Any, *, label: str, allow_empty: bool = True) -> dict[str, Any]:
    """驗證工程 metrics 為 JSON-safe、非負且具固定必要欄位。

    metrics 只用於 SERVER 資源規劃，不參與粒子物理。初始 PLANNED row 可使用空 object；
    一旦執行便必須包含 wall time、process CPU、最大常駐記憶體、輸出／checkpoint bytes 與
    particle steps。forcing cache 與 crash-window sweep 下界旗標可作為額外欄位保留；新
    cache 欄位若存在，會另外驗證 counter/gauge 的非負整數與明示的語意版本。缺少語意
    版本的歷史 progress 仍可讀取，但只能由報告層標為 legacy/unknown。
    """

    if not isinstance(value, dict):
        raise ValueError(f"{label} 必須是 object")
    if not value and allow_empty:
        return value
    missing = sorted(_METRIC_REQUIRED_KEYS - set(value))
    if missing:
        raise ValueError(f"{label} 缺少必要欄位：{','.join(missing)}")
    float_keys = ("wall_seconds", "process_cpu_seconds")
    integer_keys = ("max_rss_bytes", "output_bytes", "checkpoint_bytes", "particle_steps")
    for key in float_keys:
        item = value[key]
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            raise ValueError(f"{label}.{key} 必須是有限非負數")
        if float(item) < 0:
            raise ValueError(f"{label}.{key} 不可為負")
    for key in integer_keys:
        _nonnegative_int(value[key], label=f"{label}.{key}")
    if "checkpoint_active_bytes" in value:
        # lifetime checkpoint_bytes 是累積 logical bytes-written；active 是目前已發布普通
        # 檔案 st_size 的邏輯長度 gauge，兩者必須分開保存，且一律使用原生 int，避免大型
        # checkpoint 在 float 中失去精度。此 gauge 不能代替 NFS du/df 的實際配置空間。
        _nonnegative_int(value["checkpoint_active_bytes"], label=f"{label}.checkpoint_active_bytes")
    if "forcing_cache_stats" in value:
        semantics = value.get(_FORCING_CACHE_STATS_SEMANTICS_KEY)
        if semantics is not None and semantics not in {
            _FORCING_CACHE_STATS_SEMANTICS_V1,
            _FORCING_CACHE_STATS_LEGACY,
        }:
            raise ValueError(f"{label}.{_FORCING_CACHE_STATS_SEMANTICS_KEY} 不支援")
        if semantics == _FORCING_CACHE_STATS_SEMANTICS_V1:
            # 只有新版本語意才要求固定六欄；未標記的歷史 object 刻意保留原樣，讓舊
            # progress 可以讀取並在 benchmark report 中被標成 legacy/unknown。
            _normalize_forcing_cache_stats(
                value["forcing_cache_stats"],
                label=f"{label}.forcing_cache_stats",
            )
        elif not isinstance(value["forcing_cache_stats"], dict):
            raise ValueError(f"{label}.forcing_cache_stats 必須是 object")
    elif _FORCING_CACHE_STATS_SEMANTICS_KEY in value:
        raise ValueError(f"{label}.{_FORCING_CACHE_STATS_SEMANTICS_KEY} 缺少 forcing_cache_stats")
    if _FORCING_CACHE_STATS_STATUS_KEY in value:
        status = value[_FORCING_CACHE_STATS_STATUS_KEY]
        if type(status) is not str or status != _FORCING_CACHE_STATS_STATUS_UNAVAILABLE:
            raise ValueError(f"{label}.{_FORCING_CACHE_STATS_STATUS_KEY} 不支援")
    if "sweeps_recovered_lower_bound" in value and type(value["sweeps_recovered_lower_bound"]) is not bool:
        raise ValueError(f"{label}.sweeps_recovered_lower_bound 必須是 bool")
    return value


def _derived_run_lifecycle(shards: Mapping[str, Mapping[str, Any]]) -> str:
    """由所有 shard 狀態唯一推導 run lifecycle，避免 mutable 欄位彼此矛盾。

    RUNNING 優先代表仍有工作正在執行；其後依序是 FAILED 與 PAUSED。全部完成才可標為
    COMPLETE，全部未啟動才是 PLANNED；依 plan 順序執行時常見的 COMPLETE+PLANNED
    則代表 run 已啟動但尚未全部完成，因此歸為 RUNNING。此欄位只描述工程狀態，不改變
    粒子物理、seed 或 checkpoint 內容。
    """

    lifecycles = [row["lifecycle"] for row in shards.values()]
    if lifecycles and all(lifecycle == "COMPLETE" for lifecycle in lifecycles):
        return "COMPLETE"
    if "RUNNING" in lifecycles:
        return "RUNNING"
    if "FAILED" in lifecycles:
        return "FAILED"
    if "PAUSED" in lifecycles:
        return "PAUSED"
    if lifecycles and all(lifecycle == "PLANNED" for lifecycle in lifecycles):
        return "PLANNED"
    return "RUNNING"


def _workspace_token(root: Path, token: str, *, label: str, directory: bool = False) -> Path:
    """把 progress 中的相對 token 安全解析到 workspace 內。

    progress 是可變檔案，不能把其中的絕對路徑直接交給 validator；逐層拒絕 ``..`` 與
    symlink 可避免 progress 被竄改後讀取 workspace 以外的檔案。``directory=True``
    供已發布 shard 目錄使用。
    """

    if not isinstance(token, str) or not token or Path(token).is_absolute():
        raise ValueError(f"{label} 必須是非空相對路徑")
    path = Path(token)
    if any(part in {"", ".", ".."} for part in path.parts) or "\\" in token:
        raise ValueError(f"{label} 含不安全 path token")
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} 不允許 symlink")
    if directory:
        if not current.is_dir():
            raise ValueError(f"{label} 必須指向普通目錄")
    elif not current.is_file():
        raise ValueError(f"{label} 必須指向普通檔案")
    return current


def _hash_mapping(value: Mapping[str, str], *, label: str) -> dict[str, str]:
    """複製並驗證 component／geometry canonical hash mapping。"""

    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} 必須是非空 mapping")
    result: dict[str, str] = {}
    for key, digest in value.items():
        if type(key) is not str:
            raise ValueError(f"{label}.key 必須是字串")
        clean_key = _safe_slug(key, label=f"{label}.key")
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise ValueError(f"{label}.{clean_key} 必須是 64 位小寫 SHA-256")
        result[clean_key] = digest
    return dict(sorted(result.items()))


def checkpoint_input_binding_hash(
    *,
    raw_input_inventory_sha256: str,
    component_canonical_hashes: Mapping[str, str],
    geometry_canonical_hashes: Mapping[str, str],
    provenance: CodeProvenance | Mapping[str, Any],
    random_stream_id: str | None = None,
) -> str:
    """計算 checkpoint binding 的 composite input hash。

    composite 內同時放 raw input inventory、三份情境 component／三份 geometry canonical
    hash、deployment tree 與 ``uv.lock`` hash；因此只改 manifest 語意、部署 Python、或
    依賴 lock 都會使舊 checkpoint 無法被新 run restore。raw inventory SHA 仍另存在
    run plan，方便追溯原始檔案，不以 composite 取代它。明示 ``random_stream_id`` 時，
    也把共同亂數流識別碼納入 hash；因此即使有人只修改 immutable plan 的 seed 命名空間，
    舊 checkpoint 仍不能被錯誤地 restore。``None`` 刻意不寫入 payload，保留舊 run 的
    composite hash 與 checkpoint binding 完全相容。
    """

    if (
        not isinstance(raw_input_inventory_sha256, str)
        or _SHA256_RE.fullmatch(raw_input_inventory_sha256) is None
    ):
        raise ValueError("raw_input_inventory_sha256 必須是 64 位小寫 SHA-256")
    provenance_dict = provenance.to_dict() if isinstance(provenance, CodeProvenance) else dict(provenance)
    for key in ("deployment_tree_sha256", "uv_lock_sha256"):
        if _SHA256_RE.fullmatch(str(provenance_dict.get(key, ""))) is None:
            raise ValueError(f"provenance.{key} 必須是 64 位小寫 SHA-256")
    payload = {
        "raw_input_inventory_sha256": raw_input_inventory_sha256,
        "component_canonical_hashes": _hash_mapping(component_canonical_hashes, label="component"),
        "geometry_canonical_hashes": _hash_mapping(geometry_canonical_hashes, label="geometry"),
        "deployment_tree_sha256": provenance_dict["deployment_tree_sha256"],
        "uv_lock_sha256": provenance_dict["uv_lock_sha256"],
    }
    stream_id = validate_random_stream_id(random_stream_id)
    if stream_id is not None:
        payload["random_stream_id"] = stream_id
    return _canonical_hash(payload)


def _provenance_dict(provenance: CodeProvenance | Mapping[str, Any]) -> dict[str, Any]:
    """將 provenance snapshot 正規化並驗證必要欄位與來源旗標的一致性。

    ``git_available``、``git_commit``、``git_dirty`` 與 ``commit_source`` 是同一個
    部署事實的四種表示，不可讓測試 fixture 或手寫 JSON 任意拼接。存在 Git metadata
    時必須使用 repository HEAD、``git_repository`` 與真正的布林 dirty 狀態；沒有 Git
    但由 release record 宣告 commit 時，dirty 必須保持 ``None``（未知，不等於乾淨），
    來源固定為 ``declared_deployment``；沒有 Git 也沒有宣告 commit 的 pilot 才能使用
    ``no_git_pilot``。這個 gate 讓 formal plan 可以攜帶可稽核的無 Git 部署，同時不把
    無法判定的工作樹狀態誤寫成 ``false``。
    """

    value = provenance.to_dict() if isinstance(provenance, CodeProvenance) else dict(provenance)
    required = {
        "git_available",
        "git_commit",
        "git_dirty",
        "commit_source",
        "deployment_tree_sha256",
        "deployment_file_count",
        "uv_lock_sha256",
        "python_version",
        "platform",
        "package_version",
        "numpy_version",
        "numba_version",
        "pyarrow_version",
    }
    if set(value) != required:
        raise ValueError(
            "provenance 欄位集合不符："
            f"missing={sorted(required - set(value))}, unknown={sorted(set(value) - required)}"
        )
    if type(value["git_available"]) is not bool or not isinstance(
        value["git_dirty"], (bool, type(None))
    ):
        raise ValueError("provenance git flags 型別錯誤")
    if value["git_available"] and value["git_dirty"] is None:
        raise ValueError("有 Git metadata 時 git_dirty 不可為 None")
    if not value["git_available"] and value["git_dirty"] is not None:
        raise ValueError("無 Git metadata 時 git_dirty 必須為 None")
    if value["git_commit"] is not None and (
        type(value["git_commit"]) is not str
        or _COMMIT_RE.fullmatch(value["git_commit"]) is None
    ):
        raise ValueError("provenance.git_commit 必須是 40 位小寫 commit 或 None")
    commit_source = value["commit_source"]
    if value["git_available"]:
        if (
            value["git_commit"] is None
            or commit_source != "git_repository"
            or type(value["git_dirty"]) is not bool
        ):
            raise ValueError(
                "有 Git metadata 時 commit_source 必須是 git_repository、commit 不可為 None、"
                "git_dirty 必須是 bool"
            )
    elif value["git_commit"] is not None:
        if commit_source != "declared_deployment":
            raise ValueError(
                "無 Git 但有 declared commit 時 commit_source 必須是 declared_deployment"
            )
    elif commit_source != "no_git_pilot":
        raise ValueError("無 Git 且無 commit 時 commit_source 必須是 no_git_pilot")
    if (
        _SHA256_RE.fullmatch(str(value["deployment_tree_sha256"])) is None
        or _SHA256_RE.fullmatch(str(value["uv_lock_sha256"])) is None
    ):
        raise ValueError("provenance deployment/lock hash 格式錯誤")
    _positive_int(value["deployment_file_count"], label="provenance.deployment_file_count")
    for key in (
        "commit_source",
        "python_version",
        "platform",
        "package_version",
        "numpy_version",
        "numba_version",
        "pyarrow_version",
    ):
        _nonempty_string(value[key], label=f"provenance.{key}")
    return deepcopy(value)


def _formal_provenance_is_valid(value: Mapping[str, Any]) -> bool:
    """判定 formal plan 可接受的兩種 commit 來源，不把 dirty unknown 當作 clean。

    有 Git 的正式部署必須是 repository clean snapshot；無 Git 的正式部署則必須有
    release record 宣告的 40 位 commit，且 ``git_dirty=None`` 明確表示部署端沒有工作樹
    可供判定。兩者都已由 ``_provenance_dict`` 驗證來源／旗標配對。
    """

    if value["git_available"]:
        return value["git_commit"] is not None and value["git_dirty"] is False
    return value["git_commit"] is not None and value["git_dirty"] is None


def _scenario_row(scenario: Scenario) -> dict[str, Any]:
    """將 immutable Scenario 轉成固定欄位順序的 Parquet row。"""

    return {
        "scenario_id": scenario.scenario_id,
        "study_site_id": scenario.study_site_id,
        "analysis_region_id": scenario.analysis_region_id,
        "material_id": scenario.material_id,
        "receptor_id": scenario.receptor_id,
        "arrival_time_id": scenario.arrival_time_id,
        "settling_velocity_mps": float(scenario.settling_velocity_mps),
        "arrival_time_utc_ns": int(scenario.arrival_time_utc_ns),
        "design_version": scenario.design_version,
    }


def _scenario_hash(scenarios: Sequence[Scenario]) -> str:
    """計算 shard 內 scenario rows 的順序敏感 canonical hash。"""

    return _canonical_hash([_scenario_row(item) for item in scenarios])


def _scenario_from_row(row: Mapping[str, Any], *, label: str) -> Scenario:
    """嚴格把 scenario parquet row 還原為 domain dataclass。"""

    if set(row) != set(_SCENARIO_COLUMNS):
        raise ValueError(f"{label} 欄位集合不符")
    for key in _SCENARIO_COLUMNS:
        if key in {"settling_velocity_mps", "arrival_time_utc_ns"}:
            continue
        if not isinstance(row[key], str) or not row[key].strip():
            raise ValueError(f"{label}.{key} 必須是非空字串")
    velocity = row["settling_velocity_mps"]
    arrival_ns = row["arrival_time_utc_ns"]
    if isinstance(velocity, bool) or not isinstance(velocity, (float, int)) or not np.isfinite(velocity):
        raise ValueError(f"{label}.settling_velocity_mps 必須是有限數值")
    if isinstance(arrival_ns, bool) or not isinstance(arrival_ns, (int, np.integer)):
        raise ValueError(f"{label}.arrival_time_utc_ns 必須是整數")
    result = Scenario(
        scenario_id=row["scenario_id"],
        study_site_id=row["study_site_id"],
        analysis_region_id=row["analysis_region_id"],
        material_id=row["material_id"],
        receptor_id=row["receptor_id"],
        arrival_time_id=row["arrival_time_id"],
        settling_velocity_mps=float(velocity),
        arrival_time_utc_ns=int(arrival_ns),
        design_version=row["design_version"],
    )
    expected_id = stable_identifier(
        "scn",
        [
            result.study_site_id,
            result.material_id,
            result.receptor_id,
            result.arrival_time_id,
            result.design_version,
        ],
    )
    if result.scenario_id != expected_id:
        raise ValueError(f"{label}.scenario_id stable hash 不符")
    return result


def _inventory_bytes(inventory_file: str | Path | Mapping[str, Any]) -> tuple[bytes, dict[str, Any]]:
    """讀取 inventory 原始 bytes 與 JSON payload；mapping 以 canonical bytes 代表。"""

    if isinstance(inventory_file, Mapping):
        payload = deepcopy(dict(inventory_file))
        return _canonical_bytes(payload), payload
    path = Path(inventory_file)
    if path.is_symlink() or not path.is_file():
        raise ValueError("input inventory 必須是存在的普通檔案")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("input inventory 必須是 UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("input inventory root 必須是 object")
    _canonical_bytes(payload)
    return raw, payload


def _file_record(path: Path, *, row_count: int | None = None) -> dict[str, Any]:
    """建立 run file 的 size/checksum/optional row count contract。"""

    value: dict[str, Any] = {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
    if row_count is not None:
        value["row_count"] = row_count
    return value


def _write_parquet_rows(
    path: Path, schema: pa.Schema, rows: Sequence[dict[str, Any]], *, batch_size: int = 8192
) -> int:
    """以小批次 ParquetWriter 寫入 rows，避免把 seed table 擴成一次性 Arrow table。"""

    count = 0
    writer = pq.ParquetWriter(path, schema)
    try:
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            table = pa.Table.from_pylist(batch_rows, schema=schema)
            writer.write_table(table)
            count += len(batch_rows)
    finally:
        writer.close()
    return count


def _write_seed_table_with_seed(
    path: Path,
    shards: Sequence[ScenarioShard],
    *,
    master_seed: int,
    random_stream_id: str | None = None,
) -> int:
    """依固定 shard／scenario／member 順序串流寫入 seed table。

    未啟用共同亂數流時，欄位集合維持既有五欄，避免舊 workspace 的 schema 與 checksum
    被無意改寫。明示 stream 時，新增每列的 ``random_stream_id``，讓 seed table 自身
    就能追溯「案例身分被哪個配對命名空間取代」，而不是只依賴執行命令列或外部說明。
    兩種模式都由同一個 ``iter_run_units`` 產生 seed，確保 seed table 與實際
    ``ProductionBatch`` 使用完全相同的導出路徑。
    """

    stream_id = validate_random_stream_id(random_stream_id)
    fields = [
        pa.field("scenario_id", pa.string()),
        pa.field("experiment_case_id", pa.string()),
        pa.field("member_id", pa.int64()),
        pa.field("particle_id", pa.string()),
        pa.field("seed_128_hex", pa.string()),
    ]
    if stream_id is not None:
        fields.append(pa.field("random_stream_id", pa.string()))
    schema = pa.schema(fields)
    writer = pq.ParquetWriter(path, schema)
    pending: list[dict[str, Any]] = []
    count = 0
    try:
        for shard in shards:
            for unit in iter_run_units(
                shard,
                master_seed=master_seed,
                random_stream_id=stream_id,
            ):
                row = {
                    "scenario_id": unit.scenario.scenario_id,
                    "experiment_case_id": unit.experiment_case_id,
                    "member_id": unit.member_id,
                    "particle_id": unit.particle_id,
                    "seed_128_hex": f"{unit.seed:032x}",
                }
                if stream_id is not None:
                    row["random_stream_id"] = stream_id
                pending.append(row)
                if len(pending) >= 8192:
                    writer.write_table(pa.Table.from_pylist(pending, schema=schema))
                    count += len(pending)
                    pending.clear()
        if pending:
            writer.write_table(pa.Table.from_pylist(pending, schema=schema))
            count += len(pending)
    finally:
        writer.close()
    return count


def _create_lock_topology(root: Path, shards: Sequence[ScenarioShard]) -> None:
    """在尚未發布的 partial workspace 預建固定的零長度 lock files。

    鎖檔是 Unix ``fcntl.flock`` 的承載物，不是資料輸出；因此只要求它們是普通零長度
    檔案，且不列入四個 immutable input file checksum。先在 partial 目錄建立完整集合，
    再以同 parent 的 ``os.replace`` 發布，讓 validator 永遠看到全有或全無的 topology，
    避免正式 run 被半套鎖檔啟動。
    """

    lock_root = root / _LOCK_ROOT
    lock_root.mkdir()
    names = set(_LOCK_FIXED_NAMES) | {f"{shard.shard_id}.lock" for shard in shards}
    for name in sorted(names):
        _safe_slug(name, label="lock file name")
        path = lock_root / name
        with path.open("x", encoding="utf-8"):
            pass


def _initial_progress(run_id: str, shards: Sequence[ScenarioShard]) -> dict[str, Any]:
    """建立所有 shard 尚未啟動的 progress schema。"""

    return {
        "schema_version": RUN_PROGRESS_SCHEMA_VERSION,
        "run_id": run_id,
        "revision": 0,
        "updated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "run_lifecycle": "PLANNED",
        "shards": {
            shard.shard_id: {
                "shard_id": shard.shard_id,
                "scenario_start_index": shard.scenario_start_index,
                "scenario_stop_index": shard.scenario_stop_index,
                "lifecycle": "PLANNED",
                "checkpoint_sequence": 0,
                "checkpoint_relative_path": None,
                "output_relative_path": None,
                "attempt_count": 0,
                "sweeps_completed": 0,
                "particle_steps": 0,
                "metrics": {},
                "error_code": None,
                "failure_relative_path": None,
            }
            for shard in shards
        },
    }


def _validate_plan_shape(plan: Mapping[str, Any]) -> None:
    """嚴格檢查 immutable run plan 的型別、hash、count、range 與固定拓撲。

    schema 2.0 以舊 exact key set 唯讀相容；schema 2.1 另要求
    ``scenario_selection``，schema 2.2 再要求明示且非空的 ``random_stream_id``，並以
    selected scenario count 對齊 plan。此檢查不讀大型 Parquet payload；它先確保 JSON
    自身不可能以 ``bool`` 冒充 count、以 traversal 冒充 root，或以缺口／重疊 shard range
    改變工作分配。run file checksum 與 scenario 內容的實際比對由只讀 ``validate_run``
    接續完成。
    """

    legacy_required = {
        "schema_version",
        "run_id",
        "run_kind",
        "created_at_utc",
        "config_hash",
        "raw_input_inventory_sha256",
        "checkpoint_input_binding_hash",
        "component_canonical_hashes",
        "geometry_canonical_hashes",
        "code_provenance",
        "experiment_case_id",
        "master_seed",
        "seed_policy",
        "members_per_scenario",
        "shard_scenario_count",
        "checkpoint_interval_sweeps",
        "active_chunk_size",
        "scenario_count",
        "particle_count",
        "shard_count",
        "files",
        "shards",
        "scenario_ordering_policy",
        "lock_root",
        "output_root",
        "checkpoint_root",
        "failure_root",
    }
    if "schema_version" not in plan:
        raise ValueError("run plan 缺少欄位：schema_version")
    schema_version = plan["schema_version"]
    if schema_version == RUN_PLAN_LEGACY_SCHEMA_VERSION:
        required = legacy_required
    elif schema_version == RUN_PLAN_SCHEMA_VERSION:
        required = legacy_required | {"scenario_selection"}
    elif schema_version == RUN_PLAN_PAIRED_SCHEMA_VERSION:
        # 共同亂數流是明示的新 plan 版本；舊 2.1 plan 仍維持原 exact key set，避免
        # validator 為了補預設值而改寫既有 schema 或 seed table。
        required = legacy_required | {"scenario_selection", "random_stream_id"}
    else:
        raise ValueError("run plan schema_version 不支援")
    missing = sorted(required - set(plan))
    if missing:
        raise ValueError("run plan 缺少欄位：" + ",".join(missing))
    unknown = sorted(set(plan) - required)
    if unknown:
        raise ValueError("run plan 含未知欄位：" + ",".join(unknown))
    if plan["scenario_ordering_policy"] != SCENARIO_ORDERING_POLICY:
        raise ValueError("run_plan.scenario_ordering_policy 不支援或遭竄改")
    run_id = _safe_slug(plan["run_id"], label="run_id")
    experiment_case_id = _safe_slug(plan["experiment_case_id"], label="experiment_case_id")
    del experiment_case_id
    if type(plan["run_kind"]) is not str or plan["run_kind"] not in _RUN_KINDS:
        raise ValueError("run plan run_kind 不支援")
    _utc_timestamp(plan["created_at_utc"], label="run_plan.created_at_utc")
    for key in ("config_hash", "raw_input_inventory_sha256", "checkpoint_input_binding_hash"):
        if type(plan[key]) is not str or _SHA256_RE.fullmatch(plan[key]) is None:
            raise ValueError(f"run_plan.{key} 必須是 64 位小寫 SHA-256")
    _hash_mapping(plan["component_canonical_hashes"], label="component_canonical_hashes")
    _hash_mapping(plan["geometry_canonical_hashes"], label="geometry_canonical_hashes")
    provenance = _provenance_dict(plan["code_provenance"])
    if plan["run_kind"] == "formal" and not _formal_provenance_is_valid(provenance):
        raise ValueError(
            "formal run plan 必須是 Git clean commit，或無 Git 的 declared_deployment commit；"
            "無 Git 時 git_dirty 必須保持 None"
        )
    _nonnegative_int(plan["master_seed"], label="run_plan.master_seed")
    _safe_slug(plan["seed_policy"], label="run_plan.seed_policy")
    if schema_version == RUN_PLAN_PAIRED_SCHEMA_VERSION:
        validate_random_stream_id(plan["random_stream_id"], allow_none=False)
    members = _positive_int(plan["members_per_scenario"], label="run_plan.members_per_scenario")
    _positive_int(plan["shard_scenario_count"], label="run_plan.shard_scenario_count")
    _positive_int(plan["checkpoint_interval_sweeps"], label="run_plan.checkpoint_interval_sweeps")
    if plan["active_chunk_size"] is not None:
        _positive_int(plan["active_chunk_size"], label="run_plan.active_chunk_size")
    scenario_count = _positive_int(plan["scenario_count"], label="run_plan.scenario_count")
    particle_count = _positive_int(plan["particle_count"], label="run_plan.particle_count")
    shard_count = _positive_int(plan["shard_count"], label="run_plan.shard_count")
    if particle_count != scenario_count * members:
        raise ValueError("run plan particle_count 必須等於 scenario_count×members_per_scenario")
    if schema_version in {RUN_PLAN_SCHEMA_VERSION, RUN_PLAN_PAIRED_SCHEMA_VERSION}:
        validate_scenario_selection_binding_shape(
            plan["scenario_selection"],
            plan["run_kind"],
            scenario_count,
        )

    roots = []
    for key in ("output_root", "checkpoint_root", "failure_root"):
        roots.append(_safe_slug(plan[key], label=f"run_plan.{key}"))
    if len(set(roots)) != len(roots):
        raise ValueError("run plan output/checkpoint/failure root token 必須互異")
    if roots != ["shards", "checkpoints", "failures"]:
        raise ValueError("run plan root token 固定為 shards/checkpoints/failures")
    if _safe_slug(plan["lock_root"], label="run_plan.lock_root") != _LOCK_ROOT:
        raise ValueError("run plan lock_root 固定為 locks")

    files = plan["files"]
    if not isinstance(files, dict) or set(files) != _RUN_FILE_NAMES:
        raise ValueError("run plan files 必須是固定四個 immutable input files")
    for filename, contract in files.items():
        expected_keys = {"size_bytes", "sha256"}
        if filename.endswith(".parquet"):
            expected_keys.add("row_count")
        if not isinstance(contract, dict) or set(contract) != expected_keys:
            raise ValueError(f"run plan file contract 欄位不符：{filename}")
        _nonnegative_int(contract["size_bytes"], label=f"files.{filename}.size_bytes")
        if type(contract["sha256"]) is not str or _SHA256_RE.fullmatch(contract["sha256"]) is None:
            raise ValueError(f"files.{filename}.sha256 格式錯誤")
        if "row_count" in contract:
            _positive_int(contract["row_count"], label=f"files.{filename}.row_count")
    if files["scenario_table.parquet"]["row_count"] != scenario_count:
        raise ValueError("scenario_table row_count 與 scenario_count 不一致")
    if files["seed_table.parquet"]["row_count"] != particle_count:
        raise ValueError("seed_table row_count 與 particle_count 不一致")

    shard_rows = plan["shards"]
    if not isinstance(shard_rows, list) or len(shard_rows) != shard_count:
        raise ValueError("run plan shards 必須是與 shard_count 一致的非空 array")
    shard_keys = {
        "shard_id",
        "scenario_start_index",
        "scenario_stop_index",
        "scenario_count",
        "particle_count",
        "scenario_hash",
        "execution_group_id",
        "analysis_region_id",
        "arrival_time_utc_ns",
        "group_part_index",
        "group_part_count",
    }
    expected_start = 0
    seen_ids: set[str] = set()
    total_particles = 0
    closed_groups: set[str] = set()
    previous_group_id: str | None = None
    previous_group_pair: tuple[str, int] | None = None
    previous_part_index = -1
    previous_part_count = 0
    for index, row in enumerate(shard_rows):
        if not isinstance(row, dict) or set(row) != shard_keys:
            raise ValueError(f"run_plan.shards[{index}] 欄位集合不符")
        shard_id = _safe_slug(row["shard_id"], label=f"run_plan.shards[{index}].shard_id")
        if shard_id in seen_ids:
            raise ValueError(f"run plan shard_id 重複：{shard_id}")
        seen_ids.add(shard_id)
        start = _nonnegative_int(
            row["scenario_start_index"], label=f"run_plan.shards[{index}].scenario_start_index"
        )
        stop = _positive_int(
            row["scenario_stop_index"], label=f"run_plan.shards[{index}].scenario_stop_index"
        )
        count = _positive_int(row["scenario_count"], label=f"run_plan.shards[{index}].scenario_count")
        particles = _positive_int(row["particle_count"], label=f"run_plan.shards[{index}].particle_count")
        if start != expected_start or stop <= start or stop - start != count:
            raise ValueError(f"run plan shard range 有缺口、重疊或 count 不一致：{shard_id}")
        if particles != count * members:
            raise ValueError(f"run plan shard particle_count 不一致：{shard_id}")
        if type(row["scenario_hash"]) is not str or _SHA256_RE.fullmatch(row["scenario_hash"]) is None:
            raise ValueError(f"run plan shard scenario_hash 格式錯誤：{shard_id}")
        group_id = _safe_slug(
            row["execution_group_id"], label=f"run_plan.shards[{index}].execution_group_id"
        )
        analysis_region_id = _nonempty_string(
            row["analysis_region_id"], label=f"run_plan.shards[{index}].analysis_region_id"
        )
        arrival_time_utc_ns = row["arrival_time_utc_ns"]
        if type(arrival_time_utc_ns) is not int:
            raise ValueError(
                f"run_plan.shards[{index}].arrival_time_utc_ns 必須是非 bool 整數"
            )
        group_part_index = _nonnegative_int(
            row["group_part_index"], label=f"run_plan.shards[{index}].group_part_index"
        )
        group_part_count = _positive_int(
            row["group_part_count"], label=f"run_plan.shards[{index}].group_part_count"
        )
        if group_part_index >= group_part_count:
            raise ValueError(f"run plan group part index/count 不一致：{shard_id}")
        expected_group_id = stable_identifier(
            "grp",
            [SCENARIO_ORDERING_POLICY, analysis_region_id, str(arrival_time_utc_ns)],
            length=20,
        )
        if group_id != expected_group_id:
            raise ValueError(f"run plan execution_group_id 不符合 ordering policy：{shard_id}")
        group_pair = (analysis_region_id, arrival_time_utc_ns)
        if previous_group_id != group_id:
            if group_id in closed_groups:
                raise ValueError(f"run plan execution group 不可離開後重新出現：{group_id}")
            if previous_group_id is not None:
                if previous_part_index != previous_part_count - 1:
                    raise ValueError(f"execution group parts 未完整到達 count-1：{previous_group_id}")
                closed_groups.add(previous_group_id)
            if previous_group_pair == group_pair:
                raise ValueError("相鄰 shard group 不可共享 region+arrival")
            if group_part_index != 0:
                raise ValueError(f"execution group part 必須從 0 開始：{group_id}")
        else:
            if group_pair != previous_group_pair:
                raise ValueError(f"同一 execution group 的 region/arrival 不一致：{group_id}")
            if group_part_index != previous_part_index + 1:
                raise ValueError(f"execution group part 不連續：{group_id}")
            if group_part_count != previous_part_count:
                raise ValueError(f"execution group part_count 不一致：{group_id}")
        previous_group_id = group_id
        previous_group_pair = group_pair
        previous_part_index = group_part_index
        previous_part_count = group_part_count
        expected_start = stop
        total_particles += particles
    if expected_start != scenario_count or total_particles != particle_count:
        raise ValueError("run plan shard ranges/count 未完整覆蓋全 run")
    if previous_group_id is not None and previous_part_index != previous_part_count - 1:
        raise ValueError(f"execution group parts 未完整到達 count-1：{previous_group_id}")
    del run_id


def _snapshot_run_document(document: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    """建立與呼叫端容器完全脫鉤的普通 Python 文件快照。

    根節點必須實作映射介面（``Mapping``）；先以 ``dict`` 固定根容器，再以深拷貝遞迴
    複製其中的字典、串列與純量。這樣驗證結果不會在回傳後受到呼叫端修改巢狀容器影響，
    也不需要為了驗證來源文件而建立假的 run workspace。此步驟不轉換字串、整數或浮點數，
    亦不補任何預設欄位；無法建立快照時一律封裝成 ``ValueError``，維持公開驗證入口一致的
    拒絕邊界。
    """

    if not isinstance(document, Mapping):
        raise ValueError(f"{label} 必須是 Mapping")
    try:
        return deepcopy(dict(document))
    except Exception as exc:
        raise ValueError(f"{label} 無法建立防禦性快照") from exc


def validate_run_plan_document(document: Mapping[str, Any]) -> dict[str, Any]:
    """驗證來源 ``run_plan.json`` 文件結構並回傳不共享巢狀容器的快照。

    此公開入口供 aggregate release 等已取得文件內容的流程重用既有 run plan schema，僅驗證
    固定欄位、原生型別、雜湊、計數、分片範圍與排序拓撲。它不讀取 trajectory、checkpoint
    或 forcing 產品，也不核對 plan 宣告的檔案內容，因此不等同完整的 ``validate_run``。
    呼叫端資料不會被修改；回傳值由 ``deepcopy(dict(document))`` 建立，後續任一方修改巢狀
    字典或串列都不會影響另一方。所有非 ``Mapping``、快照失敗或 schema 錯誤皆以
    ``ValueError`` 拒絕，且不執行型別轉換或預設值填補。
    """

    snapshot = _snapshot_run_document(document, label="run plan document")
    try:
        _validate_plan_shape(snapshot)
    except ValueError:
        raise
    except Exception as exc:
        # 自訂 Mapping 或不合法巢狀值可能讓排序、索引等 Python 操作拋出其他例外；公開
        # 文件驗證 API 將它們統一視為來源文件損壞，不洩漏實作層的 TypeError/KeyError。
        raise ValueError("run plan document 結構驗證失敗") from exc
    return snapshot


def validate_run_progress_document(document: Mapping[str, Any]) -> dict[str, Any]:
    """驗證來源 ``run_progress.json`` 文件結構並回傳防禦性深拷貝快照。

    驗證範圍完整涵蓋既有 progress 的必要／未知欄位、原生數值型別、生命週期、相對路徑、
    checkpoint sequence、工程 metrics 與連續 scenario range。這只是單一來源文件的結構
    驗證：不讀取 trajectory、checkpoint 或 forcing，不與 immutable plan 交叉核對，也不
    等同完整的 ``validate_run``。回傳值由 ``deepcopy(dict(document))`` 建立，因此不保留
    呼叫端巢狀字典與串列的 alias；驗證不做字串／整數轉型、不補預設值，任何入口、複製或
    結構錯誤都以 ``ValueError`` fail closed。
    """

    progress = _snapshot_run_document(document, label="run progress document")
    try:
        _validate_run_progress_snapshot(progress)
    except ValueError:
        raise
    except Exception as exc:
        # 與 plan API 相同，來源值造成的非預期容器／比較例外仍屬文件驗證失敗。
        raise ValueError("run progress document 結構驗證失敗") from exc
    return progress


def _validate_run_progress_snapshot(progress: Mapping[str, Any]) -> None:
    """驗證已完成防禦性複製的 progress；只供公開文件驗證入口呼叫。

    ``sweeps_completed`` 代表外層對 ``ProductionBatch`` 完成或嘗試呼叫的 sweep
    次數；``particle_steps`` 則代表粒子實際成功完成的步進總數。對 COMPLETE 或
    FAILED 狀態，最後一個 sweep 可能只進行每個仍 active 粒子的步首終止判定，
    例如 forcing、邊界或最大步數條件在真正更新前就宣告 terminal，因而合法地出現
    ``particle_steps == sweeps_completed - 1``。這是一個可追溯的 terminal-only final
    sweep，不是遺漏步數；validator 只允許一個差額，避免兩個以上 sweep 被隱藏。
    PLANNED、PAUSED 與 RUNNING 尚未進入這個完成／失敗收尾語意，仍要求每個已完成
    sweep 至少包含一個成功 particle step。
    """

    required = {"schema_version", "run_id", "revision", "updated_at_utc", "run_lifecycle", "shards"}
    missing = sorted(required - set(progress))
    if missing:
        raise ValueError("run progress 缺少欄位：" + ",".join(missing))
    unknown = sorted(set(progress) - required)
    if unknown:
        raise ValueError("run progress 含未知欄位：" + ",".join(unknown))
    if progress["schema_version"] != RUN_PROGRESS_SCHEMA_VERSION:
        raise ValueError("run progress schema_version 不支援")
    _safe_slug(progress["run_id"], label="progress.run_id")
    _nonnegative_int(progress["revision"], label="progress.revision")
    _utc_timestamp(progress["updated_at_utc"], label="progress.updated_at_utc")
    if type(progress["run_lifecycle"]) is not str or progress["run_lifecycle"] not in _LIFECYCLES:
        raise ValueError("progress.run_lifecycle 不合法")
    if not isinstance(progress["shards"], dict) or not progress["shards"]:
        raise ValueError("progress.shards 必須是 object")
    shard_required = {
        "shard_id",
        "scenario_start_index",
        "scenario_stop_index",
        "lifecycle",
        "checkpoint_sequence",
        "checkpoint_relative_path",
        "output_relative_path",
        "attempt_count",
        "sweeps_completed",
        "particle_steps",
        "metrics",
        "error_code",
    }
    shard_required.add("failure_relative_path")
    for shard_id, row in progress["shards"].items():
        if not isinstance(shard_id, str) or not isinstance(row, dict):
            raise ValueError("progress shard key/row 型別錯誤")
        _safe_slug(shard_id, label="progress.shard_id")
        if set(row) != shard_required:
            raise ValueError(f"progress shard 欄位不符：{shard_id}")
        if (
            row["shard_id"] != shard_id
            or type(row["lifecycle"]) is not str
            or row["lifecycle"] not in _LIFECYCLES
        ):
            raise ValueError(f"progress shard identity/lifecycle 不符：{shard_id}")
        start = _nonnegative_int(
            row["scenario_start_index"], label=f"progress[{shard_id}].scenario_start_index"
        )
        stop = _positive_int(row["scenario_stop_index"], label=f"progress[{shard_id}].scenario_stop_index")
        if stop <= start:
            raise ValueError(f"progress[{shard_id}] scenario range 無效")
        sequence = _nonnegative_int(
            row["checkpoint_sequence"], label=f"progress[{shard_id}].checkpoint_sequence"
        )
        attempts = _nonnegative_int(
            row["attempt_count"], label=f"progress[{shard_id}].attempt_count"
        )
        sweeps = _nonnegative_int(row["sweeps_completed"], label=f"progress[{shard_id}].sweeps_completed")
        particle_steps = _nonnegative_int(row["particle_steps"], label=f"progress[{shard_id}].particle_steps")
        lifecycle = row["lifecycle"]
        if lifecycle in {"COMPLETE", "FAILED"}:
            # 完成／失敗收尾可能包含一個只做步首 terminal 判定的 sweep；只放寬一個
            # 差額，且不修正文件中的 counter，讓 validator 仍能揭露更大的不一致。
            if particle_steps + 1 < sweeps:
                raise ValueError(
                    f"progress[{shard_id}] particle_steps 與 sweeps_completed 差距超過一個"
                    " terminal-only sweep"
                )
        elif particle_steps < sweeps:
            raise ValueError(f"progress[{shard_id}] particle_steps 不可小於 sweeps_completed")
        metrics = _validate_metrics(
            row["metrics"],
            label=f"progress[{shard_id}].metrics",
            allow_empty=row["lifecycle"] in {"PLANNED", "RUNNING"},
        )
        checkpoint_path = row["checkpoint_relative_path"]
        if sequence == 0:
            if checkpoint_path is not None:
                raise ValueError(f"progress[{shard_id}] sequence=0 時 checkpoint path 必須為 None")
        else:
            _checkpoint_token(
                checkpoint_path,
                run_id=progress["run_id"],
                shard_id=shard_id,
                sequence=sequence,
                label=f"progress[{shard_id}].checkpoint_relative_path",
            )
        if lifecycle != "PLANNED" and attempts < 1:
            raise ValueError(f"progress[{shard_id}] 非 PLANNED lifecycle 必須有 attempt_count")
        output_path = row["output_relative_path"]
        failure_path = row["failure_relative_path"]
        error_code = row["error_code"]
        if lifecycle == "PLANNED":
            if any((sequence, row["attempt_count"], sweeps, particle_steps)) or metrics:
                raise ValueError(f"progress[{shard_id}] PLANNED 不可已有執行狀態")
            if output_path is not None or failure_path is not None or error_code is not None:
                raise ValueError(f"progress[{shard_id}] PLANNED path/error invariant 不符")
        elif lifecycle == "PAUSED":
            if sequence < 1 or output_path is not None or failure_path is not None or error_code is not None:
                raise ValueError(f"progress[{shard_id}] PAUSED 必須綁定 checkpoint 且無 output/failure")
        elif lifecycle == "FAILED":
            if output_path is not None or type(error_code) is not str or not error_code:
                raise ValueError(f"progress[{shard_id}] FAILED 必須有 error_code 且無 output")
            token = _relative_token(failure_path, label=f"progress[{shard_id}].failure_relative_path")
            expected_prefix = f"failures/{shard_id}/"
            if not token.startswith(expected_prefix) or _FAILURE_RE.fullmatch(Path(token).name) is None:
                raise ValueError(f"progress[{shard_id}] failure path 不合法")
        elif lifecycle == "COMPLETE":
            token = _relative_token(output_path, label=f"progress[{shard_id}].output_relative_path")
            if token != f"shards/{shard_id}" or failure_path is not None or error_code is not None:
                raise ValueError(f"progress[{shard_id}] COMPLETE output/failure invariant 不符")
        else:
            if output_path is not None or failure_path is not None or error_code is not None:
                raise ValueError(f"progress[{shard_id}] RUNNING 不可宣告 output/failure/error")
    # progress 重複保存 shard range 供 operator 不開 scenario table 即可監看；即使在尚未
    # 交叉比對 plan 前，也先要求範圍從 0 連續覆蓋，避免以兩個互相重疊的 progress row
    # 讓後續 checkpoint/output token 指向不明確的 particle subset。真正的 stop/count 與
    # immutable plan 對齊仍由 _cross_check_plan_progress 再驗一次。
    expected_start = 0
    for shard_id, row in sorted(
        progress["shards"].items(), key=lambda item: item[1]["scenario_start_index"]
    ):
        if row["scenario_start_index"] != expected_start:
            raise ValueError(f"progress shard range 有缺口或重疊：{shard_id}")
        expected_start = row["scenario_stop_index"]
    expected_run_lifecycle = _derived_run_lifecycle(progress["shards"])
    if progress["run_lifecycle"] != expected_run_lifecycle:
        raise ValueError(
            "run lifecycle 與 shard lifecycle 不一致："
            f"actual={progress['run_lifecycle']}, expected={expected_run_lifecycle}"
        )


def load_run_plan(workspace: str | Path) -> dict[str, Any]:
    """嚴格載入 immutable ``run_plan.json`` 並驗證來源文件結構。"""

    root = Path(workspace)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("run workspace 必須是非 symlink 目錄")
    plan = _read_json(root / "run_plan.json", label="run_plan")
    return validate_run_plan_document(plan)


def load_run_progress(workspace: str | Path) -> dict[str, Any]:
    """嚴格載入 mutable ``run_progress.json`` 並檢查 lifecycle/path invariants。

    檔案讀取仍由既有嚴格 JSON reader 處理；內容交給
    ``validate_run_progress_document``。此層只驗 JSON 自身；checkpoint generation 是否真的
    存在、range 是否與 immutable plan 相符，會由 controller 或 ``validate_run`` 進一步
    交叉核對。這項分層讓錯誤能清楚區分「progress schema 已損壞」與「external checkpoint
    root 指錯」。
    """

    root = Path(workspace)
    progress = _read_json(root / "run_progress.json", label="run_progress")
    return validate_run_progress_document(progress)


def _cross_check_plan_progress(plan: Mapping[str, Any], progress: Mapping[str, Any]) -> None:
    """核對 mutable progress 不可改寫 immutable plan 的 run、shard 或 range。

    progress 故意重複保存 range，讓 operator 不開 Parquet 也能監看進度；代價是每次
    controller 啟動與 run validation 都必須逐 shard 交叉核對，否則被竄改的 range 可能
    讓 checkpoint/output token 指向另一批粒子。
    """

    if progress["run_id"] != plan["run_id"]:
        raise ValueError("run plan/progress run_id 不一致")
    plan_rows = {row["shard_id"]: row for row in plan["shards"]}
    progress_rows = progress["shards"]
    if set(progress_rows) != set(plan_rows):
        raise ValueError("run plan/progress shard ID 集合不一致")
    for shard_id, plan_row in plan_rows.items():
        row = progress_rows[shard_id]
        if (
            row["scenario_start_index"] != plan_row["scenario_start_index"]
            or row["scenario_stop_index"] != plan_row["scenario_stop_index"]
        ):
            raise ValueError(f"run plan/progress shard range 不一致：{shard_id}")
        if row["output_relative_path"] is not None:
            expected_output = f"{plan['output_root']}/{shard_id}"
            if row["output_relative_path"] != expected_output:
                raise ValueError(f"progress output token 與 plan 不一致：{shard_id}")
        if row["failure_relative_path"] is not None:
            token = _relative_token(
                row["failure_relative_path"], label=f"progress[{shard_id}].failure_relative_path"
            )
            if not token.startswith(f"{plan['failure_root']}/{shard_id}/"):
                raise ValueError(f"progress failure token 與 plan 不一致：{shard_id}")


def initialize_run_workspace(
    destination: str | Path,
    *,
    run_id: str,
    scenarios: Sequence[Scenario],
    normalized_config: Mapping[str, Any],
    config_hash: str,
    input_inventory_file: str | Path | Mapping[str, Any],
    component_canonical_hashes: Mapping[str, str],
    geometry_canonical_hashes: Mapping[str, str],
    provenance: CodeProvenance | Mapping[str, Any],
    experiment_case_id: str,
    master_seed: int,
    seed_policy: str,
    random_stream_id: str | None = None,
    members_per_scenario: int,
    shard_scenario_count: int,
    checkpoint_interval_sweeps: int,
    active_chunk_size: int | None = None,
    run_kind: str = "formal",
    scenario_selection: Mapping[str, Any] | None = None,
) -> RunWorkspace:
    """建立並原子發布一個全新的 run workspace。

    ``scenarios`` 代表本次 plan 要執行的 selected scenario tuple，先依 runner 的版本化
    execution ordering policy 排序，再按 region+arrival 群組切割；scenario table 與 seed
    table 都保存這個順序。``random_stream_id`` 若明示，會以 schema 2.2 將配對命名空間
    同時寫入 immutable plan 與 seed table，並加入 checkpoint input binding；若省略，則
    發布原 schema 2.1、五欄 seed table 與舊 seed 導出規則。``scenario_selection`` 若
    省略，writer 會建立 source/selected 相等的 full binding；若提供 pilot stratified
    binding，則只驗證其 shape、selected count 與 selected ID hash 是否吻合傳入 tuple，
    完整 source 的 re-selection 必須已在 runtime initializer 或 static loader 完成。seed
    table 由 ParquetWriter 以 8192 rows
    分批寫入，
    每個由主種子、scenario、experiment、member 導出的 128-bit seed 以 32 位小寫 hex
    保存。``input_inventory_file`` 若是 Path 會保留原始 bytes 的 SHA-256；若是 mapping
    則以 canonical JSON bytes 代表。所有 temporary files 和 run directory 位於同一 parent，
    最後才用 ``os.replace`` 發布；既有 run_id（含 symlink）一律拒絕。
    """

    root_parent = Path(destination)
    _safe_slug(run_id, label="run_id")
    _safe_slug(experiment_case_id, label="experiment_case_id")
    stream_id = validate_random_stream_id(random_stream_id)
    if not isinstance(normalized_config, Mapping) or not normalized_config:
        raise ValueError("normalized_config 必須是非空 mapping")
    if type(config_hash) is not str or _SHA256_RE.fullmatch(config_hash) is None:
        raise ValueError("config_hash 必須是 64 位小寫 SHA-256")
    if isinstance(master_seed, bool) or not isinstance(master_seed, int) or master_seed < 0:
        raise ValueError("master_seed 必須是非負整數")
    integer_args = {
        "members_per_scenario": members_per_scenario,
        "shard_scenario_count": shard_scenario_count,
        "checkpoint_interval_sweeps": checkpoint_interval_sweeps,
    }
    for label, value in integer_args.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{label} 必須是正整數")
    if active_chunk_size is not None and (
        isinstance(active_chunk_size, bool) or not isinstance(active_chunk_size, int) or active_chunk_size < 1
    ):
        raise ValueError("active_chunk_size 必須是正整數或 None")
    if type(run_kind) is not str or run_kind not in _RUN_KINDS:
        raise ValueError("run_kind 必須是 formal、pilot 或 synthetic")
    if not isinstance(seed_policy, str) or not seed_policy.strip():
        raise ValueError("seed_policy 不可空白")
    scenario_values = tuple(scenarios)
    if scenario_selection is None:
        selection_binding = build_full_scenario_selection(scenario_values)
    else:
        if not isinstance(scenario_selection, Mapping):
            raise ValueError("scenario_selection 必須是 Mapping")
        selection_binding = deepcopy(dict(scenario_selection))
        validate_scenario_selection_binding_shape(
            selection_binding,
            run_kind,
            len(scenario_values),
        )
        if selection_binding["selected_scenario_count"] != len(scenario_values):
            raise ValueError("scenario_selection selected count 與傳入 scenarios 不一致")
        if selection_binding["selected_scenario_ids_sha256"] != scenario_ids_sha256(
            scenario_values
        ):
            raise ValueError("scenario_selection selected scenario hash 與傳入 scenarios 不一致")
    shards = tuple(
        plan_scenario_shards(
            scenario_values,
            members_per_scenario=members_per_scenario,
            shard_scenario_count=shard_scenario_count,
            experiment_case_id=experiment_case_id,
        )
    )
    provenance_value = _provenance_dict(provenance)
    if run_kind == "formal" and not _formal_provenance_is_valid(provenance_value):
        raise ValueError(
            "formal run 必須是 Git clean commit，或無 Git 的 declared_deployment commit；"
            "無 Git 時 git_dirty 必須保持 None"
        )
    component_hashes = _hash_mapping(component_canonical_hashes, label="component_canonical_hashes")
    geometry_hashes = _hash_mapping(geometry_canonical_hashes, label="geometry_canonical_hashes")
    raw_inventory, inventory_payload = _inventory_bytes(input_inventory_file)
    raw_inventory_hash = sha256(raw_inventory).hexdigest()
    binding_hash = checkpoint_input_binding_hash(
        raw_input_inventory_sha256=raw_inventory_hash,
        component_canonical_hashes=component_hashes,
        geometry_canonical_hashes=geometry_hashes,
        provenance=provenance_value,
        random_stream_id=stream_id,
    )
    target = root_parent / run_id
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"run workspace 已存在：{target.name}")
    root_parent.mkdir(parents=True, exist_ok=True)
    partial = root_parent / f".{run_id}.partial-{uuid4().hex}"
    partial.mkdir()
    try:
        normalized_config_path = partial / "normalized_config.json"
        inventory_path = partial / "input_inventory.json"
        scenario_path = partial / "scenario_table.parquet"
        seed_path = partial / "seed_table.parquet"
        _write_json(normalized_config_path, deepcopy(dict(normalized_config)))
        _write_json(inventory_path, inventory_payload)
        scenario_schema = pa.schema(
            [
                pa.field("scenario_id", pa.string()),
                pa.field("study_site_id", pa.string()),
                pa.field("analysis_region_id", pa.string()),
                pa.field("material_id", pa.string()),
                pa.field("receptor_id", pa.string()),
                pa.field("arrival_time_id", pa.string()),
                pa.field("settling_velocity_mps", pa.float64()),
                pa.field("arrival_time_utc_ns", pa.int64()),
                pa.field("design_version", pa.string()),
            ]
        )
        ordered_scenarios = tuple(item for shard in shards for item in shard.scenarios)
        scenario_rows = [_scenario_row(item) for item in ordered_scenarios]
        _write_parquet_rows(scenario_path, scenario_schema, scenario_rows)
        seed_count = _write_seed_table_with_seed(
            seed_path,
            shards,
            master_seed=master_seed,
            random_stream_id=stream_id,
        )
        files = {
            "normalized_config.json": _file_record(normalized_config_path),
            "input_inventory.json": _file_record(inventory_path),
            "scenario_table.parquet": _file_record(scenario_path, row_count=len(ordered_scenarios)),
            "seed_table.parquet": _file_record(seed_path, row_count=seed_count),
        }
        shard_rows = [
            {
                "shard_id": shard.shard_id,
                "scenario_start_index": shard.scenario_start_index,
                "scenario_stop_index": shard.scenario_stop_index,
                "scenario_count": shard.scenario_count,
                "particle_count": shard.particle_count,
                "scenario_hash": _scenario_hash(shard.scenarios),
                "execution_group_id": shard.execution_group_id,
                "analysis_region_id": shard.analysis_region_id,
                "arrival_time_utc_ns": shard.arrival_time_utc_ns,
                "group_part_index": shard.group_part_index,
                "group_part_count": shard.group_part_count,
            }
            for shard in shards
        ]
        plan = {
            "schema_version": (
                RUN_PLAN_SCHEMA_VERSION
                if stream_id is None
                else RUN_PLAN_PAIRED_SCHEMA_VERSION
            ),
            "run_id": run_id,
            "run_kind": run_kind,
            "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "config_hash": config_hash,
            "raw_input_inventory_sha256": raw_inventory_hash,
            "checkpoint_input_binding_hash": binding_hash,
            "component_canonical_hashes": component_hashes,
            "geometry_canonical_hashes": geometry_hashes,
            "code_provenance": provenance_value,
            "experiment_case_id": experiment_case_id,
            "master_seed": master_seed,
            "seed_policy": seed_policy,
            "members_per_scenario": members_per_scenario,
            "shard_scenario_count": shard_scenario_count,
            "checkpoint_interval_sweeps": checkpoint_interval_sweeps,
            "active_chunk_size": active_chunk_size,
            "scenario_count": len(ordered_scenarios),
            "particle_count": len(ordered_scenarios) * members_per_scenario,
            "shard_count": len(shards),
            "files": files,
            "shards": shard_rows,
            "scenario_selection": selection_binding,
            "scenario_ordering_policy": SCENARIO_ORDERING_POLICY,
            "lock_root": _LOCK_ROOT,
            "output_root": "shards",
            "checkpoint_root": "checkpoints",
            "failure_root": "failures",
        }
        if stream_id is not None:
            # 只在明示配對時加入欄位；舊模式不寫入 null，才能讓既有 plan 的 exact
            # schema／checksum 及 reader 行為維持原樣。
            plan["random_stream_id"] = stream_id
        _validate_plan_shape(plan)
        _write_json(partial / "run_plan.json", plan)
        _write_json(partial / "run_progress.json", _initial_progress(run_id, shards))
        (partial / "shards").mkdir()
        (partial / "checkpoints").mkdir()
        (partial / "failures").mkdir()
        _create_lock_topology(partial, shards)
        os.replace(partial, target)
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return RunWorkspace(path=target, plan=MappingProxyType(deepcopy(plan)))


def _rss_bytes() -> int:
    """取得目前 process 的最大 resident set，並統一成 bytes。"""

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS 回傳 bytes；Linux/多數 Unix 回傳 KiB。以執行平台區分，避免把 SERVER RSS
    # 低估 1024 倍；這是 engineering metric，不是科學結果欄位。
    import sys

    return value if sys.platform == "darwin" else value * 1024


def _safe_failure_message(error: Exception) -> str:
    """移除絕對路徑與常見 secret assignment，再限制 failure 訊息長度。

    這只能作為最後一道防線；request factory 本身仍不應把密碼或 token 放進例外。failure
    artifact 只保存錯誤類型與清理後訊息，不保存 traceback、環境變數或 current directory。
    """

    message = str(error).replace("\\", "/")
    message = re.sub(
        r"(?i)\b(password|passwd|token|secret|api[_-]?key)\s*[:=]\s*\S+",
        r"\1=<redacted>",
        message,
    )
    message = re.sub(r"(?:[A-Za-z]:)?/(?:[^\s,;:]+/?)+", "<path>", message)
    return message[:1000]


def _normalize_forcing_cache_stats(value: Mapping[str, Any], *, label: str) -> dict[str, int]:
    """驗證一份完整的流場管理器快取統計快照。

    snapshot 由同一個 process 的流場管理器提供，四個計數器（counter）是載入、命中、
    未命中與淘汰事件的非負整數；另外兩欄是取樣當下的管理器數量與常駐陣列位元組數。
    這兩欄是狀態量（gauge），不是事件總數。統計提供者（reporter）必須一次提供完整
    六欄；缺欄不可以默認成零，否則會把「未量測」誤報成精確
    的零事件。未知欄位、布林值、非有限數值、分數或負值都代表 reporter 違反資料契約，
    必須明確失敗，不能把錯誤轉成負增量。

    回傳值固定使用原生 ``int``，讓 baseline、checkpoint progress 與 JSON manifest 共享
    同一個可序列化表示。這裡只驗證工程計數，不代表整台 SERVER 的 I/O 或記憶體峰值。
    """

    if not isinstance(value, Mapping):
        raise ValueError(f"{label} 必須是 mapping")
    missing = sorted(set(_FORCING_CACHE_KEYS) - set(value))
    if missing:
        raise ValueError(f"{label} 缺少欄位：{','.join(missing)}")
    unknown = sorted(set(value) - set(_FORCING_CACHE_KEYS))
    if unknown:
        raise ValueError(f"{label} 含未知欄位：{','.join(str(item) for item in unknown)}")
    result: dict[str, int] = {}
    for key in _FORCING_CACHE_KEYS:
        raw = value[key]
        if isinstance(raw, bool) or not isinstance(raw, (int, float, np.integer, np.floating)):
            raise ValueError(f"{label}.{key} 必須是非負整數")
        if isinstance(raw, (int, np.integer)):
            integer = int(raw)
        else:
            numeric = float(raw)
            # 浮點整數的精確整數範圍有限；超出 2**53 時不能先轉 float 再宣稱
            # 還原原始 counter，因此明確拒絕，保護大計數不被悄悄捨入。
            if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer() or abs(numeric) > 2**53:
                raise ValueError(f"{label}.{key} 必須是可精確表示的非負整數")
            integer = int(numeric)
        if integer < 0:
            raise ValueError(f"{label}.{key} 必須是非負整數")
        result[key] = integer
    return result


def _forcing_cache_delta(
    snapshot: Mapping[str, Any], baseline: Mapping[str, Any], *, label: str
) -> dict[str, int]:
    """將累計 cache snapshot 轉成單次 shard invocation 的 counter delta。

    ``baseline`` 必須在 request factory、checkpoint restore 或粒子取樣前取得；因此每個
    invocation 內任何後續 snapshot 都與同一 baseline 比較。counter 若回退，表示 manager
    被重置、reporter 不穩定或資料損壞；此時直接拋出明確錯誤，避免以 ``max(0, ...)`` 把
    真實問題藏成看似合理的零值。manager 數量與常駐位元組是 gauge，保留目前樣本，不做
    減法，因為 cache 可能合法地驅逐或釋放資源。
    """

    current = _normalize_forcing_cache_stats(snapshot, label=f"{label}.snapshot")
    origin = _normalize_forcing_cache_stats(baseline, label=f"{label}.baseline")
    result: dict[str, int] = {}
    for key in _FORCING_CACHE_COUNTER_KEYS:
        if current[key] < origin[key]:
            raise ValueError(f"{label}.{key} counter regression")
        result[key] = current[key] - origin[key]
    for key in _FORCING_CACHE_GAUGE_KEYS:
        result[key] = current[key]
    return result


def _merge_invocation_forcing_snapshot(
    previous: Mapping[str, Any] | None,
    current: Mapping[str, Any] | None,
) -> dict[str, int] | None:
    """保留同一 invocation 內最新 counter delta 與 gauge 的觀測最大值。

    每次 ``current`` 都已相對同一起始讀值（baseline）計算，因此計數器（counter）不可把
    兩個 snapshot 再相加；較晚的 delta 已包含較早事件，直接採用即可。狀態量（gauge）會隨 cache 驅逐或 manager
    建立而上下變動，故從 baseline 與所有後續樣本取最大值，保存該 invocation 曾觀測到
    的最高 manager 數量／常駐位元組。這個 helper 不跨 invocation 使用；跨 invocation
    的合併由 ``_merge_forcing_cache_metrics`` 處理。
    """

    if current is None:
        return deepcopy(dict(previous)) if previous is not None else None
    normalized_current = _normalize_forcing_cache_stats(current, label="current invocation cache")
    if previous is None:
        return normalized_current
    normalized_previous = _normalize_forcing_cache_stats(previous, label="previous invocation cache")
    merged = dict(normalized_current)
    for key in _FORCING_CACHE_COUNTER_KEYS:
        if normalized_current[key] < normalized_previous[key]:
            # 即使兩者都沒有低於 invocation baseline，snapshot 自身也不應回退；否則
            # 後一個較小值會覆蓋已觀測的事件，讓 report 以錯誤的精確數字收尾。
            raise ValueError(f"current invocation {key} counter regression")
    for key in _FORCING_CACHE_GAUGE_KEYS:
        merged[key] = max(normalized_previous[key], normalized_current[key])
    return merged


def _merge_forcing_cache_metrics(
    current: Mapping[str, Any], previous: Mapping[str, Any]
) -> tuple[dict[str, Any] | None, str | None]:
    """合併本次 delta 與 resume 前 cache metrics，並保留 legacy 不確定性。

    新語意的計數器（counter）已是單次 invocation 增量，可以和前一次同分片已保存的
    增量相加；狀態量（gauge）則取兩次觀測的最大值。舊 progress 沒有
    ``forcing_cache_stats_semantics`` 時，無法知道其數字是否曾把 cumulative snapshot
    重複相加，因此仍保留原值供追溯，但整列標成 ``legacy_unknown``，下游報告不得把它
    當成精確總量。
    """

    current_stats = current.get("forcing_cache_stats")
    previous_stats = previous.get("forcing_cache_stats")
    if current_stats is None and previous_stats is None:
        return None, None
    current_semantics = current.get(_FORCING_CACHE_STATS_SEMANTICS_KEY)
    previous_semantics = previous.get(_FORCING_CACHE_STATS_SEMANTICS_KEY)
    if current_stats is None and previous_stats is not None:
        # 本次 invocation 未提供 snapshot（例如新 controller 沒有 reporter，或量測在
        # 中途失效）；即使前一次是 v1，也不能把後半段事件默認成零。保留舊值並降級
        # 為 legacy，讓 report 明示這段期間無法還原。
        if not isinstance(previous_stats, dict):
            raise ValueError("previous.metrics.forcing_cache_stats 必須是 object")
        return deepcopy(previous_stats), _FORCING_CACHE_STATS_LEGACY
    if current_stats is not None and current_semantics != _FORCING_CACHE_STATS_SEMANTICS_V1:
        # 相容讀取也可能收到另一個舊 controller 寫出的 unmarked／legacy object；保留
        # 其原樣並降級，不把它送進新六欄 snapshot validator。
        if not isinstance(current_stats, dict):
            raise ValueError("metrics.forcing_cache_stats 必須是 object")
        return deepcopy(current_stats), _FORCING_CACHE_STATS_LEGACY
    # 舊 progress 的 cache object 可能只有部分欄位或保留歷史額外診斷。不能套用新
    # snapshot 正規化而拒絕它，也不能把它和新 delta 合成看似精確的總量；保留舊 object
    # 原值並標成 legacy。這樣 resume 仍可讀，但本次 delta 不會冒充已知的跨 invocation
    # 總量，報告也不公開精確 aggregate。
    if previous_stats is not None and previous_semantics != _FORCING_CACHE_STATS_SEMANTICS_V1:
        if not isinstance(previous_stats, dict):
            raise ValueError("previous.metrics.forcing_cache_stats 必須是 object")
        return deepcopy(previous_stats), _FORCING_CACHE_STATS_LEGACY
    # previous 已經有執行 metrics 卻沒有 cache stats，代表早期 invocation 未量測
    # cache（例如舊版 KeyboardInterrupt 或 reporter=None）。即使本次取得完整新 delta，
    # 也無法補回早期事件，故保留本次數字但標示 legacy/unknown。
    if current_stats is not None and previous_stats is None and previous:
        return (
            _normalize_forcing_cache_stats(current_stats, label="metrics.forcing_cache_stats"),
            _FORCING_CACHE_STATS_LEGACY,
        )
    current_normalized = (
        _normalize_forcing_cache_stats(current_stats, label="metrics.forcing_cache_stats")
        if current_stats is not None
        else {key: 0 for key in _FORCING_CACHE_KEYS}
    )
    previous_normalized = (
        _normalize_forcing_cache_stats(previous_stats, label="previous.metrics.forcing_cache_stats")
        if previous_stats is not None
        else {key: 0 for key in _FORCING_CACHE_KEYS}
    )
    merged: dict[str, Any] = {}
    for key in _FORCING_CACHE_COUNTER_KEYS:
        merged[key] = current_normalized[key] + previous_normalized[key]
    for key in _FORCING_CACHE_GAUGE_KEYS:
        merged[key] = max(current_normalized[key], previous_normalized[key])
    semantics = (
        _FORCING_CACHE_STATS_SEMANTICS_V1
        if (current_stats is None or current_semantics == _FORCING_CACHE_STATS_SEMANTICS_V1)
        and (previous_stats is None or previous_semantics == _FORCING_CACHE_STATS_SEMANTICS_V1)
        else _FORCING_CACHE_STATS_LEGACY
    )
    return merged, semantics


def _merge_metrics(current: Mapping[str, Any], previous: Mapping[str, Any]) -> dict[str, Any]:
    """把 resume 前已保存的 engineering metrics 與本次 invocation 合併。

    經過時間、CPU 時間、輸出與 checkpoint 位元組數按執行區間累加；particle_steps
    已包含續跑前步數，不能再加一次。RSS 取所有執行區間的最大值。流場快取的
    loads/hits/misses/evictions 只有在新語意
    ``invocation_delta_v1`` 下才可跨 invocation 累加，manager 數量與 resident bytes 則取
    樣本最大值。歷史 row 沒有語意標記時保留數字但標為 ``legacy_unknown``，讓報告明示
    無法還原精確總量。這些數字只供資源規劃，不參與粒子物理或 seed 計算。
    """

    result = deepcopy(dict(current))
    # 時間可以使用浮點秒累加；檔案位元組則必須維持原生 int，避免大型 checkpoint
    # 在中間轉成 float 後遺失低位精度，進而讓 progress／benchmark 的容量不再可稽核。
    for key in ("wall_seconds", "process_cpu_seconds"):
        result[key] = float(result.get(key, 0)) + float(previous.get(key, 0))
    for key in ("output_bytes", "checkpoint_bytes"):
        result[key] = _nonnegative_int(
            result.get(key, 0), label=f"metrics.{key}"
        ) + _nonnegative_int(
            previous.get(key, 0), label=f"previous.metrics.{key}"
        )
    # 呼叫端傳入的 particle_steps 已含 resume 前累計值；不可在這裡再加一次。
    result["particle_steps"] = int(result.get("particle_steps", 0))
    result["max_rss_bytes"] = max(int(result.get("max_rss_bytes", 0)), int(previous.get("max_rss_bytes", 0)))
    if "checkpoint_active_bytes" in result or "checkpoint_active_bytes" in previous:
        result["checkpoint_active_bytes"] = max(
            int(result.get("checkpoint_active_bytes", 0)),
            int(previous.get("checkpoint_active_bytes", 0)),
        )
    if "output_bytes" in result:
        result["output_bytes"] = int(result["output_bytes"])
    if "checkpoint_bytes" in result:
        result["checkpoint_bytes"] = int(result["checkpoint_bytes"])
    cache_stats, cache_semantics = _merge_forcing_cache_metrics(result, previous)
    if cache_stats is not None:
        result["forcing_cache_stats"] = cache_stats
        result[_FORCING_CACHE_STATS_SEMANTICS_KEY] = cache_semantics
    elif "forcing_cache_stats" in result:
        # ``_merge_forcing_cache_metrics`` 只會在有 stats 時回傳 mapping；這個分支保留
        # 防禦性清理，避免 caller 傳入不完整的 arbitrary Mapping 後留下未驗證物件。
        result.pop("forcing_cache_stats", None)
        result.pop(_FORCING_CACHE_STATS_SEMANTICS_KEY, None)
    if (
        current.get(_FORCING_CACHE_STATS_STATUS_KEY) == _FORCING_CACHE_STATS_STATUS_UNAVAILABLE
        or previous.get(_FORCING_CACHE_STATS_STATUS_KEY) == _FORCING_CACHE_STATS_STATUS_UNAVAILABLE
    ):
        result[_FORCING_CACHE_STATS_STATUS_KEY] = _FORCING_CACHE_STATS_STATUS_UNAVAILABLE
    elif (
        "forcing_cache_stats" not in result
        and previous
        and "forcing_cache_stats" not in previous
    ):
        # 已有執行歷史但從未保存 cache snapshot；新 invocation 也沒有可合併的證據，
        # 因而明示 unavailable，而不是讓空 object 看起來像零事件的完整量測。
        result[_FORCING_CACHE_STATS_STATUS_KEY] = _FORCING_CACHE_STATS_STATUS_UNAVAILABLE
    return result


class RunController:
    """以 immutable run plan 執行、checkpoint、發布與 reconcile shard。

    controller 只實作 CPU/NumPy ``ProductionBatch``；``request_factory`` 由 caller 提供，
    因而可以在測試中注入 constant flow，也可以在後續 3B2 接上實值 OCM/NWW request
    builder。每次 ``run_shard`` 先取得 run gate shared、再取得 shard exclusive，並在
    progress mutation 時最後取得 progress exclusive；固定順序是 run gate → shard →
    progress，可讓不同 shard 並行而不讓 reconcile 與 worker 互撞。checkpoint generation
    不覆寫，latest pointer 與 progress 則以原子 JSON 替換。
    """

    def __init__(
        self,
        workspace: str | Path | RunWorkspace,
        *,
        request_factory: Callable[[RunUnit], ReferenceParticleRequest],
        resume: bool = False,
        checkpoint_root: str | Path | None = None,
        resource_reporter: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        """載入 plan/progress 並設定 request factory；不會自動開始物理計算。"""

        self.workspace = workspace.path if isinstance(workspace, RunWorkspace) else Path(workspace)
        self.plan = load_run_plan(self.workspace)
        self.progress = load_run_progress(self.workspace)
        _cross_check_plan_progress(self.plan, self.progress)
        if not callable(request_factory):
            raise TypeError("request_factory 必須是 callable")
        self.request_factory = request_factory
        self.resume = resume
        self.checkpoint_root = (
            Path(checkpoint_root)
            if checkpoint_root is not None
            else self.workspace / self.plan["checkpoint_root"]
        )
        self.output_root = self.workspace / self.plan["output_root"]
        self.failure_root = self.workspace / self.plan["failure_root"]
        self.resource_reporter = resource_reporter
        if (
            self.checkpoint_root.is_symlink()
            or self.output_root.is_symlink()
            or self.failure_root.is_symlink()
        ):
            raise ValueError("checkpoint/output/failure root 不允許 symlink")
        if self.checkpoint_root.exists() and not self.checkpoint_root.is_dir():
            raise ValueError("checkpoint root 必須是目錄")
        if not self.output_root.is_dir():
            raise ValueError("output root 必須是既有普通目錄")
        if not self.failure_root.is_dir():
            raise ValueError("failure root 必須是既有普通目錄")
        self.lock_root = self.workspace / self.plan["lock_root"]
        self._validate_lock_topology()
        self._validate_checkpoint_run_topology()

    def _lock_path(self, name: str) -> Path:
        """回傳已通過固定 topology 檢查的 lock path，不接受 caller 任意路徑。"""

        if name == "run_gate.lock" or name == "progress.lock":
            return self.lock_root / name
        shard_id = _safe_slug(name, label="shard lock name")
        if not shard_id.endswith(".lock"):
            raise ValueError("shard lock name 不合法")
        return self.lock_root / shard_id

    def _validate_lock_topology(self) -> None:
        """嚴格檢查 workspace 的固定 lock topology。

        lock file 不含資料且不列入 plan files checksum，但它們是跨程序狀態機的必要結構。
        缺失、未知、symlink、目錄或非零長度都 fail-closed；只讀 validator 也會呼叫這個
        檢查，不會取得鎖或修復它。檔案內容永遠應為空，實際互斥由 Unix ``flock`` 負責。
        """

        if self.lock_root.is_symlink() or not self.lock_root.is_dir():
            raise ValueError("lock root 必須是非 symlink 目錄")
        expected = set(_LOCK_FIXED_NAMES) | {
            f"{row['shard_id']}.lock" for row in self.plan["shards"]
        }
        entries = tuple(self.lock_root.iterdir())
        if {entry.name for entry in entries} != expected:
            raise ValueError("lock topology 欄位集合不符")
        for entry in entries:
            if entry.is_symlink() or not entry.is_file() or entry.stat().st_size != 0:
                raise ValueError("lock topology 必須是零長度普通檔案")

    def _refresh_progress(self) -> dict[str, Any]:
        """讀取最新 progress，避免 controller instance 使用舊 revision。"""

        self.progress = load_run_progress(self.workspace)
        _cross_check_plan_progress(self.plan, self.progress)
        return self.progress

    def _update_progress(self, updater: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        """在 progress lock 內重讀最新 revision 後原子發布一次 mutation。

        多個 shard worker 可以同時持有 shared run gate；若各自拿著舊 in-memory progress
        再寫回，會遺失另一 shard 的完成狀態。因此每次 mutation 都先以 blocking exclusive
        progress lock 重新載入 plan-bound progress，再執行 updater、遞增 revision 並以
        ``os.replace`` 發布。這裡不採用只檢查舊 revision 的 optimistic 假設；run lifecycle
        也一律由最新 shard 集合推導，避免 stale caller 值覆蓋現場狀態。
        """

        with acquire_run_lock(self._lock_path("progress.lock"), mode="exclusive", blocking=True):
            current = load_run_progress(self.workspace)
            _cross_check_plan_progress(self.plan, current)
            next_value = deepcopy(current)
            updater(next_value)
            next_value["run_lifecycle"] = _derived_run_lifecycle(next_value["shards"])
            next_value["revision"] = int(current["revision"]) + 1
            next_value["updated_at_utc"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
            _atomic_json(self.workspace / "run_progress.json", next_value)
            self.progress = next_value
            return next_value

    def _scenario_table(self) -> tuple[Scenario, ...]:
        """讀取並驗證 immutable scenario table。"""

        path = self.workspace / "scenario_table.parquet"
        if path.is_symlink() or not path.is_file():
            raise ValueError("scenario_table.parquet 缺失或不是普通檔案")
        table = pq.read_table(path)
        if tuple(table.column_names) != _SCENARIO_COLUMNS:
            raise ValueError("scenario_table.parquet 欄位順序不符")
        rows = table.to_pylist()
        scenarios = tuple(
            _scenario_from_row(row, label=f"scenario[{index}]") for index, row in enumerate(rows)
        )
        if len(scenarios) != int(self.plan["scenario_count"]):
            raise ValueError("scenario table row count 與 run plan 不一致")
        if len({scenario.scenario_id for scenario in scenarios}) != len(scenarios):
            raise ValueError("scenario table 含重複 scenario_id")
        if tuple(sorted(scenarios, key=scenario_execution_sort_key)) != scenarios:
            raise ValueError("scenario table 未依 scenario execution ordering policy 排序")
        return scenarios

    def _shard(self, shard_id: str) -> ScenarioShard:
        """依 plan range 從 scenario table 重建 immutable ScenarioShard。"""

        _safe_slug(shard_id, label="shard_id")
        row = next((item for item in self.plan["shards"] if item["shard_id"] == shard_id), None)
        if row is None:
            raise KeyError(f"未知 shard：{shard_id}")
        scenarios = self._scenario_table()
        start = int(row["scenario_start_index"])
        stop = int(row["scenario_stop_index"])
        subset = tuple(scenarios[start:stop])
        if len(subset) != int(row["scenario_count"]) or _scenario_hash(subset) != row["scenario_hash"]:
            raise ValueError(f"shard scenario range/hash 不符：{shard_id}")
        shard = ScenarioShard(
            shard_id=shard_id,
            experiment_case_id=self.plan["experiment_case_id"],
            scenario_start_index=start,
            scenario_stop_index=stop,
            scenarios=subset,
            members_per_scenario=int(self.plan["members_per_scenario"]),
            execution_group_id=row["execution_group_id"],
            analysis_region_id=row["analysis_region_id"],
            arrival_time_utc_ns=row["arrival_time_utc_ns"],
            group_part_index=row["group_part_index"],
            group_part_count=row["group_part_count"],
        )
        if (
            not subset
            or shard.analysis_region_id != subset[0].analysis_region_id
            or shard.arrival_time_utc_ns != subset[0].arrival_time_utc_ns
            or any(
                scenario.analysis_region_id != shard.analysis_region_id
                or scenario.arrival_time_utc_ns != shard.arrival_time_utc_ns
                for scenario in subset
            )
        ):
            raise ValueError(f"shard scenarios 與 execution group 不一致：{shard_id}")
        expected_group_id = stable_identifier(
            "grp",
            [SCENARIO_ORDERING_POLICY, shard.analysis_region_id, str(shard.arrival_time_utc_ns)],
            length=20,
        )
        if shard.execution_group_id != expected_group_id:
            raise ValueError(f"execution group identity 不一致：{shard_id}")
        expected_shard_id = _scenario_shard_id(
            experiment_case_id=self.plan["experiment_case_id"],
            execution_group_id=shard.execution_group_id,
            analysis_region_id=shard.analysis_region_id,
            arrival_time_utc_ns=shard.arrival_time_utc_ns,
            group_part_index=shard.group_part_index,
            group_part_count=shard.group_part_count,
            scenarios=subset,
            start=start,
            stop=stop,
        )
        if shard_id != expected_shard_id:
            raise ValueError(f"shard ID 不符合 execution ordering/range contract：{shard_id}")
        if shard.particle_count != int(row["particle_count"]):
            raise ValueError(f"shard particle count 不符：{shard_id}")
        return shard

    def _binding(self, shard: ScenarioShard) -> CheckpointBinding:
        """建立既有 schema 2 所需的 per-shard binding。"""

        provenance = self.plan["code_provenance"]
        code_commit = provenance.get("git_commit")
        # schema 2 的 code_commit 欄位為既有 str 契約；無 Git 的 pilot 以 deployment
        # tree token 綁定，formal 則在 initialize 時已保證是真實 40 位 commit。
        checkpoint_commit = code_commit or f"deployment-{provenance['deployment_tree_sha256']}"
        return CheckpointBinding(
            config_hash=self.plan["config_hash"],
            input_inventory_hash=self.plan["checkpoint_input_binding_hash"],
            experiment_case_id=shard.experiment_case_id,
            shard_id=shard.shard_id,
            seed_policy=self.plan["seed_policy"],
            code_commit=checkpoint_commit,
            random_stream_id=self.plan.get("random_stream_id"),
        )

    def _expected_metadata(self, shard: ScenarioShard) -> dict[str, Any]:
        """建立 output validator 用的不可變 metadata 子集合。"""

        provenance = self.plan["code_provenance"]
        metadata = {
            "run_id": self.plan["run_id"],
            "run_kind": self.plan["run_kind"],
            "config_hash": self.plan["config_hash"],
            "input_inventory_sha256": self.plan["raw_input_inventory_sha256"],
            "checkpoint_input_binding_hash": self.plan["checkpoint_input_binding_hash"],
            "component_canonical_hashes": self.plan["component_canonical_hashes"],
            "geometry_canonical_hashes": self.plan["geometry_canonical_hashes"],
            "code_commit": provenance.get("git_commit"),
            "deployment_tree_sha256": provenance["deployment_tree_sha256"],
            "uv_lock_sha256": provenance["uv_lock_sha256"],
            "dirty_flag": provenance.get("git_dirty"),
            "seed_policy": self.plan["seed_policy"],
            "shard_id": shard.shard_id,
            "experiment_case_id": shard.experiment_case_id,
        }
        if self.plan.get("random_stream_id") is not None:
            # paired plan 的輸出 manifest 也要留下 stream identity，讓 output validator
            # 與 seed table／checkpoint 使用同一份不可變 seed 命名空間證據。
            metadata["random_stream_id"] = self.plan["random_stream_id"]
        return metadata

    def _validate_checkpoint_run_topology(self) -> None:
        """拒絕 external checkpoint root 內本 run 的未知 shard、symlink 或特殊檔案。

        external root 可同時保存其他 run，因此只檢查 ``checkpoint_root/run_id``；該目錄內
        則只能有 plan 宣告的 shard 目錄。這項檢查在 request factory 前執行，避免 root
        指錯時先載入 forcing 或從頭計算。
        """

        root = self.checkpoint_root
        if not root.exists():
            return
        if root.is_symlink() or not root.is_dir():
            raise ValueError("checkpoint root 必須是非 symlink 目錄")
        run_dir = root / self.plan["run_id"]
        if not run_dir.exists():
            return
        if run_dir.is_symlink() or not run_dir.is_dir():
            raise ValueError("checkpoint run directory 必須是非 symlink 目錄")
        expected = {row["shard_id"] for row in self.plan["shards"]}
        for entry in run_dir.iterdir():
            if entry.is_symlink() or not entry.is_dir() or entry.name not in expected:
                raise ValueError(f"checkpoint run directory 含未知或不安全 entry：{entry.name}")

    def _checkpoint_parent(self, shard: ScenarioShard, *, create: bool = False) -> Path:
        """解析 ``checkpoint_root/run_id/shard_id``，必要時建立普通目錄。

        掃描與 restore 預設 ``create=False``，所以指定錯誤 external root 不會留下看似合法的
        空目錄；只有真正要寫新 generation 時才建立。每層已存在物件都必須是非 symlink
        目錄。
        """

        run_id = _safe_slug(self.plan["run_id"], label="run_id")
        shard_id = _safe_slug(shard.shard_id, label="shard_id")
        root = self.checkpoint_root
        run_dir = root / run_id
        parent = run_dir / shard_id
        for index, (path, label) in enumerate(
            (
                (root, "checkpoint root"),
                (run_dir, "checkpoint run"),
                (parent, "checkpoint shard"),
            )
        ):
            if path.exists() or path.is_symlink():
                if path.is_symlink() or not path.is_dir():
                    raise ValueError(f"{label} 必須是非 symlink 目錄")
            elif create:
                # 不同 shard worker 可同時第一次建立同一個 external run directory；
                # ``exist_ok`` 讓這個非資料性的父目錄建立具備 race-safe 行為，建立後仍
                # 立即重驗 symlink/目錄型別，不能把競態中的外來物件當成 checkpoint root。
                path.mkdir(parents=index == 0, exist_ok=True)
                if path.is_symlink() or not path.is_dir():
                    raise ValueError(f"{label} 必須是非 symlink 目錄")
        return parent

    @staticmethod
    def _generation_counters(loaded: Any) -> tuple[int, int]:
        """由 schema 2 execution 精確重建 particle steps 與保守 sweep 下界。"""

        step_counts = [int(execution.step_count) for execution in loaded.executions]
        return max(step_counts, default=0), sum(step_counts)

    def _scan_checkpoint_generations(
        self, shard: ScenarioShard, *, repair_latest: bool = False
    ) -> _CheckpointSelection | None:
        """掃描完整 generations，嚴格驗證 binding/order/checksum 並選最高 sequence。

        ``latest.json`` 缺失或合法但落後是允許的 crash window：controller 可由最高完整
        generation 修復 pointer。任何 malformed pointer、unknown/partial entry、symlink、
        checksum、binding 或 RunUnit order 問題都直接失敗，不以較舊 generation 掩蓋。
        """

        self._validate_checkpoint_run_topology()
        parent = self._checkpoint_parent(shard)
        if not parent.exists():
            return None
        expected_units = tuple(
            iter_run_units(
                shard,
                master_seed=int(self.plan["master_seed"]),
                random_stream_id=self.plan.get("random_stream_id"),
            )
        )
        binding = self._binding(shard)
        entries = tuple(parent.iterdir())
        generations: dict[int, tuple[Path, int, int]] = {}
        generation_schemas: dict[int, str] = {}
        legacy_sequence_zero: Path | None = None
        latest_path = parent / "latest.json"
        for entry in entries:
            if entry.is_symlink():
                raise ValueError(f"checkpoint directory 不允許 symlink：{entry.name}")
            if entry.name == "latest.json":
                if not entry.is_file():
                    raise ValueError("latest.json 必須是普通檔案")
                continue
            if entry.name.startswith(".") or entry.name.startswith(".partial"):
                raise ValueError(f"checkpoint partial/unknown entry：{entry.name}")
            match = _CHECKPOINT_RE.fullmatch(entry.name)
            if match is None or not entry.is_dir():
                raise ValueError(f"checkpoint unknown entry：{entry.name}")
            sequence = int(match.group(1))
            loaded_sequence, sweeps_lower_bound, particle_steps = inspect_execution_checkpoint(
                entry, expected_binding=binding, expected_run_units=expected_units
            )
            if loaded_sequence != sequence:
                raise ValueError(f"checkpoint dirname/metadata sequence 不一致：{entry.name}")
            metadata = _read_json(
                entry / "checkpoint.json",
                label=f"{entry.name}/checkpoint.json",
            )
            schema_version = metadata.get("schema_version")
            if type(schema_version) is not str or schema_version not in _CHECKPOINT_SCHEMA_VERSIONS:
                raise ValueError(f"checkpoint schema_version 不支援：{entry.name}")
            if sequence == 0:
                # schema 2 fixture 歷來允許 sequence=0；它只能作為同 parent 的一次性
                # migration source，不能被當作 v3 generation 或 progress 的目前狀態。
                if schema_version not in _CHECKPOINT_SCHEMA2_VERSIONS:
                    raise ValueError("checkpoint sequence 0 只能是 schema 2.x 遷移來源")
                if legacy_sequence_zero is not None:
                    raise ValueError("checkpoint sequence 0 不得重複")
                legacy_sequence_zero = entry
                continue
            if sequence < 1:
                raise ValueError("checkpoint generation sequence 必須從 1 開始")
            generations[sequence] = (entry, sweeps_lower_bound, particle_steps)
            generation_schemas[sequence] = schema_version
        if not generations:
            if legacy_sequence_zero is not None:
                raise ValueError("schema 2 sequence 0 必須由 schema 3 migration root 引用")
            if latest_path.exists() or latest_path.is_symlink():
                raise ValueError("latest.json 存在但沒有完整 checkpoint generation")
            return None

        if legacy_sequence_zero is not None:
            # 掃描前已完整驗證 schema 2 source；只有 migration root 明示以相對目錄綁定它
            # 時才允許其留在 generation parent，避免任意 sequence=0 目錄繞過連續性 gate。
            referenced = False
            for sequence in generation_schemas:
                if generation_schemas[sequence] not in _CHECKPOINT_SCHEMA3_VERSIONS:
                    continue
                metadata = _read_json(
                    generations[sequence][0] / "checkpoint.json",
                    label=f"checkpoint-{sequence:08d}/checkpoint.json",
                )
                legacy_source = metadata.get("legacy_source")
                if isinstance(legacy_source, dict) and legacy_source.get(
                    "relative_directory"
                ) == legacy_sequence_zero.name:
                    referenced = True
                    break
            if not referenced:
                raise ValueError("schema 2 sequence 0 必須由 schema 3 migration root 引用")

        highest_sequence = max(generations)
        # retention 尚未實作；只用排序後的實際 generation 相鄰比較，避免把極端高序號
        # （例如 stray checkpoint-99999999）展開成巨大 set。這仍能逐一指出第一個缺代，
        # 同時讓 scanner 的記憶體成本維持 O(G)，其中 G 是實際目錄數量。
        ordered_sequences = sorted(generations)
        expected_sequence = 1
        for sequence in ordered_sequences:
            if sequence != expected_sequence:
                raise ValueError(
                    "checkpoint generation sequence 必須連續，缺少："
                    f"checkpoint-{expected_sequence:08d}"
                )
            expected_sequence += 1

        # schema 只能保持在 2.x，或由 2.x 一次遷移到 3.x 後繼續使用 3.x。禁止在
        # schema 3 chain 中插入舊格式，否則下一次 writer 可能把它誤當 migration root，
        # 切斷前代 v3 hash chain。這個 gate 放在最高代選擇前，故 latest 落後時也不會
        # 先採認不合法的降級 generation。
        seen_schema3 = False
        seen_schema31 = False
        for sequence in ordered_sequences:
            schema_version = generation_schemas[sequence]
            if schema_version == _CHECKPOINT_SCHEMA3_VERSION:
                seen_schema3 = True
                seen_schema31 = True
            elif schema_version == _CHECKPOINT_SCHEMA30_VERSION:
                if seen_schema31:
                    raise ValueError(
                        "checkpoint schema 不可由 3.1.0 降回 3.0.0："
                        f"checkpoint-{sequence:08d}"
                    )
                seen_schema3 = True
            elif type(schema_version) is str and schema_version in _CHECKPOINT_SCHEMA2_VERSIONS:
                if seen_schema3:
                    raise ValueError(
                        "checkpoint schema 不可由 3.x 降回 2.x："
                        f"checkpoint-{sequence:08d}"
                    )
            else:  # pragma: no cover - schema version 已由上方集合 gate 限制
                raise ValueError(f"checkpoint schema_version 不支援：checkpoint-{sequence:08d}")

        selected: _CheckpointSelection | None = None
        if latest_path.exists():
            latest = _read_json(latest_path, label="checkpoint latest")
            expected_keys = {
                "schema_version",
                "run_id",
                "shard_id",
                "sequence",
                "directory",
                "checkpoint_relative_path",
                "checkpoint_json_sha256",
                "sweeps_completed",
                "particle_steps",
                "sweeps_source",
            }
            if set(latest) != expected_keys:
                raise ValueError("latest.json keys 不符")
            if (
                latest["schema_version"] != "1.0.0"
                or latest["run_id"] != self.plan["run_id"]
                or latest["shard_id"] != shard.shard_id
            ):
                raise ValueError("latest.json identity 不符")
            if (
                isinstance(latest["sequence"], bool)
                or not isinstance(latest["sequence"], int)
                or latest["sequence"] < 1
            ):
                raise ValueError("latest.json sequence 無效")
            directory = latest["directory"]
            if not isinstance(directory, str) or _CHECKPOINT_RE.fullmatch(directory) is None:
                raise ValueError("latest.json directory 不安全")
            pointed = parent / directory
            if pointed.is_symlink() or not pointed.is_dir():
                raise ValueError("latest.json 指向缺失 checkpoint")
            metadata_path = pointed / "checkpoint.json"
            expected_sha = latest["checkpoint_json_sha256"]
            if (
                not isinstance(expected_sha, str)
                or _SHA256_RE.fullmatch(expected_sha) is None
                or sha256_file(metadata_path) != expected_sha
            ):
                raise ValueError("latest.json checkpoint checksum 不符")
            pointer_sequence = int(latest["sequence"])
            if pointer_sequence != int(pointed.name.removeprefix("checkpoint-")):
                raise ValueError("latest.json sequence 與 directory 不一致")
            if pointer_sequence not in generations:
                raise ValueError("latest.json 指向未列入 checkpoint generation 的 sequence")
            expected_token = f"{self.plan['run_id']}/{shard.shard_id}/{directory}"
            if latest["checkpoint_relative_path"] != expected_token:
                raise ValueError("latest.json checkpoint_relative_path 不一致")
            sweeps = _nonnegative_int(latest["sweeps_completed"], label="latest.sweeps_completed")
            particle_steps = _nonnegative_int(latest["particle_steps"], label="latest.particle_steps")
            source = latest["sweeps_source"]
            if source not in {"controller_exact", "execution_step_count_lower_bound"}:
                raise ValueError("latest.json sweeps_source 不支援")
            _, lower_bound, expected_steps = generations[pointer_sequence]
            if particle_steps != expected_steps or sweeps < lower_bound:
                raise ValueError("latest.json sweep/particle counters 與 generation 不一致")
            selected = _CheckpointSelection(
                pointed,
                pointer_sequence,
                sweeps,
                particle_steps,
                source,
                "current",
            )
        elif latest_path.is_symlink():
            raise ValueError("latest.json 不允許 symlink")

        # generation 掃描階段逐代讀取 header 與 JSON payload，驗證 checksum、拓撲、cursor
        # 與 row count，但不建立完整 Observation/BoundaryEvent dataclass；最後對最高代完整
        # 還原一次，確保整條 immutable segment chain 的缺失、跳號、checksum、cursor 與歷史
        # row 都通過 resume 等級的 fail-closed 驗證，避免把高代孤兒當成可恢復狀態。
        highest_loaded = load_execution_checkpoint(
            generations[highest_sequence][0],
            expected_binding=binding,
            expected_run_units=expected_units,
        )
        if highest_loaded.sequence != highest_sequence:
            raise ValueError("最高 checkpoint generation sequence 不一致")
        if selected is None or highest_sequence > selected.sequence:
            highest_path, sweeps_lower_bound, particle_steps = generations[highest_sequence]
            pointer_state = "missing" if selected is None else "stale"
            selected = _CheckpointSelection(
                highest_path,
                highest_sequence,
                sweeps_lower_bound,
                particle_steps,
                "execution_step_count_lower_bound",
                pointer_state,
            )
            if repair_latest:
                self._write_latest(
                    shard,
                    highest_path,
                    highest_sequence,
                    sweeps_completed=sweeps_lower_bound,
                    particle_steps=particle_steps,
                    sweeps_source="execution_step_count_lower_bound",
                )
                selected = _CheckpointSelection(
                    highest_path,
                    highest_sequence,
                    sweeps_lower_bound,
                    particle_steps,
                    "execution_step_count_lower_bound",
                    "repaired",
                )
        if selected is not None:
            logical_bytes, active_bytes = _checkpoint_storage_metrics(parent)
            selected = replace(
                selected,
                logical_bytes=logical_bytes,
                active_bytes=active_bytes,
            )
        return selected

    def _write_latest(
        self,
        shard: ScenarioShard,
        checkpoint: Path,
        sequence: int,
        *,
        sweeps_completed: int,
        particle_steps: int,
        sweeps_source: str = "controller_exact",
    ) -> Path:
        """原子更新 latest pointer，保存相對 token 與 checkpoint 當下工程 counters。"""

        parent = self._checkpoint_parent(shard, create=True)
        if sweeps_source not in {"controller_exact", "execution_step_count_lower_bound"}:
            raise ValueError("sweeps_source 不支援")
        relative = checkpoint.name
        payload = {
            "schema_version": "1.0.0",
            "run_id": self.plan["run_id"],
            "shard_id": shard.shard_id,
            "sequence": sequence,
            "directory": relative,
            "checkpoint_relative_path": f"{self.plan['run_id']}/{shard.shard_id}/{relative}",
            "checkpoint_json_sha256": sha256_file(checkpoint / "checkpoint.json"),
            "sweeps_completed": _nonnegative_int(sweeps_completed, label="sweeps_completed"),
            "particle_steps": _nonnegative_int(particle_steps, label="particle_steps"),
            "sweeps_source": sweeps_source,
        }
        _atomic_json(parent / "latest.json", payload)
        return parent / "latest.json"

    def _checkpoint_for_progress(
        self,
        shard: ScenarioShard,
        row: Mapping[str, Any],
        selected: _CheckpointSelection | None,
    ) -> _CheckpointSelection | None:
        """交叉核對 progress 與指定 checkpoint root，並採認合法 RUNNING crash window。

        progress 已宣告 sequence/path 卻在目前 root 找不到 generation 時直接失敗；因此
        operator 若忘記傳入原 external root，不會呼叫 request factory 或從 seed 重新開始。
        PLANNED 看到任何 generation 亦視為外來狀態。只有 RUNNING 可採認高於 progress 的
        orphan generation，且只能是恰好下一代，因為合法中斷順序是「標 RUNNING→寫
        generation/latest→更新 progress」。PAUSED/FAILED 不可能合法地多出未登錄 generation。
        """

        lifecycle = row["lifecycle"]
        sequence = int(row["checkpoint_sequence"])
        token = row["checkpoint_relative_path"]
        if lifecycle == "PLANNED" and selected is not None:
            raise ValueError(f"PLANNED shard 不可已有 checkpoint generation：{shard.shard_id}")
        if selected is None:
            if sequence > 0 or token is not None:
                raise ValueError(
                    f"progress 宣告 checkpoint，但指定 checkpoint_root 找不到 generation：{shard.shard_id}"
                )
            return None

        def repair_pointer(value: _CheckpointSelection) -> _CheckpointSelection:
            """只在 progress state 合法後修復 pointer，並重建實際容量 metrics。"""

            if value.pointer_state in {"missing", "stale"}:
                self._write_latest(
                    shard,
                    value.path,
                    value.sequence,
                    sweeps_completed=value.sweeps_completed,
                    particle_steps=value.particle_steps,
                    sweeps_source=value.sweeps_source,
                )
                value = replace(value, pointer_state="repaired")
            logical_bytes, active_bytes = _checkpoint_storage_metrics(value.path.parent)
            return replace(
                value,
                logical_bytes=logical_bytes,
                active_bytes=active_bytes,
            )

        selected_token = f"{self.plan['run_id']}/{shard.shard_id}/{selected.path.name}"
        if selected.sequence < sequence:
            raise ValueError(f"checkpoint root 缺少 progress 宣告的較新 generation：{shard.shard_id}")
        if selected.sequence == sequence:
            if token != selected_token:
                raise ValueError(f"progress checkpoint path 與 selected generation 不一致：{shard.shard_id}")
            counters_invalid = (
                int(row["particle_steps"]) < selected.particle_steps
                or int(row["sweeps_completed"]) < selected.sweeps_completed
                if lifecycle == "COMPLETE"
                else int(row["particle_steps"]) != selected.particle_steps
                or int(row["sweeps_completed"]) != selected.sweeps_completed
            )
            if counters_invalid:
                raise ValueError(
                    f"progress checkpoint counters 與 selected generation 不一致：{shard.shard_id}"
                )
            return repair_pointer(selected)
        if lifecycle == "RUNNING" and selected.sequence > sequence + 1:
            # 合法 publish window 只會在同一個 checkpoint sequence 留下一個尚未登錄的
            # generation；若高出一代以上，代表中間代遺失、外來資料混入或 progress 已被
            # 回退。這裡不能直接採認最高代，避免 resume 跳過未驗證的粒子／RNG 狀態。
            raise ValueError(
                "RUNNING shard 的 orphan generation 只能是 progress 的下一代："
                f"progress={sequence}, selected={selected.sequence}, shard={shard.shard_id}"
            )
        if lifecycle != "RUNNING":
            raise ValueError(f"只有 RUNNING shard 可採認 crash-window orphan generation：{shard.shard_id}")
        selected = repair_pointer(selected)
        previous = row["metrics"] if isinstance(row["metrics"], dict) else {}
        metrics = {
            "wall_seconds": float(previous.get("wall_seconds", 0.0)),
            "process_cpu_seconds": float(previous.get("process_cpu_seconds", 0.0)),
            "max_rss_bytes": int(previous.get("max_rss_bytes", 0)),
            "output_bytes": int(previous.get("output_bytes", 0)),
            # scanner 已逐一驗證 generation payload；因目前 retention 不刪除 chain 需要的
            # generation，這個實際檔案總和就是可證明的 lifetime logical bytes。不能沿用
            # crash 前 progress 的舊值，也不能把 active gauge 冒充累計寫入量。
            "checkpoint_bytes": int(selected.logical_bytes),
            "checkpoint_active_bytes": int(selected.active_bytes),
            "particle_steps": selected.particle_steps,
            "sweeps_recovered_lower_bound": selected.sweeps_source == "execution_step_count_lower_bound",
        }
        if isinstance(previous.get("forcing_cache_stats"), dict):
            metrics["forcing_cache_stats"] = deepcopy(previous["forcing_cache_stats"])
            # 舊 progress 沒有語意欄位時不能假定它保存的是 invocation delta；即使後續
            # controller 能接續執行，也要把這個歷史邊界傳到報告，避免 resume 後冒稱精確。
            metrics[_FORCING_CACHE_STATS_SEMANTICS_KEY] = previous.get(
                _FORCING_CACHE_STATS_SEMANTICS_KEY,
                _FORCING_CACHE_STATS_LEGACY,
            )
        if previous.get(_FORCING_CACHE_STATS_STATUS_KEY) is not None:
            metrics[_FORCING_CACHE_STATS_STATUS_KEY] = previous[_FORCING_CACHE_STATS_STATUS_KEY]
        self._mark_checkpoint_running(
            shard,
            checkpoint=selected.path,
            sequence=selected.sequence,
            sweeps=selected.sweeps_completed,
            particle_steps=selected.particle_steps,
            metrics=metrics,
        )
        return selected

    def _metrics(
        self,
        started_wall: float,
        started_cpu: float,
        *,
        particle_steps: int,
        checkpoint_bytes: int = 0,
        checkpoint_active_bytes: int = 0,
        output_bytes: int = 0,
        forcing_stats: Mapping[str, Any] | None = None,
        forcing_stats_status: str | None = None,
    ) -> dict[str, Any]:
        """建立不含路徑的工程 metrics 與本次 cache snapshot 增量。

        ``forcing_stats`` 必須已由 invocation baseline 計算成 delta；本函式不自行讀取
        reporter，避免同一 invocation 的 checkpoint、output 與 failure snapshot 互相累加。
        reporter 發生錯誤時可只保存 ``forcing_stats_status``，讓原始物理例外維持為主要
        例外，同時在 progress 中明示 cache 量測不可用。
        """

        metrics: dict[str, Any] = {
            "wall_seconds": max(0.0, time.perf_counter() - started_wall),
            "process_cpu_seconds": max(0.0, time.process_time() - started_cpu),
            "max_rss_bytes": _rss_bytes(),
            "output_bytes": int(output_bytes),
            "checkpoint_bytes": int(checkpoint_bytes),
            "checkpoint_active_bytes": _nonnegative_int(
                checkpoint_active_bytes, label="checkpoint_active_bytes"
            ),
            "particle_steps": int(particle_steps),
        }
        if forcing_stats is not None:
            metrics["forcing_cache_stats"] = _normalize_forcing_cache_stats(
                forcing_stats,
                label="metrics.forcing_cache_stats",
            )
            metrics[_FORCING_CACHE_STATS_SEMANTICS_KEY] = _FORCING_CACHE_STATS_SEMANTICS_V1
        if forcing_stats_status is not None:
            if forcing_stats_status != _FORCING_CACHE_STATS_STATUS_UNAVAILABLE:
                raise ValueError("forcing_stats_status 不支援")
            metrics[_FORCING_CACHE_STATS_STATUS_KEY] = forcing_stats_status
        return metrics

    def _resource_snapshot(self) -> dict[str, int] | None:
        """讀取並驗證目前 resource reporter 的 cache 累計 snapshot。

        snapshot 發生在 controller process 內，不能解讀成整台 SERVER 的總量；它只供與
        本次 shard invocation 起點的 baseline 相減。reporter 若回傳 malformed 欄位會直接
        失敗，呼叫端在 failure/interrupt 清理路徑會保留原始例外並標示量測不可用。
        """

        if self.resource_reporter is None:
            return None
        return _normalize_forcing_cache_stats(
            self.resource_reporter(),
            label="resource_reporter",
        )

    def _resource_delta(self, baseline: Mapping[str, Any] | None) -> dict[str, int] | None:
        """以 invocation 開始前的 baseline 取得目前 cache counter delta。

        baseline 必須在 request factory 與 checkpoint restore 前建立；每次呼叫都重新讀
        snapshot，但不會重新設定 baseline。counter regression 會回報錯誤，gauge 則保留
        當下觀測值，確保 pause、resume 與 shared controller 的數字可追溯。
        """

        if baseline is None or self.resource_reporter is None:
            return None
        snapshot = self._resource_snapshot()
        if snapshot is None:
            return None
        return _forcing_cache_delta(
            snapshot,
            baseline,
            label="resource_reporter",
        )

    def _try_adopt_published_checkpoint(
        self,
        shard: ScenarioShard,
        *,
        batch: ProductionBatch,
        candidate: Path | None,
        sequence: int,
        sweeps: int,
        particle_steps: int,
        metrics: Mapping[str, Any],
        pause: bool,
    ) -> _CheckpointAdoptionStatus:
        """驗證 publish window 內的完整 generation，避免誤標 FAILED 或重寫序號。

        generation rename 已完成後，latest pointer 或 progress 更新可能因 NFS 暫時錯誤而
        中斷。此 helper 只接受預期序號、固定同 parent 路徑、binding／RunUnit 順序、完整
        segment chain 與 particle step counter 都通過 loader 的候選；不會接受 partial 或
        半寫入目錄。驗證成功後以 generation 檔案總和重建 lifetime logical bytes 與 active
        gauge，再 best-effort 發布 RUNNING；Ctrl-C 呼叫端另可要求 PAUSED。progress 更新
        若再次失敗，仍回傳 ``"adopted"`` 保留合法 orphan，讓下次 reconcile 依實體檔案復原。
        若 candidate 目錄已存在但 NFS 暫時無法讀取，回傳 ``"indeterminate"``；這時不得
        寫 FAILED，應保留 RUNNING 與現場，交由下一次 scanner 在 I/O 恢復後重新驗證。
        ``"not_adopted"`` 只代表 candidate 確定不存在、路徑不符或內容已確定違反契約。
        """

        if candidate is None:
            return "not_adopted"
        try:
            parent = self._checkpoint_parent(shard, create=False)
        except FileNotFoundError:
            return "not_adopted"
        except OSError:
            # parent 的 NFS metadata 暫時不可讀時，不能把 publish window 判定為不存在；
            # 保守保留 RUNNING，避免下一次 scanner 因 FAILED lifecycle 永久拒絕 orphan。
            return "indeterminate"
        expected = parent / f"checkpoint-{sequence:08d}"
        if candidate != expected:
            return "not_adopted"
        try:
            candidate_stat = candidate.lstat()
        except FileNotFoundError:
            return "not_adopted"
        except OSError:
            return "indeterminate"
        if stat.S_ISLNK(candidate_stat.st_mode) or not stat.S_ISDIR(candidate_stat.st_mode):
            return "not_adopted"
        try:
            loaded = load_execution_checkpoint(
                candidate,
                expected_binding=self._binding(shard),
                expected_run_units=batch.units,
            )
            if loaded.sequence != sequence:
                return "not_adopted"
            loaded_steps = sum(int(item.step_count) for item in loaded.executions)
            if loaded_steps != int(particle_steps):
                return "not_adopted"
            # publish window 內不應有另一份同序號 state 被誤認為本次結果；故障路徑可接受
            # 這次較重的逐粒子比對，確認 observation/event、step/minimum-clamp、current
            # state、RNG 與 triangle hint 都和尚在記憶體中的 batch 完全一致。
            if loaded.executions != [runtime.execution for runtime in batch.runtimes]:
                return "not_adopted"
            runtime_rng_states = [
                json.loads(
                    json.dumps(
                        runtime.rng.bit_generator.state,
                        ensure_ascii=False,
                        sort_keys=True,
                        allow_nan=False,
                    )
                )
                for runtime in batch.runtimes
            ]
            if loaded.rng_states != runtime_rng_states:
                return "not_adopted"
            if loaded.triangle_hints != [runtime.triangle_hint for runtime in batch.runtimes]:
                return "not_adopted"
            logical_bytes, active_bytes = _checkpoint_storage_metrics(parent)
        except OSError:
            # 目錄已經是 canonical ordinary directory，但 payload／NFS metadata 的讀取失敗
            # 無法在本次判斷內容真偽；保留 RUNNING 讓恢復後的 scanner 作完整 fail-closed
            # 驗證，而不是把暫時 I/O 錯誤寫成 FAILED。
            return "indeterminate"
        except (RuntimeError, TypeError, ValueError):
            # 候選可能只是 partial／checksum 尚未完成；原始 publish 例外必須保留，不能
            # 以「看起來像 generation」的目錄取代它。
            return "not_adopted"

        recovered_metrics = deepcopy(dict(metrics))
        # 這裡使用已通過 loader 且仍存在的 generation files 重算 logical counter；active
        # 則是同一 parent 的 generation 加 latest pointer，兩者不能互相冒充。
        recovered_metrics["checkpoint_bytes"] = int(logical_bytes)
        recovered_metrics["checkpoint_active_bytes"] = int(active_bytes)
        recovered_metrics["particle_steps"] = int(particle_steps)
        try:
            self._mark_checkpoint_running(
                shard,
                checkpoint=candidate,
                sequence=sequence,
                sweeps=sweeps,
                particle_steps=particle_steps,
                metrics=recovered_metrics,
            )
            if pause:
                self._mark_paused(
                    shard,
                    checkpoint=candidate,
                    sequence=sequence,
                    sweeps=sweeps,
                    particle_steps=particle_steps,
                    metrics=recovered_metrics,
                )
        except Exception:
            # generation 本身已完整通過驗證；progress lock／NFS 若再次失敗，保留 RUNNING
            # 的舊 row 與 orphan，交由下一次 scanner/reconcile 採認，不能降級成 FAILED。
            pass
        return "adopted"

    def _write_checkpoint(
        self,
        batch: ProductionBatch,
        shard: ScenarioShard,
        *,
        sequence: int,
        sweeps_completed: int,
        particle_steps: int,
        checkpoint_active_bytes_before: int | None = None,
    ) -> tuple[Path, int, int]:
        """寫出不可覆寫 checkpoint generation 並回傳 logical 與 active bytes。

        ``checkpoint_bytes`` 是本次新 generation 的 lifetime logical bytes-written；
        ``checkpoint_active_bytes`` 則是發布 latest 後目前 shard checkpoint tree 的已發布
        普通檔案 ``st_size`` 邏輯長度加總。若 caller 提供前一刻的 active gauge，這裡只加上
        新 generation 並扣除／加入 latest pointer 的新舊邏輯長度，避免每個 checkpoint 都
        重新遞迴掃描全部歷史目錄。它不代表 NFS 實際配置空間；兩者分開回傳，避免報告把
        歷代累計寫入量誤解成儲存閘門使用量。
        """

        parent = self._checkpoint_parent(shard, create=True)
        destination = parent / f"checkpoint-{sequence:08d}"
        previous_latest = parent / "latest.json"
        previous_latest_size = (
            int(previous_latest.stat().st_size)
            if previous_latest.exists() and previous_latest.is_file()
            else 0
        )
        if checkpoint_active_bytes_before is None:
            # 直接呼叫此 private helper 的維護／測試路徑沒有 controller 內的 gauge；
            # 仍以一次完整掃描建立正確基準，正式 run 會在第一次進入時建立後續增量。
            active_before = sum(
                int(item.stat().st_size) for item in parent.rglob("*") if item.is_file()
            )
        else:
            active_before = _nonnegative_int(
                checkpoint_active_bytes_before,
                label="checkpoint_active_bytes_before",
            )
        path = batch.write_checkpoint(
            destination,
            binding=self._binding(shard),
            sequence=sequence,
            previous_checkpoint=self._previous_checkpoint_for_write(
                checkpoint_parent=parent,
                sequence=sequence,
            ),
        )
        size = sum(item.stat().st_size for item in path.iterdir() if item.is_file())
        self._write_latest(
            shard,
            path,
            sequence,
            sweeps_completed=sweeps_completed,
            particle_steps=particle_steps,
        )
        latest_size = int(previous_latest.stat().st_size)
        active_size = active_before + int(size) - previous_latest_size + latest_size
        return path, size, int(active_size)

    @staticmethod
    def _previous_checkpoint_for_write(*, checkpoint_parent: Path, sequence: int) -> Path | None:
        """取得同一 shard 的上一代 generation，供 schema 3 segment chain 使用。

        run controller 只在完整 sweep/macro boundary 呼叫 ``_write_checkpoint``；若上一代
        不存在，writer 會建立 chain root。這裡不掃描或修復其他 generation，避免把孤兒
        狀態當成可續跑依據；sequence 由 controller 的 progress 綁定並必須連續。
        """

        if sequence <= 1:
            return None
        previous = checkpoint_parent / f"checkpoint-{sequence - 1:08d}"
        if not previous.is_dir() or previous.is_symlink():
            raise ValueError(f"schema 3 checkpoint 缺少上一代 generation：{previous.name}")
        return previous

    def _mark_running(self, shard: ScenarioShard) -> dict[str, Any]:
        """將可執行 shard 轉成 RUNNING，並增加 attempt count。

        已留下 RUNNING、PAUSED 或 FAILED 的 shard 都代表先前 attempt 已碰過外部狀態；只有
        caller 明示 ``resume=True`` 才能進入。這個方法在 checkpoint root 與 progress
        cross-link 完成 fail-closed 檢查後才會呼叫。
        """

        def update(progress: dict[str, Any]) -> None:
            row = progress["shards"][shard.shard_id]
            lifecycle = row["lifecycle"]
            if lifecycle == "COMPLETE":
                return
            if lifecycle in {"RUNNING", "PAUSED", "FAILED"} and not self.resume:
                raise RuntimeError(f"shard {shard.shard_id} 需以 resume=True 恢復：{lifecycle}")
            row["lifecycle"] = "RUNNING"
            row["attempt_count"] = int(row.get("attempt_count", 0)) + 1
            row["error_code"] = None
            row["failure_relative_path"] = None
            if progress["run_lifecycle"] in {"PLANNED", "PAUSED", "FAILED"}:
                progress["run_lifecycle"] = "RUNNING"

        return self._update_progress(update)

    def _mark_checkpoint_running(
        self,
        shard: ScenarioShard,
        *,
        checkpoint: Path,
        sequence: int,
        sweeps: int,
        particle_steps: int,
        metrics: Mapping[str, Any],
    ) -> None:
        """每個完整 periodic checkpoint 後立即發布 RUNNING progress。

        generation 與 latest 先完成，再原子更新 progress；若程序在兩者之間中斷，下一個
        controller 可辨識 higher orphan generation 並修復。這個順序永遠不會讓 progress
        宣告一個尚未完整發布的 checkpoint。
        """

        token = f"{self.plan['run_id']}/{shard.shard_id}/{checkpoint.name}"

        def update(progress: dict[str, Any]) -> None:
            row = progress["shards"][shard.shard_id]
            row.update(
                {
                    "lifecycle": "RUNNING",
                    "checkpoint_sequence": sequence,
                    "checkpoint_relative_path": token,
                    "sweeps_completed": sweeps,
                    "particle_steps": particle_steps,
                    "metrics": deepcopy(dict(metrics)),
                    "error_code": None,
                    "failure_relative_path": None,
                }
            )
            progress["run_lifecycle"] = _derived_run_lifecycle(progress["shards"])

        self._update_progress(update)

    def _mark_paused(
        self,
        shard: ScenarioShard,
        *,
        checkpoint: Path | None,
        sequence: int,
        sweeps: int,
        particle_steps: int,
        metrics: Mapping[str, Any],
    ) -> None:
        """checkpoint 後保存 PAUSED progress；不建立 partial output。"""

        checkpoint_token = None
        if checkpoint is not None:
            checkpoint_token = f"{self.plan['run_id']}/{shard.shard_id}/{checkpoint.name}"

        def update(progress: dict[str, Any]) -> None:
            row = progress["shards"][shard.shard_id]
            row.update(
                {
                    "lifecycle": "PAUSED",
                    "checkpoint_sequence": sequence,
                    "checkpoint_relative_path": checkpoint_token,
                    "sweeps_completed": sweeps,
                    "particle_steps": particle_steps,
                    "metrics": deepcopy(dict(metrics)),
                    "error_code": None,
                    "failure_relative_path": None,
                }
            )
            progress["run_lifecycle"] = _derived_run_lifecycle(progress["shards"])

        self._update_progress(update)

    def _mark_failed(self, shard: ScenarioShard, error: Exception, *, metrics: Mapping[str, Any]) -> None:
        """寫 immutable failure artifact 並把 shard/run progress 設為 FAILED。"""

        directory = self.failure_root / shard.shard_id
        if directory.exists() or directory.is_symlink():
            if directory.is_symlink() or not directory.is_dir():
                raise ValueError("failure shard directory 必須是非 symlink 目錄")
        else:
            directory.mkdir()
        failure_path = directory / f"failure-{uuid4().hex}.json"
        _write_json(
            failure_path,
            {
                "schema_version": "1.0.0",
                "run_id": self.plan["run_id"],
                "shard_id": shard.shard_id,
                "error_code": type(error).__name__,
                "message": _safe_failure_message(error),
            },
        )

        def update(progress: dict[str, Any]) -> None:
            row = progress["shards"][shard.shard_id]
            row["lifecycle"] = "FAILED"
            row["error_code"] = type(error).__name__
            row["failure_relative_path"] = f"{self.plan['failure_root']}/{shard.shard_id}/{failure_path.name}"
            row["metrics"] = deepcopy(dict(metrics))
            progress["run_lifecycle"] = _derived_run_lifecycle(progress["shards"])

        self._update_progress(update)

    def _mark_complete(
        self,
        shard: ScenarioShard,
        output: Path,
        *,
        sweeps: int,
        particle_steps: int,
        metrics: Mapping[str, Any],
    ) -> None:
        """output 驗證成功後保存 COMPLETE shard，並在全 shard 完成時完成 run。"""

        output_token = f"{self.plan['output_root']}/{shard.shard_id}"

        def update(progress: dict[str, Any]) -> None:
            row = progress["shards"][shard.shard_id]
            row.update(
                {
                    "lifecycle": "COMPLETE",
                    "checkpoint_relative_path": row.get("checkpoint_relative_path"),
                    "output_relative_path": output_token,
                    "sweeps_completed": sweeps,
                    "particle_steps": particle_steps,
                    "metrics": deepcopy(dict(metrics)),
                    "error_code": None,
                    "failure_relative_path": None,
                }
            )
            progress["run_lifecycle"] = _derived_run_lifecycle(progress["shards"])

        self._update_progress(update)

    def run_shard(self, shard_id: str, *, sweep_budget: int | None = None) -> RunExecutionSummary:
        """依固定 lock order 執行一個 shard，並在 contention 時於 request factory 前失敗。

        先取得 run gate 的 shared lock，再取得目標 shard 的 exclusive lock；因此不同
        shard 可同時執行，但 reconcile 的 run-gate exclusive lock 會等待所有 worker 離開。
        兩把鎖都採 non-blocking，第二個同 shard worker 會立即得到
        ``RunLockBusyError``，不會讀 progress、掃 checkpoint 或呼叫 ``request_factory``。
        progress 的細部 mutation 由 ``_update_progress`` 依固定順序再取得 progress lock。

        Args:
            shard_id: immutable plan 宣告的安全 shard identifier。
            sweep_budget: 本次最多執行的 sweep 數；若未 terminal，會先寫 checkpoint 再
                回傳 PAUSED，單位為 sweep。

        Returns:
            本次 shard 的工程狀態摘要。

        Raises:
            RunLockBusyError: run gate 或同 shard lock 已被其他程序持有。
        """

        _safe_slug(shard_id, label="shard_id")
        with acquire_run_lock(
            self._lock_path("run_gate.lock"), mode="shared", blocking=False
        ), acquire_run_lock(
            self._lock_path(f"{shard_id}.lock"), mode="exclusive", blocking=False
        ):
            return self._run_shard_locked(shard_id, sweep_budget=sweep_budget)

    def _run_shard_locked(
        self, shard_id: str, *, sweep_budget: int | None = None
    ) -> RunExecutionSummary:
        """執行一個 shard；可用 ``sweep_budget`` 強制 checkpoint 後 PAUSED。

        已 COMPLETE 且輸出通過 validator 的 shard 直接回傳，不會重跑。PAUSED/FAILED
        必須以 controller ``resume=True`` 恢復。每達到 checkpoint interval 就建立新的
        generation；只有整個 batch terminal 且 formal validator 通過後才發布 trajectory
        shard。checkpoint 只在 ``ProductionBatch.advance`` 正常回傳完整 sweep 結果後發布；
        若中途收到 Ctrl-C，會保留上一個 generation，下一次 resume 重新執行該 interval。
        任何物理例外會先寫 failure artifact，再把原例外重新拋給上層。
        """

        if sweep_budget is not None and (
            isinstance(sweep_budget, bool) or not isinstance(sweep_budget, int) or sweep_budget < 1
        ):
            raise ValueError("sweep_budget 必須是正整數或 None")
        shard = self._shard(shard_id)
        progress = self._refresh_progress()
        row = progress["shards"].get(shard_id)
        if row is None:
            raise KeyError(f"progress 缺少 shard：{shard_id}")
        lifecycle = row["lifecycle"]
        if lifecycle in {"RUNNING", "PAUSED", "FAILED"} and not self.resume:
            raise RuntimeError(f"shard {shard_id} 需以 resume=True 恢復：{lifecycle}")

        # checkpoint topology/binding/order 與 progress cross-link 全部先於 request factory。
        # 這是 external root 指錯時不載入 forcing、不從頭執行的核心 fail-closed 邊界。
        selected = self._scan_checkpoint_generations(shard, repair_latest=False)
        selected = self._checkpoint_for_progress(shard, row, selected)
        row = self._refresh_progress()["shards"][shard_id]
        if lifecycle == "COMPLETE":
            output = _workspace_token(
                self.workspace,
                str(row.get("output_relative_path", "")),
                label=f"progress[{shard_id}].output",
                directory=True,
            )
            validation = validate_trajectory_shard(
                output,
                require_formal_metadata=self.plan["run_kind"] == "formal",
                strict_run_metadata=self.plan["run_kind"] in {"formal", "pilot"},
                expected_metadata=self._expected_metadata(shard),
            )
            if not validation["valid"]:
                raise ValueError("progress COMPLETE 但 output invalid：" + ";".join(validation["errors"]))
            return RunExecutionSummary(
                self.plan["run_id"],
                shard_id,
                "COMPLETE",
                shard.scenario_count,
                shard.particle_count,
                int(row["sweeps_completed"]),
                int(row["particle_steps"]),
                str(row["output_relative_path"]),
                row["checkpoint_relative_path"],
            )

        self._mark_running(shard)
        started_wall = time.perf_counter()
        started_cpu = time.process_time()
        row = self.progress["shards"][shard_id]
        checkpoint_path = selected.path if selected is not None else None
        checkpoint_sequence = selected.sequence if selected is not None else 0
        sweeps_completed = int(row["sweeps_completed"])
        particle_steps = int(row["particle_steps"])
        previous_metrics = row["metrics"] if isinstance(row["metrics"], dict) else {}
        checkpoint_bytes_this_invocation = 0
        checkpoint_active_bytes = int(previous_metrics.get("checkpoint_active_bytes", 0))
        if selected is not None:
            # progress 可能來自 v2 舊紀錄或 generation/latest 已發布而 progress 尚未
            # 更新的 crash window；active gauge 以實際 checkpoint tree 重建，不能沿用
            # 缺欄的舊 metrics 猜測目前磁碟量。
            checkpoint_active_bytes = int(selected.active_bytes)
        # resource reporter 的計數器（counter）在 process 內累計；起始讀值（baseline）必須
        # 先於 request factory 與 checkpoint restore。狀態量（gauge）初值也納入本次單次
        # 執行（invocation）的峰值，但 counter 初值不會
        # 直接寫入結果，避免把歷史 manager 工作誤算給目前 shard。
        resource_baseline: dict[str, int] | None = None
        latest_forcing_stats: dict[str, int] | None = None
        resource_stats_unavailable = False
        # 指向本次尚未完成 progress 發布的預期 generation。只在 generation 已驗證且
        # RUNNING progress 成功更新後清除；若 latest／progress 的後續步驟失敗，外層 error
        # handler 可用它辨認已發布 orphan，而不增加 sequence 造成重寫或覆蓋。
        pending_checkpoint_path: Path | None = None
        # 記錄本輪是否已進入 checkpoint 發布嘗試。建立 checkpoint parent 目錄本身
        # 也可能因 Ctrl-C 中斷；若此旗標已設為 True，後續 recovery 必須沿用已分配的
        # sequence，而不能因 pending path 尚未取得就再次遞增，否則會跳過一代並讓
        # scanner 對 progress 與 generation 的交叉引用失去連續性。
        checkpoint_attempt_started = False
        # ``ProductionBatch.advance`` 可能在已修改部分 runtime 後才收到 Ctrl-C；在它
        # 正常回傳 ``ProductionAdvanceResult`` 前，sweep／particle counter 都尚未由
        # controller 確認。這個旗標讓 KeyboardInterrupt handler 保留上一個已發布
        # generation，而不把半個 sweep 序列化成可恢復 checkpoint。
        advance_in_progress = False
        try:
            try:
                resource_baseline = self._resource_snapshot()
            except Exception:
                resource_stats_unavailable = True
                raise
            if resource_baseline is not None:
                latest_forcing_stats = {
                    key: 0 if key in _FORCING_CACHE_COUNTER_KEYS else resource_baseline[key]
                    for key in _FORCING_CACHE_KEYS
                }
            if selected is not None:
                batch = ProductionBatch.from_checkpoint(
                    selected.path,
                    shard=shard,
                    master_seed=int(self.plan["master_seed"]),
                    random_stream_id=self.plan.get("random_stream_id"),
                    request_factory=self.request_factory,
                    expected_binding=self._binding(shard),
                    active_chunk_size=self.plan["active_chunk_size"],
                )
                checkpoint_sequence = selected.sequence
                checkpoint_path = selected.path
                batch.sweep_count = sweeps_completed
            else:
                batch = ProductionBatch(
                    shard,
                    master_seed=int(self.plan["master_seed"]),
                    random_stream_id=self.plan.get("random_stream_id"),
                    request_factory=self.request_factory,
                    active_chunk_size=self.plan["active_chunk_size"],
                )
            interval = int(self.plan["checkpoint_interval_sweeps"])
            budget_left = sweep_budget
            while not batch.terminal:
                # 每輪 advance 開始時尚未分配新的 generation。只有 advance 正常回傳後，
                # controller 才知道完整 sweep 的 counters，才可把目前 batch 寫成 checkpoint。
                checkpoint_attempt_started = False
                step_count = interval if budget_left is None else min(interval, budget_left)
                advance_in_progress = True
                result = batch.advance(step_count)
                sweeps_completed += result.sweeps_completed
                particle_steps += result.stepped_particle_count
                if budget_left is not None:
                    budget_left -= result.sweeps_completed
                # 只有完整結果與 controller counters 都已接收後才離開 safe-boundary
                # guard；此前任何 Ctrl-C 都必須捨棄目前 mutable batch，從上一代重算。
                advance_in_progress = False
                # 每次 advance 都是從上一個完整 checkpoint 起最多 interval sweeps；因此
                # off-boundary budget pause 後重新啟動仍會在下一個 interval 內產生 generation，
                # 不依無法可靠持久化的全域 sweep modulo。
                if not batch.terminal:
                    checkpoint_attempt_started = True
                    checkpoint_sequence += 1
                    pending_checkpoint_path = self._checkpoint_parent(shard, create=True) / (
                        f"checkpoint-{checkpoint_sequence:08d}"
                    )
                    checkpoint_path, checkpoint_bytes, checkpoint_active_bytes = self._write_checkpoint(
                        batch,
                        shard,
                        sequence=checkpoint_sequence,
                        sweeps_completed=sweeps_completed,
                        particle_steps=particle_steps,
                        checkpoint_active_bytes_before=checkpoint_active_bytes,
                    )
                    checkpoint_bytes_this_invocation += checkpoint_bytes
                    try:
                        stats = self._resource_delta(resource_baseline)
                    except Exception:
                        resource_stats_unavailable = True
                        raise
                    latest_forcing_stats = _merge_invocation_forcing_snapshot(
                        latest_forcing_stats,
                        stats,
                    )
                    metrics = _merge_metrics(
                        self._metrics(
                            started_wall,
                            started_cpu,
                            particle_steps=particle_steps,
                            checkpoint_bytes=checkpoint_bytes_this_invocation,
                            checkpoint_active_bytes=checkpoint_active_bytes,
                            forcing_stats=latest_forcing_stats,
                        ),
                        previous_metrics,
                    )
                    self._mark_checkpoint_running(
                        shard,
                        checkpoint=checkpoint_path,
                        sequence=checkpoint_sequence,
                        sweeps=sweeps_completed,
                        particle_steps=particle_steps,
                        metrics=metrics,
                    )
                    if budget_left is not None and budget_left == 0:
                        self._mark_paused(
                            shard,
                            checkpoint=checkpoint_path,
                            sequence=checkpoint_sequence,
                            sweeps=sweeps_completed,
                            particle_steps=particle_steps,
                            metrics=metrics,
                        )
                        # PAUSED progress 也已發布，後續才不需要把此 generation 視為待採認
                        # candidate；若 _mark_paused 失敗，pending 仍保留給 outer recovery。
                        pending_checkpoint_path = None
                        return RunExecutionSummary(
                            self.plan["run_id"],
                            shard_id,
                            "PAUSED",
                            shard.scenario_count,
                            shard.particle_count,
                            sweeps_completed,
                            particle_steps,
                            None,
                            f"{self.plan['run_id']}/{shard_id}/{checkpoint_path.name}",
                        )
                    # generation、latest 與 RUNNING progress 都已發布；下一輪 advance 會
                    # 以這個 generation 作為上一代，不需再由 recovery 採認。
                    pending_checkpoint_path = None
                    checkpoint_attempt_started = False
            results = batch.results()
            output = self.output_root / shard.shard_id
            metadata = self._expected_metadata(shard)
            provenance = self.plan["code_provenance"]
            try:
                stats = self._resource_delta(resource_baseline)
            except Exception:
                resource_stats_unavailable = True
                raise
            latest_forcing_stats = _merge_invocation_forcing_snapshot(
                latest_forcing_stats,
                stats,
            )
            resource_metrics = _merge_metrics(
                self._metrics(
                    started_wall,
                    started_cpu,
                    particle_steps=particle_steps,
                    checkpoint_bytes=checkpoint_bytes_this_invocation,
                    checkpoint_active_bytes=checkpoint_active_bytes,
                    forcing_stats=latest_forcing_stats,
                ),
                previous_metrics,
            )
            metadata.update(
                {
                    "resource_usage": resource_metrics,
                    "scenario_start_index": shard.scenario_start_index,
                    "scenario_stop_index": shard.scenario_stop_index,
                    "scenario_hash": _scenario_hash(shard.scenarios),
                    "particle_count": shard.particle_count,
                    "members_per_scenario": shard.members_per_scenario,
                    "code_commit": provenance.get("git_commit"),
                }
            )
            write_trajectory_shard(output, results, run_metadata=metadata)
            validation = validate_trajectory_shard(
                output,
                require_formal_metadata=self.plan["run_kind"] == "formal",
                strict_run_metadata=self.plan["run_kind"] in {"formal", "pilot"},
                expected_metadata=self._expected_metadata(shard),
            )
            if not validation["valid"]:
                raise ValueError("剛發布的 shard 未通過 validator：" + ";".join(validation["errors"]))
            output_bytes = sum(item.stat().st_size for item in output.rglob("*") if item.is_file())
            metrics = _merge_metrics(
                self._metrics(
                    started_wall,
                    started_cpu,
                    particle_steps=particle_steps,
                    output_bytes=output_bytes,
                    checkpoint_bytes=checkpoint_bytes_this_invocation,
                    checkpoint_active_bytes=checkpoint_active_bytes,
                    forcing_stats=latest_forcing_stats,
                ),
                previous_metrics,
            )
            self._mark_complete(
                shard, output, sweeps=sweeps_completed, particle_steps=particle_steps, metrics=metrics
            )
            return RunExecutionSummary(
                self.plan["run_id"],
                shard_id,
                "COMPLETE",
                shard.scenario_count,
                shard.particle_count,
                sweeps_completed,
                particle_steps,
                f"{self.plan['output_root']}/{shard_id}",
                f"{self.plan['run_id']}/{shard_id}/{checkpoint_path.name}" if checkpoint_path else None,
            )
        except KeyboardInterrupt:
            # Ctrl-C 可能落在 advance 或 generation rename、latest pointer、progress update
            # 的窗口。advance 尚未正常回傳時不允許把半個 sweep 寫入 checkpoint；已進入
            # publish window 才由下方 recovery 驗證並採認完整 generation。
            # 若預期 generation 已完整存在，先驗證並採認它，不能無條件 sequence += 1 再寫
            # 一份相同狀態；若尚未完成 rename，writer 的 BaseException cleanup 會清理 partial，
            # 再用同一序號做一次 best-effort checkpoint 即可。
            if "batch" in locals() and advance_in_progress:
                # advance 尚未正常回傳，當前 execution／RNG 可能只完成部分粒子或部分
                # sweep；此時不能呼叫 writer，也不能用舊 counters 假裝已完成。_mark_running
                # 已在進入 loop 前保留上一個已發布 generation，下一次 resume 會從該代重算
                # 整個 interval，避免半步狀態造成重複／遺失且無法驗證的結果。
                try:
                    stats = self._resource_delta(resource_baseline)
                    latest_forcing_stats = _merge_invocation_forcing_snapshot(
                        latest_forcing_stats,
                        stats,
                    )
                except Exception:
                    resource_stats_unavailable = True
                safe_metrics = _merge_metrics(
                    self._metrics(
                        started_wall,
                        started_cpu,
                        particle_steps=particle_steps,
                        checkpoint_active_bytes=checkpoint_active_bytes,
                        forcing_stats=latest_forcing_stats,
                        forcing_stats_status=(
                            _FORCING_CACHE_STATS_STATUS_UNAVAILABLE
                            if resource_stats_unavailable
                            else None
                        ),
                    ),
                    previous_metrics,
                )

                def save_safe_boundary_metrics(progress: dict[str, Any]) -> None:
                    """只更新可稽核量測，不把半個 sweep 宣告成 checkpoint。"""

                    progress["shards"][shard_id]["metrics"] = deepcopy(safe_metrics)

                with suppress(Exception):
                    self._update_progress(save_safe_boundary_metrics)
                    # progress lock／NFS 若在這個 best-effort metrics 更新中失敗，不能讓
                    # 它取代 operator 的 Ctrl-C；下一次 resume 仍會以舊 progress 重算。
                raise
            try:
                if "batch" in locals() and not batch.terminal:
                    recovery_metrics = _merge_metrics(
                        self._metrics(
                            started_wall,
                            started_cpu,
                            particle_steps=particle_steps,
                            checkpoint_active_bytes=checkpoint_active_bytes,
                            forcing_stats=latest_forcing_stats,
                            forcing_stats_status=(
                                _FORCING_CACHE_STATS_STATUS_UNAVAILABLE
                                if resource_stats_unavailable
                                else None
                            ),
                        ),
                        previous_metrics,
                    )
                    adoption_status = self._try_adopt_published_checkpoint(
                        shard,
                        batch=batch,
                        candidate=pending_checkpoint_path,
                        sequence=checkpoint_sequence,
                        sweeps=sweeps_completed,
                        particle_steps=particle_steps,
                        metrics=recovery_metrics,
                        pause=True,
                    )
                    if adoption_status == "adopted":
                        checkpoint_path = pending_checkpoint_path
                        pending_checkpoint_path = None
                    elif adoption_status == "indeterminate":
                        # candidate 已經是 canonical generation，但讀取時發生暫時 I/O 錯誤；
                        # 不重寫同序號，也不把 shard 降為 FAILED，finally 會保留原始 Ctrl-C
                        # 並讓下一次 resume/reconcile 重新掃描現場。
                        pass
                    else:
                        # 若上一個 writer 已留下同名普通目錄，不能覆寫未知／不完整資料；
                        # 讓原始 KeyboardInterrupt 結束，下一次 scanner 會明確報告損壞。
                        if pending_checkpoint_path is not None and pending_checkpoint_path.exists():
                            raise RuntimeError(
                                "Ctrl-C 後 checkpoint generation 未通過完整驗證，拒絕覆寫"
                            )
                        if pending_checkpoint_path is None:
                            # 若中斷發生在本輪 checkpoint parent/path 建立途中，sequence
                            # 已經先分配但 pending path 尚未寫入；沿用該 sequence。advance
                            # 期間的中斷已在 handler 開頭直接保留上一代，不會走到這裡。
                            if not checkpoint_attempt_started:
                                checkpoint_sequence += 1
                            pending_checkpoint_path = self._checkpoint_parent(
                                shard, create=True
                            ) / f"checkpoint-{checkpoint_sequence:08d}"
                        checkpoint_path, checkpoint_bytes, checkpoint_active_bytes = self._write_checkpoint(
                            batch,
                            shard,
                            sequence=checkpoint_sequence,
                            sweeps_completed=sweeps_completed,
                            particle_steps=particle_steps,
                            checkpoint_active_bytes_before=checkpoint_active_bytes,
                        )
                        checkpoint_bytes_this_invocation += checkpoint_bytes
                        try:
                            stats = self._resource_delta(resource_baseline)
                            latest_forcing_stats = _merge_invocation_forcing_snapshot(
                                latest_forcing_stats,
                                stats,
                            )
                        except Exception:
                            # Ctrl-C 是 operator 的原始動作；reporter 若同時損壞，不能以
                            # 量測錯誤取代 KeyboardInterrupt。沿用最近合法值並留下 unavailable
                            # 標記，讓後續報告不會把它當成完整精確總量。
                            resource_stats_unavailable = True
                        metrics = _merge_metrics(
                            self._metrics(
                                started_wall,
                                started_cpu,
                                particle_steps=particle_steps,
                                checkpoint_bytes=checkpoint_bytes_this_invocation,
                                checkpoint_active_bytes=checkpoint_active_bytes,
                                forcing_stats=latest_forcing_stats,
                                forcing_stats_status=(
                                    _FORCING_CACHE_STATS_STATUS_UNAVAILABLE
                                    if resource_stats_unavailable
                                    else None
                                ),
                            ),
                            previous_metrics,
                        )
                        self._mark_checkpoint_running(
                            shard,
                            checkpoint=checkpoint_path,
                            sequence=checkpoint_sequence,
                            sweeps=sweeps_completed,
                            particle_steps=particle_steps,
                            metrics=metrics,
                        )
                        pending_checkpoint_path = None
                        self._mark_paused(
                            shard,
                            checkpoint=checkpoint_path,
                            sequence=checkpoint_sequence,
                            sweeps=sweeps_completed,
                            particle_steps=particle_steps,
                            metrics=metrics,
                        )
                elif "batch" not in locals():
                    # request factory／checkpoint restore 建構期間也可能已觸發 forcing loads；
                    # 尚無可驗證 generation 時只保存 RUNNING 量測，不宣告虛假的 checkpoint。
                    try:
                        stats = self._resource_delta(resource_baseline)
                        latest_forcing_stats = _merge_invocation_forcing_snapshot(
                            latest_forcing_stats,
                            stats,
                        )
                    except Exception:
                        resource_stats_unavailable = True
                    metrics = _merge_metrics(
                        self._metrics(
                            started_wall,
                            started_cpu,
                            particle_steps=particle_steps,
                            checkpoint_active_bytes=checkpoint_active_bytes,
                            forcing_stats=latest_forcing_stats,
                            forcing_stats_status=(
                                _FORCING_CACHE_STATS_STATUS_UNAVAILABLE
                                if resource_stats_unavailable or latest_forcing_stats is None
                                else None
                            ),
                        ),
                        previous_metrics,
                    )

                    def save_running_metrics(progress: dict[str, Any]) -> None:
                        """保存無 checkpoint 的 RUNNING 量測，維持既有生命週期。"""

                        progress["shards"][shard_id]["metrics"] = deepcopy(metrics)

                    self._update_progress(save_running_metrics)
            finally:
                raise
        except Exception as error:
            try:
                # 物理／request factory 失敗前可能已經消耗 forcing；最後一次 best-effort
                # snapshot 能保留這些事件。若 reporter 自身失效，絕不可讓它覆蓋原始 error，
                # 改以 unavailable 標記交給報告層處理。
                stats = self._resource_delta(resource_baseline)
                latest_forcing_stats = _merge_invocation_forcing_snapshot(
                    latest_forcing_stats,
                    stats,
                )
            except Exception:
                resource_stats_unavailable = True
            failure_metrics = _merge_metrics(
                self._metrics(
                    started_wall,
                    started_cpu,
                    particle_steps=particle_steps,
                    checkpoint_bytes=checkpoint_bytes_this_invocation,
                    checkpoint_active_bytes=checkpoint_active_bytes,
                    forcing_stats=latest_forcing_stats,
                    forcing_stats_status=(
                        _FORCING_CACHE_STATS_STATUS_UNAVAILABLE
                        if resource_stats_unavailable
                        else None
                    ),
                ),
                previous_metrics,
            )
            # generation 已 rename 但 latest/progress 更新失敗時，candidate 仍是完整且可
            # restore 的工程狀態；先採認為 RUNNING，讓下一次 resume/reconcile 延續，而不
            # 寫 FAILED artifact 掩蓋合法 orphan。只有 candidate 確定不存在、publish 前
            # 失敗，或 loader 已確定判定內容違反契約，才進入既有 failure 語意；暫時 OSError
            # 會由 adoption helper 回報 indeterminate 並保留 RUNNING。
            adoption_status = (
                self._try_adopt_published_checkpoint(
                    shard,
                    batch=batch,
                    candidate=pending_checkpoint_path,
                    sequence=checkpoint_sequence,
                    sweeps=sweeps_completed,
                    particle_steps=particle_steps,
                    metrics=failure_metrics,
                    pause=False,
                )
                if "batch" in locals()
                else "not_adopted"
            )
            if adoption_status in {"adopted", "indeterminate"}:
                raise
            self._mark_failed(shard, error, metrics=failure_metrics)
            raise

    def run_all(self, *, sweep_budget: int | None = None) -> tuple[RunExecutionSummary, ...]:
        """依 immutable plan 順序執行全部 shard；已完成 shard 只做驗證。"""

        return tuple(
            self.run_shard(row["shard_id"], sweep_budget=sweep_budget) for row in self.plan["shards"]
        )

    def reconcile(self) -> dict[str, Any]:
        """在 run-gate exclusive lock 內執行完整 reconcile。

        reconcile 可能修復 latest pointer 或採認 published output，故不能與任何 worker
        的 shared gate 交錯。non-blocking 取得 exclusive gate 失敗時直接回傳
        ``RunLockBusyError``，不掃描、不修改 progress/latest；只讀 ``validate_run`` 則不
        取得這把鎖，也不做修復。
        """

        with acquire_run_lock(self._lock_path("run_gate.lock"), mode="exclusive", blocking=False):
            return self._reconcile_locked()

    @staticmethod
    def _published_resource_metrics(
        validation: Mapping[str, Any], fallback: Mapping[str, Any]
    ) -> dict[str, Any]:
        """從已驗證 output manifest 取回 resource usage，供 reconcile 保留量測。

        output 可能已在 progress 更新前成功發布；這時 manifest 內的
        ``run_metadata.resource_usage`` 是該完成 shard 最完整的工程 snapshot，不能被
        checkpoint 時的較舊 metrics 覆蓋。新 cache 語意與 legacy 標記會原樣保留；若是
        舊 synthetic output 沒有這個欄位，才沿用 progress fallback。此 helper 只讀已通過
        trajectory validator 的 mapping，不讀取大型 trajectory array，也不改變物理結果。
        """

        manifest = validation.get("manifest")
        metadata = manifest.get("run_metadata") if isinstance(manifest, Mapping) else None
        candidate = metadata.get("resource_usage") if isinstance(metadata, Mapping) else None
        if isinstance(candidate, dict):
            try:
                return _validate_metrics(candidate, label="output.resource_usage", allow_empty=False)
            except ValueError:
                # validator 已先保證 strict pilot/formal 的固定六個工程欄位；對舊 synthetic
                # output 保留既有 progress，避免 reconcile 因歷史 optional metrics 中斷。
                pass
        return deepcopy(dict(fallback))

    def _reconcile_locked(self) -> dict[str, Any]:
        """檢查 checkpoint/output 後採認可恢復的狀態；此內部方法已持有 exclusive gate。

        output 已原子發布但 progress 尚未更新時，只要 run metadata、binding 與 shard
        validator 通過便標記 COMPLETE；progress 已 COMPLETE 卻找不到合法 output 則
        fail-fast。latest pointer 遺失或落後時可由完整 generations 選最高 sequence，
        但 unknown directory、symlink、partial 或 checksum 損壞一律保留現場並拋錯。
        """

        progress = self._refresh_progress()
        for shard_row in self.plan["shards"]:
            shard = self._shard(shard_row["shard_id"])
            state = progress["shards"][shard.shard_id]
            selected = self._scan_checkpoint_generations(shard, repair_latest=False)
            self._checkpoint_for_progress(shard, state, selected)
            progress = self._refresh_progress()
            state = progress["shards"][shard.shard_id]
            output_token = state.get("output_relative_path")
            output = (
                _workspace_token(
                    self.workspace,
                    output_token,
                    label=f"progress[{shard.shard_id}].output",
                    directory=True,
                )
                if isinstance(output_token, str)
                else self.output_root / shard.shard_id
            )
            validation = validate_trajectory_shard(
                output,
                require_formal_metadata=self.plan["run_kind"] == "formal",
                strict_run_metadata=self.plan["run_kind"] in {"formal", "pilot"},
                expected_metadata=self._expected_metadata(shard),
            )
            if state["lifecycle"] == "COMPLETE":
                if not validation["valid"]:
                    raise ValueError(f"progress COMPLETE 但 output invalid：{shard.shard_id}")
            elif validation["valid"] and state["lifecycle"] == "RUNNING":
                metrics = self._published_resource_metrics(
                    validation,
                    state.get("metrics", {}),
                )
                self._mark_complete(
                    shard,
                    output,
                    sweeps=int(state.get("sweeps_completed", 0)),
                    particle_steps=int(state.get("particle_steps", 0)),
                    metrics=metrics,
                )
            elif validation["valid"]:
                # 只有 worker 已先標 RUNNING，才可能是「output publish 後尚未更新
                # progress」的合法 crash window。PLANNED/PAUSED/FAILED 出現 output 時
                # 不可擅自補 attempt 或改成 COMPLETE，否則會掩蓋外來／重放狀態。
                raise ValueError(
                    f"非 RUNNING shard 出現 published output，拒絕 reconcile：{shard.shard_id}"
                )
        return self._refresh_progress()


__all__ = [
    "RUN_PLAN_LEGACY_SCHEMA_VERSION",
    "RUN_PLAN_PAIRED_SCHEMA_VERSION",
    "RUN_PLAN_SCHEMA_VERSION",
    "RUN_PROGRESS_SCHEMA_VERSION",
    "RunController",
    "RunExecutionSummary",
    "RunWorkspace",
    "checkpoint_input_binding_hash",
    "initialize_run_workspace",
    "load_run_plan",
    "load_run_progress",
    "validate_run_plan_document",
    "validate_run_progress_document",
]
