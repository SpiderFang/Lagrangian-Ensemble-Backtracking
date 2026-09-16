"""正式 run 的持久 worker 分組、前置閘門與程序協調。

本模組只負責把同一份不可變 run plan 的既有 shard ID 分配給固定數量的長壽命程序，
每個子程序再透過既有 run-worker 一次連續處理自己的分片。分組只影響執行順序與
流場／月份資料的重用，不改寫 run plan、scenario、particle ID、seed 或 checkpoint cadence。
所有執行輸出、日誌與 Numba 編譯快取都必須放在操作者明示且通過儲存閘門的 scratch
樹下；程式碼根目錄與使用者暫存目錄不會作為這些可寫資料的預設位置。
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import secrets
import signal
import stat
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .provenance import collect_code_provenance
from .run_control import load_run_plan, load_run_progress
from .run_validation import validate_run

_GATE_ROOT_LABELS = (
    "result_nfs_root",
    "execution_package_root",
    "output_root",
    "scratch_root",
    "checkpoint_root",
    "uv_cache_root",
    "mpl_cache_root",
    "xdg_cache_root",
    "tmp_root",
)
_NFS_TYPES = frozenset({"nfs", "nfs4"})
_CHILD_BOOTSTRAP = r"""
import json
import os
import sys

spec = json.loads(sys.argv[1])
cpu_ids = spec.get("cpu_ids")
affinity = {
    "requested_cpu_ids": cpu_ids,
    "applied": False,
    "status": "not_requested" if spec.get("affinity_request") == "none" else "unavailable_platform_fallback",
}
if cpu_ids is not None:
    setter = getattr(os, "sched_setaffinity", None)
    if setter is None:
        affinity["status"] = "unavailable_platform_fallback"
    else:
        try:
            setter(0, set(cpu_ids))
            affinity["applied"] = True
            affinity["status"] = "applied"
        except OSError as exc:
            affinity["status"] = "unavailable_runtime_fallback"
            affinity["error_type"] = type(exc).__name__
elif spec.get("affinity_fallback_reason"):
    affinity["status"] = spec["affinity_fallback_reason"]

if spec.get("warmup_numba_backend"):
    # Dispatcher 使用 cache=False，故必須在每個長壽命 worker 內暖機一次；編譯結果留在
    # 該程序記憶體，不能宣稱由其他 worker 或下次執行共用磁碟快取。
    from lagrangian_backtracking.accelerated import warmup_numba_backend
    spec["numba_warmup_summary"] = warmup_numba_backend()

from lagrangian_backtracking.parallel_execution import worker_child_main
raise SystemExit(worker_child_main(spec, affinity))
"""
_NUMBA_WARMUP_BOOTSTRAP = r"""
import importlib
import json

accelerated = importlib.import_module("lagrangian_backtracking.accelerated")
warmup = getattr(accelerated, "warmup_numba_backend", None)
if not callable(warmup):
    raise RuntimeError("warmup_numba_backend() is unavailable")
result = warmup()
payload = {"artifact_type": "numba_warmup_child_result", "result": result}
print(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
"""


class ParallelExecutionError(RuntimeError):
    """前置閘門、分組或批次執行無法安全繼續時使用的固定錯誤型別。"""


class ParallelExecutionInterrupted(RuntimeError):
    """主程序收到 SIGINT／SIGTERM 後用於保留現場並通知子程序的控制例外。"""

    def __init__(self, signum: int) -> None:
        """保存原始訊號編號，讓批次摘要與 shell exit code 可區分 Ctrl-C／SIGTERM。"""

        super().__init__(f"收到訊號 {signum}")
        self.signum = signum


@dataclass(frozen=True, slots=True)
class WorkerGroup:
    """一個 worker 要依序執行的連續 scenario 範圍與穩定 locality 摘要。

    shard_ids 保持 immutable run plan 的原始順序；scenario_start_index 與
    scenario_stop_index 是半開區間，因此不同 worker 不會重疊或改變情境身分。
    locality_keys 只描述計畫已有的分析區域、UTC 月份與表格識別欄位，不會推測
    flow domain 或修改到達時刻。worker 編號及其分組完全由 run plan 與 worker count 決定，
    不受子程序完成先後影響。
    """

    worker_id: str
    shard_ids: tuple[str, ...]
    scenario_start_index: int
    scenario_stop_index: int
    particle_count: int
    locality_keys: tuple[tuple[str, str], ...]
    study_site_ids: tuple[str, ...]
    flow_domain_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """回傳固定欄位、與執行完成順序無關的 JSON-safe worker 分組摘要。"""

        return {
            "worker_id": self.worker_id,
            "shard_ids": list(self.shard_ids),
            "scenario_range": [self.scenario_start_index, self.scenario_stop_index],
            "particle_count": self.particle_count,
            "locality_keys": [
                {"analysis_region_id": region, "arrival_month_utc": month}
                for region, month in self.locality_keys
            ],
            "study_site_ids": list(self.study_site_ids),
            "flow_domain_ids": list(self.flow_domain_ids),
        }


def canonical_json_bytes(value: object) -> bytes:
    """以 UTF-8、固定 key 順序及緊湊分隔符產生可重現的分組位元組。"""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _strict_existing_directory(value: str | Path, *, label: str) -> Path:
    """驗證絕對既有目錄及全部路徑元件，拒絕 symlink 後回傳 canonical path。"""

    path = Path(value)
    if not path.is_absolute() or any(part in {".", ".."} for part in path.parts):
        raise ParallelExecutionError(f"{label} 必須是無相對跳脫的絕對路徑")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            entry = current.lstat()
        except OSError as exc:
            raise ParallelExecutionError(f"{label} 路徑元件不存在或不可讀") from exc
        if stat.S_ISLNK(entry.st_mode):
            raise ParallelExecutionError(f"{label} 不允許符號連結路徑元件")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ParallelExecutionError(f"{label} 無法解析") from exc
    if not resolved.is_dir() or not os.access(resolved, os.X_OK):
        raise ParallelExecutionError(f"{label} 必須是可搜尋的既有目錄")
    return resolved


def _strict_regular_file(value: str | Path, *, label: str, maximum_bytes: int | None = None) -> Path:
    """驗證絕對普通檔案與各路徑元件，避免 gate／設定經 symlink 被替換。"""

    path = Path(value)
    if not path.is_absolute() or any(part in {".", ".."} for part in path.parts):
        raise ParallelExecutionError(f"{label} 必須是無相對跳脫的絕對路徑")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            entry = current.lstat()
        except OSError as exc:
            raise ParallelExecutionError(f"{label} 路徑元件不存在或不可讀") from exc
        if stat.S_ISLNK(entry.st_mode):
            raise ParallelExecutionError(f"{label} 不允許符號連結路徑元件")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ParallelExecutionError(f"{label} 無法解析") from exc
    if not resolved.is_file():
        raise ParallelExecutionError(f"{label} 必須是普通檔案")
    if maximum_bytes is not None and resolved.stat().st_size > maximum_bytes:
        raise ParallelExecutionError(f"{label} 超過允許的檔案大小")
    return resolved


def _relative_to_root(candidate: str | Path, root: Path, *, label: str) -> tuple[Path, tuple[str, ...]]:
    """將明示資料目錄限制在已通過儲存閘門的 scratch 根之下。"""

    path = Path(candidate)
    if not path.is_absolute() or any(part in {".", ".."} for part in path.parts):
        raise ParallelExecutionError(f"{label} 必須是無相對跳脫的絕對路徑")
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ParallelExecutionError(f"{label} 必須位於已驗證 scratch root 之下") from exc
    parts = tuple(relative.parts)
    if not parts:
        raise ParallelExecutionError(f"{label} 不可等於 scratch root 本身")
    return path, parts


def _validate_storage_gate_snapshot(path: str | Path) -> tuple[dict[str, object], str]:
    """確認 SERVER 儲存快照為 PASS，且涵蓋全部受管理 NFS 根與鎖探針。"""

    evidence_path = _strict_regular_file(path, label="storage gate evidence", maximum_bytes=65_536)
    raw = evidence_path.read_bytes()
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ParallelExecutionError("storage gate evidence 不是有效 JSON") from exc
    if not isinstance(document, dict) or document.get("schema_version") != "1.0.0":
        raise ParallelExecutionError("storage gate evidence schema 不支援")
    if document.get("gate_status") != "PASS" or document.get("issues") != []:
        raise ParallelExecutionError("storage gate evidence 未通過或含問題")
    roots_raw = document.get("roots")
    if not isinstance(roots_raw, list):
        raise ParallelExecutionError("storage gate evidence 缺少 roots")
    roots = {
        row.get("label"): row
        for row in roots_raw
        if isinstance(row, dict) and isinstance(row.get("label"), str)
    }
    if set(roots) != set(_GATE_ROOT_LABELS):
        raise ParallelExecutionError("storage gate evidence roots 標籤不完整")
    for label in _GATE_ROOT_LABELS:
        if roots[label].get("gate_status") != "PASS":
            raise ParallelExecutionError(f"storage gate evidence 的 {label} 未通過")
    root_nfs = roots["result_nfs_root"]
    scratch = roots["scratch_root"]
    if root_nfs.get("fstype") not in _NFS_TYPES or scratch.get("fstype") not in _NFS_TYPES:
        raise ParallelExecutionError("storage gate evidence 未證明 scratch 位於 NFS")
    source_token = root_nfs.get("source_token_hash")
    if not isinstance(source_token, str) or not source_token:
        raise ParallelExecutionError("storage gate evidence 缺少 NFS source token")
    if any(roots[label].get("source_token_hash") != source_token for label in _GATE_ROOT_LABELS[1:]):
        raise ParallelExecutionError("storage gate evidence 根目錄並非同一個 NFS source")
    probes_raw = document.get("probes")
    if not isinstance(probes_raw, list):
        raise ParallelExecutionError("storage gate evidence 缺少寫入／鎖探針")
    probes = {
        row.get("label"): row
        for row in probes_raw
        if isinstance(row, dict) and isinstance(row.get("label"), str)
    }
    if any(
        not isinstance(probes.get(label), dict) or probes[label].get("gate_status") != "PASS"
        for label in ("write_probe", "same_host_flock_probe")
    ):
        raise ParallelExecutionError("storage gate evidence 寫入或 flock 探針未通過")
    return document, hashlib.sha256(raw).hexdigest()


def _live_mount_identity(path: Path) -> tuple[str, str]:
    """即時取得資料路徑的檔案系統型別與不可逆 NFS 來源 token。

    storage gate 快照刻意不保存 SERVER 的絕對路徑，因此 runner 不能只相信一份內容合法但
    可能屬於其他掛載點的舊快照。本函式在每次建立 log／cache 前以 ``findmnt`` 重新查詢
    實際 ``scratch_root``，選取涵蓋該路徑的最深掛載點，並使用和
    ``scripts/validate_server_storage.py`` 相同的 SHA-256 算法摘要 mount source。回傳值只
    在目前程序與 gate token 比對，不寫入原始 NFS 主機或 export 名稱。
    """

    try:
        completed = subprocess.run(
            [
                "findmnt",
                "--json",
                "--target",
                str(path),
                "--output",
                "FSTYPE,SOURCE,TARGET",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ParallelExecutionError("無法即時確認 scratch_root 的掛載來源") from exc
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ParallelExecutionError("scratch_root 的 findmnt JSON 無法解析") from exc
    filesystems = payload.get("filesystems") if isinstance(payload, Mapping) else None
    if not isinstance(filesystems, list) or not filesystems:
        raise ParallelExecutionError("scratch_root 的 findmnt 結果不完整")
    candidates: list[tuple[int, str, str]] = []
    for row in filesystems:
        if not isinstance(row, Mapping):
            continue
        fstype = row.get("fstype")
        source = row.get("source")
        target = row.get("target")
        if not all(isinstance(value, str) and value for value in (fstype, source, target)):
            continue
        try:
            resolved_target = Path(target).resolve(strict=False)
            path.relative_to(resolved_target)
        except (OSError, RuntimeError, ValueError):
            continue
        candidates.append((len(resolved_target.parts), fstype.lower(), source))
    if not candidates:
        raise ParallelExecutionError("scratch_root 沒有可驗證的涵蓋掛載點")
    _, fstype, source = max(candidates, key=lambda item: item[0])
    return fstype, hashlib.sha256(source.encode("utf-8")).hexdigest()


def validate_storage_roots(
    *,
    scratch_root: str | Path,
    log_root: str | Path,
    numba_cache_dir: str | Path | None,
    gate_evidence: str | Path,
) -> tuple[Path, Path, Path | None, str]:
    """確認日誌與 Numba cache 均位於通過儲存閘門的 scratch 樹。"""

    scratch = _strict_existing_directory(scratch_root, label="scratch_root")
    gate_document, gate_sha256 = _validate_storage_gate_snapshot(gate_evidence)
    gate_roots = {
        row.get("label"): row
        for row in gate_document["roots"]
        if isinstance(row, Mapping)
    }
    expected_source_token = gate_roots["scratch_root"].get("source_token_hash")
    live_fstype, live_source_token = _live_mount_identity(scratch)
    if live_fstype not in _NFS_TYPES:
        raise ParallelExecutionError("本次 scratch_root 的實際掛載不是 NFS")
    if live_source_token != expected_source_token:
        raise ParallelExecutionError("本次 scratch_root 與 storage gate 的 NFS 來源不一致")
    log_path, _ = _relative_to_root(log_root, scratch, label="log_root")
    # log_root 需先存在，避免任何隱含建立發生在 storage/provenance gate 之前。
    log = _strict_existing_directory(log_path, label="log_root")
    if log == scratch:
        raise ParallelExecutionError("log_root 必須是 scratch_root 的嚴格子目錄")
    cache: Path | None = None
    if numba_cache_dir is not None:
        cache_path, _ = _relative_to_root(numba_cache_dir, scratch, label="NUMBA_CACHE_DIR")
        if cache_path.exists():
            cache = _strict_existing_directory(cache_path, label="NUMBA_CACHE_DIR")
        else:
            # cache 可在首次 JIT 前建立；每一既有元件仍逐一拒絕 symlink，且只允許落在
            # 前一段已通過檢查的 scratch 樹內，不使用 Python 預設 HOME／tmp 路徑。
            cache = cache_path
        if cache == scratch:
            raise ParallelExecutionError("NUMBA_CACHE_DIR 必須是 scratch_root 的嚴格子目錄")
        try:
            cache.relative_to(log)
            overlap = True
        except ValueError:
            try:
                log.relative_to(cache)
                overlap = True
            except ValueError:
                overlap = False
        if overlap:
            raise ParallelExecutionError("NUMBA_CACHE_DIR 與 log_root 不可相等或巢狀")
    return scratch, log, cache, gate_sha256


def prepare_execution_directories(
    *,
    scratch_root: Path,
    log_root: Path,
    numba_cache_dir: Path | None,
    run_id: str,
) -> tuple[Path, Path | None]:
    """在所有唯讀 gate 通過後建立本批次 log session 與必要的 Numba cache 子目錄。"""

    scratch = _strict_existing_directory(scratch_root, label="scratch_root")
    log = _strict_existing_directory(log_root, label="log_root")
    _, log_parts = _relative_to_root(log, scratch, label="log_root")
    del log_parts
    cache: Path | None = None
    if numba_cache_dir is not None:
        cache_path, _ = _relative_to_root(numba_cache_dir, scratch, label="NUMBA_CACHE_DIR")
        cache = _create_verified_child_directory(
            scratch, cache_path, label="NUMBA_CACHE_DIR"
        )
        _write_probe(cache)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    session_name = f"{run_id}-{timestamp}-{secrets.token_hex(6)}"
    session = _create_verified_child_directory(
        log, log / session_name, label="parallel log session"
    )
    _write_probe(session)
    return session, cache


def _create_verified_child_directory(root: Path, candidate: Path, *, label: str) -> Path:
    """在已驗證 scratch 之下逐層建立目錄，並於每一層拒絕 symlink。"""

    _, parts = _relative_to_root(candidate, root, label=label)
    current = root
    for part in parts:
        current /= part
        try:
            current.mkdir()
        except FileExistsError:
            pass
        except OSError as exc:
            raise ParallelExecutionError(f"無法建立 {label} 目錄") from exc
        try:
            item = current.lstat()
        except OSError as exc:
            raise ParallelExecutionError(f"{label} 目錄建立後不可讀") from exc
        if stat.S_ISLNK(item.st_mode) or not stat.S_ISDIR(item.st_mode):
            raise ParallelExecutionError(f"{label} 目錄含符號連結或非目錄元件")
    return current.resolve(strict=True)


def _write_probe(directory: Path) -> None:
    """以實際建立、同步、原子改名及清理確認指定 NFS 目錄可寫。"""

    token = secrets.token_hex(12)
    temporary = directory / f".lbt-write-probe-{token}.partial"
    published = directory / f".lbt-write-probe-{token}.ready"
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = None
            stream.write(b"lbt-storage-probe-v1\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, published)
        published.unlink()
    except OSError as exc:
        raise ParallelExecutionError("明示 scratch 子目錄未通過實際寫入探針") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        for candidate in (temporary, published):
            with contextlib.suppress(OSError):
                candidate.unlink(missing_ok=True)


def _arrival_month_utc(value: object) -> str:
    """將 run plan 的整數 UTC 奈秒轉為不改變身分的 YYYY-MM locality 標籤。"""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ParallelExecutionError("run plan arrival_time_utc_ns 必須是整數")
    try:
        return datetime.fromtimestamp(value // 1_000_000_000, tz=UTC).strftime("%Y-%m")
    except (OverflowError, OSError, ValueError) as exc:
        raise ParallelExecutionError("run plan arrival UTC 無法轉換成月份") from exc


def _scenario_table_locality(
    workspace: Path, plan: Mapping[str, Any]
) -> tuple[dict[str, dict[str, tuple[str, ...]]], tuple[str, ...]]:
    """只讀既有情境表格的站點／流域識別欄位，不推測缺少的 flow domain。"""

    path = _strict_regular_file(workspace / "scenario_table.parquet", label="scenario_table")
    try:
        schema_names = set(pq.read_schema(path).names)
    except Exception as exc:  # noqa: BLE001 - parquet metadata 錯誤須在啟動子程序前停止
        raise ParallelExecutionError("無法讀取 scenario_table 欄位") from exc
    optional_columns = tuple(
        name for name in ("study_site_id", "flow_domain_id") if name in schema_names
    )
    if not optional_columns:
        return {}, ()
    try:
        table = pq.read_table(path, columns=list(optional_columns))
    except Exception as exc:  # noqa: BLE001 - 列內容錯誤須 fail closed
        raise ParallelExecutionError("無法讀取 scenario_table locality 欄位") from exc
    columns = {name: table[name].to_pylist() for name in optional_columns}
    result: dict[str, dict[str, tuple[str, ...]]] = {}
    for row in plan["shards"]:
        if not isinstance(row, Mapping):
            raise ParallelExecutionError("run plan shard row 必須是 object")
        start, stop = row.get("scenario_start_index"), row.get("scenario_stop_index")
        shard_id = row.get("shard_id")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(stop, bool)
            or not isinstance(stop, int)
            or not isinstance(shard_id, str)
        ):
            raise ParallelExecutionError("run plan shard range 或 ID 型別錯誤")
        values: dict[str, tuple[str, ...]] = {}
        for name, items in columns.items():
            subset = items[start:stop]
            if len(subset) != stop - start or any(
                not isinstance(item, str) or not item for item in subset
            ):
                raise ParallelExecutionError(f"scenario_table.{name} 與 shard range 不一致")
            values[name] = tuple(sorted(set(subset)))
        result[shard_id] = values
    return result, optional_columns


def build_worker_groups(
    plan: Mapping[str, Any],
    *,
    worker_count: int,
    scenario_locality: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
    locality_columns: Sequence[str] = (),
) -> tuple[WorkerGroup, ...]:
    """把全部 plan shard 確定性切成固定數量的連續 scenario 範圍。

    分組沿用 plan 原始順序；既有排序政策已令分析區域與到達時刻相鄰，因此本程序只
    依 region／UTC 月份標記 locality，再以 particle count 對連續區間做穩定平衡。站點或
    flow-domain 欄位若已存在於 scenario table，會放入摘要供稽核；它們不會觸發重新排序。
    若可用欄位缺少，只記錄 plan_order_only，不從 config、檔名或絕對路徑推測流域。
    """

    if isinstance(worker_count, bool) or not isinstance(worker_count, int) or worker_count < 1:
        raise ParallelExecutionError("worker_count 必須是正整數")
    rows = plan.get("shards")
    if not isinstance(rows, list) or not rows:
        raise ParallelExecutionError("run plan 必須含非空 shard 清單")
    if worker_count > len(rows):
        raise ParallelExecutionError("worker_count 不可大於 shard 數，避免建立空 worker")
    local_rows = scenario_locality or {}
    shard_values: list[dict[str, Any]] = []
    previous_start = -1
    previous_stop: int | None = None
    for row in rows:
        if not isinstance(row, Mapping):
            raise ParallelExecutionError("run plan shard row 必須是 object")
        shard_id = row.get("shard_id")
        start = row.get("scenario_start_index")
        stop = row.get("scenario_stop_index")
        particles = row.get("particle_count")
        region = row.get("analysis_region_id")
        month = _arrival_month_utc(row.get("arrival_time_utc_ns"))
        if (
            not isinstance(shard_id, str)
            or not shard_id
            or isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(stop, bool)
            or not isinstance(stop, int)
            or start < 0
            or stop <= start
            or start < previous_start
            or (previous_stop is not None and start != previous_stop)
            or isinstance(particles, bool)
            or not isinstance(particles, int)
            or particles < 1
            or not isinstance(region, str)
            or not region
        ):
            raise ParallelExecutionError("run plan shard ID、range 或計數不合法")
        previous_start = start
        previous_stop = stop
        locality = local_rows.get(shard_id, {})
        sites = tuple(sorted(set(locality.get("study_site_id", ()))))
        flows = tuple(sorted(set(locality.get("flow_domain_id", ()))))
        shard_values.append(
            {
                "shard_id": shard_id,
                "start": start,
                "stop": stop,
                "particles": particles,
                "locality": (region, month),
                "sites": sites,
                "flows": flows,
            }
        )

    # 對正整數工作量逐段選取最接近當前剩餘平均負載的切點；每段保留至少一片，
    # 並讓所有工作者取得原 scenario index 連續的區間。時間成本為 O(shard_count)。
    workers: list[WorkerGroup] = []
    cursor = 0
    unassigned_particles = sum(item["particles"] for item in shard_values)
    for worker_index in range(worker_count):
        workers_left = worker_count - worker_index
        maximum_end = len(shard_values) - (workers_left - 1)
        if workers_left == 1:
            end = len(shard_values)
        else:
            target = unassigned_particles / workers_left
            running = 0
            best_end = cursor + 1
            best_difference = float("inf")
            for end_candidate in range(cursor + 1, maximum_end + 1):
                running += shard_values[end_candidate - 1]["particles"]
                difference = abs(running - target)
                if difference < best_difference:
                    best_difference = difference
                    best_end = end_candidate
                if running >= target:
                    break
            end = best_end
        assigned = shard_values[cursor:end]
        if not assigned:
            raise ParallelExecutionError("分組演算法產生空 worker")
        ordered_localities = tuple(dict.fromkeys(item["locality"] for item in assigned))
        worker_sites = tuple(sorted({site for item in assigned for site in item["sites"]}))
        worker_flows = tuple(sorted({flow for item in assigned for flow in item["flows"]}))
        workers.append(
            WorkerGroup(
                worker_id=f"worker-{worker_index:04d}",
                shard_ids=tuple(item["shard_id"] for item in assigned),
                scenario_start_index=assigned[0]["start"],
                scenario_stop_index=assigned[-1]["stop"],
                particle_count=sum(item["particles"] for item in assigned),
                locality_keys=ordered_localities,
                study_site_ids=worker_sites,
                flow_domain_ids=worker_flows,
            )
        )
        unassigned_particles -= sum(item["particles"] for item in assigned)
        cursor = end
    if cursor != len(shard_values):
        raise ParallelExecutionError("分組未涵蓋 run plan 全部 shard")
    # locality_columns 影響外層 machine summary；分組本身永遠只用穩定 plan-order 欄位。
    del locality_columns
    return tuple(workers)


def worker_assignment_document(
    *,
    run_id: str,
    run_plan_sha256: str,
    worker_count: int,
    groups: Sequence[WorkerGroup],
    locality_columns: Sequence[str],
) -> dict[str, object]:
    """建立不包含完成時間與路徑的穩定分組文件，不會改寫科學 run identity。"""

    return {
        "schema_version": "1.0.0",
        "artifact_type": "formal_parallel_worker_assignment",
        "run_id": run_id,
        "run_plan_sha256": run_plan_sha256,
        "requested_worker_count": worker_count,
        "assignment_policy": "plan_order_contiguous_weighted_v1",
        "locality_policy": "analysis_region_utc_month_plan_order_v1",
        "locality_source_columns": list(locality_columns),
        "locality_fallback": "plan_order_only" if not locality_columns else None,
        "workers": [group.to_dict() for group in groups],
    }


def plan_cpu_affinity(
    request: str,
    *,
    worker_count: int,
    available_cpus: Sequence[int] | None,
) -> tuple[tuple[int, ...] | None, ...]:
    """以固定順序將可用 CPU 編號分給 worker；平台不支援時回傳未綁定。"""

    if request == "none":
        return tuple(None for _ in range(worker_count))
    if available_cpus is None:
        if request == "auto":
            return tuple(None for _ in range(worker_count))
        raise ParallelExecutionError(
            "目前平台無法讀取 CPU affinity；指定 CPU ID 時必須先能驗證可用 cpuset"
        )
    available = tuple(sorted(set(available_cpus)))
    if not available:
        return tuple(None for _ in range(worker_count))
    if request == "auto":
        selected = available
    else:
        try:
            selected = tuple(int(value.strip()) for value in request.split(","))
        except ValueError as exc:
            raise ParallelExecutionError(
                "--cpu-affinity 必須是 auto、none 或逗號分隔 CPU ID"
            ) from exc
        if not selected or len(set(selected)) != len(selected) or any(cpu < 0 for cpu in selected):
            raise ParallelExecutionError("--cpu-affinity CPU ID 不可空白、重複或為負數")
        if set(selected) - set(available):
            raise ParallelExecutionError("--cpu-affinity 包含不在目前 cpuset 的 CPU")
    groups: list[list[int]] = [[] for _ in range(worker_count)]
    for index, cpu in enumerate(selected):
        groups[index % worker_count].append(cpu)
    # worker 數超過 CPU 數時保留合法的空集合表示不綁定；不讓所有程序被綁到 CPU 0。
    return tuple(tuple(group) if group else None for group in groups)


def affinity_fallback_reason(
    request: str,
    *,
    cpu_ids: tuple[int, ...] | None,
    available_cpus: Sequence[int] | None,
) -> str | None:
    """說明未綁定 worker 的原因，避免把平台限制誤記成已套用 affinity。"""

    if request == "none":
        return None
    if available_cpus is None:
        return "unavailable_platform_fallback"
    if cpu_ids is None:
        return "no_cpu_assigned_fallback"
    return None


def current_available_cpus() -> tuple[int, ...] | None:
    """唯讀取得目前程序允許使用的 CPU 集合；非 Linux／平台缺 API 時回傳 None。"""

    getter = getattr(os, "sched_getaffinity", None)
    if getter is None:
        return None
    try:
        return tuple(sorted(getter(0)))
    except OSError:
        return None


def _worker_cli_arguments(
    *,
    workspace: Path,
    config: Path,
    group: WorkerGroup,
    ocm_native_root: Path | None,
    nww_analysis_root: Path | None,
    checkpoint_root: Path | None,
    resume: bool,
    sweep_budget: int | None,
    ocm_reconstruction_root: Path | None = None,
) -> list[str]:
    """將一個穩定 worker group 轉成唯一的一次 run-worker invocation。"""

    argv = ["run-worker", str(workspace), "--config", str(config)]
    for shard_id in group.shard_ids:
        argv.extend(("--shard-id", shard_id))
    if ocm_native_root is not None:
        argv.extend(("--ocm-native-root", str(ocm_native_root)))
    if ocm_reconstruction_root is not None:
        argv.extend(("--ocm-reconstruction-root", str(ocm_reconstruction_root)))
    if nww_analysis_root is not None:
        argv.extend(("--nww-analysis-root", str(nww_analysis_root)))
    if checkpoint_root is not None:
        argv.extend(("--checkpoint-root", str(checkpoint_root)))
    if resume:
        argv.append("--resume")
    if sweep_budget is not None:
        argv.extend(("--sweep-budget", str(sweep_budget)))
    return argv


def worker_child_main(spec: Mapping[str, Any], affinity: Mapping[str, object]) -> int:
    """在單一持久程序執行一次既有 run-worker 並回報 affinity／執行摘要。"""

    captured = io.StringIO()
    try:
        # 大型 CLI／科學模組只在 CPU affinity 與 NUMBA_CACHE_DIR 已由啟動環境設定後匯入；
        # run-worker 只建立一個 controller，再依序處理該 worker 的全部 shard。
        from .cli import main

        with contextlib.redirect_stdout(captured):
            child_exit = int(main(spec["cli_argv"]))
        output = captured.getvalue()
        try:
            child_summary: object = json.loads(output)
        except json.JSONDecodeError:
            child_summary = {"unparsed_stdout": output[-16_384:]}
        payload = {
            "artifact_type": "formal_parallel_worker_result",
            "schema_version": "1.0.0",
            "worker_id": spec["worker_id"],
            "affinity": dict(affinity),
            "numba_warmup_summary": spec.get("numba_warmup_summary"),
            "numba_disk_cache_reuse": False,
            "run_worker_exit_code": child_exit,
            "run_worker_summary": child_summary,
        }
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        return child_exit
    except Exception as exc:  # noqa: BLE001 - worker log 留下型別，主程序保留其他 worker 現場
        traceback.print_exc()
        payload = {
            "artifact_type": "formal_parallel_worker_result",
            "schema_version": "1.0.0",
            "worker_id": spec.get("worker_id"),
            "affinity": dict(affinity),
            "numba_warmup_summary": spec.get("numba_warmup_summary"),
            "numba_disk_cache_reuse": False,
            "run_worker_exit_code": 1,
            "run_worker_summary": None,
            "error_type": type(exc).__name__,
        }
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        return 1


def _parse_worker_log(path: Path) -> dict[str, Any] | None:
    """讀取 worker log 最後一筆 machine summary；中斷 log 保留為未解析狀態。"""

    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("artifact_type") == "formal_parallel_worker_result":
            return value
    return None


def _signal_process_group(process: subprocess.Popen[Any], signum: int) -> None:
    """只對本執行器建立的 worker process group 發送停止訊號。"""

    try:
        if os.name == "posix":
            os.killpg(process.pid, signum)
        else:
            process.send_signal(signum)
    except ProcessLookupError:
        pass
    except OSError:
        # process 若已自行退出，後續 wait/poll 仍會保存真實 exit code。
        pass


def _stop_active_processes(
    active: Mapping[str, subprocess.Popen[Any]], *, first_signal: int, grace_seconds: float
) -> dict[str, int | None]:
    """中止尚在執行的 worker，先給 checkpoint 正常收尾機會，再保留退出狀態。"""

    for process in active.values():
        if process.poll() is None:
            _signal_process_group(process, first_signal)
    deadline = time.monotonic() + grace_seconds
    exit_codes: dict[str, int | None] = {}
    while time.monotonic() < deadline:
        unfinished = False
        for worker_id, process in active.items():
            status = process.poll()
            if status is None:
                unfinished = True
            else:
                exit_codes[worker_id] = int(status)
        if not unfinished:
            return exit_codes
        time.sleep(0.05)
    for process in active.values():
        if process.poll() is None:
            _signal_process_group(process, signal.SIGTERM)
    second_deadline = time.monotonic() + min(5.0, grace_seconds)
    while time.monotonic() < second_deadline:
        unfinished = False
        for worker_id, process in active.items():
            status = process.poll()
            if status is None:
                unfinished = True
            else:
                exit_codes[worker_id] = int(status)
        if not unfinished:
            return exit_codes
        time.sleep(0.05)
    for worker_id, process in active.items():
        if process.poll() is None:
            _signal_process_group(process, signal.SIGKILL)
        try:
            exit_codes[worker_id] = int(process.wait(timeout=1.0))
        except (OSError, subprocess.TimeoutExpired):
            exit_codes[worker_id] = process.poll()
    return exit_codes


def execute_worker_groups(
    groups: Sequence[WorkerGroup],
    *,
    workspace: Path,
    project_root: Path,
    config: Path,
    ocm_native_root: Path | None,
    nww_analysis_root: Path | None,
    checkpoint_root: Path | None,
    numba_cache_dir: Path | None,
    log_session_dir: Path,
    resume: bool = False,
    sweep_budget: int | None = None,
    affinity_sets: Sequence[tuple[int, ...] | None] = (),
    affinity_request: str = "none",
    affinity_fallback_reasons: Sequence[str | None] = (),
    warmup_numba_backend: bool = False,
    poll_interval_seconds: float = 0.05,
    shutdown_grace_seconds: float = 30.0,
    popen_factory: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
    ocm_reconstruction_root: Path | None = None,
) -> tuple[list[dict[str, object]], int | None]:
    """啟動固定 worker groups、監看整機批次並於失敗／訊號時保留現場。"""

    if not groups:
        raise ParallelExecutionError("至少需要一個 worker group")
    if affinity_sets and len(affinity_sets) != len(groups):
        raise ParallelExecutionError("CPU affinity 分組數與 worker 數不一致")
    if affinity_fallback_reasons and len(affinity_fallback_reasons) != len(groups):
        raise ParallelExecutionError("CPU affinity fallback 摘要數與 worker 數不一致")
    if poll_interval_seconds < 0 or shutdown_grace_seconds < 0:
        raise ParallelExecutionError("worker 輪詢間隔與關閉寬限時間不可為負數")
    log_paths = [log_session_dir / f"{group.worker_id}.log" for group in groups]
    if any(path.exists() or path.is_symlink() for path in log_paths):
        raise ParallelExecutionError("worker log 已存在；拒絕覆寫既有現場")
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # child 必須匯入已通過 provenance 的本次 checkout，而不是同名舊版 site-package。
    env["PYTHONPATH"] = str(project_root / "src")
    if numba_cache_dir is not None:
        env["NUMBA_CACHE_DIR"] = str(numba_cache_dir)
    else:
        # 不讓服務帳號 shell 中預設的 HOME cache 沿用到純 NumPy worker。
        env.pop("NUMBA_CACHE_DIR", None)
    started_wall = time.perf_counter()
    active: dict[str, subprocess.Popen[Any]] = {}
    records: dict[str, dict[str, object]] = {}
    started_by_worker: dict[str, float] = {}
    failed = False
    interrupted_signal: int | None = None
    previous_handlers: dict[int, Any] = {}

    def on_signal(signum: int, frame: object) -> None:
        """將外部終止轉成可清理子程序的控制例外。"""

        del frame
        raise ParallelExecutionInterrupted(signum)

    managed_signals = [signal.SIGINT]
    if hasattr(signal, "SIGTERM"):
        managed_signals.append(signal.SIGTERM)
    for signum in managed_signals:
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, on_signal)
    launched_ids: set[str] = set()
    next_index = 0
    try:
        while next_index < len(groups) or active:
            # 以固定 worker ID 派發；每次啟動後稍作輪詢，若已知失敗就不再啟動後續組。
            if next_index < len(groups):
                group = groups[next_index]
                cpu_ids = affinity_sets[next_index] if affinity_sets else None
                affinity_fallback = (
                    affinity_fallback_reasons[next_index]
                    if affinity_fallback_reasons
                    else None
                )
                child_argv = _worker_cli_arguments(
                    workspace=workspace,
                    config=config,
                    group=group,
                    ocm_native_root=ocm_native_root,
                    ocm_reconstruction_root=ocm_reconstruction_root,
                    nww_analysis_root=nww_analysis_root,
                    checkpoint_root=checkpoint_root,
                    resume=resume,
                    sweep_budget=sweep_budget,
                )
                spec = {
                    "worker_id": group.worker_id,
                    "cpu_ids": list(cpu_ids) if cpu_ids is not None else None,
                    "affinity_request": affinity_request,
                    "affinity_fallback_reason": affinity_fallback,
                    "warmup_numba_backend": warmup_numba_backend,
                    "cli_argv": child_argv,
                }
                command = [
                    sys.executable,
                    "-c",
                    _CHILD_BOOTSTRAP,
                    json.dumps(spec, ensure_ascii=False, separators=(",", ":")),
                ]
                log_path = log_paths[next_index]
                try:
                    with log_path.open("x", encoding="utf-8") as log_stream:
                        process = popen_factory(
                            command,
                            cwd=project_root,
                            env=env,
                            stdout=log_stream,
                            stderr=subprocess.STDOUT,
                            start_new_session=(os.name == "posix"),
                        )
                except OSError as exc:
                    records[group.worker_id] = {
                        "worker_id": group.worker_id,
                        "status": "NOT_STARTED",
                        "exit_code": None,
                        "log_file": log_path.name,
                        "launch_error_type": type(exc).__name__,
                        "group": group.to_dict(),
                    }
                    failed = True
                    break
                active[group.worker_id] = process
                launched_ids.add(group.worker_id)
                records[group.worker_id] = {
                    "worker_id": group.worker_id,
                    "status": "RUNNING",
                    "exit_code": None,
                    "log_file": log_path.name,
                    "cpu_ids_requested": list(cpu_ids) if cpu_ids is not None else None,
                    "cpu_affinity_request": affinity_request,
                    "cpu_affinity_fallback_reason": affinity_fallback,
                    "child_elapsed_seconds": None,
                    "group": group.to_dict(),
                }
                started_by_worker[group.worker_id] = time.perf_counter()
                next_index += 1
                if poll_interval_seconds > 0:
                    time.sleep(poll_interval_seconds)

            for worker_id, process in tuple(active.items()):
                status = process.poll()
                if status is None:
                    continue
                status = int(status)
                active.pop(worker_id)
                records[worker_id]["exit_code"] = status
                records[worker_id]["child_elapsed_seconds"] = max(
                    0.0, time.perf_counter() - started_by_worker[worker_id]
                )
                records[worker_id]["worker_result"] = _parse_worker_log(
                    log_session_dir / str(records[worker_id]["log_file"])
                )
                records[worker_id]["status"] = "EXITED" if status == 0 else "FAILED"
                if status != 0:
                    failed = True
            if failed:
                break
            if active and next_index >= len(groups):
                time.sleep(poll_interval_seconds)
        if failed and active:
            exit_codes = _stop_active_processes(
                active,
                first_signal=signal.SIGTERM,
                grace_seconds=shutdown_grace_seconds,
            )
            for worker_id, process in active.items():
                status = exit_codes.get(worker_id, process.poll())
                records[worker_id]["exit_code"] = status
                records[worker_id]["child_elapsed_seconds"] = max(
                    0.0, time.perf_counter() - started_by_worker[worker_id]
                )
                records[worker_id]["status"] = "STOPPED_AFTER_PEER_FAILURE"
                records[worker_id]["worker_result"] = _parse_worker_log(
                    log_session_dir / str(records[worker_id]["log_file"])
                )
            active.clear()
    except ParallelExecutionInterrupted as exc:
        interrupted_signal = exc.signum
        if active:
            exit_codes = _stop_active_processes(
                active,
                first_signal=exc.signum,
                grace_seconds=shutdown_grace_seconds,
            )
            for worker_id, process in active.items():
                status = exit_codes.get(worker_id, process.poll())
                records[worker_id]["exit_code"] = status
                records[worker_id]["child_elapsed_seconds"] = max(
                    0.0, time.perf_counter() - started_by_worker[worker_id]
                )
                records[worker_id]["status"] = "STOPPED_BY_SIGNAL"
                records[worker_id]["worker_result"] = _parse_worker_log(
                    log_session_dir / str(records[worker_id]["log_file"])
                )
            active.clear()
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    for group in groups:
        if group.worker_id not in launched_ids:
            records.setdefault(
                group.worker_id,
                {
                    "worker_id": group.worker_id,
                    "status": "NOT_DISPATCHED",
                    "exit_code": None,
                    "log_file": None,
                    "group": group.to_dict(),
                },
            )
    total_wall = time.perf_counter() - started_wall
    result = [records[group.worker_id] for group in groups]
    for row in result:
        row.setdefault("whole_batch_wall_seconds", total_wall)
    return result, interrupted_signal


def validate_run_inputs(
    *,
    workspace: str | Path,
    config_path: str | Path,
    project_root: str | Path,
    checkpoint_root: str | Path | None,
) -> tuple[dict[str, Any], Any, Path, Path, Path | None]:
    """在任何子程序啟動前驗證 formal workspace、全分片、config 與部署 provenance。"""

    workspace_path = _strict_existing_directory(workspace, label="workspace")
    project_path = _strict_existing_directory(project_root, label="project_root")
    config_file = _strict_regular_file(config_path, label="config")
    checkpoint_path = (
        _strict_existing_directory(checkpoint_root, label="checkpoint_root")
        if checkpoint_root is not None
        else None
    )
    validation = validate_run(
        workspace_path,
        require_complete=False,
        checkpoint_root=checkpoint_path,
    )
    if not isinstance(validation, Mapping) or validation.get("valid") is not True:
        raise ParallelExecutionError(
            "run workspace／checkpoint pre-validation 未通過："
            + json.dumps(validation, ensure_ascii=False, sort_keys=True)
        )
    plan = load_run_plan(workspace_path)
    if plan.get("run_kind") != "formal":
        raise ParallelExecutionError("run-formal-parallel 只接受 run_kind=formal")
    shard_ids = [
        row.get("shard_id") for row in plan.get("shards", []) if isinstance(row, Mapping)
    ]
    if (
        len(shard_ids) != plan.get("shard_count")
        or not shard_ids
        or any(not isinstance(value, str) or not value for value in shard_ids)
        or len(set(shard_ids)) != len(shard_ids)
    ):
        raise ParallelExecutionError("run plan 全部 shard ID 不完整或重複")

    from .config import load_config

    config = load_config(config_file, formal_release=True)
    if config.config_hash() != plan.get("config_hash"):
        raise ParallelExecutionError("config hash 與 immutable run plan 不一致")
    try:
        normalized = json.loads(
            (workspace_path / "normalized_config.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ParallelExecutionError("workspace normalized_config 無法讀取") from exc
    if normalized != config.normalized_payload():
        raise ParallelExecutionError("workspace normalized_config 與目前 config 不一致")

    expected_provenance = plan.get("code_provenance")
    if not isinstance(expected_provenance, Mapping):
        raise ParallelExecutionError("run plan 缺少 code provenance")
    expected_commit = expected_provenance.get("git_commit")
    current_provenance = collect_code_provenance(
        project_path,
        declared_git_commit=expected_commit,
        formal=True,
    ).to_dict()
    if current_provenance != dict(expected_provenance):
        raise ParallelExecutionError("目前乾淨程式部署 provenance 與 run plan 不一致")
    return plan, config, workspace_path, project_path, checkpoint_path


def requires_numba_cache(config: object) -> bool:
    """從正式設定辨識 Numba backend，以要求安全 cache 路徑並啟動 worker 內暖機。

    目前 accelerated dispatcher 明確使用 ``cache=False``：設定此路徑是為了將 Numba
    環境固定在通過 gate 的 scratch，並避免繼承 HOME 預設值；它不表示編譯結果會寫到磁碟，
    worker 完成後也不會跨程序重用 JIT code。
    """

    execution = getattr(config, "execution", None)
    backend_values = (
        getattr(execution, "physics_kernel_backend", None),
        getattr(execution, "ocm_interpolation_backend", None),
    )
    return any(isinstance(value, str) and value.startswith("numba") for value in backend_values)


def warmup_numba_cache(
    *,
    numba_cache_dir: str | Path,
    scratch_root: str | Path,
    gate_evidence: str | Path,
    project_root: str | Path,
    log_file: str | Path,
) -> dict[str, object]:
    """用獨立程序檢查小型 JIT kernels 可編譯；此檢查不產生跨程序磁碟快取。"""

    scratch = _strict_existing_directory(scratch_root, label="scratch_root")
    cache_path, _ = _relative_to_root(numba_cache_dir, scratch, label="NUMBA_CACHE_DIR")
    _, gate_sha256 = _validate_storage_gate_snapshot(gate_evidence)
    cache = _create_verified_child_directory(scratch, cache_path, label="NUMBA_CACHE_DIR")
    _write_probe(cache)
    project = _strict_existing_directory(project_root, label="project_root")
    log_path = Path(log_file)
    log_parent = _strict_existing_directory(log_path.parent, label="warmup log directory")
    if log_path.parent != log_parent or log_path.exists() or log_path.is_symlink():
        raise ParallelExecutionError("warm-up log 必須是尚未建立的新普通檔案")
    env = os.environ.copy()
    env["NUMBA_CACHE_DIR"] = str(cache)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(project / "src")
    # 新程序會先設定 NUMBA_CACHE_DIR，再首次匯入 accelerated。現行核心 dispatcher
    # 使用 cache=False，不會產生 .nbc/.nbi；每個正式 worker 仍會在自身記憶體內另行暖機。
    try:
        with log_path.open("x", encoding="utf-8") as log_stream:
            completed = subprocess.run(
                [sys.executable, "-c", _NUMBA_WARMUP_BOOTSTRAP],
                cwd=project,
                env=env,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
    except OSError as exc:
        raise ParallelExecutionError("無法啟動獨立 Numba warm-up 程序") from exc
    if completed.returncode != 0:
        raise ParallelExecutionError(
            "Numba warm-up 檢查子程序失敗；請保留 parallel log session 內的 numba-warmup.log"
        )
    child_result: dict[str, Any] | None = None
    try:
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                decoded = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                isinstance(decoded, dict)
                and decoded.get("artifact_type") == "numba_warmup_child_result"
            ):
                child_result = decoded
    except OSError as exc:
        raise ParallelExecutionError("無法讀取 warm-up 子程序摘要") from exc
    if child_result is None:
        raise ParallelExecutionError(
            "Numba warm-up 未輸出機器可讀摘要；請保留 parallel log session 內的 numba-warmup.log"
        )
    return {
        "artifact_type": "numba_warmup_summary",
        "schema_version": "1.0.0",
        "status": "WARMUP_CHECKED",
        "disk_cache_enabled": False,
        "storage_gate_sha256": gate_sha256,
        "warmup_result": child_result.get("result"),
    }


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    """在已驗證的 log session 中以同目錄暫存檔原子保存批次摘要。"""

    if path.exists() or path.is_symlink():
        raise ParallelExecutionError("summary path 已存在；拒絕覆寫既有執行紀錄")
    encoded = json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.partial")
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = None
            stream.write(encoded)
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        raise ParallelExecutionError("無法原子保存 parallel summary") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        with contextlib.suppress(OSError):
            temporary.unlink(missing_ok=True)


def summarize_worker_completion(
    *,
    worker_records: Sequence[Mapping[str, object]],
    plan: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    """確認每個 worker 確實完整執行其分組，僅此時允許進入 final validator。"""

    errors: list[str] = []
    for record in worker_records:
        worker_id = str(record.get("worker_id"))
        expected_group = record.get("group")
        expected_ids = expected_group.get("shard_ids") if isinstance(expected_group, Mapping) else None
        worker_result = record.get("worker_result")
        if record.get("exit_code") != 0 or not isinstance(worker_result, Mapping):
            errors.append(f"{worker_id}: child_failed_or_summary_missing")
            continue
        if worker_result.get("artifact_type") != "formal_parallel_worker_result":
            errors.append(f"{worker_id}: wrapper_summary_type")
            continue
        run_payload = worker_result.get("run_worker_summary")
        if (
            not isinstance(run_payload, Mapping)
            or run_payload.get("artifact_type") != "run_worker_execution_summary"
        ):
            errors.append(f"{worker_id}: run_worker_summary_missing")
            continue
        if run_payload.get("run_id") != plan.get("run_id"):
            errors.append(f"{worker_id}: run_id_mismatch")
            continue
        if run_payload.get("requested_shard_ids") != expected_ids:
            errors.append(f"{worker_id}: shard_assignment_mismatch")
            continue
        executed = run_payload.get("executed_shards")
        if not isinstance(executed, list) or [
            item.get("shard_id") for item in executed if isinstance(item, Mapping)
        ] != expected_ids:
            errors.append(f"{worker_id}: incomplete_shard_execution")
            continue
        if any(
            not isinstance(item, Mapping) or item.get("lifecycle") != "COMPLETE"
            for item in executed
        ):
            errors.append(f"{worker_id}: shard_not_complete")
    return not errors, errors


def _input_root(
    explicit: str | Path | None,
    *,
    environment_name: str,
    label: str,
    required: bool,
) -> Path | None:
    """驗證 worker 將使用的 forcing 根目錄，避免子程序啟動後才發現路徑缺失。"""

    value: str | Path | None = explicit
    if value is None:
        value = os.environ.get(environment_name)
    if value is None or value == "":
        if required:
            raise ParallelExecutionError(f"缺少 {label}；請明示 CLI 路徑或設定 {environment_name}")
        return None
    return _strict_existing_directory(value, label=label)


def execute_formal_parallel(
    *,
    workspace: str | Path,
    config_path: str | Path,
    project_root: str | Path,
    worker_count: int,
    scratch_root: str | Path,
    log_root: str | Path,
    storage_gate_evidence: str | Path,
    checkpoint_root: str | Path | None = None,
    numba_cache_dir: str | Path | None = None,
    ocm_native_root: str | Path | None = None,
    ocm_reconstruction_root: str | Path | None = None,
    nww_analysis_root: str | Path | None = None,
    resume: bool = False,
    cpu_affinity: str = "auto",
    warmup_only: bool = False,
    popen_factory: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
    poll_interval_seconds: float = 0.05,
    shutdown_grace_seconds: float = 30.0,
) -> tuple[dict[str, object], int]:
    """驗證完整正式母體後，按 immutable plan 分組並執行持久 worker。

    此函式先驗證 workspace／全部 shard／checkpoint、正式設定與乾淨部署 provenance，
    再檢查實際 forcing 根目錄、NFS gate、log/cache 子路徑與 CPU affinity。任何 preflight
    失敗都會在建立子程序或新執行目錄之前停止。warmup-only 以獨立 JIT 子程序檢查可編譯性，
    其記憶體編譯結果不跨程序保留；正式模式每個 Numba worker 會在自身程序內暖機一次，接著
    僅啟動一次既有 run-worker，並以 run validator 作為唯一 COMPLETE 判定。
    """

    started_wall = time.perf_counter()
    if type(resume) is not bool or type(warmup_only) is not bool:
        raise ParallelExecutionError("resume 與 warmup_only 必須是 bool")
    if isinstance(worker_count, bool) or not isinstance(worker_count, int) or worker_count < 1:
        raise ParallelExecutionError("worker_count 必須是正整數")

    # run validator 先完整掃描 immutable plan、目前狀態、輸出與外部 checkpoint 拓撲；
    # formal provenance 再要求目前乾淨部署與 plan 完全相同，兩者都在啟動 child 前完成。
    plan, config, workspace_path, project_path, checkpoint_path = validate_run_inputs(
        workspace=workspace,
        config_path=config_path,
        project_root=project_root,
        checkpoint_root=checkpoint_root,
    )
    if worker_count > len(plan["shards"]):
        raise ParallelExecutionError("worker_count 不可大於 run plan shard 數")
    # 已完成、暫停或失敗過的任一 shard 都表示本 run 具有執行歷史；必須由操作者明示
    # resume，避免把部分成果誤當成初次執行而重開 checkpoint／亂數狀態。
    progress = load_run_progress(workspace_path)
    progress_shards = progress.get("shards")
    if not isinstance(progress_shards, Mapping) or set(progress_shards) != {
        row["shard_id"] for row in plan["shards"]
    }:
        raise ParallelExecutionError("run progress 與 immutable plan 的全部 shard ID 不一致")
    if progress.get("run_lifecycle") == "COMPLETE":
        raise ParallelExecutionError("run 已為 COMPLETE；請勿再次啟動同一正式母體")
    if not resume and any(
        not isinstance(row, Mapping) or row.get("lifecycle") != "PLANNED"
        for row in progress_shards.values()
    ):
        raise ParallelExecutionError("run 已有 shard 執行狀態；續跑時必須明示 --resume")

    input_config = config.inputs
    ocm_root = _input_root(
        ocm_native_root,
        environment_name=input_config.ocm_native_root_env,
        label="OCM native root",
        required=True,
    )
    nww_root = _input_root(
        nww_analysis_root,
        environment_name=input_config.nww_analysis_root_env,
        label="NWW3 analysis root",
        required=False,
    )
    reconstruction_required = bool(input_config.ocm_gap_reconstruction_manifest)
    reconstruction_root = _input_root(
        ocm_reconstruction_root,
        environment_name=input_config.ocm_reconstruction_root_env or "OCM_RECONSTRUCTION_ROOT",
        label="OCM reconstruction root",
        required=reconstruction_required,
    )

    # 情境表只讀已存在的站點／流域識別欄位；目前 run plan 固定提供區域與 UTC 到達時刻，
    # 因而可按原情境順序標示 region/month。缺少表格欄位時不推測，assignment 會記錄 fallback。
    scenario_locality, locality_columns = _scenario_table_locality(workspace_path, plan)
    groups = build_worker_groups(
        plan,
        worker_count=worker_count,
        scenario_locality=scenario_locality,
        locality_columns=locality_columns,
    )
    cache_value = numba_cache_dir
    if requires_numba_cache(config) and cache_value is None:
        raise ParallelExecutionError(
            "所選 Numba backend 必須明示 --numba-cache-dir，且目錄須位於通過 gate 的 scratch root"
        )
    if warmup_only and (not requires_numba_cache(config) or cache_value is None):
        raise ParallelExecutionError(
            "--warmup-only 僅適用於 run plan 明示 Numba backend 並指定 --numba-cache-dir"
        )

    # storage snapshot 必須證明全部受管根目錄位於同一 NFS source，實際使用的 scratch、
    # log 與 cache 路徑再逐元件拒絕 symlink；正式 SERVER 不把 cache 寫入 /home 或系統 tmp。
    scratch_path, log_path, cache_path, gate_sha256 = validate_storage_roots(
        scratch_root=scratch_root,
        log_root=log_root,
        numba_cache_dir=cache_value,
        gate_evidence=storage_gate_evidence,
    )
    available_cpus = current_available_cpus()
    affinity_sets = plan_cpu_affinity(
        cpu_affinity,
        worker_count=worker_count,
        available_cpus=available_cpus,
    )
    affinity_fallbacks = tuple(
        affinity_fallback_reason(
            cpu_affinity,
            cpu_ids=cpu_ids,
            available_cpus=available_cpus,
        )
        for cpu_ids in affinity_sets
    )

    # 只有所有唯讀檢查完成後才建立本批次獨立 log session／cache；NFS 寫入探針亦會在
    # 子程序啟動前完成。每次執行使用不可碰撞目錄，不覆寫舊摘要或失敗現場。
    run_id = plan["run_id"]
    log_session, prepared_cache = prepare_execution_directories(
        scratch_root=scratch_path,
        log_root=log_path,
        numba_cache_dir=cache_path,
        run_id=run_id,
    )
    plan_bytes = (workspace_path / "run_plan.json").read_bytes()
    plan_sha256 = hashlib.sha256(plan_bytes).hexdigest()
    assignment = worker_assignment_document(
        run_id=run_id,
        run_plan_sha256=plan_sha256,
        worker_count=worker_count,
        groups=groups,
        locality_columns=locality_columns,
    )
    atomic_write_json(log_session / "worker-assignment.json", assignment)

    if warmup_only:
        try:
            warmup_summary = warmup_numba_cache(
                numba_cache_dir=prepared_cache,
                scratch_root=scratch_path,
                gate_evidence=storage_gate_evidence,
                project_root=project_path,
                log_file=log_session / "numba-warmup.log",
            )
        except ParallelExecutionError as exc:
            summary: dict[str, object] = {
                "artifact_type": "formal_parallel_execution_summary",
                "schema_version": "1.0.0",
                "run_id": run_id,
                "run_plan_sha256": plan_sha256,
                "status": "WARMUP_FAILED",
                "valid": False,
                "whole_machine_elapsed_seconds": time.perf_counter() - started_wall,
                "storage_gate_sha256": gate_sha256,
                "worker_assignment": assignment,
                "worker_records": [],
                "error": str(exc),
                "log_session": log_session.name,
            }
            atomic_write_json(log_session / "summary.json", summary)
            return summary, 2
        summary = {
            "artifact_type": "formal_parallel_execution_summary",
            "schema_version": "1.0.0",
            "run_id": run_id,
            "run_plan_sha256": plan_sha256,
            "status": "WARMUP_CHECKED",
            "valid": True,
            "numba_disk_cache_enabled": False,
            "numba_warmup_scope": "one_off_validation_process",
            "whole_machine_elapsed_seconds": time.perf_counter() - started_wall,
            "storage_gate_sha256": gate_sha256,
            "worker_assignment": assignment,
            "worker_records": [],
            "warmup_summary": warmup_summary,
            "log_session": log_session.name,
        }
        atomic_write_json(log_session / "summary.json", summary)
        return summary, 0

    worker_records, interrupted_signal = execute_worker_groups(
        groups,
        workspace=workspace_path,
        project_root=project_path,
        config=_strict_regular_file(config_path, label="config"),
        ocm_native_root=ocm_root,
        ocm_reconstruction_root=reconstruction_root,
        nww_analysis_root=nww_root,
        checkpoint_root=checkpoint_path,
        numba_cache_dir=prepared_cache,
        log_session_dir=log_session,
        resume=resume,
        affinity_sets=affinity_sets,
        affinity_request=cpu_affinity,
        affinity_fallback_reasons=affinity_fallbacks,
        warmup_numba_backend=requires_numba_cache(config),
        poll_interval_seconds=poll_interval_seconds,
        shutdown_grace_seconds=shutdown_grace_seconds,
        popen_factory=popen_factory,
    )
    worker_complete, completion_errors = summarize_worker_completion(
        worker_records=worker_records,
        plan=plan,
    )
    final_validation: dict[str, Any] | None = None
    status = "INTERRUPTED" if interrupted_signal is not None else "INCOMPLETE"
    exit_code = 130 if interrupted_signal == signal.SIGINT else 143 if interrupted_signal else 2
    if interrupted_signal is None and worker_complete:
        final_validation = validate_run(
            workspace_path,
            require_complete=True,
            checkpoint_root=checkpoint_path,
        )
        if final_validation.get("valid") is True:
            status = "COMPLETE"
            exit_code = 0
        else:
            status = "FAILED_FINAL_VALIDATION"

    summary = {
        "artifact_type": "formal_parallel_execution_summary",
        "schema_version": "1.0.0",
        "run_id": run_id,
        "run_plan_sha256": plan_sha256,
        "status": status,
        "valid": status == "COMPLETE",
        "numba_disk_cache_enabled": False,
        "numba_warmup_scope": (
            "per_worker_process" if requires_numba_cache(config) else "not_applicable"
        ),
        "whole_machine_elapsed_seconds": time.perf_counter() - started_wall,
        "storage_gate_sha256": gate_sha256,
        "requested_worker_count": worker_count,
        "worker_assignment": assignment,
        "worker_records": worker_records,
        "worker_completion_errors": completion_errors,
        "interrupted_signal": interrupted_signal,
        "final_validation": final_validation,
        "log_session": log_session.name,
    }
    atomic_write_json(log_session / "summary.json", summary)
    return summary, exit_code


__all__ = [
    "ParallelExecutionError",
    "ParallelExecutionInterrupted",
    "WorkerGroup",
    "affinity_fallback_reason",
    "atomic_write_json",
    "build_worker_groups",
    "canonical_json_bytes",
    "current_available_cpus",
    "execute_worker_groups",
    "execute_formal_parallel",
    "plan_cpu_affinity",
    "prepare_execution_directories",
    "requires_numba_cache",
    "summarize_worker_completion",
    "validate_run_inputs",
    "validate_storage_roots",
    "warmup_numba_cache",
    "worker_assignment_document",
]
