"""正式 run 的 parent-queue dynamic persistent worker 執行器。

本模組只處理工程排程與 JSON-line IPC：不可變 run plan、scenario／particle 身分、
seed、checkpoint、lock、QC 與物理計算仍由既有 ``RunController`` 和
``ProductionBatch`` 執行。parent 只維護一份尚未派出的 shard queue；每個 child 啟動
一次 controller，之後一次接收一片、回傳一片，直到 parent 明確送出 shutdown。這個
測試／正式共用的 dynamic backend 不新增 checkpoint schema，也不把 locality hint 當成
實際派發證據。
"""

from __future__ import annotations

import contextlib
import json
import os
import secrets
import select
import signal
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from .parallel_execution import (
    FormalParallelContext,
    ParallelExecutionError,
    ParallelExecutionInterrupted,
    _arrival_month_utc,
    _stop_active_processes,
    atomic_write_json,
    requires_numba_cache,
    warmup_numba_cache,
)
from .run_control import load_run_progress
from .run_validation import validate_run

PROTOCOL_NAME = "lbt_dynamic_worker_v1"
"""parent／persistent child 之間的固定 JSON-line 協定名稱。"""

_LIFECYCLES = frozenset({"COMPLETE", "PAUSED", "FAILED"})
_STOPPED_INFLIGHT = "STOPPED_INFLIGHT"
_MIN_MEM_AVAILABLE_BYTES = 30 * 1024**3


class DynamicProtocolError(ParallelExecutionError):
    """child 回傳 EOF、非 JSON 或不符合 dynamic IPC schema 時的固定錯誤型別。"""


def _memory_available_bytes() -> int | None:
    """讀取 Linux ``MemAvailable``，供 parent 在高併發時保留安全緩衝。

    讀取失敗或非 Linux 時回傳 ``None``，不把不可觀測誤判成零記憶體；只有取得明確且
    低於 30 GiB 的數值才停止新派工並收尾 in-flight shards。此閘門不改變任何 run plan。
    """

    try:
        raw = Path("/proc/meminfo").read_text(encoding="ascii")
    except (OSError, UnicodeDecodeError):
        return None
    for line in raw.splitlines():
        if not line.startswith("MemAvailable:"):
            continue
        fields = line.split()
        if len(fields) >= 2:
            try:
                return int(fields[1]) * 1024
            except ValueError:
                return None
    return None


@dataclass(frozen=True, slots=True)
class DynamicShard:
    """parent queue 中一個 immutable shard 的最小排程摘要。

    locality 只由 run plan 的 analysis region 與 arrival UTC month 組成；它用來讓同一
    worker 優先接續相近資料，不能改變 shard ID、scenario range、particle seed 或
    checkpoint 邊界。所有物理欄位仍由 child 的既有 controller 從正式資料讀取。
    """

    shard_id: str
    locality: tuple[str, str]


@dataclass(slots=True)
class _WorkerState:
    """parent 追蹤一個 persistent child 的程序、目前 in-flight shard 與 log。"""

    worker_id: str
    process: subprocess.Popen[Any]
    log_stream: TextIO
    log_file: str
    affinity: dict[str, object]
    inflight: DynamicShard | None = None
    dispatched_at: float | None = None
    last_locality: tuple[str, str] | None = None
    # 直接從 pipe 的檔案描述元讀 bytes，避免 TextIOWrapper.readline 在只收到半行時
    # 阻塞 parent。每個 worker 保留自己的 buffer，讓同一次 read 讀到的多筆 JSON-line
    # 依序解析；它不改變 shard dispatch 順序或 checkpoint 內容。
    response_buffer: bytearray = field(default_factory=bytearray)
    response_eof: bool = False


_CHILD_BOOTSTRAP = r"""
import json
import os
import sys
import contextlib

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
    # cache=False dispatcher 的編譯結果只存在本 child 記憶體，不宣稱跨 worker 共用。
    from lagrangian_backtracking.accelerated import warmup_numba_backend
    # 暖機可能由既有加速模組輸出診斷；protocol stdout 必須只留給 JSON-line。
    with contextlib.redirect_stdout(sys.stderr):
        spec["numba_warmup_summary"] = warmup_numba_backend()

from lagrangian_backtracking.dynamic_execution import dynamic_worker_child_main
raise SystemExit(dynamic_worker_child_main(spec, sys.stdin, sys.stdout, sys.stderr, affinity))
"""


def _json_line(stream: TextIO, payload: Mapping[str, object]) -> None:
    """以單行、固定 key 順序寫入 protocol；stdout 不承載其他診斷文字。"""

    stream.write(
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )
    stream.flush()


def _protocol_error_payload(
    *,
    spec: Mapping[str, Any],
    error_code: str,
    error: str,
    shard_id: str | None = None,
    fatal: bool = True,
) -> dict[str, object]:
    """建立不包含 traceback／私有路徑的 machine-readable child error。"""

    return {
        "protocol": PROTOCOL_NAME,
        "message_type": "worker_error",
        "run_id": spec.get("run_id"),
        "worker_id": spec.get("worker_id"),
        "shard_id": shard_id,
        "error_code": error_code,
        "error_type": "DynamicProtocolError",
        "error": error,
        "fatal": fatal,
    }


def _worker_exception_is_fatal(exc: BaseException) -> bool:
    """辨識不可安全繼續派片的 worker 例外。

    execution-priority 只隔離單片物理／數值例外；記憶體不足、訊號、鎖／結構性陣列或
    網格錯誤代表 child 的共用 controller／程序環境可能已失效，必須停止整個 parent。
    其餘例外由 parent 記錄為單片 FAILED，讓同一個已初始化 controller 繼續服務後續片。
    這是保守的文字分類，實際正式 controller 若提供 ``fatal_for_dynamic`` 屬性則優先採用。
    """

    if getattr(exc, "fatal_for_dynamic", False) is True:
        return True
    if isinstance(exc, (MemoryError, KeyboardInterrupt, SystemExit)):
        return True
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    fatal_tokens = (
        "lockbusy",
        "runlock",
        "dimension",
        "shape",
        "grid",
        "array",
        "網格",
        "維度",
        "記憶體",
        "memory",
        "memavailable",
    )
    return any(token in name or token in message for token in fatal_tokens)


def dynamic_worker_child_main(
    spec: Mapping[str, Any],
    stdin: TextIO,
    stdout: TextIO,
    stderr: TextIO,
    affinity: Mapping[str, object] | None = None,
) -> int:
    """在一個 persistent child 內處理 parent JSON-line request。

    ``spec`` 只含本次 immutable run 的路徑與 binding；controller 在第一個合法
    ``run_shard`` request 建立一次，後續 request 共用它。每個 request 恰執行一片，
    child 不自行挑片、不重派片，也不把錯誤吞成成功。``affinity`` 由 bootstrap 在
    匯入大型 runtime 前設定，僅放入 in-memory error/result context。
    """

    if spec.get("protocol") not in {None, PROTOCOL_NAME}:
        _json_line(
            stdout,
            _protocol_error_payload(
                spec=spec,
                error_code="child_spec_protocol_mismatch",
                error="child spec protocol 不相容",
            ),
        )
        return 2
    controller: Any = None
    controller_resume: bool | None = None
    worker_id = spec.get("worker_id")
    run_id = spec.get("run_id")
    execution_priority = spec.get("execution_priority", False)
    if type(execution_priority) is not bool:
        _json_line(
            stdout,
            _protocol_error_payload(
                spec=spec,
                error_code="child_spec_execution_priority_invalid",
                error="child spec execution_priority 必須是 bool",
            ),
        )
        return 2
    defer_full_validation = spec.get("defer_full_validation", False)
    if type(defer_full_validation) is not bool:
        _json_line(
            stdout,
            _protocol_error_payload(
                spec=spec,
                error_code="child_spec_defer_full_validation_invalid",
                error="child spec defer_full_validation 必須是 bool",
            ),
        )
        return 2
    if defer_full_validation and not execution_priority:
        _json_line(
            stdout,
            _protocol_error_payload(
                spec=spec,
                error_code="child_spec_defer_full_validation_requires_priority",
                error="child spec defer_full_validation 需要 execution_priority=true",
            ),
        )
        return 2
    if type(worker_id) is not str or not worker_id or type(run_id) is not str or not run_id:
        _json_line(
            stdout,
            _protocol_error_payload(
                spec=spec,
                error_code="child_spec_identity_invalid",
                error="child spec 缺少合法 run_id 或 worker_id",
            ),
        )
        return 2

    try:
        while True:
            line = stdin.readline()
            if line == "":
                # parent EOF 是 bounded shutdown 的一種正常結束；若仍有 in-flight，
                # parent 會依其自己的 assignment record 將它列為未完成，不由 child 重派。
                return 0
            try:
                request = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                _json_line(
                    stdout,
                    _protocol_error_payload(
                        spec=spec,
                        error_code="invalid_json_line",
                        error=f"request JSON 無法解析：{type(exc).__name__}",
                    ),
                )
                return 2
            if not isinstance(request, Mapping):
                _json_line(
                    stdout,
                    _protocol_error_payload(
                        spec=spec,
                        error_code="request_not_object",
                        error="request 必須是 JSON object",
                    ),
                )
                return 2
            if request.get("protocol") != PROTOCOL_NAME:
                _json_line(
                    stdout,
                    _protocol_error_payload(
                        spec=spec,
                        error_code="request_protocol_mismatch",
                        error="request protocol 不相容",
                        shard_id=(
                            request.get("shard_id")
                            if isinstance(request.get("shard_id"), str)
                            else None
                        ),
                    ),
                )
                return 2
            message_type = request.get("message_type")
            if message_type == "shutdown":
                _json_line(
                    stdout,
                    {
                        "protocol": PROTOCOL_NAME,
                        "message_type": "shutdown_ack",
                        "run_id": run_id,
                        "worker_id": worker_id,
                    },
                )
                return 0
            if message_type != "run_shard":
                _json_line(
                    stdout,
                    _protocol_error_payload(
                        spec=spec,
                        error_code="unknown_message_type",
                        error="message_type 只允許 run_shard 或 shutdown",
                    ),
                )
                return 2

            request_run_id = request.get("run_id")
            request_worker_id = request.get("worker_id")
            shard_id = request.get("shard_id")
            resume = request.get("resume")
            sweep_budget = request.get("sweep_budget")
            if (
                request_run_id != run_id
                or request_worker_id != worker_id
                or type(shard_id) is not str
                or not shard_id
                or type(resume) is not bool
                or (
                    sweep_budget is not None
                    and (
                        isinstance(sweep_budget, bool)
                        or not isinstance(sweep_budget, int)
                    )
                )
            ):
                _json_line(
                    stdout,
                    _protocol_error_payload(
                        spec=spec,
                        error_code="request_fields_invalid",
                        error="run_id、worker_id、shard_id、resume 或 sweep_budget 不合法",
                        shard_id=shard_id if isinstance(shard_id, str) else None,
                    ),
                )
                return 2
            if controller_resume is None:
                controller_resume = resume
            elif controller_resume != resume:
                _json_line(
                    stdout,
                    _protocol_error_payload(
                        spec=spec,
                        error_code="request_resume_mismatch",
                        error="同一 persistent child 不可切換 controller resume binding",
                        shard_id=shard_id,
                    ),
                )
                return 2

            if controller is None:
                # 延遲一次 controller 建立，確保 child 啟動本身不會在 parent 尚未派片時
                # 讀 forcing 或碰 shard lock；之後同一 controller 服務所有 request。
                try:
                    from .cli import open_worker_controller

                    required_spec_fields = (
                        "workspace",
                        "config_path",
                        "ocm_native_root",
                    )
                    if any(
                        not isinstance(spec.get(name), str) or not spec.get(name)
                        for name in required_spec_fields
                    ):
                        raise DynamicProtocolError(
                            "child spec 缺少 workspace、config_path 或 ocm_native_root"
                        )
                    # controller 建立流程可能沿用既有 print 診斷；只把這些文字送到
                    # child log，不能污染 parent 以 stdout 解析的 protocol stream。
                    with contextlib.redirect_stdout(stderr):
                        controller_kwargs = {
                            "config_path": spec["config_path"],
                            "ocm_native_root": spec["ocm_native_root"],
                            "nww_analysis_root": spec.get("nww_analysis_root"),
                            "ocm_reconstruction_root": spec.get("ocm_reconstruction_root"),
                            "resume": resume,
                            "checkpoint_root": spec.get("checkpoint_root"),
                        }
                        if bool(spec.get("continuous_time_fallback", False)):
                            controller_kwargs["continuous_time_fallback"] = True
                        # deferred policy 只由 coordinator 的 private dynamic bootstrap 傳入；
                        # 預設 false 時省略新 keyword，保留 run-worker 與舊 controller mock
                        # 的原有呼叫形狀。controller 自己仍會在每片保留 assigned shard
                        # checkpoint／restore／output／hash／lock 驗證。
                        if defer_full_validation:
                            from .runtime import _DEFERRED_VALIDATION_PRIVATE_TOKEN

                            controller_kwargs["_defer_full_validation"] = True
                            controller_kwargs["_private_dynamic_bootstrap"] = (
                                _DEFERRED_VALIDATION_PRIVATE_TOKEN
                            )
                        controller = open_worker_controller(
                            spec["workspace"],
                            **controller_kwargs,
                        )
                except BaseException as exc:  # noqa: BLE001 - protocol 必須明確終止 child
                    traceback.print_exc(file=stderr)
                    _json_line(
                        stdout,
                        {
                            "protocol": PROTOCOL_NAME,
                            "message_type": "worker_error",
                            "run_id": run_id,
                            "worker_id": worker_id,
                            "shard_id": shard_id,
                            "error_code": "controller_setup_exception",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "fatal": True,
                        },
                    )
                    return 1

            try:
                from .cli import execute_worker_shard

                # 單片 controller 執行也可能輸出歷史診斷；維持 protocol stdout 的
                # 獨立 handle，只讓 JSON-line writer 寫入它。
                with contextlib.redirect_stdout(stderr):
                    summary = execute_worker_shard(
                        controller,
                        shard_id,
                        sweep_budget=sweep_budget,
                    )
            except BaseException as exc:  # noqa: BLE001 - parent 需取得可機讀失敗邊界
                traceback.print_exc(file=stderr)
                fatal = _worker_exception_is_fatal(exc)
                _json_line(
                    stdout,
                    {
                        "protocol": PROTOCOL_NAME,
                        "message_type": "worker_error",
                        "run_id": run_id,
                        "worker_id": worker_id,
                        "shard_id": shard_id,
                        "error_code": "run_shard_exception",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "fatal": fatal,
                    },
                )
                # execution-priority 下單片例外不應殺掉已完成初始化的 controller；
                # parent 會將此片記為 FAILED 並立即把下一片送入同一 child。結構性／
                # 記憶體／鎖錯誤則以 fatal 結束，避免污染後續片的結果。
                if bool(spec.get("execution_priority", False)) and not fatal:
                    continue
                return 1
            lifecycle = summary.get("lifecycle")
            if lifecycle not in _LIFECYCLES:
                _json_line(
                    stdout,
                    _protocol_error_payload(
                        spec=spec,
                        error_code="summary_lifecycle_invalid",
                        error="RunExecutionSummary.lifecycle 不在既有狀態集合",
                        shard_id=shard_id,
                    ),
                )
                return 2
            _json_line(
                stdout,
                {
                    "protocol": PROTOCOL_NAME,
                    "message_type": "shard_result",
                    "run_id": run_id,
                    "worker_id": worker_id,
                    "shard_id": shard_id,
                    "lifecycle": lifecycle,
                    "exit_code": 0 if lifecycle != "FAILED" else 1,
                    "summary": summary,
                    "affinity": dict(affinity or {}),
                    "numba_warmup_summary": spec.get("numba_warmup_summary"),
                },
            )
    except (BrokenPipeError, OSError) as exc:
        # stdout/stderr 關閉時不再嘗試寫第二份錯誤；stderr 仍保留 child 本地診斷。
        print(f"dynamic child I/O failure: {type(exc).__name__}: {exc}", file=stderr)
        return 1


def execute_dynamic_parallel(
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
    response_timeout_seconds: float | None = None,
    execution_priority: bool = False,
    defer_full_validation: bool = False,
) -> tuple[dict[str, object], int]:
    """薄包裝既有 formal coordinator，避免在 dynamic module 複製完整 preflight。

    coordinator 會先呼叫共用 input／provenance／storage gate；只有選定 dynamic 後才
    把已準備好的 context 交給本模組。這個 wrapper 不直接讀 plan、不開 controller，
    因而不可能繞過既有安全閘門。
    """

    from .parallel_execution import execute_formal_parallel

    return execute_formal_parallel(
        workspace=workspace,
        config_path=config_path,
        project_root=project_root,
        worker_count=worker_count,
        scratch_root=scratch_root,
        log_root=log_root,
        storage_gate_evidence=storage_gate_evidence,
        checkpoint_root=checkpoint_root,
        numba_cache_dir=numba_cache_dir,
        ocm_native_root=ocm_native_root,
        ocm_reconstruction_root=ocm_reconstruction_root,
        nww_analysis_root=nww_analysis_root,
        resume=resume,
        cpu_affinity=cpu_affinity,
        warmup_only=warmup_only,
        popen_factory=popen_factory,
        poll_interval_seconds=poll_interval_seconds,
        shutdown_grace_seconds=shutdown_grace_seconds,
        execution_mode="dynamic",
        response_timeout_seconds=response_timeout_seconds,
        execution_priority=execution_priority,
        defer_full_validation=defer_full_validation,
    )


def _pending_shards(context: FormalParallelContext) -> tuple[list[DynamicShard], set[str]]:
    """讀取 reconcile 後 progress，只排入尚未 COMPLETE 的 immutable shard。"""

    progress = load_run_progress(context.workspace_path)
    rows = progress.get("shards")
    if not isinstance(rows, Mapping):
        raise ParallelExecutionError("dynamic progress shards 必須是 mapping")
    pending: list[DynamicShard] = []
    complete: set[str] = set()
    for row in context.plan["shards"]:
        shard_id = row.get("shard_id") if isinstance(row, Mapping) else None
        if not isinstance(shard_id, str) or not shard_id:
            raise ParallelExecutionError("run plan shard_id 不合法")
        progress_row = rows.get(shard_id)
        if not isinstance(progress_row, Mapping):
            raise ParallelExecutionError(f"progress 缺少 shard：{shard_id}")
        lifecycle = progress_row.get("lifecycle")
        if lifecycle == "COMPLETE":
            complete.add(shard_id)
            continue
        if lifecycle not in {"PLANNED", "RUNNING", "PAUSED", "FAILED"}:
            raise ParallelExecutionError(f"dynamic 不支援 shard lifecycle：{shard_id}")
        region = row.get("analysis_region_id")
        month = _arrival_month_utc(row.get("arrival_time_utc_ns"))
        if not isinstance(region, str) or not region:
            raise ParallelExecutionError(f"shard locality region 不合法：{shard_id}")
        pending.append(DynamicShard(shard_id=shard_id, locality=(region, month)))
    return pending, complete


def _send_request(worker: _WorkerState, payload: Mapping[str, object]) -> None:
    """把單一 request 寫入 child；BrokenPipe 交給 parent 統一 fail-safe 收尾。"""

    if worker.process.stdin is None:
        raise DynamicProtocolError(f"{worker.worker_id} stdin 不存在")
    _json_line(worker.process.stdin, payload)


def _read_response_line(
    worker: _WorkerState,
    *,
    wait_seconds: float,
) -> tuple[bool, object | None, str | None]:
    """以非阻塞 bytes buffer 讀取一筆完整 child stdout JSON-line。

    ``select`` 只保證 pipe 目前至少有一個 byte 或已經 EOF，不能保證 newline 已到達；
    因此不可在 select 後呼叫 ``TextIOWrapper.readline``。這裡直接使用 OS file
    descriptor 的 ``os.read``，將 partial line 留在 worker 專屬 buffer，直到完整 newline
    才解析 JSON。若 EOF 時仍有殘行，明確回傳協定錯誤，避免把截斷結果當成成功。沒有
    file descriptor 的 StringIO 測試替身則以不會等待的 ``read`` 補入同一個 bytes buffer。
    """

    stream = worker.process.stdout
    if stream is None:
        return True, None, "stdout_missing"

    def consume_buffer() -> tuple[bool, object | None, str | None]:
        """只消費一個完整 newline；尚未完整時讓 parent 下輪繼續 poll。"""

        newline_index = worker.response_buffer.find(b"\n")
        if newline_index >= 0:
            raw_line = bytes(worker.response_buffer[:newline_index])
            del worker.response_buffer[: newline_index + 1]
            try:
                return True, json.loads(raw_line), None
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                return True, None, f"invalid_json_{type(exc).__name__}"
        if worker.response_eof:
            if worker.response_buffer:
                return True, None, "eof_partial_line"
            return True, None, "eof"
        return False, None, None

    buffered = consume_buffer()
    if buffered[0] or buffered[2] is not None:
        return buffered

    try:
        descriptor = stream.fileno()
    except (AttributeError, OSError, ValueError):
        # 只有沒有 OS descriptor 的測試替身會走這條路；StringIO.read 立即回傳目前
        # 內容，不會重現 subprocess pipe 的 partial-line blocking 行為。
        try:
            chunk = stream.read()
        except (OSError, ValueError) as exc:
            return True, None, f"stdout_read_{type(exc).__name__}"
        if chunk in ("", b""):
            worker.response_eof = True
        elif isinstance(chunk, str):
            worker.response_buffer.extend(chunk.encode("utf-8"))
        elif isinstance(chunk, bytes):
            worker.response_buffer.extend(chunk)
        else:
            return True, None, "stdout_read_invalid_type"
        return consume_buffer()

    try:
        ready, _, _ = select.select([descriptor], [], [], max(0.0, wait_seconds))
    except (OSError, ValueError) as exc:
        return True, None, f"select_{type(exc).__name__}"
    if not ready:
        return False, None, None
    try:
        chunk = os.read(descriptor, 64 * 1024)
    except OSError as exc:
        return True, None, f"stdout_read_{type(exc).__name__}"
    if chunk == b"":
        worker.response_eof = True
    else:
        worker.response_buffer.extend(chunk)
    return consume_buffer()


def _write_assignment_snapshot(
    path: Path,
    payload: Mapping[str, object],
    *,
    initialized: bool,
) -> bool:
    """原子保存 dynamic 實際派發快照，且拒絕把既有非本次檔案靜默覆寫。

    log session 是本次新建且不可碰撞的目錄；第一次仍沿用既有 no-overwrite writer，
    後續更新只替換同一份本次 dynamic assignment snapshot。若第一次看到既有檔案，立即
    fail closed，不把舊執行紀錄當成新派發狀態。
    """

    if not initialized:
        atomic_write_json(path, payload)
        return True
    if path.is_symlink() or not path.is_file():
        raise ParallelExecutionError("dynamic assignment snapshot 被外部替換")
    encoded = json.dumps(
        dict(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
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
        raise ParallelExecutionError("無法原子更新 dynamic assignment snapshot") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        with contextlib.suppress(OSError):
            temporary.unlink(missing_ok=True)
    return True


def _child_affinity_summary(
    *,
    cpu_ids: tuple[int, ...] | None,
    request: str,
    fallback_reason: str | None,
) -> dict[str, object]:
    """建立 parent log 可保存的 affinity request 摘要，不誤稱為已套用。"""

    return {
        "requested_cpu_ids": list(cpu_ids) if cpu_ids is not None else None,
        "request": request,
        "fallback_reason": fallback_reason,
    }


def execute_dynamic_parallel_prepared(
    context: FormalParallelContext,
    *,
    popen_factory: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
    poll_interval_seconds: float = 0.05,
    shutdown_grace_seconds: float = 30.0,
    response_timeout_seconds: float | None = None,
) -> tuple[dict[str, object], int]:
    """在既有 formal preflight context 上執行 dynamic parent queue。

    每個 worker child 只會同時持有一個 in-flight shard。一般模式維持 fail-fast；
    execution-priority 模式則只隔離 child 回報的單片物理／數值 exception 或 FAILED，
    讓同一 persistent controller 接續領取 queue，並對記憶體、訊號、鎖／陣列／網格及
    protocol 結構錯誤停止整批。任何未完成 in-flight shard 都不會被 parent 偷放回 queue；
    只有明示下一次 ``resume`` 且經既有 reconcile／validator 後，才由 progress 決定可再次
    執行的片。若 context 明示延後完整驗收，parent 不呼叫 final full-run validator，
    而以 ``COMPUTE_COMPLETE_VALIDATION_DEFERRED`` 回報計算完成但仍待獨立驗收。
    """

    if poll_interval_seconds < 0 or shutdown_grace_seconds < 0:
        raise ParallelExecutionError("dynamic poll interval 與 shutdown grace 不可為負數")
    if response_timeout_seconds is not None and response_timeout_seconds < 0:
        raise ParallelExecutionError("response_timeout_seconds 不可為負數")

    if context.warmup_only:
        try:
            warmup_summary = warmup_numba_cache(
                numba_cache_dir=context.prepared_cache,
                scratch_root=context.scratch_path,
                gate_evidence=context.storage_gate_evidence,
                project_root=context.project_path,
                log_file=context.log_session / "numba-warmup.log",
            )
        except ParallelExecutionError as exc:
            # dynamic warmup 沿用既有獨立 JIT 檢查；不得因沒有粒子 child 而宣稱正式
            # 執行完成。gate evidence 已在 coordinator preflight 驗證，這裡不再重讀。
            summary = {
                "artifact_type": "formal_dynamic_parallel_execution_summary",
                "schema_version": "1.0.0",
                "execution_mode": "dynamic",
                "run_id": context.run_id,
                "run_plan_sha256": context.plan_sha256,
                "status": "WARMUP_FAILED",
                "valid": False,
                "full_validation_deferred": context.defer_full_validation,
                "validation_scope": context.validation_scope,
                "validation_stage": (
                    "startup_validation" if context.defer_full_validation else "full_validation"
                ),
                "validation_stage_elapsed_seconds": context.validation_elapsed_seconds,
                "error": str(exc),
                "worker_records": [],
                "log_session": context.log_session.name,
            }
            atomic_write_json(context.log_session / "summary.json", summary)
            return summary, 2
        summary = {
            "artifact_type": "formal_dynamic_parallel_execution_summary",
            "schema_version": "1.0.0",
            "execution_mode": "dynamic",
            "execution_priority": context.execution_priority,
            "run_id": context.run_id,
            "run_plan_sha256": context.plan_sha256,
            "status": "WARMUP_CHECKED",
            "valid": True,
            "full_validation_deferred": context.defer_full_validation,
            "validation_scope": context.validation_scope,
            "validation_stage": (
                "startup_validation" if context.defer_full_validation else "full_validation"
            ),
            "validation_stage_elapsed_seconds": context.validation_elapsed_seconds,
            "numba_disk_cache_enabled": False,
            "numba_warmup_scope": "one_off_validation_process",
            "warmup_summary": warmup_summary,
            "worker_records": [],
            "log_session": context.log_session.name,
        }
        atomic_write_json(context.log_session / "summary.json", summary)
        return summary, 0

    pending, already_complete = _pending_shards(context)
    assignment_path = context.log_session / "dynamic-worker-assignment.json"
    assignment_rows: list[dict[str, object]] = []
    completed_ids = set(already_complete)
    failed_ids: set[str] = set()
    pending_ids = [item.shard_id for item in pending]
    inflight_ids: set[str] = set()
    stopped_inflight_ids: set[str] = set()
    dispatch_sequence = 0
    assignment_initialized = False
    stop_reason: str | None = None
    interrupted_signal: int | None = None
    child_states: dict[str, _WorkerState] = {}
    opened_logs: list[TextIO] = []
    previous_handlers: dict[int, Any] = {}
    execution_priority = context.execution_priority

    def assignment_payload() -> dict[str, object]:
        """將 parent 的實際 queue／in-flight 狀態收斂成 JSON-safe snapshot。"""

        return {
            "artifact_type": "formal_dynamic_worker_assignment",
            "schema_version": "1.0.0",
            "execution_mode": "dynamic",
            "execution_priority": execution_priority,
            "run_id": context.run_id,
            "run_plan_sha256": context.plan_sha256,
            "requested_worker_count": context.worker_count,
            "locality_policy": "soft_last_region_month_no_idle_v1",
            "assignments": assignment_rows,
            "completed_shard_ids": sorted(completed_ids),
            "failed_shard_ids": sorted(failed_ids),
            "pending_shard_ids": list(pending_ids),
            "inflight_shard_ids": sorted(inflight_ids),
            "stopped_inflight_shard_ids": sorted(stopped_inflight_ids),
            "stop_reason": stop_reason,
        }

    def save_assignment() -> None:
        """每次實際派發／完成／停止後更新 assignment artifact。"""

        nonlocal assignment_initialized
        assignment_initialized = _write_assignment_snapshot(
            assignment_path,
            assignment_payload(),
            initialized=assignment_initialized,
        )

    def on_signal(signum: int, frame: object) -> None:
        """把 parent 外部 signal 收斂成可保留 assignment 的控制例外。"""

        del frame
        raise ParallelExecutionInterrupted(signum)

    def choose_pending(worker: _WorkerState) -> DynamicShard | None:
        """優先挑相同 locality，沒有就立即挑 queue 首片，不因偏好閒置。"""

        if not pending:
            return None
        selected_index = 0
        if worker.last_locality is not None:
            for index, candidate in enumerate(pending):
                if candidate.locality == worker.last_locality:
                    selected_index = index
                    break
        return pending.pop(selected_index)

    def mark_assignment(shard_id: str, **updates: object) -> None:
        """更新唯一 dispatch row；不存在或重複都代表 parent 狀態錯誤。"""

        matches = [row for row in assignment_rows if row.get("shard_id") == shard_id]
        if len(matches) != 1:
            raise ParallelExecutionError(f"dynamic assignment row 不唯一：{shard_id}")
        matches[0].update(updates)

    def stop_inflight(reason: str) -> None:
        """停止新派工並把未收到結果的片標成 stopped in-flight，不放回 queue。"""

        nonlocal stop_reason
        if stop_reason is None:
            stop_reason = reason
        for worker in child_states.values():
            if worker.inflight is None:
                continue
            shard_id = worker.inflight.shard_id
            inflight_ids.discard(shard_id)
            stopped_inflight_ids.add(shard_id)
            mark_assignment(shard_id, status=_STOPPED_INFLIGHT, stop_reason=reason)
            worker.inflight = None
            worker.dispatched_at = None
        save_assignment()

    def launch_worker(index: int) -> None:
        """建立一個 persistent child，並將 stderr 導向本次 log session。"""

        worker_id = f"worker-{index:04d}"
        log_path = context.log_session / f"dynamic-{worker_id}.log"
        if log_path.exists() or log_path.is_symlink():
            raise ParallelExecutionError("dynamic worker log 已存在；拒絕覆寫既有現場")
        cpu_ids = context.affinity_sets[index] if index < len(context.affinity_sets) else None
        fallback = context.affinity_fallbacks[index] if index < len(context.affinity_fallbacks) else None
        spec = {
            "protocol": PROTOCOL_NAME,
            "run_id": context.run_id,
            "worker_id": worker_id,
            "workspace": str(context.workspace_path),
            "config_path": str(context.config_path),
            "ocm_native_root": str(context.ocm_root),
            "ocm_reconstruction_root": (
                str(context.reconstruction_root) if context.reconstruction_root is not None else None
            ),
            "nww_analysis_root": str(context.nww_root) if context.nww_root is not None else None,
            "checkpoint_root": (
                str(context.checkpoint_path) if context.checkpoint_path is not None else None
            ),
            "cpu_ids": list(cpu_ids) if cpu_ids is not None else None,
            "affinity_request": context.cpu_affinity,
            "affinity_fallback_reason": fallback,
            "warmup_numba_backend": requires_numba_cache(context.config),
            "execution_priority": context.execution_priority,
            "continuous_time_fallback": context.execution_priority,
            "defer_full_validation": context.defer_full_validation,
        }
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONPATH"] = str(context.project_path / "src")
        if context.prepared_cache is not None:
            env["NUMBA_CACHE_DIR"] = str(context.prepared_cache)
        else:
            env.pop("NUMBA_CACHE_DIR", None)
        log_stream = log_path.open("x", encoding="utf-8")
        opened_logs.append(log_stream)
        try:
            process = popen_factory(
                [
                    sys.executable,
                    "-c",
                    _CHILD_BOOTSTRAP,
                    json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                ],
                cwd=context.project_path,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log_stream,
                text=True,
                start_new_session=(os.name == "posix"),
            )
        except OSError as exc:
            log_stream.close()
            raise ParallelExecutionError(f"無法啟動 {worker_id} dynamic child") from exc
        child_states[worker_id] = _WorkerState(
            worker_id=worker_id,
            process=process,
            log_stream=log_stream,
            log_file=log_path.name,
            affinity=_child_affinity_summary(
                cpu_ids=cpu_ids,
                request=context.cpu_affinity,
                fallback_reason=fallback,
            ),
        )

    try:
        save_assignment()
        for signum in (signal.SIGINT, signal.SIGTERM):
            if hasattr(signal, signum.name):
                previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, on_signal)

        # resume 可能只剩少數未完成 shard；不建立沒有 queue 工作的 idle child。
        active_worker_count = min(context.worker_count, len(pending))
        for index in range(active_worker_count):
            launch_worker(index)

        while pending or inflight_ids:
            if execution_priority:
                available_memory = _memory_available_bytes()
                if (
                    available_memory is not None
                    and available_memory < _MIN_MEM_AVAILABLE_BYTES
                ):
                    stop_reason = "memory_available_below_safety_threshold"
                    stop_inflight(stop_reason)
                    break
            if stop_reason is None:
                for worker in child_states.values():
                    if worker.inflight is not None:
                        continue
                    shard = choose_pending(worker)
                    if shard is None:
                        continue
                    locality_hit = worker.last_locality == shard.locality
                    try:
                        _send_request(
                            worker,
                            {
                                "protocol": PROTOCOL_NAME,
                                "message_type": "run_shard",
                                "run_id": context.run_id,
                                "worker_id": worker.worker_id,
                                "shard_id": shard.shard_id,
                                "resume": context.resume,
                                "sweep_budget": None,
                            },
                        )
                    except (BrokenPipeError, OSError, DynamicProtocolError) as exc:
                        # 這片已由 parent 取出並嘗試派送；不可放回 queue，避免偷重派。
                        failed_ids.add(shard.shard_id)
                        assignment_rows.append(
                            {
                                "dispatch_sequence": dispatch_sequence,
                                "worker_id": worker.worker_id,
                                "shard_id": shard.shard_id,
                                "status": "FAILED",
                                "locality_preference_used": locality_hit,
                                "error_code": "dispatch_failed",
                                "error_type": type(exc).__name__,
                            }
                        )
                        pending_ids.remove(shard.shard_id)
                        stop_reason = "dispatch_failed"
                        save_assignment()
                        break
                    dispatch_sequence += 1
                    worker.inflight = shard
                    worker.dispatched_at = time.monotonic()
                    worker.last_locality = shard.locality
                    pending_ids.remove(shard.shard_id)
                    inflight_ids.add(shard.shard_id)
                    assignment_rows.append(
                        {
                            "dispatch_sequence": dispatch_sequence,
                            "worker_id": worker.worker_id,
                            "shard_id": shard.shard_id,
                            "status": "DISPATCHED",
                            "locality": {
                                "analysis_region_id": shard.locality[0],
                                "arrival_month_utc": shard.locality[1],
                            },
                            "locality_preference_used": locality_hit,
                        }
                    )
                    save_assignment()

            if stop_reason is not None:
                stop_inflight(stop_reason)
                break
            if not inflight_ids:
                continue

            # 每輪最多等待 poll interval；因此 signal 可中斷 parent，但 None timeout
            # 不會對合法長片設定人為執行上限。
            for worker in tuple(child_states.values()):
                shard = worker.inflight
                if shard is None:
                    continue
                if (
                    response_timeout_seconds is not None
                    and worker.dispatched_at is not None
                    and time.monotonic() - worker.dispatched_at >= response_timeout_seconds
                ):
                    stop_reason = "worker_response_timeout"
                    mark_assignment(
                        shard.shard_id,
                        status="FAILED",
                        error_code="worker_response_timeout",
                    )
                    failed_ids.add(shard.shard_id)
                    inflight_ids.discard(shard.shard_id)
                    worker.inflight = None
                    worker.dispatched_at = None
                    save_assignment()
                    break
                ready, payload, read_error = _read_response_line(
                    worker,
                    wait_seconds=poll_interval_seconds,
                )
                if not ready:
                    if worker.process.poll() is not None:
                        stop_reason = "worker_eof"
                        mark_assignment(
                            shard.shard_id,
                            status="FAILED",
                            error_code="worker_eof",
                        )
                        failed_ids.add(shard.shard_id)
                        inflight_ids.discard(shard.shard_id)
                        worker.inflight = None
                        worker.dispatched_at = None
                        save_assignment()
                        break
                    continue
                if read_error is not None:
                    stop_reason = read_error
                    mark_assignment(
                        shard.shard_id,
                        status="FAILED",
                        error_code=read_error,
                    )
                    failed_ids.add(shard.shard_id)
                    inflight_ids.discard(shard.shard_id)
                    worker.inflight = None
                    worker.dispatched_at = None
                    save_assignment()
                    break
                if not isinstance(payload, Mapping):
                    stop_reason = "protocol_result_not_object"
                    mark_assignment(shard.shard_id, status="FAILED", error_code=stop_reason)
                    failed_ids.add(shard.shard_id)
                    inflight_ids.discard(shard.shard_id)
                    worker.inflight = None
                    worker.dispatched_at = None
                    save_assignment()
                    break
                if (
                    payload.get("protocol") != PROTOCOL_NAME
                    or payload.get("worker_id") != worker.worker_id
                    or payload.get("run_id") != context.run_id
                    or payload.get("shard_id") != shard.shard_id
                ):
                    stop_reason = "protocol_result_identity_mismatch"
                    mark_assignment(shard.shard_id, status="FAILED", error_code=stop_reason)
                    failed_ids.add(shard.shard_id)
                    inflight_ids.discard(shard.shard_id)
                    worker.inflight = None
                    worker.dispatched_at = None
                    save_assignment()
                    break
                message_type = payload.get("message_type")
                if message_type == "shard_result" and payload.get("lifecycle") in _LIFECYCLES:
                    lifecycle = str(payload["lifecycle"])
                    result_exit_code = payload.get("exit_code")
                    result_summary = payload.get("summary")
                    result_error: str | None = None
                    if type(result_exit_code) is not int:
                        result_error = "protocol_result_exit_code_invalid"
                    elif not isinstance(result_summary, Mapping):
                        result_error = "protocol_summary_not_object"
                    elif (
                        result_summary.get("lifecycle") != lifecycle
                        or result_summary.get("shard_id") != shard.shard_id
                    ):
                        # parent 不以 child payload 的外層 lifecycle 單獨判定完成；摘要
                        # 必須仍指向同一片 immutable shard，避免錯誤結果被誤綁到目前片。
                        result_error = "protocol_summary_identity_mismatch"
                    elif lifecycle == "COMPLETE" and result_exit_code != 0:
                        # COMPLETE 的協定結果必須明確表示 child 正常完成；非零 code
                        # 即使外層 lifecycle 偽稱 COMPLETE，也只能進入 fail-stop。
                        result_error = "worker_exit_nonzero"
                    if result_error is not None:
                        stop_reason = result_error
                        mark_assignment(
                            shard.shard_id,
                            status="FAILED",
                            error_code=result_error,
                            child_exit_code=result_exit_code,
                        )
                        failed_ids.add(shard.shard_id)
                        inflight_ids.discard(shard.shard_id)
                        worker.inflight = None
                        worker.dispatched_at = None
                        save_assignment()
                        break
                    child_affinity = payload.get("affinity")
                    if isinstance(child_affinity, Mapping):
                        worker.affinity = dict(child_affinity)
                    inflight_ids.discard(shard.shard_id)
                    worker.inflight = None
                    worker.dispatched_at = None
                    mark_assignment(
                        shard.shard_id,
                        status=lifecycle,
                        child_exit_code=result_exit_code,
                    )
                    if lifecycle == "COMPLETE":
                        completed_ids.add(shard.shard_id)
                    elif lifecycle == "FAILED":
                        failed_ids.add(shard.shard_id)
                        if not execution_priority:
                            stop_reason = "worker_failed"
                    else:
                        stop_reason = "worker_paused"
                    save_assignment()
                    if stop_reason is not None:
                        break
                    continue
                error_code = str(payload.get("error_code") or "protocol_worker_error")
                fatal = payload.get("fatal", True) is True
                if message_type == "worker_error" and execution_priority and not fatal:
                    # 單片 run_shard 例外已由 child 明確封裝；保留失敗片的錯誤摘要，
                    # 但不終止同一 persistent controller，讓中央 queue 繼續補位。
                    mark_assignment(
                        shard.shard_id,
                        status="FAILED",
                        error_code=error_code,
                        error_type=payload.get("error_type"),
                        error=payload.get("error"),
                    )
                    failed_ids.add(shard.shard_id)
                    inflight_ids.discard(shard.shard_id)
                    worker.inflight = None
                    worker.dispatched_at = None
                    save_assignment()
                    continue
                stop_reason = error_code
                mark_assignment(
                    shard.shard_id,
                    status="FAILED",
                    error_code=stop_reason,
                    error_type=payload.get("error_type"),
                    error=payload.get("error"),
                )
                failed_ids.add(shard.shard_id)
                inflight_ids.discard(shard.shard_id)
                worker.inflight = None
                worker.dispatched_at = None
                save_assignment()
                break

            if stop_reason is not None:
                stop_inflight(stop_reason)
                break

        if stop_reason is None:
            # queue 與 in-flight 都清空才允許正常 shutdown；child 仍由 parent 明確關閉。
            for worker in child_states.values():
                if worker.process.poll() is None:
                    _send_request(
                        worker,
                        {
                            "protocol": PROTOCOL_NAME,
                            "message_type": "shutdown",
                            "run_id": context.run_id,
                            "worker_id": worker.worker_id,
                        },
                    )
    except ParallelExecutionInterrupted as exc:
        interrupted_signal = exc.signum
        stop_inflight("interrupted")
    except (BrokenPipeError, OSError, DynamicProtocolError) as exc:
        stop_reason = stop_reason or f"parent_{type(exc).__name__}"
        stop_inflight(stop_reason)
    finally:
        active_processes = {
            worker_id: worker.process
            for worker_id, worker in child_states.items()
            if worker.process.poll() is None
        }
        # graceful shutdown 只給已完成且 idle 的 child；in-flight／錯誤 child 走既有
        # SIGINT→SIGTERM→SIGKILL bounded helper，不能等待一個無限期合法分片。
        if stop_reason is None and interrupted_signal is None:
            deadline = time.monotonic() + shutdown_grace_seconds
            while active_processes and time.monotonic() < deadline:
                active_processes = {
                    worker_id: process
                    for worker_id, process in active_processes.items()
                    if process.poll() is None
                }
                if active_processes:
                    time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        if active_processes:
            _stop_active_processes(
                active_processes,
                first_signal=signal.SIGINT,
                grace_seconds=shutdown_grace_seconds,
            )
        for worker in child_states.values():
            if worker.process.stdin is not None:
                with contextlib.suppress(OSError):
                    worker.process.stdin.close()
            if worker.process.stdout is not None:
                with contextlib.suppress(OSError):
                    worker.process.stdout.close()
        for stream in opened_logs:
            with contextlib.suppress(OSError):
                stream.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    # 正常 shutdown 也必須核對每個 child 的實際 return code；不能只因所有 shard 曾回報
    # COMPLETE 就忽略 child 隨後以非零狀態退出。這項檢查只影響 parent orchestration
    # 摘要與 fail-stop 狀態，不改變既有 checkpoint 或 shard schema。
    worker_exit_codes = {
        worker.worker_id: worker.process.poll() for worker in child_states.values()
    }
    worker_exit_failures = [
        {"worker_id": worker_id, "exit_code": exit_code}
        for worker_id, exit_code in worker_exit_codes.items()
        if exit_code != 0
    ]
    if worker_exit_failures and stop_reason is None:
        stop_reason = "worker_exit_nonzero"

    # 若 stop reason 是 child failure/interrupt，尚未收到結果的片已在 snapshot 中標記
    # STOPPED_INFLIGHT；下一次只能由明示 resume 的既有 controller/reconcile 判定可否續跑。
    final_validation: dict[str, Any] | None = None
    all_complete = not pending_ids and not inflight_ids and not failed_ids and stop_reason is None
    status = "INTERRUPTED" if interrupted_signal is not None else "INCOMPLETE"
    exit_code = 130 if interrupted_signal == signal.SIGINT else 143 if interrupted_signal else 2
    if all_complete and context.defer_full_validation:
        # 每片 controller 已完成 assigned shard 的輸出／checkpoint／hash／lock gate；但
        # 此政策明確不把「粒子運算完成」冒充全 run 驗收，完整 validate-run 必須由 operator
        # 後補執行。保留 exit code 0 僅表示本次計算與逐片發布成功，不代表正式成果已驗收。
        status = "COMPUTE_COMPLETE_VALIDATION_DEFERRED"
        exit_code = 0
    elif all_complete:
        final_validation = validate_run(
            context.workspace_path,
            require_complete=True,
            checkpoint_root=context.checkpoint_path,
        )
        if final_validation.get("valid") is True:
            status = "COMPLETE"
            exit_code = 0
        else:
            status = "FAILED_FINAL_VALIDATION"
    elif interrupted_signal is None and stop_reason is None and failed_ids:
        status = "PARTIAL_FAILURE"
    worker_records = [
        {
            "worker_id": worker.worker_id,
            "pid": worker.process.pid,
            "exit_code": worker_exit_codes[worker.worker_id],
            "log_file": worker.log_file,
            "affinity": worker.affinity,
        }
        for worker in child_states.values()
    ]
    save_assignment()
    summary: dict[str, object] = {
        "artifact_type": "formal_dynamic_parallel_execution_summary",
        "schema_version": "1.0.0",
        "execution_mode": "dynamic",
        "execution_priority": context.execution_priority,
        "provenance_override_recorded": context.provenance_override is not None,
        "run_id": context.run_id,
        "run_plan_sha256": context.plan_sha256,
        "status": status,
        "valid": status == "COMPLETE",
        "full_validation_deferred": context.defer_full_validation,
        "validation_scope": context.validation_scope,
        "validation_stage": (
            "startup_validation" if context.defer_full_validation else "full_validation"
        ),
        "validation_stage_elapsed_seconds": context.validation_elapsed_seconds,
        "validation_status_file": context.validation_status_file,
        "numba_disk_cache_enabled": False,
        "numba_warmup_scope": (
            "per_worker_process" if requires_numba_cache(context.config) else "not_applicable"
        ),
        "whole_machine_elapsed_seconds": time.perf_counter() - context.started_wall,
        "storage_gate_sha256": context.gate_sha256,
        "requested_worker_count": context.worker_count,
        "actual_worker_count": len(child_states),
        "worker_records": worker_records,
        "worker_exit_failures": worker_exit_failures,
        "dispatch_sequence": assignment_rows,
        "assignment_file": assignment_path.name,
        "completed_shard_ids": sorted(completed_ids),
        "failed_shard_ids": sorted(failed_ids),
        "pending_shard_ids": list(pending_ids),
        "inflight_shard_ids": sorted(inflight_ids),
        "stopped_inflight_shard_ids": sorted(stopped_inflight_ids),
        "stop_reason": stop_reason,
        "interrupted_signal": interrupted_signal,
        "final_validation": final_validation,
        "log_session": context.log_session.name,
    }
    atomic_write_json(context.log_session / "summary.json", summary)
    return summary, exit_code


__all__ = [
    "DynamicProtocolError",
    "PROTOCOL_NAME",
    "dynamic_worker_child_main",
    "execute_dynamic_parallel",
    "execute_dynamic_parallel_prepared",
]
