"""persistent dynamic worker 的 IPC、queue 補位與 execution-priority 測試。

測試只使用小型 plan、假 controller 與 in-memory pipe，不讀取 OCM／NWW、不連接 SERVER，
但仍驗證 persistent child 只建立一次 controller、單片失敗可隔離，以及中央 queue 會把
完成的 slot 立即補上下一片。
"""

from __future__ import annotations

import io
import json
import signal
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from lagrangian_backtracking import cli
from lagrangian_backtracking import dynamic_execution as dynamic
from lagrangian_backtracking.parallel_execution import FormalParallelContext


def _spec(*, priority: bool = False, defer: bool = False) -> dict[str, object]:
    """建立只含協定所需欄位的 child spec；不帶任何絕對輸入資料路徑。"""

    return {
        "protocol": dynamic.PROTOCOL_NAME,
        "run_id": "dynamic-test-run",
        "worker_id": "worker-0000",
        "workspace": "/tmp/lbt-test-workspace",
        "config_path": "/tmp/lbt-test-config.yaml",
        "ocm_native_root": "/tmp/lbt-test-ocm",
        "execution_priority": priority,
        "continuous_time_fallback": priority,
        "defer_full_validation": defer,
    }


def _request(shard_id: str, *, priority: bool = False) -> str:
    """建立一筆 child JSON-line request，與 parent 產出的欄位保持一致。"""

    return (
        json.dumps(
            {
                "protocol": dynamic.PROTOCOL_NAME,
                "message_type": "run_shard",
                "run_id": "dynamic-test-run",
                "worker_id": "worker-0000",
                "shard_id": shard_id,
                "resume": priority,
                "sweep_budget": None,
            },
            separators=(",", ":"),
        )
        + "\n"
    )


def test_persistent_child_opens_controller_once_and_runs_multiple_shards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """兩片連續 request 必須共用同一 controller，stdout 只能有 JSON-line。"""

    opened: list[object] = []
    executed: list[str] = []

    class FakeController:
        pass

    controller = FakeController()

    def open_controller(*_args: Any, **_kwargs: Any) -> FakeController:
        opened.append(controller)
        return controller

    def execute(controller_value: object, shard_id: str, *, sweep_budget: int | None = None) -> dict[str, object]:
        assert controller_value is controller
        assert sweep_budget is None
        executed.append(shard_id)
        return {"shard_id": shard_id, "lifecycle": "COMPLETE"}

    monkeypatch.setattr(cli, "open_worker_controller", open_controller)
    monkeypatch.setattr(cli, "execute_worker_shard", execute)
    stdout = io.StringIO()
    stderr = io.StringIO()
    code = dynamic.dynamic_worker_child_main(
        _spec(),
        io.StringIO(_request("shard-0000") + _request("shard-0001") + '{"protocol":"'
        + dynamic.PROTOCOL_NAME
        + '","message_type":"shutdown","run_id":"dynamic-test-run","worker_id":"worker-0000"}\n'),
        stdout,
        stderr,
    )

    assert code == 0
    assert opened == [controller]
    assert executed == ["shard-0000", "shard-0001"]
    lines = stdout.getvalue().splitlines()
    assert len(lines) == 3
    assert all(isinstance(json.loads(line), dict) for line in lines)
    assert not [line for line in lines if not line.lstrip().startswith("{")]


def test_deferred_child_forwards_private_bootstrap_only_when_priority_is_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """延後驗收 child 只轉交 private capability，且不把 public skip 旗標送入 controller。"""

    from lagrangian_backtracking.runtime import _DEFERRED_VALIDATION_PRIVATE_TOKEN

    opened: list[dict[str, object]] = []
    monkeypatch.setattr(
        cli,
        "open_worker_controller",
        lambda *_args, **kwargs: opened.append(kwargs) or object(),
    )
    monkeypatch.setattr(
        cli,
        "execute_worker_shard",
        lambda _controller, shard_id, *, sweep_budget=None: {
            "shard_id": shard_id,
            "lifecycle": "COMPLETE",
        },
    )
    shutdown = json.dumps(
        {
            "protocol": dynamic.PROTOCOL_NAME,
            "message_type": "shutdown",
            "run_id": "dynamic-test-run",
            "worker_id": "worker-0000",
        },
        separators=(",", ":"),
    )

    code = dynamic.dynamic_worker_child_main(
        _spec(priority=True, defer=True),
        io.StringIO(_request("shard-0000", priority=True) + shutdown + "\n"),
        io.StringIO(),
        io.StringIO(),
    )

    assert code == 0
    assert len(opened) == 1
    assert opened[0]["_defer_full_validation"] is True
    assert opened[0]["_private_dynamic_bootstrap"] is _DEFERRED_VALIDATION_PRIVATE_TOKEN
    assert "defer_full_validation" not in opened[0]


def test_deferred_child_rejects_spec_without_execution_priority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """被竄改的 child spec 不得以 deferred bool 單獨啟用 startup-only 驗證。"""

    opened: list[object] = []
    monkeypatch.setattr(
        cli,
        "open_worker_controller",
        lambda *_args, **_kwargs: opened.append(object()),
    )
    stdout = io.StringIO()
    code = dynamic.dynamic_worker_child_main(
        _spec(priority=False, defer=True),
        io.StringIO(),
        stdout,
        io.StringIO(),
    )

    payload = json.loads(stdout.getvalue())
    assert code == 2
    assert payload["error_code"] == "child_spec_defer_full_validation_requires_priority"
    assert opened == []


def test_execution_priority_isolates_one_shard_exception_and_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """priority 模式的單片 exception 只發布 FAILED，persistent child 仍接下一片。"""

    opened: list[dict[str, object]] = []
    executed: list[str] = []

    monkeypatch.setattr(
        cli,
        "open_worker_controller",
        lambda *_args, **kwargs: opened.append(kwargs) or object(),
    )

    def execute(_controller: object, shard_id: str, *, sweep_budget: int | None = None) -> dict[str, object]:
        del sweep_budget
        executed.append(shard_id)
        if shard_id == "shard-0000":
            raise RuntimeError("synthetic numerical failure")
        return {"shard_id": shard_id, "lifecycle": "COMPLETE"}

    monkeypatch.setattr(cli, "execute_worker_shard", execute)
    requests = _request("shard-0000", priority=True) + _request("shard-0001", priority=True)
    shutdown = json.dumps(
        {
            "protocol": dynamic.PROTOCOL_NAME,
            "message_type": "shutdown",
            "run_id": "dynamic-test-run",
            "worker_id": "worker-0000",
        },
        separators=(",", ":"),
    )
    stdout = io.StringIO()
    code = dynamic.dynamic_worker_child_main(
        _spec(priority=True), io.StringIO(requests + shutdown + "\n"), stdout, io.StringIO()
    )

    payloads = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert code == 0
    assert executed == ["shard-0000", "shard-0001"]
    assert len(opened) == 1
    assert payloads[0]["message_type"] == "worker_error"
    assert payloads[0]["fatal"] is False
    assert payloads[1]["message_type"] == "shard_result"
    assert payloads[1]["lifecycle"] == "COMPLETE"
    assert payloads[-1]["message_type"] == "shutdown_ack"
    assert opened[0]["continuous_time_fallback"] is True


def test_default_child_exception_preserves_fail_fast_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """未開 priority 時，既有單片 exception 仍會結束 child，不改預設 fail-fast。"""

    executed: list[str] = []
    monkeypatch.setattr(cli, "open_worker_controller", lambda *_args, **_kwargs: object())

    def execute(*_args: Any, **_kwargs: Any) -> dict[str, object]:
        executed.append("called")
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(cli, "execute_worker_shard", execute)
    stdout = io.StringIO()
    code = dynamic.dynamic_worker_child_main(
        _spec(), io.StringIO(_request("shard-0000") + _request("shard-0001")), stdout, io.StringIO()
    )
    payloads = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert code == 1
    assert executed == ["called"]
    assert len(payloads) == 1
    assert payloads[0]["message_type"] == "worker_error"
    assert payloads[0]["fatal"] is False


class _FakePipe:
    """以 StringIO 模擬可寫入 request、可逐次讀取 response 的 child pipe。"""

    def __init__(self, on_write: Any) -> None:
        self._stream = io.StringIO()
        self._on_write = on_write

    def write(self, value: str) -> int:
        self._on_write(value)
        return len(value)

    def flush(self) -> None:
        return None

    def read(self, *_args: Any) -> str:
        return self._stream.read()

    def append_response(self, value: dict[str, object]) -> None:
        position = self._stream.tell()
        self._stream.seek(0, io.SEEK_END)
        self._stream.write(json.dumps(value, separators=(",", ":")) + "\n")
        self._stream.seek(position)

    def close(self) -> None:
        self._stream.close()


class _FakePersistentProcess:
    """parent queue 測試用的長壽命 child：每個 process 可接收任意多片。"""

    _next_pid = 60_000

    def __init__(
        self,
        spec: dict[str, object],
        *,
        failed_once: bool,
        fatal_once: bool = False,
    ) -> None:
        type(self)._next_pid += 1
        self.pid = type(self)._next_pid
        self.returncode: int | None = None
        self.spec = dict(spec)
        self.worker_id = str(spec["worker_id"])
        self.failed_once = failed_once
        self.fatal_once = fatal_once
        self.requests: list[str] = []
        self.stdout = _FakePipe(lambda _value: None)
        self.stdin = _FakePipe(self._handle_request)

    def _handle_request(self, raw: str) -> None:
        request = json.loads(raw)
        self.requests.append(request["shard_id"] if request["message_type"] == "run_shard" else "shutdown")
        if request["message_type"] == "shutdown":
            self.stdout.append_response(
                {
                    "protocol": dynamic.PROTOCOL_NAME,
                    "message_type": "shutdown_ack",
                    "run_id": request["run_id"],
                    "worker_id": self.worker_id,
                }
            )
            self.returncode = 0
            return
        shard_id = request["shard_id"]
        if self.fatal_once:
            self.fatal_once = False
            self.stdout.append_response(
                {
                    "protocol": dynamic.PROTOCOL_NAME,
                    "message_type": "worker_error",
                    "run_id": request["run_id"],
                    "worker_id": self.worker_id,
                    "shard_id": shard_id,
                    "error_code": "fatal_worker_error",
                    "error_type": "ValueError",
                    "error": "synthetic array shape failure",
                    "fatal": True,
                }
            )
            return
        if self.failed_once:
            self.failed_once = False
            self.stdout.append_response(
                {
                    "protocol": dynamic.PROTOCOL_NAME,
                    "message_type": "worker_error",
                    "run_id": request["run_id"],
                    "worker_id": self.worker_id,
                    "shard_id": shard_id,
                    "error_code": "run_shard_exception",
                    "error_type": "RuntimeError",
                    "error": "synthetic failure",
                    "fatal": False,
                }
            )
            return
        self.stdout.append_response(
            {
                "protocol": dynamic.PROTOCOL_NAME,
                "message_type": "shard_result",
                "run_id": request["run_id"],
                "worker_id": self.worker_id,
                "shard_id": shard_id,
                "lifecycle": "COMPLETE",
                "exit_code": 0,
                "summary": {"shard_id": shard_id, "lifecycle": "COMPLETE"},
            }
        )

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return 0 if self.returncode is None else self.returncode

    def send_signal(self, signum: int) -> None:
        """模擬 coordinator 的 bounded signal 收尾，避免測試留下 active child。"""

        self.returncode = -signum


def _context(
    tmp_path: Path,
    *,
    priority: bool,
    resume: bool = False,
) -> FormalParallelContext:
    """建立不觸發正式輸入 I/O 的 dynamic context。"""

    plan = {
        "run_id": "dynamic-test-run",
        "shards": [
            {
                "shard_id": f"shard-{index:04d}",
                "analysis_region_id": "region-a",
                "arrival_time_utc_ns": 1_704_067_200 * 1_000_000_000,
            }
            for index in range(5)
        ],
    }
    config = SimpleNamespace(execution=SimpleNamespace(physics_kernel_backend="numpy", ocm_interpolation_backend="numpy"))
    return FormalParallelContext(
        started_wall=0.0,
        plan=plan,
        config=config,
        config_path=tmp_path / "config.yaml",
        workspace_path=tmp_path / "workspace",
        project_path=tmp_path,
        checkpoint_path=None,
        ocm_root=tmp_path / "ocm",
        nww_root=None,
        reconstruction_root=None,
        scenario_locality={},
        locality_columns=(),
        static_groups=None,
        scratch_path=tmp_path,
        log_path=tmp_path,
        prepared_cache=None,
        storage_gate_evidence=tmp_path / "gate.json",
        gate_sha256="a" * 64,
        affinity_sets=(None, None),
        affinity_fallbacks=(None, None),
        run_id="dynamic-test-run",
        log_session=tmp_path,
        plan_sha256="b" * 64,
        worker_count=2,
        resume=resume,
        cpu_affinity="none",
        warmup_only=False,
        execution_priority=priority,
        provenance_override=None,
    )


def test_parent_queue_reuses_two_persistent_children_and_fills_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """五片工作由兩個 child 接續完成，並排除 per-shard process。"""

    monkeypatch.setattr(
        dynamic,
        "load_run_progress",
        lambda _workspace: {
            "shards": {
                f"shard-{index:04d}": {"lifecycle": "PLANNED"} for index in range(5)
            }
        },
    )
    monkeypatch.setattr(dynamic, "validate_run", lambda *_args, **_kwargs: {"valid": True})
    processes: list[_FakePersistentProcess] = []

    def popen(command: list[str], **_kwargs: Any) -> _FakePersistentProcess:
        spec = json.loads(command[-1])
        process = _FakePersistentProcess(spec, failed_once=False)
        processes.append(process)
        return process

    summary, code = dynamic.execute_dynamic_parallel_prepared(
        _context(tmp_path, priority=False),
        popen_factory=popen,
        poll_interval_seconds=0,
        shutdown_grace_seconds=0,
    )
    assert code == 0
    assert summary["status"] == "COMPLETE"
    assert len(processes) == 2
    assert sum(len(process.requests) - 1 for process in processes) == 5
    assert sorted(summary["completed_shard_ids"]) == [f"shard-{index:04d}" for index in range(5)]


def test_deferred_parent_reports_compute_complete_without_final_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """所有分片完成時 deferred policy 不呼叫全 run validator，也不冒充正式 COMPLETE。"""

    monkeypatch.setattr(
        dynamic,
        "load_run_progress",
        lambda _workspace: {
            "shards": {
                f"shard-{index:04d}": {"lifecycle": "PLANNED"} for index in range(5)
            }
        },
    )

    def reject_full_validation(*_args: Any, **_kwargs: Any) -> dict[str, object]:
        """若 deferred coordinator 仍掃描整批，測試必須立即失敗。"""

        raise AssertionError("deferred parent 不得執行 final validate_run")

    monkeypatch.setattr(dynamic, "validate_run", reject_full_validation)
    processes: list[_FakePersistentProcess] = []

    def popen(command: list[str], **_kwargs: Any) -> _FakePersistentProcess:
        process = _FakePersistentProcess(json.loads(command[-1]), failed_once=False)
        processes.append(process)
        return process

    context = replace(
        _context(tmp_path, priority=True),
        defer_full_validation=True,
        validation_scope="startup_bindings_only",
        validation_elapsed_seconds=1.25,
        validation_status_file="validation-status.json",
    )
    summary, code = dynamic.execute_dynamic_parallel_prepared(
        context,
        popen_factory=popen,
        poll_interval_seconds=0,
        shutdown_grace_seconds=0,
    )

    assert code == 0
    assert summary["status"] == "COMPUTE_COMPLETE_VALIDATION_DEFERRED"
    assert summary["valid"] is False
    assert summary["final_validation"] is None
    assert summary["full_validation_deferred"] is True
    assert summary["validation_scope"] == "startup_bindings_only"
    assert summary["validation_stage"] == "startup_validation"
    assert summary["validation_stage_elapsed_seconds"] == 1.25
    assert summary["validation_status_file"] == "validation-status.json"
    assert all(process.spec["defer_full_validation"] is True for process in processes)


def test_priority_parent_continues_after_nonfatal_worker_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """priority parent 收到 nonfatal worker_error 後仍補派 queue，並保留 failed ID。"""

    monkeypatch.setattr(
        dynamic,
        "load_run_progress",
        lambda _workspace: {
            "shards": {
                f"shard-{index:04d}": {"lifecycle": "PLANNED"} for index in range(5)
            }
        },
    )
    processes: list[_FakePersistentProcess] = []

    def popen(command: list[str], **_kwargs: Any) -> _FakePersistentProcess:
        spec = json.loads(command[-1])
        process = _FakePersistentProcess(spec, failed_once=len(processes) == 0)
        processes.append(process)
        return process

    summary, code = dynamic.execute_dynamic_parallel_prepared(
        _context(tmp_path, priority=True),
        popen_factory=popen,
        poll_interval_seconds=0,
        shutdown_grace_seconds=0,
    )
    assert code == 2
    assert summary["status"] == "PARTIAL_FAILURE"
    assert summary["failed_shard_ids"] == ["shard-0000"]
    assert len(summary["completed_shard_ids"]) == 4
    assert len(processes) == 2
    assert all(process.requests[-1] == "shutdown" for process in processes)


def test_sigterm_interrupt_preserves_stopped_inflight_without_redelivery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SIGTERM 對應的中斷必須保留 in-flight，且下一輪不可偷偷重派。"""

    monkeypatch.setattr(
        dynamic,
        "load_run_progress",
        lambda _workspace: {
            "shards": {
                f"shard-{index:04d}": {"lifecycle": "PLANNED"} for index in range(5)
            }
        },
    )
    processes: list[_FakePersistentProcess] = []

    def popen(command: list[str], **_kwargs: Any) -> _FakePersistentProcess:
        process = _FakePersistentProcess(json.loads(command[-1]), failed_once=False)
        processes.append(process)
        return process

    def interrupt(*_args: Any, **_kwargs: Any) -> tuple[bool, object | None, str | None]:
        raise dynamic.ParallelExecutionInterrupted(signal.SIGTERM)

    monkeypatch.setattr(dynamic, "_read_response_line", interrupt)
    summary, code = dynamic.execute_dynamic_parallel_prepared(
        _context(tmp_path, priority=False),
        popen_factory=popen,
        poll_interval_seconds=0,
        shutdown_grace_seconds=0,
    )

    assert code == 143
    assert summary["status"] == "INTERRUPTED"
    assert summary["interrupted_signal"] == signal.SIGTERM
    assert summary["stop_reason"] == "interrupted"
    assert summary["pending_shard_ids"] == [
        "shard-0002",
        "shard-0003",
        "shard-0004",
    ]
    assert set(summary["stopped_inflight_shard_ids"]) == {"shard-0000", "shard-0001"}
    assert [request for process in processes for request in process.requests] == [
        "shard-0000",
        "shard-0001",
    ]


def test_affinity_request_is_recorded_in_child_spec_and_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """parent 傳入的 CPU affinity 必須同時留在 child spec 與 worker 摘要。"""

    monkeypatch.setattr(
        dynamic,
        "load_run_progress",
        lambda _workspace: {
            "shards": {
                f"shard-{index:04d}": {"lifecycle": "PLANNED"} for index in range(5)
            }
        },
    )
    monkeypatch.setattr(dynamic, "validate_run", lambda *_args, **_kwargs: {"valid": True})
    processes: list[_FakePersistentProcess] = []

    def popen(command: list[str], **_kwargs: Any) -> _FakePersistentProcess:
        process = _FakePersistentProcess(json.loads(command[-1]), failed_once=False)
        processes.append(process)
        return process

    context = replace(
        _context(tmp_path, priority=False),
        affinity_sets=((2, 3), (4, 5)),
        cpu_affinity="2,3,4,5",
    )
    summary, code = dynamic.execute_dynamic_parallel_prepared(
        context,
        popen_factory=popen,
        poll_interval_seconds=0,
        shutdown_grace_seconds=0,
    )

    assert code == 0
    assert [process.spec["cpu_ids"] for process in processes] == [[2, 3], [4, 5]]
    records = {row["worker_id"]: row for row in summary["worker_records"]}
    assert records["worker-0000"]["affinity"]["requested_cpu_ids"] == [2, 3]
    assert records["worker-0001"]["affinity"]["requested_cpu_ids"] == [4, 5]


def test_fatal_worker_error_stops_queue_and_preserves_other_inflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fatal worker_error 不得被 priority 隔離，已派出的其他片只能留在 stopped-inflight。"""

    monkeypatch.setattr(
        dynamic,
        "load_run_progress",
        lambda _workspace: {
            "shards": {
                f"shard-{index:04d}": {"lifecycle": "PLANNED"} for index in range(5)
            }
        },
    )
    processes: list[_FakePersistentProcess] = []

    def popen(command: list[str], **_kwargs: Any) -> _FakePersistentProcess:
        process = _FakePersistentProcess(
            json.loads(command[-1]),
            failed_once=False,
            fatal_once=len(processes) == 0,
        )
        processes.append(process)
        return process

    summary, code = dynamic.execute_dynamic_parallel_prepared(
        _context(tmp_path, priority=True),
        popen_factory=popen,
        poll_interval_seconds=0,
        shutdown_grace_seconds=0,
    )

    assert code == 2
    assert summary["status"] == "INCOMPLETE"
    assert summary["stop_reason"] == "fatal_worker_error"
    assert summary["failed_shard_ids"] == ["shard-0000"]
    assert summary["stopped_inflight_shard_ids"] == ["shard-0001"]
    assert summary["pending_shard_ids"] == ["shard-0002", "shard-0003", "shard-0004"]
    assert [request for process in processes for request in process.requests] == [
        "shard-0000",
        "shard-0001",
    ]


def test_memory_available_threshold_stops_new_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """priority 模式偵測到低於安全 MemAvailable 時，不再派任何 shard。"""

    monkeypatch.setattr(
        dynamic,
        "load_run_progress",
        lambda _workspace: {
            "shards": {
                f"shard-{index:04d}": {"lifecycle": "PLANNED"} for index in range(5)
            }
        },
    )
    monkeypatch.setattr(
        dynamic,
        "_memory_available_bytes",
        lambda: dynamic._MIN_MEM_AVAILABLE_BYTES - 1,
    )
    processes: list[_FakePersistentProcess] = []

    def popen(command: list[str], **_kwargs: Any) -> _FakePersistentProcess:
        process = _FakePersistentProcess(json.loads(command[-1]), failed_once=False)
        processes.append(process)
        return process

    summary, code = dynamic.execute_dynamic_parallel_prepared(
        _context(tmp_path, priority=True),
        popen_factory=popen,
        poll_interval_seconds=0,
        shutdown_grace_seconds=0,
    )

    assert code == 2
    assert summary["status"] == "INCOMPLETE"
    assert summary["stop_reason"] == "memory_available_below_safety_threshold"
    assert summary["pending_shard_ids"] == [f"shard-{index:04d}" for index in range(5)]
    assert all(process.requests == [] for process in processes)


def test_resume_excludes_complete_and_preserves_resume_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """resume 只派未 COMPLETE 片，且所有實際 request 都沿用同一 resume binding。"""

    monkeypatch.setattr(
        dynamic,
        "load_run_progress",
        lambda _workspace: {
            "shards": {
                "shard-0000": {"lifecycle": "COMPLETE"},
                "shard-0001": {"lifecycle": "PAUSED"},
                "shard-0002": {"lifecycle": "PLANNED"},
                "shard-0003": {"lifecycle": "FAILED"},
                "shard-0004": {"lifecycle": "PLANNED"},
            }
        },
    )
    monkeypatch.setattr(dynamic, "validate_run", lambda *_args, **_kwargs: {"valid": True})
    processes: list[_FakePersistentProcess] = []

    def popen(command: list[str], **_kwargs: Any) -> _FakePersistentProcess:
        spec = json.loads(command[-1])
        process = _FakePersistentProcess(spec, failed_once=False)
        processes.append(process)
        return process

    summary, code = dynamic.execute_dynamic_parallel_prepared(
        _context(tmp_path, priority=False, resume=True),
        popen_factory=popen,
        poll_interval_seconds=0,
        shutdown_grace_seconds=0,
    )

    assert code == 0
    assert summary["status"] == "COMPLETE"
    assert summary["completed_shard_ids"] == [f"shard-{index:04d}" for index in range(5)]
    dispatched = [shard_id for process in processes for shard_id in process.requests[:-1]]
    assert "shard-0000" not in dispatched
    assert sorted(dispatched) == [
        "shard-0001",
        "shard-0002",
        "shard-0003",
        "shard-0004",
    ]


def test_execution_priority_persists_append_only_provenance_override_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """priority 只另存 actual/expected provenance 摘要，不修改 immutable run plan。"""

    import lagrangian_backtracking.dynamic_execution as dynamic_module
    import lagrangian_backtracking.parallel_execution as parallel

    log_session = tmp_path / "log-session"
    log_session.mkdir()
    context = replace(
        _context(tmp_path, priority=True),
        log_session=log_session,
        provenance_override={
            "artifact_type": "formal_execution_priority_provenance_warning",
            "policy": "execution_priority_v1",
            "override_applied": True,
            "immutable_run_plan_modified": False,
            "expected_provenance": {"git_commit": "expected"},
            "actual_provenance": {"git_commit": "actual"},
            "mismatched_fields": {"git_commit": {"expected": "expected", "actual": "actual"}},
        },
    )
    monkeypatch.setattr(parallel, "_prepare_formal_parallel_context", lambda **_kwargs: context)
    monkeypatch.setattr(
        dynamic_module,
        "execute_dynamic_parallel_prepared",
        lambda *_args, **_kwargs: ({"status": "COMPLETE"}, 0),
    )

    summary, code = parallel.execute_formal_parallel(
        workspace=tmp_path / "workspace",
        config_path=tmp_path / "config.yaml",
        project_root=tmp_path,
        worker_count=1,
        scratch_root=tmp_path / "scratch",
        log_root=tmp_path / "logs",
        storage_gate_evidence=tmp_path / "gate.json",
        execution_priority=True,
    )

    record_path = log_session / "execution-priority-provenance.json"
    assert code == 0
    assert summary == {"status": "COMPLETE"}
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["immutable_run_plan_modified"] is False
    assert record["actual_provenance"]["git_commit"] == "actual"
