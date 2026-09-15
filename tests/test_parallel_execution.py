"""正式平行執行器的 deterministic assignment、程序生命週期與 fail-closed 測試。

測試只建構少量 in-memory plan 與假 child process；不連接 SERVER、不開啟 forcing，也不跑
大型粒子母體。執行器的 NFS gate 由單元測試替身驗證，其 production 路徑仍要求真實 gate JSON。
"""

from __future__ import annotations

import json
import signal
import subprocess
from pathlib import Path
from typing import Any

import pytest

from lagrangian_backtracking import cli
from lagrangian_backtracking import parallel_execution as parallel


def _plan(shard_count: int = 6) -> dict[str, Any]:
    """建立連續 scenario index、含正式必要 region／UTC 欄位的小型固定 plan。"""

    rows = []
    for index in range(shard_count):
        month = 1 if index < shard_count // 2 else 2
        rows.append(
            {
                "shard_id": f"shard-{index:04d}",
                "scenario_start_index": index,
                "scenario_stop_index": index + 1,
                "particle_count": 1,
                "analysis_region_id": "region-a" if index < shard_count // 2 else "region-b",
                "arrival_time_utc_ns": (1_704_067_200 + (month - 1) * 2_678_400) * 1_000_000_000,
            }
        )
    return {
        "run_id": "formal-test-run",
        "run_kind": "formal",
        "shard_count": shard_count,
        "shards": rows,
    }


def _groups(count: int = 3) -> tuple[parallel.WorkerGroup, ...]:
    """產生供 child lifecycle 測試共用的確定分組。"""

    return parallel.build_worker_groups(_plan(), worker_count=count)


def _worker_result(group: parallel.WorkerGroup) -> dict[str, object]:
    """模擬 run-worker 完整輸出，供測試確認 finish 順序不參與 assignment。"""

    return {
        "artifact_type": "formal_parallel_worker_result",
        "run_worker_summary": {
            "artifact_type": "run_worker_execution_summary",
            "run_id": "formal-test-run",
            "requested_shard_ids": list(group.shard_ids),
            "executed_shards": [
                {"shard_id": shard_id, "lifecycle": "COMPLETE"}
                for shard_id in group.shard_ids
            ],
        },
    }


class _FakeProcess:
    """以可控 exit status 取代 Python child，避免測試啟動科學運算程序。"""

    _next_pid = 40_000

    def __init__(self, *, returncode: int | None = 0, signal_on_first_poll: int | None = None) -> None:
        """保存測試專用 return code，必要時在主執行緒送出真實 Python signal。"""

        type(self)._next_pid += 1
        self.pid = type(self)._next_pid
        self.returncode = returncode
        self.signal_on_first_poll = signal_on_first_poll
        self._signal_sent = False

    def poll(self) -> int | None:
        """模擬 Popen.poll，讓 SIGINT／SIGTERM 測試走到 coordinator 的 signal handler。"""

        if self.signal_on_first_poll is not None and not self._signal_sent:
            self._signal_sent = True
            signal.raise_signal(self.signal_on_first_poll)
        return self.returncode

    def send_signal(self, signum: int) -> None:
        """提供非 POSIX fallback 介面；測試中的停止程序另由 monkeypatch 驅動。"""

        self.returncode = -signum

    def wait(self, timeout: float | None = None) -> int:
        """回傳已設定 exit code；假程序不會等待或佔用資源。"""

        del timeout
        if self.returncode is None:
            raise RuntimeError("測試替身仍在執行")
        return self.returncode


def test_single_and_multiple_worker_assignments_are_contiguous_and_deterministic() -> None:
    """驗證單 worker 覆蓋全 plan，多 worker 的 scenario 範圍連續且 JSON 位元組固定。"""

    plan = _plan()
    one = parallel.build_worker_groups(plan, worker_count=1)
    three_a = parallel.build_worker_groups(plan, worker_count=3)
    three_b = parallel.build_worker_groups(plan, worker_count=3)

    assert len(one) == 1
    assert one[0].shard_ids == tuple(row["shard_id"] for row in plan["shards"])
    assert [group.to_dict() for group in three_a] == [group.to_dict() for group in three_b]
    assert [(group.scenario_start_index, group.scenario_stop_index) for group in three_a] == [
        (0, 2),
        (2, 4),
        (4, 6),
    ]
    assignment = parallel.worker_assignment_document(
        run_id="formal-test-run",
        run_plan_sha256="a" * 64,
        worker_count=3,
        groups=three_a,
        locality_columns=(),
    )
    repeated = parallel.worker_assignment_document(
        run_id="formal-test-run",
        run_plan_sha256="a" * 64,
        worker_count=3,
        groups=three_b,
        locality_columns=(),
    )
    assert assignment["locality_fallback"] == "plan_order_only"
    assert parallel.canonical_json_bytes(assignment) == parallel.canonical_json_bytes(repeated)


def test_plan_and_table_locality_is_reported_without_guessing_missing_fields() -> None:
    """驗證有 region/month 的 plan key，且沒有表格欄位時不推測 site 或 flow domain。"""

    groups = parallel.build_worker_groups(_plan(), worker_count=3)
    document = parallel.worker_assignment_document(
        run_id="formal-test-run",
        run_plan_sha256="b" * 64,
        worker_count=3,
        groups=groups,
        locality_columns=(),
    )

    assert groups[0].locality_keys == (("region-a", "2024-01"),)
    assert groups[-1].locality_keys == (("region-b", "2024-02"),)
    assert all(group.study_site_ids == () and group.flow_domain_ids == () for group in groups)
    assert document["locality_source_columns"] == []
    assert document["locality_fallback"] == "plan_order_only"


def test_affinity_planning_falls_back_safely_when_platform_api_is_unavailable() -> None:
    """Linux affinity API 不可用時 auto 不綁定，明示 CPU ID 則 fail closed。"""

    assert parallel.plan_cpu_affinity("auto", worker_count=2, available_cpus=None) == (
        None,
        None,
    )
    assert parallel.plan_cpu_affinity("none", worker_count=2, available_cpus=None) == (
        None,
        None,
    )
    with pytest.raises(parallel.ParallelExecutionError, match="無法讀取 CPU affinity"):
        parallel.plan_cpu_affinity("2,3", worker_count=2, available_cpus=None)
    assert (
        parallel.affinity_fallback_reason(
            "auto", cpu_ids=None, available_cpus=None
        )
        == "unavailable_platform_fallback"
    )


def test_live_mount_identity_uses_deepest_covering_mount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """即時掛載核對須選實際涵蓋 scratch 的最深掛載點並隱藏原始 source。"""

    scratch = tmp_path / "nfs" / "scratch"
    scratch.mkdir(parents=True)
    source = "server.example:/private/export/lbt"
    payload = {
        "filesystems": [
            {"fstype": "ext4", "source": "/dev/disk0", "target": str(tmp_path)},
            {"fstype": "nfs4", "source": source, "target": str(tmp_path / "nfs")},
        ]
    }

    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        """回傳兩層合成 mount，避免單元測試依賴開發機實際檔案系統。"""

        del args, kwargs
        return subprocess.CompletedProcess([], 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(parallel.subprocess, "run", fake_run)
    fstype, source_token = parallel._live_mount_identity(scratch.resolve())

    assert fstype == "nfs4"
    assert source_token == parallel.hashlib.sha256(source.encode("utf-8")).hexdigest()
    assert source not in source_token


def test_storage_root_rejects_gate_from_another_nfs_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PASS 快照若不屬本次 scratch 的 NFS export，必須在建立 session 前拒絕。"""

    scratch = tmp_path / "scratch"
    log_root = scratch / "logs"
    log_root.mkdir(parents=True)
    expected_token = "a" * 64
    gate = tmp_path / "storage-gate.json"
    gate.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "gate_status": "PASS",
                "issues": [],
                "roots": [
                    {
                        "label": label,
                        "fstype": "nfs4",
                        "source_token_hash": expected_token,
                        "gate_status": "PASS",
                    }
                    for label in parallel._GATE_ROOT_LABELS
                ],
                "probes": [
                    {"label": "write_probe", "gate_status": "PASS"},
                    {"label": "same_host_flock_probe", "gate_status": "PASS"},
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        parallel,
        "_live_mount_identity",
        lambda _path: ("nfs4", "b" * 64),
    )

    with pytest.raises(parallel.ParallelExecutionError, match="NFS 來源不一致"):
        parallel.validate_storage_roots(
            scratch_root=scratch.resolve(),
            log_root=log_root.resolve(),
            numba_cache_dir=None,
            gate_evidence=gate.resolve(),
        )


def test_worker_completion_order_does_not_change_run_assignment_identity() -> None:
    """worker 結束順序只影響執行摘要，不會改寫 plan hash 綁定的 assignment bytes。"""

    groups = _groups()
    assignment = parallel.worker_assignment_document(
        run_id="formal-test-run",
        run_plan_sha256="c" * 64,
        worker_count=len(groups),
        groups=groups,
        locality_columns=("study_site_id",),
    )
    records = [
        {
            "worker_id": group.worker_id,
            "exit_code": 0,
            "group": group.to_dict(),
            "worker_result": _worker_result(group),
        }
        for group in groups
    ]
    success_forward, errors_forward = parallel.summarize_worker_completion(
        worker_records=records,
        plan=_plan(),
    )
    success_reverse, errors_reverse = parallel.summarize_worker_completion(
        worker_records=list(reversed(records)),
        plan=_plan(),
    )

    assert success_forward and success_reverse
    assert errors_forward == errors_reverse == []
    assert parallel.canonical_json_bytes(assignment) == parallel.canonical_json_bytes(
        parallel.worker_assignment_document(
            run_id="formal-test-run",
            run_plan_sha256="c" * 64,
            worker_count=len(groups),
            groups=groups,
            locality_columns=("study_site_id",),
        )
    )


def test_executor_writes_independent_logs_and_machine_summary(tmp_path: Path) -> None:
    """假 child 各寫獨立 log，coordinator 以固定 group 順序輸出 exit／時間摘要。"""

    groups = _groups()
    spawned: list[str] = []

    def fake_popen(command: list[str], **kwargs: Any) -> _FakeProcess:
        """記錄 worker id 並把 machine-readable child summary 寫入專屬 stdout log。"""

        spec = json.loads(command[-1])
        worker_id = spec["worker_id"]
        spawned.append(worker_id)
        result = _worker_result(next(group for group in groups if group.worker_id == worker_id))
        payload = {
            **result,
            "worker_id": worker_id,
            "schema_version": "1.0.0",
            "affinity": {"status": "not_requested", "applied": False},
            "run_worker_exit_code": 0,
        }
        kwargs["stdout"].write(json.dumps(payload) + "\n")
        kwargs["stdout"].flush()
        return _FakeProcess(returncode=0)

    records, interrupted = parallel.execute_worker_groups(
        groups,
        workspace=tmp_path / "workspace",
        project_root=tmp_path,
        config=tmp_path / "config.yaml",
        ocm_native_root=None,
        nww_analysis_root=None,
        checkpoint_root=None,
        numba_cache_dir=None,
        log_session_dir=tmp_path,
        affinity_sets=(None,) * len(groups),
        affinity_request="auto",
        affinity_fallback_reasons=("unavailable_platform_fallback",) * len(groups),
        popen_factory=fake_popen,
        poll_interval_seconds=0,
    )

    assert interrupted is None
    assert spawned == [group.worker_id for group in groups]
    assert [row["status"] for row in records] == ["EXITED"] * len(groups)
    assert all(row["whole_batch_wall_seconds"] >= 0 for row in records)
    assert all(row["child_elapsed_seconds"] is not None for row in records)
    assert all(
        row["cpu_affinity_fallback_reason"] == "unavailable_platform_fallback"
        for row in records
    )
    for group in groups:
        log_path = tmp_path / f"{group.worker_id}.log"
        assert log_path.is_file()
        assert json.loads(log_path.read_text().splitlines()[-1])["worker_id"] == group.worker_id


def test_child_failure_stops_remaining_dispatch_and_preserves_logs(tmp_path: Path) -> None:
    """第一個 child 非零退出後不再派發後續 worker，並保留已建立的 log。"""

    groups = _groups()
    spawned: list[str] = []

    def failing_popen(command: list[str], **kwargs: Any) -> _FakeProcess:
        """第一個假 child 立即失敗，其餘 worker 若啟動即記錄為測試失敗。"""

        worker_id = json.loads(command[-1])["worker_id"]
        spawned.append(worker_id)
        kwargs["stdout"].write("synthetic child failure\n")
        return _FakeProcess(returncode=7)

    records, interrupted = parallel.execute_worker_groups(
        groups,
        workspace=tmp_path / "workspace",
        project_root=tmp_path,
        config=tmp_path / "config.yaml",
        ocm_native_root=None,
        nww_analysis_root=None,
        checkpoint_root=None,
        numba_cache_dir=None,
        log_session_dir=tmp_path,
        popen_factory=failing_popen,
        poll_interval_seconds=0,
    )

    assert interrupted is None
    assert spawned == [groups[0].worker_id]
    assert records[0]["status"] == "FAILED"
    assert records[0]["exit_code"] == 7
    assert all(row["status"] == "NOT_DISPATCHED" for row in records[1:])
    assert (tmp_path / f"{groups[0].worker_id}.log").read_text().startswith(
        "synthetic child failure"
    )


@pytest.mark.parametrize("requested_signal", (signal.SIGINT, signal.SIGTERM))
def test_termination_signal_stops_active_worker_and_does_not_mark_batch_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    requested_signal: int,
) -> None:
    """SIGINT／SIGTERM 會傳給 child、記錄退出原因並把未啟動 worker 留為未派發。"""

    groups = _groups()
    spawned: list[_FakeProcess] = []
    received_signals: list[int] = []

    def signal_popen(command: list[str], **kwargs: Any) -> _FakeProcess:
        """建立在首次輪詢時送出 SIGTERM 的可控程序替身。"""

        del command, kwargs
        process = _FakeProcess(returncode=None, signal_on_first_poll=requested_signal)
        spawned.append(process)
        return process

    def fake_signal_process_group(process: _FakeProcess, signum: int) -> None:
        """記錄 coordinator 實際轉送的訊號並立即模擬程序中止。"""

        del process
        received_signals.append(signum)
        spawned[0].returncode = -signum

    monkeypatch.setattr(parallel, "_signal_process_group", fake_signal_process_group)
    records, interrupted = parallel.execute_worker_groups(
        groups,
        workspace=tmp_path / "workspace",
        project_root=tmp_path,
        config=tmp_path / "config.yaml",
        ocm_native_root=None,
        nww_analysis_root=None,
        checkpoint_root=None,
        numba_cache_dir=None,
        log_session_dir=tmp_path,
        popen_factory=signal_popen,
        poll_interval_seconds=0,
        shutdown_grace_seconds=0,
    )

    assert interrupted == requested_signal
    assert received_signals[0] == requested_signal
    assert records[0]["status"] == "STOPPED_BY_SIGNAL"
    assert records[0]["exit_code"] == -requested_signal
    assert all(row["status"] == "NOT_DISPATCHED" for row in records[1:])


def test_preflight_failure_never_calls_child_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    """workspace/provenance preflight 失敗時不建立 session，也不呼叫任何 child factory。"""

    spawned: list[bool] = []

    def reject_preflight(**_: Any) -> Any:
        """在第一個 read-only gate 模擬正式部署拒絕。"""

        raise parallel.ParallelExecutionError("synthetic provenance mismatch")

    monkeypatch.setattr(parallel, "validate_run_inputs", reject_preflight)
    with pytest.raises(parallel.ParallelExecutionError, match="provenance mismatch"):
        parallel.execute_formal_parallel(
            workspace="/unused/workspace",
            config_path="/unused/config.yaml",
            project_root="/unused/project",
            worker_count=1,
            scratch_root="/unused/scratch",
            log_root="/unused/scratch/logs",
            storage_gate_evidence="/unused/gate.json",
            popen_factory=lambda *args, **kwargs: spawned.append(True),
        )

    assert spawned == []


def test_formal_parallel_cli_help_and_exception_summary_are_truthful(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI help公開正式參數，未取得 coordinator 摘要時不猜 child 是否已啟動。"""

    help_text = cli._run_formal_parallel_parser().format_help()
    assert "--worker-count" in help_text
    assert "--storage-gate-evidence" in help_text
    assert "--numba-cache-dir" in help_text
    assert "--warmup-only" in help_text

    def fail_after_unknown_phase(**_: Any) -> Any:
        """模擬無法從通用例外判斷發生在啟動前或摘要階段的 coordinator 錯誤。"""

        raise RuntimeError("synthetic coordinator failure")

    monkeypatch.setattr(parallel, "execute_formal_parallel", fail_after_unknown_phase)
    exit_code = cli.run_formal_parallel(
        [
            "/unused/workspace",
            "--config",
            "/unused/config.yaml",
            "--project-root",
            "/unused/project",
            "--worker-count",
            "1",
            "--scratch-root",
            "/unused/scratch",
            "--log-root",
            "/unused/scratch/logs",
            "--storage-gate-evidence",
            "/unused/gate.json",
        ]
    )
    payload = json.loads(capsys.readouterr().err)

    assert exit_code == 2
    assert payload["artifact_type"] == "formal_parallel_coordinator_error"
    assert payload["status"] == "COORDINATOR_ERROR"
    assert payload["children_started"] is None
