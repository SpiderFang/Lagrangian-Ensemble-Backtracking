"""Unix ``fcntl.flock`` run lock 的跨程序互斥與例外釋放測試。"""

from __future__ import annotations

import multiprocessing
from multiprocessing.connection import Connection
from pathlib import Path

import pytest

from lagrangian_backtracking.run_locking import RunLockBusyError, acquire_run_lock


def _hold_exclusive(path: str, ready: Connection, release: Connection, fail: bool = False) -> None:
    """子程序持有 exclusive lock，供 parent 驗證真正的跨 process contention。"""

    with acquire_run_lock(path, mode="exclusive", blocking=False):
        ready.send("ready")
        if fail:
            raise RuntimeError("intentional worker failure")
        release.recv()


def _try_exclusive(path: str, result: Connection) -> None:
    """子程序以 non-blocking 方式嘗試同一 lock，不能等待或取得。"""

    try:
        with acquire_run_lock(path, mode="exclusive", blocking=False):
            result.send("acquired")
    except RunLockBusyError:
        result.send("busy")


def _hold_gate_and_shard(
    gate_path: str, shard_path: str, ready: Connection, release: Connection
) -> None:
    """以正式 lock order 持有 shared gate 與一個 shard lock。"""

    with acquire_run_lock(gate_path, mode="shared", blocking=False), acquire_run_lock(
        shard_path, mode="exclusive", blocking=False
    ):
        ready.send("ready")
        release.recv()


def _context() -> multiprocessing.context.BaseContext:
    """選擇可在目前 Unix 測試環境傳遞明確 pipe 的 process context。"""

    methods = multiprocessing.get_all_start_methods()
    return multiprocessing.get_context("fork" if "fork" in methods else methods[0])


def test_same_shard_lock_is_busy_across_process_and_released_after_exit(tmp_path: Path) -> None:
    """同 shard 第二程序立即 busy，持有程序結束後 lock 可再次取得。"""

    lock = tmp_path / "shard.lock"
    lock.touch()
    context = _context()
    ready_parent, ready_child = context.Pipe()
    release_parent, release_child = context.Pipe()
    holder = context.Process(target=_hold_exclusive, args=(str(lock), ready_child, release_child))
    holder.start()
    assert ready_parent.recv() == "ready"

    result_parent, result_child = context.Pipe()
    contender = context.Process(target=_try_exclusive, args=(str(lock), result_child))
    contender.start()
    assert result_parent.recv() == "busy"
    contender.join(timeout=5)
    assert contender.exitcode == 0

    release_parent.send("release")
    holder.join(timeout=5)
    assert holder.exitcode == 0
    with acquire_run_lock(lock, mode="exclusive", blocking=False):
        pass


def test_different_shards_share_gate_and_exception_releases_lock(tmp_path: Path) -> None:
    """不同 shard 可共用 gate；worker 例外退出後核心會釋放所有 lock。"""

    gate = tmp_path / "run_gate.lock"
    shard_a = tmp_path / "a.lock"
    shard_b = tmp_path / "b.lock"
    for path in (gate, shard_a, shard_b):
        path.touch()
    context = _context()
    release_a_parent, release_a_child = context.Pipe()
    ready_a_parent, ready_a_child = context.Pipe()
    worker_a = context.Process(
        target=_hold_gate_and_shard,
        args=(str(gate), str(shard_a), ready_a_child, release_a_child),
    )
    worker_a.start()
    assert ready_a_parent.recv() == "ready"

    release_b_parent, release_b_child = context.Pipe()
    ready_b_parent, ready_b_child = context.Pipe()
    worker_b = context.Process(
        target=_hold_gate_and_shard,
        args=(str(gate), str(shard_b), ready_b_child, release_b_child),
    )
    worker_b.start()
    assert ready_b_parent.recv() == "ready"
    release_a_parent.send("release")
    release_b_parent.send("release")
    worker_a.join(timeout=5)
    worker_b.join(timeout=5)
    assert worker_a.exitcode == worker_b.exitcode == 0

    ready_fail_parent, ready_fail_child = context.Pipe()
    release_unused_parent, release_unused_child = context.Pipe()
    del release_unused_parent
    failed = context.Process(
        target=_hold_exclusive,
        args=(str(shard_a), ready_fail_child, release_unused_child, True),
    )
    failed.start()
    assert ready_fail_parent.recv() == "ready"
    failed.join(timeout=5)
    assert failed.exitcode != 0
    with acquire_run_lock(shard_a, mode="exclusive", blocking=False):
        pass


@pytest.mark.parametrize("bad_kind", ["missing", "directory", "symlink", "nonzero"])
def test_lock_file_safety_contract(tmp_path: Path, bad_kind: str) -> None:
    """不存在、目錄、symlink 或非零檔案都不能被當成 lock。"""

    lock = tmp_path / "bad.lock"
    if bad_kind == "directory":
        lock.mkdir()
    elif bad_kind == "symlink":
        target = tmp_path / "target"
        target.touch()
        lock.symlink_to(target)
    elif bad_kind == "nonzero":
        lock.write_text("not-empty", encoding="utf-8")
    with pytest.raises(ValueError), acquire_run_lock(lock, mode="exclusive", blocking=False):
        pass
