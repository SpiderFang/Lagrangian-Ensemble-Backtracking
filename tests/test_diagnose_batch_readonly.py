"""隔離診斷的最小安全契約：不發布原run、保留批次平流器及限定清理群組。"""

import importlib.util
import signal
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    'diagnose_batch_readonly', Path(__file__).parents[1] / 'scripts/diagnose_batch_readonly.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@dataclass
class Binding:
    """模擬原執行的唯讀繫結，不建立正式資料。"""
    code_commit: str = 'original'


@pytest.mark.parametrize('restore', [False, True])
def test_only_memory_batch_and_native_injection(tmp_path, restore):
    """控制器發布方法若被呼叫立即失敗；新建／恢復都傳入同一批次平流器。"""
    original = tmp_path / 'run_progress.json'
    original.write_text('original progress')
    before = original.stat().st_mtime_ns
    calls = []

    def forbidden(*args, **kwargs):
        raise AssertionError('不得呼叫原run寫入或控制器恢復')

    class Batch:
        """只記錄建構參數，避免測試另行執行物理或I/O。"""
        def __init__(self, **kwargs):
            calls.append(kwargs)

        @classmethod
        def from_checkpoint(cls, path, **kwargs):
            calls.append({'checkpoint_path': path})
            return cls(**kwargs)

    def native(tasks):
        """保留可辨識的批次平流器參照。"""
        return tasks
    controller = SimpleNamespace(
        _shard=lambda identifier: identifier, _binding=lambda shard: Binding(),
        plan=dict(master_seed=17, random_stream_id='paired', active_chunk_size=10),
        request_factory=object(), advection_batcher=native,
        run_shard=forbidden, restore=forbidden, reconcile=forbidden)
    checkpoint = tmp_path / 'checkpoint-00000043' if restore else None
    runner.build_batch(controller, 'selected', checkpoint, Batch)
    assert calls[-1]['advection_batcher'] is native
    assert calls[-1]['master_seed'] == 17
    assert calls[-1]['random_stream_id'] == 'paired'
    assert calls[-1]['active_chunk_size'] == 10
    if restore:
        assert calls[0]['checkpoint_path'] == checkpoint
        assert isinstance(calls[-1]['expected_binding'], Binding)
    assert original.read_text() == 'original progress'
    assert original.stat().st_mtime_ns == before


def test_missing_native_rejected():
    """缺少原版平流器不得悄悄改用舊核心產生不同重現。"""
    controller = SimpleNamespace(_shard=lambda _: 'shard', advection_batcher=None)
    with pytest.raises(ValueError, match='批次平流器'):
        runner.build_batch(controller, 'selected', None, object)


@pytest.mark.parametrize('group,alive,expected', [(777, True, True), (777, False, False),
                                                (60859, True, False)])
def test_cleanup_only_new_process_group(monkeypatch, group, alive, expected):
    """只對仍存活且自成群組的診斷PID送訊號，不能清理原operator群組。"""
    signals = []
    process = SimpleNamespace(pid=777, poll=lambda: None if alive else 0,
                              wait=lambda timeout: 0)
    monkeypatch.setattr(runner.os, 'getpgid', lambda _: group)
    monkeypatch.setattr(runner.os, 'getpgrp', lambda: 123)
    monkeypatch.setattr(runner.os, 'killpg', lambda *args: signals.append(args))
    if alive and group != 777:
        with pytest.raises(RuntimeError):
            runner.stop_owned(process)
    else:
        runner.stop_owned(process)
    assert signals == ([(777, signal.SIGKILL)] if expected else [])
