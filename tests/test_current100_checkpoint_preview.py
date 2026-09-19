"""未完成預覽沿用真checkpoint讀取器；不允許把損壞或無推進當作軌跡。"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_run_control import _first_shard, _request, _workspace

from lagrangian_backtracking.run_control import RunController, load_run_progress

SPEC = importlib.util.spec_from_file_location(
    "checkpoint_preview", Path(__file__).parents[1] / "scripts/build_current100_checkpoint_preview.py")
preview = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preview)


@pytest.mark.parametrize("corrupt", [False, True])
def test_saved_checkpoint_positions_are_read_only_and_checksum_checked(tmp_path, monkeypatch, corrupt):
    """只縮小測試的情境母體，checkpoint身分／內容驗證仍使用正式讀取器。"""
    workspace = _workspace(tmp_path, "current-checkpoint-preview", run_kind="formal")
    sid = _first_shard(workspace)
    RunController(workspace, request_factory=_request).run_shard(sid, sweep_budget=1)
    progress = load_run_progress(workspace)
    path = workspace / "checkpoints" / progress["shards"][sid]["checkpoint_relative_path"]
    monkeypatch.setattr(preview, "_script", lambda name: SimpleNamespace(
        select_shards=lambda plan, rows: {"gongliao": [sid]}, SELECTION={"gongliao": (1, "arrival")}))
    if corrupt:
        (path / "checkpoint.json").write_text("{}")
        with pytest.raises((ValueError, KeyError)):
            preview.extract_snapshot(workspace, workspace / "checkpoints", "gongliao")
    else:
        before = (workspace / "run_progress.json").read_bytes()
        result = preview.extract_snapshot(workspace, workspace / "checkpoints", "gongliao")
        assert result["complete"] is False
        assert result["particle_count"] > 0
        assert any(p["step_count"] > 0 for p in result["particles"])
        assert all(p["observations"] for p in result["particles"])
        assert (workspace / "run_progress.json").read_bytes() == before
