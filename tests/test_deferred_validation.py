"""延後整批驗收的起跑 gate 與單片恢復；合成公尺座標／UTC 奈秒，不使用 forcing。"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_run_control import (
    _first_shard,
    _payload_snapshot,
    _request,
    _workspace,
    _write_progress,
)

import lagrangian_backtracking.run_control as control
from lagrangian_backtracking.diffusion import DiffusionCoefficients
from lagrangian_backtracking.outputs import sha256_file
from lagrangian_backtracking.run_control import RunController, load_run_progress
from lagrangian_backtracking.run_locking import RunLockBusyError, acquire_run_lock
from lagrangian_backtracking.run_validation import validate_run, validate_run_startup


def _reject_request(unit):
    """驗證或輸出採認不得觸發物理／forcing request。"""

    raise AssertionError("不得重新計算或載入 forcing")


def _published_crash(parent: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str]:
    """實際發布合法 trajectory 後，在 progress 完成更新前模擬程序中止。"""

    workspace = _workspace(parent, "published-crash")
    shard_id = _first_shard(workspace)
    controller = RunController(workspace, request_factory=_request)

    def crash(*args, **kwargs):
        """保留已發布輸出與原 RUNNING 狀態，重現合法 crash window。"""

        raise SystemExit("published before progress")

    monkeypatch.setattr(controller, "_mark_complete", crash)
    with pytest.raises(SystemExit):
        controller.run_shard(shard_id)
    assert (workspace / "shards" / shard_id / "manifest.json").is_file()
    assert load_run_progress(workspace)["shards"][shard_id]["lifecycle"] == "RUNNING"
    return workspace, shard_id


def _snapshot(root: Path) -> dict[str, bytes]:
    """保留 fixture 檔案位元組以證明拒絕或採認不覆寫軌跡。"""

    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("damage", ["output", "history"])
def test_startup_defers_unrelated_payload_but_default_validation_remains_full(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    """不相關片的壞歷史／输出可延後，完整 validator 仍拒絕且 startup 不讀 payload。"""

    workspace = _workspace(tmp_path, "unrelated-corrupt")
    controller = RunController(workspace, request_factory=_request)
    shard_id = _first_shard(workspace)
    controller.run_shard(shard_id)
    damaged = (
        workspace / "shards" / shard_id / "manifest.json"
        if damage == "output"
        else workspace / "checkpoints" / workspace.name / shard_id / "checkpoint-00000001" / "checkpoint.json"
    )
    damaged.write_text("{broken", encoding="utf-8")
    assert not validate_run(workspace)["valid"]
    before = _snapshot(workspace)

    def no_history(*args, **kwargs):
        """起跑 gate 不可呼叫全片 scanner、restore 或 trajectory validator。"""

        raise AssertionError("startup 不可讀歷史 payload")

    with monkeypatch.context() as patch:
        patch.setattr(control, "load_execution_checkpoint", no_history)
        patch.setattr(control, "validate_trajectory_shard", no_history)
        patch.setattr(RunController, "_scan_checkpoint_generations", no_history)
        original = pq.read_table

        def static_only(path, *args, **kwargs):
            """只容許實際情境表讀取，不能悄悄開啟其他 Parquet payload。"""

            assert Path(path).name == "scenario_table.parquet"
            return original(path, *args, **kwargs)

        patch.setattr(pq, "read_table", static_only)
        result = validate_run_startup(workspace)
    assert result["valid"], result
    assert result["validation_scope"] == "startup_bindings_only"
    assert result["full_validation_deferred"] is True
    assert len(result["deferred_checks"]) == 4
    assert _snapshot(workspace) == before
    # 真正只執行另一片，證明 assigned scanner 不會還原不相關壞歷史。
    other = controller.plan["shards"][1]["shard_id"]
    assert RunController(
        workspace, request_factory=_request, defer_full_validation=True
    ).run_shard(other).lifecycle == "COMPLETE"
    assert not validate_run(workspace)["valid"]


@pytest.mark.parametrize("filename", [
    "normalized_config.json", "input_inventory.json", "scenario_table.parquet",
    "seed_table.parquet", "run_plan.json", "run_progress.json",
])
def test_startup_rejects_immutable_and_plan_progress_damage(tmp_path: Path, filename: str) -> None:
    """所有 immutable 檔案 checksum 與 plan/progress gate 均不可因延後全驗而跳過。"""

    workspace = _workspace(tmp_path, "broken-static")
    (workspace / filename).write_bytes(b"{broken")
    result = validate_run_startup(workspace)
    assert not result["valid"]
    assert result["full_validation_deferred"] is True
    assert result["validation_scope"] == "startup_bindings_only"


@pytest.mark.parametrize("filename,field,value", [
    ("seed_table.parquet", "seed_128_hex", "0" * 32),
    ("seed_table.parquet", "particle_id", "wrong-particle"),
    ("scenario_table.parquet", "settling_velocity_mps", -0.25),
])
def test_startup_rejects_rehashed_identity_seed_and_scenario_changes(
    tmp_path: Path, filename: str, field: str, value
) -> None:
    """重新計算檔案 SHA 也不能繞過 seed 導出規則與分片 scenario hash。"""

    workspace = _workspace(tmp_path, "rehash-static")
    path = workspace / filename
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows[0][field] = value
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    plan_path = workspace / "run_plan.json"
    plan = json.loads(plan_path.read_text())
    plan["files"][filename].update(size_bytes=path.stat().st_size, sha256=sha256_file(path))
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    assert not validate_run_startup(workspace)["valid"]


@pytest.mark.parametrize("damage", [
    "unknown_root", "root_symlink", "lock_symlink", "output_unknown", "output_symlink",
    "checkpoint_unknown", "checkpoint_symlink", "failure_unknown", "progress_range",
])
def test_startup_preserves_topology_and_progress_safety(tmp_path: Path, damage: str) -> None:
    """只盤點安全拓樸仍需拒絕未知項目、symlink 與 plan/progress 範圍不一致。"""

    workspace = _workspace(tmp_path, "startup-topology")
    shard_id = _first_shard(workspace)
    if damage == "unknown_root":
        (workspace / "unexpected").touch()
    elif damage == "root_symlink":
        alias = tmp_path / "alias"
        alias.symlink_to(workspace, target_is_directory=True)
        workspace = alias
    elif damage == "lock_symlink":
        path = workspace / "locks" / "run_gate.lock"
        path.unlink()
        path.symlink_to(workspace / "locks" / "progress.lock")
    elif damage.startswith("output"):
        path = workspace / "shards" / (shard_id if damage.endswith("symlink") else "unknown")
        if damage.endswith("symlink"):
            path.symlink_to(tmp_path, target_is_directory=True)
        else:
            path.mkdir()
    elif damage.startswith("checkpoint"):
        run_root = workspace / "checkpoints" / workspace.name
        run_root.mkdir(parents=True)
        path = run_root / (shard_id if damage.endswith("symlink") else "unknown")
        if damage.endswith("symlink"):
            path.symlink_to(tmp_path, target_is_directory=True)
        else:
            path.mkdir()
    elif damage == "failure_unknown":
        (workspace / "failures" / "unknown").mkdir()
    else:
        progress = load_run_progress(workspace)
        progress["shards"][shard_id]["scenario_stop_index"] += 1
        (workspace / "run_progress.json").write_text(json.dumps(progress), encoding="utf-8")
    assert not validate_run_startup(workspace)["valid"]


def test_startup_validates_external_checkpoint_topology(tmp_path: Path) -> None:
    """明示外置根時同樣拒絕未知 shard 與根 symlink，不掃其歷代 payload。"""

    workspace = _workspace(tmp_path / "run", "external-startup")
    external = tmp_path / "external"
    (external / workspace.name / "unknown").mkdir(parents=True)
    assert not validate_run_startup(workspace, checkpoint_root=external)["valid"]
    alias = tmp_path / "alias"
    alias.symlink_to(external, target_is_directory=True)
    assert not validate_run_startup(workspace, checkpoint_root=alias)["valid"]


@pytest.mark.parametrize("flag", [0, 1, "true", None])
def test_controller_deferred_flag_is_strict_bool(tmp_path: Path, flag) -> None:
    """避免字串／整數 truthiness 在未明示授權時意外進入延後驗收模式。"""

    with pytest.raises(TypeError, match="defer_full_validation"):
        RunController(tmp_path, request_factory=_reject_request, defer_full_validation=flag)


def test_deferred_adopts_running_published_output_without_recomputing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """只採認已發布且完整驗證的 RUNNING 輸出，不抽亂數、不重算、不改 payload。"""

    workspace, shard_id = _published_crash(tmp_path, monkeypatch)
    before = _snapshot(workspace / "shards")
    attempts = load_run_progress(workspace)["shards"][shard_id]["attempt_count"]
    controller = RunController(
        workspace, request_factory=_reject_request, resume=True, defer_full_validation=True
    )
    summary = controller.run_shard(shard_id)
    assert summary.lifecycle == "COMPLETE"
    assert controller.progress["shards"][shard_id]["attempt_count"] == attempts
    assert _snapshot(workspace / "shards") == before
    # 已 COMPLETE 的第二次呼叫仍走原 validator，而不是略過此片輸出驗證。
    assert controller.run_shard(shard_id) == summary
    assert validate_run(workspace)["valid"]


@pytest.mark.parametrize("lifecycle", ["PLANNED", "PAUSED", "FAILED"])
def test_deferred_never_adopts_nonrunning_published_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lifecycle: str
) -> None:
    """不合法 published-before-progress 狀態不得被 mark_running 掩蓋或覆寫輸出。"""

    workspace, shard_id = _published_crash(tmp_path, monkeypatch)
    controller = RunController(workspace, request_factory=_request, resume=True)
    if lifecycle == "PLANNED":
        # 保留 checkpoint 以確認 PLANNED + 外來歷史在原 scanner cross-link 即被拒絕。
        progress = load_run_progress(workspace)
        progress["shards"][shard_id] = load_run_progress(
            _workspace(tmp_path / "template", "template")
        )["shards"][shard_id]
        progress["run_lifecycle"] = "PLANNED"
        _write_progress(workspace, progress)
    else:
        def change(progress):
            """建立合法暫停／失敗列，僅用來驗證不能採認既有輸出。"""

            progress["shards"][shard_id]["lifecycle"] = lifecycle

        controller._update_progress(change)
    before = _snapshot(workspace)
    with pytest.raises(ValueError):
        RunController(
            workspace, request_factory=_reject_request, resume=True, defer_full_validation=True
        ).run_shard(shard_id)
    assert _snapshot(workspace) == before


@pytest.mark.parametrize("damage", ["output", "history", "metadata", "identity"])
def test_deferred_assigned_corruption_rejected_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    """指派片的完整歷史、payload 或 metadata 損壞均須在 request 前失敗。"""

    workspace, shard_id = _published_crash(tmp_path, monkeypatch)
    manifest_path = workspace / "shards" / shard_id / "manifest.json"
    if damage == "history":
        path = (
            workspace / "checkpoints" / workspace.name / shard_id
            / "checkpoint-00000001" / "checkpoint.json"
        )
        path.write_text("{broken", encoding="utf-8")
    elif damage == "output":
        manifest_path.write_text("{broken", encoding="utf-8")
    elif damage == "metadata":
        manifest = json.loads(manifest_path.read_text())
        manifest["run_metadata"]["config_hash"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    else:
        # 重新計算檔案 checksum 仍不能交換 member 的 RunUnit 次序。
        path = manifest_path.parent / "particle_table.parquet"
        table = pq.read_table(path)
        pq.write_table(pa.Table.from_pylist(list(reversed(table.to_pylist())), schema=table.schema), path)
        manifest = json.loads(manifest_path.read_text())
        manifest["files"][path.name].update(size_bytes=path.stat().st_size, sha256=sha256_file(path))
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    before = _snapshot(workspace)
    with pytest.raises(ValueError):
        RunController(
            workspace, request_factory=_reject_request, resume=True, defer_full_validation=True
        ).run_shard(shard_id)
    assert _snapshot(workspace) == before


def test_deferred_resume_keeps_nonzero_diffusion_and_rng(tmp_path: Path) -> None:
    """非零擴散合成案例的 checkpoint 續跑須與一次執行的數值與 RNG 狀態一致。"""

    def request(unit):
        """沿用合成公尺座標 request，明示非零擴散係數（平方公尺／秒）。"""

        return replace(_request(unit), diffusion=DiffusionCoefficients(0.001, 0.001, 0.001))

    baseline = _workspace(tmp_path / "baseline", "baseline")
    resumed = _workspace(tmp_path / "resumed", "resumed")
    shard_id = _first_shard(baseline)
    RunController(baseline, request_factory=request).run_shard(shard_id)
    RunController(resumed, request_factory=request).run_shard(shard_id, sweep_budget=3)
    RunController(
        resumed, request_factory=request, resume=True, defer_full_validation=True
    ).run_shard(shard_id)
    assert _payload_snapshot(baseline, shard_id) == _payload_snapshot(resumed, shard_id)
    # 末代 checkpoint 的亂數狀態亦逐粒子精確比較，不只比位置。
    def latest_rng(workspace):
        """依原 scanner／loader 讀取最高代 RNG 狀態，測試不修改 checkpoint。"""

        controller = RunController(workspace, request_factory=request)
        shard = controller._shard(shard_id)
        selected = controller._scan_checkpoint_generations(shard)
        loaded = control.load_execution_checkpoint(selected.path, expected_binding=controller._binding(shard))
        return loaded.rng_states

    assert latest_rng(baseline) == latest_rng(resumed)


def test_deferred_adoption_preserves_existing_lock_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """同片鎖與全 run gate 仍可阻止採認；不取得全 run exclusive 來繞過 worker 互斥。"""

    workspace, shard_id = _published_crash(tmp_path, monkeypatch)
    controller = RunController(
        workspace, request_factory=_reject_request, resume=True, defer_full_validation=True
    )
    before = _snapshot(workspace)
    for lock_name in ("run_gate.lock", f"{shard_id}.lock"):
        with (
            acquire_run_lock(workspace / "locks" / lock_name, mode="exclusive", blocking=False),
            pytest.raises(RunLockBusyError),
        ):
            controller.run_shard(shard_id)
    assert _snapshot(workspace) == before
