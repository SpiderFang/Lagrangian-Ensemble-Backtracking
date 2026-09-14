"""run plan/progress、external checkpoint、restart 與 reconcile 契約測試。"""

from __future__ import annotations

import json
import multiprocessing
import shutil
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from shapely.geometry import box

import lagrangian_backtracking.run_control as run_control_module
from lagrangian_backtracking.boundaries import BoundaryGeometry
from lagrangian_backtracking.diffusion import DiffusionCoefficients
from lagrangian_backtracking.engine import EngineSettings
from lagrangian_backtracking.models import ParticleState, VelocitySample
from lagrangian_backtracking.outputs import sha256_file, validate_trajectory_shard
from lagrangian_backtracking.production import ProductionBatch, ReferenceParticleRequest
from lagrangian_backtracking.provenance import CodeProvenance
from lagrangian_backtracking.run_control import (
    RUN_PROGRESS_SCHEMA_VERSION,
    RunController,
    initialize_run_workspace,
    load_run_plan,
    load_run_progress,
    validate_run_plan_document,
    validate_run_progress_document,
)
from lagrangian_backtracking.run_locking import RunLockBusyError, acquire_run_lock
from lagrangian_backtracking.run_validation import benchmark_report, validate_run
from lagrangian_backtracking.scenarios import Scenario, stable_identifier

_CONFIG_HASH = "9" * 64
_COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _progress_worker(workspace: str, shard_id: str, result_pipe) -> None:
    """子程序只更新一個 shard，驗證 progress lock 不遺失另一程序的 mutation。"""

    try:
        controller = RunController(workspace, request_factory=_request)

        def update(progress: dict) -> None:
            """建立合法 RUNNING row，模擬 worker 的 progress mutation。"""

            row = progress["shards"][shard_id]
            row["lifecycle"] = "RUNNING"
            row["attempt_count"] = 1
            row["metrics"] = {
                "wall_seconds": 0.0,
                "process_cpu_seconds": 0.0,
                "max_rss_bytes": 0,
                "output_bytes": 0,
                "checkpoint_bytes": 0,
                "particle_steps": 0,
            }

        controller._update_progress(update)
        result_pipe.send("ok")
    except BaseException as error:  # pragma: no cover - parent 會將子程序錯誤轉成 assertion
        result_pipe.send(f"error:{type(error).__name__}:{error}")
        raise


def _provenance(*, commit: str | None = _COMMIT, dirty: bool | None = False) -> CodeProvenance:
    """建立與正式來源規則一致的 deterministic provenance fixture。

    ``dirty is not None`` 代表 fixture 模擬有 Git repository；``dirty is None`` 則模擬
    無 Git 的部署，因此有 commit 時只能標成 ``declared_deployment``，沒有 commit 時
    只能標成 ``no_git_pilot``。測試不可使用任意 ``test_fixture`` source 繞過正式 gate。
    """

    commit_source = (
        "git_repository"
        if dirty is not None
        else "declared_deployment"
        if commit is not None
        else "no_git_pilot"
    )

    return CodeProvenance(
        git_available=dirty is not None,
        git_commit=commit,
        git_dirty=dirty,
        commit_source=commit_source,
        deployment_tree_sha256="1" * 64,
        deployment_file_count=3,
        uv_lock_sha256="2" * 64,
        python_version="3.12.0",
        platform="test-platform",
        package_version="0.1.0",
        numpy_version="test",
        numba_version="test",
        pyarrow_version="test",
    )


def _scenario(index: int) -> Scenario:
    """建立具有 stable scenario ID 的小型合成情境。"""

    receptor = f"receptor-{index}"
    arrival = f"arrival-{index}"
    return Scenario(
        scenario_id=stable_identifier("scn", ["gongliao", "material", receptor, arrival, "test-v1"]),
        study_site_id="gongliao",
        analysis_region_id="A",
        material_id="material",
        receptor_id=receptor,
        arrival_time_id=arrival,
        settling_velocity_mps=-0.001,
        arrival_time_utc_ns=1_700_000_000_000_000_000 + index,
        design_version="test-v1",
    )


def _request(unit) -> ReferenceParticleRequest:
    """建立零擴散、固定速度 request；七秒上限提供多個 checkpoint cadence。"""

    state = ParticleState(
        particle_id=unit.particle_id,
        scenario_id=unit.scenario.scenario_id,
        member_id=unit.member_id,
        study_site_id=unit.scenario.study_site_id,
        analysis_region_id=unit.scenario.analysis_region_id,
        receptor_id=unit.scenario.receptor_id,
        x_m=0.0,
        y_m=0.0,
        z_m=-5.0,
        time_utc_ns=unit.scenario.arrival_time_utc_ns,
    )

    def velocity(x_m: float, y_m: float, z_m: float, time_utc_ns: int) -> VelocitySample:
        """回傳有限常流樣本；輸入參數只用於符合 reference provider 介面。"""

        del x_m, y_m, z_m, time_utc_ns
        return VelocitySample(0.0, 0.0, 0.0, 0.0, -10.0, 100.0, 10.0)

    return ReferenceParticleRequest(
        initial_state=state,
        velocity=velocity,
        boundaries=BoundaryGeometry(
            box(-5.0, -5.0, 5.0, 5.0),
            box(-20.0, -20.0, 20.0, 20.0),
            {},
        ),
        behavior_class="sinking",
        diffusion=DiffusionCoefficients(0.0, 0.0, 0.0),
        settings=EngineSettings(1.0, 1.0, 1.0, 7.0, 20, 0),
    )


def _workspace(
    parent: Path,
    run_id: str,
    *,
    interval: int = 1,
    run_kind: str = "synthetic",
    provenance: CodeProvenance | None = None,
) -> Path:
    """建立兩情境×兩 member 的 tracked run workspace。"""

    workspace = initialize_run_workspace(
        parent,
        run_id=run_id,
        scenarios=(_scenario(0), _scenario(1)),
        normalized_config={"schema_version": "test", "settings": {"dt": 1.0}},
        config_hash=_CONFIG_HASH,
        input_inventory_file={"source": "synthetic", "files": []},
        component_canonical_hashes={"material": "a" * 64, "receptor": "b" * 64, "arrival": "c" * 64},
        geometry_canonical_hashes={
            "domain": "d" * 64,
            "local": "e" * 64,
            "open_boundary": "f" * 64,
        },
        provenance=provenance or _provenance(),
        experiment_case_id="baseline",
        master_seed=20260819,
        seed_policy="sha256_v1_pcg64dxsm",
        members_per_scenario=2,
        shard_scenario_count=1,
        checkpoint_interval_sweeps=interval,
        active_chunk_size=2,
        run_kind=run_kind,
    )
    return workspace.path


def _first_shard(workspace: Path) -> str:
    """取得 plan 固定排序的第一個 shard ID。"""

    return str(load_run_plan(workspace)["shards"][0]["shard_id"])


def _payload_snapshot(workspace: Path, shard_id: str) -> tuple[bytes, ...]:
    """比較排除 runtime metadata 的粒子、事件與 trajectory arrays。"""

    root = workspace / "shards" / shard_id
    table = pq.read_table(root / "particle_table.parquet").to_pylist()
    events = pq.read_table(root / "events.parquet").to_pylist()
    arrays = tuple(
        np.load(root / name, allow_pickle=False).tobytes()
        for name in (
            "trajectory_offsets.npy",
            "time_utc_ns.npy",
            "age_seconds.npy",
            "x_m.npy",
            "y_m.npy",
            "z_m.npy",
            "status_code.npy",
        )
    )
    return (repr(table).encode(), repr(events).encode(), *arrays)


def _write_progress(workspace: Path, progress: dict) -> None:
    """僅供 crash-window fixture 改寫 tmp_path 內 progress。"""

    (workspace / "run_progress.json").write_text(
        json.dumps(progress, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _counter_progress_snapshot(
    *,
    lifecycle: str,
    sweeps_completed: int,
    particle_steps: int,
) -> dict:
    """建立單一 shard 的最小合法 progress 文件，專門測試 sweep/step 下界語意。

    ``sweeps_completed`` 是外層 sweep 呼叫計數，``particle_steps`` 是實際成功粒子步進
    計數；fixture 只提供 validator 所需的 path、metrics 與 lifecycle 欄位，不建立實體
    checkpoint/output。COMPLETE／FAILED 的 output/failure token 只需符合文件 schema，
    因為本測試驗證的是純文件 counter gate，而非檔案存在性。
    """

    shard_id = "shard-0"
    row = {
        "shard_id": shard_id,
        "scenario_start_index": 0,
        "scenario_stop_index": 1,
        "lifecycle": lifecycle,
        "checkpoint_sequence": 1 if lifecycle == "PAUSED" else 0,
        "checkpoint_relative_path": (
            "counter-run/shard-0/checkpoint-00000001" if lifecycle == "PAUSED" else None
        ),
        "output_relative_path": "shards/shard-0" if lifecycle == "COMPLETE" else None,
        "attempt_count": 0 if lifecycle == "PLANNED" else 1,
        "sweeps_completed": sweeps_completed,
        "particle_steps": particle_steps,
        "metrics": {
            "wall_seconds": 0.0,
            "process_cpu_seconds": 0.0,
            "max_rss_bytes": 0,
            "output_bytes": 0,
            "checkpoint_bytes": 0,
            "particle_steps": particle_steps,
        },
        "error_code": "test_failure" if lifecycle == "FAILED" else None,
        "failure_relative_path": (
            "failures/shard-0/failure-00000000000000000000000000000000.json"
            if lifecycle == "FAILED"
            else None
        ),
    }
    return {
        "schema_version": RUN_PROGRESS_SCHEMA_VERSION,
        "run_id": "counter-run",
        "revision": 1,
        "updated_at_utc": "2026-08-31T00:00:00Z",
        "run_lifecycle": lifecycle,
        "shards": {shard_id: row},
    }


@pytest.mark.parametrize(
    ("lifecycle", "sweeps_completed", "particle_steps", "accepted"),
    [
        ("COMPLETE", 728, 727, True),
        ("COMPLETE", 1, 0, True),
        ("COMPLETE", 728, 726, False),
        ("PAUSED", 1, 0, False),
        ("COMPLETE", 728, 728, True),
        ("RUNNING", 1, 1, True),
        ("FAILED", 728, 727, True),
    ],
)
def test_progress_counter_lower_bound_depends_on_terminal_lifecycle(
    lifecycle: str,
    sweeps_completed: int,
    particle_steps: int,
    accepted: bool,
) -> None:
    """COMPLETE／FAILED 只允許一個 terminal-only sweep，PAUSED/RUNNING 不可放寬。"""

    document = _counter_progress_snapshot(
        lifecycle=lifecycle,
        sweeps_completed=sweeps_completed,
        particle_steps=particle_steps,
    )
    if accepted:
        assert validate_run_progress_document(document) == document
    else:
        with pytest.raises(ValueError, match="particle_steps"):
            validate_run_progress_document(document)


def test_initialize_rejects_unsafe_id_existing_directory_and_symlink(tmp_path: Path) -> None:
    """run_id 不可 traversal/absolute，既有目錄或 symlink target 均不可覆寫。"""

    for run_id in ("../escape", "/absolute", "a/b", ".."):
        with pytest.raises(ValueError, match="slug|traversal|separator"):
            _workspace(tmp_path / "runs", run_id)
    workspace = _workspace(tmp_path / "runs", "existing-run")
    assert workspace.is_dir()
    with pytest.raises(FileExistsError):
        _workspace(tmp_path / "runs", "existing-run")
    target = tmp_path / "symlink-target"
    target.mkdir()
    (tmp_path / "runs" / "linked-run").symlink_to(target, target_is_directory=True)
    with pytest.raises(FileExistsError):
        _workspace(tmp_path / "runs", "linked-run")


def test_initialize_plan_progress_and_seed_table_are_strictly_bound(tmp_path: Path) -> None:
    """初始 plan/progress count、range、checksum 與 128-bit seed hex 必須一致。"""

    workspace = _workspace(tmp_path, "initial-contract")
    plan = load_run_plan(workspace)
    progress = load_run_progress(workspace)
    seeds = pq.ParquetFile(workspace / "seed_table.parquet")
    assert plan["scenario_count"] == 2
    assert plan["particle_count"] == 4
    assert plan["shard_count"] == 2
    assert progress["run_lifecycle"] == "PLANNED"
    rows = [row for batch in seeds.iter_batches(batch_size=2) for row in batch.to_pylist()]
    assert len(rows) == 4
    assert all(len(row["seed_128_hex"]) == 32 for row in rows)


def test_schema21_ordering_group_and_selection_topology_are_published(tmp_path: Path) -> None:
    """schema 2.1 plan 必須保存 selection、ordering/group metadata 與完整 lock set。"""

    workspace = _workspace(tmp_path, "schema2-topology")
    plan = load_run_plan(workspace)
    assert plan["schema_version"] == "2.1.0"
    assert plan["scenario_selection"]["mode"] == "full"
    assert plan["scenario_selection"]["selected_scenario_count"] == plan["scenario_count"]
    assert plan["scenario_ordering_policy"]
    assert plan["lock_root"] == "locks"
    assert {path.name for path in (workspace / "locks").iterdir()} == {
        "run_gate.lock",
        "progress.lock",
        *(f"{row['shard_id']}.lock" for row in plan["shards"]),
    }
    assert all(
        {
            "execution_group_id",
            "analysis_region_id",
            "arrival_time_utc_ns",
            "group_part_index",
            "group_part_count",
        }
        <= set(row)
        for row in plan["shards"]
    )


def test_legacy_schema20_plan_is_read_only_full_compatible(tmp_path: Path) -> None:
    """2.0 plan 可按 full 讀取，但不可帶新 selection 或缺少 2.1 selection。"""

    workspace = _workspace(tmp_path, "legacy-plan-compatibility")
    current = load_run_plan(workspace)
    legacy = deepcopy(current)
    legacy["schema_version"] = "2.0.0"
    del legacy["scenario_selection"]
    assert validate_run_plan_document(legacy) == legacy

    legacy_with_selection = deepcopy(legacy)
    legacy_with_selection["scenario_selection"] = current["scenario_selection"]
    with pytest.raises(ValueError, match="未知欄位"):
        validate_run_plan_document(legacy_with_selection)

    missing_selection = deepcopy(current)
    del missing_selection["scenario_selection"]
    with pytest.raises(ValueError, match="scenario_selection"):
        validate_run_plan_document(missing_selection)


def test_run_shard_contention_fails_before_request_factory_and_reconcile_is_busy(
    tmp_path: Path,
) -> None:
    """同 shard contention 與 active worker reconcile 都不得進入物理或修改狀態。"""

    workspace = _workspace(tmp_path, "lock-contention")
    shard_id = _first_shard(workspace)
    calls = 0

    def counted(unit):
        """驗證同 shard lock busy 時 request factory 不會被呼叫。"""

        nonlocal calls
        calls += 1
        return _request(unit)

    controller = RunController(workspace, request_factory=counted)
    with acquire_run_lock(workspace / "locks" / f"{shard_id}.lock", mode="exclusive", blocking=False):
        with pytest.raises(RunLockBusyError):
            controller.run_shard(shard_id)
        assert calls == 0

    progress_before = (workspace / "run_progress.json").read_bytes()
    with acquire_run_lock(workspace / "locks" / "run_gate.lock", mode="shared", blocking=False):
        with pytest.raises(RunLockBusyError):
            controller.reconcile()
        assert (workspace / "run_progress.json").read_bytes() == progress_before


def test_progress_updates_from_two_processes_preserve_revision_and_lifecycle(tmp_path: Path) -> None:
    """兩個 process 更新不同 shard 後，revision 與兩列狀態都必須保留。"""

    workspace = _workspace(tmp_path, "progress-race")
    shard_ids = [str(row["shard_id"]) for row in load_run_plan(workspace)["shards"]]
    context = multiprocessing.get_context("fork")
    pipes = [context.Pipe() for _ in shard_ids]
    workers = [
        context.Process(target=_progress_worker, args=(str(workspace), shard_id, child))
        for shard_id, (parent, child) in zip(shard_ids, pipes, strict=True)
    ]
    for worker in workers:
        worker.start()
    for parent, _ in pipes:
        assert parent.poll(5)
        assert parent.recv() == "ok"
    for worker in workers:
        worker.join(timeout=5)
        assert worker.exitcode == 0
    progress = load_run_progress(workspace)
    assert progress["revision"] == 2
    assert progress["run_lifecycle"] == "RUNNING"
    assert all(row["lifecycle"] == "RUNNING" for row in progress["shards"].values())


def test_formal_provenance_gate_and_strict_output(tmp_path: Path) -> None:
    """formal 接受 Git clean 或無 Git declared commit，且 clean Git output 維持 strict gate。"""

    with pytest.raises(ValueError, match="formal"):
        _workspace(
            tmp_path / "missing",
            "formal-missing",
            run_kind="formal",
            provenance=_provenance(commit=None, dirty=None),
        )
    with pytest.raises(ValueError, match="formal"):
        _workspace(
            tmp_path / "dirty",
            "formal-dirty",
            run_kind="formal",
            provenance=_provenance(dirty=True),
        )
    declared = _workspace(
        tmp_path / "declared",
        "formal-declared",
        run_kind="formal",
        provenance=_provenance(commit=_COMMIT, dirty=None),
    )
    declared_plan = load_run_plan(declared)
    assert declared_plan["code_provenance"]["git_available"] is False
    assert declared_plan["code_provenance"]["git_commit"] == _COMMIT
    assert declared_plan["code_provenance"]["git_dirty"] is None
    assert declared_plan["code_provenance"]["commit_source"] == "declared_deployment"
    workspace = _workspace(tmp_path / "clean", "formal-clean", run_kind="formal")
    summary = RunController(workspace, request_factory=_request).run_shard(_first_shard(workspace))
    validation = validate_trajectory_shard(
        workspace / str(summary.output_relative_path),
        require_formal_metadata=True,
        strict_run_metadata=True,
    )
    assert validation["valid"], validation
    pilot = _workspace(
        tmp_path / "pilot",
        "pilot-nullable-git",
        run_kind="pilot",
        provenance=_provenance(commit=None, dirty=None),
    )
    pilot_summary = RunController(pilot, request_factory=_request).run_shard(_first_shard(pilot))
    pilot_validation = validate_trajectory_shard(
        pilot / str(pilot_summary.output_relative_path),
        strict_run_metadata=True,
    )
    assert pilot_validation["valid"], pilot_validation


def test_provenance_source_flags_cannot_use_arbitrary_fixture_source(tmp_path: Path) -> None:
    """run plan 不得接受與 Git／declared deployment 旗標矛盾的任意 source。"""

    invalid = replace(_provenance(), commit_source="test_fixture")
    with pytest.raises(ValueError, match="commit_source"):
        _workspace(tmp_path, "invalid-provenance-source", provenance=invalid)


def test_external_pause_resume_matches_uninterrupted_and_wrong_root_fails_early(tmp_path: Path) -> None:
    """external root restart 保持逐位等價，錯 root 必須在 request factory 前失敗。"""

    uninterrupted = _workspace(tmp_path / "full", "run-full", interval=2)
    external_run = _workspace(tmp_path / "paused", "run-external", interval=2)
    shard_id = _first_shard(uninterrupted)
    RunController(uninterrupted, request_factory=_request).run_shard(shard_id)
    external_root = tmp_path / "external-checkpoints"
    paused = RunController(
        external_run,
        request_factory=_request,
        checkpoint_root=external_root,
    ).run_shard(shard_id, sweep_budget=1)
    assert paused.lifecycle == "PAUSED"
    calls = 0

    def counted_request(unit):
        """記錄 wrong-root 是否錯誤進入 request factory。"""

        nonlocal calls
        calls += 1
        return _request(unit)

    with pytest.raises(ValueError, match="找不到 generation"):
        RunController(
            external_run,
            request_factory=counted_request,
            resume=True,
            checkpoint_root=tmp_path / "wrong-root",
        ).run_shard(shard_id)
    assert calls == 0
    RunController(
        external_run,
        request_factory=_request,
        resume=True,
        checkpoint_root=external_root,
    ).run_shard(shard_id)
    assert _payload_snapshot(uninterrupted, shard_id) == _payload_snapshot(external_run, shard_id)
    assert validate_run(external_run, require_complete=False, checkpoint_root=external_root)["valid"]
    assert not validate_run(external_run)["valid"]
    report = benchmark_report(external_run, checkpoint_root=external_root)
    assert report["valid"] and str(external_root) not in repr(report)


def test_running_requires_resume_and_planned_rejects_stray_generation(tmp_path: Path) -> None:
    """RUNNING 必須明示 resume；PLANNED 不可採認任何外來 generation。"""

    workspace = _workspace(tmp_path / "running", "running-run")
    shard_id = _first_shard(workspace)
    progress = load_run_progress(workspace)
    progress["run_lifecycle"] = "RUNNING"
    progress["shards"][shard_id]["lifecycle"] = "RUNNING"
    progress["shards"][shard_id]["attempt_count"] = 1
    _write_progress(workspace, progress)
    calls = 0

    def counted(unit):
        """確認未授權 resume 不建立 request。"""

        nonlocal calls
        calls += 1
        return _request(unit)

    with pytest.raises(RuntimeError, match="resume=True"):
        RunController(workspace, request_factory=counted).run_shard(shard_id)
    assert calls == 0

    stray = _workspace(tmp_path / "planned", "planned-run")
    stray_shard = _first_shard(stray)
    RunController(stray, request_factory=_request).run_shard(stray_shard, sweep_budget=1)
    progress = load_run_progress(stray)
    row = progress["shards"][stray_shard]
    row.update(
        {
            "lifecycle": "PLANNED",
            "checkpoint_sequence": 0,
            "checkpoint_relative_path": None,
            "attempt_count": 0,
            "sweeps_completed": 0,
            "particle_steps": 0,
            "metrics": {},
        }
    )
    progress["run_lifecycle"] = "PLANNED"
    _write_progress(stray, progress)
    with pytest.raises(ValueError, match="PLANNED"):
        RunController(stray, request_factory=_request, resume=True).run_shard(stray_shard)


@pytest.mark.parametrize("interval", [2, 3])
def test_off_boundary_pause_resume_keeps_periodic_checkpoint_cadence(tmp_path: Path, interval: int) -> None:
    """off-boundary pause 後仍從上一 generation 起最多 interval sweeps 建立下一代。"""

    workspace = _workspace(tmp_path, f"cadence-{interval}", interval=interval)
    shard_id = _first_shard(workspace)
    first = RunController(workspace, request_factory=_request).run_shard(shard_id, sweep_budget=1)
    assert first.lifecycle == "PAUSED"
    second = RunController(workspace, request_factory=_request, resume=True).run_shard(
        shard_id,
        sweep_budget=interval,
    )
    assert second.lifecycle == "PAUSED"
    parent = workspace / "checkpoints" / workspace.name / shard_id
    assert (parent / "checkpoint-00000001").is_dir()
    assert (parent / "checkpoint-00000002").is_dir()
    progress = load_run_progress(workspace)["shards"][shard_id]
    assert progress["checkpoint_sequence"] == 2
    assert progress["sweeps_completed"] == interval + 1
    actual_active_bytes = sum(
        path.stat().st_size
        for path in parent.rglob("*")
        if path.is_file()
    )
    assert progress["metrics"]["checkpoint_active_bytes"] == actual_active_bytes


def test_periodic_checkpoint_updates_progress_before_next_step_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """第一個 periodic checkpoint 後的 progress 必須在下一次 advance 失敗前可見。"""

    workspace = _workspace(tmp_path, "periodic-progress", interval=2)
    shard_id = _first_shard(workspace)
    original = ProductionBatch.advance
    calls = 0

    def fail_second_advance(self: ProductionBatch, sweeps: int = 1):
        """第一段正常前進，第二段模擬物理 exception。"""

        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("下一 interval 失敗")
        return original(self, sweeps)

    monkeypatch.setattr(ProductionBatch, "advance", fail_second_advance)
    with pytest.raises(RuntimeError, match="下一 interval"):
        RunController(workspace, request_factory=_request).run_shard(shard_id)
    row = load_run_progress(workspace)["shards"][shard_id]
    assert row["lifecycle"] == "FAILED"
    assert row["checkpoint_sequence"] == 1
    assert row["checkpoint_relative_path"].endswith("checkpoint-00000001")
    assert row["sweeps_completed"] == 2
    assert row["particle_steps"] > 0


def test_missing_and_stale_latest_are_repaired_without_changing_state(tmp_path: Path) -> None:
    """missing/stale latest 可由最高完整 generation 修復，progress counters 保持一致。"""

    workspace = _workspace(tmp_path / "missing", "missing-latest", interval=2)
    shard_id = _first_shard(workspace)
    RunController(workspace, request_factory=_request).run_shard(shard_id, sweep_budget=1)
    parent = workspace / "checkpoints" / workspace.name / shard_id
    (parent / "latest.json").unlink()
    RunController(workspace, request_factory=_request, resume=True).reconcile()
    assert (parent / "latest.json").is_file()

    stale = _workspace(tmp_path / "stale", "stale-latest", interval=2)
    stale_shard = _first_shard(stale)
    RunController(stale, request_factory=_request).run_shard(stale_shard, sweep_budget=1)
    stale_parent = stale / "checkpoints" / stale.name / stale_shard
    pointer_one = (stale_parent / "latest.json").read_text(encoding="utf-8")
    RunController(stale, request_factory=_request, resume=True).run_shard(stale_shard, sweep_budget=2)
    (stale_parent / "latest.json").write_text(pointer_one, encoding="utf-8")
    RunController(stale, request_factory=_request, resume=True).reconcile()
    repaired = json.loads((stale_parent / "latest.json").read_text(encoding="utf-8"))
    assert repaired["sequence"] == 2
    assert repaired["sweeps_source"] == "execution_step_count_lower_bound"


def test_orphan_generation_repairs_latest_and_running_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """generation 發布後若在 latest/progress 前中斷，只允許 RUNNING reconcile 採認。"""

    workspace = _workspace(tmp_path, "orphan-generation", interval=2)
    shard_id = _first_shard(workspace)
    RunController(workspace, request_factory=_request).run_shard(shard_id, sweep_budget=1)
    controller = RunController(workspace, request_factory=_request, resume=True)

    def stop_before_latest(*args, **kwargs):
        """模擬 generation 已完成、latest 尚未更新的不可攔截程序終止。"""

        del args, kwargs
        raise SystemExit("simulated crash")

    monkeypatch.setattr(controller, "_write_latest", stop_before_latest)
    with pytest.raises(SystemExit, match="simulated crash"):
        controller.run_shard(shard_id, sweep_budget=2)
    row = load_run_progress(workspace)["shards"][shard_id]
    assert row["lifecycle"] == "RUNNING"
    assert row["checkpoint_sequence"] == 1
    recovered = RunController(workspace, request_factory=_request, resume=True).reconcile()
    row = recovered["shards"][shard_id]
    assert row["checkpoint_sequence"] == 2
    assert row["metrics"]["sweeps_recovered_lower_bound"] is True


def test_latest_pointer_oserror_keeps_complete_generation_recoverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """latest 更新失敗時不得把已發布 generation 標成 FAILED，且容量 metrics 由實體樹重建。"""

    workspace = _workspace(tmp_path / "fault", "latest-oserror", interval=2)
    shard_id = _first_shard(workspace)
    RunController(workspace, request_factory=_request).run_shard(shard_id, sweep_budget=1)
    controller = RunController(workspace, request_factory=_request, resume=True)

    def fail_latest(*args, **kwargs):
        """模擬 NFS latest pointer 原子替換失敗。"""

        del args, kwargs
        raise OSError("simulated latest rename failure")

    monkeypatch.setattr(controller, "_write_latest", fail_latest)
    with pytest.raises(OSError, match="latest rename"):
        controller.run_shard(shard_id, sweep_budget=2)

    row = load_run_progress(workspace)["shards"][shard_id]
    parent = workspace / "checkpoints" / workspace.name / shard_id
    assert row["lifecycle"] == "RUNNING"
    assert row["checkpoint_sequence"] == 2
    assert not (workspace / "failures" / shard_id).exists()
    logical_bytes = sum(
        path.stat().st_size
        for generation in parent.glob("checkpoint-*")
        for path in generation.iterdir()
        if path.is_file()
    )
    active_bytes = sum(path.stat().st_size for path in parent.rglob("*") if path.is_file())
    assert row["metrics"]["checkpoint_bytes"] == logical_bytes
    assert row["metrics"]["checkpoint_active_bytes"] == active_bytes

    # latest pointer 仍落在第一代，下一次 reconcile 應由 RUNNING row 採認第二代 orphan，
    # 並在不改變 checkpoint state／RNG 的前提下允許完整 resume。
    reconciled = RunController(workspace, request_factory=_request, resume=True).reconcile()
    assert reconciled["shards"][shard_id]["checkpoint_sequence"] == 2
    summary = RunController(workspace, request_factory=_request, resume=True).run_shard(shard_id)
    assert summary.lifecycle == "COMPLETE"

    baseline = _workspace(tmp_path / "baseline", "latest-baseline", interval=2)
    baseline_shard_id = _first_shard(baseline)
    RunController(baseline, request_factory=_request).run_shard(baseline_shard_id)
    assert _payload_snapshot(workspace, shard_id) == _payload_snapshot(baseline, baseline_shard_id)


def test_latest_and_adoption_read_oserror_keep_running_for_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """latest 發布後的暫時讀取錯誤不得寫 FAILED，NFS 恢復後仍可續跑 orphan。"""

    workspace = _workspace(tmp_path, "latest-adoption-oserror", interval=2)
    shard_id = _first_shard(workspace)
    RunController(workspace, request_factory=_request).run_shard(shard_id, sweep_budget=1)
    controller = RunController(workspace, request_factory=_request, resume=True)

    def fail_latest(*args, **kwargs):
        """模擬 generation rename 後 latest pointer 原子替換失敗。"""

        del args, kwargs
        raise OSError("simulated latest rename failure")

    original_load = run_control_module.load_execution_checkpoint

    def fail_candidate_load(path, *args, **kwargs):
        """模擬 orphan 已存在但 NFS 暫時無法讀取其 payload。"""

        if Path(path).name == "checkpoint-00000002":
            raise OSError("simulated transient checkpoint read failure")
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(controller, "_write_latest", fail_latest)
    monkeypatch.setattr(run_control_module, "load_execution_checkpoint", fail_candidate_load)
    with pytest.raises(OSError, match="latest rename"):
        controller.run_shard(shard_id, sweep_budget=2)

    row = load_run_progress(workspace)["shards"][shard_id]
    assert row["lifecycle"] == "RUNNING"
    # progress 仍停在上一個已知 generation；candidate 必須留在現場，不能覆寫或變成 FAILED。
    assert row["checkpoint_sequence"] == 1
    assert not (workspace / "failures" / shard_id).exists()

    # NFS 恢復後恢復原 loader；新的 controller 會由最高完整 generation 修復 latest，
    # 載入同一 RNG／history，完成剩餘執行。
    monkeypatch.setattr(run_control_module, "load_execution_checkpoint", original_load)
    summary = RunController(workspace, request_factory=_request, resume=True).run_shard(shard_id)
    assert summary.lifecycle == "COMPLETE"


@pytest.mark.parametrize("damage", ["unknown", "partial", "symlink", "corrupt", "binding"])
def test_checkpoint_damage_is_fail_closed(tmp_path: Path, damage: str) -> None:
    """未知、partial、symlink、checksum 或 binding 損壞都不可退回較舊狀態。"""

    workspace = _workspace(tmp_path, f"damage-{damage}")
    shard_id = _first_shard(workspace)
    RunController(workspace, request_factory=_request).run_shard(shard_id, sweep_budget=1)
    parent = workspace / "checkpoints" / workspace.name / shard_id
    generation = parent / "checkpoint-00000001"
    if damage == "unknown":
        (parent / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    elif damage == "partial":
        (parent / ".checkpoint-00000002.partial-test").mkdir()
    elif damage == "symlink":
        (parent / "linked").symlink_to(generation, target_is_directory=True)
    elif damage == "corrupt":
        payload = generation / "history_segment.json"
        payload.write_text(payload.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    else:
        metadata_path = generation / "checkpoint.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["binding"]["config_hash"] = "0" * 64
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        latest_path = parent / "latest.json"
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        latest["checkpoint_json_sha256"] = sha256_file(metadata_path)
        latest_path.write_text(
            json.dumps(latest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    with pytest.raises((ValueError, RuntimeError)):
        RunController(workspace, request_factory=_request, resume=True).run_shard(shard_id)


def test_keyboard_interrupt_pauses_with_checkpoint_bytes_and_no_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """KeyboardInterrupt 要 best-effort checkpoint、PAUSED，且不得寫 failure artifact。"""

    workspace = _workspace(tmp_path, "keyboard-run", interval=3)
    shard_id = _first_shard(workspace)

    def interrupt(self: ProductionBatch, sweeps: int = 1):
        """模擬 operator 在 batch 已建構後中斷。"""

        del self, sweeps
        raise KeyboardInterrupt

    monkeypatch.setattr(ProductionBatch, "advance", interrupt)
    with pytest.raises(KeyboardInterrupt):
        RunController(workspace, request_factory=_request).run_shard(shard_id)
    row = load_run_progress(workspace)["shards"][shard_id]
    assert row["lifecycle"] == "PAUSED"
    assert row["checkpoint_sequence"] == 1
    assert row["metrics"]["checkpoint_bytes"] > 0
    assert row["metrics"]["checkpoint_active_bytes"] >= row["metrics"]["checkpoint_bytes"]
    assert not (workspace / "failures" / shard_id).exists()
    assert not any(path.name.startswith(".") for path in (workspace / "shards").iterdir())


def test_keyboard_interrupt_during_request_factory_stays_running(tmp_path: Path) -> None:
    """建構 batch 前中斷時保留 RUNNING，避免宣告不存在的 checkpoint。"""

    workspace = _workspace(tmp_path, "keyboard-before-batch")
    shard_id = _first_shard(workspace)

    def interrupting_factory(unit):
        """模擬 request factory 尚未產生可序列化 batch 即被中斷。"""

        del unit
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        RunController(workspace, request_factory=interrupting_factory).run_shard(shard_id)

    row = load_run_progress(workspace)["shards"][shard_id]
    assert row["lifecycle"] == "RUNNING"
    assert row["checkpoint_sequence"] == 0
    assert row["checkpoint_relative_path"] is None
    assert not (workspace / "failures" / shard_id).exists()
    validation = validate_run(workspace)
    assert validation["valid"] is True
    assert validation["summary"]["run_lifecycle"] == "RUNNING"


def test_atomic_json_cleanup_failure_preserves_original_keyboard_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """暫存檔清理遇到 NFS OSError 時仍保留原始 Ctrl-C 與現場證據。"""

    target = tmp_path / "state.json"
    original_unlink = Path.unlink

    def write_then_interrupt(path: Path, value: object) -> None:
        """留下部分內容後模擬 operator 中斷，建立可觀測 cleanup fault window。"""

        del value
        path.write_text("partial\n", encoding="utf-8")
        raise KeyboardInterrupt

    def fail_temporary_unlink(path: Path, *, missing_ok: bool = False) -> None:
        """只讓本次 atomic temporary 的清理模擬 NFS 權限／I/O 錯誤。"""

        if path.name.startswith(".state.json.partial-"):
            raise OSError("simulated temporary cleanup failure")
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(run_control_module, "_write_json", write_then_interrupt)
    monkeypatch.setattr(Path, "unlink", fail_temporary_unlink)
    with pytest.raises(KeyboardInterrupt):
        run_control_module._atomic_json(target, {})
    partials = list(tmp_path.glob(".state.json.partial-*"))
    assert len(partials) == 1
    assert not target.exists()


def test_keyboard_interrupt_during_checkpoint_parent_reuses_sequence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """建立 checkpoint parent 時中斷，recovery 不得跳過已分配的 generation 序號。"""

    workspace = _workspace(tmp_path, "keyboard-parent", interval=2)
    shard_id = _first_shard(workspace)
    controller = RunController(workspace, request_factory=_request)
    original_parent = controller._checkpoint_parent
    create_calls = 0

    def interrupt_parent(shard, *, create: bool = False):
        """只在本輪第一次真正建立 checkpoint parent 時模擬 Ctrl-C。"""

        nonlocal create_calls
        if create:
            create_calls += 1
            if create_calls == 1:
                raise KeyboardInterrupt
        return original_parent(shard, create=create)

    monkeypatch.setattr(controller, "_checkpoint_parent", interrupt_parent)
    with pytest.raises(KeyboardInterrupt):
        controller.run_shard(shard_id, sweep_budget=1)

    parent = workspace / "checkpoints" / workspace.name / shard_id
    row = load_run_progress(workspace)["shards"][shard_id]
    assert row["lifecycle"] == "PAUSED"
    assert row["checkpoint_sequence"] == 1
    assert [path.name for path in parent.glob("checkpoint-*")] == ["checkpoint-00000001"]
    assert not any(path.name.startswith(".") for path in parent.iterdir())
    assert not (workspace / "failures" / shard_id).exists()

    assert RunController(workspace, request_factory=_request, resume=True).run_shard(shard_id).lifecycle == (
        "COMPLETE"
    )


@pytest.mark.parametrize("fault", ["payload", "rename", "latest", "progress"])
def test_keyboard_interrupt_publish_windows_leave_one_resumable_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    """Ctrl-C 落在各 checkpoint 發布窗口時，不留 partial、不重複序號且可精確續跑。"""

    workspace = _workspace(tmp_path, f"keyboard-{fault}", interval=2)
    shard_id = _first_shard(workspace)
    controller = RunController(workspace, request_factory=_request)

    if fault == "payload":
        original_write_checkpoint = ProductionBatch.write_checkpoint
        calls = 0

        def interrupt_once(self: ProductionBatch, *args, **kwargs):
            """在 generation payload 尚未建立前中斷一次，第二次允許 handler 重試。"""

            nonlocal calls
            calls += 1
            if calls == 1:
                raise KeyboardInterrupt
            return original_write_checkpoint(self, *args, **kwargs)

        monkeypatch.setattr(ProductionBatch, "write_checkpoint", interrupt_once)
    elif fault == "rename":
        original_write_checkpoint = controller._write_checkpoint

        def interrupt_after_generation(*args, **kwargs):
            """generation/latest 已發布後，在 controller 收到 return 前中斷。"""

            original_write_checkpoint(*args, **kwargs)
            raise KeyboardInterrupt

        monkeypatch.setattr(controller, "_write_checkpoint", interrupt_after_generation)
    elif fault == "latest":

        def interrupt_latest(*args, **kwargs):
            """模擬 latest temporary／rename 窗口的 operator interrupt。"""

            del args, kwargs
            raise KeyboardInterrupt

        monkeypatch.setattr(controller, "_write_latest", interrupt_latest)
    else:
        original_mark_running = controller._mark_checkpoint_running
        calls = 0

        def interrupt_after_progress(*args, **kwargs):
            """progress 已原子發布後只中斷第一次，讓 handler 可再次採認。"""

            nonlocal calls
            result = original_mark_running(*args, **kwargs)
            calls += 1
            if calls == 1:
                raise KeyboardInterrupt
            return result

        monkeypatch.setattr(controller, "_mark_checkpoint_running", interrupt_after_progress)

    with pytest.raises(KeyboardInterrupt):
        controller.run_shard(shard_id, sweep_budget=1)

    parent = workspace / "checkpoints" / workspace.name / shard_id
    row = load_run_progress(workspace)["shards"][shard_id]
    assert row["lifecycle"] == "PAUSED"
    assert row["checkpoint_sequence"] == 1
    assert [path.name for path in parent.glob("checkpoint-*")] == ["checkpoint-00000001"]
    assert not any(path.name.startswith(".") for path in parent.iterdir())
    assert not (workspace / "failures" / shard_id).exists()

    summary = RunController(workspace, request_factory=_request, resume=True).run_shard(shard_id)
    assert summary.lifecycle == "COMPLETE"


def test_ordinary_error_writes_safe_failure_and_no_partial_output(tmp_path: Path) -> None:
    """一般 Exception 保存去路徑／secret artifact、FAILED progress，且不留 partial output。"""

    workspace = _workspace(tmp_path, "ordinary-failure")
    shard_id = _first_shard(workspace)

    def failing_factory(unit):
        """在 request 建立階段拋出含絕對路徑與 secret 的測試錯誤。"""

        del unit
        raise RuntimeError("forcing /private/server/input.nc password=hunter2")

    with pytest.raises(RuntimeError, match="forcing"):
        RunController(workspace, request_factory=failing_factory).run_shard(shard_id)
    row = load_run_progress(workspace)["shards"][shard_id]
    artifact = workspace / str(row["failure_relative_path"])
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert row["lifecycle"] == "FAILED"
    assert "/private/server" not in payload["message"]
    assert "hunter2" not in payload["message"]
    assert not (workspace / "shards" / shard_id).exists()
    assert not any(path.name.startswith(".") for path in (workspace / "shards").iterdir())


def test_publish_before_progress_reconcile_and_complete_output_fail_closed(tmp_path: Path) -> None:
    """合法 published-before-progress 可採認；COMPLETE output 遺失或損壞必須拒絕。"""

    workspace = _workspace(tmp_path / "reconcile", "reconcile-run")
    controller = RunController(workspace, request_factory=_request)
    summary = controller.run_shard(_first_shard(workspace))
    progress = load_run_progress(workspace)
    row = progress["shards"][summary.shard_id]
    latest = json.loads(
        (workspace / "checkpoints" / workspace.name / summary.shard_id / "latest.json").read_text(
            encoding="utf-8"
        )
    )
    row["lifecycle"] = "RUNNING"
    row["output_relative_path"] = None
    row["sweeps_completed"] = latest["sweeps_completed"]
    row["particle_steps"] = latest["particle_steps"]
    row["metrics"]["particle_steps"] = latest["particle_steps"]
    progress["run_lifecycle"] = "RUNNING"
    _write_progress(workspace, progress)
    before = validate_run(workspace)
    assert not before["valid"]
    assert any("recoverable_published_before_progress" in error for error in before["errors"])
    reconciled = RunController(workspace, request_factory=_request, resume=True).reconcile()
    assert reconciled["shards"][summary.shard_id]["lifecycle"] == "COMPLETE"

    missing = _workspace(tmp_path / "missing-output", "missing-output")
    missing_summary = RunController(missing, request_factory=_request).run_shard(_first_shard(missing))
    shutil.rmtree(missing / str(missing_summary.output_relative_path))
    with pytest.raises(ValueError, match="output"):
        RunController(missing, request_factory=_request).reconcile()

    corrupt = _workspace(tmp_path / "corrupt-output", "corrupt-output")
    corrupt_summary = RunController(corrupt, request_factory=_request).run_shard(_first_shard(corrupt))
    manifest = corrupt / str(corrupt_summary.output_relative_path) / "manifest.json"
    manifest.write_text("{bad-json", encoding="utf-8")
    with pytest.raises(ValueError, match="output"):
        RunController(corrupt, request_factory=_request).reconcile()
