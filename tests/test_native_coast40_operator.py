"""兩站岸界操作入口的有界測試；程序皆為替身，不連SERVER或啟動粒子。"""

import importlib.util
import json
import signal
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_current100_operator import fixture_plan

SCRIPT_ROOT = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_ROOT))
SPEC = importlib.util.spec_from_file_location("coast40", SCRIPT_ROOT / "run_native_coast40_subset.py")
operator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(operator)
shared = operator.shared


def selected(sites=operator.SITES):
    """每站回傳兩個唯一ID，避免程序測試混淆跨站分片。"""
    return {site: [site + "-1", site + "-2"] for site in sites}


@pytest.mark.parametrize("sites", [operator.SITES, tuple(shared.SELECTION)])
def test_completed_two_or_five_sites_exit_without_extra_wait(tmp_path, monkeypatch, sites):
    """兩站與原五站均依各自分母結束，原版／QC證據沿用原helper。"""
    import lagrangian_backtracking.parallel_execution as parallel
    import lagrangian_backtracking.run_control as control

    selection = selected(sites)
    env = shared.worker_environment({})
    spawned, stopped = [], []
    monkeypatch.setattr(shared, "memory_available", lambda: shared.MIN_MEMORY)
    monkeypatch.setattr(shared, "worker_command", lambda args, ids: ids)
    monkeypatch.setattr(shared, "process_snapshot", lambda p: {"pid": p.pid})
    monkeypatch.setattr(shared.time, "sleep", lambda _: pytest.fail("完成後不應等其他站"))
    monkeypatch.setattr(
        control,
        "load_run_progress",
        lambda _: {
            "revision": 1,
            "shards": {
                sid: {"lifecycle": "COMPLETE", "particle_steps": 100, "checkpoint_sequence": 1}
                for ids in selection.values()
                for sid in ids
            },
        },
    )
    monkeypatch.setattr(
        shared,
        "completed_evidence",
        lambda workspace, ids: {
            "native_completed_particle_steps": 200,
            "shard_reports": [{"native_rejected_reasons": {"dry_face": 1}}],
        },
    )

    def spawn(command, **kwargs):
        assert kwargs["start_new_session"] and kwargs["env"] == env
        p = SimpleNamespace(pid=91000 + len(spawned), returncode=0, poll=lambda: 0)
        spawned.append(p)
        return p

    def stop(active, **kwargs):
        assert list(active.values()) == spawned
        assert kwargs == {"first_signal": signal.SIGINT, "grace_seconds": 300}
        stopped.extend(active)
        return {site: 0 for site in active}

    monkeypatch.setattr(shared.subprocess, "Popen", spawn)
    monkeypatch.setattr(parallel, "_stop_active_processes", stop)
    args = SimpleNamespace(project_root=tmp_path, workspace=tmp_path)
    assert shared.supervise(args, selection, tmp_path, env) == 0
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert events[-1]["status"] == f"SELECTED_{len(sites) * 20}_COMPUTE_COMPLETE_FULL_RUN_INCOMPLETE"
    assert len(spawned) == len(stopped) == len(sites)
    assert all(env[k] == "1" for k in shared.THREAD_ENV)
    assert shared.frame_cache_policy(env)["max_cached_bytes"] == 4294967296
    assert shared.frame_cache_policy(env)["max_cached_frames"] == 20
    assert events[-1]["full_validation_performed"] is False


@pytest.mark.parametrize("failure", ["memory", "unobservable", "signal", "exit"])
def test_two_site_failures_cleanup_only_own_groups(tmp_path, monkeypatch, failure):
    """不足／缺失記憶體、訊號及非零退出只交本次Popen程序給原300秒helper。"""
    import lagrangian_backtracking.parallel_execution as parallel
    import lagrangian_backtracking.run_control as control

    handlers, spawned, stopped = {}, [], []
    monkeypatch.setattr(shared.signal, "signal", lambda sig, h: handlers.setdefault(sig, h))
    calls = iter(
        [
            shared.MIN_MEMORY,
            shared.MIN_MEMORY,
            None
            if failure == "unobservable"
            else shared.MIN_MEMORY - 1
            if failure == "memory"
            else shared.MIN_MEMORY,
        ]
    )
    monkeypatch.setattr(shared, "memory_available", lambda: next(calls))
    monkeypatch.setattr(shared, "worker_command", lambda *args: [])
    monkeypatch.setattr(shared, "process_snapshot", lambda p: {"pid": p.pid})
    monkeypatch.setattr(
        control,
        "load_run_progress",
        lambda _: {
            "revision": 1,
            "shards": {sid: {"lifecycle": "RUNNING"} for ids in selected().values() for sid in ids},
        },
    )

    def spawn(*args, **kwargs):
        p = SimpleNamespace(pid=92000 + len(spawned), returncode=2, poll=lambda: 2)
        spawned.append(p)
        if failure == "signal" and len(spawned) == 2:
            handlers[signal.SIGINT](signal.SIGINT, None)
        return p

    def stop(active, **kwargs):
        assert list(active.values()) == spawned and len(spawned) == 2
        assert kwargs["grace_seconds"] == 300
        stopped.extend(active)
        return {}

    monkeypatch.setattr(shared.subprocess, "Popen", spawn)
    monkeypatch.setattr(parallel, "_stop_active_processes", stop)
    assert (
        shared.supervise(SimpleNamespace(project_root=tmp_path, workspace=tmp_path), selected(), tmp_path, {})
        == 2
    )
    assert len(stopped) == 2
    assert not {65601, 65614, 65615} & {p.pid for p in spawned}


@pytest.mark.parametrize(
    "selection", [selected((*operator.SITES, "hsinchu")), {"nanwan": ["a", "b"], "lienchiang": ["a", "c"]}]
)
def test_unknown_selection_rejected_before_lifecycle(tmp_path, monkeypatch, selection):
    """第三站及重複分片不能進入共用生命週期。"""
    monkeypatch.setattr(shared.subprocess, "Popen", lambda *a, **k: pytest.fail("不可啟動"))
    with pytest.raises(ValueError):
        shared.supervise(None, selection, tmp_path, {})
    assert not list(tmp_path.iterdir())


def test_two_sites_selected_after_full_population_and_nonselected_zero():
    """先通過原五站母體再取40顆，四片外任何一片有進展即拒絕。"""
    plan, rows = fixture_plan()
    progress = {
        "shards": {s["shard_id"]: {"lifecycle": "PLANNED", "particle_steps": 0} for s in plan["shards"]}
    }
    result = operator.select_and_validate_fresh(plan, rows, progress)
    assert set(result) == set(operator.SITES) and sum(map(len, result.values())) == 4
    progress["shards"]["sh-0"]["particle_steps"] = 1
    with pytest.raises(ValueError, match="零步"):
        operator.select_and_validate_fresh(plan, rows, progress)


def binding_fixture(tmp_path, monkeypatch):
    """用小型表與主審常數替身驗證相同的雜湊檢查，不載入真實資料。"""
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    for root in (old, new):
        for name in ("run_plan.json", "scenario_table.parquet", "seed_table.parquet"):
            (root / name).write_text(name)
    coast = tmp_path / "coast.json"
    coast.write_text("coast")
    limits = dict(
        operator.REVIEWED,
        source_plan=operator.file_sha(old / "run_plan.json"),
        target_plan=operator.file_sha(new / "run_plan.json"),
        coast=operator.file_sha(coast),
    )
    monkeypatch.setattr(operator, "REVIEWED", limits)
    ref = {"path": str(coast), "sha256": limits["coast"]}
    source = {
        k: "same"
        for k in (
            "master_seed",
            "component_canonical_hashes",
            "members_per_scenario",
            "experiment_case_id",
            "seed_policy",
            "active_chunk_size",
            "checkpoint_interval_sweeps",
        )
    }
    source["geometry_canonical_hashes"] = {k: "same" for k in ("domain", "local", "open_boundary")}
    source["files"] = {
        name: {"sha256": operator.file_sha(old / name)}
        for name in ("scenario_table.parquet", "seed_table.parquet")
    }
    target = deepcopy(source)
    target.update(config_hash=limits["config"], checkpoint_input_binding_hash=limits["input_binding"])
    target["geometry_canonical_hashes"]["native_coast"] = "new"
    config = SimpleNamespace(
        config_hash=lambda: limits["config"],
        normalized_payload=lambda: {"geometry": {"native_coast_manifest": ref}},
        boundaries=SimpleNamespace(max_backtrack_days=30),
        execution=SimpleNamespace(advection_backend="baytrace_native_cpu_v1"),
    )
    args = SimpleNamespace(workspace=new, source_workspace=old, config=tmp_path / "config.yaml")
    return args, source, target, config, ref


@pytest.mark.parametrize(
    "damage",
    [None, "plan", "source_plan", "seed", "config", "input", "coast", "coast_file", "geometry", "table"],
)
def test_reviewed_pair_rejects_any_binding_change(tmp_path, monkeypatch, damage):
    """來源／目標計畫、seed、三舊幾何與新岸界皆須符合，不能略過差異。"""
    args, source, target, config, ref = binding_fixture(tmp_path, monkeypatch)
    if damage in ("plan", "source_plan"):
        root = args.workspace if damage == "plan" else args.source_workspace
        (root / "run_plan.json").write_text("changed")
    elif damage == "seed":
        target["master_seed"] = "changed"
    elif damage == "config":
        target["config_hash"] = "changed"
    elif damage == "input":
        target["checkpoint_input_binding_hash"] = "changed"
    elif damage == "coast":
        ref["sha256"] = "changed"
    elif damage == "coast_file":
        Path(ref["path"]).write_text("changed")
    elif damage == "geometry":
        target["geometry_canonical_hashes"]["local"] = "changed"
    elif damage == "table":
        (args.workspace / "seed_table.parquet").write_text("changed")
    if damage:
        with pytest.raises(ValueError):
            operator.validate_binding_pair(args, source, target, config)
    else:
        assert len(operator.validate_binding_pair(args, source, target, config)) == 2


def test_operator_lock_excludes_duplicate_and_preserves_workspace(tmp_path):
    """不同operator對同workspace不能雙啟；鎖不寫入run白名單且釋放後可取得。"""
    workspace = tmp_path / "run"
    workspace.mkdir()
    with operator.operator_lock(workspace) as path:
        assert path.parent == tmp_path and path.stat().st_size == 0
        with pytest.raises(ValueError, match="已有"), operator.operator_lock(workspace):
            pytest.fail("重複鎖不應成功")
    with operator.operator_lock(workspace):
        assert list(workspace.iterdir()) == []


def test_dry_run_uses_full_validator_and_never_spawns(tmp_path, monkeypatch):
    """只替換I/O邊界，main仍核完整母體／正式validator並保存40顆命令，不Popen。"""
    import pyarrow.parquet as pq

    import lagrangian_backtracking.parallel_execution as parallel
    import lagrangian_backtracking.run_control as control

    plan, rows = fixture_plan()
    plan.update(run_id="new", code_provenance={"git_commit": "model"})
    source = {"run_id": "old"}
    progress = {
        "revision": 0,
        "shards": {s["shard_id"]: {"lifecycle": "PLANNED", "particle_steps": 0} for s in plan["shards"]},
    }
    workspace, source_path, session = tmp_path / "target", tmp_path / "source", tmp_path / "session"
    for p in (workspace, source_path, session):
        p.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text("test")
    env = shared.worker_environment(
        {
            k: str(tmp_path)
            for k in ("TMPDIR", "MPLCONFIGDIR", "XDG_CACHE_HOME", "UV_CACHE_DIR", "NUMBA_CACHE_DIR")
        }
    )
    monkeypatch.setattr(operator, "prepare_environment", lambda args: env)
    monkeypatch.setattr(
        operator.subprocess, "check_output", lambda cmd, **kw: "actual" if "rev-parse" in cmd else ""
    )
    monkeypatch.setattr(shared.subprocess, "Popen", lambda *a, **kw: pytest.fail("dry-run不可Popen"))
    monkeypatch.setattr(
        parallel, "validate_storage_roots", lambda **kw: (tmp_path, tmp_path, tmp_path, "gate")
    )
    monkeypatch.setattr(parallel, "_live_mount_identity", lambda p: ("nfs4", "same"))
    monkeypatch.setattr(shared, "memory_available", lambda: shared.MIN_MEMORY)
    monkeypatch.setattr(control, "load_run_plan", lambda p: source if p == source_path else plan)
    monkeypatch.setattr(control, "load_run_progress", lambda p: progress)
    monkeypatch.setattr(pq, "read_table", lambda p: SimpleNamespace(to_pylist=lambda: rows))
    calls = []

    def validate(**kw):
        calls.append(kw)
        assert kw["execution_priority"] is True and "defer_full_validation" not in kw
        return plan, None, None, None, None, {"git_commit": "actual"}

    monkeypatch.setattr(parallel, "validate_run_inputs", validate)
    monkeypatch.setattr(operator, "validate_binding_pair", lambda *a: {"scenario": "same", "seed": "same"})
    monkeypatch.setattr(parallel, "prepare_execution_directories", lambda **kw: (session, tmp_path))
    argv = []
    paths = dict(
        workspace=workspace,
        source_workspace=source_path,
        config=config,
        project_root=SCRIPT_ROOT.parent,
        checkpoint_root=tmp_path,
        ocm_native_root=tmp_path,
        ocm_reconstruction_root=tmp_path,
        scratch_root=tmp_path,
        log_root=tmp_path,
        numba_cache_dir=tmp_path,
        storage_gate_evidence=tmp_path,
    )
    for key, value in paths.items():
        argv.extend(["--" + key.replace("_", "-"), str(value)])
    assert operator.main(argv + ["--expected-commit", "actual"]) == 0
    manifest = json.loads((session / "demo-manifest.json").read_text())
    assert len(calls) == 1 and manifest["execute"] is False
    assert manifest["workers"] == 2 and manifest["selected_particles"] == 40
    assert manifest["actual_commit"] != manifest["plan_code_provenance"]["git_commit"]
    for cmd in manifest["worker_commands"].values():
        assert cmd.count("--shard-id") == 2 and "--resume" not in cmd and "--sweep-budget" not in cmd


@pytest.mark.parametrize("flag", ["--resume", "--workers", "--site"])
def test_no_public_resume_or_arbitrary_selection(flag):
    """薄入口不新增任意站數／續跑選項。"""
    with pytest.raises(SystemExit):
        operator.main([flag])


@pytest.mark.parametrize("available", [None, shared.MIN_MEMORY - 1])
def test_start_guard_never_spawns_when_unavailable(tmp_path, monkeypatch, available):
    """第一站啟動前就必須能觀測至少30GiB，不可先啟動再回頭檢查。"""
    import lagrangian_backtracking.parallel_execution as parallel

    monkeypatch.setattr(shared, "memory_available", lambda: available)
    monkeypatch.setattr(shared.subprocess, "Popen", lambda *a, **k: pytest.fail("guard前不可Popen"))

    def stop(active, **kwargs):
        assert active == {} and kwargs["grace_seconds"] == 300
        return {}

    monkeypatch.setattr(parallel, "_stop_active_processes", stop)
    assert shared.supervise(SimpleNamespace(project_root=tmp_path), selected(), tmp_path, {}) == 2


@pytest.mark.parametrize("damage", [None, "target_outside", "cache_outside", "tmp_outside"])
def test_all_write_roots_bounded_to_nfs_scratch(tmp_path, monkeypatch, damage):
    """以目錄替身核對路徑防線；實際NFS來源另外由既有live mount gate核對。"""
    root = tmp_path / "nfs"
    root.mkdir()
    scratch = root / "scratch"
    scratch.mkdir()
    args = SimpleNamespace(
        workspace=root / "run",
        checkpoint_root=root / "checkpoint",
        scratch_root=scratch,
        log_root=scratch / "logs",
        numba_cache_dir=scratch / "numba",
    )
    for key in ("workspace", "checkpoint_root", "log_root", "numba_cache_dir"):
        getattr(args, key).mkdir()
    monkeypatch.setattr(operator, "Path", lambda p: root if p == "/data/LBT" else Path(p))
    for key in ("TMPDIR", "MPLCONFIGDIR", "XDG_CACHE_HOME", "UV_CACHE_DIR"):
        monkeypatch.setenv(key, str(scratch))
    if damage == "target_outside":
        args.workspace = tmp_path
    elif damage == "cache_outside":
        args.numba_cache_dir = root
    elif damage == "tmp_outside":
        monkeypatch.setenv("TMPDIR", str(tmp_path))
    if damage:
        with pytest.raises(ValueError):
            operator.prepare_environment(args)
    else:
        env = operator.prepare_environment(args)
        assert all(env[k] == "1" for k in shared.THREAD_ENV)
        assert shared.frame_cache_policy(env)["max_cached_frames"] == 20
