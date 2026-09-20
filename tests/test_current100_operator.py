"""固定100顆操作入口的選片、命令及記憶體停止測試；不連線或啟動模型。"""

import importlib.util
import json
import signal
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "current100", Path(__file__).parents[1] / "scripts/run_current100_subset.py")
operator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(operator)


def fixture_plan():
    """建出完整表的最小記憶體替身；固定組以外情境不能被派給worker。"""
    rows = []
    for site, (utc, arrival) in operator.SELECTION.items():
        rows.extend(dict(scenario_id=f"{site}-{i}", receptor_id=f"r-{i}",
                         study_site_id=site, arrival_time_id=arrival, arrival_time_utc_ns=utc,
                         material_id=operator.MATERIAL, settling_velocity_mps=-0.02) for i in range(20))
    rows.extend(dict(scenario_id=f"other-{i}", study_site_id="other", material_id="other")
                for i in range(49900))
    plan = dict(run_kind="formal", scenario_selection={"mode": "full"},
                experiment_case_id="no_stokes", members_per_scenario=1, scenario_count=50000,
                shards=[dict(shard_id=f"sh-{i}", scenario_start_index=i * 10,
                             scenario_stop_index=(i + 1) * 10, particle_count=10) for i in range(5000)])
    return plan, rows


def test_selects_only_fixed_100_and_keeps_nanwan_identity():
    """選樣固定五站各兩片；到達時間不同也保留各站事前身分。"""
    selected = operator.select_shards(*fixture_plan())
    assert list(selected) == list(operator.SELECTION)
    assert [sid for ids in selected.values() for sid in ids] == [f"sh-{i}" for i in range(10)]
    assert "nanwan" in selected and "houwan" not in selected


@pytest.mark.parametrize("damage", ["case", "mixed", "arrival", "duplicate_receptor", "settling"])
def test_rejects_changed_population_or_mixed_shards(damage):
    """波浪案例、混片、錯誤時間／受體或沉降不得進入固定純流展示。"""
    plan, rows = fixture_plan()
    if damage == "case":
        plan["experiment_case_id"] = "finite_depth_stokes"
    elif damage == "mixed":
        plan["shards"][0]["scenario_stop_index"] = 21
    elif damage == "arrival":
        rows[0]["arrival_time_id"] = "other"
    elif damage == "settling":
        rows[0]["settling_velocity_mps"] = 0
    else:
        rows[0]["receptor_id"] = rows[1]["receptor_id"]
    with pytest.raises(ValueError):
        operator.select_shards(plan, rows)


def test_existing_worker_cli_receives_only_two_shards(tmp_path):
    """命令只有指定兩片與既有CLI，不呼叫完整佇列或讀入波浪根。"""
    args = SimpleNamespace(**{key: tmp_path for key in (
        "project_root", "workspace", "config", "ocm_native_root",
        "ocm_reconstruction_root", "checkpoint_root")})
    command = operator.worker_command(args, ["first", "second"])
    assert command.count("--shard-id") == 2
    assert command[-4:] == ["--shard-id", "first", "--shard-id", "second"]
    assert "run-worker" in command and "--execution-priority" in command
    assert "run-formal-parallel" not in command and "--nww-analysis-root" not in command
    assert "--resume" not in command


@pytest.mark.parametrize("resume", [False, True])
def test_resume_only_forwards_explicit_flag_without_changing_shards(tmp_path, resume):
    """續跑只多一個既有CLI旗標，仍恰為原兩片，不新增驗證略過介面。"""
    args = SimpleNamespace(**{key: tmp_path for key in (
        "project_root", "workspace", "config", "ocm_native_root",
        "ocm_reconstruction_root", "checkpoint_root")}, resume=resume)
    command = operator.worker_command(args, ["first", "second"])
    assert command.count("--resume") == int(resume)
    assert command.count("--shard-id") == 2
    assert command[-4:] == ["--shard-id", "first", "--shard-id", "second"]
    assert "--defer-full-validation" not in command


@pytest.mark.parametrize("lifecycle", ["PLANNED", "RUNNING", "PAUSED", "FAILED", "COMPLETE"])
def test_resume_preserves_selected_progress_without_mutation(lifecycle):
    """既定十片可保留目前生命週期與耐久步數；核對函式不改計畫或進度。"""
    plan, rows = fixture_plan()
    selected = operator.select_shards(plan, rows)
    progress = {"shards": {s["shard_id"]: dict(lifecycle="PLANNED", particle_steps=0)
                           for s in plan["shards"]}}
    progress["shards"]["sh-0"] = dict(lifecycle=lifecycle,
                                     particle_steps=0 if lifecycle == "PLANNED" else 10000,
                                     checkpoint_sequence=1)
    before = deepcopy((plan, progress, selected))
    operator.validate_subset_progress(plan, progress, selected, resume=True)
    assert (plan, progress, selected) == before
    if lifecycle != "PLANNED":
        with pytest.raises(ValueError, match="未明示續跑"):
            operator.validate_subset_progress(plan, progress, selected)


@pytest.mark.parametrize("damage", ["outside_state", "outside_steps", "missing", "unknown", "non_bool"])
def test_resume_rejects_expanded_population_or_invalid_state(damage):
    """其他4,990片不得已有狀態或步數；缺片與非布林續跑值也必須拒絕。"""
    plan, rows = fixture_plan()
    selected = operator.select_shards(plan, rows)
    progress = {"shards": {s["shard_id"]: dict(lifecycle="PLANNED", particle_steps=0)
                           for s in plan["shards"]}}
    if damage == "outside_state":
        progress["shards"]["sh-10"]["lifecycle"] = "RUNNING"
    elif damage == "outside_steps":
        progress["shards"]["sh-10"]["particle_steps"] = 1
    elif damage == "missing":
        del progress["shards"]["sh-10"]
    elif damage == "unknown":
        progress["shards"]["sh-0"]["lifecycle"] = "UNKNOWN"
    with pytest.raises((ValueError, TypeError)):
        operator.validate_subset_progress(plan, progress, selected,
                                          resume=1 if damage == "non_bool" else True)


def test_low_memory_stops_only_started_workers_with_existing_safe_helper(tmp_path, monkeypatch):
    """啟動兩站後記憶體降到門檻以下，後三站不啟動，既有兩PID收到安全收尾。"""
    import lagrangian_backtracking.parallel_execution as parallel

    memories = iter([operator.MIN_MEMORY + 1, operator.MIN_MEMORY + 1, operator.MIN_MEMORY - 1])
    monkeypatch.setattr(operator, "memory_available", lambda: next(memories))
    monkeypatch.setattr(operator, "worker_command", lambda *args: ["unused"])
    started, stopped = [], []

    def spawn(*args, **kwargs):
        """程序替身只記錄PID與獨立程序群組，不建立系統程序。"""
        assert kwargs["start_new_session"] is True
        process = SimpleNamespace(pid=100 + len(started))
        started.append(process)
        return process

    def stop(active, *, first_signal, grace_seconds):
        """核對只交付本次已啟動的程序且SIGINT等待300秒。"""
        assert list(active.values()) == started
        assert first_signal == signal.SIGINT and grace_seconds == 300
        stopped.extend(active)
        return {site: 130 for site in active}

    monkeypatch.setattr(operator.subprocess, "Popen", spawn)
    monkeypatch.setattr(parallel, "_stop_active_processes", stop)
    selected = {site: ["a", "b"] for site in operator.SELECTION}
    assert operator.supervise(SimpleNamespace(project_root=tmp_path), selected, tmp_path, {}) == 2
    assert len(started) == len(stopped) == 2


def test_five_workers_receive_bounded_cache_and_manifest_policy(tmp_path, monkeypatch):
    """五席皆收到固定容量，manifest保存同一環境值；低記憶體仍經原helper收尾。"""
    import lagrangian_backtracking.parallel_execution as parallel

    inherited = {key: 'incorrect' for key in (*operator.THREAD_ENV, *operator.FRAME_CACHE_ENV)}
    env = operator.worker_environment(inherited)
    assert all(env[key] == '1' for key in operator.THREAD_ENV)
    assert all(inherited[key] == 'incorrect' for key in inherited)
    policy = operator.frame_cache_policy(env)
    assert policy['max_cached_frames'] == 20
    assert policy['max_cached_bytes'] == 4294967296
    assert policy['environment'] == operator.FRAME_CACHE_ENV
    assert '非物理設定' in policy['semantics'] and '非程序RSS' in policy['semantics']
    memories = iter([operator.MIN_MEMORY + 1] * 5 + [operator.MIN_MEMORY - 1])
    monkeypatch.setattr(operator, 'memory_available', lambda: next(memories))
    monkeypatch.setattr(operator, 'worker_command', lambda *args: ['unused'])
    passed = []

    def spawn(*args, **kwargs):
        """僅記錄每席的執行環境，不建立真實程序。"""
        passed.append(kwargs['env'])
        return SimpleNamespace(pid=700 + len(passed))

    monkeypatch.setattr(operator.subprocess, 'Popen', spawn)
    monkeypatch.setattr(parallel, '_stop_active_processes', lambda active, **kwargs: {})
    selected = {site: ['first', 'second'] for site in operator.SELECTION}
    assert operator.supervise(SimpleNamespace(project_root=tmp_path), selected, tmp_path, env) == 2
    assert len(passed) == 5 and all(operator.frame_cache_policy(e) == policy for e in passed)


@pytest.mark.parametrize("fallback,completed,expected", [(0, 20, True), (1, 20, False), (0, 0, False)])
def test_completion_requires_actual_native_steps_but_preserves_physical_qc(
    tmp_path, fallback, completed, expected
):
    """真實乾點QC保留；全早停或曾補算不能宣稱原版推進完成。"""
    report = dict(require_native=True, fallback_particle_steps=fallback,
                  native_completed_particle_steps=completed, counter_scope="shard_invocation_delta_v1",
                  source_sha256={"batch_interp": "a" * 64}, group_size_histogram={"10": 1},
                  native_rejected_reasons={"dry_face": 1})
    directory = tmp_path / "shards" / "one"
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(json.dumps({"run_metadata": {"advection_execution": report}}))
    if expected:
        evidence = operator.completed_evidence(tmp_path, ["one", "one"])
        assert evidence["native_completed_particle_steps"] == 40
        assert evidence["shard_reports"][0]["native_rejected_reasons"] == {"dry_face": 1}
    else:
        with pytest.raises(ValueError):
            operator.completed_evidence(tmp_path, ["one", "one"])
