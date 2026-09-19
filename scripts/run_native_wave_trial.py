"""有限水深波浪的單片、單次掃描真資料試跑，不改H30、步長或種子。

此操作入口只接受新的完整正式計畫；一次掃描後由既有控制器保存可恢復檢查點。
完整驗收與完整波浪派片另由正式入口處理。操作腳本可置於NFS package，數值程式
仍從明示的乾淨checkout匯入；兩者分別保存commit與檔案SHA，不污染計算checkout。
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path


def check_report(report):
    """要求真實原版批次成功與波浪節點場；物理QC原樣保留，不要求拒絕數全零。

    工程不支援會由嚴格控制器拋出例外，不能到達此成功檢查；本函式另外拒絕
    舊核心補算、零推進、未開波浪或缺來源，避免只靠正常退出碼宣告成功。
    """
    resolver = report.get("resolver_report", {})
    if (report.get("require_native") is not True
            or report.get("fallback_particle_steps") != 0
            or report.get("native_completed_particle_steps", 0) <= 0
            or not report.get("source_sha256")
            or resolver.get("include_stokes") is not True
            or resolver.get("field_policy") != "finite_depth_node_fields_v1"
            or not any(int(k) > 1 and v > 0
                       for k, v in report.get("group_size_histogram", {}).items())):
        raise ValueError("缺少原版波浪真推進／群組／來源證據，或曾退回舊核心")


def trial_worker(args):
    """只開啟一個控制器、執行指定片一個掃描；階段時間是單程序實際經過時間。"""
    from lagrangian_backtracking.cli import open_worker_controller

    start = time.monotonic()
    controller = open_worker_controller(
        args.workspace, config_path=args.config, ocm_native_root=args.ocm_native_root,
        nww_analysis_root=args.nww_analysis_root,
        ocm_reconstruction_root=args.ocm_reconstruction_root,
        checkpoint_root=args.checkpoint_root, continuous_time_fallback=True)
    setup_seconds = time.monotonic() - start
    print(json.dumps(dict(stage="static_setup_complete", elapsed_seconds=setup_seconds)), flush=True)
    advance_start = time.monotonic()
    summary = controller.run_shard(args.shard_id, sweep_budget=1)
    # 此區間包含既有步進、首次編譯、檢查點輸出；不捏造無法分離的純核心耗時。
    report = controller.advection_reporter()
    evidence = dict(utc=datetime.now(UTC).isoformat(), summary=asdict(summary), report=report,
                    setup_seconds=setup_seconds,
                    advance_and_checkpoint_seconds=time.monotonic() - advance_start,
                    worker_elapsed_seconds=time.monotonic() - start,
                    max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                    full_run_complete=False, full_validation_performed=False)
    with (args.session / "trial-evidence.json").open("x") as stream:
        json.dump(evidence, stream, ensure_ascii=False, indent=2)
    if summary.lifecycle not in {"PAUSED", "COMPLETE"} or summary.particle_steps <= 0:
        raise ValueError("控制器未保存真實推進，不能宣告小試成功")
    check_report(report)
    print(json.dumps(dict(trial_pass=True, native_completed=report["native_completed_particle_steps"])),
          flush=True)
    return 0


def main(argv=None):
    """先核對新計畫與儲存，再啟動唯一單執行緒子程序；30GiB下限安全停止。

    啟動核對包含來源情境、設定、種子與幾何繫結；不讀舊軌跡或舊檢查點。
    NFS操作紀錄不加入run工作區，避免破壞工作區白名單。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workspace", "source-workspace", "config", "project-root", "checkpoint-root",
                 "ocm-native-root", "nww-analysis-root", "ocm-reconstruction-root", "scratch-root",
                 "log-root", "numba-cache-dir", "storage-gate-evidence", "session"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--shard-id", required=True)
    parser.add_argument("--internal-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    sys.path[:0] = [str(args.project_root / "src"), str(args.project_root / "scripts")]
    from run_current100_subset import MIN_MEMORY, THREAD_ENV, memory_available, process_snapshot

    from lagrangian_backtracking.parallel_execution import (
        _stop_active_processes,
        validate_run_inputs,
        validate_storage_roots,
    )
    from lagrangian_backtracking.run_control import load_run_plan, load_run_progress

    if args.internal_worker:
        return trial_worker(args)
    scratch, _, _, gate_sha = validate_storage_roots(
        scratch_root=args.scratch_root, log_root=args.log_root,
        numba_cache_dir=args.numba_cache_dir, gate_evidence=args.storage_gate_evidence)
    if not args.session.resolve().is_relative_to(scratch):
        raise ValueError("試跑操作紀錄必須位於本次已驗證scratch")
    for key in ("TMPDIR", "MPLCONFIGDIR", "XDG_CACHE_HOME", "UV_CACHE_DIR", "NUMBA_CACHE_DIR"):
        if not Path(os.environ[key]).resolve(strict=True).is_relative_to(scratch):
            raise ValueError(f"{key}必須位於本次scratch")
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.project_root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=args.project_root, text=True)
    if dirty or actual != args.expected_commit or args.workspace.resolve() == args.source_workspace.resolve():
        raise ValueError("必須使用核准乾淨checkout與獨立新工作區")
    progress = load_run_progress(args.workspace)
    if any(r["lifecycle"] != "PLANNED" or r["particle_steps"] for r in progress["shards"].values()):
        raise ValueError("只接受新計畫；不可對已執行資料重跑此小試")
    plan, config, _, _, _, provenance = validate_run_inputs(
        workspace=args.workspace, config_path=args.config, project_root=args.project_root,
        checkpoint_root=args.checkpoint_root, execution_priority=True)
    source = load_run_plan(args.source_workspace)
    for key in ("config_hash", "master_seed", "component_canonical_hashes", "geometry_canonical_hashes"):
        if plan[key] != source[key]:
            raise ValueError(f"來源繫結不符：{key}")
    if (plan["files"]["scenario_table.parquet"] != source["files"]["scenario_table.parquet"]
            or plan["experiment_case_id"] != "finite_depth_stokes" or plan["run_kind"] != "formal"
            or plan["members_per_scenario"] != 1 or plan["scenario_selection"]["mode"] != "full"
            or config.boundaries.max_backtrack_days != 30
            or config.execution.advection_backend != "baytrace_native_cpu_v1"
            or args.shard_id not in {s["shard_id"] for s in plan["shards"]}):
        raise ValueError("必須保留H30/M1/full情境來源並使用嚴格原版波浪")
    args.session.mkdir(exist_ok=False)
    env = dict(os.environ, **{key: "1" for key in THREAD_ENV})
    command = [sys.executable, "-u", str(Path(__file__).resolve()),
               *(argv if argv is not None else sys.argv[1:]), "--internal-worker"]
    manifest = dict(utc=datetime.now(UTC).isoformat(), run_id=plan["run_id"], actual_commit=actual,
                    script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(), argv=command,
                    plan_sha256=sha256((args.workspace / "run_plan.json").read_bytes()).hexdigest(),
                    scenario_count=plan["scenario_count"], shard_id=args.shard_id, sweep_budget=1,
                    provenance=provenance, storage_gate_sha256=gate_sha,
                    discretization="finite_depth_node_fields_v1 不等同舊particle-stage取樣",
                    full_run_complete=False, full_validation_performed=False)
    with (args.session / "trial-manifest.json").open("x") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    available = memory_available()
    if available is None or available < MIN_MEMORY:
        raise RuntimeError("啟動前可用記憶體不足30GiB或不可觀測")
    stopped = []
    previous = {s: signal.signal(s, lambda number, frame: stopped.append(number))
                for s in (signal.SIGINT, signal.SIGTERM)}
    process = None
    started = time.monotonic()
    try:
        with ((args.session / "worker.log").open("x") as log,
              (args.session / "resources.jsonl").open("x") as journal):
            process = subprocess.Popen(command, cwd=args.project_root, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            print(json.dumps(dict(worker_pid=process.pid, run_id=plan["run_id"],
                                  session=str(args.session))), flush=True)
            while process.poll() is None:
                available = memory_available()
                journal.write(json.dumps(dict(utc=datetime.now(UTC).isoformat(),
                    elapsed_seconds=time.monotonic() - started, mem_available_bytes=available,
                    process=process_snapshot(process))) + "\n")
                journal.flush()
                if stopped or available is None or available < MIN_MEMORY:
                    raise RuntimeError("安全停止：記憶體保護或操作訊號；不得宣告小試通過")
                time.sleep(5)
            print(json.dumps(dict(exit_code=process.returncode,
                                  elapsed_seconds=time.monotonic() - started)), flush=True)
            return process.returncode
    finally:
        if process is not None:
            _stop_active_processes({"trial": process}, first_signal=signal.SIGINT, grace_seconds=300)
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
