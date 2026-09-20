"""五站各20受體的純流操作入口；只執行完整正式計畫中的十個固定分片。

本腳本不建立或修改run計畫、設定、受體、種子及forcing。操作者先以既有
run-create建立新的formal/full/no_stokes/M1工作區；明示--resume才允許接續
同一個固定子集，不修改原計畫或重設進度。再明示路徑與已通過的儲存
檢查證據。每站共用一個run-worker程序處理兩片，記憶體不足或任一站失敗時
對本次子程序送SIGINT並等候300秒保存checkpoint。輸出只稱100顆選取子集，
整個50,000情境計畫仍未完成；真實資料QC與工程失敗分別保留，不做全批驗收。
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

SELECTION = {
    "gongliao": (1736575200000000000, "arrival_bed_e24212dda777a178258df4c7"),
    "guishan": (1736575200000000000, "arrival_bed_5fada30731da2b6e63dbc7c5"),
    "hsinchu": (1735621200000000000, "arrival_bed_06ed73785ad899b9518066a4"),
    "lienchiang": (1736233200000000000, "arrival_bed_e8bc4436d76bac0526568ad2"),
    "nanwan": (1736794800000000000, "arrival_bed_5413cd1f81a554496c0ea31d"),
}
"""事前固定各站最早到達群組；不同站UTC不同，不依軌跡結果選樣。"""
MATERIAL = "oca_aluminum_crushed_cylinder"
MIN_MEMORY = 30 * 1024**3
"""Linux MemAvailable最低可用位元組數；讀取失敗也不允許繼續啟動。"""
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS")
FRAME_CACHE_ENV = {'LBT_BAYTRACE_FRAME_CACHE_MAX_FRAMES': '20',
                   'LBT_BAYTRACE_FRAME_CACHE_MAX_BYTES': '4294967296'}
"""僅固定100入口覆寫每worker場快取上限；不影響其他部署或物理設定。"""


def frame_cache_policy(env):
    """保存實際傳給五個worker的容量；場快取常駐上限不等於程序RSS上限。"""
    return dict(environment={k: env[k] for k in FRAME_CACHE_ENV},
                max_cached_frames=int(env['LBT_BAYTRACE_FRAME_CACHE_MAX_FRAMES']),
                max_cached_bytes=int(env['LBT_BAYTRACE_FRAME_CACHE_MAX_BYTES']),
                semantics='執行記憶體配置非物理設定；場快取常駐上限非程序RSS上限')


def worker_environment(inherited):
    """固定100入口覆寫五個執行緒限制與有界場快取，其他環境原樣保留。"""
    env = dict(inherited)
    env.update({key: '1' for key in THREAD_ENV})
    env.update(FRAME_CACHE_ENV)
    return env


def select_shards(plan, rows):
    """按既有情境表核對五組身分及完整20受體，回傳各站兩個既有分片ID。

    切片範圍沿用計畫；若既有分片混入其他情境，直接拒絕，不重建selection schema。
    另驗每站指定時間確為該鋁材的最早時間，且涵蓋站內全部受體。
    """
    if (plan.get("run_kind") != "formal" or plan.get("experiment_case_id") != "no_stokes"
            or plan.get("members_per_scenario") != 1 or plan.get("scenario_count") != 50000
            or plan.get("scenario_selection", {}).get("mode") != "full"
            or len(rows) != 50000 or len(plan["shards"]) != 5000):
        raise ValueError("必須是完整50,000情境／5,000片formal/full/no_stokes/M1計畫")
    selected = {}
    for site, (utc_ns, arrival_id) in SELECTION.items():
        material_rows = [r for r in rows if r["study_site_id"] == site and r["material_id"] == MATERIAL]
        group = [r for r in material_rows if r["arrival_time_utc_ns"] == utc_ns]
        receptors = {r["receptor_id"] for r in rows if r["study_site_id"] == site}
        if (not material_rows or min(r["arrival_time_utc_ns"] for r in material_rows) != utc_ns
                or len(group) != 20 or len(receptors) != 20
                or {r["receptor_id"] for r in group} != receptors
                or any(r["arrival_time_id"] != arrival_id or r["settling_velocity_mps"] != -0.02
                       for r in group)):
            raise ValueError(f"{site}固定鋁材／最早到達／完整20受體不符")
        wanted = {r["scenario_id"] for r in group}
        covered, ids = set(), []
        for shard in plan["shards"]:
            subset = rows[shard["scenario_start_index"]:shard["scenario_stop_index"]]
            found = {r["scenario_id"] for r in subset}
            if not found & wanted:
                continue
            if not found <= wanted or len(subset) != 10 or shard["particle_count"] != 10:
                raise ValueError(f"{site}分片混入未選情境或不是10粒子")
            covered.update(found)
            ids.append(shard["shard_id"])
        if len(ids) != 2 or covered != wanted:
            raise ValueError(f"{site}必須恰為兩片且完整涵蓋20粒子")
        selected[site] = ids
    if len({sid for ids in selected.values() for sid in ids}) != 10:
        raise ValueError("五站分片不可重複")
    return selected


def validate_subset_progress(plan, progress, selected, *, resume=False):
    """唯讀核對續跑範圍；十片以外仍須未啟動，不能以續跑擴大研究母體。

    新跑維持全部PLANNED且零步數；續跑僅允許固定十片已有合法生命週期。
    此處不採認輸出、不重設步數，也不代替控制器的片鎖、歷代檢查點及輸出驗證。
    """
    if type(resume) is not bool:
        raise TypeError("resume必須是bool")
    expected = {row["shard_id"] for row in plan["shards"]}
    chosen = {sid for ids in selected.values() for sid in ids}
    if (len(chosen) != 10 or not chosen <= expected
            or set(progress["shards"]) != expected):
        raise ValueError("固定十片或計畫／進度分片集合不符")
    for sid, row in progress["shards"].items():
        untouched = row["lifecycle"] == "PLANNED" and row["particle_steps"] == 0
        if not resume or sid not in chosen:
            if not untouched:
                raise ValueError("未明示續跑或非選取分片已有執行狀態／步數")
        elif row["lifecycle"] not in {"PLANNED", "RUNNING", "PAUSED", "FAILED", "COMPLETE"}:
            raise ValueError(f"所選分片生命週期不允許續跑：{sid}")


def memory_available():
    """讀取Linux可用記憶體，單位位元組；不可觀測時回傳None供外層停止。"""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return None


def worker_command(args, shard_ids):
    """使用同一Python及明示程式目錄呼叫既有CLI，僅明示續跑才傳--resume。

    不新增驗證略過介面；既有CLI及逐片還原驗證仍照原契約執行。
    """
    bootstrap = (
        "import sys; sys.path.insert(0,sys.argv.pop(1)); "
        "from lagrangian_backtracking.cli import main; raise SystemExit(main())"
    )
    command = [sys.executable, "-u", "-c", bootstrap, str(args.project_root / "src"),
               "run-worker", str(args.workspace), "--config", str(args.config),
               "--ocm-native-root", str(args.ocm_native_root),
               "--ocm-reconstruction-root", str(args.ocm_reconstruction_root),
               "--checkpoint-root", str(args.checkpoint_root), "--execution-priority"]
    resume = getattr(args, "resume", False)
    if type(resume) is not bool:
        raise TypeError("resume必須是bool")
    if resume:
        command.append("--resume")
    for sid in shard_ids:
        command.extend(("--shard-id", sid))
    return command


def process_snapshot(process):
    """只讀本次Popen的PID資源；CPU累積秒、RSS位元組與開始tick可核對PID身份。"""
    result = {"pid": process.pid, "exit_code": process.poll()}
    try:
        fields = Path(f"/proc/{process.pid}/stat").read_text().rsplit(")", 1)[1].split()
        result.update(cpu_seconds=(int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK"),
                      start_ticks=int(fields[19]), rss_bytes=int(fields[21]) * os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, IndexError):
        result["resource_unavailable"] = True
    return result


def completed_evidence(workspace, ids):
    """讀取已由run_shard完整驗證／發布的兩片小型manifest，保存實際native證據。

    拒絕理由可能包括真實QC，故完整保留但不要求其計數全部為零。工程例外使片
    失敗，由外層生命週期檢查拒絕；成功還須有原版推進、群組與來源證據，零舊核補算。
    """
    reports = []
    for sid in ids:
        data = json.loads((workspace / "shards" / sid / "manifest.json").read_text())
        report = data["run_metadata"]["advection_execution"]
        if (report.get("require_native") is not True or report.get("fallback_particle_steps") != 0
                or report.get("counter_scope") != "shard_invocation_delta_v1"
                or not report.get("source_sha256")):
            raise ValueError("完成輸出缺少嚴格原版／逐片／來源證據或曾使用舊核心")
        reports.append(report)
    completed = sum(r["native_completed_particle_steps"] for r in reports)
    grouped = any(int(k) > 1 and v > 0 for r in reports for k, v in r["group_size_histogram"].items())
    if completed <= 0 or not grouped:
        raise ValueError("該站沒有實際原版批次推進，不能宣告子集完成")
    return {"native_completed_particle_steps": completed, "shard_reports": reports}


def supervise(args, selected, session, env):
    """固定五席、各啟動一次；監看資源與所選進度，任何失敗安全收尾且不重試。"""
    from lagrangian_backtracking.parallel_execution import _stop_active_processes
    from lagrangian_backtracking.run_control import load_run_progress

    started = time.monotonic()
    active, streams, finished = {}, [], set()
    interrupted = []
    previous = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, lambda number, frame: interrupted.append(number))
    journal = (session / "events.jsonl").open("x", encoding="utf-8")

    def record(event, **fields):
        """追加事件並即時印出，讓啟動、原版完成數與停止原因可由操作者直接確認。"""
        row = dict(utc=datetime.now(UTC).isoformat(), elapsed_seconds=time.monotonic() - started,
                   event=event, **fields)
        journal.write(json.dumps(row, ensure_ascii=False) + "\n")
        journal.flush()
        print(json.dumps(row, ensure_ascii=False), flush=True)

    status = "FAILED"
    try:
        for site, ids in selected.items():
            available = memory_available()
            if interrupted or available is None or available < MIN_MEMORY:
                raise RuntimeError("啟動前記憶體不足／不可觀測或收到停止訊號")
            stream = (session / f"{site}.log").open("x", encoding="utf-8")
            streams.append(stream)
            process = subprocess.Popen(worker_command(args, ids), cwd=args.project_root,
                                       env=env, stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            active[site] = process
            record("worker_started", site=site, shard_ids=ids, pid=process.pid)
        last_steps = {}
        while len(finished) != 5:
            if interrupted:
                status = "INTERRUPTED"
                raise InterruptedError("操作者要求安全停止")
            available = memory_available()
            if available is None or available < MIN_MEMORY:
                status = "MEMORY_GUARD_STOP"
                raise RuntimeError("MemAvailable低於30GiB或不可觀測")
            progress = load_run_progress(args.workspace)
            rows = {sid: progress["shards"][sid] for ids in selected.values() for sid in ids}
            for sid, row in rows.items():
                now = (row.get("particle_steps", 0), row.get("checkpoint_sequence", 0))
                if sid in last_steps and any(a < b for a, b in zip(now, last_steps[sid], strict=True)):
                    raise RuntimeError("所選分片步數或checkpoint序列倒退")
                last_steps[sid] = now
            record("snapshot", mem_available_bytes=available, revision=progress["revision"],
                   processes={site: process_snapshot(p) for site, p in active.items()}, shards=rows)
            for site, process in active.items():
                if site in finished or process.poll() is None:
                    continue
                ids = selected[site]
                if process.returncode != 0 or any(rows[sid]["lifecycle"] != "COMPLETE" for sid in ids):
                    raise RuntimeError(f"{site}退出或分片未完成，見該站log")
                evidence = completed_evidence(args.workspace, ids)
                record("station_compute_complete", site=site, selected_particles=20, **evidence)
                finished.add(site)
            if len(finished) != 5:
                time.sleep(5)
        status = "SELECTED_100_COMPUTE_COMPLETE_FULL_RUN_INCOMPLETE"
        return 0
    except BaseException as exc:
        record("stopping", status=status, error_type=type(exc).__name__, error=str(exc))
        return 2
    finally:
        # 既有helper只處理本次Popen建立的獨立程序群組，先SIGINT等300秒，
        # 再採原有TERM/KILL有界收尾；不以名稱pkill，不碰舊run或其他操作者程序。
        codes = _stop_active_processes(active, first_signal=signal.SIGINT, grace_seconds=300)
        record("operator_finished", status=status, exit_codes=codes, completed_sites=sorted(finished),
               full_run_complete=False, full_validation_performed=False)
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        for stream in streams:
            stream.close()
        journal.close()


def main(argv=None):
    """核對新跑或明示續跑工作區、來源及NFS；記錄新執行來源但不改原計畫。

    --resume只放行既定十片的歷史狀態，--execute才啟動。所有原有完整驗證
    保留，不呼叫reconcile；逐片還原／完成輸出採認交由既有控制器。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workspace", "source-workspace", "config", "project-root", "checkpoint-root",
                 "ocm-native-root", "ocm-reconstruction-root", "scratch-root", "log-root",
                 "numba-cache-dir", "storage-gate-evidence"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true", help="接續同一工作區固定十片，不重建計畫")
    args = parser.parse_args(argv)
    args.project_root = args.project_root.resolve(strict=True)
    # 科學套件首次匯入前固定五個thread限制；所有可寫快取沿用操作者提供的新NFS scratch。
    env = worker_environment(os.environ)
    env.update(PYTHONDONTWRITEBYTECODE="1", NUMBA_CACHE_DIR=str(args.numba_cache_dir))
    for key in ("TMPDIR", "MPLCONFIGDIR", "XDG_CACHE_HOME", "UV_CACHE_DIR"):
        path = Path(env[key]).resolve(strict=True)
        if not path.is_relative_to(args.scratch_root.resolve(strict=True)):
            raise ValueError(f"{key}必須在本次scratch內")
    os.environ.update(env)
    sys.path.insert(0, str(args.project_root / "src"))
    import pyarrow.parquet as pq

    from lagrangian_backtracking.parallel_execution import (
        prepare_execution_directories,
        validate_run_inputs,
        validate_storage_roots,
    )
    from lagrangian_backtracking.run_control import load_run_plan, load_run_progress

    scratch, logs, cache, gate_sha = validate_storage_roots(
        scratch_root=args.scratch_root, log_root=args.log_root,
        numba_cache_dir=args.numba_cache_dir, gate_evidence=args.storage_gate_evidence)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.project_root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=args.project_root, text=True)
    if actual != args.expected_commit or dirty:
        raise ValueError("必須使用指定commit的乾淨獨立checkout")
    # 來源工作區永遠唯讀；續跑也不能把來源50k當作固定100的執行目標。
    if args.workspace.resolve() == args.source_workspace.resolve():
        raise ValueError("新工作區不可等於唯讀來源")
    initial = load_run_progress(args.workspace)
    if not args.resume and any(
        r["lifecycle"] != "PLANNED" or r["particle_steps"] != 0 for r in initial["shards"].values()
    ):
        raise ValueError("此入口只接受未執行的新工作區；續跑須另行明示")
    # 先核對選片與進度範圍，讓非選取片曾運算的工作區在完整掃描前即被拒絕；
    # 以下仍以既有validator驗證全部不可變檔與檢查點，不把這項早期檢查當驗收。
    candidate = load_run_plan(args.workspace)
    selected = select_shards(candidate, pq.read_table(args.workspace / "scenario_table.parquet").to_pylist())
    validate_subset_progress(candidate, initial, selected, resume=args.resume)
    plan, config, _, _, _, provenance = validate_run_inputs(
        workspace=args.workspace, config_path=args.config, project_root=args.project_root,
        checkpoint_root=args.checkpoint_root, execution_priority=True)
    source = load_run_plan(args.source_workspace)
    for key in ("config_hash", "master_seed", "component_canonical_hashes", "geometry_canonical_hashes"):
        if plan[key] != source[key]:
            raise ValueError(f"新run與原來源的{key}不符")
    if plan["files"]["scenario_table.parquet"] != source["files"]["scenario_table.parquet"]:
        raise ValueError("情境表與原來源不符")
    if (config.boundaries.max_backtrack_days != 30
            or config.execution.advection_backend != "baytrace_native_cpu_v1"):
        raise ValueError("須保留H30並使用原版批次後端")
    selected = select_shards(plan, pq.read_table(args.workspace / "scenario_table.parquet").to_pylist())
    validate_subset_progress(plan, load_run_progress(args.workspace), selected, resume=args.resume)
    session, _ = prepare_execution_directories(scratch_root=scratch, log_root=logs,
                                              numba_cache_dir=cache, run_id=plan["run_id"])
    manifest = dict(run_id=plan["run_id"], actual_commit=actual, actual_provenance=provenance,
                    plan_sha256=sha256((args.workspace / "run_plan.json").read_bytes()).hexdigest(),
                    storage_gate_sha256=gate_sha, selected_shards=selected, selection=SELECTION,
                    material=MATERIAL, settling_velocity_mps=-0.02, selected_particles=100,
                    resume=args.resume, initial_progress_revision=initial["revision"],
                    initial_selected_progress={sid: initial["shards"][sid]
                                               for ids in selected.values() for sid in ids},
                    immutable_run_plan_modified=False,
                    total_run_particles=50000, experiment_case="no_stokes", workers=5,
                    thread_environment={k: env[k] for k in THREAD_ENV}, full_run_complete=False,
                    frame_cache_limits=frame_cache_policy(env),
                    policy="每站固定最早到達20受體；不同站不同UTC；完整驗收另行執行")
    with (session / "demo-manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"prepared": True, "run_id": plan["run_id"], "log_session": str(session),
                      "selected_shards": selected, "execute": args.execute,
                      "resume": args.resume}), flush=True)
    return supervise(args, selected, session, env) if args.execute else 0


if __name__ == "__main__":
    raise SystemExit(main())
