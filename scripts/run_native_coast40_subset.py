"""已審模型岸界40顆的薄操作入口；預設只核對，不啟動模型。

只接受已審來源／新計畫綁定，先核完整五站母體再取南灣、連江各原兩片。
不更改計画／來源、不提供任意站數或續跑；重用固定100的原版證據、記憶體
守護及300秒安全收尾。模型岸界只覆蓋兩指定區域，不宣稱全域失敗已解決。
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import stat
import subprocess
import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import run_current100_subset as shared

REVIEWED = {
    "source_plan": "183ceebc4203b1008379f4947e9bab4cd6affb76ead78ed4b0235bbdcab874db",
    "target_plan": "bcab453d2e0bd05678a4c33c32d11b56a502ded6e316b484b95a3f625748fac7",
    "config": "6c1e136ad340dc3a37fb8cf4e493cf16a4ca788c844395700665681295611293",
    "input_binding": "222dc205a8f00bb242dba031f159c79025339379171af8c5cf26ba148423f8c7",
    "coast": "e0164decc42b765ea27ba44525cc16d2c4c672ee8f644eac5f1ee92b1a661dce",
}
"""僅保存主審的不可變身分，不將現場SERVER檔案路徑提交Git。"""
SITES = ("nanwan", "lienchiang")


def file_sha(path):
    """核對普通檔的原始位元組，不接受符號連結替换綁定。"""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"綁定來源必須是普通檔：{path.name}")
    return sha256(path.read_bytes()).hexdigest()


def validate_binding_pair(args, source, target, config):
    """精確核對主審來源／目標及兩張完整表，不能把新增岸界當成相同綁定。

    舊三項幾何及所有來源component相等；目標僅新增native coast。清冊與設定
    完整驗證仍由 validate_run_inputs 執行，本檢查不是略過驗證的替代品。
    """
    for path, expected in (
        (args.source_workspace / "run_plan.json", REVIEWED["source_plan"]),
        (args.workspace / "run_plan.json", REVIEWED["target_plan"]),
    ):
        if file_sha(path) != expected:
            raise ValueError("來源／目標plan SHA不屬於本次核准對")
    if (
        target["config_hash"] != REVIEWED["config"]
        or config.config_hash() != REVIEWED["config"]
        or target["checkpoint_input_binding_hash"] != REVIEWED["input_binding"]
    ):
        raise ValueError("目標config／input binding不符")
    ref = config.normalized_payload()["geometry"]["native_coast_manifest"]
    if ref["sha256"] != REVIEWED["coast"] or file_sha(args.config.parent / ref["path"]) != REVIEWED["coast"]:
        raise ValueError("岸界SHA不符")
    for key in (
        "master_seed",
        "component_canonical_hashes",
        "members_per_scenario",
        "experiment_case_id",
        "seed_policy",
        "active_chunk_size",
        "checkpoint_interval_sweeps",
    ):
        if source[key] != target[key]:
            raise ValueError(f"來源／目標{key}不符")
    if source.get("random_stream_id") != target.get("random_stream_id"):
        raise ValueError("共同亂數流身分不符")
    old_geometry, new_geometry = source["geometry_canonical_hashes"], target["geometry_canonical_hashes"]
    if (
        set(old_geometry) != {"domain", "local", "open_boundary"}
        or set(new_geometry) != set(old_geometry) | {"native_coast"}
        or any(new_geometry[k] != v for k, v in old_geometry.items())
    ):
        raise ValueError("幾何只能保留原三項並新增已審岸界")
    tables = {}
    for name in ("scenario_table.parquet", "seed_table.parquet"):
        before, after = file_sha(args.source_workspace / name), file_sha(args.workspace / name)
        if (
            before != after
            or before != source["files"][name]["sha256"]
            or after != target["files"][name]["sha256"]
        ):
            raise ValueError(f"來源／目標{name} SHA不符")
        tables[name] = after
    if (
        config.boundaries.max_backtrack_days != 30
        or config.execution.advection_backend != "baytrace_native_cpu_v1"
    ):
        raise ValueError("必須保留H30與原版批次後端")
    return tables


def select_and_validate_fresh(plan, rows, progress):
    """先驗原五站固定母體再取兩站；新綁定不允許四片內外任何既有進展。"""
    all_sites = shared.select_shards(plan, rows)
    selected = {site: all_sites[site] for site in SITES}
    shared.selection_size(selected)
    expected = {row["shard_id"] for row in plan["shards"]}
    if set(progress["shards"]) != expected:
        raise ValueError("進度與完整母體分片集合不符")
    for row in progress["shards"].values():
        if (
            row["lifecycle"] != "PLANNED"
            or row["particle_steps"] != 0
            or row.get("checkpoint_sequence", 0) != 0
        ):
            raise ValueError("新岸界四片及未選取片皆必須PLANNED零步／零checkpoint")
    return selected


@contextmanager
def operator_lock(workspace):
    """同一目標工作區的操作者互斥；鎖位於其父目錄，不污染run白名單。

    不使用run_gate獨占鎖（worker需要共用它）；本鎖從驗證涵蓋到子程序全部
    安全退出，既有assigned shard鎖仍由控制器取得。零長度鎖不刪除以免inode競爭，
    不寫PID於鎖；PID及啟動tick另存本次session。禁止繼承FD至worker。
    """
    path = workspace.parent / f".{workspace.name}.native-coast40-operator.lock"
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size != 0:
            raise ValueError("operator鎖必須是零長度普通檔")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("此workspace已有native coast40 operator") from exc
        yield path
    finally:
        os.close(fd)


def prepare_environment(args):
    """只容許/data/LBT內的目標與可寫根，所有快取繼承值必須位於本次scratch。

    後續仍執行既有storage gate及即時掛載驗證，不僅憑路徑前綴聲稱NFS。
    """
    root = Path("/data/LBT").resolve(strict=True)
    for key in ("workspace", "checkpoint_root", "scratch_root", "log_root", "numba_cache_dir"):
        raw = getattr(args, key)
        path = raw.resolve(strict=True)
        if raw.is_symlink() or not path.is_dir() or path == root or not path.is_relative_to(root):
            raise ValueError(f"{key}必須是/data/LBT下既有普通目錄")
        setattr(args, key, path)
    env = shared.worker_environment(os.environ)
    env.update(PYTHONDONTWRITEBYTECODE="1", NUMBA_CACHE_DIR=str(args.numba_cache_dir))
    for key in ("TMPDIR", "MPLCONFIGDIR", "XDG_CACHE_HOME", "UV_CACHE_DIR", "NUMBA_CACHE_DIR"):
        path = Path(env[key]).resolve(strict=True)
        if not path.is_dir() or not path.is_relative_to(args.scratch_root):
            raise ValueError(f"{key}必須在本次NFS scratch內")
    return env


def main(argv=None):
    """正式核對後產生40顆manifest；預設不Popen，明示execute才共用守護生命週期。"""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "workspace",
        "source-workspace",
        "config",
        "project-root",
        "checkpoint-root",
        "ocm-native-root",
        "ocm-reconstruction-root",
        "scratch-root",
        "log-root",
        "numba-cache-dir",
        "storage-gate-evidence",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    args.project_root = args.project_root.resolve(strict=True)
    args.source_workspace = args.source_workspace.resolve(strict=True)
    args.config = args.config.resolve(strict=True)
    env = prepare_environment(args)
    if args.workspace == args.source_workspace:
        raise ValueError("來源只能唯讀且必須與目標分開")
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.project_root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=args.project_root, text=True)
    if (
        actual != args.expected_commit
        or dirty
        or Path(__file__).resolve() != args.project_root / "scripts" / Path(__file__).name
    ):
        raise ValueError("必須執行指定commit乾淨checkout中的operator")
    os.environ.update(env)
    sys.path.insert(0, str(args.project_root / "src"))
    import pyarrow.parquet as pq

    from lagrangian_backtracking.parallel_execution import (
        _live_mount_identity,
        prepare_execution_directories,
        validate_run_inputs,
        validate_storage_roots,
    )
    from lagrangian_backtracking.run_control import load_run_plan, load_run_progress

    scratch, logs, cache, gate_sha = validate_storage_roots(
        scratch_root=args.scratch_root,
        log_root=args.log_root,
        numba_cache_dir=args.numba_cache_dir,
        gate_evidence=args.storage_gate_evidence,
    )
    identity = _live_mount_identity(scratch)
    paths = (args.workspace, args.workspace.parent, args.checkpoint_root, logs, cache) + tuple(
        Path(env[key])
        for key in ("TMPDIR", "MPLCONFIGDIR", "XDG_CACHE_HOME", "UV_CACHE_DIR", "NUMBA_CACHE_DIR")
    )
    if any(_live_mount_identity(path) != identity for path in paths):
        raise ValueError("目標／checkpoint／log／cache須與已驗scratch同一NFS")
    available = shared.memory_available()
    if available is None or available < shared.MIN_MEMORY:
        raise ValueError("MemAvailable不足30GiB或不可觀測")
    with operator_lock(args.workspace) as lock_path:
        initial = load_run_progress(args.workspace)
        candidate = load_run_plan(args.workspace)
        rows = pq.read_table(args.workspace / "scenario_table.parquet").to_pylist()
        selected = select_and_validate_fresh(candidate, rows, initial)
        # 完整正式驗證原樣保留；actual operator commit不可寫回immutable plan。
        plan, config, _, _, _, provenance = validate_run_inputs(
            workspace=args.workspace,
            config_path=args.config,
            project_root=args.project_root,
            checkpoint_root=args.checkpoint_root,
            execution_priority=True,
        )
        source = load_run_plan(args.source_workspace)
        tables = validate_binding_pair(args, source, plan, config)
        select_and_validate_fresh(plan, rows, load_run_progress(args.workspace))
        session, _ = prepare_execution_directories(
            scratch_root=scratch, log_root=logs, numba_cache_dir=cache, run_id=plan["run_id"]
        )
        own_process = shared.process_snapshot(
            type("OwnProcess", (), {"pid": os.getpid(), "poll": lambda self: None})()
        )
        manifest = dict(
            run_id=plan["run_id"],
            source_run_id=source["run_id"],
            reviewed_binding_pair=REVIEWED,
            utc=datetime.now(UTC).isoformat(),
            operator_process=own_process,
            operator_lock=str(lock_path),
            actual_commit=actual,
            actual_provenance=provenance,
            plan_code_provenance=plan["code_provenance"],
            immutable_run_plan_modified=False,
            table_sha256=tables,
            selected_shards=selected,
            selection={site: shared.SELECTION[site] for site in SITES},
            selected_particles=40,
            total_run_particles=50000,
            workers=2,
            experiment_case="no_stokes",
            resume=False,
            execute=args.execute,
            full_run_complete=False,
            full_science_validation_performed=False,
            initial_progress_revision=initial["revision"],
            storage_gate_sha256=gate_sha,
            thread_environment={k: env[k] for k in shared.THREAD_ENV},
            frame_cache_limits=shared.frame_cache_policy(env),
            writable_environment={
                k: env[k]
                for k in ("TMPDIR", "MPLCONFIGDIR", "XDG_CACHE_HOME", "UV_CACHE_DIR", "NUMBA_CACHE_DIR")
            },
            worker_commands={site: shared.worker_command(args, ids) for site, ids in selected.items()},
            policy="40顆選取子集；兩指定模型岸界碰岸首接觸停止；原版/QC證據保留，完整科學驗收待補",
        )
        with (session / "demo-manifest.json").open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
        print(
            json.dumps(
                {
                    "prepared": True,
                    "execute": args.execute,
                    "workers": 2,
                    "run_id": plan["run_id"],
                    "log_session": str(session),
                    "selected_shards": selected,
                }
            ),
            flush=True,
        )
        return shared.supervise(args, selected, session, env) if args.execute else 0


if __name__ == "__main__":
    raise SystemExit(main())
