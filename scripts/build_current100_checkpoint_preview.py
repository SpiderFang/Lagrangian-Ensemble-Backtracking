"""從固定100顆純流run的已發布checkpoint製作明示未完成的即時軌跡圖。

只讀指定站兩片的目前已發布世代及其必要歷史鏈，沿用既有完整checkpoint
binding／checksum／RunUnit驗證；不讀forcing、不改run、不等待或宣稱H30完成。
尚無checkpoint的粒子不畫、不補初始線，分母明示已保存數／該站20／全部100。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from lagrangian_backtracking.checkpoint import CheckpointBinding, load_execution_checkpoint
from lagrangian_backtracking.config import load_config
from lagrangian_backtracking.outputs import sha256_file
from lagrangian_backtracking.run_control import load_run_plan, load_run_progress
from lagrangian_backtracking.run_validation import _build_shards, _load_scenarios, _validate_run_files
from lagrangian_backtracking.runner import iter_run_units


def _script(name):
    """載入既有選片或配色helper；路徑固定在同一套已部署腳本內。"""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_snapshot(run, checkpoint_root, site):
    """擷取已發布世代的已保存位置；下一世代更新不會使此不可變快照失效。"""
    plan, progress = load_run_plan(run), load_run_progress(run)
    errors = []
    _validate_run_files(run, plan, errors)
    scenarios = _load_scenarios(run, plan, errors)
    if errors:
        raise ValueError(errors)
    selected = _script("run_current100_subset").select_shards(plan, [asdict(s) for s in scenarios])
    if site not in selected:
        raise ValueError("未知固定站點")
    particles, sources = [], []
    for shard in _build_shards(plan, scenarios):
        if shard.shard_id not in selected[site]:
            continue
        row = progress["shards"][shard.shard_id]
        relative = row["checkpoint_relative_path"]
        if relative is None:
            continue
        path = checkpoint_root / relative
        expected_parent = checkpoint_root / plan["run_id"] / shard.shard_id
        if path.parent != expected_parent or not path.name.startswith("checkpoint-"):
            raise ValueError("checkpoint位置未繫結固定run／分片")
        for part in (checkpoint_root, checkpoint_root / plan["run_id"], expected_parent, path):
            if part.is_symlink() or not part.is_dir():
                raise ValueError("checkpoint根或世代不是普通目錄")
        binding = CheckpointBinding(
            config_hash=plan["config_hash"], input_inventory_hash=plan["checkpoint_input_binding_hash"],
            experiment_case_id=plan["experiment_case_id"], shard_id=shard.shard_id,
            seed_policy=plan["seed_policy"], code_commit=plan["code_provenance"]["git_commit"],
            random_stream_id=plan.get("random_stream_id"))
        units = list(iter_run_units(shard, master_seed=plan["master_seed"],
                                   random_stream_id=plan.get("random_stream_id")))
        loaded = load_execution_checkpoint(path, expected_binding=binding, expected_run_units=units)
        if loaded.sequence != row["checkpoint_sequence"]:
            raise ValueError("目前progress引用的checkpoint序列不符")
        sources.append(dict(shard_id=shard.shard_id, sequence=loaded.sequence,
                            checkpoint=str(path), manifest_sha256=sha256_file(path / "checkpoint.json")))
        for execution in loaded.executions:
            particles.append(dict(particle_id=execution.state.particle_id, step_count=execution.step_count,
                                  status=execution.state.status.name,
                                  observations=[{key: getattr(obs, key) for key in
                                                 ("time_utc_ns", "age_seconds", "x_m", "y_m", "z_m")}
                                                for obs in execution.observations]))
    if not particles or not any(p["step_count"] > 0 for p in particles):
        raise ValueError("此站尚無已保存且實際推進的checkpoint")
    return dict(run_id=plan["run_id"], config_hash=plan["config_hash"], site=site, sources=sources,
                particles=particles, particle_count=len(particles), station_denominator=20,
                selected_total=100, run_total=50000, complete=False, experiment_case="no_stokes",
                arrival_utc_ns=_script("run_current100_subset").SELECTION[site][0],
                captured_utc=datetime.now(UTC).isoformat(), plan_sha256=sha256_file(run / "run_plan.json"))


def render_snapshot(payload, output, label, project_root):
    """兩面板保存水平公尺軌跡與垂向年齡；僅連接原始保存點，標示當前末端而非來源。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.text import Text

    style = _script("build_completed_subset_preview")._renderer(project_root)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    first = payload["particles"][0]["observations"][0]
    for particle in payload["particles"]:
        obs = particle["observations"]
        axes[0].plot([o["x_m"] - first["x_m"] for o in obs],
                     [o["y_m"] - first["y_m"] for o in obs], linewidth=1, color="#2166ac")
        axes[1].plot([o["age_seconds"] / 3600 for o in obs], [o["z_m"] for o in obs],
                     linewidth=1, color="#2166ac")
    axes[0].set(xlabel="相對第一受體投影東向距離（m）", ylabel="投影北向距離（m）", aspect="equal")
    axes[1].set(xlabel="實際已保存回溯年齡（小時）", ylabel="垂向位置 z（m，向上為正）")
    for ax in axes:
        ax.grid(alpha=0.25)
    arrival = datetime.fromtimestamp(payload["arrival_utc_ns"] / 1e9, UTC).isoformat()
    fig.suptitle(f"{label}｜純流未完成軌跡快照｜已保存{payload['particle_count']}/站20/總選取100\n"
                 f"到達UTC {arrival}｜鋁材 −0.02 m/s｜H30固定日曆窗與原擴散／seed未改")
    fig.text(0.5, 0.025, "僅已保存位置；線尾是當前保存末端，非來源認定。其餘粒子尚未列入；非正式驗收。",
             ha="center")
    font = style._display_font()
    for text in fig.findobj(Text):
        size = text.get_fontsize()
        text.set_fontproperties(font)
        text.set_fontsize(size)
    fig.tight_layout(rect=(0, 0.06, 1, 0.87))
    style._save_baytrace_figure(fig, output / "unfinished_trajectories.png")
    plt.close(fig)


def main():
    """明示新成果目錄，先驗證單站checkpoint再排他發布圖與可重繪位置快照。"""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "checkpoint-root", "config", "project-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--site", required=True)
    args = parser.parse_args()
    payload = extract_snapshot(args.run, args.checkpoint_root, args.site)
    config = load_config(args.config, formal_release=False)
    if config.config_hash() != payload["config_hash"]:
        raise ValueError("圖面設定未繫結原run")
    label = next(s.study_site_name_zh for s in config.study_sites if s.study_site_id == args.site)
    args.output.mkdir(parents=True, exist_ok=False)
    render_snapshot(payload, args.output, label, args.project_root)
    payload["png_sha256"] = sha256_file(args.output / "unfinished_trajectories.png")
    with (args.output / "snapshot.json").open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, allow_nan=False)
    print(json.dumps({"output": str(args.output), "particles": payload["particle_count"], "complete": False}))


if __name__ == "__main__":
    main()
