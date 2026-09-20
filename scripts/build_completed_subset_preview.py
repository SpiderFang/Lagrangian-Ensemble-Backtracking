"""唯讀擷取指定完成分片，製作非正式整批驗收的站點成果圖。

輸入為原始 formal/full run、同一設定與海岸檔；只驗指定 trajectory 的全部
checksum、內容與身分，不讀 checkpoint 或 forcing，也不修改 run_kind、選擇模式、
幾何發布資格或物理資料。公尺位置與秒制年齡原樣保存，經緯度僅供圖面交換。
輸出是獨立 completed-subset artifact；M=1 與部分完成樣本不能解讀為來源機率。
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from shapely.geometry import mapping, shape

from lagrangian_backtracking.bed_residence import resolve_bed_residence_timing
from lagrangian_backtracking.config import load_config, resolve_flow_domain_id
from lagrangian_backtracking.manifests import (
    load_boundary_geometries,
    load_scenario_inputs,
    resolve_manifest_path,
)
from lagrangian_backtracking.outputs import sha256_file, validate_trajectory_shard
from lagrangian_backtracking.run_control import (
    _cross_check_plan_progress,
    load_run_plan,
    load_run_progress,
)
from lagrangian_backtracking.run_validation import (
    _build_shards,
    _expected_output_metadata,
    _load_scenarios,
    _relative_file,
    _validate_run_files,
    _validate_seed_table,
)
from lagrangian_backtracking.runner import iter_run_units

SELECTION_RULE = "每站依 arrival_time_utc_ns、material_id 排序，第一個全部完成的20受體組"
"""選擇只依時間、材質與完成狀態，不依軌跡長度、外觀或終止類型。"""
SCOPE_NOTE = "部分已完成情境｜全批完整驗收待補｜M=1，非來源機率"
"""每張圖均展示的資格限制，不取代原始 run 的正式身分。"""
NATIVE_COAST_SCOPE_NOTE = (
    "原100同受體重算替代｜每站20顆，不新增40、不混A/B分母\n"
    "非已執行50,000顆｜全批驗收待補｜底圖僅顯示，非物理岸界"
)
"""已審兩站重算的圖面限制；不能把新run身分當成新增研究母體。"""
POSITION_FIELDS = (
    "time_utc_ns",
    "age_seconds",
    "x_m",
    "y_m",
    "z_m",
    "eta_m",
    "bed_z_m",
)
"""只擷取繪圖所需的全部保存位置／環境欄位；不重新取樣速度或事件。"""


def _json(path: Path) -> dict:
    """讀取普通 JSON 檔；拒絕 symlink，避免意外跨越來源範圍。"""
    if path.is_symlink() or not path.is_file():
        raise ValueError("來源必須為普通檔案")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value) -> None:
    """只在新成果目錄排他寫入 JSON；缺值用 null，不產生非標準 NaN。"""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _snapshot(paths) -> dict[str, str]:
    """對指定來源建立 SHA-256；不遞迴掃描 run 或 checkpoint 歷史。"""
    result = {}
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError("來源不是普通檔案")
        result[str(path)] = sha256_file(path)
    return result


def _native_coast_snapshot(config, config_path, before):
    """將可選模型岸界與其靜態來源納入擷取前快照，不重寫科學載入器。

    路徑契約沿用 native_coast.load_native_coasts：岸界引用相對設定檔，
    source_files 相對岸界檔；以既有 resolve_manifest_path 固定相同基準。
    本函式只讀來源及記錄雜湊，SHA／投影／站點／幾何資格仍由既有loader拒絕錯綁。
    無設定時不增欄位或來源，保持原三幾何行為；有設定時來源缺檔不可略過。
    """
    reference = config.geometry.get("native_coast_manifest")
    if reference is None:
        return None
    path = resolve_manifest_path(config_path, reference["path"])
    before.update(_snapshot([path]))
    document = _json(path)
    sources = [resolve_manifest_path(path, item["path"]) for item in document["source_files"]]
    source_hashes = _snapshot(sources)
    before.update(source_hashes)
    return {
        "manifest_path": str(path),
        "manifest_sha256": before[str(path)],
        "source_files_sha256": source_hashes,
        "method": document["method"],
        "coverage_policy": document["coverage_policy"],
        "components": [
            {key: record[key] for key in (
                "study_site_id", "component_id", "flow_domain_id", "projection_wkt",
                "region_sha256", "edges_sha256",
            )}
            for record in document["records"]
        ],
        "basemap_role": "display_only_not_physical_boundary",
    }


def _unchanged_snapshot(before, message):
    """擷取及渲染兩階段均核對同一組普通檔案，不漏掉第四幾何的間接來源。"""
    after = _snapshot(Path(p) for p in before)
    if before != after:
        raise ValueError(message)
    return after


def _plot_code_provenance():
    """記錄實際繪圖checkout與腳本SHA，和原計畫中的不可變模型commit分開。

    以執行腳本位置而非cwd找Git來源；非checkout不能假造commit。追蹤檔dirty另記，
    未追蹤現場證據不當成已執行程式；不修改Git或載入另一份數值實作。
    """
    script = Path(__file__).resolve()
    root = script.parents[1]
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True
    )
    loader = Path(load_boundary_geometries.__code__.co_filename).resolve()
    return {"checkout": str(root), "commit": commit, "dirty_flag": bool(dirty),
            "script_path": str(script), "script_sha256": sha256_file(script),
            "geometry_loader_path": str(loader), "geometry_loader_sha256": sha256_file(loader)}


def select_completed_groups(scenarios, shards, progress, receptors, requested_ids):
    """驗證明示分片恰為各站首個完整20受體組，回傳按站點分組情境。

    身分與垂向／水平分母由當前受體 manifest 取得，不能從 receptor 字串猜測。
    此函式只讀記憶體中的靜態表與 progress，不開啟未選輸出；非完成、混站、混材質、
    重複受體或不是最早完整組均拒絕。限本次兩站、M=1、每片10情境的展示設計。
    """
    requested = set(requested_ids)
    if len(requested) != 4 or len(requested_ids) != 4:
        raise ValueError("必須指定四個不同完成分片")
    shard_by_id = {item.shard_id: item for item in shards}
    if not requested <= shard_by_id.keys():
        raise ValueError("未知分片")
    owner = {s.scenario_id: sh.shard_id for sh in shards for s in sh.scenarios}
    groups = defaultdict(list)
    for scenario in scenarios:
        key = (scenario.study_site_id, scenario.arrival_time_utc_ns, scenario.material_id)
        groups[key].append(scenario)
    selected = defaultdict(list)
    for sid in requested_ids:
        sh = shard_by_id[sid]
        if progress["shards"][sid]["lifecycle"] != "COMPLETE":
            raise ValueError("所選分片未完成")
        if len(sh.scenarios) != 10 or sh.particle_count != 10:
            raise ValueError("所選分片非10情境M1")
        for scenario in sh.scenarios:
            selected[scenario.study_site_id].append(scenario)
    if set(selected) != {"gongliao", "guishan"}:
        raise ValueError("必須分別為貢寮與龜山島")
    for site, values in selected.items():
        identities = {(s.arrival_time_id, s.material_id) for s in values}
        expected = {r.receptor_id for r in receptors if r.study_site_id == site}
        if len(values) != 20 or len(identities) != 1 or len(expected) != 20:
            raise ValueError("同站必須同到達時間與材質且恰20受體")
        if {s.receptor_id for s in values} != expected:
            raise ValueError("受體不完整或重複")
        candidates = sorted(key for key in groups if key[0] == site)
        first = next(
            (
                groups[key]
                for key in candidates
                if len(groups[key]) == 20
                and all(
                    progress["shards"][owner[s.scenario_id]]["lifecycle"] == "COMPLETE" for s in groups[key]
                )
            ),
            None,
        )
        if first is None or {s.scenario_id for s in values} != {s.scenario_id for s in first}:
            raise ValueError("不是固定規則的首個完整組")
    return dict(selected)


def validate_selected_output(root, plan, progress, shard):
    """完整驗證單片 output，回傳粒子表與 manifest，不載入歷史 execution 物件。

    既有 validator 仍讀全部位置、事件、速度、checksum 與 metadata；後續只將位置
    陣列送往繪圖。完整 RunUnit 身分／seed 順序另與計畫重建值比對，拒絕跨片替換。
    """
    row = progress["shards"][shard.shard_id]
    if row["lifecycle"] != "COMPLETE":
        raise ValueError("只接受 COMPLETE 輸出")
    output = _relative_file(root, row["output_relative_path"], label="selected output", directory=True)
    if output != root / plan["output_root"] / shard.shard_id:
        raise ValueError("輸出位置未繫結所選分片")
    result = validate_trajectory_shard(
        output,
        require_formal_metadata=plan["run_kind"] == "formal",
        strict_run_metadata=True,
        expected_metadata=_expected_output_metadata(plan, shard),
    )
    if result.get("valid") is not True:
        raise ValueError(f"所選完整輸出驗證失敗：{result.get('errors')}")
    manifest = result["manifest"]
    if manifest["schema_version"] != "3.0.0":
        raise ValueError("本展示要求含已保存環境及速度的schema3")
    rows = pq.read_table(output / "particle_table.parquet").to_pylist()
    units = list(
        iter_run_units(shard, master_seed=plan["master_seed"], random_stream_id=plan.get("random_stream_id"))
    )
    for row, unit in zip(rows, units, strict=True):
        expected = dict(
            particle_id=unit.particle_id,
            scenario_id=unit.scenario.scenario_id,
            member_id=unit.member_id,
            study_site_id=unit.scenario.study_site_id,
            analysis_region_id=unit.scenario.analysis_region_id,
            receptor_id=unit.scenario.receptor_id,
        )
        if any(row[key] != value for key, value in expected.items()):
            raise ValueError("RunUnit 身分／順序不符")
    return output, manifest, rows, units


def panel_counts(particles):
    """由每粒子唯一紀錄重算五水平／四抽樣序位與終止分母，非物理水層統計。"""
    if len({p["particle_id"] for p in particles}) != len(particles):
        raise ValueError("粒子身分重複")
    horizontal = Counter(p["horizontal_id"] for p in particles)
    vertical = Counter(p["vertical_rank"] for p in particles)
    cells = Counter((p["horizontal_id"], p["vertical_rank"]) for p in particles)
    if len(particles) != 20 or len(horizontal) != 5 or set(horizontal.values()) != {4}:
        raise ValueError("水平受體分母不符")
    if set(vertical) != {0, 1, 2, 3} or set(vertical.values()) != {5} or set(cells.values()) != {1}:
        raise ValueError("垂向抽樣分母不符")
    return {
        "particle_count": 20,
        "curve_count": 20,
        "terminal_denominator": 20,
        "horizontal": dict(horizontal),
        "vertical": dict(vertical),
        "terminal": dict(Counter(p["final_status"] for p in particles)),
        "terminal_by_vertical": {
            str(rank): dict(Counter(p["final_status"] for p in particles if p["vertical_rank"] == rank))
            for rank in range(4)
        },
    }


def select_current_site(plan, scenarios, shards, progress, receptors, requested_ids, site):
    """重用100顆操作入口的事前選樣，僅在該站完整兩片已完成時取圖。

    不依完成先後重選時間或受體；其他四站仍可運算，無須等全run完成。
    """
    path = Path(__file__).with_name("run_current100_subset.py")
    spec = importlib.util.spec_from_file_location("current100_selection", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    selected = module.select_shards(plan, [asdict(s) for s in scenarios])
    if site not in selected or len(requested_ids) != 2 or set(requested_ids) != set(selected[site]):
        raise ValueError("必須指定固定純流站點的完整兩片")
    values = [s for sh in shards if sh.shard_id in requested_ids for s in sh.scenarios]
    expected = {r.receptor_id for r in receptors if r.study_site_id == site}
    if (len(values) != 20 or len(expected) != 20
            or {s.receptor_id for s in values} != expected
            or any(progress["shards"][sid]["lifecycle"] != "COMPLETE" for sid in requested_ids)):
        raise ValueError("該站尚未完整完成20受體")
    return {site: values}


def collect_subset(
    run: Path, config_path: Path, coastline: Path, shard_ids: list[str], *, current100_site=None
):
    """建立小型、可重繪的來源快照，不建立 forcing，也不宣稱全批驗收。

    全run只驗 immutable SHA、計畫／進度與靜態情境／seed繫結；只對指定完成分片進行
    完整 output validator。原始檔案前後 hash 必須一致；可變 progress 只核對所選列
    未變，不阻止其他 worker 推進未選片。缺失環境值以 null 保存，圖面保留斷線。
    """
    plan = load_run_plan(run)
    progress = load_run_progress(run)
    _cross_check_plan_progress(plan, progress)
    if (
        plan["run_kind"] != "formal"
        or plan["scenario_selection"]["mode"] != "full"
        or plan["members_per_scenario"] != 1
        or plan["scenario_count"] != 50000
        or plan["experiment_case_id"] != ("no_stokes" if current100_site else "finite_depth_stokes")
    ):
        raise ValueError("本展示只接受50,000情境formal/full/M1及明示的純流／有限水深波浪案例")
    initial_paths = [run / "run_plan.json", config_path, coastline, *(run / name for name in plan["files"])]
    before = _snapshot(initial_paths)
    errors = []
    _validate_run_files(run, plan, errors)
    scenarios = _load_scenarios(run, plan, errors)
    if errors:
        raise ValueError(errors)
    shards = _build_shards(plan, scenarios)
    _validate_seed_table(run, plan, shards, errors)
    if errors:
        raise ValueError(errors)
    config = load_config(config_path, formal_release=False)
    if config.config_hash() != plan["config_hash"]:
        raise ValueError("設定hash不符")
    component_refs = {
        "material": config.scenarios.material_manifest,
        "receptor": config.scenarios.receptor_manifest,
        "arrival": config.scenarios.arrival_time_manifest,
        "receptor_arrival_initial_condition": config.scenarios.receptor_arrival_initial_condition_manifest,
        "domain": config.geometry["domain_manifest"],
        "local": config.geometry["local_domain_manifest"],
        "open_boundary": config.geometry["open_boundary_manifest"],
    }
    component_paths = {key: resolve_manifest_path(config_path, ref) for key, ref in component_refs.items()}
    before.update(_snapshot(component_paths.values()))
    # 必須在既有loader讀取前固定岸界及間接來源，避免只驗三份外框而漏記物理岸界。
    native_coast = _native_coast_snapshot(config, config_path, before)
    # generated 幾何以工程讀取模式取得；以原 plan hashes 與正式 flow ID 再繫結。
    # 不升級 release_status，不經 pilot loader，不替換 scenario_selection。
    inputs = load_scenario_inputs(
        config, config_path=config_path, require_dynamic_initial_conditions=True, formal=False
    )
    geometry = load_boundary_geometries(config, config_path=config_path, formal=False)
    if (
        dict(inputs.canonical_component_hashes) != plan["component_canonical_hashes"]
        or dict(geometry.canonical_component_hashes) != plan["geometry_canonical_hashes"]
    ):
        raise ValueError("當前輸入／幾何canonical hash不符")
    if native_coast is not None:
        native_coast["canonical_sha256"] = geometry.canonical_component_hashes["native_coast"]
    expected_scenarios = {s.scenario_id: s for s in inputs.scenarios}
    if len(expected_scenarios) != len(scenarios) or any(
        expected_scenarios.get(s.scenario_id) != s for s in scenarios
    ):
        raise ValueError("當前manifest重建情境與run不符")
    groups = (
        select_current_site(plan, scenarios, shards, progress, inputs.receptors, shard_ids, current100_site)
        if current100_site else
        select_completed_groups(scenarios, shards, progress, inputs.receptors, shard_ids)
    )
    if native_coast is not None and (
        not current100_site or not set(groups) <= {"nanwan", "lienchiang"}
    ):
        raise ValueError("模型岸界展示僅接受此次南灣／連江同受體20顆重算")
    if config.boundaries.max_backtrack_days != 30 or any(
        s.material_id != "oca_aluminum_crushed_cylinder" or s.settling_velocity_mps != -0.02
        for values in groups.values()
        for s in values
    ):
        raise ValueError("圖面固定標籤只適用本次30天與指定鋁材沉降速度")
    coast = _json(coastline)
    if (
        coast.get("type") != "FeatureCollection"
        or coast.get("crs", {}).get("properties", {}).get("name") != "urn:ogc:def:crs:OGC:1.3:CRS84"
    ):
        raise ValueError("海岸需CRS84經緯度FeatureCollection")
    lands = [shape(f["geometry"]) for f in coast["features"]]
    if any(g.is_empty or not g.is_valid or g.geom_type not in {"Polygon", "MultiPolygon"} for g in lands):
        raise ValueError("無效海岸幾何")
    receptors = {r.receptor_id: r for r in inputs.receptors}
    arrivals = {a.arrival_time_id: a for a in inputs.arrival_times}
    source_shards, particles = [], defaultdict(list)
    for shard in shards:
        if shard.shard_id not in shard_ids:
            continue
        output = run / plan["output_root"] / shard.shard_id
        # 僅列固定所選目錄；validator 會拒絕未知檔案，這裡不跟隨任何 symlink。
        before.update(_snapshot(output.iterdir()))
        output, manifest, rows, units = validate_selected_output(run, plan, progress, shard)
        arrays = {name: np.load(output / f"{name}.npy", allow_pickle=False) for name in POSITION_FIELDS}
        offsets = np.load(output / "trajectory_offsets.npy", allow_pickle=False)
        for index, (row, unit) in enumerate(zip(rows, units, strict=True)):
            scenario = unit.scenario
            receptor = receptors[scenario.receptor_id]
            arrival = arrivals[scenario.arrival_time_id]
            pair = inputs.initial_conditions_by_pair[(receptor.receptor_id, arrival.arrival_time_id)]
            region = next(d for d in config.domains if d.analysis_region_id == scenario.analysis_region_id)
            if pair.flow_domain_id != resolve_flow_domain_id(config, region.analysis_region_id, formal=True):
                raise ValueError("正式flow域繫結不符")
            residence = config.scenarios.bed_residence_time
            if residence is None or residence.backtrack_mode != "fixed_calendar_window":
                raise ValueError("本展示要求固定日曆窗")
            timing = resolve_bed_residence_timing(
                arrival,
                backtrack_mode=residence.backtrack_mode,
                requested_horizon_days=config.boundaries.max_backtrack_days,
                maximum_age_days=residence.maximum_age_days,
                sampling_seed=residence.sampling_seed,
            )
            start, stop = int(offsets[index]), int(offsets[index + 1])
            positions = {
                name: [
                    None if isinstance(v, float) and not np.isfinite(v) else v
                    for v in array[start:stop].tolist()
                ]
                for name, array in arrays.items()
            }
            x0, y0 = geometry.projections[scenario.study_site_id].project(receptor.lon, receptor.lat)
            if (
                positions["time_utc_ns"][0] != scenario.arrival_time_utc_ns
                or positions["age_seconds"][0] != 0
                or not np.allclose(
                    [positions["x_m"][0], positions["y_m"][0], positions["z_m"][0]],
                    [x0, y0, pair.z_m_positive_up],
                    rtol=0,
                    atol=0.05,
                )
            ):
                raise ValueError("初始位置／時間不符當前動態受體")
            meta = receptor.metadata
            if meta.get("vertical_identity_semantics") != "random_draw_rank_not_physical_layer":
                raise ValueError("本圖例要求隨機抽樣序位，不假設固定水層")
            particles[scenario.study_site_id].append(
                {
                    **row,
                    "scenario": asdict(scenario),
                    "seed_128_hex": f"{unit.seed:032x}",
                    "horizontal_id": meta["horizontal_receptor_id"],
                    "horizontal_rank": meta["horizontal_selection_draw_order"],
                    "vertical_id": receptor.vertical_id,
                    "vertical_rank": meta["vertical_selection_draw_order"],
                    "receptor": asdict(receptor),
                    "initial_condition": asdict(pair),
                    "arrival": asdict(arrival),
                    "timing": asdict(timing),
                    "positions": positions,
                    "actual_duration_seconds": positions["age_seconds"][-1],
                    "saved_position_count": stop - start,
                }
            )
        source_shards.append(
            {
                "shard_id": shard.shard_id,
                "manifest_sha256": before[str(output / "manifest.json")],
                "particle_count": manifest["particle_count"],
                "observation_count": manifest["observation_count"],
                "event_count": manifest["event_count"],
                "manifest": manifest,
                "progress": progress["shards"][shard.shard_id],
            }
        )
        del arrays, rows, units
        gc.collect()
    domain_doc = _json(component_paths["domain"])
    sites = {}
    for site in groups:
        records = particles[site]
        counts = panel_counts(records)
        domain = next(d for d in config.domains if d.analysis_region_id == records[0]["analysis_region_id"])
        record = next(
            r for r in domain_doc["records"] if r["analysis_region_id"] == domain.analysis_region_id
        )
        if record["flow_domain_id"] != resolve_flow_domain_id(config, domain.analysis_region_id, formal=True):
            raise ValueError("圖面域框與正式flow不符")
        polygon = shape(record["geometry"])
        sites[site] = {
            "study_site_name_zh": next(
                s.study_site_name_zh for s in config.study_sites if s.study_site_id == site
            ),
            "analysis_region_id": records[0]["analysis_region_id"],
            "particles": records,
            "counts": counts,
            "center_lonlat": list(domain.center_lonlat),
            "domain": record["geometry"],
            "flow_domain_id": record["flow_domain_id"],
            "coastline": [
                mapping(g.intersection(polygon.envelope.buffer(0.15)))
                for g in lands
                if g.intersects(polygon.envelope.buffer(0.15))
            ],
            "geometry_release_status": domain_doc.get("release_status", domain_doc.get("status")),
            "domain_source_metadata": {k: v for k, v in domain_doc.items() if k != "records"},
        }
    after = _unchanged_snapshot(before, "製圖擷取期間來源改變，拒絕發布")
    current = load_run_progress(run)
    if any(current["shards"][sid] != progress["shards"][sid] for sid in shard_ids):
        raise ValueError("製圖擷取期間來源改變，拒絕發布")
    return {
        "artifact_kind": "completed-subset-preview-v1",
        "validation_scope": "selected_complete_outputs_only",
        "full_validation_deferred": True,
        "scope_note": NATIVE_COAST_SCOPE_NOTE if native_coast is not None else SCOPE_NOTE,
        **({
            "native_coast": native_coast,
            "population_role": "replacement_same_receptors_of_original100_not_additional40",
            "model_plan_commit": plan["code_commit"],
            "plot_code": _plot_code_provenance(),
        } if native_coast is not None else {}),
        "selection_rule": (
            "事前固定五站各最早指定鋁材20受體，僅擷取該站完整兩片；不同站不同UTC"
            if current100_site else SELECTION_RULE
        ),
        "run_id": plan["run_id"],
        "run_kind": plan["run_kind"],
        "scenario_selection": plan["scenario_selection"],
        "total_run_scenarios": plan["scenario_count"],
        "selected_particles": sum(len(values) for values in groups.values()),
        "members_per_scenario": 1,
        "experiment_case_id": plan["experiment_case_id"],
        "config_hash": plan["config_hash"],
        "plan": plan,
        "source_shards": source_shards,
        "sites": sites,
        "source_sha256_before": before,
        "source_sha256_after": after,
        "source_unchanged": True,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }


def _renderer(project_root: Path):
    """只載入既有純配色／字型／排他存圖 helpers；不呼叫 pilot loader 或發布 gate。"""
    path = project_root / "scripts" / "build_pilot_coastline_preview.py"
    spec = importlib.util.spec_from_file_location("completed_subset_style", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _utc(ns):
    """將原始UTC奈秒轉為圖面分鐘文字，不改變來源時間精度。"""
    return datetime.fromtimestamp(ns / 1e9, UTC).strftime("%Y-%m-%d %H:%M")


def render_subset(payload, output: Path, project_root: Path):
    """由已驗證小型快照每站產生四張PNG，保留每個保存位置與缺值斷線。

    重用 BayTrace 的配色、中文字型及排他存圖；另立標題與時間軸，因新設計的四條
    顏色是隨機垂向抽樣序位，不是舊固定物理水層。局部圖使用同一AEQD公尺座標與
    等比例軸，深度以向上正的原始z顯示，海面／床僅使用保存context，無forcing取樣。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.text import Text
    from shapely.ops import transform

    from lagrangian_backtracking.geometry import DomainProjection

    renderer = _renderer(project_root)
    colors = list(renderer._BAYTRACE_VERTICAL_COLORS.values())
    statuses = renderer._TERMINAL_STATUSES
    labels = {**renderer._BAYTRACE_STATUS_LABELS, "max_age": "達個別回溯上限"}
    font = renderer._display_font()
    files = {}

    def save(fig, name, directory):
        """统一中文字型並排他儲存；既有成果永不覆寫。"""
        for text in fig.findobj(Text):
            size = text.get_fontsize()
            text.set_fontproperties(font)
            text.set_fontsize(size)
        path = directory / name
        renderer._save_baytrace_figure(fig, path)
        plt.close(fig)
        files[str(path.relative_to(output))] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }

    def draw_land(ax, geoms, project=None, origin=(0, 0)):
        """海岸只當地理背景；保留多邊形孔洞，不以海岸重新分類粒子。"""
        for geom in geoms:
            if project:
                geom = transform(project, geom)
            polys = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
            for poly in polys:
                if poly.geom_type != "Polygon":
                    continue
                xy = np.asarray(poly.exterior.coords)
                ax.fill(xy[:, 0] - origin[0], xy[:, 1] - origin[1], color="#d3d4d4", zorder=0)
                for ring in poly.interiors:
                    xy = np.asarray(ring.coords)
                    ax.fill(xy[:, 0] - origin[0], xy[:, 1] - origin[1], color="#edf6fa", zorder=0)

    for site, data in payload["sites"].items():
        directory = output / site
        directory.mkdir()
        particles = data["particles"]
        counts = panel_counts(particles)
        if counts != data["counts"] and json.dumps(counts, sort_keys=True) != json.dumps(
            data["counts"], sort_keys=True
        ):
            raise ValueError("subplot分母與來源不符")
        if any(p["final_status"] not in statuses for p in particles):
            raise ValueError("未知終止狀態")
        projection = DomainProjection(*data["center_lonlat"])
        domain = shape(data["domain"])
        lands = [shape(g) for g in data["coastline"]]
        first = particles[0]
        arrival = first["arrival"]
        age_h = arrival["metadata"]["bed_residence_age_hours"]
        horizons = [p["timing"]["effective_horizon_seconds"] / 3600 for p in particles]
        durations = [p["actual_duration_seconds"] / 3600 for p in particles]
        site_label = data.get("study_site_name_zh", site)
        heading = f"{data.get('analysis_region_id', '')}區{site_label}｜20個已完成沉降粒子"
        wave_label = "純流，波浪關閉" if payload["experiment_case_id"] == "no_stokes" else "有限水深波浪開啟"
        meta = (
            f"沉底起點 {_utc(arrival['time_utc_ns'])} UTC；"
            f"觀測 {_utc(arrival['metadata']['observation_time_utc_ns'])} UTC\n"
            f"固定日曆窗30天；沉底停留 {age_h} 小時；"
            f"個別可回溯上限 {min(horizons):g}–{max(horizons):g} 小時；"
            f"實際保存軌跡 {min(durations):.2f}–{max(durations):.2f} 小時\n"
            f"oca_aluminum_crushed_cylinder｜沉降速度 −0.02 m/s｜{wave_label}｜5水平位置×4垂向抽樣序位"
        )
        layer_handles = [
            Line2D([], [], color=colors[i], lw=2, label=f"垂向抽樣序位{i + 1}（5粒子）") for i in range(4)
        ]
        endpoint_handles = [
            Line2D([], [], color="#1a9850", marker="o", linestyle="", label="回溯起點"),
            Line2D([], [], color="#d73027", marker="x", linestyle="", label="停止位置（非來源認定）"),
        ]

        def title(fig, subtitle, heading=heading, meta=meta):
            """所有圖面均帶時間語境與部分完成警語，避免圖脫離manifest後誤讀。"""
            fig.suptitle(heading + "\n" + subtitle, fontsize=18, y=0.99)
            fig.text(0.5, 0.895, meta, ha="center", va="top", fontsize=10)
            fig.text(
                0.5,
                0.018,
                payload["scope_note"] + "｜幾何資格照原始記錄，本圖非全批正式驗收",
                ha="center",
                fontsize=11,
                color="#8b2424",
            )

        fig, ax = plt.subplots(figsize=(14, 11))
        fig.subplots_adjust(top=0.77, bottom=0.16, left=0.10, right=0.94)
        title(fig, "水平軌跡總覽｜原始保存位置，無補值／延長／平滑")
        ax.set_facecolor("#edf6fa")
        draw_land(ax, lands)
        dx, dy = domain.exterior.xy
        ax.plot(dx, dy, color="#555555", linestyle="--", lw=1.2)
        for p in particles:
            pos = p["positions"]
            lon, lat = projection.unproject(pos["x_m"], pos["y_m"])
            ax.plot(lon, lat, color=colors[p["vertical_rank"]], lw=1.1, alpha=0.85)
            ax.scatter(lon[0], lat[0], color="#1a9850", s=20, zorder=4)
            ax.scatter(lon[-1], lat[-1], color="#d73027", marker="x", s=30, zorder=4)
        minx, miny, maxx, maxy = domain.bounds
        ax.set(
            xlim=(minx - 0.035, maxx + 0.035),
            ylim=(miny - 0.035, maxy + 0.035),
            xlabel="經度（°E）",
            ylabel="緯度（°N）",
        )
        ax.set_aspect(1 / np.cos(np.deg2rad(data["center_lonlat"][1])))
        ax.grid(alpha=0.22)
        fig.legend(
            handles=layer_handles + endpoint_handles, loc="lower center", bbox_to_anchor=(0.5, 0.052), ncol=3
        )
        save(fig, "horizontal_overview.png", directory)

        fig, axes = plt.subplots(2, 3, figsize=(16, 11))
        fig.subplots_adjust(top=0.76, bottom=0.10, hspace=0.38, wspace=0.30)
        title(fig, "各水平受體局部軌跡｜AEQD等比例公尺；每格4粒子／4曲線")
        horizontal_ids = sorted(
            counts["horizontal"],
            key=lambda key: next(p["horizontal_rank"] for p in particles if p["horizontal_id"] == key),
        )
        for ax, hid in zip(axes.flat, horizontal_ids, strict=False):
            group = [p for p in particles if p["horizontal_id"] == hid]
            origin = (group[0]["positions"]["x_m"][0], group[0]["positions"]["y_m"][0])
            extent = (
                max(
                    max(abs(v - origin[j]) for v in p["positions"][field])
                    for p in group
                    for j, field in enumerate(("x_m", "y_m"))
                )
                * 1.12
            )
            extent = max(extent, 100)
            ax.set_facecolor("#edf6fa")
            draw_land(ax, lands, projection.project, origin)
            for p in group:
                pos = p["positions"]
                x, y = np.asarray(pos["x_m"]) - origin[0], np.asarray(pos["y_m"]) - origin[1]
                ax.plot(x, y, color=colors[p["vertical_rank"]], lw=1)
                ax.scatter(x[-1], y[-1], color="#d73027", marker="x", s=25)
            ax.scatter(0, 0, color="#1a9850", s=30)
            ax.set(
                xlim=(-extent, extent),
                ylim=(-extent, extent),
                xlabel="相對起點東向（m）",
                ylabel="相對起點北向（m）",
                title=f"水平抽樣位置 {group[0]['horizontal_rank'] + 1}｜n=4",
            )
            ax.set_aspect("equal", adjustable="box")
            ax.ticklabel_format(style="sci", axis="both", scilimits=(-3, 4))
            ax.grid(alpha=0.2)
        axes.flat[-1].axis("off")
        axes.flat[-1].legend(handles=layer_handles + endpoint_handles, loc="center")
        save(fig, "horizontal_local.png", directory)

        fig, axes = plt.subplots(4, 1, figsize=(15, 14), sharex=True)
        fig.subplots_adjust(top=0.78, bottom=0.11, hspace=0.33, left=0.12, right=0.96)
        title(fig, "垂向軌跡｜抽樣序位不是固定水層；虛線海面、點線海床為已保存環境")
        for rank, ax in enumerate(axes):
            for p in particles:
                if p["vertical_rank"] != rank:
                    continue
                pos = p["positions"]
                hours = np.asarray(pos["age_seconds"]) / 3600
                ax.plot(hours, pos["z_m"], color=colors[rank], lw=1.1)
                ax.plot(
                    hours,
                    np.asarray(pos["eta_m"], dtype=float),
                    color="#617984",
                    linestyle="--",
                    lw=0.7,
                    alpha=0.55,
                )
                ax.plot(
                    hours,
                    np.asarray(pos["bed_z_m"], dtype=float),
                    color="#896743",
                    linestyle=":",
                    lw=0.8,
                    alpha=0.65,
                )
                ax.scatter(hours[0], pos["z_m"][0], color="#1a9850", s=18)
                ax.scatter(hours[-1], pos["z_m"][-1], color="#d73027", marker="x", s=22)
            ax.set(
                title=f"垂向抽樣序位 {rank + 1}｜5粒子／5曲線",
                ylabel="z（m，向上為正）",
                xlim=(0, max(durations) * 1.03),
            )
            ax.grid(alpha=0.2)
        axes[-1].set_xlabel("自沉底起點向前回溯年齡（小時）；線長是實際保存期間，不代表全部追完30天")
        save(fig, "depth_age.png", directory)

        fig, ax = plt.subplots(figsize=(15, 9))
        fig.subplots_adjust(top=0.73, bottom=0.25, left=0.20, right=0.95)
        title(fig, "停止狀態｜全部20粒子為分母；完成分片不等於完成個別時間上限")
        left = np.zeros(4)
        handles = []
        for status in statuses:
            n = [counts["terminal_by_vertical"][str(rank)].get(status, 0) for rank in range(4)]
            color = renderer._BAYTRACE_STATUS_COLORS[status]
            ax.barh(range(4), n, left=left, color=color, height=0.55)
            for rank, count in enumerate(n):
                if count:
                    ax.text(left[rank] + count / 2, rank, str(count), ha="center", va="center", fontsize=13)
            left += n
            handles.append(Line2D([], [], color=color, lw=8, label=f"{labels[status]}：{sum(n)}/20"))
        ax.set(
            yticks=range(4),
            yticklabels=[f"垂向抽樣序位 {i + 1}（n=5）" for i in range(4)],
            xticks=range(6),
            xlim=(0, 5.3),
            xlabel="粒子數（不是機率）；各列分母5，全圖分母20",
        )
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.2)
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.06), ncol=4)
        save(fig, "terminal_counts.png", directory)
    return files


def main() -> int:
    """命令列驗證指定完成資料並排他建立新目錄；不支援重算或正式整批發布。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--coastline", type=Path, required=True)
    parser.add_argument("--shard", action="append", required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--current100-site", choices=("gongliao", "guishan", "hsinchu", "lienchiang", "nanwan")
    )
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("拒絕覆寫既有成果")
    payload = collect_subset(
        args.run, args.config, args.coastline, args.shard, current100_site=args.current100_site
    )
    args.output.mkdir(parents=True, exist_ok=False)
    _write_json(args.output / "render_inputs.json", payload)
    files = render_subset(payload, args.output, args.project_root)
    # PNG完成後再核對來源，發布manifest才算本次部分展示完成；不檢查未選checkpoints。
    _unchanged_snapshot(payload["source_sha256_before"], "渲染期間來源改變，拒絕發布manifest")
    summary = {key: value for key, value in payload.items() if key not in {"sites", "plan"}}
    summary["sites"] = {
        site: {key: value for key, value in data.items() if key not in {"particles", "coastline"}}
        for site, data in payload["sites"].items()
    }
    summary["files"] = {
        **files,
        "render_inputs.json": {"sha256": sha256_file(args.output / "render_inputs.json")},
    }
    summary["script_sha256"] = sha256_file(Path(__file__))
    _write_json(args.output / "manifest.json", summary)
    with (args.output / "README.md").open("x", encoding="utf-8") as stream:
        stream.write(
            "# M1 部分完成情境展示\n\n" + payload["scope_note"] + "。\n\n"
            "來源維持 formal/full；只完整驗證所選 COMPLETE output，未做整批 checkpoint／歷史驗收。\n\n"
            + payload["selection_rule"]
            + f"。各站20受體（5水平×4隨機垂向抽樣），本圖{payload['selected_particles']}"
            + ("顆；原run計畫母體50,000情境不代表已執行或已驗收50,000顆。\n\n"
               if "native_coast" in payload else "／全run50,000情境。\n\n")
            + f"案例：{payload['experiment_case_id']}；不同站到達時間各自明示，不宣稱同時配對。\n"
            "顏色是抽樣序位，不是固定物理水層；M=1僅一成員。\n"
            "固定日曆窗以觀測錨點向前30天；扣除沉底停留後得到個別上限，實際線長由停止狀態決定。\n"
            "只連接所有保存位置，無補值、延長、平滑。海面／海床缺值保留斷線，不讀forcing。\n"
            "幾何工程生成狀態照原始metadata保存，不提升為approved。停止位置不代表污染來源。\n\n"
            "render_inputs.json 保存所有繪圖位置、受體／時間／seed身分與小型來源快照；"
            "manifest.json保存所選分片checksum、前後來源SHA與每張圖checksum。\n"
        )
        if "native_coast" in payload:
            stream.write(
                "\n本次南灣／連江是原100同受體重算替代，不是額外40顆；A與B版本分開，"
                "每站終止分母固定20，不能相加。模型岸界只覆蓋已審兩component，"
                "不等於底圖岸線；底圖僅顯示，不作物理邊界。\n"
                "native_coast欄位保存岸界檔案／canonical／來源SHA與兩component；"
                "model_plan_commit為原計畫模型來源，plot_code為實際繪圖commit與腳本SHA。\n"
            )
    print(json.dumps({"output": str(args.output), "png_count": len(files), "source_unchanged": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
