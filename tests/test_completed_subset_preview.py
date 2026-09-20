"""部分完成成果圖的身分／分母／完整輸出拒絕測試；純合成資料不代表科學驗收。"""

from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_run_control import _first_shard, _request, _workspace

from lagrangian_backtracking.outputs import sha256_file
from lagrangian_backtracking.run_control import RunController, load_run_plan, load_run_progress
from lagrangian_backtracking.run_validation import _build_shards, _load_scenarios

SCRIPT = Path(__file__).parents[1] / "scripts" / "build_completed_subset_preview.py"
spec = importlib.util.spec_from_file_location("subset_preview", SCRIPT)
preview = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preview)


def _groups():
    """以明示身分建立兩站各20受體；ID刻意不含可推測的水層／座標字串。"""
    scenarios, shards, receptors = [], [], []
    for site in ("gongliao", "guishan"):
        values = [
            SimpleNamespace(
                scenario_id=f"{site}-{i}",
                study_site_id=site,
                arrival_time_id="arrival",
                arrival_time_utc_ns=1,
                material_id="material",
                receptor_id=f"opaque-{site}-{i}",
            )
            for i in range(20)
        ]
        scenarios.extend(values)
        receptors.extend(SimpleNamespace(receptor_id=s.receptor_id, study_site_id=site) for s in values)
        for start in (0, 10):
            shards.append(
                SimpleNamespace(
                    shard_id=f"{site}-{start}", scenarios=values[start : start + 10], particle_count=10
                )
            )
    progress = {"shards": {sh.shard_id: {"lifecycle": "COMPLETE"} for sh in shards}}
    return scenarios, shards, progress, receptors, [sh.shard_id for sh in shards]


def test_fixed_selected_groups_are_read_only():
    """固定4片回傳兩站各20，呼叫後不更動來源記憶體表。"""
    values = _groups()
    before = deepcopy(values)
    result = preview.select_completed_groups(*values)
    assert {key: len(value) for key, value in result.items()} == {"gongliao": 20, "guishan": 20}
    assert values == before


def test_current_site_requires_exact_preselected_completed_pair(monkeypatch):
    """純流逐站擷取仍要求事前固定兩片；不以其他已完成片替代。"""
    from dataclasses import make_dataclass

    import test_current100_operator as current_tests

    plan, rows = current_tests.fixture_plan()
    keys = list(rows[0])
    record = make_dataclass("PreviewScenario", [(key, object) for key in keys])
    scenarios = [record(**{key: row.get(key) for key in keys}) for row in rows]
    shards = [SimpleNamespace(shard_id=s["shard_id"],
                              scenarios=scenarios[s["scenario_start_index"]:s["scenario_stop_index"]])
              for s in plan["shards"]]
    receptors = [SimpleNamespace(receptor_id=s.receptor_id, study_site_id=s.study_site_id)
                 for s in scenarios[:100]]
    progress = {"shards": {s.shard_id: {"lifecycle": "COMPLETE"} for s in shards}}
    selected = preview.select_current_site(plan, scenarios, shards, progress, receptors,
                                           ["sh-8", "sh-9"], "nanwan")
    assert len(selected["nanwan"]) == 20
    progress["shards"]["sh-9"]["lifecycle"] = "RUNNING"
    with pytest.raises(ValueError):
        preview.select_current_site(plan, scenarios, shards, progress, receptors,
                                    ["sh-8", "sh-9"], "nanwan")


@pytest.mark.parametrize(
    "damage", ["incomplete", "mixed_site", "mixed_material", "mixed_arrival", "duplicate", "unknown"]
)
def test_selection_rejects_incomplete_or_mixed_identity(damage):
    """拒絕未完成、混站／材質／到達、重複受體与未知分片，不按軌跡外觀補選。"""
    values = _groups()
    scenarios, shards, progress, receptors, ids = values
    if damage == "incomplete":
        progress["shards"][ids[0]]["lifecycle"] = "RUNNING"
    elif damage == "mixed_site":
        scenarios[0].study_site_id = "other"
    elif damage == "mixed_material":
        scenarios[0].material_id = "other"
    elif damage == "mixed_arrival":
        scenarios[0].arrival_time_id = "other"
    elif damage == "duplicate":
        scenarios[0].receptor_id = scenarios[1].receptor_id
    else:
        ids[0] = "unknown"
    with pytest.raises(ValueError):
        preview.select_completed_groups(*values)


def _particles():
    """五水平×四抽樣序位；每種停止狀態使用明確單粒子分母。"""
    return [
        {
            "particle_id": str(i),
            "horizontal_id": str(i // 4),
            "vertical_rank": i % 4,
            "final_status": "data_gap" if i < 9 else "flow_domain_open_exit",
        }
        for i in range(20)
    ]


def test_subplot_and_status_denominators():
    """全圖20、水平格4、垂向格5與停止9+11可由唯一粒子紀錄重算。"""
    counts = preview.panel_counts(_particles())
    assert counts["terminal"] == {"data_gap": 9, "flow_domain_open_exit": 11}
    assert counts["curve_count"] == counts["terminal_denominator"] == 20
    assert set(counts["horizontal"].values()) == {4}
    assert set(counts["vertical"].values()) == {5}
    assert all(sum(row.values()) == 5 for row in counts["terminal_by_vertical"].values())


@pytest.mark.parametrize("damage", ["missing", "duplicate", "horizontal", "vertical", "cell"])
def test_bad_subplot_denominators_rejected(damage):
    """不能以重複曲線湊足20，也不能把抽樣序位誤分組後仍顯示原分母。"""
    particles = _particles()
    if damage == "missing":
        particles.pop()
    elif damage == "duplicate":
        particles[0]["particle_id"] = particles[1]["particle_id"]
    elif damage == "horizontal":
        particles[0]["horizontal_id"] = "unknown"
    elif damage == "vertical":
        particles[0]["vertical_rank"] = 5
    else:
        particles[0]["vertical_rank"], particles[5]["vertical_rank"] = 1, 0
    with pytest.raises(ValueError):
        preview.panel_counts(particles)


@pytest.mark.parametrize("damage", [None, "checksum", "identity", "metadata", "incomplete", "symlink"])
def test_selected_output_full_validator_and_no_source_write(tmp_path, damage):
    """實際writer產出的片仍完整驗checksum/metadata/RunUnit；拒絕時不修復或覆寫。"""
    root = _workspace(tmp_path, "subset-output", run_kind="formal")
    sid = _first_shard(root)
    RunController(root, request_factory=_request).run_shard(sid)
    plan, progress = load_run_plan(root), load_run_progress(root)
    errors = []
    scenarios = _load_scenarios(root, plan, errors)
    shard = next(sh for sh in _build_shards(plan, scenarios) if sh.shard_id == sid)
    output = root / "shards" / sid
    if damage == "checksum":
        (output / "x_m.npy").write_bytes(b"broken")
    elif damage == "identity":
        path = output / "particle_table.parquet"
        table = pq.read_table(path)
        rows = table.to_pylist()
        rows[0]["receptor_id"] = "wrong"
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"][path.name].update(sha256=sha256_file(path), size_bytes=path.stat().st_size)
        manifest_path.write_text(json.dumps(manifest))
    elif damage == "metadata":
        path = output / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["run_metadata"]["config_hash"] = "0" * 64
        path.write_text(json.dumps(manifest))
    elif damage == "incomplete":
        progress["shards"][sid]["lifecycle"] = "RUNNING"
    elif damage == "symlink":
        (output / "unexpected").symlink_to(root / "run_plan.json")
    before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    if damage:
        with pytest.raises(ValueError):
            preview.validate_selected_output(root, plan, progress, shard)
    else:
        _, manifest, rows, units = preview.validate_selected_output(root, plan, progress, shard)
        assert len(rows) == len(units) == manifest["particle_count"]
    assert {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()} == before


def test_json_and_snapshot_refuse_symlink_and_overwrite(tmp_path):
    """來源checksum不跟隨symlink；成果JSON排他建立避免覆寫舊圖旁的證據。"""
    path = tmp_path / "source.json"
    preview._write_json(path, {"value": 1})
    with pytest.raises(FileExistsError):
        preview._write_json(path, {"value": 2})
    alias = tmp_path / "alias"
    alias.symlink_to(path)
    with pytest.raises(ValueError):
        preview._snapshot([alias])
    assert preview._json(path) == {"value": 1}


def _coast_geometry_fixture(tmp_path, *, native=True):
    """重用實際五站幾何loader與既有合成岸界，不複製loader或採樣真實流場。

    config與岸界置於不同目錄，source使用相對岸界的路徑，專門排除誤用cwd／config
    解析間接來源。兩component只供來源契約回歸，不能當成真資料岸界驗收。
    """
    from test_manifests import _config, _geometry_payloads
    from test_native_coast import manifest_fixture

    config = _config(tmp_path, with_paths=True)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("synthetic fixture")
    # 情境解析在接線測試另有探針，這些普通檔僅讓既有來源SHA快照保持真實I/O。
    for reference in (config.scenarios.material_manifest, config.scenarios.receptor_manifest,
                      config.scenarios.arrival_time_manifest,
                      config.scenarios.receptor_arrival_initial_condition_manifest):
        preview.resolve_manifest_path(config_path, reference).write_text("{}")
    for name, document in zip(("domain.json", "local.json", "open.json"), _geometry_payloads(), strict=True):
        (tmp_path / name).write_text(json.dumps(document))
    geometry = preview.load_boundary_geometries(config, config_path=config_path, formal=False)
    if not native:
        return config, config_path, geometry, None
    artifact = tmp_path / "coast"
    artifact.mkdir()
    path, document, _ = manifest_fixture(artifact)
    document["source_files"][0]["path"] = "source.json"
    document["geometry_file_sha256"] = dict(geometry.file_sha256)
    record = document["records"][0]
    other = deepcopy(record)
    other.update(study_site_id="lienchiang", component_id="lienchiang-boundary-35")
    document["records"].append(other)
    for record in document["records"]:
        site = record["study_site_id"]
        record["flow_domain_id"] = next(
            s.flow_domain_id for s in config.study_sites if s.study_site_id == site
        )
        record["projection_wkt"] = geometry.projections[site]._forward.target_crs.to_wkt()
    path.write_text(json.dumps(document))
    config = config.model_copy(update={"geometry": dict(
        config.geometry, native_coast_manifest={"path": "coast/native.json", "sha256": sha256_file(path)}
    )})
    return config, config_path, geometry, path


@pytest.mark.parametrize("native", [False, True])
def test_geometry_snapshot_three_and_four_fields(tmp_path, monkeypatch, native):
    """實際loader的三／四份canonical完整保留，所有間接來源在不同cwd仍可追溯。"""
    config, config_path, old, path = _coast_geometry_fixture(tmp_path, native=native)
    elsewhere = tmp_path / "cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    before = preview._snapshot([config_path])
    evidence = preview._native_coast_snapshot(config, config_path, before)
    geometry = preview.load_boundary_geometries(config, config_path=config_path, formal=False)
    if native:
        assert set(geometry.canonical_component_hashes) == (
            set(old.canonical_component_hashes) | {"native_coast"}
        )
        assert {k: geometry.canonical_component_hashes[k] for k in old.canonical_component_hashes} == dict(
            old.canonical_component_hashes
        )
        assert evidence["manifest_sha256"] == sha256_file(path)
        assert evidence["source_files_sha256"] == {
            str(path.parent / "source.json"): sha256_file(path.parent / "source.json")
        }
        assert {r["component_id"] for r in evidence["components"]} == {
            "nanwan-boundary-0", "lienchiang-boundary-35"
        }
        assert evidence["basemap_role"] == "display_only_not_physical_boundary"
        assert set(before) == {str(config_path), str(path), str(path.parent / "source.json")}
    else:
        assert evidence is None and before == preview._snapshot([config_path])
        assert dict(geometry.canonical_component_hashes) == dict(old.canonical_component_hashes)
    assert preview._unchanged_snapshot(before, "擷取期間來源改變") == before


@pytest.mark.parametrize("fault", ["missing", "corrupt", "source_missing", "source_sha", "projection"])
def test_coast_preview_keeps_existing_loader_rejections(tmp_path, fault):
    """繪圖前來源擷取不能讓第四幾何缺失、損壞、投影或靜態來源錯綁通過。"""
    config, config_path, _, path = _coast_geometry_fixture(tmp_path)
    if fault == "missing":
        path.unlink()
    elif fault == "corrupt":
        path.write_text("{broken")
    elif fault == "source_missing":
        (path.parent / "source.json").unlink()
    elif fault == "source_sha":
        (path.parent / "source.json").write_text('{"changed":true}')
    else:
        document = json.loads(path.read_text())
        document["records"][0]["projection_wkt"] = "wrong projection"
        path.write_text(json.dumps(document))
        config.geometry["native_coast_manifest"]["sha256"] = sha256_file(path)
    with pytest.raises((ValueError, FileNotFoundError)):
        preview._native_coast_snapshot(config, config_path, {})
        preview.load_boundary_geometries(config, config_path=config_path, formal=False)


def test_collect_requires_full_four_geometry_dictionary(tmp_path, monkeypatch):
    """沿collect_subset真實接線到已審loader；計畫若刪第四canonical仍必須被拒絕。

    僅mock不相關情境／輸出I/O，成功走過幾何檢查即由選片探針停止，避免建立假軌跡。
    """
    config, config_path, _, coast = _coast_geometry_fixture(tmp_path)
    geometry = preview.load_boundary_geometries(config, config_path=config_path, formal=False)
    plan = dict(run_kind="formal", scenario_selection={"mode": "full"}, members_per_scenario=1,
                scenario_count=50000, experiment_case_id="no_stokes", files={},
                config_hash=config.config_hash(), component_canonical_hashes={"fixture": "same"},
                geometry_canonical_hashes=dict(geometry.canonical_component_hashes))
    (tmp_path / "run_plan.json").write_text(json.dumps(plan))
    monkeypatch.setattr(preview, "load_run_plan", lambda _: plan)
    monkeypatch.setattr(preview, "load_run_progress", lambda _: {"shards": {}})
    monkeypatch.setattr(preview, "_cross_check_plan_progress", lambda *a: None)
    monkeypatch.setattr(preview, "_validate_run_files", lambda *a: None)
    monkeypatch.setattr(preview, "_validate_seed_table", lambda *a: None)
    monkeypatch.setattr(preview, "_load_scenarios", lambda *a: [])
    monkeypatch.setattr(preview, "_build_shards", lambda *a: [])
    monkeypatch.setattr(preview, "load_config", lambda *a, **k: config)
    monkeypatch.setattr(preview, "load_scenario_inputs", lambda *a, **k: SimpleNamespace(
        canonical_component_hashes={"fixture": "same"}, scenarios=[], receptors=[]))

    def reached_selection(*args):
        """只有通過真實第四幾何驗證後才會到達此測試探針。"""
        raise RuntimeError("verified all four geometry keys")

    monkeypatch.setattr(preview, "select_current_site", reached_selection)
    with pytest.raises(RuntimeError, match="verified all four"):
        preview.collect_subset(tmp_path, config_path, coast, [], current100_site="nanwan")
    plan["geometry_canonical_hashes"].pop("native_coast")
    with pytest.raises(ValueError, match="canonical hash不符"):
        preview.collect_subset(tmp_path, config_path, coast, [], current100_site="nanwan")


@pytest.mark.parametrize("source", ["native.json", "source.json"])
def test_coast_snapshot_rejects_capture_time_mutation(tmp_path, source):
    """擷取期間改岸界檔或間接來源，即使普通檔仍存在也拒絕發布；不重新接受新SHA。"""
    config, config_path, _, path = _coast_geometry_fixture(tmp_path)
    before = {}
    preview._native_coast_snapshot(config, config_path, before)
    (path.parent / source).write_text('{"changed":true}')
    with pytest.raises(ValueError, match="擷取期間來源改變"):
        preview._unchanged_snapshot(before, "擷取期間來源改變")


@pytest.mark.parametrize("mutate_during_render", [False, True])
def test_main_native_scope_and_render_source_guard(tmp_path, monkeypatch, mutate_during_render):
    """同一來源清單在渲染後再驗；成功發布才含替代母體、20分母及分開的程式來源。

    合成renderer不畫科學圖，只驗四圖出版流程及字串契約；真圖仍待主審後部署。
    """
    import sys

    config, config_path, _, coast = _coast_geometry_fixture(tmp_path)
    before = {}
    native = preview._native_coast_snapshot(config, config_path, before)
    native["canonical_sha256"] = preview.load_boundary_geometries(
        config, config_path=config_path, formal=False).canonical_component_hashes["native_coast"]
    payload = dict(source_sha256_before=before, sites={}, plan={}, native_coast=native,
                   scope_note=preview.NATIVE_COAST_SCOPE_NOTE, selection_rule="事前固定20受體",
                   selected_particles=20, experiment_case_id="no_stokes", model_plan_commit="model-source",
                   plot_code=preview._plot_code_provenance())
    output = tmp_path / "preview"
    monkeypatch.setattr(preview, "collect_subset", lambda *a, **k: payload)

    def render(data, directory, root):
        """檢查交給四圖renderer的限制文字，選擇性改來源以驗證不發布manifest。"""
        assert data["scope_note"] == preview.NATIVE_COAST_SCOPE_NOTE
        if mutate_during_render:
            (coast.parent / "source.json").write_text("changed")
        return {f"figure{i}.png": {"sha256": "fixture"} for i in range(4)}

    monkeypatch.setattr(preview, "render_subset", render)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--run", str(tmp_path), "--config", str(config_path),
                        "--coastline", str(coast), "--shard", "one", "--shard", "two",
                        "--current100-site", "nanwan", "--project-root", str(SCRIPT.parents[1]),
                        "--output", str(output)])
    if mutate_during_render:
        with pytest.raises(ValueError, match="渲染期間來源改變"):
            preview.main()
        assert not (output / "manifest.json").exists()
    else:
        assert preview.main() == 0
        manifest = json.loads((output / "manifest.json").read_text())
        rendered = json.loads((output / "render_inputs.json").read_text())
        assert manifest["native_coast"] == rendered["native_coast"] == native
        assert manifest["selected_particles"] == 20
        assert manifest["model_plan_commit"] != manifest["plot_code"]["commit"]
        text = (output / "README.md").read_text()
        for phrase in ["原100同受體重算替代", "不是額外40", "每站終止分母固定20",
                       "底圖僅顯示", "不代表已執行"]:
            assert phrase in text
