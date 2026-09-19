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
