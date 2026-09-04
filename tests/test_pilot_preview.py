"""獨立先導預覽的工程驗證，不以合成資料冒充 PI 真資料成果。

完整五萬來源由既有合成清單提供，但計畫、實際 engine 執行、檔案輸出、完整驗證與軌跡
逐片讀取均沿用正式公共路徑。此處不讀海洋 forcing，也不宣稱物理或觀測驗證通過。
"""

from __future__ import annotations

import csv
import importlib.util
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image
from shapely.geometry import box
from test_run_control import _request
from test_run_validation import _downgrade_shard_to_legacy
from test_runtime_static_inputs import (
    _initialize_exact_test_run,
    _patch_static_loaders,
    _snapshot_tree,
)
from test_runtime_static_inputs import (
    exact_runtime_data as exact_runtime_data,
)

from lagrangian_backtracking import pilot_preview as preview
from lagrangian_backtracking.boundaries import BoundaryGeometry
from lagrangian_backtracking.geometry import DomainProjection
from lagrangian_backtracking.models import SampleQC, VelocitySample
from lagrangian_backtracking.outputs import sha256_file
from lagrangian_backtracking.run_control import RunController, load_run_plan, load_run_progress
from lagrangian_backtracking.run_validation import iter_complete_run_trajectory_shards


@pytest.fixture(scope="module")
def preview_template(tmp_path_factory, exact_runtime_data):
    """真實執行四十成員，包含正常、數值上限及失敗取樣三種停止；不手造診斷事件。

    五個水平點使用來源投影位置，相隔約百公尺，七秒位移不足一公尺，刻意驗證局部圖。
    垂向僅改合成 fixture 名稱以檢查 near_bed 順序；不修改來源科學資料。
    """

    root = tmp_path_factory.mktemp("pilot-preview-source")
    data = dict(exact_runtime_data)
    verticals = dict(
        zip(
            (f"v{i}" for i in range(4)),
            (
                "upper_water_column",
                "mid_upper_water_column",
                "mid_lower_water_column",
                "near_bed",
            ),
            strict=True,
        )
    )
    inputs = data["inputs"]
    receptors = tuple(replace(item, vertical_id=verticals[item.vertical_id]) for item in inputs.receptors)
    pairs = tuple(
        replace(item, vertical_id=verticals[item.vertical_id]) for item in inputs.initial_conditions
    )
    data["inputs"] = replace(
        inputs,
        receptors=receptors,
        initial_conditions=pairs,
        initial_conditions_by_pair={(p.receptor_id, p.arrival_time_id): p for p in pairs},
    )
    by_id = {item.receptor_id: item for item in receptors}
    domain = next(item for item in data["config"].domains if item.analysis_region_id == "B")
    projection = DomainProjection(*domain.center_lonlat)
    data["geometries"] = replace(
        data["geometries"], projections={**data["geometries"].projections, "hsinchu": projection}
    )

    def request(unit):
        """固定微小平流、有限環境與明示兩種失敗，讓實際 engine 產生診斷欄位。"""

        base = _request(unit)
        receptor = by_id[unit.scenario.receptor_id]
        x, y = projection.project(receptor.lon, receptor.lat)
        state = replace(base.initial_state, x_m=float(x), y_m=float(y))
        gap = receptor.metadata["horizontal_id"] == "h1" and unit.member_id == 1
        limit = receptor.metadata["horizontal_id"] == "h0" and unit.member_id == 1

        def velocity(x_m, y_m, z_m, time_utc_ns):
            """資料缺口是真正取樣失敗，環境數值只供診斷保留，不作有效海洋樣本。"""
            return VelocitySample(
                0.05, 0.01, 0.0, 0.25, -10.0, 100.0, 10.0, qc=SampleQC.TIME_GAP if gap else SampleQC.OK
            )

        geometry = box(float(x) - 100, float(y) - 100, float(x) + 100, float(y) + 100)
        return replace(
            base,
            initial_state=state,
            velocity=velocity,
            boundaries=BoundaryGeometry(geometry, geometry, {}, local_equals_flow=True),
            settings=replace(base.settings, maximum_step_count=1 if limit else 20),
        )

    with pytest.MonkeyPatch.context() as patch:
        workspace = _initialize_exact_test_run(data, root, patch)
        controller = RunController(workspace.path, request_factory=request)
        for shard_id in load_run_progress(workspace.path)["shards"]:
            controller.run_shard(shard_id)
    config = root / "caller-config.yaml"
    config.write_text("synthetic_fixture: true\n", encoding="utf-8")
    return {"root": workspace.path, "config": config, "data": data}


@pytest.fixture
def preview_case(preview_template, tmp_path, monkeypatch):
    """每例複製私有合成結果，以便惡意改動測試不影響其他工作或共用 fixture。"""

    root = tmp_path / "run"
    shutil.copytree(preview_template["root"], root)
    config = tmp_path / "config.yaml"
    shutil.copyfile(preview_template["config"], config)
    _patch_static_loaders(monkeypatch, preview_template["data"])
    cache = tmp_path / "mpl-cache"
    cache.mkdir()
    monkeypatch.setenv("MPLCONFIGDIR", str(cache))
    return root, config, tmp_path / "preview"


def _build(case, **kwargs):
    """將測試專用位置傳入公開入口；不改寫預覽或正式報告的預設科學分母。"""
    root, config, output = case
    return preview.build_pilot_preview(root, config_path=config, output=output, **kwargs)


@pytest.fixture(scope="module")
def rendered_preview(preview_template, tmp_path_factory):
    """實際產圖一次並保存可供圖面 QA 的工程樣本；固定只畫每垂向五條但統計四十粒子。"""
    output = tmp_path_factory.mktemp("pilot-preview-product") / "synthetic-preview"
    cache = output.parent / "mpl-cache"
    cache.mkdir()
    with pytest.MonkeyPatch.context() as patch:
        _patch_static_loaders(patch, preview_template["data"])
        patch.setenv("MPLCONFIGDIR", str(cache))
        root, config = preview_template["root"], preview_template["config"]
        before = _snapshot_tree(root)
        config_before = config.read_bytes()
        manifest = _build((root, config, output), max_curves_per_vertical=5)
        assert _snapshot_tree(root) == before and config.read_bytes() == config_before
    return output, manifest


def test_complete_counts_units_and_readable_png(rendered_preview):
    """分母全量、局部圖五面板、單位及來源指紋可查，PNG 必須能由影像解碼器實讀。"""
    output, manifest = rendered_preview
    summary = json.loads((output / "summary.json").read_bytes())
    assert summary["scenario_count"] == summary["receptor_count"] == 20
    assert summary["members_per_scenario"] == 2
    assert summary["particle_count"] == summary["terminal_position_sample_count"] == 40
    assert summary["terminal_counts"]["max_age"] == 32
    assert summary["terminal_counts"]["numerical_failure"] == 4
    assert summary["terminal_counts"]["data_gap"] == 4
    assert summary["failure_particle_count"] == 8
    assert sum(summary["terminal_counts"].values()) == 40
    assert summary["drawn_particle_count"] == 20
    assert summary["vertical_order"][0] == "near_bed"
    assert len(summary["horizontal_panels"]) == 5
    assert all(
        row["particle_count"] == 8 and row["drawn_particle_count"] == 4
        for row in summary["horizontal_panels"]
    )
    assert all(sum(counts.values()) == 10 for counts in summary["terminal_counts_by_vertical"].values())
    assert summary["source"]["scenario_selection"]["source_scenario_count"] == 50_000
    assert summary["projection"]["units"] == "m" and not summary["projection"]["coastline_shown"]
    with (output / "observations.csv").open() as stream:
        observations = list(csv.DictReader(stream))
    assert len(observations) == summary["observation_count"]
    for row in observations:
        assert float(row["age_seconds"]) == pytest.approx(
            (summary["arrival_time_utc_ns"] - int(row["time_utc_ns"])) / 1e9
        )
        assert float(row["z_m"]) == -5
        if row["environment_sample_status"] == "valid":
            assert float(row["eta_m"]) == 0.25 and float(row["bed_z_m"]) == -10
        elif row["environment_sample_status"] == "not_sampled":
            assert row["eta_m"] == row["bed_z_m"] == ""
    for name in ("horizontal.png", "depth_age.png", "terminal_counts.png"):
        with Image.open(output / name) as image:
            assert image.format == "PNG" and min(image.size) >= 900
            image.verify()
    assert set(manifest["files"]) == {
        "horizontal.png",
        "depth_age.png",
        "terminal_counts.png",
        "particles.csv",
        "observations.csv",
        "summary.json",
        "README.md",
    }
    for name, contract in manifest["files"].items():
        assert sha256_file(output / name) == contract["sha256"]
        assert (output / name).stat().st_size == contract["size_bytes"]
    assert str(output.parent) not in (output / "summary.json").read_text()
    assert "逆向時間曲線變淺不表示" in (output / "README.md").read_text()


def test_real_engine_failure_event_roundtrip(preview_template, rendered_preview):
    """由 engine、輸出檔、公共讀取器至 CSV／JSON 逐欄核對，不手工偽造 stage/qc。"""
    output, _ = rendered_preview
    summary = json.loads((output / "summary.json").read_bytes())
    with (output / "particles.csv").open() as stream:
        particles = {row["particle_id"]: row for row in csv.DictReader(stream)}
    details = {row["particle_id"]: row for row in summary["failure_details"]}
    reasons = set()
    for shard in iter_complete_run_trajectory_shards(preview_template["root"]):
        for result in shard.results:
            raw = result.events[-1].attributes
            row = particles[result.final_state.particle_id]
            if raw.get("diagnostic_version") == 1:
                reasons.add(raw["failure_reason"])
                detail = details[result.final_state.particle_id]
                for key, value in raw.items():
                    assert detail[key] == value
                    assert row[key] == str(value)
                for key in (*preview._DIAGNOSTIC_FLOATS, *preview._DIAGNOSTIC_INTS):
                    if key not in raw:
                        assert detail[key] is None and row[key] == ""
            else:
                assert row["failure_stage"] == row["failure_reason"] == "unknown"
                assert row["qc_flags"] == ""
    assert reasons == {"maximum_step_count", "invalid_velocity_sample"}
    gaps = [row for row in details.values() if row["status"] == "data_gap"]
    assert all(
        row["qc_flags"] == int(SampleQC.TIME_GAP) and row["failure_stage"] == "step_start" for row in gaps
    )
    assert all(row["sampling_context_available"] is True and row["sample_eta_m"] == 0.25 for row in gaps)
    limits = [row for row in details.values() if row["status"] == "numerical_failure"]
    assert all(row["qc_available"] is False and row["qc_flags"] is None for row in limits)


@pytest.mark.parametrize("limit", [{"max_particles": 39}, {"max_observations": 1}])
def test_capacity_refuses_before_full_static_or_iterator(preview_case, monkeypatch, limit):
    """容量超限只讀計畫及小清單，不能先讓完整驗證器載入巨大軌跡陣列。"""

    def forbidden(*args, **kwargs):
        """若早期拒絕之前呼叫較昂貴驗證，直接暴露執行順序回歸。"""
        pytest.fail("full static / trajectory iterator must not run")

    monkeypatch.setattr(preview.runtime, "load_validated_run_static_inputs", forbidden)
    monkeypatch.setattr(preview, "iter_complete_run_trajectory_shards", forbidden)
    with pytest.raises(preview.PilotPreviewError, match="超過上限"):
        _build(preview_case, **limit)
    assert not preview_case[2].exists()


@pytest.mark.parametrize("kind", ["legacy", "incomplete", "binding", "source", "nonpilot"])
def test_source_gate_refuses_invalid_inputs(preview_case, monkeypatch, kind):
    """舊環境格式、未完成、非先導或來源異動均拒絕，不產生圖面或偷偷補齊來源。"""
    root, _, output = preview_case
    plan = load_run_plan(root)
    progress = load_run_progress(root)
    first = next(iter(progress["shards"].values()))
    if kind == "legacy":
        _downgrade_shard_to_legacy(root / first["output_relative_path"])
    elif kind == "incomplete":
        first["lifecycle"] = "PENDING"
        (root / "run_progress.json").write_text(json.dumps(progress))
    elif kind == "source":
        path = root / "normalized_config.json"
        path.write_bytes(path.read_bytes() + b" ")
    else:
        if kind == "binding":
            plan["scenario_selection"]["material_id"] = "not-the-source"
        else:
            plan["run_kind"] = "formal"
        (root / "run_plan.json").write_text(json.dumps(plan))
    before = _snapshot_tree(root)
    with pytest.raises(preview.PilotPreviewError) as error:
        _build(preview_case)
    assert str(root) not in str(error.value)
    assert not output.exists() and _snapshot_tree(root) == before


@pytest.mark.parametrize("kind", ["directory", "file", "symlink"])
def test_existing_output_preserved(preview_case, kind):
    """已存在的檔案、空目錄、懸空符號連結都拒絕，對方節點不得刪除或覆寫。"""
    output = preview_case[2]
    if kind == "directory":
        output.mkdir()
    elif kind == "file":
        output.write_bytes(b"someone else")
    else:
        output.symlink_to(output.parent / "missing")
    identity = output.lstat()
    with pytest.raises(FileExistsError):
        _build(preview_case)
    assert output.lstat().st_ino == identity.st_ino


@pytest.mark.parametrize("kind", ["missing", "duplicate", "identity", "age", "legacy"])
def test_iterator_records_must_match_all_source_members(preview_case, monkeypatch, kind):
    """公共讀取器之後再次核對來源成員及秒／奈秒，不能省略失敗或偷偷換身分。"""
    real = preview.iter_complete_run_trajectory_shards

    def altered(*args, **kwargs):
        """僅測試邊界故障注入；磁碟上的合成來源檔保持不變。"""
        for index, shard in enumerate(real(*args, **kwargs)):
            if index == 0:
                result = shard.results[0]
                if kind == "missing":
                    shard = replace(shard, results=shard.results[1:])
                elif kind == "duplicate":
                    shard = replace(shard, results=(*shard.results, result))
                elif kind == "legacy":
                    shard = replace(shard, trajectory_schema_version="1.0.0")
                else:
                    if kind == "identity":
                        result = replace(result, final_state=replace(result.final_state, receptor_id="wrong"))
                    else:
                        obs = list(result.observations)
                        obs[-1] = replace(obs[-1], age_seconds=999)
                        result = replace(result, observations=obs)
                    shard = replace(shard, results=(result, *shard.results[1:]))
            yield shard

    monkeypatch.setattr(preview, "iter_complete_run_trajectory_shards", altered)
    with pytest.raises(preview.PilotPreviewError):
        _build(preview_case)
    assert not preview_case[2].exists()


def test_cli_wiring_and_error_redaction(preview_case, monkeypatch, capsys):
    """獨立命令列轉送全部選項，公開錯誤只含固定摘要，不需新主 CLI 入口。"""
    path = Path(__file__).parents[1] / "scripts" / "build_pilot_preview.py"
    spec = importlib.util.spec_from_file_location("preview_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    def build(run, **kwargs):
        """只記錄接線；真實建立器已由以上完整工程測試執行。"""
        calls.append((run, kwargs))
        return {"run_id": "test", "artifact_kind": "pilot-preview-v1", "files": {}}

    monkeypatch.setattr(module, "build_pilot_preview", build)
    root, config, output = preview_case
    args = [
        "--run",
        str(root),
        "--config",
        str(config),
        "--output",
        str(output),
        "--checkpoint-root",
        str(root / "checkpoints"),
        "--max-particles",
        "100",
        "--max-observations",
        "900",
        "--max-curves-per-vertical",
        "5",
    ]
    assert module.main(args) == 0
    assert calls[0][0] == root and calls[0][1]["max_particles"] == 100
    assert calls[0][1]["max_observations"] == 900 and calls[0][1]["max_curves_per_vertical"] == 5
    assert calls[0][1]["checkpoint_root"] == root / "checkpoints"
    assert json.loads(capsys.readouterr().out)["valid"]
    monkeypatch.setattr(module, "build_pilot_preview", preview.build_pilot_preview)
    output.mkdir()
    assert module.main(args) == 2
    result = json.loads(capsys.readouterr().out)
    assert not result["valid"] and str(output) not in result["error"]
