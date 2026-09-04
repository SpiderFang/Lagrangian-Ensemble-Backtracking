"""pilot execution config 的 calibration-bound 與 immutable publish 測試。

本檔只建立小型 JSON/YAML 測試資料，不建立 OCM/NWW accepted product、SERVER 產物或
軌跡結果。release 與 calibration 的既有 validator 在 builder 單元測試中以明示的
``monkeypatch`` 隔離，讓測試專注於本 Slice 的 candidate binding、scalar contract、
path rebase 與 atomic publish；完整上游資料契約仍由各自模組的測試負責。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from test_runtime import (
    _factory,
    _matching_location,
    _patch_from_roots,
    _unit,
)
from test_runtime import runtime_fixture as runtime_fixture

import lagrangian_backtracking.cli as cli
import lagrangian_backtracking.pilot_config as pilot_config
from lagrangian_backtracking.config import ProjectConfig, load_config
from lagrangian_backtracking.input_derivation import (
    ARTIFACT_FILENAMES,
    DERIVED_INPUT_SCHEMA_VERSION,
    write_canonical_json,
)
from lagrangian_backtracking.pilot_calibration import _write_json

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = ROOT / "configs" / "lagrangian_backtracking.example.yaml"


def _write_input_fixture(root: Path) -> str:
    """寫入固定檔名的極小 accepted-like component，並回傳 artifact index raw hash。

    component 內容不參與本 Slice 的科學計算；重要的是每個固定檔名都有 canonical JSON
    與 sidecar，讓 pilot builder 仍透過既有 immutable input fingerprint 產生 release
    binding。gap-safe 檔案則另外帶入 root 與 record 的時間支援欄位，供 horizon gate
    驗證 requested backtrack window，而不是以零值或最近資料補齊時間缺口。
    """

    root.mkdir()
    for kind, filename in ARTIFACT_FILENAMES.items():
        payload: dict[str, Any] = {"manifest_kind": f"test_{kind}", "records": []}
        if kind == "ocm_gap_safe_arrival_horizon":
            payload = {
                "manifest_kind": "ocm_gap_safe_arrival_horizon_manifest",
                "schema_version": DERIVED_INPUT_SCHEMA_VERSION,
                "max_backtrack_days": 30.0,
                "records": [
                    {
                        "max_backtrack_days": 30.0,
                        "crossed_gap": False,
                        "missing_utc": [],
                        "expected_step_count": 720,
                        "supported_step_count": 720,
                    }
                ],
            }
        write_canonical_json(root / filename, payload)
    write_canonical_json(
        root / "artifact_index.json",
        {
            "manifest_kind": "derived_input_artifact_index",
            "artifacts": [
                {"kind": kind, "path": filename}
                for kind, filename in sorted(ARTIFACT_FILENAMES.items())
            ],
        },
    )
    _payload, fingerprint = pilot_config.read_canonical_json(root / "artifact_index.json")
    return str(fingerprint["sha256"])


def _replace_canonical_json(path: Path, payload: dict[str, Any]) -> None:
    """只在隔離測試目錄重建 immutable JSON，讓下一次讀取取得新的 sidecar hash。"""

    path.unlink()
    Path(f"{path}.sha256").unlink()
    write_canonical_json(path, payload)


def _write_source_config(path: Path) -> ProjectConfig:
    """由專案 example 建立保留五站／50k 設計的 release-like source YAML。"""

    payload = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    # 測試刻意保留完整 design contract，只補上 release validator 需要存在的 binding
    # root；實際 artifact hash gate 由該 validator 的專屬測試覆蓋。
    payload["release_binding"] = {"schema_version": DERIVED_INPUT_SCHEMA_VERSION}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return ProjectConfig.model_validate(payload)


def _write_calibration_fixture(root: Path, *, source_hash: str, input_hash: str) -> None:
    """建立最小 1.1.0 calibration metadata 與三個固定 payload 檔案。"""

    root.mkdir()
    report = {
        "schema_version": "1.1.0",
        "artifact_kind": "ocm_pilot_calibration_evidence",
        "evidence_class": "server_real_data_pilot_candidate",
        "completion_status": "complete",
        "recommendation_status": (
            "candidate_pending_trajectory_convergence_and_scientific_validation"
        ),
        "input_binding": {
            "config_hash": source_hash,
            "input_artifact_index_sha256": input_hash,
        },
        "candidates": {
            "constant_kh_m2ps": {"available": True, "value": 2.0},
            "constant_kz_m2ps": {"available": True, "value": 0.5},
            "floor_m2ps": {"available": True, "value": 0.0},
            "cap_m2ps": {"available": True, "value": 10.0},
        },
    }
    _write_json(root / "calibration_report.json", report)
    _write_json(root / "manifest.json", {"schema_version": "1.1.0"})
    (root / "pair_samples.parquet").write_bytes(b"test pair payload")


def _patch_external_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    """隔離上游 release/calibration semantic validator，保留本模組實際 hash 讀取。"""

    monkeypatch.setattr(
        pilot_config,
        "validate_release_config",
        lambda *args, **kwargs: {"valid": True, "summary": {}},
    )
    monkeypatch.setattr(
        pilot_config,
        "validate_pilot_calibration",
        lambda *args, **kwargs: {"valid": True, "summary": {}},
    )


def _create_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    destination_parent: str = "target",
) -> tuple[Path, Path, Path, Path, dict[str, Any]]:
    """建立一組可供 builder round-trip 使用的隔離 fixture。"""

    _patch_external_gates(monkeypatch)
    input_root = tmp_path / "accepted-input"
    input_hash = _write_input_fixture(input_root)
    source_path = tmp_path / "source" / "release.yaml"
    source = _write_source_config(source_path)
    calibration_root = tmp_path / "calibration"
    _write_calibration_fixture(
        calibration_root,
        source_hash=source.config_hash(),
        input_hash=input_hash,
    )
    destination = tmp_path / destination_parent / "pilot.yaml"
    scalars = {
        "dt_min_seconds": 60.0,
        "dt_max_seconds": 300.0,
        "output_interval_seconds": 900.0,
        "max_backtrack_days": 7.0,
        "maximum_step_count": 2016,
        "members_per_scenario": 4,
        "master_seed": 123,
        "shard_scenario_count": 100,
        "checkpoint_interval_sweeps": 10,
        "active_chunk_size": None,
        "max_resident_forcing_months": 2,
    }
    result = pilot_config.create_pilot_execution_config(
        source_path,
        input_root,
        calibration_root,
        destination,
        **scalars,
    )
    return source_path, input_root, calibration_root, destination, {**result, **scalars}


def _scalar_arguments(summary: dict[str, Any]) -> dict[str, Any]:
    """從 create summary 取出 builder API 的完整 scalar keyword 集合。"""

    return {name: summary[name] for name in pilot_config._EXECUTION_SCALAR_NAMES}


@pytest.mark.parametrize(("days", "seconds"), [(1 / 24, 3600.0), (1 / 48, 1800.0), (7 / 24, 25200.0)])
def test_builder_fractional_day_config_constructs_actual_runtime_request(
    tmp_path, monkeypatch, runtime_fixture, days, seconds,
) -> None:
    """由真實 builder 發布 YAML、重新載入，再經 RuntimeRequestFactory 產生小時制 request。

    上游清單／校準語意及 forcing manager 仍用既有工程替身，不讀海洋陣列；不再以自造
    七秒 EngineSettings 取代本次錯誤路徑。ID、UTC、種子與沉降物性原樣保留。
    """

    source, input_root, calibration_root, _, summary = _create_fixture(tmp_path, monkeypatch)
    scalars = {**_scalar_arguments(summary), "max_backtrack_days": days, "members_per_scenario": 1}
    destination = tmp_path / "hourly-pilot.yaml"
    pilot_config.create_pilot_execution_config(source, input_root, calibration_root, destination, **scalars)
    before = destination.read_bytes()
    config = load_config(destination)
    assert config.boundaries.max_backtrack_days == days
    assert config.scenarios.members_per_scenario == 1 and config.scenarios.scenario_count == 50_000
    assert config.pilot_execution_binding["execution_scalar_snapshot"]["max_backtrack_days"] == days
    data = {**runtime_fixture, "config": config}
    calls, _ = _patch_from_roots(monkeypatch, _matching_location(data))
    factory = _factory(data, tmp_path)
    unit = _unit(data["scenario"], member_id=0)
    original_seed = unit.seed
    request = factory(unit)
    assert request.settings.max_backtrack_seconds == seconds
    assert request.settings.earliest_forcing_time_utc_ns == (
        unit.scenario.arrival_time_utc_ns - int(seconds) * 1_000_000_000
    )
    assert request.initial_state.time_utc_ns == unit.scenario.arrival_time_utc_ns
    assert request.initial_state.scenario_id == unit.scenario.scenario_id
    assert request.initial_state.receptor_id == unit.scenario.receptor_id
    assert request.initial_state.study_site_id == unit.scenario.study_site_id
    assert request.behavior_class == "sinking" and unit.scenario.settling_velocity_mps < 0
    assert unit.seed == original_seed and len(calls) == 1
    assert destination.read_bytes() == before


def _walk_strings(value: object) -> list[str]:
    """收集 mapping/list 中的字串，供 binding no-path assertion 使用。"""

    if isinstance(value, dict):
        return [item for child in value.values() for item in _walk_strings(child)]
    if isinstance(value, list):
        return [item for child in value for item in _walk_strings(child)]
    return [value] if isinstance(value, str) else []


def test_round_trip_rebases_runtime_paths_and_binds_exact_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """不同 parent 的 target 必須保留五站／50k 並重建相對 runtime path。"""

    source_path, input_root, calibration_root, destination, result = _create_fixture(
        tmp_path,
        monkeypatch,
        destination_parent="nested/target",
    )
    payload = yaml.safe_load(destination.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert payload["config_status"] == "generated"
    assert payload["scenarios"]["scenario_count"] == 50_000
    assert len(payload["study_sites"]) == 5
    assert payload["physics"]["horizontal_diffusion"]["constant_kh_m2ps"] == 2.0
    assert payload["physics"]["horizontal_diffusion"]["smagorinsky"]["kh_floor_m2ps"] == 0.0
    assert payload["physics"]["horizontal_diffusion"]["smagorinsky"]["kh_cap_m2ps"] == 10.0
    assert payload["physics"]["vertical_diffusion"]["constant_kz_m2ps"] == 0.5
    assert payload["inputs"]["derived_input_artifact_index"] == "../../accepted-input/artifact_index.json"
    binding = payload["pilot_execution_binding"]
    assert set(binding) == {
        "schema_version",
        "source_config_hash",
        "input_artifact_index_sha256",
        "calibration_schema_version",
        "calibration_manifest_sha256",
        "calibration_report_sha256",
        "pair_samples_sha256",
        "candidate_values",
        "applied_fields",
        "execution_scalar_snapshot",
        "status",
    }
    assert "source_config_file_sha256" not in binding
    assert binding["schema_version"] == "1.0.0"
    assert binding["status"] == pilot_config.PILOT_EXECUTION_BINDING_STATUS
    assert binding["candidate_values"] == {
        "constant_kh_m2ps": 2.0,
        "constant_kz_m2ps": 0.5,
        "floor_m2ps": 0.0,
        "cap_m2ps": 10.0,
    }
    assert all("/" not in item and "\\" not in item for item in _walk_strings(binding))
    assert result["valid"] is True
    assert result["target_config_hash"]
    assert pilot_config.validate_pilot_execution_config(
        destination,
        input_directory=input_root,
        calibration_directory=calibration_root,
    )["valid"] is True
    assert source_path.exists()


def test_create_failure_does_not_publish_destination_or_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """hidden YAML 或 post-write validator 失敗時 final 與 partial 都不可殘留。"""

    _patch_external_gates(monkeypatch)
    input_root = tmp_path / "accepted-input"
    input_hash = _write_input_fixture(input_root)
    source_path = tmp_path / "source.yaml"
    source = _write_source_config(source_path)
    calibration_root = tmp_path / "calibration"
    _write_calibration_fixture(
        calibration_root,
        source_hash=source.config_hash(),
        input_hash=input_hash,
    )
    monkeypatch.setattr(
        pilot_config,
        "validate_pilot_execution_config",
        lambda *args, **kwargs: {"valid": False, "errors": ["forced_invalid"], "summary": {}},
    )
    destination = tmp_path / "nested" / "pilot.yaml"
    with pytest.raises(ValueError, match="target pilot execution validator failed"):
        pilot_config.create_pilot_execution_config(
            source_path,
            input_root,
            calibration_root,
            destination,
            dt_min_seconds=60.0,
            dt_max_seconds=300.0,
            output_interval_seconds=900.0,
            max_backtrack_days=7.0,
            maximum_step_count=2016,
            members_per_scenario=4,
            master_seed=123,
            shard_scenario_count=100,
            checkpoint_interval_sweeps=10,
            active_chunk_size=None,
            max_resident_forcing_months=2,
        )
    assert not destination.exists()
    assert not list(destination.parent.glob(f".{destination.name}.partial-*.yaml"))


def test_destination_existing_or_symlink_is_never_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """existing final 與 symlink final 都必須在任何 publish 前拒絕。"""

    source, input_root, calibration_root, destination, scalars = _create_fixture(
        tmp_path, monkeypatch
    )
    with pytest.raises(FileExistsError):
        pilot_config.create_pilot_execution_config(
            source,
            input_root,
            calibration_root,
            destination,
            **_scalar_arguments(scalars),
        )
    destination.unlink()
    destination.symlink_to(source)
    with pytest.raises(FileExistsError):
        pilot_config.create_pilot_execution_config(
            source,
            input_root,
            calibration_root,
            destination,
            **_scalar_arguments(scalars),
        )
    assert destination.is_symlink()


def test_atomic_replace_failure_leaves_no_destination_or_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """atomic rename 失敗時 temporary 會清除，destination 不會被假稱已發布。"""

    source, input_root, calibration_root, destination, result = _create_fixture(
        tmp_path, monkeypatch
    )
    destination.unlink()

    def reject_replace(*args: object, **kwargs: object) -> None:
        """模擬檔案系統 rename 失敗，不讓測試碰觸真正外部資料。"""

        del args, kwargs
        raise OSError("simulated replace failure")

    monkeypatch.setattr(pilot_config.os, "replace", reject_replace)
    scalar_keys = {
        "dt_min_seconds",
        "dt_max_seconds",
        "output_interval_seconds",
        "max_backtrack_days",
        "maximum_step_count",
        "members_per_scenario",
        "master_seed",
        "shard_scenario_count",
        "checkpoint_interval_sweeps",
        "active_chunk_size",
        "max_resident_forcing_months",
    }
    with pytest.raises(OSError, match="simulated replace failure"):
        pilot_config.create_pilot_execution_config(
            source,
            input_root,
            calibration_root,
            destination,
            **{key: result[key] for key in scalar_keys},
        )
    assert not destination.exists()
    assert not list(destination.parent.glob(f".{destination.name}.partial-*.yaml"))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dt_min_seconds", True),
        ("max_backtrack_days", float("nan")),
        ("members_per_scenario", 0),
        ("active_chunk_size", "none"),
    ],
)
def test_scalar_gate_rejects_non_native_or_invalid_values(field: str, value: object) -> None:
    """scalar contract 拒絕 bool、非有限值、零成員與未轉換的 ``none`` 字串。"""

    scalars: dict[str, object] = {
        "dt_min_seconds": 60.0,
        "dt_max_seconds": 300.0,
        "output_interval_seconds": 900.0,
        "max_backtrack_days": 7.0,
        "maximum_step_count": 2016,
        "members_per_scenario": 4,
        "master_seed": 123,
        "shard_scenario_count": 100,
        "checkpoint_interval_sweeps": 10,
        "active_chunk_size": None,
        "max_resident_forcing_months": 2,
    }
    scalars[field] = value
    with pytest.raises(ValueError):
        pilot_config._normalise_execution_scalars(**scalars)


@pytest.mark.parametrize("mutation", ["legacy", "partial", "evidence", "recommendation", "candidate"])
def test_create_rejects_legacy_or_incomplete_calibration_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    """builder 不得把 legacy、partial、錯誤 evidence/recommendation 或 unavailable candidate 升級。"""

    source, input_root, calibration_root, destination, result = _create_fixture(
        tmp_path, monkeypatch
    )
    report_path = calibration_root / "calibration_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert isinstance(report, dict)
    if mutation == "legacy":
        report["schema_version"] = "1.0.0"
    elif mutation == "partial":
        report["completion_status"] = "partial_engineering_sample"
    elif mutation == "evidence":
        report["evidence_class"] = "synthetic_fixture"
    elif mutation == "recommendation":
        report["recommendation_status"] = "candidate_approved"
    else:
        report["candidates"]["constant_kh_m2ps"]["available"] = False
    _write_json(report_path, report)
    destination.unlink()
    with pytest.raises(ValueError):
        pilot_config.create_pilot_execution_config(
            source,
            input_root,
            calibration_root,
            destination,
            **_scalar_arguments(result),
        )
    assert not destination.exists()


@pytest.mark.parametrize(
    "mutation",
    ["root_horizon", "record_horizon", "crossed_gap", "missing_utc", "step_count"],
)
def test_gap_safe_horizon_gate_rejects_unsupported_records(
    tmp_path: Path,
    mutation: str,
) -> None:
    """gap-safe root／record 任一缺口或步數不完整都不可進入 pilot config。"""

    input_root = tmp_path / "input"
    _write_input_fixture(input_root)
    gap_path = input_root / ARTIFACT_FILENAMES["ocm_gap_safe_arrival_horizon"]
    gap = json.loads(gap_path.read_text(encoding="utf-8"))
    assert isinstance(gap, dict)
    record = gap["records"][0]
    if mutation == "root_horizon":
        gap["max_backtrack_days"] = 5.0
    elif mutation == "record_horizon":
        record["max_backtrack_days"] = 5.0
    elif mutation == "crossed_gap":
        record["crossed_gap"] = True
    elif mutation == "missing_utc":
        record["missing_utc"] = ["2025-01-01T00:00:00Z"]
    else:
        record["supported_step_count"] = record["expected_step_count"] - 1
    _replace_canonical_json(gap_path, gap)
    with pytest.raises(ValueError):
        pilot_config._read_gap_safe_horizon(input_root, requested_max_backtrack_days=7.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"dt_min_seconds": 301.0, "dt_max_seconds": 300.0},
            "順序",
        ),
        (
            {"dt_min_seconds": 60.0, "dt_max_seconds": 900.0, "output_interval_seconds": 600.0},
            "順序",
        ),
        ({"maximum_step_count": 2015}, "不足"),
    ],
)
def test_scalar_gate_rejects_order_and_step_budget(
    kwargs: dict[str, object],
    message: str,
) -> None:
    """dt 順序與 horizon 步數下限必須在檔案寫入前 fail closed。"""

    scalars: dict[str, object] = {
        "dt_min_seconds": 60.0,
        "dt_max_seconds": 300.0,
        "output_interval_seconds": 900.0,
        "max_backtrack_days": 7.0,
        "maximum_step_count": 2016,
        "members_per_scenario": 4,
        "master_seed": 123,
        "shard_scenario_count": 100,
        "checkpoint_interval_sweeps": 10,
        "active_chunk_size": None,
        "max_resident_forcing_months": 2,
    }
    scalars.update(kwargs)
    with pytest.raises(ValueError, match=message):
        pilot_config._normalise_execution_scalars(**scalars)


def test_validator_rejects_unknown_or_missing_binding_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pilot binding topology 只接受 exact key set，新增或刪除欄位都 fail closed。"""

    _source, input_root, calibration_root, destination, _result = _create_fixture(
        tmp_path, monkeypatch
    )
    original = yaml.safe_load(destination.read_text(encoding="utf-8"))
    assert isinstance(original, dict)
    for mutation in ("unknown", "missing"):
        payload = copy.deepcopy(original)
        binding = payload["pilot_execution_binding"]
        if mutation == "unknown":
            binding["unexpected"] = True
        else:
            del binding["status"]
        destination.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        result = pilot_config.validate_pilot_execution_config(
            destination,
            input_directory=input_root,
            calibration_directory=calibration_root,
        )
        assert result["valid"] is False
        assert "pilot_execution_binding_keys_invalid" in result["errors"]


@pytest.mark.parametrize(
    "mutation",
    [
        "target_candidate",
        "binding_candidate",
        "target_scalar",
        "source_config_hash",
        "binding_hash",
        "report_hash",
        "input_artifact_hash",
    ],
)
def test_validator_rejects_candidate_scalar_and_hash_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    """candidate、scalar、calibration payload 或 input artifact 任一竄改都不得通過。"""

    _source, input_root, calibration_root, destination, _result = _create_fixture(
        tmp_path, monkeypatch
    )
    if mutation in {
        "target_candidate",
        "binding_candidate",
        "target_scalar",
        "source_config_hash",
        "binding_hash",
    }:
        payload = yaml.safe_load(destination.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        binding = payload["pilot_execution_binding"]
        if mutation == "target_candidate":
            payload["physics"]["horizontal_diffusion"]["constant_kh_m2ps"] = 999.0
        elif mutation == "binding_candidate":
            binding["candidate_values"]["constant_kh_m2ps"] = 999.0
        elif mutation == "target_scalar":
            payload["boundaries"]["max_backtrack_days"] = 8.0
        elif mutation == "source_config_hash":
            binding["source_config_hash"] = "0" * 64
        else:
            binding["calibration_report_sha256"] = "0" * 64
        destination.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    elif mutation == "report_hash":
        report_path = calibration_root / "calibration_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["candidates"]["cap_m2ps"]["value"] = 11.0
        _write_json(report_path, report)
    else:
        index_path = input_root / "artifact_index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        index["status"] = "tampered"
        _replace_canonical_json(index_path, index)
    validation = pilot_config.validate_pilot_execution_config(
        destination,
        input_directory=input_root,
        calibration_directory=calibration_root,
    )
    assert validation["valid"] is False


def test_summary_is_json_safe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """create 與 validate summary 可直接交給 CLI JSON encoder。"""

    _source, input_root, calibration_root, destination, result = _create_fixture(
        tmp_path, monkeypatch
    )
    validation = pilot_config.validate_pilot_execution_config(
        destination,
        input_directory=input_root,
        calibration_directory=calibration_root,
    )
    json.dumps(result, ensure_ascii=False, allow_nan=False)
    json.dumps(validation, ensure_ascii=False, allow_nan=False)


def test_cli_pilot_config_commands_emit_json_and_invalid_returns_two(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """兩個 CLI 子命令應保留 JSON contract，validator invalid 對應 exit code 2。"""

    monkeypatch.setattr(
        cli,
        "create_pilot_execution_config",
        lambda *args, **kwargs: {"valid": True, "status": "candidate_pending_dt_and_member_convergence"},
    )
    create_args = [
        "pilot-config-create",
        "--source-config",
        str(tmp_path / "source.yaml"),
        "--input-directory",
        str(tmp_path / "input"),
        "--calibration",
        str(tmp_path / "calibration"),
        "--output",
        str(tmp_path / "pilot.yaml"),
        "--dt-min-seconds",
        "60",
        "--dt-max-seconds",
        "300",
        "--output-interval-seconds",
        "900",
        "--max-backtrack-days",
        "7",
        "--maximum-step-count",
        "2016",
        "--members-per-scenario",
        "4",
        "--master-seed",
        "123",
        "--shard-scenario-count",
        "100",
        "--checkpoint-interval-sweeps",
        "10",
        "--active-chunk-size",
        "none",
        "--max-resident-forcing-months",
        "2",
    ]
    assert cli.main(create_args) == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True

    monkeypatch.setattr(
        cli,
        "validate_pilot_execution_config",
        lambda *args, **kwargs: {
            "valid": False,
            "errors": ["pilot_execution_binding_keys_invalid"],
            "summary": {},
        },
    )
    validate_code = cli.main(
        [
            "pilot-config-validate",
            str(tmp_path / "pilot.yaml"),
            "--input-directory",
            str(tmp_path / "input"),
            "--calibration",
            str(tmp_path / "calibration"),
        ]
    )
    assert validate_code == 2
    assert json.loads(capsys.readouterr().out)["valid"] is False
