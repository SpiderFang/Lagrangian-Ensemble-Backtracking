"""跨區 pilot 試跑共同設定 validator 的資料契約與 fail-closed 測試。"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from lagrangian_backtracking.pilot_matrix_validation import (
    MAX_PILOT_MATRIX_JSON_BYTES,
    canonical_pilot_matrix_json,
    validate_pilot_matrix,
)


def _provenance() -> dict[str, str]:
    """建立三區試跑共用的 deployment／dependency identity。"""

    return {
        "deployment_tree_sha256": "a" * 64,
        "uv_lock_sha256": "b" * 64,
        "package_version": "0.1.0",
        "numpy_version": "2.0.0",
        "numba_version": "0.61.0",
        "pyarrow_version": "18.0.0",
        "python_version": "3.11.0",
    }


def _config(*, stokes_formulation: str = "finite_depth_monochromatic_bulk") -> dict[str, object]:
    """建立含完整 integration、boundary、Stokes、material 與 pilot binding 的設定。"""

    return {
        "schema_version": "2.1.0",
        "design_version": "design_baseline_v3_non_rising_a_v3_local20_20260909",
        "inputs": {
            "input_root": "/data/site-specific/input",
            "input_sha256": "c" * 64,
        },
        "study_sites": [
            {
                "study_site_id": "site-b",
                "analysis_region_id": "B",
                "flow_domain_id": "domain-b",
            }
        ],
        "domains": [
            {
                "analysis_region_id": "B",
                "flow_domain_id": "domain-b",
                "geometry_path": "geometry/b.json",
            }
        ],
        # 下列三個區域發布 metadata 刻意不進共同 contract；B/C/D 真實 run 可各自不同。
        "geometry": {
            "domain_manifest": "geometry/b.json",
            "domain_manifest_sha256": "d" * 64,
        },
        "release_binding": {
            "input_directory": "/data/site-specific/release",
            "artifact_index_sha256": "e" * 64,
            "artifact_index_size_bytes": 100,
        },
        "release_approval": {
            "status": "generated",
            "region_candidate_note": "site-b calibration",
        },
        "integration": {
            "deterministic_method": "rk4",
            "time_direction": "backward_negative_dt",
            "stochastic_method": "operator_split_euler_maruyama_reference",
            "stochastic_variance_uses_absolute_dt": True,
            "dt_min_seconds": 30.0,
            "dt_max_seconds": 120.0,
            "output_interval_seconds": 600.0,
        },
        "boundaries": {
            "max_backtrack_days": 7.0,
            "maximum_step_count": 5040,
            "surface_suspended_or_sinking": "reflect_and_record_contact",
            "bed_sinking_or_near_bed": "deposit_on_first_contact_and_stop",
        },
        "physics": {
            "stokes": {
                "enabled": True,
                "formulation": stokes_formulation,
                "invalid_wave_policy": "stop_with_wave_data_gap",
            },
            "settling": {
                "positive_or_zero_velocity_policy": "reject_config",
                "material_classes": [
                    {
                        "material_id": "material-fishing-gear",
                        "behavior_class": "sinking",
                        "settling_velocity_mps": -0.002,
                    },
                    {
                        "material_id": "material-glass",
                        "behavior_class": "sinking",
                        "settling_velocity_mps": -0.1,
                    },
                ],
            },
            "horizontal_diffusion": {
                "constant_kh_m2ps": 12.0,
                "smagorinsky": {"kh_cap_m2ps": 100.0},
            },
            "vertical_diffusion": {"constant_kz_m2ps": 0.12},
        },
        "scenarios": {
            "material_selection_mode": "all_declared_materials",
            "material_count": 2,
            "arrival_time_path": "arrival/b.json",
        },
        "pilot_execution_binding": {
            "candidate_values": {
                "constant_kh_m2ps": 12.0,
                "constant_kz_m2ps": 0.12,
                "floor_m2ps": 0.1,
                "cap_m2ps": 100.0,
            },
            "execution_scalar_snapshot": {
                "dt_min_seconds": 30.0,
                "dt_max_seconds": 120.0,
                "output_interval_seconds": 600.0,
                "max_backtrack_days": 7.0,
                "maximum_step_count": 5040,
                "members_per_scenario": 4,
                "master_seed": 20260831,
                "shard_scenario_count": 2,
                "checkpoint_interval_sweeps": 8,
                "active_chunk_size": 16,
                "max_resident_forcing_months": 2,
            },
        },
    }


def _plan(
    run_id: str,
    *,
    experiment_case_id: str = "finite_depth_stokes",
    deployment_tree_sha256: str = "a" * 64,
    dt_max_seconds: int = 120,
    settling_velocity_mps: float = -0.002,
    region: str = "B",
) -> dict[str, object]:
    """建立可比較的兩 shard run plan，並保留可變欄位供拒絕／允許測試。"""

    return {
        "schema_version": "2.1.0",
        "run_id": run_id,
        "run_kind": "pilot",
        "experiment_case_id": experiment_case_id,
        "master_seed": 20260831,
        "seed_policy": "sha256_v1_pcg64dxsm",
        "members_per_scenario": 4,
        "shard_scenario_count": 2,
        "checkpoint_interval_sweeps": 8,
        "active_chunk_size": 16,
        "scenario_count": 4,
        "particle_count": 16,
        "shard_count": 2,
        "scenario_selection": {
            "schema_version": "1.0.0",
            "mode": "pilot_stratified",
            "ranking_policy": "pilot_site_vertical_sha256_rank_v1",
            "source_scenario_count": 100,
            "selected_scenario_count": 4,
            "selected_scenario_ids_sha256": "d" * 64,
            "source_records_sha256": "e" * 64,
        },
        "shards": [
            {
                "scenario_start_index": 0,
                "scenario_stop_index": 2,
                "scenario_count": 2,
                "particle_count": 8,
                "group_part_index": 0,
                "group_part_count": 1,
                "analysis_region_id": region,
                "arrival_time_utc_ns": 1_704_067_200_000_000_000,
            },
            {
                "scenario_start_index": 2,
                "scenario_stop_index": 4,
                "scenario_count": 2,
                "particle_count": 8,
                "group_part_index": 0,
                "group_part_count": 1,
                "analysis_region_id": region,
                "arrival_time_utc_ns": 1_704_067_206_000_000_000,
            },
        ],
        "code_provenance": {
            **_provenance(),
            "deployment_tree_sha256": deployment_tree_sha256,
        },
        "raw_input_inventory_sha256": "e" * 64,
        "geometry_canonical_hashes": {"domain": "f" * 64},
        "component_canonical_hashes": {"material": "0" * 64},
        "config_hash": "1" * 64,
        "checkpoint_input_binding_hash": "2" * 64,
        "path": f"/data/runs/{run_id}",
        "dt_max_seconds_fixture": dt_max_seconds,
        "settling_velocity_fixture": settling_velocity_mps,
    }


def _write_run(
    root: Path,
    *,
    plan: dict[str, object] | None = None,
    config: dict[str, object] | None = None,
) -> Path:
    """以 JSON bytes 建立測試用 immutable run root。"""

    root.mkdir()
    payload_plan = deepcopy(plan or _plan(root.name))
    payload_config = deepcopy(config or _config())
    # fixture 的兩個可變測試參數要實際落到 normalized config，才測到真正的物理 contract。
    if isinstance(plan, dict):
        if plan.get("dt_max_seconds") is not None:
            payload_config["integration"]["dt_max_seconds"] = plan["dt_max_seconds"]
        if plan.get("settling_velocity_mps") is not None:
            payload_config["physics"]["settling"]["material_classes"][0][
                "settling_velocity_mps"
            ] = plan["settling_velocity_mps"]
    (root / "run_plan.json").write_text(
        json.dumps(payload_plan, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    (root / "normalized_config.json").write_text(
        json.dumps(payload_config, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return root


def test_b_c_d_same_matrix_contract_passes_and_is_canonical(tmp_path: Path) -> None:
    """B/C/D 只改區域 identity、input/geometry binding 與區域校準時應通過。"""

    roots: list[Path] = []
    for region, kh, kz, cap in (("B", 12.0, 0.12, 100.0), ("C", 15.0, 0.15, 120.0), ("D", 18.0, 0.18, 140.0)):
        config = _config()
        config["study_sites"][0]["study_site_id"] = f"site-{region.lower()}"
        config["study_sites"][0]["analysis_region_id"] = region
        config["study_sites"][0]["flow_domain_id"] = f"domain-{region.lower()}"
        config["domains"][0]["analysis_region_id"] = region
        config["domains"][0]["flow_domain_id"] = f"domain-{region.lower()}"
        config["inputs"]["input_root"] = f"/data/{region.lower()}/input"
        config["inputs"]["input_sha256"] = region.lower() * 64
        config["geometry"]["domain_manifest"] = f"geometry/{region.lower()}.json"
        config["geometry"]["domain_manifest_sha256"] = region.lower() * 64
        config["release_binding"]["input_directory"] = f"/data/{region.lower()}/release"
        config["release_binding"]["artifact_index_sha256"] = region.lower() * 64
        config["release_binding"]["artifact_index_size_bytes"] = 100 + ord(region)
        config["release_approval"]["region_candidate_note"] = f"{region} calibration metadata"
        config["physics"]["horizontal_diffusion"]["constant_kh_m2ps"] = kh
        config["physics"]["vertical_diffusion"]["constant_kz_m2ps"] = kz
        config["physics"]["horizontal_diffusion"]["smagorinsky"]["kh_cap_m2ps"] = cap
        config["pilot_execution_binding"]["candidate_values"].update(
            {"constant_kh_m2ps": kh, "constant_kz_m2ps": kz, "cap_m2ps": cap}
        )
        plan = _plan(f"run-{region.lower()}", region=region)
        plan["scenario_selection"]["study_site_id"] = f"site-{region.lower()}"
        plan["scenario_selection"]["source_records_sha256"] = region.lower() * 64
        plan["raw_input_inventory_sha256"] = region.lower() * 64
        plan["geometry_canonical_hashes"] = {"domain": (region.lower() * 64)}
        roots.append(_write_run(tmp_path / f"run-{region.lower()}", plan=plan, config=config))

    report = validate_pilot_matrix(roots)
    assert report["valid"] is True
    assert report["run_count"] == 3
    assert report["errors"] == []
    assert canonical_pilot_matrix_json(report) == canonical_pilot_matrix_json(report)
    assert {run["identity"]["run_kind"] for run in report["runs"]} == {"pilot"}


def test_nested_regional_diffusion_projection_keeps_unknown_physics_strict(
    tmp_path: Path,
) -> None:
    """巢狀 Kh/Kz/cap 校準可跨區不同，但同層未知物理欄位仍須拒絕。"""

    base_config = _config()
    base_config["physics"]["horizontal_diffusion"]["regional_calibration"] = {
        "constant_kh_m2ps": 12.0,
        "nested": {
            "kh_cap_m2ps": 100.0,
            "method": "registered",
            "manifest_path": "/data/site-b/kh.json",
        },
    }
    base_config["physics"]["vertical_diffusion"]["regional_calibration"] = {
        "constant_kz_m2ps": 0.12,
        "method": "registered",
        "manifest_sha256": "a" * 64,
    }
    candidate_config = deepcopy(base_config)
    candidate_config["physics"]["horizontal_diffusion"]["regional_calibration"] = {
        "constant_kh_m2ps": 20.0,
        "nested": {
            "kh_cap_m2ps": 200.0,
            "method": "registered",
            "manifest_path": "/data/site-c/kh.json",
        },
    }
    candidate_config["physics"]["vertical_diffusion"]["regional_calibration"] = {
        "constant_kz_m2ps": 0.2,
        "method": "registered",
        "manifest_sha256": "b" * 64,
    }
    base = _write_run(tmp_path / "base", config=base_config)
    candidate = _write_run(tmp_path / "candidate", config=candidate_config)

    accepted = validate_pilot_matrix([base, candidate])
    assert accepted["valid"] is True

    candidate_config["physics"]["horizontal_diffusion"]["regional_calibration"]["nested"][
        "method"
    ] = "unregistered"
    (candidate / "normalized_config.json").write_text(
        json.dumps(candidate_config, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    rejected = validate_pilot_matrix([base, candidate])
    assert rejected["valid"] is False
    assert any(
        "config.physics.horizontal_diffusion.regional_calibration.nested.method" in error
        for error in rejected["errors"]
    )


@pytest.mark.parametrize("run_kind", ["formal", "synthetic"])
def test_matrix_rejects_non_pilot_run_kind(tmp_path: Path, run_kind: str) -> None:
    """formal 與 synthetic 不得冒充四區第一次 pilot 試跑進入比較。"""

    base = _write_run(tmp_path / "base")
    candidate_plan = _plan(f"candidate-{run_kind}")
    candidate_plan["run_kind"] = run_kind
    candidate = _write_run(tmp_path / f"candidate-{run_kind}", plan=candidate_plan)

    report = validate_pilot_matrix([base, candidate])

    assert report["valid"] is False
    candidate_report = report["runs"][1]
    assert candidate_report["valid"] is False
    assert "field_value_invalid:run_plan.run_kind" in candidate_report["errors"]


@pytest.mark.parametrize("case", ["missing", "duplicate"])
def test_matrix_rejects_invalid_pilot_exact_material_binding(tmp_path: Path, case: str) -> None:
    """pilot_exact 的選定材質必須在 normalized config 材質表中唯一存在。"""

    base = _write_run(tmp_path / "base")
    candidate_plan = _plan(f"candidate-material-{case}")
    candidate_plan["scenario_selection"]["mode"] = "pilot_exact"
    candidate_plan["scenario_selection"]["material_id"] = "material-fishing-gear"
    candidate_config = _config()
    if case == "missing":
        candidate_plan["scenario_selection"]["material_id"] = "material-not-declared"
    else:
        duplicate = deepcopy(candidate_config["physics"]["settling"]["material_classes"][0])
        candidate_config["physics"]["settling"]["material_classes"].append(duplicate)
    candidate = _write_run(
        tmp_path / f"candidate-material-{case}",
        plan=candidate_plan,
        config=candidate_config,
    )

    report = validate_pilot_matrix([base, candidate])

    assert report["valid"] is False
    candidate_report = report["runs"][1]
    assert candidate_report["valid"] is False
    expected_code = (
        "selected_material_not_found:normalized_config.physics.settling.material_classes"
        if case == "missing"
        else "selected_material_not_unique:normalized_config.physics.settling.material_classes"
    )
    assert expected_code in candidate_report["errors"]


def test_matrix_rejects_stokes_code_tree_dt_and_material_changes(tmp_path: Path) -> None:
    """研究方法、部署樹、dt、回溯 scalar 或沉降速度改變都必須拒絕。"""

    base = _write_run(tmp_path / "base")
    for label, plan_change, config_change in (
        (
            "stokes",
            {},
            {"physics": {"stokes": {"formulation": "no_stokes", "enabled": False}}},
        ),
        ("deployment", {"deployment_tree_sha256": "9" * 64}, {}),
        ("dt", {"dt_max_seconds": 240}, {}),
        ("material", {"settling_velocity_mps": -0.005}, {}),
    ):
        plan = _plan(f"candidate-{label}", **plan_change)
        config = _config()
        if config_change and "physics" in config_change:
            config["physics"]["stokes"].update(config_change["physics"]["stokes"])
        if label == "dt":
            config["integration"]["dt_max_seconds"] = 240.0
        if label == "material":
            config["physics"]["settling"]["material_classes"][0]["settling_velocity_mps"] = -0.005
        candidate = _write_run(tmp_path / f"candidate-{label}", plan=plan, config=config)
        report = validate_pilot_matrix([base, candidate])
        assert report["valid"] is False, label
        assert report["errors"], label
        if label == "stokes":
            assert any("config.physics.stokes" in error for error in report["errors"])


def test_v2_shaped_matrix_rejects_only_provenance_drift_after_allowed_region_changes(
    tmp_path: Path,
) -> None:
    """接近實際 v2 plan/config 的 B/C/D metadata 差異通過，部署與 Python 漂移才拒絕。"""

    base = _write_run(tmp_path / "b-v2")
    candidate_config = _config()
    candidate_config["study_sites"][0].update(
        {"study_site_id": "site-c", "analysis_region_id": "C", "flow_domain_id": "domain-c"}
    )
    candidate_config["domains"][0].update(
        {"analysis_region_id": "C", "flow_domain_id": "domain-c", "geometry_path": "geometry/c.json"}
    )
    candidate_config["inputs"].update(
        {"input_root": "/data/c/input", "input_sha256": "c" * 64}
    )
    candidate_config["release_binding"].update(
        {"input_directory": "/data/c/release", "artifact_index_sha256": "d" * 64}
    )
    candidate_config["physics"]["horizontal_diffusion"].update({"constant_kh_m2ps": 20.0})
    candidate_config["physics"]["vertical_diffusion"].update({"constant_kz_m2ps": 0.2})
    candidate_config["physics"]["horizontal_diffusion"]["smagorinsky"]["kh_cap_m2ps"] = 200.0
    candidate_config["pilot_execution_binding"]["candidate_values"].update(
        {"constant_kh_m2ps": 20.0, "constant_kz_m2ps": 0.2, "cap_m2ps": 200.0}
    )
    candidate_plan = _plan("c-v2", region="C")
    candidate_plan["code_provenance"]["python_version"] = "3.12.0"
    candidate = _write_run(tmp_path / "c-v2", plan=candidate_plan, config=candidate_config)

    rejected = validate_pilot_matrix([base, candidate])
    assert rejected["valid"] is False
    assert any("code_provenance.python_version" in error for error in rejected["errors"])
    assert not any("config.physics.horizontal_diffusion" in error for error in rejected["errors"])

    candidate_plan["code_provenance"]["python_version"] = "3.11.0"
    (candidate / "run_plan.json").write_text(
        json.dumps(candidate_plan, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    accepted = validate_pilot_matrix([base, candidate])
    assert accepted["valid"] is True


def test_matrix_rejects_missing_symlink_and_oversized_inputs(tmp_path: Path) -> None:
    """缺欄位、symlink root 與超過大小上限的 JSON 均不可進入比較。"""

    base = _write_run(tmp_path / "base")
    missing = _write_run(tmp_path / "missing")
    missing_config = json.loads((missing / "normalized_config.json").read_text(encoding="utf-8"))
    del missing_config["integration"]
    (missing / "normalized_config.json").write_text(json.dumps(missing_config), encoding="utf-8")
    missing_report = validate_pilot_matrix([base, missing])
    assert missing_report["valid"] is False
    assert any("pilot_matrix_run_invalid" in error for error in missing_report["errors"])

    symlink = tmp_path / "symlink"
    symlink.symlink_to(base, target_is_directory=True)
    symlink_report = validate_pilot_matrix([base, symlink])
    assert symlink_report["valid"] is False
    assert any("symlink_forbidden" in error for error in symlink_report["errors"])

    oversized = _write_run(tmp_path / "oversized")
    oversized_config = json.loads((oversized / "normalized_config.json").read_text(encoding="utf-8"))
    oversized_config["padding"] = "x" * (MAX_PILOT_MATRIX_JSON_BYTES + 1)
    (oversized / "normalized_config.json").write_text(json.dumps(oversized_config), encoding="utf-8")
    oversized_report = validate_pilot_matrix([base, oversized])
    assert oversized_report["valid"] is False
    assert any("file_too_large" in error for error in oversized_report["errors"])


def test_matrix_requires_two_runs_and_catches_scalar_snapshot_difference(tmp_path: Path) -> None:
    """單一 root 與 pilot scalar snapshot 缺欄位／差異都回報狀態 2 所需的 invalid。"""

    base = _write_run(tmp_path / "base")
    one = validate_pilot_matrix([base])
    assert one["valid"] is False
    assert "pilot_matrix_requires_at_least_two_runs" in one["errors"]

    candidate_config = _config()
    candidate_config["pilot_execution_binding"]["execution_scalar_snapshot"]["active_chunk_size"] = 32
    candidate = _write_run(tmp_path / "candidate", config=candidate_config)
    report = validate_pilot_matrix([base, candidate])
    assert report["valid"] is False
    assert any("config.pilot_execution_binding.execution_scalar_snapshot.active_chunk_size" in error
               for error in report["errors"])
