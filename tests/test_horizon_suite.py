"""共同最長回溯母體與多 horizon release suite 的工程契約測試。

測試用小型 canonical JSON 取代 OCM／NWW 大型陣列，但保留正式流程會讀取的十個
component、相鄰 SHA-256 sidecar、artifact index、release binding 與 suite manifest。
因此這些測試驗證的是一次 input build、共同 identity、原子發布與 fail-closed 邊界；
它們不把 synthetic bytes 當成海洋科學成果，也不讀取 SERVER 或 raw NetCDF。
"""

from __future__ import annotations

import copy
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

import lagrangian_backtracking.horizon_suite as horizon_suite
from lagrangian_backtracking.input_derivation import (
    ARTIFACT_FILENAMES,
    DERIVED_INPUT_SCHEMA_VERSION,
    read_canonical_json,
    write_canonical_json,
)
from lagrangian_backtracking.input_horizon import (
    BED_RESIDENCE_INPUT_SCHEMA_VERSION,
    GAP_CENSORED_BED_RESIDENCE_INPUT_SCHEMA_VERSION,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = ROOT / "configs" / "lagrangian_backtracking.example.yaml"


def _write_template(path: Path, *, support_days: int | None = None) -> bytes:
    """建立未啟用 bed-residence 的 legacy template，保留原 suite 相容性測試。"""

    payload = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    # 正式 example 現在是 2025 observation；本 helper 仍專門建立 legacy suite，
    # 因此要同步退回舊 v3 design 並移除新版 typed selection，而不是只關閉 bed block。
    payload["design_version"] = "design_baseline_v3_non_rising_a_v3_local20_20260909"
    payload.pop("arrival_time_selection", None)
    payload["scenarios"].pop("bed_residence_time", None)
    payload["inputs"].pop("backtrack_support_days", None)
    # 此 helper 專門建立 legacy suite；example 的 gap-censored policy 必須整組移除，
    # 否則沒有 bed block 的 legacy config 會被正確視為不完整宣告。
    time_axis_contract = payload["inputs"].get("time_axis_contract")
    if isinstance(time_axis_contract, dict):
        for field in ("gap_policy", "stop_at_first_gap", "denominator_policy"):
            time_axis_contract.pop(field, None)
    payload["integration"]["dt_min_seconds"] = 30.0
    if support_days is not None:
        payload["inputs"]["backtrack_support_days"] = support_days
    rendered = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False).encode("utf-8")
    path.write_bytes(rendered)
    return rendered


def _write_bed_template(path: Path) -> bytes:
    """建立正式 bed-residence config 樣板，驗證 180 日選時／90 日運算契約。"""

    payload = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    payload["integration"]["dt_min_seconds"] = 30.0
    payload["inputs"]["backtrack_support_days"] = 180
    payload["boundaries"]["max_backtrack_days"] = None
    payload["scenarios"]["bed_residence_time"]["runtime_horizon_support_days"] = None
    rendered = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False).encode("utf-8")
    path.write_bytes(rendered)
    return rendered


def _replace_canonical_json(path: Path, payload: dict[str, Any]) -> None:
    """測試專用地重簽 immutable JSON，模擬攻擊者同時改寫主檔與 sidecar。"""

    path.unlink()
    Path(f"{path}.sha256").unlink()
    write_canonical_json(path, payload)


def _load_manifest(path: Path) -> tuple[dict[str, Any], Path]:
    """讀取 suite manifest，並回傳 payload 與其路徑供測試重簽。"""

    manifest_path = path / "horizon-suite-manifest.json"
    payload, _ = read_canonical_json(manifest_path)
    return payload, manifest_path


def _install_small_suite_doubles(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_release_days: int | None = None,
    force_release_status: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """安裝只替代大型資料計算的 fake，仍寫出完整 fixed artifact closure。"""

    calls: dict[str, list[dict[str, Any]]] = {
        "build": [],
        "create": [],
        "input_validate": [],
        "release_validate": [],
    }

    def fake_build_input_derivatives(**kwargs: Any) -> None:
        """寫出十個可由 suite fingerprint reader 驗證的小型 canonical component。"""

        calls["build"].append(dict(kwargs))
        destination = Path(kwargs["destination"])
        destination.mkdir(parents=True)
        component_records: list[dict[str, Any]] = []
        for kind, filename in sorted(ARTIFACT_FILENAMES.items()):
            payload: dict[str, Any] = {
                "manifest_kind": f"test_{kind}",
                "identity": {
                    "arrival": "arrival-parent-v1",
                    "receptor": "receptor-parent-v1",
                    "material": "material-parent-v1",
                    "initial_condition": "initial-parent-v1",
                }.get(kind, f"component-{kind}-v1"),
                "records": [{"component_kind": kind, "record_id": f"{kind}-0"}],
            }
            if kind == "ocm_gap_safe_arrival_horizon":
                config_payload = yaml.safe_load(
                    Path(kwargs["config_path"]).read_text(encoding="utf-8")
                )
                bed = config_payload["scenarios"].get("bed_residence_time")
                payload["max_backtrack_days"] = 90
                payload["support_days"] = 90
                if bed is not None:
                    payload["selection_support_days"] = 180
                    payload["runtime_support_days"] = 90
            fingerprint = write_canonical_json(destination / filename, payload)
            component_records.append({"kind": kind, **fingerprint})
        index_fingerprint = write_canonical_json(
            destination / "artifact_index.json",
            {
                "manifest_kind": "derived_input_artifact_index",
                "schema_version": "1.0.0",
                "status": "immutable",
                "artifacts": component_records,
                "source_bindings": {
                    "config_hash": horizon_suite._config_hash_from_payload(
                        yaml.safe_load(Path(kwargs["config_path"]).read_text(encoding="utf-8"))
                    ),
                },
            },
        )
        write_canonical_json(
            destination / "artifact_bindings.json",
            {
                "manifest_kind": "derived_input_artifact_closure",
                "schema_version": "1.0.0",
                "artifacts": [
                    *component_records,
                    {"kind": "artifact_index", **index_fingerprint},
                ],
            },
        )

    def fake_validate_input_derivatives(*args: Any, **kwargs: Any) -> dict[str, Any]:
        """回傳 JSON-safe input gate；component bytes 已由 fake builder 真實寫入。"""

        calls["input_validate"].append({"args": args, **kwargs})
        return {"valid": True, "errors": [], "warnings": [], "summary": {"test": True}}

    def fake_create_release_config(**kwargs: Any) -> dict[str, Any]:
        """產生合理 release YAML，binding 逐項指向共同 input 的 fixed component。"""

        calls["create"].append(dict(kwargs))
        days = int(float(kwargs["max_backtrack_days"]))
        if fail_release_days == days:
            raise RuntimeError(f"測試刻意讓 {days} 日 release 建置失敗")
        template = Path(kwargs["config_template_path"])
        input_directory = Path(kwargs["input_directory"])
        output = Path(kwargs["output_path"])
        payload = yaml.safe_load(template.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        payload["boundaries"]["max_backtrack_days"] = float(days)
        payload["boundaries"]["maximum_step_count"] = int(kwargs["maximum_step_count"])
        mode = kwargs.get("backtrack_mode_override")
        bed = payload["scenarios"].get("bed_residence_time")
        if bed is not None and mode is not None:
            bed["backtrack_mode"] = mode
        horizon_suite._set_release_manifest_references(payload)
        records: list[dict[str, Any]] = []
        for kind, filename in sorted(ARTIFACT_FILENAMES.items()):
            _, fingerprint = read_canonical_json(input_directory / filename)
            records.append(
                {
                    "kind": kind,
                    **fingerprint,
                    "path": f"../common-input/{filename}",
                }
            )
        _, index_fingerprint = read_canonical_json(input_directory / "artifact_index.json")
        status = force_release_status or (
            "approved" if kwargs["formal"] else "generated"
        )
        payload["config_status"] = status
        payload["release_binding"] = {
            "schema_version": (
                GAP_CENSORED_BED_RESIDENCE_INPUT_SCHEMA_VERSION
                if bed is not None and horizon_suite._payload_gap_censoring_enabled(payload)
                else BED_RESIDENCE_INPUT_SCHEMA_VERSION
                if bed is not None
                else "1.0.0"
            ),
            "source_config_template_sha256": horizon_suite._yaml_fingerprint(
                template, "common-config.yaml"
            )["sha256"],
            "source_config_hash": horizon_suite._config_hash_from_payload(
                yaml.safe_load(template.read_text(encoding="utf-8"))
            ),
            "input_directory_artifact_index_sha256": index_fingerprint["sha256"],
            "artifacts": records,
            "approved_only_after_exact_hash_validation": True,
            "arrival_selection_binding": {
                "forcing_years": horizon_suite._arrival_population_contract(payload)[
                    "forcing_years"
                ],
                "observation_years": horizon_suite._arrival_population_contract(payload)[
                    "observation_years"
                ],
                "policy": horizon_suite._arrival_population_contract(payload)[
                    "arrival_selection_policy_id"
                ],
                "replicates": horizon_suite._arrival_population_contract(payload)[
                    "replicates_per_stratum"
                ],
            },
            "backtrack_horizon_binding": {
                "source_config_hash": horizon_suite._config_hash_from_payload(
                    yaml.safe_load(template.read_text(encoding="utf-8"))
                ),
                "source_backtrack_support_days": payload["inputs"].get(
                    "backtrack_support_days"
                ) or 90,
                "artifact_backtrack_support_days": 90.0,
                "requested_max_backtrack_days": float(days),
                "requested_maximum_step_count": int(kwargs["maximum_step_count"]),
            },
        }
        if bed is not None:
            payload["release_binding"]["backtrack_horizon_binding"].update(
                {
                    "selection_support_days": 180,
                    "runtime_support_days": 90,
                    "artifact_selection_support_days": 180,
                    "artifact_runtime_support_days": 90,
                    "backtrack_mode": mode,
                }
            )
            gap_contract = horizon_suite._payload_gap_censoring_contract(payload)
            if gap_contract is not None:
                payload["release_binding"].update(gap_contract)
        payload["release_approval"] = {
            "status": status,
            "blockers": [],
            "validated_input_summary": {"test": True},
            "formal_input_validation_summary": {"test": True} if kwargs["formal"] else None,
            "public_analysis_label_policy": {"A": "A 區分析域"},
        }
        output.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return {"config_status": status}

    def fake_validate_release_config(*args: Any, **kwargs: Any) -> dict[str, Any]:
        """提供 release validator 的 JSON-safe 通過結果。"""

        calls["release_validate"].append({"args": args, **kwargs})
        return {"valid": True, "errors": [], "warnings": [], "summary": {"test": True}}

    monkeypatch.setattr(horizon_suite, "build_input_derivatives", fake_build_input_derivatives)
    monkeypatch.setattr(horizon_suite, "validate_input_derivatives", fake_validate_input_derivatives)
    monkeypatch.setattr(horizon_suite, "create_release_config", fake_create_release_config)
    monkeypatch.setattr(horizon_suite, "validate_release_config", fake_validate_release_config)
    return calls


def test_normalize_horizons_is_canonical_and_strict() -> None:
    """horizon 只接受唯一正整數，並將比較順序固定為升冪。"""

    assert horizon_suite.normalize_horizons([90, 30, 60]) == (30, 60, 90)
    for invalid in ([], [30, 30], [True], [30.0], ["30"], [0], [-1]):
        with pytest.raises(horizon_suite.HorizonSuiteError):
            horizon_suite.normalize_horizons(invalid)  # type: ignore[arg-type]


def test_horizon_step_budget_uses_dt_min_and_rejects_invalid_dt() -> None:
    """30／60／90 日步數依最小秒數計算，無效 dt 不得被默認值掩蓋。"""

    assert horizon_suite.maximum_step_count_for_horizon(30, 30.0) == 86_401
    assert horizon_suite.maximum_step_count_for_horizon(60, 30.0) == 172_801
    assert horizon_suite.maximum_step_count_for_horizon(90, 30.0) == 259_201
    for invalid_dt in (0, -1, float("nan"), True, "30"):
        with pytest.raises(horizon_suite.HorizonSuiteError):
            horizon_suite.maximum_step_count_for_horizon(30, invalid_dt)


def test_build_horizon_suite_uses_one_common_input_and_emits_identity_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """legacy pilot suite 應只建一次 input，三份 release 共用同一 component identity。"""

    calls = _install_small_suite_doubles(monkeypatch)
    template = tmp_path / "template.yaml"
    template_before = _write_template(template, support_days=7)
    destination = tmp_path / "suite"
    result = horizon_suite.build_horizon_suite(
        template,
        [90, 30, 60],
        destination,
        tmp_path / "ocm",
        tmp_path / "surface",
        tmp_path / "nww",
        formal=False,
    )

    assert result["horizons_days"] == [30, 60, 90]
    assert result["selection_support_days"] == 90
    assert result["common_input_build_count"] == 1
    assert len(calls["build"]) == 1
    assert calls["build"][0]["strict"] is True
    assert calls["build"][0]["formal"] is False
    assert len(calls["create"]) == 3
    input_directories = {call["input_directory"] for call in calls["create"]}
    assert len(input_directories) == 1
    assert next(iter(input_directories)).name == "common-input"
    assert template.read_bytes() == template_before

    manifest, _ = read_canonical_json(destination / "horizon-suite-manifest.json")
    assert manifest["horizons_days"] == [30, 60, 90]
    assert manifest["selection_support_days"] == 90
    assert manifest["source_schema_version"] == DERIVED_INPUT_SCHEMA_VERSION
    assert manifest["maximum_step_count_by_horizon"] == {
        "30": 86_401,
        "60": 172_801,
        "90": 259_201,
    }
    assert (destination / "horizon-suite-manifest.json.sha256").is_file()
    assert (destination / "validations" / "input.json.sha256").is_file()
    release_identity = []
    for days in (30, 60, 90):
        release, _ = read_canonical_json(
            destination / "validations" / f"release-{days}d.json"
        )
        assert release["valid"] is True
        assert (destination / "validations" / f"release-{days}d.json.sha256").is_file()
        payload = yaml.safe_load(
            (destination / "release-configs" / f"release-{days}d.yaml").read_text(
                encoding="utf-8"
            )
        )
        assert payload["inputs"]["ocm_gap_safe_arrival_manifest"] == (
            "../common-input/ocm_gap_safe_arrival.json"
        )
        # legacy template 可保留 schema 的 null 欄位，但不得被 suite rewriter 綁到
        # common-input；只有 current design＋approved reconstruction policy 才建立路徑。
        assert payload["inputs"].get("ocm_gap_reconstruction_manifest") is None
        release_identity.append(
            {
                kind: {
                    field: item[field]
                    for field in ("sha256", "canonical_sha256", "size_bytes")
                }
                for kind, item in (
                    (record["kind"], record)
                    for record in payload["release_binding"]["artifacts"]
                )
                if kind in {"arrival", "receptor", "material", "initial_condition"}
            }
        )
    assert release_identity[0] == release_identity[1] == release_identity[2]
    assert horizon_suite.validate_horizon_suite(destination, formal=False)["valid"] is True


def test_bed_suite_builds_one_180_day_selection_mother_and_six_mode_releases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """90 日最大沉底年齡只增加 selection envelope，不觸發第二次 input-build。"""

    calls = _install_small_suite_doubles(monkeypatch)
    template = tmp_path / "bed-template.yaml"
    _write_bed_template(template)
    destination = tmp_path / "bed-suite"
    result = horizon_suite.build_horizon_suite(
        template,
        [90, 30, 60],
        destination,
        tmp_path / "ocm",
        tmp_path / "surface",
        tmp_path / "nww",
        formal=False,
    )

    modes = (
        "fixed_calendar_window",
        "full_horizon_from_deposition",
    )
    assert result["horizons_days"] == [30, 60, 90]
    assert result["selection_support_days"] == 180
    assert result["runtime_support_days"] == 90
    assert result["backtrack_modes"] == list(modes)
    assert result["common_input_build_count"] == 1
    assert result["release_count"] == 6
    assert len(calls["build"]) == 1
    assert len(calls["create"]) == 6
    assert {call["backtrack_mode_override"] for call in calls["create"]} == set(modes)

    common_payload = yaml.safe_load(
        (destination / "common-config.yaml").read_text(encoding="utf-8")
    )
    assert common_payload["inputs"]["backtrack_support_days"] == 180
    assert common_payload["boundaries"]["max_backtrack_days"] == 90.0
    assert common_payload["scenarios"]["bed_residence_time"][
        "runtime_horizon_support_days"
    ] == 90
    manifest, manifest_path = _load_manifest(destination)
    assert manifest["selection_support_days"] == 180
    assert manifest["runtime_support_days"] == 90
    assert manifest["source_schema_version"] == GAP_CENSORED_BED_RESIDENCE_INPUT_SCHEMA_VERSION
    assert manifest["forcing_years"] == [2024, 2025]
    assert manifest["observation_years"] == [2025]
    assert manifest["arrival_selection_policy_id"] == (
        "observation_year_stratified_48_plus_2_v1"
    )
    assert manifest["replicates_per_stratum"] == 2
    assert manifest["arrival_core_count"] == 48
    assert manifest["arrival_event_count"] == 2
    assert manifest["backtrack_modes"] == list(modes)
    assert manifest["input_build_count"] == 1
    assert len(manifest["releases"]) == 6
    wrong_source_schema = copy.deepcopy(manifest)
    wrong_source_schema["source_schema_version"] = DERIVED_INPUT_SCHEMA_VERSION
    _replace_canonical_json(manifest_path, wrong_source_schema)
    wrong_schema_result = horizon_suite.validate_horizon_suite(destination, formal=False)
    assert wrong_schema_result["valid"] is False
    assert "manifest_source_schema_version_invalid" in wrong_schema_result["errors"]
    _replace_canonical_json(manifest_path, manifest)
    identities = []
    for days in (30, 60, 90):
        for mode in modes:
            stem = f"release-{days}d-{mode}"
            release_path = destination / "release-configs" / f"{stem}.yaml"
            validation_path = destination / "validations" / f"{stem}.json"
            assert release_path.is_file()
            assert validation_path.is_file()
            validation, _ = read_canonical_json(validation_path)
            assert validation["valid"] is True
            release = yaml.safe_load(release_path.read_text(encoding="utf-8"))
            assert release["scenarios"]["bed_residence_time"]["backtrack_mode"] == mode
            assert release["inputs"]["backtrack_support_days"] == 180
            assert release["boundaries"]["max_backtrack_days"] == float(days)
            assert release["inputs"]["ocm_gap_safe_arrival_manifest"] == (
                "../common-input/ocm_gap_safe_arrival.json"
            )
            assert release["inputs"]["ocm_gap_reconstruction_manifest"] == (
                release["inputs"]["ocm_gap_safe_arrival_manifest"]
            )
            horizon_binding = release["release_binding"]["backtrack_horizon_binding"]
            assert horizon_binding["source_backtrack_support_days"] == 180
            assert horizon_binding["artifact_backtrack_support_days"] == 90.0
            assert horizon_binding["selection_support_days"] == 180
            assert horizon_binding["runtime_support_days"] == 90
            assert horizon_binding["artifact_selection_support_days"] == 180
            assert horizon_binding["artifact_runtime_support_days"] == 90
            assert horizon_binding["backtrack_mode"] == mode
            identities.append(
                {
                    kind: {
                        field: item[field]
                        for field in ("sha256", "canonical_sha256", "size_bytes")
                    }
                    for kind, item in (
                        (record["kind"], record)
                        for record in release["release_binding"]["artifacts"]
                    )
                    if kind in {"arrival", "receptor", "material", "initial_condition"}
                }
            )
    assert all(identity == identities[0] for identity in identities[1:])
    assert horizon_suite.validate_horizon_suite(destination, formal=False)["valid"] is True

    # 即使同步更新 YAML fingerprint 與 manifest sidecar，將 release mode 改成另一個合法
    # 值也不能通過 common config 的 exact expected-payload 重建。
    tampered_path = destination / "release-configs" / "release-30d-full_horizon_from_deposition.yaml"
    tampered = yaml.safe_load(tampered_path.read_text(encoding="utf-8"))
    tampered["scenarios"]["bed_residence_time"]["backtrack_mode"] = modes[0]
    tampered_path.write_text(
        yaml.safe_dump(tampered, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    for record in manifest["releases"]:
        if record["config_path"] == (
            "release-configs/release-30d-full_horizon_from_deposition.yaml"
        ):
            record["config_fingerprint"] = horizon_suite._yaml_fingerprint(
                tampered_path,
                "release-configs/release-30d-full_horizon_from_deposition.yaml",
            )
    _replace_canonical_json(manifest_path, manifest)
    tamper_result = horizon_suite.validate_horizon_suite(destination, formal=False)
    assert tamper_result["valid"] is False
    assert any(
        "release_derived_payload_mismatch" in error
        or "release_horizon_binding_invalid" in error
        for error in tamper_result["errors"]
    )


def test_resume_horizon_suite_reuses_preserved_partial_without_input_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """recovery 應保留原 partial、重建六份 release，且 input builder 呼叫次數為零。"""

    calls = _install_small_suite_doubles(monkeypatch)
    template = tmp_path / "bed-template.yaml"
    _write_bed_template(template)
    initial_destination = tmp_path / "initial-suite"
    horizon_suite.build_horizon_suite(
        template,
        [30, 60, 90],
        initial_destination,
        tmp_path / "ocm",
        tmp_path / "surface",
        tmp_path / "nww",
        formal=False,
    )
    preserved_partial = tmp_path / ".recovered-suite.partial-preserved"
    shutil.copytree(initial_destination, preserved_partial)
    before = {
        path.relative_to(preserved_partial): path.read_bytes()
        for path in preserved_partial.rglob("*")
        if path.is_file()
    }
    calls["build"].clear()
    destination = tmp_path / "recovered-suite"

    result = horizon_suite.resume_horizon_suite(
        preserved_partial,
        destination,
        [90, 30, 60],
        tmp_path / "ocm",
        tmp_path / "surface",
        tmp_path / "nww",
        formal=False,
    )

    assert result["recovery_method"] == "resume_reuse_validated_common_input_v1"
    assert result["release_count"] == 6
    assert result["common_input_build_count"] == 1
    assert calls["build"] == []
    manifest, _ = _load_manifest(destination)
    assert manifest["input_build_count"] == 1
    assert manifest["recovery_method"] == "resume_reuse_validated_common_input_v1"
    assert isinstance(manifest["recovery_source_fingerprint"], dict)
    assert horizon_suite.validate_horizon_suite(destination, formal=False)["valid"] is True
    after = {
        path.relative_to(preserved_partial): path.read_bytes()
        for path in preserved_partial.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_resume_horizon_suite_rejects_common_config_tampering_before_new_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """recovery 對 common-config 竄改 fail closed，且不建立 destination。"""

    _install_small_suite_doubles(monkeypatch)
    template = tmp_path / "bed-template.yaml"
    _write_bed_template(template)
    initial_destination = tmp_path / "initial-suite"
    horizon_suite.build_horizon_suite(
        template,
        [30, 60, 90],
        initial_destination,
        tmp_path / "ocm",
        tmp_path / "surface",
        tmp_path / "nww",
        formal=False,
    )
    preserved_partial = tmp_path / ".recovered-suite.partial-tampered"
    shutil.copytree(initial_destination, preserved_partial)
    common_path = preserved_partial / "common-config.yaml"
    common_payload = yaml.safe_load(common_path.read_text(encoding="utf-8"))
    assert isinstance(common_payload, dict)
    common_payload["boundaries"]["maximum_step_count"] = 1
    common_path.write_text(
        yaml.safe_dump(common_payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    destination = tmp_path / "recovered-suite"

    with pytest.raises(horizon_suite.HorizonSuiteError, match="common config"):
        horizon_suite.resume_horizon_suite(
            preserved_partial,
            destination,
            [30, 60, 90],
            tmp_path / "ocm",
            tmp_path / "surface",
            tmp_path / "nww",
            formal=False,
        )
    assert not destination.exists()


def test_existing_destination_symlink_and_mid_build_failure_are_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """既有目錄／symlink 拒絕，第二份 release 失敗時保留 partial 供人工稽核。"""

    template = tmp_path / "template.yaml"
    _write_template(template, support_days=90)
    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "sentinel"
    sentinel.write_text("keep", encoding="utf-8")
    _install_small_suite_doubles(monkeypatch)
    with pytest.raises(FileExistsError):
        horizon_suite.build_horizon_suite(
            template, [30, 60], existing, tmp_path / "o", tmp_path / "s", tmp_path / "n", False
        )
    assert sentinel.read_text(encoding="utf-8") == "keep"

    sentinel.unlink()
    existing.rmdir()
    target = tmp_path / "target"
    existing.symlink_to(target, target_is_directory=True)
    with pytest.raises(FileExistsError):
        horizon_suite.build_horizon_suite(
            template, [30, 60], existing, tmp_path / "o", tmp_path / "s", tmp_path / "n", False
        )
    existing.unlink()

    calls = _install_small_suite_doubles(monkeypatch, fail_release_days=60)
    with pytest.raises(RuntimeError, match="60"):
        horizon_suite.build_horizon_suite(
            template, [30, 60, 90], target, tmp_path / "o", tmp_path / "s", tmp_path / "n", False
        )
    assert not target.exists()
    assert calls["build"]
    partials = list(tmp_path.glob(".target.partial-*"))
    assert len(partials) == 1
    assert (partials[0] / "source-template.yaml").is_file()


def test_validator_returns_false_for_missing_release_and_tampered_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """缺 release 或 manifest bytes 被竄改時，公開 validator 只回 valid=false。"""

    _install_small_suite_doubles(monkeypatch)
    template = tmp_path / "template.yaml"
    _write_template(template, support_days=90)
    destination = tmp_path / "suite"
    horizon_suite.build_horizon_suite(
        template, [30, 60], destination, tmp_path / "o", tmp_path / "s", tmp_path / "n", False
    )
    (destination / "release-configs" / "release-60d.yaml").unlink()
    missing = horizon_suite.validate_horizon_suite(destination, formal=False)
    assert missing["valid"] is False
    assert missing["errors"]

    # 重新建另一份 suite，再改 manifest bytes；sidecar mismatch 必須在入口被拒絕。
    destination2 = tmp_path / "suite-tampered"
    horizon_suite.build_horizon_suite(
        template, [30, 60], destination2, tmp_path / "o2", tmp_path / "s2", tmp_path / "n2", False
    )
    manifest_path = destination2 / "horizon-suite-manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["selection_support_days"] = 31
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    tampered = horizon_suite.validate_horizon_suite(destination2, formal=False)
    assert tampered["valid"] is False
    assert tampered["errors"]


def test_example_placeholder_is_allowed_but_other_binding_is_rejected_before_builder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """example 的固定 placeholder 可建置，其他 stale binding 必須在大型 I/O 前拒絕。"""

    calls = _install_small_suite_doubles(monkeypatch)
    template = tmp_path / "template.yaml"
    _write_template(template, support_days=90)
    horizon_suite.build_horizon_suite(
        template, [30], tmp_path / "allowed", tmp_path / "o", tmp_path / "s", tmp_path / "n", False
    )
    assert len(calls["build"]) == 1

    bad_payload = yaml.safe_load(template.read_text(encoding="utf-8"))
    bad_payload["geometry"]["domain_manifest"] = "stale/domain.json"
    bad_payload["release_binding"] = {"old": True}
    bad_template = tmp_path / "bad-template.yaml"
    bad_template.write_text(
        yaml.safe_dump(bad_payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    before = len(calls["build"])
    with pytest.raises(horizon_suite.HorizonSuiteError):
        horizon_suite.build_horizon_suite(
            bad_template,
            [30],
            tmp_path / "rejected",
            tmp_path / "o2",
            tmp_path / "s2",
            tmp_path / "n2",
            False,
        )
    assert len(calls["build"]) == before
    assert not (tmp_path / "rejected").exists()


def test_formal_suite_rejects_nonapproved_release_without_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """formal gate 不接受 generated release，且失敗不得留下 final。"""

    _install_small_suite_doubles(monkeypatch, force_release_status="generated")
    template = tmp_path / "template.yaml"
    _write_template(template, support_days=90)
    destination = tmp_path / "formal-suite"
    with pytest.raises(horizon_suite.HorizonSuiteError, match="approved"):
        horizon_suite.build_horizon_suite(
            template, [30, 60, 90], destination, tmp_path / "o", tmp_path / "s", tmp_path / "n", True
        )
    assert not destination.exists()
    partials = list(tmp_path.glob(".formal-suite.partial-*"))
    assert len(partials) == 1
    assert (partials[0] / "common-config.yaml").is_file()


def test_validator_rejects_pilot_release_marked_approved_after_resigning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """pilot release 即使同步重簽 config 與 manifest，也不能冒充 approved。"""

    _install_small_suite_doubles(monkeypatch)
    template = tmp_path / "template.yaml"
    _write_template(template, support_days=90)
    destination = tmp_path / "suite"
    horizon_suite.build_horizon_suite(
        template, [30], destination, tmp_path / "o", tmp_path / "s", tmp_path / "n", False
    )
    release_path = destination / "release-configs" / "release-30d.yaml"
    release_payload = yaml.safe_load(release_path.read_text(encoding="utf-8"))
    release_payload["config_status"] = "approved"
    release_payload["release_approval"]["status"] = "approved"
    release_path.write_text(
        yaml.safe_dump(release_payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    manifest, manifest_path = _load_manifest(destination)
    record = manifest["releases"][0]
    record["config_status"] = "approved"
    record["config_fingerprint"] = horizon_suite._yaml_fingerprint(
        release_path, "release-configs/release-30d.yaml"
    )
    _replace_canonical_json(manifest_path, manifest)
    result = horizon_suite.validate_horizon_suite(destination, formal=False)
    assert result["valid"] is False
    assert any("release_status_invalid" in error for error in result["errors"])


def test_validator_rejects_pilot_approval_metadata_tamper_after_resigning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """pilot 的 blockers／summary／公開標籤必須保持 builder 登錄的 exact 值。"""

    _install_small_suite_doubles(monkeypatch)
    template = tmp_path / "template.yaml"
    _write_template(template, support_days=90)
    destination = tmp_path / "pilot-approval"
    horizon_suite.build_horizon_suite(
        template, [30], destination, tmp_path / "o", tmp_path / "s", tmp_path / "n", False
    )
    release_path = destination / "release-configs" / "release-30d.yaml"
    original_release = yaml.safe_load(release_path.read_text(encoding="utf-8"))
    original_manifest, manifest_path = _load_manifest(destination)
    mutators = (
        lambda payload: payload["release_approval"].__setitem__("blockers", ["tampered"]),
        lambda payload: payload["release_approval"].__setitem__(
            "validated_input_summary", {"tampered": True}
        ),
        lambda payload: payload["release_approval"].__setitem__(
            "public_analysis_label_policy", {"A": "other"}
        ),
    )
    for mutate in mutators:
        candidate_release = copy.deepcopy(original_release)
        mutate(candidate_release)
        release_path.write_text(
            yaml.safe_dump(candidate_release, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        candidate_manifest = copy.deepcopy(original_manifest)
        candidate_manifest["releases"][0]["config_fingerprint"] = horizon_suite._yaml_fingerprint(
            release_path, "release-configs/release-30d.yaml"
        )
        _replace_canonical_json(manifest_path, candidate_manifest)
        result = horizon_suite.validate_horizon_suite(destination, formal=False)
        assert result["valid"] is False
        assert any("release_approval" in error for error in result["errors"])
    release_path.write_text(
        yaml.safe_dump(original_release, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    _replace_canonical_json(manifest_path, original_manifest)


def test_validator_rejects_formal_approval_summary_tamper_after_resigning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """formal release 的 formal input summary 被竄改並重簽時仍必須失敗。"""

    _install_small_suite_doubles(monkeypatch)
    template = tmp_path / "template.yaml"
    _write_template(template, support_days=90)
    destination = tmp_path / "formal-approval"
    horizon_suite.build_horizon_suite(
        template, [30], destination, tmp_path / "o", tmp_path / "s", tmp_path / "n", True
    )
    release_path = destination / "release-configs" / "release-30d.yaml"
    release_payload = yaml.safe_load(release_path.read_text(encoding="utf-8"))
    release_payload["release_approval"]["formal_input_validation_summary"] = {"tampered": True}
    release_path.write_text(
        yaml.safe_dump(release_payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    manifest, manifest_path = _load_manifest(destination)
    manifest["releases"][0]["config_fingerprint"] = horizon_suite._yaml_fingerprint(
        release_path, "release-configs/release-30d.yaml"
    )
    _replace_canonical_json(manifest_path, manifest)
    result = horizon_suite.validate_horizon_suite(destination, formal=True)
    assert result["valid"] is False
    assert any("release_approval_formal_summary_mismatch" in error for error in result["errors"])


def test_validator_rejects_manifest_and_release_record_field_tampering_after_resigning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """manifest 與 record 的關鍵欄位逐一改寫後仍須 fail closed。"""

    _install_small_suite_doubles(monkeypatch)
    template = tmp_path / "template.yaml"
    _write_template(template, support_days=90)
    destination = tmp_path / "suite"
    horizon_suite.build_horizon_suite(
        template, [30], destination, tmp_path / "o", tmp_path / "s", tmp_path / "n", False
    )
    manifest, manifest_path = _load_manifest(destination)
    manifest_mutators = {
        "source_schema_version": lambda value: value.__setitem__("source_schema_version", "0.0.0"),
        "dt_min_seconds": lambda value: value.__setitem__("dt_min_seconds", 31.0),
        "observation_years": lambda value: value.__setitem__("observation_years", [2025]),
        "paths": lambda value: value["paths"].__setitem__("common_input", "other-input"),
        "input_build_count": lambda value: value.__setitem__("input_build_count", True),
    }
    original = copy.deepcopy(manifest)
    for mutate in manifest_mutators.values():
        candidate = copy.deepcopy(original)
        mutate(candidate)
        _replace_canonical_json(manifest_path, candidate)
        assert horizon_suite.validate_horizon_suite(destination, formal=False)["valid"] is False
    _replace_canonical_json(manifest_path, original)

    record_mutators = {
        "maximum_step_count": lambda value: value.__setitem__("maximum_step_count", 1),
        "validation_path": lambda value: value.__setitem__("validation_path", "wrong.json"),
        "identity_fingerprints": lambda value: value.__setitem__("identity_fingerprints", {}),
        "config_status": lambda value: value.__setitem__("config_status", "approved"),
    }
    for mutate in record_mutators.values():
        candidate = copy.deepcopy(original)
        mutate(candidate["releases"][0])
        _replace_canonical_json(manifest_path, candidate)
        assert horizon_suite.validate_horizon_suite(destination, formal=False)["valid"] is False


def test_validator_rejects_common_config_drift_even_when_hashes_are_resigned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """common config 增加未登錄設定後，更新自述 fingerprint 仍不能通過。"""

    _install_small_suite_doubles(monkeypatch)
    template = tmp_path / "template.yaml"
    _write_template(template, support_days=90)
    destination = tmp_path / "suite"
    horizon_suite.build_horizon_suite(
        template, [30], destination, tmp_path / "o", tmp_path / "s", tmp_path / "n", False
    )
    common_path = destination / "common-config.yaml"
    common_payload = yaml.safe_load(common_path.read_text(encoding="utf-8"))
    common_payload["design_version"] = "unregistered-drift"
    common_path.write_text(
        yaml.safe_dump(common_payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    manifest, manifest_path = _load_manifest(destination)
    manifest["common_config_fingerprint"] = horizon_suite._yaml_fingerprint(
        common_path, "common-config.yaml"
    )
    _replace_canonical_json(manifest_path, manifest)
    result = horizon_suite.validate_horizon_suite(destination, formal=False)
    assert result["valid"] is False
    assert "common_config_derived_payload_mismatch" in result["errors"]


def test_validator_rejects_extra_topology_and_resigned_stored_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """額外檔案與被重簽的 validation 內容都不能擴張 suite 契約。"""

    _install_small_suite_doubles(monkeypatch)
    template = tmp_path / "template.yaml"
    _write_template(template, support_days=90)
    destination = tmp_path / "suite"
    horizon_suite.build_horizon_suite(
        template, [30], destination, tmp_path / "o", tmp_path / "s", tmp_path / "n", False
    )
    extra_root = destination / "unexpected.txt"
    extra_root.write_text("extra", encoding="utf-8")
    assert horizon_suite.validate_horizon_suite(destination, formal=False)["valid"] is False
    extra_root.unlink()
    extra_release = destination / "release-configs" / "unexpected.yaml"
    extra_release.write_text("x: 1\n", encoding="utf-8")
    assert horizon_suite.validate_horizon_suite(destination, formal=False)["valid"] is False
    extra_release.unlink()

    closure_path = destination / "common-input" / "artifact_bindings.json"
    closure, _ = read_canonical_json(closure_path)
    closure["artifacts"][0]["sha256"] = "0" * 64
    _replace_canonical_json(closure_path, closure)
    assert horizon_suite.validate_horizon_suite(destination, formal=False)["valid"] is False
    # 後續 validation tamper 測試需先把 closure 恢復成可驗證內容。
    _install_small_suite_doubles(monkeypatch)
    # 以原本 component closure 重新建立 suite，避免測試自行複製未驗證 provenance。
    destination = tmp_path / "suite-validation"
    horizon_suite.build_horizon_suite(
        template, [30], destination, tmp_path / "o2", tmp_path / "s2", tmp_path / "n2", False
    )

    validation_path = destination / "validations" / "release-30d.json"
    stored, _ = read_canonical_json(validation_path)
    stored["summary"]["unregistered"] = True
    _replace_canonical_json(validation_path, stored)
    manifest, manifest_path = _load_manifest(destination)
    _, validation_fp = read_canonical_json(validation_path)
    manifest["releases"][0]["validation_fingerprint"] = validation_fp
    _replace_canonical_json(manifest_path, manifest)
    result = horizon_suite.validate_horizon_suite(destination, formal=False)
    assert result["valid"] is False
    assert any("release_validation_evidence_mismatch" in error for error in result["errors"])


def test_publish_race_preserves_existing_destination_and_partial_identity_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """final collision 不覆寫 sentinel，partial 被替換時 cleanup 不刪對方目錄。"""

    template = tmp_path / "template.yaml"
    _write_template(template, support_days=90)
    destination = tmp_path / "race"
    _install_small_suite_doubles(monkeypatch)

    def collide(
        rename_function: Any,
        *,
        parent_descriptor: int,
        source_name: str,
        destination_name: str,
        rename_flags: int,
    ) -> None:
        del rename_function, parent_descriptor, source_name, rename_flags
        final = destination.parent / destination_name
        final.mkdir()
        (final / "sentinel").write_text("keep", encoding="utf-8")
        raise FileExistsError("simulated final race")

    monkeypatch.setattr(horizon_suite, "_call_exclusive_rename", collide)
    with pytest.raises(FileExistsError):
        horizon_suite.build_horizon_suite(
            template, [30], destination, tmp_path / "o", tmp_path / "s", tmp_path / "n", False
        )
    assert (destination / "sentinel").read_text(encoding="utf-8") == "keep"
    assert len(list(tmp_path.glob(".race.partial-*"))) == 1

    monkeypatch.undo()
    _install_small_suite_doubles(monkeypatch)
    replacement_path: Path | None = None

    def swap_partial(
        partial: Path,
        final: Path,
        expected_parent_identity: tuple[int, int],
        partial_identity: tuple[int, int],
    ) -> None:
        del final, expected_parent_identity, partial_identity
        replacement = partial.with_name(f"{partial.name}.replacement")
        replacement.mkdir()
        (replacement / "sentinel").write_text("replacement", encoding="utf-8")
        backup = partial.with_name(f"{partial.name}.old")
        partial.rename(backup)
        replacement.rename(partial)
        shutil.rmtree(backup)
        nonlocal replacement_path
        replacement_path = partial
        raise RuntimeError("simulated partial identity race")

    monkeypatch.setattr(horizon_suite, "_publish_partial", swap_partial)
    with pytest.raises(RuntimeError, match="identity race"):
        horizon_suite.build_horizon_suite(
            template,
            [30],
            tmp_path / "identity",
            tmp_path / "o2",
            tmp_path / "s2",
            tmp_path / "n2",
            False,
        )
    assert replacement_path is not None
    assert (replacement_path / "sentinel").read_text(encoding="utf-8") == "replacement"


def test_publish_fsync_failure_keeps_final_and_downstream_raise_is_json_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """final rename 後 parent fsync 失敗要保留成果；validator downstream raise 不外拋。"""

    _install_small_suite_doubles(monkeypatch)
    template = tmp_path / "template.yaml"
    _write_template(template, support_days=90)
    destination = tmp_path / "durability"
    original_fsync = horizon_suite._fsync_directory

    def fail_parent(path: Path, *, expected_identity: tuple[int, int] | None = None) -> None:
        if Path(path) == tmp_path:
            raise OSError("simulated parent fsync failure")
        original_fsync(path, expected_identity=expected_identity)

    monkeypatch.setattr(horizon_suite, "_fsync_directory", fail_parent)
    with pytest.raises(RuntimeError, match="已發布但 durability 未確認"):
        horizon_suite.build_horizon_suite(
            template, [30], destination, tmp_path / "o", tmp_path / "s", tmp_path / "n", False
        )
    assert destination.is_dir()
    assert list(tmp_path.glob(".durability.partial-*")) == []

    monkeypatch.undo()
    _install_small_suite_doubles(monkeypatch)
    horizon_suite.build_horizon_suite(
        template,
        [30],
        tmp_path / "downstream",
        tmp_path / "o2",
        tmp_path / "s2",
        tmp_path / "n2",
        False,
    )

    def raise_release_validator(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        raise RuntimeError("simulated validator exception")

    monkeypatch.setattr(horizon_suite, "validate_release_config", raise_release_validator)
    result = horizon_suite.validate_horizon_suite(tmp_path / "downstream", formal=False)
    assert result["valid"] is False
    assert result["errors"]


def test_yaml_snapshot_parses_and_fingerprints_one_byte_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """source snapshot 以同一份 bytes 解析與 hash，避免共享檔案系統讀取漂移。"""

    path = tmp_path / "snapshot.yaml"
    path.write_text("a: 1\n", encoding="utf-8")
    original_read = os.read
    calls = 0

    def counted_read(descriptor: int, size: int) -> bytes:
        nonlocal calls
        calls += 1
        return original_read(descriptor, size)

    monkeypatch.setattr(os, "read", counted_read)
    payload, fingerprint, raw = horizon_suite._read_yaml_snapshot(path, "snapshot", "snapshot.yaml")
    assert calls >= 1
    assert payload == {"a": 1}
    assert raw == b"a: 1\n"
    assert fingerprint["path"] == "snapshot.yaml"


def test_validator_converts_corrupt_files_and_downstream_raises_to_json_safe_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """release YAML、JSON sidecar 或下游 validator 損壞時公開入口不得外拋。"""

    _install_small_suite_doubles(monkeypatch)
    template = tmp_path / "template.yaml"
    _write_template(template, support_days=90)
    destination = tmp_path / "suite"
    horizon_suite.build_horizon_suite(
        template, [30], destination, tmp_path / "o", tmp_path / "s", tmp_path / "n", False
    )
    release_path = destination / "release-configs" / "release-30d.yaml"
    release_path.write_text("this is not a mapping\n", encoding="utf-8")
    malformed_release = horizon_suite.validate_horizon_suite(destination, formal=False)
    assert malformed_release["valid"] is False
    assert malformed_release["errors"]

    # 重新建立一份完整 suite，單獨驗證 input sidecar 與 downstream exception 的 JSON-safe 契約。
    destination2 = tmp_path / "suite-input"
    horizon_suite.build_horizon_suite(
        template, [30], destination2, tmp_path / "o2", tmp_path / "s2", tmp_path / "n2", False
    )
    (destination2 / "validations" / "input.json.sha256").write_text("{broken", encoding="utf-8")
    corrupt_sidecar = horizon_suite.validate_horizon_suite(destination2, formal=False)
    assert corrupt_sidecar["valid"] is False
    assert corrupt_sidecar["errors"]

    monkeypatch.undo()
    _install_small_suite_doubles(monkeypatch)
    horizon_suite.build_horizon_suite(
        template,
        [30],
        tmp_path / "suite-downstream",
        tmp_path / "o3",
        tmp_path / "s3",
        tmp_path / "n3",
        False,
    )

    def raise_input_validator(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        raise RuntimeError("simulated input validator exception")

    monkeypatch.setattr(horizon_suite, "validate_input_derivatives", raise_input_validator)
    downstream = horizon_suite.validate_horizon_suite(tmp_path / "suite-downstream", formal=False)
    assert downstream["valid"] is False
    assert downstream["errors"]


def test_validator_rejects_release_config_drift_after_resigning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """release 未登錄 execution 參數漂移，即使同步重簽也必須 fail closed。"""

    _install_small_suite_doubles(monkeypatch)
    template = tmp_path / "template.yaml"
    _write_template(template, support_days=90)
    destination = tmp_path / "suite"
    horizon_suite.build_horizon_suite(
        template, [30], destination, tmp_path / "o", tmp_path / "s", tmp_path / "n", False
    )
    release_path = destination / "release-configs" / "release-30d.yaml"
    release_payload = yaml.safe_load(release_path.read_text(encoding="utf-8"))
    release_payload["execution"]["checkpoint_interval_sweeps"] = 123
    release_path.write_text(
        yaml.safe_dump(release_payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    manifest, manifest_path = _load_manifest(destination)
    manifest["releases"][0]["config_fingerprint"] = horizon_suite._yaml_fingerprint(
        release_path, "release-configs/release-30d.yaml"
    )
    _replace_canonical_json(manifest_path, manifest)
    result = horizon_suite.validate_horizon_suite(destination, formal=False)
    assert result["valid"] is False
    assert any("release_derived_payload_mismatch" in error for error in result["errors"])


def test_validator_rejects_resigned_manifest_extra_key_and_partial_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """root manifest 不得加入未登錄欄位，forcing roots 也必須 all-or-none。"""

    _install_small_suite_doubles(monkeypatch)
    template = tmp_path / "template.yaml"
    _write_template(template, support_days=90)
    destination = tmp_path / "suite"
    horizon_suite.build_horizon_suite(
        template, [30], destination, tmp_path / "o", tmp_path / "s", tmp_path / "n", False
    )
    manifest, manifest_path = _load_manifest(destination)
    manifest["unregistered_scientific_claim"] = True
    _replace_canonical_json(manifest_path, manifest)
    assert horizon_suite.validate_horizon_suite(destination, formal=False)["valid"] is False

    restored = {
        key: value for key, value in manifest.items() if key != "unregistered_scientific_claim"
    }
    _replace_canonical_json(manifest_path, restored)
    result = horizon_suite.validate_horizon_suite(
        destination,
        formal=False,
        ocm_native_root=tmp_path / "only-native",
    )
    assert result["valid"] is False
    assert "forcing_roots_partial" in result["errors"]


def test_publish_descriptor_gate_rejects_replaced_partial_without_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """最後 ownership hook 後 partial basename 被換成 sentinel 時不得發布成功。"""

    _install_small_suite_doubles(monkeypatch)
    template = tmp_path / "template.yaml"
    _write_template(template, support_days=90)
    destination = tmp_path / "race-descriptor"
    original_assert = horizon_suite._assert_owned_partial
    replacement_path: Path | None = None

    def swap_after_identity(partial: Path, identity: tuple[int, int]) -> None:
        nonlocal replacement_path
        original_assert(partial, identity)
        backup = partial.with_name(f"{partial.name}.owned")
        partial.rename(backup)
        partial.mkdir()
        (partial / "attacker-sentinel").write_text("foreign", encoding="utf-8")
        replacement_path = partial
        # owned backup is intentionally left for the builder's safe cleanup path; the
        # replacement at the original basename must never be recursively deleted.
        del backup

    monkeypatch.setattr(horizon_suite, "_assert_owned_partial", swap_after_identity)
    with pytest.raises(horizon_suite.HorizonSuiteError, match="partial"):
        horizon_suite.build_horizon_suite(
            template,
            [30],
            destination,
            tmp_path / "o",
            tmp_path / "s",
            tmp_path / "n",
            False,
        )
    assert not destination.exists()
    assert replacement_path is not None
    assert (replacement_path / "attacker-sentinel").read_text(encoding="utf-8") == "foreign"


def test_failure_cleanup_never_removes_foreign_partial_or_owned_partial(
    tmp_path: Path,
) -> None:
    """失敗清理不刪任何 partial，foreign sentinel 與本次 partial 都保留稽核。"""

    partial = tmp_path / ".manual.partial-test"
    partial.mkdir()
    (partial / "owned-marker").write_text("owned", encoding="utf-8")
    identity = horizon_suite._directory_identity(partial)
    horizon_suite._cleanup_partial(partial, identity)
    assert (partial / "owned-marker").read_text(encoding="utf-8") == "owned"

    foreign = tmp_path / ".foreign.partial-test"
    foreign.mkdir()
    (foreign / "foreign-sentinel").write_text("keep", encoding="utf-8")
    foreign_identity = (0, 0)
    horizon_suite._cleanup_partial(foreign, foreign_identity)
    assert (foreign / "foreign-sentinel").read_text(encoding="utf-8") == "keep"


def test_parent_replacement_after_publish_is_not_reported_as_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """rename 後 caller parent inode 被替換時保留 final 並回 durability 未確認。"""

    _install_small_suite_doubles(monkeypatch)
    template = tmp_path / "template.yaml"
    _write_template(template, support_days=90)
    workspace = tmp_path / "parent-root"
    workspace.mkdir()
    destination = workspace / "parent-race"
    original_publish = horizon_suite._publish_partial
    original_parent = destination.parent
    replacement_parent = tmp_path / "parent-race-old"

    def publish_then_swap(
        partial: Path,
        final: Path,
        parent_identity: tuple[int, int],
        partial_identity: tuple[int, int],
    ) -> None:
        original_publish(partial, final, parent_identity, partial_identity)
        original_parent.rename(replacement_parent)
        original_parent.mkdir()

    monkeypatch.setattr(horizon_suite, "_publish_partial", publish_then_swap)
    with pytest.raises(RuntimeError, match="已發布但 durability 未確認"):
        horizon_suite.build_horizon_suite(
            template,
            [30],
            destination,
            tmp_path / "o",
            tmp_path / "s",
            tmp_path / "n",
            False,
        )
    assert not destination.exists()
    assert (replacement_parent / "parent-race").is_dir()


def test_yaml_snapshot_rejects_symlink_leaf_without_path_read_race(tmp_path: Path) -> None:
    """YAML leaf 是 symlink 時 no-follow descriptor 必須直接拒絕。"""

    target = tmp_path / "target.yaml"
    target.write_text("a: 1\n", encoding="utf-8")
    link = tmp_path / "link.yaml"
    link.symlink_to(target)
    with pytest.raises(horizon_suite.HorizonSuiteError):
        horizon_suite._read_yaml_snapshot(link, "snapshot", "link.yaml")
