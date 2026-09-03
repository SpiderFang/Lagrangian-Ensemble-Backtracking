"""pilot calibration 的純函式與 artifact 邊界測試。

本測試檔不建立 accepted OCM/NWW product，也不模擬 SERVER 端到端資料流；只用記憶體中的
dict／Arrow table 驗證固定 schema、stable selection、統計公式、artifact validator 與
隔離的 atomic publish boundary。真實 OCM pilot 由 SERVER runbook 另行執行。
"""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import lagrangian_backtracking.pilot_calibration as pilot_calibration
from lagrangian_backtracking.models import SampleQC
from lagrangian_backtracking.pilot_calibration import (
    _PILOT_SAMPLE_CAP_M2PS,
    _QUANTILE_LEVELS,
    PAIR_SAMPLE_SCHEMA,
    _build_time_limit_candidates,
    _candidate,
    _order_pairs_for_execution,
    _pilot_pair_execution_sort_key,
    _select_pairs,
    _statistic,
    _validate_statistic,
    validate_pilot_calibration,
)


@dataclass(frozen=True)
class _Pair:
    """僅供 stable selection 單元測試的 identity stub，不代表 OCM product。"""

    study_site_id: str
    receptor_id: str
    arrival_time_id: str
    time_utc_ns: int
    flow_domain_id: str = "domain-a"
    ocm_month_yyyymm: str = "202501"


def test_pair_schema_separates_qc_from_nullable_physical_values() -> None:
    """QC 必須非空，而速度／擴散數值可用 null 表示沒有科學值。"""

    assert PAIR_SAMPLE_SCHEMA.field("ocm_qc").nullable is False
    assert PAIR_SAMPLE_SCHEMA.field("smag_cs_010_qc").nullable is False
    assert PAIR_SAMPLE_SCHEMA.field("u_mps").nullable is True
    assert PAIR_SAMPLE_SCHEMA.field("smag_cs_020_kh_m2ps").nullable is True
    assert str(PAIR_SAMPLE_SCHEMA.field("time_utc_ns").type) == "int64"


def test_quantile_and_sampling_cap_constants_are_fixed() -> None:
    """校準 quantile 與 raw Kh 取樣上限必須符合已核定資料契約。"""

    assert _QUANTILE_LEVELS == (
        0.0,
        0.005,
        0.01,
        0.05,
        0.25,
        0.5,
        0.75,
        0.95,
        0.99,
        0.995,
        1.0,
    )
    assert sys.float_info.max == _PILOT_SAMPLE_CAP_M2PS


def test_stable_pair_selection_is_independent_of_input_order() -> None:
    """同一 identity 集合換列次序後，stable SHA-256 抽樣結果不可改變。"""

    pairs = [
        _Pair("site-a", f"r{index}", f"a{index}", 1_700_000_000_000_000_000 + index)
        for index in range(8)
    ]
    first, available_first, selected_first = _select_pairs(
        pairs, pair_limit_per_site=3
    )
    second, available_second, selected_second = _select_pairs(
        list(reversed(pairs)), pair_limit_per_site=3
    )
    assert [(item.receptor_id, item.arrival_time_id) for item in first] == [
        (item.receptor_id, item.arrival_time_id) for item in second
    ]
    assert available_first == available_second == {"site-a": 8}
    assert selected_first == selected_second == {"site-a": 3}


def test_execution_order_preserves_stable_selection_set_and_groups_ocm_months() -> None:
    """locality 排序只改已選 pair 的執行順序，不改 stable-hash 選取集合。"""

    pairs = [
        _Pair(
            "site-a",
            f"r{index}",
            f"a{index}",
            1_700_000_000_000_000_000 + index,
            flow_domain_id="domain-a" if index % 2 == 0 else "domain-b",
            ocm_month_yyyymm="202501" if index % 2 == 0 else "202502",
        )
        for index in range(6)
    ]
    selected, _available, _selected_counts = _select_pairs(pairs, pair_limit_per_site=3)
    selected_reversed, _, _ = _select_pairs(list(reversed(pairs)), pair_limit_per_site=3)
    selected_identity = {(item.receptor_id, item.arrival_time_id) for item in selected}
    reversed_identity = {(item.receptor_id, item.arrival_time_id) for item in selected_reversed}
    assert selected_identity == reversed_identity

    execution = _order_pairs_for_execution(selected)
    reversed_execution = _order_pairs_for_execution(selected_reversed)
    execution_keys = [_pilot_pair_execution_sort_key(item) for item in execution]
    reversed_execution_keys = [_pilot_pair_execution_sort_key(item) for item in reversed_execution]
    assert execution_keys == reversed_execution_keys
    assert execution_keys == sorted(execution_keys)
    assert [(item.flow_domain_id, item.ocm_month_yyyymm) for item in execution] == sorted(
        (item.flow_domain_id, item.ocm_month_yyyymm) for item in execution
    )


@pytest.mark.parametrize(
    ("field", "value", "expected_message"),
    [
        ("study_site_id", "", "study_site_id"),
        ("flow_domain_id", "", "flow_domain_id"),
        ("receptor_id", "", "receptor_id"),
        ("arrival_time_id", "", "arrival_time_id"),
        ("ocm_month_yyyymm", "2025", "YYYYMM"),
        ("ocm_month_yyyymm", "202513", "YYYYMM"),
        ("ocm_month_yyyymm", "", "ocm_month_yyyymm"),
        ("time_utc_ns", "1700000000", "time_utc_ns"),
    ],
)
def test_execution_sort_key_rejects_invalid_pair_contract(
    field: str,
    value: object,
    expected_message: str,
) -> None:
    """locality key 不得替缺失 identity、非法月份或非整數 UTC 猜測 fallback。"""

    pair = _Pair("site-a", "receptor-a", "arrival-a", 1_700_000_000_000_000_000)
    invalid_pair = replace(pair, **{field: value})
    with pytest.raises(ValueError, match=expected_message):
        _pilot_pair_execution_sort_key(invalid_pair)


def test_empty_quantile_is_unavailable_and_zero_is_not_missing() -> None:
    """空集合回傳 null；有效的零值則保留為數值而非被誤判為缺值。"""

    empty = _statistic([])
    assert empty["n"] == 0
    assert empty["q50"] is None
    zero = _candidate(
        values=[0.0],
        rule="unit-test",
        metric="unit-test",
        quantile_level=0.5,
        reason_if_empty="empty",
    )
    assert zero["available"] is True
    assert zero["value"] == 0.0


def test_report_statistic_round_trip_is_monotonic() -> None:
    """validator 可重算正常 quantile，且不因 adjacent-pair 檢查誤報。"""

    errors: list[str] = []
    expected = _statistic([1.0, 2.0, 5.0])
    _validate_statistic(expected, [1.0, 2.0, 5.0], errors, "unit")
    assert errors == []


def test_time_limit_formula_excludes_zero_speed_from_finite_quantile() -> None:
    """零速度項應計入 advection unlimited_count，不可製造有限的 quantile。"""

    rows = [
        {
            "speed_mps": 2.0,
            "horizontal_scale_m": 8.0,
            "w_mps": 1.0,
            "vertical_scale_m": 4.0,
        },
        {
            "speed_mps": 0.0,
            "horizontal_scale_m": 8.0,
            "w_mps": 0.0,
            "vertical_scale_m": 4.0,
        },
    ]
    candidates = {
        "constant_kz_m2ps": {"available": False, "value": None},
        "constant_kh_m2ps": {"available": False, "value": None},
    }
    result = _build_time_limit_candidates(rows, candidates, max_settling=1.0)
    horizontal = result["horizontal_advection"]["statistics"]
    vertical = result["vertical_advection"]["statistics"]
    assert horizontal["n"] == 1
    assert horizontal["q50"] == 1.0
    assert horizontal["unlimited_count"] == 1
    assert vertical["n"] == 2
    assert vertical["unlimited_count"] == 0
    assert result["horizontal_diffusion"]["statistics"]["n"] == 0
    assert result["vertical_diffusion"]["statistics"]["n"] == 0


def test_diffusion_time_limits_use_independent_axis_formulas() -> None:
    """新 1.1.0 契約必須以各軸尺度與各軸 K 個別計算 diffusion candidate。"""

    result = _build_time_limit_candidates(
        [{"horizontal_scale_m": 8.0, "vertical_scale_m": 4.0}],
        {
            "constant_kh_m2ps": {"available": True, "value": 2.0},
            "constant_kz_m2ps": {"available": True, "value": 1.0},
        },
        max_settling=0.0,
    )
    horizontal = result["horizontal_diffusion"]
    vertical = result["vertical_diffusion"]
    assert horizontal["formula"] == "(0.25*horizontal_scale_m)^2/(2*constant_kh_m2ps)"
    assert vertical["formula"] == "(0.25*vertical_scale_m)^2/(2*constant_kz_m2ps)"
    assert horizontal["statistics"]["n"] == 1
    assert horizontal["statistics"]["q50"] == 1.0
    assert vertical["statistics"]["n"] == 1
    assert vertical["statistics"]["q50"] == 0.5
    assert horizontal["unavailable_reason"] is None
    assert vertical["unavailable_reason"] is None
    assert "diffusion" not in result


def _hand_pair_row(index: int) -> dict[str, object]:
    """建立不依賴 OCM 檔案的最小有效 pair row，供 validator 重算測試使用。"""

    row: dict[str, object] = {field.name: None for field in PAIR_SAMPLE_SCHEMA}
    row.update(
        {
            "receptor_id": f"receptor-{index}",
            "arrival_time_id": f"arrival-{index}",
            "study_site_id": "site-a",
            "analysis_region_id": "region-a",
            "flow_domain_id": "domain-a",
            "time_utc_ns": 1_735_689_600_000_000_000 + index,
            "ocm_month_yyyymm": "202501",
            "ocm_source_time_index": index,
            "ocm_time_origin": "2025-01-01T00:00:00Z",
            "vertical_id": "mid",
            "z_m_positive_up": -4.0,
            "eta_m_positive_up": 0.0,
            "bed_z_m_positive_up": -10.0,
            "water_column_height_m": 10.0,
            "height_above_bed_m": 6.0,
            "zcor_lower_m_positive_up": -5.0,
            "zcor_upper_m_positive_up": -3.0,
            "vertical_bracket_alpha": 0.5,
            "source_face_local_index": 2,
            "source_face_global_index": 12,
            "wetdry_elem_value": 1,
            "wetdry_semantics_id": "wetdry_elem_1_wet",
            "ocm_qc": int(SampleQC.OK),
            "u_mps": 2.0 + index,
            "v_mps": 0.0,
            "w_mps": 1.0,
            "max_abs_settling_velocity_mps": 0.5,
            "speed_mps": 2.0 + index,
            "horizontal_scale_m": 8.0,
            "vertical_scale_m": 4.0,
            "ocm_sampled_kz_m2ps": 1.0 + index,
        }
    )
    for token, value in (("010", 1.0), ("015", 2.0), ("020", 3.0)):
        prefix = f"smag_cs_{token}"
        row.update(
            {
                f"{prefix}_qc": int(SampleQC.OK),
                f"{prefix}_kh_m2ps": value,
                f"{prefix}_raw_current_triangle_kh_m2ps": value + 1.0,
                f"{prefix}_d_kh_dx_mps": 0.0,
                f"{prefix}_d_kh_dy_mps": 0.0,
                f"{prefix}_floor_hit": False,
                f"{prefix}_cap_hit": False,
                f"{prefix}_triangle_id": 12,
                f"{prefix}_forcing_month_id": "202501",
            }
        )
    return row


def _hand_report(
    rows: list[dict[str, object]],
    *,
    schema_version: str = pilot_calibration.PILOT_CALIBRATION_SCHEMA_VERSION,
) -> dict[str, object]:
    """依記憶體 rows 組出最小完整 report，避免透過 builder 偽造 accepted product。"""

    metrics = {
        "speed_mps": pilot_calibration._valid_metric(rows, "speed_mps", "ocm_qc"),
        "horizontal_scale_m": pilot_calibration._valid_metric(
            rows, "horizontal_scale_m", "ocm_qc"
        ),
        "vertical_scale_m": pilot_calibration._valid_metric(rows, "vertical_scale_m", "ocm_qc"),
        "ocm_sampled_kz_m2ps": pilot_calibration._valid_metric(
            rows, "ocm_sampled_kz_m2ps", "ocm_qc"
        ),
    }
    for token in ("010", "015", "020"):
        metrics[f"smag_cs_{token}_kh_m2ps"] = pilot_calibration._valid_metric(
            rows, f"smag_cs_{token}_kh_m2ps", f"smag_cs_{token}_qc"
        )
        metrics[f"smag_cs_{token}_raw_current_triangle_kh_m2ps"] = pilot_calibration._valid_metric(
            rows,
            f"smag_cs_{token}_raw_current_triangle_kh_m2ps",
            f"smag_cs_{token}_qc",
        )
    statistics = {key: pilot_calibration._statistic(value) for key, value in metrics.items()}
    constant_kz = pilot_calibration._candidate(
        values=metrics["ocm_sampled_kz_m2ps"],
        rule="pooled q50 of valid OCM sampled Kz",
        metric="ocm_sampled_kz_m2ps",
        quantile_level=0.5,
        reason_if_empty="no valid OCM sampled Kz",
    )
    constant_kh = pilot_calibration._candidate(
        values=metrics["smag_cs_015_kh_m2ps"],
        rule="pooled q50 of valid Smagorinsky particle Kh at Cs=0.15",
        metric="smag_cs_015_kh_m2ps",
        quantile_level=0.5,
        reason_if_empty="no valid Smagorinsky Kh at Cs=0.15",
    )
    floor = {
        "available": bool(metrics["smag_cs_015_kh_m2ps"]),
        "value": 0.0 if metrics["smag_cs_015_kh_m2ps"] else None,
        "unit": "m2/s",
        "rule": "fixed floor candidate 0.0 after at least one valid Cs=0.15 sample",
        "source_metric": "smag_cs_015_kh_m2ps",
        "quantile_level": None,
        "reason": (
            "valid Cs=0.15 samples available"
            if metrics["smag_cs_015_kh_m2ps"]
            else "no valid Cs=0.15 Kh"
        ),
    }
    cap = pilot_calibration._candidate(
        values=metrics["smag_cs_020_raw_current_triangle_kh_m2ps"],
        rule="pooled q99.5 of valid raw current-triangle Kh at Cs=0.20",
        metric="smag_cs_020_raw_current_triangle_kh_m2ps",
        quantile_level=0.995,
        reason_if_empty="no valid raw current-triangle Kh at Cs=0.20",
        floor=floor["value"],
    )
    candidates = {
        "constant_kz_m2ps": constant_kz,
        "constant_kh_m2ps": constant_kh,
        "floor_m2ps": floor,
        "cap_m2ps": cap,
    }
    counts = pilot_calibration._build_counts(rows, {"site-a": 2}, {"site-a": 2}, 2, 2)
    time_limits = _build_time_limit_candidates(
        rows,
        {"constant_kh_m2ps": constant_kh, "constant_kz_m2ps": constant_kz},
        0.5,
        schema_version=schema_version,
    )
    return {
        "schema_version": schema_version,
        "artifact_kind": pilot_calibration.PILOT_CALIBRATION_ARTIFACT_KIND,
        "evidence_class": "server_real_data_pilot_candidate",
        "source_status": "approved",
        "engineering_measurement_not_scientific_result": True,
        "ocm_only": True,
        "nww_accessed": False,
        "recommendation_status": "candidate_pending_trajectory_convergence_and_scientific_validation",
        "completion_status": "partial_engineering_sample",
        "selection_policy": {
            "algorithm_id": "per_site_sha256_pair_identity_sort_v1",
            "pair_limit_per_site": 2,
            "available_pair_count": 2,
            "selected_pair_count": 2,
            "status": "partial_engineering_sample",
            "full_design": {
                "receptor_count": 100,
                "arrival_count": 250,
                "pair_count": 5_000,
                "pairs_per_site": 1_000,
            },
        },
        "input_binding": {
            "config_hash": "a" * 64,
            "input_artifact_index_sha256": "b" * 64,
        },
        "code_provenance": {},
        "counts": counts,
        "statistics": statistics,
        "candidates": candidates,
        "time_limit_candidates": time_limits,
        "resource": {
            "elapsed_seconds": 0.0,
            "manager_count": 0,
            "sample_call_count": 8,
            "ocm_load_count": 0,
            "ocm_cache_hit_count": 0,
            "ocm_cache_miss_count": 0,
            "nww_load_count": 0,
            "nww_cache_hit_count": 0,
            "nww_cache_miss_count": 0,
            "eviction_count": 0,
            "resident_month_count": 0,
            "resident_ndarray_bytes": 0,
        },
    }


def _write_hand_artifact(
    root: Path,
    table: pa.Table,
    report: dict[str, object],
) -> None:
    """將記憶體中的 table/report 封裝成測試用 immutable topology。

    這個 helper 只寫入兩筆小型 Arrow row 與其 checksum，目的是讓公開 validator／reader
    真正走過 manifest、payload hash、schema fingerprint 與 report semantic dispatch；它不
    建立任何 accepted OCM/NWW product，也不代表正式校準結果。
    """

    root.mkdir()
    pair_path = root / "pair_samples.parquet"
    report_path = root / "calibration_report.json"
    pq.write_table(table, pair_path, compression="zstd", use_dictionary=False)
    pilot_calibration._write_json(report_path, report)
    manifest = {
        "schema_version": report["schema_version"],
        "artifact_kind": pilot_calibration.PILOT_CALIBRATION_ARTIFACT_KIND,
        "closure_policy": "exact_root_files_v1",
        "root_files": list(pilot_calibration.PILOT_CALIBRATION_FILES),
        "files": {
            "pair_samples.parquet": {
                "size_bytes": pair_path.stat().st_size,
                "sha256": pilot_calibration.sha256_file(pair_path),
                "row_count": table.num_rows,
                "schema_sha256": pilot_calibration._sha256_schema(PAIR_SAMPLE_SCHEMA),
            },
            "calibration_report.json": {
                "size_bytes": report_path.stat().st_size,
                "sha256": pilot_calibration.sha256_file(report_path),
            },
        },
    }
    pilot_calibration._write_json(root / "manifest.json", manifest)


def test_validator_recomputes_hand_table_statistics_candidates_and_group_counts() -> None:
    """validator 必須由 Arrow rows 重算 report，不能只信任報告內的摘要數字。"""

    table = pa.Table.from_pylist([_hand_pair_row(0), _hand_pair_row(1)], schema=PAIR_SAMPLE_SCHEMA)
    rows = table.to_pylist()
    report = _hand_report(rows)
    errors: list[str] = []
    pilot_calibration._validate_report_against_rows(report, rows, errors)
    assert errors == []

    tampered = copy.deepcopy(report)
    tampered["candidates"]["constant_kh_m2ps"]["value"] = 999.0
    tampered_errors: list[str] = []
    pilot_calibration._validate_report_against_rows(tampered, rows, tampered_errors)
    assert "candidate_contract_mismatch:constant_kh_m2ps" in tampered_errors


def test_validator_recomputes_new_axis_time_limits_and_detects_tamper() -> None:
    """1.1.0 validator 必須重算雙軸公式，竄改任一軸統計時不可放行。"""

    rows = [_hand_pair_row(0), _hand_pair_row(1)]
    report = _hand_report(rows)
    assert report["schema_version"] == "1.1.0"
    assert set(report["time_limit_candidates"]) == {
        "max_abs_settling_velocity_mps",
        "horizontal_advection",
        "vertical_advection",
        "horizontal_diffusion",
        "vertical_diffusion",
    }
    errors: list[str] = []
    pilot_calibration._validate_report_against_rows(report, rows, errors)
    assert errors == []

    tampered = copy.deepcopy(report)
    tampered["time_limit_candidates"]["horizontal_diffusion"]["statistics"]["q50"] = 999.0
    tampered_errors: list[str] = []
    pilot_calibration._validate_report_against_rows(tampered, rows, tampered_errors)
    assert "time_limit_candidates_mismatch" in tampered_errors


def test_time_limit_candidates_keep_the_other_axis_when_one_k_is_unavailable() -> None:
    """Kh／Kz 個別不可用時，只對應軸為 unavailable，另一軸仍保留數值統計。"""

    rows = [_hand_pair_row(0)]
    horizontal_unavailable = _build_time_limit_candidates(
        rows,
        {
            "constant_kh_m2ps": {"available": False, "value": None},
            "constant_kz_m2ps": {"available": True, "value": 1.0},
        },
        0.5,
    )
    assert horizontal_unavailable["horizontal_diffusion"]["statistics"]["n"] == 0
    assert horizontal_unavailable["horizontal_diffusion"]["unavailable_reason"]
    assert horizontal_unavailable["vertical_diffusion"]["statistics"]["q50"] == 0.5
    assert horizontal_unavailable["vertical_diffusion"]["unavailable_reason"] is None

    vertical_unavailable = _build_time_limit_candidates(
        rows,
        {
            "constant_kh_m2ps": {"available": True, "value": 2.0},
            "constant_kz_m2ps": {"available": False, "value": None},
        },
        0.5,
    )
    assert vertical_unavailable["vertical_diffusion"]["statistics"]["n"] == 0
    assert vertical_unavailable["vertical_diffusion"]["unavailable_reason"]
    assert vertical_unavailable["horizontal_diffusion"]["statistics"]["q50"] == 1.0
    assert vertical_unavailable["horizontal_diffusion"]["unavailable_reason"] is None


def test_public_validator_and_reader_keep_legacy_combined_diffusion_contract(tmp_path: Path) -> None:
    """公開 validator／reader 應依 1.0.0 重算舊 combined diffusion，而非套用新公式。"""

    rows = [_hand_pair_row(0), _hand_pair_row(1)]
    table = pa.Table.from_pylist(rows, schema=PAIR_SAMPLE_SCHEMA)
    report = _hand_report(
        rows,
        schema_version="1.0.0",
    )
    root = tmp_path / "legacy-pilot-calibration"
    _write_hand_artifact(root, table, report)

    validation = validate_pilot_calibration(root)
    assert validation["valid"] is True
    assert validation["summary"]["schema_version"] == "1.0.0"
    artifact = pilot_calibration.read_pilot_calibration(root)
    assert artifact.report["schema_version"] == "1.0.0"
    assert "diffusion" in artifact.report["time_limit_candidates"]
    assert "horizontal_diffusion" not in artifact.report["time_limit_candidates"]


def test_build_publishes_with_exactly_one_rename(monkeypatch, tmp_path) -> None:
    """隔離 builder 的 publish boundary，確保成功 rename 後不重複操作 final。"""

    class _Config:
        config_status = "approved"
        execution = SimpleNamespace(max_resident_forcing_months=2)

        @staticmethod
        def config_hash() -> str:
            return "a" * 64

    class _Scenario:
        initial_conditions = ()
        receptors = ()
        arrival_times = ()
        materials = ()

    class _Geometry:
        projections = {}

    monkeypatch.setattr(pilot_calibration, "load_config", lambda *args, **kwargs: _Config())
    monkeypatch.setattr(
        pilot_calibration,
        "validate_release_config",
        lambda *args, **kwargs: {"valid": True, "summary": {"config_status": "approved"}},
    )
    monkeypatch.setattr(pilot_calibration, "load_scenario_inputs", lambda *args, **kwargs: _Scenario())
    monkeypatch.setattr(pilot_calibration, "load_boundary_geometries", lambda *args, **kwargs: _Geometry())
    monkeypatch.setattr(pilot_calibration, "_input_binding", lambda *args, **kwargs: {})
    monkeypatch.setattr(pilot_calibration, "_build_report", lambda **kwargs: {"test": True})

    input_directory = tmp_path / "input"
    input_directory.mkdir()
    destination = tmp_path / "calibration"
    rename_count = 0
    real_replace = pilot_calibration.os.replace

    def track_replace(source: str | Path, target: str | Path) -> None:
        nonlocal rename_count
        rename_count += 1
        assert not Path(target).exists()
        real_replace(source, target)

    monkeypatch.setattr(pilot_calibration.os, "replace", track_replace)
    result = pilot_calibration.build_pilot_calibration(
        config_path=tmp_path / "config.yaml",
        input_directory=input_directory,
        ocm_native_root=tmp_path / "accepted-ocm",
        destination=destination,
        project_root=tmp_path,
    )
    assert result == destination
    assert destination.is_dir()
    assert rename_count == 1


def test_validator_fails_closed_without_a_complete_artifact(tmp_path) -> None:
    """空目錄與缺少 payload 時必須 invalid，不得以空結果當作校準成功。"""

    result = validate_pilot_calibration(tmp_path)
    assert result["valid"] is False
    assert "root_files_invalid" in result["errors"]


def test_sample_qc_zero_is_the_only_valid_code() -> None:
    """測試報告使用既有 QC 旗標語意，避免把其他整數當作有效樣本。"""

    assert int(SampleQC.OK) == 0
    assert int(SampleQC.TIME_GAP) != int(SampleQC.OK)
