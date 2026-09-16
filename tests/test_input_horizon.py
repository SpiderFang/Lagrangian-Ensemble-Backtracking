"""共同回溯母體的天數參數、時間窗與 gap 語意驗證測試。

測試使用小型、完全人工建立的 hourly metadata，不讀取 SERVER forcing；目的在驗證
37 日等非預列天數、閏年跨月算術，以及即使重新計算 artifact hash 也無法用偽造
``missing_utc`` 或短 row 掩蓋 canonical 時間缺口。
"""

from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import lagrangian_backtracking.input_derivation as input_derivation_module
from lagrangian_backtracking.input_horizon import (
    BED_RESIDENCE_HORIZON_METHOD_ID,
    BED_RESIDENCE_HORIZON_POLICY_ID,
    BED_RESIDENCE_INPUT_SCHEMA_VERSION,
    GENERIC_HORIZON_METHOD_ID,
    GENERIC_HORIZON_POLICY_ID,
    LEGACY_INPUT_SCHEMA_VERSION,
    HorizonContractError,
    build_horizon_window,
    compute_horizon_coverage,
    resolve_configured_horizon,
    utc_string,
    validate_generic_gap_payload,
    validate_support_days,
)


def _utc_ns(value: str) -> int:
    """把測試用的 UTC Z 字串轉成整數 epoch nanoseconds。"""

    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return int(parsed.timestamp()) * 1_000_000_000


def _gap_inventory(*, gap: bool = False) -> dict:
    """建立一個含單一 OCM region 的 metadata-only forcing inventory。"""

    start = _utc_ns("2024-01-01T00:00:00Z")
    end = _utc_ns("2024-02-29T23:00:00Z")
    gaps = (
        [
            {
                "before_utc": "2024-02-10T00:00:00Z",
                "after_utc": "2024-02-10T03:00:00Z",
                "gap_hours": 3.0,
                "missing_step_count": 2,
            }
        ]
        if gap
        else []
    )
    return {
        "expected_period": {
            "start_utc": utc_string(start),
            "end_utc": utc_string(end),
            "hourly_step_count": 1440,
        },
        "products": [
            {
                "analysis_region_id": "C",
                "flow_domain_id": "flow-c",
                "product": "ocm_native",
                "canonical_time": {
                    "canonical_time_count": 1440 - (2 if gap else 0),
                    "time_start_utc": utc_string(start),
                    "time_end_utc": utc_string(end),
                    "gaps": gaps,
                },
            }
        ],
    }


def _generic_payloads(*, gap: bool = False) -> tuple[dict, dict, dict]:
    """建立可供 generic validator 逐筆重算的最小 arrival/gap/inventory。"""

    arrival_ns = _utc_ns("2024-02-15T00:00:00Z")
    arrival = {
        "arrival_time_id": "arrival-c-1",
        "study_site_id": "houwan",
        "analysis_region_id": "C",
        "time_utc_ns": arrival_ns,
    }
    window = build_horizon_window(
        arrival_ns,
        7,
        expected_start_ns=_utc_ns("2024-01-01T00:00:00Z"),
        expected_end_ns=_utc_ns("2024-02-29T23:00:00Z"),
    )
    inventory = _gap_inventory(gap=gap)
    coverage = compute_horizon_coverage(
        window,
        expected_start_ns=_utc_ns("2024-01-01T00:00:00Z"),
        expected_end_ns=_utc_ns("2024-02-29T23:00:00Z"),
        canonical_start_ns=_utc_ns("2024-01-01T00:00:00Z"),
        canonical_end_ns=_utc_ns("2024-02-29T23:00:00Z"),
        canonical_gaps=inventory["products"][0]["canonical_time"]["gaps"],
    )
    gap_row = {
        "arrival_time_id": arrival["arrival_time_id"],
        "study_site_id": arrival["study_site_id"],
        "analysis_region_id": arrival["analysis_region_id"],
        "flow_domain_id": "flow-c",
        "arrival_time_utc": utc_string(arrival_ns),
        "horizon_start_utc": utc_string(window.start_time_ns),
        "horizon_end_utc": utc_string(window.end_time_ns),
        "max_backtrack_days": 7,
        "support_days": 7,
        "expected_step_count": window.expected_step_count,
        "supported_step_count": coverage.supported_step_count,
        "crossed_gap": coverage.crossed_gap,
        "missing_utc": [utc_string(value) for value in coverage.missing_time_ns],
    }
    gap_payload = {
        "schema_version": LEGACY_INPUT_SCHEMA_VERSION,
        "policy": GENERIC_HORIZON_POLICY_ID,
        "max_backtrack_days": 7,
        "support_days": 7,
        "provenance": {"method_id": GENERIC_HORIZON_METHOD_ID},
        "records": [gap_row],
    }
    return {"records": [arrival]}, gap_payload, inventory


def test_validate_support_days_accepts_non_prelisted_positive_integer() -> None:
    """支援窗只受整數與 int64 算術限制，不枚舉 7／30／60。"""

    assert validate_support_days(37) == 37
    assert validate_support_days(30) == 30
    with pytest.raises(HorizonContractError):
        validate_support_days(0)
    with pytest.raises(HorizonContractError):
        validate_support_days(7.0)


def test_shared_window_spans_leap_february_without_numpy_allocation() -> None:
    """30 日窗口跨閏年二月時仍有精確的 721 個 inclusive 節點。"""

    arrival = _utc_ns("2024-03-02T00:00:00Z")
    window = build_horizon_window(
        arrival,
        30,
        expected_start_ns=_utc_ns("2024-01-01T00:00:00Z"),
        expected_end_ns=_utc_ns("2024-03-31T23:00:00Z"),
    )
    assert window.expected_step_count == 721
    assert utc_string(window.start_time_ns) == "2024-02-01T00:00:00Z"
    assert utc_string(window.end_time_ns) == "2024-03-02T00:00:00Z"


def test_shared_window_rejects_support_outside_expected_period_before_iteration() -> None:
    """資料期不足時在建立窗口階段拒絕，避免先配置超長時軸。"""

    with pytest.raises(HorizonContractError, match="超出 expected forcing period"):
        build_horizon_window(
            _utc_ns("2024-01-08T00:00:00Z"),
            37,
            expected_start_ns=_utc_ns("2024-01-01T00:00:00Z"),
            expected_end_ns=_utc_ns("2024-02-29T23:00:00Z"),
        )


def test_compute_coverage_reconstructs_internal_gap_and_cross_month_window() -> None:
    """coverage 必須根據 canonical gap 找回兩個缺時節點，而非相信 row 自述。"""

    arrival_ns = _utc_ns("2024-02-15T00:00:00Z")
    window = build_horizon_window(
        arrival_ns,
        7,
        expected_start_ns=_utc_ns("2024-01-01T00:00:00Z"),
        expected_end_ns=_utc_ns("2024-02-29T23:00:00Z"),
    )
    coverage = compute_horizon_coverage(
        window,
        expected_start_ns=_utc_ns("2024-01-01T00:00:00Z"),
        expected_end_ns=_utc_ns("2024-02-29T23:00:00Z"),
        canonical_start_ns=_utc_ns("2024-01-01T00:00:00Z"),
        canonical_end_ns=_utc_ns("2024-02-29T23:00:00Z"),
        canonical_gaps=[
            {
                "before_utc": "2024-02-10T00:00:00Z",
                "after_utc": "2024-02-10T03:00:00Z",
                "missing_step_count": 2,
            }
        ],
    )
    assert coverage.crossed_gap is True
    assert [utc_string(value) for value in coverage.missing_time_ns] == [
        "2024-02-10T01:00:00Z",
        "2024-02-10T02:00:00Z",
    ]


def test_compute_coverage_allows_adjacent_gaps_sharing_observed_endpoint() -> None:
    """交替有效／缺失資料的兩個 gap 可共享中間觀測端點而不重疊。"""

    window = build_horizon_window(
        _utc_ns("2024-01-03T00:00:00Z"),
        1,
        expected_start_ns=_utc_ns("2024-01-01T00:00:00Z"),
        expected_end_ns=_utc_ns("2024-01-04T00:00:00Z"),
    )
    coverage = compute_horizon_coverage(
        window,
        expected_start_ns=_utc_ns("2024-01-01T00:00:00Z"),
        expected_end_ns=_utc_ns("2024-01-04T00:00:00Z"),
        canonical_start_ns=_utc_ns("2024-01-01T00:00:00Z"),
        canonical_end_ns=_utc_ns("2024-01-04T00:00:00Z"),
        canonical_gaps=[
            {
                "before_utc": "2024-01-02T00:00:00Z",
                "after_utc": "2024-01-02T02:00:00Z",
                "missing_step_count": 1,
            },
            {
                "before_utc": "2024-01-02T02:00:00Z",
                "after_utc": "2024-01-02T04:00:00Z",
                "missing_step_count": 1,
            },
        ],
    )
    assert len(coverage.missing_time_ns) == 2


def test_generic_validator_rejects_forged_empty_missing_even_when_nonstrict() -> None:
    """重寫 missing 與 row count 後，validator 仍由 inventory gap 重算並拒絕篡改。"""

    arrival, gap, inventory = _generic_payloads(gap=True)
    gap["records"][0]["missing_utc"] = []
    gap["records"][0]["supported_step_count"] = gap["records"][0]["expected_step_count"]
    gap["records"][0]["crossed_gap"] = False
    result = validate_generic_gap_payload(gap, arrival, inventory, strict=False)
    assert result.valid is False
    assert any("missing_nodes_mismatch" in error for error in result.errors)


def test_generic_validator_strict_rejects_real_gap_and_checks_identity() -> None:
    """strict generic master 有 gap 時失敗，並檢查 site／region／flow／UTC binding。"""

    arrival, gap, inventory = _generic_payloads(gap=True)
    result = validate_generic_gap_payload(gap, arrival, inventory, strict=True)
    assert result.valid is False
    assert any("crosses_gap" in error for error in result.errors)

    gap["records"][0]["study_site_id"] = "other-site"
    identity_result = validate_generic_gap_payload(gap, arrival, inventory, strict=False)
    assert any("site_region_mismatch" in error for error in identity_result.errors)


def test_bed_residence_gap_evidence_separates_180_day_selection_from_90_day_runtime() -> None:
    """觀測選時以 180 日重算，沉底後逐筆運算支援則獨立驗證 90 日。"""

    expected_start = _utc_ns("2024-01-01T00:00:00Z")
    expected_end = _utc_ns("2025-12-31T23:00:00Z")
    observation_ns = _utc_ns("2024-10-01T00:00:00Z")
    deposition_ns = _utc_ns("2024-07-03T00:00:00Z")
    inventory = {
        "expected_period": {
            "start_utc": utc_string(expected_start),
            "end_utc": utc_string(expected_end),
            "hourly_step_count": 17_544,
        },
        "products": [
            {
                "analysis_region_id": "C",
                "flow_domain_id": "flow-c",
                "product": "ocm_native",
                "canonical_time": {
                    "canonical_time_count": 17_544,
                    "time_start_utc": utc_string(expected_start),
                    "time_end_utc": utc_string(expected_end),
                    "expected_timestep_hours": 1.0,
                    "gaps": [],
                },
            }
        ],
    }
    inputs = SimpleNamespace(
        backtrack_support_days=180,
        model_fields_set={"backtrack_support_days"},
        years=[2024, 2025],
        time_axis_contract={"expected_timestep_hours": 1.0},
    )
    bed = SimpleNamespace(maximum_age_days=90, runtime_horizon_support_days=90)
    config = SimpleNamespace(
        inputs=inputs,
        boundaries=SimpleNamespace(max_backtrack_days=90.0),
        scenarios=SimpleNamespace(bed_residence_time=bed),
        study_sites=[SimpleNamespace(study_site_id="houwan", analysis_region_id="C")],
    )
    runtime_window = build_horizon_window(
        deposition_ns,
        90,
        expected_start_ns=expected_start,
        expected_end_ns=expected_end,
    )
    runtime_coverage = compute_horizon_coverage(
        runtime_window,
        expected_start_ns=expected_start,
        expected_end_ns=expected_end,
        canonical_start_ns=expected_start,
        canonical_end_ns=expected_end,
    )
    selection_window = build_horizon_window(
        observation_ns,
        180,
        expected_start_ns=expected_start,
        expected_end_ns=expected_end,
    )
    selection_coverage = compute_horizon_coverage(
        selection_window,
        expected_start_ns=expected_start,
        expected_end_ns=expected_end,
        canonical_start_ns=expected_start,
        canonical_end_ns=expected_end,
    )
    arrival_payload = {
        "records": [
            {
                "arrival_time_id": "bed-arrival-1",
                "study_site_id": "houwan",
                "analysis_region_id": "C",
                "time_utc_ns": deposition_ns,
                "metadata": {"observation_time_utc_ns": observation_ns},
            }
        ]
    }
    gap_row = {
        "arrival_time_id": "bed-arrival-1",
        "study_site_id": "houwan",
        "analysis_region_id": "C",
        "flow_domain_id": "flow-c",
        "arrival_time_utc": utc_string(deposition_ns),
        "horizon_start_utc": utc_string(runtime_window.start_time_ns),
        "horizon_end_utc": utc_string(deposition_ns),
        "max_backtrack_days": 90,
        "support_days": 90,
        "expected_step_count": runtime_window.expected_step_count,
        "supported_step_count": runtime_coverage.supported_step_count,
        "crossed_gap": False,
        "missing_utc": [],
        "time_support_policy": BED_RESIDENCE_HORIZON_POLICY_ID,
        "observation_time_utc": utc_string(observation_ns),
        "selection_horizon_start_utc": utc_string(selection_window.start_time_ns),
        "selection_horizon_end_utc": utc_string(observation_ns),
        "selection_support_days": 180,
        "selection_expected_step_count": selection_window.expected_step_count,
        "selection_supported_step_count": selection_coverage.supported_step_count,
        "selection_crossed_gap": False,
        "selection_missing_utc": [],
    }
    gap_payload = {
        "schema_version": BED_RESIDENCE_INPUT_SCHEMA_VERSION,
        "policy": BED_RESIDENCE_HORIZON_POLICY_ID,
        "max_backtrack_days": 90,
        "support_days": 90,
        "selection_support_days": 180,
        "runtime_support_days": 90,
        "provenance": {"method_id": BED_RESIDENCE_HORIZON_METHOD_ID},
        "records": [gap_row],
    }

    result = validate_generic_gap_payload(
        gap_payload, arrival_payload, inventory, config=config, strict=True
    )
    assert result.valid is True
    assert result.summary["selection_support_days"] == 180
    assert result.summary["runtime_support_days"] == 90

    legacy_schema_payload = dict(gap_payload)
    legacy_schema_payload["schema_version"] = LEGACY_INPUT_SCHEMA_VERSION
    legacy_schema_result = validate_generic_gap_payload(
        legacy_schema_payload, arrival_payload, inventory, config=config, strict=True
    )
    assert any("schema_version_invalid" in error for error in legacy_schema_result.errors)

    tampered = dict(gap_payload)
    tampered["selection_support_days"] = 90
    selection_result = validate_generic_gap_payload(
        tampered, arrival_payload, inventory, config=config, strict=True
    )
    assert selection_result.valid is False
    assert any("selection_support_config_mismatch" in error for error in selection_result.errors)

    tampered = dict(gap_payload)
    tampered["runtime_support_days"] = 60
    runtime_result = validate_generic_gap_payload(
        tampered, arrival_payload, inventory, config=config, strict=True
    )
    assert runtime_result.valid is False
    assert any("runtime_support" in error for error in runtime_result.errors)


def test_gap_schema_version_rejects_bed_version_for_legacy_config() -> None:
    """舊設定只接受 1.0.0 gap 文件，不能載入 bed-only 1.1.0 欄位集合。"""

    arrival, gap, inventory = _generic_payloads(gap=True)
    gap["schema_version"] = BED_RESIDENCE_INPUT_SCHEMA_VERSION
    result = validate_generic_gap_payload(gap, arrival, inventory, strict=True)
    assert result.valid is False
    assert any("schema_version_invalid" in error for error in result.errors)


def test_resolve_horizon_preserves_legacy_when_support_is_none() -> None:
    """未明示 support 或明示 None 均保留舊 7 日政策；明示整數才進 generic。"""

    legacy = SimpleNamespace(
        inputs=SimpleNamespace(backtrack_support_days=None),
        boundaries=SimpleNamespace(max_backtrack_days=7.0),
    )
    assert resolve_configured_horizon(legacy).is_generic is False

    generic = SimpleNamespace(
        inputs=SimpleNamespace(backtrack_support_days=37),
        boundaries=SimpleNamespace(max_backtrack_days=7.0),
    )
    settings = resolve_configured_horizon(generic)
    assert settings.is_generic is True
    assert settings.support_days == 37
    assert settings.selection_days == 37.0


def test_resolve_generic_support_keeps_undecided_runtime_request() -> None:
    """共同母體可先定案 support=3，runtime requested 未定案時不偷代 7 日。"""

    config = SimpleNamespace(
        inputs=SimpleNamespace(backtrack_support_days=3),
        boundaries=SimpleNamespace(max_backtrack_days=None),
    )
    settings = resolve_configured_horizon(config)
    assert settings.is_generic is True
    assert settings.support_days == 3
    assert settings.requested_days is None
    assert settings.selection_days == 3.0


def test_resolve_explicit_null_support_is_not_legacy_fallback() -> None:
    """明示 null 表示母體未定案，不能靜默產生 legacy 7 日設定。"""

    config = SimpleNamespace(
        inputs=SimpleNamespace(backtrack_support_days=None, model_fields_set={"backtrack_support_days"}),
        boundaries=SimpleNamespace(max_backtrack_days=None),
    )
    with pytest.raises(HorizonContractError, match="尚未定案"):
        resolve_configured_horizon(config)


def test_source_canonical_rebuild_rejects_tampered_gap_summary(tmp_path: Path) -> None:
    """重讀已綁定月份 time axis 後，連同 sidecar 重簽也不能偽造 canonical gap。"""

    flow = "flow-c"
    month_dir = tmp_path / flow / "months" / "202401"
    month_dir.mkdir(parents=True)
    start = _utc_ns("2024-01-01T00:00:00Z")
    times = start + np.arange(31 * 24, dtype=np.int64) * 3_600_000_000_000
    time_path = month_dir / "time_utc_ns.npy"
    np.save(time_path, times)
    time_hash = sha256(time_path.read_bytes()).hexdigest()
    time_record = {
        "path": "$ROOT/flow-c/months/202401/time_utc_ns.npy",
        "file_kind": "time_axis",
        "fingerprint_kind": "npy_header_structural",
        "size_bytes": time_path.stat().st_size,
        "npy_header": {"shape": [times.size], "dtype": "int64"},
        "sha256": time_hash,
    }
    inventory = {
        "expected_period": {
            "start_utc": "2024-01-01T00:00:00Z",
            "end_utc": "2024-01-31T23:00:00Z",
            "hourly_step_count": 31 * 24,
        },
        "products": [
            {
                "analysis_region_id": "C",
                "flow_domain_id": flow,
                "product": "ocm_native",
                "months": [
                    {
                        "month": "202401",
                        "path": "$ROOT/flow-c/months/202401",
                        "time_count": times.size,
                        "time_start_utc": "2024-01-01T00:00:00Z",
                        "time_end_utc": "2024-01-31T23:00:00Z",
                    }
                ],
                "files": [time_record],
                "canonical_time": {
                    "policy": "sort_and_deduplicate_prefer_last",
                    "expected_timestep_hours": 1.0,
                    "input_time_count": times.size,
                    "canonical_time_count": times.size,
                    "reordered_time_step_count": 0,
                    "dropped_duplicate_time_step_count": 0,
                    "expected_period_time_count": times.size,
                    "available_period_time_count": times.size,
                    "missing_period_time_count": 0,
                    "continuous_hourly": True,
                    "time_start_utc": "2024-01-01T00:00:00Z",
                    "time_end_utc": "2024-01-31T23:00:00Z",
                    "time_sha256": sha256(times.astype("<i8").tobytes()).hexdigest(),
                    "gaps": [],
                },
                "root_token": "ROOT",
            }
        ],
    }
    inventory["products"][0]["canonical_time"]["gaps"] = [
        {
            "before_utc": "2024-01-10T00:00:00Z",
            "after_utc": "2024-01-10T02:00:00Z",
            "gap_hours": 2.0,
            "missing_step_count": 1,
        }
    ]
    errors = input_derivation_module._validate_canonical_axis_bindings(
        inventory,
        roots_by_token={"ROOT": tmp_path},
    )
    assert "canonical_axis_summary_mismatch:ocm_native:C:gaps" in errors
