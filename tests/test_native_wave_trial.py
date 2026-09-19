"""小試證據閘門測試：只驗有限範圍操作契約，不取代真資料或完整科學驗收。"""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "wave_trial", Path(__file__).parents[1] / "scripts/run_native_wave_trial.py")
trial = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trial)


def report():
    """完整成功證據仍包含真實乾點QC；不得將這些拒絕抹除或要求為零。"""
    return dict(require_native=True, fallback_particle_steps=0, native_completed_particle_steps=8,
                native_rejected_particle_steps=2, native_rejected_reasons={"dry_face": 2},
                source_sha256={"batch_interp": "verified-source-hash"}, group_size_histogram={"8": 1},
                resolver_report=dict(include_stokes=True, field_policy="finite_depth_node_fields_v1"))


def test_real_qc_is_retained():
    """真實QC與成功原版推進可以並存，原始摘要不得被就緒檢查改寫。"""
    data = report()
    trial.check_report(data)
    assert data["native_rejected_reasons"] == {"dry_face": 2}


@pytest.mark.parametrize("key,value", [
    ("require_native", False), ("fallback_particle_steps", 1), ("native_completed_particle_steps", 0),
    ("source_sha256", {}), ("group_size_histogram", {"1": 8}),
    ("resolver_report", {"include_stokes": False, "field_policy": "finite_depth_node_fields_v1"}),
    ("resolver_report", {"include_stokes": True, "field_policy": "legacy"}),
])
def test_incomplete_native_evidence_rejected(key, value):
    """未開波浪、舊核補算、零推進或缺少批次來源皆不可只憑退出碼通過。"""
    data = report()
    data[key] = value
    with pytest.raises(ValueError):
        trial.check_report(data)
