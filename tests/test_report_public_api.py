"""驗證報告統計、比較統計、artifact staging、validation evidence 與 release 的公開 API。

本檔只檢查匯出名稱與物件 identity，不建立或讀寫任何 release，也不執行科學計算。
公開入口的存在代表呼叫端可以穩定找到 typed statistics、共同 staging／格式／校驗
foundation、一次串流 reducer、F11/T05 exact compatibility／核心差值純計算、F12/T06
定量 evidence schema/I/O、report-v1 writer／reader／validator 與 report pipeline 建置前
唯讀 gate；comparison release I/O、F01–F12/T01–T06 專屬 artifact adapter、完整
pipeline/render 整合與真實 sensitivity cases 尚未完成，不能稱 F11/T05
科學成果完成；解析解、dt/M、known-source、restart、NumPy/Numba、forward-validation
的正式 evidence 尚未產生，F12/T06 科學成果仍未完成。
"""

from __future__ import annotations

import lagrangian_backtracking as lbt
from lagrangian_backtracking.report_comparison_statistics import (
    ComparisonHDRStatus,
    ComparisonParameterDifference,
    ComparisonRatioDifference,
    ComparisonScalarDifference,
    ReportComparisonStatistics,
    build_report_comparison_statistics,
)
from lagrangian_backtracking.report_pipeline import ReportBuildPreflight, preflight_report_build
from lagrangian_backtracking.report_release import (
    ReportRelease,
    ReportReleaseWriter,
    read_report_registry,
    read_report_release,
    validate_report_release,
    write_report_release,
)
from lagrangian_backtracking.report_render import (
    RenderedArtifact,
    ReportStagingRenderer,
    report_render_style_context,
)
from lagrangian_backtracking.report_trajectory_stream import (
    EnvironmentCompletenessAccumulator,
    EnvironmentCompletenessSiteStatistics,
    EnvironmentCompletenessStatistics,
    TrajectoryReportAccumulator,
    TrajectoryStreamStatistics,
    build_environment_completeness_statistics,
    build_trajectory_stream_statistics,
)
from lagrangian_backtracking.report_validation_evidence import (
    VALIDATION_EVIDENCE_CATEGORIES,
    VALIDATION_EVIDENCE_SCHEMA_VERSION,
    VALIDATION_METRIC_CATEGORIES,
    ValidationEvidence,
    ValidationMetric,
    load_validation_evidence,
    validate_validation_evidence,
    write_validation_evidence,
)


def test_report_release_and_trajectory_stream_are_root_exports() -> None:
    """根目錄 ``__all__`` 應涵蓋比較統計、staging、stream 與 release API。"""

    release_symbols = {
        "ReportRelease": ReportRelease,
        "ReportReleaseWriter": ReportReleaseWriter,
        "read_report_registry": read_report_registry,
        "read_report_release": read_report_release,
        "validate_report_release": validate_report_release,
        "write_report_release": write_report_release,
    }
    stream_symbols = {
        "EnvironmentCompletenessSiteStatistics": EnvironmentCompletenessSiteStatistics,
        "EnvironmentCompletenessStatistics": EnvironmentCompletenessStatistics,
        "EnvironmentCompletenessAccumulator": EnvironmentCompletenessAccumulator,
        "build_environment_completeness_statistics": build_environment_completeness_statistics,
        "TrajectoryStreamStatistics": TrajectoryStreamStatistics,
        "TrajectoryReportAccumulator": TrajectoryReportAccumulator,
        "build_trajectory_stream_statistics": build_trajectory_stream_statistics,
    }
    render_symbols = {
        "RenderedArtifact": RenderedArtifact,
        "ReportStagingRenderer": ReportStagingRenderer,
        "report_render_style_context": report_render_style_context,
    }
    comparison_symbols = {
        "ComparisonHDRStatus": ComparisonHDRStatus,
        "ComparisonParameterDifference": ComparisonParameterDifference,
        "ComparisonRatioDifference": ComparisonRatioDifference,
        "ComparisonScalarDifference": ComparisonScalarDifference,
        "ReportComparisonStatistics": ReportComparisonStatistics,
        "build_report_comparison_statistics": build_report_comparison_statistics,
    }
    expected_symbols = {
        **release_symbols,
        **stream_symbols,
        **render_symbols,
        **comparison_symbols,
    }

    assert set(expected_symbols) <= set(lbt.__all__)
    for name, expected in expected_symbols.items():
        assert getattr(lbt, name) is expected


def test_validation_evidence_symbols_are_root_exports() -> None:
    """根目錄應公開 F12/T06 schema、immutable records 與 strict I/O API。"""

    expected_symbols = {
        "VALIDATION_EVIDENCE_CATEGORIES": VALIDATION_EVIDENCE_CATEGORIES,
        "VALIDATION_EVIDENCE_SCHEMA_VERSION": VALIDATION_EVIDENCE_SCHEMA_VERSION,
        "VALIDATION_METRIC_CATEGORIES": VALIDATION_METRIC_CATEGORIES,
        "ValidationEvidence": ValidationEvidence,
        "ValidationMetric": ValidationMetric,
        "load_validation_evidence": load_validation_evidence,
        "validate_validation_evidence": validate_validation_evidence,
        "write_validation_evidence": write_validation_evidence,
    }

    assert set(expected_symbols) <= set(lbt.__all__)
    assert VALIDATION_EVIDENCE_SCHEMA_VERSION == "1.0.0"
    assert VALIDATION_EVIDENCE_CATEGORIES == (
        "analytic_solution",
        "timestep_convergence",
        "member_convergence",
        "known_source_synthetic",
        "checkpoint_restart",
        "numpy_numba_consistency",
        "forward_validation",
    )
    assert VALIDATION_METRIC_CATEGORIES == VALIDATION_EVIDENCE_CATEGORIES
    for name, expected in expected_symbols.items():
        assert getattr(lbt, name) is expected


def test_report_pipeline_preflight_symbols_are_root_exports() -> None:
    """根目錄公開唯讀 preflight API，但不宣稱已提供完整 report-build。"""

    assert "ReportBuildPreflight" in lbt.__all__
    assert "preflight_report_build" in lbt.__all__
    assert lbt.ReportBuildPreflight is ReportBuildPreflight
    assert lbt.preflight_report_build is preflight_report_build
    assert not hasattr(lbt, "build_report_release")
