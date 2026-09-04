"""驗證離線程式架構地圖的模組目錄、流程端點與互動資料。

這組回歸測試不啟動瀏覽器，也不執行任何 Lagrangian 計算；它直接載入產圖腳本，確認
地圖資料與 ``src/lagrangian_backtracking`` 的實際 Python 檔案保持一對一。HTML 測試只
檢查產物中的節點、BayTrace 整合說明、來源統計／trajectory stream／比較統計／report release
資料流與模板替換，並確認 README 清楚區分已完成的共同 staging 基礎、F11/T05 純計算邊界
與尚未存在的專屬 science adapter／pipeline，避免以脆弱
的畫面截圖取代資料契約。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "render_source_code_architecture_map.py"


def _load_architecture_map_module() -> ModuleType:
    """以標準函式庫載入 scripts 下的產圖模組，避免替測試引入新套件。"""

    spec = importlib.util.spec_from_file_location("architecture_map_generator", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_module_catalog_covers_package_exactly() -> None:
    """確認每個非 ``__init__`` 的套件模組都有唯一且存在的地圖節點。"""

    module = _load_architecture_map_module()
    package_ids = {
        path.stem
        for path in module.PACKAGE_ROOT.glob("*.py")
        if path.stem != "__init__"
    }
    built_modules = module.build_modules()
    built_ids = [item["id"] for item in built_modules]

    assert len(built_ids) == len(set(built_ids))
    assert set(built_ids) == package_ids
    assert set(module.MODULE_INFO) == package_ids
    assert all((module.PACKAGE_ROOT / f"{module_id}.py").is_file() for module_id in built_ids)


def test_catalog_groups_and_flow_edges_are_closed() -> None:
    """確認群組、模組說明與語意流程的 source/target 不會指向未知節點。"""

    module = _load_architecture_map_module()
    package_ids = {
        path.stem
        for path in module.PACKAGE_ROOT.glob("*.py")
        if path.stem != "__init__"
    }
    group_ids = [group["id"] for group in module.GROUPS]
    grouped_ids = [module_id for group in module.GROUPS for module_id in group["modules"]]

    assert len(group_ids) == len(set(group_ids))
    assert len(grouped_ids) == len(set(grouped_ids))
    assert set(grouped_ids) == package_ids
    assert all(
        edge["source"] in package_ids and edge["target"] in package_ids
        for edge in module.FLOW_EDGES
    )
    flow_keys = [
        (edge["source"], edge["target"], edge["label"])
        for edge in module.FLOW_EDGES
    ]
    assert len(flow_keys) == len(set(flow_keys))
    runtime_info = module.MODULE_INFO["runtime"]
    assert "pilot/formal" in runtime_info["role"]
    assert "initialize_run" in runtime_info["entrypoints"]
    assert "open_run_controller" in runtime_info["entrypoints"]
    checkpoint_info = module.MODULE_INFO["checkpoint"]
    assert "write_execution_checkpoint" in checkpoint_info["entrypoints"]
    assert "load_execution_checkpoint" in checkpoint_info["entrypoints"]
    assert any(
        "write_checkpoint" in entry and "schema 1" in entry and "相容" in entry
        for entry in checkpoint_info["entrypoints"]
    )
    assert any(
        "load_checkpoint" in entry and "schema 1" in entry and "相容" in entry
        for entry in checkpoint_info["entrypoints"]
    )
    cli_info = module.MODULE_INFO["cli"]
    assert {
        "main",
        "run_preflight_command",
        "run_create",
        "run_shard",
        "run_reconcile",
        "run_validate_run",
        "run_report_validate",
        "report-validate",
    }.issubset(cli_info["entrypoints"])
    assert "formal" in cli_info["read_first"]
    assert any(
        edge == {"source": "cli", "target": "runtime", "label": "run-create／pilot+formal"}
        for edge in module.FLOW_EDGES
    )
    assert any(
        edge == {"source": "config", "target": "runtime", "label": "pilot/formal config"}
        for edge in module.FLOW_EDGES
    )

    # 三個來源統計節點必須公開各自真正承擔的 typed products，而不是把 facade、
    # renderer 與 release 混成一個已完成模組。這裡只檢查交接圖資料，不匯入或執行
    # 任一科學計算。
    matrix_info = module.MODULE_INFO["report_matrix_statistics"]
    assert {
        "OutcomeStatistics",
        "ConnectivityStatistics",
        "build_outcome_statistics",
        "build_connectivity_statistics",
    }.issubset(matrix_info["entrypoints"])
    source_receptor_info = module.MODULE_INFO["report_source_receptor_statistics"]
    assert {
        "SourceReceptorStatistics",
        "SourceReceptorStatistic",
        "TravelAgeStatistics",
        "build_source_receptor_statistics",
    }.issubset(source_receptor_info["entrypoints"])
    facade_info = module.MODULE_INFO["report_statistics"]
    assert {"ReportStatistics", "build_report_statistics"}.issubset(
        facade_info["entrypoints"]
    )
    assert "不負責繪圖" in facade_info["read_first"]

    comparison_info = module.MODULE_INFO["report_comparison_statistics"]
    assert {
        "ComparisonHDRStatus",
        "ComparisonParameterDifference",
        "ComparisonRatioDifference",
        "ComparisonScalarDifference",
        "ReportComparisonStatistics",
        "build_report_comparison_statistics",
    }.issubset(comparison_info["entrypoints"])
    assert "F11/T05 exact compatibility" in comparison_info["role"]
    assert "F11/T05 的 exact compatibility 與核心差值純計算已完成" in comparison_info[
        "read_first"
    ]
    for status_term in (
        "comparison release I/O",
        "artifact adapter",
        "pipeline/render",
        "真實 sensitivity cases",
        "不能稱 F11/T05 科學成果完成",
    ):
        assert status_term in comparison_info["read_first"]

    pipeline_info = module.MODULE_INFO["report_pipeline"]
    assert {"ReportBuildPreflight", "preflight_report_build"}.issubset(
        pipeline_info["entrypoints"]
    )
    assert "建置前唯讀 gate 已完成" in pipeline_info["role"]
    for status_term in (
        "complete run",
        "aggregate/spec binding",
        "formal trajectory v2",
        "MPLCONFIGDIR",
        "output",
        "evidence policy",
        "build_report_release",
        "F01–F12/T01–T06 專屬 artifact adapters",
        "CLI report-build",
        "正式 SERVER 科學發布",
        "不能把 preflight 稱為完整 pipeline",
    ):
        assert status_term in pipeline_info["read_first"]

    trajectory_stream_info = module.MODULE_INFO["report_trajectory_stream"]
    assert {
        "EnvironmentCompletenessSiteStatistics",
        "EnvironmentCompletenessStatistics",
        "EnvironmentCompletenessAccumulator",
        "build_environment_completeness_statistics",
        "TrajectoryStreamStatistics",
        "TrajectoryReportAccumulator",
        "build_trajectory_stream_statistics",
    }.issubset(trajectory_stream_info["entrypoints"])
    assert "一次 bounded trajectory stream" in trajectory_stream_info["role"]
    assert "不代表 renderer" in trajectory_stream_info["read_first"]

    release_info = module.MODULE_INFO["report_release"]
    assert {
        "ReportRelease",
        "ReportReleaseWriter",
        "read_report_registry",
        "read_report_release",
        "validate_report_release",
        "write_report_release",
    }.issubset(release_info["entrypoints"])
    assert "atomic" in release_info["role"]
    assert "report_pipeline.py 的建置前唯讀 gate 已完成" in release_info["read_first"]

    # renderer 節點只登錄共同輸出邊界；它不能被地圖文字誤讀成已完成所有 F/T
    # 科學版面或正式報告 pipeline。
    renderer_info = module.MODULE_INFO["report_render"]
    assert {
        "RenderedArtifact",
        "ReportStagingRenderer",
        "report_render_style_context",
        "ReportStagingRenderer.render_figure",
        "ReportStagingRenderer.render_table",
    }.issubset(renderer_info["entrypoints"])
    assert "共同 staging/格式/校驗基礎已完成" in renderer_info["read_first"]
    assert "F01–F12/T01–T06 專屬 artifact adapters" in renderer_info["read_first"]
    assert "report_pipeline" in renderer_info["read_first"]
    assert "report-build" in renderer_info["read_first"]
    assert "不能推定正式報告完成" in renderer_info["read_first"]

    validation_info = module.MODULE_INFO["report_validation_evidence"]
    assert "F12/T06 定量 evidence schema/I/O 已完成" in validation_info["role"]
    assert "schema 1.0.0" in validation_info["outputs"]
    assert {
        "VALIDATION_EVIDENCE_CATEGORIES",
        "VALIDATION_EVIDENCE_SCHEMA_VERSION",
        "VALIDATION_METRIC_CATEGORIES",
        "ValidationMetric",
        "ValidationEvidence",
        "load_validation_evidence",
        "validate_validation_evidence",
        "write_validation_evidence",
    }.issubset(validation_info["entrypoints"])
    for status_term in (
        "analytic_solution",
        "timestep_convergence",
        "member_convergence",
        "known_source_synthetic",
        "checkpoint_restart",
        "numpy_numba_consistency",
        "forward_validation",
        "正式 evidence 尚未產生",
        "F12/T06 科學成果仍未完成",
    ):
        assert status_term in validation_info["read_first"]

    required_report_edges = (
        {
            "source": "aggregate_release",
            "target": "report_matrix_statistics",
            "label": "outcome／cross-site counts",
        },
        {
            "source": "aggregate_release",
            "target": "report_source_receptor_statistics",
            "label": "source-receptor／age histograms",
        },
        {
            "source": "report_matrix_statistics",
            "target": "report_statistics",
            "label": "outcome／connectivity products",
        },
        {
            "source": "report_source_receptor_statistics",
            "target": "report_statistics",
            "label": "source-receptor products",
        },
        {
            "source": "report_spec",
            "target": "report_statistics",
            "label": "facade policy binding",
        },
        {
            "source": "production",
            "target": "report_trajectory_stream",
            "label": "complete ParticleResult iterator",
        },
        {
            "source": "report_trajectory_stream",
            "target": "report_statistics",
            "label": "material／pathway products",
        },
        {
            "source": "aggregate_release",
            "target": "report_release",
            "label": "source snapshots",
        },
        {
            "source": "report_records",
            "target": "report_release",
            "label": "registry／product identity",
        },
        {
            "source": "report_statistics",
            "target": "report_release",
            "label": "caller report products",
        },
        {
            "source": "cli",
            "target": "report_release",
            "label": "report-validate",
        },
        {
            "source": "aggregate_spec",
            "target": "report_validation_evidence",
            "label": "canonical snapshot／SHA",
        },
        {
            "source": "report_spec",
            "target": "report_validation_evidence",
            "label": "canonical snapshot／SHA",
        },
        {
            "source": "run_control",
            "target": "report_validation_evidence",
            "label": "source run plan snapshot",
        },
        {
            "source": "report_validation_evidence",
            "target": "report_records",
            "label": "F12/T06 evidence",
        },
    )
    assert all(edge in module.FLOW_EDGES for edge in required_report_edges)
    required_comparison_edges = (
        {
            "source": "aggregate_release_payload",
            "target": "report_comparison_statistics",
            "label": "exact run／axis binding",
        },
        {
            "source": "report_spec",
            "target": "report_comparison_statistics",
            "label": "policy compatibility",
        },
        {
            "source": "report_statistics",
            "target": "report_comparison_statistics",
            "label": "baseline／comparison typed products",
        },
        {
            "source": "report_comparison_statistics",
            "target": "report_records",
            "label": "F11/T05 comparison products",
        },
    )
    assert all(edge in module.FLOW_EDGES for edge in required_comparison_edges)
    required_renderer_edges = (
        {
            "source": "report_spec",
            "target": "report_render",
            "label": "renderer policy",
        },
        {
            "source": "report_style",
            "target": "report_render",
            "label": "style／font provenance",
        },
        {
            "source": "report_records",
            "target": "report_render",
            "label": "record／product contract",
        },
        {
            "source": "report_statistics",
            "target": "report_render",
            "label": "typed report products",
        },
        {
            "source": "report_render",
            "target": "report_release",
            "label": "staged artifacts",
        },
        {
            "source": "run_validation",
            "target": "report_pipeline",
            "label": "complete run／trajectory schema gate",
        },
        {
            "source": "aggregate_release",
            "target": "report_pipeline",
            "label": "aggregate release identity",
        },
        {
            "source": "report_spec",
            "target": "report_pipeline",
            "label": "ReportSpec／AggregateSpec binding",
        },
        {
            "source": "report_validation_evidence",
            "target": "report_pipeline",
            "label": "validation evidence policy",
        },
        {
            "source": "report_pipeline",
            "target": "report_release",
            "label": "preflight → future build_report_release",
        },
    )
    assert all(edge in module.FLOW_EDGES for edge in required_renderer_edges)


def test_render_html_contains_new_nodes_and_replaces_placeholders(tmp_path: Path) -> None:
    """確認重產 HTML 包含 runtime、來源統計節點與可搜尋的 BayTrace 說明。"""

    module = _load_architecture_map_module()
    output_path = tmp_path / "architecture-map.html"
    module.render_html(output_path)
    html = output_path.read_text(encoding="utf-8")

    for module_id in (
        "runtime",
        "batch_state",
        "production",
        "run_control",
        "report_matrix_statistics",
        "report_source_receptor_statistics",
        "report_statistics",
        "report_comparison_statistics",
        "report_pipeline",
        "report_trajectory_stream",
        "report_release",
        "report_render",
        "report_validation_evidence",
    ):
        assert f'"id":"{module_id}"' in html
    assert "F12/T06 定量 evidence schema/I/O 已完成" in html
    assert "正式 evidence 尚未產生" in html
    assert "F12/T06 科學成果仍未完成" in html
    assert "F11/T05 的 exact compatibility 與核心差值純計算已完成" in html
    assert "不能稱 F11/T05 科學成果完成" in html
    for status_term in (
        "建置前唯讀 gate 已完成",
        "complete run",
        "aggregate/spec binding",
        "formal trajectory v2",
        "MPLCONFIGDIR",
        "output/evidence policy",
        "build_report_release",
        "F01–F12/T01–T06 專屬 artifact adapters",
        "CLI report-build",
        "正式 SERVER 科學發布",
        "不能把 preflight 稱為完整 pipeline",
    ):
        assert status_term in html
    for category in (
        "analytic_solution",
        "timestep_convergence",
        "member_convergence",
        "known_source_synthetic",
        "checkpoint_restart",
        "numpy_numba_consistency",
        "forward_validation",
    ):
        assert category in html
    assert "BayTrace 整合關係" in html
    assert "baytrace_integration" in html
    assert 'const state = {selected: "runtime"' in html
    assert "__MODULES_JSON__" not in html
    assert "__IMPORT_EDGES_JSON__" not in html
    assert "__FLOW_EDGES_JSON__" not in html
    assert "__GROUPS_JSON__" not in html


def test_readme_states_report_boundary_without_claiming_renderer_completion() -> None:
    """README 應說明已完成的統計／release I/O 與尚未完成的 renderer 邊界。"""

    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "report_trajectory_stream.py" in readme
    assert "report_release.py" in readme
    assert "uv run lbt report-validate <run_id>.report-v1" in readme
    assert "report_render.py" in readme
    assert "report_validation_evidence.py" in readme
    assert "report_comparison_statistics.py" in readme
    assert "F12/T06 定量 evidence schema/I/O 已完成" in readme
    assert "解析解（`analytic_solution`）" in readme
    assert "dt/M 收斂（`timestep_convergence`／`member_convergence`）" in readme
    assert "known-source（`known_source_synthetic`）" in readme
    assert "restart（`checkpoint_restart`）" in readme
    assert "NumPy/Numba" in readme
    assert "forward-validation（`forward_validation`）" in readme
    assert "正式 evidence\n尚未產生" in readme
    assert "F12/T06 科學成果仍未完成" in readme
    assert "report_pipeline.py" in readme
    assert "report_pipeline.py` 的建置前唯讀 gate 已完成" in readme
    for status_term in (
        "complete run",
        "aggregate/spec binding",
        "formal trajectory v2",
        "MPLCONFIGDIR",
        "output/evidence policy",
        "`build_report_release`",
        "F01–F12/T01–T06 專屬 artifact adapters",
        "CLI `report-build`",
        "正式 SERVER 科學發布",
        "不能把 preflight 稱為完整 pipeline",
    ):
        assert status_term in readme
    assert "report-build" in readme
    assert "共同 staging/格式/校驗基礎已完成" in readme
    assert "F01–F12/T01–T06 專屬 artifact adapters" in readme
    assert "不能推定正式報告完成" in readme
    assert "F11/T05 的 exact compatibility 與核心差值純計算已完成" in readme
    assert "comparison release I/O" in readme
    assert "artifact adapter" in readme
    assert "pipeline/render" in readme
    assert "真實 sensitivity cases" in readme
    assert "不能稱 F11/T05 科學成果完成" in readme
    assert "尚未完成" in readme
    assert "解讀為正式科學成果" in readme
    assert "目前尚未完成圖表 renderer" not in readme


def test_committed_html_is_source_generated(tmp_path: Path) -> None:
    """確認提交的互動式地圖與目前 renderer 輸出的 source data 完全一致。"""

    module = _load_architecture_map_module()
    generated_path = tmp_path / "generated-map.html"
    module.render_html(generated_path)
    committed_path = PROJECT_ROOT / "docs" / "source_code_architecture_map.html"
    assert committed_path.read_bytes() == generated_path.read_bytes()
