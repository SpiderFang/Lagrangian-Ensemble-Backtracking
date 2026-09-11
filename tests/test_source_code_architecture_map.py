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
import re
import subprocess
from pathlib import Path
from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "render_source_code_architecture_map.py"

# 前 18 筆是原先位於 docs 根層的文件：其中 17 筆已依主題搬入子目錄，
# implementation_status.md 仍留在根層作為目前狀態的單一索引。第 19 筆是後續新增的
# 新竹 pilot 參數紀錄，沒有待相容的舊根層路徑，因此以 current/original 相同的
# self-alias 形式納入契約。測試同時核對現行路徑與索引中的原路徑文字，避免整理文件
# 或新增交付文件時遺失任一份歷史、現行或新註冊內容。
DOCUMENT_CATALOG: tuple[tuple[str, str], ...] = (
    ("docs/implementation_status.md", "docs/implementation_status.md"),
    (
        "docs/foundation/01_requirements_traceability.md",
        "docs/01_requirements_traceability.md",
    ),
    (
        "docs/foundation/02_architecture_and_data_contract.md",
        "docs/02_architecture_and_data_contract.md",
    ),
    (
        "docs/foundation/03_scientific_method_and_validation.md",
        "docs/03_scientific_method_and_validation.md",
    ),
    ("docs/archive/04_implementation_plan.md", "docs/04_implementation_plan.md"),
    ("docs/development/05_decisions_and_risks.md", "docs/05_decisions_and_risks.md"),
    ("docs/operations/06_server_runbook_plan.md", "docs/06_server_runbook_plan.md"),
    ("docs/results/07_results_visualization_plan.md", "docs/07_results_visualization_plan.md"),
    (
        "docs/foundation/08_design_baseline_and_derived_gates.md",
        "docs/08_design_baseline_and_derived_gates.md",
    ),
    (
        "docs/archive/09_implementation_audit_2026-08-19.md",
        "docs/09_implementation_audit_2026-08-19.md",
    ),
    (
        "docs/operations/10_available_data_time_reconstruction_and_a_expansion.md",
        "docs/10_available_data_time_reconstruction_and_a_expansion.md",
    ),
    (
        "docs/development/11_source_code_guide_and_plan_traceability.md",
        "docs/11_source_code_guide_and_plan_traceability.md",
    ),
    (
        "docs/results/12_aggregate_release_and_server_execution_plan.md",
        "docs/12_aggregate_release_and_server_execution_plan.md",
    ),
    (
        "docs/results/13_report_release_and_scientific_outputs_plan.md",
        "docs/13_report_release_and_scientific_outputs_plan.md",
    ),
    (
        "docs/results/14_hsinchu_2024-01-01_24h_pilot_parameter_record.md",
        "docs/results/14_hsinchu_2024-01-01_24h_pilot_parameter_record.md",
    ),
    (
        "docs/results/15_four_region_first_pilot_audit.md",
        "docs/results/15_four_region_first_pilot_audit.md",
    ),
    (
        "docs/operations/14_input_derivation_and_release_contract.md",
        "docs/14_input_derivation_and_release_contract.md",
    ),
    ("docs/operations/cli_reference.md", "docs/cli_reference.md"),
    ("docs/operations/git_deployment_and_data_sync.md", "docs/git_deployment_and_data_sync.md"),
    ("docs/operations/pilot_run_plan.md", "docs/pilot_run_plan.md"),
)

# 只抓取一般 Markdown 連結，不把圖片語法的開頭驚嘆號當成另一個連結；外部 URL
# 會在掃描函式中排除，保留所有本地 Markdown、圖檔、PDF 與設定檔連結的存在性檢查。
MARKDOWN_LINK_PATTERN = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]*)\)")
EXTERNAL_LINK_PREFIXES = ("http://", "https://", "mailto:")
# BayTrace、work、tmp、outputs 與虛擬環境是大型外部／執行產物，不是本專案文件拓撲；
# 排除它們可讓「全 repo 文件」驗收聚焦於可交付的 tracked 文件與本次新增索引。
NON_DELIVERABLE_MARKDOWN_DIRECTORIES = frozenset(
    {".git", ".pytest_cache", ".venv", "BayTrace", "dist", "outputs", "tmp", "work"}
)


def _load_architecture_map_module() -> ModuleType:
    """以標準函式庫載入 scripts 下的產圖模組，避免替測試引入新套件。"""

    spec = importlib.util.spec_from_file_location("architecture_map_generator", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _iter_tracked_markdown_paths() -> tuple[Path, ...]:
    """取得目前 checkout 中 Git 索引所追蹤且實際存在的 Markdown 檔案。

    本機可能保留未納入部署的文獻快取或其他忽略資料；直接以 ``rglob`` 掃描會把
    這些檔案誤算進文件拓撲，造成與 SERVER 乾淨或 detached checkout 不同的連結數。
    透過 ``git ls-files --cached`` 讀取索引，測試因此只涵蓋可由核定 checkout 部署的
    追蹤檔案。若執行環境不是完整 Git checkout，明確拋出含 repository 根目錄的錯誤，
    讓部署問題不會被誤判成連結數差異。
    """

    try:
        completed = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "ls-files", "--cached", "-z", "--", "*.md"],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"無法從 Git checkout 取得追蹤 Markdown：{PROJECT_ROOT}；"
            "請確認 SERVER 使用完整 detached checkout 且 git 可執行。"
        ) from exc

    paths: list[Path] = []
    for raw_path in completed.stdout.split(b"\0"):
        if not raw_path:
            continue
        relative_path = Path(raw_path.decode("utf-8"))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RuntimeError(f"Git 回傳不合法的 Markdown 相對路徑：{relative_path}")
        source_path = PROJECT_ROOT / relative_path
        if not source_path.is_file():
            raise RuntimeError(
                f"Git 追蹤的 Markdown 在 checkout 中不存在：{relative_path}；"
                "請重新同步完整 commit。"
            )
        paths.append(source_path)
    return tuple(sorted(paths))


def _iter_local_markdown_links() -> list[tuple[Path, str, Path]]:
    """列出全 repository Markdown 中的本地連結及其解析後目標。

    來源限定為 Git 索引中的追蹤 Markdown，避免本機忽略的資料快取改變驗收分母。
    掃描以連結所在 Markdown 檔案的父目錄為基準，這樣文件搬移後的 ``../`` 層級會
    直接受到測試約束。外部 DOI、網頁與電子郵件連結不屬於本地檔案拓撲，因此排除；
    只有去除片段識別碼後的實際路徑會交由呼叫端檢查是否存在。回傳原始檔、原始目標
    與解析後的 Path，方便失敗訊息指出哪一份文件的哪個連結斷裂。
    """

    links: list[tuple[Path, str, Path]] = []
    for source_path in _iter_tracked_markdown_paths():
        # 只以 checkout 內的相對路徑判斷執行產物目錄；SERVER checkout 常位於
        # ``/home/mustlab/work/...``，若直接檢查絕對路徑的 parts，外層部署目錄
        # ``work`` 會誤排除整個 repository，令連結數從 142 變成 0。
        relative_parts = source_path.relative_to(PROJECT_ROOT).parts
        if any(part in NON_DELIVERABLE_MARKDOWN_DIRECTORIES for part in relative_parts):
            continue
        contents = source_path.read_text(encoding="utf-8")
        for match in MARKDOWN_LINK_PATTERN.finditer(contents):
            raw_target = match.group(1).strip()
            if raw_target.startswith("<") and ">" in raw_target:
                raw_target = raw_target[1 : raw_target.index(">")]
            if raw_target.startswith(EXTERNAL_LINK_PREFIXES):
                continue
            target_without_fragment = raw_target.split("#", 1)[0].strip()
            resolved_target = (
                source_path
                if not target_without_fragment
                else (source_path.parent / target_without_fragment).resolve()
            )
            links.append((source_path, raw_target, resolved_target))
    return links


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


def test_integrator_map_records_stage_only_surface_adjustment() -> None:
    """架構地圖須揭露最小步長海面中間點調節，但不可描述成一般取樣放寬。"""

    module = _load_architecture_map_module()
    integrator_info = module.MODULE_INFO["integrators"]
    assert "SurfaceStageVelocityProvider" in integrator_info["entrypoints"]
    assert "下一次折半低於 dt_min" in integrator_info["read_first"]
    assert "完整 RK4 後" in integrator_info["read_first"]


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
    matrix_module_info = module.MODULE_INFO["pilot_matrix_validation"]
    assert {
        "validate_pilot_matrix",
        "canonical_pilot_matrix_json",
    }.issubset(matrix_module_info["entrypoints"])
    assert "只讀小型 immutable JSON" in matrix_module_info["read_first"]
    assert any(
        edge
        == {
            "source": "cli",
            "target": "pilot_matrix_validation",
            "label": "pilot-matrix-validate／跨區共同設定",
        }
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
        "pilot_matrix_validation",
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
    """README 導覽與 implementation status 應保留 report 工程邊界及正式 evidence 缺口。"""

    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    status = (PROJECT_ROOT / "docs" / "implementation_status.md").read_text(encoding="utf-8")
    requirements = (
        PROJECT_ROOT / "docs" / "foundation" / "01_requirements_traceability.md"
    ).read_text(
        encoding="utf-8"
    )
    assert len(readme.splitlines()) <= 250
    first_formal_section = next(
        index for index, line in enumerate(readme.splitlines(), start=1) if line.startswith("## ")
    )
    assert first_formal_section <= 15
    guide_section = readme.split("## 6. 文件導覽\n", 1)[1].split("\n## 7. ", 1)[0]
    guide_rows = [line for line in guide_section.splitlines() if line.startswith("| ")]
    assert len(guide_rows) == 6  # 表頭加上五個穩定入口；詳細分類移至 docs/README.md。
    for link in (
        "docs/README.md",
        "docs/implementation_status.md",
        "docs/operations/cli_reference.md",
        "docs/operations/06_server_runbook_plan.md",
        "docs/source_code_architecture_map.html",
    ):
        assert link in readme
    assert "docs/operations/pilot_run_plan.md#獨立海岸底圖重繪" in readme
    pilot_plan = (PROJECT_ROOT / "docs" / "operations" / "pilot_run_plan.md").read_text(
        encoding="utf-8"
    )
    assert "## 獨立海岸底圖重繪" in pilot_plan
    for required_term in (
        "OCM native schema `3`",
        "NWW3 analysis schema `1`",
        "trajectory shard",
        "execution checkpoint",
        "50,000×M",
        "不讀 raw NetCDF",
        "不以零值",
        "條件式來源足跡",
        "B 區 r2",
        "--style baytrace",
        "關閉單項敏感度、單位檢核",
    ):
        assert required_term in readme
    for homepage_only_term in ("F11/T05", "F12/T06"):
        assert homepage_only_term not in readme
    homepage_prose = re.sub(r"`[^`]*`", "", readme)
    homepage_prose = re.sub(r"\[[^\]]*\]\([^)]*\)", "", homepage_prose)
    for forbidden_prose_term in (
        "runtime",
        "gate",
        "forcing",
        "facade",
        "evidence",
        "immutable",
        "binding",
        "payload",
        "consumer",
        "cadence",
        "selector",
    ):
        assert re.search(
            rf"(?<![A-Za-z]){re.escape(forbidden_prose_term)}(?![A-Za-z])",
            homepage_prose,
        ) is None
    combined_status = f"{readme}\n{status}"
    for status_term in ("report-build", "F11/T05", "F12/T06"):
        assert status_term in combined_status
    assert "REQ-007" in requirements
    assert "關閉單項敏感度、單位 gate" in requirements
    assert "test_velocity_recording.py" in requirements
    assert "正式 writer／reader" in requirements

    for report_module in (
        "report_trajectory_stream.py",
        "report_statistics.py",
        "report_release.py",
        "report_render.py",
        "report_validation_evidence.py",
        "report_pipeline.py",
        "report_comparison_statistics.py",
    ):
        assert report_module in status
    assert "F12/T06 的正式 evidence 尚未產生" in status
    for evidence_category in (
        "解析解 `analytic_solution`",
        "dt／M 收斂",
        "known-source `known_source_synthetic`",
        "restart `checkpoint_restart`",
        "NumPy／Numba `numpy_numba_consistency`",
        "forward-validation `forward_validation`",
    ):
        assert evidence_category in status
    assert "v2 或 v3" in status
    assert "v1 明確拒絕正式垂向證據" in status
    assert "v2／v3 混用" in status
    assert "report-build" in status
    assert "F01–F12／T01–T06 專屬 artifact adapters" in status
    assert "不能把 preflight 稱為完整 pipeline" in status
    assert "共同 staging／格式／checksum 基礎" in status
    assert "F11/T05 的 exact compatibility 與核心差值純計算已存在" in status
    assert "comparison release I/O" in status
    assert "真實 sensitivity cases" in status
    assert "不能稱 F11/T05 科學成果完成" in status
    assert "不能將 schema／I/O 通過" in status
    assert "解讀為正式 OCM／NWW3\n科學驗證" in status
    assert "目前尚未完成圖表 renderer" not in readme
    assert "目前尚未完成圖表 renderer" not in status


def test_document_index_catalog_and_all_local_markdown_links_are_closed() -> None:
    """確認文件總入口涵蓋分類與 20 筆 catalog/alias 契約，且本地連結不斷裂。

    這裡檢查的是所有 Markdown 檔案的實際連結拓撲，包含跨分類文件、設定檔、測試、
    圖檔與 PDF；不只檢查新加入的閱讀提示，也確認新竹 pilot 文件以 self-alias
    登錄。外部 DOI 與網頁連結由掃描器排除，因為它們不是本地檔案存在性可以驗收的範圍。
    """

    index_path = PROJECT_ROOT / "docs" / "README.md"
    index = index_path.read_text(encoding="utf-8")
    assert len(DOCUMENT_CATALOG) == 20
    for category in ("foundation/", "operations/", "results/", "development/", "archive/"):
        assert category in index
    for current_path, original_path in DOCUMENT_CATALOG:
        assert (PROJECT_ROOT / current_path).is_file(), current_path
        index_relative_path = current_path.removeprefix("docs/")
        assert index_relative_path in index
        assert original_path in index
    for archive_path in (
        PROJECT_ROOT / "docs" / "archive" / "04_implementation_plan.md",
        PROJECT_ROOT / "docs" / "archive" / "09_implementation_audit_2026-08-19.md",
    ):
        archive_text = archive_path.read_text(encoding="utf-8")
        assert "今天狀態以 [實作狀態](../implementation_status.md) 為準" in archive_text

    local_links = _iter_local_markdown_links()
    # 147 是本次加入四區稽核文件與其六個導覽連結後，Git 追蹤 Markdown 的固定連結基線；
    # 小型時間重建文獻索引 README 已納入追蹤，工項 3 原始 PDF 則維持本機限定且不建立失效連結。
    # 這個數字是在收斂掃描範圍並修復兩個真實斷鏈後，由本機與 SERVER detached
    # checkout 共同核對所得，避免用改數字掩蓋兩邊拓撲差異。
    assert len(local_links) == 147
    broken_links = [
        (
            source_path.relative_to(PROJECT_ROOT).as_posix(),
            target,
            resolved_target.relative_to(PROJECT_ROOT).as_posix()
            if resolved_target.is_relative_to(PROJECT_ROOT)
            else str(resolved_target),
        )
        for source_path, target, resolved_target in local_links
        if not resolved_target.is_file()
    ]
    assert not broken_links, "本地 Markdown 連結目標不存在：" + repr(broken_links)


def test_committed_html_is_source_generated(tmp_path: Path) -> None:
    """確認提交的互動式地圖與目前 renderer 輸出的 source data 完全一致。"""

    module = _load_architecture_map_module()
    generated_path = tmp_path / "generated-map.html"
    module.render_html(generated_path)
    committed_path = PROJECT_ROOT / "docs" / "source_code_architecture_map.html"
    assert committed_path.read_bytes() == generated_path.read_bytes()
