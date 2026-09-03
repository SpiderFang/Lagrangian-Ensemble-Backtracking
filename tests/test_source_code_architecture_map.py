"""驗證離線程式架構地圖的模組目錄、流程端點與互動資料。

這組回歸測試不啟動瀏覽器，也不執行任何 Lagrangian 計算；它直接載入產圖腳本，確認
地圖資料與 ``src/lagrangian_backtracking`` 的實際 Python 檔案保持一對一。HTML 測試只
檢查產物中的節點、BayTrace 整合說明與模板替換，避免以脆弱的畫面截圖取代資料契約。
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


def test_render_html_contains_new_nodes_and_replaces_placeholders(tmp_path: Path) -> None:
    """確認重產 HTML 包含新 runtime/batch 節點與可搜尋的 BayTrace 說明。"""

    module = _load_architecture_map_module()
    output_path = tmp_path / "architecture-map.html"
    module.render_html(output_path)
    html = output_path.read_text(encoding="utf-8")

    for module_id in ("runtime", "batch_state", "production", "run_control"):
        assert f'"id":"{module_id}"' in html
    assert "BayTrace 整合關係" in html
    assert "baytrace_integration" in html
    assert 'const state = {selected: "runtime"' in html
    assert "__MODULES_JSON__" not in html
    assert "__IMPORT_EDGES_JSON__" not in html
    assert "__FLOW_EDGES_JSON__" not in html
    assert "__GROUPS_JSON__" not in html


def test_committed_html_is_source_generated(tmp_path: Path) -> None:
    """確認提交的互動式地圖與目前 renderer 輸出的 source data 完全一致。"""

    module = _load_architecture_map_module()
    generated_path = tmp_path / "generated-map.html"
    module.render_html(generated_path)
    committed_path = PROJECT_ROOT / "docs" / "source_code_architecture_map.html"
    assert committed_path.read_bytes() == generated_path.read_bytes()
