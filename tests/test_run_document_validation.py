"""run plan/progress 公開文件驗證 API 的防禦性快照與拒絕邊界測試。

測試以正式初始化流程建立最小 synthetic workspace，避免自行拼湊一份容易與 production
漂移的 progress schema。各變異只改記憶體中的文件副本，不改 workspace 內的來源檔案，
因此可同時確認純文件 API 與既有路徑 loader 的責任分界。
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import MappingProxyType

import pytest

from lagrangian_backtracking.provenance import CodeProvenance
from lagrangian_backtracking.run_control import (
    initialize_run_workspace,
    load_run_plan,
    load_run_progress,
    validate_run_plan_document,
    validate_run_progress_document,
)
from lagrangian_backtracking.scenarios import Scenario, stable_identifier

_CONFIG_HASH = "9" * 64
_COMMIT = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture(scope="module")
def synthetic_workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """用 production initializer 建立含真實 plan/progress 的最小 synthetic workspace。"""

    parent = tmp_path_factory.mktemp("run-document-validation")
    scenario = Scenario(
        scenario_id=stable_identifier(
            "scn",
            ["document-site", "material", "receptor", "arrival", "document-v1"],
        ),
        study_site_id="document-site",
        analysis_region_id="region-a",
        material_id="material",
        receptor_id="receptor",
        arrival_time_id="arrival",
        settling_velocity_mps=-0.001,
        arrival_time_utc_ns=1_700_000_000_000_000_000,
        design_version="document-v1",
    )
    provenance = CodeProvenance(
        git_available=True,
        git_commit=_COMMIT,
        git_dirty=False,
        commit_source="git_repository",
        deployment_tree_sha256="1" * 64,
        deployment_file_count=3,
        uv_lock_sha256="2" * 64,
        python_version="3.12.0",
        platform="test-platform",
        package_version="0.1.0",
        numpy_version="test",
        numba_version="test",
        pyarrow_version="test",
    )
    workspace = initialize_run_workspace(
        parent,
        run_id="document-validation-run",
        scenarios=(scenario,),
        normalized_config={"schema_version": "test", "settings": {"dt_seconds": 1.0}},
        config_hash=_CONFIG_HASH,
        input_inventory_file={"source": "synthetic", "files": []},
        component_canonical_hashes={
            "material": "a" * 64,
            "receptor": "b" * 64,
            "arrival": "c" * 64,
        },
        geometry_canonical_hashes={
            "domain": "d" * 64,
            "local": "e" * 64,
            "open_boundary": "f" * 64,
        },
        provenance=provenance,
        experiment_case_id="baseline",
        master_seed=20260829,
        seed_policy="sha256_v1_pcg64dxsm",
        members_per_scenario=1,
        shard_scenario_count=1,
        checkpoint_interval_sweeps=1,
        active_chunk_size=1,
        run_kind="synthetic",
    )
    return workspace.path


def test_real_workspace_documents_and_path_loaders_pass(synthetic_workspace: Path) -> None:
    """真實 initializer 產生的兩份文件可由公開 API 與既有路徑 loader 驗證。"""

    plan = load_run_plan(synthetic_workspace)
    progress = load_run_progress(synthetic_workspace)

    assert validate_run_plan_document(plan) == plan
    assert validate_run_progress_document(progress) == progress
    # 再走一次檔案路徑，確認 loader 改為委派公開 API 後仍維持既有成功流程。
    assert load_run_plan(synthetic_workspace) == plan
    assert load_run_progress(synthetic_workspace) == progress


def test_document_snapshots_do_not_share_nested_containers(synthetic_workspace: Path) -> None:
    """回傳內容與輸入相等，但兩種文件的巢狀字典／串列均不共享。"""

    plan_document = load_run_plan(synthetic_workspace)
    progress_document = load_run_progress(synthetic_workspace)
    # MappingProxyType 證明入口接受一般 Mapping；根節點仍應固定成可獨立持有的 dict。
    plan_snapshot = validate_run_plan_document(MappingProxyType(plan_document))
    progress_snapshot = validate_run_progress_document(MappingProxyType(progress_document))

    assert plan_snapshot == plan_document
    assert progress_snapshot == progress_document
    assert plan_snapshot is not plan_document
    assert plan_snapshot["files"] is not plan_document["files"]
    assert plan_snapshot["files"]["normalized_config.json"] is not plan_document["files"][
        "normalized_config.json"
    ]
    assert plan_snapshot["shards"] is not plan_document["shards"]
    assert plan_snapshot["shards"][0] is not plan_document["shards"][0]
    shard_id = next(iter(progress_document["shards"]))
    assert progress_snapshot["shards"] is not progress_document["shards"]
    assert progress_snapshot["shards"][shard_id] is not progress_document["shards"][shard_id]
    assert progress_snapshot["shards"][shard_id]["metrics"] is not progress_document["shards"][
        shard_id
    ]["metrics"]

    # 驗證後竄改 caller 容器，不得反向改變已驗證快照。
    plan_document["shards"][0]["scenario_count"] = 999
    plan_document["files"]["normalized_config.json"]["sha256"] = "0" * 64
    progress_document["shards"][shard_id]["metrics"]["caller_mutation"] = True
    assert plan_snapshot["shards"][0]["scenario_count"] == 1
    assert plan_snapshot["files"]["normalized_config.json"]["sha256"] != "0" * 64
    assert progress_snapshot["shards"][shard_id]["metrics"] == {}


@pytest.mark.parametrize("document", [None, [], "not-a-mapping", 1])
def test_document_validators_reject_non_mapping(document: object) -> None:
    """兩個公開入口均拒絕非 Mapping root，且固定以 ValueError 回報。"""

    with pytest.raises(ValueError):
        validate_run_plan_document(document)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        validate_run_progress_document(document)  # type: ignore[arg-type]


def test_documents_reject_unknown_missing_and_bool_counts(synthetic_workspace: Path) -> None:
    """未知／缺少欄位及可冒充整數的 bool 在 plan/progress 都必須 fail closed。"""

    valid_plan = load_run_plan(synthetic_workspace)
    valid_progress = load_run_progress(synthetic_workspace)

    invalid_plan_documents = []
    plan_unknown = deepcopy(valid_plan)
    plan_unknown["unknown"] = None
    invalid_plan_documents.append(plan_unknown)
    plan_missing = deepcopy(valid_plan)
    del plan_missing["schema_version"]
    invalid_plan_documents.append(plan_missing)
    plan_bool_count = deepcopy(valid_plan)
    plan_bool_count["scenario_count"] = True
    invalid_plan_documents.append(plan_bool_count)
    for document in invalid_plan_documents:
        with pytest.raises(ValueError):
            validate_run_plan_document(document)

    invalid_progress_documents = []
    progress_unknown = deepcopy(valid_progress)
    progress_unknown["unknown"] = None
    invalid_progress_documents.append(progress_unknown)
    progress_missing = deepcopy(valid_progress)
    del progress_missing["schema_version"]
    invalid_progress_documents.append(progress_missing)
    progress_bool_count = deepcopy(valid_progress)
    progress_bool_count["revision"] = True
    invalid_progress_documents.append(progress_bool_count)
    for document in invalid_progress_documents:
        with pytest.raises(ValueError):
            validate_run_progress_document(document)


def test_progress_rejects_invalid_lifecycle_and_range(synthetic_workspace: Path) -> None:
    """非法生命週期與有缺口的 scenario range 仍由原 progress 契約拒絕。"""

    valid_progress = load_run_progress(synthetic_workspace)
    shard_id = next(iter(valid_progress["shards"]))

    invalid_lifecycle = deepcopy(valid_progress)
    invalid_lifecycle["shards"][shard_id]["lifecycle"] = "UNKNOWN"
    with pytest.raises(ValueError):
        validate_run_progress_document(invalid_lifecycle)

    invalid_range = deepcopy(valid_progress)
    invalid_range["shards"][shard_id]["scenario_start_index"] = 1
    invalid_range["shards"][shard_id]["scenario_stop_index"] = 2
    with pytest.raises(ValueError):
        validate_run_progress_document(invalid_range)
