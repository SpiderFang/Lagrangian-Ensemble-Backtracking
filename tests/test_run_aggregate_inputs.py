"""完整 run trajectory shard iterator 與聚合輸入邊界的集中契約測試。

本檔只驗證聚合流程讀取 run workspace 時所依賴的唯讀輸入邊界，不測試聚合統計本身。
fixture 透過既有的 ``initialize_run_workspace``、``RunController`` 與 trajectory reader
建立真實的 synthetic／pilot 多 shard 輸出，再以竄改後的暫存 workspace 驗證 fail-closed
行為。測試中的絕對路徑只存在於 pytest 的暫存目錄；對外公開的 ``ValueError`` 必須不把
這些部署位置帶回錯誤訊息。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_run_control import _request, _workspace, _write_progress

from lagrangian_backtracking.engine import ParticleResult
from lagrangian_backtracking.outputs import sha256_file
from lagrangian_backtracking.run_control import (
    RunController,
    _scenario_from_row,
    load_run_plan,
    load_run_progress,
)
from lagrangian_backtracking.run_validation import (
    ValidatedTrajectoryShard,
    iter_complete_run_trajectory_shards,
    validate_run,
)
from lagrangian_backtracking.runner import iter_run_units
from lagrangian_backtracking.scenarios import Scenario


def _complete_run(
    parent: Path,
    run_id: str,
    *,
    run_kind: str = "synthetic",
    checkpoint_root: Path | None = None,
    pause_before_resume: bool = False,
) -> Path:
    """用正式 controller 路徑建立可供 iterator 讀取的完整暫存 run。

    ``pause_before_resume`` 會先讓每個 shard 寫出一個 checkpoint，再以
    ``resume=True`` 完成；這是外部 checkpoint 測試需要的實際狀態轉換。一般 fixture
    則直接依 immutable plan 順序執行全部 shard，並呼叫 reconcile，確保測試同時涵蓋
    run-create、execute、reconcile 與後續 reader，而不是只手工拼裝輸出檔案。
    """

    workspace = _workspace(
        parent,
        run_id,
        interval=1,
        run_kind=run_kind,
    )
    plan = load_run_plan(workspace)
    if pause_before_resume:
        initial = RunController(
            workspace,
            request_factory=_request,
            checkpoint_root=checkpoint_root,
        )
        for row in plan["shards"]:
            summary = initial.run_shard(row["shard_id"], sweep_budget=1)
            assert summary.lifecycle == "PAUSED"
        controller = RunController(
            workspace,
            request_factory=_request,
            resume=True,
            checkpoint_root=checkpoint_root,
        )
        controller.run_all()
    else:
        controller = RunController(
            workspace,
            request_factory=_request,
            checkpoint_root=checkpoint_root,
        )
        controller.run_all()
    reconciled = controller.reconcile()
    assert reconciled["run_lifecycle"] == "COMPLETE"
    return workspace


def _assert_iterator_rejects_without_absolute_path(
    workspace: Path,
    tmp_path: Path,
    *,
    checkpoint_root: Path | None = None,
) -> None:
    """確認 iterator 的公開 ``ValueError`` 不洩漏 run 或外部 root 絕對路徑。"""

    with pytest.raises(ValueError) as raised:
        tuple(
            iter_complete_run_trajectory_shards(
                workspace,
                checkpoint_root=checkpoint_root,
            )
        )
    message = repr(raised.value)
    assert str(workspace) not in message
    assert str(tmp_path) not in message
    if checkpoint_root is not None:
        assert str(checkpoint_root) not in message


def _rewrite_manifest(output: Path, mutate: Callable[[dict], None]) -> None:
    """以固定 JSON 格式改寫 manifest；只供測試製造明確的壞輸入。

    這個 helper 不替竄改後的 payload 重新計算任何 run-level checksum；它只在測試需要
    證明「內容 identity/order 已變但 payload checksum 被同步」時，更新 particle table
    在 trajectory manifest 中的檔案紀錄。其餘 manifest count/checksum 測試會刻意保留
    不一致，讓 iterator 必須拒絕。
    """

    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(manifest)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _rewrite_particle_rows(output: Path, mutate: Callable[[list[dict]], None]) -> None:
    """修改 particle table 後同步該檔案 manifest record，保留其餘 payload 不變。"""

    particle_path = output / "particle_table.parquet"
    table = pq.read_table(particle_path)
    rows = table.to_pylist()
    mutate(rows)
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), particle_path)

    def update_particle_record(manifest: dict) -> None:
        """同步 particle table 的 byte size 與 SHA-256，避免測試被 checksum gate 先攔截。"""

        manifest["files"]["particle_table.parquet"] = {
            "size_bytes": particle_path.stat().st_size,
            "sha256": sha256_file(particle_path),
        }

    _rewrite_manifest(output, update_particle_record)


@pytest.mark.parametrize("run_kind", ["synthetic", "pilot"])
def test_complete_multi_shard_yields_plan_order_and_complete_containers(
    tmp_path: Path,
    run_kind: str,
) -> None:
    """完整 synthetic／pilot run 必須依 plan 順序產生完整且逐欄正確的容器。"""

    workspace = _complete_run(tmp_path, f"complete-{run_kind}", run_kind=run_kind)
    plan = load_run_plan(workspace)
    plan_rows = tuple(plan["shards"])
    scenario_rows = pq.read_table(workspace / "scenario_table.parquet").to_pylist()
    ordered_scenarios = tuple(
        _scenario_from_row(row, label=f"scenario[{index}]")
        for index, row in enumerate(scenario_rows)
    )
    records = tuple(iter_complete_run_trajectory_shards(workspace))

    # plan row 的原始排列就是 immutable execution ordering；iterator 不可自行排序或
    # 依 shard id 重新建立另一套順序，否則後續聚合的 source／receptor strata 會錯位。
    assert tuple(record.shard_id for record in records) == tuple(
        row["shard_id"] for row in plan_rows
    )
    assert all(isinstance(record, ValidatedTrajectoryShard) for record in records)

    controller = RunController(workspace, request_factory=_request)
    for record, plan_row in zip(records, plan_rows, strict=True):
        start = int(plan_row["scenario_start_index"])
        stop = int(plan_row["scenario_stop_index"])
        output = workspace / "shards" / str(plan_row["shard_id"])
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        shard = controller._shard(record.shard_id)
        units = tuple(iter_run_units(shard, master_seed=int(plan["master_seed"])))

        assert record.scenario_start_index == start
        assert record.scenario_stop_index == stop
        assert record.scenarios == ordered_scenarios[start:stop]
        assert all(isinstance(scenario, Scenario) for scenario in record.scenarios)
        assert record.results and isinstance(record.results, tuple)
        assert all(isinstance(result, ParticleResult) for result in record.results)
        assert record.output_relative_path == f"shards/{record.shard_id}"
        assert record.trajectory_manifest_sha256 == sha256_file(output / "manifest.json")
        assert record.particle_count == int(manifest["particle_count"]) == len(record.results)
        assert record.observation_count == int(manifest["observation_count"]) == sum(
            len(result.observations) for result in record.results
        )
        assert record.event_count == int(manifest["event_count"]) == sum(
            len(result.events) for result in record.results
        )
        assert record.particle_count == int(plan_row["particle_count"])
        assert len(record.scenarios) == int(plan_row["scenario_count"])

        # 這裡再次以 run unit 的正式順序核對 reader 回傳的 tuple；測試不只檢查數量，
        # 也確保 scenario、member、particle、站點、區域及 receptor identity 完整對齊。
        assert len(units) == len(record.results)
        for result, unit in zip(record.results, units, strict=True):
            state = result.final_state
            assert (
                state.particle_id,
                state.scenario_id,
                state.member_id,
                state.study_site_id,
                state.analysis_region_id,
                state.receptor_id,
            ) == (
                unit.particle_id,
                unit.scenario.scenario_id,
                unit.member_id,
                unit.scenario.study_site_id,
                unit.scenario.analysis_region_id,
                unit.scenario.receptor_id,
            )


def test_require_complete_gate_rejects_planned_and_partially_completed_runs(
    tmp_path: Path,
) -> None:
    """run 尚未全部 COMPLETE 時，聚合輸入 iterator 必須在讀取前停止。"""

    planned = _workspace(tmp_path / "planned", "planned-gate")
    _assert_iterator_rejects_without_absolute_path(planned, tmp_path)

    partial = _workspace(tmp_path / "partial", "partial-gate")
    shard_id = str(load_run_plan(partial)["shards"][0]["shard_id"])
    RunController(partial, request_factory=_request).run_shard(shard_id)
    _assert_iterator_rejects_without_absolute_path(partial, tmp_path)


@pytest.mark.parametrize("tamper", ["identity", "order"])
def test_output_identity_or_order_tampering_is_rejected(
    tmp_path: Path,
    tamper: str,
) -> None:
    """即使同步 particle payload checksum，run unit identity/order 竄改仍須被拒絕。"""

    workspace = _complete_run(tmp_path, f"output-{tamper}")
    shard_id = str(load_run_plan(workspace)["shards"][0]["shard_id"])
    output = workspace / "shards" / shard_id

    def mutate(rows: list[dict]) -> None:
        """製造一個 identity 或列順序與 immutable run plan 不一致的 payload。"""

        if tamper == "identity":
            rows[0]["scenario_id"] = "scn_tampered_identity"
        else:
            rows[0], rows[1] = rows[1], rows[0]

    _rewrite_particle_rows(output, mutate)
    _assert_iterator_rejects_without_absolute_path(workspace, tmp_path)


@pytest.mark.parametrize("tamper", ["count", "checksum"])
def test_manifest_count_or_checksum_tampering_is_rejected(
    tmp_path: Path,
    tamper: str,
) -> None:
    """trajectory manifest 的計數或固定 payload checksum 被改寫時不可讀取。"""

    workspace = _complete_run(tmp_path, f"manifest-{tamper}")
    shard_id = str(load_run_plan(workspace)["shards"][0]["shard_id"])
    output = workspace / "shards" / shard_id

    def mutate(manifest: dict) -> None:
        """只修改一個 manifest contract 欄位，保留其餘結構可解析。"""

        if tamper == "count":
            manifest["observation_count"] = int(manifest["observation_count"]) + 1
        else:
            manifest["files"]["events.parquet"]["sha256"] = "0" * 64

    _rewrite_manifest(output, mutate)
    _assert_iterator_rejects_without_absolute_path(workspace, tmp_path)


@pytest.mark.parametrize("damage", ["absolute_token", "symlink"])
def test_unsafe_output_relative_path_or_symlink_is_rejected(
    tmp_path: Path,
    damage: str,
) -> None:
    """progress 的 output token 與實際 output 目錄都必須是安全的 run 內普通拓撲。"""

    workspace = _complete_run(tmp_path, f"unsafe-output-{damage}")
    shard_id = str(load_run_plan(workspace)["shards"][0]["shard_id"])
    progress = load_run_progress(workspace)
    output = workspace / "shards" / shard_id
    if damage == "absolute_token":
        progress["shards"][shard_id]["output_relative_path"] = str(output)
        _write_progress(workspace, progress)
    else:
        # 把合法輸出移到 run 外再建立 symlink，保留完整 payload 以確保拒絕原因是
        # topology，而非檔案遺失；iterator 不得跟隨這個 link 進入任意目錄。
        target = tmp_path / f"moved-{damage}"
        output.rename(target)
        output.symlink_to(target, target_is_directory=True)

    _assert_iterator_rejects_without_absolute_path(workspace, tmp_path)


def test_external_checkpoint_root_is_forwarded_and_required_for_reading(
    tmp_path: Path,
) -> None:
    """external checkpoint root 必須由 iterator 傳入完整驗證與每一個 shard 讀取邊界。"""

    external_root = tmp_path / "external-checkpoints"
    workspace = _complete_run(
        tmp_path / "external-run",
        "external-root",
        checkpoint_root=external_root,
        pause_before_resume=True,
    )
    plan = load_run_plan(workspace)
    for row in plan["shards"]:
        checkpoint_parent = external_root / workspace.name / str(row["shard_id"])
        assert checkpoint_parent.is_dir()
        assert any(path.name.startswith("checkpoint-") for path in checkpoint_parent.iterdir())

    valid_with_external_root = validate_run(
        workspace,
        require_complete=True,
        checkpoint_root=external_root,
    )
    assert valid_with_external_root["valid"] is True
    records = tuple(
        iter_complete_run_trajectory_shards(
            workspace,
            checkpoint_root=external_root,
        )
    )
    assert tuple(record.shard_id for record in records) == tuple(
        row["shard_id"] for row in plan["shards"]
    )

    # 省略 external root 時，progress 宣告的 generation 在 default workspace 找不到；
    # 這可證明 iterator 不是只把參數留在 API，而是確實轉送到完整 run gate。
    _assert_iterator_rejects_without_absolute_path(workspace, tmp_path)
