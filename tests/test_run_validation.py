"""只讀 run validator、streaming seed 與 benchmark report 契約測試。"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_run_control import _first_shard, _request, _workspace

from lagrangian_backtracking.outputs import TRAJECTORY_SHARD_SCHEMA_VERSION, sha256_file
from lagrangian_backtracking.run_control import RunController, load_run_progress
from lagrangian_backtracking.run_validation import (
    benchmark_report,
    iter_complete_run_trajectory_shards,
    validate_run,
)

_LEGACY_TRAJECTORY_SHARD_SCHEMA_VERSION = "1.0.0"
_ENVIRONMENT_PAYLOAD_FILES = (
    "environment_sample_status_code.npy",
    "eta_m.npy",
    "bed_z_m.npy",
    "forcing_month_yyyymm.npy",
    "environment_qc_flags.npy",
)


def _complete_workspace(parent: Path, run_id: str) -> Path:
    """依正式 controller 流程完成一個小型 synthetic run，供 iterator 契約測試使用。

    ``_workspace`` 建立兩個 scenario、每個 scenario 兩個 member 的 immutable run；這裡
    逐一執行其 plan 宣告的 shard，讓 output、progress、checkpoint 與 run lifecycle 都
    由正式寫入路徑產生。測試只使用少量合成粒子，因此可驗證完整資料契約而不引入正式
    OCM/NWW3 資料；回傳的路徑僅限 pytest 暫存目錄。
    """

    workspace = _workspace(parent, run_id)
    controller = RunController(workspace, request_factory=_request)
    for shard_id in load_run_progress(workspace)["shards"]:
        controller.run_shard(shard_id)
    return workspace


def _downgrade_shard_to_legacy(shard: Path) -> None:
    """將已驗證的 v2 shard 降為唯讀相容的 v1 fixture，保留 base payload 與計數。

    legacy v1 的固定拓撲沒有 environment context 五個 NPY 欄位；測試只移除這些 v2
    payload 並同步 manifest 的檔案契約與版本，藉此確認原有工程 validator/reader 能通過
    v1，而 ``iter_complete_run_trajectory_shards`` 仍會把實際版本原值交給下游辨識。
    這個 helper 不模擬或重建科學資料，也不把 v1 宣稱為正式 F03/F09 輸入。
    """

    manifest_path = shard / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for filename in _ENVIRONMENT_PAYLOAD_FILES:
        (shard / filename).unlink()
        manifest["files"].pop(filename)
    manifest["schema_version"] = _LEGACY_TRAJECTORY_SHARD_SCHEMA_VERSION
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_validate_run_missing_path_and_invalid_benchmark_are_json_safe(tmp_path: Path) -> None:
    """不存在 run 不拋例外，benchmark 仍明示工程量測與科學結果的界線。"""

    validation = validate_run(tmp_path / "does-not-exist")
    assert validation["valid"] is False
    assert isinstance(validation["errors"], list)
    assert isinstance(validation["summary"], dict)
    report = benchmark_report(tmp_path / "missing")
    assert report["valid"] is False
    assert report["engineering_measurement_not_scientific_result"] is True
    assert report["pilot_cannot_redefine_five_site_50000_baseline"] is True


@pytest.mark.parametrize("filename", ["run_plan.json", "run_progress.json"])
def test_malformed_plan_or_progress_returns_invalid(tmp_path: Path, filename: str) -> None:
    """plan/progress 壞 JSON 都應轉為 valid=false，不洩漏絕對 fixture 路徑。"""

    workspace = _workspace(tmp_path, f"malformed-{filename.split('.')[0]}")
    (workspace / filename).write_text("{not-json", encoding="utf-8")
    result = validate_run(workspace)
    assert result["valid"] is False
    assert "validator_exception" not in " ".join(result["errors"])
    assert str(tmp_path) not in repr(result)


def test_seed_validation_streams_without_read_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """seed table 必須用 Parquet batches 驗證，不能把 50,000×M rows 一次載入。"""

    workspace = _workspace(tmp_path, "streaming-seed")
    original = pq.read_table

    def reject_seed_read_table(source, *args, **kwargs):
        """若 validator 對 seed_table 呼叫 read_table，測試立即失敗。"""

        if Path(source).name == "seed_table.parquet":
            raise AssertionError("seed_table 不可使用 pq.read_table")
        return original(source, *args, **kwargs)

    monkeypatch.setattr(pq, "read_table", reject_seed_read_table)
    result = validate_run(workspace)
    assert result["valid"], result


def test_complete_trajectory_shard_exposes_current_schema_version(tmp_path: Path) -> None:
    """完整 current shard 的 container 必須逐字保存公開 v2 schema 常數。

    iterator 會先經過 run-level 與 trajectory-level validator，再從已驗證 manifest 產生
    ``ValidatedTrajectoryShard``；此測試確認下游取得的是 manifest 的資料契約版本，而不
    是由 reader 是否能解碼 payload 推測的版本。current v2 的欄位代表含完整 environment
    context 的 payload，正式 F03/F09 gate 可據此做 exact match。
    """

    workspace = _complete_workspace(tmp_path, "current-trajectory-schema")
    records = tuple(iter_complete_run_trajectory_shards(workspace))

    assert records
    assert all(shard.trajectory_schema_version == TRAJECTORY_SHARD_SCHEMA_VERSION for shard in records)


def test_legacy_trajectory_shard_schema_is_preserved_for_engineering_reader(
    tmp_path: Path,
) -> None:
    """legacy v1 通過既有 reader 時仍須回傳 1.0.0，且 iterator 不替工程聚合淘汰它。

    fixture 只把正式 writer 產生的 v2 shard 降為固定 v1 拓撲；因此 validator、reader、
    run identity/order 與計數仍走真實路徑。v1 沒有 environment context，這裡驗證的是
    相容性與版本辨識，不是把 legacy 資料提升成正式 F03/F09 的完整環境輸入。
    """

    workspace = _complete_workspace(tmp_path, "legacy-trajectory-schema")
    for shard_id in load_run_progress(workspace)["shards"]:
        _downgrade_shard_to_legacy(workspace / "shards" / shard_id)

    records = tuple(iter_complete_run_trajectory_shards(workspace))

    assert records
    assert tuple(record.trajectory_schema_version for record in records) == tuple(
        _LEGACY_TRAJECTORY_SHARD_SCHEMA_VERSION for _ in records
    )


@pytest.mark.parametrize("schema_mutation", ["missing", "non_string"])
def test_manifest_schema_boundary_fails_closed_without_exposing_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    schema_mutation: str,
) -> None:
    """validator 回傳的 manifest 缺 schema 或含非字串 schema 時，公開 iterator 必須安全失敗。

    ``outputs.validate_trajectory_shard`` 本身通常會先拒絕這兩種 manifest；這裡以
    monkeypatch 精確隔離 iterator 的「validator 回傳 manifest」邊界，確保新增的型別檢查
    不會因上游 validator 行為改變而被繞過。reader 也被設為不可呼叫，證明 malformed
    schema 尚未獲得版本資格前不會讀取 payload；公開例外只給固定訊息，不帶暫存部署路徑。
    """

    workspace = _complete_workspace(tmp_path, f"malformed-trajectory-schema-{schema_mutation}")
    manifest_path = next((workspace / "shards").glob("*/manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if schema_mutation == "missing":
        manifest.pop("schema_version")
    else:
        manifest["schema_version"] = 20260829

    def fake_validate_trajectory_shard(path, **kwargs):
        """只回傳指定 malformed manifest，讓測試命中 iterator 的版本邊界。"""

        del path, kwargs
        return {"valid": True, "errors": [], "manifest": manifest}

    reader_called = False

    def reject_reader(*args, **kwargs):
        """schema 尚未通過型別 gate 前，reader 不得開始讀 payload。"""

        nonlocal reader_called
        del args, kwargs
        reader_called = True
        raise AssertionError("malformed schema 不得進入 trajectory reader")

    monkeypatch.setattr(
        "lagrangian_backtracking.run_validation.validate_trajectory_shard",
        fake_validate_trajectory_shard,
    )
    monkeypatch.setattr("lagrangian_backtracking.run_validation.read_trajectory_shard", reject_reader)

    with pytest.raises(ValueError, match="完整 run trajectory shards 讀取失敗") as raised:
        tuple(iter_complete_run_trajectory_shards(workspace))

    assert reader_called is False
    assert str(tmp_path) not in str(raised.value)


def test_complete_output_full_identity_tamper_is_detected(tmp_path: Path) -> None:
    """重算 manifest checksum 仍不能掩蓋 scenario/member/site/receptor 身分竄改。"""

    workspace = _workspace(tmp_path, "identity-tamper")
    shard_id = _first_shard(workspace)
    RunController(workspace, request_factory=_request).run_shard(shard_id)
    output = workspace / "shards" / shard_id
    particle_path = output / "particle_table.parquet"
    table = pq.read_table(particle_path)
    rows = table.to_pylist()
    rows[0]["scenario_id"] = "scn_tampered_identity"
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), particle_path)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["particle_table.parquet"] = {
        "size_bytes": particle_path.stat().st_size,
        "sha256": sha256_file(particle_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result = validate_run(workspace)
    assert result["valid"] is False
    assert any("run_unit_identity_or_order" in error for error in result["errors"])


def test_malformed_checkpoint_and_output_are_json_safe(tmp_path: Path) -> None:
    """latest/output manifest 壞 JSON 都回 valid=false，不由 KeyError/JSONDecodeError 中斷。"""

    checkpoint_run = _workspace(tmp_path / "checkpoint", "bad-latest")
    shard_id = _first_shard(checkpoint_run)
    RunController(checkpoint_run, request_factory=_request).run_shard(shard_id, sweep_budget=1)
    latest = checkpoint_run / "checkpoints" / checkpoint_run.name / shard_id / "latest.json"
    latest.write_text("{bad-latest", encoding="utf-8")
    checkpoint_result = validate_run(checkpoint_run)
    assert checkpoint_result["valid"] is False
    assert isinstance(checkpoint_result["errors"], list)

    output_run = _workspace(tmp_path / "output", "bad-output")
    output_shard = _first_shard(output_run)
    RunController(output_run, request_factory=_request).run_shard(output_shard)
    manifest = output_run / "shards" / output_shard / "manifest.json"
    manifest.write_text("{bad-output", encoding="utf-8")
    output_result = validate_run(output_run)
    assert output_result["valid"] is False
    assert isinstance(output_result["errors"], list)


def test_progress_range_lifecycle_and_unknown_topology_are_rejected(tmp_path: Path) -> None:
    """progress range、COMPLETE iff 與 output/failure unknown entry 都必須 fail-closed。"""

    workspace = _workspace(tmp_path / "range", "bad-range")
    progress_path = workspace / "run_progress.json"
    progress = load_run_progress(workspace)
    shard_id = _first_shard(workspace)
    progress["shards"][shard_id]["scenario_stop_index"] += 1
    progress_path.write_text(json.dumps(progress), encoding="utf-8")
    assert not validate_run(workspace)["valid"]

    lifecycle = _workspace(tmp_path / "lifecycle", "bad-lifecycle")
    progress = load_run_progress(lifecycle)
    progress["run_lifecycle"] = "COMPLETE"
    (lifecycle / "run_progress.json").write_text(json.dumps(progress), encoding="utf-8")
    assert not validate_run(lifecycle)["valid"]

    topology = _workspace(tmp_path / "topology", "bad-topology")
    (topology / "shards" / "unknown-shard").mkdir()
    (topology / "failures" / "unknown-shard").mkdir()
    result = validate_run(topology)
    assert result["valid"] is False
    assert any("unknown" in error for error in result["errors"])


@pytest.mark.parametrize("damage", ["missing", "unknown", "nonzero", "symlink"])
def test_lock_topology_damage_is_json_safe_invalid(tmp_path: Path, damage: str) -> None:
    """lock topology 缺失、未知、非零或 symlink 都必須被只讀 validator 拒絕。"""

    workspace = _workspace(tmp_path, f"lock-{damage}")
    lock_root = workspace / "locks"
    if damage == "missing":
        (lock_root / "run_gate.lock").unlink()
    elif damage == "unknown":
        (lock_root / "unexpected.lock").touch()
    elif damage == "nonzero":
        (lock_root / "progress.lock").write_text("busy", encoding="utf-8")
    else:
        target = tmp_path / "lock-target"
        target.touch()
        (lock_root / "run_gate.lock").unlink()
        (lock_root / "run_gate.lock").symlink_to(target)
    result = validate_run(workspace)
    assert result["valid"] is False
    assert isinstance(result["errors"], list)


@pytest.mark.parametrize(
    "tamper",
    ["policy", "group_id", "region", "arrival", "part_index", "part_count", "row_order"],
)
def test_schema2_ordering_and_group_tamper_is_rejected(tmp_path: Path, tamper: str) -> None:
    """ordering policy、group metadata 或 plan row 順序被改寫時不可通過驗證。"""

    workspace = _workspace(tmp_path, f"plan-tamper-{tamper}")
    plan_path = workspace / "run_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if tamper == "policy":
        plan["scenario_ordering_policy"] = "old_lexical_order"
    elif tamper == "group_id":
        plan["shards"][0]["execution_group_id"] = "grp_tampered"
    elif tamper == "region":
        plan["shards"][0]["analysis_region_id"] = "B"
    elif tamper == "arrival":
        plan["shards"][0]["arrival_time_utc_ns"] += 1
    elif tamper == "part_index":
        plan["shards"][0]["group_part_index"] = 1
    elif tamper == "part_count":
        plan["shards"][0]["group_part_count"] = 2
    else:
        plan["shards"] = list(reversed(plan["shards"]))
    plan_path.write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    result = validate_run(workspace)
    assert result["valid"] is False
    assert isinstance(result["errors"], list)
