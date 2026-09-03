"""Aggregate release 來源文件語意綁定的 bounded slice 測試。

本模組建立的是一份真正由 ``initialize_run_workspace`` 與 ``RunController`` 完成的
小型 synthetic 工程 run，再把其 plan、progress、normalized config、input inventory
及一份可載入的 ``AggregateSpec`` 以 exact bytes 複製進 release。它不假造 OCM／NWW
科學資料，也不宣稱產生真實來源足跡；事件與 pathway 數值只沿用既有小型 payload
fixture 來提供 source semantic validator 所需的完整產品拓撲。

本檔只直接測試 aggregate release 的私有 source-document semantic validator：合法
來源應完整通過，來源 plan、progress、設定、inventory、AggregateSpec 或 strict JSON
bytes 的竄改應在 source stage fail closed。產品 storage、codec decoder 與 public
reader／validator 由其他測試切片負責，避免把不同責任層混在同一組測試中。
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest
import test_aggregate_release_payload as payload_fixture
from test_run_control import _provenance, _request

import lagrangian_backtracking.aggregate_release as release_module
from lagrangian_backtracking.aggregate_release import (
    _ReleaseValidationError,
    _validate_release_topology_and_checksums,
    _write_encoded_products,
)
from lagrangian_backtracking.aggregate_release_codec import (
    AggregateReleaseMetadata,
    encode_aggregate_release_payload,
    metadata_from_payload,
)
from lagrangian_backtracking.aggregate_release_payload import AggregateReleasePayload
from lagrangian_backtracking.aggregate_spec import load_aggregate_spec
from lagrangian_backtracking.outputs import sha256_file
from lagrangian_backtracking.run_control import (
    RunController,
    initialize_run_workspace,
    load_run_plan,
    load_run_progress,
)
from lagrangian_backtracking.run_validation import validate_run
from lagrangian_backtracking.scenarios import Scenario, stable_identifier

_RUN_ID = "aggregate-payload-fixture"
_EXPERIMENT_CASE_ID = "synthetic-case"
_SOURCE_FILE_ORDER = (
    "aggregate_spec.json",
    "source_run_plan.json",
    "source_run_progress.json",
    "source_normalized_config.json",
    "source_input_inventory.json",
)


@dataclass(frozen=True, slots=True)
class _SourceReleaseFixture:
    """保存完整 synthetic release 與其真實 source run 預期值。

    ``workspace`` 是由 run controller 實際完成且 reconcile 後的 run root；``root`` 是
    由該 run 的 exact source bytes 與小型 aggregate payload 組成的 release root。所有
    path 都只存在測試暫存區，資料內容不代表任何 OCM／NWW 觀測、forcing 或科學結論。
    """

    workspace: Path
    root: Path
    payload: AggregateReleasePayload
    metadata: AggregateReleaseMetadata
    contracts: dict[str, dict[str, object]]
    source_bytes: dict[str, bytes]
    manifest_document: dict[str, object]
    plan: dict[str, Any]
    progress: dict[str, Any]


def _compact_json_bytes(document: object) -> bytes:
    """建立無換行的 production config canonical JSON bytes。

    config hash 的契約是 UTF-8、排序鍵、compact separators 且不含尾端換行；這裡
    刻意在測試端獨立計算，不重用 run-control 的私有 helper，確保 fixture 能檢出
    production canonicalization 改變。``allow_nan=False`` 同時保留 strict JSON 限制。
    """

    return json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _release_json_bytes(document: object) -> bytes:
    """建立 release manifest/source snapshot 使用的 compact JSON 加單一換行。"""

    return _compact_json_bytes(document) + b"\n"


def _pretty_json_bytes(document: object) -> bytes:
    """建立 initializer 寫入 immutable run input 的 exact JSON bytes。

    ``initialize_run_workspace`` 以排序鍵、兩格縮排與單一尾端換行保存 mapping；測試
    先用相同的公開文件格式建立 inventory input file，讓 plan 的 raw inventory digest
    與 run root 實際 ``input_inventory.json`` bytes 完全一致，而不是依賴猜測或事後修補
    immutable plan。
    """

    return (
        json.dumps(
            document,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _sha256_bytes(raw_bytes: bytes) -> str:
    """獨立計算測試 source snapshot 的 exact byte SHA-256。"""

    return sha256(raw_bytes).hexdigest()


def _scenario_for_stratum(stratum: payload_fixture.ScenarioStratum) -> Scenario:
    """由既有 payload stratum 建立能通過 run schema 2 的 immutable Scenario。

    payload fixture 的 ``scenario-a`` 是容器層測試用識別碼，並非 run-control 要求的
    stable hash。這裡保留 stratum 的站點、區域、材料、受體、arrival、沉降速度與時間
    欄位，只依 production 的 stable-identifier 規則產生 Scenario ID；稍後只替換
    stratum 的 ID 以完成真實 plan／scenario table 的語意 join，不改變任何物理量或分層
    欄位。
    """

    scenario_id = stable_identifier(
        "scn",
        [
            stratum.study_site_id,
            stratum.material_id,
            stratum.receptor_id,
            stratum.arrival_time_id,
            stratum.design_version,
        ],
    )
    return Scenario(
        scenario_id=scenario_id,
        study_site_id=stratum.study_site_id,
        analysis_region_id=stratum.analysis_region_id,
        material_id=stratum.material_id,
        receptor_id=stratum.receptor_id,
        arrival_time_id=stratum.arrival_time_id,
        settling_velocity_mps=stratum.settling_velocity_mps,
        arrival_time_utc_ns=stratum.arrival_time_utc_ns,
        design_version=stratum.design_version,
    )


def _source_validator() -> Callable[..., object]:
    """解析 production agent 實際落地的 source semantic helper。

    設計名稱預期為 ``_validate_source_documents``；若 implementation worker 為了明示
    bytes 邊界採用 ``_validate_source_documents_from_bytes``，測試只在這個 bounded slice
    保留同義名稱相容層。若兩者都不存在，測試會在執行時明確失敗，不能以假 helper 或
    public API 取代尚未落地的 source semantic gate。
    """

    for name in ("_validate_source_documents", "_validate_source_documents_from_bytes"):
        candidate = getattr(release_module, name, None)
        if callable(candidate):
            return candidate
    pytest.fail(
        "aggregate_release.py 尚未提供 _validate_source_documents(_from_bytes)；"
        "本測試不會自行修改 production API"
    )


def _call_source_validator(
    root: Path,
    metadata: AggregateReleaseMetadata,
    contracts: dict[str, dict[str, object]],
    source_bytes: dict[str, bytes],
) -> object:
    """以 release root 與三項 typed source snapshot 呼叫私有 semantic validator。

    metadata 是 manifest 已驗證的 release provenance/count；contracts 保存三十二檔
    manifest file contract；source_bytes 是 checksum 通過後取得的五份 exact JSON bytes。
    ``root`` 是含固定 ``aggregate_spec.json`` 的 release 根目錄，不是外部 source run
    workspace；production 只用它依固定檔名重載 AggregateSpec，其他四份來源仍以已通過
    checksum 的 exact bytes 驗證，不得猜測未登錄的外部檔案。
    """

    return _source_validator()(
        root,
        metadata=metadata,
        contracts=contracts,
        source_json_bytes=source_bytes,
    )


def _make_valid_source_release(tmp_path_factory: pytest.TempPathFactory) -> _SourceReleaseFixture:
    """建立由真實完成 run 綁定而成的 33 檔 synthetic engineering release。

    run 只有一個 scenario 與 ``M=2``，但仍經過 initializer、兩條 member trajectory、
    run progress COMPLETE 及 reconcile。payload 的事件／pathway 數值沿用既有小型
    aggregate fixture；source plan/progress/config/inventory 與 trajectory manifest 則
    直接從實體 run 讀取。這個 helper 只驗證工程發布與來源語意綁定，絕不是 OCM／NWW
    真實科學結果生成器。
    """

    base = tmp_path_factory.mktemp("aggregate-source-semantic")
    run_parent = base / "runs"
    stratum_template = payload_fixture._scenario_stratum()
    scenario = _scenario_for_stratum(stratum_template)

    normalized_config: dict[str, object] = {
        "schema_version": "synthetic-aggregate-source-v1",
        "execution": {
            "dt_seconds": 1.0,
            "max_age_seconds": 7.0,
            "members_per_scenario": 2,
        },
        "source": {"kind": "synthetic-engineering-fixture"},
    }
    config_hash = _sha256_bytes(_compact_json_bytes(normalized_config))
    input_inventory: dict[str, object] = {
        "schema_version": "synthetic-input-inventory-v1",
        "sources": [
            {
                "source_id": "synthetic-flow-placeholder",
                "role": "engineering-fixture-only",
            }
        ],
    }
    inventory_input_path = base / "input_inventory_input.json"
    inventory_input_path.write_bytes(_pretty_json_bytes(input_inventory))

    workspace = initialize_run_workspace(
        run_parent,
        run_id=_RUN_ID,
        scenarios=(scenario,),
        normalized_config=normalized_config,
        config_hash=config_hash,
        input_inventory_file=inventory_input_path,
        component_canonical_hashes={"synthetic_component": "a" * 64},
        geometry_canonical_hashes={"synthetic_geometry": "b" * 64},
        provenance=_provenance(),
        experiment_case_id=_EXPERIMENT_CASE_ID,
        master_seed=20260829,
        seed_policy="sha256_v1_pcg64dxsm",
        members_per_scenario=2,
        shard_scenario_count=1,
        checkpoint_interval_sweeps=1,
        active_chunk_size=2,
        run_kind="synthetic",
    )
    controller = RunController(workspace, request_factory=_request)
    summaries = controller.run_all()
    assert len(summaries) == 1
    assert summaries[0].lifecycle == "COMPLETE"
    reconciled = controller.reconcile()
    assert reconciled["run_lifecycle"] == "COMPLETE"

    plan = load_run_plan(workspace)
    progress = load_run_progress(workspace)
    assert validate_run(workspace)["valid"] is True
    assert plan["run_id"] == _RUN_ID
    assert plan["run_kind"] == "synthetic"
    assert plan["experiment_case_id"] == _EXPERIMENT_CASE_ID
    assert plan["members_per_scenario"] == 2
    assert plan["scenario_count"] == 1
    assert plan["particle_count"] == 2
    assert progress["run_lifecycle"] == "COMPLETE"

    shard_row = plan["shards"][0]
    shard_id = str(shard_row["shard_id"])
    progress_row = progress["shards"][shard_id]
    output_relative_path = progress_row["output_relative_path"]
    assert output_relative_path == summaries[0].output_relative_path
    assert isinstance(output_relative_path, str)
    trajectory_root = workspace / output_relative_path
    trajectory_manifest_path = trajectory_root / "manifest.json"
    trajectory_manifest = json.loads(trajectory_manifest_path.read_bytes().decode("utf-8"))
    assert trajectory_manifest["particle_count"] == 2
    assert trajectory_manifest["observation_count"] >= 2
    assert trajectory_manifest["event_count"] >= 2

    # AggregateSpec input JSON 不攜帶兩個衍生 hash；load_aggregate_spec 重新計算後，
    # source_sha256 對應原始 bytes、canonical_sha256 對應移除 hash 欄位的設定本體。
    spec_document = payload_fixture._aggregate_spec().to_dict()
    del spec_document["source_sha256"]
    del spec_document["canonical_sha256"]
    spec_input_path = base / "aggregate_spec_input.json"
    spec_input_path.write_bytes(_release_json_bytes(spec_document))
    aggregate_spec = load_aggregate_spec(spec_input_path)

    scenario_stratum = replace(
        stratum_template,
        # run-control schema 2 的 scenario table 以 stable hash 識別；其餘 stratum 欄位
        # 仍逐欄沿用 payload fixture，確保來源 plan 與分層 join 的唯一差異只有 ID。
        scenario_id=scenario.scenario_id,
    )
    assert scenario_stratum.study_site_id == scenario.study_site_id
    assert scenario_stratum.analysis_region_id == scenario.analysis_region_id
    assert scenario_stratum.material_id == scenario.material_id
    assert scenario_stratum.receptor_id == scenario.receptor_id
    assert scenario_stratum.arrival_time_id == scenario.arrival_time_id
    assert scenario_stratum.arrival_time_utc_ns == scenario.arrival_time_utc_ns
    assert scenario_stratum.settling_velocity_mps == scenario.settling_velocity_mps

    source_bytes = {
        "aggregate_spec.json": spec_input_path.read_bytes(),
        "source_run_plan.json": (workspace / "run_plan.json").read_bytes(),
        "source_run_progress.json": (workspace / "run_progress.json").read_bytes(),
        "source_normalized_config.json": (workspace / "normalized_config.json").read_bytes(),
        "source_input_inventory.json": (workspace / "input_inventory.json").read_bytes(),
    }
    assert _sha256_bytes(source_bytes["source_input_inventory.json"]) == plan[
        "raw_input_inventory_sha256"
    ]
    assert _sha256_bytes(source_bytes["source_normalized_config.json"]) != config_hash
    assert _sha256_bytes(source_bytes["source_run_plan.json"]) == sha256_file(
        workspace / "run_plan.json"
    )

    shard_binding = replace(
        payload_fixture._shard_binding(),
        shard_id=shard_id,
        scenario_start_index=int(shard_row["scenario_start_index"]),
        scenario_stop_index=int(shard_row["scenario_stop_index"]),
        output_relative_path=output_relative_path,
        trajectory_manifest_sha256=sha256_file(trajectory_manifest_path),
        particle_count=int(trajectory_manifest["particle_count"]),
        observation_count=int(trajectory_manifest["observation_count"]),
        event_count=int(trajectory_manifest["event_count"]),
    )
    assert shard_binding.particle_count == 2

    payload_kwargs = payload_fixture._valid_payload_kwargs()
    payload_kwargs.update(
        {
            "run_id": plan["run_id"],
            "run_kind": plan["run_kind"],
            "experiment_case_id": plan["experiment_case_id"],
            "members_per_scenario": plan["members_per_scenario"],
            "config_hash": plan["config_hash"],
            "checkpoint_input_binding_hash": plan["checkpoint_input_binding_hash"],
            "source_run_plan_sha256": _sha256_bytes(source_bytes["source_run_plan.json"]),
            "source_run_progress_sha256": _sha256_bytes(
                source_bytes["source_run_progress.json"]
            ),
            "source_normalized_config_sha256": _sha256_bytes(
                source_bytes["source_normalized_config.json"]
            ),
            "source_input_inventory_sha256": _sha256_bytes(
                source_bytes["source_input_inventory.json"]
            ),
            "aggregate_spec": aggregate_spec,
            "shard_bindings": [shard_binding],
            "scenario_strata": [scenario_stratum],
        }
    )
    payload = AggregateReleasePayload(**payload_kwargs)
    assert payload.run_id == plan["run_id"]
    assert payload.run_kind == plan["run_kind"]
    assert payload.members_per_scenario == 2
    assert len(payload.scenario_strata) == 1
    assert payload.event_aggregate.input_particle_count == 2

    products = encode_aggregate_release_payload(payload)
    root = base / f"{_RUN_ID}.aggregate-v1"
    root.mkdir()
    product_contracts = _write_encoded_products(root, products)

    source_contracts: dict[str, dict[str, object]] = {}
    for file_name, raw_bytes in source_bytes.items():
        (root / file_name).write_bytes(raw_bytes)
        source_contracts[file_name] = {
            "kind": "json",
            "size_bytes": len(raw_bytes),
            "sha256": _sha256_bytes(raw_bytes),
        }
    contracts = {
        **source_contracts,
        **{file_name: dict(contract) for file_name, contract in product_contracts.items()},
    }
    metadata = metadata_from_payload(payload)
    manifest_document: dict[str, object] = {
        "schema_version": metadata.schema_version,
        "metadata": metadata.to_dict(),
        "files": deepcopy(contracts),
    }
    (root / "aggregate_manifest.json").write_bytes(_release_json_bytes(manifest_document))
    return _SourceReleaseFixture(
        workspace=workspace,
        root=root,
        payload=payload,
        metadata=metadata,
        contracts=contracts,
        source_bytes=source_bytes,
        manifest_document=manifest_document,
        plan=plan,
        progress=progress,
    )


@pytest.fixture(scope="module")
def valid_source_release(tmp_path_factory: pytest.TempPathFactory) -> _SourceReleaseFixture:
    """供各 test 複製使用的合法 source semantic release。"""

    return _make_valid_source_release(tmp_path_factory)


def _copy_release(fixture: _SourceReleaseFixture, destination: Path) -> Path:
    """複製合法 release 到單一測試暫存區，避免 tamper 互相污染。"""

    target = destination / fixture.root.name
    shutil.copytree(fixture.root, target)
    return target


def _read_manifest(root: Path) -> dict[str, object]:
    """讀取尚未竄改的 canonical manifest，供測試 helper 建立獨立副本。"""

    document = json.loads((root / "aggregate_manifest.json").read_bytes().decode("utf-8"))
    assert type(document) is dict
    return document


def _write_manifest(root: Path, document: dict[str, object]) -> None:
    """以 production manifest canonical bytes 寫回測試 release。"""

    (root / "aggregate_manifest.json").write_bytes(_release_json_bytes(document))


def _replace_source_and_sync_manifest(
    root: Path,
    file_name: str,
    raw_bytes: bytes,
) -> None:
    """只同步指定 source file 的 manifest size/hash，不改 metadata 或 run plan。

    這個操作模擬 SERVER 上 source snapshot 被替換後，操作者只重新產生了 release
    manifest contract；metadata、plan cross-binding 與其他 source digest 仍維持原值，
    因而 source semantic validator 必須在 checksum 通過後拒絕，不可把新的 bytes 當成
    原始 run 的合法來源。
    """

    (root / file_name).write_bytes(raw_bytes)
    document = _read_manifest(root)
    files = document["files"]
    assert type(files) is dict
    contract = files[file_name]
    assert type(contract) is dict
    contract["size_bytes"] = len(raw_bytes)
    contract["sha256"] = _sha256_bytes(raw_bytes)
    _write_manifest(root, document)


def _source_inputs(
    root: Path,
) -> tuple[AggregateReleaseMetadata, dict[str, dict[str, object]], dict[str, bytes]]:
    """先通過固定 topology/checksum 層，再取得 source semantic helper 的 typed inputs。"""

    return _validate_release_topology_and_checksums(root, require_final_name=True)


def _metadata_with_hashes(
    metadata: AggregateReleaseMetadata,
    **hashes: str,
) -> AggregateReleaseMetadata:
    """以 ``dataclasses.replace`` 建立只更新指定來源摘要的 typed metadata。

    這些測試直接呼叫私有 source semantic helper，因此可在 foundation 已驗證 release
    topology、contracts 與 exact source bytes 後，傳入一份獨立的 metadata copy。同步
    外層 source digest 可避免案例過早停在 metadata-byte gate，讓 plan schema、progress
    COMPLETE、plan file binding、spec run identity 與 config canonical hash 各自被驗證。
    原 fixture 與 manifest metadata 都不會被修改。
    """

    allowed_fields = {
        "source_run_plan_sha256",
        "source_run_progress_sha256",
        "source_normalized_config_sha256",
        "source_input_inventory_sha256",
        "aggregate_spec_source_sha256",
        "aggregate_spec_canonical_sha256",
    }
    if not hashes or not set(hashes) <= allowed_fields:
        raise AssertionError("metadata hash updates 必須是非空且只含已登錄來源摘要")
    return replace(metadata, **hashes)


def _assert_source_stage_rejected(
    root: Path,
    *,
    metadata_hashes: dict[str, str],
) -> None:
    """同步 typed metadata 摘要後，確認深層 source gate 固定拒絕且不洩漏路徑。"""

    metadata, contracts, source_bytes = _source_inputs(root)
    synchronized_metadata = _metadata_with_hashes(metadata, **metadata_hashes)
    with pytest.raises(_ReleaseValidationError) as error_info:
        _call_source_validator(root, synchronized_metadata, contracts, source_bytes)
    error = error_info.value
    assert error.stage == "source"
    assert str(error) == "source"
    assert str(root) not in str(error)


def test_source_validator_accepts_real_completed_synthetic_run(
    valid_source_release: _SourceReleaseFixture,
) -> None:
    """真實完成且 reconcile 的 synthetic run source snapshot 應完整通過語意綁定。"""

    metadata, contracts, source_bytes = _source_inputs(valid_source_release.root)
    assert metadata == valid_source_release.metadata
    assert contracts == valid_source_release.contracts
    assert source_bytes == valid_source_release.source_bytes

    snapshot = _call_source_validator(
        valid_source_release.root,
        metadata,
        contracts,
        source_bytes,
    )
    assert snapshot is not None
    assert type(snapshot).__name__ == "_SourceSnapshot"

    # Source snapshot 必須保留已驗證的 spec、plan、progress 與兩個小型 input object；
    # 欄位名稱允許 implementation worker 在 ``plan``／``run_plan`` 等同義命名中擇一，
    # 但不接受只回傳 None 或未解析的 raw bytes。
    def snapshot_field(*names: str) -> object:
        """從 implementation 的同義欄位名稱取得 immutable semantic snapshot。"""

        for name in names:
            if hasattr(snapshot, name):
                return getattr(snapshot, name)
        pytest.fail(f"_SourceSnapshot 缺少欄位：{names!r}")

    assert snapshot_field("aggregate_spec", "spec") == valid_source_release.payload.aggregate_spec
    assert snapshot_field("plan", "run_plan") == valid_source_release.plan
    assert snapshot_field("progress", "run_progress") == valid_source_release.progress
    assert snapshot_field("normalized_config", "config") == json.loads(
        valid_source_release.source_bytes["source_normalized_config.json"]
    )
    assert snapshot_field("input_inventory", "inventory") == json.loads(
        valid_source_release.source_bytes["source_input_inventory.json"]
    )


def test_source_validator_rejects_plan_unknown_key(
    valid_source_release: _SourceReleaseFixture,
    tmp_path: Path,
) -> None:
    """plan 加入未登錄 key 即使 checksum 合法，也必須落在 source stage。"""

    root = _copy_release(valid_source_release, tmp_path)
    document = json.loads((root / "source_run_plan.json").read_bytes().decode("utf-8"))
    assert type(document) is dict
    document["unregistered_source_key"] = "tamper"
    changed = _release_json_bytes(document)
    _replace_source_and_sync_manifest(root, "source_run_plan.json", changed)
    _assert_source_stage_rejected(
        root,
        metadata_hashes={"source_run_plan_sha256": _sha256_bytes(changed)},
    )


def test_source_validator_rejects_legal_non_complete_progress(
    valid_source_release: _SourceReleaseFixture,
    tmp_path: Path,
) -> None:
    """progress 改成結構合法但尚未 COMPLETE 的狀態，不得發布 aggregate source。"""

    root = _copy_release(valid_source_release, tmp_path)
    document = json.loads((root / "source_run_progress.json").read_bytes().decode("utf-8"))
    assert type(document) is dict
    shards = document["shards"]
    assert type(shards) is dict and len(shards) == 1
    row = next(iter(shards.values()))
    assert type(row) is dict
    row["lifecycle"] = "RUNNING"
    row["output_relative_path"] = None
    row["failure_relative_path"] = None
    row["error_code"] = None
    row["attempt_count"] = max(1, int(row["attempt_count"]))
    document["run_lifecycle"] = "RUNNING"
    changed = _release_json_bytes(document)
    _replace_source_and_sync_manifest(root, "source_run_progress.json", changed)
    _assert_source_stage_rejected(
        root,
        metadata_hashes={"source_run_progress_sha256": _sha256_bytes(changed)},
    )


def test_source_validator_rejects_normalized_config_canonical_hash_mismatch(
    valid_source_release: _SourceReleaseFixture,
    tmp_path: Path,
) -> None:
    """normalized config 改值後 canonical compact hash 不再等於 plan／metadata。"""

    root = _copy_release(valid_source_release, tmp_path)
    document = json.loads(
        (root / "source_normalized_config.json").read_bytes().decode("utf-8")
    )
    assert type(document) is dict
    execution = document["execution"]
    assert type(execution) is dict
    execution["dt_seconds"] = 2.0
    changed_config = _release_json_bytes(document)
    assert _sha256_bytes(_compact_json_bytes(document)) != valid_source_release.metadata.config_hash
    _replace_source_and_sync_manifest(
        root,
        "source_normalized_config.json",
        changed_config,
    )

    # 將 source plan 內 immutable config file contract 同步到新 bytes，使 validator 能
    # 通過 plan-file binding；plan.config_hash 與 release metadata.config_hash 刻意保持
    # 原值，唯一剩餘差異是 normalized config 的 compact canonical hash。
    plan_document = json.loads((root / "source_run_plan.json").read_bytes().decode("utf-8"))
    assert type(plan_document) is dict
    plan_files = plan_document["files"]
    assert type(plan_files) is dict
    config_contract = plan_files["normalized_config.json"]
    assert type(config_contract) is dict
    config_contract["size_bytes"] = len(changed_config)
    config_contract["sha256"] = _sha256_bytes(changed_config)
    assert plan_document["config_hash"] == valid_source_release.metadata.config_hash
    changed_plan = _release_json_bytes(plan_document)
    _replace_source_and_sync_manifest(root, "source_run_plan.json", changed_plan)

    _assert_source_stage_rejected(
        root,
        metadata_hashes={
            "source_normalized_config_sha256": _sha256_bytes(changed_config),
            "source_run_plan_sha256": _sha256_bytes(changed_plan),
        },
    )


def test_source_validator_rejects_inventory_plan_file_contract_mismatch(
    valid_source_release: _SourceReleaseFixture,
    tmp_path: Path,
) -> None:
    """inventory bytes 被替換且只同步 release contract 時，仍與 source plan contract 不同。"""

    root = _copy_release(valid_source_release, tmp_path)
    document = json.loads((root / "source_input_inventory.json").read_bytes().decode("utf-8"))
    assert type(document) is dict
    document["inventory_revision"] = 2
    changed = _release_json_bytes(document)
    assert changed != valid_source_release.source_bytes["source_input_inventory.json"]
    _replace_source_and_sync_manifest(root, "source_input_inventory.json", changed)
    _assert_source_stage_rejected(
        root,
        metadata_hashes={"source_input_inventory_sha256": _sha256_bytes(changed)},
    )


def test_source_validator_rejects_loadable_aggregate_spec_source_or_run_id_mismatch(
    valid_source_release: _SourceReleaseFixture,
    tmp_path: Path,
) -> None:
    """可載入但換 run_id 的 AggregateSpec 不得冒充原 release spec。"""

    root = _copy_release(valid_source_release, tmp_path)
    document = json.loads((root / "aggregate_spec.json").read_bytes().decode("utf-8"))
    assert type(document) is dict
    document["run_id"] = "other-synthetic-run"
    changed = _release_json_bytes(document)
    replacement_path = tmp_path / "replacement-aggregate-spec.json"
    replacement_path.write_bytes(changed)
    replacement = load_aggregate_spec(replacement_path)
    assert replacement.run_id == "other-synthetic-run"
    _replace_source_and_sync_manifest(root, "aggregate_spec.json", changed)
    _assert_source_stage_rejected(
        root,
        metadata_hashes={
            "aggregate_spec_source_sha256": replacement.source_sha256,
            "aggregate_spec_canonical_sha256": replacement.canonical_sha256,
        },
    )


@pytest.mark.parametrize(
    ("label", "raw_bytes"),
    (
        (
            "duplicate-key",
            b'{"schema_version":"synthetic-aggregate-source-v1",'
            b'"schema_version":"synthetic-aggregate-source-v1"}\n',
        ),
        (
            "nan",
            b'{"schema_version":"synthetic-aggregate-source-v1","probe":NaN}\n',
        ),
        (
            "overflow-number",
            b'{"schema_version":"synthetic-aggregate-source-v1","probe":1e999}\n',
        ),
    ),
)
def test_source_validator_rejects_strict_json_source_bytes(
    valid_source_release: _SourceReleaseFixture,
    tmp_path: Path,
    label: str,
    raw_bytes: bytes,
) -> None:
    """source JSON 的 duplicate key、NaN 與極大指數都不得繞過 strict parser。"""

    del label  # pytest id 已辨識案例；測試語意由 raw bytes 本身決定。
    root = _copy_release(valid_source_release, tmp_path)
    _replace_source_and_sync_manifest(root, "source_normalized_config.json", raw_bytes)
    _assert_source_stage_rejected(
        root,
        metadata_hashes={"source_normalized_config_sha256": _sha256_bytes(raw_bytes)},
    )
