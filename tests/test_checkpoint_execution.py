"""execution checkpoint 舊版相容、RNG continuation 與版本相容性拒絕測試。

本檔所有 request、位置與環境欄位都是本機建立的 synthetic 工程資料，只驗證序列化、
恢復、checksum 與 fail-closed 邊界，不代表真實 OCM／NWW3 forcing 或任何科學成果。
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from shapely.geometry import box

from lagrangian_backtracking.boundaries import BoundaryGeometry
from lagrangian_backtracking.checkpoint import (
    CheckpointBinding,
    _write_execution_checkpoint_schema22,
    load_execution_checkpoint,
)
from lagrangian_backtracking.diffusion import DiffusionCoefficients
from lagrangian_backtracking.engine import EngineSettings, EnvironmentSampleStatus
from lagrangian_backtracking.models import EventType, ParticleState, VelocitySampleStatus
from lagrangian_backtracking.outputs import sha256_file
from lagrangian_backtracking.production import ProductionBatch
from lagrangian_backtracking.runner import ReferenceParticleRequest, plan_scenario_shards
from lagrangian_backtracking.scenarios import Scenario


def _shard(scenario_id: str = "s0"):
    """建立一個含兩個 member 的非空 shard，供重啟測試固定使用。"""

    scenario = Scenario(
        scenario_id=scenario_id,
        study_site_id="gongliao",
        analysis_region_id="A",
        material_id="oca_nonrecyclable_flexible_sheet",
        receptor_id=f"receptor-{scenario_id}",
        arrival_time_id="arrival-0",
        settling_velocity_mps=-0.001,
        arrival_time_utc_ns=100_000_000_000,
        design_version="test-v1",
    )
    return plan_scenario_shards(
        [scenario],
        members_per_scenario=2,
        shard_scenario_count=1,
        experiment_case_id="baseline",
    )[0]


def _settings() -> EngineSettings:
    """提供足夠長的視窗，讓 checkpoint 時保留一條 active member。"""

    return EngineSettings(1.0, 1.0, 2.0, 8.0, 100, 0)


def _factory(unit) -> ReferenceParticleRequest:
    """member 0 以窄 flow domain 產生真實 boundary event，member 1 保留 Brownian 路徑。"""

    state = ParticleState(
        particle_id=unit.particle_id,
        scenario_id=unit.scenario.scenario_id,
        member_id=unit.member_id,
        study_site_id=unit.scenario.study_site_id,
        analysis_region_id=unit.scenario.analysis_region_id,
        receptor_id=unit.scenario.receptor_id,
        x_m=0.0,
        y_m=0.0,
        z_m=-10.0,
        time_utc_ns=unit.scenario.arrival_time_utc_ns,
    )

    def velocity(x_m: float, y_m: float, z_m: float, time_utc_ns: int):
        """固定向東正向流；backward 時向西，方便穩定穿越窄 flow boundary。"""

        del x_m, y_m, z_m, time_utc_ns
        from lagrangian_backtracking.models import VelocitySample

        return VelocitySample(0.2, 0.0, 0.0, 0.0, -100.0, 100.0, 10.0)

    if unit.member_id == 0:
        boundaries = BoundaryGeometry(
            own_local_domain=box(-0.05, -10.0, 100.0, 10.0),
            flow_domain=box(-0.15, -10.0, 100.0, 10.0),
            foreign_local_domains={},
        )
        diffusion = DiffusionCoefficients(0.0, 0.0, 0.0)
    else:
        boundaries = BoundaryGeometry(
            own_local_domain=box(-100.0, -100.0, 100.0, 100.0),
            flow_domain=box(-1_000.0, -1_000.0, 1_000.0, 1_000.0),
            foreign_local_domains={},
        )
        diffusion = DiffusionCoefficients(0.2, 0.1, 0.03)
    return ReferenceParticleRequest(
        initial_state=state,
        velocity=velocity,
        boundaries=boundaries,
        behavior_class="sinking",
        diffusion=diffusion,
        settings=_settings(),
    )


def _binding() -> CheckpointBinding:
    """建立與資料來源、程式版本及 seed policy 綁定的測試識別。"""

    return CheckpointBinding("config", "inventory", "baseline", "shard", "pcg64dxsm-v1", "commit")


def _write_partial_checkpoint(destination: Path, *, master_seed: int = 5) -> Path:
    """建立尚有可重啟狀態的 2.2 checkpoint，供完整性篡改測試重複使用。"""

    batch = ProductionBatch(_shard(), master_seed=master_seed, request_factory=_factory)
    batch.advance()
    return _write_execution_checkpoint_schema22(
        destination,
        binding=_binding(),
        run_units=batch.units,
        executions=[runtime.execution for runtime in batch.runtimes],
        rngs=[runtime.rng for runtime in batch.runtimes],
        triangle_hints=[runtime.triangle_hint for runtime in batch.runtimes],
        sequence=1,
    )


def _write_legacy_batch_checkpoint(
    batch: ProductionBatch,
    destination: Path,
    *,
    sequence: int,
) -> Path:
    """將已建立的 synthetic batch 寫成 2.2 fixture；正式 batch writer 固定使用 v3。"""

    return _write_execution_checkpoint_schema22(
        destination,
        binding=_binding(),
        run_units=batch.units,
        executions=[runtime.execution for runtime in batch.runtimes],
        rngs=[runtime.rng for runtime in batch.runtimes],
        triangle_hints=[runtime.triangle_hint for runtime in batch.runtimes],
        sequence=sequence,
    )


def _write_context_checkpoint(destination: Path) -> Path:
    """建立含三種環境 context 狀態的 synthetic 2.2 checkpoint。

    第一個 observation 以有限公尺制海面／海床、合法 UTC 月份與零品質旗標代表有效
    樣本；第二個以非零品質旗標代表取樣失敗；其餘觀測保留 ``NOT_SAMPLED``。這五個
    環境欄位與 11 個速度欄位只用來驗證 checkpoint 保存與 constructor cross-field gate；
    速度分項仍是 callback 能明示提供的 synthetic data，並非 OCM／NWW3 實測。
    """

    batch = ProductionBatch(_shard(), master_seed=5, request_factory=_factory)
    batch.advance()
    valid_runtime = batch.runtimes[0]
    valid_observation = valid_runtime.execution.observations[0]
    valid_runtime.execution.observations[0] = replace(
        valid_observation,
        environment_sample_status=EnvironmentSampleStatus.VALID,
        eta_m=0.0,
        bed_z_m=-20.0,
        forcing_month_id="202401",
        environment_qc_flags=0,
    )
    invalid_runtime = batch.runtimes[1]
    invalid_observation = invalid_runtime.execution.observations[0]
    invalid_runtime.execution.observations[0] = replace(
        invalid_observation,
        environment_sample_status=EnvironmentSampleStatus.INVALID,
        environment_qc_flags=1,
    )
    return _write_execution_checkpoint_schema22(
        destination,
        binding=_binding(),
        run_units=batch.units,
        executions=[runtime.execution for runtime in batch.runtimes],
        rngs=[runtime.rng for runtime in batch.runtimes],
        triangle_hints=[runtime.triangle_hint for runtime in batch.runtimes],
        sequence=1,
    )


def _refresh_payload_checksum(root: Path, filename: str) -> None:
    """篡改測試後同步更新指定 payload 的 manifest，隔離 semantic gate 與 checksum gate。"""

    payload_path = root / filename
    manifest_path = root / "checkpoint.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][filename] = {
        "size_bytes": payload_path.stat().st_size,
        "sha256": sha256_file(payload_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _set_nested_value(payload: Any, path: tuple[str | int, ...], replacement: Any) -> None:
    """在測試用 JSON 結構中修改單一欄位，讓篡改案例維持清楚的資料路徑。"""

    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement


def _rewrite_execution_payload(root: Path, mutate) -> None:
    """修改單一 synthetic execution payload 後同步 checksum，專門測 semantic gate。"""

    execution_path = root / "execution_state.json"
    payload = json.loads(execution_path.read_text(encoding="utf-8"))
    mutate(payload)
    # allow_nan=True 是 Python json 的預設，故意保留 NaN/Infinity 到 loader 的 strict
    # parser；這能證明非有限值不是被測試 helper 先轉成 None 而漏掉拒絕路徑。
    execution_path.write_text(json.dumps(payload), encoding="utf-8")
    _refresh_payload_checksum(root, "execution_state.json")


def _set_schema_version(root: Path, schema_version: str) -> None:
    """只改 metadata schema version，保留其餘 manifest 欄位與 payload checksum。"""

    metadata_path = root / "checkpoint.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["schema_version"] = schema_version
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _downgrade_fixture_to_20(root: Path) -> None:
    """把現行 2.2 fixture 安全裁成可驗證的 2.0 legacy fixture。

    2.0 的 observation 只有原始七欄，因此這裡明確移除環境五欄與速度十一欄，而不是只
    改 metadata version。helper 只在測試目錄中重算 execution_state checksum 並改 metadata
    version，模擬既有 2.0 檔案；production writer 本身沒有 downgrade 路徑。
    """

    execution_path = root / "execution_state.json"
    payload = json.loads(execution_path.read_text(encoding="utf-8"))
    for record in payload["records"]:
        for observation in record["execution"]["observations"]:
            observation.pop("environment_sample_status", None)
            observation.pop("eta_m", None)
            observation.pop("bed_z_m", None)
            observation.pop("forcing_month_id", None)
            observation.pop("environment_qc_flags", None)
            observation.pop("velocity_sample_status", None)
            observation.pop("total_u_mps", None)
            observation.pop("total_v_mps", None)
            observation.pop("total_w_mps", None)
            observation.pop("ocm_u_mps", None)
            observation.pop("ocm_v_mps", None)
            observation.pop("ocm_w_mps", None)
            observation.pop("stokes_u_mps", None)
            observation.pop("stokes_v_mps", None)
            observation.pop("settling_w_mps", None)
            observation.pop("velocity_qc_flags", None)
    execution_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _refresh_payload_checksum(root, "execution_state.json")
    _set_schema_version(root, "2.0.0")


def _downgrade_fixture_to_21(root: Path) -> None:
    """把現行 2.2 fixture 精確裁成 schema 2.1 的固定觀測欄位集合。

    2.1 允許原始七欄加環境五欄，但尚未保存速度紀錄；逐欄移除 11 個速度欄位可驗證
    loader 真正依 metadata 選擇舊拓撲，而不是讓最新 ``Observation`` dataclass 污染舊檔。
    這只是測試用的可逆 fixture 轉換，不代表 production writer 支援 downgrade。
    """

    execution_path = root / "execution_state.json"
    payload = json.loads(execution_path.read_text(encoding="utf-8"))
    for record in payload["records"]:
        for observation in record["execution"]["observations"]:
            observation.pop("velocity_sample_status", None)
            observation.pop("total_u_mps", None)
            observation.pop("total_v_mps", None)
            observation.pop("total_w_mps", None)
            observation.pop("ocm_u_mps", None)
            observation.pop("ocm_v_mps", None)
            observation.pop("ocm_w_mps", None)
            observation.pop("stokes_u_mps", None)
            observation.pop("stokes_v_mps", None)
            observation.pop("settling_w_mps", None)
            observation.pop("velocity_qc_flags", None)
    execution_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _refresh_payload_checksum(root, "execution_state.json")
    _set_schema_version(root, "2.1.0")


def test_partial_execution_checkpoint_restores_exact_results_and_rng(tmp_path: Path) -> None:
    """數個 sweep 後 schema 3 restore 必須與不中斷完成結果逐欄完全相同。"""

    shard = _shard()
    uninterrupted = ProductionBatch(shard, master_seed=123, request_factory=_factory).complete()

    interrupted = ProductionBatch(shard, master_seed=123, request_factory=_factory)
    interrupted.advance(sweeps=1)
    first_checkpoint = interrupted.write_checkpoint(
        tmp_path / "checkpoint-00000001", binding=_binding(), sequence=1
    )
    interrupted.advance(sweeps=1)
    assert interrupted.active_count == 1
    checkpoint_path = interrupted.write_checkpoint(
        tmp_path / "checkpoint-00000002",
        binding=_binding(),
        sequence=2,
        previous_checkpoint=first_checkpoint,
    )

    loaded = load_execution_checkpoint(
        checkpoint_path,
        expected_binding=_binding(),
        expected_run_units=interrupted.units,
    )
    assert loaded.sequence == 2
    assert len(loaded.executions) == 2
    assert loaded.executions[0].events
    assert [event.event_type for event in loaded.executions[0].events] == [
        EventType.LOCAL_DOMAIN_FIRST_EXIT,
        EventType.FLOW_DOMAIN_OPEN_EXIT,
    ]

    restored_factory_calls: Counter[str] = Counter()

    def restored_factory(unit):
        """記錄 restore 重建外部 request 的次數，確認每個 unit 只建立一次。"""

        restored_factory_calls[unit.particle_id] += 1
        return _factory(unit)

    restored = ProductionBatch.from_checkpoint(
        checkpoint_path,
        shard=shard,
        master_seed=123,
        request_factory=restored_factory,
        expected_binding=_binding(),
        active_chunk_size=1,
    )
    resumed = restored.complete()

    assert resumed == uninterrupted
    assert restored_factory_calls == {unit.particle_id: 1 for unit in restored.units}
    assert restored.checkpoint_sequence == 2


def test_writer_publishes_schema22_with_exact_observation_fields(tmp_path: Path) -> None:
    """舊版 fixture writer 發布 2.2.0，且 11 個速度欄位以狀態與 None 明示保存。"""

    root = _write_partial_checkpoint(tmp_path / "schema22-writer")
    metadata = json.loads((root / "checkpoint.json").read_text(encoding="utf-8"))
    assert metadata["schema_version"] == "2.2.0"
    payload = json.loads((root / "execution_state.json").read_text(encoding="utf-8"))
    observation = payload["records"][0]["execution"]["observations"][0]
    assert set(observation) == {
        "particle_id",
        "time_utc_ns",
        "age_seconds",
        "x_m",
        "y_m",
        "z_m",
        "status",
        "environment_sample_status",
        "eta_m",
        "bed_z_m",
        "forcing_month_id",
        "environment_qc_flags",
        "velocity_sample_status",
        "total_u_mps",
        "total_v_mps",
        "total_w_mps",
        "ocm_u_mps",
        "ocm_v_mps",
        "ocm_w_mps",
        "stokes_u_mps",
        "stokes_v_mps",
        "settling_w_mps",
        "velocity_qc_flags",
    }
    assert observation["environment_sample_status"] == "not_sampled"
    assert observation["eta_m"] is None
    assert observation["bed_z_m"] is None
    assert observation["forcing_month_id"] is None
    assert observation["environment_qc_flags"] is None
    assert observation["velocity_sample_status"] == "total_only"
    assert observation["total_u_mps"] == 0.2
    assert observation["total_v_mps"] == 0.0
    assert observation["total_w_mps"] == 0.0
    assert observation["ocm_u_mps"] is None
    assert observation["ocm_v_mps"] is None
    assert observation["ocm_w_mps"] is None
    assert observation["stokes_u_mps"] is None
    assert observation["stokes_v_mps"] is None
    assert observation["settling_w_mps"] is None
    assert observation["velocity_qc_flags"] == 0


def test_schema22_context_round_trip_preserves_valid_invalid_and_not_sampled(
    tmp_path: Path,
) -> None:
    """2.2 應逐欄保存環境與速度 context，且不從位置或時間補猜資料。"""

    root = _write_context_checkpoint(tmp_path / "context-round-trip")
    loaded = load_execution_checkpoint(root, expected_binding=_binding())

    valid = loaded.executions[0].observations[0]
    assert valid.environment_sample_status is EnvironmentSampleStatus.VALID
    assert valid.eta_m == 0.0
    assert valid.bed_z_m == -20.0
    assert valid.forcing_month_id == "202401"
    assert valid.environment_qc_flags == 0
    assert valid.velocity_sample_status is VelocitySampleStatus.TOTAL_ONLY
    assert valid.total_u_mps == 0.2
    assert valid.total_v_mps == 0.0
    assert valid.total_w_mps == 0.0
    assert valid.ocm_u_mps is None
    assert valid.settling_w_mps is None

    invalid = loaded.executions[1].observations[0]
    assert invalid.environment_sample_status is EnvironmentSampleStatus.INVALID
    assert invalid.eta_m is None
    assert invalid.bed_z_m is None
    assert invalid.forcing_month_id is None
    assert invalid.environment_qc_flags == 1
    assert invalid.velocity_sample_status is VelocitySampleStatus.TOTAL_ONLY
    assert invalid.total_u_mps == 0.2
    assert invalid.total_v_mps == 0.0
    assert invalid.total_w_mps == 0.0
    assert invalid.velocity_qc_flags == 0

    not_sampled = [
        observation
        for execution in loaded.executions
        for observation in execution.observations
        if observation.environment_sample_status is EnvironmentSampleStatus.NOT_SAMPLED
    ]
    assert not_sampled
    assert all(
        (
            observation.eta_m is None
            and observation.bed_z_m is None
            and observation.forcing_month_id is None
            and observation.environment_qc_flags is None
        )
        for observation in not_sampled
    )
    assert all(
        observation.velocity_sample_status is VelocitySampleStatus.NOT_SAMPLED
        and observation.total_u_mps is None
        and observation.total_v_mps is None
        and observation.total_w_mps is None
        and observation.ocm_u_mps is None
        and observation.ocm_v_mps is None
        and observation.ocm_w_mps is None
        and observation.stokes_u_mps is None
        and observation.stokes_v_mps is None
        and observation.settling_w_mps is None
        and observation.velocity_qc_flags is None
        for observation in not_sampled
    )


def test_schema20_legacy_load_defaults_context_and_preserves_resume_state(tmp_path: Path) -> None:
    """2.0 legacy fixture 只補環境／速度 NOT_SAMPLED，identity、RNG 與 step 必須不變。"""

    root = _write_partial_checkpoint(tmp_path / "legacy-20")
    before = load_execution_checkpoint(root, expected_binding=_binding())
    _downgrade_fixture_to_20(root)

    loaded = load_execution_checkpoint(root, expected_binding=_binding())
    assert loaded.sequence == before.sequence
    assert loaded.particle_order == before.particle_order
    assert loaded.run_unit_identities == before.run_unit_identities
    assert loaded.rng_states == before.rng_states
    assert [item.step_count for item in loaded.executions] == [
        item.step_count for item in before.executions
    ]
    assert [item.minimum_clamp_count for item in loaded.executions] == [
        item.minimum_clamp_count for item in before.executions
    ]
    assert all(
        observation.environment_sample_status is EnvironmentSampleStatus.NOT_SAMPLED
        and observation.eta_m is None
        and observation.bed_z_m is None
        and observation.forcing_month_id is None
        and observation.environment_qc_flags is None
        and observation.velocity_sample_status is VelocitySampleStatus.NOT_SAMPLED
        and observation.total_u_mps is None
        and observation.total_v_mps is None
        and observation.total_w_mps is None
        and observation.ocm_u_mps is None
        and observation.ocm_v_mps is None
        and observation.ocm_w_mps is None
        and observation.stokes_u_mps is None
        and observation.stokes_v_mps is None
        and observation.settling_w_mps is None
        and observation.velocity_qc_flags is None
        for execution in loaded.executions
        for observation in execution.observations
    )

    restored = ProductionBatch.from_checkpoint(
        root,
        shard=_shard(),
        master_seed=5,
        request_factory=_factory,
        expected_binding=_binding(),
    )
    assert restored.checkpoint_sequence == before.sequence
    assert [runtime.execution.state for runtime in restored.runtimes] == [
        execution.state for execution in before.executions
    ]


def test_schema21_legacy_load_defaults_velocity_without_dataclass_pollution(
    tmp_path: Path,
) -> None:
    """2.1 legacy fixture 不含速度欄位時，loader 只補未取樣，不從最新 dataclass 猜值。"""

    root = _write_partial_checkpoint(tmp_path / "legacy-21")
    _downgrade_fixture_to_21(root)
    metadata = json.loads((root / "checkpoint.json").read_text(encoding="utf-8"))
    assert metadata["schema_version"] == "2.1.0"
    payload = json.loads((root / "execution_state.json").read_text(encoding="utf-8"))
    assert all(
        set(observation)
        == {
            "particle_id",
            "time_utc_ns",
            "age_seconds",
            "x_m",
            "y_m",
            "z_m",
            "status",
            "environment_sample_status",
            "eta_m",
            "bed_z_m",
            "forcing_month_id",
            "environment_qc_flags",
        }
        for record in payload["records"]
        for observation in record["execution"]["observations"]
    )
    loaded = load_execution_checkpoint(root, expected_binding=_binding())
    assert all(
        observation.velocity_sample_status is VelocitySampleStatus.NOT_SAMPLED
        and observation.total_u_mps is None
        and observation.total_v_mps is None
        and observation.total_w_mps is None
        and observation.ocm_u_mps is None
        and observation.ocm_v_mps is None
        and observation.ocm_w_mps is None
        and observation.stokes_u_mps is None
        and observation.stokes_v_mps is None
        and observation.settling_w_mps is None
        and observation.velocity_qc_flags is None
        for execution in loaded.executions
        for observation in execution.observations
    )


def test_schema20_rejects_new_observation_fields(tmp_path: Path) -> None:
    """metadata 宣稱 2.0 時若仍帶 2.1 欄位，不能因向前相容而靜默忽略。"""

    root = _write_partial_checkpoint(tmp_path / "legacy-with-new-fields")
    _set_schema_version(root, "2.0.0")
    with pytest.raises(ValueError, match="欄位不符"):
        load_execution_checkpoint(root, expected_binding=_binding())


@pytest.mark.parametrize("schema_version", ("1.0.0", "9.9.9", True, []))
def test_execution_loader_rejects_unknown_schema_versions(
    tmp_path: Path,
    schema_version: Any,
) -> None:
    """loader 只接受精確 2.0.0／2.1.0／2.2.0 字串，不將 bool、array 或未登錄版本當成相容。"""

    root = _write_partial_checkpoint(tmp_path / f"unknown-{len(list(tmp_path.iterdir()))}")
    _set_schema_version(root, schema_version)
    with pytest.raises(ValueError, match="schema 不支援"):
        load_execution_checkpoint(root, expected_binding=_binding())


@pytest.mark.parametrize("mutation", ("missing", "unknown"))
def test_schema21_observation_keys_are_exact(tmp_path: Path, mutation: str) -> None:
    """2.1 observation 缺欄或多欄都必須 fail closed，不能讓資料版本漂移。"""

    root = _write_partial_checkpoint(tmp_path / f"schema21-{mutation}")
    _downgrade_fixture_to_21(root)

    def tamper(payload: dict[str, Any]) -> None:
        """只改測試 fixture 的 observation key，保留 manifest checksum 可驗證。"""

        observation = payload["records"][0]["execution"]["observations"][0]
        if mutation == "missing":
            del observation["environment_sample_status"]
        else:
            observation["unexpected_context"] = None

    _rewrite_execution_payload(root, tamper)
    with pytest.raises(ValueError, match="欄位不符"):
        load_execution_checkpoint(root, expected_binding=_binding())


def test_schema21_rejects_velocity_fields_after_version_downgrade(tmp_path: Path) -> None:
    """metadata 宣稱 2.1 時若仍帶速度欄位，不能把新欄位靜默塞進舊拓撲。"""

    root = _write_partial_checkpoint(tmp_path / "schema21-with-velocity-fields")
    _set_schema_version(root, "2.1.0")
    with pytest.raises(ValueError, match="欄位不符"):
        load_execution_checkpoint(root, expected_binding=_binding())


@pytest.mark.parametrize(
    "mutation",
    (
        "unknown-environment-status",
        "bool-qc",
        "nan-eta",
        "infinite-bed",
        "malformed-month",
        "valid-nonzero-qc",
        "valid-missing-environment",
        "invalid-zero-qc",
        "not-sampled-with-values",
    ),
)
def test_schema21_rejects_invalid_environment_context(tmp_path: Path, mutation: str) -> None:
    """2.1 context 的 enum、有限值、月份、品質旗標與 status cross-field gate 必須嚴格。"""

    root = _write_partial_checkpoint(tmp_path / f"context-invalid-{mutation}")
    _downgrade_fixture_to_21(root)

    def tamper(payload: dict[str, Any]) -> None:
        """將單一合法 2.1 observation 改成一種明確非法 context。"""

        observation = payload["records"][0]["execution"]["observations"][0]
        if mutation == "unknown-environment-status":
            observation["environment_sample_status"] = "provider-guessed"
        elif mutation == "bool-qc":
            observation["environment_qc_flags"] = True
        elif mutation == "nan-eta":
            observation["eta_m"] = float("nan")
        elif mutation == "infinite-bed":
            observation["bed_z_m"] = float("inf")
        elif mutation == "malformed-month":
            observation["environment_sample_status"] = "invalid"
            observation["forcing_month_id"] = "202413"
            observation["environment_qc_flags"] = 1
        elif mutation == "valid-nonzero-qc":
            observation.update(
                {
                    "environment_sample_status": "valid",
                    "eta_m": 0.0,
                    "bed_z_m": -20.0,
                    "forcing_month_id": "202401",
                    "environment_qc_flags": 1,
                }
            )
        elif mutation == "valid-missing-environment":
            observation.update(
                {
                    "environment_sample_status": "valid",
                    "eta_m": None,
                    "bed_z_m": None,
                    "forcing_month_id": None,
                    "environment_qc_flags": 0,
                }
            )
        elif mutation == "invalid-zero-qc":
            observation.update(
                {
                    "environment_sample_status": "invalid",
                    "eta_m": None,
                    "bed_z_m": None,
                    "forcing_month_id": None,
                    "environment_qc_flags": 0,
                }
            )
        else:
            observation.update(
                {
                    "environment_sample_status": "not_sampled",
                    "eta_m": 0.0,
                    "bed_z_m": None,
                    "forcing_month_id": None,
                    "environment_qc_flags": None,
                }
            )

    _rewrite_execution_payload(root, tamper)
    with pytest.raises((TypeError, ValueError)):
        load_execution_checkpoint(root, expected_binding=_binding())


def test_schema2_rejects_corrupted_rng_observation_checksum_and_binding(tmp_path: Path) -> None:
    """RNG、observation 或 manifest 被修改時均不得載入；binding 變更也必須拒絕。"""

    batch = ProductionBatch(_shard(), master_seed=5, request_factory=_factory)
    batch.advance()
    root = _write_legacy_batch_checkpoint(batch, tmp_path / "execution", sequence=1)

    rng_path = root / "rng_states.json"
    rng_payload = json.loads(rng_path.read_text(encoding="utf-8"))
    rng_payload["records"][0]["rng_state"]["bit_generator"] = "PCG64"
    rng_path.write_text(json.dumps(rng_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum|size"):
        load_execution_checkpoint(root, expected_binding=_binding())

    clean = ProductionBatch(_shard(), master_seed=5, request_factory=_factory)
    clean.advance()
    clean_root = _write_legacy_batch_checkpoint(clean, tmp_path / "execution-clean", sequence=1)
    execution_path = clean_root / "execution_state.json"
    execution_payload = json.loads(execution_path.read_text(encoding="utf-8"))
    execution_payload["records"][0]["execution"]["observations"][0]["age_seconds"] = 99.0
    execution_path.write_text(json.dumps(execution_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum|size"):
        load_execution_checkpoint(clean_root, expected_binding=_binding())

    clean_again = ProductionBatch(_shard(), master_seed=5, request_factory=_factory)
    clean_again.advance()
    binding_root = _write_legacy_batch_checkpoint(
        clean_again,
        tmp_path / "execution-binding",
        sequence=1,
    )
    incompatible = CheckpointBinding("other", "inventory", "baseline", "shard", "pcg64dxsm-v1", "commit")
    with pytest.raises(ValueError, match="binding 不相容"):
        load_execution_checkpoint(binding_root, expected_binding=incompatible)


def test_schema2_semantic_tampering_is_rejected_after_checksum_refresh(tmp_path: Path) -> None:
    """RNG 結構、observation 身分／數值與未知 status 即使重算 checksum 仍必須拒絕。"""

    rng_root = _write_partial_checkpoint(tmp_path / "semantic-rng")
    rng_path = rng_root / "rng_states.json"
    rng_payload = json.loads(rng_path.read_text(encoding="utf-8"))
    rng_payload["records"][0]["rng_state"]["bit_generator"] = "PCG64"
    rng_path.write_text(json.dumps(rng_payload), encoding="utf-8")
    _refresh_payload_checksum(rng_root, "rng_states.json")
    with pytest.raises(ValueError, match="PCG64DXSM state"):
        load_execution_checkpoint(rng_root, expected_binding=_binding())

    nested_rng_root = _write_partial_checkpoint(tmp_path / "semantic-rng-nested")
    nested_rng_path = nested_rng_root / "rng_states.json"
    nested_rng_payload = json.loads(nested_rng_path.read_text(encoding="utf-8"))
    del nested_rng_payload["records"][0]["rng_state"]["state"]["inc"]
    nested_rng_path.write_text(json.dumps(nested_rng_payload), encoding="utf-8")
    _refresh_payload_checksum(nested_rng_root, "rng_states.json")
    with pytest.raises(ValueError, match="PCG64DXSM state|結構"):
        load_execution_checkpoint(nested_rng_root, expected_binding=_binding())

    identity_root = _write_partial_checkpoint(tmp_path / "semantic-observation-identity")
    identity_path = identity_root / "execution_state.json"
    identity_payload = json.loads(identity_path.read_text(encoding="utf-8"))
    identity_payload["records"][0]["execution"]["observations"][0]["particle_id"] = "other"
    identity_path.write_text(json.dumps(identity_payload), encoding="utf-8")
    _refresh_payload_checksum(identity_root, "execution_state.json")
    with pytest.raises(ValueError, match="particle_id 與 state 不一致"):
        load_execution_checkpoint(identity_root, expected_binding=_binding())

    numeric_root = _write_partial_checkpoint(tmp_path / "semantic-observation-number")
    numeric_path = numeric_root / "execution_state.json"
    numeric_payload = json.loads(numeric_path.read_text(encoding="utf-8"))
    numeric_payload["records"][0]["execution"]["observations"][0]["x_m"] = "not-a-number"
    numeric_path.write_text(json.dumps(numeric_payload), encoding="utf-8")
    _refresh_payload_checksum(numeric_root, "execution_state.json")
    with pytest.raises(ValueError, match="有限數值"):
        load_execution_checkpoint(numeric_root, expected_binding=_binding())

    status_root = _write_partial_checkpoint(tmp_path / "semantic-observation-status")
    status_path = status_root / "execution_state.json"
    status_payload = json.loads(status_path.read_text(encoding="utf-8"))
    status_payload["records"][0]["execution"]["observations"][0]["status"] = "unknown-status"
    status_path.write_text(json.dumps(status_payload), encoding="utf-8")
    _refresh_payload_checksum(status_root, "execution_state.json")
    with pytest.raises(ValueError, match="未知 ParticleStatus"):
        load_execution_checkpoint(status_root, expected_binding=_binding())


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    (
        (("state", "particle_id"), "", "非空字串"),
        (("state", "member_id"), True, "非負整數"),
        (("state", "time_utc_ns"), 1.25, "整數"),
        (("state", "age_seconds"), -1.0, "不可為負"),
        (("state", "own_local_exit_recorded"), 1, "boolean"),
        (("observations", 0, "time_utc_ns"), True, "整數"),
        (("observations", 0, "age_seconds"), -1.0, "不可為負"),
        (("observations", 0, "z_m"), "not-a-number", "有限數值"),
        (("observations", 0, "x_m"), float("inf"), "非有限"),
    ),
)
def test_schema2_rejects_particle_and_observation_primitive_tampering(
    tmp_path: Path,
    path: tuple[str | int, ...],
    replacement: Any,
    message: str,
) -> None:
    """粒子 state 與 observation 的 identity、整數、布林、年齡及座標型別必須嚴格驗證。"""

    root = _write_partial_checkpoint(tmp_path / ("semantic-" + "-".join(map(str, path))))
    execution_path = root / "execution_state.json"
    execution_payload = json.loads(execution_path.read_text(encoding="utf-8"))
    _set_nested_value(execution_payload["records"][0]["execution"], path, replacement)
    execution_path.write_text(json.dumps(execution_payload), encoding="utf-8")
    _refresh_payload_checksum(root, "execution_state.json")

    with pytest.raises(ValueError, match=message):
        load_execution_checkpoint(root, expected_binding=_binding())


def test_schema2_event_semantics_are_rejected_after_checksum_refresh(tmp_path: Path) -> None:
    """事件 fraction 越界與 optional numeric 欄位型別錯誤不能繞過 semantic loader。"""

    fraction_root = _write_partial_checkpoint(tmp_path / "semantic-event-fraction")
    fraction_path = fraction_root / "execution_state.json"
    fraction_payload = json.loads(fraction_path.read_text(encoding="utf-8"))
    fraction_payload["records"][0]["execution"]["events"][0]["fraction"] = 1.5
    fraction_path.write_text(json.dumps(fraction_payload), encoding="utf-8")
    _refresh_payload_checksum(fraction_root, "execution_state.json")
    with pytest.raises(ValueError, match="fraction"):
        load_execution_checkpoint(fraction_root, expected_binding=_binding())

    optional_root = _write_partial_checkpoint(tmp_path / "semantic-event-optional")
    optional_path = optional_root / "execution_state.json"
    optional_payload = json.loads(optional_path.read_text(encoding="utf-8"))
    optional_payload["records"][0]["execution"]["events"][0]["boundary_s_m"] = "invalid"
    optional_path.write_text(json.dumps(optional_payload), encoding="utf-8")
    _refresh_payload_checksum(optional_root, "execution_state.json")
    with pytest.raises(ValueError, match="有限數值"):
        load_execution_checkpoint(optional_root, expected_binding=_binding())


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("x_m", float("inf"), "非有限"),
        ("source_face_id", True, "整數"),
        ("related_study_site_id", "", "非空字串"),
    ),
)
def test_schema2_rejects_event_coordinate_and_optional_tampering(
    tmp_path: Path,
    field: str,
    replacement: Any,
    message: str,
) -> None:
    """事件座標與 optional identity 欄位即使同步更新 checksum 也不得放寬型別契約。"""

    root = _write_partial_checkpoint(tmp_path / f"semantic-event-{field}")
    execution_path = root / "execution_state.json"
    execution_payload = json.loads(execution_path.read_text(encoding="utf-8"))
    execution_payload["records"][0]["execution"]["events"][0][field] = replacement
    execution_path.write_text(json.dumps(execution_payload), encoding="utf-8")
    _refresh_payload_checksum(root, "execution_state.json")

    with pytest.raises(ValueError, match=message):
        load_execution_checkpoint(root, expected_binding=_binding())


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("age_seconds", 1.0, "不得遞減"),
        ("time_utc_ns", 2_000_000_001, "不得往未來增加"),
    ),
)
def test_schema2_rejects_observation_sequence_direction_tampering(
    tmp_path: Path,
    field: str,
    replacement: Any,
    message: str,
) -> None:
    """整條軌跡的年齡只能增加、UTC 時間只能回到更早，且不以單筆型別檢查取代序列檢查。"""

    batch = ProductionBatch(_shard(), master_seed=5, request_factory=_factory)
    batch.advance(sweeps=4)
    root = _write_legacy_batch_checkpoint(
        batch,
        tmp_path / f"semantic-sequence-{field}",
        sequence=4,
    )
    execution_path = root / "execution_state.json"
    execution_payload = json.loads(execution_path.read_text(encoding="utf-8"))
    observations = execution_payload["records"][1]["execution"]["observations"]
    assert len(observations) >= 3
    if field == "age_seconds":
        observations[2][field] = replacement
    else:
        observations[2][field] = observations[1][field] + 1
    execution_path.write_text(json.dumps(execution_payload), encoding="utf-8")
    _refresh_payload_checksum(root, "execution_state.json")

    with pytest.raises(ValueError, match=message):
        load_execution_checkpoint(root, expected_binding=_binding())


def test_schema2_rejects_duplicate_particle_order_after_checksum_refresh(tmp_path: Path) -> None:
    """重複 particle order 與 execution identity 即使 payload checksum 正確也必須拒絕。"""

    root = _write_partial_checkpoint(tmp_path / "duplicate-order")
    execution_path = root / "execution_state.json"
    execution_payload = json.loads(execution_path.read_text(encoding="utf-8"))
    execution_payload["records"][1]["identity"] = dict(execution_payload["records"][0]["identity"])
    execution_path.write_text(json.dumps(execution_payload), encoding="utf-8")
    _refresh_payload_checksum(root, "execution_state.json")
    manifest_path = root / "checkpoint.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["particle_order"][1] = manifest["particle_order"][0]
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_execution_checkpoint(root, expected_binding=_binding())


def test_schema2_rejects_unknown_subdirectory_and_symlink(tmp_path: Path) -> None:
    """schema 2.x 目錄只能含三個實體檔案，不接受未知子目錄或 symlink。"""

    unknown_root = _write_partial_checkpoint(tmp_path / "unknown-directory")
    (unknown_root / "unexpected").mkdir()
    with pytest.raises(ValueError, match="未知檔案"):
        load_execution_checkpoint(unknown_root, expected_binding=_binding())

    symlink_root = _write_partial_checkpoint(tmp_path / "symlink")
    rng_path = symlink_root / "rng_states.json"
    target_path = tmp_path / "rng-target.json"
    rng_path.replace(target_path)
    rng_path.symlink_to(target_path)
    with pytest.raises(ValueError, match="symlink"):
        load_execution_checkpoint(symlink_root, expected_binding=_binding())


def test_schema2_all_complete_batch_round_trip_preserves_results(tmp_path: Path) -> None:
    """全粒子已終止時仍可寫入 schema 2.2，restore 後結果必須逐欄相同。"""

    batch = ProductionBatch(_shard(), master_seed=17, request_factory=_factory)
    expected = batch.complete()
    assert batch.terminal
    root = _write_legacy_batch_checkpoint(
        batch,
        tmp_path / "all-complete",
        sequence=batch.sweep_count,
    )

    restored = ProductionBatch.from_checkpoint(
        root,
        shard=_shard(),
        master_seed=17,
        request_factory=_factory,
        expected_binding=_binding(),
        active_chunk_size=1,
    )
    assert restored.terminal
    assert restored.results() == expected


def test_schema2_rejects_run_unit_mismatch_and_empty_checkpoint(tmp_path: Path) -> None:
    """不同 scenario identity 與空 execution 都不能冒充可恢復 checkpoint。"""

    batch = ProductionBatch(_shard(), master_seed=8, request_factory=_factory)
    root = _write_legacy_batch_checkpoint(batch, tmp_path / "execution", sequence=0)
    with pytest.raises(ValueError, match="RunUnit identity/order"):
        load_execution_checkpoint(root, expected_binding=_binding(), expected_run_units=ProductionBatch(
            _shard("other"), master_seed=8, request_factory=_factory
        ).units)

    from lagrangian_backtracking.checkpoint import write_execution_checkpoint

    with pytest.raises(ValueError, match="不允許空"):
        write_execution_checkpoint(
            tmp_path / "empty",
            binding=_binding(),
            run_units=[],
            executions=[],
            rngs=[],
            triangle_hints=[],
            sequence=0,
        )
