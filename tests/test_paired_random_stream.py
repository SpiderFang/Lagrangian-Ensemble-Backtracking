"""共同亂數流（common random number）配對 seed 的 synthetic engineering contract 測試。

本檔只使用本機合成 scenario、固定速度與 temporary workspace；測試的是 seed 導出、
immutable run plan、seed table、checkpoint binding 與 restore 的工程契約，不代表真實
OCM／NWW3 forcing 或任何科學結果。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from shapely.geometry import box

from lagrangian_backtracking.boundaries import BoundaryGeometry
from lagrangian_backtracking.checkpoint import CheckpointBinding, load_execution_checkpoint
from lagrangian_backtracking.diffusion import DiffusionCoefficients
from lagrangian_backtracking.engine import EngineSettings
from lagrangian_backtracking.models import ParticleState, VelocitySample
from lagrangian_backtracking.production import ProductionBatch, ReferenceParticleRequest
from lagrangian_backtracking.provenance import CodeProvenance
from lagrangian_backtracking.run_control import (
    RUN_PLAN_PAIRED_SCHEMA_VERSION,
    initialize_run_workspace,
    load_run_plan,
)
from lagrangian_backtracking.run_validation import validate_run
from lagrangian_backtracking.runner import iter_run_units, plan_scenario_shards
from lagrangian_backtracking.scenarios import Scenario, stable_identifier


def _provenance() -> CodeProvenance:
    """建立可通過 run plan contract 的 synthetic Git provenance。"""

    return CodeProvenance(
        git_available=True,
        git_commit="0123456789abcdef0123456789abcdef01234567",
        git_dirty=False,
        commit_source="git_repository",
        deployment_tree_sha256="1" * 64,
        deployment_file_count=3,
        uv_lock_sha256="2" * 64,
        python_version="3.12.0",
        platform="test-platform",
        package_version="0.1.0",
        numpy_version=np.__version__,
        numba_version="test",
        pyarrow_version="test",
    )


def _scenario() -> Scenario:
    """建立兩個物理案例可以共用的固定 synthetic scenario。"""

    return Scenario(
        scenario_id=stable_identifier(
            "scn", ["paired-site", "material", "receptor", "arrival", "paired-v1"]
        ),
        study_site_id="paired-site",
        analysis_region_id="paired-region",
        material_id="material",
        receptor_id="receptor",
        arrival_time_id="arrival",
        settling_velocity_mps=-0.001,
        arrival_time_utc_ns=1_700_000_000_000_000_000,
        design_version="paired-v1",
    )


def _shard(experiment_case_id: str = "no_stokes"):
    """依物理案例建立同一 scenario 的小型兩-member shard。"""

    return plan_scenario_shards(
        [_scenario()],
        members_per_scenario=2,
        shard_scenario_count=1,
        experiment_case_id=experiment_case_id,
    )[0]


def _request(unit: object) -> ReferenceParticleRequest:
    """建立不消耗外部資料的固定 request，供 ProductionBatch restore contract 使用。"""

    # RunUnit 是公開的 dataclass，但此測試只需其固定欄位；避免以 dict 或手寫 particle
    # identity 取代正式 factory，讓 restore 時仍會經過 batch 的 identity gate。
    scenario = unit.scenario  # type: ignore[attr-defined]
    state = ParticleState(
        particle_id=unit.particle_id,  # type: ignore[attr-defined]
        scenario_id=scenario.scenario_id,
        member_id=unit.member_id,  # type: ignore[attr-defined]
        study_site_id=scenario.study_site_id,
        analysis_region_id=scenario.analysis_region_id,
        receptor_id=scenario.receptor_id,
        x_m=0.0,
        y_m=0.0,
        z_m=-5.0,
        time_utc_ns=scenario.arrival_time_utc_ns,
    )

    def velocity(
        x_m: float,
        y_m: float,
        z_m: float,
        time_utc_ns: int,
    ) -> VelocitySample:
        """回傳有限常流樣本；位置與 UTC 僅用於符合 provider 介面。"""

        del x_m, y_m, z_m, time_utc_ns
        return VelocitySample(0.0, 0.0, 0.0, 0.0, -10.0, 100.0, 10.0)

    return ReferenceParticleRequest(
        initial_state=state,
        velocity=velocity,
        boundaries=BoundaryGeometry(
            own_local_domain=box(-5.0, -5.0, 5.0, 5.0),
            flow_domain=box(-20.0, -20.0, 20.0, 20.0),
            foreign_local_domains={},
        ),
        behavior_class="sinking",
        diffusion=DiffusionCoefficients(0.0, 0.0, 0.0),
        settings=EngineSettings(1.0, 1.0, 1.0, 3.0, 20, 0),
    )


def _initialize_pair_workspace(parent: Path, run_id: str) -> Path:
    """建立含 explicit paired stream 的最小 immutable workspace。"""

    workspace = initialize_run_workspace(
        parent,
        run_id=run_id,
        scenarios=(_scenario(),),
        normalized_config={"schema_version": "test", "settings": {"dt": 1.0}},
        config_hash="9" * 64,
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
        provenance=_provenance(),
        experiment_case_id="finite_depth_stokes",
        master_seed=20260914,
        seed_policy="sha256_v1_pcg64dxsm",
        random_stream_id="abcd-common-rng-v1",
        members_per_scenario=2,
        shard_scenario_count=1,
        checkpoint_interval_sweeps=1,
        active_chunk_size=1,
        run_kind="synthetic",
    )
    return workspace.path


def test_omitted_stream_preserves_case_specific_seed_and_explicit_stream_pairs_cases() -> None:
    """省略 stream 時兩案例 seed 不同；明示相同 stream 時只有 seed 相同。"""

    no_stokes = tuple(iter_run_units(_shard("no_stokes"), master_seed=17))
    finite_depth = tuple(iter_run_units(_shard("finite_depth_stokes"), master_seed=17))
    assert [unit.seed for unit in no_stokes] != [unit.seed for unit in finite_depth]
    assert [unit.particle_id for unit in no_stokes] != [unit.particle_id for unit in finite_depth]

    paired_no_stokes = tuple(
        iter_run_units(_shard("no_stokes"), master_seed=17, random_stream_id="pair-v1")
    )
    paired_finite_depth = tuple(
        iter_run_units(_shard("finite_depth_stokes"), master_seed=17, random_stream_id="pair-v1")
    )
    assert [unit.seed for unit in paired_no_stokes] == [unit.seed for unit in paired_finite_depth]
    assert [unit.particle_id for unit in paired_no_stokes] != [
        unit.particle_id for unit in paired_finite_depth
    ]


def test_paired_plan_seed_table_validator_and_production_rng_share_stream(tmp_path: Path) -> None:
    """paired plan、seed table、validator 與 ProductionBatch 的實際 RNG seed 必須一致。"""

    workspace = _initialize_pair_workspace(tmp_path, "paired-plan")
    plan = load_run_plan(workspace)
    assert plan["schema_version"] == RUN_PLAN_PAIRED_SCHEMA_VERSION
    assert plan["random_stream_id"] == "abcd-common-rng-v1"

    seed_table = pq.read_table(workspace / "seed_table.parquet")
    assert tuple(seed_table.column_names) == (
        "scenario_id",
        "experiment_case_id",
        "member_id",
        "particle_id",
        "seed_128_hex",
        "random_stream_id",
    )
    rows = seed_table.to_pylist()
    shard = _shard("finite_depth_stokes")
    batch = ProductionBatch(
        shard,
        master_seed=int(plan["master_seed"]),
        random_stream_id=plan["random_stream_id"],
        request_factory=_request,
    )
    paired_no_stokes_batch = ProductionBatch(
        _shard("no_stokes"),
        master_seed=int(plan["master_seed"]),
        random_stream_id=plan["random_stream_id"],
        request_factory=_request,
    )
    assert [row["seed_128_hex"] for row in rows] == [f"{unit.seed:032x}" for unit in batch.units]
    assert [unit.seed for unit in batch.units] == [
        unit.seed for unit in paired_no_stokes_batch.units
    ]
    assert all(row["random_stream_id"] == plan["random_stream_id"] for row in rows)
    for runtime, unit in zip(batch.runtimes, batch.units, strict=True):
        expected_rng = np.random.Generator(np.random.PCG64DXSM(unit.seed))
        assert runtime.rng.bit_generator.state == expected_rng.bit_generator.state

    validation = validate_run(workspace, require_complete=False)
    assert validation["valid"] is True, validation["errors"]


def test_paired_checkpoint_restore_binds_stream_and_rejects_change(tmp_path: Path) -> None:
    """checkpoint restore 同時保留 stream；改變 stream 必須在恢復前 fail-closed。"""

    shard = _shard("finite_depth_stokes")
    binding = CheckpointBinding(
        "config",
        "inventory",
        "finite_depth_stokes",
        shard.shard_id,
        "sha256_v1_pcg64dxsm",
        "0123456789abcdef0123456789abcdef01234567",
        "pair-v1",
    )
    interrupted = ProductionBatch(
        shard,
        master_seed=19,
        random_stream_id="pair-v1",
        request_factory=_request,
    )
    interrupted.advance()
    checkpoint = interrupted.write_checkpoint(
        tmp_path / "checkpoint-00000001", binding=binding, sequence=1
    )
    metadata = json.loads((checkpoint / "checkpoint.json").read_text(encoding="utf-8"))
    assert metadata["binding"]["random_stream_id"] == "pair-v1"
    loaded = load_execution_checkpoint(
        checkpoint,
        expected_binding=binding,
        expected_run_units=interrupted.units,
    )
    assert loaded.binding == binding

    restored = ProductionBatch.from_checkpoint(
        checkpoint,
        shard=shard,
        master_seed=19,
        random_stream_id="pair-v1",
        request_factory=_request,
        expected_binding=binding,
    )
    assert [unit.seed for unit in restored.units] == [unit.seed for unit in interrupted.units]
    with pytest.raises(ValueError, match="random_stream_id"):
        ProductionBatch.from_checkpoint(
            checkpoint,
            shard=shard,
            master_seed=19,
            random_stream_id="pair-v2",
            request_factory=_request,
            expected_binding=binding,
        )


def test_paired_plan_stream_drift_is_rejected_by_seed_table_validator(tmp_path: Path) -> None:
    """直接竄改 immutable plan 的 stream 後，validator 不允許 seed table 漂移。"""

    workspace = _initialize_pair_workspace(tmp_path, "paired-drift")
    plan_path = workspace / "run_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["random_stream_id"] = "pair-v2"
    # 測試只在 temporary workspace 模擬事故現場；正式 writer 永不覆寫已發布 plan。
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    validation = validate_run(workspace, require_complete=False)
    assert validation["valid"] is False
    assert any("seed_table.parquet" in error for error in validation["errors"])


@pytest.mark.parametrize("invalid", [None, "", "   ", 1, True])
def test_random_stream_id_is_nonempty_when_explicit(invalid: object) -> None:
    """顯式 random_stream_id 只接受非空白原生字串。"""

    if invalid is None:
        # None 代表省略，在一般 seed API 是合法的；此測試只直接驗證非法顯式值的共用
        # validator，故讓 None 維持其 backward-compatible omission 語意。
        return
    with pytest.raises((TypeError, ValueError)):
        tuple(
            iter_run_units(
                _shard("finite_depth_stokes"),
                master_seed=1,
                random_stream_id=invalid,  # type: ignore[arg-type]
            )
        )
