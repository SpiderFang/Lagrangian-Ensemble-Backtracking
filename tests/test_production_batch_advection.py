"""ProductionBatch deterministic advection rendezvous 與生命週期回歸測試。

本檔只使用小型 synthetic provider 驗證批次介面與既有 engine 的銜接；測試資料不代表
OCM／NWW3 正式產品，也不驗證 Q2 的 BayTrace native kernel 實作。每個 batcher 都只
執行既有 ``rk4_step``，故可直接比對 legacy ProductionBatch 的 observation、event、
位置與每粒子 RNG continuation。
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
from shapely.geometry import box

from lagrangian_backtracking.advection import AdvectionTask, execute_advection_task
from lagrangian_backtracking.boundaries import BoundaryGeometry
from lagrangian_backtracking.checkpoint import CheckpointBinding
from lagrangian_backtracking.diffusion import DiffusionCoefficients
from lagrangian_backtracking.engine import EngineSettings
from lagrangian_backtracking.integrators import SamplingError
from lagrangian_backtracking.models import (
    ParticleState,
    ParticleStatus,
    SampleQC,
    VelocitySample,
    VelocitySampleStatus,
)
from lagrangian_backtracking.production import ProductionBatch, run_production_shard
from lagrangian_backtracking.runner import ReferenceParticleRequest, iter_run_units, plan_scenario_shards
from lagrangian_backtracking.scenarios import Scenario


def _shard(*, members_per_scenario: int = 3):
    """建立固定 source order 的小型 synthetic shard。"""

    scenario = Scenario(
        scenario_id="s0",
        study_site_id="gongliao",
        analysis_region_id="A",
        material_id="oca_nonrecyclable_flexible_sheet",
        receptor_id="receptor-s0",
        arrival_time_id="arrival-0",
        settling_velocity_mps=-0.001,
        arrival_time_utc_ns=100_000_000_000,
        design_version="test-v1",
    )
    return plan_scenario_shards(
        [scenario],
        members_per_scenario=members_per_scenario,
        shard_scenario_count=1,
        experiment_case_id="baseline",
    )[0]


def _settings(*, max_backtrack_seconds: float = 4.0, dt_min_seconds: float = 1.0,
              dt_max_seconds: float = 1.0) -> EngineSettings:
    """以固定步長提供可重現的短 synthetic 回溯視窗。"""

    return EngineSettings(
        dt_min_seconds=dt_min_seconds,
        dt_max_seconds=dt_max_seconds,
        output_interval_seconds=2.0,
        max_backtrack_seconds=max_backtrack_seconds,
        maximum_step_count=100,
        earliest_forcing_time_utc_ns=0,
    )


def _boundaries() -> BoundaryGeometry:
    """建立不會在測試視窗內提前終止的寬廣邊界。"""

    return BoundaryGeometry(
        own_local_domain=box(-1_000.0, -1_000.0, 1_000.0, 1_000.0),
        flow_domain=box(-2_000.0, -2_000.0, 2_000.0, 2_000.0),
        foreign_local_domains={},
    )


def _request(unit, *, settings: EngineSettings | None = None) -> ReferenceParticleRequest:
    """建立與 RunUnit identity 對齊、含非零擴散的 request。"""

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

    def velocity(x_m: float, y_m: float, z_m: float, time_utc_ns: int) -> VelocitySample:
        """回傳位置依賴但固定有限的總速度，方便比對 deterministic stage。"""

        del time_utc_ns
        return VelocitySample(
            0.2 + 0.01 * x_m,
            -0.1 + 0.01 * y_m,
            0.005 + 0.0001 * z_m,
            0.0,
            -100.0,
            100.0,
            10.0,
        )

    return ReferenceParticleRequest(
        initial_state=state,
        velocity=velocity,
        boundaries=_boundaries(),
        behavior_class="sinking",
        diffusion=DiffusionCoefficients(0.2, 0.1, 0.03),
        settings=_settings() if settings is None else settings,
    )


class ExecuteSpy:
    """以既有 deterministic executor 實作測試用 batcher，並記錄批次拓撲。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.tasks: list[tuple[AdvectionTask, ...]] = []

    def __call__(self, tasks: tuple[AdvectionTask, ...]):
        """按輸入順序執行 RK4，不處理 diffusion 或 RNG。"""

        task_tuple = tuple(tasks)
        self.tasks.append(task_tuple)
        self.calls.append(tuple(task.state.particle_id for task in task_tuple))
        return [execute_advection_task(task) for task in task_tuple]


def test_batcher_receives_multiple_particles_once_per_sweep_and_matches_legacy() -> None:
    """N 粒子同一 sweep 應合併成一次 batch，且完整結果仍等於 legacy path。"""

    shard = _shard(members_per_scenario=3)
    legacy = run_production_shard(shard, master_seed=20260919, request_factory=_request)
    spy = ExecuteSpy()
    batched = run_production_shard(
        shard,
        master_seed=20260919,
        request_factory=_request,
        advection_batcher=spy,
    )

    assert batched == legacy
    assert spy.calls[0] == tuple(
        unit.particle_id for unit in iter_run_units(shard, master_seed=20260919)
    )
    assert len(spy.calls) == 4
    assert all(len(call) == 3 for call in spy.calls)


def test_nonzero_diffusion_checkpoint_resume_keeps_rng_and_output_across_chunk_sizes(
    tmp_path: Path,
) -> None:
    """非零擴散在不中斷、checkpoint resume 與不同 chunk 下逐欄相同。"""

    shard = _shard(members_per_scenario=3)
    binding = CheckpointBinding("config", "inventory", "baseline", "shard", "pcg64dxsm-v1", "commit")
    uninterrupted_spy = ExecuteSpy()
    uninterrupted = ProductionBatch(
        shard,
        master_seed=77,
        request_factory=_request,
        active_chunk_size=3,
        advection_batcher=uninterrupted_spy,
    ).complete()

    first_spy = ExecuteSpy()
    interrupted = ProductionBatch(
        shard,
        master_seed=77,
        request_factory=_request,
        active_chunk_size=1,
        advection_batcher=first_spy,
    )
    interrupted.advance(sweeps=2)
    checkpoint = interrupted.write_checkpoint(
        tmp_path / "checkpoint-00000001",
        binding=binding,
        sequence=1,
    )

    resume_spy = ExecuteSpy()
    restored = ProductionBatch.from_checkpoint(
        checkpoint,
        shard=shard,
        master_seed=77,
        request_factory=_request,
        expected_binding=binding,
        active_chunk_size=2,
        advection_batcher=resume_spy,
    )
    resumed = restored.complete()

    assert resumed == uninterrupted
    assert restored.advection_batcher is resume_spy
    assert first_spy.tasks and resume_spy.tasks


def test_short_terminal_step_and_invalid_start_keep_batch_lifecycle_and_rng_contract() -> None:
    """短終點只走一次 deterministic task；invalid start 不送 batch 且保留速度 observation。"""

    shard = _shard(members_per_scenario=1)
    short_settings = _settings(max_backtrack_seconds=0.25, dt_min_seconds=1.0, dt_max_seconds=2.0)
    short_spy = ExecuteSpy()
    short_batch = ProductionBatch(
        shard,
        master_seed=11,
        request_factory=lambda unit: _request(unit, settings=short_settings),
        advection_batcher=short_spy,
    )
    before = deepcopy(short_batch.runtimes[0].rng.bit_generator.state)
    short_batch.complete()
    after = short_batch.runtimes[0].rng.bit_generator.state
    expected = np.random.Generator(np.random.PCG64DXSM(short_batch.units[0].seed))
    expected.normal(size=3)

    assert len(short_spy.tasks) == 1
    assert short_spy.tasks[0][0].dt_seconds == -0.25
    assert before != after
    assert after == expected.bit_generator.state
    assert short_batch.runtimes[0].execution.state.status is ParticleStatus.MAX_AGE

    def invalid_request(unit) -> ReferenceParticleRequest:
        """在步首回傳 TIME_GAP，驗證不會進入 RK4 batch。"""

        request = _request(unit)

        def invalid_velocity(x_m: float, y_m: float, z_m: float, time_utc_ns: int) -> VelocitySample:
            """保留有限 context 與明示資料缺口，不能以零速度繼續。"""

            del x_m, y_m, z_m, time_utc_ns
            return VelocitySample(
                0.2,
                -0.1,
                0.0,
                0.0,
                -100.0,
                100.0,
                10.0,
                qc=SampleQC.TIME_GAP,
            )

        return ReferenceParticleRequest(
            initial_state=request.initial_state,
            velocity=invalid_velocity,
            boundaries=request.boundaries,
            behavior_class=request.behavior_class,
            diffusion=request.diffusion,
            settings=request.settings,
        )

    invalid_spy = ExecuteSpy()
    invalid_batch = ProductionBatch(
        shard,
        master_seed=11,
        request_factory=invalid_request,
        advection_batcher=invalid_spy,
    )
    invalid_result = invalid_batch.complete()[0]

    assert invalid_spy.calls == []
    assert invalid_result.final_state.status is ParticleStatus.DATA_GAP
    assert invalid_result.observations[-1].velocity_sample_status is VelocitySampleStatus.INVALID
    # Observation 的 velocity QC 描述「速度樣本本身無效」；原始 SampleQC.TIME_GAP
    # 則另由 terminal event 的 failure attributes 保存，兩者不能混在同一欄。
    assert invalid_result.observations[-1].velocity_qc_flags == 1


def test_surface_retry_errors_are_returned_in_source_order_and_do_not_consume_extra_rng() -> None:
    """多粒子 stage retry 以 source order 重送，折半錯誤本身不會多抽 Brownian。"""

    shard = _shard(members_per_scenario=2)
    settings = _settings(max_backtrack_seconds=2.0, dt_min_seconds=1.0, dt_max_seconds=2.0)
    def factory(unit) -> ReferenceParticleRequest:
        """讓兩個 member 都在相同 k4 海面越界情境觸發一次折半。"""

        base = _request(unit, settings=settings)

        def velocity(x_m: float, y_m: float, z_m: float, time_utc_ns: int) -> VelocitySample:
            """海面上方只回傳精確 VERTICAL_UNSUPPORTED，保留 retry 證據。"""

            del x_m
            if z_m > 0.0:
                return VelocitySample(
                    0.0,
                    0.0,
                    -0.4,
                    0.0,
                    -100.0,
                    100.0,
                    100.0,
                    SampleQC.VERTICAL_UNSUPPORTED,
                )
            return VelocitySample(0.0, 0.0, -0.4, 0.0, -100.0, 100.0, 100.0)

        initial = ParticleState(
            particle_id=unit.particle_id,
            scenario_id=unit.scenario.scenario_id,
            member_id=unit.member_id,
            study_site_id=unit.scenario.study_site_id,
            analysis_region_id=unit.scenario.analysis_region_id,
            receptor_id=unit.scenario.receptor_id,
            x_m=0.0,
            y_m=0.0,
            z_m=-0.5,
            time_utc_ns=unit.scenario.arrival_time_utc_ns,
        )
        return ReferenceParticleRequest(
            initial_state=initial,
            velocity=velocity,
            boundaries=base.boundaries,
            behavior_class="sinking",
            diffusion=DiffusionCoefficients(0.0, 0.0, 0.0),
            settings=settings,
        )

    class RetrySpy(ExecuteSpy):
        """把 RK4 的 SamplingError 當成 batch output 交回 generator。"""

        def __call__(self, tasks: tuple[AdvectionTask, ...]):
            task_tuple = tuple(tasks)
            self.tasks.append(task_tuple)
            self.calls.append(tuple(task.state.particle_id for task in task_tuple))
            outputs: list[ParticleState | SamplingError] = []
            for task in task_tuple:
                try:
                    outputs.append(execute_advection_task(task))
                except SamplingError as error:
                    outputs.append(error)
            return outputs

    spy = RetrySpy()
    batch = ProductionBatch(
        shard,
        master_seed=99,
        request_factory=factory,
        advection_batcher=spy,
    )
    batch.advance()

    source_order = tuple(unit.particle_id for unit in batch.units)
    assert spy.calls[:2] == [source_order, source_order]
    assert [task.dt_seconds for task in spy.tasks[0]] == [-2.0, -2.0]
    assert [task.dt_seconds for task in spy.tasks[1]] == [-1.0, -1.0]
    assert all(runtime.execution.step_count == 1 for runtime in batch.runtimes)
    assert all(len(runtime.execution.observations) == 1 for runtime in batch.runtimes)
    assert all(runtime.execution.observations[-1].velocity_sample_status is VelocitySampleStatus.TOTAL_ONLY
               for runtime in batch.runtimes)
    for runtime in batch.runtimes:
        expected_rng = np.random.Generator(np.random.PCG64DXSM(runtime.unit.seed))
        expected_rng.normal(size=3)
        assert runtime.rng.bit_generator.state == expected_rng.bit_generator.state
