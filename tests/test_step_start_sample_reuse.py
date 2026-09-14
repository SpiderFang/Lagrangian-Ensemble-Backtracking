"""步首速度樣本重用的等價、邊界與重啟契約測試。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np
from shapely.geometry import box

from lagrangian_backtracking.boundaries import BoundaryGeometry
from lagrangian_backtracking.checkpoint import (
    CheckpointBinding,
    load_execution_checkpoint,
    write_execution_checkpoint,
)
from lagrangian_backtracking.diffusion import DiffusionCoefficients
from lagrangian_backtracking.engine import (
    EngineSettings,
    advance_particle_once,
    finalize_particle_execution,
    initialize_particle_execution,
    run_particle,
)
from lagrangian_backtracking.integrators import (
    SurfaceStageVelocityProvider,
    rk4_step,
    split_rk4_brownian_step,
    supports_step_start_sample_reuse,
)
from lagrangian_backtracking.models import (
    ParticleState,
    ParticleStatus,
    SampleQC,
    VelocitySample,
)
from lagrangian_backtracking.production import run_production_shard
from lagrangian_backtracking.runner import (
    ReferenceParticleRequest,
    iter_run_units,
    plan_scenario_shards,
    run_reference_shard,
)
from lagrangian_backtracking.scenarios import Scenario


def _state(*, time_utc_ns: int = 100_000_000_000) -> ParticleState:
    """建立遠離水平邊界、可支援短步測試的 active 粒子。"""

    return ParticleState(
        particle_id="p0",
        scenario_id="s0",
        member_id=0,
        study_site_id="gongliao",
        analysis_region_id="A",
        receptor_id="r0",
        x_m=0.0,
        y_m=0.0,
        z_m=-10.0,
        time_utc_ns=time_utc_ns,
    )


def _settings(**overrides: object) -> EngineSettings:
    """提供固定一秒步長與短回溯窗，讓取樣次數與重啟結果可直接核對。"""

    values: dict[str, object] = {
        "dt_min_seconds": 1.0,
        "dt_max_seconds": 1.0,
        "output_interval_seconds": 2.0,
        "max_backtrack_seconds": 2.0,
        "maximum_step_count": 100,
        "earliest_forcing_time_utc_ns": 0,
    }
    values.update(overrides)
    return EngineSettings(**values)


def _boundaries() -> BoundaryGeometry:
    """建立不會在短測試中提前觸發水平事件的寬廣公尺制邊界。"""

    return BoundaryGeometry(
        own_local_domain=box(-100.0, -100.0, 100.0, 100.0),
        flow_domain=box(-1_000.0, -1_000.0, 1_000.0, 1_000.0),
        foreign_local_domains={},
    )


class StableVelocitySampler:
    """具明示重用能力的唯讀合成 provider，並記錄每次物理查詢。

    這個測試 provider 模擬正式 forcing facade 的關鍵性質：相同公尺位置與 UTC 時刻
    回傳固定速度，查詢計數只用來確認效能路徑，不能成為速度值或品質旗標的來源。
    """

    step_start_sample_reuse_safe = True

    def __init__(self, *, velocity: tuple[float, float, float] = (1.0, -0.5, 0.0)) -> None:
        """建立固定速度場；``velocity`` 單位為公尺／秒。"""

        self.velocity = velocity
        self.calls: list[tuple[float, float, float, int]] = []

    def __call__(self, x_m: float, y_m: float, z_m: float, time_utc_ns: int) -> VelocitySample:
        """記錄公尺制查詢並回傳有限、完整的水柱樣本。"""

        self.calls.append((x_m, y_m, z_m, time_utc_ns))
        return VelocitySample(
            *self.velocity,
            0.0,
            -100.0,
            100.0,
            10.0,
        )


class StatefulVelocitySampler:
    """沒有重用標記的有狀態 provider，保留每次呼叫都可觀察的既有語意。"""

    def __init__(self) -> None:
        """建立依查詢次數改變速度的測試 provider。"""

        self.calls = 0

    def __call__(self, x_m: float, y_m: float, z_m: float, time_utc_ns: int) -> VelocitySample:
        """以查詢次數加入可辨識的東向速度變化。"""

        del x_m, y_m, z_m, time_utc_ns
        self.calls += 1
        return VelocitySample(
            float(self.calls),
            0.0,
            0.0,
            0.0,
            -100.0,
            100.0,
            10.0,
        )


def test_explicit_step_start_sample_reuse_reduces_only_k1_query() -> None:
    """明示穩定樣本時 RK4 仍有四個導數，但 RK4 內部只查詢 k2--k4 三次。"""

    sampler = StableVelocitySampler()
    step_start = sampler(_state().x_m, _state().y_m, _state().z_m, _state().time_utc_ns)
    calls_before_rk4 = len(sampler.calls)
    result = rk4_step(
        _state(),
        dt_seconds=-1.0,
        velocity=sampler,
        step_start_sample=step_start,
    )

    assert result.x_m == -1.0
    assert result.y_m == 0.5
    assert len(sampler.calls) - calls_before_rk4 == 3  # k1 沿用步首樣本，只查 k2、k3、k4。


def test_engine_keeps_stateful_callable_query_order_and_uses_opt_in_only() -> None:
    """engine 只對標記 provider 省略 k1；普通有狀態 callable 仍完整查詢五次。"""

    stable = StableVelocitySampler()
    stable_execution = initialize_particle_execution(_state(), _settings())
    stable_result = advance_particle_once(
        stable_execution,
        velocity=stable,
        boundaries=_boundaries(),
        behavior_class="sinking",
        diffusion=DiffusionCoefficients(0.0, 0.0, 0.0),
        settings=_settings(),
        rng=np.random.default_rng(1),
    )

    stateful = StatefulVelocitySampler()
    stateful_execution = initialize_particle_execution(_state(), _settings())
    stateful_result = advance_particle_once(
        stateful_execution,
        velocity=stateful,
        boundaries=_boundaries(),
        behavior_class="sinking",
        diffusion=DiffusionCoefficients(0.0, 0.0, 0.0),
        settings=_settings(),
        rng=np.random.default_rng(1),
    )

    assert supports_step_start_sample_reuse(stable)
    assert not supports_step_start_sample_reuse(stateful)
    assert stable_result.stepped and stateful_result.stepped
    assert len(stable.calls) == 4
    assert stateful.calls == 5


def test_surface_reflection_wrapper_retains_k1_and_stage_order() -> None:
    """特殊海面反射 wrapper 不取得重用標記，故 k1 與每次中間重查順序保持完整。"""

    state = replace(_state(), z_m=-1.0)
    calls: list[tuple[float, float, float, int]] = []

    def surface_limited_velocity(
        x_m: float, y_m: float, z_m: float, time_utc_ns: int
    ) -> VelocitySample:
        """海面以上的查詢回傳垂向不支援，水柱內則回傳固定下沉速度。"""

        calls.append((x_m, y_m, z_m, time_utc_ns))
        if z_m > 0.0:
            return VelocitySample(
                0.0,
                0.0,
                -1.0,
                0.0,
                -10.0,
                100.0,
                10.0,
                SampleQC.VERTICAL_UNSUPPORTED,
            )
        return VelocitySample(0.0, 0.0, -1.0, 0.0, -10.0, 100.0, 10.0)

    step_start = surface_limited_velocity(0.0, 0.0, -1.0, state.time_utc_ns)
    wrapper = SurfaceStageVelocityProvider(
        velocity=surface_limited_velocity,
        step_start_state=state,
        step_start_sample=step_start,
        behavior_class="sinking",
    )
    assert not supports_step_start_sample_reuse(wrapper)
    result = split_rk4_brownian_step(
        state,
        dt_seconds=-4.0,
        velocity=wrapper,
        coefficients=DiffusionCoefficients(0.0, 0.0, 0.0),
        rng=np.random.default_rng(2),
    )

    assert result.z_m == 3.0
    # 先由測試建立 step_start，再由 wrapper 進行 k1、k2、k2 反射、k3、k3 反射、
    # k4、k4 反射；反射重查不推進 wrapper 的 stage counter。
    assert [round(query[2], 12) for query in calls] == [
        -1.0,
        -1.0,
        1.0,
        -1.0,
        1.0,
        -1.0,
        3.0,
        -3.0,
    ]


def test_surface_retry_requeries_k1_after_first_reused_attempt() -> None:
    """安全重用只作用於第一次嘗試；折半重試仍重新查詢 k1 以保留 hint 順序。"""

    class RetrySurfaceSampler:
        """宣告唯讀結果、但以計數器記錄 adaptive retry 的每個查詢。"""

        step_start_sample_reuse_safe = True

        def __init__(self) -> None:
            """建立查詢紀錄。"""

            self.calls: list[tuple[float, int]] = []

        def __call__(self, x_m: float, y_m: float, z_m: float, time_utc_ns: int) -> VelocitySample:
            """只拒絕超過海面容許帶的 stage，讓 engine 依既有規則折半及鏡射。"""

            del x_m
            self.calls.append((z_m, time_utc_ns))
            tolerance = 5.0e-6
            if z_m > tolerance:
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
            vertical_velocity = -0.4 if z_m >= 0.0 else 1.0
            return VelocitySample(0.0, 0.0, vertical_velocity, 0.0, -100.0, 100.0, 100.0)

    sampler = RetrySurfaceSampler()
    settings = _settings(
        dt_min_seconds=2.0,
        dt_max_seconds=4.0,
        max_backtrack_seconds=4.0,
        output_interval_seconds=2.0,
    )
    execution = initialize_particle_execution(replace(_state(), z_m=2.5e-6), settings)
    outcome = advance_particle_once(
        execution,
        velocity=sampler,
        boundaries=_boundaries(),
        behavior_class="sinking",
        diffusion=DiffusionCoefficients(0.0, 0.0, 0.0),
        settings=settings,
        rng=np.random.default_rng(4),
    )

    assert outcome.stepped and not outcome.terminal
    assert execution.state.z_m < 0.0
    assert supports_step_start_sample_reuse(sampler)
    # 初次步首 sample 供 k1 重用；4 s、2 s 折半嘗試各重新取 k1，最後才進入完整反射
    # wrapper。若 retry 錯誤沿用 reference，總查詢會少一次，無法保持原 hint 呼叫順序。
    assert len(sampler.calls) == 9
    assert sampler.calls[0][0] == 2.5e-6
    assert sampler.calls[2][0] == 2.5e-6


def test_invalid_step_start_sample_is_not_reused_or_rng_consumed() -> None:
    """即使 provider 宣告能力，無效步首樣本仍在 engine 先終止且只查詢一次。"""

    sampler = StableVelocitySampler()

    def invalid_velocity(x_m: float, y_m: float, z_m: float, time_utc_ns: int) -> VelocitySample:
        """回傳帶有時間缺口的無效 forcing 樣本。"""

        sampler.calls.append((x_m, y_m, z_m, time_utc_ns))
        return VelocitySample(
            0.0,
            0.0,
            0.0,
            np.nan,
            -100.0,
            100.0,
            10.0,
            SampleQC.TIME_GAP,
        )

    invalid_velocity.step_start_sample_reuse_safe = True  # type: ignore[attr-defined]
    execution = initialize_particle_execution(_state(), _settings())
    rng = np.random.default_rng(3)
    before = deepcopy(rng.bit_generator.state)
    result = advance_particle_once(
        execution,
        velocity=invalid_velocity,
        boundaries=_boundaries(),
        behavior_class="sinking",
        diffusion=DiffusionCoefficients(0.0, 0.0, 0.0),
        settings=_settings(),
        rng=rng,
    )

    assert result.terminal and not result.stepped
    assert execution.state.status == ParticleStatus.DATA_GAP
    assert len(sampler.calls) == 1
    assert rng.bit_generator.state == before


def test_marked_reference_and_production_match_with_short_checkpoint_restart(tmp_path) -> None:
    """標記 provider 的 reference、production 與 checkpoint restore 仍逐欄一致。"""

    scenario = Scenario(
        "s0",
        "gongliao",
        "A",
        "material",
        "r0",
        "arrival",
        -0.001,
        100_000_000_000,
        "test",
    )
    shard = plan_scenario_shards(
        [scenario],
        members_per_scenario=1,
        shard_scenario_count=1,
        experiment_case_id="baseline",
    )[0]
    sampler_instances: list[StableVelocitySampler] = []

    def request_factory(unit) -> ReferenceParticleRequest:
        """建立與 RunUnit 身分一致且可重建的唯讀合成 request。"""

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
        sampler = StableVelocitySampler()
        sampler_instances.append(sampler)
        return ReferenceParticleRequest(
            initial_state=state,
            velocity=sampler,
            boundaries=_boundaries(),
            behavior_class="sinking",
            diffusion=DiffusionCoefficients(0.2, 0.1, 0.01),
            settings=_settings(max_backtrack_seconds=3.0),
        )

    reference = run_reference_shard(
        shard,
        master_seed=19,
        request_factory=request_factory,
    )
    production = run_production_shard(shard, master_seed=19, request_factory=request_factory)
    assert production == reference
    assert sampler_instances and all(len(sampler.calls) == 12 for sampler in sampler_instances)

    checkpoint_binding = CheckpointBinding(
        "config",
        "inventory",
        "baseline",
        "shard",
        "pcg64dxsm-v1",
        "commit",
    )
    run_unit = next(iter(iter_run_units(shard, master_seed=19)))
    initial_state = replace(
        _state(),
        particle_id=run_unit.particle_id,
        scenario_id=run_unit.scenario.scenario_id,
        member_id=run_unit.member_id,
        study_site_id=run_unit.scenario.study_site_id,
        analysis_region_id=run_unit.scenario.analysis_region_id,
        receptor_id=run_unit.scenario.receptor_id,
    )
    sampler = StableVelocitySampler()
    execution = initialize_particle_execution(
        initial_state,
        _settings(max_backtrack_seconds=3.0),
    )
    rng = np.random.Generator(np.random.PCG64DXSM(run_unit.seed))
    first = advance_particle_once(
        execution,
        velocity=sampler,
        boundaries=_boundaries(),
        behavior_class="sinking",
        diffusion=DiffusionCoefficients(0.2, 0.1, 0.01),
        settings=_settings(max_backtrack_seconds=3.0),
        rng=rng,
    )
    assert first.stepped and len(sampler.calls) == 4
    run_units = list(iter_run_units(shard, master_seed=19))
    checkpoint_path = write_execution_checkpoint(
        tmp_path / "checkpoint-00000001",
        binding=checkpoint_binding,
        run_units=run_units,
        executions=[execution],
        rngs=[rng],
        triangle_hints=[None],
        sequence=1,
    )
    loaded = load_execution_checkpoint(
        checkpoint_path,
        expected_binding=checkpoint_binding,
        expected_run_units=run_units,
    )
    restored_rng = np.random.Generator(np.random.PCG64DXSM())
    restored_rng.bit_generator.state = loaded.rng_states[0]
    restored_sampler = StableVelocitySampler()
    resumed_steps = 0
    while True:
        resumed = advance_particle_once(
            loaded.executions[0],
            velocity=restored_sampler,
            boundaries=_boundaries(),
            behavior_class="sinking",
            diffusion=DiffusionCoefficients(0.2, 0.1, 0.01),
            settings=_settings(max_backtrack_seconds=3.0),
            rng=restored_rng,
        )
        resumed_steps += int(resumed.stepped)
        if resumed.terminal:
            break
    direct_sampler = StableVelocitySampler()
    direct_rng = np.random.Generator(np.random.PCG64DXSM(run_unit.seed))
    direct_result = run_particle(
        initial_state,
        velocity=direct_sampler,
        boundaries=_boundaries(),
        behavior_class="sinking",
        diffusion=DiffusionCoefficients(0.2, 0.1, 0.01),
        settings=_settings(max_backtrack_seconds=3.0),
        rng=direct_rng,
    )
    resumed_result = finalize_particle_execution(loaded.executions[0])
    assert not resumed.stepped and resumed.terminal
    assert resumed_steps == 2
    assert resumed_result == direct_result
    assert loaded.executions[0].step_count == direct_result.step_count
    assert len(restored_sampler.calls) == 8
    assert len(direct_sampler.calls) == 12
    assert restored_rng.bit_generator.state == direct_rng.bit_generator.state
