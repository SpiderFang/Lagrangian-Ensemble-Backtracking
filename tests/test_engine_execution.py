"""可暫停 engine 與既有 run_particle 語意等價測試。"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from shapely.geometry import box

from lagrangian_backtracking.boundaries import BoundaryGeometry
from lagrangian_backtracking.diffusion import DiffusionCoefficients, DiffusionSample
from lagrangian_backtracking.engine import (
    EngineSettings,
    advance_particle_once,
    finalize_particle_execution,
    initialize_particle_execution,
    run_particle,
)
from lagrangian_backtracking.models import ParticleState, ParticleStatus, SampleQC, VelocitySample


def _state(*, time_utc_ns: int = 100_000_000_000, age_seconds: float = 0.0) -> ParticleState:
    """建立具有完整 identity、且遠離測試邊界的 active 粒子。"""

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
        age_seconds=age_seconds,
    )


def _boundaries() -> BoundaryGeometry:
    """建立不會提前介入正常多步測試的 local/flow domain。"""

    return BoundaryGeometry(
        own_local_domain=box(-100.0, -100.0, 100.0, 100.0),
        flow_domain=box(-1_000.0, -1_000.0, 1_000.0, 1_000.0),
        foreign_local_domains={},
    )


def _settings(**overrides: object) -> EngineSettings:
    """提供固定時間步與上限，並允許個別測試覆蓋步首停止條件。"""

    values: dict[str, object] = {
        "dt_min_seconds": 1.0,
        "dt_max_seconds": 2.0,
        "output_interval_seconds": 2.0,
        "max_backtrack_seconds": 5.0,
        "maximum_step_count": 100,
        "earliest_forcing_time_utc_ns": 0,
    }
    values.update(overrides)
    return EngineSettings(**values)


def _position_dependent_velocity(
    x_m: float, y_m: float, z_m: float, time_utc_ns: int
) -> VelocitySample:
    """建立位置／時間皆可變但在測試域內有效的速度場，檢查 RK4 stage 仍被重複取樣。"""

    del time_utc_ns
    return VelocitySample(
        0.4 + 0.001 * x_m,
        -0.2 + 0.001 * y_m,
        0.01 + 0.0001 * z_m,
        0.0,
        -100.0,
        100.0,
        10.0,
    )


def test_stepwise_execution_is_identical_to_run_particle() -> None:
    """正常多步、非零 Brownian 與固定輸出點的逐次執行必須逐欄等於 wrapper。"""

    diffusion = DiffusionCoefficients(0.2, 0.1, 0.01)
    settings = _settings()
    callback_states: list[ParticleState] = []
    direct = run_particle(
        _state(),
        velocity=_position_dependent_velocity,
        boundaries=_boundaries(),
        behavior_class="sinking",
        diffusion=diffusion,
        settings=settings,
        rng=np.random.Generator(np.random.PCG64DXSM(20260828)),
        on_step=callback_states.append,
    )

    execution = initialize_particle_execution(_state(), settings)
    step_states: list[ParticleState] = []
    step_rng = np.random.Generator(np.random.PCG64DXSM(20260828))
    while not advance_particle_once(
        execution,
        velocity=_position_dependent_velocity,
        boundaries=_boundaries(),
        behavior_class="sinking",
        diffusion=diffusion,
        settings=settings,
        rng=step_rng,
        on_step=step_states.append,
    ):
        pass
    resumed = finalize_particle_execution(execution)

    assert resumed.final_state == direct.final_state
    assert resumed.observations == direct.observations
    assert resumed.events == direct.events
    assert resumed.step_count == direct.step_count
    assert resumed.minimum_clamp_count == direct.minimum_clamp_count
    assert step_states == callback_states
    assert len(callback_states) == direct.step_count


def test_stepwise_engine_preserves_all_step_start_stop_events() -> None:
    """max-step、max-age、forcing-start 都在步首終止且只留下 terminal observation。"""

    cases = (
        (_settings(maximum_step_count=0), _state(), ParticleStatus.NUMERICAL_FAILURE),
        (_settings(), _state(age_seconds=5.0), ParticleStatus.MAX_AGE),
        (_settings(), _state(time_utc_ns=0), ParticleStatus.FORCING_START),
    )
    for settings, state, expected_status in cases:
        callback_states: list[ParticleState] = []
        execution = initialize_particle_execution(state, settings)
        result = advance_particle_once(
            execution,
            velocity=_position_dependent_velocity,
            boundaries=_boundaries(),
            behavior_class="sinking",
            diffusion=DiffusionCoefficients(0.0, 0.0, 0.0),
            settings=settings,
            rng=np.random.Generator(np.random.PCG64DXSM(4)),
            on_step=callback_states.append,
        )
        final = finalize_particle_execution(execution)
        assert result.terminal and not result.stepped
        assert final.final_state.status == expected_status
        assert final.step_count == 0
        assert callback_states == []
        assert len(final.observations) == 1
        assert final.observations[-1].status == expected_status


def test_stepwise_engine_keeps_rk_stage_invalid_boundary_recovery() -> None:
    """RK stage 域外仍使用原 reference drift recovery，不改事件屬性與終止位置。"""

    boundaries = BoundaryGeometry(
        own_local_domain=box(-2.0, -5.0, 2.0, 5.0),
        flow_domain=box(-5.0, -5.0, 5.0, 5.0),
        foreign_local_domains={},
    )

    def clipped_velocity(x_m: float, y_m: float, z_m: float, time_utc_ns: int) -> VelocitySample:
        """只在 flow domain 內回傳有效樣本，模擬 native mesh 的保守域外結果。"""

        del y_m, z_m, time_utc_ns
        if x_m < -5.0 or x_m > 5.0:
            return VelocitySample(
                0.0,
                0.0,
                0.0,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                SampleQC.OUTSIDE_HORIZONTAL_DOMAIN,
            )
        return VelocitySample(1.0, 0.0, 0.0, 0.0, -100.0, 100.0, 10.0)

    settings = EngineSettings(1.0, 4.0, 4.0, 100.0, 100, 0)
    direct = run_particle(
        _state(),
        velocity=clipped_velocity,
        boundaries=boundaries,
        behavior_class="suspended",
        diffusion=DiffusionCoefficients(0.0, 0.0, 0.0),
        settings=settings,
        rng=np.random.Generator(np.random.PCG64DXSM(9)),
    )
    execution = initialize_particle_execution(_state(), settings)
    while not advance_particle_once(
        execution,
        velocity=clipped_velocity,
        boundaries=boundaries,
        behavior_class="suspended",
        diffusion=DiffusionCoefficients(0.0, 0.0, 0.0),
        settings=settings,
        rng=np.random.Generator(np.random.PCG64DXSM(9)),
    ):
        pass
    resumed = finalize_particle_execution(execution)

    assert resumed.final_state == direct.final_state
    assert resumed.observations == direct.observations
    assert resumed.events == direct.events
    assert resumed.events[-1].attributes["boundary_locator"] == "reference_drift_after_rk_stage_invalid"


def test_finalize_adds_terminal_observation_without_advancing() -> None:
    """手動改成 terminal 後 finalize 只補 observation，不取樣也不增加 step。"""

    execution = initialize_particle_execution(_state(), _settings(),)
    execution.state = replace(execution.state, status=ParticleStatus.MAX_AGE)
    result = finalize_particle_execution(execution)

    assert result.step_count == 0
    assert result.final_state.status == ParticleStatus.MAX_AGE
    assert result.observations[-1].status == ParticleStatus.MAX_AGE


def test_engine_samples_spatial_diffusion_once_with_step_start_triangle_hint() -> None:
    """engine 每一步只取樣一次，且把步首 reference 的 triangle ID 原樣傳給 provider。"""

    velocity_calls: list[tuple[float, float, float, int]] = []

    def velocity(x_m: float, y_m: float, z_m: float, time_utc_ns: int) -> VelocitySample:
        """以固定水平速度與 triangle ID 建立可追蹤的步首 reference。"""

        velocity_calls.append((x_m, y_m, z_m, time_utc_ns))
        return VelocitySample(
            1.0,
            0.0,
            0.0,
            0.0,
            -100.0,
            10.0,
            10.0,
            triangle_id=23,
        )

    class Provider:
        """回傳高 K 的合成 provider，專門驗證每步 call 與時間步長耦合。"""

        def __init__(self) -> None:
            self.calls: list[tuple[float, float, float, int, int | None]] = []

        def sample(
            self,
            x_m: float,
            y_m: float,
            z_m: float,
            time_utc_ns: int,
            triangle_hint: int | None = None,
        ) -> DiffusionSample:
            """保存完整步首輸入，回傳有限且有效的固定係數樣本。"""

            self.calls.append((x_m, y_m, z_m, time_utc_ns, triangle_hint))
            return DiffusionSample(DiffusionCoefficients(10.0, 10.0, 10.0), (0.0, 0.0, 0.0))

    provider = Provider()
    settings = _settings(
        dt_min_seconds=0.1,
        dt_max_seconds=10.0,
        max_backtrack_seconds=20.0,
    )
    execution = initialize_particle_execution(
        _state(),
        settings,
    )
    result = advance_particle_once(
        execution,
        velocity=velocity,
        boundaries=BoundaryGeometry(
            own_local_domain=box(-1_000_000.0, -1_000_000.0, 1_000_000.0, 1_000_000.0),
            flow_domain=box(-1_000_000.0, -1_000_000.0, 1_000_000.0, 1_000_000.0),
            foreign_local_domains={},
        ),
        behavior_class="suspended",
        diffusion=provider,
        settings=settings,
        rng=np.random.default_rng(11),
    )

    assert result.stepped is True
    assert len(provider.calls) == 1
    assert provider.calls == [(_state().x_m, _state().y_m, _state().z_m, _state().time_utc_ns, 23)]
    assert len(velocity_calls) == 5
    # min(10, advective 2.5, diffusive (0.25*10)^2/(2*10)) = 0.3125 秒。
    assert execution.state.age_seconds == 0.3125


def test_invalid_spatial_diffusion_stops_before_timestep_rk4_and_rng() -> None:
    """無效擴散 qc 應在選 dt/RK4 前終止，並完全保留粒子 RNG 狀態。"""

    velocity_calls: list[int] = []

    def velocity(x_m: float, y_m: float, z_m: float, time_utc_ns: int) -> VelocitySample:
        """只允許一次步首取樣；若 RK4 被錯誤執行，呼叫數便會暴露問題。"""

        del x_m, y_m, z_m
        velocity_calls.append(time_utc_ns)
        return VelocitySample(1.0, 0.0, 0.0, 0.0, -100.0, 10.0, 10.0, triangle_id=31)

    class InvalidProvider:
        """回傳非零品質旗標的失敗樣本，不以零係數假裝有效。"""

        def __init__(self) -> None:
            self.call_count = 0

        def sample(
            self,
            x_m: float,
            y_m: float,
            z_m: float,
            time_utc_ns: int,
            triangle_hint: int | None = None,
        ) -> DiffusionSample:
            """記錄一次呼叫並回傳 INVALID_PHYSICS。"""

            del x_m, y_m, z_m, time_utc_ns, triangle_hint
            self.call_count += 1
            return DiffusionSample(
                DiffusionCoefficients(0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                qc=SampleQC.INVALID_PHYSICS,
            )

    provider = InvalidProvider()
    execution = initialize_particle_execution(_state(), _settings())
    rng = np.random.Generator(np.random.PCG64DXSM(12))
    control_rng = np.random.Generator(np.random.PCG64DXSM(12))
    result = advance_particle_once(
        execution,
        velocity=velocity,
        boundaries=_boundaries(),
        behavior_class="suspended",
        diffusion=provider,
        settings=_settings(),
        rng=rng,
    )

    assert result.terminal is True
    assert result.stepped is False
    assert result.state.status == ParticleStatus.NUMERICAL_FAILURE
    assert execution.step_count == 0
    assert provider.call_count == 1
    assert velocity_calls == [_state().time_utc_ns]
    assert np.array_equal(rng.normal(size=3), control_rng.normal(size=3))
