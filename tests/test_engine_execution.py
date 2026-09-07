"""可暫停 engine 與既有 run_particle 語意等價測試。"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import box

from lagrangian_backtracking.boundaries import BoundaryGeometry
from lagrangian_backtracking.checkpoint import (
    CheckpointBinding,
    load_execution_checkpoint,
    write_execution_checkpoint,
)
from lagrangian_backtracking.diffusion import DiffusionCoefficients, DiffusionSample
from lagrangian_backtracking.engine import (
    EngineSettings,
    _failure_attributes,
    advance_particle_once,
    finalize_particle_execution,
    initialize_particle_execution,
    run_particle,
)
from lagrangian_backtracking.forcing import OCMNativeMonth
from lagrangian_backtracking.integrators import SamplingContext, SamplingError
from lagrangian_backtracking.mesh import NativeMesh
from lagrangian_backtracking.models import (
    EventType,
    ParticleState,
    ParticleStatus,
    SampleQC,
    VelocitySample,
    VelocitySampleStatus,
)
from lagrangian_backtracking.outputs import read_trajectory_shard, write_trajectory_shard
from lagrangian_backtracking.runner import RunUnit
from lagrangian_backtracking.scenarios import Scenario


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
    assert [item.velocity_sample_status for item in direct.observations] == [
        VelocitySampleStatus.TOTAL_ONLY,
        VelocitySampleStatus.TOTAL_ONLY,
        VelocitySampleStatus.TOTAL_ONLY,
        VelocitySampleStatus.NOT_SAMPLED,
    ]
    assert resumed.observations == direct.observations


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
    queries: list[tuple[float, float, float, int]] = []

    def clipped_velocity(x_m: float, y_m: float, z_m: float, time_utc_ns: int) -> VelocitySample:
        """只在 flow domain 內回傳有效樣本，模擬 native mesh 的保守域外結果。"""

        queries.append((x_m, y_m, z_m, time_utc_ns))
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
    direct_rng = np.random.Generator(np.random.PCG64DXSM(9))
    direct = run_particle(
        _state(),
        velocity=clipped_velocity,
        boundaries=boundaries,
        behavior_class="suspended",
        diffusion=DiffusionCoefficients(0.0, 0.0, 0.0),
        settings=settings,
        rng=direct_rng,
    )
    direct_queries = list(queries)
    queries.clear()
    execution = initialize_particle_execution(_state(), settings)
    resumed_rng = np.random.Generator(np.random.PCG64DXSM(9))
    while not advance_particle_once(
        execution,
        velocity=clipped_velocity,
        boundaries=boundaries,
        behavior_class="suspended",
        diffusion=DiffusionCoefficients(0.0, 0.0, 0.0),
        settings=settings,
        rng=resumed_rng,
    ):
        pass
    resumed = finalize_particle_execution(execution)

    assert resumed.final_state == direct.final_state
    assert resumed.observations == direct.observations
    assert resumed.events == direct.events
    assert resumed.events[-1].attributes["boundary_locator"] == "reference_drift_after_rk_stage_invalid"
    # 既有邊界恢復只完成第一步的四階／擴散；第二步 k2 域外後用步首漂移定位 x=-5。
    assert queries == direct_queries == [
        (x_m, 0.0, -10.0, time_ns)
        for x_m, time_ns in (
            (0.0, 100_000_000_000), (0.0, 100_000_000_000),
            (-2.0, 98_000_000_000), (-2.0, 98_000_000_000), (-4.0, 96_000_000_000),
            (-4.0, 96_000_000_000), (-4.0, 96_000_000_000), (-6.0, 94_000_000_000),
        )
    ]
    assert resumed.final_state.x_m == -5.0
    assert resumed.final_state.time_utc_ns == 95_000_000_000
    assert resumed.final_state.status == ParticleStatus.FLOW_DOMAIN_EXIT
    assert resumed.step_count == 2
    assert all("diagnostic_version" not in event.attributes for event in resumed.events)
    assert resumed.events[-1].attributes["requires_dt_halving_validation"] is True
    expected_rng = np.random.Generator(np.random.PCG64DXSM(9))
    expected_rng.normal(size=3)
    assert direct_rng.bit_generator.state == resumed_rng.bit_generator.state
    assert direct_rng.bit_generator.state == expected_rng.bit_generator.state


def test_backward_sinking_surface_stage_recovery_reflects_and_continues() -> None:
    """反向沉降越過已知海面時先縮短 RK4，再以既有反射定位並繼續回溯。

    合成速度的沉降分量為 -0.4 m/s，步首在海面下；大步長的 k4 會落到 eta=0 以上，
    且失效 sample 仍提供有限 bed。engine 應先在不消耗 RNG 的前提下把步長二分到
    ``dt_min``；若最小設定步長仍會越面，則沿用 reference drift 的可驗證海面定位，
    將位置鏡射後繼續下一步。每個 active accepted step 仍只取一次 Brownian；因此
    stage retry 不消耗亂數，但 fallback 反射後的非終止步也必須消耗一次。
    """

    boundaries = _boundaries()
    queries: list[tuple[float, float, float, int]] = []

    def velocity(x_m: float, y_m: float, z_m: float, time_utc_ns: int) -> VelocitySample:
        """只在海面下提供速度；海面以上回傳帶完整幾何 context 的垂向失敗。"""

        del x_m
        queries.append((0.0, 0.0, z_m, time_utc_ns))
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

    settings = _settings(
        dt_min_seconds=1.0,
        dt_max_seconds=2.0,
        max_backtrack_seconds=3.0,
        output_interval_seconds=1.0,
    )
    initial = replace(_state(), z_m=-0.5)
    rng = np.random.Generator(np.random.PCG64DXSM(20260907))
    expected_rng = np.random.Generator(np.random.PCG64DXSM(20260907))
    result = run_particle(
        initial,
        velocity=velocity,
        boundaries=boundaries,
        behavior_class="sinking",
        diffusion=DiffusionCoefficients(0.0, 0.0, 0.0),
        settings=settings,
        rng=rng,
    )

    expected_rng.normal(size=3)
    expected_rng.normal(size=3)
    expected_rng.normal(size=3)
    assert result.final_state.status == ParticleStatus.MAX_AGE
    assert result.step_count == 3
    assert result.minimum_clamp_count == 0
    assert [event.event_type for event in result.events] == [
        EventType.SURFACE_CONTACT,
        EventType.SURFACE_CONTACT,
        EventType.MAX_AGE,
    ]
    assert all(event.event_type != EventType.NUMERICAL_FAILURE for event in result.events)
    assert all(
        event.attributes["boundary_locator"] == "reference_drift_after_rk_stage_invalid"
        for event in result.events[:2]
    )
    assert rng.bit_generator.state == expected_rng.bit_generator.state
    assert any(z_m > 0.0 for _, _, z_m, _ in queries)


def test_engine_near_surface_sinking_uses_moving_surface_endpoint_hold() -> None:
    """近海面 backward sinking 不應只因 after endpoint top-layer gap 變數值失敗。

    合成 OCM 的 before 最高 ``zcor`` 為 1.77 m、after 降至 1.47 m；粒子固定 z=1.5 m
    但 query-time eta 約為 1.62 m，因此它仍在當時水柱內。engine 每個 RK stage 都由
    ``OCMNativeMonth.sample`` 取樣，after endpoint 必須使用最高層 surface hold，最後
    再由 query-time eta／bed gate 判定有效。沉降速度為 -0.001 m/s（m/s、向上為正），
    backward signed-time 只讓粒子緩慢向上移動，故本案例不應觸發海面 recovery；若
    endpoint hold 缺失，原有行為會在 stage query 產生 ``VERTICAL_UNSUPPORTED`` 並將
    粒子記成 ``NUMERICAL_FAILURE``。這只是數值工程 regression，不代表真實 SERVER
    run 或科學驗證已完成。
    """

    mesh = NativeMesh(
        node_lon=np.array([121.0, 121.001, 121.0]),
        node_lat=np.array([25.0, 25.0, 25.001]),
        node_xy=np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]]),
        source_depth_m=np.full(3, 10.0),
        source_node_bottom_index=np.zeros(3, dtype=np.int64),
        face_nodes_local=np.array([[0, 1, 2, -1]]),
        face_node_count=np.array([3]),
        source_face_global_index=np.array([99]),
    )
    time_utc_ns = np.array([0, 1_000_000_000], dtype=np.int64)
    zcor = np.empty((2, 3, 3), dtype=np.float64)
    zcor[0, :, :] = np.array([-10.0, -5.0, 1.77])
    zcor[1, :, :] = np.array([-10.0, -5.0, 1.47])
    hvel = np.zeros((2, 3, 3, 2), dtype=np.float64)
    vertical_velocity = np.full((2, 3, 3), -0.001, dtype=np.float64)
    diffusivity = np.full((2, 3, 3), 0.01, dtype=np.float64)
    elev = np.array([[1.77, 1.77, 1.77], [1.47, 1.47, 1.47]], dtype=np.float64)
    ocm = OCMNativeMonth(
        month_id="197001",
        mesh=mesh,
        time_utc_ns=time_utc_ns,
        hvel=hvel,
        vertical_velocity=vertical_velocity,
        zcor=zcor,
        elev=elev,
        wetdry_elem=np.zeros((2, 1), dtype=np.float64),
        diffusivity=diffusivity,
        maximum_time_gap_seconds=2.0,
    )
    settings = _settings(
        dt_min_seconds=0.01,
        dt_max_seconds=0.1,
        output_interval_seconds=0.1,
        max_backtrack_seconds=0.2,
        maximum_step_count=10,
    )
    initial = replace(
        _state(time_utc_ns=500_000_000),
        x_m=2.0,
        y_m=3.0,
        z_m=1.5,
    )

    result = run_particle(
        initial,
        velocity=ocm.sample,
        boundaries=_boundaries(),
        behavior_class="sinking",
        diffusion=DiffusionCoefficients(0.0, 0.0, 0.0),
        settings=settings,
        rng=np.random.Generator(np.random.PCG64DXSM(20260907)),
    )

    assert result.final_state.status == ParticleStatus.MAX_AGE
    assert all(event.event_type != EventType.NUMERICAL_FAILURE for event in result.events)
    assert all(observation.status != ParticleStatus.NUMERICAL_FAILURE for observation in result.observations)


def test_surface_stage_recovery_applies_one_nonzero_diffusion_after_reflection() -> None:
    """active 海面 recovery 必須先反射，再套用恰一次非零 Kh/Kz 擴散並解析邊界。

    本案例故意讓 ``dt=2`` 的 RK4 k4 越過已知海面，故不能靠縮短至設定的 ``dt_min=2``
    通過，只能進入 active reference-drift recovery。反射後的確定性末端 z 為 -0.3 m；
    測試以固定 seed 重算一次三軸 Brownian 位移，確認水平與垂向非零擴散都被套用，且
    stage 失敗本身沒有先消耗 RNG。此 seed 的垂向擴散會再次越過海面，因此也直接確認
    擴散後仍沿既有垂向邊界解析流程反射，而不是把第一次 reference-drift 位置當成
    accepted step 終點。
    """

    def velocity(x_m: float, y_m: float, z_m: float, time_utc_ns: int) -> VelocitySample:
        """海面以上只回傳帶有限上下界的垂向失敗，海面下回傳固定沉降速度。"""

        del x_m, y_m, time_utc_ns
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

    settings = _settings(
        dt_min_seconds=2.0,
        dt_max_seconds=2.0,
        max_backtrack_seconds=2.0,
        output_interval_seconds=2.0,
    )
    coefficients = DiffusionCoefficients(1.0, 0.25, 0.005)
    divergence = (0.1, -0.05, 0.02)

    class Provider:
        """回傳固定非零 Kh/Kz 與 div(K)，檢查 recovery 沿用完整 diffusion sample。"""

        def __init__(self) -> None:
            self.calls = 0

        def sample(
            self,
            x_m: float,
            y_m: float,
            z_m: float,
            time_utc_ns: int,
            triangle_hint: int | None = None,
        ) -> DiffusionSample:
            """只取樣一次步首 diffusion；輸入位置與 triangle 僅供介面相容。"""

            del x_m, y_m, z_m, time_utc_ns, triangle_hint
            self.calls += 1
            return DiffusionSample(coefficients, divergence)

    provider = Provider()
    initial = replace(_state(), z_m=-0.5)
    rng = np.random.Generator(np.random.PCG64DXSM(6))
    expected_rng = np.random.Generator(np.random.PCG64DXSM(6))
    normal = expected_rng.normal(size=3)
    expected_position = (
        divergence[0] * 2.0 + normal[0] * np.sqrt(2.0 * coefficients.kx_m2ps * 2.0),
        divergence[1] * 2.0 + normal[1] * np.sqrt(2.0 * coefficients.ky_m2ps * 2.0),
        -0.3 + divergence[2] * 2.0 + normal[2] * np.sqrt(2.0 * coefficients.kz_m2ps * 2.0),
    )
    expected_position = (*expected_position[:2], -expected_position[2])

    result = run_particle(
        initial,
        velocity=velocity,
        boundaries=_boundaries(),
        behavior_class="sinking",
        diffusion=provider,
        settings=settings,
        rng=rng,
    )

    assert result.step_count == 1
    assert result.final_state.status == ParticleStatus.MAX_AGE
    assert provider.calls == 1
    # max-age 是下一次步首的 terminal 狀態，沿用同一個最後位置；這裡直接檢查
    # final_state，避免把輸出 observation cadence 與 accepted step 的位置混為一談。
    assert np.isclose(result.final_state.x_m, expected_position[0])
    assert np.isclose(result.final_state.y_m, expected_position[1])
    assert np.isclose(result.final_state.z_m, expected_position[2])
    assert [event.event_type for event in result.events] == [
        EventType.SURFACE_CONTACT,
        EventType.SURFACE_CONTACT,
        EventType.MAX_AGE,
    ]
    assert result.events[0].attributes["boundary_locator"] == (
        "reference_drift_after_rk_stage_invalid"
    )
    assert rng.bit_generator.state == expected_rng.bit_generator.state


def test_step_start_surface_recovery_resamples_and_preserves_unknown_failure() -> None:
    """步首可證明海面越界時重取樣續跑；缺 eta 的真正垂向失敗仍立即停止。"""

    calls: list[float] = []

    def velocity(x_m: float, y_m: float, z_m: float, time_utc_ns: int) -> VelocitySample:
        """記錄步首反射前後的 z；未知上下界不提供任何可猜測的邊界。"""

        del x_m, time_utc_ns
        calls.append(z_m)
        if z_m > 0.0:
            return VelocitySample(
                0.0,
                0.0,
                -0.1,
                0.0,
                -100.0,
                100.0,
                100.0,
                SampleQC.VERTICAL_UNSUPPORTED,
            )
        return VelocitySample(0.0, 0.0, -0.1, 0.0, -100.0, 100.0, 100.0)

    settings = _settings(
        dt_min_seconds=1.0,
        dt_max_seconds=1.0,
        max_backtrack_seconds=1.0,
        output_interval_seconds=1.0,
    )
    execution = initialize_particle_execution(replace(_state(), z_m=0.5), settings)
    rng = np.random.Generator(np.random.PCG64DXSM(20260908))
    outcome = advance_particle_once(
        execution,
        velocity=velocity,
        boundaries=_boundaries(),
        behavior_class="sinking",
        diffusion=DiffusionCoefficients(0.0, 0.0, 0.0),
        settings=settings,
        rng=rng,
    )
    assert outcome.stepped and not outcome.terminal
    assert calls[:2] == [0.5, -0.5]
    assert execution.events[0].event_type == EventType.SURFACE_CONTACT
    assert execution.events[0].fraction == 0.0
    assert execution.state.z_m == -0.4

    missing_eta_execution = initialize_particle_execution(_state(), settings)

    def missing_eta_velocity(
        x_m: float, y_m: float, z_m: float, time_utc_ns: int
    ) -> VelocitySample:
        """保留 VERTICAL_UNSUPPORTED 但省略 eta，驗證不得把未知資料當海面。"""

        del x_m, z_m, time_utc_ns
        return VelocitySample(
            0.0,
            0.0,
            -0.1,
            np.nan,
            -100.0,
            100.0,
            100.0,
            SampleQC.VERTICAL_UNSUPPORTED,
        )

    failed = advance_particle_once(
        missing_eta_execution,
        velocity=missing_eta_velocity,
        boundaries=_boundaries(),
        behavior_class="sinking",
        diffusion=DiffusionCoefficients(0.0, 0.0, 0.0),
        settings=settings,
        rng=np.random.default_rng(20260908),
    )
    assert failed.terminal and not failed.stepped
    assert failed.state.status == ParticleStatus.NUMERICAL_FAILURE
    assert [event.event_type for event in missing_eta_execution.events] == [EventType.NUMERICAL_FAILURE]


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
    attributes = execution.events[-1].attributes
    assert attributes["failure_reason"] == "invalid_diffusion_sample"
    assert attributes["failure_stage"] == "diffusion"
    assert attributes["qc_flags"] == int(SampleQC.INVALID_PHYSICS)
    assert attributes["sampling_context_available"] is True
    assert attributes["sample_eta_available"] is False
    assert attributes["sample_bed_available"] is False
    assert attributes["attempted_dt_available"] is False
    _assert_safe_diagnostic(attributes)


def _assert_safe_diagnostic(attributes: dict[str, bool | float | int | str]) -> None:
    """確認診斷只含既有純量型別，缺值沒有被寫為 null、非有限值或任意路徑。"""

    assert attributes["diagnostic_version"] == 1
    assert all(type(value) in (bool, float, int, str) for value in attributes.values())
    encoded = json.dumps(attributes, allow_nan=False)
    assert json.loads(encoded) == attributes
    assert "/not-a-real-path" not in encoded


@pytest.mark.parametrize("qc", [
    SampleQC.VERTICAL_UNSUPPORTED, SampleQC.NUMERICAL_FAILURE,
    SampleQC.TIME_GAP, SampleQC.WAVE_UNSUPPORTED,
])
def test_step_start_failure_records_same_sample_without_advancing(qc: SampleQC) -> None:
    """步首無效樣本保留品質位元與已知海床，未知海面省略；狀態分類與查詢次數不變。"""

    calls: list[tuple[float, float, float, int]] = []

    def velocity(x_m: float, y_m: float, z_m: float, time_utc_ns: int) -> VelocitySample:
        """只回傳一次失敗樣本；任意字典內的路徑與 NaN 不得進入事件。"""

        calls.append((x_m, y_m, z_m, time_utc_ns))
        return VelocitySample(
            np.nan, np.nan, np.nan, np.nan, -100.0, 100.0, 10.0, qc,
            diagnostics={"path": "/not-a-real-path/private", "nan": np.nan},
        )

    settings = _settings()
    execution = initialize_particle_execution(_state(), settings)
    rng = np.random.Generator(np.random.PCG64DXSM(5))
    before = deepcopy(rng.bit_generator.state)
    result = advance_particle_once(
        execution, velocity=velocity, boundaries=_boundaries(), behavior_class="sinking",
        diffusion=DiffusionCoefficients(0.0, 0.0, 0.0), settings=settings, rng=rng,
    )
    expected_status = (
        ParticleStatus.DATA_GAP
        if qc & (SampleQC.TIME_GAP | SampleQC.WAVE_UNSUPPORTED)
        else ParticleStatus.NUMERICAL_FAILURE
    )
    assert result.terminal and not result.stepped
    assert execution.state == replace(_state(), status=expected_status)
    assert calls == [(0.0, 0.0, -10.0, _state().time_utc_ns)]
    assert rng.bit_generator.state == before
    attributes = execution.events[-1].attributes
    assert attributes["failure_reason"] == "invalid_velocity_sample"
    assert attributes["failure_stage"] == "step_start"
    assert attributes["qc_flags"] == int(qc)
    assert attributes["sampling_context_available"] is True
    assert attributes["sample_time_utc_ns"] == _state().time_utc_ns
    assert attributes["sample_z_m"] == -10.0
    assert attributes["sample_eta_available"] is False
    assert "sample_eta_m" not in attributes
    assert attributes["sample_bed_z_m"] == -100.0
    assert attributes["sample_bed_available"] is True
    assert attributes["attempted_dt_available"] is False
    assert "attempted_dt_seconds" not in attributes
    assert attributes["step_count"] == attributes["minimum_clamp_count"] == 0
    _assert_safe_diagnostic(attributes)


@pytest.mark.parametrize("stage_index", [2, 4])
@pytest.mark.parametrize("qc", [SampleQC.VERTICAL_UNSUPPORTED, SampleQC.TIME_GAP, SampleQC.NUMERICAL_FAILURE])
def test_unrecoverable_rk_failure_records_failed_query_not_terminal_position(
    stage_index: int, qc: SampleQC
) -> None:
    """原直線邊界判定無法恢復時，失敗中間點與最後有效位置必須分開保存。"""

    calls: list[tuple[float, float, float, int]] = []

    def velocity(x_m: float, y_m: float, z_m: float, time_utc_ns: int) -> VelocitySample:
        """步首先成功，其後在指定四階計算點回傳品質失敗，不製造物理邊界穿越。"""

        calls.append((x_m, y_m, z_m, time_utc_ns))
        return VelocitySample(
            1.0, -0.5, 0.25, 0.3, -90.0, 100.0, 10.0,
            qc if len(calls) == stage_index + 1 else SampleQC.OK,
        )

    execution = initialize_particle_execution(_state(), _settings())
    rng = np.random.Generator(np.random.PCG64DXSM(8))
    before = deepcopy(rng.bit_generator.state)
    result = advance_particle_once(
        execution, velocity=velocity, boundaries=_boundaries(), behavior_class="sinking",
        diffusion=DiffusionCoefficients(0.1, 0.1, 0.01), settings=_settings(), rng=rng,
    )
    expected_status = ParticleStatus.DATA_GAP if qc == SampleQC.TIME_GAP else ParticleStatus.NUMERICAL_FAILURE
    assert result.terminal and not result.stepped
    assert execution.state == replace(_state(), status=expected_status)
    assert len(calls) == stage_index + 1
    assert rng.bit_generator.state == before
    assert execution.step_count == execution.minimum_clamp_count == 0
    event = execution.events[-1]
    assert (event.x_m, event.y_m, event.z_m, event.time_utc_ns) == calls[0]
    attributes = event.attributes
    assert attributes["failure_reason"] == "rk_stage_unrecoverable"
    assert attributes["failure_stage"] == f"k{stage_index}"
    assert attributes["qc_flags"] == int(qc)
    assert attributes["attempted_dt_seconds"] == -2.0
    assert tuple(attributes[key] for key in (
        "sample_x_m", "sample_y_m", "sample_z_m", "sample_time_utc_ns"
    )) == calls[-1]
    assert calls[-1] != calls[0]
    assert attributes["sample_eta_m"] == 0.3
    assert attributes["sample_bed_z_m"] == -90.0
    _assert_safe_diagnostic(attributes)


@pytest.mark.parametrize("limit", ["maximum_step_count", "minimum_clamp_limit"])
def test_limit_failure_distinguishes_step_count_and_minimum_clamps(limit: str) -> None:
    """最大步數與第 101 次下限累計分開標記，不假造失敗樣本或更動任何上限。"""

    calls: list[int] = []

    def velocity(x_m: float, y_m: float, z_m: float, time_utc_ns: int) -> VelocitySample:
        """極小水平尺度只用於觸發既有步長下限，不執行四階積分。"""

        del x_m, y_m, z_m
        calls.append(time_utc_ns)
        return VelocitySample(1.0, 0.0, 0.0, 0.0, -100.0, 0.001, 10.0)

    settings = _settings()
    execution = initialize_particle_execution(_state(), settings)
    if limit == "maximum_step_count":
        execution.step_count = settings.maximum_step_count
    else:
        execution.minimum_clamp_count = settings.maximum_minimum_clamps
    rng = np.random.Generator(np.random.PCG64DXSM(9))
    before = deepcopy(rng.bit_generator.state)
    result = advance_particle_once(
        execution, velocity=velocity, boundaries=_boundaries(), behavior_class="sinking",
        diffusion=DiffusionCoefficients(0.0, 0.0, 0.0), settings=settings, rng=rng,
    )
    assert result.terminal and not result.stepped
    assert result.state == replace(_state(), status=ParticleStatus.NUMERICAL_FAILURE)
    assert len(calls) == (0 if limit == "maximum_step_count" else 1)
    assert rng.bit_generator.state == before
    attributes = execution.events[-1].attributes
    assert attributes["failure_stage"] == "limits"
    assert attributes["failure_reason"] == limit
    assert attributes["step_count"] == execution.step_count
    assert attributes["minimum_clamp_count"] == execution.minimum_clamp_count
    assert attributes["maximum_step_count"] == settings.maximum_step_count
    assert attributes["maximum_minimum_clamps"] == 100
    assert attributes["dt_min_seconds"] == 1.0
    assert attributes["dt_max_seconds"] == 2.0
    assert attributes["sampling_context_available"] is False
    assert attributes["qc_available"] is False
    assert "qc_flags" not in attributes
    if limit == "minimum_clamp_limit":
        assert attributes["minimum_clamp_count"] == 101
        assert attributes["attempted_dt_seconds"] == -1.0
    else:
        assert "attempted_dt_seconds" not in attributes
    _assert_safe_diagnostic(attributes)


@pytest.mark.parametrize("error_type", [TypeError, ValueError])
def test_diffusion_evaluation_error_omits_untrusted_exception_message(error_type: type[Exception]) -> None:
    """擴散評估例外沿用數值失敗狀態，只保存已知查詢位置，不把例外路徑當診斷。"""

    class BrokenProvider:
        """模擬現有擴散解析分支會捕捉的型別／數值錯誤。"""

        def sample(self, *args: object, **kwargs: object) -> DiffusionSample:
            """拋出含不可信文字的例外，以驗證輸出沒有複製訊息。"""

            raise error_type("/not-a-real-path/private NaN")

    execution = initialize_particle_execution(_state(), _settings())
    rng = np.random.Generator(np.random.PCG64DXSM(10))
    before = deepcopy(rng.bit_generator.state)
    advance_particle_once(
        execution, velocity=_position_dependent_velocity, boundaries=_boundaries(),
        behavior_class="sinking", diffusion=BrokenProvider(), settings=_settings(), rng=rng,
    )
    assert execution.state.status == ParticleStatus.NUMERICAL_FAILURE
    assert rng.bit_generator.state == before
    attributes = execution.events[-1].attributes
    assert attributes["failure_reason"] == "diffusion_evaluation_error"
    assert attributes["failure_stage"] == "diffusion"
    assert attributes["sampling_context_available"] is True
    assert attributes["qc_available"] is False
    assert "qc_flags" not in attributes
    _assert_safe_diagnostic(attributes)


@pytest.mark.parametrize("context", [None, SamplingContext(), SamplingContext(
    x_m=np.nan, y_m=np.inf, z_m=-np.inf, time_utc_ns=np.nan, eta_m=np.nan, bed_z_m=np.inf,
)])
def test_missing_or_nonfinite_context_omitted_from_event_and_roundtrips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, context: SamplingContext | None
) -> None:
    """缺值政策能嚴格 JSON、中途續跑與 v3 軌跡往返，舊空屬性亦仍可讀。

    使用合成資料注入外部非標準階段；事件必須標為 unknown 並省略未知數值。
    所有檔案只寫測試暫存目錄，不變更輸出實作或檔案格式版本。
    """

    def fail_step(*args: object, **kwargs: object) -> ParticleState:
        """模擬既有取樣例外，不讓外部階段文字進入序列化資料。"""

        raise SamplingError("/not-a-real-path/private", SampleQC.TIME_GAP, context=context)

    monkeypatch.setattr("lagrangian_backtracking.engine.split_rk4_brownian_step", fail_step)
    execution = initialize_particle_execution(_state(), _settings())
    rng = np.random.Generator(np.random.PCG64DXSM(12))
    advance_particle_once(
        execution, velocity=_position_dependent_velocity, boundaries=_boundaries(),
        behavior_class="sinking", diffusion=DiffusionCoefficients(0.0, 0.0, 0.0),
        settings=_settings(), rng=rng,
    )
    attributes = execution.events[-1].attributes
    assert attributes["failure_stage"] == "unknown"
    assert attributes["sampling_context_available"] is False
    assert attributes["sample_eta_available"] is False
    assert attributes["sample_bed_available"] is False
    assert not set(attributes) & {
        "sample_x_m", "sample_y_m", "sample_z_m", "sample_time_utc_ns", "sample_eta_m", "sample_bed_z_m",
    }
    _assert_safe_diagnostic(attributes)
    scenario = Scenario("s0", "gongliao", "A", "material", "r0", "arrival", -0.002, 100_000_000_000, "test")
    unit = RunUnit(scenario, "baseline", 0, "p0", 12)
    binding = CheckpointBinding("config", "inventory", "baseline", "shard", "pcg64dxsm-v1", "commit")
    checkpoint_path = write_execution_checkpoint(
        tmp_path / "checkpoint", binding=binding, run_units=[unit], executions=[execution],
        rngs=[rng], triangle_hints=[None], sequence=0,
    )
    restored = load_execution_checkpoint(checkpoint_path, expected_binding=binding, expected_run_units=[unit])
    assert restored.executions[0] == execution
    assert restored.rng_states[0] == rng.bit_generator.state
    assert json.loads((checkpoint_path / "checkpoint.json").read_text())["schema_version"] == "2.2.0"
    result = finalize_particle_execution(restored.executions[0])
    for name, event_attributes in (
        ("diagnostic", attributes), ("legacy", {}), ("ordinary", {"label": "old"}),
    ):
        candidate = replace(result, events=[replace(result.events[-1], attributes=event_attributes)])
        shard = write_trajectory_shard(tmp_path / name, [candidate], run_metadata={"run_kind": "synthetic"})
        assert json.loads((shard / "manifest.json").read_text())["schema_version"] == "3.0.0"
        assert read_trajectory_shard(shard) == (candidate,)


def test_diagnostic_whitelist_preserves_partial_context_without_coercing_unknown_values() -> None:
    """部分有限值保留原義，未知旗標／時間／步長省略；任意理由與階段退為 unknown。"""

    attributes = _failure_attributes(
        initialize_particle_execution(_state(), _settings()), _settings(),
        reason="/not-a-real-path/private", stage="/not-a-real-path/private",
        context=SamplingContext(np.float64(2.5), np.nan, -3.0, 1 << 65, 0.0, np.nan),
        attempted_dt_seconds=np.nan,
    )
    assert attributes["failure_reason"] == attributes["failure_stage"] == "unknown"
    assert attributes["sample_x_m"] == 2.5
    assert attributes["sample_z_m"] == -3.0
    assert attributes["sample_eta_m"] == 0.0  # 真正觀測到的零高程可以保留，不是補值。
    assert attributes["sample_eta_available"] is True
    assert attributes["sample_bed_available"] is False
    assert attributes["sampling_context_available"] is False
    assert attributes["attempted_dt_available"] is False
    assert not set(attributes) & {
        "sample_y_m", "sample_bed_z_m", "sample_time_utc_ns", "attempted_dt_seconds",
    }
    _assert_safe_diagnostic(attributes)


def test_successful_execution_matches_analytic_queries_positions_and_rng() -> None:
    """常流加擴散的成功案例鎖住每步查詢、位置、輸出時序與完整亂數狀態。

    對照值獨立由常流解析位移與每步一次三向布朗亂數建立，不用另一個引擎入口互比。
    三步為 2、2、1 秒，最長回溯停止仍保留舊空屬性，不因新增失敗診斷而改寫成功事件。
    """

    queries: list[tuple[float, float, float, int]] = []

    def velocity(x_m: float, y_m: float, z_m: float, time_utc_ns: int) -> VelocitySample:
        """常流讓四階中間位置可解析核對，數值尺度足以維持原設定步長。"""

        queries.append((x_m, y_m, z_m, time_utc_ns))
        return VelocitySample(1.0, -0.5, 0.0, 0.0, -100.0, 100.0, 10.0)

    rng = np.random.Generator(np.random.PCG64DXSM(22))
    expected_rng = np.random.Generator(np.random.PCG64DXSM(22))
    result = run_particle(
        _state(), velocity=velocity, boundaries=_boundaries(), behavior_class="sinking",
        diffusion=DiffusionCoefficients(0.2, 0.1, 0.01), settings=_settings(), rng=rng,
    )
    position = np.array([0.0, 0.0, -10.0])
    time_ns = _state().time_utc_ns
    expected_queries = []
    expected_positions = [tuple(position)]
    flow = np.array([1.0, -0.5, 0.0])
    for dt in (2.0, 2.0, 1.0):
        for fraction in (0.0, 0.0, 0.5, 0.5, 1.0):
            expected_queries.append((
                *(position - fraction * dt * flow), time_ns - int(fraction * dt * 1_000_000_000),
            ))
        position = position - dt * flow + np.sqrt(2.0 * np.array([0.2, 0.1, 0.01]) * dt) * (
            expected_rng.normal(size=3)
        )
        expected_positions.append(tuple(position))
        time_ns -= int(dt * 1_000_000_000)
    assert queries == expected_queries
    assert [(item.x_m, item.y_m, item.z_m) for item in result.observations] == expected_positions
    assert [item.age_seconds for item in result.observations] == [0.0, 2.0, 4.0, 5.0]
    assert [item.velocity_sample_status for item in result.observations] == [
        VelocitySampleStatus.TOTAL_ONLY,
        VelocitySampleStatus.TOTAL_ONLY,
        VelocitySampleStatus.TOTAL_ONLY,
        VelocitySampleStatus.NOT_SAMPLED,
    ]
    assert all(
        all(getattr(item, name) is None for name in (
            "ocm_u_mps", "ocm_v_mps", "ocm_w_mps",
            "stokes_u_mps", "stokes_v_mps", "settling_w_mps",
        ))
        for item in result.observations
    )
    assert result.final_state == replace(
        _state(), x_m=position[0], y_m=position[1], z_m=position[2], time_utc_ns=time_ns,
        age_seconds=5.0, status=ParticleStatus.MAX_AGE,
    )
    assert result.step_count == 3
    assert result.minimum_clamp_count == 0
    assert len(result.events) == 1
    assert result.events[0].event_type.value == "max_age"
    assert result.events[0].attributes == {}
    assert rng.bit_generator.state == expected_rng.bit_generator.state


def test_checkpoint_resume_retains_exact_rk_failure_diagnostics(tmp_path: Path) -> None:
    """先成功一步、寫中途續跑檔，再於 k4 失敗；重啟與直跑須連診斷及亂數完全一致。

    合成流場只在固定 UTC 門檻回傳垂向不支援，不依呼叫次數或亂數選失敗情境。
    同時以既有 v3 輸出讀回有限位置／時間／上下界，確認診斷沒有引入新格式。
    """

    def velocity(x_m: float, y_m: float, z_m: float, time_utc_ns: int) -> VelocitySample:
        """第二步的第四個計算點失敗；步首與較早中間點保持有效。"""

        del x_m, y_m, z_m
        qc = SampleQC.VERTICAL_UNSUPPORTED if time_utc_ns <= 96_000_000_000 else SampleQC.OK
        return VelocitySample(1.0, 0.0, 0.0, 0.0, -100.0, 100.0, 10.0, qc)

    settings = _settings()
    arguments = {
        "velocity": velocity, "boundaries": _boundaries(), "behavior_class": "sinking",
        "diffusion": DiffusionCoefficients(0.2, 0.1, 0.01), "settings": settings,
    }
    direct_rng = np.random.Generator(np.random.PCG64DXSM(16))
    direct = run_particle(_state(), rng=direct_rng, **arguments)
    rng = np.random.Generator(np.random.PCG64DXSM(16))
    execution = initialize_particle_execution(_state(), settings)
    first = advance_particle_once(execution, rng=rng, **arguments)
    assert first.stepped and not first.terminal
    scenario = Scenario("s0", "gongliao", "A", "material", "r0", "arrival", -0.002, 100_000_000_000, "test")
    unit = RunUnit(scenario, "baseline", 0, "p0", 16)
    binding = CheckpointBinding("config", "inventory", "baseline", "shard", "pcg64dxsm-v1", "commit")
    path = write_execution_checkpoint(
        tmp_path / "active", binding=binding, run_units=[unit], executions=[execution],
        rngs=[rng], triangle_hints=[None], sequence=1,
    )
    checkpoint = load_execution_checkpoint(path, expected_binding=binding, expected_run_units=[unit])
    restored_rng = np.random.Generator(np.random.PCG64DXSM())
    restored_rng.bit_generator.state = checkpoint.rng_states[0]
    restored = checkpoint.executions[0]
    outcome = advance_particle_once(restored, rng=restored_rng, **arguments)
    assert outcome.terminal and not outcome.stepped
    result = finalize_particle_execution(restored)
    assert result == direct
    assert [item.velocity_sample_status for item in result.observations] == [
        VelocitySampleStatus.TOTAL_ONLY,
        VelocitySampleStatus.TOTAL_ONLY,
    ]
    assert [item.velocity_sample_status for item in restored.observations] == [
        item.velocity_sample_status for item in direct.observations
    ]
    assert restored_rng.bit_generator.state == direct_rng.bit_generator.state
    attributes = result.events[-1].attributes
    assert attributes["failure_stage"] == "k4"
    assert attributes["sample_time_utc_ns"] == 96_000_000_000
    assert result.final_state.time_utc_ns == 98_000_000_000
    assert attributes["sample_eta_available"] is attributes["sample_bed_available"] is True
    _assert_safe_diagnostic(attributes)
    terminal_path = write_execution_checkpoint(
        tmp_path / "terminal", binding=binding, run_units=[unit], executions=[restored],
        rngs=[restored_rng], triangle_hints=[None], sequence=2,
    )
    assert load_execution_checkpoint(terminal_path, expected_binding=binding).executions[0] == restored
    shard = write_trajectory_shard(tmp_path / "output", [result], run_metadata={"run_kind": "synthetic"})
    assert read_trajectory_shard(shard) == (result,)
