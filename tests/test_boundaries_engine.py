"""own/foreign/outer crossing、垂向政策與參考引擎測試。"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from shapely.geometry import LineString, box

from lagrangian_backtracking.boundaries import (
    BoundaryGeometry,
    resolve_horizontal_boundaries,
    resolve_vertical_boundaries,
)
from lagrangian_backtracking.diffusion import DiffusionCoefficients
from lagrangian_backtracking.engine import EngineSettings, run_particle
from lagrangian_backtracking.models import EventType, ParticleState, ParticleStatus, SampleQC, VelocitySample


def _state(x_m: float, *, z_m: float = -5.0) -> ParticleState:
    """建立 A 區貢寮合成粒子。"""

    return ParticleState(
        particle_id="p0",
        scenario_id="s0",
        member_id=0,
        study_site_id="gongliao",
        analysis_region_id="A",
        receptor_id="r0",
        x_m=x_m,
        y_m=0.0,
        z_m=z_m,
        time_utc_ns=100_000_000_000,
    )


def test_own_foreign_and_outer_events_keep_site_identity() -> None:
    """同一步依序離開 own、進入 foreign、離開 outer；foreign 不得終止或改站。"""

    geometry = BoundaryGeometry(
        own_local_domain=box(-10.0, -5.0, 10.0, 5.0),
        flow_domain=box(-50.0, -10.0, 50.0, 10.0),
        foreign_local_domains={"guishan": box(20.0, -5.0, 30.0, 5.0)},
    )
    previous = _state(0.0)
    proposed = replace(previous, x_m=60.0, time_utc_ns=40_000_000_000, age_seconds=60.0)
    resolved, events = resolve_horizontal_boundaries(previous, proposed, geometry)
    assert [event.event_type for event in events] == [
        EventType.LOCAL_DOMAIN_FIRST_EXIT,
        EventType.OTHER_SITE_LOCAL_DOMAIN_ENTER,
        EventType.OTHER_SITE_LOCAL_DOMAIN_EXIT,
        EventType.FLOW_DOMAIN_OPEN_EXIT,
    ]
    assert resolved.status == ParticleStatus.FLOW_DOMAIN_EXIT
    assert resolved.study_site_id == "gongliao"
    assert events[1].related_study_site_id == "guishan"


def test_foreign_boundary_step_endpoint_is_not_double_counted() -> None:
    """前一步 fraction=1 的 foreign enter 不得在下一步 fraction=0 被誤判成 exit。"""

    geometry = BoundaryGeometry(
        own_local_domain=box(-10.0, -5.0, 10.0, 5.0),
        flow_domain=box(-50.0, -10.0, 50.0, 10.0),
        foreign_local_domains={"guishan": box(20.0, -5.0, 30.0, 5.0)},
    )
    first = _state(15.0)
    on_boundary = replace(first, x_m=20.0, time_utc_ns=95_000_000_000, age_seconds=5.0)
    _, first_events = resolve_horizontal_boundaries(first, on_boundary, geometry)
    inside = replace(on_boundary, x_m=25.0, time_utc_ns=90_000_000_000, age_seconds=10.0)
    _, second_events = resolve_horizontal_boundaries(on_boundary, inside, geometry)
    assert [event.event_type for event in first_events] == [EventType.OTHER_SITE_LOCAL_DOMAIN_ENTER]
    assert second_events == []


def test_coast_and_open_boundary_are_not_confused() -> None:
    """同一 local polygon 的東側為開放水域、北側為海岸，事件與停止政策必須分離。"""

    local = box(-10.0, -5.0, 10.0, 5.0)
    flow = box(-50.0, -10.0, 50.0, 10.0)
    geometry = BoundaryGeometry(
        own_local_domain=local,
        flow_domain=flow,
        foreign_local_domains={},
        own_local_open_boundary=LineString([(10.0, -5.0), (10.0, 5.0)]),
        flow_open_boundary=LineString([(50.0, -10.0), (50.0, 10.0)]),
    )

    previous = _state(0.0)
    through_open_water = replace(
        previous,
        x_m=20.0,
        time_utc_ns=80_000_000_000,
        age_seconds=20.0,
    )
    continued, events = resolve_horizontal_boundaries(previous, through_open_water, geometry)
    assert continued.status == ParticleStatus.ACTIVE
    assert continued.own_local_exit_recorded
    assert events[0].event_type == EventType.LOCAL_DOMAIN_FIRST_EXIT
    assert np.isclose(events[0].boundary_s_m, 5.0)

    toward_coast = replace(
        previous,
        y_m=8.0,
        time_utc_ns=80_000_000_000,
        age_seconds=20.0,
    )
    stopped, events = resolve_horizontal_boundaries(previous, toward_coast, geometry)
    assert stopped.status == ParticleStatus.COAST_CONTACT
    assert np.isclose(stopped.y_m, 5.0)
    assert [event.event_type for event in events] == [EventType.COAST_CONTACT]
    assert events[0].boundary_segment_id == "coastline"


def test_local_equals_flow_coast_writes_one_terminal_event() -> None:
    """B-D 重合邊界若穿越海岸只寫 coast，不得誤加具雙重語意的 flow exit。"""

    domain = box(-10.0, -5.0, 10.0, 5.0)
    geometry = BoundaryGeometry(
        own_local_domain=domain,
        flow_domain=domain,
        foreign_local_domains={},
        flow_open_boundary=LineString([(10.0, -5.0), (10.0, 5.0)]),
        local_equals_flow=True,
    )
    previous = _state(0.0)
    proposed = replace(previous, y_m=8.0, time_utc_ns=80_000_000_000, age_seconds=20.0)
    stopped, events = resolve_horizontal_boundaries(previous, proposed, geometry)
    assert stopped.status == ParticleStatus.COAST_CONTACT
    assert [event.event_type for event in events] == [EventType.COAST_CONTACT]


def test_rising_surface_exit_and_suspended_reflection() -> None:
    """相同越面步對 rising 是適用範圍退出，對 suspended 則鏡射。"""

    previous = _state(0.0, z_m=-1.0)
    proposed = replace(previous, z_m=1.0, time_utc_ns=90_000_000_000, age_seconds=10.0)
    sample = VelocitySample(0.0, 0.0, 0.0, 0.0, -10.0, 10.0, 1.0)
    stopped, events = resolve_vertical_boundaries(
        previous, proposed, reference_sample=sample, behavior_class="rising"
    )
    assert stopped.status == ParticleStatus.SURFACE_REGIME_EXIT
    assert events[0].event_type == EventType.SURFACE_REGIME_EXIT
    reflected, events = resolve_vertical_boundaries(
        previous, proposed, reference_sample=sample, behavior_class="suspended"
    )
    assert reflected.status == ParticleStatus.ACTIVE
    assert np.isclose(reflected.z_m, -1.0)
    assert events[0].event_type == EventType.SURFACE_CONTACT


def test_reference_engine_constant_backward_flow_exits_outer_domain() -> None:
    """正向東流的 backward 粒子應向西，先記 own exit 再於 outer boundary 停止。"""

    geometry = BoundaryGeometry(
        own_local_domain=box(-5.0, -10.0, 5.0, 10.0),
        flow_domain=box(-20.0, -10.0, 20.0, 10.0),
        foreign_local_domains={},
    )

    def velocity(x_m: float, y_m: float, z_m: float, time_utc_ns: int) -> VelocitySample:
        del x_m, y_m, z_m, time_utc_ns
        return VelocitySample(1.0, 0.0, 0.0, 0.0, -10.0, 100.0, 1.0)

    result = run_particle(
        _state(0.0),
        velocity=velocity,
        boundaries=geometry,
        behavior_class="suspended",
        diffusion=DiffusionCoefficients(0.0, 0.0, 0.0),
        settings=EngineSettings(
            dt_min_seconds=1.0,
            dt_max_seconds=4.0,
            output_interval_seconds=4.0,
            max_backtrack_seconds=100.0,
            maximum_step_count=100,
            earliest_forcing_time_utc_ns=0,
        ),
        rng=np.random.default_rng(7),
    )
    assert result.final_state.status == ParticleStatus.FLOW_DOMAIN_EXIT
    assert np.isclose(result.final_state.x_m, -20.0)
    assert [event.event_type for event in result.events] == [
        EventType.LOCAL_DOMAIN_FIRST_EXIT,
        EventType.FLOW_DOMAIN_OPEN_EXIT,
    ]


def test_rk_stage_domain_failure_recovers_provable_outer_crossing() -> None:
    """RK stage 先落到 forcing 域外時，已被步首 drift 穿越的 outer boundary 不得誤報數值失敗。"""

    geometry = BoundaryGeometry(
        own_local_domain=box(-2.0, -5.0, 2.0, 5.0),
        flow_domain=box(-5.0, -5.0, 5.0, 5.0),
        foreign_local_domains={},
    )

    def clipped_velocity(x_m: float, y_m: float, z_m: float, time_utc_ns: int) -> VelocitySample:
        """只在 flow polygon 內有效，模擬 native mesh 無域外 triangle 的保守 sampler。"""

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
        return VelocitySample(1.0, 0.0, 0.0, 0.0, -10.0, 100.0, 1.0)

    result = run_particle(
        _state(0.0),
        velocity=clipped_velocity,
        boundaries=geometry,
        behavior_class="suspended",
        diffusion=DiffusionCoefficients(0.0, 0.0, 0.0),
        settings=EngineSettings(1.0, 4.0, 4.0, 100.0, 100, 0),
        rng=np.random.default_rng(9),
    )
    assert result.final_state.status == ParticleStatus.FLOW_DOMAIN_EXIT
    assert np.isclose(result.final_state.x_m, -5.0)
    assert result.events[-1].attributes["boundary_locator"] == ("reference_drift_after_rk_stage_invalid")


def test_fraction_zero_terminal_event_replaces_same_time_active_observation() -> None:
    """恰抵 outer boundary 後下一步 fraction=0 停止，不得留下零秒 active→terminal 區段。"""

    geometry = BoundaryGeometry(
        own_local_domain=box(-2.0, -5.0, 2.0, 5.0),
        flow_domain=box(-4.0, -5.0, 4.0, 5.0),
        foreign_local_domains={},
    )

    def velocity(x_m: float, y_m: float, z_m: float, time_utc_ns: int) -> VelocitySample:
        """固定東流讓每個 4 秒步恰好抵達 -4 m outer boundary。"""

        del x_m, y_m, z_m, time_utc_ns
        return VelocitySample(1.0, 0.0, 0.0, 0.0, -10.0, 100.0, 1.0)

    result = run_particle(
        _state(0.0),
        velocity=velocity,
        boundaries=geometry,
        behavior_class="suspended",
        diffusion=DiffusionCoefficients(0.0, 0.0, 0.0),
        settings=EngineSettings(4.0, 4.0, 4.0, 100.0, 100, 0),
        rng=np.random.default_rng(9),
    )
    assert result.final_state.status == ParticleStatus.FLOW_DOMAIN_EXIT
    assert all(
        second.age_seconds > first.age_seconds
        for first, second in zip(result.observations[:-1], result.observations[1:], strict=True)
    )
    assert result.observations[-1].status == ParticleStatus.FLOW_DOMAIN_EXIT
