"""純 NumPy 參考粒子引擎、事件紀錄與停止條件。

Reference engine 的目的先是建立可驗證語意，不追求最大吞吐。每步以有效 forcing sample
決定 adaptive dt，執行完整 RK4，再加 Brownian split，最後解析垂向及水平 crossing。
任何 forcing 缺值、時間域外或定位失敗都以獨立狀態停止；正式 Numba kernel 必須逐項
對照本結果，而不是另寫一套不同物理政策。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from .boundaries import BoundaryGeometry, resolve_horizontal_boundaries, resolve_vertical_boundaries
from .diffusion import DiffusionCoefficients, choose_time_step
from .integrators import SamplingError, VelocityProvider, split_rk4_brownian_step
from .models import BoundaryEvent, EventType, ParticleState, ParticleStatus, SampleQC, VelocitySample


@dataclass(frozen=True, slots=True)
class EngineSettings:
    """一個 experiment case 的數值與停止設定。"""

    dt_min_seconds: float
    dt_max_seconds: float
    output_interval_seconds: float
    max_backtrack_seconds: float
    maximum_step_count: int
    earliest_forcing_time_utc_ns: int
    maximum_minimum_clamps: int = 100


@dataclass(frozen=True, slots=True)
class Observation:
    """ragged trajectory 的單一固定輸出點。"""

    particle_id: str
    time_utc_ns: int
    age_seconds: float
    x_m: float
    y_m: float
    z_m: float
    status: ParticleStatus


@dataclass(slots=True)
class ParticleResult:
    """一個 member 的最終狀態、固定間隔 observations、事件與步數。"""

    final_state: ParticleState
    observations: list[Observation]
    events: list[BoundaryEvent]
    step_count: int
    minimum_clamp_count: int


def _observation(state: ParticleState) -> Observation:
    """由不可變 particle state 建立輕量輸出列。"""

    return Observation(
        particle_id=state.particle_id,
        time_utc_ns=state.time_utc_ns,
        age_seconds=state.age_seconds,
        x_m=state.x_m,
        y_m=state.y_m,
        z_m=state.z_m,
        status=state.status,
    )


def _append_or_replace_observation(observations: list[Observation], state: ParticleState) -> None:
    """寫入 observation；同一 UTC/age 的狀態轉換以較新狀態取代。

    粒子可能恰停在 polygon boundary，下一步以 fraction=0 觸發 terminal event。若保留
    active 與 terminal 兩列，會形成零秒區段並破壞 residence-time 契約；座標與時間完全
    相同時只更新狀態不會遺失路徑資訊。不同時間即正常 append。
    """

    observation = _observation(state)
    if observations and (
        observations[-1].time_utc_ns == observation.time_utc_ns
        and np.isclose(observations[-1].age_seconds, observation.age_seconds, rtol=0.0, atol=1.0e-12)
    ):
        observations[-1] = observation
    else:
        observations.append(observation)


def _terminal_event(state: ParticleState, event_type: EventType) -> BoundaryEvent:
    """為非幾何停止條件建立 fraction=1 的事件列。"""

    return BoundaryEvent(
        particle_id=state.particle_id,
        scenario_id=state.scenario_id,
        member_id=state.member_id,
        study_site_id=state.study_site_id,
        analysis_region_id=state.analysis_region_id,
        receptor_id=state.receptor_id,
        event_type=event_type,
        time_utc_ns=state.time_utc_ns,
        x_m=state.x_m,
        y_m=state.y_m,
        z_m=state.z_m,
        fraction=1.0,
    )


def _status_from_sampling_error(error: SamplingError) -> tuple[ParticleStatus, EventType]:
    """把 forcing/time QC 分到 data-gap 或 numerical failure，保留科學失敗率。"""

    data_bits = SampleQC.OUTSIDE_TIME_RANGE | SampleQC.TIME_GAP | SampleQC.WAVE_UNSUPPORTED
    if error.qc & data_bits:
        return ParticleStatus.DATA_GAP, EventType.DATA_GAP
    return ParticleStatus.NUMERICAL_FAILURE, EventType.NUMERICAL_FAILURE


def _recover_terminal_boundary_from_reference_drift(
    state: ParticleState,
    *,
    reference: VelocitySample,
    dt_seconds: float,
    boundaries: BoundaryGeometry,
    behavior_class: str,
) -> tuple[ParticleState, list[BoundaryEvent]] | None:
    """在 RK stage 因陸地／域外失效時，以步首 drift 定位可證明的終止 crossing。

    native mesh 不允許穿陸或域外取樣，因此 k2–k4 可能在步末 boundary resolver 之前先
    失效。此 fallback 只建立 signed Euler 線段並接受其最早事件確實為 coast、outer、
    deposition 或 surface-regime exit 的情況；只命中非終止 local event 時不接受，避免
    用一階近似代替一般 RK4。事件明示 locator，正式驗收必須以 dt-halving 量化差異。
    """

    signed_dt = -dt_seconds
    proposed = replace(
        state,
        x_m=state.x_m + signed_dt * reference.u_mps,
        y_m=state.y_m + signed_dt * reference.v_mps,
        z_m=state.z_m + signed_dt * reference.w_mps,
        time_utc_ns=state.time_utc_ns + int(round(signed_dt * 1_000_000_000)),
        age_seconds=state.age_seconds + dt_seconds,
    )
    proposed, vertical_events = resolve_vertical_boundaries(
        state,
        proposed,
        reference_sample=reference,
        behavior_class=behavior_class,
    )
    events = list(vertical_events)
    if proposed.status == ParticleStatus.ACTIVE:
        proposed, horizontal_events = resolve_horizontal_boundaries(state, proposed, boundaries)
        events.extend(horizontal_events)
    terminal_statuses = {
        ParticleStatus.COAST_CONTACT,
        ParticleStatus.FLOW_DOMAIN_EXIT,
        ParticleStatus.DEPOSITED,
        ParticleStatus.SURFACE_REGIME_EXIT,
    }
    if proposed.status not in terminal_statuses:
        return None
    diagnosed = [
        replace(
            event,
            attributes={
                **event.attributes,
                "boundary_locator": "reference_drift_after_rk_stage_invalid",
                "requires_dt_halving_validation": True,
            },
        )
        for event in events
    ]
    return proposed, diagnosed


def run_particle(
    initial_state: ParticleState,
    *,
    velocity: VelocityProvider,
    boundaries: BoundaryGeometry,
    behavior_class: str,
    diffusion: DiffusionCoefficients,
    settings: EngineSettings,
    rng: np.random.Generator,
    on_step: Callable[[ParticleState], None] | None = None,
) -> ParticleResult:
    """從 receptor/arrival 向過去執行一個 member 至明確停止狀態。

    ``on_step`` 只供 benchmark/監控，不能修改 state。observations 依回溯 age 間隔輸出，
    事件與最終停止點無論是否落在固定輸出時刻都保留。正式大批次會以 chunked vectorized
    engine 取代逐粒子迴圈，但必須通過與本函式相同的解析及 restart 測試。
    """

    diffusion.validate()
    if initial_state.status != ParticleStatus.ACTIVE:
        raise ValueError("initial_state 必須是 ACTIVE")
    if settings.dt_min_seconds <= 0 or settings.dt_max_seconds < settings.dt_min_seconds:
        raise ValueError("engine dt 範圍無效")
    if settings.output_interval_seconds <= 0 or settings.max_backtrack_seconds <= 0:
        raise ValueError("output interval 與 max backtrack 必須為正")
    state = initial_state
    observations = [_observation(state)]
    events: list[BoundaryEvent] = []
    next_output_age = settings.output_interval_seconds
    minimum_clamps = 0
    step_count = 0
    while state.status == ParticleStatus.ACTIVE:
        if step_count >= settings.maximum_step_count:
            state = replace(state, status=ParticleStatus.NUMERICAL_FAILURE)
            events.append(_terminal_event(state, EventType.NUMERICAL_FAILURE))
            break
        remaining_age = settings.max_backtrack_seconds - state.age_seconds
        seconds_to_start = (state.time_utc_ns - settings.earliest_forcing_time_utc_ns) / 1_000_000_000
        if remaining_age <= 1e-12:
            state = replace(state, status=ParticleStatus.MAX_AGE)
            events.append(_terminal_event(state, EventType.MAX_AGE))
            break
        if seconds_to_start <= 1e-12:
            state = replace(state, status=ParticleStatus.FORCING_START)
            events.append(_terminal_event(state, EventType.FORCING_START))
            break
        reference = velocity(state.x_m, state.y_m, state.z_m, state.time_utc_ns)
        if not reference.valid:
            error = SamplingError("step_start", reference.qc)
            status, event_type = _status_from_sampling_error(error)
            state = replace(state, status=status)
            events.append(_terminal_event(state, event_type))
            break
        horizontal_speed = float(np.hypot(reference.u_mps, reference.v_mps))
        decision = choose_time_step(
            speed_horizontal_mps=horizontal_speed,
            speed_vertical_mps=abs(reference.w_mps),
            horizontal_scale_m=reference.horizontal_scale_m,
            vertical_scale_m=reference.vertical_scale_m,
            coefficients=diffusion,
            dt_min_seconds=settings.dt_min_seconds,
            dt_max_seconds=min(settings.dt_max_seconds, remaining_age, seconds_to_start),
        )
        if decision.limiting_reason == "minimum_clamp":
            minimum_clamps += 1
            if minimum_clamps > settings.maximum_minimum_clamps:
                state = replace(state, status=ParticleStatus.NUMERICAL_FAILURE)
                events.append(_terminal_event(state, EventType.NUMERICAL_FAILURE))
                break
        recovered_terminal = False
        try:
            proposed = split_rk4_brownian_step(
                state,
                dt_seconds=-decision.seconds,
                velocity=velocity,
                coefficients=diffusion,
                rng=rng,
            )
        except SamplingError as error:
            boundary_qc = (
                SampleQC.OUTSIDE_HORIZONTAL_DOMAIN | SampleQC.DRY_FACE | SampleQC.VERTICAL_UNSUPPORTED
            )
            recovery = (
                _recover_terminal_boundary_from_reference_drift(
                    state,
                    reference=reference,
                    dt_seconds=decision.seconds,
                    boundaries=boundaries,
                    behavior_class=behavior_class,
                )
                if error.qc & boundary_qc
                else None
            )
            if recovery is None:
                status, event_type = _status_from_sampling_error(error)
                state = replace(state, status=status)
                events.append(_terminal_event(state, event_type))
                break
            proposed, recovered_events = recovery
            events.extend(recovered_events)
            recovered_terminal = True
        if not recovered_terminal:
            proposed, vertical_events = resolve_vertical_boundaries(
                state, proposed, reference_sample=reference, behavior_class=behavior_class
            )
            events.extend(vertical_events)
            if proposed.status == ParticleStatus.ACTIVE:
                proposed, horizontal_events = resolve_horizontal_boundaries(state, proposed, boundaries)
                events.extend(horizontal_events)
        state = proposed
        step_count += 1
        if on_step is not None:
            on_step(state)
        if state.age_seconds + 1e-9 >= next_output_age or state.status != ParticleStatus.ACTIVE:
            _append_or_replace_observation(observations, state)
            while next_output_age <= state.age_seconds + 1e-9:
                next_output_age += settings.output_interval_seconds
    if observations[-1].time_utc_ns != state.time_utc_ns or observations[-1].status != state.status:
        _append_or_replace_observation(observations, state)
    return ParticleResult(state, observations, events, step_count, minimum_clamps)
