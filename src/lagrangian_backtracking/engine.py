"""以純 NumPy 實作的粒子回溯基準引擎、事件紀錄與停止規則。

本引擎優先確保每一項科學規則都可檢查，並非用來追求最高運算速度。每一步會依當時
可用資料決定合適的步長，以四階龍格－庫塔法計算流速移動，再加上一次隨機擴散，最後
判定海面、海床、海岸與各研究範圍的穿越事件。速度資料缺漏、超出資料時間範圍或空間
定位失敗，都會以不同停止狀態保留下來。日後加速版本必須逐項得到相同結果，不能另訂
一套物理規則。
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
    """單一分析情境的時間步長、輸出頻率與停止上限。"""

    dt_min_seconds: float
    dt_max_seconds: float
    output_interval_seconds: float
    max_backtrack_seconds: float
    maximum_step_count: int
    earliest_forcing_time_utc_ns: int
    maximum_minimum_clamps: int = 100


@dataclass(frozen=True, slots=True)
class Observation:
    """不等長軌跡中的一筆固定時間間隔位置紀錄。"""

    particle_id: str
    time_utc_ns: int
    age_seconds: float
    x_m: float
    y_m: float
    z_m: float
    status: ParticleStatus


@dataclass(slots=True)
class ParticleResult:
    """一個隨機系集成員的最終狀態、軌跡紀錄、事件與計算步數。"""

    final_state: ParticleState
    observations: list[Observation]
    events: list[BoundaryEvent]
    step_count: int
    minimum_clamp_count: int


def _observation(state: ParticleState) -> Observation:
    """將目前粒子狀態整理成可寫入軌跡檔案的一筆資料。"""

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
    """寫入軌跡紀錄；若時間與位置未變，只更新為較新的狀態。

    粒子剛好停在多邊形邊界時，下一步可能立刻判定為停止。若同一時刻同一位置同時保留
    「仍在計算」與「已停止」兩筆資料，後續計算停留時間會產生長度為零的假區段。因此
    完全相同的時間與追蹤年齡只保留較新的狀態；不同時刻則正常新增資料。
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
    """為資料缺漏等非邊界停止原因建立事件紀錄。"""

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
    """依品質檢查結果區分資料缺口與數值計算失敗，保留可解釋的失敗比例。"""

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
    """中間計算點落到陸地或資料範圍外時，補做可驗證的邊界停止判定。

    原始網格不能在陸地或範圍外提供速度，因此四階計算的中間點可能先失敗，來不及走到
    正常的邊界判定。此處只用步首速度建立一條簡化直線，且僅在第一個碰到的確實是海岸、
    流場外、沉積或離開表層這類「必須停止」事件時才採用。若只是離開局部分析區，不使用
    這個簡化結果，以免取代一般的四階計算。事件會標記此判定來源，正式驗收時應以更小
    時間步長再次檢查差異。
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
    """從一個受體與到達時刻向過去回溯一個隨機系集成員，直到明確停止。

    ``on_step`` 只供效能量測或監看進度，不能修改粒子狀態。軌跡依設定的回溯時間間隔
    輸出；即使事件或最終停止點不在固定輸出時刻，仍會完整保留。日後大量正式計算可改用
    一次處理多粒子的加速引擎，但必須通過與此函式相同的解析案例與中途續跑測試。
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
