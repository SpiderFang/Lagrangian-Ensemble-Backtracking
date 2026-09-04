"""以純 NumPy 實作的粒子回溯基準引擎、事件紀錄與停止規則。

本引擎優先確保每一項科學規則都可檢查，並非用來追求最高運算速度。每一步會依當時
可用資料決定合適的步長，以四階 Runge-Kutta 法計算流速移動，再加上一次隨機擴散，最後
判定海面、海床、海岸與各研究範圍的穿越事件。速度資料缺漏、超出資料時間範圍或空間
定位失敗，都會以不同停止狀態保留下來。日後加速版本必須逐項得到相同結果，不能另訂
一套物理規則。Observation 的環境欄位只記錄步首樣本能證明的海面、海床、forcing
月份與品質狀態：``z_m`` 採海面向上為正、長度採公尺、月份採 UTC 的 ``YYYYMM``。
缺值與品質失敗不能以零值冒充有效環境；本機 synthetic callback 的結果也只是工程測試
證據，不是正式 OCM／NWW3 海洋科學成果。
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np

from .boundaries import BoundaryGeometry, resolve_horizontal_boundaries, resolve_vertical_boundaries
from .diffusion import DiffusionCoefficients, DiffusionModel, choose_time_step, resolve_diffusion_sample
from .integrators import SamplingContext, SamplingError, VelocityProvider, split_rk4_brownian_step
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


class EnvironmentSampleStatus(StrEnum):
    """Observation 對步首環境樣本可追溯程度的固定狀態。

    ``NOT_SAMPLED`` 表示目前 observation 尚未由對應步首樣本 enrichment；``VALID`` 表示
    具有有限海面／海床、公尺制正確上下界、合法 UTC forcing 月份及 ``qc=0``；``INVALID``
    表示樣本失敗但保存了非零品質旗標與可用的部分上下文。這些狀態不改變粒子速度或
    Brownian 計算，也不把本機合成資料提升成 OCM／NWW3 正式證據。
    """

    NOT_SAMPLED = "not_sampled"
    VALID = "valid"
    INVALID = "invalid"


_ENVIRONMENT_GEOMETRY_TOLERANCE_M = 1.0e-6
_MAX_ENVIRONMENT_QC_FLAGS = (1 << 32) - 1
_YYYYMM_PATTERN = re.compile(r"^[0-9]{6}$")
EnvironmentContext = tuple[
    EnvironmentSampleStatus,
    float | None,
    float | None,
    str | None,
    int | None,
]


def _canonical_environment_float(value: object, *, label: str) -> float:
    """把環境高度轉成原生有限 ``float``，拒絕 bool、NumPy scalar 與非有限值。

    海面 ``eta_m`` 與海床 ``bed_z_m`` 都是以公尺、海面向上為正的垂向座標；只接受
    Python 原生 ``int``／``float`` 可避免不同數值 scalar 在 checkpoint、Arrow 或 JSON
    邊界產生不一致的隱式轉換。這裡不接受缺值；無效樣本的缺失上下文由專用 normalizer
    在呼叫 constructor 前明確轉成 ``None``。
    """

    if type(value) not in (int, float):
        raise TypeError(f"{label} 必須是原生有限數值")
    try:
        normalized = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{label} 必須是有限數值") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{label} 不可為非有限數值")
    return normalized


def _canonical_invalid_environment_float(value: object, *, label: str) -> float | None:
    """整理無效樣本的 optional 高度；非有限數值明確轉成缺值 ``None``。

    無效樣本的核心語意是「取樣失敗」，所以 provider 回傳的 NaN／無限值只能記成缺值，
    不能寫入 Observation；若是有限值仍須通過與 constructor 相同的原生型別限制，避免
    把任意可轉換物件誤當成可稽核的公尺數值。
    """

    if value is None:
        return None
    if type(value) is bool:
        raise TypeError(f"{label} 不可為 bool")
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise TypeError(f"{label} 必須是 None 或原生有限數值") from error
    if not math.isfinite(normalized):
        return None
    if type(value) not in (int, float):
        raise TypeError(f"{label} 必須是 None 或原生有限數值")
    return normalized


def _canonical_forcing_month(value: object, *, label: str) -> str:
    """驗證 forcing 月份為真正的四位年／兩位月 ``YYYYMM`` 字串。

    月份是 UTC forcing 的資料識別，不是可由鄰月推估的數值；因此要求 ASCII 六位數、
    年份 1--9999 且月份 01--12，拒絕空白、浮點數、非 ASCII 數字與不存在的月份。
    """

    if type(value) is not str or _YYYYMM_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} 必須是合法 YYYYMM")
    year = int(value[:4])
    month = int(value[4:])
    if not 1 <= year <= 9999 or not 1 <= month <= 12:
        raise ValueError(f"{label} 必須是合法 YYYYMM")
    return value


def _observation_matches_state(observation: Observation, state: ParticleState) -> bool:
    """確認 observation 是否正好描述目前 state，才允許綁定步首環境樣本。

    這裡刻意使用 exact identity／數值比較：enrichment 是稽核欄位寫入，不是重新取樣或
    插值，不能把某個較早輸出點的海況錯掛到已移動的粒子。status 可以不同，因為 max age、
    forcing start 與 finalize 會在同一 state 只更新生命週期狀態。
    """

    return (
        observation.particle_id == state.particle_id
        and observation.time_utc_ns == state.time_utc_ns
        and observation.age_seconds == state.age_seconds
        and observation.x_m == state.x_m
        and observation.y_m == state.y_m
        and observation.z_m == state.z_m
    )


def _validate_observation_environment(
    *,
    status: EnvironmentSampleStatus,
    eta_m: float | None,
    bed_z_m: float | None,
    forcing_month_id: str | None,
    environment_qc_flags: int | None,
    z_m: object,
) -> tuple[float | None, float | None, str | None, int | None]:
    """驗證並 canonicalize Observation 的四個環境 context 欄位。

    有效樣本必須以公尺制有限上下界包住目前的 ``z_m``，並且以合法月份證明資料來源；
    無效樣本保留非零品質旗標，允許上下界或月份缺失，但不以 ``0`` 表示缺值。回傳內容
    全部是原生 immutable values，供 frozen Observation 直接保存。
    """

    if type(status) is not EnvironmentSampleStatus:
        raise TypeError("environment_sample_status 必須是 EnvironmentSampleStatus")

    if status is EnvironmentSampleStatus.NOT_SAMPLED:
        if any(value is not None for value in (eta_m, bed_z_m, forcing_month_id, environment_qc_flags)):
            raise ValueError("not_sampled 的環境 context 必須全部是 None")
        return None, None, None, None

    if status is EnvironmentSampleStatus.VALID:
        normalized_eta = _canonical_environment_float(eta_m, label="eta_m")
        normalized_bed = _canonical_environment_float(bed_z_m, label="bed_z_m")
        normalized_month = _canonical_forcing_month(forcing_month_id, label="forcing_month_id")
        if type(environment_qc_flags) is not int or environment_qc_flags != 0:
            raise ValueError("valid 的 environment_qc_flags 必須是原生整數 0")
        try:
            normalized_z = float(z_m)
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError("valid context 的 z_m 必須是有限數值") from error
        if not math.isfinite(normalized_z):
            raise ValueError("valid context 的 z_m 必須是有限數值")
        if (
            normalized_bed > normalized_z + _ENVIRONMENT_GEOMETRY_TOLERANCE_M
            or normalized_z > normalized_eta + _ENVIRONMENT_GEOMETRY_TOLERANCE_M
            or normalized_bed > normalized_eta + _ENVIRONMENT_GEOMETRY_TOLERANCE_M
        ):
            raise ValueError("valid context 的 bed_z_m、z_m、eta_m 垂向範圍不相容")
        return normalized_eta, normalized_bed, normalized_month, 0

    if status is EnvironmentSampleStatus.INVALID:
        if type(environment_qc_flags) is not int:
            raise TypeError("invalid 的 environment_qc_flags 必須是原生整數")
        if not 1 <= environment_qc_flags <= _MAX_ENVIRONMENT_QC_FLAGS:
            raise ValueError("invalid 的 environment_qc_flags 必須介於 1 與 2^32-1")
        normalized_eta = (
            None
            if eta_m is None
            else _canonical_environment_float(eta_m, label="eta_m")
        )
        normalized_bed = (
            None
            if bed_z_m is None
            else _canonical_environment_float(bed_z_m, label="bed_z_m")
        )
        normalized_month = (
            None
            if forcing_month_id is None
            else _canonical_forcing_month(forcing_month_id, label="forcing_month_id")
        )
        return normalized_eta, normalized_bed, normalized_month, environment_qc_flags

    # Exact enum type check above already makes this unreachable, but keeping a final gate makes
    # future enum extensions fail closed instead of silently receiving a different schema.
    raise ValueError("未知 environment_sample_status")


@dataclass(frozen=True, slots=True)
class Observation:
    """不等長軌跡中的一筆固定時間間隔位置與環境 context 紀錄。

    位置 ``x_m``、``y_m``、``z_m`` 使用公尺，``z_m`` 以海面向上為正；時間使用 UTC
    奈秒，``age_seconds`` 使用秒。新增的四個環境欄位只描述同一 observation 時刻的步首
    sample：有效時上下界必須包住粒子深度，無效時保留非零品質旗標，尚未取樣時全部為
    ``None``。缺值不能以 0 混淆；即使 synthetic callback 通過工程測試，也不代表真實
    OCM／NWW3 科學資料或正式來源足跡。
    """

    particle_id: str
    time_utc_ns: int
    age_seconds: float
    x_m: float
    y_m: float
    z_m: float
    status: ParticleStatus
    environment_sample_status: EnvironmentSampleStatus = EnvironmentSampleStatus.NOT_SAMPLED
    eta_m: float | None = None
    bed_z_m: float | None = None
    forcing_month_id: str | None = None
    environment_qc_flags: int | None = None

    def __post_init__(self) -> None:
        """在建立 immutable observation 時驗證 context 並寫入原生 canonical 值。"""

        eta_m, bed_z_m, forcing_month_id, environment_qc_flags = _validate_observation_environment(
            status=self.environment_sample_status,
            eta_m=self.eta_m,
            bed_z_m=self.bed_z_m,
            forcing_month_id=self.forcing_month_id,
            environment_qc_flags=self.environment_qc_flags,
            z_m=self.z_m,
        )
        object.__setattr__(self, "eta_m", eta_m)
        object.__setattr__(self, "bed_z_m", bed_z_m)
        object.__setattr__(self, "forcing_month_id", forcing_month_id)
        object.__setattr__(self, "environment_qc_flags", environment_qc_flags)


@dataclass(slots=True)
class ParticleResult:
    """一個隨機系集成員的最終狀態、軌跡紀錄、事件與計算步數。"""

    final_state: ParticleState
    observations: list[Observation]
    events: list[BoundaryEvent]
    step_count: int
    minimum_clamp_count: int


@dataclass(slots=True)
class ParticleExecutionState:
    """可暫停、可恢復的一條粒子軌跡執行狀態。

    ``state`` 是目前唯一的粒子狀態；``observations`` 與 ``events`` 依原始 reference
    engine 的寫入順序保存，不能只保存最後位置，否則重啟後會遺失固定輸出點與邊界事件。
    ``step_count`` 是已完成的完整步數，``minimum_clamp_count`` 是時間步長被迫低於設定
    最小值的累計次數，``next_output_age_seconds`` 則是下一個應寫入固定觀測的回溯年齡，
    單位都是秒。這個資料類別不包含 forcing、幾何或亂數產生器；後三者由批次 runtime
    分開持有，才能在 checkpoint 中分別驗證輸入繫結與 RNG continuation。
    """

    state: ParticleState
    observations: list[Observation]
    events: list[BoundaryEvent]
    step_count: int
    minimum_clamp_count: int
    next_output_age_seconds: float

    @property
    def current(self) -> ParticleState:
        """提供語意化別名，讓呼叫端可用 ``current`` 讀取目前粒子狀態。"""

        return self.state

    @current.setter
    def current(self, value: ParticleState) -> None:
        """允許中途恢復程式以 ``current`` 名稱替換目前不可變粒子狀態。"""

        if not isinstance(value, ParticleState):
            raise TypeError("current 必須是 ParticleState")
        self.state = value

    @property
    def terminal(self) -> bool:
        """目前粒子是否已進入任何停止狀態。"""

        return self.state.status != ParticleStatus.ACTIVE


@dataclass(frozen=True, slots=True)
class ParticleAdvanceResult:
    """一次單步呼叫的結果，兼容布林式的 terminal 判斷。

    ``stepped`` 表示本次是否真的完成一個數值步；步首遇到 max-step、max-age、forcing
    start 或無效速度時為 ``False``。``terminal`` 表示呼叫結束時是否已停止。實作
    ``__bool__`` 是為了讓既有批次迴圈可直接寫成 ``while not advance_result``，同時保留
    ``state`` 與明確欄位給需要稽核步數的呼叫端。
    """

    terminal: bool
    stepped: bool
    state: ParticleState

    def __bool__(self) -> bool:
        """將結果轉成「是否終止」而不是「是否有前進」的布林值。"""

        return self.terminal


def _observation(
    state: ParticleState,
    *,
    preserve_context: Observation | None = None,
    environment_context: EnvironmentContext | None = None,
) -> Observation:
    """將目前粒子狀態整理成 observation，並在同一 state 時保留環境 context。

    新的時間／年齡或位置一定建立 ``NOT_SAMPLED`` observation；只有 max age、forcing
    start、finalize 等同一 state 的 status-only 更新，才可從上一筆完全相同的位置複製
    context。若呼叫端明確提供 ``environment_context``，則它代表目前步首 reference
    對同一 state 的取樣結果，優先於既有 context；這只用於無法與既有固定 observation
    對齊的 invalid terminal，避免遺失實際失敗品質旗標。這個界線避免把步首樣本誤掛到
    步末 boundary terminal，也讓中途恢復的有效 context 不會因單純改寫停止狀態而降級。
    """

    context: dict[str, object] = {}
    if environment_context is not None:
        (
            environment_sample_status,
            eta_m,
            bed_z_m,
            forcing_month_id,
            environment_qc_flags,
        ) = environment_context
        context = {
            "environment_sample_status": environment_sample_status,
            "eta_m": eta_m,
            "bed_z_m": bed_z_m,
            "forcing_month_id": forcing_month_id,
            "environment_qc_flags": environment_qc_flags,
        }
    elif preserve_context is not None and _observation_matches_state(preserve_context, state):
        context = {
            "environment_sample_status": preserve_context.environment_sample_status,
            "eta_m": preserve_context.eta_m,
            "bed_z_m": preserve_context.bed_z_m,
            "forcing_month_id": preserve_context.forcing_month_id,
            "environment_qc_flags": preserve_context.environment_qc_flags,
        }

    return Observation(
        particle_id=state.particle_id,
        time_utc_ns=state.time_utc_ns,
        age_seconds=state.age_seconds,
        x_m=state.x_m,
        y_m=state.y_m,
        z_m=state.z_m,
        status=state.status,
        **context,
    )


def _append_or_replace_observation(
    observations: list[Observation],
    state: ParticleState,
    *,
    environment_context: EnvironmentContext | None = None,
) -> None:
    """寫入軌跡紀錄；若時間與位置未變，只更新為較新的狀態。

    粒子剛好停在多邊形邊界時，下一步可能立刻判定為停止。若同一時刻同一位置同時保留
    「仍在計算」與「已停止」兩筆資料，後續計算停留時間會產生長度為零的假區段。因此
    完全相同的時間與追蹤年齡只保留較新的狀態；不同時刻則正常新增資料。明確傳入的
    ``environment_context`` 只代表目前 state 的步首樣本，供 invalid step-start 在上一個
    固定輸出點尚未更新時直接寫入 terminal observation；一般 status-only 更新仍沿用前一筆
    完全相同 state 的 context。
    """

    previous = observations[-1] if observations else None
    observation = _observation(
        state,
        preserve_context=previous,
        environment_context=environment_context,
    )
    if observations and (
        observations[-1].time_utc_ns == observation.time_utc_ns
        and np.isclose(observations[-1].age_seconds, observation.age_seconds, rtol=0.0, atol=1.0e-12)
    ):
        observations[-1] = observation
    else:
        observations.append(observation)


def _reference_environment_context(
    *,
    reference: VelocitySample,
    state: ParticleState,
) -> EnvironmentContext | None:
    """把同一次步首 ``VelocitySample`` 轉成可寫入 Observation 的環境 context。

    valid sample 只有在 provider 明示合法 UTC 月份時才可成為可追溯環境證據；月份為
    ``None`` 的 synthetic callback 代表沒有正式 forcing provenance，故不假造 VALID。
    invalid sample 則保留非零 ``qc``，並把 NaN／無限上下界明確降為缺值，不把它們當成
    物理零值。這個函式不取樣、不改 RNG，且不負責判斷 observation 是否仍位於步首。
    """

    if reference.valid:
        if reference.forcing_month_id is None:
            return None
        # 有月份即宣稱有可追溯 forcing 來源；任一環境欄位不合法都直接拒絕本次執行，
        # 不能降級成看似正常的 NOT_SAMPLED，否則正式流程會吞掉 provenance 損壞。
        eta_m = _canonical_environment_float(reference.eta_m, label="reference.eta_m")
        bed_z_m = _canonical_environment_float(reference.bed_z_m, label="reference.bed_z_m")
        forcing_month_id = _canonical_forcing_month(
            reference.forcing_month_id, label="reference.forcing_month_id"
        )
        _validate_observation_environment(
            status=EnvironmentSampleStatus.VALID,
            eta_m=eta_m,
            bed_z_m=bed_z_m,
            forcing_month_id=forcing_month_id,
            environment_qc_flags=0,
            z_m=state.z_m,
        )
        return EnvironmentSampleStatus.VALID, eta_m, bed_z_m, forcing_month_id, 0

    if type(reference.qc) is bool:
        raise TypeError("invalid reference 的 qc 不可為 bool")
    if not isinstance(reference.qc, (int, SampleQC)):
        raise TypeError("invalid reference 的 qc 必須是整數旗標")
    try:
        qc_flags = int(reference.qc)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError("invalid reference 的 qc 必須可轉成原生整數") from error
    if not 1 <= qc_flags <= _MAX_ENVIRONMENT_QC_FLAGS:
        raise ValueError("invalid reference 的 qc 必須介於 1 與 2^32-1")
    eta_m = _canonical_invalid_environment_float(reference.eta_m, label="reference.eta_m")
    bed_z_m = _canonical_invalid_environment_float(reference.bed_z_m, label="reference.bed_z_m")
    forcing_month_id = (
        None
        if reference.forcing_month_id is None
        else _canonical_forcing_month(
            reference.forcing_month_id, label="reference.forcing_month_id"
        )
    )
    return EnvironmentSampleStatus.INVALID, eta_m, bed_z_m, forcing_month_id, qc_flags


def _enrich_latest_observation_from_reference(
    observations: list[Observation],
    state: ParticleState,
    reference: VelocitySample,
) -> EnvironmentContext | None:
    """以既有步首 sample enrichment 最新且完全同 state 的 observation。

    enrichment 只替換 list 中最後一筆 immutable Observation，不建立額外觀測、不呼叫
    velocity，也不使用亂數。若最新 observation 已是較早的時間／位置，樣本不能跨觀測點
    移植；若 valid sample 沒有 forcing 月份，則維持原有 context，明確保留 synthetic 與
    正式 OCM／NWW3 可追溯證據的界線。
    """

    context = _reference_environment_context(reference=reference, state=state)
    if context is None or not observations:
        return context
    latest = observations[-1]
    if not _observation_matches_state(latest, state):
        return context
    (
        environment_sample_status,
        eta_m,
        bed_z_m,
        forcing_month_id,
        environment_qc_flags,
    ) = context
    observations[-1] = replace(
        latest,
        environment_sample_status=environment_sample_status,
        eta_m=eta_m,
        bed_z_m=bed_z_m,
        forcing_month_id=forcing_month_id,
        environment_qc_flags=environment_qc_flags,
    )
    return context


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


# 終止診斷版本只規範事件補充欄位，不變更軌跡或中途續跑檔案格式。版本 1 的缺值政策是
# 「省略數值並提供可用性旗標」；不可寫入 None、NaN、無限值，也不可用 0 代替未知值。
_FAILURE_DIAGNOSTIC_VERSION = 1
# 階段與原因採固定白名單，防止外部例外文字、檔案路徑或任意樣本字典流入公開事件。
_FAILURE_STAGES = frozenset({"k1", "k2", "k3", "k4", "step_start", "diffusion", "limits", "unknown"})
_FAILURE_REASONS = frozenset({
    "invalid_velocity_sample", "invalid_diffusion_sample", "diffusion_evaluation_error",
    "rk_stage_unrecoverable", "maximum_step_count", "minimum_clamp_limit", "unknown",
})


def _diagnostic_float(value: object) -> float | None:
    """只接受已知數值型別並轉成有限原生浮點數，不轉換任意物件或文字。

    NumPy 的浮點／整數純量來自中間座標陣列，可轉為 JSON 支援的原生數值；布林值、
    文字、未知物件與非有限數值都視為不可用，避免轉型副作用或假造物理零值。
    """

    if type(value) not in (int, float) and not isinstance(value, (np.integer, np.floating)):
        return None
    try:
        normalized = float(value)
    except (OverflowError, ValueError):
        return None
    return normalized if math.isfinite(normalized) else None


def _diagnostic_int(value: object) -> int | None:
    """只接受原生或 NumPy 整數；時間、計數與位元旗標不可由浮點數截斷猜測。"""

    if type(value) is int or isinstance(value, np.integer):
        return int(value)
    return None


def _failure_attributes(
    execution: ParticleExecutionState,
    settings: EngineSettings,
    *,
    reason: str,
    stage: str,
    qc: SampleQC | None = None,
    context: SamplingContext | None = None,
    attempted_dt_seconds: float | None = None,
) -> dict[str, bool | float | int | str]:
    """建立版本 1 的失敗診斷，只回傳既有事件格式允許的安全純量。

    原因與階段必須在白名單內，否則記為 ``unknown``；不保存例外文字或樣本任意字典。
    ``sample_x_m/y_m/z_m`` 是失敗查詢座標（公尺，向上為正），不是粒子的最後有效位置；
    ``sample_time_utc_ns`` 是該查詢的 UTC 奈秒。``sampling_context_available`` 表示這
    四欄全部可用；部分可用欄位仍保留。海面／海床只取自同次失敗樣本，各有獨立旗標。
    未知、型別不符或非有限數值省略，不寫入 null 或補零；這使舊讀取器仍能重讀事件。

    ``attempted_dt_seconds`` 帶積分方向，回溯為負；下限累計超限時表示已選但未執行的
    步長。步數、下限累計、上限及設定步長供重現停止閘門，不代表新的數值修正。最大
    步數／下限次數停止沒有失敗樣本，因此品質旗標與取樣欄位保持不可用，不假造 qc=0。
    """

    attributes: dict[str, bool | float | int | str] = {
        "diagnostic_version": _FAILURE_DIAGNOSTIC_VERSION,
        "failure_reason": reason if type(reason) is str and reason in _FAILURE_REASONS else "unknown",
        "failure_stage": stage if type(stage) is str and stage in _FAILURE_STAGES else "unknown",
    }
    # 所有數值皆逐欄轉成安全原生純量；不接受任意 mapping 擴充輸出欄位集合。
    for key, value in (
        ("step_count", execution.step_count),
        ("minimum_clamp_count", execution.minimum_clamp_count),
        ("maximum_step_count", settings.maximum_step_count),
        ("maximum_minimum_clamps", settings.maximum_minimum_clamps),
    ):
        normalized_int = _diagnostic_int(value)
        if normalized_int is not None:
            attributes[key] = normalized_int
    for key, value in (
        ("dt_min_seconds", settings.dt_min_seconds),
        ("dt_max_seconds", settings.dt_max_seconds),
        ("attempted_dt_seconds", attempted_dt_seconds),
    ):
        normalized_float = _diagnostic_float(value)
        if normalized_float is not None:
            attributes[key] = normalized_float
    attributes["attempted_dt_available"] = "attempted_dt_seconds" in attributes
    normalized_qc = int(qc) if isinstance(qc, SampleQC) else _diagnostic_int(qc)
    attributes["qc_available"] = normalized_qc is not None and 0 <= normalized_qc <= (1 << 32) - 1
    if normalized_qc is not None and attributes["qc_available"]:
        attributes["qc_flags"] = normalized_qc

    if isinstance(context, SamplingContext):
        for key, value in (
            ("sample_x_m", context.x_m), ("sample_y_m", context.y_m),
            ("sample_z_m", context.z_m), ("sample_eta_m", context.eta_m),
            ("sample_bed_z_m", context.bed_z_m),
        ):
            normalized_float = _diagnostic_float(value)
            if normalized_float is not None:
                attributes[key] = normalized_float
        sample_time = _diagnostic_int(context.time_utc_ns)
        if sample_time is not None and -(1 << 63) <= sample_time < (1 << 63):
            attributes["sample_time_utc_ns"] = sample_time
    attributes["sampling_context_available"] = all(
        key in attributes for key in ("sample_x_m", "sample_y_m", "sample_z_m", "sample_time_utc_ns")
    )
    attributes["sample_eta_available"] = "sample_eta_m" in attributes
    attributes["sample_bed_available"] = "sample_bed_z_m" in attributes
    return attributes


def _validate_engine_settings(settings: EngineSettings) -> None:
    """集中保存 reference engine 原有的設定檢查，避免重啟入口漏掉相同閘門。

    目前只重複既有 ``run_particle`` 的正值與步長範圍規則；最大步數仍由步首比較處理，
    因為該值為零或較小時的既定語意是立即以數值失敗停止，而不是在初始化階段改變事件。
    """

    if settings.dt_min_seconds <= 0 or settings.dt_max_seconds < settings.dt_min_seconds:
        raise ValueError("engine dt 範圍無效")
    if settings.output_interval_seconds <= 0 or settings.max_backtrack_seconds <= 0:
        raise ValueError("output interval 與 max backtrack 必須為正")


def initialize_particle_execution(
    initial_state: ParticleState, settings: EngineSettings
) -> ParticleExecutionState:
    """驗證並建立一條粒子執行狀態，且立即寫入初始觀測。

    初始狀態必須是 ``ACTIVE``；位置使用公尺、時間使用 UTC 奈秒，第一筆觀測的
    ``age_seconds`` 直接沿用輸入狀態。這裡不取樣 forcing，也不消耗 RNG，因此從未啟動
    與從 checkpoint 還原的粒子可以共用同一個單步核心。``next_output_age_seconds`` 從
    一個完整輸出間隔開始，與原始 while engine 的固定輸出語意一致。
    """

    _validate_engine_settings(settings)
    if initial_state.status != ParticleStatus.ACTIVE:
        raise ValueError("initial_state 必須是 ACTIVE")
    return ParticleExecutionState(
        state=initial_state,
        observations=[_observation(initial_state)],
        events=[],
        step_count=0,
        minimum_clamp_count=0,
        next_output_age_seconds=settings.output_interval_seconds,
    )


def _terminate_execution(
    execution: ParticleExecutionState,
    *,
    status: ParticleStatus,
    event_type: EventType,
    environment_context: EnvironmentContext | None = None,
    failure_attributes: dict[str, bool | float | int | str] | None = None,
) -> ParticleAdvanceResult:
    """依既有停止順序更新狀態、事件及終點觀測。

    ``environment_context`` 僅由 invalid step-start 傳入，且一定來自同一次已完成的
    reference sample。它讓 terminal observation 在最新固定 observation 尚未到達目前
    state 時仍保留失敗的品質旗標；沒有明確 context 時，仍遵守 status-only replace 的
    原有 context preservation 規則。
    ``failure_attributes`` 只能由 ``_failure_attributes`` 的白名單整理器建立，且只附加到
    終止事件，不改變粒子位置、觀測環境或物理狀態。未提供時保留既有空事件屬性。
    """

    execution.state = replace(execution.state, status=status)
    event = _terminal_event(execution.state, event_type)
    if failure_attributes is not None:
        event = replace(event, attributes=dict(failure_attributes))
    execution.events.append(event)
    _append_or_replace_observation(
        execution.observations,
        execution.state,
        environment_context=environment_context,
    )
    return ParticleAdvanceResult(terminal=True, stepped=False, state=execution.state)


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


def advance_particle_once(
    execution: ParticleExecutionState,
    *,
    velocity: VelocityProvider,
    boundaries: BoundaryGeometry,
    behavior_class: str,
    diffusion: DiffusionModel,
    settings: EngineSettings,
    rng: np.random.Generator,
    on_step: Callable[[ParticleState], None] | None = None,
) -> ParticleAdvanceResult:
    """最多完成原始粒子 while 迴圈的一次迭代，並回傳是否已終止。

    每次呼叫先執行原本的步首 max-step、max-age、forcing-start 與速度有效性判定；取得
    有效步首 reference 後，空間擴散 provider 只以同一個公尺制位置、UTC 奈秒與
    ``reference.triangle_id`` 取樣一次。若可前進，則依同一個 ``DiffusionSample`` 完成
    步長選擇、四階 Runge-Kutta stage、一次 ``+div(K)|dt|`` 加 Brownian operator split、
    垂向邊界、水平邊界、minimum clamp 計數與輸出觀測順序；固定係數仍走舊版零梯度
    Brownian 路徑。步驟方向仍由 ``split_rk4_brownian_step`` 收到負秒數決定，位置與尺度
    使用公尺、年齡使用秒、時間使用 UTC 奈秒。無效擴散樣本在 choose-time-step、RK4
    與 RNG 之前依既有 QC 終止。終止狀態會立即寫入最後觀測，因此呼叫端可以在每次
    sweep 後安全 checkpoint；``on_step`` 只在真正完成數值步時呼叫一次。
    失敗事件另附版本化的安全診斷；不增加取樣或改變步長、亂數、重試及邊界恢復政策。
    """

    if execution.terminal:
        return ParticleAdvanceResult(terminal=True, stepped=False, state=execution.state)
    _validate_engine_settings(settings)
    # 常數係數仍在既有入口先驗證，維持舊版對非法固定 K 的 fail-fast 行為；空間 provider
    # 必須等步首速度與 triangle ID 都取得後才能取樣，因此不能在這裡提前呼叫 provider。
    if isinstance(diffusion, DiffusionCoefficients):
        diffusion.validate()
    state = execution.state
    if execution.step_count >= settings.maximum_step_count:
        return _terminate_execution(
            execution,
            status=ParticleStatus.NUMERICAL_FAILURE,
            event_type=EventType.NUMERICAL_FAILURE,
            failure_attributes=_failure_attributes(
                execution, settings, reason="maximum_step_count", stage="limits",
            ),
        )
    remaining_age = settings.max_backtrack_seconds - state.age_seconds
    seconds_to_start = (state.time_utc_ns - settings.earliest_forcing_time_utc_ns) / 1_000_000_000
    if remaining_age <= 1e-12:
        return _terminate_execution(
            execution,
            status=ParticleStatus.MAX_AGE,
            event_type=EventType.MAX_AGE,
        )
    if seconds_to_start <= 1e-12:
        return _terminate_execution(
            execution,
            status=ParticleStatus.FORCING_START,
            event_type=EventType.FORCING_START,
        )
    reference = velocity(state.x_m, state.y_m, state.z_m, state.time_utc_ns)
    # reference 已是本步唯一的步首 sample；環境欄位只 enrichment 同一 state 的最新
    # observation，不另取樣或消耗 RNG，因此不會改變 RK4、Brownian 與輸出 cadence。
    environment_context = _enrich_latest_observation_from_reference(
        execution.observations, state, reference
    )
    if not reference.valid:
        error = SamplingError(
            "step_start", reference.qc,
            context=SamplingContext(
                state.x_m, state.y_m, state.z_m, state.time_utc_ns,
                reference.eta_m, reference.bed_z_m,
            ),
        )
        status, event_type = _status_from_sampling_error(error)
        return _terminate_execution(
            execution,
            status=status,
            event_type=event_type,
            environment_context=environment_context,
            failure_attributes=_failure_attributes(
                execution, settings, reason="invalid_velocity_sample", stage=error.stage,
                qc=error.qc, context=error.context,
            ),
        )
    try:
        # 擴散 provider 僅以步首狀態取樣一次；同一個 immutable sample 會同時供
        # choose_time_step 與 RK4 後 split 使用，避免步長與實際位移看到不同的 K。
        diffusion_sample = resolve_diffusion_sample(
            diffusion,
            x_m=state.x_m,
            y_m=state.y_m,
            z_m=state.z_m,
            time_utc_ns=state.time_utc_ns,
            triangle_hint=reference.triangle_id,
        )
        if not diffusion_sample.valid:
            # 擴散樣本沒有海面／海床欄位；不可把另一個速度樣本的上下界冒充成它的證據。
            raise SamplingError(
                "diffusion", diffusion_sample.qc,
                context=SamplingContext(state.x_m, state.y_m, state.z_m, state.time_utc_ns),
            )
    except SamplingError as error:
        status, event_type = _status_from_sampling_error(error)
        return _terminate_execution(
            execution,
            status=status,
            event_type=event_type,
            environment_context=environment_context,
            failure_attributes=_failure_attributes(
                execution, settings, reason="invalid_diffusion_sample", stage=error.stage,
                qc=error.qc, context=error.context,
            ),
        )
    except (TypeError, ValueError):
        # provider 回傳錯誤型別、形狀或 qc=OK 但數值不合法時，轉成既有數值失敗語意。
        # 此分支發生在 choose_time_step 與 RNG 之前，故不會產生部分步驟或消耗亂數。
        return _terminate_execution(
            execution,
            status=ParticleStatus.NUMERICAL_FAILURE,
            event_type=EventType.NUMERICAL_FAILURE,
            environment_context=environment_context,
            failure_attributes=_failure_attributes(
                execution, settings, reason="diffusion_evaluation_error", stage="diffusion",
                context=SamplingContext(state.x_m, state.y_m, state.z_m, state.time_utc_ns),
            ),
        )
    horizontal_speed = float(np.hypot(reference.u_mps, reference.v_mps))
    decision = choose_time_step(
        speed_horizontal_mps=horizontal_speed,
        speed_vertical_mps=abs(reference.w_mps),
        horizontal_scale_m=reference.horizontal_scale_m,
        vertical_scale_m=reference.vertical_scale_m,
        coefficients=diffusion_sample.coefficients,
        dt_min_seconds=settings.dt_min_seconds,
        dt_max_seconds=min(settings.dt_max_seconds, remaining_age, seconds_to_start),
    )
    if decision.limiting_reason == "minimum_clamp":
        execution.minimum_clamp_count += 1
        if execution.minimum_clamp_count > settings.maximum_minimum_clamps:
            return _terminate_execution(
                execution,
                status=ParticleStatus.NUMERICAL_FAILURE,
                event_type=EventType.NUMERICAL_FAILURE,
                failure_attributes=_failure_attributes(
                    execution, settings, reason="minimum_clamp_limit", stage="limits",
                    attempted_dt_seconds=-decision.seconds,
                ),
            )
    recovered_terminal = False
    try:
        proposed = split_rk4_brownian_step(
            state,
            dt_seconds=-decision.seconds,
            velocity=velocity,
            coefficients=diffusion_sample,
            rng=rng,
        )
    except SamplingError as error:
        boundary_qc = SampleQC.OUTSIDE_HORIZONTAL_DOMAIN | SampleQC.DRY_FACE | SampleQC.VERTICAL_UNSUPPORTED
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
            return _terminate_execution(
                execution, status=status, event_type=event_type,
                failure_attributes=_failure_attributes(
                    execution, settings, reason="rk_stage_unrecoverable", stage=error.stage,
                    qc=error.qc, context=error.context, attempted_dt_seconds=-decision.seconds,
                ),
            )
        proposed, recovered_events = recovery
        execution.events.extend(recovered_events)
        recovered_terminal = True
    if not recovered_terminal:
        proposed, vertical_events = resolve_vertical_boundaries(
            state, proposed, reference_sample=reference, behavior_class=behavior_class
        )
        execution.events.extend(vertical_events)
        if proposed.status == ParticleStatus.ACTIVE:
            proposed, horizontal_events = resolve_horizontal_boundaries(state, proposed, boundaries)
            execution.events.extend(horizontal_events)
    execution.state = proposed
    execution.step_count += 1
    if on_step is not None:
        on_step(proposed)
    if (
        proposed.age_seconds + 1e-9 >= execution.next_output_age_seconds
        or proposed.status != ParticleStatus.ACTIVE
    ):
        _append_or_replace_observation(execution.observations, proposed)
        while execution.next_output_age_seconds <= proposed.age_seconds + 1e-9:
            execution.next_output_age_seconds += settings.output_interval_seconds
    return ParticleAdvanceResult(
        terminal=proposed.status != ParticleStatus.ACTIVE,
        stepped=True,
        state=proposed,
    )


def finalize_particle_execution(execution: ParticleExecutionState) -> ParticleResult:
    """把目前執行狀態整理成既有 ``ParticleResult``，並保證終點觀測存在。

    這個函式不會再取樣、不消耗 RNG，也不會推進 active 粒子；因此可用於完整運行或
    只想檢查中途快照的結果轉換。若終點與最後一筆觀測的時間／狀態不同，補上一筆；
    相同時間與年齡則沿用既有 replace 規則，避免產生零長度軌跡區段。
    """

    state = execution.state
    if (
        not execution.observations
        or execution.observations[-1].time_utc_ns != state.time_utc_ns
        or execution.observations[-1].status != state.status
    ):
        _append_or_replace_observation(execution.observations, state)
    return ParticleResult(
        final_state=state,
        observations=execution.observations,
        events=execution.events,
        step_count=execution.step_count,
        minimum_clamp_count=execution.minimum_clamp_count,
    )


def run_particle(
    initial_state: ParticleState,
    *,
    velocity: VelocityProvider,
    boundaries: BoundaryGeometry,
    behavior_class: str,
    diffusion: DiffusionModel,
    settings: EngineSettings,
    rng: np.random.Generator,
    on_step: Callable[[ParticleState], None] | None = None,
) -> ParticleResult:
    """從一個受體與到達時刻向過去回溯一個隨機系集成員，直到明確停止。

    本公開包裝器只負責建立 execution、重複呼叫 ``advance_particle_once`` 與整理結果；
    所有物理流程、事件順序和觀測規則都集中在可暫停單步核心。這讓 reference 單粒子
    結果與 CPU production batch、checkpoint restore 共用完全相同的科學實作。
    """

    # 固定係數維持既有 run_particle 入口的立即驗證；provider 的合法性由每一步的
    # step-start sample 決定，不能在 run 建立時先取一個與實際粒子狀態無關的樣本。
    if isinstance(diffusion, DiffusionCoefficients):
        diffusion.validate()
    execution = initialize_particle_execution(initial_state, settings)
    while not advance_particle_once(
        execution,
        velocity=velocity,
        boundaries=boundaries,
        behavior_class=behavior_class,
        diffusion=diffusion,
        settings=settings,
        rng=rng,
        on_step=on_step,
    ):
        pass
    return finalize_particle_execution(execution)
