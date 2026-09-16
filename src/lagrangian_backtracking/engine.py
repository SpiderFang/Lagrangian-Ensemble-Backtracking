"""以純 NumPy 實作的粒子回溯基準引擎、事件紀錄與停止規則。

本引擎優先確保每一項科學規則都可檢查，並非用來追求最高運算速度。每一步會依當時
可用資料決定合適的步長，以四階 Runge-Kutta 法計算流速移動，再加上一次隨機擴散，最後
判定海面、海床、海岸與各研究範圍的穿越事件。速度資料缺漏、超出資料時間範圍或空間
定位失敗，都會以不同停止狀態保留下來。日後加速版本必須逐項得到相同結果，不能另訂
一套物理規則。Observation 的環境欄位只記錄步首樣本能證明的海面、海床、forcing
月份與品質狀態：``z_m`` 採海面向上為正、長度採公尺、月份採 UTC 的 ``YYYYMM``。
速度 observation 另保存同點步首 sample 的總量及 OCM／Stokes 水平／沉降分項；輸出
頻率是位置 observation cadence，不是 RK4 stage 或位移平均速度。缺值與品質失敗不能
以零值冒充有效環境或速度來源；本機 synthetic callback 的結果也只是工程測試證據，
不是正式 OCM／NWW3 海洋科學成果。
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np

from .accelerated import validate_physics_kernel_backend
from .boundaries import (
    BoundaryGeometry,
    recover_surface_boundary_at_step_start,
    resolve_horizontal_boundaries,
    resolve_vertical_boundaries,
)
from .diffusion import DiffusionCoefficients, DiffusionModel, choose_time_step, resolve_diffusion_sample
from .integrators import (
    SamplingContext,
    SamplingError,
    SurfaceStageVelocityProvider,
    VelocityProvider,
    split_rk4_brownian_step,
    supports_step_start_sample_reuse,
)
from .models import (
    SURFACE_BOUNDARY_TOLERANCE_M,
    VERTICAL_BOUNDARY_TOLERANCE_M,
    BoundaryEvent,
    EventType,
    ParticleState,
    ParticleStatus,
    SampleQC,
    VelocityComponents,
    VelocityQC,
    VelocitySample,
    VelocitySampleStatus,
)


@dataclass(frozen=True, slots=True)
class EngineSettings:
    """單一情境的時間步長、輸出頻率、停止上限與版本化純量 kernel 選擇。

    ``physics_kernel_backend`` 隨 request 的不可變設定一起傳入單步核心；既有
    手動建構與舊檢查點還原呼叫端若省略欄位，仍使用 ``numpy_v1``。選擇
    ``numba_cpu_v1`` 只替換可獨立計算的純量數值核心，不改事件、品質檢查旗標（QC）、
    RK4 階段查詢或每粒子的亂數來源。
    """

    dt_min_seconds: float
    dt_max_seconds: float
    output_interval_seconds: float
    max_backtrack_seconds: float
    maximum_step_count: int
    earliest_forcing_time_utc_ns: int
    maximum_minimum_clamps: int = 100
    physics_kernel_backend: str = "numpy_v1"


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


_MAX_ENVIRONMENT_QC_FLAGS = (1 << 32) - 1
_VELOCITY_SUM_RTOL = 1.0e-12
_VELOCITY_SUM_ATOL = 1.0e-12
_MAX_VELOCITY_QC_FLAGS = (1 << 32) - 1
_YYYYMM_PATTERN = re.compile(r"^[0-9]{6}$")
EnvironmentContext = tuple[
    EnvironmentSampleStatus,
    float | None,
    float | None,
    str | None,
    int | None,
]
VelocityObservationContext = tuple[
    VelocitySampleStatus,
    VelocityComponents | None,
    int | None,
]

_VELOCITY_COMPONENT_FIELDS = (
    "total_u_mps",
    "total_v_mps",
    "total_w_mps",
    "ocm_u_mps",
    "ocm_v_mps",
    "ocm_w_mps",
    "stokes_u_mps",
    "stokes_v_mps",
    "settling_w_mps",
)


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
            normalized_bed > normalized_z + VERTICAL_BOUNDARY_TOLERANCE_M
            or normalized_z > normalized_eta + SURFACE_BOUNDARY_TOLERANCE_M
            or normalized_bed > normalized_eta + VERTICAL_BOUNDARY_TOLERANCE_M
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


def _canonical_observation_velocity_float(value: object, *, label: str) -> float:
    """驗證 Observation 內的速度欄位為有限公尺/秒數值。

    Observation 是會進入 checkpoint、trajectory shard 與報告 consumer 的 immutable
    邊界，因此這裡接受一般 Python／NumPy 實數但拒絕 bool、文字、複數與 NaN／無限值。
    provider 的缺值在進入 Observation 前必須明確使用 ``None``，輸出 writer 才會將其
    編碼為 NaN sentinel；不把非有限數值直接留在記憶體，可避免不同 consumer 對缺值有
    不同解讀。
    """

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{label} 必須是有限公尺/秒數值")
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{label} 必須是有限公尺/秒數值") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{label} 不可為非有限數值")
    return normalized


def _validate_observation_velocity(
    *,
    status: VelocitySampleStatus,
    components: tuple[object, ...],
    velocity_qc_flags: int | None,
) -> tuple[tuple[float | None, ...], int | None]:
    """驗證 Observation 的九個速度欄位、狀態與獨立品質旗標。

    九欄固定順序是總速度 ``u/v/w``、OCM ``u/v/w``、Stokes 水平 ``u/v`` 及沉降
    ``w``，單位均為公尺/秒、方向均為正向物理方向。``COMPLETE`` 必須全部有限，並
    以 ``1e-12`` 相對／絕對容差驗證水平與垂向合成；``TOTAL_ONLY`` 只允許總速度。
    其他狀態可以保留部分有限診斷值，但必須有非零 velocity QC，不能把不完整資料當成
    有效完整分項。函式回傳原生 float／None，供 frozen ``Observation`` 寫回 canonical
    欄位。
    """

    if type(status) is not VelocitySampleStatus:
        raise TypeError("velocity_sample_status 必須是 VelocitySampleStatus")
    if len(components) != len(_VELOCITY_COMPONENT_FIELDS):
        raise ValueError("velocity 欄位數量不符")

    normalized: list[float | None] = []
    for value, field_name in zip(components, _VELOCITY_COMPONENT_FIELDS, strict=True):
        if value is None:
            normalized.append(None)
        else:
            normalized.append(
                _canonical_observation_velocity_float(value, label=field_name)
            )

    if status is VelocitySampleStatus.NOT_SAMPLED:
        if any(value is not None for value in normalized) or velocity_qc_flags is not None:
            raise ValueError("not_sampled 的速度 context 必須全部是 None")
        return tuple(normalized), None

    if type(velocity_qc_flags) is not int or not 0 <= velocity_qc_flags <= _MAX_VELOCITY_QC_FLAGS:
        raise TypeError("velocity_qc_flags 必須是 uint32 範圍內的原生整數")

    if status is VelocitySampleStatus.COMPLETE:
        if velocity_qc_flags != 0 or any(value is None for value in normalized):
            raise ValueError("complete 的速度欄位必須全部有限且 velocity_qc_flags=0")
        total_u, total_v, total_w, ocm_u, ocm_v, ocm_w, stokes_u, stokes_v, settling_w = normalized
        assert all(value is not None for value in normalized)
        if not math.isclose(
            total_u, ocm_u + stokes_u, rel_tol=_VELOCITY_SUM_RTOL, abs_tol=_VELOCITY_SUM_ATOL
        ) or not math.isclose(
            total_v, ocm_v + stokes_v, rel_tol=_VELOCITY_SUM_RTOL, abs_tol=_VELOCITY_SUM_ATOL
        ) or not math.isclose(
            total_w, ocm_w + settling_w, rel_tol=_VELOCITY_SUM_RTOL, abs_tol=_VELOCITY_SUM_ATOL
        ):
            raise ValueError("complete 的 total 與速度分項總和不符")
        return tuple(normalized), 0

    if status is VelocitySampleStatus.TOTAL_ONLY:
        if velocity_qc_flags != 0:
            raise ValueError("total_only 的 velocity_qc_flags 必須是 0")
        if any(value is None for value in normalized[:3]) or any(
            value is not None for value in normalized[3:]
        ):
            raise ValueError("total_only 只允許有限 total_u/v/w")
        return tuple(normalized), 0

    # INVALID、MISSING、NONFINITE 與 SUM_MISMATCH 都是「不完整或不可用」狀態；保留的
    # 有限欄位只能作診斷，非零 QC 才是下游判定不可用的明示依據。記憶體內仍不允許 NaN，
    # 缺值由 None 表示，輸出端才轉成 NaN。
    if velocity_qc_flags == 0:
        raise ValueError(f"{status.value} 的 velocity_qc_flags 必須非零")
    return tuple(normalized), velocity_qc_flags


def _velocity_context_kwargs(
    context: VelocityObservationContext,
) -> dict[str, object]:
    """將步首速度 context 展開成 Observation 固定欄位，不新增任何取樣。

    ``components=None`` 僅保留 ``NOT_SAMPLED`` 的全缺值語意；其餘狀態由 engine 先
    建立明示的 ``VelocityComponents``，再逐欄傳入 constructor 驗證。這個小 helper 集中
    維持九欄拓撲，避免 engine 的 enrichment、checkpoint reader 與 report clone 各自漏欄。
    """

    status, component_values, qc_flags = context
    values = (
        (None,) * len(_VELOCITY_COMPONENT_FIELDS)
        if component_values is None
        else tuple(getattr(component_values, name) for name in _VELOCITY_COMPONENT_FIELDS)
    )
    return {
        "velocity_sample_status": status,
        **dict(zip(_VELOCITY_COMPONENT_FIELDS, values, strict=True)),
        "velocity_qc_flags": qc_flags,
    }


@dataclass(frozen=True, slots=True)
class Observation:
    """不等長軌跡中的一筆固定輸出點位置、環境與速度 context 紀錄。

    位置 ``x_m``、``y_m``、``z_m`` 使用公尺，``z_m`` 以海面向上為正；時間使用 UTC
    奈秒，``age_seconds`` 使用秒。環境欄位及九個速度欄位都只描述同一 observation
    時刻能由步首 sample 證明的資料；速度總量與分項為公尺/秒、維持正向物理方向，
    不因逆向積分另行取負。速度 sample 是輸出觀測點的取樣值，不是由位移除以時間的
    平均速度，也不是 RK4 stage 或 internal step 的列表。缺值不能以 0 混淆；即使
    synthetic callback 通過工程測試，也不代表真實 OCM／NWW3 科學資料或正式來源足跡。
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
    velocity_sample_status: VelocitySampleStatus = VelocitySampleStatus.NOT_SAMPLED
    total_u_mps: float | None = None
    total_v_mps: float | None = None
    total_w_mps: float | None = None
    ocm_u_mps: float | None = None
    ocm_v_mps: float | None = None
    ocm_w_mps: float | None = None
    stokes_u_mps: float | None = None
    stokes_v_mps: float | None = None
    settling_w_mps: float | None = None
    velocity_qc_flags: int | None = None

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
        velocity_values, velocity_qc_flags = _validate_observation_velocity(
            status=self.velocity_sample_status,
            components=tuple(getattr(self, name) for name in _VELOCITY_COMPONENT_FIELDS),
            velocity_qc_flags=self.velocity_qc_flags,
        )
        for name, value in zip(_VELOCITY_COMPONENT_FIELDS, velocity_values, strict=True):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "velocity_qc_flags", velocity_qc_flags)


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
    velocity_context: VelocityObservationContext | None = None,
) -> Observation:
    """將目前粒子狀態整理成 observation，並在同一 state 時保留兩種 context。

    新的時間／年齡或位置一定建立 ``NOT_SAMPLED`` observation；只有 max age、forcing
    start、finalize 等同一 state 的 status-only 更新，才可從上一筆完全相同的位置複製
    environment／velocity context。若呼叫端明確提供任一 context，則它代表目前步首
    reference 對同一 state 的取樣結果，優先於既有 context；這只用於能證明同點的樣本，
    不把步末 boundary terminal 或 RK4 中間 sample 借掛到新位置。這個界線也讓中途恢復
    的有效 context 不會因單純改寫停止狀態而降級。
    """

    context: dict[str, object] = {}
    if preserve_context is not None and _observation_matches_state(preserve_context, state):
        context.update(
            {
                "environment_sample_status": preserve_context.environment_sample_status,
                "eta_m": preserve_context.eta_m,
                "bed_z_m": preserve_context.bed_z_m,
                "forcing_month_id": preserve_context.forcing_month_id,
                "environment_qc_flags": preserve_context.environment_qc_flags,
                "velocity_sample_status": preserve_context.velocity_sample_status,
                **{
                    name: getattr(preserve_context, name)
                    for name in _VELOCITY_COMPONENT_FIELDS
                },
                "velocity_qc_flags": preserve_context.velocity_qc_flags,
            }
        )
    if environment_context is not None:
        (
            environment_sample_status,
            eta_m,
            bed_z_m,
            forcing_month_id,
            environment_qc_flags,
        ) = environment_context
        context.update(
            {
                "environment_sample_status": environment_sample_status,
                "eta_m": eta_m,
                "bed_z_m": bed_z_m,
                "forcing_month_id": forcing_month_id,
                "environment_qc_flags": environment_qc_flags,
            }
        )
    if velocity_context is not None:
        context.update(_velocity_context_kwargs(velocity_context))

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
    velocity_context: VelocityObservationContext | None = None,
) -> None:
    """寫入軌跡紀錄；若時間與位置未變，只更新為較新的狀態。

    粒子剛好停在多邊形邊界時，下一步可能立刻判定為停止。若同一時刻同一位置同時保留
    「仍在計算」與「已停止」兩筆資料，後續計算停留時間會產生長度為零的假區段。因此
    完全相同的時間與追蹤年齡只保留較新的狀態；不同時刻則正常新增資料。明確傳入的
    ``environment_context``／``velocity_context`` 只代表目前 state 的步首樣本，供
    invalid step-start 在上一個固定輸出點尚未更新時直接寫入 terminal observation；一般
    status-only 更新仍沿用前一筆完全相同 state 的 context。步末 terminal 若沒有同點
    sample，兩種 context 都維持 ``NOT_SAMPLED`` 與 None。
    """

    previous = observations[-1] if observations else None
    observation = _observation(
        state,
        preserve_context=previous,
        environment_context=environment_context,
        velocity_context=velocity_context,
    )
    if observations and (
        observations[-1].time_utc_ns == observation.time_utc_ns
        and np.isclose(observations[-1].age_seconds, observation.age_seconds, rtol=0.0, atol=1.0e-12)
    ):
        observations[-1] = observation
    else:
        observations.append(observation)


def _provider_velocity_value(value: object) -> tuple[float | None, str]:
    """整理 provider 速度值並回傳 ``ok``／``missing``／``nonfinite``／``non_numeric``。

    速度 provider 可能來自 NumPy 內插或一般 synthetic callback；這裡允許可明確轉成
    實數的 Python／NumPy scalar，但拒絕 bool、文字、複數與無法轉換的物件。非有限值
    不會寫入 Observation，而由獨立 velocity status/QC 保存其原因。
    """

    if value is None:
        return None, "missing"
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        return None, "non_numeric"
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError):
        return None, "non_numeric"
    if not math.isfinite(normalized):
        return None, "nonfinite"
    return normalized, "ok"


def _reference_velocity_context(reference: VelocitySample) -> VelocityObservationContext:
    """把同一次步首 ``VelocitySample`` 轉成九欄速度 observation context。

    這個函式只讀取已回傳的 sample，不新增 forcing 查詢、不改呼叫順序、不消耗 RNG。
    ``CombinedMonthForcing`` 會提供完整分項；沒有分項的四參數 callback 則明示
    ``TOTAL_ONLY``，只保存有限總速度。樣本本身無效時不把 provider 可能帶有的 0 或部分
    數值升格成有效 total，而是以非零 ``SAMPLE_INVALID`` QC 及全缺值保存失敗語意。
    """

    if not isinstance(reference, VelocitySample):
        raise TypeError("reference 必須是 VelocitySample")
    # ``VelocitySample.valid`` 的既有判定維持原樣供積分使用；這裡額外防止 bool qc=0
    # 被誤當成正式速度紀錄的有效品質旗標。
    if type(reference.qc) is bool or not reference.valid:
        return (
            VelocitySampleStatus.INVALID,
            VelocityComponents(),
            int(VelocityQC.SAMPLE_INVALID),
        )

    total_values: list[float | None] = []
    total_reasons: list[str] = []
    for value in (reference.u_mps, reference.v_mps, reference.w_mps):
        normalized, reason = _provider_velocity_value(value)
        total_values.append(normalized)
        total_reasons.append(reason)
    total_components = VelocityComponents(
        total_u_mps=total_values[0],
        total_v_mps=total_values[1],
        total_w_mps=total_values[2],
    )
    if "non_numeric" in total_reasons:
        return VelocitySampleStatus.INVALID, total_components, int(VelocityQC.NON_NUMERIC)
    if "nonfinite" in total_reasons:
        return VelocitySampleStatus.NONFINITE, total_components, int(VelocityQC.NONFINITE)
    if any(value is None for value in total_values):
        return VelocitySampleStatus.MISSING, total_components, int(VelocityQC.MISSING_COMPONENT)

    # 沒有 decomposition 是合法但不完整的 synthetic／legacy callback；total_* 仍是
    # provider 明示的正向物理速度，六個來源分項維持 None，不能假造 OCM 或 Stokes。
    if reference.components is None:
        return VelocitySampleStatus.TOTAL_ONLY, total_components, 0
    if type(reference.components) is not VelocityComponents:
        return VelocitySampleStatus.INVALID, total_components, int(VelocityQC.NON_NUMERIC)

    supplied = reference.components
    component_names = (
        "ocm_u_mps",
        "ocm_v_mps",
        "ocm_w_mps",
        "stokes_u_mps",
        "stokes_v_mps",
        "settling_w_mps",
    )
    component_values: dict[str, float | None] = {}
    component_reasons: list[str] = []
    for name in component_names:
        normalized, reason = _provider_velocity_value(getattr(supplied, name))
        component_values[name] = normalized
        component_reasons.append(reason)

    # total_* 可由 typed provider 重複提供，用於檢查 typed payload 與 sample header 是否
    # 一致；若省略，則以同一次 sample 的 u/v/w 作為保存的總量，避免第二個公式來源。
    for name, sample_total in zip(
        ("total_u_mps", "total_v_mps", "total_w_mps"), total_values, strict=True
    ):
        supplied_total = getattr(supplied, name)
        if supplied_total is None:
            component_values[name] = sample_total
            continue
        normalized, reason = _provider_velocity_value(supplied_total)
        component_values[name] = normalized
        component_reasons.append(reason)
        if reason == "ok" and not math.isclose(
            normalized,
            sample_total,
            rel_tol=_VELOCITY_SUM_RTOL,
            abs_tol=_VELOCITY_SUM_ATOL,
        ):
            component_reasons.append("sum_mismatch")

    components = VelocityComponents(**component_values)
    if "non_numeric" in component_reasons:
        return VelocitySampleStatus.INVALID, components, int(VelocityQC.NON_NUMERIC)
    if "nonfinite" in component_reasons:
        return VelocitySampleStatus.NONFINITE, components, int(VelocityQC.NONFINITE)
    if "missing" in component_reasons:
        return VelocitySampleStatus.MISSING, components, int(VelocityQC.MISSING_COMPONENT)
    if "sum_mismatch" in component_reasons:
        return VelocitySampleStatus.SUM_MISMATCH, components, int(VelocityQC.SUM_MISMATCH)

    assert all(value is not None for value in component_values.values())
    assert all(value is not None for value in total_values)
    if not math.isclose(
        component_values["total_u_mps"],
        component_values["ocm_u_mps"] + component_values["stokes_u_mps"],
        rel_tol=_VELOCITY_SUM_RTOL,
        abs_tol=_VELOCITY_SUM_ATOL,
    ) or not math.isclose(
        component_values["total_v_mps"],
        component_values["ocm_v_mps"] + component_values["stokes_v_mps"],
        rel_tol=_VELOCITY_SUM_RTOL,
        abs_tol=_VELOCITY_SUM_ATOL,
    ) or not math.isclose(
        component_values["total_w_mps"],
        component_values["ocm_w_mps"] + component_values["settling_w_mps"],
        rel_tol=_VELOCITY_SUM_RTOL,
        abs_tol=_VELOCITY_SUM_ATOL,
    ):
        return VelocitySampleStatus.SUM_MISMATCH, components, int(VelocityQC.SUM_MISMATCH)
    return VelocitySampleStatus.COMPLETE, components, 0


def _invalid_total_velocity_qc(reference: VelocitySample) -> SampleQC | None:
    """檢查速度總量是否可供既有積分核心使用，並在失敗時提供非零 SampleQC。

    ``VelocitySample.valid`` 是歷史介面，主要依 provider 的 ``qc`` 判定；為避免 q=0
    的 NaN、bool 或文字在步長選擇階段才造成未捕捉例外，這裡只重讀已回傳的 ``u/v/w``
    三個值。回傳非 None 時，engine 會以既有數值失敗停止，不能進入 diffusion、RK4 或
    RNG；不檢查六個分項，因為分項缺失／不一致只影響紀錄完整度，不應改變原有總速度
    積分結果。
    """

    reasons = tuple(
        _provider_velocity_value(value)[1]
        for value in (reference.u_mps, reference.v_mps, reference.w_mps)
    )
    if all(reason == "ok" for reason in reasons):
        return None
    return SampleQC.INVALID_PHYSICS


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
    *,
    velocity_context: VelocityObservationContext | None = None,
) -> EnvironmentContext | None:
    """以既有步首 sample 更新最新且完全同 state 的環境與速度 observation 欄位。

    更新只替換 list 中最後一筆 immutable Observation，不建立額外觀測、不呼叫 velocity
    或其他 forcing，也不使用亂數。若最新 observation 已是較早的時間／位置，樣本不能
    跨觀測點移植；若 valid sample 沒有 forcing 月份，環境欄位維持原有狀態，但速度仍
    可明示 ``TOTAL_ONLY``。這保留 synthetic 與正式 OCM／NWW3 可追溯證據的界線。
    """

    context = _reference_environment_context(reference=reference, state=state)
    if velocity_context is None:
        velocity_context = _reference_velocity_context(reference)
    if not observations:
        return context
    latest = observations[-1]
    if not _observation_matches_state(latest, state):
        return context
    # 同一 observation 可能沒有 forcing 月份（例如 synthetic total-only callback），此時
    # 環境欄位保留原狀，但速度欄位仍可明示「只有總速度」。兩種 context 都只從這次
    # reference sample 取得，不透過差分或其他時間點推導。
    observations[-1] = _observation(
        state,
        preserve_context=latest,
        environment_context=context,
        velocity_context=velocity_context,
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
    validate_physics_kernel_backend(settings.physics_kernel_backend)


def initialize_particle_execution(
    initial_state: ParticleState, settings: EngineSettings
) -> ParticleExecutionState:
    """驗證並建立一條粒子執行狀態，且立即寫入初始觀測。

    一般積分初始狀態必須是 ACTIVE；唯一允許的終止初始狀態是 PRE_WINDOW_DEPOSITION，
    代表固定日曆研究窗開始前已沉底。後者會立刻保存零年齡終止觀測及同點、fraction=0
    的 terminal event，且不進入速度、forcing 或亂數步進。位置使用公尺、時間使用 UTC
    奈秒，第一筆觀測的 age_seconds 沿用輸入狀態。一般 ACTIVE 初始化亦不取樣 forcing
    或消耗 RNG。next_output_age_seconds 從完整輸出間隔開始，與既有輸出語意一致。
    """

    _validate_engine_settings(settings)
    if initial_state.status not in {
        ParticleStatus.ACTIVE,
        ParticleStatus.PRE_WINDOW_DEPOSITION,
    }:
        raise ValueError(
            "initial_state 只允許 ACTIVE 或 PRE_WINDOW_DEPOSITION"
        )
    if initial_state.status == ParticleStatus.PRE_WINDOW_DEPOSITION:
        if initial_state.age_seconds != 0.0:
            raise ValueError("PRE_WINDOW_DEPOSITION 的初始 age_seconds 必須為 0")
        terminal_event = replace(
            _terminal_event(initial_state, EventType.PRE_WINDOW_DEPOSITION),
            fraction=0.0,
        )
        return ParticleExecutionState(
            state=initial_state,
            observations=[_observation(initial_state)],
            events=[terminal_event],
            step_count=0,
            minimum_clamp_count=0,
            next_output_age_seconds=settings.output_interval_seconds,
        )
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
    velocity_context: VelocityObservationContext | None = None,
    failure_attributes: dict[str, bool | float | int | str] | None = None,
) -> ParticleAdvanceResult:
    """依既有停止順序更新狀態、事件及終點觀測。

    ``environment_context``／``velocity_context`` 僅由同一個步首 sample 傳入，且一定來自
    目前 state；它讓 invalid step-start 的 terminal observation 在最新固定 observation
    尚未到達目前 state 時仍保留失敗品質旗標。步末 boundary、max-age 或 stage failure
    沒有顯式同點 sample 時不傳入，仍遵守 status-only replace 的 context preservation
    規則，不借用步首或 RK4 中間資料。
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
        velocity_context=velocity_context,
    )
    return ParticleAdvanceResult(terminal=True, stepped=False, state=execution.state)


def _recover_boundary_from_reference_drift(
    state: ParticleState,
    *,
    reference: VelocitySample,
    dt_seconds: float,
    boundaries: BoundaryGeometry,
    behavior_class: str,
) -> tuple[ParticleState, list[BoundaryEvent]] | None:
    """中間計算點落到無法取樣的位置時，以步首漂移補做可驗證的終止邊界定位。

    原始網格不能在陸地或範圍外提供速度，因此四階計算的中間點可能先失敗，來不及走到
    正常的邊界判定。此處只用步首速度建立一條簡化直線，且僅在第一個碰到的確實是海岸、
    流場外、沉積或上浮材質離開海面等終止事件時才採用。非上浮粒子的 active 海面反射
    不再由這條一階 reference drift 近似接手；它必須先通過 adaptive halving，並只在
    下一次折半會低於 ``dt_min`` 時，以 ``SurfaceStageVelocityProvider`` 重算完整 RK4。
    若只是離開局部分析區、得到 active 海面反射，或上下界不足以證明物理邊界，均不使用
    簡化結果，以免把未知環境誤當作可繼續的邊界。所有採用的終止事件仍標記此判定來源，
    正式驗收時應以更小時間步長再次檢查差異。
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


def _is_proven_surface_stage_crossing(
    state: ParticleState,
    *,
    reference: VelocitySample,
    error: SamplingError,
    behavior_class: str,
) -> bool:
    """判斷 RK4 失敗是否確實來自步內越過已知海面，而非泛化 QC 放寬。

    OCM 的垂向取樣只接受上下兩層都能夾住 ``z_m``；因此失效 stage 若帶有有限
    ``eta_m``／``bed_z_m``，且步首與失效 stage 分別位於同一水柱的海面下／海面上，
    就能證明這是幾何邊界事件。這裡要求步首 reference 也有有限且相容上下界，避免
    只依賴失效查詢點的部分資訊猜測 crossing；時間缺口、乾點、域外、非有限上下界或
    非垂向品質旗標一律回傳 False。只有品質旗標精確等於 ``VERTICAL_UNSUPPORTED``、
    步首參考樣本有效且位於集中定義的海面／海床微米容許帶內，且中間計算點 z 嚴格高於
    當地 eta 才成立；組合旗標不會被當成純海面事件。步首容許帶只承接一般取樣介面已
    判定有效的邊界位置，不會放寬失敗中間點的海面上界。``behavior_class`` 為上浮
    （``rising``）時沿用海面退出政策，不走本函式的沉降／懸浮反射重試。
    """

    if behavior_class == "rising" or error.stage not in {"k2", "k3", "k4"}:
        return False
    if (
        not reference.valid
        or error.qc != SampleQC.VERTICAL_UNSUPPORTED
        or error.context is None
    ):
        return False
    context = error.context
    try:
        reference_eta = float(reference.eta_m)
        reference_bed = float(reference.bed_z_m)
        context_eta = float(context.eta_m)
        context_bed = float(context.bed_z_m)
        state_z = float(state.z_m)
        context_z = float(context.z_m)
    except (TypeError, ValueError, OverflowError):
        return False
    if not all(
        math.isfinite(value)
        for value in (reference_eta, reference_bed, context_eta, context_bed, state_z, context_z)
    ):
        return False
    return not (
        reference_bed > reference_eta + VERTICAL_BOUNDARY_TOLERANCE_M
        or context_bed > context_eta + VERTICAL_BOUNDARY_TOLERANCE_M
        or state_z < reference_bed - VERTICAL_BOUNDARY_TOLERANCE_M
        or state_z > reference_eta + SURFACE_BOUNDARY_TOLERANCE_M
        or context_z <= context_eta
    )


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
    與 RNG 之前依既有 QC 終止。已知海面穿越才可在 RNG 前以 ``dt`` 二分重試；若下一次
    折半會低於 ``dt_min``，只對該完整確定性 RK4 重試套用 stage 海面反射速度包裝器。
    包裝器僅鏡射 k2--k4 的合法海面查詢，任何重查失敗均 fail closed；成功完成四個 stage
    後才恰套用一次 Brownian／``+div(K)|dt|`` operator split。最後 proposed position 仍走
    一般海面／海床解析，中間 stage 不產生虛構事件；其他 stage 失敗不使用這個調節。
    終止狀態會立即寫入最後觀測，因此呼叫端可以在每次 sweep 後安全 checkpoint；
    ``on_step`` 只在真正完成數值步時呼叫一次。失敗事件另附版本化的安全診斷，所有
    boundary recovery 與重試都不在 RK4 stage 中插入隨機擴散。
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
    # OCM 對海面以上的 z 不做單側外插；若失效 sample 仍同時提供有限 eta／bed，
    # 先依明確幾何證據鏡射回水柱內，再在同一 UTC／age 重新取樣。這不是清掉 QC：
    # 只有 helper 證明「目前 z 高於海面且鏡射後仍在 bed--eta 水柱內」才會重試，
    # 且重試發生在 diffusion、RK4 及 RNG 之前。其他失效原因維持原本 fail-closed。
    if reference.qc == SampleQC.VERTICAL_UNSUPPORTED:
        recovered_start = recover_surface_boundary_at_step_start(
            state,
            reference_sample=reference,
            behavior_class=behavior_class,
        )
        if recovered_start is not None:
            state, recovered_events = recovered_start
            execution.state = state
            execution.events.extend(recovered_events)
            reference = velocity(state.x_m, state.y_m, state.z_m, state.time_utc_ns)
    # reference 若已是有效步首 sample，這裡只會有原本的一次查詢；若前一個失效步首
    # 成功被海面反射，第二次查詢是同一 state 的邊界重試，不把未知資料升格成速度，
    # 也不消耗 RNG。環境欄位只 enrichment 同一 state 的最新 observation，不另取樣或
    # 改變輸出 cadence。
    velocity_context = _reference_velocity_context(reference)
    environment_context = _enrich_latest_observation_from_reference(
        execution.observations,
        state,
        reference,
        velocity_context=velocity_context,
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
            velocity_context=velocity_context,
            failure_attributes=_failure_attributes(
                execution, settings, reason="invalid_velocity_sample", stage=error.stage,
                qc=error.qc, context=error.context,
            ),
        )
    invalid_total_qc = _invalid_total_velocity_qc(reference)
    if invalid_total_qc is not None:
        # q=0 的 NaN／bool／文字總速度仍不能進入既有 choose_time_step；把它轉成既有
        # numerical-failure 流程，並保留同一 state 的 reference 速度狀態。這不會增加
        # callback、擴散取樣或 RNG 次數。
        error = SamplingError(
            "step_start",
            invalid_total_qc,
            context=SamplingContext(
                state.x_m,
                state.y_m,
                state.z_m,
                state.time_utc_ns,
                reference.eta_m,
                reference.bed_z_m,
            ),
        )
        return _terminate_execution(
            execution,
            status=ParticleStatus.NUMERICAL_FAILURE,
            event_type=EventType.NUMERICAL_FAILURE,
            environment_context=environment_context,
            velocity_context=velocity_context,
            failure_attributes=_failure_attributes(
                execution,
                settings,
                reason="invalid_velocity_sample",
                stage=error.stage,
                qc=error.qc,
                context=error.context,
            ),
        )
    # 只有明示唯讀、相同輸入可重現的速度取樣器，才能把步首樣本交給 RK4 的 k1。
    # 一般可呼叫物件不具備這個能力，仍維持「步首查詢一次，再由 RK4 查詢 k1--k4」的
    # 原始呼叫順序；此判斷也讓後續海面階段速度包裝器重試不會誤用步首樣本。
    step_start_sample = reference if supports_step_start_sample_reuse(velocity) else None
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
            velocity_context=velocity_context,
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
            velocity_context=velocity_context,
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
        physics_kernel_backend=settings.physics_kernel_backend,
    )
    if decision.limiting_reason == "minimum_clamp":
        execution.minimum_clamp_count += 1
        if execution.minimum_clamp_count > settings.maximum_minimum_clamps:
            return _terminate_execution(
                execution,
                status=ParticleStatus.NUMERICAL_FAILURE,
                event_type=EventType.NUMERICAL_FAILURE,
                velocity_context=velocity_context,
                failure_attributes=_failure_attributes(
                    execution, settings, reason="minimum_clamp_limit", stage="limits",
                    attempted_dt_seconds=-decision.seconds,
                ),
            )
    accepted_step_seconds = decision.seconds
    surface_retry_count = 0
    stage_surface_adjustment_attempted = False
    stage_error: SamplingError | None = None
    while True:
        try:
            proposed = split_rk4_brownian_step(
                state,
                dt_seconds=-accepted_step_seconds,
                velocity=velocity,
                coefficients=diffusion_sample,
                rng=rng,
                step_start_sample=step_start_sample,
                physics_kernel_backend=settings.physics_kernel_backend,
            )
            break
        except SamplingError as error:
            next_step_seconds = accepted_step_seconds * 0.5
            proven_surface_crossing = _is_proven_surface_stage_crossing(
                state,
                reference=reference,
                error=error,
                behavior_class=behavior_class,
            )
            if proven_surface_crossing and next_step_seconds >= settings.dt_min_seconds:
                # RK4 stage 失敗發生在 Brownian operator split 之前，所以縮短步長重試不會
                # 消耗亂數。只有完整 stage 成功後才會進入一次 Brownian；若縮短後仍越面，
                # 會繼續二分直到設定的 dt_min，避免過早改用階段邊界條件。第一次
                # 嘗試以步首樣本重用 k1；一旦進入折半，後續重試重新查詢 k1，讓
                # 速度取樣器的三角形搜尋提示／其他查詢狀態維持原本重試呼叫順序。
                step_start_sample = None
                accepted_step_seconds = next_step_seconds
                surface_retry_count += 1
                continue
            if proven_surface_crossing and next_step_seconds < settings.dt_min_seconds:
                # 這是唯一允許中間計算點條件式備援的位置。新速度包裝器重新執行完整
                # k1--k4；它只在 k2--k4 的精確垂向失敗上鏡射 z，且直接以相同 x/y/t 重查。
                # 分離步驟只會在四個中間點全部成功後取一次布朗運動亂數，因此原失敗嘗試、
                # 自適應折半與包裝器內的失敗重查都不消耗亂數產生器（RNG）。
                stage_surface_adjustment_attempted = True
                adjusted_velocity = SurfaceStageVelocityProvider(
                    velocity=velocity,
                    step_start_state=state,
                    step_start_sample=reference,
                    behavior_class=behavior_class,
                )
                try:
                    proposed = split_rk4_brownian_step(
                        state,
                        dt_seconds=-accepted_step_seconds,
                        velocity=adjusted_velocity,
                        coefficients=diffusion_sample,
                        rng=rng,
                        # 反射包裝器的階段計數器必須從 k1 開始，且其 k1 可能
                        # 需要重新取得原始邊界樣本；因此不能沿用原速度取樣器的步首樣本。
                        step_start_sample=None,
                        physics_kernel_backend=settings.physics_kernel_backend,
                    )
                except SamplingError as adjusted_error:
                    stage_error = adjusted_error
                break
            stage_error = error
            break

    recovered_boundary = False
    if stage_error is not None:
        error = stage_error
        boundary_qc = SampleQC.OUTSIDE_HORIZONTAL_DOMAIN | SampleQC.DRY_FACE | SampleQC.VERTICAL_UNSUPPORTED
        recovery = (
            _recover_boundary_from_reference_drift(
                state,
                reference=reference,
                dt_seconds=accepted_step_seconds,
                boundaries=boundaries,
                behavior_class=behavior_class,
            )
            if not stage_surface_adjustment_attempted and error.qc & boundary_qc
            else None
        )
        if recovery is None:
            status, event_type = _status_from_sampling_error(error)
            return _terminate_execution(
                execution, status=status, event_type=event_type,
                velocity_context=velocity_context,
                failure_attributes=_failure_attributes(
                    execution, settings, reason="rk_stage_unrecoverable", stage=error.stage,
                    qc=error.qc, context=error.context, attempted_dt_seconds=-accepted_step_seconds,
                ),
            )
        proposed, recovered_events = recovery
        execution.events.extend(recovered_events)
        recovered_boundary = True
    if not recovered_boundary:
        proposed, vertical_events = resolve_vertical_boundaries(
            state, proposed, reference_sample=reference, behavior_class=behavior_class
        )
        if surface_retry_count:
            vertical_events = [
                replace(
                    event,
                    attributes={
                        **event.attributes,
                        "boundary_locator": "adaptive_rk4_surface_retry",
                        "retry_count": surface_retry_count,
                        "accepted_dt_seconds": accepted_step_seconds,
                    },
                )
                if event.event_type is EventType.SURFACE_CONTACT
                else event
                for event in vertical_events
            ]
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
