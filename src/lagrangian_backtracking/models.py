"""所有模組共用的粒子、速度與事件資料格式。

這裡把「粒子目前在哪裡」、「海流與波浪取樣是否可用」及「何時碰到邊界」定義成同一套
資料格式。這樣遇到缺資料、乾點或數值問題時，不會有人用 0 速度、有人用空值，導致結果
無法比較。位置一律先換算為公尺再計算；``z_m`` 向上為正，海面接近 ``eta_m``，海床為
負水深。經緯度只在讀取資料和畫圖時使用。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntFlag, StrEnum
from typing import Final

# OCM、Stokes、引擎環境 context 與輸出 validator 共用的海面上界數值政策。
# 這個尺度只處理公尺制 z 與移動海面 eta 在海面邊界定位時，由浮點運算、RK4 stage
# 積分與時空內插共同形成的數值殘差；其大小不能以單一機器精度（machine epsilon）解釋，
# 也不是網格層距、物理混合層厚度或可任意放大的邊界緩衝。checkpoint-8 觀測到的最大
# 海面邊界定位數值殘差（含浮點與積分／內插）約為 3.367686e-6 m，因此選擇 5e-6 m，
# 在涵蓋已知證據的同時仍只相當於 5 微米；超過此距離仍必須保留原本的 invalid/QC。
SURFACE_BOUNDARY_TOLERANCE_M: Final[float] = 5.0e-6

# 海床端與「海床高於海面」的幾何一致性沿用既有 1 微米契約。它與海面上界分開，
# 避免為修正移動海面的邊界定位數值殘差而對海底越界提供更寬的通行範圍。
VERTICAL_BOUNDARY_TOLERANCE_M: Final[float] = 1.0e-6


def clamp_query_z_to_surface(z_m: float, eta_m: float) -> float | None:
    """把海面容許帶內的 query z 夾回海面，並拒絕真正的海面上越。

    ``z_m`` 與 ``eta_m`` 都是公尺制、海面向上為正的垂向座標。當 query z 位於
    ``eta_m`` 上方不超過 ``SURFACE_BOUNDARY_TOLERANCE_M`` 時，回傳 ``eta_m``，讓
    OCM 表層支援與 Stokes profile 對同一個物理海面計算；海面下的 query 保留原值。
    非有限輸入或超過容許尺度的上越回傳 ``None``，呼叫端必須維持既有 fail-closed
    狀態。此函式不檢查海床、乾點、域外或時間軸，因為那些狀態必須由各自資料契約判定。
    """

    try:
        normalized_z = float(z_m)
        normalized_eta = float(eta_m)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(normalized_z) or not math.isfinite(normalized_eta):
        return None
    if normalized_z > normalized_eta + SURFACE_BOUNDARY_TOLERANCE_M:
        return None
    return normalized_eta if normalized_z > normalized_eta else normalized_z


class SampleQC(IntFlag):
    """速度或波浪取樣的品質檢查旗標（程式欄位名為 ``qc``）。

    ``OK`` 表示資料完整，可直接用來計算。其餘旗標分別記錄位置在資料範圍外、時間有
    缺口、海域已乾掉、深度無法由上下層包住、波浪不可用或數值失敗。不同原因分開保存，
    才能在成果中知道哪一種問題造成粒子停止。
    """

    OK = 0
    OUTSIDE_HORIZONTAL_DOMAIN = 1 << 0
    OUTSIDE_TIME_RANGE = 1 << 1
    TIME_GAP = 1 << 2
    DRY_FACE = 1 << 3
    VERTICAL_UNSUPPORTED = 1 << 4
    WAVE_UNSUPPORTED = 1 << 5
    INVALID_PHYSICS = 1 << 6
    NUMERICAL_FAILURE = 1 << 7


class ParticleStatus(StrEnum):
    """粒子生命週期狀態；停止原因不以布林值壓縮。

    PRE_WINDOW_DEPOSITION 表示抽樣沉底時刻早於固定日曆研究窗，因此沒有可積分的漂流
    期間；它是可追溯的研究窗分類，不是資料缺漏或數值失敗。
    """

    ACTIVE = "active"
    FLOW_DOMAIN_EXIT = "flow_domain_open_exit"
    COAST_CONTACT = "coast_contact"
    SURFACE_REGIME_EXIT = "surface_regime_exit"
    DEPOSITED = "deposited"
    FORCING_START = "forcing_start"
    DATA_GAP = "data_gap"
    MAX_AGE = "max_age"
    NUMERICAL_FAILURE = "numerical_failure"
    PRE_WINDOW_DEPOSITION = "pre_window_deposition"


class EventType(StrEnum):
    """可寫入事件表的固定事件名稱。

    ``OTHER_SITE_*`` 是非終止診斷，絕不可改變原始 ``study_site_id``；B-D 的 local
    與 flow domain 重合時，實作只寫一筆 ``FLOW_DOMAIN_OPEN_EXIT``，並在事件屬性
    保存同時具有 local-first-exit 語意，避免同一次 crossing 重複計數。
    PRE_WINDOW_DEPOSITION 是零步終止事件，fraction=0 表示它與初始沉底狀態同點。
    """

    LOCAL_DOMAIN_FIRST_EXIT = "local_domain_first_exit"
    OTHER_SITE_LOCAL_DOMAIN_ENTER = "other_site_local_domain_enter"
    OTHER_SITE_LOCAL_DOMAIN_EXIT = "other_site_local_domain_exit"
    FLOW_DOMAIN_OPEN_EXIT = "flow_domain_open_exit"
    COAST_CONTACT = "coast_contact"
    SURFACE_CONTACT = "surface_contact"
    SURFACE_REGIME_EXIT = "surface_regime_exit"
    BED_CONTACT = "bed_contact"
    DEPOSITED = "deposited"
    DATA_GAP = "data_gap"
    MAX_AGE = "max_age"
    FORCING_START = "forcing_start"
    NUMERICAL_FAILURE = "numerical_failure"
    PRE_WINDOW_DEPOSITION = "pre_window_deposition"


class VelocitySampleStatus(StrEnum):
    """Observation 速度紀錄的資料完整程度。

    ``COMPLETE`` 只用於同一個步首取樣同時提供 OCM、Stokes 水平漂流與沉降分項，且
    三個分項相加與總速度相符的樣本。一般四參數 synthetic callback 沒有分項來源時
    使用 ``TOTAL_ONLY``；其餘狀態分開表示尚未取樣、缺分項、非有限值、型別無效或
    總和不一致。這個列舉與環境樣本狀態分離，避免「有海面高度」被誤讀成「有速度
    分項」，也讓輸出端能對每個速度欄位保留明確缺值語意。
    """

    NOT_SAMPLED = "not_sampled"
    COMPLETE = "complete"
    TOTAL_ONLY = "total_only"
    INVALID = "invalid"
    MISSING = "missing"
    NONFINITE = "nonfinite"
    SUM_MISMATCH = "sum_mismatch"


class VelocityQC(IntFlag):
    """速度紀錄本身的品質檢查旗標，不與環境 ``qc`` 共用欄位。

    旗標只描述速度值或其分解，不取代 ``SampleQC`` 對 forcing 來源的取樣結果。無效、
    缺分項、非有限、型別錯誤及總和不符都必須留下非零旗標；``TOTAL_ONLY`` 是明確的
    「只有總速度」工程 callback 狀態，沒有把缺少來源誤標成速度數值錯誤。
    """

    OK = 0
    SAMPLE_INVALID = 1 << 0
    MISSING_COMPONENT = 1 << 1
    NONFINITE = 1 << 2
    NON_NUMERIC = 1 << 3
    SUM_MISMATCH = 1 << 4


@dataclass(frozen=True, slots=True)
class VelocityComponents:
    """一次速度取樣的具名、固定欄位分項資料，所有欄位單位都是公尺/秒。

    ``total_*`` 是與 ``VelocitySample.u_mps/v_mps/w_mps`` 對應的正向物理總速度；
    ``ocm_*`` 是 OCM current 的東、北、向上分量；``stokes_u_mps``／``stokes_v_mps``
    是波浪造成且實際使用的水平 Stokes 漂流；``settling_w_mps`` 是以向上為正的垂向
    沉降速度，本專案沉降案例為負值。Stokes 垂向分量與沉降水平分量不屬於本資料契約，
    因此沒有欄位，也不可用 0 假造波浪或沉降來源。欄位可以是 ``None``，讓一般 callback
    明示只有總速度或部分分項；正式 ``CombinedMonthForcing`` 的有效樣本會填滿全部九欄。

    這組欄位只承載速度提供器在同一次 UTC／位置取樣中實際提供的資料，不自行查詢
    forcing；引擎把它綁定到完全相同時間與 ``x_m/y_m/z_m`` 的位置觀測時，會檢查有限性、
    型別與 ``total = OCM + Stokes(水平) + settling`` 的明定容差。
    """

    total_u_mps: float | None = None
    total_v_mps: float | None = None
    total_w_mps: float | None = None
    ocm_u_mps: float | None = None
    ocm_v_mps: float | None = None
    ocm_w_mps: float | None = None
    stokes_u_mps: float | None = None
    stokes_v_mps: float | None = None
    settling_w_mps: float | None = None


@dataclass(frozen=True, slots=True)
class VelocitySample:
    """一次速度查詢得到的正向物理速度與附近網格資訊。

    數值都採國際單位制：速度為公尺/秒，長度為公尺。``u_mps``、``v_mps``、``w_mps``
    分別是東向、北向與向上的速度，可由海流、波浪造成的漂移和物體浮沉速度合成。
    取樣失敗一定寫入品質檢查旗標（``qc``），不可假裝成靜水。海面、海床及附近網格大小
    用於判斷粒子能否繼續前進，以及下一步最多可走多遠。``components`` 若存在，表示
    同一次查詢實際取出的 OCM、Stokes 水平與垂向沉降分項；若為 ``None``，只能在輸出
    中標為「只有總速度」，不可由總速度反推來源。
    """

    u_mps: float
    v_mps: float
    w_mps: float
    eta_m: float
    bed_z_m: float
    horizontal_scale_m: float
    vertical_scale_m: float
    qc: SampleQC = SampleQC.OK
    source_face_id: int | None = None
    triangle_id: int | None = None
    forcing_month_id: str | None = None
    diagnostics: dict[str, float | int | str] = field(default_factory=dict)
    components: VelocityComponents | None = None

    @property
    def valid(self) -> bool:
        """只有沒有任何品質問題的樣本，才可用於正式粒子計算。"""

        return self.qc == SampleQC.OK


@dataclass(frozen=True, slots=True)
class ParticleState:
    """一個系集成員在某時刻的完整狀態。

    ``time_utc_ns`` 是世界協調時間（UTC）的奈秒整數。逆向追蹤時只讓時間往過去減少，
    海流函式本身仍回傳真實世界向前流動的速度。``age_seconds`` 是已回溯多久，永遠為
    非負值，用來判斷是否達到最長回溯時間。
    """

    particle_id: str
    scenario_id: str
    member_id: int
    study_site_id: str
    analysis_region_id: str
    receptor_id: str
    x_m: float
    y_m: float
    z_m: float
    time_utc_ns: int
    age_seconds: float = 0.0
    status: ParticleStatus = ParticleStatus.ACTIVE
    own_local_exit_recorded: bool = False


@dataclass(frozen=True, slots=True)
class BoundaryEvent:
    """粒子在單一步驟中碰到邊界或停止條件的紀錄。

    交點位置、深度與時間都依同一個 ``fraction``（步首為 0、步末為 1）計算，避免時間
    和位置不相符。若粒子穿過其他站點的關注海域，只在 ``related_study_site_id`` 記下
    對方站點；粒子原本所屬的站點不會改變。``attributes`` 只放能安全寫進結果檔的補充
    說明。
    """

    particle_id: str
    scenario_id: str
    member_id: int
    study_site_id: str
    analysis_region_id: str
    receptor_id: str
    event_type: EventType
    time_utc_ns: int
    x_m: float
    y_m: float
    z_m: float
    fraction: float
    related_study_site_id: str | None = None
    boundary_segment_id: str | None = None
    boundary_s_m: float | None = None
    source_face_id: int | None = None
    triangle_id: int | None = None
    forcing_month_id: str | None = None
    attributes: dict[str, bool | float | int | str] = field(default_factory=dict)
