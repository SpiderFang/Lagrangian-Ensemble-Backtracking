"""跨 forcing、積分器、邊界與輸出模組共用的科學資料型別。

本模組集中定義粒子狀態、速度取樣與事件語意，避免各模組以裸字串或特殊數值表示
缺值。所有位置都位於 flow domain 固定的公尺制 CRS；``z_m`` 採 SCHISM 慣例的
positive-up，海面約為 ``eta``、海床約為 ``-depth``。經緯度僅在 I/O 邊界轉換。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntFlag, StrEnum


class SampleQC(IntFlag):
    """速度或波浪取樣的可組合品質旗標。

    ``OK`` 是唯一可直接送入正式積分的值。其餘 bit 分開保留域外、時間缺口、乾點、
    垂向無包夾、波浪缺值與數值失敗，避免把不同原因都變成 NaN 後失去可追溯性。
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
    """粒子生命週期狀態；停止原因不以布林值壓縮。"""

    ACTIVE = "active"
    FLOW_DOMAIN_EXIT = "flow_domain_open_exit"
    COAST_CONTACT = "coast_contact"
    SURFACE_REGIME_EXIT = "surface_regime_exit"
    DEPOSITED = "deposited"
    FORCING_START = "forcing_start"
    DATA_GAP = "data_gap"
    MAX_AGE = "max_age"
    NUMERICAL_FAILURE = "numerical_failure"


class EventType(StrEnum):
    """可寫入事件表的固定事件名稱。

    ``OTHER_SITE_*`` 是非終止診斷，絕不可改變原始 ``study_site_id``；B-D 的 local
    與 flow domain 重合時，實作只寫一筆 ``FLOW_DOMAIN_OPEN_EXIT``，並在事件屬性
    保存同時具有 local-first-exit 語意，避免同一次 crossing 重複計數。
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


@dataclass(frozen=True, slots=True)
class VelocitySample:
    """單一 RK stage 的物理時間向前速度與局地幾何資訊。

    參數皆使用 SI 單位。``u/v/w`` 是 OCM、Stokes 與浮沉尚未必全部合成的呼叫端約定；
    取樣器必須以 ``qc`` 表示失敗，不能回傳零速度替代域外或缺值。``eta_m`` 與
    ``bed_z_m`` 供海面／海床障壁判定，``horizontal_scale_m`` 與
    ``vertical_scale_m`` 供 adaptive time-step 控制。
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

    @property
    def valid(self) -> bool:
        """只有完全無 QC bit 的樣本才可進入正式積分。"""

        return self.qc == SampleQC.OK


@dataclass(frozen=True, slots=True)
class ParticleState:
    """一個 member 在某時刻的不可變狀態。

    ``time_utc_ns`` 為 UTC epoch nanoseconds；backward 積分只讓它隨負 ``dt`` 遞減，
    不改變速度函式的物理正向。``age_seconds`` 永遠非負，用來判斷 max-age censor。
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
    """步內插值後的事件紀錄。

    crossing 座標與時間對應同一線段內的 ``fraction``（0 到 1）。他站 local-domain
    事件以 ``related_study_site_id`` 指出被穿越站點，但原粒子站點保持不變。
    ``attributes`` 只放穩定、可 JSON/Parquet 序列化的輔助值。
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
