"""所有模組共用的粒子、速度與事件資料格式。

這裡把「粒子目前在哪裡」、「海流與波浪取樣是否可用」及「何時碰到邊界」定義成同一套
資料格式。這樣遇到缺資料、乾點或數值問題時，不會有人用 0 速度、有人用空值，導致結果
無法比較。位置一律先換算為公尺再計算；``z_m`` 向上為正，海面接近 ``eta_m``，海床為
負水深。經緯度只在讀取資料和畫圖時使用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntFlag, StrEnum


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
    """一次速度查詢得到的正向物理速度與附近網格資訊。

    數值都採國際單位制：速度為公尺/秒，長度為公尺。``u_mps``、``v_mps``、``w_mps``
    分別是東向、北向與向上的速度，可由海流、波浪造成的漂移和物體浮沉速度合成。
    取樣失敗一定寫入品質檢查旗標（``qc``），不可假裝成靜水。海面、海床及附近網格大小
    用於判斷粒子能否繼續前進，以及下一步最多可走多遠。
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
