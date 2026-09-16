"""CPU 批次粒子狀態的結構分離陣列（Structure of Arrays, SoA）資料契約。

本模組把每個粒子的固定身分欄位與會隨時間改變的數值欄位分開保存，讓 Phase 2
``ProductionBatch`` 可以以 NumPy 陣列完成作用中粒子選取與分塊回寫，而不必把 Python
dataclass 物件逐一解包。本容器只負責資料排列、固定狀態碼、作用中粒子壓縮（active
compaction）與分散回寫（scatter）；實際 RK4、Brownian、邊界與品質檢查仍由
``run_particle`` 共用的單步 engine 執行，因此不會另訂物理語意。

所有位置與尺度欄位均為公尺制，``time_utc_ns`` 是世界協調時間（UTC）的奈秒整數，
``triangle_hint`` 是可選的 native mesh 三角形提示，``-1`` 代表尚未知道提示。粒子
狀態碼不是由雜湊或列舉的執行期順序產生，而是由本檔明列的固定對照表編碼，確保
checkpoint、chunk 合併及不同 Python 程序之間可以穩定交換。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from operator import index
from types import MappingProxyType
from typing import ClassVar

import numpy as np

from .models import ParticleState, ParticleStatus

# 這個順序是資料契約的一部分：新增狀態時必須明確決定新整數，不能依賴 enum
# 目前的宣告順序、Python hash 或任何執行環境特性重新編號既有狀態。
_STATUS_CODE_PAIRS: tuple[tuple[ParticleStatus, int], ...] = (
    (ParticleStatus.ACTIVE, 0),
    (ParticleStatus.FLOW_DOMAIN_EXIT, 1),
    (ParticleStatus.COAST_CONTACT, 2),
    (ParticleStatus.SURFACE_REGIME_EXIT, 3),
    (ParticleStatus.DEPOSITED, 4),
    (ParticleStatus.FORCING_START, 5),
    (ParticleStatus.DATA_GAP, 6),
    (ParticleStatus.MAX_AGE, 7),
    (ParticleStatus.NUMERICAL_FAILURE, 8),
    (ParticleStatus.PRE_WINDOW_DEPOSITION, 9),
)

# MappingProxyType 只禁止呼叫端改寫對照表；數值對照本身仍是明確列出的固定契約。
PARTICLE_STATUS_TO_CODE: Mapping[ParticleStatus, int] = MappingProxyType(
    dict(_STATUS_CODE_PAIRS)
)
PARTICLE_CODE_TO_STATUS: Mapping[int, ParticleStatus] = MappingProxyType(
    {code: status for status, code in _STATUS_CODE_PAIRS}
)


def _coerce_integer_vector(values: Sequence[int] | np.ndarray, *, name: str) -> np.ndarray:
    """把索引或三角形提示轉成連續 ``int64`` 一維陣列並拒絕模糊輸入。

    浮點數即使看起來像整數也不接受，避免 ``1.9`` 被靜默截成 ``1``；布林值也不
    視為索引。空序列沒有可造成截斷的值，因此仍可安全建立空的 ``int64`` 陣列。
    """

    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{name} 必須是一維整數陣列")
    if raw.size == 0:
        return np.empty(0, dtype=np.int64)
    if raw.dtype.kind not in "iu":
        raise TypeError(f"{name} 必須使用整數 dtype，不接受浮點或布林值")
    if raw.dtype.kind == "u" and int(raw.max()) > np.iinfo(np.int64).max:
        raise ValueError(f"{name} 含有超出 int64 範圍的無號整數")
    return np.ascontiguousarray(raw, dtype=np.int64)


def _encode_status(status: ParticleStatus) -> int:
    """依固定狀態碼對照表（codebook）編碼一個粒子狀態，拒絕未登錄的狀態值。"""

    try:
        normalized = ParticleStatus(status)
    except (TypeError, ValueError) as error:
        raise ValueError(f"無法將粒子狀態轉成已登錄的 ParticleStatus：{status!r}") from error
    return PARTICLE_STATUS_TO_CODE[normalized]


@dataclass(slots=True)
class ParticleBatch:
    """可變的 SoA 粒子批次容器。

    固定身分以 ``tuple[str, ...]`` 保存，包含粒子、情境、研究站、分析區域與受體
    識別碼；``member_id`` 雖為數值，仍屬不可由作用中粒子壓縮／分散回寫
    （active compaction/scatter）改寫的身分欄位。其餘陣列是可由未來批次核心更新的
    動態欄位：位置與年齡為公尺／秒的 ``float64``、時間為 UTC 奈秒 ``int64``、狀態為
    固定狀態碼對照表（codebook）的 ``int64``、
    local exit 為 ``bool``，以及 ``-1`` 代表未知的 native mesh ``triangle_hint``。

    建構器會檢查所有陣列是一維、等長、連續且使用指定 dtype。連續性是資料契約的一
    部分，因為 ``slice_view`` 必須能回傳共享底層資料的連續一維 view；本容器不猜測
    mesh 的三角形總數，只檢查提示值不得小於 ``-1``。
    """

    particle_id: tuple[str, ...]
    scenario_id: tuple[str, ...]
    study_site_id: tuple[str, ...]
    analysis_region_id: tuple[str, ...]
    receptor_id: tuple[str, ...]
    member_id: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    z_m: np.ndarray
    age_seconds: np.ndarray
    time_utc_ns: np.ndarray
    status_code: np.ndarray
    own_local_exit_recorded: np.ndarray
    triangle_hint: np.ndarray

    _IDENTITY_FIELDS: ClassVar[tuple[str, ...]] = (
        "particle_id",
        "scenario_id",
        "study_site_id",
        "analysis_region_id",
        "receptor_id",
    )
    _ARRAY_DTYPES: ClassVar[dict[str, np.dtype]] = {
        "member_id": np.dtype(np.int64),
        "x_m": np.dtype(np.float64),
        "y_m": np.dtype(np.float64),
        "z_m": np.dtype(np.float64),
        "age_seconds": np.dtype(np.float64),
        "time_utc_ns": np.dtype(np.int64),
        "status_code": np.dtype(np.int64),
        "own_local_exit_recorded": np.dtype(np.bool_),
        "triangle_hint": np.dtype(np.int64),
    }
    _DYNAMIC_FIELDS: ClassVar[tuple[str, ...]] = (
        "x_m",
        "y_m",
        "z_m",
        "age_seconds",
        "time_utc_ns",
        "status_code",
        "own_local_exit_recorded",
        "triangle_hint",
    )

    def __post_init__(self) -> None:
        """建構後立即驗證資料契約，避免錯誤 shape 延後到批次計算才爆發。"""

        self.validate()

    def validate(self) -> None:
        """檢查身分、陣列 shape/dtype、狀態碼與三角形提示的安全條件。

        驗證只保證資料容器本身一致，不驗證位置是否落在某一份特定 mesh 內，因為
        ``ParticleBatch`` 不持有 mesh，也不應臆測三角形編號上限。呼叫端若在初始化
        後直接修改陣列，後續的 view、作用中粒子壓縮與分散回寫操作也會再次驗證。
        """

        length: int | None = None
        for field_name in self._IDENTITY_FIELDS:
            values = getattr(self, field_name)
            if not isinstance(values, tuple) or any(not isinstance(value, str) for value in values):
                raise TypeError(f"{field_name} 必須是只含字串的 tuple")
            if length is None:
                length = len(values)
            elif len(values) != length:
                raise ValueError("ParticleBatch 身分欄位長度不一致")
        assert length is not None

        for field_name, expected_dtype in self._ARRAY_DTYPES.items():
            values = getattr(self, field_name)
            if not isinstance(values, np.ndarray):
                raise TypeError(f"{field_name} 必須是 numpy.ndarray")
            if values.ndim != 1 or values.shape[0] != length:
                raise ValueError(f"{field_name} 必須是一維且長度為 {length}")
            if values.dtype != expected_dtype:
                raise TypeError(f"{field_name} dtype 必須是 {expected_dtype}，實際為 {values.dtype}")
            if not values.flags.c_contiguous:
                raise ValueError(f"{field_name} 必須是連續陣列，才能提供安全的 slice view")

        valid_codes = np.fromiter(PARTICLE_CODE_TO_STATUS, dtype=np.int64)
        if self.status_code.size and not np.all(np.isin(self.status_code, valid_codes)):
            raise ValueError("status_code 含有未登錄的 ParticleStatus code")
        if self.triangle_hint.size and np.any(self.triangle_hint < -1):
            raise ValueError("triangle_hint 只能是 -1 或非負整數；不檢查 mesh 上限")

    def __len__(self) -> int:
        """回傳批次中的粒子數；零長度批次是合法資料。"""

        return len(self.particle_id)

    @classmethod
    def from_particle_states(
        cls,
        states: Iterable[ParticleState],
        triangle_hints: Sequence[int] | np.ndarray | None = None,
    ) -> ParticleBatch:
        """由既有不可變 ``ParticleState`` 序列建立 SoA 批次。

        ``states`` 的順序就是所有欄位的粒子順序；若提供 ``triangle_hints``，其第
        ``i`` 個值只描述第 ``i`` 個 state 的 mesh 提示，未提供時全部使用 ``-1``。
        本轉換不會新增或推導任何物理量，狀態只依固定狀態碼對照表編碼，故可用於 reference
        核心結果與未來批次核心之間的明確邊界。
        """

        state_list = list(states)
        if any(not isinstance(state, ParticleState) for state in state_list):
            raise TypeError("states 必須只包含 ParticleState")
        if triangle_hints is None:
            hints = np.full(len(state_list), -1, dtype=np.int64)
        else:
            hints = _coerce_integer_vector(triangle_hints, name="triangle_hints")
            if hints.size != len(state_list):
                raise ValueError("triangle_hints 長度必須與 states 相同")

        return cls(
            particle_id=tuple(state.particle_id for state in state_list),
            scenario_id=tuple(state.scenario_id for state in state_list),
            study_site_id=tuple(state.study_site_id for state in state_list),
            analysis_region_id=tuple(state.analysis_region_id for state in state_list),
            receptor_id=tuple(state.receptor_id for state in state_list),
            member_id=np.asarray([state.member_id for state in state_list], dtype=np.int64),
            x_m=np.asarray([state.x_m for state in state_list], dtype=np.float64),
            y_m=np.asarray([state.y_m for state in state_list], dtype=np.float64),
            z_m=np.asarray([state.z_m for state in state_list], dtype=np.float64),
            age_seconds=np.asarray([state.age_seconds for state in state_list], dtype=np.float64),
            time_utc_ns=np.asarray([state.time_utc_ns for state in state_list], dtype=np.int64),
            status_code=np.asarray([_encode_status(state.status) for state in state_list], dtype=np.int64),
            own_local_exit_recorded=np.asarray(
                [state.own_local_exit_recorded for state in state_list], dtype=np.bool_
            ),
            triangle_hint=hints,
        )

    def to_particle_states(self) -> list[ParticleState]:
        """依原順序還原 ``ParticleState``，且不把 batch 專用的 hint 寫入 state schema。

        ``triangle_hint`` 是未來提示式定位快取的批次欄位，現有 ``ParticleState`` 沒有這個
        欄位；因此往返保證的是既有 state 欄位完全一致，提示則留在 batch 供下一個
        分塊核心（chunk kernel）使用。
        """

        self.validate()
        return [
            ParticleState(
                particle_id=self.particle_id[index_value],
                scenario_id=self.scenario_id[index_value],
                member_id=int(self.member_id[index_value]),
                study_site_id=self.study_site_id[index_value],
                analysis_region_id=self.analysis_region_id[index_value],
                receptor_id=self.receptor_id[index_value],
                x_m=float(self.x_m[index_value]),
                y_m=float(self.y_m[index_value]),
                z_m=float(self.z_m[index_value]),
                time_utc_ns=int(self.time_utc_ns[index_value]),
                age_seconds=float(self.age_seconds[index_value]),
                status=PARTICLE_CODE_TO_STATUS[int(self.status_code[index_value])],
                own_local_exit_recorded=bool(self.own_local_exit_recorded[index_value]),
            )
            for index_value in range(len(self))
        ]

    def slice_view(self, start: int, stop: int) -> ParticleBatch:
        """回傳共享數值底層資料的連續一維 view，身分欄位則依同一範圍切片。

        ``start`` 與 ``stop`` 採 Python slice 的負值與超界規則，但步長固定為 1，
        因此所有非空數值欄位都同時滿足「共享底層資料」與「連續 view」。修改回傳
        batch 的數值陣列會直接反映原 batch；身分 tuple 不可變，且此方法不會修改它們。
        """

        self.validate()
        start_index = index(start)
        stop_index = index(stop)
        selected = slice(start_index, stop_index, 1)
        return type(self)(
            particle_id=self.particle_id[selected],
            scenario_id=self.scenario_id[selected],
            study_site_id=self.study_site_id[selected],
            analysis_region_id=self.analysis_region_id[selected],
            receptor_id=self.receptor_id[selected],
            **{
                field_name: getattr(self, field_name)[selected]
                for field_name in self._ARRAY_DTYPES
            },
        )

    def active_indices(self) -> np.ndarray:
        """回傳 status 為 ``ACTIVE`` 的來源索引，維持原 batch 順序。"""

        self.validate()
        return np.flatnonzero(self.status_code == PARTICLE_STATUS_TO_CODE[ParticleStatus.ACTIVE]).astype(
            np.int64, copy=False
        )

    def compact_active(self) -> tuple[ParticleBatch, np.ndarray]:
        """複製作用中粒子（active）成連續 batch，並回傳其在來源 batch 的索引。

        advanced indexing 會建立獨立資料，使壓縮後 batch 可安全交給未來分塊核心
        更新；``source_indices`` 保存回寫對應，呼叫端不得依粒子目前排列自行猜測。
        沒有 active 粒子時回傳合法的空 batch 與空 ``int64`` 索引。
        """

        self.validate()
        source_indices = self.active_indices()
        arrays = {
            field_name: np.ascontiguousarray(getattr(self, field_name)[source_indices])
            for field_name in self._ARRAY_DTYPES
        }
        compacted = type(self)(
            particle_id=tuple(self.particle_id[index_value] for index_value in source_indices),
            scenario_id=tuple(self.scenario_id[index_value] for index_value in source_indices),
            study_site_id=tuple(self.study_site_id[index_value] for index_value in source_indices),
            analysis_region_id=tuple(
                self.analysis_region_id[index_value] for index_value in source_indices
            ),
            receptor_id=tuple(self.receptor_id[index_value] for index_value in source_indices),
            **arrays,
        )
        return compacted, source_indices.copy()

    def scatter_dynamic_from(
        self, compacted_batch: ParticleBatch, source_indices: Sequence[int] | np.ndarray
    ) -> None:
        """將 compacted batch 的動態欄位安全回寫來源位置。

        回寫前會驗證索引是一維、整數、唯一且在範圍內，並逐欄比對五個字串身分與
        ``member_id``。任何身分不一致都立即拒絕，避免作用中粒子壓縮後把一個
        粒子的軌跡狀態寫入另一個粒子；固定身分欄位永遠不會被此方法修改。
        """

        self.validate()
        if not isinstance(compacted_batch, ParticleBatch):
            raise TypeError("compacted_batch 必須是 ParticleBatch")
        compacted_batch.validate()
        indices = _coerce_integer_vector(source_indices, name="source_indices")
        if indices.size != len(compacted_batch):
            raise ValueError("source_indices 長度必須等於 compacted_batch 粒子數")
        if indices.size and (np.any(indices < 0) or np.any(indices >= len(self))):
            raise IndexError("source_indices 含有超出來源 batch 範圍的索引")
        if np.unique(indices).size != indices.size:
            raise ValueError("source_indices 必須唯一，避免同一粒子被不確定地多次回寫")

        for field_name in self._IDENTITY_FIELDS:
            source_identity = tuple(
                getattr(self, field_name)[int(index_value)] for index_value in indices
            )
            compacted_identity = getattr(compacted_batch, field_name)
            if source_identity != compacted_identity:
                raise ValueError(f"compacted_batch 的 {field_name} 與來源位置不一致")
        if not np.array_equal(self.member_id[indices], compacted_batch.member_id):
            raise ValueError("compacted_batch 的 member_id 與來源位置不一致")

        # 先複製動態值再寫入，讓即使 compacted_batch 是來源的 view，也不會因回寫順序
        # 造成另一個動態欄位讀到已被修改的資料；這裡仍只觸碰明列的動態欄位。
        dynamic_values = {
            field_name: np.array(getattr(compacted_batch, field_name), copy=True)
            for field_name in self._DYNAMIC_FIELDS
        }
        for field_name, values in dynamic_values.items():
            getattr(self, field_name)[indices] = values
        self.validate()


__all__ = ["PARTICLE_CODE_TO_STATUS", "PARTICLE_STATUS_TO_CODE", "ParticleBatch"]
