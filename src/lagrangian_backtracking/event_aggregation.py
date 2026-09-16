"""事件彙整的資料型別、空白拓撲初始化與串流聚合器。

本模組定義事件彙整管線使用的不可變資料容器、唯讀防禦、零值拓撲初始化，
以及將已完成事件 chunk 逐筆累加的固定記憶體 reducer。粒子事件如何由
``ParticleResult`` 轉成單一 chunk 仍由 ``aggregate_result_events`` 負責；本
模組不讀取 OCM／NWW 原始或分析產品，也不替缺值、陸地、乾點、時間缺口或
數值失敗填入物理零值。串流 reducer 只在既有 chunk 契約已成立後做整數計數
合併，不能取代正式 run 的輸入驗證、粒子互斥性證明或科學品質控制。
所有空間距離與邊界弧長均以公尺（m）表示，所有旅行年齡與年齡箱邊界均以秒
（s）表示；站點網格陣列的軸順序固定為 ``(y_cell, x_cell)``，邊界旅行年齡
直方圖則固定為 ``(s_bin, age_bin)``。

初始化會依已驗證的 :class:`~lagrangian_backtracking.aggregate_spec.AggregateSpec`
建立完整的零拓撲：每個站點都有六個事件網格計數陣列，每個站點引用的 local
或 outer 邊界段都有獨立的聚合鍵，即使同一段跨站共享或同時屬於兩種分類也不
合併。這些零值只代表尚未累加任何成員的容器，不代表觀測到零事件，也不會把
資料缺口、陸地、乾點或數值失敗誤寫成物理零值。

本模組保存的結果只能作為「條件式來源足跡」或「相對來源權重」的累加載體。
在尚未建立先驗、似然與觀測驗證前，任何後續計數都不得解釋為絕對來源機率
或因果歸因。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .aggregate_spec import AggregateSpec
from .engine import Observation, ParticleResult
from .models import BoundaryEvent, EventType, ParticleState, ParticleStatus
from .scenarios import Scenario

__all__ = [
    "BoundaryAggregateKey",
    "ReceptorAggregateKey",
    "SourceReceptorAggregateKey",
    "CrossSiteAggregateKey",
    "SiteEventGridCounts",
    "EventAggregateChunk",
    "EventAggregateAccumulator",
    "initialize_event_aggregate",
    "aggregate_result_events",
    "merge_event_aggregate_chunks",
]


# 邊界分類是資料契約的一部分；若放寬為任意字串，後續統計可能把不同物理
# 角色悄悄合併，因此所有公開邊界鍵都在建構時封閉驗證。
_BOUNDARY_KINDS = frozenset({"local", "outer"})

# 所有事件直方圖與網格計數都會落在 int64，先檢查上限才能避免 NumPy 轉型時
# 將過大的 uint64 靜默截斷或讓累計值繞回負數。
_INT64_MAX = np.iinfo(np.int64).max

# 以浮點輸入描述的長度可能使本應整除的比例出現最後幾位誤差；這個容差只
# 用於辨識「應該剛好整除」的 bin 數，不會改變非整除長度的短尾 bin。
_EXACT_BIN_RATIO_REL_TOL = 1e-12
_EXACT_BIN_RATIO_ABS_TOL = 1e-12

# 只有這些狀態代表一條 ParticleResult 的正式終止原因；每個結果必須在 events
# 中留下且只留下對應的一筆 terminal event。其餘事件是路徑診斷，不會取代最終
# 狀態的終止語意。把 mapping 固定在此處可避免以字串相似度猜測停止原因。
_TERMINAL_EVENT_TYPE_BY_STATUS: Mapping[ParticleStatus, EventType] = {
    ParticleStatus.FLOW_DOMAIN_EXIT: EventType.FLOW_DOMAIN_OPEN_EXIT,
    ParticleStatus.COAST_CONTACT: EventType.COAST_CONTACT,
    ParticleStatus.SURFACE_REGIME_EXIT: EventType.SURFACE_REGIME_EXIT,
    ParticleStatus.DEPOSITED: EventType.DEPOSITED,
    ParticleStatus.FORCING_START: EventType.FORCING_START,
    ParticleStatus.DATA_GAP: EventType.DATA_GAP,
    ParticleStatus.MAX_AGE: EventType.MAX_AGE,
    ParticleStatus.NUMERICAL_FAILURE: EventType.NUMERICAL_FAILURE,
    ParticleStatus.PRE_WINDOW_DEPOSITION: EventType.PRE_WINDOW_DEPOSITION,
}

# outcome mapping 只能保存已登錄且已終止的 ParticleStatus；把允許集合固定在
# 模組層可讓 EventAggregateChunk 在封存時拒絕 ACTIVE 或未知字串，而不讓下游
# 統計把一個未定義的終止原因當成合法分母。這裡只保存 enum 的正式 value，因為
# outcome mapping 的資料契約是可序列化字串，而不是 Python enum 物件。
_NON_ACTIVE_PARTICLE_STATUS_VALUES = frozenset(
    status.value
    for status in ParticleStatus
    if status != ParticleStatus.ACTIVE
)

# denominator_policy 排除資料／數值失敗，以及固定日曆窗前已沉底的成員。其 outcome
# 仍保留供診斷，但這些成員不能把來源、邊界或跨站事件放進 numerator；否則 numerator
# 與 valid denominator 會來自不同的成員母體。pre-window 是研究窗分類，不另記為失敗格。
_INVALID_MEMBER_STATUSES = frozenset(
    {
        ParticleStatus.DATA_GAP,
        ParticleStatus.NUMERICAL_FAILURE,
        ParticleStatus.PRE_WINDOW_DEPOSITION,
    }
)

# SiteEventGridCounts 的欄位名稱與輸出陣列一一對應；集中列舉可讓第二切片
# 使用相同的欄位順序建立可變累加副本，而不在各事件分支重複拼寫欄位。
_SITE_GRID_COUNT_FIELDS = (
    "local_first_exit_count",
    "outer_first_exit_count",
    "bed_first_contact_count",
    "bed_repeated_contact_count",
    "data_gap_failure_count",
    "numerical_failure_count",
)


def _require_nonempty_string(value: object, *, label: str) -> str:
    """驗證識別碼是非空原生字串並回傳原值。

    這些欄位是跨資料表連接用的 key，不進行去除空白或大小寫正規化，避免
    建構資料型別時偷偷改變上游已驗證的識別碼。呼叫端若需要更嚴格的 slug
    規則，應由 ``AggregateSpec`` 或對應 manifest 負責；本模組只要求此切片
    明定的「非空字串」條件。
    """

    if type(value) is not str or not value:
        raise ValueError(f"{label} 必須是非空字串。")
    return value


def _require_boundary_kind(value: object, *, label: str) -> str:
    """驗證邊界分類只能是 ``local`` 或 ``outer``。"""

    if type(value) is not str or value not in _BOUNDARY_KINDS:
        raise ValueError(f"{label} 必須是 local 或 outer。")
    return value


def _copy_mapping(value: object, *, label: str) -> dict[object, object]:
    """複製 mapping 為普通 dict，切斷呼叫端後續對原容器的修改影響。

    所有公開聚合 mapping 最終都會再以 ``MappingProxyType`` 包裝；這個中間
    副本是必要的，因為單純包裝既有 dict 仍會讓 caller 透過原始 dict 改寫
    聚合結果。自訂 mapping 若無法被穩定複製，會轉成一致的 ``ValueError``。
    """

    if not isinstance(value, Mapping):
        raise ValueError(f"{label} 必須是 mapping。")
    try:
        return dict(value)
    except Exception as error:
        raise ValueError(f"{label} 無法複製。") from error


def _nonnegative_count(value: object, *, label: str) -> int:
    """驗證非負原生 Python 整數，明確拒絕 bool 與 NumPy scalar。

    累加器的標量欄位屬於資料契約中的計數，不接受浮點近似或隱式轉型；使用
    ``type`` 而非 ``isinstance`` 也能避免 Python 將 ``bool`` 視為 ``int``。
    本層不截斷任意精度的 Python 整數，int64 陣列的固定寬度限制則由陣列
    驗證函式另外處理。
    """

    if type(value) is not int:
        raise ValueError(f"{label} 必須是非負 int，且不可為 bool。")
    if value < 0:
        raise ValueError(f"{label} 不可為負。")
    return value


def _safe_sum_int_array(value: np.ndarray, *, label: str) -> int:
    """逐值以 Python ``int`` 加總計數陣列，避免 NumPy 固定寬度溢位。

    聚合陣列雖然封存為 ``int64``，但 ``np.sum`` 或以 NumPy scalar 為累加器仍
    可能在接近 ``int64`` 上限時繞回負值。這個 helper 只在既有 dtype、形狀與
    非負性驗證完成後使用；它逐一把每個元素轉成 Python 任意精度整數，再以
    Python 加法累計，因此回傳的總量不會因 NumPy 固定寬度而溢位。陣列中的
    每個元素仍代表一個已分類事件或成員計數，負值一旦出現即 fail-closed。

    Args:
        value: 已驗證的非負整數 NumPy 陣列，可為任意維度。
        label: 錯誤訊息中描述資料語意的欄位名稱。

    Returns:
        以原生 Python ``int`` 表示的逐元素總和。

    Raises:
        ValueError: 陣列元素不是非負計數時。
    """

    total = 0
    for element in value.flat:
        count = int(element)
        if count < 0:
            raise ValueError(f"{label} 不可包含負值，無法安全加總。")
        total += count
    return total


def _safe_sum_count_mapping(
    value: Mapping[object, int],
    *,
    label: str,
) -> int:
    """逐項以 Python ``int`` 加總 mapping 計數，避免 ``np.int64`` 累加溢位。

    ``outcome``、受體分母與站點分母是 mapping 中的標量計數。即使目前建構
    流程已要求原生 Python ``int``，關聯不變量仍透過此 helper 重新逐項驗證並
    累加，讓未來替換上游 writer 或手動建構資料時不會把 NumPy scalar 的固定
    寬度帶入合計。Python ``int`` 使用任意精度，適合在比較前保存完整的總量。

    Args:
        value: 已複製且值應為非負原生整數的 mapping。
        label: 錯誤訊息中描述 mapping 語意的欄位名稱。

    Returns:
        以原生 Python ``int`` 表示的所有 value 總和。
    """

    total = 0
    for key, count_value in value.items():
        count = _nonnegative_count(
            count_value,
            label=f"{label}[{key!r}]",
        )
        total += count
    return total


def _safe_add_int64_arrays(
    total: np.ndarray,
    incoming: np.ndarray,
    *,
    label: str,
) -> None:
    """以逐元素 Python 整數預檢後，安全累加兩個 int64 計數陣列。

    ``total`` 與 ``incoming`` 都是已經過 ``EventAggregateChunk`` 驗證的非負
    ``int64`` 陣列；但是兩個合法陣列相加後仍可能超過 ``int64`` 上限。若直接
    使用 NumPy 向量化加法，固定寬度整數可能繞回負值，讓正式來源足跡統計
    靜默損壞。因此每一個 ``(y, x)``、``s`` 或 ``(s, age)`` 位置都先轉成
    Python 任意精度 ``int``，檢查 ``incoming > INT64_MAX - total``，確認安全
    後才寫回原生 ``int64`` 陣列。此函式只修改呼叫端提供的暫時累加陣列，
    不會修改已封存的輸入 chunk。

    Args:
        total: 可寫入的目前 int64 計數陣列。
        incoming: 待加入且形狀相同的 int64 計數陣列。
        label: 錯誤訊息中描述物理量與座標軸的欄位名稱。

    Raises:
        ValueError: dtype、形狀或陣列值不符合非負 int64 計數契約時。
        RuntimeError: 任一位置的結果會超過 int64 最大值時。
    """

    if total.dtype != np.dtype(np.int64) or incoming.dtype != np.dtype(np.int64):
        raise ValueError(f"{label} 必須使用 int64 dtype。")
    if total.shape != incoming.shape:
        raise ValueError(f"{label} 的累加陣列 shape 必須完全相同。")

    for index in np.ndindex(total.shape):
        current = int(total[index])
        added = int(incoming[index])
        if current < 0 or added < 0:
            raise ValueError(f"{label} 不可包含負值。")
        if added > _INT64_MAX - current:
            raise RuntimeError(f"{label} 累加將超過 int64 上限。")
        total[index] = current + added


def _readonly_nonnegative_int_array(
    value: object,
    *,
    label: str,
    ndim: int,
) -> np.ndarray:
    """複製並驗證非負整數陣列，最後固定為唯讀 int64。

    輸入必須是 NumPy 整數 dtype；布林與浮點即使數值看起來像整數也會拒絕，
    以免缺值、比例或其他連續量誤進計數欄位。轉型前先確認每個元素都在
    ``int64`` 可表示範圍內，並先檢查非負性，避免 NumPy 在轉型階段靜默繞回。
    ``copy=True`` 與 write flag 同時使用，確保 caller 既不能透過原始陣列
    修改結果，也不能直接修改回傳陣列。
    """

    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} 必須是 {ndim}D 非負整數陣列。") from error

    if raw.ndim != ndim:
        raise ValueError(f"{label} 必須是 {ndim}D 陣列。")
    if raw.dtype.kind not in "iu":
        raise ValueError(f"{label} 必須是整數 dtype，不接受 bool 或浮點。")

    try:
        if raw.size and np.any(raw < 0):
            raise ValueError(f"{label} 不可包含負值。")
        if raw.size and np.any(raw > _INT64_MAX):
            raise ValueError(f"{label} 的值超出 int64 可表示範圍。")
        copied = np.array(raw, dtype=np.int64, copy=True)
    except ValueError:
        raise
    except (TypeError, OverflowError) as error:
        raise ValueError(f"{label} 無法安全轉為 int64。") from error

    copied.setflags(write=False)
    return copied


def _readonly_boundary_edges(value: object, *, label: str) -> np.ndarray:
    """複製並驗證一組以公尺表示的邊界弧長格線。

    邊界格線是一維有限數值陣列，至少含起點與終點，第一個位置必須是 0 m，
    後續位置嚴格遞增。這裡保留 float64 是因為最後一個短 bin 可能不是整數
    公尺；它只描述空間離散化，不是事件計數。回傳陣列會關閉寫入權限，且
    不與 caller 的輸入陣列共享記憶體。
    """

    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} 必須是一維有限邊界格線。") from error
    if raw.ndim != 1 or raw.size < 2 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{label} 必須是至少兩點的一維數值格線。")

    try:
        copied = np.array(raw, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} 必須可轉為 float64 公尺格線。") from error
    if not np.all(np.isfinite(copied)):
        raise ValueError(f"{label} 必須全部為有限值。")
    if copied[0] != 0.0:
        raise ValueError(f"{label} 的第一個邊界必須是 0 m。")
    if np.any(np.diff(copied) <= 0.0):
        raise ValueError(f"{label} 必須嚴格遞增。")

    copied.setflags(write=False)
    return copied


def _readonly_array_mapping(
    value: object,
    *,
    label: str,
    ndim: int,
    validator: str = "count",
) -> MappingProxyType:
    """複製陣列 mapping 並保存每個值的唯讀副本。

    ``validator`` 只在本模組內區分計數陣列與邊界格線兩種資料語意。相同原始
    NumPy 物件若被多個 key 引用，會共用同一個防禦性副本；這讓初始化時同一
    邊界段的共享格線保有相同物件，同時不暴露 caller 的可寫記憶體。
    """

    copied_mapping = _copy_mapping(value, label=label)
    result: dict[object, np.ndarray] = {}
    copied_by_identity: dict[int, tuple[object, np.ndarray]] = {}

    for key, array_value in copied_mapping.items():
        identity = id(array_value)
        cached = copied_by_identity.get(identity)
        if cached is not None and cached[0] is array_value:
            result[key] = cached[1]
            continue

        if validator == "edges":
            copied_array = _readonly_boundary_edges(array_value, label=label)
        else:
            copied_array = _readonly_nonnegative_int_array(
                array_value,
                label=label,
                ndim=ndim,
            )
        copied_by_identity[identity] = (array_value, copied_array)
        result[key] = copied_array

    return MappingProxyType(result)


def _validate_age_bin_edges(value: object) -> np.ndarray:
    """驗證年齡箱邊界並回傳可封存於聚合結果的 float64 副本。

    年齡軸的單位是秒，至少需要兩個有限邊界；邊界必須嚴格遞增，且第一點
    必須精確為 0 s。陣列會與 caller 解除記憶體共享並關閉寫入權限，作為
    ``EventAggregateChunk`` 中所有旅行年齡直方圖的唯一 age 軸契約。
    """

    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "age_bin_edges_seconds 必須是一維有限秒數格線。"
        ) from error
    if raw.ndim != 1 or raw.size < 2 or raw.dtype.kind not in "iuf":
        raise ValueError(
            "age_bin_edges_seconds 必須是至少兩點的一維數值格線。"
        )

    try:
        edges = np.array(raw, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "age_bin_edges_seconds 必須可轉為 float64 秒數格線。"
        ) from error
    if not np.all(np.isfinite(edges)):
        raise ValueError("age_bin_edges_seconds 必須全部為有限值。")
    if edges[0] != 0.0:
        raise ValueError(
            "age_bin_edges_seconds 的第一個邊界必須精確為 0 s。"
        )
    if np.any(np.diff(edges) <= 0.0):
        raise ValueError("age_bin_edges_seconds 必須嚴格遞增。")

    edges.setflags(write=False)
    return edges


def _validate_key_mapping(
    value: object,
    *,
    label: str,
    key_type: type,
    value_type: type | None = None,
) -> dict[object, object]:
    """複製並驗證以公開聚合鍵為索引的 mapping。

    key 與 value 的類別檢查集中在此處，讓 ``EventAggregateChunk`` 的每個
    mapping 都採用同一套錯誤語意。值若是 NumPy 陣列，會由各欄位後續的
    形狀與唯讀驗證處理；此函式只負責 mapping 拓撲與資料類別的第一層隔離。
    """

    copied = _copy_mapping(value, label=label)
    for key, item in copied.items():
        if not isinstance(key, key_type):
            raise ValueError(f"{label} 的 key 類型不符。")
        if value_type is not None and not isinstance(item, value_type):
            raise ValueError(f"{label} 的 value 類型不符。")
    return copied


def _validate_site_mapping(
    value: object,
    *,
    label: str,
    value_type: type | None = None,
) -> dict[object, object]:
    """複製並驗證以非空站點識別碼為 key 的 mapping。"""

    copied = _copy_mapping(value, label=label)
    for key, item in copied.items():
        _require_nonempty_string(key, label=f"{label} key")
        if value_type is not None and not isinstance(item, value_type):
            raise ValueError(f"{label} 的 value 類型不符。")
    return copied


@dataclass(frozen=True, slots=True)
class BoundaryAggregateKey:
    """單一站點、邊界分類與邊界段的事件彙整識別鍵。

    ``study_site_id`` 與 ``boundary_segment_id`` 是非空字串，
    ``boundary_kind`` 僅可為 ``local`` 或 ``outer``。同一邊界段若在不同站點
    或兩種分類中出現，必須保留不同 key，因為它們代表不同的事件分母與空間
    角色；key 本身不宣稱任何來源機率或因果關係。
    """

    study_site_id: str
    boundary_kind: str
    boundary_segment_id: str

    def __post_init__(self) -> None:
        """固定並驗證站點、分類與邊界段識別欄位。"""

        _require_nonempty_string(self.study_site_id, label="study_site_id")
        _require_boundary_kind(self.boundary_kind, label="boundary_kind")
        _require_nonempty_string(
            self.boundary_segment_id,
            label="boundary_segment_id",
        )


@dataclass(frozen=True, slots=True)
class ReceptorAggregateKey:
    """單一站點與受體的事件分母彙整識別鍵。

    兩個識別欄位都必須是非空字串。此 key 只用於保存受體層級的有效成員
    分母，不能將分母直接解讀為受體的絕對來源機率。
    """

    study_site_id: str
    receptor_id: str

    def __post_init__(self) -> None:
        """固定並驗證站點與受體識別欄位。"""

        _require_nonempty_string(self.study_site_id, label="study_site_id")
        _require_nonempty_string(self.receptor_id, label="receptor_id")


@dataclass(frozen=True, slots=True)
class SourceReceptorAggregateKey:
    """邊界段、受體與站點組合的條件式來源足跡識別鍵。

    ``boundary_kind`` 仍只允許 ``local`` 或 ``outer``；其餘三個識別欄位必須
    是非空字串。這個 key 對應的計數描述「由特定邊界分類進入某受體」的相對
    彙整量，並不自動提供先驗、似然或觀測驗證，因此不是絕對來源機率。
    """

    study_site_id: str
    receptor_id: str
    boundary_kind: str
    boundary_segment_id: str

    def __post_init__(self) -> None:
        """固定並驗證站點、受體、分類與邊界段識別欄位。"""

        _require_nonempty_string(self.study_site_id, label="study_site_id")
        _require_nonempty_string(self.receptor_id, label="receptor_id")
        _require_boundary_kind(self.boundary_kind, label="boundary_kind")
        _require_nonempty_string(
            self.boundary_segment_id,
            label="boundary_segment_id",
        )


@dataclass(frozen=True, slots=True)
class CrossSiteAggregateKey:
    """跨站點來源與目標的事件彙整識別鍵。

    source 與 target 都是非空站點識別碼，且必須不同；相同站點的事件屬於
    站內彙整，不應被誤寫成跨站流動。此 key 的 unique-member count 只保存
    條件式跨站成員數，不代表跨站來源的絕對機率或因果歸因。
    """

    source_study_site_id: str
    target_study_site_id: str

    def __post_init__(self) -> None:
        """固定並驗證來源與目標站點識別欄位及不相等限制。"""

        source = _require_nonempty_string(
            self.source_study_site_id,
            label="source_study_site_id",
        )
        target = _require_nonempty_string(
            self.target_study_site_id,
            label="target_study_site_id",
        )
        if source == target:
            raise ValueError("source_study_site_id 與 target_study_site_id 必須不同。")


@dataclass(frozen=True, slots=True)
class SiteEventGridCounts:
    """單一站點六類事件的公尺制水平網格計數。

    每個陣列的軸順序都是 ``(y_cell, x_cell)``，不保存經緯度；六個陣列必須
    同形狀、恰為二維、使用非負整數且可安全轉成 int64。欄位依序代表 local
    首次離開、outer 首次離開、海床首次接觸、海床重複接觸、資料缺口失敗與
    數值失敗。初始化時全部為零；零只表示尚未累加，不會把缺值或失敗狀態
    轉成物理零事件。
    """

    local_first_exit_count: np.ndarray
    outer_first_exit_count: np.ndarray
    bed_first_contact_count: np.ndarray
    bed_repeated_contact_count: np.ndarray
    data_gap_failure_count: np.ndarray
    numerical_failure_count: np.ndarray

    def __post_init__(self) -> None:
        """驗證六個網格計數同形狀，並建立獨立唯讀 int64 副本。"""

        names_and_values = (
            ("local_first_exit_count", self.local_first_exit_count),
            ("outer_first_exit_count", self.outer_first_exit_count),
            ("bed_first_contact_count", self.bed_first_contact_count),
            ("bed_repeated_contact_count", self.bed_repeated_contact_count),
            ("data_gap_failure_count", self.data_gap_failure_count),
            ("numerical_failure_count", self.numerical_failure_count),
        )
        copied_arrays = tuple(
            _readonly_nonnegative_int_array(value, label=name, ndim=2)
            for name, value in names_and_values
        )
        expected_shape = copied_arrays[0].shape
        if any(array.shape != expected_shape for array in copied_arrays[1:]):
            raise ValueError("SiteEventGridCounts 的六個陣列必須同形狀。")

        for (name, _), array in zip(names_and_values, copied_arrays, strict=True):
            object.__setattr__(self, name, array)


@dataclass(frozen=True, slots=True)
class EventAggregateChunk:
    """一個可累加事件資料塊的不可變拓撲與目前計數。

    ``site_grid_counts`` 的陣列軸是 ``(y_cell, x_cell)``；邊界格線與弧長計數
    的 key 是 ``BoundaryAggregateKey``，其中弧長 bins 沿邊界由 0 m 逐段增加，
    最後一個 bin 可為短 bin。``age_bin_edges_seconds`` 是所有旅行年齡直方圖
    共用的 0 s 起始格線，且正式 age 軸由 ``AggregateSpec.age_bin_edges_seconds``
    鎖定；``boundary_travel_age_histogram`` 的形狀固定為
    ``(s_bin, age_bin)``，而 ``source_receptor_travel_age_histogram`` 是一維
    age-bin 計數，兩者的 age 軸都必須精確等於 ``edges.size - 1``。距離欄位
    以公尺、旅行年齡欄位以秒理解。

    所有 mapping 都會複製後以 ``MappingProxyType`` 暴露，巢狀 outcome mapping
    也會逐層唯讀化；所有陣列都解除與輸入的記憶體共享並關閉寫入權限。標量
    計數拒絕 bool 與負值。初始化函式的空 source-receptor 與跨站 mapping 是
    第一切片的刻意零拓撲；事件聚合函式會依輸入情境建立完整的 source-receptor、
    跨站與 outcome 細項 key，沒有事件的 key 仍保留為零。

    這個資料容器只保存條件式來源足跡或相對來源權重的原始累加載體，不把
    成員計數解讀成絕對來源機率，也不承擔先驗、似然與觀測驗證的科學推論。
    """

    site_grid_counts: Mapping[str, SiteEventGridCounts]
    boundary_bin_edges_m: Mapping[BoundaryAggregateKey, np.ndarray]
    boundary_arclength_raw_count: Mapping[BoundaryAggregateKey, np.ndarray]
    boundary_travel_age_histogram: Mapping[BoundaryAggregateKey, np.ndarray]
    age_bin_edges_seconds: np.ndarray
    source_receptor_raw_count: Mapping[SourceReceptorAggregateKey, int]
    source_receptor_travel_age_histogram: Mapping[
        SourceReceptorAggregateKey,
        np.ndarray,
    ]
    cross_site_unique_member_count: Mapping[CrossSiteAggregateKey, int]
    outcome_count_by_site: Mapping[str, Mapping[str, int]]
    valid_member_denominator_by_site: Mapping[str, int]
    total_member_count_by_site: Mapping[str, int]
    valid_member_denominator_by_receptor: Mapping[ReceptorAggregateKey, int]
    input_particle_count: int

    def __post_init__(self) -> None:
        """驗證欄位形狀與計數語意，並封存所有巢狀可變資料。

        邊界 raw count 的長度必須等於邊界格線形成的 s-bin 數；對應旅行年齡
        直方圖的第一軸也必須相同。這個關聯檢查防止後續累加器因 key 對上但
        bin 軸錯位而寫入錯誤空間位置。容器同時防禦性複製並封存 age 軸，因而
        能在建構時確認兩類旅行年齡直方圖都與同一組秒數格線完全對齊。
        """

        age_edges = _validate_age_bin_edges(self.age_bin_edges_seconds)
        object.__setattr__(self, "age_bin_edges_seconds", age_edges)
        age_bin_count = age_edges.size - 1

        site_grid_counts = _validate_site_mapping(
            self.site_grid_counts,
            label="site_grid_counts",
            value_type=SiteEventGridCounts,
        )
        object.__setattr__(
            self,
            "site_grid_counts",
            MappingProxyType(site_grid_counts),
        )

        boundary_edges = _validate_key_mapping(
            self.boundary_bin_edges_m,
            label="boundary_bin_edges_m",
            key_type=BoundaryAggregateKey,
        )
        boundary_edge_arrays = _readonly_array_mapping(
            boundary_edges,
            label="boundary_bin_edges_m value",
            ndim=1,
            validator="edges",
        )
        object.__setattr__(self, "boundary_bin_edges_m", boundary_edge_arrays)

        boundary_raw = _validate_key_mapping(
            self.boundary_arclength_raw_count,
            label="boundary_arclength_raw_count",
            key_type=BoundaryAggregateKey,
        )
        boundary_raw_arrays = _readonly_array_mapping(
            boundary_raw,
            label="boundary_arclength_raw_count value",
            ndim=1,
        )
        object.__setattr__(
            self,
            "boundary_arclength_raw_count",
            boundary_raw_arrays,
        )

        boundary_travel = _validate_key_mapping(
            self.boundary_travel_age_histogram,
            label="boundary_travel_age_histogram",
            key_type=BoundaryAggregateKey,
        )
        boundary_travel_arrays = _readonly_array_mapping(
            boundary_travel,
            label="boundary_travel_age_histogram value",
            ndim=2,
        )
        object.__setattr__(
            self,
            "boundary_travel_age_histogram",
            boundary_travel_arrays,
        )

        boundary_keys = set(boundary_edge_arrays)
        if set(boundary_raw_arrays) != boundary_keys:
            raise ValueError(
                "boundary_arclength_raw_count 的 key 必須與 boundary_bin_edges_m 相同。"
            )
        if set(boundary_travel_arrays) != boundary_keys:
            raise ValueError(
                "boundary_travel_age_histogram 的 key 必須與 boundary_bin_edges_m 相同。"
            )
        for key in boundary_keys:
            s_bin_count = boundary_edge_arrays[key].size - 1
            raw_array = boundary_raw_arrays[key]
            travel_array = boundary_travel_arrays[key]
            if raw_array.shape != (s_bin_count,):
                raise ValueError(
                    "boundary_arclength_raw_count 的 s-bin 軸長度與邊界格線不符。"
                )
            if travel_array.ndim != 2 or travel_array.shape[0] != s_bin_count:
                raise ValueError(
                    "boundary_travel_age_histogram 必須是 (s_bin, age_bin)，且 s-bin 軸對齊。"
                )
            if travel_array.shape[1] != age_bin_count:
                raise ValueError(
                    "boundary_travel_age_histogram 的 age-bin 軸長度必須等於 age edges 數量減一。"
                )

        source_raw = _validate_key_mapping(
            self.source_receptor_raw_count,
            label="source_receptor_raw_count",
            key_type=SourceReceptorAggregateKey,
        )
        source_raw_counts = {
            key: _nonnegative_count(value, label="source_receptor_raw_count value")
            for key, value in source_raw.items()
        }
        object.__setattr__(
            self,
            "source_receptor_raw_count",
            MappingProxyType(source_raw_counts),
        )

        source_travel = _validate_key_mapping(
            self.source_receptor_travel_age_histogram,
            label="source_receptor_travel_age_histogram",
            key_type=SourceReceptorAggregateKey,
        )
        source_travel_arrays = _readonly_array_mapping(
            source_travel,
            label="source_receptor_travel_age_histogram value",
            ndim=1,
        )
        object.__setattr__(
            self,
            "source_receptor_travel_age_histogram",
            source_travel_arrays,
        )
        if set(source_raw_counts) != set(source_travel_arrays):
            raise ValueError(
                "source_receptor raw count 與 travel age histogram 的 key 必須相同。"
            )
        for array in source_travel_arrays.values():
            if array.shape != (age_bin_count,):
                raise ValueError(
                    "source_receptor_travel_age_histogram 的 age-bin 軸長度必須等於 age edges 數量減一。"
                )

        cross_site = _validate_key_mapping(
            self.cross_site_unique_member_count,
            label="cross_site_unique_member_count",
            key_type=CrossSiteAggregateKey,
        )
        cross_site_counts = {
            key: _nonnegative_count(
                value,
                label="cross_site_unique_member_count value",
            )
            for key, value in cross_site.items()
        }
        object.__setattr__(
            self,
            "cross_site_unique_member_count",
            MappingProxyType(cross_site_counts),
        )

        outcome_by_site = _validate_site_mapping(
            self.outcome_count_by_site,
            label="outcome_count_by_site",
        )
        outcome_maps: dict[object, MappingProxyType] = {}
        for site_id, outcomes in outcome_by_site.items():
            copied_outcomes = _copy_mapping(
                outcomes,
                label="outcome_count_by_site nested mapping",
            )
            validated_outcomes: dict[object, int] = {}
            for outcome, count in copied_outcomes.items():
                _require_nonempty_string(
                    outcome,
                    label="outcome_count_by_site outcome key",
                )
                validated_outcomes[outcome] = _nonnegative_count(
                    count,
                    label="outcome_count_by_site value",
                )
            outcome_maps[site_id] = MappingProxyType(validated_outcomes)
        object.__setattr__(
            self,
            "outcome_count_by_site",
            MappingProxyType(outcome_maps),
        )

        valid_by_site = _validate_site_mapping(
            self.valid_member_denominator_by_site,
            label="valid_member_denominator_by_site",
        )
        object.__setattr__(
            self,
            "valid_member_denominator_by_site",
            MappingProxyType(
                {
                    site_id: _nonnegative_count(
                        count,
                        label="valid_member_denominator_by_site value",
                    )
                    for site_id, count in valid_by_site.items()
                }
            ),
        )

        total_by_site = _validate_site_mapping(
            self.total_member_count_by_site,
            label="total_member_count_by_site",
        )
        object.__setattr__(
            self,
            "total_member_count_by_site",
            MappingProxyType(
                {
                    site_id: _nonnegative_count(
                        count,
                        label="total_member_count_by_site value",
                    )
                    for site_id, count in total_by_site.items()
                }
            ),
        )

        valid_by_receptor = _validate_key_mapping(
            self.valid_member_denominator_by_receptor,
            label="valid_member_denominator_by_receptor",
            key_type=ReceptorAggregateKey,
        )
        object.__setattr__(
            self,
            "valid_member_denominator_by_receptor",
            MappingProxyType(
                {
                    key: _nonnegative_count(
                        count,
                        label="valid_member_denominator_by_receptor value",
                    )
                    for key, count in valid_by_receptor.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "input_particle_count",
            _nonnegative_count(
                self.input_particle_count,
                label="input_particle_count",
            ),
        )
        _validate_event_aggregate_relationships(
            site_grid_counts=site_grid_counts,
            boundary_bin_edges_m=boundary_edge_arrays,
            boundary_arclength_raw_count=boundary_raw_arrays,
            boundary_travel_age_histogram=boundary_travel_arrays,
            source_receptor_raw_count=source_raw_counts,
            source_receptor_travel_age_histogram=source_travel_arrays,
            cross_site_unique_member_count=cross_site_counts,
            outcome_count_by_site=outcome_maps,
            valid_member_denominator_by_site=self.valid_member_denominator_by_site,
            total_member_count_by_site=self.total_member_count_by_site,
            valid_member_denominator_by_receptor=self.valid_member_denominator_by_receptor,
            input_particle_count=self.input_particle_count,
        )


def _validate_event_aggregate_relationships(
    *,
    site_grid_counts: Mapping[str, SiteEventGridCounts],
    boundary_bin_edges_m: Mapping[BoundaryAggregateKey, np.ndarray],
    boundary_arclength_raw_count: Mapping[BoundaryAggregateKey, np.ndarray],
    boundary_travel_age_histogram: Mapping[BoundaryAggregateKey, np.ndarray],
    source_receptor_raw_count: Mapping[SourceReceptorAggregateKey, int],
    source_receptor_travel_age_histogram: Mapping[
        SourceReceptorAggregateKey,
        np.ndarray,
    ],
    cross_site_unique_member_count: Mapping[CrossSiteAggregateKey, int],
    outcome_count_by_site: Mapping[str, Mapping[str, int]],
    valid_member_denominator_by_site: Mapping[str, int],
    total_member_count_by_site: Mapping[str, int],
    valid_member_denominator_by_receptor: Mapping[ReceptorAggregateKey, int],
    input_particle_count: int,
) -> None:
    """驗證事件聚合塊內各資料表之間的關聯不變量。

    ``EventAggregateChunk`` 的每個欄位雖然各自通過型別、shape 與非負性驗證，
    仍可能因手動建構、錯誤的檔案讀取或未來 writer 變更而彼此不一致。例如
    boundary raw count 與 travel-age histogram 可能各自有合法形狀，卻描述不同
    粒子數；站點分母也可能與 outcome 或 failure grid 對不上。本函式在所有
    defensive-copy 完成後集中檢查這些跨欄位關係，任何一項不成立就拒絕封存。

    正式來源 numerator（local／outer 語意格網、邊界弧長、source-receptor、海床
    首次接觸與跨站 unique-member）必須與有效成員 denominator 來自同一母體；
    因此以下關係會額外限制每站、每受體及每個跨站 pair 的 numerator 不得超過
    對應有效成員數。DATA_GAP 與 NUMERICAL_FAILURE grid 是例外的失敗診斷欄位，
    只與 outcome 守恆，不屬於來源 numerator；PRE_WINDOW_DEPOSITION 只保留 outcome，
    不寫入失敗格或來源 numerator。

    所有合計都逐一把 NumPy scalar 轉成 Python ``int`` 後累加，或直接累加已驗證
    的 Python ``int``。不可使用 ``np.sum`` 或讓 ``np.int64`` 擔任累加器，因為
    聚合計數接近固定寬度上限時可能繞回負值，進而讓損壞結果通過後續驗證。
    這些計數仍只代表條件式來源足跡／相對來源權重的原始彙整量，不改變其
    科學解釋限制。

    Args:
        site_grid_counts: 以站點為 key 的六類事件網格計數。
        boundary_bin_edges_m: 邊界 key 到公尺制弧長格線的 mapping。
        boundary_arclength_raw_count: 邊界 key 到 s-bin raw count 的 mapping。
        boundary_travel_age_histogram: 邊界 key 到 ``(s, age)`` 計數的 mapping。
        source_receptor_raw_count: 站點／受體／邊界分類 raw count。
        source_receptor_travel_age_histogram: 對應的 age-bin 計數。
        cross_site_unique_member_count: 有序跨站 unique-member 計數。
        outcome_count_by_site: 站點到非 ACTIVE 終止狀態計數的 mapping。
        valid_member_denominator_by_site: 各站有效成員分母。
        total_member_count_by_site: 各站輸入成員總數。
        valid_member_denominator_by_receptor: 各站／受體有效成員分母。
        input_particle_count: 本資料塊輸入的粒子總數。

    Raises:
        ValueError: 任一站點、key、分母、outcome、網格或直方圖關係不一致時。
    """

    site_ids = set(site_grid_counts)
    if not (
        set(outcome_count_by_site) == site_ids
        and set(valid_member_denominator_by_site) == site_ids
        and set(total_member_count_by_site) == site_ids
    ):
        raise ValueError(
            "site_grid_counts、outcome_count_by_site、"
            "valid_member_denominator_by_site、total_member_count_by_site 的 "
            "site set 必須完全相同。"
        )

    unknown_boundary_sites = {
        key.study_site_id
        for key in boundary_bin_edges_m
        if key.study_site_id not in site_ids
    }
    if unknown_boundary_sites:
        raise ValueError(
            "boundary key 引用未知 site："
            f"{sorted(unknown_boundary_sites)!r}。"
        )

    unknown_source_sites = {
        key.study_site_id
        for key in source_receptor_raw_count
        if key.study_site_id not in site_ids
    }
    if unknown_source_sites:
        raise ValueError(
            "source-receptor key 引用未知 site："
            f"{sorted(unknown_source_sites)!r}。"
        )

    unknown_receptor_sites = {
        key.study_site_id
        for key in valid_member_denominator_by_receptor
        if key.study_site_id not in site_ids
    }
    if unknown_receptor_sites:
        raise ValueError(
            "receptor key 引用未知 site："
            f"{sorted(unknown_receptor_sites)!r}。"
        )

    unknown_cross_sites = {
        site_id
        for key in cross_site_unique_member_count
        for site_id in (
            key.source_study_site_id,
            key.target_study_site_id,
        )
        if site_id not in site_ids
    }
    if unknown_cross_sites:
        raise ValueError(
            "cross-site key 引用未知 site："
            f"{sorted(unknown_cross_sites)!r}。"
        )

    total_site_sum = _safe_sum_count_mapping(
        total_member_count_by_site,
        label="total_member_count_by_site",
    )
    if input_particle_count != total_site_sum:
        raise ValueError(
            "input_particle_count 必須等於各 site total_member_count_by_site "
            "的 Python int 合計。"
        )

    # outcome key 必須是已登錄的非 ACTIVE 終止狀態；valid 與 outcome 的關係
    # 也在此逐站確認，避免某一站的計數被誤放到另一站或漏掉終止原因。
    for site_id in site_ids:
        valid_count = valid_member_denominator_by_site[site_id]
        total_count = total_member_count_by_site[site_id]
        if valid_count > total_count:
            raise ValueError(
                f"site {site_id!r} 的 valid denominator 不可大於 total member count。"
            )

        outcomes = outcome_count_by_site[site_id]
        for outcome in outcomes:
            if outcome not in _NON_ACTIVE_PARTICLE_STATUS_VALUES:
                raise ValueError(
                    f"site {site_id!r} 的 outcome key {outcome!r} 必須是非 ACTIVE "
                    "ParticleStatus.value。"
                )
        outcome_total = _safe_sum_count_mapping(
            outcomes,
            label=f"outcome_count_by_site[{site_id!r}]",
        )
        if outcome_total != total_count:
            raise ValueError(
                f"site {site_id!r} 的 outcome count 合計必須等於 total member count。"
            )

    receptor_valid_by_site = {site_id: 0 for site_id in site_ids}
    for receptor_key, count_value in valid_member_denominator_by_receptor.items():
        receptor_count = _nonnegative_count(
            count_value,
            label=f"valid_member_denominator_by_receptor[{receptor_key!r}]",
        )
        site_total = total_member_count_by_site[receptor_key.study_site_id]
        if receptor_count > site_total:
            raise ValueError(
                f"receptor key {receptor_key!r} 的 valid count 不可大於其 site total。"
            )
        receptor_valid_by_site[receptor_key.study_site_id] += receptor_count

    for site_id in site_ids:
        if receptor_valid_by_site[site_id] != valid_member_denominator_by_site[site_id]:
            raise ValueError(
                f"site {site_id!r} 的 receptor valid 合計必須等於 site valid denominator。"
            )

    boundary_keys = set(boundary_bin_edges_m)
    boundary_raw_totals: dict[BoundaryAggregateKey, int] = {}
    for boundary_key in boundary_keys:
        raw_total = _safe_sum_int_array(
            boundary_arclength_raw_count[boundary_key],
            label=f"boundary raw count[{boundary_key!r}]",
        )
        travel_total = _safe_sum_int_array(
            boundary_travel_age_histogram[boundary_key],
            label=f"boundary travel-age histogram[{boundary_key!r}]",
        )
        if raw_total != travel_total:
            raise ValueError(
                f"boundary key {boundary_key!r} 的 raw array 合計必須等於 "
                "(s, age) histogram 合計。"
            )
        boundary_raw_totals[boundary_key] = raw_total

    source_totals_by_boundary = {
        boundary_key: 0
        for boundary_key in boundary_keys
    }
    source_totals_by_receptor_kind: dict[tuple[ReceptorAggregateKey, str], int] = {}
    for source_key, raw_value in source_receptor_raw_count.items():
        boundary_key = BoundaryAggregateKey(
            study_site_id=source_key.study_site_id,
            boundary_kind=source_key.boundary_kind,
            boundary_segment_id=source_key.boundary_segment_id,
        )
        if boundary_key not in boundary_keys:
            raise ValueError(
                f"source-receptor key {source_key!r} 找不到對應 boundary key。"
            )
        raw_count = _nonnegative_count(
            raw_value,
            label=f"source_receptor_raw_count[{source_key!r}]",
        )
        travel_total = _safe_sum_int_array(
            source_receptor_travel_age_histogram[source_key],
            label=f"source-receptor travel-age histogram[{source_key!r}]",
        )
        if raw_count != travel_total:
            raise ValueError(
                f"source-receptor key {source_key!r} 的 raw scalar 必須等於 "
                "age histogram 合計。"
            )
        source_totals_by_boundary[boundary_key] += raw_count
        receptor_key = ReceptorAggregateKey(
            study_site_id=source_key.study_site_id,
            receptor_id=source_key.receptor_id,
        )
        # source/receptor key set 與正式 strata 的 join 由上層 release payload
        # 驗證；EventAggregateChunk 的既有契約允許先封存尚未完成 join 的 source
        # key。只有存在對應 receptor denominator 時，這裡才套用本次新增的
        # numerator 上限，避免把分層拓撲責任偷移到此容器而改變既有 API。
        if receptor_key not in valid_member_denominator_by_receptor:
            continue
        receptor_kind = (receptor_key, source_key.boundary_kind)
        source_totals_by_receptor_kind[receptor_kind] = (
            source_totals_by_receptor_kind.get(receptor_kind, 0) + raw_count
        )

    # 同一受體可能有多個邊界段；這裡先跨 segment 合計，再與同一
    # site/receptor 的有效成員 denominator 比較，避免只檢查單一 key 而讓多段
    # source raw 總量超過可支持它的有效成員母體。
    for (receptor_key, boundary_kind), source_total in source_totals_by_receptor_kind.items():
        valid_receptor_count = valid_member_denominator_by_receptor[receptor_key]
        if source_total > valid_receptor_count:
            raise ValueError(
                f"source-receptor {receptor_key!r} 的 {boundary_kind} raw 跨 segment "
                "合計不可大於對應 valid denominator。"
            )

    for boundary_key, boundary_total in boundary_raw_totals.items():
        source_total = source_totals_by_boundary[boundary_key]
        if source_total != boundary_total:
            raise ValueError(
                f"site/kind/segment {boundary_key!r} 的所有 receptor source raw "
                "合計必須等於 boundary raw 合計。"
            )

    # local 與 outer grid 是事件空間位置的另一個投影；它們的總數必須與同站、
    # 同分類的所有邊界弧長 raw count 相等。只比較總量，不假定兩種投影的 cell
    # 位置一一對應，因為弧長座標與水平網格座標是不同離散化。
    grid_field_by_kind = (
        ("local", "local_first_exit_count"),
        ("outer", "outer_first_exit_count"),
    )
    for site_id, site_counts in site_grid_counts.items():
        for boundary_kind, grid_field in grid_field_by_kind:
            grid_total = _safe_sum_int_array(
                getattr(site_counts, grid_field),
                label=f"{grid_field}[{site_id!r}]",
            )
            if grid_total > valid_member_denominator_by_site[site_id]:
                raise ValueError(
                    f"site {site_id!r} 的 {boundary_kind} grid 合計不可大於 "
                    "valid denominator。"
                )
            boundary_total = 0
            for boundary_key, raw_total in boundary_raw_totals.items():
                if (
                    boundary_key.study_site_id == site_id
                    and boundary_key.boundary_kind == boundary_kind
                ):
                    boundary_total += raw_total
            if grid_total != boundary_total:
                raise ValueError(
                    f"site {site_id!r} 的 {boundary_kind} grid 合計必須等於 "
                    "同 kind 所有 boundary raw 合計。"
                )

        data_gap_grid_total = _safe_sum_int_array(
            site_counts.data_gap_failure_count,
            label=f"data_gap_failure_count[{site_id!r}]",
        )
        data_gap_outcome = outcome_count_by_site[site_id].get(
            ParticleStatus.DATA_GAP.value,
            0,
        )
        if data_gap_grid_total != data_gap_outcome:
            raise ValueError(
                f"site {site_id!r} 的 data_gap_failure grid 合計必須等於 "
                "DATA_GAP outcome count。"
            )

        numerical_grid_total = _safe_sum_int_array(
            site_counts.numerical_failure_count,
            label=f"numerical_failure_count[{site_id!r}]",
        )
        numerical_outcome = outcome_count_by_site[site_id].get(
            ParticleStatus.NUMERICAL_FAILURE.value,
            0,
        )
        if numerical_grid_total != numerical_outcome:
            raise ValueError(
                f"site {site_id!r} 的 numerical_failure grid 合計必須等於 "
                "NUMERICAL_FAILURE outcome count。"
            )

        bed_first_total = _safe_sum_int_array(
            site_counts.bed_first_contact_count,
            label=f"bed_first_contact_count[{site_id!r}]",
        )
        if bed_first_total > valid_member_denominator_by_site[site_id]:
            raise ValueError(
                f"site {site_id!r} 的 bed_first_contact_count 合計不可大於 "
                "valid denominator。"
            )

    for cross_key, count_value in cross_site_unique_member_count.items():
        cross_count = _nonnegative_count(
            count_value,
            label=f"cross_site_unique_member_count[{cross_key!r}]",
        )
        source_valid_count = valid_member_denominator_by_site[
            cross_key.source_study_site_id
        ]
        if cross_count > source_valid_count:
            raise ValueError(
                f"cross-site key {cross_key!r} 的 count 不可大於 source site "
                "valid denominator。"
            )


def _revalidate_event_aggregate_chunk(
    chunk: object,
    *,
    index: int,
) -> EventAggregateChunk:
    """以公開欄位重新建構單一 chunk，隔離繞過 dataclass 驗證的資料。

    ``EventAggregateChunk`` 是 frozen dataclass，但 ``object.__setattr__`` 或
    未來的檔案 reader 仍可能在建構後放入錯誤 mapping、可寫陣列或彼此不守恆
    的欄位。merge 不直接信任這些物件，而是逐一讀取其公開欄位並建立新的基底
    ``EventAggregateChunk``；因此會再次觸發 dtype、shape、defensive-copy 與
    跨欄位不變量檢查。所有欄位讀取或重建失敗都保留 ``chunks[index]`` 位置，
    讓正式 release 可以定位壞 shard。
    """

    if not isinstance(chunk, EventAggregateChunk):
        raise ValueError(f"chunks[{index}] 必須是 EventAggregateChunk。")

    try:
        return EventAggregateChunk(
            site_grid_counts=chunk.site_grid_counts,
            boundary_bin_edges_m=chunk.boundary_bin_edges_m,
            boundary_arclength_raw_count=chunk.boundary_arclength_raw_count,
            boundary_travel_age_histogram=chunk.boundary_travel_age_histogram,
            age_bin_edges_seconds=chunk.age_bin_edges_seconds,
            source_receptor_raw_count=chunk.source_receptor_raw_count,
            source_receptor_travel_age_histogram=(
                chunk.source_receptor_travel_age_histogram
            ),
            cross_site_unique_member_count=chunk.cross_site_unique_member_count,
            outcome_count_by_site=chunk.outcome_count_by_site,
            valid_member_denominator_by_site=chunk.valid_member_denominator_by_site,
            total_member_count_by_site=chunk.total_member_count_by_site,
            valid_member_denominator_by_receptor=(
                chunk.valid_member_denominator_by_receptor
            ),
            input_particle_count=chunk.input_particle_count,
        )
    except Exception as error:
        raise ValueError(
            f"chunks[{index}] 的公開欄位缺失或資料不符合 EventAggregateChunk 契約。"
        ) from error


class EventAggregateAccumulator:
    """以單次串流方式累加事件 chunk 的固定拓撲 reducer。

    第一個 ``add`` 會重新建構並完整驗證輸入 chunk，從它封存 site 網格、邊界
    公尺弧長格線／計數、秒制旅行年齡格線、跨站 key 與 outcome key；後續 chunk
    只能沿用這些固定拓撲，source-receptor 與受體分母 key 則可在每個 chunk
    出現時動態聯集。累加器只保留固定大小的公尺制網格與目前的 Python／int64
    計數，並在 ``add`` 返回前釋放重新驗證的 chunk 參照；峰值記憶體是
    ``O(grid topology)``，不會隨 shard 數量線性保存歷史 chunk。

    ``site_grid_counts`` 的陣列軸是 ``(y_cell, x_cell)``；boundary raw 與
    travel histogram 的軸是 ``(s_bin, age_bin)``，其中弧長以公尺、旅行年齡
    以秒表示。所有已存在的 int64 陣列都先以安全的向量化比較預檢，再以
    ``+=`` 寫入；source raw、cross-site、outcome、站點／受體分母與 input
    particle count 以 Python ``int`` 保存，避免固定寬度 scalar 溢位。這些
    結果仍只是條件式來源足跡或相對來源權重的原始累加量，不是絕對來源機率、
    因果歸因或真實 OCM/NWW 科學成果。

    ``add`` 遇到任何重新驗證、拓撲、資料或溢位錯誤後會把 reducer 封閉為
    failed state；即使 caller 捕捉原始例外，也不能把不完整的 accumulator
    繼續當作結果使用。``finalize`` 成功一次後同樣封閉，回傳值會再經完整
    ``EventAggregateChunk`` constructor 驗證與 defensive copy，不共享輸入或
    accumulator 的可寫陣列。
    """

    def __init__(self) -> None:
        """建立尚未收到 chunk 的 reducer，不配置任何 grid 或 histogram。"""

        self._chunk_count = 0
        self._failed = False
        self._finalized = False
        # 這些欄位維持 None 直到第一個合法 chunk；因此建立 accumulator 本身不會
        # 依 spec 盲目配置大網格，也不會在沒有輸入時製造一份可被誤解的零結果。
        self._age_edges: np.ndarray | None = None
        self._site_grid_arrays: dict[str, dict[str, np.ndarray]] | None = None
        self._boundary_edges: dict[BoundaryAggregateKey, np.ndarray] | None = None
        self._boundary_raw: dict[BoundaryAggregateKey, np.ndarray] | None = None
        self._boundary_travel: dict[BoundaryAggregateKey, np.ndarray] | None = None
        self._source_raw: dict[SourceReceptorAggregateKey, int] | None = None
        self._source_travel: dict[SourceReceptorAggregateKey, np.ndarray] | None = None
        self._cross_site: dict[CrossSiteAggregateKey, int] | None = None
        self._outcomes: dict[str, dict[str, int]] | None = None
        self._valid_by_site: dict[str, int] | None = None
        self._total_by_site: dict[str, int] | None = None
        self._valid_by_receptor: dict[ReceptorAggregateKey, int] | None = None
        self._input_particle_count = 0

    @property
    def chunk_count(self) -> int:
        """回傳已成功加入的 chunk 數量；失敗或封閉不會虛增此計數。"""

        return self._chunk_count

    @staticmethod
    def _source_key_sort_key(key: SourceReceptorAggregateKey) -> tuple[str, str, str, str]:
        """沿用公開 merge 的 source-receptor key 排序規則。"""

        return (
            key.study_site_id,
            key.receptor_id,
            key.boundary_kind,
            key.boundary_segment_id,
        )

    @staticmethod
    def _receptor_key_sort_key(key: ReceptorAggregateKey) -> tuple[str, str]:
        """沿用公開 merge 的站點／受體 key 排序規則。"""

        return key.study_site_id, key.receptor_id

    @staticmethod
    def _check_int64_addition(
        total: np.ndarray,
        incoming: np.ndarray,
        *,
        label: str,
    ) -> None:
        """以向量化邊界比較預檢一組 int64 陣列加法是否安全。

        輸入 chunk 已經由 ``EventAggregateChunk`` 驗證為非負 int64，private
        accumulator 也只會由同一條 commit 路徑寫入，因此這裡不再用 Python
        ``np.ndindex`` 逐格掃描。``_INT64_MAX - total`` 在已知 ``total`` 非負
        且為 int64 的前提下不會下溢；若任一 incoming cell 大於該剩餘容量，便
        在任何陣列寫入前拒絕整個 chunk，保留 ``add`` 的 transactional 語意。
        """

        if total.dtype != np.dtype(np.int64) or incoming.dtype != np.dtype(np.int64):
            raise ValueError(f"{label} 必須使用 int64 dtype。")
        if total.shape != incoming.shape:
            raise ValueError(f"{label} 的累加陣列 shape 必須完全相同。")
        if np.any(total < 0) or np.any(incoming < 0):
            raise ValueError(f"{label} 不可包含負值。")
        if np.any(incoming > (_INT64_MAX - total)):
            raise RuntimeError(f"{label} 累加將超過 int64 上限。")

    def _initialize_from(self, chunk: EventAggregateChunk) -> None:
        """由第一個合法 chunk 建立獨立可寫累加狀態，不保存 chunk 本身。"""

        self._age_edges = np.array(chunk.age_bin_edges_seconds, dtype=np.float64, copy=True)
        self._site_grid_arrays = {
            site_id: {
                field: np.array(
                    getattr(site_counts, field),
                    dtype=np.int64,
                    copy=True,
                )
                for field in _SITE_GRID_COUNT_FIELDS
            }
            for site_id, site_counts in chunk.site_grid_counts.items()
        }
        self._boundary_edges = {
            key: np.array(value, dtype=np.float64, copy=True)
            for key, value in chunk.boundary_bin_edges_m.items()
        }
        self._boundary_raw = {
            key: np.array(value, dtype=np.int64, copy=True)
            for key, value in chunk.boundary_arclength_raw_count.items()
        }
        self._boundary_travel = {
            key: np.array(value, dtype=np.int64, copy=True)
            for key, value in chunk.boundary_travel_age_histogram.items()
        }
        self._source_raw = {
            key: int(value) for key, value in chunk.source_receptor_raw_count.items()
        }
        self._source_travel = {
            key: np.array(value, dtype=np.int64, copy=True)
            for key, value in chunk.source_receptor_travel_age_histogram.items()
        }
        self._cross_site = {
            key: int(value) for key, value in chunk.cross_site_unique_member_count.items()
        }
        self._outcomes = {
            site_id: {status: int(count) for status, count in outcomes.items()}
            for site_id, outcomes in chunk.outcome_count_by_site.items()
        }
        self._valid_by_site = {
            site_id: int(value)
            for site_id, value in chunk.valid_member_denominator_by_site.items()
        }
        self._total_by_site = {
            site_id: int(value)
            for site_id, value in chunk.total_member_count_by_site.items()
        }
        self._valid_by_receptor = {
            key: int(value)
            for key, value in chunk.valid_member_denominator_by_receptor.items()
        }
        self._input_particle_count = int(chunk.input_particle_count)

    def _validate_compatible_topology(
        self,
        chunk: EventAggregateChunk,
        *,
        index: int,
    ) -> None:
        """在寫入前確認第二個以上 chunk 的固定拓撲完全相同。"""

        assert self._age_edges is not None
        assert self._site_grid_arrays is not None
        assert self._boundary_edges is not None
        assert self._boundary_raw is not None
        assert self._boundary_travel is not None
        assert self._cross_site is not None
        assert self._outcomes is not None
        if not np.array_equal(chunk.age_bin_edges_seconds, self._age_edges):
            raise ValueError(
                f"chunks[{index}] 的 age_bin_edges_seconds 與 chunks[0] 不一致。"
            )
        if set(chunk.site_grid_counts) != set(self._site_grid_arrays):
            raise ValueError(
                f"chunks[{index}] 的 site_grid_counts site key 集合與 chunks[0] 不一致。"
            )
        for site_id, fields_by_name in self._site_grid_arrays.items():
            chunk_counts = chunk.site_grid_counts[site_id]
            for field in _SITE_GRID_COUNT_FIELDS:
                if getattr(chunk_counts, field).shape != fields_by_name[field].shape:
                    raise ValueError(
                        f"chunks[{index}] site {site_id!r} 的 {field} shape 與 chunks[0] 不一致。"
                    )
        if set(chunk.boundary_bin_edges_m) != set(self._boundary_edges):
            raise ValueError(
                f"chunks[{index}] 的 boundary key 集合與 chunks[0] 不一致。"
            )
        for boundary_key, first_edges in self._boundary_edges.items():
            if not np.array_equal(chunk.boundary_bin_edges_m[boundary_key], first_edges):
                raise ValueError(
                    f"chunks[{index}] boundary key {boundary_key!r} 的公尺格線 "
                    "與 chunks[0] 不一致。"
                )
        if set(chunk.cross_site_unique_member_count) != set(self._cross_site):
            raise ValueError(
                f"chunks[{index}] 的 cross-site key 集合與 chunks[0] 不一致。"
            )
        if set(chunk.outcome_count_by_site) != set(self._outcomes):
            raise ValueError(
                f"chunks[{index}] 的 outcome site key 集合與 chunks[0] 不一致。"
            )
        for site_id, first_outcomes in self._outcomes.items():
            if set(chunk.outcome_count_by_site[site_id]) != set(first_outcomes):
                raise ValueError(
                    f"chunks[{index}] site {site_id!r} 的 outcome status key 集合 "
                    "與 chunks[0] 不一致。"
                )

    def _preflight_int64_additions(
        self,
        chunk: EventAggregateChunk,
        *,
        index: int,
    ) -> None:
        """預檢所有固定陣列與既有 dynamic histogram，避免中途才發現溢位。"""

        assert self._site_grid_arrays is not None
        assert self._boundary_raw is not None
        assert self._boundary_travel is not None
        assert self._source_travel is not None
        for site_id, accumulator_fields in self._site_grid_arrays.items():
            chunk_counts = chunk.site_grid_counts[site_id]
            for field, total in accumulator_fields.items():
                self._check_int64_addition(
                    total,
                    getattr(chunk_counts, field),
                    label=f"site grid chunks[{index}] {site_id!r}.{field}",
                )
        for boundary_key, total in self._boundary_raw.items():
            self._check_int64_addition(
                total,
                chunk.boundary_arclength_raw_count[boundary_key],
                label=f"boundary raw chunks[{index}] {boundary_key!r} (m)",
            )
            self._check_int64_addition(
                self._boundary_travel[boundary_key],
                chunk.boundary_travel_age_histogram[boundary_key],
                label=f"boundary travel histogram chunks[{index}] {boundary_key!r} (m, s)",
            )
        for source_key, incoming in chunk.source_receptor_travel_age_histogram.items():
            current = self._source_travel.get(source_key)
            if current is not None:
                self._check_int64_addition(
                    current,
                    incoming,
                    label=f"source-receptor travel histogram chunks[{index}] "
                    f"{source_key!r} (s)",
                )

    def _accumulate_validated_chunk(
        self,
        chunk: EventAggregateChunk,
    ) -> None:
        """把已完成 topology／overflow 預檢的 chunk 寫入 mutable accumulator。

        呼叫端已先完成整個 chunk 的所有 int64 預檢，因此這裡的 ``+=`` 不會
        觸發固定寬度整數繞回，也不必再呼叫保留給其他 API 的逐格 helper。新
        source-receptor key 直接複製輸入 histogram；既有 key 才做向量化 in-place
        加法。這個 commit 階段不再進行可能在中途失敗的範圍檢查。
        """

        assert self._site_grid_arrays is not None
        assert self._boundary_raw is not None
        assert self._boundary_travel is not None
        assert self._source_raw is not None
        assert self._source_travel is not None
        assert self._cross_site is not None
        assert self._outcomes is not None
        assert self._valid_by_site is not None
        assert self._total_by_site is not None
        assert self._valid_by_receptor is not None

        for site_id, accumulator_fields in self._site_grid_arrays.items():
            chunk_counts = chunk.site_grid_counts[site_id]
            for field, total in accumulator_fields.items():
                total += getattr(chunk_counts, field)
        for boundary_key, total in self._boundary_raw.items():
            total += chunk.boundary_arclength_raw_count[boundary_key]
            self._boundary_travel[boundary_key] += (
                chunk.boundary_travel_age_histogram[boundary_key]
            )

        for source_key, incoming_raw in chunk.source_receptor_raw_count.items():
            if source_key not in self._source_raw:
                self._source_raw[source_key] = int(incoming_raw)
                self._source_travel[source_key] = np.array(
                    chunk.source_receptor_travel_age_histogram[source_key],
                    dtype=np.int64,
                    copy=True,
                )
                continue
            self._source_raw[source_key] += int(incoming_raw)
            self._source_travel[source_key] += (
                chunk.source_receptor_travel_age_histogram[source_key]
            )

        for cross_key in self._cross_site:
            self._cross_site[cross_key] += int(
                chunk.cross_site_unique_member_count[cross_key]
            )
        for site_id in self._site_grid_arrays:
            self._valid_by_site[site_id] += int(
                chunk.valid_member_denominator_by_site[site_id]
            )
            self._total_by_site[site_id] += int(
                chunk.total_member_count_by_site[site_id]
            )
            for status in self._outcomes[site_id]:
                self._outcomes[site_id][status] += int(
                    chunk.outcome_count_by_site[site_id][status]
                )
        for receptor_key, incoming in chunk.valid_member_denominator_by_receptor.items():
            if receptor_key not in self._valid_by_receptor:
                self._valid_by_receptor[receptor_key] = 0
            self._valid_by_receptor[receptor_key] += int(incoming)
        self._input_particle_count += int(chunk.input_particle_count)

    def add(self, chunk: EventAggregateChunk) -> None:
        """重新驗證並加入單一 chunk；失敗後 reducer 進入不可繼續的 failed state。

        先建立當前 chunk 的獨立 ``EventAggregateChunk``，再做所有固定拓撲與
        int64 overflow 預檢；直到這些檢查全部通過才改寫 accumulator。若任一
        階段失敗，原始例外型別與訊息會原樣傳回，但 reducer 會標記 failed，
        caller 不可捕捉後繼續累加不完整的資料。``chunk`` 只代表一個已完成
        shard 的事件統計，不含粒子 ID 去重資訊；shard 互斥性仍由上游 run
        plan／validator 負責證明。
        """

        if self._failed or self._finalized:
            raise ValueError("EventAggregateAccumulator 已封閉，不能再加入 chunk。")
        index = self._chunk_count
        try:
            validated = _revalidate_event_aggregate_chunk(chunk, index=index)
            if self._chunk_count == 0:
                self._initialize_from(validated)
            else:
                self._validate_compatible_topology(validated, index=index)
                self._preflight_int64_additions(validated, index=index)
                self._accumulate_validated_chunk(validated)
        except Exception:
            self._failed = True
            raise
        self._chunk_count += 1

    def _build_result(self) -> EventAggregateChunk:
        """以固定／排序後 key 建立最終 chunk，交由 constructor 做最後驗證。"""

        assert self._age_edges is not None
        assert self._site_grid_arrays is not None
        assert self._boundary_edges is not None
        assert self._boundary_raw is not None
        assert self._boundary_travel is not None
        assert self._source_raw is not None
        assert self._source_travel is not None
        assert self._cross_site is not None
        assert self._outcomes is not None
        assert self._valid_by_site is not None
        assert self._total_by_site is not None
        assert self._valid_by_receptor is not None

        site_grid_counts = {
            site_id: SiteEventGridCounts(
                **{
                    field: np.array(array, dtype=np.int64, copy=True)
                    for field, array in fields_by_name.items()
                }
            )
            for site_id, fields_by_name in self._site_grid_arrays.items()
        }
        source_keys = sorted(self._source_raw, key=self._source_key_sort_key)
        receptor_keys = sorted(self._valid_by_receptor, key=self._receptor_key_sort_key)
        return EventAggregateChunk(
            site_grid_counts=site_grid_counts,
            boundary_bin_edges_m={
                key: np.array(value, dtype=np.float64, copy=True)
                for key, value in self._boundary_edges.items()
            },
            boundary_arclength_raw_count={
                key: np.array(self._boundary_raw[key], dtype=np.int64, copy=True)
                for key in self._boundary_raw
            },
            boundary_travel_age_histogram={
                key: np.array(self._boundary_travel[key], dtype=np.int64, copy=True)
                for key in self._boundary_travel
            },
            age_bin_edges_seconds=np.array(self._age_edges, dtype=np.float64, copy=True),
            source_receptor_raw_count={key: int(self._source_raw[key]) for key in source_keys},
            source_receptor_travel_age_histogram={
                key: np.array(self._source_travel[key], dtype=np.int64, copy=True)
                for key in source_keys
            },
            cross_site_unique_member_count={
                key: int(value) for key, value in self._cross_site.items()
            },
            outcome_count_by_site={
                site_id: {status: int(count) for status, count in outcomes.items()}
                for site_id, outcomes in self._outcomes.items()
            },
            valid_member_denominator_by_site={
                site_id: int(value) for site_id, value in self._valid_by_site.items()
            },
            total_member_count_by_site={
                site_id: int(value) for site_id, value in self._total_by_site.items()
            },
            valid_member_denominator_by_receptor={
                key: int(self._valid_by_receptor[key]) for key in receptor_keys
            },
            input_particle_count=int(self._input_particle_count),
        )

    def finalize(self) -> EventAggregateChunk:
        """封存成功累加結果；零 chunk、失敗或重複 finalize 都會拒絕。"""

        if self._failed or self._finalized:
            raise ValueError("EventAggregateAccumulator 已封閉，不能 finalize。")
        if self._chunk_count == 0:
            raise ValueError("chunks 不可為空。")
        try:
            result = self._build_result()
        except Exception:
            self._failed = True
            raise
        # result 已由 EventAggregateChunk constructor 完成最後一次 defensive
        # copy；此刻釋放 reducer 自己持有的網格、histogram 與 mapping，讓正式
        # pipeline 在 result 仍留在 scope 時不會同時保留一份大型 mutable state。
        self._release_state()
        self._finalized = True
        return result

    def _release_state(self) -> None:
        """釋放 finalize 後不再可讀取的大型 private array／mapping state。"""

        self._age_edges = None
        self._site_grid_arrays = None
        self._boundary_edges = None
        self._boundary_raw = None
        self._boundary_travel = None
        self._source_raw = None
        self._source_travel = None
        self._cross_site = None
        self._outcomes = None
        self._valid_by_site = None
        self._total_by_site = None
        self._valid_by_receptor = None
        self._input_particle_count = 0


def merge_event_aggregate_chunks(
    chunks: Iterable[EventAggregateChunk],
) -> EventAggregateChunk:
    """以單次 iterator 串流合併事件聚合 chunk，回傳新的不可變聚合結果。

    Args:
        chunks: 可逐筆迭代的 ``EventAggregateChunk``；每個 chunk 代表一組已完成
            粒子結果。函式不會把 iterator materialize 成歷史 chunk 清單，而是
            在讀到每個元素後立即重新驗證並加入固定大小的 accumulator。

    Returns:
        一個不與任何輸入陣列或 mapping 共享記憶體的 ``EventAggregateChunk``。
        site 事件網格的軸順序是 ``(y_cell, x_cell)``；邊界旅行統計的軸順序
        是 ``(s_bin, age_bin)``，其中 ``s`` 以公尺表示、age 以秒表示；
        source-receptor 旅行直方圖則只有秒制 age 軸。所有固定拓撲欄位沿用
        ``chunks[0]``，source-receptor 與受體有效分母則取所有 chunk 的 key
        聯集，缺少的 shard/key 視為零；兩個聯集都依公開字串欄位排序，因而不
        受輸入 chunk 或 mapping 插入順序影響。

    Raises:
        ValueError: ``chunks`` 為空、元素不是合法 ``EventAggregateChunk``、
            公開欄位無法重新建構，或固定拓撲／軸線不一致時。
        RuntimeError: 任一 int64 網格、弧長 raw、旅行年齡 histogram 或
            source-receptor age histogram 的逐元素加總會超過 int64 上限時。

    Notes:
        site grid、boundary raw/hist 與 source-receptor age histogram 使用逐值
        向量化比較預檢 ``incoming > INT64_MAX - total``，確認整個 chunk 安全後
        以 in-place ``+=`` 寫回 int64，因而不會讓 NumPy 固定寬度加法繞回。
        標量 outcome、各站分母、受體分母、跨站計數與 input particle count 以
        Python 任意精度整數相加；最後再以 ``EventAggregateChunk`` 重建，讓所有
        站點、邊界、source-receptor、failure 與 outcome 守恆關係再次 fail closed。

        本函式不保存 particle ID，因此無法自行偵測不同 chunk 間的重複粒子，
        也不執行 I/O、核密度估計（KDE）、ratio 或 bootstrap。正式 release 的
        caller 必須先以不可變 run plan 與 shard identity 證明各 chunk 彼此
        互斥且完整；否則合併值只能視為可能重複的條件式來源足跡／相對來源
        權重原始累加量，不得解讀為絕對來源機率或因果歸因。邊界的公尺弧長、
        網格的公尺座標及旅行 age 的秒數單位均不在 merge 中重新投影或換算。
    """

    try:
        iterator = iter(chunks)
    except Exception as error:
        raise ValueError("chunks 必須是可迭代資料。") from error

    accumulator = EventAggregateAccumulator()
    while True:
        try:
            chunk = next(iterator)
        except StopIteration:
            break
        except Exception as error:
            raise ValueError("chunks iterator 無法讀取。") from error
        # add 的 ValueError／RuntimeError 必須原樣傳回，不能被 iterator 的
        # 例外包裝邊界吞掉；因此呼叫刻意位於 try/except 之外。
        accumulator.add(chunk)
    return accumulator.finalize()


def _boundary_edges_for_segment(
    length_m: int | float,
    bin_size_m: int | float,
) -> np.ndarray:
    """依邊界段長度與 bin 尺寸建立從 0 m 到精確終點的格線。

    先產生所有完整的 ``boundary_bin_size_m`` 公尺區段，再把尚未涵蓋的尾端
    以一個短 bin 收到精確長度；若長度在浮點誤差內剛好整除，則只保留一次
    終點。最後格線一定使用輸入長度轉成 float64 的值，避免最後一點因乘法
    累積誤差偏離規格的 segment length。
    """

    try:
        length = float(length_m)
        bin_size = float(bin_size_m)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("邊界長度與 boundary bin 尺寸必須可轉為 float64。") from error
    if not math.isfinite(length) or not math.isfinite(bin_size):
        raise ValueError("邊界長度與 boundary bin 尺寸必須是有限值。")
    if length <= 0.0 or bin_size <= 0.0:
        raise ValueError("邊界長度與 boundary bin 尺寸必須大於零。")

    try:
        ratio = length / bin_size
    except (OverflowError, ZeroDivisionError) as error:
        raise ValueError("邊界長度無法換算為 bin 數。") from error
    if not math.isfinite(ratio):
        raise ValueError("邊界長度換算出的 bin 數必須是有限值。")

    nearest_integer = round(ratio)
    if nearest_integer >= 1 and math.isclose(
        ratio,
        nearest_integer,
        rel_tol=_EXACT_BIN_RATIO_REL_TOL,
        abs_tol=_EXACT_BIN_RATIO_ABS_TOL,
    ):
        full_bin_count = nearest_integer
    else:
        full_bin_count = math.floor(ratio)

    # 正常規格下 full_bin_count 很小；若輸入導致無法配置有限數量的格線，
    # 讓 NumPy/記憶體錯誤在初始化邊界前轉成明確 ValueError，而不產生半成品。
    try:
        edges_values = [float(index * bin_size) for index in range(full_bin_count + 1)]
    except (OverflowError, MemoryError) as error:
        raise ValueError("邊界 bin 數過大，無法建立弧長格線。") from error

    if not edges_values or edges_values[-1] < length:
        edges_values.append(length)
    elif edges_values[-1] > length and not math.isclose(
        edges_values[-1],
        length,
        rel_tol=_EXACT_BIN_RATIO_REL_TOL,
        abs_tol=_EXACT_BIN_RATIO_ABS_TOL,
    ):
        # 只有在浮點乘法超過終點且偏差已超出整除判定容差時才失敗；
        # 接近終點的情況會在下一行固定成精確的 segment length。
        raise ValueError("邊界 bin 格線超出 segment length。")
    edges_values[-1] = length

    edges = np.asarray(edges_values, dtype=np.float64)
    if edges.size < 2 or np.any(np.diff(edges) <= 0.0):
        raise ValueError("邊界 bin 格線必須形成嚴格遞增的有效區間。")
    edges.setflags(write=False)
    return edges


def _grid_shape(spec: AggregateSpec, site_id: str) -> tuple[int, int]:
    """由站點公尺制矩形與網格尺寸取得 ``(y_cell, x_cell)`` 形狀。

    ``AggregateSpec`` 已驗證兩軸範圍能由整數個 cell 覆蓋，因此此處只做
    最終的整數化，不進行座標裁切、平移或四捨五入修補。這保持陣列索引與
    公尺制計算座標一致；任何無法建立正整數形狀的異常都會中止初始化。
    """

    grid = spec.site_grids[site_id]
    try:
        x_count = int(
            round((grid.x_max_m - grid.x_min_m) / spec.grid_cell_size_m)
        )
        y_count = int(
            round((grid.y_max_m - grid.y_min_m) / spec.grid_cell_size_m)
        )
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"站點 {site_id} 無法轉成整數網格形狀。") from error
    if x_count <= 0 or y_count <= 0:
        raise ValueError(f"站點 {site_id} 的網格形狀必須為正整數。")
    return y_count, x_count


def initialize_event_aggregate(
    spec: AggregateSpec,
    *,
    age_bin_edges_seconds: np.ndarray,
) -> EventAggregateChunk:
    """依彙整規格建立尚未累加任何成員的事件聚合塊。

    Args:
        spec: 已驗證的 :class:`AggregateSpec`。其站點矩形與網格尺寸以公尺
            表示，邊界段長度與弧長 bin 尺寸也以公尺表示；每個站點會建立
            ``(height / cell, width / cell)`` 的六類事件網格陣列，軸順序是
            ``(y_cell, x_cell)``。
        age_bin_edges_seconds: 以秒表示的年齡箱邊界。必須先通過本模組的一維、
            有限、嚴格遞增與 0 s 起點驗證，並且必須以 ``np.array_equal``
            完全等於 ``spec.age_bin_edges_seconds``；正式 age 軸由
            ``AggregateSpec`` 鎖定，旅行年齡直方圖的 age 軸長度是邊界數減一。

    Returns:
        所有計數為零的 ``EventAggregateChunk``。每個站點的網格、valid 分母、
        total 成員數與 outcome 外層 mapping 都預先存在；本輪尚未執行粒子／
        事件累加，因此 source-receptor、跨站與受體層級細項保持空 mapping。
        同一邊界段跨站或跨 local/outer 角色時，仍各自建立 key，但共用同一
        個唯讀邊界格線副本。``input_particle_count`` 為 0。

    Raises:
        ValueError: ``spec`` 型別、年齡格線、網格形狀、邊界長度或尺寸不符合
            資料契約時。

    Notes:
        零值只表示聚合容器尚未收到事件，不能把資料缺口、陸地、乾點、域外、
        海底以下或數值失敗替換成零。完成後的結果仍只能描述條件式來源足跡
        或相對來源權重，不能直接稱為絕對來源機率或因果歸因。
    """

    if not isinstance(spec, AggregateSpec):
        raise ValueError("spec 必須是 AggregateSpec。")
    age_edges = _validate_age_bin_edges(age_bin_edges_seconds)
    spec_age_edges = np.asarray(spec.age_bin_edges_seconds, dtype=np.float64)
    if not np.array_equal(age_edges, spec_age_edges):
        raise ValueError(
            "age_bin_edges_seconds 必須與 spec.age_bin_edges_seconds 完全相等；"
            "正式 age 軸由 AggregateSpec 鎖定。"
        )
    age_bin_count = age_edges.size - 1

    site_grid_counts: dict[str, SiteEventGridCounts] = {}
    outcome_count_by_site: dict[str, Mapping[str, int]] = {}
    valid_member_denominator_by_site: dict[str, int] = {}
    total_member_count_by_site: dict[str, int] = {}

    # 先固定站點零拓撲；每個事件類別使用獨立陣列，避免後續累加一類事件
    # 時意外改寫其他類別。陣列軸以 y 在前、x 在後，與矩形高度／寬度一致。
    for site_id in spec.site_grids:
        shape = _grid_shape(spec, site_id)
        site_grid_counts[site_id] = SiteEventGridCounts(
            local_first_exit_count=np.zeros(shape, dtype=np.int64),
            outer_first_exit_count=np.zeros(shape, dtype=np.int64),
            bed_first_contact_count=np.zeros(shape, dtype=np.int64),
            bed_repeated_contact_count=np.zeros(shape, dtype=np.int64),
            data_gap_failure_count=np.zeros(shape, dtype=np.int64),
            numerical_failure_count=np.zeros(shape, dtype=np.int64),
        )
        # 外層 site key 先存在，內層 outcome 細項留給後續累加切片填入。
        outcome_count_by_site[site_id] = {}
        valid_member_denominator_by_site[site_id] = 0
        total_member_count_by_site[site_id] = 0

    boundary_bin_edges_m: dict[BoundaryAggregateKey, np.ndarray] = {}
    boundary_arclength_raw_count: dict[BoundaryAggregateKey, np.ndarray] = {}
    boundary_travel_age_histogram: dict[BoundaryAggregateKey, np.ndarray] = {}
    edges_by_segment_id: dict[str, np.ndarray] = {}

    # segment_id 是全域邊界長度表的識別碼；以它快取格線，讓同一段在多站點或
    # local/outer 兩角色出現時共用同一唯讀 edges 物件，但 key 仍完全分開。
    for site_id, segment_groups in spec.site_boundary_segment_ids.items():
        for boundary_kind, segment_ids in (
            ("local", segment_groups.local_segment_ids),
            ("outer", segment_groups.outer_segment_ids),
        ):
            for segment_id in segment_ids:
                edges = edges_by_segment_id.get(segment_id)
                if edges is None:
                    edges = _boundary_edges_for_segment(
                        spec.boundary_segment_lengths_m[segment_id],
                        spec.boundary_bin_size_m,
                    )
                    edges_by_segment_id[segment_id] = edges

                key = BoundaryAggregateKey(
                    study_site_id=site_id,
                    boundary_kind=boundary_kind,
                    boundary_segment_id=segment_id,
                )
                s_bin_count = edges.size - 1
                boundary_bin_edges_m[key] = edges
                boundary_arclength_raw_count[key] = np.zeros(
                    s_bin_count,
                    dtype=np.int64,
                )
                boundary_travel_age_histogram[key] = np.zeros(
                    (s_bin_count, age_bin_count),
                    dtype=np.int64,
                )

    return EventAggregateChunk(
        site_grid_counts=site_grid_counts,
        boundary_bin_edges_m=boundary_bin_edges_m,
        boundary_arclength_raw_count=boundary_arclength_raw_count,
        boundary_travel_age_histogram=boundary_travel_age_histogram,
        age_bin_edges_seconds=age_edges,
        source_receptor_raw_count={},
        source_receptor_travel_age_histogram={},
        cross_site_unique_member_count={},
        outcome_count_by_site=outcome_count_by_site,
        valid_member_denominator_by_site=valid_member_denominator_by_site,
        total_member_count_by_site=total_member_count_by_site,
        valid_member_denominator_by_receptor={},
        input_particle_count=0,
    )


def _finite_real_scalar(value: object, *, label: str) -> float:
    """把單一物理量驗證為有限的 float64 數值。

    事件座標、深度、年齡、交點比例與邊界弧長都必須是數值標量；布林、字串、
    複數、陣列、NaN 與無限值不能被靜默轉成可計數資料。統一轉成 Python
    ``float`` 後，後續格網索引與年齡分箱會使用同一套精度與比較規則。
    """

    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} 必須是有限數值標量。") from error
    if raw.ndim != 0 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{label} 必須是有限數值標量。")
    try:
        number = float(raw)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} 必須是有限數值標量。") from error
    if not np.isfinite(number):
        raise ValueError(f"{label} 必須是有限數值標量。")
    return number


def _utc_nanosecond(value: object, *, label: str) -> int:
    """驗證 UTC 奈秒時間是整數並轉成不失去精度的 Python ``int``。

    時間比較不能先轉成 float，因為奈秒時間通常大於 2^53，浮點化會讓本來不同
    的 observation 或 event 看起來相同。接受 Python 整數及 NumPy 整數 scalar，
    拒絕浮點、布林與其他非整數物件；負值仍可合法表示 Unix epoch 以前的 UTC。
    """

    if type(value) is int:
        return value
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} 必須是 UTC 奈秒整數。") from error
    if raw.ndim != 0 or raw.dtype.kind not in "iu":
        raise ValueError(f"{label} 必須是 UTC 奈秒整數。")
    try:
        return int(raw.item())
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} 必須是 UTC 奈秒整數。") from error


def _grid_edges_for_site(
    spec: AggregateSpec,
    site_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    """依站點矩形與公尺制 cell 尺寸建立 ``(x_edges, y_edges)``。

    ``AggregateSpec`` 已驗證每個矩形軸長度接近整數個 cell；這裡仍把最後一個
    邊界直接固定成 spec 的 ``x_max_m``／``y_max_m``，避免乘法累積誤差讓合法的
    外框座標落在最後 cell 之外。回傳格線只供目前聚合呼叫使用，真正的結果會
    保存 grid count 而不保存這兩組暫時格線。
    """

    grid = spec.site_grids[site_id]
    y_count, x_count = _grid_shape(spec, site_id)
    cell_size = float(spec.grid_cell_size_m)
    try:
        x_edges = np.arange(x_count + 1, dtype=np.float64) * cell_size + float(
            grid.x_min_m
        )
        y_edges = np.arange(y_count + 1, dtype=np.float64) * cell_size + float(
            grid.y_min_m
        )
    except (MemoryError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"站點 {site_id} 無法建立公尺制網格格線。") from error

    # 末端直接採用規格值，保證 x_max/y_max 是閉矩形的一部分，而不是被
    # 浮點乘法產生的最後幾位誤差誤判為域外。
    x_edges[-1] = float(grid.x_max_m)
    y_edges[-1] = float(grid.y_max_m)
    if (
        x_edges.ndim != 1
        or y_edges.ndim != 1
        or not np.all(np.isfinite(x_edges))
        or not np.all(np.isfinite(y_edges))
        or np.any(np.diff(x_edges) <= 0.0)
        or np.any(np.diff(y_edges) <= 0.0)
    ):
        raise ValueError(f"站點 {site_id} 的公尺制網格格線無效。")
    return x_edges, y_edges


def _grid_cell_index(value: float, edges: np.ndarray, *, label: str) -> int:
    """把閉矩形內的座標轉成 cell index，右側外框歸入最後 cell。

    一般 cell 採 ``[left, right)``；因此落在內部格線上的座標使用
    ``searchsorted(..., side="right")`` 歸到右側 cell。最右／最上外框是閉區間
    的特例，明確歸入最後 cell。所有域外與非有限值都立即失敗，不以 clip 或
    丟棄方式掩蓋輸入與事件座標錯誤。
    """

    if not np.isfinite(value) or value < edges[0] or value > edges[-1]:
        raise ValueError(f"{label} 座標超出 spec 公尺制閉矩形。")
    if value == edges[-1]:
        return edges.size - 2
    index = int(np.searchsorted(edges, value, side="right") - 1)
    if index < 0 or index >= edges.size - 1:
        raise ValueError(f"{label} 座標無法對應 spec 公尺制 cell。")
    return index


def _age_bin_index(age_seconds: float, edges: np.ndarray, *, label: str) -> int:
    """依年齡秒數格線找 bin，並將最後邊界納入最後一箱。

    一般箱是左閉右開，內部邊界上的事件歸到右側箱；最後一箱額外包含精確等於
    最後 edge 的年齡。超出 ``[0, edges[-1]]`` 的旅行年齡直接拒絕，避免用
    ``clip`` 把設定不足或事件時間錯誤藏起來。
    """

    if (
        not np.isfinite(age_seconds)
        or age_seconds < edges[0]
        or age_seconds > edges[-1]
    ):
        raise ValueError(f"{label} 必須落在 age_bin_edges_seconds 的閉範圍內。")
    if age_seconds == edges[-1]:
        return edges.size - 2
    index = int(np.searchsorted(edges, age_seconds, side="right") - 1)
    if index < 0 or index >= edges.size - 1:
        raise ValueError(f"{label} 無法對應 age bin。")
    return index


def _event_age_seconds(
    arrival_time_utc_ns: int,
    event_time_utc_ns: int,
) -> float:
    """依到達時刻與 event UTC 奈秒時間計算回溯旅行年齡。

    年齡是 ``(arrival - event) / 1e9``，不是沿 observation 累加得到的近似值；
    這確保事件直方圖直接遵守 Scenario 的到達時刻契約。整數差先在 Python
    整數中完成，再轉成 float64 並檢查有限性，避免奈秒差值先發生固定寬度溢位。
    """

    difference_ns = arrival_time_utc_ns - event_time_utc_ns
    try:
        age_seconds = float(difference_ns) / 1.0e9
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("event travel age 無法轉成有限秒數。") from error
    if not np.isfinite(age_seconds):
        raise ValueError("event travel age 必須是有限秒數。")
    return age_seconds


def _increment_int64(
    array: np.ndarray,
    index: tuple[int, ...],
    *,
    label: str,
) -> None:
    """在 int64 陣列寫入前檢查上限，拒絕計數繞回負值。

    NumPy 固定寬度整數的直接 ``+= 1`` 可能在最大值後靜默變成負數；所有事件
    grid、弧長 raw count 與旅行年齡 histogram 都經過此函式逐次累加，因此
    發生容量不足時會以 RuntimeError fail closed，而不是回傳損壞的統計。
    """

    current = int(array[index])
    if current < 0 or current >= _INT64_MAX:
        raise RuntimeError(f"{label} 累加將超過 int64 上限。")
    array[index] = current + 1


def _validate_scenarios_by_id(
    scenarios_by_id: object,
    *,
    spec: AggregateSpec,
) -> dict[str, Scenario]:
    """驗證情境 mapping，並建立不受 caller 改動影響的字典快照。

    情境 ID 是 ParticleResult、Scenario 與受體分母的連接鍵；mapping key 與
    ``Scenario.scenario_id`` 不一致時，即使其他欄位看似合法也不能繼續。站點、
    region 與 receptor 是後續輸出 key 的必要識別欄位，先確認為非空字串，並
    確認每個情境站點確實存在於本次 AggregateSpec。
    """

    if not isinstance(scenarios_by_id, Mapping):
        raise ValueError("scenarios_by_id 必須是 mapping。")
    try:
        items = tuple(scenarios_by_id.items())
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("scenarios_by_id 無法穩定讀取。") from error

    validated: dict[str, Scenario] = {}
    for mapping_key, scenario in items:
        key = _require_nonempty_string(
            mapping_key,
            label="scenarios_by_id key",
        )
        if not isinstance(scenario, Scenario):
            raise ValueError("scenarios_by_id 的 value 必須是 Scenario。")
        scenario_id = _require_nonempty_string(
            scenario.scenario_id,
            label="Scenario.scenario_id",
        )
        if key != scenario_id:
            raise ValueError("scenarios_by_id key 必須與 Scenario.scenario_id 完全一致。")
        site_id = _require_nonempty_string(
            scenario.study_site_id,
            label="Scenario.study_site_id",
        )
        if site_id not in spec.site_grids:
            raise ValueError("Scenario.study_site_id 必須存在於 spec。")
        _require_nonempty_string(
            scenario.analysis_region_id,
            label="Scenario.analysis_region_id",
        )
        _require_nonempty_string(
            scenario.receptor_id,
            label="Scenario.receptor_id",
        )
        _utc_nanosecond(
            scenario.arrival_time_utc_ns,
            label="Scenario.arrival_time_utc_ns",
        )
        validated[key] = scenario
    return validated


def _validate_particle_result(
    result: object,
    *,
    scenarios_by_id: Mapping[str, Scenario],
    spec: AggregateSpec,
) -> tuple[
    Scenario,
    ParticleState,
    tuple[Observation, ...],
    tuple[BoundaryEvent, ...],
    ParticleStatus,
    int,
]:
    """完整驗證單一粒子結果並回傳供累加使用的固定 tuple。

    驗證範圍涵蓋粒子／情境 identity、成員編號、觀測年齡與 UTC 方向、終點一致性、
    event identity、event 時間範圍、座標與比例有限性，以及 terminal status 對應
    的唯一 terminal event。函式不修改輸入的 list 或 dataclass；回傳的 tuple 只在
    目前呼叫期間使用，仍保留原始 immutable ``ParticleState``、``Observation`` 與
    ``BoundaryEvent`` 物件，以便後續按照事件類型更新不同統計。
    """

    if not isinstance(result, ParticleResult):
        raise ValueError("每個 result 必須是 ParticleResult。")
    final_state = result.final_state
    if not isinstance(final_state, ParticleState):
        raise ValueError("ParticleResult.final_state 必須是 ParticleState。")

    particle_id = _require_nonempty_string(
        final_state.particle_id,
        label="final_state.particle_id",
    )
    scenario_id = _require_nonempty_string(
        final_state.scenario_id,
        label="final_state.scenario_id",
    )
    try:
        scenario = scenarios_by_id[scenario_id]
    except (KeyError, TypeError) as error:
        raise ValueError("ParticleResult 的 scenario_id 不存在於 scenarios_by_id。") from error

    final_site_id = _require_nonempty_string(
        final_state.study_site_id,
        label="final_state.study_site_id",
    )
    final_region_id = _require_nonempty_string(
        final_state.analysis_region_id,
        label="final_state.analysis_region_id",
    )
    final_receptor_id = _require_nonempty_string(
        final_state.receptor_id,
        label="final_state.receptor_id",
    )
    if (
        final_state.scenario_id != scenario.scenario_id
        or final_site_id != scenario.study_site_id
        or final_region_id != scenario.analysis_region_id
        or final_receptor_id != scenario.receptor_id
    ):
        raise ValueError("ParticleResult 的 scenario/site/region/receptor 與 Scenario 不一致。")

    member_id = _nonnegative_count(
        final_state.member_id,
        label="final_state.member_id",
    )
    final_status = final_state.status
    if not isinstance(final_status, ParticleStatus) or final_status == ParticleStatus.ACTIVE:
        raise ValueError("ParticleResult.final_state.status 必須是非 ACTIVE 的 ParticleStatus。")
    expected_terminal_event = _TERMINAL_EVENT_TYPE_BY_STATUS.get(final_status)
    if expected_terminal_event is None:
        raise ValueError("ParticleStatus 沒有登錄的 terminal EventType。")

    final_x = _finite_real_scalar(final_state.x_m, label="final_state.x_m")
    final_y = _finite_real_scalar(final_state.y_m, label="final_state.y_m")
    final_z = _finite_real_scalar(final_state.z_m, label="final_state.z_m")
    final_time = _utc_nanosecond(
        final_state.time_utc_ns,
        label="final_state.time_utc_ns",
    )
    final_age = _finite_real_scalar(
        final_state.age_seconds,
        label="final_state.age_seconds",
    )
    if final_age < 0.0:
        raise ValueError("final_state.age_seconds 不可為負。")
    arrival_time = _utc_nanosecond(
        scenario.arrival_time_utc_ns,
        label="Scenario.arrival_time_utc_ns",
    )

    try:
        observations = tuple(result.observations)
    except (TypeError, ValueError) as error:
        raise ValueError("每個 result 的 observations 必須是非空序列。") from error
    if not observations:
        raise ValueError("每個 result 至少需要一筆 observation。")

    normalized_observation_values: list[tuple[int, float, float, float, float, ParticleStatus]] = []
    previous_age: float | None = None
    previous_time: int | None = None
    for observation in observations:
        if not isinstance(observation, Observation):
            raise ValueError("result.observations 的元素必須是 Observation。")
        observation_id = _require_nonempty_string(
            observation.particle_id,
            label="observation.particle_id",
        )
        if observation_id != particle_id:
            raise ValueError("所有 observation 必須與 final_state 使用相同 particle_id。")
        observation_time = _utc_nanosecond(
            observation.time_utc_ns,
            label="observation.time_utc_ns",
        )
        observation_age = _finite_real_scalar(
            observation.age_seconds,
            label="observation.age_seconds",
        )
        if observation_age < 0.0:
            raise ValueError("observation.age_seconds 不可為負。")
        observation_x = _finite_real_scalar(
            observation.x_m,
            label="observation.x_m",
        )
        observation_y = _finite_real_scalar(
            observation.y_m,
            label="observation.y_m",
        )
        observation_z = _finite_real_scalar(
            observation.z_m,
            label="observation.z_m",
        )
        observation_status = observation.status
        if not isinstance(observation_status, ParticleStatus):
            raise ValueError("observation.status 必須是 ParticleStatus。")
        if previous_age is None:
            if observation_age != 0.0:
                raise ValueError("第一筆 observation.age_seconds 必須從 0 開始。")
            if observation_time != arrival_time:
                raise ValueError("第一筆 observation.time_utc_ns 必須等於 Scenario arrival time。")
        else:
            if not observation_age > previous_age:
                raise ValueError("observation.age_seconds 必須嚴格增加。")
            if previous_time is None or not previous_time > observation_time:
                raise ValueError("observation.time_utc_ns 必須嚴格倒退。")
        normalized_observation_values.append(
            (
                observation_time,
                observation_age,
                observation_x,
                observation_y,
                observation_z,
                observation_status,
            )
        )
        previous_age = observation_age
        previous_time = observation_time

    last_observation = normalized_observation_values[-1]
    if (
        last_observation[0] != final_time
        or last_observation[1] != final_age
        or last_observation[2] != final_x
        or last_observation[3] != final_y
        or last_observation[4] != final_z
        or last_observation[5] != final_status
    ):
        raise ValueError("最後 observation 必須與 final_state 完全一致。")
    if final_time > arrival_time:
        raise ValueError("final_state.time_utc_ns 不可晚於 Scenario arrival time。")

    try:
        events = tuple(result.events)
    except (TypeError, ValueError) as error:
        raise ValueError("每個 result 的 events 必須是序列。") from error

    terminal_events: list[BoundaryEvent] = []
    for event in events:
        if not isinstance(event, BoundaryEvent):
            raise ValueError("result.events 的元素必須是 BoundaryEvent。")
        event_id = _require_nonempty_string(
            event.particle_id,
            label="event.particle_id",
        )
        event_scenario_id = _require_nonempty_string(
            event.scenario_id,
            label="event.scenario_id",
        )
        event_site_id = _require_nonempty_string(
            event.study_site_id,
            label="event.study_site_id",
        )
        event_region_id = _require_nonempty_string(
            event.analysis_region_id,
            label="event.analysis_region_id",
        )
        event_receptor_id = _require_nonempty_string(
            event.receptor_id,
            label="event.receptor_id",
        )
        if (
            event_id != particle_id
            or event_scenario_id != final_state.scenario_id
            or event.member_id != member_id
            or event_site_id != final_site_id
            or event_region_id != final_region_id
            or event_receptor_id != final_receptor_id
        ):
            raise ValueError("event identity 必須與 final_state 完全一致。")
        _nonnegative_count(event.member_id, label="event.member_id")
        event_type = event.event_type
        if not isinstance(event_type, EventType):
            raise ValueError("event.event_type 必須是 EventType。")
        event_time = _utc_nanosecond(
            event.time_utc_ns,
            label="event.time_utc_ns",
        )
        if event_time < final_time or event_time > arrival_time:
            raise ValueError("event.time_utc_ns 必須落在 arrival 與 final time 之間。")
        _finite_real_scalar(event.x_m, label="event.x_m")
        _finite_real_scalar(event.y_m, label="event.y_m")
        _finite_real_scalar(event.z_m, label="event.z_m")
        fraction = _finite_real_scalar(event.fraction, label="event.fraction")
        if fraction < 0.0 or fraction > 1.0:
            raise ValueError("event.fraction 必須落在 [0, 1]。")
        if not isinstance(event.attributes, Mapping):
            raise ValueError("event.attributes 必須是 mapping。")
        if "also_local_domain_first_exit" in event.attributes and type(
            event.attributes["also_local_domain_first_exit"]
        ) is not bool:
            raise ValueError("also_local_domain_first_exit 必須是原生 bool。")

        if event_type in {
            EventType.OTHER_SITE_LOCAL_DOMAIN_ENTER,
            EventType.OTHER_SITE_LOCAL_DOMAIN_EXIT,
        }:
            related_site = _require_nonempty_string(
                event.related_study_site_id,
                label="event.related_study_site_id",
            )
            if related_site == final_site_id or related_site not in spec.site_grids:
                raise ValueError(
                    "other-site event 的 related_study_site_id 必須是 spec 中不同於 origin 的站點。"
                )

        if event_type in _TERMINAL_EVENT_TYPE_BY_STATUS.values():
            terminal_events.append(event)

    if len(terminal_events) != 1 or terminal_events[0].event_type != expected_terminal_event:
        raise ValueError("final status 必須對應恰有一筆 terminal EventType。")

    return (
        scenario,
        final_state,
        observations,
        events,
        final_status,
        arrival_time,
    )


def _mutable_site_grid_arrays(
    aggregate: EventAggregateChunk,
) -> dict[str, dict[str, np.ndarray]]:
    """把初始化結果中的六類唯讀 grid 複製成目前呼叫可安全累加的陣列。

    ``EventAggregateChunk`` 對外必須不可變，因此事件聚合不能直接寫入其唯讀
    view。這個暫時結構只存在於函式內，完成所有計數後會再次交給
    ``SiteEventGridCounts`` 與 ``EventAggregateChunk`` 做防禦性封存。
    """

    mutable: dict[str, dict[str, np.ndarray]] = {}
    for site_id, counts in aggregate.site_grid_counts.items():
        mutable[site_id] = {
            field: np.array(getattr(counts, field), dtype=np.int64, copy=True)
            for field in _SITE_GRID_COUNT_FIELDS
        }
    return mutable


def _build_source_receptor_zero_topology(
    aggregate: EventAggregateChunk,
    *,
    scenario_pairs: Sequence[tuple[str, str]],
) -> tuple[
    dict[SourceReceptorAggregateKey, int],
    dict[SourceReceptorAggregateKey, np.ndarray],
    dict[ReceptorAggregateKey, int],
]:
    """建立情境 site/receptor 對應的完整 source-receptor 零拓撲。

    spec 本身沒有獨立的 receptor 清單，因此受體識別只能由已驗證的 Scenario
    提供。每個出現過的 site/receptor pair 都會與該站所有 local／outer
    ``BoundaryAggregateKey`` 做笛卡兒積；即使本批沒有任何 crossing，也保留
    raw count、age histogram 與受體分母 key，讓後續合併不必猜測缺少的零列。
    """

    source_raw: dict[SourceReceptorAggregateKey, int] = {}
    source_travel: dict[SourceReceptorAggregateKey, np.ndarray] = {}
    valid_by_receptor = {
        ReceptorAggregateKey(study_site_id=site_id, receptor_id=receptor_id): 0
        for site_id, receptor_id in scenario_pairs
    }
    age_bin_count = aggregate.age_bin_edges_seconds.size - 1
    for site_id, receptor_id in scenario_pairs:
        for boundary_key in aggregate.boundary_bin_edges_m:
            if boundary_key.study_site_id != site_id:
                continue
            source_key = SourceReceptorAggregateKey(
                study_site_id=site_id,
                receptor_id=receptor_id,
                boundary_kind=boundary_key.boundary_kind,
                boundary_segment_id=boundary_key.boundary_segment_id,
            )
            source_raw[source_key] = 0
            source_travel[source_key] = np.zeros(
                age_bin_count,
                dtype=np.int64,
            )
    return source_raw, source_travel, valid_by_receptor


def aggregate_result_events(
    results: Sequence[ParticleResult],
    *,
    scenarios_by_id: Mapping[str, Scenario],
    spec: AggregateSpec,
    age_bin_edges_seconds: np.ndarray,
) -> EventAggregateChunk:
    """將已完成的 ParticleResult 事件聚合為完整、不可變的事件資料塊。

    Args:
        results: 非空的 ``ParticleResult`` 序列。每個結果必須是非 ACTIVE 的
            terminal member，且同一呼叫中的 ``particle_id`` 不可重複。
        scenarios_by_id: 由 scenario ID 到 ``Scenario`` 的 mapping；key 必須與
            value 的 ``scenario_id`` 完全一致，且 Scenario 的站點必須存在於 spec。
        spec: 已驗證的公尺制 ``AggregateSpec``，提供站點矩形、cell size、邊界
            分類與邊界段長度。
        age_bin_edges_seconds: 以秒表示的年齡格線；先通過既有格式驗證後，還
            必須以 ``np.array_equal`` 完全等於 ``spec.age_bin_edges_seconds``。
            正式 age 軸由 ``AggregateSpec`` 鎖定，所有旅行年齡直方圖的 age 軸
            長度固定為 ``size - 1``。

    Returns:
        ``EventAggregateChunk``。結果包含所有 spec 站點、所有有序跨站 pair、
        情境中出現的 site/receptor 分母與其所有 local／outer source-receptor
        key；沒有事件的拓撲仍以零保留。所有陣列和巢狀 mapping 由資料類別
        防禦性複製並設為唯讀。

    Raises:
        ValueError: 輸入 identity、觀測順序、終止事件、座標、邊界分類、拓撲或
            年齡範圍違反契約時。
        RuntimeError: 任一 int64 計數陣列即將溢位時。

    Notes:
        ``AggregateSpec.denominator_policy`` 固定排除 DATA_GAP、
        NUMERICAL_FAILURE 與 PRE_WINDOW_DEPOSITION。前兩類成員的 outcome 與 failure
        grid 仍精確保存作為失敗診斷；pre-window 成員只記 outcome，表示研究窗內沒有
        漂流歷程。三類成員的 local／outer、boundary、source-receptor、bed contact
        與 cross-site 事件不能累加到來源 numerator；如此 numerator 與 valid
        denominator 才來自同一有效成員母體。這些計數只代表條件式來源足跡
        或相對來源權重，不是絕對來源機率。

        本函式只做 raw event/grid/histogram 累加與分母計數，不執行檔案 I/O、
        核密度估計（KDE）、ratio、bootstrap 或 release 寫出。輸出仍只代表
        指定情境下的條件式來源足跡／相對來源權重，不能直接解讀為因果歸因。
    """

    if not isinstance(spec, AggregateSpec):
        raise ValueError("spec 必須是 AggregateSpec。")
    try:
        result_items = tuple(results)
    except (TypeError, ValueError) as error:
        raise ValueError("results 必須是 ParticleResult 序列。") from error
    if not result_items:
        raise ValueError("results 不可為空。")

    age_edges = _validate_age_bin_edges(age_bin_edges_seconds)
    spec_age_edges = np.asarray(spec.age_bin_edges_seconds, dtype=np.float64)
    if not np.array_equal(age_edges, spec_age_edges):
        raise ValueError(
            "age_bin_edges_seconds 必須與 spec.age_bin_edges_seconds 完全相等；"
            "正式 age 軸由 AggregateSpec 鎖定。"
        )
    scenarios = _validate_scenarios_by_id(scenarios_by_id, spec=spec)

    # 先完成每個 result 的全部 fail-closed 驗證，再配置並寫入累加器。如此任何
    # 壞資料都不會留下可被 caller 誤用的半完成結果，也能在累加期間只使用已驗證
    # 的 identity 與時間欄位。
    validated_results: list[
        tuple[
            Scenario,
            ParticleState,
            tuple[Observation, ...],
            tuple[BoundaryEvent, ...],
            ParticleStatus,
            int,
        ]
    ] = []
    seen_particle_ids: set[str] = set()
    for result in result_items:
        validated = _validate_particle_result(
            result,
            scenarios_by_id=scenarios,
            spec=spec,
        )
        particle_id = validated[1].particle_id
        if particle_id in seen_particle_ids:
            raise ValueError("同一 call 的 particle_id 不可重複。")
        seen_particle_ids.add(particle_id)
        validated_results.append(validated)

    aggregate = initialize_event_aggregate(
        spec,
        age_bin_edges_seconds=age_edges,
    )
    mutable_site_grids = _mutable_site_grid_arrays(aggregate)
    boundary_raw = {
        key: np.array(value, dtype=np.int64, copy=True)
        for key, value in aggregate.boundary_arclength_raw_count.items()
    }
    boundary_travel = {
        key: np.array(value, dtype=np.int64, copy=True)
        for key, value in aggregate.boundary_travel_age_histogram.items()
    }
    scenario_pairs = tuple(
        sorted(
            {
                (scenario.study_site_id, scenario.receptor_id)
                for scenario in scenarios.values()
            }
        )
    )
    source_raw, source_travel, valid_by_receptor = _build_source_receptor_zero_topology(
        aggregate,
        scenario_pairs=scenario_pairs,
    )
    site_ids = tuple(spec.site_grids)
    cross_site = {
        CrossSiteAggregateKey(source, target): 0
        for source in site_ids
        for target in site_ids
        if source != target
    }
    outcome_by_site = {
        site_id: {status.value: 0 for status in ParticleStatus if status != ParticleStatus.ACTIVE}
        for site_id in site_ids
    }
    valid_by_site = {site_id: 0 for site_id in site_ids}
    total_by_site = {site_id: 0 for site_id in site_ids}
    grid_edges_by_site = {
        site_id: _grid_edges_for_site(spec, site_id)
        for site_id in site_ids
    }
    seen_cross_targets: set[tuple[str, str, str]] = set()

    for scenario, final_state, _observations, events, final_status, arrival_time in validated_results:
        site_id = scenario.study_site_id
        receptor_key = ReceptorAggregateKey(
            study_site_id=site_id,
            receptor_id=scenario.receptor_id,
        )
        if receptor_key not in valid_by_receptor:
            raise RuntimeError("已驗證的 Scenario site/receptor 不在分母拓撲中。")

        # valid member 是由最終狀態決定，而不是由某一筆較早事件是否存在決定。
        # 三種無效成員的 outcome 都保留；DATA_GAP／NUMERICAL_FAILURE 另外寫入 failure
        # grid。所有無效成員的來源 numerator 均排除，確保 numerator 與
        # denominator_policy 的有效成員母體一致。這裡只建立旗標，不以 early
        # continue 跳過後續事件驗證。
        is_valid_member = final_status not in _INVALID_MEMBER_STATUSES
        outcome_by_site[site_id][final_status.value] += 1
        total_by_site[site_id] += 1
        if is_valid_member:
            valid_by_site[site_id] += 1
            valid_by_receptor[receptor_key] += 1

        # 每粒子的 local／outer semantic 都獨立計數；FLOW event 在標註
        # also_local_domain_first_exit=True 時會同時占用兩種語意，但仍只算一個
        # event。重複 semantic 會拒絕，避免往返或重複寫出造成來源分母膨脹。
        local_semantic_count = 0
        outer_semantic_count = 0
        semantic_events: list[tuple[BoundaryEvent, str]] = []
        for event in events:
            if event.event_type == EventType.LOCAL_DOMAIN_FIRST_EXIT:
                local_semantic_count += 1
                semantic_events.append((event, "local"))
            elif event.event_type == EventType.FLOW_DOMAIN_OPEN_EXIT:
                outer_semantic_count += 1
                semantic_events.append((event, "outer"))
                if event.attributes.get("also_local_domain_first_exit", False) is True:
                    local_semantic_count += 1
                    semantic_events.append((event, "local"))
        if local_semantic_count > 1:
            raise ValueError("每個 particle 最多只能有一個 local semantic event。")
        if outer_semantic_count > 1:
            raise ValueError("每個 particle 最多只能有一個 outer semantic event。")

        for event, boundary_kind in semantic_events:
            boundary_segment_id = _require_nonempty_string(
                event.boundary_segment_id,
                label="semantic event.boundary_segment_id",
            )
            segment_groups = spec.site_boundary_segment_ids[site_id]
            allowed_segments = (
                segment_groups.local_segment_ids
                if boundary_kind == "local"
                else segment_groups.outer_segment_ids
            )
            if boundary_segment_id not in allowed_segments:
                raise ValueError("semantic event 的 boundary segment 不屬於 spec 對應 site/kind。")
            boundary_key = BoundaryAggregateKey(
                study_site_id=site_id,
                boundary_kind=boundary_kind,
                boundary_segment_id=boundary_segment_id,
            )
            if boundary_key not in boundary_raw:
                raise RuntimeError("semantic event 的 boundary key 不在零拓撲中。")
            s_value = _finite_real_scalar(
                event.boundary_s_m,
                label="semantic event.boundary_s_m",
            )
            segment_length = float(spec.boundary_segment_lengths_m[boundary_segment_id])
            if s_value < 0.0 or s_value > segment_length:
                raise ValueError("semantic event.boundary_s_m 必須落在 segment 長度閉範圍內。")
            s_bin = _age_bin_index(
                s_value,
                aggregate.boundary_bin_edges_m[boundary_key],
                label="boundary_s_m",
            )
            event_age = _event_age_seconds(
                arrival_time,
                _utc_nanosecond(event.time_utc_ns, label="event.time_utc_ns"),
            )
            age_bin = _age_bin_index(
                event_age,
                age_edges,
                label="event travel age",
            )
            event_x = _finite_real_scalar(event.x_m, label="semantic event.x_m")
            event_y = _finite_real_scalar(event.y_m, label="semantic event.y_m")
            x_edges, y_edges = grid_edges_by_site[site_id]
            ix = _grid_cell_index(event_x, x_edges, label="semantic event x")
            iy = _grid_cell_index(event_y, y_edges, label="semantic event y")
            grid_field = (
                "local_first_exit_count"
                if boundary_kind == "local"
                else "outer_first_exit_count"
            )
            source_key = SourceReceptorAggregateKey(
                study_site_id=site_id,
                receptor_id=scenario.receptor_id,
                boundary_kind=boundary_kind,
                boundary_segment_id=boundary_segment_id,
            )
            if source_key not in source_raw:
                raise RuntimeError("semantic event 的 source-receptor key 不在零拓撲中。")
            # 上述欄位與 key／座標／時間即使對 invalid member 也已完成驗證；只有
            # valid member 才能寫入同一有效母體的 local/outer 與 source numerator。
            if is_valid_member:
                _increment_int64(
                    mutable_site_grids[site_id][grid_field],
                    (iy, ix),
                    label=f"{grid_field} grid",
                )
                _increment_int64(
                    boundary_raw[boundary_key],
                    (s_bin,),
                    label="boundary_arclength_raw_count",
                )
                _increment_int64(
                    boundary_travel[boundary_key],
                    (s_bin, age_bin),
                    label="boundary_travel_age_histogram",
                )
                source_raw[source_key] += 1
                _increment_int64(
                    source_travel[source_key],
                    (age_bin,),
                    label="source_receptor_travel_age_histogram",
                )

        # BED_CONTACT 是可反射的接觸診斷，DEPOSITED 是 terminal bed contact；
        # 兩者合併後按回溯 age 排序，最小 age 明確定義為 first，其餘每筆都是
        # repeated。這裡不使用事件輸入順序，避免不同 writer 排序造成統計差異。
        bed_events = [
            event
            for event in events
            if event.event_type in {EventType.BED_CONTACT, EventType.DEPOSITED}
        ]
        bed_events.sort(
            key=lambda event: _event_age_seconds(
                arrival_time,
                _utc_nanosecond(event.time_utc_ns, label="bed event.time_utc_ns"),
            )
        )
        x_edges, y_edges = grid_edges_by_site[site_id]
        for index, event in enumerate(bed_events):
            event_x = _finite_real_scalar(event.x_m, label="bed event.x_m")
            event_y = _finite_real_scalar(event.y_m, label="bed event.y_m")
            ix = _grid_cell_index(event_x, x_edges, label="bed event x")
            iy = _grid_cell_index(event_y, y_edges, label="bed event y")
            grid_field = (
                "bed_first_contact_count"
                if index == 0
                else "bed_repeated_contact_count"
            )
            # invalid member 的 BED_CONTACT／DEPOSITED 仍在上方完成 age 排序與座標
            # 驗證，但不得混入與 valid denominator 對應的 first/repeated numerator。
            if is_valid_member:
                _increment_int64(
                    mutable_site_grids[site_id][grid_field],
                    (iy, ix),
                    label=f"{grid_field} grid",
                )

        if final_status in {
            ParticleStatus.DATA_GAP,
            ParticleStatus.NUMERICAL_FAILURE,
        }:
            ix = _grid_cell_index(final_state.x_m, x_edges, label="final failure x")
            iy = _grid_cell_index(final_state.y_m, y_edges, label="final failure y")
            grid_field = (
                "data_gap_failure_count"
                if final_status == ParticleStatus.DATA_GAP
                else "numerical_failure_count"
            )
            _increment_int64(
                mutable_site_grids[site_id][grid_field],
                (iy, ix),
                label=f"{grid_field} grid",
            )

        for event in events:
            if event.event_type not in {
                EventType.OTHER_SITE_LOCAL_DOMAIN_ENTER,
                EventType.OTHER_SITE_LOCAL_DOMAIN_EXIT,
            }:
                continue
            target_site_id = _require_nonempty_string(
                event.related_study_site_id,
                label="other-site event.related_study_site_id",
            )
            if target_site_id == site_id or target_site_id not in spec.site_grids:
                raise ValueError(
                    "other-site event 的 related_study_site_id 必須是 spec 中不同於 origin 的站點。"
                )
            if event.event_type != EventType.OTHER_SITE_LOCAL_DOMAIN_ENTER:
                continue
            cross_key = CrossSiteAggregateKey(
                source_study_site_id=site_id,
                target_study_site_id=target_site_id,
            )
            if cross_key not in cross_site:
                raise RuntimeError("cross-site event 的 pair 不在完整零拓撲中。")
            # invalid member 的 related site、event type、pair key 與重複 enter 仍需
            # 走完整驗證；只有通過 valid member policy 才能累加 unique-member
            # numerator。不能在 loop 前以 continue 跳過 invalid result。
            if is_valid_member:
                unique_key = (final_state.particle_id, site_id, target_site_id)
                if unique_key in seen_cross_targets:
                    continue
                seen_cross_targets.add(unique_key)
                cross_site[cross_key] += 1

    frozen_site_grids = {
        site_id: SiteEventGridCounts(**counts)
        for site_id, counts in mutable_site_grids.items()
    }
    return EventAggregateChunk(
        site_grid_counts=frozen_site_grids,
        boundary_bin_edges_m=aggregate.boundary_bin_edges_m,
        boundary_arclength_raw_count=boundary_raw,
        boundary_travel_age_histogram=boundary_travel,
        age_bin_edges_seconds=age_edges,
        source_receptor_raw_count=source_raw,
        source_receptor_travel_age_histogram=source_travel,
        cross_site_unique_member_count=cross_site,
        outcome_count_by_site=outcome_by_site,
        valid_member_denominator_by_site=valid_by_site,
        total_member_count_by_site=total_by_site,
        valid_member_denominator_by_receptor=valid_by_receptor,
        input_particle_count=len(validated_results),
    )
