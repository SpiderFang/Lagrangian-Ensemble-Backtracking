"""以單次遍歷方式建立路徑停留與首次進入年齡統計。

本模組只保存累加後的格網產品，不保存原始粒子點、不執行核密度估計（kernel density
estimation，KDE），也不解析邊界事件。輸入是 ``engine.py`` 產生的
``ParticleResult``；位置已是公尺制的 x、y、z 座標，``age_seconds`` 是從到達時刻向
過去回溯的秒數，``time_utc_ns`` 是用來驗證逆向時間順序的 UTC 奈秒整數。輸出的二維
陣列一律採 ``(y_cell, x_cell)``，首次進入年齡直方圖則採
``(y_cell, x_cell, age_bin)``。

每個相鄰觀測點之間的路徑被視為 x/y 位置與年齡皆線性變化，並在所有格線穿越比例處
切段；每段的中點決定其格子，故跨格線的停留秒數不會全部錯算到步末格。每個粒子在每格
只增加一次不重複粒子數，並把第一次進入該格的年齡放入 half-open 年齡箱。這些統計是
指定受體、到達條件、物性與流場下的條件式來源足跡／相對來源權重，不是絕對來源機率或
因果歸因；後續 release 層才負責事件、發布格式與其他不確定性產品。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from .engine import ParticleResult

__all__ = [
    "StreamingPathwayAccumulator",
    "StreamingPathwayAggregate",
    "first_passage_quantiles",
    "merge_streaming_pathway_aggregates",
    "stream_pathway_first_passage",
]

# 這些常數固定本模組的資料契約：計數只能落在 int64 的非負範圍，時間守恆則沿用
# 串流聚合器既有的相對與絕對容許誤差。集中定義可避免建構、合併與驗證使用不同門檻。
_INT64_MAX = int(np.iinfo(np.int64).max)
_CONSERVATION_RTOL = 1.0e-10
_CONSERVATION_ATOL = 1.0e-8


@dataclass(frozen=True, slots=True)
class StreamingPathwayAggregate:
    """保存串流路徑聚合的公尺制格網、停留秒數與首次進入年齡。

    ``x_edges_m`` 與 ``y_edges_m`` 是有限、嚴格遞增的格線；兩者分別定義 x 軸與 y 軸
    的 cell 邊界，cell 陣列的形狀固定為
    ``(len(y_edges_m) - 1, len(x_edges_m) - 1)``，第一軸是 y、第二軸是 x。年齡箱
    ``age_bin_edges_seconds`` 同樣是有限、嚴格遞增的秒數邊界，第一個邊界必須為精確
    的 0；``first_passage_age_histogram`` 的最後一軸是這些邊界形成的 age bin。

    ``unique_particle_count`` 是每格曾被多少個不同粒子訪問，並非交點數；
    ``residence_time_seconds`` 是把每個相鄰觀測區間按線性路徑分配後的累積停留秒數。
    ``first_passage_age_histogram`` 則對每一粒子在每格的第一次進入年齡計數。所有輸出
    陣列都在建立時防禦性複製並設為唯讀，避免 caller 改寫格線或統計後破壞產品定義。
    本類別不保存原始觀測點，也不包含事件、KDE 或連續時間分布的額外假設。
    """

    x_edges_m: np.ndarray
    y_edges_m: np.ndarray
    age_bin_edges_seconds: np.ndarray
    unique_particle_count: np.ndarray
    residence_time_seconds: np.ndarray
    first_passage_age_histogram: np.ndarray
    input_particle_count: int
    input_interval_seconds: float
    allocated_interval_seconds: float

    def __post_init__(self) -> None:
        """嚴格驗證直接建構的聚合資料並建立獨立、唯讀的 canonical 副本。

        ``frozen`` dataclass 只防止欄位重新綁定，不能保證 caller 傳入的 NumPy 陣列具備
        正確 dtype、軸順序、非負性或時間守恆。因此所有公開欄位都在建構當下驗證：x/y
        邊界是公尺的一維有限嚴格遞增格線，age 邊界是秒的一維有限嚴格遞增格線且從零
        開始；二維產品固定為 ``(y_cell, x_cell)``，三維 age histogram 固定為
        ``(y_cell, x_cell, age_bin)``。計數安全轉成 int64，停留時間轉成 float64，並
        檢查非負、有限及 ``input_interval_seconds``、``allocated_interval_seconds`` 與
        residence 總和的守恆。最後再次複製並關閉寫入旗標，避免輸入或回傳物件互相共享
        可寫記憶體。
        """

        validated = _validate_aggregate_fields(
            x_edges_m=self.x_edges_m,
            y_edges_m=self.y_edges_m,
            age_bin_edges_seconds=self.age_bin_edges_seconds,
            unique_particle_count=self.unique_particle_count,
            residence_time_seconds=self.residence_time_seconds,
            first_passage_age_histogram=self.first_passage_age_histogram,
            input_particle_count=self.input_particle_count,
            input_interval_seconds=self.input_interval_seconds,
            allocated_interval_seconds=self.allocated_interval_seconds,
            context="StreamingPathwayAggregate",
        )
        (
            x_edges,
            y_edges,
            age_edges,
            unique_count,
            residence,
            first_passage_histogram,
            input_particle_count,
            input_interval_seconds,
            allocated_interval_seconds,
        ) = validated
        object.__setattr__(self, "x_edges_m", _readonly_copy(x_edges, dtype=np.float64))
        object.__setattr__(self, "y_edges_m", _readonly_copy(y_edges, dtype=np.float64))
        object.__setattr__(
            self,
            "age_bin_edges_seconds",
            _readonly_copy(age_edges, dtype=np.float64),
        )
        object.__setattr__(
            self,
            "unique_particle_count",
            _readonly_copy(unique_count, dtype=np.int64),
        )
        object.__setattr__(
            self,
            "residence_time_seconds",
            _readonly_copy(residence, dtype=np.float64),
        )
        object.__setattr__(
            self,
            "first_passage_age_histogram",
            _readonly_copy(first_passage_histogram, dtype=np.int64),
        )
        object.__setattr__(self, "input_particle_count", input_particle_count)
        object.__setattr__(self, "input_interval_seconds", input_interval_seconds)
        object.__setattr__(self, "allocated_interval_seconds", allocated_interval_seconds)


def _validate_nonnegative_int_array(
    value: object,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    """驗證非負整數網格並安全複製成 int64。

    計數資料的每個元素代表粒子數或首次進入次數，不能用浮點、布林或負值代替。特別
    對 unsigned dtype 先檢查 int64 上限，再進行轉型，避免 NumPy 在轉換階段把超大值
    截斷或繞回；形狀由公尺格網的 y、x cell 數與 age bin 數精確決定。
    """

    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 必須是形狀 {expected_shape} 的非負整數陣列") from error
    if raw.shape != expected_shape:
        raise ValueError(f"{name} 形狀必須精確為 {expected_shape}")
    if raw.dtype.kind not in "iu":
        raise ValueError(f"{name} 必須使用非負整數 dtype，不接受布林或浮點")
    if raw.dtype.kind == "i" and np.any(raw < 0):
        raise ValueError(f"{name} 不可包含負值")
    if raw.dtype.kind == "u" and np.any(raw > _INT64_MAX):
        raise ValueError(f"{name} 的值必須可安全表示為 int64")
    try:
        copied = np.array(raw, dtype=np.int64, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} 的值必須可安全表示為 int64") from error
    if np.any(copied < 0):
        raise ValueError(f"{name} 不可包含負值")
    return copied


def _validate_residence_array(
    value: object,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    """驗證公尺格網的停留秒數並建立有限非負 float64 副本。

    residence 的每格數值是依線性路徑切段後累積的秒數，軸順序必須與計數產品同為
    ``(y_cell, x_cell)``。接受整數或浮點數值 dtype，但拒絕布林、字串、複數、負值與
    非有限值；轉成 float64 後再檢查一次，防止較寬浮點數在降精度時變成無限值。
    """

    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 必須是形狀 {expected_shape} 的非負數值陣列") from error
    if raw.shape != expected_shape:
        raise ValueError(f"{name} 形狀必須精確為 {expected_shape}（y_cell、x_cell）")
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} 必須使用數值 dtype，不接受布林、字串或複數")
    if not np.all(np.isfinite(raw)):
        raise ValueError(f"{name} 必須全部為有限值")
    if np.any(raw < 0):
        raise ValueError(f"{name} 不可包含負值")
    try:
        copied = np.array(raw, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} 必須可安全轉為 float64") from error
    if not np.all(np.isfinite(copied)):
        raise ValueError(f"{name} 轉為 float64 後必須全部有限")
    return copied


def _validate_nonnegative_finite_scalar(value: object, *, name: str) -> float:
    """驗證一個代表粒子數間隔的有限非負數值 scalar 並轉成 Python float。

    時間欄位的單位是秒；只接受零維整數或浮點數值，拒絕布林、字串、複數、陣列、
    缺值、無限值與負值。使用相同 helper 驗證輸入時間與格網分配時間，可確保合併前後
    的守恆比較不會混入不同 scalar 轉換規則。
    """

    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 必須是有限非負數值 scalar（秒）") from error
    if raw.ndim != 0 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} 必須是有限非負數值 scalar（秒）")
    try:
        number = float(raw)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} 必須是有限非負數值 scalar（秒）") from error
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} 必須是有限非負數值 scalar（秒）")
    return number


def _validate_input_particle_count(value: object, *, name: str) -> int:
    """驗證粒子分母是非負的真正 Python int，而非 bool 或 NumPy 整數 scalar。"""

    if type(value) is not int or value < 0:  # noqa: E721 - 此處刻意要求 Python int 的 exact type。
        raise ValueError(f"{name} 必須是非負的 Python int，不接受 bool 或 NumPy 整數")
    return value


def _finite_fsum(values: Iterable[float], *, name: str) -> float:
    """以 ``math.fsum`` 安全累加秒數，並把任何浮點溢位轉成明確 RuntimeError。

    合併後的輸入秒數與分配秒數不能使用 NumPy 固定寬度加總，否則大量 shard 可能在
    中間步驟得到無限值而延後才難以定位。所有輸入已逐項驗證為有限非負 float，這裡仍
    檢查 ``fsum`` 的結果，確保回傳時間 scalar 可供後續守恆驗證。
    """

    try:
        total = math.fsum(values)
    except (OverflowError, ValueError) as error:
        raise RuntimeError(f"{name} 累加超出有限 float64 範圍") from error
    if not np.isfinite(total):
        raise RuntimeError(f"{name} 累加結果必須保持有限")
    return float(total)


def _finite_residence_sum(values: np.ndarray, *, name: str) -> float:
    """以高精度有限累加計算 residence 總秒數，避免總和溢位被靜默接受。"""

    return _finite_fsum((float(value) for value in values.flat), name=name)


def _require_time_conservation(
    input_interval_seconds: float,
    allocated_interval_seconds: float,
    *,
    context: str,
) -> None:
    """依固定相對／絕對容許範圍檢查兩個秒數是否守恆。"""

    if not np.isclose(
        input_interval_seconds,
        allocated_interval_seconds,
        rtol=_CONSERVATION_RTOL,
        atol=_CONSERVATION_ATOL,
    ):
        raise RuntimeError(f"{context} 的 input_interval_seconds 與 allocated_interval_seconds 未守恆")


def _validate_aggregate_fields(
    *,
    x_edges_m: object,
    y_edges_m: object,
    age_bin_edges_seconds: object,
    unique_particle_count: object,
    residence_time_seconds: object,
    first_passage_age_histogram: object,
    input_particle_count: object,
    input_interval_seconds: object,
    allocated_interval_seconds: object,
    context: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float, float]:
    """驗證一份聚合欄位並回傳可安全使用的 canonical 副本。

    ``context`` 用於把錯誤定位到 dataclass 直接建構或特定 merge chunk。函式先由 y/x
    邊界推導唯一合法的二維 cell shape，再以同一 shape 驗證 unique、residence 與
    histogram 的每一軸；因此任何將 x/y 轉置、遺漏 age 軸或多出軸的資料都會在數值合併
    前被拒絕。每格 histogram 沿 age 軸的計數總和必須精確等於 unique 粒子數，且 unique
    不得超過整份 shard 的輸入粒子分母。每份資料的 residence 總和、輸入秒數與分配秒數
    都必須在固定容許範圍內守恆，不能靠合併步驟補正不一致的 shard。
    """

    x_edges = _validate_edges(x_edges_m, name=f"{context}.x_edges_m")
    y_edges = _validate_edges(y_edges_m, name=f"{context}.y_edges_m")
    age_edges = _validate_edges(
        age_bin_edges_seconds,
        name=f"{context}.age_bin_edges_seconds",
        require_zero_start=True,
    )
    cell_shape = (y_edges.size - 1, x_edges.size - 1)
    expected_histogram_shape = (*cell_shape, age_edges.size - 1)
    unique_count = _validate_nonnegative_int_array(
        unique_particle_count,
        name=f"{context}.unique_particle_count",
        expected_shape=cell_shape,
    )
    residence = _validate_residence_array(
        residence_time_seconds,
        name=f"{context}.residence_time_seconds",
        expected_shape=cell_shape,
    )
    first_passage_histogram = _validate_nonnegative_int_array(
        first_passage_age_histogram,
        name=f"{context}.first_passage_age_histogram",
        expected_shape=expected_histogram_shape,
    )
    particle_count = _validate_input_particle_count(
        input_particle_count,
        name=f"{context}.input_particle_count",
    )

    # 首次進入 histogram 的 age 軸是同一格 unique 粒子的互斥分箱，因此其總和必須
    # 精確等於該格 unique 計數。逐值轉成 Python int 再 sum，可避免 int64 沿 age 軸
    # 累加時先繞回；同時限制總和仍可由 int64 表示，維持輸出計數 dtype 的資料契約。
    for iy, ix in np.ndindex(cell_shape):
        age_axis_total = sum(
            int(value) for value in first_passage_histogram[iy, ix, :]
        )
        if age_axis_total > _INT64_MAX:
            raise ValueError(
                f"{context}.first_passage_age_histogram[{iy}, {ix}, :] "
                "沿 age 軸合計超過 int64 上限"
            )
        unique_value = int(unique_count[iy, ix])
        if age_axis_total != unique_value:
            raise ValueError(
                f"{context}.first_passage_age_histogram[{iy}, {ix}, :] "
                "沿 age 軸合計必須精確等於 unique_particle_count"
            )
        if unique_value > particle_count:
            raise ValueError(
                f"{context}.unique_particle_count[{iy}, {ix}] "
                "不可大於 input_particle_count 粒子分母"
            )
    input_interval = _validate_nonnegative_finite_scalar(
        input_interval_seconds,
        name=f"{context}.input_interval_seconds",
    )
    allocated_interval = _validate_nonnegative_finite_scalar(
        allocated_interval_seconds,
        name=f"{context}.allocated_interval_seconds",
    )
    residence_sum = _finite_residence_sum(
        residence,
        name=f"{context}.residence_time_seconds",
    )
    _require_time_conservation(input_interval, allocated_interval, context=context)
    if not np.isclose(
        residence_sum,
        allocated_interval,
        rtol=_CONSERVATION_RTOL,
        atol=_CONSERVATION_ATOL,
    ):
        raise RuntimeError(f"{context}.residence_time_seconds 總和與 allocated_interval_seconds 未守恆")
    return (
        x_edges,
        y_edges,
        age_edges,
        unique_count,
        residence,
        first_passage_histogram,
        particle_count,
        input_interval,
        allocated_interval,
    )


def _readonly_copy(value: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    """建立與輸入解除記憶體共享的 NumPy 副本，並設為不可寫。"""

    copied = np.array(value, dtype=dtype, copy=True)
    copied.setflags(write=False)
    return copied


def _validate_edges(values: np.ndarray, *, name: str, require_zero_start: bool = False) -> np.ndarray:
    """驗證並複製一組一維數值格線。

    格線數值是公尺或秒，不能含缺值、無限值或重複位置。這裡拒絕字串與布林值，而不是
    讓 NumPy 靜默轉型；如此可把資料欄位誤讀和真正的數值格線區分開來。函式不要求等距，
    因為切段比例直接使用實際格線位置，非均勻格網仍可正確分配停留時間。
    """

    try:
        raw = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 必須是有限一維數值格線") from error
    if raw.ndim != 1 or raw.size < 2 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} 必須是至少兩點的一維數值格線")
    try:
        edges = np.array(raw, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} 必須可安全轉為 float64 格線") from error
    if not np.all(np.isfinite(edges)):
        raise ValueError(f"{name} 必須全部為有限值")
    # 直接比較相鄰邊界可避免極大但仍有限的兩個 edge 相減時先溢位成 inf；契約需要的
    # 是邊界值本身有限且嚴格遞增，而不是把差值留在另一個固定寬度的暫存陣列中。
    if not np.all(edges[1:] > edges[:-1]):
        raise ValueError(f"{name} 必須嚴格遞增")
    if require_zero_start and edges[0] != 0.0:
        raise ValueError("age_bin_edges_seconds 的第一個邊界必須精確為 0")
    return edges


def _finite_real(value: object, *, name: str) -> float:
    """把單一公尺或秒欄位驗證為有限浮點數。

    ``Observation`` 的 x、y、z 與 age 都是物理量；缺值、布林、字串及複數不能被當成
    合法位置或時間。轉成 float64 是為了讓格線搜尋、線性插值和輸出統計使用同一精度。
    """

    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 必須是有限數值標量") from error
    if raw.ndim != 0 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} 必須是有限數值標量")
    try:
        number = float(raw)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} 必須是有限數值標量") from error
    if not np.isfinite(number):
        raise ValueError(f"{name} 必須是有限數值標量")
    return number


def _finite_time(value: object) -> int | float:
    """驗證 UTC 奈秒欄位並保留整數精度供嚴格遞減檢查。"""

    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError("time_utc_ns 必須是有限數值標量") from error
    if raw.ndim != 0 or raw.dtype.kind not in "iuf":
        raise ValueError("time_utc_ns 必須是有限數值標量")
    item = raw.item()
    if raw.dtype.kind in "iu":
        return int(item)
    try:
        number = float(item)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("time_utc_ns 必須是有限數值標量") from error
    if not np.isfinite(number):
        raise ValueError("time_utc_ns 必須是有限數值標量")
    return number


def _add_int64_arrays_safely(
    total: np.ndarray,
    incoming: np.ndarray,
    *,
    name: str,
) -> None:
    """逐元素檢查 int64 上限後把一份非負計數加入累加器。

    ``total`` 與 ``incoming`` 都已由聚合欄位驗證成同形狀 int64 非負陣列。加法前先對
    每個元素比較 ``incoming <= INT64_MAX - total``，再執行原地加總；因此任何一格超過
    int64 上限都會明確失敗，不會依賴 NumPy 的固定寬度繞回行為。
    """

    if np.any(incoming > (_INT64_MAX - total)):
        raise RuntimeError(f"{name} 合併將超過 int64 上限")
    total += incoming


def _add_residence_arrays_safely(total: np.ndarray, incoming: np.ndarray, *, name: str) -> None:
    """以 float64 合併 residence，並在每個 shard 後攔截非有限結果。

    residence 的元素皆為有限非負秒數；仍使用 NumPy float64 向量化相加以維持格網軸，
    並在寫回累加器前檢查整個結果。若任何 cell 因浮點加法溢位或異常變成非有限值，
    立即以 RuntimeError 終止，避免產生不可解釋的部分產品。
    """

    with np.errstate(over="ignore", invalid="ignore"):
        updated = np.add(total, incoming, dtype=np.float64)
    if not np.all(np.isfinite(updated)):
        raise RuntimeError(f"{name} 合併結果必須保持有限 float64")
    total[...] = updated


class StreamingPathwayAccumulator:
    """以固定格網拓撲逐 chunk 累加 pathway 統計的可封閉 reducer。

    這個普通 class 只保留單一站點的 x/y 公尺格線、age 秒格線，以及三個與格線等大
    的 mutable accumulator：``unique_particle_count`` 與
    ``first_passage_age_histogram`` 是 int64 計數，``residence_time_seconds`` 是
    float64 秒數。它不保存過往 ``StreamingPathwayAggregate``、ParticleResult 或
    iterator，因此記憶體只隨單站的 ``(y, x)``／``(y, x, age)`` topology 增長，不隨
    shard 數增加。

    每個 chunk 在任何累加前都會由既有 ``_validate_aggregate_fields`` 重新驗證。第一個
    chunk 建立獨立邊界與 accumulator；後續 chunk 必須有完全相同的 x、y、age 邊界。
    int64 加法先逐元素檢查上限，residence 先計算完整候選 float64 陣列並確認有限，
    所有安全檢查通過後才一次提交三組陣列與 scalar，避免錯誤只寫入部分欄位。finalize
    會再次檢查合併後的輸入秒數、配置秒數與 residence 總和守恆，並交由
    ``StreamingPathwayAggregate`` 建立最後的防禦性唯讀副本。

    這個 reducer 的結果仍是指定條件下的 pathway 工程統計與條件式來源足跡輸入，不是
    絕對來源機率或因果歸因。現有測試使用 synthetic 小資料驗證 dtype、軸順序、溢位與
    生命週期；通過測試不代表任何正式 OCM/NWW 科學成果。
    """

    _OPEN = "open"
    _FINALIZED = "finalized"
    _FAILED = "failed"

    def __init__(self) -> None:
        """建立尚未接收 chunk 的空 reducer，不配置任何未知格網的陣列。"""

        self._state = self._OPEN
        self._chunk_count = 0
        self._x_edges: np.ndarray | None = None
        self._y_edges: np.ndarray | None = None
        self._age_edges: np.ndarray | None = None
        self._unique_total: np.ndarray | None = None
        self._residence_total: np.ndarray | None = None
        self._histogram_total: np.ndarray | None = None
        self._input_particle_count = 0
        self._input_interval_seconds = 0.0
        self._allocated_interval_seconds = 0.0

    @property
    def chunk_count(self) -> int:
        """回傳成功加入的 chunk 數；失敗或封閉後仍保留此可攜整數摘要。"""

        return self._chunk_count

    def _ensure_open(self) -> None:
        """拒絕在 finalize 或失敗後繼續使用，避免 caller 忽略 reducer 狀態。"""

        if self._state != self._OPEN:
            raise ValueError("StreamingPathwayAccumulator 已關閉")

    def _mark_failed(self) -> None:
        """把任何 add/finalize 失敗轉成不可恢復狀態。"""

        self._state = self._FAILED

    def _add_validated(
        self,
        validated: tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            int,
            float,
            float,
        ],
    ) -> None:
        """在 reducer 已開啟且欄位已驗證後，原子提交一個 chunk 的所有數值。"""

        (
            x_edges,
            y_edges,
            age_edges,
            unique_count,
            residence,
            first_passage_histogram,
            input_particle_count,
            input_interval_seconds,
            allocated_interval_seconds,
        ) = validated

        if self._chunk_count == 0:
            # _validate_aggregate_fields 已建立獨立副本；再次以 array 建立 mutable copy，
            # 讓 reducer 的累加器不會與驗證 helper 或 caller 保留的任何 ndarray alias。
            self._x_edges = np.array(x_edges, dtype=np.float64, copy=True)
            self._y_edges = np.array(y_edges, dtype=np.float64, copy=True)
            self._age_edges = np.array(age_edges, dtype=np.float64, copy=True)
            self._unique_total = np.array(unique_count, dtype=np.int64, copy=True)
            self._residence_total = np.array(residence, dtype=np.float64, copy=True)
            self._histogram_total = np.array(
                first_passage_histogram,
                dtype=np.int64,
                copy=True,
            )
            self._input_particle_count = int(input_particle_count)
            self._input_interval_seconds = float(input_interval_seconds)
            self._allocated_interval_seconds = float(allocated_interval_seconds)
            self._chunk_count = 1
            return

        # 後續 chunk 的 edge 比較是 exact equality；不允許以容許值把不同公尺格網或
        # 不同秒數 age bins 混成同一個產品。這些邊界在第一個 add 時已被獨立保存。
        assert self._x_edges is not None
        assert self._y_edges is not None
        assert self._age_edges is not None
        assert self._unique_total is not None
        assert self._residence_total is not None
        assert self._histogram_total is not None
        for axis_name, expected, actual in (
            ("x_edges_m", self._x_edges, x_edges),
            ("y_edges_m", self._y_edges, y_edges),
            ("age_bin_edges_seconds", self._age_edges, age_edges),
        ):
            if not np.array_equal(expected, actual):
                raise ValueError(f"chunk[{self._chunk_count}].{axis_name} 與第一個 chunk 不完全相同")

        # 先檢查兩組 int64 計數的逐元素上限；此階段不修改既有 accumulator。residence
        # 必須先得到完整候選結果並掃描有限性，否則某一 cell 溢位時前面 cell 可能已寫入。
        if np.any(unique_count > (_INT64_MAX - self._unique_total)):
            raise RuntimeError("unique_particle_count 合併將超過 int64 上限")
        if np.any(first_passage_histogram > (_INT64_MAX - self._histogram_total)):
            raise RuntimeError("first_passage_age_histogram 合併將超過 int64 上限")
        with np.errstate(over="ignore", invalid="ignore"):
            residence_candidate = np.add(
                self._residence_total,
                residence,
                dtype=np.float64,
            )
        if not np.all(np.isfinite(residence_candidate)):
            raise RuntimeError("residence_time_seconds 合併結果必須保持有限 float64")

        # scalar 用 Python int 與有限 fsum 計算候選值；此處不把 chunk 歷史保存為 list，
        # 因而仍是固定記憶體。合併後的全域 conservation 會在 finalize 再做一次完整檢查。
        input_particle_candidate = self._input_particle_count + int(input_particle_count)
        input_interval_candidate = _finite_fsum(
            (self._input_interval_seconds, input_interval_seconds),
            name="merged input_interval_seconds",
        )
        allocated_interval_candidate = _finite_fsum(
            (self._allocated_interval_seconds, allocated_interval_seconds),
            name="merged allocated_interval_seconds",
        )

        # 所有欄位的候選值均已安全，現在才提交。這三次 in-place update 不再有可預期的
        # dtype／overflow／有限性例外；若任何外部環境造成非預期例外，add() 外層仍會把
        # reducer 標記為 failed，caller 不能把可能不完整的狀態拿來繼續累加。
        self._unique_total += unique_count
        self._residence_total[...] = residence_candidate
        self._histogram_total += first_passage_histogram
        self._input_particle_count = input_particle_candidate
        self._input_interval_seconds = input_interval_candidate
        self._allocated_interval_seconds = allocated_interval_candidate
        self._chunk_count += 1

    def add(self, chunk: StreamingPathwayAggregate) -> None:
        """驗證並原子加入一個 pathway chunk，失敗後 reducer 永久關閉。

        ``chunk`` 的計數陣列軸固定是 ``(y, x)`` 與 ``(y, x, age)``，公尺格線與 age
        邊界必須與第一個 chunk exact 相同。驗證會複製所有欄位；成功後只保留累加器，不
        保留輸入 chunk reference。任何型別、shape、edge、conservation、iterator-like
        property 或 overflow 例外都會把 reducer 轉為 failed/closed，且保留原本的
        ``ValueError``／``RuntimeError`` 類型供 caller 定位。
        """

        self._ensure_open()
        try:
            if not isinstance(chunk, StreamingPathwayAggregate):
                raise ValueError("chunk 必須是 StreamingPathwayAggregate")
            validated = _validate_aggregate_fields(
                x_edges_m=chunk.x_edges_m,
                y_edges_m=chunk.y_edges_m,
                age_bin_edges_seconds=chunk.age_bin_edges_seconds,
                unique_particle_count=chunk.unique_particle_count,
                residence_time_seconds=chunk.residence_time_seconds,
                first_passage_age_histogram=chunk.first_passage_age_histogram,
                input_particle_count=chunk.input_particle_count,
                input_interval_seconds=chunk.input_interval_seconds,
                allocated_interval_seconds=chunk.allocated_interval_seconds,
                context=f"chunk[{self._chunk_count}]",
            )
            self._add_validated(validated)
        except Exception:
            self._mark_failed()
            raise

    def finalize(self) -> StreamingPathwayAggregate:
        """完成 reducer 並回傳新的唯讀 aggregate；零 chunk 或重複呼叫皆拒絕。

        finalize 重新以 ``math.fsum`` 驗證已合併的 input／allocated 秒數，並以既有
        conservation tolerance 核對 residence 總和。成功後會釋放 reducer 內部陣列，只
        保留 ``chunk_count`` 與 finalized 狀態；回傳物件由 constructor 再次防禦性複製，
        因而不與 reducer 或任何舊 chunk alias。
        """

        self._ensure_open()
        if self._chunk_count == 0:
            self._mark_failed()
            raise ValueError("StreamingPathwayAccumulator 不可在零 chunk 時 finalize")
        assert self._x_edges is not None
        assert self._y_edges is not None
        assert self._age_edges is not None
        assert self._unique_total is not None
        assert self._residence_total is not None
        assert self._histogram_total is not None
        try:
            input_interval_seconds = _finite_fsum(
                (self._input_interval_seconds,),
                name="merged input_interval_seconds",
            )
            allocated_interval_seconds = _finite_fsum(
                (self._allocated_interval_seconds,),
                name="merged allocated_interval_seconds",
            )
            residence_sum = _finite_residence_sum(
                self._residence_total,
                name="merged residence_time_seconds",
            )
            _require_time_conservation(
                input_interval_seconds,
                allocated_interval_seconds,
                context="merged aggregate",
            )
            if not np.isclose(
                residence_sum,
                allocated_interval_seconds,
                rtol=_CONSERVATION_RTOL,
                atol=_CONSERVATION_ATOL,
            ):
                raise RuntimeError(
                    "merged residence_time_seconds 總和與 allocated_interval_seconds 未守恆"
                )
            result = StreamingPathwayAggregate(
                x_edges_m=self._x_edges,
                y_edges_m=self._y_edges,
                age_bin_edges_seconds=self._age_edges,
                unique_particle_count=self._unique_total,
                residence_time_seconds=self._residence_total,
                first_passage_age_histogram=self._histogram_total,
                input_particle_count=self._input_particle_count,
                input_interval_seconds=input_interval_seconds,
                allocated_interval_seconds=allocated_interval_seconds,
            )
        except Exception:
            self._mark_failed()
            raise

        # constructor 已完成獨立 readonly copy；釋放 reducer 對大格網的持有，避免完成後
        # 同時保留 mutable accumulator 與回傳產品。這裡不保存任何過往 chunk reference。
        self._x_edges = None
        self._y_edges = None
        self._age_edges = None
        self._unique_total = None
        self._residence_total = None
        self._histogram_total = None
        self._state = self._FINALIZED
        return result


def merge_streaming_pathway_aggregates(
    chunks: Iterable[StreamingPathwayAggregate],
) -> StreamingPathwayAggregate:
    """安全合併同一站點、同一格網與同一年齡箱定義的多個串流聚合 shard。

    Args:
        chunks: 非空的 ``StreamingPathwayAggregate`` iterable。每個 shard 必須具有完全
            相同的 x/y 公尺格線與 age 秒數邊界；二維陣列軸固定為 ``(y_cell, x_cell)``，
            三維首次進入 histogram 軸固定為 ``(y_cell, x_cell, age_bin)``。函式逐項消費
            iterable，不會先 materialize 或保存所有 shard。

    Returns:
        一個新的 immutable ``StreamingPathwayAggregate``。unique 粒子數、停留秒數與
        首次進入 histogram 逐 cell 相加；``input_particle_count`` 以 Python int 加總，
        輸入與分配秒數以 ``math.fsum`` 加總。輸出陣列皆為獨立的 float64/int64 唯讀副本，
        不與任何輸入 shard 共享可寫記憶體。

    Raises:
        ValueError: ``chunks`` 為空、無法建立或推進 iterable、元素型別不符，或任一
            shard 的格線、dtype、shape、軸順序、非負性、有限性、scalar 型別或單 shard
            時間守恆不符合資料契約時。
        RuntimeError: int64 計數相加溢位、residence float64 相加或秒數 ``math.fsum``
            產生非有限結果，或合併後輸入秒數、分配秒數與 residence 總和未在相對
            ``1e-10``、絕對 ``1e-8`` 容許範圍內守恆時。

    Notes:
        此函式只合併已完成的格網統計，不重新讀取原始粒子、不執行事件分析、核密度
        估計、ratio 或 bootstrap。合併結果仍是指定受體、到達條件、物性及流場下的
        條件式來源足跡／相對來源權重，不是絕對來源機率或因果歸因。
    """

    # reducer 只保留固定格網大小的 accumulator；不能先把 shard 轉成 tuple/list，否則
    # 正式逐 trajectory shard 的記憶體會隨 shard 數線性成長。iter() 與 next() 自身的
    # 例外是輸入 iterable 契約錯誤，統一轉為 ValueError；但 accumulator.add() 的
    # ValueError／RuntimeError 要原樣保留，讓呼叫端仍可區分欄位驗證與數值溢位。
    accumulator = StreamingPathwayAccumulator()
    try:
        iterator = iter(chunks)
    except Exception as error:
        accumulator._mark_failed()
        raise ValueError("chunks 必須是可迭代的聚合輸入") from error

    while True:
        try:
            chunk = next(iterator)
        except StopIteration:
            break
        except Exception as error:
            accumulator._mark_failed()
            raise ValueError("chunks iterator 無法取得下一筆") from error
        # 不包裝這個呼叫：add() 已負責 fail-closed，且其固定 ValueError／RuntimeError
        # 語意是既有公開 merge API 的數值契約一部分。
        accumulator.add(chunk)

    return accumulator.finalize()


def _cell_index(value: float, edges: np.ndarray, *, name: str) -> int:
    """依閉矩形與 half-open cell 定義，把一個座標轉成 cell 索引。

    內部格線採 ``searchsorted(..., side="right")``，因此格線上的位置歸入右側／上側
    cell；最右或最上外邊界則特別歸入最後 cell。這個規則與半開區間一致，且能讓閉矩形
    的合法終點不因索引等於 cell 數而被誤判為域外。呼叫前已驗證座標有限且在邊界內，若
    線性插值因極端浮點輸入跑出範圍，仍立即失敗而不默默裁切。
    """

    if not np.isfinite(value) or value < edges[0] or value > edges[-1]:
        raise ValueError(f"線性切段產生的 {name} 座標超出閉矩形格網")
    if value == edges[-1]:
        return edges.size - 2
    index = int(np.searchsorted(edges, value, side="right") - 1)
    if index < 0 or index >= edges.size - 1:
        raise ValueError(f"線性切段產生的 {name} 座標無法對應格子")
    return index


def _segment_visits(
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    *,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> list[tuple[int, int, float, float]]:
    """將一個閉矩形內的直線區間切成 ``(iy, ix, start_fraction, end_fraction)``。

    fraction 以區間起點為 0、終點為 1，且位置與 age 都依相同 fraction 線性內插。
    所有 x/y 內部格線的 crossing fraction 會合併後排序；每個相鄰 fraction 的中點
    決定該段 cell。若 x、y 都沒有位移，仍回傳完整比例 1 的所在 cell，確保零位移
    interval 的全部秒數有明確歸屬。這裡不把域外段裁到邊界，因為端點皆在凸的閉矩形內，
    若計算結果無法落入 cell 應視為數值或輸入錯誤。
    """

    x0, y0 = start_xy
    x1, y1 = end_xy
    dx = x1 - x0
    dy = y1 - y0
    if dx == 0.0 and dy == 0.0:
        # 回傳欄位契約固定為 (iy, ix, start_fraction, end_fraction)；即使粒子靜止，
        # 也必須先放 y 軸索引，避免 caller 將矩形格網的兩軸對調後錯配停留秒數。
        return [
            (
                _cell_index(y0, y_edges, name="y"),
                _cell_index(x0, x_edges, name="x"),
                0.0,
                1.0,
            )
        ]
    if not np.isfinite(dx) or not np.isfinite(dy):
        raise RuntimeError("x/y 線性切段的位移溢位，無法取得有限 crossing fraction")

    # 只考慮內部格線；外框格線只可能位於 fraction 0 或 1，不應製造零長度段。
    breaks = [0.0, 1.0]
    if dx != 0.0:
        for edge in x_edges[1:-1]:
            fraction = float((edge - x0) / dx)
            if 0.0 < fraction < 1.0:
                breaks.append(fraction)
    if dy != 0.0:
        for edge in y_edges[1:-1]:
            fraction = float((edge - y0) / dy)
            if 0.0 < fraction < 1.0:
                breaks.append(fraction)
    fractions = np.unique(np.asarray(breaks, dtype=np.float64))
    visits: list[tuple[int, int, float, float]] = []
    for start_fraction, end_fraction in zip(fractions[:-1], fractions[1:], strict=True):
        if not end_fraction > start_fraction:
            raise RuntimeError("grid crossing fraction 未保持嚴格遞增")
        midpoint_fraction = float(start_fraction + (end_fraction - start_fraction) * 0.5)
        midpoint_x = float(x0 + dx * midpoint_fraction)
        midpoint_y = float(y0 + dy * midpoint_fraction)
        ix = _cell_index(midpoint_x, x_edges, name="x")
        iy = _cell_index(midpoint_y, y_edges, name="y")
        visits.append((iy, ix, float(start_fraction), float(end_fraction)))
    if not visits:
        raise RuntimeError("非零位移 interval 未產生任何格網切段")
    return visits


def _age_bin_index(age: float, edges: np.ndarray) -> int:
    """依 half-open 年齡箱找索引，並將恰好最後邊界的值放入最後箱。

    年齡單位是秒；一般箱為 ``[left, right)``，所以剛好落在內部邊界的首次進入會進入
    右側箱。最後一個箱額外包含最終邊界。任何超過最後邊界的年齡都直接報錯，不以 clip
    掩蓋年齡箱設定不足的問題。
    """

    if not np.isfinite(age) or age < edges[0] or age > edges[-1]:
        raise ValueError("first passage age 必須落在 age_bin_edges_seconds 的閉範圍內")
    if age == edges[-1]:
        return edges.size - 2
    index = int(np.searchsorted(edges, age, side="right") - 1)
    if index < 0 or index >= edges.size - 1:
        raise ValueError("first passage age 無法對應 age bin")
    return index


def _increment_int64(array: np.ndarray, index: tuple[int, ...], *, name: str) -> None:
    """在寫入 int64 計數前檢查上限，避免固定寬度整數繞回。"""

    current = int(array[index])
    if current >= np.iinfo(np.int64).max:
        raise RuntimeError(f"{name} 累加將超過 int64 上限")
    array[index] = current + 1


def _validated_observations(
    result: ParticleResult,
    *,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    age_edges: np.ndarray,
) -> list[tuple[int | float, float, float, float, float]]:
    """驗證單一 ``ParticleResult`` 並回傳可安全計算的觀測欄位。

    回傳 tuple 的欄位依序為 ``(time_utc_ns, age_seconds, x_m, y_m, z_m)``；只在目前粒子
    的處理期間保留，不會寫入聚合結果。除了確認 age 單調增加，也確認 UTC 時間對逆向
    軌跡嚴格遞減、位置在閉矩形內、z 仍為有限公尺值，以及最後 observation 的狀態與
    ``final_state.status`` 一致。這些閘門可防止資料缺口或未完成粒子被當成正常路徑。
    """

    try:
        observations = tuple(result.observations)
        final_state = result.final_state
    except (AttributeError, TypeError) as error:
        raise ValueError("每個 result 必須包含 observations 與 final_state") from error
    if not observations:
        raise ValueError("每個 particle result 至少需要一個 observation")

    try:
        particle_id = observations[0].particle_id
        if final_state.particle_id != particle_id:
            raise ValueError("result.final_state 與 observation 的 particle_id 不一致")
    except AttributeError as error:
        raise ValueError("ParticleResult 的 particle_id 欄位不完整") from error

    validated: list[tuple[int | float, float, float, float, float]] = []
    previous_age: float | None = None
    previous_time: int | float | None = None
    for observation in observations:
        try:
            if observation.particle_id != particle_id:
                raise ValueError("同一粒子的所有 observation 必須使用相同 particle_id")
            time_utc_ns = _finite_time(observation.time_utc_ns)
            age = _finite_real(observation.age_seconds, name="observation.age_seconds")
            x_m = _finite_real(observation.x_m, name="observation.x_m")
            y_m = _finite_real(observation.y_m, name="observation.y_m")
            z_m = _finite_real(observation.z_m, name="observation.z_m")
        except AttributeError as error:
            raise ValueError("observation 缺少 particle_id、時間或位置欄位") from error
        if age < 0.0:
            raise ValueError("observation.age_seconds 必須為非負值")
        if age > age_edges[-1]:
            raise ValueError("observation.age_seconds 不可超過最後 age edge")
        if x_m < x_edges[0] or x_m > x_edges[-1] or y_m < y_edges[0] or y_m > y_edges[-1]:
            raise ValueError("所有 observation 必須落在閉矩形格網內")
        if previous_age is not None and not age > previous_age:
            raise ValueError("trajectory age_seconds 必須嚴格遞增")
        if previous_time is not None and not previous_time > time_utc_ns:
            raise ValueError("trajectory time_utc_ns 必須嚴格遞減")
        validated.append((time_utc_ns, age, x_m, y_m, z_m))
        previous_age = age
        previous_time = time_utc_ns

    try:
        final_status = final_state.status
        last_status = observations[-1].status
    except AttributeError as error:
        raise ValueError("ParticleResult 缺少 final_state.status 或 observation.status") from error
    if last_status != final_status:
        raise ValueError("最後 observation.status 必須等於 final_state.status")
    return validated


def stream_pathway_first_passage(
    results: Iterable[ParticleResult],
    *,
    x_edges_m: np.ndarray,
    y_edges_m: np.ndarray,
    age_bin_edges_seconds: np.ndarray,
) -> StreamingPathwayAggregate:
    """串流聚合粒子路徑的訪格數、停留時間與首次進入年齡。

    Args:
        results: 可重複迭代的 ``ParticleResult``；每個結果至少一筆 observation，且同一
            次呼叫中 ``particle_id`` 不可重複。函式只逐粒子保留暫時驗證資料，不保存原始
            粒子點，適合逐 shard 傳入已驗收結果。
        x_edges_m: x 軸公尺制格線，至少兩點、有限且嚴格遞增。右側外邊界屬於最後 x cell。
        y_edges_m: y 軸公尺制格線，至少兩點、有限且嚴格遞增。上側外邊界屬於最後 y cell。
        age_bin_edges_seconds: 年齡秒數格線，至少兩點、有限且嚴格遞增，第一個值必須精確
            為 0；最後邊界可被最後 age bin 包含，但超過它的觀測或首次進入會失敗。

    Returns:
        ``StreamingPathwayAggregate``。二維陣列軸順序為 ``(y_cell, x_cell)``，首次進入
        直方圖軸順序為 ``(y_cell, x_cell, age_bin)``；計數為 int64，停留秒數為 float64，
        且所有陣列都是防禦性複製的唯讀陣列。

    Raises:
        ValueError: 格線、粒子識別、觀測順序、狀態、位置、年齡或空輸入違反契約時。
        RuntimeError: 累加可能溢位、線性切段無法產生有限結果，或輸入秒數與格網分配秒數
            未在 ``1e-10`` 相對、``1e-8`` 絕對容許範圍內守恆時。

    Notes:
        路徑的每個相鄰 observation interval 都採線性位置／年齡內插；首次進入年齡因而是
        固定切段與 age bin 解析度下的統計。結果是條件式來源足跡／相對來源權重，不包含
        先驗、似然、事件因果解釋或 KDE 平滑；這些工作由後續 release 層負責。
    """

    x_edges = _validate_edges(x_edges_m, name="x_edges_m")
    y_edges = _validate_edges(y_edges_m, name="y_edges_m")
    age_edges = _validate_edges(
        age_bin_edges_seconds,
        name="age_bin_edges_seconds",
        require_zero_start=True,
    )
    shape = (y_edges.size - 1, x_edges.size - 1)
    unique_count = np.zeros(shape, dtype=np.int64)
    residence = np.zeros(shape, dtype=np.float64)
    first_passage_histogram = np.zeros((*shape, age_edges.size - 1), dtype=np.int64)

    try:
        iterator = iter(results)
    except TypeError as error:
        raise ValueError("results 必須是 ParticleResult 的可迭代集合") from error

    seen_particle_ids: set[object] = set()
    input_particle_count = 0
    input_interval_seconds = 0.0
    for result in iterator:
        observations = _validated_observations(
            result,
            x_edges=x_edges,
            y_edges=y_edges,
            age_edges=age_edges,
        )
        particle_id = result.observations[0].particle_id
        try:
            if particle_id in seen_particle_ids:
                raise ValueError("同一 call 的 particle_id 不可重複")
            seen_particle_ids.add(particle_id)
        except TypeError as error:
            raise ValueError("particle_id 必須可識別且可雜湊") from error
        input_particle_count += 1

        # 只在目前 particle 期間保存首次進入 age；key 是 (y_cell, x_cell)，value 是最小
        # age。這個字典不會被帶到回傳物件，因此聚合結果不含原始點或逐粒子軌跡。
        first_passage_by_cell: dict[tuple[int, int], float] = {}
        for first, second in zip(observations[:-1], observations[1:], strict=True):
            _, age0, x0, y0, _ = first
            _, age1, x1, y1, _ = second
            duration = age1 - age0
            if not np.isfinite(duration) or duration <= 0.0:
                raise RuntimeError("相鄰 observation 的 age interval 必須是有限正秒數")
            try:
                input_interval_seconds = math.fsum((input_interval_seconds, duration))
            except (OverflowError, ValueError) as error:
                raise RuntimeError("input_interval_seconds 累加超出有限浮點範圍") from error
            if not np.isfinite(input_interval_seconds):
                raise RuntimeError("input_interval_seconds 必須保持有限")

            visits = _segment_visits(
                (x0, y0),
                (x1, y1),
                x_edges=x_edges,
                y_edges=y_edges,
            )
            for iy, ix, start_fraction, end_fraction in visits:
                fraction = end_fraction - start_fraction
                if not np.isfinite(fraction) or fraction <= 0.0:
                    raise RuntimeError("格網切段比例必須是有限正值")
                residence_increment = duration * fraction
                if not np.isfinite(residence_increment) or residence_increment < 0.0:
                    raise RuntimeError("residence time 累加產生非有限值")
                updated_residence = residence[iy, ix] + residence_increment
                if not np.isfinite(updated_residence):
                    raise RuntimeError("residence time 累加超出有限浮點範圍")
                residence[iy, ix] = updated_residence

                # 段起點的 fraction 是該段第一次進入 cell 的位置；同一粒子重返該 cell
                # 不更新既有年齡，因為 first passage 定義是整條路徑的最小 age。
                passage_age = float(age0 + duration * start_fraction)
                cell = (iy, ix)
                if cell not in first_passage_by_cell or passage_age < first_passage_by_cell[cell]:
                    first_passage_by_cell[cell] = passage_age

        for (iy, ix), passage_age in first_passage_by_cell.items():
            _increment_int64(unique_count, (iy, ix), name="unique_particle_count")
            age_bin = _age_bin_index(passage_age, age_edges)
            _increment_int64(
                first_passage_histogram,
                (iy, ix, age_bin),
                name="first_passage_age_histogram",
            )

        # 單一 observation 沒有 interval，因此不會經過上面的切段流程；仍必須把它視為
        # 對所在 cell 的一次訪問，且第一次進入年齡就是 observation 自身的 age。
        if len(observations) == 1:
            _, age, x_m, y_m, _ = observations[0]
            ix = _cell_index(x_m, x_edges, name="x")
            iy = _cell_index(y_m, y_edges, name="y")
            cell = (iy, ix)
            first_passage_by_cell[cell] = age
            _increment_int64(unique_count, cell, name="unique_particle_count")
            age_bin = _age_bin_index(age, age_edges)
            _increment_int64(
                first_passage_histogram,
                (iy, ix, age_bin),
                name="first_passage_age_histogram",
            )

    if input_particle_count == 0:
        raise ValueError("stream pathway aggregation 不接受空 result 集合")
    allocated_interval_seconds = float(np.sum(residence, dtype=np.float64))
    if not np.isfinite(allocated_interval_seconds):
        raise RuntimeError("allocated_interval_seconds 必須保持有限")
    if not np.isclose(
        input_interval_seconds,
        allocated_interval_seconds,
        rtol=1.0e-10,
        atol=1.0e-8,
    ):
        raise RuntimeError(
            "input_interval_seconds 與 allocated_interval_seconds 未通過時間守恆檢查"
        )

    return StreamingPathwayAggregate(
        x_edges_m=x_edges,
        y_edges_m=y_edges,
        age_bin_edges_seconds=age_edges,
        unique_particle_count=unique_count,
        residence_time_seconds=residence,
        first_passage_age_histogram=first_passage_histogram,
        input_particle_count=input_particle_count,
        input_interval_seconds=input_interval_seconds,
        allocated_interval_seconds=allocated_interval_seconds,
    )


def first_passage_quantiles(
    first_passage_age_histogram: np.ndarray,
    *,
    age_bin_edges_seconds: np.ndarray,
    quantiles: Sequence[float] = (0.25, 0.5, 0.75),
) -> dict[float, np.ndarray]:
    """由首次進入年齡直方圖取得每格的固定箱解析度分位數近似。

    Args:
        first_passage_age_histogram: 非負整數三維陣列，軸順序必須為
            ``(y_cell, x_cell, age_bin)``；第三軸長度必須等於年齡邊界數減一。每個 cell
            的總數是該格有首次進入紀錄的粒子數，不是停留秒數。
        age_bin_edges_seconds: 與直方圖第三軸對應的有限、嚴格遞增秒數邊界，第一個邊界
            必須精確為 0；最後 bin 包含恰好最後邊界的首次進入。
        quantiles: 非空、唯一、有限且嚴格介於 0 與 1 的分位數序列；布林值與字串不接受。
            回傳 dictionary 的插入順序與 caller 順序相同。

    Returns:
        ``dict[float, np.ndarray]``。每個結果是形狀 ``(y_cell, x_cell)`` 的 float64 唯讀
        陣列；有樣本的 cell 回傳被選中 age bin 的中點，零樣本 cell 回傳 NaN。中點代表
        固定 age bin 解析度下的近似，不是未分箱資料的連續精確分位數，也不應解讀成更高
        時間解析度的觀測值。

    Raises:
        ValueError: 直方圖維度、dtype、非負性、形狀、年齡邊界或分位數契約不符時。
    """

    age_edges = _validate_edges(
        age_bin_edges_seconds,
        name="age_bin_edges_seconds",
        require_zero_start=True,
    )
    try:
        raw_histogram = np.asarray(first_passage_age_histogram)
    except (TypeError, ValueError) as error:
        raise ValueError("first_passage_age_histogram 必須是三維非負整數陣列") from error
    if raw_histogram.ndim != 3:
        raise ValueError("first_passage_age_histogram 必須是三維陣列")
    if raw_histogram.dtype.kind not in "iu":
        raise ValueError("first_passage_age_histogram 必須是整數 dtype，不接受布林或浮點")
    if raw_histogram.dtype.kind == "i" and np.any(raw_histogram < 0):
        raise ValueError("first_passage_age_histogram 不可包含負值")
    if raw_histogram.dtype.kind == "u" and np.any(raw_histogram > np.iinfo(np.int64).max):
        raise ValueError("first_passage_age_histogram 必須可安全表示為 int64")
    expected_age_bins = age_edges.size - 1
    if raw_histogram.shape[2] != expected_age_bins:
        raise ValueError("直方圖第三軸長度必須等於 age edges 數量減一")
    histogram = np.array(raw_histogram, dtype=np.int64, copy=True)

    try:
        quantile_values = tuple(
            _finite_real(value, name="quantile")
            for value in quantiles
        )
    except (TypeError, ValueError) as error:
        raise ValueError("quantiles 必須是非空、唯一、有限且介於 0 與 1 的數值序列") from error
    if (
        not quantile_values
        or len(set(quantile_values)) != len(quantile_values)
        or any(value <= 0.0 or value >= 1.0 for value in quantile_values)
    ):
        raise ValueError("quantiles 必須非空、唯一且嚴格介於 0 與 1")

    # 用兩個半量相加計算中點，避免兩個大型有限 edge 直接相加時溢位；中點只是分箱
    # 近似的代表值，不能被誤稱為連續資料的精確分位數。
    bin_midpoints = age_edges[:-1] * 0.5 + age_edges[1:] * 0.5
    if not np.all(np.isfinite(bin_midpoints)):
        raise ValueError("age bin 中點必須是有限秒數")

    max_int64 = np.iinfo(np.int64).max
    flat_histogram = histogram.reshape(-1, expected_age_bins)
    output_shape = histogram.shape[:2]
    outputs = {
        value: np.full(output_shape, np.nan, dtype=np.float64)
        for value in quantile_values
    }
    for flat_cell, counts in enumerate(flat_histogram):
        # 不能直接用 np.sum(int64)，因為多個合法 bin 的總數相加可能先在固定寬度
        # 整數中繞回；先用 Python int 檢查總數，通過後才使用安全的 int64 cumsum。
        total = sum(int(count) for count in counts)
        if total > max_int64:
            raise ValueError("單一 cell 的 histogram 總數不可超過 int64 上限")
        if total == 0:
            continue
        cumulative = np.cumsum(counts, dtype=np.int64)
        for quantile_value, output in outputs.items():
            target = quantile_value * total
            bin_index = int(np.searchsorted(cumulative, target, side="left"))
            if bin_index >= expected_age_bins:
                # target < total 且最後 cumulative 恰為 total；這是防禦性檢查，不能
                # 用 clip 掩蓋不一致的累積結果。
                raise RuntimeError("histogram 累積計數無法找到分位數 bin")
            output.flat[flat_cell] = bin_midpoints[bin_index]

    return {
        value: _readonly_copy(output, dtype=np.float64)
        for value, output in outputs.items()
    }
