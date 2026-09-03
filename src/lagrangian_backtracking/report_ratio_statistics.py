"""報告層的計數比例與首次通過年齡分箱分位數產品。

本模組只處理已驗證聚合資料的純統計轉換，不讀取檔案、不重新載入軌跡，也不把
缺值或低樣本資料改寫成物理零值。計數使用非負的 signed ``int64`` 契約；原始
首次通過年齡直方圖的軸順序固定為 ``(y_cell, x_cell, age_bin)``，年齡邊界與
輸出中點的單位都是秒（s）。

``CountRatio`` 以明確狀態區分可計算比例、零分母與低樣本；只有零分母時比例不可
計算而以 ``None`` 保存。低樣本仍保留精確的原始計數比例，真正的樣本警示由
``EstimateStatus`` 提供，不能只靠數值或空值猜測。
``HistogramQuantileProduct`` 重用串流聚合層的
``first_passage_quantiles``，因此分位數是固定 age bin 的中點近似，不是未分箱
觀測的連續精確分位數。所有 NumPy 陣列都在產品邊界重新複製並設為唯讀，避免
caller 後續改寫輸入而污染已建立的報告資料。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final

import numpy as np

from .streaming_aggregation import first_passage_quantiles

__all__ = [
    "EstimateStatus",
    "CountRatio",
    "build_count_ratio",
    "HistogramQuantileProduct",
    "build_histogram_quantiles",
]


_INT64_MAX: Final[int] = int(np.iinfo(np.int64).max)
_DEFAULT_QUANTILES: Final[tuple[float, ...]] = (0.25, 0.5, 0.75)


class EstimateStatus(StrEnum):
    """描述一項比例或分箱分位數估計的分母與樣本數狀態。

    ``available`` 表示分子達到設定的最低樣本數；``zero_denominator`` 表示沒有
    可用分母，不能把結果當成零；``low_count`` 表示分母為正但分子低於最低樣本
    門檻。低樣本仍可能具有可稽核的有限比例或分箱分位數，因此狀態是警示而非
    把數值刪除的缺值標記。
    這三種狀態刻意分開保存，讓 renderer 或表格輸出不必從 ``None``、NaN 或零值
    反推資料品質。小寫成員名稱與字串值都是產品交換契約；同時提供大寫別名以
    適應專案既有 Enum 的大寫使用慣例，別名不會新增第四種狀態。
    """

    available = "available"
    zero_denominator = "zero_denominator"
    low_count = "low_count"

    # 專案其他 Enum 多以大寫成員命名；這些只是同一成員的別名，不改變序列化值。
    AVAILABLE = available
    ZERO_DENOMINATOR = zero_denominator
    LOW_COUNT = low_count


def _require_int64_count(value: object, *, label: str, minimum: int = 0) -> int:
    """要求非負的原生 Python ``int``，並先檢查 signed ``int64`` 上限。

    Python ``bool`` 是 ``int`` 的子類別，NumPy 整數 scalar 也可能在比較時看似
    原生整數；兩者都不能進入報告產品，否則同一筆資料在不同 I/O 邊界可能得到
    不同的型別。先以 Python 整數檢查上限，再交給 NumPy 儲存，可避免截斷或繞回。
    ``minimum`` 只用於要求最低樣本門檻為正值，計數本身仍允許零。
    """

    if type(value) is not int:
        raise TypeError(f"{label} 必須是原生 Python int，不接受 bool 或 NumPy scalar")
    if value < minimum:
        raise ValueError(f"{label} 必須大於或等於 {minimum}")
    if value > _INT64_MAX:
        raise ValueError(f"{label} 不得超過 signed int64 上限")
    return value


def _require_native_float(value: object, *, label: str) -> float:
    """要求有限的原生 Python ``float``，拒絕 NumPy scalar 與其他可轉型物件。"""

    if type(value) is not float:
        raise TypeError(f"{label} 必須是原生 Python float，不接受 bool 或 NumPy scalar")
    if not math.isfinite(value):
        raise ValueError(f"{label} 必須是有限值")
    return value


def _snapshot_quantiles(value: object) -> tuple[float, ...]:
    """複製並驗證分位數序列，保留 caller 順序且只接受原生有限 ``float``。

    分位數鍵會成為下游圖表與 sidecar 的固定欄位識別，因此不把整數、NumPy
    scalar 或字串自動轉換成浮點；同一個 quantile 重複出現也會讓結果欄位失去
    唯一意義，必須在進入串流核心前拒絕。
    """

    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise TypeError("quantiles 必須是非字串、非 mapping 的序列")
    try:
        copied = tuple(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise TypeError("quantiles 必須是可 materialize 的序列") from error
    if not copied:
        raise ValueError("quantiles 不可為空")

    normalized: list[float] = []
    for index, item in enumerate(copied):
        quantile = _require_native_float(item, label=f"quantiles[{index}]")
        if not 0.0 < quantile < 1.0:
            raise ValueError("quantiles 必須嚴格介於 0 與 1")
        if quantile in normalized:
            raise ValueError("quantiles 不可重複")
        normalized.append(quantile)
    return tuple(normalized)


def _readonly_edges(value: object) -> np.ndarray:
    """驗證年齡秒數邊界，建立有限、嚴格遞增且從零開始的 float64 副本。"""

    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError("age_bin_edges_seconds 必須是一維數值陣列") from error
    if raw.ndim != 1 or raw.size < 2 or raw.dtype.kind not in "iuf":
        raise ValueError("age_bin_edges_seconds 必須是至少兩點的一維數值陣列")

    # 轉換可能在極寬浮點型別降成 float64 時產生 infinity；轉換後再次檢查，不能
    # 讓中點或寬度把這個問題延後到繪圖層才暴露。
    try:
        edges = np.array(raw, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("age_bin_edges_seconds 必須可安全轉為 float64") from error
    if not np.all(np.isfinite(edges)):
        raise ValueError("age_bin_edges_seconds 必須全部為有限值")
    if edges[0] != 0.0:
        raise ValueError("age_bin_edges_seconds 的第一個邊界必須精確為 0")
    if not np.all(edges[1:] > edges[:-1]):
        raise ValueError("age_bin_edges_seconds 必須嚴格遞增")
    return edges


def _readonly_int64_histogram(value: object) -> np.ndarray:
    """驗證三維非負直方圖並安全複製成 signed ``int64``。

    逐元素先用 Python ``int`` 檢查，特別涵蓋寬於 int64 的 signed/unsigned NumPy
    dtype；只有通過檢查後才轉成固定寬度陣列。這個順序可防止 uint64 大值在轉型時
    靜默變成負數，也讓後續的 cell total overflow 檢查有可信的原始計數。
    """

    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError("first_passage_age_histogram 必須是三維整數陣列") from error
    if raw.ndim != 3:
        raise ValueError("first_passage_age_histogram 必須是三維陣列")
    if raw.dtype.kind not in "iu":
        raise TypeError("first_passage_age_histogram 必須是整數 dtype，不接受 bool 或浮點")

    for element in raw.flat:
        count = int(element)
        if count < 0:
            raise ValueError("first_passage_age_histogram 不可包含負值")
        if count > _INT64_MAX:
            raise ValueError("first_passage_age_histogram 元素超出 signed int64 上限")

    try:
        return np.array(raw, dtype=np.int64, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("first_passage_age_histogram 無法安全轉為 int64") from error


def _readonly_count_grid(value: object, *, shape: tuple[int, ...], label: str) -> np.ndarray:
    """驗證非負整數 grid 並建立與 caller 記憶體分離的 int64 副本。"""

    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} 必須是形狀 {shape} 的整數陣列") from error
    if raw.shape != shape or raw.dtype.kind not in "iu":
        raise ValueError(f"{label} 必須是形狀 {shape} 的整數陣列")
    for element in raw.flat:
        count = int(element)
        if count < 0:
            raise ValueError(f"{label} 不可包含負值")
        if count > _INT64_MAX:
            raise ValueError(f"{label} 元素超出 signed int64 上限")
    try:
        return np.array(raw, dtype=np.int64, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} 無法安全轉為 int64") from error


def _cell_counts(histogram: np.ndarray) -> np.ndarray:
    """以 Python 任意精度整數計算每個 y/x cell 的 age-bin 總數。

    不能使用 ``histogram.sum(axis=2, dtype=np.int64)``：每個 bin 個別合法時，
    多個 bin 的合計仍可能超過 int64。先以 Python ``sum`` 檢查，再寫入 int64，
    使 overflow 在產品建立前明確失敗而不產生負數總樣本。
    """

    cell_counts = np.empty(histogram.shape[:2], dtype=np.int64)
    for cell in np.ndindex(cell_counts.shape):
        total = sum(int(value) for value in histogram[cell])
        if total > _INT64_MAX:
            raise ValueError("單一 cell 的 histogram 總數不可超過 signed int64 上限")
        cell_counts[cell] = total
    return cell_counts


def _bin_geometry(edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """建立分箱中點與寬度，並拒絕無法以有限 float64 表示的結果。

    中點使用 ``left * 0.5 + right * 0.5``，而不是直接計算
    ``(left + right) / 2``；前者避免兩個大型有限秒數相加時先溢位。寬度仍保留
    實際邊界差，供 renderer 標示固定箱解析度或辨識非均勻分箱。
    """

    midpoints = np.empty(edges.size - 1, dtype=np.float64)
    widths = np.empty(edges.size - 1, dtype=np.float64)
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        midpoint = float(left) * 0.5 + float(right) * 0.5
        width = float(right) - float(left)
        if not math.isfinite(midpoint) or not math.isfinite(width) or width <= 0.0:
            raise ValueError("age bin 的中點與寬度必須是有限正秒數")
        midpoints[index] = midpoint
        widths[index] = width
    return midpoints, widths


def _readonly_status_grid(
    value: object,
    *,
    shape: tuple[int, ...],
    cell_counts: np.ndarray,
    minimum_count: int,
) -> np.ndarray:
    """驗證逐 cell ``EstimateStatus`` 並核對它與 raw count 的關係。"""

    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise TypeError("status 必須是逐 cell 的 EstimateStatus 陣列") from error
    if raw.shape != shape:
        raise ValueError(f"status 形狀必須精確為 {shape}")
    for index in np.ndindex(shape):
        if type(raw[index]) is not EstimateStatus:
            raise TypeError("status 每個元素必須是 EstimateStatus")

        count = int(cell_counts[index])
        expected = (
            EstimateStatus.zero_denominator
            if count == 0
            else EstimateStatus.low_count
            if count < minimum_count
            else EstimateStatus.available
        )
        if raw[index] is not expected:
            raise ValueError("status 必須與 raw histogram 的 cell count 及最低樣本門檻一致")

    copied = np.array(raw, dtype=object, copy=True)
    copied.setflags(write=False)
    return copied


def _readonly_quantile_mapping(
    value: object,
    *,
    quantiles: tuple[float, ...],
    shape: tuple[int, int],
    status: np.ndarray,
) -> Mapping[float, np.ndarray]:
    """複製並驗證 quantile-to-grid mapping，封存為唯讀 mapping。

    可用與低樣本 cell 都必須保留有限的秒數中點；只有零分母 cell 保存為 NaN。
    這個 NaN 不是狀態本身，真正原因仍由同位置的 ``status`` 保存。低樣本不是
    缺值，保留它能讓下游在 renderer 層依明確政策決定是否遮罩。
    """

    if not isinstance(value, Mapping):
        raise TypeError("quantile_values_seconds 必須是 mapping")
    try:
        items = tuple(value.items())
    except (TypeError, ValueError) as error:
        raise ValueError("quantile_values_seconds 無法讀取 mapping") from error
    if tuple(key for key, _ in items) != quantiles:
        raise ValueError("quantile_values_seconds 的 quantile keys 必須與 quantiles 順序完全一致")

    copied: dict[float, np.ndarray] = {}
    for quantile, array in items:
        if type(quantile) is not float:
            raise TypeError("quantile_values_seconds 的 key 必須是原生 Python float")
        try:
            raw = np.asarray(array)
        except (TypeError, ValueError) as error:
            raise ValueError("quantile_values_seconds 的 value 必須是數值陣列") from error
        if raw.shape != shape or raw.dtype.kind not in "iuf":
            raise ValueError(f"quantile_values_seconds[{quantile}] 必須是形狀 {shape} 的數值陣列")
        try:
            normalized = np.array(raw, dtype=np.float64, copy=True)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"quantile_values_seconds[{quantile}] 無法轉為 float64") from error

        for index in np.ndindex(shape):
            if status[index] is EstimateStatus.zero_denominator:
                if not math.isnan(float(normalized[index])):
                    raise ValueError("zero_denominator cell 的 quantile 必須是 NaN")
            elif not math.isfinite(float(normalized[index])):
                raise ValueError("正樣本 cell 的 quantile 必須是有限秒數")
        normalized.setflags(write=False)
        copied[quantile] = normalized
    return MappingProxyType(copied)


@dataclass(frozen=True, slots=True)
class CountRatio:
    """保存一個計數比例及其明確的可估計狀態。

    ``numerator`` 與 ``denominator`` 是事件或訪格計數的原始非負數，兩者都是
    signed ``int64`` 可表示的原生 Python ``int``；本類別不把比例解讀成絕對來源
    機率。``minimum_count`` 是判定低樣本的分子門檻，不能由 renderer 另行猜測。
    當分母為零時 raw counts 仍完整保留且 ``ratio`` 為 ``None``；當分母為正但分子
    低於門檻時，``ratio`` 仍是精確的 ``numerator / denominator``，只是 status 為
    ``low_count``。frozen dataclass 防止欄位重新指派，並不允許不一致的
    status/ratio 組合。
    """

    numerator: int
    denominator: int
    ratio: float | None
    status: EstimateStatus
    minimum_count: int = 1

    def __post_init__(self) -> None:
        """完成 scalar、範圍、分母關係與狀態一致性驗證。"""

        numerator = _require_int64_count(self.numerator, label="numerator")
        denominator = _require_int64_count(self.denominator, label="denominator")
        minimum_count = _require_int64_count(self.minimum_count, label="minimum_count", minimum=1)
        if numerator > denominator:
            raise ValueError("numerator 不可大於 denominator")
        if type(self.status) is not EstimateStatus:
            raise TypeError("status 必須是 EstimateStatus")

        expected_status = (
            EstimateStatus.zero_denominator
            if denominator == 0
            else EstimateStatus.low_count
            if numerator < minimum_count
            else EstimateStatus.available
        )
        if self.status is not expected_status:
            raise ValueError("status 必須與 numerator、denominator 及 minimum_count 一致")

        if expected_status is EstimateStatus.zero_denominator:
            if self.ratio is not None:
                raise ValueError("zero_denominator 的 ratio 必須是 None")
        else:
            if type(self.ratio) is not float:
                raise TypeError("正分母的 ratio 必須是原生 Python float")
            if not math.isfinite(self.ratio):
                raise ValueError("正分母的 ratio 必須是有限值")
            expected_ratio = numerator / denominator
            if self.ratio != expected_ratio:
                raise ValueError("ratio 必須等於 numerator / denominator")

        object.__setattr__(self, "numerator", numerator)
        object.__setattr__(self, "denominator", denominator)
        object.__setattr__(self, "minimum_count", minimum_count)

    @property
    def raw_numerator(self) -> int:
        """回傳與報告欄位命名一致的 raw numerator 別名。"""

        return self.numerator

    @property
    def raw_denominator(self) -> int:
        """回傳與報告欄位命名一致的 raw denominator 別名。"""

        return self.denominator

    @property
    def fraction(self) -> float | None:
        """回傳比例欄位的語意別名；只有零分母時為 ``None``。"""

        return self.ratio


def build_count_ratio(
    numerator: int,
    denominator: int,
    *,
    minimum_count: int = 1,
) -> CountRatio:
    """由原始計數建立一個帶狀態的比例產品。

    Args:
        numerator: 非負原生 Python ``int`` 分子，必須不大於分母且不超過 signed
            ``int64`` 上限。
        denominator: 非負原生 Python ``int`` 分母；零分母會產生
            ``zero_denominator``，不會被當成比例零。
        minimum_count: 正的原生 Python ``int`` 分子門檻。分母為正且分子小於此值
            時產生 ``low_count``；raw counts 與精確比例仍會保留。

    Returns:
        frozen ``CountRatio``。只有 ``zero_denominator`` 狀態的 ``ratio`` 是
        ``None``；正分母即使是 ``low_count``，仍保存有限原生 Python ``float``
        比例，樣本警示與原因由 status 明確保存。

    Raises:
        TypeError: 任一 scalar 不是指定的原生 Python 型別。
        ValueError: 計數為負、超出 signed ``int64``、分子大於分母或門檻不合法。
    """

    normalized_numerator = _require_int64_count(numerator, label="numerator")
    normalized_denominator = _require_int64_count(denominator, label="denominator")
    normalized_minimum = _require_int64_count(minimum_count, label="minimum_count", minimum=1)
    if normalized_numerator > normalized_denominator:
        raise ValueError("numerator 不可大於 denominator")

    if normalized_denominator == 0:
        status = EstimateStatus.zero_denominator
        ratio = None
    elif normalized_numerator < normalized_minimum:
        status = EstimateStatus.low_count
        ratio = float(normalized_numerator / normalized_denominator)
    else:
        status = EstimateStatus.available
        ratio = float(normalized_numerator / normalized_denominator)

    return CountRatio(
        numerator=normalized_numerator,
        denominator=normalized_denominator,
        ratio=ratio,
        status=status,
        minimum_count=normalized_minimum,
    )


@dataclass(frozen=True, slots=True)
class HistogramQuantileProduct:
    """保存首次通過年齡 histogram 的 raw count 與分箱分位數估計。

    ``raw_count`` 是原始三維 ``int64`` 直方圖，軸順序固定為
    ``(y_cell, x_cell, age_bin)``；``cell_count`` 是沿 age 軸以 Python 任意精度
    安全加總後的每格總樣本數，軸順序為 ``(y_cell, x_cell)``。``quantiles`` 保留
    caller 指定的原生 float 順序，``quantile_values_seconds`` 以相同 quantile
    作 key，value 是每格秒數近似。估計 status 也逐格保存：只有零分母格的
    quantile value 為 NaN，低樣本格仍保留有限的中點近似，但不能忽略其警示。

    ``bin_midpoints_seconds`` 是每個 age bin 的中點，``bin_widths_seconds`` 是
    實際 bin 寬度，兩者都用秒表示；中點只代表固定分箱解析度下的近似。所有陣列
    與 quantile mapping 都在建構時 defensive-copy 並設為唯讀，frozen dataclass
    則防止欄位重新綁定。產品只描述指定資料與分母下的條件式統計，不是絕對來源
    機率或因果歸因。
    """

    raw_count: np.ndarray
    cell_count: np.ndarray
    age_bin_edges_seconds: np.ndarray
    quantiles: tuple[float, ...]
    quantile_values_seconds: Mapping[float, np.ndarray]
    bin_midpoints_seconds: np.ndarray
    bin_widths_seconds: np.ndarray
    status: np.ndarray
    minimum_count: int = 1

    def __post_init__(self) -> None:
        """驗證 shape、dtype、overflow、分箱幾何與逐格估計狀態。"""

        raw_count = _readonly_int64_histogram(self.raw_count)
        edges = _readonly_edges(self.age_bin_edges_seconds)
        if raw_count.shape[2] != edges.size - 1:
            raise ValueError("raw_count 的 age_bin 軸長度必須等於 age edges 數量減一")

        cell_count = _cell_counts(raw_count)
        provided_cell_count = _readonly_count_grid(
            self.cell_count,
            shape=cell_count.shape,
            label="cell_count",
        )
        if not np.array_equal(provided_cell_count, cell_count):
            raise ValueError("cell_count 必須等於 raw_count 沿 age_bin 軸的安全總和")

        quantiles = _snapshot_quantiles(self.quantiles)
        minimum_count = _require_int64_count(
            self.minimum_count,
            label="minimum_count",
            minimum=1,
        )
        midpoints, widths = _bin_geometry(edges)
        supplied_midpoints = _readonly_float_vector(
            self.bin_midpoints_seconds,
            size=midpoints.size,
            label="bin_midpoints_seconds",
        )
        supplied_widths = _readonly_float_vector(
            self.bin_widths_seconds,
            size=widths.size,
            label="bin_widths_seconds",
        )
        if not np.array_equal(supplied_midpoints, midpoints):
            raise ValueError("bin_midpoints_seconds 必須符合 age bin 中點定義")
        if not np.array_equal(supplied_widths, widths):
            raise ValueError("bin_widths_seconds 必須符合 age bin 邊界差")

        status = _readonly_status_grid(
            self.status,
            shape=cell_count.shape,
            cell_counts=cell_count,
            minimum_count=minimum_count,
        )
        quantile_values = _readonly_quantile_mapping(
            self.quantile_values_seconds,
            quantiles=quantiles,
            shape=cell_count.shape,
            status=status,
        )

        # 所有輸入先經過獨立 helper，這裡再以 canonical 副本封存，確保 dataclass
        # 內部不共享 caller 的可寫 buffer，也不把 mapping 的原始 dict 留在產品內。
        object.__setattr__(self, "raw_count", _readonly_copy(raw_count, dtype=np.int64))
        object.__setattr__(self, "cell_count", _readonly_copy(cell_count, dtype=np.int64))
        object.__setattr__(self, "age_bin_edges_seconds", _readonly_copy(edges, dtype=np.float64))
        object.__setattr__(self, "quantiles", quantiles)
        object.__setattr__(self, "quantile_values_seconds", quantile_values)
        object.__setattr__(self, "bin_midpoints_seconds", _readonly_copy(midpoints, dtype=np.float64))
        object.__setattr__(self, "bin_widths_seconds", _readonly_copy(widths, dtype=np.float64))
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "minimum_count", minimum_count)

    @property
    def first_passage_age_histogram(self) -> np.ndarray:
        """回傳 raw histogram 的領域語意別名。"""

        return self.raw_count

    @property
    def raw_histogram(self) -> np.ndarray:
        """回傳 raw histogram 的簡短別名。"""

        return self.raw_count

    @property
    def raw_counts(self) -> np.ndarray:
        """回傳 raw count 陣列的複數欄位別名。"""

        return self.raw_count

    @property
    def quantile_values(self) -> Mapping[float, np.ndarray]:
        """回傳 quantile 秒數 grid mapping 的簡短別名。"""

        return self.quantile_values_seconds

    @property
    def values(self) -> Mapping[float, np.ndarray]:
        """回傳 quantile 秒數 grid mapping 的通用別名。"""

        return self.quantile_values_seconds

    @property
    def age_bin_midpoints_seconds(self) -> np.ndarray:
        """回傳 age bin 中點的欄位別名。"""

        return self.bin_midpoints_seconds

    @property
    def bin_resolution_seconds(self) -> np.ndarray:
        """回傳每個 bin 寬度，作為分箱解析度的欄位別名。"""

        return self.bin_widths_seconds

    @property
    def status_codes(self) -> np.ndarray:
        """建立唯讀的字串狀態陣列，供不接受 Enum object 的交換層使用。"""

        codes = np.empty(self.status.shape, dtype="<U17")
        for index in np.ndindex(self.status.shape):
            codes[index] = self.status[index].value
        codes.setflags(write=False)
        return codes


def _readonly_float_vector(value: object, *, size: int, label: str) -> np.ndarray:
    """複製並驗證一維有限 float64 向量，供中點與寬度欄位使用。"""

    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} 必須是一維數值陣列") from error
    if raw.ndim != 1 or raw.size != size or raw.dtype.kind not in "iuf":
        raise ValueError(f"{label} 必須是長度 {size} 的數值陣列")
    try:
        copied = np.array(raw, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} 必須可安全轉為 float64") from error
    if not np.all(np.isfinite(copied)):
        raise ValueError(f"{label} 必須全部為有限值")
    copied.setflags(write=False)
    return copied


def build_histogram_quantiles(
    first_passage_age_histogram: np.ndarray,
    *,
    age_bin_edges_seconds: np.ndarray,
    quantiles: Sequence[float] = _DEFAULT_QUANTILES,
    minimum_count: int = 1,
) -> HistogramQuantileProduct:
    """建立首次通過年齡的中點分箱分位數產品。

    Args:
        first_passage_age_histogram: 非負整數三維陣列，軸順序為
            ``(y_cell, x_cell, age_bin)``；每個元素是該 cell 的首次進入計數，
            不是停留秒數。元素與沿 age 軸的總數都必須可由 signed ``int64`` 表示。
        age_bin_edges_seconds: 從精確零開始的有限嚴格遞增秒數邊界；最後一個
            edge 的包含規則沿用 ``first_passage_quantiles``。
        quantiles: 非空、唯一、嚴格介於 0 與 1 的原生 Python ``float`` 序列；
            順序會保留在輸出 mapping 與 ``quantiles`` tuple。
        minimum_count: 正的原生 Python ``int`` 每格最低樣本門檻。總數為零的
            cell 一律先標成 ``zero_denominator``；正數但低於門檻才標成
            ``low_count``。

    Returns:
        frozen ``HistogramQuantileProduct``。正樣本 cell 的每個 quantile 是被選中
        age bin 的中點秒數，即使是 low-count 也保留此有限近似；只有零分母 cell
        的 quantile grid 為 NaN。raw histogram 與 cell count 仍保留，逐格原因由
        ``status`` 保存。

    Raises:
        TypeError: histogram、quantile 或最低樣本門檻的型別不符合原生契約。
        ValueError: shape、年齡邊界、非負計數、樣本門檻或 int64 overflow 不合法。
    """

    histogram = _readonly_int64_histogram(first_passage_age_histogram)
    edges = _readonly_edges(age_bin_edges_seconds)
    if histogram.shape[2] != edges.size - 1:
        raise ValueError("first_passage_age_histogram 的 age_bin 軸長度必須等於 age edges 數量減一")
    quantile_tuple = _snapshot_quantiles(quantiles)
    normalized_minimum = _require_int64_count(
        minimum_count,
        label="minimum_count",
        minimum=1,
    )
    cell_counts = _cell_counts(histogram)
    midpoints, widths = _bin_geometry(edges)

    # 先讓既有串流核心依相同的 half-open／最後邊界包含規則選出 bin。這裡不重寫
    # quantile 搜尋，避免報告層與聚合層在 target 邊界上的定義分裂。
    quantile_values = first_passage_quantiles(
        histogram,
        age_bin_edges_seconds=edges,
        quantiles=quantile_tuple,
    )

    status = np.empty(cell_counts.shape, dtype=object)
    for index in np.ndindex(cell_counts.shape):
        count = int(cell_counts[index])
        status[index] = (
            EstimateStatus.zero_denominator
            if count == 0
            else EstimateStatus.low_count
            if count < normalized_minimum
            else EstimateStatus.available
        )

    # first_passage_quantiles 對正樣本 cell 回傳中點；低樣本也必須保留 raw count
    # 與有限中點近似，讓 renderer 之後依報告政策決定是否遮罩。核心為保護自身輸出
    # 的唯讀陣列，因此先建立 wrapper 專屬副本；這裡只把零分母 cell 明確設為 NaN。
    masked_quantile_values: dict[float, np.ndarray] = {
        quantile: np.array(values, dtype=np.float64, copy=True)
        for quantile, values in quantile_values.items()
    }
    for values in masked_quantile_values.values():
        for index in np.ndindex(cell_counts.shape):
            if status[index] is EstimateStatus.zero_denominator:
                values[index] = np.nan

    return HistogramQuantileProduct(
        raw_count=histogram,
        cell_count=cell_counts,
        age_bin_edges_seconds=edges,
        quantiles=quantile_tuple,
        quantile_values_seconds=masked_quantile_values,
        bin_midpoints_seconds=midpoints,
        bin_widths_seconds=widths,
        status=status,
        minimum_count=normalized_minimum,
    )


def _readonly_copy(value: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    """建立指定 dtype 的獨立唯讀副本，集中封存產品陣列的 memory policy。"""

    copied = np.array(value, dtype=dtype, copy=True)
    copied.setflags(write=False)
    return copied
