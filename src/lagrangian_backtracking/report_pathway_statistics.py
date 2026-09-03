"""建立報告層使用的路徑格網統計產品。

本模組只接受已完成驗證的 ``StreamingPathwayAggregate``，將其轉成報告 renderer
可直接使用的 immutable 統計快照。所有平面陣列的軸順序固定為
``(y_cell, x_cell)``；首次進入年齡直方圖的來源資料維持
``(y_cell, x_cell, age_bin)``，而分位數輸出則回到 ``(y_cell, x_cell)``。x/y
格線使用公尺（m），停留時間與首次進入年齡使用秒（s）。

``visit_fraction`` 是每格曾被訪問的有效成員 numerator 除以有效成員分母，描述指定
受體與流場條件下的條件式來源足跡／相對來源權重；它不是建立先驗、似然及觀測驗證
後的絕對來源機率，也不是因果歸因。零分母保留為 NaN，避免把「沒有可用成員」誤讀
成「沒有成員訪問」。低樣本遮罩只保存布林判定，不會把原始訪格 numerator 或比例
改成零；輸入／分配 interval 與 residence 總和則保留秒數 provenance，供 renderer 和
驗證器辨識產品涵蓋範圍。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import numpy as np

from .streaming_aggregation import (
    _CONSERVATION_ATOL,
    _CONSERVATION_RTOL,
    StreamingPathwayAggregate,
    _finite_residence_sum,
    first_passage_quantiles,
)

__all__ = ["PathwayGridStatistics", "build_pathway_grid_statistics"]

_INT64_MAX: Final[int] = int(np.iinfo(np.int64).max)


def _readonly_copy(value: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    """建立與輸入解除記憶體共享的指定 dtype 唯讀副本。

    報告產品可能會被多個 renderer 或序列化流程重複使用；若直接沿用 aggregate
    陣列或 quantile helper 的回傳陣列，任何下游誤寫都可能改變其他產品看到的數值。
    因此所有保存進 dataclass 的 NumPy 陣列都在同一個邊界重新複製並關閉寫入權限。
    """

    copied = np.array(value, dtype=dtype, copy=True)
    copied.setflags(write=False)
    return copied


def _snapshot_edges(
    value: object,
    *,
    name: str,
    require_zero_start: bool = False,
) -> np.ndarray:
    """驗證公尺或秒制的一維格線，並建立有限、嚴格遞增的唯讀副本。

    ``x_edges_m`` 與 ``y_edges_m`` 定義 cell 的公尺座標；
    ``age_bin_edges_seconds`` 定義首次進入年齡箱的秒數範圍。格線不接受布林、字串、
    複數、缺值或重複值，避免下游以隱式轉型改變 cell shape 或 age-bin 語意。
    """

    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 必須是一維數值格線") from error
    if raw.ndim != 1 or raw.size < 2 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} 必須是至少兩點的一維數值格線")
    copied = np.array(raw, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(copied)):
        raise ValueError(f"{name} 必須全部為有限值")
    if not np.all(copied[1:] > copied[:-1]):
        raise ValueError(f"{name} 必須嚴格遞增")
    if require_zero_start and copied[0] != 0.0:
        raise ValueError(f"{name} 第一個邊界必須精確為 0")
    return _readonly_copy(copied, dtype=np.float64)


def _snapshot_visit_numerator(value: object, *, shape: tuple[int, int]) -> np.ndarray:
    """驗證每格不重複訪問成員 numerator，並封存為非負 int64 唯讀陣列。

    訪格 numerator 的每個元素代表至少訪問該 cell 一次的不同有效成員數，不是一般
    事件 count、軌跡線段數或停留秒數。先檢查 unsigned 值的 int64 上限再轉型，可避免
    NumPy 轉換時靜默截斷；shape 則由 x/y 公尺格線唯一決定。
    """

    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError("visit_numerator 必須是二維非負整數陣列") from error
    if raw.shape != shape:
        raise ValueError(f"visit_numerator 形狀必須精確為 {shape}")
    if raw.dtype.kind not in "iu":
        raise TypeError("visit_numerator 必須是整數 dtype，不接受 bool 或浮點")
    if raw.dtype.kind == "i" and np.any(raw < 0):
        raise ValueError("visit_numerator 不可包含負值")
    if raw.dtype.kind == "u" and np.any(raw > _INT64_MAX):
        raise ValueError("visit_numerator 的值必須可安全表示為 int64")
    return _readonly_copy(raw, dtype=np.int64)


def _snapshot_residence(value: object, *, shape: tuple[int, int]) -> np.ndarray:
    """驗證每格累積停留秒數，並建立有限非負 float64 唯讀陣列。

    residence 的軸順序與訪格計數相同，數值單位是秒（s）；它已由串流 aggregate
    依線性路徑切段後累積，不能在報告層重新解讀為粒子數或比例。這裡仍再次驗證
    shape、有限性與非負性，讓直接建構的 frozen product 也不會保存錯誤的陣列。
    """

    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError("residence_time_seconds 必須是二維非負數值陣列") from error
    if raw.shape != shape:
        raise ValueError(f"residence_time_seconds 形狀必須精確為 {shape}")
    if raw.dtype.kind not in "iuf":
        raise TypeError("residence_time_seconds 必須是整數或浮點 dtype")
    copied = np.array(raw, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(copied)):
        raise ValueError("residence_time_seconds 必須全部為有限值")
    if np.any(copied < 0.0):
        raise ValueError("residence_time_seconds 不可包含負值")
    return _readonly_copy(copied, dtype=np.float64)


def _snapshot_fraction(value: object, *, shape: tuple[int, int]) -> np.ndarray:
    """驗證訪格比例陣列，允許零分母約定使用的 NaN 並封存為唯讀 float64。

    正常分母下比例應為有限值；只有零有效成員分母時，整張比例格網才可使用 NaN
    表示不可估計。無限值與其他 shape 都是資料契約錯誤，不能留給 renderer 猜測。
    """

    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError("visit_fraction 必須是二維數值陣列") from error
    if raw.shape != shape:
        raise ValueError(f"visit_fraction 形狀必須精確為 {shape}")
    if raw.dtype.kind not in "iuf":
        raise TypeError("visit_fraction 必須是整數或浮點 dtype")
    copied = np.array(raw, dtype=np.float64, copy=True)
    if np.any(np.isinf(copied)):
        raise ValueError("visit_fraction 不可包含無限值")
    return _readonly_copy(copied, dtype=np.float64)


def _canonical_age_bin_midpoints(age_bin_edges_seconds: np.ndarray) -> np.ndarray:
    """依既有 age-bin 邊界計算 canonical midpoint 秒數。

    ``first_passage_quantiles`` 的輸出不是未分箱資料的連續分位數，而是選中 age bin
    的代表中點；此處使用與既有函式相同的兩個半量相加方式，讓 post-init 的 provenance
    驗證與 builder 產出的數值逐位一致，也避免極大有限邊界直接相加時的中間溢位。
    """

    midpoints = age_bin_edges_seconds[:-1] * 0.5 + age_bin_edges_seconds[1:] * 0.5
    if not np.all(np.isfinite(midpoints)):
        raise ValueError("age-bin canonical midpoint 必須全部為有限秒數")
    return midpoints


def _snapshot_quantile_mapping(
    value: object,
    *,
    shape: tuple[int, int],
    visit_numerator: np.ndarray,
    age_bin_edges_seconds: np.ndarray,
) -> Mapping[float, np.ndarray]:
    """建立首次進入年齡分位數的唯讀 mapping 快照。

    mapping key 是嚴格介於 0 與 1 的原生 Python ``float`` quantile；value 是
    ``(y_cell, x_cell)`` 軸、秒（s）為單位的 age-bin midpoint 近似。這個 snapshot
    同時綁定同格的訪格 numerator 與 age edges：numerator 為零時只能是 NaN；numerator
    大於零時必須是有限且精確命中 canonical midpoint。這保留空 cell 的無樣本語意，
    也避免直接建構時塞入任意有限秒數。
    """

    if not isinstance(value, Mapping):
        raise TypeError("first_passage_quantiles 必須是 mapping")
    if not value:
        raise ValueError("first_passage_quantiles 不可為空")
    if visit_numerator.shape != shape:
        raise ValueError("visit_numerator 形狀必須與 quantile 格網一致")
    canonical_midpoints = _canonical_age_bin_midpoints(age_bin_edges_seconds)
    empty_cells = visit_numerator == 0
    visited_cells = visit_numerator > 0

    copied: dict[float, np.ndarray] = {}
    for quantile, array in value.items():
        if type(quantile) is not float:
            raise TypeError("first_passage_quantiles 的 key 必須是原生 float")
        if not np.isfinite(quantile) or not 0.0 < quantile < 1.0:
            raise ValueError("first_passage_quantiles 的 key 必須嚴格介於 0 與 1")
        try:
            raw = np.asarray(array)
        except (TypeError, ValueError) as error:
            raise ValueError("first_passage_quantiles 的 value 必須是二維數值陣列") from error
        if raw.shape != shape:
            raise ValueError(f"first_passage_quantiles[{quantile}] 形狀必須精確為 {shape}")
        if raw.dtype.kind not in "iuf":
            raise TypeError("first_passage_quantiles 的 value 必須是整數或浮點 dtype")
        normalized = np.array(raw, dtype=np.float64, copy=True)
        if np.any(np.isinf(normalized)):
            raise ValueError("first_passage_quantiles 的 value 不可包含無限值")
        if np.any(normalized[~np.isnan(normalized)] < 0.0):
            raise ValueError("first_passage_quantiles 的 value 不可包含負值")
        if np.any(empty_cells) and not np.all(np.isnan(normalized[empty_cells])):
            raise ValueError(f"first_passage_quantiles[{quantile}] 的空訪格必須全部為 NaN")
        visited_values = normalized[visited_cells]
        if visited_values.size:
            if np.any(np.isnan(visited_values)) or not np.all(np.isfinite(visited_values)):
                raise ValueError(f"first_passage_quantiles[{quantile}] 的已訪格必須全部為有限值")
            if not np.all(np.isin(visited_values, canonical_midpoints)):
                raise ValueError(f"first_passage_quantiles[{quantile}] 的已訪格必須精確命中 age-bin midpoint")
        copied[quantile] = _readonly_copy(normalized, dtype=np.float64)
    return MappingProxyType(copied)


def _snapshot_low_sample_mask(value: object, *, shape: tuple[int, int]) -> np.ndarray:
    """驗證低樣本遮罩並封存為 bool 唯讀陣列。

    ``True`` 代表該 cell 的訪格 numerator 低於 caller 指定門檻；遮罩只表示可靠度
    提醒，不會覆寫訪格計數、比例或 residence 數值。建構子只能驗證其布林 shape，
    因為門檻是 builder 的輸入而不是此衍生陣列自身可反推的資訊。
    """

    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError("low_sample_mask 必須是二維布林陣列") from error
    if raw.shape != shape:
        raise ValueError(f"low_sample_mask 形狀必須精確為 {shape}")
    if raw.dtype.kind != "b":
        raise TypeError("low_sample_mask 必須是 bool dtype")
    return _readonly_copy(raw, dtype=np.bool_)


def _require_nonnegative_native_int(value: object, *, label: str) -> int:
    """要求分母是可為零的原生 Python ``int``，拒絕 bool 與 NumPy scalar。

    有效成員分母為零時仍需產生一份可辨識的空產品，因此這個 helper 與低樣本門檻
    的正整數檢查分開；它只禁止負數，不把零誤判成錯誤輸入。
    """

    if type(value) is not int:
        raise TypeError(f"{label} 必須是原生 int，且不可是 bool 或 NumPy scalar")
    if value < 0:
        raise ValueError(f"{label} 必須是非負整數")
    return value


def _require_positive_native_int(value: object, *, label: str) -> int:
    """要求低樣本門檻是大於零的原生 Python ``int``。

    門檻直接參與 ``visit_numerator < threshold`` 的可靠度判定；不接受浮點、bool 或
    NumPy 整數，可避免相同 JSON／序列化資料在不同邊界產生不同型別語意。
    """

    if type(value) is not int:
        raise TypeError(f"{label} 必須是正的原生 int，且不可是 bool 或 NumPy scalar")
    if value <= 0:
        raise ValueError(f"{label} 必須大於 0")
    return value


def _require_nonnegative_finite_native_float(value: object, *, label: str) -> float:
    """要求時間 provenance 是有限非負的原生 Python ``float``。

    interval 的單位是秒（s），並且會在報告產品中作為輸入覆蓋範圍與格網分配範圍
    的證據欄位保存。拒絕整數、bool、NumPy scalar、NaN、無限值與負值，避免序列化
    或跨模組傳遞時悄悄改變時間資料型別與缺值語意。
    """

    if type(value) is not float:
        raise TypeError(f"{label} 必須是有限非負的原生 float（秒）")
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{label} 必須是有限非負的原生 float（秒）")
    return value


@dataclass(frozen=True, slots=True)
class PathwayGridStatistics:
    """封存有效成員路徑格網的 immutable 報告統計。

    ``x_edges_m``、``y_edges_m`` 是公尺（m）格線，平面陣列的軸順序都是
    ``(y_cell, x_cell)``；``age_bin_edges_seconds`` 是首次進入年齡箱的秒（s）
    邊界，來源 histogram 的軸順序為 ``(y_cell, x_cell, age_bin)``。
    ``visit_numerator`` 是每格曾被不同有效成員訪問的 raw numerator，
    ``visit_fraction`` 是它除以 ``valid_member_denominator`` 的條件式比例；
    ``residence_time_seconds`` 則是依軌跡區間累積的停留秒數，三者不可互相替代。
    ``input_interval_seconds`` 與 ``allocated_interval_seconds`` 都是秒（s）制時間
    provenance，前者代表 pathway 輸入涵蓋範圍，後者代表已分配到格網的範圍。

    ``first_passage_quantiles`` 保存既有分箱分位數函式產生的 age-bin midpoint
    秒數近似；它與同一格的 ``visit_numerator`` 及 ``age_bin_edges_seconds`` 綁定：
    空 cell 必須是 NaN，已訪 cell 必須是有限且精確命中某一個 canonical midpoint，
    不能保存任意有限秒數。``low_sample_mask`` 的 True cell 是
    ``visit_numerator`` 低於 ``low_sample_min_member_count`` 的格子，原始 numerator
    與比例仍完整保留。所有 NumPy 陣列與分位數 mapping 都在建構時防禦性複製；frozen
    dataclass 也防止欄位重新綁定。這些結果是條件式來源足跡／相對來源權重，不是
    絕對來源機率或因果歸因。
    """

    x_edges_m: np.ndarray
    y_edges_m: np.ndarray
    age_bin_edges_seconds: np.ndarray
    valid_member_denominator: int
    visit_numerator: np.ndarray
    visit_fraction: np.ndarray
    residence_time_seconds: np.ndarray
    first_passage_quantiles: Mapping[float, np.ndarray]
    low_sample_min_member_count: int
    low_sample_mask: np.ndarray
    input_interval_seconds: float
    allocated_interval_seconds: float

    def __post_init__(self) -> None:
        """驗證欄位關聯並把所有可變輸入封存成 canonical 唯讀快照。

        直接建構 frozen dataclass 仍可能收到 caller 持有的可寫陣列或錯誤 shape；這裡
        先由公尺格線推導唯一 cell shape，再檢查分母、訪格上限、比例範圍、低樣本
        門檻／遮罩與首次通過輸出。時間欄位則要求原生有限非負 float，並核對
        ``allocated <= input`` 以及 residence 總和與 allocated 的既有容許範圍，最後
        用 ``object.__setattr__`` 寫入獨立副本。比例的 NaN 只在零分母時合法，因為
        其他情況的 NaN 會遮蔽物理統計失敗。
        """

        x_edges = _snapshot_edges(self.x_edges_m, name="x_edges_m")
        y_edges = _snapshot_edges(self.y_edges_m, name="y_edges_m")
        age_edges = _snapshot_edges(
            self.age_bin_edges_seconds,
            name="age_bin_edges_seconds",
            require_zero_start=True,
        )
        denominator = _require_nonnegative_native_int(
            self.valid_member_denominator,
            label="valid_member_denominator",
        )
        threshold = _require_positive_native_int(
            self.low_sample_min_member_count,
            label="low_sample_min_member_count",
        )
        input_interval = _require_nonnegative_finite_native_float(
            self.input_interval_seconds,
            label="input_interval_seconds",
        )
        allocated_interval = _require_nonnegative_finite_native_float(
            self.allocated_interval_seconds,
            label="allocated_interval_seconds",
        )
        if allocated_interval > input_interval:
            raise RuntimeError("allocated_interval_seconds 不可大於 input_interval_seconds")
        shape = (y_edges.size - 1, x_edges.size - 1)
        visit_numerator = _snapshot_visit_numerator(self.visit_numerator, shape=shape)
        if visit_numerator.size and int(np.max(visit_numerator)) > denominator:
            raise ValueError("visit_numerator 不可大於 valid_member_denominator")
        residence = _snapshot_residence(self.residence_time_seconds, shape=shape)
        try:
            residence_sum = _finite_residence_sum(
                residence,
                name="PathwayGridStatistics.residence_time_seconds",
            )
        except RuntimeError as error:
            raise RuntimeError("PathwayGridStatistics residence 秒數總和無法驗證") from error
        if not np.isclose(
            residence_sum,
            allocated_interval,
            rtol=_CONSERVATION_RTOL,
            atol=_CONSERVATION_ATOL,
        ):
            conservation_error = (
                "PathwayGridStatistics.residence_time_seconds 總和與 allocated_interval_seconds 未守恆"
            )
            raise RuntimeError(conservation_error)
        fraction = _snapshot_fraction(self.visit_fraction, shape=shape)
        if denominator == 0:
            if not np.all(np.isnan(fraction)):
                raise ValueError("valid_member_denominator 為零時 visit_fraction 必須全部為 NaN")
        else:
            if np.any(np.isnan(fraction)) or np.any(fraction < 0.0) or np.any(fraction > 1.0):
                raise ValueError("visit_fraction 必須是有限且落在 0 到 1 的比例")
            expected_fraction = _visit_fraction(visit_numerator, denominator)
            if not np.array_equal(fraction, expected_fraction):
                raise ValueError("visit_fraction 必須精確等於 visit_numerator 除以有效成員分母")
        quantile_outputs = _snapshot_quantile_mapping(
            self.first_passage_quantiles,
            shape=shape,
            visit_numerator=visit_numerator,
            age_bin_edges_seconds=age_edges,
        )
        low_sample_mask = _snapshot_low_sample_mask(self.low_sample_mask, shape=shape)
        expected_low_sample_mask = np.less(visit_numerator, threshold)
        if not np.array_equal(low_sample_mask, expected_low_sample_mask):
            raise ValueError("low_sample_mask 必須精確等於 visit_numerator < low_sample_min_member_count")

        object.__setattr__(self, "x_edges_m", x_edges)
        object.__setattr__(self, "y_edges_m", y_edges)
        object.__setattr__(self, "age_bin_edges_seconds", age_edges)
        object.__setattr__(self, "valid_member_denominator", denominator)
        object.__setattr__(self, "visit_numerator", visit_numerator)
        object.__setattr__(self, "visit_fraction", fraction)
        object.__setattr__(self, "residence_time_seconds", residence)
        object.__setattr__(self, "first_passage_quantiles", quantile_outputs)
        object.__setattr__(self, "low_sample_min_member_count", threshold)
        object.__setattr__(self, "low_sample_mask", low_sample_mask)
        object.__setattr__(self, "input_interval_seconds", input_interval)
        object.__setattr__(self, "allocated_interval_seconds", allocated_interval)

    @property
    def visit_count(self) -> np.ndarray:
        """回傳舊版 ``visit_count`` 名稱的唯讀相容別名。

        正式欄位是 ``visit_numerator``，因為這個數值是 ``visit_fraction`` 的分子，
        不是可泛稱的事件 count。別名刻意直接回傳同一個唯讀陣列，既維持既有 caller
        的讀取介面，也不產生第二份可能與正式欄位分歧的統計快照。
        """

        return self.visit_numerator


def _visit_fraction(visit_numerator: np.ndarray, denominator: int) -> np.ndarray:
    """依非負有效成員分母計算比例，並在零分母時整張保留 NaN。

    NumPy 的一般除法在零分母會產生 warning 並寫入 inf/NaN 混合結果；報告契約需要
    每格都明確表示「不可估計」，所以先建立全 NaN 陣列，只在正分母時填入比例。對
    超過 float64 最大值的極大 Python 分母，使用 Python 整數除法逐格轉 float，讓
    合法但極端的原生 int 仍不會因 scalar cast 溢位而破壞建構流程。
    """

    fraction = np.full(visit_numerator.shape, np.nan, dtype=np.float64)
    if denominator == 0:
        return fraction
    try:
        denominator_float = float(denominator)
    except OverflowError:
        fraction.flat[:] = [int(value) / denominator for value in visit_numerator.flat]
    else:
        np.divide(
            visit_numerator,
            denominator_float,
            out=fraction,
            casting="unsafe",
        )
    return fraction


def _snapshot_quantiles(value: object) -> tuple[object, ...]:
    """複製 quantile 的真正 sequence，拒絕 mapping、字串與一次性 iterable。

    ``first_passage_quantiles`` 的數值驗證與 midpoint 選箱邏輯集中在既有模組；本層
    只固定輸入容器必須可重複 materialize 的 Sequence，避免 generator 被部分消費或
    mapping key 被誤當 quantile。元素型別與 0/1 邊界則交由既有函式處理。
    """

    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(value, Sequence):
        raise TypeError("quantiles 必須是非字串、非 mapping 的 sequence")
    try:
        return tuple(value)
    except Exception as error:
        raise ValueError("quantiles 無法建立 defensive tuple") from error


def build_pathway_grid_statistics(
    pathway: StreamingPathwayAggregate,
    valid_member_denominator: int,
    quantiles: Sequence[float],
    low_sample_min_member_count: int,
) -> PathwayGridStatistics:
    """由有效成員 pathway aggregate 建立報告用路徑格網統計。

    Args:
        pathway: 必須是 exact ``StreamingPathwayAggregate``。其 x/y 邊界為公尺，
            ``unique_particle_count`` 為 ``(y_cell, x_cell)`` 的每格訪格 raw numerator，
            residence 為秒，首次進入 histogram 為 ``(y_cell, x_cell, age_bin)``。
        valid_member_denominator: 有效成員分母，必須是可為零的原生 Python ``int``，且
            必須精確等於 ``pathway.input_particle_count``；零代表沒有可用成員。
        quantiles: 真正可重複使用的非字串、非 mapping ``Sequence``。內容由既有
            ``first_passage_quantiles`` 驗證，並回傳固定 age-bin midpoint 秒數近似。
        low_sample_min_member_count: 低樣本門檻，必須是正的原生 Python ``int``；每格
            以 ``visit_numerator < threshold`` 建立遮罩，遮罩不會清除原始統計。

    Returns:
        ``PathwayGridStatistics``。輸出保存公尺／秒格線、raw 訪格計數、有效分母比例、
        residence 秒數、首次進入 age quantiles 與低樣本遮罩；所有 NumPy 陣列均為
        defensive-copy 且唯讀，quantile mapping 亦不可修改。

    Raises:
        TypeError: pathway、分母、quantile 容器或門檻的型別不符合 exact contract。
        ValueError: 分母不匹配、負分母、訪格計數超過分母，或既有 quantile 函式拒絕
            age histogram／quantile 內容時。
    """

    if type(pathway) is not StreamingPathwayAggregate:
        raise TypeError("pathway 必須是 exact StreamingPathwayAggregate")
    denominator = _require_nonnegative_native_int(
        valid_member_denominator,
        label="valid_member_denominator",
    )
    if pathway.input_particle_count != denominator:
        raise ValueError("pathway.input_particle_count 必須等於 valid_member_denominator")
    quantile_values = _snapshot_quantiles(quantiles)
    threshold = _require_positive_native_int(
        low_sample_min_member_count,
        label="low_sample_min_member_count",
    )

    shape = pathway.unique_particle_count.shape
    visit_numerator = _snapshot_visit_numerator(
        pathway.unique_particle_count,
        shape=shape,
    )
    if visit_numerator.size and int(np.max(visit_numerator)) > denominator:
        raise ValueError("visit_numerator 不可大於 valid_member_denominator")

    # 呼叫既有 helper 取得 age-bin midpoint quantiles；報告層不重新實作累積計數與
    # quantile bin 選擇，避免兩個模組對同一 histogram 產生不同的空 cell／邊界語意。
    quantile_outputs = first_passage_quantiles(
        pathway.first_passage_age_histogram,
        age_bin_edges_seconds=pathway.age_bin_edges_seconds,
        quantiles=quantile_values,
    )
    fraction = _visit_fraction(visit_numerator, denominator)
    low_sample_mask = np.less(visit_numerator, threshold)

    return PathwayGridStatistics(
        x_edges_m=pathway.x_edges_m,
        y_edges_m=pathway.y_edges_m,
        age_bin_edges_seconds=pathway.age_bin_edges_seconds,
        valid_member_denominator=denominator,
        visit_numerator=visit_numerator,
        visit_fraction=fraction,
        residence_time_seconds=pathway.residence_time_seconds,
        first_passage_quantiles=quantile_outputs,
        low_sample_min_member_count=threshold,
        low_sample_mask=low_sample_mask,
        input_interval_seconds=pathway.input_interval_seconds,
        allocated_interval_seconds=pathway.allocated_interval_seconds,
    )
