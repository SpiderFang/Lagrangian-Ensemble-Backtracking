"""報告層固定規格的 KDE／HDR 頻寬敏感度產品。

本模組只接受報告層已驗證的 ``AggregateSpec``、``ReportSpec`` 與事件格網計數，
不讀取檔案、不重新載入原始 OCM／NWW3 NetCDF，也不使用 GPU。輸入計數的軸順序
固定是 ``(y_cell, x_cell)``：每一格代表固定 local metric projection 下的公尺制
平面格網，x 軸是向東的公尺座標，y 軸是向北的公尺座標；網格邊界、cell 面積與
Gaussian 頻寬全部以公尺（m）表示。這裡保存的是在指定受體、到達條件、物性與流場
條件下的格網化「條件式來源足跡」／「相對來源權重」，不是建立先驗、似然及觀測
驗證後的絕對來源機率或因果歸因。

``AggregateSpec`` 登錄三個固定 KDE 頻寬與 HDR（高密度區域，high-density region）
層級，``ReportSpec`` 登錄正文 primary bandwidth 與最低原始樣本門檻。零樣本或低於
門檻時只保留原始計數與明確狀態，不以零值假造密度；達到門檻後才把格網計數視為
格心質量，交給既有純 CPU ``binned_gaussian_kde_2d`` 建立平滑機率、面積密度與
HDR 遮罩。這種格網化與邊界內重新正規化本身仍是數值近似，不能取代 SERVER 上
已驗收 OCM schema 3／NWW3 schema 1 資料的正式科學驗證。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final

import numpy as np

from . import aggregation as aggregation_module
from .aggregate_spec import AggregateSpec
from .report_spec import ReportSpec, validate_report_spec_against_aggregate_spec

# 這個名稱讓測試與下游可以明確辨識 wrapper 使用的既有 binned core；實際呼叫透過
# 下面的 forwarding function，因此 monkeypatch core 模組或本模組的函式都能觀察呼叫。
BinnedKDEGrid = aggregation_module.BinnedKDEGrid

__all__ = [
    "BinnedKDEGrid",
    "KDEEstimateStatus",
    "KDEBandwidthLayer",
    "KDESensitivityProduct",
    "build_kde_sensitivity_product",
]


_INT64_MAX: Final[int] = int(np.iinfo(np.int64).max)
_EDGE_RTOL: Final[float] = 1.0e-12
_EDGE_ATOL: Final[float] = 1.0e-12
_GRID_ALIGNMENT_RTOL: Final[float] = 1.0e-9
_GRID_ALIGNMENT_ATOL: Final[float] = 1.0e-9


class KDEEstimateStatus(StrEnum):
    """描述固定 KDE 頻寬層是否具有足夠原始格網樣本。

    ``available`` 代表 raw count 已達 ``ReportSpec.minimum_kde_raw_count``，因此
    ``grid`` 具有可檢查的平滑結果；``zero_raw_count`` 代表整張 `(y, x)` 計數格網
    沒有原始交點；``below_minimum_raw_count`` 代表有交點但樣本太少，不建立假 KDE。
    狀態和數值分開保存，避免 renderer 以零密度或 NaN 猜測「沒有樣本」與「低樣本」
    的不同語意。大寫名稱只是既有 Enum 使用習慣的別名，不增加新的狀態。
    """

    available = "available"
    zero_raw_count = "zero_raw_count"
    below_minimum_raw_count = "below_minimum_raw_count"

    AVAILABLE = available
    ZERO_RAW_COUNT = zero_raw_count
    BELOW_MINIMUM_RAW_COUNT = below_minimum_raw_count


def _require_native_float(value: object, *, label: str) -> float:
    """要求有限、正值的原生 Python ``float``，拒絕 bool 與 NumPy scalar。"""

    if type(value) is not float:
        raise TypeError(f"{label} 必須是原生 Python float")
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{label} 必須是有限正值")
    return value


def _require_nonnegative_native_int(value: object, *, label: str) -> int:
    """要求可為零、且不超過 signed ``int64`` 的原生 Python 整數。"""

    if type(value) is not int:
        raise TypeError(f"{label} 必須是原生 Python int，不接受 bool 或 NumPy scalar")
    if value < 0:
        raise ValueError(f"{label} 不可為負值")
    if value > _INT64_MAX:
        raise ValueError(f"{label} 不得超過 signed int64 上限")
    return value


def _require_raw_point_count(value: object, *, label: str) -> int:
    """驗證 raw point 總數的 Python 整數型別與非負性。

    raw point 總數可能是多個合法 int64 cell 相加後的任意精度 Python 整數，因此
    這裡不把總和再次限制在 int64；每個 cell 的上限由 raw-count snapshot 負責。
    """

    if type(value) is not int:
        raise TypeError(f"{label} 必須是非負的原生 Python int")
    if value < 0:
        raise ValueError(f"{label} 不可為負值")
    return value


def _snapshot_edges(value: object, *, label: str) -> np.ndarray:
    """建立有限、嚴格遞增且均勻的 float64 公尺制邊界唯讀副本。

    x/y 邊界決定 `(y_cell, x_cell)` 的 shape 與每格面積；即使 caller 傳入可寫陣列，
    報告產品也必須擁有自己的記憶體，才能避免後續 renderer 或測試修改座標契約。
    """

    try:
        raw = np.asarray(value)
    except Exception:
        raise ValueError(f"{label} 必須是一維數值格線") from None
    if raw.ndim != 1 or raw.size < 2 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{label} 必須是至少兩點的一維實數格線")
    try:
        copied = np.array(raw, dtype=np.float64, copy=True)
    except Exception:
        raise ValueError(f"{label} 必須可安全轉為 float64") from None
    if not np.all(np.isfinite(copied)):
        raise ValueError(f"{label} 必須全部有限")
    widths = np.diff(copied)
    if not np.all(np.isfinite(widths)) or np.any(widths <= 0.0):
        raise ValueError(f"{label} 必須嚴格遞增")
    if not np.allclose(widths, widths[0], rtol=_EDGE_RTOL, atol=_EDGE_ATOL):
        raise ValueError(f"{label} 必須均勻")
    copied.setflags(write=False)
    return copied


def _snapshot_raw_count(value: object, *, shape: tuple[int, int], label: str) -> np.ndarray:
    """安全複製 `(y, x)` 原始計數為 signed int64 唯讀陣列。

    先逐格轉成 Python ``int`` 檢查負值與 signed int64 上限，再進行 NumPy 轉型；
    這個順序特別防止 uint64 大值先轉型而靜默繞回負數。raw count 是事件或交點的
    原始計數，不是面積密度，也不能用零替代資料缺口或未取樣狀態。
    """

    try:
        raw = np.asarray(value)
    except Exception:
        raise ValueError(f"{label} 必須是二維整數矩陣") from None
    if raw.shape != shape:
        raise ValueError(f"{label} 形狀必須精確為 {shape}")
    if raw.dtype.kind not in "iu":
        raise TypeError(f"{label} 必須是整數 dtype，不接受浮點或 bool")
    for item in raw.flat:
        count = int(item)
        if count < 0:
            raise ValueError(f"{label} 不可包含負值")
        if count > _INT64_MAX:
            raise ValueError(f"{label} 元素超過 signed int64 上限")
    try:
        copied = np.array(raw, dtype=np.int64, copy=True)
    except Exception:
        raise ValueError(f"{label} 無法安全轉成 int64") from None
    copied.setflags(write=False)
    return copied


def _python_count_sum(raw_count: np.ndarray) -> int:
    """以 Python 任意精度逐格累加 raw count，避免固定寬度總和繞回。"""

    return sum(int(item) for item in raw_count.flat)


def _snapshot_nonnegative_float_grid(
    value: object,
    *,
    shape: tuple[int, int],
    label: str,
) -> np.ndarray:
    """建立密度或機率格網的有限非負 float64 唯讀副本。"""

    try:
        raw = np.asarray(value)
    except Exception:
        raise ValueError(f"{label} 必須是二維實數格網") from None
    if raw.shape != shape or raw.dtype.kind not in "iuf":
        raise ValueError(f"{label} 必須是形狀 {shape} 的實數格網")
    try:
        copied = np.array(raw, dtype=np.float64, copy=True)
    except Exception:
        raise ValueError(f"{label} 必須可安全轉成 float64") from None
    if not np.all(np.isfinite(copied)) or np.any(copied < 0.0):
        raise ValueError(f"{label} 必須全部有限且非負")
    copied.setflags(write=False)
    return copied


def _snapshot_hdr_masks(
    value: object,
    *,
    shape: tuple[int, int],
    cell_probability: np.ndarray,
    expected_levels: tuple[float, ...] | None = None,
) -> Mapping[float, np.ndarray]:
    """依既有 core 的 exact 規則驗證 HDR mapping、shape、巢狀性與選格結果。

    HDR mask 的 `True` cell 是依機率排序後納入指定累積質量門檻的格子。這裡驗證
    它們必須精確等於既有 binned core 以 C-order flatten、stable descending sort、
    cumulative probability 與 ``searchsorted(side="left") + 1`` 產生的 mask，並
    額外維持固定 level 順序與由低至高的巢狀性。只檢查 true-count 或大約累積質量
    會允許選錯 cell 的 tamper，因此不能取代 exact array equality。mask 只是報告
    遮罩，不能把原始計數或條件式來源權重重新解讀為絕對來源機率。
    """

    if not isinstance(value, Mapping):
        raise TypeError("hdr_masks 必須是 mapping")
    try:
        items = tuple(value.items())
    except Exception:
        raise ValueError("hdr_masks mapping 無法讀取") from None
    levels = tuple(key for key, _ in items)
    if expected_levels is not None and levels != expected_levels:
        raise ValueError("hdr_masks 的 key 順序必須與 hdr_levels 完全一致")
    if not levels:
        raise ValueError("hdr_masks 不可為空")
    for index, level in enumerate(levels):
        if type(level) is not float:
            raise TypeError("hdr_masks 的 key 必須是原生 Python float")
        if not math.isfinite(level) or not 0.0 < level < 1.0:
            raise ValueError("hdr_masks 的 level 必須是介於零與一之間的有限值")
        if index and not levels[index - 1] < level:
            raise ValueError("hdr_masks 的 level 必須嚴格遞增")

    copied_masks: dict[float, np.ndarray] = {}
    for level, mask in items:
        try:
            raw = np.asarray(mask)
        except Exception:
            raise ValueError("HDR mask 必須是二維 bool 陣列") from None
        if raw.shape != shape or raw.dtype.kind != "b":
            raise TypeError("HDR mask 必須是指定 shape 的 bool 陣列")
        copied = np.array(raw, dtype=np.bool_, copy=True)
        copied.setflags(write=False)
        copied_masks[level] = copied

    # 這段刻意逐字對齊 aggregation.binned_gaussian_kde_2d 的 HDR 核心規則：
    # C-order row-major flatten 決定相同機率的 tie-break，stable sort 保留該順序，
    # cumulative mass 再以 left insertion point 加一格。如此即使 attacker 交給
    # wrapper 一組 shape、true-count 與 nested 都合法的錯誤 mask，也會因選格不同而
    # fail closed；所有 level 共用同一份排序，不能各自選另一個排序。
    flat_probability = cell_probability.ravel(order="C")
    descending_order = np.argsort(-flat_probability, kind="stable")
    cumulative_probability = np.cumsum(
        flat_probability[descending_order],
        dtype=np.float64,
    )
    expected_masks: dict[float, np.ndarray] = {}
    for level in levels:
        selected_count = int(np.searchsorted(cumulative_probability, level, side="left")) + 1
        selected_count = min(selected_count, descending_order.size)
        expected_flat = np.zeros(flat_probability.size, dtype=np.bool_)
        expected_flat[descending_order[:selected_count]] = True
        expected_masks[level] = expected_flat.reshape(shape)

    ordered_masks = tuple(copied_masks.values())
    for lower, upper in zip(ordered_masks, ordered_masks[1:], strict=False):
        if not np.all(lower <= upper):
            raise ValueError("HDR mask 必須依 level 維持巢狀關係")
    for level, mask in copied_masks.items():
        if not np.array_equal(mask, expected_masks[level]):
            raise ValueError("HDR mask 必須精確符合 stable highest-density cell 選取規則")
    return MappingProxyType(copied_masks)


def _grid_cell_area(x_edges_m: np.ndarray, y_edges_m: np.ndarray) -> np.ndarray:
    """計算 `(y, x)` 每格公尺平方面積，並拒絕溢位或非正值。"""

    with np.errstate(over="ignore", invalid="ignore"):
        cell_area = np.diff(y_edges_m)[:, None] * np.diff(x_edges_m)[None, :]
    if not np.all(np.isfinite(cell_area)) or np.any(cell_area <= 0.0):
        raise ValueError("格網 cell 面積必須是有限正值")
    return cell_area


def _snapshot_binned_grid(
    grid: object,
    *,
    expected_x_edges_m: np.ndarray | None = None,
    expected_y_edges_m: np.ndarray | None = None,
    expected_raw_count: np.ndarray | None = None,
    expected_bandwidth_m: float | None = None,
    expected_raw_point_count: int | None = None,
    expected_hdr_levels: tuple[float, ...] | None = None,
) -> BinnedKDEGrid:
    """把 shallow-frozen core 結果轉成每層獨立的深防禦 snapshot。

    ``aggregation.BinnedKDEGrid`` 只凍結 dataclass 欄位，內部 NumPy 陣列與 HDR dict
    仍可能被 caller 改寫。報告層在這裡重新複製全部陣列、關閉寫入權限、封存 mapping，
    並核對機率、密度、raw count、頻寬與 product 的固定幾何；因此三個頻寬層不會
    共享 caller 或彼此的可變 buffer。
    """

    if type(grid) is not BinnedKDEGrid:
        raise TypeError("grid 必須是 exact BinnedKDEGrid")
    x_edges = _snapshot_edges(grid.x_edges_m, label="grid.x_edges_m")
    y_edges = _snapshot_edges(grid.y_edges_m, label="grid.y_edges_m")
    if expected_x_edges_m is not None and not np.array_equal(x_edges, expected_x_edges_m):
        raise ValueError("grid.x_edges_m 必須等於 product x_edges_m")
    if expected_y_edges_m is not None and not np.array_equal(y_edges, expected_y_edges_m):
        raise ValueError("grid.y_edges_m 必須等於 product y_edges_m")

    shape = (y_edges.size - 1, x_edges.size - 1)
    raw_count = _snapshot_raw_count(grid.raw_count, shape=shape, label="grid.raw_count")
    raw_point_count = _require_raw_point_count(
        grid.raw_point_count,
        label="grid.raw_point_count",
    )
    if raw_point_count != _python_count_sum(raw_count):
        raise ValueError("grid.raw_point_count 必須等於 grid.raw_count 的逐格總和")
    if expected_raw_count is not None and not np.array_equal(raw_count, expected_raw_count):
        raise ValueError("grid.raw_count 必須等於 product raw_count")
    if expected_raw_point_count is not None and raw_point_count != expected_raw_point_count:
        raise ValueError("grid.raw_point_count 必須等於 product raw_point_count")

    bandwidth = _require_native_float(grid.bandwidth_m, label="grid.bandwidth_m")
    if expected_bandwidth_m is not None and bandwidth != expected_bandwidth_m:
        raise ValueError("grid.bandwidth_m 必須等於 layer bandwidth_m")

    density = _snapshot_nonnegative_float_grid(
        grid.density_per_m2,
        shape=shape,
        label="grid.density_per_m2",
    )
    probability = _snapshot_nonnegative_float_grid(
        grid.cell_probability,
        shape=shape,
        label="grid.cell_probability",
    )
    probability_total = float(np.sum(probability, dtype=np.float64))
    if not math.isfinite(probability_total) or not math.isclose(
        probability_total,
        1.0,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("grid.cell_probability 總和必須為 1")
    cell_area = _grid_cell_area(x_edges, y_edges)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        expected_density = probability / cell_area
    if not np.all(np.isfinite(expected_density)) or np.any(expected_density < 0.0):
        raise ValueError("由 cell_probability 推導的 density 必須有限且非負")
    if not np.array_equal(density, expected_density):
        raise ValueError("grid.density_per_m2 必須等於 cell_probability 除以 cell area")

    hdr_masks = _snapshot_hdr_masks(
        grid.hdr_masks,
        shape=shape,
        cell_probability=probability,
        expected_levels=expected_hdr_levels,
    )
    return BinnedKDEGrid(
        x_edges_m=x_edges,
        y_edges_m=y_edges,
        raw_count=raw_count,
        density_per_m2=density,
        cell_probability=probability,
        hdr_masks=hdr_masks,  # type: ignore[arg-type]
        bandwidth_m=bandwidth,
        raw_point_count=raw_point_count,
    )


def _derive_axis_edges(
    lower_value: object,
    upper_value: object,
    cell_size_value: object,
    *,
    axis_label: str,
) -> np.ndarray:
    """由站點 min/max 與公尺 cell size 建立固定 float64 邊界。

    前 ``n`` 個 edge 嚴格使用 ``min + arange(n) * cell_size``，最後一點強制寫成
    ``max``，以保存 aggregate spec 的端點契約。``n`` 必須是正整數，所有產生的
    邊界與寬度都再次經過有限、遞增及均勻檢查，避免浮點累積誤差進入既有 binned core。
    """

    try:
        lower = float(lower_value)
        upper = float(upper_value)
        cell_size = float(cell_size_value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{axis_label} 軸範圍或 cell size 無法轉成有限公尺值") from None
    if not all(math.isfinite(value) for value in (lower, upper, cell_size)):
        raise ValueError(f"{axis_label} 軸範圍與 cell size 必須有限")
    if cell_size <= 0.0 or not upper > lower:
        raise ValueError(f"{axis_label} 軸範圍與 cell size 不合法")
    width = upper - lower
    ratio = width / cell_size
    if not math.isfinite(ratio):
        raise ValueError(f"{axis_label} 軸 cell 數必須有限")
    cell_count = round(ratio)
    if cell_count <= 0 or not math.isclose(
        ratio,
        cell_count,
        rel_tol=_GRID_ALIGNMENT_RTOL,
        abs_tol=_GRID_ALIGNMENT_ATOL,
    ):
        raise ValueError(f"{axis_label} 軸 cell 數必須是正整數")
    try:
        offsets = np.arange(cell_count, dtype=np.float64)
        edges = np.empty(cell_count + 1, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            edges[:-1] = lower + offsets * cell_size
        edges[-1] = upper
    except (MemoryError, OverflowError, ValueError):
        raise ValueError(f"{axis_label} 軸邊界無法建立") from None
    if not np.all(np.isfinite(edges)):
        raise ValueError(f"{axis_label} 軸邊界必須全部有限")
    widths = np.diff(edges)
    if not np.all(np.isfinite(widths)) or np.any(widths <= 0.0):
        raise ValueError(f"{axis_label} 軸邊界必須嚴格遞增")
    if not np.allclose(widths, cell_size, rtol=_EDGE_RTOL, atol=_EDGE_ATOL):
        raise ValueError(f"{axis_label} 軸邊界必須均勻")
    edges.setflags(write=False)
    return edges


def _expected_status(raw_point_count: int, minimum_raw_count: int) -> KDEEstimateStatus:
    """依 raw total 與 report 門檻取得固定三態分類。"""

    if raw_point_count == 0:
        return KDEEstimateStatus.zero_raw_count
    if raw_point_count < minimum_raw_count:
        return KDEEstimateStatus.below_minimum_raw_count
    return KDEEstimateStatus.available


def binned_gaussian_kde_2d(
    raw_count: np.ndarray,
    *,
    x_edges_m: np.ndarray,
    y_edges_m: np.ndarray,
    bandwidth_m: float,
    hdr_levels: Sequence[float],
) -> BinnedKDEGrid:
    """轉送至既有純 CPU binned KDE core，不改變其 RuntimeError 語意。

    這個薄 forwarding 介面只為讓報告層的呼叫點可被測試觀察；真正的 Gaussian
    平滑、HDR 排序與格網內機率正規化仍完全由 ``aggregation`` 模組負責。呼叫端的
    raw count 與 x/y 邊界在 builder 中逐層另行複製，避免 core 或測試替身共享 buffer。
    """

    return aggregation_module.binned_gaussian_kde_2d(
        raw_count,
        x_edges_m=x_edges_m,
        y_edges_m=y_edges_m,
        bandwidth_m=bandwidth_m,
        hdr_levels=hdr_levels,
    )


@dataclass(frozen=True, slots=True)
class KDEBandwidthLayer:
    """單一公尺制 KDE 頻寬的 immutable 結果層。

    ``raw_point_count`` 是 `(y, x)` 原始交點計數的 Python 整數總和；只有
    ``available`` 層允許保存 ``BinnedKDEGrid``。``zero_raw_count`` 與
    ``below_minimum_raw_count`` 必須保留 `grid=None`，因為低樣本不應被圖面或報表
    誤讀為經過 KDE 估計的零密度。若有 grid，所有陣列會在此邊界重新複製成唯讀，
    其公尺邊界、機率、密度與 HDR mask 也會通過一致性檢查。
    """

    bandwidth_m: float
    status: KDEEstimateStatus
    raw_point_count: int
    grid: BinnedKDEGrid | None

    def __post_init__(self) -> None:
        """驗證狀態／樣本／grid 關係並建立獨立的 core snapshot。"""

        bandwidth = _require_native_float(self.bandwidth_m, label="bandwidth_m")
        if type(self.status) is not KDEEstimateStatus:
            raise TypeError("status 必須是 KDEEstimateStatus")
        raw_point_count = _require_raw_point_count(
            self.raw_point_count,
            label="raw_point_count",
        )
        if self.status is KDEEstimateStatus.zero_raw_count:
            if raw_point_count != 0 or self.grid is not None:
                raise ValueError("zero_raw_count 必須有零總數且 grid=None")
        elif self.status is KDEEstimateStatus.below_minimum_raw_count:
            if raw_point_count <= 0 or self.grid is not None:
                raise ValueError("below_minimum_raw_count 必須有正總數且 grid=None")
        else:
            if raw_point_count <= 0 or self.grid is None:
                raise ValueError("available 必須有正總數與 grid")

        grid = None
        if self.grid is not None:
            grid = _snapshot_binned_grid(
                self.grid,
                expected_bandwidth_m=bandwidth,
                expected_raw_point_count=raw_point_count,
            )
        object.__setattr__(self, "bandwidth_m", bandwidth)
        object.__setattr__(self, "raw_point_count", raw_point_count)
        object.__setattr__(self, "grid", grid)


@dataclass(frozen=True, slots=True)
class KDESensitivityProduct:
    """保存同一固定格網下三個頻寬的 KDE／HDR 敏感度產品。

    x/y 邊界與每格 raw count 的軸順序固定是 x 邊界、y 邊界及 `(y_cell, x_cell)`；
    邊界與密度面積單位是公尺（m、m²），raw count 是原始交點數。``layers`` 的
    mapping 順序必須與 ``bandwidths_m`` 完全相同，``primary_bandwidth_m`` 必須是
    其中一層；這使 renderer 不需臨時挑帶寬。每個 layer 都會重新建立獨立 grid
    snapshot，並檢查與 product 的邊界、raw count、HDR level、機率和密度關係。

    產品只描述條件式來源足跡／相對來源權重。零樣本和低樣本產品仍保存 raw count，
    但不提供假造的 KDE；即使 available，HDR 也只是格網內平滑質量的累積遮罩，不是
    絕對來源機率或因果歸因。
    """

    site_id: str
    x_edges_m: np.ndarray
    y_edges_m: np.ndarray
    raw_count: np.ndarray
    raw_point_count: int
    bandwidths_m: tuple[float, ...]
    hdr_levels: tuple[float, ...]
    primary_bandwidth_m: float
    minimum_raw_count: int
    layers: Mapping[float, KDEBandwidthLayer]

    def __post_init__(self) -> None:
        """對直接 dataclass 建構也執行完整的型別、shape、mapping 與數值 fail-closed 驗證。"""

        if type(self.site_id) is not str or not self.site_id:
            raise TypeError("site_id 必須是非空原生 str")
        x_edges = _snapshot_edges(self.x_edges_m, label="x_edges_m")
        y_edges = _snapshot_edges(self.y_edges_m, label="y_edges_m")
        shape = (y_edges.size - 1, x_edges.size - 1)
        raw_count = _snapshot_raw_count(self.raw_count, shape=shape, label="raw_count")
        raw_point_count = _require_raw_point_count(
            self.raw_point_count,
            label="raw_point_count",
        )
        if raw_point_count != _python_count_sum(raw_count):
            raise ValueError("raw_point_count 必須等於 raw_count 的逐格總和")

        if type(self.bandwidths_m) is not tuple:
            raise TypeError("bandwidths_m 必須是 tuple")
        bandwidths = tuple(self.bandwidths_m)
        if len(bandwidths) != 3:
            raise ValueError("bandwidths_m 必須包含 aggregate spec 的三個頻寬")
        for bandwidth in bandwidths:
            _require_native_float(bandwidth, label="bandwidths_m item")
        if not bandwidths[0] < bandwidths[1] < bandwidths[2]:
            raise ValueError("bandwidths_m 必須嚴格遞增")

        if type(self.hdr_levels) is not tuple:
            raise TypeError("hdr_levels 必須是 tuple")
        hdr_levels = tuple(self.hdr_levels)
        if not hdr_levels:
            raise ValueError("hdr_levels 不可為空")
        for level in hdr_levels:
            if type(level) is not float:
                raise TypeError("hdr_levels item 必須是原生 Python float")
            if not math.isfinite(level) or not 0.0 < level < 1.0:
                raise ValueError("hdr_levels 必須是介於零與一之間的有限值")
        if any(not left < right for left, right in zip(hdr_levels, hdr_levels[1:], strict=False)):
            raise ValueError("hdr_levels 必須嚴格遞增")

        primary_bandwidth = _require_native_float(
            self.primary_bandwidth_m,
            label="primary_bandwidth_m",
        )
        if primary_bandwidth not in bandwidths:
            raise ValueError("primary_bandwidth_m 必須精確對應 bandwidths_m 的一層")
        if type(self.minimum_raw_count) is not int:
            raise TypeError("minimum_raw_count 必須是正的原生 Python int")
        if self.minimum_raw_count <= 0:
            raise ValueError("minimum_raw_count 必須大於零")

        if not isinstance(self.layers, Mapping):
            raise TypeError("layers 必須是 mapping")
        try:
            layer_items = tuple(self.layers.items())
        except Exception:
            raise ValueError("layers mapping 無法讀取") from None
        layer_keys = tuple(key for key, _ in layer_items)
        if any(type(key) is not float for key in layer_keys):
            raise TypeError("layers key 必須是原生 Python float")
        if layer_keys != bandwidths:
            raise ValueError("layers key 與順序必須完全等於 bandwidths_m")

        expected_status = _expected_status(raw_point_count, self.minimum_raw_count)
        ordered_layers: dict[float, KDEBandwidthLayer] = {}
        for bandwidth, layer in layer_items:
            if type(layer) is not KDEBandwidthLayer:
                raise TypeError("layers value 必須是 exact KDEBandwidthLayer")
            if layer.bandwidth_m != bandwidth:
                raise ValueError("layer bandwidth_m 必須等於 mapping key")
            if layer.status is not expected_status:
                raise ValueError("所有 layer status 必須與 product raw count/minimum 一致")
            if layer.raw_point_count != raw_point_count:
                raise ValueError("所有 layer raw_point_count 必須與 product 一致")
            if expected_status is KDEEstimateStatus.available:
                if layer.grid is None:
                    raise ValueError("available layer 必須有 grid")
                grid = _snapshot_binned_grid(
                    layer.grid,
                    expected_x_edges_m=x_edges,
                    expected_y_edges_m=y_edges,
                    expected_raw_count=raw_count,
                    expected_bandwidth_m=bandwidth,
                    expected_raw_point_count=raw_point_count,
                    expected_hdr_levels=hdr_levels,
                )
            else:
                if layer.grid is not None:
                    raise ValueError("unavailable layer 的 grid 必須是 None")
                grid = None
            # 重新建構 layer 會再做一次自身契約驗證，並確保即使 caller 讓不同 key
            # 指向同一個 layer/grid，也不會把同一組 mutable buffers 帶入兩個頻寬層。
            ordered_layers[bandwidth] = KDEBandwidthLayer(
                bandwidth_m=bandwidth,
                status=layer.status,
                raw_point_count=raw_point_count,
                grid=grid,
            )

        object.__setattr__(self, "x_edges_m", x_edges)
        object.__setattr__(self, "y_edges_m", y_edges)
        object.__setattr__(self, "raw_count", raw_count)
        object.__setattr__(self, "raw_point_count", raw_point_count)
        object.__setattr__(self, "bandwidths_m", bandwidths)
        object.__setattr__(self, "hdr_levels", hdr_levels)
        object.__setattr__(self, "primary_bandwidth_m", primary_bandwidth)
        object.__setattr__(self, "minimum_raw_count", self.minimum_raw_count)
        object.__setattr__(self, "layers", MappingProxyType(ordered_layers))


def build_kde_sensitivity_product(
    raw_count: np.ndarray,
    *,
    site_id: str,
    aggregate_spec: AggregateSpec,
    report_spec: ReportSpec,
) -> KDESensitivityProduct:
    """建立固定規格的三頻寬 KDE/HDR 報告產品。

    ``raw_count`` 來自已驗證 aggregate event grid，軸順序為 `(y_cell, x_cell)`，
    每格是公尺制 local grid 中的原始交點整數計數。函式先以 exact class 與既有
    report/aggregate binding 封閉 run、canonical hash 及正文 primary bandwidth，再由
    site grid 的 min/max 與 cell size 推導邊界。總數為零或低於 report 最低樣本時，
    三個頻寬只產生明確 unavailable layer；達門檻才逐一呼叫既有純 CPU binned core。

    這個 builder 不接受 renderer 傳入臨時 bandwidth、HDR 或 minimum count，也不把
    原始格網展平成 raw points；因此適合大型系集先在 aggregate 層做 bounded count
    後再建立報告產品。local synthetic 測試只能提供 synthetic engineering evidence，
    不代表 SERVER 真實 OCM／NWW3 科學成果。

    Raises:
        TypeError／ValueError: 規格、站點、格線、計數或產品交叉欄位不符合契約時。
        RuntimeError: 既有 binned core 回報數值失敗時，保持 core 的原樣例外語意。
    """

    if type(aggregate_spec) is not AggregateSpec:
        raise TypeError("aggregate_spec 必須是 exact AggregateSpec")
    if type(report_spec) is not ReportSpec:
        raise TypeError("report_spec 必須是 exact ReportSpec")
    # 必須在讀取 site/grid 前完成既有 binding；這也避免 caller 以不同 run/hash 的
    # report spec 只替換 primary 或 minimum，讓 renderer 脫離正式固定設定。
    validate_report_spec_against_aggregate_spec(report_spec, aggregate_spec)

    if type(site_id) is not str:
        raise TypeError("site_id 必須是原生 str")
    if site_id not in aggregate_spec.site_grids:
        raise ValueError("site_id 必須精確存在於 aggregate_spec.site_grids")
    site_grid = aggregate_spec.site_grids[site_id]
    x_edges = _derive_axis_edges(
        site_grid.x_min_m,
        site_grid.x_max_m,
        aggregate_spec.grid_cell_size_m,
        axis_label="x",
    )
    y_edges = _derive_axis_edges(
        site_grid.y_min_m,
        site_grid.y_max_m,
        aggregate_spec.grid_cell_size_m,
        axis_label="y",
    )
    shape = (y_edges.size - 1, x_edges.size - 1)
    counts = _snapshot_raw_count(raw_count, shape=shape, label="raw_count")
    raw_point_count = _python_count_sum(counts)
    bandwidths = tuple(float(value) for value in aggregate_spec.kde_bandwidths_m)
    hdr_levels = tuple(float(value) for value in aggregate_spec.hdr_levels)
    minimum_raw_count = report_spec.minimum_kde_raw_count
    status = _expected_status(raw_point_count, minimum_raw_count)

    layers: dict[float, KDEBandwidthLayer] = {}
    if status is KDEEstimateStatus.available:
        for bandwidth in bandwidths:
            # 每次呼叫都傳入新的 snapshot；即使測試替身或未來 core 修改輸入，也不會
            # 污染下一個 bandwidth 或 product 的原始計數。RuntimeError 刻意不攔截，
            # 讓既有 core 的數值失敗原樣回到正式 pipeline。
            grid = binned_gaussian_kde_2d(
                np.array(counts, dtype=np.int64, copy=True),
                x_edges_m=np.array(x_edges, dtype=np.float64, copy=True),
                y_edges_m=np.array(y_edges, dtype=np.float64, copy=True),
                bandwidth_m=bandwidth,
                hdr_levels=hdr_levels,
            )
            layers[bandwidth] = KDEBandwidthLayer(
                bandwidth_m=bandwidth,
                status=status,
                raw_point_count=raw_point_count,
                grid=grid,
            )
    else:
        for bandwidth in bandwidths:
            layers[bandwidth] = KDEBandwidthLayer(
                bandwidth_m=bandwidth,
                status=status,
                raw_point_count=raw_point_count,
                grid=None,
            )

    return KDESensitivityProduct(
        site_id=site_id,
        x_edges_m=x_edges,
        y_edges_m=y_edges,
        raw_count=counts,
        raw_point_count=raw_point_count,
        bandwidths_m=bandwidths,
        hdr_levels=hdr_levels,
        primary_bandwidth_m=float(report_spec.primary_kde_bandwidth_m),
        minimum_raw_count=minimum_raw_count,
        layers=layers,
    )
