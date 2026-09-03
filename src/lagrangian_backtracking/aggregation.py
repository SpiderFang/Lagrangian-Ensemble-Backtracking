"""把大量粒子結果整理成可解讀的來源足跡統計。

所有比例都同時保留原始粒子數和有效成員分母，避免只看百分比而不知道樣本有多少。
本模組的空間密度只表示「在指定受體、到達時間、物性和流場條件下，粒子較常出現的
位置」，不是絕對來源機率。核密度估計（KDE）將離散交點轉成平滑地圖；高密度區（HDR）
則圈出累積包含 50%、75% 或 90% 質量的格網。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde

from .engine import ParticleResult
from .models import BoundaryEvent, EventType, ParticleStatus


@dataclass(frozen=True, slots=True)
class KDEGrid:
    """公尺制格網上的密度、每格機率和高密度區遮罩。"""

    x_centers_m: np.ndarray
    y_centers_m: np.ndarray
    density_per_m2: np.ndarray
    cell_probability: np.ndarray
    hdr_masks: dict[float, np.ndarray]
    bandwidth_factor: float
    raw_point_count: int


@dataclass(frozen=True, slots=True)
class BinnedKDEGrid:
    """保存格網化條件式來源足跡的公尺制產品。

    ``raw_count`` 的軸順序固定為 ``(y_cell, x_cell)``，每個元素是 caller 逐批次或逐
    shard 累加的原始交點整數計數；本資料結構不保存原始粒子點集合。``cell_probability``
    是高斯平滑後在目前格網內重新正規化的相對來源權重，``density_per_m2`` 則以每格的
    實際公尺平方面積換算為面積密度。x/y 邊界、頻寬與密度的長度單位都是公尺，故結果可
    與其他公尺制空間產品對齊。

    這個結果是把離散格網計數視為格心質量、再以高斯核平滑的計算近似；零值邊界會截去
    格網外的高斯尾部，輸出的機率只對格網內保留下來的質量重新正規化。它描述的是在
    指定受體、到達條件、物性與流場條件下的格網化條件式來源足跡／相對來源權重，不是
    建立先驗、似然和觀測驗證後的絕對來源機率，也不補回域外或未觀測來源的質量。
    """

    x_edges_m: np.ndarray
    y_edges_m: np.ndarray
    raw_count: np.ndarray
    density_per_m2: np.ndarray
    cell_probability: np.ndarray
    hdr_masks: dict[float, np.ndarray]
    bandwidth_m: float
    raw_point_count: int


@dataclass(frozen=True, slots=True)
class PathwayGrid:
    """軌跡格網中的不重複粒子數與停留秒數。

    兩個量不能互換：每條粒子軌跡在同一格最多只算一次；停留時間則依相鄰觀測點真正相隔
    的秒數分配。兩點間暫時視為直線，並在每個格線交點切開，因此粒子跨越格線時，不會把
    全部時間錯算到步末那一格。
    """

    x_edges_m: np.ndarray
    y_edges_m: np.ndarray
    unique_particle_count: np.ndarray
    residence_time_seconds: np.ndarray
    input_particle_count: int
    input_interval_seconds: float
    allocated_interval_seconds: float


@dataclass(frozen=True, slots=True)
class BoundaryArclengthHistogram:
    """單一開放邊界線段上的交點數、每公尺密度和有效成員分母。"""

    boundary_segment_id: str
    s_edges_m: np.ndarray
    raw_count: np.ndarray
    count_density_per_m: np.ndarray
    conditional_fraction_per_m: np.ndarray
    valid_member_denominator: int


def binned_gaussian_kde_2d(
    raw_count: np.ndarray,
    *,
    x_edges_m: np.ndarray,
    y_edges_m: np.ndarray,
    bandwidth_m: float,
    hdr_levels: Sequence[float] = (0.50, 0.75, 0.90),
) -> BinnedKDEGrid:
    """由二維格網計數建立高斯平滑的條件式來源足跡與高密度區遮罩。

    ``raw_count`` 的形狀與軸意義固定為
    ``(len(y_edges_m) - 1, len(x_edges_m) - 1)``；每格代表公尺制 x/y 格網內的原始
    交點數。函式只接收逐批次或逐 shard 累加後的計數矩陣，不需要再次載入原始點集合，
    適合大型系集的串流聚合。x/y 邊界必須是有限、嚴格遞增且等距的公尺制格線，才能把
    物理頻寬正確換算成兩個格網軸上的標準差。

    平滑明確使用 ``scipy.ndimage.gaussian_filter``，輸入為浮點計數，且
    ``sigma=(bandwidth_m / y_step, bandwidth_m / x_step)``、
    ``mode="constant"``、``cval=0``。零值邊界代表格網外不納入本次產品，可能截去高斯
    尾部；因此平滑後只將格網內保留的質量重新正規化，使 ``cell_probability`` 總和為 1。
    ``density_per_m2`` 是 ``cell_probability`` 除以每格真實公尺平方面積，x/y 格距不必
    相同。

    高密度區（highest-density region，HDR）依每格機率遞減排列，選取累積質量首次達到
    各門檻所需的最小格數。排序採穩定排序，且以 C-order 展平的 row-major 順序作為相同
    機率的決定性平手順序；所有門檻共用同一排列，故遮罩必然巢狀。輸入計數會防禦性複製
    為 ``int64``，避免 caller 修改原矩陣後污染結果。

    所有機率、密度與 HDR 只表示指定受體、到達條件、物性及流場條件下的格網化條件式
    來源足跡／相對來源權重；未建立先驗、似然與觀測驗證前，不得解讀為絕對來源機率或
    因果歸因。格網化與高斯平滑本身也是近似，解析度、頻寬和邊界截斷都會影響結果。

    Args:
        raw_count: 非負整數的 ``(y_cell, x_cell)`` 原始計數矩陣，總數必須大於零。
        x_edges_m: x 軸公尺制格線，至少兩點、有限、嚴格遞增且等距。
        y_edges_m: y 軸公尺制格線，至少兩點、有限、嚴格遞增且等距。
        bandwidth_m: 公尺制高斯頻寬，必須有限且大於零。
        hdr_levels: 唯一、有限且嚴格介於 0 與 1 的 HDR 累積質量門檻。

    Returns:
        含防禦性複製的格線與原始計數、平滑密度、每格機率、HDR 遮罩及原始點總數的
        ``BinnedKDEGrid``。

    Raises:
        ValueError: 輸入維度、dtype、數值、格線、頻寬或 HDR 門檻不符合契約時。
        RuntimeError: 平滑後無法得到有限且正的格網總質量時。
    """

    # 格線先複製成固定的浮點陣列；若輸入無法轉為數值，直接轉成契約一致的 ValueError，
    # 避免後續 shape 或差分運算暴露不一致的 TypeError。複製也避免 caller 改動格線後，
    # 回傳產品的座標定義被悄悄改寫。
    try:
        x_edges = np.array(x_edges_m, dtype=np.float64, copy=True)
        y_edges = np.array(y_edges_m, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError("x_edges_m 與 y_edges_m 必須可轉為數值格線") from error
    if x_edges.ndim != 1 or x_edges.size < 2 or not np.all(np.isfinite(x_edges)):
        raise ValueError("x_edges_m 必須是至少兩點的有限一維陣列")
    if y_edges.ndim != 1 or y_edges.size < 2 or not np.all(np.isfinite(y_edges)):
        raise ValueError("y_edges_m 必須是至少兩點的有限一維陣列")
    x_widths = np.diff(x_edges)
    y_widths = np.diff(y_edges)
    if (
        not np.all(np.isfinite(x_widths))
        or not np.all(np.isfinite(y_widths))
        or np.any(x_widths <= 0)
        or np.any(y_widths <= 0)
    ):
        raise ValueError("x_edges_m 與 y_edges_m 必須嚴格遞增")
    if not np.allclose(x_widths, x_widths[0], rtol=1.0e-12, atol=1.0e-12):
        raise ValueError("x_edges_m 必須具有均勻格距")
    if not np.allclose(y_widths, y_widths[0], rtol=1.0e-12, atol=1.0e-12):
        raise ValueError("y_edges_m 必須具有均勻格距")

    expected_shape = (y_edges.size - 1, x_edges.size - 1)
    try:
        counts_input = np.asarray(raw_count)
    except (TypeError, ValueError) as error:
        raise ValueError("raw_count 必須是二維整數矩陣") from error
    if counts_input.ndim != 2 or counts_input.shape != expected_shape:
        raise ValueError(f"raw_count 必須是形狀完全相符的二維矩陣 {expected_shape}")
    if counts_input.dtype.kind not in "iu":
        raise ValueError("raw_count 必須是整數 dtype，不接受浮點數或布林值")
    if counts_input.dtype.kind == "i" and np.any(counts_input < 0):
        raise ValueError("raw_count 不可包含負值")
    if counts_input.dtype.kind == "u" and np.any(counts_input > np.iinfo(np.int64).max):
        raise ValueError("raw_count 的值必須可安全表示為 int64")
    # 這個副本同時確保回傳型別固定為 int64，且 caller 修改原始矩陣不會污染結果。
    counts = np.array(counts_input, dtype=np.int64, copy=True)
    # 以 Python 整數逐格累加，避免大量非負 int64 計數相加時在 uint64 累加器中繞回；
    # raw_point_count 是原始交點總數，應忠實保存而不能因固定寬度整數溢位變小。
    raw_point_count = sum(int(value) for value in counts.flat)
    if raw_point_count <= 0:
        raise ValueError("raw_count 的總數必須大於零")

    # 頻寬是單一公尺制標量；拒絕陣列、布林與字串，避免把形狀或非物理值靜默轉成頻寬。
    try:
        bandwidth_value = np.asarray(bandwidth_m)
        if bandwidth_value.ndim != 0 or bandwidth_value.dtype.kind not in "iuf":
            raise ValueError
        bandwidth = float(bandwidth_value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("bandwidth_m 必須是有限正值") from error
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("bandwidth_m 必須是有限正值")

    # 先把門檻 materialize 成 tuple，讓 generator 也能被檢查，並逐項限制為數值標量；
    # 唯一性是必要條件，否則同一個 dictionary key 會覆蓋前一份 HDR 結果。
    try:
        level_values = tuple(hdr_levels)
        level_numbers: list[float] = []
        for level in level_values:
            level_value = np.asarray(level)
            if level_value.ndim != 0 or level_value.dtype.kind not in "iuf":
                raise ValueError
            level_numbers.append(float(level_value))
        levels = tuple(level_numbers)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("HDR levels 必須是唯一、有限且介於 0 與 1 的門檻") from error
    if (
        not levels
        or len(set(levels)) != len(levels)
        or any(not np.isfinite(level) or level <= 0.0 or level >= 1.0 for level in levels)
    ):
        raise ValueError("HDR levels 必須唯一、有限且嚴格介於 0 與 1")

    # sigma 的單位是 cell 數，因此 x/y 必須各自用自己的物理格距換算，避免矩形格網
    # 把同一個 scalar sigma 錯當成兩軸相同的公尺尺度。
    smoothed = np.asarray(
        gaussian_filter(
            counts.astype(np.float64, copy=False),
            sigma=(bandwidth / float(y_widths[0]), bandwidth / float(x_widths[0])),
            mode="constant",
            cval=0.0,
        ),
        dtype=np.float64,
    )
    if smoothed.shape != expected_shape or not np.all(np.isfinite(smoothed)) or np.any(smoothed < 0.0):
        raise RuntimeError("Gaussian 平滑結果必須與輸入同形，且全部有限非負")
    smoothed_total = float(np.sum(smoothed, dtype=np.float64))
    if not np.isfinite(smoothed_total) or smoothed_total <= 0.0:
        raise RuntimeError("Gaussian 平滑後的格網總質量無效")

    # 零值邊界造成的流失在此只對格網內保留質量重新正規化；cell area 使用實際邊界差，
    # 因此 density_per_m2 與 probability 的關係不依賴格網是否為正方形。另檢查面積是否
    # 可由有限浮點數表示，避免極端座標範圍產生無限密度。
    cell_area = y_widths[:, None] * x_widths[None, :]
    if not np.all(np.isfinite(cell_area)) or np.any(cell_area <= 0.0):
        raise ValueError("x_edges_m 與 y_edges_m 的 cell 面積必須是有限正值")
    cell_probability = smoothed / smoothed_total
    cell_probability /= float(np.sum(cell_probability, dtype=np.float64))
    probability_total = float(np.sum(cell_probability, dtype=np.float64))
    if (
        not np.all(np.isfinite(cell_probability))
        or np.any(cell_probability < 0.0)
        or not np.isclose(probability_total, 1.0, rtol=0.0, atol=1.0e-12)
    ):
        raise RuntimeError("cell_probability 必須有限、非負且總和為 1")
    density_per_m2 = np.asarray(cell_probability / cell_area, dtype=np.float64)
    if not np.all(np.isfinite(density_per_m2)) or np.any(density_per_m2 < 0.0):
        raise RuntimeError("density_per_m2 必須有限且非負")

    # flatten 使用 C-order row-major；對負機率排序並指定 stable，可在相同機率時保留
    # row-major 先後，且所有門檻共用同一順序，自然形成巢狀 HDR masks。
    flat_probability = cell_probability.ravel()
    descending_order = np.argsort(-flat_probability, kind="stable")
    cumulative_probability = np.cumsum(flat_probability[descending_order], dtype=np.float64)
    hdr_masks: dict[float, np.ndarray] = {}
    for level in levels:
        selected_count = int(np.searchsorted(cumulative_probability, level, side="left")) + 1
        selected_count = min(selected_count, descending_order.size)
        mask_flat = np.zeros(flat_probability.size, dtype=bool)
        mask_flat[descending_order[:selected_count]] = True
        hdr_masks[level] = mask_flat.reshape(expected_shape)

    return BinnedKDEGrid(
        x_edges_m=x_edges,
        y_edges_m=y_edges,
        raw_count=counts,
        density_per_m2=density_per_m2,
        cell_probability=np.asarray(cell_probability, dtype=np.float64),
        hdr_masks=hdr_masks,
        bandwidth_m=bandwidth,
        raw_point_count=raw_point_count,
    )


def conditional_kde_2d(
    points_xy_m: np.ndarray,
    *,
    x_edges_m: np.ndarray,
    y_edges_m: np.ndarray,
    hdr_levels: Sequence[float] = (0.50, 0.75, 0.90),
    bandwidth: str | float = "scott",
) -> KDEGrid:
    """建立二維平滑密度地圖與 50/75/90% 高密度區。

    至少需要三個不在同一直線上的點。樣本太少時，應保留原始交點並標示「無法估計」，
    不能為了畫圖而加入隨機擾動。格網每一格可有不同面積，函式會依真實面積重新校正。
    """

    points = np.asarray(points_xy_m, dtype=np.float64)
    x_edges = np.asarray(x_edges_m, dtype=np.float64)
    y_edges = np.asarray(y_edges_m, dtype=np.float64)
    levels = tuple(float(value) for value in hdr_levels)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3 or not np.all(np.isfinite(points)):
        raise ValueError("KDE points 必須至少三個有限二維點")
    if np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
        raise ValueError("KDE points 不可全部共線")
    if (
        x_edges.ndim != 1
        or y_edges.ndim != 1
        or np.any(np.diff(x_edges) <= 0)
        or np.any(np.diff(y_edges) <= 0)
    ):
        raise ValueError("KDE x/y edges 必須嚴格遞增")
    if any(level <= 0 or level >= 1 for level in levels):
        raise ValueError("HDR levels 必須介於 0 與 1")
    x_centers = (x_edges[:-1] + x_edges[1:]) * 0.5
    y_centers = (y_edges[:-1] + y_edges[1:]) * 0.5
    x_grid, y_grid = np.meshgrid(x_centers, y_centers)
    estimator = gaussian_kde(points.T, bw_method=bandwidth)
    density = estimator(np.vstack((x_grid.ravel(), y_grid.ravel()))).reshape(x_grid.shape)
    cell_area = np.diff(y_edges)[:, None] * np.diff(x_edges)[None, :]
    probability = density * cell_area
    total = float(probability.sum())
    if not np.isfinite(total) or total <= 0:
        raise RuntimeError("KDE 格網總質量無效")
    probability /= total
    density /= total
    flat_order = np.argsort(probability.ravel())[::-1]
    cumulative = np.cumsum(probability.ravel()[flat_order])
    masks: dict[float, np.ndarray] = {}
    for level in levels:
        count = int(np.searchsorted(cumulative, level, side="left")) + 1
        mask = np.zeros(probability.size, dtype=bool)
        mask[flat_order[:count]] = True
        masks[level] = mask.reshape(probability.shape)
    return KDEGrid(
        x_centers_m=x_centers,
        y_centers_m=y_centers,
        density_per_m2=density,
        cell_probability=probability,
        hdr_masks=masks,
        bandwidth_factor=float(estimator.factor),
        raw_point_count=points.shape[0],
    )


def _cell_index(value: float, edges: np.ndarray) -> int | None:
    """依 half-open bins 找 cell；最右邊界歸入最後一格以保存域界終點。"""

    if value < edges[0] or value > edges[-1]:
        return None
    if np.isclose(value, edges[-1], rtol=0.0, atol=1.0e-12):
        return edges.size - 2
    index = int(np.searchsorted(edges, value, side="right") - 1)
    return index if 0 <= index < edges.size - 1 else None


def _segment_cell_weights(
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    *,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> list[tuple[int, int, float]]:
    """把直線段依 grid-edge crossing 精確切成 ``(iy, ix, fraction)``。

    fraction 是原線段參數長度，也就是線性時間內插下的停留時間比例。域外小段不分配；
    因此 caller 可比較輸入與已分配秒數，判斷格網 extent 是否完整涵蓋軌跡。
    """

    x0, y0 = start_xy
    x1, y1 = end_xy
    dx = x1 - x0
    dy = y1 - y0
    if abs(dx) <= np.finfo(np.float64).eps and abs(dy) <= np.finfo(np.float64).eps:
        ix = _cell_index(x0, x_edges)
        iy = _cell_index(y0, y_edges)
        return [] if ix is None or iy is None else [(iy, ix, 1.0)]
    breaks = [0.0, 1.0]
    if abs(dx) > np.finfo(np.float64).eps:
        breaks.extend(float((edge - x0) / dx) for edge in x_edges if 0.0 < (edge - x0) / dx < 1.0)
    if abs(dy) > np.finfo(np.float64).eps:
        breaks.extend(float((edge - y0) / dy) for edge in y_edges if 0.0 < (edge - y0) / dy < 1.0)
    fractions = np.unique(np.asarray(breaks, dtype=np.float64))
    result: list[tuple[int, int, float]] = []
    for start_fraction, end_fraction in zip(fractions[:-1], fractions[1:], strict=True):
        midpoint = 0.5 * (start_fraction + end_fraction)
        ix = _cell_index(x0 + midpoint * dx, x_edges)
        iy = _cell_index(y0 + midpoint * dy, y_edges)
        if ix is not None and iy is not None:
            result.append((iy, ix, float(end_fraction - start_fraction)))
    return result


def pathway_residence_grid(
    results: Iterable[ParticleResult],
    *,
    x_edges_m: np.ndarray,
    y_edges_m: np.ndarray,
) -> PathwayGrid:
    """由 ragged trajectory results 聚合路徑覆蓋與停留時間。

    每一相鄰 observation interval 的 age 必須嚴格增加；這同時驗證 backward 軌跡時間
    座標。軌跡落在格網外的秒數不會被偷偷裁成邊界 cell，而是反映在
    ``input_interval_seconds - allocated_interval_seconds``，供發布 gate 檢查。
    """

    x_edges = np.asarray(x_edges_m, dtype=np.float64)
    y_edges = np.asarray(y_edges_m, dtype=np.float64)
    if (
        x_edges.ndim != 1
        or y_edges.ndim != 1
        or x_edges.size < 2
        or y_edges.size < 2
        or np.any(np.diff(x_edges) <= 0)
        or np.any(np.diff(y_edges) <= 0)
    ):
        raise ValueError("pathway grid edges 必須為至少兩點的嚴格遞增一維陣列")
    shape = (y_edges.size - 1, x_edges.size - 1)
    unique_count = np.zeros(shape, dtype=np.int64)
    residence = np.zeros(shape, dtype=np.float64)
    input_seconds = 0.0
    particle_count = 0
    for result in results:
        if len(result.observations) < 1:
            raise ValueError("每個 particle result 至少需要一個 observation")
        particle_count += 1
        visited: set[tuple[int, int]] = set()
        for first, second in zip(result.observations[:-1], result.observations[1:], strict=True):
            duration = second.age_seconds - first.age_seconds
            if duration <= 0:
                raise ValueError("trajectory age_seconds 必須嚴格遞增")
            input_seconds += duration
            weights = _segment_cell_weights(
                (first.x_m, first.y_m),
                (second.x_m, second.y_m),
                x_edges=x_edges,
                y_edges=y_edges,
            )
            for iy, ix, fraction in weights:
                residence[iy, ix] += duration * fraction
                visited.add((iy, ix))
        for iy, ix in visited:
            unique_count[iy, ix] += 1
    if particle_count == 0:
        raise ValueError("pathway aggregation 不接受空 result 集合")
    return PathwayGrid(
        x_edges_m=x_edges,
        y_edges_m=y_edges,
        unique_particle_count=unique_count,
        residence_time_seconds=residence,
        input_particle_count=particle_count,
        input_interval_seconds=input_seconds,
        allocated_interval_seconds=float(residence.sum()),
    )


def boundary_arclength_histogram(
    events: Iterable[BoundaryEvent],
    *,
    boundary_segment_id: str,
    s_edges_m: np.ndarray,
    valid_member_denominator: int,
    accepted_event_types: Sequence[EventType] = (
        EventType.LOCAL_DOMAIN_FIRST_EXIT,
        EventType.FLOW_DOMAIN_OPEN_EXIT,
    ),
) -> BoundaryArclengthHistogram:
    """沿指定開放邊界量測交點數和每公尺的條件式比例。

    只統計穿越開放水域邊界的事件；碰到海岸沒有「沿開放邊界位置」的意義，不能混進來。
    每筆事件都要有沿邊界的距離 ``boundary_s_m``，超出設定範圍就立刻報錯，避免漏掉交點。
    每公尺比例以有效成員數和該段長度校正，全部加總後等於穿越這條邊界的比例。
    """

    edges = np.asarray(s_edges_m, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("boundary s edges 必須嚴格遞增")
    if valid_member_denominator <= 0:
        raise ValueError("valid member 分母必須為正")
    accepted = set(accepted_event_types)
    values: list[float] = []
    for event in events:
        if event.boundary_segment_id != boundary_segment_id or event.event_type not in accepted:
            continue
        if event.boundary_s_m is None or not np.isfinite(event.boundary_s_m):
            raise ValueError("open-boundary event 缺有限 boundary_s_m")
        if event.boundary_s_m < edges[0] or event.boundary_s_m > edges[-1]:
            raise ValueError("boundary crossing 超出 s_edges_m")
        values.append(event.boundary_s_m)
    counts, _ = np.histogram(np.asarray(values, dtype=np.float64), bins=edges)
    widths = np.diff(edges)
    return BoundaryArclengthHistogram(
        boundary_segment_id=boundary_segment_id,
        s_edges_m=edges,
        raw_count=counts.astype(np.int64, copy=False),
        count_density_per_m=counts / widths,
        conditional_fraction_per_m=counts / (valid_member_denominator * widths),
        valid_member_denominator=valid_member_denominator,
    )


def cross_site_connectivity(
    events: Iterable[BoundaryEvent], *, valid_member_denominator_by_site: dict[str, int]
) -> list[dict[str, float | int | str]]:
    """計算方向性 foreign-local enter 比例，分母固定為原始站有效 members。

    同一 particle 即使重複 enter 也只計一次，避免停留／往返造成假性高連通。沒有事件的
    方向仍可由 caller 的五站矩陣補零；分母缺失或非正直接拒絕。
    """

    seen: set[tuple[str, str, str]] = set()
    counts: dict[tuple[str, str], int] = {}
    for event in events:
        if event.event_type != EventType.OTHER_SITE_LOCAL_DOMAIN_ENTER:
            continue
        target = event.related_study_site_id
        if target is None:
            raise ValueError("foreign-local event 缺 related_study_site_id")
        key = (event.particle_id, event.study_site_id, target)
        if key in seen:
            continue
        seen.add(key)
        pair = (event.study_site_id, target)
        counts[pair] = counts.get(pair, 0) + 1
    rows: list[dict[str, float | int | str]] = []
    for (source, target), count in sorted(counts.items()):
        denominator = valid_member_denominator_by_site.get(source)
        if denominator is None or denominator <= 0:
            raise ValueError(f"來源站 {source} 缺有效 member 分母")
        rows.append(
            {
                "source_study_site_id": source,
                "target_study_site_id": target,
                "raw_unique_member_count": count,
                "valid_member_denominator": denominator,
                "conditional_crossing_fraction": count / denominator,
            }
        )
    return rows


def outcome_summary(statuses: Iterable[ParticleStatus]) -> list[dict[str, float | int | str]]:
    """輸出每種停止狀態 raw count、共同 denominator 與比例。"""

    values = list(statuses)
    if not values:
        raise ValueError("outcome summary 不接受空集合")
    counts: dict[ParticleStatus, int] = {}
    for status in values:
        counts[status] = counts.get(status, 0) + 1
    denominator = len(values)
    return [
        {
            "status": status.value,
            "raw_count": count,
            "denominator": denominator,
            "fraction": count / denominator,
        }
        for status, count in sorted(counts.items(), key=lambda item: item[0].value)
    ]
