"""條件式足跡 KDE/HDR、路徑停留、停止結果與跨站 local-domain 連通聚合。

本模組的比例一律要求 caller 提供有效 member 分母，且輸出 raw count。KDE 是在指定
受體、到達時間、物性與 forcing 條件下的條件式密度，不是 posterior 或絕對來源機率。
HDR 以離散格網 cell probability 排序建立，並回報質量正規化誤差供 QC。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import gaussian_kde

from .engine import ParticleResult
from .models import BoundaryEvent, EventType, ParticleStatus


@dataclass(frozen=True, slots=True)
class KDEGrid:
    """公尺制規則格網上的條件式密度、cell probability 與 HDR masks。"""

    x_centers_m: np.ndarray
    y_centers_m: np.ndarray
    density_per_m2: np.ndarray
    cell_probability: np.ndarray
    hdr_masks: dict[float, np.ndarray]
    bandwidth_factor: float
    raw_point_count: int


@dataclass(frozen=True, slots=True)
class PathwayGrid:
    """軌跡規則格網的 unique-particle count 與 residence time。

    兩個量不可互換：前者每個 particle/cell 最多計一次，後者依相鄰 observation 的實際
    ``age_seconds`` 差分配。單段在公尺制平面假設直線，並以所有穿越的 x/y grid edge
    切段，因此停留秒數守恆，不會因輸出間隔跨過格線而全部落入步末 cell。
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
    """單一命名 open-boundary segment 的弧長 bin 統計與明示分母。"""

    boundary_segment_id: str
    s_edges_m: np.ndarray
    raw_count: np.ndarray
    count_density_per_m: np.ndarray
    conditional_fraction_per_m: np.ndarray
    valid_member_denominator: int


def conditional_kde_2d(
    points_xy_m: np.ndarray,
    *,
    x_edges_m: np.ndarray,
    y_edges_m: np.ndarray,
    hdr_levels: Sequence[float] = (0.50, 0.75, 0.90),
    bandwidth: str | float = "scott",
) -> KDEGrid:
    """建立正規化 2D Gaussian KDE 與 50/75/90% 離散 HDR。

    至少需要三個非共線點；樣本不足應由上游保留 raw points 並標示 KDE 不可估，而非
    人工加 jitter。格網 cell 面積可不等，這裡接受各軸非等距 edges 並逐格正規化。
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
    """以命名線段弧長建立 1D raw count 與條件式密度。

    只接受 open-boundary 事件；coast contact 沒有弧長 KDE 的科學語意。每個事件須有
    ``boundary_s_m``，且超出 edges 直接失敗，避免靜默遺失 crossing。條件式密度以
    有效 member 分母與 bin 長度正規化，積分後等於該 segment 的條件式 crossing 比例。
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
