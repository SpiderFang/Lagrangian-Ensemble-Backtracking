"""flow/local domain 的公尺制投影、幾何生成與步內 crossing 工具。

所有距離、buffer、邊界弧長與粒子步進均在每個 flow domain 固定的 Azimuthal
Equidistant CRS 計算；經緯度只用於上游資料交換與圖面顯示。bbox 邊界在投影前先沿
四邊加密，避免只投影四個角後把經緯線誤當投影座標中的直線。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from pyproj import CRS, Transformer
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import transform


@dataclass(frozen=True, slots=True)
class SegmentCrossing:
    """從步首到步末線段的第一次邊界交點。

    ``fraction`` 是沿原始線段的 0–1 比例，可同步內插事件時間與 z；``boundary_s_m``
    是交點沿 polygon exterior 的公尺弧長，可直接用於一維邊界密度。
    """

    x_m: float
    y_m: float
    fraction: float
    boundary_s_m: float


class DomainProjection:
    """以 domain 中心固定的 WGS84 ↔ local AEQD 雙向轉換器。"""

    def __init__(self, center_lon: float, center_lat: float) -> None:
        """建立 always-xy transformer，確保輸入順序固定為 lon、lat。"""

        if not (-180.0 <= center_lon <= 180.0 and -90.0 < center_lat < 90.0):
            raise ValueError("投影中心經緯度超出 WGS84 範圍")
        local_crs = CRS.from_proj4(
            f"+proj=aeqd +lat_0={center_lat:.12f} +lon_0={center_lon:.12f} +datum=WGS84 +units=m +no_defs"
        )
        self._forward = Transformer.from_crs("EPSG:4326", local_crs, always_xy=True)
        self._inverse = Transformer.from_crs(local_crs, "EPSG:4326", always_xy=True)

    def project(self, lon: np.ndarray | float, lat: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        """將經緯度轉為公尺；array shape 由 pyproj 原樣保留。"""

        x_m, y_m = self._forward.transform(lon, lat)
        return np.asarray(x_m, dtype=np.float64), np.asarray(y_m, dtype=np.float64)

    def unproject(self, x_m: np.ndarray | float, y_m: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        """將公尺座標轉回 WGS84 lon/lat，供 forcing adapter 與輸出使用。"""

        lon, lat = self._inverse.transform(x_m, y_m)
        return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)

    def project_geometry(self, geometry):
        """投影任意 Shapely geometry；空幾何仍保持空。"""

        return transform(self._forward.transform, geometry)

    def unproject_geometry(self, geometry):
        """反投影任意 Shapely geometry；主要用於 manifest 與展示。"""

        return transform(self._inverse.transform, geometry)


def densified_bbox_polygon(
    bbox_lon_lat: tuple[float, float, float, float], *, points_per_edge: int = 64
) -> Polygon:
    """把 WGS84 bbox 四邊加密為 polygon。

    加密只表達原始 lon/lat 常數邊界，並不增加 forcing 支撐；投影後 polygon 才能作
    outer crossing。``points_per_edge`` 至少 2，以避免退化邊界。
    """

    lon_min, lon_max, lat_min, lat_max = bbox_lon_lat
    if not (lon_min < lon_max and lat_min < lat_max):
        raise ValueError("bbox 必須嚴格遞增")
    if points_per_edge < 2:
        raise ValueError("points_per_edge 至少為 2")
    bottom = [(lon, lat_min) for lon in np.linspace(lon_min, lon_max, points_per_edge, endpoint=False)]
    right = [(lon_max, lat) for lat in np.linspace(lat_min, lat_max, points_per_edge, endpoint=False)]
    top = [(lon, lat_max) for lon in np.linspace(lon_max, lon_min, points_per_edge, endpoint=False)]
    left = [(lon_min, lat) for lat in np.linspace(lat_max, lat_min, points_per_edge, endpoint=False)]
    polygon = Polygon([*bottom, *right, *top, *left])
    if not polygon.is_valid:
        raise ValueError("bbox 加密後產生無效 polygon")
    return polygon


def build_anchor_local_domain(
    *,
    projection: DomainProjection,
    anchor_lonlat: tuple[float, float],
    radius_m: float,
    static_ocean_polygon_metric: Polygon,
    buffer_quad_segs: int = 64,
) -> Polygon:
    """建立 anchor-centered metric circle 與固定有效海域的交集。

    local domain 不使用動態 wet/dry 逐時改形，否則 50 個 arrival times 會得到不同研究
    邊界。動態遮罩只在 receptor 與 forcing stage 驗證。若交集為 MultiPolygon，保留
    包含或最接近 anchor 的連通分量，避免離散小島海域成為無關 local component。
    """

    if radius_m <= 0:
        raise ValueError("local-domain radius 必須為正")
    anchor_x, anchor_y = projection.project(*anchor_lonlat)
    anchor = Point(float(anchor_x), float(anchor_y))
    circle = anchor.buffer(radius_m, quad_segs=buffer_quad_segs)
    clipped = circle.intersection(static_ocean_polygon_metric)
    if clipped.is_empty:
        raise ValueError("local domain 與 static ocean polygon 沒有交集")
    if clipped.geom_type == "Polygon":
        return clipped
    polygons = [item for item in getattr(clipped, "geoms", ()) if item.geom_type == "Polygon"]
    if not polygons:
        raise ValueError("local domain 交集不含 polygon")
    containing = [item for item in polygons if item.covers(anchor)]
    return max(containing or polygons, key=lambda item: item.area)


def polygon_crossings(
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    polygon: Polygon,
) -> list[SegmentCrossing]:
    """回傳線段依步進順序穿越 polygon exterior 的全部離散交點。

    函式同時支援 enter、exit 與單步完整穿越 foreign local domain。步首恰位於邊界且
    往域外移動時保留 fraction=0，否則粒子可能在下一步已位於域外而永遠漏記 outer
    stop。線段若只在邊界上滑動會得到 LineString intersection，不視為 crossing。
    """

    start = np.asarray(start_xy, dtype=np.float64)
    end = np.asarray(end_xy, dtype=np.float64)
    delta = end - start
    squared_length = float(np.dot(delta, delta))
    if squared_length <= np.finfo(np.float64).eps:
        return []
    line = LineString([tuple(start), tuple(end)])
    intersection = line.intersection(polygon.boundary)
    if intersection.is_empty:
        return []
    if intersection.geom_type == "Point":
        points = [intersection]
    else:
        points = [item for item in getattr(intersection, "geoms", ()) if item.geom_type == "Point"]
    candidates: list[tuple[float, Point]] = []
    for point in points:
        vector = np.array([point.x, point.y], dtype=np.float64) - start
        fraction = float(np.dot(vector, delta) / squared_length)
        if -1e-12 <= fraction <= 1.0 + 1e-12:
            candidates.append((max(0.0, min(fraction, 1.0)), point))
    if not candidates:
        return []
    result: list[SegmentCrossing] = []
    for fraction, point in sorted(candidates, key=lambda item: item[0]):
        # polygon corner 可能由兩條 exterior segments 重複回報同一交點；依 fraction 去重，
        # 否則 foreign-local 狀態會在同一位置被錯誤切換兩次。
        if result and abs(result[-1].fraction - fraction) <= 1e-12:
            continue
        result.append(
            SegmentCrossing(
                x_m=float(point.x),
                y_m=float(point.y),
                fraction=fraction,
                boundary_s_m=float(polygon.exterior.project(point)),
            )
        )
    return result


def first_polygon_crossing(
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    polygon: Polygon,
) -> SegmentCrossing | None:
    """回傳第一個步內 polygon crossing；無離散交點時回傳 ``None``。"""

    crossings = polygon_crossings(start_xy, end_xy, polygon)
    return crossings[0] if crossings else None


def deterministic_maximin(
    candidate_xy: np.ndarray,
    *,
    count: int,
    anchor_xy: tuple[float, float],
    tie_break_keys: Iterable[tuple[float, ...]] | None = None,
) -> np.ndarray:
    """以 anchor 最近點起始，依序選取 metric-space maximin 候選 index。

    輸入 ``candidate_xy`` 為 ``(candidate,2)`` 公尺座標。每輪最大化「到已選點的最小
    距離」，同距時依 caller 提供的穩定 key，再依原 index 排序；因此 worker 數、NumPy
    iteration 或檔案順序不會改變 receptor。演算法不評估濕乾或海岸距離，這些 gate
    必須先反映在候選集合中。
    """

    points = np.asarray(candidate_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise ValueError("candidate_xy 必須是有限的 (candidate,2) 公尺座標")
    if count < 1 or count > points.shape[0]:
        raise ValueError("count 必須介於 1 與候選數之間")
    keys = (
        list(tie_break_keys) if tie_break_keys is not None else [(float(i),) for i in range(points.shape[0])]
    )
    if len(keys) != points.shape[0]:
        raise ValueError("tie_break_keys 長度必須等於候選數")
    anchor = np.asarray(anchor_xy, dtype=np.float64)
    anchor_distance2 = np.sum((points - anchor) ** 2, axis=1)
    first = min(range(points.shape[0]), key=lambda index: (anchor_distance2[index], keys[index], index))
    selected = [first]
    minimum_distance2 = np.sum((points - points[first]) ** 2, axis=1)
    minimum_distance2[first] = -np.inf
    while len(selected) < count:
        best_value = float(np.max(minimum_distance2))
        tied = np.flatnonzero(np.isclose(minimum_distance2, best_value, rtol=1e-12, atol=1e-9))
        chosen = min((int(index) for index in tied), key=lambda index: (keys[index], index))
        selected.append(chosen)
        distance2 = np.sum((points - points[chosen]) ** 2, axis=1)
        minimum_distance2 = np.minimum(minimum_distance2, distance2)
        minimum_distance2[selected] = -np.inf
    return np.asarray(selected, dtype=np.int64)
