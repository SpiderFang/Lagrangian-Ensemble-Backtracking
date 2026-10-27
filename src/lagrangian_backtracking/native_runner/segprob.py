"""WP-75：邊界／海岸分段機率圖的計算核心（取代圓形來源方位圖）。

為什麼改
--------
圓形來源方位圖（WP-60，``polar_source``）以錨點為圓心把來源點分到 36 個各 10° 的扇區，有三個問題：
大圓容易被誤認為抽樣範圍或流場範圍；流場外框是矩形，同樣 10° 的扇區在矩形邊界上對應的邊界長度不一致，
等於不均勻地加權；等深線讓圖面太雜。本模組改成直接把結果放在地圖上，分成兩張圖：

* **邊界穿越圖**：把流場範圍（矩形）的邊界沿弧長切成固定長度的分段，每段的值＝穿越該段的虛擬質點數
  ÷ 全部虛擬質點數 × 100%；
* **海岸來源圖**：把流場範圍內的底圖海岸線切成固定長度的分段，每段的值＝到達該段的虛擬質點數
  ÷ 全部虛擬質點數 × 100%。

兩張圖共用同一組離散色階，所有分段的值加總分別等於該組的穿越比例與海岸來源比例
（兩者加上觸底比例恆為 100%）。結果是條件式來源足跡（相對來源權重）的幾何表示，不是絕對來源機率或因果歸因。

分段規則
--------
邊界（:func:`boundary_ring_nw_clockwise`、:func:`cut_line`）
    流場範圍是 bbox 沿四邊加密（每邊 64 點）後投影到站專屬的方位等距座標（公尺）的多邊形，邊界總長約 499 km。
    弧長從**西北角**起算、**順時針**（先沿北界向東，再沿東界向南、南界向西、西界向北），每 10 km 一段；
    最後不足 10 km 的一段照留，真實長度寫入統計表。穿越點的弧長以「到邊界折線的最近點」投影求得，
    並與舊版弧長（``boundary_arc_length_position_m``，自西南角逆時針）換算後逐顆核對。
    邊界落在陸地上的部分（例如連江的西界）不可能被穿越，圖上以細灰線表示，不上色。

海岸（:func:`extract_coast_pieces`、:func:`coast_segments`）
    海岸線取自底圖陸地多邊形（與其他圖相同的 ``coastline-taiwan.geojson``）的邊界，只保留流場範圍內的部分，
    每一條連續的海岸線（本島、各島嶼）各自從**最西北**的端點（閉合的島則為最西北的頂點）起，閉合者順時針，
    每 5 km 切一段；海岸線總長不足 5 km 的小島整條算一段；每條線最後不足 5 km 的一段照留，
    真實長度寫入統計表。到達海岸的虛擬質點位置（方法 B 的網格海岸到達點）歸到**最近**的分段，
    歸屬距離（公尺）逐顆記錄並統計。

色階（:func:`bin_index`）
    0 單獨一階（淺灰），其餘依 ``edges`` 分階（右閉）；兩張圖與五個地點共用。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry import GeometryCollection, LinearRing, LineString, MultiLineString, Polygon, box
from shapely.ops import linemerge, substring
from shapely.strtree import STRtree

from lagrangian_backtracking.geometry import DomainProjection, densified_bbox_polygon

BOUNDARY_SEGMENT_M = 10_000.0
COAST_SEGMENT_M = 5_000.0
POINTS_PER_EDGE = 64
_TOL_M = 1e-6

# 海岸分類名稱（統計表與圖例共用）
CLASS_TAIWAN = "臺灣本島"
CLASS_NAMED = "具名島嶼"
CLASS_FUJIAN = "福建沿岸"
CLASS_MATSU = "馬祖各島"
CLASS_OTHER = "其他小島"
CLASS_ORDER = (CLASS_TAIWAN, CLASS_NAMED, CLASS_FUJIAN, CLASS_MATSU, CLASS_OTHER)

# 底圖沒有島名，以重心座標比對已知島嶼（容差 0.03°）；比對不到者歸「其他小島」。
NAMED_ISLANDS: tuple[tuple[str, float, float], ...] = (
    ("龜山島", 121.952, 24.842),
    ("基隆嶼", 121.786, 25.192),
    ("和平島", 121.766, 25.158),
    ("棉花嶼", 122.106, 25.485),
    ("蘭嶼", 121.549, 22.048),
    ("小蘭嶼", 121.610, 21.950),
    ("小琉球", 120.371, 22.340),
)
NAMED_TOL_DEG = 0.03
MATSU_BOX = (119.88, 120.55, 25.93, 26.42)  # lon_lo, lon_hi, lat_lo, lat_hi（同 coastB_tables）
MATSU_MIN_DIST_TO_MAINLAND_M = 5_000.0
MAINLAND_MIN_AREA_DEG2 = 1.0
TAIWAN_MIN_AREA_DEG2 = 3.0


# ── 邊界 ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BoundaryRing:
    """流場範圍邊界（投影公尺），弧長自西北角順時針。"""

    line: LineString  # 閉合折線，起點＝西北角
    length_m: float
    corner_s_m: dict[str, float]  # NW=0、NE、SE、SW 的弧長
    xy: np.ndarray

    def side_of(self, s_m: np.ndarray | float) -> np.ndarray:
        """弧長所在邊：N、E、S、W（角點歸下一邊）。"""

        s = np.asarray(s_m, dtype=np.float64)
        c = self.corner_s_m
        out = np.full(s.shape, "W", dtype="<U1")
        out[s < c["SW"]] = "S"
        out[s < c["SE"]] = "E"
        out[s < c["NE"]] = "N"
        return out


def boundary_ring_nw_clockwise(
    bbox: tuple[float, float, float, float], proj: DomainProjection, *, points_per_edge: int = POINTS_PER_EDGE
) -> BoundaryRing:
    """流場範圍邊界折線：加密 bbox → 投影 → 轉成自西北角起的順時針。"""

    lon_min, lon_max, lat_min, lat_max = bbox
    poly = densified_bbox_polygon(bbox, points_per_edge=points_per_edge)
    ll = np.asarray(poly.exterior.coords)[:-1]  # 自西南角逆時針
    ll = ll[::-1]  # 順時針
    is_nw = (np.abs(ll[:, 0] - lon_min) < 1e-9) & (np.abs(ll[:, 1] - lat_max) < 1e-9)
    k = int(np.flatnonzero(is_nw)[0])
    ll = np.roll(ll, -k, axis=0)
    x, y = proj.project(ll[:, 0], ll[:, 1])
    xy = np.column_stack([x, y])
    xy = np.vstack([xy, xy[:1]])
    line = LineString(xy)
    seg_len = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])

    def arc_at(lon: float, lat: float) -> float:
        hit = np.flatnonzero((np.abs(ll[:, 0] - lon) < 1e-9) & (np.abs(ll[:, 1] - lat) < 1e-9))
        return float(cum[int(hit[0])])

    corners = {
        "NW": 0.0,
        "NE": arc_at(lon_max, lat_max),
        "SE": arc_at(lon_max, lat_min),
        "SW": arc_at(lon_min, lat_min),
    }
    return BoundaryRing(line=line, length_m=float(line.length), corner_s_m=corners, xy=xy)


def old_arc_to_new(s_old_m: np.ndarray, ring: BoundaryRing) -> np.ndarray:
    """舊弧長（自西南角逆時針）換算成新弧長（自西北角順時針）：``(L − W − s_old) mod L``。

    W 是西界長度；西南角→西北角之前依序是南界、東界、北界，所以西北角的舊弧長是 ``L − W``。
    """

    length = ring.length_m
    west = length - ring.corner_s_m["SW"]
    return np.mod(length - west - np.asarray(s_old_m, dtype=np.float64), length)


def arc_positions(ring: BoundaryRing, x_m: np.ndarray, y_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """點到邊界折線的最近點的弧長，及點到折線的距離（公尺）。"""

    pts = shapely.points(np.asarray(x_m, dtype=np.float64), np.asarray(y_m, dtype=np.float64))
    s = shapely.line_locate_point(ring.line, pts)
    d = shapely.distance(ring.line, pts)
    return np.asarray(s, dtype=np.float64), np.asarray(d, dtype=np.float64)


def cut_edges(length_m: float, step_m: float) -> np.ndarray:
    """沿長度 ``length_m`` 每 ``step_m`` 一刀的分界弧長，含 0 與終點；最後不足一段者照留。"""

    n = int(math.floor(length_m / step_m + _TOL_M))
    edges = [i * step_m for i in range(n + 1)]
    if length_m - edges[-1] > _TOL_M:
        edges.append(length_m)
    else:
        edges[-1] = length_m
    if len(edges) < 2:  # 長度小於一段：整條算一段
        edges = [0.0, length_m]
    return np.asarray(edges, dtype=np.float64)


def cut_line(line: LineString, step_m: float) -> list[LineString]:
    edges = cut_edges(float(line.length), step_m)
    return [substring(line, float(a), float(b)) for a, b in zip(edges[:-1], edges[1:], strict=True)]


@dataclass(frozen=True)
class Segment:
    seg_id: str
    kind: str  # boundary | coast
    geom: LineString  # 投影公尺
    length_m: float
    group: str  # 邊界：N/E/S/W（以分段中點判定）；海岸：分類名稱
    piece: str  # 邊界：邊界；海岸：連續海岸線代號
    note: str = ""  # 海岸：島名（具名島嶼）


def boundary_segments(
    ring: BoundaryRing, land_proj: object | None, step_m: float = BOUNDARY_SEGMENT_M
) -> list[Segment]:
    """邊界分段；``land_proj`` 是投影後的陸地（可空），只用來標記落在陸地上的長度。"""

    out: list[Segment] = []
    parts = cut_line(ring.line, step_m)
    edges = cut_edges(ring.length_m, step_m)
    for i, g in enumerate(parts):
        mid = 0.5 * (edges[i] + edges[i + 1])
        side = str(ring.side_of(np.array([mid]))[0])
        out.append(Segment(f"B{i + 1:03d}", "boundary", g, float(g.length), side, "邊界"))
    return out


def land_length_m(geom: LineString, land_proj: object | None) -> float:
    if land_proj is None or land_proj.is_empty:
        return 0.0
    return float(geom.intersection(land_proj).length)


# ── 海岸 ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CoastPiece:
    feature: int  # 底圖 feature 編號
    part: int
    cls: str
    name: str
    line: LineString  # 投影公尺，已定向（自最西北端起；閉合者順時針）
    closed: bool


def _lines_of(geom: object) -> list[LineString]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom] if geom.length > 0 else []
    if isinstance(geom, (MultiLineString, GeometryCollection)):
        out: list[LineString] = []
        for g in geom.geoms:
            out.extend(_lines_of(g))
        return out
    return []


def _polys_of(geom: object) -> list[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if hasattr(geom, "geoms"):
        out: list[Polygon] = []
        for g in geom.geoms:
            out.extend(_polys_of(g))
        return out
    return []


def orient_piece(line: LineString) -> tuple[LineString, bool]:
    """定向：開放線從最西北端（y−x 較大者）起；閉合線從最西北頂點起並改為順時針。"""

    c = np.asarray(line.coords)
    closed = bool(np.allclose(c[0], c[-1]))
    if closed:
        ring = c[:-1]
        k = int(np.argmax(ring[:, 1] - ring[:, 0]))
        ring = np.roll(ring, -k, axis=0)
        if LinearRing(ring).is_ccw:
            ring = np.vstack([ring[:1], ring[1:][::-1]])
        return LineString(np.vstack([ring, ring[:1]])), True
    if (c[0, 1] - c[0, 0]) < (c[-1, 1] - c[-1, 0]):
        c = c[::-1]
    return LineString(c), False


def _named_island(cx: float, cy: float) -> str | None:
    for name, lon, lat in NAMED_ISLANDS:
        if abs(cx - lon) <= NAMED_TOL_DEG and abs(cy - lat) <= NAMED_TOL_DEG:
            return name
    return None


def extract_coast_pieces(
    land_lonlat: Sequence[object],
    bbox: tuple[float, float, float, float],
    proj: DomainProjection,
    *,
    margin_deg: float = 0.05,
) -> tuple[list[CoastPiece], object]:
    """流場範圍內的海岸線（投影公尺）與投影後的陸地（供邊界標陸地部分用）。

    先以放大 ``margin_deg`` 的方框裁切陸地多邊形（裁切邊在流場範圍之外），投影後取多邊形邊界，
    再與流場範圍多邊形相交，所以裁切邊不會混進海岸線。陸地多邊形分類見模組常數。
    """

    lon_min, lon_max, lat_min, lat_max = bbox
    clip = box(lon_min - margin_deg, lat_min - margin_deg, lon_max + margin_deg, lat_max + margin_deg)
    domain_ll = densified_bbox_polygon(bbox, points_per_edge=POINTS_PER_EDGE)
    domain = proj.project_geometry(domain_ll)
    parts: list[tuple[int, int, Polygon, Polygon]] = []  # feature, part, 原 lon/lat 多邊形, 投影後裁切多邊形
    for fi, geom in enumerate(land_lonlat):
        if not geom.intersects(domain_ll):
            continue
        for pi, poly in enumerate(_polys_of(geom)):
            if not poly.intersects(domain_ll):
                continue
            for cp in _polys_of(poly.intersection(clip)):
                parts.append((fi, pi, poly, proj.project_geometry(cp)))
    mainland = [
        (fi, p) for fi, pi, poly, p in parts if MAINLAND_MIN_AREA_DEG2 < poly.area <= TAIWAN_MIN_AREA_DEG2
    ]
    mainland_union = shapely.union_all([p for _fi, p in mainland]) if mainland else None
    pieces: list[CoastPiece] = []
    for fi, pi, poly, pp in parts:
        c = poly.centroid
        cx, cy = float(c.x), float(c.y)
        if poly.area > TAIWAN_MIN_AREA_DEG2:
            cls, name = CLASS_TAIWAN, ""
        elif poly.area > MAINLAND_MIN_AREA_DEG2:
            cls, name = CLASS_FUJIAN, ""
        else:
            nm = _named_island(cx, cy)
            in_matsu = MATSU_BOX[0] <= cx <= MATSU_BOX[1] and MATSU_BOX[2] <= cy <= MATSU_BOX[3]
            if nm is not None:
                cls, name = CLASS_NAMED, nm
            elif (
                mainland_union is not None
                and in_matsu
                and pp.distance(mainland_union) >= MATSU_MIN_DIST_TO_MAINLAND_M
            ):
                cls, name = CLASS_MATSU, ""
            elif mainland_union is not None:
                cls, name = CLASS_FUJIAN, "近岸小島"
            else:
                cls, name = CLASS_OTHER, ""
        inside = pp.boundary.intersection(domain)
        lines = _lines_of(inside)
        if not lines:
            continue
        merged = linemerge(lines)
        for ln in _lines_of(merged):
            if ln.length <= 0:
                continue
            oriented, closed = orient_piece(ln)
            pieces.append(CoastPiece(fi, pi, cls, name, oriented, closed))
    land_union = shapely.union_all([pp for _fi, _pi, _poly, pp in parts]) if parts else None
    pieces.sort(key=lambda p: (CLASS_ORDER.index(p.cls), -(p.line.coords[0][1] - p.line.coords[0][0])))
    return pieces, land_union


def coast_segments(pieces: Sequence[CoastPiece], step_m: float = COAST_SEGMENT_M) -> list[Segment]:
    out: list[Segment] = []
    for pn, pc in enumerate(pieces, start=1):
        for k, g in enumerate(cut_line(pc.line, step_m), start=1):
            out.append(
                Segment(f"C{pn:04d}-{k:02d}", "coast", g, float(g.length), pc.cls, f"L{pn:04d}", pc.name)
            )
    return out


def assign_nearest(
    segments: Sequence[Segment], x_m: np.ndarray, y_m: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """每個點歸最近的分段；回傳（分段序號，距離公尺）。"""

    tree = STRtree([s.geom for s in segments])
    pts = shapely.points(np.asarray(x_m, dtype=np.float64), np.asarray(y_m, dtype=np.float64))
    pairs, dist = tree.query_nearest(pts, return_distance=True, all_matches=False)
    idx = np.full(len(pts), -1, dtype=np.int64)
    d = np.full(len(pts), np.nan)
    idx[pairs[0]] = pairs[1]
    d[pairs[0]] = dist
    if (idx < 0).any():
        raise ValueError("有海岸到達點找不到最近分段")
    return idx, d


def distance_stats(d: np.ndarray) -> dict[str, float]:
    d = np.asarray(d, dtype=np.float64)
    if d.size == 0:
        return {
            "n": 0,
            "mean_m": float("nan"),
            "p50_m": float("nan"),
            "p90_m": float("nan"),
            "p99_m": float("nan"),
            "max_m": float("nan"),
            "n_gt_1km": 0,
        }
    return {
        "n": int(d.size),
        "mean_m": float(d.mean()),
        "p50_m": float(np.percentile(d, 50)),
        "p90_m": float(np.percentile(d, 90)),
        "p99_m": float(np.percentile(d, 99)),
        "max_m": float(d.max()),
        "n_gt_1km": int((d > 1000.0).sum()),
    }


# ── 計數與色階 ──────────────────────────────────────────────────────────


def segment_index_boundary(
    ring: BoundaryRing, s_m: np.ndarray, step_m: float = BOUNDARY_SEGMENT_M
) -> np.ndarray:
    edges = cut_edges(ring.length_m, step_m)
    idx = np.searchsorted(edges, np.asarray(s_m, dtype=np.float64), side="right") - 1
    return np.clip(idx, 0, len(edges) - 2)


def counts_by_segment(idx: np.ndarray, n_segments: int, mask: np.ndarray | None = None) -> np.ndarray:
    sel = idx if mask is None else idx[mask]
    return np.bincount(sel, minlength=n_segments).astype(np.int64)


def bin_index(pct: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    """色階序號：0＝恰為 0；其餘 1 + digitize(pct, edges, right=True)（各階右閉，最後一階為 > 最大分界）。"""

    p = np.asarray(pct, dtype=np.float64)
    out = 1 + np.digitize(p, np.asarray(edges, dtype=np.float64), right=True)
    out[p <= 0.0] = 0
    return out


def bin_labels(edges: Sequence[float]) -> list[str]:
    def f(v: float) -> str:
        return f"{v:g}"

    e = list(edges)
    labels = ["0", f"0–{f(e[0])}"]
    for a, b in zip(e[:-1], e[1:], strict=True):
        labels.append(f"{f(a)}–{f(b)}")
    labels.append(f">{f(e[-1])}")
    return labels
