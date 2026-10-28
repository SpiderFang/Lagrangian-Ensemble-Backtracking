# ruff: noqa: E501
"""WP-75：分段機率圖的註記。

* 邊界圖：連續集中段的括號畫在**邊界外側**；單張圖在括號旁寫合計 %，多面板只標小圈號（數值在標題）。
* 海岸圖：單張圖用引線＋百分比；多面板用小圈號 ①②③（標在開放海面）、數值寫進面板標題。
* 所有註記都做碰撞檢查（錨點星號、其他標籤、上色線段、陸地、圖框）。

門檻常數集中在本檔最上方（使用者可調）。所有數字由整數顆數換算：單一數值用 Decimal ROUND_HALF_UP；
一組應加總到總數的小計（四邊 → 穿越比例）用最大餘數法（:func:`largest_remainder`），使顯示值恰好加總。
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_FLOOR, ROUND_HALF_UP, Decimal

import numpy as np
from matplotlib import patheffects
from matplotlib.text import Text
from shapely.geometry import LineString, MultiLineString, Point
from shapely.geometry import box as sbox
from shapely.ops import linemerge, substring, unary_union
from shapely.prepared import prep

from lagrangian_backtracking.native_runner import segprob_figures as SF

# ── 集中段門檻（一處調整）──────────────────────────────────────────────
STRETCH_BOUNDARY_MIN_SEG_PCT = Decimal("1")  # 邊界：相鄰分段各 ≥ 1%（1–2 階以上）才併成一段，不跨越較低的分段
STRETCH_BOUNDARY_MIN_TOTAL_PCT = Decimal("3")  # 邊界：集中段合計 ≥ 3% 才標示；不設張數上限
STRETCH_BOUNDARY_MAX_NUMBERED = 3  # 多面板：只對合計最大的前 3 段編圈號、寫進標題，其餘只畫括號
STRETCH_COAST_MIN_SEG_PCT = Decimal("0.5")  # 海岸：相鄰 5 km 分段各 ≥ 0.5% 才併成一段
STRETCH_COAST_MIN_TOTAL_PCT = Decimal("3")  # 海岸：集中段合計 ≥ 3% 才標示
STRETCH_COAST_MAX_PER_PANEL = 3
SIDE_SEA_MIN_LENGTH_M = 100.0  # 邊的海上長度小於此值視為純陸地邊，不標小計
BRACKET_LEVELS_SINGLE_PT = (5.5, 9.0)  # 單張圖：括號離邊界（pt）的兩個交錯偏移；相鄰集中段輪流使用
BRACKET_LEVELS_PANEL_PT = (2.2, 4.0)  # 多面板（餘地很小）的兩個交錯偏移（pt）
BRACKET_TICK_FRAC = 0.6  # 括號端點短線長度 = 偏移距離 × 此值
BRACKET_CLEAR_PT = 1.0  # 括號之間的最小間距（pt）
BRACKET_LAND_CLEAR_DEG = 0.004  # 括號與陸地（含島嶼）的偏好間距（度，約 0.45 km）
BRACKET_LAND_CLEAR_MIN_DEG = 0.0005  # 做不到偏好間距時的最小間距（度，約 50 m）；仍相交才算與陸地重疊
SIDE_NAME_ZH = {"N": "北", "E": "東", "S": "南", "W": "西"}
SIDE_LABEL_COLOR = "#555555"  # 邊小計（邊界內側）字色，比括號合計稍淺
LAND_CLEARANCE_PT = 2.5  # 標籤與陸地、島嶼海岸線的最小距離（pt）
SIDE_LABEL_INSET_PT = 4.0  # 邊小計離邊界線的距離（pt）
CIRCLED = "①②③④⑤⑥⑦⑧⑨"

_W = {
    "axes": 100,
    "star": 100,
    "label": 100,
    "avoid": 100,
    "colored": 50,
    "land": 30,
    "frame": 40,
    "xleader": 3,
}


@dataclass
class Stretch:
    kind: str  # boundary | coast
    chain: str  # 邊界："邊界"；海岸：連續海岸線代號
    indices: list[int]  # 分段序號（沿鏈順序）
    count: int


# ── 數字：最大餘數法 ─────────────────────────────────────────────────────
def largest_remainder(counts: dict[str, int], total_count: int, n: int, decimals: int) -> dict[str, Decimal]:
    """把一組顆數換成百分比，總和恰為 ``total_count / n`` 的四捨五入值（ROUND_HALF_UP 到 ``decimals`` 位）。

    各項先取無條件捨去，再把缺的單位依餘數大小（同餘數取顆數大者，再依鍵序）逐一補 1 個單位。
    """

    unit = Decimal(1).scaleb(-decimals)
    exact = {k: Decimal(int(c)) * 100 / Decimal(int(n)) / unit for k, c in counts.items()}
    floors = {k: v.to_integral_value(rounding=ROUND_FLOOR) for k, v in exact.items()}
    target = (Decimal(int(total_count)) * 100 / Decimal(int(n)) / unit).quantize(
        Decimal(1), rounding=ROUND_HALF_UP
    )
    need = int(target - sum(floors.values()))
    order = sorted(counts, key=lambda k: (-(exact[k] - floors[k]), -counts[k], list(counts).index(k)))
    out = dict(floors)
    for k in order[: max(need, 0)]:
        out[k] += 1
    return {k: v * unit for k, v in out.items()}


def side_values(counts_by_side: dict[str, int], n: int, decimals: int) -> dict[str, Decimal]:
    """四邊穿越顆數 → 小計百分比（最大餘數法，總和 = 穿越比例四捨五入值）。"""

    return largest_remainder(counts_by_side, sum(counts_by_side.values()), n, decimals)


def side_text(count: int, value: Decimal, *, decimals: int = 2) -> str:
    """邊小計文字：0 顆寫 ``0.00``；有顆數但顯示值為 0 寫 ``<0.01``（僅 2 位小數時）；``decimals=0`` 為整數。"""  # noqa: E501

    if decimals == 2 and count > 0 and value == 0:
        return "<0.01"
    return f"{value:.{decimals}f}"


def side_counts(sd, b_s_masked: np.ndarray) -> dict[str, int]:
    """四邊穿越顆數（逐顆弧長判定所在邊）。"""

    s = b_s_masked[np.isfinite(b_s_masked)]
    sides = sd.ring.side_of(s)
    return {k: int((sides == k).sum()) for k in "NESW"}


def side_sea_geoms(sd) -> dict[str, tuple]:
    """每一邊 ``(整條邊, 海上部分)``（投影公尺）；純陸地邊的海上部分為空幾何。"""

    c = sd.ring.corner_s_m
    bounds = {
        "N": (0.0, c["NE"]),
        "E": (c["NE"], c["SE"]),
        "S": (c["SE"], c["SW"]),
        "W": (c["SW"], sd.ring.length_m),
    }
    out = {}
    for k, (a, b) in bounds.items():
        line = substring(sd.ring.line, a, b)
        sea = line if sd.land_proj is None or sd.land_proj.is_empty else line.difference(sd.land_proj)
        out[k] = (line, sea)
    return out


def side_title_variants(sd, counts_by_side: dict[str, int], n: int) -> list[str]:
    """面板標題用的邊小計字串（整數、最大餘數法），由寬到窄的候選；純陸地邊不列。

    例：``北 25　東 1　南 45　西 4`` → ``北 25 東 1 南 45 西 4`` → ``北25 東1 南45 西4``。
    """

    geoms = side_sea_geoms(sd)
    vals = side_values(counts_by_side, n, 0)
    items = [(k, vals[k]) for k in "NESW" if geoms[k][1].length >= SIDE_SEA_MIN_LENGTH_M]
    wide = "　".join(f"{SIDE_NAME_ZH[k]} {v:.0f}" for k, v in items)
    mid = " ".join(f"{SIDE_NAME_ZH[k]} {v:.0f}" for k, v in items)
    tight = " ".join(f"{SIDE_NAME_ZH[k]}{v:.0f}" for k, v in items)
    return [wide, mid, tight]


# ── 集中段合併 ───────────────────────────────────────────────────────────
def merge_stretches(
    counts: np.ndarray,
    n_total: int,
    chains: list[tuple[str, list[int], bool]],
    *,
    min_seg_pct: Decimal,
    min_total_pct: Decimal,
    max_n: int | None,
    kind: str,
) -> list[Stretch]:
    """相鄰且各 ≥ min_seg_pct 的分段併成一段；
    合計 ≥ min_total_pct 才保留，依合計由大到小，最多 max_n 個。

    ``chains``：``(鏈代號, 分段序號依序, 是否閉合)``；閉合鏈的首尾相鄰（例如邊界繞一圈、閉合的島）。
    比較一律用整數（顆數×100 對 門檻×總數），不經浮點。
    """

    out: list[Stretch] = []
    for name, idx, closed in chains:
        hot = [Decimal(int(counts[i])) * 100 >= min_seg_pct * n_total and counts[i] > 0 for i in idx]
        m = len(idx)
        if m == 0 or not any(hot):
            continue
        runs: list[list[int]] = []
        cur: list[int] = []
        for j in range(m):
            if hot[j]:
                cur.append(j)
            elif cur:
                runs.append(cur)
                cur = []
        if cur:
            runs.append(cur)
        if closed and len(runs) > 1 and hot[0] and hot[-1]:
            runs[0] = runs[-1] + runs[0]
            runs.pop()
        for r in runs:
            ids = [idx[j] for j in r]
            tot = int(sum(int(counts[i]) for i in ids))
            if Decimal(tot) * 100 >= min_total_pct * n_total:
                out.append(Stretch(kind, name, ids, tot))
    out.sort(key=lambda s: -s.count)
    return out[:max_n] if max_n is not None else out


def boundary_chains(sd) -> list[tuple[str, list[int], bool]]:
    return [("邊界", list(range(len(sd.bseg))), True)]


def coast_chains(sd) -> list[tuple[str, list[int], bool]]:
    by: dict[str, list[int]] = {}
    for i, s in enumerate(sd.cseg):
        by.setdefault(s.piece, []).append(i)
    closed = {f"L{n:04d}": p.closed for n, p in enumerate(sd.pieces, start=1)}
    return [(k, v, bool(closed.get(k, False)) and len(v) > 1) for k, v in by.items()]


def coast_stretches(sd, seg_counts: np.ndarray, n_total: int) -> list[Stretch]:
    return merge_stretches(
        seg_counts, n_total, coast_chains(sd),
        min_seg_pct=STRETCH_COAST_MIN_SEG_PCT, min_total_pct=STRETCH_COAST_MIN_TOTAL_PCT,
        max_n=STRETCH_COAST_MAX_PER_PANEL, kind="coast",
    )  # fmt: skip


def boundary_stretches(sd, seg_counts: np.ndarray, n_total: int) -> list[Stretch]:
    return merge_stretches(
        seg_counts, n_total, boundary_chains(sd),
        min_seg_pct=STRETCH_BOUNDARY_MIN_SEG_PCT, min_total_pct=STRETCH_BOUNDARY_MIN_TOTAL_PCT,
        max_n=None, kind="boundary",
    )  # fmt: skip


def circled(k: int) -> str:
    """第 k 個（從 1 起）小圈號。"""

    return CIRCLED[k - 1] if 1 <= k <= len(CIRCLED) else f"({k})"


def marker_title_line(sd, stretches: list[Stretch], n_total: int) -> str:
    """多面板標題用的集中段數值：``① 32.21%　② 4.43%``（與圖上圈號同序）。"""

    return "　".join(f"{circled(i)} {SF.pct2(s.count, n_total)}%" for i, s in enumerate(stretches, start=1))


# ── 場景：顯示座標系下的障礙物與碰撞檢查 ────────────────────────────────
class Scene:
    """某個座標軸上的障礙物（顯示座標，像素）：星號、上色線段、其他標籤、陸地、圖框。"""

    def __init__(
        self,
        fig,
        ax,
        *,
        lines_ll,
        line_lw_pt,
        star_lonlat=None,
        star_size=10.0,
        avoid_boxes=(),
        land_ll=None,
        land_pad_pt=0.0,
        frame_ll=(),
        frame_lw_pt=0.6,
    ):
        fig.canvas.draw()
        self.fig, self.ax = fig, ax
        self.r = fig.canvas.get_renderer()
        self.k = fig.dpi / 72.0
        self.axbb = ax.get_window_extent(self.r)
        self.tr, self.inv = ax.transData, ax.transData.inverted()
        parts = [LineString(self.tr.transform(np.asarray(a))) for a in lines_ll if len(a) >= 2]
        self.colored = (
            unary_union(parts).buffer((line_lw_pt / 2 + 0.8) * self.k, cap_style="flat") if parts else None
        )
        fparts = [LineString(self.tr.transform(np.asarray(a))) for a in frame_ll if len(a) >= 2]
        self.frame = (
            unary_union(fparts).buffer((frame_lw_pt / 2 + 1.0) * self.k) if fparts else None
        )  # 流場範圍細線
        self.star = None
        if star_lonlat is not None:
            sx, sy = self.tr.transform(star_lonlat)
            self.star = Point(sx, sy).buffer((star_size / 2 + 3.0) * self.k)
        av = [sbox(b.x0, b.y0, b.x1, b.y1) for b in avoid_boxes]
        self.avoid = unary_union(av) if av else None
        self.land = prep(land_ll) if land_ll is not None else None
        self.land_pad_pt = land_pad_pt
        self.land_ll = land_ll
        self._land_disp = None
        self.placed: list = []
        self.leaders: list = []
        self.leader_cross_ok = False  # True：引線與引線交叉僅記為輕微（xleader），不算標籤碰撞
        self.log: list[dict] = []

    def land_clearance_pt(self, bb) -> float | None:
        """文字外框到最近陸地（含島嶼）海岸線的距離（pt）；窗內沒有陸地回傳 None。"""

        if self.land_ll is None:
            return None
        if self._land_disp is None:
            import shapely

            ax0, ay0 = self.inv.transform((self.axbb.x0, self.axbb.y0))
            ax1, ay1 = self.inv.transform((self.axbb.x1, self.axbb.y1))
            clip = self.land_ll.intersection(
                sbox(min(ax0, ax1) - 0.2, min(ay0, ay1) - 0.2, max(ax0, ax1) + 0.2, max(ay0, ay1) + 0.2)
            )
            self._land_disp = (
                shapely.transform(clip, lambda c: self.tr.transform(c)) if not clip.is_empty else False
            )
        if self._land_disp is False:
            return None
        return float(sbox(bb.x0, bb.y0, bb.x1, bb.y1).distance(self._land_disp) / self.k)

    def violations(self, bb, anchor=None, leader=True) -> set[str]:
        tags: set[str] = set()
        b = sbox(bb.x0, bb.y0, bb.x1, bb.y1)
        ab = self.axbb
        if bb.x0 < ab.x0 + 1 or bb.x1 > ab.x1 - 1 or bb.y0 < ab.y0 + 1 or bb.y1 > ab.y1 - 1:
            tags.add("axes")
        bp = b.buffer(1.5 * self.k)
        if self.star is not None and bp.intersects(self.star):
            tags.add("star")
        if self.avoid is not None and bp.intersects(self.avoid):
            tags.add("avoid")
        if self.colored is not None and b.intersects(self.colored):
            tags.add("colored")
        if self.frame is not None and b.intersects(self.frame):
            tags.add("frame")
        if any(bp.intersects(g) for g in self.placed) or any(bp.intersects(g) for g in self.leaders):
            tags.add("label")
        if self.land is not None:
            pad = self.land_pad_pt * self.k  # 文字外框與陸地（含島嶼）海岸線至少留此距離
            x0, y0 = self.inv.transform((bb.x0 - pad, bb.y0 - pad))
            x1, y1 = self.inv.transform((bb.x1 + pad, bb.y1 + pad))
            if self.land.intersects(sbox(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))):
                tags.add("land")
        if leader and anchor is not None:
            c = ((bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2)
            ls = LineString([c, anchor]).difference(b)
            if not ls.is_empty:
                # 錨點本身落在星號禁區內（星號就壓在集中段上，如龜山島）時，引線尖端必然碰到，不計
                anchor_in_star = self.star is not None and self.star.contains(Point(anchor))
                if self.star is not None and not anchor_in_star and ls.intersects(self.star):
                    tags.add("star")
                if self.avoid is not None and ls.intersects(self.avoid):
                    tags.add("avoid")
                if any(ls.intersects(g) for g in self.placed):
                    tags.add("label")
                if any(ls.intersects(g) for g in self.leaders):
                    tags.add("xleader" if self.leader_cross_ok else "label")
                if self.colored is not None:
                    far = ls.difference(Point(anchor).buffer(6 * self.k))
                    if not far.is_empty and far.intersects(self.colored):
                        tags.add("colored")
        return tags

    def commit(self, bb, anchor=None):
        self.placed.append(sbox(bb.x0, bb.y0, bb.x1, bb.y1))
        if anchor is not None:
            c = ((bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2)
            ls = LineString([c, anchor]).difference(sbox(bb.x0, bb.y0, bb.x1, bb.y1))
            if not ls.is_empty:
                self.leaders.append(ls.buffer(0.4 * self.k))


def _text_bb(a, renderer):
    """註解文字本體（不含引線）的顯示座標外框；先更新註解的位置轉換。"""

    a.update_positions(renderer)
    return Text.get_window_extent(a, renderer)


def _marker_bb(a, renderer):
    """圈號的墨跡外框（以文字中心為準的 8.4 pt 見方；字型行高會把外框拉得比字形高很多）。"""

    from matplotlib.transforms import Bbox

    bb = _text_bb(a, renderer)
    cx, cy = (bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2
    h = 3.9 * renderer.dpi / 72.0
    return Bbox([[cx - h, cy - h], [cx + h, cy + h]])


def _place(scene: Scene, cands, make, anchor, *, leader=True, bbfun=_text_bb):
    """依序試候選位置；回傳 ``(artist, bbox, 違規標籤集合)``（無違規即停；全部違規則取最輕者）。"""

    best = None
    for c in cands:
        a = make(c)
        bb = bbfun(a, scene.r)
        tags = scene.violations(bb, anchor, leader)
        if not tags:
            return a, bb, set()
        sev = sum(_W[t] for t in tags)
        if best is None or sev < best[0]:
            best = (sev, c, tags)
        a.remove()
    a = make(best[1])
    return a, bbfun(a, scene.r), best[2]


def _lonlat(sd, xs, ys):
    lon, lat = sd.proj.unproject(np.asarray(xs), np.asarray(ys))
    return np.asarray(lon), np.asarray(lat)


def _halo():
    return [patheffects.withStroke(linewidth=1.8, foreground="white")]


def _ha_va(d, thr=0.35):
    ha = "left" if d[0] > thr else ("right" if d[0] < -thr else "center")
    va = "bottom" if d[1] > thr else ("top" if d[1] < -thr else "center")
    return ha, va


def _rot(v, deg):
    th = np.deg2rad(deg)
    return np.array([v[0] * np.cos(th) - v[1] * np.sin(th), v[0] * np.sin(th) + v[1] * np.cos(th)])


# ── 海岸：集中段幾何與標註 ───────────────────────────────────────────────
def _coast_line(sd, s: Stretch):
    geoms = [sd.cseg[i].geom for i in s.indices]
    line = linemerge(geoms) if len(geoms) > 1 else geoms[0]
    if isinstance(line, MultiLineString):
        line = max(line.geoms, key=lambda g: g.length)
    return line


def _sea_dir(sd, line, frac: float = 0.5) -> tuple[Point, np.ndarray]:
    mid = line.interpolate(frac, normalized=True)
    p1 = line.interpolate(max(line.length * frac - 300, 0))
    p2 = line.interpolate(min(line.length * frac + 300, line.length))
    t = np.array([p2.x - p1.x, p2.y - p1.y])
    t = t / (np.hypot(*t) or 1.0)
    n1 = np.array([-t[1], t[0]])
    land = sd.land_proj
    for cand in (n1, -n1):
        probe = Point(mid.x + cand[0] * 2000, mid.y + cand[1] * 2000)
        if land is None or not land.contains(probe):
            return mid, cand
    return mid, n1


def _pick_anchor(sd, line, scene: Scene) -> tuple[Point, np.ndarray]:
    """引線的落點：優先取集中段中點；若太靠近錨點星號（星號禁區內），沿集中段改取離星號夠遠的點。"""

    best = None
    for frac in (0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8, 0.1, 0.9):
        mid, sdir = _sea_dir(sd, line, frac)
        lon, lat = _lonlat(sd, [mid.x], [mid.y])
        if scene.star is None:
            return mid, sdir
        d = Point(*scene.tr.transform((lon[0], lat[0]))).distance(scene.star)
        if d >= 2.0 * scene.k:
            return mid, sdir
        if best is None or d > best[0]:
            best = (d, mid, sdir)
    return best[1], best[2]


def _stretch_record(sd, s: Stretch, n_total: int, extra: dict | None = None) -> dict:
    a0, b0 = sd.cseg[s.indices[0]].geom.coords[0], sd.cseg[s.indices[-1]].geom.coords[-1]
    (alon, alat), (blon, blat) = _lonlat(sd, [a0[0]], [a0[1]]), _lonlat(sd, [b0[0]], [b0[1]])
    rec = {
        "count": s.count, "pct": SF.pct2(s.count, n_total), "segments": [sd.cseg[i].seg_id for i in s.indices],  # noqa: E501
        "start_lonlat": [float(alon[0]), float(alat[0])], "end_lonlat": [float(blon[0]), float(blat[0])],
        "length_km": float(sum(sd.cseg[i].length_m for i in s.indices) / 1000),
    }  # fmt: skip
    if extra:
        rec.update(extra)
    return rec


def stretch_mid_lonlat(sd, s: Stretch) -> tuple[float, float]:
    mid = _coast_line(sd, s).interpolate(0.5, normalized=True)
    lon, lat = _lonlat(sd, [mid.x], [mid.y])
    return float(lon[0]), float(lat[0])


def _coast_scene(ax, fig, dr, seg_counts, *, lw_pt, star_lonlat, star_size, avoid, land_ll):
    lines = [a for i, parts in enumerate(dr.c_lines) if seg_counts[i] > 0 for a in parts]
    frame = [dr.ring_ll] if getattr(dr, "ring_ll", None) is not None else []
    return Scene(
        fig, ax, lines_ll=lines, line_lw_pt=lw_pt, star_lonlat=star_lonlat, star_size=star_size,
        avoid_boxes=avoid, land_ll=land_ll, frame_ll=frame,
    )  # fmt: skip


def annotate_coast_labels(
    ax, fig, sd, dr, seg_counts, stretches, n_total, *, font, star_lonlat, star_size=10.0, lw_pt=SF.LW_COAST,
    avoid=(), prefix="", window=None,
) -> dict:  # fmt: skip
    """單張海岸圖：每個集中段一條小引線＋合計 %（無底框，白色描邊），不寫地名。"""

    rec: dict = {"stretches": [], "collisions": []}
    scene = _coast_scene(ax, fig, dr, seg_counts, lw_pt=lw_pt, star_lonlat=star_lonlat, star_size=star_size,
                         avoid=avoid, land_ll=dr.land_union)  # fmt: skip
    dists = (13, 19, 26, 34, 44, 56, 70)
    angs = (0, 25, -25, 50, -50, 75, -75, 100, -100, 130, -130, 160, -160, 180)
    for s in stretches:
        line = _coast_line(sd, s)
        mid, sdir = _pick_anchor(sd, line, scene)
        lon, lat = _lonlat(sd, [mid.x], [mid.y])
        if window is not None and not (window[0] <= lon[0] <= window[1] and window[2] <= lat[0] <= window[3]):
            continue
        anchor = tuple(scene.tr.transform((lon[0], lat[0])))
        txt = f"{prefix}{SF.pct2(s.count, n_total)}%"
        cands = sorted(((d, a) for d in dists for a in angs), key=lambda c: (c[0], abs(c[1])))

        def make(c, txt=txt, lon=lon, lat=lat, sdir=sdir):
            d = _rot(sdir, c[1])
            ha, va = _ha_va(d)
            return ax.annotate(
                txt, xy=(lon[0], lat[0]), xytext=(d[0] * c[0], d[1] * c[0]), textcoords="offset points", ha=ha,  # noqa: E501
                va=va, fontproperties=font, fontsize=8, zorder=14, path_effects=_halo(),
                arrowprops={"arrowstyle": "-", "lw": 0.6, "color": "#222222", "shrinkA": 0, "shrinkB": 0},
            )  # fmt: skip

        a, bb, tags = _place(scene, cands, make, anchor)
        scene.commit(bb, anchor)
        rec["stretches"].append(_stretch_record(sd, s, n_total, {"label_violations": sorted(tags)}))
        if tags:
            rec["collisions"].append({"label": txt, "violations": sorted(tags)})
    return rec


def annotate_coast_markers(
    ax, fig, sd, dr, seg_counts, stretches, n_total, *, font, star_lonlat, star_size, lw_pt, avoid=(),
) -> dict:  # fmt: skip
    """多面板海岸圖：集中段旁開放海面標小圈號（①②③…，與面板標題數值同序）＋短引線。"""

    rec: dict = {"stretches": [], "collisions": []}
    scene = _coast_scene(ax, fig, dr, seg_counts, lw_pt=lw_pt, star_lonlat=star_lonlat, star_size=star_size,
                         avoid=avoid, land_ll=dr.land_union)  # fmt: skip
    dists = (9, 13, 18, 24, 32, 42, 54)
    angs = (0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180)
    for i, s in enumerate(stretches, start=1):
        line = _coast_line(sd, s)
        mid, sdir = _pick_anchor(sd, line, scene)
        lon, lat = _lonlat(sd, [mid.x], [mid.y])
        anchor = tuple(scene.tr.transform((lon[0], lat[0])))
        cands = sorted(((d, a) for d in dists for a in angs), key=lambda c: (c[0], abs(c[1])))
        glyph = circled(i)

        def make(c, glyph=glyph, lon=lon, lat=lat, sdir=sdir):
            d = _rot(sdir, c[1])
            return ax.annotate(
                glyph, xy=(lon[0], lat[0]), xytext=(d[0] * c[0], d[1] * c[0]), textcoords="offset points",
                ha="center", va="center", fontproperties=font, fontsize=8, zorder=14, path_effects=_halo(),
                arrowprops={"arrowstyle": "-", "lw": 0.5, "color": "#222222", "shrinkA": 0, "shrinkB": 0},
            )  # fmt: skip

        a, bb, tags = _place(scene, cands, make, anchor)
        scene.commit(bb, anchor)
        rec["stretches"].append(
            _stretch_record(sd, s, n_total, {"marker": glyph, "violations": sorted(tags)})
        )
        if tags:
            rec["collisions"].append({"label": glyph, "violations": sorted(tags)})
    return rec


# ── 邊界 ──────────────────────────────────────────────────────────────────
def _offset_outward(sd, line: LineString, dist_m: float) -> LineString | None:
    """把折線向流場範圍外側平移 ``dist_m``（用離中心較遠的一側）。"""

    for sign in (1, -1):
        try:
            off = line.offset_curve(sign * dist_m)
        except Exception:  # noqa: BLE001
            continue
        if off.is_empty:
            continue
        if isinstance(off, MultiLineString):
            off = linemerge(off)
            if isinstance(off, MultiLineString):
                off = max(off.geoms, key=lambda g: g.length)
        mid = off.interpolate(0.5, normalized=True)
        mid0 = line.interpolate(0.5, normalized=True)
        if np.hypot(mid.x, mid.y) > np.hypot(mid0.x, mid0.y):
            return off
    return None


def _boundary_sea_line(sd, s: Stretch):
    geoms = [sd.bseg[i].geom for i in s.indices]
    line = linemerge(geoms) if len(geoms) > 1 else geoms[0]
    if isinstance(line, MultiLineString):
        line = linemerge(list(line.geoms))
        if isinstance(line, MultiLineString):
            line = max(line.geoms, key=lambda g: g.length)
    sea = line if sd.land_proj is None or sd.land_proj.is_empty else line.difference(sd.land_proj)
    if sea.is_empty:
        return None
    if isinstance(sea, MultiLineString):
        sea = linemerge(sea)
        if isinstance(sea, MultiLineString):
            sea = max(sea.geoms, key=lambda g: g.length)
    return sea


def _m_per_pt(sd, ax, scene: Scene) -> float:
    """座標軸上 1 pt 對應的投影公尺數（以視窗中緯度的東西向長度計）。"""

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ym = 0.5 * (y0 + y1)
    px, _ = sd.proj.project(np.array([x0, x1]), np.array([ym, ym]))
    return float(abs(px[1] - px[0]) / (scene.axbb.width / scene.k))


def _land_local(sd, buf_m: float):
    """流場範圍附近的陸地（投影公尺）外擴 ``buf_m``；依距離快取。"""

    cache = sd.__dict__.setdefault("_land_buf_cache", {})
    if buf_m not in cache:
        lp = sd.land_proj
        if lp is None or lp.is_empty:
            cache[buf_m] = None
        else:
            minx, miny, maxx, maxy = sd.ring.line.bounds
            cache[buf_m] = lp.intersection(
                sbox(minx - 30_000, miny - 30_000, maxx + 30_000, maxy + 30_000)
            ).buffer(buf_m)
    return cache[buf_m]


def _bracket_candidate(sd, sea, off_m: float, land_buf_m: float | None = None):
    """外側括號（投影公尺）：平行線＋兩端向邊界的短端線；偏移失敗回傳 None。

    ``land_buf_m`` 給定時，把括號平行線上落在陸地（外擴 ``land_buf_m``）內的部分剪掉（海岸正好在集中段端點外側時），
    端線改接在剪完的兩端。回傳 ``(平行線, 端線列表, 保留比例)``。
    """

    off = _offset_outward(sd, sea, off_m)
    if off is None:
        return None
    keep = 1.0
    if land_buf_m is not None:
        lb = _land_local(sd, land_buf_m)
        if lb is not None:
            cut = off.difference(lb)
            if cut.is_empty:
                return None
            if isinstance(cut, MultiLineString):
                cut = linemerge(cut)
                if isinstance(cut, MultiLineString):
                    cut = max(cut.geoms, key=lambda g: g.length)
            if cut.geom_type != "LineString":
                return None
            keep = float(cut.length / off.length)
            off = cut
    tick = off_m * BRACKET_TICK_FRAC
    ticks = []
    for p_off in (off.coords[0], off.coords[-1]):
        near = sea.interpolate(sea.project(Point(p_off)))
        v = np.array([near.x, near.y]) - np.array(p_off)
        v = v / (np.hypot(*v) or 1.0) * tick
        ticks.append(LineString([(p_off[0], p_off[1]), (p_off[0] + v[0], p_off[1] + v[1])]))
    return off, ticks, keep


def _to_ll(sd, g: LineString) -> LineString:
    xy = np.asarray(g.coords)
    lon, lat = _lonlat(sd, xy[:, 0], xy[:, 1])
    return LineString(np.column_stack([lon, lat]))


def boundary_value_variants(
    stretches: list[Stretch], n_total: int, max_numbered: int | None = None
) -> list[str]:
    """多面板標題最後一行的候選（依序嘗試，取第一個放得下的）：

    ① 兩位小數、一行（``① 42.15%　② 17.78%``）；② 整數、一行（``① 42% ② 18%``）；③ 整數、一行緊縮（``①42% ②18%``）；
    ④ 整數、分兩行。只編前 ``max_numbered`` 個（依 % 由大到小）。
    """

    n_lab = STRETCH_BOUNDARY_MAX_NUMBERED if max_numbered is None else max_numbered
    items = list(stretches)[:n_lab]
    if not items:
        return [""]

    def dec(s: Stretch, k: int) -> str:
        return f"{circled(k)} {SF.pct2(s.count, n_total)}%"

    def integ(s: Stretch, k: int, sep: str) -> str:
        v = (Decimal(s.count) * 100 / Decimal(n_total)).quantize(Decimal(1), rounding=ROUND_HALF_UP)
        return f"{circled(k)}{sep}{v:.0f}%"

    v0 = "　".join(dec(s, k) for k, s in enumerate(items, start=1))
    v1 = " ".join(integ(s, k, " ") for k, s in enumerate(items, start=1))
    v2 = " ".join(integ(s, k, "") for k, s in enumerate(items, start=1))
    half = (len(items) + 1) // 2
    v3 = " ".join(integ(s, k, "") for k, s in enumerate(items[:half], start=1))
    if len(items) > half:
        v3 += "\n" + " ".join(integ(s, k, "") for k, s in enumerate(items[half:], start=half + 1))
    out = [v0, v1, v2, v3]
    return list(dict.fromkeys(out))


def text_width_pt(font, text: str, size: float = 8.0) -> float:
    """以報告字型量一段文字（最寬的一行）的寬度（pt）。"""

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(dpi=200)
    FigureCanvasAgg(fig)
    r = fig.canvas.get_renderer()
    w = 0.0
    for line in text.split("\n"):
        t = fig.text(0, 0, line, fontproperties=font, fontsize=size)
        w = max(w, t.get_window_extent(r).width / (fig.dpi / 72.0))
    return w


def pick_value_variant(variants: list[str], font, avail_pt: float) -> tuple[str, int]:
    """第一個寬度不超過 ``avail_pt`` 的候選；都放不下取最後一個。回傳 ``(文字, 候選序號)``。"""

    for i, v in enumerate(variants):
        if text_width_pt(font, v) <= avail_pt:
            return v, i
    return variants[-1], len(variants) - 1


def annotate_boundary(
    ax, fig, sd, dr, seg_counts, n_total, *, font, star_size=10.0, lw_pt=SF.LW_BOUNDARY, avoid=(),
    mode="label", levels_pt=BRACKET_LEVELS_SINGLE_PT, br_lw=0.8, land_pad_pt=LAND_CLEARANCE_PT,
) -> dict:  # fmt: skip
    """邊界圖：每個集中段（連續分段各 ≥ 門檻、合計 ≥ 門檻）畫外側括號。

    ``mode="label"``（單張圖）：括號旁寫該段合計 %。``mode="marker"``（多面板）：只在合計最大的前
    ``STRETCH_BOUNDARY_MAX_NUMBERED`` 段標小圈號（數值寫在面板標題），其餘只畫括號。
    相鄰集中段的括號交錯兩個偏移；括號不得與其他括號、陸地（含島嶼）、圖框重疊，否則改用其他偏移。
    單張圖的 % 優先放括號外側（放不下才放內側）；多面板的圈號放在同一段邊界線內側的開放海面（括號外側餘地太小），
    放不下才改放外側。
    """

    rec: dict = {"stretches": [], "collisions": [], "mode": mode}
    lines = list(dr.b_sea_flat) if hasattr(dr, "b_sea_flat") else [a for parts in dr.b_sea for a in parts]
    lines += list(dr.b_land)
    scene = Scene(fig, ax, lines_ll=lines, line_lw_pt=lw_pt, star_lonlat=sd.anchor, star_size=star_size,
                  avoid_boxes=avoid, land_ll=dr.land_union, land_pad_pt=land_pad_pt)  # fmt: skip
    scene.leader_cross_ok = True
    stretches = boundary_stretches(sd, seg_counts, n_total)  # 依合計由大到小
    mpp = _m_per_pt(sd, ax, scene)
    clear_m = BRACKET_CLEAR_PT * mpp
    axbb = scene.axbb
    inner = sbox(
        axbb.x0 + 0.5 * scene.k, axbb.y0 + 0.5 * scene.k, axbb.x1 - 0.5 * scene.k, axbb.y1 - 0.5 * scene.k
    )
    ring_order = sorted(range(len(stretches)), key=lambda j: stretches[j].indices[0])
    pref = {j: p % 2 for p, j in enumerate(ring_order)}  # 沿邊界相鄰的集中段交錯偏移
    chosen: dict[int, tuple] = {}
    placed_br: list = []  # 已定括號（投影公尺）
    for j in sorted(range(len(stretches)), key=lambda j: -stretches[j].count):
        s = stretches[j]
        sea = _boundary_sea_line(sd, s)
        if sea is None:
            continue
        p = pref[j]
        lv = [
            levels_pt[p],
            levels_pt[1 - p],
            levels_pt[p] * 0.75,
            levels_pt[1 - p] * 0.75,
            levels_pt[p] * 0.55,
            levels_pt[p] * 0.4,
        ]
        pick = None
        # 先要求與陸地至少 0.004°（約 450 m）；整串都做不到時，退而求其次只要求不與陸地相交（約 50 m）
        for buf_m in (BRACKET_LAND_CLEAR_DEG * 111_320.0, BRACKET_LAND_CLEAR_MIN_DEG * 111_320.0):
            for lv_pt in lv:
                cand = _bracket_candidate(sd, sea, lv_pt * mpp, buf_m)
                if cand is None:
                    continue
                off, ticks, keep = cand
                geo_m = unary_union([off, *ticks])
                geo_ll = unary_union([_to_ll(sd, g) for g in (off, *ticks)])
                hit_land = keep < 0.6 or (
                    dr.land_union is not None
                    and geo_ll.buffer(BRACKET_LAND_CLEAR_MIN_DEG * 0.5).intersects(dr.land_union)
                )
                hit_br = any(geo_m.buffer(clear_m).intersects(g) for g in placed_br)
                disp = unary_union(
                    [LineString(scene.tr.transform(np.asarray(_to_ll(sd, g).coords))) for g in (off, *ticks)]
                )
                hit_frame = not inner.contains(disp)
                pick = (lv_pt, off, ticks, geo_m, disp, hit_land, hit_br, hit_frame)
                if not (hit_land or hit_br or hit_frame):
                    break
            if pick is not None and not (pick[5] or pick[6] or pick[7]):
                break
        if pick is None:
            continue
        placed_br.append(pick[3])
        chosen[j] = (s, sea, *pick)
    numbered = (
        {
            id(stretches[j]): k
            for k, j in enumerate(range(min(len(stretches), STRETCH_BOUNDARY_MAX_NUMBERED)), start=1)
        }
        if mode == "marker"
        else {}
    )
    # 括號畫出；括號本身（顯示座標）先登錄為障礙
    for _s, _sea, _lv, off, ticks, _g, disp, *_rest in chosen.values():
        for g in (off, *ticks):
            ll_ = _to_ll(sd, g)
            ax.plot(
                *np.asarray(ll_.coords).T, color="#222222", linewidth=br_lw, solid_capstyle="butt", zorder=11
            )
        scene.placed.append(disp.buffer(0.8 * scene.k))

    def place_label(j):
        anchor = None
        a = bb = None
        s, sea, lv_pt, off, ticks, geo_m, disp, hit_land, hit_br, hit_frame = chosen[j]
        mid = off.interpolate(0.5, normalized=True)
        mid_in = sea.interpolate(0.5, normalized=True)
        d = np.array([mid.x - mid_in.x, mid.y - mid_in.y])
        d = d / (np.hypot(*d) or 1.0)
        mlon, mlat = _lonlat(sd, [mid.x], [mid.y])
        ilon, ilat = _lonlat(sd, [mid_in.x], [mid_in.y])
        horiz = abs(d[0]) < abs(d[1])
        tang = np.array([1.0, 0.0]) if horiz else np.array([0.0, 1.0])
        marker = circled(numbered[id(s)]) if id(s) in numbered else ""
        txt = f"{SF.pct2(s.count, n_total)}%" if mode == "label" else marker
        tags: set[str] = set()
        if txt:
            if mode == "label":
                shifts = [0, 24, -24, 48, -48, 80, -80, 120, -120]
                cands = [("out", sh, q) for q in (3.0, 8.0) for sh in shifts] + [
                    ("in", sh, q) for q in (4.0, 9.0) for sh in shifts
                ]
            else:
                shifts = [0, 8, -8, 16, -16, 26, -26, 40, -40, 60, -60]
                cands = [("in", sh, q) for q in (6.5, 9.5) for sh in shifts] + [
                    ("out", sh, q) for q in (4.5, 7.0) for sh in shifts[:5]
                ]

            def make(c, txt=txt, d=d, horiz=horiz, tang=tang, mlon=mlon, mlat=mlat, ilon=ilon, ilat=ilat):
                side, sh, q = c
                sgn = 1.0 if side == "out" else -1.0
                lon0, lat0 = (mlon[0], mlat[0]) if side == "out" else (ilon[0], ilat[0])
                off_pt = sgn * d * q + tang * sh
                if mode == "marker":
                    return ax.annotate(
                        txt, xy=(lon0, lat0), xytext=(off_pt[0], off_pt[1]), textcoords="offset points", ha="center", va="center",
                        fontproperties=font, fontsize=8, zorder=13, path_effects=_halo(),
                    )  # fmt: skip
                dd = sgn * d
                return ax.annotate(
                    txt, xy=(lon0, lat0), xytext=(off_pt[0], off_pt[1]), textcoords="offset points",
                    fontproperties=font, fontsize=8, zorder=13, rotation=0 if horiz else 90,
                    ha="center" if horiz else ("left" if dd[0] > 0 else "right"),
                    va=("bottom" if dd[1] > 0 else "top") if horiz else "center", path_effects=_halo(),
                )  # fmt: skip

            bbf = _marker_bb if mode == "marker" else _text_bb
            a, bb, tags = _place(scene, cands, make, None, leader=False, bbfun=bbf)
            if tags and mode == "marker":  # 直接放不下：改放開放海面並以細引線連回該段（同海岸圖的圈號）
                a.remove()
                lcands = sorted(((dist, ang) for dist in (6.5, 8, 9.5, 11, 13, 16, 20, 25, 31, 38, 47, 58) for ang in (0, 15, -15, 30, -30, 45, -45, 60, -60, 75, -75, 90, -90, 120, -120, 150, -150)),
                                key=lambda c: (c[0], abs(c[1])))  # fmt: skip
                best_l = None
                old_pad = scene.land_pad_pt
                for pad_l in (old_pad, 0.3):
                    scene.land_pad_pt = pad_l
                    for frac in (0.5, 0.3, 0.7, 0.12, 0.88):
                        pm = sea.interpolate(frac, normalized=True)
                        po = off.interpolate(frac, normalized=True)
                        dv = np.array([pm.x - po.x, pm.y - po.y])
                        dv = dv / (np.hypot(*dv) or 1.0)
                        flon, flat = _lonlat(sd, [pm.x], [pm.y])
                        anchor = tuple(scene.tr.transform((flon[0], flat[0])))

                        def make_lead(c, txt=txt, dd_in=dv, ilon=flon, ilat=flat):
                            v = _rot(dd_in, c[1])
                            return ax.annotate(
                                txt, xy=(ilon[0], ilat[0]), xytext=(v[0] * c[0], v[1] * c[0]), textcoords="offset points",
                                ha="center", va="center", fontproperties=font, fontsize=8, zorder=13, path_effects=_halo(),
                                arrowprops={"arrowstyle": "-", "lw": 0.5, "color": "#222222", "shrinkA": 0, "shrinkB": 0},
                            )  # fmt: skip

                        a_, bb_, tg_ = _place(scene, lcands, make_lead, anchor, leader=True, bbfun=_marker_bb)
                        sev = sum(_W[t] for t in tg_)
                        if best_l is None or sev < best_l[0]:
                            if best_l is not None:
                                best_l[1].remove()
                            best_l = (sev, a_, bb_, tg_, anchor)
                        else:
                            a_.remove()
                        if not tg_:
                            break
                    if best_l[3] == set():
                        break
                scene.land_pad_pt = old_pad
                _, a, bb, tags, anchor = best_l
                scene.commit(bb, anchor)
            else:
                scene.commit(bb)
        return a, bb, tags, anchor, txt, marker

    # 標籤：先由大到小；若有碰撞且不只一個圈號，試其他放置順序，取總嚴重度最小者
    base_order = sorted(chosen, key=lambda j: -stretches[j].count)
    num_js = [j for j in base_order if id(chosen[j][0]) in numbered]

    def run_order(order):
        n_p, n_l = len(scene.placed), len(scene.leaders)
        res = {j: place_label(j) for j in order}
        sev = sum(_W[t] for r in res.values() for t in r[2])
        return res, sev, n_p, n_l

    def undo(res, n_p, n_l):
        for r in res.values():
            if r[0] is not None:
                r[0].remove()
        del scene.placed[n_p:]
        del scene.leaders[n_l:]

    results, sev0, n_p0, n_l0 = run_order(base_order)
    best_sev, best_order, last_order = sev0, base_order, base_order
    if mode == "marker" and sev0 > 0 and len(num_js) > 1:
        import itertools

        others = [j for j in base_order if j not in num_js]
        for perm in itertools.permutations(num_js):
            if list(perm) == num_js:
                continue
            undo(results, n_p0, n_l0)
            last_order = list(perm) + others
            results, sev, _, _ = run_order(last_order)
            if sev < best_sev:
                best_sev, best_order = sev, last_order
            if sev == 0:
                break
        if last_order != best_order:
            undo(results, n_p0, n_l0)
            results, _, _, _ = run_order(best_order)
    for j in base_order:
        s, sea, lv_pt, off, ticks, geo_m, disp, hit_land, hit_br, hit_frame = chosen[j]
        a, bb, tags, anchor, txt, marker = results[j]
        if "xleader" in tags:
            rec.setdefault("leader_crossings", []).append(marker)
            tags = set(tags) - {"xleader"}
        if hit_land:
            tags = set(tags) | {"bracket_land"}
        if hit_br:
            tags = set(tags) | {"bracket_overlap"}
        if hit_frame:
            tags = set(tags) | {"bracket_axes"}
        a0 = sd.bseg[s.indices[0]].geom.coords[0]
        b0 = sd.bseg[s.indices[-1]].geom.coords[-1]
        (alon, alat), (blon, blat) = _lonlat(sd, [a0[0]], [a0[1]]), _lonlat(sd, [b0[0]], [b0[1]])
        rec["stretches"].append(
            {"count": s.count, "pct": SF.pct2(s.count, n_total), "segments": [sd.bseg[i].seg_id for i in s.indices],  # noqa: E501
             "start_lonlat": [float(alon[0]), float(alat[0])], "end_lonlat": [float(blon[0]), float(blat[0])],
             "bracket_offset_pt": float(lv_pt), "marker": marker, "violations": sorted(tags)}
        )  # fmt: skip
        if tags:
            rec["collisions"].append(
                {"label": txt or f"{SF.pct2(s.count, n_total)}%", "violations": sorted(tags)}
            )
    rec["n_stretches"] = len(stretches)
    rec["n_numbered"] = len(numbered)
    rec["n_unnumbered"] = (len(stretches) - len(numbered)) if mode == "marker" else 0
    return rec
