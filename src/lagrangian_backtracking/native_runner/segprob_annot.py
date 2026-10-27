"""WP-75：分段機率圖的註記。

* 邊界圖：各海上邊的小計寫在**邊界內側**（較淺的灰字）；連續集中段的括號與合計 % 畫在**邊界外側**。
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
STRETCH_BOUNDARY_MIN_SEG_PCT = Decimal("2")  # 邊界：相鄰分段各 ≥ 2%（2–5 階以上）才併成一段
STRETCH_BOUNDARY_MIN_TOTAL_PCT = Decimal("10")  # 邊界：集中段合計 ≥ 10% 才標示
STRETCH_BOUNDARY_MAX_PER_PANEL = 2
STRETCH_COAST_MIN_SEG_PCT = Decimal("0.5")  # 海岸：相鄰 5 km 分段各 ≥ 0.5% 才併成一段
STRETCH_COAST_MIN_TOTAL_PCT = Decimal("3")  # 海岸：集中段合計 ≥ 3% 才標示
STRETCH_COAST_MAX_PER_PANEL = 3
SIDE_SEA_MIN_LENGTH_M = 100.0  # 邊的海上長度小於此值視為純陸地邊，不標小計
BRACKET_OFFSETS_M = (4_000.0, 3_000.0, 2_200.0)  # 括號離邊界的距離（公尺）；碰到陸地時依序改用較小者
BRACKET_TICK_FRAC = 0.6  # 括號端點短線長度 = 偏移距離 × 此值
SIDE_NAME_ZH = {"N": "北", "E": "東", "S": "南", "W": "西"}
SIDE_LABEL_COLOR = "#555555"  # 邊小計（邊界內側）字色，比括號合計稍淺
SIDE_LABEL_INSET_PT = 4.0  # 邊小計離邊界線的距離（pt）
CIRCLED = "①②③④⑤⑥⑦⑧⑨"

_W = {"axes": 100, "star": 100, "label": 100, "avoid": 100, "colored": 50, "land": 30}


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
        max_n=STRETCH_BOUNDARY_MAX_PER_PANEL, kind="boundary",
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
        self, fig, ax, *, lines_ll, line_lw_pt, star_lonlat=None, star_size=10.0, avoid_boxes=(), land_ll=None
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
        self.star = None
        if star_lonlat is not None:
            sx, sy = self.tr.transform(star_lonlat)
            self.star = Point(sx, sy).buffer((star_size / 2 + 3.0) * self.k)
        av = [sbox(b.x0, b.y0, b.x1, b.y1) for b in avoid_boxes]
        self.avoid = unary_union(av) if av else None
        self.land = prep(land_ll) if land_ll is not None else None
        self.placed: list = []
        self.log: list[dict] = []

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
        if any(bp.intersects(g) for g in self.placed):
            tags.add("label")
        if self.land is not None:
            x0, y0 = self.inv.transform((bb.x0, bb.y0))
            x1, y1 = self.inv.transform((bb.x1, bb.y1))
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
                self.placed.append(ls.buffer(0.4 * self.k))


def _text_bb(a, renderer):
    """註解文字本體（不含引線）的顯示座標外框；先更新註解的位置轉換。"""

    a.update_positions(renderer)
    return Text.get_window_extent(a, renderer)


def _place(scene: Scene, cands, make, anchor, *, leader=True):
    """依序試候選位置；回傳 ``(artist, bbox, 違規標籤集合)``（無違規即停；全部違規則取最輕者）。"""

    best = None
    for c in cands:
        a = make(c)
        bb = _text_bb(a, scene.r)
        tags = scene.violations(bb, anchor, leader)
        if not tags:
            return a, bb, set()
        sev = sum(_W[t] for t in tags)
        if best is None or sev < best[0]:
            best = (sev, c, tags)
        a.remove()
    a = make(best[1])
    return a, _text_bb(a, scene.r), best[2]


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
    return Scene(
        fig, ax, lines_ll=lines, line_lw_pt=lw_pt, star_lonlat=star_lonlat, star_size=star_size,
        avoid_boxes=avoid, land_ll=land_ll,
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


def _bracket(sd, sea, land_ll):
    """外側括號幾何（投影公尺）；括號或端線碰到陸地時改用較小的偏移。回傳 ``(off, 端線列表, 偏移, 是否碰陸地)``。"""  # noqa: E501

    last = None
    for off_m in BRACKET_OFFSETS_M:
        off = _offset_outward(sd, sea, off_m)
        if off is None:
            continue
        tick = off_m * BRACKET_TICK_FRAC
        ticks = []
        for p_off, p_in in ((off.coords[0], sea.coords[0]), (off.coords[-1], sea.coords[-1])):
            v = np.array(p_in) - np.array(p_off)
            v = v / (np.hypot(*v) or 1.0) * tick
            ticks.append(((p_off[0], p_off[1]), (p_off[0] + v[0], p_off[1] + v[1])))
        ox, oy = np.asarray(off.coords).T
        lon, lat = _lonlat(sd, ox, oy)
        geoms = [LineString(np.column_stack([lon, lat]))]
        for a, b in ticks:
            tl, tt = _lonlat(sd, [a[0], b[0]], [a[1], b[1]])
            geoms.append(LineString(np.column_stack([tl, tt])))
        hit = land_ll is not None and unary_union(geoms).buffer(0.004).intersects(land_ll)
        last = (off, ticks, off_m, bool(hit))
        if not hit:
            return last
    return last


def annotate_boundary(
    ax, fig, sd, dr, seg_counts, side_cnt, n_total, *, font, star_size=10.0, lw_pt=SF.LW_BOUNDARY, avoid=(),
) -> dict:  # fmt: skip
    """邊界單張圖：集中段括號＋合計 %（外側）、各海上邊小計（內側、較淺灰字）；回傳註記紀錄。"""

    rec: dict = {"stretches": [], "side_labels": {}, "collisions": []}
    lines = list(dr.b_sea_flat) if hasattr(dr, "b_sea_flat") else [a for parts in dr.b_sea for a in parts]
    lines += list(dr.b_land)
    scene = Scene(fig, ax, lines_ll=lines, line_lw_pt=lw_pt, star_lonlat=sd.anchor, star_size=star_size,
                  avoid_boxes=avoid, land_ll=dr.land_union)  # fmt: skip
    # 括號與合計（外側）
    for s in boundary_stretches(sd, seg_counts, n_total):
        sea = _boundary_sea_line(sd, s)
        if sea is None:
            continue
        br = _bracket(sd, sea, dr.land_union)
        if br is None:
            continue
        off, ticks, off_m, hit = br
        ox, oy = np.asarray(off.coords).T
        lon, lat = _lonlat(sd, ox, oy)
        (ln,) = ax.plot(lon, lat, color="#222222", linewidth=0.8, solid_capstyle="butt", zorder=11)
        tick_arts = []
        for a, b in ticks:
            tl, tt = _lonlat(sd, [a[0], b[0]], [a[1], b[1]])
            (t_ln,) = ax.plot(tl, tt, color="#222222", linewidth=0.8, solid_capstyle="butt", zorder=11)
            tick_arts.append(t_ln)
        for art in (ln, *tick_arts):
            bb = art.get_window_extent(scene.r)
            scene.placed.append(sbox(bb.x0, bb.y0, bb.x1, bb.y1))
        mid = off.interpolate(0.5, normalized=True)
        mid_in = sea.interpolate(0.5, normalized=True)
        d = np.array([mid.x - mid_in.x, mid.y - mid_in.y])
        d = d / (np.hypot(*d) or 1.0)
        mlon, mlat = _lonlat(sd, [mid.x], [mid.y])
        horiz = abs(d[0]) < abs(d[1])
        txt = f"{SF.pct2(s.count, n_total)}%"
        shifts = [0, 24, -24, 48, -48, 80, -80, 120, -120]
        perps = (3.0, 8.0)
        cands = [(sh, q) for q in perps for sh in shifts]
        tang = np.array([1.0, 0.0]) if horiz else np.array([0.0, 1.0])

        def make(c, txt=txt, mlon=mlon, mlat=mlat, d=d, horiz=horiz, tang=tang):
            sh, q = c
            off_pt = d * q + tang * sh
            return ax.annotate(
                txt, xy=(mlon[0], mlat[0]), xytext=(off_pt[0], off_pt[1]), textcoords="offset points",
                fontproperties=font, fontsize=8, zorder=13, rotation=0 if horiz else 90,
                ha="center" if horiz else ("left" if d[0] > 0 else "right"),
                va=("bottom" if d[1] > 0 else "top") if horiz else "center", path_effects=_halo(),
            )  # fmt: skip

        a, bb, tags = _place(scene, cands, make, None, leader=False)
        scene.commit(bb)
        a0 = sd.bseg[s.indices[0]].geom.coords[0]
        b0 = sd.bseg[s.indices[-1]].geom.coords[-1]
        (alon, alat), (blon, blat) = _lonlat(sd, [a0[0]], [a0[1]]), _lonlat(sd, [b0[0]], [b0[1]])
        if hit:
            tags = set(tags) | {"bracket_land"}
        rec["stretches"].append(
            {"count": s.count, "pct": SF.pct2(s.count, n_total), "segments": [sd.bseg[i].seg_id for i in s.indices],  # noqa: E501
             "start_lonlat": [float(alon[0]), float(alat[0])], "end_lonlat": [float(blon[0]), float(blat[0])],
             "bracket_offset_m": off_m, "violations": sorted(tags)}
        )  # fmt: skip
        if tags:
            rec["collisions"].append({"label": txt, "violations": sorted(tags)})
    # 邊小計（內側）
    geoms = side_sea_geoms(sd)
    vals = side_values(side_cnt, n_total, 2)
    inward = {"N": (0, -1), "S": (0, 1), "E": (-1, 0), "W": (1, 0)}
    for k in "NESW":
        line, sea = geoms[k]
        if sea.length < SIDE_SEA_MIN_LENGTH_M:
            continue
        txt = f"{SIDE_NAME_ZH[k]} {side_text(side_cnt[k], vals[k])}%"
        rec["side_labels"][k] = txt
        parts = list(sea.geoms) if isinstance(sea, MultiLineString) else [sea]
        parts.sort(key=lambda g: -g.length)
        din = np.array(inward[k], float)
        horiz_side = k in "NS"
        cands = []
        for orient in ("parallel", "horizontal"):
            if horiz_side and orient == "horizontal":
                continue
            for g in parts:
                for f in (0.5, 0.3, 0.7, 0.15, 0.85, 0.05, 0.95):
                    pt = g.interpolate(f, normalized=True)
                    cands.append((orient, pt.x, pt.y))
        cx = lambda c: _lonlat(sd, [c[1]], [c[2]])  # noqa: E731

        def make(c, txt=txt, din=din, k=k, horiz_side=horiz_side):
            orient = c[0]
            lon, lat = cx(c)  # noqa: B023
            if horiz_side:
                rot, ha, va = 0, "center", ("top" if din[1] < 0 else "bottom")
            elif orient == "parallel":
                rot, ha, va = 90, ("right" if din[0] < 0 else "left"), "center"
            else:
                rot, ha, va = 0, ("right" if din[0] < 0 else "left"), "center"
            return ax.annotate(
                txt, xy=(lon[0], lat[0]), xytext=(din[0] * SIDE_LABEL_INSET_PT, din[1] * SIDE_LABEL_INSET_PT),
                textcoords="offset points", fontproperties=font, fontsize=8, zorder=13, rotation=rot, ha=ha,
                va=va, color=SIDE_LABEL_COLOR, path_effects=_halo(),
            )  # fmt: skip

        a, bb, tags = _place(scene, cands, make, None, leader=False)
        scene.commit(bb)
        if tags:
            rec["collisions"].append({"label": txt, "violations": sorted(tags)})
        rec.setdefault("side_label_violations", {})[k] = sorted(tags)
    return rec
