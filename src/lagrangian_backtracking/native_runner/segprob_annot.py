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
from types import SimpleNamespace

import numpy as np
from matplotlib import patheffects
from matplotlib.text import Text
from shapely.geometry import LineString, MultiLineString, Point, Polygon
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
STRETCH_COAST_MAX_PER_PANEL = None  # 海岸集中段不設張數上限（2026-11-02 PI 定案；v36 以前為 3）
SIDE_SEA_MIN_LENGTH_M = 100.0  # 邊的海上長度小於此值視為純陸地邊，不標小計
BRACKET_LEVELS_SINGLE_PT = (5.5, 9.0)  # 單張圖：括號離邊界（pt）的兩個交錯偏移；相鄰集中段輪流使用
BRACKET_LEVELS_PANEL_PT = (2.2, 4.0)  # 多面板（餘地很小）的兩個交錯偏移（pt）
BRACKET_TICK_FRAC = 0.6  # 括號端點短線長度 = 偏移距離 × 此值
GAP_BRACKET_MIN_TOTAL_PCT = Decimal("1")  # 空檔括號：至少 2 段且合計 ≥ 1%（每個非零片段）才畫
LABEL_CORNER_PT = 4.0  # 括號偏離兩端點連線超過此值（pt）就視為繞角括號
GAP_COLOR = "#8c8c8c"  # 空檔（低機率）括號的線色
GAP_TEXT_COLOR = "#6b6b6b"  # 空檔括號的 % 字色
GAP_LW_FRAC = 0.6  # 空檔括號線寬 = 主括號 × 此值
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


def _segs_joined(sd, i: int, k: int, tol_m: float = 50.0) -> bool:
    """兩個海上分段是否首尾相接（中間沒有陸地或缺口）。"""

    a, b = sd.bseg[i].geom, sd.bseg[k].geom
    pa = [Point(a.coords[0]), Point(a.coords[-1])]
    pb = [Point(b.coords[0]), Point(b.coords[-1])]
    return min(p.distance(q) for p in pa for q in pb) <= tol_m


def _seg_is_sea(sd, i: int, min_sea_m: float = 100.0) -> bool:
    """分段是否有海上部分（整段落在陸地上的分段屬於陸上邊界）。"""

    cache = sd.__dict__.setdefault("_seg_sea_cache", {})
    if i not in cache:
        g, lp = sd.bseg[i].geom, sd.land_proj
        cache[i] = lp is None or lp.is_empty or g.difference(lp).length >= min_sea_m
    return cache[i]


def boundary_gaps(sd, seg_counts: np.ndarray, main: list[Stretch]) -> list[list[int]]:
    """主括號之間、不屬於任何集中段的海上邊界，依連續性切成「空檔」（尚未依 0% 分段切開）。

    陸上邊界（整段落在陸地上的分段，或與鄰段不相接處）會切開空檔，但可繞過轉角。
    回傳每個空檔的分段索引串列。
    """

    m = len(sd.bseg)
    in_main = {i for s in main for i in s.indices}
    sea = [_seg_is_sea(sd, i) for i in range(m)]
    joined = [
        sea[i] and sea[(i + 1) % m] and _segs_joined(sd, i, (i + 1) % m) for i in range(m)
    ]  # i 與 i+1 相接
    free = [i for i in range(m) if i not in in_main and sea[i]]
    starts = [i for i in free if (i - 1) % m in in_main or not joined[(i - 1) % m]]
    if not starts:
        starts = free[:1]
    runs = []
    for st in starts:
        run, i = [st], st
        while joined[i] and (i + 1) % m not in in_main and (i + 1) % m != st:
            i = (i + 1) % m
            run.append(i)
        runs.append(run)
    return runs


def split_at_zero_runs(run: list[int], seg_counts: np.ndarray) -> list[list[int]]:
    """把一個空檔在 0% 分段處切開，兩端與中間的 0% 分段都去掉；回傳剩下的非零連續片段。"""

    pieces: list[list[int]] = []
    cur: list[int] = []
    for i in run:
        if int(seg_counts[i]) > 0:
            cur.append(i)
        elif cur:
            pieces.append(cur)
            cur = []
    if cur:
        pieces.append(cur)
    return pieces


def _neighbour_kind(sd, i: int, step: int, in_main: set[int], zero: set[int] | None = None) -> str:
    m = len(sd.bseg)
    j = (i + step) % m
    a, b = (i, j) if step > 0 else (j, i)
    if not (_seg_is_sea(sd, j) and _segs_joined(sd, a, b)):
        return "陸地或斷點"
    if j in in_main:
        return "主括號"
    return "0%段" if zero and j in zero else "空檔"


def boundary_gap_summary(sd, seg_counts: np.ndarray, n_total: int, main: list[Stretch] | None = None) -> dict:
    """單張邊界圖的括號：主括號、灰色空檔括號與未畫括號的餘量。

    空檔先在 0% 分段處切開（兩端與中間的 0% 都不納入），每個非零片段要同時「至少 2 段」且
    「合計 ≥ ``GAP_BRACKET_MIN_TOTAL_PCT``」才畫灰色括號；不夠的片段不畫、也不在圖上說明，
    只列在表中（``tiny``、``tiny_info``）。

    回傳 ``main``、``gaps``（要畫的片段）、``n_zero_gaps``（全為 0 的空檔數）、``tiny``（未畫括號的片段）、
    ``tiny_info``（位置、段數、原因與兩側鄰居）。單張圖的括號數字一律各自四捨五入（ROUND_HALF_UP），不做最大餘數湊數。
    """

    main = boundary_stretches(sd, seg_counts, n_total) if main is None else main
    runs = boundary_gaps(sd, seg_counts, main)
    mk = lambda r: Stretch("boundary", "邊界", r, int(sum(int(seg_counts[i]) for i in r)))  # noqa: E731
    pieces = [mk(q) for r in runs for q in split_at_zero_runs(r, seg_counts)]
    gaps = [
        g
        for g in pieces
        if len(g.indices) >= 2 and Decimal(g.count) * 100 >= GAP_BRACKET_MIN_TOTAL_PCT * n_total
    ]
    tiny = [t for t in pieces if t not in gaps]
    in_main = {i for s in main for i in s.indices}
    zero = {i for i in range(len(sd.bseg)) if int(seg_counts[i]) == 0}
    info = []
    for t in tiny:
        i0, i1 = t.indices[0], t.indices[-1]
        pt = _boundary_stretch_line(sd, t).interpolate(0.5, normalized=True)
        lon, lat = _lonlat(sd, [pt.x], [pt.y])
        ids = sd.bseg[i0].seg_id if i0 == i1 else f"{sd.bseg[i0].seg_id}–{sd.bseg[i1].seg_id}"
        info.append({
            "seg_id": ids, "n_seg": len(t.indices), "count": t.count, "pct2": SF.pct2(t.count, n_total),
            "reason": "單段" if len(t.indices) < 2 else "不到 1%",
            "lon": float(lon[0]), "lat": float(lat[0]),
            "prev": _neighbour_kind(sd, i0, -1, in_main, zero), "next": _neighbour_kind(sd, i1, 1, in_main, zero),
        })  # fmt: skip
    return {"main": main, "gaps": gaps, "n_zero_gaps": sum(1 for r in runs if not split_at_zero_runs(r, seg_counts)),
            "tiny": tiny, "tiny_info": info}  # fmt: skip


def _label_anchor(off, mpp: float, win_pt: float = 12.0, ang_deg: float = 20.0):
    """標籤的錨點（括號線上）與是否改用直段中點。

    一般取括號線的弧長中點；若括號繞過轉角（任一處前後各 ``win_pt`` 的方向差超過 ``ang_deg``），
    改取最長直段的中點，使標籤貼在括號最長的一邊中間，而不是掛在轉角附近。
    """

    length, w = off.length, win_pt * mpp
    ss = np.linspace(0.0, length, max(int(length / (2.0 * mpp)), 3))

    def straight(s: float) -> bool:
        if s < w or s > length - w:
            return True
        p0, p1, p2 = (off.interpolate(v) for v in (s - w, s, s + w))
        a1 = np.arctan2(p1.y - p0.y, p1.x - p0.x)
        a2 = np.arctan2(p2.y - p1.y, p2.x - p1.x)
        return abs((a2 - a1 + np.pi) % (2 * np.pi) - np.pi) <= np.radians(ang_deg)

    flags = [straight(float(s)) for s in ss]
    if all(flags):
        return off.interpolate(0.5, normalized=True), False
    best, run = (0, 0, 0), None
    for i, f in enumerate([*flags, False]):
        if f and run is None:
            run = i
        elif not f and run is not None:
            if i - run > best[0]:
                best = (i - run, run, i - 1)
            run = None
    _, a, b = best
    return off.interpolate(0.5 * float(ss[a] + ss[b])), True


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


def _place(scene: Scene, cands, make, anchor, *, leader=True, bbfun=_text_bb, ignore=frozenset()):
    """依序試候選位置；回傳 ``(artist, bbox, 違規標籤集合)``（無違規即停；全部違規則取最輕者）。"""

    best = None
    for c in cands:
        a = make(c)
        bb = bbfun(a, scene.r)
        tags = scene.violations(bb, anchor, leader) - set(ignore)
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


def _halo_bracket(lw: float):
    return [patheffects.withStroke(linewidth=lw + 1.6, foreground="white")]


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
def _ring_polygon(sd):
    cache = sd.__dict__
    if "_ring_poly" not in cache:
        cache["_ring_poly"] = Polygon(sd.ring.line.coords)
    return cache["_ring_poly"]


def _offset_side(sd, line: LineString, dist_m: float, outward: bool) -> LineString | None:
    """把折線平移 ``dist_m``：``outward`` 為 True 取流場範圍外側，否則取內側（海面側）。"""

    poly = _ring_polygon(sd)
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
        if off.geom_type != "LineString":
            continue
        if poly.contains(off.interpolate(0.5, normalized=True)) != outward:
            return off
    return None


def _boundary_stretch_line(sd, s: Stretch) -> LineString:
    """集中段的完整邊界線（依分段順序串接，不剪陸地）；括號必須恰好涵蓋這條線。"""

    segs = [list(sd.bseg[i].geom.coords) for i in s.indices]
    if len(segs) > 1:  # 第一段的方向要朝向第二段
        nxt = np.array(segs[1][0])
        if np.hypot(*(np.array(segs[0][0]) - nxt)) < np.hypot(*(np.array(segs[0][-1]) - nxt)):
            segs[0] = segs[0][::-1]
    pts: list = list(segs[0])
    for c in segs[1:]:
        last = np.array(pts[-1])
        if np.hypot(*(np.array(c[0]) - last)) > np.hypot(*(np.array(c[-1]) - last)):
            c = c[::-1]
        if np.hypot(*(np.array(c[0]) - last)) < 1e-6:
            c = c[1:]
        pts.extend(c)
    full = LineString(pts)
    lp = sd.land_proj
    if (
        lp is not None and not lp.is_empty and full.intersects(lp)
    ):  # 分段端點若在陸地上，括號只涵蓋畫得出顏色的海上部分
        sea = full.difference(lp)
        pieces = [g for g in getattr(sea, "geoms", [sea]) if not g.is_empty and g.length > 0]
        if pieces:
            ps = [full.project(Point(c)) for g in pieces for c in (g.coords[0], g.coords[-1])]
            if max(ps) - min(ps) > 1.0:
                full = substring(full, min(ps), max(ps))
    return full


def _span_error_m(full: LineString, off: LineString) -> float:
    """括號平行線兩端在邊界線上的垂足，離集中段兩端點的最大距離（公尺）；兩端須對應到不同端點。"""

    length = full.length
    s0, s1 = full.project(Point(off.coords[0])), full.project(Point(off.coords[-1]))
    if abs(s0 - s1) < 0.99 * length:
        return float(length)
    return float(max(min(s0, length - s0), min(s1, length - s1)))


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


def _bracket_candidate(sd, line, off_m: float, outward: bool):
    """括號（投影公尺）：與集中段邊界線等長的平行線＋兩端向邊界線的短端線；偏移失敗回傳 None。"""

    off = _offset_side(sd, line, off_m, outward)
    if off is None:
        return None
    tick = off_m * BRACKET_TICK_FRAC
    ticks = []
    for p_off in (off.coords[0], off.coords[-1]):
        near = line.interpolate(line.project(Point(p_off)))
        v = np.array([near.x, near.y]) - np.array(p_off)
        v = v / (np.hypot(*v) or 1.0) * tick
        ticks.append(LineString([(p_off[0], p_off[1]), (p_off[0] + v[0], p_off[1] + v[1])]))
    return off, ticks


def _to_ll(sd, g: LineString) -> LineString:
    xy = np.asarray(g.coords)
    lon, lat = _lonlat(sd, xy[:, 0], xy[:, 1])
    return LineString(np.column_stack([lon, lat]))


def boundary_value_variants(
    stretches: list[Stretch], n_total: int, max_numbered: int | None = None, crossing: int | None = None
) -> list[str]:
    """多面板標題最後一行的候選（依序嘗試，取第一個放得下的）。

    只編前 ``max_numbered`` 個（依 % 由大到小）。給定 ``crossing``（該面板穿越顆數）時，行尾加
    「其餘 x.xx%」（= 穿越減去已編號各段），且各數字以最大餘數法湊成穿越比例；沒給時各段各自四捨五入。
    候選：① 兩位小數、一行；② 兩位小數、分兩行；③ 整數、一行；④ 整數、一行緊縮；⑤ 整數、分兩行。
    """

    n_lab = STRETCH_BOUNDARY_MAX_NUMBERED if max_numbered is None else max_numbered
    items = list(stretches)[:n_lab]
    if not items:
        return [""]
    rest = None if crossing is None else int(crossing) - sum(s.count for s in items)

    def values(decimals: int) -> list[str]:
        if rest is None:
            q = Decimal(1).scaleb(-decimals)
            return [
                f"{(Decimal(s.count) * 100 / Decimal(n_total)).quantize(q, rounding=ROUND_HALF_UP):.{decimals}f}"
                for s in items
            ]
        keys = {f"s{k}": s.count for k, s in enumerate(items)} | {"rest": rest}
        lr = largest_remainder(keys, int(crossing), n_total, decimals)
        return [f"{lr[k]:.{decimals}f}" for k in keys]

    def build(decimals: int, sep: str, glue: str, split: bool) -> str:
        vals = values(decimals)
        ents = [f"{circled(k)}{sep}{v}%" for k, v in enumerate(vals[: len(items)], start=1)]
        if rest is not None:
            ents.append(f"其餘{' ' if sep else ''}{vals[-1]}%")
        if not split or len(ents) < 2:
            return glue.join(ents)
        half = (len(ents) + 1) // 2
        return glue.join(ents[:half]) + "\n" + glue.join(ents[half:])

    out = [
        build(2, " ", "　", False),
        build(2, " ", "　", True),
        build(0, " ", " ", False),
        build(0, "", " ", False),
        build(0, "", " ", True),
    ]
    return list(dict.fromkeys(out))


def title_value_table(stretches: list[Stretch], n_total: int, crossing: int) -> list[tuple[str, str, str]]:
    """多面板標題數值：``(編號或「其餘」, 各自四捨五入的 %, 最大餘數法的 %)``（2 位小數）。"""

    items = list(stretches)[:STRETCH_BOUNDARY_MAX_NUMBERED]
    rest = int(crossing) - sum(s.count for s in items)
    keys = {f"s{k}": s.count for k, s in enumerate(items)} | {"rest": rest}
    lr = largest_remainder(keys, int(crossing), n_total, 2)
    names = [circled(k) for k in range(1, len(items) + 1)] + ["其餘"]
    return [(nm, SF.pct2(c, n_total), f"{lr[k]:.2f}") for nm, (k, c) in zip(names, keys.items(), strict=True)]


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
    mode="label", levels_pt=BRACKET_LEVELS_SINGLE_PT, br_lw=0.8, land_pad_pt=LAND_CLEARANCE_PT, gaps=False,
) -> dict:  # fmt: skip
    """邊界圖：每個集中段（連續分段各 ≥ 門檻、合計 ≥ 門檻）畫一條與該段**等長**的括號。

    括號永遠涵蓋整段（不為避讓陸地而截短）：先試邊界外側；外側有陸地或島嶼、與其他括號或圖框衝突時改放內側（海面側）；
    都放不下才讓括號壓過極小島嶼（加白色光暈，記錄於 ``bracket_over_islet``）。

    ``mode="label"``（單張圖）：括號正中央、開放海面一側寫該段合計 %。``mode="marker"``（多面板）：只對合計最大的前
    ``STRETCH_BOUNDARY_MAX_NUMBERED`` 段在括號旁（約 6 pt 內）標小圈號（數值寫在面板標題），其餘只畫括號；
    括號旁放不下才放到最靠近該括號的圖框外緣。
    """

    rec: dict = {"stretches": [], "collisions": [], "mode": mode, "bracket_over_islet": 0}
    lines = list(dr.b_sea_flat) if hasattr(dr, "b_sea_flat") else [a for parts in dr.b_sea for a in parts]
    lines += list(dr.b_land)
    scene = Scene(fig, ax, lines_ll=lines, line_lw_pt=lw_pt, star_lonlat=sd.anchor, star_size=star_size,
                  avoid_boxes=avoid, land_ll=dr.land_union, land_pad_pt=land_pad_pt)  # fmt: skip
    stretches = boundary_stretches(sd, seg_counts, n_total)  # 依合計由大到小
    n_main = len(stretches)
    label_txt = {j: f"{SF.pct2(s.count, n_total)}%" for j, s in enumerate(stretches)}
    if gaps and mode == "label":  # 主括號之間的空檔畫灰色細括號；數字各自四捨五入
        gs = boundary_gap_summary(sd, seg_counts, n_total, stretches)
        gl_draw = gs["gaps"]
        label_txt = {j: f"{SF.pct2(st.count, n_total)}%" for j, st in enumerate(stretches)}
        label_txt.update({n_main + k: f"{SF.pct2(g.count, n_total)}%" for k, g in enumerate(gl_draw)})
        rec["gap_info"] = {
            "main_pct": [label_txt[j][:-1] for j in range(n_main)],
            "main_pct2": [SF.pct2(s.count, n_total) for s in stretches],
            "gap_pct": [label_txt[n_main + k][:-1] for k in range(len(gl_draw))],
            "gap_pct2": [SF.pct2(g.count, n_total) for g in gl_draw],
            "gap_zero_n": gs["n_zero_gaps"],
            "tiny_n": len(gs["tiny"]),
            "tiny_pct": SF.pct2(sum(t.count for t in gs["tiny"]), n_total),
            "tiny_pct2": SF.pct2(sum(t.count for t in gs["tiny"]), n_total),
            "tiny_info": gs["tiny_info"],
            "total_legend": SF.pct2(int(np.sum(seg_counts)), n_total),
        }
        stretches = stretches + gl_draw
    mpp = _m_per_pt(sd, ax, scene)
    clear_m = BRACKET_CLEAR_PT * mpp
    axbb = scene.axbb
    inner = sbox(
        axbb.x0 + 0.5 * scene.k, axbb.y0 + 0.5 * scene.k, axbb.x1 - 0.5 * scene.k, axbb.y1 - 0.5 * scene.k
    )
    ring_order = sorted(range(len(stretches)), key=lambda j: stretches[j].indices[0])
    pref = {j: p % 2 for p, j in enumerate(ring_order)}  # 沿邊界相鄰的集中段交錯偏移
    land_pref_m = BRACKET_LAND_CLEAR_DEG * 111_320.0
    land_min_m = BRACKET_LAND_CLEAR_MIN_DEG * 111_320.0
    lb_pref, lb_min = _land_local(sd, land_pref_m), _land_local(sd, land_min_m)
    chosen: dict[int, dict] = {}
    placed_br: list = []  # 已定括號（投影公尺）
    for j in sorted(range(len(stretches)), key=lambda j: (j >= n_main, -stretches[j].count)):
        s = stretches[j]
        full = _boundary_stretch_line(sd, s)
        tw_pt = text_width_pt(font, label_txt[j], 7.0 if j >= n_main else 8.0) + 1.0
        p = pref[j]
        lv = [
            levels_pt[p],
            levels_pt[1 - p],
            levels_pt[p] * 0.75,
            levels_pt[1 - p] * 0.75,
            max(levels_pt) * 1.4,
        ]
        evals: list[dict] = []
        ends_pts = [Point(full.coords[0]), Point(full.coords[-1])]
        for outward in (True, False):
            for lv_pt in lv:
                cand = _bracket_candidate(sd, full, lv_pt * mpp, outward)
                if cand is None:
                    continue
                off, ticks = cand
                geo_m = unary_union([off, *ticks])
                # 只看平行線；集中段端點本來就貼著海岸，兩端各 6 pt 內的重疊不算
                ends = unary_union([Point(off.coords[0]), Point(off.coords[-1])]).buffer(6.0 * mpp)
                core = off.difference(full.buffer(1.5 * mpp)).difference(ends)
                land_pref = 0.0 if lb_pref is None else core.intersection(lb_pref).length
                land_min = 0.0 if lb_min is None else core.intersection(lb_min).length
                hit_br = False
                for pb in placed_br:
                    shared = [e for e in ends_pts if any(e.distance(e2) < 200.0 for e2 in pb["ends"])]
                    if not shared:
                        hit_br = hit_br or geo_m.buffer(clear_m).intersects(pb["geo_m"])
                        continue
                    # 端點相接的相鄰括號：同側且偏移太接近就會連成一條，否則只比較平行線（端點附近不算）
                    if outward == pb["outward"] and abs(lv_pt - pb["lv"]) < 2.0 * (
                        1.0 if levels_pt[0] > 4 else 0.45
                    ):
                        hit_br = True
                        continue
                    zone = unary_union([e.buffer(8.0 * mpp) for e in shared])
                    hit_br = hit_br or off.difference(zone).buffer(clear_m).intersects(
                        pb["off"].difference(zone)
                    )
                disp = unary_union(
                    [LineString(scene.tr.transform(np.asarray(_to_ll(sd, g).coords))) for g in (off, *ticks)]
                )
                disp_off = LineString(scene.tr.transform(np.asarray(_to_ll(sd, off).coords)))
                hit_frame = not inner.contains(disp)
                hit_col = scene.colored is not None and disp_off.intersects(scene.colored)
                room = True
                if mode == "label":  # 括號外（開放海面側）要放得下置中的 % 文字
                    pm_ = off.interpolate(0.5, normalized=True)
                    ps_ = full.interpolate(0.5, normalized=True)
                    pm_d = np.array(
                        scene.tr.transform(tuple(float(v[0]) for v in _lonlat(sd, [pm_.x], [pm_.y])))
                    )
                    ps_d = np.array(
                        scene.tr.transform(tuple(float(v[0]) for v in _lonlat(sd, [ps_.x], [ps_.y])))
                    )
                    dd_ = pm_d.ravel() - ps_d.ravel()
                    dd_ = dd_ / (np.hypot(*dd_) or 1.0)
                    horiz_ = abs(dd_[0]) < abs(dd_[1])
                    w_, h_ = (tw_pt, 8.0) if horiz_ else (8.0, tw_pt)
                    cx_, cy_ = pm_d.ravel() + dd_ * (2.5 + 4.0) * scene.k
                    rb = SimpleNamespace(
                        x0=cx_ - w_ / 2 * scene.k,
                        x1=cx_ + w_ / 2 * scene.k,
                        y0=cy_ - h_ / 2 * scene.k,
                        y1=cy_ + h_ / 2 * scene.k,
                    )
                    room = not scene.violations(rb, None, False)
                evals.append({
                    "outward": outward, "lv": lv_pt, "off": off, "ticks": ticks, "geo_m": geo_m, "disp": disp,
                    "land_pref": land_pref, "land_min": land_min, "hit_br": hit_br, "hit_frame": hit_frame,
                    "hit_col": hit_col, "room": room,
                })  # fmt: skip
        pick = None
        far_lv = max(levels_pt) * 1.4
        # 先要求括號外放得下置中的 % 文字（僅單張圖），再要求一般偏移；都不行才放寬
        for need_room, far in ((True, False), (True, True), (False, False), (False, True)):
            if need_room and mode != "label":
                continue
            for outward, key in (
                (True, "land_pref"),
                (True, "land_min"),
                (False, "land_pref"),
                (False, "land_min"),
            ):
                for e in evals:
                    if (
                        e["outward"] == outward
                        and (e["lv"] == far_lv) == far
                        and e[key] <= 0
                        and (e["room"] or not need_room)
                        and not (e["hit_br"] or e["hit_frame"] or e["hit_col"])
                    ):
                        pick = e
                        break
                if pick is not None:
                    break
            if pick is not None:
                break
        islet = False
        if pick is None:  # 兩側、各種偏移都被陸地擋住：讓括號壓過最少的島嶼（加白色光暈）
            ok = [e for e in evals if not (e["hit_br"] or e["hit_frame"] or e["hit_col"])]
            if ok:
                pick = min(ok, key=lambda e: (e["land_min"], not e["outward"], e["lv"]))
                islet = pick["land_min"] > 0
        if pick is None and evals:  # 連括號互疊、圖框也避不開：取最輕者並記為違規
            pick = min(evals, key=lambda e: (e["hit_br"] + e["hit_frame"] + e["hit_col"], e["land_min"]))
        if pick is None:
            continue
        pick = dict(
            pick, full=full, islet=islet, gap=j >= n_main, span_err_m=_span_error_m(full, pick["off"])
        )
        placed_br.append(
            {
                "geo_m": pick["geo_m"],
                "off": pick["off"],
                "outward": pick["outward"],
                "lv": pick["lv"],
                "ends": ends_pts,
            }
        )
        chosen[j] = pick
    numbered = (
        {
            id(stretches[j]): k
            for k, j in enumerate(range(min(len(stretches), STRETCH_BOUNDARY_MAX_NUMBERED)), start=1)
        }
        if mode == "marker"
        else {}
    )
    # 括號畫出；括號本身（顯示座標）先登錄為障礙
    for c in chosen.values():
        halo = _halo_bracket(br_lw) if c["islet"] else None
        for g in (c["off"], *c["ticks"]):
            ll_ = _to_ll(sd, g)
            ax.plot(
                *np.asarray(ll_.coords).T, color=GAP_COLOR if c["gap"] else "#222222",
                linewidth=br_lw * GAP_LW_FRAC if c["gap"] else br_lw, solid_capstyle="butt", zorder=11,
                path_effects=halo,
            )  # fmt: skip
        if c["islet"]:
            rec["bracket_over_islet"] += 1
        scene.placed.append(c["disp"].buffer(0.8 * scene.k))

    def place_label(j):
        c = chosen[j]
        s, sea, off = stretches[j], c["full"], c["off"]
        mid, moved = _label_anchor(off, mpp)
        mid_in = sea.interpolate(sea.project(mid)) if moved else sea.interpolate(0.5, normalized=True)
        d = np.array([mid.x - mid_in.x, mid.y - mid_in.y])  # 由邊界線指向括號（開放海面一側）
        d = d / (np.hypot(*d) or 1.0)
        mlon, mlat = _lonlat(sd, [mid.x], [mid.y])
        ilon, ilat = _lonlat(sd, [mid_in.x], [mid_in.y])
        horiz = abs(d[0]) < abs(d[1])
        tang = np.array([1.0, 0.0]) if horiz else np.array([0.0, 1.0])
        marker = circled(numbered[id(s)]) if id(s) in numbered else ""
        txt = label_txt[j] if mode == "label" else marker
        is_gap = j >= n_main
        tags: set[str] = set()
        a = bb = None
        where = ""
        if txt:
            blen_pt = off.length / mpp
            if mode == "label":
                shifts = [0, 6, -6, 12, -12, 20, -20, 30, -30]
                near, far_sh = [sh for sh in shifts if abs(sh) <= 12], [sh for sh in shifts if abs(sh) > 12]
                cands = (  # 先在括號中點附近（沿線 ±12 pt）找，再放寬沿線位移，避免標籤被推離括號中間
                    [("A", sh, q) for q in (2.5, 5.5, 9.0) for sh in near]
                    + [("A", sh, q) for q in (2.5, 5.5, 9.0) for sh in far_sh]
                    + [("B", sh, q) for q in (4.0, 9.0) for sh in shifts]
                )
            else:
                lim = max(4.0, 0.5 * blen_pt)
                shifts = [sh for sh in (0, 4, -4, 8, -8, 12, -12, 14, -14) if abs(sh) <= lim]
                cands = [("A", sh, q) for q in (5.0, 6.2, 7.6, 9.5) for sh in shifts] + [
                    ("B", sh, q) for q in (6.5, 8.0, 10.0, 12.0) for sh in shifts
                ]

            def make(cd, txt=txt, d=d, horiz=horiz, tang=tang):
                if cd[0] == "C":
                    return ax.annotate(
                        txt, xy=(cd[3], cd[4]), xycoords="axes fraction", xytext=(cd[5], cd[6]), textcoords="offset points",
                        ha="center", va="center", fontproperties=font, fontsize=8, zorder=13, path_effects=_halo(),
                    )  # fmt: skip
                side, sh, q = cd
                sgn = 1.0 if side == "A" else -1.0
                lon0, lat0 = (mlon[0], mlat[0]) if side == "A" else (ilon[0], ilat[0])
                off_pt = sgn * d * q + tang * sh
                if mode == "marker":
                    return ax.annotate(
                        txt, xy=(lon0, lat0), xytext=(off_pt[0], off_pt[1]), textcoords="offset points", ha="center", va="center",
                        fontproperties=font, fontsize=8, zorder=13, path_effects=_halo(),
                    )  # fmt: skip
                dd = sgn * d
                return ax.annotate(
                    txt, xy=(lon0, lat0), xytext=(off_pt[0], off_pt[1]), textcoords="offset points",
                    fontproperties=font, fontsize=7 if is_gap else 8, zorder=13, rotation=0 if horiz else 90,
                    color=GAP_TEXT_COLOR if is_gap else "black",
                    ha="center" if horiz else ("left" if dd[0] > 0 else "right"),
                    va=("bottom" if dd[1] > 0 else "top") if horiz else "center", path_effects=_halo(),
                )  # fmt: skip

            bbf = _marker_bb if mode == "marker" else _text_bb
            a, bb, tags = _place(scene, cands, make, None, leader=False, bbfun=bbf)
            where = "beside"
            if tags and mode == "marker":  # 括號旁放不下：改放最靠近該括號的圖框外緣之外
                a.remove()
                mx, my = scene.tr.transform((mlon[0], mlat[0]))
                dists = {"L": mx - axbb.x0, "R": axbb.x1 - mx, "B": my - axbb.y0, "T": axbb.y1 - my}
                edges = sorted(dists, key=dists.get)
                ccands = []
                for e in edges[:3]:
                    fx, fy = (mx - axbb.x0) / axbb.width, (my - axbb.y0) / axbb.height
                    for sh in (0, 4, -4, 8, -8, 14, -14):
                        if e == "L":
                            ccands.append(("C", 0, 0, 0.0, fy, -5.4, sh))
                        elif e == "R":
                            ccands.append(("C", 0, 0, 1.0, fy, 5.4, sh))
                        elif e == "B":
                            ccands.append(("C", 0, 0, fx, 0.0, sh, -5.4))
                        else:
                            ccands.append(("C", 0, 0, fx, 1.0, sh, 5.4))
                a, bb, tags = _place(
                    scene, ccands, make, None, leader=False, bbfun=_marker_bb, ignore={"axes"}
                )
                where = "outside_axes"
            scene.commit(bb)
        return a, bb, tags, txt, marker, where

    # 標籤：先由大到小；若有碰撞且不只一個圈號，試其他放置順序，取總嚴重度最小者
    base_order = sorted(chosen, key=lambda j: (j >= n_main, -stretches[j].count))
    num_js = [j for j in base_order if id(stretches[j]) in numbered]

    def run_order(order):
        n_p = len(scene.placed)
        res = {j: place_label(j) for j in order}
        sev = sum(_W[t] for r in res.values() for t in r[2]) + 0.1 * sum(
            r[5] == "outside_axes" for r in res.values()
        )
        return res, sev, n_p

    def undo(res, n_p):
        for r in res.values():
            if r[0] is not None:
                r[0].remove()
        del scene.placed[n_p:]

    results, sev0, n_p0 = run_order(base_order)
    best_sev, best_order, last_order = sev0, base_order, base_order
    if mode == "marker" and sev0 > 0 and len(num_js) > 1:
        import itertools

        others = [j for j in base_order if j not in num_js]
        for perm in itertools.permutations(num_js):
            if list(perm) == num_js:
                continue
            undo(results, n_p0)
            last_order = list(perm) + others
            results, sev, _ = run_order(last_order)
            if sev < best_sev:
                best_sev, best_order = sev, last_order
            if sev == 0:
                break
        if last_order != best_order:
            undo(results, n_p0)
            results, _, _ = run_order(best_order)
    for j in base_order:
        s, c = stretches[j], chosen[j]
        a, bb, tags, txt, marker, where = results[j]
        tags = set(tags)
        if c["land_min"] > 0:
            tags |= set()  # 壓過極小島嶼者不算碰撞，另行計數（bracket_over_islet）
        if c["hit_br"]:
            tags.add("bracket_overlap")
        if c["hit_frame"]:
            tags.add("bracket_axes")
        if c["hit_col"]:
            tags.add("bracket_colored")
        if c["span_err_m"] > 1.0:
            tags.add("bracket_span")
        a0, b0 = c["full"].coords[0], c["full"].coords[-1]  # 集中段在海上的起訖（即括號涵蓋的範圍）
        (alon, alat), (blon, blat) = _lonlat(sd, [a0[0]], [a0[1]]), _lonlat(sd, [b0[0]], [b0[1]])
        fa = c["full"].interpolate(c["full"].project(Point(c["off"].coords[0])))
        fb = c["full"].interpolate(c["full"].project(Point(c["off"].coords[-1])))
        if c["full"].project(fa) > c["full"].project(fb):
            fa, fb = fb, fa
        (fa_lon, fa_lat), (fb_lon, fb_lat) = _lonlat(sd, [fa.x], [fa.y]), _lonlat(sd, [fb.x], [fb.y])
        seg_len = sum(sd.bseg[i].geom.length for i in s.indices)
        lab_perp = lab_along = None
        lab_corner = False
        if bb is not None:  # 標籤中心與括號的距離（pt）：垂直距離（到括號線）與沿括號方向離中點的弧長
            dline = LineString(scene.tr.transform(np.asarray(_to_ll(sd, c["off"]).coords)))
            pc = Point((bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2)
            chord = LineString([dline.coords[0], dline.coords[-1]])
            lab_corner = max(Point(q).distance(chord) for q in dline.coords) / scene.k > LABEL_CORNER_PT
            lab_perp = dline.distance(pc) / scene.k
            lab_along = abs(dline.project(pc) - dline.length / 2) / scene.k
        rec["stretches"].append(
            {"count": s.count, "pct": SF.pct2(s.count, n_total), "segments": [sd.bseg[i].seg_id for i in s.indices],  # noqa: E501
             "bracket_foot_start_lonlat": [float(fa_lon[0]), float(fa_lat[0])],
             "bracket_foot_end_lonlat": [float(fb_lon[0]), float(fb_lat[0])],
             "sea_clip_m": round(max(0.0, seg_len - c["full"].length), 3),
             "start_lonlat": [float(alon[0]), float(alat[0])], "end_lonlat": [float(blon[0]), float(blat[0])],
             "bracket_offset_pt": float(c["lv"]), "bracket_side": "out" if c["outward"] else "in",
             "bracket_over_islet": bool(c["islet"]), "style": "gap" if c["gap"] else "main",
             "shown_pct": label_txt[j][:-1] if mode == "label" else SF.pct2(s.count, n_total), "bracket_span_err_m": round(c["span_err_m"], 6),
             "marker_where": where, "marker": marker, "violations": sorted(tags),
             "label_perp_pt": None if lab_perp is None else round(lab_perp, 2),
             "label_along_pt": None if lab_along is None else round(lab_along, 2),
             "bracket_has_corner": bool(lab_corner)}
        )  # fmt: skip
        if tags:
            rec["collisions"].append(
                {"label": txt or f"{SF.pct2(s.count, n_total)}%", "violations": sorted(tags)}
            )
    rec["n_stretches"] = n_main
    rec["n_brackets"] = len(chosen)
    rec["n_inner"] = sum(1 for c in chosen.values() if not c["outward"])
    rec["n_numbered"] = len(numbered)
    rec["n_unnumbered"] = (len(stretches) - len(numbered)) if mode == "marker" else 0
    rec["max_span_err_m"] = max((c["span_err_m"] for c in chosen.values()), default=0.0)
    return rec
