"""WP-75：分段機率圖的註記——邊界四邊小計、集中段（stretch）括號與海岸集中段引線。

門檻常數都集中在本檔最上方（使用者可調）。所有註記數字由整數顆數經 :func:`segprob_figures.pct2`
（Decimal ROUND_HALF_UP）換算，與圖面其他百分比一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge, substring

from lagrangian_backtracking.native_runner import segprob_figures as SF

# ── 集中段門檻（一處調整）──────────────────────────────────────────────
STRETCH_BOUNDARY_MIN_SEG_PCT = Decimal("2")  # 邊界：相鄰分段各 ≥ 2%（2–5 階以上）才併成一段
STRETCH_BOUNDARY_MIN_TOTAL_PCT = Decimal("10")  # 邊界：集中段合計 ≥ 10% 才標示
STRETCH_BOUNDARY_MAX_PER_PANEL = 2
STRETCH_COAST_MIN_SEG_PCT = Decimal("0.5")  # 海岸：相鄰 5 km 分段各 ≥ 0.5% 才併成一段
STRETCH_COAST_MIN_TOTAL_PCT = Decimal("3")  # 海岸：集中段合計 ≥ 3% 才標示
STRETCH_COAST_MAX_PER_PANEL = 3
SIDE_SEA_MIN_LENGTH_M = 100.0  # 邊的海上長度小於此值視為純陸地邊，不標小計
BRACKET_OFFSET_M = 4_000.0  # 括號離邊界的距離（公尺）
BRACKET_TICK_M = 2_500.0  # 括號兩端短線長度（公尺）
SIDE_NAME_ZH = {"N": "北", "E": "東", "S": "南", "W": "西"}


@dataclass
class Stretch:
    kind: str  # boundary | coast
    chain: str  # 邊界："邊界"；海岸：連續海岸線代號
    indices: list[int]  # 分段序號（沿鏈順序）
    count: int


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
        elif closed and len(runs) == 1 and all(hot):
            pass
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


def side_counts(sd, b_s_masked: np.ndarray) -> dict[str, int]:
    """四邊穿越顆數（逐顆弧長判定所在邊）。"""

    s = b_s_masked[np.isfinite(b_s_masked)]
    sides = sd.ring.side_of(s)
    return {k: int((sides == k).sum()) for k in "NESW"}


def side_sea_geoms(sd) -> dict[str, object]:
    """每一邊的海上部分（投影公尺）；純陸地邊為空幾何。"""

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


def side_label(count: int, total: int, *, decimals: int = 2) -> str:
    """邊小計文字：保留 2 位小數；0 顆寫 0.00；有顆數但四捨五入為 0.00 寫 <0.01；``decimals=0`` 用於面板標題。"""  # noqa: E501

    if decimals == 0:
        return str((Decimal(count) * 100 / Decimal(total)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    t = SF.pct2(count, total)
    if count > 0 and Decimal(t) == 0:
        return "<0.01"
    return t


def side_title_string(sd, counts_by_side: dict[str, int], total: int) -> list[str]:
    """面板標題用的邊小計（0 位小數），例如 ``['北 23', '東 1', '南 45', '西 4']``；純陸地邊不列。"""

    geoms = side_sea_geoms(sd)
    return [
        f"{SIDE_NAME_ZH[k]} {side_label(counts_by_side[k], total, decimals=0)}"
        for k in "NESW"
        if geoms[k][1].length >= SIDE_SEA_MIN_LENGTH_M
    ]


# ── 繪製 ──────────────────────────────────────────────────────────────


def _bbox_overlap(a, b, pad: float = 2.0) -> bool:
    return not (a.x1 + pad < b.x0 or b.x1 + pad < a.x0 or a.y1 + pad < b.y0 or b.y1 + pad < a.y0)


class _Placer:
    """記錄已放的文字／括號外框，供後放的標籤避開。"""

    def __init__(self, fig, ax):
        self.fig, self.ax = fig, ax
        self.fig.canvas.draw()
        self.renderer = fig.canvas.get_renderer()
        self.boxes: list = []

    def ok(self, artist) -> bool:
        bb = artist.get_window_extent(self.renderer)
        axbb = self.ax.get_window_extent(self.renderer)
        inside = (
            bb.x0 >= axbb.x0 - 1 and bb.x1 <= axbb.x1 + 1 and bb.y0 >= axbb.y0 - 1 and bb.y1 <= axbb.y1 + 1
        )
        return inside and not any(_bbox_overlap(bb, o) for o in self.boxes)

    def add(self, artist) -> None:
        self.boxes.append(artist.get_window_extent(self.renderer))


def _lonlat(sd, xs, ys):
    lon, lat = sd.proj.unproject(np.asarray(xs), np.asarray(ys))
    return np.asarray(lon), np.asarray(lat)


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


def annotate_boundary(
    ax,
    fig,
    sd,
    seg_counts: np.ndarray,
    side_cnt: dict[str, int],
    n_total: int,
    *,
    font,
    stretches: bool = True,
    side_labels: bool = True,
) -> dict:
    """邊界圖註記：集中段括號（外側細線＋兩端短線＋合計 %）與各海上邊的小計；回傳註記紀錄（供表格與稽核）。"""

    rec: dict = {"stretches": [], "side_labels": {}}
    placer = _Placer(fig, ax)
    st = []
    if stretches:
        st = merge_stretches(
            seg_counts, n_total, boundary_chains(sd),
            min_seg_pct=STRETCH_BOUNDARY_MIN_SEG_PCT, min_total_pct=STRETCH_BOUNDARY_MIN_TOTAL_PCT,
            max_n=STRETCH_BOUNDARY_MAX_PER_PANEL, kind="boundary",
        )  # fmt: skip
    for s in st:
        geoms = [sd.bseg[i].geom for i in s.indices]
        line = linemerge(geoms) if len(geoms) > 1 else geoms[0]
        if isinstance(line, MultiLineString):  # 環繞西北角（首尾相接）時
            line = linemerge(list(line.geoms))
            if isinstance(line, MultiLineString):
                line = max(line.geoms, key=lambda g: g.length)
        sea = line if sd.land_proj is None or sd.land_proj.is_empty else line.difference(sd.land_proj)
        if sea.is_empty:
            continue
        if isinstance(sea, MultiLineString):
            sea = linemerge(sea)
            if isinstance(sea, MultiLineString):
                sea = max(sea.geoms, key=lambda g: g.length)
        off = _offset_outward(sd, sea, BRACKET_OFFSET_M)
        if off is None:
            continue
        ox, oy = np.asarray(off.coords).T
        lon, lat = _lonlat(sd, ox, oy)
        (ln,) = ax.plot(lon, lat, color="#222222", linewidth=0.8, solid_capstyle="butt", zorder=11)
        # 兩端短線：由括號端點朝邊界方向
        for p_off, p_in in ((off.coords[0], sea.coords[0]), (off.coords[-1], sea.coords[-1])):
            v = np.array(p_in) - np.array(p_off)
            v = v / (np.hypot(*v) or 1.0) * BRACKET_TICK_M
            x2, y2 = _lonlat(sd, [p_off[0], p_off[0] + v[0]], [p_off[1], p_off[1] + v[1]])
            ax.plot(x2, y2, color="#222222", linewidth=0.8, solid_capstyle="butt", zorder=11)
        # 合計％：括號中點、再往外
        mid = off.interpolate(0.5, normalized=True)
        mid_in = sea.interpolate(0.5, normalized=True)
        d = np.array([mid.x - mid_in.x, mid.y - mid_in.y])
        d = d / (np.hypot(*d) or 1.0)
        mlon, mlat = _lonlat(sd, [mid.x], [mid.y])
        horiz = abs(d[0]) < abs(d[1])  # 北／南邊：文字水平；東／西邊：文字轉 90°
        txt = f"{SF.pct2(s.count, n_total)}%"
        kwargs = dict(
            fontproperties=font, fontsize=8, zorder=13, rotation=0 if horiz else 90,
            ha="center" if horiz else ("left" if d[0] > 0 else "right"),
            va=("bottom" if d[1] > 0 else "top") if horiz else "center",
        )  # fmt: skip
        t = ax.annotate(
            txt, xy=(mlon[0], mlat[0]), xytext=(3 * d[0], 3 * d[1]), textcoords="offset points", **kwargs
        )
        placer.add(t)
        placer.boxes.append(ln.get_window_extent(placer.renderer))
        a = sd.bseg[s.indices[0]].geom.coords[0]
        b = sd.bseg[s.indices[-1]].geom.coords[-1]
        (alon, alat), (blon, blat) = _lonlat(sd, [a[0]], [a[1]]), _lonlat(sd, [b[0]], [b[1]])
        rec["stretches"].append(
            {"count": s.count, "pct": SF.pct2(s.count, n_total), "segments": [sd.bseg[i].seg_id for i in s.indices],  # noqa: E501
             "start_lonlat": [float(alon[0]), float(alat[0])], "end_lonlat": [float(blon[0]), float(blat[0])]}
        )  # fmt: skip
    if side_labels:
        geoms = side_sea_geoms(sd)
        for k in "NESW":
            line, sea = geoms[k]
            if sea.length < SIDE_SEA_MIN_LENGTH_M:
                continue
            txt = f"{SIDE_NAME_ZH[k]} {side_label(side_cnt[k], n_total)}%"
            rec["side_labels"][k] = txt
            parts = list(sea.geoms) if isinstance(sea, MultiLineString) else [sea]
            parts.sort(key=lambda g: -g.length)
            cand = []
            for g in parts:
                for f in (0.5, 0.3, 0.7, 0.15, 0.85, 0.05, 0.95):
                    cand.append(g.interpolate(f, normalized=True))
            outward = {"N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0)}[k]
            horiz = k in "NS"
            placed = None
            for pt in cand:
                lon, lat = _lonlat(sd, [pt.x], [pt.y])
                kwargs = dict(
                    fontproperties=font, fontsize=8, zorder=13, rotation=0 if horiz else 90,
                    ha="center" if horiz else ("left" if outward[0] > 0 else "right"),
                    va=("bottom" if outward[1] > 0 else "top") if horiz else "center",
                )  # fmt: skip
                t = ax.annotate(
                    txt,
                    xy=(lon[0], lat[0]),
                    xytext=(4 * outward[0], 4 * outward[1]),
                    textcoords="offset points",
                    **kwargs,
                )
                if placer.ok(t):
                    placed = t
                    break
                t.remove()
            if placed is None:  # 全部都擋到：退回中點（仍標出，由人工檢視）
                pt = cand[0]
                lon, lat = _lonlat(sd, [pt.x], [pt.y])
                placed = ax.annotate(txt, xy=(lon[0], lat[0]), xytext=(4 * outward[0], 4 * outward[1]), textcoords="offset points",  # noqa: E501
                                     fontproperties=font, fontsize=8, zorder=13, rotation=0 if horiz else 90,
                                     ha="center" if horiz else ("left" if outward[0] > 0 else "right"),
                                     va=("bottom" if outward[1] > 0 else "top") if horiz else "center")  # fmt: skip  # noqa: E501
                rec.setdefault("side_label_collisions", []).append(k)
            placer.add(placed)
    return rec


def annotate_coast(
    ax, fig, sd, seg_counts: np.ndarray, n_total: int, *, font, land_ll=None, star_lonlat=None
) -> dict:
    """海岸圖註記：集中段小引線＋合計 %（不寫地名）。"""

    rec: dict = {"stretches": []}
    st = merge_stretches(
        seg_counts, n_total, coast_chains(sd),
        min_seg_pct=STRETCH_COAST_MIN_SEG_PCT, min_total_pct=STRETCH_COAST_MIN_TOTAL_PCT,
        max_n=STRETCH_COAST_MAX_PER_PANEL, kind="coast",
    )  # fmt: skip
    placer = _Placer(fig, ax)
    land = sd.land_proj
    if star_lonlat is not None:  # 錨點星號當障礙物
        px, py = ax.transData.transform(star_lonlat)
        from matplotlib.transforms import Bbox

        placer.boxes.append(Bbox([[px - 9, py - 9], [px + 9, py + 9]]))
    inv = ax.transData.inverted()
    for s in st:
        geoms = [sd.cseg[i].geom for i in s.indices]
        line = linemerge(geoms) if len(geoms) > 1 else geoms[0]
        if isinstance(line, MultiLineString):
            line = max(line.geoms, key=lambda g: g.length)
        mid = line.interpolate(0.5, normalized=True)
        p1, p2 = (
            line.interpolate(max(line.length * 0.5 - 300, 0)),
            line.interpolate(min(line.length * 0.5 + 300, line.length)),
        )
        t = np.array([p2.x - p1.x, p2.y - p1.y])
        t = t / (np.hypot(*t) or 1.0)
        n1 = np.array([-t[1], t[0]])
        sea_dir = None
        for cand in (n1, -n1):
            probe = Point(mid.x + cand[0] * 2000, mid.y + cand[1] * 2000)
            if land is None or not land.contains(probe):
                sea_dir = cand
                break
        if sea_dir is None:
            sea_dir = n1
        # 閉合小島：往遠離島心的方向
        lon, lat = _lonlat(sd, [mid.x], [mid.y])
        txt = f"{SF.pct2(s.count, n_total)}%"
        done = None
        for dist in (13, 20, 28):
            for ang in (0, 35, -35, 70, -70, 110, -110, 180):
                th = np.deg2rad(ang)
                d = np.array(
                    [
                        sea_dir[0] * np.cos(th) - sea_dir[1] * np.sin(th),
                        sea_dir[0] * np.sin(th) + sea_dir[1] * np.cos(th),
                    ]
                )
                ha = "left" if d[0] > 0.35 else ("right" if d[0] < -0.35 else "center")
                va = "bottom" if d[1] > 0.35 else ("top" if d[1] < -0.35 else "center")
                a = ax.annotate(
                    txt, xy=(lon[0], lat[0]), xytext=(d[0] * dist, d[1] * dist), textcoords="offset points",
                    ha=ha, va=va, fontproperties=font, fontsize=8, zorder=14,
                    arrowprops=dict(arrowstyle="-", lw=0.6, color="#222222", shrinkA=0, shrinkB=0),
                    bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.75),
                )  # fmt: skip
                good = placer.ok(a)
                if good and land_ll is not None:  # 標籤外框的中心與四角都不可在陸地上
                    bb = a.get_window_extent(placer.renderer)
                    pts = [
                        ((bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2),
                        (bb.x0, bb.y0),
                        (bb.x0, bb.y1),
                        (bb.x1, bb.y0),
                        (bb.x1, bb.y1),
                    ]
                    good = not any(land_ll.contains(Point(*inv.transform(p))) for p in pts)
                if good:
                    done = a
                    break
                a.remove()
            if done is not None:
                break
        if done is None:  # 退回：直接放最短引線並登記碰撞
            d = sea_dir
            done = ax.annotate(
                txt, xy=(lon[0], lat[0]), xytext=(d[0] * 13, d[1] * 13), textcoords="offset points", ha="center", va="center",  # noqa: E501
                fontproperties=font, fontsize=8, zorder=14, arrowprops=dict(arrowstyle="-", lw=0.6, color="#222222"),  # noqa: E501
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.75),
            )  # fmt: skip
            rec.setdefault("label_collisions", []).append(txt)
        placer.add(done)
        a0, b0 = sd.cseg[s.indices[0]].geom.coords[0], sd.cseg[s.indices[-1]].geom.coords[-1]
        (alon, alat), (blon, blat) = _lonlat(sd, [a0[0]], [a0[1]]), _lonlat(sd, [b0[0]], [b0[1]])
        rec["stretches"].append(
            {"count": s.count, "pct": SF.pct2(s.count, n_total), "segments": [sd.cseg[i].seg_id for i in s.indices],  # noqa: E501
             "start_lonlat": [float(alon[0]), float(alat[0])], "end_lonlat": [float(blon[0]), float(blat[0])],
             "length_km": float(sum(sd.cseg[i].length_m for i in s.indices) / 1000)}
        )  # fmt: skip
    return rec
