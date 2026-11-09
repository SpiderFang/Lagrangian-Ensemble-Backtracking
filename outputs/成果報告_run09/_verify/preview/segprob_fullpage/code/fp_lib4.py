# -*- coding: utf-8 -*-
"""整頁版多面板圖第三版：單行標題（只寫組別）＋圖內數值框＋傳統共用色階。
沿用 fp_lib（Site、lw_for、star_for）與 fp_lib2（碰撞檢查工具），不改 repo 內任何檔案。

數值框（value box）：
  白底（alpha 0.85）、細灰框（#9e9e9e、0.4 pt）、8 pt、靠左對齊，內容每行一項：
  邊界「穿越 xx.xx%」「① …」「② …」「其餘 …」；海岸「海岸來源 xx.xx%」「① …」。
  位置：每個地點、每種圖（邊界／海岸）選一個固定角落；各面板同一點（角落錨點）。
  避讓：陸地、島嶼、星號、所有面板中大於 0 的上色線段（含括號與圈號所需之緣距）；0% 與無值的邊界線可被壓在框下。
色階模式：
  hbar    共用橫式色階置於面板區塊正下方並置中，標籤置於色階條下方置中，圖例在其下一行置中
  vbar    共用直式色階在區塊右側，長度＝一個面板高，垂直置中於區塊，標籤在右側
  slotLR  十類：色階在左備用格，圖例在右備用格
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import shapely
from scipy import ndimage as ndi
from shapely.geometry import LineString, Point, box as sbox
from shapely.ops import transform as shp_transform

sys.path.insert(0, str(Path(__file__).parent))
import fp_lib as L0  # noqa: E402
import fp_lib2 as L2  # noqa: E402

rs, SF, ma, sp, SA = L0.rs, L0.SF, L0.ma, L0.sp, L0.SA
W_IN, RATIO = L0.W_IN, L0.RATIO
PT = 1.0 / 72.0
BAR_H = 0.12
LEGKW = L2.LEGKW
BOX_PAD = 0.2
SOFT = True
RING_OK = False  # 最後備援（第 2 階）：數值框可蓋住海岸圖之流場範圍細灰線（zorder 3.2，不蓋陸地與上色線段）
COMPACT = False  # 備援：多個括號段之圈號項併成一行（僅在標準框放不下時由 fp_final 啟用）
BOX_FC = (1.0, 1.0, 1.0, 0.85)
BOX_EC = "#9e9e9e"
BOX_LW = 0.4
GRID = 0.01
MARGINS = (0.22, 0.18, 0.14, 0.10, 0.07, 0.05, 0.035, 0.028, 0.022)
CORNERS = ("UR", "UL", "LR", "LL")
DEBUG = False


# ------------------------------------------------------------------ 面板與框內文字
def make_panels3(S, kind: str, name: str, w: float, *, class_brackets: bool = False):
    sd = S.sd
    keys = {"season": [f"season_{k}" for k in L0.SEASONS], "tide": [f"tide_{k}" for k in L0.TIDES],
            "class": [f"class_{m}" for m in S.mats]}[name]
    lw_scale, star_size = L0.lw_for(w), L0.star_for(w)
    panels, boxes, log, pcts = [], [], [], []
    for k in keys:
        cnt = S.bcount[k] if kind == "boundary" else S.ccount[k]
        n = S.gn[k]
        t = SF.pct2(int(cnt.sum()), n)
        head = f"{L0.KIND_ZH[kind]} {t}%"
        annotate = None
        lines = [head]
        if kind == "boundary" and (name != "class" or class_brackets):
            st = SA.boundary_stretches(sd, cnt, n)
            val = SA.boundary_value_variants(st, n, crossing=int(cnt.sum()))[0]
            parts = val.split("　") if val else []
            if COMPACT and len([q for q in parts if q[:1] in "①②③④⑤"]) > 1:
                its = [q for q in parts if q[:1] in "①②③④⑤"]
                parts = ["　".join(its)] + [q for q in parts if q[:1] not in "①②③④⑤"]
            lines += parts
            log.append([S.label[k], len(st)])

            def annotate(ax, fig, f8_, cnt=cnt, n=n):
                return SA.annotate_boundary(
                    ax, fig, sd, S.dr, cnt, n, font=f8_, star_size=star_size, lw_pt=SF.LW_BOUNDARY * lw_scale,
                    mode="marker", levels_pt=SA.BRACKET_LEVELS_PANEL_PT, br_lw=max(0.4, 0.8 * lw_scale), land_pad_pt=1.0,
                )
        elif kind == "coast" and name in ("season", "tide"):
            st = SA.coast_stretches(sd, cnt, n)
            if st:
                cparts = SA.marker_title_line(sd, st, n).split("　")
                if COMPACT and len(cparts) > 2:
                    cparts = [cparts[0], "　".join(cparts[1:])]
                lines += cparts

                def annotate(ax, fig, f8_, cnt=cnt, n=n, st=st):
                    return SA.annotate_coast_markers(
                        ax, fig, sd, S.dr, cnt, st, n, font=f8_, star_lonlat=sd.anchor, star_size=star_size,
                        lw_pt=SF.LW_COAST * lw_scale,
                    )
        panels.append(SF.Panel(S.label[k], cnt / n * 100.0, annotate, ()))
        boxes.append(lines)
        pcts.append(cnt)
    return panels, boxes, log


def all_box_lines(S, kind: str, class_brackets: bool):
    out = []
    for name in ("season", "tide", "class"):
        _p, b, _l = make_panels3(S, kind, name, 2.0, class_brackets=class_brackets)
        out += b
    return out


# ------------------------------------------------------------------ 數值框位置
def _measure_boxes(S, lines_list):
    """量測各內容之框尺寸（英吋）；回傳每項 (w, h)。"""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    f8 = rs.text_font(S.font)
    out = []
    with rs.report_rc(S.font.get_name()):
        fig = plt.figure(figsize=(3, 3), dpi=150)
        ax = fig.add_axes((0, 0, 1, 1))
        ts = [ax.text(0.1, 0.1, "\n".join(ls), transform=ax.transAxes, ha="left", va="bottom", multialignment="left",
                      fontproperties=f8, fontsize=rs.TEXT_PT, linespacing=1.0, bbox=dict(boxstyle=f"square,pad={BOX_PAD}")) for ls in lines_list]
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        for t in ts:
            b = t.get_bbox_patch().get_window_extent(rend)
            out.append((b.width / fig.dpi, b.height / fig.dpi))
        plt.close(fig)
    return out


_RC: dict = {}


def _dims(w, h):
    return int(np.floor(w / GRID)), int(np.floor(h / GRID))


def _raster_lines(lines, w, h):
    nx, ny = _dims(w, h)
    g = np.zeros((ny, nx), dtype=bool)
    for a in lines:
        a = np.asarray(a, dtype=float)
        if len(a) == 0:
            continue
        pts = [a[:1]]
        for p0, p1 in zip(a[:-1], a[1:], strict=True):
            n = int(np.ceil(np.hypot(*(p1 - p0)) / (GRID / 2.0))) + 1
            pts.append(np.linspace(p0, p1, n)[1:])
        q_ = np.vstack(pts)
        ix = np.floor(q_[:, 0] / GRID).astype(int)
        iy = np.floor(q_[:, 1] / GRID).astype(int)
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        g[iy[ok], ix[ok]] = True
    return g


def _dist(g):
    if not g.any():
        return np.full(g.shape, 99.0)
    return ndi.distance_transform_edt(~g) * GRID


def _tr(S, w, h):
    win = SF.WINDOWS[S.sd.site_id]
    sx, sy = w / (win[1] - win[0]), h / (win[3] - win[2])
    return win, sx, sy


def static_occ(S, kind, w, h, land_min_area=None, ring=True):
    """陸地（含島嶼）、星號（半徑 0.10 in）、海岸圖之流場範圍細線（緣距 0.03 in）之占用格（以 (站, 種類, w) 快取）。"""
    key = ("static", S.sd.site_id, kind, round(w, 3), land_min_area, ring)
    if key in _RC:
        return _RC[key]
    dr = S.dr
    win, sx, sy = _tr(S, w, h)
    nx, ny = _dims(w, h)
    xs, ys = (np.arange(nx) + 0.5) * GRID, (np.arange(ny) + 0.5) * GRID
    X, Y = np.meshgrid(xs, ys)
    occ = np.zeros((ny, nx), dtype=bool)
    if dr.land_union is not None:
        def tr(lon, lat, z=None):
            return (np.asarray(lon) - win[0]) * sx, (np.asarray(lat) - win[2]) * sy
        land = shp_transform(tr, dr.land_union)
        if land_min_area is not None:  # 第二階：只迴避面積 ≥ land_min_area（in²）之陸地，不迴避極小島嶼
            land = shapely.union_all([g_ for g_ in shapely.get_parts(land) if g_.area >= land_min_area])
        land = land.buffer(0.01)
        occ |= shapely.contains_xy(land, X.ravel(), Y.ravel()).reshape(X.shape)
    ax_, ay_ = (S.sd.anchor[0] - win[0]) * sx, (S.sd.anchor[1] - win[2]) * sy
    occ |= np.hypot(X - ax_, Y - ay_) < 0.075
    if kind == "coast" and dr.ring_ll is not None and ring:
        r = np.column_stack([(dr.ring_ll[:, 0] - win[0]) * sx, (dr.ring_ll[:, 1] - win[2]) * sy])
        occ |= _dist(_raster_lines([r], w, h)) < 0.03
    _RC[key] = occ
    return occ


def seg_dist(S, kind, cnts, w, h, tag):
    """所有面板中 >0 之上色線段之距離場（in）。"""
    key = ("seg", S.sd.site_id, kind, tag, round(w, 3))
    if key in _RC:
        return _RC[key]
    dr = S.dr
    win, sx, sy = _tr(S, w, h)
    src = dr.b_sea if kind == "boundary" else dr.c_lines
    pos = np.zeros(len(src), dtype=bool)
    for c in cnts:
        pos |= np.asarray(c) > 0
    lines = []
    for i, parts in enumerate(src):
        if pos[i]:
            for a in parts:
                a = np.asarray(a)
                if len(a) >= 2:
                    lines.append(np.column_stack([(a[:, 0] - win[0]) * sx, (a[:, 1] - win[2]) * sy]))
    d = _dist(_raster_lines(lines, w, h))
    _RC[key] = d
    return d


def artist_occ(axes, fig, w, h):
    """實際已畫在各面板上之註記（圈號文字、括號與引線、星號）之占用格（聯集，緣距 0.03 in）。"""
    rend = fig.canvas.get_renderer()
    nx, ny = _dims(w, h)
    occ = np.zeros((ny, nx), dtype=bool)
    lines_pts = []
    for ax in axes:
        bb = ax.get_window_extent(rend)
        ox, oy, dpi = bb.x0, bb.y0, fig.dpi
        for t in ax.texts:
            if not t.get_text():
                continue
            b = t.get_window_extent(rend)
            x0, x1 = (b.x0 - ox) / dpi - 0.03, (b.x1 - ox) / dpi + 0.03
            y0, y1 = (b.y0 - oy) / dpi - 0.03, (b.y1 - oy) / dpi + 0.03
            i0, i1 = max(int(np.floor(x0 / GRID)), 0), min(int(np.ceil(x1 / GRID)), nx)
            j0, j1 = max(int(np.floor(y0 / GRID)), 0), min(int(np.ceil(y1 / GRID)), ny)
            if i1 > i0 and j1 > j0:
                occ[j0:j1, i0:i1] = True
        for ln in ax.lines:
            xy = ln.get_xydata()
            if xy is None or len(xy) == 0:
                continue
            d = ax.transData.transform(xy)
            pts = np.column_stack([(d[:, 0] - ox) / dpi, (d[:, 1] - oy) / dpi])
            r = (ln.get_markersize() / 2.0 / 72.0 + 0.02) if ln.get_marker() not in (None, "", "None", " ") else 0.0
            g = _raster_lines([pts], w, h)
            occ |= _dist(g) < max(0.03, r)
        for p_ in ax.patches:
            if p_.get_zorder() <= 1:
                continue
            v = ax.transData.transform(p_.get_path().vertices) if p_.get_transform() == ax.transData else None
            if v is not None:
                pts = np.column_stack([(v[:, 0] - ox) / dpi, (v[:, 1] - oy) / dpi])
                occ |= _dist(_raster_lines([pts], w, h)) < 0.03
    return occ


def free_positions(occ, bw: float, bh: float):
    ii = np.pad(occ.astype(np.int32).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    bwc, bhc = int(np.ceil(bw / GRID)), int(np.ceil(bh / GRID))
    ny, nx = occ.shape
    ok = np.zeros((max(ny - bhc + 1, 0), max(nx - bwc + 1, 0)), dtype=bool)
    for j in range(ok.shape[0]):
        s = ii[j + bhc, bwc: bwc + ok.shape[1]] - ii[j, bwc: bwc + ok.shape[1]] - ii[j + bhc, : ok.shape[1]] + ii[j, : ok.shape[1]]
        ok[j] = s == 0
    return ok, bwc, bhc


LAND_MIN_AREA = 0.02  # in²；第二階（極小島嶼可被框蓋住）之陸地面積門檻
MARGINS_TIGHT = (0.016, 0.012)


def place_box(S, kind, cnts, w, h, bw, bh, corner=None, margins=MARGINS, extra=None, tag="", corners=None):
    """回傳 {corner, margin, x, y, ha, va, dist, tier}（x, y 為框之錨點角，英吋；tier 0＝完全避開陸地、1＝允許蓋住面積 < 0.02 in² 之小島）。
    corner 給定時只在該角落就近找；corners（清單）時回傳 {角落: 結果}（找不到者不在其中）。
    搜尋次序：先完全避開陸地、緣距由大到小（含最後兩個貼緊值 0.016／0.012 in），再允許蓋住極小島嶼。"""
    nx, ny = _dims(w, h)
    xs, ys = (np.arange(nx) + 0.5) * GRID, (np.arange(ny) + 0.5) * GRID
    X, Y = np.meshgrid(xs, ys)
    frame = (X < 0.04) | (X > w - 0.04) | (Y < 0.04) | (Y > h - 0.04)
    dsg = seg_dist(S, kind, cnts, w, h, tag)
    want = list(corners) if corners else ([corner] if corner else list(CORNERS))
    found = {}
    tiers = [(None, True), (LAND_MIN_AREA, True)] + ([(LAND_MIN_AREA, False)] if (RING_OK and kind == "coast") else [])
    for tier, (lma, ring_) in enumerate(tiers):
        base = static_occ(S, kind, w, h, lma, ring_) | frame
        if extra is not None:
            base = base | extra
        for m in tuple(margins) + MARGINS_TIGHT:
            occ = base | (dsg < m)
            ok, bwc, bhc = free_positions(occ, bw, bh)
            jj, ii = np.nonzero(ok)
            if len(jj) == 0:
                continue
            x0, y0 = ii * GRID, jj * GRID
            for c in want:
                if c in found:
                    continue
                xr, yt = x0 + bwc * GRID, y0 + bhc * GRID
                ap = {"UR": (xr, yt, w, h), "UL": (x0, yt, 0, h), "LR": (xr, y0, w, 0), "LL": (x0, y0, 0, 0)}[c]
                d = np.hypot(ap[0] - ap[2], ap[1] - ap[3])
                k = int(np.argmin(d))
                bx, by = float(x0[k]), float(y0[k])
                bw_, bh_ = bwc * GRID, bhc * GRID
                anchor = {"UR": (bx + bw_, by + bh_), "UL": (bx, by + bh_), "LR": (bx + bw_, by), "LL": (bx, by)}[c]
                found[c] = dict(corner=c, margin=m, x=anchor[0], y=anchor[1], ha="right" if c in ("UR", "LR") else "left",
                                va="top" if c in ("UR", "UL") else "bottom", dist=float(d[k]), tier=tier)
            if all(c in found for c in want):
                break
        if all(c in found for c in want):
            break
    if corners is not None:
        return found
    return found.get(want[0]) if corner else (min(found.values(), key=lambda r_: r_["dist"]) if found else None)


# ------------------------------------------------------------------ 框與地圖元素之實測碰撞
def box_vs_artists(ax, box_text, rend, fig, land_ok_area=None):
    """框之外觀範圍（顯示座標）與面板內其他元素（文字、括號線、星號、上色線段、陸地）之重疊清單。"""
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import PathPatch

    bad = []
    bb = box_text.get_bbox_patch().get_window_extent(rend).expanded(1.0, 1.0)
    x0, y0, x1, y1 = bb.x0 - 0.5, bb.y0 - 0.5, bb.x1 + 0.5, bb.y1 + 0.5

    def inside(p):
        return (p[:, 0] >= x0) & (p[:, 0] <= x1) & (p[:, 1] >= y0) & (p[:, 1] <= y1)

    for t in ax.texts:
        if t is box_text or not t.get_text():
            continue
        b = t.get_window_extent(rend)
        if not (b.x1 < x0 or b.x0 > x1 or b.y1 < y0 or b.y0 > y1):
            bad.append(f"數值框壓到文字「{t.get_text()[:8]}」")
    for ln in ax.lines:
        xy = ln.get_xydata()
        if xy is None or len(xy) == 0:
            continue
        d = ax.transData.transform(xy)
        if len(d) >= 2:
            segs = [np.linspace(d[i], d[i + 1], 12) for i in range(len(d) - 1)]
            d = np.vstack(segs)
        r = (ln.get_markersize() / 2.0 * fig.dpi / 72.0) if ln.get_marker() not in (None, "", "None", " ") else 0.0
        hit = inside(d)
        if r > 0:
            hit = hit | ((d[:, 0] >= x0 - r) & (d[:, 0] <= x1 + r) & (d[:, 1] >= y0 - r) & (d[:, 1] <= y1 + r))
        if hit.any():
            bad.append("數值框壓到" + ("星號" if r > 0 else "括號／線"))
    for col in ax.collections:
        if isinstance(col, LineCollection) and col.get_zorder() >= 4.005:
            for seg in col.get_segments():
                d = ax.transData.transform(seg)
                d = np.vstack([np.linspace(d[i], d[i + 1], 12) for i in range(len(d) - 1)]) if len(d) >= 2 else d
                if len(d) and inside(d).any():
                    bad.append("數值框壓到上色線段")
                    break
    gx = np.linspace(x0, x1, 14)
    gy = np.linspace(y0, y1, 10)
    pts = np.array([(a, b) for a in gx for b in gy])
    inv = ax.transData.inverted().transform(pts)
    info = []
    dpi = fig.dpi
    for p in ax.patches:
        if isinstance(p, PathPatch) and p.get_zorder() <= 1:
            if p.get_path().contains_points(inv).any():
                v = ax.transData.transform(p.get_path().vertices) / dpi
                area = 0.5 * abs(np.dot(v[:, 0], np.roll(v[:, 1], -1)) - np.dot(v[:, 1], np.roll(v[:, 0], -1)))
                if land_ok_area is not None and area < land_ok_area:
                    info.append(f"蓋住小島（約 {area:.4f} in²）")
                else:
                    bad.append("數值框壓到陸地")
                    break
    return bad, info


# ------------------------------------------------------------------ 版面幾何
def band_height(mode, st):
    if mode == "hbar":
        return st["XB"] + 0.06 + BAR_H + st["BAR_TL"] + 0.02 + st["LAB_H"] + 0.07 + st["LEG_H"] + 0.015
    if mode in ("vbar", "vfull", "vrow"):
        return st["XB"] + 0.05 + st["LEG_H"] + 0.015
    return st["XB"] + 0.015


def geo3(nrow, ncol, n, th, H, mode, st, Lm, Rm, gx, gy, top, xlab_extra=0.20):
    row_gap = []
    for r in range(nrow - 1):
        need = any((r * ncol + c < n) and ((r + 1) * ncol + c >= n) for c in range(ncol))
        row_gap.append(gy + (xlab_extra if need else 0.0))
    band = band_height(mode, st)
    ww = (W_IN - Lm - Rm - gx * (ncol - 1)) / ncol
    fixed = top + band + sum(row_gap) + nrow * th
    if H is None:
        w, limit = ww, "寬"
    else:
        wh = (H - fixed) / nrow * RATIO
        w, limit = (ww, "寬") if ww <= wh else (wh, "高")
    h = w / RATIO
    fh = fixed + nrow * h
    block_w = ncol * w + (ncol - 1) * gx
    x0 = Lm + (W_IN - Lm - Rm - block_w) / 2.0
    return dict(nrow=nrow, ncol=ncol, w=w, h=h, th=th, gx=gx, gy=gy, top=top, band=band, row_gap=row_gap, fh=fh,
                x0=x0, block_w=block_w, limit=limit, L=Lm, R=Rm)


def build3(S, kind, name, nrow, ncol, H, mode, class_brackets, shared_titles, st, Lm, Rm, gx, gy, th, corner, cnts_all,
           box_sizes, bar_frac=0.65, tick_lon=0.6, tick_lat=0.4, top=0.012):
    import matplotlib.cm as mcm
    import matplotlib.pyplot as plt

    sd, dr, font = S.sd, S.dr, S.font
    n_pan = {"season": 4, "tide": 2, "class": len(S.mats)}[name]
    g = geo3(nrow, ncol, n_pan, th, H, mode, st, Lm, Rm, gx, gy, top)
    panels, boxlines, tlog = make_panels3(S, kind, name, g["w"], class_brackets=class_brackets)
    f8 = rs.text_font(font)
    win = SF.WINDOWS[sd.site_id]
    lw_scale, star_size = L0.lw_for(g["w"]), L0.star_for(g["w"])
    fw, fh = W_IN, g["fh"]
    fig = plt.figure(figsize=(fw, fh), dpi=150)
    axes, ann = [], []
    ytop = fh - g["top"]
    row_top = []
    for r in range(nrow):
        row_top.append(ytop)
        ytop -= g["th"] + g["h"] + (g["row_gap"][r] if r < nrow - 1 else 0.0)
    for i, p in enumerate(panels):
        r, c = divmod(i, ncol)
        x0 = g["x0"] + c * (g["w"] + gx)
        y0 = row_top[r] - g["th"] - g["h"]
        ax = fig.add_axes((x0 / fw, y0 / fh, g["w"] / fw, g["h"] / fh))
        SF.draw_map(ax, sd, dr, kind, p.pct, window=win, lw_scale=lw_scale, star_size=star_size)
        last_row = (r == nrow - 1) or (i + ncol >= len(panels))
        SF._format_axes(ax, win, tick_lon, tick_lat, xlabels=last_row, ylabels=(c == 0), f8=f8,  # noqa: SLF001
                        titles=not shared_titles, inward_x=True)
        ax.set_title(p.title, fontproperties=f8, fontsize=rs.TEXT_PT, pad=2.0, linespacing=1.0)
        axes.append(ax)
        if p.annotate is not None:
            ann.append(p.annotate(ax, fig, f8))
    # ---- 數值框：本圖最大框尺寸 → 固定錨點
    sizes = _measure_boxes_cached(S, boxlines)
    bw = max(s[0] for s in sizes)
    bh = max(s[1] for s in sizes)
    cnts = [S.bcount[k] if kind == "boundary" else S.ccount[k] for k in _keys(S, name)]
    fig.canvas.draw()
    xo = artist_occ(axes, fig, g["w"], g["h"])
    pl = place_box(S, kind, cnts, g["w"], g["h"], bw, bh, corner=corner, extra=xo, tag=name)
    box_texts = []
    if pl is None:
        if not SOFT:
            raise RuntimeError(f"數值框找不到位置：{kind} {name} {nrow}x{ncol}")
        c_ = corner or "UR"  # 軟失敗：版面尚未收斂（面板寬度仍在調整）時先以角落暫放，收斂後仍放不下才報錯
        pl = dict(corner=c_, margin=0.0, x=(g["w"] - 0.05) if c_ in ("UR", "LR") else 0.05, y=(g["h"] - 0.05) if c_ in ("UR", "UL") else 0.05,
                  ha="right" if c_ in ("UR", "LR") else "left", va="top" if c_ in ("UR", "UL") else "bottom", dist=0.0, tier=0, soft_fail=True)
    pad_in = BOX_PAD * 8.0 * PT
    sgn_x = -1 if pl["ha"] == "right" else 1
    sgn_y = -1 if pl["va"] == "top" else 1
    for ax, ls in zip(axes, boxlines):
        t = ax.text((pl["x"] + sgn_x * pad_in) / g["w"], (pl["y"] + sgn_y * pad_in) / g["h"], "\n".join(ls),
                    transform=ax.transAxes, ha=pl["ha"], va=pl["va"], multialignment="left", fontproperties=f8,
                    fontsize=rs.TEXT_PT, linespacing=1.0, zorder=12,
                    bbox=dict(boxstyle=f"square,pad={BOX_PAD}", fc=BOX_FC, ec=BOX_EC, lw=BOX_LW))
        box_texts.append(t)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    d = fig.dpi
    last_bottom = row_top[-1] - g["th"] - g["h"]
    first_top = row_top[0] - g["th"]
    xs_bot = []
    for i, ax in enumerate(axes):
        r = i // ncol
        if (r == nrow - 1) or (i + ncol >= len(panels)):
            for t in ax.xaxis.get_major_ticks():
                if t.label1.get_visible() and t.label1.get_text():
                    xs_bot.append(t.label1.get_window_extent(rend).y0 / d)
            if ax.xaxis.label.get_text():
                xs_bot.append(ax.xaxis.label.get_window_extent(rend).y0 / d)
    lowest = min(xs_bot) if xs_bot else last_bottom - 0.2
    meas = {}
    handles = SF._legend_handles(kind, None, dr.has_land_boundary)  # noqa: SLF001
    ftexts, legends, caxs = [], [], []
    block_cx = g["x0"] + g["block_w"] / 2.0
    if shared_titles:
        ylab_min = min(lab.get_window_extent(rend).x0 / d for ax in axes[::ncol]
                       for lab in ax.get_yticklabels() if lab.get_visible() and lab.get_text())
        ftexts.append(fig.text((ylab_min - 0.085) / fw, (first_top + last_bottom) / 2 / fh, ma.AXIS_TITLE_LAT_ZH, ha="center",
                               va="center", rotation=90, fontproperties=f8, fontsize=rs.TEXT_PT))
        tx = fig.text(block_cx / fw, (lowest - 0.012) / fh, ma.AXIS_TITLE_LON_ZH, ha="center", va="top", fontproperties=f8,
                      fontsize=rs.TEXT_PT)
        ftexts.append(tx)
        lowest = tx.get_window_extent(rend).y0 / d
    meas["XB"] = last_bottom - lowest
    spare = list(range(len(panels), nrow * ncol))
    meas["bar_frac_used"] = None
    if mode == "slotLR":
        cb_slot, lg_slot = spare[0], spare[1]
        sx = g["x0"] + (cb_slot % ncol) * (g["w"] + gx)
        cb = SF._discrete_colorbar(fig, (sx + (g["w"] - 0.75) / 2.0, last_bottom, rs.CBAR_W_IN, g["h"]), f8)  # noqa: SLF001
        caxs.append(cb.ax)
        lx = g["x0"] + (lg_slot % ncol) * (g["w"] + gx) + g["w"] / 2.0
        leg = fig.legend(handles=handles, loc="center", bbox_to_anchor=(lx / fw, (last_bottom + g["h"] / 2) / fh), ncol=1,
                         prop=f8, **LEGKW)
        legends.append(leg)
    elif mode in ("vbar", "vfull", "vrow"):
        ymid = (first_top + last_bottom) / 2.0
        if mode == "vfull":  # 與全年單張圖相同：色階高度＝格網全高（最上列面板頂到最下列面板底）
            cb_box = (g["x0"] + g["block_w"] + rs.CBAR_PAD_IN, last_bottom, rs.CBAR_W_IN, first_top - last_bottom)
        else:
            cb_box = (g["x0"] + g["block_w"] + rs.CBAR_PAD_IN, ymid - g["h"] / 2.0, rs.CBAR_W_IN, g["h"])
        if mode == "vrow":  # 每列各一條全年樣式色階，高度＝該列面板高
            for r in range(nrow):
                cbr = SF._discrete_colorbar(  # noqa: SLF001
                    fig, (g["x0"] + g["block_w"] + rs.CBAR_PAD_IN, row_top[r] - g["th"] - g["h"], rs.CBAR_W_IN, g["h"]), f8)
                caxs.append(cbr.ax)
        else:
            cb = SF._discrete_colorbar(fig, cb_box, f8)  # noqa: SLF001
            caxs.append(cb.ax)
        leg = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(block_cx / fw, (lowest - 0.05) / fh),
                         ncol=len(handles), prop=f8, **LEGKW)
        legends.append(leg)
    else:  # hbar
        cmap, norm, cols = SF.colormap_and_norm()
        labs = sp.bin_labels(SF.EDGES)
        need = len(labs) * (max(SA.text_width_pt(f8, s) for s in labs) / 72.0 + 0.05)
        bw_bar = min(g["block_w"], max(bar_frac * g["block_w"], need))
        meas["bar_frac_used"] = bw_bar / g["block_w"]
        bar_top = lowest - 0.06
        bar_y = bar_top - BAR_H
        cax = fig.add_axes(((block_cx - bw_bar / 2.0) / fw, bar_y / fh, bw_bar / fw, BAR_H / fh))
        cbar = fig.colorbar(mcm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", ticks=np.arange(len(cols)) + 0.5)
        cbar.ax.set_xticklabels(labs)
        cbar.ax.tick_params(axis="x", labelsize=rs.TEXT_PT, pad=rs.CBAR_TICK_PAD_PT, length=0)
        for lab in cbar.ax.get_xticklabels():
            lab.set_fontproperties(f8)
        for spine in cbar.ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(rs.FRAME_LW_PT)
            spine.set_edgecolor(rs.FRAME_COLOUR)
        caxs.append(cax)
        fig.canvas.draw()
        tl_bottom = min(l.get_window_extent(rend).y0 / d for l in cbar.ax.get_xticklabels())
        meas["BAR_TL"] = bar_y - tl_bottom
        lab_t = fig.text(block_cx / fw, (tl_bottom - 0.02) / fh, SF.COLORBAR_LABEL, ha="center", va="top", fontproperties=f8,
                         fontsize=rs.TEXT_PT)
        ftexts.append(lab_t)
        lb_ = L2._bbox_in(lab_t, fig, rend)
        meas["LAB_H"] = lb_[3] - lb_[1]
        leg = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(block_cx / fw, (lb_[1] - 0.07) / fh),
                         ncol=len(handles), prop=f8, **LEGKW)
        legends.append(leg)
    fig.canvas.draw()
    lb = L2._bbox_in(legends[0], fig, rend)
    meas["LEG_H"] = lb[3] - lb[1]
    texts, frames = L2._gather(fig, axes, caxs, legends, ftexts)
    # 數值框納入碰撞檢查
    box_items = []
    for i, t in enumerate(box_texts):
        bb = t.get_bbox_patch().get_window_extent(rend)
        box_items.append((f"box{i}", (bb.x0 / d, bb.y0 / d, bb.x1 / d, bb.y1 / d), i))
    minx = min(b[0] for _, b in texts)
    maxx = max(b[2] for _, b in texts)
    tit = [b[3] - b[1] for nme, b in texts if nme.endswith(".title")]
    meas.update(minx=minx, maxx=maxx, title_h=max(tit) + 2.0 * PT + 0.004, block_right=g["x0"] + g["block_w"],
                RIGHT_W=None)
    if mode in ("vbar", "vfull", "vrow"):
        meas["RIGHT_W"] = maxx - meas["block_right"] + 0.015
    return fig, g, panels, boxlines, tlog, ann, texts, frames, legends, meas, box_texts, box_items, pl, (bw, bh)


_KEYS = {}


def _keys(S, name):
    return {"season": [f"season_{k}" for k in L0.SEASONS], "tide": [f"tide_{k}" for k in L0.TIDES],
            "class": [f"class_{m}" for m in S.mats]}[name]


_MCACHE = {}


def _measure_boxes_cached(S, boxlines):
    key = (BOX_PAD, tuple(tuple(b) for b in boxlines))
    if key not in _MCACHE:
        _MCACHE[key] = _measure_boxes(S, boxlines)
    return _MCACHE[key]


def render_fit3(S, kind, name, path: Path, *, nrow, ncol, H, mode, corner, class_brackets=False, shared_titles=False,
                dpi=600, gx=0.10, gy=0.08, bar_frac=0.65, tick_lon=0.6, tick_lat=0.4) -> dict:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    f = S.font
    st = {"XB": 0.42, "LEG_H": 0.27, "BAR_TL": 0.15, "LAB_H": 0.14}
    Lm = 0.60
    Rm = 0.80 if mode in ("vbar", "vfull", "vrow") else 0.02
    th = 0.20
    res = None
    with rs.report_rc(f.get_name()):
        rs.require_report_rc(f)
        for it in range(8):
            if res is not None:
                plt.close(res[0])
            res = build3(S, kind, name, nrow, ncol, H, mode, class_brackets, shared_titles, st, Lm, Rm, gx, gy, th, corner,
                         None, None, bar_frac=bar_frac, tick_lon=tick_lon, tick_lat=tick_lat)
            (fig, g, panels, boxlines, tlog, ann, texts, frames, legends, meas, box_texts, box_items, pl, bsz) = res
            new_st = {k: meas[k] for k in ("XB", "LEG_H", "BAR_TL", "LAB_H") if k in meas}
            new_th = meas["title_h"]
            dL = (0.015 - meas["minx"]) if g["limit"] == "寬" else 0.0
            dR = 0.0
            if mode in ("vbar", "vfull", "vrow") and g["limit"] == "寬":
                dR = meas["RIGHT_W"] - Rm
            conv = (abs(dL) < 0.004 and abs(dR) < 0.004 and abs(new_th - th) < 0.004
                    and all(abs(new_st[k] - st[k]) < 0.004 for k in new_st))
            if DEBUG:
                print(f"  it{it} L={Lm:.3f} R={Rm:.3f} th={th:.3f}->{new_th:.3f} w={g['w']:.3f} fh={g['fh']:.3f} "
                      f"st={ {k: round(v, 3) for k, v in new_st.items()} }", flush=True)
            if conv:
                break
            st.update(new_st)
            th = new_th
            Lm += dL
            Rm += dR
        if pl.get("soft_fail"):
            plt.close(fig)
            raise RuntimeError(f"數值框找不到位置：{kind} {name} {nrow}x{ncol}（收斂後面板 {g['w']:.3f} in）")
        rend = fig.canvas.get_renderer()
        bad, min_sep = L2.collision_report(fig, texts, frames)
        # 數值框：與其他文字、其他面板框線、本面板邊緣、面板內元素
        axes = [a for a in fig.axes]
        islets = 0
        for (nm, bb, i) in box_items:
            for nt, bt in texts:
                gp, ov = L2._gap(bb, bt)
                if ov or gp < 1.0 * PT:
                    bad.append(f"數值框相碰：{nm} × {nt}")
            for fi, (nf, bf) in enumerate(frames):
                if nf == f"ax{i}":
                    ax_bb = bf
                    if bb[0] < ax_bb[0] + 0.03 or bb[2] > ax_bb[2] - 0.03 or bb[1] < ax_bb[1] + 0.03 or bb[3] > ax_bb[3] - 0.03:
                        bad.append(f"數值框超出面板：{nm}")
                    continue
                gp, ov = L2._gap(bb, bf)
                if ov:
                    bad.append(f"數值框壓框線：{nm} × {nf}")
            b_, i_ = box_vs_artists(fig.axes[i], box_texts[i], rend, fig, land_ok_area=(LAND_MIN_AREA if pl["tier"] >= 1 else None))
            bad += [f"面板{i}：{s}" for s in b_]
            islets += len(i_)
        rec = {
            "kind": kind, "site": S.sd.site_id, "name": name, "layout_rows_cols": [nrow, ncol], "mode": mode,
            "canvas_in": [float(W_IN), float(g["fh"])], "panel_w_h_in": [float(g["w"]), float(g["h"])], "limit": g["limit"],
            "title_h_in": g["th"], "n_panels": len(panels), "gx_in": gx, "gy_in": gy, "L_in": g["L"], "R_in": g["R"],
            "x0_in": g["x0"], "tick_step_lon_lat": [tick_lon, tick_lat], "panel_titles": [p.title for p in panels],
            "box_lines": boxlines, "box_corner": pl["corner"], "box_margin_in": pl["margin"], "box_size_in": list(bsz),
            "box_anchor_in": [pl["x"], pl["y"]], "box_land_tier": pl["tier"], "box_pad": BOX_PAD, "box_ring_ok": RING_OK, "box_compact": COMPACT, "box_islets_covered": islets, "annotations": ann, "class_brackets": class_brackets, "collisions": bad,
            "collision_count": len(bad), "min_text_sep_pt": min_sep, "title_choice": tlog,
            "legend_labels": [t.get_text() for t in legends[0].get_texts()], "lw_scale": L0.lw_for(g["w"]),
            "star_size": L0.star_for(g["w"]), "band_in": g["band"], "bar_frac_used": meas.get("bar_frac_used"),
        }
        return SF._finish(fig, Path(path), rec, dpi)  # noqa: SLF001
