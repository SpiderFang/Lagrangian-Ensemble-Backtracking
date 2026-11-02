# -*- coding: utf-8 -*-
"""整頁版多面板分段機率圖（試作）：以 segprob_figures／segprob_annot 之繪製函式組成可調版面，
不改動 repo 內任何檔案（map_axes、report_style、segprob_* 皆原樣使用）。

版面由「列數、欄數、備用格（色階與圖例放入備用格）」決定；面板大小由畫布寬 6.27 in 與整頁可用高度共同限制，
取兩者較小者（地圖面板長寬比固定為 SPEC.frame_width_over_height）。
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/wp75_segprob（wp75_common、run_v3）
import wp75_common as C  # noqa: E402
import run_v3 as R3  # noqa: E402

from lagrangian_backtracking.native_runner import figures as F  # noqa: E402
from lagrangian_backtracking.native_runner import map_axes as ma  # noqa: E402
from lagrangian_backtracking.native_runner import report_style as rs  # noqa: E402
from lagrangian_backtracking.native_runner import segprob as sp  # noqa: E402
from lagrangian_backtracking.native_runner import segprob_annot as SA  # noqa: E402
from lagrangian_backtracking.native_runner import segprob_figures as SF  # noqa: E402

W_IN = rs.REPORT_WIDTH_IN  # 6.27
RATIO = SF.SPEC.frame_width_over_height  # 地圖面板寬／高
KIND_ZH = {"boundary": "穿越", "coast": "海岸來源"}
SEASONS = R3.SEASON_ORDER
TIDES = R3.TIDE_ORDER


@dataclass
class Site:
    sd: object
    dr: object
    grp: dict
    label: dict
    mats: list
    bcount: dict
    ccount: dict
    gn: dict
    font: object
    land: list


def prep_site(sid: str) -> Site:
    regions, sites = C.load_config()
    anchors = F._load_receptor_anchors(C.RELEASE)  # noqa: SLF001
    land = C.load_land()
    font = F._resolve_font(None).font_properties  # noqa: SLF001
    sd = C.build_site(sid, land, regions, sites, anchors)
    sd.cols = C.load_site_columns(sid)
    C.assign_site(sd)
    grp, label, mats = R3.build_groups(sd.cols)
    nb, nc = len(sd.bseg), len(sd.cseg)
    gn = {k: int(m.sum()) for k, m in grp.items()}
    bcount = {k: sp.counts_by_segment(sd.b_idx[m & (sd.b_idx >= 0)], nb) for k, m in grp.items()}
    ccount = {k: sp.counts_by_segment(sd.c_idx[m & (sd.c_idx >= 0)], nc) for k, m in grp.items()}
    dr = SF.prepare_drawing(sd, land)
    return Site(sd, dr, grp, label, mats, bcount, ccount, gn, font, land)


# ---------------------------------------------------------------- 面板（標題與註記）
def lw_for(w: float) -> float:
    """線寬縮放：面板寬 0.9 in → 0.5、2.11 in → 0.85（同 v3 之 2×5、2×2），上限 1.0（等於單張圖）。"""
    return float(min(1.0, 0.5 + (w - 0.9) * (0.35 / 1.21)))


def star_for(w: float) -> float:
    return float(min(10.0, 6.0 + (w - 0.9) * 2.48))


def title_h_for(lines: int) -> float:
    return 0.30 + 0.15 * (lines - 2)


def make_panels(S: Site, kind: str, name: str, w: float, *, class_brackets: bool = False):
    """回傳 ``(panels, 標題最多行數, 標題值選擇紀錄)``；面板寬 ``w``（英吋）決定標題值換行與線寬。"""
    sd = S.sd
    keys = {
        "season": [f"season_{k}" for k in SEASONS],
        "tide": [f"tide_{k}" for k in TIDES],
        "class": [f"class_{m}" for m in S.mats],
    }[name]
    lw_scale, star_size = lw_for(w), star_for(w)
    panels, lines, log = [], [], []
    f8 = rs.text_font(S.font)
    for k in keys:
        cnt = S.bcount[k] if kind == "boundary" else S.ccount[k]
        n = S.gn[k]
        t = SF.pct2(int(cnt.sum()), n)
        base = f"{S.label[k]}\n{KIND_ZH[kind]} {t}%"
        annotate = None
        title = base
        if kind == "boundary" and (name != "class" or class_brackets):
            st = SA.boundary_stretches(sd, cnt, n)
            val, vi = SA.pick_value_variant(
                SA.boundary_value_variants(st, n, crossing=int(cnt.sum())), f8, w * 72.0 + 3.0
            )
            title = f"{base}\n{val}" if val else base
            lines.append(2 + (val.count("\n") + 1 if val else 0))
            log.append([S.label[k], vi, len(st)])

            def annotate(ax, fig, f8_, cnt=cnt, n=n):
                return SA.annotate_boundary(
                    ax, fig, sd, S.dr, cnt, n, font=f8_, star_size=star_size, lw_pt=SF.LW_BOUNDARY * lw_scale,
                    mode="marker", levels_pt=SA.BRACKET_LEVELS_PANEL_PT, br_lw=max(0.4, 0.8 * lw_scale), land_pad_pt=1.0,
                )

        elif kind == "boundary":
            lines.append(2)
        elif name in ("season", "tide"):
            st = SA.coast_stretches(sd, cnt, n)
            if st:
                title = f"{base}\n{SA.marker_title_line(sd, st, n)}"
                lines.append(3)

                def annotate(ax, fig, f8_, cnt=cnt, n=n, st=st):
                    return SA.annotate_coast_markers(
                        ax, fig, sd, S.dr, cnt, st, n, font=f8_, star_lonlat=sd.anchor, star_size=star_size,
                        lw_pt=SF.LW_COAST * lw_scale,
                    )
            else:
                lines.append(2)
        else:
            lines.append(2)
        panels.append(SF.Panel(title, cnt / n * 100.0, annotate, ()))
    return panels, max(lines), log


# ---------------------------------------------------------------- 版面幾何
@dataclass
class Geo:
    nrow: int
    ncol: int
    n: int
    w: float
    h: float
    title_h: float
    L: float
    R: float
    gx: float
    gy: float
    top: float
    bottom: float
    legend_band: float
    slot: bool
    x0: float  # 面板區塊左緣
    fh: float
    row_gap: list
    limit: str


def geometry(nrow, ncol, n, title_h, *, H=None, slot=False, L=0.66, R=0.90, gx=0.15, gy=0.10, top=0.08,
             bottom=0.45, legend_band=0.40, xlab_extra=0.22) -> Geo:
    if slot:
        R, legend_band = 0.10, 0.0
    ww = (W_IN - L - R - gx * (ncol - 1)) / ncol
    # 每列之後的列距：該列有面板位於其欄最底（下方無面板）且不是最後一列時，需為刻度標籤多留空間
    row_gap = []
    for r in range(nrow - 1):
        need = any((r * ncol + c < n) and ((r + 1) * ncol + c >= n) for c in range(ncol))
        row_gap.append(gy + (xlab_extra if need else 0.0))
    if H is None:
        w, limit = ww, "寬"
    else:
        fixed = top + bottom + legend_band + sum(row_gap) + nrow * title_h
        wh = (H - fixed) / nrow * RATIO
        w, limit = (ww, "寬") if ww <= wh else (wh, "高")
    h = w / RATIO
    fh = top + nrow * (title_h + h) + sum(row_gap) + bottom + legend_band
    block_w = ncol * w + (ncol - 1) * gx
    x0 = L + (W_IN - L - R - block_w) / 2.0
    return Geo(nrow, ncol, n, w, h, title_h, L, R, gx, gy, top, bottom, legend_band, slot, x0, fh, row_gap, limit)


# ---------------------------------------------------------------- 繪製
def render_grid(S: Site, kind: str, name: str, path: Path, *, nrow: int, ncol: int, H: float | None,
                slot: bool = False, class_brackets: bool = False, shared_titles: bool = False,
                tick_lon: float = 0.6, tick_lat: float = 0.4, inward_x: bool = True, dpi: int = 600) -> dict:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    sd, dr, font = S.sd, S.dr, S.font
    n_pan = {"season": 4, "tide": 2, "class": len(S.mats)}[name]
    th = title_h_for(2)
    for _ in range(4):  # 標題行數依面板寬決定，面板寬又依標題高度決定：迭代至穩定
        g = geometry(nrow, ncol, n_pan, th, H=H, slot=slot)
        panels, lines, tlog = make_panels(S, kind, name, g.w, class_brackets=class_brackets)
        th2 = title_h_for(lines)
        if abs(th2 - th) < 1e-9:
            break
        th = th2
    g = geometry(nrow, ncol, n_pan, th, H=H, slot=slot)
    panels, lines, tlog = make_panels(S, kind, name, g.w, class_brackets=class_brackets)
    f8 = rs.text_font(font)
    win = SF.WINDOWS[sd.site_id]
    lw_scale, star_size = lw_for(g.w), star_for(g.w)
    fw, fh = W_IN, g.fh
    with rs.report_rc(font.get_name()):
        rs.require_report_rc(font)
        fig = plt.figure(figsize=(fw, fh), dpi=150)
        axes, title_arts, ann = [], [], []
        ytop = fh - g.top
        row_top = []
        for r in range(nrow):
            row_top.append(ytop)
            ytop -= g.title_h + g.h + (g.row_gap[r] if r < nrow - 1 else 0.0)
        for i, p in enumerate(panels):
            r, c = divmod(i, ncol)
            x0 = g.x0 + c * (g.w + g.gx)
            y0 = row_top[r] - g.title_h - g.h
            ax = fig.add_axes((x0 / fw, y0 / fh, g.w / fw, g.h / fh))
            SF.draw_map(ax, sd, dr, kind, p.pct, window=win, lw_scale=lw_scale, star_size=star_size)
            last_row = (r == nrow - 1) or (i + ncol >= len(panels))
            SF._format_axes(  # noqa: SLF001
                ax, win, tick_lon, tick_lat, xlabels=last_row, ylabels=(c == 0), f8=f8,
                titles=not shared_titles, inward_x=inward_x,
            )
            title_arts.append(ax.set_title(p.title, fontproperties=f8, fontsize=rs.TEXT_PT, pad=2.0, linespacing=1.0))
            axes.append(ax)
            if p.annotate is not None:
                ann.append(p.annotate(ax, fig, f8))
        handles = SF._legend_handles(kind, None, dr.has_land_boundary)  # noqa: SLF001
        first_top = row_top[0] - g.title_h
        last_bottom = row_top[-1] - g.title_h - g.h
        block_w = ncol * g.w + (ncol - 1) * g.gx
        if not slot:
            SF._discrete_colorbar(fig, (g.x0 + block_w + rs.CBAR_PAD_IN, last_bottom, rs.CBAR_W_IN, first_top - last_bottom), f8)  # noqa: SLF001
            block_cx = (g.x0 + block_w / 2.0) / fw
            top_in = (g.legend_band - 0.02) if shared_titles else rs.axes_bottom_in(fig, axes[(nrow - 1) * ncol]) - rs.LEGEND_GAP_IN
            leg = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(block_cx, top_in / fh), ncol=len(handles),
                             prop=f8, **rs.FRAMED_LEGEND_KW)
        else:
            # 備用格：最後一列右側。色階在最右備用格（高度＝面板高），圖例在其左一格（垂直置中於面板高度）
            spare = list(range(n_pan, nrow * ncol))
            cb_slot, lg_slot = spare[-1], (spare[-2] if len(spare) > 1 else None)
            cc = cb_slot % ncol
            sx = g.x0 + cc * (g.w + g.gx)
            SF._discrete_colorbar(fig, (sx + 0.30, last_bottom, rs.CBAR_W_IN, g.h), f8)  # noqa: SLF001
            if lg_slot is not None:
                lc = lg_slot % ncol
                lx = g.x0 + lc * (g.w + g.gx) + g.w / 2.0
                leg = fig.legend(handles=handles, loc="center", bbox_to_anchor=(lx / fw, (last_bottom + g.h / 2) / fh),
                                 ncol=1, prop=f8, **rs.FRAMED_LEGEND_KW)
            else:
                leg = None
        if shared_titles:
            fig.text((g.x0 + block_w / 2) / fw, 0.06 / fh, ma.AXIS_TITLE_LON_ZH, ha="center", va="bottom", fontproperties=f8, fontsize=rs.TEXT_PT)
            fig.text(0.10 / fw, (first_top + last_bottom) / 2 / fh, ma.AXIS_TITLE_LAT_ZH, ha="center", va="center", rotation=90,
                     fontproperties=f8, fontsize=rs.TEXT_PT)
        fig.canvas.draw()
        rec = {
            "kind": kind, "site": sd.site_id, "name": name, "layout_rows_cols": [nrow, ncol], "slot_layout": slot,
            "canvas_in": [float(fw), float(fh)], "panel_w_h_in": [float(g.w), float(g.h)], "limit": g.limit,
            "title_h_in": g.title_h, "title_max_lines": lines, "n_panels": len(panels), "tick_step_lon_lat": [tick_lon, tick_lat],
            "window_lon_lat": list(win), "panel_titles": [p.title for p in panels], "annotations": ann,
            "legend_labels": [t.get_text() for t in leg.get_texts()] if leg is not None else [],
            "lw_scale": lw_scale, "star_size": star_size, "class_brackets": class_brackets, "title_choice": tlog,
        }
        return SF._finish(fig, Path(path), rec, dpi)  # noqa: SLF001
