# -*- coding: utf-8 -*-
"""整頁版多面板圖第二版：緊湊邊界＋底部橫式色階＋逐項碰撞檢查。
沿用 fp_lib 之 Site／make_panels／lw_for／star_for／title_h_for；不改 repo 內任何檔案。

版面模式（mode）：
  right  色階在右側（直式），圖例在面板區塊下方
  slot   色階（直式）與圖例放在最後一列之備用格
  bottom 色階為橫式，置於所有面板下方，標題在其左側、與面板區塊同寬；圖例在其下方或放備用格
邊界（L、R、gx、gy）自動貼緊：L 依實測最左緣、title_h 依實測標題高度；
gx、gy 由小到大嘗試，取第一組通過碰撞檢查者。
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import fp_lib as L0  # noqa: E402

rs, SF, ma, sp = L0.rs, L0.SF, L0.ma, L0.sp
W_IN, RATIO = L0.W_IN, L0.RATIO
PT = 1.0 / 72.0
GAP_CANDS = ((0.05, 0.04), (0.06, 0.05), (0.08, 0.06), (0.10, 0.08), (0.12, 0.10), (0.15, 0.12))
BAR_H = 0.12
DEBUG = False
LEGKW = {**rs.FRAMED_LEGEND_KW, "borderaxespad": 0.0}


def _bbox_in(art, fig, rend):
    b = art.get_window_extent(rend)
    d = fig.dpi
    return (b.x0 / d, b.y0 / d, b.x1 / d, b.y1 / d)


def _gather(fig, axes, cax_list, legends, ftexts):
    """蒐集所有文字（刻度標籤、軸標題、面板標題、色階標籤、圖例、圖上文字）與框線之範圍（英吋）。"""
    rend = fig.canvas.get_renderer()
    texts, frames = [], []
    for ai, ax in enumerate(axes + cax_list):
        is_cax = ai >= len(axes)
        frames.append((f"{'cax' if is_cax else 'ax'}{ai}", _bbox_in(ax, fig, rend)))
        for axis, nm in ((ax.xaxis, "x"), (ax.yaxis, "y")):
            lo, hi = sorted(axis.get_view_interval())
            for t in axis.get_major_ticks():
                if not (lo - 1e-9 <= t.get_loc() <= hi + 1e-9):
                    continue
                for lab in (t.label1, t.label2):
                    if lab.get_visible() and lab.get_text():
                        texts.append((f"{'cax' if is_cax else 'ax'}{ai}.{nm}tick[{lab.get_text()}]", _bbox_in(lab, fig, rend)))
            if axis.label.get_text() and axis.label.get_visible():
                texts.append((f"{'cax' if is_cax else 'ax'}{ai}.{nm}label", _bbox_in(axis.label, fig, rend)))
        if not is_cax and ax.title.get_text():
            texts.append((f"ax{ai}.title", _bbox_in(ax.title, fig, rend)))
    for i, lg in enumerate(legends):
        texts.append((f"legend{i}", _bbox_in(lg, fig, rend)))
    for i, t in enumerate(ftexts):
        texts.append((f"figtext{i}[{t.get_text()[:6]}]", _bbox_in(t, fig, rend)))
    return texts, frames


def _gap(a, b):
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return float(np.hypot(dx, dy)), (dx == 0.0 and dy == 0.0)


def collision_report(fig, texts, frames, sep_pt=1.0, frame_sep_pt=0.5):
    fw, fh = fig.get_size_inches()
    bad, min_sep = [], 9.0
    for (na, ba), (nb, bb) in itertools.combinations(texts, 2):
        g, ov = _gap(ba, bb)
        min_sep = min(min_sep, g)
        if ov or g < sep_pt * PT:
            bad.append(f"文字相碰：{na} × {nb}（{g / PT:.2f} pt）")
    for nt, bt in texts:
        for nf, bf in frames:
            g, ov = _gap(bt, bf)
            if ov or g < frame_sep_pt * PT:
                bad.append(f"文字壓框：{nt} × {nf}（{g / PT:.2f} pt）")
    for nt, bt in texts:
        if bt[0] < 0.005 or bt[1] < 0.005 or bt[2] > fw - 0.005 or bt[3] > fh - 0.005:
            bad.append(f"超出畫布：{nt}")
    return bad, min_sep / PT


def band_height(mode, st, legend_in_slots):
    if mode == "right":
        return st["XB"] + 0.05 + st["LEG_H"] + 0.015
    if mode == "slot":
        return st["XB"] + 0.015
    b = st["XB"] + 0.07 + BAR_H + st["BAR_TL"] + 0.015
    if not legend_in_slots:
        b += 0.07 + st["LEG_H"]
    return b


def geo2(nrow, ncol, n, th, H, mode, legend_in_slots, st, Lm, Rm, gx, gy, top, xlab_extra=0.20):
    row_gap = []
    for r in range(nrow - 1):
        need = any((r * ncol + c < n) and ((r + 1) * ncol + c >= n) for c in range(ncol))
        row_gap.append(gy + (xlab_extra if need else 0.0))
    band = band_height(mode, st, legend_in_slots)
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


def build(S, kind, name, nrow, ncol, H, mode, legend_in_slots, class_brackets, shared_titles, st, Lm, Rm, gx, gy, th,
          tick_lon=0.6, tick_lat=0.4, top=0.012):
    import matplotlib.pyplot as plt

    sd, dr, font = S.sd, S.dr, S.font
    n_pan = {"season": 4, "tide": 2, "class": len(S.mats)}[name]
    g = geo2(nrow, ncol, n_pan, th, H, mode, legend_in_slots, st, Lm, Rm, gx, gy, top)
    panels, lines, tlog = L0.make_panels(S, kind, name, g["w"], class_brackets=class_brackets)
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
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    d = fig.dpi
    last_bottom = row_top[-1] - g["th"] - g["h"]
    first_top = row_top[0] - g["th"]
    # 實測底部刻度（與每圖 x 軸標題）之下緣、左側刻度之左緣
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
    if mode == "right":
        cb = SF._discrete_colorbar(fig, (g["x0"] + g["block_w"] + rs.CBAR_PAD_IN, last_bottom, rs.CBAR_W_IN, first_top - last_bottom), f8)  # noqa: SLF001
        caxs.append(cb.ax)
        leg = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(block_cx / fw, (lowest - 0.05) / fh),
                         ncol=len(handles), prop=f8, **LEGKW)
        legends.append(leg)
    elif mode == "slot":
        cb_slot, lg_slot = spare[-1], spare[-2]
        sx = g["x0"] + (cb_slot % ncol) * (g["w"] + gx)
        cb = SF._discrete_colorbar(fig, (sx + 0.30, last_bottom, rs.CBAR_W_IN, g["h"]), f8)  # noqa: SLF001
        caxs.append(cb.ax)
        lx = g["x0"] + (lg_slot % ncol) * (g["w"] + gx) + g["w"] / 2.0
        leg = fig.legend(handles=handles, loc="center", bbox_to_anchor=(lx / fw, (last_bottom + g["h"] / 2) / fh), ncol=1,
                         prop=f8, **LEGKW)
        legends.append(leg)
    else:  # bottom
        import matplotlib.cm as mcm

        cmap, norm, cols = SF.colormap_and_norm()
        bar_top = lowest - 0.07
        bar_y = bar_top - BAR_H
        lab_t = fig.text(g["x0"] / fw, (bar_y + BAR_H / 2) / fh, SF.COLORBAR_LABEL, ha="left", va="center", fontproperties=f8,
                         fontsize=rs.TEXT_PT)
        ftexts.append(lab_t)
        lw_in = _bbox_in(lab_t, fig, rend)
        bx0 = lw_in[2] + 0.12
        bx1 = g["x0"] + g["block_w"]
        cax = fig.add_axes((bx0 / fw, bar_y / fh, (bx1 - bx0) / fw, BAR_H / fh))
        cbar = fig.colorbar(mcm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", ticks=np.arange(len(cols)) + 0.5)
        cbar.ax.set_xticklabels(sp.bin_labels(SF.EDGES))
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
        if legend_in_slots and spare:
            cxs = [g["x0"] + (s % ncol) * (g["w"] + gx) + g["w"] / 2.0 for s in spare if s // ncol == nrow - 1]
            leg = fig.legend(handles=handles, loc="center", bbox_to_anchor=(np.mean(cxs) / fw, (last_bottom + g["h"] / 2) / fh),
                             ncol=1, prop=f8, **LEGKW)
        else:
            leg = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(block_cx / fw, (tl_bottom - 0.07) / fh),
                             ncol=len(handles), prop=f8, **LEGKW)
        legends.append(leg)
    fig.canvas.draw()
    lb = _bbox_in(legends[0], fig, rend)
    meas["LEG_H"] = lb[3] - lb[1]
    texts, frames = _gather(fig, axes, caxs, legends, ftexts)
    minx = min(b[0] for _, b in texts)
    maxx = max(b[2] for _, b in texts)
    tit = [b[3] - b[1] for nme, b in texts if nme.endswith(".title")]
    meas.update(minx=minx, maxx=maxx, title_h=max(tit) + 2.0 * PT + 0.004, block_right=g["x0"] + g["block_w"], lines=lines)
    return fig, g, panels, lines, tlog, ann, texts, frames, legends, meas


def render_fit(S, kind, name, path: Path, *, nrow, ncol, H, mode, legend_in_slots=False, class_brackets=False,
               shared_titles=False, dpi=600, gaps=GAP_CANDS, tick_lon=0.6, tick_lat=0.4) -> dict:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    f = S.font
    n_pan = {"season": 4, "tide": 2, "class": len(S.mats)}[name]
    last = None
    with rs.report_rc(f.get_name()):
        rs.require_report_rc(f)
        for gx, gy in gaps:
            st = {"XB": 0.42, "LEG_H": 0.27, "BAR_TL": 0.15}
            Lm = 0.60
            Rm = 0.80 if mode == "right" else 0.02
            th = L0.title_h_for(2)
            res = None
            for it in range(7):
                if res is not None:
                    plt.close(res[0])
                res = build(S, kind, name, nrow, ncol, H, mode, legend_in_slots, class_brackets, shared_titles, st, Lm, Rm,
                            gx, gy, th, tick_lon=tick_lon, tick_lat=tick_lat)
                fig, g, panels, lines, tlog, ann, texts, frames, legends, meas = res
                new_st = {k: meas[k] for k in ("XB", "LEG_H", "BAR_TL") if k in meas}
                new_th = meas["title_h"]
                dL = (0.015 - meas["minx"]) if g["limit"] == "寬" else 0.0
                dR = 0.0
                if mode == "right":
                    need_R = meas["maxx"] - meas["block_right"] + 0.015
                    dR = need_R - Rm if g["limit"] == "寬" else 0.0
                conv = (abs(dL) < 0.004 and abs(dR) < 0.004 and abs(new_th - th) < 0.004
                        and all(abs(new_st[k] - st[k]) < 0.004 for k in new_st))
                if DEBUG:
                    print(f"  it{it} gx={gx} L={Lm:.3f} R={Rm:.3f} th={th:.3f}->{new_th:.3f} st={ {k: round(v,3) for k,v in st.items()} }->{ {k: round(v,3) for k,v in new_st.items()} } minx={meas['minx']:.3f} w={g['w']:.3f} fh={g['fh']:.3f}", flush=True)
                if conv:
                    break
                st.update(new_st)
                th = new_th
                Lm += dL
                Rm += dR
            bad, min_sep = collision_report(fig, texts, frames)
            if DEBUG:
                print(f" gap gx={gx} gy={gy}: {len(bad)} {bad[:4]}", flush=True)
            last = (g, bad, min_sep, gx, gy)
            if not bad or (gx, gy) == tuple(gaps[-1]):
                break
            plt.close(fig)
        g, bad, min_sep, gx, gy = last
        rec = {
            "kind": kind, "site": S.sd.site_id, "name": name, "layout_rows_cols": [nrow, ncol], "mode": mode,
            "legend_in_slots": legend_in_slots, "canvas_in": [float(W_IN), float(g["fh"])],
            "panel_w_h_in": [float(g["w"]), float(g["h"])], "limit": g["limit"], "title_h_in": g["th"],
            "title_max_lines": int(lines), "n_panels": len(panels), "gx_in": gx, "gy_in": gy,
            "L_in": g["L"], "R_in": g["R"], "x0_in": g["x0"], "tick_step_lon_lat": [tick_lon, tick_lat],
            "panel_titles": [p.title for p in panels], "annotations": ann, "class_brackets": class_brackets,
            "collisions": bad, "collision_count": len(bad), "min_text_sep_pt": min_sep, "title_choice": tlog,
            "legend_labels": [t.get_text() for t in legends[0].get_texts()],
            "lw_scale": L0.lw_for(g["w"]), "star_size": L0.star_for(g["w"]), "band_in": g["band"],
        }
        return SF._finish(fig, Path(path), rec, dpi)  # noqa: SLF001
