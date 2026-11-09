#!/usr/bin/env python3
"""仍在追蹤比例曲線（報告版 fs8）——選項 B2：不畫「全部類別」；十類以 10 種色相（色盲友善，不含紅）、10 種虛線型、線寬 1.0 pt 區分；圖例依表 1 之類別次序。
只改曲線與圖例；座標軸（0–180 天、y 0–100%）、90／180 天虛線、8 pt 字級、畫布與圖框尺寸全同 domain_fraction.render_site_figure(report_fs8=True)。
用法：render_B2.py <postprocess_dir> <輸出目錄> [--sites s1,s2] [--markers] [--zoom 0,20]"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

from lagrangian_backtracking.native_runner import domain_fraction as D
from lagrangian_backtracking.native_runner import report_style as rs
from lagrangian_backtracking.native_runner.display_names import material_display_name_zh
from lagrangian_backtracking.native_runner.figures import _Labeler, _resolve_font

LW = 1.0             # 十類，印刷線寬（pt）
HANDLE_LEN = 2.8     # 圖例線段長（字級倍數；原 1.6）；8 pt × 2.8 = 22.4 pt
# 表 1 之類別次序（圖例、顏色、線型皆依此次序）
TABLE1 = ["oca_nonrecyclable_flexible_sheet", "oca_wastepaper_folded_fiber", "oca_styrofoam_porous_fragment",
          "oca_fishinggear_open_mesh_bundle", "oca_pet_waterfilled_bottle", "oca_wood_waterlogged_elongated",
          "oca_other_recyclable_irregular_fragment", "oca_aluminum_crushed_cylinder", "oca_steel_rigid_cylinder",
          "oca_glass_bottle_or_fragment"]
# 色盲友善定性色盤：Okabe-Ito（去紅橘紅 D55E00、去黃 F0E442 改深黃）＋ 深紫、棕、深灰、黑；避開與紅色 90 天輔助線（#B22222）相近之紅
COLORS = ["#0072B2",  # 藍
          "#E69F00",  # 橙
          "#009E73",  # 藍綠
          "#CC79A7",  # 紫紅（粉紫）
          "#56B4E9",  # 天藍
          "#7A4A21",  # 棕
          "#6A3D9A",  # 深紫
          "#B8A800",  # 深黃
          "#000000",  # 黑
          "#8C8C8C"]  # 灰
# 虛線型（pt；on/off 交替；None＝實線）。週期整除圖例線段長 22.4 pt，使每種線型在圖例中完整呈現
PATTERNS = [None,                      # 實線
            (8.6, 2.6),                # 長虛線
            (3.0, 1.48),               # 短虛線
            (1.1, 2.63),               # 點線
            (5.2, 1.8, 1.2, 3.0),      # 點劃線
            (4.0, 1.6, 1.0, 1.6, 1.0, 2.0),   # 雙點劃線
            (0.8, 1.44),               # 密點線
            (14.0, 2.0, 2.0, 4.4),     # 長劃－點
            (5.2, 2.27),               # 中虛線
            (6.8, 1.6, 1.6, 1.2)]      # 劃－短劃
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "*"]
MSIZE = 3.0


def render(curves, path: Path, *, font, t, markers: bool, zoom):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    rs.require_report_rc(font)
    fig = plt.figure(figsize=rs.canvas_in(D.CURVE_FS8_LEGEND_ROWS), dpi=150)
    fw, fh = fig.get_size_inches()
    bx, by, bw, bh = rs.frame_box_in(D.CURVE_FS8_FRAME_W_OVER_H, D.CURVE_FS8_LEGEND_ROWS)
    ax = fig.add_axes((bx / fw, by / fh, bw / fw, bh / fh))
    grid = curves.grid_days
    ids = list(curves.class_ids)
    assert sorted(ids) == sorted(TABLE1), ids
    cur_by = dict(zip(ids, curves.class_curves, strict=True))
    lines = []
    for i, cid in enumerate(TABLE1):
        kw = {}
        if markers:   # 稀疏標記：沿曲線每隔固定視覺距離一個（座標軸對角線比例），各類起點錯開
            kw = dict(marker=MARKERS[i], markersize=MSIZE * (1.35 if MARKERS[i] == "*" else 1.0), markevery=(0.012 + 0.0085 * i, 0.085),
                      markerfacecolor=COLORS[i], markeredgecolor=COLORS[i], markeredgewidth=0.0)
        ln, = ax.plot(grid, 100.0 * cur_by[cid], color=COLORS[i], linewidth=LW, label=material_display_name_zh(cid), zorder=3 + i * 0.01, **kw)
        lines.append(ln)
    top = float(grid[-1])
    for d in (90.0, 180.0):
        ax.axvline(d, color="#B22222", linewidth=1.0, linestyle=(0, (5, 3)), zorder=2)
    ax.set_xlim(0.0, top)
    ax.set_ylim(0.0, 100.0)
    ax.set_yticks(np.arange(0.0, 100.0 + 1e-9, 20.0))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.0f}"))
    ax.set_xticks(np.arange(0.0, top + 1e-9, 30.0))
    ax.set_xlabel(t.t("回溯時間（天）", "Backtrack time (days)"))
    ax.set_ylabel(t.t("回溯期間仍在追蹤之虛擬質點比例（%）", "Fraction of particles still tracked during backtracking (%)"))
    f8 = rs.text_font(font)
    rs.scale_print_effect(ax.get_children(), rs.PRINT_SCALE_CURVE)   # 同舊版：90／180 天虛線等依印刷比例縮放
    for ln, pat in zip(lines, PATTERNS):   # 曲線：直接設定印刷線寬與絕對虛線長度（pt）
        ln.set_linewidth(LW)
        ln.set_linestyle("-" if pat is None else (0, tuple(p / LW for p in pat)))
        ln.set_solid_capstyle("butt"); ln.set_dash_capstyle("butt")
    rs.style_map_axes(ax, f8)
    rs.set_axis_titles(ax, ax.get_xlabel(), ax.get_ylabel(), f8)
    fp = D._lines_fingerprint(ax) if hasattr(D, "_lines_fingerprint") else None
    if zoom:   # 僅供檢視之局部放大（非報告圖）：改 x 範圍與刻度，其餘不變
        ax.set_xlim(*zoom); ax.set_xticks(np.arange(zoom[0], zoom[1] + 1e-9, 2.0))
    handles, labels = ax.get_legend_handles_labels()
    kw = dict(rs.FRAMED_LEGEND_KW); kw["handlelength"] = HANDLE_LEN
    pos = ax.get_position()
    cx = (pos.x0 + pos.x1) / 2.0
    topy = rs.axes_bottom_in(fig, ax) - rs.LEGEND_GAP_IN
    leg = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(cx, topy / fh),
                     ncol=rs.ncol_for_rows(len(handles), D.CURVE_FS8_LEGEND_ROWS), prop=f8, **kw)
    fig.canvas.draw()
    rec = {"canvas_in": [fw, fh], "legend": rs.legend_geometry(fig, leg, ax), "legend_labels": [x.get_text() for x in leg.get_texts()],
           "text_summary": rs.audit_summary(rs.text_audit(fig)), "markers": markers, "zoom": zoom,
           "lw_pt": LW, "colors": COLORS, "patterns_pt": [list(p) if p else None for p in PATTERNS],
           "data_fingerprint": fp}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as s:
        fig.savefig(s, format="png", dpi=600)
    if not zoom:
        with path.with_suffix(".svg").open("xb") as s:
            fig.savefig(s, format="svg")
    plt.close(fig)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("postprocess_dir", type=Path); ap.add_argument("out", type=Path)
    ap.add_argument("--sites", default="gongliao,guishan,hsinchu,nanwan,lienchiang")
    ap.add_argument("--markers", action="store_true")
    ap.add_argument("--zoom", default=None)
    ap.add_argument("--name", default="domain_fraction_vs_backtrack_time")
    a = ap.parse_args()
    zoom = tuple(float(x) for x in a.zoom.split(",")) if a.zoom else None
    curves = D.load_site_curves(a.postprocess_dir / "departures.parquet", max_days=180.0, sites=a.sites.split(","))
    fr = _resolve_font(None)
    t = _Labeler(fr.cjk_available)
    recs = {}
    for sid, c in curves.items():
        with rs.report_rc(fr.font_properties.get_name()):
            recs[sid] = render(c, a.out / sid / (a.name + ".png"), font=fr.font_properties, t=t, markers=a.markers, zoom=zoom)
        print(sid, "ok", recs[sid]["legend"], flush=True)
    (a.out / ("record_" + a.name + ".json")).write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
