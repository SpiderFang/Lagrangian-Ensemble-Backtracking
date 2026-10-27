"""WP-75：邊界／海岸分段機率圖的繪製（報告版 8 pt 版面，沿用 report_style 與 map_axes）。

單張圖（全年）與多面板圖（季節 2×2、大小潮 1×2、十類 2×5）共用同一組繪製函式與同一組離散色階。
圖上不寫研究地點名稱（由圖說辨識）；單張圖的總比例寫在圖例，多面板的總比例寫在面板標題；
沒有等深線、大圓、抽樣範圍與格線。百分比一律由整數顆數以四捨五入（half-up）到小數 2 位（:func:`pct2`）。
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

import numpy as np

from lagrangian_backtracking.native_runner import map_axes as ma
from lagrangian_backtracking.native_runner import report_style as rs
from lagrangian_backtracking.native_runner import segprob as sp

EDGES = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
COLOR_ZERO = "#bdbdbd"
COLORS = ("#fed976", "#feb24c", "#fd8d3c", "#fc4e2a", "#e31a1c", "#bd0026", "#800026")
LAND_FACE = "#d9d9d9"  # 與條件式來源足跡（停留時數密度）圖相同
LAND_EDGE = "black"
LAND_EDGE_LW = 0.35  # 足跡圖 0.5 pt × 縮放 0.70
LAND_LINE = "#6e6e6e"
STAR = dict(
    marker="*",
    markersize=10.0,
    markerfacecolor="#ffd92f",
    markeredgecolor="black",
    markeredgewidth=0.6,
    linestyle="none",
)
LW_BOUNDARY = 3.2
LW_COAST = 2.4
LW_LAND_LINE = 0.7
COLORBAR_LABEL = "各段比例（% 虛擬質點）"

# 單張圖窗口：與軌跡總覽、條件式來源足跡圖（ax2）相同（跨度 1.8°×1.2°、刻度 0.2°、參考緯度 24.1°）
SPEC = ma.AxisSpec(step_deg=0.2, lon_span_deg=1.8, lat_span_deg=1.2, ref_lat_deg=24.1)
WINDOWS = {
    "gongliao": (121.2, 123.0, 24.4, 25.6),
    "guishan": (121.2, 123.0, 24.4, 25.6),
    "hsinchu": (119.6, 121.4, 24.2, 25.4),
    "nanwan": (120.0, 121.8, 21.4, 22.6),
    "lienchiang": (119.0, 120.8, 25.6, 26.8),
}
PANEL_TICK_LON = 0.6  # 多面板：經度刻度間隔（上下限仍落在刻度上）
PANEL_TICK_LAT = 0.4


def pct2(count: int, total: int) -> str:
    """整數顆數 → 百分比字串（四捨五入 half-up 到小數 2 位），例：31085/100000 → ``31.09``。"""

    v = (Decimal(int(count)) * 100 / Decimal(int(total))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return f"{v}"


def colormap_and_norm():
    from matplotlib.colors import BoundaryNorm, ListedColormap

    cols = (COLOR_ZERO, *COLORS)
    return ListedColormap(cols), BoundaryNorm(np.arange(len(cols) + 1), len(cols)), list(cols)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


@dataclass
class Drawing:
    """一個地點的繪製幾何（經緯度）：邊界海上段／陸上段、海岸線分段與窗口內陸地。"""

    b_sea: list[list[np.ndarray]]
    b_land: list[np.ndarray]
    c_lines: list[list[np.ndarray]]
    land_polys: list[object]
    has_land_boundary: bool
    land_union: object = None  # 窗口內陸地（lon/lat），供註記避開陸地


def _unproject_lines(sd, geom) -> list[np.ndarray]:
    out = []
    for ln in sp._lines_of(geom):  # noqa: SLF001
        c = np.asarray(ln.coords)
        lon, lat = sd.proj.unproject(c[:, 0], c[:, 1])
        out.append(np.column_stack([lon, lat]))
    return out


def prepare_drawing(sd, land_lonlat: Sequence[object]) -> Drawing:
    from shapely.geometry import box

    b_sea, b_land_parts = [], []
    for seg in sd.bseg:
        if sd.land_proj is None or sd.land_proj.is_empty:
            sea, lnd = seg.geom, None
        else:
            sea, lnd = seg.geom.difference(sd.land_proj), seg.geom.intersection(sd.land_proj)
        b_sea.append(_unproject_lines(sd, sea))
        if lnd is not None:
            b_land_parts.extend(_unproject_lines(sd, lnd))
    c_lines = [_unproject_lines(sd, s.geom) for s in sd.cseg]
    w = WINDOWS[sd.site_id]
    win = box(w[0] - 0.2, w[2] - 0.2, w[1] + 0.2, w[3] + 0.2)
    polys = []
    for g in land_lonlat:
        if g.intersects(win):
            polys.extend(sp._polys_of(g))  # noqa: SLF001
    import shapely

    return Drawing(
        b_sea, b_land_parts, c_lines, polys, bool(b_land_parts), shapely.union_all(polys) if polys else None
    )


def draw_map(
    ax,
    sd,
    dr: Drawing,
    kind: str,
    pct: np.ndarray,
    *,
    window: tuple[float, float, float, float],
    star: bool = True,
    lw_scale: float = 1.0,
    star_size: float | None = None,
) -> None:
    from matplotlib.collections import LineCollection
    from shapely.plotting import plot_polygon

    _cmap, _norm, cols = colormap_and_norm()
    for poly in dr.land_polys:
        plot_polygon(
            poly,
            ax=ax,
            add_points=False,
            facecolor=LAND_FACE,
            edgecolor=LAND_EDGE,
            linewidth=LAND_EDGE_LW * lw_scale,
            zorder=1,
        )
    cls = sp.bin_index(pct, EDGES)
    if kind == "boundary":
        if dr.b_land:
            ax.add_collection(
                LineCollection(
                    dr.b_land, colors=LAND_LINE, linewidths=LW_LAND_LINE * lw_scale, zorder=3, capstyle="butt"
                )
            )
        lw, lines = LW_BOUNDARY * lw_scale, dr.b_sea
    else:
        lw, lines = LW_COAST * lw_scale, dr.c_lines
    for k in range(len(cols)):  # 先畫 0，再依色階由低到高，高值疊在上面
        segs = [a for i, parts in enumerate(lines) if cls[i] == k for a in parts]
        if segs:
            ax.add_collection(
                LineCollection(segs, colors=cols[k], linewidths=lw, zorder=4 + k * 0.01, capstyle="butt")
            )
    if star:
        st = dict(STAR)
        if star_size is not None:
            st["markersize"] = star_size
        ax.plot([sd.anchor[0]], [sd.anchor[1]], zorder=9, **st)
    ax.set_xlim(window[0], window[1])
    ax.set_ylim(window[2], window[3])
    ax.set_aspect(SPEC.aspect, adjustable="box")


def _ticks(lo: float, hi: float, step: float) -> list[float]:
    n = int(round((hi - lo) / step))
    return [round(lo + k * step, 6) for k in range(n + 1)]


def _format_axes(
    ax,
    window,
    step_lon: float,
    step_lat: float,
    *,
    xlabels: bool,
    ylabels: bool,
    f8,
    titles: bool = True,
    rotate_x: bool = False,
) -> None:
    from matplotlib.ticker import FuncFormatter

    ax.set_xticks(_ticks(window[0], window[1], step_lon))
    ax.set_yticks(_ticks(window[2], window[3], step_lat))
    ax.xaxis.set_major_formatter(FuncFormatter(ma.format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(ma.format_lat))
    rs.style_map_axes(ax, f8)
    ax.tick_params(axis="x", labelbottom=xlabels)
    ax.tick_params(axis="y", labelleft=ylabels)
    if rotate_x:
        for lab in ax.get_xticklabels():
            lab.set_rotation(90)
            lab.set_va("top")
            lab.set_ha("center")
    if titles:
        if xlabels:
            ax.set_xlabel(
                ma.AXIS_TITLE_LON_ZH, fontproperties=f8, fontsize=rs.TEXT_PT, labelpad=rs.LABEL_PAD_PT
            )
        if ylabels:
            ax.set_ylabel(
                ma.AXIS_TITLE_LAT_ZH, fontproperties=f8, fontsize=rs.TEXT_PT, labelpad=rs.LABEL_PAD_PT
            )


def _discrete_colorbar(fig, box_in: tuple[float, float, float, float], f8):
    import matplotlib.cm as mcm

    cmap, norm, cols = colormap_and_norm()
    fw, fh = fig.get_size_inches()
    x, y, w, h = box_in
    cax = fig.add_axes((x / fw, y / fh, w / fw, h / fh))
    cbar = fig.colorbar(mcm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks=np.arange(len(cols)) + 0.5)
    cbar.ax.set_yticklabels(sp.bin_labels(EDGES))
    rs.style_colorbar(cbar, COLORBAR_LABEL, f8)
    cbar.ax.tick_params(length=0)
    return cbar


def _legend_handles(kind: str, total_text: str | None, has_land_boundary: bool) -> list[Any]:
    from matplotlib.lines import Line2D

    h = [Line2D([], [], label="研究地點錨點", **STAR)]
    if kind == "boundary" and has_land_boundary:
        h.append(Line2D([], [], color=LAND_LINE, linewidth=LW_LAND_LINE * 1.6, label="陸上邊界（無法穿越）"))
    if total_text:
        h.append(Line2D([], [], linestyle="none", label=total_text))
    return h


def _finish(fig, path: Path, record: dict[str, Any], dpi: int) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    record["text_audit"] = rs.text_audit(fig)
    record["text_summary"] = rs.audit_summary(record["text_audit"])
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    record["png"] = path.name
    record["sha256"] = sha256_file(path)
    record["size_bytes"] = path.stat().st_size
    return record


def _free_inset_box(sd, dr: Drawing, kind: str, pct: np.ndarray, window, w_frac: float, frame_w_h: float):
    """在主圖右下方找一個不蓋到任何上色海岸線的位置（回傳軸比例座標 ``(x0, y0, w, h)``，含刻度標籤需要的邊）；找不到回傳 None。"""  # noqa: E501

    lon_span, lat_span = window[1] - window[0], window[3] - window[2]
    h_frac = w_frac / frame_w_h
    colored = [a for i, parts in enumerate(dr.c_lines) if pct[i] > 0 for a in parts]
    pts = np.vstack(colored) if colored else np.zeros((0, 2))
    fx = (pts[:, 0] - window[0]) / lon_span
    fy = (pts[:, 1] - window[2]) / lat_span
    for x0 in np.arange(0.99 - w_frac, 0.5, -0.01):
        for y0 in np.arange(0.10, 0.5, 0.01):
            # 保留左側 0.11（y 刻度標籤）與下側 0.09（x 刻度標籤）
            rx0, rx1, ry0, ry1 = x0 - 0.11, x0 + w_frac + 0.005, y0 - 0.09, y0 + h_frac + 0.005
            if not ((fx >= rx0) & (fx <= rx1) & (fy >= ry0) & (fy <= ry1)).any():
                return (float(x0), float(y0), w_frac, h_frac)
    return None


def render_single(
    sd,
    dr: Drawing,
    kind: str,
    pct: np.ndarray,
    total_text: str,
    path: Path,
    font,
    *,
    dpi: int = 600,
    window: tuple[float, float, float, float] | None = None,
    inset_window: tuple[float, float, float, float] | None = None,
    inset_w_frac: float = 0.27,
    annotate=None,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path = Path(path)
    f8 = rs.text_font(font)
    with rs.report_rc(font.get_name()):
        rs.require_report_rc(font)
        legend_rows = 2
        fig = plt.figure(figsize=rs.canvas_in(legend_rows), dpi=150)
        fw, fh = fig.get_size_inches()
        bx, by, bw, bh = rs.frame_box_in(SPEC.frame_width_over_height, legend_rows)
        ax = fig.add_axes((bx / fw, by / fh, bw / fw, bh / fh))
        win = window or WINDOWS[sd.site_id]
        draw_map(ax, sd, dr, kind, pct, window=win)
        _format_axes(ax, win, SPEC.step_deg, SPEC.step_deg, xlabels=True, ylabels=True, f8=f8)
        inset_rec = None
        if inset_window is not None:
            pos = _free_inset_box(sd, dr, kind, pct, win, inset_w_frac, SPEC.frame_width_over_height)
            if pos is None:
                raise RuntimeError("找不到不蓋到上色海岸線的插圖位置")
            iax = ax.inset_axes(pos)
            draw_map(iax, sd, dr, kind, pct, window=inset_window, star=False, lw_scale=0.9)
            iw = inset_window
            iax.set_xticks([iw[0], iw[1]])
            iax.set_yticks([iw[2], iw[3]])
            from matplotlib.ticker import FuncFormatter

            iax.xaxis.set_major_formatter(FuncFormatter(ma.format_lon))
            iax.yaxis.set_major_formatter(FuncFormatter(ma.format_lat))
            rs.style_map_axes(iax, f8)
            for s in iax.spines.values():
                s.set_linewidth(0.8)
            iax.set_zorder(30)
            from matplotlib.patches import Rectangle

            ax.add_patch(
                Rectangle(
                    (iw[0], iw[2]),
                    iw[1] - iw[0],
                    iw[3] - iw[2],
                    fill=False,
                    edgecolor="#222222",
                    linewidth=0.8,
                    zorder=12,
                )
            )
            inset_rec = {"window_lon_lat": list(iw), "axes_fraction_box": list(pos)}
        ann_rec = annotate(ax, fig, f8) if annotate is not None else None
        _discrete_colorbar(fig, (bx + bw + rs.CBAR_PAD_IN, by, rs.CBAR_W_IN, bh), f8)
        handles = _legend_handles(kind, total_text, dr.has_land_boundary)
        leg = rs.framed_legend(fig, ax, handles, f8, rows=legend_rows)
        rec = {
            "kind": kind,
            "site": sd.site_id,
            "canvas_in": [float(fw), float(fh)],
            "frame_box_in_left_bottom_w_h": [bx, by, bw, bh],
            "window_lon_lat": list(win),
            "legend": rs.legend_geometry(fig, leg, ax),
            "legend_labels": [t.get_text() for t in leg.get_texts()],
            "total_text": total_text,
            "inset": inset_rec,
            "annotations": ann_rec,
        }
        return _finish(fig, path, rec, dpi)


@dataclass
class Panel:
    title: str
    pct: np.ndarray
    annotate: Any = None  # 可選：``f(ax, fig, f8) -> 紀錄``，在面板畫完後加註記


@dataclass(frozen=True)
class PanelLayout:
    nrow: int
    ncol: int
    left: float = 0.70
    right_strip: float = 0.95
    gap_x: float = 0.40
    title_h: float = 0.17
    gap_y: float = 0.10
    top: float = 0.08
    bottom_axes: float = 0.45
    legend_band: float = 0.45
    lw_scale: float = 0.85
    star_size: float = 9.0
    rotate_x: bool = False
    shared_axis_titles: bool = False


LAYOUT_2x2 = PanelLayout(2, 2, title_h=0.30)
LAYOUT_1x2 = PanelLayout(1, 2, title_h=0.30)
# 2×5：面板很小，內側不留刻度標籤間距；經度刻度標籤旋轉 90°避免相鄰面板的端點標籤相撞；軸標題改為整張圖共用一組。  # noqa: E501
LAYOUT_2x5 = PanelLayout(
    2, 5, left=0.72, right_strip=0.82, gap_x=0.22, title_h=0.30, gap_y=0.10, bottom_axes=0.62, legend_band=0.40,  # noqa: E501
    lw_scale=0.5, star_size=6.0, rotate_x=True, shared_axis_titles=True,
)  # fmt: skip


def render_panels(
    sd,
    dr: Drawing,
    kind: str,
    panels: Sequence[Panel],
    layout: PanelLayout,
    path: Path,
    font,
    *,
    dpi: int = 600,
    window: tuple[float, float, float, float] | None = None,
) -> dict[str, Any]:
    """多面板：共用一條色階；只在最外側面板標刻度；每個面板標題寫組別與該組總比例。"""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    L = layout
    nrow, ncol = L.nrow, L.ncol
    path = Path(path)
    f8 = rs.text_font(font)
    w = (rs.REPORT_WIDTH_IN - L.left - L.right_strip - L.gap_x * (ncol - 1)) / ncol
    h = w / SPEC.frame_width_over_height
    fh = L.top + nrow * (L.title_h + h) + (nrow - 1) * L.gap_y + L.bottom_axes + L.legend_band
    win = window or WINDOWS[sd.site_id]
    with rs.report_rc(font.get_name()):
        rs.require_report_rc(font)
        fig = plt.figure(figsize=(rs.REPORT_WIDTH_IN, fh), dpi=150)
        fw = rs.REPORT_WIDTH_IN
        axes = []
        ann_recs: list = []
        for i, p in enumerate(panels):
            r, c = divmod(i, ncol)
            x0 = L.left + c * (w + L.gap_x)
            y0 = fh - L.top - (r + 1) * (L.title_h + h) - r * L.gap_y
            ax = fig.add_axes((x0 / fw, y0 / fh, w / fw, h / fh))
            draw_map(ax, sd, dr, kind, p.pct, window=win, lw_scale=L.lw_scale, star_size=L.star_size)
            last_row = (r == nrow - 1) or (i + ncol >= len(panels))
            _format_axes(
                ax, win, PANEL_TICK_LON, PANEL_TICK_LAT, xlabels=last_row, ylabels=(c == 0), f8=f8,
                titles=not L.shared_axis_titles, rotate_x=L.rotate_x,
            )  # fmt: skip
            ax.set_title(p.title, fontproperties=f8, fontsize=rs.TEXT_PT, pad=2.0, linespacing=1.0)
            axes.append(ax)
            if p.annotate is not None:
                ann_recs.append(p.annotate(ax, fig, f8))
        block_w = ncol * w + (ncol - 1) * L.gap_x
        cb_top = fh - L.top - L.title_h
        cb_bottom = cb_top - (nrow * h + (nrow - 1) * (L.gap_y + L.title_h))
        _discrete_colorbar(
            fig, (L.left + block_w + rs.CBAR_PAD_IN, cb_bottom, rs.CBAR_W_IN, cb_top - cb_bottom), f8
        )
        if L.shared_axis_titles:
            fig.text((L.left + block_w / 2) / fw, (L.legend_band + 0.06) / fh, ma.AXIS_TITLE_LON_ZH, ha="center", va="bottom",  # noqa: E501
                     fontproperties=f8, fontsize=rs.TEXT_PT)  # fmt: skip
            fig.text(0.10 / fw, (cb_bottom + cb_top) / 2 / fh, ma.AXIS_TITLE_LAT_ZH, ha="center", va="center", rotation=90,  # noqa: E501
                     fontproperties=f8, fontsize=rs.TEXT_PT)  # fmt: skip
        handles = _legend_handles(kind, None, dr.has_land_boundary)
        block_cx = (L.left + block_w / 2.0) / fw
        top_in = (
            (L.legend_band - 0.02)
            if L.shared_axis_titles
            else rs.axes_bottom_in(fig, axes[(nrow - 1) * ncol]) - rs.LEGEND_GAP_IN
        )
        leg = fig.legend(
            handles=handles, loc="upper center", bbox_to_anchor=(block_cx, top_in / fh), ncol=len(handles), prop=f8,  # noqa: E501
            **rs.FRAMED_LEGEND_KW,
        )  # fmt: skip
        rec = {
            "kind": kind, "site": sd.site_id, "canvas_in": [float(fw), float(fh)], "panel_w_h_in": [w, h],
            "n_panels": len(panels), "layout": [nrow, ncol], "window_lon_lat": list(win),
            "ticks_step_lon_lat": [PANEL_TICK_LON, PANEL_TICK_LAT], "panel_titles": [p.title for p in panels],
            "legend_labels": [t.get_text() for t in leg.get_texts()], "annotations": ann_recs,
        }  # fmt: skip
        return _finish(fig, path, rec, dpi)
