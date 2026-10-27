"""WP-75：邊界／海岸分段機率圖的繪製（報告版 8 pt 版面，沿用 report_style 與 map_axes）。

單張圖（全年）與多面板圖（季節 2×2、大小潮 1×2、十類、速度序位）共用同一組繪製函式與同一組離散色階。
圖上不寫研究地點名稱（由圖說辨識）；總比例寫在圖例；沒有等深線、大圓、抽樣範圍與格線。
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lagrangian_backtracking.native_runner import map_axes as ma
from lagrangian_backtracking.native_runner import report_style as rs
from lagrangian_backtracking.native_runner import segprob as sp

EDGES = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
COLOR_ZERO = "#d0d0d0"
COLORS = ("#fed976", "#feb24c", "#fd8d3c", "#fc4e2a", "#e31a1c", "#bd0026", "#800026")
LAND_FACE = "#ececec"
LAND_EDGE = "#9a9a9a"
LAND_LINE = "#8c8c8c"
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
    return Drawing(b_sea, b_land_parts, c_lines, polys, bool(b_land_parts))


def draw_map(
    ax,
    sd,
    dr: Drawing,
    kind: str,
    pct: np.ndarray,
    *,
    window: tuple[float, float, float, float],
    star: bool = True,
) -> None:
    from matplotlib.collections import LineCollection
    from shapely.plotting import plot_polygon

    cmap, _norm, cols = colormap_and_norm()
    for poly in dr.land_polys:
        plot_polygon(
            poly, ax=ax, add_points=False, facecolor=LAND_FACE, edgecolor=LAND_EDGE, linewidth=0.3, zorder=1
        )
    cls = sp.bin_index(pct, EDGES)
    if kind == "boundary":
        if dr.b_land:
            ax.add_collection(
                LineCollection(
                    dr.b_land, colors=LAND_LINE, linewidths=LW_LAND_LINE, zorder=3, capstyle="butt"
                )
            )
        # 先畫 0，再依色階由低到高，高值疊在上面
        for k in range(len(cols)):
            segs = [a for i, parts in enumerate(dr.b_sea) if cls[i] == k for a in parts]
            if segs:
                ax.add_collection(
                    LineCollection(
                        segs, colors=cols[k], linewidths=LW_BOUNDARY, zorder=4 + k * 0.01, capstyle="butt"
                    )
                )
    else:
        for k in range(len(cols)):
            segs = [a for i, parts in enumerate(dr.c_lines) if cls[i] == k for a in parts]
            if segs:
                ax.add_collection(
                    LineCollection(
                        segs, colors=cols[k], linewidths=LW_COAST, zorder=4 + k * 0.01, capstyle="butt"
                    )
                )
    if star:
        ax.plot([sd.anchor[0]], [sd.anchor[1]], zorder=9, **STAR)
    ax.set_xlim(window[0], window[1])
    ax.set_ylim(window[2], window[3])
    ax.set_aspect(SPEC.aspect, adjustable="box")


def _ticks(lo: float, hi: float, step: float) -> list[float]:
    n = int(round((hi - lo) / step))
    return [round(lo + k * step, 6) for k in range(n + 1)]


def _format_axes(ax, window, step_lon: float, step_lat: float, *, xlabels: bool, ylabels: bool, f8) -> None:
    from matplotlib.ticker import FuncFormatter

    ax.set_xticks(_ticks(window[0], window[1], step_lon))
    ax.set_yticks(_ticks(window[2], window[3], step_lat))
    ax.xaxis.set_major_formatter(FuncFormatter(ma.format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(ma.format_lat))
    rs.style_map_axes(ax, f8)
    ax.tick_params(axis="x", labelbottom=xlabels)
    ax.tick_params(axis="y", labelleft=ylabels)
    if xlabels:
        ax.set_xlabel(ma.AXIS_TITLE_LON_ZH, fontproperties=f8, fontsize=rs.TEXT_PT, labelpad=rs.LABEL_PAD_PT)
    if ylabels:
        ax.set_ylabel(ma.AXIS_TITLE_LAT_ZH, fontproperties=f8, fontsize=rs.TEXT_PT, labelpad=rs.LABEL_PAD_PT)


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


def _legend_handles(kind: str, total_text: str, has_land_boundary: bool) -> list[Any]:
    from matplotlib.lines import Line2D

    h = [Line2D([], [], label="研究地點錨點", **STAR)]
    if kind == "boundary" and has_land_boundary:
        h.append(Line2D([], [], color=LAND_LINE, linewidth=LW_LAND_LINE * 1.6, label="陸上邊界（無法穿越）"))
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


def render_single(
    sd, dr: Drawing, kind: str, pct: np.ndarray, total_pct: float, path: Path, font, *, dpi: int = 600
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
        win = WINDOWS[sd.site_id]
        draw_map(ax, sd, dr, kind, pct, window=win)
        _format_axes(ax, win, SPEC.step_deg, SPEC.step_deg, xlabels=True, ylabels=True, f8=f8)
        _discrete_colorbar(fig, (bx + bw + rs.CBAR_PAD_IN, by, rs.CBAR_W_IN, bh), f8)
        label = "穿越比例" if kind == "boundary" else "海岸來源比例"
        handles = _legend_handles(kind, f"{label} {total_pct:.2f}%", dr.has_land_boundary)
        leg = rs.framed_legend(fig, ax, handles, f8, rows=legend_rows)
        rec = {
            "kind": kind,
            "site": sd.site_id,
            "canvas_in": [float(fw), float(fh)],
            "frame_box_in_left_bottom_w_h": [bx, by, bw, bh],
            "window_lon_lat": list(win),
            "legend": rs.legend_geometry(fig, leg, ax),
            "legend_labels": [t.get_text() for t in leg.get_texts()],
            "total_pct": float(total_pct),
        }
        return _finish(fig, path, rec, dpi)


@dataclass
class Panel:
    title: str
    pct: np.ndarray
    total_pct: float


def render_panels(
    sd,
    dr: Drawing,
    kind: str,
    panels: Sequence[Panel],
    nrow: int,
    ncol: int,
    path: Path,
    font,
    *,
    dpi: int = 600,
) -> dict[str, Any]:
    """多面板：共用一條色階；只在最外側面板標刻度；每個面板標題寫組別與該組總比例。"""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path = Path(path)
    f8 = rs.text_font(font)
    left, right_strip, gap_x, title_h, gap_y, top, bottom_axes, legend_band = (
        0.70,
        0.95,
        0.40,
        0.17,
        0.10,
        0.08,
        0.45,
        0.45,
    )
    w = (rs.REPORT_WIDTH_IN - left - right_strip - gap_x * (ncol - 1)) / ncol
    h = w / SPEC.frame_width_over_height
    fh = top + nrow * (title_h + h) + (nrow - 1) * gap_y + bottom_axes + legend_band
    win = WINDOWS[sd.site_id]
    with rs.report_rc(font.get_name()):
        rs.require_report_rc(font)
        fig = plt.figure(figsize=(rs.REPORT_WIDTH_IN, fh), dpi=150)
        fw = rs.REPORT_WIDTH_IN
        axes = []
        for i, p in enumerate(panels):
            r, c = divmod(i, ncol)
            x0 = left + c * (w + gap_x)
            y0 = fh - top - (r + 1) * (title_h + h) - r * gap_y
            ax = fig.add_axes((x0 / fw, y0 / fh, w / fw, h / fh))
            draw_map(ax, sd, dr, kind, p.pct, window=win)
            last_row = (r == nrow - 1) or (i + ncol >= len(panels))
            _format_axes(ax, win, PANEL_TICK_LON, PANEL_TICK_LAT, xlabels=last_row, ylabels=(c == 0), f8=f8)
            ax.set_title(p.title, fontproperties=f8, fontsize=rs.TEXT_PT, pad=2.0)
            axes.append(ax)
        cb_top = fh - top - title_h
        cb_bottom = cb_top - (nrow * h + (nrow - 1) * (gap_y + title_h))
        _discrete_colorbar(
            fig,
            (
                left + ncol * w + (ncol - 1) * gap_x + rs.CBAR_PAD_IN,
                cb_bottom,
                rs.CBAR_W_IN,
                cb_top - cb_bottom,
            ),
            f8,
        )
        handles = _legend_handles(kind, "各面板標題為該組總比例", dr.has_land_boundary)[:-1]
        # 圖例置中於整個面板區塊正下方（頂端在最下列座標軸標題下方）
        block_cx = (left + (ncol * w + (ncol - 1) * gap_x) / 2.0) / fw
        top_in = rs.axes_bottom_in(fig, axes[(nrow - 1) * ncol]) - rs.LEGEND_GAP_IN
        leg = fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(block_cx, top_in / fh),
            ncol=len(handles),
            prop=f8,
            **rs.FRAMED_LEGEND_KW,
        )
        rec = {
            "kind": kind,
            "site": sd.site_id,
            "canvas_in": [float(fw), float(fh)],
            "panel_w_h_in": [w, h],
            "n_panels": len(panels),
            "layout": [nrow, ncol],
            "window_lon_lat": list(win),
            "ticks_step_lon_lat": [PANEL_TICK_LON, PANEL_TICK_LAT],
            "panel_titles": [p.title for p in panels],
            "legend_labels": [t.get_text() for t in leg.get_texts()],
        }
        return _finish(fig, path, rec, dpi)


def render_thumbs(
    items: Sequence[tuple[Any, Drawing, np.ndarray, np.ndarray, str, float, float]],
    path: Path,
    font,
    *,
    dpi: int = 200,
) -> dict[str, Any]:
    """五地點全年邊界（上列）與海岸（下列）並排縮圖，只供確認色階，不是報告用圖。"""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    f6 = font.copy()
    f6.set_size(6.5)
    n = len(items)
    pw = 2.0
    ph = pw / SPEC.frame_width_over_height
    left, right, top, bottom, gx, gy, title_h = 0.55, 0.9, 0.1, 0.45, 0.22, 0.35, 0.17
    fw = left + n * pw + (n - 1) * gx + right
    fh = top + 2 * (title_h + ph) + gy + bottom
    with rs.report_rc(font.get_name()):
        fig = plt.figure(figsize=(fw, fh), dpi=150)
        for j, (sd, dr, pb, pc, name, tb, tc) in enumerate(items):
            for r, (kind, pct, tot, lab) in enumerate(
                (("boundary", pb, tb, "穿越"), ("coast", pc, tc, "海岸來源"))
            ):
                x0 = left + j * (pw + gx)
                y0 = fh - top - (r + 1) * (title_h + ph) - r * gy
                ax = fig.add_axes((x0 / fw, y0 / fh, pw / fw, ph / fh))
                win = WINDOWS[sd.site_id]
                draw_map(ax, sd, dr, kind, pct, window=win)
                ax.set_xticks(_ticks(win[0], win[1], PANEL_TICK_LON))
                ax.set_yticks(_ticks(win[2], win[3], PANEL_TICK_LAT))
                from matplotlib.ticker import FuncFormatter

                ax.xaxis.set_major_formatter(FuncFormatter(ma.format_lon))
                ax.yaxis.set_major_formatter(FuncFormatter(ma.format_lat))
                ax.tick_params(labelsize=6.5, length=2.5, width=0.6, pad=1.5)
                for s in ax.spines.values():
                    s.set_linewidth(0.8)
                ax.set_title(f"{name}｜{lab} {tot:.2f}%", fontproperties=f6, fontsize=6.5, pad=2.0)
        cax = fig.add_axes(((fw - right + 0.15) / fw, bottom / fh, 0.12 / fw, (fh - top - bottom) / fh))
        import matplotlib.cm as mcm

        cmap, norm, cols = colormap_and_norm()
        cbar = fig.colorbar(
            mcm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks=np.arange(len(cols)) + 0.5
        )
        cbar.ax.set_yticklabels(sp.bin_labels(EDGES), fontproperties=f6, fontsize=6.5)
        cbar.ax.tick_params(length=0)
        cbar.set_label(COLORBAR_LABEL, fontproperties=f6, fontsize=6.5)
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
    return {
        "png": Path(path).name,
        "sha256": sha256_file(path),
        "size_bytes": Path(path).stat().st_size,
        "canvas_in": [fw, fh],
    }
