# ruff: noqa: E501
"""WP-75 試作：加上「幾何期望」虛線與新主要來源方向標示的圓形來源方位圖（報告 fs8 版面）。

沿用 :mod:`polar_source_figures` 報告版面（畫布寬 6.27 英寸、8 pt、Noto Sans CJK TC、``one_decimal`` 座標軸、
正方形主圖框、圖例在主圖正下方），差別：不畫等深線；多畫（1）期望比例虛線多邊形、（2）新規則的主要來源方向
（觀測高於期望的連續扇區，取觀測合計最大的前兩個範圍）半透明扇形。折線、同心圓與底圖完全重用既有函式。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import polar_expected as pe
from . import polar_source_figures as psf

EXPECTED_LABEL = "沿邊界及海岸均勻時之期望比例"
HIGHLIGHT_LABEL = "主要來源方向（觀測高於期望的連續方向，取前兩大）"
HIGHLIGHT_COLOR = "#f0b429"
HIGHLIGHT_ALPHA = 0.32
EXPECTED_COLOR = "#111111"
SINGLE_COLOR = "#0072B2"


@dataclass(slots=True)
class Line:
    label: str
    color: Any
    obs_pct: np.ndarray  # (36,)


@dataclass(slots=True)
class Panel:
    title: str
    lines: list[Line]
    exp_pct: np.ndarray | None  # (36,) 或 None（不畫期望）
    runs: list[list[int]]  # 要標示的範圍（空＝不標示）


def pick_scale(peak_pct: float) -> tuple[float, list[float]]:
    top, ticks = psf._nice_ceiling_axis(peak_pct * 1.05, max_intervals=6)  # noqa: SLF001
    return float(top), [float(v) for v in ticks if v > 0]


def _plan(panel: Panel, top: float, ticks: Sequence[float]) -> psf.PolarFigurePlan:
    return psf.PolarFigurePlan(
        kind="X", stem="x", title_zh="", title_en="", subtitle_zh="", subtitle_en="", legend_title_zh="", legend_title_en="",
        labels=[ln.label for ln in panel.lines], colors=[ln.color for ln in panel.lines], styles=["-"] * len(panel.lines),
        pct=np.vstack([ln.obs_pct for ln in panel.lines]), n_totals=[0] * len(panel.lines), radius_max_pct=top,
        radius_ticks_pct=list(ticks), caption_zh="",
    )  # fmt: skip


def _add_overlays(ax: Any, site_map: Any, panel: Panel, top: float, compact: bool) -> None:
    import matplotlib.patheffects as pe_
    from matplotlib.patches import Polygon as MplPolygon

    radius = site_map.radius_km
    for run in panel.runs:
        a0 = 10.0 * run[0] - 5.0
        a1 = 10.0 * run[-1] + 5.0
        if a1 < a0:
            a1 += 360.0
        az = np.linspace(a0, a1, max(8, int((a1 - a0) / 2) + 2))
        lo, la = psf.polar_xy_to_lonlat(site_map.anchor, np.full(az.shape, radius), az)
        lo0, la0 = psf.polar_xy_to_lonlat(site_map.anchor, np.array([0.0]), np.array([0.0]))
        xy = np.vstack([[lo0[0], la0[0]], np.column_stack([lo, la])])
        ax.add_patch(
            MplPolygon(
                xy, closed=True, facecolor=HIGHLIGHT_COLOR, edgecolor="none", alpha=HIGHLIGHT_ALPHA, zorder=3
            )
        )
    if panel.exp_pct is not None:
        centers = np.append(10.0 * np.arange(pe.N), 0.0)
        vals = np.append(panel.exp_pct, panel.exp_pct[0])
        lo, la = psf.polar_xy_to_lonlat(site_map.anchor, radius * vals / top, centers)
        ax.plot(lo, la, color=EXPECTED_COLOR, linestyle=(0, (4, 2)), linewidth=(0.9 if compact else 1.1), zorder=5.5,
                path_effects=[pe_.withStroke(linewidth=(1.8 if compact else 2.2), foreground="white", alpha=0.6)])  # fmt: skip


def _extra_handles(panel_has_exp: bool, panel_has_runs: bool) -> list[Any]:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    out: list[Any] = []
    if panel_has_exp:
        out.append(
            Line2D([], [], color=EXPECTED_COLOR, linestyle=(0, (4, 2)), linewidth=1.4, label=EXPECTED_LABEL)
        )
    if panel_has_runs:
        out.append(
            Patch(
                facecolor=HIGHLIGHT_COLOR,
                edgecolor="none",
                alpha=HIGHLIGHT_ALPHA + 0.1,
                label=HIGHLIGHT_LABEL,
            )
        )
    return out


def _base_handles(t: Any) -> list[Any]:
    return psf._report_base_handles(t, [], [])  # noqa: SLF001


def render_single(panel: Panel, anchor: tuple[float, float], land: Sequence[Any], top: float, ticks: Sequence[float], font: Any,
                  cjk: bool, path: Path, dpi: int = 300, legend_h: float = 0.62) -> None:  # fmt: skip
    """單張（全年）：一條觀測折線＋期望虛線＋標示範圍；圖例在主圖正下方。"""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from lagrangian_backtracking.native_runner.figures import _Labeler

    t = _Labeler(cjk)
    site_map = psf._build_report_map(anchor, 25.0, land, {})  # noqa: SLF001
    w = psf.REPORT_W_IN
    left, frame, xlab = psf._RO_LEFT_1D, psf._RO_FRAME_1D, psf._RO_XLAB_1D  # noqa: SLF001
    h = round(psf._RO_TOP + frame + xlab + psf._RO_LEGEND_GAP + legend_h + psf._RO_BOTTOM, 3)  # noqa: SLF001
    fig = plt.figure(figsize=(w, h), dpi=dpi)
    ax = fig.add_axes([left / w, 1.0 - (psf._RO_TOP + frame) / h, frame / w, frame / h])  # noqa: SLF001
    xt = psf._report_ticks(site_map.extent[0], site_map.extent[1], (0.1,), 5, inner=(0.055, 0.945))  # noqa: SLF001
    yt = psf._report_ticks(site_map.extent[2], site_map.extent[3], (0.1,), 5, inner=(0.03, 0.97))  # noqa: SLF001
    handles, _, _ = psf._draw_report_polar(ax, _plan(panel, top, ticks), site_map, font, compact=False, xticks=xt, yticks=yt,  # noqa: SLF001
                                           label_x=True, label_y=True, one_decimal=True)  # fmt: skip
    _add_overlays(ax, site_map, panel, top, compact=False)
    y_leg = psf._RO_TOP + frame + xlab + psf._RO_LEGEND_GAP  # noqa: SLF001
    leg1 = psf._fig_legend(fig, handles + _extra_handles(panel.exp_pct is not None, bool(panel.runs)), font, loc="upper left",  # noqa: SLF001
                           x_in=left, y_top_in=y_leg, h_in=h, w_in=w, title=None, ncol=1)  # fmt: skip
    leg2 = psf._fig_legend(fig, _base_handles(t), font, loc="upper right", x_in=left + frame, y_top_in=y_leg, h_in=h, w_in=w,  # noqa: SLF001
                           title=None, ncol=1)  # fmt: skip
    psf._assert_report_layout(fig, [leg1, leg2], [ax], w, h, what=path.name)  # noqa: SLF001
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def render_grid(panels: Sequence[Panel], ncols: int, anchor: tuple[float, float], land: Sequence[Any], top: float, ticks: Sequence[float],
                font: Any, cjk: bool, path: Path, legend_lines: Sequence[Line], dpi: int = 300, legend_h: float = 0.92,
                compact: bool = False, legend_title: str | None = None) -> None:  # fmt: skip
    """多格（季節 2×2 等）：每格一個 :class:`Panel`（自己的觀測、期望、標示範圍）；共用圖例在下方。"""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from lagrangian_backtracking.native_runner.figures import _Labeler

    t = _Labeler(cjk)
    site_map = psf._build_report_map(anchor, 25.0, land, {})  # noqa: SLF001
    w = psf.REPORT_W_IN
    nrows = -(-len(panels) // ncols)
    left, right, gap, title_h, rowgap, xlab = (
        psf._RG_LEFT_1D,
        psf._RG_RIGHT_MARGIN,
        psf._RG_GAP,
        psf._RG_TITLE,
        psf._RG_ROWGAP,
        psf._RG_XLAB_1D,
    )  # noqa: SLF001
    p = (w - left - right - (ncols - 1) * gap) / ncols
    top_margin = 0.04
    h = round(
        top_margin
        + nrows * (title_h + p)
        + (nrows - 1) * rowgap
        + xlab
        + psf._RO_LEGEND_GAP
        + legend_h
        + 0.04,
        3,
    )  # noqa: SLF001
    fig = plt.figure(figsize=(w, h), dpi=dpi)
    f8 = psf._text_font(font)  # noqa: SLF001
    xt = psf._report_ticks(site_map.extent[0], site_map.extent[1], (0.2, 0.5), 2, inner=(0.12, 0.88))  # noqa: SLF001
    yt = psf._report_ticks(site_map.extent[2], site_map.extent[3], (0.2, 0.5), 2, inner=(0.1, 0.9))  # noqa: SLF001
    axes = []
    for idx, panel in enumerate(panels):
        r, c = divmod(idx, ncols)
        y_top = top_margin + r * (title_h + p + rowgap) + title_h
        ax = fig.add_axes([(left + c * (p + gap)) / w, 1.0 - (y_top + p) / h, p / w, p / h])
        last_row = r == nrows - 1 or idx + ncols >= len(panels)
        psf._draw_report_polar(ax, _plan(panel, top, ticks), site_map, font, compact=compact, xticks=xt, yticks=yt,  # noqa: SLF001
                               label_x=last_row, label_y=(c == 0), one_decimal=True)  # fmt: skip
        _add_overlays(ax, site_map, panel, top, compact=compact)
        ax.set_title(panel.title, loc="left", fontproperties=f8, fontsize=psf.REPORT_TEXT_PT, pad=2.5)  # noqa: SLF001
        axes.append(ax)
    y_leg = top_margin + nrows * (title_h + p) + (nrows - 1) * rowgap + xlab + psf._RO_LEGEND_GAP  # noqa: SLF001
    obs_handles = [
        Line2D([], [], color=ln.color, linestyle="-", linewidth=1.4, label=ln.label) for ln in legend_lines
    ]
    has_exp = any(pn.exp_pct is not None for pn in panels)
    has_runs = any(pn.runs for pn in panels)
    leg1 = psf._fig_legend(
        fig,
        obs_handles,
        font,
        loc="upper left",
        x_in=left,
        y_top_in=y_leg,
        h_in=h,
        w_in=w,
        title=legend_title,
        ncol=1,
    )  # noqa: SLF001
    right_handles = _extra_handles(has_exp, has_runs) + _base_handles(t)
    leg2 = psf._fig_legend(
        fig,
        right_handles,
        font,
        loc="upper left",
        x_in=left + 1.75,
        y_top_in=y_leg,
        h_in=h,
        w_in=w,
        title=None,
        ncol=1,
    )  # noqa: SLF001
    psf._assert_report_layout(fig, [leg1, leg2], axes, w, h, what=path.name)  # noqa: SLF001
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
