"""WP-73：報告圖字級與版面規格——與 TRAPs 報告（簡要報告大圖，``traps.figure_standard`` / ``traps.concise_style``）一致。

LBT 與 TRAPs 的報告是接續的章節，閱讀視覺必須一致。本模組集中放「報告版」（``fs8``）圖面規格，
軌跡總覽、條件式來源足跡、仍在追蹤比例曲線三種圖共用；圓形來源方位圖（polar_source_figures）本來就以
6.27 吋畫布、8 pt 字級、不縮放繪製，不經過本模組。

規格（逐項對照 TRAPs）
----------------------
* 畫布寬 = 報告插入寬度 6.27 吋（TRAPs 為 6.3 吋，因其 Word 文字區寬為 6.3 吋；LBT 文字區寬 9026 twips = 6.268 吋），
  不縮放，所以圖上的字級就是印出字級。高度依圖例列數而定：2 列圖例 4.13 吋（同 TRAPs 簡要報告大圖），每多一列加 0.16 吋。
* 地圖主圖框：高 2.98 吋、寬 = 高 × 共用長寬比（1.8°×1.2° 窗口、參考緯度 24.1° → 4.082 吋），左緣 0.91 吋、
  下緣 = 圖例帶高（2 列 0.95 吋）、上方留白 0.20 吋；五個地點的畫布與主圖框完全相同。
* 字級：刻度、座標軸標題、色條、圖例一律 8 pt；字型 Noto Sans CJK TC（刻度數字與色條 10 的次方也用同一字型）。
* 框線 1.1 pt ``#222222``；刻度朝外、長 3.5 pt、寬 0.8 pt、與標籤間距 2 pt；軸標題間距 2 pt；
  色條刻度長 3.0 pt、寬 0.8、間距 1 pt；色條寬 0.12 吋、與主圖框相距 0.093 吋、與主圖框等高。
* 圖例：有外框（淺灰 0.8、白底、不透明度 0.8、圓角）、8 pt、欄距 0.9、圖例符號長 1.6，置中於主圖框正下方。
* 不畫格線。
* 線寬、記號大小以「印出效果」不變為原則：原圖以較大畫布縮小印出，這裡把線寬與記號大小乘上同一個縮放比
  （軌跡圖 6.27/9.48、足跡圖 6.27/8.93、曲線圖 6.27/8.0），所以印出的線與記號和舊圖一樣粗，只有文字放大到 8 pt。
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

REPORT_WIDTH_IN = 6.27
TEXT_PT = 8.0
FRAME_H_IN = 2.98
FRAME_LEFT_IN = 0.91
TOP_MARGIN_IN = 0.20
BAND_2ROW_IN = 0.95
BAND_PER_EXTRA_ROW_IN = 0.16
LEGEND_GAP_IN = 0.08
CBAR_W_IN = 0.12
CBAR_PAD_IN = 0.093
FRAME_COLOUR = "#222222"
FRAME_LW_PT = 1.1
TICK_LEN_PT = 3.5
TICK_W_PT = 0.8
TICK_PAD_PT = 2.0
LABEL_PAD_PT = 2.0
CBAR_TICK_LEN_PT = 3.0
CBAR_TICK_PAD_PT = 1.0

# 舊圖畫布寬（英吋）→ 縮放比（新畫布寬 6.27 ÷ 舊畫布寬）
OLD_TRAJ_CANVAS_W_IN = 9.48
OLD_FOOTPRINT_CANVAS_W_IN = 8.9267  # 5356 px / 600 dpi（足跡圖以 tight 裁切輸出）
OLD_CURVE_CANVAS_W_IN = 8.0
PRINT_SCALE_TRAJ = REPORT_WIDTH_IN / OLD_TRAJ_CANVAS_W_IN
PRINT_SCALE_FOOTPRINT = REPORT_WIDTH_IN / OLD_FOOTPRINT_CANVAS_W_IN
PRINT_SCALE_CURVE = REPORT_WIDTH_IN / OLD_CURVE_CANVAS_W_IN

FRAMED_LEGEND_KW: dict[str, Any] = {
    "frameon": True,
    "fancybox": True,
    "edgecolor": "0.8",
    "facecolor": "white",
    "framealpha": 0.8,
    "columnspacing": 0.9,
    "handlelength": 1.6,
}


def band_in(legend_rows: int) -> float:
    """主圖框下緣距畫布底端（英吋）：座標軸刻度與標題帶加圖例帶；2 列 0.95，每多一列加 0.16。"""

    return round(BAND_2ROW_IN + BAND_PER_EXTRA_ROW_IN * max(int(legend_rows) - 2, 0), 4)


def canvas_in(legend_rows: int) -> tuple[float, float]:
    """報告版畫布 ``(寬, 高)``（英吋）。"""

    return REPORT_WIDTH_IN, round(band_in(legend_rows) + FRAME_H_IN + TOP_MARGIN_IN, 4)


def frame_box_in(frame_w_over_h: float, legend_rows: int) -> tuple[float, float, float, float]:
    """主圖框 ``(左, 下, 寬, 高)``（英吋，左下角起算）。"""

    return FRAME_LEFT_IN, band_in(legend_rows), FRAME_H_IN * float(frame_w_over_h), FRAME_H_IN


@contextmanager
def report_rc(family: str) -> Iterator[None]:
    """報告版 Matplotlib 全域字型：``font.family`` ＝ [Noto Sans CJK TC, DejaVu Sans]（同 TRAPs ``cjk_font_context``），
    色條刻度的 10 的次方（mathtext）也用同一字型；不污染呼叫端 rcParams。"""

    import matplotlib

    rc = {
        "font.family": [family, "DejaVu Sans"],
        "axes.unicode_minus": False,
        "mathtext.fontset": "custom",
        "mathtext.default": "regular",
        "mathtext.rm": family,
        "mathtext.it": family,
        "mathtext.bf": family,
        "mathtext.sf": family,
        "mathtext.tt": family,
        "mathtext.cal": family,
        "font.size": TEXT_PT,
    }
    with matplotlib.rc_context(rc):
        yield


def text_font(font: Any, pt: float = TEXT_PT) -> Any:
    """字型物件的 8 pt 副本（``fontproperties=`` 傳入時字級以物件為準，所以必須是 8 pt 的副本）。"""

    f = font.copy()
    f.set_size(pt)
    return f


def scale_print_effect(artists: Sequence[Any], scale: float) -> int:
    """把線寬、記號大小與記號邊線寬乘上 ``scale``（只對傳入的 artist 做一次）。回傳處理的 artist 數。"""

    from matplotlib.collections import Collection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.spines import Spine

    n = 0
    for a in artists:
        if isinstance(a, Spine):
            continue
        if isinstance(a, Line2D):
            a.set_linewidth(a.get_linewidth() * scale)
            a.set_markersize(a.get_markersize() * scale)
            a.set_markeredgewidth(a.get_markeredgewidth() * scale)
            n += 1
        elif isinstance(a, Collection):
            lw = a.get_linewidth()
            try:
                a.set_linewidth([float(v) * scale for v in lw])
            except TypeError:
                a.set_linewidth(float(lw) * scale)
            n += 1
        elif isinstance(a, Patch):
            a.set_linewidth(a.get_linewidth() * scale)
            n += 1
    return n


def style_map_axes(ax: Any, font8: Any) -> None:
    """框線 1.1 pt #222222、刻度朝外（長 3.5、寬 0.8、間距 2）、刻度標籤與座標軸標題 8 pt（Noto）。"""

    ax.tick_params(
        axis="both", which="major", direction="out", length=TICK_LEN_PT, width=TICK_W_PT, color=FRAME_COLOUR,
        labelsize=TEXT_PT, pad=TICK_PAD_PT, bottom=True, left=True, top=False, right=False,
    )
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontproperties(font8)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(FRAME_LW_PT)
        spine.set_edgecolor(FRAME_COLOUR)
        spine.set_zorder(20)


def set_axis_titles(ax: Any, xlabel: str, ylabel: str, font8: Any) -> None:
    ax.set_xlabel(xlabel, fontproperties=font8, fontsize=TEXT_PT, labelpad=LABEL_PAD_PT)
    ax.set_ylabel(ylabel, fontproperties=font8, fontsize=TEXT_PT, labelpad=LABEL_PAD_PT)


def style_colorbar(cbar: Any, label: str, font8: Any) -> None:
    cbar.set_label(label, fontproperties=font8, fontsize=TEXT_PT, labelpad=LABEL_PAD_PT)
    cbar.ax.tick_params(labelsize=TEXT_PT, pad=CBAR_TICK_PAD_PT, length=CBAR_TICK_LEN_PT, width=TICK_W_PT, color=FRAME_COLOUR)
    for lab in cbar.ax.get_yticklabels():
        lab.set_fontproperties(font8)
    for spine in cbar.ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(FRAME_LW_PT)
        spine.set_edgecolor(FRAME_COLOUR)


def ncol_for_rows(n_items: int, rows: int) -> int:
    return max(1, math.ceil(n_items / max(int(rows), 1)))


def axes_bottom_in(fig: Any, ax: Any) -> float:
    """主圖框連同刻度標籤與座標軸標題的下緣（英吋，距畫布底端）。"""

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tb = ax.get_tightbbox(renderer)
    return float(tb.transformed(fig.dpi_scale_trans.inverted()).y0)


def framed_legend(fig: Any, ax: Any, handles: Sequence[Any], font8: Any, *, rows: int) -> Any:
    """置中於主圖框、頂端在座標軸標題下方 :data:`LEGEND_GAP_IN` 的外框圖例（8 pt，欄數使列數 ≤ ``rows``）。"""

    w, h = fig.get_size_inches()
    pos = ax.get_position()
    cx = (pos.x0 + pos.x1) / 2.0
    top = axes_bottom_in(fig, ax) - LEGEND_GAP_IN
    return fig.legend(
        handles=list(handles),
        loc="upper center",
        bbox_to_anchor=(cx, top / h),
        ncol=ncol_for_rows(len(handles), rows),
        prop=font8,
        **FRAMED_LEGEND_KW,
    )


def legend_geometry(fig: Any, legend: Any, ax: Any) -> dict[str, float]:
    """圖例外框（英吋）：左、右、下、上緣，及其與座標軸標題下緣、畫布邊界的間距；供版面檢查。"""

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.dpi_scale_trans.inverted()
    lb = legend.get_window_extent(renderer).transformed(inv)
    ab = ax.get_tightbbox(renderer).transformed(inv)
    w, _h = fig.get_size_inches()
    return {
        "left_in": float(lb.x0),
        "right_in": float(lb.x1),
        "bottom_in": float(lb.y0),
        "top_in": float(lb.y1),
        "gap_below_axes_in": float(ab.y0 - lb.y1),
        "margin_left_in": float(lb.x0),
        "margin_right_in": float(w - lb.x1),
        "margin_bottom_in": float(lb.y0),
        "rows": int(math.ceil(len(legend.get_texts()) / max(legend._ncols, 1))),  # noqa: SLF001
    }


def text_audit(fig: Any) -> list[dict[str, Any]]:
    """畫完後掃描圖上全部可見文字：內容、字級（pt）、字型檔與面（含刻度標籤、座標軸標題、圖例、色條）。"""

    from matplotlib import font_manager
    from matplotlib.text import Text

    fig.canvas.draw()
    out: list[dict[str, Any]] = []
    for t in fig.findobj(Text):
        s = t.get_text()
        if not s or not t.get_visible():
            continue
        is_math = "$" in s
        try:
            found = font_manager.findfont(t.get_fontproperties())
            path = getattr(found, "path", None) or str(found)
            index = getattr(found, "face_index", None)
            family = font_manager.FontProperties(fname=found).get_name()
        except Exception:  # noqa: BLE001
            path, index, family = "?", None, "?"
        out.append(
            {
                "text": s,
                "size_pt": float(t.get_fontsize()),
                "font_file": str(path).rsplit("/", 1)[-1],
                "font_face_index": index,
                "font_family": family,
                "mathtext": is_math,
            }
        )
    return out


def audit_summary(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    sizes = sorted({round(i["size_pt"], 3) for i in items})
    fams = sorted({i["font_family"] for i in items if not i["mathtext"]})
    import matplotlib

    return {
        "mathtext_font_rm": str(matplotlib.rcParams["mathtext.rm"]),
        "mathtext_fontset": str(matplotlib.rcParams["mathtext.fontset"]),
        "n_texts": len(items),
        "sizes_pt": sizes,
        "min_size_pt": min(sizes) if sizes else None,
        "font_families": fams,
        "mathtext_texts": [i["text"] for i in items if i["mathtext"]],
    }


def require_report_rc(font: Any) -> None:
    """報告版圖必須在 :func:`report_rc` 內繪製（否則刻度數字與色條 10 的次方會落回 DejaVu Sans）。"""

    import matplotlib

    fam = matplotlib.rcParams["font.family"]
    if not fam or str(fam[0]) != str(font.get_name()) or matplotlib.rcParams.get("mathtext.fontset") != "custom":
        raise RuntimeError("報告版（fs8）圖必須在 report_style.report_rc(字型家族) 內繪製")
