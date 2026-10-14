"""沿開放邊界之相對來源權重曲線（依組別分線）的出圖、輸出表與 manifest。

本模組把 :mod:`arc_by_group` 的計算結果畫成圖、寫成表。圖種（每站、每案）共六張：

====================================  ==========================================
檔名（不含副檔名）                    內容
====================================  ==========================================
``boundary_arc_weight_by_class_overlay``   十個沉降速度類別疊在同一張圖
``boundary_arc_weight_by_class_panels``    十類分成 2×5 格，各格縱軸範圍相同
``boundary_arc_weight_by_season_overlay``  四季（沉底時刻）疊圖
``boundary_arc_weight_by_season_panels``   四季分成 2×2 格
``boundary_arc_weight_by_tide_overlay``    大潮／小潮（沉底時刻）疊圖
``boundary_arc_weight_by_tide_panels``     大潮／小潮分成 1×2 格
====================================  ==========================================

每張圖輸出「含文字」（``figures/``，畫標題與副標）與「無文字」（``figures-notext/``，不畫
標題與副標，只留座標軸、各邊名稱與圖例，供報告排版；說明改由 ``captions.md`` 承載）兩版，
各自同時輸出 PNG 與 SVG，與既有 ``native-figures`` 的兩版慣例相同。

圖面規則
--------
* 圖上**不寫站名**；站名只出現在檔名（站子目錄）與 ``captions.md`` 的圖說。
* 橫軸、縱軸兩端都標數字（刻度包含 0 與軸上限）。橫軸為弧長 0 到邊界全長向上取整；
  縱軸為相對來源權重（%／km），單位與積分關係見 :mod:`arc_by_group`。
* 同一種圖在所有站與所有案例使用同一組軸範圍與刻度：橫軸由全部站的最大邊界全長決定，
  縱軸由該分組全部站、全部案例的曲線最大值決定（向上取到整齊的刻度）。軸範圍記錄在
  manifest 的 ``axes`` 欄位，圖面與圖說不另加說明。
* 類別名稱一律經 :mod:`display_names` 的顯示名稱對照表，不使用材質清冊的官方分類名稱。

輸出目錄（每案一個，必須是尚不存在的全新目錄）
--------------------------------------------
``figures/<站>/<圖檔>.png|svg``、``figures-notext/<站>/<圖檔>.png|svg``、
``tables/<站>/<分組>_summary.json``（各邊占比、組間兩兩總變異距離、雜訊基準）、
``tables/<站>/<分組>_curves.csv``（各組曲線格點值）、``manifest.json``、``captions.md``、
``README.md``。``manifest.json`` 內嵌每張表的摘要，供報告直接引用。
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..outputs import sha256_file
from . import arc_by_group as abg
from .arc_by_group import ARC_GROUPINGS, ArcByGroupError, ArcByGroupParams, GroupingResult
from .display_names import MATERIAL_DISPLAY_NAME_ZH
from .figures import (
    SITE_LABELS_ZH,
    _add_two_line_title,
    _corner_name_zh,
    _describe_experiment_case,
    _EDGE_NAME_ZH,
    _Labeler,
    _nice_ceiling_axis,
    _resolve_font,
    _save_exclusive,
    _title_top_margin,
)

__all__ = [
    "ARC_BY_GROUP_FIGURE_STEMS",
    "ARC_BY_GROUP_TEXT_DIRS",
    "ArcByGroupCase",
    "ArcByGroupConfig",
    "figure_stem",
    "build_arc_by_group",
]

_LAYOUTS: tuple[str, ...] = ("overlay", "panels")


def figure_stem(grouping: str, layout: str) -> str:
    """分組代號與版面（``overlay`` 疊圖或 ``panels`` 分格）→ 圖檔名（不含副檔名）。"""

    if grouping not in ARC_GROUPINGS:
        raise ArcByGroupError(f"未知的分組代號：{grouping!r}")
    if layout not in _LAYOUTS:
        raise ArcByGroupError(f"未知的版面：{layout!r}")
    return f"boundary_arc_weight_by_{grouping}_{layout}"


ARC_BY_GROUP_FIGURE_STEMS: tuple[str, ...] = tuple(
    figure_stem(g, layout) for g in ARC_GROUPINGS for layout in _LAYOUTS
)
"""本模組六張圖的固定檔名清單（不含副檔名），供 manifest、測試與 README 共用同一份。"""

ARC_BY_GROUP_TEXT_DIRS: Mapping[str, str] = {"full": "figures", "none": "figures-notext"}
"""文字層級 → 輸出子目錄名稱（與既有 ``native-figures`` 的兩版目錄命名一致）。"""

# 各分組分格版面（列數, 欄數, 圖寬 in, 圖高 in）。
_PANEL_LAYOUT: Mapping[str, tuple[int, int, float, float]] = {
    "class": (2, 5, 21.0, 8.6),
    "season": (2, 2, 12.5, 8.8),
    "tide": (1, 2, 12.5, 5.0),
}

_SEASON_COLORS = {"DJF": "#0072B2", "MAM": "#009E73", "JJA": "#D55E00", "SON": "#CC79A7"}
_TIDE_COLORS = {"spring_proxy": "#D55E00", "neap_proxy": "#0072B2", "unknown": "#7f7f7f"}
_EDGE_FACE = ("#eef3f7", "#ffffff")


# ── 設定 ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ArcByGroupCase:
    """一個物理案例：後處理目錄（只讀）與其輸出目錄（必須尚不存在）。"""

    case_id: str
    postprocess_dir: Path
    output_dir: Path

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ArcByGroupError("case_id 不可為空")
        object.__setattr__(self, "postprocess_dir", Path(self.postprocess_dir))
        object.__setattr__(self, "output_dir", Path(self.output_dir))


@dataclass(frozen=True, slots=True)
class ArcByGroupConfig:
    """一次執行的完整設定。

    ``cases``：一或多個案例；多個案例時，同一種圖的軸範圍由全部案例、全部站共同決定。
    ``sites``：只輸出指定站（預設輸出第一個案例中出現的全部站）。``groupings``：要輸出的分組。
    ``params``：計算參數（頻寬、格數、重抽次數、種子）。``font_path``：自備中文字型檔；省略時
    走 ``resolve_cjk_font``，找不到合格字型時整批圖面改用英文標籤。``dpi``：PNG 解析度。
    """

    cases: tuple[ArcByGroupCase, ...]
    sites: tuple[str, ...] | None = None
    groupings: tuple[str, ...] = ARC_GROUPINGS
    params: ArcByGroupParams = field(default_factory=ArcByGroupParams)
    font_path: Path | None = None
    dpi: int = 600

    def __post_init__(self) -> None:
        cases = tuple(self.cases)
        if not cases:
            raise ArcByGroupError("至少需要一個案例")
        if len({c.case_id for c in cases}) != len(cases):
            raise ArcByGroupError("case_id 不可重複")
        if len({c.output_dir for c in cases}) != len(cases):
            raise ArcByGroupError("各案例的輸出目錄不可相同")
        object.__setattr__(self, "cases", cases)
        bad = [g for g in self.groupings if g not in ARC_GROUPINGS]
        if bad or not self.groupings:
            raise ArcByGroupError(f"groupings 必須是 {ARC_GROUPINGS} 的非空子集，收到 {self.groupings!r}")
        object.__setattr__(self, "groupings", tuple(self.groupings))
        if self.sites is not None:
            sites = tuple(self.sites)
            if not sites or any(not isinstance(s, str) or not s for s in sites):
                raise ArcByGroupError("sites 若提供必須是非空字串序列")
            object.__setattr__(self, "sites", sites)
        if self.font_path is not None:
            object.__setattr__(self, "font_path", Path(self.font_path))
        if self.dpi < 1:
            raise ArcByGroupError("dpi 必須是正整數")


# ── 軸範圍：同一種圖在所有站、所有案例共用 ───────────────────────────────


@dataclass(frozen=True, slots=True)
class _AxesPlan:
    """同一組圖共用的軸範圍。``x_max``／``x_ticks``：橫軸；``y_max``／``y_ticks``：各分組縱軸。"""

    x_max: float
    x_ticks: np.ndarray
    panel_x_ticks: np.ndarray
    y_max: Mapping[str, float]
    y_ticks: Mapping[str, np.ndarray]


def _plan_axes(results: Sequence[GroupingResult]) -> _AxesPlan:
    """由全部站、全部案例的結果決定橫軸與各分組縱軸（向上取到整齊的刻度）。"""

    longest = max((r.length_km for r in results), default=0.0)
    x_max, x_ticks = _nice_ceiling_axis(longest, max_intervals=10)
    panel_ticks = np.arange(0.0, x_max + 1e-9, 100.0 if x_max >= 300 else x_max / 4.0)
    y_max: dict[str, float] = {}
    y_ticks: dict[str, np.ndarray] = {}
    for grouping in {r.grouping for r in results}:
        peak = max(
            max(r.max_density(), float(r.density_all_pct_per_km.max()))
            for r in results
            if r.grouping == grouping
        )
        top, ticks = _nice_ceiling_axis(peak * 1.08, max_intervals=6)
        y_max[grouping], y_ticks[grouping] = top, ticks
    return _AxesPlan(float(x_max), x_ticks, panel_ticks, y_max, y_ticks)


# ── 繪圖 ─────────────────────────────────────────────────────────────────


def _three_sig(value: float) -> str:
    """三位有效數字的數字字串（保留尾端的 0，例如 0.920、0.0142、0.000897）。"""

    return f"{value:#.3g}".rstrip(".")


def _speed_text(summary: Mapping[str, Any]) -> str:
    """類別的沉降速度文字（絕對值，m/s，三位有效數字）。

    該類穿越粒子實際使用的速度只有一個（每類單一速度的舊版資料）時，寫單一值；
    有多個速度（每顆粒子在該類 10 個速度中抽一個）時，寫「最小值–最大值」。
    """

    if int(summary["distinct_values"]) <= 1:
        return f"{_three_sig(summary['abs_median_mps'])} m/s"
    return f"{_three_sig(summary['abs_min_mps'])}–{_three_sig(summary['abs_max_mps'])} m/s"


def _group_colors(result: GroupingResult) -> list[Any]:
    """各組曲線顏色。類別依沉降速度由慢到快取 viridis 0–0.9；季節與潮況用固定色盲友善色。"""

    from matplotlib import cm

    if result.grouping == "class":
        k = len(result.group_ids)
        return [cm.viridis(0.9 * (i / max(k - 1, 1))) for i in range(k)]
    table = _SEASON_COLORS if result.grouping == "season" else _TIDE_COLORS
    return [table.get(g, "#7f7f7f") for g in result.group_ids]


def _legend_entries(result: GroupingResult) -> list[str]:
    out = []
    for k, gid in enumerate(result.group_ids):
        if result.grouping == "class" and result.velocity_summary is not None:
            out.append(f"{result.labels_zh[k]}　{_speed_text(result.velocity_summary[k])}")
        elif result.grouping == "class":
            out.append(result.labels_zh[k])
        else:
            out.append(f"{result.labels_zh[k]}　{int(result.n_crossed[k]):,} 顆")
    return out


def _has_speed_range(result: GroupingResult) -> bool:
    """類別分組中是否有任一類在穿越粒子上出現多個沉降速度（需以範圍標示）。"""

    return bool(result.velocity_summary) and any(int(v["distinct_values"]) > 1 for v in result.velocity_summary)


def _decorate_axis(ax: Any, result: GroupingResult, plan: _AxesPlan, font: Any, *, small: bool) -> None:
    """底色（交替淺色區分各邊）、轉角虛線、軸範圍與刻度（兩端標值）。"""

    from matplotlib.ticker import FuncFormatter

    for k, (_name, a, b) in enumerate(result.edges):
        ax.axvspan(a, b, facecolor=_EDGE_FACE[k % 2], zorder=0)
    for _name, s in result.corners:
        ax.axvline(s, color="#555555", linewidth=0.6 if small else 0.8, linestyle="--", zorder=1)
    y_top = plan.y_max[result.grouping]
    ax.set_xlim(0.0, plan.x_max)
    ax.set_ylim(0.0, y_top)
    ax.set_xticks(plan.panel_x_ticks if small else plan.x_ticks)
    ax.set_yticks(plan.y_ticks[result.grouping])
    number = FuncFormatter(lambda v, _p=None: f"{v:g}")
    ax.xaxis.set_major_formatter(number)
    ax.yaxis.set_major_formatter(number)
    ax.grid(axis="y", color="#dddddd", linewidth=0.5, zorder=0)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontproperties(font)
        lab.set_fontsize(8 if small else 10)


def _edge_labels_above(ax: Any, result: GroupingResult, t: _Labeler, font: Any) -> None:
    """在上框外側寫各邊名稱與轉角名稱（與既有沿弧長分布圖相同的位置與字級）。"""

    for name, a, b in result.edges:
        ax.text(
            0.5 * (a + b),
            1.012,
            t.t(f"{_EDGE_NAME_ZH.get(name, name)}邊界", f"{name} edge"),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontproperties=font,
            fontsize=10,
            color="#333333",
            clip_on=False,
        )
    for name, s in result.corners:
        ax.text(
            s,
            1.012,
            t.t(_corner_name_zh(name), name),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontproperties=font,
            fontsize=9,
            color="#444444",
            clip_on=False,
        )


def _x_label(t: _Labeler) -> str:
    return t.t("沿流場範圍開放邊界的弧長（km，自邊界折線起點量起）", "Arc length along open boundary (km from start)")


def _y_label(t: _Labeler) -> str:
    return t.t("相對來源權重（%／km）", "Relative source weight (% per km)")


def _join_corner_zh(result: GroupingResult) -> str:
    """閉合折線首尾相接處的角名（最後一邊與第一邊的交會角，例如「西南角」）。"""

    if len(result.edges) < 2:
        return ""
    return _corner_name_zh(f"{result.edges[-1][0]}-{result.edges[0][0]}")


def _end_note_zh(result: GroupingResult) -> str:
    if result.closed:
        return f"邊界折線首尾閉合，橫軸兩端同為{_join_corner_zh(result)}，平滑時首尾環狀接續"
    return "邊界折線首尾未閉合，兩端為真正的端點，平滑時於端點鏡射（質量守恆）"


def _title_pair(result: GroupingResult, t: _Labeler) -> tuple[str, str]:
    spec = abg.grouping_spec(result.grouping)
    title = t.t(
        f"沿開放邊界的相對來源權重｜依{spec.title_zh}分線",
        f"Relative source weight along open boundary | by {result.grouping}",
    )
    subtitle = t.t(
        f"穿越粒子 {result.n_all:,} 顆｜邊界全長 {result.length_km:.0f} km｜{_end_note_zh(result)}｜"
        f"高斯核頻寬 {result.params.sigma_km:g} km｜每條曲線對弧長積分為 100%",
        f"{result.n_all:,} crossed | length {result.length_km:.0f} km | sigma {result.params.sigma_km:g} km",
    )
    return title, subtitle


def _plot_overlay(
    result: GroupingResult, plan: _AxesPlan, t: _Labeler, font: Any, path: Path, dpi: int, figure_text: str
) -> None:
    """疊圖：每組一條曲線，圖例在右側。"""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13.5, 5.8), dpi=150)
    fig.subplots_adjust(left=0.07, bottom=0.14, right=0.78, top=_title_top_margin(figure_text, full=0.80, none=0.96))
    _decorate_axis(ax, result, plan, font, small=False)
    colors = _group_colors(result)
    entries = _legend_entries(result)
    for k in range(len(result.group_ids)):
        ax.plot(result.x_km, result.density_pct_per_km[k], color=colors[k], linewidth=1.6, zorder=3, label=entries[k])
    _edge_labels_above(ax, result, t, font)
    ax.set_xlabel(_x_label(t), fontproperties=font)
    ax.set_ylabel(_y_label(t), fontproperties=font)
    class_range = result.grouping == "class" and _has_speed_range(result)
    legend_title = {
        "class": (
            t.t("類別　沉降速度範圍", "Class and settling speed range")
            if class_range
            else t.t("類別　沉降速度（由慢到快）", "Class and settling speed (slow to fast)")
        ),
        "season": t.t("沉底時刻的季節　穿越粒子數", "Season of deposition"),
        "tide": t.t("沉底時刻的潮況　穿越粒子數", "Tide class at deposition"),
    }[result.grouping]
    leg = ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, prop=font, title=legend_title)
    leg.get_title().set_fontproperties(font)
    title, subtitle = _title_pair(result, t)
    _add_two_line_title(fig, t, font, title, subtitle, figure_text=figure_text)
    _save_exclusive(fig, path, dpi=dpi)
    plt.close(fig)


def _plot_panels(
    result: GroupingResult, plan: _AxesPlan, t: _Labeler, font: Any, path: Path, dpi: int, figure_text: str
) -> None:
    """分格圖：每組一格（彩色線），各格疊上全部組別合併曲線（灰線）；各格縱軸範圍相同。"""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    rows, cols, width, height = _PANEL_LAYOUT[result.grouping]
    fig, axes = plt.subplots(rows, cols, figsize=(width, height), dpi=130, squeeze=False)
    fig.subplots_adjust(
        left=0.07,
        right=0.99,
        bottom=0.12 if rows > 1 else 0.17,
        top=_title_top_margin(figure_text, full=0.78 if rows > 1 else 0.72, none=0.93),
        wspace=0.22,
        hspace=0.40,
    )
    colors = _group_colors(result)
    flat = list(axes.ravel())
    for k, ax in enumerate(flat):
        if k >= len(result.group_ids):
            ax.set_visible(False)
            continue
        _decorate_axis(ax, result, plan, font, small=True)
        ax.plot(result.x_km, result.density_all_pct_per_km, color="#888888", linewidth=0.9, zorder=2)
        ax.plot(result.x_km, result.density_pct_per_km[k], color=colors[k], linewidth=1.8, zorder=3)
        if result.grouping == "class" and result.velocity_summary is not None:
            ax.set_title(
                f"{result.labels_zh[k]}\n{_speed_text(result.velocity_summary[k])}", fontproperties=font, fontsize=11
            )
        else:
            ax.set_title(f"{result.labels_zh[k]}（{int(result.n_crossed[k]):,} 顆）", fontproperties=font, fontsize=11)
        if k == 0:
            for name, a, b in result.edges:
                ax.text(
                    0.5 * (a + b),
                    0.97,
                    t.t(_EDGE_NAME_ZH.get(name, name), name),
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontproperties=font,
                    fontsize=9,
                    color="#555555",
                )
    fig.supxlabel(_x_label(t), fontproperties=font, fontsize=12, y=0.02)
    fig.supylabel(_y_label(t), fontproperties=font, fontsize=12, x=0.012)
    title, subtitle = _title_pair(result, t)
    subtitle += t.t("\n彩色線為該組，灰線為全部組別合併；各格縱軸範圍相同", "\ncolored: this group; gray: all groups")
    _add_two_line_title(fig, t, font, title, subtitle, figure_text=figure_text)
    _save_exclusive(fig, path, dpi=dpi)
    plt.close(fig)


# ── 圖說 ─────────────────────────────────────────────────────────────────


def _caption_zh(
    result: GroupingResult, layout: str, *, site_label: str, case_text: str, table: Mapping[str, Any]
) -> str:
    """單張圖（某站 × 某案 × 某圖種）的繁體中文圖說；全部直述句，不使用該詞。"""

    spec = abg.grouping_spec(result.grouping)
    groups_n = len(result.group_ids)
    layout_zh = (
        f"{groups_n} 條曲線疊在同一張圖"
        if layout == "overlay"
        else f"{groups_n} 個分格各畫一組（彩色線），並疊上全部組別合併的曲線（灰線），各分格縱軸範圍相同"
    )
    if result.grouping == "class":
        group_zh = (
            f"沉降速度類別（{groups_n} 類；圖例由上而下依類別代表沉降速度由慢到快排序，顏色由深紫到黃綠；"
            + (
                "每類有 10 個沉降速度，各占該類粒子的十分之一，圖例與分格標題標示的是該類穿越粒子實際用到的"
                "最小與最大沉降速度（取絕對值，單位 m/s）"
                if _has_speed_range(result)
                else "每類只有一個沉降速度，標示的是該速度（取絕對值，單位 m/s）"
            )
            + "）。"
            "類別名稱為圖面顯示名稱：「不可回收塑膠」對應材質清冊的「其他/不可回收」，"
            "「可回收塑膠」對應「其他/可回收」，材質識別碼與清冊記錄不受影響"
        )
    elif result.grouping == "season":
        group_zh = (
            "粒子沉底時刻（回溯起點）所在的季節，以世界協調時間的月份劃分：冬 12–2 月、春 3–5 月、夏 6–8 月、"
            "秋 9–11 月；標籤取自沉底時刻，不是觀測錨點被選出時的時刻"
        )
    else:
        group_zh = (
            "粒子沉底時刻（回溯起點）的大潮／小潮代理分類：以該站二十五小時時窗的潮位變動強度，"
            "在同一（年、季節）內以中位數切成大潮（不小於中位數）與小潮（小於中位數）；"
            "它是以潮位資料計算的代理分類，不宣稱現場實測大小潮，標籤取自沉底時刻而非觀測時刻"
        )
    summary = table["summary"]
    pair = summary["max_tvd_pair"]
    if pair is not None:
        label_of = dict(zip(result.group_ids, result.labels_zh, strict=True))
        verdict = "超過" if summary["max_tvd"] > summary["max_tvd_pair_null_p95"] else "未超過"
        null_zh = "同釋放點交換類別標籤的配對置換" if result.grouping == "class" else "沉底事件層級的標籤置換"
        pooled = summary["half_split_particle_pooled"]
        tvd_zh = (
            f"各組兩兩總變異距離（兩分布差的絕對值總和之半）最大為 {summary['max_tvd']:.3f}"
            f"（{label_of[pair[0]]}與{label_of[pair[1]]}），該組對的{null_zh}基準第 95 百分位為 "
            f"{summary['max_tvd_pair_null_p95']:.3f}，{verdict}抽樣雜訊；同組隨機對半切的總變異距離平均為 "
            f"{pooled['mean']:.3f}。"
        )
    else:
        tvd_zh = ""
    return (
        f"穿越位置沿流場範圍開放邊界的相對來源權重曲線，依{spec.title_zh}分線，{layout_zh}。"
        f"分組依據：{group_zh}。"
        "橫軸是沿開放邊界折線自起點量起的弧長（km），底色與虛線區分南、東、北、西各邊與轉角；"
        "縱軸是相對來源權重（%／km）：每組只取該組穿越開放邊界且有弧長位置的粒子（分母為該組穿越粒子數），"
        f"依弧長分成 {result.params.n_grid} 格計數，再以頻寬 {result.params.sigma_km:g} km 的一維高斯核平滑，"
        "使每條曲線對弧長的積分都是 100%。它描述條件式來源足跡在邊界上的相對分配，尚未建立先驗、似然與觀測驗證，不可解讀為絕對來源或因果歸因。"
        f"{_end_note_zh(result)}。"
        f"{tvd_zh}"
        f"站點：{site_label}，{result.n_all:,} 顆穿越粒子，物理案例：{case_text}。"
    )


# ── 其他輸出 ─────────────────────────────────────────────────────────────


def _code_provenance() -> dict[str, Any]:
    """本程式碼所在 git 工作樹的提交與是否有未提交變更；取不到時各欄為 ``None``。"""

    root = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=20
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
                timeout=60,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": dirty}


def _input_hashes(postprocess_dir: Path) -> dict[str, str]:
    return {
        f"{stem}.parquet": sha256_file(postprocess_dir / f"{stem}.parquet")
        for stem in abg._REQUIRED_POSTPROCESS_TABLES
    }


def _write_json_exclusive(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def _readme_text(case: ArcByGroupCase, config: ArcByGroupConfig, command: str, sites: Sequence[str]) -> str:
    stems = "\n".join(f"- `{s}`" for s in ARC_BY_GROUP_FIGURE_STEMS)
    return (
        "# 沿開放邊界之相對來源權重曲線（依組別分線）\n\n"
        f"案例：`{case.case_id}`\n\n"
        "本目錄由 `lbt native-arc-by-group` 產生，只讀取 `native-postprocess` 的輸出，不改動任何輸入。\n\n"
        "## 內容\n\n"
        "- `figures/<站>/`：含文字版（標題、副標、圖例、座標軸標籤），每張圖同時輸出 PNG 與 SVG。\n"
        "- `figures-notext/<站>/`：無文字版（不畫標題與副標，只留座標軸、各邊名稱與圖例）。\n"
        "- `tables/<站>/<分組>_summary.json`：各組粒子數、各邊（南、東、北、西）占比、組間兩兩總變異距離、"
        "抽樣雜訊基準（同組隨機對半切；類別另有同釋放點交換類別標籤的配對置換，季節與潮況為沉底事件層級的標籤置換）。\n"
        "- `tables/<站>/<分組>_curves.csv`：各組曲線格點值（%／km）。\n"
        "- `manifest.json`：參數、軸範圍、輸入與輸出檢核碼，以及每張表的摘要。\n"
        "- `captions.md`：每張圖的圖說。\n\n"
        f"圖種：\n\n{stems}\n\n"
        "## 方法摘要\n\n"
        f"每組只取穿越開放邊界且有弧長位置的粒子，依弧長分成 {config.params.n_grid} 格，"
        f"以頻寬 {config.params.sigma_km:g} km 的一維高斯核平滑，曲線對弧長積分為 100%。"
        "閉合的邊界折線首尾環狀接續，未閉合者於端點鏡射。季節與大小潮使用沉底時刻計算的"
        "`deposition_season`、`deposition_tide_class`。曲線稱為相對來源權重，不可解讀為絕對來源或因果歸因。\n\n"
        f"站：{', '.join(sites)}\n\n"
        f"重建命令：\n\n```\n{command}\n```\n"
    )


# ── 主流程 ───────────────────────────────────────────────────────────────


def build_arc_by_group(config: ArcByGroupConfig, *, command: str) -> list[Path]:
    """依設定產出所有案例的圖、表、manifest 與圖說；回傳各案例輸出目錄。

    流程：確認各輸出目錄尚不存在 → 讀入各案例各站樣本並分析 → 由全部結果決定共用軸範圍 →
    逐案逐站逐分組寫表與畫圖（含文字、無文字兩版）→ 輸入檢核碼前後比對 → 寫 manifest、
    圖說與 README。任何輸入錯誤或不一致都失敗即停；已寫出的部分檔案留在新目錄供診斷。
    """

    for case in config.cases:
        if case.output_dir.exists():
            raise ArcByGroupError(f"輸出目錄必須是尚不存在的全新目錄，但已存在：{case.output_dir}")

    hashes_before = {c.case_id: _input_hashes(c.postprocess_dir) for c in config.cases}
    results: dict[str, dict[str, dict[str, GroupingResult | None]]] = {}
    unavailable: dict[str, list[dict[str, str]]] = {c.case_id: [] for c in config.cases}
    site_order: list[str] | None = None
    samples_by_case: dict[str, Mapping[str, abg.SiteArcSample]] = {}
    for case in config.cases:
        wanted = config.sites if config.sites is not None else (tuple(site_order) if site_order else None)
        samples = abg.load_site_arc_samples(
            case.postprocess_dir, wanted, closure_tolerance_m=config.params.closure_tolerance_m
        )
        if site_order is None:
            site_order = list(samples)
        samples_by_case[case.case_id] = samples
        results[case.case_id] = {}
        for site, sample in samples.items():
            results[case.case_id][site] = {}
            for grouping in config.groupings:
                res = abg.analyze_grouping(sample, grouping, config.params)
                results[case.case_id][site][grouping] = res
                if res is None:
                    unavailable[case.case_id].append(
                        {"site_id": site, "grouping": grouping, "reason": "缺少分組欄位或可分析的組別少於兩組"}
                    )
    assert site_order is not None

    all_results = [
        r for by_site in results.values() for by_group in by_site.values() for r in by_group.values() if r is not None
    ]
    if not all_results:
        raise ArcByGroupError("沒有任何可分析的站與分組，無法出圖")
    plan = _plan_axes(all_results)

    font_resolution = _resolve_font(config.font_path)
    t = _Labeler(font_resolution.cjk_available)
    font = font_resolution.font_properties
    generated_at = datetime.now(UTC)
    code = _code_provenance()
    outputs: list[Path] = []

    for case in config.cases:
        case_dir = case.output_dir
        case_dir.mkdir(parents=True)
        pp_manifest_path = case.postprocess_dir / "manifest.json"
        pp_manifest = json.loads(pp_manifest_path.read_text(encoding="utf-8")) if pp_manifest_path.is_file() else {}
        case_text = _describe_experiment_case(pp_manifest).physics_case_zh
        figure_records: dict[str, Any] = {}
        table_records: dict[str, Any] = {}
        captions: dict[str, str] = {}
        for site in samples_by_case[case.case_id]:
            for sub in (*ARC_BY_GROUP_TEXT_DIRS.values(), "tables"):
                (case_dir / sub / site).mkdir(parents=True)
            for grouping in config.groupings:
                res = results[case.case_id][site][grouping]
                if res is None:
                    continue
                table = abg.grouping_table_dict(res, case_id=case.case_id)
                summary_path = case_dir / "tables" / site / f"{grouping}_summary.json"
                curves_path = case_dir / "tables" / site / f"{grouping}_curves.csv"
                _write_json_exclusive(summary_path, table)
                abg.write_curves_csv(res, curves_path)
                table_records[f"{site}/{grouping}"] = {
                    "summary_json": {"path": f"tables/{site}/{summary_path.name}", "sha256": sha256_file(summary_path)},
                    "curves_csv": {"path": f"tables/{site}/{curves_path.name}", "sha256": sha256_file(curves_path)},
                    "summary": table["summary"],
                    "groups": table["groups"],
                    "pairwise_tvd": table["pairwise_tvd"],
                    "noise_baselines": table["noise_baselines"],
                    "boundary": table["boundary"],
                    "skipped_groups": table["skipped_groups"],
                }
                for layout in _LAYOUTS:
                    stem = figure_stem(grouping, layout)
                    captions[f"{site}/{stem}"] = _caption_zh(
                        res,
                        layout,
                        site_label=SITE_LABELS_ZH.get(site, site),
                        case_text=case_text,
                        table=table,
                    )
                    for mode, sub in ARC_BY_GROUP_TEXT_DIRS.items():
                        png_path = case_dir / sub / site / f"{stem}.png"
                        plotter = _plot_overlay if layout == "overlay" else _plot_panels
                        plotter(res, plan, t, font, png_path, config.dpi, mode)
                        for ext in ("png", "svg"):
                            fpath = png_path.with_suffix(f".{ext}")
                            figure_records[f"{sub}/{site}/{stem}.{ext}"] = {
                                "sha256": sha256_file(fpath),
                                "size_bytes": fpath.stat().st_size,
                            }

        hashes_after = _input_hashes(case.postprocess_dir)
        if hashes_after != hashes_before[case.case_id]:
            raise ArcByGroupError("輸入檔在繪圖過程中被改動，前後 SHA-256 不一致，fail-closed")

        used_ids = sorted(
            {
                gid
                for by_group in results[case.case_id].values()
                for r in by_group.values()
                if r is not None and r.grouping == "class"
                for gid in r.group_ids
            }
        )
        manifest = {
            "generated_at_utc": generated_at.isoformat(),
            "command": command,
            "case_id": case.case_id,
            "code": code,
            "sites": list(samples_by_case[case.case_id]),
            "groupings": list(config.groupings),
            "figure_stems": list(ARC_BY_GROUP_FIGURE_STEMS),
            "text_mode_dirs": dict(ARC_BY_GROUP_TEXT_DIRS),
            "output_formats": ["png", "svg"],
            "font": dict(font_resolution.manifest_record),
            "cjk_labels_used": font_resolution.cjk_available,
            "parameters": {
                "sigma_km": config.params.sigma_km,
                "n_grid": config.params.n_grid,
                "n_resamples": config.params.n_resamples,
                "seed": config.params.seed,
                "closure_tolerance_m": config.params.closure_tolerance_m,
                "min_group_particles": config.params.min_group_particles,
                "dpi": config.dpi,
            },
            "axes": {
                "x_max_km": plan.x_max,
                "x_ticks_km": [float(v) for v in plan.x_ticks],
                "panel_x_ticks_km": [float(v) for v in plan.panel_x_ticks],
                "y_max_pct_per_km": {g: float(v) for g, v in plan.y_max.items()},
                "y_ticks_pct_per_km": {g: [float(x) for x in v] for g, v in plan.y_ticks.items()},
                "scope": "同一分組的圖在本次執行的全部站與全部案例共用",
            },
            "display_names_zh": {m: MATERIAL_DISPLAY_NAME_ZH[m] for m in used_ids if m in MATERIAL_DISPLAY_NAME_ZH},
            "display_names_note": "僅圖面與圖說使用；material_id 與材質清冊的 oca_category_zh 不受影響",
            "inputs_postprocess_dir_basename": case.postprocess_dir.name,
            "inputs_sha256": hashes_after,
            "tables": table_records,
            "unavailable": unavailable[case.case_id],
            "output_figures_sha256": figure_records,
        }
        _write_json_exclusive(case_dir / "manifest.json", manifest)
        lines = ["# 沿開放邊界之相對來源權重曲線：圖說\n", f"案例：{case.case_id}\n"]
        for key, text in captions.items():
            lines.append(f"## {key}\n\n{text}\n")
        with (case_dir / "captions.md").open("x", encoding="utf-8") as stream:
            stream.write("\n".join(lines))
        with (case_dir / "README.md").open("x", encoding="utf-8") as stream:
            stream.write(_readme_text(case, config, command, list(samples_by_case[case.case_id])))
        outputs.append(case_dir)
    return outputs

