# ruff: noqa: E501, E402, SIM115
"""WP-61：50 個到達（觀測）時刻的潮位驗證圖（全年總覽＋逐時刻放大）。

圖面規則：不寫地點名稱與站碼（站碼只出現在檔名）；軸上下界取整齊刻度並標數字；中文一律傳入字型物件
（``fontproperties``，不另給 ``fontsize``）；固定畫布（不使用 ``bbox_inches="tight"``）；
有文字版與無文字版版面完全相同，無文字版只是不畫標題、季節文字、逐格說明文字，保留座標軸、刻度、圖例。
y 軸版本：U＝五站共用整齊上下界；S＝各站各自整齊上下界。

WP-64（3-4-3 圖 3–7）：全年總覽改為寬 6.2667 英寸（＝嵌入 Word 之文字區寬，1:1）、高約 6 英寸、圖內文字 9 pt、中英並列
（潮汐強度主圖加大；顯著波高、流速兩小圖各自加標題；日期軸以 2025/MM/DD 標示；標記加大並加深色邊框、略透明，
50 個時刻都在真實位置；圖例附各類數量，各季色塊頂端標該季大潮／小潮數量；圖例置於下方）。
逐時刻放大圖程式保留，但 3-4-3 不再使用（run_figs.py 以 LBT_FIGS_KINDS=overview 只出總覽）。
"""

from __future__ import annotations

import math
from pathlib import Path

import arrival_check as ac
import numpy as np

SEASON_ZH = {"DJF": "冬", "MAM": "春", "JJA": "夏", "SON": "秋"}
SEASON_RANGE_ZH = {"DJF": "冬 12–2月", "MAM": "春 3–5月", "JJA": "夏 6–8月", "SON": "秋 9–11月"}
CLASS_ZH = {"spring_proxy": "大潮", "neap_proxy": "小潮"}
PHASE_ZH = {"fastest_rising": "漲潮最快", "fastest_falling": "退潮最快", "slack_proxy": "近平潮"}
EVENT_ZH = {"high_wave_event": "高浪事件", "strong_current_event": "強流事件"}
COL_CLASS = {"spring_proxy": "#d95f02", "neap_proxy": "#1f78b4"}
MARK_PHASE = {"fastest_rising": "^", "fastest_falling": "v", "slack_proxy": "D"}
COL_EVENT = {"high_wave_event": "#7b3294", "strong_current_event": "#1b7837"}
BAND = {"DJF": "#dce8f4", "MAM": "#e2f0da", "JJA": "#f9f0cf", "SON": "#f4e0d6"}
PHASE_ORDER = ("fastest_rising", "fastest_falling", "slack_proxy")
T0 = np.datetime64("2025-01-01T00:00:00", "ns").astype("int64")
DAY_NS = 86_400_000_000_000
HALF = 13

OV_W = 6.2667  # 總覽畫布寬（英寸）＝嵌入 Word 之寬度（1:1）；高度由版面決定（見 render_overview）
OV_TITLE = 0.24  # 有文字版上方標題區；無文字版畫布不含此區（圖內位置不變）
ZM_W = 8.4
ZM_TITLE = 0.42
DPI = 300


def nice_axis(
    lo: float,
    hi: float,
    *,
    zero_based: bool = False,
    margin: float = 0.03,
    min_int: int = 3,
    max_int: int = 5,
) -> tuple[float, float, float]:
    """回傳 (下界, 上界, 刻度間距)；間距取 1、2、5、2.5 × 10^n，區間數介於 min_int 與 max_int，取涵蓋資料的最小範圍。"""

    if zero_based:
        lo = 0.0
    span = max(hi - lo, 1e-9)
    lo2 = lo if zero_based else lo - margin * span
    hi2 = hi + margin * span
    best = None
    for e in range(-4, 3):
        for order, m in enumerate((1, 2, 5, 2.5)):
            step = m * 10.0**e
            L = 0.0 if zero_based else math.floor(lo2 / step + 1e-9) * step
            U = math.ceil(hi2 / step - 1e-9) * step
            n = round((U - L) / step)
            if min_int <= n <= max_int:
                key = (round(U - L, 9), order)
                if best is None or key < best[0]:
                    best = (key, L, U, step)
    if best is None:
        raise ValueError((lo, hi))
    _, L, U, step = best
    return round(L, 9), round(U, 9), round(step, 9)


def axis_ticks(lo: float, hi: float, step: float) -> np.ndarray:
    n = round((hi - lo) / step)
    return np.round(lo + step * np.arange(n + 1), 9)


def fmt_tick(v: float) -> str:
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s.replace("-", "−")


class Fonts:
    def __init__(self) -> None:
        from lagrangian_backtracking.native_runner.figures import _resolve_font

        res = _resolve_font(None)
        self.base = res.font_properties
        self.cjk = res.cjk_available
        self.record = dict(res.manifest_record)

    def __call__(self, size: float, weight: str | None = None):
        fp = self.base.copy()
        fp.set_size(size)
        if weight:
            fp.set_weight(weight)
        return fp


# ── 各圖的 y 軸範圍 ───────────────────────────────────────────────────────────


def window_slice(s: ac.SiteSeries, i: int) -> slice:
    return slice(i - HALF, i + HALF + 1)


def window(arr: np.ndarray, i: int) -> np.ndarray:
    """以第 i 點為中心、前後各 HALF 點的視窗；超出序列首尾的部分補 NaN（run09 的觀測時刻可靠近序列末端）。"""

    lo, hi = i - HALF, i + HALF + 1
    out = np.full(hi - lo, np.nan)
    a, b = max(lo, 0), min(hi, len(arr))
    out[a - lo : b - lo] = arr[a:b]
    return out


def site_stats(s: ac.SiteSeries, rows: list[dict], idx_of: dict, ref: ac.SiteSeries | None) -> dict:
    ym = s.valid & (s.year_of == ac.YEAR)
    e_lo, e_hi = np.inf, -np.inf
    for r in rows:
        if r["tide_class"] == "event":
            continue
        i = idx_of[int(np.datetime64(r["utc"].replace("Z", ""), "ns").astype("int64"))]
        for ser in (s, ref):
            if ser is None:
                continue
            w = window(ser.eta, i)
            e_lo, e_hi = min(e_lo, np.nanmin(w)), max(e_hi, np.nanmax(w))
    return {
        "eta": (float(e_lo), float(e_hi)),
        "strength_max": float(np.nanmax(np.where(ym, s.strength, np.nan))),
        "hs_max": float(np.nanmax(np.where(ym, s.hs, np.nan))),
        "speed_max": float(np.nanmax(np.where(ym, s.speed, np.nan))),
        "hs_p99": float(np.nanpercentile(np.where(ym, s.hs, np.nan), 99)),
        "speed_p99": float(np.nanpercentile(np.where(ym, s.speed, np.nan), 99)),
    }


def axis_plan(stats: dict[str, dict], variant: str, site: str) -> dict:
    """variant U：五站共用；S：各站自己。"""

    names = list(stats) if variant == "U" else [site]
    out = {}
    out["eta"] = nice_axis(
        min(stats[n]["eta"][0] for n in names), max(stats[n]["eta"][1] for n in names), max_int=4
    )
    out["strength"] = nice_axis(0, max(stats[n]["strength_max"] for n in names), zero_based=True, max_int=5)
    # 總覽的顯著波高、流速兩小圖高度僅約 0.74 英寸，9 pt 刻度字至多容納 3 個區間
    out["hs"] = nice_axis(0, max(stats[n]["hs_max"] for n in names), zero_based=True, min_int=2, max_int=3)
    out["speed"] = nice_axis(0, max(stats[n]["speed_max"] for n in names), zero_based=True, min_int=2, max_int=3)
    out["hs_small"] = nice_axis(0, max(stats[n]["hs_max"] for n in names), zero_based=True, max_int=4)
    out["speed_small"] = nice_axis(0, max(stats[n]["speed_max"] for n in names), zero_based=True, max_int=4)
    out["strength_small"] = out["strength"]
    return out


# ── 共用繪圖小工具 ─────────────────────────────────────────────────────────────


def _setup_mpl():
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["axes.unicode_minus"] = True
    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["axes.linewidth"] = 0.6
    import matplotlib.pyplot  # noqa: F401

    return matplotlib


def _style_axes(
    ax, fonts: Fonts, size: float, *, xlim=None, ylim=None, yticks=None, xticks=None, xticklabels=None
) -> None:
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if yticks is not None:
        ax.set_yticks(yticks)
        ax.set_yticklabels([fmt_tick(v) for v in yticks], fontproperties=fonts(size))
    if xticks is not None:
        ax.set_xticks(xticks)
        ax.set_xticklabels(
            xticklabels if xticklabels is not None else [fmt_tick(v) for v in xticks],
            fontproperties=fonts(size),
        )
    ax.tick_params(length=2.0, width=0.5, pad=1.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def _save(fig, path: Path) -> None:
    with path.open("xb") as f:
        fig.savefig(f, format="png", dpi=DPI)
    with path.with_suffix(".svg").open("xb") as f:
        fig.savefig(f, format="svg")


def _legend_handles(fonts: Fonts, *, guishan: bool, zoom: bool):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    g_season = [
        Patch(facecolor=BAND[k], edgecolor="#999999", linewidth=0.4, label=SEASON_RANGE_ZH[k])
        for k in ac.SEASONS
    ]
    g_class = [
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            ms=5,
            mfc=COL_CLASS[c],
            mec="black",
            mew=0.5,
            label=f"{CLASS_ZH[c]}（強度{'≥' if c == 'spring_proxy' else '<'}季內中位數）",
        )
        for c in ac.CLASSES
    ]
    g_phase = [
        Line2D(
            [], [], ls="", marker=MARK_PHASE[p], ms=5, mfc="#bbbbbb", mec="black", mew=0.5, label=PHASE_ZH[p]
        )
        for p in PHASE_ORDER
    ]
    g_event = [
        Line2D([], [], ls="", marker="*", ms=8, mfc=COL_EVENT[e], mec="black", mew=0.4, label=EVENT_ZH[e])
        for e in ("high_wave_event", "strong_current_event")
    ]
    thr = Line2D([], [], color="black", ls=(0, (4, 2)), lw=0.9, label="季內中位數（大小潮分界）")
    p99 = Line2D([], [], color="#555555", ls=(0, (4, 2)), lw=0.8, label="全年第 99 百分位")
    mx = Line2D([], [], color="#555555", ls=(0, (1, 1.5)), lw=0.9, label="全年最大值")
    curve = Line2D([], [], color="#222222", lw=0.9, label="模式海面水位")
    ref = Line2D([], [], color="#888888", lw=0.8, ls=(0, (3, 1.5)), label="挑選依據序列（另一地點）")
    return g_season, g_class, g_phase, g_event, thr, p99, mx, curve, ref


def _draw_legends(
    fig, fonts: Fonts, *, size: float, y_in: float, canvas: tuple[float, float], zoom: bool, guishan: bool
) -> None:
    g_season, g_class, g_phase, g_event, thr, p99, mx, curve, ref = _legend_handles(
        fonts, guishan=guishan, zoom=zoom
    )
    W, H = canvas
    col1 = ([curve] + ([ref] if guishan else []) + [mx, p99]) if zoom else (g_season + [thr])
    col2 = g_class + g_phase
    col3 = g_event if zoom else (g_event + [p99])
    xs = (0.01, 0.37, 0.72)
    for x, handles in zip(xs, (col1, col2, col3), strict=True):
        leg = fig.legend(
            handles=handles,
            loc="lower left",
            bbox_to_anchor=(x, y_in / H),
            prop=fonts(size),
            frameon=False,
            handlelength=1.6,
            labelspacing=0.28,
            borderaxespad=0,
            handletextpad=0.5,
        )
        fig.add_artist(leg)


# ── 全年總覽 ───────────────────────────────────────────────────────────────────


def month_starts_days() -> tuple[np.ndarray, list[str]]:
    d = [
        (np.datetime64(f"2025-{m:02d}-01") - np.datetime64("2025-01-01")).astype(int) for m in range(1, 13)
    ] + [365]
    return np.array(d, dtype=float), [f"{m}/1" for m in list(range(1, 13)) + [1]]


def season_spans() -> list[tuple[str, float, float]]:
    return [("DJF", 0, 59), ("MAM", 59, 151), ("JJA", 151, 243), ("SON", 243, 334), ("DJF", 334, 365)]


# ── 全年總覽（WP-64：3-4-3 圖 3–7 用；寬＝嵌入 Word 之文字區寬、1:1、圖內文字 9 pt、中英並列）──

FS_OV = 9.0  # 圖內文字字級（pt）。畫布寬 = 嵌入寬，不縮放，故在 Word 裡也是 9 pt。

# 中英並列用語（依據見 WP-64 用語對照表）
BI_SEASON = {"DJF": "冬 Winter", "MAM": "春 Spring", "JJA": "夏 Summer", "SON": "秋 Autumn"}
BI_CLASS = {"spring_proxy": "大潮 Spring tide", "neap_proxy": "小潮 Neap tide"}
CLASS_SHORT = {"spring_proxy": "大潮", "neap_proxy": "小潮"}
BI_PHASE = {
    "fastest_rising": "漲潮最快 Fastest rising",
    "fastest_falling": "退潮最快 Fastest falling",
    "slack_proxy": "近平潮 Near slack",
}
BI_EVENT = {"high_wave_event": "高浪事件 High-wave event", "strong_current_event": "強流事件 Strong-current event"}
BI_MEDIAN = "季內中位數 Seasonal median"
BI_P99 = "全年第 99 百分位 Annual 99th percentile (p99)"
BI_TITLE_STRENGTH = "潮汐強度 Tidal strength（25 小時海面高度 η 均方根 25-h RMS of η, m）"
BI_TITLE_HS = "顯著波高 Significant wave height, Hs（m）"
BI_TITLE_SPEED = "流速 Current speed（m/s）"
BI_XLABEL = "日期 Date（2025 年/月/日，UTC）"
OV_TICK_MONTHS = (1, 3, 5, 7, 9, 11)  # 日期刻度：每兩個月 1 日，再加全年末端（2026/01/01）


MARK_ALPHA = 0.65  # 填色透明度（邊框不透明）：位置相近的標記疊在一起時顏色加深、邊框仍各自可見


def _rgba(color: str, alpha: float):
    from matplotlib.colors import to_rgba

    return to_rgba(color, alpha)


def _ov_counts(rows: list[dict]) -> dict:
    """由 manifest 的 50 筆時刻計算各類數量（不寫死）。"""

    c = {"class": {}, "phase": {}, "event": {}, "season_class": {}}
    for r in rows:
        if r["tide_class"] == "event":
            c["event"][r["phase_or_event"]] = c["event"].get(r["phase_or_event"], 0) + 1
        else:
            c["class"][r["tide_class"]] = c["class"].get(r["tide_class"], 0) + 1
            c["phase"][r["phase_or_event"]] = c["phase"].get(r["phase_or_event"], 0) + 1
            k = (r["season"], r["tide_class"])
            c["season_class"][k] = c["season_class"].get(k, 0) + 1
    c["total"] = len(rows)
    return c


def _ov_legend_sets(rows: list[dict]):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    n = _ov_counts(rows)
    seasons = [Patch(facecolor=BAND[k], edgecolor="#999999", linewidth=0.4, label=BI_SEASON[k]) for k in ac.SEASONS]
    cls = [
        Line2D(
            [], [], ls="", marker="o", ms=7.5, mfc=_rgba(COL_CLASS[c], MARK_ALPHA), mec="#111111", mew=0.7,
            label=f"{BI_CLASS[c]}（{n['class'].get(c, 0)}）",
        )
        for c in ac.CLASSES
    ]
    phase = [
        Line2D(
            [], [], ls="", marker=MARK_PHASE[p], ms=7.5, mfc=_rgba("#bbbbbb", MARK_ALPHA), mec="#111111", mew=0.7,
            label=f"{BI_PHASE[p]}（{n['phase'].get(p, 0)}）",
        )
        for p in PHASE_ORDER
    ]
    event = [
        Line2D(
            [], [], ls="", marker="*", ms=12, mfc=COL_EVENT[e], mec="#111111", mew=0.6,
            label=f"{BI_EVENT[e]}（{n['event'].get(e, 0)}）",
        )
        for e in ("high_wave_event", "strong_current_event")
    ]
    med = Line2D([], [], color="black", ls=(0, (4, 2)), lw=1.0, label=BI_MEDIAN)
    p99 = Line2D([], [], color="#333333", ls=(0, (4, 2)), lw=0.8, label=BI_P99)
    return seasons, [*cls, *phase], [*event, med, p99]


def _ov_make_legends(fig, fonts: Fonts, rows: list[dict]):
    seasons, left_col, right_col = _ov_legend_sets(rows)
    common = dict(
        prop=fonts(FS_OV), frameon=False, handlelength=1.5, labelspacing=0.18, borderaxespad=0, borderpad=0,
        handletextpad=0.5, loc="lower left",
    )
    legs = [
        fig.legend(handles=left_col, **common),
        fig.legend(handles=right_col, **common),
    ]
    for lg in legs:
        fig.add_artist(lg)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    return [(lg, lg.get_window_extent(rend).width / fig.dpi, lg.get_window_extent(rend).height / fig.dpi) for lg in legs]


def _ov_date_ticks() -> tuple[np.ndarray, list[str]]:
    days, _ = month_starts_days()
    pos = [days[m - 1] for m in OV_TICK_MONTHS] + [days[12]]
    lab = [f"2025/{m:02d}/01" for m in OV_TICK_MONTHS] + ["2026/01/01"]
    return np.array(pos, dtype=float), lab


def render_overview(
    site: str, s: ac.SiteSeries, res: dict, fonts: Fonts, axes_plan: dict, variant: str, text: bool, out: Path
) -> None:
    plt = _setup_mpl().pyplot
    W = OV_W
    left, right = 0.52, 0.42
    heights = (2.2, 0.68, 0.68)  # 潮汐強度主圖、顯著波高、流速（英寸）
    title_h, gap = 0.20, 0.05
    season_h = 0.34  # 主圖上方：各季名稱與大潮／小潮數量（兩行）
    xaxis_h = 0.50
    top_margin = 0.05 + (OV_TITLE if text else 0.0)
    rows = res["rows"]
    tmp = plt.figure(figsize=(W, 2.0), dpi=DPI)
    meas = _ov_make_legends(tmp, fonts, rows)
    (w_l, h_l), (w_r, h_r) = [(w, h) for _, w, h in meas]
    plt.close(tmp)
    pad_leg = 0.04
    rows_h = max(h_l, h_r)
    legend_h = pad_leg + rows_h + 0.04
    H = legend_h + xaxis_h + sum(heights) + 3 * title_h + season_h + 2 * gap + top_margin
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    y = legend_h + xaxis_h
    rects = {}
    for key, hh in (("c", heights[2]), ("h", heights[1]), ("s", heights[0])):
        rects[key] = (y, hh)
        y += hh + title_h + gap + (season_h if key == "s" else 0.0) * 0.0
        if key == "h":
            pass
    mk = lambda key: fig.add_axes((left / W, rects[key][0] / H, (W - left - right) / W, rects[key][1] / H))
    ax_c, ax_h, ax_s = mk("c"), mk("h"), mk("s")
    ax_h.sharex(ax_s)
    ax_c.sharex(ax_s)
    days = (s.times.astype(np.int64) - T0) / DAY_NS
    sel = (days >= 0) & (days <= 365)
    for ax in (ax_s, ax_h, ax_c):
        for name, a, b in season_spans():
            ax.axvspan(a, b, facecolor=BAND[name], edgecolor="none", zorder=0)
    ax_s.plot(days[sel], s.strength[sel], color="#444444", lw=0.6, zorder=2)
    for name, a, b in season_spans():
        ax_s.plot([a, b], [res["ind"]["thresholds"][name]] * 2, color="black", lw=1.0, ls=(0, (4, 2)), zorder=3)
    lo, up, st = axes_plan["strength"]
    _style_axes(ax_s, fonts, FS_OV, xlim=(0, 365), ylim=(lo, up), yticks=axis_ticks(lo, up, st))
    ax_s.tick_params(labelbottom=False)
    ax_h.plot(days[sel], s.hs[sel], color="#555555", lw=0.5, zorder=2)
    ax_c.plot(days[sel], s.speed[sel], color="#555555", lw=0.5, zorder=2)
    ym = s.valid & (s.year_of == ac.YEAR)
    for ax, key, ser in ((ax_h, "hs", s.hs), (ax_c, "speed", s.speed)):
        lo, up, st = axes_plan[key]
        ax.plot([0, 365], [np.nanpercentile(np.where(ym, ser, np.nan), 99)] * 2, color="#333333", lw=0.8, ls=(0, (4, 2)), zorder=3)
        _style_axes(ax, fonts, FS_OV, xlim=(0, 365), ylim=(lo, up), yticks=axis_ticks(lo, up, st))
    ax_h.tick_params(labelbottom=False)
    tpos, tlab = _ov_date_ticks()
    ax_c.set_xticks(tpos)
    ax_c.set_xticklabels(tlab, fontproperties=fonts(FS_OV))
    ax_c.set_xlabel(BI_XLABEL, fontproperties=fonts(FS_OV), labelpad=2)
    # 標題：潮汐強度主圖的標題在季節標籤之上
    y0, hh = rects["s"]
    fig.text(left / W, (y0 + hh + season_h + 0.025) / H, BI_TITLE_STRENGTH, ha="left", va="bottom", fontproperties=fonts(FS_OV))
    for key, ttl in (("h", BI_TITLE_HS), ("c", BI_TITLE_SPEED)):
        y0, hh = rects[key]
        fig.text(left / W, (y0 + hh + 0.025) / H, ttl, ha="left", va="bottom", fontproperties=fonts(FS_OV))
    # 各季名稱與該季大潮／小潮數量（標在季節色塊頂端）
    cnt = _ov_counts(rows)
    y0, hh = rects["s"]
    ax_w = W - left - right
    seen = set()
    for name, a, b in season_spans():
        if name in seen:
            continue
        seen.add(name)
        xc = left + ax_w * ((a + b) / 2) / 365.0
        n_sp = cnt["season_class"].get((name, "spring_proxy"), 0)
        n_nm = cnt["season_class"].get((name, "neap_proxy"), 0)
        fig.text(
            xc / W, (y0 + hh + 0.02) / H, f"{BI_SEASON[name]}\n大潮 {n_sp}／小潮 {n_nm}",
            ha="center", va="bottom", fontproperties=fonts(FS_OV), linespacing=1.05,
        )
    # 標記：位置為真實位置；加大、深色細邊框、略透明，重疊時仍可分辨兩個點
    idx_of = res["idx_of"]
    n_drawn = 0
    for r in sorted(rows, key=lambda r: r["utc"]):
        i = idx_of[int(np.datetime64(r["utc"].replace("Z", ""), "ns").astype("int64"))]
        d = days[i]
        if r["tide_class"] == "event":
            ev = r["phase_or_event"]
            col = COL_EVENT[ev]
            for ax in (ax_s, ax_h, ax_c):
                ax.axvline(d, color=col, lw=0.8, ls=(0, (1, 1.5)), zorder=1.5)
            ax_s.plot([d], [s.strength[i]], marker="*", ms=12, mfc=col, mec="#111111", mew=0.6, ls="", zorder=6, clip_on=False)
            ax, ser = (ax_h, s.hs) if ev == "high_wave_event" else (ax_c, s.speed)
            ax.plot([d], [ser[i]], marker="*", ms=12, mfc=col, mec="#111111", mew=0.6, ls="", zorder=6, clip_on=False)
        else:
            ax_s.plot(
                [d], [s.strength[i]], marker=MARK_PHASE[r["phase_or_event"]], ms=7.5,
                mfc=_rgba(COL_CLASS[r["tide_class"]], MARK_ALPHA), mec="#111111", mew=0.7, ls="", zorder=4, clip_on=False,
            )
        n_drawn += 1
    assert n_drawn == cnt["total"], (n_drawn, cnt["total"])
    if text:
        fig.text(left / W, 1 - 0.05 / H, f"{cnt['total']} 個到達時間 arrival times", ha="left", va="top", fontproperties=fonts(FS_OV, "bold"))
    legs = _ov_make_legends(fig, fonts, rows)
    (lg_l, w_l, h_l), (lg_r, w_r, h_r) = legs
    x_l = 0.04
    x_r = x_l + w_l + max(0.25, (W - 0.08 - w_l - w_r) / 2.0)
    lg_l.set_bbox_to_anchor((x_l / W, (pad_leg + (rows_h - h_l)) / H), transform=fig.transFigure)
    lg_r.set_bbox_to_anchor((x_r / W, (pad_leg + (rows_h - h_r)) / H), transform=fig.transFigure)
    _save(fig, out)
    plt.close(fig)


# ── 逐時刻放大 ─────────────────────────────────────────────────────────────────


def render_zoom(
    site: str,
    s: ac.SiteSeries,
    res: dict,
    fonts: Fonts,
    axes_plan: dict,
    stat_site: dict,
    variant: str,
    text: bool,
    out: Path,
    ref: ac.SiteSeries | None = None,
) -> None:
    plt = _setup_mpl().pyplot
    W = ZM_W
    left, pw, gx = 0.58, 1.07, 0.22
    ph, gy = 0.74, 0.40
    legend_h, xlab_h = 0.80, 0.34
    nrows = 9
    top_y = legend_h + xlab_h + nrows * ph + (nrows - 1) * gy  # 最上列上緣離底部
    H = top_y + 0.25 + (ZM_TITLE if text else 0.0)
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    rows = res["rows"]
    idx_of = res["idx_of"]
    by_key: dict[tuple, list] = {}
    events = {}
    for r in rows:
        if r["tide_class"] == "event":
            events[r["phase_or_event"]] = r
        else:
            by_key.setdefault((r["season"], r["tide_class"], r["phase_or_event"]), []).append(r)
    for v in by_key.values():
        v.sort(key=lambda r: r["replicate_rank_manifest"])
    lo, up, st = axes_plan["eta"]
    yt = axis_ticks(lo, up, st)
    xt = np.array([-HALF, 0, HALF], dtype=float)
    hrs = np.arange(-HALF, HALF + 1)
    row_defs = [(sn, cl) for sn in ac.SEASONS for cl in ac.CLASSES]
    for ri, (sn, cl) in enumerate(row_defs):
        for pj, ph_name in enumerate(PHASE_ORDER):
            for k in range(2):
                items = by_key.get((sn, cl, ph_name), [])
                col = pj * 2 + k
                x0 = left + col * (pw + gx)
                y0 = top_y - ri * (ph + gy) - ph
                ax = fig.add_axes((x0 / W, y0 / H, pw / W, ph / H))
                if k >= len(items):
                    ax.axis("off")
                    continue
                r = items[k]
                i = idx_of[int(np.datetime64(r["utc"].replace("Z", ""), "ns").astype("int64"))]
                if lo < 0 < up:
                    ax.axhline(0, color="#cccccc", lw=0.4, zorder=0)
                if ref is not None:
                    ax.plot(
                        hrs,
                        window(ref.eta, i),
                        color="#888888",
                        lw=0.7,
                        ls=(0, (3, 1.5)),
                        zorder=2,
                    )
                ax.plot(hrs, window(s.eta, i), color="#222222", lw=0.8, zorder=3)
                ax.axvline(0, color="#999999", lw=0.4, zorder=1)
                ax.plot(
                    [0],
                    [s.eta[i]],
                    marker=MARK_PHASE[ph_name],
                    ms=4.6,
                    mfc=COL_CLASS[cl],
                    mec="black",
                    mew=0.45,
                    ls="",
                    zorder=5,
                )
                _style_axes(ax, fonts, 5.6, xlim=(-HALF, HALF), ylim=(lo, up), yticks=yt, xticks=xt)
                if text:
                    ex = r["hours_to_nearest_extreme"]
                    l1 = f"{SEASON_ZH[sn]} {CLASS_ZH[cl]} {PHASE_ZH[ph_name]} #{k + 1}"
                    l2 = (
                        r["utc"][5:16].replace("T", " ")
                        + " 世界時"
                        + (f" 距高低潮{ex:.1f}h" if ph_name == "slack_proxy" else "")
                    )
                    ax.set_title(
                        l1 + "\n" + l2, fontproperties=fonts(5.3), loc="left", pad=2, linespacing=1.15
                    )
                if col == 0:
                    ax.set_ylabel("水位（m）", fontproperties=fonts(5.8), labelpad=1)
    # 事件列（第 9 列：各占三格寬）
    ri = 8
    for ci, ev in enumerate(("high_wave_event", "strong_current_event")):
        r = events[ev]
        i = idx_of[int(np.datetime64(r["utc"].replace("Z", ""), "ns").astype("int64"))]
        shift = 0.34 * ci  # 右側事件格的 y 軸標籤與刻度需要較寬的左側空間，不可壓到左格
        x0 = left + ci * 3 * (pw + gx) + shift
        w = 3 * pw + 2 * gx - shift
        y0 = top_y - ri * (ph + gy) - ph
        ax = fig.add_axes((x0 / W, y0 / H, w / W, ph / H))
        ser = s.hs if ev == "high_wave_event" else s.speed
        key = "hs_small" if ev == "high_wave_event" else "speed_small"
        elo, eup, est = axes_plan[key]
        ym = s.valid & (s.year_of == ac.YEAR)
        p99 = float(np.nanpercentile(np.where(ym, ser, np.nan), 99))
        ymax = float(np.nanmax(np.where(ym, ser, np.nan)))
        col = COL_EVENT[ev]
        ax.plot(hrs, window(ser, i), color=col, lw=0.9, zorder=3)
        ax.axvline(0, color="#999999", lw=0.4, zorder=1)
        ax.plot([-HALF, HALF], [ymax] * 2, color="#555555", lw=0.8, ls=(0, (1, 1.5)), zorder=2)
        ax.plot([-HALF, HALF], [p99] * 2, color="#555555", lw=0.8, ls=(0, (4, 2)), zorder=2)
        ax.plot(
            [0], [ser[i]], marker="*", ms=8, mfc=col, mec="black", mew=0.4, ls="", zorder=5, clip_on=False
        )
        _style_axes(
            ax, fonts, 5.6, xlim=(-HALF, HALF), ylim=(elo, eup), yticks=axis_ticks(elo, eup, est), xticks=xt
        )
        ax.set_ylabel(
            "示性波高（m）" if ev == "high_wave_event" else "流速（m/s）", fontproperties=fonts(5.8), labelpad=1
        )
        if text:
            unit = "m" if ev == "high_wave_event" else "m/s"
            ax.set_title(
                f"{EVENT_ZH[ev]}  {r['utc'][:16].replace('T', ' ')} 世界時  峰值 {ser[i]:.2f} {unit}（全年最大值 {ymax:.2f}，全年第 99 百分位 {p99:.2f}）",
                fontproperties=fonts(5.6),
                loc="left",
                pad=2,
            )
    # 共同 x 軸標籤
    fig.text(
        (left + (6 * pw + 5 * gx) / 2) / W,
        (legend_h + 0.0) / H,
        "相對到達（觀測）時刻（h）",
        ha="center",
        va="bottom",
        fontproperties=fonts(6),
    )
    if text:
        fig.text(
            left / W,
            1 - 0.06 / H,
            "50 個到達時刻前後 ±13 h：48 個分層時刻為模式海面水位，2 個事件時刻為挑選所用的波高與流速",
            ha="left",
            va="top",
            fontproperties=fonts(8, "bold"),
        )
    _draw_legends(fig, fonts, size=6.0, y_in=0.04, canvas=(W, H), zoom=True, guishan=ref is not None)
    _save(fig, out)
    plt.close(fig)
