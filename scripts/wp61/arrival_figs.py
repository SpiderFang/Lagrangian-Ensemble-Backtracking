# ruff: noqa: E501, E402, SIM115
"""WP-61：50 個到達（觀測）時刻的潮位驗證圖（全年總覽＋逐時刻放大）。

圖面規則：不寫地點名稱與站碼（站碼只出現在檔名）；軸上下界取整齊刻度並標數字；中文一律傳入字型物件
（``fontproperties``，不另給 ``fontsize``）；固定畫布（不使用 ``bbox_inches="tight"``）；
有文字版與無文字版版面完全相同，無文字版只是不畫標題、季節文字、逐格說明文字，保留座標軸、刻度、圖例。
y 軸版本：U＝五站共用整齊上下界；S＝各站各自整齊上下界。
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

OV_W = 7.6  # 總覽畫布寬（英寸）；高度由版面決定（見 render_overview）
OV_TITLE = 0.30  # 有文字版上方標題區；無文字版畫布不含此區（圖內位置不變）
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
    out["hs"] = nice_axis(0, max(stats[n]["hs_max"] for n in names), zero_based=True, max_int=5)
    out["speed"] = nice_axis(0, max(stats[n]["speed_max"] for n in names), zero_based=True, max_int=5)
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
    p99 = Line2D([], [], color="#555555", ls=(0, (4, 2)), lw=0.8, label="年 p99")
    mx = Line2D([], [], color="#555555", ls=(0, (1, 1.5)), lw=0.9, label="年最大")
    curve = Line2D([], [], color="#222222", lw=0.9, label="OCM 海面水位")
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


def render_overview(
    site: str, s: ac.SiteSeries, res: dict, fonts: Fonts, axes_plan: dict, variant: str, text: bool, out: Path
) -> None:
    plt = _setup_mpl().pyplot
    left, right = 0.72, 0.18
    legend_h, xlab_h = 0.78, 0.30
    heights = (2.25, 0.72, 0.72)
    gap = 0.07
    top_margin = 0.22
    W = OV_W
    H = legend_h + xlab_h + sum(heights) + 2 * gap + top_margin + (OV_TITLE if text else 0.0)
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    y_bottom = legend_h + xlab_h
    rects = []
    y = y_bottom
    for h in reversed(heights):
        rects.append((left / W, y / H, (W - left - right) / W, h / H))
        y += h + gap
    rects = rects[::-1]
    ax_s = fig.add_axes(rects[0])
    ax_h = fig.add_axes(rects[1], sharex=ax_s)
    ax_c = fig.add_axes(rects[2], sharex=ax_s)
    days = (s.times.astype(np.int64) - T0) / DAY_NS
    sel = (days >= 0) & (days <= 365)
    mticks, mlabels = month_starts_days()
    rows = res["rows"]
    for ax in (ax_s, ax_h, ax_c):
        for name, a, b in season_spans():
            ax.axvspan(a, b, facecolor=BAND[name], edgecolor="none", zorder=0)
    ax_s.plot(days[sel], s.strength[sel], color="#444444", lw=0.55, zorder=2)
    for name, a, b in season_spans():
        ax_s.plot(
            [a, b], [res["ind"]["thresholds"][name]] * 2, color="black", lw=0.9, ls=(0, (4, 2)), zorder=3
        )
    lo, up, st = axes_plan["strength"]
    _style_axes(ax_s, fonts, 6.5, xlim=(0, 365), ylim=(lo, up), yticks=axis_ticks(lo, up, st))
    ax_s.tick_params(labelbottom=False)
    ax_s.set_ylabel("25 h 潮汐強度（m）", fontproperties=fonts(7), labelpad=2)
    ax_h.plot(days[sel], s.hs[sel], color="#555555", lw=0.45, zorder=2)
    ax_c.plot(days[sel], s.speed[sel], color="#555555", lw=0.45, zorder=2)
    ym = s.valid & (s.year_of == ac.YEAR)
    for ax, key, ser in ((ax_h, "hs", s.hs), (ax_c, "speed", s.speed)):
        lo, up, st = axes_plan[key]
        ax.plot(
            [0, 365],
            [np.nanpercentile(np.where(ym, ser, np.nan), 99)] * 2,
            color="#333333",
            lw=0.7,
            ls=(0, (4, 2)),
            zorder=3,
        )
        _style_axes(ax, fonts, 6.5, xlim=(0, 365), ylim=(lo, up), yticks=axis_ticks(lo, up, st))
    ax_h.tick_params(labelbottom=False)
    ax_h.set_ylabel("Hs（m）", fontproperties=fonts(7), labelpad=2)
    ax_c.set_ylabel("流速（m/s）", fontproperties=fonts(7), labelpad=2)
    ax_c.set_xticks(mticks)
    ax_c.set_xticklabels(mlabels, fontproperties=fonts(6.5))
    ax_c.set_xlabel("2025 年（UTC，月/日）", fontproperties=fonts(7), labelpad=1)
    idx_of = res["idx_of"]
    for r in rows:
        i = idx_of[int(np.datetime64(r["utc"].replace("Z", ""), "ns").astype("int64"))]
        d = days[i]
        if r["tide_class"] == "event":
            ev = r["phase_or_event"]
            col = COL_EVENT[ev]
            for ax in (ax_s, ax_h, ax_c):
                ax.axvline(d, color=col, lw=0.7, ls=(0, (1, 1.5)), zorder=1.5)
            ax_s.plot(
                [d],
                [s.strength[i]],
                marker="*",
                ms=8,
                mfc=col,
                mec="black",
                mew=0.4,
                ls="",
                zorder=5,
                clip_on=False,
            )
            ax, ser = (ax_h, s.hs) if ev == "high_wave_event" else (ax_c, s.speed)
            ax.plot(
                [d], [ser[i]], marker="*", ms=9, mfc=col, mec="black", mew=0.4, ls="", zorder=5, clip_on=False
            )
        else:
            ax_s.plot(
                [d],
                [s.strength[i]],
                marker=MARK_PHASE[r["phase_or_event"]],
                ms=4.6,
                mfc=COL_CLASS[r["tide_class"]],
                mec="black",
                mew=0.45,
                ls="",
                zorder=4,
                clip_on=False,
            )
    if text:
        for name, a, b in season_spans():
            ax_s.text(
                (a + b) / 2,
                1.015,
                SEASON_ZH[name],
                transform=ax_s.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontproperties=fonts(7),
            )
        n_sp = sum(1 for r in rows if r["tide_class"] == "spring_proxy")
        n_np = sum(1 for r in rows if r["tide_class"] == "neap_proxy")
        fig.text(
            left / W,
            1 - 0.06 / H,
            f"全年 48 個分層到達時刻（大潮 {n_sp}、小潮 {n_np}）與 2 個事件時刻的潮汐強度、波高、流速",
            ha="left",
            va="top",
            fontproperties=fonts(8.5, "bold"),
        )
    _draw_legends(fig, fonts, size=6.3, y_in=0.04, canvas=(W, H), zoom=False, guishan=False)
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
                        + "Z"
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
            "Hs（m）" if ev == "high_wave_event" else "流速（m/s）", fontproperties=fonts(5.8), labelpad=1
        )
        if text:
            unit = "m" if ev == "high_wave_event" else "m/s"
            ax.set_title(
                f"{EVENT_ZH[ev]}  {r['utc'][:16].replace('T', ' ')}Z  峰值 {ser[i]:.2f} {unit}（年最大 {ymax:.2f}，年 p99 {p99:.2f}）",
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
            "50 個到達時刻前後 ±13 h：48 個分層時刻為 OCM 海面水位，2 個事件時刻為挑選所用的波高與流速",
            ha="left",
            va="top",
            fontproperties=fonts(8, "bold"),
        )
    _draw_legends(fig, fonts, size=6.0, y_in=0.04, canvas=(W, H), zoom=True, guishan=ref is not None)
    _save(fig, out)
    plt.close(fig)
