# ── 全年總覽（WP-64：3-4-3 圖 3–7 用；寬＝嵌入 Word 之文字區寬、1:1、圖內文字 9 pt、中英並列、50 個點編號）──

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
            [], [], ls="", marker="o", ms=7.5, mfc=COL_CLASS[c], mec="#111111", mew=0.7, alpha=0.82,
            label=f"{BI_CLASS[c]}（{n['class'].get(c, 0)}）",
        )
        for c in ac.CLASSES
    ]
    phase = [
        Line2D(
            [], [], ls="", marker=MARK_PHASE[p], ms=7.5, mfc="#bbbbbb", mec="#111111", mew=0.7, alpha=0.82,
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
        fig.legend(handles=seasons, ncol=4, columnspacing=0.9, **{**common, "handlelength": 1.2}),
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
    heights = (2.25, 0.72, 0.72)  # 潮汐強度主圖、顯著波高、流速（英寸）
    title_h, gap = 0.20, 0.05
    season_h = 0.34  # 主圖上方：各季名稱與大潮／小潮數量（兩行）
    xaxis_h = 0.50
    top_margin = 0.05 + (OV_TITLE if text else 0.0)
    rows = res["rows"]
    tmp = plt.figure(figsize=(W, 2.0), dpi=DPI)
    meas = _ov_make_legends(tmp, fonts, rows)
    (w_s, h_s), (w_l, h_l), (w_r, h_r) = [(w, h) for _, w, h in meas]
    plt.close(tmp)
    pad_leg = 0.04
    rows_h = max(h_l, h_r)
    legend_h = pad_leg + rows_h + 0.05 + h_s + 0.04
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
                [d], [s.strength[i]], marker=MARK_PHASE[r["phase_or_event"]], ms=7.5, mfc=COL_CLASS[r["tide_class"]],
                mec="#111111", mew=0.7, alpha=0.82, ls="", zorder=4, clip_on=False,
            )
        n_drawn += 1
    assert n_drawn == cnt["total"], (n_drawn, cnt["total"])
    if text:
        fig.text(left / W, 1 - 0.05 / H, f"{cnt['total']} 個到達時間 arrival times", ha="left", va="top", fontproperties=fonts(FS_OV, "bold"))
    legs = _ov_make_legends(fig, fonts, rows)
    (lg_s, w_s, h_s), (lg_l, w_l, h_l), (lg_r, w_r, h_r) = legs
    x_l = 0.04
    x_r = x_l + w_l + max(0.25, (W - 0.08 - w_l - w_r) / 2.0)
    lg_l.set_bbox_to_anchor((x_l / W, (pad_leg + h_s + 0.05 + (rows_h - h_l)) / H), transform=fig.transFigure)
    lg_r.set_bbox_to_anchor((x_r / W, (pad_leg + h_s + 0.05 + (rows_h - h_r)) / H), transform=fig.transFigure)
    lg_s.set_bbox_to_anchor((x_l / W, pad_leg / H), transform=fig.transFigure)
    _save(fig, out)
    plt.close(fig)


