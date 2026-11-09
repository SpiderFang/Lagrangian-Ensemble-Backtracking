# ── 全年總覽（WP-64：3-4-3 圖 3–7 用；寬 6.3 英寸、以 1:1 嵌入 Word，圖內文字 9 pt、中英並列）─────────────

FS_OV = 9.0  # 圖內文字字級（pt）。畫布寬 = 嵌入寬 = 6.3 英寸，不縮放，故在 Word 裡也是 9 pt。
OV_TICK_MONTHS = (1, 3, 5, 7, 9, 11)  # 日期刻度：每兩個月 1 日，再加全年末端（2026/01/01）

# 中英並列用語（依據見 WP-64 用語對照表）
BI_SEASON = {
    "DJF": "冬季 Winter（12–2 月）",
    "MAM": "春季 Spring（3–5 月）",
    "JJA": "夏季 Summer（6–8 月）",
    "SON": "秋季 Autumn（9–11 月）",
}
BI_CLASS = {"spring_proxy": "大潮 Spring tide", "neap_proxy": "小潮 Neap tide"}
BI_PHASE = {
    "fastest_rising": "漲潮最快 Fastest rising",
    "fastest_falling": "退潮最快 Fastest falling",
    "slack_proxy": "近平潮 Near slack",
}
BI_EVENT = {"high_wave_event": "高浪事件 High-wave event", "strong_current_event": "強流事件 Strong-current event"}
BI_MEDIAN = "季內中位數 Seasonal median"
BI_P99 = "全年第 99 百分位 Annual 99th percentile (p99)"
BI_TITLE_STRENGTH = "潮汐強度 Tidal strength（25 小時海面高度 η 均方根 25-h RMS of η）（m）"
BI_TITLE_HS = "顯著波高 Significant wave height, Hs（m）"
BI_TITLE_SPEED = "流速 Current speed（m/s）"
BI_XLABEL = "日期 Date（2025 年/月/日，UTC）"


def _ov_legend_sets(fonts: Fonts):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    seasons = [
        Patch(facecolor=BAND[k], edgecolor="#999999", linewidth=0.4, label=BI_SEASON[k]) for k in ac.SEASONS
    ]
    cls = [
        Line2D([], [], ls="", marker="o", ms=5.5, mfc=COL_CLASS[c], mec="black", mew=0.5, label=BI_CLASS[c])
        for c in ac.CLASSES
    ]
    phase = [
        Line2D(
            [], [], ls="", marker=MARK_PHASE[p], ms=5.5, mfc="#bbbbbb", mec="black", mew=0.5, label=BI_PHASE[p]
        )
        for p in PHASE_ORDER
    ]
    event = [
        Line2D([], [], ls="", marker="*", ms=9, mfc=COL_EVENT[e], mec="black", mew=0.4, label=BI_EVENT[e])
        for e in ("high_wave_event", "strong_current_event")
    ]
    med = Line2D([], [], color="black", ls=(0, (4, 2)), lw=1.0, label=BI_MEDIAN)
    p99 = Line2D([], [], color="#333333", ls=(0, (4, 2)), lw=0.8, label=BI_P99)
    # 左欄：大潮／小潮、季內中位數、p99；右欄：三種潮相與兩種事件
    return seasons, [*cls, med, p99], [*phase, *event]


def _ov_make_legends(fig, fonts: Fonts):
    """建立三組圖例（季節一列、左欄、右欄），回傳 [(legend, 寬 in, 高 in)]，位置由呼叫端再定。"""

    seasons, left_col, right_col = _ov_legend_sets(fonts)
    common = dict(
        prop=fonts(FS_OV),
        frameon=False,
        handlelength=1.5,
        labelspacing=0.18,
        borderaxespad=0,
        borderpad=0,
        handletextpad=0.5,
        loc="lower left",
    )
    legs = [
        fig.legend(handles=seasons, ncol=4, columnspacing=1.0, **common),
        fig.legend(handles=left_col, **common),
        fig.legend(handles=right_col, **common),
    ]
    for lg in legs:
        fig.add_artist(lg)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    out = []
    for lg in legs:
        bb = lg.get_window_extent(rend)
        out.append((lg, bb.width / fig.dpi, bb.height / fig.dpi))
    return out


def _ov_date_ticks() -> tuple[np.ndarray, list[str]]:
    days, _ = month_starts_days()
    pos, lab = [], []
    for m in OV_TICK_MONTHS:
        pos.append(days[m - 1])
        lab.append(f"2025/{m:02d}/01")
    pos.append(days[12])
    lab.append("2026/01/01")
    return np.array(pos, dtype=float), lab


def render_overview(
    site: str, s: ac.SiteSeries, res: dict, fonts: Fonts, axes_plan: dict, variant: str, text: bool, out: Path
) -> None:
    plt = _setup_mpl().pyplot
    W = OV_W
    left, right = 0.52, 0.42
    heights = (2.05, 0.74, 0.74)  # 潮汐強度主圖、顯著波高、流速（英寸）
    title_h, gap = 0.20, 0.05
    xaxis_h = 0.50  # 日期刻度標籤＋軸標題
    top_margin = 0.05 + (OV_TITLE if text else 0.0)
    # 先量圖例大小（以暫時畫布）
    tmp = plt.figure(figsize=(W, 2.0), dpi=DPI)
    meas = _ov_make_legends(tmp, fonts)
    (w_s, h_s), (w_l, h_l), (w_r, h_r) = [(w, h) for _, w, h in meas]
    plt.close(tmp)
    pad_leg = 0.04
    rows_h = max(h_l, h_r)
    legend_h = pad_leg + rows_h + 0.05 + h_s + 0.04
    H = legend_h + xaxis_h + sum(heights) + 3 * title_h + 2 * gap + top_margin
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    # 由下而上：流速、顯著波高、潮汐強度
    y = legend_h + xaxis_h
    rects = {}
    for key, h in (("c", heights[2]), ("h", heights[1]), ("s", heights[0])):
        rects[key] = (y, h)
        y += h + title_h + gap
    mk = lambda key: fig.add_axes((left / W, rects[key][0] / H, (W - left - right) / W, rects[key][1] / H))
    ax_c = mk("c")
    ax_h = mk("h")
    ax_s = mk("s")
    ax_h.sharex(ax_s)
    ax_c.sharex(ax_s)
    days = (s.times.astype(np.int64) - T0) / DAY_NS
    sel = (days >= 0) & (days <= 365)
    rows = res["rows"]
    for ax in (ax_s, ax_h, ax_c):
        for name, a, b in season_spans():
            ax.axvspan(a, b, facecolor=BAND[name], edgecolor="none", zorder=0)
    ax_s.plot(days[sel], s.strength[sel], color="#444444", lw=0.6, zorder=2)
    for name, a, b in season_spans():
        ax_s.plot(
            [a, b], [res["ind"]["thresholds"][name]] * 2, color="black", lw=1.0, ls=(0, (4, 2)), zorder=3
        )
    lo, up, st = axes_plan["strength"]
    _style_axes(ax_s, fonts, FS_OV, xlim=(0, 365), ylim=(lo, up), yticks=axis_ticks(lo, up, st))
    ax_s.tick_params(labelbottom=False)
    ax_h.plot(days[sel], s.hs[sel], color="#555555", lw=0.5, zorder=2)
    ax_c.plot(days[sel], s.speed[sel], color="#555555", lw=0.5, zorder=2)
    ym = s.valid & (s.year_of == ac.YEAR)
    for ax, key, ser in ((ax_h, "hs", s.hs), (ax_c, "speed", s.speed)):
        lo, up, st = axes_plan[key]
        ax.plot(
            [0, 365],
            [np.nanpercentile(np.where(ym, ser, np.nan), 99)] * 2,
            color="#333333",
            lw=0.8,
            ls=(0, (4, 2)),
            zorder=3,
        )
        _style_axes(ax, fonts, FS_OV, xlim=(0, 365), ylim=(lo, up), yticks=axis_ticks(lo, up, st))
    ax_h.tick_params(labelbottom=False)
    tpos, tlab = _ov_date_ticks()
    ax_c.set_xticks(tpos)
    ax_c.set_xticklabels(tlab, fontproperties=fonts(FS_OV))
    ax_c.set_xlabel(BI_XLABEL, fontproperties=fonts(FS_OV), labelpad=2)
    for key, ttl in (("s", BI_TITLE_STRENGTH), ("h", BI_TITLE_HS), ("c", BI_TITLE_SPEED)):
        y0, h = rects[key]
        fig.text(
            left / W, (y0 + h + 0.025) / H, ttl, ha="left", va="bottom", fontproperties=fonts(FS_OV)
        )
    idx_of = res["idx_of"]
    for r in rows:
        i = idx_of[int(np.datetime64(r["utc"].replace("Z", ""), "ns").astype("int64"))]
        d = days[i]
        if r["tide_class"] == "event":
            ev = r["phase_or_event"]
            col = COL_EVENT[ev]
            for ax in (ax_s, ax_h, ax_c):
                ax.axvline(d, color=col, lw=0.8, ls=(0, (1, 1.5)), zorder=1.5)
            ax_s.plot(
                [d], [s.strength[i]], marker="*", ms=9, mfc=col, mec="black", mew=0.4, ls="", zorder=5, clip_on=False
            )
            ax, ser = (ax_h, s.hs) if ev == "high_wave_event" else (ax_c, s.speed)
            ax.plot(
                [d], [ser[i]], marker="*", ms=10, mfc=col, mec="black", mew=0.4, ls="", zorder=5, clip_on=False
            )
        else:
            ax_s.plot(
                [d],
                [s.strength[i]],
                marker=MARK_PHASE[r["phase_or_event"]],
                ms=5.2,
                mfc=COL_CLASS[r["tide_class"]],
                mec="black",
                mew=0.5,
                ls="",
                zorder=4,
                clip_on=False,
            )
    if text:
        n_sp = sum(1 for r in rows if r["tide_class"] == "spring_proxy")
        n_np = sum(1 for r in rows if r["tide_class"] == "neap_proxy")
        fig.text(
            left / W,
            1 - 0.05 / H,
            f"全年 48 個分層到達時刻（大潮 {n_sp}、小潮 {n_np}）與 2 個事件時刻 48 stratified arrival times (spring {n_sp}, neap {n_np}) and 2 event times",
            ha="left",
            va="top",
            fontproperties=fonts(FS_OV, "bold"),
        )
    # 圖例：下方緊湊；季節一列在最下，其上為左右兩欄
    plt.close(fig) if False else None
    legs = _ov_make_legends(fig, fonts)
    (lg_s, w_s, h_s), (lg_l, w_l, h_l), (lg_r, w_r, h_r) = legs
    x_l = 0.04
    x_r = x_l + w_l + max(0.25, (W - 0.08 - w_l - w_r) / 2.0)
    lg_l.set_bbox_to_anchor((x_l / W, (pad_leg + h_s + 0.05 + (rows_h - h_l)) / H), transform=fig.transFigure)
    lg_r.set_bbox_to_anchor((x_r / W, (pad_leg + h_s + 0.05 + (rows_h - h_r)) / H), transform=fig.transFigure)
    lg_s.set_bbox_to_anchor((x_l / W, pad_leg / H), transform=fig.transFigure)
    _save(fig, out)
    plt.close(fig)


