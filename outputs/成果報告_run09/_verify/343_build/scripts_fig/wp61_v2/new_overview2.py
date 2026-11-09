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
            [], [], ls="", marker="o", ms=5.5, mfc=COL_CLASS[c], mec="black", mew=0.6,
            label=f"{BI_CLASS[c]}（{n['class'].get(c, 0)}）",
        )
        for c in ac.CLASSES
    ]
    phase = [
        Line2D(
            [], [], ls="", marker=MARK_PHASE[p], ms=5.5, mfc="#bbbbbb", mec="black", mew=0.6,
            label=f"{BI_PHASE[p]}（{n['phase'].get(p, 0)}）",
        )
        for p in PHASE_ORDER
    ]
    event = [
        Line2D(
            [], [], ls="", marker="*", ms=9, mfc=COL_EVENT[e], mec="black", mew=0.4,
            label=f"{BI_EVENT[e]}（{n['event'].get(e, 0)}）",
        )
        for e in ("high_wave_event", "strong_current_event")
    ]
    med = Line2D([], [], color="black", ls=(0, (4, 2)), lw=1.0, label=BI_MEDIAN)
    p99 = Line2D([], [], color="#333333", ls=(0, (4, 2)), lw=0.8, label=BI_P99)
    tot = Line2D([], [], ls="", marker="", label=f"編號 No. 1–{n['total']}（共 Total {n['total']}）")
    return seasons, [*cls, *phase], [*event, med, p99, tot]


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


# ── 編號標籤自動避讓 ──────────────────────────────────────────────────────────────


def _seg_hits_rect(p, q, rect, n=10) -> bool:
    x0, y0, x1, y1 = rect
    for t in np.linspace(0.0, 1.0, n):
        x, y = p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t
        if x0 <= x <= x1 and y0 <= y <= y1:
            return True
    return False


def place_number_labels(
    ax, pts: list[tuple[float, float, int, float]], curve: tuple[np.ndarray, np.ndarray] | None, font_pt: float,
    ppp: float,
) -> list[dict]:
    """標籤避讓：標記不可移動，只決定編號文字的位置（必要時加細引線）。

    ``pts``：(資料 x, 資料 y, 編號, 標記半徑 pt)；座標一律換算成圖面的點（pt）計算。
    不可重疊的對象：其他編號、全部標記（含自己）；盡量避開：曲線取樣點、離標記過遠（需引線）。
    回傳每個編號的中心（pt 座標，圖面左下為原點）與是否需要引線。
    """

    tr = ax.transData
    bb = ax.get_window_extent()
    ax_rect = (bb.x0 / ppp, bb.y0 / ppp, bb.x1 / ppp, bb.y1 / ppp)
    P = [(*(np.array(tr.transform((x, y))) / ppp), n, r) for x, y, n, r in pts]
    if curve is not None:
        cxy = tr.transform(np.column_stack(curve)) / ppp
        cx, cy = cxy[:, 0], cxy[:, 1]
    else:
        cx = cy = np.zeros(0)
    h = 0.74 * font_pt + 1.6
    wd = {n: 0.58 * font_pt * len(str(n)) + 1.8 for _, _, n, _ in P}
    pad = 0.8
    markers = [(x - r - pad, y - r - pad, x + r + pad, y + r + pad) for x, y, _, r in P]

    def rect_of(c, n):
        return (c[0] - wd[n] / 2, c[1] - h / 2, c[0] + wd[n] / 2, c[1] + h / 2)

    def ov(a, b):
        return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])

    gaps = (1.2, 3.5, 7.0, 12.0, 18.0, 26.0, 36.0, 48.0)
    angs = np.deg2rad(np.arange(0, 360, 15.0))

    def candidates(i):
        x, y, n, r = P[i]
        out = []
        for gi, g in enumerate(gaps):
            for a in angs:
                dx = (r + wd[n] / 2 + g) * np.cos(a)
                dy = (r + h / 2 + g) * np.sin(a)
                out.append((gi, (x + dx, y + dy)))
        return out

    cands = [candidates(i) for i in range(len(P))]
    chosen: dict[int, tuple[int, tuple[float, float]]] = {}

    def cost(i, gi, c, others):
        n = P[i][2]
        rect = rect_of(c, n)
        if rect[0] < ax_rect[0] + 0.5 or rect[2] > ax_rect[2] - 0.5 or rect[1] < ax_rect[1] + 0.5 or rect[3] > ax_rect[3] - 0.5:
            return None
        for m in markers:
            if ov(rect, m):
                return None
        for j, (gj, cj) in others.items():
            if ov(rect, rect_of(cj, P[j][2])):
                return None
        k = 0.0
        if cx.size:
            inside = (cx >= rect[0]) & (cx <= rect[2]) & (cy >= rect[1]) & (cy <= rect[3])
            k += 0.5 * float(inside.sum())
        k += 2.0 * gi
        if gi >= 3:  # 引線：不可穿過其他編號或其他標記
            p = (P[i][0], P[i][1])
            for j, (gj, cj) in others.items():
                if _seg_hits_rect(p, c, rect_of(cj, P[j][2])):
                    k += 25.0
            for j, m in enumerate(markers):
                if j != i and _seg_hits_rect(p, c, m, 8):
                    k += 6.0
        return k

    order = sorted(range(len(P)), key=lambda i: -sum(1 for j in range(len(P)) if j != i and abs(P[i][0] - P[j][0]) < 40 and abs(P[i][1] - P[j][1]) < 30))
    for rnd in range(6):
        for i in order:
            others = {j: v for j, v in chosen.items() if j != i}
            best = None
            for gi, c in cands[i]:
                k = cost(i, gi, c, others)
                if k is not None and (best is None or k < best[0]):
                    best = (k, gi, c)
            if best is None:
                raise RuntimeError(f"編號 {P[i][2]} 找不到不重疊的位置")
            chosen[i] = (best[1], best[2])
    res = []
    for i, (x, y, n, r) in enumerate(P):
        gi, c = chosen[i]
        res.append({"n": n, "marker": (x, y), "center": c, "r": r, "w": wd[n], "h": h, "leader": gi >= 3})
    return res


def _draw_number_labels(fig, fonts: Fonts, placed: list[dict], ppp: float, W: float, H: float, zorder=8):
    import matplotlib.patheffects as pe

    inv = fig.transFigure.inverted()
    for d in placed:
        mx, my = d["marker"]
        cx_, cy_ = d["center"]
        if d["leader"]:
            # 引線由標記邊緣到編號外框最近處
            vx, vy = cx_ - mx, cy_ - my
            L = (vx * vx + vy * vy) ** 0.5
            sx, sy = mx + vx / L * d["r"], my + vy / L * d["r"]
            # 編號外框最近點
            ex = min(max(sx, cx_ - d["w"] / 2), cx_ + d["w"] / 2)
            ey = min(max(sy, cy_ - d["h"] / 2), cy_ + d["h"] / 2)
            p0 = inv.transform((sx * ppp, sy * ppp))
            p1 = inv.transform((ex * ppp, ey * ppp))
            fig.add_artist(
                matplotlib_line(p0, p1, fig)
            )
        fig.text(
            *inv.transform((cx_ * ppp, cy_ * ppp)), str(d["n"]), ha="center", va="center",
            fontproperties=fonts(FS_OV - 0.5), color="#111111", zorder=zorder,
            path_effects=[pe.withStroke(linewidth=1.8, foreground="white")],
        )


def matplotlib_line(p0, p1, fig):
    from matplotlib.lines import Line2D

    return Line2D([p0[0], p1[0]], [p0[1], p1[1]], transform=fig.transFigure, color="#444444", lw=0.45, zorder=7)


def render_overview(
    site: str, s: ac.SiteSeries, res: dict, fonts: Fonts, axes_plan: dict, variant: str, text: bool, out: Path
) -> None:
    plt = _setup_mpl().pyplot
    W = OV_W
    left, right = 0.52, 0.42
    heights = (2.05, 0.66, 0.66)  # 潮汐強度主圖、顯著波高、流速（英寸）
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
        sp, np = cnt["season_class"].get((name, "spring_proxy"), 0), cnt["season_class"].get((name, "neap_proxy"), 0)
        fig.text(
            xc / W, (y0 + hh + 0.02) / H, f"{BI_SEASON[name]}\n大潮 {sp}／小潮 {nm if (nm := np) is not None else np}",
            ha="center", va="bottom", fontproperties=fonts(FS_OV), linespacing=1.05,
        )
    # 標記
    idx_of = res["idx_of"]
    order = sorted(rows, key=lambda r: int(np.datetime64(r["utc"].replace("Z", ""), "ns").astype("int64")))
    pts_s, pts_h, pts_c = [], [], []
    for num, r in enumerate(order, start=1):
        i = idx_of[int(np.datetime64(r["utc"].replace("Z", ""), "ns").astype("int64"))]
        d = days[i]
        if r["tide_class"] == "event":
            ev = r["phase_or_event"]
            col = COL_EVENT[ev]
            for ax in (ax_s, ax_h, ax_c):
                ax.axvline(d, color=col, lw=0.8, ls=(0, (1, 1.5)), zorder=1.5)
            ax_s.plot([d], [s.strength[i]], marker="*", ms=9, mfc=col, mec="black", mew=0.4, ls="", zorder=5, clip_on=False)
            pts_s.append((d, s.strength[i], num, 5.0))
            ax, ser, lst = (ax_h, s.hs, pts_h) if ev == "high_wave_event" else (ax_c, s.speed, pts_c)
            ax.plot([d], [ser[i]], marker="*", ms=10, mfc=col, mec="black", mew=0.4, ls="", zorder=5, clip_on=False)
            lst.append((d, ser[i], num, 5.5))
        else:
            ax_s.plot(
                [d], [s.strength[i]], marker=MARK_PHASE[r["phase_or_event"]], ms=5.4, mfc=COL_CLASS[r["tide_class"]],
                mec="black", mew=0.6, ls="", zorder=4, clip_on=False,
            )
            pts_s.append((d, s.strength[i], num, 3.4))
    assert len(pts_s) == len(order) == cnt["total"], (len(pts_s), len(order))
    ppp = fig.dpi / 72.0
    placed = place_number_labels(ax_s, pts_s, (days[sel], s.strength[sel]), FS_OV - 0.5, ppp)
    placed_h = place_number_labels(ax_h, pts_h, (days[sel], s.hs[sel]), FS_OV - 0.5, ppp) if pts_h else []
    placed_c = place_number_labels(ax_c, pts_c, (days[sel], s.speed[sel]), FS_OV - 0.5, ppp) if pts_c else []
    for pl in (placed, placed_h, placed_c):
        _draw_number_labels(fig, fonts, pl, ppp, W, H)
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


