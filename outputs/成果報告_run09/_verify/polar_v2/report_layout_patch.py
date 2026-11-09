# ── 報告版面（WP-65）：圖寬＝報告插入寬度，不縮放；圖例置於主圖下方 ─────────────────────────────
#
# 目的：run09 報告把圓形來源方位圖以 6.27 英寸寬插入 Word。舊版畫布 12.6 × 8.2 英寸（含右側圖例欄），縮放後
# 字級只剩約 4 pt，主圖只占頁寬約一半，且右欄圖例比主圖高。新版面（``PolarConfig.report_layout``）：
#
# * 畫布寬度＝6.27 英寸＝插入寬度，字級 8 pt 即印出字級（不縮放；做法同 TRAPs 報告圖：寬度即版心寬、字級至少 8 pt）；
# * 主圖（地圖）為正方形，幾乎占滿畫布寬度；地圖範圍＝錨點四方各 ``REPORT_WINDOW_KM``（圓半徑的 1.36 倍），
#   五地點的主圖大小、圓半徑與錨點位置完全相同；
# * 圖例全部放在主圖正下方，不再有右側欄，所以主圖與圖例不存在「一高一低」；
# * A 類（每類十個沉降速度）十張圖併成一張 3 欄 × 4 列的多格圖，共用一個圖例（沉降速度序位）。
#
# 只改版面，不改任何數值（方向機率、分組、色階、半徑刻度與 legacy 版完全相同）。

REPORT_W_IN = 6.27
REPORT_WINDOW_FACTOR = 1.36
"""報告版面的地圖視窗半寬＝圓半徑 × 此倍數（方位刻度標籤在 1.095 倍半徑處，仍在框內）。"""
REPORT_TEXT_PT = 8.0
"""報告版面的最小字級（pt）；畫布寬等於插入寬度，所以印出字級＝此值。"""
REPORT_STEMS: Mapping[str, str] = {
    "A": "polar_source_v2_velocity_panels",
    "B": "polar_source_v2_by_class",
    "C": "polar_source_v2_by_season",
    "D": "polar_source_v2_by_tide",
}

# 總覽圖（B 十類、C 季節、D 大小潮）：由上而下為主圖、座標刻度標籤帶、圖例帶。
_RO_LEFT = 0.58
_RO_RIGHT_MARGIN = 0.10
_RO_FRAME = REPORT_W_IN - _RO_LEFT - _RO_RIGHT_MARGIN
_RO_TOP = 0.08
_RO_XLAB = 0.30
_RO_LEGEND_GAP = 0.08
_RO_LEGEND_H = 1.30
_RO_BOTTOM = 0.04
REPORT_OVERVIEW_H = round(_RO_TOP + _RO_FRAME + _RO_XLAB + _RO_LEGEND_GAP + _RO_LEGEND_H + _RO_BOTTOM, 3)

# 多格圖（A）：3 欄 × 4 列，第 10 格在第 4 列第 1 欄，其右兩格放共用圖例。
_RG_COLS = 3
_RG_ROWS = 4
_RG_LEFT = 0.58
_RG_RIGHT_MARGIN = 0.06
_RG_GAP = 0.10
_RG_TITLE = 0.20
_RG_ROWGAP = 0.07
_RG_XLAB = 0.28
_RG_TOP = 0.04
_RG_BOTTOM = 0.04
_RG_P = (REPORT_W_IN - _RG_LEFT - _RG_RIGHT_MARGIN - (_RG_COLS - 1) * _RG_GAP) / _RG_COLS


def _rg_row_top(r: int) -> float:
    """第 r 列（0 起算）標題帶上緣，自畫布頂端起算（英寸）；第 4 列前多留一條 x 軸刻度標籤帶（第 8、9 格在其上方）。"""

    return _RG_TOP + r * (_RG_TITLE + _RG_P + _RG_ROWGAP) + (_RG_XLAB if r == _RG_ROWS - 1 else 0.0)


REPORT_PANELS_H = round(_rg_row_top(_RG_ROWS - 1) + _RG_TITLE + _RG_P + _RG_XLAB + _RG_BOTTOM, 3)


def _report_ticks(lo: float, hi: float, steps: Sequence[float], min_count: int) -> np.ndarray:
    """範圍內規則間距的刻度（度）：由 ``steps`` 依序嘗試，取第一個刻度數不少於 ``min_count`` 者。"""

    for step in steps:
        k0 = math.ceil(lo / step - 1e-9)
        k1 = math.floor(hi / step + 1e-9)
        if k1 - k0 + 1 >= min_count:
            return np.round(np.arange(k0, k1 + 1) * step, 6)
    raise PolarSourceError(f"範圍 {lo:.3f}–{hi:.3f} 找不到至少 {min_count} 個規則刻度")


def _build_report_map(
    anchor: tuple[float, float],
    radius_km: float,
    land_geometries: Sequence[Any],
    site_isobaths: Mapping[float, list[np.ndarray]],
) -> _SiteMap:
    """報告版面的地圖視窗：錨點四方各 ``radius_km × REPORT_WINDOW_FACTOR`` 公里（正方形），底圖裁到視窗內。"""

    import shapely

    lon0, lat0 = anchor
    half = radius_km * REPORT_WINDOW_FACTOR
    dlat = half / _KM_PER_DEG
    dlon = half / (_KM_PER_DEG * math.cos(math.radians(lat0)))
    extent = (lon0 - dlon, lon0 + dlon, lat0 - dlat, lat0 + dlat)
    land: list[Any] = []
    for geom in land_geometries:
        clipped = shapely.clip_by_rect(geom, extent[0], extent[2], extent[1], extent[3])
        if clipped.is_empty:
            continue
        parts = list(clipped.geoms) if hasattr(clipped, "geoms") else [clipped]
        land.extend(p for p in parts if p.geom_type in ("Polygon", "MultiPolygon") and not p.is_empty)
    empty = np.array([])
    return _SiteMap(anchor, radius_km, empty, empty, extent, land, site_isobaths)


def _text_font(font: Any) -> Any:
    f = font.copy()
    f.set_size(REPORT_TEXT_PT)
    return f


def _draw_report_polar(
    ax: Any,
    plan: PolarFigurePlan,
    site_map: _SiteMap,
    font: Any,
    *,
    compact: bool,
    xticks: np.ndarray,
    yticks: np.ndarray,
    label_x: bool,
    label_y: bool,
) -> tuple[list[Any], list[float], list[float]]:
    """在 ``ax``（正方形主圖框）畫底圖、圓與刻度、折線與錨點；回傳（折線圖例柄、實際畫出的等深線水深、全部等深線水深）。

    ``compact`` 為多格圖用的小格：線寬較細、方位刻度只標 0／90／180／270°、半徑刻度只標隔一圈；字級一律 8 pt。
    """

    import matplotlib.patheffects as pe
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter
    from shapely.plotting import plot_polygon

    radius_km = site_map.radius_km
    rmax = plan.radius_max_pct
    f8 = _text_font(font)
    x0, x1, y0, y1 = site_map.extent
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("auto")  # 框為正方形、視窗為正方形公里範圍，圓為正圓
    ax.set_facecolor("white")
    for land in site_map.land:
        plot_polygon(land, ax=ax, add_points=False, facecolor=_LAND_COLOR, edgecolor="black", linewidth=0.4, zorder=2)
    levels = sorted(site_map.isobaths)
    drawn = _draw_isobaths(ax, site_map.isobaths, site_map.extent, levels) if levels else []

    lw_scale = 0.75 if compact else 1.0
    az_ring = np.linspace(0.0, 360.0, 721)
    lo, la = polar_xy_to_lonlat(site_map.anchor, radius_km, az_ring)
    ax.plot(lo, la, color="#222222", linewidth=0.9 * lw_scale, zorder=4)
    for v in plan.radius_ticks_pct:
        if abs(v - rmax) < 1e-9:
            continue
        lo, la = polar_xy_to_lonlat(site_map.anchor, radius_km * v / rmax, az_ring)
        ax.plot(lo, la, color="#555555", linewidth=0.45 * lw_scale, linestyle=(0, (4, 3)), alpha=0.8, zorder=4)
    for az in (0.0, 90.0, 180.0, 270.0):
        lo, la = polar_xy_to_lonlat(site_map.anchor, np.array([0.0, radius_km]), np.array([az, az]))
        ax.plot(lo, la, color="#555555", linewidth=0.45 * lw_scale, alpha=0.8, zorder=4)
    for k in range(ps.N_SECTORS):
        az = ps.SECTOR_WIDTH_DEG * k - ps.SECTOR_WIDTH_DEG / 2.0
        lo, la = polar_xy_to_lonlat(site_map.anchor, np.array([radius_km, radius_km * 1.03]), np.array([az, az]))
        ax.plot(lo, la, color="#222222", linewidth=0.5 * lw_scale, zorder=4)
    az_labels = (0, 90, 180, 270) if compact else tuple(range(0, 360, 30))
    for az in az_labels:
        lo, la = polar_xy_to_lonlat(site_map.anchor, radius_km * (1.12 if compact else 1.095), float(az))
        ax.text(float(lo), float(la), f"{az}°", ha="center", va="center", fontproperties=f8, fontsize=REPORT_TEXT_PT,
                color="#222222", zorder=9)  # fmt: skip
    ring_labels = list(plan.radius_ticks_pct)
    if compact and len(ring_labels) > 3:
        ring_labels = ring_labels[1::2]
    for v in ring_labels:
        lo, la = polar_xy_to_lonlat(site_map.anchor, radius_km * v / rmax, 45.0)
        ax.text(float(lo), float(la), f"{v:g}%", ha="left", va="bottom", fontproperties=f8, fontsize=REPORT_TEXT_PT,
                color="#111111", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.4},
                zorder=9)  # fmt: skip

    centers = np.append(ps.SECTOR_WIDTH_DEG * np.arange(ps.N_SECTORS), 0.0)
    handles: list[Any] = []
    for i, label in enumerate(plan.labels):
        values = np.append(plan.pct[i], plan.pct[i][0])
        lo, la = polar_xy_to_lonlat(site_map.anchor, radius_km * values / rmax, centers)
        ax.plot(
            lo, la, color=plan.colors[i], linestyle=plan.styles[i], linewidth=(0.9 if compact else 1.2), zorder=6,
            path_effects=[pe.withStroke(linewidth=(1.8 if compact else 2.4), foreground="white", alpha=0.65)],
        )  # fmt: skip
        handles.append(Line2D([], [], color=plan.colors[i], linestyle=plan.styles[i], linewidth=1.4, label=label))
    lon0, lat0 = site_map.anchor
    ax.plot(
        lon0, lat0, marker="*", markersize=(8 if compact else 11), markerfacecolor="#ffd92f", markeredgecolor="black",
        markeredgewidth=0.6, linestyle="none", zorder=8,
    )  # fmt: skip

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.xaxis.set_major_formatter(FuncFormatter(_lon_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(_lat_formatter))
    ax.tick_params(axis="both", length=2.5, width=0.6, pad=2, labelsize=REPORT_TEXT_PT)
    ax.tick_params(axis="x", labelbottom=label_x)
    ax.tick_params(axis="y", labelleft=label_y)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontproperties(f8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    return handles, drawn, levels


def _report_base_handles(t: _Labeler, drawn: Sequence[float], levels: Sequence[float]) -> list[Any]:
    """錨點、同心圓與等深線的圖例柄（全中文）。"""

    from matplotlib.lines import Line2D

    extra: list[Any] = [
        Line2D([], [], marker="*", markersize=10, markerfacecolor="#ffd92f", markeredgecolor="black", markeredgewidth=0.6,
               linestyle="none", label=t.t("研究地點錨點（圓心）", "Site anchor (centre)")),
        Line2D([], [], color="#555555", linewidth=0.8, linestyle=(0, (4, 3)), label=t.t("同心圓：方向機率（%）", "Concentric circles: share (%)")),
    ]  # fmt: skip
    for depth in sorted(drawn):
        k = _ISOBATH_LINESTYLES.index(_isobath_linestyle(depth, levels))
        extra.append(
            Line2D([], [], color=_ISOBATH_COLOR, linewidth=0.8, linestyle=_ISOBATH_LINESTYLES[k],
                   label=t.t(f"等深線 {depth:g} m（{_ISOBATH_STYLE_ZH[k]}）", f"Isobath {depth:g} m"))
        )  # fmt: skip
    return extra


def _fig_legend(fig: Any, handles: Sequence[Any], font: Any, *, loc: str, x_in: float, y_top_in: float, h_in: float, w_in: float,
                title: str | None, ncol: int) -> Any:
    """畫布層級圖例：``(x_in, y_top_in)`` 為圖例錨點（自畫布左上角起算，英寸）。"""

    f8 = _text_font(font)
    leg = fig.legend(
        handles=list(handles), loc=loc, bbox_to_anchor=(x_in / w_in, 1.0 - y_top_in / h_in), bbox_transform=fig.transFigure,
        frameon=False, prop=f8, title=title, handlelength=2.2, handletextpad=0.6, labelspacing=0.28, columnspacing=1.2,
        borderaxespad=0.0, borderpad=0.0, ncol=ncol, alignment="left",
    )  # fmt: skip
    if title is not None:
        leg.get_title().set_fontproperties(f8)
    return leg


def _assert_report_layout(fig: Any, legends: Sequence[Any], axes: Sequence[Any], w_in: float, h_in: float, *, what: str) -> None:
    """版面 gate：圖例與座標軸文字不超出畫布，圖例彼此、圖例與主圖框不重疊。"""

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    dpi = fig.dpi
    tol = 1.0  # 像素
    boxes: list[tuple[str, Any]] = []
    for i, leg in enumerate(legends):
        boxes.append((f"legend{i}", leg.get_window_extent(renderer)))
    for ax in axes:
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            if lab.get_visible() and lab.get_text():
                boxes.append(("tick", lab.get_window_extent(renderer)))
    for name, bb in boxes:
        if bb.x0 < -tol or bb.y0 < -tol or bb.x1 > w_in * dpi + tol or bb.y1 > h_in * dpi + tol:
            raise PolarSourceError(f"{what}：{name} 超出畫布（{bb.x0 / dpi:.2f},{bb.y0 / dpi:.2f},{bb.x1 / dpi:.2f},{bb.y1 / dpi:.2f} in）")
    lboxes = [bb for name, bb in boxes if name.startswith("legend")]
    for i in range(len(lboxes)):
        for j in range(i + 1, len(lboxes)):
            if lboxes[i].overlaps(lboxes[j]):
                raise PolarSourceError(f"{what}：圖例 {i} 與 {j} 重疊")
    for ax in axes:
        frame = ax.get_window_extent(renderer)
        for i, bb in enumerate(lboxes):
            if bb.overlaps(frame):
                raise PolarSourceError(f"{what}：圖例 {i} 與主圖框重疊")


def render_report_overview(
    plan: PolarFigurePlan, site_map: _SiteMap, t: _Labeler, font: Any, path: Path, dpi: int, png_only: bool = False
) -> None:
    """報告版面的總覽圖（B 十類、C 季節、D 大小潮）：正方形主圖占滿寬度，圖例全部在主圖正下方。"""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    w, h = REPORT_W_IN, REPORT_OVERVIEW_H
    fig = plt.figure(figsize=(w, h), dpi=150)
    ax = fig.add_axes([_RO_LEFT / w, 1.0 - (_RO_TOP + _RO_FRAME) / h, _RO_FRAME / w, _RO_FRAME / h])
    xt = _report_ticks(site_map.extent[0], site_map.extent[1], (0.1, 0.2, 0.25, 0.5), 5)
    yt = _report_ticks(site_map.extent[2], site_map.extent[3], (0.1, 0.2, 0.25, 0.5), 5)
    handles, drawn, levels = _draw_report_polar(ax, plan, site_map, font, compact=False, xticks=xt, yticks=yt, label_x=True, label_y=True)
    y_leg = _RO_TOP + _RO_FRAME + _RO_XLAB + _RO_LEGEND_GAP
    ncol = 2 if len(handles) > 5 else 1
    leg1 = _fig_legend(fig, handles, font, loc="upper left", x_in=_RO_LEFT, y_top_in=y_leg, h_in=h, w_in=w,
                       title=t.t(plan.legend_title_zh, plan.legend_title_en), ncol=ncol)  # fmt: skip
    leg2 = _fig_legend(fig, _report_base_handles(t, drawn, levels), font, loc="upper right", x_in=_RO_LEFT + _RO_FRAME, y_top_in=y_leg,
                       h_in=h, w_in=w, title=None, ncol=1)  # fmt: skip
    _assert_report_layout(fig, [leg1, leg2], [ax], w, h, what=plan.stem)
    _save_fixed_canvas(fig, path, dpi=dpi, crop_in=None, svg=not png_only)
    plt.close(fig)


def render_report_panels(
    plans: Sequence[PolarFigurePlan], site_map: _SiteMap, t: _Labeler, font: Any, path: Path, dpi: int, png_only: bool = False
) -> None:
    """報告版面的多格圖：每類一格（格內十條線依沉降速度由慢至快），3 欄 × 4 列，共用一個圖例（沉降速度序位）。"""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if len(plans) != 10 or any(p.kind != "A" for p in plans):
        raise PolarSourceError("多格圖需要十張 A 圖計畫（每類一張）")
    w, h = REPORT_W_IN, REPORT_PANELS_H
    p = _RG_P
    col_x = [_RG_LEFT + c * (p + _RG_GAP) for c in range(_RG_COLS)]
    fig = plt.figure(figsize=(w, h), dpi=150)
    f8 = _text_font(font)
    xt = _report_ticks(site_map.extent[0], site_map.extent[1], (0.25, 0.5), 2)
    yt = _report_ticks(site_map.extent[2], site_map.extent[3], (0.25, 0.5), 2)
    axes: list[Any] = []
    drawn: list[float] = []
    levels: list[float] = []
    for idx, plan in enumerate(plans):
        r, c = divmod(idx, _RG_COLS)
        top = _rg_row_top(r) + _RG_TITLE
        ax = fig.add_axes([col_x[c] / w, 1.0 - (top + p) / h, p / w, p / h])
        bottom_most = (r == _RG_ROWS - 1) or (idx + _RG_COLS >= len(plans))
        _, drawn, levels = _draw_report_polar(ax, plan, site_map, font, compact=True, xticks=xt, yticks=yt, label_x=bottom_most, label_y=(c == 0))
        cname = material_display_name_zh(plan.stem.removeprefix("polar_source_velocity_"))
        ax.set_title(f"{cname}　{plan.labels[0]}–{plan.labels[-1]}", loc="left", fontproperties=f8, fontsize=REPORT_TEXT_PT, pad=2.5)
        axes.append(ax)
    ranks = []
    for i in range(10):
        label = f"第 {i + 1}" + ("（最慢）" if i == 0 else "（最快）" if i == 9 else "")
        ranks.append(Line2D([], [], color=plans[0].colors[i], linestyle=plans[0].styles[i], linewidth=1.4, label=label))
    x_leg = col_x[1]
    y_leg = _rg_row_top(_RG_ROWS - 1) + 0.02
    title = t.t("格內十條線之沉降速度序位", "Rank of settling speed") + "\n" + t.t("格標題數字為該類十個沉降速度之範圍（m/s）", "Panel title: range (m/s)")
    leg1 = _fig_legend(fig, ranks, font, loc="upper left", x_in=x_leg, y_top_in=y_leg, h_in=h, w_in=w, title=title, ncol=2)
    leg2 = _fig_legend(fig, _report_base_handles(t, drawn, levels), font, loc="lower left", x_in=x_leg,
                       y_top_in=_rg_row_top(_RG_ROWS - 1) + _RG_TITLE + _RG_P, h_in=h, w_in=w, title=None, ncol=2)  # fmt: skip
    _assert_report_layout(fig, [leg1, leg2], axes, w, h, what="polar_source_v2_velocity_panels")
    _save_fixed_canvas(fig, path, dpi=dpi, crop_in=None, svg=not png_only)
    plt.close(fig)


def _render_report_job(job: Mapping[str, Any]) -> list[str]:
    """子行程：報告版面，畫一個（案例, 地點）的 B、C、D 總覽與 A 多格圖。回傳寫出的 PNG 路徑。"""

    import matplotlib

    matplotlib.use("Agg", force=True)
    font_res = _resolve_font(Path(job["font_path"]) if job["font_path"] else None)
    t = _Labeler(font_res.cjk_available)
    land = _load_coastline(Path(job["coastline"]))
    iso_all, _ = _load_isobaths(Path(job["bathymetry"]) if job["bathymetry"] else None)
    site_map = _build_report_map(tuple(job["anchor"]), job["radius_km"], land, iso_all.get(job["site"], {}))
    out_dir = Path(job["out_dir"])
    written: list[str] = []
    a_plans = [pl for pl in job["plans"] if pl.kind == "A"]
    for plan in job["plans"]:
        if plan.kind == "A":
            continue
        path = out_dir / f"{REPORT_STEMS[plan.kind]}.png"
        render_report_overview(plan, site_map, t, font_res.font_properties, path, job["dpi"])
        written.append(str(path))
    if a_plans:
        path = out_dir / f"{REPORT_STEMS['A']}.png"
        render_report_panels(a_plans, site_map, t, font_res.font_properties, path, job["dpi"])
        written.append(str(path))
    return written
