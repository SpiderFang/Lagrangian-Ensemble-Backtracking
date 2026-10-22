#!/usr/bin/env python3
"""WP-59：重畫 3-4-3 的圖 1（流場範圍總覽）與圖 2（受體抽樣範圍）。

和 run07 新版（流場範圍總覽_新版／受體抽樣範圍_新版）的版面、字型、色彩、線寬、
座標範圍一致（規格取自原 SVG），
改變的只有：

* 抽樣範圍由矩形虛線框改為以研究地點錨點為圓心、半徑 12.5 km（receptor_core_radius_m）的測地線圓
  （可用 --radius-km 改半徑，例如 10；非 12.5 時那不是抽樣範圍，圖例改標「距錨點 N km」）；
* 圖 2 每個地點加畫 5 個受體水平位置（取自 common-input/receptor.json，每個位置 4 個深度，
  同一水平點只畫一點）；
* 圖例：「研究地點錨點」「抽樣範圍」（圖 2 另有「受體水平位置」）；地點名稱是「龜山島」；
* 龜山島使用吸附後的海上錨點（121.939148, 24.850326）。

WP-74（v5，3-4-3 圖 1–2 字級與字型）：圖內全部文字改為 8 pt（TRAPs 報告圖規格；畫布 pt 尺寸＝Word 嵌入尺寸、不縮放，印出即 8 pt），
字型 Noto Sans CJK TC；框線、刻度依 TRAPs 慣例（框線 1.1 pt #222222；刻度朝外、長 3.5、寬 0.8、pad 2、labelpad 2）。
畫布、axes 位置與大小、座標範圍、圖例項目與用語、資料與線寬不變（圖 1 之 axes 與畫布尺寸、圖 2 之版面均與 v4 相同）。
manifest 新增 ``text_audit``（全部文字之字級、字型）、``data_fingerprint``（資料指紋，供與 v4 比對）、``clearance_pt``。

選項：

* ``--radius-km``：圓半徑（預設 12.5＝實際抽樣半徑，圖例「抽樣範圍」）；其他半徑仍是測地線圓、同樣式，
  但圖例改標「距錨點 N km」（因為它不是抽樣範圍）；
* ``--label-points``：圖 2 的 5 個受體水平位置旁加註 1～5（依 horizontal_selection_draw_order）；
* ``--out-prefix``：輸出檔名前綴（預設 343，輸出 343_fig1_circle.* 等）。

受體水平位置永遠取 receptor.json 的實際座標，不移動、不縮放、不裁切，也沒有任何調整點位的選項；
圓外的點照原位置畫，程式只回報圓外點數與距離（不再要求全部在圓內）。

資料來源：海岸線 coastline-taiwan.geojson（含龜山島輪廓）、流場範圍 common-input/domain.json、
錨點 release config、受體 common-input/receptor.json。程式不改任何輸入。

用法::

    PYTHONPATH=src python scripts/wp59_figs_343/make_343_figs.py \\
        --coastline ~/Workspace/lbt-inputs/coastline-taiwan.geojson \\
        --domain-json <suite>/common-input/domain.json \\
        --receptor-json <suite>/common-input/receptor.json \\
        --release-config <suite>/release-configs/release-90d-full_horizon_from_deposition.yaml \\
        --out-dir <輸出資料夾>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PatchCollection  # noqa: E402
from matplotlib.legend_handler import HandlerPatch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Ellipse, Patch, PathPatch  # noqa: E402
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.path import Path as MplPath  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig_audit as fa  # noqa: E402
from lagrangian_backtracking.native_runner import figures as nf  # noqa: E402

SITES = ("gongliao", "guishan", "hsinchu", "nanwan", "lienchiang")
PANEL = dict(zip(SITES, "abcde", strict=True))
RADIUS_M = 10000.0  # release config 的 receptor_core_radius_m（只用來核對設定檔，繪圖半徑看 --radius-km）
DEFAULT_RADIUS_KM = 10.0
LABEL_FS = 8.0
LABEL_OFFSET_PT = 8.0

# ---- 與原圖（run07 新版 SVG）一致的樣式 ----
LAND_FC, LAND_EC, LAND_LW = "#d9d9d9", "#333333", 0.4
GRID_C, GRID_LW = "#e0e0e0", 0.4
FLOW_FC, FLOW_EC, FLOW_ALPHA, FLOW_LW = "#9ecae1", "#3182bd", 0.35, 1.0
STAR_FC = "#ffd92f"
CIRCLE_C = "#1F4E79"
FONT = "Noto Sans CJK TC"

# 圖 1（v4）：長寬依期中報告圖 2-17（期中報告(0814)全.pdf 第 43 頁，印刷頁碼 35）。
# 量測（PDF 內容串流 CTM 與 600 dpi 影像）：整張嵌入影像 3.7500 × 6.2067 in（270.0 × 446.88 pt，749×1241 px，200 ppi）；
# 地圖 axes 框（119–123°E、20–27°N）3.370 × 5.893 in（約 242.6 × 424.3 pt），寬高比 0.5718＝4/7（每度同長）；
# axes 左緣離影像左 0.286 in、右緣離影像右 0.095 in、上緣離影像上 0.050 in、下緣離影像下 0.270 in。
# 本圖 axes 取 242.5 × 424.375 pt（3.368 × 5.894 in；每度 60.625 pt，經緯度同長），與參考圖差 −0.05%／+0.02%。
# 畫布 = axes + 四邊空白；四邊空白取「刻度標籤與中文軸標題的墨跡範圍」再外加約 0.5 pt 之最小餘裕（manifest 記載實測餘裕）
# （字級仍為 8／9 pt、刻度長 3 pt，與 v3 相同；參考圖字級較小，所以空白不可能完全一樣，故整張圖略大於參考圖）。
# 範圍仍為 119–123°E、20–27°N。
FIG1_EXTENT = (119.0, 123.0, 20.0, 27.0)  # lon0, lon1, lat0, lat1
REF_FIG_IN = (3.7500, 6.2067)  # 期中報告圖 2-17 嵌入影像在頁面上的尺寸（in）
REF_AX_IN = (3.370, 5.893)  # 期中報告圖 2-17 axes 框尺寸（in）
FIG1_PT_PER_DEG = 60.625
FIG1_AX_W = FIG1_PT_PER_DEG * (FIG1_EXTENT[1] - FIG1_EXTENT[0])  # 242.5 pt
FIG1_AX_H = FIG1_PT_PER_DEG * (FIG1_EXTENT[3] - FIG1_EXTENT[2])  # 424.375 pt
FIG1_LEFT_MARGIN, FIG1_RIGHT_MARGIN = 26.1, 7.2
FIG1_TOP, FIG1_BOTTOM_MARGIN = 4.0, 25.2
FIG1_PT = (
    FIG1_LEFT_MARGIN + FIG1_AX_W + FIG1_RIGHT_MARGIN,
    FIG1_TOP + FIG1_AX_H + FIG1_BOTTOM_MARGIN,
)
FIG1_BOTTOM = FIG1_TOP + FIG1_AX_H  # axes 下緣（pt，由上往下量）
FIG1_AX = (FIG1_LEFT_MARGIN, FIG1_TOP, FIG1_LEFT_MARGIN + FIG1_AX_W, FIG1_BOTTOM)
FIG1_MEASURE: dict = {}  # fig1() 量得之文字外框離畫布邊距（pt）
FIG1_LAYOUT = {"lw": 0.9, "dash": (3, 1.6), "star_s": 58, "star_lw": 0.4}
# 圖 2：6.2677 × 8.9216 in；3 列 2 欄的 0.5° × 0.5° 方形 axes（各 168.758 pt）＋右下的圖例 axes
FIG2_PT = (451.275591, 642.353386)
FIG2_AX_W = 168.757795
FIG2_COLS = (44.64, 268.117795)
FIG2_ROWS_TOP = (17.28, 230.677795, 444.075591)  # axes 上緣（由上往下量，pt）
FIG2_LEGEND = (268.117795, 444.075591, 436.875591, 548.475591)  # 左、上、右、下（pt，由上往下量）
FIG2_LAYOUT = {"lw": 1.2, "dash": (4, 2), "star_s": 105, "star_lw": 0.6}
FIG2_WINDOWS = {  # (lon0, lon1, lat0, lat1)，與原圖相同
    "gongliao": (121.7, 122.2, 24.9, 25.4),
    "guishan": (121.7, 122.2, 24.6, 25.1),
    "hsinchu": (120.3, 120.8, 24.4, 24.9),
    "nanwan": (120.5, 121.0, 21.7, 22.2),
    "lienchiang": (119.7, 120.2, 25.9, 26.4),
}
FIG2_CELL = {
    "gongliao": (0, 0),
    "guishan": (0, 1),
    "hsinchu": (1, 0),
    "nanwan": (1, 1),
    "lienchiang": (2, 0),
}  # (列, 欄)
RECEPTOR_MS = 5.0
# 刻度標籤與軸標籤的間距（pt）：由原圖 SVG 與重繪圖逐像素比對校正
PANEL_LABEL_FS = 8.0  # 圖 1 的 (a)～(e) 標籤字級（v5：全部文字 8 pt）
TEXT_PT = 8.0  # v5：圖內全部文字字級（pt，TRAPs 規格）
SPINE_LW, SPINE_COLOR = 1.1, "#222222"  # TRAPs 框線
TICK_LEN, TICK_W, TICK_PAD, LABEL_PAD = 3.5, 0.8, 2.0, 2.0  # TRAPs 刻度與標籤間距


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rc(fig_pt, left, top, right, bottom):
    """由上往下的 pt 座標 → matplotlib figure 比例的 [left, bottom, width, height]。"""

    w, h = fig_pt
    return [left / w, 1.0 - bottom / h, (right - left) / w, (bottom - top) / h]


def _split_polygons(geoms):
    out = []
    for g in geoms:
        out.extend(g.geoms if g.geom_type == "MultiPolygon" else [g])
    return out


def _poly_patch(poly):
    verts, codes = [], []
    for ring in [poly.exterior, *poly.interiors]:
        xy = np.asarray(ring.coords)
        verts.append(xy)
        codes.append([MplPath.MOVETO] + [MplPath.LINETO] * (len(xy) - 2) + [MplPath.CLOSEPOLY])
    return PathPatch(MplPath(np.vstack(verts), np.concatenate(codes)))


def _land_collection(polys, window=None):
    patches = []
    for p in polys:
        if window is not None:
            x0, y0, x1, y1 = p.bounds
            if x1 < window[0] or x0 > window[1] or y1 < window[2] or y0 > window[3]:
                continue
        patches.append(_poly_patch(p))
    return PatchCollection(
        patches, facecolor=LAND_FC, edgecolor=LAND_EC, linewidth=LAND_LW, joinstyle="miter", zorder=2
    )


def _style_axes(ax, *, tick_fs=TEXT_PT, label_fs=TEXT_PT, pad=TICK_PAD, labelpad=LABEL_PAD):
    for sp in ax.spines.values():
        sp.set_linewidth(SPINE_LW)
        sp.set_edgecolor(SPINE_COLOR)
    ax.tick_params(length=TICK_LEN, width=TICK_W, labelsize=tick_fs, direction="out", pad=pad)
    ax.grid(True, color=GRID_C, linewidth=GRID_LW)
    ax.set_axisbelow(True)
    ax.set_xlabel("經度 (°E)", fontsize=label_fs, labelpad=labelpad)
    ax.set_ylabel("緯度 (°N)", fontsize=label_fs, labelpad=labelpad)


class CircleHandle(Patch):
    """圖例用的「圓」記號（抽樣範圍）；用子類別是為了只讓這一項套用圓形 handler。"""


def _circle_handler():
    def make(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
        d = height
        return Ellipse((xdescent + width / 2.0, ydescent + height / 2.0), d, d)

    return HandlerPatch(patch_func=make)


def _circle_handle(lw, dash):
    return CircleHandle(facecolor="none", edgecolor=CIRCLE_C, linewidth=lw, linestyle=(0, dash))


RUN09_ANCHORS = {  # run09 母體 release config 之 study_sites[].anchor_lonlat
    "gongliao": (121.9223889, 25.0964444),
    "guishan": (121.939148, 24.850326),
    "hsinchu": (120.45, 24.75),
    "nanwan": (120.770675, 21.92041),
    "lienchiang": (119.95, 26.2),
}


def load_inputs(args):
    anchors = nf._load_receptor_anchors(args.release_config)
    rj = json.loads(args.receptor_json.read_text(encoding="utf-8"))
    prov = rj["provenance"]["random_horizontal_receptors_by_site"]
    for site in SITES:
        lon, lat, _cfg_radius = anchors[site]
        exp = RUN09_ANCHORS[site]
        if abs(lon - exp[0]) > 1e-7 or abs(lat - exp[1]) > 1e-7:
            raise SystemExit(f"{site} 錨點 {(lon, lat)} 與 run09 預期 {exp} 不一致")
        # 實際抽樣半徑以受體清單記載之 radius_m 為準（run09 設定檔之 receptor_core_radius_m 仍為舊值 12500）
        rad = float(prov[site]["stratified_sampling"]["radius_m"])
        if abs(rad - args.radius_km * 1000.0) > 1e-6:
            raise SystemExit(f"{site} 受體清單半徑 {rad} m 與 --radius-km={args.radius_km} 不一致")
        anchors[site] = (lon, lat, rad)
    coast = _split_polygons(nf._load_coastline(args.coastline))
    dom = json.loads(args.domain_json.read_text(encoding="utf-8"))["records"]
    flow = [np.asarray(json.loads(json.dumps(r["geometry"]))["coordinates"][0]) for r in dom]
    rec = json.loads(args.receptor_json.read_text(encoding="utf-8"))["records"]
    pos = {s: {} for s in SITES}
    for r in sorted(rec, key=lambda r: r["metadata"]["horizontal_selection_draw_order"]):
        hid = r["metadata"]["horizontal_receptor_id"]
        d = pos[r["study_site_id"]].setdefault(
            hid,
            {
                "lon": r["lon"],
                "lat": r["lat"],
                "depths_m": [],
                "draw_order": r["metadata"]["horizontal_selection_draw_order"],
            },
        )
        assert abs(d["lon"] - r["lon"]) < 1e-12 and abs(d["lat"] - r["lat"]) < 1e-12
        d["depths_m"].append(r["z_m_positive_up"])
    return anchors, coast, flow, pos


def horizontal_distances(anchors, pos):
    """25 個水平位置到各自錨點的測地線距離。回傳 [(site, 繪製順序 1～5, lon, lat, 距離 km)]。"""

    rows = []
    for site in SITES:
        lon0, lat0, _ = anchors[site]
        ps = sorted(pos[site].values(), key=lambda p: p["draw_order"])
        assert len(ps) == 5, (site, len(ps))
        assert all(len(p["depths_m"]) == 4 for p in ps)
        d = nf.geodesic_distance_km(lon0, lat0, [p["lon"] for p in ps], [p["lat"] for p in ps])
        for i, (p, dk) in enumerate(zip(ps, d, strict=True), start=1):
            rows.append((site, i, p["lon"], p["lat"], float(dk)))
    return rows


def outside_report(rows, radius_km):
    """圓外的水平位置（距離 > 半徑）。只回報，不改位置。"""

    return [r for r in rows if r[4] > radius_km]


def check_in_window(pos):
    """圖 2 的繪圖範圍必須容得下所有點（不裁切、不移動；超出就報錯讓人改視窗）。"""

    for site in SITES:
        x0, x1, y0, y1 = FIG2_WINDOWS[site]
        for p in pos[site].values():
            if not (x0 <= p["lon"] <= x1 and y0 <= p["lat"] <= y1):
                raise SystemExit(f"{site} 的水平位置 {(p['lon'], p['lat'])} 超出圖 2 的繪圖範圍")


def _label_points(ax, site, ps, lon0, lat0):
    """在每個受體點旁加註 1～5：8 個方向挑離其他點、星號、已放數字最遠者（點位置不動）。"""

    fig = ax.figure
    k = 72.0 / fig.dpi  # 螢幕像素 → pt
    to_pt = lambda lon, lat: np.asarray(ax.transData.transform((lon, lat))) * k  # noqa: E731
    pts = [to_pt(p["lon"], p["lat"]) for p in ps]
    star = to_pt(lon0, lat0)
    bb = ax.get_window_extent()
    box = (bb.x0 * k, bb.y0 * k, bb.x1 * k, bb.y1 * k)
    margin = 5.0
    angles = [45, 135, 315, 225, 90, 0, 270, 180]  # 偏好：右上、左上、右下、左下、上、右、下、左
    placed = []
    for i, p in enumerate(ps):
        best, best_score = None, -1e9
        for rank, a in enumerate(angles):
            c = pts[i] + LABEL_OFFSET_PT * np.array([np.cos(np.radians(a)), np.sin(np.radians(a))])
            if not (
                box[0] + margin <= c[0] <= box[2] - margin and box[1] + margin <= c[1] <= box[3] - margin
            ):
                continue
            dists = [np.hypot(*(c - q)) for j, q in enumerate(pts) if j != i]
            dists += [np.hypot(*(c - star)) * 0.8]
            dists += [np.hypot(*(c - q)) for q in placed]
            score = min(dists) - 0.05 * rank
            if score > best_score:
                best, best_score, best_a = c, score, a
        a = best_a
        placed.append(best)
        ax.annotate(
            str(i + 1),
            xy=(p["lon"], p["lat"]),
            xytext=(LABEL_OFFSET_PT * np.cos(np.radians(a)), LABEL_OFFSET_PT * np.sin(np.radians(a))),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=LABEL_FS,
            fontweight="bold",
            color="#000000",
            path_effects=[pe.withStroke(linewidth=2.0, foreground="#ffffff")],
            zorder=8,
        )


def _circle_label(radius_m):
    """圓的圖例文字：run09 之圓即實際抽樣範圍，依實際半徑標示。"""

    return "抽樣範圍"  # 半徑寫在正文與表 2，不放進圖例（與 run08 圖 1／圖 2 之圖例同寬）


def fig1(anchors, coast, flow, out_dir, dpi, radius_m, prefix):
    L = FIG1_LAYOUT
    fig = plt.figure(figsize=(FIG1_PT[0] / 72, FIG1_PT[1] / 72), dpi=dpi)
    ax = fig.add_axes(_rc(FIG1_PT, *FIG1_AX))
    ax.set_xlim(FIG1_EXTENT[0], FIG1_EXTENT[1])
    ax.set_ylim(FIG1_EXTENT[2], FIG1_EXTENT[3])
    ax.set_xticks([119, 120, 121, 122, 123])
    ax.set_yticks([20, 21, 22, 23, 24, 25, 26, 27])
    _style_axes(ax)
    for xy in flow:
        ax.add_patch(
            MplPolygon(xy, closed=True, facecolor=FLOW_FC, alpha=FLOW_ALPHA, edgecolor="none", zorder=1)
        )
    ax.add_collection(_land_collection(coast))
    for xy in flow:
        ax.add_patch(
            MplPolygon(xy, closed=True, facecolor="none", edgecolor=FLOW_EC, linewidth=FLOW_LW, zorder=3)
        )
    west = {"hsinchu", "nanwan"}  # 圖 1 原圖中 (c)(d) 的標籤放在左側
    for site in SITES:
        lon0, lat0, _ = anchors[site]
        c = nf.geodesic_circle_lonlat(lon0, lat0, radius_m)
        ax.plot(
            c[:, 0],
            c[:, 1],
            color=CIRCLE_C,
            linewidth=L["lw"],
            linestyle=(0, L["dash"]),
            dash_capstyle="butt",
            zorder=4,
        )
        ax.scatter(
            [lon0],
            [lat0],
            marker="*",
            s=L["star_s"],
            facecolor=STAR_FC,
            edgecolor="#000000",
            linewidth=L["star_lw"],
            zorder=5,
        )
        edge_lon = c[:, 0].min() if site in west else c[:, 0].max()
        ax.annotate(
            f"({PANEL[site]})",
            xy=(edge_lon, lat0),
            xytext=(-3.0 if site in west else 3.0, 0.0),
            textcoords="offset points",
            ha="right" if site in west else "left",
            va="center",
            fontsize=PANEL_LABEL_FS,
            fontweight="bold",
            color="#000000",
            alpha=0.9,
            path_effects=[pe.withStroke(linewidth=1.8, foreground="#ffffff")],
            zorder=6,
        )
    handles = [
        (
            Patch(facecolor=FLOW_FC, alpha=FLOW_ALPHA, edgecolor="none"),
            Patch(facecolor="none", edgecolor=FLOW_EC, linewidth=FLOW_LW),
        ),
        Line2D(
            [],
            [],
            marker="*",
            linestyle="none",
            markersize=9,
            markerfacecolor=STAR_FC,
            markeredgecolor="#000000",
            markeredgewidth=0.5,
        ),
        _circle_handle(L["lw"], L["dash"]),
    ]
    leg = ax.legend(
        handles,
        ["流場範圍", "研究地點錨點", _circle_label(radius_m)],
        title="圖例",
        loc="lower right",  # v3：放在 axes 右下角（20–21°N 的開放海域），框大小與 v2 相同
        bbox_to_anchor=(FIG1_AX_RIGHT_LEG / FIG1_PT[0], 1.0 - FIG1_LEG_BOTTOM / FIG1_PT[1]),
        bbox_transform=fig.transFigure,
        borderaxespad=0.0,
        fontsize=TEXT_PT,
        title_fontsize=TEXT_PT,
        borderpad=0.75,
        labelspacing=0.8,
        handlelength=1.75,
        handleheight=1.125,
        handletextpad=0.625,
        fancybox=False,
        framealpha=1.0,
        edgecolor="#555555",
        facecolor="#ffffff",
        handler_map={CircleHandle: _circle_handler()},
    )
    leg.get_frame().set_linewidth(0.5)
    # v4：量出圖上所有文字、刻度、圖例的外框（含字體下伸部分）離畫布四邊的距離（pt），供核對沒有被裁切
    fig.canvas.draw()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())  # inch，由左下起算
    FIG1_MEASURE.update(
        {
            "clearance_pt_left_top_right_bottom": [
                round(tb.x0 * 72, 3),
                round(FIG1_PT[1] - tb.y1 * 72, 3),
                round(FIG1_PT[0] - tb.x1 * 72, 3),
                round(tb.y0 * 72, 3),
            ]
        }
    )
    if min(FIG1_MEASURE["clearance_pt_left_top_right_bottom"]) < 0.0:
        raise SystemExit(f"圖 1 的文字或刻度超出畫布：{FIG1_MEASURE}")
    _save(fig, out_dir / f"{prefix}_fig1_circle", dpi)


FIG1_LEG_GAP = 3.672  # 圖例框與 axes 邊的距離（pt；v2 為右緣離 axes 右緣 289.44−285.768）
FIG1_AX_RIGHT_LEG = FIG1_AX[2] - FIG1_LEG_GAP  # 圖例框右緣（pt，由左往右）
FIG1_LEG_BOTTOM = FIG1_BOTTOM - FIG1_LEG_GAP  # 圖例框下緣（pt，由上往下）


def fig2(anchors, coast, pos, out_dir, dpi, radius_m, prefix, label_points):
    L = FIG2_LAYOUT
    fig = plt.figure(figsize=(FIG2_PT[0] / 72, FIG2_PT[1] / 72), dpi=dpi)
    for site in SITES:
        row, col = FIG2_CELL[site]
        left, top = FIG2_COLS[col], FIG2_ROWS_TOP[row]
        ax = fig.add_axes(_rc(FIG2_PT, left, top, left + FIG2_AX_W, top + FIG2_AX_W))
        x0, x1, y0, y1 = FIG2_WINDOWS[site]
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_xticks(np.round(np.arange(x0, x1 + 1e-9, 0.1), 1))
        ax.set_yticks(np.round(np.arange(y0, y1 + 1e-9, 0.1), 1))
        ax.set_xticklabels([f"{v:.1f}" for v in ax.get_xticks()])
        ax.set_yticklabels([f"{v:.1f}" for v in ax.get_yticks()])
        _style_axes(ax)
        ax.add_collection(_land_collection(coast, window=(x0 - 0.1, x1 + 0.1, y0 - 0.1, y1 + 0.1)))
        lon0, lat0, _ = anchors[site]
        c = nf.geodesic_circle_lonlat(lon0, lat0, radius_m)
        ax.plot(
            c[:, 0],
            c[:, 1],
            color=CIRCLE_C,
            linewidth=L["lw"],
            linestyle=(0, L["dash"]),
            dash_capstyle="butt",
            zorder=4,
        )
        ps = sorted(pos[site].values(), key=lambda p: p["draw_order"])
        ax.plot(
            [p["lon"] for p in ps],
            [p["lat"] for p in ps],
            linestyle="none",
            marker="o",
            markersize=RECEPTOR_MS,
            markerfacecolor="#ffffff",
            markeredgecolor="#000000",
            markeredgewidth=0.7,
            zorder=5,
        )
        ax.scatter(
            [lon0],
            [lat0],
            marker="*",
            s=L["star_s"],
            facecolor=STAR_FC,
            edgecolor="#000000",
            linewidth=L["star_lw"],
            zorder=6,
        )
        ax.set_title(f"({PANEL[site]})", fontsize=TEXT_PT, fontweight="bold", pad=3)
        if label_points:
            _label_points(ax, site, ps, lon0, lat0)
    # 圖例 axes（右下）：單位用 pt，由上往下
    x_l, y_t, x_r, y_b = FIG2_LEGEND
    lax = fig.add_axes(_rc(FIG2_PT, x_l, y_t, x_r, y_b))
    W, H = x_r - x_l, y_b - y_t
    lax.set_xlim(0, W)
    lax.set_ylim(H, 0)
    lax.set_xticks([])
    lax.set_yticks([])
    for sp in lax.spines.values():
        sp.set_linewidth(0.5)
        sp.set_edgecolor("#555555")
    lax.set_title("圖例", fontsize=TEXT_PT, pad=4)
    cx, tx, rows = 27.0, 50.627, (20.5, 52.2, 83.9)
    lax.plot(
        [cx],
        [rows[0]],
        marker="*",
        linestyle="none",
        markersize=11,
        markerfacecolor=STAR_FC,
        markeredgecolor="#000000",
        markeredgewidth=0.6,
    )
    lax.add_patch(
        Ellipse(
            (cx, rows[1]),
            13.57,
            13.57,
            facecolor="none",
            edgecolor=CIRCLE_C,
            linewidth=L["lw"],
            linestyle=(0, L["dash"]),
        )
    )
    lax.plot(
        [cx],
        [rows[2]],
        marker="o",
        linestyle="none",
        markersize=RECEPTOR_MS + 1,
        markerfacecolor="#ffffff",
        markeredgecolor="#000000",
        markeredgewidth=0.7,
    )
    for y, text in zip(rows, ("研究地點錨點", _circle_label(radius_m), "受體水平位置"), strict=True):
        lax.text(tx, y + 3.23, text, fontsize=TEXT_PT, ha="left", va="baseline")
    _save(fig, out_dir / f"{prefix}_fig2_circle", dpi)


AUDIT: dict = {}  # 圖名 → {"text_audit", "text_summary", "data_fingerprint", "clearance_pt"}


def _save(fig, stem: Path, dpi: int):
    audit = fa.text_audit(fig)
    cl = fa.clearance_in(fig)
    AUDIT[stem.name] = {
        "text_summary": fa.text_summary(audit),
        "text_audit": audit,
        "data_fingerprint": fa.fingerprint(fig),
        "clearance_pt_left_top_right_bottom": [round(cl[k] * 72, 3) for k in ("left", "top", "right", "bottom")],
        "canvas_in": [round(float(v), 4) for v in fig.get_size_inches()],
    }
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, facecolor="#ffffff")
    fig.savefig(stem.with_suffix(".svg"), facecolor="#ffffff")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coastline", type=Path, required=True)
    ap.add_argument("--domain-json", type=Path, required=True)
    ap.add_argument("--receptor-json", type=Path, required=True)
    ap.add_argument("--release-config", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument(
        "--radius-km",
        type=float,
        default=DEFAULT_RADIUS_KM,
        help="圓半徑（km，測地線圓；預設 12.5＝抽樣範圍，其他值圖例改標距錨點 N km）",
    )
    ap.add_argument(
        "--label-points", action="store_true", help="圖 2 的受體水平位置旁加註 1～5（依 draw order）"
    )
    ap.add_argument("--out-prefix", default="343", help="輸出檔名前綴（預設 343）")
    ap.add_argument("--panel-label-fontsize", type=float, default=None, help="僅供校正字級用")
    args = ap.parse_args(argv)
    if args.panel_label_fontsize is not None:
        global PANEL_LABEL_FS
        PANEL_LABEL_FS = args.panel_label_fontsize
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": [FONT, "DejaVu Sans"],
            "svg.fonttype": "path",
            "svg.hashsalt": "wp59-343",
            "axes.unicode_minus": False,
        }
    )
    if not 0.0 < args.radius_km <= 50.0:
        raise SystemExit(f"--radius-km={args.radius_km} 不合理（須在 0～50 之間）")
    radius_m = args.radius_km * 1000.0
    anchors, coast, flow, pos = load_inputs(args)
    rows = horizontal_distances(anchors, pos)
    check_in_window(pos)
    far = max(rows, key=lambda r: r[4])
    out = outside_report(rows, args.radius_km)
    print(
        f"半徑 {args.radius_km:g} km；25 個水平位置中 {25 - len(out)} 個在圓內、{len(out)} 個在圓外；"
        f"最遠 {far[0]} #{far[1]} = {far[4]:.3f} km"
    )
    for site, i, _lo, _la, d in out:
        print(f"  圓外：{site} #{i}  距離 {d:.3f} km（超出 {d - args.radius_km:.3f} km）")
    fig1(anchors, coast, flow, args.out_dir, args.dpi, radius_m, args.out_prefix)
    fig2(anchors, coast, pos, args.out_dir, args.dpi, radius_m, args.out_prefix, args.label_points)
    manifest = {
        "radius_m": radius_m,
        "radius_km": args.radius_km,
        "label_points": args.label_points,
        "circle_legend_label": _circle_label(radius_m),
        "circle_is_sampling_range": True,
        "circle": "WGS84 測地線，方位角每 1° 一點（pyproj Geod.fwd）",
        "anchors_lonlat": {s: list(anchors[s][:2]) for s in SITES},
        "horizontal_positions": [
            {
                "site": s,
                "index": i,
                "lon": lo,
                "lat": la,
                "distance_km": d,
                "inside_circle": d <= args.radius_km,
            }
            for s, i, lo, la, d in rows
        ],
        "outside_circle": {
            "count": len(out),
            "by_site": {
                s: [{"index": i, "distance_km": d} for s2, i, _lo, _la, d in out if s2 == s]
                for s in SITES
                if any(r[0] == s for r in out)
            },
        },
        "max_distance_km": far[4],
        "fig1_extent_lon0_lon1_lat0_lat1": list(FIG1_EXTENT),
        "fig1_canvas_pt": list(FIG1_PT),
        "fig1_axes_pt_left_top_right_bottom": [round(v, 3) for v in FIG1_AX],
        "fig1_axes_size_in": [round((FIG1_AX[2] - FIG1_AX[0]) / 72, 4), round((FIG1_AX[3] - FIG1_AX[1]) / 72, 4)],
        "fig1_pt_per_degree": round(FIG1_PT_PER_DEG, 4),
        "fig1_canvas_in": [round(FIG1_PT[0] / 72, 4), round(FIG1_PT[1] / 72, 4)],
        "fig1_text_clearance_pt_left_top_right_bottom": FIG1_MEASURE["clearance_pt_left_top_right_bottom"],
        "fig1_ref_midterm_fig2_17_embedded_in": list(REF_FIG_IN),
        "fig1_ref_midterm_fig2_17_axes_in": list(REF_AX_IN),
        "fig1_axes_vs_ref_pct": [
            round(((FIG1_AX[2] - FIG1_AX[0]) / 72 / REF_AX_IN[0] - 1) * 100, 3),
            round(((FIG1_AX[3] - FIG1_AX[1]) / 72 / REF_AX_IN[1] - 1) * 100, 3),
        ],
        "fig1_canvas_vs_ref_pct": [
            round((FIG1_PT[0] / 72 / REF_FIG_IN[0] - 1) * 100, 3),
            round((FIG1_PT[1] / 72 / REF_FIG_IN[1] - 1) * 100, 3),
        ],
        "inputs_sha256": {
            p.name: _sha(p)
            for p in (args.coastline, args.domain_json, args.receptor_json, args.release_config)
        },
        "dpi": args.dpi,
        "text_pt": TEXT_PT,
        "style": {"spine_lw": SPINE_LW, "spine_color": SPINE_COLOR, "tick_len": TICK_LEN, "tick_w": TICK_W, "tick_pad": TICK_PAD, "label_pad": LABEL_PAD, "font": FONT},
        "figures_audit": AUDIT,
    }
    (args.out_dir / f"{args.out_prefix}_figs_circle_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
