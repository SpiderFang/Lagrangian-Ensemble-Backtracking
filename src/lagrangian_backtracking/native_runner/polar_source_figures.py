"""圓形來源方位圖（WP-60）的出圖、輸出表、摘要與 manifest。

圖的定義（PI 構想）
------------------
以研究地點的錨點為圓心，在有海岸線、等深線與經緯度軸的地圖上疊一個圓（半徑 ``radius_km``，預設 25 km，
在地圖上是真正的圓，因為地圖長寬比已依緯度校正）；圓分成 36 個各寬 10° 的扇區，扇區以 0°、10°、…、350°
為中心。每個扇區的值是「來源方位落在該扇區的虛擬質點占該組全部虛擬質點的比例（%）」，36 個值依序連成
**封閉的極座標折線**（半徑 = 比例，不是玫瑰圖的扇形長條）。折線的半徑與比例的對照由同心圓與半徑刻度
（如 5%、10%、…）給出。來源方位的計算、擱淺與其他終止類別的處理見 :mod:`polar_source`。

每案每地點 13 張圖（含文字版與無文字版各一套，各自輸出 PNG 與 SVG）：

* A（10 張）：每個沉降速度類別一張，圖上 10 條線是該類 10 個候選沉降速度（每條分母 1,000 顆）；
* B（1 張）：十類各一條線（每條分母 1 萬顆）；
* C（1 張）：依沉底時刻的季節分線；
* D（1 張）：依沉底時刻的大潮／小潮分線。

圖面規則
--------
* 圖面、圖例、標題**不寫地點名稱，也不寫英文站碼**；站碼只出現在檔名（地點子目錄與檔名）。
* 經度、緯度軸的上下界一律有刻度數字，軸範圍取刻度的整齊倍數（0.1°、0.2°、0.25°、0.5°、1°）。
* 半徑刻度（比例，%）在五個地點、兩個案例使用同一組整齊上限與刻度，使各地點視覺一致；
  上限依各圖種全部地點、全部案例的實際最大值向上取整（見 :func:`plan_radius_scales`）。
* 含中文的刻度、圖例、文字一律傳入字型物件（``fontproperties`` 置於 ``fontsize`` 之前）。
* 用詞：「比例（%）」「虛擬質點」「研究地點」；不使用「機率」。
* 無文字版不畫標題與副標，保留經緯度刻度、方位刻度、半徑刻度與圖例。
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..outputs import sha256_file
from . import polar_source as ps
from .display_names import MATERIAL_DISPLAY_NAME_ZH, material_display_name_zh
from .figures import (
    SITE_LABELS_ZH,
    _add_two_line_title,
    _axis_ticks,
    _describe_experiment_case,
    _draw_isobaths,
    _ISOBATH_COLOR,
    _ISOBATH_LINESTYLES,
    _ISOBATH_LINEWIDTH,
    _ISOBATH_STYLE_ZH,
    _isobath_linestyle,
    _Labeler,
    _land_lonlat_patches,
    _lat_formatter,
    _load_coastline,
    _load_isobaths,
    _load_receptor_anchors,
    _lon_formatter,
    _nice_ceiling_axis,
    _resolve_font,
    _save_exclusive,
    _tick_font_kwargs,
    _title_top_margin,
)
from .polar_source import PolarParams, PolarSourceError, SiteAnalysis

__all__ = [
    "FIGURE_KINDS",
    "PolarCase",
    "PolarConfig",
    "PolarFigurePlan",
    "build_polar_source",
    "plan_radius_scales",
    "figure_stem",
    "polar_xy_to_lonlat",
    "build_figure_plans",
    "render_figure",
    "summary_markdown",
]

FIGURE_KINDS: tuple[str, ...] = ("A", "B", "C", "D")
"""圖種：A 類別內十速度、B 十類、C 季節、D 大小潮。"""

_KIND_GROUPING: Mapping[str, str] = {"A": "class_velocity", "B": "class", "C": "season", "D": "tide"}
_KM_PER_DEG = 111.195
"""每度緯度的公里數（地球半徑 6371.0088 km）；地圖上把圓與折線的公里距離換成經緯度差用。"""

_SEASON_COLORS = {"DJF": "#0072B2", "MAM": "#009E73", "JJA": "#D55E00", "SON": "#CC79A7"}
_TIDE_COLORS = {"spring_proxy": "#D55E00", "neap_proxy": "#0072B2", "unknown": "#7f7f7f"}
_LINE_STYLES: tuple[Any, ...] = ("-", "--", "-.", ":", (0, (5, 1, 1, 1, 1, 1)), (0, (1, 1.2)), (0, (6, 2)), (0, (3, 1, 1, 1)))
"""折線線型循環：實線、虛線、點虛線、點線與三種自訂虛線；與色階並用，使黑白列印仍可區辨。"""


def figure_stem(kind: str, material_id: str | None = None) -> str:
    """圖種（與 A 圖的類別）→ 圖檔名（不含副檔名）。"""

    if kind == "A":
        if not material_id:
            raise PolarSourceError("A 圖需要 material_id")
        return f"polar_source_velocity_{material_id}"
    if kind == "B":
        return "polar_source_by_class"
    if kind == "C":
        return "polar_source_by_season"
    if kind == "D":
        return "polar_source_by_tide"
    raise PolarSourceError(f"未知的圖種：{kind!r}")


# ── 設定 ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class PolarCase:
    """一個物理案例：後處理目錄（只讀）、含文字版與無文字版的輸出目錄（必須尚不存在）。"""

    case_id: str
    postprocess_dir: Path
    text_dir: Path
    notext_dir: Path

    def __post_init__(self) -> None:
        if not self.case_id:
            raise PolarSourceError("case_id 不可為空")
        for name in ("postprocess_dir", "text_dir", "notext_dir"):
            object.__setattr__(self, name, Path(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class PolarConfig:
    """一次執行的完整設定。

    ``release_config``：正式釋放設定 YAML（取錨點）。``coastline``、``bathymetry``：海岸線與等深線 GeoJSON。
    ``radius_km``：圓的半徑（公里）。``radius_max_pct``：若給定，所有圖種共用這一個半徑上限（%）；
    省略時 A、B、D 與 C 各依實際最大值取整（見 :func:`plan_radius_scales`）。
    ``sites`` / ``only_cases`` / ``kinds`` / ``classes``：只畫指定子集（不影響半徑上限的計算，
    試作與全量因此用同一組刻度）。``modes``：``full`` 含文字、``none`` 無文字。``write_tables``：是否輸出表與摘要。
    """

    cases: tuple[PolarCase, ...]
    release_config: Path
    coastline: Path
    bathymetry: Path | None
    radius_km: float = 25.0
    radius_max_pct: float | None = None
    sites: tuple[str, ...] | None = None
    only_cases: tuple[str, ...] | None = None
    kinds: tuple[str, ...] = FIGURE_KINDS
    classes: tuple[str, ...] | None = None
    modes: tuple[str, ...] = ("full", "none")
    params: PolarParams = field(default_factory=PolarParams)
    font_path: Path | None = None
    dpi: int = 600
    workers: int = 8
    write_tables: bool = True

    def __post_init__(self) -> None:
        if not self.cases:
            raise PolarSourceError("至少需要一個案例")
        if len({c.case_id for c in self.cases}) != len(self.cases):
            raise PolarSourceError("case_id 不可重複")
        bad = [k for k in self.kinds if k not in FIGURE_KINDS]
        if bad or not self.kinds:
            raise PolarSourceError(f"kinds 必須是 {FIGURE_KINDS} 的非空子集")
        badm = [m for m in self.modes if m not in ("full", "none")]
        if badm or not self.modes:
            raise PolarSourceError("modes 必須是 full／none 的非空子集")
        if not math.isfinite(self.radius_km) or self.radius_km <= 0:
            raise PolarSourceError("radius_km 必須是有限正數")
        if self.radius_max_pct is not None and not (math.isfinite(self.radius_max_pct) and self.radius_max_pct > 0):
            raise PolarSourceError("radius_max_pct 必須是有限正數")
        if self.dpi < 1 or self.workers < 1:
            raise PolarSourceError("dpi 與 workers 必須是正整數")


# ── 半徑刻度：各圖種共用、跨地點跨案例一致 ───────────────────────────────


def plan_radius_scales(
    analyses_by_case: Mapping[str, Sequence[SiteAnalysis]], *, override_max_pct: float | None = None
) -> dict[str, tuple[float, list[float]]]:
    """決定各圖種的半徑上限與刻度，回傳 ``{圖種: (上限 %, [刻度 %])}``。

    規則：同一圖種在全部地點、全部案例共用一組整齊的上限與刻度（最多 6 個間隔，間距取 1、2、2.5、5 的 10 的冪），
    上限由該圖種全部折線的最大扇區比例乘 1.05 後向上取整。A（類別內十速度）、B（十類）、D（大小潮）的最大值相近，
    共用同一個上限；C（季節）的單一扇區比例可高出一倍以上，若與其他圖種共用同一上限，其餘圖的折線會縮得太小，
    因此 C 單獨取一個上限。``override_max_pct`` 給定時，四種圖一律使用該上限。
    """

    peaks: dict[str, float] = {}
    for kind, grouping in _KIND_GROUPING.items():
        peaks[kind] = max(
            float(an.group_sets[grouping].pct_sectors().max()) for ans in analyses_by_case.values() for an in ans
        )

    def axis(value: float) -> tuple[float, list[float]]:
        top, ticks = _nice_ceiling_axis(value * 1.05, max_intervals=6)
        return float(top), [float(v) for v in ticks if v > 0]

    if override_max_pct is not None:
        top, ticks = axis(override_max_pct / 1.05)
        return {k: (top, ticks) for k in FIGURE_KINDS}
    shared = axis(max(peaks["A"], peaks["B"], peaks["D"]))
    season = axis(peaks["C"])
    return {"A": shared, "B": shared, "D": shared, "C": season}


# ── 幾何：極座標 → 經緯度 ────────────────────────────────────────────────


def polar_xy_to_lonlat(
    anchor_lonlat: tuple[float, float], radius_km: float | np.ndarray, azimuth_deg: float | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """以錨點為原點的局部平面：距離（km）與方位角（自正北順時針，度）→ 經度、緯度（度）。

    只用於畫圖：圖面長寬比已依錨點緯度校正（``1 / cos(緯度)``），所以這樣畫出的圓在螢幕上是正圓。
    """

    lon0, lat0 = anchor_lonlat
    az = np.radians(np.asarray(azimuth_deg, dtype=np.float64))
    r = np.asarray(radius_km, dtype=np.float64)
    lat = lat0 + r * np.cos(az) / _KM_PER_DEG
    lon = lon0 + r * np.sin(az) / (_KM_PER_DEG * math.cos(math.radians(lat0)))
    return lon, lat


# ── 圖面計畫（可序列化給子行程）─────────────────────────────────────────


@dataclass(slots=True)
class PolarFigurePlan:
    """一張圖需要的全部內容（純 Python／numpy 資料，可傳給子行程繪製）。"""

    kind: str
    stem: str
    title_zh: str
    title_en: str
    subtitle_zh: str
    subtitle_en: str
    legend_title_zh: str
    legend_title_en: str
    labels: list[str]
    colors: list[Any]
    styles: list[Any]
    pct: np.ndarray  # (線數, 36)
    n_totals: list[int]
    radius_max_pct: float
    radius_ticks_pct: list[float]
    stranded_lonlat: np.ndarray  # (n, 2)
    caption_zh: str


def _line_colors(kind: str, group_ids: Sequence[str]) -> list[Any]:
    from matplotlib import cm

    if kind == "C":
        return [_SEASON_COLORS.get(g, "#7f7f7f") for g in group_ids]
    if kind == "D":
        return [_TIDE_COLORS.get(g, "#7f7f7f") for g in group_ids]
    k = len(group_ids)
    return [cm.viridis(0.88 * (i / max(k - 1, 1))) for i in range(k)]


def _coverage(an: SiteAnalysis, grouping: str, members: Sequence[int]) -> tuple[int, float, float, float]:
    gs = an.group_sets[grouping]
    sel = np.isin(gs.group_code, list(members))
    n = int(sel.sum())
    kinds = an.sample.kind[sel]
    return (
        n,
        100.0 * float(np.count_nonzero(kinds == ps.KIND_CROSSED)) / n,
        100.0 * float(np.count_nonzero(kinds == ps.KIND_STRANDED)) / n,
        100.0 * float(np.count_nonzero(kinds == ps.KIND_OTHER)) / n,
    )


def _thousands(n: int) -> str:
    return f"{n:,}"


def build_figure_plans(
    an: SiteAnalysis,
    case_text: str,
    scales: Mapping[str, tuple[float, list[float]]],
    *,
    kinds: Sequence[str] = FIGURE_KINDS,
    classes: Sequence[str] | None = None,
) -> list[PolarFigurePlan]:
    """為單站建立圖面計畫（A 每類一張，B、C、D 各一張）。"""

    plans: list[PolarFigurePlan] = []
    site_label = SITE_LABELS_ZH.get(an.site_id, an.site_id)
    for kind in kinds:
        grouping = _KIND_GROUPING[kind]
        gs = an.group_sets[grouping]
        pct_all = gs.pct_sectors()
        top, ticks = scales[kind]
        if kind == "A":
            materials = list(dict.fromkeys(an.group_sets["class"].group_ids))
            if classes is not None:
                materials = [m for m in materials if m in classes]
            subsets = [(m, [i for i, mm in enumerate(gs.material_id) if mm == m]) for m in materials]
        else:
            subsets = [("", list(range(len(gs.group_ids))))]
        for material, members in subsets:
            n, crossed, stranded, other = _coverage(an, grouping, members)
            ids = [gs.group_ids[i] for i in members]
            if kind == "A":
                labels = [gs.labels_zh[i] for i in members]
                cname = material_display_name_zh(material)
                title_zh = f"來源方位的比例｜{cname}｜十個沉降速度分線"
                title_en = f"Source bearing share | {material} | by settling speed"
                legend_zh, legend_en = "沉降速度（m/s）", "Settling speed (m/s)"
                per_line = f"每條線的分母為 {_thousands(int(gs.n_total[members[0]]))} 顆虛擬質點"
                scope = f"類別：{cname}"
            elif kind == "B":
                labels = [
                    f"{gs.labels_zh[i]}　{ps._speed_label(gs.speed_min_mps[i])}–{ps._speed_label(gs.speed_max_mps[i])} m/s"
                    for i in members
                ]
                title_zh, title_en = "來源方位的比例｜依沉降速度類別分線", "Source bearing share | by class"
                legend_zh, legend_en = "類別　沉降速度範圍", "Class and settling speed range"
                per_line = f"每條線的分母為 {_thousands(int(gs.n_total[members[0]]))} 顆虛擬質點"
                scope = "全部十類"
            elif kind == "C":
                labels = [f"{gs.labels_zh[i]}　{_thousands(int(gs.n_total[i]))} 顆" for i in members]
                title_zh, title_en = "來源方位的比例｜依沉底時刻的季節分線", "Source bearing share | by season"
                legend_zh, legend_en = "沉底時刻的季節　虛擬質點數", "Season at deposition"
                per_line = "每條線的分母為該季節的全部虛擬質點"
                scope = "全部虛擬質點依季節分組"
            else:
                labels = [f"{gs.labels_zh[i]}　{_thousands(int(gs.n_total[i]))} 顆" for i in members]
                title_zh, title_en = "來源方位的比例｜依沉底時刻的大潮／小潮分線", "Source bearing share | by tide"
                legend_zh, legend_en = "沉底時刻的潮況　虛擬質點數", "Tide class at deposition"
                per_line = "每條線的分母為該潮況的全部虛擬質點"
                scope = "全部虛擬質點依大小潮分組"
            subtitle_zh = (
                f"圓心為研究地點錨點｜36 個 10° 扇區，正北扇區為 355°–5°，方位自正北順時針｜"
                f"半徑為來源方位落在該扇區的比例（%）\n{per_line}｜扇區合計為穿越開放邊界與擱淺：穿越 {crossed:.1f}%、"
                f"擱淺 {stranded:.1f}%，其餘 {other:.1f}% 未計入扇區（回溯達上限、沉底等）"
            )
            subtitle_en = (
                f"Centre: site anchor | 36 sectors of 10 deg, bearing clockwise from north | radius: share in sector (%)\n"
                f"crossed {crossed:.1f}%, stranded {stranded:.1f}%, not counted {other:.1f}%"
            )
            stranded_ll = np.zeros((0, 2))
            sel = np.isin(gs.group_code, members) & (an.sample.kind == ps.KIND_STRANDED)
            if sel.any():
                stranded_ll = np.column_stack([an.sample.source_lon[sel], an.sample.source_lat[sel]])
            caption = (
                f"{site_label}，{case_text}。{scope}（{n:,} 顆虛擬質點，{len(members)} 條線；{per_line}）。"
                "以研究地點錨點為圓心，把每顆虛擬質點的來源點相對錨點的方位角（大圓初始方位角，自正北順時針）分到 36 個"
                "各寬 10° 的扇區（0°、10°、…、350° 為中心，正北扇區涵蓋 355°–5°），折線的半徑為落在該扇區的比例（%），"
                "36 個值連成封閉折線。來源點為穿越開放邊界者的離開點，與擱淺者的擱淺位置；其餘終止類別不計入扇區但留在分母。"
                f"本圖穿越 {crossed:.2f}%、擱淺 {stranded:.2f}%、未計入 {other:.2f}%。"
                f"半徑上限 {top:g}%。比例描述條件式來源足跡的相對分配，不是絕對來源量化，也不可解讀為因果歸因。"
            )
            plans.append(
                PolarFigurePlan(
                    kind=kind,
                    stem=figure_stem(kind, material or None),
                    title_zh=title_zh,
                    title_en=title_en,
                    subtitle_zh=subtitle_zh,
                    subtitle_en=subtitle_en,
                    legend_title_zh=legend_zh,
                    legend_title_en=legend_en,
                    labels=labels,
                    colors=_line_colors(kind, ids),
                    styles=[_LINE_STYLES[j % len(_LINE_STYLES)] for j in range(len(members))],
                    pct=pct_all[members],
                    n_totals=[int(gs.n_total[i]) for i in members],
                    radius_max_pct=top,
                    radius_ticks_pct=list(ticks),
                    stranded_lonlat=stranded_ll,
                    caption_zh=caption,
                )
            )
    return plans


# ── 繪圖 ─────────────────────────────────────────────────────────────────

_LAND_COLOR = "#d9d9d9"
_WINDOW_FACTOR = 1.32
"""地圖視窗半寬 = 圓半徑 × 此倍數（留出方位刻度與標籤的空間）。"""


@dataclass(slots=True)
class _SiteMap:
    """單站的地圖視窗與底圖（已裁到視窗內）。"""

    anchor: tuple[float, float]
    radius_km: float
    xticks: np.ndarray
    yticks: np.ndarray
    extent: tuple[float, float, float, float]
    land: list[Any]
    isobaths: Mapping[float, list[np.ndarray]]


def _build_site_map(
    anchor: tuple[float, float],
    radius_km: float,
    land_geometries: Sequence[Any],
    site_isobaths: Mapping[float, list[np.ndarray]],
) -> _SiteMap:
    import shapely

    lon0, lat0 = anchor
    half = radius_km * _WINDOW_FACTOR
    dlat = half / _KM_PER_DEG
    dlon = half / (_KM_PER_DEG * math.cos(math.radians(lat0)))
    xticks = _axis_ticks(lon0 - dlon, lon0 + dlon, max_intervals=8)
    yticks = _axis_ticks(lat0 - dlat, lat0 + dlat, max_intervals=8)
    extent = (float(xticks[0]), float(xticks[-1]), float(yticks[0]), float(yticks[-1]))
    land: list[Any] = []
    for geom in land_geometries:
        clipped = shapely.clip_by_rect(geom, extent[0], extent[2], extent[1], extent[3])
        if clipped.is_empty:
            continue
        parts = list(clipped.geoms) if hasattr(clipped, "geoms") else [clipped]
        land.extend(p for p in parts if p.geom_type in ("Polygon", "MultiPolygon") and not p.is_empty)
    return _SiteMap(anchor, radius_km, xticks, yticks, extent, land, site_isobaths)


def render_figure(
    plan: PolarFigurePlan,
    site_map: _SiteMap,
    t: _Labeler,
    font: Any,
    path: Path,
    dpi: int,
    figure_text: str,
) -> None:
    """畫單張圓形來源方位圖（含地圖底圖、圓與扇區刻度、半徑刻度、10 條以內折線與圖例），並落檔（PNG＋SVG）。"""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter

    lon0, lat0 = site_map.anchor
    radius_km = site_map.radius_km
    rmax = plan.radius_max_pct
    fig = plt.figure(figsize=(13.4, 7.6), dpi=150)
    fig.subplots_adjust(left=0.075, right=0.66, bottom=0.075, top=_title_top_margin(figure_text, full=0.855, none=0.97))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_anchor("W")
    x0, x1, y0, y1 = site_map.extent
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect(1.0 / math.cos(math.radians(lat0)), adjustable="box")
    ax.set_facecolor("white")

    # 底圖：陸地（灰）與海岸線（黑），等深線（中灰細線）。
    from shapely.plotting import plot_polygon

    for land in site_map.land:
        plot_polygon(land, ax=ax, add_points=False, facecolor=_LAND_COLOR, edgecolor="black", linewidth=0.5, zorder=2)
    levels = sorted(site_map.isobaths)
    drawn = _draw_isobaths(ax, site_map.isobaths, site_map.extent, levels) if levels else []

    # 圓、同心半徑刻度圈、十字軸線與扇區刻度。
    az_ring = np.linspace(0.0, 360.0, 721)
    lo, la = polar_xy_to_lonlat(site_map.anchor, radius_km, az_ring)
    ax.plot(lo, la, color="#222222", linewidth=1.0, zorder=4)
    for v in plan.radius_ticks_pct:
        if abs(v - rmax) < 1e-9:
            continue
        lo, la = polar_xy_to_lonlat(site_map.anchor, radius_km * v / rmax, az_ring)
        ax.plot(lo, la, color="#555555", linewidth=0.5, linestyle=(0, (4, 3)), alpha=0.8, zorder=4)
    for az in (0.0, 90.0, 180.0, 270.0):
        lo, la = polar_xy_to_lonlat(site_map.anchor, np.array([0.0, radius_km]), np.array([az, az]))
        ax.plot(lo, la, color="#555555", linewidth=0.5, alpha=0.8, zorder=4)
    for k in range(ps.N_SECTORS):
        az = ps.SECTOR_WIDTH_DEG * k - ps.SECTOR_WIDTH_DEG / 2.0  # 扇區邊界（5°、15°、…）
        lo, la = polar_xy_to_lonlat(site_map.anchor, np.array([radius_km, radius_km * 1.03]), np.array([az, az]))
        ax.plot(lo, la, color="#222222", linewidth=0.6, zorder=4)
    for az in range(0, 360, 30):
        lo, la = polar_xy_to_lonlat(site_map.anchor, radius_km * 1.095, float(az))
        ax.text(
            float(lo), float(la), f"{az}°", ha="center", va="center", fontproperties=font, fontsize=8.5, color="#222222", zorder=9
        )
    # 半徑刻度標籤：沿方位 45° 的半徑，白底以免被折線遮住。
    for v in plan.radius_ticks_pct:
        lo, la = polar_xy_to_lonlat(site_map.anchor, radius_km * v / rmax, 45.0)
        ax.text(
            float(lo),
            float(la),
            f"{v:g}%",
            ha="left",
            va="bottom",
            fontproperties=font,
            fontsize=8,
            color="#111111",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.6},
            zorder=9,
        )

    # 折線：36 個扇區中心的值連成封閉折線。
    centers = np.append(ps.SECTOR_WIDTH_DEG * np.arange(ps.N_SECTORS), 0.0)
    handles = []
    for i, label in enumerate(plan.labels):
        values = np.append(plan.pct[i], plan.pct[i][0])
        lo, la = polar_xy_to_lonlat(site_map.anchor, radius_km * values / rmax, centers)
        (line,) = ax.plot(
            lo,
            la,
            color=plan.colors[i],
            linestyle=plan.styles[i],
            linewidth=1.5,
            zorder=6,
            path_effects=[pe.withStroke(linewidth=2.8, foreground="white", alpha=0.65)],
        )
        handles.append(Line2D([], [], color=plan.colors[i], linestyle=plan.styles[i], linewidth=1.8, label=label))

    # 錨點（圓心）與擱淺點標記。
    ax.plot(
        lon0, lat0, marker="*", markersize=13, markerfacecolor="#ffd92f", markeredgecolor="black", markeredgewidth=0.7,
        linestyle="none", zorder=8,
    )  # fmt: skip
    extra: list[Any] = [
        Line2D([], [], marker="*", markersize=12, markerfacecolor="#ffd92f", markeredgecolor="black", linestyle="none",
               label=t.t("研究地點錨點（圓心）", "Site anchor (centre)"))
    ]  # fmt: skip
    if plan.stranded_lonlat.shape[0]:
        inside = (
            (plan.stranded_lonlat[:, 0] >= x0) & (plan.stranded_lonlat[:, 0] <= x1)
            & (plan.stranded_lonlat[:, 1] >= y0) & (plan.stranded_lonlat[:, 1] <= y1)
        )  # fmt: skip
        if inside.any():
            ax.plot(
                plan.stranded_lonlat[inside, 0], plan.stranded_lonlat[inside, 1], marker="v", markersize=4.5,
                markerfacecolor="#e45756", markeredgecolor="black", markeredgewidth=0.4, linestyle="none", zorder=7,
            )  # fmt: skip
        extra.append(
            Line2D([], [], marker="v", markersize=6, markerfacecolor="#e45756", markeredgecolor="black", linestyle="none",
                   label=t.t(f"擱淺點（共 {plan.stranded_lonlat.shape[0]:,} 顆）", f"Stranded points ({plan.stranded_lonlat.shape[0]:,})"))
        )  # fmt: skip
    for depth in sorted(drawn):
        k = _ISOBATH_LINESTYLES.index(_isobath_linestyle(depth, levels))
        extra.append(
            Line2D([], [], color=_ISOBATH_COLOR, linewidth=0.8, linestyle=_ISOBATH_LINESTYLES[k],
                   label=t.t(f"等深線 {depth:g} m（{_ISOBATH_STYLE_ZH[k]}）", f"Isobath {depth:g} m"))
        )  # fmt: skip

    # 軸：首末刻度即框邊，兩端都有數字。
    ax.set_xticks(site_map.xticks)
    ax.set_yticks(site_map.yticks)
    ax.xaxis.set_major_formatter(FuncFormatter(_lon_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(_lat_formatter))
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontproperties(font)
        lab.set_fontsize(9)

    # 主圖例（折線）用 ax.legend 建立，使 bbox_inches="tight" 一定把它納入裁切範圍；
    # 次圖例（錨點、擱淺點、等深線）較短，以 add_artist 加入。
    from matplotlib.legend import Legend

    leg = ax.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(1.03, 1.0), frameon=False, prop=font,
        title=t.t(plan.legend_title_zh, plan.legend_title_en), handlelength=3.4, borderaxespad=0.0, alignment="left",
    )  # fmt: skip
    leg.get_title().set_fontproperties(font)
    leg2 = Legend(
        ax, extra, [h.get_label() for h in extra], loc="lower left", bbox_to_anchor=(1.03, 0.0),
        bbox_transform=ax.transAxes, frameon=False, prop=font, handlelength=3.4, borderaxespad=0.0,
    )  # fmt: skip
    ax.add_artist(leg2)
    _add_two_line_title(
        fig, t, font, t.t(plan.title_zh, plan.title_en), t.t(plan.subtitle_zh, plan.subtitle_en), figure_text=figure_text
    )
    _save_exclusive(fig, path, dpi=dpi)
    plt.close(fig)


# ── 子行程工作 ───────────────────────────────────────────────────────────


def _render_job(job: Mapping[str, Any]) -> list[str]:
    """子行程：載入底圖並畫出一個（案例, 地點, 文字層級）的全部圖。回傳寫出的 PNG 路徑。"""

    import matplotlib

    matplotlib.use("Agg", force=True)
    font_res = _resolve_font(Path(job["font_path"]) if job["font_path"] else None)
    t = _Labeler(font_res.cjk_available)
    land = _load_coastline(Path(job["coastline"]))
    iso_all, _ = _load_isobaths(Path(job["bathymetry"]) if job["bathymetry"] else None)
    site_map = _build_site_map(tuple(job["anchor"]), job["radius_km"], land, iso_all.get(job["site"], {}))
    written: list[str] = []
    out_dir = Path(job["out_dir"])
    for plan in job["plans"]:
        path = out_dir / f"{plan.stem}.png"
        render_figure(plan, site_map, t, font_res.font_properties, path, job["dpi"], job["mode"])
        written.append(str(path))
    return written


# ── 摘要 ─────────────────────────────────────────────────────────────────

_CASE_LABEL_ZH = {"no_stokes": "不含斯托克斯漂流", "finite_depth_stokes": "含斯托克斯漂流"}
_VERDICT_ZH = {"over_noise": "明顯超過雜訊", "partial": "僅部分組對超過雜訊", "within_noise": "在雜訊內"}


def _site_conclusion(an: SiteAnalysis) -> str:
    """單站的一句話結論：哪些分組差異明顯超過雜訊（可放報告），哪些在雜訊內。"""

    by_kind: dict[str, list[ps.Comparison]] = {}
    for c in an.comparisons:
        by_kind.setdefault(c.kind, []).append(c)
    over: list[str] = []
    partial: list[str] = []
    within: list[str] = []

    def put(name: str, verdicts: Sequence[str]) -> None:
        if all(v == "over_noise" for v in verdicts):
            over.append(name)
        elif all(v == "within_noise" for v in verdicts):
            within.append(name)
        else:
            n_o = sum(v == "over_noise" for v in verdicts)
            n_p = sum(v == "partial" for v in verdicts)
            partial.append(f"{name}（{len(verdicts)} 類中 {n_o} 類明顯超過、{n_p} 類僅部分組對超過）" if len(verdicts) > 1 else name)

    for kind, name in (
        ("velocity_within_class", "同類 10 個速度之間"),
        ("class", "十類之間"),
        ("season", "季節之間"),
        ("tide", "大小潮之間"),
    ):
        if kind in by_kind:
            put(name, [c.verdict for c in by_kind[kind]])
    parts = []
    if over:
        parts.append("明顯超過雜訊（可放報告）：" + "、".join(over))
    if partial:
        parts.append("僅部分超過雜訊：" + "、".join(partial))
    if within:
        parts.append("在雜訊內：" + "、".join(within))
    return "；".join(parts) + "。"


def summary_markdown(
    case_id: str,
    analyses: Sequence[SiteAnalysis],
    scales: Mapping[str, tuple[float, list[float]]],
    config: PolarConfig,
) -> str:
    """單一案例的 summary.md 全文：方法、終止類別、每地點結論與可區辨性明細。"""

    case_zh = _CASE_LABEL_ZH.get(case_id, case_id)
    lines: list[str] = [
        f"# 圓形來源方位圖（WP-60）統計摘要：{case_zh}（`{case_id}`）",
        "",
        "## 方法摘要",
        "",
        "- 圓心為研究地點錨點；36 個各寬 10° 的扇區，以 0°、10°、…、350° 為中心（正北扇區為 355°–5°），方位自正北順時針；"
        "方位角用大圓初始方位角計算。",
        "- 來源點：穿越開放邊界者用開放邊界離開點；擱淺者用擱淺位置（朝向陸地一側）；其餘終止類別不計入扇區但留在分母。"
        "因此 36 扇區合計 = 穿越比例 + 擱淺比例。",
        "- 季節與大小潮使用沉底時刻計算的 `deposition_season`、`deposition_tide_class`。",
        f"- 可區辨性：每組以 37 格向量（36 扇區 + 未計入）描述；兩組距離用總變異距離（TVD）。雜訊基準 = 同組內依到達時刻單元"
        f"（`arrival_time_id`）整單元隨機對半切後兩半的 TVD，重抽 {config.params.n_resamples} 次，同一比較範圍內各組合併後取第 99 百分位為門檻；"
        "某組對的 TVD 超過門檻即判定超過雜訊。對半切每半只有一半樣本，門檻偏保守。"
        "判定：至少半數組對超過門檻為「明顯超過雜訊」；有超過但不到半數為「僅部分組對超過」；沒有組對超過為「在雜訊內」。",
        "- 結果稱為相對來源權重／條件式來源足跡，不是絕對來源量化或因果歸因。",
        "- 半徑上限（%）："
        + "、".join(f"{'/'.join(k for k in 'ABCD' if scales[k][0] == top)} 圖 {top:g}%" for top in dict.fromkeys(v[0] for v in scales.values()))
        + f"；圓半徑 {config.radius_km:g} km。",
        "",
        "## 每地點結論",
        "",
    ]
    for an in analyses:
        site = SITE_LABELS_ZH.get(an.site_id, an.site_id)
        lines.append(f"- **{site}（`{an.site_id}`）**：{_site_conclusion(an)}")
    lines += ["", "## 終止類別（每地點 10 萬顆虛擬質點）", ""]
    lines.append("| 地點 | 穿越開放邊界 | 擱淺 | " + " | ".join(ps.STATUS_NAME_ZH[c] for c in ps._OTHER_STATUS_CODES) + " | 計入扇區比例 |")
    lines.append("|" + "---|" * (4 + len(ps._OTHER_STATUS_CODES)))
    for an in analyses:
        site = SITE_LABELS_ZH.get(an.site_id, an.site_id)
        st = an.sample.status
        n = st.size
        cnt = {c: int(np.count_nonzero(st == c)) for c in ps._STATUS_CODES}
        cells = [f"{cnt[c]:,}" for c in ps._OTHER_STATUS_CODES]
        counted = cnt[1] + cnt[3]
        lines.append(f"| {site} | {cnt[1]:,} | {cnt[3]:,} | " + " | ".join(cells) + f" | {100.0 * counted / n:.2f}% |")
    lines += ["", "## 可區辨性明細", ""]
    for an in analyses:
        site = SITE_LABELS_ZH.get(an.site_id, an.site_id)
        lines += [f"### {site}（`{an.site_id}`）", ""]
        lines.append("| 比較範圍 | 組數／組對數 | 每組顆數 | 雜訊 TVD p99 | 最大 TVD | 超過 p99 的組對 | 最大扇區差（百分點） | 判定 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for c in an.comparisons:
            if c.kind == "velocity_within_class":
                name = f"同類 10 速度：{material_display_name_zh(c.scope_material)}"
            else:
                name = {"class": "十類之間", "season": "季節之間", "tide": "大小潮之間"}[c.kind]
            size = f"{c.group_n_min:,}" if c.group_n_min == c.group_n_max else f"{c.group_n_min:,}–{c.group_n_max:,}"
            lines.append(
                f"| {name} | {c.n_groups}／{c.n_pairs} | {size} | {c.noise_tvd['p99']:.3f} | {c.max_tvd:.3f} | "
                f"{c.pairs_over_p99}／{c.n_pairs} | {c.max_maxsec_pp:.2f} | {_VERDICT_ZH[c.verdict]} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


# ── 主流程 ───────────────────────────────────────────────────────────────


def _write_json_exclusive(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        stream.write("\n")


def _write_text_exclusive(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def _code_provenance() -> dict[str, Any]:
    from .arc_by_group_figures import _code_provenance as base

    return base()


def build_polar_source(config: PolarConfig, *, command: str) -> list[Path]:
    """依設定產出圓形來源方位圖、統計表、summary.md、manifest 與圖說；回傳各案例含文字版輸出目錄。

    流程：確認輸出目錄尚不存在 → 讀錨點與各案各站樣本並分析（含可區辨性檢定）→ 由全部案例全部地點決定半徑刻度 →
    建立目錄、寫表與摘要 → 以多行程繪圖（每個「案例×地點×文字層級」一個工作）→ 計算輸出檢核碼並寫 manifest。
    """

    targets = [c for c in config.cases if config.only_cases is None or c.case_id in config.only_cases]
    if not targets:
        raise PolarSourceError("only_cases 沒有對應到任何案例")
    for case in targets:
        for d, active in ((case.text_dir, True), (case.notext_dir, "none" in config.modes)):
            if active and d.exists():
                raise PolarSourceError(f"輸出目錄必須是尚不存在的全新目錄，但已存在：{d}")
        if "full" not in config.modes and case.text_dir.exists():
            raise PolarSourceError(f"輸出目錄必須是尚不存在的全新目錄，但已存在：{case.text_dir}")

    anchors = {k: (v[0], v[1]) for k, v in _load_receptor_anchors(config.release_config).items()}
    hashes_before = {c.case_id: sha256_file(c.postprocess_dir / "departures.parquet") for c in config.cases}
    analyses: dict[str, list[SiteAnalysis]] = {}
    for case in config.cases:
        samples = ps.load_site_polar_samples(case.postprocess_dir, anchors, None)
        analyses[case.case_id] = [ps.analyze_site(s, case.case_id, config.params) for s in samples.values()]
    scales = plan_radius_scales(analyses, override_max_pct=config.radius_max_pct)

    font_res = _resolve_font(config.font_path)
    generated_at = datetime.now(UTC)
    code = _code_provenance()
    render_jobs: list[dict[str, Any]] = []
    plan_records: dict[str, dict[str, Any]] = {}
    case_info: dict[str, dict[str, Any]] = {}
    for case in targets:
        import json as _json

        pp_manifest_path = case.postprocess_dir / "manifest.json"
        pp_manifest = _json.loads(pp_manifest_path.read_text(encoding="utf-8")) if pp_manifest_path.is_file() else {}
        case_text = _describe_experiment_case(pp_manifest).physics_case_zh
        shown = [an for an in analyses[case.case_id] if config.sites is None or an.site_id in config.sites]
        if not shown:
            raise PolarSourceError("sites 沒有對應到任何地點")
        captions: dict[str, str] = {}
        for an in shown:
            plans = build_figure_plans(an, case_text, scales, kinds=config.kinds, classes=config.classes)
            for plan in plans:
                captions[f"{an.site_id}/{plan.stem}"] = plan.caption_zh
            for mode, base in (("full", case.text_dir), ("none", case.notext_dir)):
                if mode not in config.modes:
                    continue
                out_dir = base / an.site_id
                out_dir.mkdir(parents=True)
                render_jobs.append(
                    {
                        "case": case.case_id,
                        "site": an.site_id,
                        "mode": mode,
                        "plans": plans,
                        "out_dir": str(out_dir),
                        "anchor": an.sample.anchor_lonlat,
                        "radius_km": config.radius_km,
                        "coastline": str(config.coastline),
                        "bathymetry": str(config.bathymetry) if config.bathymetry else None,
                        "font_path": str(config.font_path) if config.font_path else None,
                        "dpi": config.dpi,
                    }
                )
        case_info[case.case_id] = {"shown": shown, "captions": captions, "case_text": case_text}
        if config.write_tables:
            tdir = case.text_dir / "tables"
            tdir.mkdir(parents=True, exist_ok=True)
            all_an = analyses[case.case_id]
            ps.write_sector_long_csv(all_an, case.case_id, tdir / "sector_shares_long.csv")
            ps.write_termination_csv(all_an, case.case_id, tdir / "termination_by_group.csv")
            ps.write_pairs_csv(all_an, case.case_id, tdir / "distinguishability_pairs.csv")
            ps.write_comparison_csv(all_an, case.case_id, tdir / "distinguishability_summary.csv")
            ps.write_stranded_csv(all_an, case.case_id, tdir / "stranded_points.csv")
            _write_text_exclusive(case.text_dir / "summary.md", summary_markdown(case.case_id, all_an, scales, config))

    written: list[str] = []
    if config.workers <= 1 or len(render_jobs) <= 1:
        for job in render_jobs:
            written.extend(_render_job(job))
    else:
        import multiprocessing as mp

        with ProcessPoolExecutor(max_workers=min(config.workers, len(render_jobs)), mp_context=mp.get_context("fork")) as pool:
            for out in pool.map(_render_job, render_jobs):
                written.extend(out)

    outputs: list[Path] = []
    for case in targets:
        if sha256_file(case.postprocess_dir / "departures.parquet") != hashes_before[case.case_id]:
            raise PolarSourceError("輸入檔在繪圖過程中被改動，前後 SHA-256 不一致，fail-closed")
        info = case_info[case.case_id]
        figure_records: dict[str, Any] = {}
        for base in (case.text_dir, case.notext_dir):
            if not base.exists():
                continue
            for p in sorted(base.rglob("*.png")) + sorted(base.rglob("*.svg")):
                figure_records[str(p.relative_to(base.parent))] = {"sha256": sha256_file(p), "size_bytes": p.stat().st_size}
        manifest = {
            "generated_at_utc": generated_at.isoformat(),
            "command": command,
            "case_id": case.case_id,
            "code": code,
            "sites": [an.site_id for an in info["shown"]],
            "kinds": list(config.kinds),
            "modes": list(config.modes),
            "radius_km": config.radius_km,
            "radius_scales_pct": {k: {"max": v[0], "ticks": v[1]} for k, v in scales.items()},
            "radius_scale_override_pct": config.radius_max_pct,
            "sector": {"count": ps.N_SECTORS, "width_deg": ps.SECTOR_WIDTH_DEG, "center_azimuths_deg": [10 * k for k in range(36)]},
            "bearing": "great_circle_initial_bearing_from_anchor_clockwise_from_north",
            "anchors_lonlat": {an.site_id: list(an.sample.anchor_lonlat) for an in analyses[case.case_id]},
            "params": {"n_resamples": config.params.n_resamples, "seed": config.params.seed, "dpi": config.dpi},
            "font": dict(font_res.manifest_record),
            "cjk_labels_used": font_res.cjk_available,
            "input_departures_sha256": hashes_before[case.case_id],
            "release_config_basename": config.release_config.name,
            "status_counts_by_site": {
                an.site_id: {str(c): int(np.count_nonzero(an.sample.status == c)) for c in ps._STATUS_CODES}
                for an in analyses[case.case_id]
            },
            "output_figures_sha256": figure_records,
        }
        _write_json_exclusive(case.text_dir / "manifest.json", manifest)
        lines = ["# 圓形來源方位圖：圖說\n", f"案例：{case.case_id}\n"]
        for key, text in info["captions"].items():
            lines.append(f"## {key}\n\n{text}\n")
        _write_text_exclusive(case.text_dir / "captions.md", "\n".join(lines))
        _write_text_exclusive(
            case.text_dir / "README.md",
            "# 圓形來源方位圖（WP-60）\n\n"
            f"案例：`{case.case_id}`。本目錄由 `lbt native-polar-source` 產生，只讀取 `native-postprocess` 的 `departures.parquet`。\n\n"
            "- 含文字版圖：本目錄 `<地點>/`；無文字版圖：同層的 `figures-polar-notext/<地點>/`。\n"
            "- `tables/`：扇區比例長表、終止類別表、可區辨性組對與摘要表、擱淺點清單。\n"
            "- `summary.md`：統計摘要與每地點結論；`captions.md`：圖說；`manifest.json`：參數與檢核碼。\n\n"
            f"重建命令：\n\n```\n{command}\n```\n",
        )
        outputs.append(case.text_dir)
    return outputs
