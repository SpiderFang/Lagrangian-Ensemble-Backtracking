"""圓形來源方位圖（WP-60）的出圖、輸出表、摘要與 manifest。

圖的定義（PI 構想）
------------------
以研究地點的錨點為圓心，在有海岸線、等深線與經緯度軸的地圖上疊一個圓（半徑 ``radius_km``，預設 25 km，
在地圖上是真正的圓，因為地圖長寬比已依緯度校正）；圓分成 36 個各寬 10° 的扇區，扇區以 0°、10°、…、350°
為中心。每個扇區的值是「來源方位落在該扇區的虛擬質點占該組全部虛擬質點的比例（%）」，36 個值依序連成
**封閉的極座標折線**（半徑 = 比例，不是玫瑰圖的扇形長條）。折線的半徑與比例的對照由同心圓與半徑刻度
（如 5%、10%、…）給出。來源方位的計算見 :mod:`polar_source`：穿越開放邊界者用離開點，其餘全部（未穿越者）以最終位置的方位計入，視為海岸來源，
每組 36 扇區合計 100%；總曲線就是這一條，兩種來源的拆分只寫在統計表與副標。

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
* **無文字版共同裁切框**：無文字版沿用含文字版的畫面版型（圓的位置與大小一致），落檔時以同一個固定裁切框
  ``NOTEXT_CROP_IN``（英寸，由 130 張無文字版內容外框的聯集加約 0.08 英寸邊界決定）裁掉上方標題空白與右側多餘的圖例欄寬，
  使報告中固定寬度時地圖主體較大。裁切框是常數，不依各圖內容裁切，因此 130 張尺寸相同、圓心與半徑的像素位置相同。
* **固定畫布**：含文字版與無文字版、全部地點與圖種的畫布尺寸完全相同（``CANVAS_W_IN`` × ``CANVAS_H_IN`` 英寸，
  600 dpi 時 7560 × 4920 像素），落檔不使用 ``bbox_inches="tight"``（那會依內容裁切，使各圖寬度不一，
  也曾把圖例裁到畫布之外）。錨點（圓心）在畫布上的位置與圓的半徑像素數對每張圖都一樣；地圖框依各地點整齊刻度的範圍
  決定大小，落在固定的包絡內（見 :func:`_axes_rect`）；圖例放在右側固定欄位，欄寬足以容納最長圖例。
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
    _tick_font_kwargs,
)
from . import map_axes as _map_axes
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
    "FigureLayout",
    "RUN08_LAYOUT",
    "ENVELOPE_SIDES",
    "make_layout",
    "plan_envelope",
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

    版面（固定畫布的包絡）：預設沿用 run08 的包絡（左 43、右 39、下 45、上 40 km，半徑 25 km 時）與無文字版共同裁切框，
    run08 的圖因此可逐位元重現。``report_layout``：報告版面（畫布寬 6.27 英寸、圖例在主圖下方、A 類十圖併成一張多格圖，見 :func:`render_report_overview`），只輸出無文字版；
    ``envelope_km``（四個方向的公里數）明示包絡；``envelope_auto`` 依本次全部錨點自動決定包絡
    （:func:`plan_envelope`）。包絡不是 run08 預設時，無文字版共同裁切框由本次全部無文字版圖的內容外框聯集量測決定
    （四邊各加 0.08 英寸），也可用 ``notext_crop_in`` 明示。
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
    envelope_km: Mapping[str, float] | None = None
    envelope_auto: bool = False
    notext_crop_in: tuple[float, float, float, float] | None = None
    report_layout: bool = False
    # WP-71：報告版面的座標軸樣式。legacy＝沿用既有（刻度二位小數、無座標軸標題）；
    # one_decimal＝刻度標籤一位小數（只標框內 0.1°／0.2° 整數倍的刻度，不把非整刻度四捨五入標示）、
    # 加座標軸標題「經度」「緯度」。圓一律在框正中（框以錨點為中心、五地點同為 ±34 km），兩端不強制落在刻度上。
    report_axes_style: str = "legacy"

    def __post_init__(self) -> None:
        if self.report_axes_style not in ("legacy", "one_decimal"):
            raise PolarSourceError("report_axes_style 必須是 legacy 或 one_decimal")
        if self.report_axes_style != "legacy" and not self.report_layout:
            raise PolarSourceError("report_axes_style=one_decimal 只能搭配 report_layout")
        if self.report_layout:
            if self.modes != ("none",):
                raise PolarSourceError("report_layout 只輸出無文字版：modes 必須是 ('none',)")
            if self.envelope_km is not None or self.envelope_auto or self.notext_crop_in is not None:
                raise PolarSourceError("report_layout 不使用包絡與裁切框設定")
            if abs(self.radius_km - 25.0) > 1e-9:
                raise PolarSourceError("report_layout 的半徑固定為 25 km")
        if self.envelope_km is not None and self.envelope_auto:
            raise PolarSourceError("envelope_km 與 envelope_auto 只能擇一")
        if self.envelope_km is not None:
            make_layout(self.envelope_km)  # 提早驗證
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
    caption_zh: str


def _line_colors(kind: str, group_ids: Sequence[str]) -> list[Any]:
    from matplotlib import cm

    if kind == "C":
        return [_SEASON_COLORS.get(g, "#7f7f7f") for g in group_ids]
    if kind == "D":
        return [_TIDE_COLORS.get(g, "#7f7f7f") for g in group_ids]
    k = len(group_ids)
    return [cm.viridis(0.88 * (i / max(k - 1, 1))) for i in range(k)]


def _coverage(an: SiteAnalysis, grouping: str, members: Sequence[int]) -> tuple[int, float, float]:
    """所選組別全部質點數，與穿越開放邊界、海岸來源（未穿越者）各占的百分比（兩者合計 100）。"""

    gs = an.group_sets[grouping]
    sel = np.isin(gs.group_code, list(members))
    n = int(sel.sum())
    crossed = 100.0 * float(np.count_nonzero(an.sample.kind[sel] == ps.KIND_CROSSED)) / n
    return n, crossed, 100.0 - crossed


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
            n, crossed, coast = _coverage(an, grouping, members)
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
                f"半徑為來源方位落在該扇區的比例（%），36 扇區合計 100%\n{per_line}｜"
                f"穿越開放邊界 {crossed:.1f}%｜海岸來源（抵達海岸者以抵達位置方位計入）{coast:.1f}%"
            )
            subtitle_en = (
                f"Centre: site anchor | 36 sectors of 10 deg, bearing clockwise from north | radius: share in sector (%), sectors sum to 100%\n"
                f"crossed open boundary {crossed:.1f}% | coast source (non-crossing, counted by final-position bearing) {coast:.1f}%"
            )
            caption = (
                f"{site_label}，{case_text}。{scope}（{n:,} 顆虛擬質點，{len(members)} 條線；{per_line}）。"
                "以研究地點錨點為圓心，把每顆虛擬質點的來源點相對錨點的方位角（大圓初始方位角，自正北順時針）分到 36 個"
                "各寬 10° 的扇區（0°、10°、…、350° 為中心，正北扇區涵蓋 355°–5°），折線的半徑為落在該扇區的比例（%），"
                "36 個值連成封閉折線，合計 100%。來源點：穿越開放邊界者為開放邊界離開點；回溯途中抵達海岸者（狀態碼 8，方法 B）為首次抵達位置，視為海岸來源；其餘未穿越者（沉底、達回溯上限）"
                "一律以最終位置的方位計入。不設距海岸距離門檻。"
                f"本圖穿越開放邊界 {crossed:.2f}%、海岸來源 {coast:.2f}%。"
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
                    caption_zh=caption,
                )
            )
    return plans


# ── 繪圖 ─────────────────────────────────────────────────────────────────

_LAND_COLOR = "#d9d9d9"
_WINDOW_FACTOR = 1.32
"""地圖視窗半寬 = 圓半徑 × 此倍數（留出方位刻度與標籤的空間）。"""

CANVAS_W_IN = 12.6
CANVAS_H_IN = 8.2
"""固定畫布尺寸（英寸）；PNG 以 600 dpi 輸出時為 7560 × 4920 像素，全部圖相同。"""

_LEFT_MARGIN_IN = 0.85
_BOTTOM_MARGIN_IN = 0.65
_TOP_FRAC = 0.855
"""地圖區頂端占畫布高度的比例（其上為標題與副標的空間；無文字版沿用同一版面，使圓的位置一致）。"""
_LEGEND_X_IN = 7.25
"""圖例欄左緣（英寸）；欄寬到畫布右緣前 0.25 英寸，約 5.1 英寸，足以容納最長圖例。"""
NOTEXT_CROP_IN = (0.26, 0.43, 10.21, 7.08)
"""無文字版的共同裁切框 ``(x0, y0, x1, y1)``（英寸，自畫布左下角起算）。由 run08 兩案 130 張無文字版（含最長的十類圖圖例）
非白色像素外框的聯集，四邊各加約 0.08 英寸而得；600 dpi 時為 5970 × 3990 像素。"""
_ENVELOPE_KM = {"left": 43.0, "right": 39.0, "down": 45.0, "up": 40.0}
"""半徑 25 km 時，各地點「整齊刻度範圍」自錨點向四方延伸公里數的包絡上限（五個研究地點的實測最大值為
左 42.5、右 38.0、下 44.1、上 39.3 km）；超出即失敗，以保證地圖框與圖例不超出固定畫布。"""
_IN_PER_KM_25 = (_TOP_FRAC * CANVAS_H_IN - _BOTTOM_MARGIN_IN) / (_ENVELOPE_KM["down"] + _ENVELOPE_KM["up"])
"""半徑 25 km 時地圖每公里的英寸數（圓半徑 = 25 × 此值）；半徑不是 25 km 時反比縮放，使圓的像素大小不變。"""
_ANCHOR_X_IN = _LEFT_MARGIN_IN + _ENVELOPE_KM["left"] * _IN_PER_KM_25
_ANCHOR_Y_IN = _BOTTOM_MARGIN_IN + _ENVELOPE_KM["down"] * _IN_PER_KM_25
"""錨點（圓心）在畫布上的固定位置（自左下角起算的英寸）。"""

ENVELOPE_SIDES = ("left", "right", "down", "up")
_NOTEXT_MARGIN_IN = 0.08
"""無文字版共同裁切框在內容聯集外各加的邊界（英寸）。"""
_MAP_TO_LEGEND_GAP_IN = 0.2
"""地圖框右緣與圖例欄左緣之間至少保留的間距（英寸）。"""


@dataclass(frozen=True, slots=True)
class FigureLayout:
    """固定畫布版面：包絡（半徑 25 km 時各方向公里數）、比例尺、錨點位置與無文字版共同裁切框。

    預設版面 :data:`RUN08_LAYOUT` 與 run08（WP-60）逐位元相同。包絡可由設定給定或依一次執行的全部錨點自動決定
    （:func:`plan_envelope`）；包絡改變時比例尺與錨點位置隨之改變，但同一次執行內的全部圖仍使用同一個版面
    （圓心與圓半徑的像素位置一致、畫布尺寸一致）。``notext_crop_in`` 為 ``None`` 表示輸出整張畫布（量測用）。
    """

    left: float
    right: float
    down: float
    up: float
    in_per_km_25: float
    anchor_x_in: float
    anchor_y_in: float
    notext_crop_in: tuple[float, float, float, float] | None

    def envelope(self) -> dict[str, float]:
        return {"left": self.left, "right": self.right, "down": self.down, "up": self.up}


def make_layout(
    envelope_km: Mapping[str, float],
    notext_crop_in: tuple[float, float, float, float] | None = NOTEXT_CROP_IN,
) -> FigureLayout:
    """由包絡（半徑 25 km 時各方向公里數）建立版面；包絡不合理（缺方向、非正數）時失敗；比例尺取垂直放得下與不壓到圖例欄兩者較小者。"""

    if set(envelope_km) != set(ENVELOPE_SIDES):
        raise PolarSourceError(f"包絡必須恰有 {ENVELOPE_SIDES} 四個方向")
    env = {k: float(envelope_km[k]) for k in ENVELOPE_SIDES}
    if any(not (math.isfinite(v) and v > 0.0) for v in env.values()):
        raise PolarSourceError("包絡各方向必須是有限正數（公里）")
    fit_v = (_TOP_FRAC * CANVAS_H_IN - _BOTTOM_MARGIN_IN) / (env["down"] + env["up"])
    fit_h = (_LEGEND_X_IN - _MAP_TO_LEGEND_GAP_IN - _LEFT_MARGIN_IN) / (env["left"] + env["right"])
    in_per_km = min(fit_v, fit_h)  # 垂直方向放得下為準；左右很寬時改以不壓到圖例欄為準（run08 包絡由垂直方向決定）
    anchor_x = _LEFT_MARGIN_IN + env["left"] * in_per_km
    anchor_y = _BOTTOM_MARGIN_IN + env["down"] * in_per_km
    return FigureLayout(env["left"], env["right"], env["down"], env["up"], in_per_km, anchor_x, anchor_y, notext_crop_in)


RUN08_LAYOUT = make_layout(_ENVELOPE_KM, NOTEXT_CROP_IN)
"""run08（WP-60）的版面；預設值，行為與改版前完全相同。"""


def _axes_rect(
    anchor: tuple[float, float],
    radius_km: float,
    extent: tuple[float, float, float, float],
    layout: FigureLayout | None = None,
) -> tuple[float, float, float, float]:
    """地圖框在固定畫布上的位置 ``(left, bottom, width, height)``（畫布比例）。

    把錨點固定放在畫布的 ``(_ANCHOR_X_IN, _ANCHOR_Y_IN)``，比例尺固定（圓半徑像素數不變），
    框的四邊依該地點整齊刻度範圍到錨點的實際公里數決定；經緯度的公里換算與圖面長寬比校正一致
    （緯度每度 111.195 km、經度再乘 cos 緯度），所以圓是正圓。範圍超出包絡時失敗。
    """

    lay = RUN08_LAYOUT if layout is None else layout
    env = lay.envelope()
    lon0, lat0 = anchor
    x0, x1, y0, y1 = extent
    kx = _KM_PER_DEG * math.cos(math.radians(lat0))
    km = {"left": (lon0 - x0) * kx, "right": (x1 - lon0) * kx, "down": (lat0 - y0) * _KM_PER_DEG, "up": (y1 - lat0) * _KM_PER_DEG}
    scale = radius_km / 25.0
    for side, value in km.items():
        if value > env[side] * scale + 1e-6:
            raise PolarSourceError(f"地圖範圍超出固定畫布包絡（{side} {value:.1f} km > {env[side] * scale:.1f} km）")
    s_in = lay.in_per_km_25 / scale
    left = lay.anchor_x_in - km["left"] * s_in
    bottom = lay.anchor_y_in - km["down"] * s_in
    width = (km["left"] + km["right"]) * s_in
    height = (km["down"] + km["up"]) * s_in
    return left / CANVAS_W_IN, bottom / CANVAS_H_IN, width / CANVAS_W_IN, height / CANVAS_H_IN


def _save_fixed_canvas(
    fig: Any, path: Path, *, dpi: int, crop_in: tuple[float, float, float, float] | None = None, svg: bool = True
) -> None:
    """排他建立 PNG 與同名 SVG；**不使用 ``bbox_inches="tight"``**（不依內容裁切）。

    ``crop_in`` 為 ``None`` 時輸出整張畫布（``figsize``）；給定時以這個固定的共同裁切框（英寸）輸出，
    同一版本全部圖使用同一個框，所以尺寸相同。
    """

    from matplotlib.transforms import Bbox

    box = None if crop_in is None else Bbox([[crop_in[0], crop_in[1]], [crop_in[2], crop_in[3]]])
    with path.open("xb") as stream:
        fig.savefig(stream, format="png", dpi=dpi, bbox_inches=box)
    if svg:
        with path.with_suffix(".svg").open("xb") as stream:
            fig.savefig(stream, format="svg", dpi=dpi, bbox_inches=box)


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
    layout: FigureLayout | None = None,
) -> _SiteMap:
    import shapely

    lon0, lat0 = anchor
    half = radius_km * _WINDOW_FACTOR
    dlat = half / _KM_PER_DEG
    dlon = half / (_KM_PER_DEG * math.cos(math.radians(lat0)))
    xticks = _axis_ticks(lon0 - dlon, lon0 + dlon, max_intervals=8)
    yticks = _axis_ticks(lat0 - dlat, lat0 + dlat, max_intervals=8)
    extent = (float(xticks[0]), float(xticks[-1]), float(yticks[0]), float(yticks[-1]))
    _axes_rect(anchor, radius_km, extent, layout)  # 範圍超出固定畫布包絡時及早失敗
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
    layout: FigureLayout | None = None,
    png_only: bool = False,
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
    fig = plt.figure(figsize=(CANVAS_W_IN, CANVAS_H_IN), dpi=150)
    lay = RUN08_LAYOUT if layout is None else layout
    ax = fig.add_axes(_axes_rect(site_map.anchor, radius_km, site_map.extent, lay))
    x0, x1, y0, y1 = site_map.extent
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("auto")  # 框的長寬已依公里比例尺算好（見 _axes_rect），圓為正圓
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

    # 錨點（圓心）。
    ax.plot(
        lon0, lat0, marker="*", markersize=13, markerfacecolor="#ffd92f", markeredgecolor="black", markeredgewidth=0.7,
        linestyle="none", zorder=8,
    )  # fmt: skip
    extra: list[Any] = [
        Line2D([], [], marker="*", markersize=12, markerfacecolor="#ffd92f", markeredgecolor="black", linestyle="none",
               label=t.t("研究地點錨點（圓心）", "Site anchor (centre)"))
    ]  # fmt: skip
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

    # 主圖例（折線）用 ax.legend 建立；兩個圖例都放在畫布右側固定欄位（畫布比例座標），不依內容裁切；
    # 次圖例（錨點、等深線）較短，以 add_artist 加入。
    from matplotlib.legend import Legend

    leg = ax.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(_LEGEND_X_IN / CANVAS_W_IN, _TOP_FRAC), bbox_transform=fig.transFigure,
        frameon=False, prop=font,
        title=t.t(plan.legend_title_zh, plan.legend_title_en), handlelength=3.4, borderaxespad=0.0, alignment="left",
    )  # fmt: skip
    leg.get_title().set_fontproperties(font)
    leg2 = Legend(
        ax, extra, [h.get_label() for h in extra], loc="lower left",
        bbox_to_anchor=(_LEGEND_X_IN / CANVAS_W_IN, _BOTTOM_MARGIN_IN / CANVAS_H_IN),
        bbox_transform=fig.transFigure, frameon=False, prop=font, handlelength=3.4, borderaxespad=0.0,
    )  # fmt: skip
    ax.add_artist(leg2)
    _add_two_line_title(
        fig, t, font, t.t(plan.title_zh, plan.title_en), t.t(plan.subtitle_zh, plan.subtitle_en), figure_text=figure_text
    )
    _save_fixed_canvas(fig, path, dpi=dpi, crop_in=lay.notext_crop_in if figure_text == "none" else None, svg=not png_only)
    plt.close(fig)


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
_RO_LEGEND_H_BY_KIND: Mapping[str, float] = {"B": 0.92, "C": 0.76, "D": 0.52}
"""圖例帶高度（英寸）：B 十類（兩欄 5 列＋右側 6 列）、C 季節（4 列）、D 大小潮（2 列）；C、D 的錨點與等深線圖例改兩欄。"""
_RO_BOTTOM = 0.04


# WP-71（one_decimal 樣式）：多一條縱軸標題（左側加寬）與橫軸標題（下方加高），畫布寬仍為 6.27 英寸。
_RO_LEFT_1D = 0.75
_RO_FRAME_1D = REPORT_W_IN - _RO_LEFT_1D - _RO_RIGHT_MARGIN
_RO_XLAB_1D = 0.50


def report_overview_height(kind: str, one_decimal: bool = False) -> float:
    """總覽圖（B、C、D）的畫布高度（英寸）；同一圖種五地點相同，圖例帶依圖種的列數決定，不留大片空白。"""

    frame, xlab = (_RO_FRAME_1D, _RO_XLAB_1D) if one_decimal else (_RO_FRAME, _RO_XLAB)
    return round(_RO_TOP + frame + xlab + _RO_LEGEND_GAP + _RO_LEGEND_H_BY_KIND[kind] + _RO_BOTTOM, 3)


REPORT_OVERVIEW_H = report_overview_height("B")

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


_RG_LEFT_1D = 0.75
_RG_P_1D = (REPORT_W_IN - _RG_LEFT_1D - _RG_RIGHT_MARGIN - (_RG_COLS - 1) * _RG_GAP) / _RG_COLS
_RG_XLAB_1D = 0.42


def _rg_row_top(r: int, one_decimal: bool = False) -> float:
    """第 r 列（0 起算）標題帶上緣，自畫布頂端起算（英寸）；第 4 列前多留一條 x 軸刻度標籤帶（第 8、9 格在其上方）。"""

    p, xlab = (_RG_P_1D, _RG_XLAB_1D) if one_decimal else (_RG_P, _RG_XLAB)
    return _RG_TOP + r * (_RG_TITLE + p + _RG_ROWGAP) + (xlab if r == _RG_ROWS - 1 else 0.0)


def report_panels_height(one_decimal: bool = False) -> float:
    """多格圖的畫布高度（英寸）。"""

    p, xlab = (_RG_P_1D, _RG_XLAB_1D) if one_decimal else (_RG_P, _RG_XLAB)
    return round(_rg_row_top(_RG_ROWS - 1, one_decimal) + _RG_TITLE + p + xlab + _RG_BOTTOM, 3)


REPORT_PANELS_H = report_panels_height()


def _report_ticks(lo: float, hi: float, steps: Sequence[float], min_count: int, inner: tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    """範圍內規則間距的刻度（度）：由 ``steps`` 依序嘗試，取第一個（落在範圍 ``inner`` 比例內的）刻度數不少於 ``min_count`` 者。

    ``inner`` 把靠近框邊的刻度排除（小格圖的刻度標籤寬度接近格寬的三分之一，貼邊會超出畫布）。
    """

    for step in steps:
        k0 = math.ceil(lo / step - 1e-9)
        k1 = math.floor(hi / step + 1e-9)
        ticks = np.round(np.arange(k0, k1 + 1) * step, 6)
        frac = (ticks - lo) / (hi - lo)
        ticks = ticks[(frac >= inner[0] - 1e-9) & (frac <= inner[1] + 1e-9)]
        if len(ticks) >= min_count:
            return ticks
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
    one_decimal: bool = False,
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
                color="#222222", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 0.3}, zorder=9)  # fmt: skip
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
    ax.xaxis.set_major_formatter(FuncFormatter(_map_axes.format_lon if one_decimal else _lon_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(_map_axes.format_lat if one_decimal else _lat_formatter))
    ax.tick_params(axis="both", length=2.5, width=0.6, pad=2, labelsize=REPORT_TEXT_PT)
    ax.tick_params(axis="x", labelbottom=label_x)
    ax.tick_params(axis="y", labelleft=label_y)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontproperties(f8)
    if one_decimal:
        # 座標軸標題全報告統一為「經度 (°E)」「緯度 (°N)」（刻度標籤只寫數字）；只在有刻度標籤的邊畫
        if label_x:
            ax.set_xlabel(_map_axes.AXIS_TITLE_LON_ZH, fontproperties=f8, fontsize=REPORT_TEXT_PT, labelpad=1.5)
        if label_y:
            ax.set_ylabel(_map_axes.AXIS_TITLE_LAT_ZH, fontproperties=f8, fontsize=REPORT_TEXT_PT, labelpad=1.5)
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
        frameon=False, prop=f8, title=title, handlelength=2.0, handletextpad=0.6, labelspacing=0.28, columnspacing=0.9,
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
                raise PolarSourceError(f"{what}：圖例 {i} 與 {j} 重疊（{[round(v / dpi, 2) for v in lboxes[i].extents]} / {[round(v / dpi, 2) for v in lboxes[j].extents]} in）")
    for ax in axes:
        frame = ax.get_window_extent(renderer)
        for i, bb in enumerate(lboxes):
            if bb.overlaps(frame):
                raise PolarSourceError(f"{what}：圖例 {i} 與主圖框重疊（{[round(v / dpi, 2) for v in bb.extents]} / {[round(v / dpi, 2) for v in frame.extents]} in）")


def render_report_overview(
    plan: PolarFigurePlan, site_map: _SiteMap, t: _Labeler, font: Any, path: Path, dpi: int, png_only: bool = False,
    one_decimal: bool = False,
) -> None:
    """報告版面的總覽圖（B 十類、C 季節、D 大小潮）：正方形主圖占滿寬度，圖例全部在主圖正下方。"""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    ro_left, ro_frame, ro_xlab = (_RO_LEFT_1D, _RO_FRAME_1D, _RO_XLAB_1D) if one_decimal else (_RO_LEFT, _RO_FRAME, _RO_XLAB)
    w, h = REPORT_W_IN, report_overview_height(plan.kind, one_decimal)
    fig = plt.figure(figsize=(w, h), dpi=150)
    ax = fig.add_axes([ro_left / w, 1.0 - (_RO_TOP + ro_frame) / h, ro_frame / w, ro_frame / h])
    steps = (0.1,) if one_decimal else (0.1, 0.2, 0.25, 0.5)
    xt = _report_ticks(site_map.extent[0], site_map.extent[1], steps, 5, inner=(0.055, 0.945))
    yt = _report_ticks(site_map.extent[2], site_map.extent[3], steps, 5, inner=(0.03, 0.97))
    handles, drawn, levels = _draw_report_polar(ax, plan, site_map, font, compact=False, xticks=xt, yticks=yt, label_x=True, label_y=True,
                                                one_decimal=one_decimal)  # fmt: skip
    y_leg = _RO_TOP + ro_frame + ro_xlab + _RO_LEGEND_GAP
    ncol = 2 if len(handles) > 5 else 1
    legend_title = t.t(plan.legend_title_zh, plan.legend_title_en)
    if plan.kind == "B":  # 單位移到圖例標題，每列少寫一次 m/s，兩欄圖例才放得進主圖下方
        for hd in handles:
            hd.set_label(hd.get_label().removesuffix(" m/s"))
        legend_title += t.t("（m/s）", " (m/s)")
    leg1 = _fig_legend(fig, handles, font, loc="upper left", x_in=ro_left, y_top_in=y_leg, h_in=h, w_in=w,
                       title=legend_title, ncol=ncol)  # fmt: skip
    leg2 = _fig_legend(fig, _report_base_handles(t, drawn, levels), font, loc="upper right", x_in=ro_left + ro_frame, y_top_in=y_leg,
                       h_in=h, w_in=w, title=None, ncol=(1 if plan.kind == "B" else 2))  # fmt: skip
    _assert_report_layout(fig, [leg1, leg2], [ax], w, h, what=plan.stem)
    _save_fixed_canvas(fig, path, dpi=dpi, crop_in=None, svg=not png_only)
    plt.close(fig)


def render_report_panels(
    plans: Sequence[PolarFigurePlan], site_map: _SiteMap, t: _Labeler, font: Any, path: Path, dpi: int, png_only: bool = False,
    one_decimal: bool = False,
) -> None:
    """報告版面的多格圖：每類一格（格內十條線依沉降速度由慢至快），3 欄 × 4 列，共用一個圖例（沉降速度序位）。"""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if len(plans) != 10 or any(p.kind != "A" for p in plans):
        raise PolarSourceError("多格圖需要十張 A 圖計畫（每類一張）")
    w, h = REPORT_W_IN, report_panels_height(one_decimal)
    p = _RG_P_1D if one_decimal else _RG_P
    rg_left = _RG_LEFT_1D if one_decimal else _RG_LEFT
    col_x = [rg_left + c * (p + _RG_GAP) for c in range(_RG_COLS)]
    fig = plt.figure(figsize=(w, h), dpi=150)
    f8 = _text_font(font)
    steps = (0.2, 0.5) if one_decimal else (0.2, 0.25, 0.5)
    xt = _report_ticks(site_map.extent[0], site_map.extent[1], steps, 2, inner=(0.17, 0.83))
    yt = _report_ticks(site_map.extent[2], site_map.extent[3], steps, 2, inner=(0.1, 0.9))
    axes: list[Any] = []
    drawn: list[float] = []
    levels: list[float] = []
    for idx, plan in enumerate(plans):
        r, c = divmod(idx, _RG_COLS)
        top = _rg_row_top(r, one_decimal) + _RG_TITLE
        ax = fig.add_axes([col_x[c] / w, 1.0 - (top + p) / h, p / w, p / h])
        bottom_most = (r == _RG_ROWS - 1) or (idx + _RG_COLS >= len(plans))
        _, drawn, levels = _draw_report_polar(ax, plan, site_map, font, compact=True, xticks=xt, yticks=yt, label_x=bottom_most, label_y=(c == 0),
                                              one_decimal=one_decimal)  # fmt: skip
        cname = material_display_name_zh(plan.stem.removeprefix("polar_source_velocity_"))
        ax.set_title(f"{cname}　{plan.labels[0]}–{plan.labels[-1]}", loc="left", fontproperties=f8, fontsize=REPORT_TEXT_PT, pad=2.5)
        axes.append(ax)
    ranks = []
    for i in range(10):
        label = f"第 {i + 1}" + ("（最慢）" if i == 0 else "（最快）" if i == 9 else "")
        ranks.append(Line2D([], [], color=plans[0].colors[i], linestyle=plans[0].styles[i], linewidth=1.4, label=label))
    x_leg = col_x[1]
    y_leg = _rg_row_top(_RG_ROWS - 1, one_decimal) + 0.02
    title = t.t("格內十條線之沉降速度序位", "Rank of settling speed") + "\n" + t.t("格標題數字為該類十個沉降速度之範圍（m/s）", "Panel title: range (m/s)")
    leg1 = _fig_legend(fig, ranks, font, loc="upper left", x_in=x_leg, y_top_in=y_leg, h_in=h, w_in=w, title=title, ncol=2)
    leg2 = _fig_legend(fig, _report_base_handles(t, drawn, levels), font, loc="lower left", x_in=x_leg,
                       y_top_in=_rg_row_top(_RG_ROWS - 1, one_decimal) + _RG_TITLE + p, h_in=h, w_in=w, title=None, ncol=2)  # fmt: skip
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
        render_report_overview(plan, site_map, t, font_res.font_properties, path, job["dpi"], one_decimal=job.get("axes_style") == "one_decimal")
        written.append(str(path))
    if a_plans:
        path = out_dir / f"{REPORT_STEMS['A']}.png"
        render_report_panels(a_plans, site_map, t, font_res.font_properties, path, job["dpi"], one_decimal=job.get("axes_style") == "one_decimal")
        written.append(str(path))
    return written


# ── 子行程工作 ───────────────────────────────────────────────────────────


def _render_job(job: Mapping[str, Any]) -> list[str]:
    """子行程：載入底圖並畫出一個（案例, 地點, 文字層級）的全部圖。回傳寫出的 PNG 路徑。"""

    import matplotlib

    matplotlib.use("Agg", force=True)
    font_res = _resolve_font(Path(job["font_path"]) if job["font_path"] else None)
    t = _Labeler(font_res.cjk_available)
    land = _load_coastline(Path(job["coastline"]))
    iso_all, _ = _load_isobaths(Path(job["bathymetry"]) if job["bathymetry"] else None)
    layout = job.get("layout")
    site_map = _build_site_map(tuple(job["anchor"]), job["radius_km"], land, iso_all.get(job["site"], {}), layout)
    written: list[str] = []
    out_dir = Path(job["out_dir"])
    for plan in job["plans"]:
        path = out_dir / f"{plan.stem}.png"
        render_figure(plan, site_map, t, font_res.font_properties, path, job["dpi"], job["mode"], layout)
        written.append(str(path))
    return written


MEASURE_DPI = 60


def union_content_box_in(png_paths: Sequence[Path], dpi: int, canvas_in: tuple[float, float]) -> tuple[float, float, float, float]:
    """整張畫布 PNG 的非白色像素外框聯集（英寸，自左下角起算 ``(x0, y0, x1, y1)``）。"""

    import matplotlib.image as mpimg

    x0 = y0 = math.inf
    x1 = y1 = -math.inf
    for path in png_paths:
        img = np.asarray(mpimg.imread(str(path)))
        mask = np.any(img[..., :3] < 0.99, axis=-1)
        if img.shape[-1] == 4:
            mask |= img[..., 3] < 0.99
        rows = np.flatnonzero(mask.any(axis=1))
        cols = np.flatnonzero(mask.any(axis=0))
        if rows.size == 0 or cols.size == 0:
            continue
        h = img.shape[0]
        x0 = min(x0, cols[0] / dpi)
        x1 = max(x1, (cols[-1] + 1) / dpi)
        y0 = min(y0, (h - rows[-1] - 1) / dpi)
        y1 = max(y1, (h - rows[0]) / dpi)
    if not math.isfinite(x0):
        raise PolarSourceError("量測無文字版內容外框失敗：圖是空白的")
    return x0, y0, x1, y1


def crop_from_content_box(box: tuple[float, float, float, float], canvas_in: tuple[float, float]) -> tuple[float, float, float, float]:
    """內容聯集外框 → 共同裁切框：四邊各加 :data:`_NOTEXT_MARGIN_IN`，再向外取整到 0.01 英寸，並限制在畫布內。"""

    x0, y0, x1, y1 = box
    m = _NOTEXT_MARGIN_IN
    out = (
        max(0.0, math.floor((x0 - m) * 100.0 + 1e-6) / 100.0),
        max(0.0, math.floor((y0 - m) * 100.0 + 1e-6) / 100.0),
        min(canvas_in[0], math.ceil((x1 + m) * 100.0 - 1e-6) / 100.0),
        min(canvas_in[1], math.ceil((y1 + m) * 100.0 - 1e-6) / 100.0),
    )
    return out


def _measure_job(job: Mapping[str, Any]) -> list[str]:
    """子行程：以低解析度把一個無文字版工作的全部圖畫成整張畫布（不裁切）PNG，供量測共同裁切框。"""

    import dataclasses

    measured = dict(job)
    measured["layout"] = dataclasses.replace(job["layout"], notext_crop_in=None)
    measured["dpi"] = MEASURE_DPI
    # 沿用 _render_job 的版面建立邏輯，但只存 PNG
    import matplotlib

    matplotlib.use("Agg", force=True)
    font_res = _resolve_font(Path(measured["font_path"]) if measured["font_path"] else None)
    t = _Labeler(font_res.cjk_available)
    land = _load_coastline(Path(measured["coastline"]))
    iso_all, _ = _load_isobaths(Path(measured["bathymetry"]) if measured["bathymetry"] else None)
    layout = measured["layout"]
    site_map = _build_site_map(tuple(measured["anchor"]), measured["radius_km"], land, iso_all.get(measured["site"], {}), layout)
    out_dir = Path(measured["measure_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for plan in measured["plans"]:
        path = out_dir / f"{measured['case']}__{measured['site']}__{plan.stem}.png"
        render_figure(plan, site_map, t, font_res.font_properties, path, MEASURE_DPI, "none", layout, png_only=True)
        written.append(str(path))
    return written


def plan_envelope(anchors: Mapping[str, tuple[float, float]], radius_km: float) -> dict[str, float]:
    """依一次執行的全部錨點決定包絡（半徑 25 km 時各方向公里數）。

    每個地點的整齊刻度範圍（與 :func:`_build_site_map` 同一規則）到錨點的公里數，換算成半徑 25 km 的等價值後，
    各方向取全部地點的最大值，再向上取整到整數公里。只依錨點與半徑，不依資料。
    """

    if not anchors:
        raise PolarSourceError("沒有錨點，無法決定包絡")
    env = dict.fromkeys(ENVELOPE_SIDES, 0.0)
    half = radius_km * _WINDOW_FACTOR
    for lon0, lat0 in anchors.values():
        dlat = half / _KM_PER_DEG
        dlon = half / (_KM_PER_DEG * math.cos(math.radians(lat0)))
        xt = _axis_ticks(lon0 - dlon, lon0 + dlon, max_intervals=8)
        yt = _axis_ticks(lat0 - dlat, lat0 + dlat, max_intervals=8)
        kx = _KM_PER_DEG * math.cos(math.radians(lat0))
        km = {
            "left": (lon0 - float(xt[0])) * kx,
            "right": (float(xt[-1]) - lon0) * kx,
            "down": (lat0 - float(yt[0])) * _KM_PER_DEG,
            "up": (float(yt[-1]) - lat0) * _KM_PER_DEG,
        }
        for side, value in km.items():
            env[side] = max(env[side], value * 25.0 / radius_km)
    return {side: float(math.ceil(value - 1e-6)) for side, value in env.items()}


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
        "- 來源點（PI 定案規則，方法 B）：穿越開放邊界者用開放邊界離開點；回溯途中首次抵達海岸者（狀態 8）視為找到來源並停止追蹤，用抵達位置；"
        "**其餘沒有穿越開放邊界也沒有抵達海岸的虛擬質點**（觸底、達回溯上限等）一律以最終位置相對錨點的方位角計入 36 扇區，視為海岸來源；不設距海岸距離門檻。"
        "沒有「未計入」類別，每組 36 扇區比例合計為 100%（程式以浮點容差檢查，不符即失敗）。圖上的總曲線就是這一條；"
        "穿越開放邊界與海岸來源兩種來源的拆分寫在 `tables/termination_by_group.csv` 與 `tables/sector_shares_long.csv`（欄位分開）。",
        "- 季節與大小潮使用沉底時刻計算的 `deposition_season`、`deposition_tide_class`。",
        f"- 可區辨性：每組以 36 格向量（36 個扇區，合計 1）描述；兩組距離用總變異距離（TVD）。雜訊基準 = 同組內依到達時刻單元"
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
    lines += ["", "## 穿越開放邊界與海岸來源（每地點 10 萬顆虛擬質點；括號為比例）", ""]
    coast_names = [ps.STATUS_NAME_ZH[c] for c in ps._COAST_STATUS_CODES]
    lines.append("| 地點 | 穿越開放邊界 | 海岸來源合計 | " + " | ".join(f"其中：{nm}" for nm in coast_names) + " | 36 扇區合計 |")
    lines.append("|" + "---|" * (4 + len(coast_names)))
    for an in analyses:
        site = SITE_LABELS_ZH.get(an.site_id, an.site_id)
        st = an.sample.status
        n = st.size
        cnt = {c: int(np.count_nonzero(st == c)) for c in ps._STATUS_CODES}
        n_coast = n - cnt[1]
        cells = [f"{cnt[c]:,}（{100.0 * cnt[c] / n:.3f}%）" for c in ps._COAST_STATUS_CODES]
        sector_sum = 100.0 * float(np.count_nonzero((an.sample.sector >= 0) & (an.sample.sector < ps.N_SECTORS))) / n
        lines.append(
            f"| {site} | {cnt[1]:,}（{100.0 * cnt[1] / n:.3f}%） | {n_coast:,}（{100.0 * n_coast / n:.3f}%） | "
            + " | ".join(cells)
            + f" | {sector_sum:.4f}% |"
        )
    lines += ["", "## 各地點最大 TVD 與雜訊 p99（格式：最大 TVD／雜訊 p99）", ""]
    lines.append("| 地點 | 季節之間 | 大小潮之間 | 十類之間 | 同類 10 速度之間（取十類中最大者） |")
    lines.append("|---|---|---|---|---|")
    for an in analyses:
        site = SITE_LABELS_ZH.get(an.site_id, an.site_id)
        cells = []
        for kind in ("season", "tide", "class", "velocity_within_class"):
            cs = [c for c in an.comparisons if c.kind == kind]
            if not cs:
                cells.append("—")
                continue
            c = max(cs, key=lambda x: x.max_tvd)
            suffix = f"（{material_display_name_zh(c.scope_material)}）" if kind == "velocity_within_class" else ""
            cells.append(f"{c.max_tvd:.3f}／{c.noise_tvd['p99']:.3f}{suffix}")
        lines.append(f"| {site} | " + " | ".join(cells) + " |")
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
    if config.report_layout:
        envelope, envelope_source = dict(_ENVELOPE_KM), "not_applicable_report_layout"
    elif config.envelope_km is not None:
        envelope, envelope_source = {k: float(config.envelope_km[k]) for k in ENVELOPE_SIDES}, "explicit"
    elif config.envelope_auto:
        envelope, envelope_source = plan_envelope(anchors, config.radius_km), "auto_from_anchors"
    else:
        envelope, envelope_source = dict(_ENVELOPE_KM), "run08_default"
    is_default_envelope = envelope == dict(_ENVELOPE_KM)
    if config.report_layout:
        layout_crop, crop_source = NOTEXT_CROP_IN, "not_applicable_report_layout"
    elif config.notext_crop_in is not None:
        layout_crop, crop_source = tuple(config.notext_crop_in), "explicit"
    elif is_default_envelope:
        layout_crop, crop_source = NOTEXT_CROP_IN, "run08_default"
    else:
        layout_crop, crop_source = None, "measured_from_notext_figures"
    layout = make_layout(envelope, layout_crop if layout_crop is not None else NOTEXT_CROP_IN)
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
                        "layout": layout,
                        "axes_style": config.report_axes_style,
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
            _write_text_exclusive(case.text_dir / "summary.md", summary_markdown(case.case_id, all_an, scales, config))

    if crop_source == "measured_from_notext_figures":
        none_jobs = [j for j in render_jobs if j["mode"] == "none"]
        if none_jobs:
            import dataclasses
            import tempfile

            with tempfile.TemporaryDirectory(prefix="polar-measure-") as tmp:
                mjobs = [dict(j, measure_dir=str(Path(tmp) / f"j{i}")) for i, j in enumerate(none_jobs)]
                if config.workers <= 1 or len(mjobs) <= 1:
                    measured = [p for j in mjobs for p in _measure_job(j)]
                else:
                    import multiprocessing as mp

                    with ProcessPoolExecutor(
                        max_workers=min(config.workers, len(mjobs)), mp_context=mp.get_context("fork")
                    ) as pool:
                        measured = [p for out in pool.map(_measure_job, mjobs) for p in out]
                box = union_content_box_in([Path(p) for p in measured], MEASURE_DPI, (CANVAS_W_IN, CANVAS_H_IN))
                layout_crop = crop_from_content_box(box, (CANVAS_W_IN, CANVAS_H_IN))
            layout = dataclasses.replace(layout, notext_crop_in=layout_crop)
            for j in render_jobs:
                j["layout"] = layout
        else:
            crop_source = "not_needed_no_notext_mode"
    if config.report_layout:
        one_dec = config.report_axes_style == "one_decimal"
        figure_layout_record = {
            "report_layout": True,
            "axes_style": config.report_axes_style,
            "canvas_in": {**{k: [REPORT_W_IN, report_overview_height(k, one_dec)] for k in ("B", "C", "D")}, "panels": [REPORT_W_IN, report_panels_height(one_dec)]},
            "window_half_km": config.radius_km * REPORT_WINDOW_FACTOR,
            "overview_frame_in": _RO_FRAME_1D if one_dec else _RO_FRAME,
            "panel_frame_in": _RG_P_1D if one_dec else _RG_P,
            "text_pt": REPORT_TEXT_PT,
            "stems": dict(REPORT_STEMS),
        }
    else:
        figure_layout_record = {
        "envelope_km_at_radius_25": layout.envelope(),
        "envelope_source": envelope_source,
        "notext_crop_in": list(layout.notext_crop_in) if layout.notext_crop_in is not None else None,
        "notext_crop_source": crop_source,
        "canvas_in": [CANVAS_W_IN, CANVAS_H_IN],
        "in_per_km_at_radius_25": layout.in_per_km_25,
        "anchor_position_in": [layout.anchor_x_in, layout.anchor_y_in],
        }
    written: list[str] = []
    job_fn = _render_report_job if config.report_layout else _render_job
    if config.workers <= 1 or len(render_jobs) <= 1:
        for job in render_jobs:
            written.extend(job_fn(job))
    else:
        import multiprocessing as mp

        with ProcessPoolExecutor(max_workers=min(config.workers, len(render_jobs)), mp_context=mp.get_context("fork")) as pool:
            for out in pool.map(job_fn, render_jobs):
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
            "figure_layout": figure_layout_record,
            "radius_scales_pct": {k: {"max": v[0], "ticks": v[1]} for k, v in scales.items()},
            "radius_scale_override_pct": config.radius_max_pct,
            "sector": {"count": ps.N_SECTORS, "width_deg": ps.SECTOR_WIDTH_DEG, "center_azimuths_deg": [10 * k for k in range(36)]},
            "bearing": "great_circle_initial_bearing_from_anchor_clockwise_from_north",
            "source_rule": "crossed: flow-domain boundary crossing point; reached coast (status 8, method B): first-arrival position (coast source); other non-crossing statuses (deposited, max backtrack): final position; no distance threshold; 36 sectors sum to 100%",
            "crossed_share_pct_by_site": {
                an.site_id: 100.0 * float(np.count_nonzero(an.sample.kind == ps.KIND_CROSSED)) / an.sample.n_rows
                for an in analyses[case.case_id]
            },
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
        case.text_dir.mkdir(parents=True, exist_ok=True)
        manifest["report_layout"] = bool(config.report_layout)
        _write_json_exclusive(case.text_dir / "manifest.json", manifest)
        lines = ["# 圓形來源方位圖：圖說\n", f"案例：{case.case_id}\n"]
        for key, text in info["captions"].items():
            lines.append(f"## {key}\n\n{text}\n")
        _write_text_exclusive(case.text_dir / "captions.md", "\n".join(lines))
        if config.report_layout:
            _write_text_exclusive(
                case.text_dir / "README.md",
                "# 圓形來源方位圖：報告版面（WP-65）\n\n"
                f"案例：`{case.case_id}`。只改版面、不改數值：畫布寬 6.27 英寸（＝報告插入寬度），字級 8 pt，圖例在主圖正下方；"
                "每地點 4 張圖（十類總覽、十類各自十個沉降速度之多格圖、季節、大小潮），圖檔在同層 `figures-polar-v2-notext/<地點>/`。"
                "統計表、summary.md 與 captions.md 沿用 `figures-polar/`（數值相同）。\n\n"
                f"重建命令：\n\n```\n{command}\n```\n",
            )
            outputs.append(case.text_dir)
            continue
        _write_text_exclusive(
            case.text_dir / "README.md",
            "# 圓形來源方位圖（WP-60）\n\n"
            f"案例：`{case.case_id}`。本目錄由 `lbt native-polar-source` 產生，只讀取 `native-postprocess` 的 `departures.parquet`。\n\n"
            "- 含文字版圖：本目錄 `<地點>/`；無文字版圖：同層的 `figures-polar-notext/<地點>/`。\n"
            "- `tables/`：扇區比例長表（含穿越開放邊界與海岸來源分欄）、穿越／海岸來源比例表、可區辨性組對與摘要表。\n"
            "- `summary.md`：統計摘要與每地點結論；`captions.md`：圖說；`manifest.json`：參數與檢核碼。\n\n"
            f"重建命令：\n\n```\n{command}\n```\n",
        )
        outputs.append(case.text_dir)
    return outputs
