"""WP-21：由 native 後處理輸出（``postprocess.py`` 的 Parquet／CSV 成果）獨立繪製
每站八張正式成果圖。

背景與定位
----------
``native_runner.postprocess`` 只輸出表格（離開紀錄、邊界通量、條件式來源足跡的
格網數值），沒有任何圖。舊版 ``scripts/build_pilot_coastline_preview.py`` 的四張
圖只認得舊引擎 preview 目錄（``particles.csv``／``observations.csv``／
``summary.json``），沒有 native 後處理輸出可讀的欄位，也不認得方位扇區、弧段等
WP-20 新增產品。本模組是**獨立的 native 專用繪圖管線**：只讀
``postprocess.py`` 已經寫出的檔案與呼叫端明示的海岸 GeoJSON，**不讀
forcing、不重新積分、不改寫任何輸入**，輸出五站各八張 PNG、一份 README 與一份
manifest。舊版管線的任何行為都不因本模組存在而改變。

視覺風格沿用 ``build_pilot_coastline_preview.py --style baytrace`` 已核准的版面
語彙（標題兩行、副標列出情境身分與設定、灰色陸地、虛線 domain／開放邊界外框、
綠點＝回溯起點、紅叉／狀態標記＝停止位置、圖例在下方），但資料來源完全不同，
兩份程式碼刻意不共用任何私有函式，避免舊版契約被新用途悄悄改動。

WP-24b：圖面不畫警語、圖說另存、文字層級開關
--------------------------------------------
研究團隊裁決：正式成果圖面不畫任何警語文字（含舊版底部紅字警語），圖面本身只
承載資料（座標軸、圖例、可選標題／副標）；每張圖對應的文字說明（圖是什麼、
資料來源欄位、關鍵設定）改由本模組另外輸出的 ``captions.md`` 承載，並把同一段
文字與該圖的 SHA-256 一起寫進 ``manifest.json`` 的 ``captions`` 欄位，供報告
排版時原文引用、也供事後稽核「這張圖當初配的文字是不是這段」。``--figure-text``
命令列選項另外控制標題／副標是否畫在圖面上：``full``（預設）維持標題、副標、
圖例、座標軸標籤全部都畫，供內部檢視；``none`` 只畫座標軸與圖例，不畫標題與
副標，供正式報告排版——報告本身用 ``captions.md`` 的文字做圖說，不需要圖面上
再重複一次標題。兩種模式都會輸出 ``captions.md``，且 ``none`` 模式會把
``figure_text: "none"`` 記進 manifest，供人工確認這批圖是哪種文字層級。

真實輸入的已知落差（誠實揭露，見 ``reports/WP-21.md``）
--------------------------------------------------------
一次完整的 ``postprocess.py`` 執行會輸出比本模組實際依賴更多的表格（例如按
站×沉降速度細分的方位扇區／弧段相對來源權重表）。但這些表都是**可選**的：
``departures.parquet``（每顆粒子一列的離開紀錄表）本身已經保存
``departure_azimuth_sector_index/_name``、``boundary_arc_segment_index``、
``boundary_arc_length_position_m``、``boundary_segment_id`` 與
``settling_velocity_mps`` 等全部原始欄位，因此本模組對「方位扇區」「弧段」
「沉降速度細分」三張圖一律**直接從離開紀錄表重新聚合**（聚合公式與
``postprocess._sector_entry_table``／``_arc_segment_entry_table`` 完全相同：
分母是同一分組內成功穿越邊界的粒子數），不依賴那些屬於可選、允許缺席的聚合表。唯一
無法從離開紀錄表重建、且本模組必須讀取的**幾何**，是開放邊界折線本身的實際
座標，來自 ``boundary_arc_segments_geometry_by_site``。

本模組因此只要求八個輸入存在：``departures.parquet``、
``trajectory_samples_by_site.parquet``、``kde_by_site.parquet``、
``boundary_summary_by_site``、``boundary_arc_segments_geometry_by_site``、
``local_domain_departures.parquet``、``local_domain_first_exit_summary_by_site``
（``boundary_summary_by_site``／``boundary_arc_segments_geometry_by_site``／
``local_domain_first_exit_summary_by_site`` 皆可以是 ``.parquet`` 或 ``.csv``，
優先讀 ``.parquet``；``local_domain_departures.parquet`` 只接受 Parquet，理由同
``departures.parquet``）、``manifest.json``（``postprocess.py`` 自己輸出的執行
紀錄，供讀取方位扇區數、弧段數等設定）。缺少任一者一律 fail-closed，不静默略過。

WP-23：停止邊界與 local domain 首次離開的雙重語意
----------------------------------------------------
研究團隊裁決（見 ``AGENTS.md``、``docs/development/05_decisions_and_risks.md``
D008）：**終止一律以 flow domain 開放邊界為準；離開站點 local domain 是非終止
事件，只記錄、粒子繼續回溯**。``departures.parquet`` 的
``crossed_interest_region_boundary``／終止標記因此**已經是 flow domain 語意**，
五站可以直接互相比較；本模組另外從 ``local_domain_departures.parquet``（僅
``local_equals_flow=False`` 的測站——目前是貢寮、龜山島——才有列）與
``departures.parquet`` 的 ``also_local_domain_first_exit``（僅
``local_equals_flow=True`` 的測站——新竹、南灣、連江——才會為
``True``，代表這一列同時是 local 首次離開與 flow 終止）取得 local 首次離開的
位置，在水平總覽／局部兩張圖上用**另一種標記**畫出，並在圖例與副標明確區分
「local domain 邊界（記錄用，非終止）」與「flow domain 邊界（終止邊界）」。
``local_equals_flow=True`` 的站點兩者是同一次穿越，圖上只畫一個標記（沿用
終止標記本身），不另外疊加 local 標記，避免造成「同一顆粒子被畫了兩次」的
錯覺。

經緯度還原（無 domain manifest 輸入下的投影復原）
--------------------------------------------------
本模組命令列只有 ``--postprocess-dir`` 與 ``--coastline``，不讀 domain／open
manifest（避免疊床架屋，也避免正式圖表管線另外維護一份與 postprocess 輸出站點
不保證同步的幾何來源）。但代表軌跡樣本表只保存公尺制 ``x_m``／``y_m``／``z_m``，
沒有經緯度欄位；要疊圖到海岸底圖就必須知道各站「等距方位投影」（AEQD）的中心
經緯度。``departures.parquet`` 剛好同時保存同一列粒子的
``departure_x_m``／``departure_y_m``（公尺）與
``departure_lon_deg``／``departure_lat_deg``（經緯度）——這是同一個 AEQD 投影的
精確樣本對。``_recover_site_projection_center`` 因此重用既有
``geometry.DomainProjection``（與正式管線完全相同的投影公式），以
``scipy.optimize.least_squares`` 對每站的少量樣本點解出投影中心，使正投影後的
公尺座標與 ``departure_x_m``／``y_m`` 吻合到公尺等級以內；解不出時 fail-closed，
不用任何近似或線性內插代替。解出中心後才能用
``DomainProjection.unproject`` 把任何軌跡樣本或邊界幾何座標轉回經緯度。

科學語意與用語（依 AGENTS.md）
--------------------------------
本模組產出的方位扇區／弧段統計一律稱為「相對來源權重」，條件式核密度足跡一律
稱為「條件式來源足跡」；不使用「機率」「probability of origin」等字樣。這兩個
量的定義是：給定站點、受體、沉降速度與到達時刻等情境設定後，穿越各方位扇區／
各邊界弧段的相對頻率——是描述這一批 ``M=1`` 系集在既定設定下的分布，需要先驗、
概似與觀測驗證後才能延伸為來源機率或因果歸因，本模組不提供、也不宣稱提供這一
層推論。每張圖的副標與 ``captions.md`` 固定寫出站點、粒子數、``M=1``、終止邊界
定義（flow domain 開放邊界）等已知設定，供讀者自行核對適用範圍，不使用警語式
的「不得」句型。

字型與雙語退場
----------------
``--font-path`` 未提供時呼叫 ``report_font.resolve_cjk_font()``；找不到合格繁體
中文字型時**不讓整個管線失敗**，改把全部圖面文字換成英文（README 仍固定使用
繁體中文，因為 README 是 UTF-8 文字檔，不需要系統字型就能正確顯示中文，只有
Matplotlib 算繪的 PNG 才會因缺字型而出現方框／豆腐字），並在 manifest 的
``font`` 欄位明確記錄「改用英文標籤」與原因，供人工稽核。
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq
from scipy.optimize import least_squares
from shapely.geometry import MultiPolygon, Polygon, shape

from ..geometry import DomainProjection
from ..outputs import sha256_file
from ..report_font import resolve_cjk_font
from .backward_driver import (
    NATIVE_STATUS_DEPOSITED,
    NATIVE_STATUS_FORCING_MISSING,
    NATIVE_STATUS_LEFT_INTEREST_REGION,
    NATIVE_STATUS_MAX_BACKTRACK,
    NATIVE_STATUS_NUMERICAL_FAILURE,
    NATIVE_STATUS_PRE_WINDOW_DEPOSITION,
    NATIVE_STATUS_STRANDED,
)
from .postprocess import DEFAULT_AZIMUTH_SECTOR_COUNT, NON_CROSSING_STATUS_CATEGORY_ZH

__all__ = [
    "FiguresError",
    "NativeFiguresConfig",
    "FIGURE_STEMS",
    "SITE_LABELS_ZH",
    "build_native_figures",
]


class FiguresError(RuntimeError):
    """native-figures 圖表管線的一般性資料契約或一致性錯誤，一律 fail-closed。"""


# ── 站點中文名稱：與 configs/*.yaml 的 study_site_name_zh 一致 ──────────────
#
# 目前正式設計版本五站是 gongliao／guishan／hsinchu／nanwan／lienchiang（``nanwan``
# 取代了舊 pilot 的 ``houwan``，見 ``scenarios.py`` C 區站點交叉檢查）。未登錄在此
# 表的站點（例如測試用的合成站名）直接使用 site_id 本身作標籤，不 fail-closed——
# 這是純呈現用的中文別名表，不是資料契約，未知站名不代表輸入有誤。
SITE_LABELS_ZH: Mapping[str, str] = {
    "gongliao": "貢寮",
    "guishan": "龜山島西側",
    "hsinchu": "新竹外海",
    "nanwan": "南灣",
    "lienchiang": "連江",
}


def _site_label_zh(site_id: str) -> str:
    return SITE_LABELS_ZH.get(site_id, site_id)


# ── 停止狀態：中英文標籤、固定順序、顏色與標記形狀 ──────────────────────────
#
# ``NON_CROSSING_STATUS_CATEGORY_ZH`` 直接重用 postprocess 既有的六種「未穿越」
# 中文分類；這裡只另外補上「成功穿越」（``NATIVE_STATUS_LEFT_INTEREST_REGION``）
# 與對應的英文標籤、順序、顏色、標記，讓 native-figures 的八張圖與 postprocess
# 輸出表共用完全相同的狀態語意，不創造另一套分類。

_CROSSED_LABEL_ZH = "穿越開放邊界（正常結束）"
_CROSSED_LABEL_EN = "Crossed open boundary"

_STATUS_LABELS_ZH: Mapping[int, str] = {
    NATIVE_STATUS_LEFT_INTEREST_REGION: _CROSSED_LABEL_ZH,
    **NON_CROSSING_STATUS_CATEGORY_ZH,
}

_STATUS_LABELS_EN: Mapping[int, str] = {
    NATIVE_STATUS_LEFT_INTEREST_REGION: _CROSSED_LABEL_EN,
    NATIVE_STATUS_MAX_BACKTRACK: "Max backtrack reached",
    NATIVE_STATUS_STRANDED: "Stranded",
    NATIVE_STATUS_DEPOSITED: "Deposited",
    NATIVE_STATUS_FORCING_MISSING: "Forcing data gap",
    NATIVE_STATUS_NUMERICAL_FAILURE: "Numerical failure",
    NATIVE_STATUS_PRE_WINDOW_DEPOSITION: "Pre-window deposition",
}

# 固定順序：成功穿越永遠第一欄，其餘六種依 postprocess 既有分類順序排列，讓多張
# 圖與表格的欄位／圖例順序一致，方便人工比對。
_STATUS_ORDER: tuple[int, ...] = (
    NATIVE_STATUS_LEFT_INTEREST_REGION,
    NATIVE_STATUS_MAX_BACKTRACK,
    NATIVE_STATUS_STRANDED,
    NATIVE_STATUS_DEPOSITED,
    NATIVE_STATUS_FORCING_MISSING,
    NATIVE_STATUS_NUMERICAL_FAILURE,
    NATIVE_STATUS_PRE_WINDOW_DEPOSITION,
)

_STATUS_COLORS: Mapping[int, str] = {
    NATIVE_STATUS_LEFT_INTEREST_REGION: "#4c78a8",
    NATIVE_STATUS_MAX_BACKTRACK: "#54a24b",
    NATIVE_STATUS_STRANDED: "#f58518",
    NATIVE_STATUS_DEPOSITED: "#72b7b2",
    NATIVE_STATUS_FORCING_MISSING: "#ff9da6",
    NATIVE_STATUS_NUMERICAL_FAILURE: "#d62728",
    NATIVE_STATUS_PRE_WINDOW_DEPOSITION: "#b279a2",
}

_STATUS_MARKERS: Mapping[int, str] = {
    NATIVE_STATUS_LEFT_INTEREST_REGION: "s",
    NATIVE_STATUS_MAX_BACKTRACK: "^",
    NATIVE_STATUS_STRANDED: "P",
    NATIVE_STATUS_DEPOSITED: "d",
    NATIVE_STATUS_FORCING_MISSING: "h",
    NATIVE_STATUS_NUMERICAL_FAILURE: "X",
    NATIVE_STATUS_PRE_WINDOW_DEPOSITION: "v",
}

FIGURE_STEMS: tuple[str, ...] = (
    "horizontal_overview",
    "horizontal_local",
    "depth_age",
    "terminal_counts",
    "boundary_azimuth_rose",
    "boundary_arc_segments",
    "source_footprint_kde",
    "settling_velocity_breakdown",
)
"""八張圖固定檔名（不含副檔名），供 manifest、測試與 README 共用同一份清單。"""


class _Labeler:
    """依字型可用性在繁體中文與英文標籤之間切換的最小工具。

    只在「找不到合格繁體中文字型」時才切換成英文（見模組 docstring「字型與雙語
    退場」一節）；README 一律使用中文，不經過這個物件。``t(zh, en)`` 回傳其中
    一種語言的固定字串，呼叫端仍需自行把動態數值（站名、次數、百分比）組進字串。
    """

    __slots__ = ("cjk_available",)

    def __init__(self, cjk_available: bool) -> None:
        self.cjk_available = cjk_available

    def t(self, zh: str, en: str) -> str:
        return zh if self.cjk_available else en


# ── 設定 ─────────────────────────────────────────────────────────────────

FIGURE_TEXT_MODES: tuple[str, ...] = ("full", "none")
"""``--figure-text`` 合法值：``full``＝標題／副標／圖例／座標軸標籤全部畫出
（預設，供內部檢視）；``none``＝只畫座標軸與圖例，不畫標題與副標（供正式報告
排版，文字改由 ``captions.md`` 承載）。見模組 docstring「WP-24b」一節。"""


@dataclass(frozen=True, slots=True)
class NativeFiguresConfig:
    """一次 ``build_native_figures`` 執行的完整設定；全部欄位在建構時驗證。

    ``postprocess_dir`` 是 ``postprocess.PostprocessAccumulator.finalize`` 寫出的
    目錄（只讀）；``coastline_path`` 是 CRS84 ``FeatureCollection`` GeoJSON（只讀）；
    ``output_dir`` 必須是尚不存在的全新目錄。``sites`` 為 ``None`` 時輸出全部在
    ``departures.parquet`` 出現過的站；否則只輸出交集，任何一個站不存在於資料中
    都會 fail-closed（避免打錯站名卻被靜默忽略）。``font_path`` 提供時直接使用該
    字型檔，不經過 ``resolve_cjk_font`` 的必要字元檢查（呼叫端對自備字型負責）；
    省略時走 ``resolve_cjk_font()``，找不到合格字型時整批圖面改用英文標籤（見模組
    docstring）。``max_trajectories_per_site`` 上限代表軌跡樣本數（水平總覽／局部
    ／深度年齡三張圖共用同一批已排序抽樣，避免同一顆粒子在不同圖之間對不上）；
    ``dpi`` 是輸出 PNG 的解析度。``figure_text`` 是 WP-24b 新增的文字層級開關，
    合法值見 :data:`FIGURE_TEXT_MODES`；``"full"``（預設）維持標題／副標／圖例／
    座標軸標籤全部畫出，``"none"`` 只畫座標軸與圖例，不畫標題與副標。兩種模式都
    會輸出 ``captions.md``（見模組 docstring）。
    """

    postprocess_dir: Path
    coastline_path: Path
    output_dir: Path
    sites: tuple[str, ...] | None = None
    font_path: Path | None = None
    max_trajectories_per_site: int = 24
    dpi: int = 150
    figure_text: str = "full"

    def __post_init__(self) -> None:
        object.__setattr__(self, "postprocess_dir", Path(self.postprocess_dir))
        object.__setattr__(self, "coastline_path", Path(self.coastline_path))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.font_path is not None:
            object.__setattr__(self, "font_path", Path(self.font_path))
        if self.sites is not None:
            sites = tuple(self.sites)
            if not sites or any(not isinstance(s, str) or not s for s in sites):
                raise FiguresError("sites 若提供必須是非空字串序列")
            object.__setattr__(self, "sites", sites)
        if self.max_trajectories_per_site < 1:
            raise FiguresError("max_trajectories_per_site 必須是正整數")
        if self.dpi < 1:
            raise FiguresError("dpi 必須是正整數")
        if self.figure_text not in FIGURE_TEXT_MODES:
            raise FiguresError(
                f"figure_text 必須是 {FIGURE_TEXT_MODES} 之一，收到：{self.figure_text!r}"
            )


# ── 表格讀取：優先 Parquet，缺 Parquet 才退回同名 CSV ───────────────────────


def _read_table_any(
    directory: Path, stem: str, *, required: bool = True
) -> tuple[pa.Table | None, Path | None]:
    """讀取 ``<stem>.parquet``（優先）或 ``<stem>.csv``；都不存在時依 required 決定。

    正式 ``postprocess.py`` 兩種格式都會寫（見該模組 ``_write_table_with_csv``），
    這裡容許只有 CSV 的目錄（例如工作包驗收用的縮減版真實輸出），但一律優先信任
    Parquet 的型別（CSV 經 ``pyarrow.csv`` 讀回後數值型別仍會正確推斷，字串型別
    也一致，因為兩者由同一張表寫出）。``required=False`` 且都不存在時回傳
    ``(None, None)``，交由呼叫端決定是否要有替代資料來源；``required=True`` 時
    直接 fail-closed，訊息明確列出嘗試過的兩個檔名。
    """

    parquet_path = directory / f"{stem}.parquet"
    if parquet_path.is_file():
        return pq.read_table(parquet_path), parquet_path
    csv_path = directory / f"{stem}.csv"
    if csv_path.is_file():
        return pa_csv.read_csv(csv_path), csv_path
    if required:
        raise FiguresError(f"後處理目錄缺少必要輸入表：{stem}.parquet 或 {stem}.csv")
    return None, None


def _require_columns(table: pa.Table, names: Sequence[str], stem: str) -> None:
    """驗證表格含全部必要欄位，缺欄位時列出確切欄名，不讓後續存取用 KeyError 帶過。"""

    missing = [name for name in names if name not in table.column_names]
    if missing:
        raise FiguresError(f"{stem} 缺少必要欄位：{', '.join(missing)}")


# ── 海岸線：CRS84 FeatureCollection，只作地理參照 ───────────────────────────


def _load_coastline(path: Path) -> list[Polygon | MultiPolygon]:
    """讀取並驗證海岸 GeoJSON，回傳有效 land geometry 清單。

    只驗證結構（CRS84、FeatureCollection、每個 feature 是有效非空 Polygon／
    MultiPolygon），不像舊版 preview 管線那樣鎖定特定 SHA-256 或固定 1905 筆
    ——本模組是通用 native 管線，呼叫端提供的可以是不同版本或測試用的縮小海岸檔。
    海岸只作底圖地理參照，不參與任何粒子有效性判定。
    """

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise FiguresError(f"無法讀取海岸 GeoJSON：{exc}") from exc
    except json.JSONDecodeError as exc:
        raise FiguresError(f"海岸 GeoJSON 不是合法 JSON：{exc}") from exc
    crs_name = None
    crs_block = payload.get("crs") if isinstance(payload, dict) else None
    if isinstance(crs_block, dict):
        crs_name = crs_block.get("properties", {}).get("name")
    if not isinstance(payload, dict) or payload.get("type") != "FeatureCollection":
        raise FiguresError("海岸 GeoJSON 必須是 FeatureCollection")
    if crs_name != "urn:ogc:def:crs:OGC:1.3:CRS84":
        raise FiguresError("海岸 GeoJSON 必須明示 CRS84（經度、緯度）座標順序")
    features = payload.get("features")
    if not isinstance(features, list) or not features:
        raise FiguresError("海岸 GeoJSON features 不得為空")
    geometries: list[Polygon | MultiPolygon] = []
    for index, feature in enumerate(features):
        try:
            geometry = shape(feature["geometry"])
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise FiguresError(f"海岸 GeoJSON feature[{index}] geometry 無法解析") from exc
        if not isinstance(geometry, (Polygon, MultiPolygon)) or not geometry.is_valid or geometry.is_empty:
            raise FiguresError(f"海岸 GeoJSON feature[{index}] 不是有效 Polygon／MultiPolygon")
        geometries.append(geometry)
    return geometries


# ── 字型解析：CJK 優先，找不到就整批改英文（見模組 docstring）──────────────


@dataclass(frozen=True, slots=True)
class _FontResolution:
    font_properties: Any
    cjk_available: bool
    manifest_record: Mapping[str, Any]


def _resolve_font(font_path: Path | None) -> _FontResolution:
    """解析圖面用字型；找不到 CJK 字型時回傳英文退場旗標而不是拋例外。"""

    from matplotlib import font_manager
    from matplotlib.ft2font import FT2Font

    if font_path is not None:
        path = Path(font_path)
        if not path.is_file():
            raise FiguresError(f"指定的 --font-path 不是既有檔案：{path.name}")
        try:
            FT2Font(str(path))
        except Exception as exc:  # FreeType 例外型別依平台而異，統一收斂
            raise FiguresError(f"指定的 --font-path 無法由 FreeType 解析：{path.name}") from exc
        properties = font_manager.FontProperties(fname=str(path))
        record = {
            "mode": "explicit_font_path",
            "filename": path.name,
            "file_sha256": sha256_file(path),
        }
        return _FontResolution(properties, True, record)

    try:
        selection = resolve_cjk_font()
    except ValueError as exc:
        return _FontResolution(
            font_manager.FontProperties(),
            False,
            {
                "mode": "english_fallback",
                "reason": str(exc),
            },
        )
    resolved_path = font_manager.findfont(
        font_manager.FontProperties(family=[selection.family]), fallback_to_default=False
    )
    properties = font_manager.FontProperties(fname=resolved_path)
    record = {
        "mode": "resolved_cjk_font",
        "family": selection.family,
        "filename": selection.filename,
        "file_sha256": selection.file_sha256,
    }
    return _FontResolution(properties, True, record)


# ── 投影中心還原（見模組 docstring「經緯度還原」一節）───────────────────────

_PROJECTION_FIT_SAMPLE_SIZE = 50
"""每站用來解投影中心的樣本點數；50 點遠超過 2 個未知數所需，用來降低單點座標
若剛好落在數值病態位置（理論上不會發生，AEQD 對這個站範圍內處處良態）的風險，
仍遠小於整站粒子數，解算時間可忽略。"""

_PROJECTION_FIT_MAX_RESIDUAL_M = 1.0
"""投影中心反解殘差的可接受上限（公尺）。真實資料驗證顯示殘差可達 1e-9 公尺
（機器精度等級）；1 公尺門檻已經非常寬鬆，只用來攔截「經緯度與公尺座標根本不是
同一個 AEQD 投影」這種資料損毀情況，不是精度目標。"""


def _recover_site_projection_center(
    lon_deg: np.ndarray, lat_deg: np.ndarray, x_m: np.ndarray, y_m: np.ndarray
) -> DomainProjection:
    """由一批 ``(經緯度, 公尺座標)`` 精確樣本對，反解出該站 AEQD 投影中心。

    ``departures.parquet`` 的 ``departure_lon_deg/lat_deg`` 與
    ``departure_x_m/y_m`` 是同一個 AEQD 投影的精確樣本對（不是量測值，沒有雜訊），
    因此只要找到中心經緯度 ``(lon0, lat0)`` 使
    ``DomainProjection(lon0, lat0).project(lon, lat) == (x_m, y_m)`` 對全部樣本點
    同時成立，就找回了正式管線實際使用的投影中心，不是近似或線性內插。初始猜測
    直接用樣本點經緯度平均，最小平方法通常一兩次疊代就收斂到浮點極限；若殘差超過
    ``_PROJECTION_FIT_MAX_RESIDUAL_M``，代表輸入的經緯度與公尺欄位不是一致的
    AEQD 樣本（資料損毀或欄位對錯），直接 fail-closed，不回傳可疑的投影。
    """

    if lon_deg.size == 0:
        raise FiguresError("無法反解投影中心：站內沒有任何含經緯度的離開紀錄")
    guess = (float(np.mean(lon_deg)), float(np.mean(lat_deg)))

    def residuals(center: np.ndarray) -> np.ndarray:
        lon0, lat0 = float(center[0]), float(center[1])
        if not (-180.0 <= lon0 <= 180.0 and -90.0 < lat0 < 90.0):
            return np.full(2 * lon_deg.size, 1.0e9)
        projection = DomainProjection(lon0, lat0)
        proj_x, proj_y = projection.project(lon_deg, lat_deg)
        return np.concatenate([proj_x - x_m, proj_y - y_m])

    result = least_squares(residuals, guess, method="lm", xtol=1e-14, ftol=1e-14, max_nfev=200)
    max_residual = float(np.max(np.abs(residuals(result.x)))) if lon_deg.size else math.inf
    if not result.success or not math.isfinite(max_residual) or max_residual > _PROJECTION_FIT_MAX_RESIDUAL_M:
        raise FiguresError(
            "無法由離開紀錄反解出一致的 AEQD 投影中心（殘差 "
            f"{max_residual:.6g} 公尺，超過門檻 {_PROJECTION_FIT_MAX_RESIDUAL_M} 公尺）；"
            "經緯度與公尺欄位不是同一個投影的一致樣本對"
        )
    return DomainProjection(float(result.x[0]), float(result.x[1]))


# ── 每站資料結構 ─────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class TrajectorySample:
    """單顆代表粒子的軌跡樣本，已轉回經緯度並標好色階序位。

    ``age_hours`` 是相對到達時刻的回溯時齡（小時，0 為到達／回溯起點，隨陣列
    遞增），直接由 ``trajectory_utc_ns`` 換算，不重新取樣或插值。``rank_index``
    是這顆粒子在同一站已抽樣批次中依起始深度（``z_m[0]``，海面為零、向上為正，
    越負越深）排序後的序位（1 起算），**只用來分色與畫圖，不代表固定物理水層**
    ——native 後處理沒有保留 ``vertical_id``（見 ``postprocess`` 模組 docstring
    第 3 點揭露），序位只是「這批代表粒子裡，這顆的起始深度排第幾」。

    ``local_first_exit_lon_deg``／``lat_deg``（WP-23 新增）是這顆粒子**首次離開
    自己站點 local domain**（非終止，粒子當時仍繼續回溯）的經緯度；只有
    ``local_equals_flow=False`` 的測站（目前是貢寮、龜山島）且這顆粒子確實有對應
    ``local_domain_departures.parquet`` 列時才有值，其餘一律 ``None``——
    ``local_equals_flow=True`` 的測站（新竹／南灣／連江）不需要這兩個欄位，因為
    那個事件與終止事件是同一個點，直接沿用終止標記即可，見模組 docstring
    「WP-23」一節。
    """

    particle_id: str
    receptor_id: str | None
    final_status_code: int
    final_status_description_zh: str
    age_hours: np.ndarray
    z_m: np.ndarray
    lon_deg: np.ndarray
    lat_deg: np.ndarray
    rank_index: int
    rank_total: int
    local_first_exit_lon_deg: float | None = None
    local_first_exit_lat_deg: float | None = None


@dataclass(frozen=True, slots=True)
class SiteData:
    """單一站點的全部繪圖輸入，已完成投影、篩選與型別轉換。

    ``departures`` 是這一站全部離開紀錄，攤平成欄名到 NumPy 陣列的 dict（保留
    ``departures.parquet`` 原始列順序），供各圖表獨立重新聚合方位扇區、弧段、
    沉降速度等統計，不依賴那些屬於可選、允許缺席的聚合表（見模組 docstring）。``kde`` 為
    ``None`` 表示 ``kde_by_site`` 沒有這一站的列（理論上不應發生，
    ``PostprocessAccumulator`` 對每個處理過的站一定會有一列，但仍保守處理避免
    對不存在的鍵盲目索引）。

    以下四個欄位是 WP-23 新增，皆來自 ``local_domain_first_exit_summary_by_site``：
    ``local_equals_flow`` 為 ``True``／``False``／``None``（``None``＝
    ``postprocess.py`` 執行時沒有提供 ``boundary_geometries``，無法判定，見該表
    ``local_first_exit_source="unknown_local_equals_flow"``）；
    ``local_first_exit_particle_count``／``local_first_exit_fraction`` 是這一站
    「曾經離開過自己 local domain」的粒子數與佔全站比例，分子依 ``local_equals_
    flow`` 取自兩種互斥來源之一（見 ``postprocess`` 模組 docstring「WP-23」一節），
    不是本模組自行推算；``local_departures_by_row`` 是
    ``row_particle_index -> local_domain_departures.parquet`` 該列欄位 dict 的
    對照，只有 ``local_equals_flow=False`` 的測站才會非空（``local_equals_
    flow=True`` 的測站，這個事件已經是 ``departures.parquet`` 該列
    ``also_local_domain_first_exit=True``，不會出現在 local_domain_departures，
    對照表本身就是空的，不代表資料缺漏）。
    """

    site_id: str
    site_label_zh: str
    projection: DomainProjection
    departures: Mapping[str, np.ndarray]
    trajectories: tuple[TrajectorySample, ...]
    kde: Mapping[str, Any] | None
    summary: Mapping[str, Any]
    arc_segments_by_boundary: Mapping[str, tuple[Mapping[str, Any], ...]]
    azimuth_sector_count: int
    boundary_arc_segment_count: int
    local_equals_flow: bool | None
    local_first_exit_particle_count: int
    local_first_exit_fraction: float
    local_departures_by_row: Mapping[int, Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class NativeFiguresInputs:
    """一次載入完成的全部輸入：站點資料、postprocess 執行 manifest 與來源快照。"""

    sites: tuple[SiteData, ...]
    postprocess_manifest: Mapping[str, Any]
    sources_before: Mapping[str, Mapping[str, Any]]
    source_paths: Mapping[str, Path]


_DEPARTURE_REQUIRED_COLUMNS: tuple[str, ...] = (
    "particle_id",
    "study_site_id",
    "receptor_id",
    "settling_velocity_mps",
    "final_status_code",
    "final_status_description_zh",
    "crossed_interest_region_boundary",
    "also_local_domain_first_exit",
    "departure_lon_deg",
    "departure_lat_deg",
    "departure_x_m",
    "departure_y_m",
    "row_particle_index",
    "backtrack_duration_seconds",
    "departure_azimuth_sector_index",
    "departure_azimuth_sector_name",
    "boundary_segment_id",
    "boundary_arc_segment_index",
    "boundary_arc_length_position_m",
)

_TRAJECTORY_REQUIRED_COLUMNS: tuple[str, ...] = (
    "study_site_id",
    "row_particle_index",
    "particle_id",
    "trajectory_utc_ns",
    "trajectory_x_m",
    "trajectory_y_m",
    "trajectory_z_m",
)

_KDE_REQUIRED_COLUMNS: tuple[str, ...] = (
    "study_site_id",
    "status",
    "raw_point_count",
    "bandwidth_factor",
    "hdr_levels",
    "grid_rows",
    "grid_cols",
    "x_centers_m",
    "y_centers_m",
    "density_per_m2_flat",
    "cell_probability_flat",
    "hdr_masks_flat",
)

_SUMMARY_REQUIRED_COLUMNS: tuple[str, ...] = (
    "study_site_id",
    "total_particle_count",
    "fraction_crossed_interest_region_boundary",
    "fraction_max_backtrack",
    "fraction_stranded",
    "fraction_deposited",
    "fraction_forcing_missing",
    "fraction_numerical_failure",
    "fraction_pre_window_deposition",
)

_ARC_GEOMETRY_REQUIRED_COLUMNS: tuple[str, ...] = (
    "study_site_id",
    "base_boundary_segment_id",
    "arc_segment_index",
    "arc_segment_id",
    "start_x_m",
    "start_y_m",
    "end_x_m",
    "end_y_m",
    "midpoint_x_m",
    "midpoint_y_m",
    "start_lon_deg",
    "start_lat_deg",
    "end_lon_deg",
    "end_lat_deg",
    "midpoint_lon_deg",
    "midpoint_lat_deg",
)

# ── WP-23：local domain 首次離開（非終止事件）相關兩張表的必要欄位 ──────────

_LOCAL_DEPARTURE_REQUIRED_COLUMNS: tuple[str, ...] = (
    "study_site_id",
    "row_particle_index",
    "local_first_exit_x_m",
    "local_first_exit_y_m",
    "local_first_exit_lon_deg",
    "local_first_exit_lat_deg",
    "local_departure_azimuth_sector_index",
    "local_departure_azimuth_sector_name",
    "local_boundary_segment_id",
)
"""``local_domain_departures.parquet``（``postprocess.LOCAL_DOMAIN_DEPARTURE_TABLE_
SCHEMA``）只有 ``local_equals_flow=False`` 的測站（目前是貢寮、龜山島）才會有列；
本模組只取繪圖需要的欄位子集，不要求呼叫端提供 ``material_id``／``season`` 等與
繪圖無關的分類欄位。"""

_LOCAL_SUMMARY_REQUIRED_COLUMNS: tuple[str, ...] = (
    "study_site_id",
    "local_equals_flow",
    "total_particle_count",
    "local_first_exit_particle_count",
    "fraction_local_first_exit",
    "local_first_exit_source",
)


def _column_to_numpy(table: pa.Table, name: str) -> np.ndarray:
    """把一欄轉成 NumPy 陣列；字串或允許含 null 的欄位改用逐元素 Python list 轉換，
    避免 ``to_numpy(zero_copy_only=True)`` 對非固定寬度型別拋例外。"""

    column = table.column(name)
    return np.array(column.to_pylist())


def _site_ids_in_table(table: pa.Table, column: str = "study_site_id") -> list[str]:
    values = pc.unique(table.column(column)).to_pylist()
    return sorted(str(v) for v in values if v is not None)


def _snapshot_sources(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    """記錄每個輸入檔的 basename、位元組數與 SHA-256；不保存絕對路徑。

    manifest 不得含牆鐘時間以外無法重現的欄位（工作包明確要求）；本機絕對路徑在
    不同機器上不可重現，因此只保存 basename，可重建性完全由 SHA-256 保證。
    """

    result: dict[str, dict[str, Any]] = {}
    for label, path in paths.items():
        result[label] = {
            "basename": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return result


def _load_manifest_json(postprocess_dir: Path) -> tuple[dict[str, Any], Path]:
    path = postprocess_dir / "manifest.json"
    if not path.is_file():
        raise FiguresError("後處理目錄缺少必要輸入：manifest.json")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FiguresError(f"manifest.json 不是合法 JSON：{exc}") from exc
    if not isinstance(payload, dict):
        raise FiguresError("manifest.json 頂層必須是物件")
    return payload, path


def _require_strict_parquet(directory: Path, stem: str) -> tuple[pa.Table, Path]:
    """讀取只能是 Parquet 的表（含陣列欄位，CSV 無法安全round-trip），缺檔即 fail-closed。"""

    path = directory / f"{stem}.parquet"
    if not path.is_file():
        raise FiguresError(f"後處理目錄缺少必要輸入表（僅接受 Parquet）：{stem}.parquet")
    return pq.read_table(path), path


def _filter_site(table: pa.Table, site_id: str) -> pa.Table:
    mask = pc.equal(table.column("study_site_id"), site_id)
    return table.filter(mask)


def _build_departure_arrays(site_table: pa.Table) -> dict[str, np.ndarray]:
    """把一站的離開紀錄表攤平成欄名到 NumPy 陣列的 dict，供各圖獨立重新聚合。"""

    return {name: _column_to_numpy(site_table, name) for name in _DEPARTURE_REQUIRED_COLUMNS}


def _build_trajectory_samples(
    trajectory_site_table: pa.Table,
    departure_lookup: Mapping[int, tuple[str | None, int, str]],
    *,
    site_id: str,
    max_count: int,
    projection: DomainProjection,
    local_departures_by_row: Mapping[int, Mapping[str, Any]] | None = None,
) -> tuple[TrajectorySample, ...]:
    """建立一站的代表軌跡樣本：換算年齡與經緯度、依 ``particle_id`` 取固定子集、
    再依起始深度排序賦予色階序位（見 :class:`TrajectorySample` docstring）。

    ``departure_lookup`` 是 ``row_particle_index -> (receptor_id, final_status_code,
    final_status_description_zh)`` 的對照，來自同一批離開紀錄；若某個代表粒子的
    ``row_particle_index`` 查不到對應離開紀錄，代表兩張表不是同一次後處理輸出，
    直接 fail-closed，不用任何預設值頂替終止狀態。``local_departures_by_row``
    （WP-23 新增，預設空 dict）是同一站的 ``row_particle_index -> local_domain_
    departures.parquet`` 列欄位 dict 對照；查得到就把該筆代表粒子的 local domain
    首次離開經緯度填進 :class:`TrajectorySample`，查不到（多數情況，因為只有
    ``local_equals_flow=False`` 的測站才會有內容，且代表粒子只是全站的一個子集）
    保持 ``None``，不是錯誤。
    """

    local_departures_by_row = local_departures_by_row or {}
    n = trajectory_site_table.num_rows
    if n == 0:
        return ()
    particle_ids = trajectory_site_table.column("particle_id").to_pylist()
    row_indices = trajectory_site_table.column("row_particle_index").to_pylist()
    utc_ns_lists = trajectory_site_table.column("trajectory_utc_ns").to_pylist()
    x_lists = trajectory_site_table.column("trajectory_x_m").to_pylist()
    y_lists = trajectory_site_table.column("trajectory_y_m").to_pylist()
    z_lists = trajectory_site_table.column("trajectory_z_m").to_pylist()

    order = sorted(range(n), key=lambda i: particle_ids[i])[:max_count]

    prepared: list[tuple[float, dict[str, Any]]] = []
    for i in order:
        row_particle_index = int(row_indices[i])
        looked_up = departure_lookup.get(row_particle_index)
        if looked_up is None:
            raise FiguresError(
                f"{site_id} 站代表軌跡樣本（row_particle_index={row_particle_index}）"
                "在離開紀錄表中找不到對應列；trajectory_samples_by_site 與 departures"
                "不是同一次後處理輸出"
            )
        receptor_id, final_status_code, final_status_description_zh = looked_up
        utc_ns = np.asarray(utc_ns_lists[i], dtype=np.int64)
        x_m = np.asarray(x_lists[i], dtype=np.float64)
        y_m = np.asarray(y_lists[i], dtype=np.float64)
        z_m = np.asarray(z_lists[i], dtype=np.float64)
        if utc_ns.size == 0:
            raise FiguresError(f"{site_id} 站代表粒子 {particle_ids[i]} 的軌跡樣本是空陣列")
        age_hours = (float(utc_ns[0]) - utc_ns.astype(np.float64)) / 3.6e12
        lon_deg, lat_deg = projection.unproject(x_m, y_m)
        local_record = local_departures_by_row.get(row_particle_index)
        local_first_exit_lon_deg: float | None = None
        local_first_exit_lat_deg: float | None = None
        if local_record is not None:
            raw_lon = local_record["local_first_exit_lon_deg"]
            raw_lat = local_record["local_first_exit_lat_deg"]
            if raw_lon is not None and raw_lat is not None:
                local_first_exit_lon_deg = float(raw_lon)
                local_first_exit_lat_deg = float(raw_lat)
        prepared.append(
            (
                float(z_m[0]),
                {
                    "particle_id": str(particle_ids[i]),
                    "receptor_id": receptor_id,
                    "final_status_code": final_status_code,
                    "final_status_description_zh": final_status_description_zh,
                    "age_hours": age_hours,
                    "z_m": z_m,
                    "lon_deg": lon_deg,
                    "lat_deg": lat_deg,
                    "local_first_exit_lon_deg": local_first_exit_lon_deg,
                    "local_first_exit_lat_deg": local_first_exit_lat_deg,
                },
            )
        )

    prepared.sort(key=lambda item: item[0])
    total = len(prepared)
    samples: list[TrajectorySample] = []
    for rank, (_initial_z, fields) in enumerate(prepared, start=1):
        samples.append(TrajectorySample(rank_index=rank, rank_total=total, **fields))
    return tuple(samples)


def _build_kde_row(kde_table: pa.Table, site_id: str) -> dict[str, Any] | None:
    mask = pc.equal(kde_table.column("study_site_id"), site_id)
    matched = kde_table.filter(mask)
    if matched.num_rows == 0:
        return None
    if matched.num_rows > 1:
        raise FiguresError(f"kde_by_site 對 {site_id} 有超過一列，資料不一致")
    return {name: matched.column(name)[0].as_py() for name in _KDE_REQUIRED_COLUMNS}


def _build_summary_row(summary_table: pa.Table, site_id: str) -> dict[str, Any]:
    mask = pc.equal(summary_table.column("study_site_id"), site_id)
    matched = summary_table.filter(mask)
    if matched.num_rows == 0:
        raise FiguresError(f"boundary_summary_by_site 沒有 {site_id} 站的列")
    if matched.num_rows > 1:
        raise FiguresError(f"boundary_summary_by_site 對 {site_id} 有超過一列，資料不一致")
    return {name: matched.column(name)[0].as_py() for name in _SUMMARY_REQUIRED_COLUMNS}


def _build_local_departures_by_row(
    local_table: pa.Table, site_id: str
) -> dict[int, dict[str, Any]]:
    """把一站的 ``local_domain_departures.parquet`` 列攤平成
    ``row_particle_index -> 欄位 dict`` 對照（見 :class:`SiteData` docstring）。

    ``local_equals_flow=True`` 的測站在這張表裡完全沒有列，回傳空 dict——這是
    預期行為，不是缺漏（見 ``postprocess`` 模組 docstring「WP-23」一節），呼叫端
    不應對空字典視為錯誤。
    """

    mask = pc.equal(local_table.column("study_site_id"), site_id)
    matched = local_table.filter(mask)
    result: dict[int, dict[str, Any]] = {}
    for index in range(matched.num_rows):
        row = {name: matched.column(name)[index].as_py() for name in _LOCAL_DEPARTURE_REQUIRED_COLUMNS}
        result[int(row["row_particle_index"])] = row
    return result


def _build_local_summary_row(summary_table: pa.Table, site_id: str) -> dict[str, Any]:
    """讀 ``local_domain_first_exit_summary_by_site`` 這一站的一列。

    ``postprocess.py`` 的 ``_write_local_domain_first_exit_summary_table`` 在
    ``finalize()`` 無條件呼叫，對每個處理過的站一定會有一列（即使
    ``local_equals_flow`` 未知），因此這裡與 ``_build_summary_row`` 一樣，找不到
    就 fail-closed，不用任何預設值頂替。
    """

    mask = pc.equal(summary_table.column("study_site_id"), site_id)
    matched = summary_table.filter(mask)
    if matched.num_rows == 0:
        raise FiguresError(f"local_domain_first_exit_summary_by_site 沒有 {site_id} 站的列")
    if matched.num_rows > 1:
        raise FiguresError(f"local_domain_first_exit_summary_by_site 對 {site_id} 有超過一列，資料不一致")
    return {name: matched.column(name)[0].as_py() for name in _LOCAL_SUMMARY_REQUIRED_COLUMNS}


def _build_arc_segments(
    arc_table: pa.Table, site_id: str
) -> dict[str, tuple[dict[str, Any], ...]]:
    """依 ``base_boundary_segment_id`` 分組、依 ``arc_segment_index`` 排序的弧段幾何。"""

    mask = pc.equal(arc_table.column("study_site_id"), site_id)
    matched = arc_table.filter(mask)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for index in range(matched.num_rows):
        row = {name: matched.column(name)[index].as_py() for name in _ARC_GEOMETRY_REQUIRED_COLUMNS}
        grouped.setdefault(str(row["base_boundary_segment_id"]), []).append(row)
    return {
        base_id: tuple(sorted(rows, key=lambda r: int(r["arc_segment_index"])))
        for base_id, rows in grouped.items()
    }


def load_native_figures_inputs(config: NativeFiguresConfig) -> NativeFiguresInputs:
    """讀取並核對一次 native-figures 執行的全部輸入，回傳每站已就緒的繪圖資料。

    只讀取模組 docstring 列出的八個必要輸入與呼叫端的海岸 GeoJSON；不讀取
    forcing、不重新積分、不改寫任何檔案。任何必要欄位缺席、``--sites`` 指定了
    資料中不存在的站、或投影中心無法一致反解，都在這裡 fail-closed，不進入
    繪圖階段。
    """

    postprocess_dir = config.postprocess_dir
    if not postprocess_dir.is_dir():
        raise FiguresError(f"--postprocess-dir 不是既有目錄：{postprocess_dir}")

    manifest_payload, manifest_path = _load_manifest_json(postprocess_dir)
    departures_table, departures_path = _require_strict_parquet(postprocess_dir, "departures")
    _require_columns(departures_table, _DEPARTURE_REQUIRED_COLUMNS, "departures.parquet")
    trajectory_table, trajectory_path = _require_strict_parquet(
        postprocess_dir, "trajectory_samples_by_site"
    )
    _require_columns(trajectory_table, _TRAJECTORY_REQUIRED_COLUMNS, "trajectory_samples_by_site.parquet")
    kde_table, kde_path = _require_strict_parquet(postprocess_dir, "kde_by_site")
    _require_columns(kde_table, _KDE_REQUIRED_COLUMNS, "kde_by_site.parquet")
    summary_table, summary_path = _read_table_any(postprocess_dir, "boundary_summary_by_site")
    _require_columns(summary_table, _SUMMARY_REQUIRED_COLUMNS, "boundary_summary_by_site")
    arc_table, arc_path = _read_table_any(postprocess_dir, "boundary_arc_segments_geometry_by_site")
    _require_columns(arc_table, _ARC_GEOMETRY_REQUIRED_COLUMNS, "boundary_arc_segments_geometry_by_site")
    local_departures_table, local_departures_path = _require_strict_parquet(
        postprocess_dir, "local_domain_departures"
    )
    _require_columns(
        local_departures_table, _LOCAL_DEPARTURE_REQUIRED_COLUMNS, "local_domain_departures.parquet"
    )
    local_summary_table, local_summary_path = _read_table_any(
        postprocess_dir, "local_domain_first_exit_summary_by_site"
    )
    _require_columns(
        local_summary_table, _LOCAL_SUMMARY_REQUIRED_COLUMNS, "local_domain_first_exit_summary_by_site"
    )

    available_sites = _site_ids_in_table(departures_table)
    if not available_sites:
        raise FiguresError("departures.parquet 沒有任何站點資料")
    if config.sites is not None:
        missing_sites = sorted(set(config.sites) - set(available_sites))
        if missing_sites:
            raise FiguresError(f"--sites 指定了資料中不存在的站點：{', '.join(missing_sites)}")
        selected_sites = [site for site in available_sites if site in set(config.sites)]
    else:
        selected_sites = available_sites

    azimuth_sector_count = DEFAULT_AZIMUTH_SECTOR_COUNT
    boundary_arc_segment_count = 20
    manifest_config = manifest_payload.get("config")
    if isinstance(manifest_config, dict):
        raw_sector_count = manifest_config.get("azimuth_sector_count")
        if isinstance(raw_sector_count, int) and raw_sector_count > 0:
            azimuth_sector_count = raw_sector_count
        raw_arc_count = manifest_config.get("boundary_arc_segment_count")
        if isinstance(raw_arc_count, int) and raw_arc_count > 0:
            boundary_arc_segment_count = raw_arc_count

    # 全表建立一次 row_particle_index -> (receptor_id, final_status_code, 中文說明)
    # 對照，避免逐站重複掃描 5 萬列；用 study_site_id 一併核對避免跨站誤配。
    full_site_ids = departures_table.column("study_site_id").to_pylist()
    full_row_indices = departures_table.column("row_particle_index").to_pylist()
    full_receptor_ids = departures_table.column("receptor_id").to_pylist()
    full_status_codes = departures_table.column("final_status_code").to_pylist()
    full_status_desc = departures_table.column("final_status_description_zh").to_pylist()
    departure_lookup_by_site: dict[str, dict[int, tuple[str | None, int, str]]] = {}
    for site, row_index, receptor_id, status_code, status_desc in zip(
        full_site_ids, full_row_indices, full_receptor_ids, full_status_codes, full_status_desc, strict=True
    ):
        departure_lookup_by_site.setdefault(str(site), {})[int(row_index)] = (
            receptor_id,
            int(status_code),
            str(status_desc),
        )

    sites: list[SiteData] = []
    for site_id in selected_sites:
        site_departures_table = _filter_site(departures_table, site_id)
        departure_arrays = _build_departure_arrays(site_departures_table)
        lon = departure_arrays["departure_lon_deg"].astype(np.float64)
        lat = departure_arrays["departure_lat_deg"].astype(np.float64)
        x_m = departure_arrays["departure_x_m"].astype(np.float64)
        y_m = departure_arrays["departure_y_m"].astype(np.float64)
        if np.any(np.isnan(lon)) or np.any(np.isnan(lat)):
            raise FiguresError(
                f"{site_id} 站的離開紀錄缺少經緯度（departure_lon_deg/lat_deg 為 null）；"
                "native-figures 需要 postprocess 執行時已提供 lonlat_projector"
            )
        sample_size = min(_PROJECTION_FIT_SAMPLE_SIZE, lon.size)
        projection = _recover_site_projection_center(
            lon[:sample_size], lat[:sample_size], x_m[:sample_size], y_m[:sample_size]
        )

        local_departures_by_row = _build_local_departures_by_row(local_departures_table, site_id)

        site_trajectory_table = _filter_site(trajectory_table, site_id)
        trajectories = _build_trajectory_samples(
            site_trajectory_table,
            departure_lookup_by_site.get(site_id, {}),
            site_id=site_id,
            max_count=config.max_trajectories_per_site,
            projection=projection,
            local_departures_by_row=local_departures_by_row,
        )

        summary_row = _build_summary_row(summary_table, site_id)
        total_particle_count = int(summary_row["total_particle_count"])
        if total_particle_count != departure_arrays["final_status_code"].size:
            raise FiguresError(
                f"{site_id} 站 boundary_summary_by_site 的 total_particle_count="
                f"{total_particle_count} 與 departures.parquet 實際列數"
                f"{departure_arrays['final_status_code'].size} 不一致"
            )

        local_summary_row = _build_local_summary_row(local_summary_table, site_id)

        sites.append(
            SiteData(
                site_id=site_id,
                site_label_zh=_site_label_zh(site_id),
                projection=projection,
                departures=departure_arrays,
                trajectories=trajectories,
                kde=_build_kde_row(kde_table, site_id),
                summary=summary_row,
                arc_segments_by_boundary=_build_arc_segments(arc_table, site_id),
                azimuth_sector_count=azimuth_sector_count,
                boundary_arc_segment_count=boundary_arc_segment_count,
                local_equals_flow=local_summary_row["local_equals_flow"],
                local_first_exit_particle_count=int(local_summary_row["local_first_exit_particle_count"]),
                local_first_exit_fraction=float(local_summary_row["fraction_local_first_exit"]),
                local_departures_by_row=local_departures_by_row,
            )
        )

    source_paths: dict[str, Path] = {
        "manifest.json": manifest_path,
        departures_path.name: departures_path,
        trajectory_path.name: trajectory_path,
        kde_path.name: kde_path,
        summary_path.name: summary_path,
        arc_path.name: arc_path,
        local_departures_path.name: local_departures_path,
        local_summary_path.name: local_summary_path,
        "coastline": config.coastline_path,
    }
    sources_before = _snapshot_sources(source_paths)

    return NativeFiguresInputs(
        sites=tuple(sites),
        postprocess_manifest=manifest_payload,
        sources_before=sources_before,
        source_paths=source_paths,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 繪圖層：以下函式全部只讀 ``SiteData``／字型／海岸線，不再碰任何檔案 I/O
# （檔案落地統一交給 :func:`build_native_figures` 排他建立）。
# ═══════════════════════════════════════════════════════════════════════════

_COMPASS_16_ORDER: tuple[str, ...] = (
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
)
"""與 ``postprocess.COMPASS_16_SECTOR_NAMES`` 完全相同的 16 方位順序，這裡獨立
覆寫一份常數只是避免繪圖層與後處理層產生模組間的隱性耦合；數值上必須永遠一致，
兩邊皆由同一份 AGENTS.md 方位定義（正北 0 度、順時針）產生。"""


def _sector_center_azimuth_deg(sector_index: int, sector_count: int) -> float:
    """回傳扇區中心方位角（度），公式與 ``postprocess.classify_azimuth_sector``
    的扇區寬度定義完全一致（``sector_index`` 落在
    ``[sector_index * width, (sector_index + 1) * width)``，中心即中點）。"""

    width = 360.0 / sector_count
    return (sector_index + 0.5) * width


def _crossed_mask(departures: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.asarray(departures["crossed_interest_region_boundary"], dtype=bool)


def _masked_int_array(values: np.ndarray, mask: np.ndarray, *, field_name: str, site_id: str) -> np.ndarray:
    """把已用 ``mask``（通常是「已穿越邊界」）篩選過的欄位轉成 int64 陣列。

    篩選後理論上不應再有缺值（未穿越邊界的列才會是 null，見
    ``postprocess`` 模組對方位扇區／弧段欄位的 nullable 定義）；若仍出現 ``None``，
    代表資料契約被破壞（例如穿越旗標與扇區欄位不同步），直接 fail-closed，不用
    -1 或 0 頂替，避免把資料錯誤誤算進某個扇區或弧段的統計。
    """

    selected = np.asarray(values, dtype=object)[mask]
    if selected.size and any(v is None for v in selected):
        raise FiguresError(
            f"{site_id} 站 {field_name} 在已穿越邊界的列中出現缺值，資料契約不一致"
        )
    return selected.astype(np.int64) if selected.size else np.array([], dtype=np.int64)


@dataclass(frozen=True, slots=True)
class SectorWeight:
    sector_index: int
    sector_name: str
    crossed_count: int
    relative_source_weight: float


def _compute_sector_weights(
    departures: Mapping[str, np.ndarray],
    *,
    sector_count: int,
    site_id: str,
    extra_mask: np.ndarray | None = None,
) -> tuple[SectorWeight, ...]:
    """由離開紀錄逐列重新聚合方位扇區「相對來源權重」（見模組 docstring）。

    聚合公式與 ``postprocess._sector_entry_table`` 完全相同：分母是同一分組內
    （這裡的分組＝該站，或該站再疊上 ``extra_mask`` 額外條件，例如單一沉降速度）
    全部成功穿越開放邊界的粒子數，不是該分組的全部粒子數。只回傳「至少有一顆
    粒子」的扇區；缺席的扇區代表沒有觀測，不是零權重與有觀測的零權重同義——
    呼叫端在畫圖時仍應把兩者都畫成 0，只是這裡不虛構一列。
    """

    mask = _crossed_mask(departures)
    if extra_mask is not None:
        mask = mask & extra_mask
    total = int(np.sum(mask))
    if total == 0:
        return ()
    sector_index = _masked_int_array(
        departures["departure_azimuth_sector_index"], mask, field_name="departure_azimuth_sector_index",
        site_id=site_id,
    )
    counts: dict[int, int] = {}
    for value in sector_index:
        counts[int(value)] = counts.get(int(value), 0) + 1
    results = []
    for index in sorted(counts):
        if not (0 <= index < sector_count):
            raise FiguresError(f"{site_id} 站方位扇區索引 {index} 超出 sector_count={sector_count} 範圍")
        name = (
            _COMPASS_16_ORDER[index]
            if sector_count == len(_COMPASS_16_ORDER)
            else f"sector_{index:03d}"
        )
        count = counts[index]
        results.append(SectorWeight(index, name, count, float(count) / float(total)))
    return tuple(results)


@dataclass(frozen=True, slots=True)
class ArcSegmentWeight:
    base_boundary_segment_id: str
    arc_segment_index: int
    crossed_count: int
    relative_source_weight: float


def _compute_arc_segment_weights(
    departures: Mapping[str, np.ndarray], *, site_id: str
) -> tuple[ArcSegmentWeight, ...]:
    """由離開紀錄逐列重新聚合弧段「相對來源權重」，公式與
    ``postprocess._arc_segment_entry_table`` 完全相同：分母是該站全部成功穿越
    開放邊界的粒子數（跨全部開放邊界線加總，不是各線各自的分母）。"""

    mask = _crossed_mask(departures)
    total = int(np.sum(mask))
    if total == 0:
        return ()
    base_ids = np.asarray(departures["boundary_segment_id"], dtype=object)[mask]
    arc_index = _masked_int_array(
        departures["boundary_arc_segment_index"], mask,
        field_name="boundary_arc_segment_index", site_id=site_id,
    )
    counts: dict[tuple[str, int], int] = {}
    for base_id, index in zip(base_ids, arc_index, strict=True):
        if base_id is None:
            raise FiguresError(f"{site_id} 站已穿越邊界的列缺少 boundary_segment_id")
        key = (str(base_id), int(index))
        counts[key] = counts.get(key, 0) + 1
    results = []
    for (base_id, index) in sorted(counts):
        count = counts[(base_id, index)]
        results.append(ArcSegmentWeight(base_id, index, count, float(count) / float(total)))
    return tuple(results)


def _status_counts(departures: Mapping[str, np.ndarray]) -> dict[int, int]:
    codes = np.asarray(departures["final_status_code"], dtype=np.int64)
    counts: dict[int, int] = {}
    for code in codes:
        counts[int(code)] = counts.get(int(code), 0) + 1
    return counts


# ── 共用版面元件（標題兩行、副標、底部紅字警語、排他落檔）───────────────────


def _boundary_kind(base_boundary_segment_id: str) -> str:
    """回傳一條開放邊界線是 ``"local"``（站點 local domain，記錄用、非終止）、
    ``"flow"``（flow domain，終止邊界）還是 ``"other"``（無法辨識的命名）。

    判斷依據是 ``manifests.py``／``production_wiring.py`` 既有命名慣例：
    ``f"{site_id}_local_open_boundary"``／``f"{flow_domain_id}_flow_open_boundary"``
    （見 ``src/lagrangian_backtracking/manifests.py`` 建構 ``BoundaryGeometry`` 處），
    與 ``postprocess.classify_boundary_segment`` 判別「離開點屬於哪一條線」時看的
    是同一組後綴，不是本模組另創的規則。``local_equals_flow=True`` 的測站（新竹／
    南灣／連江）兩條線座標完全相同但識別碼不同，仍會各自回傳 ``"local"``／
    ``"flow"``——繪圖層因此會疊出兩條重疊但線型不同的線，視覺上剛好呈現「這兩條
    邊界其實是同一條」，不需要特殊判斷去跳過其中一條。
    """

    if base_boundary_segment_id.endswith("_local_open_boundary"):
        return "local"
    if base_boundary_segment_id.endswith("_flow_open_boundary"):
        return "flow"
    return "other"


# 水平總覽／局部／條件式來源足跡三張圖共用的邊界線樣式：flow domain（終止邊界）
# 維持既有藍色虛線；local domain（記錄用、非終止）改用橘色點線，避免與既有
# 「開放邊界＝終止」的視覺記憶混淆；無法辨識類別的線退回灰色點劃線，仍畫出來
# 不隱藏，只是明確標示「未知類別」。
_BOUNDARY_LINE_STYLE_BY_KIND: Mapping[str, Mapping[str, Any]] = {
    "flow": {"color": "#174a70", "linewidth": 1.4, "linestyle": "--"},
    "local": {"color": "#b35806", "linewidth": 1.3, "linestyle": ":"},
    "other": {"color": "#666666", "linewidth": 1.0, "linestyle": "-."},
}

# local domain 首次離開（非終止事件）的標記樣式：空心菱形，邊框色與 local 邊界線
# 同色，讓「這個標記」與「哪條邊界」在視覺上互相對應；刻意用空心（不像終止標記
# 那樣實心）強調「粒子在這裡沒有停止，仍繼續回溯」。
_LOCAL_FIRST_EXIT_MARKER_STYLE: Mapping[str, Any] = {
    "marker": "D",
    "markersize": 6.5,
    "markerfacecolor": "none",
    "markeredgecolor": "#b35806",
    "markeredgewidth": 1.4,
}


def _local_flow_semantics_caption(site: SiteData, t: _Labeler) -> str:
    """回傳水平總覽／局部兩張圖共用的副標第三行：說明 local／flow 兩種邊界與
    標記的差異（WP-23 要求「副標或圖說註明兩者的差別」，見模組 docstring）。

    依 ``site.local_equals_flow`` 給三種文字之一：``True``（同一次穿越，圖上
    只有一個標記）、``False``（兩個事件、兩個標記）、``None``（未知，僅提示
    「本次 postprocess 未提供 boundary_geometries，無法判定」，不臆測）。
    """

    if site.local_equals_flow is True:
        return t.t(
            "local domain 與 flow domain 邊界相同：離開 local domain＝離開 flow domain，"
            "圖上只有一個終止標記，不另畫 local 首次離開標記。",
            "local domain equals flow domain here: leaving local = leaving flow. "
            "Only one termination marker is drawn (no separate local-exit marker).",
        )
    if site.local_equals_flow is False:
        return t.t(
            "空心菱形＝首次離開 local domain（僅記錄，粒子當時仍繼續回溯）；"
            "實心終止標記＝離開 flow domain 開放邊界（真正停止處）。兩者是不同時刻、不同位置的事件。",
            "Open diamond = first exit from local domain (recorded only, tracking continued). "
            "Filled terminal marker = exit from the flow-domain open boundary (actual stop). "
            "These are two distinct events at different times and locations.",
        )
    return t.t(
        "本次後處理未提供 boundary_geometries，無法判定 local domain 與 flow domain 是否相同"
        "（local_equals_flow 未知），圖上不畫 local 首次離開標記。",
        "This postprocess run did not provide boundary_geometries, so local_equals_flow is "
        "unknown; no local-exit marker is drawn.",
    )


def _resolve_dashed_open_boundary_lines(site: SiteData) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """由弧段幾何重建每條開放邊界折線的經緯度點序列（依 arc_segment_index 排序，
    以「首段起點＋各段終點」連成整條折線，不插入任何未登錄的額外點）。"""

    lines: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for base_id, arcs in site.arc_segments_by_boundary.items():
        if not arcs:
            continue
        lon = [float(arcs[0]["start_lon_deg"])] + [float(a["end_lon_deg"]) for a in arcs]
        lat = [float(arcs[0]["start_lat_deg"])] + [float(a["end_lat_deg"]) for a in arcs]
        lines[base_id] = (np.asarray(lon), np.asarray(lat))
    return lines


def _resolve_open_boundary_lines_xy(site: SiteData) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """同 :func:`_resolve_dashed_open_boundary_lines`，但回傳公尺制座標，供局部圖使用。"""

    lines: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for base_id, arcs in site.arc_segments_by_boundary.items():
        if not arcs:
            continue
        x = [float(arcs[0]["start_x_m"])] + [float(a["end_x_m"]) for a in arcs]
        y = [float(arcs[0]["start_y_m"])] + [float(a["end_y_m"]) for a in arcs]
        lines[base_id] = (np.asarray(x), np.asarray(y))
    return lines


def _add_two_line_title(
    fig, t: _Labeler, font: Any, title: str, subtitle: str, *, figure_text: str
) -> None:
    """畫兩行標題（主標＋副標）；``figure_text=="none"`` 時什麼都不畫（見
    :data:`FIGURE_TEXT_MODES`），供 WP-24b 正式報告排版模式使用——文字改由
    ``captions.md`` 承載，圖面本身只留座標軸與圖例。"""

    if figure_text == "none":
        return
    fig.suptitle(title, fontproperties=font, fontsize=17, y=0.985)
    fig.text(0.5, 0.945, subtitle, ha="center", va="top", fontproperties=font, fontsize=10)


def _title_top_margin(figure_text: str, *, full: float, none: float = 0.97) -> float:
    """依文字層級回傳 ``fig.subplots_adjust(top=...)`` 用的頂部邊界。

    ``full`` 模式沿用各圖原本的邊界（留給標題／副標的畫面空間）；``none`` 模式
    不畫標題，因此把繪圖區延伸到更接近畫面頂端，避免留下沒有文字的大片空白。
    """

    return full if figure_text == "full" else none


def _site_subtitle(site: SiteData, t: _Labeler, *, extra: str = "") -> str:
    total = int(site.summary["total_particle_count"])
    crossed_fraction = float(site.summary["fraction_crossed_interest_region_boundary"])
    base = t.t(
        f"站點：{site.site_label_zh}（{site.site_id}）｜粒子數：{total}｜穿越開放邊界比例："
        f"{crossed_fraction * 100:.1f}%｜M=1（含隨機擴散，單一系集成員）",
        f"Site: {site.site_id} | N={total} | crossed fraction: "
        f"{crossed_fraction * 100:.1f}% | M=1 (single ensemble member)",
    )
    return f"{base}{extra}"


def _save_exclusive(fig, path: Path, *, dpi: int) -> None:
    """以排他建立方式落檔（``xb``），輸出目錄若已存在同名檔一律 fail-closed。"""

    with path.open("xb") as stream:
        fig.savefig(stream, format="png", dpi=dpi, bbox_inches="tight")


def _land_lonlat_patches(land_geometries: Sequence[Polygon | MultiPolygon]):
    from shapely.plotting import plot_polygon

    def _draw(ax) -> None:
        for land in land_geometries:
            plot_polygon(
                land, ax=ax, add_points=False, facecolor="#c9c9c9", edgecolor="#858585",
                linewidth=0.25, alpha=0.85, zorder=1,
            )

    return _draw


# ── 八張圖各自的繪製函式 ─────────────────────────────────────────────────


def _plot_horizontal_overview(
    site: SiteData,
    land_geometries: Sequence[Polygon | MultiPolygon],
    t: _Labeler,
    font: Any,
    path: Path,
    dpi: int,
    figure_text: str,
) -> None:
    """代表粒子水平軌跡總覽（整個 flow domain 範圍，經緯度顯示）。"""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 8.5), dpi=150)
    ax.set_facecolor("#eaf3f8")
    _land_lonlat_patches(land_geometries)(ax)
    boundary_lines = _resolve_dashed_open_boundary_lines(site)
    boundary_kinds_present: set[str] = set()
    for base_id, (lon, lat) in boundary_lines.items():
        kind = _boundary_kind(base_id)
        boundary_kinds_present.add(kind)
        style = _BOUNDARY_LINE_STYLE_BY_KIND[kind]
        ax.plot(lon, lat, zorder=2, **style)

    cmap = plt.get_cmap("viridis")
    any_local_marker = False
    for sample in site.trajectories:
        color = cmap(0.0 if sample.rank_total <= 1 else (sample.rank_index - 1) / (sample.rank_total - 1))
        ax.plot(sample.lon_deg, sample.lat_deg, color=color, linewidth=1.0, alpha=0.85, zorder=4)
        ax.plot(
            sample.lon_deg[0], sample.lat_deg[0], marker="o", markersize=5.2, markerfacecolor="#1a9850",
            markeredgecolor="white", markeredgewidth=0.6, linestyle="None", zorder=6,
        )
        # local domain 首次離開（非終止）：只有 local_equals_flow=False 的測站、
        # 且這顆代表粒子確實有對應紀錄時才畫；與下面的終止標記是兩個不同事件
        # （見 _local_flow_semantics_caption），local_equals_flow=True 的測站
        # 這兩個欄位恆為 None，不會多畫一個重疊標記。
        if sample.local_first_exit_lon_deg is not None and sample.local_first_exit_lat_deg is not None:
            any_local_marker = True
            ax.plot(
                sample.local_first_exit_lon_deg, sample.local_first_exit_lat_deg,
                linestyle="None", zorder=6.5, **_LOCAL_FIRST_EXIT_MARKER_STYLE,
            )
        marker = _STATUS_MARKERS.get(sample.final_status_code, "X")
        color_end = _STATUS_COLORS.get(sample.final_status_code, "#d62728")
        ax.plot(
            sample.lon_deg[-1], sample.lat_deg[-1], marker=marker, markersize=7.0, markerfacecolor=color_end,
            markeredgecolor=color_end, linestyle="None", zorder=7,
        )

    all_lon = np.concatenate([s.lon_deg for s in site.trajectories]) if site.trajectories else np.array([0.0])
    all_lat = np.concatenate([s.lat_deg for s in site.trajectories]) if site.trajectories else np.array([0.0])
    for lon, lat in boundary_lines.values():
        all_lon = np.concatenate([all_lon, lon])
        all_lat = np.concatenate([all_lat, lat])
    lon_margin = max((all_lon.max() - all_lon.min()) * 0.08, 0.01)
    lat_margin = max((all_lat.max() - all_lat.min()) * 0.08, 0.01)
    ax.set_xlim(all_lon.min() - lon_margin, all_lon.max() + lon_margin)
    ax.set_ylim(all_lat.min() - lat_margin, all_lat.max() + lat_margin)
    center_lat = float(np.mean(all_lat))
    ax.set_aspect(1.0 / math.cos(math.radians(center_lat)), adjustable="box")
    ax.set_xlabel(t.t("經度（°E）", "Longitude (deg E)"), fontproperties=font)
    ax.set_ylabel(t.t("緯度（°N）", "Latitude (deg N)"), fontproperties=font)
    ax.grid(color="white", linewidth=0.7, alpha=0.8)

    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], marker="o", color="#1a9850", markerfacecolor="#1a9850", linestyle="None",
               label=t.t("回溯起點（受體到達位置）", "Backtrack start (receptor arrival)")),
    ]
    if "flow" in boundary_kinds_present:
        handles.append(
            Line2D(
                [], [], **_BOUNDARY_LINE_STYLE_BY_KIND["flow"],
                label=t.t("flow domain 開放邊界（終止邊界）", "Flow-domain open boundary (terminal)"),
            )
        )
    if "local" in boundary_kinds_present:
        handles.append(
            Line2D(
                [], [], **_BOUNDARY_LINE_STYLE_BY_KIND["local"],
                label=t.t(
                    "local domain 開放邊界（記錄用，非終止）",
                    "Local-domain open boundary (record only, non-terminal)",
                ),
            )
        )
    if "other" in boundary_kinds_present:
        handles.append(
            Line2D(
                [], [], **_BOUNDARY_LINE_STYLE_BY_KIND["other"],
                label=t.t("未分類開放邊界", "Unclassified open boundary"),
            )
        )
    if any_local_marker:
        handles.append(
            Line2D(
                [], [], linestyle="None", **_LOCAL_FIRST_EXIT_MARKER_STYLE,
                label=t.t(
                    "local domain 首次離開（非終止，繼續回溯）",
                    "First exit from local domain (non-terminal, tracking continued)",
                ),
            )
        )
    for code in _STATUS_ORDER:
        handles.append(
            Line2D(
                [], [], marker=_STATUS_MARKERS.get(code, "X"), color=_STATUS_COLORS.get(code, "#d62728"),
                linestyle="None", markersize=7,
                label=t.t(_STATUS_LABELS_ZH.get(code, str(code)), _STATUS_LABELS_EN.get(code, str(code))),
            )
        )
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.05), ncol=4, prop=font, frameon=True,
        fontsize=8,
    )

    _add_two_line_title(
        fig, t, font,
        t.t(f"水平軌跡總覽｜{site.site_label_zh}", f"Horizontal overview | {site.site_id}"),
        _site_subtitle(site, t, extra=t.t("｜線色：代表粒子起始深度序位（淺→深）", "")),
        figure_text=figure_text,
    )
    if figure_text == "full":
        fig.text(
            0.5, 0.905, _local_flow_semantics_caption(site, t), ha="center", va="top",
            fontproperties=font, fontsize=8, wrap=True,
        )
    # 改用 subplots_adjust 明確指定邊界，不用 tight_layout：tight_layout 與
    # ax.set_aspect(..., adjustable="box") 併用時，在某些「經緯度範圍很扁」的
    # 站點（例如本圖含寬闊 flow_open_boundary 外框）會於內部多重繪製階段留下
    # 殘影（已實測，圖例列會疊到一個淡灰色殘影字元），改用固定邊界可穩定避免。
    fig.subplots_adjust(left=0.08, bottom=0.30, right=0.97, top=_title_top_margin(figure_text, full=0.84))
    _save_exclusive(fig, path, dpi=dpi)
    plt.close(fig)


def _plot_horizontal_local(
    site: SiteData, t: _Labeler, font: Any, path: Path, dpi: int, figure_text: str
) -> None:
    """受體附近局部放大（公尺制平移座標，原點取代表粒子回溯起點的平均位置）。"""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 8), dpi=150)
    if not site.trajectories:
        ax.text(0.5, 0.5, t.t("無代表軌跡樣本", "No trajectory samples"), ha="center", va="center")
        _save_exclusive(fig, path, dpi=dpi)
        plt.close(fig)
        return

    x0 = float(np.mean([s.lon_deg[0] for s in site.trajectories]))
    y0 = float(np.mean([s.lat_deg[0] for s in site.trajectories]))
    # 局部圖用經緯度平移到以起點群平均為原點，再乘上等距修正因子換成近似公尺，
    # 只為了「局部放大好讀」，不是新的座標系統——最終仍以度為單位標軸。
    ax.set_facecolor("#eaf3f8")
    boundary_lines = _resolve_dashed_open_boundary_lines(site)
    boundary_kinds_present: set[str] = set()
    for base_id, (lon, lat) in boundary_lines.items():
        kind = _boundary_kind(base_id)
        boundary_kinds_present.add(kind)
        style = _BOUNDARY_LINE_STYLE_BY_KIND[kind]
        # flow domain 邊界通常遠在局部圖視窗之外（局部圖只依代表粒子範圍縮放，
        # 不強制含入邊界，見下方 span 計算只用軌跡點），畫出來只是讓視窗剛好
        # 涵蓋到時能看見；不在範圍內時 matplotlib 自動裁掉，不影響軸範圍。
        ax.plot(lon - x0, lat - y0, zorder=2, **style)
    cmap = plt.get_cmap("viridis")
    any_local_marker = False
    all_lon: list[float] = []
    all_lat: list[float] = []
    for sample in site.trajectories:
        color = cmap(0.0 if sample.rank_total <= 1 else (sample.rank_index - 1) / (sample.rank_total - 1))
        ax.plot(sample.lon_deg - x0, sample.lat_deg - y0, color=color, linewidth=1.3, alpha=0.85, zorder=4)
        ax.plot(
            sample.lon_deg[0] - x0, sample.lat_deg[0] - y0, marker="o", markersize=6.2,
            markerfacecolor="#1a9850", markeredgecolor="white", markeredgewidth=0.7,
            linestyle="None", zorder=6,
        )
        if sample.local_first_exit_lon_deg is not None and sample.local_first_exit_lat_deg is not None:
            any_local_marker = True
            ax.plot(
                sample.local_first_exit_lon_deg - x0, sample.local_first_exit_lat_deg - y0,
                linestyle="None", zorder=6.5, **_LOCAL_FIRST_EXIT_MARKER_STYLE,
            )
        marker = _STATUS_MARKERS.get(sample.final_status_code, "X")
        color_end = _STATUS_COLORS.get(sample.final_status_code, "#d62728")
        ax.plot(
            sample.lon_deg[-1] - x0, sample.lat_deg[-1] - y0, marker=marker, markersize=8.2,
            markerfacecolor=color_end, markeredgecolor=color_end, linestyle="None", zorder=7,
        )
        all_lon.extend((sample.lon_deg - x0).tolist())
        all_lat.extend((sample.lat_deg - y0).tolist())

    span = max(max(all_lon) - min(all_lon), max(all_lat) - min(all_lat), 1e-4) * 1.25
    cx = (max(all_lon) + min(all_lon)) / 2
    cy = (max(all_lat) + min(all_lat)) / 2
    ax.set_xlim(cx - span / 2, cx + span / 2)
    ax.set_ylim(cy - span / 2, cy + span / 2)
    center_lat = y0
    ax.set_aspect(1.0 / math.cos(math.radians(center_lat)), adjustable="box")
    ax.set_xlabel(t.t("經度偏移（度，相對代表粒子起點平均）", "Longitude offset (deg, rel. mean start)"),
                  fontproperties=font)
    ax.set_ylabel(t.t("緯度偏移（度，相對代表粒子起點平均）", "Latitude offset (deg, rel. mean start)"),
                  fontproperties=font)
    ax.grid(color="white", linewidth=0.8)

    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], marker="o", color="#1a9850", markerfacecolor="#1a9850", linestyle="None",
               label=t.t("回溯起點（受體到達位置）", "Backtrack start (receptor arrival)")),
    ]
    if "flow" in boundary_kinds_present:
        handles.append(
            Line2D(
                [], [], **_BOUNDARY_LINE_STYLE_BY_KIND["flow"],
                label=t.t("flow domain 開放邊界（終止邊界）", "Flow-domain open boundary (terminal)"),
            )
        )
    if "local" in boundary_kinds_present:
        handles.append(
            Line2D(
                [], [], **_BOUNDARY_LINE_STYLE_BY_KIND["local"],
                label=t.t(
                    "local domain 開放邊界（記錄用，非終止）",
                    "Local-domain open boundary (record only, non-terminal)",
                ),
            )
        )
    if any_local_marker:
        handles.append(
            Line2D(
                [], [], linestyle="None", **_LOCAL_FIRST_EXIT_MARKER_STYLE,
                label=t.t(
                    "local domain 首次離開（非終止，繼續回溯）",
                    "First exit from local domain (non-terminal, tracking continued)",
                ),
            )
        )
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.045), ncol=2, prop=font, frameon=True,
        fontsize=8,
    )

    _add_two_line_title(
        fig, t, font,
        t.t(f"水平局部放大｜{site.site_label_zh}", f"Horizontal local view | {site.site_id}"),
        _site_subtitle(site, t),
        figure_text=figure_text,
    )
    if figure_text == "full":
        fig.text(
            0.5, 0.905, _local_flow_semantics_caption(site, t), ha="center", va="top",
            fontproperties=font, fontsize=8, wrap=True,
        )
    fig.subplots_adjust(left=0.05, bottom=0.24, right=0.98, top=_title_top_margin(figure_text, full=0.84))
    _save_exclusive(fig, path, dpi=dpi)
    plt.close(fig)


def _plot_depth_age(
    site: SiteData, t: _Labeler, font: Any, path: Path, dpi: int, figure_text: str
) -> None:
    """深度對回溯時齡曲線；線色依垂向抽樣序位（不代表固定物理水層）。"""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=150)
    cmap = plt.get_cmap("viridis")
    for sample in site.trajectories:
        color = cmap(0.0 if sample.rank_total <= 1 else (sample.rank_index - 1) / (sample.rank_total - 1))
        ax.plot(sample.age_hours, sample.z_m, color=color, linewidth=1.2, alpha=0.85)
    ax.axhline(0.0, color="#888888", linewidth=0.8, linestyle=":")
    ax.set_xlabel(t.t("回溯時齡（小時）", "Backtrack age (hours)"), fontproperties=font)
    ax.set_ylabel(
        t.t("深度 z（公尺，海面為 0，向上為正）", "Depth z (m, sea surface = 0, up positive)"),
        fontproperties=font,
    )
    ax.grid(color="#dddddd", linewidth=0.6)

    if site.trajectories:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1, max(site.trajectories[0].rank_total, 1)))
        cbar = fig.colorbar(sm, ax=ax, fraction=0.05, pad=0.02)
        cbar.set_label(
            t.t(
                "起始深度序位（僅供分色，不代表固定物理水層）",
                "Initial-depth rank (color only, not a fixed layer)",
            ),
            fontproperties=font,
        )

    _add_two_line_title(
        fig, t, font,
        t.t(f"深度—時齡曲線｜{site.site_label_zh}", f"Depth vs. age | {site.site_id}"),
        _site_subtitle(site, t),
        figure_text=figure_text,
    )
    fig.subplots_adjust(left=0.02, bottom=0.12, right=0.98, top=_title_top_margin(figure_text, full=0.90))
    _save_exclusive(fig, path, dpi=dpi)
    plt.close(fig)


def _plot_terminal_counts(
    site: SiteData, t: _Labeler, font: Any, path: Path, dpi: int, figure_text: str
) -> None:
    """停止類別計數／比例長條圖；分母與 ``boundary_summary_by_site`` 一致。"""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    counts = _status_counts(site.departures)
    total = int(site.summary["total_particle_count"])
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    labels = [t.t(_STATUS_LABELS_ZH[code], _STATUS_LABELS_EN[code]) for code in _STATUS_ORDER]
    values = [counts.get(code, 0) for code in _STATUS_ORDER]
    fractions = [v / total if total else 0.0 for v in values]
    colors = [_STATUS_COLORS[code] for code in _STATUS_ORDER]
    bars = ax.bar(range(len(_STATUS_ORDER)), fractions, color=colors, edgecolor="white")
    for bar, value, fraction in zip(bars, values, fractions, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{value}\n({fraction * 100:.1f}%)",
            ha="center", va="bottom", fontsize=8, fontproperties=font,
        )
    ax.set_xticks(range(len(_STATUS_ORDER)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontproperties=font, fontsize=8)
    ax.set_ylabel(t.t("佔全站粒子數比例", "Fraction of all particles at this site"), fontproperties=font)
    ax.set_ylim(0, max(fractions + [0.05]) * 1.25)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)

    _add_two_line_title(
        fig, t, font,
        t.t(f"停止類別統計｜{site.site_label_zh}", f"Terminal status counts | {site.site_id}"),
        _site_subtitle(site, t),
        figure_text=figure_text,
    )
    fig.subplots_adjust(left=0.08, bottom=0.34, right=0.98, top=_title_top_margin(figure_text, full=0.90))
    _save_exclusive(fig, path, dpi=dpi)
    plt.close(fig)


def _plot_boundary_azimuth_rose(
    site: SiteData, t: _Labeler, font: Any, path: Path, dpi: int, figure_text: str
) -> None:
    """16 方位扇區相對來源權重極座標玫瑰圖（正北向上、順時針）。"""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    weights = _compute_sector_weights(
        site.departures, sector_count=site.azimuth_sector_count, site_id=site.site_id
    )
    by_index = {w.sector_index: w for w in weights}
    n = site.azimuth_sector_count
    theta = np.radians([_sector_center_azimuth_deg(i, n) for i in range(n)])
    radii = np.array([by_index[i].relative_source_weight if i in by_index else 0.0 for i in range(n)])
    width = 2 * math.pi / n * 0.92

    fig = plt.figure(figsize=(8, 8), dpi=150)
    ax = fig.add_subplot(projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    cmap = plt.get_cmap("viridis")
    max_radius = float(radii.max()) if radii.size and radii.max() > 0 else 1.0
    ax.bar(
        theta, radii, width=width, color=[cmap(r / max_radius) for r in radii],
        edgecolor="white", linewidth=0.6,
    )
    ax.set_xticks(np.radians([i * 360.0 / 16 for i in range(16)]))
    ax.set_xticklabels(
        _COMPASS_16_ORDER if n == 16 else [f"{i}" for i in range(n)], fontproperties=font, fontsize=8
    )
    ax.set_ylabel("")

    top3 = sorted(weights, key=lambda w: w.relative_source_weight, reverse=True)[:3]
    separator = t.t("；", "; ")
    top_text = (
        separator.join(f"{w.sector_name} {w.relative_source_weight * 100:.1f}%" for w in top3)
        if top3
        else t.t("無穿越邊界粒子", "No particles crossed the boundary")
    )
    if figure_text == "full":
        fig.suptitle(
            t.t(
                f"方位扇區相對來源權重（玫瑰圖）｜{site.site_label_zh}",
                f"Azimuth-sector relative source weight | {site.site_id}",
            ),
            fontproperties=font, fontsize=16, y=0.985,
        )
        fig.text(
            0.5, 0.945, t.t(f"前三大方位：{top_text}", f"Top-3 sectors: {top_text}"), ha="center", va="top",
            fontproperties=font, fontsize=10,
        )
        fig.text(0.5, 0.915, _site_subtitle(site, t), ha="center", va="top", fontproperties=font, fontsize=9)
    top_margin = _title_top_margin(figure_text, full=0.82, none=0.92)
    fig.subplots_adjust(left=0.03, bottom=0.14, right=0.97, top=top_margin)
    _save_exclusive(fig, path, dpi=dpi)
    plt.close(fig)


def _short_boundary_label(base_boundary_segment_id: str) -> str:
    """把 ``base_boundary_segment_id`` 縮寫成長條圖 x 軸容得下的短標籤。

    完整 ID（例如 ``northeast_taiwan_common_cache_v3_flow_open_boundary``）是
    跨站共用的快取幾何識別碼，旋轉 90 度全文顯示在 20 段 × 2 條邊界的長條圖
    上會撐爆版面；這裡只依「local」／「flow」兩種既有開放邊界類別給簡稱，
    看不出類別時原樣保留（不臆測縮寫，避免誤導）。完整 ID 仍保留在
    ``boundary_arc_segments_geometry_by_site`` 原始表格中，未遺失任何資訊。
    """

    kind = _boundary_kind(base_boundary_segment_id)
    return kind if kind in ("local", "flow") else base_boundary_segment_id


def _plot_boundary_arc_segments(
    site: SiteData, t: _Labeler, font: Any, path: Path, dpi: int, figure_text: str
) -> None:
    """開放邊界折線依弧段著色（權重色階）＋弧段權重長條圖。"""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    weights = _compute_arc_segment_weights(site.departures, site_id=site.site_id)
    by_key = {(w.base_boundary_segment_id, w.arc_segment_index): w for w in weights}
    max_weight = max((w.relative_source_weight for w in weights), default=0.0) or 1.0
    cmap = plt.get_cmap("viridis")

    fig, (ax_map, ax_bar) = plt.subplots(
        1, 2, figsize=(15, 7), dpi=150, gridspec_kw={"width_ratios": [1.1, 1.0]}
    )
    ax_map.set_facecolor("#eaf3f8")
    all_labels: list[str] = []
    all_values: list[float] = []
    for base_id, arcs in site.arc_segments_by_boundary.items():
        for row in arcs:
            index = int(row["arc_segment_index"])
            weight = by_key.get((base_id, index))
            value = weight.relative_source_weight if weight is not None else 0.0
            color = cmap(value / max_weight)
            ax_map.plot(
                [row["start_lon_deg"], row["end_lon_deg"]], [row["start_lat_deg"], row["end_lat_deg"]],
                color=color, linewidth=3.5, solid_capstyle="butt", zorder=3,
            )
            all_labels.append(f"{_short_boundary_label(base_id)}#{index}")
            all_values.append(value)
    ax_map.set_xlabel(t.t("經度（°E）", "Longitude (deg E)"), fontproperties=font)
    ax_map.set_ylabel(t.t("緯度（°N）", "Latitude (deg N)"), fontproperties=font)
    ax_map.grid(color="white", linewidth=0.6, alpha=0.7)

    # 地圖預設縮放到「有非零權重」的弧段範圍（通常只是其中一條開放邊界線的一段），
    # 不是整個 flow domain 外框——否則權重最高的弧段在全域尺度下往往只有幾個像素
    # 寬，肉眼幾乎看不出顏色差異。全部弧段權重皆為零（該站完全沒有穿越紀錄）時才
    # 退回顯示全部弧段幾何，讓使用者至少看得到邊界線本身在哪裡。
    weighted_points: list[tuple[float, float]] = []
    all_points: list[tuple[float, float]] = []
    for arcs in site.arc_segments_by_boundary.values():
        for row in arcs:
            all_points.append((row["start_lon_deg"], row["start_lat_deg"]))
            all_points.append((row["end_lon_deg"], row["end_lat_deg"]))
            key = (row["base_boundary_segment_id"], int(row["arc_segment_index"]))
            weight = by_key.get(key)
            if weight is not None and weight.relative_source_weight > 0.0:
                weighted_points.append((row["start_lon_deg"], row["start_lat_deg"]))
    focus_points = weighted_points if weighted_points else all_points
    if focus_points:
        lons = [p[0] for p in focus_points]
        lats = [p[1] for p in focus_points]
        lon_margin = max((max(lons) - min(lons)) * 0.25, 0.01)
        lat_margin = max((max(lats) - min(lats)) * 0.25, 0.01)
        ax_map.set_xlim(min(lons) - lon_margin, max(lons) + lon_margin)
        ax_map.set_ylim(min(lats) - lat_margin, max(lats) + lat_margin)
        center_lat = (min(lats) + max(lats)) / 2
    else:
        center_lat = 25.0
    ax_map.set_aspect(1.0 / math.cos(math.radians(center_lat)), adjustable="box")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_weight))
    cbar = fig.colorbar(sm, ax=ax_map, fraction=0.05, pad=0.02)
    cbar.set_label(t.t("相對來源權重", "Relative source weight"), fontproperties=font)

    order = sorted(range(len(all_labels)), key=lambda i: -all_values[i])
    ax_bar.bar(
        range(len(order)), [all_values[i] for i in order],
        color=[cmap(all_values[i] / max_weight) for i in order],
    )
    ax_bar.set_xticks(range(len(order)))
    ax_bar.set_xticklabels([all_labels[i] for i in order], rotation=90, fontsize=6)
    ax_bar.set_ylabel(t.t("相對來源權重", "Relative source weight"), fontproperties=font)
    ax_bar.grid(axis="y", color="#dddddd", linewidth=0.6)

    _add_two_line_title(
        fig, t, font,
        t.t(
            f"弧段相對來源權重｜{site.site_label_zh}",
            f"Boundary-arc relative source weight | {site.site_id}",
        ),
        _site_subtitle(site, t),
        figure_text=figure_text,
    )
    fig.subplots_adjust(left=0.06, bottom=0.2, right=0.98, top=_title_top_margin(figure_text, full=0.88))
    _save_exclusive(fig, path, dpi=dpi)
    plt.close(fig)


def _plot_source_footprint_kde(
    site: SiteData,
    land_geometries: Sequence[Polygon | MultiPolygon],
    t: _Labeler,
    font: Any,
    path: Path,
    dpi: int,
    figure_text: str,
) -> None:
    """由 ``kde_by_site`` 畫條件式來源足跡：密度色階＋HDR 等值線，疊海岸線與開放邊界。"""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8.5), dpi=150)
    ax.set_facecolor("#eaf3f8")

    if site.kde is None or site.kde["status"] != "available":
        status_text = "" if site.kde is None else str(site.kde["status"])
        ax.text(
            0.5, 0.5, t.t(f"此站無可用條件式來源足跡（狀態：{status_text or '缺席'}）",
                          f"No source footprint available (status: {status_text or 'missing'})"),
            ha="center", va="center", transform=ax.transAxes, fontproperties=font,
        )
    else:
        kde = site.kde
        rows, cols = int(kde["grid_rows"]), int(kde["grid_cols"])
        x_centers = np.asarray(kde["x_centers_m"], dtype=np.float64)
        y_centers = np.asarray(kde["y_centers_m"], dtype=np.float64)
        density = np.asarray(kde["density_per_m2_flat"], dtype=np.float64).reshape(rows, cols)
        xx, yy = np.meshgrid(x_centers, y_centers)
        lon_grid, lat_grid = site.projection.unproject(xx.ravel(), yy.ravel())
        lon_grid = lon_grid.reshape(rows, cols)
        lat_grid = lat_grid.reshape(rows, cols)

        mesh = ax.pcolormesh(lon_grid, lat_grid, density, cmap="viridis", shading="auto", zorder=2, alpha=0.9)
        cbar = fig.colorbar(mesh, ax=ax, fraction=0.045, pad=0.02)
        cbar.set_label(
            t.t("條件式來源足跡密度（每平方公尺）", "Conditional source footprint density (per m^2)"),
            fontproperties=font,
        )

        hdr_levels = list(kde["hdr_levels"])
        hdr_masks_flat = np.asarray(kde["hdr_masks_flat"], dtype=bool)
        cell_count = rows * cols
        hdr_colors = ["#d62728", "#f58518", "#4c78a8"]
        for level_index, _level in enumerate(hdr_levels):
            start, stop = level_index * cell_count, (level_index + 1) * cell_count
            mask = hdr_masks_flat[start:stop].reshape(rows, cols)
            if not mask.any():
                continue
            ax.contour(
                lon_grid, lat_grid, mask.astype(np.float64), levels=[0.5],
                colors=[hdr_colors[level_index % len(hdr_colors)]], linewidths=1.6, zorder=5,
            )

        from matplotlib.lines import Line2D

        hdr_handles = [
            Line2D(
                [], [], color=hdr_colors[i % len(hdr_colors)], linewidth=1.6,
                label=f"HDR {level * 100:.0f}%",
            )
            for i, level in enumerate(hdr_levels)
        ]
        fig.legend(
            handles=hdr_handles, loc="lower center", bbox_to_anchor=(0.5, 0.05),
            ncol=len(hdr_handles), prop=font, frameon=True,
        )

    _land_lonlat_patches(land_geometries)(ax)
    boundary_lines = _resolve_dashed_open_boundary_lines(site)
    for base_id, (lon, lat) in boundary_lines.items():
        style = _BOUNDARY_LINE_STYLE_BY_KIND[_boundary_kind(base_id)]
        ax.plot(lon, lat, zorder=6, **style)

    if site.kde is not None and site.kde["status"] == "available":
        x_centers = np.asarray(site.kde["x_centers_m"], dtype=np.float64)
        y_centers = np.asarray(site.kde["y_centers_m"], dtype=np.float64)
        lon_min, lat_min = site.projection.unproject(x_centers.min(), y_centers.min())
        lon_max, lat_max = site.projection.unproject(x_centers.max(), y_centers.max())
        ax.set_xlim(min(float(lon_min), float(lon_max)), max(float(lon_min), float(lon_max)))
        ax.set_ylim(min(float(lat_min), float(lat_max)), max(float(lat_min), float(lat_max)))
        ax.set_aspect(1.0 / math.cos(math.radians(float((lat_min + lat_max) / 2))), adjustable="box")

    ax.set_xlabel(t.t("經度（°E）", "Longitude (deg E)"), fontproperties=font)
    ax.set_ylabel(t.t("緯度（°N）", "Latitude (deg N)"), fontproperties=font)
    ax.grid(color="white", linewidth=0.6, alpha=0.6)

    _add_two_line_title(
        fig, t, font,
        t.t(
            f"條件式來源足跡（核密度）｜{site.site_label_zh}",
            f"Conditional source footprint (KDE) | {site.site_id}",
        ),
        _site_subtitle(site, t),
        figure_text=figure_text,
    )
    fig.subplots_adjust(left=0.02, bottom=0.14, right=0.98, top=_title_top_margin(figure_text, full=0.88))
    _save_exclusive(fig, path, dpi=dpi)
    plt.close(fig)


def _plot_settling_velocity_breakdown(
    site: SiteData, t: _Labeler, font: Any, path: Path, dpi: int, figure_text: str
) -> None:
    """十種沉降速度的穿越比例（長條圖）與主要方位扇區權重比較（熱圖）。"""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    sv_values = sorted(set(float(v) for v in site.departures["settling_velocity_mps"]))
    crossed = _crossed_mask(site.departures)
    sv_array = np.asarray(site.departures["settling_velocity_mps"], dtype=np.float64)

    crossing_fractions = []
    heatmap = np.zeros((len(sv_values), site.azimuth_sector_count))
    for row_index, sv in enumerate(sv_values):
        sv_mask = np.isclose(sv_array, sv)
        total_sv = int(np.sum(sv_mask))
        crossed_sv = int(np.sum(sv_mask & crossed))
        crossing_fractions.append(crossed_sv / total_sv if total_sv else 0.0)
        weights = _compute_sector_weights(
            site.departures, sector_count=site.azimuth_sector_count, site_id=site.site_id,
            extra_mask=sv_mask,
        )
        for w in weights:
            heatmap[row_index, w.sector_index] = w.relative_source_weight

    fig, (ax_bar, ax_heat) = plt.subplots(
        1, 2, figsize=(15, 6.5), dpi=150, gridspec_kw={"width_ratios": [0.55, 1.0]}
    )
    sv_labels = [f"{sv:g}" for sv in sv_values]
    ax_bar.barh(range(len(sv_values)), crossing_fractions, color="#4c78a8")
    ax_bar.set_yticks(range(len(sv_values)))
    ax_bar.set_yticklabels(sv_labels, fontsize=8)
    ax_bar.set_xlabel(t.t("穿越開放邊界比例", "Fraction crossing open boundary"), fontproperties=font)
    ax_bar.set_ylabel(
        t.t("沉降速度（公尺/秒，負值為下沉）", "Settling velocity (m/s, negative = sinking)"),
        fontproperties=font,
    )
    ax_bar.grid(axis="x", color="#dddddd", linewidth=0.6)

    sector_labels = _COMPASS_16_ORDER if site.azimuth_sector_count == 16 else [
        f"{i}" for i in range(site.azimuth_sector_count)
    ]
    # origin="lower"讓熱圖第 0 列（sv_values 最小、下沉最快）畫在圖底部，
    # 與左側 barh 預設「索引 0 在最下面」的列順序一致，兩張子圖才能逐列對照。
    im = ax_heat.imshow(heatmap, aspect="auto", cmap="viridis", origin="lower")
    ax_heat.set_xticks(range(len(sector_labels)))
    ax_heat.set_xticklabels(sector_labels, fontsize=7, rotation=90)
    ax_heat.set_yticks(range(len(sv_values)))
    ax_heat.set_yticklabels(sv_labels, fontsize=8)
    ax_heat.set_xlabel(t.t("方位扇區", "Azimuth sector"), fontproperties=font)
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.045, pad=0.02)
    cbar.set_label(
        t.t(
            "相對來源權重（同沉降速度分組內）",
            "Relative source weight (within same settling velocity)",
        ),
        fontproperties=font,
    )

    _add_two_line_title(
        fig, t, font,
        t.t(f"沉降速度細分｜{site.site_label_zh}", f"Settling-velocity breakdown | {site.site_id}"),
        _site_subtitle(site, t),
        figure_text=figure_text,
    )
    fig.subplots_adjust(left=0.02, bottom=0.14, right=0.98, top=_title_top_margin(figure_text, full=0.88))
    _save_exclusive(fig, path, dpi=dpi)
    plt.close(fig)


def _dispatch_plot(
    stem: str,
    site: SiteData,
    land_geometries: Sequence[Polygon | MultiPolygon],
    t: _Labeler,
    font: Any,
    path: Path,
    dpi: int,
    figure_text: str,
) -> None:
    if stem == "horizontal_overview":
        _plot_horizontal_overview(site, land_geometries, t, font, path, dpi, figure_text)
    elif stem == "horizontal_local":
        _plot_horizontal_local(site, t, font, path, dpi, figure_text)
    elif stem == "depth_age":
        _plot_depth_age(site, t, font, path, dpi, figure_text)
    elif stem == "terminal_counts":
        _plot_terminal_counts(site, t, font, path, dpi, figure_text)
    elif stem == "boundary_azimuth_rose":
        _plot_boundary_azimuth_rose(site, t, font, path, dpi, figure_text)
    elif stem == "boundary_arc_segments":
        _plot_boundary_arc_segments(site, t, font, path, dpi, figure_text)
    elif stem == "source_footprint_kde":
        _plot_source_footprint_kde(site, land_geometries, t, font, path, dpi, figure_text)
    elif stem == "settling_velocity_breakdown":
        _plot_settling_velocity_breakdown(site, t, font, path, dpi, figure_text)
    else:  # pragma: no cover - FIGURE_STEMS 是唯一呼叫來源，這裡只是防禦
        raise FiguresError(f"未知圖表代號：{stem}")


# ═══════════════════════════════════════════════════════════════════════════
# 圖說（captions.md）：WP-24b 新增，圖面不畫警語後，文字改由這裡承載
# ═══════════════════════════════════════════════════════════════════════════


def _experiment_case_label(postprocess_manifest: Mapping[str, Any]) -> str:
    """回傳這批圖對應的物理案例識別碼（供圖說列出「關鍵設定」的物理案例）。

    直接讀 ``postprocess.py`` 自己輸出的 ``manifest.json`` 的
    ``experiment_case_id``；沒有這個欄位時如實寫「未於後處理 manifest 記錄」，
    不臆測案例名稱。
    """

    case_id = postprocess_manifest.get("experiment_case_id")
    return str(case_id) if case_id else "未於後處理 manifest 記錄"


def _figure_caption_zh(stem: str, site: SiteData, *, experiment_case_label: str) -> str:
    """回傳單張圖（某站 × 某圖種）的繁體中文圖說，全部用直述句。

    每段固定交代三件事（見 WP-24b 任務要求）：這張圖是什麼、資料來源欄位、
    關鍵設定（站點、粒子數、``M``、終止邊界定義、物理案例）。全部用直述句陳述
    已量測的事實與方法定義——量得到的數字直接寫數字，量不到的（例如個別離開
    點分類的實際誤差率）不在此揣測，只寫清楚的方法定義。
    """

    total = int(site.summary["total_particle_count"])
    site_desc = f"{site.site_label_zh}（{site.site_id}）"
    settings_tail = (
        f"{site_desc}，{total:,} 顆粒子，M=1（每情境一個系集成員，含隨機擴散），"
        f"終止邊界為 flow domain 開放邊界，物理案例：{experiment_case_label}。"
    )
    if stem == "horizontal_overview":
        body = (
            "代表粒子水平軌跡總覽，涵蓋整個 flow domain 範圍，同時標示 local domain 與 "
            "flow domain 兩條開放邊界、依 final_status_code 著色的終止標記，以及 local "
            "domain 首次離開（非終止）標記。資料來源：trajectory_samples_by_site.parquet "
            "的軌跡座標與時間、departures.parquet 的 final_status_code 與 "
            "also_local_domain_first_exit、local_domain_departures.parquet 的 local 首次 "
            "離開位置、boundary_arc_segments_geometry_by_site 的開放邊界折線幾何。"
        )
    elif stem == "horizontal_local":
        body = (
            "受體附近局部放大圖，座標為經緯度相對代表粒子回溯起點平均位置的偏移量，同樣 "
            "標示 local／flow 兩條開放邊界與 local domain 首次離開標記。資料來源與水平軌跡 "
            "總覽圖相同：trajectory_samples_by_site.parquet、departures.parquet、"
            "local_domain_departures.parquet、boundary_arc_segments_geometry_by_site。"
        )
    elif stem == "depth_age":
        body = (
            "代表粒子深度對回溯時齡曲線，深度為海面以下之公尺數（海面為零、向上為正），"
            "顏色依代表粒子起始深度在同批抽樣中的排序序位分色，序位只用來分色，不對應固定 "
            "物理水層。資料來源：trajectory_samples_by_site.parquet 的 trajectory_z_m 與 "
            "trajectory_utc_ns。"
        )
    elif stem == "terminal_counts":
        body = (
            "七種未穿越邊界的停止類別與一種成功穿越開放邊界的粒子計數與比例長條圖，分母為 "
            "boundary_summary_by_site.total_particle_count。資料來源：departures.parquet 的 "
            "final_status_code 逐列統計，並與 boundary_summary_by_site 的 fraction_* 欄位 "
            "互相核對。"
        )
    elif stem == "boundary_azimuth_rose":
        body = (
            f"{site.azimuth_sector_count} 方位扇區的相對來源權重極座標玫瑰圖（正北向上、"
            "順時針），標出前三大方位扇區名稱與百分比。相對來源權重＝該扇區成功穿越次數除以 "
            "全站成功穿越開放邊界的總次數。資料來源：departures.parquet 的 "
            "departure_azimuth_sector_index 與 departure_azimuth_sector_name。"
        )
    elif stem == "boundary_arc_segments":
        arc_count = sum(len(rows) for rows in site.arc_segments_by_boundary.values())
        body = (
            f"開放邊界弧段相對來源權重，本站共 {arc_count} 段弧段，依權重色階著色並以長條圖 "
            "並列比較。相對來源權重＝該弧段成功穿越次數除以全站成功穿越開放邊界的總次數。"
            "資料來源：boundary_arc_segments_geometry_by_site 的弧段幾何與 "
            "departures.parquet 的 boundary_segment_id、boundary_arc_segment_index。"
        )
    elif stem == "source_footprint_kde":
        body = (
            "條件式來源足跡核密度圖，背景為每平方公尺密度，疊加 50%／75%／90% 高密度區間"
            "（HDR，Highest Density Region）等值線，並疊海岸線與開放邊界作地理參照。資料來源："
            "kde_by_site.parquet。"
        )
    elif stem == "settling_velocity_breakdown":
        body = (
            "十個未校準沉降速度敏感度格點各自的穿越開放邊界比例長條圖，與各沉降速度分組內 "
            "方位扇區相對來源權重熱圖並列比較。資料來源：departures.parquet 的 "
            "settling_velocity_mps 與 departure_azimuth_sector_index，逐沉降速度重新分組 "
            "聚合。"
        )
    else:  # pragma: no cover - FIGURE_STEMS 是唯一呼叫來源，這裡只是防禦
        raise FiguresError(f"未知圖表代號：{stem}")
    return f"{body} {settings_tail}"


def _build_captions(inputs: NativeFiguresInputs) -> dict[str, str]:
    """建立「``<site_id>/<stem>.png`` → 圖說」對照，供 ``captions.md`` 與
    manifest 的 ``output_png_sha256.<key>.caption`` 共用同一份文字（見模組
    docstring「WP-24b」一節：文字只寫一次，圖說與 manifest 互相核對用同一個
    來源，不會出現兩處文字不一致的情況）。"""

    experiment_case_label = _experiment_case_label(inputs.postprocess_manifest)
    captions: dict[str, str] = {}
    for site in inputs.sites:
        for stem in FIGURE_STEMS:
            captions[f"{site.site_id}/{stem}.png"] = _figure_caption_zh(
                stem, site, experiment_case_label=experiment_case_label
            )
    return captions


def _render_captions_markdown(
    inputs: NativeFiguresInputs, captions: Mapping[str, str], *, generated_at_utc: str, figure_text: str
) -> str:
    """把 :func:`_build_captions` 的對照表排版成人類可讀的 Markdown。

    每站一個章節、每張圖一個小節，順序固定依 ``FIGURE_STEMS``，方便報告排版時
    逐段複製圖說。
    """

    lines = [
        f"# WP-21 native 正式成果圖表：圖說（{generated_at_utc}）",
        "",
        f"文字層級：`--figure-text {figure_text}`。本檔案逐圖列出圖說——這張圖是什麼、"
        "資料來源欄位、關鍵設定（站點、粒子數、M、終止邊界定義、物理案例）；文字與 "
        "`manifest.json` 對應圖檔的 `caption` 欄位完全相同，可用 SHA-256 核對圖檔與圖說"
        "是否對得上同一個版本。",
        "",
    ]
    for site in inputs.sites:
        lines.append(f"## {site.site_label_zh}（{site.site_id}）")
        lines.append("")
        for stem in FIGURE_STEMS:
            lines.append(f"### `{stem}.png`")
            lines.append("")
            lines.append(captions[f"{site.site_id}/{stem}.png"])
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ═══════════════════════════════════════════════════════════════════════════
# README 與整體 manifest
# ═══════════════════════════════════════════════════════════════════════════

_README_TEMPLATE = """\
# WP-21：native 正式成果圖表（{generated_at_utc}）

本目錄由 `native-figures` 命令列由既有 native 後處理輸出（`departures.parquet`
等，見下方「輸入來源」）獨立繪製，**不重跑任何積分、不讀取 forcing、不改寫任何
輸入檔**；輸入檔前後 SHA-256 已核對一致（見 `manifest.json`）。文字層級為
`--figure-text {figure_text}`（見下方「每站八張圖」一節）。

## 科學定義與範圍

- **M=1**：每個情境（站點 × 受體 × 沉降速度 × 到達時刻等）只有一個系集成員，
  含隨機擴散。
- 「條件式來源足跡」與「相對來源權重」的定義：給定站點、受體、沉降速度與到達
  時刻等情境設定後，代表粒子穿越各方位扇區／各邊界弧段的相對頻率，是描述這批
  `M=1` 系集在既定設定下分布的統計量。來源機率與因果歸因是另一層科學推論，需要
  另外建立先驗、概似與觀測驗證，本批圖表不提供這一層推論。
- 圖上的終止標記位置是回溯積分停止處的座標，對應 `final_status_code` 七種
  互斥狀態之一（成功穿越 flow domain 開放邊界、達回溯上限、擱淺、沉底、流場
  缺值、數值失敗、`pre_window_deposition`）；停止處是積分終止的座標，不是另外
  查證過的實際污染來源。
- 海岸與開放邊界幾何目前的版本狀態是工程生成版：由既有海岸資料與弧段幾何直接
  取得，尚未經研究團隊另立正式核准發布流程。

## 站點

{site_list}

## WP-23：兩種開放邊界與兩種「離開」事件（請先讀，影響前兩張圖）

研究團隊裁決（`AGENTS.md`、`docs/development/05_decisions_and_risks.md`
D008）：**終止一律以 flow domain 開放邊界為準；離開站點 local domain 是非終止
事件，只記錄、粒子繼續回溯**。因此各站依 `local_equals_flow` 而定，會有一條或
兩條開放邊界線，圖上用不同線型區分：

- **flow domain 開放邊界**（藍色虛線）：真正的終止邊界，粒子離開這條線才停止
  回溯，`terminal_counts.png` 等圖的「穿越開放邊界（正常結束）」指的就是這裡。
- **local domain 開放邊界**（橘色點線）：只是記錄用，粒子離開這條線後仍繼續
  回溯，不影響 `final_status_code`。

對應地，粒子的終點標記（依 `final_status_code` 著色的實心標記）永遠畫在 flow
domain 邊界上；若這顆代表粒子還有一筆「首次離開 local domain」的紀錄
（`local_domain_departures.parquet` 或 `departures.parquet` 的
`also_local_domain_first_exit=True`），會另外疊一個**空心橘色菱形**標記非終止
事件。若某站 `local_equals_flow=True`（目前是新竹、南灣、連江——local domain
與 flow domain 是同一個多邊形），兩個事件是同一次穿越，圖上只會看到一個標記
（終止標記本身），不會另外畫空心菱形；若 `local_equals_flow=False`（目前是
貢寮、龜山島），兩個標記通常出現在軌跡上不同位置。詳細語意見每張圖下方的
「local domain 與 flow domain 的差別」副標文字。

## 每站八張圖

各站資料夾（`<site_id>/`）內固定輸出以下八張 PNG。`--figure-text full`（預設）
在圖面上同時畫出標題、副標、圖例與座標軸標籤；`--figure-text none` 只畫座標軸
與圖例，不畫標題與副標，供正式報告排版——每張圖對應的文字改由 `captions.md`
承載（逐圖一段，內容與下列條列一致，並與 `manifest.json` 的
`output_png_sha256.<site_id>/<stem>.png.caption` 完全相同）：

1. `horizontal_overview.png`：代表粒子水平軌跡總覽（經緯度顯示，涵蓋整個 flow
   domain 範圍），**同時畫出 local domain 與 flow domain 兩條開放邊界**（見上
   一節）。資料來源：`trajectory_samples_by_site`（軌跡座標、狀態）、
   `departures.parquet`（起訖狀態對照，含 `also_local_domain_first_exit`）、
   `local_domain_departures.parquet`（local 首次離開位置）、
   `boundary_arc_segments_geometry_by_site`（開放邊界折線）。綠點＝回溯起點
   （受體到達位置）；flow domain 終點依 `final_status_code` 用不同顏色與標記
   形狀表示；空心橘色菱形＝local domain 首次離開（非終止）；線色依代表粒子
   起始深度序位（淺→深）。
2. `horizontal_local.png`：受體附近局部放大，座標為經緯度相對代表粒子回溯起點
   平均位置的偏移量，**同樣疊 local／flow 兩條開放邊界與 local 首次離開標記**
   （舊版本圖完全不畫邊界，WP-23 起新增）。資料來源同上。
3. `depth_age.png`：深度（`z_m`，海面為 0、向上為正）對回溯時齡（小時）曲線，
   顏色依代表粒子的垂向抽樣序位分色——**序位不代表固定物理水層**，native 後
   處理沒有保留原始的 `vertical_id`。資料來源：`trajectory_samples_by_site`。
4. `terminal_counts.png`：七種未穿越邊界的停止類別＋一種成功穿越開放邊界，計數
   與佔全站粒子比例；分母＝`boundary_summary_by_site.total_particle_count`，與
   該表的 `fraction_*` 欄位互相印證。資料來源：`departures.parquet`（逐列統計）、
   `boundary_summary_by_site`（分母核對）。
5. `boundary_azimuth_rose.png`：16 方位扇區「相對來源權重」極座標玫瑰圖（正北
   向上、順時針），並標出前三大扇區名稱與百分比。**相對來源權重**＝該扇區穿越
   次數 ÷ 全站成功穿越邊界的總次數，聚合公式與 `postprocess._sector_entry_table`
   完全相同，只是直接由 `departures.parquet` 的
   `departure_azimuth_sector_index`／`_name` 逐列重新聚合，不依賴另外的聚合表。
6. `boundary_arc_segments.png`：開放邊界折線依弧段著色（權重色階）＋弧段權重長
   條圖，弧段編號與座標取自 `boundary_arc_segments_geometry_by_site`；權重聚合
   公式與 `postprocess._arc_segment_entry_table` 相同，一樣直接由
   `departures.parquet` 的 `boundary_segment_id`／`boundary_arc_segment_index`
   逐列重新聚合。
7. `source_footprint_kde.png`：由 `kde_by_site` 畫出**條件式來源足跡**——背景是
   核密度（每平方公尺），疊加 50%／75%／90% HDR（高密度區間，Highest Density
   Region）等值線，再疊海岸線與開放邊界作地理參照。**這是條件式來源足跡，不是
   絕對來源機率**。
8. `settling_velocity_breakdown.png`：十種沉降速度（`settling_velocity_mps`）各
   自的穿越開放邊界比例（長條圖）與各沉降速度分組內的方位扇區相對來源權重
   （熱圖）。資料來源：`departures.parquet`（逐沉降速度重新分組聚合）。

## 已知限制與方法揭露

- **M=1**：每個情境（站點 × 受體 × 沉降速度 × 到達時刻等）只有一個系集成員
  （含隨機擴散），本批圖表不呈現、也不能呈現系集離散度或不確定性區間。
- 方位扇區與弧段的「相對來源權重」分母是**同一分組內成功穿越開放邊界的粒子
  數**，不是該分組全部粒子數；未穿越邊界的粒子不參與方位／弧段統計，但仍計入
  `terminal_counts.png` 與 `boundary_summary_by_site` 的整體分母。
- `boundary_segment_id`（離開點最終判定所屬的開放邊界）採用「離開點到候選開放
  邊界線最短距離」的啟發式（見 `postprocess.classify_boundary_segment`
  docstring），是近似判定，不是精確幾何交點。
- 海岸與開放邊界幾何目前的版本狀態見上方「科學定義與範圍」；`horizontal_overview`
  等圖以「開放邊界折線（由弧段起訖點重建）」近似完整 domain 外框，本管線不讀
  domain／open manifest，因此不畫閉合陸域邊界，只畫開放水域邊界。
- 深度—時齡曲線的「序位」只是同一批代表粒子按起始深度排序後的名次，同一序位
  在不同站、不同受體之間**不對應同一水層**。
- `boundary_azimuth_rose.png`／`boundary_arc_segments.png` 目前只呈現 flow
  domain 語意的相對來源權重；local domain 首次離開的方位／弧段分布**尚未**
  另外出圖，只能從 `local_domain_departures.parquet`（或
  `local_domain_first_exit_summary_by_site` 的 `fraction_local_first_exit`）
  自行查閱。

## 輸入來源

見 `manifest.json` 的 `inputs_sha256_before`／`inputs_sha256_after`（相同代表本
批繪圖沒有改動任何輸入檔）與 `output_png_sha256`（每張 PNG 的 SHA-256）；八個
必要輸入表見本模組 docstring（`departures.parquet`、
`trajectory_samples_by_site.parquet`、`kde_by_site.parquet`、
`boundary_summary_by_site`、`boundary_arc_segments_geometry_by_site`、
`local_domain_departures.parquet`、`local_domain_first_exit_summary_by_site`、
`manifest.json`）。

## 重建指令

```
{command}
```
"""


def _build_readme(inputs: NativeFiguresInputs, config: NativeFiguresConfig, command: str) -> str:
    site_list = "\n".join(f"- `{s.site_id}`（{s.site_label_zh}）" for s in inputs.sites)
    return _README_TEMPLATE.format(
        generated_at_utc=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        figure_text=config.figure_text,
        site_list=site_list,
        command=command,
    )


def build_native_figures(config: NativeFiguresConfig, *, command: str) -> Path:
    """執行一次完整 native-figures 產出：讀輸入、畫八張圖、寫 README 與 manifest。

    ``command`` 是呼叫端（通常是 ``cli.py``）重建出的完整命令列字串，直接寫進
    manifest 與 README，供人工重建；本函式本身不解析命令列參數。``output_dir``
    必須尚不存在，整個目錄樹（含每站子目錄與全部 PNG）都以排他方式建立，任何
    一步失敗都不會覆寫既有檔案（但已寫出的部分檔案會留在半完成的新目錄裡，供
    診斷，呼叫端可自行刪除後重試）。
    """

    output_dir = config.output_dir
    if output_dir.exists():
        raise FiguresError(f"--output-dir 必須是尚不存在的全新目錄，但已存在：{output_dir}")

    inputs = load_native_figures_inputs(config)
    land_geometries = _load_coastline(config.coastline_path)
    font_resolution = _resolve_font(config.font_path)
    labeler = _Labeler(font_resolution.cjk_available)

    output_dir.mkdir(parents=True)
    captions = _build_captions(inputs)
    png_records: dict[str, dict[str, Any]] = {}
    for site in inputs.sites:
        site_dir = output_dir / site.site_id
        site_dir.mkdir()
        for stem in FIGURE_STEMS:
            png_path = site_dir / f"{stem}.png"
            _dispatch_plot(
                stem, site, land_geometries, labeler, font_resolution.font_properties, png_path, config.dpi,
                config.figure_text,
            )
            key = f"{site.site_id}/{stem}.png"
            png_records[key] = {
                "sha256": sha256_file(png_path),
                "size_bytes": png_path.stat().st_size,
                "caption": captions[key],
            }

    sources_after = _snapshot_sources(inputs.source_paths)
    if sources_after != dict(inputs.sources_before):
        raise FiguresError("輸入檔在繪圖過程中被改動，前後 SHA-256 不一致，fail-closed")

    generated_at_utc = datetime.now(UTC)
    manifest = {
        "generated_at_utc": generated_at_utc.isoformat(),
        "command": command,
        "sites": [s.site_id for s in inputs.sites],
        "site_labels_zh": {s.site_id: s.site_label_zh for s in inputs.sites},
        "figure_stems": list(FIGURE_STEMS),
        "font": dict(font_resolution.manifest_record),
        "cjk_labels_used": font_resolution.cjk_available,
        "inputs_sha256_before": dict(inputs.sources_before),
        "inputs_sha256_after": sources_after,
        "output_png_sha256": png_records,
        "postprocess_manifest_experiment_case_id": inputs.postprocess_manifest.get("experiment_case_id"),
        "postprocess_manifest_master_seed": inputs.postprocess_manifest.get("master_seed"),
        "config": {
            "postprocess_dir_basename": config.postprocess_dir.name,
            "coastline_basename": config.coastline_path.name,
            "max_trajectories_per_site": config.max_trajectories_per_site,
            "dpi": config.dpi,
            "sites_filter": list(config.sites) if config.sites is not None else None,
            "figure_text": config.figure_text,
        },
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")

    captions_path = output_dir / "captions.md"
    captions_text = _render_captions_markdown(
        inputs, captions,
        generated_at_utc=generated_at_utc.strftime("%Y-%m-%d %H:%M UTC"),
        figure_text=config.figure_text,
    )
    with captions_path.open("x", encoding="utf-8") as stream:
        stream.write(captions_text)

    readme_path = output_dir / "README.md"
    readme_text = _build_readme(inputs, config, command)
    with readme_path.open("x", encoding="utf-8") as stream:
        stream.write(readme_text)

    return output_dir
