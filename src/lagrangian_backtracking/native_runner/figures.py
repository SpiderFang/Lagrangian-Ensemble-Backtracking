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
forcing、不重新積分、不改寫任何輸入**，輸出五站各八張圖（WP-25 起每張圖同時
輸出 PNG 與 SVG 兩種格式，見「WP-25：對齊 OCM-SVD-Analysis 正式成圖規範」
一節）、一份 README 與一份 manifest。舊版管線的任何行為都不因本模組存在而
改變。

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

WP-25：水平軌跡圖移除 local domain 圖層（僅圖面，資料層不變）
--------------------------------------------------------------
研究團隊裁決：``horizontal_overview.png``／``horizontal_local.png`` 不再畫出
local domain 的橘色點線邊界與 local 首次離開的橘色空心菱形標記，讓貢寮／
龜山島（A 區，``local_equals_flow=False``）與新竹／南灣／連江
（``local_equals_flow=True``）五站的圖面版式完全一致——圖上一律只保留 flow
domain 開放邊界（終止邊界）與依 ``final_status_code`` 著色的終止標記。這是
**純圖面裁決**：``local_domain_departures.parquet``、
``local_domain_first_exit_summary_by_site``、``departures.parquet`` 的
``also_local_domain_first_exit`` 欄位一律照常產出（見上一節「WP-23」），
:class:`SiteData` 與 :class:`TrajectorySample` 的 ``local_first_exit_*`` 欄位
也維持照常填值；只是本模組不再把這兩個欄位畫成任何圖層。若需要 local domain
首次離開的空間分布，仍可直接查詢上述三個資料產品，不受本次圖面裁決影響。
其餘六種圖（`source_footprint_kde` 等）本來就不受這項裁決規範，維持原樣，
`source_footprint_kde` 仍會畫出 local domain 開放邊界線（見該函式）。

WP-25：對齊 OCM-SVD-Analysis 正式成圖規範（輸出格式與色階）
--------------------------------------------------------------
本模組另依 ``docs/results/17_native_figures_formal_output_rules.md``（機器可讀
契約：``docs/data/native_figures_formal_output_policy_v1.json``）新增兩類規則，
細節見該文件，這裡只列摘要：

1. **輸出格式**：每張圖同時輸出 ``.png``（``dpi`` 由 ``NativeFiguresConfig.dpi``
   指定，正式交付預設 600）與同名 ``.svg``（向量、不受 ``dpi`` 影響解析度）；
   兩者的 SHA-256 都寫入 ``manifest.json``。``--figure-text`` 與
   ``captions.md`` 的既有行為完全不變，兩種格式共用同一份標題／副標／圖例
   邏輯，不是分別繪製兩次。
2. **固定色階與跨站色條刻度一致化**：``source_footprint_kde``（密度）、
   ``boundary_arc_segments``（弧段相對來源權重）、``boundary_azimuth_rose``
   （方位扇區相對來源權重）、``settling_velocity_breakdown``（熱圖相對來源
   權重）四張圖，改用同一批被選取站點共同的色階上下限與刻度（見
   :func:`_compute_shared_color_scales`），不再各站各自依自己的資料最大值
   正規化色階。超過上限的值仍以最高色顯示，不做視覺裁切；每個「站×圖」組合
   的 ``over_limit_count``／``over_limit_fraction``／``display_saturation``
   寫入 ``manifest.json`` 的 ``color_scales`` 區塊。``horizontal_overview``／
   ``horizontal_local``／``depth_age`` 的色階是「代表粒子起始深度序位」
   （``rank_index``），不是可跨站比較的物理量或比例——不同站的代表粒子抽樣
   數（``rank_total``）本來就可能不同，因此不適用跨站固定色階，這三張圖的
   色階規則維持原樣（見規則文件「不適用條目」一節的說明）。

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
各邊界弧段的相對頻率——是描述這一批 ``M`` 個系集成員在既定設定下的分布，需要
先驗、概似與觀測驗證後才能延伸為來源機率或因果歸因，本模組不提供、也不宣稱
提供這一層推論。每張圖的副標與 ``captions.md`` 固定寫出站點、粒子數、``M``、
終止邊界定義（flow domain 開放邊界）等已知設定，供讀者自行核對適用範圍，不使用
警語式的「不得」句型。``M``（WP-37 起）一律讀 ``boundary_summary_by_site.
ensemble_member_count``（後處理實測的相異 ``member_id`` 數量），不是任何寫死的
常數——同一批圖表可能來自 ``M=1`` 的確定性試驗，也可能來自 ``M>1`` 的隨機系集，
由後處理實際資料決定，本模組不假設固定為 1。

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
from dataclasses import dataclass, replace
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
from .footprint import cell_area_km2_by_row
from .postprocess import DEFAULT_AZIMUTH_SECTOR_COUNT, NON_CROSSING_STATUS_CATEGORY_ZH

__all__ = [
    "FiguresError",
    "NativeFiguresConfig",
    "FIGURE_STEMS",
    "FOOTPRINT_FIGURE_STEMS",
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

ISOBATH_LEVELS_DEFAULT: tuple[float, ...] = (200.0, 1000.0)
"""WP-38 預設只畫兩條等深線：200 m（陸棚邊緣，摘要表的陸棚占比即以此為界）與 1000 m；
等深線密了會和高密度區間（HDR）互相干擾，所以其他水深預設不畫，需要時由參數指定。"""

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

    ``bathymetry_path``（WP-38）是 ``scripts/export_isobaths.py`` 匯出的等深線 GeoJSON
    （CRS84）；省略或檔案不存在時，停留時數密度等新圖仍照常繪製，只是不畫等深線，
    並在 ``manifest.json`` 的 ``footprint_figures`` 區塊記錄原因。``isobath_levels_m`` 是要畫的
    等深線水深（預設 200、1000 公尺，見 :data:`ISOBATH_LEVELS_DEFAULT`）。
    ``receptor_config_path`` 是正式釋放設定 YAML（讀 ``study_sites[].anchor_lonlat``）：星號受體
    位置以此為準。後處理目錄含足跡產品而沒有提供 ``receptor_config_path`` 時 fail-closed（正式出圖必須帶
    釋放設定）；只有明示 ``allow_unanchored_receptor=True``（診斷用）才退回後處理由釋放點反推的
    平均位置，並在圖例、圖說與 manifest 標明這不是站點錨點。
    """

    postprocess_dir: Path
    coastline_path: Path
    output_dir: Path
    sites: tuple[str, ...] | None = None
    font_path: Path | None = None
    max_trajectories_per_site: int = 24
    dpi: int = 150
    figure_text: str = "full"
    bathymetry_path: Path | None = None
    isobath_levels_m: tuple[float, ...] = ISOBATH_LEVELS_DEFAULT
    receptor_config_path: Path | None = None
    allow_unanchored_receptor: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "postprocess_dir", Path(self.postprocess_dir))
        object.__setattr__(self, "coastline_path", Path(self.coastline_path))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.font_path is not None:
            object.__setattr__(self, "font_path", Path(self.font_path))
        if self.bathymetry_path is not None:
            object.__setattr__(self, "bathymetry_path", Path(self.bathymetry_path))
        if self.receptor_config_path is not None:
            object.__setattr__(self, "receptor_config_path", Path(self.receptor_config_path))
        levels = tuple(sorted({float(v) for v in self.isobath_levels_m}))
        if any(not math.isfinite(v) or v <= 0 for v in levels):
            raise FiguresError("isobath_levels_m 必須是有限正數（水深，公尺）")
        object.__setattr__(self, "isobath_levels_m", levels)
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
            raise FiguresError(f"figure_text 必須是 {FIGURE_TEXT_MODES} 之一，收到：{self.figure_text!r}")


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
    footprint: Mapping[str, FootprintSiteFigureData] | None = None
    footprint_manifest: Mapping[str, Any] | None = None


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

# WP-37：這一站實測的系集成員數（M，相異 member_id 數量），由
# ``postprocess._write_site_summary_table`` 寫入。刻意**不**放進上面的
# ``_SUMMARY_REQUIRED_COLUMNS``（那個 tuple 缺欄位時只會得到 _require_columns
# 的通用訊息「缺少必要欄位」），而是在 load_native_figures_inputs 另外用專屬
# 訊息檢查，明確指出這是舊版後處理輸出、需要重跑 native-postprocess，避免使用
# 預設值 1 靜默帶過（那正是 WP-37 修的缺陷成因，見模組 docstring）。
_SUMMARY_ENSEMBLE_MEMBER_COUNT_COLUMN = "ensemble_member_count"

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
    """讀 ``boundary_summary_by_site`` 這一站的一列，並附上 WP-37 的
    ``ensemble_member_count``（呼叫端 :func:`load_native_figures_inputs` 已經
    確認這欄存在才會呼叫本函式，這裡不重複檢查）。"""

    mask = pc.equal(summary_table.column("study_site_id"), site_id)
    matched = summary_table.filter(mask)
    if matched.num_rows == 0:
        raise FiguresError(f"boundary_summary_by_site 沒有 {site_id} 站的列")
    if matched.num_rows > 1:
        raise FiguresError(f"boundary_summary_by_site 對 {site_id} 有超過一列，資料不一致")
    columns = (*_SUMMARY_REQUIRED_COLUMNS, _SUMMARY_ENSEMBLE_MEMBER_COUNT_COLUMN)
    return {name: matched.column(name)[0].as_py() for name in columns}


def _build_local_departures_by_row(local_table: pa.Table, site_id: str) -> dict[int, dict[str, Any]]:
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


def _build_arc_segments(arc_table: pa.Table, site_id: str) -> dict[str, tuple[dict[str, Any], ...]]:
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


def _load_receptor_anchors(path: Path) -> dict[str, tuple[float, float, float | None]]:
    """讀正式釋放設定 YAML 的 ``study_sites[].anchor_lonlat``（權威來源）。

    回傳 ``{站: (經度, 緯度, receptor_core_radius_m 或 None)}``。這是研究點的定義位置
    （見 ``config.StudySiteConfig``）；後處理由各受體釋放點平均反推的位置不是權威值，
    因為 20 個受體是在錨點周圍的核心半徑內隨機取點。任何缺漏或非有限值都 fail-closed。
    """

    import yaml

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise FiguresError(f"無法讀取釋放設定檔：{exc}") from exc
    sites = payload.get("study_sites") if isinstance(payload, dict) else None
    if not isinstance(sites, list) or not sites:
        raise FiguresError("釋放設定檔缺少 study_sites 清單")
    result: dict[str, tuple[float, float, float | None]] = {}
    for index, item in enumerate(sites):
        try:
            site_id = str(item["study_site_id"])
            lon, lat = (float(v) for v in item["anchor_lonlat"])
        except (KeyError, TypeError, ValueError) as exc:
            raise FiguresError(f"釋放設定檔 study_sites[{index}] 缺少 study_site_id／anchor_lonlat") from exc
        if not (math.isfinite(lon) and math.isfinite(lat)):
            raise FiguresError(f"釋放設定檔 study_sites[{index}] 的 anchor_lonlat 必須是有限數值")
        radius_raw = item.get("receptor_core_radius_m")
        result[site_id] = (lon, lat, float(radius_raw) if radius_raw is not None else None)
    return result


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
    trajectory_table, trajectory_path = _require_strict_parquet(postprocess_dir, "trajectory_samples_by_site")
    _require_columns(trajectory_table, _TRAJECTORY_REQUIRED_COLUMNS, "trajectory_samples_by_site.parquet")
    kde_table, kde_path = _require_strict_parquet(postprocess_dir, "kde_by_site")
    _require_columns(kde_table, _KDE_REQUIRED_COLUMNS, "kde_by_site.parquet")
    summary_table, summary_path = _read_table_any(postprocess_dir, "boundary_summary_by_site")
    _require_columns(summary_table, _SUMMARY_REQUIRED_COLUMNS, "boundary_summary_by_site")
    if _SUMMARY_ENSEMBLE_MEMBER_COUNT_COLUMN not in summary_table.column_names:
        # WP-37：這是舊版 native-postprocess 輸出（尚未把系集成員數寫進
        # boundary_summary_by_site），不是普通的缺欄位——fail-closed 拒絕出圖，
        # 不用預設值 1 靜默帶過，否則正是這次正式成果把 M 印錯的成因重演。
        raise FiguresError(
            f"boundary_summary_by_site 缺少 {_SUMMARY_ENSEMBLE_MEMBER_COUNT_COLUMN} 欄位——"
            "這是舊版 native-postprocess 輸出，尚未記錄系集成員數（M），請重新執行 "
            "native-postprocess 重新產生 boundary_summary_by_site 後再出圖，不可用預設值 1 "
            "靜默帶過"
        )
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

    footprint_data: dict[str, FootprintSiteFigureData] | None = None
    footprint_paths: dict[str, Path] = {}
    if _footprint_tables_present(postprocess_dir):
        footprint_data, footprint_paths = _load_footprint_figure_inputs(
            postprocess_dir, [site.site_id for site in sites]
        )

    unanchored = config.receptor_config_path is None and not config.allow_unanchored_receptor
    if footprint_data is not None and unanchored:
        raise FiguresError(
            "後處理目錄含足跡產品，但未提供釋放設定（--config）：星號受體必須取自正式設定的 "
            "study_sites[].anchor_lonlat；僅診斷用途可加 --allow-unanchored-receptor 退回釋放點平均位置"
        )
    if config.receptor_config_path is not None:
        anchors = _load_receptor_anchors(config.receptor_config_path)
        if footprint_data is not None:
            missing = [sid for sid in footprint_data if sid not in anchors]
            if missing:
                raise FiguresError(f"釋放設定檔沒有這些站的 anchor_lonlat：{missing}")
            footprint_data = {
                sid: replace(
                    fp,
                    receptor_anchor=(anchors[sid][0], anchors[sid][1]),
                    receptor_core_radius_m=anchors[sid][2],
                    receptor_source="release_config_anchor_lonlat",
                )
                for sid, fp in footprint_data.items()
            }
        footprint_paths = {**footprint_paths, "receptor_config": config.receptor_config_path}

    source_paths: dict[str, Path] = {
        **footprint_paths,
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
    if config.bathymetry_path is not None and config.bathymetry_path.is_file():
        source_paths["bathymetry"] = config.bathymetry_path
    sources_before = _snapshot_sources(source_paths)

    return NativeFiguresInputs(
        sites=tuple(sites),
        postprocess_manifest=manifest_payload,
        sources_before=sources_before,
        source_paths=source_paths,
        footprint=footprint_data,
        footprint_manifest=(
            manifest_payload.get("footprint") if isinstance(manifest_payload.get("footprint"), dict) else None
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════
# 繪圖層：以下函式全部只讀 ``SiteData``／字型／海岸線，不再碰任何檔案 I/O
# （檔案落地統一交給 :func:`build_native_figures` 排他建立）。
# ═══════════════════════════════════════════════════════════════════════════

_COMPASS_16_ORDER: tuple[str, ...] = (
    "N",
    "NNE",
    "NE",
    "ENE",
    "E",
    "ESE",
    "SE",
    "SSE",
    "S",
    "SSW",
    "SW",
    "WSW",
    "W",
    "WNW",
    "NW",
    "NNW",
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
        raise FiguresError(f"{site_id} 站 {field_name} 在已穿越邊界的列中出現缺值，資料契約不一致")
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
        departures["departure_azimuth_sector_index"],
        mask,
        field_name="departure_azimuth_sector_index",
        site_id=site_id,
    )
    counts: dict[int, int] = {}
    for value in sector_index:
        counts[int(value)] = counts.get(int(value), 0) + 1
    results = []
    for index in sorted(counts):
        if not (0 <= index < sector_count):
            raise FiguresError(f"{site_id} 站方位扇區索引 {index} 超出 sector_count={sector_count} 範圍")
        name = _COMPASS_16_ORDER[index] if sector_count == len(_COMPASS_16_ORDER) else f"sector_{index:03d}"
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
        departures["boundary_arc_segment_index"],
        mask,
        field_name="boundary_arc_segment_index",
        site_id=site_id,
    )
    counts: dict[tuple[str, int], int] = {}
    for base_id, index in zip(base_ids, arc_index, strict=True):
        if base_id is None:
            raise FiguresError(f"{site_id} 站已穿越邊界的列缺少 boundary_segment_id")
        key = (str(base_id), int(index))
        counts[key] = counts.get(key, 0) + 1
    results = []
    for base_id, index in sorted(counts):
        count = counts[(base_id, index)]
        results.append(ArcSegmentWeight(base_id, index, count, float(count) / float(total)))
    return tuple(results)


def _status_counts(departures: Mapping[str, np.ndarray]) -> dict[int, int]:
    codes = np.asarray(departures["final_status_code"], dtype=np.int64)
    counts: dict[int, int] = {}
    for code in codes:
        counts[int(code)] = counts.get(int(code), 0) + 1
    return counts


# ═══════════════════════════════════════════════════════════════════════════
# WP-25：跨站共用色階（對齊 OCM-SVD-Analysis 的「四海域共同尺度」規則，見
# docs/results/17_native_figures_formal_output_rules.md 與模組 docstring）
# ═══════════════════════════════════════════════════════════════════════════

_SHARED_COLOR_SCALE_STEMS: tuple[str, ...] = (
    "source_footprint_kde",
    "boundary_arc_segments",
    "boundary_azimuth_rose",
    "settling_velocity_breakdown",
)
"""這四張圖的色階／色條在同一批被選取站點之間共用同一組上下限與刻度，讓讀者
可以直接比較不同站點的顏色深淺（見 :func:`_compute_shared_color_scales`）。

其餘四張圖（``horizontal_overview``／``horizontal_local``／``depth_age``／
``terminal_counts``）刻意不在這裡：前三張的顏色是「代表粒子起始深度序位」
（``rank_index``／``rank_total``），不同站的代表粒子抽樣數本來就可能不同，
序位不是可跨站比較的物理量或比例；``terminal_counts`` 用的是固定的狀態別
類別色（``_STATUS_COLORS``），本來就與資料範圍無關、天生就跨站一致，不需要
額外的「共用色階」機制。"""


@dataclass(frozen=True, slots=True)
class ColorScale:
    """一張圖在跨站比較情境下共用的色階上下限與刻度。

    ``vmin``／``vmax`` 是這批被選取站點共同套用的色階範圍；``tick_count``
    個刻度在 ``[vmin, vmax]`` 之間等距分布（含頭尾兩端），供色條標示。超過
    ``vmax`` 的值不會被裁掉或改寫，只是顯示時飽和成色階最頂端的顏色（呼叫端
    對 ``matplotlib`` 的 ``Normalize``／``imshow``／``pcolormesh`` 傳入同一組
    ``vmin``／``vmax`` 即可得到這個行為，這是 ``matplotlib`` 內建語意，本類別
    不需要自己做視覺裁切）；實際超限狀況另由 :meth:`over_limit` 算出並寫進
    manifest，供人工稽核「這批資料有沒有真的超過目前的色階上限」。
    """

    vmin: float
    vmax: float
    tick_count: int = 7

    def __post_init__(self) -> None:
        if self.tick_count < 2:
            raise FiguresError("ColorScale.tick_count 必須至少為 2")
        if not (math.isfinite(self.vmin) and math.isfinite(self.vmax)) or self.vmax <= self.vmin:
            raise FiguresError(
                f"ColorScale 上下限必須是有限數值且 vmax > vmin，收到 vmin={self.vmin}, vmax={self.vmax}"
            )

    def ticks(self) -> tuple[float, ...]:
        step = (self.vmax - self.vmin) / (self.tick_count - 1)
        return tuple(self.vmin + step * i for i in range(self.tick_count))

    def over_limit(self, values: np.ndarray) -> tuple[int, float, bool]:
        """回傳 ``(over_limit_count, over_limit_fraction, display_saturation)``。

        分母是 ``values`` 中的有限值筆數（``NaN``／``inf`` 不計入，也不算超限，
        因為那些本來就不是可比較的數值）；``values`` 全空或全非有限值時回傳
        ``(0, 0.0, False)``，代表這一站這張圖沒有可統計的資料，不是「沒有超限」
        的積極宣稱。
        """

        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return 0, 0.0, False
        count = int(np.sum(finite > self.vmax))
        fraction = count / finite.size
        return count, fraction, count > 0


def _color_scale_source_values(stem: str, site: SiteData) -> np.ndarray | None:
    """回傳某張圖在某一站實際餵進共用色階的原始數值陣列。

    :func:`_compute_shared_color_scales`（跨站 preflight，決定共同 vmax）與
    manifest 的 over-limit 統計（單站紀錄）都呼叫這個函式，確保兩處用的是
    同一份「這張圖的色階量到底是什麼」定義，不會算出兩種不同答案。回傳
    ``None`` 表示這一站這張圖沒有色階資料可比（目前只有 KDE 狀態不是
    ``"available"`` 時會發生），呼叫端不應把這解讀成「有資料但剛好是 0 筆」。
    """

    if stem == "source_footprint_kde":
        if site.kde is None or site.kde.get("status") != "available":
            return None
        return np.asarray(site.kde["density_per_m2_flat"], dtype=np.float64)
    if stem == "boundary_arc_segments":
        weights = _compute_arc_segment_weights(site.departures, site_id=site.site_id)
        return np.asarray([w.relative_source_weight for w in weights], dtype=np.float64)
    if stem == "boundary_azimuth_rose":
        weights = _compute_sector_weights(
            site.departures, sector_count=site.azimuth_sector_count, site_id=site.site_id
        )
        return np.asarray([w.relative_source_weight for w in weights], dtype=np.float64)
    if stem == "settling_velocity_breakdown":
        sv_array = np.asarray(site.departures["settling_velocity_mps"], dtype=np.float64)
        collected: list[float] = []
        for sv in sorted(set(sv_array.tolist())):
            sv_mask = np.isclose(sv_array, sv)
            weights = _compute_sector_weights(
                site.departures,
                sector_count=site.azimuth_sector_count,
                site_id=site.site_id,
                extra_mask=sv_mask,
            )
            collected.extend(w.relative_source_weight for w in weights)
        return np.asarray(collected, dtype=np.float64)
    raise FiguresError(f"{stem} 不是支援共用色階的圖種")  # pragma: no cover - 呼叫端已限制範圍


_COLOR_SCALE_DEFAULT_VMAX = 1.0
"""某張圖在這批站點完全沒有任何可用數值時（例如全部站點都沒有粒子成功穿越
邊界）的保底 ``vmax``。``relative_source_weight`` 與 KDE 密度都是非負量，用
``1.0`` 只是避免 ``ColorScale`` 因 ``vmax<=vmin`` 而拒絕建構，不代表任何量測
到的數值；這種退化情況下色條會整片顯示同一色階最低色，如實反映「這批圖沒有
任何非零訊號」，不是刻意誤導。"""


def _compute_shared_color_scales(sites: tuple[SiteData, ...]) -> dict[str, ColorScale]:
    """對這批被選取的站點做一次跨站 preflight，決定
    :data:`_SHARED_COLOR_SCALE_STEMS` 四張圖各自的共同色階。

    ``vmin`` 固定為 ``0.0``（KDE 密度與相對來源權重都是非負量，見
    ``postprocess`` 與模組 docstring 的定義）；``vmax`` 取這批站點『這次執行』
    實際觀測到的跨站最大值。

    **與 OCM-SVD-Analysis 的差異，誠實揭露**：那邊的 ``vmax=0.024`` 是先用
    真實四海域 science run 資料做過一次 preflight、經過審查後寫死在規則文件
    裡的常數，之後每次重繪都固定使用同一個數字，因此新資料若超過它才會真的
    出現非零的 over-limit 紀錄。LBT 目前是第一次建立這套機制，沒有這樣一個
    經過審查、可以直接套用到未來所有 run 的歷史常數；本函式因此把 vmax 定義
    為「這次執行、這批站點的實際最大值」，這樣做在本次執行內一定不會有
    over-limit（因為上限本來就取自這批資料本身），但已經達成規則要求的
    「同一批圖之間色階刻度一致化、超限只記在 metadata、不做視覺裁切」的
    核心行為；``over_limit_count`` 等欄位在架構上完整保留，未來若研究團隊
    審閱後要另外核定一個跨 run 永遠固定的常數，只需要在呼叫端改用那個常數
    建構 :class:`ColorScale` 即可，不必更動這裡的統計或繪圖邏輯。
    """

    scales: dict[str, ColorScale] = {}
    for stem in _SHARED_COLOR_SCALE_STEMS:
        collected: list[np.ndarray] = []
        for site in sites:
            values = _color_scale_source_values(stem, site)
            if values is not None and values.size:
                finite = values[np.isfinite(values)]
                if finite.size:
                    collected.append(finite)
        pooled = np.concatenate(collected) if collected else np.array([], dtype=np.float64)
        vmax = float(pooled.max()) if pooled.size else _COLOR_SCALE_DEFAULT_VMAX
        if not math.isfinite(vmax) or vmax <= 0.0:
            vmax = _COLOR_SCALE_DEFAULT_VMAX
        scales[stem] = ColorScale(vmin=0.0, vmax=vmax)
    return scales


def _build_color_scale_manifest(
    sites: tuple[SiteData, ...], color_scales: Mapping[str, ColorScale]
) -> dict[str, Any]:
    """把 :func:`_compute_shared_color_scales` 的結果與每站實際超限狀況整理成
    manifest 的 ``color_scales`` 區塊（見規則文件「固定色階與超限處理」一節）。"""

    result: dict[str, Any] = {}
    for stem in _SHARED_COLOR_SCALE_STEMS:
        scale = color_scales[stem]
        per_site: dict[str, Any] = {}
        for site in sites:
            values = _color_scale_source_values(stem, site)
            if values is None:
                per_site[site.site_id] = {"status": "not_applicable"}
                continue
            count, fraction, saturated = scale.over_limit(values)
            per_site[site.site_id] = {
                "status": "available",
                "sample_count": int(values.size),
                "over_limit_count": count,
                "over_limit_fraction": fraction,
                "display_saturation": saturated,
            }
        result[stem] = {
            "vmin": scale.vmin,
            "vmax": scale.vmax,
            "vmax_basis": "max_of_finite_values_across_selected_sites_this_run",
            "ticks": list(scale.ticks()),
            "colorbar_extend": "neither",
            "per_site": per_site,
        }
    return result


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

# WP-25：local domain 首次離開標記樣式（``_LOCAL_FIRST_EXIT_MARKER_STYLE``）與
# 對應的副標說明函式（``_local_flow_semantics_caption``）已隨「水平軌跡圖移除
# local domain 圖層」裁決一併移除（見模組 docstring「WP-25：水平軌跡圖移除
# local domain 圖層」一節）；資料層（``local_first_exit_lon_deg`` 等欄位）仍在
# :class:`TrajectorySample`／:class:`SiteData` 保留，只是不再有圖面樣式常數。


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


def _add_two_line_title(fig, t: _Labeler, font: Any, title: str, subtitle: str, *, figure_text: str) -> None:
    """畫兩行標題（主標＋副標）；``figure_text=="none"`` 時什麼都不畫（見
    :data:`FIGURE_TEXT_MODES`），供 WP-24b 正式報告排版模式使用——文字改由
    ``captions.md`` 承載，圖面本身只留座標軸與圖例。"""

    if figure_text == "none":
        return
    fig.suptitle(title, fontproperties=font, fontsize=17, y=0.985)
    subtitle_y = 0.945 if "\n" not in subtitle else 0.915
    fig.text(0.5, subtitle_y, subtitle, ha="center", va="top", fontproperties=font, fontsize=10)


def _title_top_margin(figure_text: str, *, full: float, none: float = 0.97) -> float:
    """依文字層級回傳 ``fig.subplots_adjust(top=...)`` 用的頂部邊界。

    ``full`` 模式沿用各圖原本的邊界（留給標題／副標的畫面空間）；``none`` 模式
    不畫標題，因此把繪圖區延伸到更接近畫面頂端，避免留下沒有文字的大片空白。
    """

    return full if figure_text == "full" else none


def _ensemble_member_count_text(n: int) -> tuple[str, str]:
    """回傳系集成員數（``M``）的中／英文描述短句，供圖面副標與 ``captions.md``
    共用同一份措辭（WP-37：兩處過去各自寫死 ``M=1``，這裡改成唯一來源）。

    ``n`` 是後處理實測的相異 ``member_id`` 數量（見
    ``postprocess._ensemble_member_count_by_site``），不是任何寫死的常數；
    ``n == 1`` 時維持既有語句（正式成果曾經真的是確定性單一系集試驗時用這句），
    其餘一律寫「每情境 n 個獨立隨機系集成員」。
    """

    if n == 1:
        return (
            "M=1（每情境一個系集成員，含隨機擴散）",
            "M=1 (single ensemble member)",
        )
    return (
        f"M={n}（每情境 {n} 個獨立隨機系集成員，含隨機擴散）",
        f"M={n} ({n} ensemble members per scenario)",
    )


def _site_subtitle(site: SiteData, t: _Labeler, *, extra: str = "") -> str:
    total = int(site.summary["total_particle_count"])
    crossed_fraction = float(site.summary["fraction_crossed_interest_region_boundary"])
    member_count = int(site.summary["ensemble_member_count"])
    member_text_zh, member_text_en = _ensemble_member_count_text(member_count)
    base = t.t(
        f"站點：{site.site_label_zh}（{site.site_id}）｜粒子數：{total}｜穿越開放邊界比例："
        f"{crossed_fraction * 100:.1f}%｜{member_text_zh}",
        f"Site: {site.site_id} | N={total} | crossed fraction: "
        f"{crossed_fraction * 100:.1f}% | {member_text_en}",
    )
    return f"{base}{extra}"


def _save_exclusive(fig, path: Path, *, dpi: int) -> None:
    """以排他建立方式落檔（``xb``），輸出目錄若已存在同名檔一律 fail-closed。

    WP-25：對齊 OCM-SVD-Analysis 正式成圖規範「PNG 600 dpi ＋ SVG 都出」的
    輸出格式決定，``path``（``.png``）與其同名 ``.svg``（向量，不受 ``dpi``
    影響解析度）在這裡一次寫出，共用同一個已經畫好的 ``Figure`` 物件，不是
    分別呼叫繪圖函式兩次；呼叫端（:func:`build_native_figures`）另外把兩者的
    SHA-256 都記進 manifest。任何一步失敗都不會留下另一個格式的半完成檔——
    PNG 排他建立失敗時整個函式直接拋例外，不會嘗試寫 SVG。
    """

    with path.open("xb") as stream:
        fig.savefig(stream, format="png", dpi=dpi, bbox_inches="tight")
    svg_path = path.with_suffix(".svg")
    with svg_path.open("xb") as stream:
        fig.savefig(stream, format="svg", dpi=dpi, bbox_inches="tight")


def _land_lonlat_patches(land_geometries: Sequence[Polygon | MultiPolygon]):
    from shapely.plotting import plot_polygon

    def _draw(ax) -> None:
        for land in land_geometries:
            plot_polygon(
                land,
                ax=ax,
                add_points=False,
                facecolor="#c9c9c9",
                edgecolor="#858585",
                linewidth=0.25,
                alpha=0.85,
                zorder=1,
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
        if kind == "local":
            # WP-25：水平總覽圖不再畫 local domain 邊界（見模組 docstring
            # 「WP-25：水平軌跡圖移除 local domain 圖層」一節）；資料仍完整保存
            # 在 local_domain_departures.parquet 等表，只是不進這張圖。
            continue
        boundary_kinds_present.add(kind)
        style = _BOUNDARY_LINE_STYLE_BY_KIND[kind]
        ax.plot(lon, lat, zorder=2, **style)

    cmap = plt.get_cmap("viridis")
    for sample in site.trajectories:
        color = cmap(0.0 if sample.rank_total <= 1 else (sample.rank_index - 1) / (sample.rank_total - 1))
        ax.plot(sample.lon_deg, sample.lat_deg, color=color, linewidth=1.0, alpha=0.85, zorder=4)
        ax.plot(
            sample.lon_deg[0],
            sample.lat_deg[0],
            marker="o",
            markersize=5.2,
            markerfacecolor="#1a9850",
            markeredgecolor="white",
            markeredgewidth=0.6,
            linestyle="None",
            zorder=6,
        )
        # WP-25：不再畫 local domain 首次離開標記（資料仍在 sample.local_first_
        # exit_lon_deg／lat_deg，只是這張圖不使用），圖上只保留下面的終止標記。
        marker = _STATUS_MARKERS.get(sample.final_status_code, "X")
        color_end = _STATUS_COLORS.get(sample.final_status_code, "#d62728")
        ax.plot(
            sample.lon_deg[-1],
            sample.lat_deg[-1],
            marker=marker,
            markersize=7.0,
            markerfacecolor=color_end,
            markeredgecolor=color_end,
            linestyle="None",
            zorder=7,
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
        Line2D(
            [],
            [],
            marker="o",
            color="#1a9850",
            markerfacecolor="#1a9850",
            linestyle="None",
            label=t.t("回溯起點（受體到達位置）", "Backtrack start (receptor arrival)"),
        ),
    ]
    if "flow" in boundary_kinds_present:
        handles.append(
            Line2D(
                [],
                [],
                **_BOUNDARY_LINE_STYLE_BY_KIND["flow"],
                label=t.t("流場範圍開放邊界（終止邊界）", "Flow-domain open boundary (terminal)"),
            )
        )
    if "other" in boundary_kinds_present:
        handles.append(
            Line2D(
                [],
                [],
                **_BOUNDARY_LINE_STYLE_BY_KIND["other"],
                label=t.t("未分類開放邊界", "Unclassified open boundary"),
            )
        )
    for code in _STATUS_ORDER:
        handles.append(
            Line2D(
                [],
                [],
                marker=_STATUS_MARKERS.get(code, "X"),
                color=_STATUS_COLORS.get(code, "#d62728"),
                linestyle="None",
                markersize=7,
                label=t.t(_STATUS_LABELS_ZH.get(code, str(code)), _STATUS_LABELS_EN.get(code, str(code))),
            )
        )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=4,
        prop=font,
        frameon=True,
        fontsize=8,
    )

    _add_two_line_title(
        fig,
        t,
        font,
        t.t(f"水平軌跡總覽｜{site.site_label_zh}", f"Horizontal overview | {site.site_id}"),
        _site_subtitle(site, t, extra=t.t("｜線色：代表粒子起始深度序位（淺→深）", "")),
        figure_text=figure_text,
    )
    # WP-25：不再畫 local／flow 差異說明副標（圖上已經只剩 flow domain 邊界，
    # 不需要再解釋兩者差異）。
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
        if kind == "local":
            # WP-25：局部放大圖同樣不再畫 local domain 邊界（見水平總覽圖同一
            # 段說明）；資料仍完整保存，只是不進這張圖。
            continue
        boundary_kinds_present.add(kind)
        style = _BOUNDARY_LINE_STYLE_BY_KIND[kind]
        # flow domain 邊界通常遠在局部圖視窗之外（局部圖只依代表粒子範圍縮放，
        # 不強制含入邊界，見下方 span 計算只用軌跡點），畫出來只是讓視窗剛好
        # 涵蓋到時能看見；不在範圍內時 matplotlib 自動裁掉，不影響軸範圍。
        ax.plot(lon - x0, lat - y0, zorder=2, **style)
    cmap = plt.get_cmap("viridis")
    all_lon: list[float] = []
    all_lat: list[float] = []
    for sample in site.trajectories:
        color = cmap(0.0 if sample.rank_total <= 1 else (sample.rank_index - 1) / (sample.rank_total - 1))
        ax.plot(sample.lon_deg - x0, sample.lat_deg - y0, color=color, linewidth=1.3, alpha=0.85, zorder=4)
        ax.plot(
            sample.lon_deg[0] - x0,
            sample.lat_deg[0] - y0,
            marker="o",
            markersize=6.2,
            markerfacecolor="#1a9850",
            markeredgecolor="white",
            markeredgewidth=0.7,
            linestyle="None",
            zorder=6,
        )
        # WP-25：不再畫 local domain 首次離開標記（理由同水平總覽圖）。
        marker = _STATUS_MARKERS.get(sample.final_status_code, "X")
        color_end = _STATUS_COLORS.get(sample.final_status_code, "#d62728")
        ax.plot(
            sample.lon_deg[-1] - x0,
            sample.lat_deg[-1] - y0,
            marker=marker,
            markersize=8.2,
            markerfacecolor=color_end,
            markeredgecolor=color_end,
            linestyle="None",
            zorder=7,
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
    ax.set_xlabel(
        t.t("經度偏移（度，相對代表粒子起點平均）", "Longitude offset (deg, rel. mean start)"),
        fontproperties=font,
    )
    ax.set_ylabel(
        t.t("緯度偏移（度，相對代表粒子起點平均）", "Latitude offset (deg, rel. mean start)"),
        fontproperties=font,
    )
    ax.grid(color="white", linewidth=0.8)

    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            color="#1a9850",
            markerfacecolor="#1a9850",
            linestyle="None",
            label=t.t("回溯起點（受體到達位置）", "Backtrack start (receptor arrival)"),
        ),
    ]
    if "flow" in boundary_kinds_present:
        handles.append(
            Line2D(
                [],
                [],
                **_BOUNDARY_LINE_STYLE_BY_KIND["flow"],
                label=t.t("流場範圍開放邊界（終止邊界）", "Flow-domain open boundary (terminal)"),
            )
        )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=2,
        prop=font,
        frameon=True,
        fontsize=8,
    )

    _add_two_line_title(
        fig,
        t,
        font,
        t.t(f"水平局部放大｜{site.site_label_zh}", f"Horizontal local view | {site.site_id}"),
        _site_subtitle(site, t),
        figure_text=figure_text,
    )
    # WP-25：不再畫 local／flow 差異說明副標（理由同水平總覽圖）。
    fig.subplots_adjust(left=0.05, bottom=0.24, right=0.98, top=_title_top_margin(figure_text, full=0.84))
    _save_exclusive(fig, path, dpi=dpi)
    plt.close(fig)


def _plot_depth_age(site: SiteData, t: _Labeler, font: Any, path: Path, dpi: int, figure_text: str) -> None:
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
        fig,
        t,
        font,
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
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{value}\n({fraction * 100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=8,
            fontproperties=font,
        )
    ax.set_xticks(range(len(_STATUS_ORDER)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontproperties=font, fontsize=8)
    ax.set_ylabel(t.t("佔全站粒子數比例", "Fraction of all particles at this site"), fontproperties=font)
    ax.set_ylim(0, max(fractions + [0.05]) * 1.25)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)

    _add_two_line_title(
        fig,
        t,
        font,
        t.t(f"停止類別統計｜{site.site_label_zh}", f"Terminal status counts | {site.site_id}"),
        _site_subtitle(site, t),
        figure_text=figure_text,
    )
    fig.subplots_adjust(left=0.08, bottom=0.34, right=0.98, top=_title_top_margin(figure_text, full=0.90))
    _save_exclusive(fig, path, dpi=dpi)
    plt.close(fig)


def _plot_boundary_azimuth_rose(
    site: SiteData,
    t: _Labeler,
    font: Any,
    path: Path,
    dpi: int,
    figure_text: str,
    color_scale: ColorScale,
) -> None:
    """16 方位扇區相對來源權重極座標玫瑰圖（正北向上、順時針）。

    ``color_scale``（WP-25）同時固定極座標半徑軸上限與長條顏色映射的
    ``vmin``／``vmax``，讓五站的玫瑰圖用同一把尺——半徑本身已經是可直接比較
    的相對來源權重，顏色只是輔助辨識，見 :data:`_SHARED_COLOR_SCALE_STEMS`。
    """

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
    norm = plt.Normalize(color_scale.vmin, color_scale.vmax)
    ax.bar(
        theta,
        radii,
        width=width,
        color=[cmap(norm(r)) for r in radii],
        edgecolor="white",
        linewidth=0.6,
    )
    ax.set_ylim(color_scale.vmin, color_scale.vmax)
    ax.set_xticks(np.radians([i * 360.0 / 16 for i in range(16)]))
    ax.set_xticklabels(
        _COMPASS_16_ORDER if n == 16 else [f"{i}" for i in range(n)], fontproperties=font, fontsize=8
    )
    ax.set_ylabel("")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.09, shrink=0.75)
    cbar.set_ticks(color_scale.ticks())
    cbar.set_label(
        t.t("相對來源權重（跨站共用色階）", "Relative source weight (shared cross-site scale)"),
        fontproperties=font,
    )

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
            fontproperties=font,
            fontsize=16,
            y=0.985,
        )
        fig.text(
            0.5,
            0.945,
            t.t(f"前三大方位：{top_text}", f"Top-3 sectors: {top_text}"),
            ha="center",
            va="top",
            fontproperties=font,
            fontsize=10,
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
    site: SiteData,
    t: _Labeler,
    font: Any,
    path: Path,
    dpi: int,
    figure_text: str,
    color_scale: ColorScale,
) -> None:
    """開放邊界折線依弧段著色（權重色階）＋弧段權重長條圖。

    ``color_scale``（WP-25）固定色階上下限與刻度，讓五站可直接比較顏色深淺；
    見 :data:`_SHARED_COLOR_SCALE_STEMS`。
    """

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    weights = _compute_arc_segment_weights(site.departures, site_id=site.site_id)
    by_key = {(w.base_boundary_segment_id, w.arc_segment_index): w for w in weights}
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(color_scale.vmin, color_scale.vmax)

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
            color = cmap(norm(value))
            ax_map.plot(
                [row["start_lon_deg"], row["end_lon_deg"]],
                [row["start_lat_deg"], row["end_lat_deg"]],
                color=color,
                linewidth=3.5,
                solid_capstyle="butt",
                zorder=3,
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
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax_map, fraction=0.05, pad=0.02)
    cbar.set_ticks(color_scale.ticks())
    cbar.set_label(
        t.t("相對來源權重（跨站共用色階）", "Relative source weight (shared cross-site scale)"),
        fontproperties=font,
    )

    order = sorted(range(len(all_labels)), key=lambda i: -all_values[i])
    ax_bar.bar(
        range(len(order)),
        [all_values[i] for i in order],
        color=[cmap(norm(all_values[i])) for i in order],
    )
    ax_bar.set_xticks(range(len(order)))
    ax_bar.set_xticklabels([all_labels[i] for i in order], rotation=90, fontsize=6)
    ax_bar.set_ylim(color_scale.vmin, color_scale.vmax)
    ax_bar.set_ylabel(
        t.t("相對來源權重（跨站共用色階）", "Relative source weight (shared cross-site scale)"),
        fontproperties=font,
    )
    ax_bar.grid(axis="y", color="#dddddd", linewidth=0.6)

    _add_two_line_title(
        fig,
        t,
        font,
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
    color_scale: ColorScale,
) -> None:
    """由 ``kde_by_site`` 畫條件式來源足跡：密度色階＋HDR 等值線，疊海岸線與開放邊界。

    ``color_scale``（WP-25）固定密度色階的 ``vmin``／``vmax``，讓五站可直接
    比較密度深淺；超過 ``vmax`` 的格點仍以最高色顯示，不裁切資料（見
    :data:`_SHARED_COLOR_SCALE_STEMS`）。
    """

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8.5), dpi=150)
    ax.set_facecolor("#eaf3f8")

    if site.kde is None or site.kde["status"] != "available":
        status_text = "" if site.kde is None else str(site.kde["status"])
        ax.text(
            0.5,
            0.5,
            t.t(
                f"此站無可用條件式來源足跡（狀態：{status_text or '缺席'}）",
                f"No source footprint available (status: {status_text or 'missing'})",
            ),
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontproperties=font,
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

        mesh = ax.pcolormesh(
            lon_grid,
            lat_grid,
            density,
            cmap="viridis",
            shading="auto",
            zorder=2,
            alpha=0.9,
            vmin=color_scale.vmin,
            vmax=color_scale.vmax,
        )
        cbar = fig.colorbar(mesh, ax=ax, fraction=0.045, pad=0.02)
        cbar.set_ticks(color_scale.ticks())
        cbar.set_label(
            t.t(
                "條件式來源足跡密度（每平方公尺，跨站共用色階）",
                "Conditional source footprint density (per m^2, shared cross-site scale)",
            ),
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
                lon_grid,
                lat_grid,
                mask.astype(np.float64),
                levels=[0.5],
                colors=[hdr_colors[level_index % len(hdr_colors)]],
                linewidths=1.6,
                zorder=5,
            )

        from matplotlib.lines import Line2D

        hdr_handles = [
            Line2D(
                [],
                [],
                color=hdr_colors[i % len(hdr_colors)],
                linewidth=1.6,
                label=t.t(f"高密度區間 {level * 100:.0f}%", f"HDR {level * 100:.0f}%"),
            )
            for i, level in enumerate(hdr_levels)
        ]
        fig.legend(
            handles=hdr_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.05),
            ncol=len(hdr_handles),
            prop=font,
            frameon=True,
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
        fig,
        t,
        font,
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
    site: SiteData,
    t: _Labeler,
    font: Any,
    path: Path,
    dpi: int,
    figure_text: str,
    color_scale: ColorScale,
) -> None:
    """十種沉降速度的穿越比例（長條圖）與主要方位扇區權重比較（熱圖）。

    ``color_scale``（WP-25）固定熱圖色階上下限與刻度，讓五站可直接比較顏色
    深淺（見 :data:`_SHARED_COLOR_SCALE_STEMS`）。
    """

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
            site.departures,
            sector_count=site.azimuth_sector_count,
            site_id=site.site_id,
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

    sector_labels = (
        _COMPASS_16_ORDER
        if site.azimuth_sector_count == 16
        else [f"{i}" for i in range(site.azimuth_sector_count)]
    )
    # origin="lower"讓熱圖第 0 列（sv_values 最小、下沉最快）畫在圖底部，
    # 與左側 barh 預設「索引 0 在最下面」的列順序一致，兩張子圖才能逐列對照。
    im = ax_heat.imshow(
        heatmap,
        aspect="auto",
        cmap="viridis",
        origin="lower",
        vmin=color_scale.vmin,
        vmax=color_scale.vmax,
    )
    ax_heat.set_xticks(range(len(sector_labels)))
    ax_heat.set_xticklabels(sector_labels, fontsize=7, rotation=90)
    ax_heat.set_yticks(range(len(sv_values)))
    ax_heat.set_yticklabels(sv_labels, fontsize=8)
    ax_heat.set_xlabel(t.t("方位扇區", "Azimuth sector"), fontproperties=font)
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.045, pad=0.02)
    cbar.set_ticks(color_scale.ticks())
    cbar.set_label(
        t.t(
            "相對來源權重（同沉降速度分組內，跨站共用色階）",
            "Relative source weight (within same settling velocity, shared cross-site scale)",
        ),
        fontproperties=font,
    )

    _add_two_line_title(
        fig,
        t,
        font,
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
    color_scales: Mapping[str, ColorScale],
) -> None:
    """依 ``stem`` 呼叫對應的繪圖函式；``color_scales`` 是
    :func:`_compute_shared_color_scales` 算好的跨站共用色階，只有
    :data:`_SHARED_COLOR_SCALE_STEMS` 內的四種圖會用到，其餘圖種忽略此參數
    （見各圖種色階規則的說明）。
    """

    if stem == "horizontal_overview":
        _plot_horizontal_overview(site, land_geometries, t, font, path, dpi, figure_text)
    elif stem == "horizontal_local":
        _plot_horizontal_local(site, t, font, path, dpi, figure_text)
    elif stem == "depth_age":
        _plot_depth_age(site, t, font, path, dpi, figure_text)
    elif stem == "terminal_counts":
        _plot_terminal_counts(site, t, font, path, dpi, figure_text)
    elif stem == "boundary_azimuth_rose":
        _plot_boundary_azimuth_rose(site, t, font, path, dpi, figure_text, color_scales[stem])
    elif stem == "boundary_arc_segments":
        _plot_boundary_arc_segments(site, t, font, path, dpi, figure_text, color_scales[stem])
    elif stem == "source_footprint_kde":
        _plot_source_footprint_kde(site, land_geometries, t, font, path, dpi, figure_text, color_scales[stem])
    elif stem == "settling_velocity_breakdown":
        _plot_settling_velocity_breakdown(site, t, font, path, dpi, figure_text, color_scales[stem])
    else:  # pragma: no cover - FIGURE_STEMS 是唯一呼叫來源，這裡只是防禦
        raise FiguresError(f"未知圖表代號：{stem}")


# ═══════════════════════════════════════════════════════════════════════════
# WP-38：停留時數密度、回溯快照與邊界弧長分布三種新圖
# ═══════════════════════════════════════════════════════════════════════════
#
# 這一節的圖只讀 ``postprocess`` 在 ``--with-footprint`` 下多輸出的表（見
# ``native_runner.footprint``），後處理目錄沒有這些表時完全不出這三種圖，八張既有圖
# 的行為不變。舊的 ``source_footprint_kde`` 仍照常輸出，但不再是報告主圖：它的輸入
# 是粒子穿越開放邊界的那一點，這些點全部落在一維的邊界線上，二維核密度不適用。

FOOTPRINT_FIGURE_STEMS: tuple[str, ...] = (
    "source_footprint_residence",
    "source_footprint_residence_site_scale",
    "lookback_snapshot",
    "boundary_departure_along_arc",
)
"""WP-38 新增的圖種檔名（不含副檔名）。``source_footprint_residence`` 使用跨站共用色階，
``source_footprint_residence_site_scale`` 使用各站自己的色階，兩者資料相同。"""

_FOOTPRINT_TABLE_STEMS: tuple[str, ...] = (
    "residence_density_by_site",
    "residence_hdr_by_site",
    "footprint_grid_by_site",
    "lookback_snapshot_by_site",
    "lookback_snapshot_density_by_site",
    "boundary_departure_along_arc_by_site",
    "boundary_edge_intervals_by_site",
    "footprint_geometry_by_site",
    "footprint_summary_by_site",
)

RESIDENCE_COLOR_FLOOR_FRACTION = 1.0e-4
"""對數色階的下限占上限的比例；低於下限的網格不著色（透明）。"""

_RESIDENCE_CMAP = "YlGnBu"

# 高密度區間（HDR）50%／75%／90%：同一色系（棕色）由深到淺，線型為實線／虛線／點線。
_HDR_LINE_STYLES: tuple[Mapping[str, Any], ...] = (
    {"color": "#3b2314", "linestyle": "-", "linewidth": 1.5},
    {"color": "#7a5230", "linestyle": "--", "linewidth": 1.3},
    {"color": "#a77c4e", "linestyle": ":", "linewidth": 1.5},
)

_OPEN_BOUNDARY_STYLE: Mapping[str, Any] = {"color": "#d0009f", "linewidth": 1.2, "linestyle": "--"}
_OPEN_BOUNDARY_LABEL_ZH = "流場範圍開放邊界（終止邊界）"
_OPEN_BOUNDARY_LABEL_EN = "Flow-domain open boundary (terminating boundary)"
_LAND_COLOR = "#d9d9d9"
_ISOBATH_COLOR = "#7f7f7f"
_ISOBATH_LINEWIDTH = 0.4
# 等深線線型依水深由淺到深：實線、虛線、點虛線、點線（預設 200 m 實線、1000 m 虛線）；
# 顏色是中灰、線寬 0.4 pt，和 HDR 的深棕色粗線明顯區隔，HDR 才是圖上的主角。
_ISOBATH_LINESTYLES: tuple[str, ...] = ("-", "--", "-.", ":")
_ISOBATH_STYLE_ZH: tuple[str, ...] = ("實線", "虛線", "點虛線", "點線")
_ISOBATH_STYLE_EN: tuple[str, ...] = ("solid", "dashed", "dash-dot", "dotted")

# 座標軸範圍向外擴到刻度間距的整數倍；間距依範圍大小在下列值中挑最小且刻度間隔數不超過上限者。
# 每軸刻度數 4～7 個（間隔數 3～6）；2×2 快照的小面板上限放寬為最多 6 個刻度（間隔數 5）。
_TICK_STEP_CHOICES_DEG: tuple[float, ...] = (0.1, 0.2, 0.25, 0.5, 1.0)
_MIN_INTERVALS = 3
_RESIDENCE_MAX_INTERVALS = 6
_SNAPSHOT_MAX_INTERVALS = 5

# 粒子釋放範圍（全部釋放點的凸包）：淡紫半透明填色加細框，畫在密度與 HDR 之上、陸地與星號之間，
# 只是位置參考，不搶 HDR 等值線的視覺重心。
_RELEASE_FACE = (0.42, 0.24, 0.60, 0.14)
_RELEASE_EDGE = "#6a3d9a"
_RELEASE_LINEWIDTH = 0.7
COLORBAR_ALIGNMENT_TOLERANCE_PX = 1.0
"""色條與主圖上下緣對齊的驗收容差（顯示座標，像素）。"""

_EDGE_NAME_ZH: Mapping[str, str] = {"west": "西", "east": "東", "south": "南", "north": "北"}


def _corner_name_zh(name: str) -> str:
    """``"south-east"`` -> ``"東南角"``（中文慣例東西在前、南北在後）。"""

    parts = name.split("-")
    ordered = sorted(parts, key=lambda p: 0 if p in ("east", "west") else 1)
    return "".join(_EDGE_NAME_ZH.get(p, p) for p in ordered) + "角"


@dataclass(frozen=True, slots=True)
class FootprintSiteFigureData:
    """單站 WP-38 繪圖輸入：停留時數密度場、HDR、回溯快照、弧長分布與幾何。"""

    site_id: str
    grid: Mapping[str, Any]
    density_pct_per_km2: np.ndarray
    hdr_threshold_pct_per_km2: Mapping[float, float]
    hdr_area_km2: Mapping[float, float]
    snapshots: tuple[Mapping[str, Any], ...]
    arc_bins: tuple[Mapping[str, Any], ...]
    edge_intervals: tuple[Mapping[str, Any], ...]
    domain_lonlat: np.ndarray
    open_boundary_parts: tuple[np.ndarray, ...]
    receptor_center: tuple[float, float] | None
    summary: Mapping[str, Any]
    receptor_anchor: tuple[float, float] | None = None
    receptor_core_radius_m: float | None = None
    receptor_source: str = "postprocess_release_point_mean"
    release_points: np.ndarray | None = None
    """全部相異釋放點（受體水平位置）的經緯度，形狀 ``(n, 2)``；來自後處理的足跡幾何表。"""

    @property
    def receptor_lonlat(self) -> tuple[float, float] | None:
        """星號位置：有正式釋放設定的錨點就用錨點，否則退回釋放點平均位置。"""

        return self.receptor_anchor if self.receptor_anchor is not None else self.receptor_center

    @property
    def lon_edges(self) -> np.ndarray:
        g = self.grid
        return float(g["lon0_deg"]) + np.arange(int(g["nx"]) + 1) * float(g["resolution_deg"])

    @property
    def lat_edges(self) -> np.ndarray:
        g = self.grid
        return float(g["lat0_deg"]) + np.arange(int(g["ny"]) + 1) * float(g["resolution_deg"])

    def density_max(self) -> float:
        return float(np.max(self.density_pct_per_km2)) if self.density_pct_per_km2.size else 0.0


def _footprint_tables_present(postprocess_dir: Path) -> bool:
    return (postprocess_dir / "residence_density_by_site.parquet").is_file()


def _rows_for_site(table: pa.Table, site_id: str) -> list[dict[str, Any]]:
    return _filter_site(table, site_id).to_pylist()


def _load_footprint_figure_inputs(
    postprocess_dir: Path, site_ids: Sequence[str]
) -> tuple[dict[str, FootprintSiteFigureData], dict[str, Path]]:
    """讀取 WP-38 的表；已有停留時數密度表卻缺其他表時 fail-closed。"""

    tables: dict[str, pa.Table] = {}
    paths: dict[str, Path] = {}
    for stem in _FOOTPRINT_TABLE_STEMS:
        path = postprocess_dir / f"{stem}.parquet"
        if not path.is_file():
            raise FiguresError(f"後處理目錄有停留時數密度表，但缺少配套輸入表：{stem}.parquet")
        tables[stem] = pq.read_table(path)
        paths[path.name] = path

    result: dict[str, FootprintSiteFigureData] = {}
    for site_id in site_ids:
        grid_rows = _rows_for_site(tables["footprint_grid_by_site"], site_id)
        if len(grid_rows) != 1:
            raise FiguresError(f"footprint_grid_by_site 對站 {site_id} 應恰有一列，實際 {len(grid_rows)} 列")
        grid = grid_rows[0]
        nx, ny = int(grid["nx"]), int(grid["ny"])

        residence = [
            r
            for r in _rows_for_site(tables["residence_density_by_site"], site_id)
            if r["group_kind"] == "all"
        ]
        density = np.zeros((ny, nx), dtype=np.float64)
        for row in residence:
            density[int(row["cell_iy"]), int(row["cell_ix"])] = float(row["density_fraction_per_km2"]) * 100.0
        if not residence:
            raise FiguresError(f"residence_density_by_site 沒有站 {site_id} 的全體停留時數列")

        hdr_rows = [
            r for r in _rows_for_site(tables["residence_hdr_by_site"], site_id) if r["group_kind"] == "all"
        ]
        thresholds = {
            round(float(r["level"]), 6): float(r["density_threshold_fraction_per_km2"]) * 100.0
            for r in hdr_rows
        }
        areas = {round(float(r["level"]), 6): float(r["area_km2"]) for r in hdr_rows}

        snap_density_rows = _rows_for_site(tables["lookback_snapshot_density_by_site"], site_id)
        snap_hdr_rows = [
            r
            for r in _rows_for_site(tables["residence_hdr_by_site"], site_id)
            if r["group_kind"] == "snapshot"
        ]
        snapshot_list = []
        for status_row in sorted(
            _rows_for_site(tables["lookback_snapshot_by_site"], site_id), key=lambda r: r["snapshot_hours"]
        ):
            hours_ = int(status_row["snapshot_hours"])
            snap_density: np.ndarray | None = None
            if status_row["status"] == "available":
                snap_density = np.zeros((ny, nx), dtype=np.float64)
                for r in snap_density_rows:
                    if int(r["snapshot_hours"]) == hours_:
                        snap_density[int(r["cell_iy"]), int(r["cell_ix"])] = (
                            float(r["density_fraction_per_km2"]) * 100.0
                        )
            snapshot_list.append(
                {
                    **status_row,
                    "density_pct_per_km2": snap_density,
                    "hdr_threshold_pct_per_km2": {
                        round(float(r["level"]), 6): float(r["density_threshold_fraction_per_km2"]) * 100.0
                        for r in snap_hdr_rows
                        if r["group_value"] == str(hours_)
                    },
                }
            )
        snapshots = tuple(snapshot_list)
        arc_bins = tuple(
            sorted(
                _rows_for_site(tables["boundary_departure_along_arc_by_site"], site_id),
                key=lambda r: r["bin_index"],
            )
        )
        edges = tuple(
            sorted(
                _rows_for_site(tables["boundary_edge_intervals_by_site"], site_id),
                key=lambda r: (r["s_start_km"], r["kind"] != "edge"),
            )
        )

        geometry = _rows_for_site(tables["footprint_geometry_by_site"], site_id)
        domain = sorted(
            (r for r in geometry if r["element"] == "flow_domain"), key=lambda r: r["vertex_index"]
        )
        domain_lonlat = np.asarray([[r["lon_deg"], r["lat_deg"]] for r in domain], dtype=np.float64)
        if domain_lonlat.shape[0] < 3:
            raise FiguresError(f"footprint_geometry_by_site 對站 {site_id} 缺少 flow domain 外框")
        parts: dict[int, list[dict[str, Any]]] = {}
        for r in geometry:
            if r["element"] == "open_boundary":
                parts.setdefault(int(r["part_index"]), []).append(r)
        boundary_parts = tuple(
            np.asarray(
                [
                    [r["lon_deg"], r["lat_deg"], r["s_km"]]
                    for r in sorted(parts[p], key=lambda r: r["vertex_index"])
                ],
                dtype=np.float64,
            )
            for p in sorted(parts)
        )
        release = [r for r in geometry if r["element"] == "receptor"]
        release_points = (
            np.asarray(
                [[r["lon_deg"], r["lat_deg"]] for r in sorted(release, key=lambda r: r["vertex_index"])],
                dtype=np.float64,
            )
            if release
            else None
        )
        centers = [r for r in geometry if r["element"] == "receptor_center"]
        receptor_center = (float(centers[0]["lon_deg"]), float(centers[0]["lat_deg"])) if centers else None

        summary_rows = _rows_for_site(tables["footprint_summary_by_site"], site_id)
        if len(summary_rows) != 1:
            raise FiguresError(f"footprint_summary_by_site 對站 {site_id} 應恰有一列")

        result[site_id] = FootprintSiteFigureData(
            site_id=site_id,
            grid=grid,
            density_pct_per_km2=density,
            hdr_threshold_pct_per_km2=thresholds,
            hdr_area_km2=areas,
            snapshots=snapshots,
            arc_bins=arc_bins,
            edge_intervals=edges,
            domain_lonlat=domain_lonlat,
            open_boundary_parts=boundary_parts,
            receptor_center=receptor_center,
            summary=summary_rows[0],
            release_points=release_points,
        )
    return result, paths


# ── 等深線 ───────────────────────────────────────────────────────────────


def _load_isobaths(path: Path | None) -> tuple[dict[str, dict[float, list[np.ndarray]]], dict[str, Any]]:
    """讀取等深線 GeoJSON（``scripts/export_isobaths.py`` 的輸出）。

    回傳 ``({站: {水深: [線段陣列 (n, 2)]}}, 紀錄)``。沒有提供路徑或檔案不存在時
    回傳空對照並在紀錄註明原因（圖仍照常繪製，只是沒有等深線）；檔案存在但格式
    不符則 fail-closed。
    """

    if path is None:
        return {}, {"status": "not_provided", "reason": "未提供 --bathymetry"}
    if not path.is_file():
        return {}, {"status": "file_not_found", "reason": f"等深線檔案不存在：{path.name}"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FiguresError(f"無法讀取等深線 GeoJSON：{exc}") from exc
    crs_block = payload.get("crs") if isinstance(payload, dict) else None
    crs_name = crs_block.get("properties", {}).get("name") if isinstance(crs_block, dict) else None
    if not isinstance(payload, dict) or payload.get("type") != "FeatureCollection":
        raise FiguresError("等深線 GeoJSON 必須是 FeatureCollection")
    if crs_name != "urn:ogc:def:crs:OGC:1.3:CRS84":
        raise FiguresError("等深線 GeoJSON 必須明示 CRS84（經度、緯度）座標順序")
    result: dict[str, dict[float, list[np.ndarray]]] = {}
    for index, feature in enumerate(payload.get("features", [])):
        try:
            props = feature["properties"]
            site_id, depth = str(props["site_id"]), float(props["depth_m"])
            geometry = feature["geometry"]
            coords = geometry["coordinates"]
            gtype = geometry["type"]
        except (KeyError, TypeError, ValueError) as exc:
            raise FiguresError(f"等深線 GeoJSON feature[{index}] 缺少 site_id／depth_m／geometry") from exc
        if gtype == "LineString":
            coords = [coords]
        elif gtype != "MultiLineString":
            raise FiguresError(f"等深線 GeoJSON feature[{index}] 必須是 LineString／MultiLineString")
        segments = [np.asarray(seg, dtype=np.float64) for seg in coords if len(seg) >= 2]
        if any(seg.ndim != 2 or seg.shape[1] != 2 or not np.all(np.isfinite(seg)) for seg in segments):
            raise FiguresError(f"等深線 GeoJSON feature[{index}] 座標必須是有限的 (經度, 緯度)")
        result.setdefault(site_id, {}).setdefault(depth, []).extend(segments)
    return result, {"status": "loaded", "features": len(payload.get("features", []))}


def _axis_ticks(lo: float, hi: float, *, max_intervals: int) -> np.ndarray:
    """把 ``[lo, hi]`` 向外擴到刻度間距的整數倍，回傳等間距刻度（首末刻度即軸範圍）。

    間距在 :data:`_TICK_STEP_CHOICES_DEG`（0.1、0.2、0.25、0.5、1 度）中挑最小、且刻度間隔數不超過
    ``max_intervals`` 者；範圍太小、間隔數不到 :data:`_MIN_INTERVALS`（即刻度少於 4 個）時，
    在兩側輪流多擴一格。範圍一定完整包含 ``[lo, hi]``，首末刻度正好落在框邊，沒有無刻度的空白段。
    """

    step = _TICK_STEP_CHOICES_DEG[-1]
    k0 = k1 = 0
    for step in _TICK_STEP_CHOICES_DEG:
        k0 = math.floor(lo / step + 1e-9)
        k1 = math.ceil(hi / step - 1e-9)
        if k1 - k0 <= max_intervals:
            break
    if k1 <= k0:
        k1 = k0 + 1
    grow_high = True
    while k1 - k0 < _MIN_INTERVALS:
        if grow_high:
            k1 += 1
        else:
            k0 -= 1
        grow_high = not grow_high
    return np.round(np.arange(k0, k1 + 1) * step, 10)


def _site_axis_ticks(
    fp: FootprintSiteFigureData, margin_fraction: float = 0.04, max_intervals: int = _RESIDENCE_MAX_INTERVALS
) -> tuple[np.ndarray, np.ndarray]:
    """地圖的經度、緯度刻度；範圍是流場範圍外框加邊界餘量後向外擴到刻度倍數。"""

    lon = fp.domain_lonlat[:, 0]
    lat = fp.domain_lonlat[:, 1]
    dx = (float(lon.max()) - float(lon.min())) * margin_fraction
    dy = (float(lat.max()) - float(lat.min())) * margin_fraction
    return (
        _axis_ticks(float(lon.min()) - dx, float(lon.max()) + dx, max_intervals=max_intervals),
        _axis_ticks(float(lat.min()) - dy, float(lat.max()) + dy, max_intervals=max_intervals),
    )


def _site_extent(
    fp: FootprintSiteFigureData, margin_fraction: float = 0.04, max_intervals: int = _RESIDENCE_MAX_INTERVALS
) -> tuple[float, float, float, float]:
    xt, yt = _site_axis_ticks(fp, margin_fraction, max_intervals)
    return float(xt[0]), float(xt[-1]), float(yt[0]), float(yt[-1])


def _nice_ceiling_axis(value: float, *, max_intervals: int) -> tuple[float, np.ndarray]:
    """從 0 起算、向上取整到「漂亮間距」的軸：回傳 ``(軸上限, 刻度)``，刻度間隔數不超過上限。"""

    if not math.isfinite(value) or value <= 0:
        return 1.0, np.asarray([0.0, 1.0])
    exponent = math.floor(math.log10(value / max_intervals)) - 1
    while True:
        for mantissa in (1.0, 2.0, 2.5, 5.0):
            step = mantissa * 10.0**exponent
            n = math.ceil(value / step - 1e-9)
            if n <= max_intervals:
                n = max(n, 1)
                return round(n * step, 10), np.round(np.arange(n + 1) * step, 10)
        exponent += 1


def _inside_runs(seg: np.ndarray, extent: tuple[float, float, float, float]) -> list[np.ndarray]:
    """把折線裁成落在範圍內的連續段（含只有兩個頂點但穿過範圍的長線段）。"""

    import shapely

    x0, x1, y0, y1 = extent
    clipped = shapely.clip_by_rect(shapely.linestrings(seg), x0, y0, x1, y1)
    if clipped.is_empty:
        return []
    parts = list(clipped.geoms) if hasattr(clipped, "geoms") else [clipped]
    runs: list[np.ndarray] = []
    for part in parts:
        if part.geom_type != "LineString":
            continue
        coords = np.asarray(part.coords, dtype=np.float64)
        if coords.shape[0] >= 2:
            runs.append(coords)
    return runs


def _isobath_linestyle(depth: float, levels: Sequence[float]) -> str:
    ordered = sorted(levels)
    idx = min(range(len(ordered)), key=lambda i: abs(ordered[i] - depth)) if ordered else 0
    return _ISOBATH_LINESTYLES[idx % len(_ISOBATH_LINESTYLES)]


def _draw_isobaths(
    ax,
    site_isobaths: Mapping[float, list[np.ndarray]],
    extent: tuple[float, float, float, float],
    levels: Sequence[float],
) -> list[float]:
    """畫指定水深的等深線（中灰 0.4 pt，最淺一條實線、其次虛線），回傳實際畫出的水深。

    地圖上不放水深數字標籤（標籤是干擾高密度區間的主要來源）；水深與線型改由圖例與圖說交代。
    圖層順序在密度色塊之上、HDR 等值線之下。
    """

    drawn: list[float] = []
    for depth in sorted(levels):
        key = next((d for d in site_isobaths if abs(d - depth) < 1e-9), None)
        if key is None:
            continue
        runs: list[np.ndarray] = []
        for seg in site_isobaths[key]:
            runs.extend(_inside_runs(seg, extent))
        if not runs:
            continue
        style = _isobath_linestyle(depth, levels)
        for run in runs:
            ax.plot(
                run[:, 0],
                run[:, 1],
                color=_ISOBATH_COLOR,
                linewidth=_ISOBATH_LINEWIDTH,
                linestyle=style,
                zorder=3,
            )
        drawn.append(depth)
    return drawn


def _lon_formatter(value: float, _pos: int | None = None) -> str:
    return f"{value:.2f}°E"


def _lat_formatter(value: float, _pos: int | None = None) -> str:
    return f"{value:.2f}°N"


def _release_hull(points: np.ndarray | None):
    """全部釋放點的凸包（shapely 幾何）；設定檔只定義受體核心圓（錨點加半徑），沒有釋放多邊形，
    所以「實際釋放範圍」取後處理表中各受體釋放點的外包絡。沒有釋放點時回傳 ``None``。"""

    from shapely.geometry import MultiPoint

    if points is None or len(points) == 0:
        return None
    return MultiPoint([(float(x), float(y)) for x, y in points]).convex_hull


_EARTH_RADIUS_KM = 6371.0088


def _haversine_km(lon1, lat1, lon2, lat2):
    """兩點（或點與陣列）的球面大圓距離（km，地球半徑取 6371.0088 km）。"""

    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(np.asarray(lon2) - lon1)
    h = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2.0) ** 2
    return 2.0 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(h))


def _release_geometry_record(
    fp: FootprintSiteFigureData, land_geometries: Sequence[Polygon | MultiPolygon]
) -> dict[str, Any]:
    """粒子釋放範圍與錨點的核對數值，寫入 manifest。

    ``anchor_to_release_centroid_km``：錨點到釋放點重心（經緯度算術平均）的距離；
    ``release_max_radius_km``：釋放點到重心的最大距離；``release_max_distance_from_anchor_km``：
    釋放點到錨點的最大距離（應不超過受體核心半徑）；``anchor_on_sea``：錨點不在海岸線 GeoJSON
    的任何陸地多邊形內。沒有釋放設定（無錨點）或沒有釋放點時對應欄位為 ``None``。
    """

    from shapely.geometry import Point

    pts = fp.release_points
    anchor = fp.receptor_anchor
    record: dict[str, Any] = {
        "release_point_count": int(len(pts)) if pts is not None else 0,
        "release_points_lonlat": [[float(x), float(y)] for x, y in pts] if pts is not None else [],
        "distance_method": f"球面大圓（haversine，地球半徑 {_EARTH_RADIUS_KM} km）",
        "release_area_definition": (
            "全部相異釋放點（後處理足跡幾何表的受體水平位置）的凸包；設定檔只定義受體核心圓"
            "（錨點加 receptor_core_radius_m），沒有釋放多邊形"
        ),
    }
    if pts is not None and len(pts):
        centroid = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
        record["release_centroid_lonlat"] = list(centroid)
        to_centroid = _haversine_km(centroid[0], centroid[1], pts[:, 0], pts[:, 1])
        record["release_max_radius_km"] = float(np.max(to_centroid))
        hull = _release_hull(pts)
        record["release_hull_vertex_count"] = (
            len(hull.exterior.coords) - 1 if hull is not None and hull.geom_type == "Polygon" else 0
        )
        record["release_points_on_land_count"] = int(
            sum(any(g.contains(Point(float(x), float(y))) for g in land_geometries) for x, y in pts)
        )
    else:
        centroid = None
    if anchor is not None:
        record["anchor_lonlat"] = list(anchor)
        record["anchor_on_sea"] = not any(g.contains(Point(anchor[0], anchor[1])) for g in land_geometries)
        if centroid is not None and pts is not None:
            record["anchor_to_release_centroid_km"] = float(
                _haversine_km(anchor[0], anchor[1], centroid[0], centroid[1])
            )
            record["release_max_distance_from_anchor_km"] = float(
                np.max(_haversine_km(anchor[0], anchor[1], pts[:, 0], pts[:, 1]))
            )
    else:
        record["anchor_lonlat"] = None
        record["anchor_on_sea"] = None
    return record


def _hdr_masks(density: np.ndarray, thresholds: Mapping[float, float]) -> list[np.ndarray]:
    return [density >= thresholds[level] * (1.0 - 1e-9) for level in sorted(thresholds)]


def _draw_footprint_base_map(
    ax,
    fp: FootprintSiteFigureData,
    land_geometries: Sequence[Polygon | MultiPolygon],
    site_isobaths: Mapping[float, list[np.ndarray]],
    *,
    isobath_levels: Sequence[float] = ISOBATH_LEVELS_DEFAULT,
    max_intervals: int = _RESIDENCE_MAX_INTERVALS,
    tick_fontsize: float = 8.0,
) -> tuple[tuple[float, float, float, float], list[float]]:
    """陸地（灰）、海岸線（黑 0.5 pt）、等深線、開放邊界（洋紅虛線）、粒子釋放範圍與受體星號。

    圖面不寫站名（星號旁沒有文字標籤，圖例只寫「受體（研究點錨點）」）；受體位置用
    ``fp.receptor_lonlat``，粒子釋放範圍是全部釋放點的凸包（見 :func:`_release_hull`）。
    座標軸範圍與刻度見 :func:`_axis_ticks`：首末刻度就是四個框邊。回傳
    ``(座標軸範圍, 實際畫出的等深線水深)``。
    """

    from matplotlib.ticker import FuncFormatter
    from shapely.plotting import plot_polygon

    xticks, yticks = _site_axis_ticks(fp, max_intervals=max_intervals)
    extent = (float(xticks[0]), float(xticks[-1]), float(yticks[0]), float(yticks[-1]))
    for land in land_geometries:
        plot_polygon(
            land,
            ax=ax,
            add_points=False,
            facecolor=_LAND_COLOR,
            edgecolor="black",
            linewidth=0.5,
            zorder=5.5,  # 陸地蓋在密度與 HDR 線之上，密度不會畫到陸地上
        )
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect(1.0 / math.cos(math.radians((extent[2] + extent[3]) / 2.0)), adjustable="box")
    drawn = _draw_isobaths(ax, site_isobaths, extent, isobath_levels)
    for part in fp.open_boundary_parts:
        ax.plot(part[:, 0], part[:, 1], zorder=6, **_OPEN_BOUNDARY_STYLE)
    hull = _release_hull(fp.release_points)
    if hull is not None:
        if hull.geom_type == "Polygon":
            ax.fill(
                *hull.exterior.xy,
                facecolor=_RELEASE_FACE,
                edgecolor=_RELEASE_EDGE,
                linewidth=_RELEASE_LINEWIDTH,
                zorder=7,
            )
        else:  # 釋放點不足三個或共線：以線段或點表示
            xy = np.asarray(hull.coords)
            ax.plot(xy[:, 0], xy[:, 1], color=_RELEASE_EDGE, linewidth=_RELEASE_LINEWIDTH, zorder=7)
    receptor = fp.receptor_lonlat
    if receptor is not None:
        ax.plot(
            receptor[0],
            receptor[1],
            marker="*",
            markersize=13,
            markerfacecolor="#ffd92f",
            markeredgecolor="black",
            markeredgewidth=0.7,
            linestyle="none",
            zorder=8,
        )
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.xaxis.set_major_formatter(FuncFormatter(_lon_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(_lat_formatter))
    ax.tick_params(labelsize=tick_fontsize)
    ax.set_facecolor("white")
    return extent, drawn


def _footprint_legend_handles(
    t: _Labeler,
    hdr_levels: Sequence[float],
    *,
    isobath_levels_drawn: Sequence[float],
    isobath_levels: Sequence[float],
    receptor_is_anchor: bool = True,
    has_release_range: bool = True,
) -> list[Any]:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [
        Line2D(
            [],
            [],
            label=t.t(f"高密度區間 {level * 100:.0f}%", f"HDR {level * 100:.0f}%"),
            **_HDR_LINE_STYLES[i % len(_HDR_LINE_STYLES)],
        )
        for i, level in enumerate(hdr_levels)
    ]
    handles.append(
        Line2D([], [], label=t.t(_OPEN_BOUNDARY_LABEL_ZH, _OPEN_BOUNDARY_LABEL_EN), **_OPEN_BOUNDARY_STYLE)
    )
    handles.append(
        Line2D(
            [],
            [],
            marker="*",
            markersize=12,
            markerfacecolor="#ffd92f",
            markeredgecolor="black",
            linestyle="none",
            label=(
                t.t("受體（研究點錨點）", "Receptor (study-site anchor)")
                if receptor_is_anchor
                else t.t("受體（釋放點平均位置）", "Receptor (mean release position)")
            ),
        )
    )
    if has_release_range:
        handles.append(
            Patch(
                facecolor=_RELEASE_FACE,
                edgecolor=_RELEASE_EDGE,
                linewidth=_RELEASE_LINEWIDTH,
                label=t.t("粒子釋放範圍", "Particle release area"),
            )
        )
    ordered = sorted(isobath_levels)
    for depth in sorted(isobath_levels_drawn):
        idx = min(range(len(ordered)), key=lambda i: abs(ordered[i] - depth))
        k = idx % len(_ISOBATH_LINESTYLES)
        handles.append(
            Line2D(
                [],
                [],
                color=_ISOBATH_COLOR,
                linewidth=_ISOBATH_LINEWIDTH,
                linestyle=_ISOBATH_LINESTYLES[k],
                label=t.t(
                    f"等深線 {depth:g} m（{_ISOBATH_STYLE_ZH[k]}）",
                    f"Isobath {depth:g} m ({_ISOBATH_STYLE_EN[k]})",
                ),
            )
        )
    return handles


def _log_colorbar_ticks(vmin: float, vmax: float) -> list[float]:
    """對數色條刻度：上下限之間每個 10 的整數次方一個刻度，兩端一定在內。"""

    lo, hi = round(math.log10(vmin)), round(math.log10(vmax))
    ticks = [10.0**e for e in range(lo, hi + 1)]
    if len(ticks) < 2:
        return [vmin, vmax]
    ticks[0], ticks[-1] = vmin, vmax
    return ticks


def _decade_formatter(value: float, _pos: int | None = None) -> str:
    """色條刻度文字：``10^-4`` 的數學排版形式。"""

    if value <= 0:
        return ""
    return f"$10^{{{round(math.log10(value))}}}$"


def _decade_limits(vmax_raw: float) -> tuple[float, float]:
    """對數色階上下限：上限＝大於或等於最大值的 10 的整數次方，下限＝上限的 1/10000。

    上限一定 >= 資料最大值（浮點誤差時多進一位），所以不會有格子高於上限。
    """

    if not math.isfinite(vmax_raw) or vmax_raw <= 0:
        return 10.0**-4, 1.0
    exponent = math.ceil(math.log10(vmax_raw) - 1e-9)
    while 10.0**exponent < vmax_raw:
        exponent += 1
    return 10.0 ** (exponent - 4), 10.0**exponent


def _residence_color_limits(vmax: float) -> tuple[float, float]:
    return _decade_limits(vmax)


def _pow10_text(value: float) -> str:
    return f"10^{round(math.log10(value))}"


def _snapshot_color_limits(fp: FootprintSiteFigureData) -> tuple[float, float]:
    """快照圖四個面板共用的上下限：四個面板最大密度向上取整到 10 的整數次方。"""

    peaks = [
        float(s["density_pct_per_km2"].max())
        for s in fp.snapshots[:4]
        if s["density_pct_per_km2"] is not None
    ]
    return _decade_limits(max(peaks, default=0.0))


def _limit_stats(
    fp: FootprintSiteFigureData, density: np.ndarray, vmin: float, vmax: float
) -> dict[str, Any]:
    """低於下限與高於上限的網格數，及其占全部停留時數（或該面板質量）的比例。

    色階不做視覺裁切（低於下限者透明、高於上限者本應為 0），越界情形只記錄在這裡。
    """

    res = float(fp.grid["resolution_deg"])
    lat_centers = float(fp.grid["lat0_deg"]) + (np.arange(int(fp.grid["ny"])) + 0.5) * res
    area = cell_area_km2_by_row(lat_centers, res, res)[:, None]
    mass = np.where(np.isfinite(density), density, 0.0) * area
    total = float(mass.sum())
    positive = density > 0
    below = positive & (density < vmin)
    above = density > vmax
    return {
        "cells_with_density": int(positive.sum()),
        "cells_below_lower_limit": int(below.sum()),
        "mass_fraction_below_lower_limit": float(mass[below].sum() / total) if total > 0 else 0.0,
        "cells_above_upper_limit": int(above.sum()),
        "mass_fraction_above_upper_limit": float(mass[above].sum() / total) if total > 0 else 0.0,
    }


def _add_aligned_colorbar(
    fig,
    mesh,
    map_axes: Sequence[Any],
    *,
    vmin: float,
    vmax: float,
    label: str,
    font: Any,
    tick_fontsize: float = 8.0,
    width: float = 0.018,
    pad: float = 0.014,
):
    """矩形色條（``extend='neither'``），上緣、下緣與實際畫出的地圖軸對齊。

    先讓每個地圖軸套用等比例（aspect）取得實際位置，再依這些位置決定色條軸的位置；
    不使用 ``fig.colorbar`` 的預設版面。多個地圖軸（2×2 快照）時，色條從最上排的上緣
    量到最下排的下緣，四個面板共用同一條色條。
    """

    from matplotlib.ticker import FuncFormatter

    for a in map_axes:
        a.apply_aspect()
    pos = [a.get_position() for a in map_axes]
    bottom = min(p.y0 for p in pos)
    top = max(p.y1 for p in pos)
    right = max(p.x1 for p in pos)
    cax = fig.add_axes((right + pad, bottom, width, top - bottom))
    cbar = fig.colorbar(mesh, cax=cax, extend="neither")
    cbar.set_ticks(_log_colorbar_ticks(vmin, vmax))
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(_decade_formatter))
    cbar.ax.minorticks_off()
    cbar.ax.set_ylim(vmin, vmax)
    cbar.ax.tick_params(labelsize=tick_fontsize)
    cbar.set_label(label, fontproperties=font)
    return cbar


def _layout_record(fig, map_axes: Sequence[Any], label_axes, cbar=None) -> dict[str, Any]:
    """畫完後量測並記錄版面：色條與地圖軸的上下緣顯示座標、座標軸範圍與刻度、色條端點刻度。

    ``label_axes`` 是有刻度文字的那個軸（2×2 快照為左下面板）。像素是 ``fig.dpi`` 下的顯示
    座標；版面以圖的比例座標決定，不隨輸出 dpi 改變。
    """

    fig.canvas.draw()
    top_map = max(a.get_window_extent().y1 for a in map_axes)
    bottom_map = min(a.get_window_extent().y0 for a in map_axes)
    ax = label_axes
    record: dict[str, Any] = {
        "measure_dpi": float(fig.dpi),
        "map_axes_top_px": float(top_map),
        "map_axes_bottom_px": float(bottom_map),
        "xlim": [float(v) for v in ax.get_xlim()],
        "ylim": [float(v) for v in ax.get_ylim()],
        "xticks": [float(v) for v in ax.get_xticks()],
        "yticks": [float(v) for v in ax.get_yticks()],
        "xtick_labels": [tl.get_text() for tl in ax.get_xticklabels()],
        "ytick_labels": [tl.get_text() for tl in ax.get_yticklabels()],
    }
    if cbar is not None:
        cb = cbar.ax.get_window_extent()
        record.update(
            {
                "colorbar_top_px": float(cb.y1),
                "colorbar_bottom_px": float(cb.y0),
                "top_diff_px": float(cb.y1 - top_map),
                "bottom_diff_px": float(cb.y0 - bottom_map),
                "colorbar": {
                    "extend": "neither",
                    "ylim": [float(v) for v in cbar.ax.get_ylim()],
                    "ticks": [float(v) for v in cbar.ax.get_yticks()],
                    "tick_labels": [tl.get_text() for tl in cbar.ax.get_yticklabels()],
                },
            }
        )
    return record


def _domain_clip_patch(ax, fp: FootprintSiteFigureData):
    """流場範圍（flow domain）外框的裁切用多邊形；密度只畫在流場範圍內。"""

    from matplotlib.patches import Polygon as MplPolygon

    patch = MplPolygon(
        fp.domain_lonlat, closed=True, transform=ax.transData, facecolor="none", edgecolor="none"
    )
    ax.add_patch(patch)
    return patch


def _draw_hdr_contours(
    ax, lon: np.ndarray, lat: np.ndarray, masks: Sequence[np.ndarray], clip_patch: Any = None
) -> None:
    for i, mask in enumerate(masks):
        if not mask.any():
            continue
        style = _HDR_LINE_STYLES[i % len(_HDR_LINE_STYLES)]
        contour = ax.contour(
            lon,
            lat,
            mask.astype(np.float64),
            levels=[0.5],
            colors=[style["color"]],
            linestyles=[style["linestyle"]],
            linewidths=[style["linewidth"]],
            zorder=5,
        )
        if clip_patch is not None:
            contour.set_clip_path(clip_patch)


def _plot_source_footprint_residence(
    site: SiteData,
    fp: FootprintSiteFigureData,
    land_geometries: Sequence[Polygon | MultiPolygon],
    site_isobaths: Mapping[float, list[np.ndarray]],
    t: _Labeler,
    font: Any,
    path: Path,
    dpi: int,
    figure_text: str,
    vmax: float,
    *,
    scale_kind: str,
    isobath_levels: Sequence[float] = ISOBATH_LEVELS_DEFAULT,
) -> dict[str, Any]:
    """停留時數密度圖：對數色階（上限取 10 的整數次方、下限為上限的 1/10000）、HDR 50／75／90%、
    底圖與受體。色條與地圖軸同高。回傳繪圖紀錄（含色階越界統計與版面量測）。"""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    fig, ax = plt.subplots(figsize=(9.6, 8.6), dpi=150)
    fig.subplots_adjust(left=0.1, bottom=0.14, right=0.86, top=_title_top_margin(figure_text, full=0.86))
    density = fp.density_pct_per_km2
    levels = sorted(fp.hdr_threshold_pct_per_km2)
    masks = _hdr_masks(density, fp.hdr_threshold_pct_per_km2)
    _extent, drawn = _draw_footprint_base_map(
        ax, fp, land_geometries, site_isobaths, isobath_levels=isobath_levels
    )
    vmin, vmax_used = _residence_color_limits(vmax)
    masked = np.ma.masked_less(np.ma.masked_invalid(density), vmin)
    cmap = plt.get_cmap(_RESIDENCE_CMAP).with_extremes(bad=(1.0, 1.0, 1.0, 0.0))
    mesh = ax.pcolormesh(
        fp.lon_edges,
        fp.lat_edges,
        masked,
        cmap=cmap,
        norm=LogNorm(vmin=vmin, vmax=vmax_used),
        shading="flat",
        zorder=2,
    )
    clip_patch = _domain_clip_patch(ax, fp)
    mesh.set_clip_path(clip_patch)
    cbar = _add_aligned_colorbar(
        fig,
        mesh,
        [ax],
        vmin=vmin,
        vmax=vmax_used,
        label=t.t("停留時數密度（% 停留時數／km²）", "Residence density (% of particle-hours / km^2)"),
        font=font,
    )
    lon_c = 0.5 * (fp.lon_edges[:-1] + fp.lon_edges[1:])
    lat_c = 0.5 * (fp.lat_edges[:-1] + fp.lat_edges[1:])
    _draw_hdr_contours(ax, lon_c, lat_c, masks, clip_patch)

    ax.set_xlabel(t.t("經度", "Longitude"), fontproperties=font)
    ax.set_ylabel(t.t("緯度", "Latitude"), fontproperties=font)
    ax.legend(
        handles=_footprint_legend_handles(
            t,
            levels,
            isobath_levels_drawn=drawn,
            isobath_levels=isobath_levels,
            receptor_is_anchor=fp.receptor_anchor is not None,
            has_release_range=fp.release_points is not None,
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.09),
        ncol=3,
        prop=font,
        frameon=True,
    )
    total_hours = int(fp.grid["total_particle_hours"])
    resolution = float(fp.grid["resolution_deg"])
    scale_text = (
        t.t("跨站共用色階", "shared cross-site scale")
        if scale_kind == "shared"
        else t.t("本站自訂色階", "per-site scale")
    )
    _add_two_line_title(
        fig,
        t,
        font,
        t.t(f"停留時數密度｜{site.site_label_zh}", f"Residence density | {site.site_id}"),
        _site_subtitle(
            site,
            t,
            extra=t.t(
                f"\n總停留時數：{total_hours:,} 粒子小時｜網格 {resolution:.2f}°｜{scale_text}",
                f"\n{total_hours:,} particle-hours | grid {resolution:.2f} deg | {scale_text}",
            ),
        ),
        figure_text=figure_text,
    )
    layout = _layout_record(fig, [ax], ax, cbar)
    _save_exclusive(fig, path, dpi=dpi)
    plt.close(fig)
    return {
        "isobath_levels_m": drawn,
        "vmin": vmin,
        "vmax": vmax_used,
        "raw_max": float(vmax),
        "scale": scale_kind,
        "limit_stats": _limit_stats(fp, density, vmin, vmax_used),
        "layout": layout,
    }


def _plot_lookback_snapshot(
    site: SiteData,
    fp: FootprintSiteFigureData,
    land_geometries: Sequence[Polygon | MultiPolygon],
    site_isobaths: Mapping[float, list[np.ndarray]],
    t: _Labeler,
    font: Any,
    path: Path,
    dpi: int,
    figure_text: str,
    *,
    isobath_levels: Sequence[float] = ISOBATH_LEVELS_DEFAULT,
) -> dict[str, Any]:
    """2×2 回溯快照圖：各面板是該回溯時刻仍在漂流的粒子的條件式來源足跡（與停留時數密度同網格）。
    四個面板共用一條色條，從最上排上緣量到最下排下緣。"""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    snapshots = list(fp.snapshots)[:4]
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.6), dpi=150, sharex=True, sharey=True)
    fig.subplots_adjust(
        left=0.07,
        bottom=0.13,
        right=0.86,
        top=_title_top_margin(figure_text, full=0.91, none=0.96),
        hspace=0.10,
        wspace=0.17,
    )
    flat_axes = list(axes.ravel())

    densities: list[np.ndarray | None] = [snap["density_pct_per_km2"] for snap in snapshots]
    vmin, vmax = _snapshot_color_limits(fp)
    has_density = any(d is not None for d in densities)
    cmap = plt.get_cmap(_RESIDENCE_CMAP).with_extremes(bad=(1.0, 1.0, 1.0, 0.0))
    mesh = None
    drawn_levels: list[float] = []
    panel_stats: list[dict[str, Any]] = []
    levels = sorted(fp.hdr_threshold_pct_per_km2) or [0.5, 0.75, 0.9]
    lon_c = 0.5 * (fp.lon_edges[:-1] + fp.lon_edges[1:])
    lat_c = 0.5 * (fp.lat_edges[:-1] + fp.lat_edges[1:])

    for ax, snap, density in zip(flat_axes, snapshots, densities, strict=False):
        masks: list[np.ndarray] = []
        if density is not None:
            thresholds = snap["hdr_threshold_pct_per_km2"]
            masks = _hdr_masks(density, thresholds)
        _extent, drawn = _draw_footprint_base_map(
            ax,
            fp,
            land_geometries,
            site_isobaths,
            isobath_levels=isobath_levels,
            max_intervals=_SNAPSHOT_MAX_INTERVALS,
            tick_fontsize=7.5,
        )
        drawn_levels = drawn or drawn_levels
        hours = int(snap["snapshot_hours"])
        active_pct = float(snap["active_fraction"]) * 100.0
        title = t.t(
            f"回溯 {hours} 小時（{hours / 24:g} 天）｜仍在漂流 {active_pct:.1f}%",
            f"{hours} h back ({hours / 24:g} d) | still drifting {active_pct:.1f}%",
        )
        if density is not None:
            masked = np.ma.masked_less(np.ma.masked_invalid(density), vmin)
            mesh = ax.pcolormesh(
                fp.lon_edges,
                fp.lat_edges,
                masked,
                cmap=cmap,
                norm=LogNorm(vmin=vmin, vmax=vmax),
                shading="flat",
                zorder=2,
            )
            clip_patch = _domain_clip_patch(ax, fp)
            mesh.set_clip_path(clip_patch)
            _draw_hdr_contours(ax, lon_c, lat_c, masks, clip_patch)
            panel_stats.append({"snapshot_hours": hours, **_limit_stats(fp, density, vmin, vmax)})
        else:
            ax.text(
                0.5,
                0.9,
                t.t(
                    f"仍在漂流粒子數不足（n={int(snap['active_particle_count'])}）",
                    f"Too few particles still drifting (n={int(snap['active_particle_count'])})",
                ),
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontproperties=font,
                fontsize=10,
                zorder=7,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 3.0},
            )
        ax.set_title(title, fontproperties=font, fontsize=10.5)

    for ax in flat_axes[len(snapshots) :]:
        ax.set_visible(False)
    for ax in axes[1, :]:
        ax.set_xlabel(t.t("經度", "Longitude"), fontproperties=font)
    for ax in axes[:, 0]:
        ax.set_ylabel(t.t("緯度", "Latitude"), fontproperties=font)
    visible_axes = [a for a in flat_axes if a.get_visible()]
    cbar = None
    if mesh is not None and has_density:
        cbar = _add_aligned_colorbar(
            fig,
            mesh,
            visible_axes,
            vmin=vmin,
            vmax=vmax,
            label=t.t(
                "條件式來源足跡密度（% 仍在漂流粒子／km²）",
                "Conditional footprint density (% of drifting particles / km^2)",
            ),
            font=font,
            tick_fontsize=7.5,
        )
    fig.legend(
        handles=_footprint_legend_handles(
            t,
            levels,
            isobath_levels_drawn=drawn_levels,
            isobath_levels=isobath_levels,
            receptor_is_anchor=fp.receptor_anchor is not None,
            has_release_range=fp.release_points is not None,
        ),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=4,
        prop=font,
        frameon=True,
    )
    _add_two_line_title(
        fig,
        t,
        font,
        t.t(
            f"回溯快照的條件式來源足跡｜{site.site_label_zh}",
            f"Look-back snapshot footprints | {site.site_id}",
        ),
        _site_subtitle(site, t),
        figure_text=figure_text,
    )
    layout = _layout_record(fig, visible_axes, axes[1, 0], cbar)
    _save_exclusive(fig, path, dpi=dpi)
    plt.close(fig)
    return {
        "isobath_levels_m": drawn_levels,
        "snapshot_hours": [int(s["snapshot_hours"]) for s in snapshots],
        "vmin": vmin if mesh is not None else None,
        "vmax": vmax if mesh is not None else None,
        "panels_estimated": [
            int(s["snapshot_hours"]) for s, d in zip(snapshots, densities, strict=False) if d is not None
        ],
        "limit_stats_by_panel": panel_stats,
        "smoothing_sigma_km": float(snapshots[0]["smoothing_sigma_km"]) if snapshots else None,
        "minimum_active_particle_count": (
            int(snapshots[0]["minimum_active_particle_count"]) if snapshots else None
        ),
        "layout": layout,
    }


def _plot_boundary_departure_along_arc(
    site: SiteData,
    fp: FootprintSiteFigureData,
    t: _Labeler,
    font: Any,
    path: Path,
    dpi: int,
    figure_text: str,
) -> dict[str, Any]:
    """沿開放邊界弧長的一維穿越分布：直條圖（每 bin 占比）＋累積曲線，標出各邊與轉角。

    三條軸都從 0 起算並向上取整到刻度倍數：橫軸 0 到邊界全長向上取整；左縱軸 0 到最大占比
    向上取整；右縱軸 0–100%；首末刻度就是軸的兩端，兩端都標值。各邊名稱與轉角名稱寫在上框外側，
    不壓在直條與累積曲線上。
    """

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    fig, ax = plt.subplots(figsize=(12.0, 5.6), dpi=150)
    fig.subplots_adjust(
        left=0.07, bottom=0.14, right=0.93, top=_title_top_margin(figure_text, full=0.80, none=0.96)
    )
    bins = list(fp.arc_bins)
    total = sum(int(b["crossed_particle_count"]) for b in bins)
    length_km = float(bins[0]["boundary_length_km"]) if bins else 0.0
    bin_km = float(bins[0]["bin_km"]) if bins else 0.0
    xmax, xticks = _nice_ceiling_axis(length_km, max_intervals=10)
    ax2 = ax.twinx()
    ymax = 1.0
    if not bins or total == 0:
        ax.text(
            0.5,
            0.5,
            t.t("本站沒有粒子穿越流場範圍開放邊界", "No particle crossed the open boundary"),
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontproperties=font,
        )
        yticks = np.asarray([0.0, 1.0])
    else:
        starts = np.asarray([float(b["s_start_km"]) for b in bins])
        ends = np.asarray([float(b["s_end_km"]) for b in bins])
        share = np.asarray([float(b["fraction"]) for b in bins]) * 100.0
        cum = np.asarray([float(b["cumulative_fraction"]) for b in bins]) * 100.0

        edge_colors = ("#eef3f7", "#ffffff")
        edges = [e for e in fp.edge_intervals if e["kind"] == "edge"]
        for k, e in enumerate(edges):
            ax.axvspan(float(e["s_start_km"]), float(e["s_end_km"]), facecolor=edge_colors[k % 2], zorder=0)
        ax.bar(
            starts,
            share,
            width=ends - starts,
            align="edge",
            color="#3a7ca5",
            edgecolor="white",
            linewidth=0.4,
            zorder=2,
        )
        ax2.plot(ends, cum, color="#3b2314", linewidth=1.6, zorder=3)
        ax2.plot([0.0, float(ends[0])], [0.0, float(cum[0])], color="#3b2314", linewidth=1.6, zorder=3)
        ymax, yticks = _nice_ceiling_axis(float(share.max()) * 1.2, max_intervals=5)
        for e in edges:
            centre = 0.5 * (float(e["s_start_km"]) + float(e["s_end_km"]))
            ax.text(
                centre,
                1.012,
                t.t(f"{_EDGE_NAME_ZH.get(e['name'], e['name'])}邊界", f"{e['name']} edge"),
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontproperties=font,
                fontsize=10,
                color="#333333",
                clip_on=False,
            )
        for c in (e for e in fp.edge_intervals if e["kind"] == "corner"):
            sk = float(c["s_start_km"])
            ax.axvline(sk, color="#555555", linewidth=0.8, linestyle="--", zorder=1)
            ax.text(
                sk,
                1.012,
                t.t(_corner_name_zh(str(c["name"])), str(c["name"])),
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontproperties=font,
                fontsize=9,
                color="#444444",
                clip_on=False,
            )
    number = FuncFormatter(lambda v, _p=None: f"{v:g}")
    ax.set_xlim(0.0, float(xticks[-1]))
    ax.set_xticks(xticks)
    ax.set_ylim(0.0, float(yticks[-1]))
    ax.set_yticks(yticks)
    ax.xaxis.set_major_formatter(number)
    ax.yaxis.set_major_formatter(number)
    ax2.set_ylim(0.0, 100.0)
    ax2.set_yticks(np.arange(0.0, 100.0 + 1e-9, 20.0))
    ax2.yaxis.set_major_formatter(number)
    ax2.set_ylabel(t.t("累積比例（%）", "Cumulative share (%)"), fontproperties=font)
    ax.set_xlabel(
        t.t(
            "沿流場範圍開放邊界的弧長（km，自邊界折線起點量起）",
            "Arc length along open boundary (km from start)",
        ),
        fontproperties=font,
    )
    ax.set_ylabel(
        t.t(f"穿越粒子占比（%，每 {bin_km:g} km）", f"Share of crossings (%, per {bin_km:g} km)"),
        fontproperties=font,
    )
    ax.grid(axis="y", color="#dddddd", linewidth=0.5, zorder=0)
    origin = fp.open_boundary_parts[0][0] if fp.open_boundary_parts else None
    origin_zh = (
        f"｜弧長起點：{origin[0]:.3f}°E, {origin[1]:.3f}°N（依邊界折線頂點順序前進）"
        if origin is not None
        else ""
    )
    origin_en = f" | start: {origin[0]:.3f}E, {origin[1]:.3f}N" if origin is not None else ""
    _add_two_line_title(
        fig,
        t,
        font,
        t.t(
            f"穿越位置沿開放邊界的分布｜{site.site_label_zh}",
            f"Crossing position along open boundary | {site.site_id}",
        ),
        _site_subtitle(
            site,
            t,
            extra=t.t(
                f"\n穿越粒子 {total:,} 顆｜邊界長 {length_km:.0f} km{origin_zh}",
                f"\n{total:,} crossed | length {length_km:.0f} km{origin_en}",
            ),
        ),
        figure_text=figure_text,
    )
    fig.canvas.draw()
    layout = {
        "measure_dpi": float(fig.dpi),
        "xlim": [float(v) for v in ax.get_xlim()],
        "ylim_left": [float(v) for v in ax.get_ylim()],
        "ylim_right": [float(v) for v in ax2.get_ylim()],
        "xticks": [float(v) for v in ax.get_xticks()],
        "yticks_left": [float(v) for v in ax.get_yticks()],
        "yticks_right": [float(v) for v in ax2.get_yticks()],
        "xtick_labels": [tl.get_text() for tl in ax.get_xticklabels()],
        "ytick_labels_left": [tl.get_text() for tl in ax.get_yticklabels()],
        "ytick_labels_right": [tl.get_text() for tl in ax2.get_yticklabels()],
    }
    _save_exclusive(fig, path, dpi=dpi)
    plt.close(fig)
    return {
        "crossed_particle_count": total,
        "boundary_length_km": length_km,
        "bin_km": bin_km,
        "layout": layout,
    }


def _footprint_residence_color_scales(
    footprint: Mapping[str, FootprintSiteFigureData],
) -> tuple[float, dict[str, float]]:
    """回傳 ``(跨站共用上限, {站: 各站上限})``，單位 % 停留時數／km²。"""

    per_site = {site_id: fp.density_max() for site_id, fp in footprint.items()}
    shared = max(per_site.values(), default=0.0)
    return shared, per_site


def _hdr_area_text(fp: FootprintSiteFigureData) -> str:
    parts = []
    for level in sorted(fp.hdr_area_km2):
        parts.append(f"{level * 100:.0f}% 為 {fp.hdr_area_km2[level]:,.0f} km²")
    return "、".join(parts)


def _receptor_text_zh(fp: FootprintSiteFigureData) -> str:
    """圖說用的受體位置說明；有正式釋放設定就寫研究點錨點，否則明示是釋放點平均位置。"""

    lonlat = fp.receptor_lonlat
    if lonlat is None:
        return "受體位置未記錄"
    pos = f"{lonlat[0]:.4f}°E, {lonlat[1]:.4f}°N"
    if fp.receptor_anchor is not None:
        radius = (
            f"半徑 {fp.receptor_core_radius_m / 1000.0:g} km 的受體核心範圍內的各受體點"
            if fp.receptor_core_radius_m
            else "周圍的各受體點"
        )
        return f"受體位置（星號）為研究點錨點 {pos}（取自正式釋放設定），粒子自錨點周圍{radius}釋放"
    return f"受體位置（星號）為各釋放點的平均位置 {pos}（本批圖未提供釋放設定，不是研究點錨點）"


def _footprint_caption_zh(
    stem: str,
    site: SiteData,
    fp: FootprintSiteFigureData,
    *,
    experiment_case: _ExperimentCaseDisclosure,
    footprint_manifest: Mapping[str, Any] | None,
    isobath_levels: Sequence[float],
    isobath_reason: str | None,
    shared_vmax: float,
    site_vmax: float,
    isobath_requested: Sequence[float] = ISOBATH_LEVELS_DEFAULT,
) -> str:
    """WP-38 三種新圖的圖說。全部用直述句，不出現程式識別字與檔名。"""

    total = int(site.summary["total_particle_count"])
    member_count = int(site.summary["ensemble_member_count"])
    member_text_zh, _ = _ensemble_member_count_text(member_count)
    settings_tail = (
        f"站點：{site.site_label_zh}（{site.site_id}），{_receptor_text_zh(fp)}，"
        f"{total:,} 顆粒子，{member_text_zh}，終止邊界為流場範圍開放邊界，"
        f"物理案例：{experiment_case.physics_case_zh}"
    )
    if experiment_case.bed_no_deposition_zh is not None:
        settings_tail += f"，{experiment_case.bed_no_deposition_zh}"
    settings_tail += "。"

    arrival_included = True
    if footprint_manifest is not None:
        arrival_included = bool(footprint_manifest.get("arrival_sample_included", True))
    arrival_zh = "回溯起點（到達時刻）的位置計入" if arrival_included else "回溯起點（到達時刻）的位置不計入"
    res = float(fp.grid["resolution_deg"])
    km = res * 111.19
    release_zh = (
        f"淡紫色半透明範圍為粒子釋放範圍（{len(fp.release_points)} 個釋放點的凸包）"
        if fp.release_points is not None and len(fp.release_points)
        else "本站沒有可用的釋放點資料，未繪粒子釋放範圍"
    )
    if isobath_levels:
        ordered = sorted(isobath_requested)
        style_parts = []
        for v in sorted(isobath_levels):
            k = min(range(len(ordered)), key=lambda i: abs(ordered[i] - v)) % len(_ISOBATH_STYLE_ZH)
            style_parts.append(f"{v:g} 公尺（{_ISOBATH_STYLE_ZH[k]}）")
        shelf_zh = "，200 公尺為陸棚邊緣" if any(abs(v - 200.0) < 1e-9 for v in isobath_levels) else ""
        base_zh = (
            "底圖為陸地（灰）與海岸線（黑），中灰細線為 "
            f"{'、'.join(style_parts)}等深線{shelf_zh}（水深取自計算網格的節點水深；"
            "為避免干擾高密度區間，只畫這些水深，地圖上不標水深數字，水深與線型見圖例），"
            f"星號為受體位置，{release_zh}，洋紅色虛線為流場範圍開放邊界（終止邊界）。"
        )
    else:
        base_zh = (
            "底圖為陸地（灰）與海岸線（黑），"
            f"本圖未繪等深線（{isobath_reason or '本站沒有可用的水深資料'}），"
            f"星號為受體位置，{release_zh}，洋紅色虛線為流場範圍開放邊界（終止邊界）。"
        )

    if stem in ("source_footprint_residence", "source_footprint_residence_site_scale"):
        if stem == "source_footprint_residence":
            scale_zh = (
                f"本圖採用跨站共用色階（各站最大值中的最大者為 {shared_vmax:.3g} % 停留時數／km²），"
                "各站顏色深淺可直接比較"
            )
        else:
            scale_zh = (
                f"本圖採用本站自訂色階（本站最大值為 {site_vmax:.3g} % 停留時數／km²），"
                "只適合看本站內部的分布形狀；跨站比較請看共用色階版本"
            )
        scale_vmax = shared_vmax if stem == "source_footprint_residence" else site_vmax
        lo_c, hi_c = _residence_color_limits(scale_vmax)
        body = (
            "停留時數密度圖，呈現回溯過程中這一站全部粒子逐小時所在位置的累計分布。"
            "停留時數的定義：每顆粒子在回溯的每個整點時刻，只要仍在漂流（尚未穿越開放邊界、擱淺、沉底或因其他原因"
            f"終止），就在它所在的網格記 1 個粒子小時；{arrival_zh}；粒子終止之後凍結的位置不再計入。"
            "各網格的粒子小時數除以本站粒子小時總數，再除以網格面積（km²），單位為「% 停留時數／km²」，"
            "全部網格對面積積分為 100%。"
            f"網格為經緯度規則網格，解析度 {res:.2f}°（約 {km:.1f} km），面積依緯度以 cos φ 計算。"
            f"色階為對數色階，上限為最大值向上取整到 10 的整數次方（{_pow10_text(hi_c)}），"
            f"下限為上限的 1/10000（{_pow10_text(lo_c)}），每個 10 的次方標一個刻度，色條與地圖同高；"
            f"低於下限的網格不著色（其數量與所占停留時數比例記錄在 manifest）；{scale_zh}。"
            "圖上的線為高密度區間（Highest Density Region, HDR）50%、75%、90%：依密度由高到低累加網格，"
            f"累計到該比例的停留時數所需的最小面積範圍；本站三個區間的面積依序為 {_hdr_area_text(fp)}。"
            "與舊版邊界核密度圖的差別：舊圖的輸入只有粒子穿越開放邊界的那一個點，這些點全部落在一維的邊界線上，"
            "二維核密度不適用；本圖改用回溯全程逐小時的位置，反映粒子在流場範圍內實際停留的地方。"
            f"{base_zh}"
        )
    elif stem == "lookback_snapshot":
        hours = "、".join(str(int(s["snapshot_hours"])) for s in fp.snapshots)
        sigma = float(fp.snapshots[0]["smoothing_sigma_km"]) if fp.snapshots else 0.0
        min_n = int(fp.snapshots[0]["minimum_active_particle_count"]) if fp.snapshots else 0
        smooth_zh = (
            f"密度先累加到與停留時數密度相同的 {res:.2f}° 網格，再以寬度 {sigma:g} km 的高斯核做輕度平滑；"
            "平滑只在海域內進行（分子分母都乘上海域遮罩，陸地與流場範圍外的格不參與也不會得到密度），"
            "平滑後總質量重新正規化為 1。"
            if sigma > 0
            else f"密度直接累加到與停留時數密度相同的 {res:.2f}° 網格，不做平滑。"
        )
        lo_s, hi_s = _snapshot_color_limits(fp)
        body = (
            f"回溯快照圖，2×2 四個面板依序為回溯第 {hours} 小時仍在漂流的粒子所在位置的條件式來源足跡。"
            f"{smooth_zh}"
            "每個面板只計入該時刻仍在漂流的粒子，密度的分母是該時刻仍在漂流的粒子數，"
            "因此各面板的總和都是 1；已經終止的粒子（含回溯總時間不足該時刻者）不在面板內，"
            "各面板標題列出仍在漂流的比例（分母是本站全部粒子）。"
            f"仍在漂流的粒子少於 {min_n} 顆時該面板不畫密度，並在面板中註明粒子數。"
            "色階為對數色階，四個面板共用同一組上下限與一條和面板同高的色條，單位為「% 仍在漂流粒子／km²」，"
            f"上限為四個面板最大值向上取整到 10 的整數次方（{_pow10_text(hi_s)}），"
            f"下限為上限的 1/10000（{_pow10_text(lo_s)}），低於下限的位置不著色；"
            "線為該面板的高密度區間（HDR）50%、75%、90%，算法與停留時數密度相同。"
            "陸地畫在密度之上，密度不會出現在陸地上。"
            f"{base_zh}"
        )
    elif stem == "boundary_departure_along_arc":
        length_km = float(fp.arc_bins[0]["boundary_length_km"]) if fp.arc_bins else 0.0
        bin_km = float(fp.arc_bins[0]["bin_km"]) if fp.arc_bins else 0.0
        crossed = sum(int(b["crossed_particle_count"]) for b in fp.arc_bins)
        body = (
            "穿越位置沿開放邊界的一維分布圖。橫軸是沿流場範圍開放邊界折線自起點量起的弧長（km），"
            f"起點與前進方向取自邊界折線的頂點順序，邊界全長 {length_km:.0f} km；"
            f"直條為每 {bin_km:g} km 弧長內穿越的粒子占全部穿越粒子的百分比，棕色曲線與右軸為累積百分比；"
            f"淡色底色區分西、南、東、北各邊，虛線標出各邊之間的轉角。本站共 {crossed:,} 顆粒子穿越開放邊界。"
            "橫軸從 0 起算到邊界全長向上取整，左縱軸從 0 起算到最大占比向上取整，右縱軸為 0 到 100%，"
            "三條軸的兩端都標值。"
            "穿越點全部落在一維的邊界線上，因此以一維分布呈現，取代舊版把這些點做二維核密度的圖。"
        )
    else:  # pragma: no cover - FOOTPRINT_FIGURE_STEMS 是唯一呼叫來源
        raise FiguresError(f"未知圖表代號：{stem}")
    return f"{body} {settings_tail}"


# ═══════════════════════════════════════════════════════════════════════════
# 圖說（captions.md）：WP-24b 新增，圖面不畫警語後，文字改由這裡承載
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class _ExperimentCaseDisclosure:
    """WP-37：把「物理案例」與「粒子身分表標籤」分開陳述的兩段中文文字。

    這兩個量語意不同、且刻意分開命名，理由見 ``postprocess.py`` 模組 docstring
    「WP-18」一節與 ``PostprocessAccumulator.__init__`` 對
    ``runtime_physics_case`` 的說明：

    - ``physics_case_zh``：**物理案例**（是否含 Stokes 漂移、擴散方法），取自
      ``manifest.json`` 的 ``config.runtime_physics_case``——這是
      ``native-postprocess`` 呼叫端提供 ``--plan`` 時，原樣複製自計畫檔
      ``physics_case`` 區塊的內容，只有這個欄位才是「這次執行實際採用的物理
      設定」的證據。呼叫端沒有提供 ``--plan`` 時這個欄位是 ``None``，**不代表
      沒有 Stokes**，只代表無法從 manifest 核對，如實寫「未於後處理 manifest
      記錄」，不退回去用粒子身分表標籤假冒。
    - ``identity_table_label_zh``：**粒子身分表標籤**，取自 manifest 頂層的
      ``experiment_case_id``——這是建立粒子身分表時的字串標籤（seed 命名空間
      用），只保證同一批粒子部署使用同一個標籤，**不保證**等於實際執行時的
      物理設定，因此文字明確標示「這是身分表標籤、不是物理設定的證據」。
    - ``bed_no_deposition_zh``：``config.runtime_bed_no_deposition`` 非
      ``None`` 時，這批結果是否啟用「海床不沉積」敏感度案例；``None``（呼叫端
      未提供 ``--plan``）時本欄位也是 ``None``（圖說不列出這一項，而不是猜測
      未啟用）。
    """

    physics_case_zh: str
    identity_table_label_zh: str
    bed_no_deposition_zh: str | None


def _describe_experiment_case(postprocess_manifest: Mapping[str, Any]) -> _ExperimentCaseDisclosure:
    """由 ``postprocess.py`` 輸出的 ``manifest.json`` 建立 :class:`_ExperimentCaseDisclosure`。

    WP-37 修正：舊版 ``_experiment_case_label`` 的 docstring 宣稱回傳「物理案例
    識別碼」，實際讀的卻是 ``experiment_case_id``（粒子身分表標籤），把兩個語意
    不同的量混成一個字串、還掛錯名字——這裡改成分開回傳，呼叫端各自標明用途，
    不再有任何一個字串同時被兩種語意共用。
    """

    config = postprocess_manifest.get("config")
    config = config if isinstance(config, Mapping) else {}

    runtime_physics_case = config.get("runtime_physics_case")
    if isinstance(runtime_physics_case, Mapping):
        case_id = runtime_physics_case.get("experiment_case_id")
        include_stokes = runtime_physics_case.get("include_stokes")
        diffusion_method = runtime_physics_case.get("diffusion_method")
        if include_stokes is True:
            stokes_zh = "含 Stokes 漂移"
        elif include_stokes is False:
            stokes_zh = "不含 Stokes 漂移"
        else:
            stokes_zh = "是否含 Stokes 漂移未記錄"
        physics_case_zh = f"{case_id}（{stokes_zh}"
        if diffusion_method:
            physics_case_zh += f"，擴散方法：{diffusion_method}"
        physics_case_zh += "）"
    else:
        physics_case_zh = "未於後處理 manifest 記錄（`native-postprocess` 未提供 `--plan`）"

    identity_case_id = postprocess_manifest.get("experiment_case_id")
    identity_table_label_zh = (
        f"{identity_case_id}（粒子身分表標籤，僅供 seed 命名空間辨識，不是物理設定的證據）"
        if identity_case_id
        else "未於後處理 manifest 記錄"
    )

    runtime_bed_no_deposition = config.get("runtime_bed_no_deposition")
    bed_no_deposition_zh: str | None = None
    if isinstance(runtime_bed_no_deposition, Mapping):
        enabled = bool(runtime_bed_no_deposition.get("bed_no_deposition", False))
        bed_no_deposition_zh = (
            "已啟用「海床不沉積」敏感度案例" if enabled else "未啟用「海床不沉積」敏感度案例"
        )

    return _ExperimentCaseDisclosure(
        physics_case_zh=physics_case_zh,
        identity_table_label_zh=identity_table_label_zh,
        bed_no_deposition_zh=bed_no_deposition_zh,
    )


def _figure_caption_zh(stem: str, site: SiteData, *, experiment_case: _ExperimentCaseDisclosure) -> str:
    """回傳單張圖（某站 × 某圖種）的繁體中文圖說，全部用直述句。

    每段固定交代三件事（見 WP-24b 任務要求）：這張圖是什麼、資料來源欄位、
    關鍵設定（站點、粒子數、``M``、終止邊界定義、物理案例）。全部用直述句陳述
    已量測的事實與方法定義——量得到的數字直接寫數字，量不到的（例如個別離開
    點分類的實際誤差率）不在此揣測，只寫清楚的方法定義。

    WP-37：``M`` 改讀該站實測的 ``ensemble_member_count``（不再寫死 1），且
    「物理案例」與「粒子身分表標籤」分開陳述（見 :class:`_ExperimentCaseDisclosure`
    docstring），避免讀者把身分表標籤誤當成物理設定的證據。
    """

    total = int(site.summary["total_particle_count"])
    member_count = int(site.summary["ensemble_member_count"])
    member_text_zh, _member_text_en = _ensemble_member_count_text(member_count)
    site_desc = f"{site.site_label_zh}（{site.site_id}）"
    settings_tail = (
        f"{site_desc}，{total:,} 顆粒子，{member_text_zh}，"
        f"終止邊界為 flow domain 開放邊界，物理案例：{experiment_case.physics_case_zh}，"
        f"粒子身分表標籤：{experiment_case.identity_table_label_zh}"
    )
    if experiment_case.bed_no_deposition_zh is not None:
        settings_tail += f"，{experiment_case.bed_no_deposition_zh}"
    settings_tail += "。"
    if stem == "horizontal_overview":
        body = (
            "代表粒子水平軌跡總覽，涵蓋整個 flow domain 範圍，標示 flow domain 開放邊界"
            "（終止邊界）與依 final_status_code 著色的終止標記；五站圖面版式一致，不畫 "
            "local domain 邊界或 local domain 首次離開標記（WP-25 裁決，該事件仍完整"
            "保存於 local_domain_departures.parquet 等資料表，只是不在這張圖呈現）。"
            "資料來源：trajectory_samples_by_site.parquet 的軌跡座標與時間、"
            "departures.parquet 的 final_status_code、"
            "boundary_arc_segments_geometry_by_site 的開放邊界折線幾何。"
        )
    elif stem == "horizontal_local":
        body = (
            "受體附近局部放大圖，座標為經緯度相對代表粒子回溯起點平均位置的偏移量，同樣"
            "只畫 flow domain 開放邊界與終止標記，不畫 local domain 邊界或首次離開標記"
            "（WP-25 裁決，理由同水平軌跡總覽圖）。資料來源與水平軌跡總覽圖相同："
            "trajectory_samples_by_site.parquet、departures.parquet、"
            "boundary_arc_segments_geometry_by_site。"
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
        # WP-29：boundary_arc_segments_geometry_by_site 對每站同時登記終止邊界
        # （flow domain 開放邊界）與 local domain 開放邊界各自的弧段幾何，但只有
        # 終止邊界那一組弧段會有非零的相對來源權重——粒子的終止事件（穿越開放
        # 邊界）一律歸屬到 flow domain 開放邊界（見 postprocess.classify_
        # boundary_segment docstring「候選集合先依 local_equals_flow 決定」一
        # 節），local domain 開放邊界那一組弧段純粹是保留給地圖畫線用的幾何
        # 參照，不對應任何穿越次數。舊圖說籠統寫「本站共 40 段弧段」，把兩組
        # 20 段混在一起講，容易被誤讀成「權重分布在 40 個弧段上」；這裡改成
        # 分別列出終止邊界與 local 邊界各自的段數，精確對應資料實際登記的內容。
        terminal_ids = [
            base_id for base_id in site.arc_segments_by_boundary if _boundary_kind(base_id) == "flow"
        ]
        reference_ids = [
            base_id for base_id in site.arc_segments_by_boundary if _boundary_kind(base_id) != "flow"
        ]
        terminal_count = sum(len(site.arc_segments_by_boundary[base_id]) for base_id in terminal_ids)
        reference_count = sum(len(site.arc_segments_by_boundary[base_id]) for base_id in reference_ids)
        body = (
            f"終止邊界（flow domain 開放邊界）依權重色階著色，共 {terminal_count} 段弧段，"
            "並以長條圖並列比較相對來源權重；相對來源權重＝該弧段成功穿越次數除以全站成功"
            "穿越開放邊界的總次數，只計在終止邊界的弧段上。"
            f"boundary_arc_segments_geometry_by_site 另外登記 local domain 開放邊界的弧段"
            f"幾何（共 {reference_count} 段），純供地圖疊線的地理參照，權重恆為 0、不計入"
            "終止統計；local_equals_flow=True 的測站這條 local 邊界與上面的終止邊界是同一個"
            "幾何（見 boundary_summary_by_site 的 local_equals_flow 欄位），local_equals_"
            "flow=False 的測站則是各自站點獨立的 local domain 邊界。資料來源："
            "boundary_arc_segments_geometry_by_site 的弧段幾何與 departures.parquet 的 "
            "boundary_segment_id、boundary_arc_segment_index。"
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


@dataclass(frozen=True, slots=True)
class _FootprintFigurePlan:
    """WP-38 新圖的繪製計畫：等深線、色階上限與每站實際會畫出的等深線水深。

    圖說在繪圖前建立，因此「會畫哪些等深線」要在繪圖前就用與繪圖相同的判準算好
    （:func:`_isobath_levels_in_extent`），圖說與圖面才會一致。
    """

    isobaths: Mapping[str, Mapping[float, list[np.ndarray]]]
    bathymetry_record: Mapping[str, Any]
    site_isobath_levels: Mapping[str, tuple[float, ...]]
    site_isobath_reason: Mapping[str, str | None]
    shared_vmax: float
    site_vmax: Mapping[str, float]
    isobath_levels_requested: tuple[float, ...] = ISOBATH_LEVELS_DEFAULT


def _isobath_levels_in_extent(
    site_isobaths: Mapping[float, list[np.ndarray]],
    extent: tuple[float, float, float, float],
    levels: Sequence[float] = ISOBATH_LEVELS_DEFAULT,
) -> tuple[float, ...]:
    return tuple(
        depth
        for depth in sorted(site_isobaths)
        if any(abs(depth - v) < 1e-9 for v in levels)
        and any(_inside_runs(seg, extent) for seg in site_isobaths[depth])
    )


def _plan_footprint_figures(
    inputs: NativeFiguresInputs, config: NativeFiguresConfig
) -> _FootprintFigurePlan | None:
    if inputs.footprint is None:
        return None
    isobaths, record = _load_isobaths(config.bathymetry_path)
    levels: dict[str, tuple[float, ...]] = {}
    reasons: dict[str, str | None] = {}
    for site in inputs.sites:
        fp = inputs.footprint[site.site_id]
        site_isobaths = isobaths.get(site.site_id, {})
        found = tuple(
            sorted(
                {
                    depth
                    for max_intervals in (_RESIDENCE_MAX_INTERVALS, _SNAPSHOT_MAX_INTERVALS)
                    for depth in _isobath_levels_in_extent(
                        site_isobaths, _site_extent(fp, max_intervals=max_intervals), config.isobath_levels_m
                    )
                }
            )
        )
        levels[site.site_id] = found
        if found:
            reasons[site.site_id] = None
        elif record["status"] != "loaded":
            reasons[site.site_id] = str(record["reason"])
        elif not site_isobaths:
            reasons[site.site_id] = "水深檔沒有本站的等深線"
        else:
            reasons[site.site_id] = "本站繪圖範圍內沒有所要求水深的等深線"
    shared_vmax, site_vmax = _footprint_residence_color_scales(inputs.footprint)
    return _FootprintFigurePlan(
        isobaths=isobaths,
        bathymetry_record=record,
        site_isobath_levels=levels,
        site_isobath_reason=reasons,
        shared_vmax=shared_vmax,
        site_vmax=site_vmax,
        isobath_levels_requested=tuple(config.isobath_levels_m),
    )


def _build_captions(inputs: NativeFiguresInputs, plan: _FootprintFigurePlan | None = None) -> dict[str, str]:
    """建立「``<site_id>/<stem>.png`` → 圖說」對照，供 ``captions.md`` 與
    manifest 的 ``output_png_sha256.<key>.caption`` 共用同一份文字（見模組
    docstring「WP-24b」一節：文字只寫一次，圖說與 manifest 互相核對用同一個
    來源，不會出現兩處文字不一致的情況）。``plan`` 不為 ``None`` 時另外建立
    WP-38 三種新圖的圖說。"""

    experiment_case = _describe_experiment_case(inputs.postprocess_manifest)
    captions: dict[str, str] = {}
    for site in inputs.sites:
        for stem in FIGURE_STEMS:
            captions[f"{site.site_id}/{stem}.png"] = _figure_caption_zh(
                stem, site, experiment_case=experiment_case
            )
        if plan is not None and inputs.footprint is not None:
            for stem in FOOTPRINT_FIGURE_STEMS:
                captions[f"{site.site_id}/{stem}.png"] = _footprint_caption_zh(
                    stem,
                    site,
                    inputs.footprint[site.site_id],
                    experiment_case=experiment_case,
                    footprint_manifest=inputs.footprint_manifest,
                    isobath_levels=plan.site_isobath_levels[site.site_id],
                    isobath_reason=plan.site_isobath_reason[site.site_id],
                    shared_vmax=plan.shared_vmax,
                    site_vmax=plan.site_vmax[site.site_id],
                    isobath_requested=plan.isobath_levels_requested,
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
        stems = FIGURE_STEMS
        if inputs.footprint is not None:
            stems = (*FIGURE_STEMS, *FOOTPRINT_FIGURE_STEMS)
        for stem in stems:
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

**WP-25：輸出格式與跨站色階**——每張圖同時輸出 `.png`（{dpi} dpi）與同名
`.svg`（向量），對齊相鄰專案 `OCM-SVD-Analysis` 的正式成圖規範（見
`docs/results/17_native_figures_formal_output_rules.md`）；`source_footprint_
kde`、`boundary_arc_segments`、`boundary_azimuth_rose`、`settling_velocity_
breakdown` 四張圖改用同一批被選取站點共同的色階上下限與刻度，讓站間可直接
比較，超過上限的值以最高色顯示、不裁切資料，實際狀態記錄於 `manifest.json`
的 `color_scales` 區塊（`over_limit_count`／`over_limit_fraction`／
`display_saturation`）。

## 科學定義與範圍

- **系集成員數（M）**：{ensemble_member_summary}；每個系集成員皆含隨機擴散，
  差異只來自隨機數種子。M 是後處理實測值（`boundary_summary_by_site.
  ensemble_member_count`，相異 `member_id` 數量），不是本模組寫死的常數
  （WP-37）——同一套管線可能產出 `M=1` 的確定性試驗，也可能產出 `M>1` 的
  隨機系集。
- 「條件式來源足跡」與「相對來源權重」的定義：給定站點、受體、沉降速度與到達
  時刻等情境設定後，代表粒子穿越各方位扇區／各邊界弧段的相對頻率，是描述這批
  M 個系集成員在既定設定下分布的統計量。來源機率與因果歸因是另一層科學推論，
  需要另外建立先驗、概似與觀測驗證，本批圖表不提供這一層推論。
- 圖上的終止標記位置是回溯積分停止處的座標，對應 `final_status_code` 七種
  互斥狀態之一（成功穿越 flow domain 開放邊界、達回溯上限、擱淺、沉底、流場
  缺值、數值失敗、`pre_window_deposition`）；停止處是積分終止的座標，不是另外
  查證過的實際污染來源。
- 海岸與開放邊界幾何目前的版本狀態是工程生成版：由既有海岸資料與弧段幾何直接
  取得，尚未經研究團隊另立正式核准發布流程。

## 站點

{site_list}

## WP-23／WP-25：flow domain 終止邊界與 local domain 首次離開（資料仍在，圖上已不畫）

研究團隊裁決（`AGENTS.md`、`docs/development/05_decisions_and_risks.md`
D008）：**終止一律以 flow domain 開放邊界為準；離開站點 local domain 是非終止
事件，只記錄、粒子繼續回溯**。`departures.parquet` 的 `crossed_interest_region_
boundary`／`final_status_code` 因此一律是 flow domain 語意，五站可直接互相
比較；`local_domain_departures.parquet`、`local_domain_first_exit_summary_by_
site` 與 `departures.parquet` 的 `also_local_domain_first_exit` 欄位完整保存
每顆粒子「首次離開自己站點 local domain」的位置與統計，不受下一段圖面裁決
影響。

**WP-25 裁決**：`horizontal_overview.png`／`horizontal_local.png` 不再畫 local
domain 邊界或 local domain 首次離開標記，讓貢寮／龜山島（`local_equals_
flow=False`）與新竹／南灣／連江（`local_equals_flow=True`）五站的圖面版式
完全一致——圖上只保留 **flow domain 開放邊界**（藍色虛線，真正的終止邊界，
`terminal_counts.png` 等圖的「穿越開放邊界（正常結束）」指的就是這裡）與依
`final_status_code` 著色的終止標記。若需要 local domain 首次離開的空間分布，
請直接查詢 `local_domain_departures.parquet`（僅 `local_equals_flow=False`
的站有列）或 `local_domain_first_exit_summary_by_site` 的
`fraction_local_first_exit`（各站皆有一列彙總統計）。

## 每站八張圖

各站資料夾（`<site_id>/`）內固定輸出以下八張 PNG。`--figure-text full`（預設）
在圖面上同時畫出標題、副標、圖例與座標軸標籤；`--figure-text none` 只畫座標軸
與圖例，不畫標題與副標，供正式報告排版——每張圖對應的文字改由 `captions.md`
承載（逐圖一段，內容與下列條列一致，並與 `manifest.json` 的
`output_png_sha256.<site_id>/<stem>.png.caption` 完全相同）：

1. `horizontal_overview.png`：代表粒子水平軌跡總覽（經緯度顯示，涵蓋整個 flow
   domain 範圍），**只畫 flow domain 開放邊界**（見上一節；WP-25 起不再畫
   local domain 邊界或首次離開標記，五站版式一致）。資料來源：
   `trajectory_samples_by_site`（軌跡座標、狀態）、`departures.parquet`
   （`final_status_code`）、`boundary_arc_segments_geometry_by_site`
   （開放邊界折線）。綠點＝回溯起點（受體到達位置）；終點依
   `final_status_code` 用不同顏色與標記形狀表示；線色依代表粒子起始深度序位
   （淺→深）。
2. `horizontal_local.png`：受體附近局部放大，座標為經緯度相對代表粒子回溯起點
   平均位置的偏移量，**同樣只畫 flow domain 開放邊界與終止標記**（WP-25 起不
   再疊 local domain 邊界或首次離開標記）。資料來源同上。
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

{footprint_readme}## 已知限制與方法揭露

- **系集成員數（M）**：{ensemble_member_summary}（見上方「科學定義與範圍」）；
  本批圖表不論 M 為何，一律把同一情境下的全部系集成員直接併入同一批統計，
  不呈現、也不能呈現系集內部的離散度或不確定性區間。
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


def _ensemble_member_summary_zh(inputs: NativeFiguresInputs) -> str:
    """回傳 README 用的系集成員數摘要文字（WP-37）。

    正常情況下全部被選取站點的 ``ensemble_member_count`` 應該相同（後處理
    ``_ensemble_member_count_by_site`` 已經 fail-closed 檢核過一致性），這裡
    直接寫成單一 ``M=n``；萬一讀到不一致的舊資料（例如手動拼接的後處理輸出），
    仍誠實逐站列出，不假裝一致或悄悄取其中一個值。
    """

    member_counts = {s.site_id: int(s.summary["ensemble_member_count"]) for s in inputs.sites}
    distinct_counts = sorted(set(member_counts.values()))
    if len(distinct_counts) == 1:
        return f"M={distinct_counts[0]}"
    detail = "、".join(f"{site_id}: M={count}" for site_id, count in sorted(member_counts.items()))
    return f"各站不一致（{detail}）"


def _footprint_readme_section(inputs: NativeFiguresInputs, plan: _FootprintFigurePlan | None) -> str:
    """WP-38 新圖的 README 段落；後處理目錄沒有足跡產品時回傳空字串。"""

    if plan is None or inputs.footprint is None:
        return ""
    bathy = plan.bathymetry_record
    bathy_line = (
        "等深線取自 `--bathymetry` 指定的 GeoJSON（由 `scripts/export_isobaths.py` 依 SCHISM 計算網格的節點"
        "水深匯出）。"
        if bathy.get("status") == "loaded"
        else f"本批圖未繪等深線：{bathy.get('reason')}。"
    )
    return (
        "## WP-38：停留時數密度、回溯快照與邊界弧長分布\n\n"
        "後處理目錄含足跡產品（`native-postprocess --with-footprint`）時，每站另外輸出四張圖：\n\n"
        "1. `source_footprint_residence.png`：停留時數密度（跨站共用對數色階）。停留時數＝回溯過程中逐小時、"
        "仍在漂流（尚未終止）的粒子所在網格的粒子小時數，終止後凍結的位置不計入；色階單位 % 停留時數／km²，"
        "上限為最大值向上取整到 10 的整數次方、下限為上限的 1/10000，每個 10 的次方標一個刻度，"
        "低於下限不著色；疊高密度區間（HDR）50%／75%／90%。\n"
        "2. `source_footprint_residence_site_scale.png`：同一份資料，色階上限改用各站自己的最大值，"
        "只適合看單站"
        "內部形狀；跨站比較請看上一張。\n"
        "3. `lookback_snapshot.png`：回溯 72／168／336／720 小時仍在漂流的粒子的條件式來源足跡（2×2 面板，"
        "與停留時數密度同網格的直方圖，可選海域遮罩下的輕度平滑），標題列出仍在漂流的比例；"
        "仍在漂流粒子數低於門檻的面板不畫密度。\n"
        "4. `boundary_departure_along_arc.png`：穿越位置沿開放邊界弧長的一維分布"
        "（直條＋累積曲線），標出各邊與"
        "轉角。\n\n"
        "舊的 `source_footprint_kde.png` 照常輸出，但不再是報告主圖：它的輸入只有粒子穿越開放邊界的那一點，"
        "這些點全部落在一維的邊界線上，二維核密度不適用。\n\n"
        f"{bathy_line}底圖為陸地（灰）、海岸線（黑 0.5 pt）、中灰細等深線"
        "（預設只畫 200 m 實線與 1000 m 虛線，地圖上不標水深數字）、淡紫色粒子釋放範圍、"
        "星號（受體，位置取自正式釋放設定的研究點錨點，圖面不寫站名）與洋紅色虛線（流場範圍開放邊界）。"
        "色條與地圖軸同高，座標軸範圍向外擴到刻度倍數、首末刻度落在框邊。"
        "`manifest.json` 的 `footprint_figures` 區塊記錄各圖的色階上下限、低於下限的網格統計、"
        "色條與地圖軸的上下緣量測值、座標軸範圍與刻度、等深線狀態，以及每站受體位置與來源。\n\n"
    )


def _build_readme(
    inputs: NativeFiguresInputs,
    config: NativeFiguresConfig,
    command: str,
    plan: _FootprintFigurePlan | None = None,
) -> str:
    site_list = "\n".join(f"- `{s.site_id}`（{s.site_label_zh}）" for s in inputs.sites)
    return _README_TEMPLATE.format(
        footprint_readme=_footprint_readme_section(inputs, plan),
        generated_at_utc=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        figure_text=config.figure_text,
        site_list=site_list,
        ensemble_member_summary=_ensemble_member_summary_zh(inputs),
        command=command,
        dpi=config.dpi,
    )


def _receptor_manifest_record(
    fp: FootprintSiteFigureData, land_geometries: Sequence[Polygon | MultiPolygon]
) -> dict[str, Any]:
    star = fp.receptor_lonlat
    return {
        "source": fp.receptor_source,
        "anchor_lonlat": list(fp.receptor_anchor) if fp.receptor_anchor is not None else None,
        "receptor_core_radius_m": fp.receptor_core_radius_m,
        "star_lonlat": list(star) if star is not None else None,
        "postprocess_release_point_mean_lonlat": (
            list(fp.receptor_center) if fp.receptor_center is not None else None
        ),
        "formal_anchor_used": fp.receptor_anchor is not None,
        "release_geometry": _release_geometry_record(fp, land_geometries),
    }


def _build_footprint_figures_manifest(
    inputs: NativeFiguresInputs,
    plan: _FootprintFigurePlan,
    records: Mapping[str, Any],
    land_geometries: Sequence[Polygon | MultiPolygon] = (),
    receptor_config_path: Path | None = None,
) -> dict[str, Any]:
    """``manifest.json`` 的 ``footprint_figures`` 區塊：色階類型與上下限、等深線狀態。"""

    lower = RESIDENCE_COLOR_FLOOR_FRACTION
    return {
        "residence_color_scale": {
            "unit": "% 停留時數／km²",
            "norm": "log",
            "lower_limit_fraction_of_upper_limit": lower,
            "limit_rule": (
                "上限＝大於或等於最大值的 10 的整數次方；下限＝上限的 1/10000；每個 10 的次方一個刻度"
            ),
            "cells_below_lower_limit": "transparent",
            "out_of_limit_handling": "metadata_only_no_visual_clipping",
            "colorbar": {"shape": "rectangular", "extend": "neither"},
            "colormap": _RESIDENCE_CMAP,
            "variants": {
                "source_footprint_residence": {
                    "scale": "shared_across_sites",
                    "raw_max": plan.shared_vmax,
                    "upper_limit": _decade_limits(plan.shared_vmax)[1],
                    "lower_limit": _decade_limits(plan.shared_vmax)[0],
                },
                "source_footprint_residence_site_scale": {
                    "scale": "per_site",
                    "raw_max_by_site": dict(plan.site_vmax),
                    "upper_limit_by_site": {k: _decade_limits(v)[1] for k, v in plan.site_vmax.items()},
                    "lower_limit_by_site": {k: _decade_limits(v)[0] for k, v in plan.site_vmax.items()},
                },
            },
        },
        "receptors": {
            "authority": "正式釋放設定 study_sites[].anchor_lonlat（config.StudySiteConfig）",
            "by_site": {
                site.site_id: _receptor_manifest_record(inputs.footprint[site.site_id], land_geometries)
                for site in inputs.sites
            },
            "config_path": str(receptor_config_path) if receptor_config_path is not None else None,
            "config_sha256": (
                inputs.sources_before["receptor_config"]["sha256"]
                if receptor_config_path is not None and "receptor_config" in inputs.sources_before
                else None
            ),
            "land_geometry_for_sea_check": "海岸線 GeoJSON 的陸地多邊形；點落在任一多邊形內即視為在陸上",
        },
        "axes_rule": "座標軸範圍向外擴到刻度間距（0.1／0.2／0.5°）的整數倍，首末刻度即框邊，四個端點都標值",
        "colorbar_alignment_tolerance_px": COLORBAR_ALIGNMENT_TOLERANCE_PX,
        "snapshot_color_scale": {
            "unit": "% 仍在漂流粒子／km²",
            "norm": "log",
            "lower_limit_fraction_of_upper_limit": lower,
            "scale": "shared_across_the_four_panels_of_each_site",
            "limit_rule": "上限＝四個面板最大值向上取整到 10 的整數次方；下限＝上限的 1/10000",
            "limits_by_site": {
                site.site_id: dict(
                    zip(
                        ("lower_limit", "upper_limit"),
                        _snapshot_color_limits(inputs.footprint[site.site_id]),
                        strict=True,
                    )
                )
                for site in inputs.sites
            },
        },
        "bathymetry": {
            **dict(plan.bathymetry_record),
            "requested_levels_m": list(plan.isobath_levels_requested),
            "style": {
                "color": _ISOBATH_COLOR,
                "linewidth_pt": _ISOBATH_LINEWIDTH,
                "linestyles_by_level_order": list(_ISOBATH_LINESTYLES),
            },
            "by_site": {
                site.site_id: {
                    "status": "drawn" if plan.site_isobath_levels[site.site_id] else "not_drawn",
                    "levels_m": list(plan.site_isobath_levels[site.site_id]),
                    "reason": plan.site_isobath_reason[site.site_id],
                }
                for site in inputs.sites
            },
        },
        "hdr_levels": [0.5, 0.75, 0.9],
        "text_modes": {
            "full": "標題、副標、面板標題、圖例、座標軸標籤全部畫出",
            "none": "不畫標題與副標；面板標題（快照時刻）、圖例與座標軸標籤保留",
        },
        "figures": dict(records),
    }


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
    color_scales = _compute_shared_color_scales(inputs.sites)

    footprint_plan = _plan_footprint_figures(inputs, config)

    output_dir.mkdir(parents=True)
    captions = _build_captions(inputs, footprint_plan)
    png_records: dict[str, dict[str, Any]] = {}
    svg_records: dict[str, dict[str, Any]] = {}
    footprint_figure_records: dict[str, Any] = {}
    for site in inputs.sites:
        site_dir = output_dir / site.site_id
        site_dir.mkdir()
        for stem in FIGURE_STEMS:
            png_path = site_dir / f"{stem}.png"
            svg_path = png_path.with_suffix(".svg")
            _dispatch_plot(
                stem,
                site,
                land_geometries,
                labeler,
                font_resolution.font_properties,
                png_path,
                config.dpi,
                config.figure_text,
                color_scales,
            )
            key = f"{site.site_id}/{stem}.png"
            png_records[key] = {
                "sha256": sha256_file(png_path),
                "size_bytes": png_path.stat().st_size,
                "caption": captions[key],
            }
            svg_key = f"{site.site_id}/{stem}.svg"
            svg_records[svg_key] = {
                "sha256": sha256_file(svg_path),
                "size_bytes": svg_path.stat().st_size,
            }
        if footprint_plan is not None and inputs.footprint is not None:
            fp = inputs.footprint[site.site_id]
            site_isobaths = footprint_plan.isobaths.get(site.site_id, {})
            for stem in FOOTPRINT_FIGURE_STEMS:
                png_path = site_dir / f"{stem}.png"
                svg_path = png_path.with_suffix(".svg")
                if stem == "source_footprint_residence":
                    record = _plot_source_footprint_residence(
                        site,
                        fp,
                        land_geometries,
                        site_isobaths,
                        labeler,
                        font_resolution.font_properties,
                        png_path,
                        config.dpi,
                        config.figure_text,
                        footprint_plan.shared_vmax,
                        scale_kind="shared",
                        isobath_levels=config.isobath_levels_m,
                    )
                elif stem == "source_footprint_residence_site_scale":
                    record = _plot_source_footprint_residence(
                        site,
                        fp,
                        land_geometries,
                        site_isobaths,
                        labeler,
                        font_resolution.font_properties,
                        png_path,
                        config.dpi,
                        config.figure_text,
                        footprint_plan.site_vmax[site.site_id],
                        scale_kind="per_site",
                        isobath_levels=config.isobath_levels_m,
                    )
                elif stem == "lookback_snapshot":
                    record = _plot_lookback_snapshot(
                        site,
                        fp,
                        land_geometries,
                        site_isobaths,
                        labeler,
                        font_resolution.font_properties,
                        png_path,
                        config.dpi,
                        config.figure_text,
                        isobath_levels=config.isobath_levels_m,
                    )
                else:
                    record = _plot_boundary_departure_along_arc(
                        site,
                        fp,
                        labeler,
                        font_resolution.font_properties,
                        png_path,
                        config.dpi,
                        config.figure_text,
                    )
                key = f"{site.site_id}/{stem}.png"
                png_records[key] = {
                    "sha256": sha256_file(png_path),
                    "size_bytes": png_path.stat().st_size,
                    "caption": captions[key],
                }
                svg_records[f"{site.site_id}/{stem}.svg"] = {
                    "sha256": sha256_file(svg_path),
                    "size_bytes": svg_path.stat().st_size,
                }
                footprint_figure_records[key] = record

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
        "output_formats": ["png", "svg"],
        "font": dict(font_resolution.manifest_record),
        "cjk_labels_used": font_resolution.cjk_available,
        "inputs_sha256_before": dict(inputs.sources_before),
        "inputs_sha256_after": sources_after,
        "output_png_sha256": png_records,
        "output_svg_sha256": svg_records,
        "color_scales": _build_color_scale_manifest(inputs.sites, color_scales),
        "postprocess_manifest_experiment_case_id": inputs.postprocess_manifest.get("experiment_case_id"),
        "postprocess_manifest_master_seed": inputs.postprocess_manifest.get("master_seed"),
        "config": {
            "bathymetry_basename": config.bathymetry_path.name
            if config.bathymetry_path is not None
            else None,
            "postprocess_dir_basename": config.postprocess_dir.name,
            "coastline_basename": config.coastline_path.name,
            "max_trajectories_per_site": config.max_trajectories_per_site,
            "dpi": config.dpi,
            "sites_filter": list(config.sites) if config.sites is not None else None,
            "figure_text": config.figure_text,
            "isobath_levels_m": list(config.isobath_levels_m),
            "allow_unanchored_receptor": config.allow_unanchored_receptor,
            "receptor_config_basename": config.receptor_config_path.name
            if config.receptor_config_path is not None
            else None,
        },
    }
    if footprint_plan is not None:
        manifest["footprint_figure_stems"] = list(FOOTPRINT_FIGURE_STEMS)
        manifest["footprint_figures"] = _build_footprint_figures_manifest(
            inputs,
            footprint_plan,
            footprint_figure_records,
            land_geometries,
            config.receptor_config_path,
        )
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")

    captions_path = output_dir / "captions.md"
    captions_text = _render_captions_markdown(
        inputs,
        captions,
        generated_at_utc=generated_at_utc.strftime("%Y-%m-%d %H:%M UTC"),
        figure_text=config.figure_text,
    )
    with captions_path.open("x", encoding="utf-8") as stream:
        stream.write(captions_text)

    readme_path = output_dir / "README.md"
    readme_text = _build_readme(inputs, config, command, footprint_plan)
    with readme_path.open("x", encoding="utf-8") as stream:
        stream.write(readme_text)

    return output_dir
