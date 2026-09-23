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
綠點＝回溯起點、紅叉／狀態標記＝停止位置、圖例在下方、底部紅字警語），但資料
來源完全不同，兩份程式碼刻意不共用任何私有函式，避免舊版契約被新用途悄悄改動。

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
分母是同一分組內成功穿越邊界的粒子數），不依賴那些可能缺席的聚合表。唯一
無法從離開紀錄表重建、且本模組必須讀取的**幾何**，是開放邊界折線本身的實際
座標，來自 ``boundary_arc_segments_geometry_by_site``。

本模組因此只要求六個輸入存在：``departures.parquet``、
``trajectory_samples_by_site.parquet``、``kde_by_site.parquet``、
``boundary_summary_by_site``、``boundary_arc_segments_geometry_by_site``
（兩者皆可以是 ``.parquet`` 或 ``.csv``，優先讀 ``.parquet``）、
``manifest.json``（``postprocess.py`` 自己輸出的執行紀錄，供讀取方位扇區數、
弧段數等設定）。缺少任一者一律 fail-closed，不静默略過。

經緯度還原（無 domain manifest 輸入下的投影復原）
--------------------------------------------------
本模組命令列只有 ``--postprocess-dir`` 與 ``--coastline``，不讀 domain／open
manifest（避免疊床架屋，也避免正式圖表管線多一份可能與 postprocess 輸出的站點
不同步的幾何來源）。但代表軌跡樣本表只保存公尺制 ``x_m``／``y_m``／``z_m``，
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
稱為「條件式來源足跡」；不得使用「機率」「probability of origin」等字樣。
每張圖與 README 都固定附上 M=1（每情境僅一個系集成員）、幾何為工程生成、停止
位置不等於污染來源等警語。

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


def _disclaimer_text(t: _Labeler) -> str:
    """回傳每張圖與 README 共用的固定警語（中英文擇一）。"""

    return t.t(
        "警語：M=1（每情境僅一個系集成員，含隨機擴散）；邊界與足跡統計是「條件式來源足跡」"
        "與「相對來源權重」，不得解讀為絕對來源機率或因果歸因；停止位置是追蹤停止處，不是已知"
        "污染來源；海岸與開放邊界幾何為工程生成，尚未提升為核准發布幾何。",
        "Caution: M=1 (one ensemble member per scenario, includes random diffusion). Boundary "
        "and footprint statistics are a conditional source footprint / relative source weight "
        "only, not an absolute probability of origin or causal attribution. Endpoints mark where "
        "tracking stopped, not a known pollution source. Coastline and boundary geometry are "
        "engineering-generated, not approved for formal release.",
    )


# ── 設定 ─────────────────────────────────────────────────────────────────


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
    ``dpi`` 是輸出 PNG 的解析度。
    """

    postprocess_dir: Path
    coastline_path: Path
    output_dir: Path
    sites: tuple[str, ...] | None = None
    font_path: Path | None = None
    max_trajectories_per_site: int = 24
    dpi: int = 150

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
    ——本模組是通用 native 管線，呼叫端可能提供不同版本或測試用的縮小海岸檔。
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
            "經緯度與公尺欄位可能不是同一個投影的樣本對"
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


@dataclass(frozen=True, slots=True)
class SiteData:
    """單一站點的全部繪圖輸入，已完成投影、篩選與型別轉換。

    ``departures`` 是這一站全部離開紀錄，攤平成欄名到 NumPy 陣列的 dict（保留
    ``departures.parquet`` 原始列順序），供各圖表獨立重新聚合方位扇區、弧段、
    沉降速度等統計，不依賴可能缺席的聚合表（見模組 docstring）。``kde`` 為
    ``None`` 表示 ``kde_by_site`` 沒有這一站的列（理論上不應發生，
    ``PostprocessAccumulator`` 對每個處理過的站一定會有一列，但仍保守處理避免
    對不存在的鍵盲目索引）。
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


def _column_to_numpy(table: pa.Table, name: str) -> np.ndarray:
    """把一欄轉成 NumPy 陣列；字串／可能含 null 的欄位改用逐元素 Python list 轉換，
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
) -> tuple[TrajectorySample, ...]:
    """建立一站的代表軌跡樣本：換算年齡與經緯度、依 ``particle_id`` 取固定子集、
    再依起始深度排序賦予色階序位（見 :class:`TrajectorySample` docstring）。

    ``departure_lookup`` 是 ``row_particle_index -> (receptor_id, final_status_code,
    final_status_description_zh)`` 的對照，來自同一批離開紀錄；若某個代表粒子的
    ``row_particle_index`` 查不到對應離開紀錄，代表兩張表不是同一次後處理輸出，
    直接 fail-closed，不用任何預設值頂替終止狀態。
    """

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
                "可能不是同一次後處理輸出"
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

    只讀取模組 docstring 列出的六個必要輸入與呼叫端的海岸 GeoJSON；不讀取
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

        site_trajectory_table = _filter_site(trajectory_table, site_id)
        trajectories = _build_trajectory_samples(
            site_trajectory_table,
            departure_lookup_by_site.get(site_id, {}),
            site_id=site_id,
            max_count=config.max_trajectories_per_site,
            projection=projection,
        )

        summary_row = _build_summary_row(summary_table, site_id)
        total_particle_count = int(summary_row["total_particle_count"])
        if total_particle_count != departure_arrays["final_status_code"].size:
            raise FiguresError(
                f"{site_id} 站 boundary_summary_by_site 的 total_particle_count="
                f"{total_particle_count} 與 departures.parquet 實際列數"
                f"{departure_arrays['final_status_code'].size} 不一致"
            )

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
            )
        )

    source_paths: dict[str, Path] = {
        "manifest.json": manifest_path,
        departures_path.name: departures_path,
        trajectory_path.name: trajectory_path,
        kde_path.name: kde_path,
        summary_path.name: summary_path,
        arc_path.name: arc_path,
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


def _add_two_line_title(fig, t: _Labeler, font: Any, title: str, subtitle: str) -> None:
    fig.suptitle(title, fontproperties=font, fontsize=17, y=0.985)
    fig.text(0.5, 0.945, subtitle, ha="center", va="top", fontproperties=font, fontsize=10)


def _add_disclaimer(fig, t: _Labeler, font: Any) -> None:
    fig.text(
        0.5,
        0.012,
        _disclaimer_text(t),
        ha="center",
        va="bottom",
        fontproperties=font,
        fontsize=7.6,
        color="#b3261e",
        wrap=True,
    )


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
) -> None:
    """代表粒子水平軌跡總覽（整個 flow domain 範圍，經緯度顯示）。"""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 8.5), dpi=150)
    ax.set_facecolor("#eaf3f8")
    _land_lonlat_patches(land_geometries)(ax)
    boundary_lines = _resolve_dashed_open_boundary_lines(site)
    for lon, lat in boundary_lines.values():
        ax.plot(lon, lat, color="#174a70", linewidth=1.4, linestyle="--", zorder=2)

    cmap = plt.get_cmap("viridis")
    for sample in site.trajectories:
        color = cmap(0.0 if sample.rank_total <= 1 else (sample.rank_index - 1) / (sample.rank_total - 1))
        ax.plot(sample.lon_deg, sample.lat_deg, color=color, linewidth=1.0, alpha=0.85, zorder=4)
        ax.plot(
            sample.lon_deg[0], sample.lat_deg[0], marker="o", markersize=5.2, markerfacecolor="#1a9850",
            markeredgecolor="white", markeredgewidth=0.6, linestyle="None", zorder=6,
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
        Line2D([], [], color="#174a70", linestyle="--", linewidth=1.4,
               label=t.t("開放邊界（工程生成幾何）", "Open boundary (engineering geometry)")),
    ]
    for code in _STATUS_ORDER:
        handles.append(
            Line2D(
                [], [], marker=_STATUS_MARKERS.get(code, "X"), color=_STATUS_COLORS.get(code, "#d62728"),
                linestyle="None", markersize=7,
                label=t.t(_STATUS_LABELS_ZH.get(code, str(code)), _STATUS_LABELS_EN.get(code, str(code))),
            )
        )
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.06), ncol=3, prop=font, frameon=True
    )

    _add_two_line_title(
        fig, t, font,
        t.t(f"水平軌跡總覽｜{site.site_label_zh}", f"Horizontal overview | {site.site_id}"),
        _site_subtitle(site, t, extra=t.t("｜線色：代表粒子起始深度序位（淺→深）", "")),
    )
    _add_disclaimer(fig, t, font)
    # 改用 subplots_adjust 明確指定邊界，不用 tight_layout：tight_layout 與
    # ax.set_aspect(..., adjustable="box") 併用時，在某些「經緯度範圍很扁」的
    # 站點（例如本圖含寬闊 flow_open_boundary 外框）會於內部多重繪製階段留下
    # 殘影（已實測，圖例列會疊到一個淡灰色殘影字元），改用固定邊界可穩定避免。
    fig.subplots_adjust(left=0.08, bottom=0.26, right=0.97, top=0.87)
    _save_exclusive(fig, path, dpi=dpi)
    plt.close(fig)


def _plot_horizontal_local(
    site: SiteData, t: _Labeler, font: Any, path: Path, dpi: int
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
    cmap = plt.get_cmap("viridis")
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

    _add_two_line_title(
        fig, t, font,
        t.t(f"水平局部放大｜{site.site_label_zh}", f"Horizontal local view | {site.site_id}"),
        _site_subtitle(site, t),
    )
    _add_disclaimer(fig, t, font)
    fig.subplots_adjust(left=0.02, bottom=0.12, right=0.98, top=0.90)
    _save_exclusive(fig, path, dpi=dpi)
    plt.close(fig)


def _plot_depth_age(site: SiteData, t: _Labeler, font: Any, path: Path, dpi: int) -> None:
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
    )
    _add_disclaimer(fig, t, font)
    fig.subplots_adjust(left=0.02, bottom=0.12, right=0.98, top=0.90)
    _save_exclusive(fig, path, dpi=dpi)
    plt.close(fig)


def _plot_terminal_counts(site: SiteData, t: _Labeler, font: Any, path: Path, dpi: int) -> None:
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
    )
    _add_disclaimer(fig, t, font)
    fig.subplots_adjust(left=0.08, bottom=0.34, right=0.98, top=0.90)
    _save_exclusive(fig, path, dpi=dpi)
    plt.close(fig)


def _plot_boundary_azimuth_rose(site: SiteData, t: _Labeler, font: Any, path: Path, dpi: int) -> None:
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
    _add_disclaimer(fig, t, font)
    fig.subplots_adjust(left=0.03, bottom=0.14, right=0.97, top=0.82)
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

    if base_boundary_segment_id.endswith("_local_open_boundary"):
        return "local"
    if base_boundary_segment_id.endswith("_flow_open_boundary"):
        return "flow"
    return base_boundary_segment_id


def _plot_boundary_arc_segments(site: SiteData, t: _Labeler, font: Any, path: Path, dpi: int) -> None:
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
    )
    _add_disclaimer(fig, t, font)
    fig.subplots_adjust(left=0.06, bottom=0.2, right=0.98, top=0.88)
    _save_exclusive(fig, path, dpi=dpi)
    plt.close(fig)


def _plot_source_footprint_kde(
    site: SiteData,
    land_geometries: Sequence[Polygon | MultiPolygon],
    t: _Labeler,
    font: Any,
    path: Path,
    dpi: int,
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
    for lon, lat in boundary_lines.values():
        ax.plot(lon, lat, color="#174a70", linewidth=1.4, linestyle="--", zorder=6)

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
    )
    _add_disclaimer(fig, t, font)
    fig.subplots_adjust(left=0.02, bottom=0.14, right=0.98, top=0.88)
    _save_exclusive(fig, path, dpi=dpi)
    plt.close(fig)


def _plot_settling_velocity_breakdown(site: SiteData, t: _Labeler, font: Any, path: Path, dpi: int) -> None:
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
    )
    _add_disclaimer(fig, t, font)
    fig.subplots_adjust(left=0.02, bottom=0.14, right=0.98, top=0.88)
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
) -> None:
    if stem == "horizontal_overview":
        _plot_horizontal_overview(site, land_geometries, t, font, path, dpi)
    elif stem == "horizontal_local":
        _plot_horizontal_local(site, t, font, path, dpi)
    elif stem == "depth_age":
        _plot_depth_age(site, t, font, path, dpi)
    elif stem == "terminal_counts":
        _plot_terminal_counts(site, t, font, path, dpi)
    elif stem == "boundary_azimuth_rose":
        _plot_boundary_azimuth_rose(site, t, font, path, dpi)
    elif stem == "boundary_arc_segments":
        _plot_boundary_arc_segments(site, t, font, path, dpi)
    elif stem == "source_footprint_kde":
        _plot_source_footprint_kde(site, land_geometries, t, font, path, dpi)
    elif stem == "settling_velocity_breakdown":
        _plot_settling_velocity_breakdown(site, t, font, path, dpi)
    else:  # pragma: no cover - FIGURE_STEMS 是唯一呼叫來源，這裡只是防禦
        raise FiguresError(f"未知圖表代號：{stem}")


# ═══════════════════════════════════════════════════════════════════════════
# README 與整體 manifest
# ═══════════════════════════════════════════════════════════════════════════

_README_TEMPLATE = """\
# WP-21：native 正式成果圖表（{generated_at_utc}）

本目錄由 `native-figures` 命令列由既有 native 後處理輸出（`departures.parquet`
等，見下方「輸入來源」）獨立繪製，**不重跑任何積分、不讀取 forcing、不改寫任何
輸入檔**；輸入檔前後 SHA-256 已核對一致（見 `manifest.json`）。

## 重要警語（請先讀）

{disclaimer}

## 站點

{site_list}

## 每站八張圖

各站資料夾（`<site_id>/`）內固定輸出以下八張 PNG，圖面標題、副標與底部紅字警語
與本 README 一致：

1. `horizontal_overview.png`：代表粒子水平軌跡總覽（經緯度顯示，涵蓋整個 flow
   domain 範圍）。資料來源：`trajectory_samples_by_site`（軌跡座標、狀態）、
   `departures.parquet`（起訖狀態對照）、`boundary_arc_segments_geometry_by_site`
   （開放邊界折線）。綠點＝回溯起點（受體到達位置）；終點依 `final_status_code`
   用不同顏色與標記形狀表示；線色依代表粒子起始深度序位（淺→深）。
2. `horizontal_local.png`：受體附近局部放大，座標為經緯度相對代表粒子回溯起點
   平均位置的偏移量。資料來源同上。
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
  docstring），不是精確幾何交點，個別離開點的分類可能有誤差。
- 海岸與開放邊界幾何為工程生成，尚未提升為核准發布幾何；`horizontal_overview`
  等圖以「開放邊界折線（由弧段起訖點重建）」近似完整 domain 外框，本管線不讀
  domain／open manifest，因此不畫閉合陸域邊界，只畫開放水域邊界。
- 深度—時齡曲線的「序位」只是同一批代表粒子按起始深度排序後的名次，同一序位
  在不同站、不同受體之間**不對應同一水層**。

## 輸入來源

見 `manifest.json` 的 `inputs_sha256_before`／`inputs_sha256_after`（相同代表本
批繪圖沒有改動任何輸入檔）與 `output_png_sha256`（每張 PNG 的 SHA-256）。

## 重建指令

```
{command}
```
"""


def _build_readme(
    inputs: NativeFiguresInputs, config: NativeFiguresConfig, command: str, disclaimer_zh: str
) -> str:
    site_list = "\n".join(f"- `{s.site_id}`（{s.site_label_zh}）" for s in inputs.sites)
    return _README_TEMPLATE.format(
        generated_at_utc=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        disclaimer=disclaimer_zh,
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
    png_records: dict[str, dict[str, Any]] = {}
    for site in inputs.sites:
        site_dir = output_dir / site.site_id
        site_dir.mkdir()
        for stem in FIGURE_STEMS:
            png_path = site_dir / f"{stem}.png"
            _dispatch_plot(
                stem, site, land_geometries, labeler, font_resolution.font_properties, png_path, config.dpi
            )
            png_records[f"{site.site_id}/{stem}.png"] = {
                "sha256": sha256_file(png_path),
                "size_bytes": png_path.stat().st_size,
            }

    sources_after = _snapshot_sources(inputs.source_paths)
    if sources_after != dict(inputs.sources_before):
        raise FiguresError("輸入檔在繪圖過程中被改動，前後 SHA-256 不一致，fail-closed")

    disclaimer_zh = _disclaimer_text(_Labeler(True))
    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
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
        },
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")

    readme_path = output_dir / "README.md"
    readme_text = _build_readme(inputs, config, command, disclaimer_zh)
    with readme_path.open("x", encoding="utf-8") as stream:
        stream.write(readme_text)

    return output_dir
