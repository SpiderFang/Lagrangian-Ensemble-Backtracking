"""WP-6：把大量 ``run_backward_batch`` 批次成果彙整成可交付的後處理產品。

背景與範圍
----------
``native_runner.backward_driver.run_backward_batch`` 每次呼叫只處理「同一流場
區域、同一到達時刻、同一沉降速度」的一批粒子（20 到 2,000 顆不等），回傳一個
``BackwardDriverResult``。正式全案（五站 × 每站 10 沉降速度 × 20 受體 × 50 到達
時刻 × M 個系集成員）會產生 2,000 批以上的結果，粒子總數在 ``M=1`` 時是
50,000 顆、``M=10`` 時是 500,000 顆；每顆粒子的完整軌跡（最長 30 天回溯、每小時
一筆）有 720 筆以上的樣本，若把全部軌跡一次攤平成陣列，50 萬顆 × 720 小時 ×
多個欄位會達到數十 GB，遠超單一行程可用記憶體。本模組因此把「彙整」設計成
**單一 pass 的串流處理**：呼叫端把 ``BackwardDriverResult`` 一批一批餵進來（例如
直接串接 ``run_backward_batch`` 的產生器），本模組只保留下列三類「大小與粒子
總數同量級、但遠小於軌跡總量」的狀態，從不在記憶體中累積軌跡陣列本身：

1. 每顆粒子一份的小型純量摘要（離開狀態、離開位置、離開時刻等），大小是
   ``O(粒子數)``，不隨軌跡樣本數成長；
2. 每個分組（例如「同一站」）固定筆數的軌跡抽樣暫存區，大小是
   ``O(分組數 × 抽樣上限)``，不隨已處理粒子數成長；
3. 每個分組成功穿越邊界的離開座標點列（供條件式來源足跡核密度估計使用），
   大小最多是 ``O(粒子數)``，同樣遠小於軌跡總量。

粒子身分由 ``particle_input.ParticleIdentityTable`` 保存，本模組完全信任該表
已通過的內容雜湊與唯一性驗證，只用 ``row_particle_index`` 做整數列位置對回（見
``particle_input.build_baytrace_particle_soa`` docstring：這是刻意設計成不需要
額外 join 鍵的對應方式）。

科學語意（依 AGENTS.md 與工作書）
------------------------------------
- 只有 ``NATIVE_STATUS_LEFT_INTEREST_REGION``（成功穿越關注區域邊界）的離開點
  才會進入條件式來源足跡（式 (11)）的核密度估計；其餘六種終止原因（達回溯
  上限、擱淺、沉底、流場缺值、數值失敗、固定日曆回溯窗之外的沉底
  ``pre_window_deposition``，見 WP-17）是不同的資料狀態，一律分開列出人數與
  比例，**不與穿越邊界的分母混在一起，也不以零值頂替**。``pre_window_deposition``
  依 ``docs/operations/14_input_derivation_and_release_contract.md`` 第 3.1 節，
  保留在固定的情境／成員 outcome 分母（即這裡的 ``total_particle_count``）內單獨
  計數，但不貢獻任何來源足跡分子，語意上與「資料缺口」「數值失敗」平行、但
  成因不同（不是運算失敗，是這顆粒子的沉底時刻本身就在本次固定日曆窗之外）。
- 本模組產出的所有空間密度、機率與高密度區（HDR）一律稱為「條件式來源足跡」
  或「相對來源權重」，不得解讀為絕對來源機率或因果歸因（AGENTS.md、
  ``aggregation.binned_gaussian_kde_2d``／``conditional_kde_2d`` docstring 已有
  相同宣告，這裡沿用同一用語，不另創新詞）。
- 所有物理量（離開位置、頻寬、格網）都以公尺制運算座標為準；經緯度欄位只在
  提供 ``lonlat_projector`` 時才會被填入，且只作為資料交換／圖面顯示用途，不
  參與任何統計或核密度計算。

既有模組重用（依工作要求「先搜尋並優先重用」逐一交代）
--------------------------------------------------------
- **核密度估計（式 (11)）**：直接呼叫既有 ``aggregation.conditional_kde_2d``，
  完全不重寫核密度數學。這個既有函式已經是「給定原始交點座標＋公尺制格網
  邊界」直接算出平滑機率密度與 HDR 遮罩的實作，形式上就是式 (11)
  ``f̂(x,y) = 1/(n·hx·hy) Σ K((x−xi)/hx, (y−yi)/hy)`` 的二維乘積高斯核，只是
  頻寬透過 ``scipy.stats.gaussian_kde`` 的共變異數縮放實現（見下方「頻寬選擇」
  一節），不是重新發明的公式。本模組只新增一層薄轉接：（a）依每個分組自己的
  離開點座標範圍推導公尺制格網邊界（``_derive_grid_edges``），（b）在樣本數
  過少或所有點共線（``conditional_kde_2d`` 會對這兩種情況丟出
  ``ValueError``）時，回傳明確的「不可估計」狀態而不是讓例外往外傳，因為批次
  後處理不應該因為某一個分組樣本太少就讓整個 run 失敗。
  另外考慮過 ``aggregation.binned_gaussian_kde_2d``（格網計數＋高斯濾波的近似
  版本，``report_kde_statistics.build_kde_sensitivity_product`` 用的就是它）與
  該報告層 wrapper 本身；但兩者都要求呼叫端先建立 ``AggregateSpec``／
  ``ReportSpec``——這兩個規格檔描述的是既有「事件式」管線（``boundaries.py``、
  ``BoundaryGeometryBundle``）用 JSON 定義的每站固定網格與三段頻寬敏感度，
  屬於另一條尚未涵蓋 native_runner 資料的管線；若為了套用這層 wrapper 而憑空
  捏造一份 ``AggregateSpec``／``ReportSpec`` JSON，只是形式上「重用了程式碼」，
  實質上是在編造一份沒有科學依據的規格，因此本模組選擇直接重用真正的數值
  核心（``conditional_kde_2d``），不重用需要憑空規格才能啟動的報告層 wrapper。
- **季節分類**：直接重用 ``arrival_times.SEASON_BY_MONTH``（既有
  ``DJF``／``MAM``／``JJA``／``SON`` 四季對照表），不重新定義月份到季節的
  對應。
- **程式部署指紋（Git commit／dirty／依賴版本）**：直接重用
  ``provenance.collect_code_provenance``；**檔案檢查碼（checksum）**：直接重用
  ``outputs.sha256_file``。兩者都是本專案既有、已測試過的通用工具，本模組不
  重寫雜湊或 Git 邏輯。
- **粒子終止狀態碼與中文說明**：直接重用
  ``backward_driver.NATIVE_STATUS_DESCRIPTIONS`` 與六個 ``NATIVE_STATUS_*``
  常數，不另建一套字串或整數狀態碼。

頻寬選擇（``conditional_kde_2d`` 的 ``bandwidth="scott"``）
--------------------------------------------------------------
本模組固定要求 ``conditional_kde_2d`` 使用 Scott 法則（``scipy.stats.gaussian_kde``
的預設頻寬選擇）：頻寬係數 ``factor = n**(-1/(d+4))``（``d=2`` 是空間維度），再由
``factor`` 縮放樣本點的共變異數矩陣得到實際核寬度。這是文獻上最常見、也是
``scipy`` 內建的資料驅動頻寬公式，優點是完全由樣本點的離散程度與樣本數自動
決定、不需要人工試誤；代價是假設離開點分布近似橢圓對稱的單峰或平緩多峰分布，
若真實分布是尖銳的多群聚（multi-modal、群間距遠大於群內離散度），Scott 頻寬
可能過度平滑，掩蓋群聚間的細節。工作書要求的固定敏感度分析（多個頻寬對照）
屬於報告層 ``report_kde_statistics`` 的既有契約，本模組的postprocess輸出的
``bandwidth_factor`` 已足以讓下游報告層另外做頻寬敏感度比較，不在 WP-6 這一層
重複建立。

不做的事（誠實揭露）
----------------------
1. 本模組不知道也不假設任何 ``(lon, lat) ↔ (x, y)`` 的投影方式；``lonlat_projector``
   是選擇性依賴注入（風格比照 ``particle_input.InitialPositionResolver``），
   不提供時經緯度欄位一律是 Parquet 的 ``null``（不是 ``0`` 或任何頂替值）。
2. 邊界穿越通量統計只提供「各站」「各沉降速度」「各受體深度（以 z0_m 站內
   四分位數為代理）」「各季節」四個**各自獨立**的邊際分組，不提供四維聯合
   交叉分表；理由是 ``M=1`` 正式設計下（50,000 顆粒子）若做四維聯合交叉，
   每格分母可能只有數十顆粒子，比例估計噪聲很大且容易誤導，一維邊際分組的
   分母遠大於聯合分表，統計上更穩健。若下游報告確實需要聯合交叉，應該在
   report 層另外從本模組輸出的離開紀錄表（``departures.parquet``，含全部原始
   欄位）重新聚合，而不是由 WP-6 預先算好一份可能誤導的聯合表。
3. 「各受體深度」沒有獨立的垂向 receptor 分類欄位可用——``ParticleIdentityTable``
   只保留連續的初始深度 ``z0_m``（公尺，海面為零、向上為正），沒有保留
   ``receptors.RandomVerticalDraw`` 的 ``vertical_id``。本模組以「站內
   ``z0_m`` 四分位數」作為深度分組代理（``depth_quartile`` 欄位，0 是最靠近
   海面的一組），這只是後處理層的分組代理，不是官方定義的四個垂向抽樣位置；
   如需精確對應，需要上游在身分表中新增 ``vertical_id`` 欄位。
4. 軌跡抽樣使用「以 SHA-256 雜湊為優先序、保留每組雜湊值最小的 K 筆」演算法
   （見 ``_stable_sample_priority``），設計精神比照既有
   ``report_trajectory_identity.representative_priority_digest``（同樣是雜湊
   最小優先序、固定容量選樣），但沒有直接呼叫該函式——它要求
   ``season``／``tide_class`` 必須精確屬於報告層定義的核心分層
   （``CORE_SEASONS``／``CORE_TIDE_CLASSES``，後者是潮汐 spring/neap 分類），
   而 native_runner 目前沒有潮汐分類的產出；勉強塞一個假的 ``tide_class`` 只是
   為了呼叫既有函式而捏造不存在的資料，因此本模組改為獨立撰寫一個更簡單、
   不需要潮汐分層的同精神版本，並在此明確揭露這個設計選擇的取捨。
"""

from __future__ import annotations

import hashlib
import heapq
import json
import resource
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq
from shapely.geometry import Point

from .. import aggregation as aggregation_module
from ..arrival_times import SEASON_BY_MONTH
from ..boundaries import BoundaryGeometry
from ..outputs import sha256_file
from ..provenance import CodeProvenance, collect_code_provenance
from .backward_driver import (
    NATIVE_STATUS_ACTIVE,
    NATIVE_STATUS_DEPOSITED,
    NATIVE_STATUS_DESCRIPTIONS,
    NATIVE_STATUS_FORCING_MISSING,
    NATIVE_STATUS_LEFT_INTEREST_REGION,
    NATIVE_STATUS_MAX_BACKTRACK,
    NATIVE_STATUS_NUMERICAL_FAILURE,
    NATIVE_STATUS_PRE_WINDOW_DEPOSITION,
    NATIVE_STATUS_STRANDED,
    BackwardDriverResult,
)
from .particle_input import ParticleIdentityTable

__all__ = [
    "PostprocessError",
    "NON_CROSSING_STATUS_CATEGORY_ZH",
    "DEPARTURE_TABLE_SCHEMA",
    "LonLatProjector",
    "PostprocessConfig",
    "KDE_STATUS_AVAILABLE",
    "KDE_STATUS_INSUFFICIENT_POINTS",
    "KDE_STATUS_COLLINEAR_POINTS",
    "SourceFootprintKDEProduct",
    "build_source_footprint_kde",
    "compute_batch_result_fingerprint",
    "BatchRecord",
    "PostprocessManifest",
    "PostprocessAccumulator",
    "classify_boundary_segment",
]


class PostprocessError(RuntimeError):
    """WP-6 後處理階段的一般性資料契約或一致性錯誤，一律 fail-closed。"""


# ── 狀態碼詞彙：直接重用 backward_driver 既有常數，不另建一套 ───────────────

_NON_CROSSING_STATUS_CODES: tuple[int, ...] = (
    NATIVE_STATUS_MAX_BACKTRACK,
    NATIVE_STATUS_STRANDED,
    NATIVE_STATUS_DEPOSITED,
    NATIVE_STATUS_FORCING_MISSING,
    NATIVE_STATUS_NUMERICAL_FAILURE,
    NATIVE_STATUS_PRE_WINDOW_DEPOSITION,
)

NON_CROSSING_STATUS_CATEGORY_ZH: Mapping[int, str] = MappingProxyType(
    {
        NATIVE_STATUS_MAX_BACKTRACK: "達回溯上限",
        NATIVE_STATUS_STRANDED: "擱淺",
        NATIVE_STATUS_DEPOSITED: "沉底",
        NATIVE_STATUS_FORCING_MISSING: "流場缺值",
        NATIVE_STATUS_NUMERICAL_FAILURE: "數值失敗",
        NATIVE_STATUS_PRE_WINDOW_DEPOSITION: "固定日曆回溯窗之外的沉底（pre_window_deposition）",
    }
)
"""六種「未穿越關注區域邊界」終止原因的中文分類標籤，鍵為
``backward_driver.NATIVE_STATUS_*`` 狀態碼，供邊界通量統計表的欄位命名與人類
可讀報告使用；刻意不包含 ``NATIVE_STATUS_ACTIVE``（正式輸出不應出現尚在追蹤中
的粒子，出現即代表上游 run 未完整結束，見 ``PostprocessAccumulator.add_batch``
的檢查）與 ``NATIVE_STATUS_LEFT_INTEREST_REGION``（成功穿越，不屬於「未穿越」
分類）。``NATIVE_STATUS_PRE_WINDOW_DEPOSITION``（WP-17 新增）與其餘五種狀態不同
的地方是：它不代表運算過程中發生了什麼（不是資料缺口、也不是數值失敗），而是
這顆粒子的沉底時刻本身就落在 ``fixed_calendar_window`` 模式的研究窗之外，因此
從一開始就不會進入 forcing 運算（見 ``backward_driver.build_pre_window_deposition_
result``）；把它與其餘五種一起列在「未穿越」分類，只是沿用同一張表的分母與欄位
慣例，不代表把它與資料缺口或數值失敗混為同一種資料品質問題。"""

_FLUX_STATUS_ORDER: tuple[int, ...] = (
    NATIVE_STATUS_LEFT_INTEREST_REGION,
    *_NON_CROSSING_STATUS_CODES,
)
"""邊界通量統計表固定的狀態欄位順序：第一欄永遠是成功穿越，其餘六欄依
``NON_CROSSING_STATUS_CATEGORY_ZH`` 的分類順序排列，讓不同分組輸出的欄位順序
一致，方便下游程式直接依位置讀取，不需要每次都查欄名。"""


# 欄名沿用既有 ``status_{整數碼}`` 慣例（例如 ``count_status_7``）：既有下游／測試
# 已經依這個慣例讀欄，WP-17 只新增 ``NATIVE_STATUS_PRE_WINDOW_DEPOSITION``（碼 7）
# 這一種狀態，不重新命名既有五種欄位。欄位語意一律以 ``NON_CROSSING_STATUS_
# CATEGORY_ZH`` 的中文標籤為準，裸整數只是穩定的欄名尾碼。


# ── 離開紀錄表：固定 Parquet schema，供串流逐批寫入 ─────────────────────────

DEPARTURE_TABLE_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("particle_id", pa.string(), nullable=False),
        pa.field("scenario_id", pa.string(), nullable=False),
        pa.field("member_id", pa.int32(), nullable=False),
        pa.field("study_site_id", pa.string(), nullable=False),
        pa.field("analysis_region_id", pa.string(), nullable=False),
        pa.field("receptor_id", pa.string(), nullable=False),
        pa.field("material_id", pa.string(), nullable=False),
        pa.field("arrival_time_id", pa.string(), nullable=False),
        pa.field("arrival_time_utc_ns", pa.int64(), nullable=False),
        pa.field("settling_velocity_mps", pa.float64(), nullable=False),
        pa.field("season", pa.string(), nullable=False),
        pa.field("depth_quartile", pa.int8(), nullable=False),
        pa.field("final_status_code", pa.int32(), nullable=False),
        pa.field("final_status_description_zh", pa.string(), nullable=False),
        pa.field("crossed_interest_region_boundary", pa.bool_(), nullable=False),
        pa.field("departure_x_m", pa.float64(), nullable=False),
        pa.field("departure_y_m", pa.float64(), nullable=False),
        pa.field("departure_z_m", pa.float64(), nullable=False),
        pa.field("departure_lon_deg", pa.float64(), nullable=True),
        pa.field("departure_lat_deg", pa.float64(), nullable=True),
        pa.field("departure_utc_ns", pa.int64(), nullable=False),
        pa.field("backtrack_duration_seconds", pa.float64(), nullable=False),
        pa.field("row_particle_index", pa.int64(), nullable=False),
        pa.field("boundary_segment_id", pa.string(), nullable=True),
        pa.field("boundary_classification_distance_m", pa.float64(), nullable=True),
    ]
)
"""每顆粒子一列的離開紀錄表 schema。``departure_x_m``／``y_m``／``z_m`` 是公尺制
運算座標（與 ``BackwardDriverResult.final_x_m`` 等欄位同一系統）；
``departure_lon_deg``／``lat_deg`` 只在呼叫端提供 ``lonlat_projector`` 時才有值，
只作資料交換與圖面顯示用途，不參與任何統計或核密度計算（見模組 docstring）。
``depth_quartile`` 是站內 ``z0_m`` 四分位數代理分組（0 最靠近海面），不是官方
垂向 receptor 分類，見模組 docstring 第 3 點揭露。``backtrack_duration_seconds``
是 ``(arrival_time_utc_ns - departure_utc_ns) / 1e9``，恆為非負值（逆向追蹤，
離開時刻必然早於或等於到達時刻）。``boundary_segment_id``／
``boundary_classification_distance_m``（WP-17 新增）只在建構時提供
``boundary_geometries`` 且該列 ``crossed_interest_region_boundary=True`` 時才有值，
見 ``classify_boundary_segment`` 的方法與已知限制說明；其餘情況一律是明確的
``null``（未提供邊界幾何，或這列根本沒有穿越邊界），不得以任何分段代碼頂替。"""


LonLatProjector = Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
"""選擇性依賴注入：給定 ``(study_site_id 陣列, x_m 陣列, y_m 陣列)``，回傳同長度
的 ``(lon_deg 陣列, lat_deg 陣列)``。呼叫端應綁定各站實際使用的公尺制投影反轉換
（例如工作書「每站明示的固定本地區域方位等距投影」）；本模組不假設、也不內建
任何特定投影，風格比照 ``particle_input.InitialPositionResolver`` 的依賴注入
設計。不提供時，輸出表的經緯度欄位一律是明確的 Parquet ``null``。"""


def _describe_status(code: int) -> str:
    """回傳狀態碼的中文說明，未知狀態碼直接 fail-closed（不得靜默給預設字串）。"""

    try:
        return NATIVE_STATUS_DESCRIPTIONS[int(code)]
    except KeyError as exc:
        raise PostprocessError(f"未知的粒子終止狀態碼：{code!r}") from exc


# ── 條件式來源足跡（式 11）：薄轉接層包裝既有 aggregation.conditional_kde_2d ──

KDE_STATUS_AVAILABLE = "available"
KDE_STATUS_INSUFFICIENT_POINTS = "insufficient_points"
KDE_STATUS_COLLINEAR_POINTS = "collinear_points"


@dataclass(frozen=True, slots=True)
class SourceFootprintKDEProduct:
    """單一分組（例如某一站，或某一站×某一沉降速度）的條件式來源足跡估計結果。

    只有 ``status == KDE_STATUS_AVAILABLE`` 時，格網與密度欄位才不是 ``None``；
    樣本太少或全部共線時，只保留 ``raw_point_count`` 讓報告層知道「有多少原始
    穿越點但無法估計」，不得以零密度或均勻密度假造結果（AGENTS.md：缺值不可
    用零值替代，這裡的「無法估計」屬於同一精神的狀態分離）。
    """

    status: str
    raw_point_count: int
    bandwidth_factor: float | None
    x_edges_m: np.ndarray | None
    y_edges_m: np.ndarray | None
    x_centers_m: np.ndarray | None
    y_centers_m: np.ndarray | None
    density_per_m2: np.ndarray | None
    cell_probability: np.ndarray | None
    hdr_levels: tuple[float, ...]
    hdr_masks: Mapping[float, np.ndarray] | None


def build_source_footprint_kde(
    points_xy_m: np.ndarray,
    *,
    grid_cell_count: int = 64,
    padding_fraction: float = 0.15,
    minimum_point_count: int = 30,
    hdr_levels: Sequence[float] = (0.50, 0.75, 0.90),
) -> SourceFootprintKDEProduct:
    """對一組成功穿越關注區域邊界的離開點，建立式 (11) 條件式來源足跡。

    這是本模組唯一的 KDE 入口，直接呼叫既有 ``aggregation.conditional_kde_2d``
    做真正的核密度數學（見模組 docstring「既有模組重用」一節），本函式只負責
    三件既有函式沒有做的事：

    1. 從這組點自己的座標範圍推導公尺制格網邊界（既有函式要求呼叫端先給定
       邊界，不會自動決定範圍）：以點的最小／最大值各外擴
       ``padding_fraction``（預設 15%）的範圍寬度，讓核密度尾端不會被格網邊界
       硬生生截斷太多；若某一軸的點範圍是零（例如全部點 x 座標剛好相同），
       改以 1 公尺當最小外擴寬度，避免產生寬度為零的退化格網。
    2. 在樣本數低於 ``minimum_point_count`` 時，直接回傳「樣本不足」狀態，不
       呼叫既有函式（該函式本身只檢查「至少 3 點」，比報告層需要的最低樣本
       門檻更寬鬆）。
    3. 攔截既有函式對「全部點共線」丟出的 ``ValueError``，轉成明確的
       ``KDE_STATUS_COLLINEAR_POINTS`` 狀態，而不是讓例外中止整個批次後處理
       （不同分組的樣本退化情況不應該互相拖累）。

    Args:
        points_xy_m: ``(n, 2)`` 公尺制離開座標點，每列一顆粒子的 ``(x, y)``。
        grid_cell_count: 格網每軸的格數（正方形格網），預設 64；純粹是報告用
            解析度的取捨，格數越多，KDE 峰值定位越精細，但陣列越大。
        padding_fraction: 格網邊界相對於點範圍的外擴比例。
        minimum_point_count: 低於此樣本數視為不可估計（見上方第 2 點）。
        hdr_levels: 高密度區（HDR）累積機率門檻，預設沿用專案既有慣例
            ``(0.50, 0.75, 0.90)``（``aggregate_spec._HDR_LEVELS`` 與
            ``aggregation`` 兩個既有模組的預設值一致）。

    Returns:
        ``SourceFootprintKDEProduct``。
    """

    points = np.asarray(points_xy_m, dtype=np.float64)
    raw_point_count = int(points.shape[0]) if points.ndim == 2 else 0
    levels = tuple(float(value) for value in hdr_levels)

    if raw_point_count < minimum_point_count:
        return SourceFootprintKDEProduct(
            status=KDE_STATUS_INSUFFICIENT_POINTS,
            raw_point_count=raw_point_count,
            bandwidth_factor=None,
            x_edges_m=None,
            y_edges_m=None,
            x_centers_m=None,
            y_centers_m=None,
            density_per_m2=None,
            cell_probability=None,
            hdr_levels=levels,
            hdr_masks=None,
        )

    x_min, x_max = float(points[:, 0].min()), float(points[:, 0].max())
    y_min, y_max = float(points[:, 1].min()), float(points[:, 1].max())
    x_pad = max((x_max - x_min) * padding_fraction, 1.0)
    y_pad = max((y_max - y_min) * padding_fraction, 1.0)
    x_edges = np.linspace(x_min - x_pad, x_max + x_pad, grid_cell_count + 1)
    y_edges = np.linspace(y_min - y_pad, y_max + y_pad, grid_cell_count + 1)

    try:
        grid = aggregation_module.conditional_kde_2d(
            points,
            x_edges_m=x_edges,
            y_edges_m=y_edges,
            hdr_levels=levels,
            bandwidth="scott",
        )
    except ValueError:
        # 既有函式對「全部點共線」（矩陣秩 < 2）與其他資料退化情況一律丟
        # ValueError；本函式已先過濾樣本數不足，這裡剩下的唯一已知成因是
        # 共線，因此歸類為 KDE_STATUS_COLLINEAR_POINTS，而不是讓例外往外傳。
        return SourceFootprintKDEProduct(
            status=KDE_STATUS_COLLINEAR_POINTS,
            raw_point_count=raw_point_count,
            bandwidth_factor=None,
            x_edges_m=None,
            y_edges_m=None,
            x_centers_m=None,
            y_centers_m=None,
            density_per_m2=None,
            cell_probability=None,
            hdr_levels=levels,
            hdr_masks=None,
        )

    return SourceFootprintKDEProduct(
        status=KDE_STATUS_AVAILABLE,
        raw_point_count=raw_point_count,
        bandwidth_factor=float(grid.bandwidth_factor),
        x_edges_m=x_edges,
        y_edges_m=y_edges,
        x_centers_m=grid.x_centers_m,
        y_centers_m=grid.y_centers_m,
        density_per_m2=grid.density_per_m2,
        cell_probability=grid.cell_probability,
        hdr_levels=levels,
        hdr_masks=grid.hdr_masks,
    )


# ── 軌跡抽樣：雜湊最小優先序、固定容量、與批次到達順序無關 ─────────────────


def _stable_sample_priority(seed: int, particle_id: str) -> int:
    """由抽樣種子與粒子識別碼推導一顆粒子的抽樣優先序（數值越小越優先保留）。

    設計精神比照 ``backward_driver.derive_batch_diffusion_seed`` 與
    ``report_trajectory_identity.representative_priority_digest``：以 SHA-256
    （不使用受 ``PYTHONHASHSEED`` 影響的 Python 內建 ``hash``）雜湊
    ``f"{seed}:{particle_id}"``，取前 8 個位元組轉成無號 64-bit 整數。因為雜湊值
    只由「種子＋粒子識別碼」決定，與這顆粒子被哪一批、以什麼順序處理完全無關，
    所以「保留每個分組雜湊值最小的 K 顆」這個規則，不論批次到達順序為何、
    也不論同一個分組的粒子分散在多少個批次裡，重跑後必然選出完全相同的 K 顆
    粒子（見驗收：可重現性），且不需要事先知道分組總大小（不像傳統
    reservoir sampling 需要依序處理才能保證均勻性）。
    """

    if not isinstance(particle_id, str) or not particle_id:
        raise PostprocessError("particle_id 必須是非空字串")
    digest = hashlib.sha256(f"{seed}:{particle_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


@dataclass(slots=True)
class _TrajectorySampleReservoir:
    """單一分組鍵值的固定容量軌跡抽樣暫存區（以優先序最大堆維護，見上方函式）。

    ``heap`` 內每個元素是 ``(-priority, row_particle_index, snapshot)``；因為
    Python ``heapq`` 是最小堆，儲存 ``-priority`` 讓堆頂（``heap[0]``）永遠是
    目前保留集合中「優先序最差（priority 最大）」的一筆，插入新元素時只需要
    跟堆頂比較就能決定是否要汰換，不需要每次掃描整個保留集合。
    """

    sample_size: int
    heap: list[tuple[int, int, dict[str, np.ndarray]]] = field(default_factory=list)

    def offer(
        self,
        *,
        priority: int,
        row_particle_index: int,
        extract_snapshot: Callable[[], dict[str, np.ndarray]],
    ) -> None:
        """依優先序決定是否把這顆粒子的軌跡快照納入保留集合。

        ``extract_snapshot`` 是延遲呼叫的 callable（而不是直接傳入陣列），
        讓呼叫端只在真的需要保留這顆粒子時，才從 ``BackwardDriverResult`` 的
        軌跡陣列切出對應欄位，避免對注定會被捨棄的粒子做不必要的陣列複製。
        """

        neg_priority = -priority
        if len(self.heap) < self.sample_size:
            heapq.heappush(self.heap, (neg_priority, row_particle_index, extract_snapshot()))
        elif neg_priority > self.heap[0][0]:
            heapq.heapreplace(self.heap, (neg_priority, row_particle_index, extract_snapshot()))

    def snapshots(self) -> list[tuple[int, dict[str, np.ndarray]]]:
        """回傳 ``(row_particle_index, snapshot)`` 清單，依 ``row_particle_index`` 排序。

        排序只是為了讓輸出檔案的列序穩定、方便人工檢視／diff，不影響「選中哪
        K 筆」這個由雜湊優先序決定的核心結果。
        """

        return sorted(((row, snapshot) for _, row, snapshot in self.heap), key=lambda item: item[0])


def _extract_trajectory_snapshot(result: BackwardDriverResult, local_index: int) -> dict[str, np.ndarray]:
    """從一個批次結果切出單一粒子（批次內第 ``local_index`` 欄）的完整軌跡。

    複製（而非切片視圖）陣列，避免保留區間接持有整個 ``BackwardDriverResult``
    底層記憶體，讓呼叫端在這次 ``add_batch`` 結束後可以釋放整批軌跡陣列。
    """

    return {
        "trajectory_utc_ns": np.array(result.trajectory_utc_ns, dtype=np.int64, copy=True),
        "trajectory_x_m": np.array(result.trajectory_x_m[:, local_index], dtype=np.float64, copy=True),
        "trajectory_y_m": np.array(result.trajectory_y_m[:, local_index], dtype=np.float64, copy=True),
        "trajectory_z_m": np.array(result.trajectory_z_m[:, local_index], dtype=np.float64, copy=True),
        "trajectory_status": np.array(result.trajectory_status[:, local_index], dtype=np.int32, copy=True),
    }


# ── 批次指紋與 manifest 資料類別 ────────────────────────────────────────────


def compute_batch_result_fingerprint(result: BackwardDriverResult) -> str:
    """對一個 ``BackwardDriverResult`` 的「摘要」欄位計算穩定 SHA-256 指紋。

    只雜湊 ``row_particle_index``、``final_status``、``final_x/y/z_m``、
    ``final_utc_ns`` 與 ``hours_run``，**不**雜湊完整 ``trajectory_*`` 陣列內容
    ——這個指紋的用途是 manifest 記錄「這次後處理實際讀到了哪些批次結果」的
    provenance（同一個 run 若重新產生批次結果，指紋應該一致），不是要取代
    ``run_backward_batch`` 自己對軌跡數值正確性的驗收（那是 WP-5 的責任，見
    ``tests/test_native_backward_driver.py``）；把完整軌跡（可能每批次上萬個
    浮點數）也納入雜湊，只會讓每個批次的指紋計算成本大幅增加，卻不會讓
    「這批是不是同一批」這個判斷更準確（``final_*``＋``row_particle_index``
    已經足以識別批次身分與其最終結果）。
    """

    digest = hashlib.sha256()
    for name in ("row_particle_index", "final_status", "final_x_m", "final_y_m", "final_z_m", "final_utc_ns"):
        array = np.ascontiguousarray(getattr(result, name))
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(array.tobytes())
    digest.update(f"hours_run={int(result.hours_run)}".encode())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class BatchRecord:
    """manifest 中一筆輸入批次的紀錄：呼叫端提供的批次識別碼與結果指紋。"""

    batch_id: str
    particle_count: int
    fingerprint_sha256: str


@dataclass(frozen=True, slots=True)
class PostprocessConfig:
    """一次 ``PostprocessAccumulator`` 執行的設定；全部欄位在建構時驗證。

    ``trajectory_sample_seed`` 建議直接沿用該 run 的 ``master_seed``（與
    ``ParticleIdentityTable.master_seed`` 相同），但刻意獨立成另一個欄位而不是
    強制相等：抽樣種子改變只影響「畫哪幾條代表軌跡」這個報告用途，不是物理
    亂數串流的一部分，允許重跑報告時單獨調整抽樣而不影響任何物理計算。
    """

    trajectory_sample_size_per_group: int = 24
    trajectory_sample_seed: int = 0
    kde_grid_cell_count: int = 64
    kde_grid_padding_fraction: float = 0.15
    kde_minimum_point_count: int = 30
    kde_hdr_levels: tuple[float, ...] = (0.50, 0.75, 0.90)
    depth_quartile_count: int = 4
    project_root: str | None = None
    declared_git_commit: str | None = None
    formal_provenance: bool = False

    def __post_init__(self) -> None:
        if self.trajectory_sample_size_per_group < 1:
            raise PostprocessError("trajectory_sample_size_per_group 必須是正整數")
        if self.kde_grid_cell_count < 4:
            raise PostprocessError("kde_grid_cell_count 太小，無法建立有意義的格網")
        if not (0.0 <= self.kde_grid_padding_fraction < 5.0):
            raise PostprocessError("kde_grid_padding_fraction 不合理")
        if self.kde_minimum_point_count < 3:
            raise PostprocessError("kde_minimum_point_count 必須至少為 3（conditional_kde_2d 的硬性下限）")
        if self.depth_quartile_count < 1:
            raise PostprocessError("depth_quartile_count 必須是正整數")
        levels = tuple(float(v) for v in self.kde_hdr_levels)
        if not levels or any(not (0.0 < v < 1.0) for v in levels):
            raise PostprocessError("kde_hdr_levels 必須是介於 0 與 1 之間的非空序列")
        object.__setattr__(self, "kde_hdr_levels", levels)


@dataclass(frozen=True, slots=True)
class PostprocessManifest:
    """一次 WP-6 後處理執行的完整 provenance 與成果摘要。"""

    experiment_case_id: str
    master_seed: int
    identity_table_content_sha256: str
    particle_count_total: int
    batches: tuple[BatchRecord, ...]
    status_counts: Mapping[int, int]
    status_descriptions_zh: Mapping[int, str]
    output_files_sha256: Mapping[str, str]
    code_provenance: Mapping[str, Any]
    config: Mapping[str, Any]
    kde_bandwidth_method: str
    trajectory_sample_seed_policy: str
    resource_usage: Mapping[str, float]
    generated_at_utc_ns: int

    def to_json_dict(self) -> dict[str, Any]:
        """轉成可直接 ``json.dumps`` 的 dict（所有 key 轉成字串、值只用原生型別）。"""

        return {
            "experiment_case_id": self.experiment_case_id,
            "master_seed": self.master_seed,
            "identity_table_content_sha256": self.identity_table_content_sha256,
            "particle_count_total": self.particle_count_total,
            "batches": [
                {
                    "batch_id": item.batch_id,
                    "particle_count": item.particle_count,
                    "fingerprint_sha256": item.fingerprint_sha256,
                }
                for item in self.batches
            ],
            "status_counts": {str(code): count for code, count in self.status_counts.items()},
            "status_descriptions_zh": {str(code): text for code, text in self.status_descriptions_zh.items()},
            "output_files_sha256": dict(self.output_files_sha256),
            "code_provenance": dict(self.code_provenance),
            "config": dict(self.config),
            "kde_bandwidth_method": self.kde_bandwidth_method,
            "trajectory_sample_seed_policy": self.trajectory_sample_seed_policy,
            "resource_usage": dict(self.resource_usage),
            "generated_at_utc_ns": self.generated_at_utc_ns,
        }


# ── 依站內 z0_m 四分位數建立「受體深度」分組代理（見模組 docstring 第 3 點）──


def _assign_depth_quartile(identity_table: ParticleIdentityTable, *, bin_count: int) -> np.ndarray:
    """對每個 ``study_site_id`` 各自依 ``z0_m`` 分位數指定 0..bin_count-1 的深度分組。

    分位數邊界只由該站自己的粒子分布決定（不同站的深度分布、水深都不同，混在
    一起算分位數沒有意義）。若某一站全部粒子的 ``z0_m`` 完全相同（退化情況），
    ``np.quantile`` 會得到重複邊界，``np.digitize`` 仍能正確把全部點分進同一箱
    （不會拋例外），只是該站的深度分組退化成單一組，這是資料本身的限制，不是
    程式錯誤。
    """

    n = len(identity_table)
    bins = np.full(n, -1, dtype=np.int8)
    quantile_points = np.linspace(0.0, 1.0, bin_count + 1)[1:-1]
    for site in np.unique(identity_table.study_site_id):
        mask = identity_table.study_site_id == site
        depths = identity_table.z0_m[mask]
        if quantile_points.size == 0:
            bins[mask] = 0
            continue
        edges = np.quantile(depths, quantile_points)
        bins[mask] = np.digitize(depths, edges).astype(np.int8)
    if np.any(bins < 0):
        raise PostprocessError("深度分組未能覆蓋全部粒子，這是內部錯誤")
    return bins


def _seasons_for_arrival(identity_table: ParticleIdentityTable) -> np.ndarray:
    """向量化計算每顆粒子到達時刻對應的季節字串（重用既有 SEASON_BY_MONTH）。

    先把 UTC 奈秒轉成 ``datetime64[ns]``，再取整月份（``datetime64[M]``）換算成
    1-12 月，最後用查表陣列一次映射成季節字串；比逐列呼叫
    ``datetime.fromtimestamp`` 快非常多（50 萬列規模下差異明顯），但換算結果
    與逐列版本完全等價，因為兩者都是先取出 UTC 月份、再套用同一份
    ``SEASON_BY_MONTH``。
    """

    utc_ns = identity_table.arrival_time_utc_ns
    months = utc_ns.astype("datetime64[ns]").astype("datetime64[M]").astype(np.int64) % 12 + 1
    season_by_month_index = np.array([SEASON_BY_MONTH[month] for month in range(1, 13)])
    return season_by_month_index[months - 1]


def _write_table_with_csv(table: pa.Table, output_dir: Path, stem: str) -> None:
    """把同一張表同時寫成 ``<stem>.parquet``（正式資料契約）與 ``<stem>.csv``
    （人可讀，供任何試算表或文字編輯器直接開啟；欄位順序與 Parquet 完全相同）。
    """

    pq.write_table(table, str(output_dir / f"{stem}.parquet"))
    pa_csv.write_csv(table, str(output_dir / f"{stem}.csv"))


# ── 邊界進入統計：依開放邊界分段分類離開點（WP-17 任務 3，見模組 docstring
#    「邊界進入統計」一節的方法與已知限制揭露）───────────────────────────────


def classify_boundary_segment(
    x_m: float, y_m: float, geometry: BoundaryGeometry
) -> tuple[str, float]:
    """把一個「已離開關注區域」的離開點分類到最近的開放邊界分段。

    **方法（近似，已知限制，誠實揭露）**：native_runner 記錄的離開座標是子步
    結束後、已經在關注區域**之外**的原始粒子位置（見
    ``backward_driver.mirror_rk4_substep_loop`` 與 ``substep_numba`` 的關注區域
    判定），不是離開瞬間與 ``own_local_domain`` 邊界的精確交點；因此這裡**不能**
    重用既有 ``boundaries._open_boundary_crossing``（它要求輸入點幾乎剛好落在
    邊界線上，預設容忍度只有 ``1e-6`` 公尺）。本函式改用「離開點分別到
    ``own_local_open_boundary``／``flow_open_boundary`` 兩條開放水域線的最短
    直線距離，取較近者」這個較粗但穩健的啟發式；沒有開放邊界線可比對的站點
    （理論上不應發生，``own_local_open_boundary`` 缺席會由呼叫端的
    ``manifests.load_boundary_geometries`` 更早拒絕）回傳
    ``geometry.coastline_segment_id``。

    這個分類**不會**判斷離開點是不是真的離開海岸（``NATIVE_STATUS_STRANDED``
    走的是完全不同、以網格乾濕元素判定的路徑，不會呼叫本函式）；它只回答「這個
    已經確定成功穿越關注區域邊界的離開點，比較靠近哪一段開放邊界」。回傳的
    ``distance_m``（離開點到判定分段的直線距離）保留在輸出表中，供使用者評估
    這個啟發式在特定站點的典型誤差量級——距離越大，代表離開點離任何一段開放
    邊界線都比較遠（例如剛好在兩條線的交角附近，或一次子步位移特別大），分類
    結果的可信度也越低；本函式不對這個距離設任何拒絕門檻，是否要在下游報告中
    排除高距離的分類，留給使用者依實際量測到的距離分布決定。

    Returns:
        ``(boundary_segment_id, distance_m)``。
    """

    point = Point(float(x_m), float(y_m))
    candidates: list[tuple[str, float]] = []
    if geometry.own_local_open_boundary is not None:
        candidates.append(
            (geometry.own_local_boundary_segment_id, float(point.distance(geometry.own_local_open_boundary)))
        )
    if geometry.flow_open_boundary is not None:
        candidates.append(
            (geometry.flow_boundary_segment_id, float(point.distance(geometry.flow_open_boundary)))
        )
    if not candidates:
        return geometry.coastline_segment_id, float("nan")
    candidates.sort(key=lambda item: item[1])
    return candidates[0]


def _boundary_entry_table(
    counts: Mapping[tuple[Any, ...], int], key_columns: tuple[str, ...]
) -> pa.Table:
    """把 ``{(分組鍵..., boundary_segment_id): 穿越次數}`` 轉成人可讀的相對來源權重表。

    每一列是一個 ``(分組鍵..., boundary_segment_id)`` 組合；``relative_source_weight``
    是「這個分段的穿越次數」除以「同一分組鍵底下全部分段的穿越次數總和」——即
    AGENTS.md 定義的相對來源權重，只在同一分組（例如同一站）內部比較，不跨分組
    比較，也不宣稱是絕對來源機率。
    """

    totals: dict[tuple[Any, ...], int] = {}
    for key, count in counts.items():
        group_key = key[:-1]
        totals[group_key] = totals.get(group_key, 0) + count

    rows: list[dict[str, Any]] = []
    for key in sorted(counts):
        group_key = key[:-1]
        segment_id = key[-1]
        count = counts[key]
        total = totals[group_key]
        row: dict[str, Any] = dict(zip(key_columns, group_key, strict=True))
        row["boundary_segment_id"] = segment_id
        row["crossed_count"] = count
        row["relative_source_weight_within_group"] = float(count) / float(total) if total else float("nan")
        rows.append(row)
    if not rows:
        fields = [pa.field(name, pa.string()) for name in key_columns] + [
            pa.field("boundary_segment_id", pa.string()),
            pa.field("crossed_count", pa.int64()),
            pa.field("relative_source_weight_within_group", pa.float64()),
        ]
        return pa.table({f.name: pa.array([], type=f.type) for f in fields}, schema=pa.schema(fields))
    return pa.Table.from_pylist(rows)


# ── 邊界通量統計：向量化的一維邊際分組計數（見模組 docstring 第 2 點限制）───


def _flux_table_for_grouping(
    group_values: np.ndarray,
    group_column_name: str,
    final_status: np.ndarray,
) -> pa.Table:
    """依單一分組欄位（例如 ``study_site_id``）建立邊界通量統計表。

    以 ``np.unique(..., return_inverse=True)`` 把分組值轉成連續整數編碼，再用
    一次 ``np.bincount`` 對「分組編碼 × 狀態欄位索引」的攤平索引計數，取代
    「對每個分組值、每個狀態各掃描一次陣列」的寫法（後者是
    ``O(分組數 × 狀態數 × N)``，前者是 ``O(N + 分組數 × 狀態數)``），在 50 萬列
    規模下差異明顯。
    """

    unique_groups, group_codes = np.unique(group_values, return_inverse=True)
    n_groups = unique_groups.shape[0]
    n_status = len(_FLUX_STATUS_ORDER)
    status_index = {code: index for index, code in enumerate(_FLUX_STATUS_ORDER)}
    status_codes = np.asarray([status_index[int(value)] for value in final_status], dtype=np.int64)
    flat_index = group_codes.astype(np.int64) * n_status + status_codes
    counts = np.bincount(flat_index, minlength=n_groups * n_status).reshape(n_groups, n_status)

    totals = counts.sum(axis=1)
    if np.any(totals <= 0):
        raise PostprocessError("邊界通量統計出現分母為零的分組，這是內部錯誤")

    columns: dict[str, Any] = {group_column_name: pa.array(unique_groups)}
    columns["total_particle_count"] = pa.array(totals.astype(np.int64))
    for status_position, status_code in enumerate(_FLUX_STATUS_ORDER):
        count_column = counts[:, status_position].astype(np.int64)
        fraction_column = count_column.astype(np.float64) / totals.astype(np.float64)
        suffix = "crossed" if status_code == NATIVE_STATUS_LEFT_INTEREST_REGION else f"status_{status_code}"
        columns[f"count_{suffix}"] = pa.array(count_column)
        columns[f"fraction_{suffix}"] = pa.array(fraction_column)
    return pa.table(columns)


# ── 主要串流累加器 ───────────────────────────────────────────────────────


class PostprocessAccumulator:
    """單一 pass 串流彙整 ``BackwardDriverResult`` 批次的主要入口。

    使用方式：對每個批次結果呼叫一次 :meth:`add_batch`（可以是
    ``run_backward_batch`` 的即時輸出，處理完立刻丟給這裡、不需要先全部收集
    起來），全部批次處理完後呼叫一次 :meth:`finalize` 取得
    :class:`PostprocessManifest` 並把所有成果寫到 ``output_dir``。

    記憶體足跡：本物件在 ``__init__`` 時配置的陣列全部是 ``O(粒子總數)``
    （離開狀態、離開位置、離開時刻、季節、深度分組），大小與
    ``ParticleIdentityTable`` 本身同量級；:meth:`add_batch` 之後不會持有任何
    ``BackwardDriverResult`` 的參照（只複製「這一批次需要的少量摘要」），批次
    物件在呼叫端可以立刻被垃圾回收。軌跡抽樣暫存區是 ``O(分組數 × 抽樣上限)``，
    KDE 離開點暫存是最多 ``O(粒子總數)``（只保留座標，不保留軌跡）。
    """

    def __init__(
        self,
        identity_table: ParticleIdentityTable,
        *,
        output_dir: str | Path,
        config: PostprocessConfig | None = None,
        lonlat_projector: LonLatProjector | None = None,
        boundary_geometries: Mapping[str, BoundaryGeometry] | None = None,
    ) -> None:
        self._identity_table = identity_table
        self._config = config if config is not None else PostprocessConfig()
        self._lonlat_projector = lonlat_projector
        self._boundary_geometries = boundary_geometries
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # WP-17 任務 3：依 (study_site_id, boundary_segment_id) 與
        # (study_site_id, settling_velocity_mps, boundary_segment_id) 累計穿越次數，
        # 只在提供 boundary_geometries 時才有內容（見 classify_boundary_segment）。
        self._boundary_entry_counts_by_site: dict[tuple[str, str], int] = {}
        self._boundary_entry_counts_by_site_velocity: dict[tuple[str, float, str], int] = {}

        n = len(identity_table)
        self._n_total = n
        self._seen = np.zeros(n, dtype=bool)
        self._final_status = np.full(n, NATIVE_STATUS_ACTIVE, dtype=np.int32)
        self._final_x_m = np.full(n, np.nan, dtype=np.float64)
        self._final_y_m = np.full(n, np.nan, dtype=np.float64)
        self._final_z_m = np.full(n, np.nan, dtype=np.float64)
        self._final_utc_ns = np.zeros(n, dtype=np.int64)

        self._season = _seasons_for_arrival(identity_table)
        self._depth_quartile = _assign_depth_quartile(
            identity_table, bin_count=self._config.depth_quartile_count
        )

        self._batches: list[BatchRecord] = []
        self._start_wall_time = time.perf_counter()

        # 軌跡抽樣：兩種分組方案，各自獨立維護固定容量的雜湊優先序保留集合。
        self._trajectory_reservoirs_by_site: dict[str, _TrajectorySampleReservoir] = {}
        self._trajectory_reservoirs_by_site_velocity: dict[tuple[str, float], _TrajectorySampleReservoir] = {}

        # KDE 離開點暫存：只保留成功穿越邊界的粒子座標，依站與站×沉降速度分組。
        self._kde_points_by_site: dict[str, list[np.ndarray]] = {}
        self._kde_points_by_site_velocity: dict[tuple[str, float], list[np.ndarray]] = {}

        departure_path = self._output_dir / "departures.parquet"
        self._departure_path = departure_path
        self._departure_writer = pq.ParquetWriter(str(departure_path), DEPARTURE_TABLE_SCHEMA)

    # -- 批次寫入 ---------------------------------------------------------

    def add_batch(self, batch_id: str, result: BackwardDriverResult) -> None:
        """處理一個批次結果：驗證一致性、寫入離開紀錄、更新抽樣與 KDE 暫存區。

        Raises:
            PostprocessError: 出現任何一致性違反（重複粒子、超出範圍的列位置、
                ``final_status`` 仍是 ``NATIVE_STATUS_ACTIVE``、離開紀錄與
                ``final_status`` 不一致），一律 fail-closed，不嘗試修補或跳過。
        """

        rows = np.asarray(result.row_particle_index, dtype=np.int64)
        if rows.ndim != 1 or rows.size == 0:
            raise PostprocessError(f"批次 {batch_id!r} 的 row_particle_index 必須是非空一維陣列")
        if np.any((rows < 0) | (rows >= self._n_total)):
            raise PostprocessError(f"批次 {batch_id!r} 出現超出身分表範圍的 row_particle_index")
        if np.unique(rows).size != rows.size:
            raise PostprocessError(f"批次 {batch_id!r} 內部出現重複的 row_particle_index")
        if np.any(self._seen[rows]):
            duplicated = rows[self._seen[rows]]
            raise PostprocessError(
                f"批次 {batch_id!r} 內含先前批次已處理過的粒子（列位置例如 {int(duplicated[0])}），"
                "違反「每顆粒子恰好出現一次」的契約"
            )

        final_status = np.asarray(result.final_status, dtype=np.int32)
        if final_status.shape != rows.shape:
            raise PostprocessError(f"批次 {batch_id!r} 的 final_status 形狀與 row_particle_index 不一致")
        if np.any(final_status == NATIVE_STATUS_ACTIVE):
            raise PostprocessError(
                f"批次 {batch_id!r} 出現 final_status 仍是 NATIVE_STATUS_ACTIVE 的粒子，"
                "代表這個 run 尚未完整結束（run_backward_batch 只在全部粒子終止或觸及回溯"
                "上限時才回傳），後處理拒絕接受未完成的批次結果"
            )

        # 兩種列位置不可混用：``result.row_particle_index`` 是全體身分表的列位置（全域），
        # 而驅動層（``backward_driver``／``hour_major``）寫進 ``DepartureRecord`` 的
        # ``row_particle_index`` 是「這個批次內」的局部位置 0..n-1。因此先以
        # ``rows[局部位置]`` 轉成全域列位置，再與 ``final_status`` 逐顆比對。
        departure_status_by_row: dict[int, int] = {}
        for record in result.departures:
            local = int(record.row_particle_index)
            if local < 0 or local >= rows.size:
                raise PostprocessError(
                    f"批次 {batch_id!r} 的 DepartureRecord 局部列位置 {local} 超出批次範圍 0..{rows.size - 1}"
                )
            global_row = int(rows[local])
            if global_row in departure_status_by_row:
                raise PostprocessError(f"批次 {batch_id!r} 對同一顆粒子回報了超過一筆 DepartureRecord")
            departure_status_by_row[global_row] = int(record.status)
        for row_value, status_value in zip(rows.tolist(), final_status.tolist(), strict=True):
            departure_status = departure_status_by_row.get(row_value)
            if departure_status is None:
                raise PostprocessError(
                    f"批次 {batch_id!r} 的粒子 {row_value} final_status={status_value} 但沒有對應的 "
                    "DepartureRecord，departures 與 final_status 不一致"
                )
            if departure_status != status_value:
                raise PostprocessError(
                    f"批次 {batch_id!r} 的粒子 {row_value} DepartureRecord.status={departure_status} 與 "
                    f"final_status={status_value} 不一致"
                )

        # 到這裡才真正寫入內部狀態，確保上面任何一項檢查失敗時，這個批次完全
        # 不會留下部分寫入的痕跡（seen 陣列與已終止的 parquet writer 沒有回滾
        # 機制，因此驗證必須在任何寫入動作之前全部通過）。
        self._seen[rows] = True
        self._final_status[rows] = final_status
        self._final_x_m[rows] = np.asarray(result.final_x_m, dtype=np.float64)
        self._final_y_m[rows] = np.asarray(result.final_y_m, dtype=np.float64)
        self._final_z_m[rows] = np.asarray(result.final_z_m, dtype=np.float64)
        self._final_utc_ns[rows] = np.asarray(result.final_utc_ns, dtype=np.int64)

        self._write_departure_batch(rows, final_status, result)
        self._update_trajectory_samples(rows, result)
        self._update_kde_points(rows, final_status)

        self._batches.append(
            BatchRecord(
                batch_id=str(batch_id),
                particle_count=int(rows.size),
                fingerprint_sha256=compute_batch_result_fingerprint(result),
            )
        )

    def _write_departure_batch(
        self, rows: np.ndarray, final_status: np.ndarray, result: BackwardDriverResult
    ) -> None:
        """把這一批粒子的離開紀錄寫進 Parquet（一次一批，串流寫入不緩存全表）。"""

        identity = self._identity_table
        arrival_utc = identity.arrival_time_utc_ns[rows]
        departure_utc = np.asarray(result.final_utc_ns, dtype=np.int64)
        backtrack_seconds = (arrival_utc - departure_utc).astype(np.float64) / 1.0e9
        if np.any(backtrack_seconds < -1.0e-6):
            raise PostprocessError("出現離開時刻晚於到達時刻的粒子，違反逆向追蹤的時間方向")
        backtrack_seconds = np.maximum(backtrack_seconds, 0.0)

        crossed = final_status == NATIVE_STATUS_LEFT_INTEREST_REGION
        status_desc = np.array([_describe_status(code) for code in final_status.tolist()], dtype=object)

        final_x = np.asarray(result.final_x_m, dtype=np.float64)
        final_y = np.asarray(result.final_y_m, dtype=np.float64)
        final_z = np.asarray(result.final_z_m, dtype=np.float64)

        if self._lonlat_projector is not None:
            lon, lat = self._lonlat_projector(identity.study_site_id[rows], final_x, final_y)
            lon_array = pa.array(np.asarray(lon, dtype=np.float64))
            lat_array = pa.array(np.asarray(lat, dtype=np.float64))
        else:
            lon_array = pa.array([None] * rows.size, type=pa.float64())
            lat_array = pa.array([None] * rows.size, type=pa.float64())

        segment_ids: list[str | None] = [None] * rows.size
        segment_distances: list[float | None] = [None] * rows.size
        if self._boundary_geometries is not None:
            site_ids = identity.study_site_id[rows]
            velocities = identity.settling_velocity_mps[rows]
            for local_index in np.flatnonzero(crossed):
                site_id = str(site_ids[local_index])
                geometry = self._boundary_geometries.get(site_id)
                if geometry is None:
                    raise PostprocessError(
                        f"boundary_geometries 缺少站點 {site_id!r} 的邊界幾何，"
                        "無法分類這顆已穿越關注區域邊界的粒子"
                    )
                segment_id, distance_m = classify_boundary_segment(
                    float(final_x[local_index]), float(final_y[local_index]), geometry
                )
                segment_ids[local_index] = segment_id
                segment_distances[local_index] = distance_m
                site_key = (site_id, segment_id)
                self._boundary_entry_counts_by_site[site_key] = (
                    self._boundary_entry_counts_by_site.get(site_key, 0) + 1
                )
                velocity_key = (site_id, float(velocities[local_index]), segment_id)
                self._boundary_entry_counts_by_site_velocity[velocity_key] = (
                    self._boundary_entry_counts_by_site_velocity.get(velocity_key, 0) + 1
                )

        columns = {
            "particle_id": pa.array(identity.particle_id[rows]),
            "scenario_id": pa.array(identity.scenario_id[rows]),
            "member_id": pa.array(identity.member_id[rows]),
            "study_site_id": pa.array(identity.study_site_id[rows]),
            "analysis_region_id": pa.array(identity.analysis_region_id[rows]),
            "receptor_id": pa.array(identity.receptor_id[rows]),
            "material_id": pa.array(identity.material_id[rows]),
            "arrival_time_id": pa.array(identity.arrival_time_id[rows]),
            "arrival_time_utc_ns": pa.array(arrival_utc),
            "settling_velocity_mps": pa.array(identity.settling_velocity_mps[rows]),
            "season": pa.array(self._season[rows]),
            "depth_quartile": pa.array(self._depth_quartile[rows]),
            "final_status_code": pa.array(final_status),
            "final_status_description_zh": pa.array(status_desc, type=pa.string()),
            "crossed_interest_region_boundary": pa.array(crossed),
            "departure_x_m": pa.array(final_x),
            "departure_y_m": pa.array(final_y),
            "departure_z_m": pa.array(final_z),
            "departure_lon_deg": lon_array,
            "departure_lat_deg": lat_array,
            "departure_utc_ns": pa.array(departure_utc),
            "backtrack_duration_seconds": pa.array(backtrack_seconds),
            "row_particle_index": pa.array(rows),
            "boundary_segment_id": pa.array(segment_ids, type=pa.string()),
            "boundary_classification_distance_m": pa.array(segment_distances, type=pa.float64()),
        }
        table = pa.table(columns, schema=DEPARTURE_TABLE_SCHEMA)
        self._departure_writer.write_table(table)

    def _update_trajectory_samples(self, rows: np.ndarray, result: BackwardDriverResult) -> None:
        """對這一批每顆粒子計算抽樣優先序，更新兩種分組方案各自的保留集合。"""

        identity = self._identity_table
        seed = self._config.trajectory_sample_seed
        size = self._config.trajectory_sample_size_per_group
        for local_index, row in enumerate(rows.tolist()):
            particle_id = str(identity.particle_id[row])
            priority = _stable_sample_priority(seed, particle_id)
            site_key = str(identity.study_site_id[row])
            velocity_key = (site_key, float(identity.settling_velocity_mps[row]))

            site_reservoir = self._trajectory_reservoirs_by_site.setdefault(
                site_key, _TrajectorySampleReservoir(sample_size=size)
            )
            site_reservoir.offer(
                priority=priority,
                row_particle_index=row,
                extract_snapshot=lambda idx=local_index: _extract_trajectory_snapshot(result, idx),
            )

            velocity_reservoir = self._trajectory_reservoirs_by_site_velocity.setdefault(
                velocity_key, _TrajectorySampleReservoir(sample_size=size)
            )
            velocity_reservoir.offer(
                priority=priority,
                row_particle_index=row,
                extract_snapshot=lambda idx=local_index: _extract_trajectory_snapshot(result, idx),
            )

    def _update_kde_points(self, rows: np.ndarray, final_status: np.ndarray) -> None:
        """把這一批成功穿越邊界的離開座標點，累加進對應分組的暫存清單。"""

        crossed_mask = final_status == NATIVE_STATUS_LEFT_INTEREST_REGION
        if not np.any(crossed_mask):
            return
        identity = self._identity_table
        crossed_rows = rows[crossed_mask]
        xy = np.stack(
            [self._final_x_m[crossed_rows], self._final_y_m[crossed_rows]],
            axis=1,
        )
        sites = identity.study_site_id[crossed_rows]
        velocities = identity.settling_velocity_mps[crossed_rows]
        for site in np.unique(sites):
            site_mask = sites == site
            self._kde_points_by_site.setdefault(str(site), []).append(xy[site_mask])
            site_velocities = velocities[site_mask]
            site_xy = xy[site_mask]
            for velocity in np.unique(site_velocities):
                velocity_mask = site_velocities == velocity
                key = (str(site), float(velocity))
                self._kde_points_by_site_velocity.setdefault(key, []).append(site_xy[velocity_mask])

    # -- 收尾：一致性檢查、統計輸出、manifest -----------------------------

    def finalize(self) -> PostprocessManifest:
        """關閉離開紀錄表、執行全域一致性檢查、輸出全部統計產品與 manifest。

        Raises:
            PostprocessError: 有粒子完全沒有出現在任何批次中（``self._seen``
                未全部為真），或狀態計數總和與粒子總數不符。
        """

        self._departure_writer.close()

        if not np.all(self._seen):
            missing = int(np.count_nonzero(~self._seen))
            raise PostprocessError(
                f"有 {missing} 顆粒子從未出現在任何批次結果中，後處理視為不完整，拒絕產出成果"
            )

        status_counts = {
            int(code): int(np.count_nonzero(self._final_status == code)) for code in _FLUX_STATUS_ORDER
        }
        if sum(status_counts.values()) != self._n_total:
            raise PostprocessError("各狀態計數總和與粒子總數不一致，這是嚴重的內部錯誤")

        self._write_flux_tables()
        kde_summary = self._write_kde_products()
        self._write_trajectory_samples()

        output_files_sha256 = {
            path.name: sha256_file(path) for path in sorted(self._output_dir.glob("*.parquet"))
        }

        code_provenance: Mapping[str, Any]
        if self._config.project_root is not None:
            provenance: CodeProvenance = collect_code_provenance(
                self._config.project_root,
                self._config.declared_git_commit,
                formal=self._config.formal_provenance,
            )
            code_provenance = provenance.to_dict()
        else:
            code_provenance = {"note": "未提供 project_root，略過程式部署指紋收集（僅限測試或探索用途）"}

        peak_rss_kib = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        resource_usage = {
            "wall_seconds": time.perf_counter() - self._start_wall_time,
            "peak_rss_kib": peak_rss_kib,
        }

        manifest = PostprocessManifest(
            experiment_case_id=self._identity_table.experiment_case_id,
            master_seed=self._identity_table.master_seed,
            identity_table_content_sha256=self._identity_table.content_sha256,
            particle_count_total=self._n_total,
            batches=tuple(self._batches),
            status_counts=status_counts,
            status_descriptions_zh={code: _describe_status(code) for code in _FLUX_STATUS_ORDER},
            output_files_sha256=output_files_sha256,
            code_provenance=code_provenance,
            config={
                "trajectory_sample_size_per_group": self._config.trajectory_sample_size_per_group,
                "trajectory_sample_seed": self._config.trajectory_sample_seed,
                "kde_grid_cell_count": self._config.kde_grid_cell_count,
                "kde_grid_padding_fraction": self._config.kde_grid_padding_fraction,
                "kde_minimum_point_count": self._config.kde_minimum_point_count,
                "kde_hdr_levels": list(self._config.kde_hdr_levels),
                "depth_quartile_count": self._config.depth_quartile_count,
                "kde_group_summary": kde_summary,
                "boundary_segment_classification_available": self._boundary_geometries is not None,
                "boundary_segment_classification_method": (
                    "nearest_open_boundary_line_by_euclidean_distance_v1（見"
                    " postprocess.classify_boundary_segment docstring 的近似方法與已知限制）"
                    if self._boundary_geometries is not None
                    else None
                ),
            },
            kde_bandwidth_method=(
                "scipy.stats.gaussian_kde 預設 Scott 法則（factor = n**(-1/(d+4))，d=2），"
                "經既有 aggregation.conditional_kde_2d 轉呼叫，見模組 docstring「頻寬選擇」一節"
            ),
            trajectory_sample_seed_policy=(
                "SHA-256(f'{trajectory_sample_seed}:{particle_id}') 前 8 bytes 轉 uint64 當優先序，"
                "各分組保留優先序最小的固定 K 筆，與批次到達順序無關，見 _stable_sample_priority docstring"
            ),
            resource_usage=resource_usage,
            generated_at_utc_ns=time.time_ns(),
        )

        manifest_path = self._output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest.to_json_dict(), ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        return manifest

    def _write_flux_tables(self) -> None:
        """輸出四個一維邊際邊界通量統計表（見模組 docstring 第 2 點的範圍限制），
        以及（提供 ``boundary_geometries`` 時）依開放邊界分段的相對來源權重表
        （WP-17 任務 3）。每張表同時寫出 Parquet 與 CSV 兩種格式。
        """

        identity = self._identity_table
        groupings = (
            ("study_site_id", identity.study_site_id, "boundary_flux_by_site"),
            (
                "settling_velocity_mps",
                identity.settling_velocity_mps,
                "boundary_flux_by_settling_velocity",
            ),
            ("depth_quartile", self._depth_quartile, "boundary_flux_by_depth_quartile"),
            ("season", self._season, "boundary_flux_by_season"),
        )
        for column_name, values, stem in groupings:
            table = _flux_table_for_grouping(values, column_name, self._final_status)
            _write_table_with_csv(table, self._output_dir, stem)

        if self._boundary_geometries is not None:
            by_site_table = _boundary_entry_table(
                self._boundary_entry_counts_by_site, ("study_site_id",)
            )
            _write_table_with_csv(by_site_table, self._output_dir, "boundary_entry_by_site")
            by_site_velocity_table = _boundary_entry_table(
                self._boundary_entry_counts_by_site_velocity,
                ("study_site_id", "settling_velocity_mps"),
            )
            _write_table_with_csv(
                by_site_velocity_table, self._output_dir, "boundary_entry_by_site_settling_velocity"
            )

    def _write_kde_products(self) -> dict[str, dict[str, Any]]:
        """對兩種分組（各站；各站×沉降速度）建立條件式來源足跡並寫成 Parquet。

        Returns:
            分組鍵值到 ``{"status": ..., "raw_point_count": ...}`` 的摘要 dict，
            供 manifest 快速呈現各分組的可估計性，不需要重新讀取 Parquet 檔案。
        """

        summary: dict[str, dict[str, Any]] = {}

        site_rows: list[dict[str, Any]] = []
        for site, chunks in sorted(self._kde_points_by_site.items()):
            points = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 2), dtype=np.float64)
            product = build_source_footprint_kde(
                points,
                grid_cell_count=self._config.kde_grid_cell_count,
                padding_fraction=self._config.kde_grid_padding_fraction,
                minimum_point_count=self._config.kde_minimum_point_count,
                hdr_levels=self._config.kde_hdr_levels,
            )
            site_rows.append(_kde_product_to_row({"study_site_id": site}, product))
            summary[f"site={site}"] = {"status": product.status, "raw_point_count": product.raw_point_count}
        pq.write_table(pa.Table.from_pylist(site_rows), str(self._output_dir / "kde_by_site.parquet"))

        site_velocity_rows: list[dict[str, Any]] = []
        for (site, velocity), chunks in sorted(self._kde_points_by_site_velocity.items()):
            points = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 2), dtype=np.float64)
            product = build_source_footprint_kde(
                points,
                grid_cell_count=self._config.kde_grid_cell_count,
                padding_fraction=self._config.kde_grid_padding_fraction,
                minimum_point_count=self._config.kde_minimum_point_count,
                hdr_levels=self._config.kde_hdr_levels,
            )
            site_velocity_rows.append(
                _kde_product_to_row(
                    {"study_site_id": site, "settling_velocity_mps": velocity},
                    product,
                )
            )
            summary[f"site={site},settling_velocity_mps={velocity}"] = {
                "status": product.status,
                "raw_point_count": product.raw_point_count,
            }
        pq.write_table(
            pa.Table.from_pylist(site_velocity_rows),
            str(self._output_dir / "kde_by_site_settling_velocity.parquet"),
        )
        return summary

    def _write_trajectory_samples(self) -> None:
        """輸出兩種分組方案（各站；各站×沉降速度）的軌跡抽樣 Parquet 表。"""

        identity = self._identity_table

        def _rows_for(
            reservoirs: Mapping[Any, _TrajectorySampleReservoir], key_fields: Sequence[str]
        ) -> list[dict[str, Any]]:
            rows: list[dict[str, Any]] = []
            for key, reservoir in reservoirs.items():
                key_tuple = key if isinstance(key, tuple) else (key,)
                for row_particle_index, snapshot in reservoir.snapshots():
                    record: dict[str, Any] = dict(zip(key_fields, key_tuple, strict=True))
                    record["row_particle_index"] = int(row_particle_index)
                    record["particle_id"] = str(identity.particle_id[row_particle_index])
                    record["trajectory_utc_ns"] = snapshot["trajectory_utc_ns"].tolist()
                    record["trajectory_x_m"] = snapshot["trajectory_x_m"].tolist()
                    record["trajectory_y_m"] = snapshot["trajectory_y_m"].tolist()
                    record["trajectory_z_m"] = snapshot["trajectory_z_m"].tolist()
                    record["trajectory_status_code"] = snapshot["trajectory_status"].tolist()
                    rows.append(record)
            return rows

        by_site_rows = _rows_for(self._trajectory_reservoirs_by_site, ("study_site_id",))
        pq.write_table(
            pa.Table.from_pylist(by_site_rows),
            str(self._output_dir / "trajectory_samples_by_site.parquet"),
        )

        by_site_velocity_rows = _rows_for(
            self._trajectory_reservoirs_by_site_velocity, ("study_site_id", "settling_velocity_mps")
        )
        pq.write_table(
            pa.Table.from_pylist(by_site_velocity_rows),
            str(self._output_dir / "trajectory_samples_by_site_settling_velocity.parquet"),
        )


def _kde_product_to_row(key_fields: dict[str, Any], product: SourceFootprintKDEProduct) -> dict[str, Any]:
    """把 :class:`SourceFootprintKDEProduct` 攤平成一列可寫進 Parquet 的 dict。

    格網欄位一律以 row-major（``x`` 為內層、``y`` 為外層）攤平成一維 list，並
    另外保存 ``grid_rows``／``grid_cols`` 供還原成二維陣列；HDR 遮罩依
    ``hdr_levels`` 的順序串接成單一 ``hdr_masks_flat`` list，每個 level 各佔
    ``grid_rows × grid_cols`` 個元素，解碼方式見模組頂端 docstring 與
    ``manifest.json`` 的 ``kde_bandwidth_method`` 說明鄰近段落。狀態不是
    ``KDE_STATUS_AVAILABLE`` 時，全部格網／密度／HDR 欄位都是空 list，只保留
    ``raw_point_count``，不得以空格網暗示零密度。
    """

    row: dict[str, Any] = dict(key_fields)
    row["status"] = product.status
    row["raw_point_count"] = product.raw_point_count
    row["bandwidth_factor"] = product.bandwidth_factor
    row["hdr_levels"] = list(product.hdr_levels)
    if product.status != KDE_STATUS_AVAILABLE:
        row["grid_rows"] = 0
        row["grid_cols"] = 0
        row["x_centers_m"] = []
        row["y_centers_m"] = []
        row["density_per_m2_flat"] = []
        row["cell_probability_flat"] = []
        row["hdr_masks_flat"] = []
        return row

    assert product.density_per_m2 is not None  # 供型別檢查器確認，前面的狀態判斷已保證
    grid_rows, grid_cols = product.density_per_m2.shape
    row["grid_rows"] = int(grid_rows)
    row["grid_cols"] = int(grid_cols)
    row["x_centers_m"] = product.x_centers_m.tolist()
    row["y_centers_m"] = product.y_centers_m.tolist()
    row["density_per_m2_flat"] = product.density_per_m2.reshape(-1).tolist()
    row["cell_probability_flat"] = product.cell_probability.reshape(-1).tolist()
    masks_flat: list[bool] = []
    for level in product.hdr_levels:
        masks_flat.extend(bool(value) for value in product.hdr_masks[level].reshape(-1).tolist())
    row["hdr_masks_flat"] = masks_flat
    return row
