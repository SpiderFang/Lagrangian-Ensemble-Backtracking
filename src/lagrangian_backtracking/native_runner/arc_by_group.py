"""沿開放邊界之「相對來源權重」曲線（依組別分線）的計算核心。

這個模組回答什麼問題
--------------------
逆向系集把粒子從受體往回追，直到粒子穿越流場範圍的開放邊界為止；穿越點全部落在一條
一維的邊界折線上。既有圖 ``boundary_departure_along_arc`` 把「全部」穿越粒子沿邊界弧長
畫成直條圖。本模組進一步把穿越粒子依下列三種方式分組，每一組各算一條沿弧長的平滑曲線，
用來檢視「不同組別的來源位置分布是否有差異」：

* ``class``：十個沉降速度類別（``material_id``）；
* ``season``：沉底時刻所在的季節（欄位 ``deposition_season``）；
* ``tide``：沉底時刻的大潮／小潮代理分類（欄位 ``deposition_tide_class``）。

季節與潮況**必須**使用以沉底時刻（粒子回溯起點）計算的 ``deposition_*`` 欄位；觀測錨點被
選出時的舊標籤 ``observation_*`` 與粒子軌跡沒有物理關聯，本模組一律不讀（見
``postprocess`` 的 ``LABEL_FLUX_TABLE_NOTE_ZH``）。標籤已在後處理階段寫進逐粒子表
``departures.parquet``，本模組只讀取、不重算。

曲線的定義（縱軸單位為「% 每公里」）
------------------------------------
* 對每一組，取該組所有「穿越開放邊界且有弧長位置」的粒子（分母就是該組的穿越粒子數），
  把弧長 ``[0, 邊界全長)`` 等分成 ``n_grid`` 格，先數每格粒子數，再用一維高斯核
  （頻寬 ``sigma_km``，單位公里）平滑，最後除以格寬並乘 100，使「曲線對弧長積分 = 100%」。
* 曲線值稱為「相對來源權重」，**不是絕對的來源量化值**：它只表示在「本次回溯的釋放設計」下，穿越
  粒子在邊界各處的相對占比，沒有建立先驗、似然與觀測驗證，不可解讀為絕對來源或因果歸因
  （見 ``AGENTS.md`` 科學與資料契約）。

邊界端點處理
------------
* 邊界折線**首尾閉合**（第一個頂點與最後一個頂點距離不超過 ``closure_tolerance_m``）時，
  弧長 0 與全長是同一個點，平滑時首尾環狀接續（核函數繞過端點）。
* 邊界折線**未閉合**時，兩端是真正的終點，粒子不可能「繞過去」；改採**鏡射**處理
  （端點外的核函數質量折回邊界內），因此質量守恆、積分仍為 100%，端點附近不會被人為壓低。
  兩種情況都在輸出表的 ``end_treatment`` 欄位與圖說中註明。

雜訊基準（回答「差異是否超過抽樣雜訊」）
--------------------------------------
兩組曲線之間的距離用總變異距離（total variation distance，TVD）：兩個分布差的絕對值
總和之半，範圍 0 到 1。因為每組只是有限樣本，即使兩組的真實分布完全相同，TVD 也不會是
0，必須有雜訊基準才能判讀。本模組提供三種基準（皆以同一套平滑流程計算 TVD）：

1. **同組隨機對半切（粒子層級）**：把同一組粒子隨機分成兩半，算兩半的 TVD。沒有任何組別
   差異，純粹是抽樣雜訊。注意對半後每半只有一半樣本，TVD 會比「全樣本對全樣本」略大，
   是保守（偏嚴）的基準。
2. **同組隨機對半切（沉底事件層級）**：以「沉底事件」（同一到達時刻、同一系集成員，
   ``arrival_time_id`` ＋ ``member_id``）為單位整批對半切。同一沉底事件的粒子共用同一個
   沉底時刻，彼此並不獨立，事件層級的雜訊通常明顯大於粒子層級，是更貼近實際獨立單位的基準。
3. **組別標籤的置換基準**：
   類別分組使用「同一釋放點交換類別標籤」的配對置換：各類別共用同一批釋放點
   （受體 × 到達時刻 × 成員），虛無假設為「類別不影響某釋放點粒子的穿越位置」，故對兩類
   共有的釋放點以二分之一的隨機機會交換其弧長位置，再算兩組 TVD；季節與潮況分組的組別互不相交，
   改用「沉底事件層級的標籤置換」：把兩組的事件混合後，依原事件數隨機重新分派。

觀測 TVD 大於對應置換基準的第 95 百分位時，標示為「超過雜訊」。第 99 百分位僅供參考
（重抽次數不多時尾端估計粗糙）。

資料契約
--------
讀取 ``native-postprocess`` 輸出目錄（只讀）：``departures.parquet``（逐粒子表）、
``boundary_departure_along_arc_by_site.parquet``（邊界全長）、
``boundary_edge_intervals_by_site.parquet``（各邊與轉角的弧長區間）、
``footprint_geometry_by_site.parquet``（邊界折線頂點，用來檢查是否閉合）。任一表缺漏、
欄位缺漏或站內穿越粒子數與弧長分布表不一致，一律拋出 :class:`ArcByGroupError`（失敗即停）。
"""

from __future__ import annotations

import csv
import math
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy.ndimage import gaussian_filter1d

from .display_names import material_display_name_zh

__all__ = [
    "ARC_GROUPINGS",
    "ARC_EDGE_ORDER",
    "ArcByGroupError",
    "ArcByGroupParams",
    "GroupingSpec",
    "GroupingResult",
    "SiteArcSample",
    "grouping_spec",
    "boundary_is_closed",
    "bin_index_for_arc",
    "smooth_arc_mass",
    "smooth_arc_density",
    "total_variation_distance",
    "edge_share_percent",
    "analyze_grouping",
    "load_site_arc_samples",
    "grouping_table_dict",
    "write_curves_csv",
]


class ArcByGroupError(RuntimeError):
    """沿弧長分組曲線管線的資料契約或一致性錯誤，一律失敗即停。"""


ARC_GROUPINGS: tuple[str, ...] = ("class", "season", "tide")
"""三種分組方式的代號（輸出表與圖檔名使用）。"""

ARC_EDGE_ORDER: tuple[str, ...] = ("south", "east", "north", "west")
"""輸出表中各邊的固定欄位順序：南、東、北、西。"""

_REQUIRED_POSTPROCESS_TABLES: tuple[str, ...] = (
    "departures",
    "boundary_departure_along_arc_by_site",
    "boundary_edge_intervals_by_site",
    "footprint_geometry_by_site",
)


# ── 參數 ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ArcByGroupParams:
    """一次分析的參數；全部欄位在建構時驗證。

    ``sigma_km``：高斯核頻寬（公里）。``n_grid``：弧長等分格數（格寬 = 邊界全長 / n_grid，
    預設 2000 格約 0.25 km；曲線輸出與總變異距離都在這組格點上）。``n_resamples``：每個
    雜訊基準的重抽次數。``seed``：亂數種子（實際種子再混入站代號與分組代號，使各站各分組
    獨立且可重現）。``closure_tolerance_m``：判定折線首尾閉合的距離容許值（公尺）。
    ``min_group_particles``：組別穿越粒子數少於此值時不分析（TVD 與雜訊基準不可靠），並在
    輸出中列為略過的組別。
    """

    sigma_km: float = 5.0
    n_grid: int = 2000
    n_resamples: int = 200
    seed: int = 20261014
    closure_tolerance_m: float = 100.0
    min_group_particles: int = 50

    def __post_init__(self) -> None:
        if not math.isfinite(self.sigma_km) or self.sigma_km <= 0:
            raise ArcByGroupError("sigma_km 必須是有限正數（公里）")
        if self.n_grid < 100:
            raise ArcByGroupError("n_grid 至少 100")
        if self.n_resamples < 1:
            raise ArcByGroupError("n_resamples 必須是正整數")
        if not math.isfinite(self.closure_tolerance_m) or self.closure_tolerance_m < 0:
            raise ArcByGroupError("closure_tolerance_m 必須是有限非負數（公尺）")
        if self.min_group_particles < 2:
            raise ArcByGroupError("min_group_particles 至少 2（對半切需要至少兩顆）")


# ── 分組定義 ─────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class GroupingSpec:
    """一種分組方式的定義。

    ``key``：分組代號。``sample_attr``：:class:`SiteArcSample` 上存放組別標籤的屬性名稱。
    ``source_column``：逐粒子表 ``departures.parquet`` 的來源欄位。``title_zh``：中文名稱。
    ``canonical_order``：固定的組別順序（類別分組為 ``None``，改依實測沉降速度排序）。
    ``paired_release``：各組是否共用同一批釋放點（類別分組為真，用配對置換；其餘用事件置換）。
    """

    key: str
    sample_attr: str
    source_column: str
    title_zh: str
    canonical_order: tuple[str, ...] | None
    paired_release: bool


_SEASON_LABELS_ZH: Mapping[str, str] = {
    "DJF": "冬季（12–2 月）",
    "MAM": "春季（3–5 月）",
    "JJA": "夏季（6–8 月）",
    "SON": "秋季（9–11 月）",
}
_TIDE_LABELS_ZH: Mapping[str, str] = {
    "spring_proxy": "大潮",
    "neap_proxy": "小潮",
    "unknown": "未分類",
}

_GROUPING_SPECS: Mapping[str, GroupingSpec] = {
    "class": GroupingSpec("class", "material_id", "material_id", "沉降速度類別", None, True),
    "season": GroupingSpec(
        "season", "season", "deposition_season", "沉底時刻的季節", ("DJF", "MAM", "JJA", "SON"), False
    ),
    "tide": GroupingSpec(
        "tide",
        "tide",
        "deposition_tide_class",
        "沉底時刻的大潮／小潮",
        ("spring_proxy", "neap_proxy", "unknown"),
        False,
    ),
}


def grouping_spec(key: str) -> GroupingSpec:
    """依代號取得分組定義；不認得的代號拋出 :class:`ArcByGroupError`。"""

    try:
        return _GROUPING_SPECS[key]
    except KeyError as exc:
        raise ArcByGroupError(f"未知的分組代號：{key!r}（合法值：{ARC_GROUPINGS}）") from exc


def group_label_zh(grouping: str, group_id: str) -> str:
    """組別的圖面顯示名稱。類別走 :mod:`display_names` 對照表，其餘用本模組的固定名稱。"""

    if grouping == "class":
        return material_display_name_zh(group_id)
    if grouping == "season":
        return _SEASON_LABELS_ZH.get(group_id, group_id)
    return _TIDE_LABELS_ZH.get(group_id, group_id)


# ── 純數值函式（可獨立測試）─────────────────────────────────────────────


def boundary_is_closed(vertices_lonlat: np.ndarray | None, *, tolerance_m: float = 100.0) -> tuple[bool, float]:
    """判斷開放邊界折線是否首尾閉合；回傳 ``(是否閉合, 首尾距離公尺)``。

    ``vertices_lonlat`` 形狀 ``(n, >=2)``，前兩欄為經度、緯度（度），依折線頂點順序排列。
    首尾距離用球面大圓距離（半徑 6371.0088 km）。少於 3 個頂點、或為 ``None``，視為未閉合
    （距離回傳 ``nan``）。多段折線（多個 part）不在此函式處理：呼叫端遇到多段一律視為未閉合。
    """

    if vertices_lonlat is None:
        return False, float("nan")
    pts = np.asarray(vertices_lonlat, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] < 2:
        return False, float("nan")
    lon1, lat1 = np.radians(pts[0, 0]), np.radians(pts[0, 1])
    lon2, lat2 = np.radians(pts[-1, 0]), np.radians(pts[-1, 1])
    a = math.sin((lat2 - lat1) / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2.0) ** 2
    gap_m = 2.0 * 6371008.8 * math.asin(min(1.0, math.sqrt(a)))
    return bool(gap_m <= tolerance_m), float(gap_m)


def bin_index_for_arc(s_km: np.ndarray, length_km: float, n_grid: int, *, closed: bool) -> np.ndarray:
    """弧長（公里）→ 格編號（0 到 ``n_grid - 1``）。

    閉合折線：弧長恰等於全長的點與弧長 0 是同一點，編號取模回到 0；未閉合折線：超出
    ``[0, 全長]`` 的極小數值誤差夾回端點格。
    """

    dx = length_km / n_grid
    idx = np.floor(np.asarray(s_km, dtype=np.float64) / dx).astype(np.int64)
    if closed:
        return np.mod(idx, n_grid)
    return np.clip(idx, 0, n_grid - 1)


def _smooth_counts_to_mass(counts: np.ndarray, *, sigma_bins: float, closed: bool) -> np.ndarray:
    """格計數 → 平滑後的質量向量（總和為 1）。閉合用環狀接續，未閉合用鏡射。"""

    total = float(np.sum(counts))
    if total <= 0:
        raise ArcByGroupError("沒有任何粒子，無法計算平滑分布")
    mode = "wrap" if closed else "reflect"
    smooth = gaussian_filter1d(counts.astype(np.float64), sigma_bins, mode=mode, truncate=4.0)
    smooth = np.maximum(smooth, 0.0)
    return smooth / smooth.sum()


def smooth_arc_mass(
    s_km: np.ndarray, length_km: float, *, sigma_km: float, n_grid: int, closed: bool
) -> np.ndarray:
    """一組穿越粒子的弧長位置 → 平滑後的質量向量（長度 ``n_grid``、總和 1）。"""

    idx = bin_index_for_arc(s_km, length_km, n_grid, closed=closed)
    counts = np.bincount(idx, minlength=n_grid)
    return _smooth_counts_to_mass(counts, sigma_bins=sigma_km / (length_km / n_grid), closed=closed)


def smooth_arc_density(
    s_km: np.ndarray, length_km: float, *, sigma_km: float, n_grid: int, closed: bool
) -> tuple[np.ndarray, np.ndarray]:
    """回傳 ``(格中心弧長 km, 相對來源權重 %／km)``；曲線對弧長的積分恆為 100%。"""

    dx = length_km / n_grid
    mass = smooth_arc_mass(s_km, length_km, sigma_km=sigma_km, n_grid=n_grid, closed=closed)
    centers = (np.arange(n_grid) + 0.5) * dx
    return centers, 100.0 * mass / dx


def total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    """兩個質量向量的總變異距離：``0.5 * sum(|p - q|)``，範圍 0 到 1。

    ``p``、``q`` 必須同長，且各自總和相同（通常都是 1）。完全重合為 0，完全不重疊為 1。
    """

    pa_ = np.asarray(p, dtype=np.float64)
    qa_ = np.asarray(q, dtype=np.float64)
    if pa_.shape != qa_.shape:
        raise ArcByGroupError("總變異距離的兩個向量長度必須相同")
    return 0.5 * float(np.abs(pa_ - qa_).sum())


def edge_share_percent(s_km: np.ndarray, edges: Sequence[tuple[float, float]]) -> np.ndarray:
    """各邊占比（%）：落在 ``[起點, 終點)`` 的粒子數占全部的百分比；最後一邊含終點。

    使用未平滑的原始粒子位置，分母是傳入的全部粒子；沒有粒子時回傳全 0。
    """

    s = np.asarray(s_km, dtype=np.float64)
    out = np.zeros(len(edges), dtype=np.float64)
    if s.size == 0:
        return out
    last = len(edges) - 1
    for i, (a, b) in enumerate(edges):
        mask = (s >= a) & ((s <= b + 1.0e-9) if i == last else (s < b))
        out[i] = 100.0 * float(np.count_nonzero(mask)) / s.size
    return out


# ── 單站資料 ─────────────────────────────────────────────────────────────


@dataclass(slots=True)
class SiteArcSample:
    """單站的穿越粒子樣本與邊界幾何（只含「穿越開放邊界且有弧長位置」的粒子）。

    ``s_km``：沿邊界折線自起點量起的弧長（公里）。``material_id`` / ``season`` / ``tide``：
    逐粒子組別標籤（``season`` 與 ``tide`` 來自 ``deposition_*`` 欄位，缺欄位時為 ``None``）。
    ``velocity_mps``：該粒子實際使用的沉降速度（負值，m/s）。``release_code``：釋放點編號
    （受體 × 到達時刻 × 成員，同值代表同一釋放點，供類別配對置換）。``event_code``：沉底事件
    編號（到達時刻 × 成員）。``edges``：各邊 ``(名稱, 起點 km, 終點 km)``；``corners``：各轉角
    ``(名稱, 弧長 km)``。``closed`` / ``closure_gap_m``：折線首尾是否閉合與首尾距離。
    ``n_departure_rows``：該站逐粒子表總列數；``n_crossed``：穿越且有弧長的粒子數。
    """

    site_id: str
    length_km: float
    closed: bool
    closure_gap_m: float
    edges: tuple[tuple[str, float, float], ...]
    corners: tuple[tuple[str, float], ...]
    s_km: np.ndarray
    material_id: np.ndarray
    velocity_mps: np.ndarray
    season: np.ndarray | None
    tide: np.ndarray | None
    release_code: np.ndarray
    event_code: np.ndarray
    n_departure_rows: int
    n_crossed: int
    arc_table_crossed_total: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def end_treatment(self) -> str:
        """端點處理方式的代號：閉合為 ``circular``，未閉合為 ``reflect``。"""

        return "circular" if self.closed else "reflect"


def _codes(*arrays: np.ndarray) -> np.ndarray:
    """把幾個字串（或數值）欄位的組合編成連續整數編號（相同組合同一編號）。"""

    joined = np.asarray(arrays[0]).astype(str)
    for extra in arrays[1:]:
        joined = np.char.add(np.char.add(joined, "|"), np.asarray(extra).astype(str))
    _, inverse = np.unique(joined, return_inverse=True)
    return inverse.astype(np.int64)


def _read_table(postprocess_dir: Path, stem: str, columns: Sequence[str] | None = None) -> pa.Table:
    path = postprocess_dir / f"{stem}.parquet"
    if not path.is_file():
        raise ArcByGroupError(f"後處理目錄缺少必要輸入：{stem}.parquet")
    return pq.read_table(path, columns=list(columns) if columns is not None else None)


def load_site_arc_samples(
    postprocess_dir: Path | str,
    site_ids: Sequence[str] | None = None,
    *,
    closure_tolerance_m: float = 100.0,
) -> dict[str, SiteArcSample]:
    """讀取後處理目錄，建立各站的 :class:`SiteArcSample`。

    ``site_ids`` 為 ``None`` 時輸出逐粒子表中出現的全部站；指定了不存在的站則拋出錯誤
    （避免打錯站名被靜默忽略）。只讀不寫。
    """

    root = Path(postprocess_dir)
    for stem in _REQUIRED_POSTPROCESS_TABLES:
        if not (root / f"{stem}.parquet").is_file():
            raise ArcByGroupError(f"後處理目錄缺少必要輸入：{stem}.parquet（需用 --with-footprint 產生）")

    schema_names = set(pq.read_schema(root / "departures.parquet").names)
    base_columns = [
        "study_site_id",
        "material_id",
        "settling_velocity_mps",
        "crossed_interest_region_boundary",
        "boundary_arc_length_position_m",
        "receptor_id",
        "arrival_time_id",
        "member_id",
    ]
    missing = [c for c in base_columns if c not in schema_names]
    if missing:
        raise ArcByGroupError(f"departures.parquet 缺少必要欄位：{missing}")
    optional = [c for c in ("deposition_season", "deposition_tide_class") if c in schema_names]
    dep = _read_table(root, "departures", base_columns + optional)

    sites_in_data = sorted(set(pc.unique(dep.column("study_site_id")).to_pylist()))
    if site_ids is None:
        wanted = sites_in_data
    else:
        wanted = list(site_ids)
        unknown = [s for s in wanted if s not in sites_in_data]
        if unknown:
            raise ArcByGroupError(f"指定的站不在逐粒子表中：{unknown}（現有：{sites_in_data}）")

    arc_rows = _read_table(root, "boundary_departure_along_arc_by_site").to_pylist()
    edge_rows = _read_table(root, "boundary_edge_intervals_by_site").to_pylist()
    geom_rows = [r for r in _read_table(root, "footprint_geometry_by_site").to_pylist() if r["element"] == "open_boundary"]

    site_col = dep.column("study_site_id").to_numpy(zero_copy_only=False)
    arc_all = dep.column("boundary_arc_length_position_m").to_numpy(zero_copy_only=False).astype(np.float64)
    crossed_all = dep.column("crossed_interest_region_boundary").to_numpy(zero_copy_only=False).astype(bool)

    result: dict[str, SiteArcSample] = {}
    for site in wanted:
        arcs = sorted((r for r in arc_rows if r["study_site_id"] == site), key=lambda r: r["bin_index"])
        if not arcs:
            raise ArcByGroupError(f"boundary_departure_along_arc_by_site 沒有站 {site} 的資料")
        length_km = float(arcs[0]["boundary_length_km"])
        arc_total = int(sum(int(r["crossed_particle_count"]) for r in arcs))

        edges_site = sorted((r for r in edge_rows if r["study_site_id"] == site), key=lambda r: r["s_start_km"])
        edges = tuple(
            (str(r["name"]), float(r["s_start_km"]), float(r["s_end_km"])) for r in edges_site if r["kind"] == "edge"
        )
        corners = tuple((str(r["name"]), float(r["s_start_km"])) for r in edges_site if r["kind"] == "corner")
        if not edges:
            raise ArcByGroupError(f"boundary_edge_intervals_by_site 沒有站 {site} 的邊區間")

        parts: dict[int, list[Mapping[str, Any]]] = {}
        for r in geom_rows:
            if r["study_site_id"] == site:
                parts.setdefault(int(r["part_index"]), []).append(r)
        if len(parts) == 1:
            only = sorted(next(iter(parts.values())), key=lambda r: r["vertex_index"])
            vertices = np.asarray([[r["lon_deg"], r["lat_deg"]] for r in only], dtype=np.float64)
            closed, gap_m = boundary_is_closed(vertices, tolerance_m=closure_tolerance_m)
        else:
            # 沒有折線或有多段：無法保證首尾相接，保守視為未閉合（改用鏡射）。
            closed, gap_m = False, float("nan")

        mask = (site_col == site) & crossed_all & np.isfinite(arc_all)
        sel = np.flatnonzero(mask)
        n_rows = int(np.count_nonzero(site_col == site))
        if int(sel.size) != arc_total:
            raise ArcByGroupError(
                f"站 {site}：逐粒子表的穿越粒子數 {int(sel.size)} 與弧長分布表合計 {arc_total} 不一致，fail-closed"
            )
        sub = dep.take(pa.array(sel))

        def col(name: str, _sub: pa.Table = sub) -> np.ndarray:
            return _sub.column(name).to_numpy(zero_copy_only=False)

        s_km = col("boundary_arc_length_position_m").astype(np.float64) / 1000.0
        if s_km.size and (s_km.min() < -1.0e-6 or s_km.max() > length_km + 1.0e-6):
            raise ArcByGroupError(f"站 {site}：弧長位置超出 [0, 邊界全長]，fail-closed")
        receptor = col("receptor_id")
        arrival = col("arrival_time_id")
        member = col("member_id")
        result[site] = SiteArcSample(
            site_id=site,
            length_km=length_km,
            closed=closed,
            closure_gap_m=gap_m,
            edges=edges,
            corners=corners,
            s_km=s_km,
            material_id=col("material_id").astype(str),
            velocity_mps=col("settling_velocity_mps").astype(np.float64),
            season=col("deposition_season").astype(str) if "deposition_season" in optional else None,
            tide=col("deposition_tide_class").astype(str) if "deposition_tide_class" in optional else None,
            release_code=_codes(receptor, arrival, member),
            event_code=_codes(arrival, member),
            n_departure_rows=n_rows,
            n_crossed=int(sel.size),
            arc_table_crossed_total=arc_total,
        )
    return result


# ── 單站單分組分析 ───────────────────────────────────────────────────────


@dataclass(slots=True)
class GroupingResult:
    """單站、單一分組方式的完整分析結果。

    ``edges`` / ``corners``：各邊 ``(名稱, 起點 km, 終點 km)`` 與各轉角 ``(名稱, 弧長 km)``，供繪圖畫底色與虛線。
    陣列的第一軸都是組別（順序同 ``group_ids``）。``density_pct_per_km``：形狀
    ``(組數, n_grid)``，每列積分 100%。``edge_share_pct``：形狀 ``(組數, 邊數)``。
    ``tvd``：``(組數, 組數)`` 的對稱矩陣，對角為 0。``pair_rows``：每一對的觀測 TVD 與置換
    基準。``noise``：同組對半切基準。``skipped``：因粒子數不足而略過的組別。
    """

    grouping: str
    site_id: str
    params: ArcByGroupParams
    length_km: float
    closed: bool
    closure_gap_m: float
    end_treatment: str
    edge_names: tuple[str, ...]
    edges: tuple[tuple[str, float, float], ...]
    corners: tuple[tuple[str, float], ...]
    group_ids: tuple[str, ...]
    labels_zh: tuple[str, ...]
    n_crossed: np.ndarray
    velocity_summary: tuple[Mapping[str, float], ...] | None
    x_km: np.ndarray
    density_pct_per_km: np.ndarray
    density_all_pct_per_km: np.ndarray
    n_all: int
    edge_share_pct: np.ndarray
    edge_share_all_pct: np.ndarray
    tvd: np.ndarray
    pair_rows: tuple[Mapping[str, Any], ...]
    noise: Mapping[str, Any]
    label_null_kind: str
    skipped: tuple[Mapping[str, Any], ...]

    def max_density(self) -> float:
        """所有組別曲線（不含合併曲線）的最大值，供決定縱軸範圍。"""

        return float(self.density_pct_per_km.max()) if self.density_pct_per_km.size else 0.0


def _stable_seed(seed: int, *labels: str) -> list[int]:
    """把使用者種子與站、分組代號混成 ``numpy`` 可用的種子序列（跨平台、跨行程穩定）。"""

    return [int(seed), *[zlib.crc32(label.encode("utf-8")) for label in labels]]


def _percentiles(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def analyze_grouping(
    sample: SiteArcSample, grouping: str, params: ArcByGroupParams
) -> GroupingResult | None:
    """對單站做一種分組的曲線、各邊占比、組間 TVD 與雜訊基準。

    該分組需要的標籤欄位不存在（例如舊版後處理沒有 ``deposition_*``）、或可分析的組別少於
    兩組時，回傳 ``None``（呼叫端應在 manifest 記錄原因）。
    """

    spec = grouping_spec(grouping)
    labels = getattr(sample, spec.sample_attr)
    if labels is None:
        return None
    n_grid = params.n_grid
    length_km = sample.length_km
    closed = sample.closed
    sigma_bins = params.sigma_km / (length_km / n_grid)
    bins = bin_index_for_arc(sample.s_km, length_km, n_grid, closed=closed)

    def mass_of(sel: np.ndarray) -> np.ndarray:
        return _smooth_counts_to_mass(np.bincount(bins[sel], minlength=n_grid), sigma_bins=sigma_bins, closed=closed)

    present = [str(g) for g in np.unique(labels)]
    skipped: list[dict[str, Any]] = []
    kept: dict[str, np.ndarray] = {}
    for g in present:
        sel = np.flatnonzero(labels == g)
        if sel.size < params.min_group_particles:
            skipped.append({"group_id": g, "n_crossed": int(sel.size), "reason": "穿越粒子數低於下限"})
        else:
            kept[g] = sel

    velocity_summary: tuple[Mapping[str, float], ...] | None = None
    if spec.canonical_order is not None:
        order = [g for g in spec.canonical_order if g in kept] + sorted(g for g in kept if g not in spec.canonical_order)
    else:
        med = {g: float(np.median(np.abs(sample.velocity_mps[kept[g]]))) for g in kept}
        order = sorted(kept, key=lambda g: (med[g], g))
        velocity_summary = tuple(
            {
                "abs_median_mps": med[g],
                "abs_min_mps": float(np.min(np.abs(sample.velocity_mps[kept[g]]))),
                "abs_max_mps": float(np.max(np.abs(sample.velocity_mps[kept[g]]))),
                "distinct_values": int(np.unique(sample.velocity_mps[kept[g]]).size),
            }
            for g in order
        )
    if len(order) < 2:
        return None

    n_groups = len(order)
    centers = (np.arange(n_grid) + 0.5) * (length_km / n_grid)
    dx = length_km / n_grid
    masses = np.stack([mass_of(kept[g]) for g in order])
    all_sel = np.arange(sample.s_km.size)
    mass_all = mass_of(all_sel)
    edge_bounds = [(a, b) for (_n, a, b) in sample.edges]
    edge_names = tuple(n for (n, _a, _b) in sample.edges)
    edge_share = np.stack([edge_share_percent(sample.s_km[kept[g]], edge_bounds) for g in order])
    edge_share_all = edge_share_percent(sample.s_km, edge_bounds)

    tvd = np.zeros((n_groups, n_groups))
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            tvd[i, j] = tvd[j, i] = total_variation_distance(masses[i], masses[j])

    rng = np.random.default_rng(_stable_seed(params.seed, sample.site_id, grouping))
    ev = sample.event_code

    # 基準 1、2：同組隨機對半切（粒子層級、沉底事件層級）。
    half_particle: dict[str, dict[str, float]] = {}
    half_event: dict[str, dict[str, float]] = {}
    pooled_particle: list[float] = []
    pooled_event: list[float] = []
    for g in order:
        sel = kept[g]
        h = sel.size // 2
        vals = []
        for _ in range(params.n_resamples):
            perm = rng.permutation(sel.size)
            vals.append(total_variation_distance(mass_of(sel[perm[:h]]), mass_of(sel[perm[h : 2 * h]])))
        half_particle[g] = _percentiles(vals)
        pooled_particle += vals

        ev_sel = ev[sel]
        uniq, inv = np.unique(ev_sel, return_inverse=True)
        if uniq.size >= 2:
            vals_e = []
            for _ in range(params.n_resamples):
                pick = np.zeros(uniq.size, dtype=bool)
                pick[rng.permutation(uniq.size)[: uniq.size // 2]] = True
                m = pick[inv]
                vals_e.append(total_variation_distance(mass_of(sel[m]), mass_of(sel[~m])))
            half_event[g] = _percentiles(vals_e)
            pooled_event += vals_e
    noise: dict[str, Any] = {
        "half_split_particle": {"per_group": half_particle, "pooled": _percentiles(pooled_particle)},
        "half_split_event": {
            "per_group": half_event,
            "pooled": _percentiles(pooled_event) if pooled_event else None,
            "event_unit": "同一到達時刻且同一系集成員（共用同一個沉底時刻）",
        },
        "note_zh": (
            "對半切後每半只有一半樣本，TVD 比全樣本對全樣本略大，為偏保守的雜訊基準；"
            "事件層級以共用沉底時刻的粒子為不可分割單位，反映實際獨立單位數較少的情形。"
        ),
    }

    # 基準 3：組別標籤置換。
    pair_rows: list[dict[str, Any]] = []
    null_kind = "paired_release_swap" if spec.paired_release else "event_label_permutation"
    rel = sample.release_code
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            ga, gb = order[i], order[j]
            sa, sb = kept[ga], kept[gb]
            vals_n: list[float] = []
            if spec.paired_release:
                common, ia, ib = np.intersect1d(rel[sa], rel[sb], assume_unique=False, return_indices=True)
                # np.intersect1d 在重複值時只回傳第一個索引；釋放點在同一類別內唯一，故一一對應。
                ca, cb = sa[ia], sb[ib]
                only_a = np.setdiff1d(sa, ca, assume_unique=True)
                only_b = np.setdiff1d(sb, cb, assume_unique=True)
                for _ in range(params.n_resamples):
                    swap = rng.random(ca.size) < 0.5
                    a_idx = np.concatenate([np.where(swap, cb, ca), only_a])
                    b_idx = np.concatenate([np.where(swap, ca, cb), only_b])
                    vals_n.append(total_variation_distance(mass_of(a_idx), mass_of(b_idx)))
                n_pairs_shared = int(common.size)
            else:
                pool = np.concatenate([sa, sb])
                uniq, inv = np.unique(ev[pool], return_inverse=True)
                n_ev_a = int(np.unique(ev[sa]).size)
                for _ in range(params.n_resamples):
                    pick = np.zeros(uniq.size, dtype=bool)
                    pick[rng.permutation(uniq.size)[:n_ev_a]] = True
                    m = pick[inv]
                    vals_n.append(total_variation_distance(mass_of(pool[m]), mass_of(pool[~m])))
                n_pairs_shared = 0
            stats = _percentiles(vals_n)
            obs = float(tvd[i, j])
            pair_rows.append(
                {
                    "group_a": ga,
                    "group_b": gb,
                    "tvd": obs,
                    "null_kind": null_kind,
                    "null_mean": stats["mean"],
                    "null_p95": stats["p95"],
                    "null_p99": stats["p99"],
                    "tvd_over_null_mean": obs / stats["mean"] if stats["mean"] > 0 else float("nan"),
                    "exceeds_null_p95": bool(obs > stats["p95"]),
                    "exceeds_null_p99": bool(obs > stats["p99"]),
                    "shared_release_points": n_pairs_shared,
                }
            )

    return GroupingResult(
        grouping=grouping,
        site_id=sample.site_id,
        params=params,
        length_km=length_km,
        closed=closed,
        closure_gap_m=sample.closure_gap_m,
        end_treatment=sample.end_treatment,
        edge_names=edge_names,
        edges=sample.edges,
        corners=sample.corners,
        group_ids=tuple(order),
        labels_zh=tuple(group_label_zh(grouping, g) for g in order),
        n_crossed=np.asarray([kept[g].size for g in order], dtype=np.int64),
        velocity_summary=velocity_summary,
        x_km=centers,
        density_pct_per_km=100.0 * masses / dx,
        density_all_pct_per_km=100.0 * mass_all / dx,
        n_all=int(sample.s_km.size),
        edge_share_pct=edge_share,
        edge_share_all_pct=edge_share_all,
        tvd=tvd,
        pair_rows=tuple(pair_rows),
        noise=noise,
        label_null_kind=null_kind,
        skipped=tuple(skipped),
    )


# ── 輸出表 ───────────────────────────────────────────────────────────────


def grouping_table_dict(result: GroupingResult, *, case_id: str) -> dict[str, Any]:
    """整理成可寫入 JSON／manifest 的表格字典（不含曲線格點，曲線另存 CSV）。

    內容：邊界與參數、各組粒子數與各邊占比、組間兩兩 TVD 與置換基準、同組對半切雜訊基準，
    以及摘要（最大 TVD 的組對、超過置換基準第 95／99 百分位的組對數）。
    """

    dx = result.length_km / result.params.n_grid
    groups = []
    for k, gid in enumerate(result.group_ids):
        rec: dict[str, Any] = {
            "group_id": gid,
            "label_zh": result.labels_zh[k],
            "n_crossed": int(result.n_crossed[k]),
            "edge_share_pct": {n: float(result.edge_share_pct[k, e]) for e, n in enumerate(result.edge_names)},
            "curve_integral_pct": float(result.density_pct_per_km[k].sum() * dx),
        }
        if result.velocity_summary is not None:
            rec["settling_speed_mps"] = dict(result.velocity_summary[k])
        groups.append(rec)
    pairs = [dict(r) for r in result.pair_rows]
    worst = max(pairs, key=lambda r: r["tvd"]) if pairs else None
    pooled_p = result.noise["half_split_particle"]["pooled"]
    pooled_e = result.noise["half_split_event"]["pooled"]
    summary = {
        "pair_count": len(pairs),
        "max_tvd": worst["tvd"] if worst else None,
        "max_tvd_pair": [worst["group_a"], worst["group_b"]] if worst else None,
        "max_tvd_pair_labels_zh": (
            [group_label_zh(result.grouping, worst["group_a"]), group_label_zh(result.grouping, worst["group_b"])]
            if worst
            else None
        ),
        "max_tvd_pair_null_p95": worst["null_p95"] if worst else None,
        "mean_tvd": float(np.mean([r["tvd"] for r in pairs])) if pairs else None,
        "pairs_exceeding_null_p95": int(sum(r["exceeds_null_p95"] for r in pairs)),
        "pairs_exceeding_null_p99": int(sum(r["exceeds_null_p99"] for r in pairs)),
        "null_kind": result.label_null_kind,
        "null_p95_pooled_mean_over_pairs": float(np.mean([r["null_p95"] for r in pairs])) if pairs else None,
        "half_split_particle_pooled": pooled_p,
        "half_split_event_pooled": pooled_e,
    }
    return {
        "case_id": case_id,
        "site_id": result.site_id,
        "grouping": result.grouping,
        "grouping_title_zh": grouping_spec(result.grouping).title_zh,
        "source_column": grouping_spec(result.grouping).source_column,
        "boundary": {
            "length_km": result.length_km,
            "closed": result.closed,
            "closure_gap_m": result.closure_gap_m,
            "end_treatment": result.end_treatment,
        },
        "parameters": {
            "sigma_km": result.params.sigma_km,
            "n_grid": result.params.n_grid,
            "grid_step_km": dx,
            "n_resamples": result.params.n_resamples,
            "seed": result.params.seed,
            "min_group_particles": result.params.min_group_particles,
            "closure_tolerance_m": result.params.closure_tolerance_m,
        },
        "edge_names": list(result.edge_names),
        "groups": groups,
        "all_groups_combined": {
            "n_crossed": result.n_all,
            "edge_share_pct": {n: float(result.edge_share_all_pct[e]) for e, n in enumerate(result.edge_names)},
        },
        "skipped_groups": [dict(s) for s in result.skipped],
        "pairwise_tvd": pairs,
        "noise_baselines": result.noise,
        "summary": summary,
    }


def write_curves_csv(result: GroupingResult, path: Path) -> None:
    """寫出各組曲線格點值（寬表）：``x_km`` ＋ 每組一欄（單位 %／km）＋ 合併曲線欄。

    以排他方式建立檔案（已存在即失敗）。欄名使用組別代號，組別的中文顯示名稱記在第二列
    註解之外的 JSON 表，避免 CSV 標頭含逗號或特殊字元。
    """

    header = ["x_km", *[f"weight_pct_per_km__{g}" for g in result.group_ids], "weight_pct_per_km__all_groups"]
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for k in range(result.x_km.size):
            writer.writerow(
                [f"{result.x_km[k]:.6f}"]
                + [f"{result.density_pct_per_km[g, k]:.8g}" for g in range(len(result.group_ids))]
                + [f"{result.density_all_pct_per_km[k]:.8g}"]
            )
