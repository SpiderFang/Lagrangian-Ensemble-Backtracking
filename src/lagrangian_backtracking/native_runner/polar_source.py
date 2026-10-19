"""圓形來源方位圖（WP-60）的計算核心：來源方位、36 扇區比例、終止類別與可區辨性檢定。

這個模組回答什麼問題
--------------------
逆向系集把虛擬質點從受體（研究點）往回追，每顆質點最後會以某種方式「終止」：穿越流場範圍的
開放邊界、擱淺在海岸、回溯到上限仍在域內、沉底等。本模組把每顆質點的「來源點」換算成
**相對於研究點錨點的方位角**（自正北順時針，單位度），再把方位角分到 36 個各寬 10° 的扇區，
求各扇區占該組虛擬質點的百分比。這些百分比連成封閉的極座標折線，就是 PI 構想的
「圓形來源方位圖」（繪圖見 :mod:`polar_source_figures`）。

來源點的定義（逐顆質點）
------------------------
1. **穿越開放邊界**（狀態碼 1，``NATIVE_STATUS_LEFT_FLOW_DOMAIN``）：來源點是它的開放邊界離開點
   （欄位 ``departure_lon_deg``、``departure_lat_deg``）。
2. **擱淺**（狀態碼 3，``NATIVE_STATUS_STRANDED``）：質點卡在海岸（所在元素被判定為乾）。PI 的指示是
   「內插到陸地，表示它確實沉積在海岸」：以擱淺位置（同樣是 ``departure_lon_deg``、``departure_lat_deg``，
   即質點停止時的位置，已在海岸或陸地一側）計算相對錨點的方位角並計入該方位扇區；
   繪圖時另以小標記畫出擱淺點，讓海岸來源可辨認。
3. **其他終止類別**（達回溯上限仍在域內、沉底、流場缺值、數值失敗、回溯窗外沉底）：沒有可靠的
   「來源方向」，**不計入 36 個扇區**，但仍留在分母（分母永遠是該組全部虛擬質點），
   並在統計表中逐類記錄其比例。因此 36 扇區合計 =（穿越 + 擱淺）的比例，不一定是 100%。

方位角算法（程式註解即設計說明）
--------------------------------
使用**大圓初始方位角**（球面上由錨點朝來源點出發的航向，``atan2(sinΔλ·cosφ₂, cosφ₁·sinφ₂ − sinφ₁·cosφ₂·cosΔλ)``），
自正北 0° 順時針增加，結果規範到 [0°, 360°)。選它的理由：定義與地圖投影無關、全向量化、不需要挑選
局部平面的參考緯度。與「局部等距離平面近似」（地圖上直接量角度）在本研究的距離（約 30–110 km）
相差不到 1°，遠小於 10° 的扇區寬，不影響分扇區結果。正東（同緯度、經度增加的赤道上）恰為 90°。

扇區定義
--------
36 個扇區，編號 0–35，扇區 k 的中心方位是 10°·k，涵蓋 [10°·k − 5°, 10°·k + 5°)（左閉右開）。
所以正北扇區（編號 0，中心 0°）涵蓋 [355°, 360°) ∪ [0°, 5°)；4.99° 屬編號 0，5.00° 屬編號 1，
355.00° 屬編號 0，354.99° 屬編號 35。

可區辨性檢定（沿用 WP-55 的雜訊基準概念）
------------------------------------------
每組虛擬質點的方位分布寫成 37 格向量：36 個扇區加第 37 格「未計入」（分母為該組全部質點，
因此向量總和為 1，與圖上折線畫的比例一致）。兩組分布的距離用總變異距離（total variation
distance，TVD，兩向量差絕對值總和之半，範圍 0–1），另附最大扇區差（百分點）。

雜訊基準：先前已證實抽樣不確定性主要來自「到達時刻」的選取而非系集成員，所以把同一組內的虛擬質點
依「到達時刻單元」（``arrival_time_id``，同一單元的質點共用同一個到達時刻的流場而彼此相關）
整單元隨機對半切，算兩半的 TVD，重複 ``n_resamples`` 次。這是沒有任何組別差異、純抽樣雜訊的
距離分布。對半切後每半只有一半樣本，TVD 會比「全樣本對全樣本」略大（約 √2 倍），
所以這個基準偏保守（偏嚴）。同一比較的所有組合併計算後取第 95、99 百分位為門檻；
某組對的觀測 TVD 超過第 99 百分位門檻，即判定該組對的差異超過雜訊。

比較範圍：同一類別內 10 個候選速度之間、十類之間、季節之間、大小潮之間；季節與大小潮使用
沉底時刻計算的 ``deposition_season``、``deposition_tide_class``（與 WP-55 相同，
不使用與軌跡無物理關聯的 ``observation_*`` 舊標籤）。

用詞限制：結果是「條件式來源足跡／相對來源權重」，不是絕對來源機率或因果歸因
（見 ``AGENTS.md`` 科學與資料契約）。
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

from . import arc_by_group as abg
from .backward_driver import (
    NATIVE_STATUS_DEPOSITED,
    NATIVE_STATUS_FORCING_MISSING,
    NATIVE_STATUS_LEFT_FLOW_DOMAIN,
    NATIVE_STATUS_MAX_BACKTRACK,
    NATIVE_STATUS_NUMERICAL_FAILURE,
    NATIVE_STATUS_PRE_WINDOW_DEPOSITION,
    NATIVE_STATUS_STRANDED,
)
from .display_names import material_display_name_zh

__all__ = [
    "N_SECTORS",
    "SECTOR_WIDTH_DEG",
    "N_BINS",
    "UNCOUNTED_BIN",
    "POLAR_GROUPINGS",
    "STATUS_NAME_ZH",
    "KIND_CROSSED",
    "KIND_STRANDED",
    "KIND_OTHER",
    "PolarSourceError",
    "PolarParams",
    "SitePolarSample",
    "GroupSet",
    "Comparison",
    "SiteAnalysis",
    "bearing_deg",
    "sector_index",
    "sector_center_deg",
    "classify_sources",
    "load_site_polar_samples",
    "build_group_set",
    "analyze_site",
    "tvd_and_max_sector",
    "half_split_noise",
    "write_sector_long_csv",
    "write_termination_csv",
    "write_pairs_csv",
    "write_comparison_csv",
    "write_stranded_csv",
]


class PolarSourceError(RuntimeError):
    """圓形來源方位圖管線的資料契約或一致性錯誤，一律失敗即停。"""


N_SECTORS = 36
"""扇區數；扇區 k 的中心方位為 ``SECTOR_WIDTH_DEG * k`` 度。"""

SECTOR_WIDTH_DEG = 10.0
"""每個扇區的角寬（度）。"""

UNCOUNTED_BIN = N_SECTORS
"""第 37 格（編號 36）：不計入任何扇區的質點（其他終止類別），只在檢定與統計表使用。"""

N_BINS = N_SECTORS + 1

KIND_CROSSED = 0
"""來源點種類：穿越開放邊界。"""
KIND_STRANDED = 1
"""來源點種類：擱淺（計入方位扇區）。"""
KIND_OTHER = 2
"""來源點種類：其他終止類別（不計入扇區）。"""

STATUS_NAME_ZH: Mapping[int, str] = {
    NATIVE_STATUS_LEFT_FLOW_DOMAIN: "穿越開放邊界",
    NATIVE_STATUS_MAX_BACKTRACK: "達回溯上限而仍在流場範圍內",
    NATIVE_STATUS_STRANDED: "擱淺",
    NATIVE_STATUS_DEPOSITED: "沉底",
    NATIVE_STATUS_FORCING_MISSING: "流場缺值",
    NATIVE_STATUS_NUMERICAL_FAILURE: "數值失敗",
    NATIVE_STATUS_PRE_WINDOW_DEPOSITION: "固定日曆回溯窗之外的沉底",
}
"""終止狀態碼 → 中文名稱（狀態碼定義見 ``backward_driver.NATIVE_STATUS_*``）。"""

_STATUS_CODES: tuple[int, ...] = tuple(sorted(STATUS_NAME_ZH))
_OTHER_STATUS_CODES: tuple[int, ...] = tuple(
    c for c in _STATUS_CODES if c not in (NATIVE_STATUS_LEFT_FLOW_DOMAIN, NATIVE_STATUS_STRANDED)
)

POLAR_GROUPINGS: tuple[str, ...] = ("class_velocity", "class", "season", "tide")
"""四種分組：類別×候選速度（每組 1,000 顆）、類別（每組 1 萬顆）、季節、大小潮。"""

_CANONICAL_ORDER: Mapping[str, tuple[str, ...]] = {
    "season": ("DJF", "MAM", "JJA", "SON"),
    "tide": ("spring_proxy", "neap_proxy", "unknown"),
}


# ── 參數 ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class PolarParams:
    """一次分析的參數。

    ``n_resamples``：每個組的「到達時刻單元對半切」重抽次數。``seed``：亂數種子（實際種子再混入案例、
    站與組別代號，使各組獨立且可重現）。``min_units``：組內有質點的到達時刻單元少於此數時不做對半切。
    """

    n_resamples: int = 1000
    seed: int = 20261019
    min_units: int = 2

    def __post_init__(self) -> None:
        if self.n_resamples < 10:
            raise PolarSourceError("n_resamples 至少 10")
        if self.min_units < 2:
            raise PolarSourceError("min_units 至少 2（對半切需要至少兩個單元）")


# ── 純數值函式（可獨立測試）─────────────────────────────────────────────


def bearing_deg(lon0_deg: float, lat0_deg: float, lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    """錨點 ``(lon0, lat0)`` 到各目標點的大圓初始方位角（度，自正北順時針，範圍 [0, 360)）。

    公式：``atan2(sinΔλ·cosφ₂, cosφ₁·sinφ₂ − sinφ₁·cosφ₂·cosΔλ)``，Δλ 為經度差。全向量化。
    目標點與錨點完全重合時方位角無定義，這裡回傳 0（此情形在正式資料中不會出現，
    呼叫端另以有限值檢查擋下 NaN）。輸入含 NaN 時對應輸出為 NaN。
    """

    phi1 = math.radians(float(lat0_deg))
    phi2 = np.radians(np.asarray(lat_deg, dtype=np.float64))
    dlon = np.radians(np.asarray(lon_deg, dtype=np.float64) - float(lon0_deg))
    y = np.sin(dlon) * np.cos(phi2)
    x = math.cos(phi1) * np.sin(phi2) - math.sin(phi1) * np.cos(phi2) * np.cos(dlon)
    az = np.degrees(np.arctan2(y, x))
    az = np.mod(az, 360.0)
    # np.mod 對極小負數可能回傳恰為 360.0；規範回 [0, 360)。
    return np.where(az >= 360.0, 0.0, az)


def sector_index(azimuth_deg: np.ndarray) -> np.ndarray:
    """方位角（度）→ 扇區編號 0–35。

    扇區 k 涵蓋 ``[10k − 5, 10k + 5)`` 度：把方位角加 5° 後除以 10° 取下整，再對 36 取餘。
    例：355° → 0、354.99° → 35、4.99° → 0、5° → 1、359.99° → 0。
    """

    shifted = np.mod(np.asarray(azimuth_deg, dtype=np.float64) + SECTOR_WIDTH_DEG / 2.0, 360.0)
    return np.floor(shifted / SECTOR_WIDTH_DEG).astype(np.int64) % N_SECTORS


def sector_center_deg(index: int | np.ndarray) -> float | np.ndarray:
    """扇區編號 → 中心方位角（度）。"""

    return SECTOR_WIDTH_DEG * np.asarray(index) if isinstance(index, np.ndarray) else SECTOR_WIDTH_DEG * float(index)


def classify_sources(
    status: np.ndarray,
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    anchor_lonlat: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """依終止狀態與位置，求每顆質點的 ``(來源種類, 方位角, 格編號 0–36)``。

    * 狀態 1（穿越）、3（擱淺）：用離開／擱淺位置算方位角並分扇區；位置缺值或非有限時拋出錯誤
      （不得以零值替代，見 ``AGENTS.md``）。
    * 其他狀態：種類為 :data:`KIND_OTHER`，方位角為 NaN，格編號為 :data:`UNCOUNTED_BIN`。
    """

    st = np.asarray(status)
    kind = np.full(st.shape, KIND_OTHER, dtype=np.int8)
    kind[st == NATIVE_STATUS_LEFT_FLOW_DOMAIN] = KIND_CROSSED
    kind[st == NATIVE_STATUS_STRANDED] = KIND_STRANDED
    counted = kind != KIND_OTHER
    lon = np.asarray(lon_deg, dtype=np.float64)
    lat = np.asarray(lat_deg, dtype=np.float64)
    if counted.any() and not (np.all(np.isfinite(lon[counted])) and np.all(np.isfinite(lat[counted]))):
        raise PolarSourceError("穿越或擱淺的質點位置含缺值或非有限值，fail-closed")
    az = np.full(st.shape, np.nan, dtype=np.float64)
    bins = np.full(st.shape, UNCOUNTED_BIN, dtype=np.int64)
    if counted.any():
        az[counted] = bearing_deg(anchor_lonlat[0], anchor_lonlat[1], lon[counted], lat[counted])
        bins[counted] = sector_index(az[counted])
    return kind, az, bins


def tvd_and_max_sector(p: np.ndarray, q: np.ndarray) -> tuple[float, float]:
    """兩個 37 格比例向量的 ``(總變異距離, 最大扇區差百分點)``；最大扇區差只看前 36 格。"""

    d = np.abs(np.asarray(p, dtype=np.float64) - np.asarray(q, dtype=np.float64))
    return 0.5 * float(d.sum()), 100.0 * float(d[:N_SECTORS].max())


def _stable_seed(seed: int, *labels: str) -> list[int]:
    return [int(seed), *[zlib.crc32(label.encode("utf-8")) for label in labels]]


def half_split_noise(
    counts_unit_bin: np.ndarray, n_resamples: int, rng: np.random.Generator, *, min_units: int = 2
) -> tuple[np.ndarray, np.ndarray] | None:
    """同組依到達時刻單元隨機對半切的雜訊基準，全向量化。

    ``counts_unit_bin``：形狀 ``(單元數, 37)`` 的計數，列為到達時刻單元、欄為 37 格。只有「有質點的單元」
    參與切分；每次重抽隨機選出其中 ``單元數 // 2`` 個單元為第一半，其餘為第二半。
    回傳 ``(每次重抽的 TVD, 每次重抽的最大扇區差百分點)``；有質點的單元少於 ``min_units`` 時回傳 ``None``。
    """

    c = np.asarray(counts_unit_bin, dtype=np.float64)
    active = np.flatnonzero(c.sum(axis=1) > 0)
    if active.size < max(min_units, 2):
        return None
    c = c[active]
    k = active.size // 2
    scores = rng.random((n_resamples, active.size))
    ranks = np.argsort(np.argsort(scores, axis=1), axis=1)
    pick = (ranks < k).astype(np.float64)
    h1 = pick @ c
    h2 = c.sum(axis=0)[None, :] - h1
    p1 = h1 / h1.sum(axis=1, keepdims=True)
    p2 = h2 / h2.sum(axis=1, keepdims=True)
    d = np.abs(p1 - p2)
    return 0.5 * d.sum(axis=1), 100.0 * d[:, :N_SECTORS].max(axis=1)


# ── 單站樣本 ─────────────────────────────────────────────────────────────


@dataclass(slots=True)
class SitePolarSample:
    """單站全部虛擬質點的來源方位資料（所有質點，含未計入者）。

    ``anchor_lonlat``：研究點錨點（正式釋放設定的 ``anchor_lonlat``），方位角的原點。
    ``material_id``、``candidate_index``、``speed_abs_mps``：類別、該類 10 個候選速度的編號（依速度由小到大）
    與質點實際使用的沉降速度絕對值（m/s）；``class_speed_abs_mps``：類別代表速度絕對值（排序用）。
    ``season``、``tide``：沉底時刻的季節與大小潮標籤（``deposition_*``）。``unit``：到達時刻單元編號。
    ``status``：終止狀態碼。``kind``：來源點種類（穿越、擱淺、其他）。``azimuth_deg``：方位角（未計入者 NaN）。
    ``bin37``：格編號（0–35 扇區，36 未計入）。``source_lon``、``source_lat``：來源點經緯度（用於地圖上的擱淺標記）。
    ``particle_id``：質點識別碼（擱淺點清單用）。
    """

    site_id: str
    anchor_lonlat: tuple[float, float]
    n_rows: int
    material_id: np.ndarray
    candidate_index: np.ndarray
    speed_abs_mps: np.ndarray
    class_speed_abs_mps: np.ndarray
    season: np.ndarray
    tide: np.ndarray
    unit: np.ndarray
    n_units: int
    status: np.ndarray
    kind: np.ndarray
    azimuth_deg: np.ndarray
    bin37: np.ndarray
    source_lon: np.ndarray
    source_lat: np.ndarray
    particle_id: np.ndarray
    notes: list[str] = field(default_factory=list)


_REQUIRED_COLUMNS: tuple[str, ...] = (
    "particle_id",
    "study_site_id",
    "material_id",
    "settling_velocity_mps",
    "particle_settling_velocity_mps",
    "settling_candidate_index",
    "arrival_time_id",
    "final_status_code",
    "crossed_interest_region_boundary",
    "departure_lon_deg",
    "departure_lat_deg",
    "deposition_season",
    "deposition_tide_class",
)


def load_site_polar_samples(
    postprocess_dir: Path | str,
    anchors: Mapping[str, tuple[float, float]],
    site_ids: Sequence[str] | None = None,
) -> dict[str, SitePolarSample]:
    """讀取後處理目錄的逐質點表 ``departures.parquet``，建立各站的 :class:`SitePolarSample`。

    ``anchors``：``{站: (經度, 緯度)}``，缺少某站錨點即失敗。``site_ids`` 為 ``None`` 時輸出表中出現的全部站；
    指定了不存在的站同樣失敗。只讀不寫。檢查：必要欄位齊全、狀態碼都是已知值、
    ``crossed_interest_region_boundary`` 與「狀態為穿越」逐顆一致。
    """

    path = Path(postprocess_dir) / "departures.parquet"
    if not path.is_file():
        raise PolarSourceError(f"後處理目錄缺少必要輸入：departures.parquet（{postprocess_dir}）")
    names = set(pq.read_schema(path).names)
    missing = [c for c in _REQUIRED_COLUMNS if c not in names]
    if missing:
        raise PolarSourceError(f"departures.parquet 缺少必要欄位：{missing}")
    table = pq.read_table(path, columns=list(_REQUIRED_COLUMNS))
    site_col = table.column("study_site_id").to_numpy(zero_copy_only=False).astype(str)
    present = sorted(set(site_col.tolist()))
    wanted = present if site_ids is None else list(site_ids)
    unknown = [s for s in wanted if s not in present]
    if unknown:
        raise PolarSourceError(f"指定的站不在逐質點表中：{unknown}（現有：{present}）")
    result: dict[str, SitePolarSample] = {}
    for site in wanted:
        if site not in anchors:
            raise PolarSourceError(f"缺少站 {site} 的錨點（釋放設定檔 anchor_lonlat）")
        sel = np.flatnonzero(site_col == site)
        sub = table.take(pa.array(sel))

        def col(name: str, _sub: pa.Table = sub) -> np.ndarray:
            return _sub.column(name).to_numpy(zero_copy_only=False)

        status = col("final_status_code").astype(np.int64)
        bad = sorted(set(np.unique(status).tolist()) - set(_STATUS_CODES))
        if bad:
            raise PolarSourceError(f"站 {site}：出現未知的終止狀態碼 {bad}，fail-closed")
        crossed_flag = col("crossed_interest_region_boundary").astype(bool)
        if not np.array_equal(crossed_flag, status == NATIVE_STATUS_LEFT_FLOW_DOMAIN):
            raise PolarSourceError(f"站 {site}：crossed_interest_region_boundary 與終止狀態碼不一致，fail-closed")
        lon = col("departure_lon_deg").astype(np.float64)
        lat = col("departure_lat_deg").astype(np.float64)
        kind, az, bins = classify_sources(status, lon, lat, anchors[site])
        speed = np.abs(col("particle_settling_velocity_mps").astype(np.float64))
        if not np.all(np.isfinite(speed)):
            raise PolarSourceError(f"站 {site}：逐粒子沉降速度含缺值或非有限值，fail-closed")
        _, unit = np.unique(col("arrival_time_id").astype(str), return_inverse=True)
        counted = kind != KIND_OTHER
        result[site] = SitePolarSample(
            site_id=site,
            anchor_lonlat=(float(anchors[site][0]), float(anchors[site][1])),
            n_rows=int(sel.size),
            material_id=col("material_id").astype(str),
            candidate_index=col("settling_candidate_index").astype(np.int64),
            speed_abs_mps=speed,
            class_speed_abs_mps=np.abs(col("settling_velocity_mps").astype(np.float64)),
            season=col("deposition_season").astype(str),
            tide=col("deposition_tide_class").astype(str),
            unit=unit.astype(np.int64),
            n_units=int(unit.max()) + 1 if unit.size else 0,
            status=status,
            kind=kind,
            azimuth_deg=az,
            bin37=bins,
            source_lon=np.where(counted, lon, np.nan),
            source_lat=np.where(counted, lat, np.nan),
            particle_id=col("particle_id").astype(str),
        )
    return result


# ── 分組 ─────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class GroupSet:
    """單站、單一分組方式的全部組別。

    ``group_ids``、``labels_zh``：組別代號與中文顯示名稱（類別走顯示名稱對照表；類別×速度的標籤是速度數值）。
    ``material_id``：各組所屬類別（只有類別×速度與類別分組有意義，其餘為空字串）。
    ``speed_abs_mps``：各組代表速度絕對值（類別×速度為該速度；類別為實際速度中位數）；``speed_min_mps``、
    ``speed_max_mps``：各組實際速度絕對值的最小、最大。``n_total``：各組全部質點數（分母）。
    ``counts``：形狀 ``(組數, 到達時刻單元數, 37)``。``group_code``：每顆質點的組別編號（不在任何組者為 -1）。
    """

    grouping: str
    site_id: str
    group_ids: tuple[str, ...]
    labels_zh: tuple[str, ...]
    material_id: tuple[str, ...]
    speed_abs_mps: np.ndarray
    speed_min_mps: np.ndarray
    speed_max_mps: np.ndarray
    n_total: np.ndarray
    counts: np.ndarray
    group_code: np.ndarray

    def pct_sectors(self) -> np.ndarray:
        """各組 36 扇區比例（%），形狀 ``(組數, 36)``；分母為該組全部質點（含未計入者）。"""

        total = self.counts.sum(axis=1)
        return 100.0 * total[:, :N_SECTORS] / self.n_total[:, None]

    def proportions37(self) -> np.ndarray:
        """各組 37 格比例（總和 1），形狀 ``(組數, 37)``。"""

        return self.counts.sum(axis=1) / self.n_total[:, None]


def _speed_label(value: float) -> str:
    return f"{value:#.3g}".rstrip(".")


def build_group_set(sample: SitePolarSample, grouping: str) -> GroupSet:
    """依分組方式建立 :class:`GroupSet`（含 ``(組, 到達時刻單元, 37 格)`` 計數張量）。"""

    if grouping not in POLAR_GROUPINGS:
        raise PolarSourceError(f"未知的分組代號：{grouping!r}（合法值：{POLAR_GROUPINGS}）")
    mat = sample.material_id
    if grouping == "class_velocity":
        key = np.char.add(np.char.add(mat, "|"), sample.candidate_index.astype(str))
        raw_ids = sorted(set(key.tolist()))
    elif grouping == "class":
        key = mat
        raw_ids = sorted(set(key.tolist()))
    elif grouping == "season":
        key = sample.season
        present = set(key.tolist())
        raw_ids = [g for g in _CANONICAL_ORDER["season"] if g in present] + sorted(present - set(_CANONICAL_ORDER["season"]))
    else:
        key = sample.tide
        present = set(key.tolist())
        raw_ids = [g for g in _CANONICAL_ORDER["tide"] if g in present] + sorted(present - set(_CANONICAL_ORDER["tide"]))

    if grouping in ("class_velocity", "class"):
        # 類別一律依類別代表沉降速度（絕對值）由慢到快排序；類別內的候選速度依實際速度由慢到快。
        cls_speed = {m: float(np.median(sample.class_speed_abs_mps[mat == m])) for m in sorted(set(mat.tolist()))}
        if grouping == "class":
            raw_ids = sorted(raw_ids, key=lambda g: (cls_speed[g], g))
        else:
            first_speed = {g: float(np.median(sample.speed_abs_mps[key == g])) for g in raw_ids}
            raw_ids = sorted(raw_ids, key=lambda g: (cls_speed[g.split("|")[0]], first_speed[g], g))
    index_of = {g: i for i, g in enumerate(raw_ids)}
    code = np.fromiter((index_of.get(k, -1) for k in key.tolist()), dtype=np.int64, count=key.size)
    g_n = len(raw_ids)
    u_n = max(sample.n_units, 1)
    flat = (code * u_n + sample.unit) * N_BINS + sample.bin37
    if (code < 0).any():
        raise PolarSourceError("出現不在分組清單內的標籤，fail-closed")
    counts = np.bincount(flat, minlength=g_n * u_n * N_BINS).reshape(g_n, u_n, N_BINS)
    n_total = counts.sum(axis=(1, 2)).astype(np.int64)

    labels: list[str] = []
    materials: list[str] = []
    spd = np.zeros(g_n)
    smin = np.zeros(g_n)
    smax = np.zeros(g_n)
    for i, g in enumerate(raw_ids):
        sel = code == i
        s = sample.speed_abs_mps[sel]
        spd[i], smin[i], smax[i] = float(np.median(s)), float(s.min()), float(s.max())
        if grouping == "class_velocity":
            m = g.split("|")[0]
            materials.append(m)
            labels.append(_speed_label(spd[i]))
        elif grouping == "class":
            materials.append(g)
            labels.append(material_display_name_zh(g))
        else:
            materials.append("")
            labels.append(abg.group_label_zh(grouping, g))
    return GroupSet(
        grouping=grouping,
        site_id=sample.site_id,
        group_ids=tuple(raw_ids),
        labels_zh=tuple(labels),
        material_id=tuple(materials),
        speed_abs_mps=spd,
        speed_min_mps=smin,
        speed_max_mps=smax,
        n_total=n_total,
        counts=counts,
        group_code=code,
    )


# ── 可區辨性：比較、組對與雜訊基準 ──────────────────────────────────────


@dataclass(slots=True)
class Comparison:
    """一個比較範圍（例如「某類別的 10 個速度之間」「十類之間」）的檢定結果。

    ``comparison_id``：例如 ``velocity_within_class:oca_steel_rigid_cylinder``、``class``、``season``、``tide``。
    ``kind``：``velocity_within_class`` / ``class`` / ``season`` / ``tide``。``pairs``：每一組對的觀測值。
    ``noise``：同組依到達時刻單元對半切之 TVD 的合併分布摘要（含 p50、p95、p99、max）。
    ``verdict``：``over_noise``（至少半數組對超過 p99 門檻）/ ``partial``（有但不到半數）/ ``within_noise``。
    """

    comparison_id: str
    kind: str
    scope_material: str
    n_groups: int
    group_ids: tuple[str, ...]
    n_pairs: int
    group_n_min: int
    group_n_max: int
    pairs: list[dict[str, Any]]
    noise_tvd: dict[str, float]
    noise_maxsec: dict[str, float]
    max_tvd: float
    mean_tvd: float
    max_pair: tuple[str, str]
    max_maxsec_pp: float
    pairs_over_p99: int
    pairs_over_p95: int
    maxsec_pairs_over_p99: int
    max_tvd_over_p99: bool
    verdict: str


def _pct_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "n": int(values.size),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def _compare_groups(
    gs: GroupSet,
    members: Sequence[int],
    comparison_id: str,
    kind: str,
    scope_material: str,
    case_id: str,
    params: PolarParams,
) -> Comparison | None:
    """對 ``members``（組別編號）做雜訊基準與兩兩 TVD。少於兩組、或任一組無法對半切時回傳 ``None``。"""

    if len(members) < 2:
        return None
    tvd_pool: list[np.ndarray] = []
    max_pool: list[np.ndarray] = []
    for gi in members:
        rng = np.random.default_rng(_stable_seed(params.seed, case_id, gs.site_id, gs.grouping, gs.group_ids[gi]))
        out = half_split_noise(gs.counts[gi], params.n_resamples, rng, min_units=params.min_units)
        if out is None:
            continue
        tvd_pool.append(out[0])
        max_pool.append(out[1])
    if not tvd_pool:
        return None
    noise_tvd = _pct_summary(np.concatenate(tvd_pool))
    noise_max = _pct_summary(np.concatenate(max_pool))
    p = gs.proportions37()[list(members)]
    diff = np.abs(p[:, None, :] - p[None, :, :])
    tvd_m = 0.5 * diff.sum(axis=-1)
    max_m = 100.0 * diff[:, :, :N_SECTORS].max(axis=-1)
    argmax_sector = diff[:, :, :N_SECTORS].argmax(axis=-1)
    pairs: list[dict[str, Any]] = []
    for a in range(len(members)):
        for b in range(a + 1, len(members)):
            pairs.append(
                {
                    "group_a": gs.group_ids[members[a]],
                    "group_b": gs.group_ids[members[b]],
                    "label_a": gs.labels_zh[members[a]],
                    "label_b": gs.labels_zh[members[b]],
                    "tvd": float(tvd_m[a, b]),
                    "max_sector_diff_pp": float(max_m[a, b]),
                    "max_sector_diff_sector": int(argmax_sector[a, b]),
                    "over_p95": bool(tvd_m[a, b] > noise_tvd["p95"]),
                    "over_p99": bool(tvd_m[a, b] > noise_tvd["p99"]),
                    "maxsec_over_p99": bool(max_m[a, b] > noise_max["p99"]),
                }
            )
    worst = max(pairs, key=lambda r: r["tvd"])
    n_over99 = sum(r["over_p99"] for r in pairs)
    if n_over99 * 2 >= len(pairs):
        verdict = "over_noise"
    elif n_over99 > 0:
        verdict = "partial"
    else:
        verdict = "within_noise"
    ns = gs.n_total[list(members)]
    return Comparison(
        comparison_id=comparison_id,
        kind=kind,
        scope_material=scope_material,
        n_groups=len(members),
        group_ids=tuple(gs.group_ids[m] for m in members),
        n_pairs=len(pairs),
        group_n_min=int(ns.min()),
        group_n_max=int(ns.max()),
        pairs=pairs,
        noise_tvd=noise_tvd,
        noise_maxsec=noise_max,
        max_tvd=float(worst["tvd"]),
        mean_tvd=float(np.mean([r["tvd"] for r in pairs])),
        max_pair=(worst["group_a"], worst["group_b"]),
        max_maxsec_pp=float(max(r["max_sector_diff_pp"] for r in pairs)),
        pairs_over_p99=int(n_over99),
        pairs_over_p95=int(sum(r["over_p95"] for r in pairs)),
        maxsec_pairs_over_p99=int(sum(r["maxsec_over_p99"] for r in pairs)),
        max_tvd_over_p99=bool(worst["tvd"] > noise_tvd["p99"]),
        verdict=verdict,
    )


@dataclass(slots=True)
class SiteAnalysis:
    """單站完整分析：四種分組的組別、終止類別統計與全部比較。"""

    site_id: str
    sample: SitePolarSample
    group_sets: dict[str, GroupSet]
    comparisons: list[Comparison]


def analyze_site(sample: SitePolarSample, case_id: str, params: PolarParams) -> SiteAnalysis:
    """建立四種分組並執行 :data:`POLAR_GROUPINGS` 對應的全部可區辨性比較。"""

    sets = {g: build_group_set(sample, g) for g in POLAR_GROUPINGS}
    comps: list[Comparison] = []
    cv = sets["class_velocity"]
    for material in dict.fromkeys(sets["class"].group_ids):
        members = [i for i, m in enumerate(cv.material_id) if m == material]
        c = _compare_groups(cv, members, f"velocity_within_class:{material}", "velocity_within_class", material, case_id, params)
        if c is not None:
            comps.append(c)
    for kind in ("class", "season", "tide"):
        gs = sets[kind]
        c = _compare_groups(gs, list(range(len(gs.group_ids))), kind, kind, "", case_id, params)
        if c is not None:
            comps.append(c)
    return SiteAnalysis(site_id=sample.site_id, sample=sample, group_sets=sets, comparisons=comps)


def termination_rows(analysis: SiteAnalysis, case_id: str) -> list[dict[str, Any]]:
    """每組（含「全站」一組）的終止類別計數與比例，供 CSV。

    欄位：全部質點數、穿越數、擱淺數、其他終止類別各自的數量，以及各項占該組全部質點的百分比，
    ``counted_pct`` = 計入扇區（穿越＋擱淺）的比例 = 36 扇區比例合計。
    """

    s = analysis.sample
    rows: list[dict[str, Any]] = []

    def one(grouping: str, gid: str, label: str, material: str, speed: float | None, sel: np.ndarray | slice) -> None:
        st = s.status[sel]
        n = int(st.size)
        rec: dict[str, Any] = {
            "case": case_id,
            "site": s.site_id,
            "grouping": grouping,
            "group_id": gid,
            "group_label_zh": label,
            "material_id": material,
            "settling_speed_abs_mps": "" if speed is None else f"{speed:.6g}",
            "n_total": n,
        }
        counts = {c: int(np.count_nonzero(st == c)) for c in _STATUS_CODES}
        rec["n_crossed"] = counts[NATIVE_STATUS_LEFT_FLOW_DOMAIN]
        rec["n_stranded"] = counts[NATIVE_STATUS_STRANDED]
        rec["n_counted"] = rec["n_crossed"] + rec["n_stranded"]
        rec["n_uncounted"] = n - rec["n_counted"]
        for c in _OTHER_STATUS_CODES:
            rec[f"n_status_{c}"] = counts[c]
        rec["crossed_pct"] = 100.0 * rec["n_crossed"] / n
        rec["stranded_pct"] = 100.0 * rec["n_stranded"] / n
        rec["counted_pct"] = 100.0 * rec["n_counted"] / n
        rec["uncounted_pct"] = 100.0 * rec["n_uncounted"] / n
        for c in _OTHER_STATUS_CODES:
            rec[f"status_{c}_pct"] = 100.0 * counts[c] / n
        rows.append(rec)

    one("site_all", "all", "全部虛擬質點", "", None, slice(None))
    for grouping, gs in analysis.group_sets.items():
        for i, gid in enumerate(gs.group_ids):
            one(grouping, gid, gs.labels_zh[i], gs.material_id[i], float(gs.speed_abs_mps[i]) if grouping == "class_velocity" else None, gs.group_code == i)
    return rows


# ── CSV 輸出（以排他方式建立，已存在即失敗）──────────────────────────────


def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(header), extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_sector_long_csv(analyses: Sequence[SiteAnalysis], case_id: str, path: Path) -> None:
    """各案/地點/分組/扇區的粒子數與比例長表（四種分組：類別×速度、類別、季節、大小潮）。"""

    rows: list[dict[str, Any]] = []
    for an in analyses:
        for grouping, gs in an.group_sets.items():
            tot = gs.counts.sum(axis=1)
            for i, gid in enumerate(gs.group_ids):
                for k in range(N_SECTORS):
                    rows.append(
                        {
                            "case": case_id,
                            "site": an.site_id,
                            "grouping": grouping,
                            "group_id": gid,
                            "group_label_zh": gs.labels_zh[i],
                            "material_id": gs.material_id[i],
                            "settling_speed_abs_mps": f"{gs.speed_abs_mps[i]:.6g}" if grouping == "class_velocity" else "",
                            "sector_index": k,
                            "sector_center_azimuth_deg": SECTOR_WIDTH_DEG * k,
                            "count": int(tot[i, k]),
                            "group_total": int(gs.n_total[i]),
                            "pct_of_group": f"{100.0 * tot[i, k] / gs.n_total[i]:.6f}",
                        }
                    )
    header = [
        "case", "site", "grouping", "group_id", "group_label_zh", "material_id", "settling_speed_abs_mps",
        "sector_index", "sector_center_azimuth_deg", "count", "group_total", "pct_of_group",
    ]  # fmt: skip
    _write_csv(path, header, rows)


def write_termination_csv(analyses: Sequence[SiteAnalysis], case_id: str, path: Path) -> None:
    """各案/地點/分組的擱淺比例與其他終止類別比例表。"""

    rows = [r for an in analyses for r in termination_rows(an, case_id)]
    header = [
        "case", "site", "grouping", "group_id", "group_label_zh", "material_id", "settling_speed_abs_mps",
        "n_total", "n_crossed", "n_stranded", "n_counted", "n_uncounted",
        *[f"n_status_{c}" for c in _OTHER_STATUS_CODES],
        "crossed_pct", "stranded_pct", "counted_pct", "uncounted_pct",
        *[f"status_{c}_pct" for c in _OTHER_STATUS_CODES],
    ]  # fmt: skip
    for r in rows:
        for key in ("crossed_pct", "stranded_pct", "counted_pct", "uncounted_pct", *[f"status_{c}_pct" for c in _OTHER_STATUS_CODES]):
            r[key] = f"{r[key]:.6f}"
    _write_csv(path, header, rows)


def write_pairs_csv(analyses: Sequence[SiteAnalysis], case_id: str, path: Path) -> None:
    """每個組對的觀測 TVD、最大扇區差，以及是否超過該比較的雜訊基準門檻。"""

    rows: list[dict[str, Any]] = []
    for an in analyses:
        for c in an.comparisons:
            for r in c.pairs:
                rows.append(
                    {
                        "case": case_id,
                        "site": an.site_id,
                        "comparison_id": c.comparison_id,
                        "kind": c.kind,
                        **{k: r[k] for k in ("group_a", "group_b", "label_a", "label_b")},
                        "tvd": f"{r['tvd']:.6f}",
                        "noise_tvd_p95": f"{c.noise_tvd['p95']:.6f}",
                        "noise_tvd_p99": f"{c.noise_tvd['p99']:.6f}",
                        "over_p95": r["over_p95"],
                        "over_p99": r["over_p99"],
                        "max_sector_diff_pp": f"{r['max_sector_diff_pp']:.4f}",
                        "max_sector_diff_sector": r["max_sector_diff_sector"],
                        "noise_maxsec_p99_pp": f"{c.noise_maxsec['p99']:.4f}",
                        "maxsec_over_p99": r["maxsec_over_p99"],
                    }
                )
    header = [
        "case", "site", "comparison_id", "kind", "group_a", "group_b", "label_a", "label_b", "tvd", "noise_tvd_p95",
        "noise_tvd_p99", "over_p95", "over_p99", "max_sector_diff_pp", "max_sector_diff_sector", "noise_maxsec_p99_pp",
        "maxsec_over_p99",
    ]  # fmt: skip
    _write_csv(path, header, rows)


def write_comparison_csv(analyses: Sequence[SiteAnalysis], case_id: str, path: Path) -> None:
    """每個比較範圍一列：雜訊基準（half-split）、觀測最大 TVD、是否通過 99% 門檻與判定。"""

    rows: list[dict[str, Any]] = []
    for an in analyses:
        for c in an.comparisons:
            rows.append(
                {
                    "case": case_id,
                    "site": an.site_id,
                    "comparison_id": c.comparison_id,
                    "kind": c.kind,
                    "material_id": c.scope_material,
                    "n_groups": c.n_groups,
                    "n_pairs": c.n_pairs,
                    "group_n_min": c.group_n_min,
                    "group_n_max": c.group_n_max,
                    "noise_tvd_p50": f"{c.noise_tvd['p50']:.6f}",
                    "noise_tvd_p95": f"{c.noise_tvd['p95']:.6f}",
                    "noise_tvd_p99": f"{c.noise_tvd['p99']:.6f}",
                    "noise_tvd_max": f"{c.noise_tvd['max']:.6f}",
                    "noise_resamples_pooled": c.noise_tvd["n"],
                    "max_tvd": f"{c.max_tvd:.6f}",
                    "mean_tvd": f"{c.mean_tvd:.6f}",
                    "max_tvd_pair": f"{c.max_pair[0]} vs {c.max_pair[1]}",
                    "max_tvd_over_p99": c.max_tvd_over_p99,
                    "pairs_over_p99": c.pairs_over_p99,
                    "pairs_over_p95": c.pairs_over_p95,
                    "frac_pairs_over_p99": f"{c.pairs_over_p99 / c.n_pairs:.4f}",
                    "max_sector_diff_pp": f"{c.max_maxsec_pp:.4f}",
                    "noise_maxsec_p99_pp": f"{c.noise_maxsec['p99']:.4f}",
                    "maxsec_pairs_over_p99": c.maxsec_pairs_over_p99,
                    "verdict": c.verdict,
                }
            )
    header = list(rows[0].keys()) if rows else ["case"]
    _write_csv(path, header, rows)


def write_stranded_csv(analyses: Sequence[SiteAnalysis], case_id: str, path: Path) -> None:
    """全部擱淺質點清單（地圖上畫小標記的依據）；沒有擱淺時只有標頭。"""

    rows: list[dict[str, Any]] = []
    for an in analyses:
        s = an.sample
        for i in np.flatnonzero(s.kind == KIND_STRANDED):
            rows.append(
                {
                    "case": case_id,
                    "site": s.site_id,
                    "particle_id": s.particle_id[i],
                    "material_id": s.material_id[i],
                    "lon_deg": f"{s.source_lon[i]:.6f}",
                    "lat_deg": f"{s.source_lat[i]:.6f}",
                    "azimuth_deg": f"{s.azimuth_deg[i]:.4f}",
                    "sector_index": int(s.bin37[i]),
                }
            )
    _write_csv(path, ["case", "site", "particle_id", "material_id", "lon_deg", "lat_deg", "azimuth_deg", "sector_index"], rows)
