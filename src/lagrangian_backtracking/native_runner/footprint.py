"""WP-38：以「停留時數密度」取代邊界點核密度，作為條件式來源足跡的主產品。

動機
----
``postprocess.kde_by_site`` 的輸入是粒子「穿越開放邊界時的離開座標」。這些點全部
落在一維邊界上，二維核密度（Scott 頻寬）會把它們抹成跨出計算域的圓團，高密度區間
（HDR）大片落在域外，物理上站不住。本模組改用每顆粒子的**完整軌跡**：逐小時把
「仍在漂流」的粒子位置累加到固定經緯度規則網格，得到「停留時數密度」，再由密度
導出 HDR；另外提供回溯特定天數時的位置快照、以及沿開放邊界弧長（一維）的離開分布。

定義
----
- **停留時數**：一顆粒子在某網格內的「逐小時樣本數」。只計 ``status == 0`` 的樣本
  （尚在漂流的有效位置）。粒子終止後 x／y 凍結在終止位置、status 變成非 0；若把
  這些樣本也算進去，凍結在邊界上的粒子會灌入大量假停留，因此必須排除
  （``tests/test_footprint.py`` 有專門測試保護）。
- **到達時刻樣本**：軌跡第 0 筆（到達時刻、受體位置，status 為 0）**計入**；也就是
  一顆漂流了 h 小時後終止的粒子貢獻 ``h + 1`` 筆停留時數（第 0 到第 h 小時，
  終止那一筆因 status 非 0 而不計，若粒子在第 h 筆之前一直有效則有效筆數為 h）。
  這個選擇寫入 manifest 的 ``footprint.arrival_sample_included``。
- **正規化**：``fraction_of_total`` ＝ 網格停留時數 ÷ 該站（或該分群）總停留時數，
  全場總和為 1（容差 1e-9，否則 fail-closed）；``density_fraction_per_km2`` ＝
  ``fraction_of_total`` ÷ 網格面積（km²），面積依緯度以 ``R²·Δλ·Δφ·cos φ`` 計算
  （``R`` 取 6371.0088 km，``φ`` 為網格中心緯度）。圖面色條單位「% 停留時數／km²」
  即 ``100 × density_fraction_per_km2``。
- **HDR**：依 ``density_fraction_per_km2`` 由高到低累加網格，累積停留時數比例首次
  達到 50／75／90% 時停止；被累加的網格集合就是「包含該比例停留時數的最小面積
  網格集合」，密度門檻是最後加入網格的密度，面積是該集合的網格面積和。
- **回溯快照**：回溯 ``h`` 小時時 ``status == 0`` 的粒子位置，直接累加到與停留時數密度
  **同一個**經緯度網格（整數計數、同樣的面積權重），不做核密度估計。分母是該時刻仍在漂流
  的粒子數（另記錄仍在漂流比例＝仍在漂流粒子數 ÷ 全站粒子數；軌跡長度不足 ``h + 1`` 筆
  的單元視為已全部終止）。可選的輕度平滑是「海域遮罩下的正規化卷積」：
  ``G*(d·m) / G*m``，``d`` 是密度、``m`` 是海域遮罩（陸地與 flow domain 外一律為 0）、
  ``G`` 是以公里指定寬度的高斯核（預設 2 km，約一個網格）；平滑後再依總質量重新正規化為 1，
  陸地格的密度恆為 0。仍在漂流粒子數低於門檻（預設 200）時不出密度。
  海域遮罩＝flow domain 內、且海域面積占格面積不小於 0.5 的格（另加：曾有粒子出現的格
  必為海域），陸地來源是海岸線 GeoJSON（與出圖所用同一檔）。
- **沿開放邊界弧長的離開分布**：終止於 flow domain 開放邊界的粒子，其離開座標投影
  到開放邊界折線上的弧長（km），以 10 km 為預設組距做直方圖，附累積分布；並依
  flow domain 外框標出北、東、南、西各邊與角落的弧長區間。

串流與記憶體
------------
``FootprintAccumulator.add_batch`` 每批只做整數計數（``np.bincount``），不保留軌跡；
狀態大小為「網格格數 × (沉降速度數 × 系集成員數 + 到達單元數)」的整數陣列，與粒子
總數與軌跡長度無關。投影座標（公尺）到經緯度網格的對照，用每站一張預先計算的查表
（``lut_cell_m``，預設 200 m，只在建構時呼叫一次投影反轉換），逐點只做整數索引。
計數為整數（快照也是整數直方圖），因此分塊累加與一次累加的結果位元相同；弧長直方圖
在 finalize 時先把點依座標排序，輸出同樣與批次切分無關。

收斂與敏感度
------------
- 成員分半：``member_id`` 排序後前半與後半各自的正規化停留場，計算總變異距離
  ``TVD = 0.5·Σ|p1 − p2|`` 與 HDR50 面積的相對差。
- 雜訊底：（a）把成員隨機分成大小相同的兩半，重複多次；（b）把到達單元隨機分成
  大小相同的兩半，重複多次。兩者的 TVD 平均與第 95 百分位數，作為分半 TVD 的參照值。
- 網格敏感度：0.01°／0.02°／0.04° 三種解析度的 HDR50 與 HDR90 面積。

公尺網格（WP-70）
-----------------
``FootprintConfig.grid_cell_km`` 不為 ``None`` 時，改用「流場區域局部等距方位座標」下的 km 級規則公尺
網格取代經緯度網格。粒子軌跡座標 ``(x, y)`` 本來就是各流場區域以區域中心為投影中心的等距方位
（AEQD）平面座標（公尺），網格直接切在這個平面上：格線落在 ``x``、``y`` 為格寬整數倍之處（原點
＝投影中心＝流場區域中心，是一個格角），每格在該平面上是邊長 ``grid_cell_km`` 的正方形，面積正好
``grid_cell_km`` 平方公里，不依緯度而變，也不需要查表（粒子座標直接取整數索引）。網格涵蓋流場區域外框
加 ``padding_km``，邊緣對齊最大格寬的整數倍，所以敏感度網格（預設 0.5 與 2 km）與主網格巢狀對齊。
密度、正規化、HDR、回溯快照、海域遮罩的定義與經緯度網格版本相同，只是面積恆為常數；格心與格角的經緯度
由區域投影反轉換得到，寫進表格供繪圖與水深查詢使用。未指定 ``grid_cell_km`` 時，所有行為與輸出與本
修改之前位元相同。
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pyarrow as pa
import shapely
from scipy import ndimage
from shapely.geometry.base import BaseGeometry

from .backward_driver import NATIVE_STATUS_LEFT_FLOW_DOMAIN
from .particle_input import ParticleIdentityTable

__all__ = [
    "FootprintError",
    "FootprintConfig",
    "FootprintSiteSpec",
    "FootprintGrid",
    "FootprintMeterGrid",
    "FootprintProducts",
    "FootprintAccumulator",
    "EARTH_RADIUS_KM",
    "cell_area_km2_by_row",
    "compute_hdr",
    "smooth_masked_density",
    "load_land_lonlat",
    "total_variation_distance",
    "haversine_km_and_bearing_deg",
]

EARTH_RADIUS_KM = 6371.0088
"""計算網格面積與大圓距離用的地球半徑（km，IUGG 平均半徑）。"""

NORMALIZED_SUM_TOLERANCE = 1.0e-9
"""正規化停留場總和與 1 的容許誤差；超過即 fail-closed。"""

METER_GRID_KIND = "regular_meter_local_aeqd"
"""WP-70：公尺網格在 ``footprint_grid_by_site`` 與 manifest 的種類識別。"""

GROUP_ALL = "all"
GROUP_SETTLING_VELOCITY = "settling_velocity"
GROUP_MEMBER_HALF = "member_half"
GROUP_SNAPSHOT = "snapshot"

KDE_NOTE_ZH = (
    "kde_by_site 的輸入點位於一維開放邊界，二維 KDE 不適用，不再作報告主圖；"
    "報告主圖改用停留時數密度（residence_density_by_site）。"
)

LonLatProjector = Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


class FootprintError(RuntimeError):
    """足跡產品的資料契約或一致性錯誤，一律 fail-closed。"""


# ── 設定 ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class FootprintConfig:
    """足跡產品設定；全部欄位在建構時驗證。"""

    grid_resolution_deg: float = 0.02
    sensitivity_resolutions_deg: tuple[float, ...] = (0.01, 0.04)
    padding_deg: float = 0.05
    lut_cell_m: float = 200.0
    hdr_levels: tuple[float, ...] = (0.50, 0.75, 0.90)
    lookback_snapshot_hours: tuple[int, ...] = (72, 168, 336, 720)
    snapshot_smoothing_km: float = 2.0
    snapshot_minimum_particle_count: int = 200
    land_mask_min_sea_fraction: float = 0.5
    arc_bin_km: float = 10.0
    noise_floor_repeats: int = 64
    noise_floor_seed: int = 20260929
    depth_threshold_m: float = 200.0
    count_arrival_sample: bool = True
    grid_cell_km: float | None = None
    """WP-70：不為 ``None`` 時改用邊長此值（km）的規則公尺網格，``grid_resolution_deg`` 與
    ``sensitivity_resolutions_deg`` 在公尺網格模式下不使用。"""
    sensitivity_cells_km: tuple[float, ...] = (0.5, 2.0)
    """WP-70：公尺網格模式的敏感度格寬（km，不含主網格）。"""
    padding_km: float = 5.0
    """WP-70：公尺網格涵蓋範圍＝流場區域外框外推的距離（km）。"""

    def __post_init__(self) -> None:
        if self.grid_cell_km is not None:
            if not (0.0 < float(self.grid_cell_km) <= 100.0):
                raise FootprintError("grid_cell_km 必須介於 0（不含）與 100 km 之間")
            for cell in self.sensitivity_cells_km:
                if not (0.0 < float(cell) <= 100.0):
                    raise FootprintError("sensitivity_cells_km 每個值都必須介於 0（不含）與 100 km 之間")
            if self.padding_km < 0.0:
                raise FootprintError("padding_km 不可為負")
        if not (0.0 < self.grid_resolution_deg <= 1.0):
            raise FootprintError("grid_resolution_deg 必須介於 0 與 1 度之間")
        for res in self.sensitivity_resolutions_deg:
            if not (0.0 < res <= 1.0):
                raise FootprintError("sensitivity_resolutions_deg 每個值都必須介於 0 與 1 度之間")
        if self.padding_deg < 0.0:
            raise FootprintError("padding_deg 不可為負")
        if self.lut_cell_m <= 0.0:
            raise FootprintError("lut_cell_m 必須為正")
        levels = tuple(float(v) for v in self.hdr_levels)
        if not levels or any(not (0.0 < v < 1.0) for v in levels) or list(levels) != sorted(levels):
            raise FootprintError("hdr_levels 必須是介於 0 與 1 之間、遞增的非空序列")
        object.__setattr__(self, "hdr_levels", levels)
        hours = tuple(int(h) for h in self.lookback_snapshot_hours)
        if any(h < 0 for h in hours):
            raise FootprintError("lookback_snapshot_hours 不可為負")
        object.__setattr__(self, "lookback_snapshot_hours", hours)
        if self.snapshot_smoothing_km < 0.0:
            raise FootprintError("snapshot_smoothing_km 不可為負（0 表示不平滑）")
        if self.snapshot_minimum_particle_count < 1:
            raise FootprintError("snapshot_minimum_particle_count 必須為正整數")
        if not (0.0 < self.land_mask_min_sea_fraction <= 1.0):
            raise FootprintError("land_mask_min_sea_fraction 必須介於 0（不含）與 1 之間")
        if self.arc_bin_km <= 0.0:
            raise FootprintError("arc_bin_km 必須為正")
        if self.noise_floor_repeats < 1:
            raise FootprintError("noise_floor_repeats 必須為正整數")

    def to_manifest_dict(self) -> dict[str, Any]:
        out = self._base_manifest_dict()
        if self.grid_cell_km is not None:
            # 公尺網格欄位只在啟用時才出現，預設模式的 manifest 與修改前完全相同。
            out["grid_cell_km"] = float(self.grid_cell_km)
            out["sensitivity_cells_km"] = [float(v) for v in self.sensitivity_cells_km]
            out["padding_km"] = float(self.padding_km)
        return out

    def _base_manifest_dict(self) -> dict[str, Any]:
        return {
            "grid_resolution_deg": self.grid_resolution_deg,
            "sensitivity_resolutions_deg": list(self.sensitivity_resolutions_deg),
            "padding_deg": self.padding_deg,
            "lut_cell_m": self.lut_cell_m,
            "hdr_levels": list(self.hdr_levels),
            "lookback_snapshot_hours": list(self.lookback_snapshot_hours),
            "snapshot_smoothing_km": self.snapshot_smoothing_km,
            "snapshot_minimum_particle_count": self.snapshot_minimum_particle_count,
            "land_mask_min_sea_fraction": self.land_mask_min_sea_fraction,
            "arc_bin_km": self.arc_bin_km,
            "noise_floor_repeats": self.noise_floor_repeats,
            "noise_floor_seed": self.noise_floor_seed,
            "depth_threshold_m": self.depth_threshold_m,
            "count_arrival_sample": self.count_arrival_sample,
        }


@dataclass(frozen=True, slots=True)
class FootprintSiteSpec:
    """單站足跡所需的公尺制幾何：flow domain 多邊形與 flow domain 開放邊界折線。"""

    flow_domain: BaseGeometry
    flow_open_boundary: BaseGeometry | None = None


# ── 網格與面積 ───────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class FootprintGrid:
    """固定經緯度規則網格：``lon0``／``lat0`` 是西南角，格寬 ``resolution_deg`` 度。"""

    lon0: float
    lat0: float
    resolution_deg: float
    nx: int
    ny: int

    @property
    def cell_count(self) -> int:
        return self.nx * self.ny

    def lon_centers(self) -> np.ndarray:
        return self.lon0 + (np.arange(self.nx) + 0.5) * self.resolution_deg

    def lat_centers(self) -> np.ndarray:
        return self.lat0 + (np.arange(self.ny) + 0.5) * self.resolution_deg

    def flat_index(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """經緯度到網格編號（``iy * nx + ix``，int32）；網格外為 ``-1``。"""

        fx = np.floor((np.asarray(lon, dtype=np.float64) - self.lon0) / self.resolution_deg)
        fy = np.floor((np.asarray(lat, dtype=np.float64) - self.lat0) / self.resolution_deg)
        ok = (fx >= 0) & (fx < self.nx) & (fy >= 0) & (fy < self.ny)
        flat = np.where(ok, fy * self.nx + fx, -1.0)
        return flat.astype(np.int32)

    def row_area_km2(self) -> np.ndarray:
        """每個緯度列的網格面積（km²），長度為 ``ny``。"""

        return cell_area_km2_by_row(self.lat_centers(), self.resolution_deg, self.resolution_deg)

    def cell_area_km2(self) -> np.ndarray:
        """攤平成 ``iy * nx + ix`` 順序的每格面積（km²）。"""

        return np.repeat(self.row_area_km2(), self.nx)

    def cell_centers_lonlat(self) -> tuple[np.ndarray, np.ndarray]:
        lon = np.tile(self.lon_centers(), self.ny)
        lat = np.repeat(self.lat_centers(), self.nx)
        return lon, lat


def cell_area_km2_by_row(lat_center_deg: np.ndarray, dlon_deg: float, dlat_deg: float) -> np.ndarray:
    """經緯度網格面積：``R²·Δλ·Δφ·cos φ``（Δ 為弧度、φ 為列中心緯度）。"""

    lat = np.asarray(lat_center_deg, dtype=np.float64)
    return EARTH_RADIUS_KM**2 * math.radians(dlon_deg) * math.radians(dlat_deg) * np.cos(np.radians(lat))


def make_grid(
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    *,
    resolution_deg: float,
    align_deg: float,
) -> FootprintGrid:
    """建立涵蓋指定範圍的網格，西南角對齊 ``align_deg`` 的整數倍，讓不同解析度的網格巢狀對齊。"""

    lon0 = round(math.floor(lon_min / align_deg + 1e-9) * align_deg, 10)
    lat0 = round(math.floor(lat_min / align_deg + 1e-9) * align_deg, 10)
    nx = max(1, int(math.ceil((lon_max - lon0) / resolution_deg - 1e-9)))
    ny = max(1, int(math.ceil((lat_max - lat0) / resolution_deg - 1e-9)))
    return FootprintGrid(lon0=lon0, lat0=lat0, resolution_deg=resolution_deg, nx=nx, ny=ny)


class FootprintMeterGrid:
    """WP-70：流場區域局部等距方位平面上的規則公尺網格。

    ``x0_m``／``y0_m`` 是西南角（公尺，區域投影平面座標），格寬 ``cell_m`` 公尺，格 ``(ix, iy)`` 涵蓋
    ``[x0 + ix·cell, x0 + (ix+1)·cell) × [y0 + iy·cell, y0 + (iy+1)·cell)``。每格面積是常數
    ``(cell_m/1000)²`` km²。經緯度只用於輸出與繪圖：``project`` 是把平面公尺轉成 ``(lon, lat)`` 的函式
    （該區域投影的反轉換），格心與格角的經緯度第一次用到時才計算並快取。
    """

    kind = "meter"

    def __init__(
        self,
        *,
        x0_m: float,
        y0_m: float,
        cell_m: float,
        nx: int,
        ny: int,
        project: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    ) -> None:
        self.x0_m = float(x0_m)
        self.y0_m = float(y0_m)
        self.cell_m = float(cell_m)
        self.nx = int(nx)
        self.ny = int(ny)
        self._project = project
        self._centers_lonlat: tuple[np.ndarray, np.ndarray] | None = None
        self._corners_lonlat: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def cell_count(self) -> int:
        return self.nx * self.ny

    @property
    def cell_km(self) -> float:
        return self.cell_m / 1000.0

    def flat_index_xy(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """平面公尺座標到網格編號（``iy * nx + ix``，int32）；網格外為 ``-1``。"""

        fx = np.floor((np.asarray(x, dtype=np.float64) - self.x0_m) / self.cell_m)
        fy = np.floor((np.asarray(y, dtype=np.float64) - self.y0_m) / self.cell_m)
        ok = (fx >= 0) & (fx < self.nx) & (fy >= 0) & (fy < self.ny)
        flat = np.where(ok, fy * self.nx + fx, -1.0)
        return flat.astype(np.int32)

    def cell_area_km2(self) -> np.ndarray:
        """攤平成 ``iy * nx + ix`` 順序的每格面積（km²，全部相同）。"""

        return np.full(self.cell_count, self.cell_km**2, dtype=np.float64)

    def centers_xy(self) -> tuple[np.ndarray, np.ndarray]:
        """攤平成 ``iy * nx + ix`` 順序的格心平面座標（m）。"""

        xs = self.x0_m + (np.arange(self.nx) + 0.5) * self.cell_m
        ys = self.y0_m + (np.arange(self.ny) + 0.5) * self.cell_m
        return np.tile(xs, self.ny), np.repeat(ys, self.nx)

    def corners_xy(self) -> tuple[np.ndarray, np.ndarray]:
        """格角平面座標（m），形狀 ``(ny + 1, nx + 1)``。"""

        xs = self.x0_m + np.arange(self.nx + 1) * self.cell_m
        ys = self.y0_m + np.arange(self.ny + 1) * self.cell_m
        xx, yy = np.meshgrid(xs, ys)
        return xx, yy

    def cell_centers_lonlat(self) -> tuple[np.ndarray, np.ndarray]:
        if self._centers_lonlat is None:
            xc, yc = self.centers_xy()
            lon, lat = self._project(xc, yc)
            self._centers_lonlat = (np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64))
        return self._centers_lonlat

    def corners_lonlat(self) -> tuple[np.ndarray, np.ndarray]:
        """格角經緯度，形狀 ``(ny + 1, nx + 1)``。"""

        if self._corners_lonlat is None:
            xx, yy = self.corners_xy()
            lon, lat = self._project(xx.ravel(), yy.ravel())
            self._corners_lonlat = (
                np.asarray(lon, dtype=np.float64).reshape(xx.shape),
                np.asarray(lat, dtype=np.float64).reshape(xx.shape),
            )
        return self._corners_lonlat

    def cell_boxes_lonlat(self) -> Any:
        """每格四個格角在經緯度上連成的多邊形（shapely 陣列，``iy * nx + ix`` 順序）。"""

        lon, lat = self.corners_lonlat()
        ring = np.stack(
            [
                np.stack([lon[:-1, :-1].ravel(), lat[:-1, :-1].ravel()], axis=-1),
                np.stack([lon[:-1, 1:].ravel(), lat[:-1, 1:].ravel()], axis=-1),
                np.stack([lon[1:, 1:].ravel(), lat[1:, 1:].ravel()], axis=-1),
                np.stack([lon[1:, :-1].ravel(), lat[1:, :-1].ravel()], axis=-1),
                np.stack([lon[:-1, :-1].ravel(), lat[:-1, :-1].ravel()], axis=-1),
            ],
            axis=1,
        )
        return shapely.polygons(ring)


def make_meter_grid(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    *,
    cell_m: float,
    align_m: float,
    project: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
) -> FootprintMeterGrid:
    """建立涵蓋指定平面範圍的公尺網格，西南角對齊 ``align_m`` 的整數倍（巢狀對齊）。"""

    x0 = round(math.floor(x_min / align_m + 1e-9) * align_m, 6)
    y0 = round(math.floor(y_min / align_m + 1e-9) * align_m, 6)
    nx = max(1, int(math.ceil((x_max - x0) / cell_m - 1e-9)))
    ny = max(1, int(math.ceil((y_max - y0) / cell_m - 1e-9)))
    return FootprintMeterGrid(x0_m=x0, y0_m=y0, cell_m=cell_m, nx=nx, ny=ny, project=project)


# ── 純函式：HDR、TVD、大圓距離 ───────────────────────────────────────────


def compute_hdr(hours: np.ndarray, area_km2: np.ndarray, levels: Sequence[float]) -> list[dict[str, Any]]:
    """由停留時數（或其他非負權重）與網格面積求 HDR。

    依密度（權重 ÷ 面積）由高到低累加網格，累積權重首次不小於 ``level × 總權重`` 時停止。
    權重為整數時以整數比較（避免浮點誤差）；為浮點數（例如平滑後的快照）時目標值乘上
    ``1 - 1e-12`` 容差。回傳每個等級一筆
    ``{"level", "density_threshold_fraction_per_km2", "area_km2", "cell_count",
    "achieved_fraction", "cell_indices"}``。總權重為 0 時回傳空清單。
    """

    weights = np.asarray(hours)
    is_int = weights.dtype.kind in "iu"
    weights = weights.astype(np.int64 if is_int else np.float64)
    area = np.asarray(area_km2, dtype=np.float64)
    total = weights.sum()
    if total <= 0:
        return []
    positive = np.flatnonzero(weights > 0)
    density = weights[positive] / area[positive]
    order = positive[np.argsort(-density, kind="stable")]
    cumulative = np.cumsum(weights[order])
    result: list[dict[str, Any]] = []
    for level in levels:
        target = float(level) * total if is_int else float(level) * float(total) * (1.0 - 1e-12)
        k = int(np.searchsorted(cumulative, target, side="left")) + 1
        k = min(k, order.size)
        chosen = order[:k]
        last = chosen[-1]
        result.append(
            {
                "level": float(level),
                "density_threshold_fraction_per_km2": float(weights[last] / total / area[last]),
                "area_km2": float(area[chosen].sum()),
                "cell_count": int(k),
                "achieved_fraction": float(cumulative[k - 1] / total),
                "cell_indices": chosen,
            }
        )
    return result


def smooth_masked_density(
    density: np.ndarray, mask: np.ndarray, sigma_cells: tuple[float, float]
) -> np.ndarray:
    """海域遮罩下的正規化卷積 ``G*(d·m) / G*m``；遮罩為 0 的格輸出恆為 0。

    ``density`` 與 ``mask`` 形狀為 ``(ny, nx)``；``sigma_cells`` 是 ``(沿緯度方向, 沿經度方向)``
    的高斯核標準差（單位：格）。兩者皆 0 時不平滑，直接回傳 ``d·m``。遮罩外的密度不會被抹進海域，
    也不會有質量流失到陸地：``G*m`` 只在海域格上作分母。
    """

    d = np.asarray(density, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    if d.shape != m.shape:
        raise FootprintError("density 與 mask 形狀不一致")
    dm = np.where(m, d, 0.0)
    if sigma_cells[0] <= 0.0 and sigma_cells[1] <= 0.0:
        return dm
    num = ndimage.gaussian_filter(dm, sigma=sigma_cells, mode="constant", cval=0.0, truncate=3.0)
    den = ndimage.gaussian_filter(
        m.astype(np.float64), sigma=sigma_cells, mode="constant", cval=0.0, truncate=3.0
    )
    out = np.zeros_like(dm)
    ok = m & (den > 0.0)
    out[ok] = num[ok] / den[ok]
    return out


def load_land_lonlat(path: Any) -> tuple[BaseGeometry, dict[str, Any]]:
    """讀取海岸線 GeoJSON（CRS84 的 Polygon／MultiPolygon），回傳合併後的陸地幾何與來源紀錄。"""

    import hashlib
    import json
    from pathlib import Path

    from shapely.geometry import shape

    file = Path(path)
    raw = file.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict) or payload.get("type") != "FeatureCollection":
        raise FootprintError("海岸線 GeoJSON 必須是 FeatureCollection")
    geoms = []
    for index, feature in enumerate(payload.get("features") or []):
        try:
            geom = shape(feature["geometry"])
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise FootprintError(f"海岸線 GeoJSON feature[{index}] geometry 無法解析") from exc
        if geom.geom_type not in ("Polygon", "MultiPolygon") or geom.is_empty:
            raise FootprintError(f"海岸線 GeoJSON feature[{index}] 必須是非空 Polygon／MultiPolygon")
        geoms.append(shapely.make_valid(geom) if not geom.is_valid else geom)
    if not geoms:
        raise FootprintError("海岸線 GeoJSON features 不得為空")
    land = shapely.union_all(geoms)
    return land, {
        "file_name": file.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "feature_count": len(geoms),
    }


def total_variation_distance(hours_a: np.ndarray, hours_b: np.ndarray) -> float | None:
    """兩個停留時數場各自正規化後的總變異距離 ``0.5·Σ|p1 − p2|``；任一方總和為 0 時回傳 ``None``。"""

    a = np.asarray(hours_a, dtype=np.float64)
    b = np.asarray(hours_b, dtype=np.float64)
    sa, sb = a.sum(), b.sum()
    if sa <= 0 or sb <= 0:
        return None
    return float(0.5 * np.abs(a / sa - b / sb).sum())


def haversine_km_and_bearing_deg(lon1: float, lat1: float, lon2: float, lat2: float) -> tuple[float, float]:
    """回傳點 1 到點 2 的大圓距離（km）與方位角（度，正北 0、順時針）。"""

    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    distance = 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    bearing = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
    return distance, bearing


def _fmt_res(res: float) -> str:
    return f"{res:.2f}".replace(".", "p")


def _fmt_km(cell_km: float) -> str:
    """公尺網格敏感度欄位名稱用的格寬字串，例如 0.5 km -> ``0p50km``。"""

    return f"{cell_km:.2f}".replace(".", "p") + "km"


# ── 每站狀態 ─────────────────────────────────────────────────────────────


@dataclass(slots=True)
class _Lut:
    """公尺制座標到各解析度網格編號的查表。"""

    x0: float
    y0: float
    cell_m: float
    nx: int
    ny: int
    cells: dict[float, np.ndarray]

    def lookup(self, x: np.ndarray, y: np.ndarray, resolution_deg: float) -> tuple[np.ndarray, np.ndarray]:
        """回傳 ``(cell, in_lut)``：``cell`` 為網格編號（網格外 ``-1``），``in_lut`` 標示是否在查表範圍內。"""

        fx = np.floor((x - self.x0) / self.cell_m)
        fy = np.floor((y - self.y0) / self.cell_m)
        in_lut = (fx >= 0) & (fx < self.nx) & (fy >= 0) & (fy < self.ny)
        flat = np.where(in_lut, fy * self.nx + fx, 0.0).astype(np.int64)
        cell = np.where(in_lut, self.cells[resolution_deg][flat], -1)
        return cell.astype(np.int64), in_lut


@dataclass(slots=True)
class _SiteState:
    site_id: str
    spec: FootprintSiteSpec
    grid: FootprintGrid | FootprintMeterGrid
    sens_grids: dict[float, Any]
    lut: _Lut | None
    velocities: np.ndarray
    members: np.ndarray
    unit_count: int
    total_particles: int
    domain_lonlat_bounds: tuple[float, float, float, float]
    hours_vm: np.ndarray
    hours_unit: np.ndarray
    sens_hours: dict[float, np.ndarray]
    snapshot_hist: dict[int, np.ndarray] = field(default_factory=dict)
    snapshot_active: dict[int, int] = field(default_factory=dict)
    snapshot_outside: dict[int, int] = field(default_factory=dict)
    arc_s_m: list[np.ndarray] = field(default_factory=list)
    hours_outside_lut: int = 0
    hours_outside_grid: int = 0
    crossed_count: int = 0


@dataclass(slots=True)
class FootprintProducts:
    """``finalize`` 的輸出：各表（``pyarrow.Table``）、逐站摘要列與 manifest 區塊。"""

    tables: dict[str, pa.Table]
    summary_rows: list[dict[str, Any]]
    manifest: dict[str, Any]


class FootprintAccumulator:
    """串流累加停留時數、回溯快照與邊界弧長分布。

    Args:
        identity: 粒子身分表。
        config: :class:`FootprintConfig`。
        specs: 每站的公尺制幾何。
        lonlat_projector: ``(站 id 陣列, x, y) -> (lon, lat)``。
        bathymetry_by_site: 可選，每站水深場（``bathymetry.BathymetryField``）。
        land_lonlat: 可選，CRS84 的陸地幾何（海岸線）；提供時用於回溯快照的海域遮罩。
        land_source: 陸地幾何的來源紀錄（檔名、SHA-256），寫進 manifest。
    """

    def __init__(
        self,
        identity: ParticleIdentityTable,
        *,
        config: FootprintConfig,
        specs: Mapping[str, FootprintSiteSpec],
        lonlat_projector: LonLatProjector,
        bathymetry_by_site: Mapping[str, Any] | None = None,
        land_lonlat: BaseGeometry | None = None,
        land_source: Mapping[str, Any] | None = None,
    ) -> None:
        self._land = land_lonlat
        self._land_source = dict(land_source) if land_source else None
        self._identity = identity
        self._config = config
        self._projector = lonlat_projector
        self._bathymetry = dict(bathymetry_by_site) if bathymetry_by_site else {}
        n = len(identity)
        site_ids = np.asarray(identity.study_site_id)
        self._site_names = sorted(str(s) for s in np.unique(site_ids))
        missing = [s for s in self._site_names if s not in specs]
        if missing:
            raise FootprintError(f"足跡產品缺少下列站點的 flow domain 幾何：{', '.join(missing)}")

        self._site_of_row = np.full(n, -1, dtype=np.int16)
        self._slot_of_row = np.zeros(n, dtype=np.int64)
        self._unit_of_row = np.zeros(n, dtype=np.int64)
        self._states: dict[str, _SiteState] = {}
        align = max([config.grid_resolution_deg, *config.sensitivity_resolutions_deg])
        resolutions = sorted({config.grid_resolution_deg, *config.sensitivity_resolutions_deg})

        for site_index, site in enumerate(self._site_names):
            spec = specs[site]
            mask = site_ids == site
            rows_site = np.flatnonzero(mask)
            self._site_of_row[rows_site] = site_index
            velocities = np.unique(identity.settling_velocity_mps[rows_site])
            members = np.unique(identity.member_id[rows_site])
            v_idx = np.searchsorted(velocities, identity.settling_velocity_mps[rows_site])
            m_idx = np.searchsorted(members, identity.member_id[rows_site])
            self._slot_of_row[rows_site] = v_idx * members.size + m_idx
            unit_keys = np.char.add(
                np.char.add(identity.arrival_time_id[rows_site].astype(str), "|"),
                np.asarray([repr(float(v)) for v in identity.settling_velocity_mps[rows_site]]),
            )
            _, unit_inverse = np.unique(unit_keys, return_inverse=True)
            self._unit_of_row[rows_site] = unit_inverse
            unit_count = int(unit_inverse.max()) + 1

            bounds = self._domain_lonlat_bounds(site, spec)
            if config.grid_cell_km is not None:
                # WP-70：公尺網格。粒子的平面座標直接取整數索引，不需要查表。
                cells_km = sorted(
                    {float(config.grid_cell_km), *(float(c) for c in config.sensitivity_cells_km)}
                )
                x_min, y_min, x_max, y_max = spec.flow_domain.bounds
                pad_m = config.padding_km * 1000.0
                align_m = max(cells_km) * 1000.0
                project = self._site_project(site)
                mgrids = {
                    c: make_meter_grid(
                        x_min - pad_m,
                        x_max + pad_m,
                        y_min - pad_m,
                        y_max + pad_m,
                        cell_m=c * 1000.0,
                        align_m=align_m,
                        project=project,
                    )
                    for c in cells_km
                }
                grid = mgrids[float(config.grid_cell_km)]
                lut = None
                sens = {c: mgrids[c] for c in cells_km if c != float(config.grid_cell_km)}
            else:
                pad = config.padding_deg
                lon_min, lon_max, lat_min, lat_max = bounds
                grids = {
                    res: make_grid(
                        lon_min - pad,
                        lon_max + pad,
                        lat_min - pad,
                        lat_max + pad,
                        resolution_deg=res,
                        align_deg=align,
                    )
                    for res in resolutions
                }
                grid = grids[config.grid_resolution_deg]
                lut = self._build_lut(site, spec, grids, bounds)
                sens = {res: grids[res] for res in resolutions if res != config.grid_resolution_deg}
            self._states[site] = _SiteState(
                site_id=site,
                spec=spec,
                grid=grid,
                sens_grids=sens,
                lut=lut,
                velocities=velocities,
                members=members,
                unit_count=unit_count,
                total_particles=int(rows_site.size),
                domain_lonlat_bounds=bounds,
                hours_vm=np.zeros(velocities.size * members.size * grid.cell_count, dtype=np.int64),
                hours_unit=np.zeros(unit_count * grid.cell_count, dtype=np.int64),
                sens_hours={res: np.zeros(g.cell_count, dtype=np.int64) for res, g in sens.items()},
                snapshot_hist={
                    h: np.zeros(grid.cell_count, dtype=np.int64) for h in config.lookback_snapshot_hours
                },
                snapshot_active={h: 0 for h in config.lookback_snapshot_hours},
                snapshot_outside={h: 0 for h in config.lookback_snapshot_hours},
            )

    # -- 建構輔助 ---------------------------------------------------------

    def _site_project(self, site: str) -> Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        """該站的「平面公尺 → (lon, lat)」函式（包住 ``lonlat_projector``）。"""

        def project(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            x = np.asarray(x, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)
            return self._projector(np.full(x.shape, site), x, y)

        return project

    def _domain_lonlat_bounds(self, site: str, spec: FootprintSiteSpec) -> tuple[float, float, float, float]:
        coords = np.asarray(shapely.get_coordinates(spec.flow_domain.boundary), dtype=np.float64)
        if coords.size == 0:
            raise FootprintError(f"站 {site} 的 flow domain 幾何為空")
        lon, lat = self._projector(np.full(coords.shape[0], site), coords[:, 0], coords[:, 1])
        return float(np.min(lon)), float(np.max(lon)), float(np.min(lat)), float(np.max(lat))

    def _build_lut(
        self,
        site: str,
        spec: FootprintSiteSpec,
        grids: Mapping[float, FootprintGrid],
        lonlat_bounds: tuple[float, float, float, float],
    ) -> _Lut:
        x_min, y_min, x_max, y_max = spec.flow_domain.bounds
        pad = max(0.05 * max(x_max - x_min, y_max - y_min), 2000.0)
        cell_m = self._config.lut_cell_m
        x0, y0 = x_min - pad, y_min - pad
        nx = int(math.ceil((x_max + pad - x0) / cell_m))
        ny = int(math.ceil((y_max + pad - y0) / cell_m))
        xs = x0 + (np.arange(nx) + 0.5) * cell_m
        ys = y0 + (np.arange(ny) + 0.5) * cell_m
        xx, yy = np.meshgrid(xs, ys)
        lon, lat = self._projector(np.full(xx.size, site), xx.ravel(), yy.ravel())
        cells = {res: grid.flat_index(lon, lat) for res, grid in grids.items()}
        return _Lut(x0=x0, y0=y0, cell_m=cell_m, nx=nx, ny=ny, cells=cells)

    # -- 串流累加 ---------------------------------------------------------

    def add_batch(self, batch_id: str, rows: np.ndarray, result: Any) -> None:
        """累加一個批次：``rows`` 為批次粒子在身分表的全域列位置。"""

        status = np.asarray(result.trajectory_status)
        tx = np.asarray(result.trajectory_x_m)
        ty = np.asarray(result.trajectory_y_m)
        if status.ndim != 2 or status.shape != tx.shape or status.shape != ty.shape:
            raise FootprintError(f"批次 {batch_id!r} 的軌跡陣列形狀不一致")
        if status.shape[1] != rows.size:
            raise FootprintError(f"批次 {batch_id!r} 的軌跡欄數與 row_particle_index 長度不一致")
        final_status = np.asarray(result.final_status)
        final_x = np.asarray(result.final_x_m)
        final_y = np.asarray(result.final_y_m)
        site_of_cols = self._site_of_row[rows]
        for site_index in np.unique(site_of_cols):
            state = self._states[self._site_names[int(site_index)]]
            cols = None if np.all(site_of_cols == site_index) else np.flatnonzero(site_of_cols == site_index)
            self._add_site(state, rows, cols, status, tx, ty, final_status, final_x, final_y)

    def _add_site(
        self,
        state: _SiteState,
        rows: np.ndarray,
        cols: np.ndarray | None,
        status: np.ndarray,
        tx: np.ndarray,
        ty: np.ndarray,
        final_status: np.ndarray,
        final_x: np.ndarray,
        final_y: np.ndarray,
    ) -> None:
        cfg = self._config
        if cols is not None:
            rows = rows[cols]
            status, tx, ty = status[:, cols], tx[:, cols], ty[:, cols]
            final_status, final_x, final_y = final_status[cols], final_x[cols], final_y[cols]

        valid = status == 0
        if not cfg.count_arrival_sample:
            valid = valid.copy()
            valid[0, :] = False
        t_idx, c_idx = np.nonzero(valid)
        if t_idx.size:
            x = tx[t_idx, c_idx]
            y = ty[t_idx, c_idx]
            if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
                raise FootprintError(f"{state.site_id}：status==0 的樣本含非有限座標")
            if state.lut is None:
                cell = state.grid.flat_index_xy(x, y).astype(np.int64)
                in_lut = np.ones(cell.shape, dtype=bool)
            else:
                cell, in_lut = state.lut.lookup(x, y, state.grid.resolution_deg)
            state.hours_outside_lut += int(np.count_nonzero(~in_lut))
            in_grid = cell >= 0
            state.hours_outside_grid += int(np.count_nonzero(in_lut & ~in_grid))
            ncell = state.grid.cell_count
            slot_pts = self._slot_of_row[rows][c_idx][in_grid]
            unit_pts = self._unit_of_row[rows][c_idx][in_grid]
            cell_ok = cell[in_grid]
            state.hours_vm += np.bincount(slot_pts * ncell + cell_ok, minlength=state.hours_vm.size)
            state.hours_unit += np.bincount(unit_pts * ncell + cell_ok, minlength=state.hours_unit.size)
            for res, hours in state.sens_hours.items():
                if state.lut is None:
                    cell_r = state.sens_grids[res].flat_index_xy(x, y)
                else:
                    cell_r, _ = state.lut.lookup(x, y, res)
                cell_r = cell_r[cell_r >= 0]
                hours += np.bincount(cell_r, minlength=hours.size)

        n_time = status.shape[0]
        for h in cfg.lookback_snapshot_hours:
            if n_time <= h:
                continue
            sel = status[h] == 0
            count = int(np.count_nonzero(sel))
            if count:
                xs, ys = tx[h][sel], ty[h][sel]
                if not (np.all(np.isfinite(xs)) and np.all(np.isfinite(ys))):
                    raise FootprintError(f"{state.site_id}：快照 {h} 小時 status==0 的樣本含非有限座標")
                if state.lut is None:
                    snap_cell = state.grid.flat_index_xy(xs, ys).astype(np.int64)
                else:
                    snap_cell, _ = state.lut.lookup(xs, ys, state.grid.resolution_deg)
                inside = snap_cell >= 0
                state.snapshot_hist[h] += np.bincount(snap_cell[inside], minlength=state.grid.cell_count)
                state.snapshot_outside[h] += int(np.count_nonzero(~inside))
                state.snapshot_active[h] += count

        crossed = final_status == NATIVE_STATUS_LEFT_FLOW_DOMAIN
        if np.any(crossed) and state.spec.flow_open_boundary is not None:
            points = shapely.points(final_x[crossed], final_y[crossed])
            state.arc_s_m.append(np.asarray(shapely.line_locate_point(state.spec.flow_open_boundary, points)))
            state.crossed_count += int(np.count_nonzero(crossed))

    # -- 收尾 -------------------------------------------------------------

    def finalize(
        self,
        *,
        receptor_centers_xy: Mapping[str, tuple[float, float]],
        physical_case_id: str | None = None,
        ensemble_member_count_by_site: Mapping[str, int] | None = None,
    ) -> FootprintProducts:
        """輸出全部表格與 manifest 區塊；正規化總和不為 1 時 fail-closed。"""

        from . import postprocess as pp  # 延後匯入以避免循環相依

        cfg = self._config
        tables_rows: dict[str, list[dict[str, Any]]] = {
            "residence": [],
            "residence_sv": [],
            "residence_half": [],
            "hdr": [],
            "grid": [],
            "grid_corners": [],
            "arc": [],
            "edge": [],
            "geometry": [],
            "snapshot": [],
            "snapshot_density": [],
            "sea_mask": [],
        }
        summary_rows: list[dict[str, Any]] = []
        manifest_sites: dict[str, Any] = {}
        warnings: list[str] = []

        for site in self._site_names:
            state = self._states[site]
            info = self._finalize_site(
                state,
                pp,
                tables_rows,
                receptor_xy=receptor_centers_xy.get(site),
                physical_case_id=physical_case_id,
                member_count=(ensemble_member_count_by_site or {}).get(site),
                warnings=warnings,
            )
            summary_rows.append(info["summary"])
            manifest_sites[site] = info["manifest"]

        tables = {
            "residence_density_by_site": _residence_table(tables_rows["residence"], velocity=False),
            "residence_density_by_site_settling_velocity": _residence_table(
                tables_rows["residence_sv"], velocity=True
            ),
            "residence_density_by_site_member_half": _residence_half_table(tables_rows["residence_half"]),
            "residence_hdr_by_site": _hdr_table(tables_rows["hdr"]),
            "footprint_grid_by_site": pa.Table.from_pylist(tables_rows["grid"]),
            "lookback_snapshot_by_site": _snapshot_table(tables_rows["snapshot"]),
            "lookback_snapshot_density_by_site": _snapshot_density_table(tables_rows["snapshot_density"]),
            "footprint_sea_mask_by_site": _sea_mask_table(tables_rows["sea_mask"]),
            "boundary_departure_along_arc_by_site": _arc_table(tables_rows["arc"]),
            "boundary_edge_intervals_by_site": _edge_table(tables_rows["edge"]),
            "footprint_geometry_by_site": _geometry_table(tables_rows["geometry"]),
        }
        if cfg.grid_cell_km is not None:
            tables["footprint_grid_corners_by_site"] = _grid_corners_table(tables_rows["grid_corners"])
        manifest = {
            "definition_zh": (
                "停留時數＝逐小時、仍在漂流（status==0）的粒子樣本數；終止後凍結的位置不計入；"
                "正規化＝網格停留時數÷該站總停留時數，密度＝正規化值÷網格面積（km²）；"
                "HDR＝依密度由高到低累加至 50／75／90% 停留時數的最小面積網格集合。"
            ),
            "snapshot_definition_zh": (
                "回溯快照＝該回溯時刻仍在漂流（status==0）的粒子位置，累加到與停留時數密度相同的網格"
                "（整數計數、面積加權），分母是該時刻仍在漂流的粒子數；可選的平滑為海域遮罩下的正規化卷積"
                "G*(d·m)/G*m（高斯核，寬度以公里指定），平滑後依總質量重新正規化；"
                "陸地與 flow domain 外的格密度恆為 0；HDR 算法與停留時數密度相同；"
                "仍在漂流粒子數低於門檻時不出密度。"
            ),
            "sea_mask_source": self._land_source,
            "arrival_sample_included": bool(cfg.count_arrival_sample),
            "kde_by_site_note_zh": KDE_NOTE_ZH,
            "config": cfg.to_manifest_dict(),
            "earth_radius_km": EARTH_RADIUS_KM,
            "normalized_sum_tolerance": NORMALIZED_SUM_TOLERANCE,
            "sites": manifest_sites,
            "warnings": warnings,
            "bathymetry": {
                "available_sites": sorted(self._bathymetry),
                "missing_sites": sorted(set(self._site_names) - set(self._bathymetry)),
            },
        }
        if cfg.grid_cell_km is not None:
            manifest["grid_definition_zh"] = (
                f"網格＝流場區域局部等距方位（AEQD）平面上邊長 {cfg.grid_cell_km:g} km 的正方形規則網格，"
                f"格線落在平面座標 x、y 為 {cfg.grid_cell_km:g} km 整數倍之處（原點為投影中心），"
                f"每格面積正好 {cfg.grid_cell_km**2:g} km²；格心與格角的經緯度由區域投影反轉換得到。"
            )
        return FootprintProducts(tables=tables, summary_rows=summary_rows, manifest=manifest)

    # -- 單站收尾 ---------------------------------------------------------

    def _finalize_site(
        self,
        state: _SiteState,
        pp: Any,
        rows_out: dict[str, list[dict[str, Any]]],
        *,
        receptor_xy: tuple[float, float] | None,
        physical_case_id: str | None,
        member_count: int | None,
        warnings: list[str],
    ) -> dict[str, Any]:
        cfg = self._config
        site = state.site_id
        grid = state.grid
        ncell = grid.cell_count
        area = grid.cell_area_km2()
        lon_c, lat_c = grid.cell_centers_lonlat()
        V, M = state.velocities.size, state.members.size
        hours_vm = state.hours_vm.reshape(V, M, ncell)
        hours_all = hours_vm.sum(axis=(0, 1))
        total_hours = int(hours_all.sum())
        if total_hours <= 0:
            raise FootprintError(f"{site}：沒有任何有效（status==0）停留時數，無法建立停留時數密度")

        # 一致性：兩種累加方式的總時數必須相同。
        if int(state.hours_unit.sum()) != total_hours:
            raise FootprintError(f"{site}：以系集成員與以到達單元累加的總停留時數不一致")

        def emit(
            group_kind: str, group_value: str, hours: np.ndarray, extra: dict[str, Any], key: str
        ) -> float:
            total = int(hours.sum())
            idx = np.flatnonzero(hours > 0)
            if total == 0:
                return 0.0
            fraction = hours[idx] / total
            density = fraction / area[idx]
            norm_sum = float(fraction.sum())
            if (
                abs(norm_sum - 1.0) > NORMALIZED_SUM_TOLERANCE
                or abs(float((density * area[idx]).sum()) - 1.0) > NORMALIZED_SUM_TOLERANCE
            ):
                raise FootprintError(
                    f"{site}／{group_kind}={group_value}：正規化停留場總和 {norm_sum!r} 與 1 相差超過 "
                    f"{NORMALIZED_SUM_TOLERANCE}"
                )
            for i, cell in enumerate(idx.tolist()):
                rows_out[key].append(
                    {
                        "study_site_id": site,
                        "group_kind": group_kind,
                        "group_value": group_value,
                        **extra,
                        "cell_ix": int(cell % grid.nx),
                        "cell_iy": int(cell // grid.nx),
                        "lon_center_deg": float(lon_c[cell]),
                        "lat_center_deg": float(lat_c[cell]),
                        "cell_area_km2": float(area[cell]),
                        "particle_hours": int(hours[cell]),
                        "fraction_of_total": float(fraction[i]),
                        "density_fraction_per_km2": float(density[i]),
                    }
                )
            for hdr in compute_hdr(hours, area, cfg.hdr_levels):
                rows_out["hdr"].append(
                    {
                        "study_site_id": site,
                        "group_kind": group_kind,
                        "group_value": group_value,
                        "level": hdr["level"],
                        "density_threshold_fraction_per_km2": hdr["density_threshold_fraction_per_km2"],
                        "area_km2": hdr["area_km2"],
                        "cell_count": hdr["cell_count"],
                        "achieved_fraction": hdr["achieved_fraction"],
                    }
                )
            return norm_sum

        norm_sum_all = emit(GROUP_ALL, "all", hours_all, {}, "residence")
        for v_i, velocity in enumerate(state.velocities.tolist()):
            emit(
                GROUP_SETTLING_VELOCITY,
                repr(float(velocity)),
                hours_vm[v_i].sum(axis=0),
                {"settling_velocity_mps": float(velocity)},
                "residence_sv",
            )
        half = M // 2
        first_members, second_members = state.members[:half], state.members[half:]
        hours_first = hours_vm[:, :half].sum(axis=(0, 1)) if half > 0 else np.zeros(ncell, dtype=np.int64)
        hours_second = hours_vm[:, half:].sum(axis=(0, 1))
        for name, hrs, ids in (
            ("first", hours_first, first_members),
            ("second", hours_second, second_members),
        ):
            emit(
                GROUP_MEMBER_HALF,
                name,
                hrs,
                {"member_ids": ",".join(str(int(m)) for m in ids)},
                "residence_half",
            )

        hdr_all = compute_hdr(hours_all, area, cfg.hdr_levels)
        hdr_area = {round(h["level"], 6): h["area_km2"] for h in hdr_all}

        # 質心、與受體的距離／方位。
        fraction_all = hours_all / total_hours
        centroid_lon = float(np.sum(fraction_all * lon_c))
        centroid_lat = float(np.sum(fraction_all * lat_c))
        receptor_lonlat: tuple[float, float] | None = None
        distance_km = bearing = None
        if receptor_xy is not None:
            r_lon, r_lat = self._projector(
                np.asarray([site]), np.asarray([receptor_xy[0]]), np.asarray([receptor_xy[1]])
            )
            receptor_lonlat = (float(r_lon[0]), float(r_lat[0]))
            distance_km, bearing = haversine_km_and_bearing_deg(
                receptor_lonlat[0], receptor_lonlat[1], centroid_lon, centroid_lat
            )

        # 水深。
        depth_fraction = depth_missing = None
        bathy = self._bathymetry.get(site)
        if bathy is not None:
            occupied = np.flatnonzero(hours_all > 0)
            depth = bathy.depth_at(lon_c[occupied], lat_c[occupied])
            finite = np.isfinite(depth)
            occupied_hours = hours_all[occupied].astype(np.float64)
            finite_hours = float(occupied_hours[finite].sum())
            depth_missing = float(1.0 - finite_hours / occupied_hours.sum())
            if finite_hours > 0:
                depth_fraction = float(
                    occupied_hours[finite & (depth < cfg.depth_threshold_m)].sum() / finite_hours
                )

        # 成員分半與雜訊底。
        half_tvd = half_hdr50_rel = None
        if first_members.size and second_members.size:
            half_tvd = total_variation_distance(hours_first, hours_second)
            a1 = _hdr_area(hours_first, area, 0.5)
            a2 = _hdr_area(hours_second, area, 0.5)
            if a1 is not None and a2 is not None and (a1 + a2) > 0:
                half_hdr50_rel = abs(a1 - a2) / ((a1 + a2) / 2.0)
        rng = np.random.default_rng([cfg.noise_floor_seed, zlib.crc32(site.encode("utf-8"))])
        by_member = hours_vm.sum(axis=0)  # (M, ncell)
        noise_member = _noise_floor(by_member, cfg.noise_floor_repeats, rng)
        by_unit = state.hours_unit.reshape(state.unit_count, ncell)
        noise_unit = _noise_floor(by_unit, cfg.noise_floor_repeats, rng)

        # 網格敏感度。
        sens_hdr: dict[float, dict[float, float | None]] = {}
        is_meter = isinstance(grid, FootprintMeterGrid)
        primary_key = grid.cell_km if is_meter else grid.resolution_deg
        all_grids = {
            primary_key: (grid, hours_all),
            **{r: (g, state.sens_hours[r]) for r, g in state.sens_grids.items()},
        }
        for res, (g, hrs) in sorted(all_grids.items()):
            g_area = g.cell_area_km2()
            sens_hdr[res] = {lv: _hdr_area(hrs, g_area, lv) for lv in (0.5, 0.9)}
            if int(hrs.sum()) != total_hours and res != primary_key:
                unit_text = f"{res:g} km" if is_meter else f"{res}°"
                warnings.append(
                    f"{site}：{unit_text} 網格的總停留時數 {int(hrs.sum())} 與主網格 {total_hours} 不同"
                    "（部分停留時數落在該解析度網格外）"
                )

        # 回溯快照。
        snapshot_info, sea_info = self._finalize_snapshots(state, rows_out, hours_all, area, warnings)

        # 邊界弧長分布。
        arc_info = self._finalize_arc(state, rows_out["arc"], rows_out["edge"], rows_out["geometry"])

        # 幾何：flow domain、受體。
        self._emit_geometry(state, rows_out["geometry"], receptor_lonlat, receptor_xy)

        if is_meter:
            rows_out["grid"].append(
                {
                    "study_site_id": site,
                    "grid_kind": METER_GRID_KIND,
                    "cell_m": grid.cell_m,
                    "x0_m": grid.x0_m,
                    "y0_m": grid.y0_m,
                    "nx": grid.nx,
                    "ny": grid.ny,
                    "domain_lon_min_deg": state.domain_lonlat_bounds[0],
                    "domain_lon_max_deg": state.domain_lonlat_bounds[1],
                    "domain_lat_min_deg": state.domain_lonlat_bounds[2],
                    "domain_lat_max_deg": state.domain_lonlat_bounds[3],
                    "total_particle_hours": total_hours,
                    "hours_outside_grid": state.hours_outside_grid,
                    "hours_outside_lookup": 0,
                }
            )
            corner_lon, corner_lat = grid.corners_lonlat()
            for iy in range(grid.ny + 1):
                for ix in range(grid.nx + 1):
                    rows_out["grid_corners"].append(
                        {
                            "study_site_id": site,
                            "corner_ix": ix,
                            "corner_iy": iy,
                            "lon_deg": float(corner_lon[iy, ix]),
                            "lat_deg": float(corner_lat[iy, ix]),
                        }
                    )
        else:
            rows_out["grid"].append(
                {
                    "study_site_id": site,
                    "resolution_deg": grid.resolution_deg,
                    "lon0_deg": grid.lon0,
                    "lat0_deg": grid.lat0,
                    "nx": grid.nx,
                    "ny": grid.ny,
                    "domain_lon_min_deg": state.domain_lonlat_bounds[0],
                    "domain_lon_max_deg": state.domain_lonlat_bounds[1],
                    "domain_lat_min_deg": state.domain_lonlat_bounds[2],
                    "domain_lat_max_deg": state.domain_lonlat_bounds[3],
                    "total_particle_hours": total_hours,
                    "hours_outside_grid": state.hours_outside_grid,
                    "hours_outside_lookup": state.hours_outside_lut,
                    "lookup_cell_m": cfg.lut_cell_m,
                }
            )
        if state.hours_outside_grid or state.hours_outside_lut:
            warnings.append(
                f"{site}：有 {state.hours_outside_grid + state.hours_outside_lut} 筆有效停留時數落在網格外，"
                "未計入正規化分母"
            )

        summary = {
            "study_site_id": site,
            "physical_case_id": physical_case_id,
            "ensemble_member_count": int(member_count) if member_count is not None else int(M),
            "total_particle_count": state.total_particles,
            "total_particle_hours": total_hours,
            **(
                {"grid_cell_km": grid.cell_km}
                if is_meter
                else {"grid_resolution_deg": grid.resolution_deg}
            ),
            "hdr50_area_km2": hdr_area.get(0.5),
            "hdr75_area_km2": hdr_area.get(0.75),
            "hdr90_area_km2": hdr_area.get(0.9),
            "centroid_lon_deg": centroid_lon,
            "centroid_lat_deg": centroid_lat,
            "receptor_center_lon_deg": receptor_lonlat[0] if receptor_lonlat else None,
            "receptor_center_lat_deg": receptor_lonlat[1] if receptor_lonlat else None,
            "centroid_distance_from_receptor_km": distance_km,
            "centroid_bearing_from_receptor_deg": bearing,
            "hours_fraction_depth_below_threshold": depth_fraction,
            "depth_threshold_m": cfg.depth_threshold_m,
            "hours_fraction_depth_missing": depth_missing,
        }
        for h in cfg.lookback_snapshot_hours:
            info = snapshot_info[h]
            summary[f"active_fraction_{h}h"] = info["active_fraction"]
            summary[f"active_particle_count_{h}h"] = info["active_particle_count"]
            summary[f"snapshot_hdr50_area_km2_{h}h"] = info["hdr_area_km2"].get("0.5")
        summary.update(
            {
                "member_half_tvd": half_tvd,
                "member_half_hdr50_area_relative_difference": half_hdr50_rel,
                "noise_tvd_member_partition_mean": noise_member[0],
                "noise_tvd_member_partition_p95": noise_member[1],
                "noise_tvd_unit_split_mean": noise_unit[0],
                "noise_tvd_unit_split_p95": noise_unit[1],
            }
        )
        for res, values in sens_hdr.items():
            tag = _fmt_km(res) if is_meter else _fmt_res(res)
            summary[f"grid_sensitivity_hdr50_area_km2_{tag}"] = values[0.5]
            summary[f"grid_sensitivity_hdr90_area_km2_{tag}"] = values[0.9]

        manifest = {
            "grid": (
                {
                    "kind": METER_GRID_KIND,
                    "cell_m": grid.cell_m,
                    "x0_m": grid.x0_m,
                    "y0_m": grid.y0_m,
                    "nx": grid.nx,
                    "ny": grid.ny,
                    "cell_area_km2": grid.cell_km**2,
                }
                if is_meter
                else {
                    "resolution_deg": grid.resolution_deg,
                    "lon0_deg": grid.lon0,
                    "lat0_deg": grid.lat0,
                    "nx": grid.nx,
                    "ny": grid.ny,
                }
            ),
            "total_particle_hours": total_hours,
            "normalized_sum": norm_sum_all,
            "hours_outside_grid": state.hours_outside_grid,
            "hours_outside_lookup": state.hours_outside_lut,
            "settling_velocity_count": int(V),
            "ensemble_member_count": int(M),
            "member_half": {
                "first_member_ids": [int(m) for m in first_members],
                "second_member_ids": [int(m) for m in second_members],
            },
            "arrival_unit_count": state.unit_count,
            "snapshots": {str(h): snapshot_info[h] for h in cfg.lookback_snapshot_hours},
            "sea_mask": sea_info,
            "boundary_departure": arc_info,
            "bathymetry_available": bathy is not None,
        }
        return {"summary": summary, "manifest": manifest}

    def _domain_lonlat_polygon(self, state: _SiteState) -> BaseGeometry:
        """flow domain（公尺制）投影到經緯度後的多邊形（含內環）。"""

        site = state.site_id

        def project(coords: np.ndarray) -> np.ndarray:
            lon, lat = self._projector(np.full(coords.shape[0], site), coords[:, 0], coords[:, 1])
            return np.column_stack((lon, lat))

        domain = state.spec.flow_domain
        polys = list(domain.geoms) if hasattr(domain, "geoms") else [domain]
        out = []
        for poly in polys:
            shell = project(np.asarray(poly.exterior.coords, dtype=np.float64))
            holes = [project(np.asarray(r.coords, dtype=np.float64)) for r in poly.interiors]
            out.append(shapely.Polygon(shell, holes))
        geom = shapely.union_all(out)
        return shapely.make_valid(geom) if not geom.is_valid else geom

    def _sea_mask(
        self, state: _SiteState, observed: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """回傳 ``(海域遮罩, 海域面積占比, 統計)``，陣列長度為網格格數。

        海域＝flow domain 內、且扣掉陸地（海岸線）之後；海域面積占比不小於
        ``land_mask_min_sea_fraction`` 的格為海域格。曾有粒子出現的格（``observed``）必為海域，
        避免海岸邊被 200 m 查表量化到相鄰陸地格的粒子質量被遮罩吃掉。
        """

        cfg = self._config
        grid = state.grid
        n = grid.cell_count
        if isinstance(grid, FootprintMeterGrid):
            boxes = grid.cell_boxes_lonlat()
            corner_lon, corner_lat = grid.corners_lonlat()
            margin = 0.05
            clip_rect = (
                float(corner_lon.min()) - margin,
                float(corner_lat.min()) - margin,
                float(corner_lon.max()) + margin,
                float(corner_lat.max()) + margin,
            )
        else:
            idx = np.arange(n)
            ix, iy = idx % grid.nx, idx // grid.nx
            res = grid.resolution_deg
            boxes = shapely.box(
                grid.lon0 + ix * res,
                grid.lat0 + iy * res,
                grid.lon0 + (ix + 1) * res,
                grid.lat0 + (iy + 1) * res,
            )
            lon_min, lat_min = grid.lon0, grid.lat0
            clip_rect = (
                lon_min - res,
                lat_min - res,
                lon_min + (grid.nx + 1) * res,
                lat_min + (grid.ny + 1) * res,
            )
        sea = self._domain_lonlat_polygon(state)
        if self._land is not None:
            clip = shapely.clip_by_rect(self._land, *clip_rect)
            if not clip.is_empty:
                sea = shapely.difference(sea, clip)
        fraction = np.zeros(n, dtype=np.float64)
        if not sea.is_empty:
            shapely.prepare(sea)
            full = shapely.contains(sea, boxes)
            partial = shapely.intersects(sea, boxes) & ~full
            fraction[full] = 1.0
            if partial.any():
                inter = shapely.intersection(boxes[partial], sea)
                fraction[partial] = shapely.area(inter) / shapely.area(boxes[partial])
        geometric = fraction >= cfg.land_mask_min_sea_fraction
        mask = geometric | observed
        stats = {
            "min_sea_fraction": cfg.land_mask_min_sea_fraction,
            "land_source": self._land_source,
            "cell_count": int(n),
            "sea_cell_count": int(mask.sum()),
            "land_or_outside_domain_cell_count": int(n - mask.sum()),
            "cells_sea_by_observation_only": int((mask & ~geometric).sum()),
            "cells_partly_land_but_sea": int((mask & (fraction < 1.0)).sum()),
        }
        return mask, fraction, stats

    def _finalize_snapshots(
        self,
        state: _SiteState,
        rows_out: dict[str, list[dict[str, Any]]],
        hours_all: np.ndarray,
        area: np.ndarray,
        warnings: list[str],
    ) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
        cfg = self._config
        site = state.site_id
        grid = state.grid
        ncell = grid.cell_count
        lon_c, lat_c = grid.cell_centers_lonlat()

        observed = hours_all > 0
        for h in cfg.lookback_snapshot_hours:
            observed = observed | (state.snapshot_hist[h] > 0)
        mask, sea_fraction, sea_info = self._sea_mask(state, observed)
        for cell in range(ncell):
            rows_out["sea_mask"].append(
                {
                    "study_site_id": site,
                    "cell_ix": int(cell % grid.nx),
                    "cell_iy": int(cell // grid.nx),
                    "lon_center_deg": float(lon_c[cell]),
                    "lat_center_deg": float(lat_c[cell]),
                    "sea_area_fraction": float(sea_fraction[cell]),
                    "is_sea_cell": bool(mask[cell]),
                    "observed_particle": bool(observed[cell]),
                }
            )

        # 平滑核：公里 -> 格。緯度方向每格 res°×111.195 km，經度方向再乘 cos(平均緯度)。
        if isinstance(grid, FootprintMeterGrid):
            sigma_cells = (cfg.snapshot_smoothing_km / grid.cell_km, cfg.snapshot_smoothing_km / grid.cell_km)
        else:
            dy_km = grid.resolution_deg * math.radians(1.0) * EARTH_RADIUS_KM
            mean_lat = float(lat_c.mean())
            dx_km = dy_km * math.cos(math.radians(mean_lat))
            sigma_cells = (cfg.snapshot_smoothing_km / dy_km, cfg.snapshot_smoothing_km / dx_km)
        sea_info = {
            **sea_info,
            "smoothing_sigma_km": cfg.snapshot_smoothing_km,
            "smoothing_sigma_cells_lat_lon": [sigma_cells[0], sigma_cells[1]],
        }

        info_by_hour: dict[int, dict[str, Any]] = {}
        for h in cfg.lookback_snapshot_hours:
            counts = state.snapshot_hist[h]
            n_active = int(state.snapshot_active[h])
            n_outside = int(state.snapshot_outside[h])
            n_in = int(counts.sum())
            if n_in + n_outside != n_active:
                raise FootprintError(
                    f"{site}：回溯 {h} 小時快照的網格計數 {n_in}＋網格外 {n_outside} 不等於 {n_active}"
                )
            if n_outside:
                warnings.append(f"{site}：回溯 {h} 小時快照有 {n_outside} 顆粒子落在網格範圍外")
            active_fraction = n_active / state.total_particles
            available = n_active >= cfg.snapshot_minimum_particle_count and n_in > 0
            row: dict[str, Any] = {
                "study_site_id": site,
                "snapshot_hours": int(h),
                "snapshot_days": h / 24.0,
                "site_particle_count": state.total_particles,
                "active_particle_count": n_active,
                "active_fraction": active_fraction,
                "status": "available" if available else "insufficient_active_particles",
                "minimum_active_particle_count": cfg.snapshot_minimum_particle_count,
                "smoothing_sigma_km": cfg.snapshot_smoothing_km,
                "mass_rescale_factor": None,
                "hdr50_area_km2": None,
                "hdr75_area_km2": None,
                "hdr90_area_km2": None,
                "particles_outside_grid": n_outside,
            }
            info = {
                "active_fraction": active_fraction,
                "active_particle_count": n_active,
                "status": row["status"],
                "minimum_active_particle_count": cfg.snapshot_minimum_particle_count,
                "mass_rescale_factor": None,
                "hdr_area_km2": {},
            }
            if available:
                raw_density = (counts / n_in / area).reshape(grid.ny, grid.nx)
                smooth = smooth_masked_density(
                    raw_density, mask.reshape(grid.ny, grid.nx), sigma_cells
                ).ravel()
                mass = float((smooth * area).sum())
                if not mass > 0.0:
                    raise FootprintError(f"{site}：回溯 {h} 小時快照平滑後總質量為 0")
                factor = 1.0 / mass
                fraction = smooth * area * factor
                density = fraction / area
                if abs(float(fraction.sum()) - 1.0) > NORMALIZED_SUM_TOLERANCE:
                    raise FootprintError(
                        f"{site}：回溯 {h} 小時快照正規化總和 {fraction.sum()!r} 與 1 相差超過容差"
                    )
                if np.any(fraction[~mask] != 0.0):
                    raise FootprintError(f"{site}：回溯 {h} 小時快照在陸地格出現非零密度")
                row["mass_rescale_factor"] = info["mass_rescale_factor"] = factor
                for cell in np.flatnonzero(fraction > 0):
                    rows_out["snapshot_density"].append(
                        {
                            "study_site_id": site,
                            "snapshot_hours": int(h),
                            "cell_ix": int(cell % grid.nx),
                            "cell_iy": int(cell // grid.nx),
                            "lon_center_deg": float(lon_c[cell]),
                            "lat_center_deg": float(lat_c[cell]),
                            "cell_area_km2": float(area[cell]),
                            "particle_count": int(counts[cell]),
                            "fraction_of_active": float(fraction[cell]),
                            "density_fraction_per_km2": float(density[cell]),
                        }
                    )
                for hdr in compute_hdr(fraction, area, cfg.hdr_levels):
                    rows_out["hdr"].append(
                        {
                            "study_site_id": site,
                            "group_kind": GROUP_SNAPSHOT,
                            "group_value": str(h),
                            "level": hdr["level"],
                            "density_threshold_fraction_per_km2": hdr["density_threshold_fraction_per_km2"],
                            "area_km2": hdr["area_km2"],
                            "cell_count": hdr["cell_count"],
                            "achieved_fraction": hdr["achieved_fraction"],
                        }
                    )
                    row[f"hdr{int(round(hdr['level'] * 100))}_area_km2"] = hdr["area_km2"]
                    info["hdr_area_km2"][str(hdr["level"])] = hdr["area_km2"]
            rows_out["snapshot"].append(row)
            info_by_hour[h] = info
        return info_by_hour, sea_info

    def _finalize_arc(
        self,
        state: _SiteState,
        arc_rows: list[dict[str, Any]],
        edge_rows: list[dict[str, Any]],
        geometry_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        cfg = self._config
        site = state.site_id
        line = state.spec.flow_open_boundary
        if line is None:
            return {"available": False}
        length_km = float(line.length) / 1000.0
        s_km = np.sort(np.concatenate(state.arc_s_m)) / 1000.0 if state.arc_s_m else np.empty(0)
        n_bins = max(1, int(math.ceil(length_km / cfg.arc_bin_km - 1e-9)))
        edges = np.arange(n_bins + 1) * cfg.arc_bin_km
        edges[-1] = max(edges[-1], length_km)
        counts = np.histogram(np.clip(s_km, 0.0, edges[-1]), bins=edges)[0]
        total = int(counts.sum())
        cum = np.cumsum(counts)
        for i in range(n_bins):
            arc_rows.append(
                {
                    "study_site_id": site,
                    "bin_index": i,
                    "s_start_km": float(edges[i]),
                    "s_end_km": float(edges[i + 1]),
                    "crossed_particle_count": int(counts[i]),
                    "fraction": float(counts[i] / total) if total else 0.0,
                    "cumulative_fraction": float(cum[i] / total) if total else 0.0,
                    "boundary_length_km": length_km,
                    "bin_km": cfg.arc_bin_km,
                }
            )
        for kind, name, a, b in self._edge_intervals(state, length_km):
            edge_rows.append(
                {"study_site_id": site, "kind": kind, "name": name, "s_start_km": a, "s_end_km": b}
            )
        return {
            "available": True,
            "boundary_length_km": length_km,
            "crossed_particle_count": total,
            "bin_km": cfg.arc_bin_km,
        }

    def _edge_intervals(self, state: _SiteState, length_km: float) -> list[tuple[str, str, float, float]]:
        """把開放邊界依弧長切成北／東／南／西各邊的區間，並標出角落位置。"""

        site = state.site_id
        line = state.spec.flow_open_boundary
        step_km = min(1.0, length_km / 2000.0) if length_km > 0 else 1.0
        s_km = np.append(np.arange(0.0, length_km, step_km), length_km)
        pts = shapely.line_interpolate_point(line, s_km * 1000.0)
        xy = shapely.get_coordinates(pts)
        lon, lat = self._projector(np.full(xy.shape[0], site), xy[:, 0], xy[:, 1])
        lon_min, lon_max, lat_min, lat_max = state.domain_lonlat_bounds
        mid_lat = math.radians((lat_min + lat_max) / 2.0)
        scale = math.cos(mid_lat)
        dist = np.stack(
            [
                (lon - lon_min) * scale,  # west
                (lon_max - lon) * scale,  # east
                (lat - lat_min),  # south
                (lat_max - lat),  # north
            ]
        )
        names = np.array(["west", "east", "south", "north"])
        label = names[np.argmin(dist, axis=0)]
        # 短於 3 個取樣的碎段併入前一段，避免角落附近的雜訊。
        runs: list[list[Any]] = []
        for i, lab in enumerate(label.tolist()):
            if runs and runs[-1][0] == lab:
                runs[-1][2] = i
            else:
                runs.append([lab, i, i])
        merged: list[list[Any]] = []
        for run in runs:
            if merged and ((run[2] - run[1] + 1) < 3 or merged[-1][0] == run[0]):
                merged[-1][2] = run[2]
            else:
                merged.append(run)
        if len(merged) > 1 and (merged[0][2] - merged[0][1] + 1) < 3:
            # 起點恰在角落時，第一個取樣可能被歸到相鄰邊；過短的開頭併入下一段。
            merged[1][1] = merged[0][1]
            del merged[0]
        last = s_km.size - 1
        corners = [float(s_km[run[2]] + s_km[min(run[2] + 1, last)]) / 2.0 for run in merged[:-1]]
        bounds = [0.0, *corners, float(length_km)]
        result: list[tuple[str, str, float, float]] = []
        for k, run in enumerate(merged):
            result.append(("edge", run[0], bounds[k], bounds[k + 1]))
        for k, corner_s in enumerate(corners):
            result.append(("corner", f"{merged[k][0]}-{merged[k + 1][0]}", corner_s, corner_s))
        return result

    def _emit_geometry(
        self,
        state: _SiteState,
        out_rows: list[dict[str, Any]],
        receptor_lonlat: tuple[float, float] | None,
        receptor_xy: tuple[float, float] | None,
    ) -> None:
        site = state.site_id
        exterior = shapely.get_coordinates(state.spec.flow_domain.boundary)
        lon, lat = self._projector(np.full(exterior.shape[0], site), exterior[:, 0], exterior[:, 1])
        for i in range(exterior.shape[0]):
            out_rows.append(_geom_row(site, "flow_domain", 0, i, lon[i], lat[i], None, None))
        line = state.spec.flow_open_boundary
        if line is not None:
            offset_m = 0.0
            parts = list(line.geoms) if hasattr(line, "geoms") else [line]
            for p, part in enumerate(parts):
                coords = np.asarray(part.coords, dtype=np.float64)
                p_lon, p_lat = self._projector(np.full(coords.shape[0], site), coords[:, 0], coords[:, 1])
                seg = np.hypot(np.diff(coords[:, 0]), np.diff(coords[:, 1]))
                cum = offset_m + np.concatenate([[0.0], np.cumsum(seg)])
                for i in range(coords.shape[0]):
                    out_rows.append(
                        _geom_row(
                            site, "open_boundary", p, i, p_lon[i], p_lat[i], float(cum[i]) / 1000.0, None
                        )
                    )
                offset_m += float(part.length)
        identity = self._identity
        mask = np.asarray(identity.study_site_id) == site
        rec_ids, first = np.unique(np.asarray(identity.receptor_id)[mask], return_index=True)
        rx = identity.x0_m[mask][first]
        ry = identity.y0_m[mask][first]
        r_lon, r_lat = self._projector(np.full(rx.size, site), rx, ry)
        for i, rid in enumerate(rec_ids.tolist()):
            out_rows.append(_geom_row(site, "receptor", 0, i, r_lon[i], r_lat[i], None, str(rid)))
        if receptor_lonlat is not None:
            out_rows.append(
                _geom_row(site, "receptor_center", 0, 0, receptor_lonlat[0], receptor_lonlat[1], None, site)
            )


# ── 表格建構 ─────────────────────────────────────────────────────────────


def _hdr_area(hours: np.ndarray, area: np.ndarray, level: float) -> float | None:
    hdr = compute_hdr(hours, area, [level])
    return hdr[0]["area_km2"] if hdr else None


def _noise_floor(
    matrix: np.ndarray, repeats: int, rng: np.random.Generator
) -> tuple[float | None, float | None]:
    """把矩陣的列（成員或到達單元）隨機分成大小相同的兩半，重複 ``repeats`` 次，回傳 TVD 的平均與 p95。"""

    n = matrix.shape[0]
    if n < 2:
        return None, None
    half = n // 2
    values: list[float] = []
    for _ in range(repeats):
        perm = rng.permutation(n)
        tvd = total_variation_distance(matrix[perm[:half]].sum(axis=0), matrix[perm[half:]].sum(axis=0))
        if tvd is not None:
            values.append(tvd)
    if not values:
        return None, None
    return float(np.mean(values)), float(np.percentile(values, 95))


def _geom_row(
    site: str,
    element: str,
    part: int,
    vertex: int,
    lon: float,
    lat: float,
    s_km: float | None,
    label: str | None,
) -> dict[str, Any]:
    return {
        "study_site_id": site,
        "element": element,
        "part_index": part,
        "vertex_index": vertex,
        "lon_deg": float(lon),
        "lat_deg": float(lat),
        "s_km": s_km,
        "label": label,
    }


_RESIDENCE_FIELDS = [
    ("study_site_id", pa.string()),
    ("group_kind", pa.string()),
    ("group_value", pa.string()),
]
_RESIDENCE_TAIL = [
    ("cell_ix", pa.int32()),
    ("cell_iy", pa.int32()),
    ("lon_center_deg", pa.float64()),
    ("lat_center_deg", pa.float64()),
    ("cell_area_km2", pa.float64()),
    ("particle_hours", pa.int64()),
    ("fraction_of_total", pa.float64()),
    ("density_fraction_per_km2", pa.float64()),
]


def _table_from_rows(rows: list[dict[str, Any]], fields: list[tuple[str, pa.DataType]]) -> pa.Table:
    schema = pa.schema([pa.field(name, dtype) for name, dtype in fields])
    return pa.Table.from_pylist(rows, schema=schema)


def _residence_table(rows: list[dict[str, Any]], *, velocity: bool) -> pa.Table:
    fields = list(_RESIDENCE_FIELDS)
    if velocity:
        fields.append(("settling_velocity_mps", pa.float64()))
    return _table_from_rows(rows, fields + _RESIDENCE_TAIL)


def _residence_half_table(rows: list[dict[str, Any]]) -> pa.Table:
    return _table_from_rows(rows, [*_RESIDENCE_FIELDS, ("member_ids", pa.string()), *_RESIDENCE_TAIL])


def _hdr_table(rows: list[dict[str, Any]]) -> pa.Table:
    return _table_from_rows(
        rows,
        [
            *_RESIDENCE_FIELDS,
            ("level", pa.float64()),
            ("density_threshold_fraction_per_km2", pa.float64()),
            ("area_km2", pa.float64()),
            ("cell_count", pa.int64()),
            ("achieved_fraction", pa.float64()),
        ],
    )


def _snapshot_table(rows: list[dict[str, Any]]) -> pa.Table:
    return _table_from_rows(
        rows,
        [
            ("study_site_id", pa.string()),
            ("snapshot_hours", pa.int64()),
            ("snapshot_days", pa.float64()),
            ("site_particle_count", pa.int64()),
            ("active_particle_count", pa.int64()),
            ("active_fraction", pa.float64()),
            ("status", pa.string()),
            ("minimum_active_particle_count", pa.int64()),
            ("smoothing_sigma_km", pa.float64()),
            ("mass_rescale_factor", pa.float64()),
            ("hdr50_area_km2", pa.float64()),
            ("hdr75_area_km2", pa.float64()),
            ("hdr90_area_km2", pa.float64()),
            ("particles_outside_grid", pa.int64()),
        ],
    )


def _snapshot_density_table(rows: list[dict[str, Any]]) -> pa.Table:
    return _table_from_rows(
        rows,
        [
            ("study_site_id", pa.string()),
            ("snapshot_hours", pa.int64()),
            ("cell_ix", pa.int32()),
            ("cell_iy", pa.int32()),
            ("lon_center_deg", pa.float64()),
            ("lat_center_deg", pa.float64()),
            ("cell_area_km2", pa.float64()),
            ("particle_count", pa.int64()),
            ("fraction_of_active", pa.float64()),
            ("density_fraction_per_km2", pa.float64()),
        ],
    )


def _grid_corners_table(rows: list[dict[str, Any]]) -> pa.Table:
    return _table_from_rows(
        rows,
        [
            ("study_site_id", pa.string()),
            ("corner_ix", pa.int32()),
            ("corner_iy", pa.int32()),
            ("lon_deg", pa.float64()),
            ("lat_deg", pa.float64()),
        ],
    )


def _sea_mask_table(rows: list[dict[str, Any]]) -> pa.Table:
    return _table_from_rows(
        rows,
        [
            ("study_site_id", pa.string()),
            ("cell_ix", pa.int32()),
            ("cell_iy", pa.int32()),
            ("lon_center_deg", pa.float64()),
            ("lat_center_deg", pa.float64()),
            ("sea_area_fraction", pa.float64()),
            ("is_sea_cell", pa.bool_()),
            ("observed_particle", pa.bool_()),
        ],
    )


def _arc_table(rows: list[dict[str, Any]]) -> pa.Table:
    return _table_from_rows(
        rows,
        [
            ("study_site_id", pa.string()),
            ("bin_index", pa.int32()),
            ("s_start_km", pa.float64()),
            ("s_end_km", pa.float64()),
            ("crossed_particle_count", pa.int64()),
            ("fraction", pa.float64()),
            ("cumulative_fraction", pa.float64()),
            ("boundary_length_km", pa.float64()),
            ("bin_km", pa.float64()),
        ],
    )


def _edge_table(rows: list[dict[str, Any]]) -> pa.Table:
    return _table_from_rows(
        rows,
        [
            ("study_site_id", pa.string()),
            ("kind", pa.string()),
            ("name", pa.string()),
            ("s_start_km", pa.float64()),
            ("s_end_km", pa.float64()),
        ],
    )


def _geometry_table(rows: list[dict[str, Any]]) -> pa.Table:
    return _table_from_rows(
        rows,
        [
            ("study_site_id", pa.string()),
            ("element", pa.string()),
            ("part_index", pa.int32()),
            ("vertex_index", pa.int32()),
            ("lon_deg", pa.float64()),
            ("lat_deg", pa.float64()),
            ("s_km", pa.float64()),
            ("label", pa.string()),
        ],
    )


def summary_table(rows: list[dict[str, Any]]) -> pa.Table:
    """把逐站摘要列轉成 ``pyarrow.Table``（缺值為 null）。"""

    return pa.Table.from_pylist(rows)
