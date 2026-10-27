# ruff: noqa: E501
"""WP-75 試作：圓形來源方位圖的「幾何期望」與新主要來源方向規則。

為什麼需要期望值
----------------
圓形來源方位圖把來源點依相對錨點的方位分到 36 個各 10° 的扇區。若來源點沿流場範圍邊界（穿越者）與
範圍內海岸（抵達海岸者）「每公里一樣多」地均勻分布，各扇區仍會因幾何而得到不同的比例：
朝向矩形角落的 10° 扇區涵蓋的邊界長度，遠大於正對近邊的 10° 扇區；朝向陸地的扇區則沒有邊界也沒有海岸可到達。
舊規則（各扇區比例高於 1/36 者相鄰合併）沒有扣掉這個幾何差異，也把朝陸的扇區算進平均。

期望值（每站只由幾何決定）
--------------------------
``e_i = P_cross × (扇區 i 內的海上邊界長度 ÷ 海上邊界總長) + P_coast × (扇區 i 內的範圍內海岸長度 ÷ 海岸總長)``

* ``P_cross``、``P_coast``：該組虛擬質點的穿越比例與抵達海岸比例（觀測總量，%）。
* 海上邊界：流場範圍矩形邊界扣掉落在陸地上的部分（陸上邊界不可能被穿越，貢獻 0）。
* 範圍內海岸：與邊界分段機率圖相同的底圖海岸線（本島與各島），只取流場範圍內。
* 扇區歸屬：沿邊界與海岸以不超過 100 m 的步長走過，取每一小步中點相對錨點的大圓初始方位角
  （與舊版 :func:`polar_source.bearing_deg` 相同），依 :func:`polar_source.sector_index` 歸扇區，累加長度。
* 不做視線遮蔽（不扣掉被陸地擋住的方向）。

新主要來源方向規則
------------------
觀測比例大於期望比例（``obs_i > e_i``）的連續扇區（首尾相接）合併為一個範圍，範圍的觀測比例為各扇區觀測比例合計；
取合計最大者為最主要來源方向，並列出第二大者。舊規則為 ``obs_i > 100/36 %``。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import shapely

from . import polar_source as ps

N = ps.N_SECTORS
STEP_M = 100.0
OLD_THRESHOLD_PCT = 100.0 / N


def walk_length_by_sector(
    geom, proj, anchor_lonlat: tuple[float, float], step_m: float = STEP_M
) -> np.ndarray:
    """沿（投影公尺的）線幾何以不超過 ``step_m`` 的步長走過，回傳 36 個扇區各自的長度（公尺）。"""

    out = np.zeros(N)
    if geom is None or geom.is_empty:
        return out
    parts = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
    for part in parts:
        if part.geom_type != "LineString" or part.length <= 0:
            continue
        length = float(part.length)
        n = max(1, math.ceil(length / step_m))
        frac = (np.arange(n) + 0.5) / n
        pts = shapely.line_interpolate_point(part, frac * length)
        x, y = shapely.get_x(pts), shapely.get_y(pts)
        lon, lat = proj.unproject(np.asarray(x), np.asarray(y))
        az = ps.bearing_deg(anchor_lonlat[0], anchor_lonlat[1], lon, lat)
        idx = ps.sector_index(az)
        out += np.bincount(idx, minlength=N) * (length / n)
    return out


@dataclass(frozen=True)
class SiteGeometry:
    """單站幾何：各扇區的海上邊界長度與範圍內海岸長度（公尺）。"""

    boundary_sea_m: np.ndarray
    coast_m: np.ndarray
    boundary_total_m: float
    boundary_sea_total_m: float
    coast_total_m: float

    @property
    def frac_boundary(self) -> np.ndarray:
        return (
            self.boundary_sea_m / self.boundary_sea_total_m if self.boundary_sea_total_m > 0 else np.zeros(N)
        )

    @property
    def frac_coast(self) -> np.ndarray:
        return self.coast_m / self.coast_total_m if self.coast_total_m > 0 else np.zeros(N)


def site_geometry(sd, step_m: float = STEP_M) -> SiteGeometry:
    """由分段機率圖的站資料（:class:`wp75_common.SiteData`）算各扇區的海上邊界與海岸長度。"""

    ring = sd.ring.line
    land = sd.land_proj
    sea = ring if land is None or land.is_empty else ring.difference(land)
    b = walk_length_by_sector(sea, sd.proj, sd.anchor, step_m)
    c = np.zeros(N)
    c_total = 0.0
    for piece in sd.pieces:
        c += walk_length_by_sector(piece.line, sd.proj, sd.anchor, step_m)
        c_total += float(piece.line.length)
    return SiteGeometry(b, c, float(ring.length), float(sea.length), c_total)


def expected_pct(geo: SiteGeometry, p_cross_pct: float, p_coast_pct: float) -> np.ndarray:
    """36 扇區的期望比例（%）；陸上邊界與沒有海岸的扇區為 0。"""

    return p_cross_pct * geo.frac_boundary + p_coast_pct * geo.frac_coast


def circular_runs(flags: np.ndarray) -> list[list[int]]:
    """布林向量中連續為真的扇區編號串（首尾相接），每串由小到大依環狀順序列出。"""

    f = np.asarray(flags, dtype=bool)
    if f.all():
        return [list(range(N))]
    if not f.any():
        return []
    start = int(np.flatnonzero(~f)[0])  # 從一個假值之後開始掃，環狀不會切斷連續段
    order = [(start + 1 + k) % N for k in range(N)]
    runs: list[list[int]] = []
    cur: list[int] = []
    for i in order:
        if f[i]:
            cur.append(i)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return runs


def run_label(run: list[int]) -> tuple[str, str]:
    """(扇區編號範圍, 方位範圍)。方位範圍取扇區邊界（中心 ±5°），例：扇區 12–18 → ``12–18``、``115°–185°``。"""

    a, b = run[0], run[-1]
    sec = f"{a}" if a == b else f"{a}–{b}"
    lo = (10 * a - 5) % 360
    hi = (10 * b + 5) % 360
    return sec, f"{lo}°–{hi}°"


def ranked_runs(flags: np.ndarray, obs_pct: np.ndarray) -> list[list[int]]:
    """連續範圍依觀測比例合計由大到小排序（同值取起點編號小者）。"""

    runs = circular_runs(flags)
    return sorted(runs, key=lambda r: (-float(obs_pct[r].sum()), r[0]))


def old_flags(counts: np.ndarray, n: int) -> np.ndarray:
    """舊規則：扇區比例高於 1/36（以整數顆數比較，避免浮點誤差）。"""

    return counts.astype(np.int64) * N > int(n)


def new_flags(obs_pct: np.ndarray, exp_pct: np.ndarray) -> np.ndarray:
    """新規則：觀測比例大於期望比例。"""

    return np.asarray(obs_pct) > np.asarray(exp_pct)
