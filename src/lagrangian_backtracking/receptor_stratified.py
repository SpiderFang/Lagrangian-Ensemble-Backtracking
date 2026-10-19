"""WP-63：受體水平位置的「方位 × 距離」等面積拉丁方格分層隨機抽樣。

目的：在以錨點為中心、半徑 ``radius_m`` 內的候選面（persistent-wet、全部到達時刻皆濕潤、通過邊界緩衝）
中，抽出 ``n``（預設 5）個水平位置，使它們環繞錨點、由內圈到外圈都有，而且每個位置相當於在其所屬
海域塊內以面積加權均勻隨機取點。

方法
----
1. 方位分層：以種子決定的起始方位為起點，依方位順時針累加候選面面積，切成 n 個面積相等的方位區間
   （陸地占去的方向不改變每區海域面積）。
2. 距離分層：依距錨點距離由近到遠累加候選面面積，切成 n 個面積相等的距離區間。
3. 配對（拉丁方格）：以種子產生 0..n-1 的排列 pi，方位區間 i 配距離區間 pi[i]；每個方位區間、每個
   距離區間恰好一個位置（必有最內圈與最外圈各一個）。
4. 在「方位區間 ∩ 距離區間」的區塊內以面積加權隨機抽一個候選面：對全體候選面一次產生面積加權的隨機
   順序鍵（Efraimidis–Spirakis：key = -ln(u) / area，鍵小者先），區塊內依鍵由小到大取第一個通過
   ``is_supported`` 的面；沒通過者即「依隨機順序遞補」。鍵與排列嘗試無關，同一種子下的結果不受其他區塊
   或曾嘗試過的排列影響。
5. 某排列使某區塊沒有任何（通過支援檢查的）候選面時，改試種子順序中的下一個排列；n! 個排列都不行時
   丟出 ``ValueError``（fail closed，不放寬、不補點）。

面歸屬：每個面以其「累加面積中點」決定歸屬區間；方位與距離兩個方向各自獨立計算。方位與距離一律用
WGS84 測地線（錨點到面中心的經緯度）計算。隨機性只來自 ``seed``，由呼叫端以 master_seed、站點、設計
版本、policy 與 stream scope 經 ``derive_receptor_selection_seed`` 派生。

本模組的核心（``sample_azimuth_distance_latin``）只用 numpy，來源是另一個工作寫的預覽函式
``lbt-wp62/receptor_sampling_preview.py``；差異只有：不匯入預覽專用的 ``Candidates`` 以外的東西、
增加 ``n`` 的上限檢查，並以 ``select_horizontal_receptors_azimuth_distance_latin`` 接上候選池。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import permutations
from typing import Any, Final

import numpy as np

HORIZONTAL_RECEPTOR_SELECTION_POLICY_AZIMUTH_DISTANCE_LATIN_ID: Final[str] = (
    "azimuth_distance_equal_area_latin_stratified_v1"
)
AZIMUTH_DISTANCE_LATIN_STREAM_SCOPE: Final[str] = "horizontal:azimuth_distance_latin"
AZIMUTH_DISTANCE_LATIN_SELECTION_SCOPE: Final[str] = "azimuth_distance_latin"
MAX_STRATA_COUNT: Final[int] = 8


@dataclass(frozen=True)
class Candidates:
    """候選面（全部陣列等長）。azimuth_deg 為自北順時針 [0,360)；distance_m 為距錨點距離。"""

    face_id: np.ndarray
    azimuth_deg: np.ndarray
    distance_m: np.ndarray
    area_m2: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.face_id)
        if not (len(self.azimuth_deg) == len(self.distance_m) == len(self.area_m2) == n):
            raise ValueError("候選面陣列長度不一致")
        if n == 0:
            raise ValueError("沒有候選面")
        if not np.all(np.asarray(self.area_m2) > 0):
            raise ValueError("候選面面積必須為正")
        if len(set(np.asarray(self.face_id).tolist())) != n:
            raise ValueError("face_id 不可重複")


@dataclass(frozen=True)
class SamplingResult:
    start_azimuth_deg: float
    azimuth_edges_deg: tuple[float, ...]  # 長度 n+1，絕對方位（自北順時針），第 0 與第 n 個相同（= 起始方位）
    distance_edges_m: tuple[float, ...]  # 長度 n+1
    permutation: tuple[int, ...]  # 方位區間 i（1 起算）-> 距離區間 permutation[i]（1 起算）
    attempts: int  # 實際試了幾個排列（含成功的那個）
    attempt_log: tuple[str, ...]  # 失敗排列的原因
    picks: tuple[dict, ...]  # 依方位區間編號排序


def equal_area_bins(
    values: np.ndarray,
    area: np.ndarray,
    face_id: np.ndarray,
    n: int,
    *,
    start: float = 0.0,
    period: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """依 values 排序累加 area，切成 n 個面積相等的區間。

    period 不為 None 時 values 為循環量（方位，度），累加從 ``start`` 開始。回傳
    ``(bin_index 每個面的區間編號, edges 長度 n+1 的邊界)``；邊界取相鄰兩區最近兩個面的 values 中點，
    第 0 邊界為 0（相對 start），第 n 邊界為 period（循環）或 values 最大值。
    """

    values = np.asarray(values, dtype=float)
    area = np.asarray(area, dtype=float)
    rel = np.mod(values - start, period) if period is not None else values
    order = np.lexsort((np.asarray(face_id), rel))  # 主鍵 rel，次鍵 face_id（可重現）
    cum = np.cumsum(area[order])
    total = cum[-1]
    mid = cum - 0.5 * area[order]
    idx_sorted = np.minimum((mid / total * n).astype(int), n - 1)
    bin_index = np.empty(len(values), dtype=int)
    bin_index[order] = idx_sorted
    edges = np.empty(n + 1)
    edges[0] = 0.0
    edges[n] = float(period) if period is not None else float(rel.max())
    for k in range(1, n):
        left = rel[order][idx_sorted == k - 1]
        right = rel[order][idx_sorted == k]
        if left.size == 0 or right.size == 0:
            raise ValueError("候選面太少，無法切成面積相等的區間")
        edges[k] = 0.5 * (left.max() + right.min())
    return bin_index, edges


def _weighted_random_keys(rng: np.random.Generator, area: np.ndarray, face_id: np.ndarray) -> np.ndarray:
    """面積加權隨機順序鍵：鍵小者先被抽中。先依 face_id 排序再取亂數，結果與輸入順序無關。"""

    order = np.argsort(face_id, kind="stable")
    u = rng.random(len(face_id))
    key_sorted = -np.log(u) / area[order]
    keys = np.empty(len(face_id))
    keys[order] = key_sorted
    return keys


def validate_azimuth_sectors(sectors: Sequence[tuple[float, float, int]], n: int) -> tuple[tuple[float, float, int], ...]:
    """方位扇區必須由小到大、首尾相接地蓋滿 [0, 360)，每區至少 1 個方位區間，總數等於 n。"""

    checked = tuple((float(lo), float(hi), int(k)) for lo, hi, k in sectors)
    if (
        not checked
        or checked[0][0] != 0.0
        or checked[-1][1] != 360.0
        or any(hi <= lo or k < 1 for lo, hi, k in checked)
        or any(checked[i][1] != checked[i + 1][0] for i in range(len(checked) - 1))
        or sum(k for _lo, _hi, k in checked) != n
    ):
        raise ValueError(f"方位扇區必須連續蓋滿 [0,360) 且區間數總和為 {n}：{checked}")
    return checked


def sample_azimuth_distance_latin(
    candidates: Candidates,
    *,
    seed: int,
    n: int = 5,
    is_supported: Callable[[int], bool] | None = None,
    azimuth_sectors: Sequence[tuple[float, float, int]] | None = None,
) -> SamplingResult:
    """見模組說明。``is_supported(face_id) -> bool`` 為逐面空間支援檢查（惰性呼叫並快取）。

    ``azimuth_sectors``（預設 ``None``＝隨機起始方位的 n 等分）是 ``[(起始方位, 結束方位, 區間數), ...]``：
    候選面先依方位分進各扇區，各扇區內再依方位切成指定個數的等面積區間；方位區間編號由方位小到大，
    起始方位固定為 0（不隨機）。距離分層與拉丁方格配對不變。
    """

    if n < 2 or n > MAX_STRATA_COUNT:
        raise ValueError(f"n 必須在 2..{MAX_STRATA_COUNT}")
    ids = np.asarray(candidates.face_id)
    az = np.asarray(candidates.azimuth_deg, dtype=float)
    dist = np.asarray(candidates.distance_m, dtype=float)
    area = np.asarray(candidates.area_m2, dtype=float)

    def stream(tag: int) -> np.random.Generator:
        return np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence([int(seed), tag])))

    if azimuth_sectors is None:
        start = float(stream(0).uniform(0.0, 360.0))
        az_bin, az_rel_edges = equal_area_bins(az, area, ids, n, start=start, period=360.0)
        az_abs_edges = tuple(float((start + e) % 360.0) for e in az_rel_edges)
    else:
        sectors = validate_azimuth_sectors(azimuth_sectors, n)
        start = 0.0
        az_bin = np.full(len(ids), -1, dtype=int)
        edge_list = [0.0]
        offset = 0
        for lo, hi, k in sectors:
            mask = (az >= lo) & (az < hi)
            if int(mask.sum()) < k:
                raise ValueError(f"方位扇區 [{lo:g},{hi:g}) 的候選面只有 {int(mask.sum())} 個，少於 {k} 個區間")
            sector_bin, sector_edges = equal_area_bins(az[mask] - lo, area[mask], ids[mask], k)
            az_bin[mask] = sector_bin + offset
            edge_list.extend(float(lo + sector_edges[j]) for j in range(1, k))
            edge_list.append(float(hi))
            offset += k
        az_abs_edges = tuple(edge_list)
    d_bin, d_edges = equal_area_bins(dist, area, ids, n)
    d_edges[0] = 0.0
    keys = _weighted_random_keys(stream(2), area, ids)

    perms = list(permutations(range(n)))
    perm_order = stream(1).permutation(len(perms))

    support_cache: dict[int, bool] = {}

    def supported(face_id: int) -> bool:
        if is_supported is None:
            return True
        if face_id not in support_cache:
            support_cache[face_id] = bool(is_supported(face_id))
        return support_cache[face_id]

    log: list[str] = []
    for attempt, pi in enumerate(perm_order, start=1):
        perm = perms[int(pi)]
        picks: list[dict] = []
        failed = None
        for i in range(n):
            in_cell = np.flatnonzero((az_bin == i) & (d_bin == perm[i]))
            if in_cell.size == 0:
                failed = f"方位區間 {i + 1} ∩ 距離區間 {perm[i] + 1} 沒有候選面"
                break
            skipped = 0
            chosen = None
            for j in in_cell[np.argsort(keys[in_cell], kind="stable")]:
                if supported(int(ids[j])):
                    chosen = int(j)
                    break
                skipped += 1
            if chosen is None:
                failed = f"方位區間 {i + 1} ∩ 距離區間 {perm[i] + 1} 的 {in_cell.size} 個候選面皆未通過支援檢查"
                break
            picks.append(
                {
                    "face_id": int(ids[chosen]),
                    "azimuth_bin": i + 1,
                    "distance_bin": perm[i] + 1,
                    "azimuth_deg": float(az[chosen]),
                    "distance_m": float(dist[chosen]),
                    "cell_candidate_count": int(in_cell.size),
                    "skipped_unsupported": skipped,
                }
            )
        if failed is None:
            return SamplingResult(
                start_azimuth_deg=start,
                azimuth_edges_deg=az_abs_edges,
                distance_edges_m=tuple(float(e) for e in d_edges),
                permutation=tuple(int(p) + 1 for p in perm),
                attempts=attempt,
                attempt_log=tuple(log),
                picks=tuple(picks),
            )
        log.append(f"排列 {tuple(p + 1 for p in perm)}：{failed}")
    counts = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            counts[i, j] = int(np.sum((az_bin == i) & (d_bin == j)))
    err = ValueError(
        f"所有 {len(perms)} 個排列都有空區塊（或區塊內無通過支援檢查的面），無法完成抽樣；"
        f"各區塊候選面數（列=方位區間，欄=距離區間）：{counts.tolist()}"
    )
    err.cell_counts = counts  # type: ignore[attr-defined]
    err.attempt_log = tuple(log)  # type: ignore[attr-defined]
    raise err


def geodesic_azimuth_distance(
    anchor_lonlat: tuple[float, float], lon: np.ndarray, lat: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """錨點到各點的 WGS84 測地線方位（自北順時針 [0,360)）與距離（m）。"""

    from pyproj import Geod

    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    az, _back, dist = Geod(ellps="WGS84").inv(
        np.full(lon.shape, float(anchor_lonlat[0])),
        np.full(lat.shape, float(anchor_lonlat[1])),
        lon,
        lat,
    )
    return np.mod(np.asarray(az, dtype=float), 360.0), np.asarray(dist, dtype=float)


def select_horizontal_receptors_azimuth_distance_latin(
    pool: Any,
    *,
    anchor_lonlat: tuple[float, float],
    master_seed: int,
    design_version: str,
    radius_m: float,
    count: int = 5,
    is_supported: Callable[[Any], bool] | None = None,
    selection_policy_id: str = HORIZONTAL_RECEPTOR_SELECTION_POLICY_AZIMUTH_DISTANCE_LATIN_ID,
    stream_scope: str = AZIMUTH_DISTANCE_LATIN_STREAM_SCOPE,
    excluded_face_indices: frozenset[int] | set[int] = frozenset(),
    azimuth_sectors: Sequence[tuple[float, float, int]] | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    """在候選池中抽 ``count`` 個水平受體；回傳 ``(依方位區間排序的受體, 可寫進 manifest 的稽核)``。

    ``pool`` 是 ``receptors.HorizontalReceptorCandidatePool``（需帶 ``candidate_area_m2``）；
    ``is_supported(receptor)`` 是呼叫端的逐面閘門（NWW exact-hour 與 OCM 垂向支援），丟出
    ``InputDerivationError`` 或回傳 False 都算未通過（呼叫端自行把例外轉成 False）。
    """

    from .receptors import (  # 延遲匯入，避免 receptors 與本模組循環
        derive_receptor_selection_seed,
        horizontal_receptor_from_pool_position,
    )

    if pool.candidate_area_m2 is None:
        raise ValueError("候選池缺少 candidate_area_m2，無法做等面積分層")
    radius = float(radius_m)
    if not (math.isfinite(radius) and radius > 0.0):
        raise ValueError("radius_m 必須是有限正數")
    lon = np.asarray(pool.candidate_lon, dtype=float)
    lat = np.asarray(pool.candidate_lat, dtype=float)
    az, dist = geodesic_azimuth_distance(anchor_lonlat, lon, lat)
    local = np.asarray(pool.candidate_face_local_indices)
    keep = (dist <= radius) & np.asarray([int(v) not in excluded_face_indices for v in local])
    positions = np.flatnonzero(keep)
    if positions.size < count:
        raise ValueError(f"{pool.study_site_id} 半徑 {radius:g} m 內候選面只有 {positions.size} 個，少於 {count}")
    face_ids = np.asarray(pool.candidate_face_global_indices)[positions]
    candidates = Candidates(
        face_id=face_ids,
        azimuth_deg=az[positions],
        distance_m=dist[positions],
        area_m2=np.asarray(pool.candidate_area_m2, dtype=float)[positions],
    )
    position_by_face = {int(f): int(p) for f, p in zip(face_ids, positions, strict=True)}
    seed, seed_digest = derive_receptor_selection_seed(
        master_seed=int(master_seed),
        study_site_id=pool.study_site_id.split(":", 1)[0],
        design_version=design_version,
        selection_policy_id=selection_policy_id,
        stream_scope=stream_scope,
    )

    def gate(face_id: int) -> bool:
        if is_supported is None:
            return True
        return bool(is_supported(horizontal_receptor_from_pool_position(pool, position_by_face[int(face_id)])))

    result = sample_azimuth_distance_latin(
        candidates, seed=seed, n=count, is_supported=gate, azimuth_sectors=azimuth_sectors
    )
    receptors = [horizontal_receptor_from_pool_position(pool, position_by_face[int(p["face_id"])]) for p in result.picks]
    record = {
        "selection_policy": selection_policy_id,
        "stream_scope": stream_scope,
        "derived_seed_hex": f"{seed:032x}",
        "seed_derivation_sha256": seed_digest,
        "radius_m": radius,
        "strata_count": int(count),
        "candidate_face_count_within_radius": int(positions.size),
        "candidate_area_m2_within_radius": float(np.sum(candidates.area_m2)),
        "start_azimuth_deg": result.start_azimuth_deg,
        "azimuth_edges_deg": list(result.azimuth_edges_deg),
        "distance_edges_m": list(result.distance_edges_m),
        "permutation_azimuth_bin_to_distance_bin": list(result.permutation),
        "permutation_attempts": int(result.attempts),
        "permutation_attempt_log": list(result.attempt_log),
        "picks": [dict(p) for p in result.picks],
        "tie_break": "weighted_random_order_key_efraimidis_spirakis_then_next_in_same_cell",
    }
    if azimuth_sectors is not None:
        record["azimuth_sectors"] = [
            {"from_deg": float(lo), "to_deg": float(hi), "bin_count": int(k)}
            for lo, hi, k in validate_azimuth_sectors(azimuth_sectors, int(count))
        ]
    return receptors, record
