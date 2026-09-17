"""persistent-wet 水平受體與四個垂向層位的決定性生成。

每站先在固定 candidate polygon 內選 5 個 source-face 中心，再建立 upper 10%、40%、
70% 與 near-bed 四個垂向模板，形成 20 個 receptor IDs。水平候選必須在所有 50 個
arrival times 都是 wet，且與 candidate boundary 保留核定 margin；maximin 只在已通過
這些資料與海岸 gate 的候選上運作。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from shapely.geometry import Point, Polygon

from .geometry import deterministic_maximin
from .mesh import NativeMesh
from .scenarios import stable_identifier


@dataclass(frozen=True, slots=True)
class HorizontalReceptor:
    """一個站點的水平 receptor 位置與 OCM face provenance。"""

    horizontal_receptor_id: str
    study_site_id: str
    x_m: float
    y_m: float
    lon: float
    lat: float
    source_face_local_index: int
    source_face_global_index: int
    anchor_snap_distance_m: float | None


@dataclass(frozen=True, slots=True)
class HorizontalReceptorCandidatePool:
    """單站一次建立、可供多輪 deterministic 重選的水平候選池。

    candidate pool 已完成 persistent-wet、candidate polygon 與 boundary margin gate；
    陣列分別保存 source face local/global index、公尺制中心、經緯度與 tie-break key。
    所有 numpy 欄位都是 defensive copy 且設為唯讀，後續 NWW／OCM 垂向 blacklist 只能透過
    ``excluded_face_indices`` 傳入 selector，不能改寫原始候選池或重新掃描 Shapely
    geometry。經緯度只作 receptor 資料交換與 NWW 空間 gate，maximin 距離仍在公尺制
    ``candidate_xy_m`` 與 ``anchor_xy`` 上計算。
    """

    study_site_id: str
    anchor_xy: tuple[float, float]
    maximum_anchor_snap_distance_m: float | None
    candidate_face_local_indices: np.ndarray
    candidate_face_global_indices: np.ndarray
    candidate_xy_m: np.ndarray
    candidate_lon: np.ndarray
    candidate_lat: np.ndarray
    tie_break_keys: tuple[tuple[float, float, int], ...]


@dataclass(frozen=True, slots=True)
class VerticalTarget:
    """單一 arrival/水平位置的實際 positive-up z 與 HAB。"""

    vertical_id: str
    target_fraction_below_surface: float | None
    z_m_positive_up: float
    height_above_bed_m: float
    bracket_span_m: float


def _face_centers(mesh: NativeMesh) -> np.ndarray:
    """依每個原生 face 的 3/4 個有效 node 計算公尺中心。"""

    centers = np.empty((mesh.face_nodes_local.shape[0], 2), dtype=np.float64)
    for face_index, count in enumerate(mesh.face_node_count):
        nodes = mesh.face_nodes_local[face_index, : int(count)]
        centers[face_index] = mesh.node_xy[nodes].mean(axis=0)
    return centers


def _readonly_copy(values: np.ndarray) -> np.ndarray:
    """建立候選池專用的唯讀 array copy，避免 excluded 狀態污染 immutable pool。"""

    result = np.array(values, copy=True)
    result.setflags(write=False)
    return result


def prepare_horizontal_receptor_candidates(
    *,
    study_site_id: str,
    mesh: NativeMesh,
    candidate_polygon_metric: Polygon,
    anchor_xy: tuple[float, float],
    wetdry_at_arrivals: np.ndarray,
    boundary_margin_m: float = 0.0,
    wet_value: float = 0.0,
    maximum_anchor_snap_distance_m: float | None = None,
) -> HorizontalReceptorCandidatePool:
    """一次掃描 geometry，建立單站 persistent-wet 的 immutable 水平候選池。

    ``wetdry_at_arrivals`` shape 為 ``(arrival,source_face)``；50 個時次任一非有限或非 wet
    即剔除。``boundary_margin_m`` 同時避開 candidate polygon 外界與被 ocean clipping
    形成的岸線。這個階段不接受 count，也不進行 maximin；候選數量與 NWW／垂向
    blacklist 由 ``select_horizontal_receptors_from_pool`` 在不重做 geometry scan 的
    情況下處理。pool 保存的座標與 tie-break key 已經是原 wrapper 的同一份資料，故
    不會因重選輪次或 excluded set 的輸入順序改變結果。
    """

    wetdry = np.asarray(wetdry_at_arrivals, dtype=np.float64)
    if wetdry.ndim != 2 or wetdry.shape[1] != mesh.face_nodes_local.shape[0]:
        raise ValueError("wetdry_at_arrivals 必須是 (arrival,source_face)")
    if boundary_margin_m < 0:
        raise ValueError("boundary margin 不可為負")
    persistent_wet = np.all(np.isfinite(wetdry) & np.isclose(wetdry, wet_value, atol=0.1), axis=0)
    centers = _face_centers(mesh)
    candidate_indices: list[int] = []
    for face_index, (x_m, y_m) in enumerate(centers):
        point = Point(float(x_m), float(y_m))
        if not persistent_wet[face_index] or not candidate_polygon_metric.covers(point):
            continue
        if boundary_margin_m > 0 and point.distance(candidate_polygon_metric.boundary) < boundary_margin_m:
            continue
        candidate_indices.append(face_index)
    indices = np.asarray(candidate_indices, dtype=np.int64)
    candidate_xy = centers[indices]
    lon = np.empty(indices.size, dtype=np.float64)
    lat = np.empty(indices.size, dtype=np.float64)
    for position, face_index in enumerate(indices):
        node_count = int(mesh.face_node_count[face_index])
        nodes = mesh.face_nodes_local[face_index, :node_count]
        lon[position] = float(mesh.node_lon[nodes].mean())
        lat[position] = float(mesh.node_lat[nodes].mean())
    tie_keys = [
        (float(lon[index]), float(lat[index]), int(mesh.source_face_global_index[face_index]))
        for index, face_index in enumerate(indices)
    ]
    return HorizontalReceptorCandidatePool(
        study_site_id=study_site_id,
        anchor_xy=(float(anchor_xy[0]), float(anchor_xy[1])),
        maximum_anchor_snap_distance_m=(
            float(maximum_anchor_snap_distance_m)
            if maximum_anchor_snap_distance_m is not None
            else None
        ),
        candidate_face_local_indices=_readonly_copy(indices),
        candidate_face_global_indices=_readonly_copy(mesh.source_face_global_index[indices]),
        candidate_xy_m=_readonly_copy(candidate_xy),
        candidate_lon=_readonly_copy(lon),
        candidate_lat=_readonly_copy(lat),
        tie_break_keys=tuple(tie_keys),
    )


def select_horizontal_receptors_from_pool(
    pool: HorizontalReceptorCandidatePool,
    *,
    count: int = 5,
    excluded_face_indices: Iterable[int] = (),
) -> list[HorizontalReceptor]:
    """在既有 immutable candidate pool 上重跑 anchor-first deterministic maximin。

    ``excluded_face_indices`` 是目前 study site 已被 NWW 或 OCM 垂向 gate 淘汰的 source
    face local index 集合；selector 只建立一份布林選取 mask，不修改 pool 的任何 array。
    因此每輪只做數值 maximin，不重新建立 Shapely Point、重新判斷 polygon 或讀取 wetdry。
    候選不足仍以 ``ValueError`` fail closed；未被排除的候選、tie-break、anchor snap 與
    receptor ID 格式均保持原 ``select_horizontal_receptors`` 的結果。
    """

    if not isinstance(pool, HorizontalReceptorCandidatePool):
        raise ValueError("pool 必須是 HorizontalReceptorCandidatePool")
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)):
        raise ValueError("count 必須為正整數")
    count_value = int(count)
    if count_value < 1:
        raise ValueError("count 必須為正整數")
    try:
        excluded_values = tuple(excluded_face_indices)
    except TypeError as exc:
        raise ValueError("excluded_face_indices 必須是整數 iterable") from exc
    if any(
        isinstance(face_index, (bool, np.bool_))
        or not isinstance(face_index, (int, np.integer))
        for face_index in excluded_values
    ):
        raise ValueError("excluded_face_indices 必須是整數 iterable")
    excluded = frozenset(int(face_index) for face_index in excluded_values)
    include = np.asarray(
        [int(face_index) not in excluded for face_index in pool.candidate_face_local_indices],
        dtype=bool,
    )
    remaining_count = int(np.count_nonzero(include))
    if remaining_count < count_value:
        raise ValueError(
            f"{pool.study_site_id} persistent-wet/margin 候選不足：需要 {count_value}，實際 {remaining_count}"
        )
    candidate_xy = pool.candidate_xy_m[include]
    candidate_lon = pool.candidate_lon[include]
    candidate_lat = pool.candidate_lat[include]
    candidate_local_indices = pool.candidate_face_local_indices[include]
    candidate_global_indices = pool.candidate_face_global_indices[include]
    candidate_tie_keys = tuple(
        key for key, included in zip(pool.tie_break_keys, include, strict=True) if included
    )
    local_selected = deterministic_maximin(
        candidate_xy,
        count=count_value,
        anchor_xy=pool.anchor_xy,
        tie_break_keys=candidate_tie_keys,
    )
    selected: list[HorizontalReceptor] = []
    anchor = np.asarray(pool.anchor_xy, dtype=np.float64)
    for order, local_index in enumerate(local_selected):
        index = int(local_index)
        x_m, y_m = candidate_xy[index]
        snap_distance = float(np.linalg.norm(candidate_xy[index] - anchor)) if order == 0 else None
        if (
            order == 0
            and pool.maximum_anchor_snap_distance_m is not None
            and snap_distance > pool.maximum_anchor_snap_distance_m
        ):
            raise ValueError(
                f"{pool.study_site_id} anchor snap {snap_distance:.3f} m "
                f"超過 {pool.maximum_anchor_snap_distance_m:.3f} m"
            )
        selected.append(
            HorizontalReceptor(
                horizontal_receptor_id=stable_identifier(
                    "hr",
                    [
                        pool.study_site_id,
                        str(int(candidate_global_indices[index])),
                        f"{x_m:.6f}",
                        f"{y_m:.6f}",
                    ],
                ),
                study_site_id=pool.study_site_id,
                x_m=float(x_m),
                y_m=float(y_m),
                lon=float(candidate_lon[index]),
                lat=float(candidate_lat[index]),
                source_face_local_index=int(candidate_local_indices[index]),
                source_face_global_index=int(candidate_global_indices[index]),
                anchor_snap_distance_m=snap_distance,
            )
        )
    return selected


def select_horizontal_receptors_from_coordinates(
    pool: HorizontalReceptorCandidatePool,
    *,
    coordinates_lonlat: Sequence[tuple[float, float]],
    coordinates_xy: Sequence[tuple[float, float]],
    tolerance_m: float = 1.0,
) -> list[HorizontalReceptor]:
    """依已核定的 WGS84 順序逐點映射 persistent-wet 原生 OCM face。

    ``coordinates_lonlat`` 是研究者已核定的資料交換座標，``coordinates_xy`` 是同一批
    座標依 flow-domain 的固定公尺制投影所得的位置；兩者只用來尋找 face，不會取代
    OCM 的垂向、NWW 或 arrival 支援檢查。``pool`` 已先完成 candidate polygon、邊界
    margin 與所有 arrival 的 persistent-wet 篩選，因此每個宣告點都必須在該 immutable
    pool 中找到唯一 source face。此函式保留宣告順序，拒絕重複 face、超過公尺制公差
    或不在候選池的座標；它不會退回 maximin、最近值或其他未核定位置。

    Args:
        pool: 同一份 OCM 原生 mesh 建立的 persistent-wet candidate pool。
        coordinates_lonlat: 五個固定水平受體的 WGS84 經度／緯度，順序不可改變。
        coordinates_xy: 上述座標在相同 flow-domain projection 下的公尺制位置。
        tolerance_m: 宣告座標與 source-face 中心允許的公尺制差異；必須為有限正數。

    Returns:
        依宣告順序排列的五個 ``HorizontalReceptor``，其 lon/lat 與 source face
        provenance 取自原生 mesh，而非由輸入座標臨時插值。

    Raises:
        ValueError: 固定點數量、投影座標、匹配公差或唯一 face 契約不成立時。
    """

    if not isinstance(tolerance_m, (int, float)) or isinstance(tolerance_m, bool):
        raise ValueError("固定水平受體 tolerance_m 必須是數值")
    tolerance = float(tolerance_m)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("固定水平受體 tolerance_m 必須是有限正數")
    if len(coordinates_lonlat) != len(coordinates_xy) or len(coordinates_lonlat) != 5:
        raise ValueError("固定水平受體必須恰有順序一致的五個 lon/lat 與公尺制座標")
    declared_lonlat = np.asarray(coordinates_lonlat, dtype=np.float64)
    declared_xy = np.asarray(coordinates_xy, dtype=np.float64)
    if declared_lonlat.shape != (5, 2) or declared_xy.shape != (5, 2):
        raise ValueError("固定水平受體座標必須是 (5, 2) 數值陣列")
    if not np.all(np.isfinite(declared_lonlat)) or not np.all(np.isfinite(declared_xy)):
        raise ValueError("固定水平受體座標不可含非有限值")
    if np.any(declared_lonlat[:, 0] < -180.0) or np.any(declared_lonlat[:, 0] > 180.0):
        raise ValueError("固定水平受體經度超出 WGS84 bounds")
    if np.any(declared_lonlat[:, 1] < -90.0) or np.any(declared_lonlat[:, 1] > 90.0):
        raise ValueError("固定水平受體緯度超出 WGS84 bounds")

    candidate_xy = np.asarray(pool.candidate_xy_m, dtype=np.float64)
    if candidate_xy.ndim != 2 or candidate_xy.shape[1] != 2:
        raise ValueError("candidate pool 公尺制座標形狀不符")
    selected: list[HorizontalReceptor] = []
    used_faces: set[int] = set()
    anchor = np.asarray(pool.anchor_xy, dtype=np.float64)
    for order, (_lonlat, target_xy) in enumerate(zip(declared_lonlat, declared_xy, strict=True)):
        distances = np.linalg.norm(candidate_xy - target_xy, axis=1)
        if distances.size == 0:
            raise ValueError(f"{pool.study_site_id} 固定受體候選池為空")
        candidate_position = int(np.argmin(distances))
        distance_m = float(distances[candidate_position])
        if distance_m > tolerance:
            raise ValueError(
                f"{pool.study_site_id} 固定受體第 {order + 1} 點無法映射到同一 OCM mesh face："
                f"distance={distance_m:.6f} m > tolerance={tolerance:.6f} m"
            )
        local_index = int(pool.candidate_face_local_indices[candidate_position])
        if local_index in used_faces:
            raise ValueError(
                f"{pool.study_site_id} 固定受體第 {order + 1} 點與既有點映射到重複 OCM face："
                f"{local_index}"
            )
        used_faces.add(local_index)
        x_m, y_m = candidate_xy[candidate_position]
        snap_distance = (
            float(np.linalg.norm(candidate_xy[candidate_position] - anchor))
            if order == 0
            else None
        )
        selected.append(
            HorizontalReceptor(
                horizontal_receptor_id=stable_identifier(
                    "hr",
                    [
                        pool.study_site_id,
                        str(int(pool.candidate_face_global_indices[candidate_position])),
                        f"{x_m:.6f}",
                        f"{y_m:.6f}",
                    ],
                ),
                study_site_id=pool.study_site_id,
                x_m=float(x_m),
                y_m=float(y_m),
                lon=float(pool.candidate_lon[candidate_position]),
                lat=float(pool.candidate_lat[candidate_position]),
                source_face_local_index=local_index,
                source_face_global_index=int(pool.candidate_face_global_indices[candidate_position]),
                anchor_snap_distance_m=snap_distance,
            )
        )
    return selected


def select_horizontal_receptors(
    *,
    study_site_id: str,
    mesh: NativeMesh,
    candidate_polygon_metric: Polygon,
    anchor_xy: tuple[float, float],
    wetdry_at_arrivals: np.ndarray,
    count: int = 5,
    boundary_margin_m: float = 0.0,
    wet_value: float = 0.0,
    maximum_anchor_snap_distance_m: float | None = None,
) -> list[HorizontalReceptor]:
    """相容既有 API：prepare 一次候選池後執行無 exclusion 的 deterministic selector。

    wrapper 保持原參數與回傳型別，讓既有 caller 不需知道 pool 內部拆分；候選 pool 的
    geometry、persistent-wet、anchor-first、tie-break、boundary margin 與 snap distance
    語意完全由新的兩階段 API 共用。input derivation 在需要 NWW／垂向 blacklist 時，
    直接呼叫兩階段 API 以避免每輪重建幾何候選。
    """

    pool = prepare_horizontal_receptor_candidates(
        study_site_id=study_site_id,
        mesh=mesh,
        candidate_polygon_metric=candidate_polygon_metric,
        anchor_xy=anchor_xy,
        wetdry_at_arrivals=wetdry_at_arrivals,
        boundary_margin_m=boundary_margin_m,
        wet_value=wet_value,
        maximum_anchor_snap_distance_m=maximum_anchor_snap_distance_m,
    )
    return select_horizontal_receptors_from_pool(pool, count=count)


def build_vertical_targets(
    *,
    surface_z_m: float,
    bed_z_m: float,
    valid_layer_z_m: np.ndarray,
) -> list[VerticalTarget]:
    """建立 10/40/70% depth 與 near-bed 四個可包夾、互異垂向目標。

    前三個目標保持連續物理 z，不硬 snap 到 layer；``bracket_span_m`` 保存實際上下層
    間距。near-bed 使用最低兩個有效 full levels 的中點，代表最低 layer center。若水柱
    太淺或有效層不足以形成四個互異目標，水平位置應由上游淘汰重選。
    """

    layers = np.unique(np.asarray(valid_layer_z_m, dtype=np.float64))
    layers = np.sort(layers[np.isfinite(layers)])
    if not np.isfinite(surface_z_m) or not np.isfinite(bed_z_m) or surface_z_m <= bed_z_m:
        raise ValueError("surface/bed z 必須有限且 surface 高於 bed")
    layers = layers[(layers >= bed_z_m - 1e-9) & (layers <= surface_z_m + 1e-9)]
    if layers.size < 4:
        raise ValueError("有效 z layers 不足以建立四個垂向 receptor")

    def bracket_span(target: float) -> float:
        """回傳目標 z 最近上下有效層距；無雙側支撐時拒絕垂向外插。"""

        below = layers[layers <= target]
        above = layers[layers >= target]
        if below.size == 0 or above.size == 0:
            raise ValueError("垂向目標無上下包夾")
        return float(above.min() - below.max())

    depth = surface_z_m - bed_z_m
    specs = [
        ("upper_water_column", 0.10),
        ("mid_upper_water_column", 0.40),
        ("mid_lower_water_column", 0.70),
    ]
    targets = []
    for vertical_id, fraction in specs:
        target = surface_z_m - fraction * depth
        targets.append(
            VerticalTarget(
                vertical_id,
                fraction,
                float(target),
                float(target - bed_z_m),
                bracket_span(target),
            )
        )
    near_bed = float((layers[0] + layers[1]) * 0.5)
    targets.append(
        VerticalTarget(
            "near_bed",
            None,
            near_bed,
            near_bed - bed_z_m,
            float(layers[1] - layers[0]),
        )
    )
    if len({round(item.z_m_positive_up, 9) for item in targets}) != 4:
        raise ValueError("四個垂向目標不互異，需淘汰此水平候選")
    return targets
