"""persistent-wet 水平受體與四個垂向層位的決定性生成。

每站先在固定 candidate polygon 內選 5 個 source-face 中心，再建立 upper 10%、40%、
70% 與 near-bed 四個垂向模板，形成 20 個 receptor IDs。水平候選必須在所有 50 個
arrival times 都是 wet，且與 candidate boundary 保留核定 margin；maximin 只在已通過
這些資料與海岸 gate 的候選上運作。
"""

from __future__ import annotations

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
    """從 persistent-wet faces 以 anchor-first deterministic maximin 選水平點。

    ``wetdry_at_arrivals`` shape 為 ``(arrival,source_face)``；50 個時次任一非有限或非 wet
    即剔除。``boundary_margin_m`` 同時避開 candidate polygon 外界與被 ocean clipping
    形成的岸線，若候選不足 caller 可依文件規則另立半 margin QC case，不在此靜默放寬。
    """

    wetdry = np.asarray(wetdry_at_arrivals, dtype=np.float64)
    if wetdry.ndim != 2 or wetdry.shape[1] != mesh.face_nodes_local.shape[0]:
        raise ValueError("wetdry_at_arrivals 必須是 (arrival,source_face)")
    if count < 1 or boundary_margin_m < 0:
        raise ValueError("count 必須為正且 boundary margin 不可為負")
    persistent_wet = np.all(np.isfinite(wetdry) & np.isclose(wetdry, wet_value, atol=0.1), axis=0)
    centers = _face_centers(mesh)
    candidate_indices = []
    for face_index, (x_m, y_m) in enumerate(centers):
        point = Point(float(x_m), float(y_m))
        if not persistent_wet[face_index] or not candidate_polygon_metric.covers(point):
            continue
        if boundary_margin_m > 0 and point.distance(candidate_polygon_metric.boundary) < boundary_margin_m:
            continue
        candidate_indices.append(face_index)
    if len(candidate_indices) < count:
        raise ValueError(
            f"{study_site_id} persistent-wet/margin 候選不足：需要 {count}，實際 {len(candidate_indices)}"
        )
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
    local_selected = deterministic_maximin(
        candidate_xy, count=count, anchor_xy=anchor_xy, tie_break_keys=tie_keys
    )
    selected: list[HorizontalReceptor] = []
    anchor = np.asarray(anchor_xy, dtype=np.float64)
    for order, local_index in enumerate(local_selected):
        face_index = int(indices[local_index])
        x_m, y_m = candidate_xy[local_index]
        snap_distance = float(np.linalg.norm(candidate_xy[local_index] - anchor)) if order == 0 else None
        if (
            order == 0
            and maximum_anchor_snap_distance_m is not None
            and snap_distance > maximum_anchor_snap_distance_m
        ):
            raise ValueError(
                f"{study_site_id} anchor snap {snap_distance:.3f} m "
                f"超過 {maximum_anchor_snap_distance_m:.3f} m"
            )
        identifier = stable_identifier(
            "hr",
            [
                study_site_id,
                str(int(mesh.source_face_global_index[face_index])),
                f"{x_m:.6f}",
                f"{y_m:.6f}",
            ],
        )
        selected.append(
            HorizontalReceptor(
                horizontal_receptor_id=identifier,
                study_site_id=study_site_id,
                x_m=float(x_m),
                y_m=float(y_m),
                lon=float(lon[local_index]),
                lat=float(lat[local_index]),
                source_face_local_index=face_index,
                source_face_global_index=int(mesh.source_face_global_index[face_index]),
                anchor_snap_distance_m=snap_distance,
            )
        )
    return selected


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
