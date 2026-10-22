"""WP-72：方法 B——虛擬質點回溯途中「首次抵達海岸」即視為找到來源（逐顆終止列覆寫）。

這個模組回答什麼問題
--------------------
原本的判定只看質點如何離開流場範圍：穿越範圍邊界者以穿越位置為來源，回溯滿上限仍在範圍內者以最終位置為來源。
模式海岸是反射式（質點碰岸後彈回、不停止），因此質點可能在海岸邊反覆被彈回而一直不離開範圍。
方法 B 改為：質點在回溯途中**第一次抵達海岸**，就視為已找到來源，該質點的回溯到此結束；
來源方向以抵達位置相對錨點的方向計入。未抵達海岸者維持原判定（穿越者依穿越位置、觸底者依最終位置）。

「抵達海岸」的定義（沒有任意距離門檻）
------------------------------------
質點所在的原生網格元素「有一條網格海岸邊」。網格海岸邊是只屬於一個元素的邊界邊（沒有相鄰元素），
且邊的中點落在流場範圍（flow domain）多邊形內；落在範圍外的網格外緣不算（質點在到達那裡之前就已穿越範圍邊界）。
這正是模式反射質點的邊界條件，所以不需要距離門檻。取樣是逐時輸出（3600 s），以輸出列的位置判定；
起點（第 0 列）若已在海岸元素內，也照實計入（第 0 列即抵達）。

覆寫的內容（對每顆抵達海岸的質點，抵達列為 h）
----------------------------------------------
* 終止狀態碼改為 ``NATIVE_STATUS_REACHED_COAST``（8），離開紀錄的位置與時刻改為抵達列的位置與時刻；
* 軌跡第 h 列起狀態碼改為 8，座標與深度凍結在抵達位置（與既有終止凍結慣例相同：非 0 狀態列不計入停留時數與回溯快照）；
* 原本是穿越者，其穿越紀錄改為本類別，``also_local_domain_first_exit`` 清為 False；
* 抵達時刻之後才發生的 local domain 首次離開事件（非終止事件）刪除，因為質點已在抵達當下停止。
輸入檔（``*.driver_result.npz``）完全不改；覆寫只發生在後處理讀入之後。
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .backward_driver import (
    NATIVE_STATUS_ACTIVE,
    NATIVE_STATUS_REACHED_COAST,
    BackwardDriverResult,
    DepartureRecord,
)

__all__ = [
    "CoastArrivalError",
    "CoastGeometry",
    "coast_geometry_from_mesh",
    "load_coast_geometry",
    "find_first_coast_row",
    "apply_coast_arrival",
    "CoastArrivalOverride",
]


class CoastArrivalError(RuntimeError):
    """方法 B（首次抵達海岸）的資料契約錯誤，一律 fail-closed。"""


@dataclass(frozen=True)
class CoastGeometry:
    """單一流場網域的網格海岸幾何。

    ``finder``：由原生網格三角形建立的 matplotlib 三角形搜尋器（座標與軌跡同為該網域的局部等距方位投影，公尺）。
    ``coast_element``：每個元素是否有網格海岸邊（布林）。``n_boundary_edges``／``n_coast_edges``：網格邊界邊數與其中
    落在流場範圍內的網格海岸邊數；``n_coast_elements``：有網格海岸邊的元素數。
    """

    finder: Any
    coast_element: np.ndarray
    n_elements: int
    n_boundary_edges: int
    n_coast_edges: int
    n_coast_elements: int


def coast_geometry_from_mesh(x: np.ndarray, y: np.ndarray, tri: np.ndarray, flow_polygon_metric: Any) -> CoastGeometry:
    """由三角網格（公尺座標）與流場範圍多邊形（同座標）建立 :class:`CoastGeometry`。"""

    import shapely
    from matplotlib.tri import Triangulation

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    tri = np.asarray(tri, dtype=np.int64)
    if tri.ndim != 2 or tri.shape[1] != 3:
        raise CoastArrivalError("方法 B 目前只支援全三角形網格（tri 形狀必須是 (元素數, 3)）")
    ne = tri.shape[0]
    nn = x.size
    e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    owner = np.tile(np.arange(ne), 3)
    a = np.minimum(e[:, 0], e[:, 1])
    b = np.maximum(e[:, 0], e[:, 1])
    key = a * (nn + 1) + b
    _uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    boundary = cnt[inv] == 1
    ea, eb, eo = a[boundary], b[boundary], owner[boundary]
    mx = 0.5 * (x[ea] + x[eb])
    my = 0.5 * (y[ea] + y[eb])
    inside = shapely.contains_xy(flow_polygon_metric, mx, my)
    coast_el = np.zeros(ne, dtype=bool)
    coast_el[eo[inside]] = True
    finder = Triangulation(x, y, tri).get_trifinder()
    return CoastGeometry(
        finder=finder,
        coast_element=coast_el,
        n_elements=int(ne),
        n_boundary_edges=int(boundary.sum()),
        n_coast_edges=int(inside.sum()),
        n_coast_elements=int(coast_el.sum()),
    )


def load_coast_geometry(
    grid_dir: str | Path, center_lonlat: tuple[float, float], bbox_lon_lat: tuple[float, float, float, float]
) -> CoastGeometry:
    """讀 OCM 網格目錄（``source_lon／source_lat／source_face_nodes_local／source_face_node_count``）並建立海岸幾何。

    投影與執行時相同：以該網域 ``center_lonlat`` 為中心的局部等距方位投影；流場範圍多邊形為 ``bbox_lon_lat`` 加密後投影。
    """

    from ..geometry import DomainProjection, densified_bbox_polygon

    grid = Path(grid_dir)
    lon = np.load(grid / "source_lon.npy").astype(np.float64)
    lat = np.load(grid / "source_lat.npy").astype(np.float64)
    faces = np.load(grid / "source_face_nodes_local.npy")
    count = np.load(grid / "source_face_node_count.npy")
    if not np.all(count == 3):
        raise CoastArrivalError(f"{grid}：含非三角形元素，方法 B 不支援")
    projection = DomainProjection(float(center_lonlat[0]), float(center_lonlat[1]))
    x, y = projection.project(lon, lat)
    polygon = projection.project_geometry(densified_bbox_polygon(tuple(float(v) for v in bbox_lon_lat), points_per_edge=64))
    return coast_geometry_from_mesh(x, y, faces[:, :3].astype(np.int64), polygon)


def find_first_coast_row(
    trajectory_x_m: np.ndarray,
    trajectory_y_m: np.ndarray,
    trajectory_status: np.ndarray,
    geometry: CoastGeometry,
) -> tuple[np.ndarray, int]:
    """每顆質點首次抵達海岸的軌跡列（0 起算；沒有則 -1），以及追蹤中取樣點落在網格外的點數。

    只看狀態碼為 0（追蹤中）的列；每顆質點取第一個所在元素是海岸元素的列。
    """

    status = np.asarray(trajectory_status)
    active = status == NATIVE_STATUS_ACTIVE
    elem = np.full(status.shape, -1, dtype=np.int64)
    elem[active] = geometry.finder(
        np.asarray(trajectory_x_m, dtype=np.float64)[active], np.asarray(trajectory_y_m, dtype=np.float64)[active]
    )
    outside = int(np.count_nonzero(active & (elem < 0)))
    hit = active & (elem >= 0) & geometry.coast_element[np.maximum(elem, 0)]
    first = np.where(hit.any(axis=0), hit.argmax(axis=0), -1).astype(np.int64)
    return first, outside


def apply_coast_arrival(result: BackwardDriverResult, hit_row: np.ndarray) -> BackwardDriverResult:
    """把「首次抵達海岸列」覆寫進批次結果（見模組說明）；``hit_row`` 為每顆質點的抵達列，-1 表示沒有抵達。"""

    hit_row = np.asarray(hit_row, dtype=np.int64)
    n = int(result.row_particle_index.size)
    if hit_row.shape != (n,):
        raise CoastArrivalError("hit_row 長度與批次粒子數不一致")
    hit = hit_row >= 0
    if not hit.any():
        return result
    tx = np.array(result.trajectory_x_m, copy=True)
    ty = np.array(result.trajectory_y_m, copy=True)
    tz = np.array(result.trajectory_z_m, copy=True)
    ts = np.array(result.trajectory_status, copy=True)
    final_status = np.array(result.final_status, copy=True)
    fx = np.array(result.final_x_m, copy=True)
    fy = np.array(result.final_y_m, copy=True)
    fz = np.array(result.final_z_m, copy=True)
    futc = np.array(result.final_utc_ns, copy=True)
    utc = np.asarray(result.trajectory_utc_ns)
    new_dep: dict[int, DepartureRecord] = {}
    for p in np.flatnonzero(hit):
        h = int(hit_row[p])
        if h >= ts.shape[0]:
            raise CoastArrivalError("抵達列超出軌跡長度")
        tx[h:, p] = tx[h, p]
        ty[h:, p] = ty[h, p]
        tz[h:, p] = tz[h, p]
        ts[h:, p] = NATIVE_STATUS_REACHED_COAST
        final_status[p] = NATIVE_STATUS_REACHED_COAST
        fx[p], fy[p], fz[p], futc[p] = tx[h, p], ty[h, p], tz[h, p], int(utc[h])
        new_dep[int(p)] = DepartureRecord(
            row_particle_index=int(p),
            status=NATIVE_STATUS_REACHED_COAST,
            x_m=float(tx[h, p]),
            y_m=float(ty[h, p]),
            z_m=float(tz[h, p]),
            utc_ns=int(utc[h]),
            also_local_domain_first_exit=False,
        )
    departures = tuple(new_dep.get(int(r.row_particle_index), r) for r in result.departures)
    seen = {int(r.row_particle_index) for r in departures}
    if seen != set(range(n)) and any(int(p) not in seen for p in new_dep):
        raise CoastArrivalError("抵達海岸的質點在原結果中沒有離開紀錄，結果不一致")
    hit_utc = {int(p): int(utc[int(hit_row[p])]) for p in np.flatnonzero(hit)}
    local = tuple(
        r for r in result.local_domain_departures
        if int(r.row_particle_index) not in hit_utc or int(r.utc_ns) >= hit_utc[int(r.row_particle_index)]
    )
    return dataclasses.replace(
        result,
        final_status=final_status,
        final_x_m=fx,
        final_y_m=fy,
        final_z_m=fz,
        final_utc_ns=futc,
        trajectory_x_m=tx,
        trajectory_y_m=ty,
        trajectory_z_m=tz,
        trajectory_status=ts,
        departures=departures,
        local_domain_departures=local,
    )


class CoastArrivalOverride:
    """各站的網格海岸幾何與覆寫入口；後處理對每個批次呼叫 :meth:`apply`，並累計摘要。"""

    def __init__(self, geometry_by_site: Mapping[str, CoastGeometry]) -> None:
        self._geometry = dict(geometry_by_site)
        self._stats: dict[str, dict[str, int]] = {
            s: {"particles": 0, "reached_coast": 0, "reached_at_row0": 0, "points_outside_mesh": 0} for s in self._geometry
        }

    def apply(self, site_id: str, result: BackwardDriverResult) -> tuple[BackwardDriverResult, np.ndarray]:
        geometry = self._geometry.get(site_id)
        if geometry is None:
            raise CoastArrivalError(f"缺少站 {site_id!r} 的網格海岸幾何")
        hit_row, outside = find_first_coast_row(
            result.trajectory_x_m, result.trajectory_y_m, result.trajectory_status, geometry
        )
        st = self._stats[site_id]
        st["particles"] += int(hit_row.size)
        st["reached_coast"] += int((hit_row >= 0).sum())
        st["reached_at_row0"] += int((hit_row == 0).sum())
        st["points_outside_mesh"] += outside
        return apply_coast_arrival(result, hit_row), hit_row

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for site, g in self._geometry.items():
            out[site] = {
                **self._stats[site],
                "mesh_elements": g.n_elements,
                "mesh_boundary_edges": g.n_boundary_edges,
                "coast_edges_inside_flow_domain": g.n_coast_edges,
                "coast_elements": g.n_coast_elements,
            }
        return out
