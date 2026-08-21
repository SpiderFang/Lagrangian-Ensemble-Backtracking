"""保留原始海洋模式網格拓撲，並快速找出粒子所在三角形。

本模組只使用前處理資料提供的網格面與節點關係，不以通用插值工具重新建網，因此每個
三角形都能追溯回原始 SCHISM 模式網格面。四邊形依公尺制較短的對角線拆成兩個三角形；
兩條對角線等長時固定選節點 0 至 2，確保每次結果一致。內部的方格索引只用來加快候選
三角形查找，不會改變網格連接關係或三角形內插權重。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import DomainProjection


@dataclass(frozen=True, slots=True)
class MeshLocation:
    """查詢位置所在三角形、原始網格面與三個節點的內插權重。"""

    triangle_id: int
    source_face_local_index: int
    source_face_global_index: int
    node_indices: tuple[int, int, int]
    barycentric_weights: tuple[float, float, float]
    triangle_area_m2: float


def _signed_double_area(vertices_xy: np.ndarray) -> float:
    """回傳帶正負號的兩倍三角形面積，用來判定節點方向與退化面。"""

    a, b, c = vertices_xy
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _oriented_triangle(nodes: tuple[int, int, int], node_xy: np.ndarray) -> tuple[int, int, int]:
    """把三角形節點固定為逆時針順序；面積近零時立即拒絕使用。"""

    vertices = node_xy[np.asarray(nodes)]
    area2 = _signed_double_area(vertices)
    if abs(area2) <= 1e-8:
        raise ValueError(f"發現退化 SCHISM triangle：nodes={nodes}")
    return nodes if area2 > 0 else (nodes[0], nodes[2], nodes[1])


def triangulate_faces(
    face_nodes_local: np.ndarray, face_node_count: np.ndarray, node_xy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """將三角形或四邊形網格面轉成固定方式的三角形，並保留來源面索引。

    四邊形的節點 0 至 2 對角線不長於 1 至 3 時，拆成 ``(0,1,2)`` 與 ``(0,2,3)``；
    否則使用另一條對角線。``-1`` 只允許出現在三角形資料未使用的第四欄，避免錯誤的
    節點連接關係被陣列的負索引規則默默接受。
    """

    connectivity = np.asarray(face_nodes_local, dtype=np.int64)
    counts = np.asarray(face_node_count, dtype=np.int64)
    coordinates = np.asarray(node_xy, dtype=np.float64)
    if connectivity.ndim != 2 or connectivity.shape[1] != 4 or counts.shape != (connectivity.shape[0],):
        raise ValueError("face connectivity 必須是 (face,4)，count 必須是 (face,)")
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or not np.all(np.isfinite(coordinates)):
        raise ValueError("node_xy 必須是有限的 (node,2)")
    triangles: list[tuple[int, int, int]] = []
    sources: list[int] = []
    for face_index, (row, count) in enumerate(zip(connectivity, counts, strict=True)):
        if count not in (3, 4):
            raise ValueError(f"face {face_index} node_count 只能是 3 或 4")
        valid = tuple(int(value) for value in row[:count])
        if any(value < 0 or value >= coordinates.shape[0] for value in valid) or len(set(valid)) != count:
            raise ValueError(f"face {face_index} connectivity 無效")
        if count == 3:
            split = [valid]
        else:
            diagonal_02 = float(np.sum((coordinates[valid[0]] - coordinates[valid[2]]) ** 2))
            diagonal_13 = float(np.sum((coordinates[valid[1]] - coordinates[valid[3]]) ** 2))
            split = (
                [(valid[0], valid[1], valid[2]), (valid[0], valid[2], valid[3])]
                if diagonal_02 <= diagonal_13
                else [(valid[0], valid[1], valid[3]), (valid[1], valid[2], valid[3])]
            )
        for triangle in split:
            triangles.append(_oriented_triangle(triangle, coordinates))
            sources.append(face_index)
    return np.asarray(triangles, dtype=np.int64), np.asarray(sources, dtype=np.int64)


class NativeMesh:
    """已投影為公尺座標的 OCM 子網格，以及快速位置查找索引。"""

    def __init__(
        self,
        *,
        node_lon: np.ndarray,
        node_lat: np.ndarray,
        node_xy: np.ndarray,
        source_depth_m: np.ndarray,
        source_node_bottom_index: np.ndarray,
        face_nodes_local: np.ndarray,
        face_node_count: np.ndarray,
        source_face_global_index: np.ndarray,
        bin_size_m: float | None = None,
    ) -> None:
        """檢查不隨時間改變的網格資料、拆分三角形並建立查找索引。"""

        self.node_lon = np.asarray(node_lon, dtype=np.float64)
        self.node_lat = np.asarray(node_lat, dtype=np.float64)
        self.node_xy = np.asarray(node_xy, dtype=np.float64)
        self.source_depth_m = np.asarray(source_depth_m, dtype=np.float64)
        self.source_node_bottom_index = np.asarray(source_node_bottom_index, dtype=np.int64)
        self.source_face_global_index = np.asarray(source_face_global_index, dtype=np.int64)
        self.face_nodes_local = np.asarray(face_nodes_local, dtype=np.int64)
        self.face_node_count = np.asarray(face_node_count, dtype=np.int64)
        node_count = self.node_lon.size
        if (
            self.node_lat.shape != (node_count,)
            or self.node_xy.shape != (node_count, 2)
            or self.source_depth_m.shape != (node_count,)
            or self.source_node_bottom_index.shape != (node_count,)
        ):
            raise ValueError("native node 靜態陣列 shape 不一致")
        if np.any(~np.isfinite(self.node_xy)) or np.any(~np.isfinite(self.source_depth_m)):
            raise ValueError("native node 座標與水深必須有限")
        self.triangle_nodes, self.triangle_face_local = triangulate_faces(
            self.face_nodes_local, self.face_node_count, self.node_xy
        )
        if self.source_face_global_index.shape != (self.face_nodes_local.shape[0],):
            raise ValueError("source_face_global_index shape 不符")
        vertices = self.node_xy[self.triangle_nodes]
        self.triangle_bbox_min = vertices.min(axis=1)
        self.triangle_bbox_max = vertices.max(axis=1)
        area2 = np.array([_signed_double_area(item) for item in vertices])
        self.triangle_area_m2 = area2 * 0.5
        edge_lengths = np.linalg.norm(vertices - np.roll(vertices, -1, axis=1), axis=2)
        representative = float(np.median(edge_lengths[edge_lengths > 0]))
        self.bin_size_m = float(bin_size_m or max(representative * 4.0, 100.0))
        if not np.isfinite(self.bin_size_m) or self.bin_size_m <= 0:
            raise ValueError("bin_size_m 必須為有限正值")
        self._origin = self.triangle_bbox_min.min(axis=0)
        self._bins: dict[tuple[int, int], list[int]] = {}
        for triangle_id, (lower, upper) in enumerate(
            zip(self.triangle_bbox_min, self.triangle_bbox_max, strict=True)
        ):
            low_cell = np.floor((lower - self._origin) / self.bin_size_m).astype(np.int64)
            high_cell = np.floor((upper - self._origin) / self.bin_size_m).astype(np.int64)
            for ix in range(int(low_cell[0]), int(high_cell[0]) + 1):
                for iy in range(int(low_cell[1]), int(high_cell[1]) + 1):
                    self._bins.setdefault((ix, iy), []).append(triangle_id)

    @classmethod
    def from_directory(cls, grid_dir: str | Path, *, projection: DomainProjection) -> NativeMesh:
        """從 OCM 網格資料夾唯讀開啟靜態陣列，並將經緯度投影為公尺座標。"""

        root = Path(grid_dir)

        def load(name: str) -> np.ndarray:
            """唯讀開啟一個網格陣列；缺檔時顯示完整路徑以利補齊資料。"""

            path = root / name
            if not path.is_file():
                raise FileNotFoundError(f"缺少 OCM grid array：{path}")
            return np.load(path, mmap_mode="r", allow_pickle=False)

        lon = load("source_lon.npy")
        lat = load("source_lat.npy")
        x_m, y_m = projection.project(lon, lat)
        return cls(
            node_lon=lon,
            node_lat=lat,
            node_xy=np.column_stack((x_m, y_m)),
            source_depth_m=load("source_depth_m.npy"),
            source_node_bottom_index=load("source_node_bottom_index.npy"),
            face_nodes_local=load("source_face_nodes_local.npy"),
            face_node_count=load("source_face_node_count.npy"),
            source_face_global_index=load("source_face_global_index.npy"),
        )

    def locate(self, x_m: float, y_m: float, *, tolerance: float = 1e-10) -> MeshLocation | None:
        """先用方格索引縮小候選，再以三角形內插權重判定位置是否落在面內。

        共用邊上的點可能同時屬於兩個三角形；固定選編號較小者，讓重新啟動與不同工作
        處理程序都得到相同的來源網格記錄。位置在網格外時回傳 ``None``，不以最近三角形
        硬做外插，以免製造不存在的流速。
        """

        point = np.array([x_m, y_m], dtype=np.float64)
        if not np.all(np.isfinite(point)):
            return None
        cell = tuple(np.floor((point - self._origin) / self.bin_size_m).astype(np.int64))
        candidates = self._bins.get((int(cell[0]), int(cell[1])), [])
        for triangle_id in sorted(candidates):
            lower = self.triangle_bbox_min[triangle_id]
            upper = self.triangle_bbox_max[triangle_id]
            if np.any(point < lower - tolerance) or np.any(point > upper + tolerance):
                continue
            nodes = self.triangle_nodes[triangle_id]
            vertices = self.node_xy[nodes]
            a, b, c = vertices
            denominator = _signed_double_area(vertices)
            weight_a = (
                (b[0] - point[0]) * (c[1] - point[1]) - (b[1] - point[1]) * (c[0] - point[0])
            ) / denominator
            weight_b = (
                (c[0] - point[0]) * (a[1] - point[1]) - (c[1] - point[1]) * (a[0] - point[0])
            ) / denominator
            weight_c = 1.0 - weight_a - weight_b
            weights = np.array([weight_a, weight_b, weight_c])
            if np.all(weights >= -tolerance) and np.all(weights <= 1.0 + tolerance):
                face_local = int(self.triangle_face_local[triangle_id])
                return MeshLocation(
                    triangle_id=triangle_id,
                    source_face_local_index=face_local,
                    source_face_global_index=int(self.source_face_global_index[face_local]),
                    node_indices=tuple(int(value) for value in nodes),
                    barycentric_weights=tuple(float(value) for value in weights),
                    triangle_area_m2=float(self.triangle_area_m2[triangle_id]),
                )
        return None
