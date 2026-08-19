"""SCHISM 原生 face 拓撲的決定性三角化與公尺制 point locator。

本模組只使用上游發布的 ``source_face_nodes_local``，不以 SciPy Delaunay 重建網格，
因此 ``triangle_id`` 可追溯回原始 SCHISM face。四邊形依公尺制較短對角線切分，同距
時固定選 0–2 對角線；uniform-bin index 只加速候選查找，不改變拓撲或 barycentric
權重。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import DomainProjection


@dataclass(frozen=True, slots=True)
class MeshLocation:
    """查詢點所在三角形、來源 face 與重心權重。"""

    triangle_id: int
    source_face_local_index: int
    source_face_global_index: int
    node_indices: tuple[int, int, int]
    barycentric_weights: tuple[float, float, float]
    triangle_area_m2: float


def _signed_double_area(vertices_xy: np.ndarray) -> float:
    """回傳三角形有向面積的兩倍，供方向與退化判斷。"""

    a, b, c = vertices_xy
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _oriented_triangle(nodes: tuple[int, int, int], node_xy: np.ndarray) -> tuple[int, int, int]:
    """將 triangle 固定為 counter-clockwise，退化面立即拒絕。"""

    vertices = node_xy[np.asarray(nodes)]
    area2 = _signed_double_area(vertices)
    if abs(area2) <= 1e-8:
        raise ValueError(f"發現退化 SCHISM triangle：nodes={nodes}")
    return nodes if area2 > 0 else (nodes[0], nodes[2], nodes[1])


def triangulate_faces(
    face_nodes_local: np.ndarray, face_node_count: np.ndarray, node_xy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """把 triangle/quad faces 轉成決定性三角形與來源 face index。

    quad 若 0–2 對角線不長於 1–3，切成 ``(0,1,2)+(0,2,3)``；否則採另一對角線。
    ``-1`` 只允許出現在 triangle 的第四欄，避免壞 connectivity 被負索引吞掉。
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
    """已投影的 OCM native 子網格與 uniform-bin locator。"""

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
        """驗證靜態陣列、三角化並建立 bbox-to-bin index。"""

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
        """由 OCM schema 3 ``grid/`` memory-map 靜態陣列並投影 node。"""

        root = Path(grid_dir)

        def load(name: str) -> np.ndarray:
            """以唯讀 mmap 載入單一 schema-3 grid array，缺檔時保留完整路徑失敗。"""

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
        """以 uniform bin 找候選，再用 barycentric 判定 point-in-triangle。

        共邊點可能同時落入兩個 triangle；固定選最小 triangle ID，確保 restart 與不同
        worker 得到一致 provenance。域外回傳 ``None``，不做最近 triangle 外插。
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
