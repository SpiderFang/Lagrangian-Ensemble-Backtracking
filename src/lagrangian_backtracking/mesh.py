"""保留原始海洋模式網格拓撲，並快速找出粒子所在三角形。

本模組只使用前處理資料提供的網格面與節點關係，不以通用插值工具重新建網，因此每個
三角形都能追溯回原始 SCHISM 模式網格面。四邊形依公尺制較短的對角線拆成兩個三角形；
兩條對角線等長時固定選節點 0 至 2，確保每次結果一致。內部的方格索引只用來加快候選
三角形查找，不會改變網格連接關係或三角形內插權重。建構時另預計算每個三角形的
barycentric shape-function 公尺制梯度，以及 deterministic、sorted 的 node-to-triangle
incident adjacency，供 native current 的 P1 nodal 擴散取樣使用。
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import index
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


def _build_triangle_neighbors(triangle_nodes: np.ndarray) -> np.ndarray:
    """建立「對頂點的共邊鄰居」表，並保守處理外邊界與非流形共邊。

    回傳陣列 shape 為 ``(triangle, 3)``；第 0、1、2 欄分別表示跨過該三角形
    對應頂點的對邊後所到的三角形編號。外邊界沒有鄰居，記為 ``-1``；同一條邊若
    被三個以上三角形共用，無法唯一決定穿越方向，該共邊的所有鄰接值也都保留
    ``-1``。非流形資料因此不會在建構器中被臆測成任意拓撲，也不會新增破壞性拒絕。
    """

    connectivity = np.asarray(triangle_nodes, dtype=np.int64)
    if connectivity.ndim != 2 or connectivity.shape[1] != 3:
        raise ValueError("triangle_nodes 必須是 (triangle,3)")
    neighbors = np.full(connectivity.shape, -1, dtype=np.int64)
    edge_owners: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for triangle_id, nodes in enumerate(connectivity):
        for opposite_vertex in range(3):
            edge_nodes = tuple(
                sorted(
                    (
                        int(nodes[(opposite_vertex + 1) % 3]),
                        int(nodes[(opposite_vertex + 2) % 3]),
                    )
                )
            )
            edge_owners.setdefault(edge_nodes, []).append((triangle_id, opposite_vertex))
    for owners in edge_owners.values():
        if len(owners) != 2:
            # len=1 是外邊界；len>2 是非流形共邊，兩者都不能走到唯一鄰居。
            continue
        (first_triangle, first_opposite), (second_triangle, second_opposite) = owners
        neighbors[first_triangle, first_opposite] = second_triangle
        neighbors[second_triangle, second_opposite] = first_triangle
    return neighbors


def _build_triangle_shape_gradients(vertices: np.ndarray, area2: np.ndarray) -> np.ndarray:
    """預計算每個逆時針三角形三個形函數的 ``(dNi/dx,dNi/dy)``。

    ``vertices`` 的 shape 是 ``(triangle,3,2)``，座標單位為公尺；``area2`` 是帶方向的
    兩倍面積，應已由 ``_oriented_triangle`` 保證為正。回傳陣列 shape 為
    ``(triangle,3,2)``，每一列的單位為 1/m。這裡集中檢查 shape、有限性與正面積，讓
    gradient 計算不會在退化元素上產生無限值或悄悄除以零。
    """

    coordinates = np.asarray(vertices, dtype=np.float64)
    signed_areas = np.asarray(area2, dtype=np.float64)
    if coordinates.ndim != 3 or coordinates.shape[1:] != (3, 2):
        raise ValueError("vertices 必須是有限的 (triangle,3,2)")
    if signed_areas.shape != (coordinates.shape[0],):
        raise ValueError("area2 shape 必須與 triangle 數量一致")
    if not np.all(np.isfinite(coordinates)) or not np.all(np.isfinite(signed_areas)):
        raise ValueError("vertices 與 area2 必須全部有限")
    if np.any(signed_areas <= 1.0e-8):
        raise ValueError("triangle area2 必須是正值且不可退化")
    gradients = np.empty((coordinates.shape[0], 3, 2), dtype=np.float64)
    x0 = coordinates[:, 0, 0]
    y0 = coordinates[:, 0, 1]
    x1 = coordinates[:, 1, 0]
    y1 = coordinates[:, 1, 1]
    x2 = coordinates[:, 2, 0]
    y2 = coordinates[:, 2, 1]
    gradients[:, 0, 0] = (y1 - y2) / signed_areas
    gradients[:, 0, 1] = (x2 - x1) / signed_areas
    gradients[:, 1, 0] = (y2 - y0) / signed_areas
    gradients[:, 1, 1] = (x0 - x2) / signed_areas
    gradients[:, 2, 0] = (y0 - y1) / signed_areas
    gradients[:, 2, 1] = (x1 - x0) / signed_areas
    if not np.all(np.isfinite(gradients)):
        raise ValueError("triangle shape gradients 必須全部有限")
    return gradients


def _build_node_incident_triangles(
    triangle_nodes: np.ndarray, *, node_count: int
) -> tuple[tuple[int, ...], ...]:
    """建立每個 node 對應的排序 incident triangle immutable adjacency。

    ``triangle_nodes`` 的每列是已完成方向固定的 local node ID；同一 triangle 只會對
    每個節點加入一次。建立過程使用 triangle ID 的自然遍歷後再次排序，故不依賴輸入
    adjacency 的偶然順序；節點沒有三角形支撐時保留空 tuple，供上層 fail closed。
    """

    connectivity = np.asarray(triangle_nodes, dtype=np.int64)
    if connectivity.ndim != 2 or connectivity.shape[1] != 3:
        raise ValueError("triangle_nodes 必須是 (triangle,3)")
    if type(node_count) is not int or node_count < 1:
        raise ValueError("node_count 必須是正整數")
    if connectivity.size and (
        np.any(connectivity < 0) or np.any(connectivity >= node_count)
    ):
        raise ValueError("triangle_nodes 含有超出 node_count 的索引")
    incident: list[list[int]] = [[] for _ in range(node_count)]
    for triangle_id, nodes in enumerate(connectivity):
        for node in nodes:
            incident[int(node)].append(triangle_id)
    return tuple(tuple(sorted(triangles)) for triangles in incident)


class NativeMesh:
    """已投影為公尺座標的 OCM 子網格、位置查找索引與線性形函數幾何資料。

    ``triangle_shape_gradients`` 的每列保存三個 barycentric 形函數的
    ``(dNi/dx,dNi/dy)``（單位 1/m）；``node_incident_triangles`` 則是每個 local node
    的排序 incident triangle immutable tuple。兩者只依靜態 native mesh 建立，不讀取
    時間變動 forcing，也不改變既有 locate/hint 的選擇結果。
    """

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
        # 這份鄰接表只依靜態 connectivity 建立一次；hint locator 會利用它做局部走訪，
        # 但遇到邊界、非流形或歧義位置時仍回到原本的 uniform-bin 全候選搜尋。
        self.triangle_neighbors = _build_triangle_neighbors(self.triangle_nodes)
        if self.source_face_global_index.shape != (self.face_nodes_local.shape[0],):
            raise ValueError("source_face_global_index shape 不符")
        vertices = self.node_xy[self.triangle_nodes]
        self.triangle_bbox_min = vertices.min(axis=1)
        self.triangle_bbox_max = vertices.max(axis=1)
        area2 = np.array([_signed_double_area(item) for item in vertices])
        self.triangle_area_m2 = area2 * 0.5
        # 對線性三角形形函數預先計算每個節點權重的公尺制梯度。這組值只由靜態
        # node_xy 決定，後續 OCM 速度或 Kh 取樣可直接以 ``values @ gradients`` 得到
        # 解析的一階梯度，不必在每次粒子取樣時重算，也不會誤把經緯度當成差分距離。
        self.triangle_shape_gradients = _build_triangle_shape_gradients(vertices, area2)
        self.triangle_shape_gradients.setflags(write=False)
        # 一個節點可能被多個三角形共用；固定以 triangle ID 排序並轉成巢狀 tuple，讓
        # area-weighted nodal Kh 在不同程序與不同輸入遍歷順序下都使用相同支撐集合，
        # 同時避免取樣後 caller 改寫 adjacency 破壞 provenance。
        self.node_incident_triangles = _build_node_incident_triangles(
            self.triangle_nodes, node_count=node_count
        )
        # 這個別名保留「node -> triangle」的直觀命名，兩者指向同一份 immutable tuple。
        self.node_to_incident_triangles = self.node_incident_triangles
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

    def incident_triangles(self, node_index: int) -> tuple[int, ...]:
        """回傳指定節點的排序 incident triangle ID，並拒絕非法節點索引。

        回傳值是 immutable tuple；triangle ID 的排序是 deterministic contract，供 nodal
        擴散係數的 area weighting 使用。此方法只讀靜態拓撲，不會依粒子位置做最近鄰
        推測，也不會把沒有支撐的節點補成零值。
        """

        try:
            normalized = index(node_index)
        except TypeError as error:
            raise TypeError("node_index 必須是整數") from error
        if isinstance(node_index, (bool, np.bool_)):
            raise TypeError("node_index 不可為 bool")
        if normalized < 0 or normalized >= len(self.node_incident_triangles):
            raise IndexError(f"node_index 超出 native mesh 範圍：{normalized}")
        return self.node_incident_triangles[normalized]

    def triangle_linear_gradient(
        self, triangle_id: int, scalar_values: tuple[float, float, float] | list[float] | np.ndarray
    ) -> tuple[float, float]:
        """由目前三角形三個節點 scalar 值計算公尺制線性梯度。

        ``scalar_values`` 的三個元素必須依 ``triangle_nodes[triangle_id]`` 的節點順序
        提供；回傳 ``(dscalar/dx, dscalar/dy)``，單位取決於 scalar 的單位除以公尺。
        方法嚴格檢查 triangle ID、向量 shape 與有限數值，因為 gradient 若含 NaN 或把
        其他三角形的節點順序混入，後續 Smagorinsky Kh 會失去物理可追溯性。
        """

        try:
            normalized_triangle = index(triangle_id)
        except TypeError as error:
            raise TypeError("triangle_id 必須是整數") from error
        if isinstance(triangle_id, (bool, np.bool_)):
            raise TypeError("triangle_id 不可為 bool")
        if normalized_triangle < 0 or normalized_triangle >= self.triangle_nodes.shape[0]:
            raise IndexError(f"triangle_id 超出 native mesh 範圍：{normalized_triangle}")
        try:
            values = np.asarray(scalar_values, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise TypeError("scalar_values 必須是三元素數值序列") from error
        if values.shape != (3,):
            raise ValueError("scalar_values 必須是 shape=(3,) 的 local node 向量")
        if not np.all(np.isfinite(values)):
            raise ValueError("scalar_values 必須全部有限")
        gradient = values @ self.triangle_shape_gradients[normalized_triangle]
        if not np.all(np.isfinite(gradient)):
            raise ValueError("triangle linear gradient 必須全部有限")
        return float(gradient[0]), float(gradient[1])

    # 以較短名稱提供同一個明確幾何運算；wrapper 保留文件字串，避免 caller 依賴內部
    # shape-gradient 陣列的記憶體表示，也讓「三角形 scalar gradient」的意圖易於搜尋。
    def triangle_scalar_gradient(
        self, triangle_id: int, scalar_values: tuple[float, float, float] | list[float] | np.ndarray
    ) -> tuple[float, float]:
        """回傳 ``triangle_linear_gradient`` 的同義結果，單位仍為 scalar／公尺。"""

        return self.triangle_linear_gradient(triangle_id, scalar_values)

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

    def _barycentric_weights(self, point: np.ndarray, triangle_id: int) -> np.ndarray:
        """計算指定三角形的重心座標；呼叫端已保證三角形拓撲與座標有效。"""

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
        return np.array([weight_a, weight_b, weight_c], dtype=np.float64)

    def _location_from_triangle(
        self, point: np.ndarray, triangle_id: int, *, tolerance: float
    ) -> tuple[MeshLocation | None, np.ndarray]:
        """以共用的 bbox 與重心計算判定三角形，避免 hint 與原搜尋產生兩套公式。"""

        lower = self.triangle_bbox_min[triangle_id]
        upper = self.triangle_bbox_max[triangle_id]
        if np.any(point < lower - tolerance) or np.any(point > upper + tolerance):
            return None, self._barycentric_weights(point, triangle_id)
        weights = self._barycentric_weights(point, triangle_id)
        if not np.all(weights >= -tolerance) or not np.all(weights <= 1.0 + tolerance):
            return None, weights
        face_local = int(self.triangle_face_local[triangle_id])
        nodes = self.triangle_nodes[triangle_id]
        return (
            MeshLocation(
                triangle_id=triangle_id,
                source_face_local_index=face_local,
                source_face_global_index=int(self.source_face_global_index[face_local]),
                node_indices=tuple(int(value) for value in nodes),
                barycentric_weights=tuple(float(value) for value in weights),
                triangle_area_m2=float(self.triangle_area_m2[triangle_id]),
            ),
            weights,
        )

    @staticmethod
    def _is_boundary_location(weights: np.ndarray, *, tolerance: float) -> bool:
        """判斷位置是否在三角形共邊或頂點附近，這些位置必須交回全域候選排序。"""

        return bool(
            np.any(np.abs(weights) <= tolerance)
            or np.any(np.abs(weights - 1.0) <= tolerance)
        )

    def _locate_by_uniform_bins(
        self, point: np.ndarray, *, tolerance: float
    ) -> MeshLocation | None:
        """執行既有 uniform-bin 候選搜尋，並固定採最小 triangle_id 的符合者。"""

        cell = tuple(np.floor((point - self._origin) / self.bin_size_m).astype(np.int64))
        candidates = self._bins.get((int(cell[0]), int(cell[1])), [])
        for triangle_id in sorted(candidates):
            # 先保留舊版的 bbox 篩選，避免對明顯不可能命中的候選重算重心座標；
            # hint path 則仍需在 bbox 外計算重心座標，以決定下一條對邊。
            lower = self.triangle_bbox_min[triangle_id]
            upper = self.triangle_bbox_max[triangle_id]
            if np.any(point < lower - tolerance) or np.any(point > upper + tolerance):
                continue
            location, _ = self._location_from_triangle(point, triangle_id, tolerance=tolerance)
            if location is not None:
                return location
        return None

    def _locate_from_hint(
        self,
        point: np.ndarray,
        triangle_hint: int,
        *,
        tolerance: float,
    ) -> MeshLocation | None:
        """沿鄰接三角形走訪 hint；無法可靠判定時回傳 ``None`` 交由 bins fallback。"""

        triangle_count = self.triangle_nodes.shape[0]
        if triangle_hint < 0 or triangle_hint >= triangle_count:
            return None
        # 走訪上限避免壞拓撲或循環鄰接讓單次取樣無限迴圈；上限不是三角形總數假設。
        max_steps = max(8, min(64, triangle_count))
        current = triangle_hint
        visited: set[int] = set()
        for _ in range(max_steps):
            if current in visited:
                return None
            visited.add(current)
            location, weights = self._location_from_triangle(point, current, tolerance=tolerance)
            if location is not None:
                # 邊界點可能同時屬於多面；只有原本的全域候選排序能保證 provenance。
                if self._is_boundary_location(weights, tolerance=tolerance):
                    return None
                return location
            opposite_vertex = int(np.argmin(weights))
            neighbor = int(self.triangle_neighbors[current, opposite_vertex])
            if neighbor < 0 or neighbor in visited:
                return None
            current = neighbor
        return None

    def locate(
        self,
        x_m: float,
        y_m: float,
        *,
        tolerance: float = 1e-10,
        triangle_hint: int | None = None,
    ) -> MeshLocation | None:
        """先用方格索引縮小候選，再以三角形內插權重判定位置是否落在面內。

        共用邊上的點可能同時屬於兩個三角形；固定選編號較小者，讓重新啟動與不同工作
        處理程序都得到相同的來源網格記錄。位置在網格外時回傳 ``None``，不以最近三角形
        硬做外插，以免製造不存在的流速。若提供合法的 ``triangle_hint``，先沿三角形
        對邊鄰接表走訪以縮小查找成本；走訪遇到共邊、頂點、非流形或失效提示時，會回退
        到同一套 uniform-bin 候選排序，因此不改變既有位置與 provenance 語意。
        """

        point = np.array([x_m, y_m], dtype=np.float64)
        if not np.all(np.isfinite(point)):
            return None
        if triangle_hint is not None:
            try:
                normalized_hint = index(triangle_hint)
            except TypeError:
                normalized_hint = -1
            if normalized_hint >= 0:
                hinted_location = self._locate_from_hint(
                    point, normalized_hint, tolerance=tolerance
                )
                if hinted_location is not None:
                    return hinted_location
        return self._locate_by_uniform_bins(point, tolerance=tolerance)
