"""向量化前 b9d5456 的逐節點／元素參考；只供bitwise與首個拒絕理由測試。

保留原運算順序與dtype語意，不作正式執行入口，也不讀外部流場或PI原始資料。
"""

from typing import Any

import numpy as np

from lagrangian_backtracking.baytrace_native_cpu import (
    BayTraceFieldPair,
    _float_array,
    _int_array,
    _positive_int,
    _require_finite,
    _signed_triangle_area,
)
from lagrangian_backtracking.baytrace_native_cpu import (
    BayTraceGridArrays as ActualGrid,
)
from lagrangian_backtracking.baytrace_ocm_adapter import _SIGMA_TOLERANCE, _FramePreparationFallback


def reference_grid_init(self) -> None:
    """一次性轉成 Numba 可安全讀取的連續陣列並驗證網格契約。"""

    i34 = _int_array(self.i34, "i34", 1)
    elnode = _int_array(self.elnode, "elnode", 2)
    grid_x = _float_array(self.grid_x, "grid_x", 1)
    grid_y = _float_array(self.grid_y, "grid_y", 1)
    ic3 = _int_array(self.ic3, "ic3", 2)
    kbp = _int_array(self.kbp, "kbp", 1)
    kbe = _int_array(self.kbe, "kbe", 1)
    sigma_lcl = _float_array(self.sigma_lcl, "sigma_lcl", 2)
    dp = _float_array(self.dp, "dp", 1)
    nvrt = _positive_int(self.nvrt, "nvrt")

    ne = i34.size
    nnode = grid_x.size
    if ne == 0:
        raise ValueError("i34 不得為空")
    if grid_y.shape != grid_x.shape:
        raise ValueError("grid_x 與 grid_y shape 必須一致")
    if elnode.shape != (ne, 4):
        raise ValueError("elnode 必須是 (元素數, 4) 的零起始節點索引")
    if ic3.shape != (4, ne):
        raise ValueError("ic3 必須是 BayTrace 原始布局 (4, 元素數)")
    if kbp.shape != (nnode,):
        raise ValueError("kbp shape 必須等於節點數")
    if kbe.shape != (ne,):
        raise ValueError("kbe shape 必須等於元素數")
    if sigma_lcl.shape != (nvrt, nnode):
        raise ValueError("sigma_lcl shape 必須是 (nvrt, 節點數)")
    if dp.shape != (nnode,):
        raise ValueError("dp shape 必須等於節點數")
    if np.any(~np.isin(i34, (3, 4))):
        raise ValueError("i34 只能包含 3 或 4")
    if np.any(kbp < 0) or np.any(kbp >= nvrt) or np.any(kbe < 0) or np.any(kbe >= nvrt):
        raise ValueError("kbp/kbe 含超出垂向層數的索引")
    if np.any(dp <= 0.0):
        raise ValueError("dp 必須是有限的正水深")
    if np.any((ic3 < -1) | (ic3 >= ne)):
        raise ValueError("ic3 只能使用 -1 或有效元素索引")

    # 只檢查每個元素實際使用的節點；三角形第四欄固定為 -1，避免
    # 不小心把無效欄位送進原始 kernel 的節點索引運算。
    for element, node_count in enumerate(i34):
        used_nodes = elnode[element, :node_count]
        if np.any(used_nodes < 0) or np.any(used_nodes >= nnode):
            raise ValueError(f"elnode[{element}] 含無效節點索引")
        if node_count == 3 and elnode[element, 3] != -1:
            raise ValueError("三角形 elnode 的未使用第四欄必須是 -1")

        # 原始 _pt_in_poly3 只接受正向且非退化三角形；在 Python 層
        # 先擋住退化/反向面，避免編譯後才產生難追的錯誤內插。
        first_area = _signed_triangle_area(grid_x, grid_y, used_nodes[:3])
        if first_area <= 1.0e-12:
            raise ValueError(f"元素 {element} 的第一個三角形方向或面積無效")
        if node_count == 4:
            second_area = _signed_triangle_area(grid_x, grid_y, used_nodes[[0, 2, 3]])
            if second_area <= 1.0e-12:
                raise ValueError(f"元素 {element} 的第二個三角形方向或面積無效")

    for name, array in (
        ("grid_x", grid_x),
        ("grid_y", grid_y),
        ("sigma_lcl", sigma_lcl),
        ("dp", dp),
    ):
        _require_finite(array, name)

    object.__setattr__(self, "i34", i34)
    object.__setattr__(self, "elnode", elnode)
    object.__setattr__(self, "grid_x", grid_x)
    object.__setattr__(self, "grid_y", grid_y)
    object.__setattr__(self, "ic3", ic3)
    object.__setattr__(self, "nvrt", nvrt)
    object.__setattr__(self, "kbp", kbp)
    object.__setattr__(self, "kbe", kbe)
    object.__setattr__(self, "sigma_lcl", sigma_lcl)
    object.__setattr__(self, "dp", dp)



def BayTraceGridArrays(**kwargs):
    """直接套用保存的舊檢核，避免參考路徑意外使用新版向量化驗證。"""
    grid = object.__new__(ActualGrid)
    for name, value in kwargs.items():
        object.__setattr__(grid, name, value)
    reference_grid_init(grid)
    return grid

def reference_build_grid_and_fields(
    self,
    mesh: Any,
    rows: dict[str, np.ndarray],
    *,
    settling_velocity_mps: float,
) -> tuple[BayTraceGridArrays, BayTraceFieldPair]:
    """由兩端 row 建立 OCM native grid/field；所有物理軸仍是公尺與 m/s。"""

    node_xy = np.asarray(mesh.node_xy)
    dp = np.asarray(mesh.source_depth_m)
    triangle_nodes = np.asarray(mesh.triangle_nodes)
    neighbors = np.asarray(mesh.triangle_neighbors)
    z_before = np.asarray(rows["z_before"])
    z_after = np.asarray(rows["z_after"])
    eta_before = np.asarray(rows["eta_before"])
    eta_after = np.asarray(rows["eta_after"])
    hvel_before = np.asarray(rows["hvel_before"])
    hvel_after = np.asarray(rows["hvel_after"])
    vertical_before = np.asarray(rows["vertical_before"])
    vertical_after = np.asarray(rows["vertical_after"])

    nnode = int(node_xy.shape[0])
    if (
        node_xy.shape != (nnode, 2)
        or dp.shape != (nnode,)
        or triangle_nodes.ndim != 2
        or triangle_nodes.shape[1] != 3
        or neighbors.shape != triangle_nodes.shape
        or z_before.shape != z_after.shape
        or z_before.shape[0] != nnode
        or eta_before.shape != (nnode,)
        or eta_after.shape != (nnode,)
        or hvel_before.ndim != 3
        or hvel_after.shape != hvel_before.shape
        or hvel_before.shape[:2] != z_before.shape
        or hvel_before.shape[2] < 2
        or vertical_before.shape != z_before.shape
        or vertical_after.shape != z_after.shape
    ):
        raise _FramePreparationFallback("frame_shape_invalid")
    nvrt = int(z_before.shape[1])
    if nvrt < 2:
        raise _FramePreparationFallback("vertical_support_invalid")
    if np.any(~np.isfinite(dp)) or np.any(dp <= 0.0):
        raise _FramePreparationFallback("invalid_depth")
    if np.any(~np.isfinite(eta_before)) or np.any(~np.isfinite(eta_after)):
        raise _FramePreparationFallback("nonfinite_field")

    kbp = np.empty(nnode, dtype=np.int64)
    sigma_lcl = np.empty((nvrt, nnode), dtype=np.float64)
    for node in range(nnode):
        finite_before = np.flatnonzero(np.isfinite(z_before[node]))
        finite_after = np.flatnonzero(np.isfinite(z_after[node]))
        if finite_before.size == 0 or finite_after.size == 0:
            raise _FramePreparationFallback("invalid_vertical_support")
        first = int(finite_before[0])
        if int(finite_after[0]) != first:
            raise _FramePreparationFallback("vertical_support_mismatch")
        # first finite layer 是原版 kbp；其下未使用層只在輸出場中重複此
        # kbp 值，等價於原版 find_vlevel 的 kbp clamp，不以 0 補物理量。
        if not np.all(np.isfinite(z_before[node, first:])) or not np.all(
            np.isfinite(z_after[node, first:])
        ):
            raise _FramePreparationFallback("invalid_vertical_support")
        denominator_before = float(dp[node] + eta_before[node])
        denominator_after = float(dp[node] + eta_after[node])
        if denominator_before <= 0.0 or denominator_after <= 0.0:
            raise _FramePreparationFallback("invalid_depth")
        sigma_before = (z_before[node, first:] - eta_before[node]) / denominator_before
        sigma_after = (z_after[node, first:] - eta_after[node]) / denominator_after
        if not np.all(np.isfinite(sigma_before)) or not np.all(np.isfinite(sigma_after)):
            raise _FramePreparationFallback("invalid_vertical_support")
        if np.any(np.diff(z_before[node, first:]) < -_SIGMA_TOLERANCE) or np.any(
            np.diff(z_after[node, first:]) < -_SIGMA_TOLERANCE
        ):
            raise _FramePreparationFallback("vertical_nonmonotonic")
        if np.any(np.diff(sigma_before) < -_SIGMA_TOLERANCE) or np.any(
            np.diff(sigma_after) < -_SIGMA_TOLERANCE
        ):
            raise _FramePreparationFallback("sigma_nonmonotonic")
        if (
            np.any(sigma_before < -1.0 - _SIGMA_TOLERANCE)
            or np.any(sigma_before > _SIGMA_TOLERANCE)
            or np.any(sigma_after < -1.0 - _SIGMA_TOLERANCE)
            or np.any(sigma_after > _SIGMA_TOLERANCE)
        ):
            raise _FramePreparationFallback("sigma_out_of_range")
        reconstructed_before = eta_before[node] + sigma_before * denominator_before
        reconstructed_after = eta_after[node] + sigma_after * denominator_after
        if not np.allclose(
            reconstructed_before,
            z_before[node, first:],
            rtol=1.0e-6,
            atol=1.0e-6,
        ) or not np.allclose(
            reconstructed_after,
            z_after[node, first:],
            rtol=1.0e-6,
            atol=1.0e-6,
        ):
            raise _FramePreparationFallback("zcor_reconstruction_mismatch")
        kbp[node] = first
        sigma_lcl[first:, node] = sigma_before
        sigma_lcl[:first, node] = sigma_before[0]

        for values, label in (
            (hvel_before[node, first:, :2], "hvel_before"),
            (hvel_after[node, first:, :2], "hvel_after"),
            (vertical_before[node, first:], "vertical_before"),
            (vertical_after[node, first:], "vertical_after"),
        ):
            if not np.all(np.isfinite(values)):
                raise _FramePreparationFallback("nonfinite_field", label)

    element_nodes = triangle_nodes
    if np.any(element_nodes < 0) or np.any(element_nodes >= nnode):
        raise _FramePreparationFallback("mesh_topology_invalid")
    kbe = np.min(kbp[element_nodes], axis=1).astype(np.int64, copy=False)
    ne = int(triangle_nodes.shape[0])
    i34 = np.full(ne, 3, dtype=np.int64)
    elnode = np.full((ne, 4), -1, dtype=np.int64)
    elnode[:, :3] = triangle_nodes.astype(np.int64, copy=False)
    ic3 = np.full((4, ne), -1, dtype=np.int64)
    ic3[:3, :] = neighbors.T.astype(np.int64, copy=False)

    def transpose_with_kbp(values: np.ndarray, *, add: float = 0.0) -> np.ndarray:
        """轉成 BayTrace (layer,node)，並以 kbp clamp 重複未使用底層。"""

        result = np.empty((nvrt, nnode), dtype=np.float64)
        for node in range(nnode):
            first = int(kbp[node])
            result[first:, node] = np.asarray(values[node, first:], dtype=np.float64) + add
            result[:first, node] = result[first, node]
        if not np.all(np.isfinite(result)):
            raise _FramePreparationFallback("nonfinite_result")
        return result

    u1 = transpose_with_kbp(hvel_before[:, :, 0])
    v1 = transpose_with_kbp(hvel_before[:, :, 1])
    w1 = transpose_with_kbp(vertical_before, add=settling_velocity_mps)
    u2 = transpose_with_kbp(hvel_after[:, :, 0])
    v2 = transpose_with_kbp(hvel_after[:, :, 1])
    w2 = transpose_with_kbp(vertical_after, add=settling_velocity_mps)
    eta1 = np.asarray(eta_before, dtype=np.float64).copy()
    if not np.all(np.isfinite(eta1)):
        raise _FramePreparationFallback("nonfinite_field")

    grid = BayTraceGridArrays(
        i34=i34,
        elnode=elnode,
        grid_x=np.asarray(node_xy[:, 0], dtype=np.float64),
        grid_y=np.asarray(node_xy[:, 1], dtype=np.float64),
        ic3=ic3,
        nvrt=nvrt,
        kbp=kbp,
        kbe=kbe,
        sigma_lcl=sigma_lcl,
        dp=np.asarray(dp, dtype=np.float64),
    )
    fields = BayTraceFieldPair(
        u1=u1,
        v1=v1,
        w1=w1,
        eta1=eta1,
        u2=u2,
        v2=v2,
        w2=w2,
    )
    return grid, fields

