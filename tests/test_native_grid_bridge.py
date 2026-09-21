"""WP-1 靜態網格橋接（``native_runner.grid_bridge``）的驗收測試。

測試涵蓋：三角網格 ``ic3`` 邊鄰接、輸出陣列 shape／dtype／contiguity、
zcor→sigma_lcl→z 往返容差、故意用隨時間變動資料證明檢核函式會回報偏差、
把產出陣列真的餵進 BayTrace 原版 ``batch_interpolate_velocity``，以及
乾點／海底以下／缺值等邊界情況。
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from lagrangian_backtracking.mesh import NativeMesh
from lagrangian_backtracking.native_runner import (
    NativeGridBridgeError,
    build_native_grid_arrays,
    evaluate_static_sigma_error,
)

_NX = 4
_NY = 4
_SPACING_M = 100.0
_NVRT = 6
_ETA_REFERENCE = 0.2


def _node_index(i: int, j: int) -> int:
    """規則格網節點的一維索引：沿 x 先走，再沿 y 走。"""

    return j * _NX + i


def _kbp_true(i: int, j: int) -> int:
    """合成資料的「真實」kbp：依 (i+j) 對 3 取餘數，讓不同節點層數不同。"""

    return (i + j) % 3


def _sigma_true(node_i: int, node_j: int) -> np.ndarray:
    """回傳節點 (i, j) 在其有效層範圍內的固定分率 sigma（由海床到海面線性）。

    只在有效層（``k >= kbp``）內定義；長度是 ``_NVRT - kbp``。這代表一個
    真正符合 LSC2 假設的合成世界：sigma 分率不隨時間或海面高度改變。
    """

    kbp = _kbp_true(node_i, node_j)
    return np.linspace(-1.0, 0.0, _NVRT - kbp)


def _build_regular_mesh() -> tuple[NativeMesh, np.ndarray]:
    """建立 4x4 節點、3x3 格、每格切 2 個三角形的規則公尺制網格。

    回傳 mesh 與每個節點的正值水深 ``dp``（供測試直接重算合成 zcor 使用）。
    """

    n_nodes = _NX * _NY
    node_xy = np.zeros((n_nodes, 2), dtype=np.float64)
    dp = np.zeros(n_nodes, dtype=np.float64)
    for j in range(_NY):
        for i in range(_NX):
            idx = _node_index(i, j)
            node_xy[idx] = (i * _SPACING_M, j * _SPACING_M)
            dp[idx] = 15.0 + 2.0 * (i + j)

    triangles: list[tuple[int, int, int]] = []
    for j in range(_NY - 1):
        for i in range(_NX - 1):
            n00 = _node_index(i, j)
            n10 = _node_index(i + 1, j)
            n11 = _node_index(i + 1, j + 1)
            n01 = _node_index(i, j + 1)
            # 兩個三角形皆為逆時針（x 右、y 上），與 mesh.py 的方向慣例一致。
            triangles.append((n00, n10, n11))
            triangles.append((n00, n11, n01))

    n_faces = len(triangles)
    face_nodes_local = np.full((n_faces, 4), -1, dtype=np.int64)
    for face_index, nodes in enumerate(triangles):
        face_nodes_local[face_index, :3] = nodes
    face_node_count = np.full(n_faces, 3, dtype=np.int64)

    mesh = NativeMesh(
        node_lon=np.linspace(120.0, 120.01, n_nodes),
        node_lat=np.linspace(24.0, 24.01, n_nodes),
        node_xy=node_xy,
        source_depth_m=dp,
        source_node_bottom_index=np.zeros(n_nodes, dtype=np.int64),
        face_nodes_local=face_nodes_local,
        face_node_count=face_node_count,
        source_face_global_index=np.arange(n_faces, dtype=np.int64),
        bin_size_m=200.0,
    )
    return mesh, dp


def _synthetic_zcor(mesh: NativeMesh, dp: np.ndarray, eta: float) -> np.ndarray:
    """依「真實」固定分率 sigma 產生某一時刻的合成 zcor，shape (節點數, nvrt)。

    海床以下（``k < kbp``）填 NaN，代表 OCM 對該柱缺乏支撐，不得以零值頂替。
    """

    n_nodes = mesh.node_xy.shape[0]
    zcor = np.full((n_nodes, _NVRT), np.nan, dtype=np.float64)
    for j in range(_NY):
        for i in range(_NX):
            idx = _node_index(i, j)
            kbp = _kbp_true(i, j)
            sigma = _sigma_true(i, j)
            zcor[idx, kbp:] = eta + sigma * (dp[idx] + eta)
    return zcor


def _reference_grid():
    """回傳供多個測試共用的 mesh、dp 與參考時刻建成的 NativeGridArrays。"""

    mesh, dp = _build_regular_mesh()
    zcor_reference = _synthetic_zcor(mesh, dp, _ETA_REFERENCE)
    eta_reference = np.full(mesh.node_xy.shape[0], _ETA_REFERENCE, dtype=np.float64)
    grid = build_native_grid_arrays(
        mesh, zcor_reference=zcor_reference, eta_reference=eta_reference
    )
    return mesh, dp, grid


# ── 1. ic3 邊鄰接：每條內部邊剛好兩個擁有者互指 ──────────────────────────


def test_ic3_edge_adjacency_is_reciprocal_on_regular_triangulation() -> None:
    """獨立重算每條邊的擁有者，驗證 ic3 對內部邊的互指關係，外邊界為 -1。"""

    _, _, grid = _reference_grid()
    ne = grid.i34.shape[0]
    assert np.all(grid.i34 == 3)

    edge_owners: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for elem in range(ne):
        nodes = grid.elnode[elem, :3]
        for local_vertex in range(3):
            a = int(nodes[(local_vertex + 1) % 3])
            b = int(nodes[(local_vertex + 2) % 3])
            key = (min(a, b), max(a, b))
            edge_owners.setdefault(key, []).append((elem, local_vertex))

    interior_edges = 0
    boundary_edges = 0
    for owners in edge_owners.values():
        if len(owners) == 2:
            interior_edges += 1
            (elem_a, edge_a), (elem_b, edge_b) = owners
            assert int(grid.ic3[edge_a, elem_a]) == elem_b
            assert int(grid.ic3[edge_b, elem_b]) == elem_a
        elif len(owners) == 1:
            boundary_edges += 1
            (elem_a, edge_a) = owners[0]
            assert int(grid.ic3[edge_a, elem_a]) == -1
        else:
            pytest.fail("規則三角化不應出現三個以上共邊三角形")

    assert interior_edges > 0 and boundary_edges > 0
    # 3x3 格、每格 2 個三角形 => 18 個三角形，每個三角形第 4 欄固定 -1。
    assert np.all(grid.ic3[3, :] == -1)


# ── 2. elnode/i34/kbe 的 shape、dtype、contiguity ────────────────────────


def test_output_array_shapes_dtypes_and_contiguity() -> None:
    """所有輸出陣列必須是 C-contiguous 的 int64／float64，shape 符合 kernel 契約。"""

    mesh, dp, grid = _reference_grid()
    n_nodes = mesh.node_xy.shape[0]
    ne = grid.i34.shape[0]

    assert grid.i34.shape == (ne,) and grid.i34.dtype == np.int64
    assert grid.elnode.shape == (ne, 4) and grid.elnode.dtype == np.int64
    assert grid.grid_x.shape == (n_nodes,) and grid.grid_x.dtype == np.float64
    assert grid.grid_y.shape == (n_nodes,) and grid.grid_y.dtype == np.float64
    assert grid.ic3.shape == (4, ne) and grid.ic3.dtype == np.int64
    assert grid.kbp.shape == (n_nodes,) and grid.kbp.dtype == np.int64
    assert grid.kbe.shape == (ne,) and grid.kbe.dtype == np.int64
    assert grid.sigma_lcl.shape == (_NVRT, n_nodes) and grid.sigma_lcl.dtype == np.float64
    assert grid.dp.shape == (n_nodes,) and grid.dp.dtype == np.float64
    assert grid.nvrt == _NVRT

    for array in (
        grid.i34, grid.elnode, grid.grid_x, grid.grid_y, grid.ic3,
        grid.kbp, grid.kbe, grid.sigma_lcl, grid.dp,
    ):
        assert array.flags["C_CONTIGUOUS"]

    # kbe 必須是三角形三個節點 kbp 的最小值（元素層級底層）。
    for elem in range(ne):
        nodes = grid.elnode[elem, :3]
        assert int(grid.kbe[elem]) == int(grid.kbp[nodes].min())

    np.testing.assert_array_equal(grid.dp, dp)


# ── 3. zcor → sigma_lcl → z 往返容差 ─────────────────────────────────────


def test_round_trip_reconstruction_matches_within_tight_tolerance() -> None:
    """合成世界真的符合固定分率 LSC2 時，靜態 sigma 在其他時刻幾乎零誤差。"""

    mesh, dp, grid = _reference_grid()
    n_nodes = mesh.node_xy.shape[0]

    other_etas = np.array([-0.3, 0.0, 0.5, 1.1])
    zcor_series = np.stack(
        [_synthetic_zcor(mesh, dp, float(eta)) for eta in other_etas], axis=0
    )
    eta_series = np.tile(other_etas[:, None], (1, n_nodes))

    report = evaluate_static_sigma_error(
        grid, zcor_series=zcor_series, eta_series=eta_series
    )

    # 合成資料是精確的固定分率模型，重建誤差只剩浮點捨入，遠低於任何物理量級。
    assert report.max_abs_z_error_m < 1.0e-8
    assert report.max_abs_sigma_error < 1.0e-10
    assert report.evaluated_point_count > 0
    assert report.skipped_point_count >= 0


# ── 4. 故意用隨時間變動資料，證明檢核函式會回報偏差 ─────────────────────


def test_evaluator_detects_drift_when_sigma_is_not_actually_static() -> None:
    """真實垂向座標若不是固定分率，檢核函式必須回報明顯偏差，不能被靜默放行。"""

    mesh, dp, grid = _reference_grid()
    n_nodes = mesh.node_xy.shape[0]

    other_etas = np.array([-0.3, 0.0, 0.5, 1.1])
    zcor_series = np.stack(
        [_synthetic_zcor(mesh, dp, float(eta)) for eta in other_etas], axis=0
    )
    eta_series = np.tile(other_etas[:, None], (1, n_nodes))

    # 刻意疊加一個「非固定分率」的額外位移，模擬垂向座標本身隨海面高度
    # 額外變動（例如混合 Z/sigma 座標在轉換帶附近的行為）；只加在有效層，
    # 且保持隨層遞增以維持單調，避免污染成「資料本身不合法」而非「靜態化
    # 誤差」。
    drift_series = zcor_series.copy()
    for t_index, _eta in enumerate(other_etas):
        drift_series[t_index] += 0.4 * (t_index + 1)

    report = evaluate_static_sigma_error(
        grid, zcor_series=drift_series, eta_series=eta_series
    )

    # 漂移量級是「幾十公分乘上時刻編號」，遠高於往返測試量到的浮點雜訊，
    # 證明檢核函式確實偵測到「靜態 sigma 不再精確吻合」而不是靜默通過。
    assert report.max_abs_z_error_m > 0.1
    assert report.max_abs_sigma_error > 1.0e-4


# ── 5. 產出陣列真的餵進 BayTrace batch_interpolate_velocity ─────────────


@pytest.fixture
def _original_batch_interp(tmp_path: Path):
    """只有明確設定 LBT_BAYTRACE_ROOT 時才載入原版 njit kernel。"""

    from lagrangian_backtracking.baytrace_native_cpu import load_original_cpu

    raw_root = os.environ.get("LBT_BAYTRACE_ROOT")
    if not raw_root:
        pytest.skip("未設定 LBT_BAYTRACE_ROOT；略過需要 BayTrace 原始 artifact 的測試")
    source_root = Path(raw_root)
    if not source_root.exists():
        pytest.fail(f"LBT_BAYTRACE_ROOT 不存在: {source_root}")
    kernels = load_original_cpu(source_root, tmp_path / "numba-cache")
    return kernels.batch_interpolate_velocity


def test_grid_arrays_feed_real_batch_interpolate_velocity(_original_batch_interp) -> None:
    """把 build_native_grid_arrays 的產出直接餵進原版 kernel，取得有限速度。"""

    mesh, dp, grid = _reference_grid()
    n_nodes = mesh.node_xy.shape[0]

    u1 = np.full((_NVRT, n_nodes), 0.30)
    v1 = np.full((_NVRT, n_nodes), -0.10)
    w1 = np.zeros((_NVRT, n_nodes))
    eta1 = np.full(n_nodes, _ETA_REFERENCE)
    u2 = np.full((_NVRT, n_nodes), 0.35)
    v2 = np.full((_NVRT, n_nodes), -0.05)
    w2 = np.zeros((_NVRT, n_nodes))

    # 三角形 0 的節點是 (0,0)-(100,0)-(100,100)，取形心並置於半水深處。
    px = np.array([200.0 / 3.0])
    py = np.array([100.0 / 3.0])
    pz = np.array([-5.0])
    p_elem = np.array([0], dtype=np.int64)
    p_status = np.array([1], dtype=np.int64)

    u_out = np.zeros(1)
    v_out = np.zeros(1)
    w_out = np.zeros(1)

    _original_batch_interp(
        px, py, pz, p_elem, p_status,
        grid.i34, grid.elnode, grid.grid_x, grid.grid_y,
        grid.nvrt, grid.kbp, grid.kbe, grid.sigma_lcl, grid.dp,
        u1, v1, w1, eta1,
        u2, v2, w2,
        0.5, 0.0,
        u_out, v_out, w_out,
    )

    assert np.all(np.isfinite(u_out))
    assert np.all(np.isfinite(v_out))
    assert np.all(np.isfinite(w_out))
    # 兩個時刻皆為常數場，trat=0.5 內插後應落在兩者之間。
    assert 0.30 <= float(u_out[0]) <= 0.35
    assert -0.10 <= float(v_out[0]) <= -0.05


# ── 6. 邊界情況：乾點、海底以下、缺值 ────────────────────────────────────


def test_rejects_node_with_no_finite_zcor_layer() -> None:
    """整柱缺值（例如全乾）必須明確拒絕，不得以零值或最近鄰頂替。"""

    mesh, dp = _build_regular_mesh()
    eta_reference = np.full(mesh.node_xy.shape[0], _ETA_REFERENCE, dtype=np.float64)
    zcor_reference = _synthetic_zcor(mesh, dp, _ETA_REFERENCE)
    zcor_reference[0, :] = np.nan

    with pytest.raises(NativeGridBridgeError, match="沒有任何有限 zcor 層"):
        build_native_grid_arrays(
            mesh, zcor_reference=zcor_reference, eta_reference=eta_reference
        )


def test_rejects_dry_point_with_nonpositive_depth_plus_eta() -> None:
    """水深加海面高程非正（乾點）必須拒絕，不能反推出無意義的 sigma。"""

    mesh, dp = _build_regular_mesh()
    eta_reference = np.full(mesh.node_xy.shape[0], _ETA_REFERENCE, dtype=np.float64)
    zcor_reference = _synthetic_zcor(mesh, dp, _ETA_REFERENCE)
    eta_reference[0] = -(dp[0] + 1.0)  # 讓 depth + eta < 0，模擬乾點

    with pytest.raises(NativeGridBridgeError, match="水深加海面高程非正"):
        build_native_grid_arrays(
            mesh, zcor_reference=zcor_reference, eta_reference=eta_reference
        )


def test_below_seabed_layers_are_clamped_not_zero_filled() -> None:
    """海床以下的 sigma 是複製 kbp 當層的值，不是零值，且與 kbp 定義一致。"""

    mesh, dp, grid = _reference_grid()
    for j in range(_NY):
        for i in range(_NX):
            idx = _node_index(i, j)
            kbp = _kbp_true(i, j)
            assert int(grid.kbp[idx]) == kbp
            if kbp > 0:
                # 海床以下每一層都應等於 kbp 當層的 sigma（clamp），不是 0。
                below = grid.sigma_lcl[:kbp, idx]
                at_kbp = grid.sigma_lcl[kbp, idx]
                assert np.all(below == at_kbp)
                assert at_kbp != 0.0


def test_rejects_shape_mismatch_between_mesh_and_zcor_reference() -> None:
    """zcor_reference 節點數與 mesh 不一致時必須拒絕，不得靜默截斷或補齊。"""

    mesh, dp = _build_regular_mesh()
    eta_reference = np.full(mesh.node_xy.shape[0], _ETA_REFERENCE, dtype=np.float64)
    zcor_reference = _synthetic_zcor(mesh, dp, _ETA_REFERENCE)[:-1, :]

    with pytest.raises(NativeGridBridgeError, match="節點數"):
        build_native_grid_arrays(
            mesh, zcor_reference=zcor_reference, eta_reference=eta_reference
        )
