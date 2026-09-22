"""WP-16 任務 2：融合子步迴圈（``native_runner.substep_numba``）的驗收測試。

涵蓋（見 ``substep_numba.py`` 模組文件「與原版行為的已知差異」一節逐項
編號）：

1. 不啟用擴散、不啟用關注區域判定時，與 ``mirror_rk4_substep_loop``
   逐位元相同（不涉及任何 RNG／Shapely，理論上必然相同，用來確認 RK4／
   邊界／clamp／內插的呼叫順序與參數完全沒有搬錯）。
2. SplitMix64 純量版本與向量化版本的整數部分逐位元相同。
3. Box-Muller（``log``／``cos``／``sin``）純量（Numba／``math``）與向量化
   （NumPy ufunc）版本**量化** ULP 級差異，不假設逐位元相同。
4. 射線法點在多邊形內測試與 Shapely ``contains_xy`` 的一致率量化。
5. 含擴散＋關注區域判定的完整子步迴圈，與 ``mirror_rk4_substep_loop``
   比較最終狀態、departures，確認差異只源自上面第 3 點的 ULP 差異、
   不會被放大成肉眼可見的軌跡偏差（統計等價）。
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from shapely import contains_xy
from shapely.geometry import Polygon

from lagrangian_backtracking.mesh import NativeMesh
from lagrangian_backtracking.native_runner import backward_driver as bd
from lagrangian_backtracking.native_runner import build_native_grid_arrays
from lagrangian_backtracking.native_runner.substep_numba import (
    _counter_based_gaussian_pair_scalar,
    _point_in_polygon_numba,
    _splitmix64_finalize_scalar,
    mirror_rk4_substep_loop_numba,
    prepare_interest_region_arrays,
)

_NX = 10
_NY = 10
_SPACING_M = 200.0
_NVRT = 3
_DP_M = 20.0
_DOMAIN_X = (_NX - 1) * _SPACING_M
_DOMAIN_Y = (_NY - 1) * _SPACING_M


def _node_index(i: int, j: int) -> int:
    return j * _NX + i


def _build_mesh() -> NativeMesh:
    n_nodes = _NX * _NY
    node_xy = np.zeros((n_nodes, 2), dtype=np.float64)
    for j in range(_NY):
        for i in range(_NX):
            node_xy[_node_index(i, j)] = (i * _SPACING_M, j * _SPACING_M)
    dp = np.full(n_nodes, _DP_M, dtype=np.float64)

    triangles: list[tuple[int, int, int]] = []
    for j in range(_NY - 1):
        for i in range(_NX - 1):
            n00 = _node_index(i, j)
            n10 = _node_index(i + 1, j)
            n11 = _node_index(i + 1, j + 1)
            n01 = _node_index(i, j + 1)
            triangles.append((n00, n10, n11))
            triangles.append((n00, n11, n01))
    n_faces = len(triangles)
    face_nodes_local = np.full((n_faces, 4), -1, dtype=np.int64)
    for face_index, nodes in enumerate(triangles):
        face_nodes_local[face_index, :3] = nodes
    face_node_count = np.full(n_faces, 3, dtype=np.int64)

    return NativeMesh(
        node_lon=np.linspace(120.0, 120.01, n_nodes),
        node_lat=np.linspace(24.0, 24.01, n_nodes),
        node_xy=node_xy,
        source_depth_m=dp,
        source_node_bottom_index=np.zeros(n_nodes, dtype=np.int64),
        face_nodes_local=face_nodes_local,
        face_node_count=face_node_count,
        source_face_global_index=np.arange(n_faces, dtype=np.int64),
        bin_size_m=400.0,
    )


def _resolve_baytrace_root() -> str:
    root = os.environ.get("LBT_BAYTRACE_ROOT")
    if not root:
        pytest.skip("需要 LBT_BAYTRACE_ROOT 指向 BayTrace 原始碼根目錄才能執行本測試")
    return root


@pytest.fixture(scope="module")
def handles() -> bd.BayTraceModuleHandles:
    return bd.load_baytrace_modules(_resolve_baytrace_root())


@pytest.fixture(scope="module")
def driver_grid(handles: bd.BayTraceModuleHandles) -> bd.NativeBackwardDriverGrid:
    mesh = _build_mesh()
    n_nodes = mesh.node_xy.shape[0]
    zcor_reference = np.zeros((n_nodes, _NVRT), dtype=np.float64)
    zcor_reference[:, 0] = -_DP_M
    zcor_reference[:, -1] = 0.0
    eta_reference = np.zeros(n_nodes, dtype=np.float64)
    grid = build_native_grid_arrays(mesh, zcor_reference=zcor_reference, eta_reference=eta_reference)
    built = bd.build_native_backward_driver_grid(grid, handles, h0=0.0)
    assert built.ic3_cross_check_ok
    return built


def _n_nodes(driver_grid: bd.NativeBackwardDriverGrid) -> int:
    return int(driver_grid.grid.grid_x.shape[0])


def _make_composed(driver_grid, *, u: float, v: float) -> bd.ComposedFieldPair:
    n_nodes = _n_nodes(driver_grid)
    nvrt = int(driver_grid.schism_grid.nvrt)
    shape = (nvrt, n_nodes)
    u_field = np.full(shape, u)
    v_field = np.full(shape, v)
    w_field = np.zeros(shape)
    eta = np.zeros(n_nodes)
    return bd.ComposedFieldPair(
        u_field1=u_field, v_field1=v_field, w_field1=w_field, eta_field1=eta,
        u_field2=u_field, v_field2=v_field, w_field2=w_field,
        node_is_dry=np.zeros(n_nodes, dtype=bool),
        elem_is_dry=np.zeros(driver_grid.grid.elnode.shape[0], dtype=bool),
        stokes_ok_fraction=1.0,
    )


def _make_pa(driver_grid, handles, *, x0, y0, z0, composed):
    n = x0.shape[0]
    pa = handles.particle_arrays_cls(n)
    pa.x[:] = x0
    pa.y[:] = y0
    pa.z[:] = z0
    pa.status[:] = 0
    not_found = bd.locate_initial_elements(
        driver_grid, handles, pa,
        eta_m=composed.eta_field1, u_current_mps=composed.u_field1,
        v_current_mps=composed.v_field1, w_current_mps=composed.w_field1,
    )
    assert not np.any(not_found)
    return pa


# ══════════════════════════════════════════════════════════════════════════
# 1. 不啟用擴散、不啟用關注區域：與 mirror_rk4_substep_loop 逐位元相同
# ══════════════════════════════════════════════════════════════════════════


def test_no_diffusion_bit_identical_to_python_loop(driver_grid, handles) -> None:
    composed = _make_composed(driver_grid, u=0.02, v=-0.01)
    rng = np.random.default_rng(1)
    x0 = rng.uniform(300.0, _DOMAIN_X - 300.0, size=6)
    y0 = rng.uniform(300.0, _DOMAIN_Y - 300.0, size=6)

    pa_py = _make_pa(driver_grid, handles, x0=x0, y0=y0, z0=-5.0, composed=composed)
    bd.mirror_rk4_substep_loop(
        driver_grid, handles, pa_py, ibf=-1, ndeltp=40, dtb=90.0, composed=composed, enable_diffusion=False,
    )

    pa_nb = _make_pa(driver_grid, handles, x0=x0, y0=y0, z0=-5.0, composed=composed)
    mirror_rk4_substep_loop_numba(
        driver_grid, handles, pa_nb, ibf=-1, ndeltp=40, dtb=90.0, composed=composed, enable_diffusion=False,
    )

    assert np.array_equal(pa_py.x, pa_nb.x)
    assert np.array_equal(pa_py.y, pa_nb.y)
    assert np.array_equal(pa_py.z, pa_nb.z)
    assert np.array_equal(pa_py.status, pa_nb.status)
    assert np.array_equal(pa_py.element, pa_nb.element)


# ══════════════════════════════════════════════════════════════════════════
# 2-3. SplitMix64／Box-Muller 純量與向量化版本的一致性量化
# ══════════════════════════════════════════════════════════════════════════


def test_splitmix64_scalar_matches_vectorized_bit_identical() -> None:
    """整數運算（模 2**64 環域）沒有任何浮點求和順序或 libm 後端差異的
    餘地，純量與向量化版本必須逐位元相同（無例外，見模組文件差異第 1 點）。
    """

    from numba import njit

    @njit(cache=True)
    def _scalar_batch(state, out):
        for i in range(state.shape[0]):
            out[i] = _splitmix64_finalize_scalar(state[i])

    rng = np.random.default_rng(2)
    n = 50_000
    state = rng.integers(0, 2**63, size=n, dtype=np.int64).astype(np.uint64)
    vec_result = bd._splitmix64_finalize(state.copy())
    scalar_result = np.empty(n, dtype=np.uint64)
    _scalar_batch(state, scalar_result)
    assert np.array_equal(vec_result, scalar_result)


def test_box_muller_scalar_vs_vectorized_ulp_difference_quantified() -> None:
    """量化 Numba／``math``（純量）與 NumPy ufunc（向量化）算出的高斯亂數
    差異，不假設逐位元相同（見模組文件差異第 2 點）——這個測試本身**不會
    因為出現差異而失敗**，只在差異超出「1-2 個 float64 ulp」這個誠實
    預期時才失敗，避免真正的邏輯錯誤被誤判成「這本來就會有的 ULP 差異」
    而被掩蓋過去。
    """

    from numba import njit

    @njit(cache=True)
    def _scalar_batch(seed_lo, seed_hi, counter, z0_out, z1_out):
        for i in range(seed_lo.shape[0]):
            z0, z1 = _counter_based_gaussian_pair_scalar(seed_lo[i], seed_hi[i], np.uint64(counter[i]))
            z0_out[i] = z0
            z1_out[i] = z1

    n = 200_000
    seed_hex = [f"{i:032x}" for i in range(n)]
    seed_lo, seed_hi = bd.derive_particle_diffusion_seed_lo_hi_u64(seed_hex, master_seed=123)
    rng = np.random.default_rng(3)
    counter = rng.integers(0, 5000, size=n).astype(np.int64)

    z0_vec, z1_vec = bd.counter_based_gaussian_pairs(seed_lo, seed_hi, counter)
    z0_sc = np.empty(n)
    z1_sc = np.empty(n)
    _scalar_batch(seed_lo, seed_hi, counter, z0_sc, z1_sc)

    max_abs_diff = max(np.max(np.abs(z0_vec - z0_sc)), np.max(np.abs(z1_vec - z1_sc)))
    n_mismatch = int(np.count_nonzero(z0_vec != z0_sc)) + int(np.count_nonzero(z1_vec != z1_sc))
    mismatch_ratio = n_mismatch / (2 * n)

    # float64 machine epsilon 約 2.22e-16；1-2 個 ulp 級差異上界約 5e-16，
    # 若超過這個量級代表可能是邏輯錯誤而不是求和順序／libm 後端差異。
    assert max_abs_diff < 5.0e-15, max_abs_diff
    assert mismatch_ratio < 0.02, mismatch_ratio


# ══════════════════════════════════════════════════════════════════════════
# 4. 射線法 vs Shapely contains_xy 一致率量化
# ══════════════════════════════════════════════════════════════════════════


def test_point_in_polygon_matches_shapely_away_from_boundary() -> None:
    poly = Polygon([(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)])
    region = prepare_interest_region_arrays(poly)
    poly_x = region.poly_x[:, 0]
    poly_y = region.poly_y[:, 0]
    nvert = int(region.poly_nvert[0])

    rng = np.random.default_rng(4)
    n = 20_000
    xs = rng.uniform(-20.0, 120.0, size=n)
    ys = rng.uniform(-20.0, 120.0, size=n)
    shapely_result = contains_xy(poly, xs, ys)
    numba_result = np.array(
        [_point_in_polygon_numba(x, y, poly_x, poly_y, nvert) for x, y in zip(xs, ys, strict=True)]
    )
    assert np.array_equal(shapely_result, numba_result)


def test_point_in_polygon_boundary_mismatch_is_only_on_exact_boundary() -> None:
    """在邊界**附近**（含浮點捨入級的微小擾動）兩者必須完全一致；只有
    座標**恰好**落在邊界上時，兩種演算法的邊界收斂慣例才可能不同——這正是
    模組文件差異第 3 點聲明的範圍，這裡直接量化上界，不留模糊空間。
    """

    poly = Polygon([(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)])
    region = prepare_interest_region_arrays(poly)
    poly_x = region.poly_x[:, 0]
    poly_y = region.poly_y[:, 0]
    nvert = int(region.poly_nvert[0])

    rng = np.random.default_rng(5)
    n = 20_000
    edge = rng.integers(0, 4, size=n)
    t = rng.uniform(0.0, 100.0, size=n)
    eps = rng.uniform(-1.0e-9, 1.0e-9, size=n)
    bx = np.zeros(n)
    by = np.zeros(n)
    for i in range(n):
        e = edge[i]
        if e == 0:
            bx[i], by[i] = t[i], 0.0 + eps[i]
        elif e == 1:
            bx[i], by[i] = 100.0 + eps[i], t[i]
        elif e == 2:
            bx[i], by[i] = t[i], 100.0 + eps[i]
        else:
            bx[i], by[i] = 0.0 + eps[i], t[i]

    shapely_result = contains_xy(poly, bx, by)
    numba_result = np.array(
        [_point_in_polygon_numba(x, y, poly_x, poly_y, nvert) for x, y in zip(bx, by, strict=True)]
    )
    # 有微小（非零）浮點擾動時，點實際上不在邊界「恰好」上，兩種演算法
    # 應該完全一致；真正的分歧只在擾動剛好是 0（機率為零的事件，這裡的
    # 隨機擾動幾乎必然非零）。
    assert np.array_equal(shapely_result, numba_result)


# ══════════════════════════════════════════════════════════════════════════
# 5. 含擴散＋關注區域：與 mirror_rk4_substep_loop 統計等價
# ══════════════════════════════════════════════════════════════════════════


def test_diffusion_and_interest_region_matches_python_loop_statistically(driver_grid, handles) -> None:
    composed = _make_composed(driver_grid, u=0.02, v=-0.01)
    n_nodes = _n_nodes(driver_grid)
    nvrt = int(driver_grid.schism_grid.nvrt)
    kh_field = np.full((nvrt, n_nodes), 1.0e-4)
    poly = Polygon(
        [
            (400.0, 400.0),
            (_DOMAIN_X - 400.0, 400.0),
            (_DOMAIN_X - 400.0, _DOMAIN_Y - 400.0),
            (400.0, _DOMAIN_Y - 400.0),
        ]
    )

    n_particles = 150
    rng = np.random.default_rng(6)
    x0 = rng.uniform(200.0, _DOMAIN_X - 200.0, size=n_particles)
    y0 = rng.uniform(200.0, _DOMAIN_Y - 200.0, size=n_particles)
    seed_hex = [f"{i:032x}" for i in range(n_particles)]
    seed_lo, seed_hi = bd.derive_particle_diffusion_seed_lo_hi_u64(seed_hex, master_seed=42)

    ndeltp = 300
    dtb = 3600.0 / ndeltp
    hour_start = 1_700_000_000 * 1_000_000_000

    def substep_utc(idt: int, _start: int = hour_start, _dtb: float = dtb) -> int:
        return _start - int(round(idt * _dtb * 1.0e9))

    pa_py = _make_pa(driver_grid, handles, x0=x0, y0=y0, z0=-5.0, composed=composed)
    counter_py = np.zeros(n_particles, dtype=np.int64)
    native_status_py = np.zeros(n_particles, dtype=np.int32)
    departures_py: list[bd.DepartureRecord] = []
    bd.mirror_rk4_substep_loop(
        driver_grid, handles, pa_py, ibf=-1, ndeltp=ndeltp, dtb=dtb, composed=composed, enable_diffusion=True,
        hf1_field=kh_field, hf2_field=kh_field,
        particle_seed_lo=seed_lo, particle_seed_hi=seed_hi, particle_step_counter=counter_py,
        native_status=native_status_py, substep_utc_ns=substep_utc,
        interest_region_polygon=poly, departures=departures_py,
    )

    pa_nb = _make_pa(driver_grid, handles, x0=x0, y0=y0, z0=-5.0, composed=composed)
    counter_nb = np.zeros(n_particles, dtype=np.int64)
    native_status_nb = np.zeros(n_particles, dtype=np.int32)
    departures_nb: list[bd.DepartureRecord] = []
    mirror_rk4_substep_loop_numba(
        driver_grid, handles, pa_nb, ibf=-1, ndeltp=ndeltp, dtb=dtb, composed=composed, enable_diffusion=True,
        hf1_field=kh_field, hf2_field=kh_field,
        particle_seed_lo=seed_lo, particle_seed_hi=seed_hi, particle_step_counter=counter_nb,
        native_status=native_status_nb, substep_utc_ns=substep_utc,
        interest_region_polygon=poly, departures=departures_nb,
    )

    # 逐步累積的擴散位移經過 300 個子步後，可能因為極少數子步的 1-ulp
    # Box-Muller 差異被 RK4 的非線性平流路徑放大，但幅度應該遠小於粒子
    # 實際位移量級（見模組文件差異第 2 點）；這裡用寬鬆但有意義的門檻。
    displacement = np.hypot(pa_py.x - x0, pa_py.y - y0)
    position_diff = np.hypot(pa_py.x - pa_nb.x, pa_py.y - pa_nb.y)
    assert np.all(position_diff < 1.0e-2 * np.maximum(displacement, 1.0)), (
        position_diff.max(), displacement.max()
    )

    # departures 的「哪些粒子在哪個子步離開」應該幾乎完全一致（極少數可能
    # 因為 1-ulp 差異剛好落在關注區域邊界附近的子步差 1 個 idt，屬預期
    # 範圍，這裡只要求絕大多數一致）。
    dep_py_rows = {d.row_particle_index for d in departures_py}
    dep_nb_rows = {d.row_particle_index for d in departures_nb}
    mismatch = dep_py_rows.symmetric_difference(dep_nb_rows)
    assert len(mismatch) <= max(1, int(0.02 * n_particles)), (
        mismatch, len(departures_py), len(departures_nb),
    )


def test_single_polygon_broadcasts_to_all_particles(driver_grid, handles) -> None:
    """WP-16 開發期間踩過的臭蟲回歸測試：``interest_region_polygon``
    （單一多邊形套用到全部粒子）的 ``poly_id`` 必須廣播成 ``(n,)``，不能
    停留在 ``prepare_interest_region_arrays`` 回傳的 ``(1,)``——否則粒子
    數大於 1 時，kernel 內對 ``poly_id[ip]``（``ip`` 可以到 ``n-1``）的
    存取會超出陣列邊界（在 Numba 編譯碼裡不會拋 Python 例外，而是直接
    讀到不相關記憶體或造成 segfault，見最終報告開發過程記錄）。
    """

    composed = _make_composed(driver_grid, u=0.0, v=0.0)
    poly = Polygon([(-1.0e6, -1.0e6), (1.0e6, -1.0e6), (1.0e6, 1.0e6), (-1.0e6, 1.0e6)])
    n_particles = 37  # 遠大於 1，才能真正測出廣播是否正確
    rng = np.random.default_rng(7)
    x0 = rng.uniform(300.0, _DOMAIN_X - 300.0, size=n_particles)
    y0 = rng.uniform(300.0, _DOMAIN_Y - 300.0, size=n_particles)
    pa = _make_pa(driver_grid, handles, x0=x0, y0=y0, z0=-5.0, composed=composed)

    native_status = np.zeros(n_particles, dtype=np.int32)
    departures: list[bd.DepartureRecord] = []

    def substep_utc(idt: int) -> int:
        return 0

    mirror_rk4_substep_loop_numba(
        driver_grid, handles, pa, ibf=-1, ndeltp=5, dtb=10.0, composed=composed, enable_diffusion=False,
        native_status=native_status, substep_utc_ns=substep_utc,
        interest_region_polygon=poly, departures=departures,
    )
    # 大多邊形涵蓋整個網格，且流速為 0（粒子不會移動出網格），不應該有
    # 任何粒子離開關注區域；這個斷言本身不是重點，重點是呼叫過程沒有
    # segfault（pytest 行程存活就代表沒有越界存取）。
    assert len(departures) == 0
    assert np.all(native_status == bd.NATIVE_STATUS_ACTIVE)


def test_legacy_rng_mode_not_supported(driver_grid, handles) -> None:
    """``mirror_rk4_substep_loop_numba`` 沒有 ``rng`` 參數（不支援 legacy
    共用串流模式，見模組文件差異第 4 點）；啟用擴散時若沒有提供計數器型
    三個參數，必須明確拒絕，不能靜默用零種子跑出一組沒有意義的擴散序列。
    """

    composed = _make_composed(driver_grid, u=0.0, v=0.0)
    pa = _make_pa(
        driver_grid, handles, x0=np.array([500.0]), y0=np.array([500.0]), z0=-5.0, composed=composed,
    )
    n_nodes = _n_nodes(driver_grid)
    nvrt = int(driver_grid.schism_grid.nvrt)
    kh_field = np.full((nvrt, n_nodes), 1.0e-4)
    with pytest.raises(bd.BackwardDriverError, match="particle_seed_lo"):
        mirror_rk4_substep_loop_numba(
            driver_grid, handles, pa, ibf=-1, ndeltp=5, dtb=10.0, composed=composed, enable_diffusion=True,
            hf1_field=kh_field, hf2_field=kh_field,
        )
