"""WP-14 任務 C：``smagorinsky_batch``（Numba 版 Smagorinsky 節點平均）的
驗收測試。

涵蓋：純 Python 參考核心與 Numba 核心互相一致（同一演算法、只差有沒有經過
Numba 編譯）、兩者與 BayTrace 原版（透過 ``backward_driver.
compute_smagorinsky_kh_field``）在多種網格規模與流速情境下比對，**誠實
記錄比對結果**——見模組 docstring「位元相同性」一節，本模組事先聲明
不假設兩者位元相同，測試的目的是量測差異、不是強行通過。含全零流速
（strain 恆為 0，理論上必然逐位元相同）、邊界節點（``nne`` 較小）、退化
（零面積）元素等邊界情況。
"""

from __future__ import annotations

import numpy as np
import pytest

from lagrangian_backtracking.mesh import NativeMesh
from lagrangian_backtracking.native_runner import backward_driver as bd
from lagrangian_backtracking.native_runner import build_native_grid_arrays
from lagrangian_backtracking.native_runner.smagorinsky_batch import (
    SmagorinskyGridArrays,
    compute_smagorinsky_kh_field_numba,
    compute_smagorinsky_kh_field_reference,
    prepare_smagorinsky_grid,
)


def _resolve_baytrace_root() -> str:
    import os

    root = os.environ.get("LBT_BAYTRACE_ROOT")
    if not root:
        pytest.skip("需要 LBT_BAYTRACE_ROOT 指向 BayTrace 原始碼根目錄才能執行本測試")
    return root


@pytest.fixture(scope="module")
def handles() -> bd.BayTraceModuleHandles:
    return bd.load_baytrace_modules(_resolve_baytrace_root())


def _build_regular_grid_mesh(nx: int, ny: int, *, spacing_m: float, dp: np.ndarray) -> NativeMesh:
    n_nodes = nx * ny
    node_xy = np.zeros((n_nodes, 2), dtype=np.float64)
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy")
    node_xy[:, 0] = (ii * spacing_m).ravel()
    node_xy[:, 1] = (jj * spacing_m).ravel()

    idx = np.arange(n_nodes).reshape(ny, nx)
    n00 = idx[:-1, :-1].ravel()
    n10 = idx[:-1, 1:].ravel()
    n11 = idx[1:, 1:].ravel()
    n01 = idx[1:, :-1].ravel()
    triangles = np.concatenate(
        [np.stack([n00, n10, n11], axis=1), np.stack([n00, n11, n01], axis=1)], axis=0
    )
    n_faces = triangles.shape[0]
    face_nodes_local = np.full((n_faces, 4), -1, dtype=np.int64)
    face_nodes_local[:, :3] = triangles
    face_node_count = np.full(n_faces, 3, dtype=np.int64)

    return NativeMesh(
        node_lon=np.linspace(120.0, 120.5, n_nodes),
        node_lat=np.linspace(24.0, 24.5, n_nodes),
        node_xy=node_xy,
        source_depth_m=dp,
        source_node_bottom_index=np.zeros(n_nodes, dtype=np.int64),
        face_nodes_local=face_nodes_local,
        face_node_count=face_node_count,
        source_face_global_index=np.arange(n_faces, dtype=np.int64),
        bin_size_m=400.0,
    )


def _build_driver_grid(handles: bd.BayTraceModuleHandles, *, nx: int, ny: int, nvrt: int, spacing_m: float):
    rng = np.random.default_rng(20260921)
    n_nodes = nx * ny
    dp = np.full(n_nodes, 30.0) * rng.uniform(0.3, 2.0, size=n_nodes)
    mesh = _build_regular_grid_mesh(nx, ny, spacing_m=spacing_m, dp=dp)
    sigma_1d = np.linspace(-1.0, 0.0, nvrt)
    zcor_reference = sigma_1d[None, :] * dp[:, None]
    eta_reference = np.zeros(n_nodes, dtype=np.float64)
    grid = build_native_grid_arrays(mesh, zcor_reference=zcor_reference, eta_reference=eta_reference)
    driver_grid = bd.build_native_backward_driver_grid(grid, handles, h0=0.0)
    assert driver_grid.ic3_cross_check_ok
    return driver_grid


def _grid_arrays(driver_grid: bd.NativeBackwardDriverGrid) -> SmagorinskyGridArrays:
    g = driver_grid.schism_grid
    return SmagorinskyGridArrays(
        elnode=g.elnode, area=g.area, dldxy=g.dldxy, nne=g.nne, indel=g.indel,
        x=g.x, y=g.y, xctr=g.xctr, yctr=g.yctr,
    )


# ── 1. 純 Python 參考核心與 Numba 核心：同一演算法，必須逐位元相同 ─────────


def test_reference_and_numba_agree_bit_identical(handles) -> None:
    """參考核心與 Numba 核心是同一套運算式與求和順序的兩種實作，兩者之間
    （不涉及 BayTrace 原版／NumPy 內積的求和順序差異）必須逐位元相同。
    """

    driver_grid = _build_driver_grid(handles, nx=10, ny=10, nvrt=6, spacing_m=150.0)
    grid_arrays = _grid_arrays(driver_grid)
    n_nodes = driver_grid.grid.grid_x.shape[0]
    rng = np.random.default_rng(1)
    shape = (6, n_nodes)
    u2 = rng.uniform(-0.3, 0.3, size=shape)
    v2 = rng.uniform(-0.3, 0.3, size=shape)

    ref = compute_smagorinsky_kh_field_reference(grid_arrays, u_field2=u2, v_field2=v2, horcon=0.2)
    numba = compute_smagorinsky_kh_field_numba(grid_arrays, u_field2=u2, v_field2=v2, horcon=0.2)

    assert np.array_equal(ref, numba)


def test_all_zero_velocity_matches_baytrace_exactly(driver_grid_small, handles) -> None:
    """全零流速時應變率恆為 0，內積求和順序的差異不影響結果（全部項都是
    0），這是唯一保證與 BayTrace 原版逐位元相同的情境——用來確認差異
    確實只來自浮點求和順序，而不是本模組某處算錯了公式。
    """

    grid_arrays = _grid_arrays(driver_grid_small)
    n_nodes = driver_grid_small.grid.grid_x.shape[0]
    nvrt = int(driver_grid_small.schism_grid.nvrt)
    shape = (nvrt, n_nodes)
    u2 = np.zeros(shape)
    v2 = np.zeros(shape)

    baytrace_result = bd.compute_smagorinsky_kh_field(
        driver_grid_small, handles, u_current_mps=u2, v_current_mps=v2, horcon=0.2
    )
    numba_result = compute_smagorinsky_kh_field_numba(grid_arrays, u_field2=u2, v_field2=v2, horcon=0.2)

    assert np.array_equal(baytrace_result, numba_result)


@pytest.fixture
def driver_grid_small(handles):
    return _build_driver_grid(handles, nx=10, ny=10, nvrt=6, spacing_m=150.0)


# ── 2. 與 BayTrace 原版比對：誠實記錄差異，不強行斷言位元相同 ─────────────


@pytest.mark.parametrize(
    ("nx", "ny", "nvrt", "spacing_m", "velocity_scale", "label"),
    [
        (10, 10, 6, 150.0, 0.3, "規則網格-中等流速"),
        (24, 24, 12, 100.0, 0.3, "規則網格-較大規模"),
        (10, 10, 6, 150.0, 5.0, "規則網格-極端大流速"),
    ],
)
def test_numba_vs_baytrace_reports_honest_diff(
    handles, nx, ny, nvrt, spacing_m, velocity_scale, label, capsys
) -> None:
    """與 BayTrace 原版逐點比對，把最大絕對／相對誤差與不相符點數印出來，
    供最終報告引用；**刻意不斷言 ``np.array_equal``**（見模組 docstring：
    NumPy 內積的求和順序與本模組手寫序列求和不保證逐位元相同），改成
    「差異必須落在機器精度等級（相對誤差 <= 1e-12）」這個遠比正確性寬鬆、
    只用來攔截「公式根本算錯」這種重大回歸的上界。
    """

    driver_grid = _build_driver_grid(handles, nx=nx, ny=ny, nvrt=nvrt, spacing_m=spacing_m)
    grid_arrays = _grid_arrays(driver_grid)
    n_nodes = driver_grid.grid.grid_x.shape[0]
    rng = np.random.default_rng(7)
    shape = (nvrt, n_nodes)
    u2 = rng.uniform(-velocity_scale, velocity_scale, size=shape)
    v2 = rng.uniform(-velocity_scale, velocity_scale, size=shape)

    baytrace_result = bd.compute_smagorinsky_kh_field(
        driver_grid, handles, u_current_mps=u2, v_current_mps=v2, horcon=0.2
    )
    numba_result = compute_smagorinsky_kh_field_numba(grid_arrays, u_field2=u2, v_field2=v2, horcon=0.2)

    exactly_equal = np.array_equal(baytrace_result, numba_result)
    diff = np.abs(baytrace_result - numba_result)
    denom = np.maximum(np.abs(baytrace_result), 1e-300)
    max_rel_err = float(np.max(diff / denom))
    max_abs_err = float(np.max(diff))
    n_mismatch = int(np.count_nonzero(diff != 0.0))
    n_total = int(baytrace_result.size)

    print(
        f"\n[smagorinsky_batch] {label}：np.array_equal={exactly_equal}，"
        f"max_abs_err={max_abs_err:.3e}，max_rel_err={max_rel_err:.3e}，"
        f"不相符點數={n_mismatch}/{n_total} ({100.0 * n_mismatch / n_total:.4f}%)"
    )

    # 只攔截「公式錯了」這種數量級的重大回歸；機器精度等級的差異（已知
    # 來自 NumPy 內積與手寫序列求和的求和順序不同）是預期內、可接受的。
    assert max_rel_err <= 1.0e-12, (
        f"{label}：與 BayTrace 原版差異超出機器精度等級（{max_rel_err:.3e}），"
        "可能是公式或索引寫錯，不是單純的浮點求和順序差異"
    )


# ── 3. 邊界情況：邊界節點（nne 較小）、退化（零面積）元素 ─────────────────


def test_boundary_nodes_with_few_neighbors_do_not_crash_or_produce_nan(handles) -> None:
    """規則網格的角落／邊界節點 ``nne`` 明顯小於內部節點（內部通常是 6），
    確認這些節點仍然正確算出有限值，不是 NaN 或者被跳過。
    """

    driver_grid = _build_driver_grid(handles, nx=8, ny=8, nvrt=4, spacing_m=200.0)
    grid_arrays = _grid_arrays(driver_grid)
    n_nodes = driver_grid.grid.grid_x.shape[0]
    rng = np.random.default_rng(3)
    shape = (4, n_nodes)
    u2 = rng.uniform(-0.2, 0.2, size=shape)
    v2 = rng.uniform(-0.2, 0.2, size=shape)

    numba_result = compute_smagorinsky_kh_field_numba(grid_arrays, u_field2=u2, v_field2=v2, horcon=0.2)

    assert np.all(np.isfinite(numba_result))
    # 角落節點的 nne 應該明顯小於內部節點（規則網格內部是 6），確認測試
    # 真的涵蓋到邊界情況，而不是全部節點鄰接數都一樣。
    nne = grid_arrays.nne
    assert nne.min() < nne.max()


def test_degenerate_zero_area_element_does_not_poison_neighbors(handles) -> None:
    """人為把一個元素的面積與形函數導數歸零（模擬退化三角形），確認只有
    直接依賴它的節點數值改變，其餘節點不受影響、且不產生 NaN／Inf。
    """

    driver_grid = _build_driver_grid(handles, nx=8, ny=8, nvrt=3, spacing_m=200.0)
    grid_arrays = _grid_arrays(driver_grid)
    n_nodes = driver_grid.grid.grid_x.shape[0]
    rng = np.random.default_rng(9)
    shape = (3, n_nodes)
    u2 = rng.uniform(-0.2, 0.2, size=shape)
    v2 = rng.uniform(-0.2, 0.2, size=shape)

    baseline = compute_smagorinsky_kh_field_numba(grid_arrays, u_field2=u2, v_field2=v2, horcon=0.2)

    degenerate_area = grid_arrays.area.copy()
    degenerate_dldxy = grid_arrays.dldxy.copy()
    target_elem = 0
    degenerate_area[target_elem] = 0.0
    degenerate_dldxy[:, :, target_elem] = 0.0
    degenerate_grid = SmagorinskyGridArrays(
        elnode=grid_arrays.elnode, area=degenerate_area, dldxy=degenerate_dldxy,
        nne=grid_arrays.nne, indel=grid_arrays.indel, x=grid_arrays.x, y=grid_arrays.y,
        xctr=grid_arrays.xctr, yctr=grid_arrays.yctr,
    )
    degenerate_result = compute_smagorinsky_kh_field_numba(
        degenerate_grid, u_field2=u2, v_field2=v2, horcon=0.2
    )

    assert np.all(np.isfinite(degenerate_result))

    # 找出哪些節點鄰接到退化元素，只有這些節點的值應該改變。
    affected_nodes = set()
    nne = grid_arrays.nne
    indel = grid_arrays.indel
    for i in range(n_nodes):
        nn = int(nne[i])
        if target_elem in indel[:nn, i].tolist():
            affected_nodes.add(i)
    assert affected_nodes, "測試前提：這個網格應該至少有節點鄰接到元素 0"

    unaffected = np.array([i for i in range(n_nodes) if i not in affected_nodes])
    assert np.array_equal(baseline[:, unaffected], degenerate_result[:, unaffected])


# ── 4. WP-16 任務 3：迴圈重排＋權重快取必須與「不快取」路徑逐位元相同 ──────


def test_prepared_grid_cache_matches_uncached_bit_identical(handles) -> None:
    """``prepared=None``（每次呼叫都重算 ascontiguousarray 轉換＋反距離權重，
    WP-16 任務 3 新增這個參數前的唯一行為）與``prepared=prepare_smagorinsky_
    grid(...)``（只算一次、重複傳入）必須逐位元相同——這是任務 3「快取只是
    把同一組運算搬到只呼叫一次，不改變任何數值」的直接證明，兩種路徑內部
    都是同一個外層 k、內層 i／e 的迴圈重排版本（見
    ``test_reference_and_numba_agree_bit_identical`` 對純 Python 參考核心
    的獨立交叉驗證，涵蓋迴圈重排本身的正確性）。多組規模、多組隨機流速，
    降低「剛好在某個簡單案例碰巧一樣」的疑慮。
    """

    for nx, ny, nvrt, spacing_m in ((10, 10, 6, 150.0), (24, 24, 12, 100.0), (5, 5, 3, 80.0)):
        driver_grid = _build_driver_grid(handles, nx=nx, ny=ny, nvrt=nvrt, spacing_m=spacing_m)
        grid_arrays = _grid_arrays(driver_grid)
        n_nodes = driver_grid.grid.grid_x.shape[0]
        rng = np.random.default_rng(nx * 1000 + ny)
        shape = (nvrt, n_nodes)
        u2 = rng.uniform(-0.5, 0.5, size=shape)
        v2 = rng.uniform(-0.5, 0.5, size=shape)

        prepared = prepare_smagorinsky_grid(grid_arrays)
        uncached = compute_smagorinsky_kh_field_numba(grid_arrays, u_field2=u2, v_field2=v2, horcon=0.2)
        cached = compute_smagorinsky_kh_field_numba(
            grid_arrays, u_field2=u2, v_field2=v2, horcon=0.2, prepared=prepared
        )
        assert np.array_equal(uncached, cached), (nx, ny, nvrt)

        # 同一份 prepared 重複用在兩組不同流速上，彼此獨立、不互相污染
        # （prepared 只含網格拓樸資訊，不含任何流場相關的中間狀態）。
        u2b = rng.uniform(-0.5, 0.5, size=shape)
        v2b = rng.uniform(-0.5, 0.5, size=shape)
        cached_b = compute_smagorinsky_kh_field_numba(
            grid_arrays, u_field2=u2b, v_field2=v2b, horcon=0.2, prepared=prepared
        )
        uncached_b = compute_smagorinsky_kh_field_numba(grid_arrays, u_field2=u2b, v_field2=v2b, horcon=0.2)
        assert np.array_equal(uncached_b, cached_b), (nx, ny, nvrt, "second call")
        assert not np.array_equal(cached, cached_b), "測試前提：兩組流速應該給出不同結果"


def test_prepared_grid_weights_independent_of_horcon_and_velocity(handles) -> None:
    """``prepare_smagorinsky_grid`` 算出的 ``weights``／``slm`` 只跟網格拓樸
    有關，換 ``horcon``、換流速都不應該影響它們——這是「可以在網格建立時
    算一次、跨全部小時與全部沉降分組重複使用」這個快取假設成立的前提。
    """

    driver_grid = _build_driver_grid(handles, nx=10, ny=10, nvrt=4, spacing_m=150.0)
    grid_arrays = _grid_arrays(driver_grid)
    prepared_a = prepare_smagorinsky_grid(grid_arrays)
    prepared_b = prepare_smagorinsky_grid(grid_arrays)
    assert np.array_equal(prepared_a.weights, prepared_b.weights)
    assert np.array_equal(prepared_a.slm, prepared_b.slm)
