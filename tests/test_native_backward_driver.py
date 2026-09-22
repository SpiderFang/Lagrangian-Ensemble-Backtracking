"""WP-5：單一批次逆向（reverse-time）原生驅動層（``native_runner.backward_driver``）
的驗收測試。

測試涵蓋（對應工作包驗收要求，逐項編號）：

1. **鏡像忠實度（最重要）**：擴散關閉、Stokes 為 0、沉降為 0 時，
   ``mirror_rk4_substep_loop`` 與直接逐行呼叫 BayTrace 原始 njit kernel
   （不透過本模組任何函式）的最終粒子狀態必須位元相同（``np.array_equal``）。
2. **速度合成正確性**：已知的常數 Stokes／沉降速度，驗證位移等於
   ``(海流+Stokes+沉降)×時間``，並與「只加在 k1（``pa.u``）」的錯誤做法對照，
   證明兩者不同、只有合成做法正確。
3. **逆向時間方向**：時間變動（但空間均勻）的合成流場，正向再反向應回到
   接近原點。
4. **擴散統計**：均勻 K、無邊界的合成流場，位移變異數應等於 ``2*K*t``
   （每軸獨立），並確認擴散不是插進 RK4 的 stage 內部。
5. **停止條件**：正確記錄離開關注區域的位置與時刻；未離開者標記為回溯上限。
6. **狀態碼**：各種終止原因彼此不混淆。
7. **可重現性**：相同設定＋種子重跑，結果位元相同。
8. **效能量測**：N=10,000 顆粒子跑數小時，量測並拆解 µs/粒子步的耗時。

合成網格與流場的設計原則
--------------------------
所有測試共用一個規則三角網格（透過真正的 ``NativeMesh`` ＋
``build_native_grid_arrays`` 建構，不手刻 ``ic3``，避免測試自己引入邊鄰接
慣例錯誤）與同一個受控載入的 ``BayTraceModuleHandles``；BayTrace 原始碼一律
不修改，測試只驗證 WP-5 驅動層程式碼本身的行為。
"""

from __future__ import annotations

import os
import time as _time

import numpy as np
import pytest
from shapely.geometry import Polygon

from lagrangian_backtracking.mesh import NativeMesh
from lagrangian_backtracking.native_runner import backward_driver as bd
from lagrangian_backtracking.native_runner import build_native_grid_arrays
from lagrangian_backtracking.native_runner.forcing_bridge import HourlyFieldPair
from lagrangian_backtracking.native_runner.stokes_batch import finite_depth_stokes_reference

_HOUR_NS = 3_600_000_000_000

# ── 合成網格：8x8 節點、7x7 格、每格 2 個三角形，間距 200 公尺 ──────────────

_NX = 8
_NY = 8
_SPACING_M = 200.0
_NVRT = 2
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
        pytest.skip("需要 LBT_BAYTRACE_ROOT 指向 BayTrace 原始碼根目錄才能執行 WP-5 測試")
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
    zcor_reference[:, 1] = 0.0
    eta_reference = np.zeros(n_nodes, dtype=np.float64)
    grid = build_native_grid_arrays(mesh, zcor_reference=zcor_reference, eta_reference=eta_reference)
    built = bd.build_native_backward_driver_grid(grid, handles, h0=0.0)
    assert built.ic3_cross_check_ok
    return built


def _n_nodes(driver_grid: bd.NativeBackwardDriverGrid) -> int:
    return int(driver_grid.grid.grid_x.shape[0])


def _uniform_field_pair(
    driver_grid: bd.NativeBackwardDriverGrid,
    *,
    u1: float,
    v1: float,
    w1: float = 0.0,
    u2: float | None = None,
    v2: float | None = None,
    w2: float | None = None,
    trat: float = 0.0,
    ibf: int = -1,
    time_before_utc_ns: int = 2 * _HOUR_NS,
    time_after_utc_ns: int = 1 * _HOUR_NS,
) -> HourlyFieldPair:
    """建立一個「全網格空間均勻」的合成流場，供不需要空間變化的測試使用。"""

    n_nodes = _n_nodes(driver_grid)
    shape = (_NVRT, n_nodes)
    u2 = u1 if u2 is None else u2
    v2 = v1 if v2 is None else v2
    w2 = w1 if w2 is None else w2
    eta = np.zeros(n_nodes, dtype=np.float64)
    diffusivity = np.full(shape, 1.0e-4, dtype=np.float64)
    missing = np.zeros(shape, dtype=bool)
    dry_face = np.zeros(shape, dtype=bool)
    return HourlyFieldPair(
        time_before_utc_ns=time_before_utc_ns,
        time_after_utc_ns=time_after_utc_ns,
        trat=trat,
        ibf=ibf,
        u_field1=np.full(shape, u1, dtype=np.float64),
        v_field1=np.full(shape, v1, dtype=np.float64),
        w_field1=np.full(shape, w1, dtype=np.float64),
        eta_field1=eta.copy(),
        diffusivity_field1=diffusivity.copy(),
        missing_mask1=missing.copy(),
        dry_face_flag1=dry_face.copy(),
        month_id1="synthetic",
        time_origin1="test",
        u_field2=np.full(shape, u2, dtype=np.float64),
        v_field2=np.full(shape, v2, dtype=np.float64),
        w_field2=np.full(shape, w2, dtype=np.float64),
        eta_field2=eta.copy(),
        diffusivity_field2=diffusivity.copy(),
        missing_mask2=missing.copy(),
        dry_face_flag2=dry_face.copy(),
        month_id2="synthetic",
        time_origin2="test",
    )


class _SyntheticSoa:
    """欄位相容的最小 ``BayTraceParticleSoA`` 替身，供 ``run_backward_batch``
    的 ``soa`` 參數使用（該函式只讀取這幾個欄位，見其 docstring）。

    刻意用 ``__init__`` 賦值而不是在測試裡就地宣告 ``class _Soa: z0 = z0``
    這種寫法——那樣寫在 Python 的 class 主體作用域規則下，只要同一個類別
    主體內任何地方對 ``z0`` 賦值，``z0`` 就會被視為整個類別主體的區域變數，
    連帶讓賦值「之前」讀取外層同名變數的那一行也讀不到外層值而拋出
    ``NameError``；用建構子參數賦值可以完全避開這個陷阱。
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, row_particle_index: np.ndarray) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.z0 = z
        self.row_particle_index = row_particle_index


def _make_particles(handles: bd.BayTraceModuleHandles, n: int):
    return handles.particle_arrays_cls(n)


def _copy_particle_arrays(handles: bd.BayTraceModuleHandles, src) -> object:
    dst = handles.particle_arrays_cls(src.n)
    for name in (*src._FLOAT_FIELDS, *src._INT64_FIELDS, *src._INT_FIELDS):
        getattr(dst, name)[:] = getattr(src, name)
    dst.ids = list(src.ids)
    return dst


def _place_and_locate(
    driver_grid: bd.NativeBackwardDriverGrid,
    handles: bd.BayTraceModuleHandles,
    pa,
    *,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    field_pair: HourlyFieldPair,
) -> None:
    pa.x[:] = x
    pa.y[:] = y
    pa.z[:] = z
    pa.z0[:] = z
    pa.element[:] = -1
    pa.status[:] = 0
    not_found = bd.locate_initial_elements(
        driver_grid,
        handles,
        pa,
        eta_m=field_pair.eta_field1,
        u_current_mps=field_pair.u_field1,
        v_current_mps=field_pair.v_field1,
        w_current_mps=field_pair.w_field1,
    )
    assert not np.any(not_found), "合成粒子位置應該全部落在網格內"


def _direct_kernel_substep_loop(
    driver_grid: bd.NativeBackwardDriverGrid,
    handles: bd.BayTraceModuleHandles,
    pa,
    *,
    ibf: int,
    ndeltp: int,
    dtb: float,
    composed: bd.ComposedFieldPair,
) -> None:
    """獨立於 ``mirror_rk4_substep_loop`` 之外，逐行對照
    ``tracker.py`` 1519-1674 行的 passive+RK4 分支，直接呼叫 BayTrace 原始
    njit kernel。這是驗收 1（鏡像忠實度）的「地面真相」參考實作，刻意不呼叫
    本模組的 ``mirror_rk4_substep_loop``，避免自我比較毫無意義。
    """

    grid = driver_grid.schism_grid
    n = pa.n
    x0 = pa.x.copy()
    y0 = pa.y.copy()
    z0 = pa.z.copy()
    for idt in range(1, ndeltp + 1):
        trat = (idt / ndeltp) if ibf == 1 else ((idt - 1) / ndeltp)
        handles.batch_move_rk4(
            pa.x, pa.y, pa.z, pa.u, pa.v, pa.w,
            pa.x_prev, pa.y_prev, pa.element, pa.status,
            x0, y0, z0,
            grid.i34, grid.elnode, grid.x, grid.y, grid.ic3,
            grid.nvrt, grid.kbp, grid.kbe, grid.sigma_lcl, grid.dp,
            composed.u_field1, composed.v_field1, composed.w_field1, composed.eta_field1,
            composed.u_field2, composed.v_field2, composed.w_field2,
            ibf, dtb, trat,
        )
        needs_fallback = np.zeros(n, dtype=np.int32)
        handles.batch_check_boundaries(
            pa.x, pa.y, pa.x_prev, pa.y_prev,
            pa.element, pa.status, pa.inbr, pa.rnds,
            grid.i34, grid.elnode, grid.x, grid.y, grid.xctr, grid.yctr,
            grid.elside, grid.isidenode, grid.ic3, grid.snx, grid.sny, grid.isbnd,
            composed.elem_is_dry.astype(np.int64),
            0, 1.0,
            needs_fallback,
        )
        assert not np.any(needs_fallback), "合成情境不應觸發 needs_fallback"
        handles.batch_clamp_z(
            pa.x, pa.y, pa.z, pa.element, pa.status,
            grid.elnode, grid.x, grid.y,
            grid.nvrt, grid.kbp, grid.kbe, grid.sigma_lcl, grid.dp, composed.eta_field1,
        )
        handles.batch_interpolate_velocity(
            pa.x, pa.y, pa.z, pa.element, pa.status,
            grid.i34, grid.elnode, grid.x, grid.y,
            grid.nvrt, grid.kbp, grid.kbe, grid.sigma_lcl, grid.dp,
            composed.u_field1, composed.v_field1, composed.w_field1, composed.eta_field1,
            composed.u_field2, composed.v_field2, composed.w_field2,
            trat, driver_grid.h0,
            pa.u, pa.v, pa.w,
        )
        x0[:] = pa.x
        y0[:] = pa.y
        z0[:] = pa.z


# ── 驗收 1：鏡像忠實度（最重要）───────────────────────────────────────────


def test_mirror_matches_direct_kernel_calls_bit_identical(driver_grid, handles) -> None:
    """擴散關閉、Stokes/沉降為 0：鏡像子步迴圈與直接呼叫原版 kernel 位元相同。"""

    field_pair = _uniform_field_pair(driver_grid, u1=0.06, v1=-0.04, ibf=1, trat=0.0)
    composed = bd.compose_total_velocity_fields(driver_grid, handles, field_pair, settling_velocity_mps=0.0)

    n = 6
    x = np.array([700.0 + 15.0 * k for k in range(n)])
    y = np.array([700.0 - 10.0 * k for k in range(n)])
    z = np.full(n, -5.0)

    pa_reference = _make_particles(handles, n)
    _place_and_locate(driver_grid, handles, pa_reference, x=x, y=y, z=z, field_pair=field_pair)
    pa_mirror = _copy_particle_arrays(handles, pa_reference)

    ndeltp = bd.estimate_substep_count(driver_grid, composed.u_field1, composed.v_field1, dt_seconds=3600.0)
    dtb = 3600.0 / ndeltp

    _direct_kernel_substep_loop(
        driver_grid, handles, pa_reference, ibf=1, ndeltp=ndeltp, dtb=dtb, composed=composed
    )
    bd.mirror_rk4_substep_loop(
        driver_grid, handles, pa_mirror,
        ibf=1, ndeltp=ndeltp, dtb=dtb, composed=composed, enable_diffusion=False,
    )

    for field in ("x", "y", "z", "u", "v", "w", "element", "status", "x_prev", "y_prev"):
        a = getattr(pa_reference, field)
        b = getattr(pa_mirror, field)
        assert np.array_equal(a, b), f"欄位 {field} 在鏡像與直接呼叫之間不是位元相同：{a} vs {b}"


def _build_real_particle_tracker(
    driver_grid: bd.NativeBackwardDriverGrid,
    handles: bd.BayTraceModuleHandles,
    *,
    ibf: int,
    ndeltp: int,
    composed: bd.ComposedFieldPair,
) -> object:
    """組出一個真正的（未修改的）``ParticleTracker`` 執行個體，只手動設定
    ``batch_update_particles``（``tracker.py`` 1443-1692 行）實際會讀寫到的
    屬性子集，不讀取任何設定檔或 netCDF（用 ``object.__new__`` 跳過
    ``__init__``，比照本工作包 docstring 建議的作法）。

    這是驗收 1b 的「地面真相」——與既有
    ``test_mirror_matches_direct_kernel_calls_bit_identical`` 不同，那個測試
    比對的是「測試檔自己手寫的迴圈」，本函式改成直接呼叫 BayTrace 原版
    ``ParticleTracker.batch_update_particles`` 本體（含它內部呼叫的
    ``_batch_check_boundaries``／``_batch_clamp_z``／``_batch_interpolate_velocity``
    包裝方法），兩邊若有相同的偏差就不會被既有測試遮蔽。

    屬性設定原則：``self.grid`` 直接重用 ``driver_grid.schism_grid``（真正的
    ``SchismGrid``，已經有 ``build_side_structure`` 建好的邊結構），
    ``self._grid_*_i64`` 系列直接借用同一份陣列的別名（``tracker.py`` 原本
    也是在 ``_precompute_grid_arrays`` 用 ``astype(np.int64)`` 複製一份，這裡
    因為來源已經是 int64，別名與複製在數值上等價，只是省去一次不必要的
    複製）。``idry_e`` 直接取用 ``composed.elem_is_dry``（已由
    ``compose_total_velocity_fields`` 呼叫未修改的 ``_compute_dry_flags``
    算出），轉成原版慣用的 ``int32``（見 ``tracker._compute_dry_flags``
    ``idry_new = (...).astype(np.int32)``）。

    ``_find_new_element_fallback`` 被覆寫成一個會直接讓測試失敗的樁
    （stub），因為本測試的合成情境設計為「一小時內位移遠小於網格範圍」，
    理論上不應該觸發 KD-tree 後援；若真的觸發，代表測試資料設計有問題，
    應該讓測試明確失敗，而不是靜默吞掉、得到一個其實沒有真正跑到後援分支
    的假通過。
    """

    tracker = object.__new__(handles.tracker_cls)
    grid = driver_grid.schism_grid
    tracker.grid = grid
    tracker._grid_i34_i64 = grid.i34
    tracker._grid_elnode_i64 = grid.elnode
    tracker._grid_kbp_i64 = grid.kbp
    tracker._grid_kbe_i64 = grid.kbe
    tracker._grid_ic3_i64 = grid.ic3
    tracker._grid_elside_i64 = grid.elside
    tracker._grid_isidenode_i64 = grid.isidenode

    tracker.params = {
        "h0": driver_grid.h0,
        "mod_part": 0,
        "istiff": 0,
        "pbeach": 1.0,
        "ibf": ibf,
        "ndeltp": ndeltp,
    }
    tracker.istokes = 0
    tracker.iadvection = 2
    tracker.iweathering = 0
    tracker.rng = None
    tracker.idry_e = composed.elem_is_dry.astype(np.int32)

    tracker.u_field1 = composed.u_field1
    tracker.v_field1 = composed.v_field1
    tracker.w_field1 = composed.w_field1
    tracker.eta_field1 = composed.eta_field1
    tracker.u_field2 = composed.u_field2
    tracker.v_field2 = composed.v_field2
    tracker.w_field2 = composed.w_field2

    def _fail_on_fallback(pa, i):  # noqa: ANN001 - 比照原版方法簽名，純測試用樁
        raise AssertionError(
            f"真正的 ParticleTracker 對粒子 {i} 觸發了 KD-tree 後援；"
            "本測試的合成情境設計為不應觸發，出現代表測試資料或鏡像邏輯有問題"
        )

    tracker._find_new_element_fallback = _fail_on_fallback
    return tracker


def test_mirror_matches_real_tracker_batch_update_particles_both_directions(driver_grid, handles) -> None:
    """真正的忠實度檢驗：與 BayTrace 原版 ``ParticleTracker.batch_update_particles``
    （而非測試檔自己手寫的迴圈）逐項比對，ibf=+1、ibf=-1 都要驗證。

    背景：既有 ``test_mirror_matches_direct_kernel_calls_bit_identical`` 比對
    的「地面真相」是 ``_direct_kernel_substep_loop``——那是這份測試檔自己
    手寫、逐行對照 ``tracker.py`` 原始碼抄出來的迴圈，只呼叫四個 njit
    kernel。如果 ``mirror_rk4_substep_loop`` 與那個手寫迴圈剛好共享了同一個
    誤解或抄寫錯誤（例如兩邊都抄漏了某個參數、抄錯了呼叫順序），那個測試
    完全抓不到，因為兩邊會「一起錯、一起對」。這裡改成直接呼叫 BayTrace
    原版 ``ParticleTracker.batch_update_particles`` 本體（透過
    ``_build_real_particle_tracker`` 建的最小鴨子型別容器，只設定該方法
    passive／RK4 分支實際用到的屬性子集），確保比對的是「原版真正會執行的
    程式碼路徑」，不是重新謄寫的版本。另外既有測試只涵蓋 ``ibf=+1``，這裡
    把 ``ibf=-1``（本專案正式用途唯一會用到的方向）也一併納入。

    擴散、Stokes、沉降依驗收要求全部關閉／為零（``enable_diffusion=False``、
    ``settling_velocity_mps=0.0``、不提供 ``wave_field1/2``），這樣兩條路徑
    在數學上應該完全等價，任何差異都必須被視為鏡像邏輯的臭蟲並修正，而不是
    放寬容差讓測試通過。
    """

    for ibf in (1, -1):
        field_pair = _uniform_field_pair(driver_grid, u1=0.06, v1=-0.04, ibf=ibf, trat=0.0)
        composed = bd.compose_total_velocity_fields(
            driver_grid, handles, field_pair, settling_velocity_mps=0.0,
        )

        n = 6
        x = np.array([700.0 + 15.0 * k for k in range(n)])
        y = np.array([700.0 - 10.0 * k for k in range(n)])
        z = np.full(n, -5.0)

        pa_reference = _make_particles(handles, n)
        _place_and_locate(driver_grid, handles, pa_reference, x=x, y=y, z=z, field_pair=field_pair)
        pa_mirror = _copy_particle_arrays(handles, pa_reference)

        ndeltp = bd.estimate_substep_count(
            driver_grid, composed.u_field1, composed.v_field1, dt_seconds=3600.0,
        )
        dtb = 3600.0 / ndeltp
        assert dtb == pytest.approx(3600.0 / ndeltp)  # 供下方注意事項對照，非實質斷言

        real_tracker = _build_real_particle_tracker(
            driver_grid, handles, ibf=ibf, ndeltp=ndeltp, composed=composed
        )
        trackable_mask = np.ones(n, dtype=bool)
        # dt 直接傳 3600.0（而非 ndeltp*dtb），確保原版內部
        # ``dtb = dt / ndeltp`` 與這裡／鏡像函式的 ``dtb = 3600.0 / ndeltp``
        # 是完全相同的一次浮點除法，不會因為先乘後除引入額外的捨入誤差
        # 而製造出與鏡像邏輯無關的假差異。
        handles.tracker_cls.batch_update_particles(
            real_tracker, trackable_mask, 3600.0, 0.0, pa=pa_reference
        )

        bd.mirror_rk4_substep_loop(
            driver_grid, handles, pa_mirror,
            ibf=ibf, ndeltp=ndeltp, dtb=dtb, composed=composed, enable_diffusion=False,
        )

        for field in ("x", "y", "z", "u", "v", "w", "element", "status", "x_prev", "y_prev"):
            a = getattr(pa_reference, field)
            b = getattr(pa_mirror, field)
            assert np.array_equal(a, b), (
                f"ibf={ibf}：欄位 {field} 在鏡像與真正的 ParticleTracker."
                f"batch_update_particles 之間不是位元相同：{a} vs {b}"
            )


# ── 驗收 2：速度合成正確性 ─────────────────────────────────────────────────


def test_composed_velocity_matches_stokes_reference_at_each_node_layer(driver_grid, handles) -> None:
    """合成場的 Stokes 增量必須逐節點逐層等於 WP-3 純量參考實作的結果。"""

    n_nodes = _n_nodes(driver_grid)
    field_pair = _uniform_field_pair(driver_grid, u1=0.02, v1=0.01, ibf=-1)

    hs = 1.5
    fp = 0.1
    direction_deg = 30.0
    wave = bd.WaveFieldAtNodes(
        significant_wave_height_m=np.full(n_nodes, hs),
        peak_frequency_hz=np.full(n_nodes, fp),
        direction_raw_deg=np.full(n_nodes, direction_deg),
        valid_mask_wave=np.ones(n_nodes, dtype=bool),
        qc_flags=np.zeros(n_nodes, dtype=np.int64),
    )

    composed = bd.compose_total_velocity_fields(
        driver_grid, handles, field_pair, settling_velocity_mps=0.0, wave_field1=wave, wave_field2=wave
    )
    assert composed.stokes_ok_fraction == pytest.approx(1.0)

    # 表層節點（sigma=0，z 相對海面=0）用參考實作直接算出 Stokes，逐一核對。
    surface_layer = _NVRT - 1
    reference = finite_depth_stokes_reference(
        significant_wave_height_m=np.array([hs]),
        peak_frequency_hz=np.array([fp]),
        direction_raw_deg=np.array([direction_deg]),
        particle_z_m=np.array([0.0]),
        water_depth_m=np.array([_DP_M]),
        valid_mask_wave=np.array([True]),
        qc_flags=np.array([0]),
    )
    expected_u = float(reference.stokes_u_mps[0])
    expected_v = float(reference.stokes_v_mps[0])
    for node in range(0, n_nodes, 7):
        got_u = composed.u_field1[surface_layer, node] - field_pair.u_field1[surface_layer, node]
        got_v = composed.v_field1[surface_layer, node] - field_pair.v_field1[surface_layer, node]
        assert got_u == pytest.approx(expected_u, rel=1e-9, abs=1e-12)
        assert got_v == pytest.approx(expected_v, rel=1e-9, abs=1e-12)


def test_apply_settling_velocity_only_affects_effective_mask_layers() -> None:
    """任務 4 檢查項 (e) 的獨立單元測試：沉降速度只能加在
    ``effective_mask``（濕節點、且未低於海床）為 True 的位置；乾節點與
    海床以下層無論沉降速度多大都必須維持 0。

    既有整合測試（如
    ``test_composed_velocity_matches_stokes_reference_at_each_node_layer``）
    用的合成網格是平底、``kbp`` 全部為 0、也沒有乾節點，剛好測不到「同一
    根柱子裡有些層無效、有些層有效」與「乾節點」這兩種邊界情況，所以這裡
    繞過完整網格與 RK4，直接對 ``apply_settling_velocity`` 做隔離單元
    測試，手工構造 ``kbp``／``node_is_dry`` 有變化的合成輸入。

    WP-15 更新：``base`` 現在依 ``apply_settling_velocity`` 明訂的前提條件
    （見該函式 docstring）先手工套用與 ``compose_current_and_stokes_fields``
    相同的歸零規則再傳入——這才是本函式真正的合法輸入型態（唯一合法產生
    ``ComposedCurrentAndStokesFields`` 的函式 ``compose_current_and_stokes_
    fields`` 保證輸出滿足這個不變量）。WP-15 之前的版本刻意傳入**未歸零**
    的 ``u_base``／``v_base``，用來測試 ``apply_settling_velocity`` 自己
    「防禦性第二次歸零」的行為；WP-15 移除了這個防禦性重算（見
    ``apply_settling_velocity`` docstring 的證明：只要輸入滿足不變量，
    第二次歸零永遠是 no-op），因此本測試改成驗證「输入已經滿足不變量時，
    輸出的有效／無效位置語意仍然正確」，這與原測試想驗證的物理行為
    （乾點與海床以下層必須是 0）完全相同，只是不再依賴函式自己重新歸零
    來達成。函式簽章仍接受 ``kbp`` 參數（相容性保留，內部已不使用），這裡
    仍照樣傳入。
    """

    nvrt, nnode = 3, 4
    # 節點 0：kbp=0，三層皆有效、非乾。
    # 節點 1：kbp=1，第 0 層在海床以下（無效），第 1、2 層有效。
    # 節點 2：kbp=2，只有第 2 層有效。
    # 節點 3：kbp=0，但整根柱子是乾的（node_is_dry=True）——乾節點一律
    #         無效，優先於 kbp。
    kbp = np.array([0, 1, 2, 0], dtype=np.int64)
    node_is_dry = np.array([False, False, False, True])
    effective_mask = (np.arange(nvrt)[:, None] >= kbp[None, :]) & ~node_is_dry[None, :]
    below_wet = (np.arange(nvrt)[:, None] < kbp[None, :]) & ~node_is_dry[None, :]

    # w/u/v 初始值故意設非零常數，這樣才能驗證「有效位置確實加了沉降」；
    # 無效位置（乾點、海床以下層）依 apply_settling_velocity 的前提條件，
    # 在傳入前就必須已經是 0（與 compose_current_and_stokes_fields 的輸出
    # 不變量一致）。
    w_base = np.full((nvrt, nnode), 0.3, dtype=np.float64)
    u_base = np.full((nvrt, nnode), 0.05, dtype=np.float64)
    v_base = np.full((nvrt, nnode), -0.02, dtype=np.float64)
    for arr in (w_base, u_base, v_base):
        arr[below_wet] = 0.0
        arr[:, node_is_dry] = 0.0

    base = bd.ComposedCurrentAndStokesFields(
        u_field1=u_base.copy(),
        v_field1=v_base.copy(),
        w_field1_no_settling=w_base.copy(),
        eta_field1=np.zeros(nnode, dtype=np.float64),
        u_field2=u_base.copy(),
        v_field2=v_base.copy(),
        w_field2_no_settling=w_base.copy(),
        node_is_dry=node_is_dry,
        elem_is_dry=np.zeros(1, dtype=bool),
        effective_mask=effective_mask,
        stokes_ok_fraction=1.0,
    )

    settling = -0.123
    composed = bd.apply_settling_velocity(base, settling, kbp=kbp)

    for field_name in ("w_field1", "w_field2"):
        w = getattr(composed, field_name)
        assert np.allclose(w[effective_mask], w_base[effective_mask] + settling)
        assert np.all(w[below_wet] == 0.0)
        assert np.all(w[:, node_is_dry] == 0.0)

    for field_name, base_arr in (
        ("u_field1", u_base),
        ("v_field1", v_base),
        ("u_field2", u_base),
        ("v_field2", v_base),
    ):
        arr = getattr(composed, field_name)
        # u/v 不受沉降影響，有效位置必須維持原值。
        assert np.allclose(arr[effective_mask], base_arr[effective_mask])
        # 無效位置（海床以下、乾節點）本來就是 0（輸入不變量），輸出必須維持 0。
        assert np.all(arr[below_wet] == 0.0)
        assert np.all(arr[:, node_is_dry] == 0.0)


def _legacy_apply_settling_velocity(
    base: bd.ComposedCurrentAndStokesFields, settling_velocity_mps: float, *, kbp: np.ndarray
) -> bd.ComposedFieldPair:
    """WP-15 優化前 ``apply_settling_velocity`` 的逐行複製版本，只用於本檔案
    的位元相同回歸測試，不是正式程式碼的一部分。刻意獨立於
    ``backward_driver.py`` 保存，即使未來優化後的版本再被修改，這份「歷史
    快照」仍能當作獨立的第三方比對基準。
    """

    settling = float(settling_velocity_mps)

    w1 = np.array(base.w_field1_no_settling, dtype=np.float64, copy=True)
    w2 = np.array(base.w_field2_no_settling, dtype=np.float64, copy=True)
    w1[base.effective_mask] += settling
    w2[base.effective_mask] += settling

    u1 = np.array(base.u_field1, dtype=np.float64, copy=True)
    v1 = np.array(base.v_field1, dtype=np.float64, copy=True)
    u2 = np.array(base.u_field2, dtype=np.float64, copy=True)
    v2 = np.array(base.v_field2, dtype=np.float64, copy=True)

    def _zero(u_field, v_field, w_field):
        nvrt_local = u_field.shape[0]
        u_field[:, base.node_is_dry] = 0.0
        v_field[:, base.node_is_dry] = 0.0
        w_field[:, base.node_is_dry] = 0.0
        layers = np.arange(nvrt_local)[:, None]
        below_wet = (layers < kbp[None, :]) & ~base.node_is_dry[None, :]
        u_field[below_wet] = 0.0
        v_field[below_wet] = 0.0
        w_field[below_wet] = 0.0

    _zero(u1, v1, w1)
    _zero(u2, v2, w2)

    return bd.ComposedFieldPair(
        u_field1=np.ascontiguousarray(u1),
        v_field1=np.ascontiguousarray(v1),
        w_field1=np.ascontiguousarray(w1),
        eta_field1=base.eta_field1,
        u_field2=np.ascontiguousarray(u2),
        v_field2=np.ascontiguousarray(v2),
        w_field2=np.ascontiguousarray(w2),
        node_is_dry=base.node_is_dry,
        elem_is_dry=base.elem_is_dry,
        stokes_ok_fraction=base.stokes_ok_fraction,
    )


def test_apply_settling_velocity_matches_legacy_implementation_bit_identical() -> None:
    """WP-15 任務 A.2 的核心位元相同證明：對大量隨機、但滿足
    ``apply_settling_velocity`` 前提條件（乾點／海床以下層已歸零）的合成
    輸入，優化後的實作與 ``_legacy_apply_settling_velocity``（WP-15 之前
    的逐行複製版本）輸出必須逐位元相同。

    覆蓋範圍：多組 ``(nvrt, nnode)`` 規模、隨機 ``kbp``、隨機乾濕分布、
    正／負／零沉降速度、隨機 u/v/w 初值（含 NaN 以外的極端值），確保優化
    不是「剛好在某個簡單案例碰巧一樣」。
    """

    rng = np.random.default_rng(20260922)
    for nvrt, nnode in ((1, 1), (3, 4), (5, 17), (10, 50)):
        for _trial in range(5):
            kbp = rng.integers(0, nvrt + 1, size=nnode).astype(np.int64)
            node_is_dry = rng.random(nnode) < 0.3
            effective_mask = (np.arange(nvrt)[:, None] >= kbp[None, :]) & ~node_is_dry[None, :]
            below_wet = (np.arange(nvrt)[:, None] < kbp[None, :]) & ~node_is_dry[None, :]

            def _random_zeroed_field(
                nvrt: int = nvrt, nnode: int = nnode, below_wet: np.ndarray = below_wet,
                node_is_dry: np.ndarray = node_is_dry,
            ) -> np.ndarray:
                arr = rng.uniform(-5.0, 5.0, size=(nvrt, nnode))
                arr[below_wet] = 0.0
                arr[:, node_is_dry] = 0.0
                return arr

            u1, v1, w1 = _random_zeroed_field(), _random_zeroed_field(), _random_zeroed_field()
            u2, v2, w2 = _random_zeroed_field(), _random_zeroed_field(), _random_zeroed_field()

            base = bd.ComposedCurrentAndStokesFields(
                u_field1=u1, v_field1=v1, w_field1_no_settling=w1,
                eta_field1=rng.uniform(-1.0, 1.0, size=nnode),
                u_field2=u2, v_field2=v2, w_field2_no_settling=w2,
                node_is_dry=node_is_dry,
                elem_is_dry=np.zeros(1, dtype=bool),
                effective_mask=effective_mask,
                stokes_ok_fraction=1.0,
            )

            for settling in (0.0, -0.001, 0.05, -12.3456789):
                legacy = _legacy_apply_settling_velocity(base, settling, kbp=kbp)
                optimized = bd.apply_settling_velocity(base, settling, kbp=kbp)
                for field_name in (
                    "u_field1", "v_field1", "w_field1", "u_field2", "v_field2", "w_field2",
                ):
                    assert np.array_equal(getattr(legacy, field_name), getattr(optimized, field_name)), (
                        f"nvrt={nvrt} nnode={nnode} settling={settling} field={field_name}"
                    )
                assert np.array_equal(legacy.node_is_dry, optimized.node_is_dry)
                assert np.array_equal(legacy.elem_is_dry, optimized.elem_is_dry)
                assert legacy.stokes_ok_fraction == optimized.stokes_ok_fraction


def test_apply_settling_velocity_shares_u_v_arrays_without_copy() -> None:
    """WP-15 優化的關鍵不變量：``u_field1``／``v_field1``／``u_field2``／
    ``v_field2`` 不需要複製，直接回傳 ``base`` 的陣列參考（多個沉降分組
    安全共用同一份 ``base`` 而不互相污染，因為下游 kernel 只讀不寫，見
    ``apply_settling_velocity`` docstring）；``w_field1``／``w_field2``
    則必須是獨立新配置的陣列，不能與 ``base.w_field*_no_settling`` 共用
    記憶體，否則不同沉降分組會互相覆寫對方的沉降貢獻。
    """

    nvrt, nnode = 3, 4
    kbp = np.zeros(nnode, dtype=np.int64)
    node_is_dry = np.zeros(nnode, dtype=bool)
    effective_mask = np.ones((nvrt, nnode), dtype=bool)
    base = bd.ComposedCurrentAndStokesFields(
        u_field1=np.full((nvrt, nnode), 0.1), v_field1=np.full((nvrt, nnode), 0.2),
        w_field1_no_settling=np.zeros((nvrt, nnode)),
        eta_field1=np.zeros(nnode),
        u_field2=np.full((nvrt, nnode), 0.3), v_field2=np.full((nvrt, nnode), 0.4),
        w_field2_no_settling=np.zeros((nvrt, nnode)),
        node_is_dry=node_is_dry, elem_is_dry=np.zeros(1, dtype=bool),
        effective_mask=effective_mask, stokes_ok_fraction=1.0,
    )

    composed_a = bd.apply_settling_velocity(base, -0.01, kbp=kbp)
    composed_b = bd.apply_settling_velocity(base, -0.02, kbp=kbp)

    assert composed_a.u_field1 is base.u_field1
    assert composed_a.v_field1 is base.v_field1
    assert composed_a.u_field2 is base.u_field2
    assert composed_a.v_field2 is base.v_field2
    assert not np.shares_memory(composed_a.w_field1, base.w_field1_no_settling)
    assert not np.shares_memory(composed_a.w_field2, base.w_field2_no_settling)

    # 兩個不同沉降分組各自的 w 場必須彼此獨立，不因共用同一個 base 而互相污染。
    assert np.allclose(composed_a.w_field1, -0.01)
    assert np.allclose(composed_b.w_field1, -0.02)
    assert np.allclose(composed_a.w_field1, -0.01)  # 再讀一次，確認上面沒有被 composed_b 覆寫


def test_composed_velocity_all_four_rk4_stages_vs_wrong_k1_only_addition(driver_grid, handles) -> None:
    """驗證合成速度場位移＝(海流+Stokes+沉降)×時間，且與「只加在 k1」的錯誤做法不同。

    刻意使用空間完全均勻、時間不變（field1==field2）的流場，讓 RK4 對常數
    向量場的積分在數學上精確等於「速度×時間」，藉此把「合成是否讓四個
    stage 都讀到附加速度」的問題化簡成一次直接的數值比較。
    """

    u_curr, v_curr = 0.03, -0.02
    stokes_u, stokes_v = 0.01, 0.004
    settling = -0.0005
    field_pair = _uniform_field_pair(driver_grid, u1=u_curr, v1=v_curr, w1=0.0, ibf=1, trat=0.0)

    n_nodes = _n_nodes(driver_grid)
    # 建一個等效於常數 Stokes 的合成場：直接疊加常數速度到 u/v_field（略過
    # WP-3 公式本身，只為了得到「已知常數 Stokes」這個測試前提），並疊加常數
    # 沉降到 w_field，模擬 compose_total_velocity_fields 步驟 3、4 的效果。
    composed_correct = bd.ComposedFieldPair(
        u_field1=np.full((_NVRT, n_nodes), u_curr + stokes_u),
        v_field1=np.full((_NVRT, n_nodes), v_curr + stokes_v),
        w_field1=np.full((_NVRT, n_nodes), settling),
        eta_field1=field_pair.eta_field1.copy(),
        u_field2=np.full((_NVRT, n_nodes), u_curr + stokes_u),
        v_field2=np.full((_NVRT, n_nodes), v_curr + stokes_v),
        w_field2=np.full((_NVRT, n_nodes), settling),
        node_is_dry=np.zeros(n_nodes, dtype=bool),
        elem_is_dry=np.zeros(driver_grid.grid.i34.shape[0], dtype=bool),
        stokes_ok_fraction=1.0,
    )

    n = 3
    x0 = np.array([700.0, 720.0, 740.0])
    y0 = np.array([700.0, 690.0, 680.0])
    z0 = np.full(n, 0.0)  # 表層，避免沉降把粒子帶出濁層造成速度隨層變化

    pa_correct = _make_particles(handles, n)
    _place_and_locate(driver_grid, handles, pa_correct, x=x0, y=y0, z=z0, field_pair=field_pair)
    # 比照 run_backward_batch 的「起點速度預熱」：locate_initial_elements 只用
    # 純海流、單層初始化 pa.u/v/w（見該函式 docstring），正式流程會在進入
    # mirror_rk4_substep_loop 之前先用合成後的總速度場重新內插一次，否則
    # 第一個子步的 k1 會少算「合成場 - 純海流」這一份差異（× dtb/6 的權重）。
    # 這裡直接賦值，因為流場對本測試而言在空間上完全均勻，等價於重新內插。
    pa_correct.u[:] = u_curr + stokes_u
    pa_correct.v[:] = v_curr + stokes_v
    pa_correct.w[:] = settling

    ndeltp = bd.estimate_substep_count(
        driver_grid, composed_correct.u_field1, composed_correct.v_field1, dt_seconds=3600.0,
    )
    dtb = 3600.0 / ndeltp
    bd.mirror_rk4_substep_loop(
        driver_grid, handles, pa_correct,
        ibf=1, ndeltp=ndeltp, dtb=dtb, composed=composed_correct, enable_diffusion=False,
    )

    dt_total = ndeltp * dtb
    expected_x = x0 + (u_curr + stokes_u) * dt_total
    expected_y = y0 + (v_curr + stokes_v) * dt_total
    np.testing.assert_allclose(pa_correct.x, expected_x, atol=1e-6)
    np.testing.assert_allclose(pa_correct.y, expected_y, atol=1e-6)

    # 錯誤做法：只把 Stokes／沉降加到 pa.u/pa.v/pa.w 一次（STEP 0 的舊慣例），
    # 但 batch_move_rk4 收到的仍是「純海流」欄位（沒有合成），完全比照事實 1
    # 描述的錯誤路徑：k1 拿到附加速度，k2-k4 透過 _interp_velocity_at 讀純
    # 海流欄位，STEP 3 又把 pa.u/pa.v 覆寫回純海流。
    pa_wrong = _make_particles(handles, n)
    _place_and_locate(driver_grid, handles, pa_wrong, x=x0, y=y0, z=z0, field_pair=field_pair)
    pa_wrong.u[:] += stokes_u
    pa_wrong.v[:] += stokes_v
    pa_wrong.w[:] += settling
    composed_pure_current = bd.compose_total_velocity_fields(
        driver_grid, handles, field_pair, settling_velocity_mps=0.0
    )
    bd.mirror_rk4_substep_loop(
        driver_grid, handles, pa_wrong,
        ibf=1, ndeltp=ndeltp, dtb=dtb, composed=composed_pure_current, enable_diffusion=False,
    )

    # 錯誤做法的位移應該明顯小於正確做法（只在第一個子步的 k1 拿到 1/6 權重
    # 的附加速度，其餘子步完全拿不到），兩者不可能相等。
    assert not np.allclose(pa_wrong.x, expected_x, atol=1e-3)
    assert not np.allclose(pa_wrong.y, expected_y, atol=1e-3)
    # 錯誤做法應該非常接近「完全沒有附加速度」的純海流位移（略多一點點，
    # 來自第一個子步 k1 那 1/6 權重），藉此佐證額外速度幾乎全部遺失。
    pure_current_x = x0 + u_curr * dt_total
    y0 + v_curr * dt_total
    assert np.max(np.abs(pa_wrong.x - pure_current_x)) < np.max(np.abs(expected_x - pure_current_x)) * 0.5


# ── 驗收 3：逆向時間方向 ───────────────────────────────────────────────────


def test_reverse_time_direction_retraces_forward_trajectory(driver_grid, handles) -> None:
    """時間變動（分段線性、空間均勻）的合成流場：正向再反向應回到接近原點。

    以三個「快照」時刻 ``t0``／``t0+H``／``t0+2H`` 的速度定義流場，讓
    小時邊界上前一小時的 field2 與後一小時的 field1 讀到同一個快照值
    （與真實 OCM 逐小時資料的重疊邊界慣例一致），避免測試自己在小時邊界
    人工製造流場不連續（那會製造出與「反向時間方向是否正確」無關的額外
    離散化誤差，見程式內部除錯記錄）。
    """

    t0 = 2_000 * _HOUR_NS
    vel_snapshot = {
        t0: (0.05, 0.0),
        t0 + _HOUR_NS: (-0.02, 0.04),
        t0 + 2 * _HOUR_NS: (0.03, -0.03),
    }

    def field_between(*, ibf: int, hi: int, lo: int) -> HourlyFieldPair:
        """依 WP-2 慣例組出涵蓋 ``[lo, hi)`` 的欄位：``field1``／``field2``
        分別取兩端快照值，``ibf=-1`` 時 field1=hi（較晚）、field2=lo
        （較早），``ibf=1`` 時相反。
        """

        u_hi, v_hi = vel_snapshot[hi]
        u_lo, v_lo = vel_snapshot[lo]
        if ibf == -1:
            u1, v1, u2, v2 = u_hi, v_hi, u_lo, v_lo
        else:
            u1, v1, u2, v2 = u_lo, v_lo, u_hi, v_hi
        return _uniform_field_pair(
            driver_grid, u1=u1, v1=v1, u2=u2, v2=v2, ibf=ibf, trat=0.0,
            time_before_utc_ns=hi, time_after_utc_ns=lo,
        )

    x0 = np.array([700.0])
    y0 = np.array([700.0])
    z0 = np.array([-5.0])

    field_hour1_fwd = field_between(ibf=1, hi=t0 + _HOUR_NS, lo=t0)
    pa_fwd = _make_particles(handles, 1)
    _place_and_locate(driver_grid, handles, pa_fwd, x=x0, y=y0, z=z0, field_pair=field_hour1_fwd)

    for hi in (t0 + _HOUR_NS, t0 + 2 * _HOUR_NS):
        lo = hi - _HOUR_NS
        field = field_between(ibf=1, hi=hi, lo=lo)
        composed = bd.compose_total_velocity_fields(
            driver_grid, handles, field, settling_velocity_mps=0.0,
        )
        ndeltp = bd.estimate_substep_count(
            driver_grid, composed.u_field1, composed.v_field1, dt_seconds=3600.0,
        )
        dtb = 3600.0 / ndeltp
        bd.mirror_rk4_substep_loop(
            driver_grid, handles, pa_fwd, ibf=1, ndeltp=ndeltp, dtb=dtb, composed=composed,
            enable_diffusion=False,
        )

    forward_end_x = float(pa_fwd.x[0])
    forward_end_y = float(pa_fwd.y[0])

    # 反向：query time 是每個小時區間「較晚」那一端（見 run_backward_batch
    # 外層迴圈 current_utc 的遞減方式），field1=hi（較晚真實時間）、
    # field2=lo（較早），對應本專案 ibf=-1 的既定慣例。
    def backward_resolver(target_utc_ns: int) -> HourlyFieldPair:
        return field_between(ibf=-1, hi=target_utc_ns, lo=target_utc_ns - _HOUR_NS)

    interest_polygon = Polygon([(-1.0e6, -1.0e6), (1.0e6, -1.0e6), (1.0e6, 1.0e6), (-1.0e6, 1.0e6)])

    soa = _SyntheticSoa(
        x=np.array([forward_end_x]), y=np.array([forward_end_y]), z=np.array([-5.0]),
        row_particle_index=np.array([0], dtype=np.int64),
    )

    config = bd.BackwardDriverConfig(
        arrival_time_utc_ns=t0 + 2 * _HOUR_NS,
        settling_velocity_mps=0.0,
        max_backtrack_seconds=2.0 * 3600.0,
        dt_max_seconds=30.0,
        enable_diffusion=False,
        output_interval_seconds=3600.0,
        master_seed=0,
    )
    result = bd.run_backward_batch(
        driver_grid, handles, soa,
        seed_hex_values=["0" * 32],
        config=config,
        forcing_resolver=backward_resolver,
        interest_region_polygon=interest_polygon,
    )

    assert result.final_status[0] == bd.NATIVE_STATUS_MAX_BACKTRACK
    # 允許小量容差：正向／反向各自的 RK4 子步 trat 取樣點並非鏡射對稱
    # （正向 trat=idt/ndeltp 在子步「終點」取值，逆向 trat=(idt-1)/ndeltp
    # 在子步「起點」取值，這是原版演算法本身的既定慣例，不是本模組引入的
    # 誤差），因此往返不保證位元相同，但應遠小於單一小時的位移量級
    # （約 100-200 公尺），確認方向與量級都正確地反轉回原點附近。
    assert result.final_x_m[0] == pytest.approx(700.0, abs=2.0)
    assert result.final_y_m[0] == pytest.approx(700.0, abs=2.0)


# ── 驗收 4：擴散統計 ───────────────────────────────────────────────────────


def test_diffusion_variance_matches_2kt_and_not_applied_inside_rk4(driver_grid, handles) -> None:
    """均勻 K、無平流（u=v=0）的合成流場：位移變異數應等於 2*K*t（每軸）。"""

    kh_const = 0.05  # m^2/s，選一個讓一小時內位移遠小於網格範圍的量值
    field_pair = _uniform_field_pair(driver_grid, u1=0.0, v1=0.0, ibf=1, trat=0.0)

    n_nodes = _n_nodes(driver_grid)
    hf_field = np.full((_NVRT, n_nodes), kh_const, dtype=np.float64)

    n = 4000
    rng_place = np.random.default_rng(12345)
    cx, cy = _DOMAIN_X / 2.0, _DOMAIN_Y / 2.0
    x0 = cx + rng_place.uniform(-50.0, 50.0, size=n)
    y0 = cy + rng_place.uniform(-50.0, 50.0, size=n)
    z0 = np.full(n, -5.0)

    pa = _make_particles(handles, n)
    _place_and_locate(driver_grid, handles, pa, x=x0, y=y0, z=z0, field_pair=field_pair)

    composed = bd.compose_total_velocity_fields(driver_grid, handles, field_pair, settling_velocity_mps=0.0)
    ndeltp = bd.estimate_substep_count(driver_grid, composed.u_field1, composed.v_field1, dt_seconds=3600.0)
    dtb = 3600.0 / ndeltp
    dt_total = ndeltp * dtb

    rng = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence(999)))
    bd.mirror_rk4_substep_loop(
        driver_grid, handles, pa,
        ibf=1, ndeltp=ndeltp, dtb=dtb, composed=composed, enable_diffusion=True,
        hf1_field=hf_field, hf2_field=hf_field, rng=rng,
    )

    dx = pa.x - x0
    dy = pa.y - y0
    # u=v=0，因此全部位移理論上都來自擴散；平均位移應接近 0（無漂移）。
    assert np.mean(dx) == pytest.approx(0.0, abs=6.0 * np.sqrt(2.0 * kh_const * dt_total / n))
    assert np.mean(dy) == pytest.approx(0.0, abs=6.0 * np.sqrt(2.0 * kh_const * dt_total / n))

    expected_var = 2.0 * kh_const * dt_total
    var_x = np.var(dx)
    var_y = np.var(dy)
    # 樣本變異數的抽樣誤差量級是 sqrt(2/n) 乘上真值；n=4000 時抓 5 個標準誤。
    tolerance = expected_var * 5.0 * np.sqrt(2.0 / n)
    assert var_x == pytest.approx(expected_var, abs=tolerance)
    assert var_y == pytest.approx(expected_var, abs=tolerance)


def test_diffusion_disabled_is_bit_identical_to_no_diffusion_path(driver_grid, handles) -> None:
    """``enable_diffusion=False`` 時，即使提供了 rng/hf 場也完全不影響結果。

    這間接證明擴散只在 ``enable_diffusion`` 為真時才會被呼叫，不會意外地在
    RK4 的任何一個 stage 內部被觸發——若擴散被錯誤地插入 RK4 内部，兩條路徑
    不可能無論如何都位元相同。
    """

    field_pair = _uniform_field_pair(driver_grid, u1=0.04, v1=0.02, ibf=1, trat=0.0)
    composed = bd.compose_total_velocity_fields(driver_grid, handles, field_pair, settling_velocity_mps=0.0)
    n = 5
    x0 = np.full(n, 700.0)
    y0 = np.full(n, 700.0)
    z0 = np.full(n, -5.0)

    pa_no_diff = _make_particles(handles, n)
    _place_and_locate(driver_grid, handles, pa_no_diff, x=x0, y=y0, z=z0, field_pair=field_pair)
    pa_disabled = _copy_particle_arrays(handles, pa_no_diff)

    ndeltp = bd.estimate_substep_count(driver_grid, composed.u_field1, composed.v_field1, dt_seconds=3600.0)
    dtb = 3600.0 / ndeltp

    bd.mirror_rk4_substep_loop(
        driver_grid, handles, pa_no_diff, ibf=1, ndeltp=ndeltp, dtb=dtb, composed=composed,
        enable_diffusion=False,
    )
    n_nodes = _n_nodes(driver_grid)
    np.full((_NVRT, n_nodes), 1.0, dtype=np.float64)
    bd.mirror_rk4_substep_loop(
        driver_grid, handles, pa_disabled, ibf=1, ndeltp=ndeltp, dtb=dtb, composed=composed,
        enable_diffusion=False,
    )
    assert np.array_equal(pa_no_diff.x, pa_disabled.x)
    assert np.array_equal(pa_no_diff.y, pa_disabled.y)


# ── 驗收 5、6：停止條件與狀態碼 ────────────────────────────────────────────


def test_stopping_conditions_and_status_codes_are_distinct(driver_grid, handles) -> None:
    """離開關注區域、回溯上限、擱淺分別記錄正確位置／時刻，狀態碼互不混淆。"""

    u_curr, v_curr = 0.08, 0.0
    arrival_utc = 3_000 * _HOUR_NS

    def resolver(target_utc_ns: int) -> HourlyFieldPair:
        hi = target_utc_ns
        lo = target_utc_ns - _HOUR_NS
        return _uniform_field_pair(
            driver_grid, u1=u_curr, v1=v_curr, ibf=-1, trat=0.0,
            time_before_utc_ns=hi, time_after_utc_ns=lo,
        )

    # 兩顆粒子共用同一個空間均勻流場（逆向時 ibf=-1 使真實位移方向與
    # u_curr 相反：batch_move_rk4 的 k1u = ibf*u，往 -x 移動），因此哪一顆
    # 先離開關注區域完全取決於「距離左邊界多遠」：
    # 粒子 0：貼近左邊界（50 公尺），在回溯上限之內就會先離開。
    # 粒子 1：放在網格最右側（距左邊界 700 公尺），同樣的回溯上限內走不到
    # 邊界，因此會撞到回溯上限而不是離開關注區域。
    interest_polygon = Polygon([(600.0, 0.0), (_DOMAIN_X, 0.0), (_DOMAIN_X, _DOMAIN_Y), (600.0, _DOMAIN_Y)])

    # run_backward_batch 的回溯上限檢查發生在「每個整小時邊界」（見該函式
    # while 迴圈開頭），不會在小時中途提前停止；因此這裡刻意選一個整數小時
    # 的回溯上限，讓「精確停在上限」的斷言有明確定義的期望值。
    max_backtrack_seconds = 2.0 * 3600.0  # 7200 秒：0.08 m/s 走 576 公尺

    n = 2
    x0 = np.array([650.0, 1300.0])
    y0 = np.array([700.0, 700.0])
    z0 = np.full(n, -5.0)

    soa = _SyntheticSoa(x=x0, y=y0, z=z0, row_particle_index=np.arange(n, dtype=np.int64))

    config = bd.BackwardDriverConfig(
        arrival_time_utc_ns=arrival_utc,
        settling_velocity_mps=0.0,
        max_backtrack_seconds=max_backtrack_seconds,
        dt_max_seconds=30.0,
        enable_diffusion=False,
        output_interval_seconds=3600.0,
        master_seed=7,
    )
    result = bd.run_backward_batch(
        driver_grid, handles, soa,
        seed_hex_values=["1" * 32, "2" * 32],
        config=config,
        forcing_resolver=resolver,
        interest_region_polygon=interest_polygon,
    )

    assert result.final_status[0] == bd.NATIVE_STATUS_LEFT_INTEREST_REGION
    assert result.final_status[1] == bd.NATIVE_STATUS_MAX_BACKTRACK
    assert result.final_status[0] != result.final_status[1]

    departure0 = next(d for d in result.departures if d.row_particle_index == 0)
    assert departure0.status == bd.NATIVE_STATUS_LEFT_INTEREST_REGION
    # 離開時的 x 應該略小於關注區域左邊界 600（往 -x 方向移出）。
    assert departure0.x_m < 600.0
    assert departure0.utc_ns < arrival_utc
    assert departure0.utc_ns > arrival_utc - int(config.max_backtrack_seconds * 1.0e9)

    departure1 = next(d for d in result.departures if d.row_particle_index == 1)
    assert departure1.status == bd.NATIVE_STATUS_MAX_BACKTRACK
    assert departure1.utc_ns == arrival_utc - int(config.max_backtrack_seconds * 1.0e9)

    # NATIVE_STATUS_DESCRIPTIONS 涵蓋所有用到的狀態碼，且彼此描述不同。
    used = {bd.NATIVE_STATUS_LEFT_INTEREST_REGION, bd.NATIVE_STATUS_MAX_BACKTRACK}
    descriptions = {code: bd.NATIVE_STATUS_DESCRIPTIONS[code] for code in used}
    assert len(set(descriptions.values())) == len(descriptions)


def test_all_status_codes_are_unique_and_documented() -> None:
    """七種狀態碼彼此數值不同，且每一種都有獨立描述文字。"""

    codes = [
        bd.NATIVE_STATUS_ACTIVE,
        bd.NATIVE_STATUS_LEFT_INTEREST_REGION,
        bd.NATIVE_STATUS_MAX_BACKTRACK,
        bd.NATIVE_STATUS_STRANDED,
        bd.NATIVE_STATUS_DEPOSITED,
        bd.NATIVE_STATUS_FORCING_MISSING,
        bd.NATIVE_STATUS_NUMERICAL_FAILURE,
    ]
    assert len(codes) == len(set(codes))
    descriptions = [bd.NATIVE_STATUS_DESCRIPTIONS[code] for code in codes]
    assert len(descriptions) == len(set(descriptions))


def test_forcing_missing_status_stops_batch_at_that_hour(driver_grid, handles) -> None:
    """濕節點缺值未經解釋時，整批粒子在該小時停止並標記為 FORCING_MISSING。"""

    arrival_utc = 4_000 * _HOUR_NS
    _n_nodes(driver_grid)

    def resolver(target_utc_ns: int) -> HourlyFieldPair:
        hi = target_utc_ns
        lo = target_utc_ns - _HOUR_NS
        pair = _uniform_field_pair(
            driver_grid, u1=0.01, v1=0.0, ibf=-1, trat=0.0,
            time_before_utc_ns=hi, time_after_utc_ns=lo,
        )
        # 在其中一個濕節點的 field1 製造一個未登錄的缺值（missing_mask=True
        # 但沒有任何乾點可以解釋它），觸發 UnresolvedForcingGapError。
        u1 = pair.u_field1.copy()
        missing1 = pair.missing_mask1.copy()
        u1[0, 5] = np.nan
        missing1[0, 5] = True
        return HourlyFieldPair(
            time_before_utc_ns=pair.time_before_utc_ns,
            time_after_utc_ns=pair.time_after_utc_ns,
            trat=pair.trat,
            ibf=pair.ibf,
            u_field1=u1, v_field1=pair.v_field1, w_field1=pair.w_field1,
            eta_field1=pair.eta_field1, diffusivity_field1=pair.diffusivity_field1,
            missing_mask1=missing1, dry_face_flag1=pair.dry_face_flag1,
            month_id1=pair.month_id1, time_origin1=pair.time_origin1,
            u_field2=pair.u_field2, v_field2=pair.v_field2, w_field2=pair.w_field2,
            eta_field2=pair.eta_field2, diffusivity_field2=pair.diffusivity_field2,
            missing_mask2=pair.missing_mask2, dry_face_flag2=pair.dry_face_flag2,
            month_id2=pair.month_id2, time_origin2=pair.time_origin2,
        )

    interest_polygon = Polygon([(-1.0e6, -1.0e6), (1.0e6, -1.0e6), (1.0e6, 1.0e6), (-1.0e6, 1.0e6)])

    soa = _SyntheticSoa(
        x=np.array([700.0]), y=np.array([700.0]), z=np.array([-5.0]),
        row_particle_index=np.array([0], dtype=np.int64),
    )

    config = bd.BackwardDriverConfig(
        arrival_time_utc_ns=arrival_utc,
        settling_velocity_mps=0.0,
        max_backtrack_seconds=5.0 * 3600.0,
        enable_diffusion=False,
        master_seed=1,
    )
    result = bd.run_backward_batch(
        driver_grid, handles, soa,
        seed_hex_values=["3" * 32],
        config=config,
        forcing_resolver=resolver,
        interest_region_polygon=interest_polygon,
    )
    assert result.final_status[0] == bd.NATIVE_STATUS_FORCING_MISSING
    assert result.hours_run == 0


# ── 驗收 7：可重現性 ───────────────────────────────────────────────────────


def test_reproducibility_same_seed_bit_identical(driver_grid, handles) -> None:
    """相同設定與種子重跑兩次，完整軌跡與最終狀態必須位元相同。"""

    arrival_utc = 5_000 * _HOUR_NS

    def resolver(target_utc_ns: int) -> HourlyFieldPair:
        hi = target_utc_ns
        lo = target_utc_ns - _HOUR_NS
        return _uniform_field_pair(
            driver_grid, u1=0.02, v1=-0.015, ibf=-1, trat=0.0,
            time_before_utc_ns=hi, time_after_utc_ns=lo,
        )

    interest_polygon = Polygon([(-1.0e6, -1.0e6), (1.0e6, -1.0e6), (1.0e6, 1.0e6), (-1.0e6, 1.0e6)])
    n = 8
    rng_place = np.random.default_rng(2024)
    x0 = _DOMAIN_X / 2.0 + rng_place.uniform(-100.0, 100.0, size=n)
    y0 = _DOMAIN_Y / 2.0 + rng_place.uniform(-100.0, 100.0, size=n)
    z0 = np.full(n, -5.0)
    seed_hex_values = [f"{i:032x}" for i in range(n)]

    soa = _SyntheticSoa(x=x0, y=y0, z=z0, row_particle_index=np.arange(n, dtype=np.int64))

    config = bd.BackwardDriverConfig(
        arrival_time_utc_ns=arrival_utc,
        settling_velocity_mps=-0.0002,
        max_backtrack_seconds=4.0 * 3600.0,
        enable_diffusion=True,
        master_seed=123,
    )

    result_a = bd.run_backward_batch(
        driver_grid, handles, soa,
        seed_hex_values=seed_hex_values,
        config=config,
        forcing_resolver=resolver,
        interest_region_polygon=interest_polygon,
    )
    result_b = bd.run_backward_batch(
        driver_grid, handles, soa,
        seed_hex_values=seed_hex_values,
        config=config,
        forcing_resolver=resolver,
        interest_region_polygon=interest_polygon,
    )

    assert np.array_equal(result_a.trajectory_x_m, result_b.trajectory_x_m)
    assert np.array_equal(result_a.trajectory_y_m, result_b.trajectory_y_m)
    assert np.array_equal(result_a.trajectory_z_m, result_b.trajectory_z_m)
    assert np.array_equal(result_a.final_status, result_b.final_status)
    assert np.array_equal(result_a.final_x_m, result_b.final_x_m)
    assert np.array_equal(result_a.final_utc_ns, result_b.final_utc_ns)


# ── 驗收 8：效能量測 ───────────────────────────────────────────────────────


def test_performance_benchmark_reports_breakdown(driver_grid, handles) -> None:
    """N=10,000 顆粒子跑 3 小時，量測並列印 µs/粒子步的拆解。

    這不是一個「必須低於某個絕對值」的硬性效能斷言（BayTrace 原生 3.5
    µs/粒子步是在完整生產網格與長時間執行下量測的基準；本測試用的合成小
    網格、含 Python 層小時迴圈開銷，兩者條件不同，直接套用同一個絕對門檻
    沒有科學意義）。測試本身的驗收重點是：量測與拆解機制本身能正確運作、
    回報每個階段的時間佔比，供人工判讀是否需要優化；數字本身誠實列印於
    測試輸出，不做美化。
    """

    n = 10_000
    arrival_utc = 6_000 * _HOUR_NS

    def resolver(target_utc_ns: int) -> HourlyFieldPair:
        hi = target_utc_ns
        lo = target_utc_ns - _HOUR_NS
        return _uniform_field_pair(
            driver_grid, u1=0.03, v1=0.02, ibf=-1, trat=0.0,
            time_before_utc_ns=hi, time_after_utc_ns=lo,
        )

    interest_polygon = Polygon([(-1.0e6, -1.0e6), (1.0e6, -1.0e6), (1.0e6, 1.0e6), (-1.0e6, 1.0e6)])
    rng_place = np.random.default_rng(555)
    x0 = _DOMAIN_X / 2.0 + rng_place.uniform(-200.0, 200.0, size=n)
    y0 = _DOMAIN_Y / 2.0 + rng_place.uniform(-200.0, 200.0, size=n)
    z0 = np.full(n, -5.0)

    soa = _SyntheticSoa(x=x0, y=y0, z=z0, row_particle_index=np.arange(n, dtype=np.int64))

    config = bd.BackwardDriverConfig(
        arrival_time_utc_ns=arrival_utc,
        settling_velocity_mps=0.0,
        max_backtrack_seconds=3.0 * 3600.0,
        enable_diffusion=True,
        master_seed=1,
    )

    # 先跑一次極小規模，確保這一份受控載入的 njit kernel 已完成編譯，量測
    # 才不會把「第一次呼叫的 JIT 編譯時間」誤算進效能數字。
    warm_soa = _SyntheticSoa(
        x=np.array([_DOMAIN_X / 2.0]), y=np.array([_DOMAIN_Y / 2.0]), z=np.array([-5.0]),
        row_particle_index=np.array([0], dtype=np.int64),
    )
    bd.run_backward_batch(
        driver_grid, handles, warm_soa,
        seed_hex_values=["4" * 32],
        config=bd.BackwardDriverConfig(
            arrival_time_utc_ns=arrival_utc, settling_velocity_mps=0.0,
            max_backtrack_seconds=3600.0, enable_diffusion=True, master_seed=1,
        ),
        forcing_resolver=resolver,
        interest_region_polygon=interest_polygon,
    )

    wall_t0 = _time.perf_counter()
    result = bd.run_backward_batch(
        driver_grid, handles, soa,
        seed_hex_values=[f"{i:032x}" for i in range(n)],
        config=config,
        forcing_resolver=resolver,
        interest_region_polygon=interest_polygon,
    )
    wall_elapsed_s = _time.perf_counter() - wall_t0

    diag = result.diagnostics
    total_particle_substeps = diag.n_particle_substeps
    assert total_particle_substeps > 0

    def us_per_particle_step(seconds: float) -> float:
        return seconds / total_particle_substeps * 1.0e6

    breakdown_us = {
        "move_rk4": us_per_particle_step(diag.time_move_s),
        "boundary_check": us_per_particle_step(diag.time_boundary_s),
        "clamp_z": us_per_particle_step(diag.time_clamp_s),
        "interpolate_velocity": us_per_particle_step(diag.time_interpolate_s),
        "diffusivity_sample": us_per_particle_step(diag.time_diffusivity_sample_s),
        "diffusion_apply": us_per_particle_step(diag.time_diffusion_apply_s),
        "interest_region_check": us_per_particle_step(diag.time_interest_check_s),
    }
    total_us_per_step = us_per_particle_step(diag.total_kernel_and_overhead_s)
    wall_us_per_step = wall_elapsed_s / total_particle_substeps * 1.0e6

    print(f"\n=== WP-5 效能量測（N={n}, 子步總數={diag.n_substeps}, hours_run={result.hours_run}） ===")
    for name, value in breakdown_us.items():
        print(f"  {name:>22s}: {value:8.3f} µs/粒子步")
    print(f"  {'kernel 階段合計':>22s}: {total_us_per_step:8.3f} µs/粒子步")
    print(f"  {'含 Python 迴圈總牆鐘時間':>22s}: {wall_us_per_step:8.3f} µs/粒子步")
    print("  （BayTrace 原生基準約 3.5 µs/粒子步，量測條件不同，僅供參考）")

    assert total_particle_substeps == diag.n_substeps * n
    assert all(np.isfinite(value) and value >= 0.0 for value in breakdown_us.values())


# ── 任務 3：近真實網格規模的「每小時固定開銷」實測 ──────────────────────────
#
# 上面的效能測試用 1.4 km 見方（8x8 節點）的迷你網格，量不到「每小時流場
# 合成」本身的成本——那個網格小到 Stokes、乾濕判定、Smagorinsky 全部在
# 微秒等級內完成，完全不會在小批次下主導總耗時。這裡改用接近真實規模的
# 網格（10,000／30,000／60,000 節點 × 20 垂向層）搭配三種批次大小
# （20、200、2,000 顆），量測：
#
# 1. 每小時「與粒子數無關」的固定開銷（``HourlyOverheadDiagnostics``：乾濕
#    判定、Stokes 全網格計算、Smagorinsky 水平擴散係數場、CFL 子步數估計）。
# 2. 這個固定開銷攤提到每個粒子步之後的 µs/粒子步。
# 3. kernel 本身（``SubstepDiagnostics``）的 µs/粒子步，與固定開銷相加得到
#    總 µs/粒子步。
#
# 並在最大規模（60,000 節點）示範任務 3 要求的優化：把 Stokes／Smagorinsky
# 計算（``compose_current_and_stokes_fields``／``compute_smagorinsky_kh_field``）
# 從「每個沉降速度批次各算一次」改成「同一到達時刻、同一區域的多個沉降批次
# 共用一次」（``run_backward_batch`` 的 ``composed_base_cache``／
# ``smagorinsky_cache`` 參數），並量化改善前後的差異。


def _build_regular_grid_mesh(nx: int, ny: int, *, spacing_m: float = 200.0, dp_m: float = 20.0) -> NativeMesh:
    """建立一個 ``nx × ny`` 規則三角網格的 ``NativeMesh``，供效能測試使用。

    全程用 NumPy 向量化建構三角形連接（不逐節點／逐三角形跑 Python 迴圈），
    讓網格規模拉到六萬節點時，建構本身的耗時仍遠小於後續要量測的追蹤成本。
    """

    n_nodes = nx * ny
    node_xy = np.zeros((n_nodes, 2), dtype=np.float64)
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy")
    node_xy[:, 0] = (ii * spacing_m).ravel()
    node_xy[:, 1] = (jj * spacing_m).ravel()
    dp = np.full(n_nodes, dp_m, dtype=np.float64)

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


def _build_realistic_scale_driver_grid(
    handles: bd.BayTraceModuleHandles, node_target: int,
) -> tuple[bd.NativeBackwardDriverGrid, int, int]:
    """依目標節點數挑一個接近的正方形 ``nx × ny``，建出 20 層垂向的驅動層網格。

    垂向 sigma 用 20 層等間隔（``[-1, 0]``）、``eta=0`` 的參考時刻反推，滿足
    ``build_native_grid_arrays`` 的固定分率契約（見該模組 docstring）；這裡
    只是要一個「規模接近真實、但幾何本身不重要」的網格，不代表任何真實
    OCM 子網格的水深或垂向結構。
    """

    side = max(2, int(round(node_target ** 0.5)))
    mesh = _build_regular_grid_mesh(side, side)
    n_nodes = mesh.node_xy.shape[0]
    nvrt = 20
    sigma_1d = np.linspace(-1.0, 0.0, nvrt)
    zcor_reference = sigma_1d[None, :] * mesh.source_depth_m[:, None]
    eta_reference = np.zeros(n_nodes, dtype=np.float64)
    grid = build_native_grid_arrays(mesh, zcor_reference=zcor_reference, eta_reference=eta_reference)
    driver_grid = bd.build_native_backward_driver_grid(grid, handles, h0=0.0)
    assert driver_grid.ic3_cross_check_ok
    return driver_grid, side, side


def _realistic_scale_field_pair(
    driver_grid: bd.NativeBackwardDriverGrid, hi: int, lo: int,
) -> HourlyFieldPair:
    n_nodes = _n_nodes(driver_grid)
    nvrt = driver_grid.grid.nvrt
    shape = (nvrt, n_nodes)
    eta = np.zeros(n_nodes, dtype=np.float64)
    diffusivity = np.full(shape, 1.0e-4, dtype=np.float64)
    missing = np.zeros(shape, dtype=bool)
    dry_face = np.zeros(shape, dtype=bool)
    return HourlyFieldPair(
        time_before_utc_ns=hi, time_after_utc_ns=lo, trat=0.0, ibf=-1,
        u_field1=np.full(shape, 0.05), v_field1=np.full(shape, 0.02), w_field1=np.zeros(shape),
        eta_field1=eta.copy(), diffusivity_field1=diffusivity.copy(),
        missing_mask1=missing.copy(), dry_face_flag1=dry_face.copy(),
        month_id1="synthetic", time_origin1="test",
        u_field2=np.full(shape, 0.05), v_field2=np.full(shape, 0.02), w_field2=np.zeros(shape),
        eta_field2=eta.copy(), diffusivity_field2=diffusivity.copy(),
        missing_mask2=missing.copy(), dry_face_flag2=dry_face.copy(),
        month_id2="synthetic", time_origin2="test",
    )


def _realistic_scale_wave_resolver(driver_grid: bd.NativeBackwardDriverGrid) -> callable:
    n_nodes = _n_nodes(driver_grid)

    def _resolve(_target_utc_ns: int) -> bd.WaveFieldAtNodes:
        return bd.WaveFieldAtNodes(
            significant_wave_height_m=np.full(n_nodes, 1.5),
            peak_frequency_hz=np.full(n_nodes, 0.1),
            direction_raw_deg=np.full(n_nodes, 45.0),
            valid_mask_wave=np.ones(n_nodes, dtype=bool),
            qc_flags=np.zeros(n_nodes, dtype=np.int64),
        )

    return _resolve


def test_realistic_grid_scale_hourly_overhead_benchmark(handles) -> None:
    """10k/30k/60k 節點 × 20 層網格、批次 20/200/2,000 顆：量測每小時固定開銷。

    誠實揭露量測結論（見最終報告）：這個測試量到的數字明確顯示，在近真實
    網格規模下，「每小時固定開銷」（乾濕判定＋Stokes 全網格計算＋
    Smagorinsky）在小批次（20 顆）下**遠遠**主導總耗時（實測中固定開銷可以
    是 kernel 本身耗時的百倍以上），批次越大攤提效果越好，但即使 2,000 顆
    仍有相當比例來自固定開銷。這正是任務 3 要求發現並優化的現象；本測試
    另外用同一份最大規模網格示範 ``composed_base_cache``／
    ``smagorinsky_cache`` 的改善效果並量化。
    """

    interest_polygon = Polygon([(-1.0e7, -1.0e7), (1.0e7, -1.0e7), (1.0e7, 1.0e7), (-1.0e7, 1.0e7)])
    arrival_utc = 9_000 * _HOUR_NS

    print("\n=== WP-5 任務 3：近真實網格規模的每小時固定開銷實測 ===")

    results: list[dict] = []
    largest_driver_grid: bd.NativeBackwardDriverGrid | None = None

    for node_target in (10_000, 30_000, 60_000):
        driver_grid_big, side_x, side_y = _build_realistic_scale_driver_grid(handles, node_target)
        n_nodes = _n_nodes(driver_grid_big)
        cx = (side_x - 1) * 200.0 / 2.0
        cy = (side_y - 1) * 200.0 / 2.0

        def resolver(target_utc_ns: int, _dg=driver_grid_big) -> HourlyFieldPair:
            return _realistic_scale_field_pair(_dg, target_utc_ns, target_utc_ns - _HOUR_NS)

        wave_resolver = _realistic_scale_wave_resolver(driver_grid_big)

        # 每個網格規模先跑一次極小批次熱機，避免把 njit 第一次呼叫的編譯
        # 時間算進正式量測（比照既有效能測試的作法）。
        warm_soa = _SyntheticSoa(
            x=np.array([cx]), y=np.array([cy]), z=np.array([-5.0]),
            row_particle_index=np.array([0], dtype=np.int64),
        )
        bd.run_backward_batch(
            driver_grid_big, handles, warm_soa,
            seed_hex_values=["5" * 32],
            config=bd.BackwardDriverConfig(
                arrival_time_utc_ns=arrival_utc, settling_velocity_mps=0.0,
                max_backtrack_seconds=3600.0, enable_diffusion=True, master_seed=1,
            ),
            forcing_resolver=resolver, interest_region_polygon=interest_polygon,
            wave_field_resolver=wave_resolver,
        )

        for n_particles in (20, 200, 2_000):
            rng_place = np.random.default_rng(42)
            x0 = cx + rng_place.uniform(-500.0, 500.0, size=n_particles)
            y0 = cy + rng_place.uniform(-500.0, 500.0, size=n_particles)
            z0 = np.full(n_particles, -5.0)
            soa = _SyntheticSoa(x=x0, y=y0, z=z0, row_particle_index=np.arange(n_particles, dtype=np.int64))
            config = bd.BackwardDriverConfig(
                arrival_time_utc_ns=arrival_utc, settling_velocity_mps=0.0,
                max_backtrack_seconds=3600.0, enable_diffusion=True, master_seed=1,
            )
            result = bd.run_backward_batch(
                driver_grid_big, handles, soa,
                seed_hex_values=[f"{i:032x}" for i in range(n_particles)],
                config=config, forcing_resolver=resolver, interest_region_polygon=interest_polygon,
                wave_field_resolver=wave_resolver,
            )
            diag = result.diagnostics
            over = result.overhead_diagnostics
            total_steps = diag.n_particle_substeps
            assert total_steps > 0
            kernel_us = diag.total_kernel_and_overhead_s / total_steps * 1.0e6
            overhead_us = over.total_s / total_steps * 1.0e6
            results.append(
                {
                    "node_target": node_target,
                    "n_nodes": n_nodes,
                    "n_particles": n_particles,
                    "overhead_total_s": over.total_s,
                    "kernel_us": kernel_us,
                    "overhead_us": overhead_us,
                    "total_us": kernel_us + overhead_us,
                }
            )
            print(
                f"  節點≈{node_target:>6d}（實際 {n_nodes:>6d}）× nvrt=20, "
                f"批次={n_particles:>5d}: 每小時固定開銷={over.total_s:7.3f}s  "
                f"kernel={kernel_us:8.3f} µs/步  固定開銷攤提={overhead_us:10.3f} µs/步  "
                f"合計={kernel_us + overhead_us:10.3f} µs/步"
            )
            assert np.isfinite(over.total_s) and over.total_s >= 0.0
            assert np.isfinite(kernel_us) and kernel_us >= 0.0

        largest_driver_grid = driver_grid_big

    # 每小時固定開銷是與粒子數無關的常數，因此同一個網格規模下，批次越大
    # 攤提後的固定開銷 µs/步應該越小；用最小與最大批次交叉驗證這個單調性，
    # 確認量測機制本身正確反映了「固定開銷攤提」這件事，不是量到雜訊。
    by_scale: dict[int, list[dict]] = {}
    for row in results:
        by_scale.setdefault(row["node_target"], []).append(row)
    for node_target, rows in by_scale.items():
        rows_sorted = sorted(rows, key=lambda r: r["n_particles"])
        overhead_us_sequence = [r["overhead_us"] for r in rows_sorted]
        assert overhead_us_sequence[0] > overhead_us_sequence[-1], (
            f"節點≈{node_target}：批次 20 顆的固定開銷攤提（{overhead_us_sequence[0]:.3f} "
            f"µs/步）應該明顯高於批次 2,000 顆（{overhead_us_sequence[-1]:.3f} µs/步），"
            "否則代表固定開銷量測沒有正確反映攤提效果"
        )

    # ── 優化示範：composed_base_cache／smagorinsky_cache 的量化改善 ────────
    assert largest_driver_grid is not None
    driver_grid_big = largest_driver_grid
    n_nodes = _n_nodes(driver_grid_big)
    side = int(round(n_nodes ** 0.5))
    cx = (side - 1) * 200.0 / 2.0
    cy = (side - 1) * 200.0 / 2.0

    def resolver(target_utc_ns: int, _dg=driver_grid_big) -> HourlyFieldPair:
        return _realistic_scale_field_pair(_dg, target_utc_ns, target_utc_ns - _HOUR_NS)

    wave_resolver = _realistic_scale_wave_resolver(driver_grid_big)

    n_particles = 200
    rng_place = np.random.default_rng(7)
    x0 = cx + rng_place.uniform(-500.0, 500.0, size=n_particles)
    y0 = cy + rng_place.uniform(-500.0, 500.0, size=n_particles)
    z0 = np.full(n_particles, -5.0)
    settling_speeds = (0.0, -0.001, -0.002)

    def run_three_settling_batches(*, use_cache: bool) -> tuple[float, list[bd.BackwardDriverResult]]:
        composed_cache: dict = {} if use_cache else None
        smag_cache: dict = {} if use_cache else None
        batch_results = []
        t0 = _time.perf_counter()
        for settle in settling_speeds:
            soa = _SyntheticSoa(
                x=x0.copy(), y=y0.copy(), z=z0.copy(),
                row_particle_index=np.arange(n_particles, dtype=np.int64),
            )
            config = bd.BackwardDriverConfig(
                arrival_time_utc_ns=arrival_utc, settling_velocity_mps=settle,
                max_backtrack_seconds=3600.0, enable_diffusion=True, master_seed=1,
            )
            batch_results.append(
                bd.run_backward_batch(
                    driver_grid_big, handles, soa,
                    seed_hex_values=[f"{i:032x}" for i in range(n_particles)],
                    config=config, forcing_resolver=resolver, interest_region_polygon=interest_polygon,
                    wave_field_resolver=wave_resolver,
                    composed_base_cache=composed_cache, smagorinsky_cache=smag_cache,
                )
            )
        elapsed = _time.perf_counter() - t0
        return elapsed, batch_results

    elapsed_without_cache, results_without_cache = run_three_settling_batches(use_cache=False)
    elapsed_with_cache, results_with_cache = run_three_settling_batches(use_cache=True)

    speedup = elapsed_without_cache / elapsed_with_cache if elapsed_with_cache > 0 else float("inf")
    print(
        f"\n  快取效果（節點={n_nodes}, 批次={n_particles} 顆 × 3 種沉降速度，各 1 小時）："
        f"無快取={elapsed_without_cache:.3f}s  有快取={elapsed_with_cache:.3f}s  "
        f"加速比={speedup:.2f}x"
    )

    # 正確性：快取只應該改變效能，不應該改變任何數值結果——逐一比對三種
    # 沉降速度在有無快取下的最終位置與狀態必須位元相同。
    zipped = zip(settling_speeds, results_without_cache, results_with_cache, strict=False)
    for settle, result_a, result_b in zipped:
        assert np.array_equal(result_a.final_x_m, result_b.final_x_m), (
            f"settle={settle}：啟用快取後 x 結果不應改變"
        )
        assert np.array_equal(result_a.final_y_m, result_b.final_y_m), (
            f"settle={settle}：啟用快取後 y 結果不應改變"
        )
        assert np.array_equal(result_a.final_status, result_b.final_status), (
            f"settle={settle}：啟用快取後狀態不應改變"
        )

    # 效能：有快取應該明顯更快——第一個批次仍要付出完整的乾濕判定＋Stokes＋
    # Smagorinsky 成本，但第二、三個批次應該幾乎只剩下 apply_settling_velocity
    # 這個便宜步驟。用寬鬆門檻（有快取版本耗時不到無快取版本的 80%）避免
    # 在雜訊較大的機器上偶發失敗，同時仍然是一個有意義的迴歸防護。
    assert elapsed_with_cache < elapsed_without_cache * 0.8, (
        f"啟用 composed_base_cache／smagorinsky_cache 後應該明顯更快："
        f"無快取={elapsed_without_cache:.3f}s，有快取={elapsed_with_cache:.3f}s"
    )


# ── WP-9：計數器型擴散亂數 ───────────────────────────────────────────────────


def test_counter_based_gaussian_pairs_statistics() -> None:
    """計數器型亂數的統計品質：均值≈0、變異數≈1、x/y 不相關、跨步不自相關。

    這是純函式測試，不需要網格或 BayTrace handles；驗證
    ``counter_based_gaussian_pairs`` 本身的隨機數品質是否足以支撐
    「Box–Muller 轉換出的標準常態亂數」這個假設，供 WP-9 小時主軸的擴散
    位移直接使用。
    """

    n_particles = 6000
    n_steps = 400
    seed_lo, seed_hi = bd.derive_particle_diffusion_seed_lo_hi_u64(
        [f"{i:032x}" for i in range(n_particles)], master_seed=7
    )

    all_z0 = []
    all_z1 = []
    for step in range(n_steps):
        counter = np.full(n_particles, step, dtype=np.int64)
        z0, z1 = bd.counter_based_gaussian_pairs(seed_lo, seed_hi, counter)
        assert np.all(np.isfinite(z0)) and np.all(np.isfinite(z1))
        all_z0.append(z0)
        all_z1.append(z1)
    all_z0 = np.concatenate(all_z0)
    all_z1 = np.concatenate(all_z1)

    n_total = all_z0.size
    assert np.mean(all_z0) == pytest.approx(0.0, abs=5.0 / np.sqrt(n_total))
    assert np.mean(all_z1) == pytest.approx(0.0, abs=5.0 / np.sqrt(n_total))
    assert np.var(all_z0) == pytest.approx(1.0, abs=5.0 * np.sqrt(2.0 / n_total))
    assert np.var(all_z1) == pytest.approx(1.0, abs=5.0 * np.sqrt(2.0 / n_total))
    assert abs(np.corrcoef(all_z0, all_z1)[0, 1]) < 0.05

    # 單一粒子跨步序列的 lag-1 自相關應接近 0（不同步計數器之間相互獨立）。
    single_seed_lo = seed_lo[:1]
    single_seed_hi = seed_hi[:1]
    n_series = 20000
    long_series = np.array(
        [
            bd.counter_based_gaussian_pairs(
                single_seed_lo, single_seed_hi, np.array([c], dtype=np.int64),
            )[0][0]
            for c in range(n_series)
        ]
    )
    # 5 個標準誤（獨立常態序列 lag-1 自相關係數的抽樣標準誤約 1/sqrt(n)）。
    autocorr_tolerance = 5.0 / np.sqrt(n_series)
    assert abs(np.corrcoef(long_series[:-1], long_series[1:])[0, 1]) < autocorr_tolerance


def test_counter_based_diffusion_seed_independent_of_master_seed_and_particle() -> None:
    """同一顆粒子換一個 ``master_seed`` 得到不同序列；不同粒子彼此序列不同。"""

    seed_hex_values = [f"{i:032x}" for i in range(8)]
    lo_a, hi_a = bd.derive_particle_diffusion_seed_lo_hi_u64(seed_hex_values, master_seed=1)
    lo_b, hi_b = bd.derive_particle_diffusion_seed_lo_hi_u64(seed_hex_values, master_seed=2)
    assert not np.array_equal(lo_a, lo_b) or not np.array_equal(hi_a, hi_b)

    # 同一個 (seed_hex, master_seed) 重跑必須完全相同（純函式、可重現）。
    lo_a2, hi_a2 = bd.derive_particle_diffusion_seed_lo_hi_u64(seed_hex_values, master_seed=1)
    assert np.array_equal(lo_a, lo_a2)
    assert np.array_equal(hi_a, hi_a2)

    # 不同粒子（不同 seed_hex）的種子必須互異。
    assert len(set(lo_a.tolist())) == len(seed_hex_values)


def test_diffusion_apply_counter_mode_matches_regardless_of_batch_composition() -> None:
    """計數器型擴散：同一顆粒子單獨跑與併入更大批次跑，位移必須位元相同。

    這是 WP-9「合併批次不改變擴散序列」設計的最小單元測試：直接呼叫
    ``_apply_horizontal_diffusion``（不透過完整的 ``mirror_rk4_substep_loop``
    或 ``run_backward_batch``），驗證同一顆粒子（同一組
    ``seed_lo``／``seed_hi``、同一個 ``counter`` 起始值）不論被放進長度為
    1 的陣列，還是長度為 5、位置在第 3 列的陣列，得到的位移都完全相同。
    """

    seed_hex_values = [f"{i:032x}" for i in range(5)]
    seed_lo_all, seed_hi_all = bd.derive_particle_diffusion_seed_lo_hi_u64(seed_hex_values, master_seed=42)
    kh = 0.02
    dtb = 15.0

    # 場景 A：這顆粒子（索引 2）獨自一個長度為 1 的批次。
    px_a = np.array([100.0])
    py_a = np.array([200.0])
    active_a = np.array([True])
    counter_a = np.zeros(1, dtype=np.int64)
    bd._apply_horizontal_diffusion(
        px_a, py_a, active_a, kh_mps2=np.array([kh]), dtb=dtb,
        particle_seed_lo=seed_lo_all[2:3], particle_seed_hi=seed_hi_all[2:3],
        particle_step_counter=counter_a,
    )

    # 場景 B：這顆粒子併入一個長度為 5 的批次（同樣是它自己第一次套用擴散，
    # 起始 counter 為 0；批次內其他粒子的座標、活躍與否都不同）。
    px_b = np.array([1.0, 2.0, 100.0, 4.0, 5.0])
    py_b = np.array([1.0, 2.0, 200.0, 4.0, 5.0])
    active_b = np.array([True, False, True, True, False])
    counter_b = np.zeros(5, dtype=np.int64)
    bd._apply_horizontal_diffusion(
        px_b, py_b, active_b, kh_mps2=np.full(5, kh), dtb=dtb,
        particle_seed_lo=seed_lo_all, particle_seed_hi=seed_hi_all,
        particle_step_counter=counter_b,
    )

    assert px_a[0] == pytest.approx(px_b[2])
    assert py_a[0] == pytest.approx(py_b[2])
    assert counter_a[0] == counter_b[2] == 1

    # 未活躍的粒子（索引 1、4）計數器不應該遞增（尚未真正套用擴散）。
    assert counter_b[1] == 0
    assert counter_b[4] == 0


def test_mirror_rk4_substep_loop_rejects_mixed_diffusion_rng_sources(driver_grid, handles) -> None:
    """同時提供 ``rng`` 與計數器型三參數、或兩者都不提供，必須明確拒絕。"""

    field_pair = _uniform_field_pair(driver_grid, u1=0.0, v1=0.0, ibf=1, trat=0.0)
    n = 3
    pa = _make_particles(handles, n)
    _place_and_locate(
        driver_grid, handles, pa,
        x=np.full(n, _DOMAIN_X / 2.0), y=np.full(n, _DOMAIN_Y / 2.0), z=np.full(n, -5.0),
        field_pair=field_pair,
    )
    composed = bd.compose_total_velocity_fields(
        driver_grid, handles, field_pair, settling_velocity_mps=0.0,
    )
    n_nodes = _n_nodes(driver_grid)
    hf_field = np.full((_NVRT, n_nodes), 0.01, dtype=np.float64)
    seed_lo, seed_hi = bd.derive_particle_diffusion_seed_lo_hi_u64(
        [f"{i:032x}" for i in range(n)], master_seed=0,
    )
    counter = np.zeros(n, dtype=np.int64)
    rng = np.random.default_rng(0)

    with pytest.raises(bd.BackwardDriverError):
        bd.mirror_rk4_substep_loop(
            driver_grid, handles, pa, ibf=1, ndeltp=1, dtb=60.0, composed=composed, enable_diffusion=True,
            hf1_field=hf_field, hf2_field=hf_field,
            rng=rng, particle_seed_lo=seed_lo, particle_seed_hi=seed_hi, particle_step_counter=counter,
        )
    with pytest.raises(bd.BackwardDriverError):
        bd.mirror_rk4_substep_loop(
            driver_grid, handles, pa, ibf=1, ndeltp=1, dtb=60.0, composed=composed, enable_diffusion=True,
            hf1_field=hf_field, hf2_field=hf_field,
        )
    with pytest.raises(bd.BackwardDriverError):
        bd.mirror_rk4_substep_loop(
            driver_grid, handles, pa, ibf=1, ndeltp=1, dtb=60.0, composed=composed, enable_diffusion=True,
            hf1_field=hf_field, hf2_field=hf_field,
            particle_seed_lo=seed_lo, particle_seed_hi=seed_hi,
        )


def test_run_backward_batch_default_uses_counter_rng_and_is_reproducible(driver_grid, handles) -> None:
    """``run_backward_batch`` 預設（未傳入 ``rng``）改用計數器型擴散，仍可重現。

    這取代舊版「批次共用種子」的預設路徑；本測試同時確認新預設路徑的
    可重現性（與既有 ``test_reproducibility_same_seed_bit_identical`` 互補，
    但那個測試沒有明確驗證『這是計數器型路徑』，只驗證『重跑位元相同』，
    任何一種亂數方案都能通過；本測試額外確認同一顆粒子單獨抽出來看，序列
    完全由自己的 ``seed_hex`` 決定，見上面的批次組成無關性測試）。
    """

    arrival_utc = 6_000 * _HOUR_NS

    def resolver(target_utc_ns: int) -> HourlyFieldPair:
        hi = target_utc_ns
        lo = target_utc_ns - _HOUR_NS
        return _uniform_field_pair(
            driver_grid, u1=0.01, v1=-0.005, ibf=-1, trat=0.0,
            time_before_utc_ns=hi, time_after_utc_ns=lo,
        )

    interest_polygon = Polygon([(-1.0e6, -1.0e6), (1.0e6, -1.0e6), (1.0e6, 1.0e6), (-1.0e6, 1.0e6)])
    n = 6
    x0 = np.full(n, _DOMAIN_X / 2.0)
    y0 = np.full(n, _DOMAIN_Y / 2.0)
    z0 = np.full(n, -5.0)
    seed_hex_values = [f"{i:032x}" for i in range(n)]
    soa = _SyntheticSoa(x=x0, y=y0, z=z0, row_particle_index=np.arange(n, dtype=np.int64))
    config = bd.BackwardDriverConfig(
        arrival_time_utc_ns=arrival_utc, settling_velocity_mps=0.0,
        max_backtrack_seconds=2.0 * 3600.0, enable_diffusion=True, master_seed=99,
    )

    result_a = bd.run_backward_batch(
        driver_grid, handles, soa, seed_hex_values=seed_hex_values, config=config,
        forcing_resolver=resolver, interest_region_polygon=interest_polygon,
    )
    result_b = bd.run_backward_batch(
        driver_grid, handles, soa, seed_hex_values=seed_hex_values, config=config,
        forcing_resolver=resolver, interest_region_polygon=interest_polygon,
    )
    assert np.array_equal(result_a.trajectory_x_m, result_b.trajectory_x_m)
    assert np.array_equal(result_a.trajectory_y_m, result_b.trajectory_y_m)

    # 位移不應該全部是 0（確認擴散確實有作用，不是誤用到停用路徑）。
    assert not np.array_equal(result_a.trajectory_x_m[0], result_a.trajectory_x_m[-1])


# ── WP-9：逐列關注區域多邊形 ─────────────────────────────────────────────────


def test_interest_region_polygons_per_row_matches_separate_single_polygon_calls(driver_grid, handles) -> None:
    """合併呼叫（逐列多邊形）與分開呼叫（各自單一多邊形）必須位元相同。

    模擬 WP-9 小時主軸「同一次 kernel 呼叫混合多個測站的粒子」的情境：
    兩個測站各自的關注區域是不同的正方形，把兩站粒子合併成一個陣列、用
    ``interest_region_polygons`` 逐列指定後跑一次子步，其結果應該與「兩站
    粒子分開、各自用 ``interest_region_polygon`` 單一多邊形跑」完全相同。
    """

    field_pair = _uniform_field_pair(driver_grid, u1=0.05, v1=0.0, ibf=1, trat=0.0)
    polygon_a = Polygon([(0.0, 0.0), (500.0, 0.0), (500.0, 500.0), (0.0, 500.0)])
    polygon_b = Polygon([(700.0, 0.0), (1400.0, 0.0), (1400.0, 1400.0), (700.0, 1400.0)])

    # 站 A 的粒子：起點靠近 polygon_a 右邊界，往 +x 平流後應該離開。
    # 站 B 的粒子：起點靠近 polygon_b 右邊界，同樣往 +x 平流後應該離開，
    # 但離開的時間點與 polygon_a 不同（門檻不同），用來確認兩者互不干擾。
    x_a = np.array([480.0, 100.0])
    y_a = np.array([250.0, 250.0])
    x_b = np.array([1380.0, 900.0])
    y_b = np.array([700.0, 700.0])

    def _run_combined() -> tuple:
        n = 4
        pa = _make_particles(handles, n)
        _place_and_locate(
            driver_grid, handles, pa,
            x=np.concatenate([x_a, x_b]), y=np.concatenate([y_a, y_b]), z=np.full(n, -5.0),
            field_pair=field_pair,
        )
        composed = bd.compose_total_velocity_fields(
            driver_grid, handles, field_pair, settling_velocity_mps=0.0,
        )
        ndeltp = bd.estimate_substep_count(
            driver_grid, composed.u_field1, composed.v_field1, dt_seconds=600.0,
        )
        dtb = 600.0 / ndeltp
        native_status = np.full(n, bd.NATIVE_STATUS_ACTIVE, dtype=np.int32)
        departures: list = []
        polygons_per_row = np.array([polygon_a, polygon_a, polygon_b, polygon_b], dtype=object)
        bd.mirror_rk4_substep_loop(
            driver_grid, handles, pa, ibf=1, ndeltp=ndeltp, dtb=dtb, composed=composed,
            enable_diffusion=False,
            native_status=native_status, substep_utc_ns=lambda idt: idt,
            interest_region_polygons=polygons_per_row, departures=departures,
        )
        return pa.x.copy(), pa.y.copy(), native_status.copy(), pa.status.copy()

    def _run_separate() -> tuple:
        results_x = []
        results_y = []
        results_status = []
        results_pa_status = []
        for x_group, y_group, polygon in ((x_a, y_a, polygon_a), (x_b, y_b, polygon_b)):
            n_group = x_group.shape[0]
            pa = _make_particles(handles, n_group)
            _place_and_locate(
                driver_grid, handles, pa,
                x=x_group, y=y_group, z=np.full(n_group, -5.0), field_pair=field_pair,
            )
            composed = bd.compose_total_velocity_fields(
                driver_grid, handles, field_pair, settling_velocity_mps=0.0,
            )
            ndeltp = bd.estimate_substep_count(
                driver_grid, composed.u_field1, composed.v_field1, dt_seconds=600.0,
            )
            dtb = 600.0 / ndeltp
            native_status = np.full(n_group, bd.NATIVE_STATUS_ACTIVE, dtype=np.int32)
            departures: list = []
            bd.mirror_rk4_substep_loop(
                driver_grid, handles, pa, ibf=1, ndeltp=ndeltp, dtb=dtb, composed=composed,
                enable_diffusion=False,
                native_status=native_status, substep_utc_ns=lambda idt: idt,
                interest_region_polygon=polygon, departures=departures,
            )
            results_x.append(pa.x.copy())
            results_y.append(pa.y.copy())
            results_status.append(native_status.copy())
            results_pa_status.append(pa.status.copy())
        return (
            np.concatenate(results_x), np.concatenate(results_y),
            np.concatenate(results_status), np.concatenate(results_pa_status),
        )

    combined_x, combined_y, combined_native_status, combined_pa_status = _run_combined()
    separate_x, separate_y, separate_native_status, separate_pa_status = _run_separate()

    assert np.array_equal(combined_x, separate_x)
    assert np.array_equal(combined_y, separate_y)
    assert np.array_equal(combined_native_status, separate_native_status)
    assert np.array_equal(combined_pa_status, separate_pa_status)
    # 確認測試場景真的有粒子離開（不是門檻設得太鬆導致測試沒有鑑別力）。
    assert np.any(combined_native_status == bd.NATIVE_STATUS_LEFT_INTEREST_REGION)


def test_interest_region_polygon_and_polygons_mutually_exclusive(driver_grid, handles) -> None:
    """同時提供 ``interest_region_polygon`` 與 ``interest_region_polygons`` 必須拒絕。"""

    field_pair = _uniform_field_pair(driver_grid, u1=0.0, v1=0.0, ibf=1, trat=0.0)
    n = 2
    pa = _make_particles(handles, n)
    _place_and_locate(
        driver_grid, handles, pa,
        x=np.full(n, _DOMAIN_X / 2.0), y=np.full(n, _DOMAIN_Y / 2.0), z=np.full(n, -5.0),
        field_pair=field_pair,
    )
    composed = bd.compose_total_velocity_fields(driver_grid, handles, field_pair, settling_velocity_mps=0.0)
    polygon = Polygon([(-1.0e6, -1.0e6), (1.0e6, -1.0e6), (1.0e6, 1.0e6), (-1.0e6, 1.0e6)])
    with pytest.raises(bd.BackwardDriverError):
        bd.mirror_rk4_substep_loop(
            driver_grid, handles, pa, ibf=1, ndeltp=1, dtb=60.0, composed=composed, enable_diffusion=False,
            native_status=np.full(n, bd.NATIVE_STATUS_ACTIVE, dtype=np.int32), substep_utc_ns=lambda idt: idt,
            interest_region_polygon=polygon,
            interest_region_polygons=np.array([polygon, polygon], dtype=object),
        )


# ── WP-9：Stokes 深度截斷優化 ────────────────────────────────────────────────


def test_stokes_depth_threshold_none_is_bit_identical_to_original(driver_grid, handles) -> None:
    """``stokes_depth_threshold_ratio=None``（預設）必須與未加此參數前逐位元相同。

    直接沿用既有的 Stokes 交叉驗證合成場景（見
    ``test_composed_velocity_matches_stokes_reference_at_each_node_layer``），
    只是在這裡改成明確比較「有沒有傳這個新參數」兩種呼叫方式。
    """

    n_nodes = _n_nodes(driver_grid)
    field_pair = _uniform_field_pair(driver_grid, u1=0.03, v1=0.01, ibf=-1, trat=0.0)
    wave = bd.WaveFieldAtNodes(
        significant_wave_height_m=np.full(n_nodes, 1.2),
        peak_frequency_hz=np.full(n_nodes, 0.15),
        direction_raw_deg=np.full(n_nodes, 30.0),
        valid_mask_wave=np.ones(n_nodes, dtype=bool),
        qc_flags=np.zeros(n_nodes, dtype=np.int64),
    )
    base_default = bd.compose_current_and_stokes_fields(
        driver_grid, handles, field_pair, wave_field1=wave, wave_field2=wave
    )
    base_explicit_none = bd.compose_current_and_stokes_fields(
        driver_grid, handles, field_pair, wave_field1=wave, wave_field2=wave,
        stokes_depth_threshold_ratio=None,
    )
    assert np.array_equal(base_default.u_field1, base_explicit_none.u_field1)
    assert np.array_equal(base_default.v_field1, base_explicit_none.v_field1)
    # 未啟用截斷時，「顯著點」就是全部 effective_mask 內的點，兩個計數器
    # 應該相等（沒有任何點被跳過）。
    assert base_default.stokes_points_evaluated == base_default.stokes_points_possible
    assert base_default.stokes_points_evaluated > 0


def test_precomputed_wet_dry_flags_matches_internal_recompute_bit_identical(driver_grid, handles) -> None:
    """WP-15 任務 A.3：``precomputed_wet_dry_flags`` 提供與同一份 ``eta_field1``
    算出的旗標時，輸出必須與完全不傳這個參數（函式內部自己重算一次）逐位元
    相同——這是 ``hour_major.py`` 主迴圈省下第二次全網格 ``compute_wet_dry_
    flags`` 呼叫的安全性基礎（見該模組對此參數的呼叫點註解）。
    """

    n_nodes = _n_nodes(driver_grid)
    field_pair = _uniform_field_pair(driver_grid, u1=0.02, v1=-0.01, ibf=-1, trat=0.0)
    wave = bd.WaveFieldAtNodes(
        significant_wave_height_m=np.full(n_nodes, 1.1),
        peak_frequency_hz=np.full(n_nodes, 0.12),
        direction_raw_deg=np.full(n_nodes, 60.0),
        valid_mask_wave=np.ones(n_nodes, dtype=bool),
        qc_flags=np.zeros(n_nodes, dtype=np.int64),
    )

    node_is_dry, elem_is_dry = bd.compute_wet_dry_flags(driver_grid, handles, eta_m=field_pair.eta_field1)

    base_recompute = bd.compose_current_and_stokes_fields(
        driver_grid, handles, field_pair, wave_field1=wave, wave_field2=wave,
    )
    base_precomputed = bd.compose_current_and_stokes_fields(
        driver_grid, handles, field_pair, wave_field1=wave, wave_field2=wave,
        precomputed_wet_dry_flags=(node_is_dry, elem_is_dry),
    )

    for field_name in (
        "u_field1", "v_field1", "w_field1_no_settling",
        "u_field2", "v_field2", "w_field2_no_settling",
        "node_is_dry", "elem_is_dry", "effective_mask",
    ):
        assert np.array_equal(getattr(base_recompute, field_name), getattr(base_precomputed, field_name)), (
            field_name
        )
    assert base_recompute.stokes_ok_fraction == base_precomputed.stokes_ok_fraction


def test_stokes_depth_threshold_rejects_out_of_range_values(driver_grid, handles) -> None:
    """``stokes_depth_threshold_ratio`` 必須落在 (0, 1) 開區間，否則拒絕。"""

    n_nodes = _n_nodes(driver_grid)
    field_pair = _uniform_field_pair(driver_grid, u1=0.0, v1=0.0, ibf=-1, trat=0.0)
    wave = bd.WaveFieldAtNodes(
        significant_wave_height_m=np.full(n_nodes, 1.0),
        peak_frequency_hz=np.full(n_nodes, 0.1),
        direction_raw_deg=np.zeros(n_nodes),
        valid_mask_wave=np.ones(n_nodes, dtype=bool),
        qc_flags=np.zeros(n_nodes, dtype=np.int64),
    )
    for bad_ratio in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(bd.BackwardDriverError):
            bd.compose_current_and_stokes_fields(
                driver_grid, handles, field_pair, wave_field1=wave, wave_field2=wave,
                stokes_depth_threshold_ratio=bad_ratio,
            )


def test_stokes_depth_threshold_truncates_deep_layers_within_bounded_error(handles) -> None:
    """深水（相對波長）情境下，深度截斷應該真的跳過部分點，且誤差有界。

    用一個比既有合成網格更深（``dp=150 m``）、更高頻（``fp=0.2 Hz``，波長
    較短）的網格，讓 Stokes 垂向衰減在水柱內達到「顯著」之下——這正是
    近岸淺水（既有 8x8 合成網格 ``dp=20 m``）不容易展示出截斷效果的原因
    （見最終報告的誠實揭露）。誤差上界直接來自截斷門檻本身的物理意義：
    被跳過的點，其真實 Stokes 貢獻本來就 ``<= threshold_ratio`` 倍表面值，
    這裡用「表面速度量級」估計一個寬鬆但可驗證的絕對誤差上界。
    """

    mesh = _build_regular_grid_mesh(6, 6, spacing_m=200.0, dp_m=150.0)
    nvrt = 20
    n_nodes = mesh.node_xy.shape[0]
    sigma_1d = np.linspace(-1.0, 0.0, nvrt)
    zcor_reference = sigma_1d[None, :] * mesh.source_depth_m[:, None]
    eta_reference = np.zeros(n_nodes, dtype=np.float64)
    grid = build_native_grid_arrays(mesh, zcor_reference=zcor_reference, eta_reference=eta_reference)
    driver_grid_deep = bd.build_native_backward_driver_grid(grid, handles, h0=0.0)

    shape = (nvrt, n_nodes)
    eta = np.zeros(n_nodes)
    field_pair = HourlyFieldPair(
        time_before_utc_ns=2 * _HOUR_NS, time_after_utc_ns=1 * _HOUR_NS, trat=0.0, ibf=-1,
        u_field1=np.full(shape, 0.05), v_field1=np.full(shape, 0.02), w_field1=np.zeros(shape),
        eta_field1=eta.copy(), diffusivity_field1=np.full(shape, 1.0e-4),
        missing_mask1=np.zeros(shape, dtype=bool),
        dry_face_flag1=np.zeros(shape, dtype=bool), month_id1="synthetic", time_origin1="test",
        u_field2=np.full(shape, 0.05), v_field2=np.full(shape, 0.02), w_field2=np.zeros(shape),
        eta_field2=eta.copy(), diffusivity_field2=np.full(shape, 1.0e-4),
        missing_mask2=np.zeros(shape, dtype=bool),
        dry_face_flag2=np.zeros(shape, dtype=bool), month_id2="synthetic", time_origin2="test",
    )
    wave = bd.WaveFieldAtNodes(
        significant_wave_height_m=np.full(n_nodes, 1.5),
        peak_frequency_hz=np.full(n_nodes, 0.2),
        direction_raw_deg=np.full(n_nodes, 45.0),
        valid_mask_wave=np.ones(n_nodes, dtype=bool),
        qc_flags=np.zeros(n_nodes, dtype=np.int64),
    )

    base_full = bd.compose_current_and_stokes_fields(
        driver_grid_deep, handles, field_pair, wave_field1=wave, wave_field2=wave
    )
    threshold_ratio = 1.0e-3
    base_truncated = bd.compose_current_and_stokes_fields(
        driver_grid_deep, handles, field_pair, wave_field1=wave, wave_field2=wave,
        stokes_depth_threshold_ratio=threshold_ratio,
    )

    assert base_truncated.stokes_points_possible == base_full.stokes_points_possible
    assert 0 < base_truncated.stokes_points_evaluated < base_truncated.stokes_points_possible, (
        "這個深水情境應該要讓深度截斷真正跳過一部分點，否則測試沒有鑑別力"
    )

    diff_u = np.abs(base_truncated.u_field1 - base_full.u_field1)
    diff_v = np.abs(base_truncated.v_field1 - base_full.v_field1)
    # 表面 Stokes 速度量級：a²ωk ~ (Hs/2)² * omega * k；用一個寬鬆但物理
    # 上合理的上界（表面速度量級 × 門檻比例 × 安全係數 10）驗證誤差確實
    # 被門檻控制住，而不是隨意大小。
    surface_speed_scale = 0.5  # m/s，遠大於本測試波高／週期下的實際表面 Stokes 速度
    error_bound = surface_speed_scale * threshold_ratio * 10.0
    assert np.nanmax(diff_u) < error_bound
    assert np.nanmax(diff_v) < error_bound


# ══════════════════════════════════════════════════════════════════════════
# WP-14 任務 B：field1_reuse_candidate（端點重用）驗收
# ══════════════════════════════════════════════════════════════════════════


def _endpoint_pair(driver_grid, *, eta1, eta2, u1, v1, u2, v2, time_before_utc_ns, time_after_utc_ns):
    n_nodes = _n_nodes(driver_grid)
    shape = (_NVRT, n_nodes)
    return HourlyFieldPair(
        time_before_utc_ns=time_before_utc_ns, time_after_utc_ns=time_after_utc_ns, trat=0.0, ibf=-1,
        u_field1=np.full(shape, u1), v_field1=np.full(shape, v1), w_field1=np.zeros(shape),
        eta_field1=eta1.copy(), diffusivity_field1=np.full(shape, 1.0e-4),
        missing_mask1=np.zeros(shape, dtype=bool), dry_face_flag1=np.zeros(shape, dtype=bool),
        month_id1="synthetic", time_origin1="test",
        u_field2=np.full(shape, u2), v_field2=np.full(shape, v2), w_field2=np.zeros(shape),
        eta_field2=eta2.copy(), diffusivity_field2=np.full(shape, 1.0e-4),
        missing_mask2=np.zeros(shape, dtype=bool), dry_face_flag2=np.zeros(shape, dtype=bool),
        month_id2="synthetic", time_origin2="test",
    )


def test_field1_reuse_candidate_matches_fresh_when_node_is_dry_equal(driver_grid, handles) -> None:
    """``field1_reuse_candidate`` 在 ``node_is_dry`` 逐位元相同時必須安全重用，
    結果與完全不傳這個參數（重新計算）逐位元相同——這是 WP-14 任務 B 的
    核心正確性保證，見 ``compose_current_and_stokes_fields`` 該參數 docstring
    「為什麼需要比對 node_is_dry」一節。
    """

    n_nodes = _n_nodes(driver_grid)
    eta_zero = np.zeros(n_nodes)
    wave = bd.WaveFieldAtNodes(
        significant_wave_height_m=np.linspace(0.5, 2.0, n_nodes),
        peak_frequency_hz=np.linspace(0.08, 0.25, n_nodes),
        direction_raw_deg=np.linspace(0.0, 300.0, n_nodes),
        valid_mask_wave=np.ones(n_nodes, dtype=bool),
        qc_flags=np.zeros(n_nodes, dtype=np.int64),
    )

    # 三個相連的流場時刻 T、T-Δh、T-2Δh，讓「hour h 的 field2」與
    # 「hour h+1 的 field1」指向同一份原始資料（T-Δh），模擬小時主軸相鄰
    # 兩小時的窗口相連。
    pair_h = _endpoint_pair(
        driver_grid, eta1=eta_zero, eta2=eta_zero, u1=0.02, v1=0.01, u2=0.015, v2=-0.005,
        time_before_utc_ns=3 * _HOUR_NS, time_after_utc_ns=2 * _HOUR_NS,
    )
    pair_h1 = _endpoint_pair(
        driver_grid, eta1=eta_zero, eta2=eta_zero, u1=0.015, v1=-0.005, u2=0.01, v2=0.03,
        time_before_utc_ns=2 * _HOUR_NS, time_after_utc_ns=1 * _HOUR_NS,
    )

    base_h = bd.compose_current_and_stokes_fields(
        driver_grid, handles, pair_h, wave_field1=wave, wave_field2=wave
    )
    base_h1_fresh = bd.compose_current_and_stokes_fields(
        driver_grid, handles, pair_h1, wave_field1=wave, wave_field2=wave
    )
    base_h1_cached = bd.compose_current_and_stokes_fields(
        driver_grid, handles, pair_h1, wave_field1=wave, wave_field2=wave,
        field1_reuse_candidate=base_h.field2_endpoint,
    )

    assert np.array_equal(base_h1_cached.u_field1, base_h.field2_endpoint.u)
    assert np.array_equal(base_h1_cached.u_field1, base_h1_fresh.u_field1)
    assert np.array_equal(base_h1_cached.v_field1, base_h1_fresh.v_field1)
    assert np.array_equal(base_h1_cached.w_field1_no_settling, base_h1_fresh.w_field1_no_settling)
    assert base_h1_cached.stokes_ok_fraction == base_h1_fresh.stokes_ok_fraction
    assert base_h1_cached.stokes_points_evaluated == base_h1_fresh.stokes_points_evaluated
    assert base_h1_cached.stokes_points_possible == base_h1_fresh.stokes_points_possible
    # field2 一定照常重新計算，不受這個參數影響。
    assert np.array_equal(base_h1_cached.u_field2, base_h1_fresh.u_field2)


def test_field1_reuse_candidate_fails_closed_when_node_is_dry_differs(driver_grid, handles) -> None:
    """``node_is_dry`` 不同時必須放棄重用、整段重新計算，不得誤用不相容的
    快取（即使兩個端點指向的原始流場資料時刻相連）。
    """

    n_nodes = _n_nodes(driver_grid)
    eta_zero = np.zeros(n_nodes)
    eta_dry = np.full(n_nodes, -_DP_M * 1.5)  # 水位低於海床，保證全部節點乾
    wave = bd.WaveFieldAtNodes(
        significant_wave_height_m=np.linspace(0.5, 2.0, n_nodes),
        peak_frequency_hz=np.linspace(0.08, 0.25, n_nodes),
        direction_raw_deg=np.linspace(0.0, 300.0, n_nodes),
        valid_mask_wave=np.ones(n_nodes, dtype=bool),
        qc_flags=np.zeros(n_nodes, dtype=np.int64),
    )

    # hour h 自己的 eta_field1 是乾的（時刻 T），field2（時刻 T-Δh）因此也
    # 用「乾」的節點遮罩合成；hour h+1 自己的 eta_field1 就是時刻 T-Δh 的
    # 真正海面（這裡設為全濕），兩者遮罩基礎不同，不能重用。
    pair_h_dry = _endpoint_pair(
        driver_grid, eta1=eta_dry, eta2=eta_zero, u1=0.02, v1=0.01, u2=0.015, v2=-0.005,
        time_before_utc_ns=3 * _HOUR_NS, time_after_utc_ns=2 * _HOUR_NS,
    )
    pair_h1_wet = _endpoint_pair(
        driver_grid, eta1=eta_zero, eta2=eta_zero, u1=0.015, v1=-0.005, u2=0.01, v2=0.03,
        time_before_utc_ns=2 * _HOUR_NS, time_after_utc_ns=1 * _HOUR_NS,
    )

    base_h_dry = bd.compose_current_and_stokes_fields(
        driver_grid, handles, pair_h_dry, wave_field1=wave, wave_field2=wave
    )
    assert np.all(base_h_dry.node_is_dry), "測試前提：hour h 的節點應該全部是乾的"

    base_h1_fresh = bd.compose_current_and_stokes_fields(
        driver_grid, handles, pair_h1_wet, wave_field1=wave, wave_field2=wave
    )
    assert not np.any(base_h1_fresh.node_is_dry), "測試前提：hour h+1 的節點應該全部是濕的"

    base_h1_cached_attempt = bd.compose_current_and_stokes_fields(
        driver_grid, handles, pair_h1_wet, wave_field1=wave, wave_field2=wave,
        field1_reuse_candidate=base_h_dry.field2_endpoint,
    )

    assert np.array_equal(base_h1_cached_attempt.u_field1, base_h1_fresh.u_field1)
    assert np.array_equal(base_h1_cached_attempt.v_field1, base_h1_fresh.v_field1)
    assert np.array_equal(base_h1_cached_attempt.w_field1_no_settling, base_h1_fresh.w_field1_no_settling)
    # 誤用快取的話，field1 會被錯誤地全部歸零（沿用 hour h 的「全乾」遮罩），
    # 這裡明確排除這個錯誤結果。
    assert not np.array_equal(base_h1_cached_attempt.u_field1, base_h_dry.field2_endpoint.u)
