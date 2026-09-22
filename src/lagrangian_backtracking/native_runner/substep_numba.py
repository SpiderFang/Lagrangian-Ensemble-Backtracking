"""WP-16 任務 2：把 ``mirror_rk4_substep_loop`` 的整個子步迴圈融合成單一
Numba（``@njit``）呼叫，消除每個子步的 Python 直譯器開銷。

背景與動機
----------
主代理在伺服器 .203 用 cProfile 對後灣網格（24,447 節點×48 層）做的量測
顯示：``mirror_rk4_substep_loop`` 每個流場小時、每個沉降分組平均花
1.63 秒，10 個分組合計約 16 秒，幾乎是單小時成本的全部——但這個函式
呼叫的 BayTrace 原版批次核心（``batch_move_rk4``／``batch_check_
boundaries``／``batch_clamp_z``／``batch_interpolate_velocity``／
``batch_interpolate_diffusivity``，全部是 ``@njit(cache=True)``、逐粒子
迴圈已經編譯過的機器碼）本身並不慢；真正的成本來自**每個子步之間**留在
Python 層的工作：計數器型亂數（SplitMix64＋Box-Muller）、
``_apply_horizontal_diffusion`` 的陣列配置與 ``astype``、Shapely
``contains_xy``（關注區域判定）、``np.any(needs_fallback != 0)`` 等——
這些單獨看都很便宜，但一個流場小時要重複 4,000+ 次（``ndeltp`` 隨
CFL 估計而定），逐次的 Python 直譯器與 NumPy 函式呼叫固定開銷加總起來
就成了主要成本（見最終報告的量測表）。

本模組的解法：把「一個子步」在 Python 層做的全部工作（計數器型亂數、
水平擴散施加、關注區域判定、``needs_fallback`` 檢查、型別轉換）連同對
BayTrace 原版批次核心的呼叫，全部包進**同一個** ``ndeltp`` 迴圈、跑在
**同一次** ``@njit`` 呼叫裡——Python 直譯器只在「這一小時、這一個沉降
分組」開始前被呼叫一次，不是每個子步呼叫好幾次。

BayTrace 原版批次核心怎麼在 ``@njit`` 函式裡呼叫另一個 ``@njit`` 函式
------------------------------------------------------------------------
``batch_move_rk4`` 等五個函式是 BayTrace 原版（唯讀，不得修改）用
``@njit(cache=True)`` 裝飾的模組層級函式，但 LBT 不能在本模組頂層寫
``from baytrace.batch_interp import batch_move_rk4``——BayTrace 原始碼的
實際載入位置由 ``LBT_BAYTRACE_ROOT`` 環境變數決定、透過
``backward_driver.load_baytrace_modules`` 的受控載入器（SHA-256 驗證來源
檔案、``sys.modules`` 衝突檢查）在**執行期**才載入，不是可以靜態 import
的固定路徑；本模組也不重新實作一份受控載入邏輯。

解法是「njit 函式工廠」：``build_fused_substep_kernel(handles)`` 在
執行期接收已受控載入好的 ``BayTraceModuleHandles``，把
``handles.batch_move_rk4`` 等五個 dispatcher 物件當成**閉包變數**，定義
並回傳一個新的 ``@njit`` 函式，函式本體內直接呼叫這些閉包變數——Numba
的型別推斷在編譯這個新函式時，會把閉包裡的 ``CPUDispatcher`` 物件當成
「呼叫另一個已編譯的 njit 函式」正確解析、內聯呼叫，不需要透過 Python
屬性查找（已用最小可重現腳本驗證這個模式確實可行，見最終報告）。編譯
出來的 kernel 函式物件依 ``id(handles)`` 快取在
``_KERNEL_CACHE``（見該全域變數的快取生命週期假設說明），確保整個
run 只編譯一次，不是每小時、每個分組都重新觸發 Numba JIT 編譯。

與原版行為的已知差異（誠實揭露，不迴避）
------------------------------------------------------------------------
1. **SplitMix64 整數部分**：與 ``backward_driver.counter_based_gaussian_
   pairs`` 的 NumPy 向量化版本使用完全相同的整數常數與位元運算，兩者都是
   在明定語意的 64-bit 無號整數環域（模 2**64）上運算，**保證逐位元
   相同**（見 ``tests/test_substep_numba.py`` 的直接比對）。
2. **Box-Muller 的 log／cos／sin**：本模組在 ``@njit`` 函式內用 Python
   標準庫 ``math`` 模組的純量版本（Numba 會把它們編譯成對應的 LLVM
   intrinsic／libm 呼叫），``backward_driver`` 的向量化版本用
   ``numpy`` 的 ufunc（可能使用不同的 SIMD／libm 後端，視 NumPy 版本、
   是否連結 Intel SVML 等而定）。兩者在數學上算同一個轉換，但**不保證**
   逐位元相同——與 WP-14 任務 C（Smagorinsky）觀察到的 Numba/NumPy
   浮點運算 ULP 差異是同一類現象。本模組不假設兩者相同，一律由測試檔
   實測最大 ULP 差異、誠實記錄在最終報告，不強行斷言相等。
3. **關注區域判定**：本模組用標準射線法（crossing-number algorithm，
   PNPOLY）取代 Shapely 的 ``contains_xy``。一般內部點兩者結果相同；
   點剛好落在多邊形邊界上時，兩種演算法對「邊界算不算內部」的浮點捨入
   可能不同，見 ``_point_in_polygon_numba`` docstring 與測試檔的量化
   比例（預期只在測度為零的邊界情況出現，不影響統計等價）。**已知限制**：
   本模組的射線法只處理多邊形的外環（exterior ring），**不支援內環
   （洞，interior holes）**；``prepare_interest_region_arrays`` 會在
   偵測到任何一個多邊形有內環時直接拒絕（fail-closed），不會靜默算錯。
4. **只支援計數器型擴散亂數**：``mirror_rk4_substep_loop`` 的「legacy／
   批次共用串流」（``rng`` 參數）模式在本模組**沒有**對應實作——
   ``hour_major.py``（本模組唯一的正式呼叫端）一律使用計數器型亂數，
   legacy 模式只有 ``run_backward_batch``（逐單元路徑，本工作包未接線
   Numba 融合版本）與少數低階測試會用到，不在本次優化範圍內。
5. **``SubstepDiagnostics`` 的細項拆解不再可用**：融合後的迴圈跑在單一
   ``@njit`` 呼叫裡，無法在不重新插入 Python 層計時（那正是本模組要消除
   的開銷）的前提下，繼續拆解「移動／邊界／擴散取樣／擴散施加／關注區域
   判定」各佔多少時間——``mirror_rk4_substep_loop_numba`` 回傳的
   ``SubstepDiagnostics.time_move_s`` 改為這整個融合 kernel 呼叫的總秒數
   （其餘細項欄位固定為 0），呼叫端若需要細項拆解，仍必須改用
   ``mirror_rk4_substep_loop``（見該函式與本模組差異的權衡說明）。

科學與資料契約
----------------
本模組不改變任何物理假設、公式或求和語意，只是把既有 ``backward_driver``
運算式（SplitMix64／Box-Muller／算子分裂擴散／RK4 呼叫順序）搬進單一
``@njit`` 函式；BayTrace 原始碼（``batch_move_rk4`` 等五個函式）完全
未修改、逐一以原始簽章呼叫。
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numba import njit
from shapely.geometry import Polygon

from .backward_driver import (
    NATIVE_STATUS_LEFT_INTEREST_REGION,
    BackwardDriverError,
    BayTraceModuleHandles,
    ComposedFieldPair,
    DepartureRecord,
    NativeBackwardDriverGrid,
    SubstepDiagnostics,
)

__all__ = [
    "InterestRegionArrays",
    "prepare_interest_region_arrays",
    "build_fused_substep_kernel",
    "mirror_rk4_substep_loop_numba",
]

# ── SplitMix64／Box-Muller 常數：必須與 backward_driver.py 的向量化版本
# 逐位元相同（同一組十六進位常數），見模組文件差異第 1 點。────────────────

_SPLITMIX64_MUL1 = np.uint64(0xBF58476D1CE4E5B9)
_SPLITMIX64_MUL2 = np.uint64(0x94D049BB133111EB)
_SPLITMIX64_GOLDEN_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_SPLITMIX64_SALT = np.uint64(0xD6E8FEB86659FD93)
_SPLITMIX64_MAX53 = np.float64(2.0**53)


@njit(cache=True)
def _splitmix64_finalize_scalar(state: np.uint64) -> np.uint64:
    """``backward_driver._splitmix64_finalize`` 的純量版本，運算式逐行相同。"""

    state = state ^ (state >> np.uint64(30))
    state = state * _SPLITMIX64_MUL1
    state = state ^ (state >> np.uint64(27))
    state = state * _SPLITMIX64_MUL2
    state = state ^ (state >> np.uint64(31))
    return state


@njit(cache=True)
def _counter_based_gaussian_pair_scalar(
    seed_lo: np.uint64, seed_hi: np.uint64, counter: np.uint64
) -> tuple[float, float]:
    """``backward_driver.counter_based_gaussian_pairs`` 的逐粒子純量版本。

    整數部分（SplitMix64 混合、狀態推進）與向量化版本逐位元相同；
    ``log``／``cos``／``sin`` 改用 ``math`` 模組的純量函式，見模組文件
    差異第 2 點——ULP 級差異由呼叫端測試量化，本函式不假設兩者相同。
    """

    seed_mixed = _splitmix64_finalize_scalar(seed_lo ^ _splitmix64_finalize_scalar(seed_hi))
    state = seed_mixed + counter * _SPLITMIX64_GOLDEN_GAMMA
    bits1 = _splitmix64_finalize_scalar(state)
    bits2 = _splitmix64_finalize_scalar(state ^ _SPLITMIX64_SALT)
    u1 = (np.float64(bits1 >> np.uint64(11)) + 1.0) / (_SPLITMIX64_MAX53 + 1.0)
    u2 = (np.float64(bits2 >> np.uint64(11)) + 1.0) / (_SPLITMIX64_MAX53 + 1.0)
    radius = math.sqrt(-2.0 * math.log(u1))
    angle = 2.0 * math.pi * u2
    z0 = radius * math.cos(angle)
    z1 = radius * math.sin(angle)
    return z0, z1


@njit(cache=True)
def _point_in_polygon_numba(
    x: float, y: float, poly_x: np.ndarray, poly_y: np.ndarray, nvert: int
) -> bool:
    """標準射線法（crossing-number，PNPOLY 演算法）點在多邊形內測試。

    只看外環（``poly_x[:nvert]``／``poly_y[:nvert]``，呼叫端保證這是不含
    重複收尾點的開放頂點序列），不支援內環，見模組文件差異第 3 點。與
    Shapely ``contains_xy`` 在一般內部點結果相同，僅邊界上可能因浮點捨入
    不同，由 ``tests/test_substep_numba.py`` 實測量化。
    """

    inside = False
    j = nvert - 1
    for i in range(nvert):
        xi = poly_x[i]
        yi = poly_y[i]
        xj = poly_x[j]
        yj = poly_y[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


@dataclass(frozen=True, slots=True)
class InterestRegionArrays:
    """把逐粒子 Shapely ``Polygon`` 陣列轉成 njit 可用的緊湊陣列表示。

    多顆粒子常常共用同一個測站（同一個 ``Polygon`` 物件），本類別依
    ``id(polygon)`` 去重，只保存**相異**多邊形的頂點，用 ``poly_id``
    （``(n,)``）標出每顆粒子對應哪一個相異多邊形，避免對重複的多邊形
    浪費記憶體與轉換成本。
    """

    poly_id: np.ndarray
    """``(n,)`` int64，每顆粒子對應的相異多邊形索引。"""
    poly_x: np.ndarray
    poly_y: np.ndarray
    """``(max_vertices, n_polygons)`` float64，外環頂點座標（公尺），
    padding 部分的值不會被讀取（見 ``poly_nvert``）。"""
    poly_nvert: np.ndarray
    """``(n_polygons,)`` int64，每個相異多邊形實際的外環頂點數。"""


def prepare_interest_region_arrays(
    polygons_per_particle: np.ndarray | Polygon,
) -> InterestRegionArrays:
    """把 ``interest_region_polygon``（單一）或 ``interest_region_polygons``
    （逐粒子陣列）轉成 ``InterestRegionArrays``，只在每小時呼叫融合 kernel
    前做一次（不是每個子步），見模組文件「本模組的解法」一節。

    Raises:
        BackwardDriverError: 任何一個多邊形含內環（洞），本模組的射線法
            實作不支援，fail-closed 拒絕而不是靜默算錯（見模組文件差異
            第 3 點）。
    """

    if isinstance(polygons_per_particle, Polygon):
        polygons = np.asarray([polygons_per_particle], dtype=object)
        poly_id = np.zeros(1, dtype=np.int64)
    else:
        polygons_per_particle = np.asarray(polygons_per_particle, dtype=object)
        n = polygons_per_particle.shape[0]
        distinct: list[Polygon] = []
        distinct_ids: dict[int, int] = {}
        poly_id = np.empty(n, dtype=np.int64)
        for row in range(n):
            poly = polygons_per_particle[row]
            key = id(poly)
            idx = distinct_ids.get(key)
            if idx is None:
                idx = len(distinct)
                distinct_ids[key] = idx
                distinct.append(poly)
            poly_id[row] = idx
        polygons = np.asarray(distinct, dtype=object)

    n_polygons = polygons.shape[0]
    rings: list[np.ndarray] = []
    for poly in polygons:
        if len(poly.interiors) > 0:
            raise BackwardDriverError(
                "substep_numba 的射線法只支援單純多邊形（無內環／洞），"
                "偵測到一個含內環的關注區域多邊形，fail-closed 拒絕"
            )
        coords = np.asarray(poly.exterior.coords, dtype=np.float64)
        # Shapely 的 exterior.coords 是收尾封閉環（首尾點相同），射線法用
        # 不含重複收尾點的開放頂點序列，見 _point_in_polygon_numba。
        if coords.shape[0] > 1 and np.array_equal(coords[0], coords[-1]):
            coords = coords[:-1]
        rings.append(coords)

    max_vertices = max((ring.shape[0] for ring in rings), default=0)
    if max_vertices == 0:
        raise BackwardDriverError("interest_region 多邊形沒有任何頂點")
    poly_x = np.zeros((max_vertices, n_polygons), dtype=np.float64)
    poly_y = np.zeros((max_vertices, n_polygons), dtype=np.float64)
    poly_nvert = np.zeros(n_polygons, dtype=np.int64)
    for col, ring in enumerate(rings):
        nv = ring.shape[0]
        poly_x[:nv, col] = ring[:, 0]
        poly_y[:nv, col] = ring[:, 1]
        poly_nvert[col] = nv

    return InterestRegionArrays(poly_id=poly_id, poly_x=poly_x, poly_y=poly_y, poly_nvert=poly_nvert)


# ── njit 融合 kernel 工廠：見模組文件「BayTrace 原版批次核心怎麼在 njit
# 函式裡呼叫另一個 njit 函式」一節。──────────────────────────────────────

_KERNEL_CACHE: dict[int, Callable[..., None]] = {}
"""以 ``id(handles)`` 為鍵的融合 kernel 快取。``handles``（
``BayTraceModuleHandles``）在正式用途裡由 ``load_baytrace_modules`` 在
整個 run 開始時建立一次、之後一路傳遞使用，生命週期涵蓋整個 run，
``id()`` 重用（該物件被回收、記憶體位置被另一個物件取代）的風險可忽略；
測試若在同一個行程內對多組不同 ``handles`` 反覆呼叫，仍然安全（只是會
分別各自快取一份、不會用錯 kernel），只有在 ``handles`` 物件被垃圾回收
後、剛好有新物件重用同一個記憶體位置時才可能誤用快取，這在本專案實際
呼叫模式（``handles`` 長駐整個 run）下不會發生。"""


def build_fused_substep_kernel(handles: BayTraceModuleHandles) -> Callable[..., None]:
    """建立（或取回快取的）融合子步迴圈 njit kernel，見模組文件。"""

    cache_key = id(handles)
    cached = _KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    batch_move_rk4 = handles.batch_move_rk4
    batch_check_boundaries = handles.batch_check_boundaries
    batch_clamp_z = handles.batch_clamp_z
    batch_interpolate_velocity = handles.batch_interpolate_velocity
    batch_interpolate_diffusivity = handles.batch_interpolate_diffusivity

    native_status_left_interest_region = np.int32(NATIVE_STATUS_LEFT_INTEREST_REGION)

    @njit(cache=False)
    def _kernel(
        px, py, pz, p_u, p_v, p_w, p_x_prev, p_y_prev, p_elem, p_status, p_inbr, p_rnds,
        p_dhfx, p_dhfy, p_dhfz, p_grdx, p_grdy, p_grdz, p_wind_x, p_wind_y,
        x0, y0, z0,
        i34, elnode, grid_x, grid_y, ic3, nvrt, kbp, kbe, sigma_lcl, dp,
        xctr, yctr, elside, isidenode, snx, sny, isbnd,
        u_field1, v_field1, w_field1, eta_field1, u_field2, v_field2, w_field2,
        elem_is_dry_i64,
        ibf, ndeltp, dtb, hdc, h0, enable_diffusion,
        hf1_field, hf2_field,
        particle_seed_lo, particle_seed_hi, particle_step_counter,
        has_interest_region, poly_id, poly_x, poly_y, poly_nvert,
        native_status, substep_utc_ns_values,
        dep_row, dep_x, dep_y, dep_z, dep_utc_ns,
        needs_fallback,
    ):
        n = px.shape[0]
        n_grid_nodes = grid_x.shape[0]
        dep_count = 0
        fail_substep = -1
        fail_count = 0
        zero_diff_field = np.zeros((nvrt, n_grid_nodes), dtype=np.float64)

        for idt in range(1, ndeltp + 1):
            trat = (idt / ndeltp) if ibf == 1 else ((idt - 1) / ndeltp)

            if enable_diffusion:
                batch_interpolate_diffusivity(
                    px, py, pz, p_elem, p_status,
                    i34, elnode, grid_x, grid_y,
                    nvrt, kbp, kbe, sigma_lcl, dp,
                    zero_diff_field, zero_diff_field, eta_field1,
                    hf1_field, hf2_field,
                    trat, h0, 1, hdc,
                    p_dhfx, p_dhfy, p_dhfz,
                    p_grdx, p_grdy, p_grdz,
                    0, 1.2e-5, 3.0e-4, 30.0,
                    p_wind_x, p_wind_y,
                )

            batch_move_rk4(
                px, py, pz, p_u, p_v, p_w,
                p_x_prev, p_y_prev, p_elem, p_status,
                x0, y0, z0,
                i34, elnode, grid_x, grid_y, ic3,
                nvrt, kbp, kbe, sigma_lcl, dp,
                u_field1, v_field1, w_field1, eta_field1,
                u_field2, v_field2, w_field2,
                ibf, dtb, trat,
            )

            if enable_diffusion:
                for ip in range(n):
                    if p_status[ip] != 1:
                        continue
                    z0g, z1g = _counter_based_gaussian_pair_scalar(
                        particle_seed_lo[ip], particle_seed_hi[ip],
                        np.uint64(particle_step_counter[ip]),
                    )
                    particle_step_counter[ip] += 1
                    kh = p_dhfx[ip]
                    if kh < 0.0:
                        kh = 0.0
                    scale = math.sqrt(2.0 * kh * dtb)
                    px[ip] += z0g * scale
                    py[ip] += z1g * scale

            for ip in range(n):
                needs_fallback[ip] = 0
            batch_check_boundaries(
                px, py, p_x_prev, p_y_prev,
                p_elem, p_status, p_inbr, p_rnds,
                i34, elnode, grid_x, grid_y, xctr, yctr,
                elside, isidenode, ic3, snx, sny, isbnd,
                elem_is_dry_i64,
                0, 1.0,
                needs_fallback,
            )
            local_fail_count = 0
            for ip in range(n):
                if needs_fallback[ip] != 0:
                    local_fail_count += 1
            if local_fail_count > 0:
                fail_substep = idt
                fail_count = local_fail_count
                break

            batch_clamp_z(
                px, py, pz, p_elem, p_status,
                elnode, grid_x, grid_y,
                nvrt, kbp, kbe, sigma_lcl, dp, eta_field1,
            )

            batch_interpolate_velocity(
                px, py, pz, p_elem, p_status,
                i34, elnode, grid_x, grid_y,
                nvrt, kbp, kbe, sigma_lcl, dp,
                u_field1, v_field1, w_field1, eta_field1,
                u_field2, v_field2, w_field2,
                trat, h0,
                p_u, p_v, p_w,
            )

            if has_interest_region:
                utc_ns = substep_utc_ns_values[idt - 1]
                for ip in range(n):
                    if p_status[ip] != 1:
                        continue
                    pid = poly_id[ip]
                    nvert = poly_nvert[pid]
                    inside = _point_in_polygon_numba(px[ip], py[ip], poly_x[:, pid], poly_y[:, pid], nvert)
                    if not inside:
                        dep_row[dep_count] = ip
                        dep_x[dep_count] = px[ip]
                        dep_y[dep_count] = py[ip]
                        dep_z[dep_count] = pz[ip]
                        dep_utc_ns[dep_count] = utc_ns
                        dep_count += 1
                        native_status[ip] = native_status_left_interest_region
                        # pa.status（BayTrace 原版粒子狀態，1=作用中）的 0 代表
                        # 「不再作用中」；與 NATIVE_STATUS_ACTIVE（native_status
                        # 語意下的「仍在追蹤中」）恰好都是整數 0 純屬巧合，兩個
                        # 欄位語意完全不同，這裡直接寫字面值 0 避免混淆。
                        p_status[ip] = 0

            for ip in range(n):
                x0[ip] = px[ip]
                y0[ip] = py[ip]
                z0[ip] = pz[ip]

        return dep_count, fail_substep, fail_count

    _KERNEL_CACHE[cache_key] = _kernel
    return _kernel


def mirror_rk4_substep_loop_numba(
    driver_grid: NativeBackwardDriverGrid,
    handles: BayTraceModuleHandles,
    pa: Any,
    *,
    ibf: int,
    ndeltp: int,
    dtb: float,
    composed: ComposedFieldPair,
    enable_diffusion: bool,
    hf1_field: np.ndarray | None = None,
    hf2_field: np.ndarray | None = None,
    hdc: float = 1.0,
    particle_seed_lo: np.ndarray | None = None,
    particle_seed_hi: np.ndarray | None = None,
    particle_step_counter: np.ndarray | None = None,
    native_status: np.ndarray | None = None,
    substep_utc_ns: Callable[[int], int] | None = None,
    interest_region_polygon: Polygon | None = None,
    interest_region_polygons: np.ndarray | None = None,
    departures: list[DepartureRecord] | None = None,
) -> SubstepDiagnostics:
    """``mirror_rk4_substep_loop`` 的 WP-16 任務 2 融合版本：整個子步迴圈
    （含對 BayTrace 原版批次核心的呼叫）跑在單一 ``@njit`` 呼叫裡，見模組
    文件。介面盡量與 ``mirror_rk4_substep_loop`` 對應，供 ``hour_major.py``
    透過 ``HourMajorConfig.substep_loop_impl`` 切換；已知差異見模組文件
    「與原版行為的已知差異」一節，最重要的是**只支援計數器型擴散亂數**
    （不支援 ``rng`` 參數）。

    Returns:
        ``SubstepDiagnostics``，``time_move_s`` 是整個融合 kernel 呼叫的
        總秒數（其餘細項欄位固定為 0，見模組文件差異第 5 點）。
    """

    if ibf not in (1, -1):
        raise BackwardDriverError("ibf 只能是 1 或 -1")
    if ndeltp < 1:
        raise BackwardDriverError("ndeltp 必須是正整數")
    if interest_region_polygon is not None and interest_region_polygons is not None:
        raise BackwardDriverError("interest_region_polygon 與 interest_region_polygons 只能擇一提供")
    has_interest_region = interest_region_polygon is not None or interest_region_polygons is not None
    if has_interest_region and (native_status is None or substep_utc_ns is None):
        raise BackwardDriverError(
            "提供 interest_region_polygon／interest_region_polygons 時必須同時提供 "
            "native_status 與 substep_utc_ns"
        )
    if enable_diffusion:
        if hf1_field is None or hf2_field is None:
            raise BackwardDriverError("啟用擴散時必須提供 hf1_field 與 hf2_field")
        counter_fields = (particle_seed_lo, particle_seed_hi, particle_step_counter)
        if any(field is None for field in counter_fields):
            raise BackwardDriverError(
                "mirror_rk4_substep_loop_numba 啟用擴散時必須同時提供 particle_seed_lo／"
                "particle_seed_hi／particle_step_counter 三者（不支援 legacy rng 模式，"
                "見模組文件差異第 4 點）"
            )

    import time as _time

    t_start = _time.perf_counter()

    grid = driver_grid.schism_grid
    n = pa.n
    kernel = build_fused_substep_kernel(handles)

    x0 = pa.x.copy()
    y0 = pa.y.copy()
    z0 = pa.z.copy()
    needs_fallback = np.zeros(n, dtype=np.int32)

    if enable_diffusion:
        hf1_arr = np.ascontiguousarray(hf1_field, dtype=np.float64)
        hf2_arr = np.ascontiguousarray(hf2_field, dtype=np.float64)
        seed_lo = np.ascontiguousarray(particle_seed_lo, dtype=np.uint64)
        seed_hi = np.ascontiguousarray(particle_seed_hi, dtype=np.uint64)
        step_counter = particle_step_counter
    else:
        hf1_arr = np.zeros((grid.nvrt, grid.np), dtype=np.float64)
        hf2_arr = hf1_arr
        seed_lo = np.zeros(n, dtype=np.uint64)
        seed_hi = np.zeros(n, dtype=np.uint64)
        step_counter = np.zeros(n, dtype=np.int64)

    if has_interest_region:
        region_arrays = prepare_interest_region_arrays(
            interest_region_polygon if interest_region_polygon is not None else interest_region_polygons
        )
        if interest_region_polygon is not None:
            # 單一多邊形套用到全部 n 顆粒子：prepare_interest_region_arrays
            # 只看得到這一個 Polygon 物件，不知道粒子數，這裡才把 poly_id
            # 廣播成 (n,)（全部指向唯一的相異多邊形索引 0）。
            poly_id = np.zeros(n, dtype=np.int64)
        else:
            poly_id = region_arrays.poly_id
        poly_x = region_arrays.poly_x
        poly_y = region_arrays.poly_y
        poly_nvert = region_arrays.poly_nvert
        substep_utc_ns_values = np.asarray(
            [int(substep_utc_ns(idt)) for idt in range(1, ndeltp + 1)], dtype=np.int64
        )
    else:
        poly_id = np.zeros(n, dtype=np.int64)
        poly_x = np.zeros((1, 1), dtype=np.float64)
        poly_y = np.zeros((1, 1), dtype=np.float64)
        poly_nvert = np.zeros(1, dtype=np.int64)
        substep_utc_ns_values = np.zeros(max(ndeltp, 1), dtype=np.int64)

    dep_row = np.full(n, -1, dtype=np.int64)
    dep_x = np.zeros(n, dtype=np.float64)
    dep_y = np.zeros(n, dtype=np.float64)
    dep_z = np.zeros(n, dtype=np.float64)
    dep_utc_ns = np.zeros(n, dtype=np.int64)

    elem_is_dry_i64 = np.ascontiguousarray(composed.elem_is_dry, dtype=np.int64)

    dep_count, fail_substep, fail_count = kernel(
        pa.x, pa.y, pa.z, pa.u, pa.v, pa.w, pa.x_prev, pa.y_prev, pa.element, pa.status, pa.inbr, pa.rnds,
        pa.dhfx, pa.dhfy, pa.dhfz, pa.grdx, pa.grdy, pa.grdz, pa.wind_x, pa.wind_y,
        x0, y0, z0,
        grid.i34, grid.elnode, grid.x, grid.y, grid.ic3,
        grid.nvrt, grid.kbp, grid.kbe, grid.sigma_lcl, grid.dp,
        grid.xctr, grid.yctr, grid.elside, grid.isidenode, grid.snx, grid.sny, grid.isbnd,
        composed.u_field1, composed.v_field1, composed.w_field1, composed.eta_field1,
        composed.u_field2, composed.v_field2, composed.w_field2,
        elem_is_dry_i64,
        int(ibf), int(ndeltp), float(dtb), float(hdc), float(driver_grid.h0), bool(enable_diffusion),
        hf1_arr, hf2_arr,
        seed_lo, seed_hi, step_counter,
        bool(has_interest_region), poly_id, poly_x, poly_y, poly_nvert,
        native_status if native_status is not None else np.zeros(n, dtype=np.int32),
        substep_utc_ns_values,
        dep_row, dep_x, dep_y, dep_z, dep_utc_ns,
        needs_fallback,
    )

    if fail_substep >= 0:
        raise BackwardDriverError(
            f"{int(fail_count)} 顆粒子在子步 {int(fail_substep)} 觸發 "
            "needs_fallback（一般後援或 death trap），本驅動層不含 Python 層 "
            "KD-tree 後援（見模組 docstring 已知限制第 1 點），fail-closed 中止"
        )

    if departures is not None and dep_count > 0:
        for i in range(dep_count):
            departures.append(
                DepartureRecord(
                    row_particle_index=int(dep_row[i]),
                    status=NATIVE_STATUS_LEFT_INTEREST_REGION,
                    x_m=float(dep_x[i]),
                    y_m=float(dep_y[i]),
                    z_m=float(dep_z[i]),
                    utc_ns=int(dep_utc_ns[i]),
                )
            )

    diag = SubstepDiagnostics()
    diag.n_substeps = ndeltp
    diag.n_particle_substeps = ndeltp * n
    diag.time_move_s = _time.perf_counter() - t_start
    return diag
