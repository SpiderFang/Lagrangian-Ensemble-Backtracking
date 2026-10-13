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
   可能不同，見 ``_ray_cast_numba`` docstring 與測試檔的量化比例
   （預期只在測度為零的邊界情況出現，不影響統計等價）。射線法支援含
   內環（洞，interior holes，例如東北區關注區域內的龜山島）的 ``Polygon``
   與 ``MultiPolygon``：判定「點在幾何內」＝「點在某個子多邊形的外環
   （exterior ring）內，且不在該子多邊形任何一個內環內」，
   ``MultiPolygon`` 則取其任一子多邊形滿足即可；語意與 Shapely
   ``contains_xy`` 一致（邊界情況除外，同上）。``prepare_interest_region_
   arrays`` 只在幾何不是 ``Polygon``／``MultiPolygon``、或幾何本身／其
   任何子多邊形是空幾何時才 fail-closed 拒絕。
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

WP-51 D：對齊 BayTrace 的擴散雜訊與垂向隨機漫步
------------------------------------------------
BayTrace v4.7.2 的 **passive 模式（``mod_part=0``）完全沒有隨機漫步**
（``batch_move_passive``，``batch_interp.py`` 1086–1119 行，只做 ``x0 + ibf*dtb*u``）；
擴散只存在於 oil 模式（``mod_part=1``，``batch_move_oil``，1397–1549 行），而 LBT 的
Smagorinsky 水平擴散本來就取自 oil 模式的機制，因此本專案對齊的是「oil 模式、
無風、無浮力、無風化」的子集：

- 水平與垂向都用均勻雜訊 ``rnd = 2u - 1``、位移 ``rnd * sqrt(6*K*dtb)``
  （1510–1532 行）；舊版是 ``sqrt(2*K*dtb)`` 乘常態亂數，單步變異數同為 ``2*K*dtb``。
- 垂向 Kv 來源：BayTrace 預設 ``ikvmodel=0``（SCHISM ``diffusivity`` 場，``tracker.py``
  740 行讀入，且只在 ``mod_part=1`` 時讀）；本專案 OCM 目前沒有該場，所以提供
  ``vertical_diffusion="constant"``（預設 3.0e-4 m²/s，即 BayTrace ``kv_const``／初始 ``dhfz``
  預設值）與 ``"ocm_field"``（介面＋單元測試，餵 Kv 場給 BayTrace 的
  ``batch_interpolate_diffusivity``，含垂向梯度）。
- Visser (1997) 修正（1524–1530 行）只在 ``imixing=1`` 時啟用；BayTrace 預設
  ``imixing=0``（``tracker.py`` 89 行、``main.py`` 37–41 行），故預設不啟用
  （``visser_correction=False``）；常數 Kv 時修正量為零。
- 垂向位移之後由既有 ``batch_clamp_z`` 夾回水柱（BayTrace ``tracker.py`` 1617 行
  同樣在移動之後呼叫 ``_batch_clamp_z``）。
- 計數器型亂數維持：每顆粒子每子步消耗一個計數器，``(種子, 計數器)`` 決定三個獨立均勻
  亂數（見 ``_counter_based_uniform_triple_scalar``），可重現性與批次組成無關性不變。
- 預設參數（``diffusion_noise="gaussian"``、``vertical_diffusion="off"``）完全保持舊行為。

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
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from .backward_driver import (
    NATIVE_STATUS_LEFT_FLOW_DOMAIN,
    BackwardDriverError,
    BayTraceModuleHandles,
    ComposedFieldPair,
    DepartureRecord,
    LocalDomainFirstExitRecord,
    NativeBackwardDriverGrid,
    SubstepDiagnostics,
)
from .settling_kernels import build_settling_kernels


def _empty_region_arrays(
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """沒有提供任何區域（``has_interest_region``／``has_local_region`` 皆假）時，
    仍要餵給 njit kernel 型別穩定的空陣列（kernel 內部不會讀取這些內容，
    但 numba 需要看到與「有提供」分支完全相同的 dtype／維度才能共用同一份
    編譯後的機器碼，見模組文件「本模組的解法」一節）。"""

    poly_id = np.zeros(n, dtype=np.int64)
    ring_x = np.zeros(0, dtype=np.float64)
    ring_y = np.zeros(0, dtype=np.float64)
    ring_vertex_offsets = np.zeros(1, dtype=np.int64)
    subpoly_ring_offsets = np.zeros(1, dtype=np.int64)
    poly_subpoly_offsets = np.zeros(1, dtype=np.int64)
    return poly_id, ring_x, ring_y, ring_vertex_offsets, subpoly_ring_offsets, poly_subpoly_offsets

__all__ = [
    "NOISE_KIND_GAUSSIAN_LEGACY",
    "NOISE_KIND_UNIFORM_BAYTRACE",
    "VERTICAL_MODE_CONSTANT",
    "VERTICAL_MODE_OCM_FIELD",
    "VERTICAL_MODE_OFF",
    "baytrace_uniform_diffusion_displacement",
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
# WP-51 D：均勻擴散亂數的第三個獨立位元流（垂向）用的鹽值。與 ``_SPLITMIX64_SALT`` 不同，
# 讓同一個計數器產生的三個均勻亂數（x、y、z）各自來自獨立的 SplitMix64 輸出。
_SPLITMIX64_SALT_Z = np.uint64(0xC2B2AE3D27D4EB4F)

# 擴散雜訊種類（``noise_kind``）與垂向擴散模式（``vertical_mode``）的整數代碼。
# 用整數而不是字串，是因為它們要直接傳進 njit kernel（字串在 njit 內不利於分支）。
NOISE_KIND_GAUSSIAN_LEGACY = 0
NOISE_KIND_UNIFORM_BAYTRACE = 1
VERTICAL_MODE_OFF = 0
VERTICAL_MODE_CONSTANT = 1
VERTICAL_MODE_OCM_FIELD = 2


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
def _counter_based_uniform_triple_scalar(
    seed_lo: np.uint64, seed_hi: np.uint64, counter: np.uint64
) -> tuple[float, float, float]:
    """WP-51 D：同一個計數器產生三個獨立均勻亂數 ``(ux, uy, uz)``，皆落在 ``(0, 1)``。

    與 ``_counter_based_gaussian_pair_scalar`` 共用同一組種子混合與計數器推進
    （``state = seed_mixed + counter * golden_gamma``），前兩個均勻亂數
    ``u1``／``u2`` 的位元運算與 Box-Muller 版本完全相同（只是不再做
    Box-Muller 轉換）；第三個來自 ``state ^ _SPLITMIX64_SALT_Z`` 的獨立
    SplitMix64 輸出。每個子步每顆粒子只消耗一個計數器，計數器語意與舊版相同，
    因此「以 (種子, 計數器) 決定亂數」的可重現性與分批獨立性不變。
    """

    seed_mixed = _splitmix64_finalize_scalar(seed_lo ^ _splitmix64_finalize_scalar(seed_hi))
    state = seed_mixed + counter * _SPLITMIX64_GOLDEN_GAMMA
    bits1 = _splitmix64_finalize_scalar(state)
    bits2 = _splitmix64_finalize_scalar(state ^ _SPLITMIX64_SALT)
    bits3 = _splitmix64_finalize_scalar(state ^ _SPLITMIX64_SALT_Z)
    ux = (np.float64(bits1 >> np.uint64(11)) + 1.0) / (_SPLITMIX64_MAX53 + 1.0)
    uy = (np.float64(bits2 >> np.uint64(11)) + 1.0) / (_SPLITMIX64_MAX53 + 1.0)
    uz = (np.float64(bits3 >> np.uint64(11)) + 1.0) / (_SPLITMIX64_MAX53 + 1.0)
    return ux, uy, uz


@njit(cache=True)
def baytrace_uniform_diffusion_displacement(
    ux: float,
    uy: float,
    uz: float,
    kh: float,
    kv: float,
    dkv_dz: float,
    dtb: float,
    vertical_on: bool,
    visser: bool,
) -> tuple[float, float, float]:
    """BayTrace ``batch_move_oil`` 的隨機漫步位移（``batch_interp.py`` 1510–1537 行）。

    對 ``ux``／``uy``／``uz`` ∈ (0, 1) 取 ``rnd = 2*u - 1 ∈ (-1, 1)``（BayTrace
    1510–1512 行 ``2.0 * dx - 1.0``），位移 ``rnd * sqrt(6 * K * dtb)``
    （1520–1532 行）。均勻分布在 ``(-1, 1)`` 的變異數是 1/3，所以單一子步位移
    變異數是 ``6*K*dtb/3 = 2*K*dtb``，與高斯雜訊的 ``sqrt(2*K*dtb)`` 同階，
    累積 ``t`` 秒後變異數為 ``2*K*t``。

    水平 ``kh`` 負值夾到 0（與舊版高斯路徑一致）。垂向（``vertical_on``）：
    ``visser=False`` 時 ``kv`` 直接代入（BayTrace 預設 ``imixing=0``，1531–1532 行）；
    ``visser=True`` 時用 Visser (1997) 修正後的 ``kv* = kv + 0.5*dkv_dz**2*dtb``
    （1524–1530 行，下限 1e-6）。垂向另外加梯度漂移 ``dkv_dz * dtb``（BayTrace 的
    ``w_total = w + rising + grad_z``，1481 行附近）；常數 Kv 時 ``dkv_dz = 0``，
    漂移與 Visser 修正皆為零。回傳 ``(dx, dy, dz)``，不含平流。
    """

    if kh < 0.0:
        kh = 0.0
    scale = math.sqrt(6.0 * kh * dtb)
    dx = (2.0 * ux - 1.0) * scale
    dy = (2.0 * uy - 1.0) * scale
    dz = 0.0
    if vertical_on:
        kv_eff = kv
        if visser:
            kv_eff = kv + dkv_dz * dkv_dz * dtb * 0.5
        if kv_eff < 1.0e-6:
            kv_eff = 1.0e-6
        dz = (2.0 * uz - 1.0) * math.sqrt(6.0 * kv_eff * dtb) + dkv_dz * dtb
    return dx, dy, dz


@njit(cache=True)
def _ray_cast_numba(x: float, y: float, ring_x: np.ndarray, ring_y: np.ndarray) -> bool:
    """標準射線法（crossing-number，PNPOLY 演算法）點在**單一環**內測試。

    ``ring_x``／``ring_y`` 是同一個環（外環或內環皆可）的頂點座標（公尺），
    呼叫端保證這是不含重複收尾點的開放頂點序列。與 Shapely 逐環判定在
    一般內部點結果相同，僅邊界上可能因浮點捨入不同，由
    ``tests/test_substep_numba.py`` 實測量化——本函式只判定「點在這個環
    圍出的區域內」，不處理洞或多重多邊形，那是 ``_point_in_geometry_numba``
    的責任。
    """

    nvert = ring_x.shape[0]
    inside = False
    j = nvert - 1
    for i in range(nvert):
        xi = ring_x[i]
        yi = ring_y[i]
        xj = ring_x[j]
        yj = ring_y[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


@njit(cache=True)
def _point_in_geometry_numba(
    x: float,
    y: float,
    ring_x: np.ndarray,
    ring_y: np.ndarray,
    ring_vertex_offsets: np.ndarray,
    subpoly_ring_offsets: np.ndarray,
    poly_subpoly_offsets: np.ndarray,
    poly_idx: int,
) -> bool:
    """點在（可能含內環的）``Polygon`` 或 ``MultiPolygon`` 內測試。

    語意（與 Shapely ``contains_xy`` 一致，邊界情況除外，見模組文件差異
    第 3 點）：一個幾何（``poly_idx`` 指定）由一個或多個「子多邊形」
    （``MultiPolygon`` 的每個 ``geoms`` 元素，單一 ``Polygon`` 則只有一個）
    組成；點在幾何內＝點在**任一**子多邊形內；點在子多邊形內＝點在該子
    多邊形外環（exterior ring）內、且不在該子多邊形**任何一個**內環
    （洞，interior ring）內。

    四層扁平陣列＋偏移（``prepare_interest_region_arrays`` 建構，見該函式
    docstring）：幾何 → 子多邊形（``poly_subpoly_offsets``）→ 環
    （``subpoly_ring_offsets``，每個子多邊形範圍內第一個環固定是外環，
    其餘是內環）→ 頂點（``ring_vertex_offsets``，索引進 ``ring_x``／
    ``ring_y``）。用偏移而不是固定寬度 padding，是因為不同環的頂點數、
    不同子多邊形的環數差異可能很大（例如東北關注區域的龜山島洞只有幾個
    頂點，外環卻有數百個），padding 到最大值會浪費大量記憶體。
    """

    sp_start = poly_subpoly_offsets[poly_idx]
    sp_end = poly_subpoly_offsets[poly_idx + 1]
    for sp in range(sp_start, sp_end):
        ring_start = subpoly_ring_offsets[sp]
        ring_end = subpoly_ring_offsets[sp + 1]
        ext_v_start = ring_vertex_offsets[ring_start]
        ext_v_end = ring_vertex_offsets[ring_start + 1]
        if not _ray_cast_numba(x, y, ring_x[ext_v_start:ext_v_end], ring_y[ext_v_start:ext_v_end]):
            continue
        in_hole = False
        for r in range(ring_start + 1, ring_end):
            v_start = ring_vertex_offsets[r]
            v_end = ring_vertex_offsets[r + 1]
            if _ray_cast_numba(x, y, ring_x[v_start:v_end], ring_y[v_start:v_end]):
                in_hole = True
                break
        if not in_hole:
            return True
    return False


@dataclass(frozen=True, slots=True)
class InterestRegionArrays:
    """把逐粒子 Shapely ``Polygon``／``MultiPolygon`` 陣列轉成 njit 可用的
    緊湊扁平陣列表示（四層：幾何 → 子多邊形 → 環 → 頂點，見
    ``_point_in_geometry_numba`` docstring）。

    多顆粒子常常共用同一個測站（同一個幾何物件），本類別依 ``id(polygon)``
    去重，只保存**相異**幾何的頂點，用 ``poly_id``（``(n,)``）標出每顆
    粒子對應哪一個相異幾何，避免對重複的幾何浪費記憶體與轉換成本。
    """

    poly_id: np.ndarray
    """``(n,)`` int64，每顆粒子對應的相異幾何索引。"""
    ring_x: np.ndarray
    ring_y: np.ndarray
    """``(total_vertices,)`` float64，所有環（外環＋內環，跨全部相異幾何
    與其子多邊形依序串接）的頂點座標（公尺），不含重複收尾點。個別環的
    頂點範圍由 ``ring_vertex_offsets`` 標定。"""
    ring_vertex_offsets: np.ndarray
    """``(n_rings + 1,)`` int64，第 ``r`` 個環的頂點是
    ``ring_x[ring_vertex_offsets[r]:ring_vertex_offsets[r+1]]``。"""
    subpoly_ring_offsets: np.ndarray
    """``(n_subpolygons + 1,)`` int64，第 ``sp`` 個子多邊形的環範圍
    ``[subpoly_ring_offsets[sp], subpoly_ring_offsets[sp+1])`` 中，第一個
    環固定是外環，其餘（若有）是內環（洞）。"""
    poly_subpoly_offsets: np.ndarray
    """``(n_polygons + 1,)`` int64，第 ``p`` 個相異幾何的子多邊形範圍
    ``[poly_subpoly_offsets[p], poly_subpoly_offsets[p+1])``——單一
    ``Polygon`` 只有一個子多邊形，``MultiPolygon`` 則對應其
    ``geoms`` 逐一展開。"""


def _append_ring_coords(
    coords: np.ndarray, x_parts: list[np.ndarray], y_parts: list[np.ndarray], vertex_offsets: list[int]
) -> None:
    """把一個環（外環或內環）的收尾封閉座標，去除重複收尾點後，接到扁平
    頂點陣列末端，並登記新的頂點偏移。"""

    if coords.shape[0] > 1 and np.array_equal(coords[0], coords[-1]):
        coords = coords[:-1]
    x_parts.append(coords[:, 0])
    y_parts.append(coords[:, 1])
    vertex_offsets.append(vertex_offsets[-1] + coords.shape[0])


def prepare_interest_region_arrays(
    polygons_per_particle: np.ndarray | Polygon | MultiPolygon,
) -> InterestRegionArrays:
    """把 ``interest_region_polygon``（單一）或 ``interest_region_polygons``
    （逐粒子陣列）轉成 ``InterestRegionArrays``，只在每小時呼叫融合 kernel
    前做一次（不是每個子步），見模組文件「本模組的解法」一節。支援含內環
    （洞，例如東北關注區域內的龜山島）的 ``Polygon`` 與 ``MultiPolygon``，
    見模組文件差異第 3 點。

    Raises:
        BackwardDriverError: 任何一個幾何不是 ``Polygon``／``MultiPolygon``、
            或幾何本身／其任一子多邊形是空幾何，fail-closed 拒絕而不是
            靜默算錯。
    """

    if isinstance(polygons_per_particle, BaseGeometry):
        # 單一幾何（``interest_region_polygon``）套用到全部粒子：即使不是
        # Polygon／MultiPolygon（例如誤傳 LineString），也先包成長度 1 的
        # 陣列送進下面共用的逐幾何檢查迴圈，才能給出正確的錯誤訊息，而不是
        # 讓 0 維 ``np.asarray`` 在下面的陣列分支裡因為沒有 ``shape[0]``
        # 而拋出跟本意無關的 IndexError。
        polygons = np.asarray([polygons_per_particle], dtype=object)
        poly_id = np.zeros(1, dtype=np.int64)
    else:
        polygons_per_particle = np.asarray(polygons_per_particle, dtype=object)
        n = polygons_per_particle.shape[0]
        distinct: list[Polygon | MultiPolygon] = []
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

    ring_x_parts: list[np.ndarray] = []
    ring_y_parts: list[np.ndarray] = []
    ring_vertex_offsets: list[int] = [0]
    subpoly_ring_offsets: list[int] = [0]
    poly_subpoly_offsets: list[int] = [0]

    for poly in polygons:
        if isinstance(poly, MultiPolygon):
            sub_polys = list(poly.geoms)
        elif isinstance(poly, Polygon):
            sub_polys = [poly]
        else:
            raise BackwardDriverError(
                "interest_region 幾何只能是 Polygon 或 MultiPolygon，"
                f"偵測到 {type(poly).__name__}，fail-closed 拒絕"
            )
        if poly.is_empty or len(sub_polys) == 0:
            raise BackwardDriverError("interest_region 多邊形是空幾何，fail-closed 拒絕")
        for sub in sub_polys:
            if sub.is_empty or len(sub.exterior.coords) == 0:
                raise BackwardDriverError("interest_region 多邊形是空幾何，fail-closed 拒絕")
            # Shapely 的 exterior／interior coords 是收尾封閉環（首尾點
            # 相同），射線法用不含重複收尾點的開放頂點序列，見
            # _ray_cast_numba；每個子多邊形的第一個環固定是外環，之後
            # （若有）依序是內環（洞），語意見 _point_in_geometry_numba。
            ext_coords = np.asarray(sub.exterior.coords, dtype=np.float64)
            _append_ring_coords(ext_coords, ring_x_parts, ring_y_parts, ring_vertex_offsets)
            for interior in sub.interiors:
                int_coords = np.asarray(interior.coords, dtype=np.float64)
                _append_ring_coords(int_coords, ring_x_parts, ring_y_parts, ring_vertex_offsets)
            subpoly_ring_offsets.append(len(ring_vertex_offsets) - 1)
        poly_subpoly_offsets.append(len(subpoly_ring_offsets) - 1)

    if len(ring_x_parts) == 0:
        raise BackwardDriverError("interest_region 多邊形沒有任何頂點")

    return InterestRegionArrays(
        poly_id=poly_id,
        ring_x=np.concatenate(ring_x_parts),
        ring_y=np.concatenate(ring_y_parts),
        ring_vertex_offsets=np.asarray(ring_vertex_offsets, dtype=np.int64),
        subpoly_ring_offsets=np.asarray(subpoly_ring_offsets, dtype=np.int64),
        poly_subpoly_offsets=np.asarray(poly_subpoly_offsets, dtype=np.int64),
    )


# ── njit 融合 kernel 工廠：見模組文件「BayTrace 原版批次核心怎麼在 njit
# 函式裡呼叫另一個 njit 函式」一節。──────────────────────────────────────

_KERNEL_CACHE: dict[tuple[int, bool], Callable[..., None]] = {}
"""以 ``id(handles)`` 為鍵的融合 kernel 快取。``handles``（
``BayTraceModuleHandles``）在正式用途裡由 ``load_baytrace_modules`` 在
整個 run 開始時建立一次、之後一路傳遞使用，生命週期涵蓋整個 run，
``id()`` 重用（該物件被回收、記憶體位置被另一個物件取代）的風險可忽略；
測試若在同一個行程內對多組不同 ``handles`` 反覆呼叫，仍然安全（只是會
分別各自快取一份、不會用錯 kernel），只有在 ``handles`` 物件被垃圾回收
後、剛好有新物件重用同一個記憶體位置時才可能誤用快取，這在本專案實際
呼叫模式（``handles`` 長駐整個 run）下不會發生。"""


def build_fused_substep_kernel(
    handles: BayTraceModuleHandles, per_particle_settling: bool = False
) -> Callable[..., None]:
    """建立（或取回快取的）融合子步迴圈 njit kernel，見模組文件。

    ``per_particle_settling``（WP-54）：``False`` 時行為與先前完全相同，速度與
    位移全部走 BayTrace 原版 ``batch_move_rk4``／``batch_interpolate_velocity``，
    kernel 末端兩個額外引數 ``settle_mask``／``settle`` 被忽略；``True`` 時改走
    ``settling_kernels`` 的逐粒子沉降版本，``settle_mask`` 是 ``(nvrt, np)`` bool
    的 ``effective_mask``，``settle`` 是 ``(n,)`` float64 的每顆粒子自己的沉降
    速度，且 ``w_field`` 必須是**沒有**烤進沉降速度的合成場，見
    ``settling_kernels`` 模組文件。兩種模式的 kernel 簽章相同，各自獨立快取。
    """

    cache_key = (id(handles), bool(per_particle_settling))
    cached = _KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    batch_check_boundaries = handles.batch_check_boundaries
    batch_clamp_z = handles.batch_clamp_z
    batch_interpolate_diffusivity = handles.batch_interpolate_diffusivity

    if per_particle_settling:
        _settle_kernels = build_settling_kernels(handles)
        _move_impl = _settle_kernels.batch_move_rk4
        _interp_impl = _settle_kernels.batch_interpolate_velocity

        @njit(cache=False)
        def batch_move_rk4(
            px, py, pz, p_u, p_v, p_w, p_x_prev, p_y_prev, p_elem, p_status, x0, y0, z0,
            i34, elnode, grid_x, grid_y, ic3, nvrt, kbp, kbe, sigma_lcl, dp,
            u_field1, v_field1, w_field1, eta_field1, u_field2, v_field2, w_field2,
            ibf, dtb, trat, settle_mask, settle,
        ):
            _move_impl(
                px, py, pz, p_u, p_v, p_w, p_x_prev, p_y_prev, p_elem, p_status, x0, y0, z0,
                i34, elnode, grid_x, grid_y, ic3, nvrt, kbp, kbe, sigma_lcl, dp,
                u_field1, v_field1, w_field1, eta_field1, u_field2, v_field2, w_field2,
                ibf, dtb, trat, settle_mask, settle,
            )

        @njit(cache=False)
        def batch_interpolate_velocity(
            px, py, pz, p_elem, p_status, i34, elnode, grid_x, grid_y,
            nvrt, kbp, kbe, sigma_lcl, dp,
            u_field1, v_field1, w_field1, eta_field1, u_field2, v_field2, w_field2,
            trat, h0, p_u, p_v, p_w, settle_mask, settle,
        ):
            _interp_impl(
                px, py, pz, p_elem, p_status, i34, elnode, grid_x, grid_y,
                nvrt, kbp, kbe, sigma_lcl, dp,
                u_field1, v_field1, w_field1, eta_field1, u_field2, v_field2, w_field2,
                trat, h0, p_u, p_v, p_w, settle_mask, settle,
            )
    else:
        _move_orig = handles.batch_move_rk4
        _interp_orig = handles.batch_interpolate_velocity

        @njit(cache=False)
        def batch_move_rk4(
            px, py, pz, p_u, p_v, p_w, p_x_prev, p_y_prev, p_elem, p_status, x0, y0, z0,
            i34, elnode, grid_x, grid_y, ic3, nvrt, kbp, kbe, sigma_lcl, dp,
            u_field1, v_field1, w_field1, eta_field1, u_field2, v_field2, w_field2,
            ibf, dtb, trat, settle_mask, settle,
        ):
            _move_orig(
                px, py, pz, p_u, p_v, p_w, p_x_prev, p_y_prev, p_elem, p_status, x0, y0, z0,
                i34, elnode, grid_x, grid_y, ic3, nvrt, kbp, kbe, sigma_lcl, dp,
                u_field1, v_field1, w_field1, eta_field1, u_field2, v_field2, w_field2,
                ibf, dtb, trat,
            )

        @njit(cache=False)
        def batch_interpolate_velocity(
            px, py, pz, p_elem, p_status, i34, elnode, grid_x, grid_y,
            nvrt, kbp, kbe, sigma_lcl, dp,
            u_field1, v_field1, w_field1, eta_field1, u_field2, v_field2, w_field2,
            trat, h0, p_u, p_v, p_w, settle_mask, settle,
        ):
            _interp_orig(
                px, py, pz, p_elem, p_status, i34, elnode, grid_x, grid_y,
                nvrt, kbp, kbe, sigma_lcl, dp,
                u_field1, v_field1, w_field1, eta_field1, u_field2, v_field2, w_field2,
                trat, h0, p_u, p_v, p_w,
            )

    native_status_left_flow_domain = np.int32(NATIVE_STATUS_LEFT_FLOW_DOMAIN)

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
        noise_kind, vertical_mode, kz_const, visser,
        hf1_field, hf2_field, kv_field1, kv_field2,
        particle_seed_lo, particle_seed_hi, particle_step_counter,
        has_interest_region, poly_id,
        region_ring_x, region_ring_y, region_ring_vertex_offsets,
        region_subpoly_ring_offsets, region_poly_subpoly_offsets,
        has_local_region, local_poly_id,
        local_region_ring_x, local_region_ring_y, local_region_ring_vertex_offsets,
        local_region_subpoly_ring_offsets, local_region_poly_subpoly_offsets,
        local_first_exit_pending,
        native_status, substep_utc_ns_values,
        dep_row, dep_x, dep_y, dep_z, dep_utc_ns,
        local_dep_row, local_dep_x, local_dep_y, local_dep_z, local_dep_utc_ns,
        needs_fallback, settle_mask, settle,
    ):
        n = px.shape[0]
        n_grid_nodes = grid_x.shape[0]
        dep_count = 0
        local_dep_count = 0
        fail_substep = -1
        fail_count = 0
        zero_diff_field = np.zeros((nvrt, n_grid_nodes), dtype=np.float64)
        # WP-51 D：垂向 Kv 來源。``ocm_field`` 模式把呼叫端提供的 Kv 場餵給 BayTrace
        # ``batch_interpolate_diffusivity``（``ikvmodel=0``，SCHISM 場，含 trat/2 時間內插
        # 與垂向有限差分梯度）；``constant`` 模式用 ``ikvmodel=3``（使用者常數，梯度為 0）；
        # 其餘維持舊行為（零場，``ikvmodel=0``）。兩個分支的陣列型別相同，Numba 可合併。
        if vertical_mode == 2:
            kv_use1 = kv_field1
            kv_use2 = kv_field2
        else:
            kv_use1 = zero_diff_field
            kv_use2 = zero_diff_field
        ikvmodel = 3 if vertical_mode == 1 else 0
        vertical_on = vertical_mode != 0

        for idt in range(1, ndeltp + 1):
            trat = (idt / ndeltp) if ibf == 1 else ((idt - 1) / ndeltp)

            if enable_diffusion:
                batch_interpolate_diffusivity(
                    px, py, pz, p_elem, p_status,
                    i34, elnode, grid_x, grid_y,
                    nvrt, kbp, kbe, sigma_lcl, dp,
                    kv_use1, kv_use2, eta_field1,
                    hf1_field, hf2_field,
                    trat, h0, 1, hdc,
                    p_dhfx, p_dhfy, p_dhfz,
                    p_grdx, p_grdy, p_grdz,
                    ikvmodel, 1.2e-5, kz_const, 30.0,
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
                ibf, dtb, trat, settle_mask, settle,
            )

            if enable_diffusion:
                if noise_kind == 1:
                    # WP-51 D：BayTrace 均勻雜訊 ``rnd*sqrt(6*K*dtb)``，水平＋（選用）垂向。
                    # 每顆粒子每子步仍只消耗一個計數器；垂向位移隨後由
                    # ``batch_clamp_z`` 夾回水柱內。
                    for ip in range(n):
                        if p_status[ip] != 1:
                            continue
                        ux, uy, uz = _counter_based_uniform_triple_scalar(
                            particle_seed_lo[ip], particle_seed_hi[ip],
                            np.uint64(particle_step_counter[ip]),
                        )
                        particle_step_counter[ip] += 1
                        ddx, ddy, ddz = baytrace_uniform_diffusion_displacement(
                            ux, uy, uz, p_dhfx[ip], p_dhfz[ip], p_grdz[ip], dtb,
                            vertical_on, visser,
                        )
                        px[ip] += ddx
                        py[ip] += ddy
                        pz[ip] += ddz
                else:
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
                p_u, p_v, p_w, settle_mask, settle,
            )

            if has_local_region:
                # WP-23：local domain 判定必須排在 flow domain 判定之前，且
                # 只看「仍作用中、且尚未記錄過首次離開」（local_first_exit_
                # pending[ip]）的粒子；離開時只翻轉 pending 旗標、記一筆非
                # 終止事件，完全不動 p_status／native_status——這是與下面
                # flow domain 判定「記錄並終止」的唯一差異。
                local_utc_ns = substep_utc_ns_values[idt - 1]
                for ip in range(n):
                    if p_status[ip] != 1 or not local_first_exit_pending[ip]:
                        continue
                    lpid = local_poly_id[ip]
                    local_inside = _point_in_geometry_numba(
                        px[ip], py[ip],
                        local_region_ring_x, local_region_ring_y, local_region_ring_vertex_offsets,
                        local_region_subpoly_ring_offsets, local_region_poly_subpoly_offsets,
                        lpid,
                    )
                    if not local_inside:
                        local_dep_row[local_dep_count] = ip
                        local_dep_x[local_dep_count] = px[ip]
                        local_dep_y[local_dep_count] = py[ip]
                        local_dep_z[local_dep_count] = pz[ip]
                        local_dep_utc_ns[local_dep_count] = local_utc_ns
                        local_dep_count += 1
                        local_first_exit_pending[ip] = False

            if has_interest_region:
                utc_ns = substep_utc_ns_values[idt - 1]
                for ip in range(n):
                    if p_status[ip] != 1:
                        continue
                    pid = poly_id[ip]
                    inside = _point_in_geometry_numba(
                        px[ip], py[ip],
                        region_ring_x, region_ring_y, region_ring_vertex_offsets,
                        region_subpoly_ring_offsets, region_poly_subpoly_offsets,
                        pid,
                    )
                    if not inside:
                        dep_row[dep_count] = ip
                        dep_x[dep_count] = px[ip]
                        dep_y[dep_count] = py[ip]
                        dep_z[dep_count] = pz[ip]
                        dep_utc_ns[dep_count] = utc_ns
                        dep_count += 1
                        native_status[ip] = native_status_left_flow_domain
                        # pa.status（BayTrace 原版粒子狀態，1=作用中）的 0 代表
                        # 「不再作用中」；與 NATIVE_STATUS_ACTIVE（native_status
                        # 語意下的「仍在追蹤中」）恰好都是整數 0 純屬巧合，兩個
                        # 欄位語意完全不同，這裡直接寫字面值 0 避免混淆。
                        p_status[ip] = 0

            for ip in range(n):
                x0[ip] = px[ip]
                y0[ip] = py[ip]
                z0[ip] = pz[ip]

        return dep_count, local_dep_count, fail_substep, fail_count

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
    interest_region_polygon: Polygon | MultiPolygon | None = None,
    interest_region_polygons: np.ndarray | None = None,
    departures: list[DepartureRecord] | None = None,
    local_region_polygon: Polygon | MultiPolygon | None = None,
    local_region_polygons: np.ndarray | None = None,
    local_first_exit_pending: np.ndarray | None = None,
    local_departures: list[LocalDomainFirstExitRecord] | None = None,
    diffusion_noise: str = "gaussian",
    vertical_diffusion: str = "off",
    kz_constant_m2ps: float = 3.0e-4,
    visser_correction: bool = False,
    kv_field1: np.ndarray | None = None,
    kv_field2: np.ndarray | None = None,
    particle_settling_mps: np.ndarray | None = None,
    settling_effective_mask: np.ndarray | None = None,
) -> SubstepDiagnostics:
    """``mirror_rk4_substep_loop`` 的 WP-16 任務 2 融合版本：整個子步迴圈
    （含對 BayTrace 原版批次核心的呼叫）跑在單一 ``@njit`` 呼叫裡，見模組
    文件。介面盡量與 ``mirror_rk4_substep_loop`` 對應，供 ``hour_major.py``
    透過 ``HourMajorConfig.substep_loop_impl`` 切換；已知差異見模組文件
    「與原版行為的已知差異」一節，最重要的是**只支援計數器型擴散亂數**
    （不支援 ``rng`` 參數）。

    WP-23：``local_region_polygon``／``local_region_polygons``／
    ``local_first_exit_pending``／``local_departures`` 四個參數與
    ``mirror_rk4_substep_loop`` 的同名參數語意完全對應（見該函式
    docstring）——``interest_region_*`` 系列 WP-23 起代表 **flow domain**
    開放邊界（離開即終止），``local_region_*`` 系列代表每顆粒子自己站點
    的 **local domain**（``own_local_domain``，離開時只記錄、不終止）。
    兩層判定在融合 kernel 內部依相同順序（local 先於 flow）執行，見
    ``build_fused_substep_kernel`` 內的 ``_kernel`` 實作。

    WP-51 D 新增的四組參數（預設值全部保持舊行為，逐位元相同）：

    - ``diffusion_noise``：``"gaussian"``（預設，舊版，``sqrt(2*K*dtb)`` 乘常態亂數）
      或 ``"uniform"``（BayTrace 對應：``(2u-1)*sqrt(6*K*dtb)``，``u`` 為均勻亂數；
      單一子步變異數同為 ``2*K*dtb``，但分布是有界均勻而非常態）。
    - ``vertical_diffusion``：``"off"``（預設，不做垂向隨機漫步）、``"constant"``
      （常數 ``kz_constant_m2ps``，預設 3.0e-4 m²/s，與 BayTrace ``kv_const`` 預設相同）
      或 ``"ocm_field"``（由 ``kv_field1``／``kv_field2`` 這兩個 ``(nvrt, np)`` Kv 場經
      BayTrace ``batch_interpolate_diffusivity``（``ikvmodel=0``）做 trat/2 時間內插與
      垂向梯度；本專案目前 OCM 沒有 diffusivity 場，僅提供介面與單元測試）。
      垂向擴散只在 ``diffusion_noise="uniform"`` 時可用（高斯舊路徑沒有垂向實作）。
    - ``visser_correction``：``True`` 時用 Visser (1997) 修正 Kv（BayTrace ``imixing=1``）；
      預設 ``False`` 對應 BayTrace 預設 ``imixing=0``。常數 Kv 時修正量為零。
    - 垂向位移之後由 kernel 內的 ``batch_clamp_z`` 夾回水柱內，與 BayTrace 一致。

    WP-54：``particle_settling_mps``（``(n,)`` float64，每顆粒子自己的沉降速度，
    負值向下）與 ``settling_effective_mask``（``(nvrt, np)`` bool，``base.
    effective_mask``）必須同時提供或同時省略。提供時 ``composed`` 的 ``w_field1``／
    ``w_field2`` 必須是**沒有**烤進沉降速度的合成場（``backward_driver.
    composed_pair_without_settling``），kernel 改走 ``settling_kernels`` 的逐粒子
    沉降版本；省略時行為與先前逐位元相同。

    Returns:
        ``SubstepDiagnostics``，``time_move_s`` 是整個融合 kernel 呼叫的
        總秒數（其餘細項欄位固定為 0，見模組文件差異第 5 點）。
    """

    if diffusion_noise not in ("gaussian", "uniform"):
        raise BackwardDriverError(f"diffusion_noise 只能是 'gaussian' 或 'uniform'，收到 {diffusion_noise!r}")
    if vertical_diffusion not in ("off", "constant", "ocm_field"):
        raise BackwardDriverError(
            f"vertical_diffusion 只能是 'off'、'constant' 或 'ocm_field'，收到 {vertical_diffusion!r}"
        )
    if vertical_diffusion != "off" and diffusion_noise != "uniform":
        raise BackwardDriverError("垂向擴散只支援 diffusion_noise='uniform'（BayTrace 對應路徑）")
    if vertical_diffusion != "off" and not enable_diffusion:
        raise BackwardDriverError("啟用垂向擴散必須同時 enable_diffusion=True")
    if vertical_diffusion == "constant" and not (math.isfinite(kz_constant_m2ps) and kz_constant_m2ps > 0.0):
        raise BackwardDriverError("kz_constant_m2ps 必須是有限的正值")
    if vertical_diffusion == "ocm_field":
        if kv_field1 is None or kv_field2 is None:
            raise BackwardDriverError("vertical_diffusion='ocm_field' 必須同時提供 kv_field1 與 kv_field2")
        expected_shape = (driver_grid.schism_grid.nvrt, driver_grid.schism_grid.np)
        for label, field in (("kv_field1", kv_field1), ("kv_field2", kv_field2)):
            if tuple(np.shape(field)) != expected_shape:
                raise BackwardDriverError(
                    f"{label} 形狀必須是 (nvrt, np)={expected_shape}，收到 {tuple(np.shape(field))}"
                )
            if not np.all(np.isfinite(field)):
                raise BackwardDriverError(f"{label} 含非有限值，fail-closed 拒絕")
    if (particle_settling_mps is None) != (settling_effective_mask is None):
        raise BackwardDriverError(
            "particle_settling_mps 與 settling_effective_mask 必須同時提供或同時省略"
        )
    per_particle_settling = particle_settling_mps is not None
    if per_particle_settling:
        if particle_settling_mps.shape != (pa.n,) or particle_settling_mps.dtype != np.float64:
            raise BackwardDriverError("particle_settling_mps 必須是長度等於粒子數的 float64 一維陣列")
        if not np.all(np.isfinite(particle_settling_mps)):
            raise BackwardDriverError("particle_settling_mps 含非有限值，fail-closed 拒絕")
        expected_mask_shape = (driver_grid.schism_grid.nvrt, driver_grid.schism_grid.np)
        if settling_effective_mask.shape != expected_mask_shape or settling_effective_mask.dtype != np.bool_:
            raise BackwardDriverError(
                f"settling_effective_mask 必須是形狀 {expected_mask_shape} 的 bool 陣列"
            )
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
    if local_region_polygon is not None and local_region_polygons is not None:
        raise BackwardDriverError("local_region_polygon 與 local_region_polygons 只能擇一提供")
    has_local_region = local_region_polygon is not None or local_region_polygons is not None
    if has_local_region and (local_first_exit_pending is None or substep_utc_ns is None):
        raise BackwardDriverError(
            "提供 local_region_polygon／local_region_polygons 時必須同時提供 "
            "local_first_exit_pending 與 substep_utc_ns"
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
    kernel = build_fused_substep_kernel(handles, per_particle_settling)

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

    noise_kind_code = NOISE_KIND_UNIFORM_BAYTRACE if diffusion_noise == "uniform" else NOISE_KIND_GAUSSIAN_LEGACY
    vertical_mode_code = {
        "off": VERTICAL_MODE_OFF, "constant": VERTICAL_MODE_CONSTANT, "ocm_field": VERTICAL_MODE_OCM_FIELD,
    }[vertical_diffusion]
    if vertical_diffusion == "ocm_field":
        kv1_arr = np.ascontiguousarray(kv_field1, dtype=np.float64)
        kv2_arr = np.ascontiguousarray(kv_field2, dtype=np.float64)
    else:
        # 非 ocm_field 模式 kernel 不會讀這兩個陣列；給同型別（float64、2 維、C 連續）的
        # 極小陣列，避免每次呼叫配置 (nvrt, np) 大小的暫存。
        kv1_arr = np.zeros((1, 1), dtype=np.float64)
        kv2_arr = kv1_arr

    needs_utc_ns_values = has_interest_region or has_local_region

    if has_interest_region:
        region_arrays = prepare_interest_region_arrays(
            interest_region_polygon if interest_region_polygon is not None else interest_region_polygons
        )
        if interest_region_polygon is not None:
            # 單一幾何套用到全部 n 顆粒子：prepare_interest_region_arrays
            # 只看得到這一個 Polygon／MultiPolygon 物件，不知道粒子數，
            # 這裡才把 poly_id 廣播成 (n,)（全部指向唯一的相異幾何索引 0）。
            poly_id = np.zeros(n, dtype=np.int64)
        else:
            poly_id = region_arrays.poly_id
        region_ring_x = region_arrays.ring_x
        region_ring_y = region_arrays.ring_y
        region_ring_vertex_offsets = region_arrays.ring_vertex_offsets
        region_subpoly_ring_offsets = region_arrays.subpoly_ring_offsets
        region_poly_subpoly_offsets = region_arrays.poly_subpoly_offsets
    else:
        (
            poly_id, region_ring_x, region_ring_y,
            region_ring_vertex_offsets, region_subpoly_ring_offsets, region_poly_subpoly_offsets,
        ) = _empty_region_arrays(n)

    if has_local_region:
        local_arrays = prepare_interest_region_arrays(
            local_region_polygon if local_region_polygon is not None else local_region_polygons
        )
        if local_region_polygon is not None:
            local_poly_id = np.zeros(n, dtype=np.int64)
        else:
            local_poly_id = local_arrays.poly_id
        local_region_ring_x = local_arrays.ring_x
        local_region_ring_y = local_arrays.ring_y
        local_region_ring_vertex_offsets = local_arrays.ring_vertex_offsets
        local_region_subpoly_ring_offsets = local_arrays.subpoly_ring_offsets
        local_region_poly_subpoly_offsets = local_arrays.poly_subpoly_offsets
    else:
        (
            local_poly_id, local_region_ring_x, local_region_ring_y,
            local_region_ring_vertex_offsets, local_region_subpoly_ring_offsets,
            local_region_poly_subpoly_offsets,
        ) = _empty_region_arrays(n)

    if needs_utc_ns_values:
        substep_utc_ns_values = np.asarray(
            [int(substep_utc_ns(idt)) for idt in range(1, ndeltp + 1)], dtype=np.int64
        )
    else:
        substep_utc_ns_values = np.zeros(max(ndeltp, 1), dtype=np.int64)

    dep_row = np.full(n, -1, dtype=np.int64)
    dep_x = np.zeros(n, dtype=np.float64)
    dep_y = np.zeros(n, dtype=np.float64)
    dep_z = np.zeros(n, dtype=np.float64)
    dep_utc_ns = np.zeros(n, dtype=np.int64)

    local_dep_row = np.full(n, -1, dtype=np.int64)
    local_dep_x = np.zeros(n, dtype=np.float64)
    local_dep_y = np.zeros(n, dtype=np.float64)
    local_dep_z = np.zeros(n, dtype=np.float64)
    local_dep_utc_ns = np.zeros(n, dtype=np.int64)

    # ``local_first_exit_pending`` 由呼叫端配置並跨小時持續傳入（見
    # ``mirror_rk4_substep_loop`` docstring 同名參數）；沒有 local domain
    # 判定時仍要餵一個型別穩定的陣列給 kernel，但 kernel 不會讀寫它
    # （``has_local_region`` 為假時整段判定被跳過）。
    local_pending_arg = (
        local_first_exit_pending if local_first_exit_pending is not None
        else np.zeros(n, dtype=bool)
    )

    elem_is_dry_i64 = np.ascontiguousarray(composed.elem_is_dry, dtype=np.int64)

    if per_particle_settling:
        settle_mask_arg = np.ascontiguousarray(settling_effective_mask)
        settle_arg = np.ascontiguousarray(particle_settling_mps)
    else:
        # 非逐粒子模式 kernel 不會讀這兩個引數；給同型別的極小陣列即可。
        settle_mask_arg = np.zeros((1, 1), dtype=np.bool_)
        settle_arg = np.zeros(1, dtype=np.float64)

    dep_count, local_dep_count, fail_substep, fail_count = kernel(
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
        noise_kind_code, vertical_mode_code, float(kz_constant_m2ps), bool(visser_correction),
        hf1_arr, hf2_arr, kv1_arr, kv2_arr,
        seed_lo, seed_hi, step_counter,
        bool(has_interest_region), poly_id,
        region_ring_x, region_ring_y, region_ring_vertex_offsets,
        region_subpoly_ring_offsets, region_poly_subpoly_offsets,
        bool(has_local_region), local_poly_id,
        local_region_ring_x, local_region_ring_y, local_region_ring_vertex_offsets,
        local_region_subpoly_ring_offsets, local_region_poly_subpoly_offsets,
        local_pending_arg,
        native_status if native_status is not None else np.zeros(n, dtype=np.int32),
        substep_utc_ns_values,
        dep_row, dep_x, dep_y, dep_z, dep_utc_ns,
        local_dep_row, local_dep_x, local_dep_y, local_dep_z, local_dep_utc_ns,
        needs_fallback, settle_mask_arg, settle_arg,
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
                    status=NATIVE_STATUS_LEFT_FLOW_DOMAIN,
                    x_m=float(dep_x[i]),
                    y_m=float(dep_y[i]),
                    z_m=float(dep_z[i]),
                    utc_ns=int(dep_utc_ns[i]),
                )
            )

    if local_departures is not None and local_dep_count > 0:
        for i in range(local_dep_count):
            local_departures.append(
                LocalDomainFirstExitRecord(
                    row_particle_index=int(local_dep_row[i]),
                    x_m=float(local_dep_x[i]),
                    y_m=float(local_dep_y[i]),
                    z_m=float(local_dep_z[i]),
                    utc_ns=int(local_dep_utc_ns[i]),
                )
            )

    diag = SubstepDiagnostics()
    diag.n_substeps = ndeltp
    diag.n_particle_substeps = ndeltp * n
    diag.time_move_s = _time.perf_counter() - t_start
    return diag
