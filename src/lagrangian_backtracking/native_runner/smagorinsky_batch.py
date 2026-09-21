"""WP-14 任務 C：Smagorinsky 水平擴散係數場的 Numba 忠實移植（實驗性）。

背景與動機
----------
``backward_driver.compute_smagorinsky_kh_field`` 透過 ``_TrackerMethodShim``
呼叫 BayTrace 原版、未經修改的 ``ParticleTracker._compute_smagorinsky_hf2``
（``baytrace/tracker.py`` 1381-1441 行）。WP-13 量測：67,600 節點×48 層的
單小時合成成本裡，Smagorinsky 占 36.8%（8.4 秒），僅次於 Stokes（62.2%）。

讀原版原始碼（見上方行號）發現耗時的根源：該方法分兩步——

1. **元素級應變率**（``hvis_e``）：對每一層 ``k``（Python 迴圈，``nvrt``
   次疊代），用 NumPy 向量化一次算完全部 ``ne`` 個元素的速度梯度與應變率。
   這一步本身已經向量化，成本大致與 ``nvrt * ne`` 成正比，不是本模組要
   解決的瓶頸。
2. **節點級反距離加權平均**（``hf2_field``）：**對每一個節點** ``i``
   （Python 迴圈，``n_nodes`` 次疊代）取出鄰接元素清單，**再對每一層**
   ``k``（巢狀 Python 迴圈，``nvrt`` 次疊代）呼叫一次
   ``np.dot(hvis_e[k, elems], weights) / slm``——這是「Python 迴圈裡呼叫
   極小陣列（通常只有 3-8 個元素）的 NumPy 函式」的典型反模式：
   ``n_nodes * nvrt`` 次疊代（67,600×48 ≈ 324 萬次）各自付出一次 Python
   直譯器開銷＋NumPy 函式呼叫開銷＋（視 BLAS 後端而定的）極短陣列 BLAS
   呼叫開銷，這些開銷與真正的浮點運算量完全不成比例，是本模組要消除的
   對象。

本模組把這個節點級迴圈（連同它依賴的元素級應變率）整段改寫成單一
``@njit`` 批次函式：迴圈本身還在（``nvrt * n_nodes`` 個組合仍然要逐一算），
但迴圈跑在編譯後的機器碼裡，不是逐次從 Python 直譯器與 NumPy C 層之間來回
呼叫，因此可以消除上述反模式的固定開銷。**運算式本身不改**：向量分量、
權重公式、加總順序都逐行對應原版程式碼，見下方 ``compute_smagorinsky_
kh_field_numba`` 各步驟的行內註解與原版行號對照。

位元相同性：誠實但未知，需要實測證明
----------------------------------------
原版節點級加總使用 ``np.dot(hvis_e[k, elems], weights)``——這是 NumPy 對
一維 ``float64`` 陣列的內積，實際呼叫哪一種加總順序（是否呼叫 BLAS
``ddot``、BLAS 內部是否用迴圈展開／SIMD 分段合計）由 NumPy 版本與連結的
BLAS 後端決定，屬於「未在語言層明訂求和結合律」的浮點運算，**不保證**與
本模組手寫的簡單序列迴圈（``total += hvis_e[k, elem] * weight``）逐位元
相同——即使兩者在數學上是同一個內積，浮點加法不滿足結合律，不同的求和
分組方式可能在最後一兩個 ulp 產生差異。``nn``（鄰接元素數）在典型三角網格
通常只有 4-8，差異若存在，預期非常小，但「預期很小」不等於「證明位元
相同」，因此本模組**不假設**兩者相同，一律由
``tests/test_native_smagorinsky_batch.py`` 對真實／合成網格實測
``np.array_equal``，並在無法通過時忠實記錄最大相對誤差——是否要接線到
正式路徑（``compute_smagorinsky_kh_field`` 的 ``use_numba`` 選項）由測試
結果與最終報告的結論決定，本模組本身只負責提供這個獨立實作與量測入口，
不自行斷言「應該足夠接近」。

科學與資料契約
----------------
本模組只搬移既有 Smagorinsky 公式的**計算方式**，不改變、不重新推導
物理假設；輸入輸出的物理意義、單位（m²/s）與
``backward_driver.compute_smagorinsky_kh_field`` 完全一致。依規範不修改
BayTrace 原始碼，本模組是 LBT 自己的獨立實作，供任務 C 的位元相同性測試
使用。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

__all__ = [
    "SmagorinskyGridArrays",
    "compute_smagorinsky_kh_field_reference",
    "compute_smagorinsky_kh_field_numba",
]


@dataclass(frozen=True, slots=True)
class SmagorinskyGridArrays:
    """``compute_smagorinsky_kh_field_numba`` 所需的靜態網格拓樸欄位。

    全部欄位都對應 BayTrace ``SchismGrid`` 同名屬性（見
    ``baytrace/grid.py`` ``compute_dldxy``／``compute_node_to_element``／
    ``__init__``），呼叫端（``backward_driver.py``）從已受控載入的
    ``driver_grid.schism_grid`` 原樣抽出，不重新計算、不修改任何一個值。

    Attributes:
        elnode: ``(ne, >=3)`` 元素節點連接（本模組只用前三欄，即三角形
            的三個節點索引；BayTrace 允許四邊形，但 Smagorinsky 公式
            （見 ``tracker.py`` 1399-1401 行）本身只讀前三欄，本模組忠實
            沿用這個既有限制，不擴充成四邊形）。
        area: ``(ne,)`` 元素**帶號**面積（原版 ``self.area``；本模組會在
            內部取絕對值，與原版 ``hvis_e`` 公式的 ``area_abs`` 一致）。
        dldxy: ``(3, 2, ne)`` 形函數導數（原版 ``compute_dldxy`` 輸出）。
        nne: ``(n_nodes,)`` 每個節點的鄰接元素數。
        indel: ``(max_nei, n_nodes)`` 每個節點的鄰接元素索引；只有前
            ``nne[i]`` 列對節點 ``i`` 有效，其餘列（含未使用的 padding）
            不會被讀取（比照原版 ``elems = indel[:nn, i]`` 的切片方式）。
        x, y: ``(n_nodes,)`` 節點座標（公尺，投影座標，非經緯度）。
        xctr, yctr: ``(ne,)`` 元素形心座標（公尺）。
    """

    elnode: np.ndarray
    area: np.ndarray
    dldxy: np.ndarray
    nne: np.ndarray
    indel: np.ndarray
    x: np.ndarray
    y: np.ndarray
    xctr: np.ndarray
    yctr: np.ndarray


def compute_smagorinsky_kh_field_reference(
    grid: SmagorinskyGridArrays,
    *,
    u_field2: np.ndarray,
    v_field2: np.ndarray,
    horcon: float,
) -> np.ndarray:
    """純 Python／``math`` 逐點參考實作（不經 NumPy 向量化、不經 Numba）。

    依 ``AGENTS.md``「先建立純 NumPy 參考核心」的要求撰寫，但因為本模組
    要驗證的正是「逐節點逐層 Python 迴圈」這個計算形狀本身，這裡刻意連
    NumPy 向量化都不用（純 ``math`` 逐元素運算），確保與
    ``compute_smagorinsky_kh_field_numba`` 除了「有沒有經過 Numba 編譯」
    以外，計算邏輯逐行對應、容易人工核對，作為獨立於 BayTrace 原版、也
    獨立於 Numba 版本的第三方交叉驗證基準（三方兩兩互相驗證：本函式 vs
    Numba 版本 vs BayTrace 原版，見測試檔）。

    Args:
        grid: 見 ``SmagorinskyGridArrays``。
        u_field2, v_field2: ``(nvrt, n_nodes)`` 純海流速度（不含 Stokes、
            不含沉降），語意與 ``compute_smagorinsky_kh_field`` 相同。
        horcon: Smagorinsky 係數。

    Returns:
        ``(nvrt, n_nodes)`` 水平擴散係數場，單位 m²/s。
    """

    import math

    elnode = grid.elnode
    ne = elnode.shape[0]
    nvrt = u_field2.shape[0]
    n_nodes = grid.x.shape[0]

    hvis_e = [[0.0] * ne for _ in range(nvrt)]
    for e in range(ne):
        n1 = int(elnode[e, 0])
        n2 = int(elnode[e, 1])
        n3 = int(elnode[e, 2])
        area_abs = abs(float(grid.area[e]))
        d00 = float(grid.dldxy[0, 0, e])
        d01 = float(grid.dldxy[0, 1, e])
        d10 = float(grid.dldxy[1, 0, e])
        d11 = float(grid.dldxy[1, 1, e])
        d20 = float(grid.dldxy[2, 0, e])
        d21 = float(grid.dldxy[2, 1, e])
        for k in range(nvrt):
            u1 = float(u_field2[k, n1])
            u2 = float(u_field2[k, n2])
            u3 = float(u_field2[k, n3])
            v1 = float(v_field2[k, n1])
            v2 = float(v_field2[k, n2])
            v3 = float(v_field2[k, n3])
            dudx = u1 * d00 + u2 * d10 + u3 * d20
            dudy = u1 * d01 + u2 * d11 + u3 * d21
            dvdx = v1 * d00 + v2 * d10 + v3 * d20
            dvdy = v1 * d01 + v2 * d11 + v3 * d21
            strain = math.sqrt(dudx * dudx + dvdy * dvdy + 0.5 * (dudy + dvdx) ** 2)
            hvis_e[k][e] = horcon * area_abs * strain

    hf2_field = np.zeros((nvrt, n_nodes), dtype=np.float64)
    for i in range(n_nodes):
        nn = int(grid.nne[i])
        if nn == 0:
            continue
        elems = [int(grid.indel[j, i]) for j in range(nn)]
        xi = float(grid.x[i])
        yi = float(grid.y[i])
        weights = []
        for e in elems:
            dx = xi - float(grid.xctr[e])
            dy = yi - float(grid.yctr[e])
            rl = math.sqrt(dx * dx + dy * dy)
            rl = max(rl, 1.0e-10)
            weights.append(1.0 / rl)
        slm = sum(weights)
        if slm > 0.0:
            for k in range(nvrt):
                total = 0.0
                for j, e in enumerate(elems):
                    total += hvis_e[k][e] * weights[j]
                hf2_field[k, i] = total / slm

    return hf2_field


@njit(cache=True)
def _smagorinsky_element_strain_numba(
    elnode: np.ndarray,
    area: np.ndarray,
    dldxy: np.ndarray,
    u_field2: np.ndarray,
    v_field2: np.ndarray,
    horcon: float,
    hvis_e_out: np.ndarray,
) -> None:
    """元素級應變率（原版 ``_compute_smagorinsky_hf2`` 步驟 1，
    ``tracker.py`` 1397-1420 行）。逐元素、逐層計算，運算式與求和順序
    （``u1*d0 + u2*d1 + u3*d2``，逐項相加，非另外分組）逐行對應原版。
    """

    ne = elnode.shape[0]
    nvrt = u_field2.shape[0]
    for e in range(ne):
        n1 = elnode[e, 0]
        n2 = elnode[e, 1]
        n3 = elnode[e, 2]
        area_abs = abs(area[e])
        d00 = dldxy[0, 0, e]
        d01 = dldxy[0, 1, e]
        d10 = dldxy[1, 0, e]
        d11 = dldxy[1, 1, e]
        d20 = dldxy[2, 0, e]
        d21 = dldxy[2, 1, e]
        for k in range(nvrt):
            u1 = u_field2[k, n1]
            u2 = u_field2[k, n2]
            u3 = u_field2[k, n3]
            v1 = v_field2[k, n1]
            v2 = v_field2[k, n2]
            v3 = v_field2[k, n3]
            dudx = u1 * d00 + u2 * d10 + u3 * d20
            dudy = u1 * d01 + u2 * d11 + u3 * d21
            dvdx = v1 * d00 + v2 * d10 + v3 * d20
            dvdy = v1 * d01 + v2 * d11 + v3 * d21
            strain = np.sqrt(dudx * dudx + dvdy * dvdy + 0.5 * (dudy + dvdx) ** 2)
            hvis_e_out[k, e] = horcon * area_abs * strain


@njit(cache=True)
def _smagorinsky_node_average_numba(
    nne: np.ndarray,
    indel: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    xctr: np.ndarray,
    yctr: np.ndarray,
    hvis_e: np.ndarray,
    hf2_field_out: np.ndarray,
) -> None:
    """節點級反距離加權平均（原版步驟 2，``tracker.py`` 1428-1441 行）。

    這是原版效能瓶頸所在（``n_nodes * nvrt`` 個 Python 迴圈疊代、各自呼叫
    一次極小陣列的 ``np.dot``），本函式把同一組運算式搬進 njit 迴圈，
    權重計算與內積求和順序（``for j in range(nn): total += hvis_e[k,elem]
    *weight[j]``，逐項循序相加）逐行對應原版 ``np.dot`` 想表達的內積定義；
    **是否與原版 NumPy／BLAS 實際執行的求和順序逐位元相同，由呼叫端的
    測試實測決定，本函式不自行假設**（見模組 docstring「位元相同性」
    一節）。
    """

    n_nodes = x.shape[0]
    nvrt = hvis_e.shape[0]
    max_nei = indel.shape[0]
    weights = np.empty(max_nei, dtype=np.float64)
    for i in range(n_nodes):
        nn = nne[i]
        if nn == 0:
            continue
        xi = x[i]
        yi = y[i]
        slm = 0.0
        for j in range(nn):
            e = indel[j, i]
            dx = xi - xctr[e]
            dy = yi - yctr[e]
            rl = np.sqrt(dx * dx + dy * dy)
            if rl < 1.0e-10:
                rl = 1.0e-10
            w = 1.0 / rl
            weights[j] = w
            slm += w
        if slm > 0.0:
            for k in range(nvrt):
                total = 0.0
                for j in range(nn):
                    e = indel[j, i]
                    total += hvis_e[k, e] * weights[j]
                hf2_field_out[k, i] = total / slm


def compute_smagorinsky_kh_field_numba(
    grid: SmagorinskyGridArrays,
    *,
    u_field2: np.ndarray,
    v_field2: np.ndarray,
    horcon: float,
) -> np.ndarray:
    """``compute_smagorinsky_kh_field_reference`` 的 Numba 批次版本，
    參數與回傳語意完全相同，供 ``backward_driver.compute_smagorinsky_
    kh_field`` 視測試結果決定要不要接線使用（見該函式 ``use_numba``
    參數與模組層級「位元相同性」說明）。

    分兩個 ``@njit`` 函式呼叫（``_smagorinsky_element_strain_numba``／
    ``_smagorinsky_node_average_numba``），對應原版的兩個步驟，中間的
    ``hvis_e`` 中繼陣列由本函式配置、傳遞，呼叫端看不到這個細節。
    """

    elnode = np.ascontiguousarray(grid.elnode[:, :3], dtype=np.int64)
    area = np.ascontiguousarray(grid.area, dtype=np.float64)
    dldxy = np.ascontiguousarray(grid.dldxy, dtype=np.float64)
    nne = np.ascontiguousarray(grid.nne, dtype=np.int64)
    indel = np.ascontiguousarray(grid.indel, dtype=np.int64)
    x = np.ascontiguousarray(grid.x, dtype=np.float64)
    y = np.ascontiguousarray(grid.y, dtype=np.float64)
    xctr = np.ascontiguousarray(grid.xctr, dtype=np.float64)
    yctr = np.ascontiguousarray(grid.yctr, dtype=np.float64)
    u2 = np.ascontiguousarray(u_field2, dtype=np.float64)
    v2 = np.ascontiguousarray(v_field2, dtype=np.float64)

    ne = elnode.shape[0]
    nvrt = u2.shape[0]
    n_nodes = x.shape[0]

    hvis_e = np.empty((nvrt, ne), dtype=np.float64)
    _smagorinsky_element_strain_numba(elnode, area, dldxy, u2, v2, float(horcon), hvis_e)

    hf2_field = np.zeros((nvrt, n_nodes), dtype=np.float64)
    _smagorinsky_node_average_numba(nne, indel, x, y, xctr, yctr, hvis_e, hf2_field)
    return hf2_field
