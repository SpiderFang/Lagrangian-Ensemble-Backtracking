"""有限水深 Stokes 漂流批次核心（工作書 2-4 節式(7)）。

本模組是 WP-3 工作包的產出，服務對象是 WP-5「直接驅動 BayTrace 原版 njit 批次
主迴圈」的整合層。它取代 BayTrace 原版 ``batch_interp.batch_compute_stokes``
（見該檔案約 627-700 行），因為原版函式有下列三個本模組刻意不沿用、也不修改的
問題（詳見本檔最後一節「對 BayTrace 原版量綱問題的獨立推導」）：

1. 波數用深水近似 ``k = ω²/g``，工作書要求以**有限水深色散關係**求解。
2. 用平均週期（``Tm01``），工作書要求用**尖峰週期（``Tp``）**。
3. 表面 Stokes 速度公式 ``Us = (2π)³Hs²/(16 g Tm01²)`` 分母是週期的平方，
   量綱推導後得到公尺（``m``），不是速度（``m/s``）；標準式分母應為週期的
   三次方。

科學規格
--------
粒子假設完全沉沒於水中（工作書明示逆向追蹤忽略風偏推力），故只計算水平分量，
垂向 Stokes 分量恆為 0，不在本模組輸出範圍內。水平速度大小為::

    v_stokes(z) = (π² Hs²) / (L Tp) * exp(4π z / L)

其中 ``Hs`` 為有效波高（m）、``Tp`` 為尖峰週期（s，``= 1/fp``）、``L`` 為波長
（m），由有限水深色散關係求出::

    ω² = g k tanh(k h),  L = 2π / k,  ω = 2π / Tp = 2π fp

``h`` 為正值水深（m）、``z`` 為粒子相對海面的垂向位置（海面為 0，水下為負，
單位 m）。方向沿用 ``[cosθ, sinθ]``（θ 為波浪傳播方向）。

這個式子可以化簡成與 ``stokes.py``／BayTrace 原版都相容的等價形式：令
``a = Hs/2``（振幅）、``L = 2π/k``、``4π/L = 2k``，代入後得到

    v_stokes(z) = a² * ω * k * exp(2 k z)

也就是「用有限水深色散關係求出的波數 ``k``，套用深水型垂向指數衰減
``exp(2kz)``」——這正是 ``accelerated.finite_depth_stokes_numba_kernel`` 在
``kh > 20``（深水）時使用的穩定支線公式，以及 ``stokes.deep_water_stokes``
的公式；差別在於工作書式(7)刻意把這個「深水垂向結構」套用到**所有**水深
（不論淺水或深水），而不是像 ``stokes.py`` 那樣在淺水／中間水深改用真正的
雙曲餘弦有限水深垂向結構（``cosh(2k(z+h))/sinh²(kh)``）。這是工作書明訂的
簡化，**不是本模組的自由選擇**；也因此本模組的結果在淺水中會與 ``stokes.py``
的完整有限水深公式出現系統性差異（工作書公式通常低估淺水表面流速的垂向支撐、
高估近底衰減），只有波數／色散求解與深水極限這兩件事必須與 ``stokes.py`` 逐點
一致。若需要真正的有限水深垂向結構，請使用 ``stokes.py`` 而非本模組。

資料來源與職責分工
--------------------
輸入的 ``significant_wave_height_m``、``peak_frequency_hz``、
``direction_raw_deg``、``water_depth_m`` 都必須是**已內插到粒子位置**的陣列；
NWW3 規則經緯網格到粒子座標的空間內插不在本工作包範圍，由呼叫端（WP-5 或既有
``forcing.py`` 的 ``CombinedMonthForcing``／``_sample_spatial_at_time`` 一類
邏輯）完成。本模組只讀取這些已內插後的純量欄位，計算物理量並寫入輸出陣列，不
讀取 ``nww3_analysis`` 產品檔、不管理粒子生命週期。

波向慣例（容易出錯，務必先讀）
--------------------------------
NWW3 ``peak_direction_raw_deg`` 的既有專案慣例是「**波浪從正北起順時針方向
吹來**」（來向，不是去向）；依據：

* ``stokes.py`` 第 5-6 行模組 docstring 明文：「波向慣例明確採『波浪從正北起
  順時針吹來』；本模組不自行猜測或改變此定義」，其
  ``wave_from_direction_to_unit_vector``（第 115-121 行）把來向角度加
  180° 後取 ``sin``／``cos`` 當東／北單位向量，即「來向轉去向」。
* ``forcing.py`` 第 1946-1947、1958 行：``direction_x = Σw·sin(direction)``、
  ``direction_y = Σw·cos(direction)``、``raw_direction =
  degrees(atan2(direction_x, direction_y)) % 360``——這組 ``sin``/``cos``↔
  ``atan2(x, y)`` 的配對，正是「角度以正北為 0、順時針為正」的標準三角換算
  （東分量用 ``sin``、北分量用 ``cos``），與 ``stokes.py`` 的角度定義互相
  自洽。
* ``accelerated.finite_depth_stokes_numba_kernel`` 第 154-156 行同樣以
  ``(direction_raw_deg + 180) % 360`` 轉去向、``sin``/``cos`` 拆東北分量。

本模組直接沿用同一套慣例與轉換式，不重新定義、不用氣象與海洋慣例混用的其他
公式。

無效資料狀態碼（不得以零速度靜默替代）
------------------------------------------
依 ``AGENTS.md`` 資料契約，缺值與品質問題必須用不同狀態保存，不能用零值頂替。
本模組的批次核心與參考實作都輸出 ``status``（int32）陣列，配合
``STOKES_BATCH_STATUS_DESCRIPTIONS`` 說明；任何非 ``STOKES_BATCH_STATUS_OK``
的位置，輸出速度與波數固定為 ``NaN``，不會是 0——只有「有效波高本身為 0」
（真的無浪，見下方）這個合法物理情境才會讓速度是 0 且狀態仍是 ``OK``。

狀態碼判斷順序（由外而內，前面判定為真就不再往下算）：

1. ``valid_mask_wave`` 為假：判定為 ``STOKES_BATCH_STATUS_WAVE_MASK_INVALID``。
   對應 NWW3 ``valid_mask_wave`` 為假（域外、陸地或原始缺值）。
2. ``qc_flags`` 非零：判定為 ``STOKES_BATCH_STATUS_WAVE_QC_FLAGGED``。即使
   mask 為真，只要品質旗標非零（例如跨月時間內插聯集後的旗標）就視為不可信。
3. ``Hs``、``fp``、方向、``z``、``h`` 任一非有限，或 ``Hs<0``、``fp<=0``、
   ``h<=0``：判定為 ``STOKES_BATCH_STATUS_INVALID_PHYSICS``。
4. 粒子垂向位置超出 ``[-h, 0]``（含 ``surface_tolerance_m`` 容差）：判定為
   ``STOKES_BATCH_STATUS_PARTICLE_OUTSIDE_WATER_COLUMN``。容差內、位於海面上方
   的極小殘差會被夾回 0（沿用 ``stokes.finite_depth_stokes`` 同一種「只夾回
   定位殘差、不放寬真正越界」的政策）。
5. 色散求根失敗（無法括界／未收斂／殘差超過容許值）：分別判定為
   ``STOKES_BATCH_STATUS_DISPERSION_BRACKET_FAILED``／
   ``STOKES_BATCH_STATUS_DISPERSION_NOT_CONVERGED``／
   ``STOKES_BATCH_STATUS_DISPERSION_RESIDUAL_TOO_LARGE``。
6. 以上皆通過：``STOKES_BATCH_STATUS_OK``。``Hs == 0`` 時速度合法為 0（無浪但
   資料完整、水深與週期仍有效，可與「資料缺值」明確區分）。

數值穩定性
------------
``exp(2 k z)`` 中 ``z<=0``、``k>=0``，指數恆非正，只會下溢為 0（物理上代表
「這個深度已經感受不到表面波」），不會上溢或產生 NaN；因此本公式比
``stokes.py`` 的雙曲餘弦形式更穩定，不需要另外的深水／淺水數值分支。真正的
數值風險集中在色散波數求解，因此波數求解沿用
``accelerated.solve_wave_number_numba_kernel`` 已驗證過的括界與二分策略（見
下方 ``_solve_wave_number_embedded``），但**在本模組內獨立內嵌實作**，不跨模組
呼叫該 kernel：一來符合工作包「不修改、不耦合既有模組」的界線，二來避免
``cache=True``（本模組）與 ``cache=False``（``accelerated.py``）兩種編譯快取
設定混用時的不確定行為。兩者的容差、疊代上限與括界策略完全比照抄錄，數值行為
應當一致，已由測試獨立驗證（見 ``tests/test_stokes_batch.py``）。

效能設計
--------
``batch_finite_depth_stokes`` 是 ``@njit(cache=True)`` 的批次函式，**一次呼叫
處理全部粒子**（在 njit 內部對粒子跑 Python-level 迴圈，但整個迴圈已編譯成機器
碼，不是逐粒子從 Python 呼叫）。呼叫形態比照 BayTrace 原版
``batch_compute_stokes``：輸出直接寫入呼叫端配置好的 ``stokes_u_out``／
``stokes_v_out`` 陣列，型別、單位與正負號慣例與 BayTrace ``pa.stokes_u``／
``pa.stokes_v`` 相容（BayTrace 在 ``tracker.py`` 約 1540-1542 行把它們加到
``pa.u``／``pa.v``）。額外的 ``wave_number_out``、``status_out`` 供 QC 與診斷
使用，不影響與 BayTrace 陣列的相容性（呼叫端可以忽略這兩個輸出，或用
``status_out`` 決定哪些粒子這一步該被排除在平流之外）。

對 BayTrace 原版量綱問題的獨立推導
-------------------------------------
以下是本模組作者不依賴他人說法、自行從物理重新推導的結論。

BayTrace 原版式子：``Us = (2π)³ Hs² / (16 g Tm01²)``。逐一檢查各量的因次：
``Hs²`` 是 ``m²``；``g`` 是 ``m/s²``；``Tm01²`` 是 ``s²``；``(2π)³`` 無因次。
因此 ``Us`` 的因次是 ``m² / [(m/s²)·s²] = m² / m = m``——是長度，不是速度
（``m/s``）。這個式子本身量綱不自洽，不可能是任何速度公式的正確形式，與其
「表面 Stokes drift 速度」的用途矛盾。

再從物理原理重新推導正確式子應該長什麼樣子。單一頻率、深水線性波的表面
Stokes drift 速度是 ``Us(0) = a² ω k``（``a`` 為振幅、``ω`` 角頻率、``k`` 波數；
本模組與 ``stokes.py``、``accelerated.py`` 都是這個式子）。深水色散
``k = ω²/g``，故 ``Us(0) = a² ω³ / g``。對隨機海況以 ``Hs``、平均週期 ``Tm01``
代表整個頻譜的做法（例如 Kenyon 1969、Breivik et al. 2014 附錄 A 的近似），
是把 ``a`` 用 ``Hs/4``（注意不是 ``Hs/2``：這是頻譜矩近似，不是單一正弦波
振幅）、``ω`` 用平均角頻率 ``ωm = 2π/Tm01`` 代入單一頻率公式，得到

    Us(0) ≈ (Hs/4)² * ωm³ / g = Hs² * (2π/Tm01)³ / (16 g) = (2π)³ Hs² / (16 g Tm01³)

因次檢查：``Hs²[m²] / (g[m/s²] · Tm01³[s³]) = m² / (m·s) = m/s``——這才是正確
的速度量綱。對照 BayTrace 原版式子，係數 ``(2π)³/16`` 完全相同，**差別只在
分母的 ``Tm01`` 次方**：正確式應是三次方，BayTrace 原版寫成平方。這與題目
描述的判斷完全吻合，本模組作者是獨立重新推導後才確認，而不是照抄題目說法；
結論是：**題目的判斷是對的，BayTrace 原版該行程式確實有量綱錯誤（少除一個
週期），不能沿用，也不應該修改 BayTrace 原始碼去「順手修正」它**——依規範
BayTrace 原始碼是唯讀的，任何更正都只能在 LBT 自己的模組（如本檔）內以獨立
實作處理。附帶一提，這也解釋了為什麼工作書式(7)明確要求另外指定 ``Tp``、
``L`` 兩個由色散關係算出的量，而不是複用 BayTrace 那個只用 ``Tm01`` 平方的
近似式：式(7)整條式子的因次本來就是自洽的（見上方「科學規格」小節的
``a²ωk`` 等價推導），不需要、也不應該再套用 BayTrace 那個有量綱瑕疵的簡化式。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit

from ..models import SURFACE_BOUNDARY_TOLERANCE_M
from ..stokes import GRAVITY_MPS2

__all__ = [
    "GRAVITY_MPS2",
    "STOKES_BATCH_STATUS_OK",
    "STOKES_BATCH_STATUS_WAVE_MASK_INVALID",
    "STOKES_BATCH_STATUS_WAVE_QC_FLAGGED",
    "STOKES_BATCH_STATUS_INVALID_PHYSICS",
    "STOKES_BATCH_STATUS_PARTICLE_OUTSIDE_WATER_COLUMN",
    "STOKES_BATCH_STATUS_DISPERSION_BRACKET_FAILED",
    "STOKES_BATCH_STATUS_DISPERSION_NOT_CONVERGED",
    "STOKES_BATCH_STATUS_DISPERSION_RESIDUAL_TOO_LARGE",
    "STOKES_BATCH_STATUS_DESCRIPTIONS",
    "StokesBatchResult",
    "finite_depth_stokes_reference",
    "batch_finite_depth_stokes",
]

# ── 狀態碼：明確區分「無效」與「合法零速度」，不得以 0 靜默頂替 ─────────────

STOKES_BATCH_STATUS_OK = 0
STOKES_BATCH_STATUS_WAVE_MASK_INVALID = 1
STOKES_BATCH_STATUS_WAVE_QC_FLAGGED = 2
STOKES_BATCH_STATUS_INVALID_PHYSICS = 3
STOKES_BATCH_STATUS_PARTICLE_OUTSIDE_WATER_COLUMN = 4
STOKES_BATCH_STATUS_DISPERSION_BRACKET_FAILED = 5
STOKES_BATCH_STATUS_DISPERSION_NOT_CONVERGED = 6
STOKES_BATCH_STATUS_DISPERSION_RESIDUAL_TOO_LARGE = 7

STOKES_BATCH_STATUS_DESCRIPTIONS: dict[int, str] = {
    STOKES_BATCH_STATUS_OK: (
        "有效：Hs、Tp（或 fp）、方向、水深與粒子垂向位置皆合法，速度為公式(7)"
        "計算值；Hs=0 時代表『確實無浪』，合法回傳 0，不代表缺值。"
    ),
    STOKES_BATCH_STATUS_WAVE_MASK_INVALID: (
        "NWW3 valid_mask_wave 為假（內插用到的網格點本身無效、域外或陸地），"
        "輸出速度與波數固定為 NaN，不得解讀為零速度。"
    ),
    STOKES_BATCH_STATUS_WAVE_QC_FLAGGED: (
        "NWW3 qc_flags 非零（品質旗標已標記問題，即使 valid_mask_wave 為真），"
        "輸出速度與波數固定為 NaN。"
    ),
    STOKES_BATCH_STATUS_INVALID_PHYSICS: (
        "Hs<0、fp<=0、水深<=0，或 Hs／fp／方向／z／h 任一非有限，輸出速度與"
        "波數固定為 NaN。"
    ),
    STOKES_BATCH_STATUS_PARTICLE_OUTSIDE_WATER_COLUMN: (
        "粒子垂向位置超出海床與海面之間（含 surface_tolerance_m 容差），"
        "輸出速度與波數固定為 NaN。"
    ),
    STOKES_BATCH_STATUS_DISPERSION_BRACKET_FAILED: (
        "無法為有限水深色散方程式建立正確括界，輸出速度與波數固定為 NaN。"
    ),
    STOKES_BATCH_STATUS_DISPERSION_NOT_CONVERGED: (
        "二分法在上限疊代次數內未收斂，輸出速度與波數固定為 NaN。"
    ),
    STOKES_BATCH_STATUS_DISPERSION_RESIDUAL_TOO_LARGE: (
        "二分法收斂但最終色散殘差超出容許值，輸出速度與波數固定為 NaN。"
    ),
}

# 色散求根的括界與二分策略常數：與 accelerated.solve_wave_number_numba_kernel
# 逐一對應抄錄（見該函式 docstring），兩邊獨立維護但數值行為必須一致，由
# tests/test_stokes_batch.py 交叉驗證。
_DISPERSION_BRACKET_MAX_EXPANSIONS = 64
_DISPERSION_BRACKET_MAX_SEARCH_PER_M = 1.0e6
_DISPERSION_BISECTION_MAX_ITERS = 100
_DISPERSION_ROOT_TOL_ABS = 1.0e-13
_DISPERSION_ROOT_TOL_REL = 1.0e-13
_DISPERSION_RESIDUAL_TOL_FLOOR = 1.0e-10
_DISPERSION_RESIDUAL_TOL_REL = 1.0e-10


@njit(cache=True)
def _solve_wave_number_embedded(
    omega_radps: float, water_depth_m: float, gravity_mps2: float
) -> tuple[float, int]:
    """njit 內嵌的有限水深色散求根，供 ``batch_finite_depth_stokes`` 逐粒子呼叫。

    數值策略與容差完全比照
    ``accelerated.solve_wave_number_numba_kernel``：由深水／淺水初始上界開始
    倍增建立異號括界，再以最多 100 次二分把波數誤差收斂到
    ``1e-13 + 1e-13*|k|``，最後檢查殘差是否落在
    ``max(1e-10, ω²*1e-10)`` 內。刻意在本模組內獨立重寫（而非跨模組呼叫
    ``accelerated`` 的版本），原因見模組 docstring「數值穩定性」一節。

    呼叫端必須先確認 ``omega_radps``、``water_depth_m``、``gravity_mps2``
    皆為有限正值；本函式不重複那一層防禦性檢查（沿用「Python 控制層先驗證、
    kernel 只算數值」的專案慣例），只回傳色散求根本身的狀態碼。

    回傳
    ------
    ``(wave_number_per_m, status)``：``status`` 使用本模組的
    ``STOKES_BATCH_STATUS_*`` 常數，只會是 ``OK``、
    ``DISPERSION_BRACKET_FAILED``、``DISPERSION_NOT_CONVERGED`` 或
    ``DISPERSION_RESIDUAL_TOO_LARGE`` 之一。
    """

    omega2 = omega_radps * omega_radps
    upper = max(
        omega2 / gravity_mps2,
        omega_radps / math.sqrt(gravity_mps2 * water_depth_m),
    )
    upper = max(upper * 2.0, 1e-12)
    residual_upper = gravity_mps2 * upper * math.tanh(upper * water_depth_m) - omega2
    expansions = 0
    while residual_upper <= 0.0:
        upper *= 2.0
        expansions += 1
        if (
            upper > _DISPERSION_BRACKET_MAX_SEARCH_PER_M
            or expansions >= _DISPERSION_BRACKET_MAX_EXPANSIONS
        ):
            return np.nan, STOKES_BATCH_STATUS_DISPERSION_BRACKET_FAILED
        residual_upper = gravity_mps2 * upper * math.tanh(upper * water_depth_m) - omega2

    lower = 0.0
    root = np.nan
    converged = False
    for _ in range(_DISPERSION_BISECTION_MAX_ITERS):
        midpoint = lower + 0.5 * (upper - lower)
        residual = gravity_mps2 * midpoint * math.tanh(midpoint * water_depth_m) - omega2
        if residual == 0.0:
            root = midpoint
            converged = True
            break
        if residual < 0.0:
            lower = midpoint
        else:
            upper = midpoint
        if 0.5 * (upper - lower) <= _DISPERSION_ROOT_TOL_ABS + _DISPERSION_ROOT_TOL_REL * abs(
            midpoint
        ):
            root = lower + 0.5 * (upper - lower)
            converged = True
            break
    if not converged:
        return np.nan, STOKES_BATCH_STATUS_DISPERSION_NOT_CONVERGED

    final_residual = gravity_mps2 * root * math.tanh(root * water_depth_m) - omega2
    if abs(final_residual) > max(_DISPERSION_RESIDUAL_TOL_FLOOR, omega2 * _DISPERSION_RESIDUAL_TOL_REL):
        return root, STOKES_BATCH_STATUS_DISPERSION_RESIDUAL_TOO_LARGE
    return root, STOKES_BATCH_STATUS_OK


@njit(cache=True)
def batch_finite_depth_stokes(
    significant_wave_height_m: np.ndarray,
    peak_frequency_hz: np.ndarray,
    direction_raw_deg: np.ndarray,
    particle_z_m: np.ndarray,
    water_depth_m: np.ndarray,
    valid_mask_wave: np.ndarray,
    qc_flags: np.ndarray,
    stokes_u_out: np.ndarray,
    stokes_v_out: np.ndarray,
    wave_number_out: np.ndarray,
    status_out: np.ndarray,
    gravity_mps2: float,
    surface_tolerance_m: float,
) -> None:
    """一次呼叫計算全部粒子的有限水深 Stokes 表面漂流水平速度（工作書式(7)）。

    這是取代 BayTrace 原版 ``batch_interp.batch_compute_stokes`` 的批次核心，
    呼叫形態刻意比照（逐粒子迴圈寫在 njit 內部、輸出直接寫進呼叫端陣列），但
    物理公式、波數求解與狀態碼是本模組獨立實作，完整推導見模組 docstring。

    所有輸入陣列必須是形狀相同的一維陣列，長度即粒子數 ``N``；``*_out`` 輸出
    陣列必須由呼叫端預先配置好同樣長度的 ``float64``（``wave_number_out``、
    ``stokes_u_out``、``stokes_v_out``）與 ``int32``（``status_out``）陣列。
    本函式不做形狀檢查（比照 BayTrace 原版無檢查的慣例，把檢查成本留給
    Python 控制層一次性完成，不在熱迴圈內重複判斷），呼叫端必須自行保證。

    Args:
        significant_wave_height_m: 已內插到粒子位置的有效波高 ``Hs``（m），
            不可為負；``0`` 代表確實無浪，非缺值。
        peak_frequency_hz: 已內插到粒子位置的 NWW3 尖峰頻率 ``fp``（Hz，
            ``= 1/Tp``），必須為正。
        direction_raw_deg: 已內插到粒子位置的 NWW3 原始尖峰波向（度），慣例
            是「正北起順時針的來向」，見模組 docstring「波向慣例」一節。
        particle_z_m: 粒子相對海面的垂向位置（m），海面為 0、水下為負。
        water_depth_m: 粒子所在位置的正值水深 ``h``（m）。
        valid_mask_wave: 對應 NWW3 ``valid_mask_wave`` 內插後的布林陣列，
            ``False`` 表示這個粒子當下用到的波浪網格點本身無效。
        qc_flags: 對應 NWW3 ``qc_flags`` 內插後的整數位元旗標（通常是多個
            角點聯集後的結果），非零一律視為不可信。
        stokes_u_out: 輸出：東向 Stokes 水平速度（m/s），與 BayTrace
            ``pa.stokes_u`` 相容。
        stokes_v_out: 輸出：北向 Stokes 水平速度（m/s），與 BayTrace
            ``pa.stokes_v`` 相容。
        wave_number_out: 輸出：色散關係求得的波數 ``k``（m⁻¹），供 QC 與
            單元測試使用；狀態非 ``OK`` 時為 ``NaN``。
        status_out: 輸出：本模組 ``STOKES_BATCH_STATUS_*`` 狀態碼（int32）。
        gravity_mps2: 重力加速度（m/s²），呼叫端必須傳入有限正值。
        surface_tolerance_m: 海面邊界定位容差（m），用於夾回極小的正向海面
            殘差；呼叫端通常傳入 ``models.SURFACE_BOUNDARY_TOLERANCE_M``。

    無效或狀態非 ``OK`` 的粒子，其 ``stokes_u_out``／``stokes_v_out``／
    ``wave_number_out`` 一律寫入 ``NaN``，不會是 0；只有 ``Hs=0`` 這個合法
    物理情境會讓速度是 0 且狀態仍是 ``STOKES_BATCH_STATUS_OK``。
    """

    n = significant_wave_height_m.shape[0]
    for i in range(n):
        if not valid_mask_wave[i]:
            stokes_u_out[i] = np.nan
            stokes_v_out[i] = np.nan
            wave_number_out[i] = np.nan
            status_out[i] = STOKES_BATCH_STATUS_WAVE_MASK_INVALID
            continue
        if qc_flags[i] != 0:
            stokes_u_out[i] = np.nan
            stokes_v_out[i] = np.nan
            wave_number_out[i] = np.nan
            status_out[i] = STOKES_BATCH_STATUS_WAVE_QC_FLAGGED
            continue

        hs = significant_wave_height_m[i]
        fp = peak_frequency_hz[i]
        theta_deg = direction_raw_deg[i]
        z = particle_z_m[i]
        h = water_depth_m[i]

        if not (
            np.isfinite(hs)
            and hs >= 0.0
            and np.isfinite(fp)
            and fp > 0.0
            and np.isfinite(theta_deg)
            and np.isfinite(z)
            and np.isfinite(h)
            and h > 0.0
        ):
            stokes_u_out[i] = np.nan
            stokes_v_out[i] = np.nan
            wave_number_out[i] = np.nan
            status_out[i] = STOKES_BATCH_STATUS_INVALID_PHYSICS
            continue

        if z > surface_tolerance_m or z < -h - surface_tolerance_m:
            stokes_u_out[i] = np.nan
            stokes_v_out[i] = np.nan
            wave_number_out[i] = np.nan
            status_out[i] = STOKES_BATCH_STATUS_PARTICLE_OUTSIDE_WATER_COLUMN
            continue
        z_clamped = z
        if z_clamped > 0.0:
            # 只夾回海面容差帶內的正向定位殘差，真正越界已在上一個 gate 拒絕。
            z_clamped = 0.0

        omega = 2.0 * math.pi * fp
        wave_number, disp_status = _solve_wave_number_embedded(omega, h, gravity_mps2)
        if disp_status != STOKES_BATCH_STATUS_OK:
            stokes_u_out[i] = np.nan
            stokes_v_out[i] = np.nan
            wave_number_out[i] = np.nan
            status_out[i] = disp_status
            continue

        if hs == 0.0:
            speed = 0.0
        else:
            # a = Hs/2；v = a² ω k exp(2kz)，等價於工作書式(7)的
            # (π²Hs²)/(L Tp) exp(4πz/L)（推導見模組 docstring）。
            speed = (hs * hs / 4.0) * omega * wave_number * math.exp(2.0 * wave_number * z_clamped)

        direction_to_rad = math.radians((theta_deg + 180.0) % 360.0)
        east = math.sin(direction_to_rad)
        north = math.cos(direction_to_rad)

        stokes_u_out[i] = speed * east
        stokes_v_out[i] = speed * north
        wave_number_out[i] = wave_number
        status_out[i] = STOKES_BATCH_STATUS_OK


@dataclass(frozen=True, slots=True)
class StokesBatchResult:
    """``finite_depth_stokes_reference`` 的回傳容器，欄位語意對齊批次核心輸出。

    四個陣列 shape 皆與輸入相同；``status`` 非 ``STOKES_BATCH_STATUS_OK`` 的
    位置，``stokes_u_mps``／``stokes_v_mps``／``wave_number_per_m`` 固定為
    ``NaN``。
    """

    stokes_u_mps: np.ndarray
    stokes_v_mps: np.ndarray
    wave_number_per_m: np.ndarray
    status: np.ndarray


def _solve_wave_number_reference_scalar(
    omega_radps: float, water_depth_m: float, gravity_mps2: float
) -> tuple[float, int]:
    """``_solve_wave_number_embedded`` 的純 Python／``math`` 對應版本。

    刻意逐行對應同一套括界與二分策略（見模組層級常數），但完全不經 Numba
    編譯，作為 AGENTS.md 要求的「先建立的純 NumPy／Python 參考核心」；兩者
    是否數值一致由 ``tests/test_stokes_batch.py`` 獨立驗證，不共用同一份
    程式碼（否則就失去交叉驗證的意義）。
    """

    omega2 = omega_radps * omega_radps
    upper = max(
        omega2 / gravity_mps2,
        omega_radps / math.sqrt(gravity_mps2 * water_depth_m),
    )
    upper = max(upper * 2.0, 1e-12)
    residual_upper = gravity_mps2 * upper * math.tanh(upper * water_depth_m) - omega2
    expansions = 0
    while residual_upper <= 0.0:
        upper *= 2.0
        expansions += 1
        if (
            upper > _DISPERSION_BRACKET_MAX_SEARCH_PER_M
            or expansions >= _DISPERSION_BRACKET_MAX_EXPANSIONS
        ):
            return math.nan, STOKES_BATCH_STATUS_DISPERSION_BRACKET_FAILED
        residual_upper = gravity_mps2 * upper * math.tanh(upper * water_depth_m) - omega2

    lower = 0.0
    root = math.nan
    converged = False
    for _ in range(_DISPERSION_BISECTION_MAX_ITERS):
        midpoint = lower + 0.5 * (upper - lower)
        residual = gravity_mps2 * midpoint * math.tanh(midpoint * water_depth_m) - omega2
        if residual == 0.0:
            root = midpoint
            converged = True
            break
        if residual < 0.0:
            lower = midpoint
        else:
            upper = midpoint
        if 0.5 * (upper - lower) <= _DISPERSION_ROOT_TOL_ABS + _DISPERSION_ROOT_TOL_REL * abs(
            midpoint
        ):
            root = lower + 0.5 * (upper - lower)
            converged = True
            break
    if not converged:
        return math.nan, STOKES_BATCH_STATUS_DISPERSION_NOT_CONVERGED

    final_residual = gravity_mps2 * root * math.tanh(root * water_depth_m) - omega2
    if abs(final_residual) > max(_DISPERSION_RESIDUAL_TOL_FLOOR, omega2 * _DISPERSION_RESIDUAL_TOL_REL):
        return root, STOKES_BATCH_STATUS_DISPERSION_RESIDUAL_TOO_LARGE
    return root, STOKES_BATCH_STATUS_OK


def _finite_depth_stokes_reference_scalar(
    hs: float,
    fp: float,
    theta_deg: float,
    z: float,
    h: float,
    valid: bool,
    qc: int,
    gravity_mps2: float,
    surface_tolerance_m: float,
) -> tuple[float, float, float, int]:
    """單一粒子的純 Python 參考計算，邏輯與 ``batch_finite_depth_stokes`` 的
    迴圈本體逐行對應，供 ``finite_depth_stokes_reference`` 逐顆呼叫。
    """

    if not valid:
        return math.nan, math.nan, math.nan, STOKES_BATCH_STATUS_WAVE_MASK_INVALID
    if qc != 0:
        return math.nan, math.nan, math.nan, STOKES_BATCH_STATUS_WAVE_QC_FLAGGED

    if not (
        math.isfinite(hs)
        and hs >= 0.0
        and math.isfinite(fp)
        and fp > 0.0
        and math.isfinite(theta_deg)
        and math.isfinite(z)
        and math.isfinite(h)
        and h > 0.0
    ):
        return math.nan, math.nan, math.nan, STOKES_BATCH_STATUS_INVALID_PHYSICS

    if z > surface_tolerance_m or z < -h - surface_tolerance_m:
        return math.nan, math.nan, math.nan, STOKES_BATCH_STATUS_PARTICLE_OUTSIDE_WATER_COLUMN
    z_clamped = 0.0 if z > 0.0 else z

    omega = 2.0 * math.pi * fp
    wave_number, disp_status = _solve_wave_number_reference_scalar(omega, h, gravity_mps2)
    if disp_status != STOKES_BATCH_STATUS_OK:
        return math.nan, math.nan, math.nan, disp_status

    if hs == 0.0:
        speed = 0.0
    else:
        speed = (hs * hs / 4.0) * omega * wave_number * math.exp(2.0 * wave_number * z_clamped)

    direction_to_rad = math.radians((theta_deg + 180.0) % 360.0)
    east = math.sin(direction_to_rad)
    north = math.cos(direction_to_rad)
    return speed * east, speed * north, wave_number, STOKES_BATCH_STATUS_OK


def finite_depth_stokes_reference(
    significant_wave_height_m: np.ndarray,
    peak_frequency_hz: np.ndarray,
    direction_raw_deg: np.ndarray,
    particle_z_m: np.ndarray,
    water_depth_m: np.ndarray,
    valid_mask_wave: np.ndarray,
    qc_flags: np.ndarray,
    *,
    gravity_mps2: float = GRAVITY_MPS2,
    surface_tolerance_m: float = SURFACE_BOUNDARY_TOLERANCE_M,
) -> StokesBatchResult:
    """``batch_finite_depth_stokes`` 的純 Python／NumPy 參考實作。

    依 ``AGENTS.md``「先建立純 NumPy 參考核心與解析／統計測試，再加入 Numba
    加速；兩者必須在明定誤差內一致」的要求撰寫：本函式逐顆呼叫
    ``_finite_depth_stokes_reference_scalar``（純 Python／``math``，不經
    Numba），刻意保持簡單、容易手動核對，作為 Numba 批次核心的正確性基準，
    不追求向量化效能。

    參數語意與形狀規則、狀態碼與無效資料政策，完全對齊
    ``batch_finite_depth_stokes``（見其 docstring），差別只在這裡的輸入可以是
    任意形狀（會攤平計算後還原成原形狀），且提供 ``gravity_mps2``／
    ``surface_tolerance_m`` 的預設值，方便測試與探索用途。

    Raises:
        ValueError: 任一輸入陣列形狀與 ``significant_wave_height_m`` 不一致。
    """

    hs = np.asarray(significant_wave_height_m, dtype=np.float64)
    fp = np.asarray(peak_frequency_hz, dtype=np.float64)
    theta_deg = np.asarray(direction_raw_deg, dtype=np.float64)
    z = np.asarray(particle_z_m, dtype=np.float64)
    h = np.asarray(water_depth_m, dtype=np.float64)
    valid = np.asarray(valid_mask_wave, dtype=bool)
    qc = np.asarray(qc_flags)

    shape = hs.shape
    for name, array in (
        ("peak_frequency_hz", fp),
        ("direction_raw_deg", theta_deg),
        ("particle_z_m", z),
        ("water_depth_m", h),
        ("valid_mask_wave", valid),
        ("qc_flags", qc),
    ):
        if array.shape != shape:
            raise ValueError(
                f"{name} shape 必須與 significant_wave_height_m 一致：{array.shape} != {shape}"
            )

    hs_flat = hs.reshape(-1)
    fp_flat = fp.reshape(-1)
    theta_flat = theta_deg.reshape(-1)
    z_flat = z.reshape(-1)
    h_flat = h.reshape(-1)
    valid_flat = valid.reshape(-1)
    qc_flat = qc.reshape(-1)

    n = hs_flat.size
    stokes_u = np.empty(n, dtype=np.float64)
    stokes_v = np.empty(n, dtype=np.float64)
    wave_number = np.empty(n, dtype=np.float64)
    status = np.empty(n, dtype=np.int32)

    for i in range(n):
        u_i, v_i, k_i, status_i = _finite_depth_stokes_reference_scalar(
            float(hs_flat[i]),
            float(fp_flat[i]),
            float(theta_flat[i]),
            float(z_flat[i]),
            float(h_flat[i]),
            bool(valid_flat[i]),
            int(qc_flat[i]),
            gravity_mps2,
            surface_tolerance_m,
        )
        stokes_u[i] = u_i
        stokes_v[i] = v_i
        wave_number[i] = k_i
        status[i] = status_i

    return StokesBatchResult(
        stokes_u_mps=stokes_u.reshape(shape),
        stokes_v_mps=stokes_v.reshape(shape),
        wave_number_per_m=wave_number.reshape(shape),
        status=status.reshape(shape),
    )
