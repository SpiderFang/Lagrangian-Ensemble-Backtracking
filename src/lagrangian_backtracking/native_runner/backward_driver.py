"""WP-5：以 BayTrace 原版批次核心驅動的單一批次逆向追蹤驅動層。

背景與範圍
----------
本模組是「原版 BayTrace 核心 + LBT 逆向追蹤」整合鏈的最後一塊：給定同一個
「流場區域、到達時刻、沉降速度」的一批粒子（見 ``particle_input.py`` 的
``group_indices_by_settling_velocity_and_arrival_time``），從到達時刻開始，
以 UTC 遞減的方向逐小時往回推進，直到粒子離開關注區域、達到 30 天回溯上限，
或發生擱淺、沉底、流場缺值、數值失敗等終止事件為止。

使用者的硬性要求是「運算一律採用原版 BayTrace 核心，kernel 一行都不改」。
本模組因此把所有需要逐步驟數值運算的部分，全部交給 BayTrace 原始
``batch_interp.py`` 的 njit 批次函式（``batch_move_rk4``、
``batch_check_boundaries``、``batch_clamp_z``、``batch_interpolate_velocity``）
與 ``tracker.py``、``grid.py`` 裡「不是 njit、但同樣未經修改」的既有輔助方法
（``ParticleTracker._compute_smagorinsky_hf2``、
``ParticleTracker._compute_dry_flags``、``SchismGrid`` 的網格拓撲建構方法）；
本模組自己撰寫的程式碼只負責：

1. 把這些原始函式／方法以「明確來源＋SHA-256」的方式受控載入（見下方
   ``load_baytrace_modules``），風格比照既有 ``baytrace_native_cpu.py`` 的
   ``load_original_cpu``，但因為需要用到 ``batch_interp`` 以外、且該模組
   刻意不涵蓋的 ``grid.py``／``tracker.py``，所以是獨立撰寫的載入器，
   不修改、不呼叫該檔案裡任何底線開頭的私有函式。
2. 把海流、Stokes 漂流（WP-3）、沉降速度合成為 BayTrace kernel 看得懂的
   「總合速度場」陣列（見下方「事實 1」）。
3. 逐小時鏡像 ``ParticleTracker.batch_update_particles`` 的子步迴圈，在
   MOVE 之後插入算子分裂的隨機擴散位移（見下方「事實 2、3」）。
4. 以遞減 UTC 驅動外層小時迴圈、管理粒子狀態機、記錄軌跡與離開事件、
   判斷停止條件。

四項關鍵事實（由主代理確認，已對照原始碼逐行驗證，實作全程遵守）
--------------------------------------------------------------------

**事實 1（速度合成）**：BayTrace 每個子步的 STEP 0 只把 Stokes 加到
``pa.u``／``pa.v``（``tracker.py`` 約 1540-1542 行），但 RK4 的 k2-k4 由
``_interp_velocity_at``（``batch_interp.py`` 1187-1258 行）直接讀
``u_field``／``v_field``／``w_field``，完全看不到那筆加總；STEP 3
（1622 行）又會用純海流內插覆寫掉 ``pa.u``／``pa.v``。因此若比照原版把
Stokes／沉降加在 ``pa.*``，只會拿到 1/6 的 RK4 權重。本模組改為在
每個流場時刻，把「海流＋Stokes＋沉降」合成後的總速度直接寫進交給
kernel 的 ``u_field1/2``／``v_field1/2``／``w_field1/2`` 陣列（見
``compose_total_velocity_fields``），這樣 RK4 全部四階都讀到同一組合成
場，四階都正確，且 kernel 一行都不改。

**事實 2（算子分裂）**：passive 模式（``mod_part=0``，本模組唯一使用的
模式）的 RK4 路徑完全沒有隨機擴散；oil 模式雖然有擴散，但走
``batch_move_oil`` 而非 RK4，且夾帶風偏與浮力（本專案不需要，也不使用）。
依 AGENTS.md「確定性平流與隨機擴散分開驗證，隨機位移不得插入 RK4 的
stage」，本模組讓 ``batch_move_rk4``（確定性、完全未修改）先跑完整個
RK4 步，**之後**才另外施加一次隨機擴散位移，兩者在同一子步內依序執行、
但分屬不同運算階段，不混入 RK4 内部的 k1-k4 加權。

**事實 3（逐子步鏡像）**：``batch_update_particles``
（``tracker.py`` 1443-1692 行）內建完整的
``for idt in range(1, ndeltp+1)`` 子步迴圈，子步之間沒有掛勾可插入擴散；
若只在每小時邊界施加一次擴散，會讓大步跳躍越過小步原本會觸發的邊界反射，
破壞統計等價。本模組因此在 ``mirror_rk4_substep_loop`` 內逐行對應同一個
子步迴圈（呼叫同一批 kernel、同一呼叫順序、同一 ``trat`` 推導公式：
逆向 ``trat=(idt-1)/ndeltp``），只在 MOVE 之後插入擴散這一件事。

**事實 4（遞減 UTC）**：BayTrace 原版主迴圈永遠順向讀檔、逆向只翻轉速度
符號，因此要求 forcing 以反時序供給；WP-2
（``forcing_bridge.NativeForcingBridgeReader``，``ibf=-1``）已經處理好
field1/field2 指派與 ``trat`` 的方向推導。本模組的外層小時迴圈因此以
**遞減 UTC** 依序查詢（見 ``run_backward_batch`` 的 ``forcing_resolver``
參數），從到達時刻往回走到到達時刻減 30 天。

受控載入 BayTrace 原始模組
----------------------------
``load_baytrace_modules`` 需要 ``geometry``、``vertical``、``random``、
``coordinate``、``calendar``、``physics``、``particle_arrays``、``config``、
``batch_interp``、``grid``、``tracker`` 十一個原始檔案（``tracker.py``
在模組層級 import 了前八個，``grid.py``／``batch_interp.py`` 各自 import
``geometry``／``vertical``），比既有 ``baytrace_native_cpu.py`` 的
``_MODULE_NAMES``（只有四個）多，因此無法重用該私有函式；載入邏輯獨立
撰寫但採同一設計原則：明確來源根目錄、拒絕 symlink、記錄每個檔案的
SHA-256、以 ``importlib.util.spec_from_file_location`` 手動依拓樸序
（先被依賴者、後依賴者）建立並註冊到 ``sys.modules['baytrace.<name>']``，
與既有載入器共用同一個 ``baytrace`` package sentinel（若已存在且來源一致
則直接沿用，不重複載入；來源不一致則明確拒絕），讓兩個載入器可以在同一
個行程內共存而不互相踩踏。

本模組只使用下列原始物件，其餘一律不呼叫：

- ``batch_interp.batch_move_rk4``／``batch_check_boundaries``／
  ``batch_clamp_z``／``batch_interpolate_velocity``／
  ``batch_interpolate_diffusivity``（njit，完全未修改，直接以 tracker.py
  逐行核對過的參數順序呼叫）。
- ``grid.SchismGrid``：只呼叫其 ``_compute_element_areas``／
  ``compute_dldxy``／``compute_node_to_element``／``build_side_structure``
  四個「網格拓樸與幾何前處理」方法（純 Python／NumPy，非 njit，不是
  「物理積分核心」），把 WP-1 產出的 ``NativeGridArrays`` 轉成 BayTrace
  這幾個方法期待的屬性配置後呼叫，藉此重用原版的邊結構建構演算法，而不
  是自己重寫一份容易與原版行為分歧的複本。
- ``ParticleTracker._compute_smagorinsky_hf2``／``_compute_dry_flags``：
  這兩個是 ``ParticleTracker`` 的方法，但本身是純 NumPy 向量化計算（非
  njit），只讀取 ``self.params``／``self.grid``／少數流場與狀態屬性；
  本模組不建立完整 ``ParticleTracker``（需要設定檔、netCDF 等大量無關
  相依），而是用一個只帶最少必要屬性的鴨子型別容器（``_TrackerMethodShim``）
  以「未綁定方法」的方式呼叫這兩個方法本體，讓 Smagorinsky 公式與乾濕
  判定完全重用原版程式碼、不修改、不重寫。
- ``particle_arrays.ParticleArrays``：直接使用這個原生 SoA 容器裝粒子
  狀態，不自己另外設計一份欄位不同的容器，確保與 kernel 的欄位、dtype
  期待完全相容。

已知限制（誠實揭露）
----------------------
1. **不含 Python 層 KD-tree 後援**：``ParticleTracker._batch_check_boundaries``
   包裝方法在 njit 級 ``batch_check_boundaries`` 回報
   ``needs_fallback!=0``（一般 fallback 或 death trap）時，會呼叫另一個
   方法 ``_find_new_element_fallback`` 做 Python 層 KD-tree 後援搜尋。
   該方法需要完整 ``ParticleTracker``（``self.geometry``、惰性建立的
   ``self._kdtree`` 等），為避免把整個受控載入與 shim 複雜度大幅推高，
   本模組刻意不重建這一層：``mirror_rk4_substep_loop`` 直接呼叫 njit
   級 ``batch_check_boundaries``，若任何粒子回報 ``needs_fallback!=0``
   就以 ``BackwardDriverError`` fail-closed 中止，不嘗試後援定位。正式
   資料的網格解析度與 ``dtb``（本模組保證 ``<= dt_max_seconds``，預設
   30 秒）應該讓子步位移遠小於元素尺度、極少觸發這個情況，但這件事尚未
   在真實 OCM 網格上驗證，是本工作包尚未消除的風險，請 WP-6／正式跑前
   QC 特別留意 ``BackwardDriverError`` 是否曾經出現。
2. **NWW3 波浪場「內插到節點」的橋接模組已補上，但驗證範圍有限**：
   ``compose_total_velocity_fields``／``compose_current_and_stokes_fields``
   需要的 ``WaveFieldAtNodes``（每節點的 ``Hs``／``fp``／來向／
   ``valid_mask``／``qc_flags``）本來必須由呼叫端自行提供；此限制已由
   同目錄新增的 ``wave_bridge.py``（``NativeWaveBridgeReader`` ／
   ``build_nww3_node_interpolation_weights`` ／ ``gather_nww3_hour_at_nodes``）
   補上：它讀取已驗收的 ``nww3_analysis`` schema 1 月檔（``NWWAnalysisMonth``），
   對追蹤網格節點做一次性雙線性內插權重預計算，之後每小時只做
   向量化 gather＋加權求和（無逐節點 Python 迴圈），波向採 sin/cos
   分量平均後 ``atan2`` 還原（與 ``forcing.py``／``stokes_batch.py`` 的
   「來向、以北為 0 度順時針」慣例一致），四角只要有一角無效或節點
   落在 NWW3 覆蓋範圍外，``valid_mask_wave`` 一律回報 False（視為缺失，
   絕不用 0 靜默填補），時間採「精確整點對齊、缺點即 fail-closed」政策。
   ``NativeWaveBridgeReader.as_wave_field_resolver()`` 回傳的函式可以
   直接當成 ``run_backward_batch(wave_field_resolver=...)`` 的參數。
   **驗證範圍的誠實揭露**：本工作包的測試（``tests/test_native_wave_bridge.py``）
   把 ``gather_nww3_hour_at_nodes`` 的內插結果與 ``forcing.py`` 自身的
   ``NWWAnalysisMonth._sample_spatial_at_time``／``sample_at_time_index``
   參考算法逐點交叉比對、並用合成（非真實）NWW3 月檔驗證遮罩、跨
   0/360 度環向平均、缺時、快取等邊界情況，也有一個端對端測試證明
   接上此橋接後 Stokes 確實會改變粒子軌跡；但**尚未**在真實近岸
   NWW3 產品資料上驗證過覆蓋範圍與資料品質（例如近岸網格是否大量
   落在 ``node_covered_by_nww3=False``、``qc_flags`` 實際分布），
   正式跑之前建議先用真實月檔跑一次 ``NativeWaveBridgeReader`` 並
   統計 ``valid_mask_wave`` 的有效節點比例，確認 Stokes 缺失的空間
   範圍在可接受程度內。
3. **擱淺／沉底判定是簡化的啟發式，不是嚴謹的物理臨界剪應力模型**：見
   ``_apply_stranding_and_deposition`` 的 docstring；工作書也只要求把
   沉降速度當成常數代理物理行為，本身就是簡化，這裡的簡化與其一致但
   仍需在正式報告中揭露。
4. **關注區域穿越偵測解析度是「每個子步後檢查」而非「精確幾何內插交點」**：
   與 ``boundaries.py``（既有逐粒子 NumPy 引擎）不同，本模組為了避免
   逐粒子建立 Shapely 物件（先前量測 48.8 µs/粒子/步），改用 Shapely 2.0
   向量化的 ``contains_xy`` 在**每個子步結束時**對全部粒子一次性判斷是否
   仍在關注區域內；一旦偵測到離開，記錄的「離開位置」就是該子步結束時
   的位置（已在關注區域外），而不是位置與邊界的精確幾何交點。位置誤差
   上界是「一個子步內的最大位移」，因 ``dtb<=dt_max_seconds`` 而受控，
   但這不是 ``boundaries.first_polygon_crossing`` 那種精確線段—多邊形
   交點；如需精確交點，需在偵測到離開後另外用該函式對該粒子的最後一段
   軌跡做一次性後處理（不在本模組範圍內）。
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import threading
import types
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from shapely import contains_xy
from shapely.geometry import Polygon

from ..stokes import GRAVITY_MPS2
from .forcing_bridge import HourlyFieldPair, UnresolvedForcingGapError
from .grid_bridge import NativeGridArrays
from .stokes_batch import (
    STOKES_BATCH_STATUS_OK,
    batch_finite_depth_stokes,
)

# ══════════════════════════════════════════════════════════════════════════
# WP-13 任務 2：底場合成的小時內平行化——只針對 Stokes 全網格計算
# ══════════════════════════════════════════════════════════════════════════
#
# 實測（見最終報告）：``stokes_batch.batch_finite_depth_stokes`` 是
# ``@njit(cache=True)``、**沒有**設定 ``nogil=True`` 的函式；本專案規定不得
# 修改 ``stokes_batch.py``（不在 WP-13 可修改檔案清單內），因此無法在那個
# 檔案裡加上 ``nogil=True`` 或 ``parallel=True``。實測（2 核心沙盒環境，見
# 最終報告的機器規格誠實揭露）證實：4 個 Python 執行緒同時呼叫這個函式，
# 總耗時與序列呼叫完全相同（4.542s vs 4.518s，4 執行緒沒有任何加速）——
# GIL 確實沒有被釋放，``ThreadPoolExecutor`` 對這個函式完全無效。
#
# 因此本模組改用**行程**（``ProcessPoolExecutor``）而非執行緒平行化 Stokes
# 計算：把 (層,節點) 點集合切成 N 段，各自送進行程池呼叫「同一個」未經修改
# 的 ``batch_finite_depth_stokes``，再依原本的索引寫回結果陣列。這個計算
# 是逐點獨立、完全沒有跨點歸約（每個點的 Stokes 解只由自己的波高／週期／
# 波向／水深／深度決定，不依賴其他點），因此**不論怎麼切、切幾段、哪個
# 行程算哪一段，結果都是逐位元相同**——不是「機率上通常一樣」，是這個計算
# 本身的數學結構保證的，這也是任務書「若某項無法位元相同，改用不涉及歸約
# 順序的平行化方式」這句話在本模組唯一能夠安全套用平行化的地方（比較：
# ``compute_smagorinsky_kh_field`` 的節點平均步驟有跨點依賴、且是 BayTrace
# 原始碼的一部分、本模組不得修改，因此本 WP 沒有平行化它，見最終報告的
# 誠實揭露）。
#
# 預設 ``stokes_pool=None``：完全不啟用，呼叫路徑與此優化新增前逐位元
# 相同，不影響任何既有呼叫端或測試。呼叫端只有在明確傳入一個
# ``concurrent.futures.ProcessPoolExecutor``（通常是尾端只剩少數工作單元、
# 其餘核心閒置時才建立／啟用，見 ``hour_major.py``／``hour_major_provider.py``
# 的整合點）時才會真正切段平行送出。


def _stokes_chunk_task(
    significant_wave_height_m: np.ndarray,
    peak_frequency_hz: np.ndarray,
    direction_raw_deg: np.ndarray,
    particle_z_m: np.ndarray,
    water_depth_m: np.ndarray,
    valid_mask_wave: np.ndarray,
    qc_flags: np.ndarray,
    gravity_mps2: float,
    surface_tolerance_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """在（可能是另一個）行程裡對一個點子集合呼叫 ``batch_finite_depth_stokes``。

    模組層級函式（非閉包）：``ProcessPoolExecutor`` 預設 ``spawn`` 啟動方式
    要求任務函式可以被 pickle 匯入，理由與 ``orchestrator._run_single_unit_task``
    的既有說明相同。這裡呼叫的是與序列路徑完全相同、未經任何修改的
    ``batch_finite_depth_stokes``，因此每個點的計算結果與序列呼叫逐位元相同
    （見上方模組層級說明）；本函式只負責配置輸出陣列與呼叫，不含任何額外
    的數值邏輯。
    """

    n = significant_wave_height_m.shape[0]
    stokes_u = np.empty(n, dtype=np.float64)
    stokes_v = np.empty(n, dtype=np.float64)
    wave_number = np.empty(n, dtype=np.float64)
    status = np.empty(n, dtype=np.int32)
    batch_finite_depth_stokes(
        significant_wave_height_m,
        peak_frequency_hz,
        direction_raw_deg,
        particle_z_m,
        water_depth_m,
        valid_mask_wave,
        qc_flags,
        stokes_u,
        stokes_v,
        wave_number,
        status,
        gravity_mps2,
        surface_tolerance_m,
    )
    return stokes_u, stokes_v, wave_number, status


def _evaluate_batch_finite_depth_stokes(
    significant_wave_height_m: np.ndarray,
    peak_frequency_hz: np.ndarray,
    direction_raw_deg: np.ndarray,
    particle_z_m: np.ndarray,
    water_depth_m: np.ndarray,
    valid_mask_wave: np.ndarray,
    qc_flags: np.ndarray,
    *,
    gravity_mps2: float,
    surface_tolerance_m: float,
    stokes_pool: Any | None,
    stokes_worker_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``compose_current_and_stokes_fields`` 的 Stokes 求解分派點：序列或平行。

    ``stokes_pool is None`` 或 ``stokes_worker_count <= 1`` 或點數少於
    ``stokes_worker_count``（切了也沒有意義，行程呼叫的固定開銷會蓋過收益）
    時，走原本的單次呼叫路徑，與這個優化新增前逐位元相同。否則把點集合切成
    ``stokes_worker_count`` 個連續區段（``np.array_split`` 的切法），各自
    ``pool.submit`` 一次 ``_stokes_chunk_task``，收集結果依原索引寫回——
    每個點的計算彼此獨立，切法與收集順序不影響任何一個點的數值結果。
    """

    n = significant_wave_height_m.shape[0]
    if stokes_pool is None or stokes_worker_count <= 1 or n < stokes_worker_count:
        stokes_u = np.empty(n, dtype=np.float64)
        stokes_v = np.empty(n, dtype=np.float64)
        wave_number = np.empty(n, dtype=np.float64)
        status = np.empty(n, dtype=np.int32)
        batch_finite_depth_stokes(
            significant_wave_height_m,
            peak_frequency_hz,
            direction_raw_deg,
            particle_z_m,
            water_depth_m,
            valid_mask_wave,
            qc_flags,
            stokes_u,
            stokes_v,
            wave_number,
            status,
            gravity_mps2,
            surface_tolerance_m,
        )
        return stokes_u, stokes_v, wave_number, status

    boundaries = np.linspace(0, n, stokes_worker_count + 1).astype(np.int64)
    pending: list[tuple[int, int, Any]] = []
    for i in range(stokes_worker_count):
        start, stop = int(boundaries[i]), int(boundaries[i + 1])
        if start == stop:
            continue
        future = stokes_pool.submit(
            _stokes_chunk_task,
            np.ascontiguousarray(significant_wave_height_m[start:stop]),
            np.ascontiguousarray(peak_frequency_hz[start:stop]),
            np.ascontiguousarray(direction_raw_deg[start:stop]),
            np.ascontiguousarray(particle_z_m[start:stop]),
            np.ascontiguousarray(water_depth_m[start:stop]),
            np.ascontiguousarray(valid_mask_wave[start:stop]),
            np.ascontiguousarray(qc_flags[start:stop]),
            gravity_mps2,
            surface_tolerance_m,
        )
        pending.append((start, stop, future))

    stokes_u = np.empty(n, dtype=np.float64)
    stokes_v = np.empty(n, dtype=np.float64)
    wave_number = np.empty(n, dtype=np.float64)
    status = np.empty(n, dtype=np.int32)
    for start, stop, future in pending:
        u_chunk, v_chunk, k_chunk, status_chunk = future.result()
        stokes_u[start:stop] = u_chunk
        stokes_v[start:stop] = v_chunk
        wave_number[start:stop] = k_chunk
        status[start:stop] = status_chunk
    return stokes_u, stokes_v, wave_number, status

__all__ = [
    "BackwardDriverError",
    "NATIVE_STATUS_ACTIVE",
    "NATIVE_STATUS_LEFT_INTEREST_REGION",
    "NATIVE_STATUS_MAX_BACKTRACK",
    "NATIVE_STATUS_STRANDED",
    "NATIVE_STATUS_DEPOSITED",
    "NATIVE_STATUS_FORCING_MISSING",
    "NATIVE_STATUS_NUMERICAL_FAILURE",
    "NATIVE_STATUS_DESCRIPTIONS",
    "BayTraceModuleHandles",
    "load_baytrace_modules",
    "NativeBackwardDriverGrid",
    "build_native_backward_driver_grid",
    "WaveFieldAtNodes",
    "ComposedFieldPair",
    "ComposedCurrentAndStokesFields",
    "compose_current_and_stokes_fields",
    "apply_settling_velocity",
    "compose_total_velocity_fields",
    "compute_smagorinsky_kh_field",
    "compute_wet_dry_flags",
    "estimate_substep_count",
    "particle_bed_z_levels",
    "derive_batch_diffusion_seed",
    "derive_particle_diffusion_seed_lo_hi_u64",
    "counter_based_gaussian_pairs",
    "SubstepDiagnostics",
    "HourlyOverheadDiagnostics",
    "mirror_rk4_substep_loop",
    "BackwardDriverConfig",
    "BackwardDriverResult",
    "run_backward_batch",
]


# ── 狀態碼：與 ``models.ParticleStatus`` 概念對應，但獨立定義 ────────────────
#
# 為什麼不直接重用 ``models.ParticleStatus``：那個 StrEnum 是既有逐粒子
# NumPy 引擎（``engine.py``／``boundaries.py``）的資料契約，其中
# ``FLOW_DOMAIN_EXIT``／``COAST_CONTACT`` 等語意綁定「以 Shapely 精確幾何
# 交點」與「共同流場範圍 vs. 局部分析區」兩層邊界的完整狀態機。本模組是
# 單一批次、單一關注區域、以網格乾濕與子步位置判斷終止的簡化模型，語意
# 不完全相同（例如「離開關注區域」在這裡沒有精確交點、「擱淺」是網格
# 乾元素而非海岸線 Shapely 幾何）。為避免同一個字串常數在兩處被賦予不同
# 精確度的意涵，這裡改用獨立的整數狀態碼（存在 ``numpy`` 陣列中，效能
# 也遠優於逐粒子 Python 物件），並在下表註明與 ``ParticleStatus`` 的概念
# 對應，供 WP-6 後處理時參考如何映射。

NATIVE_STATUS_ACTIVE = 0
"""仍在追蹤中，尚未終止（對應 ``ParticleStatus.ACTIVE``）。"""

NATIVE_STATUS_LEFT_INTEREST_REGION = 1
"""已離開關注區域（``own_local_domain``），逆向追蹤在此正常結束。"""

NATIVE_STATUS_MAX_BACKTRACK = 2
"""回溯超過設定上限（預設 30 天）仍未離開關注區域，強制停止。"""

NATIVE_STATUS_STRANDED = 3
"""擱淺上岸：粒子目前所在元素被 BayTrace 乾濕判定標記為乾。"""

NATIVE_STATUS_DEPOSITED = 4
"""沉底：具下沉行為（沉降速度為負）的粒子連續停留在海床附近。"""

NATIVE_STATUS_FORCING_MISSING = 5
"""流場缺值：本小時的濕節點上出現未經解釋的缺值，整批粒子在此小時停止。"""

NATIVE_STATUS_NUMERICAL_FAILURE = 6
"""數值失敗：粒子位置在某一步之後出現非有限值。"""

NATIVE_STATUS_DESCRIPTIONS: dict[int, str] = {
    NATIVE_STATUS_ACTIVE: "仍在追蹤中，尚未終止。",
    NATIVE_STATUS_LEFT_INTEREST_REGION: "已離開關注區域（own_local_domain），正常結束。",
    NATIVE_STATUS_MAX_BACKTRACK: "回溯已達設定上限，粒子仍在關注區域內，強制停止。",
    NATIVE_STATUS_STRANDED: "擱淺上岸：所在元素被判定為乾。",
    NATIVE_STATUS_DEPOSITED: "沉底：下沉行為粒子持續停留在海床附近。",
    NATIVE_STATUS_FORCING_MISSING: "流場缺值：本小時濕節點出現未登錄缺值，該批粒子在此停止。",
    NATIVE_STATUS_NUMERICAL_FAILURE: "數值失敗：粒子位置出現非有限值。",
}


class BackwardDriverError(RuntimeError):
    """WP-5 驅動層的一般性錯誤：受控載入失敗、鏡像迴圈遇到未處理情況等。"""


# ── 受控載入 BayTrace 原始模組（風格比照 baytrace_native_cpu.load_original_cpu）──

# 依 import 相依關係排序：前面的模組不依賴後面的模組。tracker.py 在模組
# 層級 import 了 particle_arrays／calendar／geometry／coordinate／grid／
# physics／config；grid.py import geometry／vertical；batch_interp.py
# import geometry／vertical／random；config.py import calendar／coordinate。
_BAYTRACE_MODULE_ORDER: tuple[str, ...] = (
    "geometry",
    "vertical",
    "random",
    "coordinate",
    "calendar",
    "physics",
    "particle_arrays",
    "config",
    "batch_interp",
    "grid",
    "tracker",
)

_LOAD_LOCK = threading.RLock()


def _sha256(path: Path) -> str:
    """以串流方式計算檔案 SHA-256，避免把整份原始碼一次讀進記憶體。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_baytrace_package_dir(source_root: Path) -> Path:
    """解析明確的 BayTrace package 目錄，不從其他位置猜測來源。"""

    package_dir = source_root if source_root.name == "baytrace" else source_root / "baytrace"
    package_dir = package_dir.resolve(strict=True)
    if not package_dir.is_dir() or package_dir.name != "baytrace":
        raise BackwardDriverError("source_root 必須包含 baytrace package 目錄")
    if not (package_dir / "__init__.py").is_file():
        raise BackwardDriverError("BayTrace package 缺少 __init__.py")
    return package_dir


@dataclass(frozen=True, slots=True)
class BayTraceModuleHandles:
    """受控載入後保留的原始物件與其來源證據。

    只暴露本模組實際會呼叫的四個 njit 函式與兩個類別；其餘已載入的原始
    模組（例如 ``physics``、``config``）只是滿足 import 相依，不對外
    暴露，避免呼叫端誤用未經本模組驗證過用途的原始碼路徑。
    """

    batch_move_rk4: Callable[..., None]
    batch_check_boundaries: Callable[..., None]
    batch_clamp_z: Callable[..., None]
    batch_interpolate_velocity: Callable[..., None]
    batch_interpolate_diffusivity: Callable[..., None]
    tracker_cls: type
    grid_cls: type
    particle_arrays_cls: type
    source_root: Path
    package_dir: Path
    dependency_sha256: Mapping[str, str]


def load_baytrace_modules(source_root: str | Path) -> BayTraceModuleHandles:
    """受控載入 WP-5 需要的 BayTrace 原始模組，記錄來源路徑與 SHA-256。

    ``source_root`` 必須明確指向含 ``baytrace/`` 的 artifact 根目錄（或
    直接指向該 package 目錄），不從工作目錄、``sys.path`` 或環境變數猜
    來源；呼叫端（例如測試或正式 CLI）自行從 ``LBT_BAYTRACE_ROOT`` 一類
    環境變數解析出這個路徑再傳入，本函式本身不讀取任何環境變數。

    載入方式：依 ``_BAYTRACE_MODULE_ORDER`` 的拓樸序，對每個檔案用
    ``importlib.util.spec_from_file_location`` 建立模組並手動
    ``exec_module``，同時登記到 ``sys.modules['baytrace.<name>']`` 與一個
    不執行 ``__init__.py`` 的 ``baytrace`` package sentinel（技巧與既有
    ``baytrace_native_cpu.load_original_cpu`` 相同，讓兩者可以在同一個
    行程內共存）。若 ``sys.modules`` 已存在來源不同的同名模組，直接拒絕，
    避免同名 package／模組被冒名替換。

    Returns:
        ``BayTraceModuleHandles``。

    Raises:
        BackwardDriverError: 來源目錄不存在、缺少必要檔案，或
            ``sys.modules`` 已被不同來源的 ``baytrace`` 佔用。
    """

    source_root = Path(source_root).expanduser().resolve(strict=True)
    package_dir = _resolve_baytrace_package_dir(source_root)

    dependency_paths = {name: package_dir / f"{name}.py" for name in _BAYTRACE_MODULE_ORDER}
    for name, path in dependency_paths.items():
        if path.is_symlink() or not path.is_file():
            raise BackwardDriverError(f"缺少或不允許的 BayTrace 原始檔案: {name}.py")
    dependency_sha256 = {name: _sha256(path) for name, path in dependency_paths.items()}

    with _LOAD_LOCK:
        package = sys.modules.get("baytrace")
        if package is not None:
            if getattr(package, "__lbt_controlled_source__", None) != str(package_dir):
                raise BackwardDriverError("sys.modules 已存在非本來源的 baytrace package")
        else:
            package = types.ModuleType("baytrace")
            package.__file__ = str(package_dir / "__init__.py")
            package.__package__ = "baytrace"
            package.__path__ = [str(package_dir)]
            package.__lbt_controlled_source__ = str(package_dir)
            # 既有 baytrace_native_cpu.load_original_cpu 用同一個 sentinel
            # 技巧，但只載入四個模組（_MODULE_NAMES），且會檢查
            # ``package.__lbt_dependency_sha256__`` 是否與它自己重算的四項
            # SHA-256 完全相等，才允許沿用既有 sentinel。若這裡建立
            # sentinel 時不寫入這個屬性，換成先呼叫本函式、再於同一行程
            # 呼叫 ``load_original_cpu`` 時，那邊的相等性檢查會因為
            # 「屬性不存在（None）」而永遠失敗，即使兩邊讀的是同一份原始
            # 檔案。這裡補上限定四個共同模組的子集雜湊，內容與
            # ``load_original_cpu`` 自己算出來的完全相同（同一份檔案、同一種
            # SHA-256 演算法），讓兩個獨立撰寫的受控載入器無論先後呼叫順序
            # 為何都能正確共用同一個 sentinel，不必互相依賴或匯入對方。
            package.__lbt_dependency_sha256__ = {
                name: dependency_sha256[name]
                for name in ("geometry", "vertical", "random", "batch_interp")
            }
            sys.modules["baytrace"] = package

        loaded: dict[str, types.ModuleType] = {}
        for name in _BAYTRACE_MODULE_ORDER:
            full_name = f"baytrace.{name}"
            module = sys.modules.get(full_name)
            if module is not None:
                existing_source = getattr(module, "__lbt_controlled_source__", None)
                if existing_source != str(dependency_paths[name]):
                    raise BackwardDriverError(f"sys.modules 已存在非本來源的 {full_name}")
                loaded[name] = module
                continue
            spec = importlib.util.spec_from_file_location(full_name, dependency_paths[name])
            if spec is None or spec.loader is None:
                raise BackwardDriverError(f"無法建立 {full_name} 的明確 source loader")
            module = importlib.util.module_from_spec(spec)
            sys.modules[full_name] = module
            spec.loader.exec_module(module)
            module.__lbt_controlled_source__ = str(dependency_paths[name])
            setattr(package, name, module)
            loaded[name] = module

    batch_interp_module = loaded["batch_interp"]
    tracker_module = loaded["tracker"]
    grid_module = loaded["grid"]
    particle_arrays_module = loaded["particle_arrays"]

    for name in (
        "batch_move_rk4",
        "batch_check_boundaries",
        "batch_clamp_z",
        "batch_interpolate_velocity",
        "batch_interpolate_diffusivity",
    ):
        if not hasattr(batch_interp_module, name):
            raise BackwardDriverError(f"baytrace.batch_interp 缺少必要函式 {name}")
    for module, cls_name in ((tracker_module, "ParticleTracker"), (grid_module, "SchismGrid")):
        if not hasattr(module, cls_name):
            raise BackwardDriverError(f"{module.__name__} 缺少必要類別 {cls_name}")

    return BayTraceModuleHandles(
        batch_move_rk4=batch_interp_module.batch_move_rk4,
        batch_check_boundaries=batch_interp_module.batch_check_boundaries,
        batch_clamp_z=batch_interp_module.batch_clamp_z,
        batch_interpolate_velocity=batch_interp_module.batch_interpolate_velocity,
        batch_interpolate_diffusivity=batch_interp_module.batch_interpolate_diffusivity,
        tracker_cls=tracker_module.ParticleTracker,
        grid_cls=grid_module.SchismGrid,
        particle_arrays_cls=particle_arrays_module.ParticleArrays,
        source_root=source_root,
        package_dir=package_dir,
        dependency_sha256=dependency_sha256,
    )


# ── 網格拓樸：重用 SchismGrid 的邊結構建構方法，不重寫演算法 ────────────────


@dataclass(frozen=True, slots=True)
class NativeBackwardDriverGrid:
    """驅動層需要的完整網格拓樸：WP-1 的靜態網格 + BayTrace 邊結構。

    ``schism_grid`` 是一個真正的、未經修改的 ``SchismGrid`` 執行個體，只是
    屬性由 ``NativeGridArrays`` 填入（不讀檔），再呼叫該類別自己的
    ``_compute_element_areas``／``compute_dldxy``／``compute_node_to_element``／
    ``build_side_structure`` 四個方法建立 ``area``、``xctr``、``yctr``、
    ``dldxy``、``nne``、``indel``、``elside``、``isidenode``、``ic3``、
    ``snx``、``sny``；``isbnd``（開放／陸地邊界分段代碼）目前一律填 0，
    因為 ``batch_check_boundaries`` 只在 ``mod_part==1``（油污模式，本模組
    不使用）的分支內讀取 ``isbnd``——已對照 ``batch_interp.py`` 761-1053
    行逐行確認：passive 模式（``mod_part==0``）遇到邊界一律走「反射」，
    從不參照 ``isbnd``，因此填 0 不影響任何行為，也不需要真正的開放／
    陸地邊界資料。

    ``ic3_cross_check_ok`` 記錄 ``schism_grid.build_side_structure()`` 重新
    算出的 ``ic3`` 是否與 ``grid.ic3``（``NativeGridArrays``，源自
    ``NativeMesh.triangle_neighbors``）逐元素相等；建構時一併驗證，兩者
    使用同一種「局部邊 j 對應跨過與局部頂點 j 相對那條邊的鄰居元素」慣例
    （已對照 ``mesh.py`` 的 ``_build_triangle_neighbors`` 與
    ``grid.py`` 的 ``build_side_structure`` 逐行確認相同），理論上必須
    相等；若不相等會直接拒絕建構，而不是靜默選用其中一份，因為那代表
    兩個工作包對「哪個元素是哪個」的認定不一致，是必須先解決的資料契約
    問題，不能在這裡自行決定要相信誰。

    ``min_edge_length_m`` 是全網格三角形三邊中最短的一段，供
    ``estimate_substep_count`` 的 CFL 判斷使用；只在建構時計算一次。
    """

    grid: NativeGridArrays
    schism_grid: Any
    h0: float
    min_edge_length_m: float
    ic3_cross_check_ok: bool


def build_native_backward_driver_grid(
    grid: NativeGridArrays,
    handles: BayTraceModuleHandles,
    *,
    h0: float = 0.0,
) -> NativeBackwardDriverGrid:
    """把 WP-1 的 ``NativeGridArrays`` 轉成含完整邊結構的驅動層網格。

    Args:
        grid: WP-1 ``build_native_grid_arrays`` 的輸出。
        handles: ``load_baytrace_modules`` 的回傳值，取用其中的
            ``grid_cls``（即 ``SchismGrid``）。
        h0: 濕乾判定的最小水深門檻（m），沿用 BayTrace ``config.py`` 的
            初始值慣例（0.0，見既有 ``baytrace_native_cpu.py`` 的
            ``BAYTRACE_DEFAULT_H0`` 註解），呼叫端可依正式設定覆寫。

    Raises:
        BackwardDriverError: 重新建構的 ``ic3`` 與 ``grid.ic3`` 不一致。
    """

    nnode = int(grid.grid_x.shape[0])
    ne = int(grid.i34.shape[0])

    sg = handles.grid_cls()
    sg.np = nnode
    sg.ne = ne
    sg.x = np.ascontiguousarray(grid.grid_x, dtype=np.float64)
    sg.y = np.ascontiguousarray(grid.grid_y, dtype=np.float64)
    sg.dp = np.ascontiguousarray(grid.dp, dtype=np.float64)
    sg.i34 = np.ascontiguousarray(grid.i34, dtype=np.int64)
    sg.elnode = np.ascontiguousarray(grid.elnode, dtype=np.int64)
    sg.nvrt = int(grid.nvrt)
    sg.kbp = np.ascontiguousarray(grid.kbp, dtype=np.int64)
    sg.kbe = np.ascontiguousarray(grid.kbe, dtype=np.int64)
    sg.sigma_lcl = np.ascontiguousarray(grid.sigma_lcl, dtype=np.float64)

    # 呼叫原版未修改的網格前處理方法：面積／元素中心 → 形函數梯度
    # （Smagorinsky 需要）→ 節點-元素鄰接（Smagorinsky 反距離加權需要）
    # → 邊結構（batch_check_boundaries 需要）。
    sg._compute_element_areas()
    sg.compute_dldxy()
    sg.compute_node_to_element()
    sg.build_side_structure()
    sg.isbnd = np.zeros(nnode, dtype=np.int64)

    # SchismGrid.build_side_structure() 把 ic3 配置成 (max_sides, ne)；
    # 全三角形網格時 max_sides=3，比 NativeGridArrays.ic3 固定的 (4, ne)
    # 少一列（第 4 列固定 -1，供四邊形使用）。比較前先把兩者攤成同一個
    # (4, ne) 形狀，第 4 列不足的部分補 -1，而不是直接比較兩個不同 shape
    # 的陣列（那樣 ``np.array_equal`` 一律回傳 False，會誤判為不一致）。
    rebuilt_ic3_raw = np.ascontiguousarray(sg.ic3, dtype=np.int64)
    rebuilt_ic3 = np.full((4, ne), -1, dtype=np.int64)
    rebuilt_ic3[: rebuilt_ic3_raw.shape[0], :] = rebuilt_ic3_raw
    ic3_cross_check_ok = bool(np.array_equal(rebuilt_ic3, grid.ic3))
    if not ic3_cross_check_ok:
        raise BackwardDriverError(
            "SchismGrid.build_side_structure() 重新算出的 ic3 與 NativeGridArrays.ic3 "
            "不一致：兩個工作包對三角形鄰接的認定不同，這是必須先解決的資料契約問題，"
            "驅動層不會靜默選用其中一份"
        )
    # 邊結構建構完成後，一律採用與 batch_move_rk4／batch_interpolate_velocity
    # 相同來源（grid.ic3）的陣列，確保同一次呼叫全程只有一份 ic3。
    sg.ic3 = np.ascontiguousarray(grid.ic3, dtype=np.int64)
    sg.elside = np.ascontiguousarray(sg.elside, dtype=np.int64)
    sg.isidenode = np.ascontiguousarray(sg.isidenode, dtype=np.int64)
    sg.snx = np.ascontiguousarray(sg.snx, dtype=np.float64)
    sg.sny = np.ascontiguousarray(sg.sny, dtype=np.float64)
    sg.xctr = np.ascontiguousarray(sg.xctr, dtype=np.float64)
    sg.yctr = np.ascontiguousarray(sg.yctr, dtype=np.float64)
    sg.area = np.ascontiguousarray(sg.area, dtype=np.float64)

    n1 = grid.elnode[:, 0]
    n2 = grid.elnode[:, 1]
    n3 = grid.elnode[:, 2]
    edge_lengths = np.concatenate(
        [
            np.hypot(grid.grid_x[n1] - grid.grid_x[n2], grid.grid_y[n1] - grid.grid_y[n2]),
            np.hypot(grid.grid_x[n2] - grid.grid_x[n3], grid.grid_y[n2] - grid.grid_y[n3]),
            np.hypot(grid.grid_x[n3] - grid.grid_x[n1], grid.grid_y[n3] - grid.grid_y[n1]),
        ]
    )
    min_edge_length_m = float(np.min(edge_lengths))
    if not np.isfinite(min_edge_length_m) or min_edge_length_m <= 0.0:
        raise BackwardDriverError("網格三角形邊長計算出非正值，無法用於 CFL 估計")

    return NativeBackwardDriverGrid(
        grid=grid,
        schism_grid=sg,
        h0=float(h0),
        min_edge_length_m=min_edge_length_m,
        ic3_cross_check_ok=ic3_cross_check_ok,
    )


# ── 未綁定方法呼叫：重用 ParticleTracker 的兩個純 NumPy 輔助方法 ────────────


class _TrackerMethodShim:
    """呼叫 ``ParticleTracker._compute_smagorinsky_hf2``／``_compute_dry_flags``
    這兩個未修改方法所需的最小鴨子型別容器。

    這兩個方法都只讀寫下列屬性（已逐行核對 ``tracker.py`` 原始碼），本
    類別因此只提供這些屬性，不建立完整 ``ParticleTracker``（會需要設定檔、
    netCDF I/O 等大量無關相依）。本類別本身不包含任何物理或數值邏輯。
    """

    def __init__(self, schism_grid: Any, *, h0: float, horcon: float) -> None:
        self.params: dict[str, float] = {"h0": float(h0), "horcon": float(horcon)}
        self.grid = schism_grid
        self.u_field2: np.ndarray | None = None
        self.v_field2: np.ndarray | None = None
        self.w_field2: np.ndarray | None = None
        self.eta_field1: np.ndarray | None = None
        self.hf2_field: np.ndarray | None = None
        self.idry_e: np.ndarray | None = None
        self.idry: np.ndarray | None = None
        # 供 ``find_initial_elements`` 使用；schism_grid 的這些欄位已在
        # ``build_native_backward_driver_grid`` 統一轉成 int64，這裡只是
        # 沿用同一份陣列的別名，不重新複製。
        self._grid_i34_i64 = schism_grid.i34
        self._grid_elnode_i64 = schism_grid.elnode
        self._grid_kbp_i64 = schism_grid.kbp
        self._grid_kbe_i64 = schism_grid.kbe
        self.windx_field2: np.ndarray | None = None
        self.windy_field2: np.ndarray | None = None
        self.pa: Any = None

    def _report_element_search_failure(self, px: float, py: float) -> None:
        """吞掉原版 ``find_initial_elements`` 找不到元素時的診斷列印。

        原版方法只用它印出除錯資訊，不影響任何狀態；本 shim 不需要那些
        列印雜訊，但仍保留一個可呼叫的同名方法，避免 ``AttributeError``。
        """

        return None


def compute_wet_dry_flags(
    driver_grid: NativeBackwardDriverGrid,
    handles: BayTraceModuleHandles,
    *,
    eta_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """重用 ``ParticleTracker._compute_dry_flags`` 計算節點與元素乾濕旗標。

    Args:
        eta_m: ``(nnode,)`` 該小時的節點海面高程，即 BayTrace 原版的
            ``eta_field1``（乾濕判定只用這一個時刻的海面，不混合
            ``eta_field2``，見模組 docstring 對 ``eta_field1`` 角色的說明）。

    Returns:
        ``(node_is_dry, elem_is_dry)``：分別是 ``(nnode,)``／``(ne,)`` 的
        布林陣列。``node_is_dry`` 是「重新檢查後」的節點乾濕（只要周邊有
        任一濕元素就算濕，對應原版 ``self.idry``，語意是「乾到連一個相鄰
        元素都不濕」）；``elem_is_dry`` 對應原版 ``self.idry_e``，直接是
        ``batch_check_boundaries`` 需要的元素級旗標。
    """

    shim = _TrackerMethodShim(driver_grid.schism_grid, h0=driver_grid.h0, horcon=0.2)
    shim.eta_field1 = np.ascontiguousarray(eta_m, dtype=np.float64)
    handles.tracker_cls._compute_dry_flags(shim)
    node_is_dry = shim.idry.astype(bool)
    elem_is_dry = shim.idry_e.astype(bool)
    return node_is_dry, elem_is_dry


def compute_smagorinsky_kh_field(
    driver_grid: NativeBackwardDriverGrid,
    handles: BayTraceModuleHandles,
    *,
    u_current_mps: np.ndarray,
    v_current_mps: np.ndarray,
    horcon: float = 0.2,
) -> np.ndarray:
    """重用 ``ParticleTracker._compute_smagorinsky_hf2`` 計算水平擴散係數場。

    刻意接收「純海流」速度（不含 Stokes、不含沉降），對應工作書式(10)
    Smagorinsky 公式的物理原意：以**已解析流場的應變率**估計次網格紊流
    黏性／擴散，Stokes 漂流的空間梯度（隨波高、波向變化）與沉降速度
    （空間均勻、梯度恆為零）都不是這個公式想估計的對象；若改用「合成後
    的總速度場」計算應變率，會讓 Stokes 造成的虛假水平剪切污染擴散係數
    估計，這不是原版 ``mod_part=1`` 使用這個方法時的用法（該路徑的
    Stokes 只加在逐粒子的 ``pa.u``／``pa.v``，從未進入
    ``u_field2``／``v_field2``，因此原版的 Smagorinsky 本來就只看純海流）。
    本模組刻意維持同一慣例，是需要在報告中特別說明的設計決策。

    Args:
        u_current_mps, v_current_mps: ``(nvrt, nnode)`` 純海流速度，即
            對應這一小時 field2（原版程式碼固定讀 ``self.u_field2``／
            ``self.v_field2``，見 ``tracker.py`` 1410-1411 行；本模組沿用
            同一時間層選擇，不代表 field2 比 field1 更「正確」，只是原版
            的既定用法）。
        horcon: Smagorinsky 係數，BayTrace 預設 0.2（``params.get('horcon',
            0.2)``）。

    Returns:
        ``(nvrt, nnode)`` 的水平擴散係數場（``hf2_field``），單位 m²/s。
    """

    shim = _TrackerMethodShim(driver_grid.schism_grid, h0=driver_grid.h0, horcon=horcon)
    shim.u_field2 = np.ascontiguousarray(u_current_mps, dtype=np.float64)
    shim.v_field2 = np.ascontiguousarray(v_current_mps, dtype=np.float64)
    handles.tracker_cls._compute_smagorinsky_hf2(shim)
    return shim.hf2_field


# ── 垂向 z：與 compute_z_levels_lsc 相同公式的向量化版本（僅供場合成用）────


def _vectorized_z_levels(
    sigma_lcl: np.ndarray, kbp: np.ndarray, dp: np.ndarray, eta: np.ndarray
) -> np.ndarray:
    """向量化計算 ``(nvrt, nnode)`` 的節點垂向 z（公尺），公式與
    ``vertical.compute_z_levels_lsc`` 完全相同：
    ``z = eta + sigma_lcl[k] * (dp + eta)``，``k < kbp`` 的層 clamp 成
    ``kbp`` 當層的值。這不是另一個「kernel」——它只是同一條純數學公式
    對所有節點一次算完，而不是逐節點呼叫 njit 函式；用來一次建立整個
    節點網格的 z（Stokes 需要），不是取代任何粒子層級的內插。
    """

    nvrt, nnode = sigma_lcl.shape
    layers = np.arange(nvrt)[:, None]
    z_at_kbp = eta[None, :] + sigma_lcl[kbp, np.arange(nnode)][None, :] * (dp[None, :] + eta[None, :])
    z_full = eta[None, :] + sigma_lcl * (dp[None, :] + eta[None, :])
    below = layers < kbp[None, :]
    return np.where(below, z_at_kbp, z_full)


# ── 波浪場輸入（WP-5 的介面缺口：無現成 NWW3→節點橋接，見模組 docstring）────


@dataclass(frozen=True, slots=True)
class WaveFieldAtNodes:
    """已內插到追蹤網格節點的 NWW3 波浪參數，供 Stokes 合成使用。

    所有欄位皆為 ``(nnode,)``；欄位語意與 ``stokes_batch.batch_finite_depth_stokes``
    的純量輸入完全一致（見該模組 docstring），只是這裡是「每個節點」而非
    「每個粒子」。``qc_flags`` 使用 int，非零視為不可信。
    """

    significant_wave_height_m: np.ndarray
    peak_frequency_hz: np.ndarray
    direction_raw_deg: np.ndarray
    valid_mask_wave: np.ndarray
    qc_flags: np.ndarray


# ── 速度場合成：事實 1 的具體實作 ────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ComposedFieldPair:
    """已合成「海流＋Stokes＋沉降」、且已套用乾濕與海床以下歸零的總速度場。

    欄位 shape 與 ``HourlyFieldPair`` 相同，可直接交給
    ``batch_move_rk4``／``batch_interpolate_velocity``。``node_is_dry``、
    ``elem_is_dry`` 一併回傳供呼叫端做擱淺判定，不必重算一次。
    """

    u_field1: np.ndarray
    v_field1: np.ndarray
    w_field1: np.ndarray
    eta_field1: np.ndarray
    u_field2: np.ndarray
    v_field2: np.ndarray
    w_field2: np.ndarray
    node_is_dry: np.ndarray
    elem_is_dry: np.ndarray
    stokes_ok_fraction: float


@dataclass(frozen=True, slots=True)
class ComposedCurrentAndStokesFields:
    """``compose_total_velocity_fields`` 拆成兩階段後的「前半段」中間結果。

    背景（任務 3 效能優化）
    --------------------------
    ``compose_total_velocity_fields`` 原本一次做完「乾濕判定→歸零→加
    Stokes→加沉降→再歸零」五個步驟；其中前三步（乾濕判定、對整個
    ``(nvrt, nnode)`` 陣列做 Stokes 全網格計算、歸零）的成本只跟**網格規模**
    有關，與這一批粒子的沉降速度完全無關。但 WP-4 的分批策略
    （``group_indices_by_settling_velocity_and_arrival_time``）刻意把「同一
    到達時刻、不同沉降速度」的粒子拆成不同批次分別呼叫
    ``run_backward_batch``——這代表同一個流場小時、同一個區域，過去會因為
    有 N 種沉降速度就重算 N 次昂貴的 Stokes 全網格計算，而其實只有最後一步
    「把常數沉降速度加進 ``w_field``」才真的需要每個沉降批次各做一次。

    本類別就是拆分後「與沉降速度無關」的那一半：已經套用乾濕與海床以下
    歸零、已經加總 Stokes 貢獻的 ``u/v`` 場，以及**尚未加沉降**的
    ``w_field``。呼叫端應該以「這一小時的兩個流場時刻（``field_pair.
    time_before_utc_ns``／``time_after_utc_ns``）」為快取鍵，讓同一到達
    時刻、同一區域的不同沉降批次共用同一份 ``ComposedCurrentAndStokesFields``
    （見 ``run_backward_batch`` 的 ``composed_base_cache`` 參數），只在
    ``apply_settling_velocity``（便宜：一次純量加法＋既有遮罩）這一步才
    依各自的沉降速度分岔。

    ``effective_mask`` 是 ``(nvrt, nnode)`` 布林陣列，``True`` 代表「濕節點
    且在該節點的有效垂向層（``k>=kbp``）」，沉降速度只應該加在這個遮罩內
    ——直接暴露出來，讓 ``apply_settling_velocity`` 不必重新計算一次。

    ``stokes_points_evaluated``／``stokes_points_possible``（WP-9 效能診斷）
    ------------------------------------------------------------------------
    見 ``compose_current_and_stokes_fields`` 的 ``stokes_depth_threshold_ratio``
    參數：啟用「Stokes 深度截斷」優化時，這兩個欄位分別是「真正呼叫過
    ``batch_finite_depth_stokes`` 求解色散關係與垂向 Stokes 分佈的
    (層,節點) 點數」與「``effective_mask`` 內的 (層,節點) 總點數」；比值
    即為深度截斷後真正需要重算 Stokes 的點數比例，供效能報告量化優化效果
    （見模組文件與最終報告）。未提供波浪場（不需要算 Stokes）時，兩者
    一律是 0；提供波浪場但未啟用截斷（``stokes_depth_threshold_ratio=
    None``）時，兩者相等且等於 ``effective_mask`` 內的點數（代表「全部
    點都照舊計算，沒有跳過任何一個」，不是「未使用這組計數」）。
    """

    u_field1: np.ndarray
    v_field1: np.ndarray
    w_field1_no_settling: np.ndarray
    eta_field1: np.ndarray
    u_field2: np.ndarray
    v_field2: np.ndarray
    w_field2_no_settling: np.ndarray
    node_is_dry: np.ndarray
    elem_is_dry: np.ndarray
    effective_mask: np.ndarray
    stokes_ok_fraction: float
    stokes_points_evaluated: int = 0
    stokes_points_possible: int = 0


def _zero_dry_and_below_kbp(
    u_field: np.ndarray, v_field: np.ndarray, w_field: np.ndarray, *, kbp: np.ndarray, node_is_dry: np.ndarray
) -> None:
    """原地把乾節點與海床以下層的速度歸零，公式對應
    ``ParticleTracker._zero_dry_velocities``／``_zero_below_kbp``
    （``tracker.py`` 2333-2389 行），純 NumPy 布林遮罩運算，不是 njit
    kernel；獨立在此重寫（而非透過 shim 呼叫原版方法）是因為原版方法會
    同時處理 ``diff_field``、``include_field1`` 等本模組用不到的分支，
    這裡只保留「乾節點歸零」與「``k<kbp`` 歸零」兩條與本模組相關的規則，
    公式本身完全比照，行為由 ``tests/test_native_backward_driver.py`` 的
    合成測試交叉驗證。
    """

    nvrt = u_field.shape[0]
    u_field[:, node_is_dry] = 0.0
    v_field[:, node_is_dry] = 0.0
    w_field[:, node_is_dry] = 0.0
    layers = np.arange(nvrt)[:, None]
    below_wet = (layers < kbp[None, :]) & ~node_is_dry[None, :]
    u_field[below_wet] = 0.0
    v_field[below_wet] = 0.0
    w_field[below_wet] = 0.0


def compose_current_and_stokes_fields(
    driver_grid: NativeBackwardDriverGrid,
    handles: BayTraceModuleHandles,
    field_pair: HourlyFieldPair,
    *,
    wave_field1: WaveFieldAtNodes | None = None,
    wave_field2: WaveFieldAtNodes | None = None,
    gravity_mps2: float = GRAVITY_MPS2,
    surface_tolerance_m: float = 5.0e-6,
    stokes_depth_threshold_ratio: float | None = None,
    stokes_pool: Any | None = None,
    stokes_worker_count: int = 1,
) -> ComposedCurrentAndStokesFields:
    """``compose_total_velocity_fields`` 拆成兩階段後，計算「與沉降速度無關」
    的那一半（乾濕判定＋歸零＋Stokes 全網格計算），供多個沉降速度批次共用。

    這是任務 3 效能優化的核心：這個函式的成本只跟網格規模
    （``nvrt × nnode``）與是否提供波浪場有關，與呼叫端最終要用哪個沉降速度
    完全無關，因此可以用 ``(field_pair.time_before_utc_ns,
    field_pair.time_after_utc_ns)`` 當快取鍵，讓同一到達時刻、同一區域、
    不同沉降速度的批次（見 WP-4
    ``group_indices_by_settling_velocity_and_arrival_time``）在同一個流場
    小時只呼叫一次，而不是「有幾種沉降速度就重算幾次」；``run_backward_batch``
    的 ``composed_base_cache`` 參數就是接這個快取用途。

    步驟與 ``compose_total_velocity_fields`` 原本的步驟 1-3 完全相同（乾濕
    判定、歸零、視情況加 Stokes），只是把「加沉降、再歸零一次」（原步驟
    4、5）切出去，交給 ``apply_settling_velocity`` 處理，見該函式與
    ``ComposedCurrentAndStokesFields`` 的 docstring。

    Args:
        與 ``compose_total_velocity_fields`` 相同，唯獨不接受
        ``settling_velocity_mps``。
        stokes_depth_threshold_ratio: WP-9 新增的「Stokes 深度截斷」優化
            開關與門檻。``None``（預設）：完全不啟用，行為與此參數新增前
            逐位元相同（``compose_total_velocity_fields`` 原本 1-3 步）—
            —這是 ``run_backward_batch`` 的預設值，確保既有測試與呼叫端
            不受影響。給定 ``(0, 1)`` 開區間的值時，啟用截斷：

            物理依據——有限水深 Stokes 表面漂流的垂向結構是
            ``exp(2 k z)``（``k`` 為波數、``z`` 為相對海面的深度，負值），
            等價於工作書式(7)常寫成的 ``exp(4π z / L)``（``L=2π/k`` 為
            波長）；這是隨深度單調遞減的指數衰減，與粒子的沉降速度、到達
            時刻、測站完全無關，只跟這一小時、這一節點的波浪參數與水深
            有關。當 ``exp(2kz)`` 相對表面值（``z=0``）已經小於
            ``stokes_depth_threshold_ratio`` 時，即使不計算，也已經比
            NWW3 輸入的 ``Hs``／``fp``／波向本身的觀測與內插不確定度小了
            幾個數量級，直接視為 0 不會造成有意義的科學誤差；門檻越小，
            截斷越保守（保留越多層），``1e-3``（千分之一）是一個合理、
            保守的預設，對應「衰減到千分之一以下視為可忽略」，可由呼叫端
            依實際近岸波浪不確定度調整。

            實作：**先**對每個節點在 ``z=0`` 呼叫一次
            ``batch_finite_depth_stokes``（``nnode`` 個點，而不是
            ``nvrt*nnode`` 個點）求出波數 ``k``——色散關係求根本身完全不
            依賴 ``z``（見 ``stokes_batch.py`` ``_solve_wave_number_embedded``
            docstring 與原始碼，同一節點在不同垂向層的 ``Hs``／``fp``／
            水深皆相同，因此 ``k`` 對同一節點的每一層都相同），這一步的
            成本只有原本「對全部 ``nvrt`` 層都各自重新求解一次色散關係」
            的 ``1/nvrt``；**再**用求出的 ``k`` 算出每個節點的截斷深度
            （``z_cutoff = ln(threshold) / (2k)``），只對「淺於截斷深度」
            且在 ``effective_mask`` 內的 (層,節點) 組合，第二次呼叫
            ``batch_finite_depth_stokes``（陣列縮小成只含這些點，用布林
            fancy indexing 選取、計算後再散回原陣列，深於截斷深度的位置
            維持 0，效果等同「衰減後貢獻視為 0」）。第二次呼叫仍然會對
            保留下來的每個點重新求解一次色散關係（``batch_finite_depth_stokes``
            把求解 ``k`` 與套用垂向結構耦合在同一個 njit 函式內，本模組
            依專案規定不修改該檔案，因此無法直接重用第一次呼叫已經解出的
            ``k``），但因為保留點數遠少於 ``nvrt*nnode``（近岸淺水站點
            可能仍需保留全部或大部分層，外海深水站點通常只需保留表層附近
            少數幾層），總求解次數仍遠低於未截斷時的 ``nvrt*nnode`` 次，
            見 ``stokes_points_evaluated``／``stokes_points_possible`` 與
            最終報告的量化結果。任何節點若第一次呼叫本身狀態非
            ``STOKES_BATCH_STATUS_OK``（例如波浪缺值、QC 不通過、色散求根
            失敗），一律**不截斷**（該節點全部層都納入第二次呼叫，交由
            正常的逐層狀態碼判定），避免在「不確定波數」的節點錯誤地跳過
            本來就該正常計算（或正常回報缺值）的層——寧可多算，不可少算。
        stokes_pool: WP-13 任務 2 新增。``None``（預設）：Stokes 求解在呼叫
            端自己的程序內單次呼叫，行為與此參數新增前逐位元相同。提供一個
            ``concurrent.futures.ProcessPoolExecutor`` 時，搭配
            ``stokes_worker_count`` 把 Stokes 求解切成多段、分派給這個行程
            池平行計算（見模組層級「WP-13 任務 2」說明：Stokes 逐點獨立、
            沒有跨點歸約，切幾段、哪個行程算哪一段都不影響結果，保證逐位元
            相同）。呼叫端負責建立與關閉這個 pool（通常只在尾端剩餘工作
            單元數少於可用核心數時才建立，見 ``hour_major_provider.py`` 的
            整合點），本函式不擁有它的生命週期。
        stokes_worker_count: 搭配 ``stokes_pool`` 使用，Stokes 點集合要切成
            幾段；``<= 1`` 時等同不啟用（即使提供了 ``stokes_pool``）。

    Raises:
        BackwardDriverError: ``wave_field1``／``wave_field2`` 只提供其中
            一個（必須同時提供或同時不提供），或
            ``stokes_depth_threshold_ratio`` 不在 ``(0, 1)`` 開區間內。
    """

    if (wave_field1 is None) != (wave_field2 is None):
        raise BackwardDriverError("wave_field1 與 wave_field2 必須同時提供或同時不提供")
    if stokes_depth_threshold_ratio is not None and not (0.0 < stokes_depth_threshold_ratio < 1.0):
        raise BackwardDriverError("stokes_depth_threshold_ratio 必須落在 (0, 1) 開區間內")

    grid = driver_grid.grid
    node_is_dry, elem_is_dry = compute_wet_dry_flags(driver_grid, handles, eta_m=field_pair.eta_field1)
    kbp = grid.kbp
    dp = grid.dp

    u1 = np.array(field_pair.u_field1, dtype=np.float64, copy=True)
    v1 = np.array(field_pair.v_field1, dtype=np.float64, copy=True)
    w1 = np.array(field_pair.w_field1, dtype=np.float64, copy=True)
    u2 = np.array(field_pair.u_field2, dtype=np.float64, copy=True)
    v2 = np.array(field_pair.v_field2, dtype=np.float64, copy=True)
    w2 = np.array(field_pair.w_field2, dtype=np.float64, copy=True)

    _zero_dry_and_below_kbp(u1, v1, w1, kbp=kbp, node_is_dry=node_is_dry)
    _zero_dry_and_below_kbp(u2, v2, w2, kbp=kbp, node_is_dry=node_is_dry)

    nvrt, nnode = u1.shape
    effective_mask = (np.arange(nvrt)[:, None] >= kbp[None, :]) & ~node_is_dry[None, :]

    stokes_ok_count = 0
    stokes_total_count = 0
    stokes_points_evaluated = 0
    stokes_points_possible = 0
    if wave_field1 is not None and wave_field2 is not None:
        z1 = (
            _vectorized_z_levels(grid.sigma_lcl, kbp, dp, field_pair.eta_field1)
            - field_pair.eta_field1[None, :]
        )
        z2 = (
            _vectorized_z_levels(grid.sigma_lcl, kbp, dp, field_pair.eta_field2)
            - field_pair.eta_field2[None, :]
        )
        for z_rel, u_field, v_field, wave in (
            (z1, u1, v1, wave_field1),
            (z2, u2, v2, wave_field2),
        ):
            depth_grid = np.broadcast_to(dp[None, :], (nvrt, nnode))
            hs_grid = np.broadcast_to(wave.significant_wave_height_m[None, :], (nvrt, nnode))
            fp_grid = np.broadcast_to(wave.peak_frequency_hz[None, :], (nvrt, nnode))
            dir_grid = np.broadcast_to(wave.direction_raw_deg[None, :], (nvrt, nnode))
            valid_grid = np.broadcast_to(wave.valid_mask_wave[None, :], (nvrt, nnode))
            qc_grid = np.broadcast_to(wave.qc_flags[None, :], (nvrt, nnode))

            # 決定這一次呼叫實際要計算 Stokes 的 (層,節點) 遮罩：未啟用深度
            # 截斷時就是整個 effective_mask（與此優化新增前完全相同）；啟用
            # 時先以 nnode 個點的表層 (z=0) 呼叫求出每個節點的波數，據此縮小
            # 遮罩，詳見本函式 docstring「stokes_depth_threshold_ratio」一節。
            if stokes_depth_threshold_ratio is None:
                stokes_mask = effective_mask
            else:
                zero_z_at_nodes = np.zeros(nnode, dtype=np.float64)
                surface_u = np.empty(nnode, dtype=np.float64)
                surface_v = np.empty(nnode, dtype=np.float64)
                surface_k = np.empty(nnode, dtype=np.float64)
                surface_status = np.empty(nnode, dtype=np.int32)
                batch_finite_depth_stokes(
                    np.ascontiguousarray(wave.significant_wave_height_m, dtype=np.float64),
                    np.ascontiguousarray(wave.peak_frequency_hz, dtype=np.float64),
                    np.ascontiguousarray(wave.direction_raw_deg, dtype=np.float64),
                    zero_z_at_nodes,
                    np.ascontiguousarray(dp, dtype=np.float64),
                    np.ascontiguousarray(wave.valid_mask_wave, dtype=bool),
                    np.ascontiguousarray(wave.qc_flags, dtype=np.int64),
                    surface_u,
                    surface_v,
                    surface_k,
                    surface_status,
                    float(gravity_mps2),
                    float(surface_tolerance_m),
                )
                k_known = (
                    (surface_status == STOKES_BATCH_STATUS_OK)
                    & np.isfinite(surface_k)
                    & (surface_k > 0.0)
                )
                # 波數不確定的節點一律不截斷（cutoff 設成 -inf，等同「全部
                # 深度都視為顯著」），寧可多算也不可漏算；波數已知的節點才
                # 依 exp(2kz)>=threshold 反推出截斷深度 z_cutoff（負值）。
                cutoff_z_node = np.where(
                    k_known,
                    np.log(stokes_depth_threshold_ratio) / (2.0 * np.where(k_known, surface_k, 1.0)),
                    -np.inf,
                )
                cutoff_z_grid = np.broadcast_to(cutoff_z_node[None, :], (nvrt, nnode))
                stokes_mask = effective_mask & (z_rel >= cutoff_z_grid)

            n_kept = int(np.count_nonzero(stokes_mask))
            stokes_points_evaluated += n_kept
            stokes_points_possible += int(np.count_nonzero(effective_mask))
            if n_kept > 0:
                stokes_u, stokes_v, wave_number, status = _evaluate_batch_finite_depth_stokes(
                    np.ascontiguousarray(hs_grid[stokes_mask], dtype=np.float64),
                    np.ascontiguousarray(fp_grid[stokes_mask], dtype=np.float64),
                    np.ascontiguousarray(dir_grid[stokes_mask], dtype=np.float64),
                    np.ascontiguousarray(z_rel[stokes_mask], dtype=np.float64),
                    np.ascontiguousarray(depth_grid[stokes_mask], dtype=np.float64),
                    np.ascontiguousarray(valid_grid[stokes_mask], dtype=bool),
                    np.ascontiguousarray(qc_grid[stokes_mask], dtype=np.int64),
                    gravity_mps2=float(gravity_mps2),
                    surface_tolerance_m=float(surface_tolerance_m),
                    stokes_pool=stokes_pool,
                    stokes_worker_count=stokes_worker_count,
                )
                ok_kept = status == STOKES_BATCH_STATUS_OK
                ok = np.zeros((nvrt, nnode), dtype=bool)
                ok[stokes_mask] = ok_kept
                u_field[ok] += stokes_u[ok_kept]
                v_field[ok] += stokes_v[ok_kept]
                stokes_ok_count += int(np.count_nonzero(ok_kept))
            stokes_total_count += int(np.count_nonzero(effective_mask))

    # 這裡不加沉降，因此不需要在此再做一次歸零：u1/v1/u2/v2 已在上面歸零過，
    # 而 Stokes 只加在 effective_mask（已經排除乾點與海床以下）之內，不會
    # 弄髒無效層；w1/w2 本身也已經歸零過，尚未加沉降前不會再被弄髒。真正
    # 「加完沉降後再防禦性歸零一次」的動作留給 apply_settling_velocity。
    stokes_ok_fraction = (
        float(stokes_ok_count) / float(stokes_total_count) if stokes_total_count > 0 else float("nan")
    )

    return ComposedCurrentAndStokesFields(
        u_field1=np.ascontiguousarray(u1),
        v_field1=np.ascontiguousarray(v1),
        w_field1_no_settling=np.ascontiguousarray(w1),
        eta_field1=np.ascontiguousarray(field_pair.eta_field1, dtype=np.float64),
        u_field2=np.ascontiguousarray(u2),
        v_field2=np.ascontiguousarray(v2),
        w_field2_no_settling=np.ascontiguousarray(w2),
        node_is_dry=node_is_dry,
        elem_is_dry=elem_is_dry,
        effective_mask=effective_mask,
        stokes_ok_fraction=stokes_ok_fraction,
        stokes_points_evaluated=stokes_points_evaluated,
        stokes_points_possible=stokes_points_possible,
    )


def apply_settling_velocity(
    base: ComposedCurrentAndStokesFields,
    settling_velocity_mps: float,
    *,
    kbp: np.ndarray,
) -> ComposedFieldPair:
    """把常數沉降速度加進 ``compose_current_and_stokes_fields`` 的結果，
    完成 ``compose_total_velocity_fields`` 原本的步驟 4、5。

    這一步刻意做得非常便宜（只有陣列複製＋一次遮罩內純量加法＋一次防禦性
    歸零），因為它是「每個沉降批次都要各自做一次」的部分（見
    ``ComposedCurrentAndStokesFields`` docstring）；真正昂貴的乾濕判定與
    Stokes 全網格計算已經在 ``base`` 裡算好，這裡不重算。

    Args:
        base: ``compose_current_and_stokes_fields`` 的輸出，可以是跨多個
            沉降批次共用的同一個物件（本函式不修改 ``base`` 的任何陣列，
            都是先複製再修改，安全併發共用）。
        settling_velocity_mps: 這一批粒子共用的沉降速度（m/s），語意同
            ``compose_total_velocity_fields``。
        kbp: ``(nnode,)`` 每個節點第一個有效垂向層索引，用於防禦性歸零；
            呼叫端傳入 ``driver_grid.grid.kbp``。

    Raises:
        BackwardDriverError: ``settling_velocity_mps`` 非有限值。
    """

    settling = float(settling_velocity_mps)
    if not np.isfinite(settling):
        raise BackwardDriverError("settling_velocity_mps 必須是有限值")

    w1 = np.array(base.w_field1_no_settling, dtype=np.float64, copy=True)
    w2 = np.array(base.w_field2_no_settling, dtype=np.float64, copy=True)
    w1[base.effective_mask] += settling
    w2[base.effective_mask] += settling

    u1 = np.array(base.u_field1, dtype=np.float64, copy=True)
    v1 = np.array(base.v_field1, dtype=np.float64, copy=True)
    u2 = np.array(base.u_field2, dtype=np.float64, copy=True)
    v2 = np.array(base.v_field2, dtype=np.float64, copy=True)
    # 防禦性第二道歸零（對應原本步驟 5）：理論上加沉降不會弄髒乾點或海床
    # 以下（effective_mask 已經排除這些位置），但仍保留這一步以維持與
    # compose_total_velocity_fields 拆分前完全相同的輸出，不因重構而改變
    # 任何數值行為。
    _zero_dry_and_below_kbp(u1, v1, w1, kbp=kbp, node_is_dry=base.node_is_dry)
    _zero_dry_and_below_kbp(u2, v2, w2, kbp=kbp, node_is_dry=base.node_is_dry)

    return ComposedFieldPair(
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


def compose_total_velocity_fields(
    driver_grid: NativeBackwardDriverGrid,
    handles: BayTraceModuleHandles,
    field_pair: HourlyFieldPair,
    *,
    settling_velocity_mps: float,
    wave_field1: WaveFieldAtNodes | None = None,
    wave_field2: WaveFieldAtNodes | None = None,
    gravity_mps2: float = GRAVITY_MPS2,
    surface_tolerance_m: float = 5.0e-6,
    stokes_depth_threshold_ratio: float | None = None,
) -> ComposedFieldPair:
    """把海流、Stokes 漂流（WP-3）、沉降速度合成為 kernel 用的總速度場。

    步驟（對應模組 docstring「事實 1」）：

    1. 以 ``eta_field1`` 計算節點／元素乾濕旗標（``compute_wet_dry_flags``，
       重用 ``ParticleTracker._compute_dry_flags``），field1／field2 共用
       同一份乾濕判定（比照原版 ``_zero_dry_velocities(include_field1=True)``
       的用法，見 ``tracker.py`` 2929-2931 行）。
    2. 複製（絕不原地修改）``field_pair`` 的 ``u/v/w_field1/2``，先套用
       乾濕與海床以下歸零，讓後續加總不會把 Stokes／沉降疊到不該有速度
       的乾點或無效層。
    3. 若提供 ``wave_field1``／``wave_field2``，在每個 (節點, 層) 呼叫
       WP-3 的 ``batch_finite_depth_stokes``（把網格點當「粒子」餵入：
       ``z`` 為該層相對海面的 z、``h`` 為節點水深），把回傳的水平 Stokes
       速度加進 ``u_field``／``v_field``；狀態非 OK（缺值、QC 不通過、
       水柱外、色散求解失敗）的位置不加（維持純海流），並回報
       ``stokes_ok_fraction`` 供 QC。深層不特別跳過：
       ``batch_finite_depth_stokes`` 的指數垂向結構在深層只會安全下溢為
       極小值，不會變慢或不穩定（見該模組 docstring「數值穩定性」一節），
       全部 ``nvrt*nnode`` 個點一次算完即可，效能上不需要跳過優化。
    4. 沉降速度是常數，只加到「有效層」（濕節點、``k>=kbp``）的
       ``w_field``；乾點與海床以下已在步驟 2 歸零，這裡直接對整個陣列
       加常數再套用同一個遮罩即可達成「只加有效層」的效果。
    5. 再套用一次乾濕／海床以下歸零，確保 Stokes／沉降沒有意外污染到
       無效層（理論上不會，因為步驟 3/4 已經跳過或已遮罩，這是防禦性
       的第二道保險，成本可忽略）。

    任務 3 效能優化後的實作備註
    ------------------------------
    本函式現在只是 ``compose_current_and_stokes_fields`` ＋
    ``apply_settling_velocity`` 的薄包裝，行為（含回傳值的每一個位元）與
    拆分前完全相同；拆成兩段是為了讓「同一到達時刻、同一區域、不同沉降
    速度」的多個批次可以共用前段的乾濕判定與 Stokes 全網格計算（見
    ``run_backward_batch`` 的 ``composed_base_cache`` 參數與模組 docstring
    「任務 3」一節的量化結果）。**只呼叫這個函式本身（不使用
    ``composed_base_cache``）不會有任何效能差異**，因為前段一樣只算一次；
    效能改善只發生在「重複呼叫、且共用同一個 base 物件」的情境。

    Args:
        settling_velocity_mps: 這一整批粒子共用的沉降速度（m/s，正值代表
            向上浮、負值代表下沉），來自 WP-4 ``BatchGroup`` 分組後的
            單一常數；呼叫端必須先以
            ``validate_settling_velocity_homogeneous_batches`` 確認同批
            粒子共用同一個值。
        stokes_depth_threshold_ratio: 直接透傳給
            ``compose_current_and_stokes_fields``，見該函式 docstring；
            預設 ``None`` 不啟用截斷，行為與此參數新增前完全相同。

    Raises:
        BackwardDriverError: ``wave_field1``／``wave_field2`` 只提供其中
            一個（必須同時提供或同時不提供），或 ``settling_velocity_mps``
            非有限值。
    """

    base = compose_current_and_stokes_fields(
        driver_grid,
        handles,
        field_pair,
        wave_field1=wave_field1,
        wave_field2=wave_field2,
        gravity_mps2=gravity_mps2,
        surface_tolerance_m=surface_tolerance_m,
        stokes_depth_threshold_ratio=stokes_depth_threshold_ratio,
    )
    return apply_settling_velocity(base, settling_velocity_mps, kbp=driver_grid.grid.kbp)


# ── 全域同步時鐘：CFL 決定 ndeltp，並保證 dtb <= dt_max_seconds ─────────────


def estimate_substep_count(
    driver_grid: NativeBackwardDriverGrid,
    u_field: np.ndarray,
    v_field: np.ndarray,
    *,
    dt_seconds: float = 3600.0,
    dt_max_seconds: float = 30.0,
    cfl_number: float = 0.5,
) -> int:
    """依 CFL 條件與 ``dt_max_seconds`` 上限，估計這一小時要切成幾個子步。

    這不是嚴謹的 RK4＋隨機擴散穩定性證明，只是工作書要求的「明確下限讓
    ``dtb<=30 秒``」搭配一個標準的平流 CFL 安全係數（預設
    ``cfl_number=0.5``，即子步位移不超過半個最小網格邊長）；真正決定
    ``ndeltp`` 的是兩者中較嚴格（``ndeltp`` 較大）的那一個：

    ``ndeltp = max(ceil(dt_seconds / dt_max_seconds),``
    ``              ceil(dt_seconds * max_speed / (cfl_number * min_edge_length)))``

    ``max_speed`` 取這一小時合成後總速度場（含 Stokes、沉降）在所有濕、
    有效層節點上的最大水平速度量值；若場中沒有任何有限速度（例如整個
    區域都乾），退化為只用 ``dt_max_seconds`` 上限。

    Returns:
        ``ndeltp``（正整數），保證 ``dt_seconds / ndeltp <= dt_max_seconds``。
    """

    if dt_seconds <= 0.0 or dt_max_seconds <= 0.0 or cfl_number <= 0.0:
        raise BackwardDriverError("dt_seconds／dt_max_seconds／cfl_number 必須是正值")

    speed = np.hypot(u_field, v_field)
    finite_speed = speed[np.isfinite(speed)]
    max_speed = float(np.max(finite_speed)) if finite_speed.size > 0 else 0.0

    ndeltp_from_dt_max = int(np.ceil(dt_seconds / dt_max_seconds))
    if max_speed <= 0.0:
        ndeltp_cfl = 1
    else:
        dtb_cfl = cfl_number * driver_grid.min_edge_length_m / max_speed
        ndeltp_cfl = int(np.ceil(dt_seconds / dtb_cfl))

    ndeltp = max(1, ndeltp_from_dt_max, ndeltp_cfl)
    dtb = dt_seconds / ndeltp
    if dtb > dt_max_seconds * (1.0 + 1.0e-9):
        raise BackwardDriverError("內部錯誤：估計出的 dtb 超過 dt_max_seconds 上限")
    return ndeltp


# ── 批次向量化隨機擴散：seed 設計與取捨（見模組 docstring／最終報告）───────


def derive_batch_diffusion_seed(seed_hex_values: Sequence[str], *, master_seed: int) -> int:
    """從一批粒子的 128-bit 種子（``seed_hex``）推導出這一批共用的擴散亂數種子。

    設計取捨（誠實揭露，不迴避）
    ------------------------------
    向量化與「同一顆粒子的亂數序列與批次組成無關」這兩個目標無法同時完全
    達成：本函式選擇**向量化優先**——同一批粒子共用一個
    ``numpy.random.Generator``（``PCG64DXSM``），每個子步對「整批」粒子
    一次呼叫 ``standard_normal(size=(n,2))``，而不是每顆粒子各自建立一個
    獨立的 ``Generator``（那正是先前效能量測指出的瓶頸來源之一）。

    代價是：同一顆粒子若被分到不同組成的批次（例如換一種分批策略、或
    批次內粒子順序不同），會從共用串流的不同位置取數，得到不同的擴散
    序列。**在本專案實際會用到的分批策略下**（
    ``group_indices_by_settling_velocity_and_arrival_time``：沉降速度與
    到達時刻都是每個情境固定的物理屬性），同一顆粒子每次重新執行同一個
    實驗設計時，必然落在組成相同、列序也相同（``BatchGroup.indices`` 是
    確定性排序）的同一批，因此**同一個實驗設計重跑具有完全的位元可重現
    性**（見驗收 7）；只有當比較「同一顆粒子在兩種不同分批方案下」的
    擴散序列時，兩者才會不同——這是刻意接受的取捨，不是缺陷。

    種子推導本身：對批次粒子的 ``seed_hex``（不依賴列序，因為排序後才
    雜湊）與 ``master_seed`` 一起做 SHA-256，取前 8 個位元組當成
    ``numpy.random.SeedSequence`` 的輸入，與既有 ``derive_member_seed``
    （``scenarios.py``）同樣使用 SHA-256、不使用 Python 內建 hash（會受
    ``PYTHONHASHSEED`` 影響）的原則一致。

    Args:
        seed_hex_values: 這一批全部粒子的 32 字元十六進位種子（不必先
            排序，本函式會自己排序以確保與列序無關）。
        master_seed: 與 ``derive_member_seed`` 相同角色的主種子。
    """

    if not seed_hex_values:
        raise BackwardDriverError("seed_hex_values 不可為空")
    payload = ",".join(sorted(seed_hex_values)) + f"|{int(master_seed)}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


# ── WP-9：計數器型（counter-based）擴散亂數 ──────────────────────────────────
#
# 背景（為什麼要新增這一套，取代 derive_batch_diffusion_seed 的預設用途）
# ------------------------------------------------------------------------
# ``derive_batch_diffusion_seed``＋共用 ``Generator`` 的方案，刻意選擇
# 「向量化優先」，代價是同一顆粒子的擴散序列與「這顆粒子被分到哪一個批次、
# 批次內排在第幾列」綁定（見該函式 docstring 的取捨說明）。WP-9 的小時主軸
# 排程刻意把「同一到達時刻、同一測站」的粒子，與「其他到達時刻、其他測站」
# 但沉降速度相同的粒子合併成同一次 kernel 呼叫（見 ``hour_major.py`` 模組
# 文件），批次的組成與列序因此不再是恆定不變的實驗設計參數，而是排程演算法
# 的實作細節（例如換一種排程策略、換一種 worker 數，同一顆粒子仍應得到完全
# 相同的擴散序列）。若繼續用共用 ``Generator`` 的方案，排程細節的變動會讓
# 擴散序列跟著變，這既違反「正確性不可妥協」，也讓「小時主軸與逐單元路徑
# 位元相同」這條驗收標準無法成立。
#
# 解法：**計數器型（counter-based）亂數**——不維護任何有狀態的 RNG 物件、
# 不依賴呼叫順序，而是把「這顆粒子的固定種子」與「這顆粒子自己的子步計數器
# （從牠被啟動那一刻起算，每套用一次擴散就加一，累加跨越小時邊界）」一起
# 雜湊，直接算出這一步要用的常態亂數。這樣一來，只要滿足兩個physically
# 保證成立的條件——(a) 同一顆粒子每次重新執行同一個實驗設計，都會在同一個
# UTC 走到同一個累積子步序號（因為 ``estimate_substep_count`` 算出的
# ``ndeltp`` 只跟「流場區域、這一個小時」有關，見該函式 docstring，與批次
# 組成、沉降速度、到達時刻本身都無關）；(b) 每顆粒子有自己獨一無二、與其他
# 粒子無關的種子——這顆粒子的擴散序列就與批次組成、列序、worker 數完全
# 無關，可以在合併／不合併、單執行緒／多執行緒之間位元相同地重現。
#
# 演算法：SplitMix64（Steele, Lea & Flood 2014 提出，Java
# ``SplittableRandom``／多個現代 PRNG 函式庫採用的 64-bit 狀態終結函式，
# 已發表且廣泛測試過雪崩效應與統計品質）的終結混合函式
# （``_splitmix64_finalize``），先把粒子的 128-bit 種子（拆成 ``seed_lo``／
# ``seed_hi`` 兩個 64-bit 半段）二次混合成一個 64-bit 「粒子專屬狀態種子」，
# 再對每個子步計數器 ``counter`` 算出
# ``state = seed_mixed + counter * GOLDEN_GAMMA``（``GOLDEN_GAMMA`` 是
# SplitMix64 標準的黃金比例增量常數，讓連續計數器的狀態在 64-bit 空間均勻
# 散開），對 ``state`` 與 ``state XOR SALT``（``SALT`` 只是另一個任意的
# 64-bit 常數，用來讓兩個輸出彼此獨立，不是安全性用途）分別再做一次終結
# 混合，得到兩個獨立的 64-bit 雜湊值，取其高 53 個位元轉成
# ``(0,1)``（開區間，見下方避免 ``log(0)`` 的處理）均勻亂數 ``u1``／``u2``，
# 最後以標準 Box–Muller 轉換
# （``z0=sqrt(-2 ln u1) cos(2π u2)``、``z1=sqrt(-2 ln u1) sin(2π u2)``）
# 得到兩個獨立標準常態亂數，分別當作 x／y 方向的擴散抽樣。
#
# 全程都是 NumPy 向量化的整數位元運算與浮點運算，沒有逐粒子 Python 迴圈，
# 也沒有任何有狀態物件（純函式：同樣的輸入永遠得到同樣的輸出），可以直接
# 對「這一小時所有作用中粒子」一次算完，效能與現有共用 ``Generator`` 方案
# 相當（皆為 O(n) 向量化呼叫），統計品質見
# ``tests/test_native_hour_major.py`` 的均值／變異數／自相關檢定。
#
# 誠實揭露：這不是密碼學安全的雜湊，也不是像 PCG64／Philox 那樣經過完整
# TestU01／PractRand 統計套件驗證過的正式亂數演算法函式庫；SplitMix64
# 本身作為「不需要密碼學安全性的高品質 64-bit 狀態終結函式」在業界與學術界
# 已有相當長的使用歷史（例如作為 xoshiro／xoroshiro 系列的種子初始化器），
# 本模組的統計驗證（見測試檔）只涵蓋均值、變異數、成對相關與單一粒子跨步
# 自相關，不是完整的隨機性套件驗證；若未來對更嚴謹的統計品質有疑慮，可以
# 用同一個 ``(particle_step_counter)`` 介面替換成 Philox／Threefry 等
# 「計數器型」正式演算法，呼叫端（``mirror_rk4_substep_loop``／
# ``run_backward_batch``／``hour_major.py``）都不需要跟著修改。

_SPLITMIX64_MUL1 = np.uint64(0xBF58476D1CE4E5B9)
_SPLITMIX64_MUL2 = np.uint64(0x94D049BB133111EB)
_SPLITMIX64_GOLDEN_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_SPLITMIX64_SALT = np.uint64(0xD6E8FEB86659FD93)
_SPLITMIX64_MAX53 = np.float64(2.0**53)


def _splitmix64_finalize(state: np.ndarray) -> np.ndarray:
    """SplitMix64 的終結混合函式，把任意 64-bit 狀態打散成看似均勻的 64-bit 值。

    輸入輸出皆為 ``numpy.uint64`` 陣列；乘法、位移在 numpy 的 ``uint64``
    dtype 下天然以模 ``2**64`` 迴繞（與 C 的無號 64-bit 整數運算語意相同，
    已在開發期以互動測試確認 numpy 不會對這個 dtype 的溢位丟出例外或警告），
    因此可以放心把整個函式當成模 ``2**64`` 的純數論運算，不需要另外處理
    溢位或負值。
    """

    state = state.astype(np.uint64, copy=False)
    state = state ^ (state >> np.uint64(30))
    state = state * _SPLITMIX64_MUL1
    state = state ^ (state >> np.uint64(27))
    state = state * _SPLITMIX64_MUL2
    state = state ^ (state >> np.uint64(31))
    return state


def derive_particle_diffusion_seed_lo_hi_u64(
    seed_hex_values: Sequence[str], *, master_seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """為每一顆粒子各自推導一組獨立的 64-bit 擴散種子半段 ``(seed_lo, seed_hi)``。

    與 ``derive_batch_diffusion_seed``（批次共用一個種子）不同，這裡**每一列
    各自**以自己的 ``seed_hex``（``ParticleIdentityTable.seed_hex``，即
    ``RunUnit.seed`` 的 128-bit 十六進位表示，見 ``particle_input.py``）與
    共同的 ``master_seed`` 一起做 SHA-256，取雜湊摘要的前 16 個位元組拆成
    兩個 64-bit 半段；同一個 ``(seed_hex, master_seed)`` 永遠得到同一組
    ``(lo, hi)``，且與這顆粒子在陣列中的位置、與其他粒子的種子完全無關
    （SHA-256 逐列獨立計算，不像 ``derive_batch_diffusion_seed`` 那樣先把
    整批排序、串接再雜湊一次）。

    這個函式本身**不是**熱路徑（只在一批粒子第一次啟動時呼叫一次，往後每
    小時的擴散抽樣直接用回傳的 ``(seed_lo, seed_hi)`` 陣列＋
    ``counter_based_gaussian_pairs``，不重新雜湊種子），因此即使對 50 萬顆
    粒子逐列呼叫 Python 的 ``hashlib.sha256``（而非向量化位元運算），效能
    仍可接受；換取的是雜湊輸入與既有 ``derive_member_seed``
    （``scenarios.py``）、``derive_batch_diffusion_seed`` 一致，全部走
    SHA-256、不使用受 ``PYTHONHASHSEED`` 影響的 Python 內建 ``hash``。

    Returns:
        ``(seed_lo, seed_hi)``：兩個 ``(n,)`` 的 ``numpy.uint64`` 陣列，
        ``n = len(seed_hex_values)``。
    """

    n = len(seed_hex_values)
    if n == 0:
        raise BackwardDriverError("seed_hex_values 不可為空")
    seed_lo = np.empty(n, dtype=np.uint64)
    seed_hi = np.empty(n, dtype=np.uint64)
    master_suffix = f"|{int(master_seed)}".encode()
    for row, seed_hex in enumerate(seed_hex_values):
        digest = hashlib.sha256(seed_hex.encode("utf-8") + master_suffix).digest()
        seed_lo[row] = int.from_bytes(digest[0:8], "big", signed=False)
        seed_hi[row] = int.from_bytes(digest[8:16], "big", signed=False)
    return seed_lo, seed_hi


def counter_based_gaussian_pairs(
    seed_lo: np.ndarray, seed_hi: np.ndarray, counter: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """依「粒子種子＋粒子自己的子步計數器」算出一對獨立標準常態亂數。

    純函式、全程向量化（見模組文件「計數器型擴散亂數」一節的完整設計說明
    與取捨）：同樣的 ``(seed_lo[i], seed_hi[i], counter[i])`` 永遠得到同樣的
    ``(z0[i], z1[i])``，與陣列中其他列的內容、陣列長度、呼叫順序完全無關。

    Args:
        seed_lo, seed_hi: ``(n,)`` ``numpy.uint64``，見
            ``derive_particle_diffusion_seed_lo_hi_u64``。
        counter: ``(n,)`` 整數陣列（呼叫端通常是 ``int64``），這顆粒子從
            啟動以來已經套用過幾次擴散（0-based）；呼叫端負責只在真正對該
            粒子套用擴散時才遞增，見 ``_apply_horizontal_diffusion``。

    Returns:
        ``(z0, z1)``：兩個 ``(n,)`` 的 ``float64`` 標準常態亂數陣列，分別
        供 x／y 方向的擴散位移使用。
    """

    seed_lo64 = np.asarray(seed_lo, dtype=np.uint64)
    seed_hi64 = np.asarray(seed_hi, dtype=np.uint64)
    counter_u64 = np.asarray(counter, dtype=np.int64).astype(np.uint64)
    if seed_lo64.shape != seed_hi64.shape or seed_lo64.shape != counter_u64.shape:
        raise BackwardDriverError("seed_lo／seed_hi／counter 必須是相同 shape 的一維陣列")

    seed_mixed = _splitmix64_finalize(seed_lo64 ^ _splitmix64_finalize(seed_hi64))
    state = seed_mixed + counter_u64 * _SPLITMIX64_GOLDEN_GAMMA
    bits1 = _splitmix64_finalize(state)
    bits2 = _splitmix64_finalize(state ^ _SPLITMIX64_SALT)
    # 取高 53 個位元（對應 float64 尾數精度）轉成 (0,1) 開區間均勻亂數：
    # 分子加 1、分母加 1，讓值域嚴格落在 (0,1) 之間（不含 0 與 1），避免
    # u1 剛好等於 0 時 Box–Muller 的 ``log(u1)`` 產生 -inf。
    u1 = ((bits1 >> np.uint64(11)).astype(np.float64) + 1.0) / (_SPLITMIX64_MAX53 + 1.0)
    u2 = ((bits2 >> np.uint64(11)).astype(np.float64) + 1.0) / (_SPLITMIX64_MAX53 + 1.0)
    radius = np.sqrt(-2.0 * np.log(u1))
    angle = 2.0 * np.pi * u2
    z0 = radius * np.cos(angle)
    z1 = radius * np.sin(angle)
    return z0, z1


def _apply_horizontal_diffusion(
    px: np.ndarray,
    py: np.ndarray,
    active_mask: np.ndarray,
    *,
    kh_mps2: np.ndarray,
    dtb: float,
    rng: np.random.Generator | None = None,
    particle_seed_lo: np.ndarray | None = None,
    particle_seed_hi: np.ndarray | None = None,
    particle_step_counter: np.ndarray | None = None,
) -> None:
    """對作用中粒子施加一次算子分裂的水平隨機擴散位移（工作書式(9)）。

    ``Δx = N(0,1) * sqrt(2 * Kh * dtb)``，x／y 各自獨立抽樣一次常態亂數；
    這是標準布朗運動離散化，均方位移 ``E[Δx^2] = 2*Kh*dtb`` 正是
    ``mirror_rk4_substep_loop`` 逐子步累加後、經 ``dtb`` 加總得到
    ``2*Kh*t``（``t`` 為總擴散時間）的來源，供驗收 4 的統計檢定直接對照。

    兩種互斥的亂數來源（呼叫端必須恰好提供其中一組，見
    ``mirror_rk4_substep_loop`` 的參數驗證）：

    1. **legacy／批次共用串流**（``rng`` 非 ``None``）：與本函式最初版本
       完全相同的行為，供既有直接呼叫 ``mirror_rk4_substep_loop`` 的低階
       測試與呼叫端維持位元相容，見 ``derive_batch_diffusion_seed``
       docstring 的取捨說明。這裡永遠對**全部** ``n`` 顆粒子抽樣（不論是否
       ``active_mask``），只有套用時才用 ``active_mask`` 遮罩，理由同上。
    2. **計數器型**（``particle_seed_lo``／``particle_seed_hi``／
       ``particle_step_counter`` 三者皆非 ``None``）：見
       ``counter_based_gaussian_pairs`` 與模組文件「計數器型擴散亂數」
       一節。這裡**只**對 ``active_mask`` 為真的粒子抽樣＋遞增計數器；
       不需要「對全部粒子抽樣」的技巧，因為每顆粒子的亂數只跟自己的種子與
       自己的計數器有關，其他粒子是否抽樣、抽了幾次都不影響它。
       ``particle_step_counter`` 會被原地遞增（``+= 1``，只加在
       ``active_mask`` 為真的列），呼叫端應該用同一個陣列跨越多個小時
       重複傳入，讓計數器連續累加。

    ``kh_mps2`` 允許逐粒子不同（水平擴散係數本來就隨位置變化）。
    """

    n = px.shape[0]
    if dtb < 0.0:
        raise BackwardDriverError("dtb 不可為負")

    if rng is not None:
        draws = rng.standard_normal(size=(n, 2))
        dx_std = draws[:, 0]
        dy_std = draws[:, 1]
    else:
        if particle_seed_lo is None or particle_seed_hi is None or particle_step_counter is None:
            raise BackwardDriverError(
                "_apply_horizontal_diffusion 必須提供 rng，或同時提供 particle_seed_lo／"
                "particle_seed_hi／particle_step_counter 三者"
            )
        dx_std, dy_std = counter_based_gaussian_pairs(
            particle_seed_lo, particle_seed_hi, particle_step_counter
        )
        particle_step_counter[active_mask] += 1

    scale = np.sqrt(2.0 * np.maximum(kh_mps2, 0.0) * dtb)
    dx = dx_std * scale
    dy = dy_std * scale
    px[active_mask] += dx[active_mask]
    py[active_mask] += dy[active_mask]


# ── 逐子步鏡像迴圈：事實 2、3 的具體實作 ────────────────────────────────


@dataclass
class SubstepDiagnostics:
    """一次 ``mirror_rk4_substep_loop`` 呼叫（即一個流場小時）的效能拆解。

    每個欄位是這個小時內、該階段耗費的總秒數（跨全部 ``ndeltp`` 個子步
    累加），供驗收 8 的「µs/粒子步」拆解使用。``n_substeps``、
    ``n_particle_substeps`` 讓呼叫端可以自行換算成每粒子每步的平均耗時。
    """

    n_substeps: int = 0
    n_particle_substeps: int = 0
    time_move_s: float = 0.0
    time_boundary_s: float = 0.0
    time_clamp_s: float = 0.0
    time_interpolate_s: float = 0.0
    time_diffusivity_sample_s: float = 0.0
    time_diffusion_apply_s: float = 0.0
    time_interest_check_s: float = 0.0

    def add(self, other: SubstepDiagnostics) -> None:
        """把另一個小時的拆解累加進本物件，供多小時彙整成整批總覽。"""

        self.n_substeps += other.n_substeps
        self.n_particle_substeps += other.n_particle_substeps
        self.time_move_s += other.time_move_s
        self.time_boundary_s += other.time_boundary_s
        self.time_clamp_s += other.time_clamp_s
        self.time_interpolate_s += other.time_interpolate_s
        self.time_diffusivity_sample_s += other.time_diffusivity_sample_s
        self.time_diffusion_apply_s += other.time_diffusion_apply_s
        self.time_interest_check_s += other.time_interest_check_s

    @property
    def total_kernel_and_overhead_s(self) -> float:
        """回傳這個小時全部階段的總秒數，作為分母換算 µs/粒子步的基準。"""

        return (
            self.time_move_s
            + self.time_boundary_s
            + self.time_clamp_s
            + self.time_interpolate_s
            + self.time_diffusivity_sample_s
            + self.time_diffusion_apply_s
            + self.time_interest_check_s
        )


@dataclass
class HourlyOverheadDiagnostics:
    """逐小時「與粒子數無關」的固定開銷拆解（任務 3 效能量測用）。

    背景
    ------
    ``SubstepDiagnostics`` 拆解的是 njit kernel 本身的耗時，這些成本隨粒子
    數（更精確地說，隨「粒子數 × 子步數」）線性增加，用「µs/粒子步」表示
    才有意義。但 ``run_backward_batch`` 每個流場小時還要做一系列**只跟網格
    規模有關、與這一小時要驅動幾顆粒子完全無關**的工作：把 OCM 與 NWW3
    內插到節點（在真實部署中由 ``forcing_bridge``／``wave_bridge`` 負責，
    這裡量測的是呼叫 ``forcing_resolver``／``wave_field_resolver`` 本身的
    耗時）、對整個 ``(nvrt, nnode)`` 陣列做乾濕判定與 Stokes 全網格計算
    （``compose_current_and_stokes_fields``）、Smagorinsky 水平擴散係數場
    （``compute_smagorinsky_kh_field``）、CFL 子步數估計
    （``estimate_substep_count``）。

    這些「每小時固定開銷」在批次很大（例如兩千顆粒子）時，相對於
    kernel 總耗時可以忽略；但在批次很小（例如二十顆粒子）時可能反過來
    主導總耗時——這正是任務 3 要求量測與（若需要）優化的對象。本類別把
    這些開銷獨立拆出來，讓呼叫端可以分別看到「攤提後的每粒子步成本」與
    「完全與粒子數無關的固定成本」，不會被混在一起誤讀。

    ``*_cache_hit``／``*_cache_miss`` 計數搭配 ``composed_base_cache``／
    ``smagorinsky_cache``（見 ``run_backward_batch``）使用：多個沉降速度
    批次共用同一個快取字典時，只有第一個批次會是 cache miss（付出
    ``time_compose_base_s``／``time_smagorinsky_s`` 的完整成本），其餘批次
    命中快取，幾乎不需要額外時間（只有 ``time_apply_settling_s`` 這個便宜
    步驟每批次都要做一次）。
    """

    n_hours: int = 0
    time_forcing_resolve_s: float = 0.0
    time_wave_resolve_s: float = 0.0
    time_wet_dry_check_s: float = 0.0
    time_compose_base_s: float = 0.0
    compose_base_cache_hits: int = 0
    compose_base_cache_misses: int = 0
    time_apply_settling_s: float = 0.0
    time_smagorinsky_s: float = 0.0
    smagorinsky_cache_hits: int = 0
    smagorinsky_cache_misses: int = 0
    time_substep_count_s: float = 0.0
    stokes_points_evaluated: int = 0
    """WP-9 效能診斷：只在 ``compose_base_cache`` 未命中（真正呼叫過
    ``compose_current_and_stokes_fields``）的小時才累加，累加值就是那些小時
    ``ComposedCurrentAndStokesFields.stokes_points_evaluated`` 的總和；搭配
    ``stokes_points_possible`` 可以算出深度截斷優化後，全部小時平均下來
    真正重新求解 Stokes 的 (層,節點) 點數比例。"""
    stokes_points_possible: int = 0
    """與 ``stokes_points_evaluated`` 對應的分母，見上。"""

    def add(self, other: HourlyOverheadDiagnostics) -> None:
        """把另一個小時的拆解累加進本物件，供多小時彙整成整批總覽。"""

        self.n_hours += other.n_hours
        self.time_forcing_resolve_s += other.time_forcing_resolve_s
        self.time_wave_resolve_s += other.time_wave_resolve_s
        self.time_wet_dry_check_s += other.time_wet_dry_check_s
        self.time_compose_base_s += other.time_compose_base_s
        self.compose_base_cache_hits += other.compose_base_cache_hits
        self.compose_base_cache_misses += other.compose_base_cache_misses
        self.time_apply_settling_s += other.time_apply_settling_s
        self.time_smagorinsky_s += other.time_smagorinsky_s
        self.smagorinsky_cache_hits += other.smagorinsky_cache_hits
        self.smagorinsky_cache_misses += other.smagorinsky_cache_misses
        self.time_substep_count_s += other.time_substep_count_s
        self.stokes_points_evaluated += other.stokes_points_evaluated
        self.stokes_points_possible += other.stokes_points_possible

    @property
    def total_s(self) -> float:
        """這一批次全部小時、全部「與粒子數無關」開銷的總秒數。"""

        return (
            self.time_forcing_resolve_s
            + self.time_wave_resolve_s
            + self.time_wet_dry_check_s
            + self.time_compose_base_s
            + self.time_apply_settling_s
            + self.time_smagorinsky_s
            + self.time_substep_count_s
        )


@dataclass(frozen=True, slots=True)
class DepartureRecord:
    """一顆粒子離開追蹤（因任何終止原因）時的位置、時刻與狀態碼。"""

    row_particle_index: int
    status: int
    x_m: float
    y_m: float
    z_m: float
    utc_ns: int


def mirror_rk4_substep_loop(
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
    rng: np.random.Generator | None = None,
    particle_seed_lo: np.ndarray | None = None,
    particle_seed_hi: np.ndarray | None = None,
    particle_step_counter: np.ndarray | None = None,
    native_status: np.ndarray | None = None,
    substep_utc_ns: Callable[[int], int] | None = None,
    interest_region_polygon: Polygon | None = None,
    interest_region_polygons: np.ndarray | None = None,
    departures: list[DepartureRecord] | None = None,
) -> SubstepDiagnostics:
    """鏡像 ``ParticleTracker.batch_update_particles`` 的子步迴圈（passive＋RK4）。

    逐段對應 ``tracker.py`` 1519-1674 行（僅保留本模組會用到的分支：
    ``mod_part=0``、``iadvection=2``、``istiff=0``、無 debug dump、無
    oil weathering），並在 MOVE 之後插入算子分裂的擴散（``enable_diffusion``
    為真時）：

    - STEP 0（Stokes，``tracker.py`` 1525-1542 行）：**略過**。本函式假設
      Stokes 與沉降已經在呼叫端經 ``compose_total_velocity_fields`` 併入
      ``composed`` 的欄位陣列，因此不需要、也不應該再對 ``pa.u``／``pa.v``
      額外加總（那正是事實 1 指出的錯誤做法）。
    - STEP 1 MOVE（1544-1600 行，只保留 RK4 分支）：呼叫未修改的
      ``handles.batch_move_rk4``。
    - **［新增]** 擴散：若 ``enable_diffusion``，先以未修改的
      ``handles.batch_interpolate_diffusivity``（``ihdf=1``）在 MOVE 之前
      的位置／元素取樣 Kh（見 ``compute_smagorinsky_kh_field`` 準備的
      ``hf1``／``hf2`` 場），再以 ``_apply_horizontal_diffusion`` 施加位移；
      這一步發生在 RK4 完整算完之後，不在 k1-k4 任何一個 stage 內部
      （見模組 docstring「事實 2」）。停用時（``enable_diffusion=False``）
      這一段完全不執行，讓本函式在停用擴散時與逐行對應
      ``batch_update_particles`` 的行為位元相同（驗收 1）。
    - STEP 2 SEARCH（1602-1603 行）：呼叫未修改的
      ``handles.batch_check_boundaries``；若任何粒子回報
      ``needs_fallback!=0``，直接 fail-closed（見模組 docstring「已知
      限制」第 1 點，本函式不含 Python 層 KD-tree 後援）。
    - STEP 2b clamp（1617 行）：呼叫未修改的 ``handles.batch_clamp_z``。
    - STEP 3 INTERPOLATE（1622 行）：呼叫未修改的
      ``handles.batch_interpolate_velocity``，更新 ``pa.u/v/w`` 供下一個
      子步的 k1 使用。
    - 每個子步結束後（若提供 ``interest_region_polygon`` 或
      ``interest_region_polygons``）：以 Shapely 2.0 向量化 ``contains_xy``
      一次判斷全部作用中粒子是否仍在關注區域內（不對任何粒子逐一建立
      Shapely 物件），新離開者記錄 ``DepartureRecord`` 並停用
      （``pa.status=0``、``native_status=NATIVE_STATUS_LEFT_INTEREST_REGION``）。

    Args:
        pa: 真正的（未修改的）``ParticleArrays`` 執行個體，``x/y/z/u/v/w/
            x_prev/y_prev/element/status/inbr/rnds`` 等欄位會被原地修改。
        ibf: ``+1``（正向，僅供測試使用）或 ``-1``（逆向，正式用途）。
        ndeltp、dtb: 這一小時的子步數與子步秒數，``ndeltp*dtb`` 應等於
            這一小時的秒數（呼叫端保證，本函式不重複檢查）。
        composed: ``compose_total_velocity_fields`` 的輸出。
        rng: 啟用擴散時的**legacy／批次共用串流**亂數來源，與
            ``particle_seed_lo``／``particle_seed_hi``／
            ``particle_step_counter`` 三者擇一提供（不可同時提供、也不可
            都不提供），見 ``_apply_horizontal_diffusion`` docstring 兩種
            模式的說明。
        particle_seed_lo, particle_seed_hi, particle_step_counter:
            啟用擴散時的**計數器型**亂數來源，三者必須同時提供（見
            ``derive_particle_diffusion_seed_lo_hi_u64``／
            ``counter_based_gaussian_pairs``）；``particle_step_counter``
            會被原地遞增，呼叫端應該用同一個陣列跨越多個小時重複傳入。
            這是 WP-9 小時主軸排程使用的模式，讓合併批次的擴散序列與批次
            組成、列序、worker 數完全無關（見模組文件「計數器型擴散亂數」
            一節）。
        native_status: ``(n,)`` int32，本函式只在偵測到離開關注區域時
            寫入 ``NATIVE_STATUS_LEFT_INTEREST_REGION``；其餘終止原因由
            呼叫端（``run_backward_batch``）在小時邊界另行判定。提供
            ``interest_region_polygon``／``interest_region_polygons`` 時
            必須同時提供本參數。
        substep_utc_ns: 給定子步索引 ``idt``（``1..ndeltp``），回傳該子步
            結束時對應的 UTC 奈秒，供 ``DepartureRecord.utc_ns`` 使用；
            提供 ``interest_region_polygon``／``interest_region_polygons``
            時必須同時提供。
        interest_region_polygon: **單一**關注區域多邊形，套用到全部 ``n``
            顆粒子——適用於「整批粒子共用同一個測站」的既有逐單元路徑
            （``run_backward_batch``）。與 ``interest_region_polygons``
            互斥。
        interest_region_polygons: ``(n,)`` 的 Shapely ``Polygon`` object
            陣列，**逐列**指定每顆粒子自己的關注區域——適用於 WP-9
            小時主軸排程「同一次 kernel 呼叫混合多個測站的粒子」的情境
            （見 ``hour_major.py``）。Shapely 2.0 的向量化函式（例如
            ``contains_xy``）支援「geometry 引數本身也是與座標陣列等長的
            object 陣列」，逐點配對計算，因此這裡對每個粒子使用正確多邊形
            的計算結果，與「把每個測站的粒子分開、各自呼叫一次
            ``interest_region_polygon``」完全等價（幾何述詞是逐點獨立
            計算，不受同一次呼叫還有哪些其他點／多邊形影響），可以安全
            合併呼叫而不改變任何一顆粒子的判定結果。與
            ``interest_region_polygon`` 互斥。

    Returns:
        ``SubstepDiagnostics``，這個小時的效能拆解。
    """

    if ibf not in (1, -1):
        raise BackwardDriverError("ibf 只能是 1 或 -1")
    if ndeltp < 1:
        raise BackwardDriverError("ndeltp 必須是正整數")
    if interest_region_polygon is not None and interest_region_polygons is not None:
        raise BackwardDriverError("interest_region_polygon 與 interest_region_polygons 只能擇一提供")
    if (interest_region_polygon is not None or interest_region_polygons is not None) and (
        native_status is None or substep_utc_ns is None
    ):
        raise BackwardDriverError(
            "提供 interest_region_polygon／interest_region_polygons 時必須同時提供 "
            "native_status 與 substep_utc_ns"
        )
    if enable_diffusion:
        if hf1_field is None or hf2_field is None:
            raise BackwardDriverError("啟用擴散時必須提供 hf1_field 與 hf2_field")
        using_rng = rng is not None
        counter_fields = (particle_seed_lo, particle_seed_hi, particle_step_counter)
        using_counter = any(field is not None for field in counter_fields)
        if using_rng and using_counter:
            raise BackwardDriverError(
                "啟用擴散時 rng 與計數器型參數（particle_seed_lo／particle_seed_hi／"
                "particle_step_counter）只能擇一提供"
            )
        if using_counter and any(field is None for field in counter_fields):
            raise BackwardDriverError(
                "計數器型擴散必須同時提供 particle_seed_lo／particle_seed_hi／"
                "particle_step_counter 三者，不可只提供部分"
            )
        if not using_rng and not using_counter:
            raise BackwardDriverError(
                "啟用擴散時必須提供 rng，或同時提供 particle_seed_lo／particle_seed_hi／"
                "particle_step_counter 三者"
            )

    import time as _time

    # 注意：這裡刻意全部取自 ``driver_grid.schism_grid``（真正的 SchismGrid
    # 執行個體），而不是 ``driver_grid.grid``（WP-1 的 ``NativeGridArrays``）。
    # 後者只保留 kernel 需要的最小欄位子集（沒有 ``x``／``xctr``／``yctr``／
    # ``elside``／``isidenode``／``snx``／``sny``／``isbnd``，這些是
    # ``build_native_backward_driver_grid`` 呼叫 ``SchismGrid`` 自己的方法
    # 才建立出來的邊結構欄位）；``schism_grid`` 在建構時已把 ``i34``／
    # ``elnode``／``x``／``y``／``ic3``／``nvrt``／``kbp``／``kbe``／
    # ``sigma_lcl``／``dp`` 直接設成與 ``NativeGridArrays`` 相同的陣列（見
    # ``build_native_backward_driver_grid``），因此兩者對這些共用欄位的值
    #永遠一致，統一從 ``schism_grid`` 讀取只是為了同時取得邊結構欄位。
    grid = driver_grid.schism_grid
    i34 = grid.i34
    elnode = grid.elnode
    grid_x = grid.x
    grid_y = grid.y
    ic3 = grid.ic3
    nvrt = grid.nvrt
    kbp = grid.kbp
    kbe = grid.kbe
    sigma_lcl = grid.sigma_lcl
    dp = grid.dp
    xctr = grid.xctr
    yctr = grid.yctr
    elside = grid.elside
    isidenode = grid.isidenode
    snx = grid.snx
    sny = grid.sny
    isbnd = grid.isbnd

    diag = SubstepDiagnostics()
    n = pa.n
    x0 = pa.x.copy()
    y0 = pa.y.copy()
    z0 = pa.z.copy()
    needs_fallback = np.zeros(n, dtype=np.int32)

    zero_diff_field = None
    if enable_diffusion:
        zero_diff_field = np.zeros((nvrt, grid.np), dtype=np.float64)

    for idt in range(1, ndeltp + 1):
        trat = (idt / ndeltp) if ibf == 1 else ((idt - 1) / ndeltp)

        if enable_diffusion:
            t0 = _time.perf_counter()
            handles.batch_interpolate_diffusivity(
                pa.x, pa.y, pa.z, pa.element, pa.status,
                i34, elnode, grid_x, grid_y,
                nvrt, kbp, kbe, sigma_lcl, dp,
                zero_diff_field, zero_diff_field, composed.eta_field1,
                hf1_field, hf2_field,
                trat, driver_grid.h0, 1, hdc,
                pa.dhfx, pa.dhfy, pa.dhfz,
                pa.grdx, pa.grdy, pa.grdz,
                0, 1.2e-5, 3.0e-4, 30.0,
                pa.wind_x, pa.wind_y,
            )
            diag.time_diffusivity_sample_s += _time.perf_counter() - t0

        t0 = _time.perf_counter()
        handles.batch_move_rk4(
            pa.x, pa.y, pa.z, pa.u, pa.v, pa.w,
            pa.x_prev, pa.y_prev, pa.element, pa.status,
            x0, y0, z0,
            i34, elnode, grid_x, grid_y, ic3,
            nvrt, kbp, kbe, sigma_lcl, dp,
            composed.u_field1, composed.v_field1, composed.w_field1, composed.eta_field1,
            composed.u_field2, composed.v_field2, composed.w_field2,
            ibf, dtb, trat,
        )
        diag.time_move_s += _time.perf_counter() - t0

        if enable_diffusion:
            t0 = _time.perf_counter()
            active_mask = pa.status == 1
            _apply_horizontal_diffusion(
                pa.x, pa.y, active_mask, kh_mps2=pa.dhfx, dtb=dtb, rng=rng,
                particle_seed_lo=particle_seed_lo,
                particle_seed_hi=particle_seed_hi,
                particle_step_counter=particle_step_counter,
            )
            diag.time_diffusion_apply_s += _time.perf_counter() - t0

        t0 = _time.perf_counter()
        needs_fallback[:] = 0
        handles.batch_check_boundaries(
            pa.x, pa.y, pa.x_prev, pa.y_prev,
            pa.element, pa.status, pa.inbr, pa.rnds,
            i34, elnode, grid_x, grid_y, xctr, yctr,
            elside, isidenode, ic3, snx, sny, isbnd,
            composed.elem_is_dry.astype(np.int64),
            0, 1.0,
            needs_fallback,
        )
        if np.any(needs_fallback != 0):
            raise BackwardDriverError(
                f"{int(np.count_nonzero(needs_fallback))} 顆粒子在子步 {idt} 觸發 "
                "needs_fallback（一般後援或 death trap），本驅動層不含 Python 層 "
                "KD-tree 後援（見模組 docstring 已知限制第 1 點），fail-closed 中止"
            )
        diag.time_boundary_s += _time.perf_counter() - t0

        t0 = _time.perf_counter()
        handles.batch_clamp_z(
            pa.x, pa.y, pa.z, pa.element, pa.status,
            elnode, grid_x, grid_y,
            nvrt, kbp, kbe, sigma_lcl, dp, composed.eta_field1,
        )
        diag.time_clamp_s += _time.perf_counter() - t0

        t0 = _time.perf_counter()
        handles.batch_interpolate_velocity(
            pa.x, pa.y, pa.z, pa.element, pa.status,
            i34, elnode, grid_x, grid_y,
            nvrt, kbp, kbe, sigma_lcl, dp,
            composed.u_field1, composed.v_field1, composed.w_field1, composed.eta_field1,
            composed.u_field2, composed.v_field2, composed.w_field2,
            trat, driver_grid.h0,
            pa.u, pa.v, pa.w,
        )
        diag.time_interpolate_s += _time.perf_counter() - t0

        if interest_region_polygon is not None or interest_region_polygons is not None:
            t0 = _time.perf_counter()
            active_mask = pa.status == 1
            if np.any(active_mask):
                inside = np.ones(n, dtype=bool)
                if interest_region_polygon is not None:
                    inside[active_mask] = contains_xy(
                        interest_region_polygon, pa.x[active_mask], pa.y[active_mask]
                    )
                else:
                    inside[active_mask] = contains_xy(
                        interest_region_polygons[active_mask], pa.x[active_mask], pa.y[active_mask]
                    )
                left_now = active_mask & ~inside
                if np.any(left_now):
                    utc_ns = int(substep_utc_ns(idt))
                    if departures is not None:
                        for row in np.flatnonzero(left_now):
                            departures.append(
                                DepartureRecord(
                                    row_particle_index=int(row),
                                    status=NATIVE_STATUS_LEFT_INTEREST_REGION,
                                    x_m=float(pa.x[row]),
                                    y_m=float(pa.y[row]),
                                    z_m=float(pa.z[row]),
                                    utc_ns=utc_ns,
                                )
                            )
                    native_status[left_now] = NATIVE_STATUS_LEFT_INTEREST_REGION
                    pa.status[left_now] = 0
            diag.time_interest_check_s += _time.perf_counter() - t0

        x0[:] = pa.x
        y0[:] = pa.y
        z0[:] = pa.z

    diag.n_substeps = ndeltp
    diag.n_particle_substeps = ndeltp * n
    return diag


# ── 初始元素定位：重用 ParticleTracker.find_initial_elements ────────────────


def locate_initial_elements(
    driver_grid: NativeBackwardDriverGrid,
    handles: BayTraceModuleHandles,
    pa: Any,
    *,
    eta_m: np.ndarray,
    u_current_mps: np.ndarray,
    v_current_mps: np.ndarray,
    w_current_mps: np.ndarray,
) -> np.ndarray:
    """重用未修改的 ``ParticleTracker.find_initial_elements`` 定位起始元素。

    這是 WP-4 明確保留給 WP-5 的工作（見 ``particle_input.py`` docstring
    「本模組刻意不做的事」第 2 點）。呼叫後會原地修改 ``pa.element``
    （找到的元素）、``pa.status``（找到者設為 1，找不到者設為 -1）、
    ``pa.z``（依 kbe 夾限的初始垂向位置）與 ``pa.u/v/w``（起點速度，注意
    這裡使用**純海流**速度初始化，不含 Stokes／沉降——因為
    ``find_initial_elements`` 內部的 ``batch_init_velocity`` 不做垂向
    內插，只讀 ``jlev`` 單層；下一次呼叫 ``mirror_rk4_substep_loop`` 的
    第一個子步會立刻用 ``compose_total_velocity_fields`` 的合成場重新
    內插覆寫，這裡的初始值只是 RK4 k1 的起點快取，很快就會被取代）。

    Args:
        eta_m, u_current_mps, v_current_mps, w_current_mps: 到達時刻（第一
            個流場小時）的節點海面高程與純海流速度，``(nnode,)``／
            ``(nvrt,nnode)``。

    Returns:
        ``(n,)`` 布林陣列，``True`` 表示該粒子起始位置落在網格外，找不到
        元素；呼叫端應把這些粒子標成 ``NATIVE_STATUS_NUMERICAL_FAILURE``
        （最接近的既有分類：起始位置資料契約異常，不是六種正常終止原因
        之一，詳見模組文件與最終報告的誠實揭露）。
    """

    shim = _TrackerMethodShim(driver_grid.schism_grid, h0=driver_grid.h0, horcon=0.2)
    shim.pa = pa
    shim.eta_field1 = np.ascontiguousarray(eta_m, dtype=np.float64)
    shim.u_field2 = np.ascontiguousarray(u_current_mps, dtype=np.float64)
    shim.v_field2 = np.ascontiguousarray(v_current_mps, dtype=np.float64)
    shim.w_field2 = np.ascontiguousarray(w_current_mps, dtype=np.float64)
    handles.tracker_cls.find_initial_elements(shim)
    return pa.status == -1


# ── 停留於海床附近的偵測：供沉底判定使用 ────────────────────────────────


def particle_bed_z_levels(
    driver_grid: NativeBackwardDriverGrid, eta_m: np.ndarray, elements: np.ndarray
) -> np.ndarray:
    """回傳每顆粒子目前所在元素的海床 z（公尺），供沉底判定比較。

    簡化自 ``batch_clamp_z``（``batch_interp.py`` 703-756 行）的元素級
    公式：原版以粒子實際 (x,y) 的面積座標加權平均三個節點的 z；這裡為了
    避免重算面積座標，改用元素三個節點的等權重平均（1/3、1/3、1/3）取
    ``kbe``（元素底層）那一層當作海床 z 的近似值。這只是「是否停留在海床
    附近」的粗略診斷閾值比較，不是粒子位置本身的計算（粒子真正的 z 仍然
    完全由未修改的 ``batch_clamp_z``／``batch_move_rk4`` 決定），因此這個
    近似不影響任何粒子的實際軌跡，只影響「沉底」狀態碼何時被標記。
    """

    grid = driver_grid.grid
    n = elements.shape[0]
    bed_z = np.full(n, np.nan, dtype=np.float64)
    valid = elements >= 0
    if not np.any(valid):
        return bed_z
    elem_valid = elements[valid]
    nodes = grid.elnode[elem_valid, :3]
    z_nodes = _vectorized_z_levels(grid.sigma_lcl, grid.kbp, grid.dp, eta_m)
    kbe_valid = grid.kbe[elem_valid]
    bed_z[valid] = z_nodes[kbe_valid, nodes[:, 0]] / 3.0 + z_nodes[kbe_valid, nodes[:, 1]] / 3.0 + z_nodes[
        kbe_valid, nodes[:, 2]
    ] / 3.0
    return bed_z


# ── 主要驅動器：逐小時遞減 UTC，直到離開關注區域或觸發任一終止條件 ─────────


@dataclass(frozen=True, slots=True)
class BackwardDriverConfig:
    """一次 ``run_backward_batch`` 呼叫的設定；同一批粒子共用同一組設定。

    ``arrival_time_utc_ns``／``settling_velocity_mps`` 必須與這批粒子的
    ``BatchGroup`` 鍵值一致（呼叫端應先以
    ``validate_arrival_time_homogeneous_batches``／
    ``validate_settling_velocity_homogeneous_batches`` 確認）。

    ``stokes_depth_threshold_ratio``：見
    ``compose_current_and_stokes_fields`` docstring 的「Stokes 深度截斷」
    優化；預設 ``None`` 不啟用，維持此欄位新增前完全相同的行為。WP-9
    小時主軸路徑（``hour_major.py``）預設啟用（見該模組文件），若要跟
    ``run_backward_batch`` 做位元相同的等價驗收，兩邊必須明確傳入同一個
    數值。
    """

    arrival_time_utc_ns: int
    settling_velocity_mps: float
    max_backtrack_seconds: float = 30.0 * 86400.0
    dt_max_seconds: float = 30.0
    cfl_number: float = 0.5
    h0: float = 0.0
    horcon: float = 0.2
    enable_diffusion: bool = True
    output_interval_seconds: float = 3600.0
    deposition_consecutive_hours: int = 3
    deposition_tolerance_m: float = 0.5
    master_seed: int = 0
    stokes_depth_threshold_ratio: float | None = None


@dataclass(frozen=True, slots=True)
class BackwardDriverResult:
    """一次 ``run_backward_batch`` 呼叫的完整輸出。

    ``trajectory_*`` 的 shape 都是 ``(n_samples, n_particles)``（含第 0 筆
    即到達時刻的初始狀態），``trajectory_utc_ns`` 是 ``(n_samples,)``。
    ``departures`` 只包含真正發生終止事件的粒子，一顆粒子至多出現一次；
    未在其中出現、且 ``hours_run`` 未達 ``max_backtrack_seconds`` 就結束
    （理論上不會發生，因為主迴圈只在全部粒子都終止或觸及上限才結束）的
    情況不應存在，可作為驗收時的一致性檢查。
    """

    row_particle_index: np.ndarray
    final_status: np.ndarray
    final_x_m: np.ndarray
    final_y_m: np.ndarray
    final_z_m: np.ndarray
    final_utc_ns: np.ndarray
    trajectory_utc_ns: np.ndarray
    trajectory_x_m: np.ndarray
    trajectory_y_m: np.ndarray
    trajectory_z_m: np.ndarray
    trajectory_status: np.ndarray
    departures: tuple[DepartureRecord, ...]
    diagnostics: SubstepDiagnostics
    overhead_diagnostics: HourlyOverheadDiagnostics
    hours_run: int


_HOUR_NS = 3_600_000_000_000


def run_backward_batch(
    driver_grid: NativeBackwardDriverGrid,
    handles: BayTraceModuleHandles,
    soa: Any,
    *,
    seed_hex_values: Sequence[str],
    config: BackwardDriverConfig,
    forcing_resolver: Callable[[int], HourlyFieldPair],
    interest_region_polygon: Polygon,
    wave_field_resolver: Callable[[int], WaveFieldAtNodes] | None = None,
    composed_base_cache: MutableMapping[tuple[int, int], ComposedCurrentAndStokesFields] | None = None,
    smagorinsky_cache: MutableMapping[int, np.ndarray] | None = None,
) -> BackwardDriverResult:
    """對一批「同到達時刻、同沉降速度」的粒子執行完整的逆向批次追蹤。

    Args:
        soa: WP-4 ``build_baytrace_particle_soa`` 的輸出（或欄位相容的
            物件），提供 ``x/y/z/z0/settling_velocity_mps/row_particle_index``
            等起始資料；本函式不修改傳入物件，只讀取其欄位值。
        seed_hex_values: 這批粒子的 ``ParticleIdentityTable.seed_hex``
            （用於推導批次共用的擴散亂數種子，見
            ``derive_batch_diffusion_seed``）。
        forcing_resolver: 給定目標 UTC 奈秒，回傳涵蓋該時刻的
            ``HourlyFieldPair``；正式用途應綁定一個 ``ibf=-1`` 的
            ``NativeForcingBridgeReader.resolve``，測試可用合成函式替代
            （見模組文件「已知限制」第 2 點與測試檔）。
        interest_region_polygon: 關注區域（``own_local_domain``），公尺制
            座標，與網格同一投影。
        wave_field_resolver: 給定目標 UTC 奈秒，回傳該時刻已內插到追蹤
            網格節點的波浪場；``None`` 時完全不啟用 Stokes（見模組文件
            「已知限制」第 2 點：目前沒有 NWW3→節點的橋接模組）。
        composed_base_cache: 選用的外部可變映射，鍵是
            ``(time_before_utc_ns, time_after_utc_ns)``，值是
            ``compose_current_and_stokes_fields`` 的輸出。**同一到達時刻、
            同一關注區域、不同沉降速度**的多個批次（見 WP-4
            ``group_indices_by_settling_velocity_and_arrival_time``）若呼叫
            端在多次 ``run_backward_batch`` 之間傳入同一個字典物件，本函式
            會在第一個批次算好每個小時的乾濕判定＋Stokes 全網格結果後寫進
            這個字典，後續批次直接命中快取，只需另外做一次便宜的
            ``apply_settling_velocity``（見任務 3 效能量測，
            ``HourlyOverheadDiagnostics`` docstring）。``None``（預設）時
            完全不快取，行為與拆分前的 ``compose_total_velocity_fields``
            完全一致，只是內部改成呼叫兩段式函式。呼叫端必須確保共用同一個
            快取的多次呼叫確實對應同一個網格、同一個 ``forcing_resolver``／
            ``wave_field_resolver``（否則快取內容張冠李戴）；本函式不驗證
            這件事，因為快取鍵只有 UTC 時刻，無法判斷「這是不是同一個
            forcing 來源」。
        smagorinsky_cache: 選用的外部可變映射，鍵是
            ``field_pair.time_after_utc_ns``（``compute_smagorinsky_kh_field``
            實際讀取的是 field2，見該函式 docstring），值是計算出的 Kh 場
            （``(nvrt, nnode)``）。用途、共用條件與 ``composed_base_cache``
            相同：Smagorinsky 場同樣只跟純海流與網格有關，與沉降速度無關，
            可以在同一到達時刻的不同沉降批次間共用。``None``（預設）時不
            快取。

    Returns:
        ``BackwardDriverResult``。
    """

    if config.arrival_time_utc_ns <= 0:
        raise BackwardDriverError("arrival_time_utc_ns 必須是正值 UTC 奈秒")

    n = int(np.asarray(soa.x).shape[0])
    pa = handles.particle_arrays_cls(n)
    pa.x[:] = np.asarray(soa.x, dtype=np.float64)
    pa.y[:] = np.asarray(soa.y, dtype=np.float64)
    pa.z[:] = np.asarray(soa.z, dtype=np.float64)
    pa.z0[:] = np.asarray(soa.z0, dtype=np.float64)
    pa.element[:] = -1
    pa.status[:] = 0
    row_particle_index = np.asarray(soa.row_particle_index, dtype=np.int64)
    pa.ids = [str(int(value)) for value in row_particle_index]

    import time as _time

    native_status = np.full(n, NATIVE_STATUS_ACTIVE, dtype=np.int32)
    departures: list[DepartureRecord] = []
    total_diag = SubstepDiagnostics()
    total_overhead = HourlyOverheadDiagnostics()

    arrival_utc = int(config.arrival_time_utc_ns)
    max_backtrack_ns = int(round(config.max_backtrack_seconds * 1.0e9))
    output_interval_ns = int(round(config.output_interval_seconds * 1.0e9))
    if output_interval_ns <= 0:
        raise BackwardDriverError("output_interval_seconds 必須是正值")

    first_pair = forcing_resolver(arrival_utc)
    not_found = locate_initial_elements(
        driver_grid,
        handles,
        pa,
        eta_m=first_pair.eta_field1,
        u_current_mps=first_pair.u_field1,
        v_current_mps=first_pair.v_field1,
        w_current_mps=first_pair.w_field1,
    )
    if np.any(not_found):
        native_status[not_found] = NATIVE_STATUS_NUMERICAL_FAILURE
        for row in np.flatnonzero(not_found):
            departures.append(
                DepartureRecord(
                    row_particle_index=int(row),
                    status=NATIVE_STATUS_NUMERICAL_FAILURE,
                    x_m=float(pa.x[row]),
                    y_m=float(pa.y[row]),
                    z_m=float(pa.z[row]),
                    utc_ns=arrival_utc,
                )
            )
    pa.status[:] = (native_status == NATIVE_STATUS_ACTIVE).astype(np.int64)

    # 擴散亂數：WP-9 起改用計數器型（見 backward_driver 模組文件「計數器型
    # 擴散亂數」一節），每顆粒子各自的種子只跟自己的 seed_hex 與
    # master_seed 有關，與批次組成、列序完全無關；``particle_step_counter``
    # 從 0 開始，只在真正套用擴散的子步才遞增（``_apply_horizontal_diffusion``
    # 負責），並跨越整個回溯過程（含跨小時邊界）持續累加，讓「單獨呼叫
    # run_backward_batch」與「WP-9 小時主軸把同一顆粒子併入更大批次」得到
    # 位元相同的擴散序列（見 ``hour_major.py`` 模組文件與驗收 2）。
    #
    # 舊版 ``derive_batch_diffusion_seed``＋批次共用 ``Generator`` 的方案仍
    # 保留（見該函式 docstring），但不再是本函式的預設路徑：那個方案的
    # 擴散序列與批次組成、列序綁定，無法滿足「合併／不合併批次得到相同
    # 結果」這個 WP-9 的驗收要求。
    particle_seed_lo: np.ndarray | None = None
    particle_seed_hi: np.ndarray | None = None
    particle_step_counter: np.ndarray | None = None
    if config.enable_diffusion:
        particle_seed_lo, particle_seed_hi = derive_particle_diffusion_seed_lo_hi_u64(
            seed_hex_values, master_seed=config.master_seed
        )
        particle_step_counter = np.zeros(n, dtype=np.int64)

    trajectory_utc = [arrival_utc]
    trajectory_x = [pa.x.copy()]
    trajectory_y = [pa.y.copy()]
    trajectory_z = [pa.z.copy()]
    trajectory_status = [native_status.copy()]

    prev_hf2: np.ndarray | None = None
    deposit_streak = np.zeros(n, dtype=np.int64)
    current_utc = arrival_utc
    next_output_utc = arrival_utc - output_interval_ns
    hours_run = 0
    primed = False

    while np.any(native_status == NATIVE_STATUS_ACTIVE):
        elapsed_ns = arrival_utc - current_utc
        if elapsed_ns >= max_backtrack_ns:
            active = native_status == NATIVE_STATUS_ACTIVE
            native_status[active] = NATIVE_STATUS_MAX_BACKTRACK
            pa.status[active] = 0
            for row in np.flatnonzero(active):
                departures.append(
                    DepartureRecord(
                        int(row), NATIVE_STATUS_MAX_BACKTRACK,
                        float(pa.x[row]), float(pa.y[row]), float(pa.z[row]), int(current_utc),
                    )
                )
            break

        hour_overhead = HourlyOverheadDiagnostics(n_hours=1)

        t0 = _time.perf_counter()
        field_pair = forcing_resolver(current_utc)
        hour_overhead.time_forcing_resolve_s += _time.perf_counter() - t0

        t0 = _time.perf_counter()
        node_is_dry_check, _elem_dry_check = compute_wet_dry_flags(
            driver_grid, handles, eta_m=field_pair.eta_field1
        )
        hour_overhead.time_wet_dry_check_s += _time.perf_counter() - t0
        try:
            field_pair.raise_if_unresolved_gap(node_is_dry1=node_is_dry_check, node_is_dry2=node_is_dry_check)
        except UnresolvedForcingGapError:
            active = native_status == NATIVE_STATUS_ACTIVE
            native_status[active] = NATIVE_STATUS_FORCING_MISSING
            pa.status[active] = 0
            for row in np.flatnonzero(active):
                departures.append(
                    DepartureRecord(
                        int(row), NATIVE_STATUS_FORCING_MISSING,
                        float(pa.x[row]), float(pa.y[row]), float(pa.z[row]), int(current_utc),
                    )
                )
            break

        t0 = _time.perf_counter()
        wave1 = (
            wave_field_resolver(int(field_pair.time_before_utc_ns))
            if wave_field_resolver is not None
            else None
        )
        wave2 = (
            wave_field_resolver(int(field_pair.time_after_utc_ns))
            if wave_field_resolver is not None
            else None
        )
        hour_overhead.time_wave_resolve_s += _time.perf_counter() - t0

        # 任務 3 效能優化：乾濕判定＋Stokes 全網格計算（compose_current_and_
        # stokes_fields）只跟這一小時的流場與網格有關，與這一批粒子的沉降
        # 速度無關；composed_base_cache 非 None 時，讓「同一到達時刻、同一
        # 區域、不同沉降速度」的多次 run_backward_batch 呼叫共用同一份結果，
        # 只在每個批次各自做便宜的 apply_settling_velocity（見該參數與
        # HourlyOverheadDiagnostics 的 docstring、模組文件「任務 3」一節的
        # 量化結果）。composed_base_cache 為 None 時（預設）行為與拆分前的
        # compose_total_velocity_fields 完全相同，只是內部呼叫兩段式函式。
        cache_key = (int(field_pair.time_before_utc_ns), int(field_pair.time_after_utc_ns))
        base = composed_base_cache.get(cache_key) if composed_base_cache is not None else None
        if base is not None:
            hour_overhead.compose_base_cache_hits += 1
        else:
            t0 = _time.perf_counter()
            base = compose_current_and_stokes_fields(
                driver_grid,
                handles,
                field_pair,
                wave_field1=wave1,
                wave_field2=wave2,
                stokes_depth_threshold_ratio=config.stokes_depth_threshold_ratio,
            )
            hour_overhead.time_compose_base_s += _time.perf_counter() - t0
            hour_overhead.compose_base_cache_misses += 1
            hour_overhead.stokes_points_evaluated += base.stokes_points_evaluated
            hour_overhead.stokes_points_possible += base.stokes_points_possible
            if composed_base_cache is not None:
                composed_base_cache[cache_key] = base

        t0 = _time.perf_counter()
        composed = apply_settling_velocity(base, config.settling_velocity_mps, kbp=driver_grid.grid.kbp)
        hour_overhead.time_apply_settling_s += _time.perf_counter() - t0

        if not primed:
            # 用合成後的總速度場重新內插一次起點速度，取代
            # ``locate_initial_elements`` 內部只用純海流、單層（無垂向內插）
            # 的初始快取，避免第一個子步的 k1 漏掉 Stokes／沉降貢獻（見
            # ``locate_initial_elements`` docstring）。
            active_mask0 = pa.status == 1
            if np.any(active_mask0):
                handles.batch_interpolate_velocity(
                    pa.x, pa.y, pa.z, pa.element, pa.status,
                    driver_grid.schism_grid.i34, driver_grid.schism_grid.elnode,
                    driver_grid.schism_grid.x, driver_grid.schism_grid.y,
                    driver_grid.schism_grid.nvrt, driver_grid.schism_grid.kbp, driver_grid.schism_grid.kbe,
                    driver_grid.schism_grid.sigma_lcl, driver_grid.schism_grid.dp,
                    composed.u_field1, composed.v_field1, composed.w_field1, composed.eta_field1,
                    composed.u_field2, composed.v_field2, composed.w_field2,
                    0.0, driver_grid.h0,
                    pa.u, pa.v, pa.w,
                )
            primed = True

        hf1_now = hf2_now = None
        if config.enable_diffusion:
            # 同樣可跨沉降批次共用（見 composed_base_cache 的說明）：Smagorinsky
            # 只讀純海流 field2 與網格，鍵用 time_after_utc_ns（對應
            # compute_smagorinsky_kh_field 實際讀取的時刻，見該函式 docstring）。
            smagorinsky_key = int(field_pair.time_after_utc_ns)
            cached_hf2 = smagorinsky_cache.get(smagorinsky_key) if smagorinsky_cache is not None else None
            if cached_hf2 is not None:
                hf2_now = cached_hf2
                hour_overhead.smagorinsky_cache_hits += 1
            else:
                t0 = _time.perf_counter()
                hf2_now = compute_smagorinsky_kh_field(
                    driver_grid, handles,
                    u_current_mps=field_pair.u_field2, v_current_mps=field_pair.v_field2,
                    horcon=config.horcon,
                )
                hour_overhead.time_smagorinsky_s += _time.perf_counter() - t0
                hour_overhead.smagorinsky_cache_misses += 1
                if smagorinsky_cache is not None:
                    smagorinsky_cache[smagorinsky_key] = hf2_now
            hf1_now = prev_hf2 if prev_hf2 is not None else hf2_now
            prev_hf2 = hf2_now

        t0 = _time.perf_counter()
        ndeltp = estimate_substep_count(
            driver_grid, composed.u_field1, composed.v_field1,
            dt_seconds=3600.0, dt_max_seconds=config.dt_max_seconds, cfl_number=config.cfl_number,
        )
        hour_overhead.time_substep_count_s += _time.perf_counter() - t0
        dtb = 3600.0 / ndeltp
        hour_start_utc = current_utc
        hour_end_utc = hour_start_utc - _HOUR_NS

        def _substep_utc_ns(idt: int, _start: int = hour_start_utc, _dtb: float = dtb) -> int:
            return _start - int(round(idt * _dtb * 1.0e9))

        diag = mirror_rk4_substep_loop(
            driver_grid,
            handles,
            pa,
            ibf=-1,
            ndeltp=ndeltp,
            dtb=dtb,
            composed=composed,
            enable_diffusion=config.enable_diffusion,
            hf1_field=hf1_now,
            hf2_field=hf2_now,
            particle_seed_lo=particle_seed_lo,
            particle_seed_hi=particle_seed_hi,
            particle_step_counter=particle_step_counter,
            native_status=native_status,
            substep_utc_ns=_substep_utc_ns,
            interest_region_polygon=interest_region_polygon,
            departures=departures,
        )
        total_diag.add(diag)
        total_overhead.add(hour_overhead)

        active = native_status == NATIVE_STATUS_ACTIVE
        bad = active & ~(np.isfinite(pa.x) & np.isfinite(pa.y) & np.isfinite(pa.z))
        if np.any(bad):
            native_status[bad] = NATIVE_STATUS_NUMERICAL_FAILURE
            pa.status[bad] = 0
            for row in np.flatnonzero(bad):
                departures.append(
                    DepartureRecord(
                        int(row), NATIVE_STATUS_NUMERICAL_FAILURE,
                        float(pa.x[row]), float(pa.y[row]), float(pa.z[row]), int(hour_end_utc),
                    )
                )

        active = native_status == NATIVE_STATUS_ACTIVE
        if np.any(active):
            elem = pa.element
            valid_elem = elem >= 0
            elem_dry_lookup = np.zeros(n, dtype=bool)
            elem_dry_lookup[valid_elem] = composed.elem_is_dry[elem[valid_elem]]
            stranded = active & valid_elem & elem_dry_lookup
            if np.any(stranded):
                native_status[stranded] = NATIVE_STATUS_STRANDED
                pa.status[stranded] = 0
                for row in np.flatnonzero(stranded):
                    departures.append(
                        DepartureRecord(
                            int(row), NATIVE_STATUS_STRANDED,
                            float(pa.x[row]), float(pa.y[row]), float(pa.z[row]), int(hour_end_utc),
                        )
                    )

        if config.settling_velocity_mps < 0.0:
            active = native_status == NATIVE_STATUS_ACTIVE
            if np.any(active):
                bed_z = particle_bed_z_levels(driver_grid, field_pair.eta_field1, pa.element)
                at_bed = active & np.isfinite(bed_z) & ((pa.z - bed_z) <= config.deposition_tolerance_m)
                deposit_streak[at_bed] += 1
                deposit_streak[active & ~at_bed] = 0
                newly_deposited = active & (deposit_streak >= config.deposition_consecutive_hours)
                if np.any(newly_deposited):
                    native_status[newly_deposited] = NATIVE_STATUS_DEPOSITED
                    pa.status[newly_deposited] = 0
                    for row in np.flatnonzero(newly_deposited):
                        departures.append(
                            DepartureRecord(
                                int(row), NATIVE_STATUS_DEPOSITED,
                                float(pa.x[row]), float(pa.y[row]), float(pa.z[row]), int(hour_end_utc),
                            )
                        )

        current_utc = hour_end_utc
        hours_run += 1

        if current_utc <= next_output_utc:
            trajectory_utc.append(current_utc)
            trajectory_x.append(pa.x.copy())
            trajectory_y.append(pa.y.copy())
            trajectory_z.append(pa.z.copy())
            trajectory_status.append(native_status.copy())
            next_output_utc -= output_interval_ns

    departure_by_row: dict[int, DepartureRecord] = {}
    for record in departures:
        departure_by_row[record.row_particle_index] = record
    final_utc_ns = np.full(n, current_utc, dtype=np.int64)
    for row, record in departure_by_row.items():
        final_utc_ns[row] = record.utc_ns

    return BackwardDriverResult(
        row_particle_index=row_particle_index,
        final_status=native_status.copy(),
        final_x_m=pa.x.copy(),
        final_y_m=pa.y.copy(),
        final_z_m=pa.z.copy(),
        final_utc_ns=final_utc_ns,
        trajectory_utc_ns=np.asarray(trajectory_utc, dtype=np.int64),
        trajectory_x_m=np.stack(trajectory_x, axis=0),
        trajectory_y_m=np.stack(trajectory_y, axis=0),
        trajectory_z_m=np.stack(trajectory_z, axis=0),
        trajectory_status=np.stack(trajectory_status, axis=0),
        departures=tuple(departures),
        diagnostics=total_diag,
        overhead_diagnostics=total_overhead,
        hours_run=hours_run,
    )
