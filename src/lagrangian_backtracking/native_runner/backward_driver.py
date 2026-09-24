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

WP-26／WP-28：``bed_no_deposition`` 海床不沉積敏感度案例（正名、誠實範圍揭露）
--------------------------------------------------------------------------------
``BackwardDriverConfig.bed_no_deposition``（預設 ``False``，見該欄位 docstring；
WP-26 引入時暫名 ``bed_reflect``，WP-28 依 ``reports/WP-26.md`` 第 6 節「沒把握
或沒做完的部分」第 2 點的誠實揭露正名，見下方「正名說明」）只關閉「連續停留
海床即終止」這一個判定，讓粒子留在 ``NATIVE_STATUS_ACTIVE`` 繼續交給**完全
未修改**的 ``mirror_rk4_substep_loop``／``batch_clamp_z`` 驅動——**這不是在
海床做一次真正的鏡射速度／位置反射，也不是任何形式的再懸浮動力學**（沉積物
被水流或紊流重新捲起的物理過程）：本模組完全沒有實作、也沒有嘗試近似後者的
任何運動學或動力學公式，那需要改寫原版 ``batch_clamp_z``／子步 kernel 本身
（本模組刻意不修改原版物理核心，見本檔模組說明最上方「模組結構」原則）。
這個敏感度案例唯一做的事，就是它現在的名字所講的：**粒子接觸海床時不視為
沉積、不終止**，由既有物理（平流＋Smagorinsky 擴散＋既有 ``batch_clamp_z``
垂向夾擠）繼續驅動，量測「若不假設沉積物永久停留海床」這個基線假設讓來源
足跡偏近多少。native 路徑每個子步本來就會把 z 夾限在 ``[bed_z, eta]`` 之間
（``batch_clamp_z``），粒子在數值上本來就無法穿透海床，這只是恰好造成一個
可觀察的副作用（粒子不會憑空「掉出」海床以下），但這個副作用是既有夾擠
物理的結果，不是本模組刻意設計的「反射」機制，因此不應該用「反射」這個字
描述它，以免讓審查者誤以為這裡實作了海床鏡射速度反轉或沉積物再懸浮的
物理／統計模型。

**正名說明（WP-28）**：``bed_reflect`` 這個舊名字暗示「鏡射反射」，但實作
行為从來就是「關掉沉積終止判定」，名不副實、容易誤導審查（見
``reports/WP-26.md`` 對這個落差的誠實記錄）。WP-28 把 ``BackwardDriverConfig.
bed_reflect`` 改名為 ``bed_no_deposition``、``BedReflectEventRecord`` 改名為
``BedNoDepositionEventRecord``、``BackwardDriverResult.bed_reflect_events``
改名為 ``bed_no_deposition_events``；**只改名稱與文件敘述，不改行為**——預設
``bed_no_deposition=False`` 時的全部行為與改名前的 ``bed_reflect=False`` 逐位元
相同（見 ``tests/test_native_backward_driver.py`` 的位元相同回歸測試）。

**已知已完成的整合工作（WP-28 補完，取代 WP-26 report 第 6 節第 1 點的「未完成」
狀態）**：``bed_no_deposition`` 現在已經接上 ``native-plan``／``native-run`` CLI
（``--bed-no-deposition`` 旗標／``LBT_NATIVE_BED_NO_DEPOSITION`` 環境變數，見
``physics_case.py`` 同款「CLI 優先於環境變數、不一致 fail closed」介面）、
``hour_major.HourMajorConfig``／``hour_major_provider.py`` 的參數傳遞、
``postprocess.py`` 結果 manifest（見該模組 ``PostprocessManifest.config`` 的
``runtime_bed_no_deposition`` 欄位，由 ``native-postprocess --plan`` 讀回
``native-plan`` 計畫檔的 ``bed_no_deposition`` 區塊寫入），以及輸出目錄命名
慣例（見 ``docs/operations/19_native_backward_batch_orchestrator.md`` 9.15 節
與 CLI 說明：兩種案例的結果必須放在不同的 ``--output`` 目錄，manifest 記錄
本次是否啟用這個案例，供人工核對兩批結果沒有被混在同一份統計）。
``particle_input.py`` 的批次身分雜湊**刻意沒有**把 ``bed_no_deposition`` 納入
（見該模組本節新增說明）：粒子初始位置與 ``experiment_case_id``（是否含
Stokes）決定的物理設定無關於「海床接觸後要不要終止」這個回溯動力學層級的
選擇，兩個敏感度案例本來就應該共用同一份粒子部署（相同起點、相同種子）才能
做公平比較，把 ``bed_no_deposition`` 混進粒子身分雜湊反而會強迫兩案例各自
重建一份不必要的粒子表。
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import threading
import types
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, replace
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
    finite_depth_stokes_from_wave_number_batch,
    solve_wave_number_batch,
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


# ══════════════════════════════════════════════════════════════════════════
# WP-14 任務 A：每節點只求一次波數，套用到各層時改呼叫「給定波數」的批次核心
# ══════════════════════════════════════════════════════════════════════════
#
# 見 ``stokes_batch.py`` 模組層級「WP-14 任務 A」說明與
# ``compose_current_and_stokes_fields`` docstring「wave_number 快取」一節的
# 完整推導。這裡只負責把 WP-13 既有的「序列或行程池」分派邏輯，從呼叫
# ``batch_finite_depth_stokes``（每個點都重新求解色散關係）改成呼叫
# ``finite_depth_stokes_from_wave_number_batch``（套用已經在節點層級算好的
# 波數，不重新求解），行程池切段與寫回索引的邏輯與 WP-13 完全相同、原樣
# 保留，只是換了要平行呼叫的目標函式。


def _stokes_chunk_task_given_wave_number(
    node_wave_number: np.ndarray,
    node_status: np.ndarray,
    significant_wave_height_m: np.ndarray,
    peak_frequency_hz: np.ndarray,
    direction_raw_deg: np.ndarray,
    particle_z_m: np.ndarray,
    water_depth_m: np.ndarray,
    surface_tolerance_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``_stokes_chunk_task`` 的「給定波數」版本，呼叫未經修改的
    ``finite_depth_stokes_from_wave_number_batch``。模組層級函式（非閉包），
    理由同 ``_stokes_chunk_task``：``ProcessPoolExecutor`` 的 ``spawn``
    啟動方式要求任務函式可以被 pickle 匯入。
    """

    n = significant_wave_height_m.shape[0]
    stokes_u = np.empty(n, dtype=np.float64)
    stokes_v = np.empty(n, dtype=np.float64)
    wave_number = np.empty(n, dtype=np.float64)
    status = np.empty(n, dtype=np.int32)
    finite_depth_stokes_from_wave_number_batch(
        node_wave_number,
        node_status,
        significant_wave_height_m,
        peak_frequency_hz,
        direction_raw_deg,
        particle_z_m,
        water_depth_m,
        stokes_u,
        stokes_v,
        wave_number,
        status,
        surface_tolerance_m,
    )
    return stokes_u, stokes_v, wave_number, status


def _evaluate_finite_depth_stokes_given_wave_number(
    node_wave_number: np.ndarray,
    node_status: np.ndarray,
    significant_wave_height_m: np.ndarray,
    peak_frequency_hz: np.ndarray,
    direction_raw_deg: np.ndarray,
    particle_z_m: np.ndarray,
    water_depth_m: np.ndarray,
    *,
    surface_tolerance_m: float,
    stokes_pool: Any | None,
    stokes_worker_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``_evaluate_batch_finite_depth_stokes`` 的「給定波數」版本：分派點
    完全相同（序列或依 ``stokes_worker_count`` 切段的行程池），差別只在
    呼叫 ``finite_depth_stokes_from_wave_number_batch`` 而非
    ``batch_finite_depth_stokes``——因為波數已經在節點層級算好、不再需要
    每個點各自重新求解色散關係，這一步比 WP-13 原本的版本便宜得多（見
    最終報告的量測），但切段平行邏輯仍保留給呼叫端在需要時使用。
    """

    n = significant_wave_height_m.shape[0]
    if stokes_pool is None or stokes_worker_count <= 1 or n < stokes_worker_count:
        stokes_u = np.empty(n, dtype=np.float64)
        stokes_v = np.empty(n, dtype=np.float64)
        wave_number = np.empty(n, dtype=np.float64)
        status = np.empty(n, dtype=np.int32)
        finite_depth_stokes_from_wave_number_batch(
            node_wave_number,
            node_status,
            significant_wave_height_m,
            peak_frequency_hz,
            direction_raw_deg,
            particle_z_m,
            water_depth_m,
            stokes_u,
            stokes_v,
            wave_number,
            status,
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
            _stokes_chunk_task_given_wave_number,
            np.ascontiguousarray(node_wave_number[start:stop]),
            np.ascontiguousarray(node_status[start:stop]),
            np.ascontiguousarray(significant_wave_height_m[start:stop]),
            np.ascontiguousarray(peak_frequency_hz[start:stop]),
            np.ascontiguousarray(direction_raw_deg[start:stop]),
            np.ascontiguousarray(particle_z_m[start:stop]),
            np.ascontiguousarray(water_depth_m[start:stop]),
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
    "NATIVE_STATUS_LEFT_FLOW_DOMAIN",
    "NATIVE_STATUS_LEFT_INTEREST_REGION",
    "NATIVE_STATUS_MAX_BACKTRACK",
    "NATIVE_STATUS_STRANDED",
    "NATIVE_STATUS_DEPOSITED",
    "NATIVE_STATUS_FORCING_MISSING",
    "NATIVE_STATUS_NUMERICAL_FAILURE",
    "NATIVE_STATUS_PRE_WINDOW_DEPOSITION",
    "NATIVE_STATUS_DESCRIPTIONS",
    "BayTraceModuleHandles",
    "load_baytrace_modules",
    "NativeBackwardDriverGrid",
    "build_native_backward_driver_grid",
    "WaveFieldAtNodes",
    "ComposedFieldPair",
    "ComposedCurrentAndStokesFields",
    "ComposedFieldEndpoint",
    "compose_current_and_stokes_fields",
    "apply_settling_velocity",
    "compose_total_velocity_fields",
    "compute_smagorinsky_kh_field",
    "zeroed_current_field2_for_smagorinsky",
    "compute_wet_dry_flags",
    "estimate_substep_count",
    "estimate_substep_count_local_cfl",
    "LocalCflSpatialIndex",
    "build_local_cfl_spatial_index",
    "particle_bed_z_levels",
    "derive_batch_diffusion_seed",
    "derive_particle_diffusion_seed_lo_hi_u64",
    "counter_based_gaussian_pairs",
    "SubstepDiagnostics",
    "HourlyOverheadDiagnostics",
    "DepartureRecord",
    "LocalDomainFirstExitRecord",
    "BedNoDepositionEventRecord",
    "mirror_rk4_substep_loop",
    "BackwardDriverConfig",
    "BackwardDriverResult",
    "build_pre_window_deposition_result",
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

NATIVE_STATUS_LEFT_FLOW_DOMAIN = 1
"""已離開流場區域（``flow_domain``）開放邊界，逆向追蹤在此正常結束。

WP-23 修正：此狀態碼過去誤把「離開站點 local domain（``own_local_domain``）」
當成終止條件（貢寮／龜山島因此在 20 km local domain 邊界就被錯誤截斷，而不是在
共用的 A 區 flow domain 開放邊界）。裁決後（見
``docs/foundation/08_design_baseline_and_derived_gates.md`` 第 8 節與
``docs/development/05_decisions_and_risks.md`` D008）：終止一律以 ``flow_domain``
開放邊界為準；「離開 ``own_local_domain``」改成**非終止**事件，見
``LocalDomainFirstExitRecord`` 與 ``BackwardDriverResult.local_domain_departures``。
對新竹／南灣／連江三站（``local_equals_flow=True``），local domain 與 flow domain
是同一個多邊形，離開的當下同時觸發本狀態碼**並**在該筆 ``DepartureRecord`` 標記
``also_local_domain_first_exit=True``，避免重複計數（詳見模組文件「WP-23：
local domain 與 flow domain 的兩階段終止語意」一節）。"""

NATIVE_STATUS_LEFT_INTEREST_REGION = NATIVE_STATUS_LEFT_FLOW_DOMAIN
"""``NATIVE_STATUS_LEFT_FLOW_DOMAIN`` 的舊名別名，避免打破既有匯入；新程式碼請改用
新名稱，語意見該常數的 docstring（WP-23 之前這裡的語意其實是「離開 own_local_domain」，
WP-23 之後語意已改為「離開 flow_domain」——舊名保留但語意已變更，這正是 WP-23 要
修正的問題本身，故不建議新程式碼繼續使用舊名）。"""

NATIVE_STATUS_MAX_BACKTRACK = 2
"""回溯超過設定上限（預設 30 天）仍未離開流場區域，強制停止。"""

NATIVE_STATUS_STRANDED = 3
"""擱淺上岸：粒子目前所在元素被 BayTrace 乾濕判定標記為乾。"""

NATIVE_STATUS_DEPOSITED = 4
"""沉底：具下沉行為（沉降速度為負）的粒子連續停留在海床附近。"""

NATIVE_STATUS_FORCING_MISSING = 5
"""流場缺值：本小時的濕節點上出現未經解釋的缺值，整批粒子在此小時停止。"""

NATIVE_STATUS_NUMERICAL_FAILURE = 6
"""數值失敗：粒子位置在某一步之後出現非有限值。"""

NATIVE_STATUS_PRE_WINDOW_DEPOSITION = 7
"""固定日曆回溯模式（``fixed_calendar_window``）專屬：沉底時刻早於「觀測日往前 H 天」的研究窗，
有效回溯期間（``H - bed_residence_age_hours``）不為正，因此這顆粒子完全不進入 forcing 運算
（不定位初始元素、不查詢任何流場、不消耗任何子步）。這不是資料缺口，也不是數值失敗——粒子的
沉底時刻本身就落在本次固定日曆窗之外，見 ``bed_residence.resolve_bed_residence_timing`` 與
``docs/operations/14_input_derivation_and_release_contract.md`` 第 3.1 節「pre-window 成員」的
定義。這類粒子從有效來源比例的分母排除，但仍保留在固定的情境／成員 outcome 總分母中單獨計數
（見 ``native_runner.postprocess`` 對這個狀態碼的分母處理）。"""

NATIVE_STATUS_DESCRIPTIONS: dict[int, str] = {
    NATIVE_STATUS_ACTIVE: "仍在追蹤中，尚未終止。",
    NATIVE_STATUS_LEFT_FLOW_DOMAIN: "已離開流場區域（flow_domain）開放邊界，正常結束。",
    NATIVE_STATUS_MAX_BACKTRACK: "回溯已達設定上限，粒子仍在流場區域內，強制停止。",
    NATIVE_STATUS_STRANDED: "擱淺上岸：所在元素被判定為乾。",
    NATIVE_STATUS_DEPOSITED: "沉底：下沉行為粒子持續停留在海床附近。",
    NATIVE_STATUS_FORCING_MISSING: "流場缺值：本小時濕節點出現未登錄缺值，該批粒子在此停止。",
    NATIVE_STATUS_NUMERICAL_FAILURE: "數值失敗：粒子位置出現非有限值。",
    NATIVE_STATUS_PRE_WINDOW_DEPOSITION: (
        "固定日曆回溯窗之外的沉底（pre_window_deposition）：沉底時刻早於觀測日往前 H 天，"
        "未進入運算。"
    ),
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
    use_numba: bool = False,
    smagorinsky_prepared_grid: Any | None = None,
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
        use_numba: WP-14 任務 C 新增。``False``（預設）：完全不變，呼叫
            BayTrace 原版 ``_compute_smagorinsky_hf2``（本函式新增這個參數
            前的唯一行為）。``True``：改呼叫
            ``smagorinsky_batch.compute_smagorinsky_kh_field_numba``，這個
            獨立 Numba 移植版本**已知與原版不是逐位元相同**（實測相對誤差
            落在 ``3e-16``～``5e-16``，即 1-2 個 ulp，來自 NumPy 內積與本
            模組手寫序列求和的浮點加法結合律差異，不是公式錯誤，見
            ``smagorinsky_batch.py`` 模組 docstring「位元相同性」一節與
            ``tests/test_native_smagorinsky_batch.py`` 的量測），因此**預設
            關閉**、不接線到任何正式呼叫路徑（``hour_major.py``／
            ``run_backward_batch`` 都不會傳這個參數，恆為 ``False``）；只
            提供給需要自行評估這個 ulp 級別誤差是否可接受的呼叫端明確
            選用，選用後果由呼叫端自行承擔。
        smagorinsky_prepared_grid: WP-16 任務 3 新增，只在 ``use_numba=True``
            時有意義。``None``（預設）：完全不變，每次呼叫都對網格陣列做
            一次 ``ascontiguousarray`` 轉換＋反距離權重計算（見
            ``smagorinsky_batch.compute_smagorinsky_kh_field_numba`` 的
            ``prepared`` 參數說明）。提供
            ``smagorinsky_batch.SmagorinskyPreparedGrid``（呼叫端用
            ``smagorinsky_batch.prepare_smagorinsky_grid`` 對同一個
            ``driver_grid`` 算一次、跨全部小時重複傳入）時跳過這些只跟
            網格拓樸有關、與流場無關的重複計算——``hour_major.py`` 的
            小時主軸迴圈接了這個參數（見該模組），``run_backward_batch``
            （逐單元路徑）目前未接線，仍是 ``None``（誠實記錄於最終
            報告：逐單元路徑不是量測重點，見任務書量測情境）。

    Returns:
        ``(nvrt, nnode)`` 的水平擴散係數場（``hf2_field``），單位 m²/s。
    """

    if use_numba:
        from .smagorinsky_batch import SmagorinskyGridArrays, compute_smagorinsky_kh_field_numba

        g = driver_grid.schism_grid
        grid_arrays = SmagorinskyGridArrays(
            elnode=g.elnode, area=g.area, dldxy=g.dldxy, nne=g.nne, indel=g.indel,
            x=g.x, y=g.y, xctr=g.xctr, yctr=g.yctr,
        )
        return compute_smagorinsky_kh_field_numba(
            grid_arrays,
            u_field2=np.ascontiguousarray(u_current_mps, dtype=np.float64),
            v_field2=np.ascontiguousarray(v_current_mps, dtype=np.float64),
            horcon=horcon,
            prepared=smagorinsky_prepared_grid,
        )

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

    ``field2_endpoint``（WP-14 任務 B）
    -------------------------------------
    這一小時 ``field2`` 端點的合成結果快取單元（``ComposedFieldEndpoint``），
    供呼叫端保留給**下一小時**當 ``field1_reuse_candidate`` 候選使用，見
    ``compose_current_and_stokes_fields`` 該參數 docstring 的完整推導。
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
    field2_endpoint: ComposedFieldEndpoint | None = None


@dataclass(frozen=True, slots=True)
class ComposedFieldEndpoint:
    """WP-14 任務 B：單一流場時刻「端點」的合成結果快取單元。

    背景與動機
    ----------
    小時主軸迴圈（``hour_major.run_flow_domain_hour_major``）逐小時遞減
    UTC 呼叫 ``compose_current_and_stokes_fields``，每次都同時合成該小時的
    兩個端點（``field_pair.time_before_utc_ns``／``time_after_utc_ns``，即
    ``field1``／``field2``）。但相鄰兩個小時的窗口是相連的：第 h 小時的
    ``field2``（較早的端點）與第 h+1 小時的 ``field1``（較晚的端點）通常是
    **同一個絕對時刻的同一份流場資料**（``field_pair.time_after_utc_ns`` ==
    下一小時的 ``time_before_utc_ns``，且來自同一個月份／同一種資料來源，
    見 ``forcing_bridge.HourlyFieldPair`` 的 ``month_id``／``time_origin``
    欄位），因此把第 h 小時已經算好的 ``field2``（含乾濕歸零、含 Stokes）
    快取起來，下一小時直接當 ``field1`` 重用，理論上可以省下一半的乾濕＋
    Stokes 合成成本。

    這裡的欄位就是「重用一個端點所需的最小資訊」：不是重新暴露
    ``ComposedCurrentAndStokesFields`` 全部欄位，而是只保留「已經套用某個
    ``node_is_dry`` 遮罩、已經加總 Stokes 貢獻」的 ``u/v/w`` 陣列，以及
    **那個遮罩本身**（``node_is_dry``）——重用是否安全，完全取決於下一
    小時自己重新算出的 ``node_is_dry`` 是否與這裡記錄的逐位元相同，見
    ``compose_current_and_stokes_fields`` 參數 ``field1_reuse_candidate``
    docstring「為什麼需要比對 node_is_dry」一節的完整推導。呼叫端
    （``hour_major.py``）另外自行記錄 UTC 時刻／``month_id``／
    ``time_origin``／波浪時刻等「資料來源」身分資訊，決定要不要把這個
    快取單元當候選傳進下一小時；本類別本身不含這些身分欄位，因為
    ``compose_current_and_stokes_fields`` 不知道、也不需要知道呼叫端的
    快取鍵設計。
    """

    node_is_dry: np.ndarray
    u: np.ndarray
    v: np.ndarray
    w_no_settling: np.ndarray
    stokes_ok_count: int
    stokes_total_count: int
    stokes_points_evaluated: int
    stokes_points_possible: int
    # 主代理補強：產生這個端點的「原始輸入陣列」（海流 u/v/w、水位 eta、以及有波浪時的
    # 波高／尖峰頻率／波向／有效遮罩／品質旗標）。重用前逐一以 ``np.array_equal`` 比對下一小時
    # field1 的原始輸入，確保「輸入完全相同＋乾濕遮罩完全相同」才重用；合成是確定性函式，
    # 兩者皆相同即保證輸出逐位元相同，不再只依賴呼叫端的時刻／來源身分預過濾。
    # 預設空元組代表「沒有記錄來源」，此時一律不重用（fail-closed）。
    source_arrays: tuple = ()


def _endpoint_source_arrays(
    u: np.ndarray, v: np.ndarray, w: np.ndarray, eta: np.ndarray, wave: Any | None
) -> tuple:
    """收集單一端點合成所需的全部原始輸入陣列，供端點重用時做逐位元相同比對。"""

    arrays = [u, v, w, eta]
    if wave is not None:
        arrays.extend(
            [
                wave.significant_wave_height_m,
                wave.peak_frequency_hz,
                wave.direction_raw_deg,
                wave.valid_mask_wave,
                wave.qc_flags,
            ]
        )
    return tuple(arrays)


def _same_source_arrays(recorded: tuple, current: tuple) -> bool:
    """兩組原始輸入陣列是否數量相同且逐一形狀、內容完全相同（NaN 位置也須相同）。"""

    if not recorded or len(recorded) != len(current):
        return False
    for a, b in zip(recorded, current, strict=True):
        a = np.asarray(a)
        b = np.asarray(b)
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        if a.dtype.kind == "f":
            if not np.array_equal(a, b, equal_nan=True):
                return False
        elif not np.array_equal(a, b):
            return False
    return True


def zeroed_current_field2_for_smagorinsky(
    field_pair: HourlyFieldPair, *, kbp: np.ndarray, node_is_dry: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """回傳乾節點與海床以下層已歸零的純海流 field2（u, v），供 Smagorinsky 使用。

    BayTrace 原版在 ``_compute_smagorinsky_hf2`` 之前一定先呼叫
    ``_zero_dry_velocities``（乾節點與 ``k<kbp`` 層歸零），因此原版的應變率
    從不讀到海床以下的數值。真實 ``ocm_native`` 產品在海床以下層存 ``NaN``，
    若直接把原始 ``field_pair.u_field2`` 傳給 Smagorinsky，相鄰節點海床深度
    不同的元素在該層會算出 ``NaN`` 擴散係數；這裡以複本歸零（不改動
    ``field_pair``），與 ``compose_current_and_stokes_fields`` 對 field2 的
    歸零規則、乾濕遮罩完全一致。
    """

    u = np.array(field_pair.u_field2, dtype=np.float64, copy=True)
    v = np.array(field_pair.v_field2, dtype=np.float64, copy=True)
    w = np.array(field_pair.w_field2, dtype=np.float64, copy=True)
    _zero_dry_and_below_kbp(u, v, w, kbp=kbp, node_is_dry=node_is_dry)
    return u, v


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


def _compose_single_endpoint(
    grid: NativeGridArrays,
    u_field: np.ndarray,
    v_field: np.ndarray,
    w_field: np.ndarray,
    *,
    eta: np.ndarray,
    wave: WaveFieldAtNodes | None,
    kbp: np.ndarray,
    dp: np.ndarray,
    effective_mask: np.ndarray,
    nvrt: int,
    nnode: int,
    gravity_mps2: float,
    surface_tolerance_m: float,
    stokes_depth_threshold_ratio: float | None,
    stokes_pool: Any | None,
    stokes_worker_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int, int]:
    """單一流場端點（field1 或 field2 其中之一）的「視情況加 Stokes」計算。

    ``compose_current_and_stokes_fields`` 對 field1／field2 原本是同一段
    程式碼跑兩次（一個 for 迴圈），WP-14 任務 B 需要讓 field1 可以視情況
    整段跳過（改用快取），field2 則永遠要重新計算，因此把這段邏輯抽成
    可以各自獨立呼叫一次的輔助函式；運算式與執行順序完全比照 WP-14 任務 A
    修改後的版本（每節點只求一次波數、套用 ``stokes_depth_threshold_ratio``
    深度截斷），只是把原本「同一個 for 迴圈跑兩次」的寫法改成「呼叫端呼叫
    兩次」，不是重新設計計算內容，逐位元相同由
    ``tests/test_native_backward_driver.py`` 交叉驗證。

    Args:
        u_field, v_field, w_field: 呼叫端已經複製、且已經套用過
            ``_zero_dry_and_below_kbp`` 的速度場；本函式會原地疊加 Stokes
            貢獻，回傳的就是同一個陣列物件（呼叫端不需要另外接手）。
        eta: 這個端點自己的海面高程（對應 ``field_pair.eta_field1`` 或
            ``eta_field2``），只用來計算 Stokes 垂向結構所需的相對深度
            ``z_rel = z_levels(eta) - eta``；**不**用於乾濕判定——乾濕
            判定的遮罩（``effective_mask``）由呼叫端傳入，完全由「這一
            小時」自己的 ``eta_field1`` 決定，兩個端點共用同一份，見
            ``compose_current_and_stokes_fields`` docstring
            ``field1_reuse_candidate`` 一節「為什麼需要比對 node_is_dry」
            的完整推導。
        wave: 這個端點的波浪場；``None`` 代表不計算 Stokes（呼叫端已經把
            「wave_field1／wave_field2 是否同時提供」的檢查做完，這裡只
            單純依 ``None``/非 ``None`` 決定要不要算）。

    Returns:
        ``(u_field, v_field, w_field, stokes_ok_count, stokes_total_count,
        stokes_points_evaluated, stokes_points_possible)``——後四個是這個
        單一端點的統計，呼叫端把 field1／field2 兩次呼叫的結果相加即為
        ``ComposedCurrentAndStokesFields`` 最終回報的欄位。
    """

    stokes_ok_count = 0
    stokes_total_count = 0
    stokes_points_evaluated = 0
    stokes_points_possible = 0

    if wave is None:
        return (
            u_field, v_field, w_field,
            stokes_ok_count, stokes_total_count, stokes_points_evaluated, stokes_points_possible,
        )

    z_rel = _vectorized_z_levels(grid.sigma_lcl, kbp, dp, eta) - eta[None, :]

    depth_grid = np.broadcast_to(dp[None, :], (nvrt, nnode))
    hs_grid = np.broadcast_to(wave.significant_wave_height_m[None, :], (nvrt, nnode))
    fp_grid = np.broadcast_to(wave.peak_frequency_hz[None, :], (nvrt, nnode))
    dir_grid = np.broadcast_to(wave.direction_raw_deg[None, :], (nvrt, nnode))

    # WP-14 任務 A：每個節點的 fp／h／Hs／方向／valid_mask_wave／qc_flags
    # 在 nvrt 個垂向層之間完全相同（上面都是 np.broadcast_to 展開），色散
    # 關係求根本身也完全不依賴 z，因此對每個節點只求一次波數，而不是對每個
    # (層,節點) 組合都各自重新求解——這一步同時取代了 WP-9 原本為了
    # stokes_depth_threshold_ratio 而額外做的「z=0 表層呼叫」，兩者是同一個
    # 計算（z=0 的物理有效性判斷與節點級判斷完全等價，見
    # ``stokes_batch.solve_wave_number_batch`` docstring），不需要再算兩次。
    node_wave_number = np.empty(nnode, dtype=np.float64)
    node_status = np.empty(nnode, dtype=np.int32)
    solve_wave_number_batch(
        np.ascontiguousarray(wave.significant_wave_height_m, dtype=np.float64),
        np.ascontiguousarray(wave.peak_frequency_hz, dtype=np.float64),
        np.ascontiguousarray(wave.direction_raw_deg, dtype=np.float64),
        np.ascontiguousarray(dp, dtype=np.float64),
        np.ascontiguousarray(wave.valid_mask_wave, dtype=bool),
        np.ascontiguousarray(wave.qc_flags, dtype=np.int64),
        node_wave_number,
        node_status,
        float(gravity_mps2),
    )
    node_wave_number_grid = np.broadcast_to(node_wave_number[None, :], (nvrt, nnode))
    node_status_grid = np.broadcast_to(node_status[None, :], (nvrt, nnode))

    if stokes_depth_threshold_ratio is None:
        stokes_mask = effective_mask
    else:
        k_known = (
            (node_status == STOKES_BATCH_STATUS_OK)
            & np.isfinite(node_wave_number)
            & (node_wave_number > 0.0)
        )
        cutoff_z_node = np.where(
            k_known,
            np.log(stokes_depth_threshold_ratio) / (2.0 * np.where(k_known, node_wave_number, 1.0)),
            -np.inf,
        )
        cutoff_z_grid = np.broadcast_to(cutoff_z_node[None, :], (nvrt, nnode))
        stokes_mask = effective_mask & (z_rel >= cutoff_z_grid)

    n_kept = int(np.count_nonzero(stokes_mask))
    stokes_points_evaluated = n_kept
    stokes_points_possible = int(np.count_nonzero(effective_mask))
    stokes_total_count = int(np.count_nonzero(effective_mask))
    if n_kept > 0:
        stokes_u, stokes_v, wave_number, status = _evaluate_finite_depth_stokes_given_wave_number(
            np.ascontiguousarray(node_wave_number_grid[stokes_mask], dtype=np.float64),
            np.ascontiguousarray(node_status_grid[stokes_mask], dtype=np.int32),
            np.ascontiguousarray(hs_grid[stokes_mask], dtype=np.float64),
            np.ascontiguousarray(fp_grid[stokes_mask], dtype=np.float64),
            np.ascontiguousarray(dir_grid[stokes_mask], dtype=np.float64),
            np.ascontiguousarray(z_rel[stokes_mask], dtype=np.float64),
            np.ascontiguousarray(depth_grid[stokes_mask], dtype=np.float64),
            surface_tolerance_m=float(surface_tolerance_m),
            stokes_pool=stokes_pool,
            stokes_worker_count=stokes_worker_count,
        )
        ok_kept = status == STOKES_BATCH_STATUS_OK
        ok = np.zeros((nvrt, nnode), dtype=bool)
        ok[stokes_mask] = ok_kept
        u_field[ok] += stokes_u[ok_kept]
        v_field[ok] += stokes_v[ok_kept]
        stokes_ok_count = int(np.count_nonzero(ok_kept))

    return (
        u_field, v_field, w_field,
        stokes_ok_count, stokes_total_count, stokes_points_evaluated, stokes_points_possible,
    )


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
    field1_reuse_candidate: ComposedFieldEndpoint | None = None,
    precomputed_wet_dry_flags: tuple[np.ndarray, np.ndarray] | None = None,
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
        field1_reuse_candidate: WP-14 任務 B 新增。``None``（預設）：完全
            不啟用，``field1`` 一律照原本路徑重新計算乾濕歸零＋Stokes，
            行為與此參數新增前逐位元相同。呼叫端（通常是
            ``hour_major.py`` 的小時主軸迴圈）可以傳入**上一小時**回傳的
            ``ComposedCurrentAndStokesFields.field2_endpoint``，供本函式
            嘗試直接拿來當這一小時的 ``field1``，省下一次乾濕歸零＋Stokes
            全網格計算——前提是相鄰兩小時的窗口相連（第 h 小時的
            ``field2`` 端點＝第 h+1 小時的 ``field1`` 端點，同一絕對時刻、
            同一資料來源），這件事本函式無從得知，完全由呼叫端自行判斷
            （比對 ``field_pair`` 的 ``time_before_utc_ns``／
            ``month_id1``／``time_origin1`` 是否與上一小時 ``field2`` 端點
            的身分相符，以及波浪時刻是否相符），本函式只負責**傳進來的
            候選是否在數值上真的可以安全重用**這件事。

            **為什麼需要比對 node_is_dry**：``compose_current_and_
            stokes_fields`` 對同一小時的 ``field1``／``field2`` 套用
            **同一份**乾濕遮罩（``node_is_dry``，只由這一小時自己的
            ``field_pair.eta_field1`` 決定，見上方步驟 1），不是「每個端點
            各自用自己時刻的海面高程判定乾濕」。這代表：即使第 h 小時的
            ``field2``（端點時刻 ``T-Δh``）與第 h+1 小時的 ``field1``
            （同一個端點時刻 ``T-Δh``）指向同一份原始流場資料，兩者在各自
            小時裡實際套用的乾濕遮罩未必相同——前者用的是第 h 小時自己的
            ``eta_field1``（時刻 ``T``）算出的遮罩，後者用的是第 h+1 小時
            自己的 ``eta_field1``（時刻 ``T-Δh``，即端點自己的海面）算出的
            遮罩，兩個遮罩來自不同時刻的海面高程，只有在這兩個時刻的乾濕
            狀態剛好完全一致時才會逐位元相同。因此本函式**一律**先算出
            這一小時自己的 ``node_is_dry``，再與候選記錄的
            ``node_is_dry`` 逐點比對（``np.array_equal``）；只有完全相同
            才重用，任何一個節點不同就整個放棄重用、退回正常重新計算——
            這是刻意保守的 fail-closed 設計，寧可少省一次重用機會，也不
            冒任何逐位元不一致的風險。這個檢查本身很便宜（一次布林陣列
            比較，遠比重新算一次 Stokes 便宜），不會抵銷重用帶來的效益。
        precomputed_wet_dry_flags: WP-15 任務 A.3 新增。``None``（預設）：
            完全不啟用，行為與此參數新增前逐位元相同——本函式自己呼叫
            一次 ``compute_wet_dry_flags(eta_m=field_pair.eta_field1)``。
            提供 ``(node_is_dry, elem_is_dry)`` tuple 時，本函式**信賴**
            呼叫端保證這是用**同一個** ``field_pair.eta_field1`` 算出來的
            結果（本函式不會重新驗證），直接使用、跳過重算——這是為了
            ``hour_major.py`` 主迴圈的既有情境：主迴圈在呼叫本函式之前，
            已經為了「是否有未登錄缺口」的檢查而呼叫過一次
            ``compute_wet_dry_flags(eta_m=field_pair.eta_field1)``，兩次
            呼叫的輸入完全相同、``compute_wet_dry_flags`` 是純函式（同樣
            輸入必然得到同樣輸出，見該函式實作只讀 ``eta_m`` 與網格拓樸，
            没有隱藏狀態），因此第二次呼叫必然得到逐位元相同的結果，
            傳入預先算好的結果不會改變任何輸出，只省下一次全網格乾濕
            判定的成本（見最終報告量測）。呼叫端若傳入與
            ``field_pair.eta_field1`` 不一致的旗標，屬於誤用（本函式不做
            防禦性重新驗證，成本考量與 ``field1_reuse_candidate`` 的
            ``node_is_dry`` 比對不同——那裡的重算候選是「可能不對」的
            跨小時資料，這裡的旗標是「呼叫端自己在同一小時、同一輸入下
            已經算過一次」，性質不同，不需要重複比對）。

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
    if precomputed_wet_dry_flags is not None:
        node_is_dry, elem_is_dry = precomputed_wet_dry_flags
    else:
        node_is_dry, elem_is_dry = compute_wet_dry_flags(driver_grid, handles, eta_m=field_pair.eta_field1)
    kbp = grid.kbp
    dp = grid.dp
    nvrt, nnode = field_pair.u_field1.shape
    effective_mask = (np.arange(nvrt)[:, None] >= kbp[None, :]) & ~node_is_dry[None, :]
    has_wave = wave_field1 is not None and wave_field2 is not None

    # WP-14 任務 B：field1 是否可以直接沿用上一小時已經算好的 field2 端點，
    # 見 ``field1_reuse_candidate`` docstring「為什麼需要比對 node_is_dry」
    # 一節；只有在呼叫端提供候選、且候選記錄的遮罩與這一小時剛算好的
    # ``node_is_dry`` 逐位元相同時才重用，否則照原本路徑重新計算——不確定
    # 就重算，不冒險假設兩者一定一樣。
    field1_reused = (
        field1_reuse_candidate is not None
        and np.array_equal(field1_reuse_candidate.node_is_dry, node_is_dry)
        and _same_source_arrays(
            field1_reuse_candidate.source_arrays,
            _endpoint_source_arrays(
                field_pair.u_field1, field_pair.v_field1, field_pair.w_field1, field_pair.eta_field1,
                wave_field1 if has_wave else None,
            ),
        )
    )
    if field1_reused:
        u1 = np.array(field1_reuse_candidate.u, dtype=np.float64, copy=True)
        v1 = np.array(field1_reuse_candidate.v, dtype=np.float64, copy=True)
        w1 = np.array(field1_reuse_candidate.w_no_settling, dtype=np.float64, copy=True)
        stokes_ok_count = field1_reuse_candidate.stokes_ok_count
        stokes_total_count = field1_reuse_candidate.stokes_total_count
        stokes_points_evaluated = field1_reuse_candidate.stokes_points_evaluated
        stokes_points_possible = field1_reuse_candidate.stokes_points_possible
    else:
        u1 = np.array(field_pair.u_field1, dtype=np.float64, copy=True)
        v1 = np.array(field_pair.v_field1, dtype=np.float64, copy=True)
        w1 = np.array(field_pair.w_field1, dtype=np.float64, copy=True)
        _zero_dry_and_below_kbp(u1, v1, w1, kbp=kbp, node_is_dry=node_is_dry)
        u1, v1, w1, ok1, total1, eval1, possible1 = _compose_single_endpoint(
            grid, u1, v1, w1, eta=field_pair.eta_field1, wave=wave_field1 if has_wave else None,
            kbp=kbp, dp=dp, effective_mask=effective_mask, nvrt=nvrt, nnode=nnode,
            gravity_mps2=gravity_mps2, surface_tolerance_m=surface_tolerance_m,
            stokes_depth_threshold_ratio=stokes_depth_threshold_ratio,
            stokes_pool=stokes_pool, stokes_worker_count=stokes_worker_count,
        )
        stokes_ok_count, stokes_total_count = ok1, total1
        stokes_points_evaluated, stokes_points_possible = eval1, possible1

    u2 = np.array(field_pair.u_field2, dtype=np.float64, copy=True)
    v2 = np.array(field_pair.v_field2, dtype=np.float64, copy=True)
    w2 = np.array(field_pair.w_field2, dtype=np.float64, copy=True)
    _zero_dry_and_below_kbp(u2, v2, w2, kbp=kbp, node_is_dry=node_is_dry)
    u2, v2, w2, ok2, total2, eval2, possible2 = _compose_single_endpoint(
        grid, u2, v2, w2, eta=field_pair.eta_field2, wave=wave_field2 if has_wave else None,
        kbp=kbp, dp=dp, effective_mask=effective_mask, nvrt=nvrt, nnode=nnode,
        gravity_mps2=gravity_mps2, surface_tolerance_m=surface_tolerance_m,
        stokes_depth_threshold_ratio=stokes_depth_threshold_ratio,
        stokes_pool=stokes_pool, stokes_worker_count=stokes_worker_count,
    )
    stokes_ok_count += ok2
    stokes_total_count += total2
    stokes_points_evaluated += eval2
    stokes_points_possible += possible2

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
        field2_endpoint=ComposedFieldEndpoint(
            node_is_dry=node_is_dry,
            u=np.ascontiguousarray(u2),
            v=np.ascontiguousarray(v2),
            w_no_settling=np.ascontiguousarray(w2),
            stokes_ok_count=ok2,
            stokes_total_count=total2,
            stokes_points_evaluated=eval2,
            stokes_points_possible=possible2,
            source_arrays=_endpoint_source_arrays(
                field_pair.u_field2, field_pair.v_field2, field_pair.w_field2, field_pair.eta_field2,
                wave_field2 if has_wave else None,
            ),
        ),
    )


def apply_settling_velocity(
    base: ComposedCurrentAndStokesFields,
    settling_velocity_mps: float,
    *,
    kbp: np.ndarray,
) -> ComposedFieldPair:
    """把常數沉降速度加進 ``compose_current_and_stokes_fields`` 的結果，
    完成 ``compose_total_velocity_fields`` 原本的步驟 4、5。

    這一步刻意做得非常便宜，因為它是「每個沉降批次都要各自做一次」的部分
    （見 ``ComposedCurrentAndStokesFields`` docstring；正式規模每個到達
    時刻有 10 個沉降速度分組，本函式即為每小時要重複 10 次的部分）；真正
    昂貴的乾濕判定與 Stokes 全網格計算已經在 ``base`` 裡算好，這裡不重算。

    WP-15 任務 A.2 優化（誠實記錄，位元相同由
    ``tests/test_native_backward_driver.py`` 的
    ``test_apply_settling_velocity_matches_pre_optimization_reference``
    系列測試證明）：本函式原本對 ``u1/v1/u2/v2`` 各自複製一份、再對
    ``u1/v1/w1`` 與 ``u2/v2/w2`` 各呼叫一次 ``_zero_dry_and_below_kbp``
    （每次呼叫內部又是好幾個布林 fancy-indexing 寫入），合計 6 次
    ``(nvrt, nnode)`` 陣列複製＋12 次布林遮罩寫入，但這些操作對 ``u/v``
    全部都是**可證明的沒有作用（no-op）**：

    1. ``u1/v1/u2/v2`` 在 ``base`` 裡已經是
       ``compose_current_and_stokes_fields`` 呼叫
       ``_zero_dry_and_below_kbp`` 的結果（乾點、海床以下層都已經是
       0），本函式從頭到尾不曾對 ``u/v`` 做任何加法或其他修改，因此
       「複製一份再原地歸零一次已經是 0 的位置」與「直接沿用原陣列」
       在數值上逐位元相同——這不是近似，是同一個陣列的兩種讀法。
       因此本函式**不再複製 ``u1/v1/u2/v2``**，直接把 ``base`` 的參考
       原樣放進回傳值；下游唯一會讀到這些陣列的呼叫端
       （``handles.batch_move_rk4``／``handles.batch_interpolate_velocity``，
       見 BayTrace 原版 ``batch_interp.py`` 逐行核對）只把它們當
       **唯讀**輸入（傳給 njit kernel 做速度場內插，函式簽章裡這幾個
       陣列自始至終沒有被賦值過），因此多個沉降批次安全共用同一份
       ``base`` 陣列參考，不會互相污染。
    2. ``w1/w2`` 確實需要複製＋修改（加沉降速度），但**加沉降之後
       的第二次防禦性歸零同樣是 no-op**：沉降只加在
       ``base.effective_mask``（濕節點且 ``k>=kbp``）內，而
       ``w_field1_no_settling``／``w_field2_no_settling`` 在進入本函式
       前，乾點與 ``k<kbp`` 的位置**已經**是 0（``compose_current_and_
       stokes_fields`` 保證的不變量，見該函式與
       ``ComposedCurrentAndStokesFields`` docstring）；``effective_mask``
       的定義恰好是「乾濕遮罩」與「``k>=kbp``」的交集，與
       ``_zero_dry_and_below_kbp`` 要歸零的位置（乾點∪``k<kbp``）互斥，
       因此 ``w1[effective_mask] += settling`` 完全不會碰到需要被
       第二次歸零的位置——第二次歸零因此對 ``w1/w2`` 也是 no-op，
       可以安全移除，只保留「複製＋在遮罩內加常數」這兩步。

    因此本函式現在只做 2 次 ``(nvrt, nnode)`` 陣列複製（``w1``／``w2``）＋
    2 次遮罩內純量加法，省下另外 4 次複製與全部 12 次布林遮罩寫入；在
    67,593 節點×48 層、10 沉降分組的實測中，這一步的每小時合計成本從約
    569 ms 降到約 60 ms（見最終報告 Part A 量測表）。

    Args:
        base: ``compose_current_and_stokes_fields`` 的輸出，可以是跨多個
            沉降批次共用的同一個物件（本函式不修改 ``base`` 的任何陣列，
            回傳值裡的 ``u_field1``／``v_field1``／``u_field2``／
            ``v_field2`` 直接是 ``base`` 的陣列參考，安全併發共用；只有
            ``w_field1``／``w_field2`` 是新配置的獨立複本）。
        settling_velocity_mps: 這一批粒子共用的沉降速度（m/s），語意同
            ``compose_total_velocity_fields``。
        kbp: ``(nnode,)`` 每個節點第一個有效垂向層索引，保留這個參數是
            為了維持呼叫端介面相容；本函式優化後不再需要它來做防禦性
            歸零（見上方推導），但仍保留在簽章中，避免影響既有呼叫端。

    Raises:
        BackwardDriverError: ``settling_velocity_mps`` 非有限值。
    """

    del kbp  # 見 docstring：優化後不再需要，保留參數只為介面相容。

    settling = float(settling_velocity_mps)
    if not np.isfinite(settling):
        raise BackwardDriverError("settling_velocity_mps 必須是有限值")

    w1 = np.array(base.w_field1_no_settling, dtype=np.float64, copy=True)
    w2 = np.array(base.w_field2_no_settling, dtype=np.float64, copy=True)
    w1[base.effective_mask] += settling
    w2[base.effective_mask] += settling

    return ComposedFieldPair(
        u_field1=base.u_field1,
        v_field1=base.v_field1,
        w_field1=np.ascontiguousarray(w1),
        eta_field1=base.eta_field1,
        u_field2=base.u_field2,
        v_field2=base.v_field2,
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


# ── WP-16 任務 1：依「活動粒子局部鄰域」的 CFL 子步政策 ──────────────────────
#
# 背景與動機
# ------------------------------------------------------------------------
# ``estimate_substep_count``（上面，本次更名為「``global_cfl``」政策，程式
# 完全未修改，繼續保留作為位元相同的既有行為）用「全網格」的最大速度與
# 最短邊長估計 ``ndeltp``，這對地形複雜、含極小近岸元素的網格極度保守：
# 主代理在伺服器 .203 的實測（見最終報告）显示，全網格版本量出的
# ``ndeltp`` 遠超過活動粒子實際所在區域需要的子步數（後灣網格單小時約
# 4,250 個子步，其中每個子步的 Python 開銷是目前最大的效能瓶頸）。
#
# 「局部 CFL」政策改成只看「活動粒子這一小時可能到達的元素」：以每顆
# 活動粒子目前位置為圓心、半徑取「這一小時全域最大速度 × 3600 秒 ×
# 安全係數」（``radius_safety_factor``，預設 3 倍，見
# ``estimate_substep_count_local_cfl`` 參數說明）畫一個圓，落在圓內（或
# 與圓相交，見下方查詢半徑的保守外擴）的元素才會被用來計算「局部最短邊」
# 與「局部最大速度」；圓的半徑仍然用**全域**最大速度計算（不是局部速度），
# 這是刻意的保守設計——即使某顆粒子目前所在區域流速很慢，只要網格上其他
# 地方有更快的速度，這顆粒子理論上仍有可能在下一個 RK4 子步被平流到那個
# 更快的區域（雖然機率極低，一個子步的位移不可能真的遠達整個網格對角，
# 但「用全域最大速度定半徑」本身已經是一個遠比實際可能位移更寬鬆的上界，
# 足以涵蓋任何合理的單一小時內平流路徑），因此半徑本身的保守性不依賴局部
# 速度假設，只依賴「全域最大速度 × 3600 秒」這個物理上不可能超過的位移
# 上界，再乘上安全係數進一步放寬。
#
# 仍然保證 ``dtb <= dt_max_seconds``（呼叫端未提供任何活動粒子，或全場
# 速度為零時，退化為只用 ``dt_max_seconds`` 上限，與原版
# ``estimate_substep_count`` 在 ``max_speed<=0`` 時的退化行為一致）。
#
# 是否可以完全不做「局部 CFL」、只靠 ``dt_max_seconds`` 上限？
# ------------------------------------------------------------------------
# 讀 BayTrace 原版 ``batch_move_rk4``（``batch_interp.py`` 1263-1345 行）：
# RK4 每一個 stage（k1-k4）都會呼叫 ``_find_element_at`` 重新定位到當時
# 的半步／全步暫定位置所在元素，再用該元素當下的流場做速度內插——也就是
# 說，**單一子步內部本來就正確處理了粒子跨越多個元素**的情形，不是「假設
# 粒子整個子步都停留在同一個元素」的簡化模型。這代表 CFL 條件在這裡真正
# 要保證的，不是「粒子不能跨元素」（跨元素本來就有正確處理），而是「RK4
# 用來近似連續運動的離散時間步長，相對於局部流場變化尺度／網格解析度不能
# 太粗」這個標準的數值穩定性／精度考量（見 ``estimate_substep_count`` 既有
# docstring：這本來就「不是嚴謹的 RK4＋隨機擴散穩定性證明，只是工作書要求
# 的明確下限搭配一個標準的平流 CFL 安全係數」）。因此，理論上确实存在「只
# 靠 ``dt_max_seconds`` 上限，不做任何 CFL 判斷」的可能性——依工作書給的
# 例子，``dt_max=30`` 秒時只需要 120 個子步（相對於實測的 4,250 個子步，
# 若可行將是巨大的效益）。
#
# **本模組的建議與判斷**：不採用「只靠 dt_max_seconds」作為預設值，原因
# 有二。第一，``dt_max_seconds`` 上限完全不隨流速變化，若未來換一個流速
# 明顯更強（例如颱風期間的海流）或網格解析度明顯更細的場景，「120 個子步
# 是否仍然足夠精確」沒有任何流速相關的保證，純粹依賴使用者自己手動調低
# ``dt_max_seconds``；相較之下，CFL 條件（不論全域或局部）會隨流速自動
# 提高子步數，是更穩健、更不容易被使用者疏忽而低估精度的預設值。第二，
# ``dt_max_seconds`` 上限只約束「時間離散」的粗細，完全不管「空間離散」
# （子步位移相對於網格元素大小的比例）——即使 RK4 的 ``_find_element_at``
# 能正確處理跨元素，若一個子步的位移遠大於局部元素尺寸，插值使用的「子步
# 開始時刻」流場在子步結束時的位置可能已經不能代表該處真實流場（尤其在
# 近岸複雜地形，流速方向與量值在很短距離內劇烈變化），這是 CFL 條件真正
# 要防範的物理／數值精度風險，不是「跨元素會不會出錯」的問題。因此本模組
# 預設 ``substep_policy="local_cfl"``（局部 CFL＋``dt_max`` 上限，比全域
# CFL 更寬鬆但仍然保守），供 PI／主代理視實際場景另外決定是否要在充分驗證
# 後改用「只靠 dt_max_seconds」這個更激進的選項（本模組未提供這個選項，
# 因為那已經不是「CFL 政策的實作細節」，而是「要不要保留 CFL 條件本身」
# 這個更高層次的科學判斷，不屬於本工作包被授權自行決定的範圍）。


@dataclass(frozen=True, slots=True)
class LocalCflSpatialIndex:
    """WP-16 任務 1：局部 CFL 子步政策用的靜態空間索引與逐元素最短邊快取。

    全部欄位只跟網格拓樸有關（元素形心座標、逐元素三個節點索引、逐元素
    最短邊、全域元素「最大延伸半徑」），與任何一小時的流場或粒子位置完全
    無關，只需要在網格建立後計算一次，供
    ``estimate_substep_count_local_cfl`` 在每小時重複查詢時避免重算——
    呼叫端（``hour_major.py``）應該在整個 ``run_flow_domain_hour_major``
    呼叫開始時建立一次、跨全部小時重複傳入。

    Attributes:
        element_centroid_kdtree: ``scipy.spatial.cKDTree``，索引全部元素的
            形心座標 ``(xctr, yctr)``。
        element_node0/1/2: ``(ne,)`` int64，每個元素三角形的三個節點全域
            索引（與 ``grid.elnode[:, :3]`` 相同，本模組只支援三角形，與
            ``smagorinsky_batch.SmagorinskyGridArrays`` 的既有限制一致）。
        element_min_edge_length: ``(ne,)`` float64，每個元素三邊中最短的
            一段（公尺），與 ``build_native_backward_driver_grid`` 計算
            ``min_edge_length_m``（全域最小值）用的同一組邊長定義，只是
            這裡保留逐元素的值，不取全域最小。
        element_max_reach_m: 純量，全部元素中「形心到自己三個頂點最遠距離」
            的最大值（公尺）——查詢空間索引時，半徑會加上這個值，確保「形心
            落在查詢半徑外、但元素本體仍可能與查詢圓相交」的元素不會被
            漏掉（見 ``estimate_substep_count_local_cfl`` 的查詢半徑推導）。
    """

    element_centroid_kdtree: Any
    element_node0: np.ndarray
    element_node1: np.ndarray
    element_node2: np.ndarray
    element_min_edge_length: np.ndarray
    element_max_reach_m: float


def build_local_cfl_spatial_index(driver_grid: NativeBackwardDriverGrid) -> LocalCflSpatialIndex:
    """建立 ``LocalCflSpatialIndex``，只依賴網格拓樸，呼叫端應該只呼叫一次。"""

    from scipy.spatial import cKDTree

    grid = driver_grid.schism_grid
    elnode = np.asarray(grid.elnode, dtype=np.int64)
    n1 = elnode[:, 0]
    n2 = elnode[:, 1]
    n3 = elnode[:, 2]
    x = np.asarray(grid.x, dtype=np.float64)
    y = np.asarray(grid.y, dtype=np.float64)
    xctr = np.asarray(grid.xctr, dtype=np.float64)
    yctr = np.asarray(grid.yctr, dtype=np.float64)

    edge01 = np.hypot(x[n1] - x[n2], y[n1] - y[n2])
    edge12 = np.hypot(x[n2] - x[n3], y[n2] - y[n3])
    edge20 = np.hypot(x[n3] - x[n1], y[n3] - y[n1])
    element_min_edge_length = np.minimum(np.minimum(edge01, edge12), edge20)

    reach0 = np.hypot(x[n1] - xctr, y[n1] - yctr)
    reach1 = np.hypot(x[n2] - xctr, y[n2] - yctr)
    reach2 = np.hypot(x[n3] - xctr, y[n3] - yctr)
    element_max_reach_m = float(np.max(np.maximum(np.maximum(reach0, reach1), reach2)))

    kdtree = cKDTree(np.column_stack([xctr, yctr]))

    return LocalCflSpatialIndex(
        element_centroid_kdtree=kdtree,
        element_node0=n1,
        element_node1=n2,
        element_node2=n3,
        element_min_edge_length=element_min_edge_length,
        element_max_reach_m=element_max_reach_m,
    )


def estimate_substep_count_local_cfl(
    driver_grid: NativeBackwardDriverGrid,
    u_field: np.ndarray,
    v_field: np.ndarray,
    *,
    active_x: np.ndarray,
    active_y: np.ndarray,
    spatial_index: LocalCflSpatialIndex,
    dt_seconds: float = 3600.0,
    dt_max_seconds: float = 30.0,
    cfl_number: float = 0.5,
    radius_safety_factor: float = 3.0,
) -> int:
    """WP-16 任務 1：只看「活動粒子這一小時可能到達的元素」估計 ``ndeltp``。

    與 ``estimate_substep_count``（「``global_cfl``」政策）相同的
    ``ndeltp = max(ceil(dt/dt_max), ceil(dt*max_speed/(cfl*min_edge)))``
    公式，差別只在 ``max_speed``／``min_edge`` 的取值範圍：這裡限定在
    「活動粒子鄰域」內的節點／元素，不是全網格（見模組文件「WP-16 任務 1」
    一節的完整推導與保守性論證）。

    Args:
        active_x, active_y: 這一小時全部（跨所有沉降分組合併的）活動粒子
            目前位置，公尺制。空陣列代表這一小時沒有任何活動粒子。
        spatial_index: ``build_local_cfl_spatial_index`` 的輸出，呼叫端
            應該跨小時重複傳入同一份，不要每小時重建。
        radius_safety_factor: 查詢半徑＝``radius_safety_factor *
            這一小時全域最大速度 * dt_seconds``，見模組文件說明為何半徑
            本身仍用全域最大速度（而非局部速度）計算。

    Returns:
        ``ndeltp``（正整數），保證 ``dt_seconds / ndeltp <= dt_max_seconds``，
        且行為在「沒有活動粒子」或「全場速度為零」時退化為與
        ``estimate_substep_count`` 相同的 ``dt_max_seconds``-only 下限。
    """

    if dt_seconds <= 0.0 or dt_max_seconds <= 0.0 or cfl_number <= 0.0:
        raise BackwardDriverError("dt_seconds／dt_max_seconds／cfl_number 必須是正值")
    if radius_safety_factor <= 0.0:
        raise BackwardDriverError("radius_safety_factor 必須是正值")
    active_x = np.asarray(active_x, dtype=np.float64)
    active_y = np.asarray(active_y, dtype=np.float64)
    if active_x.shape != active_y.shape:
        raise BackwardDriverError("active_x／active_y 必須是相同 shape 的一維陣列")

    ndeltp_from_dt_max = int(np.ceil(dt_seconds / dt_max_seconds))

    speed = np.hypot(u_field, v_field)
    finite_speed = speed[np.isfinite(speed)]
    global_max_speed = float(np.max(finite_speed)) if finite_speed.size > 0 else 0.0

    # 沒有活動粒子，或這一小時全場速度都是零／非有限值：沒有東西需要局部
    # CFL 判斷（沒有粒子可以移動、或移動不了），與全域版本的退化行為一致，
    # 只用 dt_max_seconds 上限。
    if active_x.size == 0 or global_max_speed <= 0.0:
        return max(1, ndeltp_from_dt_max)

    radius = radius_safety_factor * global_max_speed * dt_seconds
    query_radius = radius + spatial_index.element_max_reach_m
    points = np.column_stack([active_x, active_y])
    neighbor_lists = spatial_index.element_centroid_kdtree.query_ball_point(points, r=query_radius)

    nearby_elements: set[int] = set()
    for neighbors in neighbor_lists:
        nearby_elements.update(neighbors)

    if not nearby_elements:
        # 理論上不應該發生（活動粒子必然位於某個元素內部，該元素形心離它
        # 自己不可能超過 element_max_reach_m，一定會被查到）；若真的發生，
        # fail-closed 退回全網格版本，不冒任何低估 ndeltp 的風險。
        return estimate_substep_count(
            driver_grid, u_field, v_field,
            dt_seconds=dt_seconds, dt_max_seconds=dt_max_seconds, cfl_number=cfl_number,
        )

    idx = np.fromiter(nearby_elements, dtype=np.int64, count=len(nearby_elements))
    local_min_edge_length = float(np.min(spatial_index.element_min_edge_length[idx]))

    local_nodes = np.unique(
        np.concatenate(
            [
                spatial_index.element_node0[idx],
                spatial_index.element_node1[idx],
                spatial_index.element_node2[idx],
            ]
        )
    )
    local_speed = speed[:, local_nodes]
    local_finite = local_speed[np.isfinite(local_speed)]
    local_max_speed = float(np.max(local_finite)) if local_finite.size > 0 else 0.0

    if local_max_speed <= 0.0:
        ndeltp_cfl = 1
    else:
        dtb_cfl = cfl_number * local_min_edge_length / local_max_speed
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
    endpoint_reuse_hits: int = 0
    """WP-14 任務 B 效能診斷：這一小時的 ``field1`` 成功沿用上一小時已經
    算好的 ``field2`` 端點（``compose_current_and_stokes_fields`` 的
    ``field1_reuse_candidate`` 命中，省下一次乾濕歸零＋Stokes 全網格計算）
    的次數；只在 ``compose_base_cache`` 未命中（真正呼叫過
    ``compose_current_and_stokes_fields``）的小時才有意義，只用於
    ``hour_major.py`` 的小時主軸路徑，``run_backward_batch`` 的逐單元路徑
    不會用到端點重用，恆為 0。"""
    endpoint_reuse_misses: int = 0
    """與 ``endpoint_reuse_hits`` 對應：有候選但因為身分不符（不同流場
    時刻／資料來源／波浪時刻）或 ``node_is_dry`` 遮罩不同而 fail-closed
    重新計算的次數；不含「根本沒有候選」（例如這一小時是整個 run 的第一
    個小時，或緊接在缺值缺口、checkpoint 續跑之後）的情形，那些情形不計入
    任何一個計數器，避免把「本來就不該重用」的正常情形誤讀成「重用失敗」。
    """

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
        self.endpoint_reuse_hits += other.endpoint_reuse_hits
        self.endpoint_reuse_misses += other.endpoint_reuse_misses

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
    """一顆粒子離開追蹤（因任何終止原因）時的位置、時刻與狀態碼。

    ``also_local_domain_first_exit``（WP-23 新增）：只有在這筆記錄的
    ``status == NATIVE_STATUS_LEFT_FLOW_DOMAIN`` **且**這顆粒子所屬測站的
    local domain 與 flow domain 相同（``local_equals_flow=True``，目前是新竹／
    南灣／連江）時才可能是 ``True``——代表「離開 flow domain」與「離開 local
    domain」是同一次穿越，依規格只寫這一筆事件、不得另外產生
    ``LocalDomainFirstExitRecord``。對貢寮／龜山島（``local_equals_flow=False``），
    這個欄位恆為 ``False``；local 首次離開一律另外記在
    ``BackwardDriverResult.local_domain_departures``。此欄位由呼叫端
    （``run_backward_batch``／``hour_major.run_flow_domain_hour_major``）在收集完
    ``departures`` 後依站點身分回填，子步迴圈本身不知道這個站點層級的事實。
    """

    row_particle_index: int
    status: int
    x_m: float
    y_m: float
    z_m: float
    utc_ns: int
    also_local_domain_first_exit: bool = False


@dataclass(frozen=True, slots=True)
class LocalDomainFirstExitRecord:
    """一顆粒子首次離開自己站點 local domain（``own_local_domain``）的非終止事件。

    WP-23 新增。只適用於 local domain 與 flow domain 不同的測站（目前是貢寮、
    龜山島）；離開後粒子**不終止**、繼續回溯，因此本記錄與
    ``DepartureRecord``（終止事件）是兩個獨立的資料流，一顆粒子最多各出現一次
    （local 一次、flow 一次），彼此互不覆蓋。``local_equals_flow=True`` 的測站
    （新竹／南灣／連江）不會產生這個記錄——那些站點的「離開 local domain」與
    「離開 flow domain」是同一次穿越，直接由 ``DepartureRecord.
    also_local_domain_first_exit=True`` 表達，避免重複計數。
    欄位語意與 ``DepartureRecord`` 對應（位置為公尺制運算座標，``utc_ns`` 是
    離開當下的 UTC 奈秒）；backward age、所屬邊界分段與弧長位置由後處理層
    （``postprocess.py``）依 ``arrival_time_utc_ns`` 與 ``BoundaryGeometry.
    own_local_open_boundary`` 另行推導，本記錄本身不重複保存這些可衍生欄位。
    """

    row_particle_index: int
    x_m: float
    y_m: float
    z_m: float
    utc_ns: int


@dataclass(frozen=True, slots=True)
class BedNoDepositionEventRecord:
    """``BackwardDriverConfig.bed_no_deposition=True`` 時，一次「本應沉底、改為不沉積」的非終止事件。

    WP-26 新增、WP-28 正名（舊名 ``BedReflectEventRecord``，見模組文件「WP-26／
    WP-28」一節的正名說明——這裡記錄的從來就不是反射事件，是「不沉積」事件）。
    語意對應 ``LocalDomainFirstExitRecord``：只在明示啟用 ``bed_no_deposition``
    敏感度案例時才會產生，``bed_no_deposition=False``（預設基線）時
    ``BackwardDriverResult.bed_no_deposition_events`` 恆為空 tuple。記錄的時刻
    是這顆粒子連續 ``deposition_consecutive_hours`` 小時停留在海床附近、依
    基線政策「應該」被標記 ``NATIVE_STATUS_DEPOSITED`` 的那個小時終點；粒子
    本身**不**終止，連續小時計數會重設為 0（見 ``run_backward_batch``），因此
    同一顆粒子若之後又再次連續停留滿足門檻，會再產生一筆新記錄——一顆粒子
    在同一次回溯中可能出現多筆，不像 ``DepartureRecord``／
    ``LocalDomainFirstExitRecord`` 那樣至多一筆。欄位語意與
    ``LocalDomainFirstExitRecord`` 相同（公尺制運算座標、UTC 奈秒）。
    """

    row_particle_index: int
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
    local_region_polygon: Polygon | None = None,
    local_region_polygons: np.ndarray | None = None,
    local_first_exit_pending: np.ndarray | None = None,
    local_departures: list[LocalDomainFirstExitRecord] | None = None,
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
    - 每個子步結束後，依序做兩層邊界判定（WP-23：兩層語意不同，local 在前、
      flow 在後，見模組文件「WP-23：local domain 與 flow domain 的兩階段
      終止語意」一節）：
      1. 若提供 ``local_region_polygon``／``local_region_polygons``：以
         Shapely 2.0 向量化 ``contains_xy`` 判斷「尚未記錄過首次離開」
         （``local_first_exit_pending`` 為真）且仍作用中的粒子是否仍在自己
         站點的 local domain 內；新離開者記錄 ``LocalDomainFirstExitRecord``
         並把 ``local_first_exit_pending`` 該列改為假——**不終止**，粒子
         （``pa.status``）不變、``native_status`` 不變。
      2. 若提供 ``interest_region_polygon``／``interest_region_polygons``
         （WP-23 起代表 flow domain 開放邊界）：以同一種向量化方式判斷全部
         作用中粒子是否仍在流場區域內，新離開者記錄 ``DepartureRecord`` 並
         停用（``pa.status=0``、``native_status=NATIVE_STATUS_LEFT_FLOW_DOMAIN``）。
      因為 local domain 幾何上是 flow domain 的子集，正常情況下第 1 層一定
      早於（或至少不晚於）第 2 層對同一顆粒子觸發；若某一步同時滿足兩層
      條件（例如子步位移很大），兩個事件仍會在同一子步各自正確記錄一次，
      互不覆蓋。

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
        native_status: ``(n,)`` int32，本函式只在偵測到離開流場區域時
            寫入 ``NATIVE_STATUS_LEFT_FLOW_DOMAIN``；其餘終止原因由
            呼叫端（``run_backward_batch``）在小時邊界另行判定。提供
            ``interest_region_polygon``／``interest_region_polygons`` 時
            必須同時提供本參數。
        substep_utc_ns: 給定子步索引 ``idt``（``1..ndeltp``），回傳該子步
            結束時對應的 UTC 奈秒，供 ``DepartureRecord.utc_ns``／
            ``LocalDomainFirstExitRecord.utc_ns`` 使用；提供
            ``interest_region_polygon``／``interest_region_polygons``
            或 ``local_region_polygon``／``local_region_polygons`` 任一者
            時都必須同時提供。
        interest_region_polygon: **單一**流場區域（flow domain）開放邊界
            多邊形，套用到全部 ``n`` 顆粒子——適用於「整批粒子共用同一個
            測站」的既有逐單元路徑（``run_backward_batch``）。離開此多邊形
            即終止（WP-23 起語意固定是 flow domain，不是 local domain，
            見模組文件）。與 ``interest_region_polygons`` 互斥。
        interest_region_polygons: ``(n,)`` 的 Shapely ``Polygon`` object
            陣列，**逐列**指定每顆粒子自己的流場區域邊界——適用於 WP-9
            小時主軸排程「同一次 kernel 呼叫混合多個測站的粒子」的情境
            （見 ``hour_major.py``）。同一個 flow domain 內的不同站點通常
            填同一個多邊形物件。Shapely 2.0 的向量化函式（例如
            ``contains_xy``）支援「geometry 引數本身也是與座標陣列等長的
            object 陣列」，逐點配對計算，因此這裡對每個粒子使用正確多邊形
            的計算結果，與「把每個測站的粒子分開、各自呼叫一次
            ``interest_region_polygon``」完全等價（幾何述詞是逐點獨立
            計算，不受同一次呼叫還有哪些其他點／多邊形影響），可以安全
            合併呼叫而不改變任何一顆粒子的判定結果。與
            ``interest_region_polygon`` 互斥。
        local_region_polygon, local_region_polygons: WP-23 新增，語意與
            ``interest_region_polygon``／``interest_region_polygons`` 對應，
            但幾何是每個測站**自己的** local domain（``own_local_domain``），
            離開時**不終止**、只記錄一次 ``LocalDomainFirstExitRecord``
            （見 ``local_first_exit_pending``）。``local_equals_flow=True``
            的測站（新竹／南灣／連江）不應傳入這兩個參數（由呼叫端在收集
            ``departures`` 後改用 ``DepartureRecord.
            also_local_domain_first_exit`` 表達，見 ``run_backward_batch``）。
            兩者互斥；提供任一者時必須同時提供 ``local_first_exit_pending``
            與 ``substep_utc_ns``。
        local_first_exit_pending: ``(n,)`` bool，**呼叫端配置並跨小時持續
            傳入的持久陣列**（初始值全 ``True``）；本函式只在偵測到「該列
            仍為 ``True``、且該粒子仍作用中、且已離開自己的 local domain」
            時把該列改成 ``False`` 並記一筆 ``LocalDomainFirstExitRecord``，
            確保同一顆粒子的 local 首次離開全程只記錄一次。提供
            ``local_region_polygon``／``local_region_polygons`` 時必須同時
            提供本參數。
        local_departures: 若提供，本函式會把每個 local domain 首次離開事件
            以 ``LocalDomainFirstExitRecord`` 附加到這個 list（呼叫端負責
            跨小時累積，行為與既有 ``departures`` 完全對應）。

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
    if local_region_polygon is not None and local_region_polygons is not None:
        raise BackwardDriverError("local_region_polygon 與 local_region_polygons 只能擇一提供")
    if (local_region_polygon is not None or local_region_polygons is not None) and (
        local_first_exit_pending is None or substep_utc_ns is None
    ):
        raise BackwardDriverError(
            "提供 local_region_polygon／local_region_polygons 時必須同時提供 "
            "local_first_exit_pending 與 substep_utc_ns"
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

        if local_region_polygon is not None or local_region_polygons is not None:
            t0 = _time.perf_counter()
            # WP-23：local domain 判定必須在 flow domain 判定「之前」，並且只
            # 看「仍作用中、且尚未記錄過 local 首次離開」的粒子（不能只用
            # active_mask，否則同一顆粒子每個子步都會重覆命中）；離開時只翻轉
            # local_first_exit_pending、完全不動 pa.status／native_status，這是
            # 「記錄但不終止」與下面 flow domain 判定「記錄並終止」的唯一差異。
            check_mask = (pa.status == 1) & local_first_exit_pending
            if np.any(check_mask):
                inside = np.ones(n, dtype=bool)
                if local_region_polygon is not None:
                    inside[check_mask] = contains_xy(
                        local_region_polygon, pa.x[check_mask], pa.y[check_mask]
                    )
                else:
                    inside[check_mask] = contains_xy(
                        local_region_polygons[check_mask], pa.x[check_mask], pa.y[check_mask]
                    )
                left_local_now = check_mask & ~inside
                if np.any(left_local_now):
                    local_utc_ns = int(substep_utc_ns(idt))
                    if local_departures is not None:
                        for row in np.flatnonzero(left_local_now):
                            local_departures.append(
                                LocalDomainFirstExitRecord(
                                    row_particle_index=int(row),
                                    x_m=float(pa.x[row]),
                                    y_m=float(pa.y[row]),
                                    z_m=float(pa.z[row]),
                                    utc_ns=local_utc_ns,
                                )
                            )
                    local_first_exit_pending[left_local_now] = False
            diag.time_interest_check_s += _time.perf_counter() - t0

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
                                    status=NATIVE_STATUS_LEFT_FLOW_DOMAIN,
                                    x_m=float(pa.x[row]),
                                    y_m=float(pa.y[row]),
                                    z_m=float(pa.z[row]),
                                    utc_ns=utc_ns,
                                )
                            )
                    native_status[left_now] = NATIVE_STATUS_LEFT_FLOW_DOMAIN
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
    # 比照原版在第一步先 ``_zero_dry_velocities(include_field1=True)`` 再定位：乾節點與
    # 海床以下層以複本歸零（真實 ocm_native 在海床以下存 NaN），不改動呼叫端陣列。
    node_is_dry_init, _ = compute_wet_dry_flags(driver_grid, handles, eta_m=shim.eta_field1)
    u_init = np.array(u_current_mps, dtype=np.float64, copy=True)
    v_init = np.array(v_current_mps, dtype=np.float64, copy=True)
    w_init = np.array(w_current_mps, dtype=np.float64, copy=True)
    _zero_dry_and_below_kbp(u_init, v_init, w_init, kbp=driver_grid.grid.kbp, node_is_dry=node_is_dry_init)
    shim.u_field2 = np.ascontiguousarray(u_init)
    shim.v_field2 = np.ascontiguousarray(v_init)
    shim.w_field2 = np.ascontiguousarray(w_init)
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

    ``bed_no_deposition``（WP-26 新增、WP-28 正名，舊名 ``bed_reflect``，見模組
    文件「WP-26／WP-28」一節）：預設 ``False``，行為與新增此欄位前逐位元
    相同——粒子連續 ``deposition_consecutive_hours`` 小時停留在海床附近（見
    ``particle_bed_z_levels``／``deposition_tolerance_m``）即終止並標記
    ``NATIVE_STATUS_DEPOSITED``（現行基線 ``deposit_on_first_contact_and_stop``
    的 native 對應實作）。設為 ``True`` 時關閉這個終止判定：符合條件的粒子
    改為重設連續小時計數、**不**終止、繼續留在 ``NATIVE_STATUS_ACTIVE`` 接受
    後續正常物理（平流＋Smagorinsky 擴散＋既有 ``batch_clamp_z`` 垂向夾限）
    驅動——**這只是「不視為沉積、不終止」，不是任何形式的鏡射反射或再懸浮
    動力學**（見模組文件的正名說明），量測「若不假設沉積物永久停留海床」
    這個基線假設讓來源足跡偏近多少。每次符合條件都會在回傳值的
    ``BackwardDriverResult.bed_no_deposition_events`` 多寫一筆
    ``BedNoDepositionEventRecord`` 稽核（不是終止事件，不進入
    ``departures``），供下游統計或人工核對這個案例確實發生過、發生在哪些
    粒子與小時。
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
    bed_no_deposition: bool = False


@dataclass(frozen=True, slots=True)
class BackwardDriverResult:
    """一次 ``run_backward_batch`` 呼叫的完整輸出。

    ``trajectory_*`` 的 shape 都是 ``(n_samples, n_particles)``（含第 0 筆
    即到達時刻的初始狀態），``trajectory_utc_ns`` 是 ``(n_samples,)``。
    ``departures`` 只包含真正發生終止事件的粒子，一顆粒子至多出現一次；
    未在其中出現、且 ``hours_run`` 未達 ``max_backtrack_seconds`` 就結束
    （理論上不會發生，因為主迴圈只在全部粒子都終止或觸及上限才結束）的
    情況不應存在，可作為驗收時的一致性檢查。

    ``local_domain_departures``（WP-23 新增）是**非終止**事件：只記錄
    ``local_equals_flow=False`` 的測站（目前是貢寮、龜山島）首次離開自己
    local domain 的位置與時刻，一顆粒子至多出現一次，且與 ``departures``
    互不重疊（一顆粒子可以同時出現在兩邊：local 一次、flow 一次）。
    ``local_equals_flow=True`` 的測站（新竹／南灣／連江）不會在這裡出現，
    改由 ``departures`` 對應那筆 ``DepartureRecord`` 的
    ``also_local_domain_first_exit=True`` 表達，見該欄位 docstring。

    ``bed_no_deposition_events``（WP-26 新增、WP-28 正名，舊名
    ``bed_reflect_events``）同樣是**非終止**事件：只在
    ``BackwardDriverConfig.bed_no_deposition=True`` 時才可能非空，見
    ``BedNoDepositionEventRecord`` docstring；``bed_no_deposition=False``
    （省略此欄位的既有呼叫端）時恆為空 tuple，不影響 ``departures``／
    ``final_status`` 等其餘欄位的既有內容或型別。
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
    local_domain_departures: tuple[LocalDomainFirstExitRecord, ...] = ()
    bed_no_deposition_events: tuple[BedNoDepositionEventRecord, ...] = ()


_HOUR_NS = 3_600_000_000_000


def build_pre_window_deposition_result(
    *,
    row_particle_index: np.ndarray,
    x0_m: np.ndarray,
    y0_m: np.ndarray,
    z0_m: np.ndarray,
    arrival_time_utc_ns: int,
) -> BackwardDriverResult:
    """為固定日曆回溯窗之外的沉底粒子組出一份「完全未進入運算」的結果。

    ``bed_residence.resolve_bed_residence_timing`` 在 ``fixed_calendar_window`` 模式判定
    ``pre_window_deposition=True`` 時，有效回溯期間為零——這批粒子的沉底時刻本身就早於
    「觀測日往前 H 天」的研究窗，因此呼叫端（``orchestrator``／``hour_major_provider``）
    不應該把它們送進 ``run_backward_batch``／``run_flow_domain_hour_major``：不定位初始
    元素、不查詢任何流場、不消耗任何子步、不產生任何 forcing 讀取成本。本函式只是把這個
    「零運算」決定包成與正常結果同型別的 ``BackwardDriverResult``（``final_status`` 全部
    是 ``NATIVE_STATUS_PRE_WINDOW_DEPOSITION``，位置固定在沉底起點 ``x0_m``／``y0_m``／
    ``z0_m``，``hours_run=0``，軌跡只有沉底當下這一筆），讓下游（後處理、npz 存檔、計畫檔
    摘要）可以用同一套介面消費，不需要另外特判「這個單元完全沒有結果檔」的情況。

    Args:
        row_particle_index: 這批粒子在其所屬 ``NativeWorkUnit`` 陣列裡的局部列位置
            （``0..n_unit-1``），語意與 ``BackwardDriverResult.row_particle_index`` 一致。
        x0_m, y0_m, z0_m: 逆向追蹤起點（＝沉底位置），與 ``ParticleIdentityTable`` 同一套
            公尺制座標；本函式不做任何座標轉換或有效性檢查（呼叫端已在身分表建構時驗證）。
        arrival_time_utc_ns: 這批粒子共用的到達（沉底）UTC 奈秒。
    """

    n = int(row_particle_index.shape[0])
    if n == 0:
        raise BackwardDriverError("build_pre_window_deposition_result 不可處理空批次")
    for name, array in (("x0_m", x0_m), ("y0_m", y0_m), ("z0_m", z0_m)):
        if array.shape != (n,):
            raise BackwardDriverError(f"{name} 形狀必須是 ({n},)，實際 {array.shape}")
    status = np.full(n, NATIVE_STATUS_PRE_WINDOW_DEPOSITION, dtype=np.int32)
    final_utc = np.full(n, int(arrival_time_utc_ns), dtype=np.int64)
    departures = tuple(
        DepartureRecord(
            row_particle_index=int(row),
            status=NATIVE_STATUS_PRE_WINDOW_DEPOSITION,
            x_m=float(x0_m[row]),
            y_m=float(y0_m[row]),
            z_m=float(z0_m[row]),
            utc_ns=int(arrival_time_utc_ns),
        )
        for row in range(n)
    )
    return BackwardDriverResult(
        row_particle_index=np.asarray(row_particle_index, dtype=np.int64).copy(),
        final_status=status,
        final_x_m=np.asarray(x0_m, dtype=np.float64).copy(),
        final_y_m=np.asarray(y0_m, dtype=np.float64).copy(),
        final_z_m=np.asarray(z0_m, dtype=np.float64).copy(),
        final_utc_ns=final_utc,
        trajectory_utc_ns=np.array([int(arrival_time_utc_ns)], dtype=np.int64),
        trajectory_x_m=np.asarray(x0_m, dtype=np.float64).reshape(1, n).copy(),
        trajectory_y_m=np.asarray(y0_m, dtype=np.float64).reshape(1, n).copy(),
        trajectory_z_m=np.asarray(z0_m, dtype=np.float64).reshape(1, n).copy(),
        trajectory_status=status.reshape(1, n).copy(),
        departures=departures,
        diagnostics=SubstepDiagnostics(),
        overhead_diagnostics=HourlyOverheadDiagnostics(),
        hours_run=0,
    )


def run_backward_batch(
    driver_grid: NativeBackwardDriverGrid,
    handles: BayTraceModuleHandles,
    soa: Any,
    *,
    seed_hex_values: Sequence[str],
    config: BackwardDriverConfig,
    forcing_resolver: Callable[[int], HourlyFieldPair],
    interest_region_polygon: Polygon,
    local_equals_flow: bool = True,
    local_domain_polygon: Polygon | None = None,
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
        interest_region_polygon: 流場區域（``flow_domain``）開放邊界，公尺制
            座標，與網格同一投影；離開此多邊形即終止（WP-23 起固定語意，
            見模組文件「WP-23」一節）。這批粒子必須全部屬於同一個測站
            （``run_backward_batch`` 是逐單元路徑）。
        local_equals_flow: 這批粒子所屬測站的 local domain 是否與 flow domain
            相同（``boundaries.BoundaryGeometry.local_equals_flow``，目前新竹／
            南灣／連江為 ``True``，貢寮／龜山島為 ``False``）。``True`` 時
            ``local_domain_polygon`` 必須是 ``None``（local 首次離開與 flow
            開放邊界離開是同一次穿越，直接在對應 ``DepartureRecord`` 標記
            ``also_local_domain_first_exit=True``，不呼叫子步迴圈的 local
            檢查）；``False`` 時必須提供 ``local_domain_polygon``，fail-closed
            拒絕漏傳（避免默默把貢寮／龜山島當成 local==flow 處理）。
        local_domain_polygon: 這批粒子所屬測站自己的 local domain
            （``own_local_domain``），公尺制座標；只在 ``local_equals_flow=
            False`` 時提供，離開時記錄非終止的 ``LocalDomainFirstExitRecord``
            （見 ``BackwardDriverResult.local_domain_departures``），粒子繼續
            回溯直到真正離開 ``interest_region_polygon``（flow domain）。
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
    if local_equals_flow and local_domain_polygon is not None:
        raise BackwardDriverError(
            "local_equals_flow=True 時 local domain 與 flow domain 是同一個多邊形，"
            "不可另外提供 local_domain_polygon（fail-closed，避免兩份幾何不小心不一致）"
        )
    if not local_equals_flow and local_domain_polygon is None:
        raise BackwardDriverError(
            "local_equals_flow=False 時必須提供 local_domain_polygon，fail-closed 拒絕："
            "這代表呼叫端沒有正確辨識這個測站是貢寮／龜山島這類 local≠flow 的站點"
        )

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
    local_departures: list[LocalDomainFirstExitRecord] = []
    bed_no_deposition_events: list[BedNoDepositionEventRecord] = []
    local_first_exit_pending = (
        np.ones(n, dtype=bool) if local_domain_polygon is not None else None
    )
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
            field_pair.raise_if_unresolved_gap(
                node_is_dry1=node_is_dry_check, node_is_dry2=node_is_dry_check, kbp=driver_grid.grid.kbp
            )
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
                u_smag, v_smag = zeroed_current_field2_for_smagorinsky(
                    field_pair, kbp=driver_grid.grid.kbp, node_is_dry=node_is_dry_check
                )
                hf2_now = compute_smagorinsky_kh_field(
                    driver_grid, handles,
                    u_current_mps=u_smag, v_current_mps=v_smag,
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
            local_region_polygon=local_domain_polygon,
            local_first_exit_pending=local_first_exit_pending,
            local_departures=local_departures if local_domain_polygon is not None else None,
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
                    if config.bed_no_deposition:
                        # WP-26／WP-28 敏感度案例：不終止、不寫 DepartureRecord，只
                        # 重設連續小時計數（讓粒子需要再連續停留滿門檻才會被記下
                        # 一次「不沉積」事件）並記一筆非終止的
                        # BedNoDepositionEventRecord。native_status／pa.status 完全
                        # 不變，粒子維持 ACTIVE，之後這個小時循環照常用未修改的
                        # mirror_rk4_substep_loop／batch_clamp_z 繼續驅動——見模組
                        # 文件「WP-26／WP-28」一節與 BackwardDriverConfig.
                        # bed_no_deposition docstring：這裡只是不沉積，不是反射、
                        # 也不是再懸浮動力學。
                        deposit_streak[newly_deposited] = 0
                        for row in np.flatnonzero(newly_deposited):
                            bed_no_deposition_events.append(
                                BedNoDepositionEventRecord(
                                    int(row),
                                    float(pa.x[row]), float(pa.y[row]), float(pa.z[row]), int(hour_end_utc),
                                )
                            )
                    else:
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

    if local_equals_flow:
        # WP-23：local domain 與 flow domain 是同一個多邊形時，「離開 flow
        # domain」這個終止事件本身就是「離開 local domain」——不呼叫子步迴圈
        # 的 local 檢查（上面 local_domain_polygon 恆為 None），改在這裡對
        # 每一筆 flow-exit 終止事件回填 also_local_domain_first_exit=True，
        # 避免另外產生一筆重複的 LocalDomainFirstExitRecord。
        departures = [
            replace(record, also_local_domain_first_exit=True)
            if record.status == NATIVE_STATUS_LEFT_FLOW_DOMAIN
            else record
            for record in departures
        ]

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
        local_domain_departures=tuple(local_departures),
        bed_no_deposition_events=tuple(bed_no_deposition_events),
    )
