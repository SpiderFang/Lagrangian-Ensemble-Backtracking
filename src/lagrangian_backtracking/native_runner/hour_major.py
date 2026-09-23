"""WP-9：以「小時主軸」重新排程逆向追蹤，取代逐工作單元各自重算每小時基礎場。

背景與問題
----------
WP-7／WP-8 的既有路徑（``orchestrator.plan_native_work_units`` ＋
``backward_driver.run_backward_batch``）把全案切成最多 2,500 個工作單元
（區域 × 到達時刻 × 沉降速度 × 測站），每個工作單元各自呼叫一次
``run_backward_batch``，從自己的到達時刻開始，逐小時往回走到終止。這代表
「同一個流場區域、同一個小時」的乾濕判定＋Stokes 全網格計算＋Smagorinsky
擴散係數場（即 ``backward_driver.HourlyOverheadDiagnostics`` 量測的「與粒子數
無關的固定開銷」），只要有 N 個工作單元共用該小時，就會被重算 N 次；WP-5
的 ``composed_base_cache``／``smagorinsky_cache`` 只能在**同一次
``run_backward_batch`` 呼叫序列、呼叫端主動共用同一個字典**時才有效，但
既有 ``orchestrator.run_native_plan`` 排程是「同一個 worker 依局部性群組
（``analysis_region_id``、``arrival_time_utc_ns``）批次處理」，不同到達時刻
仍是分開的 ``run_backward_batch`` 呼叫，且每個到達時刻可能跨 30 天窗口與
其他到達時刻大量重疊，實測（見 ``tests/test_native_backward_driver.py`` 的
``test_realistic_grid_scale_hourly_overhead_benchmark``）在小批次（``M=1``、
20 顆粒子）情境下，這個固定開銷可以佔單一批次總耗時的 99% 以上。

本模組的解法：對同一個流場區域，把外層迴圈從「工作單元」改成「UTC 小時」，
每個小時只呼叫一次 ``compose_current_and_stokes_fields``／
``compute_smagorinsky_kh_field``，同一小時內所有仍在追蹤中的粒子（不論屬於
哪個到達時刻或測站），只要沉降速度相同，就合併進同一次
``mirror_rk4_substep_loop`` 呼叫（沉降速度必須疊進 ``w_field``，因此仍是
唯一的「必要」分批維度，見 ``particle_input.py`` 模組文件「沉降速度與
BayTrace RK4 的相容性」一節）。這樣「固定開銷」全案只需要付一次（每個流場
區域、每個實際被追蹤到的小時各一次），而不是每個工作單元各付一次。

本模組完全不改動 ``batch_move_rk4`` 等原版 kernel（一行都沒有動），也不改
``mirror_rk4_substep_loop``／``compose_current_and_stokes_fields`` 等 WP-5
既有函式的任何行為；本模組是在**呼叫這些既有函式的順序與批次組成**上做
文章，所有物理與數值邏輯都原封不動地重用 ``backward_driver`` 模組。

四個支撐本設計「合併不改變任何粒子結果」的關鍵事實（已對照原始碼與既有
測試逐一確認，詳見模組內對應函式 docstring 或行內註解）：

1. ``estimate_substep_count`` 只讀 ``composed.u_field1``／``v_field1``
   （純海流＋Stokes 的水平分量），``apply_settling_velocity`` 只加到
   ``w_field``（垂向）——因此 ``ndeltp``／``dtb`` 只跟（流場區域、小時）
   有關，與沉降速度無關，可以在同一小時內跨全部沉降速度分組共用同一次
   計算結果。
2. BayTrace 全部 ``batch_*`` njit kernel 都以 ``if p_status[ip] != 1:
   continue`` 起手（見 ``baytrace/batch_interp.py`` 逐一核對），對
   ``status != 1`` 的列完全不讀不寫（除了 ``batch_move_rk4`` 會先寫
   ``p_x_prev[ip] = x0[ip]``，但這一行也在同一個 ``if`` 保護內）。這代表
   可以安全地把「尚未到達」與「已終止」的粒子，以 ``status=0`` 常駐在同一個
   合併陣列裡，kernel 呼又成本只隨「該次呼叫要處理的行數」而不是「陣列總
   長度」線性增加（迴圈仍會掃過全部列，但非作用中列的成本是一次整數比較，
   可忽略；詳見模組文件「已知限制」的效能誠實揭露）。
3. ``ParticleTracker.find_initial_elements``（``locate_initial_elements`` 包裝
   的原版方法）**不是**列級 ``status`` 保護的函式：它會對傳入的
   ``pa`` 整個陣列重新做 KD-tree 定位、覆寫 ``pa.element``／``pa.status``／
   ``pa.z``／``pa.u/v/w``／``pa.dhfx`` 等欄位（``tracker.py`` 606-635 行，
   逐一核對過），且 ``pa.x_prev[:] = pa.x`` 這一行更是無條件覆寫整個陣列。
   因此**絕對不能**對一個已經包含「其他到達時刻、已經在追蹤中」粒子的合併
   陣列直接呼叫這個方法——那會把進行中粒子的元素、速度、垂向位置全部用
   「起點初始化」的近似值覆寫掉。本模組的因應方式：每次只對「這一小時
   剛好到達、需要初始化」的那一小撮粒子，另外配置一個**獨立、只含這些
   粒子**的暫時 ``ParticleArrays``，呼叫 ``locate_initial_elements`` 後，
   再把結果的少數幾個欄位（``element``／``status``／``z``／``u``／``v``／
   ``w``／``dhfx``／``dhfy``／``dhfz``／``grdx``／``grdy``／``grdz``／
   ``rnds``／``x_prev``／``y_prev``）逐一寫回合併陣列的對應列，不觸碰其他
   任何列。
4. ``run_backward_batch`` 的「priming」（用合成後總速度場重新內插一次起點
   速度，見該函式 docstring）與 Smagorinsky 的 ``hf1_now``／``hf2_now``
   前後小時鏈結，兩者都是**針對單一批次自己的第一個小時**才有的特殊處理，
   對「合併批次裡摻雜了不同到達時刻」的情況並不能直接套用同一個共用變數
   （見下方「新到達粒子與已在途粒子的擴散場鏈結」一節，這是本模組實作時
   發現、且必須明確處理的正確性風險，不是事後才補的修正）。

新到達粒子與已在途粒子的擴散場鏈結（重要正確性風險，已妥善處理）
----------------------------------------------------------------------
``run_backward_batch`` 對 Smagorinsky 擴散係數場維持一個「跨小時鏈結」：
第一個小時 ``hf1_now = hf2_now``（同一小時，沒有「前一小時」可用），之後
每個小時 ``hf1_now = 前一小時的 hf2_now``。這個鏈結是**針對單一批次自己的
生命週期**定義的，不是絕對 UTC 時間軸上的鏈結。

本模組把同一個沉降速度、多個到達時刻的粒子合併進同一個持久陣列後，若直接
沿用「這個流場區域這一小時、上一小時算出的 hf2」當作全部作用中粒子共用的
``hf1``，會讓**這一小時才剛到達**的粒子（牠們自己的「第一個小時」）錯誤地
拿到「這個區域上一個小時」的擴散場，而不是標準版會給的「跟自己同一小時」
的擴散場——兩者在 ``trat=0`` 的第一個子步（``batch_interpolate_diffusivity``
在該子步的溫度內插權重完全落在 ``hf1`` 上）會讓這些粒子的第一次擴散取樣
使用錯誤的（雖然數值上通常很接近的）擴散係數場，破壞「合併與否結果必須
位元相同」的驗收要求。

解法：``mirror_rk4_substep_loop`` 的 ``hf1_field``／``hf2_field`` 是整個呼叫
共用的網格陣列，無法在同一次呼叫內對不同列給不同的 ``hf1``。因此，只有在
**同一小時、同一沉降速度分組**裡，「這一小時剛到達」與「已在追蹤中、延續
上一小時」兩種粒子同時存在、且啟用擴散（``config.enable_diffusion=True``）
時，本模組才會對該分組**拆成兩次** ``mirror_rk4_substep_loop`` 呼叫：
一次只處理剛到達的列（暫時把其餘列的 ``pa.status`` 設為 0，呼叫時
``hf1_field=hf2_field=`` 這一小時的值），另一次只處理延續中的列
（把剛到達那些列的 ``pa.status`` 暫時設為 0，呼叫時 ``hf1_field=`` 前一小時
的值），呼叫後再把暫時歸零的 ``status`` 復原。兩次呼叫合計仍然遠少於「每個
到達時刻各自一次」，只有在「新到達＋延續中同時存在＋啟用擴散」這個相對
少見的交集小時才會變成 2 次（不啟用擴散、或該小時沒有混合兩種粒子時，仍然
只有 1 次）。這是為了不犧牲正確性而接受的、經過量化的複雜度取捨，詳見
最終報告與 ``tests/test_native_hour_major.py`` 的對應驗收測試。

分批粒度與 ``NativeWorkUnit`` 的關係
--------------------------------------
本模組刻意**不**重新定義粒子分批鍵值，而是直接消費 WP-7
``orchestrator.plan_native_work_units`` 已經算好的 ``NativeWorkUnit`` 序列
（限同一個 ``flow_domain_id``）：每個 ``NativeWorkUnit`` 對應輸出裡的一個
「虛擬單元」（沿用其 ``unit_id``），本模組只是改變「怎麼把這些單元的粒子
排進 kernel 呼叫」，輸出仍然是「每個 ``NativeWorkUnit`` 一個
``backward_driver.BackwardDriverResult``」，欄位定義、``row_particle_index``
語意、``departures`` 的 ``row_particle_index``（該單元自己陣列裡的 0..n-1
局部列位置，與 ``run_backward_batch`` 完全一致）都不變，讓
``postprocess.PostprocessAccumulator``（消費 ``BackwardDriverResult`` 的既有
模組，不屬於本工作包）不需要任何修改就能接上（見模組文件「輸出相容性與
已知限制」一節的誠實揭露）。

平行化設計：一個流場區域一個行程，不做小時預取的共享記憶體管線
--------------------------------------------------------------------
工作書要求設計「一個流場區域一個 stepping 行程（至多 4 個並行）＋可選的
預取 worker 透過共享記憶體預先算好未來小時的基礎場」，或者提出並量化一個
同樣快、但更簡單的替代方案。本模組採用**後者**：只做「一個流場區域一個
行程」（直接重用 ``orchestrator.run_native_plan`` 既有的多行程執行引擎，見
``plan_native_hour_major_work_units``），不實作小時預取的共享記憶體管線。
理由（量化見最終報告，這裡摘要邏輯）：

1. 本模組已經把「同一小時、同一區域」的乾濕判定＋Stokes 全網格計算＋
   Smagorinsky 場（``HourlyOverheadDiagnostics`` 量測的固定開銷）從「每個
   工作單元各付一次」降到「每個小時只付一次」，實測（見既有
   ``test_realistic_grid_scale_hourly_overhead_benchmark``）在 60,000 節點
   規模下，讓 3 個沉降速度分組共用同一小時基礎場，總耗時從 27.9 秒降到
   9.8 秒（2.85 倍），且分組數越多，攤提效果越好（漸近於「1 次昂貴計算 +
   (G-1) 次便宜的 ``apply_settling_velocity``」）。在正式規模（每區域最多
   10 個沉降速度分組）下，這個攤提後的單一區域每小時成本，估計已經逼近
   「只算一次基礎場＋10 次便宜疊加」的下界，此時再花力氣把同一區域內的
   10 個沉降分組拆到不同行程平行執行，每個行程都要**各自重算**一次基礎場
   （除非額外實作跨行程共享記憶體，見下一點），淨效果是用 10 倍的 CPU
   核心時間換取遠小於 10 倍的實際壁鐘時間縮短（因為基礎場計算本來就已經
   只需要算一次，拆開平行反而失去共用的好處）——這是用**更多核心數**換取
   **更少壁鐘時間**的典型報酬遞減案例，量化細節見最終報告。
2. 用「跨行程共享記憶體」讓多個行程共用同一份基礎場，技術上可行
   （``multiprocessing.shared_memory``），但需要每小時做一次「主行程算好→
   寫進共享記憶體→通知子行程→子行程讀取」的交握，交握本身的實作複雜度與
   除錯難度（尤其是「子行程讀到一半、主行程正在覆寫下一小時」的競態條件、
   以及子行程崩潰時共享記憶體段的清理責任）相對本模組其餘部分明顯提高，
   而由上一點的量化分析，其可能帶來的邊際壁鐘時間改善有限，因此本模組
   判斷「不值得在這個時間點投入」，但誠實記錄為**未來可考慮的優化方向**，
   若正式規模下（4 區域、50 到達時刻、``M=10``）的實測結果顯示基礎場成本
   仍然主導總耗時（例如粒子步淘汰速度遠低於預期，30 天回溯窗大多數時間
   都有大量粒子仍在追蹤中），則這個判斷應該重新評估。
3. 剩下真正「只能用更多行程換取壁鐘時間」的槓桿，是**跨流場區域**的平行化
   （最多 4 個區域可以完全獨立平行跑，因為不同區域的網格、流場、粒子毫無
   交集）；這正是工作書要求的「一個流場區域一個 stepping 行程」，本模組
   透過 ``plan_native_hour_major_work_units`` 直接重用 ``run_native_plan``
   既有的多行程佇列引擎達成，不需要另外實作。若正式伺服器有遠多於 4 個
   核心（例如 32、64 核），的確會有核心閒置；本模組對此的誠實立場是：
   閒置的核心無法在不引入額外複雜度（見上一點）的前提下被本模組安全地
   利用，這是本設計的已知限制，不是被忽略的問題。

輸出相容性與已知限制（誠實揭露）
------------------------------------
1. **逐虛擬單元的效能診斷不再是精確量測，改為明確歸零＋另外提供區域級
   彙總**：``BackwardDriverResult.diagnostics``（``SubstepDiagnostics``）與
   ``overhead_diagnostics``（``HourlyOverheadDiagnostics``）原本是「這個
   工作單元自己耗費多少時間」的精確量測；合併呼叫之後，一次
   ``mirror_rk4_substep_loop`` 呼叫的耗時同時服務多個虛擬單元，無法在不
   失真的情況下精確拆回「這個單元自己花了多少毫秒」（若用粒子數比例分攤，
   只是一個粗略估計值，可能被誤讀成精確量測，不符合「誠實回報」原則）。
   本模組因此讓每個虛擬單元的 ``diagnostics``／``overhead_diagnostics``
   保持欄位全零（``SubstepDiagnostics()``／``HourlyOverheadDiagnostics()``
   的預設值），並改由 ``run_flow_domain_hour_major`` 的回傳值另外提供一份
   **整個區域彙總**的 ``HourlyOverheadDiagnostics`` 與核心呼叫次數統計
   （``HourMajorRunResult.domain_overhead_diagnostics``／
   ``domain_kernel_call_count``），供效能報告使用。若下游程式碼假設這兩個
   欄位一定非零，需要另行確認並調整（本工作包未發現既有測試依賴這個
   假設，但 ``postprocess.py`` 不屬於本工作包範圍，需要另行確認）。
2. **合併陣列常駐已終止粒子的迴圈開銷**：見上方「支撐事實 2」，
   ``status=0`` 的列雖然不會被 kernel 實際處理，但 njit 迴圈仍會逐列掃過
   做一次 ``status`` 檢查，因此極端情況下（例如某個沉降速度分組裡，大多數
   粒子早早就終止、只剩極少數還在追蹤 30 天上限）合併陣列的「陣列長度」
   本身會變成一個固定的、隨全案粒子數而非「目前仍在追蹤的粒子數」增長的
   迴圈開銷。本模組不做「定期壓縮陣列、剔除已終止列」的優化，這是已知、
   可能在極端情境下侵蝕部分效能收益的限制，量化見最終報告的效能實測。
3. **到達時刻必須精確落在小時邊界**：本模組的小時主軸迴圈假設每顆粒子的
   ``arrival_time_utc_ns`` 恰好等於某個小時邊界（與 ``arrival_times.py``
   的既有保證一致，見該模組 docstring），並在 ``build_hour_major_domain_batch``
   建構期間 fail-closed 驗證這件事，不假設呼叫端一定正確傳入。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import Polygon

from .backward_driver import (
    NATIVE_STATUS_ACTIVE,
    NATIVE_STATUS_DEPOSITED,
    NATIVE_STATUS_FORCING_MISSING,
    NATIVE_STATUS_LEFT_FLOW_DOMAIN,
    NATIVE_STATUS_MAX_BACKTRACK,
    NATIVE_STATUS_NUMERICAL_FAILURE,
    NATIVE_STATUS_STRANDED,
    BackwardDriverConfig,
    BackwardDriverResult,
    BayTraceModuleHandles,
    ComposedCurrentAndStokesFields,
    ComposedFieldEndpoint,
    DepartureRecord,
    HourlyOverheadDiagnostics,
    LocalCflSpatialIndex,
    LocalDomainFirstExitRecord,
    NativeBackwardDriverGrid,
    SubstepDiagnostics,
    WaveFieldAtNodes,
    apply_settling_velocity,
    build_local_cfl_spatial_index,
    compose_current_and_stokes_fields,
    compute_smagorinsky_kh_field,
    compute_wet_dry_flags,
    derive_particle_diffusion_seed_lo_hi_u64,
    estimate_substep_count,
    estimate_substep_count_local_cfl,
    locate_initial_elements,
    mirror_rk4_substep_loop,
    particle_bed_z_levels,
    zeroed_current_field2_for_smagorinsky,
)
from .forcing_bridge import HourlyFieldPair, UnresolvedForcingGapError
from .orchestrator import NativeWorkUnit
from .particle_input import ParticleIdentityTable
from .substep_numba import mirror_rk4_substep_loop_numba

__all__ = [
    "HourMajorError",
    "HourMajorConfig",
    "HourMajorDomainBatch",
    "HourMajorSettlingGroupInput",
    "HourMajorUnitInput",
    "build_hour_major_domain_batch",
    "HourMajorRunResult",
    "run_flow_domain_hour_major",
    "save_hour_major_checkpoint",
    "load_hour_major_checkpoint",
    "hour_major_config_to_backward_driver_config",
]


_HOUR_NS = 3_600_000_000_000


class HourMajorError(RuntimeError):
    """本模組專用例外，與 ``BackwardDriverError``／``NativeOrchestratorError`` 區分責任層。"""


# ══════════════════════════════════════════════════════════════════════════
# 1. 設定：與 BackwardDriverConfig 對應，但是「整個流場區域」共用一份
# ══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class HourMajorConfig:
    """整個流場區域一次小時主軸執行共用的設定。

    欄位語意與 ``backward_driver.BackwardDriverConfig`` 逐一對應（該類別是
    「單一到達時刻＋單一沉降速度」批次的設定，本類別是「整個流場區域、
    全部到達時刻與沉降速度」共用的設定，兩者在驗收測試裡必須填入完全相同的
    數值，才能比較「合併」與「逐單元」兩條路徑是否位元相同，見
    ``hour_major_config_to_backward_driver_config``）。
    """

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
    # 水平擴散係數（Smagorinsky，工作書式(10)）的計算實作：
    # ``"numba"``（預設）＝``smagorinsky_batch`` 的 Numba 移植，同一演算法、同一運算式，但浮點
    # 加總順序不同，與原版差 1–2 個最後有效位元（真實東北網格實測最大相對差 7e-16），屬統計
    # 等價而非位元相同；東北網格每小時 0.27 秒（原版 11.67 秒），且幾乎不產生暫存陣列。
    # ``"baytrace"``＝透過 shim 呼叫 BayTrace 原版 ``_compute_smagorinsky_hf2``（需要與原版
    # 位元相同時使用）。專案決定：預設採運算最快的版本。
    smagorinsky_impl: str = "numba"
    # WP-16 任務 1：CFL 子步數估計政策。
    # ``"local_cfl"``（預設）＝``estimate_substep_count_local_cfl``，只看這一小時活動粒子
    # 鄰域內的元素，對含極小近岸元素的網格遠比全網格版本省子步數（見最終報告量測）；仍保證
    # ``dtb<=dt_max_seconds``，且行為與統計等價測試證明的舊行為一致（不是位元相同，因為
    # ``ndeltp`` 本身通常會變小）。
    # ``"global_cfl"``＝``estimate_substep_count``（WP-5 既有版本，完全未修改），用全網格
    # 最大速度與最短邊，是本欄位新增前的唯一行為，與新增前逐位元相同，供需要與舊版本比對、
    # 或不信任局部 CFL 假設的場景使用。
    substep_policy: str = "local_cfl"
    # WP-16 任務 2：子步迴圈實作。
    # ``"numba"``（預設）＝``substep_numba.mirror_rk4_substep_loop_numba``，整個 ndeltp 迴圈
    # （含計數器型亂數、水平擴散、關注區域判定、對 BayTrace 原版批次核心的呼叫）融合成單一
    # ``@njit`` 呼叫，消除逐子步的 Python 直譯器開銷；與 ``"python"`` 統計等價但非位元相同
    # （SplitMix64 整數部分位元相同，Box-Muller 的 log/cos/sin 與關注區域判定的射線法各有
    # ulp 級／測度為零邊界情況的已知差異，見 substep_numba.py 模組文件與最終報告）。
    # ``"python"``＝``backward_driver.mirror_rk4_substep_loop``（WP-5 既有版本，完全未修改），
    # 本欄位新增前的唯一行為，與新增前逐位元相同。
    substep_loop_impl: str = "numba"


def hour_major_config_to_backward_driver_config(
    config: HourMajorConfig, *, arrival_time_utc_ns: int, settling_velocity_mps: float
) -> BackwardDriverConfig:
    """把區域級設定轉成單一子批次的 ``BackwardDriverConfig``。

    供驗收測試建構「單獨呼叫 ``run_backward_batch``」的對照組時使用，確保
    兩條路徑除了 ``arrival_time_utc_ns``／``settling_velocity_mps``（子批次
    本來就各自不同的鍵值）以外，其餘設定逐項相同。
    """

    return BackwardDriverConfig(
        arrival_time_utc_ns=arrival_time_utc_ns,
        settling_velocity_mps=settling_velocity_mps,
        max_backtrack_seconds=config.max_backtrack_seconds,
        dt_max_seconds=config.dt_max_seconds,
        cfl_number=config.cfl_number,
        h0=config.h0,
        horcon=config.horcon,
        enable_diffusion=config.enable_diffusion,
        output_interval_seconds=config.output_interval_seconds,
        deposition_consecutive_hours=config.deposition_consecutive_hours,
        deposition_tolerance_m=config.deposition_tolerance_m,
        master_seed=config.master_seed,
        stokes_depth_threshold_ratio=config.stokes_depth_threshold_ratio,
    )


# ══════════════════════════════════════════════════════════════════════════
# 2. 輸入：從既有 NativeWorkUnit ＋ ParticleIdentityTable 建構合併批次
# ══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class HourMajorUnitInput:
    """一個虛擬單元（對應一個 ``NativeWorkUnit``）在合併陣列裡的位置與身分。

    ``start``／``stop`` 是這個單元的粒子在其所屬沉降速度分組的合併陣列裡
    佔用的列區間（左閉右開，``[start, stop)``），建構時依單元逐一串接，
    因此同一個單元的列永遠連續，方便切片還原成獨立的
    ``BackwardDriverResult``。
    """

    unit_id: str
    arrival_time_utc_ns: int
    study_site_id: str
    start: int
    stop: int
    row_particle_index: np.ndarray


@dataclass(frozen=True, slots=True)
class HourMajorSettlingGroupInput:
    """一個沉降速度分組的合併粒子輸入（可能來自多個到達時刻／測站）。

    ``max_backtrack_ns``：WP-17 新增，逐列（逐粒子）回溯時間上限（奈秒）。因為一個
    分組可能合併多個到達時刻（見 ``arrival_time_utc_ns`` 的說明），而不同到達時刻
    在 ``fixed_calendar_window`` 模式下可能有不同的沉底年齡、因而有不同的有效回溯
    時間（``H - age``），所以這個欄位必須是逐列陣列，不能像 ``HourMajorConfig.
    max_backtrack_seconds`` 一樣只是整個流場區域共用一個純量。值為 ``-1`` 表示
    「這個單元沒有明示值」（即 WP-17 之前建構、或直接由測試建構、``resolved_max_
    backtrack_seconds=None`` 的 ``NativeWorkUnit``）——``run_flow_domain_hour_major``
    在每個小時的回溯上限檢查時，遇到 ``-1`` 才會退回讀取自己收到的
    ``config.max_backtrack_seconds``（見該函式「步驟 1」），刻意不在
    ``build_hour_major_domain_batch`` 這裡就烤進退回值：這個函式沒有 ``config``
    參數，若在這裡就展開退回值，任何呼叫端忘記同步傳入 ``config.max_backtrack_
    seconds`` 都會被靜默用錯的預設值（``HourMajorConfig()`` 的 class 預設 30 天）
    覆蓋掉自己實際配置的值，而不是在明顯的地方壞掉。
    """

    settling_velocity_mps: float
    x0_m: np.ndarray
    y0_m: np.ndarray
    z0_m: np.ndarray
    seed_hex: np.ndarray
    arrival_time_utc_ns: np.ndarray
    max_backtrack_ns: np.ndarray
    interest_region_polygons: np.ndarray
    local_region_polygons: np.ndarray
    """WP-23 新增。``(n,)`` object 陣列，每顆粒子所屬測站自己的 local domain
    （``own_local_domain``）多邊形；語意與 ``interest_region_polygons``
    （WP-23 起代表 flow domain）對應但用途不同，見
    ``backward_driver.mirror_rk4_substep_loop`` 的 ``local_region_polygons``
    參數。``local_equals_flow_mask`` 為真的列，這裡仍然填有效多邊形物件
    （與該列的 flow domain 相同，保持陣列型別穩定），但**不會**被子步迴圈
    實際判定——``_new_group_state`` 把這些列的 ``local_first_exit_pending``
    初始值設成 ``False``，子步迴圈的 check_mask 天生就會跳過它們（見該
    docstring），終止時改由 ``_finalize_unit_result`` 對它們的 flow-exit
    ``DepartureRecord`` 回填 ``also_local_domain_first_exit=True``。"""
    local_equals_flow_mask: np.ndarray
    """WP-23 新增。``(n,)`` bool 陣列，每顆粒子所屬測站的 local domain 是否與
    flow domain 相同（``resources.DomainResources.local_equals_flow``）。一個
    ``HourMajorUnitInput`` 對應單一測站，因此同一個單元底下的列這個值必為
    常數。"""
    units: tuple[HourMajorUnitInput, ...]

    @property
    def particle_count(self) -> int:
        return int(self.x0_m.shape[0])

    @property
    def needs_local_domain_check(self) -> bool:
        """這個分組裡是否至少有一顆粒子屬於 ``local_equals_flow=False`` 的測站
        （目前是貢寮／龜山島）。為假時（例如整個分組全部來自新竹／南灣／
        連江）呼叫端應該完全不傳遞 local domain 相關參數給子步迴圈——不是
        傳一個「反正也不會命中」的陣列，是刻意完全不啟用那段程式碼路徑，
        確保這三站在新程式下的行為與 WP-23 之前逐位元等價（見
        ``reports/WP-23.md`` 等價性論證一節）。"""

        return bool(np.any(~self.local_equals_flow_mask))


@dataclass(frozen=True, slots=True)
class HourMajorDomainBatch:
    """一個流場區域的完整小時主軸輸入：依沉降速度分組後的全部粒子。"""

    flow_domain_id: str
    groups: tuple[HourMajorSettlingGroupInput, ...]

    @property
    def max_arrival_time_utc_ns(self) -> int:
        return max(int(np.max(g.arrival_time_utc_ns)) for g in self.groups)


def build_hour_major_domain_batch(
    table: ParticleIdentityTable,
    units: Sequence[NativeWorkUnit],
    *,
    interest_regions: Mapping[str, Polygon],
    local_domains: Mapping[str, Polygon],
    local_equals_flow: Mapping[str, bool],
) -> HourMajorDomainBatch:
    """把同一個流場區域的一批 ``NativeWorkUnit`` 轉成小時主軸輸入。

    Args:
        table: 這些工作單元的粒子列位置所屬的身分表（提供 ``x0_m``／
            ``y0_m``／``z0_m``／``seed_hex``）。
        units: 必須全部屬於同一個 ``flow_domain_id``，否則 fail-closed
            拒絕（不同流場區域的粒子必須分開呼叫
            ``run_flow_domain_hour_major``，本函式不代為合併）。呼叫端
            不應該把 ``unit.pre_window_deposition=True`` 的單元傳進來
            （見 ``hour_major_provider._run_unit_entrypoint``：這些單元
            完全不進入小時主軸批次，改用
            ``backward_driver.build_pre_window_deposition_result`` 直接
            組出結果）；若真的傳入會 fail-closed 拒絕，避免默默把
            pre-window 粒子當成一般粒子運算。
        interest_regions: ``{study_site_id: flow domain 開放邊界多邊形}``
            （WP-23 起固定語意，見 ``resources.DomainResources.
            interest_regions`` docstring），來自
            ``resources.DomainResources.interest_regions``；離開此多邊形
            即終止。任何單元的 ``study_site_id`` 不在這個對照表裡都會
            fail-closed 拒絕。
        local_domains: WP-23 新增。``{study_site_id: 該測站自己的 local
            domain 多邊形（own_local_domain）}``，來自
            ``resources.DomainResources.local_domains``；離開此多邊形是
            **非終止**事件，只記錄、不停止（見
            ``backward_driver.LocalDomainFirstExitRecord``）。
        local_equals_flow: WP-23 新增。``{study_site_id: 該測站 local
            domain 是否與 flow domain 相同}``，來自
            ``resources.DomainResources.local_equals_flow``。

    Raises:
        HourMajorError: ``units`` 為空、跨流場區域、到達時刻未精確落在
            小時邊界、含 pre-window 單元，或缺少對應的關注區域／local
            domain 多邊形。
    """

    if not units:
        raise HourMajorError("units 不可為空")
    for unit in units:
        if unit.pre_window_deposition:
            raise HourMajorError(
                f"單元 {unit.unit_id} 是 pre_window_deposition（固定日曆回溯窗之外的沉底），"
                "不可送入 build_hour_major_domain_batch；呼叫端必須先過濾掉這些單元，改用"
                "backward_driver.build_pre_window_deposition_result 直接組出結果"
            )

    domain_ids = {u.flow_domain_id for u in units}
    if len(domain_ids) != 1:
        raise HourMajorError(
            f"build_hour_major_domain_batch 只接受單一 flow_domain_id，實際發現 {sorted(domain_ids)}"
        )
    flow_domain_id = next(iter(domain_ids))

    for unit in units:
        if int(unit.arrival_time_utc_ns) % _HOUR_NS != 0:
            raise HourMajorError(
                f"單元 {unit.unit_id} 的到達時刻 {unit.arrival_time_utc_ns} 未精確落在小時邊界，"
                "小時主軸排程要求到達時刻必須恰為整點（見 arrival_times.py 的既有保證）"
            )
        if unit.study_site_id not in interest_regions:
            raise HourMajorError(
                f"單元 {unit.unit_id} 的測站 {unit.study_site_id!r} 沒有對應的關注區域多邊形"
            )
        if unit.study_site_id not in local_domains or unit.study_site_id not in local_equals_flow:
            raise HourMajorError(
                f"單元 {unit.unit_id} 的測站 {unit.study_site_id!r} 沒有對應的 local domain "
                "多邊形或 local_equals_flow 旗標（WP-23 起 fail-closed 必要）"
            )

    by_settling: dict[float, list[NativeWorkUnit]] = {}
    for unit in units:
        by_settling.setdefault(float(unit.settling_velocity_mps), []).append(unit)

    groups: list[HourMajorSettlingGroupInput] = []
    for settling in sorted(by_settling):
        group_units = sorted(
            by_settling[settling], key=lambda u: (u.arrival_time_utc_ns, u.study_site_id, u.unit_id)
        )
        x0_parts: list[np.ndarray] = []
        y0_parts: list[np.ndarray] = []
        z0_parts: list[np.ndarray] = []
        seed_parts: list[np.ndarray] = []
        arrival_parts: list[np.ndarray] = []
        max_backtrack_parts: list[np.ndarray] = []
        polygon_parts: list[np.ndarray] = []
        local_polygon_parts: list[np.ndarray] = []
        local_equals_flow_parts: list[np.ndarray] = []
        unit_inputs: list[HourMajorUnitInput] = []
        cursor = 0
        for unit in group_units:
            rows = unit.particle_row_indices
            n_unit = int(rows.shape[0])
            x0_parts.append(table.x0_m[rows])
            y0_parts.append(table.y0_m[rows])
            z0_parts.append(table.z0_m[rows])
            seed_parts.append(table.seed_hex[rows])
            arrival_parts.append(np.full(n_unit, int(unit.arrival_time_utc_ns), dtype=np.int64))
            unit_max_backtrack_ns = (
                int(round(unit.resolved_max_backtrack_seconds * 1.0e9))
                if unit.resolved_max_backtrack_seconds is not None
                else -1
            )
            max_backtrack_parts.append(np.full(n_unit, unit_max_backtrack_ns, dtype=np.int64))
            polygon = interest_regions[unit.study_site_id]
            polygon_parts.append(np.full(n_unit, polygon, dtype=object))
            local_polygon = local_domains[unit.study_site_id]
            local_polygon_parts.append(np.full(n_unit, local_polygon, dtype=object))
            local_equals_flow_parts.append(
                np.full(n_unit, bool(local_equals_flow[unit.study_site_id]), dtype=bool)
            )
            unit_inputs.append(
                HourMajorUnitInput(
                    unit_id=unit.unit_id,
                    arrival_time_utc_ns=int(unit.arrival_time_utc_ns),
                    study_site_id=unit.study_site_id,
                    start=cursor,
                    stop=cursor + n_unit,
                    row_particle_index=rows.copy(),
                )
            )
            cursor += n_unit
        groups.append(
            HourMajorSettlingGroupInput(
                settling_velocity_mps=settling,
                x0_m=np.concatenate(x0_parts),
                y0_m=np.concatenate(y0_parts),
                z0_m=np.concatenate(z0_parts),
                seed_hex=np.concatenate(seed_parts),
                arrival_time_utc_ns=np.concatenate(arrival_parts),
                max_backtrack_ns=np.concatenate(max_backtrack_parts),
                interest_region_polygons=np.concatenate(polygon_parts),
                local_region_polygons=np.concatenate(local_polygon_parts),
                local_equals_flow_mask=np.concatenate(local_equals_flow_parts),
                units=tuple(unit_inputs),
            )
        )

    return HourMajorDomainBatch(flow_domain_id=flow_domain_id, groups=tuple(groups))


# ══════════════════════════════════════════════════════════════════════════
# 3. 執行期狀態：每個沉降速度分組一個持久 ParticleArrays ＋逐單元進度
# ══════════════════════════════════════════════════════════════════════════


@dataclass
class _UnitProgress:
    """一個虛擬單元在小時主軸推進過程中持續更新的進度與輸出緩衝。"""

    info: HourMajorUnitInput
    finished: bool = False
    hours_run: int = 0
    next_output_utc_ns: int = 0
    trajectory_utc: list[int] = field(default_factory=list)
    trajectory_x: list[np.ndarray] = field(default_factory=list)
    trajectory_y: list[np.ndarray] = field(default_factory=list)
    trajectory_z: list[np.ndarray] = field(default_factory=list)
    trajectory_status: list[np.ndarray] = field(default_factory=list)
    departures: list[DepartureRecord] = field(default_factory=list)
    local_departures: list[LocalDomainFirstExitRecord] = field(default_factory=list)
    """WP-23 新增：這個單元底下、``local_equals_flow=False`` 的粒子首次離開
    自己 local domain 的非終止事件。``local_equals_flow=True`` 的單元這裡
    永遠是空 list——那些單元的 local 首次離開改由 ``departures`` 對應那筆
    ``DepartureRecord.also_local_domain_first_exit=True`` 表達，見
    ``_finalize_unit_result``。"""


@dataclass
class _GroupState:
    """一個沉降速度分組的完整執行期狀態（持久陣列＋逐單元進度）。"""

    inp: HourMajorSettlingGroupInput
    pa: Any
    native_status: np.ndarray
    released: np.ndarray
    particle_seed_lo: np.ndarray | None
    particle_seed_hi: np.ndarray | None
    particle_step_counter: np.ndarray | None
    deposit_streak: np.ndarray
    local_first_exit_pending: np.ndarray
    """WP-23 新增。``(n,)`` bool，跨小時持續傳入子步迴圈的持久陣列（語意見
    ``backward_driver.mirror_rk4_substep_loop`` 同名參數）。初始值是
    ``~inp.local_equals_flow_mask``：``local_equals_flow=True`` 的列一開始
    就是 ``False``（不需要 local 判定，子步迴圈的 check_mask 天生跳過），
    ``local_equals_flow=False`` 的列初始為 ``True``，離開自己 local domain
    後被子步迴圈翻成 ``False``（全程只翻一次）。"""
    prev_hf2: np.ndarray | None = None
    prev_hf2_utc_ns: int | None = None
    units: dict[str, _UnitProgress] = field(default_factory=dict)


def _new_group_state(
    handles: BayTraceModuleHandles, inp: HourMajorSettlingGroupInput, *, config: HourMajorConfig
) -> _GroupState:
    n = inp.particle_count
    pa = handles.particle_arrays_cls(n)
    pa.x[:] = inp.x0_m
    pa.y[:] = inp.y0_m
    pa.z[:] = inp.z0_m
    pa.z0[:] = inp.z0_m
    pa.element[:] = -1
    pa.status[:] = 0

    particle_seed_lo = particle_seed_hi = particle_step_counter = None
    if config.enable_diffusion:
        particle_seed_lo, particle_seed_hi = derive_particle_diffusion_seed_lo_hi_u64(
            [str(value) for value in inp.seed_hex], master_seed=config.master_seed
        )
        particle_step_counter = np.zeros(n, dtype=np.int64)

    output_interval_ns = int(round(config.output_interval_seconds * 1.0e9))
    units = {
        u.unit_id: _UnitProgress(info=u, next_output_utc_ns=u.arrival_time_utc_ns - output_interval_ns)
        for u in inp.units
    }

    return _GroupState(
        inp=inp,
        pa=pa,
        native_status=np.full(n, NATIVE_STATUS_ACTIVE, dtype=np.int32),
        released=np.zeros(n, dtype=bool),
        particle_seed_lo=particle_seed_lo,
        particle_seed_hi=particle_seed_hi,
        particle_step_counter=particle_step_counter,
        deposit_streak=np.zeros(n, dtype=np.int64),
        local_first_exit_pending=~inp.local_equals_flow_mask,
        units=units,
    )


# ══════════════════════════════════════════════════════════════════════════
# 4. 核心迴圈
# ══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class HourMajorRunResult:
    """一次 ``run_flow_domain_hour_major`` 呼叫的完整輸出。

    ``unit_results`` 的每個值都是完整的 ``BackwardDriverResult``，除了
    ``diagnostics``／``overhead_diagnostics`` 兩個效能診斷欄位固定為全零
    （見模組文件「輸出相容性與已知限制」第 1 點），其餘欄位（最終位置、
    狀態、軌跡、departures、``hours_run``）與獨立呼叫
    ``run_backward_batch`` 對同一子批次得到的結果應該位元相同（診斷等於
    的驗收見 ``tests/test_native_hour_major.py``）。
    """

    unit_results: Mapping[str, BackwardDriverResult]
    domain_overhead_diagnostics: HourlyOverheadDiagnostics
    domain_kernel_call_count: int
    domain_hours_walked: int


def _finalize_unit_result(group: _GroupState, progress: _UnitProgress) -> BackwardDriverResult:
    """組出一個虛擬單元的最終 ``BackwardDriverResult``。

    ``final_status``／``final_x_m``／``final_y_m``／``final_z_m`` 直接讀
    ``group.pa``／``group.native_status`` 目前的值，**不是**
    ``progress.trajectory_*[-1]``：鏡像 ``run_backward_batch`` 的
    ``final_status = native_status.copy()``／``final_x_m = pa.x.copy()``
    在 while 迴圈結束「當下」直接取值，不是取最後一次有取樣進 trajectory
    的那一筆——回溯上限／缺值缺口都是在本小時任何移動、任何軌跡取樣
    **之前**就 ``break``，此時 ``trajectory_status[-1]`` 還停在終止前一小時
    的 ``ACTIVE``，只有直接讀當下的 ``native_status``／``pa.x`` 才會是
    正確的終止狀態與終止位置（見驗收 1／4 的臭蟲修正說明）。這個讀法之所以
    安全，是因為粒子一旦終止（``native_status`` 不再是 ``ACTIVE``），本模組
    後續不會再對它的列做任何寫入（所有 batch kernel 都以 ``status==1`` 為
    作用中條件，見模組文件支撐事實 1），因此呼叫本函式時（一定是在整個
    分組的主迴圈已經跑完、或 ``_max_master_hours`` 測試鉤子提前中止之後）
    這些列的值就是它終止當下凍結的最終值。
    """

    info = progress.info
    if not progress.trajectory_utc:
        raise HourMajorError(f"單元 {info.unit_id} 從未被排入任何一個小時，這是內部錯誤")
    start, stop = info.start, info.stop

    # WP-23：一個單元只對應單一測站，因此 local_equals_flow 在這個列區間內
    # 必為常數——用 all()／any() 都應該得到相同答案，這裡用 all() 並在不一致
    # 時 fail-closed，防止 build_hour_major_domain_batch 未來被改壞、悄悄把
    # 不同測站的列混進同一個單元卻沒被發現。
    unit_local_equals_flow_slice = group.inp.local_equals_flow_mask[start:stop]
    if bool(np.all(unit_local_equals_flow_slice)) != bool(np.any(unit_local_equals_flow_slice)):
        raise HourMajorError(
            f"單元 {info.unit_id} 底下的列 local_equals_flow 不一致，這違反「一個單元只屬於"
            "單一測站」的假設，是內部錯誤"
        )
    unit_local_equals_flow = (
        bool(unit_local_equals_flow_slice[0]) if unit_local_equals_flow_slice.size > 0 else True
    )

    departures = list(progress.departures)
    if unit_local_equals_flow:
        # local domain 與 flow domain 是同一個多邊形：這個單元底下不會（也不應該）
        # 產生獨立的 LocalDomainFirstExitRecord（group.local_first_exit_pending
        # 對這些列從一開始就是 False，子步迴圈天生跳過），改對每一筆 flow-exit
        # 終止事件回填 also_local_domain_first_exit=True，語意與
        # run_backward_batch 的同名分支完全對應。
        departures = [
            replace(record, also_local_domain_first_exit=True)
            if record.status == NATIVE_STATUS_LEFT_FLOW_DOMAIN
            else record
            for record in departures
        ]

    departure_by_row: dict[int, DepartureRecord] = {}
    for record in departures:
        departure_by_row[record.row_particle_index] = record
    n_unit = int(info.row_particle_index.shape[0])
    final_utc_ns = np.full(n_unit, progress.trajectory_utc[-1], dtype=np.int64)
    for row, record in departure_by_row.items():
        final_utc_ns[row] = record.utc_ns
    return BackwardDriverResult(
        row_particle_index=info.row_particle_index.copy(),
        final_status=group.native_status[start:stop].copy(),
        final_x_m=group.pa.x[start:stop].copy(),
        final_y_m=group.pa.y[start:stop].copy(),
        final_z_m=group.pa.z[start:stop].copy(),
        final_utc_ns=final_utc_ns,
        trajectory_utc_ns=np.asarray(progress.trajectory_utc, dtype=np.int64),
        trajectory_x_m=np.stack(progress.trajectory_x, axis=0),
        trajectory_y_m=np.stack(progress.trajectory_y, axis=0),
        trajectory_z_m=np.stack(progress.trajectory_z, axis=0),
        trajectory_status=np.stack(progress.trajectory_status, axis=0),
        departures=tuple(departures),
        diagnostics=SubstepDiagnostics(),
        overhead_diagnostics=HourlyOverheadDiagnostics(),
        hours_run=progress.hours_run,
        local_domain_departures=tuple(progress.local_departures),
    )


def _record_departures(
    group: _GroupState, rows: np.ndarray, *, status: int, utc_ns: int
) -> None:
    """把一批列位置標記為終止，並把 departure 記錄分派到各自所屬的單元。

    ``rows`` 是分組合併陣列裡的列位置（``0..n_group-1``）；
    ``DepartureRecord.row_particle_index`` 依 ``run_backward_batch`` 的既有
    語意，必須是「該單元自己陣列裡」的局部列位置（``row - unit.start``），
    不是分組合併陣列裡的列位置，見模組文件「分批粒度與 NativeWorkUnit 的
    關係」一節。
    """

    if rows.size == 0:
        return
    group.native_status[rows] = status
    group.pa.status[rows] = 0
    for row in rows:
        row = int(row)
        for progress in group.units.values():
            if progress.info.start <= row < progress.info.stop:
                progress.departures.append(
                    DepartureRecord(
                        row_particle_index=row - progress.info.start,
                        status=status,
                        x_m=float(group.pa.x[row]),
                        y_m=float(group.pa.y[row]),
                        z_m=float(group.pa.z[row]),
                        utc_ns=int(utc_ns),
                    )
                )
                break


@dataclass(frozen=True, slots=True)
class _FieldEndpointIdentity:
    """WP-14 任務 B：辨識「這是不是同一個流場端點」的身分鍵。

    只記錄可以廉價比對、且足以代表「同一份已核准產品資料」的欄位：流場
    時刻（UTC 奈秒）、月份 ID、資料來源（``observed``／``reconstructed``，
    見 ``forcing_bridge.HourlyFieldPair`` 的 ``month_id``／``time_origin``
    docstring），以及（若有提供波浪場）波浪時刻。刻意**不**比對實際陣列內容
    ——逐點比對陣列內容的成本與直接重算 Stokes 相近，會抵銷重用的效益；
    改用「同一個 UTC 時刻 + 同一份月份資料 + 同一種來源」代表「這兩次呼叫
    ``forcing_resolver``／``wave_field_resolver`` 一定回傳相同的陣列內容」
    ——這是對 ``forcing_resolver``／``wave_field_resolver`` 必須是純函式
    （同樣輸入必然得到同樣輸出，不依賴呼叫次序或隱藏狀態）的信賴，也是
    ``AGENTS.md`` 對已驗收產品「版本化重建方法」與 provenance 要求的自然
    延伸。真正決定能不能安全重用的「數值」判斷仍然是
    ``compose_current_and_stokes_fields`` 內部對 ``node_is_dry`` 的逐位元
    比對（見該函式 docstring），這裡的身分鍵只是「值不值得去試」的第一層
    篩選，篩選錯了也不會產生錯誤結果，只是浪費一次白算的重用嘗試。
    """

    utc_ns: int
    month_id: str
    time_origin: str
    wave_utc_ns: int | None


def _field1_identity(field_pair: HourlyFieldPair, *, has_wave: bool) -> _FieldEndpointIdentity:
    return _FieldEndpointIdentity(
        utc_ns=int(field_pair.time_before_utc_ns),
        month_id=field_pair.month_id1,
        time_origin=field_pair.time_origin1,
        wave_utc_ns=int(field_pair.time_before_utc_ns) if has_wave else None,
    )


def _field2_identity(field_pair: HourlyFieldPair, *, has_wave: bool) -> _FieldEndpointIdentity:
    return _FieldEndpointIdentity(
        utc_ns=int(field_pair.time_after_utc_ns),
        month_id=field_pair.month_id2,
        time_origin=field_pair.time_origin2,
        wave_utc_ns=int(field_pair.time_after_utc_ns) if has_wave else None,
    )


def run_flow_domain_hour_major(
    driver_grid: NativeBackwardDriverGrid,
    handles: BayTraceModuleHandles,
    batch: HourMajorDomainBatch,
    *,
    config: HourMajorConfig,
    forcing_resolver: Callable[[int], HourlyFieldPair],
    wave_field_resolver: Callable[[int], WaveFieldAtNodes] | None = None,
    checkpoint_dir: str | Path | None = None,
    checkpoint_every_hours: int = 24,
    _max_master_hours: int | None = None,
    stokes_pool: Any | None = None,
    stokes_worker_count: int = 1,
) -> HourMajorRunResult:
    """以小時主軸推進一個流場區域的全部沉降速度分組，直到全部單元終止。

    Args:
        batch: ``build_hour_major_domain_batch`` 的輸出，同一個流場區域的
            全部沉降速度分組。
        config: 見 ``HourMajorConfig``。
        forcing_resolver／wave_field_resolver: 語意與
            ``run_backward_batch`` 完全相同，見該函式 docstring；本函式
            每個小時只呼叫一次（不是每個沉降速度分組各呼叫一次）。
        checkpoint_dir: 提供時，每 ``checkpoint_every_hours`` 個小時（且在
            正常結束或提前中止時）把完整執行期狀態寫進這個目錄；下次以
            相同 ``checkpoint_dir`` 呼叫本函式時，若目錄內已有可用的
            checkpoint，會先讀回並從中斷處繼續，而不是從頭開始。續跑後的
            最終結果與未中斷執行必須位元相同（見
            ``tests/test_native_hour_major.py`` 的續跑驗收）。
        _max_master_hours: 僅供測試使用：處理滿這麼多個「主迴圈小時」後
            立即停止（並寫一次 checkpoint，若有提供 ``checkpoint_dir``），
            用於模擬「執行到一半被中斷」，驗證續跑的位元相同性。正式用途
            不應該傳入這個參數（``None`` 代表跑到全部單元終止為止）。
        stokes_pool／stokes_worker_count: 原樣轉交給每小時呼叫一次的
            ``compose_current_and_stokes_fields``（見該函式與
            ``backward_driver`` 模組頂端的設計說明）：把史托克斯漂移
            （Stokes drift）逐點求解切成 ``stokes_worker_count`` 塊、分派到
            呼叫端預先建立好的行程池（process pool，``stokes_pool``）平行
            計算，藉由行程重用避開「每小時建立/銷毀行程池」的固定開銷。
            由於每個節點的史托克斯解互相獨立、沒有跨節點的化簡
            （reduction），切塊方式不影響數值結果，因此無論
            ``stokes_worker_count`` 是多少、或根本不提供 ``stokes_pool``
            （沿用單一行程逐點計算），輸出必須位元相同——這個等價性由
            ``tests/test_native_hour_major.py`` 驗證。呼叫端（目前是
            ``hour_major_provider``）決定要不要建立、何時建立行程池；本函式
            不負責行程池的生命週期管理。預設 ``stokes_pool=None``、
            ``stokes_worker_count=1``，即維持 WP-13 之前逐點單一行程執行的
            行為。

    Returns:
        ``HourMajorRunResult``。
    """

    output_interval_ns = int(round(config.output_interval_seconds * 1.0e9))
    if output_interval_ns <= 0:
        raise HourMajorError("output_interval_seconds 必須是正值")
    # WP-17：``config.max_backtrack_seconds`` 只在分組的 ``max_backtrack_ns`` 陣列
    # 有 ``-1`` sentinel（單元沒有明示 per-arrival 解析值）時才當退回值使用，見
    # ``HourMajorSettlingGroupInput.max_backtrack_ns`` docstring 與步驟 1 的比較。
    config_max_backtrack_ns = int(round(config.max_backtrack_seconds * 1.0e9))

    groups: list[_GroupState] = []
    current_utc = batch.max_arrival_time_utc_ns
    master_hours_walked = 0

    if checkpoint_dir is not None:
        loaded = load_hour_major_checkpoint(Path(checkpoint_dir), batch=batch, handles=handles)
        if loaded is not None:
            groups, current_utc, master_hours_walked = loaded

    if not groups:
        groups = [_new_group_state(handles, g, config=config) for g in batch.groups]

    composed_base_cache: MutableMapping[tuple[int, int], ComposedCurrentAndStokesFields] = {}
    smagorinsky_cache: MutableMapping[int, np.ndarray] = {}
    # WP-16 任務 1：只跟網格拓樸有關，整個 run_flow_domain_hour_major 呼叫只建立一次，
    # 跨全部小時重複傳入 estimate_substep_count_local_cfl，見該函式與 LocalCflSpatialIndex
    # docstring。"global_cfl" 政策不需要這個索引，不建立（省下一次 KD-tree 建構成本）。
    local_cfl_spatial_index: LocalCflSpatialIndex | None = None
    if config.substep_policy == "local_cfl":
        local_cfl_spatial_index = build_local_cfl_spatial_index(driver_grid)
    elif config.substep_policy != "global_cfl":
        raise HourMajorError(
            f"substep_policy 必須是 'local_cfl' 或 'global_cfl'，收到 {config.substep_policy!r}"
        )

    # WP-16 任務 3：Smagorinsky 的反距離權重與網格陣列轉換只跟網格拓樸有關，
    # 整個 run_flow_domain_hour_major 呼叫只準備一次，跨全部小時重複傳入
    # compute_smagorinsky_kh_field，見 smagorinsky_batch.SmagorinskyPreparedGrid
    # docstring。只在啟用擴散且採用 numba 實作時才需要建立。
    smagorinsky_prepared_grid = None
    if config.enable_diffusion and config.smagorinsky_impl == "numba":
        from .smagorinsky_batch import SmagorinskyGridArrays, prepare_smagorinsky_grid

        _sg = driver_grid.schism_grid
        smagorinsky_prepared_grid = prepare_smagorinsky_grid(
            SmagorinskyGridArrays(
                elnode=_sg.elnode, area=_sg.area, dldxy=_sg.dldxy, nne=_sg.nne, indel=_sg.indel,
                x=_sg.x, y=_sg.y, xctr=_sg.xctr, yctr=_sg.yctr,
            )
        )
    domain_overhead = HourlyOverheadDiagnostics()
    kernel_call_count = 0

    # WP-14 任務 B：上一個「真正算過」的小時，其 field2 端點的合成結果與
    # 身分鍵；``None`` 代表「還沒有可用候選」（run 剛開始、checkpoint 續跑
    # 後第一個小時、缺值缺口或完全沒有工作的小時之後），此時
    # ``field1_reuse_candidate`` 一律不傳、直接重新計算，等同任務 B 新增前
    # 的行為——這也是為什麼續跑後第一個小時「直接重算即可」自動成立，不需要
    # 額外的續跑專用邏輯：checkpoint 不持久化這兩個變數，重新呼叫本函式時
    # 它們必然是 ``None``。
    prev_field2_endpoint: ComposedFieldEndpoint | None = None
    prev_field2_identity: _FieldEndpointIdentity | None = None

    import time as _time

    while True:
        if all(_group_finished(g) for g in groups):
            break

        hour_overhead = HourlyOverheadDiagnostics(n_hours=1)

        # ── 步驟 1：回溯上限——先於本小時任何流場查詢，鏡像 run_backward_batch
        # 在迴圈頂端就檢查 elapsed_ns 的順序。 ──────────────────────────────
        for group in groups:
            elapsed_ns = group.inp.arrival_time_utc_ns - current_utc
            active = group.native_status == NATIVE_STATUS_ACTIVE
            # WP-17：逐列比較（不是共用純量），讓 ``fixed_calendar_window`` 模式下
            # 同一分組裡不同到達時刻（不同沉底年齡）各自在自己的有效回溯時間停止；
            # ``-1`` sentinel（單元沒有明示解析值）退回這次呼叫收到的
            # ``config.max_backtrack_seconds``，見 ``HourMajorSettlingGroupInput.
            # max_backtrack_ns`` docstring。
            unit_cap_ns = np.where(
                group.inp.max_backtrack_ns >= 0, group.inp.max_backtrack_ns, config_max_backtrack_ns
            )
            capped = active & group.released & (elapsed_ns >= unit_cap_ns)
            if np.any(capped):
                _record_departures(
                    group, np.flatnonzero(capped), status=NATIVE_STATUS_MAX_BACKTRACK, utc_ns=current_utc
                )

        # ── 步驟 2：這一小時是否有任何工作要做（新到達或延續中的作用中列）。
        any_work = False
        for group in groups:
            newly = (~group.released) & (group.inp.arrival_time_utc_ns == current_utc)
            continuing = group.released & (group.native_status == NATIVE_STATUS_ACTIVE)
            if np.any(newly) or np.any(continuing):
                any_work = True
                break

        # 這一小時「實際跑過移動 kernel」的列遮罩，逐分組記錄；只有這個
        # 遮罩涵蓋到的列才算真正跑了這一小時（鏡像 ``run_backward_batch``
        # 的 ``hours_run`` 只在整個 while 迴圈本體真正執行完才遞增，遇到
        # 回溯上限或缺值缺口都在遞增之前就 ``break``，見步驟 4 與模組文件
        # 相關臭蟲修正說明）。預設全 False，遇到缺值缺口或完全沒有工作
        # 的小時都維持預設值，不需要另外特判。
        attempted_by_group: dict[int, np.ndarray] = {
            id(group): np.zeros(group.pa.n, dtype=bool) for group in groups
        }

        if any_work:
            t0 = _time.perf_counter()
            field_pair = forcing_resolver(current_utc)
            hour_overhead.time_forcing_resolve_s += _time.perf_counter() - t0

            # ── 步驟 3：新到達的粒子先定位＋初始化（獨立暫時陣列，見模組
            # 文件支撐事實 3），不論本小時流場是否有缺值都要做——標準版
            # ``run_backward_batch`` 的 locate_initial_elements 也是在缺值
            # 檢查之前、以無條件方式呼叫的。 ──────────────────────────────
            for group in groups:
                newly_mask = (~group.released) & (group.inp.arrival_time_utc_ns == current_utc)
                if not np.any(newly_mask):
                    continue
                _release_new_arrivals(
                    driver_grid, handles, group, newly_mask, field_pair=field_pair, utc_ns=current_utc
                )
                # 鏡像 ``run_backward_batch`` 在 locate_initial_elements（含
                # not_found 的狀態標記）之後、任何本小時移動之前，就把「到達
                # 當下」的原始狀態存成軌跡第一筆取樣（``trajectory_utc =
                # [arrival_utc]``）；本函式在小時主軸下按「每個單元自己的
                # 到達時刻」各自補這一筆，否則小時主軸的軌跡會比逐單元路徑
                # 少一筆、且其餘各筆整體錯位一個小時（見驗收 1／2 的臭蟲
                # 修正說明）。
                for progress in group.units.values():
                    start, stop = progress.info.start, progress.info.stop
                    if not np.any(newly_mask[start:stop]):
                        continue
                    progress.trajectory_utc.append(int(current_utc))
                    progress.trajectory_x.append(group.pa.x[start:stop].copy())
                    progress.trajectory_y.append(group.pa.y[start:stop].copy())
                    progress.trajectory_z.append(group.pa.z[start:stop].copy())
                    progress.trajectory_status.append(group.native_status[start:stop].copy())

            t0 = _time.perf_counter()
            node_is_dry_check, elem_is_dry_check = compute_wet_dry_flags(
                driver_grid, handles, eta_m=field_pair.eta_field1
            )
            hour_overhead.time_wet_dry_check_s += _time.perf_counter() - t0
            gap_detected = False
            try:
                field_pair.raise_if_unresolved_gap(
                    node_is_dry1=node_is_dry_check, node_is_dry2=node_is_dry_check, kbp=driver_grid.grid.kbp
                )
            except UnresolvedForcingGapError:
                gap_detected = True

            if gap_detected:
                for group in groups:
                    active_now = group.released & (group.native_status == NATIVE_STATUS_ACTIVE)
                    if np.any(active_now):
                        _record_departures(
                            group,
                            np.flatnonzero(active_now),
                            status=NATIVE_STATUS_FORCING_MISSING,
                            utc_ns=current_utc,
                        )
            else:
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

                has_wave = wave_field_resolver is not None
                cache_key = (int(field_pair.time_before_utc_ns), int(field_pair.time_after_utc_ns))
                # 快取只在「同一個流場小時」內有意義（逆向逐小時遞減，已走過的小時不會再用到）；
                # 換到新的小時就先清空，避免整個回溯窗（數百小時 × 每小時數十至數百 MB 的合成
                # 場）全部留在記憶體中。只影響記憶體，不影響任何數值。
                if cache_key not in composed_base_cache:
                    composed_base_cache.clear()
                base = composed_base_cache.get(cache_key)
                if base is not None:
                    hour_overhead.compose_base_cache_hits += 1
                else:
                    # WP-14 任務 B：只有在「上一個真正算過的小時」的 field2
                    # 端點，身分（流場時刻／資料來源／波浪時刻）與這一小時
                    # 的 field1 相符時，才把它當候選傳進去；真正能不能安全
                    # 重用（node_is_dry 是否逐位元相同）由
                    # compose_current_and_stokes_fields 內部判斷，這裡的
                    # 身分比對只是「值不值得嘗試」的第一層篩選，篩選錯了
                    # 也不會產生錯誤結果（見 ``_FieldEndpointIdentity``
                    # docstring）。
                    this_field1_identity = _field1_identity(field_pair, has_wave=has_wave)
                    reuse_candidate = (
                        prev_field2_endpoint
                        if prev_field2_identity is not None and prev_field2_identity == this_field1_identity
                        else None
                    )

                    t0 = _time.perf_counter()
                    base = compose_current_and_stokes_fields(
                        driver_grid, handles, field_pair, wave_field1=wave1, wave_field2=wave2,
                        stokes_depth_threshold_ratio=config.stokes_depth_threshold_ratio,
                        stokes_pool=stokes_pool,
                        stokes_worker_count=stokes_worker_count,
                        field1_reuse_candidate=reuse_candidate,
                        # WP-15 任務 A.3：主迴圈剛才已經用同一個
                        # field_pair.eta_field1 算過一次乾濕旗標（上面的
                        # node_is_dry_check／elem_is_dry_check，用於未登錄
                        # 缺口檢查），這裡直接重用，省下第二次全網格乾濕
                        # 判定（見 compose_current_and_stokes_fields 的
                        # precomputed_wet_dry_flags 參數 docstring）。
                        precomputed_wet_dry_flags=(node_is_dry_check, elem_is_dry_check),
                    )
                    hour_overhead.time_compose_base_s += _time.perf_counter() - t0
                    hour_overhead.compose_base_cache_misses += 1
                    hour_overhead.stokes_points_evaluated += base.stokes_points_evaluated
                    hour_overhead.stokes_points_possible += base.stokes_points_possible
                    if reuse_candidate is not None:
                        # compose_current_and_stokes_fields 內部用同一份
                        # node_is_dry 邏輯決定是否真的重用；這裡用同一個
                        # 比較重新得出布林值只是為了診斷計數，不影響正確性
                        # （即使誤判，也只是統計數字不準，不會用錯資料）。
                        if np.array_equal(reuse_candidate.node_is_dry, base.node_is_dry):
                            hour_overhead.endpoint_reuse_hits += 1
                        else:
                            hour_overhead.endpoint_reuse_misses += 1
                    composed_base_cache[cache_key] = base

                prev_field2_endpoint = base.field2_endpoint
                prev_field2_identity = _field2_identity(field_pair, has_wave=has_wave)

                t0 = _time.perf_counter()
                if config.substep_policy == "local_cfl":
                    assert local_cfl_spatial_index is not None  # 見上方建構區塊
                    # WP-16 任務 1：同一小時內所有分組共用同一個 ndeltp（全域同步時鐘不變，
                    # 見任務書），因此活動粒子位置要跨「全部」分組合併，不是只看某一個分組。
                    active_x_parts = [g.pa.x[g.pa.status == 1] for g in groups]
                    active_y_parts = [g.pa.y[g.pa.status == 1] for g in groups]
                    active_x = (
                        np.concatenate(active_x_parts) if active_x_parts else np.zeros(0, dtype=np.float64)
                    )
                    active_y = (
                        np.concatenate(active_y_parts) if active_y_parts else np.zeros(0, dtype=np.float64)
                    )
                    ndeltp = estimate_substep_count_local_cfl(
                        driver_grid, base.u_field1, base.v_field1,
                        active_x=active_x, active_y=active_y,
                        spatial_index=local_cfl_spatial_index,
                        dt_seconds=3600.0, dt_max_seconds=config.dt_max_seconds,
                        cfl_number=config.cfl_number,
                    )
                else:
                    ndeltp = estimate_substep_count(
                        driver_grid, base.u_field1, base.v_field1,
                        dt_seconds=3600.0, dt_max_seconds=config.dt_max_seconds,
                        cfl_number=config.cfl_number,
                    )
                hour_overhead.time_substep_count_s += _time.perf_counter() - t0
                dtb = 3600.0 / ndeltp
                hour_start_utc = current_utc

                hf2_now = None
                if config.enable_diffusion:
                    smagorinsky_key = int(field_pair.time_after_utc_ns)
                    cached = smagorinsky_cache.get(smagorinsky_key)
                    if cached is not None:
                        hf2_now = cached
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
                            use_numba=config.smagorinsky_impl == "numba",
                            smagorinsky_prepared_grid=smagorinsky_prepared_grid,
                        )
                        hour_overhead.time_smagorinsky_s += _time.perf_counter() - t0
                        hour_overhead.smagorinsky_cache_misses += 1
                        smagorinsky_cache.clear()  # 同上：只保留目前這一小時
                        smagorinsky_cache[smagorinsky_key] = hf2_now

                for group in groups:
                    newly_mask = (
                        group.released
                        & (group.inp.arrival_time_utc_ns == current_utc)
                        & (group.native_status == NATIVE_STATUS_ACTIVE)
                    )
                    continuing_mask = (
                        group.released
                        & (group.inp.arrival_time_utc_ns != current_utc)
                        & (group.native_status == NATIVE_STATUS_ACTIVE)
                    )
                    attempted_by_group[id(group)] = newly_mask | continuing_mask
                    n_new = int(np.count_nonzero(newly_mask))
                    n_continuing = int(np.count_nonzero(continuing_mask))
                    if n_new == 0 and n_continuing == 0:
                        continue

                    t0 = _time.perf_counter()
                    composed = apply_settling_velocity(
                        base, group.inp.settling_velocity_mps, kbp=driver_grid.grid.kbp
                    )
                    hour_overhead.time_apply_settling_s += _time.perf_counter() - t0

                    if n_new > 0:
                        _prime_new_arrivals(handles, driver_grid, group, newly_mask, composed=composed)

                    needs_split = config.enable_diffusion and n_new > 0 and n_continuing > 0
                    kernel_call_count += _run_group_hour(
                        driver_grid, handles, group,
                        composed=composed, config=config, ndeltp=ndeltp, dtb=dtb,
                        hour_start_utc=hour_start_utc, hf2_now=hf2_now,
                        newly_mask=newly_mask if needs_split else None,
                        continuing_mask=continuing_mask if needs_split else None,
                    )

                    group.prev_hf2 = hf2_now
                    group.prev_hf2_utc_ns = hour_start_utc

                    _post_hour_checks(
                        driver_grid, group, composed=composed, field_pair=field_pair,
                        config=config, hour_end_utc=current_utc - _HOUR_NS,
                    )

        domain_overhead.add(hour_overhead)

        # ── 步驟 4：小時邊界後續——軌跡取樣、hours_run、finished 判定。 ──
        hour_end_utc = current_utc - _HOUR_NS
        for group in groups:
            attempted_mask = attempted_by_group[id(group)]
            for progress in group.units.values():
                if progress.finished:
                    continue
                start, stop = progress.info.start, progress.info.stop
                was_in_play = np.any(attempted_mask[start:stop])
                if was_in_play:
                    progress.hours_run += 1
                # 只有這一小時真的跑過移動 kernel 的單元才取樣軌跡，鏡像
                # ``run_backward_batch`` 的取樣邏輯只出現在 while 迴圈本體
                # 真正執行完的那幾次（見 ``was_in_play`` 與 hours_run 的
                # 臭蟲修正說明）；否則像回溯上限剛好在本小時開頭觸發、這一
                # 小時完全沒有任何工作的情況，會多出一筆與前一筆重複、且
                # 逐單元路徑本來就不會有的軌跡取樣。
                if was_in_play and hour_end_utc <= progress.next_output_utc_ns:
                    progress.trajectory_utc.append(int(hour_end_utc))
                    progress.trajectory_x.append(group.pa.x[start:stop].copy())
                    progress.trajectory_y.append(group.pa.y[start:stop].copy())
                    progress.trajectory_z.append(group.pa.z[start:stop].copy())
                    progress.trajectory_status.append(group.native_status[start:stop].copy())
                    progress.next_output_utc_ns -= output_interval_ns
                is_released_now = bool(group.released[start]) if stop > start else False
                if is_released_now and not np.any(group.native_status[start:stop] == NATIVE_STATUS_ACTIVE):
                    progress.finished = True

        current_utc = hour_end_utc
        master_hours_walked += 1

        if _max_master_hours is not None and master_hours_walked >= _max_master_hours:
            if checkpoint_dir is not None:
                save_hour_major_checkpoint(
                    Path(checkpoint_dir), groups=groups, current_utc=current_utc,
                    master_hours_walked=master_hours_walked,
                )
            return _package_partial_result(groups, domain_overhead, kernel_call_count, master_hours_walked)

        if (
            checkpoint_dir is not None
            and checkpoint_every_hours > 0
            and master_hours_walked % checkpoint_every_hours == 0
        ):
            save_hour_major_checkpoint(
                Path(checkpoint_dir), groups=groups, current_utc=current_utc,
                master_hours_walked=master_hours_walked,
            )

    if checkpoint_dir is not None:
        save_hour_major_checkpoint(
            Path(checkpoint_dir), groups=groups, current_utc=current_utc,
            master_hours_walked=master_hours_walked,
        )

    unit_results = {
        progress.info.unit_id: _finalize_unit_result(group, progress)
        for group in groups
        for progress in group.units.values()
    }
    return HourMajorRunResult(
        unit_results=unit_results,
        domain_overhead_diagnostics=domain_overhead,
        domain_kernel_call_count=kernel_call_count,
        domain_hours_walked=master_hours_walked,
    )


def _package_partial_result(
    groups: Sequence[_GroupState],
    domain_overhead: HourlyOverheadDiagnostics,
    kernel_call_count: int,
    master_hours_walked: int,
) -> HourMajorRunResult:
    """給 ``_max_master_hours`` 測試鉤子用：把尚未全部終止的狀態包成結果。

    只有已經至少被取樣過一次軌跡的單元才能安全地呼叫
    ``_finalize_unit_result``（該函式要求 ``trajectory_utc`` 非空）；尚未
    釋放、或釋放後還沒碰到任何一次輸出取樣點的單元會被跳過，不會出現在
    回傳的 ``unit_results`` 裡——這是測試鉤子專用的部分結果，正式呼叫不應
    該依賴這個提前中止的路徑取得最終輸出，只用來驗證 checkpoint／續跑的
    位元相同性。
    """

    unit_results: dict[str, BackwardDriverResult] = {}
    for group in groups:
        for progress in group.units.values():
            if progress.trajectory_utc:
                unit_results[progress.info.unit_id] = _finalize_unit_result(group, progress)
    return HourMajorRunResult(
        unit_results=unit_results,
        domain_overhead_diagnostics=domain_overhead,
        domain_kernel_call_count=kernel_call_count,
        domain_hours_walked=master_hours_walked,
    )


def _group_finished(group: _GroupState) -> bool:
    return all(progress.finished for progress in group.units.values())


def _release_new_arrivals(
    driver_grid: NativeBackwardDriverGrid,
    handles: BayTraceModuleHandles,
    group: _GroupState,
    newly_mask: np.ndarray,
    *,
    field_pair: HourlyFieldPair,
    utc_ns: int,
) -> None:
    """對這一小時剛到達的列，用獨立暫時陣列呼叫 locate_initial_elements。

    見模組文件支撐事實 3：絕不能對整個合併陣列直接呼叫
    ``locate_initial_elements``，只能對這裡挑出的、尚未定位過的少數列另外
    配置一個乾淨的暫時 ``ParticleArrays``，再把結果寫回。
    """

    rows = np.flatnonzero(newly_mask)
    n_new = rows.size
    temp_pa = handles.particle_arrays_cls(n_new)
    temp_pa.x[:] = group.pa.x[rows]
    temp_pa.y[:] = group.pa.y[rows]
    temp_pa.z[:] = group.pa.z[rows]
    temp_pa.z0[:] = group.pa.z0[rows]
    temp_pa.element[:] = -1
    temp_pa.status[:] = 0

    not_found = locate_initial_elements(
        driver_grid,
        handles,
        temp_pa,
        eta_m=field_pair.eta_field1,
        u_current_mps=field_pair.u_field1,
        v_current_mps=field_pair.v_field1,
        w_current_mps=field_pair.w_field1,
    )

    group.released[rows] = True
    group.pa.element[rows] = temp_pa.element
    group.pa.status[rows] = temp_pa.status.clip(min=0)
    group.pa.z[rows] = temp_pa.z
    group.pa.u[rows] = temp_pa.u
    group.pa.v[rows] = temp_pa.v
    group.pa.w[rows] = temp_pa.w
    group.pa.dhfx[rows] = temp_pa.dhfx
    group.pa.dhfy[rows] = temp_pa.dhfy
    group.pa.dhfz[rows] = temp_pa.dhfz
    group.pa.grdx[rows] = temp_pa.grdx
    group.pa.grdy[rows] = temp_pa.grdy
    group.pa.grdz[rows] = temp_pa.grdz
    group.pa.rnds[rows] = temp_pa.rnds
    group.pa.x_prev[rows] = temp_pa.x_prev
    group.pa.y_prev[rows] = temp_pa.y_prev

    if np.any(not_found):
        bad_rows = rows[not_found]
        _record_departures(group, bad_rows, status=NATIVE_STATUS_NUMERICAL_FAILURE, utc_ns=utc_ns)


_VELOCITY_FIELDS = ("u", "v", "w")
_DIFFUSIVITY_FIELDS = ("dhfx", "dhfy", "dhfz", "grdx", "grdy", "grdz")


def _snapshot_fields(pa: Any, rows: np.ndarray, field_names: tuple[str, ...]) -> dict[str, np.ndarray]:
    """把 ``pa`` 在 ``rows`` 位置上、指定欄位的目前值另外存一份副本。

    配合 ``_restore_fields`` 使用，見兩者共同的呼叫端 docstring：
    ``handles.batch_interpolate_velocity``／``batch_interpolate_diffusivity``
    這兩個 BayTrace 原生 kernel，與本模組已知的「``status!=1`` 的列完全不
    讀寫」慣例不同——它們會對 ``status!=1`` 的列**明確寫入預設值**（速度歸
    零；擴散係數寫入 ``hdc``／``3.0e-4``、梯度寫入 ``0.0``，見
    ``batch_interp.py`` 對應函式第 76-80 行、457-463 行），不是單純跳過。
    因此本模組任何「暫時把其他列的 status 設成 0，藉此把某個 kernel 呼叫
    的作用範圍限制在一部分列」的技巧，只要呼叫鏈裡可能包含這兩個 kernel
    （``batch_interpolate_velocity`` 一定會、``batch_interpolate_diffusivity``
    只在啟用擴散時才會），就必須在暫時停用前存一份「被停用列」的這些欄位、
    事後連同 ``status`` 一起還原，否則會把延續中粒子已經正確累積的速度／
    擴散狀態覆寫成錯誤的預設值（這正是本函式修正的臭蟲，見模組文件「已知
    限制」一節與 ``tests/test_native_hour_major.py`` 的等價驗收）。
    """

    return {name: getattr(pa, name)[rows].copy() for name in field_names}


def _restore_fields(pa: Any, rows: np.ndarray, snapshot: dict[str, np.ndarray]) -> None:
    for name, values in snapshot.items():
        getattr(pa, name)[rows] = values


def _prime_new_arrivals(
    handles: BayTraceModuleHandles,
    driver_grid: NativeBackwardDriverGrid,
    group: _GroupState,
    newly_mask: np.ndarray,
    *,
    composed: Any,
) -> None:
    """對剛釋放且定位成功的列，用合成後總速度場重新內插一次起點速度。

    對應 ``run_backward_batch`` 的「priming」步驟；用暫時把其餘列的
    ``pa.status`` 設成 0 的方式，把 ``batch_interpolate_velocity`` 的作用
    範圍限制在這批列。``batch_interpolate_velocity`` 對 ``status!=1`` 的列
    會明確把速度寫成 0（見 ``_snapshot_fields`` docstring），因此必須在
    停用前後另外保存／還原這些列的 ``u/v/w``，否則會把延續中粒子已經正確
    累積的速度覆寫成 0（見模組文件支撐事實 4 與相關臭蟲修正說明）。
    """

    found_mask = newly_mask & (group.pa.status == 1)
    if not np.any(found_mask):
        return
    other_active = (group.pa.status == 1) & ~found_mask
    other_rows = np.flatnonzero(other_active)
    saved_status = group.pa.status[other_active].copy()
    saved_velocity = _snapshot_fields(group.pa, other_rows, _VELOCITY_FIELDS)
    group.pa.status[other_active] = 0
    try:
        grid = driver_grid.schism_grid
        handles.batch_interpolate_velocity(
            group.pa.x, group.pa.y, group.pa.z, group.pa.element, group.pa.status,
            grid.i34, grid.elnode, grid.x, grid.y,
            grid.nvrt, grid.kbp, grid.kbe, grid.sigma_lcl, grid.dp,
            composed.u_field1, composed.v_field1, composed.w_field1, composed.eta_field1,
            composed.u_field2, composed.v_field2, composed.w_field2,
            0.0, driver_grid.h0,
            group.pa.u, group.pa.v, group.pa.w,
        )
    finally:
        group.pa.status[other_active] = saved_status
        _restore_fields(group.pa, other_rows, saved_velocity)

def _dispatch_departure_records(group: _GroupState, records: Sequence[DepartureRecord]) -> None:
    """把一批 ``DepartureRecord``（``row_particle_index`` 是分組合併陣列的
    列位置）依所屬單元拆開，轉成該單元自己陣列裡的局部列位置後append進去。

    供 ``mirror_rk4_substep_loop`` 內部因「離開關注區域」而產生的
    departures 使用（該函式回傳時 ``row_particle_index`` 就是呼叫時傳入
    ``pa`` 的列位置，本模組呼叫時 ``pa`` 一律是整個分組的合併陣列，因此
    這裡的位置換算與 ``_record_departures`` 完全相同）。
    """

    for record in records:
        row = record.row_particle_index
        for progress in group.units.values():
            if progress.info.start <= row < progress.info.stop:
                progress.departures.append(
                    DepartureRecord(
                        row_particle_index=row - progress.info.start,
                        status=record.status,
                        x_m=record.x_m,
                        y_m=record.y_m,
                        z_m=record.z_m,
                        utc_ns=record.utc_ns,
                    )
                )
                break


def _dispatch_local_departure_records(
    group: _GroupState, records: Sequence[LocalDomainFirstExitRecord]
) -> None:
    """``_dispatch_departure_records`` 的 local domain 版本（WP-23 新增）。

    把子步迴圈回傳的 ``LocalDomainFirstExitRecord``（非終止：離開自己站點
    local domain）依所屬單元拆開，換算成該單元自己陣列裡的局部列位置。
    只會收到 ``local_equals_flow=False`` 測站的粒子——``local_equals_flow=
    True`` 的列因為 ``group.local_first_exit_pending`` 從一開始就是
    ``False``，子步迴圈的 check_mask 根本不會命中它們（見
    ``HourMajorSettlingGroupInput.local_region_polygons`` docstring）。
    """

    for record in records:
        row = record.row_particle_index
        for progress in group.units.values():
            if progress.info.start <= row < progress.info.stop:
                progress.local_departures.append(
                    LocalDomainFirstExitRecord(
                        row_particle_index=row - progress.info.start,
                        x_m=record.x_m,
                        y_m=record.y_m,
                        z_m=record.z_m,
                        utc_ns=record.utc_ns,
                    )
                )
                break


def _mirror_rk4_substep_loop_dispatch(
    driver_grid: NativeBackwardDriverGrid,
    handles: BayTraceModuleHandles,
    pa: Any,
    *,
    config: HourMajorConfig,
    ibf: int,
    ndeltp: int,
    dtb: float,
    composed: Any,
    enable_diffusion: bool,
    hf1_field: np.ndarray | None,
    hf2_field: np.ndarray | None,
    particle_seed_lo: np.ndarray,
    particle_seed_hi: np.ndarray,
    particle_step_counter: np.ndarray,
    native_status: np.ndarray,
    substep_utc_ns: Callable[[int], int],
    interest_region_polygons: np.ndarray,
    departures: list[DepartureRecord],
    local_region_polygons: np.ndarray | None = None,
    local_first_exit_pending: np.ndarray | None = None,
    local_departures: list[LocalDomainFirstExitRecord] | None = None,
) -> SubstepDiagnostics:
    """WP-16 任務 2：依 ``config.substep_loop_impl`` 選擇子步迴圈實作。

    兩個呼叫端（``_run_group_hour`` 的「單次呼叫」與「拆成新到達／延續中
    兩次呼叫」分支）共用同一個參數組合，這裡集中一次，避免兩處分岔各自
    重複 if/else。

    ``local_region_polygons``／``local_first_exit_pending``／
    ``local_departures``（WP-23 新增）三者要嘛全部是 ``None``（這個分組完全
    不含 ``local_equals_flow=False`` 的測站，見
    ``HourMajorSettlingGroupInput.needs_local_domain_check``），要嘛全部有值；
    呼叫端負責維持這個不變量，本函式只原樣透傳給底層子步迴圈。
    """

    if config.substep_loop_impl == "python":
        return mirror_rk4_substep_loop(
            driver_grid, handles, pa,
            ibf=ibf, ndeltp=ndeltp, dtb=dtb, composed=composed,
            enable_diffusion=enable_diffusion,
            hf1_field=hf1_field, hf2_field=hf2_field,
            particle_seed_lo=particle_seed_lo,
            particle_seed_hi=particle_seed_hi,
            particle_step_counter=particle_step_counter,
            native_status=native_status,
            substep_utc_ns=substep_utc_ns,
            interest_region_polygons=interest_region_polygons,
            departures=departures,
            local_region_polygons=local_region_polygons,
            local_first_exit_pending=local_first_exit_pending,
            local_departures=local_departures,
        )
    if config.substep_loop_impl == "numba":
        return mirror_rk4_substep_loop_numba(
            driver_grid, handles, pa,
            ibf=ibf, ndeltp=ndeltp, dtb=dtb, composed=composed,
            enable_diffusion=enable_diffusion,
            hf1_field=hf1_field, hf2_field=hf2_field,
            particle_seed_lo=particle_seed_lo,
            particle_seed_hi=particle_seed_hi,
            particle_step_counter=particle_step_counter,
            native_status=native_status,
            substep_utc_ns=substep_utc_ns,
            interest_region_polygons=interest_region_polygons,
            departures=departures,
            local_region_polygons=local_region_polygons,
            local_first_exit_pending=local_first_exit_pending,
            local_departures=local_departures,
        )
    raise HourMajorError(
        f"substep_loop_impl 必須是 'python' 或 'numba'，收到 {config.substep_loop_impl!r}"
    )


def _run_group_hour(
    driver_grid: NativeBackwardDriverGrid,
    handles: BayTraceModuleHandles,
    group: _GroupState,
    *,
    composed: Any,
    config: HourMajorConfig,
    ndeltp: int,
    dtb: float,
    hour_start_utc: int,
    hf2_now: np.ndarray | None,
    newly_mask: np.ndarray | None,
    continuing_mask: np.ndarray | None,
) -> int:
    """對這個分組跑完這一小時的子步迴圈，回傳實際發出的 kernel 呼叫次數。

    ``newly_mask``／``continuing_mask`` 只在「這一小時同時有新到達與延續中
    的作用中粒子、且啟用擴散」時才會非 ``None``（見模組文件「新到達粒子與
    已在途粒子的擴散場鏈結」一節），此時拆成兩次呼叫，分別給予正確的
    ``hf1_field``；其餘情況只需要一次呼叫。呼叫產生的「離開關注區域」
    departures 會被拆回各自所屬的虛擬單元（見
    ``_dispatch_departure_records``）。
    """

    def _substep_utc_ns(idt: int, _start: int = hour_start_utc, _dtb: float = dtb) -> int:
        return _start - int(round(idt * _dtb * 1.0e9))

    # 上一小時的 Smagorinsky 場只有在「恰好是絕對 UTC 上的前一小時」才可信；
    # 若這個分組曾經有一整段小時完全沒有作用中粒子（見模組文件「已知限制」
    # 與下方 docstring 補述），``prev_hf2`` 會是好幾個小時之前留下的過期值，
    # 此時必須視同沒有「前一小時」可用，退回 ``hf2_now``（同一小時），否則
    # 會讓「間隔一段時間後才又有新到達」的粒子錯誤地拿到過期的擴散場。
    prev_hf2_fresh = (
        group.prev_hf2 is not None and group.prev_hf2_utc_ns == hour_start_utc + _HOUR_NS
    )

    needs_local = group.inp.needs_local_domain_check

    if newly_mask is None and continuing_mask is None:
        hf1_now = None
        if config.enable_diffusion:
            hf1_now = group.prev_hf2 if prev_hf2_fresh else hf2_now
        departures: list[DepartureRecord] = []
        local_departures: list[LocalDomainFirstExitRecord] = []
        _mirror_rk4_substep_loop_dispatch(
            driver_grid, handles, group.pa,
            config=config,
            ibf=-1, ndeltp=ndeltp, dtb=dtb, composed=composed,
            enable_diffusion=config.enable_diffusion,
            hf1_field=hf1_now, hf2_field=hf2_now,
            particle_seed_lo=group.particle_seed_lo,
            particle_seed_hi=group.particle_seed_hi,
            particle_step_counter=group.particle_step_counter,
            native_status=group.native_status,
            substep_utc_ns=_substep_utc_ns,
            interest_region_polygons=group.inp.interest_region_polygons,
            departures=departures,
            local_region_polygons=group.inp.local_region_polygons if needs_local else None,
            local_first_exit_pending=group.local_first_exit_pending if needs_local else None,
            local_departures=local_departures if needs_local else None,
        )
        _dispatch_departure_records(group, departures)
        if needs_local:
            _dispatch_local_departure_records(group, local_departures)
        return 1

    calls = 0
    for mask, hf1_now in (
        (newly_mask, hf2_now),
        (continuing_mask, group.prev_hf2 if prev_hf2_fresh else hf2_now),
    ):
        if mask is None or not np.any(mask & (group.pa.status == 1)):
            continue
        other_active = (group.pa.status == 1) & ~mask
        other_rows = np.flatnonzero(other_active)
        saved_status = group.pa.status[other_active].copy()
        # ``mirror_rk4_substep_loop`` 內部一定會呼叫
        # ``batch_interpolate_velocity``（STEP 3），啟用擴散時還會呼叫
        # ``batch_interpolate_diffusivity``；兩者都對暫時停用（status!=1）
        # 的列明確寫入預設值而非略過，因此必須連同 status 一起保存／還原
        # 這些欄位，否則會把「另一個世代」粒子的速度／擴散狀態覆寫成預設值
        # （見 ``_snapshot_fields`` docstring 與相關臭蟲修正說明）。
        saved_fields = _snapshot_fields(group.pa, other_rows, _VELOCITY_FIELDS)
        if config.enable_diffusion:
            saved_fields.update(_snapshot_fields(group.pa, other_rows, _DIFFUSIVITY_FIELDS))
        group.pa.status[other_active] = 0
        departures = []
        local_departures = []
        try:
            _mirror_rk4_substep_loop_dispatch(
                driver_grid, handles, group.pa,
                config=config,
                ibf=-1, ndeltp=ndeltp, dtb=dtb, composed=composed,
                enable_diffusion=config.enable_diffusion,
                hf1_field=hf1_now, hf2_field=hf2_now,
                particle_seed_lo=group.particle_seed_lo,
                particle_seed_hi=group.particle_seed_hi,
                particle_step_counter=group.particle_step_counter,
                native_status=group.native_status,
                substep_utc_ns=_substep_utc_ns,
                interest_region_polygons=group.inp.interest_region_polygons,
                departures=departures,
                local_region_polygons=group.inp.local_region_polygons if needs_local else None,
                local_first_exit_pending=group.local_first_exit_pending if needs_local else None,
                local_departures=local_departures if needs_local else None,
            )
        finally:
            group.pa.status[other_active] = saved_status
            _restore_fields(group.pa, other_rows, saved_fields)
        _dispatch_departure_records(group, departures)
        if needs_local:
            _dispatch_local_departure_records(group, local_departures)
        calls += 1
    return calls


def _post_hour_checks(
    driver_grid: NativeBackwardDriverGrid,
    group: _GroupState,
    *,
    composed: Any,
    field_pair: HourlyFieldPair,
    config: HourMajorConfig,
    hour_end_utc: int,
) -> None:
    """鏡像 ``run_backward_batch`` 小時邊界後的數值失敗／擱淺／沉底判定。

    逐行對應 ``run_backward_batch`` 小時迴圈尾端同樣三段檢查，唯一差異是
    這裡對整個分組（可能混合多個到達時刻）一次做完，判定條件全部以
    ``native_status == NATIVE_STATUS_ACTIVE`` 為「作用中」遮罩——尚未釋放
    的列雖然 ``native_status`` 也是 ``ACTIVE``，但 ``pa.element`` 仍是
    ``-1``、位置仍是有限的 ``x0/y0/z0``，因此下面三個判定式（分別需要
    「非有限位置」「有效元素」「有限海床 z」）都會自然把尚未釋放的列排除，
    不需要額外的 ``released`` 遮罩（見模組文件支撐事實的延伸推論，
    已在設計階段逐一驗證）。
    """

    pa = group.pa

    active = group.native_status == NATIVE_STATUS_ACTIVE
    bad = active & ~(np.isfinite(pa.x) & np.isfinite(pa.y) & np.isfinite(pa.z))
    if np.any(bad):
        _record_departures(
            group, np.flatnonzero(bad), status=NATIVE_STATUS_NUMERICAL_FAILURE, utc_ns=hour_end_utc
        )

    active = group.native_status == NATIVE_STATUS_ACTIVE
    if np.any(active):
        elem = pa.element
        valid_elem = elem >= 0
        elem_dry_lookup = np.zeros(pa.n, dtype=bool)
        elem_dry_lookup[valid_elem] = composed.elem_is_dry[elem[valid_elem]]
        stranded = active & valid_elem & elem_dry_lookup
        if np.any(stranded):
            _record_departures(
                group, np.flatnonzero(stranded), status=NATIVE_STATUS_STRANDED, utc_ns=hour_end_utc
            )

    if group.inp.settling_velocity_mps < 0.0:
        active = group.native_status == NATIVE_STATUS_ACTIVE
        if np.any(active):
            bed_z = particle_bed_z_levels(driver_grid, field_pair.eta_field1, pa.element)
            at_bed = active & np.isfinite(bed_z) & ((pa.z - bed_z) <= config.deposition_tolerance_m)
            group.deposit_streak[at_bed] += 1
            group.deposit_streak[active & ~at_bed] = 0
            newly_deposited = active & (group.deposit_streak >= config.deposition_consecutive_hours)
            if np.any(newly_deposited):
                _record_departures(
                    group, np.flatnonzero(newly_deposited),
                    status=NATIVE_STATUS_DEPOSITED, utc_ns=hour_end_utc,
                )


# ══════════════════════════════════════════════════════════════════════════
# 5. Checkpoint：每個分組一個 .npz（陣列狀態）＋一份 JSON（逐單元進度）
# ══════════════════════════════════════════════════════════════════════════

_PA_FLOAT_FIELDS = (
    "x", "y", "z", "z0", "u", "v", "w",
    "dhfx", "dhfy", "dhfz", "grdx", "grdy", "grdz", "rnds", "x_prev", "y_prev",
)
_PA_INT_FIELDS = ("element", "status")


def save_hour_major_checkpoint(
    checkpoint_dir: Path,
    *,
    groups: Sequence[_GroupState],
    current_utc: int,
    master_hours_walked: int,
) -> None:
    """把目前完整的小時主軸執行期狀態原子性寫進 ``checkpoint_dir``。

    每個沉降速度分組各自一個 ``group_<i>.npz``（粒子陣列、``native_status``
    ／``released``／``deposit_streak``／計數器型擴散種子與步數、上一小時
    的 Smagorinsky 場、每個單元已取樣的軌跡陣列），加上一份 ``meta.json``
    （純量進度、``departures`` 記錄、軌跡取樣的 UTC 時刻）。``meta.json``
    以「先寫暫存檔、再 ``os.replace``」的方式原子性寫出，續跑時只信任
    ``meta.json`` 存在且可解析——半寫的 ``.npz`` 若恰好在寫 ``meta.json``
    之前中斷，會因為 ``meta.json`` 尚未更新而被視為「這次 checkpoint 沒有
    完成」，下次仍會讀到上一次成功的 checkpoint（``.npz`` 本身不是原子性
    寫出，但只要 ``meta.json`` 沒有指向它，就不會被讀取）。
    """

    import os

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {
        # WP-23：從 v1 升到 v2，新增 local_first_exit_pending／local_departures；
        # 舊版 v1 checkpoint 是在終止邊界誤用 local domain 的程式下寫出的，粒子
        # 可能已經在 local domain 邊界被錯誤截斷，續跑沒有意義，見
        # load_hour_major_checkpoint 的明確拒絕與 reports/WP-23.md。
        "checkpoint_kind": "hour_major_domain_checkpoint_v2",
        "current_utc_ns": int(current_utc),
        "master_hours_walked": int(master_hours_walked),
        "groups": [],
    }
    for gi, group in enumerate(groups):
        npz_payload: dict[str, np.ndarray] = {}
        for name in _PA_FLOAT_FIELDS:
            npz_payload[f"pa_{name}"] = getattr(group.pa, name)
        for name in _PA_INT_FIELDS:
            npz_payload[f"pa_{name}"] = getattr(group.pa, name)
        npz_payload["native_status"] = group.native_status
        npz_payload["released"] = group.released
        npz_payload["deposit_streak"] = group.deposit_streak
        npz_payload["local_first_exit_pending"] = group.local_first_exit_pending
        has_diffusion_state = group.particle_seed_lo is not None
        if has_diffusion_state:
            npz_payload["particle_seed_lo"] = group.particle_seed_lo
            npz_payload["particle_seed_hi"] = group.particle_seed_hi
            npz_payload["particle_step_counter"] = group.particle_step_counter
        has_prev_hf2 = group.prev_hf2 is not None
        if has_prev_hf2:
            npz_payload["prev_hf2"] = group.prev_hf2
            npz_payload["prev_hf2_utc_ns"] = np.asarray([group.prev_hf2_utc_ns], dtype=np.int64)

        units_meta: list[dict[str, Any]] = []
        for ui, unit_info in enumerate(group.inp.units):
            progress = group.units[unit_info.unit_id]
            units_meta.append(
                {
                    "unit_id": unit_info.unit_id,
                    "finished": progress.finished,
                    "hours_run": progress.hours_run,
                    "next_output_utc_ns": progress.next_output_utc_ns,
                    "trajectory_utc_ns": list(progress.trajectory_utc),
                    "departures": [
                        {
                            "row_particle_index": d.row_particle_index,
                            "status": d.status,
                            "x_m": d.x_m,
                            "y_m": d.y_m,
                            "z_m": d.z_m,
                            "utc_ns": d.utc_ns,
                        }
                        for d in progress.departures
                    ],
                    "local_departures": [
                        {
                            "row_particle_index": d.row_particle_index,
                            "x_m": d.x_m,
                            "y_m": d.y_m,
                            "z_m": d.z_m,
                            "utc_ns": d.utc_ns,
                        }
                        for d in progress.local_departures
                    ],
                    "has_trajectory": bool(progress.trajectory_x),
                }
            )
            if progress.trajectory_x:
                npz_payload[f"traj_x_{ui}"] = np.stack(progress.trajectory_x, axis=0)
                npz_payload[f"traj_y_{ui}"] = np.stack(progress.trajectory_y, axis=0)
                npz_payload[f"traj_z_{ui}"] = np.stack(progress.trajectory_z, axis=0)
                npz_payload[f"traj_status_{ui}"] = np.stack(progress.trajectory_status, axis=0)

        np.savez(str(checkpoint_dir / f"group_{gi}.npz"), **npz_payload)
        meta["groups"].append(
            {
                "settling_velocity_mps": group.inp.settling_velocity_mps,
                "has_diffusion_state": has_diffusion_state,
                "has_prev_hf2": has_prev_hf2,
                "units": units_meta,
            }
        )

    tmp_path = checkpoint_dir / "meta.json.tmp"
    tmp_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, checkpoint_dir / "meta.json")


def load_hour_major_checkpoint(
    checkpoint_dir: Path,
    *,
    batch: HourMajorDomainBatch,
    handles: BayTraceModuleHandles,
) -> tuple[list[_GroupState], int, int] | None:
    """讀回 ``save_hour_major_checkpoint`` 寫出的狀態；沒有 checkpoint 時回傳 ``None``。

    讀回後會核對分組數、每組的沉降速度與單元順序是否與目前傳入的
    ``batch`` 完全一致，任何不一致都 fail-closed 直接拒絕（代表呼叫端傳入
    了與寫出當時不同的粒子輸入，續跑沒有意義）。
    """

    meta_path = checkpoint_dir / "meta.json"
    if not meta_path.is_file():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("checkpoint_kind") == "hour_major_domain_checkpoint_v1":
        raise HourMajorError(
            f"{meta_path} 是 WP-23 之前的 v1 checkpoint（終止邊界誤用 local domain，"
            "而非 flow domain）；此 checkpoint 裡的粒子可能已經在站點 local domain "
            "邊界被錯誤截斷，語意不正確，續跑沒有意義，請刪除 checkpoint 目錄後"
            "重新從頭跑（見 reports/WP-23.md），fail-closed 拒絕"
        )
    if meta.get("checkpoint_kind") != "hour_major_domain_checkpoint_v2":
        raise HourMajorError(f"{meta_path} 不是本模組的 checkpoint 格式")
    if len(meta["groups"]) != len(batch.groups):
        raise HourMajorError("checkpoint 的沉降速度分組數與目前輸入不一致，拒絕續跑")

    groups: list[_GroupState] = []
    for gi, (group_meta, inp) in enumerate(zip(meta["groups"], batch.groups, strict=True)):
        if float(group_meta["settling_velocity_mps"]) != inp.settling_velocity_mps:
            raise HourMajorError(f"checkpoint 分組 {gi} 的沉降速度與目前輸入不一致，拒絕續跑")

        with np.load(str(checkpoint_dir / f"group_{gi}.npz"), allow_pickle=False) as npz:
            n = inp.particle_count
            pa = handles.particle_arrays_cls(n)
            for name in _PA_FLOAT_FIELDS:
                getattr(pa, name)[:] = npz[f"pa_{name}"]
            for name in _PA_INT_FIELDS:
                getattr(pa, name)[:] = npz[f"pa_{name}"]
            native_status = npz["native_status"].astype(np.int32).copy()
            released = npz["released"].astype(bool).copy()
            deposit_streak = npz["deposit_streak"].astype(np.int64).copy()
            local_first_exit_pending = npz["local_first_exit_pending"].astype(bool).copy()

            particle_seed_lo = particle_seed_hi = particle_step_counter = None
            if group_meta["has_diffusion_state"]:
                particle_seed_lo = npz["particle_seed_lo"].copy()
                particle_seed_hi = npz["particle_seed_hi"].copy()
                particle_step_counter = npz["particle_step_counter"].copy()
            prev_hf2 = npz["prev_hf2"].copy() if group_meta["has_prev_hf2"] else None
            prev_hf2_utc_ns = (
                int(npz["prev_hf2_utc_ns"][0]) if group_meta["has_prev_hf2"] else None
            )

            units: dict[str, _UnitProgress] = {}
            for ui, (unit_meta, unit_info) in enumerate(zip(group_meta["units"], inp.units, strict=True)):
                if unit_meta["unit_id"] != unit_info.unit_id:
                    raise HourMajorError(
                        f"checkpoint 分組 {gi} 第 {ui} 個單元的 unit_id 與目前輸入不一致，拒絕續跑"
                    )
                if unit_meta["has_trajectory"]:
                    trajectory_x = list(npz[f"traj_x_{ui}"])
                    trajectory_y = list(npz[f"traj_y_{ui}"])
                    trajectory_z = list(npz[f"traj_z_{ui}"])
                    trajectory_status = list(npz[f"traj_status_{ui}"])
                else:
                    # 注意：這四個必須各自是獨立的 list 物件，不能寫成
                    # ``a = b = c = d = []`` 這種鏈式賦值——那樣四個名字會
                    # 指到同一個 list，後續對 trajectory_x／y／z／status 的
                    # ``append`` 全部會疊進同一份底層資料，讀出來的
                    # ``trajectory_x_m`` 會混進 y／z／status 的值（見驗收 6
                    # 的臭蟲修正說明：尚未釋放就中斷、續跑後才第一次取樣的
                    # 單元最容易踩到，因為只有這種單元會走到這個分支）。
                    trajectory_x = []
                    trajectory_y = []
                    trajectory_z = []
                    trajectory_status = []
                departures = [
                    DepartureRecord(
                        row_particle_index=int(d["row_particle_index"]),
                        status=int(d["status"]),
                        x_m=float(d["x_m"]),
                        y_m=float(d["y_m"]),
                        z_m=float(d["z_m"]),
                        utc_ns=int(d["utc_ns"]),
                    )
                    for d in unit_meta["departures"]
                ]
                local_departures = [
                    LocalDomainFirstExitRecord(
                        row_particle_index=int(d["row_particle_index"]),
                        x_m=float(d["x_m"]),
                        y_m=float(d["y_m"]),
                        z_m=float(d["z_m"]),
                        utc_ns=int(d["utc_ns"]),
                    )
                    for d in unit_meta["local_departures"]
                ]
                units[unit_info.unit_id] = _UnitProgress(
                    info=unit_info,
                    finished=bool(unit_meta["finished"]),
                    hours_run=int(unit_meta["hours_run"]),
                    next_output_utc_ns=int(unit_meta["next_output_utc_ns"]),
                    trajectory_utc=list(unit_meta["trajectory_utc_ns"]),
                    trajectory_x=trajectory_x,
                    trajectory_y=trajectory_y,
                    trajectory_z=trajectory_z,
                    trajectory_status=trajectory_status,
                    departures=departures,
                    local_departures=local_departures,
                )

        groups.append(
            _GroupState(
                inp=inp,
                pa=pa,
                native_status=native_status,
                released=released,
                particle_seed_lo=particle_seed_lo,
                particle_seed_hi=particle_seed_hi,
                particle_step_counter=particle_step_counter,
                deposit_streak=deposit_streak,
                local_first_exit_pending=local_first_exit_pending,
                prev_hf2=prev_hf2,
                prev_hf2_utc_ns=prev_hf2_utc_ns,
                units=units,
            )
        )

    return groups, int(meta["current_utc_ns"]), int(meta["master_hours_walked"])
