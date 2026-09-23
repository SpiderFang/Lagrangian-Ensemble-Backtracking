"""WP-11：把 WP-8 正式接線（``production_wiring``）與 WP-9 小時主軸排程
（``hour_major``）接成 ``lbt native-run --resource-provider`` 可直接使用的
模組層級、可 pickle 工廠，讓「切段平行」（見 ``orchestrator.
plan_native_hour_major_segments``）真正能在伺服器上一條指令跑完。

為什麼需要這個模組
--------------------
``orchestrator.py`` 文件第 9.8 節（WP-9 留下的接線步驟）與 ``production_
wiring.py`` 文件都明確指出：「``--resource-provider`` 指定的 ``run_unit``
需要能辨識 ``hourmajor::`` 前綴單元」這件事，刻意留給不屬於 WP-7／WP-8／
WP-9 職責範圍、但需要同時 import 三者的另一個模組負責——這正是本模組的
唯一職責。本模組**不新增任何科學邏輯**，只做三件事：

1. ``build_worker_resources(worker_id)``：每個長駐 worker 子程序只呼叫
   一次，載入 BayTrace 模組、讀回粒子身分表與計畫檔的 ``hour_major_
   members`` 對照表，並準備一個「按需建構、建過快取」的
   ``{analysis_region_id: DomainResources}`` 字典（鍵值是
   ``analysis_region_id`` 不是 ``flow_domain_id``，見
   ``_domain_resources_for`` docstring 的完整說明；``production_wiring.
   build_domain_resources`` 對同一個區域只需要呼叫一次；見
   ``resources.DomainResources`` 文件「每個工作程序各自呼叫一次」的
   既有契約）。
2. ``run_unit(resources, spec)``：辨識 ``spec.unit_id`` 是否帶
   ``hourmajor::`` 前綴，若是則查出底下原始工作單元、組成
   ``hour_major.HourMajorDomainBatch``、呼叫
   ``hour_major.run_flow_domain_hour_major``，並把每個原始工作單元自己的
   ``BackwardDriverResult`` 序列化成 ``.npz``（見「重量級輸出的存放格式」
   一節），供 ``lbt native-postprocess`` 之後讀回、餵給
   ``postprocess.PostprocessAccumulator``。
3. ``make_hour_major_resource_provider()``：模組層級、呼叫時不帶參數的
   factory（符合 ``lbt native-run --resource-provider`` 既有呼叫慣例，見
   ``orchestrator.py`` 文件第 9.8 節最後一段「刻意沒有改動」的邊界），
   回傳 ``(build_worker_resources, run_unit)`` 這一對可 pickle 的
   callable。

環境變數契約（正式接線唯一的參數傳遞管道，絕不硬編伺服器路徑）
------------------------------------------------------------------
``make_hour_major_resource_provider`` 呼叫時不帶參數，所有需要的設定改由
環境變數傳入——這是刻意的設計，理由與 ``production_wiring.py`` 模組文件
「環境變數與路徑」一節相同：``multiprocessing`` ``spawn`` 啟動的子程序需要
能夠 pickle 匯入 ``build_worker_resources``／``run_unit``，若把設定綁進
閉包會無法序列化；改成在**呼叫 factory 的當下**（一定在父程序、``lbt
native-run`` 剛啟動時）讀一次環境變數，把結果凍結進一個可 pickle 的
``functools.partial`` 引數，之後 spawn 出的子程序不需要重新讀取環境（也就
不受「子程序繼承到的環境變數是否與父程序一致」這個平台差異影響）。

必要環境變數：

- ``LBT_NATIVE_CONFIG``：正式設定 YAML 路徑，直接透傳給
  ``production_wiring.make_domain_resources_factory``。
- ``LBT_NATIVE_PLAN``：``lbt native-plan --schedule hour-major``（或切段
  規劃）產生的計畫檔路徑，用來重建 ``hour_major_members`` 對照表（見
  ``orchestrator.read_native_hour_major_members``）。
- ``LBT_NATIVE_IDENTITY_TABLE``：WP-4 粒子身分表 npz 路徑（與產生上述計畫
  檔時用的必須是同一份，否則 ``spec.particle_row_indices`` 與身分表列位置
  對不上；本模組不驗證兩者是否確實同源，呼叫端必須自行保證，是本模組的
  已知限制，見下方「已知限制」）。
- ``LBT_NATIVE_RESULTS_DIR``：存放逐原始工作單元 ``BackwardDriverResult``
  ``.npz`` 的目錄，供 ``lbt native-postprocess`` 讀取。

選用環境變數（未設定時使用列出的預設值；**誠實揭露**：以下物理／數值參數
目前不在 ``config.ProjectConfig`` schema 中——``ExecutionConfig``／
``IntegrationConfig``／``BoundaryConfig`` 只定義了 ``dt_max_seconds``／
``output_interval_seconds``／``max_backtrack_days``，沒有 ``cfl_number``／
``horcon``／``h0``／``enable_diffusion``／deposition 門檻／Stokes 深度門檻
比例欄位；把這些欄位加進 ``config.py`` 屬於另一個工作包的範圍，不在本次
WP-11 可修改檔案清單內，因此本模組退回 ``hour_major.HourMajorConfig`` 自身
的既有預設值，並要求呼叫端如果需要非預設值就明確以環境變數覆寫，不會
靜默假裝這些值來自已核定的正式設定）：

- ``LBT_NATIVE_FORMAL``：``"1"``（預設）／``"0"``，透傳給
  ``production_wiring`` 的 ``formal`` 參數。
- ``LBT_NATIVE_BACKTRACK_MODE``（WP-17 新增）：``"full_horizon_from_deposition"``
  （預設）或 ``"fixed_calendar_window"``，見 ``backtrack_horizon.py`` 模組文件。
  這是「本次 native-run 實際採用哪個回溯模式」的最終判定，只用來核對與計畫檔
  （``native-plan`` 階段已經用同一個模式解析過每個工作單元的
  ``resolved_max_backtrack_seconds``／``pre_window_deposition``）是否一致，以及
  與設定檔 ``bed_residence_time.backtrack_mode`` 是否一致；**不會**在這一步重新
  規劃工作單元。
- ``LBT_NATIVE_ACKNOWLEDGE_CONFIG_MODE_MISMATCH``（WP-17 新增）：``"1"``／
  ``"0"``（預設）。設定檔宣稱的 ``bed_residence_time.backtrack_mode`` 與
  ``LBT_NATIVE_BACKTRACK_MODE`` 實際採用值不一致時，預設 fail-closed 拒絕啟動
  （見 ``backtrack_horizon.require_mode_acknowledged_if_mismatched``）；只有明確
  設成 ``"1"`` 才會放行，且這個承認狀態本身也會寫進每個結果摘要。這同一個環境
  變數也用於核對 WP-18 實驗案例（見下）與計畫檔是否一致，不另外新增一個
  「承認」旗標。
- ``LBT_NATIVE_EXPERIMENT_CASE``（WP-18 新增）：``"finite_depth_stokes"``
  （預設，含有限水深 Stokes 漂移）或 ``"no_stokes"``（只有流，完全不讀波浪
  資料）；亦可為 ``runtime.EXPERIMENT_CASE_SPECS`` 其他已登錄案例（此時
  ``include_stokes`` 由該案例的 ``ExperimentCaseSpec.include_stokes`` 決定，
  但擴散仍固定用 Smagorinsky，不採用該案例的 ``diffusion_kind``）。見
  ``physics_case.py`` 模組文件。
- ``LBT_NATIVE_REQUIRE_WAVE_DATA``（WP-18 新增）：``"1"``／``"0"``（預設）。
  ``"1"`` 時，``include_stokes=True`` 但 NWW3 波浪資料缺漏會直接 fail-closed，
  不再靜默退化成沒有 Stokes（見 ``production_wiring.build_domain_resources``
  的 ``require_wave_data`` 參數）。
- ``LBT_NATIVE_CHECKPOINT_ROOT``：提供時，每個切段排程單元各自在其下建立
  一個以 ``unit_id`` 命名的子目錄當 ``hour_major`` 的 ``checkpoint_dir``；
  省略時不做 checkpoint（中斷後整段必須從頭重算）。
- ``LBT_NATIVE_CHECKPOINT_EVERY_HOURS``：整數，預設 ``24``。
- ``LBT_NATIVE_CFL_NUMBER``：浮點，預設沿用
  ``HourMajorConfig.cfl_number``（``0.5``）。
- ``LBT_NATIVE_HORCON``：浮點，預設沿用 ``HourMajorConfig.horcon``
  （``0.2``）。
- ``LBT_NATIVE_H0``：浮點，預設沿用 ``HourMajorConfig.h0``（``0.0``）。
- ``LBT_NATIVE_ENABLE_DIFFUSION``：``"1"``（預設）／``"0"``。
- ``LBT_NATIVE_DEPOSITION_CONSECUTIVE_HOURS``：整數，預設 ``3``。
- ``LBT_NATIVE_DEPOSITION_TOLERANCE_M``：浮點，預設 ``0.5``。
- ``LBT_NATIVE_STOKES_DEPTH_THRESHOLD_RATIO``：浮點或空字串（``None``），
  預設 ``None``（沿用 ``run_backward_batch`` 自身預設）。

  **WP-13 誠實記載的已知限制（史托克斯子行程池未接線到正式路徑）**：
  ``backward_driver.compose_current_and_stokes_fields``／
  ``hour_major.run_flow_domain_hour_major`` 都已支援 ``stokes_pool``／
  ``stokes_worker_count`` 參數，把每小時只算一次的史托克斯漂移（Stokes
  drift）逐點求解切塊分派到呼叫端自備的子行程池平行計算，且已用直接呼叫
  （不巢狀在任何其他 ``ProcessPoolExecutor`` 的 worker 之內）驗證過位元
  相同（見 ``tests/test_native_hour_major.py::
  test_hour_major_stokes_pool_matches_sequential_bit_identical``）。但本
  模組**刻意不**提供環境變數旋鈕在 ``native-run`` 的每個 worker 子程序內
  再自動建立這個子行程池——因為 ``native-run`` 的每個 worker 子程序本身
  就已經是 ``orchestrator.run_native_plan`` 用 ``ProcessPoolExecutor``
  （spawn）開出來的子程序，實測在這種「子程序裡面再建一個
  ``ProcessPoolExecutor``」（巢狀行程池）的組態下，會不穩定地出現兩種
  失敗模式：(1) 整個 worker 卡死（觀察到的 CPU 時間在數分鐘內完全不再
  增加，符合死結特徵，而非單純運算緩慢）；(2) 孫行程在 bootstrap 階段
  丟出 ``FileNotFoundError``（``SemLock._rebuild`` 找不到底層號誌檔），
  兩者在同一份程式碼、同一組參數下都曾經重現，屬於競態（race），不是
  每次都會發生。受限於本工作包的時間範圍，未能查明根本原因（懷疑與
  ``concurrent.futures.process`` 的佇列管理執行緒／號誌在巢狀行程池組態
  下的既知脆弱性有關，但未證實），因此判斷**不能**把這個旋鈕接上正式
  ``native-run`` 管線——正式管線的可靠性優先於這項優化的效能收益。詳細
  重現腳本與分析見 ``reports/WP-13.md``。這不影響
  ``LBT_NATIVE_STOKES_DEPTH_THRESHOLD_RATIO``（上面這個環境變數）本身的
  行為；只是本模組沒有另外提供、也不建議自行加上
  「在 worker 子程序內建立史托克斯子行程池」的環境變數。
- ``LBT_NATIVE_TEST_FORCE_INTERRUPT_AFTER_HOURS``：整數，預設不設定
  （``None``）。**測試專用、正式部署禁止設定**：見下方「測試專用：模擬
  工作程序中途消失」一節，用來在自動化測試裡驗證「子程序被系統砍掉後、
  下次重跑能否從 checkpoint 續跑到位元相同的最終結果」，不是任何正式
  管線會用到的旋鈕。

測試專用：模擬工作程序中途消失（``LBT_NATIVE_TEST_FORCE_INTERRUPT_AFTER_HOURS``）
------------------------------------------------------------------------------------
``native-run`` 是用 ``multiprocessing``（預設 ``spawn``）平行呼叫 worker 子
程序的；真實伺服器上子程序可能被作業系統中途砍掉（記憶體不足、機器重開），
此時我們期望下一次重跑 ``lbt native-run``（同一個 ``--output`` 目錄）能夠從
``hour_major.run_flow_domain_hour_major`` 既有的 checkpoint 機制（
``checkpoint_dir``／``checkpoint_every_hours``，見 ``hour_major.py`` 模組文件
與 ``tests/test_native_hour_major.py::test_hour_major_checkpoint_resume_bit_
identical``）續跑，而不是把整段重算一次。要在自動化測試裡誠實地驗證這件
事，需要一個方法讓 ``run_unit`` 在「checkpoint 已經寫入部分小時、但最終結果
還沒寫成 ``.driver_result.npz``、``orchestrator.run_native_plan`` 也還沒把
這個排程單元記成完成」的狀態下「消失」。

設定此環境變數後，``_run_unit_entrypoint`` 會：

1. 先確認 ``LBT_NATIVE_CHECKPOINT_ROOT`` 也有設定——沒有 checkpoint 的情況
   下模擬「中斷後續跑」沒有意義（下次重跑只會從頭開始，不是真的在驗證續跑
   機制），這是呼叫端的設定錯誤，本模組選擇 fail-closed（直接 raise）而不
   是靜默忽略這個旋鈕、跑出一個誤導測試者的「假中斷」。
2. 以 ``_max_master_hours=<這個環境變數的值>`` 呼叫
   ``run_flow_domain_hour_major``；這個呼叫本身會照常把 checkpoint 正常寫進
   ``checkpoint_dir``（``hour_major.py`` 既有行為，本模組不改動），但回傳的
   是**尚未全部終止**的部分結果。
3. 刻意**不**把這個部分結果寫成 ``.driver_result.npz``、也**不** ``return``
   ——因為一旦 ``run_unit`` 正常 return，``orchestrator.run_native_plan`` 就
   會判定這個排程單元完成、寫下完成證據；改為 raise
   :class:`HourMajorProviderError`，讓 ``run_native_plan`` 把這個單元記成
   失敗（``*.done.json`` 的 ``status`` 欄位是 ``"failed"``，不滿足
   ``filter_pending_units`` 判定「可跳過」所需要的 ``status == "completed"``
   條件），下一次呼叫 ``native-run`` 時會被重新排入待執行清單。
4. 下一次呼叫（不設定這個環境變數）走正常路徑，``run_flow_domain_hour_
   major`` 讀到既有 checkpoint 後自動從最後完成的小時續跑到底，正常寫出完
   整的 ``.driver_result.npz`` 並正常 return，``run_native_plan`` 這次才會
   把這個單元標記為完成。

**正式部署絕不可設定這個環境變數**：它只是把每個排程單元原本會跑到底的
``run_flow_domain_hour_major`` 呼叫，人為截斷成一次注定失敗、且不產出任何
可用結果的呼叫，唯一用途是讓測試套件能夠在合理時間內（不需要真的等到系統
OOM killer 出手）重現「子程序中途消失」這個場景。

重量級輸出的存放格式（``.driver_result.npz``）
-------------------------------------------------
``run_unit`` 回傳給 ``orchestrator._persist_outcome`` 的摘要字典必須是可以
``json.dumps`` 序列化的輕量摘要（見 ``orchestrator.py`` 模組文件「執行
引擎」一節），不能直接塞入完整的軌跡陣列。因此每個原始工作單元的完整
``BackwardDriverResult``（含軌跡、departures、最終狀態）改用
``save_backward_driver_result_npz`` 寫成單獨的 ``.npz`` 檔（檔名
``<original_unit_id>.driver_result.npz``，落在 ``LBT_NATIVE_RESULTS_DIR``
底下），``run_unit`` 只把「這個排程單元底下有哪些原始 unit_id、各自最終
狀態的計數分布」寫進輕量摘要；``lbt native-postprocess``
（``load_backward_driver_result_npz``）逐一讀回，餵給
``postprocess.PostprocessAccumulator.add_batch``。

已知限制（誠實揭露）
--------------------
1. ``LBT_NATIVE_IDENTITY_TABLE`` 與產生計畫檔時使用的身分表必須同源，本
   模組不驗證兩者一致（``ParticleIdentityTable.content_sha256`` 沒有被
   寫進計畫檔可供比對）；不一致時 ``spec.particle_row_indices`` 取出的
   起點座標會是錯的，且不會有任何例外提示，這是留給呼叫端（伺服器操作
   腳本）自行保證的操作紀律，建議在 ``scripts/run_native_backtrack_
   server.sh`` 內用同一個檔案路徑貫穿 plan／run／postprocess 三步。
2. 物理參數（cfl／horcon／deposition／Stokes 門檻）目前只能透過環境變數
   覆寫、不是從 ``ProjectConfig`` 讀取（見上方「環境變數契約」的完整
   說明）；若研究團隊核定了與 ``HourMajorConfig`` 預設不同的正式數值，
   必須在啟動 ``native-run`` 前明確設定對應環境變數，本模組不會自動同步
   任何 YAML 欄位。
3. 一個 worker 若被排到同一個流場區域的多個段（切段後常見情形），本模組
   的 ``build_worker_resources`` 只在第一次遇到該區域時呼叫一次
   ``production_wiring.build_domain_resources``，之後的段重用同一份
   ``DomainResources``（同一個 forcing reader 實例、同一份滑動快取）；
   但若同一個 worker 被排到**不同**流場區域的段（``plan_worker_schedule``
   的 LPT 裝箱在段數遠多於 worker 數時可能發生），該 worker 會依序建構
   多個區域的資源，記憶體用量會疊加，``--per-worker-memory-bytes`` 的
   估算需要把這種情況考慮進去，本模組不會自動偵測或警告。
4. ``LBT_NATIVE_TEST_FORCE_INTERRUPT_AFTER_HOURS`` 是純測試旋鈕（見上方
   「測試專用：模擬工作程序中途消失」一節），本模組不會檢查呼叫端是不是
   在正式伺服器環境誤設了它——這與其他環境變數「未設定時退回安全預設值」
   的哲學不同，一旦設定就會讓每個排程單元的 ``run_unit`` 必定 raise、必定
   不產出結果，正式部署的操作腳本必須確保這個環境變數不在環境裡（不只是
   空字串，必須完全不存在或留白，見 ``_env_optional_int`` 的判斷方式）。
5. 【WP-11 修正紀錄，誠實揭露】``_run_unit_entrypoint`` 原本把
   ``spec.flow_domain_id``（實際 OCM 產品目錄名稱，例如
   ``northeast_taiwan_common_cache_v3``）直接當成 ``production_wiring.
   build_domain_resources`` 的 ``domain_id`` 參數傳入，但該參數依其 docstring
   明確要求是 ``analysis_region_id``（例如 ``"A"``）；兩者只有在
   ``native-plan`` 省略 ``--region-to-flow-domain``（退化為
   ``flow_domain_id == analysis_region_id``，只在合成測試會發生）時才會剛好
   相同，正式部署一定會提供 ``--region-to-flow-domain``，兩者不同，若不修正
   會讓每個排程單元都用錯誤的 ``domain_id`` 查 ``ProjectConfig.domains``。
   現在改成從排程單元底下原始工作單元的 ``analysis_region_id`` 欄位取得正確
   值（見 ``_domain_resources_for`` 與 ``_run_unit_entrypoint`` 內的相關
   說明），且會 fail-closed 檢查同一段不會橫跨多個 ``analysis_region_id``。
   這是本次工作包在撰寫端對端測試時發現、並在可修改檔案清單內直接修正的
   既有缺陷，不是新增的科學邏輯。
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..config import load_config
from . import orchestrator as native_orchestrator
from . import production_wiring
from .backtrack_horizon import (
    require_mode_acknowledged_if_mismatched,
    resolve_backtrack_mode,
)
from .backward_driver import (
    NATIVE_STATUS_DESCRIPTIONS,
    BackwardDriverResult,
    BayTraceModuleHandles,
    DepartureRecord,
    HourlyOverheadDiagnostics,
    SubstepDiagnostics,
    build_pre_window_deposition_result,
    load_baytrace_modules,
)
from .hour_major import (
    HourMajorConfig,
    build_hour_major_domain_batch,
    run_flow_domain_hour_major,
)
from .orchestrator import NativeWorkUnit, NativeWorkUnitSpec
from .particle_input import ParticleIdentityTable, load_particle_identity_table_npz
from .physics_case import resolve_physics_case
from .resources import DomainResources

__all__ = [
    "HourMajorProviderError",
    "HourMajorProviderParams",
    "HourMajorWorkerResources",
    "save_backward_driver_result_npz",
    "load_backward_driver_result_npz",
    "make_hour_major_resource_provider",
]

_HOUR_MAJOR_PREFIX = "hourmajor::"


class HourMajorProviderError(RuntimeError):
    """本模組專用例外，與 ``HourMajorError``／``WiringError`` 區分責任層。"""


def _require_env(env: Mapping[str, str], name: str) -> str:
    value = env.get(name)
    if value is None or not str(value).strip():
        raise HourMajorProviderError(
            f"環境變數 {name} 未設定或為空白；hour_major_provider 的正式接線只透過環境變數"
            "傳遞路徑與參數，本模組不接受硬編伺服器路徑，也不會用預設路徑頂替"
        )
    return str(value)


def _env_flag(env: Mapping[str, str], name: str, *, default: bool) -> bool:
    value = env.get(name)
    if value is None or not str(value).strip():
        return default
    return str(value).strip() not in ("0", "false", "False", "")


def _env_float(env: Mapping[str, str], name: str, *, default: float) -> float:
    value = env.get(name)
    if value is None or not str(value).strip():
        return default
    return float(value)


def _env_int(env: Mapping[str, str], name: str, *, default: int) -> int:
    value = env.get(name)
    if value is None or not str(value).strip():
        return default
    return int(value)


def _env_optional_float(env: Mapping[str, str], name: str) -> float | None:
    value = env.get(name)
    if value is None or not str(value).strip():
        return None
    return float(value)


def _env_optional_int(env: Mapping[str, str], name: str) -> int | None:
    """未設定或空白時回傳 ``None``；供測試專用旋鈕等「預設不啟用」的整數欄位使用。"""

    value = env.get(name)
    if value is None or not str(value).strip():
        return None
    return int(value)


# ══════════════════════════════════════════════════════════════════════════
# 一、參數凍結：factory 呼叫當下（父程序）讀一次環境變數，之後只靠這個可
#    pickle 的 frozen dataclass 傳遞給 spawn 出的子程序，不依賴子程序自己
#    重新讀取環境（見模組文件「環境變數契約」）。
# ══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class HourMajorProviderParams:
    """``make_hour_major_resource_provider`` 呼叫當下凍結的全部設定。

    ``test_force_interrupt_after_hours`` 是**測試專用、正式部署禁止設定**的欄位：
    非 ``None`` 時，``run_unit`` 會在把 checkpoint 寫入部分小時之後主動 raise，
    模擬工作程序中途消失（見模組文件「測試專用：模擬工作程序中途消失」一節）；
    由 ``LBT_NATIVE_TEST_FORCE_INTERRUPT_AFTER_HOURS`` 解析而來，預設 ``None``
    時完全不影響既有正常路徑的行為。
    """

    config_path: str
    plan_path: str
    identity_table_path: str
    results_dir: str
    formal: bool
    checkpoint_root: str | None
    checkpoint_every_hours: int
    hour_major_config: HourMajorConfig
    # WP-17：實際採用的回溯模式與設定檔宣稱的模式（見 backtrack_horizon.py）。
    # 兩者不一致時 ``_resolve_params`` 已經 fail-closed 檢查過
    # ``LBT_NATIVE_ACKNOWLEDGE_CONFIG_MODE_MISMATCH``，這裡只是把結果凍結下來，
    # 供 ``_run_unit_entrypoint`` 寫進每個結果摘要（不重新檢查一次）。
    effective_backtrack_mode: str = "full_horizon_from_deposition"
    config_declared_backtrack_mode: str | None = None
    requested_horizon_days: float = 30.0
    test_force_interrupt_after_hours: int | None = None
    # WP-18：實際採用的實驗案例（見 physics_case.py）。``include_stokes=False``
    # （``no_stokes``）時 ``_build_worker_resources_entrypoint`` 會把它傳給
    # ``production_wiring.make_domain_resources_factory``，讓該區域完全不讀波浪
    # 資料；``require_wave_data`` 只在 ``include_stokes=True`` 時有意義，見該
    # 函式 docstring。預設值維持 WP-18 之前「永遠嘗試 Stokes、資料缺漏時靜默
    # 退化」的既有行為，不破壞既有呼叫端。
    experiment_case_id: str = "finite_depth_stokes"
    include_stokes: bool = True
    diffusion_method: str = "smagorinsky_workbook_eq10"
    require_wave_data: bool = False


def _env_smagorinsky_impl(env: Mapping[str, str]) -> str:
    """``LBT_NATIVE_SMAGORINSKY_IMPL``：``numba``（預設，統計等價的加速移植）或 ``baytrace``（原版）。"""

    value = (env.get("LBT_NATIVE_SMAGORINSKY_IMPL") or "numba").strip().lower()
    if value not in {"baytrace", "numba"}:
        raise HourMajorProviderError(
            f"LBT_NATIVE_SMAGORINSKY_IMPL 只接受 baytrace 或 numba，收到 {value!r}"
        )
    return value


def _resolve_params(env: Mapping[str, str]) -> HourMajorProviderParams:
    """從環境變數解析出 :class:`HourMajorProviderParams`（見模組文件環境變數契約）。"""

    config_path = _require_env(env, "LBT_NATIVE_CONFIG")
    config = load_config(config_path, formal_release=False)
    master_seed = config.scenarios.master_seed if config.scenarios.master_seed is not None else 0
    max_backtrack_seconds = (
        config.boundaries.max_backtrack_days * 86400.0
        if config.boundaries.max_backtrack_days is not None
        else HourMajorConfig().max_backtrack_seconds
    )
    requested_horizon_days = (
        float(config.boundaries.max_backtrack_days)
        if config.boundaries.max_backtrack_days is not None
        else HourMajorConfig().max_backtrack_seconds / 86400.0
    )

    # WP-17：回溯模式明確化——native-run 這一步不重新規劃工作單元（單元的
    # resolved_max_backtrack_seconds／pre_window_deposition 已經由 native-plan
    # 決定並寫進計畫檔，見 orchestrator.resolve_native_work_unit_backtrack_horizons），
    # 但仍必須重新核對「這次執行實際採用的模式」與設定檔宣稱的模式是否一致，
    # 避免有人用同一份計畫檔搭配改過 backtrack_mode 的設定檔重跑、卻沒有人注意到。
    bed_residence_time = getattr(config.scenarios, "bed_residence_time", None)
    config_declared_backtrack_mode = (
        getattr(bed_residence_time, "backtrack_mode", None) if bed_residence_time is not None else None
    )
    effective_backtrack_mode = resolve_backtrack_mode(cli_backtrack_mode=None, env=dict(env))
    require_mode_acknowledged_if_mismatched(
        effective_backtrack_mode=effective_backtrack_mode,
        config_declared_backtrack_mode=config_declared_backtrack_mode,
        acknowledge_config_mode_mismatch=_env_flag(
            env, "LBT_NATIVE_ACKNOWLEDGE_CONFIG_MODE_MISMATCH", default=False
        ),
    )
    # WP-18：實驗案例明確化——與上面的回溯模式核對同一個理由，native-run 這一步
    # 不重新規劃工作單元，但每個 worker 都要知道這次實際要不要建構波浪解析器
    # （見 physics_case.py／production_wiring.build_domain_resources 的
    # include_stokes 參數）。
    resolved_physics_case = resolve_physics_case(cli_experiment_case_id=None, env=dict(env))
    require_wave_data = _env_flag(env, "LBT_NATIVE_REQUIRE_WAVE_DATA", default=False)

    dt_max_seconds = (
        config.integration.dt_max_seconds
        if config.integration.dt_max_seconds is not None
        else HourMajorConfig().dt_max_seconds
    )
    output_interval_seconds = (
        config.integration.output_interval_seconds
        if config.integration.output_interval_seconds is not None
        else HourMajorConfig().output_interval_seconds
    )
    hour_major_config = HourMajorConfig(
        max_backtrack_seconds=float(max_backtrack_seconds),
        dt_max_seconds=float(dt_max_seconds),
        cfl_number=_env_float(env, "LBT_NATIVE_CFL_NUMBER", default=HourMajorConfig().cfl_number),
        h0=_env_float(env, "LBT_NATIVE_H0", default=HourMajorConfig().h0),
        horcon=_env_float(env, "LBT_NATIVE_HORCON", default=HourMajorConfig().horcon),
        enable_diffusion=_env_flag(env, "LBT_NATIVE_ENABLE_DIFFUSION", default=True),
        output_interval_seconds=float(output_interval_seconds),
        deposition_consecutive_hours=_env_int(
            env,
            "LBT_NATIVE_DEPOSITION_CONSECUTIVE_HOURS",
            default=HourMajorConfig().deposition_consecutive_hours,
        ),
        deposition_tolerance_m=_env_float(
            env, "LBT_NATIVE_DEPOSITION_TOLERANCE_M", default=HourMajorConfig().deposition_tolerance_m
        ),
        master_seed=int(master_seed),
        stokes_depth_threshold_ratio=_env_optional_float(env, "LBT_NATIVE_STOKES_DEPTH_THRESHOLD_RATIO"),
        smagorinsky_impl=_env_smagorinsky_impl(env),
    )
    return HourMajorProviderParams(
        config_path=config_path,
        plan_path=_require_env(env, "LBT_NATIVE_PLAN"),
        identity_table_path=_require_env(env, "LBT_NATIVE_IDENTITY_TABLE"),
        results_dir=_require_env(env, "LBT_NATIVE_RESULTS_DIR"),
        formal=_env_flag(env, "LBT_NATIVE_FORMAL", default=True),
        checkpoint_root=env.get("LBT_NATIVE_CHECKPOINT_ROOT") or None,
        checkpoint_every_hours=_env_int(env, "LBT_NATIVE_CHECKPOINT_EVERY_HOURS", default=24),
        hour_major_config=hour_major_config,
        effective_backtrack_mode=effective_backtrack_mode,
        config_declared_backtrack_mode=config_declared_backtrack_mode,
        requested_horizon_days=requested_horizon_days,
        test_force_interrupt_after_hours=_env_optional_int(
            env, "LBT_NATIVE_TEST_FORCE_INTERRUPT_AFTER_HOURS"
        ),
        experiment_case_id=resolved_physics_case.experiment_case_id,
        include_stokes=resolved_physics_case.include_stokes,
        diffusion_method=resolved_physics_case.diffusion_method,
        require_wave_data=require_wave_data,
    )


# ══════════════════════════════════════════════════════════════════════════
# 二、每個 worker 長駐的資源：只在第一次遇到某個流場區域時呼叫一次
#    ``build_domain_resources``，之後的段重用同一份。
# ══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class HourMajorWorkerResources:
    """一個長駐 worker 子程序持有的全部資源；``run_unit`` 每次呼叫都重用。"""

    params: HourMajorProviderParams
    handles: BayTraceModuleHandles
    identity_table: ParticleIdentityTable
    members: Mapping[str, tuple[NativeWorkUnit, ...]]
    domain_resources_factory: Any  # production_wiring.DomainResourcesFactory
    domain_resources_cache: dict[str, DomainResources]


def _build_worker_resources_entrypoint(
    params: HourMajorProviderParams, worker_id: int
) -> HourMajorWorkerResources:
    """``build_worker_resources(worker_id)``：每個 worker 子程序只呼叫一次。

    模組層級函式（非閉包）：``multiprocessing`` 的 ``spawn`` 需要能夠 pickle
    匯入它；``functools.partial`` 綁定 ``params`` 後仍然可以被 pickle（見
    ``production_wiring._domain_resources_factory_entrypoint`` 同一個理由）。
    """

    del worker_id  # 目前不需要依 worker_id 做任何差異化；保留參數只為符合既有簽章。
    baytrace_root = os.environ.get("LBT_BAYTRACE_ROOT")
    if not baytrace_root:
        raise HourMajorProviderError(
            "環境變數 LBT_BAYTRACE_ROOT 未設定；載入 BayTrace 原版模組必須明示其原始碼根目錄"
        )
    handles = load_baytrace_modules(baytrace_root)
    identity_table = load_particle_identity_table_npz(params.identity_table_path)
    _, payload = native_orchestrator.read_native_plan_file(params.plan_path)
    members = native_orchestrator.read_native_hour_major_members(payload)
    if not members:
        raise HourMajorProviderError(
            f"計畫檔 {params.plan_path!r} 沒有 hour_major_members 欄位；"
            "hour_major_provider 只能執行 --schedule hour-major（或切段）產生的計畫檔"
        )
    domain_resources_factory = production_wiring.make_domain_resources_factory(
        params.config_path,
        formal=params.formal,
        include_stokes=params.include_stokes,
        require_wave_data=params.require_wave_data,
    )
    return HourMajorWorkerResources(
        params=params,
        handles=handles,
        identity_table=identity_table,
        members=members,
        domain_resources_factory=domain_resources_factory,
        domain_resources_cache={},
    )


def _domain_resources_for(resources: HourMajorWorkerResources, analysis_region_id: str) -> DomainResources:
    """按需建構、建過即快取的區域資源查詢（見模組文件「已知限制」第 3 點）。

    **重要**：這裡的鍵值與傳給 ``domain_resources_factory`` 的參數必須是
    ``analysis_region_id``（例如 ``"A"``），不是 ``flow_domain_id``（例如
    ``"northeast_taiwan_common_cache_v3"``）——見 ``production_wiring.
    build_domain_resources`` docstring 明確記載的契約：其 ``domain_id`` 參數
    「是 ``analysis_region_id``……不是 ``flow_domain_id``」。這兩者只有在呼叫端
    省略 ``--region-to-flow-domain``（即 ``plan_native_work_units`` 退化為
    ``flow_domain_id == analysis_region_id``）時才會剛好相同；正式部署一定會
    提供 ``--region-to-flow-domain``（例如區域 ``"A"`` 對應
    ``northeast_taiwan_common_cache_v3``），兩者不同，呼叫端若傳錯會讓
    ``build_domain_resources`` 用錯誤的 ``domain_id`` 去查 ``ProjectConfig.
    domains``，得到錯誤的 local domain／interest region，甚至直接找不到對應
    區域而拋例外——呼叫端（``_run_unit_entrypoint``）必須傳入真正的
    ``analysis_region_id``，本函式不再接受把 ``flow_domain_id`` 當參數名稱，
    避免呼叫端望文生義傳錯。
    """

    cached = resources.domain_resources_cache.get(analysis_region_id)
    if cached is not None:
        return cached
    # 每個 worker 只保留「目前這個區域」的資源：資料依區域分批就緒時，worker 會先後處理
    # 不同區域，若全部留在記憶體，56 個 worker 各自累積四區網格與讀取器會超過伺服器記憶體。
    # 換區時重建資源只需數秒，遠小於一個排程單元的運算時間。
    resources.domain_resources_cache.clear()
    built = resources.domain_resources_factory(analysis_region_id)
    resources.domain_resources_cache[analysis_region_id] = built
    return built


# ══════════════════════════════════════════════════════════════════════════
# 三、重量級輸出序列化：BackwardDriverResult ←→ .npz
# ══════════════════════════════════════════════════════════════════════════


def save_backward_driver_result_npz(path: str | Path, result: BackwardDriverResult) -> None:
    """把一個原始工作單元的完整 ``BackwardDriverResult`` 寫成單一 ``.npz``。

    ``departures`` 是 ``DepartureRecord`` 的 tuple（純量欄位的 dataclass），
    改存成等長的並排陣列（``departure_row``／``departure_status``／
    ``departure_x_m``／...），讀回時再逐列重建成 ``DepartureRecord`` tuple。
    ``diagnostics``／``overhead_diagnostics`` 在小時主軸路徑下固定是歸零的
    值（見 ``hour_major`` 模組文件「分批粒度」一節），本函式仍誠實寫入
    （而非省略），避免讀回時對欄位是否存在做特殊判斷。

    寫入方式：先寫到帶亂數後綴的暫存路徑，再以 ``os.replace`` 原子改名，
    與 ``orchestrator._atomic_write_json`` 相同的「不留半寫檔案」原則。
    """

    import os as _os
    import uuid as _uuid

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f".{target.name}.tmp-{_os.getpid()}-{_uuid.uuid4().hex[:8]}")

    n_departures = len(result.departures)
    departure_row = np.array([d.row_particle_index for d in result.departures], dtype=np.int64)
    departure_status = np.array([d.status for d in result.departures], dtype=np.int32)
    departure_x = np.array([d.x_m for d in result.departures], dtype=np.float64)
    departure_y = np.array([d.y_m for d in result.departures], dtype=np.float64)
    departure_z = np.array([d.z_m for d in result.departures], dtype=np.float64)
    departure_utc = np.array([d.utc_ns for d in result.departures], dtype=np.int64)

    payload = {
        "row_particle_index": np.asarray(result.row_particle_index, dtype=np.int64),
        "final_status": np.asarray(result.final_status, dtype=np.int32),
        "final_x_m": np.asarray(result.final_x_m, dtype=np.float64),
        "final_y_m": np.asarray(result.final_y_m, dtype=np.float64),
        "final_z_m": np.asarray(result.final_z_m, dtype=np.float64),
        "final_utc_ns": np.asarray(result.final_utc_ns, dtype=np.int64),
        "trajectory_utc_ns": np.asarray(result.trajectory_utc_ns, dtype=np.int64),
        "trajectory_x_m": np.asarray(result.trajectory_x_m, dtype=np.float64),
        "trajectory_y_m": np.asarray(result.trajectory_y_m, dtype=np.float64),
        "trajectory_z_m": np.asarray(result.trajectory_z_m, dtype=np.float64),
        "trajectory_status": np.asarray(result.trajectory_status, dtype=np.int32),
        "departure_row": departure_row,
        "departure_status": departure_status,
        "departure_x_m": departure_x,
        "departure_y_m": departure_y,
        "departure_z_m": departure_z,
        "departure_utc_ns": departure_utc,
        "n_departures": np.asarray(n_departures, dtype=np.int64),
        "hours_run": np.asarray(int(result.hours_run), dtype=np.int64),
    }
    with tmp_path.open("wb") as handle:
        np.savez(handle, **payload)
    _os.replace(tmp_path, target)


def load_backward_driver_result_npz(path: str | Path) -> BackwardDriverResult:
    """讀回 :func:`save_backward_driver_result_npz` 寫出的 ``.npz``。"""

    with np.load(str(path), allow_pickle=False) as npz:
        n_departures = int(npz["n_departures"])
        departures = tuple(
            DepartureRecord(
                row_particle_index=int(npz["departure_row"][i]),
                status=int(npz["departure_status"][i]),
                x_m=float(npz["departure_x_m"][i]),
                y_m=float(npz["departure_y_m"][i]),
                z_m=float(npz["departure_z_m"][i]),
                utc_ns=int(npz["departure_utc_ns"][i]),
            )
            for i in range(n_departures)
        )
        return BackwardDriverResult(
            row_particle_index=np.asarray(npz["row_particle_index"], dtype=np.int64),
            final_status=np.asarray(npz["final_status"], dtype=np.int32),
            final_x_m=np.asarray(npz["final_x_m"], dtype=np.float64),
            final_y_m=np.asarray(npz["final_y_m"], dtype=np.float64),
            final_z_m=np.asarray(npz["final_z_m"], dtype=np.float64),
            final_utc_ns=np.asarray(npz["final_utc_ns"], dtype=np.int64),
            trajectory_utc_ns=np.asarray(npz["trajectory_utc_ns"], dtype=np.int64),
            trajectory_x_m=np.asarray(npz["trajectory_x_m"], dtype=np.float64),
            trajectory_y_m=np.asarray(npz["trajectory_y_m"], dtype=np.float64),
            trajectory_z_m=np.asarray(npz["trajectory_z_m"], dtype=np.float64),
            trajectory_status=np.asarray(npz["trajectory_status"], dtype=np.int32),
            departures=departures,
            diagnostics=SubstepDiagnostics(),
            overhead_diagnostics=HourlyOverheadDiagnostics(),
            hours_run=int(npz["hours_run"]),
        )


def result_npz_path(results_dir: str | Path, original_unit_id: str) -> Path:
    """統一的檔名規則，``native-postprocess`` 用同一個函式重建路徑。"""

    return Path(results_dir) / f"{original_unit_id}.driver_result.npz"


# ══════════════════════════════════════════════════════════════════════════
# 四、run_unit：辨識 hourmajor:: 前綴，執行整個流場區域（或其中一段）
# ══════════════════════════════════════════════════════════════════════════


def _status_counts(final_status: np.ndarray) -> dict[str, int]:
    counts: dict[str, int] = {}
    values, tallies = np.unique(np.asarray(final_status, dtype=np.int32), return_counts=True)
    for value, tally in zip(values.tolist(), tallies.tolist(), strict=True):
        counts[NATIVE_STATUS_DESCRIPTIONS.get(int(value), str(value))] = int(tally)
    return counts


def _run_unit_entrypoint(
    params: HourMajorProviderParams, resources: HourMajorWorkerResources, spec: NativeWorkUnitSpec
) -> dict[str, Any]:
    """``run_unit(resources, spec)``：見模組文件「為什麼需要這個模組」第 2 點。"""

    if not spec.unit_id.startswith(_HOUR_MAJOR_PREFIX):
        raise HourMajorProviderError(
            f"spec.unit_id={spec.unit_id!r} 不是 {_HOUR_MAJOR_PREFIX!r} 前綴的小時主軸排程單元；"
            "hour_major_provider 的 run_unit 只接受 orchestrator.plan_native_hour_major_work_units"
            "／plan_native_hour_major_segments 摺疊出的排程單元，其餘 unit_id 請改用其他"
            "（例如逐工作單元路徑的）resource-provider 實作"
        )
    member_units = resources.members.get(spec.unit_id)
    if member_units is None:
        raise HourMajorProviderError(
            f"計畫檔的 hour_major_members 找不到 {spec.unit_id!r} 底下的原始工作單元；"
            "請確認 native-run 使用的計畫檔與 native-plan 產生的是同一份"
        )

    # ``spec.analysis_region_id`` 是切段時摺疊出的 sentinel 值
    # （``"__hour_major_domain__"``，見 ``orchestrator.plan_native_hour_major_
    # segments``），不能拿來查詢區域資源；真正的 ``analysis_region_id`` 要從
    # 段底下原始工作單元自己的欄位取得。同一段理論上只會來自同一個
    # ``flow_domain_id``（切段就是依 ``flow_domain_id`` 分組），但
    # ``analysis_region_id → flow_domain_id`` 的對照表由呼叫端在 native-plan
    # 階段提供，理論上允許多個 ``analysis_region_id`` 共用同一個
    # ``flow_domain_id``；若真的發生本函式無法決定該用哪一個
    # ``analysis_region_id`` 建構資源，fail-closed 明確報錯，不猜測。
    member_regions = {unit.analysis_region_id for unit in member_units}
    if len(member_regions) != 1:
        raise HourMajorProviderError(
            f"排程單元 {spec.unit_id!r} 底下的原始工作單元橫跨多個 analysis_region_id"
            f"（{sorted(member_regions)!r}），hour_major_provider 目前只支援一個排程單元"
            "對應單一 analysis_region_id；若正式設定真的讓多個區域共用同一個"
            "flow_domain_id，需要另外設計資源查詢方式，此處拒絕猜測"
        )
    analysis_region_id = next(iter(member_regions))
    domain_resources = _domain_resources_for(resources, analysis_region_id)

    # WP-17：native-plan 已經依實際採用的回溯模式，把每個原始工作單元標成
    # 「進入運算」或「pre_window_deposition」（沉底年齡 ≥ H，固定日曆窗之外，
    # 不進入運算），見 orchestrator.resolve_native_work_unit_backtrack_horizons。
    # 這裡先把兩者分開：pre-window 單元完全不建構小時主軸批次、不查詢任何流場，
    # 直接組出結果；其餘單元才照舊路徑跑 run_flow_domain_hour_major。
    pre_window_units = [unit for unit in member_units if unit.pre_window_deposition]
    runnable_units = [unit for unit in member_units if not unit.pre_window_deposition]

    results_dir = Path(params.results_dir)
    member_summaries: dict[str, dict[str, Any]] = {}

    def _unit_horizon_provenance(unit: NativeWorkUnit) -> dict[str, Any]:
        return {
            "declared_config_backtrack_mode": params.config_declared_backtrack_mode,
            "effective_backtrack_mode": params.effective_backtrack_mode,
            "requested_horizon_days": params.requested_horizon_days,
            "resolved_max_backtrack_seconds": unit.resolved_max_backtrack_seconds,
            "pre_window_deposition": bool(unit.pre_window_deposition),
            "bed_residence_age_hours": unit.bed_residence_age_hours,
        }

    def _physics_case_provenance() -> dict[str, Any]:
        # WP-18：每個結果摘要都記錄實際採用的實驗案例，避免事後只看摺疊後的
        # 排程單元猜不出這批結果到底有沒有 Stokes（見 physics_case.py）。
        return {
            "experiment_case_id": params.experiment_case_id,
            "include_stokes": params.include_stokes,
            "diffusion_method": params.diffusion_method,
        }

    for unit in pre_window_units:
        table = resources.identity_table
        driver_result = build_pre_window_deposition_result(
            row_particle_index=np.arange(unit.particle_count, dtype=np.int64),
            x0_m=table.x0_m[unit.particle_row_indices],
            y0_m=table.y0_m[unit.particle_row_indices],
            z0_m=table.z0_m[unit.particle_row_indices],
            arrival_time_utc_ns=int(unit.arrival_time_utc_ns),
        )
        save_backward_driver_result_npz(result_npz_path(results_dir, unit.unit_id), driver_result)
        member_summaries[unit.unit_id] = {
            "particle_count": unit.particle_count,
            "hours_run": 0,
            "final_status_counts": _status_counts(driver_result.final_status),
            "result_npz": str(result_npz_path(results_dir, unit.unit_id)),
            **_unit_horizon_provenance(unit),
            **_physics_case_provenance(),
        }

    if not runnable_units:
        # 整個排程單元底下全部到達時刻都是 pre-window：不需要（也不應該）呼叫
        # run_flow_domain_hour_major——沒有任何粒子需要查詢流場。
        return {
            "domain_unit_id": spec.unit_id,
            "flow_domain_id": spec.flow_domain_id,
            "member_unit_ids": sorted(member_summaries),
            "member_unit_count": len(member_summaries),
            "member_summaries": member_summaries,
            "domain_hours_walked": 0,
            "domain_kernel_call_count": 0,
            "wave_data_available": None,
            "smagorinsky_impl": params.hour_major_config.smagorinsky_impl,
            "all_members_pre_window_deposition": True,
            "declared_config_backtrack_mode": params.config_declared_backtrack_mode,
            "effective_backtrack_mode": params.effective_backtrack_mode,
            "requested_horizon_days": params.requested_horizon_days,
            **_physics_case_provenance(),
        }

    batch = build_hour_major_domain_batch(
        resources.identity_table,
        runnable_units,
        interest_regions=domain_resources.interest_regions,
        local_domains=domain_resources.local_domains,
        local_equals_flow=domain_resources.local_equals_flow,
    )

    checkpoint_dir: str | None = None
    if params.checkpoint_root:
        checkpoint_dir = str(Path(params.checkpoint_root) / spec.unit_id)

    wave_resolver = (
        domain_resources.wave_resolver_factory()
        if domain_resources.wave_resolver_factory is not None
        else None
    )

    if params.test_force_interrupt_after_hours is not None:
        # ── 測試專用中斷模擬（見模組文件「測試專用：模擬工作程序中途消失」
        # 一節）；正式部署絕不會設定 test_force_interrupt_after_hours，這個
        # 分支的唯一目的是讓測試套件能重現「子程序被系統砍掉」的場景。 ──
        if checkpoint_dir is None:
            raise HourMajorProviderError(
                "LBT_NATIVE_TEST_FORCE_INTERRUPT_AFTER_HOURS 需要同時設定"
                " LBT_NATIVE_CHECKPOINT_ROOT；沒有 checkpoint 的情況下模擬"
                "「中斷後續跑」沒有意義（下次重跑只會從頭重算，不是真的在"
                "驗證續跑機制），這是呼叫端的設定錯誤，本模組刻意 fail-closed"
                "，不會靜默忽略這個測試旋鈕"
            )
        # 這次呼叫本身會把 checkpoint 正常寫進 checkpoint_dir（hour_major.py
        # 既有行為，見該函式 docstring 的 _max_master_hours 說明），但回傳的
        # 是尚未全部終止的部分結果——刻意不使用、不寫成 .driver_result.npz。
        run_flow_domain_hour_major(
            domain_resources.driver_grid,
            resources.handles,
            batch,
            config=params.hour_major_config,
            forcing_resolver=domain_resources.forcing_resolver_factory(),
            wave_field_resolver=wave_resolver,
            checkpoint_dir=checkpoint_dir,
            checkpoint_every_hours=params.checkpoint_every_hours,
            _max_master_hours=params.test_force_interrupt_after_hours,
        )
        raise HourMajorProviderError(
            f"[測試專用中斷模擬] 排程單元 {spec.unit_id!r} 已在處理滿"
            f" {params.test_force_interrupt_after_hours} 個小時主軸小時、並把"
            f" checkpoint 寫入 {checkpoint_dir!r} 之後，模擬工作程序在完成前"
            "失去存活（例如被 OOM killer 砍掉、或機器重開）；此例外是刻意"
            "觸發，不代表真正的執行失敗——下一次不設定"
            " LBT_NATIVE_TEST_FORCE_INTERRUPT_AFTER_HOURS 重跑 native-run 時"
            "，run_flow_domain_hour_major 會讀到這份 checkpoint 並從中斷處"
            "續跑到底"
        )

    run_result = run_flow_domain_hour_major(
        domain_resources.driver_grid,
        resources.handles,
        batch,
        config=params.hour_major_config,
        forcing_resolver=domain_resources.forcing_resolver_factory(),
        wave_field_resolver=wave_resolver,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every_hours=params.checkpoint_every_hours,
    )

    for unit in runnable_units:
        driver_result = run_result.unit_results.get(unit.unit_id)
        if driver_result is None:
            raise HourMajorProviderError(
                f"HourMajorRunResult 缺少原始工作單元 {unit.unit_id!r} 的結果，這是內部錯誤"
            )
        save_backward_driver_result_npz(result_npz_path(results_dir, unit.unit_id), driver_result)
        member_summaries[unit.unit_id] = {
            "particle_count": unit.particle_count,
            "hours_run": int(driver_result.hours_run),
            "final_status_counts": _status_counts(driver_result.final_status),
            "result_npz": str(result_npz_path(results_dir, unit.unit_id)),
            **_unit_horizon_provenance(unit),
            **_physics_case_provenance(),
        }

    return {
        "domain_unit_id": spec.unit_id,
        "flow_domain_id": spec.flow_domain_id,
        "member_unit_ids": sorted(member_summaries),
        "member_unit_count": len(member_summaries),
        "member_summaries": member_summaries,
        "domain_hours_walked": int(run_result.domain_hours_walked),
        "domain_kernel_call_count": int(run_result.domain_kernel_call_count),
        "wave_data_available": wave_resolver is not None,
        # 每個排程單元都記錄實際使用的 Smagorinsky 實作，後處理與報告可據此區分。
        "smagorinsky_impl": params.hour_major_config.smagorinsky_impl,
        "declared_config_backtrack_mode": params.config_declared_backtrack_mode,
        "effective_backtrack_mode": params.effective_backtrack_mode,
        "requested_horizon_days": params.requested_horizon_days,
        **_physics_case_provenance(),
    }


def make_hour_major_resource_provider() -> tuple[Any, Any]:
    """``lbt native-run --resource-provider`` 指定的模組層級 factory 函式。

    呼叫時不帶參數（見模組文件「環境變數契約」），在呼叫當下（父程序）讀一次
    ``os.environ`` 凍結出 :class:`HourMajorProviderParams`，回傳綁定了這份
    參數的 ``(build_worker_resources, run_unit)``；兩者都是對模組層級函式
    的 ``functools.partial``，可以安全地在 ``multiprocessing`` ``spawn``
    context 下傳給子程序（與 ``production_wiring.make_domain_resources_
    factory`` 相同的 pickle 策略，見該函式 docstring）。
    """

    import functools

    params = _resolve_params(os.environ)
    build_worker_resources = functools.partial(_build_worker_resources_entrypoint, params)
    run_unit = functools.partial(_run_unit_entrypoint, params)
    return build_worker_resources, run_unit
