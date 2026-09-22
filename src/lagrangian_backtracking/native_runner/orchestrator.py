"""WP-7：把已建好的粒子身分表切成工作單元，跨核心長駐平行驅動 ``run_backward_batch``。

背景與定位
----------
``backward_driver.run_backward_batch`` 已經是效能核心（實測約 5 µs/粒子步），本模組
不改動任何物理或數值邏輯，只負責「怎麼把 50,000×M（正式進階 500,000×M）顆粒子分配
給有限的 CPU 核心、以最快速度跑完，且中斷後能安全接續」這件工程問題。因此本模組
完全不 import 任何會觸發 Numba JIT 編譯或讀取大型陣列的模組，讓 ``native-plan``／
單元測試可以在不到一秒內完成規劃與缺口檢查；真正呼叫 BayTrace 核心的動作，全部
封裝在呼叫端注入的 ``build_worker_resources``／``run_unit`` 兩個 callable 裡（見下方
「執行引擎」一節），本模組只負責在子程序裡呼叫它們、記錄結果與處理失敗。

工作單元的切分依據（対應工作書「依流場區域、到達時刻、沉降速度」）
--------------------------------------------------------------------
一次 ``run_backward_batch`` 呼叫要求同批粒子共用「同一個沉降速度」（RK4 全部四階
都要用同一個疊加常數，見 ``backward_driver`` 模組文件事實 1）與「同一個到達時刻」
（同步的 BayTrace 全域時鐘），也只能傳入單一 ``interest_region_polygon``（見其
docstring）。而 ``interest_region_polygon`` 是逐站（``study_site_id``）定義的
``own_local_domain``——貢寮與龜山島雖然共用同一個流場區域（``analysis_region_id
== "A"``）與可能相同的到達時刻，但兩站的關注區域中心不同，不能合成一次 kernel
呼叫。因此本模組定義的「工作單元」（``NativeWorkUnit``）鍵值是四元組
``(analysis_region_id, arrival_time_utc_ns, settling_velocity_mps, study_site_id)``：
前三者合乎「同批同沉降同到達」的物理正確性要求，第四者則對應
``interest_region_polygon`` 的實際粒度。

排程時另外定義較粗的「局部性群組」鍵值 ``(analysis_region_id, arrival_time_utc_ns)``：
同一群組內的所有工作單元（不同沉降速度、不同站別）都需要讀取「同一個流場區域、
同一個 30 天回溯窗」的逐時 OCM 資料（即 ``forcing_bridge.NativeForcingBridgeReader``
一次建構後可重複使用的同一個實例），差別只在於粒子起點與沉降速度。把同一個局部性
群組的所有工作單元排給同一個 worker、且緊接著執行，可以讓 forcing reader、其記憶體
內快照快取（見 ``forcing_bridge`` 模組文件）與作業系統分頁快取全部重複利用，不必
為每個沉降速度或站別重新開檔、重新掃描月份 memmap。

執行引擎
--------
每個 worker 是一個長駐子程序：程序啟動時只呼叫一次呼叫端提供的
``build_worker_resources(worker_id)``（於其中載入 BayTrace 模組、建構網格、視需要
建構 forcing reader 等重量級資源），之後在同一個程序記憶體內連續處理排定給它的
一整串工作單元，呼叫呼叫端提供的 ``run_unit(resources, unit)``；不像舊架構「每個
shard 重開一次 CLI 子程序」，也不使用 ``ProcessPoolExecutor`` 預設的「工作竊取」
語意（那會打散刻意安排的局部性排程）。

``run_unit`` 只需要回傳一個可以被 ``json.dumps`` 序列化的摘要字典（例如各終止狀態碼
的粒子數、實際回溯小時數）；真正的重量級輸出（軌跡 shard、checkpoint 等）由
``run_unit`` 自行以其他模組的既有寫出邏輯負責，本模組不假設、不重新定義那些格式。
本模組只負責把這個輕量摘要，連同單元識別碼、輸入雜湊與時間戳記，原子性寫成一筆
「完成證據」，供續跑時判斷是否可以跳過。

續跑與失敗隔離
--------------
每個工作單元完成時，先把摘要寫到 ``<unit_id>.result.json.tmp-<pid>-<random>``，
再以 ``os.replace``（同檔案系統內保證原子）改名為正式檔案，避免半寫檔案被誤判為
完成。完成證據包含 ``input_hash``（涵蓋粒子身分表整體內容雜湊、工作單元鍵值與該
單元實際包含的列位置），續跑時只有雜湊完全相符才跳過重算，任何輸入變動（重新建表、
換了不同粒子子集）都會讓舊完成證據失效、重新計算該單元。

單一工作單元擲出例外時，worker 迴圈只記錄錯誤與完整 traceback 字串，繼續處理同一
worker 佇列裡的下一個單元；``build_worker_resources`` 本身失敗（例如某個 worker
分到的流場區域資料損毀）則讓該 worker 全部單元一併記為失敗、但完全不影響其他
worker，最終由 ``run_native_plan`` 彙整全部失敗清單，不靜默吞掉任何錯誤。

已知限制（誠實列出，不用假的完整性掩蓋）
------------------------------------------
1. 本模組不知道、也不假設如何從 ``analysis_region_id`` 找出實際的 OCM 月份目錄、
   如何建構逐站 ``interest_region_polygon``、或如何取得 Stokes 波浪場——這些是
   ``config.py``、``geometry.py``、``mesh.py`` 與（另一工作包負責的）``wave_bridge``
   的責任。``run_native_plan`` 透過 ``build_worker_resources``／``run_unit`` 兩個
   純函式介面接受這些邏輯的具體實作，測試與 CLI 可以用合成資源或真正的生產資源
   注入，本模組本身兩者都不 import。
2. 排程演算法（見 ``plan_worker_schedule``）用「每工作單元粒子數」當作計算成本的
   近似代理；不同沉降速度可能因為擱淺、沉底或缺值提早停止而實際耗時不同，本模組
   不宣稱這是精確的執行時間估計，只是比「完全忽略局部性」更好的啟發式。
3. 到達時刻缺口安全性檢查（見下方「缺口安全的到達時刻」一節）只在呼叫端提供完整
   候選序列時，才能真正重用既有 ``arrival_times.select_arrival_times`` 重新選點；
   若只有最終身分表、沒有原始候選序列，本模組只能誠實回報「此到達時刻不安全、且
   無法在這一層自動替換」，不會發明新的選點規則。
"""

from __future__ import annotations

import json
import multiprocessing as _multiprocessing
import os
import time
import traceback
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool as _BrokenProcessPool
from dataclasses import asdict, dataclass
from dataclasses import replace as _dataclass_replace
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from ..arrival_times import ArrivalTime, select_arrival_times
from ..provenance import CodeProvenance, collect_code_provenance
from .backtrack_horizon import (
    BED_RESIDENCE_MODE_FIXED_CALENDAR_WINDOW,
    resolve_unit_backtrack_horizon,
)
from .forcing_bridge import ForcingGapInventoryReport
from .particle_input import BatchGroup, ParticleIdentityTable, validate_batch_partition

__all__ = [
    "NativeWorkUnit",
    "NativeWorkUnitSpec",
    "plan_native_work_units",
    "plan_native_hour_major_work_units",
    "HourMajorSegmentPlanReport",
    "estimate_segment_extra_cost_fraction",
    "recommend_hour_major_segment_counts",
    "estimate_hour_major_unit_cost_seconds",
    "estimate_hour_major_segment_costs",
    "estimate_total_wall_clock_seconds",
    "choose_hour_major_segment_count",
    "recommend_hour_major_segment_counts_by_workload",
    "plan_native_hour_major_segments",
    "NativeWorkPlanSummary",
    "summarize_native_work_plan",
    "estimate_formal_native_work_plan",
    "plan_worker_schedule",
    "read_available_memory_bytes",
    "estimate_worker_count",
    "ArrivalGapCheck",
    "check_arrival_gap_safety",
    "ArrivalReselectionInputs",
    "derive_backward_window_available",
    "rebuild_gap_safe_arrival_times",
    "StratumCoverageDiff",
    "summarize_stratum_coverage",
    "diff_stratum_coverage",
    "build_native_run_manifest",
    "NativeUnitOutcome",
    "WorkerRunReport",
    "NativeRunSummary",
    "run_native_plan",
    "read_progress_snapshot",
    "write_native_plan_file",
    "read_native_plan_file",
    "read_native_hour_major_members",
]


_HOUR_NS = 3_600_000_000_000
_DEFAULT_MAX_BACKTRACK_SECONDS = 30.0 * 86400.0


class NativeOrchestratorError(RuntimeError):
    """本模組專用例外，與 ``ParticleInputError``／``ForcingBridgeError`` 區分責任層。"""


# ══════════════════════════════════════════════════════════════════════════
# 1. 工作單元切分
# ══════════════════════════════════════════════════════════════════════════


def _slug_float(value: float) -> str:
    """把浮點數轉成穩定、檔名安全的十六進位片段，用於組合 ``unit_id``。

    直接把浮點數格式化進檔名（例如 ``-0.0001``）在不同平台的 repr 可能不穩定
    （尾數、科學記號門檻），改以 IEEE-754 位元表示的十六進位字串，確保同一個
    浮點值永遠對應同一個檔名片段，且不同浮點值幾乎不可能碰撞。
    """

    bits = np.float64(value).view(np.int64)
    return format(int(bits) & 0xFFFFFFFFFFFFFFFF, "016x")


@dataclass(frozen=True, slots=True)
class NativeWorkUnit:
    """一組可以直接餵給單次 ``run_backward_batch`` 呼叫的粒子列位置集合。

    ``particle_row_indices`` 是遞增排序、無重複的 ``int64`` 一維陣列，內容是這批
    粒子在來源 ``ParticleIdentityTable`` 中的列位置（語意與 ``BatchGroup.indices``
    完全一致，可直接用於 fancy indexing）。``input_hash`` 涵蓋來源身分表的整體內容
    雜湊、本單元的四個鍵值與 ``particle_row_indices`` 本身，任何一項改變都會讓雜湊
    改變，是續跑判斷「這個單元的輸入是否与上次完全相同」的唯一依據。
    """

    unit_id: str
    analysis_region_id: str
    flow_domain_id: str
    study_site_id: str
    arrival_time_utc_ns: int
    settling_velocity_mps: float
    particle_row_indices: np.ndarray
    input_hash: str
    # WP-17：回溯模式明確化（見 ``backtrack_horizon.py``）。``resolved_max_backtrack_
    # seconds=None`` 是刻意的向後相容預設值——沿用「不知道回溯模式」的舊呼叫端（例如
    # 尚未更新的測試或工具直接建構 ``NativeWorkUnit``）時，``build_hour_major_domain_
    # batch`` 會退回讀取 ``HourMajorConfig.max_backtrack_seconds``（即 WP-17 之前對
    # 全部單元一律套用同一個 H 的行為），不會因為新增這兩個欄位而改變既有結果。
    # ``pre_window_deposition=True`` 只應該由 ``lbt native-plan`` 在使用者明確要求
    # ``fixed_calendar_window`` 且該到達時刻的沉底年齡 ≥ H 時設定；為 True 時這個
    # 單元完全不會被送進 ``run_flow_domain_hour_major``（見 ``hour_major_provider.
    # _run_unit_entrypoint``），改用 ``backward_driver.build_pre_window_deposition_
    # result`` 直接組出結果。
    resolved_max_backtrack_seconds: float | None = None
    pre_window_deposition: bool = False
    bed_residence_age_hours: int | None = None

    def __post_init__(self) -> None:
        indices = self.particle_row_indices
        if not isinstance(indices, np.ndarray) or indices.ndim != 1:
            raise NativeOrchestratorError("particle_row_indices 必須是一維陣列")
        if indices.dtype != np.int64:
            object.__setattr__(self, "particle_row_indices", indices.astype(np.int64))
            indices = self.particle_row_indices
        if indices.size == 0:
            raise NativeOrchestratorError(f"工作單元 {self.unit_id} 不可為空批次")
        if np.any(np.diff(indices) <= 0):
            raise NativeOrchestratorError(f"工作單元 {self.unit_id} 的列位置必須嚴格遞增、無重複")

    @property
    def particle_count(self) -> int:
        """本單元的粒子數，供排程成本估計與報告使用。"""

        return int(self.particle_row_indices.shape[0])

    @property
    def locality_key(self) -> tuple[str, int]:
        """排程用的較粗鍵值：同一流場區域、同一到達時刻共用同一份 720 小時流場。"""

        return (self.flow_domain_id, int(self.arrival_time_utc_ns))

    def to_spec(self) -> NativeWorkUnitSpec:
        """轉成傳給子程序用的輕量、明確可序列化 spec（見 ``NativeWorkUnitSpec``）。"""

        return NativeWorkUnitSpec(
            unit_id=self.unit_id,
            analysis_region_id=self.analysis_region_id,
            flow_domain_id=self.flow_domain_id,
            study_site_id=self.study_site_id,
            arrival_time_utc_ns=int(self.arrival_time_utc_ns),
            settling_velocity_mps=float(self.settling_velocity_mps),
            particle_row_indices=self.particle_row_indices.copy(),
            input_hash=self.input_hash,
            resolved_max_backtrack_seconds=self.resolved_max_backtrack_seconds,
            pre_window_deposition=self.pre_window_deposition,
            bed_residence_age_hours=self.bed_residence_age_hours,
        )


@dataclass(frozen=True, slots=True)
class NativeWorkUnitSpec:
    """``NativeWorkUnit`` 的純資料版本，明確用於跨程序傳遞（multiprocessing pickling）。

    與 ``NativeWorkUnit`` 分開定義，是為了讓「規劃期使用的完整物件」與「跨程序邊界
    實際會被序列化的資料」在型別上明確區分；目前兩者欄位相同，但規劃期物件未來若
    加入不可序列化的輔助欄位（例如快取的 numpy view），不會意外把它們也序列化過去。
    """

    unit_id: str
    analysis_region_id: str
    flow_domain_id: str
    study_site_id: str
    arrival_time_utc_ns: int
    settling_velocity_mps: float
    particle_row_indices: np.ndarray
    input_hash: str
    resolved_max_backtrack_seconds: float | None = None
    pre_window_deposition: bool = False
    bed_residence_age_hours: int | None = None


def _split_indices_by_columns(
    table: ParticleIdentityTable, indices: np.ndarray, columns: tuple[str, ...]
) -> list[tuple[tuple[str, ...], np.ndarray]]:
    """把給定列位置依一或多個字串欄位的值分組，回傳 ``(欄位值, 列位置)`` 清單。

    邏輯與 ``particle_input._group_by_columns`` 相同（lexsort＋邊界偵測一次到位，
    避免對每個唯一值各自掃描一次），但本函式只對呼叫端已經篩出的子集操作，且只
    支援字串欄位（本模組唯一需要在 ``group_indices_by_settling_velocity_and_arrival_time``
    之後再切分的欄位是 ``analysis_region_id``／``study_site_id``），刻意不重用
    ``particle_input`` 的私有函式，避免 WP-7 依賴另一個工作包內部尚可能調整的
    實作細節。
    """

    if indices.size == 0:
        return []
    values = [getattr(table, name)[indices] for name in columns]
    order = np.lexsort(tuple(reversed(values)))
    sorted_values = [value[order] for value in values]
    boundaries = np.zeros(indices.size, dtype=bool)
    boundaries[0] = True
    for value in sorted_values:
        boundaries[1:] |= value[1:] != value[:-1]
    starts = np.flatnonzero(boundaries)
    stops = np.append(starts[1:], indices.size)
    groups: list[tuple[tuple[str, ...], np.ndarray]] = []
    for start, stop in zip(starts, stops, strict=True):
        key = tuple(str(sorted_values[column_index][start]) for column_index in range(len(columns)))
        group_indices = np.sort(indices[order[start:stop]]).astype(np.int64)
        groups.append((key, group_indices))
    return groups


def _unit_input_hash(table_content_sha256: str, key: tuple[Any, ...], indices: np.ndarray) -> str:
    """建立涵蓋來源身分表內容雜湊、單元鍵值與實際列位置的 SHA-256。"""

    digest = sha256()
    digest.update(table_content_sha256.encode("utf-8"))
    digest.update(json.dumps(key, ensure_ascii=False, sort_keys=False, separators=(",", ":")).encode("utf-8"))
    digest.update(np.ascontiguousarray(indices, dtype=np.int64).tobytes())
    return digest.hexdigest()


def plan_native_work_units(
    table: ParticleIdentityTable,
    *,
    region_to_flow_domain: Mapping[str, str] | None = None,
) -> tuple[NativeWorkUnit, ...]:
    """依 ``(analysis_region_id, arrival_time_utc_ns, settling_velocity_mps,
    study_site_id)`` 把粒子身分表切成可直接驅動的工作單元。

    參數
    ------
    table:
        WP-4 ``build_particle_identity_table`` 的輸出。
    region_to_flow_domain:
        ``{analysis_region_id: flow_domain_id}``；正式呼叫應由呼叫端以
        ``config.resolve_flow_domain_id`` 對每個 ``ProjectConfig.domains`` 建出這個
        對照表再傳入（本模組刻意不 import ``config.py``，見模組文件「已知限制」）。
        省略時退化為 ``flow_domain_id == analysis_region_id``，只適用於合成測試。

    回傳的工作單元依鍵值排序（先 ``analysis_region_id``，再 ``arrival_time_utc_ns``，
    再 ``settling_velocity_mps``，再 ``study_site_id``），確保同一次規劃永遠得到
    確定性的順序，重跑不會因為字典或集合的迭代順序改變而得到不同的 ``unit_id`` 生成
    序列（雖然 ``unit_id`` 本身也是內容決定的，不依賴順序）。
    """

    if len(table) == 0:
        raise NativeOrchestratorError("粒子身分表不可為空")

    region_to_flow_domain = dict(region_to_flow_domain or {})
    order = np.lexsort(
        (
            table.study_site_id,
            table.settling_velocity_mps,
            table.arrival_time_utc_ns,
            table.analysis_region_id,
        )
    )
    sorted_region = table.analysis_region_id[order]
    sorted_arrival = table.arrival_time_utc_ns[order]
    sorted_settling = table.settling_velocity_mps[order]
    sorted_site = table.study_site_id[order]
    n = order.size
    boundaries = np.zeros(n, dtype=bool)
    boundaries[0] = True
    boundaries[1:] |= sorted_region[1:] != sorted_region[:-1]
    boundaries[1:] |= sorted_arrival[1:] != sorted_arrival[:-1]
    boundaries[1:] |= sorted_settling[1:] != sorted_settling[:-1]
    boundaries[1:] |= sorted_site[1:] != sorted_site[:-1]
    starts = np.flatnonzero(boundaries)
    stops = np.append(starts[1:], n)

    units: list[NativeWorkUnit] = []
    for start, stop in zip(starts, stops, strict=True):
        region = str(sorted_region[start])
        arrival = int(sorted_arrival[start])
        settling = float(sorted_settling[start])
        site = str(sorted_site[start])
        indices = np.sort(order[start:stop]).astype(np.int64)
        flow_domain_id = region_to_flow_domain.get(region, region)
        key = (region, arrival, settling, site)
        input_hash = _unit_input_hash(table.content_sha256, key, indices)
        unit_id = f"{region}__{site}__arr{arrival}__sv{_slug_float(settling)}__{input_hash[:12]}"
        units.append(
            NativeWorkUnit(
                unit_id=unit_id,
                analysis_region_id=region,
                flow_domain_id=flow_domain_id,
                study_site_id=site,
                arrival_time_utc_ns=arrival,
                settling_velocity_mps=settling,
                particle_row_indices=indices,
                input_hash=input_hash,
            )
        )
    units.sort(
        key=lambda u: (u.analysis_region_id, u.arrival_time_utc_ns, u.settling_velocity_mps, u.study_site_id)
    )
    _validate_units_cover_table(table, units)
    return tuple(units)


def _validate_units_cover_table(table: ParticleIdentityTable, units: Sequence[NativeWorkUnit]) -> None:
    """重用既有 ``validate_batch_partition`` 確認分批無遺漏、無重疊。"""

    groups = [BatchGroup(key=((("unit_id", u.unit_id),)), indices=u.particle_row_indices) for u in units]
    validate_batch_partition(table, groups)


def resolve_native_work_unit_backtrack_horizons(
    units: Sequence[NativeWorkUnit],
    *,
    backtrack_mode: str,
    requested_horizon_days: float,
    arrivals_by_time_utc_ns: Mapping[int, ArrivalTime] | None = None,
    maximum_age_days: int | None = None,
    sampling_seed: int | None = None,
) -> tuple[NativeWorkUnit, ...]:
    """WP-17：把每個工作單元的有效回溯時間、pre-window 判定寫進 ``NativeWorkUnit``。

    每個工作單元只對應**一個**到達時刻（``arrival_time_utc_ns``，見 ``plan_native_
    work_units`` 的分組鍵值），而沉底年齡是到達時刻的屬性，因此「有效回溯時間」是
    per-arrival（等價於 per-unit），不需要真正的 per-particle 計算——同一個到達
    時刻底下不論多少顆粒子、多少個工作單元（不同沉降速度會拆成不同單元），都共用
    同一個 ``bed_residence.resolve_bed_residence_timing`` 結果，本函式按
    ``arrival_time_utc_ns`` 快取，只呼叫一次。

    ``full_horizon_from_deposition``（預設）刻意**不**把 H 烤進
    ``unit.resolved_max_backtrack_seconds``（維持 ``None``／``-1`` sentinel）：這個
    模式對全部單元套用同一個 H，與 WP-17 之前「完全不知道回溯模式、一律讀
    ``config.boundaries.max_backtrack_days``」的現行行為在數值上等價，因此不需要、
    也不應該由 native-plan 另外決定一個獨立數值——``native-plan --max-backtrack-days``
    歷來只是缺口檢查窗口的估計值（見 ``run_native_plan_command`` 既有用途），與
    ``native-run`` 實際讀取的 ``LBT_NATIVE_CONFIG`` 是不同來源；若在這裡把它烤進
    ``resolved_max_backtrack_seconds``，兩個來源不一致時（例如測試用小型
    ``HourMajorConfig`` 但 native-plan 用預設 30 天）會讓 plan 端的值靜默覆蓋掉
    run-time 設定的真正值。``full_horizon_from_deposition`` 的 ``ResolvedUnitHorizon.
    effective_horizon_seconds`` 仍然算出來、寫進計畫檔／結果摘要供人核對，只是不
    用來覆寫 ``NativeWorkUnit.resolved_max_backtrack_seconds``。

    ``fixed_calendar_window`` 則不同：有效回溯時間本來就是逐到達時刻各自不同
    （``H - age``），沒有「run-time 設定裡已經有的單一值」可以退回，因此必須提供
    ``arrivals_by_time_utc_ns``（每個到達時刻對應的 ``ArrivalTime``，含完整沉底
    metadata）、``maximum_age_days`` 與 ``sampling_seed``，並把解析結果**明確寫入**
    ``unit.resolved_max_backtrack_seconds``／``pre_window_deposition``；缺少對應到達
    時刻的 metadata 時 fail closed（不臆測年齡）。

    回傳新的 ``NativeWorkUnit`` tuple（原物件不可變，這裡一律回傳 ``dataclasses.
    replace`` 產生的新實例），保留原本的排序與其餘欄位。
    """

    cache: dict[int, Any] = {}
    output: list[NativeWorkUnit] = []
    for unit in units:
        arrival_utc = int(unit.arrival_time_utc_ns)
        if arrival_utc not in cache:
            arrival = (
                arrivals_by_time_utc_ns.get(arrival_utc)
                if arrivals_by_time_utc_ns is not None
                else None
            )
            if arrival is None and arrivals_by_time_utc_ns is not None:
                raise NativeOrchestratorError(
                    f"到達時刻 {arrival_utc} 沒有對應的 ArrivalTime metadata，"
                    "無法在 fixed_calendar_window 模式下解析有效回溯時間"
                )
            cache[arrival_utc] = resolve_unit_backtrack_horizon(
                backtrack_mode=backtrack_mode,
                requested_horizon_days=requested_horizon_days,
                arrival=arrival,
                maximum_age_days=maximum_age_days,
                sampling_seed=sampling_seed,
            )
        resolved = cache[arrival_utc]
        unit_resolved_max_backtrack_seconds = (
            resolved.effective_horizon_seconds
            if backtrack_mode == BED_RESIDENCE_MODE_FIXED_CALENDAR_WINDOW
            else None
        )
        output.append(
            _dataclass_replace(
                unit,
                resolved_max_backtrack_seconds=unit_resolved_max_backtrack_seconds,
                pre_window_deposition=resolved.pre_window_deposition,
                bed_residence_age_hours=resolved.bed_residence_age_hours,
            )
        )
    return tuple(output)


_HOUR_MAJOR_DOMAIN_UNIT_PREFIX = "hourmajor::"


def plan_native_hour_major_work_units(
    units: Sequence[NativeWorkUnit],
) -> tuple[tuple[NativeWorkUnit, ...], Mapping[str, tuple[NativeWorkUnit, ...]]]:
    """把 ``plan_native_work_units`` 產生的逐工作單元，依 ``flow_domain_id`` 併成
    小時主軸排程（``hour_major.run_flow_domain_hour_major``）要用的「每個流場區域
    一個」排程單元，供 ``run_native_plan`` 的既有執行引擎直接重用。

    背景
    ----
    ``run_native_plan``／``plan_worker_schedule``／``filter_pending_units`` 這一整套
    執行引擎（見模組文件「執行引擎」一節）完全是靠 ``NativeWorkUnit`` 的欄位在運作
    （worker 指派看 ``particle_count``、續跑判斷看 ``input_hash``、完成證據檔名看
    ``unit_id``），沒有任何一處假設「一個 ``NativeWorkUnit`` 只對應一次
    ``run_backward_batch`` 呼叫」。因此小時主軸排程不需要另外重寫一套執行引擎，只要
    把「同一個流場區域底下全部的逐工作單元」**摺疊成一個新的 ``NativeWorkUnit``**
    （``unit_id`` 帶 ``"hourmajor::"`` 前綴、``flow_domain_id`` 不變、
    ``particle_row_indices`` 是底下全部單元列位置的聯集），其餘四個鍵值
    （``analysis_region_id``／``study_site_id``／``arrival_time_utc_ns``／
    ``settling_velocity_mps``）在小時主軸這個更粗的粒度下沒有單一值可以代表，因此
    填入本函式定義的 sentinel 值（``"__hour_major_domain__"``／``1``／``0.0``，
    呼叫端一律不應該讀取這幾個欄位的值，只應該讀 ``unit_id``／
    ``flow_domain_id``／``particle_row_indices``），把這批摺疊出來的
    ``NativeWorkUnit`` 原封不動交給 ``run_native_plan`` 排程執行，就能重用它既有的
    worker 指派、原子性完成證據、續跑判斷與進度快照，不必重寫。

    真正「摺疊回去的原始逐工作單元是哪些」這個對照關係，``NativeWorkUnit`` 本身
    （凍結 dataclass、欄位固定）放不下，所以另外用回傳值的第二項
    ``members``（``{domain_unit_id: (原始 NativeWorkUnit, ...)}``）回傳；呼叫端的
    ``run_unit(resources, spec)`` 回呼要能在子程序裡看到這個對照表，才能對
    ``spec.unit_id`` 查出這個流場區域底下全部的到達時刻／沉降速度／測站組合，逐一
    建出 ``hour_major.HourMajorSettlingGroupInput``（見 ``hour_major`` 模組文件
    「分批粒度與 NativeWorkUnit 的關係」一節），再呼叫一次
    ``hour_major.build_hour_major_domain_batch`` ＋
    ``hour_major.run_flow_domain_hour_major`` 取得整個流場區域全部單元的結果，最後
    把每個原始單元自己的 ``BackwardDriverResult`` 轉成 ``run_unit`` 該回傳的摘要字典
    （即 ``NativeUnitOutcome``／``*.result.json`` 的內容）。這個轉接（如何把
    ``members`` 這個對照表用可以跨 ``multiprocessing.spawn`` 序列化的方式送進
    worker、如何從 ``ParticleIdentityTable`` 取得小時主軸需要的 ``x0_m``／``y0_m``／
    ``z0_m``／``seed_hex``、如何取得各測站的 ``interest_regions`` 多邊形、如何取得
    ``forcing_resolver``）需要 import 本模組刻意不依賴的重量級模組（見模組文件
    「背景與定位」一節），因此留給 ``build_worker_resources``／``run_unit`` 的實作端
    （即 ``production_wiring.py``）負責，不在本函式的職責範圍內。

    參數與回傳
    ----------
    units:
        ``plan_native_work_units`` 的輸出（或任何欄位相容的 ``NativeWorkUnit``
        序列）。

    回傳 ``(domain_units, members)``：``domain_units`` 依 ``flow_domain_id`` 字串
    排序，每個流場區域恰好一個；``members[domain_unit.unit_id]`` 是該流場區域底下
    的原始工作單元，依 ``unit_id`` 排序（確定性順序，與呼叫時 ``units`` 的原始
    順序無關）。
    """

    if not units:
        raise NativeOrchestratorError("小時主軸排程不可對空的工作單元清單規劃")

    by_domain: dict[str, list[NativeWorkUnit]] = {}
    for unit in units:
        by_domain.setdefault(unit.flow_domain_id, []).append(unit)

    domain_units: list[NativeWorkUnit] = []
    members: dict[str, tuple[NativeWorkUnit, ...]] = {}
    for flow_domain_id in sorted(by_domain):
        member_units = tuple(sorted(by_domain[flow_domain_id], key=lambda u: u.unit_id))
        all_rows = np.unique(np.concatenate([m.particle_row_indices for m in member_units]))
        combined_hash = sha256(
            "|".join(f"{m.unit_id}:{m.input_hash}" for m in member_units).encode("utf-8")
        ).hexdigest()
        domain_unit_id = f"{_HOUR_MAJOR_DOMAIN_UNIT_PREFIX}{flow_domain_id}__{combined_hash[:12]}"
        domain_unit = NativeWorkUnit(
            unit_id=domain_unit_id,
            analysis_region_id="__hour_major_domain__",
            flow_domain_id=flow_domain_id,
            study_site_id="__hour_major_domain__",
            arrival_time_utc_ns=1,
            settling_velocity_mps=0.0,
            particle_row_indices=all_rows,
            input_hash=combined_hash,
        )
        domain_units.append(domain_unit)
        members[domain_unit_id] = member_units

    return tuple(domain_units), members


# ══════════════════════════════════════════════════════════════════════════
# WP-11：切段平行——把「每個流場區域一個」的小時主軸排程單元，依到達時刻
# 進一步切成 K 段，讓核心數遠多於流場區域數的伺服器也能吃滿全部核心。
# ══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class HourMajorSegmentPlanReport:
    """單一流場區域切段規劃的報表：段數、涵蓋範圍與估計的重疊額外成本。

    「額外成本」是指段與段之間因為各自獨立回溯 ``max_backtrack_seconds``
    （預設 30 天）而重複合成底場的小時數，相對於「完全不切段（``K=1``）只
    合成一次」所多付出的比例；只是**分析式估計**（假設每小時固定開銷佔比
    均勻），不是實際量測，見 ``estimate_segment_extra_cost_fraction`` 的說明。
    """

    flow_domain_id: str
    segment_count: int
    min_arrival_utc_ns: int
    max_arrival_utc_ns: int
    arrival_span_days: float
    max_backtrack_days: float
    estimated_extra_cost_fraction: float
    segment_unit_ids: tuple[str, ...]
    segment_arrival_counts: tuple[int, ...]
    segment_hours_to_synthesize: tuple[int, ...] = ()
    """WP-13 新增：每段自己需要合成底場的流場小時數（該段最早到達時刻減去
    ``max_backtrack_days`` 到該段最晚到達時刻的整段小時數），供
    ``estimate_hour_major_segment_costs`` 套用正確成本模型；預設空 tuple
    只是為了不打破舊呼叫端以位置引數建構本 dataclass 的既有測試，
    ``plan_native_hour_major_segments`` 一律會填入與 ``segment_unit_ids``
    等長的值。"""


def estimate_segment_extra_cost_fraction(
    *, arrival_span_days: float, segment_count: int, max_backtrack_days: float
) -> float:
    """估計切成 ``segment_count`` 段後，因回溯窗重疊而多付出的小時計算比例。

    推導：一個流場區域的到達時刻分布在 ``[t_min, t_max]``（跨距
    ``arrival_span_days`` 天）；不切段時，小時主軸排程沿絕對時鐘從
    ``t_max`` 走回 ``t_min - max_backtrack_days``，總共約
    ``arrival_span_days + max_backtrack_days`` 天的「固定開銷小時」（乾濕＋
    Stokes＋CFL 估計，與粒子數無關，見 ``hour_major`` 模組文件 9.1 節）。
    切成 ``K`` 段、每段各自涵蓋約 ``arrival_span_days / K`` 天的到達時刻，
    但各自仍要獨立回溯 ``max_backtrack_days`` 天，因此 ``K`` 段合計的固定
    開銷小時約為 ``K × (arrival_span_days / K + max_backtrack_days)
    = arrival_span_days + K × max_backtrack_days``。與不切段（``K=1``）的
    ``arrival_span_days + max_backtrack_days`` 相比，多出的比例是：

    ``(K - 1) × max_backtrack_days / (arrival_span_days + max_backtrack_days)``

    這是**固定開銷小時數**的額外比例，不是總牆鐘時間的額外比例——後者還要
    看固定開銷在總耗時裡的佔比（見 ``hour_major`` 模組文件 9.7 節「固定
    開銷攤提的相對收益隨粒子數增加而下降」），本函式刻意只估計前者，呼叫端
    在報告中應該連同粒子數規模的脈絡一起呈現，不應該單獨當成總牆鐘時間的
    倍率使用。

    範例：兩年資料（``arrival_span_days≈730``）切成 8 段、``max_backtrack_
    days=30``：``(8-1)×30/(730+30) = 210/760 ≈ 0.276``，與工作包描述的
    「額外成本約 1/3」量級一致（差異來自「跨距」定義用 730 或 760 當分母的
    近似取捨，此函式選擇較保守、分母較小的 ``arrival_span_days +
    max_backtrack_days``）。
    """

    if segment_count < 1:
        raise NativeOrchestratorError("segment_count 必須是正整數")
    if arrival_span_days < 0 or max_backtrack_days < 0:
        raise NativeOrchestratorError("arrival_span_days／max_backtrack_days 不可為負")
    denominator = arrival_span_days + max_backtrack_days
    if denominator <= 0:
        return 0.0
    return (segment_count - 1) * max_backtrack_days / denominator


def recommend_hour_major_segment_counts(
    units: Sequence[NativeWorkUnit],
    *,
    target_worker_count: int,
    max_backtrack_seconds: float = _DEFAULT_MAX_BACKTRACK_SECONDS,
    max_segments_per_domain: int | None = None,
) -> dict[str, int]:
    """依目標平行度自動決定每個流場區域要切成幾段。

    策略：把 ``target_worker_count`` 個「段」名額，依各流場區域的粒子總數
    佔比分配（粒子數愈多的區域分到愈多段，讓每段的計算量盡量平衡），每個
    區域至少分到 1 段；受限於這個區域實際的相異到達時刻數（不能切出比到達
    時刻數更多的段，否則會有空段）。這是**啟發式**分配，不保證嚴格最佳
    makespan——精確排程仍需搭配 ``plan_worker_schedule`` 對切段後的實際
    工作單元重新做 LPT 裝箱。

    ``max_segments_per_domain`` 提供時額外夾住每個區域的段數上限，避免
    ``estimate_segment_extra_cost_fraction`` 估計的重疊額外成本過高（呼叫端
    可以先用該函式估算不同 ``K`` 的額外成本，再據此設定這個上限）。
    """

    if target_worker_count < 1:
        raise NativeOrchestratorError("target_worker_count 必須是正整數")
    if not units:
        raise NativeOrchestratorError("units 不可為空")

    by_domain: dict[str, list[NativeWorkUnit]] = {}
    for unit in units:
        by_domain.setdefault(unit.flow_domain_id, []).append(unit)

    particle_counts = {
        domain: sum(u.particle_count for u in domain_units) for domain, domain_units in by_domain.items()
    }
    distinct_arrival_counts = {
        domain: len({u.arrival_time_utc_ns for u in domain_units})
        for domain, domain_units in by_domain.items()
    }
    total_particles = sum(particle_counts.values()) or 1

    result: dict[str, int] = {}
    for domain in sorted(by_domain):
        share = particle_counts[domain] / total_particles
        raw = max(1, round(target_worker_count * share))
        cap = distinct_arrival_counts[domain]
        if max_segments_per_domain is not None:
            cap = min(cap, max_segments_per_domain)
        result[domain] = max(1, min(raw, cap))
    return result


# ══════════════════════════════════════════════════════════════════════════
# WP-13：正確成本模型與自動選段數
#
# 舊版 ``recommend_hour_major_segment_counts``（上面）以「粒子數」估計每個
# 流場區域的計算成本；但 ``hour_major.py`` 模組文件與本次 WP-13 實測都已
# 證明，小時主軸排程的成本幾乎全部來自「每個流場小時合成一次底場」（乾濕、
# Stokes、Smagorinsky、sigma、轉置），這個成本正比於「需要合成底場的流場
# 小時數 × 節點數 × 垂向層數」，幾乎與粒子數無關（見最終報告 29,929／
# 67,600 節點 × 48 層的量測：每小時成本分別是 10.008 秒／22.553 秒，正規化
# 後幾乎相同，強烈支持這個線性模型）。以下函式提供正確的成本模型與依此
# 模型自動選段數的邏輯，供 ``plan_native_hour_major_segments`` 與
# ``native-plan`` CLI 使用；舊函式保留不動，避免打破既有呼叫端與測試。
# ══════════════════════════════════════════════════════════════════════════

# WP-14 後（每節點只求一次波數＋端點重用）重新校準：29,929／67,600 節點×48 層每小時
# 3.96／9.00 秒，約 2.77e-6（沙箱單執行緒值；正式伺服器應以 --seconds-per-node-layer-hour 覆寫）。
_DEFAULT_SECONDS_PER_NODE_LAYER_HOUR = 2.77e-6
"""WP-13 實測校準常數：每小時、每個 (節點, 層) 組合的底場合成成本（秒）。

來源（見最終報告完整量測表與機器規格誠實揭露）：在 29,929 節點×48 層與
67,600 節點×48 層的合成網格上，量測 ``backward_driver.
compose_current_and_stokes_fields``（提供波浪場、啟用 Stokes）＋
``compute_smagorinsky_kh_field``＋``apply_settling_velocity``＋
``estimate_substep_count`` 的單一流場小時總耗時，分別是 10.008 秒與
22.553 秒；除以各自的節點數×層數（1,436,592 與 3,244,800）得到 6.968e-6
秒與 6.950e-6 秒——兩個相差 2.3 倍規模的網格算出幾乎相同的「每節點層小時」
成本，這是本模組選擇「成本正比於節點數×層數×小時數」這個線性模型、而不是
更複雜模型的實證依據。7.0e-6 秒是這兩個實測值的簡單捨入。

**誠實揭露**：這是在這次 WP-13 量測用的機器（2 核心沙盒環境，非正式伺服器，
見最終報告）上的實測值；正式伺服器的 CPU 時脈、快取、記憶體頻寬都不同，
绝對秒數一定不同（東北區實際節點數比這裡量測的 67,600 略多，見下方
``NORTHEAST_NODE_COUNT`` 一類的常數由呼叫端自行提供，本模組不內建）。
呼叫端應該在正式伺服器上重新量測、以 ``seconds_per_node_layer_hour``
參數覆寫這個預設值；這裡只是提供一個「量級正確、不必每次呼叫前都先跑一次
量測」的合理起點。
"""

_DEFAULT_SECONDS_PER_PARTICLE_SUBSTEP = 5.0e-6
"""既有量測：``run_backward_batch`` 粒子步 kernel 本身的成本量級約
5 µs/步（見 ``native_runner/orchestrator.py`` 模組文件開頭引用的既有效能
量測，及 WP-9 ``hour_major`` 模組文件「背景與問題」一節引用的同一個
數字）。"""

_DEFAULT_SUBSTEPS_PER_HOUR = 4.0
"""粗略估計：一個流場小時內 CFL 子步數（``ndeltp``）的量級。真正的子步數
由每小時實際流速決定（見 ``estimate_substep_count``），逐小時、逐區域都會
變動；本模組不假裝能在排程前精確預知它，這裡只需要一個量級正確的常數，
確保成本模型裡「粒子步」這一項不會被完全設成零、而是如任務書要求「加上
極小的粒子步項」，同時不會喧賓奪主蓋過真正主導成本的節點數×層數×小時數
這一項（見 ``estimate_hour_major_unit_cost_seconds`` 的兩項合計）。"""


def estimate_hour_major_unit_cost_seconds(
    *,
    hours_to_synthesize: int,
    node_count: int,
    layer_count: int,
    particle_count: int,
    seconds_per_node_layer_hour: float = _DEFAULT_SECONDS_PER_NODE_LAYER_HOUR,
    seconds_per_particle_substep: float = _DEFAULT_SECONDS_PER_PARTICLE_SUBSTEP,
    substeps_per_hour: float = _DEFAULT_SUBSTEPS_PER_HOUR,
) -> float:
    """WP-13 正確成本模型：估計一個小時主軸排程單元（整個流場區域或其中一段）
    的計算成本（秒）。

    ``estimated_cost = hours_to_synthesize × node_count × layer_count ×
    seconds_per_node_layer_hour``（底場合成主項，見上方
    ``_DEFAULT_SECONDS_PER_NODE_LAYER_HOUR`` 的量測依據）``+
    hours_to_synthesize × substeps_per_hour × particle_count ×
    seconds_per_particle_substep``（粒子步次項，量級遠小於主項，只在段內
    粒子數差距懸殊時才會影響排序）。

    這是工作書「已由主代理確認」問題描述要求的成本模型（「幾乎與粒子數無關」
    的固定開銷正比於「該段需走過的流場小時數 × 網格節點數 × 層數」），取代
    舊版 ``plan_worker_schedule`` 以粒子數估計成本的近似——後者對逐工作單元
    路徑（每個工作單元一次 kernel 呼叫）勉強成立，但對小時主軸排程（見
    ``hour_major.py``）完全失真：兩個粒子數差 100 倍、但涵蓋相同流場小時數
    的段，實際成本幾乎相同。
    """

    if hours_to_synthesize < 0:
        raise NativeOrchestratorError("hours_to_synthesize 不可為負")
    if node_count < 0 or layer_count < 0 or particle_count < 0:
        raise NativeOrchestratorError("node_count／layer_count／particle_count 不可為負")
    field_synthesis_cost = hours_to_synthesize * node_count * layer_count * seconds_per_node_layer_hour
    particle_step_cost = (
        hours_to_synthesize * substeps_per_hour * particle_count * seconds_per_particle_substep
    )
    return field_synthesis_cost + particle_step_cost


def estimate_hour_major_segment_costs(
    reports: Sequence[HourMajorSegmentPlanReport],
    domain_units: Sequence[NativeWorkUnit],
    *,
    node_counts: Mapping[str, int],
    layer_counts: Mapping[str, int],
    seconds_per_node_layer_hour: float = _DEFAULT_SECONDS_PER_NODE_LAYER_HOUR,
    seconds_per_particle_substep: float = _DEFAULT_SECONDS_PER_PARTICLE_SUBSTEP,
    substeps_per_hour: float = _DEFAULT_SUBSTEPS_PER_HOUR,
) -> dict[str, float]:
    """對 ``plan_native_hour_major_segments`` 的輸出，逐排程單元套用 WP-13
    成本模型，回傳 ``{domain_unit_id: 估計成本秒數}``。

    Args:
        reports: ``plan_native_hour_major_segments`` 回傳的 ``reports``。
        domain_units: 同一次呼叫回傳的 ``domain_units``（用來查每個排程
            單元自己的 ``particle_count``，即成本模型的次要項）。
        node_counts / layer_counts: ``{flow_domain_id: 節點數／層數}``；
            缺少任何一個出現在 ``reports`` 裡的 ``flow_domain_id`` 都會
            fail-closed，不會用臆測值頂替（節點數／層數是決定成本的主要
            因子，猜錯會讓整個排程與報告的預估牆鐘時間失真）。

    Returns:
        ``{domain_unit_id: 估計成本秒數}``，鍵值涵蓋 ``reports`` 內全部
        ``segment_unit_ids``。
    """

    particle_count_by_unit_id = {u.unit_id: u.particle_count for u in domain_units}
    costs: dict[str, float] = {}
    for report in reports:
        if report.flow_domain_id not in node_counts or report.flow_domain_id not in layer_counts:
            raise NativeOrchestratorError(
                f"缺少流場區域 {report.flow_domain_id!r} 的 node_counts／layer_counts，"
                "無法套用 WP-13 成本模型（節點數／層數是成本模型的主要因子，本函式"
                "拒絕以臆測值頂替）"
            )
        node_count = node_counts[report.flow_domain_id]
        layer_count = layer_counts[report.flow_domain_id]
        for unit_id, hours in zip(
            report.segment_unit_ids, report.segment_hours_to_synthesize, strict=True
        ):
            particle_count = particle_count_by_unit_id.get(unit_id, 0)
            costs[unit_id] = estimate_hour_major_unit_cost_seconds(
                hours_to_synthesize=hours,
                node_count=node_count,
                layer_count=layer_count,
                particle_count=particle_count,
                seconds_per_node_layer_hour=seconds_per_node_layer_hour,
                seconds_per_particle_substep=seconds_per_particle_substep,
                substeps_per_hour=substeps_per_hour,
            )
    return costs


def estimate_total_wall_clock_seconds(
    unit_costs: Mapping[str, float], *, worker_count: int
) -> tuple[float, float]:
    """工作書要求的總牆鐘時間估算：``max(總成本／核心數, 最大單段成本)``。

    第一項是「假設完美負載平衡、核心永不閒置」下的理論下界；第二項是工作書
    「不可壓縮的下限」——單一到達時刻的回溯窗長度決定的成本，即使核心再多
    也壓不下去這個數字（因為任何一段都必須從自己的到達時刻依時間順序走到
    回溯上限，不能拆給另一個核心分攤），除非用任務 2 的小時內平行化壓低
    這個最大單段自己的成本。兩者取最大值，是對「總牆鐘時間」最粗略但方向
    正確的估計：實際值一定不會低於這兩個下界中較大的一個，但因為裝箱
    （bin packing）不可能永遠完美平衡，實際值通常會比這個估計更高一些
    （見 ``plan_worker_schedule`` 的 LPT 裝箱有界誤差說明）。

    Returns:
        ``(estimated_wall_clock_seconds, max_single_unit_cost_seconds)``；
        第二個回傳值供報告單獨列出「不可壓縮下限」。
    """

    if worker_count < 1:
        raise NativeOrchestratorError("worker_count 必須是正整數")
    if not unit_costs:
        raise NativeOrchestratorError("unit_costs 不可為空")
    total = sum(unit_costs.values())
    max_cost = max(unit_costs.values())
    return max(total / worker_count, max_cost), max_cost


def choose_hour_major_segment_count(
    *,
    arrival_span_days: float,
    distinct_arrival_count: int,
    max_backtrack_days: float,
    node_count: int,
    layer_count: int,
    available_cores: int,
    seconds_per_node_layer_hour: float = _DEFAULT_SECONDS_PER_NODE_LAYER_HOUR,
    max_segment_count: int | None = None,
) -> tuple[int, dict[int, dict[str, float]]]:
    """WP-13：依單一流場區域的網格規模、到達時刻分布、回溯天數與可用核心數，
    自動選擇這個區域要切成幾段（``K``）。

    對每個候選 ``K``（``1`` 到 ``distinct_arrival_count``，或
    ``max_segment_count`` 進一步夾住的上限），假設 ``K`` 段把全部到達時刻
    跨距平分（與 ``_split_sorted_into_k_chunks`` 的實際切法一致），估計：

    - ``per_segment_hours = arrival_span_hours / K + max_backtrack_hours``
      （每段自己的小時數：平分後的到達時刻跨距，加上每段都要各自付出的
      回溯窗長度）
    - ``total_hours = K × per_segment_hours``（``K`` 段合計的小時數，
      ``K`` 越大，段與段之間回溯窗重疊而重複計算的比例越高，見
      ``estimate_segment_extra_cost_fraction``）
    - ``wall_clock = max(total_hours × 每節點層小時成本 / min(K,
      available_cores), per_segment_hours × 每節點層小時成本)``（與
      ``estimate_total_wall_clock_seconds`` 相同的下界公式，套用到「假設
      這個區域自己能用到 ``available_cores`` 個核心」這個樂觀假設——實際
      上跟其他區域共用同一個 worker pool，見 ``run_native_plan`` 的動態
      佇列設計，因此這個函式選出的 ``K`` 是「合理」而非「全域最佳」，真正
      的整體牆鐘時間仍要用 ``estimate_total_wall_clock_seconds`` 對全部
      區域、全部段的實際成本重新估一次）

    選出讓 ``wall_clock`` 最小的 ``K``；``K`` 太大時 ``total_hours``（進而
    重複計算的固定開銷）會持續增加，但 ``per_segment_hours``／
    ``total_hours/K`` 會持續下降，兩者的極小值通常落在
    ``K ≈ available_cores`` 附近但可能因為回溯窗的固定開銷而更小，這正是
    本函式要用數值搜尋而不是拍腦袋公式的原因。

    Returns:
        ``(建議段數, {K: 該候選 K 的診斷字典（wall_clock_seconds／
        extra_cost_fraction／total_cost_seconds／max_segment_cost_seconds／
        cores_used）})``；診斷字典供 ``native-plan`` 報表列出候選比較，不是
        只回傳最終選擇，方便人工覆寫時參考。
    """

    if distinct_arrival_count < 1:
        raise NativeOrchestratorError("distinct_arrival_count 必須是正整數")
    if available_cores < 1:
        raise NativeOrchestratorError("available_cores 必須是正整數")
    if node_count < 0 or layer_count < 0:
        raise NativeOrchestratorError("node_count／layer_count 不可為負")
    if arrival_span_days < 0 or max_backtrack_days < 0:
        raise NativeOrchestratorError("arrival_span_days／max_backtrack_days 不可為負")

    cap = (
        distinct_arrival_count
        if max_segment_count is None
        else min(distinct_arrival_count, max_segment_count)
    )
    cap = max(1, cap)
    arrival_span_hours = arrival_span_days * 24.0
    max_backtrack_hours = max_backtrack_days * 24.0
    cost_per_hour = node_count * layer_count * seconds_per_node_layer_hour
    no_split_hours = arrival_span_hours + max_backtrack_hours

    diagnostics: dict[int, dict[str, float]] = {}
    best_k = 1
    best_wall_clock: float | None = None
    for k in range(1, cap + 1):
        per_segment_hours = arrival_span_hours / k + max_backtrack_hours
        total_hours = per_segment_hours * k
        extra_fraction = (total_hours - no_split_hours) / no_split_hours if no_split_hours > 0 else 0.0
        cores_used = min(k, available_cores)
        total_cost = total_hours * cost_per_hour
        max_segment_cost = per_segment_hours * cost_per_hour
        wall_clock = max(total_cost / cores_used, max_segment_cost)
        diagnostics[k] = {
            "wall_clock_seconds": wall_clock,
            "extra_cost_fraction": extra_fraction,
            "total_cost_seconds": total_cost,
            "max_segment_cost_seconds": max_segment_cost,
            "cores_used": float(cores_used),
        }
        if best_wall_clock is None or wall_clock < best_wall_clock - 1.0e-9:
            best_wall_clock = wall_clock
            best_k = k
    return best_k, diagnostics


def recommend_hour_major_segment_counts_by_workload(
    units: Sequence[NativeWorkUnit],
    *,
    available_cores: int,
    node_counts: Mapping[str, int],
    layer_counts: Mapping[str, int],
    max_backtrack_seconds: float = _DEFAULT_MAX_BACKTRACK_SECONDS,
    seconds_per_node_layer_hour: float = _DEFAULT_SECONDS_PER_NODE_LAYER_HOUR,
    max_segments_per_domain: int | None = None,
) -> dict[str, int]:
    """WP-13：``recommend_hour_major_segment_counts`` 的正確成本模型版本。

    把 ``units`` 依 ``flow_domain_id`` 分組後，逐區域呼叫
    ``choose_hour_major_segment_count`` 決定段數；與舊版不同，**不**依粒子數
    比例分配段數名額，而是每個區域各自依自己的網格規模、到達時刻跨距與
    回溯天數，獨立選出讓自己的估計牆鐘時間最小的段數（見
    ``choose_hour_major_segment_count`` docstring 對「合理而非全域最佳」的
    說明）。回傳 ``{flow_domain_id: K}``，可直接餵給
    ``plan_native_hour_major_segments`` 的 ``segment_count`` 參數。
    """

    if not units:
        raise NativeOrchestratorError("units 不可為空")
    by_domain: dict[str, list[NativeWorkUnit]] = {}
    for unit in units:
        by_domain.setdefault(unit.flow_domain_id, []).append(unit)
    max_backtrack_days = max_backtrack_seconds / 86400.0

    result: dict[str, int] = {}
    for domain in sorted(by_domain):
        if domain not in node_counts or domain not in layer_counts:
            raise NativeOrchestratorError(f"缺少流場區域 {domain!r} 的 node_counts／layer_counts")
        domain_units = by_domain[domain]
        distinct_arrivals = sorted({int(u.arrival_time_utc_ns) for u in domain_units})
        arrival_span_days = (
            (distinct_arrivals[-1] - distinct_arrivals[0]) / (86400.0 * 1_000_000_000.0)
            if len(distinct_arrivals) > 1
            else 0.0
        )
        k, _diagnostics = choose_hour_major_segment_count(
            arrival_span_days=arrival_span_days,
            distinct_arrival_count=len(distinct_arrivals),
            max_backtrack_days=max_backtrack_days,
            node_count=node_counts[domain],
            layer_count=layer_counts[domain],
            available_cores=available_cores,
            seconds_per_node_layer_hour=seconds_per_node_layer_hour,
            max_segment_count=max_segments_per_domain,
        )
        result[domain] = k
    return result


def plan_native_hour_major_segments(
    units: Sequence[NativeWorkUnit],
    *,
    segment_count: int | Mapping[str, int] | None = None,
    target_worker_count: int | None = None,
    max_backtrack_seconds: float = _DEFAULT_MAX_BACKTRACK_SECONDS,
    node_counts: Mapping[str, int] | None = None,
    layer_counts: Mapping[str, int] | None = None,
    seconds_per_node_layer_hour: float = _DEFAULT_SECONDS_PER_NODE_LAYER_HOUR,
) -> tuple[
    tuple[NativeWorkUnit, ...],
    Mapping[str, tuple[NativeWorkUnit, ...]],
    tuple[HourMajorSegmentPlanReport, ...],
]:
    """``plan_native_hour_major_work_units`` 的切段版本：每個流場區域切成 K 段。

    每段各自摺疊成一個 ``hourmajor::`` 排程單元（沿用同一套 sentinel 欄位
    與 ``members`` 對照表機制，見 ``plan_native_hour_major_work_units``
    docstring），因此可以原封不動交給既有 ``run_native_plan`` 執行引擎，唯一
    差異是同一個流場區域現在對應多個排程單元（``K`` 個），可以分派給不同
    worker 平行執行。

    **正確性保證（切段不得改變任何粒子的結果）**：切段只改變「哪些
    ``NativeWorkUnit`` 被摺進同一次 ``run_flow_domain_hour_major`` 呼叫」，
    不改變任何粒子的起點、seed 或亂數串流；``hour_major`` 模組的計數器型
    亂數與逐單元位元等價驗收（``tests/test_native_hour_major.py``）已經
    證明「同一顆粒子併入不同批次組成跑，結果位元相同」，切段是這個既有
    不變量的直接應用——``K=1``（全部單元併一段，等同
    ``plan_native_hour_major_work_units`` 的行為）與 ``K>1`` 對同一顆粒子
    必須得到位元相同的最終狀態，這是本函式的核心驗收要求，見
    ``tests/test_native_hour_major_provider.py``。

    切段依據：對每個流場區域，把該區域內全部相異 ``arrival_time_utc_ns``
    由舊到新排序後，切成 ``K`` 個盡量等長（相差至多 1 個到達時刻）的連續
    區段；每個 ``NativeWorkUnit`` 依自己的到達時刻歸入對應區段。這保證每段
    的回溯窗聯集是「該段最早到達時刻 − max_backtrack_seconds 至該段最晚
    到達時刻」的一個連續區間，段與段之間只在回溯窗重疊處重複計算，不會有
    「同一到達時刻被拆進兩段」這種需要額外去重邏輯的情況。

    Args:
        units: ``plan_native_work_units`` 的輸出。
        segment_count: 每個流場區域要切的段數；可以是單一整數（套用到全部
            區域）、``{flow_domain_id: K}`` 對照表（逐區域指定），或
            ``None``（改用 ``target_worker_count`` 透過
            ``recommend_hour_major_segment_counts`` 自動決定）。
        target_worker_count: ``segment_count`` 為 ``None`` 時的目標平行度
            （通常是可用核心數）。兩者都是 ``None`` 時等同 ``segment_count=1``
            （不切段，行為與 ``plan_native_hour_major_work_units`` 相同）。
        max_backtrack_seconds: 用於 ``HourMajorSegmentPlanReport`` 的額外
            成本估計與（WP-13）每段成本模型的回溯窗小時數，不影響實際切段
            （實際回溯窗長度仍由 ``hour_major`` 執行時的
            ``HourMajorConfig.max_backtrack_seconds`` 決定）。
        node_counts / layer_counts: WP-13 新增。``{flow_domain_id: 節點數}``／
            ``{flow_domain_id: 垂向層數}``；``segment_count`` 為 ``None`` 且
            ``target_worker_count`` 提供時，若兩者都提供，改用 WP-13 正確
            成本模型（``recommend_hour_major_segment_counts_by_workload``，
            依「流場小時數×節點數×層數」而非粒子數選段數）；若只提供
            ``target_worker_count`` 而缺少 ``node_counts``／``layer_counts``，
            退回舊版 ``recommend_hour_major_segment_counts``（純粒子數啟發式，
            見該函式文件對「成本模型不精確」的既有誠實揭露——WP-13 之前的
            版本不知道底場合成成本幾乎與粒子數無關這件事）。
        seconds_per_node_layer_hour: WP-13 成本模型的校準常數，見
            ``estimate_hour_major_unit_cost_seconds`` 文件；只在使用新成本
            模型（``node_counts``／``layer_counts`` 都有提供）時生效。

    Returns:
        ``(domain_units, members, reports)``；``domain_units``／``members``
        語意與 ``plan_native_hour_major_work_units`` 相同（只是每個流場
        區域可能對應多個排程單元），``reports`` 是逐區域的切段規劃報表，
        依 ``flow_domain_id`` 排序。
    """

    if not units:
        raise NativeOrchestratorError("小時主軸切段排程不可對空的工作單元清單規劃")

    by_domain: dict[str, list[NativeWorkUnit]] = {}
    for unit in units:
        by_domain.setdefault(unit.flow_domain_id, []).append(unit)

    if segment_count is None:
        if target_worker_count is None:
            resolved_counts: dict[str, int] = {domain: 1 for domain in by_domain}
        elif node_counts is not None and layer_counts is not None:
            resolved_counts = recommend_hour_major_segment_counts_by_workload(
                units,
                available_cores=target_worker_count,
                node_counts=node_counts,
                layer_counts=layer_counts,
                max_backtrack_seconds=max_backtrack_seconds,
                seconds_per_node_layer_hour=seconds_per_node_layer_hour,
            )
        else:
            resolved_counts = recommend_hour_major_segment_counts(
                units, target_worker_count=target_worker_count, max_backtrack_seconds=max_backtrack_seconds
            )
    elif isinstance(segment_count, Mapping):
        resolved_counts = {domain: int(segment_count.get(domain, 1)) for domain in by_domain}
    else:
        resolved_counts = {domain: int(segment_count) for domain in by_domain}

    max_backtrack_days = max_backtrack_seconds / 86400.0
    domain_units: list[NativeWorkUnit] = []
    members: dict[str, tuple[NativeWorkUnit, ...]] = {}
    reports: list[HourMajorSegmentPlanReport] = []

    for flow_domain_id in sorted(by_domain):
        domain_member_units = by_domain[flow_domain_id]
        distinct_arrivals = sorted({int(u.arrival_time_utc_ns) for u in domain_member_units})
        k = resolved_counts.get(flow_domain_id, 1)
        if k < 1:
            raise NativeOrchestratorError(f"流場區域 {flow_domain_id!r} 的段數必須是正整數，實際={k}")
        k = min(k, len(distinct_arrivals))  # 不切出比相異到達時刻數更多的段，避免空段

        chunks = _split_sorted_into_k_chunks(distinct_arrivals, k)
        arrival_to_segment: dict[int, int] = {}
        for segment_index, chunk in enumerate(chunks):
            for arrival in chunk:
                arrival_to_segment[arrival] = segment_index

        segment_members: list[list[NativeWorkUnit]] = [[] for _ in chunks]
        for unit in domain_member_units:
            segment_members[arrival_to_segment[int(unit.arrival_time_utc_ns)]].append(unit)

        segment_unit_ids: list[str] = []
        arrival_span_days = (distinct_arrivals[-1] - distinct_arrivals[0]) / (86400.0 * 1_000_000_000.0)
        segment_arrival_counts: list[int] = []
        segment_hours_to_synthesize: list[int] = []
        max_backtrack_hours = int(round(max_backtrack_days * 24.0))
        for segment_index, member_units in enumerate(segment_members):
            member_units_sorted = tuple(sorted(member_units, key=lambda u: u.unit_id))
            all_rows = np.unique(np.concatenate([m.particle_row_indices for m in member_units_sorted]))
            combined_hash = sha256(
                "|".join(f"{m.unit_id}:{m.input_hash}" for m in member_units_sorted).encode("utf-8")
            ).hexdigest()
            domain_unit_id = (
                f"{_HOUR_MAJOR_DOMAIN_UNIT_PREFIX}{flow_domain_id}__seg{segment_index}of{len(chunks)}"
                f"__{combined_hash[:12]}"
            )
            domain_unit = NativeWorkUnit(
                unit_id=domain_unit_id,
                analysis_region_id="__hour_major_domain__",
                flow_domain_id=flow_domain_id,
                study_site_id="__hour_major_domain__",
                arrival_time_utc_ns=1,
                settling_velocity_mps=0.0,
                particle_row_indices=all_rows,
                input_hash=combined_hash,
            )
            domain_units.append(domain_unit)
            members[domain_unit_id] = member_units_sorted
            segment_unit_ids.append(domain_unit_id)
            segment_arrival_counts.append(len(member_units_sorted))
            # WP-13 正確成本模型：這一段自己需要合成底場的流場小時數，是「這一段
            # 涵蓋的到達時刻跨距（小時）」加上「回溯窗長度（小時）」，與段內粒子
            # 數無關——這正是 hour_major.py 固定開銷與粒子數無關這件事的排程層
            # 對應。``chunk`` 是 ``_split_sorted_into_k_chunks`` 產生的這一段到達
            # 時刻（已排序、單位 ns），只有一個到達時刻時跨距為 0。
            chunk = chunks[segment_index]
            segment_span_hours = int(round((chunk[-1] - chunk[0]) / _HOUR_NS)) if len(chunk) > 1 else 0
            segment_hours_to_synthesize.append(segment_span_hours + max_backtrack_hours + 1)

        reports.append(
            HourMajorSegmentPlanReport(
                flow_domain_id=flow_domain_id,
                segment_count=len(chunks),
                min_arrival_utc_ns=distinct_arrivals[0],
                max_arrival_utc_ns=distinct_arrivals[-1],
                arrival_span_days=arrival_span_days,
                max_backtrack_days=max_backtrack_days,
                estimated_extra_cost_fraction=estimate_segment_extra_cost_fraction(
                    arrival_span_days=arrival_span_days,
                    segment_count=len(chunks),
                    max_backtrack_days=max_backtrack_days,
                ),
                segment_unit_ids=tuple(segment_unit_ids),
                segment_arrival_counts=tuple(segment_arrival_counts),
                segment_hours_to_synthesize=tuple(segment_hours_to_synthesize),
            )
        )

    return tuple(domain_units), members, tuple(reports)


def _split_sorted_into_k_chunks(sorted_values: Sequence[int], k: int) -> list[list[int]]:
    """把已排序的序列切成 ``k`` 個盡量等長（相差至多 1）的連續區塊。

    前 ``n % k`` 個區塊多分到 1 個元素，其餘區塊平分，確保 ``k`` 個區塊的
    聯集恰好是原序列、彼此不重疊、且保持原本的排序（每個區塊內部與區塊之間
    都遞增），這是切段要求「每段涵蓋連續到達時刻區間」的直接實作。
    """

    n = len(sorted_values)
    if k < 1 or k > n:
        raise NativeOrchestratorError(f"k 必須介於 1 與序列長度 {n} 之間，實際 k={k}")
    base, extra = divmod(n, k)
    chunks: list[list[int]] = []
    cursor = 0
    for index in range(k):
        size = base + (1 if index < extra else 0)
        chunks.append(list(sorted_values[cursor : cursor + size]))
        cursor += size
    return chunks


@dataclass(frozen=True, slots=True)
class NativeWorkPlanSummary:
    """一次規劃結果的粒子數與單元數分佈摘要，供回報估算使用。"""

    unit_count: int
    locality_group_count: int
    particle_count_total: int
    min_particles_per_unit: int
    max_particles_per_unit: int
    mean_particles_per_unit: float
    region_ids: tuple[str, ...]


def summarize_native_work_plan(units: Sequence[NativeWorkUnit]) -> NativeWorkPlanSummary:
    """統計一組工作單元的規模分佈，不重新掃描來源身分表。"""

    if not units:
        raise NativeOrchestratorError("units 不可為空，無法統計規劃摘要")
    counts = [u.particle_count for u in units]
    localities = {u.locality_key for u in units}
    regions = sorted({u.analysis_region_id for u in units})
    total = sum(counts)
    return NativeWorkPlanSummary(
        unit_count=len(units),
        locality_group_count=len(localities),
        particle_count_total=total,
        min_particles_per_unit=min(counts),
        max_particles_per_unit=max(counts),
        mean_particles_per_unit=total / len(units),
        region_ids=tuple(regions),
    )


def estimate_formal_native_work_plan(
    members_per_scenario: int,
    *,
    study_site_count: int = 5,
    settling_velocity_count: int = 10,
    arrival_time_count_per_site: int = 50,
    receptor_count_per_site: int = 20,
    region_count: int = 4,
) -> NativeWorkPlanSummary:
    """依正式設計常數（見 README「五站四區的沉降情境」）分析式估算工作單元規模。

    這是**設計層級**的估算，不讀取任何真正的身分表，只反映「10 種沉降代理 × 20 個
    受體 × 50 個到達時刻 × 5 站」的規劃契約（``members_per_scenario=M`` 時總粒子數
    為 ``5×10×50×20×M = 50,000×M``，與 README 第 2 節一致）。實際 ``native-plan``
    仍必須對真正建好的 ``ParticleIdentityTable`` 呼叫 ``plan_native_work_units``
    重新統計；若真實到達時刻在站與站之間並非完全獨立（例如 A 區兩站共用同一組到達
    UTC），實際 ``locality_group_count`` 會低於本函式假設「每站到達時刻互不相同」
    所給出的上界，本函式因此把 ``locality_group_count`` 保守地取「未共用到達時刻」
    的上界（``region_count × arrival_time_count_per_site``），並在文件中明確標注
    這一點，避免回報時誤稱為精確值。
    """

    if members_per_scenario < 1:
        raise NativeOrchestratorError("members_per_scenario 必須是正整數")
    unit_count = study_site_count * settling_velocity_count * arrival_time_count_per_site
    particles_per_unit = receptor_count_per_site * members_per_scenario
    total = unit_count * particles_per_unit
    locality_group_count_upper_bound = region_count * arrival_time_count_per_site
    return NativeWorkPlanSummary(
        unit_count=unit_count,
        locality_group_count=locality_group_count_upper_bound,
        particle_count_total=total,
        min_particles_per_unit=particles_per_unit,
        max_particles_per_unit=particles_per_unit,
        mean_particles_per_unit=float(particles_per_unit),
        region_ids=(),
    )


# ══════════════════════════════════════════════════════════════════════════
# 2. 排程：把工作單元分給固定數量的長駐 worker
# ══════════════════════════════════════════════════════════════════════════


def plan_worker_schedule(
    units: Sequence[NativeWorkUnit], *, worker_count: int
) -> dict[int, tuple[str, ...]]:
    """把工作單元分給 ``worker_count`` 個 worker，回傳 ``{worker_index: (unit_id,...)}``。

    排程策略分兩步：

    1. **局部性分組＋負載平衡（LPT 貪婪裝箱）**：先把工作單元依
       ``locality_key``（流場區域＋到達時刻）分組——同一組的所有工作單元共用同一份
       720 小時流場，模組文件已說明理由。以「組內粒子數總和」近似該組計算成本，
       依成本遞減排序後，每次把整組指派給目前累積成本最低的 worker（Longest
       Processing Time first，一種常見且有界的裝箱近似：最終 makespan 不超過最佳解
       的 ``4/3 - 1/(3×worker_count)`` 倍）。這一步保證同一個到達時刻的所有沉降速度
       與站別一定分到同一個 worker，不會被拆散到不同程序、各自重複建構 forcing
       reader。
    2. **同 worker 內依區域、到達時刻排序**：指派完成後，把每個 worker 佇列內的
       單元改依 ``(flow_domain_id, arrival_time_utc_ns, settling_velocity_mps,
       study_site_id)`` 排序執行順序（不影響步驟 1 決定的成本平衡，只改變同一個
       worker 內「先做哪一組」）。到達時刻由舊到新排序，讓相鄰處理的兩個到達時刻
       盡量在時間軸上相鄰，增加 OCM 月份 memmap 頁面與作業系統分頁快取的重疊機率。

    回傳的 dict 只包含至少分到一個單元的 worker index（``0`` 起算），worker 數多於
    局部性群組數時，多出來的 worker 不會出現在回傳值中，呼叫端應據此決定實際要
    啟動幾個子程序，不必為空佇列的 worker 額外開一個什麼都不做的程序。
    """

    if worker_count < 1:
        raise NativeOrchestratorError("worker_count 必須是正整數")
    if not units:
        raise NativeOrchestratorError("units 不可為空")

    by_locality: dict[tuple[str, int], list[NativeWorkUnit]] = {}
    for unit in units:
        by_locality.setdefault(unit.locality_key, []).append(unit)

    locality_items = sorted(
        by_locality.items(),
        key=lambda item: (-sum(u.particle_count for u in item[1]), item[0]),
    )

    worker_cost = [0] * worker_count
    worker_units: list[list[NativeWorkUnit]] = [[] for _ in range(worker_count)]
    for _key, group_units in locality_items:
        target = min(range(worker_count), key=lambda index: (worker_cost[index], index))
        worker_units[target].extend(group_units)
        worker_cost[target] += sum(u.particle_count for u in group_units)

    schedule: dict[int, tuple[str, ...]] = {}
    for worker_index, assigned in enumerate(worker_units):
        if not assigned:
            continue
        assigned.sort(
            key=lambda u: (
                u.flow_domain_id,
                u.arrival_time_utc_ns,
                u.settling_velocity_mps,
                u.study_site_id,
            )
        )
        schedule[worker_index] = tuple(u.unit_id for u in assigned)
    return schedule


# ══════════════════════════════════════════════════════════════════════════
# 3. 記憶體估計（沿用 ``dynamic_execution._memory_available_bytes`` 的讀取思路，
#    但改為可設定門檻，不照搬固定 30 GiB）
# ══════════════════════════════════════════════════════════════════════════


def read_available_memory_bytes() -> int | None:
    """讀取 Linux ``/proc/meminfo`` 的 ``MemAvailable``（位元組）。

    做法與既有 ``dynamic_execution._memory_available_bytes`` 相同（同一個核算式，
    避免不同模組各自對「可用記憶體」有不同定義），但本模組不在這裡內建任何固定
    門檻——門檻改由 ``estimate_worker_count`` 的參數決定，讓正式與測試環境可以依
    實際量測結果分別設定，而不是照搬既有 30 GiB 常數。非 Linux 或讀取失敗時回傳
    ``None``，呼叫端必須自行決定保守或拒絕啟動。
    """

    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if not line.startswith("MemAvailable:"):
                    continue
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    return int(parts[1]) * 1024
    except OSError:
        return None
    return None


def estimate_worker_count(
    *,
    available_memory_bytes: int | None,
    per_worker_memory_bytes: int,
    cpu_count: int | None = None,
    reserved_memory_bytes: int = 0,
    max_workers: int | None = None,
    min_workers: int = 1,
) -> int:
    """依實際量測的可用記憶體與每 worker 預估用量，估算安全的 worker 數。

    參數
    ------
    available_memory_bytes:
        通常是 ``read_available_memory_bytes()`` 的結果；``None``（無法量測）時
        本函式保守回傳 ``min_workers``，不假設有足夠記憶體開更多 worker。
    per_worker_memory_bytes:
        單一 worker 長駐後的預估記憶體用量（BayTrace handles、網格、forcing reader
        快照快取與正在處理批次的粒子陣列），必須由呼叫端依實測結果提供——本模組
        不內建任何物理常數估計值，因為這高度依賴流場區域網格大小與批次粒子數。
    reserved_memory_bytes:
        額外保留給作業系統、其他程序或安全邊際的位元組數，從可用記憶體中先扣除。
    cpu_count / max_workers:
        CPU 核心數上限；``max_workers`` 可再進一步限制（例如伺服器契約要求單執行緒
        限制或保留核心給其他服務）。

    回傳 ``max(min_workers, min(記憶體允許數, CPU 上限, max_workers))``；任何一個
    上限為零或負值都會被夾到 ``min_workers``，不會回傳非正數。
    """

    if per_worker_memory_bytes <= 0:
        raise NativeOrchestratorError("per_worker_memory_bytes 必須是正值")
    if min_workers < 1:
        raise NativeOrchestratorError("min_workers 必須是正整數")

    if available_memory_bytes is None:
        return min_workers

    usable = max(int(available_memory_bytes) - int(reserved_memory_bytes), 0)
    by_memory = max(usable // per_worker_memory_bytes, 0)

    upper_bounds = [value for value in (cpu_count, max_workers) if value is not None]
    upper = min(upper_bounds) if upper_bounds else by_memory
    candidate = min(by_memory, upper) if upper_bounds else by_memory
    return max(min_workers, candidate)


# ══════════════════════════════════════════════════════════════════════════
# 4. 缺口安全的到達時刻：以 ``inventory_forcing_gaps`` 為前置檢查，
#    需要重新選點時重用既有 ``arrival_times.select_arrival_times``
# ══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class ArrivalGapCheck:
    """單一（站點、到達時刻）在其 30 天回溯窗內的缺口安全性檢查結果。"""

    study_site_id: str
    analysis_region_id: str
    arrival_time_utc_ns: int
    backward_window_start_utc_ns: int
    is_gap_safe: bool
    overlapping_gap_ids: tuple[str, ...]
    reason: str


def _region_gap_hours(gap_report: ForcingGapInventoryReport) -> np.ndarray:
    """把一份缺口盤點報告的全部未登錄缺口小時攤平成排序後的一維陣列，供二分搜尋。"""

    if not gap_report.unregistered_gaps:
        return np.empty(0, dtype=np.int64)
    return np.sort(
        np.concatenate(
            [np.asarray(gap.missing_times_ns, dtype=np.int64) for gap in gap_report.unregistered_gaps]
        )
    )


def check_arrival_gap_safety(
    table: ParticleIdentityTable,
    *,
    gap_reports_by_region: Mapping[str, ForcingGapInventoryReport],
    max_backtrack_seconds: float = _DEFAULT_MAX_BACKTRACK_SECONDS,
) -> tuple[ArrivalGapCheck, ...]:
    """對身分表內每個唯一 (站點, 到達時刻) 檢查其 30 天回溯窗是否落在未登錄缺口內。

    這是 native-run 開跑前的第二道防線：``NativeForcingBridgeReader`` 本身在真正
    查到缺口時會拋出 ``UnregisteredForcingGapError`` 讓那個工作單元 fail-closed
    停止（見 ``forcing_bridge`` 模組文件），但那個時間點已經花掉部分計算資源，也
    要等到逆向追蹤真正推進到缺口時刻才會發現。本函式改用
    ``inventory_forcing_gaps``（只讀時間軸、秒級完成，見其 docstring）在**開始
    任何 kernel 計算之前**先算出「這個到達時刻的完整回溯窗是否有任何一小時落在
    未登錄缺口內」，讓 ``native-plan`` 可以在報告裡明確列出受影響的到達時刻，而
    不必真的跑一次才知道。

    ``gap_reports_by_region`` 必須至少涵蓋身分表內出現的每一個
    ``analysis_region_id``；缺少對應報告的區域會讓該區域全部到達時刻回報
    ``is_gap_safe=False``、``reason="no_gap_report_for_region"``，而不是靜默略過。
    """

    max_backtrack_ns = int(round(max_backtrack_seconds * 1.0e9))
    seen: set[tuple[str, int]] = set()
    checks: list[ArrivalGapCheck] = []
    n = len(table)
    for row in range(n):
        site = str(table.study_site_id[row])
        region = str(table.analysis_region_id[row])
        arrival = int(table.arrival_time_utc_ns[row])
        key = (site, arrival)
        if key in seen:
            continue
        seen.add(key)
        window_start = arrival - max_backtrack_ns
        report = gap_reports_by_region.get(region)
        if report is None:
            checks.append(
                ArrivalGapCheck(
                    study_site_id=site,
                    analysis_region_id=region,
                    arrival_time_utc_ns=arrival,
                    backward_window_start_utc_ns=window_start,
                    is_gap_safe=False,
                    overlapping_gap_ids=(),
                    reason="no_gap_report_for_region",
                )
            )
            continue
        overlapping = tuple(
            gap.gap_id
            for gap in report.unregistered_gaps
            if int(gap.missing_times_ns[-1]) >= window_start and int(gap.missing_times_ns[0]) <= arrival
        )
        window_covered = (
            window_start >= report.expected_start_utc_ns and arrival <= report.expected_end_utc_ns
        )
        if not window_covered:
            checks.append(
                ArrivalGapCheck(
                    study_site_id=site,
                    analysis_region_id=region,
                    arrival_time_utc_ns=arrival,
                    backward_window_start_utc_ns=window_start,
                    is_gap_safe=False,
                    overlapping_gap_ids=overlapping,
                    reason="backward_window_exceeds_inventoried_range",
                )
            )
            continue
        checks.append(
            ArrivalGapCheck(
                study_site_id=site,
                analysis_region_id=region,
                arrival_time_utc_ns=arrival,
                backward_window_start_utc_ns=window_start,
                is_gap_safe=len(overlapping) == 0,
                overlapping_gap_ids=overlapping,
                reason="ok" if len(overlapping) == 0 else "backward_window_overlaps_unregistered_gap",
            )
        )
    checks.sort(key=lambda c: (c.analysis_region_id, c.study_site_id, c.arrival_time_utc_ns))
    return tuple(checks)


@dataclass(frozen=True, slots=True)
class ArrivalReselectionInputs:
    """重新呼叫既有 ``select_arrival_times`` 所需、依站保存的原始候選序列。

    這些欄位對應 ``select_arrival_times`` 除了 ``backward_window_available`` 以外
    的全部輸入；本模組**不會**自己決定季節、潮況或事件分層規則，重新選點時完整
    重用既有函式，只由本模組依最新缺口盤點重新算出 ``backward_window_available``
    （見 ``derive_backward_window_available``）。
    """

    time_utc_ns: np.ndarray
    elevation_m: np.ndarray
    significant_wave_height_m: np.ndarray
    current_speed_mps: np.ndarray
    valid_forcing: np.ndarray
    design_version: str
    rolling_window_hours: float = 24.0 * 14.0
    observation_years: tuple[int, ...] | None = None
    replicates: int | None = None
    selection_policy: str | None = None
    policy: str | None = None
    event_supplement_count: int = 2


def derive_backward_window_available(
    time_utc_ns: np.ndarray,
    gap_report: ForcingGapInventoryReport,
    *,
    max_backtrack_seconds: float = _DEFAULT_MAX_BACKTRACK_SECONDS,
) -> np.ndarray:
    """依缺口盤點報告，逐候選時刻算出其完整回溯窗是否完全避開未登錄缺口。

    這是餵給 ``select_arrival_times`` 的 ``backward_window_available`` 遮罩；本函式
    只依據 ``inventory_forcing_gaps`` 的輸出計算，不重新讀取任何大型 forcing 陣列，
    也不涉入選點規則本身（季節、潮況分層完全由 ``select_arrival_times`` 決定）。
    """

    times = np.asarray(time_utc_ns, dtype=np.int64)
    max_backtrack_ns = int(round(max_backtrack_seconds * 1.0e9))
    window_start = times - max_backtrack_ns
    covered = (window_start >= gap_report.expected_start_utc_ns) & (times <= gap_report.expected_end_utc_ns)
    gap_hours = _region_gap_hours(gap_report)
    if gap_hours.size == 0:
        return covered
    # 對每個候選時刻，檢查是否存在任一缺口小時落在 [window_start, time] 內：
    # 用 searchsorted 找出「第一個 >= window_start 的缺口小時」位置，若該位置存在
    # 且其值 <= time，就代表視窗內至少有一個缺口小時。
    overlap = np.zeros(times.shape, dtype=bool)
    for i in range(times.size):
        position = int(np.searchsorted(gap_hours, window_start[i], side="left"))
        overlap[i] = position < gap_hours.size and int(gap_hours[position]) <= int(times[i])
    return covered & ~overlap


def rebuild_gap_safe_arrival_times(
    study_site_id: str,
    inputs: ArrivalReselectionInputs,
    *,
    gap_report: ForcingGapInventoryReport,
    max_backtrack_seconds: float = _DEFAULT_MAX_BACKTRACK_SECONDS,
) -> tuple[ArrivalTime, ...]:
    """用最新缺口盤點重新推導 ``backward_window_available``，重新呼叫既有
    ``arrival_times.select_arrival_times`` 完整重選 50 個到達時刻。

    這個函式**完全重用**既有的 48+2 分層選點邏輯（見 ``arrival_times`` 模組
    docstring），本模組唯一新增的邏輯是「怎麼把缺口盤點結果轉成
    backward_window_available 遮罩」；選點規則、tie-break 與地層定義都不在這裡
    重新實作。若原始候選序列不足 50 個可用時刻，``select_arrival_times`` 會照既有
    行為直接拋出例外，本函式不吞掉、不改寫這個失敗。
    """

    backward_window_available = derive_backward_window_available(
        inputs.time_utc_ns, gap_report, max_backtrack_seconds=max_backtrack_seconds
    )
    return tuple(
        select_arrival_times(
            study_site_id=study_site_id,
            time_utc_ns=inputs.time_utc_ns,
            elevation_m=inputs.elevation_m,
            significant_wave_height_m=inputs.significant_wave_height_m,
            current_speed_mps=inputs.current_speed_mps,
            valid_forcing=inputs.valid_forcing,
            backward_window_available=backward_window_available,
            design_version=inputs.design_version,
            rolling_window_hours=inputs.rolling_window_hours,
            observation_years=inputs.observation_years,
            replicates=inputs.replicates,
            selection_policy=inputs.selection_policy,
            policy=inputs.policy,
            event_supplement_count=inputs.event_supplement_count,
        )
    )


@dataclass(frozen=True, slots=True)
class StratumCoverageDiff:
    """比較替換前後 season×tide_class 分層涵蓋範圍的差異報告。

    只依 ``ArrivalTime.season``／``tide_class`` 這兩個既有欄位粗略分組，**不是**
    ``arrival_times.py`` 內部完整的 48+2（年份×季節×spring/neap×三相位）分層定義；
    本模組刻意不重建那個完整定義，避免用不完整的重建結果冒充官方分層驗收。這裡
    只用來偵測「替換後是否有分層完全消失」這種明顯劣化，任何缺口都必須明確回報，
    不能靜默視為可接受。
    """

    lost_strata: tuple[tuple[str, str], ...]
    before_counts: Mapping[tuple[str, str], int]
    after_counts: Mapping[tuple[str, str], int]

    @property
    def is_degraded(self) -> bool:
        return len(self.lost_strata) > 0


def summarize_stratum_coverage(arrivals: Sequence[ArrivalTime]) -> dict[tuple[str, str], int]:
    """統計一組到達時刻在 (season, tide_class) 上的粗略分層計數。"""

    counts: dict[tuple[str, str], int] = {}
    for item in arrivals:
        key = (item.season, item.tide_class)
        counts[key] = counts.get(key, 0) + 1
    return counts


def diff_stratum_coverage(
    before: Sequence[ArrivalTime], after: Sequence[ArrivalTime]
) -> StratumCoverageDiff:
    """比較替換前後的粗略分層計數，回報是否有分層從「有」變成「無」。"""

    before_counts = summarize_stratum_coverage(before)
    after_counts = summarize_stratum_coverage(after)
    lost = tuple(sorted(key for key in before_counts if after_counts.get(key, 0) == 0))
    return StratumCoverageDiff(lost_strata=lost, before_counts=before_counts, after_counts=after_counts)


# ══════════════════════════════════════════════════════════════════════════
# 5. 執行 manifest：Git commit、dirty flag、設定雜湊、seed 政策、BayTrace 來源雜湊
# ══════════════════════════════════════════════════════════════════════════


def build_native_run_manifest(
    *,
    project_root: str | Path,
    declared_git_commit: str | None = None,
    formal: bool = False,
    config_sha256: str,
    seed_policy: str,
    master_seed: int,
    baytrace_source_root: str | None = None,
    baytrace_dependency_sha256: Mapping[str, str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """組合一次正式 native 執行需要保存的最小 manifest 內容。

    ``config_sha256`` 由呼叫端對實際使用的設定檔內容計算（本模組不假設設定檔的
    格式，只要求呼叫端提供已算好的雜湊，避免本模組被迫理解 ``ProjectConfig`` 的
    完整結構）。``baytrace_dependency_sha256`` 建議直接傳入
    ``BayTraceModuleHandles.dependency_sha256``，把「這次 run 實際載入的 BayTrace
    原始碼版本」與 LBT 自己的 Git commit 分開保存，兩者缺一都無法回答「這個結果
    是用哪一份程式算出來的」。
    """

    provenance: CodeProvenance = collect_code_provenance(
        project_root, declared_git_commit, formal=formal
    )
    manifest: dict[str, Any] = {
        "manifest_kind": "native_backward_batch_run_manifest_v1",
        "code_provenance": asdict(provenance),
        "config_sha256": config_sha256,
        "seed_policy": seed_policy,
        "master_seed": int(master_seed),
        "baytrace_source_root": str(baytrace_source_root) if baytrace_source_root else None,
        "baytrace_dependency_sha256": dict(baytrace_dependency_sha256 or {}),
    }
    if extra:
        manifest["extra"] = dict(extra)
    return manifest


# ══════════════════════════════════════════════════════════════════════════
# 6. 執行引擎：長駐 worker、原子性完成證據、進度回報、失敗隔離
# ══════════════════════════════════════════════════════════════════════════

WorkerResourceBuilder = Callable[[int], Any]
"""``(worker_id) -> resources``；每個 worker 子程序啟動時只呼叫一次。"""

UnitRunner = Callable[[Any, NativeWorkUnitSpec], Mapping[str, Any]]
"""``(resources, unit) -> summary``；``summary`` 必須是可以 ``json.dumps`` 的字典。"""


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """先寫暫存檔、再以 ``os.replace`` 改名，避免續跑時讀到半寫的完成證據。"""

    tmp_path = path.with_name(f"{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp_path, path)


def _marker_path(output_dir: Path, unit_id: str) -> Path:
    return output_dir / f"{unit_id}.done.json"


def _result_path(output_dir: Path, unit_id: str) -> Path:
    return output_dir / f"{unit_id}.result.json"


@dataclass(frozen=True, slots=True)
class NativeUnitOutcome:
    """單一工作單元的執行結果，是完成證據（``*.done.json``）的內容來源。"""

    unit_id: str
    status: str  # "completed" | "failed" | "skipped"
    input_hash: str
    worker_id: int
    elapsed_seconds: float
    particle_count: int
    completed_at_utc_ns: int
    error: str | None = None


def _load_existing_marker(output_dir: Path, unit_id: str) -> Mapping[str, Any] | None:
    marker = _marker_path(output_dir, unit_id)
    if not marker.is_file():
        return None
    try:
        return json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # 半寫或損毀的舊完成證據視為不存在，交由呼叫端重算，不假裝已完成。
        return None


def filter_pending_units(
    units: Sequence[NativeWorkUnit], output_dir: str | Path
) -> tuple[tuple[NativeWorkUnit, ...], tuple[NativeUnitOutcome, ...]]:
    """依既有完成證據把工作單元分成「待執行」與「已跳過（續跑命中）」兩組。

    只有完成證據的 ``input_hash`` 與目前規劃出的單元完全相同才視為可跳過；任何
    輸入變動（重新建表、換了不同的粒子子集或缺口盤點結果）都會讓舊證據失效，
    該單元重新排入待執行清單，不會被誤判為「已完成」。
    """

    output_path = Path(output_dir)
    pending: list[NativeWorkUnit] = []
    skipped: list[NativeUnitOutcome] = []
    for unit in units:
        marker = _load_existing_marker(output_path, unit.unit_id)
        is_completed_match = (
            marker is not None
            and marker.get("input_hash") == unit.input_hash
            and marker.get("status") == "completed"
        )
        if is_completed_match:
            skipped.append(
                NativeUnitOutcome(
                    unit_id=unit.unit_id,
                    status="skipped",
                    input_hash=unit.input_hash,
                    worker_id=int(marker.get("worker_id", -1)),
                    elapsed_seconds=0.0,
                    particle_count=unit.particle_count,
                    completed_at_utc_ns=int(marker.get("completed_at_utc_ns", 0)),
                )
            )
        else:
            pending.append(unit)
    return tuple(pending), tuple(skipped)


@dataclass(frozen=True, slots=True)
class WorkerRunReport:
    """單一 worker 程序（依 ``os.getpid()`` 分組）處理過的工作單元回報。

    WP-13 改成動態拉取佇列（見 ``run_native_plan`` 模組文件「執行引擎」一節）
    之後，「worker」不再對應一份事先排定的整串清單，而是「同一個長駐子
    程序，在其存活期間依序拉取並處理過的工作單元」；本 dataclass 的欄位
    語意因此從「這個 worker 被指派了什麼」變成「這個 worker 實際做了什麼」，
    在事後（全部單元執行完）依 ``NativeUnitOutcome.worker_id`` 分組重建，
    純粹為了保留既有回報結構，不影響任何呼叫端讀取這個欄位的既有程式碼。
    """

    worker_id: int
    outcomes: tuple[NativeUnitOutcome, ...]
    fatal_error: str | None = None


# 每個長駐子程序自己的資源快取（見 ``_run_single_unit_task`` docstring）。
# ``multiprocessing`` 的 ``spawn`` 啟動方式讓每個子程序有獨立的模組狀態，
# 這個模組層級變數天生就是「每個程序各自一份」，不需要額外用鎖保護。
_WORKER_RESOURCE_CACHE: list[tuple[str, Any]] = []


def _reset_worker_resource_cache_for_tests() -> None:
    """測試專用：在同一個程序內重跑多次 ``run_native_plan``（例如
    ``mp_context_method`` 用 fork 或直接呼叫本模組內部函式做單元測試）時，
    清掉上一次呼叫殘留的資源快取，避免不同次呼叫的 ``build_worker_resources``
    互相污染。正式路徑（``spawn``）每次呼叫都會產生全新的子程序，不需要
    呼叫這個函式；只有測試需要在同一個直譯器程序內模擬「新的一批 worker」
    時才使用。
    """

    _WORKER_RESOURCE_CACHE.clear()


def _run_single_unit_task(
    spec: NativeWorkUnitSpec,
    build_worker_resources: WorkerResourceBuilder,
    run_unit: UnitRunner,
    output_dir: str,
    numba_cache_dir: str | None,
) -> NativeUnitOutcome:
    """動態佇列的執行單元：處理**一個**工作單元，供 ``ProcessPoolExecutor``
    的內部工作佇列逐一派給任一目前空閒的長駐子程序。

    這是 WP-13 取代舊版「每個 worker 一次 submit 整串清單」（見模組文件
    「已知限制」曾經記載、現已修正的靜態排程問題）的核心：``run_native_plan``
    對**每一個**待執行工作單元各自呼叫一次 ``pool.submit(_run_single_unit_task,
    ...)``，而不是先用 ``plan_worker_schedule`` 把清單靜態切給固定的
    worker。``ProcessPoolExecutor`` 內建的呼叫佇列本來就是「任何一個子程序
    處理完手上的任務，立刻去佇列拿下一個」的行為（這正是 CPython
    ``concurrent.futures.process`` 的既有實作，不需要另外實作一套佇列協定）
    ——這就是「常駐 worker 從共享佇列拉取下一個工作單元，做完立刻拿下一個」
    的直接體現，消除了舊版「部分 worker 做完自己的整串清單就退出、不能接手
    其他 worker 剩餘工作」的尾端閒置問題。

    每個子程序只在**第一次**被呼叫時建構資源（``build_worker_resources``），
    之後同一個子程序處理的每一個工作單元都重用同一份快取（見
    ``_WORKER_RESOURCE_CACHE``），與舊版「每個 worker 只呼叫一次
    ``build_worker_resources``」的資源重用行為完全相同，只是觸發時機從
    「程序啟動時」改成「程序第一次真正拿到工作單元時」（差異不影響任何
    正確性或效能特性：worker 啟動後在拿到第一個單元之前本來就沒有其他
    事可做）。

    失敗隔離語意與舊版相同：``build_worker_resources`` 失敗時，這個子程序
    往後收到的**每一個**工作單元都會立即回報失敗（快取住失敗原因，不會
    對同一個注定失敗的子程序重複嘗試重建資源），但完全不影響其他子程序
    ——因為現在是動態拉取，這個「毒化」的子程序仍然會繼續從佇列拉工作單元
    （只是每個都立刻失敗），不像舊版整批工作單元從一開始就固定綁死在它
    身上；淨效果是這個子程序等同於少了一個能真正做事的 worker，但不會
    卡住整體排程或讓其他子程序等待它。單一工作單元執行失敗（``run_unit``
    拋例外）則只讓那一個單元失敗，同一個子程序的資源快取不受影響，繼續
    處理佇列裡的下一個單元——與舊版逐單元失敗隔離語意相同。

    若子程序本身異常終止（作業系統層級的 kill、segfault，不是這裡任何
    ``except`` 能攔截的 Python 例外），``ProcessPoolExecutor`` 會把整個
    pool 標記成 ``BrokenProcessPool``；呼叫端 ``run_native_plan`` 對這個
    情況的處理見該函式文件「已知限制」一節，本函式不處理、也無法處理這個
    情況（此時本函式所在的程序已經不存在了）。
    """

    output_path = Path(output_dir)
    if numba_cache_dir:
        os.environ["NUMBA_CACHE_DIR"] = numba_cache_dir
    worker_id = os.getpid()

    if not _WORKER_RESOURCE_CACHE:
        try:
            _WORKER_RESOURCE_CACHE.append(("ok", build_worker_resources(worker_id)))
        except Exception:  # noqa: BLE001 - 資源建構失敗必須讓這個子程序往後的單元都回報失敗，不能讓程序直接崩潰
            _WORKER_RESOURCE_CACHE.append(("failed", traceback.format_exc()))

    cache_status, cache_payload = _WORKER_RESOURCE_CACHE[0]
    if cache_status == "failed":
        outcome = NativeUnitOutcome(
            unit_id=spec.unit_id,
            status="failed",
            input_hash=spec.input_hash,
            worker_id=worker_id,
            elapsed_seconds=0.0,
            particle_count=int(spec.particle_row_indices.shape[0]),
            completed_at_utc_ns=time.time_ns(),
            error=cache_payload,
        )
        _persist_outcome(output_path, outcome, extra_summary=None)
        return outcome

    resources = cache_payload
    start = time.monotonic()
    try:
        summary = run_unit(resources, spec)
        elapsed = time.monotonic() - start
        outcome = NativeUnitOutcome(
            unit_id=spec.unit_id,
            status="completed",
            input_hash=spec.input_hash,
            worker_id=worker_id,
            elapsed_seconds=elapsed,
            particle_count=int(spec.particle_row_indices.shape[0]),
            completed_at_utc_ns=time.time_ns(),
        )
        _persist_outcome(output_path, outcome, extra_summary=summary)
    except Exception:  # noqa: BLE001 - 單一單元失敗必須被隔離，不能拖垮同一子程序後續要處理的其他單元
        elapsed = time.monotonic() - start
        outcome = NativeUnitOutcome(
            unit_id=spec.unit_id,
            status="failed",
            input_hash=spec.input_hash,
            worker_id=worker_id,
            elapsed_seconds=elapsed,
            particle_count=int(spec.particle_row_indices.shape[0]),
            completed_at_utc_ns=time.time_ns(),
            error=traceback.format_exc(),
        )
        _persist_outcome(output_path, outcome, extra_summary=None)
    return outcome


def _persist_outcome(
    output_dir: Path, outcome: NativeUnitOutcome, *, extra_summary: Mapping[str, Any] | None
) -> None:
    """原子性寫出單一工作單元的結果摘要與完成標記。"""

    if extra_summary is not None:
        _atomic_write_json(_result_path(output_dir, outcome.unit_id), dict(extra_summary))
    marker_payload = {
        "unit_id": outcome.unit_id,
        "status": outcome.status,
        "input_hash": outcome.input_hash,
        "worker_id": outcome.worker_id,
        "elapsed_seconds": outcome.elapsed_seconds,
        "particle_count": outcome.particle_count,
        "completed_at_utc_ns": outcome.completed_at_utc_ns,
        "error": outcome.error,
    }
    _atomic_write_json(_marker_path(output_dir, outcome.unit_id), marker_payload)


@dataclass(frozen=True, slots=True)
class NativeRunSummary:
    """一次 ``run_native_plan`` 呼叫的最終彙整結果。"""

    total_units: int
    completed_unit_count: int
    failed_unit_count: int
    skipped_unit_count: int
    failed_unit_ids: tuple[str, ...]
    worker_reports: tuple[WorkerRunReport, ...]
    elapsed_seconds: float

    @property
    def all_succeeded(self) -> bool:
        """完成加跳過等於全部單元、且沒有任何失敗，才代表這次呼叫整批成功。"""

        finished = self.completed_unit_count + self.skipped_unit_count
        return self.failed_unit_count == 0 and finished == self.total_units


def _write_progress_snapshot(
    progress_path: Path,
    *,
    total_units: int,
    skipped: Sequence[NativeUnitOutcome],
    live_counts: dict[str, int],
    live_elapsed_seconds: list[float],
    started_at_utc_ns: int,
) -> None:
    """把目前已知的進度（完成、失敗、跳過、估計剩餘時間）原子性寫成快照檔。

    剩餘時間估計只用「目前為止已完成單元的平均耗時 × 尚未完成單元數」這個粗略
    外插，模組文件已說明各單元實際耗時可能因終止原因不同而有落差；這裡明確只
    承諾「粗估」，不宣稱精確排程。
    """

    completed = live_counts.get("completed", 0) + len(skipped)
    failed = live_counts.get("failed", 0)
    pending = max(total_units - completed - failed, 0)
    mean_elapsed = (sum(live_elapsed_seconds) / len(live_elapsed_seconds)) if live_elapsed_seconds else None
    estimated_remaining_seconds = (mean_elapsed * pending) if mean_elapsed is not None else None
    payload = {
        "total_units": total_units,
        "completed_units": completed,
        "failed_units": failed,
        "skipped_units": len(skipped),
        "pending_units": pending,
        "mean_unit_elapsed_seconds": mean_elapsed,
        "estimated_remaining_seconds": estimated_remaining_seconds,
        "started_at_utc_ns": started_at_utc_ns,
        "updated_at_utc_ns": time.time_ns(),
    }
    _atomic_write_json(progress_path, payload)


def read_progress_snapshot(
    output_dir: str | Path, *, progress_filename: str = "progress.json"
) -> Mapping[str, Any] | None:
    """讀取 ``run_native_plan`` 定期寫出的進度快照；供 ``lbt native-status`` 使用。"""

    path = Path(output_dir) / progress_filename
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _domain_staged_marker_path(staged_ready_dir: Path, flow_domain_id: str, suffix: str) -> Path:
    return staged_ready_dir / f"{flow_domain_id}{suffix}"


def run_native_plan(
    units: Sequence[NativeWorkUnit],
    *,
    worker_count: int,
    build_worker_resources: WorkerResourceBuilder,
    run_unit: UnitRunner,
    output_dir: str | Path,
    numba_cache_dir: str | None = None,
    mp_context_method: str = "spawn",
    poll_interval_seconds: float = 0.2,
    progress_filename: str = "progress.json",
    unit_costs: Mapping[str, float] | None = None,
    staged_ready_dir: str | Path | None = None,
    staged_flow_domain_ids: Sequence[str] | None = None,
    staged_marker_suffix: str = ".done",
) -> NativeRunSummary:
    """規劃、以動態拉取佇列平行執行一批 native 工作單元，支援續跑與失敗隔離。

    WP-13 動態工作佇列
    --------------------
    取代 WP-7 原本「``plan_worker_schedule`` 靜態 LPT 裝箱、每個 worker 一次
    ``submit`` 整串清單」的排程方式（見該函式文件與模組文件「已知限制」曾經
    記載的問題：部分 worker 做完自己的清單就退出、不能接手其他 worker 剩餘
    的工作，造成尾端 CPU 閒置）。本函式改成對**每一個**待執行工作單元各自
    ``pool.submit(_run_single_unit_task, ...)``（見該函式文件），讓
    ``ProcessPoolExecutor`` 內建的呼叫佇列自然達成「常駐子程序做完手上的
    單元立刻拿下一個，佇列清空才沒有下一個可拿」的動態拉取語意——不需要
    另外實作一套佇列協定（比較 ``dynamic_execution.py`` 的 subprocess
    JSON-line 協定：本模組的排程單元本身已經是「一整個流場區域或其中一段」
    這麼粗的粒度，用 ``ProcessPoolExecutor`` 的內建佇列已經足夠穩健，不需要
    重新發明一套機制）。

    提交順序依 ``unit_costs`` 由大到小排序（見下方 ``unit_costs`` 參數）：
    讓成本最高的工作單元最早開始執行，是動態拉取佇列搭配「工作單元數量
    多於 worker 數」時常見的負載平衡手法（避免最後才輪到一個超大工作單元、
    此時其他 worker 已經閒置卻無事可做的情況；若工作單元數不多於 worker
    數，提交順序本來就不影響總牆鐘時間）。

    參數
    ------
    build_worker_resources / run_unit:
        見模組文件「執行引擎」一節；必須是模組層級可匯入的函式（或
        ``functools.partial`` 包住模組層級函式），因為 ``mp_context_method="spawn"``
        （預設）需要能夠 pickle 匯入它們。呼叫端若確定在 fork-safe 環境下執行且
        接受 fork 的已知風險（例如與執行緒或已開檔的 Numba 狀態互動），可傳入
        ``mp_context_method="fork"``，但這不是本模組的預設建議。
    output_dir:
        存放完成證據（``*.done.json``）、結果摘要（``*.result.json``）與進度快照
        （預設 ``progress.json``）的目錄；呼叫端負責先建立它，並確保它與其他 run
        的輸出目錄不共用（否則不同 run 的完成證據會互相干擾續跑判斷）。
    unit_costs:
        ``{unit_id: 估計成本}``（見 ``estimate_hour_major_segment_costs``）；
        提供時依此由大到小排序提交順序，未提供或某個 ``unit_id`` 不在裡面時
        退回該單元的 ``particle_count`` 當作成本代理（與 WP-7 舊版
        ``plan_worker_schedule`` 相同的退化行為，只影響提交順序，不影響
        排程結構本身）。
    staged_ready_dir / staged_flow_domain_ids / staged_marker_suffix:
        WP-13「依資料就緒動態加入」：``staged_ready_dir`` 提供時，本函式不會
        一開始就把全部工作單元送進佇列，而是先只送出
        ``flow_domain_id`` 已經就緒（``<staged_ready_dir>/<flow_domain_id>
        {staged_marker_suffix}`` 存在）的工作單元；尚未就緒的區域，其工作
        單元先保留在父程序記憶體內，主迴圈每一輪（每 ``poll_interval_seconds``
        秒）重新檢查尚未就緒區域的完成標記，一旦某個區域就緒就立刻把它的
        全部工作單元送進佇列（任何目前空閒的子程序都能立刻拿到，不需要
        等其他區域也就緒）。``staged_flow_domain_ids`` 必須列出**全部**
        預期會出現的 ``flow_domain_id``（即使某些現在已經就緒）：本函式用
        它決定「全部區域都就緒了嗎」，藉此判斷何時可以正常結束；``pending``
        內出現任何不在這個集合裡的 ``flow_domain_id`` 都會 fail-closed 拒絕
        （防止呼叫端忘記列出某個區域，導致該區域的工作單元被誤判永遠等待
        或永遠不會被視為「還在等」）。全部區域就緒且佇列清空、全部工作單元
        都有結果之後才會結束；只要還有任何區域未就緒，即使目前佇列暫時是
        空的（已就緒區域的單元都做完了），也會繼續等待、不會提前結束或讓
        子程序判斷「沒事做了」而退出（子程序本身完全不知道 staged 這件事，
        是父程序決定何時 submit，子程序只是持續存在、等待下一次
        ``submit``）。

    續跑行為：本函式一律先呼叫 ``filter_pending_units`` 排除輸入雜湊完全相符的
    已完成單元，只把真正待執行的單元排入排程；就算所有單元都已完成（``units`` 全部
    命中既有完成證據），本函式仍會回傳一份 ``NativeRunSummary``（不會啟動任何
    worker 子程序），讓呼叫端可以安全地把「重跑一次確認完成度」與「第一次執行」
    寫成同一段程式碼路徑。

    已知限制（誠實揭露）
    --------------------
    若某個子程序本身異常終止（作業系統層級的 kill、segfault，不是
    ``run_unit`` 內部例外），``ProcessPoolExecutor`` 會把整個 pool 標記成
    broken；本函式偵測到這個情況時，會把當時仍在佇列裡（已提交但尚未拿到
    結果）與仍在等待資料就緒、尚未提交的工作單元**全部**記為失敗（不會嘗試
    在同一個 pool 內重試或啟動替代子程序），並立即停止、回傳目前為止的
    ``NativeRunSummary``——這與單一工作單元計算失敗（只隔離那一個單元）是
    不同層級的失敗，因為 pool 一旦 broken，父程序已經無法確認其他子程序是
    否還存活、是否已經寫出部分結果。呼叫端應該把這種情況視為「這次 run 需要
    重跑」，重新呼叫 ``run_native_plan``（同一個 ``output_dir``）會透過
    ``filter_pending_units`` 自動跳過已經正確完成的單元，只重算真正失敗或
    未完成的部分。
    """

    if worker_count < 1:
        raise NativeOrchestratorError("worker_count 必須是正整數")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    progress_path = output_path / progress_filename

    pending, skipped = filter_pending_units(units, output_path)
    started_at = time.time_ns()
    start_monotonic = time.monotonic()
    _write_progress_snapshot(
        progress_path,
        total_units=len(units),
        skipped=skipped,
        live_counts={},
        live_elapsed_seconds=[],
        started_at_utc_ns=started_at,
    )

    if not pending:
        return NativeRunSummary(
            total_units=len(units),
            completed_unit_count=0,
            failed_unit_count=0,
            skipped_unit_count=len(skipped),
            failed_unit_ids=(),
            worker_reports=(),
            elapsed_seconds=time.monotonic() - start_monotonic,
        )

    def _cost_of(unit: NativeWorkUnit) -> float:
        if unit_costs is not None and unit.unit_id in unit_costs:
            return float(unit_costs[unit.unit_id])
        return float(unit.particle_count)

    ordered_pending = sorted(pending, key=_cost_of, reverse=True)

    staged_ready_path: Path | None = None
    not_ready_domains: set[str] = set()
    if staged_ready_dir is not None:
        if not staged_flow_domain_ids:
            raise NativeOrchestratorError("staged_ready_dir 提供時必須同時提供 staged_flow_domain_ids")
        staged_ready_path = Path(staged_ready_dir)
        expected_domains = set(staged_flow_domain_ids)
        present_domains = {u.flow_domain_id for u in ordered_pending}
        unexpected = present_domains - expected_domains
        if unexpected:
            raise NativeOrchestratorError(
                f"pending 出現不在 staged_flow_domain_ids 內的流場區域：{sorted(unexpected)}"
            )
        not_ready_domains = {
            domain
            for domain in expected_domains
            if not _domain_staged_marker_path(staged_ready_path, domain, staged_marker_suffix).is_file()
        }

    ready_now = [u for u in ordered_pending if u.flow_domain_id not in not_ready_domains]
    held_back = [u for u in ordered_pending if u.flow_domain_id in not_ready_domains]

    n_workers = min(worker_count, len(ordered_pending))
    ctx = _multiprocessing.get_context(mp_context_method)

    outcomes: dict[str, NativeUnitOutcome] = {}
    live_counts: dict[str, int] = {"completed": 0, "failed": 0}
    live_elapsed: list[float] = []
    broken_pool_error: str | None = None

    def _record_outcome(unit_id: str, outcome: NativeUnitOutcome) -> None:
        outcomes[unit_id] = outcome
        if outcome.status == "completed":
            live_counts["completed"] = live_counts.get("completed", 0) + 1
            live_elapsed.append(outcome.elapsed_seconds)
        elif outcome.status == "failed":
            live_counts["failed"] = live_counts.get("failed", 0) + 1

    def _mark_unsubmitted_as_failed(remaining: Sequence[NativeWorkUnit], *, reason: str) -> None:
        now_ns = time.time_ns()
        for unit in remaining:
            outcome = NativeUnitOutcome(
                unit_id=unit.unit_id,
                status="failed",
                input_hash=unit.input_hash,
                worker_id=-1,
                elapsed_seconds=0.0,
                particle_count=unit.particle_count,
                completed_at_utc_ns=now_ns,
                error=reason,
            )
            _persist_outcome(output_path, outcome, extra_summary=None)
            _record_outcome(unit.unit_id, outcome)

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        futures: dict[Any, str] = {}
        for unit in ready_now:
            future = pool.submit(
                _run_single_unit_task,
                unit.to_spec(),
                build_worker_resources,
                run_unit,
                str(output_path),
                numba_cache_dir,
            )
            futures[future] = unit.unit_id

        while futures or held_back or not_ready_domains:
            if staged_ready_path is not None and not_ready_domains:
                newly_ready = {
                    domain
                    for domain in not_ready_domains
                    if _domain_staged_marker_path(staged_ready_path, domain, staged_marker_suffix).is_file()
                }
                if newly_ready:
                    not_ready_domains -= newly_ready
                    still_held_back: list[NativeWorkUnit] = []
                    for unit in held_back:
                        if unit.flow_domain_id in newly_ready:
                            future = pool.submit(
                                _run_single_unit_task,
                                unit.to_spec(),
                                build_worker_resources,
                                run_unit,
                                str(output_path),
                                numba_cache_dir,
                            )
                            futures[future] = unit.unit_id
                        else:
                            still_held_back.append(unit)
                    held_back = still_held_back

            if not futures:
                # 佇列暫時空了，但還有區域未就緒（或還有工作單元在等待它們的
                # 區域就緒）：子程序繼續存活、等待下一輪 submit，不視為完成。
                _write_progress_snapshot(
                    progress_path, total_units=len(units), skipped=skipped,
                    live_counts=live_counts, live_elapsed_seconds=live_elapsed,
                    started_at_utc_ns=started_at,
                )
                if held_back or not_ready_domains:
                    time.sleep(poll_interval_seconds)
                    continue
                break

            done_now = [future for future in list(futures) if future.done()]
            if not done_now:
                _write_progress_snapshot(
                    progress_path, total_units=len(units), skipped=skipped,
                    live_counts=live_counts, live_elapsed_seconds=live_elapsed,
                    started_at_utc_ns=started_at,
                )
                time.sleep(poll_interval_seconds)
                continue

            for future in done_now:
                unit_id = futures.pop(future)
                try:
                    outcome = future.result()
                except _BrokenProcessPool as exc:
                    # 整個 pool 已經損壞（子程序被作業系統砍掉），父程序無法再
                    # 信任其他仍在佇列裡的子程序狀態；見本函式文件「已知限制」。
                    broken_pool_error = f"{type(exc).__name__}: {exc}"
                    break
                _record_outcome(unit_id, outcome)

            if broken_pool_error is not None:
                break

            _write_progress_snapshot(
                progress_path, total_units=len(units), skipped=skipped,
                live_counts=live_counts, live_elapsed_seconds=live_elapsed,
                started_at_utc_ns=started_at,
            )

        if broken_pool_error is not None:
            unresolved_unit_ids = set(futures.values()) | {u.unit_id for u in held_back}
            unresolved_units = [u for u in ordered_pending if u.unit_id in unresolved_unit_ids]
            _mark_unsubmitted_as_failed(
                unresolved_units, reason=f"worker_process_pool_broken: {broken_pool_error}"
            )

    all_outcomes = list(outcomes.values())
    completed = sum(1 for outcome in all_outcomes if outcome.status == "completed")
    failed_outcomes = [outcome for outcome in all_outcomes if outcome.status == "failed"]
    _write_progress_snapshot(
        progress_path,
        total_units=len(units),
        skipped=skipped,
        live_counts={"completed": completed, "failed": len(failed_outcomes)},
        live_elapsed_seconds=[
            outcome.elapsed_seconds for outcome in all_outcomes if outcome.status == "completed"
        ],
        started_at_utc_ns=started_at,
    )

    by_worker: dict[int, list[NativeUnitOutcome]] = {}
    for outcome in all_outcomes:
        by_worker.setdefault(outcome.worker_id, []).append(outcome)
    worker_reports = tuple(
        WorkerRunReport(worker_id=worker_id, outcomes=tuple(worker_outcomes))
        for worker_id, worker_outcomes in sorted(by_worker.items())
    )

    return NativeRunSummary(
        total_units=len(units),
        completed_unit_count=completed,
        failed_unit_count=len(failed_outcomes),
        skipped_unit_count=len(skipped),
        failed_unit_ids=tuple(sorted(outcome.unit_id for outcome in failed_outcomes)),
        worker_reports=worker_reports,
        elapsed_seconds=time.monotonic() - start_monotonic,
    )


# ══════════════════════════════════════════════════════════════════════════
# 7. 規劃檔序列化：供 ``lbt native-plan``／``lbt native-run`` 之間交換工作單元
# ══════════════════════════════════════════════════════════════════════════


def _work_unit_to_record(unit: NativeWorkUnit) -> dict[str, Any]:
    return {
        "unit_id": unit.unit_id,
        "analysis_region_id": unit.analysis_region_id,
        "flow_domain_id": unit.flow_domain_id,
        "study_site_id": unit.study_site_id,
        "arrival_time_utc_ns": int(unit.arrival_time_utc_ns),
        "settling_velocity_mps": float(unit.settling_velocity_mps),
        "particle_row_indices": unit.particle_row_indices.tolist(),
        "input_hash": unit.input_hash,
        "resolved_max_backtrack_seconds": unit.resolved_max_backtrack_seconds,
        "pre_window_deposition": bool(unit.pre_window_deposition),
        "bed_residence_age_hours": unit.bed_residence_age_hours,
    }


def _work_unit_from_record(record: Mapping[str, Any]) -> NativeWorkUnit:
    resolved_max_backtrack_seconds = record.get("resolved_max_backtrack_seconds")
    bed_residence_age_hours = record.get("bed_residence_age_hours")
    return NativeWorkUnit(
        unit_id=str(record["unit_id"]),
        analysis_region_id=str(record["analysis_region_id"]),
        flow_domain_id=str(record["flow_domain_id"]),
        study_site_id=str(record["study_site_id"]),
        arrival_time_utc_ns=int(record["arrival_time_utc_ns"]),
        settling_velocity_mps=float(record["settling_velocity_mps"]),
        particle_row_indices=np.asarray(record["particle_row_indices"], dtype=np.int64),
        input_hash=str(record["input_hash"]),
        resolved_max_backtrack_seconds=(
            None if resolved_max_backtrack_seconds is None else float(resolved_max_backtrack_seconds)
        ),
        pre_window_deposition=bool(record.get("pre_window_deposition", False)),
        bed_residence_age_hours=(
            None if bed_residence_age_hours is None else int(bed_residence_age_hours)
        ),
    )


def write_native_plan_file(
    path: str | Path,
    *,
    units: Sequence[NativeWorkUnit],
    gap_checks: Sequence[ArrivalGapCheck],
    summary: NativeWorkPlanSummary,
    manifest: Mapping[str, Any] | None = None,
    hour_major_members: Mapping[str, Sequence[NativeWorkUnit]] | None = None,
    backtrack_mode_declaration: Mapping[str, Any] | None = None,
) -> None:
    """把規劃結果（工作單元、缺口檢查、規模摘要、manifest）寫成單一 JSON 計畫檔。

    ``lbt native-plan`` 只產生這個檔案、不執行任何計算；``lbt native-run`` 讀入
    這個檔案重建 ``NativeWorkUnit`` 序列後才開始排程與執行，兩個命令因此可以在
    不同時間、甚至不同人核准後才分別執行。

    ``hour_major_members``：只在 ``units`` 是 ``plan_native_hour_major_work_units``
    摺疊過的「每個流場區域一個」排程單元時才提供，內容是該函式回傳的
    ``members`` 對照表（``{domain_unit_id: (原始逐工作單元 NativeWorkUnit, ...)}``）
    原封不動寫進去；``lbt native-run`` 本身不需要讀這個欄位（``run_native_plan``
    只依賴 ``units`` 本身），但 ``--resource-provider`` 指定的 ``run_unit`` 實作
    （即 ``production_wiring.py``）需要它才能對 ``hourmajor::`` 開頭的
    ``spec.unit_id`` 查出底下實際涵蓋哪些到達時刻／沉降速度／測站組合——它可以
    用 ``read_native_plan_file`` 再讀一次同一個計畫檔路徑（native-run 呼叫時已經
    知道這個路徑），取回的 payload 裡就有這個欄位，見
    ``read_native_hour_major_members`` 的重建方式。
    """

    unsafe = [check for check in gap_checks if not check.is_gap_safe]
    payload = {
        "plan_kind": "native_backward_batch_plan_v1",
        "is_gap_safe_ready": len(unsafe) == 0,
        "summary": asdict(summary),
        "gap_checks": [asdict(check) for check in gap_checks],
        "units": [_work_unit_to_record(unit) for unit in units],
    }
    if manifest is not None:
        payload["manifest"] = dict(manifest)
    if hour_major_members is not None:
        payload["hour_major_members"] = {
            domain_unit_id: [_work_unit_to_record(member) for member in members]
            for domain_unit_id, members in hour_major_members.items()
        }
    if backtrack_mode_declaration is not None:
        # WP-17：把「設定檔宣稱的模式」與「本次規劃實際採用的模式與 H」一起寫進計畫檔，
        # 避免 native-run／native-postprocess 只看到摺疊後的工作單元、猜不出到底用了
        # 哪個回溯模式（見 backtrack_horizon.py 模組文件）。
        payload["backtrack_horizon"] = dict(backtrack_mode_declaration)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(output_path, payload)


def read_native_plan_file(path: str | Path) -> tuple[tuple[NativeWorkUnit, ...], Mapping[str, Any]]:
    """讀回 ``write_native_plan_file`` 產生的計畫檔，回傳工作單元序列與完整 payload。"""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    units = tuple(_work_unit_from_record(record) for record in payload["units"])
    return units, payload


def read_native_hour_major_members(
    payload: Mapping[str, Any],
) -> Mapping[str, tuple[NativeWorkUnit, ...]]:
    """從 ``read_native_plan_file`` 回傳的 payload 重建 ``hour_major_members``。

    只有用 ``--schedule hour-major`` 寫出的計畫檔才有這個欄位；其餘情況回傳空
    字典（呼叫端可以用 ``bool(members)`` 判斷這份計畫是不是小時主軸排程）。
    """

    raw = payload.get("hour_major_members")
    if not raw:
        return {}
    return {
        domain_unit_id: tuple(_work_unit_from_record(record) for record in records)
        for domain_unit_id, records in raw.items()
    }
