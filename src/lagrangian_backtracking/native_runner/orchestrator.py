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

import contextlib
import json
import multiprocessing as _multiprocessing
import os
import queue
import time
import traceback
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from ..arrival_times import ArrivalTime, select_arrival_times
from ..provenance import CodeProvenance, collect_code_provenance
from .forcing_bridge import ForcingGapInventoryReport
from .particle_input import BatchGroup, ParticleIdentityTable, validate_batch_partition

__all__ = [
    "NativeWorkUnit",
    "NativeWorkUnitSpec",
    "plan_native_work_units",
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
    """單一 worker 處理完其被指派的整串工作單元後的完整回報。"""

    worker_id: int
    outcomes: tuple[NativeUnitOutcome, ...]
    fatal_error: str | None = None


def _run_assigned_units(
    worker_id: int,
    unit_specs: tuple[NativeWorkUnitSpec, ...],
    *,
    build_worker_resources: WorkerResourceBuilder,
    run_unit: UnitRunner,
    output_dir: str,
    numba_cache_dir: str | None,
    progress_queue: Any,
) -> WorkerRunReport:
    """單一 worker 子程序的主迴圈：建構一次資源、連續處理整串工作單元。

    這是模組層級（非巢狀）函式，是刻意的設計：``multiprocessing`` 的 ``spawn``／
    ``forkserver`` 啟動方式要求任務函式可以被 pickle 匯入，巢狀函式或閉包在
    ``spawn`` 下無法可靠序列化；``build_worker_resources``／``run_unit`` 由呼叫端
    以模組層級函式（或其 ``functools.partial``）提供，同樣是為了滿足這個限制。
    """

    output_path = Path(output_dir)
    if numba_cache_dir:
        os.environ["NUMBA_CACHE_DIR"] = numba_cache_dir

    try:
        resources = build_worker_resources(worker_id)
    except Exception:  # noqa: BLE001 - 資源建構失敗必須讓整個 worker 的單元都回報失敗，不能讓程序直接崩潰
        error_text = traceback.format_exc()
        now_ns = time.time_ns()
        outcomes = tuple(
            NativeUnitOutcome(
                unit_id=spec.unit_id,
                status="failed",
                input_hash=spec.input_hash,
                worker_id=worker_id,
                elapsed_seconds=0.0,
                particle_count=int(spec.particle_row_indices.shape[0]),
                completed_at_utc_ns=now_ns,
                error=error_text,
            )
            for spec in unit_specs
        )
        for outcome in outcomes:
            _persist_outcome(output_path, outcome, extra_summary=None)
            _safe_put(
                progress_queue, {"worker_id": worker_id, "unit_id": outcome.unit_id, "status": "failed"}
            )
        return WorkerRunReport(worker_id=worker_id, outcomes=outcomes, fatal_error=error_text)

    outcomes: list[NativeUnitOutcome] = []
    for spec in unit_specs:
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
        except Exception:  # noqa: BLE001 - 單一單元失敗必須被隔離，不能拖垮同 worker 佇列的其餘單元
            elapsed = time.monotonic() - start
            error_text = traceback.format_exc()
            outcome = NativeUnitOutcome(
                unit_id=spec.unit_id,
                status="failed",
                input_hash=spec.input_hash,
                worker_id=worker_id,
                elapsed_seconds=elapsed,
                particle_count=int(spec.particle_row_indices.shape[0]),
                completed_at_utc_ns=time.time_ns(),
                error=error_text,
            )
            _persist_outcome(output_path, outcome, extra_summary=None)
        outcomes.append(outcome)
        _safe_put(
            progress_queue,
            {
                "worker_id": worker_id,
                "unit_id": outcome.unit_id,
                "status": outcome.status,
                "elapsed_seconds": outcome.elapsed_seconds,
                "particle_count": outcome.particle_count,
            },
        )
    return WorkerRunReport(worker_id=worker_id, outcomes=tuple(outcomes))


def _safe_put(progress_queue: Any, message: Mapping[str, Any]) -> None:
    """進度回報是盡力而為的觀測用途，佇列已滿或已關閉都不應該讓計算失敗。"""

    with contextlib.suppress(Exception):
        progress_queue.put_nowait(dict(message))


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
) -> NativeRunSummary:
    """規劃、排程並平行執行一批 native 工作單元，支援續跑與失敗隔離。

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

    續跑行為：本函式一律先呼叫 ``filter_pending_units`` 排除輸入雜湊完全相符的
    已完成單元，只把真正待執行的單元排入排程；就算所有單元都已完成（``units`` 全部
    命中既有完成證據），本函式仍會回傳一份 ``NativeRunSummary``（不會啟動任何
    worker 子程序），讓呼叫端可以安全地把「重跑一次確認完成度」與「第一次執行」
    寫成同一段程式碼路徑。
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

    schedule = plan_worker_schedule(pending, worker_count=worker_count)
    unit_by_id = {u.unit_id: u for u in pending}

    ctx = _multiprocessing.get_context(mp_context_method)
    manager = ctx.Manager()
    progress_queue = manager.Queue()
    try:
        worker_reports: dict[int, WorkerRunReport] = {}
        live_counts: dict[str, int] = {"completed": 0, "failed": 0}
        live_elapsed: list[float] = []
        with ProcessPoolExecutor(max_workers=len(schedule), mp_context=ctx) as pool:
            futures = {}
            for worker_index, unit_ids in schedule.items():
                specs = tuple(unit_by_id[unit_id].to_spec() for unit_id in unit_ids)
                future = pool.submit(
                    _run_assigned_units,
                    worker_index,
                    specs,
                    build_worker_resources=build_worker_resources,
                    run_unit=run_unit,
                    output_dir=str(output_path),
                    numba_cache_dir=numba_cache_dir,
                    progress_queue=progress_queue,
                )
                futures[future] = worker_index

            pending_futures = set(futures)
            while pending_futures:
                _drain_progress_queue(progress_queue, live_counts, live_elapsed)
                _write_progress_snapshot(
                    progress_path,
                    total_units=len(units),
                    skipped=skipped,
                    live_counts=live_counts,
                    live_elapsed_seconds=live_elapsed,
                    started_at_utc_ns=started_at,
                )
                done_now = {future for future in pending_futures if future.done()}
                for future in done_now:
                    pending_futures.discard(future)
                    worker_reports[futures[future]] = future.result()
                if pending_futures:
                    time.sleep(poll_interval_seconds)
            _drain_progress_queue(progress_queue, live_counts, live_elapsed)
    finally:
        manager.shutdown()

    all_outcomes = [outcome for report in worker_reports.values() for outcome in report.outcomes]
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

    return NativeRunSummary(
        total_units=len(units),
        completed_unit_count=completed,
        failed_unit_count=len(failed_outcomes),
        skipped_unit_count=len(skipped),
        failed_unit_ids=tuple(sorted(outcome.unit_id for outcome in failed_outcomes)),
        worker_reports=tuple(worker_reports[index] for index in sorted(worker_reports)),
        elapsed_seconds=time.monotonic() - start_monotonic,
    )


def _drain_progress_queue(
    progress_queue: Any, live_counts: dict[str, int], live_elapsed: list[float]
) -> None:
    """非阻塞地清空目前累積的進度訊息，更新父程序端的即時統計。"""

    while True:
        try:
            message = progress_queue.get_nowait()
        except queue.Empty:
            return
        except (OSError, EOFError):  # pragma: no cover - manager 已關閉時的防禦性處理
            return
        status = message.get("status")
        if status in ("completed", "failed"):
            live_counts[status] = live_counts.get(status, 0) + 1
            if status == "completed" and "elapsed_seconds" in message:
                live_elapsed.append(float(message["elapsed_seconds"]))


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
    }


def _work_unit_from_record(record: Mapping[str, Any]) -> NativeWorkUnit:
    return NativeWorkUnit(
        unit_id=str(record["unit_id"]),
        analysis_region_id=str(record["analysis_region_id"]),
        flow_domain_id=str(record["flow_domain_id"]),
        study_site_id=str(record["study_site_id"]),
        arrival_time_utc_ns=int(record["arrival_time_utc_ns"]),
        settling_velocity_mps=float(record["settling_velocity_mps"]),
        particle_row_indices=np.asarray(record["particle_row_indices"], dtype=np.int64),
        input_hash=str(record["input_hash"]),
    )


def write_native_plan_file(
    path: str | Path,
    *,
    units: Sequence[NativeWorkUnit],
    gap_checks: Sequence[ArrivalGapCheck],
    summary: NativeWorkPlanSummary,
    manifest: Mapping[str, Any] | None = None,
) -> None:
    """把規劃結果（工作單元、缺口檢查、規模摘要、manifest）寫成單一 JSON 計畫檔。

    ``lbt native-plan`` 只產生這個檔案、不執行任何計算；``lbt native-run`` 讀入
    這個檔案重建 ``NativeWorkUnit`` 序列後才開始排程與執行，兩個命令因此可以在
    不同時間、甚至不同人核准後才分別執行。
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
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(output_path, payload)


def read_native_plan_file(path: str | Path) -> tuple[tuple[NativeWorkUnit, ...], Mapping[str, Any]]:
    """讀回 ``write_native_plan_file`` 產生的計畫檔，回傳工作單元序列與完整 payload。"""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    units = tuple(_work_unit_from_record(record) for record in payload["units"])
    return units, payload
