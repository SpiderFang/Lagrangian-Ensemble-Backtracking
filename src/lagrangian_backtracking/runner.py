"""安排情境批次、隨機系集成員與可重現的基準計算。

計畫書的 10×20×50 是每個研究站點的基礎情境數；每個情境外面再配置 ``M`` 個獨立的
隨機系集成員。本模組明確建立「情境 × 成員」的執行單位，因此每站總軌跡數是
``10,000 × M``。不論一次處理多少情境、使用多少工作程序或輸入清單原有順序如何，
粒子識別碼與亂數種子都不會改變。此處的 NumPy 逐粒子計算是科學比對基準；大量正式
計算的加速版本必須先證明與它逐項一致並完成效能驗證。
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from .boundaries import BoundaryGeometry
from .diffusion import DiffusionModel
from .engine import EngineSettings, ParticleResult, run_particle
from .integrators import VelocityProvider
from .models import ParticleState
from .scenarios import Scenario, derive_member_seed, stable_identifier

# 此政策只規範 I/O 與批次的固定遍歷順序，不改變 Scenario 的內容、scenario_id、粒子
# 識別碼或種子。將排序鍵公開並版本化，是為了讓不同程序重建同一 run plan 時，能辨識
# 「區域／到達時刻」的執行 locality 契約，而不是各自猜測輸入檔的列順序。
SCENARIO_ORDERING_POLICY = "analysis_region_arrival_utc_site_material_receptor_scenario_v1"


def scenario_execution_sort_key(scenario: Scenario) -> tuple[str, int, str, str, str, str]:
    """回傳情境的正式執行排序鍵。

    欄位依序代表分析區域、到達時間（UTC 奈秒）、研究站點、材料、受體與情境識別碼。
    這個純函式只讀取既有 ``Scenario`` 欄位；排序的目的是讓同一流場區域與到達時刻
    儘量相鄰，改善 I/O locality，並不改變科學樣本、物理公式、粒子識別碼或亂數種子。
    文字欄位必須是非空字串，到達時間必須是非 ``bool`` 的整數，否則立即拒絕不完整的
    manifest，避免錯誤資料在不同 Python 程序中產生不同的批次順序。

    Args:
        scenario: 已載入且含固定識別欄位的情境資料。

    Returns:
        可直接交給 ``sorted`` 的六元素 tuple；時間單位為 UTC 奈秒。

    Raises:
        ValueError: 欄位空白或到達時間型別不符合資料契約時 raised。
    """

    text_fields = (
        scenario.analysis_region_id,
        scenario.study_site_id,
        scenario.material_id,
        scenario.receptor_id,
        scenario.scenario_id,
    )
    if any(not isinstance(value, str) or not value.strip() for value in text_fields):
        raise ValueError("scenario execution ordering 欄位必須是非空字串")
    arrival_time = scenario.arrival_time_utc_ns
    if isinstance(arrival_time, bool) or not isinstance(arrival_time, int):
        raise ValueError("scenario arrival_time_utc_ns 必須是非 bool 整數")
    return (
        scenario.analysis_region_id,
        arrival_time,
        scenario.study_site_id,
        scenario.material_id,
        scenario.receptor_id,
        scenario.scenario_id,
    )


def _scenario_shard_id(
    *,
    experiment_case_id: str,
    execution_group_id: str,
    analysis_region_id: str,
    arrival_time_utc_ns: int,
    group_part_index: int,
    group_part_count: int,
    scenarios: Sequence[Scenario],
    start: int,
    stop: int,
) -> str:
    """依固定 group/part/range 內容產生 shard ID；供 plan 與 validator 共用。

    shard ID 不是科學欄位，但它是 checkpoint、output、lock 與 progress 的路徑根。把
    policy、group metadata、全域範圍與 subset 首尾／數量一併放入穩定雜湊，能在 plan 或
    scenario table 被替換時拒絕把另一批粒子誤認成原 shard；此 helper 不改粒子 ID 或 seed。
    """

    if not scenarios:
        raise ValueError("scenario shard 不可為空")
    shard_hash = stable_identifier(
        "shd",
        [
            experiment_case_id,
            SCENARIO_ORDERING_POLICY,
            execution_group_id,
            analysis_region_id,
            str(arrival_time_utc_ns),
            str(group_part_index),
            str(group_part_count),
            str(start),
            str(stop),
            scenarios[0].scenario_id,
            scenarios[-1].scenario_id,
            str(len(scenarios)),
        ],
        length=20,
    )
    return f"{start:08d}-{stop:08d}_{shard_hash}"


@dataclass(frozen=True, slots=True)
class RunUnit:
    """一條可單獨重跑的軌跡所需情境、成員編號、粒子識別碼與亂數種子。"""

    scenario: Scenario
    experiment_case_id: str
    member_id: int
    particle_id: str
    seed: int


@dataclass(frozen=True, slots=True)
class ScenarioShard:
    """以完整情境為單位、不互相重疊的一批工作。

    同一情境的 ``M`` 個成員不分到不同批次，才能直接檢查每個情境的成員是否完整，並讓
    各情境的統計分母一致。若試算顯示一個情境的成員數大到必須拆開，必須另外定義並記錄
    拆分格式，不能在這裡悄悄拆分。``execution_group_id`` 與五個 group metadata 固定
    保存此批次的分析區域、到達 UTC 奈秒與 group part；它們是 I/O locality 與 checkpoint
    邊界的稽核欄位，不是新的物理狀態。
    """

    shard_id: str
    experiment_case_id: str
    scenario_start_index: int
    scenario_stop_index: int
    scenarios: tuple[Scenario, ...]
    members_per_scenario: int
    execution_group_id: str
    analysis_region_id: str
    arrival_time_utc_ns: int
    group_part_index: int
    group_part_count: int

    @property
    def scenario_count(self) -> int:
        """回傳此批次的基礎情境數，不乘上每情境成員數。"""

        return len(self.scenarios)

    @property
    def particle_count(self) -> int:
        """回傳實際需要積分的軌跡數，即基礎情境數乘上每情境成員數。"""

        return self.scenario_count * self.members_per_scenario


@dataclass(frozen=True, slots=True)
class ReferenceParticleRequest:
    """計算一條基準粒子軌跡所需的物理資料與設定。

    建立請求的函式可依情境所屬流場共用已開啟的資料，並依材料行為指定垂向速度與
    擴散。``diffusion`` 可以是固定的 ``DiffusionCoefficients``，也可以是依步首位置、
    UTC 時間與 mesh triangle hint 回傳 ``DiffusionSample`` 的空間 provider；runner 只
    傳遞這個模型，不在批次層預先取樣或改變 checkpoint/output schema。本模組不硬寫
    伺服器路徑或資料載入方式，讓正式執行環境可由設定檔指定。
    """

    initial_state: ParticleState
    velocity: VelocityProvider
    boundaries: BoundaryGeometry
    behavior_class: str
    diffusion: DiffusionModel
    settings: EngineSettings


def plan_scenario_shards(
    scenarios: Sequence[Scenario],
    *,
    members_per_scenario: int,
    shard_scenario_count: int,
    experiment_case_id: str,
) -> list[ScenarioShard]:
    """排序、檢查並把情境清單切成固定且可重現的批次。

    先依 ``SCENARIO_ORDERING_POLICY`` 的完整排序鍵排序，再按
    ``(analysis_region_id, arrival_time_utc_ns)`` 建立 execution group；每個 group 只在
    自己的範圍內切割，永遠不會為填滿一個 shard 而跨越群組。這只影響資料讀取 locality
    與恢復邊界，不改 Scenario、粒子識別碼或 seed。批次識別碼包含群組、part 與首尾
    scenario 的穩定雜湊；重複情境、空清單、每情境成員數小於一或不合理的批次大小都會
    立即報錯。
    """

    if not scenarios:
        raise ValueError("scenario manifest 不可為空")
    if (
        type(members_per_scenario) is not int
        or type(shard_scenario_count) is not int
        or members_per_scenario < 1
        or shard_scenario_count < 1
    ):
        raise ValueError("members_per_scenario 與 shard_scenario_count 必須為正")
    if type(experiment_case_id) is not str or not experiment_case_id.strip():
        raise ValueError("experiment_case_id 不可空白")
    ordered = sorted(scenarios, key=scenario_execution_sort_key)
    identifiers = [item.scenario_id for item in ordered]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("scenario manifest 含重複 scenario_id")
    shards: list[ScenarioShard] = []
    # 以連續的區域／時間鍵分群；因為 ordered 已先按完整鍵排序，每個群組的列天然
    # contiguous。此處刻意不把下一群組搬進尚有空間的 shard，避免 checkpoint 恢復時
    # 一個批次同時依賴兩個不同流場到達時刻。
    grouped: list[tuple[tuple[str, int], list[Scenario]]] = []
    for scenario in ordered:
        group_key = (scenario.analysis_region_id, scenario.arrival_time_utc_ns)
        if not grouped or grouped[-1][0] != group_key:
            grouped.append((group_key, []))
        grouped[-1][1].append(scenario)

    global_start = 0
    for (analysis_region_id, arrival_time_utc_ns), group_scenarios in grouped:
        group_part_count = (len(group_scenarios) + shard_scenario_count - 1) // shard_scenario_count
        execution_group_id = stable_identifier(
            "grp",
            [SCENARIO_ORDERING_POLICY, analysis_region_id, str(arrival_time_utc_ns)],
            length=20,
        )
        for group_part_index in range(group_part_count):
            local_start = group_part_index * shard_scenario_count
            local_stop = min(local_start + shard_scenario_count, len(group_scenarios))
            subset = tuple(group_scenarios[local_start:local_stop])
            start = global_start + local_start
            stop = global_start + local_stop
            shards.append(
                ScenarioShard(
                    shard_id=_scenario_shard_id(
                        experiment_case_id=experiment_case_id,
                        execution_group_id=execution_group_id,
                        analysis_region_id=analysis_region_id,
                        arrival_time_utc_ns=arrival_time_utc_ns,
                        group_part_index=group_part_index,
                        group_part_count=group_part_count,
                        scenarios=subset,
                        start=start,
                        stop=stop,
                    ),
                    experiment_case_id=experiment_case_id,
                    scenario_start_index=start,
                    scenario_stop_index=stop,
                    scenarios=subset,
                    members_per_scenario=members_per_scenario,
                    execution_group_id=execution_group_id,
                    analysis_region_id=analysis_region_id,
                    arrival_time_utc_ns=arrival_time_utc_ns,
                    group_part_index=group_part_index,
                    group_part_count=group_part_count,
                )
            )
        global_start += len(group_scenarios)
    return shards


def iter_run_units(shard: ScenarioShard, *, master_seed: int) -> Iterator[RunUnit]:
    """依固定情境與成員順序逐一產生執行單位，避免一次建立龐大清單。"""

    if type(master_seed) is not int or master_seed < 0:
        raise ValueError("master_seed 必須是非負整數，且不可為 bool")
    for scenario in shard.scenarios:
        for member_id in range(shard.members_per_scenario):
            particle_id = stable_identifier(
                "prt",
                [scenario.scenario_id, shard.experiment_case_id, str(member_id)],
            )
            yield RunUnit(
                scenario=scenario,
                experiment_case_id=shard.experiment_case_id,
                member_id=member_id,
                particle_id=particle_id,
                seed=derive_member_seed(
                    master_seed=master_seed,
                    scenario_id=scenario.scenario_id,
                    experiment_case_id=shard.experiment_case_id,
                    member_id=member_id,
                ),
            )


def run_reference_shard(
    shard: ScenarioShard,
    *,
    master_seed: int,
    request_factory: Callable[[RunUnit], ReferenceParticleRequest],
    on_result: Callable[[RunUnit, ParticleResult], None] | None = None,
) -> list[ParticleResult]:
    """以 NumPy 逐粒子引擎執行一批工作，供驗證與小型試算。

    建立函式回傳的初始粒子狀態必須和執行單位逐欄一致，避免錯誤受體、實驗案例或成員的
    初始資料被寫成看似正確的結果。亂數產生器使用由主種子、情境、案例與成員共同導出的
    128 位元種子，因此單獨重跑某一成員與整批計算會得到相同的隨機擴散序列。
    """

    results: list[ParticleResult] = []
    for unit in iter_run_units(shard, master_seed=master_seed):
        request = request_factory(unit)
        state = request.initial_state
        identity = (
            state.particle_id,
            state.scenario_id,
            state.member_id,
            state.study_site_id,
            state.analysis_region_id,
            state.receptor_id,
            state.time_utc_ns,
        )
        expected = (
            unit.particle_id,
            unit.scenario.scenario_id,
            unit.member_id,
            unit.scenario.study_site_id,
            unit.scenario.analysis_region_id,
            unit.scenario.receptor_id,
            unit.scenario.arrival_time_utc_ns,
        )
        if identity != expected:
            raise ValueError(
                f"request initial_state 與 run unit 不一致：actual={identity}, expected={expected}"
            )
        result = run_particle(
            state,
            velocity=request.velocity,
            boundaries=request.boundaries,
            behavior_class=request.behavior_class,
            diffusion=request.diffusion,
            settings=request.settings,
            rng=np.random.Generator(np.random.PCG64DXSM(unit.seed)),
        )
        results.append(result)
        if on_result is not None:
            on_result(unit, result)
    if len(results) != shard.particle_count:
        raise RuntimeError("reference shard 結果數與 scenario×M 契約不符")
    return results


__all__ = [
    "SCENARIO_ORDERING_POLICY",
    "ReferenceParticleRequest",
    "RunUnit",
    "ScenarioShard",
    "iter_run_units",
    "plan_scenario_shards",
    "run_reference_shard",
    "scenario_execution_sort_key",
]
