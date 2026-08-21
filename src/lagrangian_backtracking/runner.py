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
from .diffusion import DiffusionCoefficients
from .engine import EngineSettings, ParticleResult, run_particle
from .integrators import VelocityProvider
from .models import ParticleState
from .scenarios import Scenario, derive_member_seed, stable_identifier


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
    拆分格式，不能在這裡悄悄拆分。
    """

    shard_id: str
    experiment_case_id: str
    scenario_start_index: int
    scenario_stop_index: int
    scenarios: tuple[Scenario, ...]
    members_per_scenario: int

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

    建立請求的函式可依情境所屬流場共用已開啟的資料，並依材料行為指定垂向速度與擴散。
    本模組不硬寫伺服器路徑或資料載入方式，讓正式執行環境可由設定檔指定。
    """

    initial_state: ParticleState
    velocity: VelocityProvider
    boundaries: BoundaryGeometry
    behavior_class: str
    diffusion: DiffusionCoefficients
    settings: EngineSettings


def plan_scenario_shards(
    scenarios: Sequence[Scenario],
    *,
    members_per_scenario: int,
    shard_scenario_count: int,
    experiment_case_id: str,
) -> list[ScenarioShard]:
    """排序、檢查並把情境清單切成固定且可重現的批次。

    僅依唯一的 ``scenario_id`` 排序，所以輸入資料檔的列順序不影響批次內容。批次識別碼
    包含實驗案例與情境識別碼範圍的雜湊值；重複情境、空清單、每情境成員數小於一或不合理
    的批次大小都會立即報錯。
    """

    if not scenarios:
        raise ValueError("scenario manifest 不可為空")
    if members_per_scenario < 1 or shard_scenario_count < 1:
        raise ValueError("members_per_scenario 與 shard_scenario_count 必須為正")
    if not experiment_case_id:
        raise ValueError("experiment_case_id 不可空白")
    ordered = sorted(scenarios, key=lambda item: item.scenario_id)
    identifiers = [item.scenario_id for item in ordered]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("scenario manifest 含重複 scenario_id")
    shards: list[ScenarioShard] = []
    for start in range(0, len(ordered), shard_scenario_count):
        stop = min(start + shard_scenario_count, len(ordered))
        subset = tuple(ordered[start:stop])
        shard_hash = stable_identifier(
            "shd",
            [experiment_case_id, subset[0].scenario_id, subset[-1].scenario_id, str(len(subset))],
            length=20,
        )
        shards.append(
            ScenarioShard(
                shard_id=f"{start:08d}-{stop:08d}_{shard_hash}",
                experiment_case_id=experiment_case_id,
                scenario_start_index=start,
                scenario_stop_index=stop,
                scenarios=subset,
                members_per_scenario=members_per_scenario,
            )
        )
    return shards


def iter_run_units(shard: ScenarioShard, *, master_seed: int) -> Iterator[RunUnit]:
    """依固定情境與成員順序逐一產生執行單位，避免一次建立龐大清單。"""

    if master_seed < 0:
        raise ValueError("master_seed 不可為負")
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
