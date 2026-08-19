"""情境分片、member run unit 與可重現 reference batch executor。

計畫書的 10×20×50 是每站基礎 scenario 數；隨機系集成員是 scenario 外層維度。本模組
明確建立 ``Scenario × member_id`` run unit，使每站總軌跡為 ``10,000×M``，並保證
修改 shard 大小、worker 數或輸入 manifest 列順序不會改變 particle ID 與亂數 seed。
reference executor 以逐粒子 NumPy 引擎建立科學基準；正式大批次仍須在 Numba backend
通過逐項等價與 benchmark gate 後才能宣稱 production ready。
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
    """一條可獨立重跑的 scenario/member 軌跡識別與 seed。"""

    scenario: Scenario
    experiment_case_id: str
    member_id: int
    particle_id: str
    seed: int


@dataclass(frozen=True, slots=True)
class ScenarioShard:
    """以完整 scenario 為邊界的不可重疊分片。

    同一 scenario 的 M 個 members 不跨 shard，便於檢查情境完整率與一致分母；若 pilot
    顯示單一 scenario 的 M 過大，須另建立已記錄的 member-subshard schema，不可在此
    靜默拆分。
    """

    shard_id: str
    experiment_case_id: str
    scenario_start_index: int
    scenario_stop_index: int
    scenarios: tuple[Scenario, ...]
    members_per_scenario: int

    @property
    def scenario_count(self) -> int:
        """回傳 shard 中的基礎 scenario 數，不乘 M。"""

        return len(self.scenarios)

    @property
    def particle_count(self) -> int:
        """回傳實際需積分的 trajectory 數，即 scenario_count×M。"""

        return self.scenario_count * self.members_per_scenario


@dataclass(frozen=True, slots=True)
class ReferenceParticleRequest:
    """reference executor 所需的單粒子物理依賴。

    factory 可依 scenario 的 flow domain 共用 forcing cache，並依 behavior/material 指派
    垂向速度與擴散；這裡不在 runner 內硬編碼 SERVER 路徑或資料載入策略。
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
    """排序、驗證並把 scenario manifest 切成穩定分片。

    排序只用唯一 ``scenario_id``，因此來源 Parquet/YAML 的列順序不影響 shard。分片 ID
    含 experiment case 與 scenario ID 範圍的 hash；同一情境不可重複，空 manifest、
    ``M<1`` 或無效 shard size 皆立即失敗。
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
    """依 scenario、member 穩定順序逐一產生 run units，不一次配置大型清單。"""

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
    """以逐粒子 NumPy engine 執行一個 shard，供驗證與小型 pilot。

    factory 產出的 initial state 必須逐項符合 run unit；這可阻止錯 receptor、case 或
    member 的初始列被寫入正確 seed 的結果。PCG64DXSM 接受 128-bit 派生 seed，確保
    單獨重跑某 member 與批次執行得到相同 Brownian 序列。
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
