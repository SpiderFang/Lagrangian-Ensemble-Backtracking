"""scenario×member 分片、seed 與 reference batch executor 測試。"""

from __future__ import annotations

from dataclasses import replace

from shapely.geometry import box

from lagrangian_backtracking.boundaries import BoundaryGeometry
from lagrangian_backtracking.diffusion import DiffusionCoefficients
from lagrangian_backtracking.engine import EngineSettings
from lagrangian_backtracking.models import ParticleState, ParticleStatus, VelocitySample
from lagrangian_backtracking.runner import (
    ReferenceParticleRequest,
    iter_run_units,
    plan_scenario_shards,
    run_reference_shard,
)
from lagrangian_backtracking.scenarios import Scenario


def _scenario(identifier: str, arrival_time_utc_ns: int = 100_000_000_000) -> Scenario:
    """建立可排序的貢寮合成 scenario。"""

    return Scenario(
        scenario_id=identifier,
        study_site_id="gongliao",
        analysis_region_id="A",
        material_id="neutral_000mmps",
        receptor_id=f"receptor-{identifier}",
        arrival_time_id="arrival-0",
        settling_velocity_mps=0.0,
        arrival_time_utc_ns=arrival_time_utc_ns,
        design_version="test-v1",
    )


def test_shard_size_does_not_change_run_unit_identity_or_seed() -> None:
    """重新排列 manifest 或調整 shard 大小時，scenario/member 的 ID 與 seed 皆相同。"""

    scenarios = [_scenario("s2"), _scenario("s0"), _scenario("s1")]
    plans_a = plan_scenario_shards(
        scenarios,
        members_per_scenario=3,
        shard_scenario_count=1,
        experiment_case_id="baseline",
    )
    plans_b = plan_scenario_shards(
        list(reversed(scenarios)),
        members_per_scenario=3,
        shard_scenario_count=2,
        experiment_case_id="baseline",
    )

    def flatten(plans):
        """提取可跨分片比較的不可變 run-unit 欄位。"""

        return [
            (unit.scenario.scenario_id, unit.member_id, unit.particle_id, unit.seed)
            for plan in plans
            for unit in iter_run_units(plan, master_seed=20260819)
        ]

    assert flatten(plans_a) == flatten(plans_b)
    assert sum(plan.particle_count for plan in plans_a) == 3 * 3


def test_reference_shard_runs_every_member_once() -> None:
    """reference executor 對每個 scenario/member 各跑一次並保留穩定粒子識別。"""

    shard = plan_scenario_shards(
        [_scenario("s0")],
        members_per_scenario=2,
        shard_scenario_count=1,
        experiment_case_id="baseline",
    )[0]
    boundaries = BoundaryGeometry(
        own_local_domain=box(-5.0, -5.0, 5.0, 5.0),
        flow_domain=box(-20.0, -5.0, 20.0, 5.0),
        foreign_local_domains={},
    )

    def velocity(x_m: float, y_m: float, z_m: float, time_utc_ns: int) -> VelocitySample:
        """固定東向流供 signed-time 常流測試；輸入只用於符合 provider 介面。"""

        del x_m, y_m, z_m, time_utc_ns
        return VelocitySample(1.0, 0.0, 0.0, 0.0, -10.0, 100.0, 1.0)

    def request(unit) -> ReferenceParticleRequest:
        """由 run unit 建立一致的 receptor 初始狀態與物理設定。"""

        state = ParticleState(
            particle_id=unit.particle_id,
            scenario_id=unit.scenario.scenario_id,
            member_id=unit.member_id,
            study_site_id=unit.scenario.study_site_id,
            analysis_region_id=unit.scenario.analysis_region_id,
            receptor_id=unit.scenario.receptor_id,
            x_m=0.0,
            y_m=0.0,
            z_m=-5.0,
            time_utc_ns=unit.scenario.arrival_time_utc_ns,
        )
        return ReferenceParticleRequest(
            initial_state=state,
            velocity=velocity,
            boundaries=boundaries,
            behavior_class="suspended",
            diffusion=DiffusionCoefficients(0.0, 0.0, 0.0),
            settings=EngineSettings(1.0, 4.0, 4.0, 100.0, 100, 0),
        )

    first = run_reference_shard(shard, master_seed=9, request_factory=request)
    second = run_reference_shard(shard, master_seed=9, request_factory=request)
    assert len(first) == 2
    assert len({result.final_state.particle_id for result in first}) == 2
    assert all(result.final_state.status == ParticleStatus.FLOW_DOMAIN_EXIT for result in first)
    assert [result.final_state for result in first] == [result.final_state for result in second]


def test_reference_executor_rejects_mismatched_initial_identity() -> None:
    """factory 若把 member ID 配錯，executor 必須在積分前 fail-fast。"""

    shard = plan_scenario_shards(
        [_scenario("s0")],
        members_per_scenario=1,
        shard_scenario_count=1,
        experiment_case_id="baseline",
    )[0]

    def request(unit) -> ReferenceParticleRequest:
        """刻意建立錯誤 member ID，驗證 identity gate。"""

        state = ParticleState(
            particle_id=unit.particle_id,
            scenario_id=unit.scenario.scenario_id,
            member_id=99,
            study_site_id=unit.scenario.study_site_id,
            analysis_region_id=unit.scenario.analysis_region_id,
            receptor_id=unit.scenario.receptor_id,
            x_m=0.0,
            y_m=0.0,
            z_m=-1.0,
            time_utc_ns=unit.scenario.arrival_time_utc_ns,
        )
        return ReferenceParticleRequest(
            initial_state=replace(state),
            velocity=lambda *_: VelocitySample(0.0, 0.0, 0.0, 0.0, -10.0, 1.0, 1.0),
            boundaries=BoundaryGeometry(box(-1, -1, 1, 1), box(-2, -2, 2, 2), {}),
            behavior_class="suspended",
            diffusion=DiffusionCoefficients(0.0, 0.0, 0.0),
            settings=EngineSettings(1.0, 1.0, 1.0, 1.0, 1, 0),
        )

    try:
        run_reference_shard(shard, master_seed=1, request_factory=request)
    except ValueError as error:
        assert "run unit 不一致" in str(error)
    else:
        raise AssertionError("identity mismatch 應被拒絕")
