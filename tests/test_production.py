"""CPU production batch、hint tracking 與 active compaction 測試。"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy

from shapely.geometry import box

from lagrangian_backtracking.boundaries import BoundaryGeometry
from lagrangian_backtracking.diffusion import DiffusionCoefficients
from lagrangian_backtracking.engine import EngineSettings
from lagrangian_backtracking.models import ParticleState, ParticleStatus, VelocitySample
from lagrangian_backtracking.production import (
    HintTrackingVelocityProvider,
    ProductionBatch,
    run_production_shard,
)
from lagrangian_backtracking.runner import ReferenceParticleRequest, plan_scenario_shards, run_reference_shard
from lagrangian_backtracking.scenarios import Scenario


def _scenario(identifier: str, arrival_time_utc_ns: int = 100_000_000_000) -> Scenario:
    """建立可排序的 synthetic scenario，保留正式 RunUnit 所需欄位。"""

    return Scenario(
        scenario_id=identifier,
        study_site_id="gongliao",
        analysis_region_id="A",
        material_id="oca_nonrecyclable_flexible_sheet",
        receptor_id=f"receptor-{identifier}",
        arrival_time_id="arrival-0",
        settling_velocity_mps=-0.001,
        arrival_time_utc_ns=arrival_time_utc_ns,
        design_version="test-v1",
    )


def _shard(*, scenario_count: int = 2, members_per_scenario: int = 2):
    """建立小型、固定順序的 scenario shard，避免測試依賴大型 manifest。"""

    return plan_scenario_shards(
        [_scenario(f"s{index}") for index in range(scenario_count)],
        members_per_scenario=members_per_scenario,
        shard_scenario_count=scenario_count,
        experiment_case_id="baseline",
    )[0]


def _settings() -> EngineSettings:
    """以固定一秒步長與短回溯窗提供快速且可重現的測試設定。"""

    return EngineSettings(
        dt_min_seconds=1.0,
        dt_max_seconds=1.0,
        output_interval_seconds=2.0,
        max_backtrack_seconds=4.0,
        maximum_step_count=100,
        earliest_forcing_time_utc_ns=0,
    )


def _boundaries() -> BoundaryGeometry:
    """建立足以支撐完整 4 秒測試的寬廣 local/flow domain。"""

    return BoundaryGeometry(
        own_local_domain=box(-100.0, -100.0, 100.0, 100.0),
        flow_domain=box(-1_000.0, -1_000.0, 1_000.0, 1_000.0),
        foreign_local_domains={},
    )


def _request(unit, *, narrow: bool = False) -> ReferenceParticleRequest:
    """依 unit 建立與 scenario identity 完全一致的 request。"""

    state = ParticleState(
        particle_id=unit.particle_id,
        scenario_id=unit.scenario.scenario_id,
        member_id=unit.member_id,
        study_site_id=unit.scenario.study_site_id,
        analysis_region_id=unit.scenario.analysis_region_id,
        receptor_id=unit.scenario.receptor_id,
        x_m=0.0,
        y_m=0.0,
        z_m=-10.0,
        time_utc_ns=unit.scenario.arrival_time_utc_ns,
    )

    def velocity(x_m: float, y_m: float, z_m: float, time_utc_ns: int) -> VelocitySample:
        """位置依賴速度場；它不讀 hint，方便比較 wrapper 前後的科學結果。"""

        del time_utc_ns
        return VelocitySample(
            0.2 + 0.01 * x_m,
            -0.1 + 0.01 * y_m,
            0.005 + 0.0001 * z_m,
            0.0,
            -100.0,
            100.0,
            10.0,
        )

    if narrow:
        boundaries = BoundaryGeometry(
            own_local_domain=box(-0.25, -10.0, 100.0, 10.0),
            flow_domain=box(-0.5, -10.0, 100.0, 10.0),
            foreign_local_domains={},
        )
        diffusion = DiffusionCoefficients(0.0, 0.0, 0.0)
    else:
        boundaries = _boundaries()
        diffusion = DiffusionCoefficients(0.1, 0.05, 0.02)
    return ReferenceParticleRequest(
        initial_state=state,
        velocity=velocity,
        boundaries=boundaries,
        behavior_class="sinking",
        diffusion=diffusion,
        settings=_settings(),
    )


def _factory(unit):
    """標準 factory，每次呼叫都建立全新的 closure 與物理 request。"""

    return _request(unit)


def test_reference_and_production_are_bitwise_equivalent_for_position_flow_and_brownian() -> None:
    """同 seed 的 reference 與 production 必須完整相等，不只 final position 相近。"""

    shard = _shard(scenario_count=2, members_per_scenario=2)
    reference = run_reference_shard(shard, master_seed=20260828, request_factory=_factory)
    production = run_production_shard(
        shard,
        master_seed=20260828,
        request_factory=_factory,
        active_chunk_size=2,
    )

    assert production == reference
    assert [item.final_state for item in production] == [item.final_state for item in reference]
    assert [item.observations for item in production] == [item.observations for item in reference]
    assert [item.events for item in production] == [item.events for item in reference]


def test_active_chunk_size_does_not_change_results_or_rng_sequence() -> None:
    """chunk=1 與大 chunk 仍採相同 source order，每條獨立 RNG 的結果必須完全一致。"""

    shard = _shard(scenario_count=3, members_per_scenario=2)
    one = run_production_shard(
        shard, master_seed=77, request_factory=_factory, active_chunk_size=1
    )
    large = run_production_shard(
        shard, master_seed=77, request_factory=_factory, active_chunk_size=100
    )
    assert one == large


def test_early_stops_compact_and_scatter_to_original_particle_identity() -> None:
    """第一個粒子提早離開不應讓後續 member 的 state 寫入錯誤 source index。"""

    shard = _shard(scenario_count=1, members_per_scenario=3)
    calls: Counter[str] = Counter()

    def factory(unit):
        """只讓 member 0 使用窄域，其他粒子維持完整回溯。"""

        calls[unit.particle_id] += 1
        return _request(unit, narrow=unit.member_id == 0)

    batch = ProductionBatch(shard, master_seed=19, request_factory=factory, active_chunk_size=1)
    results = batch.complete()

    assert all(count == 1 for count in calls.values())
    assert [result.final_state.particle_id for result in results] == [
        unit.particle_id for unit in batch.units
    ]
    assert len({result.final_state.particle_id for result in results}) == 3
    assert results[0].final_state.status == ParticleStatus.FLOW_DOMAIN_EXIT
    assert results[1].final_state.status == ParticleStatus.MAX_AGE
    assert results[2].final_state.status == ParticleStatus.MAX_AGE
    assert batch.particle_batch.to_particle_states() == [result.final_state for result in results]


def test_pre_window_production_batch_outputs_terminal_results_without_forcing_or_rng_use() -> None:
    """ProductionBatch 保留 pre-window terminal 結果，且不呼叫 provider、sweep 或 RNG。"""

    shard = _shard(scenario_count=1, members_per_scenario=2)
    velocity_calls: list[tuple[float, float, float, int]] = []

    def request_factory(unit) -> ReferenceParticleRequest:
        """建立可序列化的 terminal request，provider 僅作意外取樣哨兵。"""

        state = ParticleState(
            particle_id=unit.particle_id,
            scenario_id=unit.scenario.scenario_id,
            member_id=unit.member_id,
            study_site_id=unit.scenario.study_site_id,
            analysis_region_id=unit.scenario.analysis_region_id,
            receptor_id=unit.scenario.receptor_id,
            x_m=0.0,
            y_m=0.0,
            z_m=-10.0,
            time_utc_ns=unit.scenario.arrival_time_utc_ns,
            status=ParticleStatus.PRE_WINDOW_DEPOSITION,
        )

        def forbidden_velocity(
            x_m: float, y_m: float, z_m: float, time_utc_ns: int
        ) -> VelocitySample:
            """記錄若 terminal 粒子被誤送進數值取樣。"""

            velocity_calls.append((x_m, y_m, z_m, time_utc_ns))
            raise AssertionError("pre-window member 不得取樣 forcing")

        return ReferenceParticleRequest(
            initial_state=state,
            velocity=forbidden_velocity,
            boundaries=_boundaries(),
            behavior_class="sinking",
            diffusion=DiffusionCoefficients(0.0, 0.0, 0.0),
            settings=_settings(),
        )

    batch = ProductionBatch(
        shard,
        master_seed=20260916,
        request_factory=request_factory,
        active_chunk_size=1,
    )
    rng_states_before = [
        deepcopy(runtime.rng.bit_generator.state) for runtime in batch.runtimes
    ]
    progress = batch.advance()
    results = batch.results()

    assert progress.terminal is True
    assert progress.stepped_particle_count == 0
    assert progress.sweeps_completed == 0
    assert progress.active_particle_count == 0
    assert batch.sweep_count == 0
    assert velocity_calls == []
    assert [
        runtime.rng.bit_generator.state for runtime in batch.runtimes
    ] == rng_states_before
    assert len(results) == 2
    assert all(
        result.final_state.status is ParticleStatus.PRE_WINDOW_DEPOSITION
        and result.step_count == 0
        and len(result.observations) == 1
        and len(result.events) == 1
        and result.events[0].fraction == 0.0
        for result in results
    )


def test_hint_tracking_provider_forwards_and_updates_hint_and_keeps_plain_callable() -> None:
    """支援 hint 的 sample 收到逐次 hint；普通四參數 callable 仍可直接取樣。"""

    class HintSampler:
        """記錄輸入 hint 並回傳下一個 triangle_id 的最小測試 provider。"""

        def __init__(self) -> None:
            self.received: list[int | None] = []

        def sample(
            self,
            x_m: float,
            y_m: float,
            z_m: float,
            time_utc_ns: int,
            *,
            triangle_hint: int | None,
        ) -> VelocitySample:
            """回傳固定速度，僅用輸入座標符合 provider 介面。"""

            del x_m, y_m, z_m, time_utc_ns
            self.received.append(triangle_hint)
            return VelocitySample(0.0, 0.0, 0.0, 0.0, -10.0, 1.0, 1.0, triangle_id=7)

    sampler = HintSampler()
    provider = HintTrackingVelocityProvider(sampler)
    assert provider(0.0, 0.0, -1.0, 0).triangle_id == 7
    assert provider(0.0, 0.0, -1.0, 1).triangle_id == 7
    assert sampler.received == [None, 7]
    assert provider.triangle_hint == 7

    plain = HintTrackingVelocityProvider(
        lambda x_m, y_m, z_m, time_utc_ns: VelocitySample(1.0, 0.0, 0.0, 0.0, -10.0, 1.0, 1.0)
    )
    assert plain(0.0, 0.0, -1.0, 0).u_mps == 1.0
    assert plain.triangle_hint is None


def test_request_factory_is_called_once_per_run_unit() -> None:
    """批次初始化一次建立 request，後續 sweep 與 complete 不得重建外部 forcing。"""

    shard = _shard(scenario_count=2, members_per_scenario=3)
    calls: Counter[str] = Counter()

    def factory(unit):
        """記錄每個固定 particle_id 的 factory 次數。"""

        calls[unit.particle_id] += 1
        return _request(unit)

    batch = ProductionBatch(shard, master_seed=3, request_factory=factory, active_chunk_size=1)
    for _ in range(3):
        batch.advance()
    batch.complete()
    assert calls == {unit.particle_id: 1 for unit in batch.units}
