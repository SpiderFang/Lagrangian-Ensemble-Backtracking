"""粒子引擎步首環境 context 的型別、綁定與不變性測試。

本檔案使用小型 synthetic velocity callback 驗證資料流與停止語意；synthetic callback
只代表本機工程測試證據，不能視為真實 OCM／NWW3 資料或正式來源足跡結果。
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from shapely.geometry import box

from lagrangian_backtracking import EnvironmentSampleStatus
from lagrangian_backtracking.boundaries import BoundaryGeometry
from lagrangian_backtracking.diffusion import DiffusionCoefficients
from lagrangian_backtracking.engine import (
    EngineSettings,
    Observation,
    advance_particle_once,
    finalize_particle_execution,
    initialize_particle_execution,
    run_particle,
)
from lagrangian_backtracking.models import ParticleState, ParticleStatus, SampleQC, VelocitySample


def _state(*, age_seconds: float = 0.0, time_utc_ns: int = 100_000_000_000) -> ParticleState:
    """建立遠離垂向與水平邊界的 active 粒子，座標單位為公尺。"""

    return ParticleState(
        particle_id="p0",
        scenario_id="s0",
        member_id=0,
        study_site_id="gongliao",
        analysis_region_id="A",
        receptor_id="r0",
        x_m=0.0,
        y_m=0.0,
        z_m=-10.0,
        time_utc_ns=time_utc_ns,
        age_seconds=age_seconds,
    )


def _settings(**overrides: object) -> EngineSettings:
    """建立固定一秒步長的測試設定，並讓個別案例覆蓋輸出或停止條件。"""

    values: dict[str, object] = {
        "dt_min_seconds": 1.0,
        "dt_max_seconds": 1.0,
        "output_interval_seconds": 10.0,
        "max_backtrack_seconds": 5.0,
        "maximum_step_count": 100,
        "earliest_forcing_time_utc_ns": 0,
    }
    values.update(overrides)
    return EngineSettings(**values)


def _boundaries() -> BoundaryGeometry:
    """建立不會在一般 context 測試中提前停止的公尺制研究範圍。"""

    return BoundaryGeometry(
        own_local_domain=box(-100.0, -100.0, 100.0, 100.0),
        flow_domain=box(-1_000.0, -1_000.0, 1_000.0, 1_000.0),
        foreign_local_domains={},
    )


def _sample(
    *,
    qc: SampleQC = SampleQC.OK,
    eta_m: float = 0.0,
    bed_z_m: float = -100.0,
    forcing_month_id: str | None = "202401",
    u_mps: float = 0.0,
    w_mps: float = 0.0,
) -> VelocitySample:
    """建立測試用速度樣本；所有高度採海面向上為正的公尺座標。"""

    return VelocitySample(
        u_mps=u_mps,
        v_mps=0.0,
        w_mps=w_mps,
        eta_m=eta_m,
        bed_z_m=bed_z_m,
        horizontal_scale_m=100.0,
        vertical_scale_m=10.0,
        qc=qc,
        forcing_month_id=forcing_month_id,
    )


def _observation_with_context(
    state: ParticleState,
    *,
    status: ParticleStatus = ParticleStatus.ACTIVE,
) -> Observation:
    """建立與 state 完全同 identity 的 valid observation，供 status-only 測試使用。"""

    return Observation(
        particle_id=state.particle_id,
        time_utc_ns=state.time_utc_ns,
        age_seconds=state.age_seconds,
        x_m=state.x_m,
        y_m=state.y_m,
        z_m=state.z_m,
        status=status,
        environment_sample_status=EnvironmentSampleStatus.VALID,
        eta_m=0.0,
        bed_z_m=-100.0,
        forcing_month_id="202401",
        environment_qc_flags=0,
    )


def test_environment_sample_status_is_public_and_observation_has_three_states() -> None:
    """公開 enum 與 not-sampled、valid、invalid 三種 context 狀態必須可明確區分。"""

    assert EnvironmentSampleStatus.NOT_SAMPLED.value == "not_sampled"
    assert EnvironmentSampleStatus.VALID.value == "valid"
    assert EnvironmentSampleStatus.INVALID.value == "invalid"

    state = _state()
    not_sampled = Observation(
        state.particle_id,
        state.time_utc_ns,
        state.age_seconds,
        state.x_m,
        state.y_m,
        state.z_m,
        ParticleStatus.ACTIVE,
    )
    assert not_sampled.environment_sample_status is EnvironmentSampleStatus.NOT_SAMPLED
    assert (not_sampled.eta_m, not_sampled.bed_z_m, not_sampled.forcing_month_id) == (None, None, None)
    assert not_sampled.environment_qc_flags is None

    valid = _observation_with_context(state)
    assert valid.environment_sample_status is EnvironmentSampleStatus.VALID
    assert type(valid.eta_m) is float
    assert type(valid.bed_z_m) is float
    assert type(valid.environment_qc_flags) is int

    invalid = replace(
        valid,
        environment_sample_status=EnvironmentSampleStatus.INVALID,
        eta_m=None,
        bed_z_m=-20.0,
        forcing_month_id=None,
        environment_qc_flags=4,
    )
    assert invalid.environment_sample_status is EnvironmentSampleStatus.INVALID
    assert invalid.eta_m is None
    assert invalid.bed_z_m == -20.0
    assert invalid.environment_qc_flags == 4


@pytest.mark.parametrize(
    "changes",
    (
        {"environment_sample_status": "valid"},
        {"environment_sample_status": EnvironmentSampleStatus.NOT_SAMPLED, "eta_m": 0.0},
        {
            "environment_sample_status": EnvironmentSampleStatus.VALID,
            "eta_m": np.nan,
        },
        {
            "environment_sample_status": EnvironmentSampleStatus.VALID,
            "eta_m": True,
        },
        {
            "environment_sample_status": EnvironmentSampleStatus.VALID,
            "forcing_month_id": "202413",
        },
        {
            "environment_sample_status": EnvironmentSampleStatus.VALID,
            "environment_qc_flags": True,
        },
        {
            "environment_sample_status": EnvironmentSampleStatus.VALID,
            "eta_m": -20.0,
        },
        {
            "environment_sample_status": EnvironmentSampleStatus.INVALID,
            "environment_qc_flags": 0,
        },
        {
            "environment_sample_status": EnvironmentSampleStatus.INVALID,
            "environment_qc_flags": np.int64(1),
        },
        {
            "environment_sample_status": EnvironmentSampleStatus.INVALID,
            "eta_m": np.inf,
            "environment_qc_flags": 1,
        },
        {
            "environment_sample_status": EnvironmentSampleStatus.INVALID,
            "forcing_month_id": "202400",
            "environment_qc_flags": 1,
        },
    ),
)
def test_observation_rejects_unknown_status_malformed_context_and_nonfinite_values(
    changes: dict[str, object],
) -> None:
    """constructor 必須拒絕未知狀態、缺值政策違反、bool、非有限值與非法月份。"""

    state = _state()
    kwargs: dict[str, object] = {
        "particle_id": state.particle_id,
        "time_utc_ns": state.time_utc_ns,
        "age_seconds": state.age_seconds,
        "x_m": state.x_m,
        "y_m": state.y_m,
        "z_m": state.z_m,
        "status": ParticleStatus.ACTIVE,
        "environment_sample_status": EnvironmentSampleStatus.VALID,
        "eta_m": 0.0,
        "bed_z_m": -100.0,
        "forcing_month_id": "202401",
        "environment_qc_flags": 0,
    }
    kwargs.update(changes)
    with pytest.raises((TypeError, ValueError)):
        Observation(**kwargs)


def test_valid_step_start_enriches_existing_observation_without_extra_sample_or_observation() -> None:
    """合法步首 sample 只 enrichment 初始 observation，且不改變數值步的五次取樣契約。"""

    calls = 0

    def velocity(*_args: object) -> VelocitySample:
        """回傳具有完整月份 provenance 的 synthetic 樣本；不代表真實海洋資料。"""

        nonlocal calls
        calls += 1
        return _sample()

    execution = initialize_particle_execution(_state(), _settings(),)
    rng = np.random.Generator(np.random.PCG64DXSM(20260829))
    result = advance_particle_once(
        execution,
        velocity=velocity,
        boundaries=_boundaries(),
        behavior_class="sinking",
        diffusion=DiffusionCoefficients(0.0, 0.0, 0.0),
        settings=_settings(),
        rng=rng,
    )

    assert result.stepped and not result.terminal
    assert calls == 5  # 步首 reference 加上 RK4 的 k1--k4；enrichment 不增加取樣。
    assert len(execution.observations) == 1
    observation = execution.observations[0]
    assert observation.environment_sample_status is EnvironmentSampleStatus.VALID
    assert (observation.eta_m, observation.bed_z_m, observation.forcing_month_id) == (
        0.0,
        -100.0,
        "202401",
    )
    assert observation.environment_qc_flags == 0
    assert execution.observations[0].time_utc_ns != execution.state.time_utc_ns


def test_invalid_step_start_terminal_keeps_context_when_latest_observation_matches() -> None:
    """已有 matching observation 時，invalid terminal 只 replace status 並保留失敗 context。"""

    def velocity(*_args: object) -> VelocitySample:
        """回傳時間缺口樣本；缺值只作為 synthetic 工程測試輸入。"""

        return _sample(
            qc=SampleQC.TIME_GAP,
            eta_m=np.nan,
            bed_z_m=-20.0,
            forcing_month_id=None,
        )

    execution = initialize_particle_execution(_state(), _settings(),)
    result = advance_particle_once(
        execution,
        velocity=velocity,
        boundaries=_boundaries(),
        behavior_class="sinking",
        diffusion=DiffusionCoefficients(0.0, 0.0, 0.0),
        settings=_settings(),
        rng=np.random.Generator(np.random.PCG64DXSM(1)),
    )

    assert result.terminal and not result.stepped
    assert len(execution.observations) == 1
    observation = execution.observations[-1]
    assert observation.status is ParticleStatus.DATA_GAP
    assert observation.environment_sample_status is EnvironmentSampleStatus.INVALID
    assert observation.eta_m is None
    assert observation.bed_z_m == -20.0
    assert observation.forcing_month_id is None
    assert observation.environment_qc_flags == int(SampleQC.TIME_GAP)


def test_invalid_step_start_after_unoutput_step_appends_only_contextual_terminal_observation() -> None:
    """先完成未達固定輸出點的步驟，再失敗時 terminal 必須帶同一次 invalid reference context。"""

    calls = 0

    def velocity(*_args: object) -> VelocitySample:
        """前五次供第一個 RK4 步驟，下一次在真正 step-start 產生資料缺口。"""

        nonlocal calls
        calls += 1
        if calls <= 5:
            return _sample()
        return _sample(
            qc=SampleQC.TIME_GAP,
            eta_m=0.25,
            bed_z_m=-50.0,
            forcing_month_id="202402",
        )

    settings = _settings(output_interval_seconds=10.0)
    execution = initialize_particle_execution(_state(), settings)
    rng = np.random.Generator(np.random.PCG64DXSM(2))
    first = advance_particle_once(
        execution,
        velocity=velocity,
        boundaries=_boundaries(),
        behavior_class="sinking",
        diffusion=DiffusionCoefficients(0.0, 0.0, 0.0),
        settings=settings,
        rng=rng,
    )
    assert first.stepped and not first.terminal
    assert len(execution.observations) == 1  # 第一個數值步未達固定輸出 age。

    second = advance_particle_once(
        execution,
        velocity=velocity,
        boundaries=_boundaries(),
        behavior_class="sinking",
        diffusion=DiffusionCoefficients(0.0, 0.0, 0.0),
        settings=settings,
        rng=rng,
    )

    assert second.terminal and not second.stepped
    assert calls == 6  # 第二次只有 step-start reference，沒有進入 RK4。
    assert len(execution.observations) == 2
    assert execution.observations[0].environment_sample_status is EnvironmentSampleStatus.VALID
    terminal = execution.observations[-1]
    assert terminal.status is ParticleStatus.DATA_GAP
    assert terminal.environment_sample_status is EnvironmentSampleStatus.INVALID
    assert (terminal.eta_m, terminal.bed_z_m, terminal.forcing_month_id) == (
        0.25,
        -50.0,
        "202402",
    )
    assert terminal.environment_qc_flags == int(SampleQC.TIME_GAP)


def test_valid_synthetic_month_none_keeps_existing_context_and_new_observation_unsampled() -> None:
    """沒有月份 provenance 的 synthetic sample 不得猜測月份或宣稱 VALID。"""

    def velocity(*_args: object) -> VelocitySample:
        """回傳沒有 forcing 月份的 synthetic callback，僅供資料流測試。"""

        return _sample(forcing_month_id=None)

    settings = _settings()
    new_execution = initialize_particle_execution(_state(), settings)
    advance_particle_once(
        new_execution,
        velocity=velocity,
        boundaries=_boundaries(),
        behavior_class="sinking",
        diffusion=DiffusionCoefficients(0.0, 0.0, 0.0),
        settings=settings,
        rng=np.random.Generator(np.random.PCG64DXSM(3)),
    )
    assert new_execution.observations[0].environment_sample_status is EnvironmentSampleStatus.NOT_SAMPLED

    preserved_execution = initialize_particle_execution(_state(), settings)
    existing = _observation_with_context(_state())
    preserved_execution.observations[0] = existing
    advance_particle_once(
        preserved_execution,
        velocity=velocity,
        boundaries=_boundaries(),
        behavior_class="sinking",
        diffusion=DiffusionCoefficients(0.0, 0.0, 0.0),
        settings=settings,
        rng=np.random.Generator(np.random.PCG64DXSM(4)),
    )
    assert preserved_execution.observations == [existing]


@pytest.mark.parametrize(
    "sample",
    (
        _sample(forcing_month_id="202413"),
        _sample(eta_m=np.nan),
        _sample(bed_z_m=np.inf),
    ),
)
def test_valid_reference_with_unusable_environment_metadata_fails_closed(sample: VelocitySample) -> None:
    """月份或環境幾何不可稽核時，必須拒絕執行而不寫入看似有效的 context。"""

    execution = initialize_particle_execution(_state(), _settings())

    def velocity(*_args: object) -> VelocitySample:
        """提供 metadata 故意損壞的 synthetic 樣本，確認不會被當成正式證據。"""

        return sample

    with pytest.raises((TypeError, ValueError)):
        advance_particle_once(
            execution,
            velocity=velocity,
            boundaries=_boundaries(),
            behavior_class="sinking",
            diffusion=DiffusionCoefficients(0.0, 0.0, 0.0),
            settings=_settings(),
            rng=np.random.Generator(np.random.PCG64DXSM(5)),
        )
    assert execution.observations[0].environment_sample_status is EnvironmentSampleStatus.NOT_SAMPLED


def test_status_only_replace_and_finalize_preserve_matching_context() -> None:
    """max-age status replacement 與 finalize 不得因同 state 改狀態而清除 context。"""

    state = _state(age_seconds=5.0)
    settings = _settings(max_backtrack_seconds=5.0)
    execution = initialize_particle_execution(state, settings)
    existing = _observation_with_context(state)
    execution.observations[0] = existing

    result = advance_particle_once(
        execution,
        velocity=lambda *_args: pytest.fail("status-only stop 不應呼叫 velocity"),
        boundaries=_boundaries(),
        behavior_class="sinking",
        diffusion=DiffusionCoefficients(0.0, 0.0, 0.0),
        settings=settings,
        rng=np.random.Generator(np.random.PCG64DXSM(6)),
    )
    assert result.terminal and not result.stepped
    assert len(execution.observations) == 1
    assert execution.observations[0].status is ParticleStatus.MAX_AGE
    assert execution.observations[0].environment_sample_status is EnvironmentSampleStatus.VALID

    execution.state = replace(execution.state, status=ParticleStatus.FORCING_START)
    finalized = finalize_particle_execution(execution)
    assert len(finalized.observations) == 1
    assert finalized.observations[0].status is ParticleStatus.FORCING_START
    assert finalized.observations[0].environment_sample_status is EnvironmentSampleStatus.VALID


def test_boundary_terminal_does_not_inherit_step_start_context_from_different_state() -> None:
    """步首 VALID context 不得錯掛到位置／時間已改變的 boundary terminal observation。"""

    def velocity(*_args: object) -> VelocitySample:
        """以無擴散向西回溯穿越 flow boundary 的 synthetic 速度。"""

        return _sample(u_mps=1.0)

    boundaries = BoundaryGeometry(
        own_local_domain=box(-0.5, -100.0, 100.0, 100.0),
        flow_domain=box(-0.5, -100.0, 100.0, 100.0),
        foreign_local_domains={},
        local_equals_flow=True,
    )
    settings = _settings(max_backtrack_seconds=5.0, output_interval_seconds=10.0)
    execution = initialize_particle_execution(_state(), settings)
    result = advance_particle_once(
        execution,
        velocity=velocity,
        boundaries=boundaries,
        behavior_class="sinking",
        diffusion=DiffusionCoefficients(0.0, 0.0, 0.0),
        settings=settings,
        rng=np.random.Generator(np.random.PCG64DXSM(7)),
    )

    assert result.terminal and result.stepped
    assert execution.observations[0].environment_sample_status is EnvironmentSampleStatus.VALID
    terminal = execution.observations[-1]
    assert terminal.status is ParticleStatus.FLOW_DOMAIN_EXIT
    assert terminal.environment_sample_status is EnvironmentSampleStatus.NOT_SAMPLED
    assert len(execution.observations) == 2


def test_direct_and_stepwise_execution_match_with_exact_sample_and_rng_cadence() -> None:
    """加入 context 後 direct／stepwise 結果、速度取樣次數與 RNG 消耗仍完全一致。"""

    settings = _settings(output_interval_seconds=2.0, max_backtrack_seconds=2.0)
    direct_calls = 0
    stepwise_calls = 0

    def direct_velocity(*_args: object) -> VelocitySample:
        """direct runner 的 deterministic synthetic callback。"""

        nonlocal direct_calls
        direct_calls += 1
        return _sample()

    def stepwise_velocity(*_args: object) -> VelocitySample:
        """stepwise runner 的同值 callback，用於隔離 call count 而非科學資料。"""

        nonlocal stepwise_calls
        stepwise_calls += 1
        return _sample()

    direct_rng = np.random.Generator(np.random.PCG64DXSM(8))
    direct = run_particle(
        _state(),
        velocity=direct_velocity,
        boundaries=_boundaries(),
        behavior_class="sinking",
        diffusion=DiffusionCoefficients(0.1, 0.05, 0.0),
        settings=settings,
        rng=direct_rng,
    )

    stepwise_rng = np.random.Generator(np.random.PCG64DXSM(8))
    execution = initialize_particle_execution(_state(), settings)
    while not advance_particle_once(
        execution,
        velocity=stepwise_velocity,
        boundaries=_boundaries(),
        behavior_class="sinking",
        diffusion=DiffusionCoefficients(0.1, 0.05, 0.0),
        settings=settings,
        rng=stepwise_rng,
    ):
        pass
    stepwise = finalize_particle_execution(execution)

    assert stepwise == direct
    assert direct_calls == direct.step_count * 5
    assert stepwise_calls == stepwise.step_count * 5
    assert direct_rng.bit_generator.state == stepwise_rng.bit_generator.state
