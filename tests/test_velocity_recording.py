"""逐點總速度與分項速度紀錄的核心資料流測試。

本檔只使用小型 synthetic provider，驗證速度欄位的物理方向、同點綁定、缺值狀態與
終止分支；這些 fixture 不讀取 OCM／NWW3 正式產品，也不代表正式來源足跡結果。速度
紀錄只應來自每個 engine step 的步首 reference，故測試同時檢查 RK4 失敗或擴散失敗
時不會把中間 stage 速度誤寫入 terminal observation，也不會因紀錄而多查詢或多消耗 RNG。
"""

from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest
from shapely.geometry import box

from lagrangian_backtracking.boundaries import BoundaryGeometry
from lagrangian_backtracking.diffusion import DiffusionCoefficients, DiffusionModel, DiffusionSample
from lagrangian_backtracking.engine import (
    EngineSettings,
    Observation,
    advance_particle_once,
    initialize_particle_execution,
    run_particle,
)
from lagrangian_backtracking.models import (
    EventType,
    ParticleState,
    ParticleStatus,
    SampleQC,
    VelocityComponents,
    VelocityQC,
    VelocitySample,
    VelocitySampleStatus,
)


def _state(*, age_seconds: float = 0.0) -> ParticleState:
    """建立使用公尺座標與 UTC 奈秒、遠離一般邊界的 active 粒子。"""

    return ParticleState(
        particle_id="velocity-p0",
        scenario_id="velocity-s0",
        member_id=0,
        study_site_id="gongliao",
        analysis_region_id="A",
        receptor_id="r0",
        x_m=0.0,
        y_m=0.0,
        z_m=-5.0,
        time_utc_ns=100_000_000_000,
        age_seconds=age_seconds,
    )


def _settings(**overrides: object) -> EngineSettings:
    """建立一秒步長、兩秒輸出 cadence 的測試設定。"""

    values: dict[str, object] = {
        "dt_min_seconds": 1.0,
        "dt_max_seconds": 1.0,
        "output_interval_seconds": 2.0,
        "max_backtrack_seconds": 5.0,
        "maximum_step_count": 100,
        "earliest_forcing_time_utc_ns": 0,
    }
    values.update(overrides)
    return EngineSettings(**values)


def _boundaries() -> BoundaryGeometry:
    """建立不會干擾速度紀錄測試的公尺制 local／flow domain。"""

    return BoundaryGeometry(
        own_local_domain=box(-100.0, -100.0, 100.0, 100.0),
        flow_domain=box(-1_000.0, -1_000.0, 1_000.0, 1_000.0),
        foreign_local_domains={},
    )


def _total_sample(
    *,
    u_mps: object = 0.0,
    v_mps: object = 0.0,
    w_mps: object = 0.0,
    qc: SampleQC = SampleQC.OK,
    forcing_month_id: str | None = "202401",
) -> VelocitySample:
    """建立沒有來源分項的 callback sample，供 total-only 或失敗輸入測試使用。"""

    return VelocitySample(
        u_mps=u_mps,  # type: ignore[arg-type]
        v_mps=v_mps,  # type: ignore[arg-type]
        w_mps=w_mps,  # type: ignore[arg-type]
        eta_m=0.0,
        bed_z_m=-20.0,
        horizontal_scale_m=100.0,
        vertical_scale_m=10.0,
        qc=qc,
        forcing_month_id=forcing_month_id,
    )


def _complete_sample(
    *,
    ocm: tuple[float, float, float] = (0.8, -0.4, 0.0),
    stokes: tuple[float, float] = (0.1, 0.05),
    settling_w_mps: float = -0.2,
    forcing_month_id: str | None = "202401",
) -> VelocitySample:
    """建立同一次查詢可核對合成公式的完整 OCM／Stokes／沉降 sample。"""

    total = (ocm[0] + stokes[0], ocm[1] + stokes[1], ocm[2] + settling_w_mps)
    components = VelocityComponents(
        total_u_mps=total[0],
        total_v_mps=total[1],
        total_w_mps=total[2],
        ocm_u_mps=ocm[0],
        ocm_v_mps=ocm[1],
        ocm_w_mps=ocm[2],
        stokes_u_mps=stokes[0],
        stokes_v_mps=stokes[1],
        settling_w_mps=settling_w_mps,
    )
    return VelocitySample(
        u_mps=total[0],
        v_mps=total[1],
        w_mps=total[2],
        eta_m=0.0,
        bed_z_m=-20.0,
        horizontal_scale_m=100.0,
        vertical_scale_m=10.0,
        forcing_month_id=forcing_month_id,
        components=components,
    )


def _velocity_fields(observation: Observation) -> tuple[float | None, ...]:
    """依固定資料契約順序取出 Observation 的九個速度欄位。"""

    return tuple(
        getattr(observation, field_name)
        for field_name in (
            "total_u_mps",
            "total_v_mps",
            "total_w_mps",
            "ocm_u_mps",
            "ocm_v_mps",
            "ocm_w_mps",
            "stokes_u_mps",
            "stokes_v_mps",
            "settling_w_mps",
        )
    )


def _advance(
    execution,
    *,
    velocity,
    settings: EngineSettings,
    diffusion: DiffusionModel | None = None,
    rng: np.random.Generator,
):
    """以固定 boundary 呼叫單步核心，集中保留測試的既有 engine 入口。"""

    if diffusion is None:
        diffusion = DiffusionCoefficients(0.0, 0.0, 0.0)
    return advance_particle_once(
        execution,
        velocity=velocity,
        boundaries=_boundaries(),
        behavior_class="sinking",
        diffusion=diffusion,
        settings=settings,
        rng=rng,
    )


def test_complete_reference_is_bound_to_initial_observation_without_average_or_reverse_sign() -> None:
    """完整 reference 應在同點初始觀測保存，逆向位置則使用物理正向速度的相反時間。"""

    sample = _complete_sample()
    calls: list[tuple[float, float, float, int]] = []

    def velocity(x_m: float, y_m: float, z_m: float, time_utc_ns: int) -> VelocitySample:
        """記錄每次 callback 引數；回傳值始終是同一個完整 synthetic sample。"""

        calls.append((x_m, y_m, z_m, time_utc_ns))
        return sample

    settings = _settings(output_interval_seconds=10.0, max_backtrack_seconds=1.0)
    execution = initialize_particle_execution(_state(), settings)
    result = _advance(
        execution,
        velocity=velocity,
        settings=settings,
        rng=np.random.Generator(np.random.PCG64DXSM(20260905)),
    )

    assert result.stepped and not result.terminal
    assert len(calls) == 5  # 步首 reference 加上原本 RK4 的 k1--k4，紀錄不新增查詢。
    observation = execution.observations[0]
    assert observation.velocity_sample_status is VelocitySampleStatus.COMPLETE
    assert observation.velocity_qc_flags == 0
    assert _velocity_fields(observation) == (
        0.9,
        -0.35000000000000003,
        -0.2,
        0.8,
        -0.4,
        0.0,
        0.1,
        0.05,
        -0.2,
    )
    # 速度欄位仍是物理時間向前的正向值；回溯只把 dt 設為負，故 x 向負向移動。
    assert execution.state.x_m == pytest.approx(-0.9)
    assert execution.state.y_m == pytest.approx(0.35)
    assert execution.state.z_m == pytest.approx(-4.8)
    assert execution.observations[0].time_utc_ns == _state().time_utc_ns


def test_total_only_callback_is_explicit_and_does_not_fabricate_ocm_or_stokes() -> None:
    """只有總速度的 callback 必須標為 total-only，六個來源分項維持缺值。"""

    sample = _total_sample(u_mps=0.3, v_mps=-0.1, w_mps=-0.2)
    execution = initialize_particle_execution(_state(), _settings())
    _advance(
        execution,
        velocity=lambda *_args: sample,
        settings=_settings(),
        rng=np.random.Generator(np.random.PCG64DXSM(20260906)),
    )

    observation = execution.observations[0]
    assert observation.velocity_sample_status is VelocitySampleStatus.TOTAL_ONLY
    assert _velocity_fields(observation) == (0.3, -0.1, -0.2, None, None, None, None, None, None)
    assert observation.velocity_qc_flags == 0


@pytest.mark.parametrize(
    ("bad_value", "expected_status", "expected_qc"),
    (
        (None, VelocitySampleStatus.MISSING, VelocityQC.MISSING_COMPONENT),
        (np.nan, VelocitySampleStatus.NONFINITE, VelocityQC.NONFINITE),
        (True, VelocitySampleStatus.INVALID, VelocityQC.NON_NUMERIC),
    ),
)
def test_bad_total_values_are_none_with_independent_velocity_qc_before_diffusion_or_rng(
    bad_value: object,
    expected_status: VelocitySampleStatus,
    expected_qc: VelocityQC,
) -> None:
    """總速度缺值、NaN 與 bool 必須各自降級，且不能進入擴散或消耗 RNG。"""

    calls = 0

    def velocity(*_args: object) -> VelocitySample:
        """每次只供應故意損壞的步首 sample，確認引擎不會重試或補值。"""

        nonlocal calls
        calls += 1
        return _total_sample(u_mps=bad_value)

    execution = initialize_particle_execution(_state(), _settings())
    rng = np.random.Generator(np.random.PCG64DXSM(20260907))
    state_before = execution.state
    rng_before = copy.deepcopy(rng.bit_generator.state)
    result = _advance(execution, velocity=velocity, settings=_settings(), rng=rng)

    assert result.terminal and not result.stepped
    assert calls == 1
    assert execution.state == replace(state_before, status=ParticleStatus.NUMERICAL_FAILURE)
    assert rng.bit_generator.state == rng_before
    observation = execution.observations[-1]
    assert observation.velocity_sample_status is expected_status
    assert observation.velocity_qc_flags == int(expected_qc)
    assert observation.total_u_mps is None
    assert observation.total_v_mps == 0.0
    assert observation.total_w_mps == 0.0
    assert all(value is None for value in _velocity_fields(observation)[3:])


def test_typed_component_sum_mismatch_is_recorded_but_does_not_change_total_physics() -> None:
    """分項總和不符時保存非零 QC，但既有有限 total 的數值步仍維持原本物理結果。"""

    valid = _complete_sample()
    assert valid.components is not None
    mismatched = replace(
        valid,
        components=replace(valid.components, stokes_u_mps=0.5),
    )
    calls = 0

    def velocity(*_args: object) -> VelocitySample:
        """回傳同一個故意 sum mismatch 的完整 payload，隔離紀錄驗證與積分公式。"""

        nonlocal calls
        calls += 1
        return mismatched

    execution = initialize_particle_execution(_state(), _settings())
    result = _advance(
        execution,
        velocity=velocity,
        settings=_settings(),
        rng=np.random.Generator(np.random.PCG64DXSM(20260908)),
    )

    assert result.stepped and not result.terminal
    assert calls == 5
    observation = execution.observations[0]
    assert observation.velocity_sample_status is VelocitySampleStatus.SUM_MISMATCH
    assert observation.velocity_qc_flags == int(VelocityQC.SUM_MISMATCH)
    assert observation.total_u_mps == pytest.approx(0.9)
    assert observation.stokes_u_mps == pytest.approx(0.5)
    # 積分仍只讀取 VelocitySample 的既有總量，不從分項重新合成。
    assert execution.state.x_m == pytest.approx(-0.9)


def test_reverse_run_keeps_physical_forward_velocity_sign() -> None:
    """完整回溯結果的輸出速度不應因逆向積分被另行取負。"""

    sample = _complete_sample(ocm=(1.0, 0.0, 0.0), stokes=(0.0, 0.0), settling_w_mps=0.0)
    calls = 0

    def velocity(*_args: object) -> VelocitySample:
        """提供正東向的物理 forward current；引擎本身負責逆向時間方向。"""

        nonlocal calls
        calls += 1
        return sample

    result = run_particle(
        _state(),
        velocity=velocity,
        boundaries=_boundaries(),
        behavior_class="sinking",
        diffusion=DiffusionCoefficients(0.0, 0.0, 0.0),
        settings=_settings(output_interval_seconds=10.0, max_backtrack_seconds=1.0),
        rng=np.random.Generator(np.random.PCG64DXSM(20260909)),
    )

    assert calls == 5
    assert result.final_state.x_m == pytest.approx(-1.0)
    assert result.observations[0].total_u_mps == pytest.approx(1.0)
    assert result.observations[0].velocity_sample_status is VelocitySampleStatus.COMPLETE
    assert result.observations[-1].status is ParticleStatus.MAX_AGE
    assert result.observations[-1].velocity_sample_status is VelocitySampleStatus.NOT_SAMPLED


def test_stage_failure_on_nonoutput_step_uses_only_step_start_reference_velocity() -> None:
    """cadence=2 的第二步 RK4 stage failure 應保留步首 reference，不借用失敗 stage。"""

    reference = _complete_sample(ocm=(0.6, 0.2, 0.0), stokes=(0.05, -0.02), settling_w_mps=-0.1)
    invalid_stage = _total_sample(qc=SampleQC.TIME_GAP)
    calls: list[tuple[float, float, float, int]] = []

    def velocity(x_m: float, y_m: float, z_m: float, time_utc_ns: int) -> VelocitySample:
        """前一步完整成功；第二步在 k2 失敗，並記錄所有真實 callback 次數。"""

        calls.append((x_m, y_m, z_m, time_utc_ns))
        return invalid_stage if len(calls) == 8 else reference

    settings = _settings(output_interval_seconds=2.0)
    execution = initialize_particle_execution(_state(), settings)
    rng = np.random.Generator(np.random.PCG64DXSM(20260910))
    first = _advance(execution, velocity=velocity, settings=settings, rng=rng)
    assert first.stepped and not first.terminal
    assert len(execution.observations) == 1  # age=1 尚未到 cadence=2。
    state_before_failure = execution.state
    rng_before_failure = copy.deepcopy(rng.bit_generator.state)
    calls_before_failure = len(calls)

    second = _advance(execution, velocity=velocity, settings=settings, rng=rng)

    assert second.terminal and not second.stepped
    assert execution.state == replace(state_before_failure, status=ParticleStatus.DATA_GAP)
    assert len(calls) == calls_before_failure + 3  # reference、k1、k2；沒有紀錄用額外查詢。
    assert calls[calls_before_failure] == (
        state_before_failure.x_m,
        state_before_failure.y_m,
        state_before_failure.z_m,
        state_before_failure.time_utc_ns,
    )
    assert rng.bit_generator.state == rng_before_failure  # 失敗在 Brownian 前，不消耗 RNG。
    terminal = execution.observations[-1]
    assert (terminal.time_utc_ns, terminal.age_seconds) == (
        state_before_failure.time_utc_ns,
        state_before_failure.age_seconds,
    )
    assert (terminal.x_m, terminal.y_m, terminal.z_m) == (
        state_before_failure.x_m,
        state_before_failure.y_m,
        state_before_failure.z_m,
    )
    assert terminal.status is ParticleStatus.DATA_GAP
    assert terminal.velocity_sample_status is VelocitySampleStatus.COMPLETE
    assert terminal.velocity_qc_flags == 0
    assert _velocity_fields(terminal) == _velocity_fields(execution.observations[0])
    assert execution.events[-1].event_type is EventType.DATA_GAP


class _FailingDiffusionProvider:
    """在步首回傳無效擴散或拋出例外的 synthetic provider，完全不產生 RNG。"""

    def __init__(self, mode: str) -> None:
        """設定唯一故障模式；兩種模式都只應被 engine 呼叫一次。"""

        self.mode = mode
        self.calls = 0

    def sample(
        self,
        _x_m: float,
        _y_m: float,
        _z_m: float,
        _time_utc_ns: int,
        *,
        triangle_hint: int | None = None,
    ) -> DiffusionSample:
        """回傳非零 QC 樣本或在 choose-time-step 前拋出 ValueError。"""

        del triangle_hint
        self.calls += 1
        if self.mode == "sample_invalid":
            return DiffusionSample(
                DiffusionCoefficients(0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                qc=SampleQC.TIME_GAP,
            )
        raise ValueError("synthetic diffusion evaluation failure")


@pytest.mark.parametrize(
    ("mode", "expected_status"),
    (
        ("sample_invalid", ParticleStatus.DATA_GAP),
        ("exception", ParticleStatus.NUMERICAL_FAILURE),
    ),
)
def test_diffusion_failure_on_nonoutput_step_keeps_reference_velocity_without_extra_call_or_rng(
    mode: str,
    expected_status: ParticleStatus,
) -> None:
    """擴散樣本 invalid／exception 均在同點 terminal 保存 reference，且不進 RK4。"""

    reference = _complete_sample(ocm=(0.4, -0.2, 0.0), stokes=(0.02, 0.03), settling_w_mps=-0.15)
    calls = 0

    def velocity(*_args: object) -> VelocitySample:
        """先完成一個未輸出的步，再供應第二步唯一的步首 reference。"""

        nonlocal calls
        calls += 1
        return reference

    settings = _settings(output_interval_seconds=2.0)
    execution = initialize_particle_execution(_state(), settings)
    rng = np.random.Generator(np.random.PCG64DXSM(20260911))
    first = _advance(execution, velocity=velocity, settings=settings, rng=rng)
    assert first.stepped and not first.terminal
    state_before_failure = execution.state
    rng_before_failure = copy.deepcopy(rng.bit_generator.state)
    calls_before_failure = calls
    diffusion_provider = _FailingDiffusionProvider(mode)

    second = _advance(
        execution,
        velocity=velocity,
        settings=settings,
        diffusion=diffusion_provider,
        rng=rng,
    )

    assert second.terminal and not second.stepped
    assert second.state == replace(state_before_failure, status=expected_status)
    assert calls == calls_before_failure + 1  # reference only；沒有 RK4 或紀錄額外查詢。
    assert diffusion_provider.calls == 1
    assert rng.bit_generator.state == rng_before_failure
    terminal = execution.observations[-1]
    assert terminal.status is expected_status
    assert terminal.velocity_sample_status is VelocitySampleStatus.COMPLETE
    assert terminal.velocity_qc_flags == 0
    assert _velocity_fields(terminal) == (
        reference.components.total_u_mps,
        reference.components.total_v_mps,
        reference.components.total_w_mps,
        reference.components.ocm_u_mps,
        reference.components.ocm_v_mps,
        reference.components.ocm_w_mps,
        reference.components.stokes_u_mps,
        reference.components.stokes_v_mps,
        reference.components.settling_w_mps,
    )


def test_status_only_max_age_without_reference_remains_not_sampled() -> None:
    """沒有新步首 sample 的 max-age status-only 終點不得借用任何速度資料。"""

    settings = _settings(max_backtrack_seconds=2.0)
    execution = initialize_particle_execution(_state(age_seconds=2.0), settings)
    rng = np.random.Generator(np.random.PCG64DXSM(20260912))
    rng_before = copy.deepcopy(rng.bit_generator.state)

    result = _advance(
        execution,
        velocity=lambda *_args: pytest.fail("max_age 前不應呼叫 velocity"),
        settings=settings,
        rng=rng,
    )

    assert result.terminal and not result.stepped
    assert rng.bit_generator.state == rng_before
    observation = execution.observations[-1]
    assert observation.status is ParticleStatus.MAX_AGE
    assert observation.velocity_sample_status is VelocitySampleStatus.NOT_SAMPLED
    assert _velocity_fields(observation) == (None,) * 9
    assert observation.velocity_qc_flags is None


def test_observation_allows_explicit_missing_diagnostics_but_rejects_nan_and_bool() -> None:
    """模型層接受明示 None 缺分項，並在輸出邊界拒絕 NaN／bool 以免狀態被偽造。"""

    state = _state()
    missing = Observation(
        particle_id=state.particle_id,
        time_utc_ns=state.time_utc_ns,
        age_seconds=state.age_seconds,
        x_m=state.x_m,
        y_m=state.y_m,
        z_m=state.z_m,
        status=ParticleStatus.ACTIVE,
        velocity_sample_status=VelocitySampleStatus.MISSING,
        total_u_mps=None,
        total_v_mps=0.0,
        total_w_mps=0.0,
        velocity_qc_flags=int(VelocityQC.MISSING_COMPONENT),
    )
    assert missing.total_u_mps is None
    assert missing.velocity_qc_flags == int(VelocityQC.MISSING_COMPONENT)

    for bad_value in (np.nan, True):
        with pytest.raises((TypeError, ValueError)):
            Observation(
                particle_id=state.particle_id,
                time_utc_ns=state.time_utc_ns,
                age_seconds=state.age_seconds,
                x_m=state.x_m,
                y_m=state.y_m,
                z_m=state.z_m,
                status=ParticleStatus.ACTIVE,
                velocity_sample_status=VelocitySampleStatus.TOTAL_ONLY,
                total_u_mps=bad_value,
                total_v_mps=0.0,
                total_w_mps=0.0,
                velocity_qc_flags=0,
            )
