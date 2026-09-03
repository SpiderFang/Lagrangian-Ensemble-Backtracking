"""signed-time RK4、浮沉與 Brownian 統計參考測試。"""

from __future__ import annotations

import numpy as np
import pytest

from lagrangian_backtracking.diffusion import (
    DiffusionCoefficients,
    DiffusionSample,
    SmagorinskySettings,
    brownian_displacement,
    choose_time_step,
    diffusion_displacement,
    smagorinsky_horizontal_diffusivity,
)
from lagrangian_backtracking.integrators import rk4_step, split_rk4_brownian_step
from lagrangian_backtracking.models import ParticleState, SampleQC, VelocitySample


def _state() -> ParticleState:
    """建立無邊界影響的合成 active particle。"""

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
        time_utc_ns=1_704_067_200_000_000_000,
    )


def test_backward_rk4_uses_negative_dt_once() -> None:
    """常流 backward 1 小時應沿物理速度反方向移動，z 沉降項自然反向。"""

    def velocity(x_m: float, y_m: float, z_m: float, time_utc_ns: int) -> VelocitySample:
        del x_m, y_m, z_m, time_utc_ns
        return VelocitySample(1.0, -0.5, -0.01, 0.0, -100.0, 1_000.0, 1.0)

    result = rk4_step(_state(), dt_seconds=-3_600.0, velocity=velocity)
    assert np.isclose(result.x_m, -3_600.0)
    assert np.isclose(result.y_m, 1_800.0)
    assert np.isclose(result.z_m, 26.0)
    assert result.age_seconds == 3_600.0


def test_rk4_calls_all_four_stages() -> None:
    """線性時間速度若每 stage 取樣，單步積分會精確得到解析時間積分。"""

    start_ns = _state().time_utc_ns
    calls: list[int] = []

    def velocity(x_m: float, y_m: float, z_m: float, time_utc_ns: int) -> VelocitySample:
        del x_m, y_m, z_m
        calls.append(time_utc_ns)
        elapsed = (time_utc_ns - start_ns) / 1_000_000_000
        return VelocitySample(elapsed, 0.0, 0.0, 0.0, -100.0, 1_000.0, 1.0)

    result = rk4_step(_state(), dt_seconds=10.0, velocity=velocity)
    assert len(calls) == 4
    assert np.isclose(result.x_m, 50.0)


def test_brownian_variance_matches_2kdt() -> None:
    """大量獨立實現的三軸變異應落在 2K|dt| 的統計容許範圍。"""

    rng = np.random.default_rng(20260819)
    coefficients = DiffusionCoefficients(4.0, 2.0, 0.01)
    samples = np.stack(
        [brownian_displacement(coefficients, dt_seconds=-60.0, rng=rng) for _ in range(80_000)]
    )
    expected = 2.0 * np.array([4.0, 2.0, 0.01]) * 60.0
    assert np.allclose(samples.mean(axis=0), 0.0, atol=np.sqrt(expected / samples.shape[0]) * 5.0)
    assert np.allclose(samples.var(axis=0), expected, rtol=0.02)


def test_choose_time_step_does_not_cross_pair_horizontal_k_with_vertical_scale() -> None:
    """大水平 Kh 搭配很小垂向尺度時，不能產生跨軸的過度小步長。"""

    decision = choose_time_step(
        speed_horizontal_mps=0.0,
        speed_vertical_mps=0.0,
        horizontal_scale_m=100.0,
        vertical_scale_m=0.1,
        coefficients=DiffusionCoefficients(100.0, 100.0, 0.0),
        dt_min_seconds=1.0e-6,
        dt_max_seconds=100.0,
    )
    # 水平公式為 (0.25*100 m)^2/(2*100 m²/s)=3.125 s；Kz=0 不建立垂向限制。
    assert decision.seconds == 3.125
    assert decision.limiting_reason == "horizontal_diffusion"


def test_choose_time_step_can_select_each_diffusion_axis_independently() -> None:
    """水平與垂向 diffusion candidate 應各自使用自己的尺度與係數。"""

    horizontal = choose_time_step(
        speed_horizontal_mps=0.0,
        speed_vertical_mps=0.0,
        horizontal_scale_m=8.0,
        vertical_scale_m=100.0,
        coefficients=DiffusionCoefficients(4.0, 4.0, 0.0),
        dt_min_seconds=1.0e-6,
        dt_max_seconds=100.0,
    )
    vertical = choose_time_step(
        speed_horizontal_mps=0.0,
        speed_vertical_mps=0.0,
        horizontal_scale_m=100.0,
        vertical_scale_m=4.0,
        coefficients=DiffusionCoefficients(0.0, 0.0, 1.0),
        dt_min_seconds=1.0e-6,
        dt_max_seconds=100.0,
    )
    assert horizontal == type(horizontal)(0.5, "horizontal_diffusion")
    assert vertical == type(vertical)(0.5, "vertical_diffusion")


def test_choose_time_step_keeps_zero_axis_and_boundary_clamp_policies() -> None:
    """某軸 K=0 只移除該軸限制，minimum clamp 與 forcing boundary 仍維持原政策。"""

    horizontal_only = choose_time_step(
        speed_horizontal_mps=0.0,
        speed_vertical_mps=0.0,
        horizontal_scale_m=8.0,
        vertical_scale_m=0.1,
        coefficients=DiffusionCoefficients(4.0, 4.0, 0.0),
        dt_min_seconds=1.0e-6,
        dt_max_seconds=100.0,
    )
    vertical_only = choose_time_step(
        speed_horizontal_mps=0.0,
        speed_vertical_mps=0.0,
        horizontal_scale_m=0.1,
        vertical_scale_m=8.0,
        coefficients=DiffusionCoefficients(0.0, 0.0, 1.0),
        dt_min_seconds=1.0e-6,
        dt_max_seconds=100.0,
    )
    clamped = choose_time_step(
        speed_horizontal_mps=0.0,
        speed_vertical_mps=0.0,
        horizontal_scale_m=1.0,
        vertical_scale_m=100.0,
        coefficients=DiffusionCoefficients(100.0, 100.0, 0.0),
        dt_min_seconds=0.1,
        dt_max_seconds=100.0,
    )
    boundary = choose_time_step(
        speed_horizontal_mps=0.0,
        speed_vertical_mps=0.0,
        horizontal_scale_m=100.0,
        vertical_scale_m=100.0,
        coefficients=DiffusionCoefficients(0.0, 0.0, 0.0),
        dt_min_seconds=1.0,
        dt_max_seconds=100.0,
        seconds_to_forcing_boundary=2.0,
    )
    assert horizontal_only.limiting_reason == "horizontal_diffusion"
    assert vertical_only.limiting_reason == "vertical_diffusion"
    assert clamped == type(clamped)(0.1, "minimum_clamp")
    assert boundary == type(boundary)(2.0, "forcing_boundary")


def test_constant_split_keeps_legacy_fixed_seed_displacement() -> None:
    """舊版常數係數 split 應逐位元等於直接 Brownian helper 的結果。"""

    coefficients = DiffusionCoefficients(4.0, 2.0, 0.01)
    state = _state()

    def velocity(x_m: float, y_m: float, z_m: float, time_utc_ns: int) -> VelocitySample:
        """提供零確定性速度，讓測試只比較擴散亂數的相容性。"""

        del x_m, y_m, z_m, time_utc_ns
        return VelocitySample(0.0, 0.0, 0.0, 0.0, -100.0, 1000.0, 100.0)

    split_rng = np.random.default_rng(20260831)
    brownian_rng = np.random.default_rng(20260831)
    split_result = split_rk4_brownian_step(
        state,
        dt_seconds=-60.0,
        velocity=velocity,
        coefficients=coefficients,
        rng=split_rng,
    )
    expected_displacement = brownian_displacement(
        coefficients,
        dt_seconds=-60.0,
        rng=brownian_rng,
    )
    assert np.array_equal(
        np.array([split_result.x_m, split_result.y_m, split_result.z_m]),
        np.array([state.x_m, state.y_m, state.z_m]) + expected_displacement,
    )


def test_diffusion_sample_positive_and_negative_dt_are_identical() -> None:
    """同一 RNG 狀態的正負 pseudo-time 應有相同梯度漂移與 Brownian 增量。"""

    sample = DiffusionSample(
        DiffusionCoefficients(0.5, 0.25, 0.125),
        diffusivity_divergence_mps=(1.5, -0.75, 0.25),
    )
    backward_rng = np.random.default_rng(7)
    forward_rng = np.random.default_rng(7)
    backward = diffusion_displacement(sample, -4.0, backward_rng)
    forward = diffusion_displacement(sample, 4.0, forward_rng)
    assert np.array_equal(backward, forward)


def test_linear_diffusivity_gradient_has_expected_deterministic_sign() -> None:
    """K=0 時位移只剩 ``+div(K)|dt|``，可直接驗證梯度符號與絕對時間。"""

    sample = DiffusionSample(
        DiffusionCoefficients(0.0, 0.0, 0.0),
        diffusivity_divergence_mps=(2.0, -3.0, 0.5),
    )
    rng = np.random.default_rng(8)
    displacement = diffusion_displacement(sample, -4.0, rng)
    assert np.array_equal(displacement, np.array([8.0, -12.0, 2.0]))


def test_diffusion_sample_validation_rejects_invalid_valid_payload() -> None:
    """qc=OK 不得容納負 K、非有限 K、錯誤梯度維度或非有限梯度。"""

    with pytest.raises(ValueError, match="有限非負"):
        DiffusionSample(DiffusionCoefficients(-1.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="有限非負"):
        DiffusionSample(DiffusionCoefficients(np.nan, 0.0, 0.0), (0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="長度 3"):
        DiffusionSample(DiffusionCoefficients(0.0, 0.0, 0.0), (0.0, 0.0))
    with pytest.raises(ValueError, match="必須有限"):
        DiffusionSample(DiffusionCoefficients(0.0, 0.0, 0.0), (np.inf, 0.0, 0.0))

    invalid = DiffusionSample(
        DiffusionCoefficients(-1.0, 0.0, 0.0),
        (np.nan, 0.0, 0.0),
        qc=SampleQC.INVALID_PHYSICS,
    )
    assert invalid.valid is False
    assert invalid.qc != SampleQC.OK


def test_smagorinsky_bounds_are_finite_nonnegative_and_ordered() -> None:
    """上下限不是物理零值替代品，必須先通過有限、非負與 floor<=cap 閘門。"""

    kwargs = {
        "du_dx_per_s": 1.0,
        "du_dy_per_s": 0.0,
        "dv_dx_per_s": 0.0,
        "dv_dy_per_s": 0.0,
        "triangle_area_m2": 100.0,
        "coefficient_cs": 0.2,
    }
    with pytest.raises(ValueError, match="floor_m2ps"):
        smagorinsky_horizontal_diffusivity(**kwargs, floor_m2ps=-1.0)
    with pytest.raises(ValueError, match="cap_m2ps"):
        smagorinsky_horizontal_diffusivity(**kwargs, cap_m2ps=np.inf)
    with pytest.raises(ValueError, match="不可大於"):
        smagorinsky_horizontal_diffusivity(**kwargs, floor_m2ps=2.0, cap_m2ps=1.0)


def test_smagorinsky_settings_validate_all_scalar_contracts() -> None:
    """SmagorinskySettings 應拒絕 Cs=0、負值、非有限值與反向上下限。"""

    valid = SmagorinskySettings(0.2, 0.0, 10.0, 0.01)
    assert valid == SmagorinskySettings(0.2, 0.0, 10.0, 0.01)
    assert valid.constant_kz_m2ps == 0.01
    with pytest.raises(ValueError, match="大於 0"):
        SmagorinskySettings(0.0, 0.0, 10.0, 0.01)
    with pytest.raises(ValueError, match="非負"):
        SmagorinskySettings(0.2, -1.0, 10.0, 0.01)
    with pytest.raises(ValueError, match="有限"):
        SmagorinskySettings(0.2, 0.0, np.inf, 0.01)
    with pytest.raises(ValueError, match="不可大於"):
        SmagorinskySettings(0.2, 2.0, 1.0, 0.01)
