"""signed-time RK4、浮沉與 Brownian 統計參考測試。"""

from __future__ import annotations

import numpy as np

from lagrangian_backtracking.diffusion import DiffusionCoefficients, brownian_displacement
from lagrangian_backtracking.integrators import rk4_step
from lagrangian_backtracking.models import ParticleState, VelocitySample


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
