"""dispersion、深水極限與 wave-from 方向測試。"""

from __future__ import annotations

import math

import numpy as np

from lagrangian_backtracking.stokes import (
    GRAVITY_MPS2,
    deep_water_stokes,
    finite_depth_stokes,
    solve_wave_number,
    wave_from_direction_to_unit_vector,
)


def test_dispersion_root_has_small_residual() -> None:
    """有限水深波數代回分散式後，殘差應接近浮點精度。"""

    omega = 2.0 * math.pi / 8.0
    wave_number = solve_wave_number(angular_frequency_radps=omega, water_depth_m=20.0)
    residual = GRAVITY_MPS2 * wave_number * math.tanh(wave_number * 20.0) - omega**2
    assert abs(residual) < 1e-11


def test_finite_profile_converges_to_deep_water_formula() -> None:
    """足夠深水時有限水深式應回復同一 Hs/Tp 的式 (7)。"""

    finite = finite_depth_stokes(
        significant_wave_height_m=2.0,
        peak_frequency_hz=0.125,
        direction_raw_deg=0.0,
        particle_z_m=-2.0,
        surface_z_m=0.0,
        bed_z_m=-500.0,
    )
    deep_u, deep_v = deep_water_stokes(
        significant_wave_height_m=2.0,
        peak_frequency_hz=0.125,
        direction_raw_deg=0.0,
        relative_z_m=-2.0,
    )
    assert np.isclose(finite.u_mps, deep_u, rtol=1e-10, atol=1e-12)
    assert np.isclose(finite.v_mps, deep_v, rtol=1e-10, atol=1e-12)


def test_wave_from_cardinal_directions_are_reversed_to_propagation() -> None:
    """來自北方的波向南傳，來自東方的波向西傳。"""

    east, north = wave_from_direction_to_unit_vector(0.0)
    assert abs(east) < 1e-12 and np.isclose(north, -1.0)
    east, north = wave_from_direction_to_unit_vector(90.0)
    assert np.isclose(east, -1.0) and abs(north) < 1e-12
