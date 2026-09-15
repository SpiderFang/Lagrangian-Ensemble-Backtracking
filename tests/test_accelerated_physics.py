"""版本化 CPU 數值核心與純 NumPy 參考路徑的解析及執行等價測試。"""

from __future__ import annotations

import copy
import json
import math

import numpy as np
import pytest
from shapely.geometry import box

import lagrangian_backtracking.forcing as forcing_module
from lagrangian_backtracking.accelerated import (
    PHYSICS_KERNEL_BACKEND_NUMBA_CPU_V1,
    PHYSICS_KERNEL_BACKEND_NUMPY_V1,
    warmup_numba_backend,
)
from lagrangian_backtracking.boundaries import BoundaryGeometry
from lagrangian_backtracking.diffusion import (
    DiffusionCoefficients,
    DiffusionSample,
    TimeStepDecision,
    brownian_displacement,
    choose_time_step,
    diffusion_displacement,
)
from lagrangian_backtracking.engine import EngineSettings, run_particle
from lagrangian_backtracking.forcing import CombinedMonthForcing, WaveSample
from lagrangian_backtracking.geometry import DomainProjection
from lagrangian_backtracking.models import (
    EventType,
    ParticleState,
    ParticleStatus,
    SampleQC,
    VelocitySample,
)
from lagrangian_backtracking.stokes import finite_depth_stokes, solve_wave_number


@pytest.mark.parametrize(
    ("omega_radps", "depth_m"),
    [(0.7853981633974483, 0.05), (1.1, 12.0), (0.35, 500.0), (4.0, 2.0)],
)
def test_numba_wave_number_matches_brent_reference_with_declared_tolerance(
    omega_radps: float, depth_m: float
) -> None:
    """淺水、中間水深與深水色散根須落在設定的絕對／相對根誤差內。"""

    reference = solve_wave_number(
        angular_frequency_radps=omega_radps,
        water_depth_m=depth_m,
        physics_kernel_backend=PHYSICS_KERNEL_BACKEND_NUMPY_V1,
    )
    accelerated = solve_wave_number(
        angular_frequency_radps=omega_radps,
        water_depth_m=depth_m,
        physics_kernel_backend=PHYSICS_KERNEL_BACKEND_NUMBA_CPU_V1,
    )
    allowed_root_error = 1.0e-13 + 1.0e-13 * abs(reference)
    assert abs(accelerated - reference) <= allowed_root_error


def test_numba_wave_number_randomized_log_scale_matches_reference() -> None:
    """固定測試 seed 的對數尺度頻率／水深樣本補足解析代表點之外的數值分層。"""

    test_rng = np.random.default_rng(20260915)
    frequencies_hz = np.exp(test_rng.uniform(np.log(0.03), np.log(1.5), size=32))
    depths_m = np.exp(test_rng.uniform(np.log(0.05), np.log(2_000.0), size=32))
    for frequency_hz, depth_m in zip(frequencies_hz, depths_m, strict=True):
        omega_radps = 2.0 * math.pi * float(frequency_hz)
        reference = solve_wave_number(
            angular_frequency_radps=omega_radps,
            water_depth_m=float(depth_m),
            physics_kernel_backend=PHYSICS_KERNEL_BACKEND_NUMPY_V1,
        )
        accelerated = solve_wave_number(
            angular_frequency_radps=omega_radps,
            water_depth_m=float(depth_m),
            physics_kernel_backend=PHYSICS_KERNEL_BACKEND_NUMBA_CPU_V1,
        )
        assert abs(accelerated - reference) <= 1.0e-13 + 1.0e-13 * abs(reference)
        residual = 9.80665 * accelerated * math.tanh(accelerated * float(depth_m)) - omega_radps**2
        assert abs(residual) <= max(1.0e-10, omega_radps**2 * 1.0e-10)


@pytest.mark.parametrize(
    ("wave_height_m", "frequency_hz", "direction_deg", "z_m", "bed_z_m"),
    [
        (2.0, 0.125, 33.0, -2.0, -20.0),
        (0.8, 0.22, 217.0, -0.3, -4.0),
        (2.0, 0.125, 71.0, -2.0, -500.0),
        (0.0, 0.125, 33.0, -2.0, -20.0),
    ],
)
def test_numba_finite_depth_stokes_preserves_profile_and_diagnostics(
    wave_height_m: float,
    frequency_hz: float,
    direction_deg: float,
    z_m: float,
    bed_z_m: float,
) -> None:
    """有限水深與深水穩定支線的速度及診斷量遵循同一個根容差。"""

    arguments = {
        "significant_wave_height_m": wave_height_m,
        "peak_frequency_hz": frequency_hz,
        "direction_raw_deg": direction_deg,
        "particle_z_m": z_m,
        "surface_z_m": 0.0,
        "bed_z_m": bed_z_m,
    }
    reference = finite_depth_stokes(
        **arguments, physics_kernel_backend=PHYSICS_KERNEL_BACKEND_NUMPY_V1
    )
    accelerated = finite_depth_stokes(
        **arguments, physics_kernel_backend=PHYSICS_KERNEL_BACKEND_NUMBA_CPU_V1
    )
    reference_wave_number = reference.wave_number_per_m
    accelerated_wave_number = accelerated.wave_number_per_m
    root_difference = abs(accelerated_wave_number - reference_wave_number)
    root_tolerance = 1.0e-13 + 1.0e-13 * abs(reference_wave_number)
    assert root_difference <= root_tolerance

    # 其餘與波數有關的欄位依公式對波數的一階敏感度推導誤差上界，而非另訂較寬的
    # 任意相對容差。最後加上的項僅覆蓋雙精度基本運算與 sin/cos 的數個 ULP 捨入。
    epsilon = np.finfo(np.float64).eps
    roundoff = 16.0 * epsilon
    assert abs(accelerated.wavelength_m - reference.wavelength_m) <= (
        root_difference
        * 2.0
        * math.pi
        / min(reference_wave_number, accelerated_wave_number) ** 2
        + roundoff * max(abs(reference.wavelength_m), abs(accelerated.wavelength_m))
    )
    water_depth_m = -bed_z_m
    assert abs(accelerated.kh - reference.kh) <= (
        root_difference * water_depth_m
        + roundoff * max(abs(reference.kh), abs(accelerated.kh))
    )
    amplitude_m = wave_height_m / 2.0
    assert abs(accelerated.steepness_ka - reference.steepness_ka) <= (
        root_difference * amplitude_m
        + roundoff * max(abs(reference.steepness_ka), abs(accelerated.steepness_ka))
    )
    assert accelerated.relative_depth == reference.relative_depth

    def logarithmic_speed_sensitivity(wave_number: float) -> float:
        """回傳既有 Stokes 公式對波數的對數導數，作局部誤差傳播上界。"""

        if wave_number * water_depth_m > 20.0:
            return 1.0 / wave_number + 2.0 * z_m
        return (
            1.0 / wave_number
            + 2.0 * (z_m + water_depth_m) * math.tanh(2.0 * wave_number * (z_m + water_depth_m))
            - 2.0 * water_depth_m / math.tanh(wave_number * water_depth_m)
        )

    speed_error = root_difference * max(
        abs(reference.u_mps * logarithmic_speed_sensitivity(reference_wave_number)),
        abs(accelerated.u_mps * logarithmic_speed_sensitivity(accelerated_wave_number)),
        abs(reference.v_mps * logarithmic_speed_sensitivity(reference_wave_number)),
        abs(accelerated.v_mps * logarithmic_speed_sensitivity(accelerated_wave_number)),
    )
    directional_roundoff = roundoff * max(
        abs(reference.u_mps),
        abs(reference.v_mps),
        abs(accelerated.u_mps),
        abs(accelerated.v_mps),
        abs((reference.u_mps**2 + reference.v_mps**2) ** 0.5),
        abs((accelerated.u_mps**2 + accelerated.v_mps**2) ** 0.5),
    )
    assert abs(accelerated.u_mps - reference.u_mps) <= speed_error + directional_roundoff
    assert abs(accelerated.v_mps - reference.v_mps) <= speed_error + directional_roundoff


@pytest.mark.parametrize(
    "invalid_arguments",
    [
        {"angular_frequency_radps": 0.0, "water_depth_m": 10.0},
        {"angular_frequency_radps": 1.0, "water_depth_m": 0.0},
    ],
)
def test_numba_wave_solver_keeps_invalid_input_exception_contract(
    invalid_arguments: dict[str, float],
) -> None:
    """Numba wave solver 不將非正頻率或水深轉成狀態碼回傳給公開呼叫端。"""

    with pytest.raises(ValueError, match="有限正值"):
        solve_wave_number(
            **invalid_arguments,
            physics_kernel_backend=PHYSICS_KERNEL_BACKEND_NUMBA_CPU_V1,
        )


def test_numba_stokes_keeps_surface_qc_gate_before_physics_kernel() -> None:
    """超出海面容許帶仍由 Python 幾何 gate 拒絕，不進入加速公式。"""

    with pytest.raises(ValueError, match="有效海床與海面"):
        finite_depth_stokes(
            significant_wave_height_m=2.0,
            peak_frequency_hz=0.125,
            direction_raw_deg=33.0,
            particle_z_m=1.0e-4,
            surface_z_m=0.0,
            bed_z_m=-20.0,
            physics_kernel_backend=PHYSICS_KERNEL_BACKEND_NUMBA_CPU_V1,
        )


def test_combined_forcing_passes_versioned_backend_into_finite_depth_stokes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """forcing provider 將已驗證後端傳入 Stokes，並保留原波浪/QC/速度合成契約。"""

    class ValidCurrent:
        """提供一筆公尺座標、海面海床及三向流速均有效的合成 OCM 樣本。"""

        def sample(self, *_: object, **__: object) -> VelocitySample:
            """回傳固定 current；本測試只觀察 finite-depth Stokes 的核心選擇。"""

            return VelocitySample(0.1, -0.05, 0.0, 0.0, -20.0, 100.0, 10.0)

    class ValidWave:
        """提供一筆固定且具有物理意義的 NWW3 波浪摘要，不讀取產品檔案。"""

        def sample(self, *_: object) -> WaveSample:
            """回傳有效波高、峰值頻率與來波方向，維持正北順時針角度慣例。"""

            return WaveSample(2.0, 0.125, 33.0, 0, SampleQC.OK)

    observed_arguments: list[dict[str, object]] = []
    reference_stokes = finite_depth_stokes

    def observe_stokes(**arguments: object):
        """保存 provider 實際選擇的後端，再呼叫真實數值實作完成樣本。"""

        observed_arguments.append(arguments)
        return reference_stokes(**arguments)  # type: ignore[arg-type]

    monkeypatch.setattr(forcing_module, "finite_depth_stokes", observe_stokes)
    provider = CombinedMonthForcing(
        ocm=ValidCurrent(),  # type: ignore[arg-type]
        nww=ValidWave(),  # type: ignore[arg-type]
        projection=DomainProjection(121.0, 25.0),
        settling_velocity_mps=-0.2,
        include_stokes=True,
        physics_kernel_backend=PHYSICS_KERNEL_BACKEND_NUMBA_CPU_V1,
    )

    result = provider.sample(0.0, 0.0, -2.0, 1_000_000_000)

    assert result.valid
    assert len(observed_arguments) == 1
    assert observed_arguments[0]["physics_kernel_backend"] == PHYSICS_KERNEL_BACKEND_NUMBA_CPU_V1
    assert result.components is not None
    expected = reference_stokes(**observed_arguments[0])
    assert result.components.stokes_u_mps == expected.u_mps
    assert result.components.stokes_v_mps == expected.v_mps


@pytest.mark.parametrize(
    ("coefficients", "divergence"),
    [
        (DiffusionCoefficients(0.2, 0.1, 0.01), (0.0, 0.0, 0.0)),
        (DiffusionCoefficients(0.2, 0.1, 0.01), (0.003, -0.002, 0.0004)),
        (DiffusionCoefficients(0.0, 0.0, 0.0), (0.003, -0.002, 0.0004)),
    ],
)
def test_numba_diffusion_preserves_displacement_and_rng_state(
    coefficients: DiffusionCoefficients, divergence: tuple[float, float, float]
) -> None:
    """固定 PCG64DXSM seed 下位移逐位元一致，且兩路徑消耗相同數目的常態數。"""

    sample = DiffusionSample(coefficients, diffusivity_divergence_mps=divergence)
    reference_rng = np.random.Generator(np.random.PCG64DXSM(20260915))
    accelerated_rng = np.random.Generator(np.random.PCG64DXSM(20260915))
    if divergence == (0.0, 0.0, 0.0):
        reference = brownian_displacement(
            coefficients,
            dt_seconds=-3.75,
            rng=reference_rng,
            physics_kernel_backend=PHYSICS_KERNEL_BACKEND_NUMPY_V1,
        )
        accelerated = brownian_displacement(
            coefficients,
            dt_seconds=-3.75,
            rng=accelerated_rng,
            physics_kernel_backend=PHYSICS_KERNEL_BACKEND_NUMBA_CPU_V1,
        )
    else:
        reference = diffusion_displacement(
            sample,
            -3.75,
            reference_rng,
            physics_kernel_backend=PHYSICS_KERNEL_BACKEND_NUMPY_V1,
        )
        accelerated = diffusion_displacement(
            sample,
            -3.75,
            accelerated_rng,
            physics_kernel_backend=PHYSICS_KERNEL_BACKEND_NUMBA_CPU_V1,
        )
    assert np.array_equal(accelerated, reference)
    assert accelerated_rng.bit_generator.state == reference_rng.bit_generator.state


@pytest.mark.parametrize(
    "values",
    [
        (0.0, 0.0, 100.0, 10.0, DiffusionCoefficients(0.0, 0.0, 0.0), 1.0, 50.0, None),
        (0.4, 0.01, 100.0, 10.0, DiffusionCoefficients(0.2, 0.1, 0.01), 1.0, 50.0, 4.0),
        (0.0, 0.0, 8.0, 0.1, DiffusionCoefficients(4.0, 4.0, 0.0), 0.5, 50.0, None),
        (0.0, 0.0, 100.0, 10.0, DiffusionCoefficients(0.0, 0.0, 0.0), 1.0, 50.0, 1.0),
    ],
)
def test_numba_time_step_decisions_match_candidate_order_and_clamps(
    values: tuple[object, ...],
) -> None:
    """每種候選、平手優先序與最小步長夾制均須產生同一秒數及原因。"""

    (speed_h, speed_v, scale_h, scale_v, coefficients, dt_min, dt_max, boundary) = values
    arguments = {
        "speed_horizontal_mps": speed_h,
        "speed_vertical_mps": speed_v,
        "horizontal_scale_m": scale_h,
        "vertical_scale_m": scale_v,
        "coefficients": coefficients,
        "dt_min_seconds": dt_min,
        "dt_max_seconds": dt_max,
        "seconds_to_forcing_boundary": boundary,
    }
    reference = choose_time_step(
        **arguments, physics_kernel_backend=PHYSICS_KERNEL_BACKEND_NUMPY_V1
    )
    accelerated = choose_time_step(
        **arguments, physics_kernel_backend=PHYSICS_KERNEL_BACKEND_NUMBA_CPU_V1
    )
    assert accelerated == reference
    assert isinstance(accelerated, TimeStepDecision)


def _run_particle_with_backend(backend: str):
    """以固定三向速度場與 seed 執行短軌跡，供逐欄比較 engine 輸出與呼叫順序。"""

    state = ParticleState(
        particle_id="p0",
        scenario_id="s0",
        member_id=0,
        study_site_id="gongliao",
        analysis_region_id="A",
        receptor_id="r0",
        x_m=0.0,
        y_m=0.0,
        z_m=-10.0,
        time_utc_ns=100_000_000_000,
    )
    calls: list[tuple[float, float, float, int]] = []

    def velocity(x_m: float, y_m: float, z_m: float, time_utc_ns: int) -> VelocitySample:
        """回傳位置相依但 QC 有效的速度，所有 stage 呼叫均留作順序證據。"""

        calls.append((x_m, y_m, z_m, time_utc_ns))
        return VelocitySample(
            0.4 + 0.001 * x_m,
            -0.2 + 0.001 * y_m,
            0.01 + 0.0001 * z_m,
            0.0,
            -100.0,
            100.0,
            10.0,
        )

    settings = EngineSettings(
        dt_min_seconds=0.5,
        dt_max_seconds=2.0,
        output_interval_seconds=2.0,
        max_backtrack_seconds=5.0,
        maximum_step_count=100,
        earliest_forcing_time_utc_ns=0,
        physics_kernel_backend=backend,
    )
    rng = np.random.Generator(np.random.PCG64DXSM(20260915))
    result = run_particle(
        state,
        velocity=velocity,
        boundaries=BoundaryGeometry(
            own_local_domain=box(-100.0, -100.0, 100.0, 100.0),
            flow_domain=box(-1_000.0, -1_000.0, 1_000.0, 1_000.0),
            foreign_local_domains={},
        ),
        behavior_class="sinking",
        diffusion=DiffusionCoefficients(0.2, 0.1, 0.01),
        settings=settings,
        rng=rng,
    )
    return result, calls, copy.deepcopy(rng.bit_generator.state)


def test_numba_engine_preserves_trajectory_step_count_events_and_rng() -> None:
    """同 seed 下整條軌跡、所有觀測與終止事件分類逐欄一致。"""

    reference, reference_calls, reference_rng_state = _run_particle_with_backend(
        PHYSICS_KERNEL_BACKEND_NUMPY_V1
    )
    accelerated, accelerated_calls, accelerated_rng_state = _run_particle_with_backend(
        PHYSICS_KERNEL_BACKEND_NUMBA_CPU_V1
    )
    assert accelerated.final_state == reference.final_state
    assert accelerated.observations == reference.observations
    assert accelerated.events == reference.events
    assert accelerated.step_count == reference.step_count
    assert accelerated.minimum_clamp_count == reference.minimum_clamp_count
    assert accelerated_calls == reference_calls
    assert accelerated_rng_state == reference_rng_state
    assert accelerated.final_state.status == ParticleStatus.MAX_AGE
    assert [event.event_type for event in accelerated.events][-1] == EventType.MAX_AGE


def test_warmup_compiles_every_kernel_without_rng_or_disk_cache_writes(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Worker warmup 只編譯微型合成資料，回傳 JSON 摘要且不讀檔、不耗 RNG、不寫 cache。"""

    cache_root = tmp_path / "explicit-numba-cache"
    cache_root.mkdir()
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(cache_root))
    rng = np.random.Generator(np.random.PCG64DXSM(20260915))
    state_before = copy.deepcopy(rng.bit_generator.state)

    summary = warmup_numba_backend()

    assert summary["backend"] == PHYSICS_KERNEL_BACKEND_NUMBA_CPU_V1
    assert summary["status"] == "compiled"
    assert summary["disk_cache_enabled"] is False
    assert summary["kernels"]
    assert set(summary["kernels"].values()) == {"compiled"}
    json.dumps(summary, sort_keys=True)
    assert rng.bit_generator.state == state_before
    assert list(cache_root.iterdir()) == []


def test_backend_validation_rejects_unknown_token_without_reference_fallback() -> None:
    """拼錯或未登錄版本不得默默回退 NumPy，避免正式設定與實際後端不一致。"""

    with pytest.raises(ValueError, match="不支援"):
        solve_wave_number(
            angular_frequency_radps=1.0,
            water_depth_m=10.0,
            physics_kernel_backend="numba_cpu_v2",
        )
