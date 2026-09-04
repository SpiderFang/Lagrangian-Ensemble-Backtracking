"""signed-time RK4、浮沉與 Brownian 統計參考測試。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

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
from lagrangian_backtracking.integrators import (
    SamplingContext,
    SamplingError,
    rk4_step,
    split_rk4_brownian_step,
)
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


@pytest.mark.parametrize("stage_index", [1, 2, 3, 4])
@pytest.mark.parametrize("nonfinite_velocity", [False, True])
def test_failed_rk_stage_context_uses_exact_existing_query(
    stage_index: int, nonfinite_velocity: bool
) -> None:
    """失敗證據必須來自原查詢，不增加查詢或消耗擴散亂數。

    合成常流為 (1, -0.5, 0.25) 公尺／秒，逆向兩秒使中間位置可解析核對。
    分別注入垂向不支援與 qc=OK 但速度非有限的樣本，確保錯誤品質旗標仍沿用舊規則。
    """

    state = _state()
    queries: list[tuple[float, float, float, int]] = []

    def velocity(x_m: float, y_m: float, z_m: float, time_utc_ns: int) -> VelocitySample:
        """在指定既有計算點回傳失敗樣本；任意診斷字典不可被上下文複製。"""

        queries.append((x_m, y_m, z_m, time_utc_ns))
        sample = VelocitySample(1.0, -0.5, 0.25, 0.3, -90.0, 1000.0, 10.0)
        if len(queries) == stage_index:
            return replace(
                sample,
                u_mps=np.nan if nonfinite_velocity else sample.u_mps,
                qc=SampleQC.OK if nonfinite_velocity else SampleQC.VERTICAL_UNSUPPORTED,
                diagnostics={"untrusted": "/not-a-real-path/private", "bad": np.nan},
            )
        return sample

    rng = np.random.Generator(np.random.PCG64DXSM(13))
    before = deepcopy(rng.bit_generator.state)
    with pytest.raises(SamplingError) as caught:
        split_rk4_brownian_step(
            state, dt_seconds=-2.0, velocity=velocity,
            coefficients=DiffusionCoefficients(0.2, 0.1, 0.01), rng=rng,
        )
    expected_queries = [
        (0.0, 0.0, -10.0, state.time_utc_ns),
        (-1.0, 0.5, -10.25, state.time_utc_ns - 1_000_000_000),
        (-1.0, 0.5, -10.25, state.time_utc_ns - 1_000_000_000),
        (-2.0, 1.0, -10.5, state.time_utc_ns - 2_000_000_000),
    ]
    assert queries == expected_queries[:stage_index]
    assert caught.value.stage == f"k{stage_index}"
    expected_qc = SampleQC.NUMERICAL_FAILURE if nonfinite_velocity else SampleQC.VERTICAL_UNSUPPORTED
    assert caught.value.qc == expected_qc
    assert caught.value.context == SamplingContext(*expected_queries[stage_index - 1], 0.3, -90.0)
    assert rng.bit_generator.state == before
    assert state == _state()


def test_sampling_error_keeps_legacy_constructor_and_optional_context() -> None:
    """舊兩參數例外仍可用，新增可選上下文不強迫外部呼叫端假造未知數值。"""

    legacy = SamplingError("external-stage", SampleQC.TIME_GAP)
    assert legacy.context is None
    assert legacy.stage == "external-stage"
    assert legacy.qc == SampleQC.TIME_GAP
    context = SamplingContext(z_m=np.nan)
    enriched = SamplingError("k2", SampleQC.VERTICAL_UNSUPPORTED, context=context)
    assert enriched.context is context


def test_rk4_rotation_field_closes_after_forward_and_reverse_steps() -> None:
    """旋轉場以正、負 signed ``dt`` 往返後，位置應在明定公尺誤差內閉合。

    測試場為二維剛體旋轉 ``u=-omega*y``、``v=omega*x``，其中座標是 m、速度是 m/s、
    ``omega=0.1 s^-1``、每一步 ``dt=1 s``。經典四階 Runge-Kutta 並非精確可逆，因此
    容許正向再反向一個步長的殘差不超過 ``1e-7 m``；時間必須精確回到原本的 UTC ns。
    ``age_seconds`` 是累計經過量而非可逆座標，往返後預期為 ``2 s``。
    """

    initial = replace(_state(), x_m=1.0, y_m=4.0)
    omega_per_s = 0.1

    def velocity(x_m: float, y_m: float, z_m: float, time_utc_ns: int) -> VelocitySample:
        """回傳不依時間變化的剛體旋轉速度；z 軸維持靜止以隔離水平閉合。"""

        del z_m, time_utc_ns
        return VelocitySample(
            -omega_per_s * y_m,
            omega_per_s * x_m,
            0.0,
            0.0,
            -100.0,
            1_000.0,
            1.0,
        )

    forward = rk4_step(initial, dt_seconds=1.0, velocity=velocity)
    round_trip = rk4_step(forward, dt_seconds=-1.0, velocity=velocity)
    assert np.allclose(
        [round_trip.x_m, round_trip.y_m, round_trip.z_m],
        [initial.x_m, initial.y_m, initial.z_m],
        rtol=0.0,
        atol=1e-7,
    )
    assert round_trip.time_utc_ns == initial.time_utc_ns
    assert round_trip.age_seconds == 2.0


def test_rk4_spatial_shear_field_closes_after_forward_and_reverse_steps() -> None:
    """空間剪切場的正向與反向 RK4 步驟應閉合，且 stage 位置確實會被重新取樣。

    採用 ``u=gamma*y``、``v=0`` 的水平剪切場；``gamma=0.35 s^-1`` 與 ``y``（m）相乘
    產生 m/s。這個解析場的 y 不變、x 線性漂移，所以正向 ``dt=2 s`` 後再以 ``-2 s``
    回溯理論上精確回到步首；浮點運算的位置殘差容許 ``1e-12 m``。另外記錄八次 stage
    取樣，確認至少有一個 stage 看到已更新的 x 座標，而非每次都重用步首位置。
    """

    initial = replace(_state(), x_m=1.0, y_m=4.0)
    gamma_per_s = 0.35
    sampled_positions: list[tuple[float, float]] = []

    def velocity(x_m: float, y_m: float, z_m: float, time_utc_ns: int) -> VelocitySample:
        """記錄每個 RK4 stage 的水平位置，並回傳單純剪切速度。"""

        del z_m, time_utc_ns
        sampled_positions.append((x_m, y_m))
        return VelocitySample(
            gamma_per_s * y_m,
            0.0,
            0.0,
            0.0,
            -100.0,
            1_000.0,
            1.0,
        )

    forward = rk4_step(initial, dt_seconds=2.0, velocity=velocity)
    round_trip = rk4_step(forward, dt_seconds=-2.0, velocity=velocity)
    assert np.allclose(
        [round_trip.x_m, round_trip.y_m, round_trip.z_m],
        [initial.x_m, initial.y_m, initial.z_m],
        rtol=0.0,
        atol=1e-12,
    )
    assert round_trip.time_utc_ns == initial.time_utc_ns
    assert round_trip.age_seconds == 4.0
    assert len(sampled_positions) == 8
    assert any(abs(x_m - initial.x_m) > 0.0 for x_m, _ in sampled_positions[1:4])


def test_rk4_has_fourth_order_convergence_for_position_and_time_dependent_field() -> None:
    """用有解析解的 ``x'=x/tau+a*t`` 驗證全域四階收斂與每 stage 的時空取樣。

    ``x`` 是 m、時間是 s、速度是 m/s；取 ``tau=1 s``、``a=1 m/s²``、初值
    ``x(0)=1 m``，故解析解為 ``x(t)=2*exp(t)-t-1 m``。先以一個 ``0.75 s`` 步驟
    逐一核對四個 stage 的 x（m）與 UTC ns，再用 4、8、16、32 個等距正向步驟計算
    ``t=1 s`` 的誤差。相鄰加倍解析誤差的觀察階數需大於 3.5（理想值 4），且最後誤差
    以 ``2e-7 m`` 作為絕對數值上限；這些門檻同時避免把「只取步首位置／時間」誤判為
    合格的四階實作。
    """

    start_ns = _state().time_utc_ns
    tau_seconds = 1.0
    acceleration_mps2 = 1.0
    initial_x_m = 1.0

    def make_state() -> ParticleState:
        """建立固定 UTC 起點與解析初值，其他欄位保持可積分的有效水域資訊。"""

        return ParticleState(
            particle_id="analytic",
            scenario_id="rk4_order",
            member_id=0,
            study_site_id="synthetic",
            analysis_region_id="A",
            receptor_id="r0",
            x_m=initial_x_m,
            y_m=0.0,
            z_m=-10.0,
            time_utc_ns=start_ns,
        )

    stage_calls: list[tuple[float, float, float, int]] = []

    def velocity_with_trace(
        x_m: float, y_m: float, z_m: float, time_utc_ns: int
    ) -> VelocitySample:
        """以收到的 stage 位置與 UTC 時間計算 m/s，並保存取樣順序供契約檢查。"""

        elapsed_seconds = (time_utc_ns - start_ns) / 1_000_000_000
        stage_calls.append((x_m, y_m, z_m, time_utc_ns))
        return VelocitySample(
            x_m / tau_seconds + acceleration_mps2 * elapsed_seconds,
            0.0,
            0.0,
            0.0,
            -100.0,
            1_000.0,
            1.0,
        )

    stage_dt_seconds = 0.75
    one_step = rk4_step(make_state(), dt_seconds=stage_dt_seconds, velocity=velocity_with_trace)
    assert len(stage_calls) == 4
    expected_stage_times = [
        start_ns,
        start_ns + 375_000_000,
        start_ns + 375_000_000,
        start_ns + 750_000_000,
    ]
    expected_stage_x = [1.0, 1.375, 1.65625, 2.5234375]
    assert [call[3] for call in stage_calls] == expected_stage_times
    assert np.allclose(
        [call[0] for call in stage_calls], expected_stage_x, rtol=0.0, atol=1e-12
    )
    assert np.allclose(
        [call[1] for call in stage_calls], 0.0, rtol=0.0, atol=1e-15
    )
    assert np.allclose(
        [call[2] for call in stage_calls], -10.0, rtol=0.0, atol=1e-15
    )
    assert one_step.time_utc_ns == start_ns + 750_000_000

    def integrate(step_count: int) -> tuple[float, ParticleState]:
        """以指定步數完成 1 s 正向積分，回傳 m 誤差與最終狀態。"""

        state = make_state()
        dt_seconds = 1.0 / step_count

        def velocity(
            x_m: float, y_m: float, z_m: float, time_utc_ns: int
        ) -> VelocitySample:
            """在每個步驟的四個 stage 重新使用 m 與 UTC ns 計算 m/s。"""

            elapsed_seconds = (time_utc_ns - start_ns) / 1_000_000_000
            return VelocitySample(
                x_m / tau_seconds + acceleration_mps2 * elapsed_seconds,
                0.0,
                0.0,
                0.0,
                -100.0,
                1_000.0,
                1.0,
            )

        for _ in range(step_count):
            state = rk4_step(state, dt_seconds=dt_seconds, velocity=velocity)
        exact_x_m = (initial_x_m + acceleration_mps2 * tau_seconds**2) * np.exp(
            1.0 / tau_seconds
        ) - acceleration_mps2 * tau_seconds * 1.0 - acceleration_mps2 * tau_seconds**2
        return abs(state.x_m - exact_x_m), state

    step_counts = np.array([4, 8, 16, 32], dtype=np.int64)
    errors_m, final_states = zip(*(integrate(int(count)) for count in step_counts), strict=True)
    errors = np.asarray(errors_m, dtype=np.float64)
    observed_orders = np.log2(errors[:-1] / errors[1:])
    assert np.all(observed_orders > 3.5)
    assert errors[-1] < 2.0e-7
    assert all(state.time_utc_ns == start_ns + 1_000_000_000 for state in final_states)
    assert all(state.age_seconds == 1.0 for state in final_states)


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

    calls: list[tuple[float, float, float, int]] = []

    def velocity(x_m: float, y_m: float, z_m: float, time_utc_ns: int) -> VelocitySample:
        """提供零確定性速度，讓測試只比較擴散亂數的相容性。"""

        calls.append((x_m, y_m, z_m, time_utc_ns))
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
    # 零流場的四個查詢仍依原時序，並且只在其後消耗一次三向布朗位移的亂數。
    assert calls == [
        (state.x_m, state.y_m, state.z_m, state.time_utc_ns + offset)
        for offset in (0, -30_000_000_000, -30_000_000_000, -60_000_000_000)
    ]
    assert split_rng.bit_generator.state == brownian_rng.bit_generator.state


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
