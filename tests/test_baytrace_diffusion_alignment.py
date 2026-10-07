"""WP-51 D：垂向隨機漫步與水平均勻雜訊（對齊 BayTrace ``batch_move_oil``）的驗收測試。

涵蓋：

1. 均勻亂數產生器：三個亂數都落在開區間 (0, 1)、平均 1/2、變異數 1/12、彼此不相關；
   前兩個與舊 Box-Muller 版本共用同一組位元（``z0**2 + z1**2 == -2 ln(ux)``）。
2. 位移公式：邊界值、``sqrt(6*K*dtb)`` 上界、Visser 修正與梯度漂移的公式。
3. 融合 kernel 內的統計行為（零流場、常數擴散係數）：
   水平變異數 ``2*K*t``、垂向變異數 ``2*Kv*t``（Kv 預設 3.0e-4 m²/s）、
   Visser 在常數 Kv 時不改變結果、垂向位移被 ``batch_clamp_z`` 夾回水柱、
   ``off`` 模式不動 z、舊高斯路徑仍是預設。
4. ``ocm_field`` 介面：形狀／有限值檢查、常數場的變異數、線性 Kv 場的梯度漂移方向與量級。
5. 小時主軸：新設定的批次組成無關性（位元相同）、垂向擴散確實讓 z 變動、
   不支援組合（python 實作＋uniform、ocm_field）fail-closed。
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import test_native_hour_major as thm
import test_substep_numba as tsn
from lagrangian_backtracking.native_runner import backward_driver as bd
from lagrangian_backtracking.native_runner import build_native_grid_arrays
from lagrangian_backtracking.native_runner import hour_major as hm
from lagrangian_backtracking.native_runner import hour_major_provider as hmp
from lagrangian_backtracking.native_runner.substep_numba import (
    _counter_based_gaussian_pair_scalar,
    _counter_based_uniform_triple_scalar,
    baytrace_uniform_diffusion_displacement,
    mirror_rk4_substep_loop_numba,
)


# ══════════════════════════════════════════════════════════════════════════
# 夾具
# ══════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def handles() -> bd.BayTraceModuleHandles:
    root = os.environ.get("LBT_BAYTRACE_ROOT")
    if not root:
        pytest.skip("需要 LBT_BAYTRACE_ROOT 指向 BayTrace 原始碼根目錄才能執行本測試")
    return bd.load_baytrace_modules(root)


@pytest.fixture(scope="module")
def grid3(handles):
    """3 層（z=-20, -10, 0）、200 m 間距、水深 20 m 的 10×10 合成網格。"""

    mesh = tsn._build_mesh()
    n_nodes = mesh.node_xy.shape[0]
    zcor = np.zeros((n_nodes, tsn._NVRT), dtype=np.float64)
    zcor[:, 0] = -tsn._DP_M
    zcor[:, 1] = -tsn._DP_M / 2.0  # test_substep_numba 的網格中間層是 0（退化），這裡要真正的 3 個等間距層
    grid = build_native_grid_arrays(mesh, zcor_reference=zcor, eta_reference=np.zeros(n_nodes))
    built = bd.build_native_backward_driver_grid(grid, handles, h0=0.0)
    assert built.ic3_cross_check_ok
    return built


@pytest.fixture(scope="module")
def grid2(handles):
    """小時主軸測試用的 2 層合成網格（與 test_native_hour_major 相同）。"""

    mesh = thm._build_mesh()
    n_nodes = mesh.node_xy.shape[0]
    zcor = np.zeros((n_nodes, thm._NVRT), dtype=np.float64)
    zcor[:, 0] = -thm._DP_M
    grid = build_native_grid_arrays(mesh, zcor_reference=zcor, eta_reference=np.zeros(n_nodes))
    built = bd.build_native_backward_driver_grid(grid, handles, h0=0.0)
    assert built.ic3_cross_check_ok
    return built


def _run_kernel(
    grid,
    handles,
    *,
    n: int,
    z0: float,
    seconds: float,
    ndeltp: int = 60,
    kh: float = 0.05,
    master_seed: int = 42,
    **kwargs,
):
    """零流場、常數水平 K 的融合 kernel 一次呼叫（一個「流場小時」），回傳 pa 與初始位置。"""

    composed = tsn._make_composed(grid, u=0.0, v=0.0)
    nvrt = int(grid.schism_grid.nvrt)
    n_nodes = tsn._n_nodes(grid)
    kh_field = np.full((nvrt, n_nodes), kh)
    x0 = np.full(n, tsn._DOMAIN_X / 2.0)
    y0 = np.full(n, tsn._DOMAIN_Y / 2.0)
    # BayTrace 的初始化以 ``pa.z0``（相對海面的深度）決定起始 z（``batch_init_velocity``：
    # ``z = z0 + z_sfc``，並夾在海床之上），所以 z0 也必須設定，不能只設 ``pa.z``。
    pa = handles.particle_arrays_cls(n)
    pa.x[:] = x0
    pa.y[:] = y0
    pa.z[:] = z0
    pa.z0[:] = z0
    pa.status[:] = 0
    not_found = bd.locate_initial_elements(
        grid, handles, pa, eta_m=composed.eta_field1, u_current_mps=composed.u_field1,
        v_current_mps=composed.v_field1, w_current_mps=composed.w_field1,
    )
    assert not np.any(not_found)
    assert np.allclose(pa.z, z0)
    seed_lo, seed_hi = bd.derive_particle_diffusion_seed_lo_hi_u64(
        [f"{i:032x}" for i in range(n)], master_seed=master_seed
    )
    counter = np.zeros(n, dtype=np.int64)
    mirror_rk4_substep_loop_numba(
        grid, handles, pa, ibf=-1, ndeltp=ndeltp, dtb=seconds / ndeltp, composed=composed,
        enable_diffusion=True, hf1_field=kh_field, hf2_field=kh_field,
        particle_seed_lo=seed_lo, particle_seed_hi=seed_hi, particle_step_counter=counter,
        **kwargs,
    )
    return pa, x0, y0, counter


# ══════════════════════════════════════════════════════════════════════════
# 1. 均勻亂數產生器
# ══════════════════════════════════════════════════════════════════════════


def test_uniform_triple_open_interval_moments_and_independence() -> None:
    seed_lo, seed_hi = np.uint64(0x1234567890ABCDEF), np.uint64(0x0FEDCBA987654321)
    n = 200_000
    u = np.empty((n, 3))
    for c in range(n):
        u[c] = _counter_based_uniform_triple_scalar(seed_lo, seed_hi, np.uint64(c))
    assert np.all(u > 0.0) and np.all(u < 1.0)
    assert np.allclose(u.mean(axis=0), 0.5, atol=4.0 * np.sqrt(1.0 / 12.0 / n))
    assert np.allclose(u.var(axis=0), 1.0 / 12.0, rtol=0.02)
    corr = np.corrcoef(u.T)
    assert np.all(np.abs(corr[np.triu_indices(3, k=1)]) < 0.01)
    # 相鄰計數器也不相關（不同子步的雜訊彼此獨立）
    assert abs(np.corrcoef(u[:-1, 0], u[1:, 0])[0, 1]) < 0.01


def test_uniform_triple_first_two_share_bits_with_box_muller() -> None:
    """ux、uy 與舊 Box-Muller 的 u1、u2 是同一組位元：z0²+z1² = -2 ln(u1)。"""

    seed_lo, seed_hi = np.uint64(11), np.uint64(22)
    for c in range(200):
        ux, _uy, _uz = _counter_based_uniform_triple_scalar(seed_lo, seed_hi, np.uint64(c))
        z0, z1 = _counter_based_gaussian_pair_scalar(seed_lo, seed_hi, np.uint64(c))
        assert z0 * z0 + z1 * z1 == pytest.approx(-2.0 * np.log(ux), rel=1e-12)


def test_uniform_triple_is_reproducible_and_counter_dependent() -> None:
    a = _counter_based_uniform_triple_scalar(np.uint64(5), np.uint64(6), np.uint64(7))
    b = _counter_based_uniform_triple_scalar(np.uint64(5), np.uint64(6), np.uint64(7))
    c = _counter_based_uniform_triple_scalar(np.uint64(5), np.uint64(6), np.uint64(8))
    assert a == b
    assert a != c


# ══════════════════════════════════════════════════════════════════════════
# 2. 位移公式
# ══════════════════════════════════════════════════════════════════════════


def test_displacement_formula_bounds_and_scale() -> None:
    dtb, kh, kv = 30.0, 0.04, 3.0e-4
    # u=0.5 → rnd=0 → 無隨機位移
    assert baytrace_uniform_diffusion_displacement(0.5, 0.5, 0.5, kh, kv, 0.0, dtb, True, False) == (0.0, 0.0, 0.0)
    # u→1 → rnd→1 → 位移 → sqrt(6*K*dtb)
    dx, dy, dz = baytrace_uniform_diffusion_displacement(1.0, 0.0, 1.0, kh, kv, 0.0, dtb, True, False)
    assert dx == pytest.approx(np.sqrt(6.0 * kh * dtb))
    assert dy == pytest.approx(-np.sqrt(6.0 * kh * dtb))
    assert dz == pytest.approx(np.sqrt(6.0 * kv * dtb))
    # 垂向關閉時 dz 恆為 0
    assert baytrace_uniform_diffusion_displacement(0.9, 0.9, 0.9, kh, kv, 1.0e-3, dtb, False, True)[2] == 0.0
    # 負的水平 K 夾到 0
    assert baytrace_uniform_diffusion_displacement(0.9, 0.9, 0.9, -1.0, kv, 0.0, dtb, True, False)[:2] == (0.0, 0.0)


def test_displacement_visser_correction_and_gradient_drift() -> None:
    dtb, kv, dk = 60.0, 2.0e-3, 5.0e-4
    plain = baytrace_uniform_diffusion_displacement(1.0, 0.5, 1.0, 0.0, kv, dk, dtb, True, False)[2]
    visser = baytrace_uniform_diffusion_displacement(1.0, 0.5, 1.0, 0.0, kv, dk, dtb, True, True)[2]
    assert plain == pytest.approx(np.sqrt(6.0 * kv * dtb) + dk * dtb)
    kv_star = kv + 0.5 * dk * dk * dtb
    assert visser == pytest.approx(np.sqrt(6.0 * kv_star * dtb) + dk * dtb)
    assert visser > plain
    # 修正後 Kv 低於下限 1e-6 時夾到 1e-6
    tiny = baytrace_uniform_diffusion_displacement(1.0, 0.5, 1.0, 0.0, 0.0, 0.0, dtb, True, False)[2]
    assert tiny == pytest.approx(np.sqrt(6.0 * 1.0e-6 * dtb))


def test_displacement_per_step_variance_is_two_k_dt() -> None:
    dtb, kh, kv = 20.0, 0.03, 4.0e-4
    seed_lo, seed_hi = np.uint64(99), np.uint64(100)
    n = 100_000
    d = np.empty((n, 3))
    for c in range(n):
        ux, uy, uz = _counter_based_uniform_triple_scalar(seed_lo, seed_hi, np.uint64(c))
        d[c] = baytrace_uniform_diffusion_displacement(ux, uy, uz, kh, kv, 0.0, dtb, True, False)
    assert d[:, 0].var() == pytest.approx(2.0 * kh * dtb, rel=0.02)
    assert d[:, 1].var() == pytest.approx(2.0 * kh * dtb, rel=0.02)
    assert d[:, 2].var() == pytest.approx(2.0 * kv * dtb, rel=0.02)
    assert np.all(np.abs(d[:, 0]) <= np.sqrt(6.0 * kh * dtb))


# ══════════════════════════════════════════════════════════════════════════
# 3. 融合 kernel 內的統計行為
# ══════════════════════════════════════════════════════════════════════════


def test_horizontal_uniform_noise_variance_is_two_k_t(grid3, handles) -> None:
    kh, seconds, n = 0.05, 3600.0, 6000
    pa, x0, y0, counter = _run_kernel(
        grid3, handles, n=n, z0=-10.0, seconds=seconds, kh=kh, diffusion_noise="uniform",
    )
    assert np.all(pa.status == 1)
    assert np.all(counter == 60)  # 每個子步每顆粒子恰消耗一個計數器
    dx, dy = pa.x - x0, pa.y - y0
    expected = 2.0 * kh * seconds
    assert dx.var() == pytest.approx(expected, rel=0.08)
    assert dy.var() == pytest.approx(expected, rel=0.08)
    assert abs(dx.mean()) < 4.0 * np.sqrt(expected / n)
    assert abs(dy.mean()) < 4.0 * np.sqrt(expected / n)
    # 沒開垂向擴散：z 完全不動
    assert np.all(pa.z == -10.0)


@pytest.mark.parametrize("kz,seconds", [(3.0e-4, 3600.0), (2.0e-3, 900.0)])
def test_vertical_random_walk_variance_is_two_kv_t(grid3, handles, kz, seconds) -> None:
    n = 6000
    pa, _x0, _y0, _c = _run_kernel(
        grid3, handles, n=n, z0=-10.0, seconds=seconds, diffusion_noise="uniform",
        vertical_diffusion="constant", kz_constant_m2ps=kz,
    )
    dz = pa.z - (-10.0)
    expected = 2.0 * kz * seconds
    assert dz.var() == pytest.approx(expected, rel=0.08)
    assert abs(dz.mean()) < 4.0 * np.sqrt(expected / n)
    assert np.max(np.abs(dz)) <= 10.0  # 未碰到邊界，沒有夾制偏誤


def test_vertical_default_kz_is_baytrace_constant(grid3, handles) -> None:
    """不指定 ``kz_constant_m2ps`` 時預設 3.0e-4 m²/s（BayTrace ``kv_const`` 預設）。"""

    seconds, n = 3600.0, 6000
    pa, _x0, _y0, _c = _run_kernel(
        grid3, handles, n=n, z0=-10.0, seconds=seconds, diffusion_noise="uniform",
        vertical_diffusion="constant",
    )
    assert (pa.z + 10.0).var() == pytest.approx(2.0 * 3.0e-4 * seconds, rel=0.08)


def test_visser_correction_is_noop_for_constant_kv(grid3, handles) -> None:
    kwargs = dict(n=200, z0=-10.0, seconds=1800.0, diffusion_noise="uniform", vertical_diffusion="constant")
    plain, *_ = _run_kernel(grid3, handles, visser_correction=False, **kwargs)
    visser, *_ = _run_kernel(grid3, handles, visser_correction=True, **kwargs)
    assert np.array_equal(plain.z, visser.z)
    assert np.array_equal(plain.x, visser.x)


def test_vertical_displacement_is_clamped_to_water_column(grid3, handles) -> None:
    pa, *_ = _run_kernel(
        grid3, handles, n=2000, z0=-0.2, seconds=600.0, diffusion_noise="uniform",
        vertical_diffusion="constant", kz_constant_m2ps=5.0e-2,
    )
    assert np.all(pa.z <= 0.0 + 1.0e-12)
    assert np.all(pa.z >= -tsn._DP_M)
    assert not np.all(pa.z == -0.2)
    assert np.any(pa.z > -0.2)  # 有粒子被夾在或貼近表面，而不是全部往下散


def test_vertical_off_leaves_z_untouched_and_default_is_legacy(grid3, handles) -> None:
    uniform_off, *_ = _run_kernel(
        grid3, handles, n=100, z0=-7.0, seconds=600.0, diffusion_noise="uniform", vertical_diffusion="off",
    )
    assert np.all(uniform_off.z == -7.0)
    # 預設（不指定任何 WP-51 參數）＝舊高斯、無垂向：z 不動，水平變異數同為 2*K*t
    legacy, x0, _y0, _c = _run_kernel(grid3, handles, n=6000, z0=-7.0, seconds=1800.0)
    assert np.all(legacy.z == -7.0)
    assert (legacy.x - x0).var() == pytest.approx(2.0 * 0.05 * 1800.0, rel=0.08)


def test_uniform_noise_differs_from_gaussian_but_same_variance(grid3, handles) -> None:
    gauss, x0, _y0, _c = _run_kernel(grid3, handles, n=6000, z0=-7.0, seconds=1800.0, diffusion_noise="gaussian")
    unif, _x, _y, _c2 = _run_kernel(grid3, handles, n=6000, z0=-7.0, seconds=1800.0, diffusion_noise="uniform")
    assert not np.array_equal(gauss.x, unif.x)
    assert (gauss.x - x0).var() == pytest.approx((unif.x - x0).var(), rel=0.10)
    # 均勻雜訊的尾端有界：單一子步位移不超過 sqrt(6*K*dtb)；累積 60 步後的峰度接近 3（中央極限）
    dxu = unif.x - x0
    kurt = np.mean((dxu - dxu.mean()) ** 4) / dxu.var() ** 2
    assert 2.7 < kurt < 3.3


def test_substep_numba_rejects_invalid_diffusion_arguments(grid3, handles) -> None:
    base = dict(n=10, z0=-10.0, seconds=60.0, ndeltp=2)
    with pytest.raises(bd.BackwardDriverError, match="diffusion_noise"):
        _run_kernel(grid3, handles, diffusion_noise="laplace", **base)
    with pytest.raises(bd.BackwardDriverError, match="vertical_diffusion"):
        _run_kernel(grid3, handles, diffusion_noise="uniform", vertical_diffusion="bogus", **base)
    with pytest.raises(bd.BackwardDriverError, match="uniform"):
        _run_kernel(grid3, handles, diffusion_noise="gaussian", vertical_diffusion="constant", **base)
    with pytest.raises(bd.BackwardDriverError, match="kz_constant_m2ps"):
        _run_kernel(
            grid3, handles, diffusion_noise="uniform", vertical_diffusion="constant",
            kz_constant_m2ps=0.0, **base,
        )


# ══════════════════════════════════════════════════════════════════════════
# 4. ocm_field 介面
# ══════════════════════════════════════════════════════════════════════════


def _kv_fields(grid, per_level: list[float]):
    nvrt = int(grid.schism_grid.nvrt)
    n_nodes = tsn._n_nodes(grid)
    field = np.zeros((nvrt, n_nodes))
    for k, value in enumerate(per_level):
        field[k, :] = value
    return field


def test_ocm_field_requires_both_fields_with_correct_shape_and_finite(grid3, handles) -> None:
    base = dict(n=10, z0=-10.0, seconds=60.0, ndeltp=2, diffusion_noise="uniform", vertical_diffusion="ocm_field")
    good = _kv_fields(grid3, [2e-3, 2e-3, 2e-3])
    with pytest.raises(bd.BackwardDriverError, match="kv_field1"):
        _run_kernel(grid3, handles, **base)
    with pytest.raises(bd.BackwardDriverError, match="形狀"):
        _run_kernel(grid3, handles, kv_field1=good[:, :-1], kv_field2=good, **base)
    bad = good.copy()
    bad[0, 0] = np.nan
    with pytest.raises(bd.BackwardDriverError, match="非有限"):
        _run_kernel(grid3, handles, kv_field1=bad, kv_field2=good, **base)


def test_ocm_field_constant_field_gives_two_kv_t_variance(grid3, handles) -> None:
    kv, seconds, n = 2.0e-3, 900.0, 6000
    field = _kv_fields(grid3, [kv, kv, kv])
    pa, *_ = _run_kernel(
        grid3, handles, n=n, z0=-10.0, seconds=seconds, diffusion_noise="uniform",
        vertical_diffusion="ocm_field", kv_field1=field, kv_field2=field,
    )
    assert (pa.z + 10.0).var() == pytest.approx(2.0 * kv * seconds, rel=0.08)


def test_ocm_field_linear_kv_profile_gives_upward_gradient_drift(grid3, handles) -> None:
    """Kv 隨 z 增加（-20 m 1e-3 → 0 m 9e-3）時，隨機漫步需加 dKv/dz 漂移，平均位移應向上（正）。"""

    field = _kv_fields(grid3, [1.0e-3, 5.0e-3, 9.0e-3])  # dKv/dz = 4e-3 / 10 m = 4e-4 m/s
    seconds, n = 600.0, 20000
    pa, *_ = _run_kernel(
        grid3, handles, n=n, z0=-10.0, seconds=seconds, diffusion_noise="uniform",
        vertical_diffusion="ocm_field", kv_field1=field, kv_field2=field,
    )
    mean_dz = float((pa.z + 10.0).mean())
    expected = 4.0e-4 * seconds  # 0.24 m
    assert 0.6 * expected < mean_dz < 1.5 * expected, (mean_dz, expected)
    # 對照：常數 Kv 沒有漂移
    const = _kv_fields(grid3, [5.0e-3, 5.0e-3, 5.0e-3])
    pa2, *_ = _run_kernel(
        grid3, handles, n=n, z0=-10.0, seconds=seconds, diffusion_noise="uniform",
        vertical_diffusion="ocm_field", kv_field1=const, kv_field2=const,
    )
    assert abs(float((pa2.z + 10.0).mean())) < 0.08


# ══════════════════════════════════════════════════════════════════════════
# 5. 小時主軸與提供者
# ══════════════════════════════════════════════════════════════════════════


def _uniform_vertical_config(**overrides) -> hm.HourMajorConfig:
    params = dict(
        max_backtrack_seconds=3.0 * 3600.0, enable_diffusion=True, master_seed=99,
        diffusion_noise="uniform", vertical_diffusion="constant", kz_constant_m2ps=2.0e-3,
    )
    params.update(overrides)
    return hm.HourMajorConfig(**params)


def test_hour_major_vertical_diffusion_batch_composition_invariance(grid2, handles) -> None:
    table, units, interest_regions, _t0 = thm._build_staggered_domain()
    unit_a0, _unit_a1, unit_b0 = units
    resolver = thm._make_resolver(grid2, u=0.02, v=0.01, kh=2.0e-4)
    config = _uniform_vertical_config()
    kwargs = thm._flow_only_kwargs(interest_regions)

    def run(unit_list):
        batch = hm.build_hour_major_domain_batch(table, unit_list, interest_regions=interest_regions, **kwargs)
        return hm.run_flow_domain_hour_major(grid2, handles, batch, config=config, forcing_resolver=resolver)

    all_units = run(units)
    alone = run([unit_a0])
    pair = run([unit_a0, unit_b0])
    thm._assert_results_bit_identical(all_units.unit_results["A0"], alone.unit_results["A0"], label="A0 all/alone")
    thm._assert_results_bit_identical(all_units.unit_results["A0"], pair.unit_results["A0"], label="A0 all/pair")


def test_hour_major_vertical_diffusion_moves_z_while_legacy_does_not(grid2, handles) -> None:
    table, units, interest_regions, _t0 = thm._build_staggered_domain(n_per_unit=8)
    resolver = thm._make_resolver(grid2, u=0.02, v=0.01, kh=2.0e-4)
    kwargs = thm._flow_only_kwargs(interest_regions)
    batch = hm.build_hour_major_domain_batch(table, units, interest_regions=interest_regions, **kwargs)
    new = hm.run_flow_domain_hour_major(
        grid2, handles, batch, config=_uniform_vertical_config(), forcing_resolver=resolver,
    )
    old = hm.run_flow_domain_hour_major(
        grid2, handles, batch,
        config=hm.HourMajorConfig(max_backtrack_seconds=3.0 * 3600.0, enable_diffusion=True, master_seed=99),
        forcing_resolver=resolver,
    )
    for unit in units:
        z_new = new.unit_results[unit.unit_id].final_z_m
        z_old = old.unit_results[unit.unit_id].final_z_m
        assert np.all(np.isfinite(z_new))
        assert np.any(z_new != z_old), unit.unit_id
        assert np.all(z_new <= 0.0) and np.all(z_new >= -thm._DP_M)


def test_hour_major_scheme_validation_fails_closed() -> None:
    with pytest.raises(hm.HourMajorError, match="python"):
        hm.validate_diffusion_scheme(hm.HourMajorConfig(diffusion_noise="uniform", substep_loop_impl="python"))
    with pytest.raises(hm.HourMajorError, match="ocm_field"):
        hm.validate_diffusion_scheme(_uniform_vertical_config(vertical_diffusion="ocm_field"))
    with pytest.raises(hm.HourMajorError, match="uniform"):
        hm.validate_diffusion_scheme(hm.HourMajorConfig(vertical_diffusion="constant"))
    with pytest.raises(hm.HourMajorError, match="enable_diffusion"):
        hm.validate_diffusion_scheme(_uniform_vertical_config(enable_diffusion=False))
    with pytest.raises(hm.HourMajorError, match="kz_constant_m2ps"):
        hm.validate_diffusion_scheme(_uniform_vertical_config(kz_constant_m2ps=-1.0))
    with pytest.raises(hm.HourMajorError, match="diffusion_noise"):
        hm.validate_diffusion_scheme(hm.HourMajorConfig(diffusion_noise="laplace"))
    # 合法組合不丟例外
    hm.validate_diffusion_scheme(hm.HourMajorConfig())
    hm.validate_diffusion_scheme(_uniform_vertical_config())
    hm.validate_diffusion_scheme(hm.HourMajorConfig(diffusion_noise="uniform"))


def test_hour_major_library_default_is_legacy_scheme() -> None:
    config = hm.HourMajorConfig()
    assert config.diffusion_noise == "gaussian"
    assert config.vertical_diffusion == "off"
    assert config.kz_constant_m2ps == 3.0e-4
    assert config.visser_correction is False
    assert config.cfl_number == 0.5  # 實際子步上限＝0.5 × 局部網格邊長（設定檔 WP-51 起也寫 0.5）


def test_provider_env_defaults_choice_and_provenance() -> None:
    assert hmp._env_choice({}, "X", default="uniform", allowed=("uniform", "gaussian")) == "uniform"
    assert hmp._env_choice({"X": " Gaussian "}, "X", default="uniform", allowed=("uniform", "gaussian")) == "gaussian"
    with pytest.raises(hmp.HourMajorProviderError, match="X"):
        hmp._env_choice({"X": "laplace"}, "X", default="uniform", allowed=("uniform", "gaussian"))
    prov = hmp._diffusion_scheme_provenance(_uniform_vertical_config(kz_constant_m2ps=3.0e-4))
    assert prov == {
        "diffusion_noise": "uniform", "vertical_diffusion": "constant",
        "kz_constant_m2ps": 3.0e-4, "visser_correction": False,
    }
