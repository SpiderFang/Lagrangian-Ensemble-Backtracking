"""WP-54：逐粒子沉降速度（v7 ``class_candidates_balanced``）進入平流核心的驗收測試。

涵蓋：

1. **kernel 層級位元等價**：``settling_kernels`` 的逐粒子版本，在 ``effective_mask``
   部分為假、場為隨機值的情況下，輸出與「先把沉降速度烤進 ``w_field`` 再呼叫
   BayTrace 原版 kernel」逐位元相同（內插、單點內插、RK4 平流三者）。
2. **小時主軸等價**：整組粒子的逐粒子速度全部等於同一個值時，與烤進場的既有分組
   逐位元相同（擴散開／關、新到達與延續中混合小時）。
3. **混合速度等價**：同一分組內 4 種速度的粒子，結果與「各速度各自一個烤進場的
   分組」逐粒子位元相同。
4. **沉降速度真的進入垂向位移**：均勻流場、無擴散，回溯 2 小時的垂向位移等於
   ``|w|·t``，且依速度單調。
5. **續跑**：帶逐粒子速度的 checkpoint 續跑位元相同；速度指紋不符時拒絕。
6. **python 子步迴圈**搭配逐粒子速度 fail-closed。
"""

from __future__ import annotations

import numpy as np
import pytest

from lagrangian_backtracking.native_runner import backward_driver as bd
from lagrangian_backtracking.native_runner import hour_major as hm
from lagrangian_backtracking.native_runner.forcing_bridge import HourlyFieldPair
from lagrangian_backtracking.native_runner.settling_kernels import (
    FORKED_BATCH_INTERP_SHA256,
    build_settling_kernels,
)

from test_native_hour_major import (  # noqa: F401  (fixtures 需要匯入才會被 pytest 找到)
    _BIG_POLYGON,
    _DOMAIN_X,
    _DOMAIN_Y,
    _DP_M,
    _HOUR_NS,
    _assert_results_bit_identical,
    _FakeIdentityTable,
    _flow_only_kwargs,
    _make_unit,
    _n_nodes,
    _seed_hex_for,
    driver_grid,
    handles,
)

# ══════════════════════════════════════════════════════════════════════════
# 1. kernel 層級位元等價
# ══════════════════════════════════════════════════════════════════════════


def _random_inputs(driver_grid, seed: int):
    """隨機（但固定種子）的場、遮罩與落在各三角形內部的粒子。"""

    grid = driver_grid.schism_grid
    rng = np.random.default_rng(seed)
    nvrt, nnode = int(grid.nvrt), int(grid.x.shape[0])
    shape = (nvrt, nnode)
    fields = {
        name: rng.normal(0.0, 0.1, size=shape)
        for name in ("u1", "v1", "w1", "u2", "v2", "w2")
    }
    # 遮罩部分為假：模擬乾點與海床以下層，驗證只在 effective_mask 內加沉降速度。
    mask = rng.random(shape) > 0.25
    for key in ("w1", "w2"):
        fields[key] = np.where(mask, fields[key], 0.0)
    ne = int(grid.i34.shape[0])
    elems = np.arange(ne, dtype=np.int32)
    centroid_x = np.array([grid.x[grid.elnode[e, :3]].mean() for e in elems])
    centroid_y = np.array([grid.y[grid.elnode[e, :3]].mean() for e in elems])
    z = rng.uniform(-float(_DP_M), 0.0, size=ne)
    return fields, mask, elems, centroid_x, centroid_y, z


def test_forked_batch_interp_hash_matches_loaded_baytrace(handles) -> None:
    assert handles.dependency_sha256["batch_interp"] == FORKED_BATCH_INTERP_SHA256


@pytest.mark.parametrize("seed", [1, 2])
def test_interpolate_velocity_matches_baked_field_bit_identical(driver_grid, handles, seed) -> None:
    grid = driver_grid.schism_grid
    fields, mask, elems, px, py, pz = _random_inputs(driver_grid, seed)
    n = elems.shape[0]
    rng = np.random.default_rng(100 + seed)
    settle = rng.choice([-0.0003, -0.004, -0.03, -0.12, -0.4], size=n).astype(np.float64)
    eta = np.zeros(grid.x.shape[0], dtype=np.float64)
    kernels = build_settling_kernels(handles)

    out_new = [np.zeros(n) for _ in range(3)]
    kernels.batch_interpolate_velocity(
        px, py, pz, elems.copy(), np.ones(n, dtype=np.int32),
        grid.i34, grid.elnode, grid.x, grid.y,
        grid.nvrt, grid.kbp, grid.kbe, grid.sigma_lcl, grid.dp,
        fields["u1"], fields["v1"], fields["w1"], eta,
        fields["u2"], fields["v2"], fields["w2"],
        0.37, 0.0, *out_new, mask, settle,
    )

    # 對照：每顆粒子的速度各自烤進 w 場後，用 BayTrace 原版 kernel 單顆內插。
    for ip in range(n):
        w1 = fields["w1"].copy()
        w2 = fields["w2"].copy()
        w1[mask] += settle[ip]
        w2[mask] += settle[ip]
        out_ref = [np.zeros(1) for _ in range(3)]
        handles.batch_interpolate_velocity(
            px[ip : ip + 1], py[ip : ip + 1], pz[ip : ip + 1], elems[ip : ip + 1].copy(),
            np.ones(1, dtype=np.int32),
            grid.i34, grid.elnode, grid.x, grid.y,
            grid.nvrt, grid.kbp, grid.kbe, grid.sigma_lcl, grid.dp,
            fields["u1"], fields["v1"], w1, eta,
            fields["u2"], fields["v2"], w2,
            0.37, 0.0, *out_ref,
        )
        for new, ref in zip(out_new, out_ref, strict=True):
            assert new[ip] == ref[0], f"粒子 {ip} 內插結果與烤進場的原版不是逐位元相同"


def test_move_rk4_matches_baked_field_bit_identical(driver_grid, handles) -> None:
    grid = driver_grid.schism_grid
    fields, mask, elems, px, py, pz = _random_inputs(driver_grid, 5)
    n = elems.shape[0]
    rng = np.random.default_rng(55)
    settle = rng.choice([-0.0003, -0.004, -0.03, -0.12], size=n).astype(np.float64)
    eta = np.zeros(grid.x.shape[0], dtype=np.float64)
    kernels = build_settling_kernels(handles)
    status = np.ones(n, dtype=np.int32)

    def run(move, w1, w2, settle_args):
        x = px.copy()
        y = py.copy()
        z = pz.copy()
        p_u = np.full(n, 0.01)
        p_v = np.full(n, -0.02)
        p_w = np.full(n, -0.003)
        x_prev = x.copy()
        y_prev = y.copy()
        elem = elems.copy()
        x0, y0, z0 = x.copy(), y.copy(), z.copy()
        move(
            x, y, z, p_u, p_v, p_w, x_prev, y_prev, elem, status.copy(), x0, y0, z0,
            grid.i34, grid.elnode, grid.x, grid.y, grid.ic3,
            grid.nvrt, grid.kbp, grid.kbe, grid.sigma_lcl, grid.dp,
            fields["u1"], fields["v1"], w1, eta, fields["u2"], fields["v2"], w2,
            -1, 60.0, 0.4, *settle_args,
        )
        return x, y, z, elem

    new = run(kernels.batch_move_rk4, fields["w1"], fields["w2"], (mask, settle))
    for ip in range(n):
        w1 = fields["w1"].copy()
        w2 = fields["w2"].copy()
        w1[mask] += settle[ip]
        w2[mask] += settle[ip]
        # 單顆對照：只讓第 ip 顆 status==1。
        status_one = np.zeros(n, dtype=np.int32)
        status_one[ip] = 1
        x = px.copy(); y = py.copy(); z = pz.copy()
        p_u = np.full(n, 0.01); p_v = np.full(n, -0.02); p_w = np.full(n, -0.003)
        x_prev = x.copy(); y_prev = y.copy(); elem = elems.copy()
        x0, y0, z0 = x.copy(), y.copy(), z.copy()
        handles.batch_move_rk4(
            x, y, z, p_u, p_v, p_w, x_prev, y_prev, elem, status_one, x0, y0, z0,
            grid.i34, grid.elnode, grid.x, grid.y, grid.ic3,
            grid.nvrt, grid.kbp, grid.kbe, grid.sigma_lcl, grid.dp,
            fields["u1"], fields["v1"], w1, eta, fields["u2"], fields["v2"], w2,
            -1, 60.0, 0.4,
        )
        assert new[0][ip] == x[ip] and new[1][ip] == y[ip] and new[2][ip] == z[ip], (
            f"粒子 {ip} 的 RK4 結果與烤進場的原版不是逐位元相同"
        )


# ══════════════════════════════════════════════════════════════════════════
# 小時主軸共用工具
# ══════════════════════════════════════════════════════════════════════════


class _FakeSettlingIdentityTable(_FakeIdentityTable):
    """多一個 ``particle_settling_velocity_mps`` 欄位的假身分表。"""

    def __init__(self, base: _FakeIdentityTable, settling: np.ndarray | None) -> None:
        super().__init__(base.x0_m, base.y0_m, base.z0_m, base.seed_hex)
        self.particle_settling_velocity_mps = settling


def _sheared_resolver(driver_grid, *, u: float, shear: float, w: float = 0.0, kh: float = 1.0e-4):
    """水平流速隨垂向層變化（有剪切）的合成流場，讓沉降造成的深度改變會影響水平位移。"""

    n_nodes = _n_nodes(driver_grid)
    nvrt = int(driver_grid.schism_grid.nvrt)
    shape = (nvrt, n_nodes)

    def resolver(target_utc_ns: int) -> HourlyFieldPair:
        u_field = np.zeros(shape)
        for level in range(nvrt):
            u_field[level, :] = u + shear * level
        v_field = np.full(shape, 0.004)
        w_field = np.full(shape, w)
        eta = np.zeros(n_nodes)
        diffusivity = np.full(shape, kh)
        missing = np.zeros(shape, dtype=bool)
        dry = np.zeros(shape, dtype=bool)
        return HourlyFieldPair(
            time_before_utc_ns=target_utc_ns, time_after_utc_ns=target_utc_ns - _HOUR_NS, trat=0.0, ibf=-1,
            u_field1=u_field.copy(), v_field1=v_field.copy(), w_field1=w_field.copy(),
            eta_field1=eta.copy(), diffusivity_field1=diffusivity.copy(),
            missing_mask1=missing.copy(), dry_face_flag1=dry.copy(), month_id1="synthetic", time_origin1="test",
            u_field2=u_field.copy(), v_field2=v_field.copy(), w_field2=w_field.copy(),
            eta_field2=eta.copy(), diffusivity_field2=diffusivity.copy(),
            missing_mask2=missing.copy(), dry_face_flag2=dry.copy(), month_id2="synthetic", time_origin2="test",
        )

    return resolver


def _staggered_inputs(speeds: np.ndarray, *, class_speed: float, z0: float = -12.0):
    """兩個到達時刻（T0 與 T0-1hr）各 len(speeds)//2 顆粒子，同一個類別分組。"""

    n_total = speeds.shape[0]
    assert n_total % 2 == 0
    half = n_total // 2
    t0 = 20_000 * _HOUR_NS
    rng = np.random.default_rng(31)
    x0 = _DOMAIN_X / 2.0 + rng.uniform(-60.0, 60.0, size=n_total)
    y0 = _DOMAIN_Y / 2.0 + rng.uniform(-60.0, 60.0, size=n_total)
    z = np.full(n_total, z0)
    row_ids = np.arange(n_total, dtype=np.int64)
    base = _FakeIdentityTable(x0_m=x0, y0_m=y0, z0_m=z, seed_hex=_seed_hex_for(row_ids))
    units = [
        _make_unit(unit_id="U0", region="R", domain="D", site="S1", arrival_utc=t0,
                   settling=class_speed, rows=np.arange(0, half)),
        _make_unit(unit_id="U1", region="R", domain="D", site="S1", arrival_utc=t0 - _HOUR_NS,
                   settling=class_speed, rows=np.arange(half, n_total)),
    ]
    return base, units, {"S1": _BIG_POLYGON}


def _run(driver_grid, handles, table, units, interest_regions, resolver, config, **kwargs):
    batch = hm.build_hour_major_domain_batch(
        table, units, interest_regions=interest_regions, **_flow_only_kwargs(interest_regions)
    )
    return hm.run_flow_domain_hour_major(
        driver_grid, handles, batch, config=config, forcing_resolver=resolver, **kwargs
    )


# ══════════════════════════════════════════════════════════════════════════
# 2. 所有粒子速度相同 → 與烤進場分組逐位元相同
# ══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("enable_diffusion", [False, True])
def test_uniform_per_particle_speed_matches_baked_group(driver_grid, handles, enable_diffusion) -> None:
    speed = -0.012
    speeds = np.full(6, speed)
    base, units, regions = _staggered_inputs(speeds, class_speed=speed)
    resolver = _sheared_resolver(driver_grid, u=0.02, shear=0.03, w=0.0004)
    config = hm.HourMajorConfig(
        max_backtrack_seconds=3.0 * 3600.0, enable_diffusion=enable_diffusion, master_seed=77,
    )

    baked = _run(driver_grid, handles, _FakeSettlingIdentityTable(base, None), units, regions, resolver, config)
    per_particle = _run(
        driver_grid, handles, _FakeSettlingIdentityTable(base, speeds.copy()), units, regions, resolver, config
    )
    for unit_id in ("U0", "U1"):
        _assert_results_bit_identical(
            baked.unit_results[unit_id], per_particle.unit_results[unit_id], label=unit_id
        )
    # 兩種路徑每個分組每小時都只有一次（或拆成新到達／延續中兩次）kernel 呼叫，數量相同。
    assert baked.domain_kernel_call_count == per_particle.domain_kernel_call_count


# ══════════════════════════════════════════════════════════════════════════
# 3. 同一分組內混合 4 種速度 → 與各速度各自的烤進場分組逐粒子位元相同
# ══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("enable_diffusion", [False, True])
def test_mixed_speeds_match_separate_baked_groups_per_particle(driver_grid, handles, enable_diffusion) -> None:
    candidates = np.array([-0.002, -0.01, -0.03, -0.08])
    speeds = np.tile(candidates, 3)  # 12 顆，每種速度 3 顆，兩個到達時刻各 6 顆
    class_speed = -0.0185  # 類別代表速度：分組鍵，不參與位移
    base, units, regions = _staggered_inputs(speeds, class_speed=class_speed)
    resolver = _sheared_resolver(driver_grid, u=0.02, shear=0.03, w=0.0004)
    config = hm.HourMajorConfig(
        max_backtrack_seconds=3.0 * 3600.0, enable_diffusion=enable_diffusion, master_seed=78,
    )

    mixed = _run(
        driver_grid, handles, _FakeSettlingIdentityTable(base, speeds.copy()), units, regions, resolver, config
    )

    # 對照組：依（到達時刻, 速度）切成多個單元，各自烤進場（既有路徑）。
    ref_units = []
    ref_index: dict[str, tuple[str, int]] = {}
    for unit in units:
        rows = unit.particle_row_indices
        for speed in candidates:
            sub_rows = rows[speeds[rows] == speed]
            sub_id = f"{unit.unit_id}_{speed}"
            ref_units.append(
                _make_unit(unit_id=sub_id, region="R", domain="D", site="S1",
                           arrival_utc=unit.arrival_time_utc_ns, settling=float(speed), rows=sub_rows)
            )
            for local, row in enumerate(sub_rows):
                ref_index[f"{unit.unit_id}:{local_to_global(unit, row)}"] = (sub_id, local)
    reference = _run(
        driver_grid, handles, _FakeSettlingIdentityTable(base, None), ref_units, regions, resolver, config
    )

    checked = 0
    for unit in units:
        result = mixed.unit_results[unit.unit_id]
        for local, row in enumerate(unit.particle_row_indices):
            sub_id, sub_local = ref_index[f"{unit.unit_id}:{local_to_global(unit, row)}"]
            ref = reference.unit_results[sub_id]
            assert result.final_x_m[local] == ref.final_x_m[sub_local]
            assert result.final_y_m[local] == ref.final_y_m[sub_local]
            assert result.final_z_m[local] == ref.final_z_m[sub_local]
            assert result.final_status[local] == ref.final_status[sub_local]
            assert np.array_equal(result.trajectory_z_m[:, local], ref.trajectory_z_m[:, sub_local])
            checked += 1
    assert checked == speeds.shape[0]


def local_to_global(unit, row) -> int:
    return int(row)


# ══════════════════════════════════════════════════════════════════════════
# 4. 沉降速度真的進入垂向位移（均勻流場、無擴散）
# ══════════════════════════════════════════════════════════════════════════


def test_per_particle_speed_sets_vertical_displacement(driver_grid, handles) -> None:
    candidates = np.array([-0.0003, -0.0006, -0.0009, -0.0012])
    speeds = np.tile(candidates, 2)
    base, units, regions = _staggered_inputs(speeds, class_speed=-0.0007, z0=-12.0)
    # 兩個到達時刻都只走 2 小時（回溯上限 2 小時），避免碰到海面夾制（最大位移 8.64 m < 12 m）。
    resolver = _sheared_resolver(driver_grid, u=0.0, shear=0.0, w=0.0)
    config = hm.HourMajorConfig(max_backtrack_seconds=2.0 * 3600.0, enable_diffusion=False, master_seed=5)
    result = _run(
        driver_grid, handles, _FakeSettlingIdentityTable(base, speeds.copy()), units, regions, resolver, config
    )
    for unit in units:
        res = result.unit_results[unit.unit_id]
        unit_speeds = speeds[unit.particle_row_indices]
        # 逆向追蹤：正向向下沉（w<0）在回溯時間軸上等於向上浮，位移為 +|w|·t。
        expected = -12.0 + np.abs(unit_speeds) * 2.0 * 3600.0
        assert np.allclose(res.final_z_m, expected, atol=1.0e-6), (res.final_z_m, expected)
        order = np.argsort(np.abs(unit_speeds))
        assert np.all(np.diff(res.final_z_m[order]) > 0.0)


# ══════════════════════════════════════════════════════════════════════════
# 5. 續跑
# ══════════════════════════════════════════════════════════════════════════


def test_checkpoint_resume_with_per_particle_speed_is_bit_identical(driver_grid, handles, tmp_path) -> None:
    candidates = np.array([-0.002, -0.01, -0.03, -0.08])
    speeds = np.tile(candidates, 3)
    base, units, regions = _staggered_inputs(speeds, class_speed=-0.0185)
    table = _FakeSettlingIdentityTable(base, speeds.copy())
    resolver = _sheared_resolver(driver_grid, u=0.02, shear=0.03, w=0.0004)
    config = hm.HourMajorConfig(max_backtrack_seconds=3.0 * 3600.0, enable_diffusion=True, master_seed=9)

    full = _run(driver_grid, handles, table, units, regions, resolver, config)
    checkpoint_dir = tmp_path / "ckpt"
    partial = _run(
        driver_grid, handles, table, units, regions, resolver, config,
        checkpoint_dir=checkpoint_dir, checkpoint_every_hours=1, _max_master_hours=1,
    )
    assert partial.domain_hours_walked == 1
    resumed = _run(
        driver_grid, handles, table, units, regions, resolver, config,
        checkpoint_dir=checkpoint_dir, checkpoint_every_hours=1,
    )
    for unit_id in full.unit_results:
        _assert_results_bit_identical(full.unit_results[unit_id], resumed.unit_results[unit_id], label=unit_id)

    # 逐粒子速度不同（指紋不符）時必須拒絕續跑。
    other_speeds = speeds.copy()
    other_speeds[0] = -0.05
    with pytest.raises(hm.HourMajorError, match="逐粒子沉降速度"):
        _run(
            driver_grid, handles, _FakeSettlingIdentityTable(base, other_speeds), units, regions, resolver,
            config, checkpoint_dir=checkpoint_dir, checkpoint_every_hours=1,
        )
    # 原本有逐粒子速度的 checkpoint 不可被沒有逐粒子速度的輸入續跑。
    with pytest.raises(hm.HourMajorError, match="逐粒子沉降速度"):
        _run(
            driver_grid, handles, _FakeSettlingIdentityTable(base, None), units, regions, resolver,
            config, checkpoint_dir=checkpoint_dir, checkpoint_every_hours=1,
        )


# ══════════════════════════════════════════════════════════════════════════
# 6. python 子步迴圈不支援逐粒子速度
# ══════════════════════════════════════════════════════════════════════════


def test_python_substep_loop_rejects_per_particle_speed(driver_grid, handles) -> None:
    speeds = np.array([-0.002, -0.01, -0.03, -0.08])
    base, units, regions = _staggered_inputs(speeds, class_speed=-0.0185)
    resolver = _sheared_resolver(driver_grid, u=0.02, shear=0.03)
    config = hm.HourMajorConfig(
        max_backtrack_seconds=3600.0, enable_diffusion=False, master_seed=1, substep_loop_impl="python",
    )
    with pytest.raises(hm.HourMajorError, match="numba"):
        _run(driver_grid, handles, _FakeSettlingIdentityTable(base, speeds), units, regions, resolver, config)
