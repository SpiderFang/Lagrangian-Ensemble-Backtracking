"""BayTrace 原始 CPU passive-advection adapter 的實源比較測試。"""

from __future__ import annotations

import hashlib
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from numba.core.registry import CPUDispatcher

from lagrangian_backtracking.baytrace_native_cpu import (
    BayTraceFieldPair,
    BayTraceGridArrays,
    BayTraceParticleArrays,
    load_original_cpu,
    step_native,
)


@pytest.fixture
def original_kernels(tmp_path: Path):
    """只有明確設定 LBT_BAYTRACE_ROOT 時才執行實源測試。"""

    raw_root = os.environ.get("LBT_BAYTRACE_ROOT")
    if not raw_root:
        pytest.skip("未設定 LBT_BAYTRACE_ROOT；略過需要 PI BayTrace 原始 artifact 的測試")
    source_root = Path(raw_root)
    if not source_root.exists():
        pytest.fail(f"LBT_BAYTRACE_ROOT 不存在: {source_root}")
    return load_original_cpu(source_root, tmp_path / "numba-cache")


def _grid() -> BayTraceGridArrays:
    """建立一個全程同面、正向且不含邊界穿越的最小三角網格。"""

    return BayTraceGridArrays(
        i34=np.array([3], dtype=np.int32),
        elnode=np.array([[0, 1, 2, -1]], dtype=np.int32),
        grid_x=np.array([0.0, 10.0, 0.0]),
        grid_y=np.array([0.0, 0.0, 10.0]),
        ic3=np.full((4, 1), -1, dtype=np.int32),
        nvrt=3,
        kbp=np.zeros(3, dtype=np.int32),
        kbe=np.zeros(1, dtype=np.int32),
        sigma_lcl=np.array([[-1.0, -1.0, -1.0], [-0.5, -0.5, -0.5], [0.0, 0.0, 0.0]]),
        dp=np.full(3, 10.0),
    )


def _fields() -> BayTraceFieldPair:
    """建立空間與時間均固定的速度場，便於檢查正反向位移。"""

    constant_u = np.full((3, 3), 2.0)
    constant_v = np.full((3, 3), -1.0)
    constant_w = np.full((3, 3), 0.25)
    return BayTraceFieldPair(
        u1=constant_u,
        v1=constant_v,
        w1=constant_w,
        eta1=np.zeros(3),
        u2=constant_u.copy(),
        v2=constant_v.copy(),
        w2=constant_w.copy(),
    )


@pytest.mark.parametrize("ibf", [1, -1])
def test_real_original_cpu_moves_in_both_directions_and_records_snapshots(original_kernels, ibf: int) -> None:
    """實際原始 dispatcher 在正、逆方向都使用正 dt 與 ibf 符號。"""

    grid = _grid()
    fields = _fields()
    particles = BayTraceParticleArrays.from_positions(
        x=np.array([2.0]),
        y=np.array([2.0]),
        z=np.array([-5.0]),
        elem=np.array([0], dtype=np.int32),
    )
    initial = np.array([2.0, 2.0, -5.0])

    step_native(original_kernels, grid, fields, particles, dt_seconds=0.5, trat=0.5, ibf=ibf)

    expected = initial + ibf * 0.5 * np.array([2.0, -1.0, 0.25])
    np.testing.assert_allclose([particles.x[0], particles.y[0], particles.z[0]], expected)
    np.testing.assert_allclose([particles.x0[0], particles.y0[0], particles.z0[0]], initial)
    np.testing.assert_allclose([particles.x_prev[0], particles.y_prev[0], particles.z_prev[0]], initial)
    np.testing.assert_allclose([particles.u[0], particles.v[0], particles.w[0]], [2.0, -1.0, 0.25])
    np.testing.assert_allclose(
        [particles.u_prev[0], particles.v_prev[0], particles.w_prev[0]],
        [0.0, 0.0, 0.0],
    )
    assert particles.elem[0] == 0
    assert original_kernels.batch_interpolate_velocity.signatures
    assert original_kernels.batch_move_rk4.signatures
    assert isinstance(original_kernels.batch_interpolate_velocity, CPUDispatcher)
    assert isinstance(original_kernels.batch_move_rk4, CPUDispatcher)


def test_inactive_particle_is_not_moved_by_original_kernels(original_kernels) -> None:
    """原始 status!=1 粒子保持位置，且內插輸出被原函式寫成零。"""

    particles = BayTraceParticleArrays.from_positions(
        x=np.array([2.0, 3.0]),
        y=np.array([2.0, 2.0]),
        z=np.array([-5.0, -4.0]),
        elem=np.array([0, -1], dtype=np.int32),
        status=np.array([1, 0], dtype=np.int32),
    )
    initial_inactive = np.array([particles.x[1], particles.y[1], particles.z[1]])

    step_native(original_kernels, _grid(), _fields(), particles, dt_seconds=0.5, trat=0.0, ibf=1)

    np.testing.assert_allclose([particles.x[1], particles.y[1], particles.z[1]], initial_inactive)
    np.testing.assert_allclose([particles.u[1], particles.v[1], particles.w[1]], [0.0, 0.0, 0.0])


def test_loader_records_source_hash_skips_init_and_rejects_conflict(original_kernels, tmp_path: Path) -> None:
    """來源 hash、外部 cache 與已存在的異來源 package 都有明確證據。"""

    provenance = original_kernels.provenance
    assert provenance.cache_dir == (tmp_path / "numba-cache").resolve()
    assert set(provenance.dependency_paths) == {"geometry", "vertical", "random", "batch_interp"}
    for name, path in provenance.dependency_paths.items():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert provenance.dependency_sha256[name] == digest
        assert path.is_absolute()
    assert sys.modules["baytrace"].__lbt_controlled_source__ == str(provenance.package_dir)
    assert not hasattr(sys.modules["baytrace"], "ParticleTracker")

    # loader 不應覆蓋其他來源的同名 package；測試在 finally 還原 process global。
    sentinel = types.ModuleType("baytrace")
    original_package = sys.modules["baytrace"]
    sys.modules["baytrace"] = sentinel
    try:
        with pytest.raises(ImportError, match="非本來源"):
            load_original_cpu(provenance.source_root, tmp_path / "other-cache")
    finally:
        sys.modules["baytrace"] = original_package


def test_validation_rejects_invalid_adjacency_before_native_call() -> None:
    """錯誤 adjacency 在建構階段被拒絕，不送入 Numba compiled code。"""

    with pytest.raises(ValueError, match="ic3"):
        BayTraceGridArrays(
            i34=np.array([3], dtype=np.int32),
            elnode=np.array([[0, 1, 2, -1]], dtype=np.int32),
            grid_x=np.array([0.0, 10.0, 0.0]),
            grid_y=np.array([0.0, 0.0, 10.0]),
            ic3=np.array([[9], [-1], [-1], [-1]], dtype=np.int32),
            nvrt=3,
            kbp=np.zeros(3, dtype=np.int32),
            kbe=np.zeros(1, dtype=np.int32),
            sigma_lcl=np.array([[-1.0] * 3, [-0.5] * 3, [0.0] * 3]),
            dp=np.full(3, 10.0),
        )
