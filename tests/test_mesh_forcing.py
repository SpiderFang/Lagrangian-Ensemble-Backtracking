"""SCHISM 拓撲、OCM 4D 與 NWW circular interpolation 測試。"""

from __future__ import annotations

import numpy as np

from lagrangian_backtracking.forcing import NWWAnalysisMonth, OCMNativeMonth
from lagrangian_backtracking.mesh import NativeMesh, triangulate_faces
from lagrangian_backtracking.models import SampleQC


def _triangle_mesh() -> NativeMesh:
    """建立一個 10×10 m 直角三角形，node lon/lat 只作占位 provenance。"""

    return NativeMesh(
        node_lon=np.array([121.0, 121.001, 121.0]),
        node_lat=np.array([25.0, 25.0, 25.001]),
        node_xy=np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]]),
        source_depth_m=np.array([10.0, 10.0, 10.0]),
        source_node_bottom_index=np.array([0, 0, 0]),
        face_nodes_local=np.array([[0, 1, 2, -1]]),
        face_node_count=np.array([3]),
        source_face_global_index=np.array([99]),
        bin_size_m=20.0,
    )


def test_quad_split_uses_shorter_diagonal_and_preserves_face() -> None:
    """非對稱 quad 應依較短對角線切成兩面，兩者都指回同一來源 face。"""

    xy = np.array([[0.0, 0.0], [3.0, 0.0], [2.0, 1.0], [0.0, 2.0]])
    triangles, source = triangulate_faces(np.array([[0, 1, 2, 3]]), np.array([4]), xy)
    assert triangles.shape == (2, 3)
    assert np.array_equal(source, [0, 0])
    assert all(0 in triangle and 2 in triangle for triangle in triangles)


def test_mesh_locator_returns_barycentric_weights_and_global_face() -> None:
    """triangle 內點的權重總和為 1，域外不得最近鄰外插。"""

    mesh = _triangle_mesh()
    location = mesh.locate(2.0, 3.0)
    assert location is not None
    assert location.source_face_global_index == 99
    assert np.allclose(location.barycentric_weights, [0.5, 0.2, 0.3])
    assert mesh.locate(9.0, 9.0) is None


def test_ocm_sampler_is_exact_for_linear_xyzt_field() -> None:
    """垂向、水平、時間皆線性的合成場應被 OCM sampler 精確重建。"""

    mesh = _triangle_mesh()
    times = np.array([0, 10_000_000_000], dtype=np.int64)
    z_levels = np.array([-10.0, -5.0, 0.0])
    zcor = np.broadcast_to(z_levels, (2, 3, 3)).copy()
    hvel = np.empty((2, 3, 3, 2), dtype=np.float64)
    w = np.empty((2, 3, 3), dtype=np.float64)
    kz = np.empty_like(w)
    for time_index, seconds in enumerate([0.0, 10.0]):
        for node, (x_m, y_m) in enumerate(mesh.node_xy):
            for layer, z_m in enumerate(z_levels):
                hvel[time_index, node, layer, 0] = 0.01 * x_m + 0.1 * z_m + 0.2 * seconds
                hvel[time_index, node, layer, 1] = 0.02 * y_m - 0.05 * z_m
                w[time_index, node, layer] = 0.001 * z_m
                kz[time_index, node, layer] = 0.01
    sampler = OCMNativeMonth(
        month_id="197001",
        mesh=mesh,
        time_utc_ns=times,
        hvel=hvel,
        vertical_velocity=w,
        zcor=zcor,
        elev=np.zeros((2, 3)),
        wetdry_elem=np.zeros((2, 1)),
        diffusivity=kz,
        maximum_time_gap_seconds=20.0,
    )
    sample = sampler.sample(2.0, 3.0, -7.5, 5_000_000_000)
    assert sample.valid
    assert np.isclose(sample.u_mps, 0.02 - 0.75 + 1.0)
    assert np.isclose(sample.v_mps, 0.06 + 0.375)
    assert np.isclose(sample.w_mps, -0.0075)
    assert np.isclose(sample.diagnostics["kz_m2ps"], 0.01)

    accelerated = OCMNativeMonth(
        month_id="197001",
        mesh=mesh,
        time_utc_ns=times,
        hvel=hvel,
        vertical_velocity=w,
        zcor=zcor,
        elev=np.zeros((2, 3)),
        wetdry_elem=np.zeros((2, 1)),
        diffusivity=kz,
        maximum_time_gap_seconds=20.0,
        use_numba_kernel=True,
    ).sample(2.0, 3.0, -7.5, 5_000_000_000)
    assert accelerated.valid
    assert np.allclose(
        [accelerated.u_mps, accelerated.v_mps, accelerated.w_mps, accelerated.diagnostics["kz_m2ps"]],
        [sample.u_mps, sample.v_mps, sample.w_mps, sample.diagnostics["kz_m2ps"]],
        rtol=1e-12,
        atol=1e-12,
    )


def test_ocm_sampler_rejects_dry_face() -> None:
    """wetdry 非 wet value 時不得回傳零流速冒充有效場。"""

    mesh = _triangle_mesh()
    arrays = np.zeros((2, 3, 2))
    sampler = OCMNativeMonth(
        month_id="197001",
        mesh=mesh,
        time_utc_ns=np.array([0, 1_000_000_000], dtype=np.int64),
        hvel=np.zeros((2, 3, 2, 2)),
        vertical_velocity=arrays,
        zcor=np.broadcast_to(np.array([-10.0, 0.0]), (2, 3, 2)),
        elev=np.zeros((2, 3)),
        wetdry_elem=np.ones((2, 1)),
        diffusivity=arrays,
    )
    sample = sampler.sample(2.0, 2.0, -5.0, 0)
    assert not sample.valid
    assert sample.qc & SampleQC.DRY_FACE


def test_nww_direction_uses_circular_interpolation() -> None:
    """350° 與 10° 空間平均應接近 0°，不能線性變成 180°。"""

    shape = (2, 2, 2)
    directions = np.array([[[350.0, 10.0], [350.0, 10.0]], [[350.0, 10.0], [350.0, 10.0]]])
    sampler = NWWAnalysisMonth(
        month_id="202401",
        lon=np.array([121.0, 122.0]),
        lat=np.array([24.0, 25.0]),
        time_utc_ns=np.array([0, 3_600_000_000_000], dtype=np.int64),
        significant_wave_height=np.full(shape, 2.0),
        peak_frequency=np.full(shape, 0.125),
        peak_direction_raw_deg=directions,
        valid_mask_wave=np.ones(shape, dtype=bool),
        qc_flags=np.zeros(shape, dtype=np.uint16),
    )
    sample = sampler.sample(121.5, 24.5, 1_800_000_000_000)
    assert sample.valid
    assert min(abs(sample.peak_direction_raw_deg), abs(sample.peak_direction_raw_deg - 360.0)) < 1e-10
