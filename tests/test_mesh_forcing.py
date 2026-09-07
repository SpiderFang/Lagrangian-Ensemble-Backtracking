"""SCHISM 拓撲、OCM 4D 與 NWW circular interpolation 測試。"""

from __future__ import annotations

import numpy as np
import pytest

from lagrangian_backtracking.diffusion import SmagorinskySettings, smagorinsky_horizontal_diffusivity
from lagrangian_backtracking.forcing import (
    CombinedMonthForcing,
    MonthlyCombinedForcing,
    NWWAnalysisMonth,
    OCMNativeMonth,
)
from lagrangian_backtracking.geometry import DomainProjection
from lagrangian_backtracking.mesh import NativeMesh, _build_triangle_neighbors, triangulate_faces
from lagrangian_backtracking.models import SURFACE_BOUNDARY_TOLERANCE_M, SampleQC
from lagrangian_backtracking.stokes import finite_depth_stokes


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


def test_native_mesh_shape_gradient_is_exact_for_linear_scalar() -> None:
    """任意線性 scalar 在公尺制三角形上應精確回傳其 x/y 係數。"""

    mesh = _triangle_mesh()
    coefficients = (2.5, -1.25, 9.0)
    scalar_values = np.asarray(
        [coefficients[0] * x + coefficients[1] * y + coefficients[2] for x, y in mesh.node_xy]
    )
    assert mesh.triangle_shape_gradients.shape == (1, 3, 2)
    assert np.allclose(mesh.triangle_linear_gradient(0, scalar_values), coefficients[:2])
    assert np.allclose(mesh.triangle_scalar_gradient(0, scalar_values), coefficients[:2])
    assert np.allclose(mesh.triangle_shape_gradients.sum(axis=1), 0.0)


def test_native_mesh_incident_adjacency_is_sorted_and_immutable() -> None:
    """node-to-triangle adjacency 應固定排序、保留孤立 node 空 tuple 並不可 append。"""

    mesh = _adjacent_mesh()
    assert mesh.node_incident_triangles == ((0, 1), (0,), (0, 1), (1,))
    assert mesh.node_to_incident_triangles is mesh.node_incident_triangles
    assert mesh.incident_triangles(2) == (0, 1)
    with pytest.raises(TypeError):
        mesh.node_incident_triangles[0] += (3,)
    with pytest.raises(IndexError):
        mesh.incident_triangles(99)


def test_native_mesh_gradient_rejects_invalid_triangle_or_scalar_inputs() -> None:
    """triangle ID、local scalar shape 與非有限值錯誤都必須在梯度計算前拒絕。"""

    mesh = _triangle_mesh()
    with pytest.raises(IndexError):
        mesh.triangle_linear_gradient(99, [0.0, 0.0, 0.0])
    with pytest.raises(TypeError):
        mesh.triangle_linear_gradient(True, [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match=r"shape=\(3,\)"):
        mesh.triangle_linear_gradient(0, [0.0, 0.0])
    with pytest.raises(ValueError, match="全部有限"):
        mesh.triangle_linear_gradient(0, [0.0, np.nan, 0.0])


def test_native_mesh_rejects_degenerate_triangle_for_shape_gradient() -> None:
    """退化三角形不應產生除以零的形函數梯度。"""

    with pytest.raises(ValueError, match="退化"):
        NativeMesh(
            node_lon=np.array([121.0, 121.001, 121.002]),
            node_lat=np.array([25.0, 25.0, 25.0]),
            node_xy=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
            source_depth_m=np.full(3, 10.0),
            source_node_bottom_index=np.zeros(3, dtype=np.int64),
            face_nodes_local=np.array([[0, 1, 2, -1]]),
            face_node_count=np.array([3]),
            source_face_global_index=np.array([1]),
        )


def _moving_surface_ocm_sampler(
    *,
    use_numba_kernel: bool = False,
    missing_eta: bool = False,
    missing_top_field: str | None = None,
) -> OCMNativeMonth:
    """建立固定 z 查詢會遇到移動海面支援洞的 OCM synthetic month。

    before endpoint 的最高 ``zcor`` 為 1.77 m，after endpoint 降至 1.47 m；查詢時刻
    ``z=1.683 m`` 位於兩端高度之間，並選在 query-time eta 約 1.726 m 的時刻。這讓
    同一固定 z 在 before 仍可由中層與 top layer 內插、在 after 則必須使用 top-layer
    surface hold，正好覆蓋 SERVER 診斷的移動海面案例。所有速度分量與 Kz 的單位分別
    是 m/s 與 m²/s；三個 node 使用相同數值，測試重點是垂向支援與時間 gate，不是
    triangle 水平梯度。

    ``missing_eta`` 模擬 endpoint 海面缺值；``missing_top_field`` 模擬 after top
    layer 的必要物理量缺值。兩者都必須 fail closed，不能退回更深 layer 或補零。
    """

    mesh = _triangle_mesh()
    times = np.array([0, 1_000_000_000], dtype=np.int64)
    zcor = np.empty((2, 3, 3), dtype=np.float64)
    zcor[0, :, :] = np.array([-10.0, -5.0, 1.77])
    zcor[1, :, :] = np.array([-10.0, -5.0, 1.47])
    hvel = np.zeros((2, 3, 3, 2), dtype=np.float64)
    vertical_velocity = np.zeros((2, 3, 3), dtype=np.float64)
    diffusivity = np.zeros((2, 3, 3), dtype=np.float64)
    # 中層值用來計算 before endpoint 的合法雙側內插；最高層值用來驗證 after hold。
    hvel[:, :, 1, :] = np.array([0.5, -0.5])
    vertical_velocity[:, :, 1] = 0.05
    diffusivity[:, :, 1] = 0.02
    hvel[0, :, 2, :] = np.array([1.0, 2.0])
    hvel[1, :, 2, :] = np.array([3.0, 4.0])
    vertical_velocity[0, :, 2] = 0.10
    vertical_velocity[1, :, 2] = 0.20
    diffusivity[0, :, 2] = 0.03
    diffusivity[1, :, 2] = 0.04
    elev = np.empty((2, 3), dtype=np.float64)
    elev[0, :] = 1.77
    elev[1, :] = 1.47
    if missing_eta:
        elev[1, 0] = np.nan
    if missing_top_field == "hvel":
        hvel[1, 0, 2, 0] = np.nan
    elif missing_top_field == "vertical_velocity":
        vertical_velocity[1, 0, 2] = np.nan
    elif missing_top_field == "diffusivity":
        diffusivity[1, 0, 2] = np.nan
    elif missing_top_field is not None:
        raise ValueError(f"未知的 synthetic top 欄位：{missing_top_field}")
    return OCMNativeMonth(
        month_id="197001",
        mesh=mesh,
        time_utc_ns=times,
        hvel=hvel,
        vertical_velocity=vertical_velocity,
        zcor=zcor,
        elev=elev,
        wetdry_elem=np.zeros((2, 1), dtype=np.float64),
        diffusivity=diffusivity,
        maximum_time_gap_seconds=2.0,
        use_numba_kernel=use_numba_kernel,
    )


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


@pytest.mark.parametrize("use_numba_kernel", [False, True])
def test_ocm_vertical_unsupported_preserves_finite_geometric_bounds(
    use_numba_kernel: bool,
) -> None:
    """垂向速度不支援時保留可獨立證明的 eta／bed，卻不產生可用速度。

    查詢深度 ``z_m=1`` 高於合成海面 ``eta_m=0``，故速度不能由 OCM 上下兩層夾住，
    必須回傳 ``VERTICAL_UNSUPPORTED``。海面來自 ``elev``、海床來自 native mesh
    ``source_depth_m``，兩者仍是有限且相容的幾何證據；此欄位讓 engine 能辨認海面
    反射，但不會把 invalid sample 的零速度誤當成可積分速度。NumPy 與 Numba 路徑都
    必須維持同一缺值／上下界契約。
    """

    mesh = _triangle_mesh()
    times = np.array([0, 1_000_000_000], dtype=np.int64)
    z_levels = np.array([-10.0, 0.0])
    zcor = np.broadcast_to(z_levels, (2, 3, 2)).copy()
    hvel = np.zeros((2, 3, 2, 2), dtype=np.float64)
    vertical_velocity = np.zeros((2, 3, 2), dtype=np.float64)
    diffusivity = np.full((2, 3, 2), 0.01, dtype=np.float64)
    sampler = OCMNativeMonth(
        month_id="197001",
        mesh=mesh,
        time_utc_ns=times,
        hvel=hvel,
        vertical_velocity=vertical_velocity,
        zcor=zcor,
        elev=np.zeros((2, 3), dtype=np.float64),
        wetdry_elem=np.zeros((2, 1), dtype=np.float64),
        diffusivity=diffusivity,
        use_numba_kernel=use_numba_kernel,
    )

    sample = sampler.sample(2.0, 3.0, 1.0, 0)
    assert not sample.valid
    assert sample.qc == SampleQC.VERTICAL_UNSUPPORTED
    assert sample.eta_m == 0.0
    assert sample.bed_z_m == -10.0
    assert sample.u_mps == sample.v_mps == sample.w_mps == 0.0


@pytest.mark.parametrize("use_numba_kernel", [False, True])
def test_ocm_moving_surface_uses_endpoint_surface_hold_and_query_time_gate(
    use_numba_kernel: bool,
) -> None:
    """移動海面造成的 endpoint 支援洞應由 top hold 修復，再由 query-time eta 把關。

    查詢時間的 alpha 約為 0.146666667，因此 eta 約為 1.726 m；固定 z=1.683 m
    在 query-time 水柱內。before endpoint 的 z=1.683 m 仍落在 -5 m 與 1.77 m
    之間，after endpoint 則高於 1.47 m top layer，必須使用 after top-layer 值。
    測試同時比較一般 NumPy reference 與 Numba kernel，包含速度、Kz、幾何上下界與
    垂向尺度，確保 surface hold 沒有只修到單一路徑。
    """

    sampler = _moving_surface_ocm_sampler(use_numba_kernel=use_numba_kernel)
    query_time_ns = 146_666_667
    query_alpha = query_time_ns / 1_000_000_000.0
    target_z_m = 1.683
    before_vertical_alpha = (target_z_m + 5.0) / (1.77 + 5.0)
    before_values = np.array(
        [
            0.5 + before_vertical_alpha * (1.0 - 0.5),
            -0.5 + before_vertical_alpha * (2.0 + 0.5),
            0.05 + before_vertical_alpha * (0.10 - 0.05),
            0.02 + before_vertical_alpha * (0.03 - 0.02),
        ]
    )
    after_values = np.array([3.0, 4.0, 0.20, 0.04])
    expected_values = before_values + query_alpha * (after_values - before_values)
    expected_eta = 1.77 + query_alpha * (1.47 - 1.77)

    sample = sampler.sample(2.0, 3.0, target_z_m, query_time_ns)

    assert sample.valid
    assert np.allclose(
        [sample.u_mps, sample.v_mps, sample.w_mps, sample.diagnostics["kz_m2ps"]],
        expected_values,
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.isclose(sample.eta_m, expected_eta)
    assert sample.bed_z_m == -10.0
    # after endpoint 全部使用 surface hold，故該端點尺度回到保守 0.1 m。
    assert sample.vertical_scale_m == 0.1


@pytest.mark.parametrize("use_numba_kernel", [False, True])
def test_ocm_surface_boundary_residual_clamps_query_but_preserves_endpoint_top_support(
    use_numba_kernel: bool,
) -> None:
    """同一海面容許尺度必須同時修正 query gate 與 endpoint top-layer 支援。

    第一個案例把 after endpoint 的 ``eta`` 設在最高 ``zcor`` 上方約 3.368 微米，
    對應 checkpoint-8 的最大移動海面邊界定位數值殘差；若 top-layer 支援仍使用舊的 1 微米門檻，
    即使 query z 已夾回 eta 也會回傳 ``VERTICAL_UNSUPPORTED``。第二個案例把 query z
    放在 query-time eta 上方但仍在 5 微米容許帶內，驗證最後的幾何 gate 與 NumPy／Numba
    兩條路徑都把它視為海面，而不是將微小正差交給後續物理公式。
    """

    sampler = _moving_surface_ocm_sampler(use_numba_kernel=use_numba_kernel)
    endpoint_top_offset_m = 3.367686e-6
    sampler.elev[1, :] = 1.47 + endpoint_top_offset_m
    endpoint_surface_sample = sampler.sample(
        2.0,
        3.0,
        1.47 + endpoint_top_offset_m,
        1_000_000_000,
    )
    assert endpoint_surface_sample.valid
    assert endpoint_surface_sample.eta_m == 1.47 + endpoint_top_offset_m

    sampler = _moving_surface_ocm_sampler(use_numba_kernel=use_numba_kernel)
    query_time_ns = 146_666_667
    query_alpha = query_time_ns / 1_000_000_000.0
    query_eta_m = 1.77 + query_alpha * (1.47 - 1.77)
    near_surface_sample = sampler.sample(
        2.0,
        3.0,
        query_eta_m + SURFACE_BOUNDARY_TOLERANCE_M - 1.0e-9,
        query_time_ns,
    )
    assert near_surface_sample.valid
    assert near_surface_sample.eta_m == pytest.approx(query_eta_m)


@pytest.mark.parametrize("use_numba_kernel", [False, True])
def test_ocm_surface_boundary_residual_above_tolerance_and_bed_overshoot_remain_unsupported(
    use_numba_kernel: bool,
) -> None:
    """超過 5 微米的海面上越與任何海床下越都不得被表層容差掩蓋。"""

    sampler = _moving_surface_ocm_sampler(use_numba_kernel=use_numba_kernel)
    query_time_ns = 146_666_667
    query_alpha = query_time_ns / 1_000_000_000.0
    query_eta_m = 1.77 + query_alpha * (1.47 - 1.77)
    above_surface = sampler.sample(
        2.0,
        3.0,
        query_eta_m + SURFACE_BOUNDARY_TOLERANCE_M + 1.0e-9,
        query_time_ns,
    )
    assert not above_surface.valid
    assert above_surface.qc == SampleQC.VERTICAL_UNSUPPORTED

    below_bed = sampler.sample(
        2.0,
        3.0,
        -10.0 - 0.5 * SURFACE_BOUNDARY_TOLERANCE_M,
        query_time_ns,
    )
    assert not below_bed.valid
    assert below_bed.qc == SampleQC.VERTICAL_UNSUPPORTED


@pytest.mark.parametrize("use_numba_kernel", [False, True])
def test_combined_forcing_clamps_surface_boundary_residual_before_stokes(
    use_numba_kernel: bool,
) -> None:
    """OCM 已接受的海面邊界定位數值殘差在合成 Stokes 時不得再變成 ``INVALID_PHYSICS``。"""

    ocm = _moving_surface_ocm_sampler(use_numba_kernel=use_numba_kernel)
    times = np.array([0, 1_000_000_000], dtype=np.int64)
    nww = NWWAnalysisMonth(
        month_id="197001",
        lon=np.array([120.0, 122.0]),
        lat=np.array([24.0, 26.0]),
        time_utc_ns=times,
        significant_wave_height=np.full((2, 2, 2), 1.5),
        peak_frequency=np.full((2, 2, 2), 0.125),
        peak_direction_raw_deg=np.full((2, 2, 2), 225.0),
        valid_mask_wave=np.ones((2, 2, 2), dtype=bool),
        qc_flags=np.zeros((2, 2, 2), dtype=np.uint16),
        maximum_time_gap_seconds=2.0,
    )
    forcing = CombinedMonthForcing(
        ocm=ocm,
        nww=nww,
        projection=DomainProjection(121.0, 25.0),
        settling_velocity_mps=0.0,
        include_stokes=True,
    )
    query_time_ns = 146_666_667
    query_alpha = query_time_ns / 1_000_000_000.0
    query_eta_m = 1.77 + query_alpha * (1.47 - 1.77)
    near_surface = forcing.sample(
        2.0,
        3.0,
        query_eta_m + SURFACE_BOUNDARY_TOLERANCE_M - 1.0e-9,
        query_time_ns,
    )
    at_surface = forcing.sample(2.0, 3.0, query_eta_m, query_time_ns)
    assert near_surface.valid and at_surface.valid
    assert near_surface.qc == SampleQC.OK
    assert np.allclose(
        [near_surface.u_mps, near_surface.v_mps, near_surface.w_mps],
        [at_surface.u_mps, at_surface.v_mps, at_surface.w_mps],
        rtol=0.0,
        atol=1.0e-15,
    )


@pytest.mark.parametrize("use_numba_kernel", [False, True])
@pytest.mark.parametrize("target_z_m", [1.8, -11.0])
def test_ocm_query_time_surface_or_bed_crossing_remains_fail_closed(
    use_numba_kernel: bool,
    target_z_m: float,
) -> None:
    """endpoint hold 不得取代 query-time 海面／海床 gate。"""

    sampler = _moving_surface_ocm_sampler(use_numba_kernel=use_numba_kernel)
    sample = sampler.sample(2.0, 3.0, target_z_m, 146_666_667)

    assert not sample.valid
    assert sample.qc == SampleQC.VERTICAL_UNSUPPORTED
    assert np.isclose(sample.eta_m, 1.77 + (146_666_667 / 1_000_000_000.0) * (1.47 - 1.77))
    assert sample.bed_z_m == -10.0


@pytest.mark.parametrize("use_numba_kernel", [False, True])
def test_ocm_missing_eta_rejects_surface_hold_without_boundary_privilege(
    use_numba_kernel: bool,
) -> None:
    """缺少任一 endpoint eta 時，不能把有限 top-layer velocity 當成合法水柱。"""

    sampler = _moving_surface_ocm_sampler(use_numba_kernel=use_numba_kernel, missing_eta=True)
    sample = sampler.sample(2.0, 3.0, 1.683, 146_666_667)

    assert not sample.valid
    assert sample.qc == SampleQC.VERTICAL_UNSUPPORTED
    assert np.isnan(sample.eta_m)
    assert np.isnan(sample.bed_z_m)


@pytest.mark.parametrize("use_numba_kernel", [False, True])
@pytest.mark.parametrize("missing_top_field", ["hvel", "vertical_velocity", "diffusivity"])
def test_ocm_missing_after_top_layer_quantity_rejects_surface_hold(
    use_numba_kernel: bool,
    missing_top_field: str,
) -> None:
    """最高 zcor layer 的任一必要物理量缺值時，不得回退更深層補洞。"""

    sampler = _moving_surface_ocm_sampler(
        use_numba_kernel=use_numba_kernel,
        missing_top_field=missing_top_field,
    )
    sample = sampler.sample(2.0, 3.0, 1.683, 146_666_667)

    assert not sample.valid
    assert sample.qc == SampleQC.VERTICAL_UNSUPPORTED
    assert np.isclose(sample.eta_m, 1.77 + (146_666_667 / 1_000_000_000.0) * (1.47 - 1.77))
    assert sample.bed_z_m == -10.0


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


def _adjacent_mesh() -> NativeMesh:
    """建立由兩個三角形組成的正方形，供 hint 跨共邊走訪測試使用。"""

    return NativeMesh(
        node_lon=np.array([121.0, 121.001, 121.001, 121.0]),
        node_lat=np.array([25.0, 25.0, 25.001, 25.001]),
        node_xy=np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
        source_depth_m=np.full(4, 10.0),
        source_node_bottom_index=np.zeros(4, dtype=np.int64),
        face_nodes_local=np.array([[0, 1, 2, -1], [0, 2, 3, -1]]),
        face_node_count=np.array([3, 3]),
        source_face_global_index=np.array([101, 102]),
        bin_size_m=10.0,
    )


def _adjacent_ocm_sampler(mesh: NativeMesh, *, use_numba_kernel: bool = False) -> OCMNativeMonth:
    """建立線性且兩個 face 都濕潤的 OCM 測試場，以檢查 hint 不改變物理取樣。"""

    times = np.array([0, 10_000_000_000], dtype=np.int64)
    z_levels = np.array([-10.0, -5.0, 0.0])
    zcor = np.broadcast_to(z_levels, (2, 4, 3)).copy()
    hvel = np.empty((2, 4, 3, 2), dtype=np.float64)
    vertical_velocity = np.empty((2, 4, 3), dtype=np.float64)
    diffusivity = np.full((2, 4, 3), 0.01, dtype=np.float64)
    for time_index, seconds in enumerate([0.0, 10.0]):
        for node, (x_m, y_m) in enumerate(mesh.node_xy):
            for layer, z_m in enumerate(z_levels):
                hvel[time_index, node, layer, 0] = 0.01 * x_m + 0.02 * y_m + 0.1 * z_m + seconds
                hvel[time_index, node, layer, 1] = 0.03 * x_m - 0.04 * y_m - 0.05 * z_m
                vertical_velocity[time_index, node, layer] = 0.001 * z_m
    return OCMNativeMonth(
        month_id="197001",
        mesh=mesh,
        time_utc_ns=times,
        hvel=hvel,
        vertical_velocity=vertical_velocity,
        zcor=zcor,
        elev=np.zeros((2, 4)),
        wetdry_elem=np.zeros((2, 2)),
        diffusivity=diffusivity,
        maximum_time_gap_seconds=20.0,
        use_numba_kernel=use_numba_kernel,
    )


def _assert_velocity_samples_equal(first, second) -> None:
    """逐一比較速度、分項、尺度、QC 與 provenance，避免只比較 valid 布林值。"""

    assert np.allclose(
        [
            first.u_mps,
            first.v_mps,
            first.w_mps,
            first.eta_m,
            first.bed_z_m,
            first.horizontal_scale_m,
            first.vertical_scale_m,
        ],
        [
            second.u_mps,
            second.v_mps,
            second.w_mps,
            second.eta_m,
            second.bed_z_m,
            second.horizontal_scale_m,
            second.vertical_scale_m,
        ],
        equal_nan=True,
    )
    assert first.qc == second.qc
    assert first.source_face_id == second.source_face_id
    assert first.triangle_id == second.triangle_id
    assert first.forcing_month_id == second.forcing_month_id
    assert first.diagnostics == second.diagnostics
    assert first.components == second.components


def test_triangle_neighbors_are_opposite_vertex_adjacency() -> None:
    """共邊鄰接欄位必須對應對頂點的邊，外邊界使用 -1。"""

    mesh = _adjacent_mesh()
    assert mesh.triangle_neighbors.tolist() == [[-1, 1, -1], [-1, -1, 0]]


def test_non_manifold_shared_edge_is_marked_unknown_without_rejection() -> None:
    """三個面共用同一邊時不任意選鄰居，且 adjacency 建構仍可完成。"""

    neighbors = _build_triangle_neighbors(
        np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]], dtype=np.int64)
    )
    assert np.all(neighbors == -1)


def test_triangle_hint_follows_adjacent_triangle_without_uniform_bin_candidates() -> None:
    """清空局部 mesh 的 bins 後，public hint path 仍須從 triangle 0 走到 triangle 1。"""

    mesh = _adjacent_mesh()
    target_point = (0.2, 0.8)
    assert mesh.locate(*target_point).triangle_id == 1
    mesh._bins.clear()

    # 測試仍呼叫 public locate；只將這個局部 fixture 的候選索引清空，藉此隔離 hint
    # 走訪是否真的有效。測試結束後 fixture 被丟棄，不影響其他測試或正式 mesh。
    assert mesh.locate(*target_point) is None
    hinted_location = mesh.locate(*target_point, triangle_hint=0)
    assert hinted_location is not None
    assert hinted_location.triangle_id == 1


def test_triangle_hint_fallback_handles_stale_outside_and_nonfinite_points() -> None:
    """失效 hint、域外與非有限座標都維持原 locator 的安全結果。"""

    mesh = _adjacent_mesh()
    baseline = mesh.locate(0.2, 0.8)
    assert baseline is not None
    assert mesh.locate(0.2, 0.8, triangle_hint=-1) == baseline
    assert mesh.locate(0.2, 0.8, triangle_hint=999) == baseline
    assert mesh.locate(2.0, 2.0, triangle_hint=0) is None
    assert mesh.locate(np.nan, 0.0, triangle_hint=0) is None
    assert mesh.locate(0.0, np.inf, triangle_hint=0) is None


def test_edge_and_vertex_hints_fallback_to_smallest_triangle_id() -> None:
    """共邊與頂點不採 hint 直接回傳，必須維持全域搜尋的最小 triangle_id。"""

    mesh = _adjacent_mesh()
    for point in ((0.5, 0.5), (0.0, 0.0)):
        baseline = mesh.locate(*point)
        hinted_from_first = mesh.locate(*point, triangle_hint=0)
        hinted_from_second = mesh.locate(*point, triangle_hint=1)
        assert baseline is not None and baseline.triangle_id == 0
        assert hinted_from_first == baseline
        assert hinted_from_second == baseline


def test_ocm_hint_sampling_preserves_physics_and_provenance() -> None:
    """OCM 有效、stale 與無 hint 的速度、QC、face/triangle provenance 必須一致。"""

    mesh = _adjacent_mesh()
    sampler = _adjacent_ocm_sampler(mesh)
    baseline = sampler.sample(0.2, 0.8, -7.5, 5_000_000_000)
    walked = sampler.sample(0.2, 0.8, -7.5, 5_000_000_000, triangle_hint=0)
    stale = sampler.sample(0.2, 0.8, -7.5, 5_000_000_000, triangle_hint=999)

    assert baseline.valid
    assert baseline.triangle_id == 1
    assert baseline.source_face_id == 102
    _assert_velocity_samples_equal(baseline, walked)
    _assert_velocity_samples_equal(baseline, stale)


def test_combined_and_monthly_sample_forward_hint_without_changing_call_api() -> None:
    """新增 sample keyword 不改變既有 __call__ 的四參數相容介面。"""

    sampler = _adjacent_ocm_sampler(_adjacent_mesh())
    combined = CombinedMonthForcing(
        ocm=sampler,
        nww=None,
        projection=DomainProjection(121.0, 25.0),
        settling_velocity_mps=0.0,
        include_stokes=False,
    )
    explicit = combined.sample(0.2, 0.8, -7.5, 5_000_000_000, triangle_hint=0)
    via_call = combined(0.2, 0.8, -7.5, 5_000_000_000)
    _assert_velocity_samples_equal(explicit, via_call)

    monthly = MonthlyCombinedForcing({"197001": combined})
    monthly_explicit = monthly.sample(0.2, 0.8, -7.5, 5_000_000_000, triangle_hint=0)
    monthly_via_call = monthly(0.2, 0.8, -7.5, 5_000_000_000)
    _assert_velocity_samples_equal(explicit, monthly_explicit)
    _assert_velocity_samples_equal(explicit, monthly_via_call)


def _smagorinsky_sampler_from_node_values(
    mesh: NativeMesh,
    *,
    u_values_by_time: list[np.ndarray],
    v_values_by_time: list[np.ndarray],
    wetdry_elem: np.ndarray | None = None,
    maximum_time_gap_seconds: float = 20.0,
    use_numba_kernel: bool = False,
) -> OCMNativeMonth:
    """把每個時間／節點的水平 current 值展開成固定 z OCM synthetic month。"""

    time_count = len(u_values_by_time)
    node_count = mesh.node_xy.shape[0]
    z_levels = np.array([-10.0, 0.0])
    zcor = np.broadcast_to(z_levels, (time_count, node_count, z_levels.size)).copy()
    hvel = np.empty((time_count, node_count, z_levels.size, 2), dtype=np.float64)
    for time_index in range(time_count):
        hvel[time_index, :, :, 0] = np.asarray(u_values_by_time[time_index])[:, None]
        hvel[time_index, :, :, 1] = np.asarray(v_values_by_time[time_index])[:, None]
    vertical_velocity = np.zeros((time_count, node_count, z_levels.size), dtype=np.float64)
    diffusivity = np.full_like(vertical_velocity, 0.01)
    return OCMNativeMonth(
        month_id="197001",
        mesh=mesh,
        time_utc_ns=np.arange(time_count, dtype=np.int64) * 10_000_000_000,
        hvel=hvel,
        vertical_velocity=vertical_velocity,
        zcor=zcor,
        elev=np.zeros((time_count, node_count)),
        wetdry_elem=(
            np.zeros((time_count, mesh.source_face_global_index.size))
            if wetdry_elem is None
            else wetdry_elem
        ),
        diffusivity=diffusivity,
        maximum_time_gap_seconds=maximum_time_gap_seconds,
        use_numba_kernel=use_numba_kernel,
    )


def test_smagorinsky_triangle_formula_and_constant_kz_are_exact() -> None:
    """native current 線性 shear 的 raw/current Kh 應等於既有 equation 10。"""

    mesh = _triangle_mesh()
    u = np.array([0.01 * x + 0.02 * y for x, y in mesh.node_xy])
    v = np.array([0.03 * x - 0.04 * y for x, y in mesh.node_xy])
    sampler = _smagorinsky_sampler_from_node_values(mesh, u_values_by_time=[u, u], v_values_by_time=[v, v])
    settings = SmagorinskySettings(0.2, 0.0, 10.0, 0.07)
    sample = sampler.sample_smagorinsky_diffusion(2.0, 3.0, -5.0, 5_000_000_000, settings)
    expected, floor_hit, cap_hit = smagorinsky_horizontal_diffusivity(
        du_dx_per_s=0.01,
        du_dy_per_s=0.02,
        dv_dx_per_s=0.03,
        dv_dy_per_s=-0.04,
        triangle_area_m2=50.0,
        coefficient_cs=0.2,
        floor_m2ps=0.0,
        cap_m2ps=10.0,
    )
    assert sample.valid
    assert sample.coefficients.kx_m2ps == expected
    assert sample.coefficients.ky_m2ps == expected
    assert sample.coefficients.kz_m2ps == 0.07
    assert np.allclose(sample.diffusivity_divergence_mps, (0.0, 0.0, 0.0), atol=1e-18)
    assert sample.diagnostics["raw_current_triangle_kh_m2ps"] == expected
    assert sample.diagnostics["floor_hit"] is floor_hit
    assert sample.diagnostics["cap_hit"] is cap_hit
    assert sample.diagnostics["triangle_id"] == 0


def test_smagorinsky_solid_body_rotation_has_zero_strain_at_zero_floor() -> None:
    """剛體旋轉的反對稱速度梯度不應被誤判為剪切擴散。"""

    mesh = _triangle_mesh()
    omega = 0.3
    u = np.array([-omega * y for _, y in mesh.node_xy])
    v = np.array([omega * x for x, _ in mesh.node_xy])
    sampler = _smagorinsky_sampler_from_node_values(mesh, u_values_by_time=[u, u], v_values_by_time=[v, v])
    sample = sampler.sample_smagorinsky_diffusion(
        2.0,
        3.0,
        -5.0,
        0,
        SmagorinskySettings(0.2, 0.0, 10.0, 0.01),
    )
    assert sample.valid
    assert sample.coefficients.kx_m2ps == 0.0
    assert sample.diffusivity_divergence_mps == (0.0, 0.0, 0.0)
    assert sample.diagnostics["raw_current_triangle_kh_m2ps"] == 0.0


def test_smagorinsky_strain_invariant_under_orthogonal_coordinate_rotation() -> None:
    """同一線性速度場以旋轉座標表示時，equation 10 的 strain/Kh 應保持不變。"""

    gradient = np.array([[0.4, -0.7], [0.2, 0.1]])
    angle = 0.61
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated = rotation.T @ gradient @ rotation
    original_kh, _, _ = smagorinsky_horizontal_diffusivity(
        du_dx_per_s=gradient[0, 0],
        du_dy_per_s=gradient[0, 1],
        dv_dx_per_s=gradient[1, 0],
        dv_dy_per_s=gradient[1, 1],
        triangle_area_m2=2.0,
        coefficient_cs=0.15,
    )
    rotated_kh, _, _ = smagorinsky_horizontal_diffusivity(
        du_dx_per_s=rotated[0, 0],
        du_dy_per_s=rotated[0, 1],
        dv_dx_per_s=rotated[1, 0],
        dv_dy_per_s=rotated[1, 1],
        triangle_area_m2=2.0,
        coefficient_cs=0.15,
    )
    assert np.isclose(original_kh, rotated_kh, rtol=1e-12, atol=1e-12)


def test_smagorinsky_nodal_area_weighting_and_gradient_use_all_support_triangles() -> None:
    """兩三角形不同 shear 時，particle K 與 grad(K) 應符合手算 nodal area weighting。"""

    mesh = _adjacent_mesh()
    u = np.array([0.0, 0.0, 0.0, 2.0])
    v = np.zeros(4)
    sampler = _smagorinsky_sampler_from_node_values(mesh, u_values_by_time=[u, u], v_values_by_time=[v, v])
    settings = SmagorinskySettings(0.2, 0.0, 100.0, 0.01)
    sample = sampler.sample_smagorinsky_diffusion(0.2, 0.8, -5.0, 0, settings)

    # triangle 0 為零 strain；triangle 1 的梯度為 (-2,2)，故 Kh1=.02*sqrt(8)。
    # 兩面積都是 0.5，current triangle 1 的三個 nodal K 為 [Kh1/2,Kh1/2,Kh1]。
    expected_triangle_kh = 0.02 * np.sqrt(8.0)
    expected_particle_kh = 0.8 * expected_triangle_kh
    assert sample.valid
    assert np.isclose(sample.coefficients.kx_m2ps, expected_particle_kh)
    assert np.isclose(sample.diagnostics["d_kh_dx_mps"], -expected_triangle_kh / 2.0)
    assert np.isclose(sample.diagnostics["d_kh_dy_mps"], expected_triangle_kh / 2.0)
    assert np.isclose(sample.diagnostics["raw_current_triangle_kh_m2ps"], expected_triangle_kh)
    assert sample.diagnostics["valid_incident_triangle_count"] == 2
    assert sample.diagnostics["excluded_incident_triangle_count"] == 0


def test_smagorinsky_time_interpolation_is_after_each_slice_formula() -> None:
    """before/after 的非線性 Kh 各自計算後，raw current triangle Kh 才線性內插。"""

    mesh = _triangle_mesh()
    zero = np.zeros(3)
    shear = np.array([0.0, 20.0, 0.0])
    sampler = _smagorinsky_sampler_from_node_values(
        mesh,
        u_values_by_time=[zero, shear],
        v_values_by_time=[zero, zero],
    )
    settings = SmagorinskySettings(0.2, 0.0, 100.0, 0.01)
    sample = sampler.sample_smagorinsky_diffusion(2.0, 3.0, -5.0, 5_000_000_000, settings)
    # 這個 triangle 的 shear 值在 node 2 造成 du/dx=2；面積 50、Cs=.2 時 Kh=4.0。
    expected_after, _, _ = smagorinsky_horizontal_diffusivity(
        du_dx_per_s=2.0,
        du_dy_per_s=0.0,
        dv_dx_per_s=0.0,
        dv_dy_per_s=0.0,
        triangle_area_m2=50.0,
        coefficient_cs=0.2,
    )
    assert sample.valid
    assert np.isclose(sample.diagnostics["raw_current_triangle_kh_m2ps"], expected_after * 0.5)
    assert np.isclose(sample.coefficients.kx_m2ps, expected_after * 0.5)


def test_smagorinsky_floor_cap_and_numba_velocity_switch_have_stable_diagnostics() -> None:
    """floor/cap 命中需可追蹤，且 Smag reference 不受 OCM velocity switch 影響。"""

    mesh = _triangle_mesh()
    u = np.zeros(3)
    v = np.zeros(3)
    numpy_sampler = _smagorinsky_sampler_from_node_values(
        mesh, u_values_by_time=[u, u], v_values_by_time=[v, v], use_numba_kernel=False
    )
    numba_sampler = _smagorinsky_sampler_from_node_values(
        mesh, u_values_by_time=[u, u], v_values_by_time=[v, v], use_numba_kernel=True
    )
    settings = SmagorinskySettings(0.2, 0.1, 0.2, 0.01)
    numpy_sample = numpy_sampler.sample_smagorinsky_diffusion(2.0, 3.0, -5.0, 0, settings)
    numba_sample = numba_sampler.sample_smagorinsky_diffusion(2.0, 3.0, -5.0, 0, settings)
    assert numpy_sample.valid and numba_sample.valid
    assert numpy_sample.diagnostics == numba_sample.diagnostics
    assert numpy_sample.diagnostics["floor_hit"] is True
    assert numpy_sample.diagnostics["cap_hit"] is False
    assert numpy_sample.coefficients.kx_m2ps == 0.1


def test_smagorinsky_excludes_dry_support_but_fails_current_triangle() -> None:
    """非目前 incident dry face 可排除；目前 triangle 乾涸則必須回 DRY_FACE。"""

    mesh = _adjacent_mesh()
    u = np.array([0.0, 0.0, 0.0, 2.0])
    v = np.zeros(4)
    wet = np.zeros((2, 2))
    wet[:, 0] = 1.0
    sampler = _smagorinsky_sampler_from_node_values(
        mesh, u_values_by_time=[u, u], v_values_by_time=[v, v], wetdry_elem=wet
    )
    settings = SmagorinskySettings(0.2, 0.0, 100.0, 0.01)
    supported = sampler.sample_smagorinsky_diffusion(0.2, 0.8, -5.0, 0, settings)
    assert supported.valid
    assert supported.diagnostics["valid_incident_triangle_count"] == 1
    assert supported.diagnostics["excluded_incident_triangle_count"] == 1

    current_dry = _smagorinsky_sampler_from_node_values(
        mesh,
        u_values_by_time=[u, u],
        v_values_by_time=[v, v],
        wetdry_elem=np.ones((2, 2)),
    )
    failed = current_dry.sample_smagorinsky_diffusion(0.2, 0.8, -5.0, 0, settings)
    assert not failed.valid
    assert failed.qc & SampleQC.DRY_FACE


def test_smagorinsky_returns_vertical_time_gap_and_outside_qc_without_fill() -> None:
    """固定 z unsupported、時間 gap 與域外位置都應各自保留非零 QC。"""

    mesh = _triangle_mesh()
    zeros = np.zeros(3)
    settings = SmagorinskySettings(0.2, 0.0, 100.0, 0.01)
    vertical = _smagorinsky_sampler_from_node_values(
        mesh, u_values_by_time=[zeros, zeros], v_values_by_time=[zeros, zeros]
    )
    unsupported = vertical.sample_smagorinsky_diffusion(2.0, 3.0, 1.0, 0, settings)
    assert not unsupported.valid
    assert unsupported.qc & SampleQC.VERTICAL_UNSUPPORTED

    gap = _smagorinsky_sampler_from_node_values(
        mesh,
        u_values_by_time=[zeros, zeros],
        v_values_by_time=[zeros, zeros],
        maximum_time_gap_seconds=1.0,
    )
    gap_sample = gap.sample_smagorinsky_diffusion(2.0, 3.0, -5.0, 5_000_000_000, settings)
    assert not gap_sample.valid
    assert gap_sample.qc & SampleQC.TIME_GAP

    outside = vertical.sample_smagorinsky_diffusion(99.0, 99.0, -5.0, 0, settings)
    assert not outside.valid
    assert outside.qc & SampleQC.OUTSIDE_HORIZONTAL_DOMAIN


@pytest.mark.parametrize("use_numba_kernel", [False, True])
def test_smagorinsky_surface_hold_shares_query_time_vertical_gate(
    use_numba_kernel: bool,
) -> None:
    """Smagorinsky 水平速度共用 top hold，且不可繞過一般速度的 query-time gate。

    Smagorinsky reference 只取水平 ``hvel``，但其 endpoint 垂向支援仍面對相同移動
    海面。因此 z=1.683 m 應在 after endpoint 使用最高層水平 current 並得到合法 Kh；
    z=1.8 m 雖然也能觸發 endpoint hold，卻高於 query-time eta，必須回傳
    ``VERTICAL_UNSUPPORTED``。兩個 ``use_numba_kernel`` 設定用來確認一般 OCM 路徑
    的切換不會改變這個 NumPy Smagorinsky reference 的結果。
    """

    sampler = _moving_surface_ocm_sampler(use_numba_kernel=use_numba_kernel)
    settings = SmagorinskySettings(0.2, 0.0, 100.0, 0.01)
    valid = sampler.sample_smagorinsky_diffusion(
        2.0,
        3.0,
        1.683,
        146_666_667,
        settings,
    )
    above_surface = sampler.sample_smagorinsky_diffusion(
        2.0,
        3.0,
        1.8,
        146_666_667,
        settings,
    )

    assert valid.valid
    assert np.isfinite(valid.coefficients.kx_m2ps)
    assert valid.diagnostics["valid_incident_triangle_count"] == 1
    assert not above_surface.valid
    assert above_surface.qc == SampleQC.VERTICAL_UNSUPPORTED


def test_smagorinsky_caches_unique_node_columns_per_time_slice() -> None:
    """同一 time slice 的每個 unique node 固定 z 取樣只能呼叫一次。"""

    mesh = _adjacent_mesh()
    u = np.zeros(4)
    v = np.zeros(4)
    sampler = _smagorinsky_sampler_from_node_values(mesh, u_values_by_time=[u, u], v_values_by_time=[v, v])
    calls: list[tuple[int, int]] = []
    original = sampler._vertical_horizontal_velocity_node_sample

    def wrapped(*, time_index: int, node_index: int, z_m: float):
        """記錄 helper 呼叫後交回原本的保守垂向取樣。"""

        calls.append((time_index, node_index))
        return original(time_index=time_index, node_index=node_index, z_m=z_m)

    sampler._vertical_horizontal_velocity_node_sample = wrapped
    sample = sampler.sample_smagorinsky_diffusion(
        0.2,
        0.8,
        -5.0,
        5_000_000_000,
        SmagorinskySettings(0.2, 0.0, 100.0, 0.01),
    )
    assert sample.valid
    assert sorted(calls) == [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]


def test_combined_forcing_sums_3d_ocm_stokes_and_strict_sinking_without_windage() -> None:
    """驗證三維 OCM、非零 NWW3 Stokes 與嚴格負沉降速度的解析三分量合成。

    OCM 測試場在公尺座標 ``(x,y,z)`` 與秒單位時間上各自為線性函式，因此三角形水平
    形函數、垂向夾層及時間內插都能得到解析值。NWW3 則提供非零有效波高與斜向來波，
    由有限水深 Stokes 公式產生兩個水平分量。粒子位於 ``-4 m``，嚴格低於海面 ``0 m``
    且高於海床 ``-10 m``，所以這個測試驗證的是完全浸沒粒子：水平只含 OCM current
    加 Stokes，垂向只把嚴格小於零的沉降速度加到 OCM ``w``，不得偷偷加入風致漂移。

    單位與容許誤差：位置是 m、速度是 m/s、時間是 UTC ns；解析合成以 ``rtol=1e-12``
    與 ``atol=1e-12 m/s`` 比較。``settling_velocity_mps=-0.07`` 明確代表向下速度，
    而不是零值或未定義的浮力項。
    """

    mesh = _triangle_mesh()
    times_ns = np.array([0, 10_000_000_000], dtype=np.int64)
    z_levels_m = np.array([-10.0, -5.0, 0.0])
    zcor = np.broadcast_to(z_levels_m, (2, 3, 3)).copy()
    hvel = np.empty((2, 3, 3, 2), dtype=np.float64)
    vertical_velocity = np.empty((2, 3, 3), dtype=np.float64)
    diffusivity = np.full((2, 3, 3), 0.01, dtype=np.float64)
    for time_index, time_seconds in enumerate((0.0, 10.0)):
        for node_index, (x_m, y_m) in enumerate(mesh.node_xy):
            for layer_index, z_m in enumerate(z_levels_m):
                # 這三個線性場分別是東、北與向上速度；時間項用來確認 OCM 四維
                # 內插也參與合成，但預期值仍可由解析函式直接計算。
                hvel[time_index, node_index, layer_index, 0] = (
                    0.4 + 0.01 * x_m + 0.02 * y_m + 0.03 * z_m + 0.01 * time_seconds
                )
                hvel[time_index, node_index, layer_index, 1] = (
                    -0.2 + 0.015 * x_m - 0.01 * y_m + 0.02 * z_m - 0.005 * time_seconds
                )
                vertical_velocity[time_index, node_index, layer_index] = (
                    0.05 + 0.001 * x_m - 0.002 * y_m + 0.004 * z_m + 0.002 * time_seconds
                )
    ocm = OCMNativeMonth(
        month_id="197001",
        mesh=mesh,
        time_utc_ns=times_ns,
        hvel=hvel,
        vertical_velocity=vertical_velocity,
        zcor=zcor,
        elev=np.zeros((2, 3), dtype=np.float64),
        wetdry_elem=np.zeros((2, 1), dtype=np.float64),
        diffusivity=diffusivity,
        maximum_time_gap_seconds=20.0,
    )
    nww = NWWAnalysisMonth(
        month_id="197001",
        lon=np.array([120.0, 122.0]),
        lat=np.array([24.0, 26.0]),
        time_utc_ns=times_ns,
        significant_wave_height=np.full((2, 2, 2), 1.5, dtype=np.float64),
        peak_frequency=np.full((2, 2, 2), 0.125, dtype=np.float64),
        peak_direction_raw_deg=np.full((2, 2, 2), 225.0, dtype=np.float64),
        valid_mask_wave=np.ones((2, 2, 2), dtype=bool),
        qc_flags=np.zeros((2, 2, 2), dtype=np.uint16),
        maximum_time_gap_seconds=20.0,
    )
    projection = DomainProjection(121.0, 25.0)
    sinking_velocity_mps = -0.07
    with_stokes = CombinedMonthForcing(
        ocm=ocm,
        nww=nww,
        projection=projection,
        settling_velocity_mps=sinking_velocity_mps,
        include_stokes=True,
    )
    without_stokes = CombinedMonthForcing(
        ocm=ocm,
        nww=None,
        projection=projection,
        settling_velocity_mps=sinking_velocity_mps,
        include_stokes=False,
    )

    x_m, y_m, z_m, time_ns = 2.0, 3.0, -4.0, 5_000_000_000
    sample = with_stokes.sample(x_m, y_m, z_m, time_ns)
    no_stokes_sample = without_stokes.sample(x_m, y_m, z_m, time_ns)
    assert sample.valid and no_stokes_sample.valid
    assert no_stokes_sample.eta_m == 0.0
    assert no_stokes_sample.bed_z_m == -10.0
    assert no_stokes_sample.bed_z_m < z_m < no_stokes_sample.eta_m

    elapsed_seconds = 5.0
    expected_ocm = np.array(
        [
            0.4 + 0.01 * x_m + 0.02 * y_m + 0.03 * z_m + 0.01 * elapsed_seconds,
            -0.2 + 0.015 * x_m - 0.01 * y_m + 0.02 * z_m - 0.005 * elapsed_seconds,
            0.05 + 0.001 * x_m - 0.002 * y_m + 0.004 * z_m + 0.002 * elapsed_seconds,
        ],
        dtype=np.float64,
    )
    expected_stokes = finite_depth_stokes(
        significant_wave_height_m=1.5,
        peak_frequency_hz=0.125,
        direction_raw_deg=225.0,
        particle_z_m=z_m,
        surface_z_m=0.0,
        bed_z_m=-10.0,
    )
    expected_total = expected_ocm + np.array(
        [expected_stokes.u_mps, expected_stokes.v_mps, sinking_velocity_mps]
    )
    assert np.linalg.norm([expected_stokes.u_mps, expected_stokes.v_mps]) > 0.0
    assert np.allclose(
        [sample.u_mps, sample.v_mps, sample.w_mps],
        expected_total,
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.allclose(
        [no_stokes_sample.u_mps, no_stokes_sample.v_mps, no_stokes_sample.w_mps],
        expected_ocm + np.array([0.0, 0.0, sinking_velocity_mps]),
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.allclose(
        [sample.u_mps - no_stokes_sample.u_mps, sample.v_mps - no_stokes_sample.v_mps],
        [expected_stokes.u_mps, expected_stokes.v_mps],
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.isclose(sample.w_mps, no_stokes_sample.w_mps, rtol=0.0, atol=1e-12)
    assert sample.w_mps < expected_ocm[2]
    assert np.isclose(sample.diagnostics["stokes_u_mps"], expected_stokes.u_mps, atol=1e-12)
    assert np.isclose(sample.diagnostics["stokes_v_mps"], expected_stokes.v_mps, atol=1e-12)
    assert sample.components is not None
    assert np.allclose(
        [
            sample.components.total_u_mps,
            sample.components.total_v_mps,
            sample.components.total_w_mps,
            sample.components.ocm_u_mps,
            sample.components.ocm_v_mps,
            sample.components.ocm_w_mps,
            sample.components.stokes_u_mps,
            sample.components.stokes_v_mps,
            sample.components.settling_w_mps,
        ],
        [
            expected_total[0],
            expected_total[1],
            expected_total[2],
            expected_ocm[0],
            expected_ocm[1],
            expected_ocm[2],
            expected_stokes.u_mps,
            expected_stokes.v_mps,
            sinking_velocity_mps,
        ],
        rtol=1e-12,
        atol=1e-12,
    )
    assert no_stokes_sample.components is not None
    assert no_stokes_sample.components.stokes_u_mps == 0.0
    assert no_stokes_sample.components.stokes_v_mps == 0.0
    assert no_stokes_sample.components.settling_w_mps == sinking_velocity_mps
