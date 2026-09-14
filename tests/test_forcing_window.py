"""lazy forcing window 的月份選擇、LRU、NWW lazy loading 與 hint 傳遞測試。"""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from shapely.geometry import box

import lagrangian_backtracking.forcing_window as forcing_window_module
from lagrangian_backtracking.boundaries import BoundaryGeometry
from lagrangian_backtracking.checkpoint import CheckpointBinding
from lagrangian_backtracking.config import load_config
from lagrangian_backtracking.diffusion import DiffusionCoefficients, SmagorinskySettings
from lagrangian_backtracking.engine import (
    EngineSettings,
    advance_particle_once,
    initialize_particle_execution,
)
from lagrangian_backtracking.forcing import (
    CombinedMonthForcing,
    NWWAnalysisMonth,
    OCMNativeMonth,
)
from lagrangian_backtracking.forcing_window import (
    ForcingWindowManager,
    MissingForcingMonth,
)
from lagrangian_backtracking.geometry import DomainProjection
from lagrangian_backtracking.integrators import supports_step_start_sample_reuse
from lagrangian_backtracking.mesh import NativeMesh
from lagrangian_backtracking.models import ParticleState, SampleQC
from lagrangian_backtracking.production import HintTrackingVelocityProvider, ProductionBatch
from lagrangian_backtracking.runner import ReferenceParticleRequest, plan_scenario_shards
from lagrangian_backtracking.scenarios import Scenario


def _mesh() -> NativeMesh:
    """建立可容納測試粒子的單一三角形 native mesh。"""

    return NativeMesh(
        node_lon=np.array([121.0, 121.001, 121.0]),
        node_lat=np.array([25.0, 25.0, 25.001]),
        node_xy=np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]]),
        source_depth_m=np.full(3, 10.0),
        source_node_bottom_index=np.zeros(3, dtype=np.int64),
        face_nodes_local=np.array([[0, 1, 2, -1]]),
        face_node_count=np.array([3]),
        source_face_global_index=np.array([99]),
        bin_size_m=20.0,
    )


def _month_start_ns(month_id: str) -> int:
    """將 YYYYMM 轉成 UTC 月初奈秒，避免測試依賴本機時區。"""

    start = datetime.strptime(month_id, "%Y%m").replace(tzinfo=UTC)
    return int(start.timestamp()) * 1_000_000_000


def _utc_ns(value: str) -> int:
    """將 exact UTC ISO 時刻轉成整數奈秒，避免測試依賴本機時區或浮點四捨五入。"""

    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("測試 UTC 時刻必須帶有時區")
    return int(parsed.timestamp()) * 1_000_000_000


def _ocm(month_id: str, mesh: NativeMesh, *, eastward_mps: float = 1.0) -> OCMNativeMonth:
    """建立兩個逐時支撐點的有效 OCM month；所有面保持 wet。"""

    times = np.array(
        [_month_start_ns(month_id), _month_start_ns(month_id) + 3_600_000_000_000], dtype=np.int64
    )
    z_levels = np.array([-10.0, 0.0])
    zcor = np.broadcast_to(z_levels, (2, 3, 2)).copy()
    hvel = np.zeros((2, 3, 2, 2), dtype=np.float64)
    hvel[..., 0] = eastward_mps
    vertical_velocity = np.zeros((2, 3, 2), dtype=np.float64)
    diffusivity = np.full((2, 3, 2), 0.01, dtype=np.float64)
    return OCMNativeMonth(
        month_id=month_id,
        mesh=mesh,
        time_utc_ns=times,
        hvel=hvel,
        vertical_velocity=vertical_velocity,
        zcor=zcor,
        elev=np.zeros((2, 3)),
        wetdry_elem=np.zeros((2, 1)),
        diffusivity=diffusivity,
        maximum_time_gap_seconds=7_200.0,
    )


def _ocm_cross_month(
    month_id: str,
    times_ns: list[int],
    *,
    maximum_time_gap_seconds: float = 7_200.0,
    reference_ns: int | None = None,
    use_numba_kernel: bool = False,
) -> OCMNativeMonth:
    """建立跨月端點共用同一解析場的 OCM 月份。

    每個欄位都依絕對 UTC 小時產生，因此 July 尾端 00:00 與 August 首端 01:00
    在分月產品和連續參考產品中具有完全相同的 endpoint 值。速度、海面高程與
    ``diffusivity`` 都保留時間變化，才能檢查跨月內插沒有退回任一側最近值。
    ``use_numba_kernel`` 只切換 OCM 端點的加速實作，數值結果仍應與 NumPy 路徑一致。
    """

    mesh = _mesh()
    times = np.asarray(times_ns, dtype=np.int64)
    origin_ns = (
        _utc_ns("2024-08-01T00:00:00Z") if reference_ns is None else int(reference_ns)
    )
    hours = (times.astype(np.float64) - origin_ns) / 3_600_000_000_000.0
    z_levels = np.array([-10.0, 0.0])
    zcor = np.broadcast_to(z_levels, (times.size, 3, 2)).copy()
    hvel = np.empty((times.size, 3, 2, 2), dtype=np.float64)
    vertical_velocity = np.empty((times.size, 3, 2), dtype=np.float64)
    diffusivity = np.empty((times.size, 3, 2), dtype=np.float64)
    elev = np.empty((times.size, 3), dtype=np.float64)
    for time_index, hour in enumerate(hours):
        for node_index in range(3):
            for layer_index in range(2):
                hvel[time_index, node_index, layer_index, 0] = (
                    0.15 + 0.02 * hour + 0.01 * node_index + 0.005 * layer_index
                )
                hvel[time_index, node_index, layer_index, 1] = (
                    -0.03 + 0.01 * hour + 0.003 * node_index + 0.002 * layer_index
                )
                vertical_velocity[time_index, node_index, layer_index] = (
                    0.001 + 0.0002 * hour + 0.0001 * layer_index
                )
                diffusivity[time_index, node_index, layer_index] = (
                    0.004 + 0.0005 * hour + 0.0001 * node_index + 0.0002 * layer_index
                )
            elev[time_index, node_index] = 0.3 + 0.01 * hour + 0.001 * node_index
    return OCMNativeMonth(
        month_id=month_id,
        mesh=mesh,
        time_utc_ns=times,
        hvel=hvel,
        vertical_velocity=vertical_velocity,
        zcor=zcor,
        elev=elev,
        wetdry_elem=np.zeros((times.size, 1)),
        diffusivity=diffusivity,
        maximum_time_gap_seconds=maximum_time_gap_seconds,
        use_numba_kernel=use_numba_kernel,
    )


def _nww(month_id: str) -> NWWAnalysisMonth:
    """建立覆蓋測試位置的有效 NWW analysis month。"""

    shape = (2, 2, 2)
    return NWWAnalysisMonth(
        month_id=month_id,
        lon=np.array([120.0, 122.0]),
        lat=np.array([24.0, 26.0]),
        time_utc_ns=np.array(
            [_month_start_ns(month_id), _month_start_ns(month_id) + 3_600_000_000_000],
            dtype=np.int64,
        ),
        significant_wave_height=np.ones(shape),
        peak_frequency=np.full(shape, 0.1),
        peak_direction_raw_deg=np.zeros(shape),
        valid_mask_wave=np.ones(shape, dtype=bool),
        qc_flags=np.zeros(shape, dtype=np.uint16),
        maximum_time_gap_seconds=7_200.0,
    )


def _nww_cross_month(
    month_id: str,
    times_ns: list[int],
    *,
    maximum_time_gap_seconds: float = 7_200.0,
    reference_ns: int | None = None,
) -> NWWAnalysisMonth:
    """建立跨月四角有效的 NWW fixture，並讓波向在 0/360 度附近連續變化。"""

    times = np.asarray(times_ns, dtype=np.int64)
    origin_ns = (
        _utc_ns("2024-08-01T00:00:00Z") if reference_ns is None else int(reference_ns)
    )
    hours = (times.astype(np.float64) - origin_ns) / 3_600_000_000_000.0
    corner_hs = np.array([[1.0, 1.4], [1.8, 2.2]], dtype=np.float64)
    corner_fp = np.array([[0.08, 0.10], [0.12, 0.14]], dtype=np.float64)
    corner_direction = np.array([[350.0, 10.0], [355.0, 5.0]], dtype=np.float64)
    hs = np.stack([corner_hs + 0.1 * hour for hour in hours])
    fp = np.stack([corner_fp + 0.005 * hour for hour in hours])
    directions = np.stack([(corner_direction + 4.0 * hour) % 360.0 for hour in hours])
    shape = (times.size, 2, 2)
    return NWWAnalysisMonth(
        month_id=month_id,
        lon=np.array([120.0, 122.0]),
        lat=np.array([24.0, 26.0]),
        time_utc_ns=times,
        significant_wave_height=hs,
        peak_frequency=fp,
        peak_direction_raw_deg=directions,
        valid_mask_wave=np.ones(shape, dtype=bool),
        qc_flags=np.zeros(shape, dtype=np.uint16),
        maximum_time_gap_seconds=maximum_time_gap_seconds,
    )


def _assert_velocity_equivalent(actual, expected) -> None:
    """逐欄比對跨月與連續參考速度，排除只代表來源月份名稱的 provenance 欄位。"""

    for field in (
        "u_mps",
        "v_mps",
        "w_mps",
        "eta_m",
        "bed_z_m",
        "horizontal_scale_m",
        "vertical_scale_m",
    ):
        assert np.isclose(getattr(actual, field), getattr(expected, field), rtol=1e-12, atol=1e-12)
    assert actual.qc == expected.qc
    assert actual.source_face_id == expected.source_face_id
    assert actual.triangle_id == expected.triangle_id
    assert actual.components is not None and expected.components is not None
    for field in (
        "total_u_mps",
        "total_v_mps",
        "total_w_mps",
        "ocm_u_mps",
        "ocm_v_mps",
        "ocm_w_mps",
        "stokes_u_mps",
        "stokes_v_mps",
        "settling_w_mps",
    ):
        assert np.isclose(
            getattr(actual.components, field),
            getattr(expected.components, field),
            rtol=1e-12,
            atol=1e-12,
        )
    assert actual.diagnostics.keys() == expected.diagnostics.keys()
    for key, actual_value in actual.diagnostics.items():
        expected_value = expected.diagnostics[key]
        if isinstance(actual_value, (float, int)) and isinstance(expected_value, (float, int)):
            assert np.isclose(actual_value, expected_value, rtol=1e-12, atol=1e-12)
        else:
            assert actual_value == expected_value


def test_cross_month_velocity_matches_single_continuous_provider_at_all_boundary_times() -> None:
    """OCM/NWW 分月產品在月界四個時刻應等價於同端點的連續參考 provider。

    OCM July 尾端固定包含 2024-08-01 00:00，August 首端固定包含 01:00；00:30
    與 00:59:45 必須使用兩個月份的實際端點做時間內插。NWW 同時以四角不同的
    ``Hs``、``fp`` 與 350/10 度附近波向測試空間、時間及 circular interpolation。
    連續參考產品保留完全相同的絕對時間端點，故比較涵蓋 primitive 派生欄位、有限水深
    Stokes、沉降後總速度、海面、海床、垂向擴散與品質旗標；月份名稱本身不屬於數值等價條件。
    """

    july_end = [_utc_ns("2024-07-31T23:00:00Z"), _utc_ns("2024-08-01T00:00:00Z")]
    august_start = [_utc_ns("2024-08-01T01:00:00Z"), _utc_ns("2024-08-01T02:00:00Z")]
    continuous_times = july_end[:1] + july_end[1:] + august_start
    split_ocm = {
        "202407": _ocm_cross_month("202407", july_end),
        "202408": _ocm_cross_month("202408", august_start),
    }
    split_nww = {
        "202407": _nww_cross_month("202407", july_end),
        "202408": _nww_cross_month("202408", august_start),
    }
    manager, ocm_loader, nww_loader = _manager(split_ocm, split_nww)
    continuous_ocm = _ocm_cross_month("continuous", continuous_times)
    continuous_nww = _nww_cross_month("continuous", continuous_times)
    reference = CombinedMonthForcing(
        ocm=continuous_ocm,
        nww=continuous_nww,
        projection=DomainProjection(121.0, 25.0),
        settling_velocity_mps=-0.002,
        include_stokes=True,
    )

    query_times = (
        _utc_ns("2024-08-01T00:00:00Z"),
        _utc_ns("2024-08-01T00:30:00Z"),
        _utc_ns("2024-08-01T00:59:45Z"),
        _utc_ns("2024-08-01T01:00:00Z"),
    )
    provider = manager.provider(-0.002, include_stokes=True)
    for time_ns in query_times:
        actual = provider.sample(1.0, 1.0, -5.0, time_ns, triangle_hint=0)
        expected = reference.sample(1.0, 1.0, -5.0, time_ns, triangle_hint=0)
        assert actual.valid, (time_ns, actual.qc, actual.diagnostics)
        assert expected.valid
        _assert_velocity_equivalent(actual, expected)
        primitive = continuous_ocm.sample(1.0, 1.0, -5.0, time_ns, triangle_hint=0)
        assert primitive.valid
        assert actual.components is not None
        assert np.isclose(actual.components.ocm_u_mps, primitive.u_mps, rtol=1e-12, atol=1e-12)
        assert np.isclose(actual.components.ocm_v_mps, primitive.v_mps, rtol=1e-12, atol=1e-12)
        assert np.isclose(actual.components.ocm_w_mps, primitive.w_mps, rtol=1e-12, atol=1e-12)
        assert np.isclose(actual.eta_m, primitive.eta_m, rtol=1e-12, atol=1e-12)
        assert np.isclose(actual.bed_z_m, primitive.bed_z_m, rtol=1e-12, atol=1e-12)
        assert np.isclose(
            actual.diagnostics["kz_m2ps"],
            primitive.diagnostics["kz_m2ps"],
            rtol=1e-12,
            atol=1e-12,
        )
    wave = continuous_nww.sample(121.0, 25.0, _utc_ns("2024-08-01T00:59:45Z"))
    assert wave.valid
    assert min(wave.peak_direction_raw_deg, 360.0 - wave.peak_direction_raw_deg) < 10.0
    assert ocm_loader.call_count >= 2
    assert nww_loader.call_count >= 2


def test_leading_halo_changes_before_boundary_query_and_exact_duplicate_prefers_last() -> None:
    """下一月份的前置 halo 必須參與月界前查詢，且 exact duplicate 採後月份資料。

    這個 fixture 刻意讓 July 只有 22:00／23:00，August 則以不同數值重複 23:00，
    再保存 00:00。若 manager 只看 query 所屬曆月與 July 的首末列，22:30 會錯誤
    沿用 July 內插結果；正確的 canonical prefer-last 應使用 August 的 23:00 作為
    after endpoint。23:00 exact 也必須直接選 August 那筆重複資料，不能因為它位於
    July 的末列而保留較早月份的值。
    """

    july_times = [
        _utc_ns("2024-07-31T22:00:00Z"),
        _utc_ns("2024-07-31T23:00:00Z"),
    ]
    august_times = [
        _utc_ns("2024-07-31T23:00:00Z"),
        _utc_ns("2024-08-01T00:00:00Z"),
    ]
    july = _ocm_cross_month("202407", july_times)
    august = _ocm_cross_month("202408", august_times)
    # August 的 leading halo 與 July 的末列是同一 UTC，但內容刻意不同，才能驗證
    # prefer-last 的來源選擇，而不是只驗證兩端時間相同。
    august.hvel[0, ..., 0] += 1.0
    manager, _, _ = _manager({"202407": july, "202408": august})

    continuous = _ocm_cross_month(
        "continuous",
        [
            _utc_ns("2024-07-31T22:00:00Z"),
            _utc_ns("2024-07-31T23:00:00Z"),
            _utc_ns("2024-08-01T00:00:00Z"),
        ],
    )
    continuous.hvel[1, ..., 0] = august.hvel[0, ..., 0]
    reference = CombinedMonthForcing(
        ocm=continuous,
        nww=None,
        projection=DomainProjection(121.0, 25.0),
        settling_velocity_mps=-0.002,
        include_stokes=False,
    )
    provider = manager.provider(-0.002, include_stokes=False)

    leading_query = _utc_ns("2024-07-31T22:30:00Z")
    actual_leading = provider.sample(1.0, 1.0, -5.0, leading_query, triangle_hint=0)
    expected_leading = reference.sample(1.0, 1.0, -5.0, leading_query, triangle_hint=0)
    assert actual_leading.valid
    assert expected_leading.valid
    assert np.isclose(actual_leading.u_mps, expected_leading.u_mps, rtol=1e-12, atol=1e-12)
    assert actual_leading.forcing_month_id == "202408"

    exact_query = _utc_ns("2024-07-31T23:00:00Z")
    actual_exact = provider.sample(1.0, 1.0, -5.0, exact_query, triangle_hint=0)
    expected_exact = reference.sample(1.0, 1.0, -5.0, exact_query, triangle_hint=0)
    assert actual_exact.valid
    assert expected_exact.valid
    assert np.isclose(actual_exact.u_mps, expected_exact.u_mps, rtol=1e-12, atol=1e-12)
    assert actual_exact.forcing_month_id == "202408"


def test_staggered_ocm_and_nww_exact_endpoint_uses_each_product_month() -> None:
    """OCM／NWW 月界端點錯開時，exact query 仍須分別採用各自實際月份資料。"""

    july_ocm_times = [
        _utc_ns("2024-07-31T23:00:00Z"),
        _utc_ns("2024-08-01T00:00:00Z"),
    ]
    august_ocm_times = [
        _utc_ns("2024-08-01T01:00:00Z"),
        _utc_ns("2024-08-01T02:00:00Z"),
    ]
    july_nww_times = [
        _utc_ns("2024-07-31T22:00:00Z"),
        _utc_ns("2024-07-31T23:00:00Z"),
    ]
    august_nww_times = [
        _utc_ns("2024-08-01T00:00:00Z"),
        _utc_ns("2024-08-01T01:00:00Z"),
    ]
    manager, _, _ = _manager(
        {
            "202407": _ocm_cross_month("202407", july_ocm_times),
            "202408": _ocm_cross_month("202408", august_ocm_times),
        },
        {
            "202407": _nww_cross_month("202407", july_nww_times),
            "202408": _nww_cross_month("202408", august_nww_times),
        },
    )
    continuous_times = [
        _utc_ns("2024-07-31T23:00:00Z"),
        _utc_ns("2024-08-01T00:00:00Z"),
        _utc_ns("2024-08-01T01:00:00Z"),
    ]
    reference = CombinedMonthForcing(
        ocm=_ocm_cross_month("continuous", continuous_times),
        nww=_nww_cross_month("continuous", continuous_times),
        projection=DomainProjection(121.0, 25.0),
        settling_velocity_mps=-0.002,
        include_stokes=True,
    )
    target_ns = _utc_ns("2024-08-01T00:00:00Z")
    actual = manager.provider(-0.002, include_stokes=True).sample(
        1.0,
        1.0,
        -5.0,
        target_ns,
        triangle_hint=0,
    )
    expected = reference.sample(1.0, 1.0, -5.0, target_ns, triangle_hint=0)
    assert actual.valid
    assert expected.valid
    _assert_velocity_equivalent(actual, expected)
    # OCM exact endpoint 來自 7 月 halo；不能因 query 的曆月是 8 月而改用 8 月 OCM。
    assert actual.forcing_month_id == "202407"


def test_cross_month_triangle_hint_reaches_both_ocm_endpoints() -> None:
    """跨月 endpoint 取樣仍須把同一 triangle hint 傳給兩個 OCM 時間列。"""

    july_end = [_utc_ns("2024-07-31T23:00:00Z"), _utc_ns("2024-08-01T00:00:00Z")]
    august_start = [_utc_ns("2024-08-01T01:00:00Z"), _utc_ns("2024-08-01T02:00:00Z")]
    july_ocm = _ocm_cross_month("202407", july_end)
    august_ocm = _ocm_cross_month("202408", august_start)
    geometry_spies = {}
    endpoint_spies = {}
    for month_id, ocm in (("202407", july_ocm), ("202408", august_ocm)):
        geometry_spies[month_id] = Mock(wraps=ocm.geometry_at_time_index)
        endpoint_spies[month_id] = Mock(wraps=ocm.sample_at_time_index)
        ocm.geometry_at_time_index = geometry_spies[month_id]  # type: ignore[method-assign]
        ocm.sample_at_time_index = endpoint_spies[month_id]  # type: ignore[method-assign]
    manager, _, _ = _manager({"202407": july_ocm, "202408": august_ocm})
    result = manager.provider(-0.002, include_stokes=False).sample(
        1.0,
        1.0,
        -5.0,
        _utc_ns("2024-08-01T00:30:00Z"),
        triangle_hint=0,
    )
    assert result.valid
    for spy in (*geometry_spies.values(), *endpoint_spies.values()):
        assert spy.call_count == 1
        assert spy.call_args.kwargs["triangle_hint"] == 0


def test_cross_month_numba_ocm_matches_continuous_provider() -> None:
    """跨月 OCM 啟用 Numba 時，兩端結果仍應等同連續 NumPy 參考 provider。"""

    july_end = [_utc_ns("2024-07-31T23:00:00Z"), _utc_ns("2024-08-01T00:00:00Z")]
    august_start = [_utc_ns("2024-08-01T01:00:00Z"), _utc_ns("2024-08-01T02:00:00Z")]
    manager, _, _ = _manager(
        {
            "202407": _ocm_cross_month("202407", july_end, use_numba_kernel=True),
            "202408": _ocm_cross_month("202408", august_start, use_numba_kernel=True),
        }
    )
    continuous = _ocm_cross_month(
        "continuous",
        july_end + august_start,
        use_numba_kernel=False,
    )
    reference = CombinedMonthForcing(
        ocm=continuous,
        nww=None,
        projection=DomainProjection(121.0, 25.0),
        settling_velocity_mps=-0.002,
        include_stokes=False,
    )
    target_ns = _utc_ns("2024-08-01T00:30:00Z")
    actual = manager.provider(-0.002, include_stokes=False).sample(
        1.0,
        1.0,
        -5.0,
        target_ns,
        triangle_hint=0,
    )
    expected = reference.sample(1.0, 1.0, -5.0, target_ns, triangle_hint=0)
    assert actual.valid
    assert expected.valid
    _assert_velocity_equivalent(actual, expected)


def test_cross_month_gap_over_maximum_time_gap_remains_time_gap() -> None:
    """跨月端點間隔超過上限時必須回傳 TIME_GAP，不可採最近值或外插。"""

    july_end = [_utc_ns("2024-07-31T23:00:00Z"), _utc_ns("2024-08-01T00:00:00Z")]
    august_start = [_utc_ns("2024-08-01T02:00:00Z"), _utc_ns("2024-08-01T03:00:00Z")]
    manager, _, _ = _manager(
        {
            "202407": _ocm_cross_month("202407", july_end, maximum_time_gap_seconds=3_600.0),
            "202408": _ocm_cross_month(
                "202408", august_start, maximum_time_gap_seconds=3_600.0
            ),
        }
    )
    actual = manager.provider(-0.002, include_stokes=False).sample(
        1.0,
        1.0,
        -5.0,
        _utc_ns("2024-08-01T01:00:00Z"),
        triangle_hint=0,
    )
    continuous = _ocm_cross_month(
        "continuous",
        july_end[:1] + july_end[1:] + august_start,
        maximum_time_gap_seconds=3_600.0,
    ).sample(1.0, 1.0, -5.0, _utc_ns("2024-08-01T01:00:00Z"), triangle_hint=0)
    assert not continuous.valid
    assert continuous.qc == SampleQC.TIME_GAP
    assert not actual.valid
    assert actual.qc == SampleQC.TIME_GAP
    assert np.isnan(actual.eta_m)
    assert np.isnan(actual.bed_z_m)
    assert actual.components is None


def test_cross_month_nww_invalid_corner_remains_wave_unsupported() -> None:
    """跨月 NWW 任一必要四角失效時仍須回傳 WAVE_UNSUPPORTED，不可補零 Stokes。"""

    july_end = [_utc_ns("2024-07-31T23:00:00Z"), _utc_ns("2024-08-01T00:00:00Z")]
    august_start = [_utc_ns("2024-08-01T01:00:00Z"), _utc_ns("2024-08-01T02:00:00Z")]
    july_nww = _nww_cross_month("202407", july_end)
    august_nww = _nww_cross_month("202408", august_start)
    # 只遮罩一個四角即可使保守 bilinear 規則失效；其餘角仍保留有效數值，避免
    # 測試退化成整月缺失（那應由另一個既有測試負責）。
    august_nww.valid_mask_wave[0, 0, 0] = False
    manager, _, nww_loader = _manager(
        {
            "202407": _ocm_cross_month("202407", july_end),
            "202408": _ocm_cross_month("202408", august_start),
        },
        {"202407": july_nww, "202408": august_nww},
    )
    sample = manager.provider(-0.002, include_stokes=True).sample(
        1.0,
        1.0,
        -5.0,
        _utc_ns("2024-08-01T00:59:45Z"),
        triangle_hint=0,
    )
    assert not sample.valid
    assert sample.qc == SampleQC.WAVE_UNSUPPORTED
    assert sample.components is None
    assert nww_loader.call_count >= 2


def test_cross_month_nww_gap_over_maximum_time_gap_remains_time_gap() -> None:
    """NWW 跨月時間端點間隔過大時應保留 TIME_GAP，而非沿用鄰近波浪值。"""

    july_end = [_utc_ns("2024-07-31T23:00:00Z"), _utc_ns("2024-08-01T00:00:00Z")]
    august_start = [_utc_ns("2024-08-01T02:00:00Z"), _utc_ns("2024-08-01T03:00:00Z")]
    manager, _, _ = _manager(
        {
            "202407": _ocm_cross_month("202407", july_end),
            "202408": _ocm_cross_month("202408", august_start),
        },
        {
            "202407": _nww_cross_month(
                "202407", july_end, maximum_time_gap_seconds=3_600.0
            ),
            "202408": _nww_cross_month(
                "202408", august_start, maximum_time_gap_seconds=3_600.0
            ),
        },
    )
    sample = manager.provider(-0.002, include_stokes=True).sample(
        1.0,
        1.0,
        -5.0,
        _utc_ns("2024-08-01T01:00:00Z"),
        triangle_hint=0,
    )
    assert not sample.valid
    assert sample.qc == SampleQC.TIME_GAP
    assert not np.isnan(sample.eta_m)
    assert sample.components is None


def test_nww_second_endpoint_failure_preserves_prior_qc_flags() -> None:
    """NWW 第二時間端點失效時，仍保留第一端已累積的角點 QC flags。"""

    times = [_utc_ns("2024-08-01T00:00:00Z"), _utc_ns("2024-08-01T01:00:00Z")]
    nww = _nww_cross_month("202408", times)
    nww.qc_flags[0, ...] = np.uint16(0x0001)
    nww.qc_flags[1, ...] = np.uint16(0x0002)
    nww.valid_mask_wave[1, 0, 0] = False
    result = nww.sample(121.0, 25.0, _utc_ns("2024-08-01T00:30:00Z"))
    assert not result.valid
    assert result.qc == SampleQC.WAVE_UNSUPPORTED
    assert result.qc_flags == 0x0003


def _assert_diffusion_equivalent(actual, expected) -> None:
    """逐欄比對跨月 Smagorinsky sample 與連續 OCM 參考結果。"""

    assert actual.qc == expected.qc
    for field in ("kx_m2ps", "ky_m2ps", "kz_m2ps"):
        assert np.isclose(
            getattr(actual.coefficients, field),
            getattr(expected.coefficients, field),
            rtol=1e-12,
            atol=1e-12,
        )
    assert np.allclose(
        actual.diffusivity_divergence_mps,
        expected.diffusivity_divergence_mps,
        rtol=1e-12,
        atol=1e-12,
    )
    assert actual.diagnostics.keys() == expected.diagnostics.keys()
    for key, actual_value in actual.diagnostics.items():
        expected_value = expected.diagnostics[key]
        if isinstance(actual_value, (float, int)) and isinstance(expected_value, (float, int)):
            assert np.isclose(actual_value, expected_value, rtol=1e-12, atol=1e-12)
        else:
            assert actual_value == expected_value


def test_cross_month_smagorinsky_matches_continuous_ocm_and_keeps_nww_lazy() -> None:
    """Smagorinsky 跨月取樣應等價於連續 OCM，且仍不載入 NWW。"""

    july_end = [_utc_ns("2024-07-31T23:00:00Z"), _utc_ns("2024-08-01T00:00:00Z")]
    august_start = [_utc_ns("2024-08-01T01:00:00Z"), _utc_ns("2024-08-01T02:00:00Z")]
    manager, _, nww_loader = _manager(
        {
            "202407": _ocm_cross_month("202407", july_end),
            "202408": _ocm_cross_month("202408", august_start),
        },
        {
            "202407": _nww_cross_month("202407", july_end),
            "202408": _nww_cross_month("202408", august_start),
        },
    )
    settings = SmagorinskySettings(
        coefficient_cs=0.15,
        floor_m2ps=0.0001,
        cap_m2ps=10.0,
        constant_kz_m2ps=0.02,
    )
    continuous = _ocm_cross_month(
        "continuous",
        july_end + august_start,
    )
    provider = manager.smagorinsky_provider(settings)
    for time_ns in (
        _utc_ns("2024-08-01T00:30:00Z"),
        _utc_ns("2024-08-01T00:59:45Z"),
    ):
        actual = provider.sample(1.0, 1.0, -5.0, time_ns, triangle_hint=0)
        expected = continuous.sample_smagorinsky_diffusion(
            1.0,
            1.0,
            -5.0,
            time_ns,
            settings,
            triangle_hint=0,
        )
        assert actual.valid
        assert expected.valid
        _assert_diffusion_equivalent(actual, expected)
    assert nww_loader.call_count == 0


@pytest.mark.parametrize("max_resident_months", [1, 2])
def test_cross_month_lru_capacity_keeps_samples_correct_and_stats_bounded(
    max_resident_months: int,
) -> None:
    """cache 容量為 1／2 時跨月載入與淘汰不應改變取樣值或突破 resident 上限。"""

    july_end = [_utc_ns("2024-07-31T23:00:00Z"), _utc_ns("2024-08-01T00:00:00Z")]
    august_start = [_utc_ns("2024-08-01T01:00:00Z"), _utc_ns("2024-08-01T02:00:00Z")]
    manager, ocm_loader, _ = _manager(
        {
            "202407": _ocm_cross_month("202407", july_end),
            "202408": _ocm_cross_month("202408", august_start),
        },
        max_resident_months=max_resident_months,
    )
    continuous = _ocm_cross_month("continuous", july_end + august_start)
    provider = manager.provider(-0.002, include_stokes=False)
    for time_ns in (
        _utc_ns("2024-07-31T23:30:00Z"),
        _utc_ns("2024-08-01T00:30:00Z"),
        _utc_ns("2024-08-01T01:30:00Z"),
        _utc_ns("2024-08-01T00:59:45Z"),
    ):
        actual = provider.sample(1.0, 1.0, -5.0, time_ns, triangle_hint=0)
        expected = CombinedMonthForcing(
            ocm=continuous,
            nww=None,
            projection=DomainProjection(121.0, 25.0),
            settling_velocity_mps=-0.002,
            include_stokes=False,
        ).sample(1.0, 1.0, -5.0, time_ns, triangle_hint=0)
        assert actual.valid
        _assert_velocity_equivalent(actual, expected)
    stats = manager.cache_stats
    assert len(stats.resident_month_ids) <= max_resident_months
    assert stats.ocm_load_count >= 2
    assert stats.ocm_cache_miss_count >= 2
    assert stats.resident_ndarray_bytes > 0
    if max_resident_months == 1:
        assert stats.eviction_count >= 1
    assert ocm_loader.call_count == stats.ocm_load_count


def test_cross_year_duplicate_endpoint_prefers_later_month_and_keeps_december_january() -> None:
    """跨年月界若兩產品重複同一 exact 時刻，應採後月份 endpoint 並維持連續結果。"""

    december_times = [
        _utc_ns("2024-12-31T23:00:00Z"),
        _utc_ns("2025-01-01T00:00:00Z"),
    ]
    january_times = [
        _utc_ns("2025-01-01T00:00:00Z"),
        _utc_ns("2025-01-01T01:00:00Z"),
    ]
    january_origin_ns = january_times[0]
    december_ocm = _ocm_cross_month(
        "202412", december_times, reference_ns=january_origin_ns
    )
    january_ocm = _ocm_cross_month("202501", january_times, reference_ns=january_origin_ns)
    december_nww = _nww_cross_month("202412", december_times, reference_ns=january_origin_ns)
    january_nww = _nww_cross_month("202501", january_times, reference_ns=january_origin_ns)
    # 後月份的重複 endpoint 特意使用不同值；若合併器採前一筆，exact 00:00 與
    # 00:30 的 current／wave／Stokes 都會出現可觀察的不連續。
    january_ocm.hvel[0, ..., 0] += 0.4
    january_ocm.hvel[0, ..., 1] -= 0.2
    january_ocm.vertical_velocity[0] += 0.01
    january_ocm.diffusivity[0] += 0.02
    january_ocm.elev[0] += 0.4
    january_nww.significant_wave_height[0] += 0.4
    january_nww.peak_frequency[0] += 0.01
    january_nww.peak_direction_raw_deg[0] = (
        january_nww.peak_direction_raw_deg[0] + 20.0
    ) % 360.0

    manager, _, _ = _manager(
        {"202412": december_ocm, "202501": january_ocm},
        {"202412": december_nww, "202501": january_nww},
    )
    continuous_ocm = _ocm_cross_month(
        "continuous",
        [december_times[0], january_times[0], january_times[1]],
        reference_ns=january_origin_ns,
    )
    continuous_nww = _nww_cross_month(
        "continuous",
        [december_times[0], january_times[0], january_times[1]],
        reference_ns=january_origin_ns,
    )
    for target, source in (
        (continuous_ocm.hvel[1], january_ocm.hvel[0]),
        (continuous_ocm.vertical_velocity[1], january_ocm.vertical_velocity[0]),
        (continuous_ocm.zcor[1], january_ocm.zcor[0]),
        (continuous_ocm.elev[1], january_ocm.elev[0]),
        (continuous_ocm.wetdry_elem[1], january_ocm.wetdry_elem[0]),
        (continuous_ocm.diffusivity[1], january_ocm.diffusivity[0]),
    ):
        target[...] = source
    for target, source in (
        (continuous_nww.significant_wave_height[1], january_nww.significant_wave_height[0]),
        (continuous_nww.peak_frequency[1], january_nww.peak_frequency[0]),
        (continuous_nww.peak_direction_raw_deg[1], january_nww.peak_direction_raw_deg[0]),
        (continuous_nww.valid_mask_wave[1], january_nww.valid_mask_wave[0]),
        (continuous_nww.qc_flags[1], january_nww.qc_flags[0]),
    ):
        target[...] = source
    reference = CombinedMonthForcing(
        ocm=continuous_ocm,
        nww=continuous_nww,
        projection=DomainProjection(121.0, 25.0),
        settling_velocity_mps=-0.002,
        include_stokes=True,
    )
    provider = manager.provider(-0.002, include_stokes=True)
    for time_ns in (
        _utc_ns("2024-12-31T23:59:59Z"),
        _utc_ns("2025-01-01T00:00:00Z"),
        _utc_ns("2025-01-01T00:30:00Z"),
    ):
        actual = provider.sample(1.0, 1.0, -5.0, time_ns, triangle_hint=0)
        expected = reference.sample(1.0, 1.0, -5.0, time_ns, triangle_hint=0)
        assert actual.valid
        assert expected.valid
        _assert_velocity_equivalent(actual, expected)
    assert manager.cache_stats.ocm_load_count == 2
    assert manager.cache_stats.resident_month_ids == ("202412", "202501")


def _manager(
    ocm_by_month: dict[str, OCMNativeMonth],
    nww_by_month: dict[str, NWWAnalysisMonth] | None = None,
    *,
    max_resident_months: int = 2,
) -> tuple[ForcingWindowManager, Mock, Mock]:
    """建立注入式 manager 並回傳 loader mocks，方便檢查 lazy load 次數。"""

    ocm_loader = Mock(side_effect=lambda month_id: ocm_by_month.get(month_id))
    nww_loader = Mock(
        side_effect=lambda month_id: None if nww_by_month is None else nww_by_month.get(month_id)
    )
    manager = ForcingWindowManager(
        flow_domain_id="synthetic_domain",
        projection=DomainProjection(121.0, 25.0),
        mesh=_mesh(),
        ocm_loader=ocm_loader,
        nww_loader=nww_loader,
        max_resident_months=max_resident_months,
    )
    return manager, ocm_loader, nww_loader


def _sample_time(month_id: str, *, offset_seconds: int = 1_800) -> int:
    """回傳月份內的 RK stage 測試時間。"""

    return _month_start_ns(month_id) + offset_seconds * 1_000_000_000


def _ocm_backend_parity_shard():
    """建立兩個 backend 共用的固定 scenario／seed 工作單位。"""

    scenario = Scenario(
        scenario_id="ocm-backend-parity",
        study_site_id="synthetic-site",
        analysis_region_id="synthetic-region",
        material_id="synthetic-material",
        receptor_id="synthetic-receptor",
        arrival_time_id="synthetic-arrival",
        settling_velocity_mps=-0.002,
        arrival_time_utc_ns=_utc_ns("2024-08-01T00:00:30Z"),
        design_version="ocm-backend-parity-v1",
    )
    return plan_scenario_shards(
        [scenario],
        members_per_scenario=2,
        shard_scenario_count=1,
        experiment_case_id="no_stokes",
    )[0]


def _ocm_backend_request_factory(use_numba_kernel: bool):
    """以相同跨月 OCM fixture 建立 production request factory。

    July 的最後一個時間列與 August 的第一個時間列分別提供月界前後端點；兩種
    backend 只改變 OCM 內層垂向／水平／時間插值的實作。固定一秒步長、四秒回溯、
    非零水平擴散與窄 local domain 讓測試同時產生多步觀測、亂數與 local 邊界事件，
    但粒子始終留在 synthetic native triangle 內，不把域外資料當成可用速度。
    """

    july_end = [_utc_ns("2024-07-31T23:00:00Z"), _utc_ns("2024-08-01T00:00:00Z")]
    august_start = [_utc_ns("2024-08-01T01:00:00Z"), _utc_ns("2024-08-01T02:00:00Z")]
    manager, _, _ = _manager(
        {
            "202407": _ocm_cross_month(
                "202407", july_end, use_numba_kernel=use_numba_kernel
            ),
            "202408": _ocm_cross_month(
                "202408", august_start, use_numba_kernel=use_numba_kernel
            ),
        }
    )
    provider = manager.provider(-0.002, include_stokes=False)
    boundaries = BoundaryGeometry(
        own_local_domain=box(1.95, 0.5, 8.5, 8.5),
        flow_domain=box(0.5, 0.5, 9.5, 9.5),
        foreign_local_domains={},
    )
    settings = EngineSettings(
        dt_min_seconds=1.0,
        dt_max_seconds=1.0,
        output_interval_seconds=1.0,
        max_backtrack_seconds=4.0,
        maximum_step_count=32,
        earliest_forcing_time_utc_ns=0,
    )

    def factory(unit):
        """對兩個 member 使用同一 forcing，但由 unit 保留獨立 seed 與 identity。"""

        state = ParticleState(
            particle_id=unit.particle_id,
            scenario_id=unit.scenario.scenario_id,
            member_id=unit.member_id,
            study_site_id=unit.scenario.study_site_id,
            analysis_region_id=unit.scenario.analysis_region_id,
            receptor_id=unit.scenario.receptor_id,
            x_m=2.0,
            y_m=2.0,
            z_m=-5.0,
            time_utc_ns=unit.scenario.arrival_time_utc_ns,
        )
        return ReferenceParticleRequest(
            initial_state=state,
            velocity=provider,
            boundaries=boundaries,
            behavior_class="sinking",
            diffusion=DiffusionCoefficients(1.0e-4, 1.0e-4, 0.0),
            settings=settings,
        )

    return factory


def _ocm_backend_batch(shard, *, use_numba_kernel: bool) -> ProductionBatch:
    """以固定 scenario／master seed 建立指定 OCM backend 的 production batch。"""

    return ProductionBatch(
        shard,
        master_seed=20260914,
        request_factory=_ocm_backend_request_factory(use_numba_kernel),
        active_chunk_size=2,
    )


def _ocm_backend_config_hash(backend: str) -> str:
    """從範例設定衍生明示 backend hash，供 checkpoint binding 使用。"""

    config_path = Path(__file__).resolve().parents[1] / "configs" / "lagrangian_backtracking.example.yaml"
    config = load_config(config_path)
    execution = config.execution.model_copy(update={"ocm_interpolation_backend": backend})
    return config.model_copy(update={"execution": execution}).config_hash()


def _ocm_backend_binding(shard, backend: str) -> CheckpointBinding:
    """建立只差 OCM backend config hash 的 checkpoint binding。"""

    return CheckpointBinding(
        config_hash=_ocm_backend_config_hash(backend),
        input_inventory_hash="synthetic-ocm-inventory",
        experiment_case_id=shard.experiment_case_id,
        shard_id=shard.shard_id,
        seed_policy="pcg64dxsm-v1",
        code_commit="synthetic-ocm-commit",
    )


def _assert_backend_values_equivalent(actual, expected) -> None:
    """遞迴比較完整結果；缺值／狀態 exact，有限浮點採固定嚴格容差。"""

    if actual is None or expected is None:
        assert actual == expected
    elif isinstance(actual, float) or isinstance(expected, float):
        np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)
    elif isinstance(actual, dict) and isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_backend_values_equivalent(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_backend_values_equivalent(actual_item, expected_item)
    else:
        assert actual == expected


def _assert_particle_results_backend_equivalent(actual, expected) -> None:
    """比較兩 backend 的完整狀態、觀測、事件、QC 與步數。"""

    assert len(actual) == len(expected)
    for actual_result, expected_result in zip(actual, expected, strict=True):
        _assert_backend_values_equivalent(asdict(actual_result), asdict(expected_result))


def _assert_batch_rng_states_equal(actual: ProductionBatch, expected: ProductionBatch) -> None:
    """確認每個固定 particle seed 的 PCG64DXSM continuation state 完全一致。"""

    assert [runtime.rng.bit_generator.state for runtime in actual.runtimes] == [
        runtime.rng.bit_generator.state for runtime in expected.runtimes
    ]


def test_ocm_backends_match_full_multistep_trajectory_and_rng_state() -> None:
    """同一跨月 forcing、scenario 與 seed 的 NumPy／Numba 完整軌跡應保持一致。"""

    shard = _ocm_backend_parity_shard()
    numpy_batch = _ocm_backend_batch(shard, use_numba_kernel=False)
    numba_batch = _ocm_backend_batch(shard, use_numba_kernel=True)
    numpy_results = numpy_batch.complete()
    numba_results = numba_batch.complete()

    _assert_particle_results_backend_equivalent(numba_results, numpy_results)
    _assert_batch_rng_states_equal(numba_batch, numpy_batch)
    assert any(result.events for result in numpy_results)
    assert all(result.observations for result in numpy_results)
    assert all(
        observation.velocity_qc_flags == 0
        for result in numpy_results
        for observation in result.observations
        if observation.velocity_sample_status.value == "complete"
    )


@pytest.mark.parametrize(
    ("backend", "use_numba_kernel"),
    [("numpy_v1", False), ("numba_ocm_v1", True)],
)
def test_each_ocm_backend_checkpoint_resume_and_cross_backend_binding(
    tmp_path: Path, backend: str, use_numba_kernel: bool
) -> None:
    """各 backend 的 checkpoint resume 應等於不中斷結果，跨 backend binding 必須拒絕。"""

    shard = _ocm_backend_parity_shard()
    binding = _ocm_backend_binding(shard, backend)
    other_backend = "numba_ocm_v1" if backend == "numpy_v1" else "numpy_v1"
    other_binding = _ocm_backend_binding(shard, other_backend)
    assert binding.config_hash != other_binding.config_hash

    uninterrupted = _ocm_backend_batch(shard, use_numba_kernel=use_numba_kernel)
    uninterrupted_results = uninterrupted.complete()
    interrupted = _ocm_backend_batch(shard, use_numba_kernel=use_numba_kernel)
    interrupted.advance(sweeps=2)
    assert interrupted.active_count > 0
    checkpoint_path = interrupted.write_checkpoint(
        tmp_path / backend,
        binding=binding,
        # checkpoint sequence 表示不可變 generation 的連續編號，不是已完成的
        # sweep 數；首次寫入固定從 1 開始，才能驗證 schema 3 hash chain。
        sequence=1,
    )
    resumed = ProductionBatch.from_checkpoint(
        checkpoint_path,
        shard=shard,
        master_seed=20260914,
        request_factory=_ocm_backend_request_factory(use_numba_kernel),
        expected_binding=binding,
        active_chunk_size=2,
    )
    resumed_results = resumed.complete()
    _assert_particle_results_backend_equivalent(resumed_results, uninterrupted_results)
    _assert_batch_rng_states_equal(resumed, uninterrupted)

    with pytest.raises(ValueError, match="config_hash"):
        ProductionBatch.from_checkpoint(
            checkpoint_path,
            shard=shard,
            master_seed=20260914,
            request_factory=_ocm_backend_request_factory(not use_numba_kernel),
            expected_binding=other_binding,
            active_chunk_size=2,
        )


def test_same_month_material_facades_share_ocm_and_no_stokes_never_loads_nww() -> None:
    """不同沉降速度只建立輕量合併 facade，同月 OCM 及 no-Stokes NWW 都只讀一次／零次。"""

    month = "197001"
    manager, ocm_loader, nww_loader = _manager({month: _ocm(month, _mesh())})
    first = manager.provider(-0.1, include_stokes=False)
    second = manager.provider(-0.2, include_stokes=False)
    time_ns = _sample_time(month)
    assert first.sample(1.0, 1.0, -5.0, time_ns, triangle_hint=0).valid
    assert second.sample(1.0, 1.0, -5.0, time_ns, triangle_hint=0).valid
    assert ocm_loader.call_count == 1
    assert nww_loader.call_count == 0
    stats = manager.cache_stats
    assert stats.ocm_load_count == 1
    assert stats.ocm_cache_miss_count == 1
    assert stats.ocm_cache_hit_count >= 1
    assert stats.resident_ndarray_bytes > 0


def test_same_month_hot_path_counts_one_hit_per_product_after_warmup() -> None:
    """同月安全查詢暖機後每次只記一次 OCM hit，Stokes 再記一次 NWW hit。

    測試使用月份內的半小時查詢，避免月尾 halo 與跨月 endpoint；因此若 manager 在
    建立全域 bracket 前直接沿用單月 ``CombinedMonthForcing``，統計增量應只包含
    一次 query 月 OCM cache hit。Stokes 先由 no-Stokes 查詢暖機 OCM，再以一次 NWW
    load 建立 wave facade，後續暖機查詢應只增加一個 OCM 與一個 NWW hit，不能因
    重複探查同一月份污染正式 Phase 3B 的 cache 指標。
    """

    month = "197001"
    manager, _, _ = _manager({month: _ocm(month, _mesh())}, {month: _nww(month)})
    time_ns = _sample_time(month)
    no_stokes = manager.provider(-0.1, include_stokes=False)
    assert no_stokes.sample(1.0, 1.0, -5.0, time_ns, triangle_hint=0).valid
    before_no_stokes = manager.cache_stats
    assert no_stokes.sample(1.0, 1.0, -5.0, time_ns, triangle_hint=0).valid
    after_no_stokes = manager.cache_stats
    assert after_no_stokes.ocm_cache_hit_count - before_no_stokes.ocm_cache_hit_count == 1
    assert after_no_stokes.nww_cache_hit_count - before_no_stokes.nww_cache_hit_count == 0

    stokes = manager.provider(-0.1, include_stokes=True)
    assert stokes.sample(1.0, 1.0, -5.0, time_ns, triangle_hint=0).valid
    before_stokes = manager.cache_stats
    assert stokes.sample(1.0, 1.0, -5.0, time_ns, triangle_hint=0).valid
    after_stokes = manager.cache_stats
    assert after_stokes.ocm_cache_hit_count - before_stokes.ocm_cache_hit_count == 1
    assert after_stokes.nww_cache_hit_count - before_stokes.nww_cache_hit_count == 1


def test_month_metadata_cache_reuses_hot_path_parse_and_next_month_start() -> None:
    """同一月份反覆判斷熱路徑時，解析與下一月月初計算只在首次建立快取項目時發生。

    直接觀察 production helper 的 ``cache_info``，而不是在測試中重做月份計算；首次
    呼叫會建立查詢月份與下一月份兩個 metadata，第二次呼叫只增加命中數。這保留月界
    halo 判斷依賴實際時間軸的原則，同時避免每個 OCM/NWW 速度樣本再次呼叫
    ``datetime.strptime``。
    """

    forcing_window_module._cached_month_metadata.cache_clear()
    month = "202412"
    forcing = _ocm(month, _mesh())
    target_ns = _month_start_ns(month) + 1_800 * 1_000_000_000

    assert forcing_window_module.ForcingWindowManager._is_same_month_hot_path_safe(
        month, target_ns, forcing
    )
    first = forcing_window_module._cached_month_metadata.cache_info()
    assert first.misses == 2
    assert first.currsize == 2

    assert forcing_window_module.ForcingWindowManager._is_same_month_hot_path_safe(
        month, target_ns, forcing
    )
    second = forcing_window_module._cached_month_metadata.cache_info()
    assert second.misses == first.misses
    assert second.hits == first.hits + 2

    forcing_window_module._cached_month_metadata.cache_clear()


@pytest.mark.parametrize("month_id", ["202400", "202413", "20A401", "20241"])
def test_month_metadata_cache_rejects_invalid_month_without_caching(month_id: str) -> None:
    """非法月份維持既有 ValueError，且不因錯誤輸入占用有限月份快取。"""

    forcing_window_module._cached_month_metadata.cache_clear()
    with pytest.raises(ValueError, match="month_id"):
        forcing_window_module._validate_month_id(month_id)
    assert forcing_window_module._cached_month_metadata.cache_info().currsize == 0


def test_month_metadata_cache_rejects_non_string_month_id_without_lru_type_error() -> None:
    """非字串 month_id 仍由驗證層回報 ValueError，不暴露 lru_cache 的雜湊錯誤。"""

    forcing_window_module._cached_month_metadata.cache_clear()
    with pytest.raises(ValueError, match="month_id 必須是 YYYYMM"):
        forcing_window_module._validate_month_id(202412)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="month_id 必須是 YYYYMM"):
        forcing_window_module._validate_month_id(["202412"])  # type: ignore[arg-type]
    assert forcing_window_module._cached_month_metadata.cache_info().currsize == 0


def test_adjacent_month_metadata_cache_preserves_cross_year_and_datetime_bounds() -> None:
    """跨年相鄰月份與 datetime 首尾界線維持原本的曆法結果與錯誤契約。"""

    forcing_window_module._cached_month_metadata.cache_clear()
    assert forcing_window_module._adjacent_month_id("202412", 1) == "202501"
    assert forcing_window_module._adjacent_month_id("202501", -1) == "202412"
    assert forcing_window_module._month_start_ns("202501") == _month_start_ns("202501")
    assert forcing_window_module._month_start_ns("000101") == _month_start_ns("000101")
    assert forcing_window_module._month_start_ns("999912") == _month_start_ns("999912")
    with pytest.raises(ValueError, match="超出 datetime 支援範圍"):
        forcing_window_module._adjacent_month_id("000101", -1)
    with pytest.raises(ValueError, match="超出 datetime 支援範圍"):
        forcing_window_module._adjacent_month_id("999912", 1)
    forcing_window_module._cached_month_metadata.cache_clear()


def test_stokes_loads_nww_lazily_once_after_no_stokes() -> None:
    """先使用 no-Stokes 不碰 NWW；同月後續 Stokes 只觸發一次 NWW load。"""

    month = "197001"
    manager, ocm_loader, nww_loader = _manager(
        {month: _ocm(month, _mesh())}, {month: _nww(month)}
    )
    time_ns = _sample_time(month)
    manager.provider(-0.1, False).sample(1.0, 1.0, -5.0, time_ns)
    assert nww_loader.call_count == 0
    stokes = manager.provider(-0.1, True)
    result = stokes.sample(1.0, 1.0, -5.0, time_ns, triangle_hint=0)
    assert result.valid
    stokes.sample(1.0, 1.0, -5.0, time_ns, triangle_hint=0)
    assert ocm_loader.call_count == 1
    assert nww_loader.call_count == 1
    assert manager.cache_stats.nww_load_count == 1


def test_missing_ocm_and_missing_nww_have_distinct_qc() -> None:
    """OCM 月份缺失是 OUTSIDE_TIME_RANGE，OCM 存在但 NWW 缺月是 WAVE_UNSUPPORTED。"""

    manager, _, nww_loader = _manager({})
    missing_ocm = manager.provider(-0.1, False).sample(
        1.0, 1.0, -5.0, _sample_time("197001")
    )
    assert missing_ocm.qc == SampleQC.OUTSIDE_TIME_RANGE
    assert nww_loader.call_count == 0

    month = "197002"
    manager, _, nww_loader = _manager({month: _ocm(month, _mesh())}, {})
    missing_nww = manager.provider(-0.1, True).sample(
        1.0, 1.0, -5.0, _sample_time(month)
    )
    assert missing_nww.qc == SampleQC.WAVE_UNSUPPORTED
    assert nww_loader.call_count == 1


def test_lru_eviction_reloads_month_and_preload_uses_same_contract() -> None:
    """容量一個時跨月會淘汰並重新載入；preload 仍沿用同一 LRU，不繞過月份驗證。"""

    jan, feb = "197001", "197002"
    manager, ocm_loader, _ = _manager(
        {jan: _ocm(jan, _mesh()), feb: _ocm(feb, _mesh())}, max_resident_months=1
    )
    provider = manager.provider(-0.1, False)
    for month in (jan, feb, jan):
        assert provider.sample(1.0, 1.0, -5.0, _sample_time(month)).valid
    manager.preload([feb])
    assert ocm_loader.call_count == 4
    assert manager.cache_stats.resident_month_ids == (feb,)
    assert manager.cache_stats.eviction_count == 3


def test_loader_error_for_existing_month_is_not_converted_to_qc() -> None:
    """注入 loader 的產品錯誤必須上拋，不能與整月不存在混為 OUTSIDE_TIME_RANGE。"""

    def broken_loader(month_id: str):
        """模擬已存在月份的 schema/array 讀取錯誤。"""

        del month_id
        raise ValueError("schema damaged")

    manager = ForcingWindowManager(
        flow_domain_id="synthetic_domain",
        projection=DomainProjection(121.0, 25.0),
        mesh=_mesh(),
        ocm_loader=broken_loader,
    )
    with pytest.raises(ValueError, match="schema damaged"):
        manager.provider(-0.1, False).sample(1.0, 1.0, -5.0, _sample_time("197001"))


def test_hint_is_forwarded_through_managed_facade() -> None:
    """managed facade 的 triangle_hint 應完整抵達 OCM sampler，不改變 forcing physics。"""

    month = "197001"
    ocm = _ocm(month, _mesh())
    spy = Mock(wraps=ocm.sample)
    ocm.sample = spy  # type: ignore[method-assign]
    manager, _, _ = _manager({month: ocm})
    result = manager.provider(-0.1, False).sample(
        1.0, 1.0, -5.0, _sample_time(month), triangle_hint=0
    )
    assert result.valid
    assert spy.call_args.kwargs["triangle_hint"] == 0


def test_managed_provider_and_hint_wrapper_enable_safe_k1_reuse() -> None:
    """真實 managed forcing facade 經 hint wrapper 後，engine 只查三個後續 RK stage。"""

    month = "197001"
    manager, ocm_loader, _ = _manager({month: _ocm(month, _mesh())})
    managed = manager.provider(-0.1, include_stokes=False)
    wrapped = HintTrackingVelocityProvider(managed)
    assert managed.step_start_sample_reuse_safe is True
    assert wrapped.step_start_sample_reuse_safe is True
    assert supports_step_start_sample_reuse(wrapped)

    start_time = _sample_time(month)
    state = ParticleState(
        particle_id="p0",
        scenario_id="s0",
        member_id=0,
        study_site_id="gongliao",
        analysis_region_id="A",
        receptor_id="r0",
        x_m=5.0,
        y_m=1.0,
        z_m=-5.0,
        time_utc_ns=start_time,
    )
    settings = EngineSettings(
        dt_min_seconds=1.0,
        dt_max_seconds=1.0,
        output_interval_seconds=1.0,
        max_backtrack_seconds=1.0,
        maximum_step_count=10,
        earliest_forcing_time_utc_ns=_month_start_ns(month),
    )
    execution = initialize_particle_execution(state, settings)
    result = advance_particle_once(
        execution,
        velocity=wrapped,
        boundaries=BoundaryGeometry(
            own_local_domain=box(-100.0, -100.0, 100.0, 100.0),
            flow_domain=box(-100.0, -100.0, 100.0, 100.0),
            foreign_local_domains={},
        ),
        behavior_class="sinking",
        diffusion=DiffusionCoefficients(0.0, 0.0, 0.0),
        settings=settings,
        rng=np.random.default_rng(11),
    )

    assert result.stepped
    assert execution.state.x_m == 4.0
    assert ocm_loader.call_count == 1
    # 初始 step-start sample 觸發一次 miss；k2、k3、k4 各命中一次，k1 沿用初始樣本。
    assert manager.cache_stats.ocm_cache_hit_count == 3


def test_smagorinsky_facade_shares_ocm_lru_and_never_loads_nww() -> None:
    """Smagorinsky facade 應共用 OCM LRU，且擴散取樣不建立 NWW／Combined forcing。"""

    month = "197001"
    manager, ocm_loader, nww_loader = _manager({month: _ocm(month, _mesh())})
    settings = SmagorinskySettings(
        coefficient_cs=0.10,
        floor_m2ps=0.01,
        cap_m2ps=10.0,
        constant_kz_m2ps=0.02,
    )
    first = manager.smagorinsky_provider(settings)
    second = manager.smagorinsky_provider(settings)
    first_result = first.sample(1.0, 1.0, -5.0, _sample_time(month), triangle_hint=0)
    second_result = second.sample(1.0, 1.0, -5.0, _sample_time(month), triangle_hint=0)
    assert first_result.valid
    assert second_result.valid
    assert first.manager is manager
    assert first.settings is settings
    assert ocm_loader.call_count == 1
    assert nww_loader.call_count == 0
    assert manager.cache_stats.nww_cache_miss_count == 0


def test_smagorinsky_facade_forwards_triangle_hint_and_missing_ocm_is_explicit_qc() -> None:
    """擴散 hint 必須到達 OCM；缺少整月 OCM 時需保留方法／月份／缺失診斷。"""

    month = "197001"
    ocm = _ocm(month, _mesh())
    spy = Mock(wraps=ocm.sample_smagorinsky_diffusion)
    ocm.sample_smagorinsky_diffusion = spy  # type: ignore[method-assign]
    manager, _, nww_loader = _manager({month: ocm})
    settings = SmagorinskySettings(0.15, 0.01, 10.0, 0.02)
    result = manager.smagorinsky_provider(settings).sample(
        1.0, 1.0, -5.0, _sample_time(month), triangle_hint=0
    )
    assert result.valid
    assert spy.call_args.kwargs["triangle_hint"] == 0
    assert spy.call_args.args[4] is settings
    assert nww_loader.call_count == 0

    missing_manager, _, missing_nww_loader = _manager({})
    missing = missing_manager.smagorinsky_provider(settings).sample(
        1.0, 1.0, -5.0, _sample_time(month), triangle_hint=0
    )
    assert not missing.valid
    assert missing.qc == SampleQC.OUTSIDE_TIME_RANGE
    assert missing.diagnostics["method"] == "smagorinsky_native_mesh_p1_nodal"
    assert missing.diagnostics["forcing_month_id"] == month
    assert missing.diagnostics["ocm_month_missing"] is True
    assert missing_nww_loader.call_count == 0


def test_smagorinsky_four_argument_call_keeps_existing_provider_shape() -> None:
    """facade 的四參數 ``__call__`` 介面不應偷偷需要 triangle hint。"""

    month = "197001"
    manager, _, _ = _manager({month: _ocm(month, _mesh())})
    provider = manager.smagorinsky_provider(SmagorinskySettings(0.10, 0.01, 10.0, 0.02))
    result = provider(1.0, 1.0, -5.0, _sample_time(month))
    assert result.valid


def _save_npy(path: Path, name: str, value: np.ndarray) -> None:
    """將測試 array 寫成 production reader 使用的 .npy 檔。"""

    np.save(path / name, value)


def _write_production_fixture(root: Path, month_id: str) -> None:
    """建立正式 root layout 的最小 OCM/NWW 陣列，驗證 from_roots 的檔案路徑契約。"""

    ocm_grid = root / "ocm" / "synthetic_domain" / "grid"
    ocm_month = root / "ocm" / "synthetic_domain" / "months" / month_id
    nww_grid = root / "nww" / "synthetic_domain" / "grid"
    nww_month = root / "nww" / "synthetic_domain" / "months" / month_id
    for directory in (ocm_grid, ocm_month, nww_grid, nww_month):
        directory.mkdir(parents=True)
    _save_npy(ocm_grid, "source_lon.npy", np.array([121.0, 121.001, 121.0]))
    _save_npy(ocm_grid, "source_lat.npy", np.array([25.0, 25.0, 25.001]))
    _save_npy(ocm_grid, "source_depth_m.npy", np.full(3, 10.0))
    _save_npy(ocm_grid, "source_node_bottom_index.npy", np.zeros(3, dtype=np.int64))
    _save_npy(ocm_grid, "source_face_nodes_local.npy", np.array([[0, 1, 2, -1]], dtype=np.int64))
    _save_npy(ocm_grid, "source_face_node_count.npy", np.array([3], dtype=np.int64))
    _save_npy(ocm_grid, "source_face_global_index.npy", np.array([99], dtype=np.int64))
    month_data = _ocm(month_id, _mesh())
    for name in (
        "time_utc_ns",
        "hvel",
        "vertical_velocity",
        "zcor",
        "elev",
        "wetdry_elem",
        "diffusivity",
    ):
        _save_npy(ocm_month, f"{name}.npy", getattr(month_data, name))
    nww_data = _nww(month_id)
    for name in (
        "time_utc_ns",
        "significant_wave_height",
        "peak_frequency",
        "peak_direction_raw_deg",
        "valid_mask_wave",
        "qc_flags",
    ):
        _save_npy(nww_month, f"{name}.npy", getattr(nww_data, name))
    _save_npy(nww_grid, "lon.npy", nww_data.lon)
    _save_npy(nww_grid, "lat.npy", nww_data.lat)


@pytest.mark.parametrize("use_numba_kernel", [False, True])
def test_from_roots_loads_mesh_once_and_production_missing_month_is_qc(
    tmp_path: Path, use_numba_kernel: bool
) -> None:
    """from_roots 使用固定 root layout，並把 OCM kernel 選擇傳到月份 reader。"""

    month = "197001"
    _write_production_fixture(tmp_path, month)
    manager = ForcingWindowManager.from_roots(
        flow_domain_id="synthetic_domain",
        projection=DomainProjection(121.0, 25.0),
        ocm_root=tmp_path / "ocm",
        nww_root=tmp_path / "nww",
        use_numba_kernel=use_numba_kernel,
    )
    assert manager.provider(-0.1, True).sample(1.0, 1.0, -5.0, _sample_time(month)).valid
    assert manager._months[month].ocm is not None
    assert manager._months[month].ocm.use_numba_kernel is use_numba_kernel
    missing = manager.provider(-0.1, False).sample(1.0, 1.0, -5.0, _sample_time("197002"))
    assert missing.qc == SampleQC.OUTSIDE_TIME_RANGE
    assert manager.cache_stats.ocm_load_count == 1


def test_from_roots_rejects_non_boolean_ocm_kernel_switch(tmp_path: Path) -> None:
    """OCM kernel 開關若不是真正 bool，應在讀取任何 root 前拒絕。"""

    with pytest.raises(TypeError, match="use_numba_kernel 必須是 bool"):
        ForcingWindowManager.from_roots(
            flow_domain_id="synthetic_domain",
            projection=DomainProjection(121.0, 25.0),
            ocm_root=tmp_path / "ocm",
            nww_root=None,
            use_numba_kernel=1,  # type: ignore[arg-type]
        )


def test_missing_month_exception_is_supported_by_injected_loader() -> None:
    """注入 loader 可用 MissingForcingMonth 明確標示整個月份目錄缺失。"""

    def missing_loader(month_id: str):
        """模擬不存在的 YYYYMM directory。"""

        raise MissingForcingMonth(month_id)

    manager = ForcingWindowManager(
        flow_domain_id="synthetic_domain",
        projection=DomainProjection(121.0, 25.0),
        mesh=_mesh(),
        ocm_loader=missing_loader,
    )
    assert (
        manager.provider(-0.1, False).sample(1.0, 1.0, -5.0, _sample_time("197001")).qc
        == SampleQC.OUTSIDE_TIME_RANGE
    )
