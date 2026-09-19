"""有限水深節點場與真正PI原版批次的整合驗證；合成資料不建立任何forcing產品。

常場驗證四段全量、沉降與identity；變動場明示量化兩種離散政策差異，不要求假等價。
所有位置單位m、時間UTC ns、Hs為m、fp為Hz；測試不抽RNG、不改PI原檔。
"""

import hashlib
from dataclasses import replace

import numpy as np
import pytest
from test_baytrace_ocm_adapter import _far_mesh, _mesh, _month, _sample, _state, _task
from test_baytrace_ocm_adapter import original_kernels as original_kernels

from lagrangian_backtracking.advection import execute_advection_task
from lagrangian_backtracking.baytrace_batch_advection import (
    NativeBayTraceBatchAdvector,
    NativeBayTraceUnsupportedError,
)
from lagrangian_backtracking.baytrace_ocm_adapter import OCMPreparedFrameResolver
from lagrangian_backtracking.baytrace_wave_frame import _node_wave_samples
from lagrangian_backtracking.forcing import NWWAnalysisMonth, OCMReconstructionPatchMonth
from lagrangian_backtracking.forcing_window import ForcingWindowManager, ManagedForcingProvider
from lagrangian_backtracking.geometry import DomainProjection
from lagrangian_backtracking.integrators import SamplingError
from lagrangian_backtracking.models import ParticleState, SampleQC
from lagrangian_backtracking.stokes import finite_depth_stokes


def _inputs(*, varying=False, wave_times=None, fallback=False, mesh=None, patch=None):
    """建立20m水深、三層OCM與2×2 NWW格網；零合成垂速保證粒子停在中層。

    變動案例僅改已存Hs/fp及OCM水平流速，不在轉接器補值。格網覆蓋公尺測試域。
    """
    mesh = _mesh() if mesh is None else mesh
    ocean = _month(mesh, patch=patch)
    if not varying:
        ocean.hvel[..., 0] = 0.1
        ocean.hvel[..., 1] = 0.05
    ocean.vertical_velocity[...] = 0.02
    times = ocean.time_utc_ns if wave_times is None else np.asarray(wave_times, dtype=np.int64)
    shape = (len(times), 2, 2)
    hs, fp = np.full(shape, 2.0), np.full(shape, 0.1)
    if varying:
        hs[:] = np.asarray([[1.0, 2.0], [2.0, 3.0]])
        hs[-1] *= 2
        fp[-1] *= 1.4
    wave = NWWAnalysisMonth(
        month_id="202401",
        lon=np.asarray([120.999, 121.002]),
        lat=np.asarray([24.999, 25.002]),
        time_utc_ns=times,
        significant_wave_height=hs,
        peak_frequency=fp,
        peak_direction_raw_deg=np.full(shape, 270.0),
        valid_mask_wave=np.ones(shape, dtype=bool),
        qc_flags=np.zeros(shape, dtype=np.uint16),
        continuous_time_fallback=fallback,
    )
    manager = ForcingWindowManager(
        flow_domain_id="wave-test",
        projection=DomainProjection(121.0, 25.0),
        mesh=mesh,
        ocm_loader=lambda month: ocean if month == ocean.month_id else None,
        nww_loader=lambda month: wave if month == wave.month_id else None,
        continuous_time_fallback=fallback,
        available_ocm_month_ids=("202401",),
        available_nww_month_ids=("202401",),
    )
    return ocean, wave, manager


def _provider(manager, wave=True):
    """固定沉降-0.02m/s只抵銷OCM +0.02m/s；不改任何隨機過程。"""
    return ManagedForcingProvider(manager, settling_velocity_mps=-0.02, include_stokes=wave)


def _expected(z=-10.0, hs=2.0, fp=0.1, direction=270.0):
    """以既有NumPy有限水深標量公式作獨立後端比對，不複製物理公式。"""
    return finite_depth_stokes(
        significant_wave_height_m=hs,
        peak_frequency_hz=fp,
        direction_raw_deg=direction,
        particle_z_m=z,
        surface_z_m=0.0,
        bed_z_m=-20.0,
    )


@pytest.mark.parametrize("with_wave", [False, True])
def test_actual_original_all_stages_and_settling_once(original_kernels, monkeypatch, with_wave):
    """四顆同時真推進；漂移為全量而非k1-only的1/6，純流不准讀NWW或退舊核心。"""
    _, waves, manager = _inputs()
    provider = _provider(manager, with_wave)

    def forbidden(*args, **kwargs):
        raise AssertionError("不可執行舊核心／純流讀波浪")

    monkeypatch.setattr(
        "lagrangian_backtracking.baytrace_batch_advection._execute_legacy_advection_task", forbidden
    )
    if not with_wave:
        monkeypatch.setattr(waves, "sample_at_time_index", forbidden)
        monkeypatch.setattr(manager, "_ensure_nww", forbidden)
    tasks = [
        _task(provider, state=_state(particle_id=f"p{i}", x_m=30.0 + i), dt_seconds=-10.0) for i in range(4)
    ]
    resolver = OCMPreparedFrameResolver()
    runner = NativeBayTraceBatchAdvector(original_kernels, resolver, require_native=True)
    results = runner(tasks)
    drift = _expected().u_mps if with_wave else 0.0
    for task, result in zip(tasks, results, strict=True):
        assert isinstance(result, ParticleState)
        assert result.particle_id == task.state.particle_id
        assert result.x_m - task.state.x_m == pytest.approx(-10.0 * (0.1 + drift), abs=1e-10)
        assert result.y_m - task.state.y_m == pytest.approx(-0.5, abs=1e-10)
        assert result.z_m == pytest.approx(-10.0, abs=1e-12)
        assert result.age_seconds == 10.0
        assert result.time_utc_ns == task.state.time_utc_ns - 10_000_000_000
    report = runner.report()
    assert report["native_completed_particle_steps"] == 4
    assert report["fallback_particle_steps"] == report["native_rejected_particle_steps"] == 0
    assert report["group_size_histogram"] == {"4": 1}
    assert resolver.report()["include_stokes"] is with_wave
    if with_wave:
        assert resolver.report()["field_policy"] == "finite_depth_node_fields_v1"
    # 實際dispatcher須已編譯，四個受控來源檔案在執行後SHA仍與載入證據一致。
    assert original_kernels.batch_interpolate_velocity.signatures
    assert original_kernels.batch_move_rk4.signatures
    for name, path in original_kernels.provenance.dependency_paths.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == report["source_sha256"][name]


def test_layers_match_existing_formula_and_cache_is_separate(monkeypatch):
    """兩端各層皆用有限水深Hs/fp公式；快取命中不重新取波浪，純流鍵獨立。"""
    _, waves, manager = _inputs()
    resolver = OCMPreparedFrameResolver()
    task = _task(_provider(manager))
    frame = resolver(task)
    assert frame is not None
    for field in (frame.fields.u1, frame.fields.u2):
        for layer, z in enumerate([-20.0, -10.0, 0.0]):
            np.testing.assert_allclose(field[layer], 0.1 + _expected(z).u_mps, rtol=2e-12)

    def forbidden(*args, **kwargs):
        raise AssertionError("命中端點快取不可重取NWW")

    monkeypatch.setattr(waves, "sample_at_time_index", forbidden)
    assert (
        resolver(replace(task, state=_state(time_utc_ns=task.state.time_utc_ns + 1))).fields is frame.fields
    )
    current = resolver(_task(_provider(manager, False)))
    np.testing.assert_allclose(current.fields.u1, 0.1)
    assert resolver.report()["frame_builds"] == 2


def test_circular_wave_direction_and_peak_frequency():
    """359/1度在時間與空間皆沿圓向量跨北，不成180度；fp仍為Hz。"""
    _, wave, manager = _inputs()
    wave.peak_direction_raw_deg[0] = 359.0
    wave.peak_direction_raw_deg[1] = 1.0
    wave.peak_frequency[1] = 0.2
    values, qc, _ = _node_wave_samples(manager, manager.mesh.node_xy, _state().time_utc_ns)
    assert np.all(qc == 0)
    np.testing.assert_allclose(values[:, 1], 0.15)
    assert np.all(np.minimum(values[:, 2], 360 - values[:, 2]) < 1e-10)
    wave.peak_direction_raw_deg[:, :, 0] = 359.0
    wave.peak_direction_raw_deg[:, :, 1] = 1.0
    values, qc, _ = _node_wave_samples(manager, manager.mesh.node_xy, _state().time_utc_ns)
    assert np.all(np.minimum(values[:, 2], 360 - values[:, 2]) < 1.01)


@pytest.mark.parametrize("bad", ["mask", "hs", "fp", "direction"])
def test_wave_missing_preserves_qc_and_never_zero_or_success(original_kernels, bad):
    """任一必要四角的缺值均拒絕；重複快取拒絕仍保留原QC，不偽裝成功或零波浪。"""
    _, waves, manager = _inputs()
    arrays = {
        "mask": waves.valid_mask_wave,
        "hs": waves.significant_wave_height,
        "fp": waves.peak_frequency,
        "direction": waves.peak_direction_raw_deg,
    }
    arrays[bad][0, 0, 0] = False if bad == "mask" else np.nan
    resolver = OCMPreparedFrameResolver()
    runner = NativeBayTraceBatchAdvector(original_kernels, resolver, require_native=True)
    task = _task(_provider(manager))
    for _ in range(2):
        result = runner([task])[0]
        assert isinstance(result, SamplingError)
        assert result.qc != SampleQC.OK
        assert not result.qc & SampleQC.NUMERICAL_FAILURE
    assert runner.report()["native_completed_particle_steps"] == 0
    assert runner.report()["fallback_particle_steps"] == 0


def test_varying_field_records_discretization_difference(original_kernels, record_property):
    """非均勻／變時Hs及fp在節點先算漂移，不等於逐stage先內插參數再算漂移。"""
    _, _, manager = _inputs(varying=True)
    provider = _provider(manager)
    state = _state(x_m=35.0, y_m=20.0)
    sample = provider(state.x_m, state.y_m, state.z_m, state.time_utc_ns)
    task = _task(provider, state=state, sample=sample, dt_seconds=-30.0)
    resolver = OCMPreparedFrameResolver()
    runner = NativeBayTraceBatchAdvector(original_kernels, resolver, require_native=True)
    result = runner([task])[0]
    reference = execute_advection_task(task)
    assert isinstance(result, ParticleState)
    delta = np.linalg.norm([result.x_m - reference.x_m, result.y_m - reference.y_m])
    record_property("native_minus_legacy_horizontal_m", float(delta))
    record_property("native_displacement_m", float(np.hypot(result.x_m - state.x_m, result.y_m - state.y_m)))
    assert 1e-6 < delta < 10.0
    assert runner.report()["native_completed_particle_steps"] == 1
    assert resolver.report()["last_frame_metadata"]["discretization"].endswith("not_legacy_stage_sampling")


@pytest.mark.parametrize("fallback", [False, True])
def test_nww_global_time_policy_is_not_invented(fallback):
    """OCM端點沒有NWW覆蓋時，只能由既有明示補位政策決定；禁止自訂最近值。"""
    start = 1_704_067_200_000_000_000
    _, _, manager = _inputs(wave_times=[start + 600_000_000_000, start + 3000_000_000_000], fallback=fallback)
    resolver = OCMPreparedFrameResolver()
    frame = resolver(_task(_provider(manager)))
    if fallback:
        assert frame is not None
        assert all(
            row["time_fallback"] for row in resolver.report()["last_frame_metadata"]["wave_endpoint_sources"]
        )
    else:
        assert frame is None
        assert resolver.last_rejection_qc & SampleQC.OUTSIDE_TIME_RANGE


def test_unrelated_wave_missing_and_cached_local_qc(original_kernels, monkeypatch):
    """遠端波浪域外不污染可算片；快取命中無效片仍回原域外QC，不變工程拒絕。"""
    _, waves, manager = _inputs(mesh=_far_mesh())
    provider = _provider(manager)
    resolver = OCMPreparedFrameResolver()
    runner = NativeBayTraceBatchAdvector(original_kernels, resolver, require_native=True)
    valid = _task(provider)
    assert isinstance(runner([valid])[0], ParticleState)

    def forbidden(*args, **kwargs):
        raise AssertionError("已建波浪場不得因遠端缺值重新取樣")

    monkeypatch.setattr(waves, "sample_at_time_index", forbidden)
    invalid = _task(provider, state=_state(x_m=1020.0, y_m=1020.0), sample=_sample(triangle_id=1))
    for _ in range(2):
        error = runner([invalid])[0]
        assert isinstance(error, SamplingError)
        assert error.qc & SampleQC.OUTSIDE_HORIZONTAL_DOMAIN
        assert not error.qc & SampleQC.NUMERICAL_FAILURE
    assert isinstance(runner([valid])[0], ParticleState)
    assert resolver.report()["frame_builds"] == 1
    assert runner.report()["fallback_particle_steps"] == 0


def test_wave_cache_capacity_fails_before_nww_and_counts_no_success(original_kernels, monkeypatch):
    """容量不足是工程例外，先於昂貴波浪準備拒絕，不終止粒子也不假算成功。"""
    _, _, manager = _inputs()

    def forbidden(*args, **kwargs):
        raise AssertionError("容量拒絕應先於波浪讀取")

    monkeypatch.setattr(manager, "_ensure_nww", forbidden)
    runner = NativeBayTraceBatchAdvector(
        original_kernels, OCMPreparedFrameResolver(max_cached_bytes=1), require_native=True
    )
    with pytest.raises(NativeBayTraceUnsupportedError, match="frame_bytes_exceeds_limit"):
        runner([_task(_provider(manager))])
    assert runner.report()["native_completed_particle_steps"] == 0
    assert runner.report()["fallback_particle_steps"] == 0


def test_wave_exact_dt_utc_groups_identity_and_no_random(original_kernels):
    """同端點不同UTC／dt各自成組，保留精確步長與來源順序，亂數狀態完全不動。"""
    _, _, manager = _inputs()
    provider = _provider(manager)
    tasks = [
        _task(
            provider,
            state=_state(particle_id=f"id{i}", time_utc_ns=_state().time_utc_ns + (1 if i == 3 else 0)),
            dt_seconds=-2.0 if i == 2 else -1.0,
        )
        for i in range(4)
    ]
    rng_before = np.random.get_state()
    resolver = OCMPreparedFrameResolver()
    runner = NativeBayTraceBatchAdvector(original_kernels, resolver, require_native=True)
    results = runner(tasks)
    for task, result in zip(tasks, results, strict=True):
        assert isinstance(result, ParticleState)
        assert result.particle_id == task.state.particle_id
        assert result.scenario_id == task.state.scenario_id
        assert result.member_id == task.state.member_id
        assert result.time_utc_ns == task.state.time_utc_ns + round(task.dt_seconds * 1e9)
    rng_after = np.random.get_state()
    for before, after in zip(rng_before, rng_after, strict=True):
        np.testing.assert_equal(before, after)
    assert runner.report()["group_size_histogram"] == {"1": 2, "2": 1}
    assert resolver.report()["cache_warmup_builds"] == 1


def test_reconstructed_ocm_source_is_preserved_with_wave():
    """啟用波浪不把OCM重建列來源改成實測；兩端NWW UTC另列保存。"""
    mesh = _mesh()
    times = np.asarray([_state().time_utc_ns], dtype=np.int64)
    base = _month(mesh, times=times)
    patch = OCMReconstructionPatchMonth(
        month_id="202401",
        time_utc_ns=times,
        hvel=base.hvel,
        vertical_velocity=base.vertical_velocity,
        zcor=base.zcor,
        elev=base.elev,
        wetdry_elem=base.wetdry_elem,
        diffusivity=base.diffusivity,
        origin_code=np.asarray([1], dtype=np.int8),
        quality_flags=np.asarray([1], dtype=np.uint16),
        metadata={"method": "synthetic-wave-binding", "gap_id": "test-only"},
    )
    _, _, manager = _inputs(mesh=mesh, patch=patch)
    resolver = OCMPreparedFrameResolver()
    assert resolver(_task(_provider(manager))) is not None
    meta = resolver.report()["last_frame_metadata"]
    assert meta["before_source"] == meta["after_source"] == "reconstructed"
    assert meta["before_row_diagnostics"]["ocm_reconstruction_method"] == "synthetic-wave-binding"
    assert all(row["ocm_endpoint_utc_ns"] == times[0] for row in meta["wave_endpoint_sources"])


def test_nww_month_halo_uses_known_ocm_endpoint_month():
    """下月UTC可來自前月已存halo；不能因下月無檔誤報缺波浪，也不造新列。"""
    _, wave, _ = _inputs()
    mesh = _mesh()
    times = np.asarray([1_706_742_000_000_000_000, 1_706_745_600_000_000_000], dtype=np.int64)
    ocean = _month(mesh, times=times)
    wave.time_utc_ns = times
    manager = ForcingWindowManager(
        flow_domain_id="known-halo",
        projection=DomainProjection(121.0, 25.0),
        mesh=mesh,
        ocm_loader=lambda month: ocean if month == "202401" else None,
        nww_loader=lambda month: wave if month == "202401" else None,
        max_resident_months=1,
    )
    resolver = OCMPreparedFrameResolver()
    assert resolver(_task(_provider(manager), state=_state(time_utc_ns=int(times[-1])))) is not None
    sources = resolver.report()["last_frame_metadata"]["wave_endpoint_sources"]
    assert all(row["before_month_id"] == row["after_month_id"] == "202401" for row in sources)
    assert all(row["before_utc_ns"] == times[-1] and not row["time_fallback"] for row in sources)


def test_vertical_reference_uses_endpoint_eta_and_stored_zcor():
    """各端點漂移用zcor-eta及dp+eta，非把高程當相對深度；底下未使用層不補0。"""
    ocean, _, manager = _inputs()
    ocean.elev[1] = 1.0
    ocean.zcor[1, :, :] = [-20.0, -9.5, 1.0]
    ocean.zcor[:, :, 0] = np.nan
    resolver = OCMPreparedFrameResolver()
    frame = resolver(_task(_provider(manager)))
    assert frame is not None
    expected = finite_depth_stokes(
        significant_wave_height_m=2.0,
        peak_frequency_hz=0.1,
        direction_raw_deg=270.0,
        particle_z_m=-9.5,
        surface_z_m=1.0,
        bed_z_m=-20.0,
    )
    np.testing.assert_allclose(frame.fields.u2[1], 0.1 + expected.u_mps, rtol=2e-12)
    np.testing.assert_allclose(frame.fields.u2[0], frame.fields.u2[1])
    assert frame.grid.kbp.tolist() == [1, 1, 1, 1]
