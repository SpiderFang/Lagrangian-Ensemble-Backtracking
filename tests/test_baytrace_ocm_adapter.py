"""Q3 OCM frame resolver 與原版 BayTrace CPU 的整合測試。

測試故意使用真正的 PI BayTrace v4.7.2 CPU dispatcher、LBT
``ForcingWindowManager`` 與 ``OCMNativeMonth`` synthetic 資料；不以 mock kernel
替代原始 ``batch_interpolate_velocity``／``batch_move_rk4``。資料只包含小型兩三角形
網格，讓測試可以驗證跨面、時間 bracket、沉降與 fail-closed gate，而不啟動全量 OCM
或 production benchmark。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from lagrangian_backtracking.advection import AdvectionTask
from lagrangian_backtracking.baytrace_batch_advection import NativeBayTraceBatchAdvector
from lagrangian_backtracking.baytrace_native_cpu import load_original_cpu
from lagrangian_backtracking.baytrace_ocm_adapter import OCMPreparedFrameResolver
from lagrangian_backtracking.forcing import OCMNativeMonth, OCMReconstructionPatchMonth
from lagrangian_backtracking.forcing_window import (
    ForcingWindowManager,
    ManagedForcingProvider,
)
from lagrangian_backtracking.geometry import DomainProjection
from lagrangian_backtracking.integrators import PHYSICS_KERNEL_BACKEND_NUMPY_V1
from lagrangian_backtracking.mesh import NativeMesh
from lagrangian_backtracking.models import ParticleState, ParticleStatus, SampleQC, VelocitySample
from lagrangian_backtracking.production import HintTrackingVelocityProvider

_BAYTRACE_ROOT = Path(
    "/Users/mustlab/Workspace/Lagrangian-Ensemble-Backtracking/"
    "BayTrace/Baytech_BayTrace_v4.7.2"
)
_HOUR_NS = 3_600_000_000_000


@pytest.fixture(scope="module")
def original_kernels(tmp_path_factory: pytest.TempPathFactory):
    """以明確的 PI artifact 載入實際原版 CPU dispatcher。"""

    if not _BAYTRACE_ROOT.is_dir():
        pytest.skip(f"未找到指定 BayTrace root: {_BAYTRACE_ROOT}")
    return load_original_cpu(_BAYTRACE_ROOT, tmp_path_factory.mktemp("q3-baytrace-cache"))


def _mesh() -> NativeMesh:
    """建立兩個相鄰、逆時針方向的三角形與固定 20 m 水深。"""

    return NativeMesh(
        node_lon=np.asarray([121.0, 121.001, 121.001, 121.0]),
        node_lat=np.asarray([25.0, 25.0, 25.001, 25.001]),
        node_xy=np.asarray([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]]),
        source_depth_m=np.full(4, 20.0),
        # 這個欄位刻意不參與 adapter 的 kbp 推導；Q3 要求由 zcor 首個有效層推導。
        source_node_bottom_index=np.full(4, 99, dtype=np.int64),
        face_nodes_local=np.asarray([[0, 1, 2, -1], [0, 2, 3, -1]]),
        face_node_count=np.asarray([3, 3]),
        source_face_global_index=np.asarray([10, 11]),
        bin_size_m=200.0,
    )


def _far_mesh() -> NativeMesh:
    """建立兩個互不相鄰的三角形，供局部乾面 negative cache 測試使用。"""

    return NativeMesh(
        node_lon=np.asarray([121.0, 121.001, 121.0, 121.01, 121.011, 121.01]),
        node_lat=np.asarray([25.0, 25.0, 25.001, 25.01, 25.01, 25.011]),
        node_xy=np.asarray(
            [
                [0.0, 0.0],
                [100.0, 0.0],
                [0.0, 100.0],
                [1_000.0, 1_000.0],
                [1_100.0, 1_000.0],
                [1_000.0, 1_100.0],
            ]
        ),
        source_depth_m=np.full(6, 20.0),
        source_node_bottom_index=np.full(6, 99, dtype=np.int64),
        face_nodes_local=np.asarray([[0, 1, 2, -1], [3, 4, 5, -1]]),
        face_node_count=np.asarray([3, 3]),
        source_face_global_index=np.asarray([10, 11]),
        bin_size_m=200.0,
    )


def _month(
    mesh: NativeMesh,
    *,
    month_id: str = "202401",
    times: np.ndarray | None = None,
    settling_current_mps: float = 0.03,
    wetdry: np.ndarray | None = None,
    zcor: np.ndarray | None = None,
    elev: np.ndarray | None = None,
    patch: OCMReconstructionPatchMonth | None = None,
    continuous_time_fallback: bool = False,
) -> OCMNativeMonth:
    """建立 node×layer×component 的 OCM row，包含非均勻水平／垂向速度。"""

    if times is None:
        times = np.asarray([1_704_067_200_000_000_000, 1_704_070_800_000_000_000], dtype=np.int64)
    times = np.asarray(times, dtype=np.int64)
    row_count = int(times.size)
    nvrt = 3
    nnode = int(mesh.node_xy.shape[0])
    nface = int(mesh.source_face_global_index.size)
    layer = np.arange(nvrt, dtype=np.float64)[None, :, None]
    node = np.arange(nnode, dtype=np.float64)[None, None, :]
    hvel = np.empty((row_count, nnode, nvrt, 2), dtype=np.float64)
    for row in range(row_count):
        # 三個速度分量都保持有限但不均勻，避免 synthetic case 只測常數平移。
        hvel[row, :, :, 0] = (0.04 + 0.002 * node + 0.003 * layer + row * 0.001)[0].T
        hvel[row, :, :, 1] = (0.02 - 0.001 * node + 0.002 * layer - row * 0.0005)[0].T
    vertical = np.empty((row_count, nnode, nvrt), dtype=np.float64)
    vertical[...] = settling_current_mps
    vertical += 0.002 * layer.reshape(1, 1, nvrt)
    if zcor is None:
        base_z = np.asarray([-20.0, -10.0, 0.0], dtype=np.float64)
        zcor = np.broadcast_to(base_z[None, None, :], (row_count, nnode, nvrt)).copy()
    if elev is None:
        elev = np.zeros((row_count, nnode), dtype=np.float64)
    if wetdry is None:
        wetdry = np.zeros((row_count, nface), dtype=np.float64)
    return OCMNativeMonth(
        month_id=month_id,
        mesh=mesh,
        time_utc_ns=times,
        hvel=hvel,
        vertical_velocity=vertical,
        zcor=np.asarray(zcor, dtype=np.float64),
        elev=np.asarray(elev, dtype=np.float64),
        wetdry_elem=np.asarray(wetdry, dtype=np.float64),
        diffusivity=np.zeros((row_count, nnode, nvrt), dtype=np.float64),
        reconstruction_patch=patch,
        continuous_time_fallback=continuous_time_fallback,
    )


def _manager(
    mesh: NativeMesh,
    months: dict[str, OCMNativeMonth],
    *,
    continuous_time_fallback: bool = False,
    available_month_ids: tuple[str, ...] | None = None,
    max_resident_months: int = 2,
) -> ForcingWindowManager:
    """建立不讀磁碟的 manager，仍使用正式 global bracket 私有流程。"""

    return ForcingWindowManager(
        flow_domain_id="q3-synthetic",
        projection=DomainProjection(121.0, 25.0),
        mesh=mesh,
        ocm_loader=lambda month_id: months.get(month_id),
        max_resident_months=max_resident_months,
        continuous_time_fallback=continuous_time_fallback,
        available_ocm_month_ids=available_month_ids,
    )


def _state(
    *,
    particle_id: str = "p0",
    time_utc_ns: int = 1_704_069_000_000_000_000,
    x_m: float = 20.0,
    y_m: float = 20.0,
    z_m: float = -10.0,
    status: ParticleStatus = ParticleStatus.ACTIVE,
) -> ParticleState:
    """建立可辨識 identity 的公尺制 active particle state。"""

    return ParticleState(
        particle_id=particle_id,
        scenario_id="q3-scenario",
        member_id=0,
        study_site_id="site-a",
        analysis_region_id="region-a",
        receptor_id="receptor-a",
        x_m=x_m,
        y_m=y_m,
        z_m=z_m,
        time_utc_ns=time_utc_ns,
        age_seconds=0.0,
        status=status,
    )


def _sample(*, triangle_id: int | None = 0, qc: SampleQC = SampleQC.OK) -> VelocitySample:
    """建立僅作步首 reference／triangle hint 的有限 OCM sample。"""

    return VelocitySample(
        u_mps=0.05,
        v_mps=0.02,
        w_mps=0.01,
        eta_m=0.0,
        bed_z_m=-20.0,
        horizontal_scale_m=100.0,
        vertical_scale_m=10.0,
        qc=qc,
        triangle_id=triangle_id,
    )


def _task(
    provider: object,
    *,
    state: ParticleState | None = None,
    sample: VelocitySample | None = None,
    dt_seconds: float = 1.0,
) -> AdvectionTask:
    """建立 Q1 frozen advection task，模擬 production 的 wrapper chain。"""

    return AdvectionTask(
        state=state or _state(),
        dt_seconds=dt_seconds,
        velocity=provider,
        step_start_sample=sample or _sample(),
        physics_kernel_backend=PHYSICS_KERNEL_BACKEND_NUMPY_V1,
    )


def test_real_original_kernel_ocm_frame_and_nonzero_settling_run(original_kernels) -> None:
    """實際原版 kernel 應使用 OCM frame 並把非零沉降放進四個 RK4 stage。"""

    mesh = _mesh()
    month = _month(mesh, settling_current_mps=0.03)
    manager = _manager(mesh, {month.month_id: month})
    managed = ManagedForcingProvider(manager, settling_velocity_mps=-0.02, include_stokes=False)
    provider = HintTrackingVelocityProvider(managed, triangle_hint=0)
    task = _task(provider)

    resolver = OCMPreparedFrameResolver()
    frame = resolver(task)
    assert frame is not None
    assert np.allclose(frame.fields.w1, 0.01 + 0.002 * np.arange(3)[:, None])
    assert frame.grid.kbp.tolist() == [0, 0, 0, 0]
    assert frame.grid.kbe.tolist() == [0, 0]
    assert np.allclose(frame.grid.sigma_lcl[:, 0], [-1.0, -0.5, 0.0])
    assert resolver.report()["last_frame_metadata"]["data_source"] == "observed"

    batcher = NativeBayTraceBatchAdvector(original_kernels, resolver)
    result = batcher([task])[0]
    assert isinstance(result, ParticleState)
    assert result.particle_id == task.state.particle_id
    assert result.time_utc_ns == task.state.time_utc_ns + 1_000_000_000
    assert result.age_seconds == 1.0
    assert result.z_m > task.state.z_m
    assert batcher.report()["native_particle_steps"] == 1
    assert batcher.report()["native_batch_calls"] == 1
    assert resolver.report()["frame_builds"] == 1


def test_real_resolver_batch_groups_exact_utc_trat_and_signed_dt(original_kernels) -> None:
    """同 frame／dt 才合批；不同 UTC 或有號 dt 要分組且保留 source order。"""

    mesh = _mesh()
    month = _month(mesh)
    manager = _manager(mesh, {month.month_id: month})
    managed = ManagedForcingProvider(manager, settling_velocity_mps=-0.01, include_stokes=False)
    resolver = OCMPreparedFrameResolver(max_cached_frames=4)
    batcher = NativeBayTraceBatchAdvector(original_kernels, resolver)
    first_time = int(month.time_utc_ns[0]) + 1_800_000_000_000
    second_time = int(month.time_utc_ns[0]) + 900_000_000_000
    tasks = [
        _task(managed, state=_state(particle_id="p0", time_utc_ns=first_time), dt_seconds=1.0),
        _task(managed, state=_state(particle_id="p1", time_utc_ns=first_time, x_m=25.0), dt_seconds=1.0),
        _task(
            managed,
            state=_state(particle_id="p2", time_utc_ns=first_time, x_m=30.0),
            dt_seconds=-1.0,
        ),
        _task(
            managed,
            state=_state(particle_id="p3", time_utc_ns=second_time, x_m=20.0, y_m=80.0),
            sample=_sample(triangle_id=1),
            dt_seconds=1.0,
        ),
    ]
    results = batcher(tasks)
    assert all(isinstance(result, ParticleState) for result in results)
    assert [result.particle_id for result in results] == [task.state.particle_id for task in tasks]
    assert results[2].time_utc_ns == first_time - 1_000_000_000
    assert results[2].age_seconds == 1.0
    assert batcher.report()["native_particle_steps"] == 4
    assert batcher.report()["native_batch_calls"] == 3
    assert batcher.report()["fallback_particle_steps"] == 0
    assert resolver.report()["frame_builds"] == 1
    assert resolver.report()["cache_hits"] >= 3


def test_same_endpoints_different_utc_reuse_arrays_and_mesh_check_once(monkeypatch) -> None:
    """相同 OCM 端點只建一次 array，查詢 UTC 仍更新 trat 與 metadata。"""

    mesh = _mesh()
    month = _month(mesh)
    manager = _manager(mesh, {month.month_id: month})
    provider = ManagedForcingProvider(manager, settling_velocity_mps=0.0, include_stokes=False)
    resolver = OCMPreparedFrameResolver()
    mesh_match_calls = 0
    original_mesh_match = manager._meshes_match

    def spy_mesh_match(first, second):
        nonlocal mesh_match_calls
        mesh_match_calls += 1
        return original_mesh_match(first, second)

    monkeypatch.setattr(manager, "_meshes_match", spy_mesh_match)
    first_time = int(month.time_utc_ns[0]) + 900_000_000_000
    second_time = int(month.time_utc_ns[0]) + 1_800_000_000_000
    first = resolver(_task(provider, state=_state(time_utc_ns=first_time)))
    # manager 可能在載入月份時做一次自己的 mesh provenance 檢查；只觀察
    # frame 建立完成後的多個 cache hit，確認 adapter 不會每粒子重新掃 mesh。
    mesh_match_calls = 0
    second = resolver(_task(provider, state=_state(particle_id="p1", time_utc_ns=second_time)))
    for index in range(2, 6):
        resolver(
            _task(
                provider,
                state=_state(particle_id=f"p{index}", time_utc_ns=second_time),
            )
        )
    assert first is not None and second is not None
    assert first.grid is second.grid
    assert first.fields is second.fields
    assert first.trat == pytest.approx(0.25)
    assert second.trat == pytest.approx(0.5)
    report = resolver.report()
    assert report["frame_builds"] == 1
    assert report["cache_hits"] == 5
    assert report["last_frame_metadata"]["target_utc_ns"] == second_time
    assert report["last_frame_metadata"]["trat"] == pytest.approx(second.trat)
    assert report["last_frame_metadata"]["frame_cache_reused"] is True
    assert report["last_frame_metadata"]["frame_build_target_utc_ns"] == first_time
    assert report["last_frame_metadata"]["frame_build_trat"] == pytest.approx(first.trat)
    # 同一 manager.mesh 直接以 identity 通過，N 個 cache hit 不應掃全 mesh。
    assert mesh_match_calls == 0


def test_resolver_cache_reuse_and_exact_treatment_key() -> None:
    """相同 endpoint／settling 應 reuse；不同沉降或 trat 不得共用錯場。"""

    mesh = _mesh()
    month = _month(mesh)
    manager = _manager(mesh, {month.month_id: month})
    first_provider = ManagedForcingProvider(manager, settling_velocity_mps=-0.01, include_stokes=False)
    second_provider = ManagedForcingProvider(manager, settling_velocity_mps=-0.02, include_stokes=False)
    resolver = OCMPreparedFrameResolver(max_cached_frames=2)
    first = resolver(_task(first_provider))
    second = resolver(_task(first_provider, state=_state(particle_id="p1")))
    third = resolver(_task(second_provider, state=_state(particle_id="p2")))
    assert first is not None and second is not None and third is not None
    assert first.grid is second.grid
    assert first.fields is second.fields
    assert first.fields.w1[0, 0] != third.fields.w1[0, 0]
    report = resolver.report()
    assert report["cache_hits"] >= 1
    assert report["frame_builds"] == 2
    assert report["cached_frames"] <= 2


def test_cross_month_bracket_and_missing_month_continuous_fallback_metadata() -> None:
    """跨月 endpoint 與 manager continuous fallback 應保留來源與策略欄位。"""

    mesh = _mesh()
    january_times = np.asarray([1_706_700_000_000_000_000], dtype=np.int64)
    february_times = np.asarray([1_706_710_800_000_000_000], dtype=np.int64)
    january = _month(mesh, month_id="202401", times=january_times)
    february = _month(mesh, month_id="202402", times=february_times)
    manager = _manager(
        mesh,
        {"202401": january, "202402": february},
        continuous_time_fallback=True,
        available_month_ids=("202401", "202402", "202403"),
    )
    provider = ManagedForcingProvider(manager, settling_velocity_mps=0.0, include_stokes=False)
    target = int(january_times[0] + 1_800_000_000_000)
    resolver = OCMPreparedFrameResolver()
    frame = resolver(_task(provider, state=_state(time_utc_ns=target)))
    assert frame is not None
    metadata = resolver.report()["last_frame_metadata"]
    assert metadata["before_month_id"] == "202401"
    assert metadata["after_month_id"] == "202402"
    assert metadata["time_fallback"] == 1
    assert metadata["time_fallback_policy"] in {"linear_interpolation", "nearest_valid_temporal_endpoint"}


def test_reconstruction_patch_provenance_is_not_copied_as_observed() -> None:
    """merged patch row 應讀單列並明示 reconstructed provenance。"""

    mesh = _mesh()
    observed_time = np.asarray([1_704_067_200_000_000_000], dtype=np.int64)
    patch_time = np.asarray([1_704_069_000_000_000_000], dtype=np.int64)
    observed = _month(mesh, month_id="202401", times=observed_time)
    patch_base = _month(mesh, month_id="202401", times=patch_time)
    patch = OCMReconstructionPatchMonth(
        month_id="202401",
        time_utc_ns=patch_time,
        hvel=patch_base.hvel.copy(),
        vertical_velocity=patch_base.vertical_velocity.copy(),
        zcor=patch_base.zcor.copy(),
        elev=patch_base.elev.copy(),
        wetdry_elem=patch_base.wetdry_elem.copy(),
        diffusivity=patch_base.diffusivity.copy(),
        origin_code=np.asarray([1], dtype=np.int8),
        quality_flags=np.asarray([1], dtype=np.uint16),
        metadata={"method": "q3-test", "gap_id": "gap-1"},
    )
    observed = _month(mesh, month_id="202401", times=observed_time, patch=patch)
    manager = _manager(mesh, {"202401": observed})
    provider = ManagedForcingProvider(manager, settling_velocity_mps=0.0, include_stokes=False)
    resolver = OCMPreparedFrameResolver()
    frame = resolver(_task(provider, state=_state(time_utc_ns=int(patch_time[0]))))
    assert frame is not None
    metadata = resolver.report()["last_frame_metadata"]
    assert metadata["before_source"] == "reconstructed"
    assert metadata["after_source"] == "reconstructed"
    assert metadata["before_row_diagnostics"]["ocm_reconstruction_method"] == "q3-test"


def test_invalid_vertical_or_depth_is_bounded_negative_cache_and_fallback() -> None:
    """有效層 NaN／非法水深不可建 frame，重試只命中 finite negative cache。"""

    mesh = _mesh()
    bad_z = np.broadcast_to(np.asarray([-20.0, np.nan, 0.0])[None, None, :], (2, 4, 3)).copy()
    month = _month(mesh, zcor=bad_z)
    manager = _manager(mesh, {month.month_id: month})
    provider = ManagedForcingProvider(manager, settling_velocity_mps=0.0, include_stokes=False)
    resolver = OCMPreparedFrameResolver(max_cached_frames=2)
    assert resolver(_task(provider)) is None
    assert resolver(_task(provider, state=_state(particle_id="p1"))) is None
    report = resolver.report()
    assert report["fallback_reasons"]["no_valid_wet_support"] == 2
    assert report["negative_cache_hits"] == 1
    assert report["negative_cache_entries"] == 1


def test_dry_face_preserves_qc_with_current_and_wave() -> None:
    """純流／流波均先拒絕實際乾面；不因啟用波浪而掩蓋原乾面原因。"""

    mesh = _mesh()
    dry = np.zeros((2, 2), dtype=np.float64)
    dry[:, 0] = 1.0
    month = _month(mesh, wetdry=dry)
    manager = _manager(mesh, {month.month_id: month})
    resolver = OCMPreparedFrameResolver()
    current = ManagedForcingProvider(manager, settling_velocity_mps=0.0, include_stokes=False)
    waves = ManagedForcingProvider(manager, settling_velocity_mps=0.0, include_stokes=True)
    assert resolver(_task(current)) is None
    assert resolver(_task(waves, state=_state(particle_id="p1"))) is None
    reasons = resolver.report()["fallback_reasons"]
    assert reasons["dry_face"] == 2
    assert resolver.last_rejection_qc == SampleQC.DRY_FACE


def test_local_dry_face_does_not_poison_disconnected_wet_element(original_kernels) -> None:
    """局部乾面只影響該 element，遠端全濕面仍可進入原版 native。"""

    mesh = _far_mesh()
    dry = np.zeros((2, 2), dtype=np.float64)
    dry[:, 0] = 1.0
    month = _month(mesh, wetdry=dry)
    manager = _manager(mesh, {month.month_id: month})
    provider = ManagedForcingProvider(manager, settling_velocity_mps=0.0, include_stokes=False)
    resolver = OCMPreparedFrameResolver()
    dry_task = _task(provider, state=_state(x_m=20.0, y_m=20.0), sample=_sample(triangle_id=0))
    wet_task = _task(
        provider,
        state=_state(particle_id="wet", x_m=1_020.0, y_m=1_020.0),
        sample=_sample(triangle_id=1),
    )
    assert resolver(dry_task) is None
    wet_frame = resolver(wet_task)
    assert wet_frame is not None
    result = NativeBayTraceBatchAdvector(original_kernels, resolver)([wet_task])[0]
    assert isinstance(result, ParticleState)
    report = resolver.report()
    assert report["fallback_reasons"]["dry_face"] == 1
    assert report["frame_builds"] == 1
    assert report["negative_cache_hits"] == 0


def test_missing_triangle_or_unsupported_wrapper_does_not_build_frame() -> None:
    """沒有 LBT 步首 triangle hint 或未註冊 provider 時不可猜測 element。"""

    mesh = _mesh()
    month = _month(mesh)
    manager = _manager(mesh, {month.month_id: month})
    managed = ManagedForcingProvider(manager, settling_velocity_mps=0.0, include_stokes=False)
    resolver = OCMPreparedFrameResolver()
    assert resolver(replace(_task(managed), step_start_sample=None)) is None

    class UnsupportedWrapper:
        """模擬非 ManagedForcingProvider 的波浪／材質 wrapper。"""

        def __init__(self) -> None:
            self.provider = lambda *_args: None

    assert resolver(_task(UnsupportedWrapper(), state=_state(particle_id="p1"))) is None

    class SpeedChangingWrapper:
        """模擬雖暴露 provider 但會改寫速度的未註冊 wrapper。"""

        def __init__(self, wrapped: ManagedForcingProvider) -> None:
            self.provider = wrapped

    assert resolver(_task(SpeedChangingWrapper(managed), state=_state(particle_id="p2"))) is None
    reasons = resolver.report()["fallback_reasons"]
    assert reasons["unsupported_provider_wrapper"] == 2


def test_different_mesh_is_explicit_and_does_not_abort_other_task() -> None:
    """manager loader 的跨 mesh endpoint 應 fallback；另一個有效 resolver call 仍可成功。"""

    mesh = _mesh()
    other_mesh = _mesh()
    other_mesh.node_xy[0, 0] = 1.0
    valid_month = _month(mesh, month_id="202401")
    invalid_month = _month(
        other_mesh,
        month_id="202402",
        times=np.asarray([1_706_745_600_000_000_000, 1_706_749_200_000_000_000], dtype=np.int64),
    )
    manager = _manager(
        mesh,
        {"202401": valid_month, "202402": invalid_month},
        continuous_time_fallback=False,
    )
    provider = ManagedForcingProvider(manager, settling_velocity_mps=0.0, include_stokes=False)
    resolver = OCMPreparedFrameResolver()
    valid = resolver(_task(provider))
    assert valid is not None
    invalid_time = int(invalid_month.time_utc_ns[0])
    invalid = resolver(_task(provider, state=_state(particle_id="p1", time_utc_ns=invalid_time)))
    assert invalid is None
    assert resolver.report()["fallback_reasons"]["different_mesh"] == 1


@pytest.mark.parametrize("field", ["depth", "eta", "zcor", "hvel", "vertical"])
def test_unrelated_invalid_nodes_compact_once_and_run_strict_native(original_kernels, monkeypatch, field):
    """遠端乾／缺節點不否決有效三角形；原hint1映為native0，步後身份仍保留。"""
    import lagrangian_backtracking.baytrace_batch_advection as batch_module

    mesh = _far_mesh()
    month = _month(mesh)
    manager = _manager(mesh, {month.month_id: month})
    # 先讓既有manager完成mesh binding，再模擬已載入row內的無關缺值；
    # 不放寬manager本身的輸入產品驗證契約。
    manager.preload((month.month_id,))
    if field == "depth":
        mesh.source_depth_m[:3] = np.nan
    elif field == "eta":
        month.elev[:, :3] = np.nan
    elif field == "zcor":
        month.zcor[:, :3] = np.nan
    elif field == "hvel":
        month.hvel[:, :3] = np.nan
    else:
        month.vertical_velocity[:, :3] = np.nan
    provider = manager.provider(-0.02, False)
    resolver = OCMPreparedFrameResolver()
    tasks = [_task(provider, state=_state(particle_id=f"p{i}", x_m=1020, y_m=1020),
                   sample=_sample(triangle_id=1)) for i in range(4)]

    def forbidden(*args, **kwargs):
        """纯流不得讀取波浪或以legacy掩蓋轉接失敗。"""
        raise AssertionError("unexpected NWW/legacy")

    monkeypatch.setattr(manager, "_ensure_nww", forbidden)
    monkeypatch.setattr(batch_module, "_execute_legacy_advection_task", forbidden)
    frame = resolver(tasks[0])
    assert frame is not None and frame.element_id == 0 and frame.grid.grid_x.size == 3
    batcher = NativeBayTraceBatchAdvector(original_kernels, resolver, require_native=True)
    results = batcher(tasks)
    assert all(isinstance(result, ParticleState) for result in results)
    assert [result.particle_id for result in results] == [f"p{i}" for i in range(4)]
    assert all(task.step_start_sample.triangle_id == 1 for task in tasks)
    assert resolver.report()["frame_builds"] == 1
    assert batcher.report()["group_size_histogram"] == {"4": 1}
    assert batcher.native_particle_steps == 4 and batcher.fallback_particle_steps == 0


def test_strict_capacity_and_surface_wrapper_are_explicit_rejection(original_kernels):
    """過小記憶體上限及帶鏡射物理的wrapper均拒絕，不假native／退legacy。"""
    from lagrangian_backtracking.baytrace_batch_advection import NativeBayTraceUnsupportedError
    from lagrangian_backtracking.integrators import SurfaceStageVelocityProvider

    mesh = _mesh()
    month = _month(mesh)
    manager = _manager(mesh, {month.month_id: month})
    provider = manager.provider(0, False)
    resolver = OCMPreparedFrameResolver(max_cached_bytes=1)
    batcher = NativeBayTraceBatchAdvector(original_kernels, resolver, require_native=True)
    for _ in range(3):
        with pytest.raises(NativeBayTraceUnsupportedError, match="frame_bytes_exceeds_limit"):
            batcher([_task(provider)])
    assert resolver.report()["frame_builds"] == resolver.report()["cached_bytes"] == 0
    assert resolver.report()["negative_cache_hits"] == 2
    wrapper = SurfaceStageVelocityProvider(provider, _state(), _sample(), "non_rising")
    with pytest.raises(NativeBayTraceUnsupportedError, match="surface_stage_wrapper_unsupported"):
        batcher([_task(wrapper)])
    assert batcher.fallback_particle_steps == batcher.native_particle_steps == 0


@pytest.mark.parametrize("qc", [SampleQC.DRY_FACE, SampleQC.TIME_GAP, SampleQC.WAVE_UNSUPPORTED])
def test_strict_keeps_known_sample_qc(original_kernels, qc):
    """真正已有取樣證據的QC原樣保留，不把dry/time/wave變成數值失敗。"""
    from lagrangian_backtracking.integrators import SamplingError
    mesh = _mesh()
    month = _month(mesh)
    manager = _manager(mesh, {month.month_id: month})
    resolver = OCMPreparedFrameResolver()
    batcher = NativeBayTraceBatchAdvector(original_kernels, resolver, require_native=True)
    result = batcher([_task(manager.provider(0, False), sample=_sample(qc=qc))])[0]
    assert isinstance(result, SamplingError) and result.qc == qc
    assert batcher.native_particle_steps == batcher.fallback_particle_steps == 0
