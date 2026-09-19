"""Q2 原版 BayTrace batch advection 的實源與資料契約測試。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from lagrangian_backtracking.advection import AdvectionTask, execute_advection_task
from lagrangian_backtracking.baytrace_batch_advection import (
    NativeBayTraceBatchAdvector,
    PreparedBayTraceFrame,
)
from lagrangian_backtracking.baytrace_native_cpu import (
    BayTraceFieldPair,
    BayTraceGridArrays,
    BayTraceParticleArrays,
    load_original_cpu,
    step_native,
)
from lagrangian_backtracking.integrators import (
    PHYSICS_KERNEL_BACKEND_NUMPY_V1,
    SamplingError,
)
from lagrangian_backtracking.models import (
    ParticleState,
    ParticleStatus,
    SampleQC,
    VelocitySample,
)

_BAYTRACE_ROOT = Path(
    "/Users/mustlab/Workspace/Lagrangian-Ensemble-Backtracking/"
    "BayTrace/Baytech_BayTrace_v4.7.2"
)


@pytest.fixture(scope="module")
def original_kernels(tmp_path_factory: pytest.TempPathFactory):
    """只以明確的 PI artifact 執行原版 dispatcher 測試。"""

    if not _BAYTRACE_ROOT.is_dir():
        pytest.skip(f"未找到指定 BayTrace root: {_BAYTRACE_ROOT}")
    return load_original_cpu(_BAYTRACE_ROOT, tmp_path_factory.mktemp("baytrace-cache"))


def _grid() -> BayTraceGridArrays:
    """建立兩個正向相鄰三角形，供測試原版一環 neighbor hint。"""

    return BayTraceGridArrays(
        i34=np.array([3, 3], dtype=np.int32),
        elnode=np.array([[0, 1, 2, -1], [0, 2, 3, -1]], dtype=np.int32),
        grid_x=np.array([0.0, 1.0, 1.0, 0.0]),
        grid_y=np.array([0.0, 0.0, 1.0, 1.0]),
        # 每個三角形的第一個 local edge 設為另一面；其餘邊界鄰居保持 -1。
        ic3=np.array([[1, 0], [-1, -1], [-1, -1], [-1, -1]], dtype=np.int32),
        nvrt=3,
        kbp=np.zeros(4, dtype=np.int32),
        kbe=np.zeros(2, dtype=np.int32),
        sigma_lcl=np.array(
            [[-1.0, -1.0, -1.0, -1.0], [-0.5, -0.5, -0.5, -0.5], [0.0, 0.0, 0.0, 0.0]]
        ),
        dp=np.full(4, 10.0),
    )


def _fields(*, sinking_mps: float, nonuniform: bool = True) -> BayTraceFieldPair:
    """建立靜態、垂向剪切且可含固定沉降的兩端速度場。"""

    if nonuniform:
        node = np.arange(4, dtype=np.float64)[None, :]
        layer = np.arange(3, dtype=np.float64)[:, None]
        u1 = 0.03 + 0.005 * node + 0.004 * layer
        v1 = 0.02 - 0.003 * node + 0.002 * layer
        w1 = sinking_mps + 0.005 * node - 0.01 * layer
    else:
        u1 = np.full((3, 4), -0.4)
        v1 = np.full((3, 4), 0.4)
        w1 = np.full((3, 4), sinking_mps)
    return BayTraceFieldPair(
        u1=u1,
        v1=v1,
        w1=w1,
        eta1=np.zeros(4),
        u2=u1.copy(),
        v2=v1.copy(),
        w2=w1.copy(),
    )


def _state(
    particle_id: str,
    *,
    x_m: float = 0.25,
    y_m: float = 0.25,
    z_m: float = -5.0,
    status: ParticleStatus = ParticleStatus.ACTIVE,
) -> ParticleState:
    """建立具有可辨識 identity 的公尺制測試粒子。"""

    return ParticleState(
        particle_id=particle_id,
        scenario_id=f"scenario-{particle_id}",
        member_id=int(particle_id.removeprefix("p")) if particle_id.startswith("p") else 0,
        study_site_id="site-a",
        analysis_region_id="region-a",
        receptor_id="receptor-a",
        x_m=x_m,
        y_m=y_m,
        z_m=z_m,
        time_utc_ns=1_700_000_000_000_000_000,
        age_seconds=0.0,
        status=status,
    )


def _velocity(
    _x_m: float,
    _y_m: float,
    _z_m: float,
    _time_utc_ns: int,
) -> VelocitySample:
    """提供 fallback legacy RK4 所需的有限 synthetic sample。"""

    return VelocitySample(
        u_mps=0.2,
        v_mps=-0.1,
        w_mps=-0.03,
        eta_m=0.0,
        bed_z_m=-10.0,
        horizontal_scale_m=1.0,
        vertical_scale_m=1.0,
    )


def _task(state: ParticleState, dt_seconds: float) -> AdvectionTask:
    """建立與另一 worker frozen task 完全相同欄位的 advection 工作。"""

    return AdvectionTask(
        state=state,
        dt_seconds=dt_seconds,
        velocity=_velocity,
        step_start_sample=None,
        physics_kernel_backend=PHYSICS_KERNEL_BACKEND_NUMPY_V1,
    )


def _frame(
    grid: BayTraceGridArrays,
    fields: BayTraceFieldPair,
    *,
    trat: float = 0.5,
    element_id: int = 0,
) -> PreparedBayTraceFrame:
    """建立一個固定時間比例與步首元素 hint 的 frame。"""

    return PreparedBayTraceFrame(grid, fields, trat=trat, element_id=element_id)


def _direct_states(
    kernels,
    grid: BayTraceGridArrays,
    fields: BayTraceFieldPair,
    states: list[ParticleState],
    element_ids: list[int],
    *,
    dt_seconds: float,
    trat: float,
) -> list[ParticleState]:
    """直接以同一個原版 step_native group 建立比較基準。"""

    particles = BayTraceParticleArrays.from_positions(
        np.array([state.x_m for state in states]),
        np.array([state.y_m for state in states]),
        np.array([state.z_m for state in states]),
        np.array(element_ids, dtype=np.int64),
    )
    step_native(
        kernels,
        grid,
        fields,
        particles,
        dt_seconds=abs(dt_seconds),
        trat=trat,
        ibf=1 if dt_seconds > 0 else -1,
    )
    return [
        replace(
            state,
            x_m=float(particles.x[index]),
            y_m=float(particles.y[index]),
            z_m=float(particles.z[index]),
            time_utc_ns=state.time_utc_ns + int(round(dt_seconds * 1_000_000_000)),
            age_seconds=state.age_seconds + abs(dt_seconds),
        )
        for index, state in enumerate(states)
    ]


def test_native_group_matches_direct_kernel_across_adjacent_triangles_and_preserves_hash(
    original_kernels,
) -> None:
    """相鄰三角形跨面、source scatter 與原始 source hash 都必須可追溯。"""

    grid = _grid()
    fields = _fields(sinking_mps=0.0, nonuniform=False)
    frame_by_id = {
        "p0": _frame(grid, fields, element_id=0),
        "p1": _frame(grid, fields, element_id=1),
    }
    states = [
        _state("p0", x_m=0.75, y_m=0.20),
        _state("p1", x_m=0.20, y_m=0.75),
    ]
    tasks = [_task(states[0], 1.0), _task(states[1], 1.0)]

    def resolver(task: AdvectionTask) -> PreparedBayTraceFrame:
        return frame_by_id[task.state.particle_id]

    before = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in original_kernels.provenance.dependency_paths.items()
    }

    batcher = NativeBayTraceBatchAdvector(original_kernels, resolver)
    actual = batcher(tasks)
    expected = _direct_states(
        original_kernels,
        grid,
        fields,
        states,
        [0, 1],
        dt_seconds=1.0,
        trat=0.5,
    )

    assert all(isinstance(result, ParticleState) for result in actual)
    for result, reference in zip(actual, expected, strict=True):
        assert isinstance(result, ParticleState)
        np.testing.assert_allclose(
            [result.x_m, result.y_m, result.z_m],
            [reference.x_m, reference.y_m, reference.z_m],
            rtol=0.0,
            atol=1.0e-12,
        )
        assert result.particle_id == reference.particle_id
        assert result.time_utc_ns == reference.time_utc_ns
        assert result.age_seconds == reference.age_seconds
    assert actual[0].y_m > actual[0].x_m  # p0 已跨入相鄰的第二個三角形
    assert actual[1].particle_id == "p1"  # source order 未因 native group 壓縮改變
    assert batcher.native_batch_calls == 1
    assert batcher.native_particle_steps == 2
    assert batcher.fallback_particle_steps == 0
    assert batcher.max_group_size == 2

    after = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in original_kernels.provenance.dependency_paths.items()
    }
    assert after == before == dict(original_kernels.provenance.source_sha256)
    assert json.dumps(batcher.report(), ensure_ascii=False)


def test_native_nonuniform_vertical_shear_supports_sinking_and_reverse_sign(original_kernels) -> None:
    """非均勻水平／垂向剪切與定值沉降在正反向都沿用同一原版 RK4。"""

    grid = _grid()
    fields = _fields(sinking_mps=-0.08, nonuniform=True)
    state_forward = _state("p0")
    state_backward = _state("p1")
    frame = _frame(grid, fields, element_id=0, trat=0.25)
    batcher = NativeBayTraceBatchAdvector(original_kernels, lambda _task: frame)

    actual = batcher([_task(state_forward, 0.5), _task(state_backward, -0.5)])
    expected_forward = _direct_states(
        original_kernels,
        grid,
        fields,
        [state_forward],
        [0],
        dt_seconds=0.5,
        trat=0.25,
    )[0]
    expected_backward = _direct_states(
        original_kernels,
        grid,
        fields,
        [state_backward],
        [0],
        dt_seconds=-0.5,
        trat=0.25,
    )[0]

    assert all(isinstance(result, ParticleState) for result in actual)
    assert isinstance(actual[0], ParticleState)
    assert isinstance(actual[1], ParticleState)
    np.testing.assert_allclose(
        [actual[0].x_m, actual[0].y_m, actual[0].z_m],
        [expected_forward.x_m, expected_forward.y_m, expected_forward.z_m],
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        [actual[1].x_m, actual[1].y_m, actual[1].z_m],
        [expected_backward.x_m, expected_backward.y_m, expected_backward.z_m],
        rtol=0.0,
        atol=1.0e-12,
    )
    assert actual[0].z_m < state_forward.z_m
    assert actual[1].z_m > state_backward.z_m
    assert actual[0].time_utc_ns > state_forward.time_utc_ns
    assert actual[1].time_utc_ns < state_backward.time_utc_ns
    assert actual[0].age_seconds == actual[1].age_seconds == 0.5
    assert batcher.native_batch_calls == 2  # signed dt 不可被量化或合併


def test_grouping_inactive_scatter_and_report_counters(original_kernels) -> None:
    """不同 dt／trat 各自成組，inactive 不進 native，report 可序列化。"""

    grid = _grid()
    fields = _fields(sinking_mps=-0.02)
    frame_a = _frame(grid, fields, trat=0.25, element_id=0)
    frame_b = _frame(grid, fields, trat=0.75, element_id=0)
    states = [
        _state("p0"),
        _state("p1", x_m=0.30, y_m=0.30),
        _state("p2", x_m=0.35, y_m=0.25),
        _state("p3", status=ParticleStatus.COAST_CONTACT),
    ]
    frames = {"p0": frame_a, "p1": frame_b, "p2": frame_a}
    resolver_calls: list[str] = []

    def resolver(task: AdvectionTask) -> PreparedBayTraceFrame:
        resolver_calls.append(task.state.particle_id)
        return frames[task.state.particle_id]

    batcher = NativeBayTraceBatchAdvector(original_kernels, resolver)
    actual = batcher([
        _task(states[0], 0.25),
        _task(states[1], 0.5),
        _task(states[2], 0.25),
        _task(states[3], 0.25),
    ])

    assert [result.particle_id for result in actual if isinstance(result, ParticleState)] == [
        "p0",
        "p1",
        "p2",
        "p3",
    ]
    assert actual[3] is states[3]
    assert resolver_calls == ["p0", "p1", "p2"]
    assert batcher.native_batch_calls == 2
    assert batcher.native_particle_steps == 3
    assert batcher.max_group_size == 2
    report = batcher.report()
    assert report["source_sha256"] == dict(original_kernels.provenance.source_sha256)
    assert report["native_particle_steps"] == 3
    assert report["native_batch_calls"] == 2
    assert report["fallback_particle_steps"] == 0
    json.dumps(report, ensure_ascii=False)


def test_fallback_is_legacy_counted_and_none_does_not_claim_native(original_kernels) -> None:
    """resolver 的 None 明示保留既有 LBT fallback，不能冒充原版 kernel。"""

    state = _state("p0")
    task = _task(state, -0.25)
    batcher = NativeBayTraceBatchAdvector(original_kernels, lambda _task: None)
    actual = batcher([task])
    expected = execute_advection_task(task)

    assert isinstance(actual[0], ParticleState)
    assert isinstance(expected, ParticleState)
    np.testing.assert_allclose(
        [actual[0].x_m, actual[0].y_m, actual[0].z_m],
        [expected.x_m, expected.y_m, expected.z_m],
        rtol=0.0,
        atol=1.0e-12,
    )
    assert batcher.native_particle_steps == 0
    assert batcher.native_batch_calls == 0
    assert batcher.fallback_particle_steps == 1
    assert batcher.report()["fallback_backend"] == "lbt_legacy_execute_advection_task"


def test_invalid_dt_or_element_is_rejected_before_any_native_group(original_kernels, monkeypatch) -> None:
    """非法 dt／element 在預檢階段失敗，不可讓前一組先成功。"""

    import importlib

    adapter_module = importlib.import_module("lagrangian_backtracking.baytrace_batch_advection")
    grid = _grid()
    fields = _fields(sinking_mps=0.0)
    frame = _frame(grid, fields, element_id=0)
    state = _state("p0")
    calls = 0
    original_step = adapter_module.step_native

    def spy_step(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_step(*args, **kwargs)

    monkeypatch.setattr(adapter_module, "step_native", spy_step)
    batcher = NativeBayTraceBatchAdvector(original_kernels, lambda _task: frame)
    with pytest.raises(ValueError, match="dt_seconds"):
        batcher([_task(state, 0.25), _task(_state("p1"), 0.0)])
    assert calls == 0
    assert batcher.native_batch_calls == 0

    bad_frame = _frame(grid, fields, element_id=grid.i34.size)
    bad_batcher = NativeBayTraceBatchAdvector(original_kernels, lambda _task: bad_frame)
    with pytest.raises(ValueError, match="element_id"):
        bad_batcher([_task(state, 0.25)])
    assert calls == 0
    assert bad_batcher.native_batch_calls == 0

    invalid_group_batcher = NativeBayTraceBatchAdvector(original_kernels, lambda _task: object())
    with pytest.raises(TypeError, match="PreparedBayTraceFrame"):
        invalid_group_batcher([_task(state, 0.25)])
    assert invalid_group_batcher.native_batch_calls == 0


def test_loader_rejects_source_internal_cache_before_creating_directory(
    original_kernels,
    tmp_path: Path,
) -> None:
    """cache_dir 位於原始 source 內時，失敗前不得留下新目錄。"""

    forbidden_cache = original_kernels.provenance.package_dir / f".q2-cache-{tmp_path.name}"
    assert not forbidden_cache.exists()
    with pytest.raises(ValueError, match="source 目錄"):
        load_original_cpu(_BAYTRACE_ROOT, forbidden_cache)
    assert not forbidden_cache.exists()


def test_inactive_is_not_passed_and_nonfinite_native_output_becomes_sampling_error(
    original_kernels,
    monkeypatch,
) -> None:
    """inactive 不傳入；原始 kernel 結果若非有限則回傳 NUMERICAL_FAILURE。"""

    import importlib

    adapter_module = importlib.import_module("lagrangian_backtracking.baytrace_batch_advection")
    grid = _grid()
    fields = _fields(sinking_mps=0.0)
    active = _state("p0")
    inactive = _state("p1", status=ParticleStatus.COAST_CONTACT)
    frame = _frame(grid, fields, element_id=0)
    real_step = adapter_module.step_native

    def corrupt_step(*args, **kwargs):
        real_step(*args, **kwargs)
        particles = args[3]
        particles.x[0] = np.nan

    monkeypatch.setattr(adapter_module, "step_native", corrupt_step)
    batcher = NativeBayTraceBatchAdvector(original_kernels, lambda task: frame)
    actual = batcher([_task(active, 0.25), _task(inactive, 0.25)])

    assert isinstance(actual[0], SamplingError)
    assert actual[0].qc == SampleQC.NUMERICAL_FAILURE
    assert actual[0].stage == "native"
    assert actual[1] is inactive
    assert batcher.native_particle_steps == 0
    assert batcher.native_attempted_particle_steps == 1
    assert batcher.native_rejected_particle_steps == 1


def test_strict_native_rejects_without_legacy_and_reports_attempts(original_kernels, monkeypatch):
    """嚴格模式無frame不呼叫舊核心；成功組保持來源順序且計數只含實際完成。"""
    import lagrangian_backtracking.baytrace_batch_advection as module

    def forbidden(task):
        """任何legacy呼叫均表示違反本次native-only部署契約。"""
        raise AssertionError("strict mode不得呼叫legacy")

    monkeypatch.setattr(module, "_execute_legacy_advection_task", forbidden)
    frame = _frame(_grid(), _fields(sinking_mps=0.0), element_id=0)
    batcher = NativeBayTraceBatchAdvector(
        original_kernels, lambda task: None if task.state.particle_id == "bad" else frame,
        require_native=True,
    )
    results = batcher([_task(_state("p0"), 0.25), _task(_state("p1"), 0.25)])
    assert [r.particle_id for r in results] == ["p0", "p1"]
    with pytest.raises(module.NativeBayTraceUnsupportedError, match="frame_unavailable"):
        batcher([_task(_state("p0"), 0.25), _task(_state("bad"), 0.25)])
    report = batcher.report()
    assert report["native_completed_particle_steps"] == report["native_attempted_particle_steps"] == 2
    assert report["native_rejected_particle_steps"] == 1
    assert report["fallback_particle_steps"] == 0
    assert report["group_size_histogram"] == {"2": 1}


def test_native_kernel_exception_is_not_success(original_kernels, monkeypatch):
    """核心拋例外必須保留錯誤且不得預先增加成功計數。"""
    import lagrangian_backtracking.baytrace_batch_advection as module

    def fail(*args, **kwargs):
        """模擬原版dispatcher失敗，不嘗試另一物理核心。"""
        raise RuntimeError("kernel failure")

    monkeypatch.setattr(module, "step_native", fail)
    frame = _frame(_grid(), _fields(sinking_mps=0.0), element_id=0)
    batcher = NativeBayTraceBatchAdvector(original_kernels, lambda task: frame, require_native=True)
    with pytest.raises(RuntimeError, match="kernel failure"):
        batcher([_task(_state("p0"), 0.25)])
    assert batcher.native_particle_steps == batcher.fallback_particle_steps == 0
    assert batcher.native_attempted_particle_steps == batcher.native_rejected_particle_steps == 1


@pytest.mark.parametrize("value", [1, "true", None])
def test_require_native_is_strict_bool(original_kernels, value):
    """部署旗標不可用真假值隱式轉型，避免字串false卻開啟嚴格模式。"""
    with pytest.raises(TypeError, match="require_native"):
        NativeBayTraceBatchAdvector(original_kernels, lambda task: None, require_native=value)


def test_exact_utc_splits_native_groups_without_changing_identity(original_kernels):
    """同dt/trat但UTC差1奈秒不可同組；輸出身份、精確UTC與分組counter均保留。"""
    frame = _frame(_grid(), _fields(sinking_mps=0.0), element_id=0)
    first = _state("p0")
    second = replace(_state("p1"), time_utc_ns=first.time_utc_ns + 1)
    batcher = NativeBayTraceBatchAdvector(original_kernels, lambda task: frame, require_native=True)
    results = batcher([_task(first, 0.25), _task(second, 0.25)])
    assert [r.particle_id for r in results] == ["p0", "p1"]
    assert results[1].time_utc_ns - results[0].time_utc_ns == 1
    assert batcher.report()["group_size_histogram"] == {"1": 2}
    assert batcher.native_particle_steps == 2
