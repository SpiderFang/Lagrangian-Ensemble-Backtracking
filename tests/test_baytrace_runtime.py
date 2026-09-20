"""Q4 runtime wiring、設定 hash 與 advection execution metadata 測試。"""

from __future__ import annotations

import json
import sys
import types
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

import lagrangian_backtracking.baytrace_batch_advection as batch_advection_module
import lagrangian_backtracking.baytrace_native_cpu as native_cpu_module
import lagrangian_backtracking.runtime as runtime_module
from lagrangian_backtracking.advection import execute_advection_task
from lagrangian_backtracking.config import (
    ADVECTION_BACKEND_BAYTRACE_NATIVE_CPU_V1,
    ADVECTION_BACKEND_LBT_STAGE_RK4_V1,
    ProjectConfig,
)
from lagrangian_backtracking.run_control import (
    RunController,
    initialize_run_workspace,
    load_run_plan,
)


@pytest.mark.parametrize('frames,byte_limit', [
    ('20', None), (None, '1024'), ('', '1024'), ('2', ''), ('0', '1024'),
    ('2', '0'), ('-1', '1024'), ('2', '-1'), ('65', '1024'), ('20', '4294967297'),
    ('2.0', '1024'), (' 2', '1024'), ('2', '1e9'), ('true', '1024'), ('２', '1024')])
def test_native_frame_cache_env_rejects_invalid_pair(monkeypatch, frames, byte_limit):
    """半套、空值、非ASCII正整數及超限環境必須清楚拒絕，不得改成無界快取。"""
    for name, value in zip(('LBT_BAYTRACE_FRAME_CACHE_MAX_FRAMES',
                            'LBT_BAYTRACE_FRAME_CACHE_MAX_BYTES'), (frames, byte_limit), strict=True):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match='LBT_BAYTRACE_FRAME_CACHE_MAX_'):
        runtime_module._native_frame_cache_limits()


def _runtime_fixture_data() -> dict[str, Any]:
    """重用既有 runtime synthetic manifest builder，避免另造一套 request schema。"""

    # test_runtime 已包含完整的 Scenario／pair／geometry synthetic fixture；這裡只重用
    # 建構 helper，不讀取 OCM/NWW 陣列，也不把 synthetic 資料當成正式產品證據。
    import test_runtime as runtime_tests

    config = runtime_tests._configured_project_config()
    behavior, receptor, arrival, pair, scenario = runtime_tests._scenario_records(config)
    inputs = runtime_tests._scenario_inputs(behavior, receptor, arrival, pair, (scenario,))
    geometries = runtime_tests._geometry_bundle(receptor)
    return {
        "config": config,
        "inputs": inputs,
        "geometries": geometries,
        "scenario": scenario,
        "behavior": behavior,
        "receptor": receptor,
        "arrival": arrival,
        "pair": pair,
    }


def _native_config(config: ProjectConfig) -> ProjectConfig:
    """只替換 execution backend，保留其餘 fixture 的物理與輸入設定。"""

    execution = config.execution.model_copy(
        update={"advection_backend": ADVECTION_BACKEND_BAYTRACE_NATIVE_CPU_V1}
    )
    return config.model_copy(update={"execution": execution})


def test_advection_backend_default_preserves_old_hash_and_explicit_token_enters_payload() -> None:
    """舊 YAML 的 canonical bytes 不漂移，明示 native 才形成新的 config identity。"""

    import test_config as config_tests

    payload = config_tests._legacy_example_payload()
    assert isinstance(payload, dict)
    old = ProjectConfig.model_validate(payload)
    old_hash = "cf9dd4951d01708a69aa516189aa431754900feca65b2787c0ffa3bd36836c3a"

    assert old.execution.advection_backend == ADVECTION_BACKEND_LBT_STAGE_RK4_V1
    assert "advection_backend" not in old.normalized_payload()["execution"]
    assert old.config_hash() == old_hash

    explicit_payload = deepcopy(payload)
    explicit_payload["execution"]["advection_backend"] = (
        ADVECTION_BACKEND_BAYTRACE_NATIVE_CPU_V1
    )
    explicit = ProjectConfig.model_validate(explicit_payload)
    assert (
        explicit.normalized_payload()["execution"]["advection_backend"]
        == ADVECTION_BACKEND_BAYTRACE_NATIVE_CPU_V1
    )
    assert explicit.config_hash() != old_hash


@pytest.mark.parametrize("experiment_case_id", ["no_stokes", "finite_depth_stokes"])
@pytest.mark.parametrize('cache_limits', [None, (20, 4294967296)])
def test_runtime_factory_native_loader_uses_env_and_reports_source_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, experiment_case_id: str, cache_limits
) -> None:
    """純流／流波均強制原版批次器，且如實保留成功與工程拒絕的不同證據。"""

    data = _runtime_fixture_data()
    source_root = tmp_path / "baytrace-artifact"
    source_root.mkdir()
    (tmp_path / "ocm-native").mkdir()
    (tmp_path / "nww-analysis").mkdir()
    monkeypatch.setenv("LBT_BAYTRACE_ROOT", str(source_root))
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    for index, name in enumerate(('LBT_BAYTRACE_FRAME_CACHE_MAX_FRAMES',
                                  'LBT_BAYTRACE_FRAME_CACHE_MAX_BYTES')):
        if cache_limits is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, str(cache_limits[index]))

    class FakeResolver:
        """只驗證 Q3 建構參數與 report 邊界的 resolver 替身。"""

        init_kwargs: dict[str, Any] = {}

        def __init__(self, **kwargs: Any) -> None:
            type(self).init_kwargs = dict(kwargs)

        def __call__(self, _task: Any) -> None:
            return None

        def report(self) -> dict[str, Any]:
            return {"native_frame_count": 0, "fallback_frame_count": 0}

    class FakeBatcher:
        """只驗證 factory 接上 Q2 介面的最小 batcher。"""

        def __init__(self, kernels: Any, resolver: Any, *, require_native: bool) -> None:
            self.kernels = kernels
            self.resolver = resolver
            self.require_native = require_native

        def __call__(self, _tasks: Any) -> list[Any]:
            return []

        def report(self) -> dict[str, Any]:
            return {
                "source_root": str(source_root),
                "cache_dir": str(self.kernels["cache_dir"]),
                "source_sha256": {"geometry": "a" * 64},
                "native_particle_steps": 3,
                "native_batch_calls": 2,
                "fallback_particle_steps": 1,
                "max_group_size": 2,
                "native_attempted_particle_steps": 4,
                "native_completed_particle_steps": 3,
                "native_rejected_particle_steps": 1,
                "native_rejected_reasons": {"native_kernel_exception": 1},
                "group_size_histogram": {"2": 2},
                "counter_semantics": "native_steps_completed_only_v2",
                "require_native": self.require_native,
            }

    def fake_loader(root: Path, cache_dir: Path) -> dict[str, Path]:
        """避免 Q4 wiring test 依賴本機是否存在 PI artifact。"""

        return {"source_root": root, "cache_dir": cache_dir}

    q3_module = types.ModuleType("lagrangian_backtracking.baytrace_ocm_adapter")
    q3_module.OCMPreparedFrameResolver = FakeResolver
    monkeypatch.setitem(sys.modules, q3_module.__name__, q3_module)
    monkeypatch.setattr(native_cpu_module, "load_original_cpu", fake_loader)
    monkeypatch.setattr(batch_advection_module, "NativeBayTraceBatchAdvector", FakeBatcher)

    factory = runtime_module.RuntimeRequestFactory(
        config=_native_config(data["config"]),
        scenario_inputs=data["inputs"],
        geometries=data["geometries"],
        ocm_native_root=tmp_path / "ocm-native",
        nww_analysis_root=tmp_path / "nww-analysis" if experiment_case_id != "no_stokes" else None,
        experiment_case_id=experiment_case_id,
    )

    assert isinstance(factory.advection_batcher, FakeBatcher)
    assert factory.advection_batcher.require_native is True
    assert FakeResolver.init_kwargs == {
        "max_cached_frames": 2 if cache_limits is None else cache_limits[0],
        "max_cached_bytes": 512 * 1024 * 1024 if cache_limits is None else cache_limits[1],
    }
    report = factory.advection_report()
    assert report["backend"] == ADVECTION_BACKEND_BAYTRACE_NATIVE_CPU_V1
    assert report["source_environment"]["LBT_BAYTRACE_ROOT"] == str(source_root)
    assert report["source_environment"]["NUMBA_CACHE_DIR"] is None
    cache_path = Path(report["cache_dir"])
    assert cache_path.is_dir()
    assert cache_path != source_root
    assert source_root not in cache_path.parents
    assert report["source_sha256"] == {"geometry": "a" * 64}
    assert report["native_particle_steps"] == 3
    for key in (
        "native_attempted_particle_steps", "native_completed_particle_steps",
        "native_rejected_particle_steps", "native_rejected_reasons",
        "group_size_histogram", "counter_semantics", "require_native",
    ):
        assert report[key] == factory.advection_batcher.report()[key]
    assert report["resolver_report"]["native_frame_count"] == 0


def test_native_factory_missing_source_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """明示 native 卻缺少 source env 時不可靜默回到 legacy。"""

    data = _runtime_fixture_data()
    (tmp_path / "ocm-native").mkdir()
    monkeypatch.delenv("LBT_BAYTRACE_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="LBT_BAYTRACE_ROOT"):
        runtime_module.RuntimeRequestFactory(
            config=_native_config(data["config"]),
            scenario_inputs=data["inputs"],
            geometries=data["geometries"],
            ocm_native_root=tmp_path / "ocm-native",
            nww_analysis_root=None,
            experiment_case_id="no_stokes",
        )
    monkeypatch.setenv("LBT_BAYTRACE_ROOT", str(tmp_path / "missing-baytrace"))
    with pytest.raises(FileNotFoundError, match="LBT_BAYTRACE_ROOT"):
        runtime_module.RuntimeRequestFactory(
            config=_native_config(data["config"]),
            scenario_inputs=data["inputs"],
            geometries=data["geometries"],
            ocm_native_root=tmp_path / "ocm-native",
            nww_analysis_root=None,
            experiment_case_id="no_stokes",
        )


def test_open_run_controller_passes_factory_batcher_reporter_and_keeps_time_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """open controller 應把 factory 的兩個 native callback 傳入，且保留時間 fallback。"""

    captured: dict[str, Any] = {}
    static_inputs = types.SimpleNamespace(
        plan={"run_kind": "pilot", "experiment_case_id": "no_stokes"},
        config=object(),
        scenario_inputs=object(),
        geometries=object(),
    )
    monkeypatch.setattr(
        runtime_module,
        "load_validated_run_static_inputs",
        lambda *args, **kwargs: static_inputs,
    )

    class FakeFactory:
        """只記錄 open controller 的 factory／reporter wiring。"""

        def __init__(self, **kwargs: Any) -> None:
            captured["factory_kwargs"] = kwargs
            self.advection_batcher = lambda _tasks: []

        def __call__(self, _unit: Any) -> Any:
            return None

        def resource_stats(self) -> dict[str, int]:
            return {
                "manager_count": 0,
                "loads": 0,
                "hits": 0,
                "misses": 0,
                "evictions": 0,
                "resident_bytes": 0,
            }

        def advection_report(self) -> dict[str, Any]:
            return {"backend": ADVECTION_BACKEND_BAYTRACE_NATIVE_CPU_V1}

    class FakeController:
        """避免此 wiring test 啟動 shard，只驗證 constructor 參數。"""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            captured["controller_args"] = args
            captured["controller_kwargs"] = kwargs

    monkeypatch.setattr(runtime_module, "RuntimeRequestFactory", FakeFactory)
    monkeypatch.setattr(runtime_module, "RunController", FakeController)

    controller = runtime_module._open_run_controller(
        tmp_path,
        config_path=tmp_path / "config.yaml",
        ocm_native_root=tmp_path / "ocm-native",
        nww_analysis_root=None,
        continuous_time_fallback=True,
    )

    assert isinstance(controller, FakeController)
    assert captured["factory_kwargs"]["continuous_time_fallback"] is True
    controller_kwargs = captured["controller_kwargs"]
    assert controller_kwargs["advection_batcher"] is not None
    assert controller_kwargs["advection_reporter"]() == {
        "backend": ADVECTION_BACKEND_BAYTRACE_NATIVE_CPU_V1
    }


def test_runtime_factory_real_source_and_synthetic_ocm_reach_native_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """真實 PI 來源與合成流場須形成多粒子群組並推進，工程拒絕不得當成完成。"""

    data = _runtime_fixture_data()
    source_root = Path(
        "/Users/mustlab/Workspace/Lagrangian-Ensemble-Backtracking/"
        "BayTrace/Baytech_BayTrace_v4.7.2"
    )
    if not source_root.is_dir():
        pytest.skip("未找到明確指定的 PI BayTrace artifact")
    (tmp_path / "ocm-native").mkdir()
    monkeypatch.setenv("LBT_BAYTRACE_ROOT", str(source_root))
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(tmp_path / "numba-cache"))

    factory = runtime_module.RuntimeRequestFactory(
        config=_native_config(data["config"]),
        scenario_inputs=data["inputs"],
        geometries=data["geometries"],
        ocm_native_root=tmp_path / "ocm-native",
        nww_analysis_root=None,
        experiment_case_id="no_stokes",
    )

    # Q3 test helper 只建立小型 synthetic OCM month／manager；真正的 kernel 仍由
    # factory 依 LBT_BAYTRACE_ROOT 載入，沒有以 mock 取代 BayTrace dispatcher。
    import test_baytrace_ocm_adapter as q3_tests

    from lagrangian_backtracking.forcing_window import ManagedForcingProvider
    from lagrangian_backtracking.production import HintTrackingVelocityProvider

    mesh = q3_tests._mesh()
    month = q3_tests._month(mesh, settling_current_mps=0.03)
    manager = q3_tests._manager(mesh, {month.month_id: month})
    managed = ManagedForcingProvider(
        manager,
        settling_velocity_mps=-0.02,
        include_stokes=False,
    )
    provider = HintTrackingVelocityProvider(managed, triangle_hint=0)
    task = q3_tests._task(provider)
    assert factory.advection_batcher is not None
    from dataclasses import replace

    from numba.core.registry import CPUDispatcher

    second_task = replace(task, state=replace(task.state, particle_id="p1"))
    results = factory.advection_batcher([task, second_task])
    for result in results:
        assert result.__class__.__name__ == "ParticleState"
        assert result.time_utc_ns != task.state.time_utc_ns
        assert (result.x_m, result.y_m, result.z_m) != (
            task.state.x_m, task.state.y_m, task.state.z_m
        )
    assert isinstance(factory.advection_batcher.kernels.batch_move_rk4, CPUDispatcher)
    assert factory.advection_batcher.kernels.batch_move_rk4.signatures

    report = factory.advection_report()
    assert report["native_particle_steps"] > 0, report
    assert report["native_batch_calls"] > 0
    assert report["fallback_particle_steps"] == 0
    assert report["native_rejected_particle_steps"] == 0
    assert report["native_completed_particle_steps"] == 2
    assert report["native_attempted_particle_steps"] == 2
    assert report["group_size_histogram"] == {"2": 1}
    assert report["resolver_report"]["frame_builds"] > 0
    assert set(report["source_sha256"]) == {"geometry", "vertical", "random", "batch_interp"}


def _scenario(index: int) -> Any:
    """建立 controller synthetic workspace 所需的固定 scenario。"""

    from lagrangian_backtracking.scenarios import Scenario, stable_identifier

    receptor_id = f"receptor-{index}"
    arrival_id = f"arrival-{index}"
    return Scenario(
        scenario_id=stable_identifier(
            "scn", ["gongliao", "material", receptor_id, arrival_id, "q4-test-v1"]
        ),
        study_site_id="gongliao",
        analysis_region_id="A",
        material_id="material",
        receptor_id=receptor_id,
        arrival_time_id=arrival_id,
        settling_velocity_mps=-0.001,
        arrival_time_utc_ns=1_700_000_000_000_000_000 + index,
        design_version="q4-test-v1",
    )


def _controller_request(unit: Any) -> Any:
    """以零流速、零擴散 request 驗證 controller 不改變 legacy lifecycle。"""

    from shapely.geometry import box

    from lagrangian_backtracking.boundaries import BoundaryGeometry
    from lagrangian_backtracking.diffusion import DiffusionCoefficients
    from lagrangian_backtracking.engine import EngineSettings
    from lagrangian_backtracking.models import ParticleState, VelocitySample
    from lagrangian_backtracking.production import ReferenceParticleRequest

    state = ParticleState(
        particle_id=unit.particle_id,
        scenario_id=unit.scenario.scenario_id,
        member_id=unit.member_id,
        study_site_id=unit.scenario.study_site_id,
        analysis_region_id=unit.scenario.analysis_region_id,
        receptor_id=unit.scenario.receptor_id,
        x_m=0.0,
        y_m=0.0,
        z_m=-5.0,
        time_utc_ns=unit.scenario.arrival_time_utc_ns,
    )

    def velocity(
        _x_m: float, _y_m: float, _z_m: float, _time_utc_ns: int
    ) -> VelocitySample:
        """提供有限 synthetic sample；所有物理步仍由既有 engine 執行。"""

        return VelocitySample(0.0, 0.0, 0.0, 0.0, -10.0, 100.0, 10.0)

    return ReferenceParticleRequest(
        initial_state=state,
        velocity=velocity,
        boundaries=BoundaryGeometry(box(-5, -5, 5, 5), box(-20, -20, 20, 20), {}),
        behavior_class="sinking",
        diffusion=DiffusionCoefficients(0.0, 0.0, 0.0),
        settings=EngineSettings(1.0, 1.0, 1.0, 3.0, 20, 0),
    )


def _controller_workspace(parent: Path, run_id: str) -> Path:
    """建立兩 member、單 shard 的 synthetic controller workspace。"""

    from lagrangian_backtracking.provenance import CodeProvenance

    return initialize_run_workspace(
        parent,
        run_id=run_id,
        scenarios=(_scenario(0),),
        normalized_config={"schema_version": "q4-test", "settings": {"dt": 1.0}},
        config_hash="9" * 64,
        input_inventory_file={"source": "synthetic", "files": []},
        component_canonical_hashes={"material": "a" * 64, "receptor": "b" * 64},
        geometry_canonical_hashes={"domain": "c" * 64, "local": "d" * 64},
        provenance=CodeProvenance(
            git_available=False,
            git_commit=None,
            git_dirty=None,
            commit_source="no_git_pilot",
            deployment_tree_sha256="1" * 64,
            deployment_file_count=1,
            uv_lock_sha256="2" * 64,
            python_version="3.12.0",
            platform="test",
            package_version="0.1.0",
            numpy_version="test",
            numba_version="test",
            pyarrow_version="test",
        ),
        experiment_case_id="baseline",
        master_seed=20260919,
        seed_policy="sha256_v1_pcg64dxsm",
        members_per_scenario=2,
        shard_scenario_count=1,
        checkpoint_interval_sweeps=1,
        active_chunk_size=2,
        run_kind="synthetic",
    ).path


class _SpyAdvector:
    """模擬 Q2 batcher 的 deterministic callable 與可稽核累計 report。"""

    def __init__(self) -> None:
        self.calls = 0
        self.steps = 0

    def __call__(self, tasks: Any) -> list[Any]:
        tasks = tuple(tasks)
        self.calls += 1
        self.steps += len(tasks)
        return [execute_advection_task(task) for task in tasks]

    def report(self) -> dict[str, Any]:
        return {
            "backend": "native_baytrace_batch_advector",
            "source_root": "/tmp/q4-test-baytrace",
            "cache_dir": "/tmp/q4-test-cache",
            "source_sha256": {"geometry": "a" * 64},
            "native_particle_steps": self.steps,
            "native_batch_calls": self.calls,
            "fallback_particle_steps": 0,
            "max_group_size": 2,
            "resolver_report": {
                "native_frame_count": self.steps,
                "fallback_frame_count": 0,
            },
        }


def test_controller_injects_batcher_on_new_and_restore_paths_and_writes_scoped_metrics(
    tmp_path: Path,
) -> None:
    """新建與 checkpoint restore 都注入 batcher，且 output 只寫當次 invocation evidence。"""

    workspace = _controller_workspace(tmp_path, "q4-native-controller")
    shard_id = load_run_plan(workspace)["shards"][0]["shard_id"]
    first_batcher = _SpyAdvector()
    first = RunController(
        workspace,
        request_factory=_controller_request,
        advection_batcher=first_batcher,
        advection_reporter=first_batcher.report,
    )
    paused = first.run_shard(shard_id, sweep_budget=1)
    assert paused.lifecycle == "PAUSED"
    assert first_batcher.calls > 0

    restored_batcher = _SpyAdvector()
    restored = RunController(
        workspace,
        request_factory=_controller_request,
        resume=True,
        advection_batcher=restored_batcher,
        advection_reporter=restored_batcher.report,
    )
    completed = restored.run_shard(shard_id)
    assert completed.lifecycle == "COMPLETE"
    assert restored_batcher.calls > 0

    manifest = json.loads(
        (workspace / "shards" / shard_id / "manifest.json").read_text(encoding="utf-8")
    )
    execution = manifest["run_metadata"]["advection_execution"]
    assert execution["counter_scope"] == "shard_invocation_delta_v1"
    assert execution["invocation"] == {
        "scope": "single_shard_controller_invocation",
        "shard_id": shard_id,
    }
    assert execution["native_particle_steps"] > 0
    assert execution["native_particle_steps"] == restored_batcher.steps
    assert execution["native_batch_calls"] == restored_batcher.calls
    assert execution["resolver_report"]["native_frame_count"] == restored_batcher.steps
    assert "forcing_cache_stats" not in execution


def test_advection_delta_uses_explicit_paths_for_nonzero_process_baseline() -> None:
    """跨 shard／resume 的 process snapshot 只扣除已登錄 counter，保留 metadata 與 gauge。"""

    baseline = {
        "native_particle_steps": 100,
        "native_batch_calls": 12,
        "fallback_particle_steps": 4,
        "max_group_size": 8,
        "source_metadata": {"frame_builds": 30, "bracket_queries": 40},
        "resolver_report": {
            "cache_hits": 20,
            "cache_misses": 21,
            "frame_builds": 7,
            "frame_evictions": 2,
            "negative_cache_hits": 3,
            "negative_cache_stores": 4,
            "negative_cache_evictions": 5,
            "bracket_cache_hits": 6,
            "bracket_queries": 10,
            "time_fallback_count": 8,
            "fallback_count": 9,
            "fallback_reasons": {"unsupported_provider_wrapper": 3},
            "cached_frames": 1,
            "cached_bytes": 100,
            "max_cached_frames": 2,
            "max_cached_bytes": 512,
            "negative_cache_entries": 1,
            "negative_cache_capacity": 2,
            "native_frame_count": 50,
            "source_metadata": {"frame_builds": 60, "bracket_queries": 70},
        },
    }
    current = deepcopy(baseline)
    current.update(
        {
            "native_particle_steps": 107,
            "native_batch_calls": 15,
            "fallback_particle_steps": 6,
            "max_group_size": 16,
            "source_metadata": {"frame_builds": 31, "bracket_queries": 41},
        }
    )
    resolver = current["resolver_report"]
    assert isinstance(resolver, dict)
    resolver.update(
        {
            "cache_hits": 22,
            "cache_misses": 24,
            "frame_builds": 9,
            "frame_evictions": 3,
            "negative_cache_hits": 7,
            "negative_cache_stores": 6,
            "negative_cache_evictions": 8,
            "bracket_cache_hits": 9,
            "bracket_queries": 15,
            "time_fallback_count": 10,
            "fallback_count": 11,
            "fallback_reasons": {
                "unsupported_provider_wrapper": 5,
                "different_mesh": 1,
            },
            "cached_frames": 2,
            "cached_bytes": 300,
            "max_cached_frames": 4,
            "max_cached_bytes": 1024,
            "negative_cache_entries": 2,
            "negative_cache_capacity": 4,
            # 舊 substring 猜測會把這個未登錄欄位也扣成 2；新契約必須保留 55。
            "native_frame_count": 55,
            "source_metadata": {"frame_builds": 61, "bracket_queries": 71},
        }
    )

    delta = RunController._advection_counter_delta(current, baseline)

    assert delta["native_particle_steps"] == 7
    assert delta["native_batch_calls"] == 3
    assert delta["fallback_particle_steps"] == 2
    assert delta["max_group_size"] == 16
    assert delta["source_metadata"] == current["source_metadata"]
    resolver_delta = delta["resolver_report"]
    assert isinstance(resolver_delta, dict)
    assert resolver_delta["frame_builds"] == 2
    assert resolver_delta["bracket_queries"] == 5
    assert resolver_delta["cache_hits"] == 2
    assert resolver_delta["negative_cache_stores"] == 2
    assert resolver_delta["fallback_count"] == 2
    assert resolver_delta["fallback_reasons"] == {
        "unsupported_provider_wrapper": 2,
        "different_mesh": 1,
    }
    assert resolver_delta["cached_frames"] == 2
    assert resolver_delta["cached_bytes"] == 300
    assert resolver_delta["max_cached_frames"] == 4
    assert resolver_delta["max_cached_bytes"] == 1024
    assert resolver_delta["negative_cache_entries"] == 2
    assert resolver_delta["negative_cache_capacity"] == 4
    assert resolver_delta["native_frame_count"] == 55
    assert resolver_delta["source_metadata"] == resolver["source_metadata"]

    regressed = deepcopy(current)
    regressed["native_batch_calls"] = 11
    with pytest.raises(ValueError, match="advection counter 不得倒退"):
        RunController._advection_counter_delta(regressed, baseline)

    regressed = deepcopy(current)
    regressed["resolver_report"]["frame_builds"] = 6
    with pytest.raises(ValueError, match="resolver_report.frame_builds"):
        RunController._advection_counter_delta(regressed, baseline)


def test_native_report_new_counters_are_per_invocation_and_metadata_is_preserved() -> None:
    """持久 worker 的第二片只保存新事件；拒絕理由與群組大小也需扣除前片累計。"""

    baseline = {
        "native_attempted_particle_steps": 10,
        "native_completed_particle_steps": 8,
        "native_rejected_particle_steps": 2,
        "native_rejected_reasons": {"native_kernel_exception": 2},
        "group_size_histogram": {"2": 5},
        "resolver_report": {"cache_warmup_builds": 3},
    }
    current = {
        "native_attempted_particle_steps": 14,
        "native_completed_particle_steps": 11,
        "native_rejected_particle_steps": 3,
        "native_rejected_reasons": {"native_kernel_exception": 2, "frame_unavailable": 1},
        "group_size_histogram": {"2": 6, "4": 1},
        "resolver_report": {
            "cache_warmup_builds": 4,
            "field_policy": "ocm_wet_support_node_fields_v1",
            "last_rejection_reason": "frame_unavailable",
            "last_rejection_qc": None,
            "include_stokes": False,
        },
        "require_native": True,
        "counter_semantics": "native_steps_completed_only_v2",
        "max_group_size": 4,
        "source_metadata": {"native_attempted_particle_steps": 99},
    }
    delta = RunController._advection_counter_delta(current, baseline)
    assert delta["native_attempted_particle_steps"] == 4
    assert delta["native_completed_particle_steps"] == 3
    assert delta["native_rejected_particle_steps"] == 1
    assert delta["native_rejected_reasons"] == {"native_kernel_exception": 0, "frame_unavailable": 1}
    assert delta["group_size_histogram"] == {"2": 1, "4": 1}
    assert delta["resolver_report"] == {**current["resolver_report"], "cache_warmup_builds": 1}
    for key in ("require_native", "counter_semantics", "max_group_size", "source_metadata"):
        assert delta[key] == current[key]
    for key, nested_key in (("native_rejected_reasons", "native_kernel_exception"),
                            ("group_size_histogram", "2")):
        regressed = deepcopy(current)
        regressed[key][nested_key] = 0
        with pytest.raises(ValueError, match="advection counter 不得倒退"):
            RunController._advection_counter_delta(regressed, baseline)


def test_legacy_controller_output_has_no_fake_native_metadata(tmp_path: Path) -> None:
    """未注入 batcher 的舊 controller 不新增 advection execution 欄位。"""

    workspace = _controller_workspace(tmp_path, "q4-legacy-controller")
    shard_id = load_run_plan(workspace)["shards"][0]["shard_id"]
    result = RunController(workspace, request_factory=_controller_request).run_shard(shard_id)
    assert result.lifecycle == "COMPLETE"
    manifest = json.loads(
        (workspace / "shards" / shard_id / "manifest.json").read_text(encoding="utf-8")
    )
    assert "advection_execution" not in manifest["run_metadata"]
