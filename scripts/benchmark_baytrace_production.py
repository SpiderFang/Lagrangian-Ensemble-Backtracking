#!/usr/bin/env python3
"""Q3：以同一 ProductionBatch lifecycle 比較 LBT legacy 與原版 BayTrace batch。

本腳本重用 Q2 的小型 synthetic OCM case 建構器，但完整路徑的 request 使用
``ManagedForcingProvider``，native 路徑再經 ``OCMPreparedFrameResolver`` 由 OCM
兩端 row 實際建立／轉置 frame；不把預先建好的 ``PreparedBayTraceFrame`` 或 lambda
當成 forcing 轉接。兩條路徑都執行同一個 ``ProductionBatch``，包含 reference engine
的步首取樣、RK4、非零 settling、非零 diffusion、boundary check、observation 與
event/output container。只比較這兩條 production orchestration，不能解讀成完整
BayTrace tracker、真實 OCM/NWW 或 SERVER throughput 證據。

預設跑 128×16×3 與 512×16×3（粒子×sweep×repeat）兩個小型 case。首次 frame
build／轉置的成本包含在 native advance wall time，同時由 resolver report 分開列出
``frame_builds``、``cache_hits``、``fallback_reasons``；warmup 另行執行且不納入
steady timing。輸出目錄預設為指定的
``/private/tmp/lbt-baytrace-production-evidence-20260919``，若目標 JSON 已存在即拒絕
覆寫。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# 腳本不是 package module；明確把 scripts 目錄放入 import path，重用 Q2 已驗證的
# BayTrace case／環境 helper，而不在 Q3 另寫第二份原始 kernel loader。
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import benchmark_baytrace_native_cpu as q2  # noqa: E402

from lagrangian_backtracking.baytrace_batch_advection import (  # noqa: E402
    NativeBayTraceBatchAdvector,
)
from lagrangian_backtracking.baytrace_ocm_adapter import OCMPreparedFrameResolver  # noqa: E402
from lagrangian_backtracking.boundaries import BoundaryGeometry  # noqa: E402
from lagrangian_backtracking.diffusion import DiffusionCoefficients  # noqa: E402
from lagrangian_backtracking.engine import EngineSettings  # noqa: E402
from lagrangian_backtracking.forcing_window import (  # noqa: E402
    ForcingWindowManager,
    ManagedForcingProvider,
)
from lagrangian_backtracking.geometry import DomainProjection  # noqa: E402
from lagrangian_backtracking.models import ParticleState  # noqa: E402
from lagrangian_backtracking.production import ProductionBatch  # noqa: E402
from lagrangian_backtracking.runner import (  # noqa: E402
    ReferenceParticleRequest,
    RunUnit,
    plan_scenario_shards,
)
from lagrangian_backtracking.scenarios import Scenario  # noqa: E402

SCHEMA = "q3_baytrace_production_benchmark_v1"
DEFAULT_OUTPUT_DIR = Path("/private/tmp/lbt-baytrace-production-evidence-20260919")
DEFAULT_CASES = ((128, 16, 3), (512, 16, 3))
SETTLING_VELOCITY_MPS = -0.0004
DIFFUSION_M2PS = 1.0e-4
MAX_POSITION_ERROR_M = 5.0e-5


@dataclass(slots=True)
class PipelineRun:
    """保存一次 ProductionBatch 的 timing、輸出摘要與 native resolver evidence。"""

    mode: str
    setup_seconds: float
    advance_seconds: float
    batch: ProductionBatch
    resolver_report: dict[str, Any] | None
    batcher_report: dict[str, Any] | None


def _manager_for_case(case: Any) -> ForcingWindowManager:
    """以同一 synthetic OCM month 建立正式 manager facade，不預先轉成 BayTrace frame。"""

    month = case.month

    def load_ocm(month_id: str) -> Any:
        """只提供 Q2 synthetic 的 197001 month；其他月份保持缺月語意。"""

        return month if month_id == month.month_id else None

    return ForcingWindowManager(
        flow_domain_id="q3-production-synthetic",
        projection=DomainProjection(121.0, 25.0),
        mesh=month.mesh,
        ocm_loader=load_ocm,
        max_resident_months=2,
        continuous_time_fallback=False,
    )


def _make_batch(
    case: Any,
    *,
    steps: int,
    kernels: Any | None,
    mode: str,
) -> tuple[ProductionBatch, OCMPreparedFrameResolver | None, NativeBayTraceBatchAdvector | None]:
    """建立 old／native 兩個同設定 ProductionBatch，唯一差別是 advection callback。"""

    if mode not in {"legacy", "native"}:
        raise ValueError(f"未知 production mode：{mode}")
    manager = _manager_for_case(case)
    scenarios = tuple(
        Scenario(
            # old/native 兩條 pipeline 必須共用相同 scenario identity，才能讓
            # ProductionBatch 分配完全相同的 PCG64DXSM diffusion stream。
            scenario_id=f"q3-{index:08d}",
            study_site_id="synthetic",
            analysis_region_id="synthetic",
            material_id="q3-nonzero-settling",
            receptor_id=f"r-{index:08d}",
            arrival_time_id="arrival-0",
            settling_velocity_mps=SETTLING_VELOCITY_MPS,
            arrival_time_utc_ns=q2.INITIAL_TIME_NS,
            design_version=SCHEMA,
        )
        for index in range(case.n)
    )
    shard = plan_scenario_shards(
        scenarios,
        members_per_scenario=1,
        shard_scenario_count=case.n,
        experiment_case_id="q3-production-comparison",
    )[0]
    from shapely.geometry import box

    boundary = BoundaryGeometry(
        # 此 domain 刻意包住所有 synthetic 粒子；每一步仍執行 boundary code，但不把
        # 終止型海岸線案例混入數值最大誤差比較，避免輸出拓撲差異掩蓋平流結果。
        own_local_domain=box(-1.0e9, -1.0e9, 1.0e9, 1.0e9),
        flow_domain=box(-1.0e9, -1.0e9, 1.0e9, 1.0e9),
        foreign_local_domains={},
        local_equals_flow=True,
    )
    settings = EngineSettings(
        dt_min_seconds=case.dt_seconds,
        dt_max_seconds=case.dt_seconds,
        output_interval_seconds=case.dt_seconds,
        max_backtrack_seconds=(steps + 1) * case.dt_seconds,
        maximum_step_count=steps + 2,
        earliest_forcing_time_utc_ns=0,
        physics_kernel_backend="numba_cpu_v1",
    )
    diffusion = DiffusionCoefficients(
        DIFFUSION_M2PS,
        DIFFUSION_M2PS,
        DIFFUSION_M2PS,
    )

    def request_factory(unit: RunUnit) -> ReferenceParticleRequest:
        """用 unit suffix 對回相同 initial position，保留 stable identity。"""

        index = int(unit.scenario.scenario_id.rsplit("-", 1)[1])
        provider = ManagedForcingProvider(
            manager,
            settling_velocity_mps=SETTLING_VELOCITY_MPS,
            include_stokes=False,
        )
        return ReferenceParticleRequest(
            initial_state=ParticleState(
                particle_id=unit.particle_id,
                scenario_id=unit.scenario.scenario_id,
                member_id=unit.member_id,
                study_site_id="synthetic",
                analysis_region_id="synthetic",
                receptor_id=unit.scenario.receptor_id,
                x_m=float(case.x[index]),
                y_m=float(case.y[index]),
                z_m=float(case.z[index]),
                time_utc_ns=q2.INITIAL_TIME_NS,
            ),
            velocity=provider,
            boundaries=boundary,
            behavior_class="sinking",
            diffusion=diffusion,
            settings=settings,
        )

    resolver: OCMPreparedFrameResolver | None = None
    batcher: NativeBayTraceBatchAdvector | None = None
    if mode == "native":
        if kernels is None:
            raise ValueError("native production batch 必須提供原版 CPU kernels")
        resolver = OCMPreparedFrameResolver(max_cached_frames=2)
        batcher = NativeBayTraceBatchAdvector(kernels, resolver)
    batch = ProductionBatch(
        shard,
        master_seed=20260919,
        request_factory=request_factory,
        active_chunk_size=case.n,
        advection_batcher=batcher,
    )
    return batch, resolver, batcher


def _output_snapshot(batch: ProductionBatch) -> dict[str, Any]:
    """整理兩條 pipeline 的 state、observation、boundary event 輸出摘要。"""

    executions = batch.execution_states
    final_positions = np.asarray(
        [[item.state.x_m, item.state.y_m, item.state.z_m] for item in executions],
        dtype=np.float64,
    )
    final_time_age_status = [
        {
            "particle_id": item.state.particle_id,
            "time_utc_ns": int(item.state.time_utc_ns),
            "age_seconds": float(item.state.age_seconds),
            "status": str(item.state.status),
            "observation_count": len(item.observations),
            "event_count": len(item.events),
            "event_types": [str(event.event_type) for event in item.events],
        }
        for item in executions
    ]
    observation_rows = [
        [
            int(observation.time_utc_ns),
            float(observation.age_seconds),
            float(observation.x_m),
            float(observation.y_m),
            float(observation.z_m),
            str(observation.status),
        ]
        for item in executions
        for observation in item.observations
    ]
    event_rows = [
        [
            event.particle_id,
            str(event.event_type),
            int(event.time_utc_ns),
            float(event.fraction),
        ]
        for item in executions
        for event in item.events
    ]
    return {
        "final_positions": final_positions,
        "final_time_age_status": final_time_age_status,
        "observations": observation_rows,
        "events": event_rows,
        "active_count": int(batch.active_count),
        "sweeps_completed": int(batch.sweep_count),
    }


def _jsonable_output(snapshot: dict[str, Any]) -> dict[str, Any]:
    """將摘要轉成可寫入 JSON 的 bounded metadata，不寫整批大型 arrays。"""

    positions = np.asarray(snapshot["final_positions"], dtype=np.float64)
    return {
        "final_position_shape": list(positions.shape),
        "final_position_finite": bool(np.all(np.isfinite(positions))),
        "final_time_age_status": snapshot["final_time_age_status"],
        "observation_count_total": len(snapshot["observations"]),
        "event_count_total": len(snapshot["events"]),
        "event_types": sorted({row[1] for row in snapshot["events"]}),
        "active_count": snapshot["active_count"],
        "sweeps_completed": snapshot["sweeps_completed"],
    }


def _run_once(case: Any, *, steps: int, kernels: Any, mode: str) -> PipelineRun:
    """建立並執行一次完整 ProductionBatch；timing 包含各自設定建立與 advance。"""

    setup_start = time.perf_counter()
    batch, resolver, batcher = _make_batch(case, steps=steps, kernels=kernels, mode=mode)
    setup_seconds = time.perf_counter() - setup_start
    advance_start = time.perf_counter()
    result = batch.advance(steps)
    advance_seconds = time.perf_counter() - advance_start
    if result.sweeps_completed != steps or result.stepped_particle_count != case.n * steps:
        raise RuntimeError(f"{mode} ProductionBatch step count 不符：{result}")
    # 明確觸發 output/state validation；不呼叫 complete，避免額外 terminal sweep 改變
    # benchmark 的固定 16-sweep lifecycle。
    batch._validate_alignment()
    return PipelineRun(
        mode=mode,
        setup_seconds=setup_seconds,
        advance_seconds=advance_seconds,
        batch=batch,
        resolver_report=resolver.report() if resolver is not None else None,
        batcher_report=batcher.report() if batcher is not None else None,
    )


def _warmup(case: Any, *, steps: int, kernels: Any) -> dict[str, float]:
    """各路徑先跑一次，將 JIT 與原版 resolver 建置成本獨立列出。"""

    timings: dict[str, float] = {}
    for mode in ("legacy", "native"):
        start = time.perf_counter()
        _run_once(case, steps=1, kernels=kernels, mode=mode)
        timings[mode] = time.perf_counter() - start
    return timings


def _compare_outputs(legacy: dict[str, Any], native: dict[str, Any]) -> dict[str, Any]:
    """比較同一 ProductionBatch lifecycle 的位置與 output topology。"""

    legacy_positions = np.asarray(legacy["final_positions"], dtype=np.float64)
    native_positions = np.asarray(native["final_positions"], dtype=np.float64)
    if legacy_positions.shape != native_positions.shape:
        return {
            "pass": False,
            "max_abs_position_error_m": None,
            "output_topology_equal": False,
            "reason": "final_position_shape_mismatch",
        }
    max_error = float(np.max(np.abs(legacy_positions - native_positions)))
    legacy_observations = legacy["observations"]
    native_observations = native["observations"]
    observation_shapes_equal = len(legacy_observations) == len(native_observations)
    observation_position_error = 0.0
    observation_discrete_equal = False
    if observation_shapes_equal:
        observation_discrete_equal = all(
            (left[0], left[5]) == (right[0], right[5])
            for left, right in zip(legacy_observations, native_observations, strict=True)
        )
        if legacy_observations:
            legacy_observation_positions = np.asarray(
                [row[2:5] for row in legacy_observations], dtype=np.float64
            )
            native_observation_positions = np.asarray(
                [row[2:5] for row in native_observations], dtype=np.float64
            )
            observation_position_error = float(
                np.max(np.abs(legacy_observation_positions - native_observation_positions))
            )
    topology_equal = (
        legacy["final_time_age_status"] == native["final_time_age_status"]
        and observation_shapes_equal
        and observation_discrete_equal
        and legacy["events"] == native["events"]
        and legacy["sweeps_completed"] == native["sweeps_completed"]
    )
    return {
        "pass": bool(
            topology_equal
            and max_error <= MAX_POSITION_ERROR_M
            and observation_position_error <= MAX_POSITION_ERROR_M
        ),
        "max_abs_position_error_m": max_error,
        "max_abs_observation_position_error_m": observation_position_error,
        "position_tolerance_m": MAX_POSITION_ERROR_M,
        "output_topology_equal": bool(topology_equal),
        "legacy_output": _jsonable_output(legacy),
        "native_output": _jsonable_output(native),
    }


def _sha256(path: Path) -> str:
    """以串流方式保存 benchmark artifact 的 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_benchmark(
    *,
    baytrace_root: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    output_name: str = "q3_baytrace_production_benchmark.json",
    cases: tuple[tuple[int, int, int], ...] = DEFAULT_CASES,
    dt_seconds: float = q2.DEFAULT_DT_SECONDS,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """執行兩個小型 production case 並以 exclusive JSON 寫出 evidence。"""

    repo_root = Path(__file__).resolve().parents[1]
    baytrace_root = baytrace_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not output_name or Path(output_name).name != output_name:
        raise ValueError("output_name 必須是單一檔名，不可包含目錄")
    output = output_dir / output_name
    if output.exists():
        raise FileExistsError(f"拒絕覆寫既有 benchmark evidence：{output}")
    if not baytrace_root.is_dir():
        raise FileNotFoundError(f"BayTrace root 不存在：{baytrace_root}")
    if not cases:
        raise ValueError("cases 不可為空")
    for particles, steps, repeats in cases:
        if min(particles, steps, repeats) < 1:
            raise ValueError("particles/steps/repeats 必須是正整數")
    if dt_seconds not in (0.25, 1.0):
        raise ValueError("dt_seconds 只允許 Q2 synthetic case 支援的 0.25 或 1.0")

    cache_dir = (cache_dir or output_dir / ".q3-baytrace-cache").resolve()
    bridge, kernels = q2._load_bridge(baytrace_root, cache_dir)
    all_cases: list[dict[str, Any]] = []
    for particles, steps, repeats in cases:
        case = q2._build_case(bridge, kernels, particles, dt_seconds)
        warmup = _warmup(case, steps=steps, kernels=kernels)
        measurements: dict[str, list[dict[str, Any]]] = {"legacy": [], "native": []}
        comparisons: list[dict[str, Any]] = []
        for repeat in range(repeats):
            legacy_run = _run_once(case, steps=steps, kernels=kernels, mode="legacy")
            native_run = _run_once(case, steps=steps, kernels=kernels, mode="native")
            legacy_output = _output_snapshot(legacy_run.batch)
            native_output = _output_snapshot(native_run.batch)
            comparisons.append(_compare_outputs(legacy_output, native_output))
            measurements["legacy"].append(
                {
                    "repeat": repeat,
                    "setup_seconds": legacy_run.setup_seconds,
                    "advance_seconds": legacy_run.advance_seconds,
                    "total_seconds": legacy_run.setup_seconds + legacy_run.advance_seconds,
                    "output": _jsonable_output(legacy_output),
                }
            )
            measurements["native"].append(
                {
                    "repeat": repeat,
                    "setup_seconds": native_run.setup_seconds,
                    "advance_seconds": native_run.advance_seconds,
                    "total_seconds": native_run.setup_seconds + native_run.advance_seconds,
                    "output": _jsonable_output(native_output),
                    "resolver": native_run.resolver_report,
                    "batcher": native_run.batcher_report,
                }
            )
        for mode in ("legacy", "native"):
            advance_values = [item["advance_seconds"] for item in measurements[mode]]
            measurements[mode + "_median"] = {
                "advance_seconds": float(np.median(advance_values)),
                "setup_seconds": float(
                    np.median([item["setup_seconds"] for item in measurements[mode]])
                ),
                "total_seconds": float(
                    np.median([item["total_seconds"] for item in measurements[mode]])
                ),
            }
        native_reports = [item["resolver"] for item in measurements["native"]]
        batcher_reports = [item["batcher"] for item in measurements["native"]]
        all_cases.append(
            {
                "particles": particles,
                "steps": steps,
                "repeats": repeats,
                "warmup_seconds": warmup,
                "measurements": measurements,
                "comparisons": comparisons,
                "comparison_pass": all(item["pass"] for item in comparisons),
                "native_evidence": {
                    "frame_builds": [int(item["frame_builds"]) for item in native_reports],
                    "frame_cache_hits": [int(item["cache_hits"]) for item in native_reports],
                    "fallback_counts": [int(item["fallback_count"]) for item in native_reports],
                    "fallback_reasons": [item["fallback_reasons"] for item in native_reports],
                    "native_particle_steps": [
                        int(item["native_particle_steps"]) for item in batcher_reports
                    ],
                    "native_batch_calls": [
                        int(item["native_batch_calls"]) for item in batcher_reports
                    ],
                    "max_group_size": [int(item["max_group_size"]) for item in batcher_reports],
                },
            }
        )

    provenance = kernels.provenance
    tracked_sources = (
        "src/lagrangian_backtracking/advection.py",
        "src/lagrangian_backtracking/baytrace_batch_advection.py",
        "src/lagrangian_backtracking/baytrace_native_cpu.py",
        "src/lagrangian_backtracking/baytrace_ocm_adapter.py",
        "src/lagrangian_backtracking/production.py",
        "src/lagrangian_backtracking/forcing.py",
        "src/lagrangian_backtracking/forcing_window.py",
        "src/lagrangian_backtracking/engine.py",
        "src/lagrangian_backtracking/diffusion.py",
        "src/lagrangian_backtracking/boundaries.py",
        "scripts/benchmark_baytrace_native_cpu.py",
    )
    source_hashes = {
        relative: q2._sha256(repo_root / relative)
        for relative in tracked_sources
        if (repo_root / relative).is_file()
    }
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "source": {
            "lbt": q2._git_info(repo_root),
            "baytrace_root": str(provenance.source_root),
            "baytrace_dependencies_sha256": dict(provenance.dependency_sha256),
            "q3_script": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__).resolve())},
            "lbt_sources_sha256": source_hashes,
        },
        "environment": q2._environment(),
        "config": {
            "cases_particles_steps_repeats": [list(item) for item in cases],
            "dt_seconds": dt_seconds,
            "settling_velocity_mps": SETTLING_VELOCITY_MPS,
            "diffusion_m2ps": [DIFFUSION_M2PS] * 3,
            "same_production_batch_lifecycle": True,
            "native_path": (
                "ManagedForcingProvider -> OCMPreparedFrameResolver -> "
                "NativeBayTraceBatchAdvector"
            ),
            "legacy_path": "ManagedForcingProvider -> ProductionBatch legacy execute_advection_task",
            "frame_build_cost_included_in_native_advance": True,
            "raw_kernel_comparison": False,
        },
        "cases": all_cases,
        "limitations": [
            "synthetic OCM month only; real OCM/NWW products have not been tested here",
            "Stokes/wave adapter remains explicit stokes_adapter_pending fallback",
            "mixed LBT lifecycle retains step_start_reference, boundary and diffusion ownership",
            "not a full BayTrace tracker or SERVER throughput claim",
        ],
    }
    report["comparison_pass"] = bool(all(item["comparison_pass"] for item in all_cases))
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        stream.write("\n")
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint；預設尺寸固定為使用者要求的兩個小型 case。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baytrace-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output-name",
        default="q3_baytrace_production_benchmark.json",
        help="輸出檔名；既有檔案拒絕覆寫",
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--dt-seconds", type=float, choices=(0.25, 1.0), default=q2.DEFAULT_DT_SECONDS)
    parser.add_argument("--skip-large-case", action="store_true")
    args = parser.parse_args(argv)
    cases = DEFAULT_CASES[:1] if args.skip_large_case else DEFAULT_CASES
    report = run_benchmark(
        baytrace_root=args.baytrace_root,
        output_dir=args.output_dir,
        output_name=args.output_name,
        cases=cases,
        dt_seconds=args.dt_seconds,
        cache_dir=args.cache_dir,
    )
    print(
        json.dumps(
            {
                "comparison_pass": report["comparison_pass"],
                "output_dir": str(args.output_dir.resolve()),
                "cases": [
                    {
                        "particles": item["particles"],
                        "frame_builds": item["native_evidence"]["frame_builds"],
                        "cache_hits": item["native_evidence"]["frame_cache_hits"],
                        "fallback_counts": item["native_evidence"]["fallback_counts"],
                    }
                    for item in report["cases"]
                ],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if report["comparison_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
