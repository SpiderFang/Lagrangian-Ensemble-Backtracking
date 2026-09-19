#!/usr/bin/env python3
"""Q2：BayTrace 原始 CPU passive-advection 與 LBT Numba 的合成 benchmark。

本工具只比較同一個可控三角網格、公尺座標、固定水深、平面 sigma 與靜態非均勻速度
場上的被動平流 RK4。BayTrace 路徑直接呼叫 PI 原始 compiled
``batch_interpolate_velocity``／``batch_move_rk4``；thin adapter 只呼叫 Q1 的
``step_native``；LBT 路徑重用既有 ``OCMNativeMonth.sample``、``rk4_step``、
``numba_ocm_v1`` 與 ``numba_cpu_v1``。不重寫任何數值公式、不啟動 full tracker、
不讀真實 forcing，也不把結果解讀成 SERVER throughput。

兩個 forcing time rows 的值完全相同，並固定 BayTrace ``trat=0.5``，所以共同 case 不會
因 alpha 變化造成假等價。time-varying case 下，BayTrace 原版 RK4 對所有 stage 使用
同一 trat，而 LBT 會依各 stage UTC 重新決定時間 alpha；差異是已知演算法語意，不是
本工具要修正的 bug。
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "q2_baytrace_native_cpu_benchmark_v1"
DEFAULT_PARTICLES = 128
DEFAULT_STEPS = 16
DEFAULT_REPEATS = 3
DEFAULT_DT_SECONDS = 0.25
TRAT = 0.5
TOLERANCE_M = 1.0e-9
DEPTH_M = 20.0
TRIANGLE_SIZE_M = 1_000_000.0
TIME_ROWS_NS = np.asarray([0, 1_000_000_000_000], dtype=np.int64)
INITIAL_TIME_NS = 500_000_000_000


@dataclass(slots=True)
class SyntheticVelocityProvider:
    """把 synthetic OCM month 轉成正式速度 provider 的最小 typed facade。

    ``OCMNativeMonth.sample`` 本身是 bound method，沒有保存「可重用步首樣本」的能力
    宣告，也不能由 ``HintTrackingVelocityProvider`` 取得 triangle hint。此 facade 只
    轉交同一個 month，不改速度、時間或 QC；它明示唯讀 sample 可安全重用，並保存實際
    sample 次數與收到的 hint，讓 benchmark 以 evidence 驗證優化確實生效，而不是把
    預期的 4 或 5 次呼叫寫死成報告數字。
    """

    month: Any
    sample_calls: int = 0
    received_triangle_hints: list[int | None] = field(default_factory=list)

    @property
    def step_start_sample_reuse_safe(self) -> bool:
        """宣告固定 synthetic month 在相同輸入下可重用步首樣本。"""

        return True

    def sample(
        self,
        x_m: float,
        y_m: float,
        z_m: float,
        time_utc_ns: int,
        *,
        triangle_hint: int | None = None,
    ) -> Any:
        """記錄 hint／呼叫次數後，逐字轉交 ``OCMNativeMonth.sample``。"""

        self.sample_calls += 1
        self.received_triangle_hints.append(triangle_hint)
        return self.month.sample(
            x_m,
            y_m,
            z_m,
            time_utc_ns,
            triangle_hint=triangle_hint,
        )

    def __call__(self, x_m: float, y_m: float, z_m: float, time_utc_ns: int) -> Any:
        """提供既有四參數 ``VelocityProvider`` 呼叫形式；無 hint 時採 ``None``。"""

        return self.sample(x_m, y_m, z_m, time_utc_ns, triangle_hint=None)


@dataclass(slots=True)
class Case:
    """保存同一組 BayTrace/LBT synthetic input；不保存大型或真實資料。"""

    bridge: Any
    kernels: Any
    grid: Any
    fields: Any
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    elem: np.ndarray
    status: np.ndarray
    month: Any
    dt_seconds: float

    @property
    def n(self) -> int:
        """回傳粒子數。"""

        return int(self.x.size)

    def new_particles(self) -> Any:
        """每個 repeat 以獨立初始 arrays 建立 Q1 SOA 粒子容器。"""

        return self.bridge.BayTraceParticleArrays.from_positions(
            self.x.copy(), self.y.copy(), self.z.copy(), self.elem.copy(), status=self.status.copy()
        )


def _sha256(path: Path) -> str:
    """計算來源檔案的串流 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_info(root: Path) -> dict[str, Any]:
    """唯讀保存目前 LBT checkout 的 commit 與 dirty 狀態。"""

    def git(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    # status 的第一欄可能以空白表示 unstaged clean；不能使用整體 ``strip``，否則
    # 第一個 modified path 會少掉首字母，破壞 dirty provenance 的可讀性。
    try:
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.rstrip("\n")
    except (OSError, subprocess.CalledProcessError):
        status = ""
    return {
        "root": str(root),
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(status),
        "dirty_paths": [line[3:] for line in status.splitlines()],
    }


def _environment() -> dict[str, Any]:
    """保存 Python、NumPy、Numba、CPU 與可能影響 thread 的環境欄位。"""

    import numba

    keys = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMBA_NUM_THREADS",
        "NUMBA_THREADING_LAYER",
    )
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "numba": numba.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "thread_environment": {key: os.environ.get(key) for key in keys},
    }


def _load_bridge(root: Path, cache_dir: Path) -> tuple[Any, Any]:
    """lazy import Q1 bridge，並由它受控載入 PI 原始 CPU dispatchers。"""

    try:
        bridge = importlib.import_module("lagrangian_backtracking.baytrace_native_cpu")
        kernels = bridge.load_original_cpu(root, cache_dir)
    except (ImportError, AttributeError) as error:
        raise RuntimeError("Q1 baytrace_native_cpu bridge 尚未可用") from error
    for name in ("batch_interpolate_velocity", "batch_move_rk4"):
        if not hasattr(kernels, name):
            raise RuntimeError(f"Q1 kernels 缺少 {name}")
    return bridge, kernels


def _build_case(bridge: Any, kernels: Any, n: int, dt_seconds: float) -> Case:
    """建立全程留在同一個三角形內的 stationary interior case。

    三個節點的 ``u/v/w`` 不同，因此速度是平面 P1 非均勻場；同一節點的三個垂向層
    值相同，只消除垂向場的額外變化，不把流場變成零或均勻常數。兩個時間 rows 的
    所有速度與 eta 完全相同，正反向都使用同一個 positive dt 與 ``ibf`` 符號。
    """

    if type(n) is not int or n < 1:
        raise ValueError("particles 必須是正整數")
    if dt_seconds not in (0.25, 1.0):
        raise ValueError("dt_seconds 只允許 0.25 或 1.0")

    from lagrangian_backtracking.forcing import OCMNativeMonth
    from lagrangian_backtracking.mesh import NativeMesh

    node_xy = np.asarray([[0.0, 0.0], [TRIANGLE_SIZE_M, 0.0], [0.0, TRIANGLE_SIZE_M]], dtype=np.float64)
    rng = np.random.default_rng(20260919)
    x = TRIANGLE_SIZE_M * (0.20 + 0.15 * rng.random(n))
    y = TRIANGLE_SIZE_M * (0.20 + 0.15 * rng.random(n))
    z = np.full(n, -5.0, dtype=np.float64)
    elem = np.zeros(n, dtype=np.int64)
    status = np.ones(n, dtype=np.int32)

    i34 = np.asarray([3], dtype=np.int64)
    elnode = np.asarray([[0, 1, 2, -1]], dtype=np.int64)
    ic3 = np.full((4, 1), -1, dtype=np.int64)
    nvrt = 3
    kbp = np.zeros(3, dtype=np.int64)
    kbe = np.zeros(1, dtype=np.int64)
    sigma = np.asarray([[-1.0] * 3, [-0.5] * 3, [0.0] * 3], dtype=np.float64)
    dp = np.full(3, DEPTH_M, dtype=np.float64)

    u = np.asarray([0.08, 0.14, 0.20], dtype=np.float64)
    v = np.asarray([0.06, 0.015, -0.03], dtype=np.float64)
    w = np.asarray([0.0010, 0.0015, 0.0020], dtype=np.float64)
    u_layer = np.tile(u, (nvrt, 1))
    v_layer = np.tile(v, (nvrt, 1))
    w_layer = np.tile(w, (nvrt, 1))

    grid = bridge.BayTraceGridArrays(
        i34=i34,
        elnode=elnode,
        grid_x=node_xy[:, 0],
        grid_y=node_xy[:, 1],
        ic3=ic3,
        nvrt=nvrt,
        kbp=kbp,
        kbe=kbe,
        sigma_lcl=sigma,
        dp=dp,
    )
    fields = bridge.BayTraceFieldPair(
        u1=u_layer,
        v1=v_layer,
        w1=w_layer,
        eta1=np.zeros(3),
        u2=u_layer.copy(),
        v2=v_layer.copy(),
        w2=w_layer.copy(),
    )

    mesh = NativeMesh(
        node_lon=np.zeros(3),
        node_lat=np.zeros(3),
        node_xy=node_xy,
        source_depth_m=dp,
        source_node_bottom_index=kbp,
        face_nodes_local=elnode,
        face_node_count=i34,
        source_face_global_index=np.asarray([0], dtype=np.int64),
        bin_size_m=TRIANGLE_SIZE_M,
    )
    hvel = np.zeros((2, 3, nvrt, 2), dtype=np.float64)
    # BayTrace 場陣列是 (layer, node)，OCMNativeMonth 則是 (time, node,
    # layer, component)；三個軸長恰好同為 3 時，錯誤轉置不會被 shape 檢查抓到，
    # 卻會把節點非均勻性誤放到垂向層。這裡明示逐節點／逐層轉置，保留共同場的物理語意。
    hvel[:, :, :, 0] = u_layer.T[None, :, :]
    hvel[:, :, :, 1] = v_layer.T[None, :, :]
    month = OCMNativeMonth(
        # Unix 0 與 1000 秒屬 1970-01；月份僅是 synthetic provenance 標籤，不代表真實 forcing。
        month_id="197001",
        mesh=mesh,
        time_utc_ns=TIME_ROWS_NS.copy(),
        hvel=hvel,
        vertical_velocity=np.broadcast_to(w_layer.T[None, :, :], (2, 3, nvrt)).copy(),
        zcor=np.broadcast_to(np.asarray([-20.0, -10.0, 0.0])[None, None, :], (2, 3, nvrt)).copy(),
        elev=np.zeros((2, 3)),
        wetdry_elem=np.zeros((2, 1)),
        diffusivity=np.zeros((2, 3, nvrt)),
        use_numba_kernel=True,
    )
    return Case(bridge, kernels, grid, fields, x, y, z, elem, status, month, float(dt_seconds))


def _advance_raw(case: Case, particles: Any, ibf: int, steps: int) -> None:
    """直接呼叫原始 interpolate + move；每步本地重取 k1 並計入 x0 copies。"""

    grid, fields = case.grid, case.fields
    for _ in range(steps):
        x0, y0, z0 = particles.x.copy(), particles.y.copy(), particles.z.copy()
        case.kernels.batch_interpolate_velocity(
            particles.x,
            particles.y,
            particles.z,
            particles.elem,
            particles.status,
            grid.i34,
            grid.elnode,
            grid.grid_x,
            grid.grid_y,
            grid.nvrt,
            grid.kbp,
            grid.kbe,
            grid.sigma_lcl,
            grid.dp,
            fields.u1,
            fields.v1,
            fields.w1,
            fields.eta1,
            fields.u2,
            fields.v2,
            fields.w2,
            TRAT,
            0.0,
            particles.u,
            particles.v,
            particles.w,
        )
        case.kernels.batch_move_rk4(
            particles.x,
            particles.y,
            particles.z,
            particles.u,
            particles.v,
            particles.w,
            particles.x_prev,
            particles.y_prev,
            particles.elem,
            particles.status,
            x0,
            y0,
            z0,
            grid.i34,
            grid.elnode,
            grid.grid_x,
            grid.grid_y,
            grid.ic3,
            grid.nvrt,
            grid.kbp,
            grid.kbe,
            grid.sigma_lcl,
            grid.dp,
            fields.u1,
            fields.v1,
            fields.w1,
            fields.eta1,
            fields.u2,
            fields.v2,
            fields.w2,
            ibf,
            case.dt_seconds,
            TRAT,
        )


def _advance_adapter(case: Case, particles: Any, ibf: int, steps: int) -> None:
    """只呼叫 Q1 ``step_native``；其內部保存等價的 x0/y0/z0 copies。"""

    for _ in range(steps):
        case.bridge.step_native(
            case.kernels,
            case.grid,
            case.fields,
            particles,
            dt_seconds=case.dt_seconds,
            trat=TRAT,
            ibf=ibf,
        )


def _new_lbt_states(case: Case) -> list[Any]:
    """建立 LBT direct RK4 的 immutable ParticleState 清單。"""

    from lagrangian_backtracking.models import ParticleState

    return [
        ParticleState(
            particle_id=f"q2-{i:08d}",
            scenario_id=f"q2-{i:08d}",
            member_id=0,
            study_site_id="synthetic",
            analysis_region_id="synthetic",
            receptor_id=f"r-{i:08d}",
            x_m=float(x),
            y_m=float(y),
            z_m=float(z),
            time_utc_ns=INITIAL_TIME_NS,
        )
        for i, (x, y, z) in enumerate(zip(case.x, case.y, case.z, strict=True))
    ]


def _new_lbt_providers(case: Case) -> tuple[list[SyntheticVelocityProvider], list[Any]]:
    """每顆核心粒子建立一個 typed facade 與既有 hint wrapper，避免 hint 互相污染。"""

    from lagrangian_backtracking.production import HintTrackingVelocityProvider

    sources: list[SyntheticVelocityProvider] = []
    providers: list[Any] = []
    for _ in range(case.n):
        source = SyntheticVelocityProvider(case.month)
        sources.append(source)
        providers.append(HintTrackingVelocityProvider(source, triangle_hint=0))
    return sources, providers


def _advance_lbt(
    case: Case,
    states: list[Any],
    providers: list[Any],
    ibf: int,
    steps: int,
) -> None:
    """以既有 hint wrapper 執行 LBT RK4，並顯式重用每步步首 k1 sample。"""

    from lagrangian_backtracking.accelerated import PHYSICS_KERNEL_BACKEND_NUMBA_CPU_V1
    from lagrangian_backtracking.integrators import rk4_step

    for _ in range(steps):
        for i, state in enumerate(states):
            provider = providers[i]
            step_start_sample = provider.sample(
                state.x_m,
                state.y_m,
                state.z_m,
                state.time_utc_ns,
            )
            states[i] = rk4_step(
                state,
                dt_seconds=ibf * case.dt_seconds,
                velocity=provider,
                step_start_sample=step_start_sample,
                physics_kernel_backend=PHYSICS_KERNEL_BACKEND_NUMBA_CPU_V1,
            )


def _production_batch(case: Case, steps: int) -> tuple[Any, tuple[SyntheticVelocityProvider, ...]]:
    """建立 ProductionBatch；其額外功能成本另列，不納入 BayTrace 等量。"""

    from shapely.geometry import box

    from lagrangian_backtracking.boundaries import BoundaryGeometry
    from lagrangian_backtracking.diffusion import DiffusionCoefficients
    from lagrangian_backtracking.engine import EngineSettings
    from lagrangian_backtracking.models import ParticleState
    from lagrangian_backtracking.production import HintTrackingVelocityProvider, ProductionBatch
    from lagrangian_backtracking.runner import ReferenceParticleRequest, plan_scenario_shards
    from lagrangian_backtracking.scenarios import Scenario

    provider_sources: dict[int, SyntheticVelocityProvider] = {}
    scenarios = tuple(
        Scenario(
            scenario_id=f"q2-{i:08d}",
            study_site_id="synthetic",
            analysis_region_id="synthetic",
            material_id="passive",
            receptor_id=f"r-{i:08d}",
            arrival_time_id="arrival-0",
            settling_velocity_mps=0.0,
            arrival_time_utc_ns=INITIAL_TIME_NS,
            design_version=SCHEMA,
        )
        for i in range(case.n)
    )
    shard = plan_scenario_shards(
        scenarios,
        members_per_scenario=1,
        shard_scenario_count=case.n,
        experiment_case_id="q2-production-lifecycle",
    )[0]
    boundary = BoundaryGeometry(
        own_local_domain=box(-1e9, -1e9, 1e9, 1e9),
        flow_domain=box(-1e9, -1e9, 1e9, 1e9),
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

    def request_factory(unit: Any) -> ReferenceParticleRequest:
        """依 stable scenario suffix 建立與 BayTrace 相同的 initial position。"""

        i = int(unit.scenario.scenario_id.rsplit("-", 1)[1])
        source = SyntheticVelocityProvider(case.month)
        provider_sources[i] = source
        provider = HintTrackingVelocityProvider(source, triangle_hint=0)
        return ReferenceParticleRequest(
            initial_state=ParticleState(
                particle_id=unit.particle_id,
                scenario_id=unit.scenario.scenario_id,
                member_id=unit.member_id,
                study_site_id="synthetic",
                analysis_region_id="synthetic",
                receptor_id=unit.scenario.receptor_id,
                x_m=float(case.x[i]),
                y_m=float(case.y[i]),
                z_m=float(case.z[i]),
                time_utc_ns=INITIAL_TIME_NS,
            ),
            velocity=provider,
            boundaries=boundary,
            behavior_class="sinking",
            # K=0 保留 operator-split diffusion API/RNG lifecycle，但不改 passive position。
            diffusion=DiffusionCoefficients(0.0, 0.0, 0.0),
            settings=settings,
        )

    batch = ProductionBatch(
        shard,
        master_seed=20260919,
        request_factory=request_factory,
        active_chunk_size=case.n,
    )
    # ProductionBatch 內層仍會建立既有 HintTrackingVelocityProvider；把外層與 request
    # provider 都從同一個三角形開始，確保首個查詢也是真正的 hint-enabled 路徑。
    for runtime in batch.runtimes:
        runtime.velocity.set_triangle_hint(0)
    return batch, tuple(provider_sources[i] for i in range(case.n))


def _positions(path: str, state: Any) -> np.ndarray:
    """讀取四條路徑的共同 (N,3) final position。"""

    if path.startswith("baytrace"):
        return np.column_stack((state.x, state.y, state.z)).astype(np.float64, copy=True)
    if path == "lbt_numba_passive_rk4":
        return np.asarray([[item.x_m, item.y_m, item.z_m] for item in state], dtype=np.float64)
    return np.column_stack((state.batch.x_m, state.batch.y_m, state.batch.z_m)).astype(np.float64, copy=True)


def _margin(position: np.ndarray) -> float:
    """回傳三角形三條邊的最小內部距離尺度。"""

    return float(
        min(
            position[:, 0].min(),
            position[:, 1].min(),
            (TRIANGLE_SIZE_M - position[:, 0] - position[:, 1]).min(),
        )
    )


def _status(path: str, state: Any) -> dict[str, int]:
    """保存 BayTrace/LBT status count；不把 status 壓縮成單一成功旗標。"""

    values = (
        state.status
        if path.startswith("baytrace")
        else (["active"] * len(state) if path == "lbt_numba_passive_rk4" else state.batch.status_code)
    )
    unique, counts = np.unique(np.asarray(values), return_counts=True)
    return {
        str(value.item() if hasattr(value, "item") else value): int(count)
        for value, count in zip(unique, counts, strict=True)
    }


def _run_path(case: Case, path: str, ibf: int, steps: int, repeats: int) -> tuple[dict[str, Any], np.ndarray]:
    """每個 repeat 重置初始陣列，量測 steady wall；setup/JIT 不進 ratio。"""

    walls: list[float] = []
    setups: list[float] = []
    sample_call_counts: list[int] = []
    hint_call_counts: list[int] = []
    final: np.ndarray | None = None
    final_status: dict[str, int] | None = None
    for _ in range(repeats):
        setup_start = time.perf_counter()
        if path.startswith("baytrace"):
            state = case.new_particles()
            sample_sources: tuple[SyntheticVelocityProvider, ...] = ()
            providers: list[Any] = []
        elif path == "lbt_numba_passive_rk4":
            state = _new_lbt_states(case)
            sample_sources, providers = _new_lbt_providers(case)
        else:
            state, sample_sources = _production_batch(case, steps)
            providers = []
        setups.append(time.perf_counter() - setup_start)
        start = time.perf_counter()
        if path == "baytrace_native_raw":
            _advance_raw(case, state, ibf, steps)
        elif path == "baytrace_native_adapter":
            _advance_adapter(case, state, ibf, steps)
        elif path == "lbt_numba_passive_rk4":
            _advance_lbt(case, state, providers, ibf, steps)
        else:
            result = state.advance(steps)
            if result.sweeps_completed != steps or result.stepped_particle_count != case.n * steps:
                raise RuntimeError(f"ProductionBatch step count 不符：{result}")
        walls.append(time.perf_counter() - start)
        final, final_status = _positions(path, state), _status(path, state)
        if sample_sources:
            sample_call_counts.append(sum(source.sample_calls for source in sample_sources))
            hint_call_counts.append(
                sum(
                    sum(hint is not None for hint in source.received_triangle_hints)
                    for source in sample_sources
                )
            )

    assert final is not None and final_status is not None
    median = float(np.median(walls))
    metric = {
        "wall_seconds": {"repeats": walls, "median": median},
        "setup_seconds_median": float(np.median(setups)),
        "particle_steps_per_second": case.n * steps / median,
        "position_finite": bool(np.all(np.isfinite(final))),
        "status_counts": final_status,
        "common_interior_margin_m": _margin(final),
        "steady_excludes_setup_and_jit_warmup": True,
    }
    if not metric["position_finite"] or metric["common_interior_margin_m"] <= 0:
        raise RuntimeError(f"{path} 離開 common stationary interior case")
    if sample_call_counts:
        metric.update(
            {
                "velocity_sample_calls": {
                    "repeats": sample_call_counts,
                    "median": float(np.median(sample_call_counts)),
                },
                "velocity_sample_calls_per_particle_step": float(
                    np.median(sample_call_counts) / (case.n * steps)
                ),
                "triangle_hint_non_null_calls": {
                    "repeats": hint_call_counts,
                    "median": float(np.median(hint_call_counts)),
                },
                "hint_enabled_provider": True,
            }
        )
    return metric, final


def _warmup(case: Case, steps: int) -> dict[str, Any]:
    """先編譯所有會進入 steady 的 dispatcher，另列 warmup wall。"""

    def warm_lbt() -> None:
        """暖機 direct LBT provider、hint wrapper 與明示 k1 reuse 路徑。"""

        states = _new_lbt_states(case)
        _, providers = _new_lbt_providers(case)
        _advance_lbt(case, states, providers, -1, 1)

    calls = (
        ("baytrace_native_raw", lambda: _advance_raw(case, case.new_particles(), -1, 1)),
        ("baytrace_native_adapter", lambda: _advance_adapter(case, case.new_particles(), -1, 1)),
        ("lbt_numba_passive_rk4", warm_lbt),
        (
            "lbt_production_batch_lifecycle",
            lambda: _production_batch(case, steps)[0].advance(1),
        ),
    )
    seconds: dict[str, float] = {}
    for name, call in calls:
        start = time.perf_counter()
        call()
        seconds[name] = time.perf_counter() - start
    from lagrangian_backtracking.accelerated import warmup_numba_backend

    start = time.perf_counter()
    backend = warmup_numba_backend()
    seconds["lbt_warmup_numba_backend"] = time.perf_counter() - start
    return {"wall_seconds": seconds, "excluded_from_ratios": True, "lbt_backend": backend}


def _signatures(case: Case) -> dict[str, list[str]]:
    """保存原始 BayTrace 與 LBT Numba 已編譯 dispatcher signatures。"""

    from lagrangian_backtracking import accelerated

    return {
        "baytrace_batch_interpolate_velocity": [
            str(item) for item in case.kernels.batch_interpolate_velocity.signatures
        ],
        "baytrace_batch_move_rk4": [str(item) for item in case.kernels.batch_move_rk4.signatures],
        "lbt_interpolate_ocm_support_numba": [
            str(item) for item in accelerated.interpolate_ocm_support_numba.signatures
        ],
        "lbt_rk4_finalize_numba_kernel": [
            str(item) for item in accelerated.rk4_finalize_numba_kernel.signatures
        ],
    }


def _profile(case: Case, steps: int, path: Path) -> None:
    """只對暖機後的 ProductionBatch lifecycle 寫 profile，不回傳 instrumented wall。"""

    profile = cProfile.Profile()
    profile.enable()
    result = _production_batch(case, steps)[0].advance(steps)
    profile.disable()
    if result.sweeps_completed != steps:
        raise RuntimeError("profile ProductionBatch sweep count 不符")
    profile.dump_stats(str(path))


def run_benchmark(
    *,
    baytrace_root: Path,
    output: Path,
    particles: int = DEFAULT_PARTICLES,
    steps: int = DEFAULT_STEPS,
    repeats: int = DEFAULT_REPEATS,
    dt_seconds: float = DEFAULT_DT_SECONDS,
    cache_dir: Path | None = None,
    profile: bool = False,
) -> dict[str, Any]:
    """執行正反向 benchmark 並以 exclusive JSON 寫出新的 evidence。"""

    repo_root = Path(__file__).resolve().parents[1]
    baytrace_root, output = baytrace_root.resolve(), output.resolve()
    profile_path = output.with_suffix(output.suffix + ".production.cprofile") if profile else None
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or (profile_path is not None and profile_path.exists()):
        raise FileExistsError("拒絕覆蓋既有 benchmark/profile evidence；請指定新的 --output")
    if not baytrace_root.is_dir():
        raise FileNotFoundError(f"BayTrace root 不存在：{baytrace_root}")
    invalid_counts = (
        type(particles) is not int
        or particles < 1
        or type(steps) is not int
        or steps < 1
        or type(repeats) is not int
        or repeats < 1
    )
    if invalid_counts:
        raise ValueError("particles/steps/repeats 必須是正整數")
    cache_dir = (cache_dir or output.parent / f".{output.stem}.baytrace-cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    bridge, kernels = _load_bridge(baytrace_root, cache_dir)
    case = _build_case(bridge, kernels, particles, dt_seconds)
    print(f"warmup start: particles={particles}, steps={steps}, repeats={repeats}", flush=True)
    warmup = _warmup(case, steps)
    print(json.dumps({"warmup_seconds": warmup["wall_seconds"]}, ensure_ascii=False), flush=True)

    core_paths = ("baytrace_native_raw", "baytrace_native_adapter", "lbt_numba_passive_rk4")
    production_path = "lbt_production_batch_lifecycle"
    # ProductionBatch 是逆向引擎；只在 backward 實際量測，避免把 lifecycle
    # 成本誤列為 forward 方向或與核心 passive-advection 等量。
    paths_by_direction = {"forward": core_paths, "backward": core_paths + (production_path,)}
    measurements: dict[str, Any] = {}
    finals: dict[str, dict[str, np.ndarray]] = {}
    for direction, ibf in (("forward", 1), ("backward", -1)):
        measurements[direction], finals[direction] = {}, {}
        paths = paths_by_direction[direction]
        for path in paths:
            measurements[direction][path], finals[direction][path] = _run_path(
                case, path, ibf, steps, repeats
            )
        wall_medians = {path: measurements[direction][path]["wall_seconds"]["median"] for path in paths}
        print(
            json.dumps({"direction": direction, "wall_medians": wall_medians}, ensure_ascii=False),
            flush=True,
        )

    errors: dict[str, Any] = {}
    ratios: dict[str, Any] = {}
    for direction in measurements:
        raw = finals[direction]["baytrace_native_raw"]
        errors[direction], ratios[direction] = {}, {}
        baseline = measurements[direction]["baytrace_native_raw"]["wall_seconds"]["median"]
        for path in paths_by_direction[direction]:
            median = measurements[direction][path]["wall_seconds"]["median"]
            ratios[direction][f"raw_over_{path}"] = baseline / median
            if path != production_path:
                error = float(np.max(np.abs(raw - finals[direction][path])))
                errors[direction][f"raw_vs_{path}"] = {
                    "max_abs_position_error_m": error,
                    "pass": error <= TOLERANCE_M,
                }

    if profile_path is not None:
        _profile(case, steps, profile_path)

    provenance = kernels.provenance
    bridge_path = Path(bridge.__file__).resolve()
    lbt_rel = (
        "src/lagrangian_backtracking/accelerated.py",
        "src/lagrangian_backtracking/boundaries.py",
        "src/lagrangian_backtracking/diffusion.py",
        "src/lagrangian_backtracking/engine.py",
        "src/lagrangian_backtracking/forcing.py",
        "src/lagrangian_backtracking/integrators.py",
        "src/lagrangian_backtracking/mesh.py",
        "src/lagrangian_backtracking/models.py",
        "src/lagrangian_backtracking/production.py",
        "src/lagrangian_backtracking/runner.py",
    )
    lbt_hashes = {item: _sha256(repo_root / item) for item in lbt_rel if (repo_root / item).is_file()}
    report = {
        "schema": SCHEMA,
        "source": {
            "lbt": _git_info(repo_root),
            "baytrace_root": str(provenance.source_root),
            "benchmark_tool": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256(Path(__file__).resolve()),
            },
            "bridge": {"path": str(bridge_path), "sha256": _sha256(bridge_path)},
            "baytrace_dependencies": dict(provenance.dependency_sha256),
            "lbt_sources": lbt_hashes,
        },
        "environment": _environment(),
        "config": {
            "particles": particles,
            "steps": steps,
            "repeats": repeats,
            "dt_seconds": dt_seconds,
            "directions": {"forward": 1, "backward": -1},
            "trat": TRAT,
            "units": {"position": "m", "velocity": "m/s", "time_step": "s"},
            "numerical_pass_tolerance_m": TOLERANCE_M,
            "k1_evaluations_per_path": particles * steps,
            "rk4_stage_evaluations_core_paths": 4 * particles * steps,
            "raw_x0_copies_in_timing": True,
            "adapter_x0_copies_in_timing": "inside step_native",
            "initialization_and_jit_warmup_excluded_from_steady": True,
        },
        "synthetic_case": {
            "type": "common_stationary_interior_triangle",
            "triangle_size_m": TRIANGLE_SIZE_M,
            "fixed_depth_m": DEPTH_M,
            "zcor_m": [-20.0, -10.0, 0.0],
            "time_rows": 2,
            "time_rows_identical": True,
            "flow": "static non-uniform nodal P1 u/v/w; same values at both rows and all layers",
        },
        "warmup": warmup,
        "numba_signatures": _signatures(case),
        "measurements": measurements,
        "max_abs_position_errors": errors,
        "speed_ratios_raw_over_path": ratios,
        "production_batch_lifecycle": {
            "separate_measurement": True,
            "included": [
                "ProductionBatch construction/request identity and initial observation (setup median)",
                "active compaction/scatter, alignment/QC/boundary checks, observations and event container",
                "numba_cpu_v1 RK4 with OCM sample",
                "operator-split diffusion Kx=Ky=Kz=0 m2/s; RNG path called, net displacement zero",
            ],
            "excluded": [
                "checkpoint/I-O",
                "trajectory/report/aggregation",
                "NWW3/Stokes",
                "SERVER scheduling/throughput",
            ],
            "actual_direction": "backward",
            "not_measured_as_forward": True,
            "core_comparison_equivalent": False,
            "core_velocity_calls_per_particle_step_contract": 4,
            "velocity_sample_counts_measured_in_each_path": True,
            "production_extra_sample_policy": "report observed count; do not assume four or five",
        },
        "time_varying_limitation": {
            "status": "not_run_as_equivalence_case",
            "baytrace": "original batch RK4 uses one trat for all stages",
            "lbt": "OCMNativeMonth.sample can use stage-specific UTC alpha",
            "interpretation": "time-varying differences are declared algorithm semantics, not a bug",
        },
        "profile": {
            "enabled": profile_path is not None,
            "path": str(profile_path) if profile_path else None,
            "scope": "warmed ProductionBatch only",
            "instrumented_wall_not_used_in_ratios": True,
        },
        "limitations": [
            "common stationary interior synthetic case only; not full tracker or formal science evidence",
            "BayTrace direct path excludes full-tracker SEARCH/clamp/output/event lifecycle; "
            "original intermediate lookup inside batch_move_rk4 remains",
            "LBT direct path excludes boundary/events/observations/diffusion; "
            "ProductionBatch lifecycle is separate and not BayTrace-equivalent",
            "no unit-test wall-time hard threshold; local timing cannot be generalized to SERVER throughput",
        ],
    }
    comparison_checks = [check for direction_checks in errors.values() for check in direction_checks.values()]
    report["comparison"] = {
        "pass": bool(comparison_checks) and all(check["pass"] for check in comparison_checks),
        "tolerance_m": TOLERANCE_M,
        "checked_core_paths_only": True,
        "failed_checks": sum(not check["pass"] for check in comparison_checks),
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baytrace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--particles", type=int, default=DEFAULT_PARTICLES)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--dt-seconds", type=float, choices=(0.25, 1.0), default=DEFAULT_DT_SECONDS)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args(argv)
    report = run_benchmark(
        baytrace_root=args.baytrace_root,
        output=args.output,
        particles=args.particles,
        steps=args.steps,
        repeats=args.repeats,
        dt_seconds=args.dt_seconds,
        cache_dir=args.cache_dir,
        profile=args.profile,
    )
    if not report["comparison"]["pass"]:
        print(
            json.dumps(
                {
                    "comparison_pass": False,
                    "failed_checks": report["comparison"]["failed_checks"],
                    "output": str(args.output.resolve()),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
