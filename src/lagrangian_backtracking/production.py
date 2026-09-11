"""CPU/NumPy 分塊粒子執行器與可重啟批次邏輯。

本模組把既有 reference ``run_particle`` 的單步核心套在多條粒子軌跡外面：每條軌跡仍
保有獨立的 ``ParticleExecutionState``、PCG64DXSM 亂數產生器與 native mesh 三角形提示，
批次本身只負責固定順序、作用中粒子壓縮／分塊／分散回寫與 identity gate。這個 Phase 2
實作是 CPU/NumPy orchestration，不是假裝已經有完整的 Numba physics kernel；所有 RK4、
Brownian、邊界與品質檢查仍直接呼叫同一份 reference engine。
``ReferenceParticleRequest.diffusion`` 可攜帶固定係數或空間擴散 provider；本模組只將
該模型逐粒子傳給 engine，不在 batch orchestration 預先取樣，也不為它新增 checkpoint
或輸出欄位。
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .batch_state import PARTICLE_STATUS_TO_CODE, ParticleBatch
from .engine import (
    ParticleAdvanceResult,
    ParticleExecutionState,
    ParticleResult,
    advance_particle_once,
    finalize_particle_execution,
    initialize_particle_execution,
)
from .integrators import VelocityProvider, supports_step_start_sample_reuse
from .models import ParticleState, ParticleStatus, VelocitySample
from .runner import (
    ReferenceParticleRequest,
    RunUnit,
    ScenarioShard,
    iter_run_units,
)

if TYPE_CHECKING:
    from .checkpoint import CheckpointBinding, ExecutionCheckpoint


def _supports_triangle_hint(method: Callable[..., object]) -> bool:
    """判斷 provider 的 sample 方法是否明確接受 triangle hint 關鍵字。

    以簽名檢查區分「不支援 hint」和「方法內部真的發生 TypeError」，避免用寬鬆的
    try/except 把 forcing 內部錯誤誤當成介面相容性問題。部分 C-extension callable 可能
    沒有可讀簽名，這種情況採保守的四參數路徑，不會猜測額外參數。
    """

    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return False
    parameters = signature.parameters.values()
    return "triangle_hint" in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters
    )


class HintTrackingVelocityProvider:
    """替粒子保存最後成功取得的 native mesh triangle hint。

    若底層物理 provider 有 ``sample(x, y, z, time, triangle_hint=...)``，每一個 RK4 stage
    都會收到目前提示；若回傳的 ``VelocitySample.triangle_id`` 有值，便立即更新提示，讓
    下一個 stage 優先從相鄰三角形搜尋。普通四參數 callable 維持原呼叫方式，且 hint
    不會參與速度、擴散或邊界計算，因此只能影響搜尋效率，不能改變物理結果。
    """

    def __init__(self, provider: VelocityProvider, *, triangle_hint: int | None = None) -> None:
        """建立 provider wrapper；``None`` 與 ``-1`` 都代表尚未有三角形提示。"""

        self.provider = provider
        self.triangle_hint = self._normalize_hint(triangle_hint)
        sample_method = getattr(provider, "sample", None)
        self._sample_method = sample_method if callable(sample_method) else None
        self._sample_accepts_hint = (
            self._sample_method is not None and _supports_triangle_hint(self._sample_method)
        )

    @staticmethod
    def _normalize_hint(value: int | None) -> int | None:
        """把批次使用的 ``-1`` 空值轉成 provider 使用的 ``None``。"""

        if value is None or value == -1:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError("triangle_hint 必須是 None、-1 或非負整數")
        normalized = int(value)
        if normalized < 0:
            raise ValueError("triangle_hint 不可小於 -1")
        return normalized

    def set_triangle_hint(self, value: int | None) -> None:
        """在 restore 或下一個 sweep 前設定目前粒子的 mesh hint。"""

        self.triangle_hint = self._normalize_hint(value)

    @property
    def step_start_sample_reuse_safe(self) -> bool:
        """回傳底層速度取樣器是否明示允許步首樣本重用。

        包裝器本身會更新三角形搜尋提示，但提示只影響原生網格搜尋順序，不是物理結果的
        來源。只有底層速度取樣器實作 ``StepStartSampleReuseProvider`` 且明示回傳 ``True``
        時，才把這項能力傳給粒子引擎；普通四參數可呼叫物件仍保留每個階段的原始查詢次數
        與副作用。
        """

        # 共用 integrators 的保守檢查；連同能力屬性的屬性讀取例外，都回到原始查詢路徑。
        return supports_step_start_sample_reuse(self.provider)

    def sample(
        self,
        x_m: float,
        y_m: float,
        z_m: float,
        time_utc_ns: int,
    ) -> VelocitySample:
        """依底層 provider 能力取樣並以回傳 triangle ID 更新 hint。"""

        if self._sample_method is not None:
            if self._sample_accepts_hint:
                result = self._sample_method(
                    x_m,
                    y_m,
                    z_m,
                    time_utc_ns,
                    triangle_hint=self.triangle_hint,
                )
            else:
                result = self._sample_method(x_m, y_m, z_m, time_utc_ns)
        else:
            result = self.provider(x_m, y_m, z_m, time_utc_ns)
        if not isinstance(result, VelocitySample):
            raise TypeError("velocity provider 必須回傳 VelocitySample")
        if result.triangle_id is not None:
            self.triangle_hint = self._normalize_hint(result.triangle_id)
        return result

    def __call__(self, x_m: float, y_m: float, z_m: float, time_utc_ns: int) -> VelocitySample:
        """提供既有四參數 ``VelocityProvider`` 介面給 RK4 核心。"""

        return self.sample(x_m, y_m, z_m, time_utc_ns)


@dataclass(slots=True)
class ProductionParticleRuntime:
    """一個 RunUnit 的執行資源與可恢復狀態。

    ``rng`` 只供這條粒子使用，不能由多條 runtime 共用；``triangle_hint`` 是批次欄位的
    mirror，真正取樣時由 ``velocity`` 持有並在每個 stage 後同步。``request`` 保留 forcing、
    邊界及設定物件的原始參照，這些大型或外部資源不寫入 checkpoint，restore 時會由同一
    ``request_factory`` 重新建立並核對 RunUnit identity。
    """

    unit: RunUnit
    request: ReferenceParticleRequest
    execution: ParticleExecutionState
    rng: np.random.Generator
    velocity: HintTrackingVelocityProvider
    triangle_hint: int = -1

    def sync_triangle_hint(self) -> None:
        """將 wrapper 的最新提示同步到可序列化的 runtime 欄位。"""

        self.triangle_hint = self.velocity.triangle_hint if self.velocity.triangle_hint is not None else -1


@dataclass(frozen=True, slots=True)
class ProductionAdvanceResult:
    """一或多個 sweep 後的批次進度，布林值代表是否全數終止。"""

    terminal: bool
    stepped_particle_count: int
    sweeps_completed: int
    active_particle_count: int

    def __bool__(self) -> bool:
        """讓呼叫端可直接以 ``if batch.advance():`` 判斷批次是否完成。"""

        return self.terminal


def _state_identity(state: ParticleState) -> tuple[object, ...]:
    """提取運行期間不會改變、需要和 RunUnit 逐欄核對的粒子身分。"""

    return (
        state.particle_id,
        state.scenario_id,
        state.member_id,
        state.study_site_id,
        state.analysis_region_id,
        state.receptor_id,
    )


def _unit_identity(unit: RunUnit) -> tuple[object, ...]:
    """提取固定 RunUnit 身分，避免 request factory 偷換站點或 member。"""

    return (
        unit.particle_id,
        unit.scenario.scenario_id,
        unit.member_id,
        unit.scenario.study_site_id,
        unit.scenario.analysis_region_id,
        unit.scenario.receptor_id,
    )


def _validate_request(unit: RunUnit, request: ReferenceParticleRequest) -> None:
    """在建立 runtime 前驗證 request 的完整身分與初始狀態。"""

    actual = (*_state_identity(request.initial_state), request.initial_state.time_utc_ns)
    expected = (*_unit_identity(unit), unit.scenario.arrival_time_utc_ns)
    if actual != expected:
        raise ValueError(f"request initial_state 與 run unit 不一致：actual={actual}, expected={expected}")
    if request.initial_state.status != ParticleStatus.ACTIVE:
        raise ValueError("request initial_state 必須是 ACTIVE")


class ProductionBatch:
    """依固定 RunUnit 順序執行、可中途 checkpoint 的 CPU/NumPy 粒子批次。

    建構時依 ``iter_run_units`` 產生固定順序，``request_factory`` 對每個 unit 恰好呼叫
    一次。每個 sweep 先從 ``ParticleBatch`` 壓縮 ACTIVE 粒子，依 source order 對每一條
    runtime 最多呼叫一次 ``advance_particle_once``，再把位置、年齡、時間、狀態、local exit
    與 triangle hint 分散回原始索引；因此 active 粒子提早停止不會造成 identity 錯位。
    ``active_chunk_size`` 只改變一次處理的切片大小，不改變 source order、每條粒子的 RNG
    或物理呼叫次數。``request.diffusion`` 仍原樣交給單步 engine；空間 provider 的
    步首取樣次數與 triangle hint 語意由 engine 集中保證，避免 batch 層複製一份不同的
    擴散狀態。
    """

    def __init__(
        self,
        shard: ScenarioShard,
        *,
        master_seed: int,
        request_factory: Callable[[RunUnit], ReferenceParticleRequest],
        active_chunk_size: int | None = None,
    ) -> None:
        """建立非空批次並一次完成所有 request 建立與初始觀測。"""

        if isinstance(master_seed, bool) or not isinstance(master_seed, int):
            raise TypeError("master_seed 必須是非負整數")
        if master_seed < 0:
            raise ValueError("master_seed 不可為負")
        if shard.particle_count < 1:
            raise ValueError("ProductionBatch 不允許空 shard")
        if (
            active_chunk_size is not None
            and (isinstance(active_chunk_size, bool) or not isinstance(active_chunk_size, int))
        ):
            raise TypeError("active_chunk_size 必須是正整數或 None")
        if active_chunk_size is not None and active_chunk_size < 1:
            raise ValueError("active_chunk_size 必須為正整數或 None")
        self.shard = shard
        self.master_seed = master_seed
        self.active_chunk_size = active_chunk_size
        self.units = tuple(iter_run_units(shard, master_seed=master_seed))
        if len(self.units) != shard.particle_count:
            raise RuntimeError("RunUnit 數量與 scenario×M 契約不符")
        self.runtimes: list[ProductionParticleRuntime] = []
        for unit in self.units:
            request = request_factory(unit)
            _validate_request(unit, request)
            execution = initialize_particle_execution(request.initial_state, request.settings)
            velocity = HintTrackingVelocityProvider(request.velocity)
            self.runtimes.append(
                ProductionParticleRuntime(
                    unit=unit,
                    request=request,
                    execution=execution,
                    rng=np.random.Generator(np.random.PCG64DXSM(unit.seed)),
                    velocity=velocity,
                )
            )
        self.particle_batch = ParticleBatch.from_particle_states(
            [runtime.execution.state for runtime in self.runtimes],
            triangle_hints=[runtime.triangle_hint for runtime in self.runtimes],
        )
        self.sweep_count = 0
        self.checkpoint_sequence = 0
        self._validate_alignment()

    @property
    def batch(self) -> ParticleBatch:
        """回傳目前 SoA 狀態；保留 ``particle_batch`` 作為明確名稱。"""

        return self.particle_batch

    @property
    def active_count(self) -> int:
        """回傳目前仍可前進的粒子數。"""

        return int(self.particle_batch.active_indices().size)

    @property
    def terminal(self) -> bool:
        """回傳批次是否已沒有 ACTIVE 粒子。"""

        return self.active_count == 0

    @property
    def execution_states(self) -> tuple[ParticleExecutionState, ...]:
        """依固定 RunUnit 順序回傳目前 execution 物件，供監看與 snapshot 使用。"""

        return tuple(runtime.execution for runtime in self.runtimes)

    def _validate_alignment(self) -> None:
        """逐粒子核對 SoA 與 execution，任何身分／狀態／動態欄位錯位立即拒絕。

        這個檢查故意放在每次 sweep 前後；雖然增加少量 Python orchestration 成本，卻能在
        還沒有把錯誤 scatter 到輸出前指出粒子、member 或狀態來源，避免長時間 SERVER run
        產生不可追溯的軌跡交叉。真正的物理 kernel 尚未在本 Phase 由 Numba 實作。
        """

        self.particle_batch.validate()
        if len(self.runtimes) != len(self.particle_batch):
            raise ValueError("ParticleBatch 與 runtime count 不一致")
        for index, runtime in enumerate(self.runtimes):
            expected_identity = _unit_identity(runtime.unit)
            state_identity = _state_identity(runtime.execution.state)
            batch_identity = (
                self.particle_batch.particle_id[index],
                self.particle_batch.scenario_id[index],
                int(self.particle_batch.member_id[index]),
                self.particle_batch.study_site_id[index],
                self.particle_batch.analysis_region_id[index],
                self.particle_batch.receptor_id[index],
            )
            if state_identity != expected_identity or batch_identity != expected_identity:
                raise ValueError(
                    f"ParticleBatch/runtime identity 不一致：index={index}, "
                    f"state={state_identity}, batch={batch_identity}, expected={expected_identity}"
                )
            expected_status = PARTICLE_STATUS_TO_CODE[runtime.execution.state.status]
            if int(self.particle_batch.status_code[index]) != expected_status:
                raise ValueError(
                    f"ParticleBatch/runtime status 不一致：index={index}, "
                    f"batch={int(self.particle_batch.status_code[index])}, expected={expected_status}"
                )
            state = runtime.execution.state
            dynamic = (
                (self.particle_batch.x_m[index], state.x_m),
                (self.particle_batch.y_m[index], state.y_m),
                (self.particle_batch.z_m[index], state.z_m),
                (self.particle_batch.age_seconds[index], state.age_seconds),
                (self.particle_batch.time_utc_ns[index], state.time_utc_ns),
                (self.particle_batch.own_local_exit_recorded[index], state.own_local_exit_recorded),
                (self.particle_batch.triangle_hint[index], runtime.triangle_hint),
            )
            if any(actual != expected for actual, expected in dynamic):
                raise ValueError(f"ParticleBatch/runtime state 不一致：index={index}")

    @staticmethod
    def _write_state_to_batch(
        batch: ParticleBatch,
        index: int,
        state: ParticleState,
        triangle_hint: int,
    ) -> None:
        """把單一 execution 的動態結果寫入 compacted batch，不碰固定身分欄位。"""

        batch.x_m[index] = state.x_m
        batch.y_m[index] = state.y_m
        batch.z_m[index] = state.z_m
        batch.age_seconds[index] = state.age_seconds
        batch.time_utc_ns[index] = state.time_utc_ns
        batch.status_code[index] = PARTICLE_STATUS_TO_CODE[state.status]
        batch.own_local_exit_recorded[index] = state.own_local_exit_recorded
        batch.triangle_hint[index] = triangle_hint

    def _advance_one_sweep(self) -> ProductionAdvanceResult:
        """依 source order 讓每條 active runtime 最多前進一步。"""

        self._validate_alignment()
        compacted, source_indices = self.particle_batch.compact_active()
        if source_indices.size == 0:
            return ProductionAdvanceResult(True, 0, 0, 0)
        chunk_size = self.active_chunk_size or int(source_indices.size)
        stepped_count = 0
        for chunk_start in range(0, int(source_indices.size), chunk_size):
            chunk_stop = min(chunk_start + chunk_size, int(source_indices.size))
            chunk = compacted.slice_view(chunk_start, chunk_stop)
            for local_index, source_index_value in enumerate(source_indices[chunk_start:chunk_stop]):
                source_index = int(source_index_value)
                runtime = self.runtimes[source_index]
                compacted_index = local_index
                runtime.velocity.set_triangle_hint(runtime.triangle_hint)
                result: ParticleAdvanceResult = advance_particle_once(
                    runtime.execution,
                    velocity=runtime.velocity,
                    boundaries=runtime.request.boundaries,
                    behavior_class=runtime.request.behavior_class,
                    diffusion=runtime.request.diffusion,
                    settings=runtime.request.settings,
                    rng=runtime.rng,
                )
                runtime.sync_triangle_hint()
                self._write_state_to_batch(chunk, compacted_index, result.state, runtime.triangle_hint)
                stepped_count += int(result.stepped)
        self.particle_batch.scatter_dynamic_from(compacted, source_indices)
        self.sweep_count += 1
        self._validate_alignment()
        return ProductionAdvanceResult(
            terminal=self.terminal,
            stepped_particle_count=stepped_count,
            sweeps_completed=1,
            active_particle_count=self.active_count,
        )

    def advance(self, sweeps: int = 1) -> ProductionAdvanceResult:
        """執行指定數量 sweep；每個 active particle 在每個 sweep 最多前進一步。

        ``sweeps`` 只控制外層呼叫次數，不能改變粒子排序或 RNG 分配。若批次在要求的
        sweep 數之前全數終止，會立即回傳；因此 complete 與固定次數 checkpoint 都可共用
        這個介面。
        """

        if isinstance(sweeps, bool) or not isinstance(sweeps, int):
            raise TypeError("sweeps 必須是正整數")
        if sweeps < 1:
            raise ValueError("sweeps 必須為正整數")
        stepped_count = 0
        completed = 0
        for _ in range(sweeps):
            result = self._advance_one_sweep()
            stepped_count += result.stepped_particle_count
            completed += result.sweeps_completed
            if result.terminal:
                break
        return ProductionAdvanceResult(
            terminal=self.terminal,
            stepped_particle_count=stepped_count,
            sweeps_completed=completed,
            active_particle_count=self.active_count,
        )

    def complete(self) -> list[ParticleResult]:
        """持續 sweep 到所有粒子終止，並依固定 RunUnit 順序回傳結果。"""

        while not self.terminal:
            self.advance()
        return self.results()

    def results(self) -> list[ParticleResult]:
        """回傳固定順序的完成結果；partial batch 不可誤當成正式輸出。"""

        if not self.terminal:
            raise RuntimeError("ProductionBatch 尚有 ACTIVE 粒子，必須 complete 後才能取 results")
        self._validate_alignment()
        return [finalize_particle_execution(runtime.execution) for runtime in self.runtimes]

    def snapshot(
        self,
        *,
        sequence: int | None = None,
        binding: CheckpointBinding | None = None,
    ) -> ExecutionCheckpoint:
        """建立記憶體中的 execution checkpoint snapshot，不序列化 forcing 或 geometry。

        ``binding`` 可省略以便測試或檢查中途狀態；真正寫入磁碟時必須提供
        ``CheckpointBinding``，由 ``write_checkpoint`` 送入 schema 2 manifest。
        """

        from .checkpoint import build_execution_checkpoint

        self._validate_alignment()
        return build_execution_checkpoint(
            binding=binding,
            sequence=self.checkpoint_sequence if sequence is None else sequence,
            run_units=self.units,
            executions=[runtime.execution for runtime in self.runtimes],
            rngs=[runtime.rng for runtime in self.runtimes],
            triangle_hints=[runtime.triangle_hint for runtime in self.runtimes],
        )

    def write_checkpoint(
        self,
        destination: str | Path,
        *,
        binding: CheckpointBinding,
        sequence: int,
    ) -> Path:
        """寫出不可覆寫、原子完成且含 RNG continuation 的 schema 2 checkpoint。"""

        from .checkpoint import write_execution_checkpoint

        self._validate_alignment()
        path = write_execution_checkpoint(
            destination,
            binding=binding,
            sequence=sequence,
            run_units=self.units,
            executions=[runtime.execution for runtime in self.runtimes],
            rngs=[runtime.rng for runtime in self.runtimes],
            triangle_hints=[runtime.triangle_hint for runtime in self.runtimes],
        )
        self.checkpoint_sequence = sequence
        return path

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        shard: ScenarioShard,
        master_seed: int,
        request_factory: Callable[[RunUnit], ReferenceParticleRequest],
        expected_binding: CheckpointBinding,
        active_chunk_size: int | None = None,
    ) -> ProductionBatch:
        """以同一 shard、master seed 與 request factory 重建外部資源並恢復 checkpoint。"""

        from .checkpoint import load_execution_checkpoint

        batch = cls(
            shard,
            master_seed=master_seed,
            request_factory=request_factory,
            active_chunk_size=active_chunk_size,
        )
        loaded = load_execution_checkpoint(
            path,
            expected_binding=expected_binding,
            expected_run_units=batch.units,
        )
        if len(loaded.executions) != len(batch.runtimes):
            raise ValueError("checkpoint execution count 與 batch 不一致")
        for index, (runtime, execution, rng_state, triangle_hint) in enumerate(
            zip(
                batch.runtimes,
                loaded.executions,
                loaded.rng_states,
                loaded.triangle_hints,
                strict=True,
            )
        ):
            if _state_identity(execution.state) != _unit_identity(runtime.unit):
                raise ValueError(f"checkpoint execution identity 不一致：index={index}")
            runtime.execution = execution
            runtime.rng.bit_generator.state = rng_state
            runtime.triangle_hint = triangle_hint
            runtime.velocity.set_triangle_hint(triangle_hint)
        batch.particle_batch = ParticleBatch.from_particle_states(
            [runtime.execution.state for runtime in batch.runtimes],
            triangle_hints=[runtime.triangle_hint for runtime in batch.runtimes],
        )
        batch.checkpoint_sequence = loaded.sequence
        batch._validate_alignment()
        return batch

    restore = from_checkpoint
    restore_from_checkpoint = from_checkpoint


def run_production_shard(
    shard: ScenarioShard,
    *,
    master_seed: int,
    request_factory: Callable[[RunUnit], ReferenceParticleRequest],
    active_chunk_size: int | None = None,
    on_result: Callable[[RunUnit, ParticleResult], None] | None = None,
) -> list[ParticleResult]:
    """以 CPU/NumPy active-compacted orchestration 完成一個 scenario shard。"""

    batch = ProductionBatch(
        shard,
        master_seed=master_seed,
        request_factory=request_factory,
        active_chunk_size=active_chunk_size,
    )
    results = batch.complete()
    if on_result is not None:
        for unit, result in zip(batch.units, results, strict=True):
            on_result(unit, result)
    if len(results) != shard.particle_count:
        raise RuntimeError("production shard 結果數與 scenario×M 契約不符")
    return results


__all__ = [
    "HintTrackingVelocityProvider",
    "ProductionAdvanceResult",
    "ProductionBatch",
    "ProductionParticleRuntime",
    "ReferenceParticleRequest",
    "RunUnit",
    "ScenarioShard",
    "run_production_shard",
]
