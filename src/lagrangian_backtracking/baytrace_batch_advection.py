"""以原版 BayTrace CPU kernel 批次執行確定性平流。

本模組是 ``ProductionBatch`` 可選用的 advection batcher，不是新的物理核心。
當呼叫端的 frame resolver 能為一個 task 提供已準備好的 BayTrace 網格與兩端
速度場時，本模組只建立該批次的小型粒子陣列，並把整組交給既有
``step_native``；因此真正的內插與四階 Runge--Kutta（RK4）公式仍由 PI 提供的
``batch_interpolate_velocity``／``batch_move_rk4`` 執行。resolver 回傳 ``None``
時，僅預設相容模式走既有 LBT legacy/reference fallback；``require_native=True``
必須附理由拒絕，不能轉舊核心。``native_particle_steps`` 僅計完成成功的粒子步數，
attempted另計核心嘗試，避免非有限或例外被冒稱成功。

這裡只處理確定性平流的一步：不做 OCM 時間列轉接、Stokes 轉接，也不在 Python
batcher 內另做元素搜尋、海岸／海面／海床 boundary lifecycle 或隨機擴散。原版
kernel 內部的 stage hint 只先查目前元素與 ``ic3`` 一環鄰居，找不到時退回原 hint；
四個 stage 共用同一 ``trat``，垂向層則沿用原版 ``find_vlevel`` 與 ``kbe`` clamp。
這些 degraded 語意必須由既有 LBT lifecycle 在步後另行處理，不能在本 batcher 內補
一套不同規則。
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from .baytrace_native_cpu import (
    BayTraceCPUKernels,
    BayTraceFieldPair,
    BayTraceGridArrays,
    BayTraceParticleArrays,
    step_native,
)
from .integrators import SamplingContext, SamplingError
from .models import ParticleState, ParticleStatus, SampleQC

if TYPE_CHECKING:
    from .advection import AdvectionTask


class NativeBayTraceUnsupportedError(RuntimeError):
    """工程未支援的可修復分片失敗；不是粒子物理數值失敗或終止狀態。

    reason為固定診斷識別碼。嚴格模式在任何核心執行前拋出，讓controller保留
    checkpoint並標記failed shard；不呼叫legacy、不消耗diffusion亂數。
    """

    def __init__(self, reason: str) -> None:
        """保存未支援原因供部署檢核，不夾帶路徑或改寫粒子QC。"""
        self.reason = reason
        super().__init__(f"native unsupported: {reason}")


@dataclass(frozen=True, slots=True)
class PreparedBayTraceFrame:
    """一個可直接交給原版 kernel 的固定網格／時間框架。

    ``grid`` 與 ``fields`` 必須由呼叫端快取並重用；batcher 只依兩個物件的
    identity 分組，不會對大型網格或速度場做 hash。``trat`` 是同一個 native
    step 四個 RK4 stage 共用的時間內插比例，不能在 batcher 內依粒子重新推導。
    ``element_id`` 是該 task 的步首元素提示，步後的相鄰元素搜尋與 boundary
    lifecycle 仍由既有 LBT 流程負責。

    沉降若已由 resolver 把固定值加入 ``w1``／``w2``，原版 kernel 的四個 stage
    就會自然使用該垂向速度；本類別不另加一步沉降位移，也不承諾 Stokes 或 OCM
    來源已完成轉接。
    """

    grid: BayTraceGridArrays
    fields: BayTraceFieldPair
    trat: float
    element_id: int

    def __post_init__(self) -> None:
        """在任何 batch 執行前驗證 frame 的小型 scalar 契約。"""

        if not isinstance(self.grid, BayTraceGridArrays):
            raise TypeError("PreparedBayTraceFrame.grid 必須是 BayTraceGridArrays")
        if not isinstance(self.fields, BayTraceFieldPair):
            raise TypeError("PreparedBayTraceFrame.fields 必須是 BayTraceFieldPair")
        try:
            trat = float(self.trat)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("PreparedBayTraceFrame.trat 必須是有限數值") from exc
        if not np.isfinite(trat) or not 0.0 <= trat <= 1.0:
            raise ValueError("PreparedBayTraceFrame.trat 必須是 [0, 1] 內的有限數值")
        if (
            isinstance(self.element_id, bool)
            or not isinstance(self.element_id, (int, np.integer))
        ):
            raise TypeError("PreparedBayTraceFrame.element_id 必須是整數")
        object.__setattr__(self, "trat", trat)
        object.__setattr__(self, "element_id", int(self.element_id))


@dataclass(frozen=True, slots=True)
class _PreparedTask:
    """完成預檢後的 task；保存 group 所需 scalar，避免執行中再次推導。"""

    source_index: int
    task: Any
    state: ParticleState
    dt_seconds: float
    dt_nanoseconds: int
    frame: PreparedBayTraceFrame | None


class NativeBayTraceBatchAdvector:
    """把支援 native 的 advection task 依固定 frame 批次送入原版 kernel。

    呼叫物件符合另一個 worker 所定義的 advection batcher 形狀：輸入是保持
    source order 的 ``Sequence[AdvectionTask]``，輸出是同順序的
    ``ParticleState`` 或 ``SamplingError``。作用中且有 frame 的 task 依
    ``grid``／``fields`` 物件 identity、精確 ``trat`` 與精確 signed
    ``dt_seconds`` 分組；每組恰好呼叫一次 ``step_native``，不在每粒子重新觸發
    JIT，也不掃描整個網格；UTC亦精確相同才同組。無frame在嚴格模式拒絕，只有
    預設相容模式交给既有legacy；群組直方圖與嘗試／成功／拒絕計數區分實際路徑。

    原生步驟只回傳平流後的位置、UTC 時間與正值 age；identity、status 及
    ``own_local_exit_recorded`` 由原 ``ParticleState`` 保留。負的
    ``dt_seconds`` 代表逆向回溯，但 ``age_seconds`` 仍增加正的步長絕對值。
    元素搜尋、垂向／邊界事件與 diffusion 不在此處處理。
    """

    def __init__(
        self,
        kernels: BayTraceCPUKernels,
        frame_resolver: Callable[[AdvectionTask], PreparedBayTraceFrame | None],
        *,
        require_native: bool = False,
    ) -> None:
        """建立批次器；嚴格模式禁止任何未支援任務退回舊核心。

        ``require_native`` 只接受真正布林值，預設False保留歷史參考路徑。
        True時工程無frame拋NativeBayTraceUnsupportedError；只有已知資料QC回傳
        原QC的SamplingError，不改步長、身份或隨機數。
        """

        if not isinstance(kernels, BayTraceCPUKernels):
            raise TypeError("kernels 必須是 BayTraceCPUKernels")
        if not callable(frame_resolver):
            raise TypeError("frame_resolver 必須是 callable")
        if type(require_native) is not bool:
            raise TypeError("require_native 必須是 bool")
        self.kernels = kernels
        self.frame_resolver = frame_resolver
        self.require_native = require_native

        # 這些公開整數是效能 evidence；它們只記錄實際進入 native／fallback 的
        # 粒子步數，不參與物理計算，也不把 fallback 的工作冒充成原版 kernel。
        self.native_particle_steps = 0
        self.native_batch_calls = 0
        self.fallback_particle_steps = 0
        self.max_group_size = 0
        # 舊native_particle_steps改為成功完成數；attempted包含核心失敗，不能
        # 因已進入dispatcher就宣稱成功。直方圖記實際送入核心的群組大小。
        self.native_attempted_particle_steps = 0
        self.native_rejected_particle_steps = 0
        self.group_size_histogram: Counter[int] = Counter()
        self.rejected_reasons: Counter[str] = Counter()
        self._unsupported_reasons: dict[int, str] = {}
        self._unsupported_qc: dict[int, SampleQC | None] = {}

    def __call__(
        self,
        tasks: Sequence[AdvectionTask],
    ) -> list[ParticleState | SamplingError]:
        """依 source order 執行一批 task，並將各結果 scatter 回原索引。

        所有 task 的 dt、state、frame scalar 與 element index 會在第一次
        ``step_native`` 前完成預檢；因此同一批中只要有非法 dt、frame 或 index，
        就會先整批拒絕，不會出現前半批已偷偷成功、後半批才失敗的狀態。inactive
        state 不呼叫 resolver、不送進 native，也不增加任何 step counter。
        """

        try:
            task_list = tuple(tasks)
        except TypeError as exc:
            raise TypeError("tasks 必須是可迭代的 AdvectionTask sequence") from exc
        if not task_list:
            return []

        prepared: list[_PreparedTask] = []
        groups: dict[tuple[int, int, float, float, int], list[_PreparedTask]] = {}
        fallback: list[_PreparedTask] = []

        # 這一輪只讀取並驗證輸入；任何 group 仍尚未執行，確保 scalar/index
        # 錯誤不會在已完成一部分 native batch 後才被發現。
        for source_index, task in enumerate(task_list):
            record = self._prepare_task(source_index, task)
            prepared.append(record)
            if record.state.status != ParticleStatus.ACTIVE:
                continue
            if record.frame is None:
                if self.require_native and self._unsupported_qc[source_index] is None:
                    reason = self._unsupported_reasons[source_index]
                    self.native_rejected_particle_steps += 1
                    self.rejected_reasons[reason] += 1
                    self._unsupported_reasons.clear()
                    self._unsupported_qc.clear()
                    raise NativeBayTraceUnsupportedError(reason)
                fallback.append(record)
                continue
            frame = record.frame
            key = (id(frame.grid), id(frame.fields), frame.trat, record.dt_seconds, record.state.time_utc_ns)
            groups.setdefault(key, []).append(record)

        results: list[ParticleState | SamplingError | None] = [None] * len(prepared)
        for record in prepared:
            if record.state.status != ParticleStatus.ACTIVE:
                # 不傳入 inactive，也不呼叫 fallback；既有生命週期已把它標示為
                # terminal 或其他非 active 狀態，本 adapter 不得重新啟動它。
                results[record.source_index] = record.state

        # group 的成員維持原 source order；最後仍以 source_index scatter，故不同
        # dt／trat 或 native/fallback 交錯時，identity 不會因壓縮而錯位。
        for group in groups.values():
            self._run_native_group(group, results)
        for record in fallback:
            if self.require_native:
                reason = self._unsupported_reasons.pop(record.source_index, "frame_unavailable")
                self.native_rejected_particle_steps += 1
                self.rejected_reasons[reason] += 1
                results[record.source_index] = SamplingError(
                    f"native_rejected:{reason}", self._unsupported_qc[record.source_index],
                    context=SamplingContext(x_m=record.state.x_m, y_m=record.state.y_m,
                                            z_m=record.state.z_m, time_utc_ns=record.state.time_utc_ns),
                )
            else:
                self._run_fallback(record, results)
        self._unsupported_reasons.clear()
        self._unsupported_qc.clear()

        if any(result is None for result in results):
            raise RuntimeError("batch advection 結果未完整 scatter")
        return [result for result in results if result is not None]

    def report(self) -> dict[str, Any]:
        """回傳可直接 JSON 序列化的來源 hash 與本程序計數 evidence。"""

        provenance = self.kernels.provenance
        return {
            "backend": "native_baytrace_batch_advector",
            "fallback_backend": "lbt_legacy_execute_advection_task",
            "source_root": str(provenance.source_root),
            "package_dir": str(provenance.package_dir),
            "cache_dir": str(provenance.cache_dir),
            "source_sha256": dict(provenance.dependency_sha256),
            "native_particle_steps": int(self.native_particle_steps),
            "native_batch_calls": int(self.native_batch_calls),
            "fallback_particle_steps": int(self.fallback_particle_steps),
            "max_group_size": int(self.max_group_size),
            "require_native": self.require_native,
            "native_attempted_particle_steps": self.native_attempted_particle_steps,
            "native_completed_particle_steps": self.native_particle_steps,
            "native_rejected_particle_steps": self.native_rejected_particle_steps,
            "native_rejected_reasons": dict(self.rejected_reasons),
            "group_size_histogram": {str(k): v for k, v in sorted(self.group_size_histogram.items())},
            "counter_semantics": "native_steps_completed_only_v2",
        }

    def _prepare_task(self, source_index: int, task: Any) -> _PreparedTask:
        """驗證單一 task 的狀態／步長，並在 active 時解析 frame。"""

        state = getattr(task, "state", None)
        if not isinstance(state, ParticleState):
            raise TypeError(f"tasks[{source_index}].state 必須是 ParticleState")
        self._validate_state(source_index, state)

        raw_dt = getattr(task, "dt_seconds", None)
        if isinstance(raw_dt, bool):
            raise TypeError(f"tasks[{source_index}].dt_seconds 不可為 bool")
        try:
            dt_seconds = float(raw_dt)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"tasks[{source_index}].dt_seconds 必須是有限非零數值") from exc
        if not np.isfinite(dt_seconds) or dt_seconds == 0.0:
            raise ValueError(f"tasks[{source_index}].dt_seconds 必須是有限非零數值")
        dt_ns_float = dt_seconds * 1_000_000_000.0
        if not np.isfinite(dt_ns_float):
            raise ValueError(f"tasks[{source_index}].dt_seconds 超出 UTC 奈秒轉換範圍")
        dt_nanoseconds = int(round(dt_ns_float))

        frame: PreparedBayTraceFrame | None = None
        if state.status == ParticleStatus.ACTIVE:
            frame = self.frame_resolver(task)
            if frame is None:
                self._unsupported_reasons[source_index] = str(
                    getattr(self.frame_resolver, "last_rejection_reason", None) or "frame_unavailable"
                )
                self._unsupported_qc[source_index] = getattr(self.frame_resolver, "last_rejection_qc", None)
            if frame is not None and not isinstance(frame, PreparedBayTraceFrame):
                raise TypeError(
                    f"frame_resolver(tasks[{source_index}]) 必須回傳 PreparedBayTraceFrame 或 None"
                )
            if frame is not None:
                self._validate_frame(source_index, frame)

        return _PreparedTask(
            source_index=source_index,
            task=task,
            state=state,
            dt_seconds=dt_seconds,
            dt_nanoseconds=dt_nanoseconds,
            frame=frame,
        )

    @staticmethod
    def _validate_state(source_index: int, state: ParticleState) -> None:
        """拒絕會在 native 後產生不可追溯座標／age 的輸入狀態。"""

        for name in ("x_m", "y_m", "z_m", "age_seconds"):
            try:
                value = float(getattr(state, name))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"tasks[{source_index}].state.{name} 必須是有限數值") from exc
            if not math.isfinite(value):
                raise ValueError(f"tasks[{source_index}].state.{name} 不可為 NaN 或無限值")
        if state.age_seconds < 0.0:
            raise ValueError(f"tasks[{source_index}].state.age_seconds 不可為負")
        if isinstance(state.time_utc_ns, bool) or not isinstance(
            state.time_utc_ns, (int, np.integer)
        ):
            raise TypeError(f"tasks[{source_index}].state.time_utc_ns 必須是整數奈秒")

    @staticmethod
    def _validate_frame(source_index: int, frame: PreparedBayTraceFrame) -> None:
        """驗證 frame 與 grid 尺寸／element hint 相容，但不重掃大型資料。"""

        grid = frame.grid
        fields = frame.fields
        if frame.element_id < 0 or frame.element_id >= grid.i34.size:
            raise ValueError(
                f"tasks[{source_index}].frame.element_id 超出 grid 元素範圍: {frame.element_id}"
            )
        expected_shape = (grid.nvrt, grid.grid_x.size)
        if fields.u1.shape != expected_shape:
            raise ValueError(
                f"tasks[{source_index}].frame.fields 與 grid shape 不一致: "
                f"{fields.u1.shape} != {expected_shape}"
            )
        if fields.eta1.shape != (grid.grid_x.size,):
            raise ValueError(f"tasks[{source_index}].frame.fields.eta1 與 grid 節點數不一致")

    def _run_native_group(
        self,
        group: list[_PreparedTask],
        results: list[ParticleState | SamplingError | None],
    ) -> None:
        """以一次原始 ``step_native`` 處理同一 exact group。"""

        frame = group[0].frame
        if frame is None:  # pragma: no cover - group 建立時已由預檢排除
            raise RuntimeError("native group 不得沒有 PreparedBayTraceFrame")
        grid = frame.grid
        fields = frame.fields
        element_ids: list[int] = []
        for record in group:
            if record.frame is None:  # pragma: no cover - group 建立時已由預檢排除
                raise RuntimeError("native group 成員不得沒有 PreparedBayTraceFrame")
            element_ids.append(record.frame.element_id)
        particles = BayTraceParticleArrays.from_positions(
            x=np.asarray([record.state.x_m for record in group], dtype=np.float64),
            y=np.asarray([record.state.y_m for record in group], dtype=np.float64),
            z=np.asarray([record.state.z_m for record in group], dtype=np.float64),
            elem=np.asarray(element_ids, dtype=np.int64),
        )

        # 同一 group 的 signed dt 完全相等；step_native 接受正 magnitude，方向
        # 由 ibf 表示，避免在 adapter 內改寫原版 RK4 公式。
        dt_seconds = group[0].dt_seconds
        ibf = 1 if dt_seconds > 0.0 else -1
        self.native_batch_calls += 1
        self.native_attempted_particle_steps += len(group)
        self.max_group_size = max(self.max_group_size, len(group))
        self.group_size_histogram[len(group)] += 1
        try:
            step_native(
                self.kernels, grid, fields, particles,
                dt_seconds=abs(dt_seconds), trat=frame.trat, ibf=ibf,
            )
        except Exception:
            # dispatcher異常不等於成功；保留原例外供上游fail-closed，不偷偷重算。
            self.native_rejected_particle_steps += len(group)
            self.rejected_reasons["native_kernel_exception"] += len(group)
            raise

        for local_index, record in enumerate(group):
            result = self._native_result(record, particles, local_index)
            if isinstance(result, SamplingError):
                self.native_rejected_particle_steps += 1
                self.rejected_reasons["native_nonfinite_result"] += 1
            else:
                self.native_particle_steps += 1
            results[record.source_index] = result

    @staticmethod
    def _native_result(
        record: _PreparedTask,
        particles: BayTraceParticleArrays,
        local_index: int,
    ) -> ParticleState | SamplingError:
        """把 native arrays 轉回 state，並將非有限結果分類為數值失敗。"""

        finite_values = np.array(
            [
                particles.x[local_index],
                particles.y[local_index],
                particles.z[local_index],
                particles.u[local_index],
                particles.v[local_index],
                particles.w[local_index],
            ],
            dtype=np.float64,
        )
        new_age = float(record.state.age_seconds) + abs(record.dt_seconds)
        if not np.all(np.isfinite(finite_values)) or not math.isfinite(new_age):
            return _numerical_failure(record.state, stage="native")
        return replace(
            record.state,
            x_m=float(particles.x[local_index]),
            y_m=float(particles.y[local_index]),
            z_m=float(particles.z[local_index]),
            time_utc_ns=int(record.state.time_utc_ns) + record.dt_nanoseconds,
            age_seconds=new_age,
        )

    def _run_fallback(
        self,
        record: _PreparedTask,
        results: list[ParticleState | SamplingError | None],
    ) -> None:
        """執行明示的 LBT legacy/reference fallback 並驗證回傳型別。"""

        self.fallback_particle_steps += 1
        try:
            result = _execute_legacy_advection_task(record.task)
        except SamplingError as error:
            result = error
        if isinstance(result, SamplingError):
            results[record.source_index] = result
            return
        if not isinstance(result, ParticleState):
            raise TypeError("execute_advection_task 必須回傳 ParticleState 或 SamplingError")
        if _state_identity(result) != _state_identity(record.state):
            raise ValueError("legacy fallback 不得改變粒子 identity")
        if not _state_finite(result):
            results[record.source_index] = _numerical_failure(record.state, stage="legacy_fallback")
            return
        results[record.source_index] = result


def _execute_legacy_advection_task(task: Any) -> ParticleState | SamplingError:
    """延遲載入另一 worker 提供的既有 LBT advection entry point。

    延遲 import 讓本模組可以在 ``advection.py`` 尚未合併時先被編譯／測試；
    真正使用 fallback 時仍會要求該既有 entry point 存在，不以本模組偷偷重寫
    RK4 或宣稱 native 支援。
    """

    from .advection import execute_advection_task

    return execute_advection_task(task)


def _state_identity(state: ParticleState) -> tuple[object, ...]:
    """取得 fallback 不可改變的粒子身分欄位。"""

    return (
        state.particle_id,
        state.scenario_id,
        state.member_id,
        state.study_site_id,
        state.analysis_region_id,
        state.receptor_id,
    )


def _state_finite(state: ParticleState) -> bool:
    """檢查 fallback state 的位置與 age，不讓 NaN 進入下游 lifecycle。"""

    try:
        return all(
            math.isfinite(float(getattr(state, name)))
            for name in ("x_m", "y_m", "z_m", "age_seconds")
        )
    except (TypeError, ValueError, OverflowError):
        return False


def _numerical_failure(state: ParticleState, *, stage: str) -> SamplingError:
    """將 native／fallback 非有限結果轉成可由既有引擎分類的 SamplingError。"""

    return SamplingError(
        stage,
        SampleQC.NUMERICAL_FAILURE,
        context=SamplingContext(
            x_m=float(state.x_m),
            y_m=float(state.y_m),
            z_m=float(state.z_m),
            time_utc_ns=int(state.time_utc_ns),
        ),
    )


__all__ = ["NativeBayTraceBatchAdvector", "NativeBayTraceUnsupportedError", "PreparedBayTraceFrame"]
