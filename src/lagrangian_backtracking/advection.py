"""確定性平流工作的單粒子與批次介面。

本模組只描述四階 Runge--Kutta（RK4）平流工作的輸入與輸出，不處理隨機擴散、邊界
事件、觀測或粒子生命週期。這個邊界讓 ``ProductionBatch`` 可以把多條已由 engine
決定步長的工作交給同一個 batcher，同時保留 engine 對步長、品質檢查旗標（QC）、
亂數產生器（RNG）與 checkpoint 的唯一所有權。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from .integrators import SamplingError, VelocityProvider, rk4_step
from .models import ParticleState, VelocitySample


@dataclass(frozen=True, slots=True)
class AdvectionTask:
    """一個已完成步長決策、只含確定性 RK4 平流的工作描述。

    ``state`` 是步首粒子狀態，位置使用公尺、時間使用 UTC 奈秒；``dt_seconds`` 保留
    engine 決定的有號秒數，回溯時為負值。``velocity`` 與可選的
    ``step_start_sample`` 沿用既有取樣器與步首樣本重用契約，不能在 batch 層自行新增
    查詢或改變步長。``physics_kernel_backend`` 只選擇既有 RK4 純數值後端，不包含
    Brownian 或空間擴散的任何隨機操作。

    這個類別刻意是 frozen：batcher 可以在不同執行後端間傳遞同一份不可變工作描述，
    而不會把一條粒子的步首資料、時間軸或三角形提示在批次處理中途改成另一條粒子。
    """

    state: ParticleState
    dt_seconds: float
    velocity: VelocityProvider
    step_start_sample: VelocitySample | None
    physics_kernel_backend: str


class BatchAdvector(Protocol):
    """多粒子確定性平流 batcher 的 callable 契約。

    呼叫端依原始 source order 傳入一個或多個 ``AdvectionTask``。回傳序列必須與輸入
    等長且逐項對齊：成功項是只完成 RK4 的 ``ParticleState``，取樣失敗項是原本
    ``SamplingError``。batcher 不得加入擴散、邊界解析、觀測、步長決策或 RNG；這些
    行為仍由 engine 在每條 generator 的原有生命週期中完成。
    """

    def __call__(
        self, tasks: Sequence[AdvectionTask]
    ) -> Sequence[ParticleState | SamplingError]:
        """批次執行確定性 RK4，並依輸入順序回傳成功狀態或取樣錯誤。"""


def execute_advection_task(task: AdvectionTask) -> ParticleState:
    """以既有 RK4 核心執行一個確定性平流工作。

    此函式只呼叫 ``integrators.rk4_step``；它不抽取 RNG、不套用擴散、不判定邊界，也
    不寫入 observation。``SamplingError`` 會原樣交回呼叫端，讓 engine 可以以
    ``generator.throw`` 注入同一個錯誤並保留原有的自適應折半、海面 stage 調節與
    terminal 診斷順序。
    """

    return rk4_step(
        task.state,
        dt_seconds=task.dt_seconds,
        velocity=task.velocity,
        step_start_sample=task.step_start_sample,
        physics_kernel_backend=task.physics_kernel_backend,
    )


__all__ = ["AdvectionTask", "BatchAdvector", "execute_advection_task"]
