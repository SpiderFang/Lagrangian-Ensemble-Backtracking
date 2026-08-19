"""signed-time RK4 與獨立 stochastic split 的純 NumPy 參考實作。

速度 provider 永遠回傳物理時間向前的速度；backward 只由負 ``dt_seconds`` 表達。
每個 RK stage 都重新取樣位置、z 與時間，任何 stage 無效即整步失敗，不能沿用步首速度
或把缺值當零。隨機位移在完整 RK4 後加入，符合文件的 operator-split 基線。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

import numpy as np

from .diffusion import DiffusionCoefficients, brownian_displacement
from .models import ParticleState, SampleQC, VelocitySample


class VelocityProvider(Protocol):
    """OCM/Stokes/浮沉合成速度必須實作的 stage 取樣介面。"""

    def __call__(self, x_m: float, y_m: float, z_m: float, time_utc_ns: int) -> VelocitySample:
        """回傳指定位置與 UTC 的物理時間向前速度。"""


class SamplingError(RuntimeError):
    """RK stage 因 forcing/QC 無效而中止，保留 bitmask 供 engine 分類。"""

    def __init__(self, stage: str, qc: SampleQC) -> None:
        super().__init__(f"RK4 {stage} 取樣無效：qc={int(qc)}")
        self.stage = stage
        self.qc = qc


def _velocity_vector(sample: VelocitySample, stage: str) -> np.ndarray:
    """把有效速度轉成 float64 三向量，無效則保留 stage/QC 丟例外。"""

    if not sample.valid:
        raise SamplingError(stage, sample.qc)
    vector = np.array([sample.u_mps, sample.v_mps, sample.w_mps], dtype=np.float64)
    if not np.all(np.isfinite(vector)):
        raise SamplingError(stage, SampleQC.NUMERICAL_FAILURE)
    return vector


def rk4_step(state: ParticleState, *, dt_seconds: float, velocity: VelocityProvider) -> ParticleState:
    """執行一個完整的 signed-time RK4 確定性步。

    ``age_seconds`` 增加 ``abs(dt)``，UTC 則依 signed dt 改變。ID、站點與 local-exit
    狀態保持不變；海面／海床與水平 boundary policy 由 engine 在步後依 crossing 處理。
    """

    if not np.isfinite(dt_seconds) or dt_seconds == 0:
        raise ValueError("RK4 dt_seconds 必須是有限非零值")
    position = np.array([state.x_m, state.y_m, state.z_m], dtype=np.float64)
    dt_ns = int(round(dt_seconds * 1_000_000_000))
    half_ns = int(round(dt_seconds * 0.5 * 1_000_000_000))
    k1 = _velocity_vector(velocity(*position, state.time_utc_ns), "k1")
    p2 = position + 0.5 * dt_seconds * k1
    k2 = _velocity_vector(velocity(*p2, state.time_utc_ns + half_ns), "k2")
    p3 = position + 0.5 * dt_seconds * k2
    k3 = _velocity_vector(velocity(*p3, state.time_utc_ns + half_ns), "k3")
    p4 = position + dt_seconds * k3
    k4 = _velocity_vector(velocity(*p4, state.time_utc_ns + dt_ns), "k4")
    advanced = position + dt_seconds * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return replace(
        state,
        x_m=float(advanced[0]),
        y_m=float(advanced[1]),
        z_m=float(advanced[2]),
        time_utc_ns=state.time_utc_ns + dt_ns,
        age_seconds=state.age_seconds + abs(dt_seconds),
    )


def split_rk4_brownian_step(
    state: ParticleState,
    *,
    dt_seconds: float,
    velocity: VelocityProvider,
    coefficients: DiffusionCoefficients,
    rng: np.random.Generator,
) -> ParticleState:
    """先做完整 RK4，再加入一次 Brownian 位移的基線 Lie split。"""

    advanced = rk4_step(state, dt_seconds=dt_seconds, velocity=velocity)
    displacement = brownian_displacement(coefficients, dt_seconds=dt_seconds, rng=rng)
    return replace(
        advanced,
        x_m=advanced.x_m + float(displacement[0]),
        y_m=advanced.y_m + float(displacement[1]),
        z_m=advanced.z_m + float(displacement[2]),
    )
