"""計算單一粒子一步移動的基準方法。

本模組以四階 Runge-Kutta 法（RK4）計算海流、波浪造成的確定移動，再另外加入隨機擴散
造成的位移。速度資料一律表示「物理時間往後」的流速；逆向溯源時只要給負的時間步長，
便會沿相反時間方向回推。每個中間計算點都必須重新讀取速度，若資料缺漏或位置無效，
整步便停止，絕不把缺值當成零速度。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

import numpy as np

from .diffusion import DiffusionCoefficients, brownian_displacement
from .models import ParticleState, SampleQC, VelocitySample


class VelocityProvider(Protocol):
    """取得某位置、深度與時刻速度的共同介面。

    海流、波浪造成的表面漂移、浮沉速度可以先各自處理，再由呼叫端合成為此介面需要的
    三個方向速度。輸入座標使用公尺，深度 ``z_m`` 以海面為零且水下為負，時間使用世界
    協調時間（UTC）的奈秒整數；回傳值中的品質旗標會說明資料是否可用。
    """

    def __call__(self, x_m: float, y_m: float, z_m: float, time_utc_ns: int) -> VelocitySample:
        """回傳指定位置與 UTC 時刻、物理時間往後的三向速度。"""


class SamplingError(RuntimeError):
    """中間計算點無法取得可用速度時拋出的例外。

    ``stage`` 說明失敗發生在四階計算的哪一個中間點；``qc`` 保留品質檢查旗標，讓粒子
    引擎可區分「資料缺口」與「數值計算失敗」，而非把兩者混為同一種停止原因。
    """

    def __init__(self, stage: str, qc: SampleQC) -> None:
        super().__init__(f"RK4 {stage} 取樣無效：qc={int(qc)}")
        self.stage = stage
        self.qc = qc


def _velocity_vector(sample: VelocitySample, stage: str) -> np.ndarray:
    """把已通過檢查的速度轉為三個浮點數；無效時保留原因並停止這一步。"""

    if not sample.valid:
        raise SamplingError(stage, sample.qc)
    vector = np.array([sample.u_mps, sample.v_mps, sample.w_mps], dtype=np.float64)
    if not np.all(np.isfinite(vector)):
        raise SamplingError(stage, SampleQC.NUMERICAL_FAILURE)
    return vector


def rk4_step(state: ParticleState, *, dt_seconds: float, velocity: VelocityProvider) -> ParticleState:
    """以四階 Runge-Kutta 法計算一次不含隨機擴散的粒子移動。

    ``dt_seconds`` 為正代表往未來推進，為負代表往過去回溯。粒子的已追蹤時間
    ``age_seconds`` 永遠增加正值，UTC 時刻則依時間步長的正負方向改變。此函式只改變
    位置、深度與時間；碰到海面、海床、海岸或研究範圍邊界的處理，交由粒子引擎在本步
    完成後統一判定，避免不同規則互相覆蓋。
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
    """先依流速移動，再加入一次隨機擴散位移。

    將流速移動與隨機擴散分開計算，可清楚檢查兩種影響各自是否正確；隨機位移只在完整
    的四階流速計算完成後加入一次，因此不會在同一時間步中被重複套用。
    """

    advanced = rk4_step(state, dt_seconds=dt_seconds, velocity=velocity)
    displacement = brownian_displacement(coefficients, dt_seconds=dt_seconds, rng=rng)
    return replace(
        advanced,
        x_m=advanced.x_m + float(displacement[0]),
        y_m=advanced.y_m + float(displacement[1]),
        z_m=advanced.z_m + float(displacement[2]),
    )
