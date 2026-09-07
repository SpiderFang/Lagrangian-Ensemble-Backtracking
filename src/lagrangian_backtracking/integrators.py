"""計算單一粒子一步移動的基準方法。

本模組以四階 Runge-Kutta 法（RK4）計算海流、波浪造成的確定移動，再另外加入隨機擴散
造成的位移。速度資料一律表示「物理時間往後」的流速；逆向溯源時只要給負的時間步長，
便會沿相反時間方向回推。每個中間計算點都必須重新讀取速度，若資料缺漏或位置無效，
整步便停止，絕不把缺值當成零速度。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np

from .diffusion import (
    DiffusionCoefficients,
    DiffusionSample,
    brownian_displacement,
    diffusion_displacement,
)
from .models import ParticleState, SampleQC, VelocitySample


class VelocityProvider(Protocol):
    """取得某位置、深度與時刻速度的共同介面。

    海流、波浪造成的表面漂移、浮沉速度可以先各自處理，再由呼叫端合成為此介面需要的
    三個方向速度。輸入座標使用公尺，深度 ``z_m`` 以海面為零且水下為負，時間使用世界
    協調時間（UTC）的奈秒整數；回傳值中的品質旗標會說明資料是否可用。
    """

    def __call__(self, x_m: float, y_m: float, z_m: float, time_utc_ns: int) -> VelocitySample:
        """回傳指定位置與 UTC 時刻、物理時間往後的三向速度。"""


@dataclass(frozen=True, slots=True)
class SamplingContext:
    """保留失敗查詢當下已知的位置、時間及樣本上下界，不重新取樣。

    三軸位置與海面／海床高程均為公尺，垂向向上為正；時間是世界協調時間（UTC）
    奈秒整數。欄位只來自該次查詢的引數與回傳樣本，不讀取樣本的任意診斷字典。
    記憶體中允許未知值 ``None`` 或失敗樣本的非有限值；引擎寫入事件時會省略它們並
    標示不可用，絕不補零或將非有限值寫成 JSON。這些欄位不參與積分或邊界判定。
    """

    x_m: float | None = None
    y_m: float | None = None
    z_m: float | None = None
    time_utc_ns: int | None = None
    eta_m: float | None = None
    bed_z_m: float | None = None


class SamplingError(RuntimeError):
    """中間計算點無法取得可用速度時拋出的例外。

    ``stage`` 說明失敗發生在四階計算的哪一個中間點；``qc`` 保留品質檢查旗標，讓粒子
    引擎可區分「資料缺口」與「數值計算失敗」，而非把兩者混為同一種停止原因。
    可選的具型別上下文（``context``）只補充既有查詢證據；舊的兩參數呼叫仍有效。
    外部傳入的 ``stage`` 與例外文字不保證安全，事件序列化必須使用階段白名單，
    不得直接複製例外訊息。新增上下文不改變原有例外觸發條件或恢復策略。
    """

    def __init__(
        self, stage: str, qc: SampleQC, *, context: SamplingContext | None = None
    ) -> None:
        """保存品質旗標與可選查詢證據；不藉診斷資料改變原有失敗分類。"""

        super().__init__(f"RK4 {stage} 取樣無效：qc={int(qc)}")
        self.stage = stage
        self.qc = qc
        self.context = context


def _velocity_vector(
    sample: VelocitySample, stage: str, *, position: np.ndarray, time_utc_ns: int
) -> np.ndarray:
    """檢查已取得的速度，僅在失敗時附上同次查詢的公尺位置與 UTC 奈秒。

    有效分支保持原三向速度陣列與有限值檢查；失敗分支只讀取已在記憶體的座標與
    海面／海床，不新增速度查詢，也不改寫樣本或四階中間位置。
    """

    if not sample.valid:
        raise SamplingError(
            stage, sample.qc,
            context=SamplingContext(*position, time_utc_ns, sample.eta_m, sample.bed_z_m),
        )
    vector = np.array([sample.u_mps, sample.v_mps, sample.w_mps], dtype=np.float64)
    if not np.all(np.isfinite(vector)):
        raise SamplingError(
            stage, SampleQC.NUMERICAL_FAILURE,
            context=SamplingContext(*position, time_utc_ns, sample.eta_m, sample.bed_z_m),
        )
    return vector


def rk4_step(state: ParticleState, *, dt_seconds: float, velocity: VelocityProvider) -> ParticleState:
    """以四階 Runge-Kutta 法計算一次不含隨機擴散的粒子移動。

    ``dt_seconds`` 為正代表往未來推進，為負代表往過去回溯。粒子的已追蹤時間
    ``age_seconds`` 永遠增加正值，UTC 時刻則依時間步長的正負方向改變。此函式只改變
    位置、深度與時間；碰到海面、海床、海岸或研究範圍邊界的處理，交由粒子引擎在本步
    完成後統一判定，避免不同規則互相覆蓋。失敗上下文綁定原本 k1--k4 查詢的位置與
    時刻，不為診斷增加查詢、重試或亂數消耗。
    """

    if not np.isfinite(dt_seconds) or dt_seconds == 0:
        raise ValueError("RK4 dt_seconds 必須是有限非零值")
    position = np.array([state.x_m, state.y_m, state.z_m], dtype=np.float64)
    dt_ns = int(round(dt_seconds * 1_000_000_000))
    half_ns = int(round(dt_seconds * 0.5 * 1_000_000_000))
    k1 = _velocity_vector(
        velocity(*position, state.time_utc_ns), "k1",
        position=position, time_utc_ns=state.time_utc_ns,
    )
    p2 = position + 0.5 * dt_seconds * k1
    k2 = _velocity_vector(
        velocity(*p2, state.time_utc_ns + half_ns), "k2",
        position=p2, time_utc_ns=state.time_utc_ns + half_ns,
    )
    p3 = position + 0.5 * dt_seconds * k2
    k3 = _velocity_vector(
        velocity(*p3, state.time_utc_ns + half_ns), "k3",
        position=p3, time_utc_ns=state.time_utc_ns + half_ns,
    )
    p4 = position + dt_seconds * k3
    k4 = _velocity_vector(
        velocity(*p4, state.time_utc_ns + dt_ns), "k4",
        position=p4, time_utc_ns=state.time_utc_ns + dt_ns,
    )
    advanced = position + dt_seconds * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return replace(
        state,
        x_m=float(advanced[0]),
        y_m=float(advanced[1]),
        z_m=float(advanced[2]),
        time_utc_ns=state.time_utc_ns + dt_ns,
        age_seconds=state.age_seconds + abs(dt_seconds),
    )


def apply_diffusion_step(
    state: ParticleState,
    *,
    dt_seconds: float,
    coefficients: DiffusionCoefficients | DiffusionSample,
    rng: np.random.Generator,
) -> ParticleState:
    """對已完成確定性步驟的狀態套用一次 operator-split 擴散位移。

    這個入口把 Brownian 隨機位移及空間擴散的 ``+div(K)|dt|`` 漂移集中在同一處，
    讓一般 RK4 成功路徑與「RK stage 已知越過海面、先做幾何反射」的 recovery 路徑
    使用完全相同的擴散契約。``state`` 的時間與年齡應已代表本次確定性步驟的末端；
    本函式只改變三個公尺制位置，不再次取樣速度、不在 RK4 stage 中插入亂數，而且每次
    呼叫恰消耗一次三軸 ``normal(size=3)``。若 caller 已判定邊界為終止狀態，禁止呼叫
    本函式，因為終止 recovery 不應在停止時間之後追加 diffusion。
    """

    if isinstance(coefficients, DiffusionCoefficients):
        displacement = brownian_displacement(coefficients, dt_seconds=dt_seconds, rng=rng)
    elif isinstance(coefficients, DiffusionSample):
        displacement = diffusion_displacement(coefficients, dt_seconds, rng)
    else:
        raise TypeError("coefficients 必須是 DiffusionCoefficients 或 DiffusionSample")
    return replace(
        state,
        x_m=state.x_m + float(displacement[0]),
        y_m=state.y_m + float(displacement[1]),
        z_m=state.z_m + float(displacement[2]),
    )


def split_rk4_brownian_step(
    state: ParticleState,
    *,
    dt_seconds: float,
    velocity: VelocityProvider,
    coefficients: DiffusionCoefficients | DiffusionSample,
    rng: np.random.Generator,
) -> ParticleState:
    """先依流速移動，再加入一次隨機擴散位移。

    將流速移動與隨機擴散分開計算，可清楚檢查兩種影響各自是否正確；隨機位移只在完整
    的四階流速計算完成後加入一次，因此不會在同一時間步中被重複套用。``coefficients``
    若是舊版常數 ``DiffusionCoefficients``，只加入 ``sqrt(2K|dt|)N``；若是步首
    ``DiffusionSample``，則另外加入該樣本的 ``+div(K)|dt|`` pseudo-time 漂移。兩種
    路徑都不會在 RK4 stage 中讀取或消耗擴散亂數。
    """

    advanced = rk4_step(state, dt_seconds=dt_seconds, velocity=velocity)
    # 一般完整 RK4 路徑與特殊 recovery 路徑都共用同一個 helper，確保 Brownian 與
    # +div(K)|dt| 只在確定性計算完成後套用一次，且維持既有 seed／亂數消耗順序。
    return apply_diffusion_step(
        advanced,
        dt_seconds=dt_seconds,
        coefficients=coefficients,
        rng=rng,
    )
