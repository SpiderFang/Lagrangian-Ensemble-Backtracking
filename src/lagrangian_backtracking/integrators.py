"""計算單一粒子一步移動的基準方法。

本模組以四階 Runge-Kutta 法（RK4）計算海流、波浪造成的確定移動，再另外加入隨機擴散
造成的位移。速度資料一律表示「物理時間往後」的流速；逆向溯源時只要給負的時間步長，
便會沿相反時間方向回推。每個中間計算點都必須重新讀取速度，若資料缺漏或位置無效，
整步便停止，絕不把缺值當成零速度。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
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


_RK4_STAGE_NAMES = ("k1", "k2", "k3", "k4")


@dataclass(slots=True)
class SurfaceStageVelocityProvider:
    """只供單次完整 RK4 重試使用的海面 stage 反射速度包裝器。

    一般 ``VelocityProvider`` 必須拒絕超過其海面數值容許帶的查詢，不能為了讓積分通過
    而接受任意水面以上位置。本包裝器因此只由 engine 在兩項條件同時成立時建立：原始
    k2、k3 或 k4 已由同次失敗樣本證明越過當地移動海面，且再把時間步長折半會低於
    ``dt_min``。它只服務接下來那一次完整的確定性四階 Runge-Kutta 法（RK4）重試，
    不改變底層 forcing API，也不修改粒子 state。

    每個 stage 先以原始 ``x/y/z/t`` 查詢；僅當品質檢查旗標（``qc``）精確等於
    ``VERTICAL_UNSUPPORTED``、海面與海床皆為有限相容的公尺制正向向上座標、步首嚴格
    位於實際水柱內、行為不是上浮，且中間計算點的 z 嚴格高於 eta 時，才以
    ``z_reflected = 2*eta - z`` 鏡射。鏡射深度必須嚴格落在該次樣本的 ``[bed, eta]``，
    再於完全相同的 x、y、UTC 奈秒與鏡射 z 重查速度。這相當於只對中間導數施加反射
    數值邊界條件；RK4 的原始中間幾何與最後提議位置不會被夾回，步末仍由既有垂向邊界
    解析器判斷是否真的接觸海面。一般取樣介面的 5 微米海面容許帶不會在這裡再套一次。

    重查若仍無效、速度非有限或回傳幾何與鏡射深度不相容，會立即以該鏡射查詢的實際
    座標拋出 ``SamplingError``。這可保留 dry、域外、時間缺口及其他 QC，且失敗發生在
    Brownian 隨機擴散之前，不消耗亂數。包裝器依一次 RK4 固定的四次 stage 呼叫順序
    辨識 k1--k4；同一實例不得跨兩次 RK4 嘗試重用。
    """

    velocity: VelocityProvider
    step_start_state: ParticleState
    step_start_sample: VelocitySample
    behavior_class: str
    _stage_index: int = field(default=0, init=False, repr=False)

    def __call__(
        self, x_m: float, y_m: float, z_m: float, time_utc_ns: int
    ) -> VelocitySample:
        """查詢一個 RK4 stage，僅在已完整證明海面越界時改以鏡射深度重查。"""

        stage = _RK4_STAGE_NAMES[self._stage_index] if self._stage_index < 4 else None
        self._stage_index += 1
        sample = self.velocity(x_m, y_m, z_m, time_utc_ns)
        reflected_z = self._reflected_stage_z(stage=stage, z_m=z_m, sample=sample)
        if reflected_z is None:
            return sample
        # ``_reflected_stage_z`` 只會替 k2--k4 回傳數值；明示此不變量可避免失敗診斷
        # 在未來調整呼叫順序時悄悄退化成未知階段。
        assert stage is not None

        # 直接呼叫底層 provider，避免鏡射重查被誤計為下一個 RK4 stage。x/y/t 完全保留，
        # 唯一改變的是垂向查詢位置；這項內部調節不寫入或擴張輸出 schema。
        reflected_sample = self.velocity(x_m, y_m, reflected_z, time_utc_ns)
        context = SamplingContext(
            x_m,
            y_m,
            reflected_z,
            time_utc_ns,
            reflected_sample.eta_m,
            reflected_sample.bed_z_m,
        )
        if not reflected_sample.valid:
            raise SamplingError(stage, reflected_sample.qc, context=context)
        if not self._reflected_sample_contains_z(reflected_sample, reflected_z):
            raise SamplingError(
                stage,
                SampleQC.VERTICAL_UNSUPPORTED,
                context=context,
            )
        try:
            reflected_velocity = tuple(
                float(value)
                for value in (
                    reflected_sample.u_mps,
                    reflected_sample.v_mps,
                    reflected_sample.w_mps,
                )
            )
        except (TypeError, ValueError, OverflowError):
            raise SamplingError(stage, SampleQC.NUMERICAL_FAILURE, context=context) from None
        if not all(math.isfinite(value) for value in reflected_velocity):
            raise SamplingError(stage, SampleQC.NUMERICAL_FAILURE, context=context)
        return reflected_sample

    def _reflected_stage_z(
        self,
        *,
        stage: str | None,
        z_m: float,
        sample: VelocitySample,
    ) -> float | None:
        """核對 stage crossing 的完整幾何證據，回傳合法鏡射深度或 ``None``。

        ``None`` 表示包裝器沒有權限調節該查詢，呼叫端會把原樣本交回既有 RK4 錯誤
        流程。特別是 k1、組合 QC、缺值、上浮行為、步首海床以下或鏡射後海床以下，
        都不能藉由海面反射轉成有效速度。
        """

        if (
            stage not in {"k2", "k3", "k4"}
            or self.behavior_class == "rising"
            or not self.step_start_sample.valid
            or sample.qc != SampleQC.VERTICAL_UNSUPPORTED
        ):
            return None
        try:
            start_z = float(self.step_start_state.z_m)
            start_eta = float(self.step_start_sample.eta_m)
            start_bed = float(self.step_start_sample.bed_z_m)
            stage_z = float(z_m)
            stage_eta = float(sample.eta_m)
            stage_bed = float(sample.bed_z_m)
        except (TypeError, ValueError, OverflowError):
            return None
        if not all(
            math.isfinite(value)
            for value in (start_z, start_eta, start_bed, stage_z, stage_eta, stage_bed)
        ):
            return None
        if (
            start_bed > start_eta
            or start_z < start_bed
            or start_z > start_eta
            or stage_bed > stage_eta
            or stage_z <= stage_eta
        ):
            return None
        reflected_z = 2.0 * stage_eta - stage_z
        # 這裡採嚴格水柱範圍而不再使用容許帶：海面公式本身保證 reflected_z <= eta，
        # 若大步長讓鏡射位置低於 bed，就代表不能以海面條件掩蓋海床越界。
        if not math.isfinite(reflected_z) or reflected_z < stage_bed or reflected_z > stage_eta:
            return None
        return reflected_z

    @staticmethod
    def _reflected_sample_contains_z(sample: VelocitySample, reflected_z: float) -> bool:
        """確認重查的有效樣本仍以有限相容上下界支撐鏡射深度。

        真實 OCM 的 eta／bed 不隨查詢 z 改變，但合成取樣器或損毀資料可能違反此前提；
        因此品質旗標有效仍須重驗幾何，而且採嚴格水柱範圍，不新增物理緩衝。
        """

        try:
            eta = float(sample.eta_m)
            bed = float(sample.bed_z_m)
        except (TypeError, ValueError, OverflowError):
            return False
        return (
            math.isfinite(eta)
            and math.isfinite(bed)
            and bed <= eta
            and reflected_z >= bed
            and reflected_z <= eta
        )


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

    這個入口把布朗運動（Brownian）隨機位移及空間擴散的 ``+div(K)|dt|`` 漂移集中在
    同一處，讓一般 RK4 與海面中間計算點經反射速度重查後成功的路徑使用完全相同的
    擴散契約。``state`` 的時間與年齡應已代表本次確定性步驟的末端；
    本函式只改變三個公尺制位置，不再次取樣速度、不在 RK4 stage 中插入亂數，而且每次
    呼叫恰消耗一次三軸 ``normal(size=3)``。若呼叫端已判定邊界為終止狀態，禁止呼叫
    本函式，因為終止邊界定位不應在停止時間之後追加擴散。
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
