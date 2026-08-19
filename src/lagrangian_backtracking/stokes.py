"""有限水深單色 bulk Stokes drift 與方向轉換。

本方法以 NWW3 ``Hs``、``fp``、raw peak direction 與 OCM 瞬時水深建立水平 Stokes
profile。它不能重建方向頻譜，正式成果必須與 no-Stokes、deep-water 案例並列。方向
慣例由 config 明示為「自正北順時針 wave-from」，本模組不隱藏推定。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.optimize import brentq

GRAVITY_MPS2 = 9.80665


@dataclass(frozen=True, slots=True)
class StokesResult:
    """Stokes 水平速度與可供 QC/敏感度報告的診斷量。"""

    u_mps: float
    v_mps: float
    wave_number_per_m: float
    wavelength_m: float
    kh: float
    steepness_ka: float
    relative_depth: float


def solve_wave_number(
    *, angular_frequency_radps: float, water_depth_m: float, gravity_mps2: float = GRAVITY_MPS2
) -> float:
    """解 ``omega²=g k tanh(kh)`` 的唯一正根。

    bracket 由深水估計起始並倍增到殘差轉正；``brentq`` 保證 bracket 內收斂。極淺水
    或非有限輸入直接拒絕，避免 root solver 回傳可疑值後污染整條軌跡。
    """

    if not (
        math.isfinite(angular_frequency_radps)
        and angular_frequency_radps > 0
        and math.isfinite(water_depth_m)
        and water_depth_m > 0
        and math.isfinite(gravity_mps2)
        and gravity_mps2 > 0
    ):
        raise ValueError("omega、water depth 與 gravity 必須為有限正值")
    omega2 = angular_frequency_radps**2

    def residual(wave_number: float) -> float:
        """回傳有限水深 dispersion equation 左右兩側差，供 bracket root solver 使用。"""

        return gravity_mps2 * wave_number * math.tanh(wave_number * water_depth_m) - omega2

    upper = max(omega2 / gravity_mps2, angular_frequency_radps / math.sqrt(gravity_mps2 * water_depth_m))
    upper = max(upper * 2.0, 1e-12)
    while residual(upper) <= 0:
        upper *= 2.0
        if upper > 1e6:
            raise RuntimeError("無法建立 dispersion root bracket")
    root = float(brentq(residual, 0.0, upper, xtol=1e-13, rtol=1e-13, maxiter=100))
    if abs(residual(root)) > max(1e-10, omega2 * 1e-10):
        raise RuntimeError("dispersion root residual 超出容許值")
    return root


def wave_from_direction_to_unit_vector(direction_raw_deg: float) -> tuple[float, float]:
    """將自北順時針 wave-from 角度轉成傳播去向的 east/north 單位向量。"""

    if not math.isfinite(direction_raw_deg):
        raise ValueError("波向必須有限")
    theta_to_rad = math.radians((direction_raw_deg + 180.0) % 360.0)
    return math.sin(theta_to_rad), math.cos(theta_to_rad)


def finite_depth_stokes(
    *,
    significant_wave_height_m: float,
    peak_frequency_hz: float,
    direction_raw_deg: float,
    particle_z_m: float,
    surface_z_m: float,
    bed_z_m: float,
    gravity_mps2: float = GRAVITY_MPS2,
) -> StokesResult:
    """計算文件式 (7) 對應的有限水深 bulk Stokes 水平速度。

    ``particle_z_m/surface_z_m/bed_z_m`` 都是 positive-up。粒子必須位於閉區間
    ``[bed,surface]``；Hs=0 合法且回傳零速度，但 fp、方向與水深仍須有效，讓無波與
    缺波可被區分。為避免 ``sinh(kh)`` 在深水溢位，``kh>20`` 直接使用已驗證的深水
    極限；這是數值穩定轉換，不是更換 physics case。
    """

    values = [
        significant_wave_height_m,
        peak_frequency_hz,
        direction_raw_deg,
        particle_z_m,
        surface_z_m,
        bed_z_m,
    ]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Stokes 輸入必須全部有限")
    if significant_wave_height_m < 0 or peak_frequency_hz <= 0:
        raise ValueError("Hs 不可為負且 fp 必須為正")
    water_depth = surface_z_m - bed_z_m
    relative_z = particle_z_m - surface_z_m
    if water_depth <= 0 or relative_z > 1e-9 or relative_z < -water_depth - 1e-9:
        raise ValueError("粒子必須位於有效海床與海面之間")
    omega = 2.0 * math.pi * peak_frequency_hz
    wave_number = solve_wave_number(
        angular_frequency_radps=omega, water_depth_m=water_depth, gravity_mps2=gravity_mps2
    )
    amplitude = significant_wave_height_m / 2.0
    kh = wave_number * water_depth
    if significant_wave_height_m == 0:
        speed = 0.0
    elif kh > 20.0:
        speed = amplitude**2 * omega * wave_number * math.exp(2.0 * wave_number * relative_z)
    else:
        numerator = (
            amplitude**2 * omega * wave_number * math.cosh(2.0 * wave_number * (relative_z + water_depth))
        )
        speed = numerator / (2.0 * math.sinh(kh) ** 2)
    east, north = wave_from_direction_to_unit_vector(direction_raw_deg)
    return StokesResult(
        u_mps=speed * east,
        v_mps=speed * north,
        wave_number_per_m=wave_number,
        wavelength_m=2.0 * math.pi / wave_number,
        kh=kh,
        steepness_ka=wave_number * amplitude,
        relative_depth=-relative_z / water_depth,
    )


def deep_water_stokes(
    *,
    significant_wave_height_m: float,
    peak_frequency_hz: float,
    direction_raw_deg: float,
    relative_z_m: float,
    gravity_mps2: float = GRAVITY_MPS2,
) -> tuple[float, float]:
    """計算附檔式 (7) 的深水 profile，供必要敏感度與極限測試。"""

    if significant_wave_height_m < 0 or peak_frequency_hz <= 0 or relative_z_m > 0:
        raise ValueError("深水 Stokes 要求 Hs>=0、fp>0 且 relative_z<=0")
    omega = 2.0 * math.pi * peak_frequency_hz
    wave_number = omega**2 / gravity_mps2
    amplitude = significant_wave_height_m / 2.0
    speed = amplitude**2 * omega * wave_number * math.exp(2.0 * wave_number * relative_z_m)
    east, north = wave_from_direction_to_unit_vector(direction_raw_deg)
    return speed * east, speed * north
