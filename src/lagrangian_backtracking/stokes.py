"""依有限水深波浪資料計算表面漂移，並轉換波向。

本方法以 NWW3 的有效波高（``Hs``）、峰值頻率（``fp``）、原始峰值波向與 OCM 當時
水深，計算波浪造成的水平表面漂移。它不能重建完整方向頻譜，因此正式成果必須另列
「不納入波浪表面漂移」與「深水近似」結果比較。波向慣例明確採「波浪從正北起順時針
吹來」；本模組不自行猜測或改變此定義。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.optimize import brentq

GRAVITY_MPS2 = 9.80665


@dataclass(frozen=True, slots=True)
class StokesResult:
    """波浪表面漂移的東、北速度及供品質檢查與敏感度分析使用的診斷量。"""

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
    """求解有限水深波浪色散關係 ``omega²=g k tanh(kh)`` 的唯一正波數。

    搜尋範圍從深水近似開始逐步放大，直到方程式兩側差值跨過零；接著使用有收斂保證的
    數值方法求根。極淺水或非有限輸入會立即拒絕，避免不可靠結果污染整條粒子軌跡。
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
        """回傳有限水深色散關係兩側的差值，供求根步驟判定方向。"""

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
    """將「從正北起順時針」的來波角度，轉為波浪傳播方向的東、北單位向量。"""

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
    """計算文件公式（7）對應的有限水深波浪表面漂移水平速度。

    ``particle_z_m``、``surface_z_m`` 與 ``bed_z_m`` 都以海面向上為正。粒子必須位於
    海床與海面之間；有效波高為零時可合法回傳零速度，但峰值頻率、波向與水深仍必須有效，
    才能區分「確實無波」和「缺少波浪資料」。水很深時為避免雙曲函數數值溢位，會改用
    已驗證的深水極限公式；這只是穩定計算方式，不是改變物理情境。
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
