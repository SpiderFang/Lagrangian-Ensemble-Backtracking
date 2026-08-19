"""常數擴散參考、Smagorinsky 候選與 adaptive time-step 限制。

隨機位移使用 ``sqrt(2K|dt|)``，永遠不把 backward 設為負 diffusivity，也不把亂數
增量放入 RK4 stage。空變 K 的 gradient drift 尚未在此啟用；Smagorinsky 函式只輸出
候選 Kh 與 cap/floor 診斷，正式 baseline 仍需 well-mixed 驗證。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class DiffusionCoefficients:
    """公尺平方每秒的對角擴散係數。"""

    kx_m2ps: float
    ky_m2ps: float
    kz_m2ps: float

    def validate(self) -> None:
        """負值或非有限 K 代表病態模型，必須在產生亂數前拒絕。"""

        values = (self.kx_m2ps, self.ky_m2ps, self.kz_m2ps)
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError("Kx/Ky/Kz 必須是有限非負 m²/s")


@dataclass(frozen=True, slots=True)
class TimeStepDecision:
    """adaptive step 的絕對秒數與主導限制，方向由 caller 另加正負號。"""

    seconds: float
    limiting_reason: str


def brownian_displacement(
    coefficients: DiffusionCoefficients, *, dt_seconds: float, rng: np.random.Generator
) -> np.ndarray:
    """產生三軸獨立的 Euler–Maruyama 位移，shape 固定為 ``(3,)`` 公尺。"""

    coefficients.validate()
    if not math.isfinite(dt_seconds) or dt_seconds == 0:
        raise ValueError("dt_seconds 必須是有限非零值")
    diffusivity = np.array(
        [coefficients.kx_m2ps, coefficients.ky_m2ps, coefficients.kz_m2ps], dtype=np.float64
    )
    return rng.normal(size=3) * np.sqrt(2.0 * diffusivity * abs(dt_seconds))


def smagorinsky_horizontal_diffusivity(
    *,
    du_dx_per_s: float,
    du_dy_per_s: float,
    dv_dx_per_s: float,
    dv_dy_per_s: float,
    triangle_area_m2: float,
    coefficient_cs: float,
    floor_m2ps: float | None = None,
    cap_m2ps: float | None = None,
) -> tuple[float, bool, bool]:
    """依文件式 (10) 計算 triangle 內候選 Kh 與 floor/cap 命中狀態。"""

    values = [du_dx_per_s, du_dy_per_s, dv_dx_per_s, dv_dy_per_s, triangle_area_m2, coefficient_cs]
    if not all(math.isfinite(value) for value in values) or triangle_area_m2 <= 0 or coefficient_cs < 0:
        raise ValueError("速度梯度需有限、triangle area 正值且 Cs 非負")
    delta_m = math.sqrt(triangle_area_m2)
    strain = math.sqrt((du_dx_per_s - dv_dy_per_s) ** 2 + (dv_dx_per_s + du_dy_per_s) ** 2)
    value = (coefficient_cs * delta_m) ** 2 * strain
    hit_floor = floor_m2ps is not None and value < floor_m2ps
    hit_cap = cap_m2ps is not None and value > cap_m2ps
    if floor_m2ps is not None:
        value = max(value, floor_m2ps)
    if cap_m2ps is not None:
        value = min(value, cap_m2ps)
    return value, hit_floor, hit_cap


def choose_time_step(
    *,
    speed_horizontal_mps: float,
    speed_vertical_mps: float,
    horizontal_scale_m: float,
    vertical_scale_m: float,
    coefficients: DiffusionCoefficients,
    dt_min_seconds: float,
    dt_max_seconds: float,
    seconds_to_forcing_boundary: float | None = None,
    advective_fraction: float = 0.25,
    vertical_fraction: float = 0.25,
    diffusive_fraction: float = 0.25,
) -> TimeStepDecision:
    """由水平平流、垂向平流、擴散與 forcing 邊界選最小合法步長。

    ``seconds_to_forcing_boundary`` 是沿目前積分方向到下一資料時界的正秒數。若所需步長
    低於 ``dt_min``，函式仍回傳 ``dt_min`` 並標示 ``minimum_clamp``；engine 必須計數，
    超過核定次數時停止為 numerical failure，而不是無限縮步。
    """

    coefficients.validate()
    numeric = [
        speed_horizontal_mps,
        speed_vertical_mps,
        horizontal_scale_m,
        vertical_scale_m,
        dt_min_seconds,
        dt_max_seconds,
    ]
    if not all(math.isfinite(value) and value >= 0 for value in numeric):
        raise ValueError("time-step 輸入必須有限非負")
    if (
        horizontal_scale_m <= 0
        or vertical_scale_m <= 0
        or dt_min_seconds <= 0
        or dt_max_seconds < dt_min_seconds
    ):
        raise ValueError("尺度與 dt 範圍無效")
    candidates: list[tuple[float, str]] = [(dt_max_seconds, "maximum")]
    if speed_horizontal_mps > 0:
        candidates.append(
            (advective_fraction * horizontal_scale_m / speed_horizontal_mps, "horizontal_advection")
        )
    if speed_vertical_mps > 0:
        candidates.append((vertical_fraction * vertical_scale_m / speed_vertical_mps, "vertical_advection"))
    maximum_k = max(coefficients.kx_m2ps, coefficients.ky_m2ps, coefficients.kz_m2ps)
    if maximum_k > 0:
        scale = min(horizontal_scale_m, vertical_scale_m)
        candidates.append(((diffusive_fraction * scale) ** 2 / (2.0 * maximum_k), "diffusion"))
    if seconds_to_forcing_boundary is not None:
        if not math.isfinite(seconds_to_forcing_boundary) or seconds_to_forcing_boundary <= 0:
            raise ValueError("seconds_to_forcing_boundary 必須是有限正值")
        candidates.append((seconds_to_forcing_boundary, "forcing_boundary"))
    seconds, reason = min(candidates, key=lambda item: item[0])
    if seconds < dt_min_seconds:
        return TimeStepDecision(dt_min_seconds, "minimum_clamp")
    return TimeStepDecision(seconds, reason)
