"""Numba production primitives 與純 NumPy/Python 參考介面的等價層。

此模組只加速已由解析測試固定的內層迴圈，不在 Numba 路徑重新定義物理。首批 kernel
處理 OCM 三個支撐 node、前後兩時次的垂向包夾與 barycentric/time interpolation；
域外、濕乾、月份選擇與事件仍由可稽核 Python 控制層判定。``cache=False`` 避免在正式
repository 或唯讀 SERVER 安裝目錄寫入編譯快取，部署可另以明示 Numba cache root 管理。
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=False)
def _vertical_column_kernel(
    hvel: np.ndarray,
    vertical_velocity: np.ndarray,
    zcor: np.ndarray,
    diffusivity: np.ndarray,
    time_index: int,
    node_index: int,
    target_z_m: float,
) -> tuple[np.ndarray, float, bool]:
    """在單柱搜尋上下包夾層；Numba 內部函式不直接作公開 API。"""

    layer_count = zcor.shape[2]
    lower_index = -1
    upper_index = -1
    lower_z = -np.inf
    upper_z = np.inf
    for layer in range(layer_count):
        z_value = zcor[time_index, node_index, layer]
        u_value = hvel[time_index, node_index, layer, 0]
        v_value = hvel[time_index, node_index, layer, 1]
        w_value = vertical_velocity[time_index, node_index, layer]
        k_value = diffusivity[time_index, node_index, layer]
        if not (
            np.isfinite(z_value)
            and np.isfinite(u_value)
            and np.isfinite(v_value)
            and np.isfinite(w_value)
            and np.isfinite(k_value)
        ):
            continue
        if z_value <= target_z_m and z_value > lower_z:
            lower_z = z_value
            lower_index = layer
        if z_value >= target_z_m and z_value < upper_z:
            upper_z = z_value
            upper_index = layer
    result = np.zeros(4, dtype=np.float64)
    if lower_index < 0 or upper_index < 0:
        return result, 0.0, False
    lower_values = np.array(
        [
            hvel[time_index, node_index, lower_index, 0],
            hvel[time_index, node_index, lower_index, 1],
            vertical_velocity[time_index, node_index, lower_index],
            diffusivity[time_index, node_index, lower_index],
        ],
        dtype=np.float64,
    )
    span = upper_z - lower_z
    if abs(span) <= np.finfo(np.float64).eps * 16.0:
        return lower_values, 0.0, True
    upper_values = np.array(
        [
            hvel[time_index, node_index, upper_index, 0],
            hvel[time_index, node_index, upper_index, 1],
            vertical_velocity[time_index, node_index, upper_index],
            diffusivity[time_index, node_index, upper_index],
        ],
        dtype=np.float64,
    )
    alpha = (target_z_m - lower_z) / span
    return lower_values + alpha * (upper_values - lower_values), span, True


@njit(cache=False)
def interpolate_ocm_support_numba(
    hvel: np.ndarray,
    vertical_velocity: np.ndarray,
    zcor: np.ndarray,
    diffusivity: np.ndarray,
    before_time_index: int,
    after_time_index: int,
    time_alpha: float,
    node_indices: np.ndarray,
    barycentric_weights: np.ndarray,
    target_z_m: float,
) -> tuple[np.ndarray, float, bool]:
    """加速兩時次 × 三 node 的 OCM 垂向、水平與時間內插。

    回傳 ``([u,v,w,kz], minimum_positive_bracket_span, valid)``。任一支撐柱無上下包夾即
    ``valid=False``，不重新正規化其餘 node 權重；這與純 Python reference 完全相同。
    """

    first = np.zeros(4, dtype=np.float64)
    second = np.zeros(4, dtype=np.float64)
    minimum_span = np.inf
    for support_index in range(3):
        node = int(node_indices[support_index])
        weight = barycentric_weights[support_index]
        values, span, valid = _vertical_column_kernel(
            hvel, vertical_velocity, zcor, diffusivity, before_time_index, node, target_z_m
        )
        if not valid:
            return first, 0.0, False
        first += weight * values
        if span > 0.0 and span < minimum_span:
            minimum_span = span
        if after_time_index == before_time_index:
            second += weight * values
        else:
            values, span, valid = _vertical_column_kernel(
                hvel, vertical_velocity, zcor, diffusivity, after_time_index, node, target_z_m
            )
            if not valid:
                return first, 0.0, False
            second += weight * values
            if span > 0.0 and span < minimum_span:
                minimum_span = span
    if not np.isfinite(minimum_span):
        minimum_span = 0.1
    return first + time_alpha * (second - first), max(minimum_span, 0.1), True
