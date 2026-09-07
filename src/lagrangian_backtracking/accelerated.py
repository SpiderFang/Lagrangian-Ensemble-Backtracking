"""Numba production primitives 與純 NumPy/Python 參考介面的等價層。

此模組只加速已由解析測試固定的內層迴圈，不在 Numba 路徑重新定義物理。首批 kernel
處理 OCM 三個支撐 node、前後兩時次的垂向包夾與 barycentric/time interpolation。
移動海面時，若固定 target z 超過某一端點最高有限 ``zcor``，kernel 只暫用該端點
最高層的有限物理量作為表層控制體（surface hold）；query-time ``eta``／海床的最終
幾何 gate 仍由可稽核 Python 控制層判定，不在此 kernel 放寬海面邊界。域外、濕乾、
月份選擇與事件仍由 Python 控制層判定。``cache=False`` 避免在正式 repository 或
唯讀 SERVER 安裝目錄寫入編譯快取，部署可另以明示 Numba cache root 管理。
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
    vertical_boundary_tolerance_m: float,
) -> tuple[np.ndarray, float, bool]:
    """在單柱搜尋上下支撐層，並實作與 NumPy reference 相同的表層 hold。

    ``target_z_m`` 是固定的 query z；它可以高於某一端點的最高 ``zcor``，因為移動
    海面會讓同一固定 z 在 before／after 端點落在不同的垂向支援區間。此時只要該端點
    的最高有限 ``zcor`` layer 所需的 ``hvel``、``vertical_velocity`` 與 Kz 都有限，便
    回傳該層值與零垂向夾層跨度；真正是否位於 query-time 水柱內由外層共同幾何 gate
    決定。底層沒有對稱 hold，任何必要 layer 缺值仍回傳 ``valid=False``。
    """

    layer_count = zcor.shape[2]
    usable = np.zeros(layer_count, dtype=np.bool_)
    top_index = -1
    top_z = -np.inf
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
        if np.isfinite(z_value) and z_value > top_z:
            top_z = z_value
            top_index = layer
        usable[layer] = (
            np.isfinite(z_value)
            and np.isfinite(u_value)
            and np.isfinite(v_value)
            and np.isfinite(w_value)
            and np.isfinite(k_value)
        )
    if top_index < 0 or not np.isfinite(target_z_m):
        return np.zeros(4, dtype=np.float64), 0.0, False
    # 端點最高層以上的固定 z 不是任意最近值外插，而是明確代表 OCM 最上層控制體。
    # 最高 zcor 的必要物理量若缺值，不能退回更深層冒充表層支援。
    if target_z_m >= top_z - vertical_boundary_tolerance_m:
        if not usable[top_index]:
            return np.zeros(4, dtype=np.float64), 0.0, False
        result = np.array(
            [
                hvel[time_index, node_index, top_index, 0],
                hvel[time_index, node_index, top_index, 1],
                vertical_velocity[time_index, node_index, top_index],
                diffusivity[time_index, node_index, top_index],
            ],
            dtype=np.float64,
        )
        return result, 0.0, True
    for layer in range(layer_count):
        z_value = zcor[time_index, node_index, layer]
        if not usable[layer]:
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
    vertical_boundary_tolerance_m: float,
) -> tuple[np.ndarray, float, bool]:
    """加速兩時次 × 三 node 的 OCM 垂向、水平與時間內插。

    回傳 ``([u,v,w,kz], minimum_positive_bracket_span, valid)``。端點 target z 高於
    最高有效 ``zcor`` 時沿用最高層控制體值；底層不做對稱 hold。任一支撐柱沒有
    雙側包夾或缺少其最高層必要物理量即 ``valid=False``，不重新正規化其餘 node 權重；
    query-time ``eta``／海床的最後範圍檢查由 Python caller 與 NumPy reference 共用，
    因此這個 kernel 不會把端點 hold 誤當成海面以上的有效速度。
    """

    first = np.zeros(4, dtype=np.float64)
    second = np.zeros(4, dtype=np.float64)
    # NumPy reference 會先在每個 endpoint 內忽略零跨度的 exact-layer／surface-hold，
    # 若該 endpoint 沒有正跨度才使用 0.1 m 的保守尺度，最後再取 before／after 較小者。
    # 分開保存兩端，避免 after 全部採 surface hold 時錯誤沿用 before 的大跨度。
    first_minimum_span = np.inf
    second_minimum_span = np.inf
    for support_index in range(3):
        node = int(node_indices[support_index])
        weight = barycentric_weights[support_index]
        values, span, valid = _vertical_column_kernel(
            hvel,
            vertical_velocity,
            zcor,
            diffusivity,
            before_time_index,
            node,
            target_z_m,
            vertical_boundary_tolerance_m,
        )
        if not valid:
            return first, 0.0, False
        first += weight * values
        if span > 0.0 and span < first_minimum_span:
            first_minimum_span = span
        if after_time_index == before_time_index:
            second += weight * values
        else:
            values, span, valid = _vertical_column_kernel(
                hvel,
                vertical_velocity,
                zcor,
                diffusivity,
                after_time_index,
                node,
                target_z_m,
                vertical_boundary_tolerance_m,
            )
            if not valid:
                return first, 0.0, False
            second += weight * values
            if span > 0.0 and span < second_minimum_span:
                second_minimum_span = span
    if after_time_index == before_time_index:
        second_minimum_span = first_minimum_span
    first_vertical_scale = 0.1
    second_vertical_scale = 0.1
    if np.isfinite(first_minimum_span):
        first_vertical_scale = max(first_minimum_span, 0.1)
    if np.isfinite(second_minimum_span):
        second_vertical_scale = max(second_minimum_span, 0.1)
    vertical_scale = min(first_vertical_scale, second_vertical_scale)
    return first + time_alpha * (second - first), vertical_scale, True
