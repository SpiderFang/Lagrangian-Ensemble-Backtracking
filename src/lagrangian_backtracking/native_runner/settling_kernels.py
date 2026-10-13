"""逐粒子沉降速度的 njit 內插／平流核心（WP-54，v7 類別候選速度均衡抽樣用）。

為什麼需要這個模組
------------------
既有路徑把「這一批粒子共用的常數沉降速度」直接烤進 ``w_field``
（``backward_driver.apply_settling_velocity``：複製 ``w_field``、在
``effective_mask`` 內加上純量），BayTrace 的 ``batch_move_rk4`` 與
``batch_interpolate_velocity`` 再從這份場內插出每顆粒子的 ``w``。這個做法
要求「同一次 kernel 呼叫的所有粒子沉降速度相同」。

v7 每顆粒子有自己的沉降速度（同一類別 10 個候選值均衡分配給粒子），若沿用
烤進場的做法，每個小時必須為每個相異速度值各複製一份 ``(nvrt, nnode)``
的 ``w_field`` 並分別呼叫 kernel，每小時 100 次場複製加 100 次 kernel 呼叫，
成本遠高於既有的 10 次，違反「單一流場小時成本不得明顯上升」的要求。

本模組的解法
------------
BayTrace 的 ``batch_move_rk4`` 與 ``batch_interpolate_velocity`` 是唯讀的
njit 函式，不能修改，也不能在外部對它的 ``w`` 內插結果加修正（RK4 每個階段
的中間位置會用到 ``w``，修正必須發生在階段內部）。因此本模組把這兩個函式與
其內部使用的 ``_interp_velocity_at`` **逐行複製**成 LBT 自有版本，唯一的差異
是節點層級讀取 ``w`` 時改為::

    w_node_level = w_field[k, node]                # 沒有沉降的合成場
    if effective_mask[k, node]:
        w_node_level = w_node_level + p_settle     # 這顆粒子自己的沉降速度

這與 ``apply_settling_velocity`` 的 ``w1[effective_mask] += settling``
做的是**同一個浮點運算**（同一個 ``float64`` 加法、同樣只加在
``effective_mask`` 內），之後的垂向、時間、水平內插算式與 BayTrace 原版完全
相同，所以當所有粒子的 ``p_settle`` 等於同一個值 ``v`` 時，本模組的輸出與
「先 ``apply_settling_velocity(base, v)`` 再呼叫 BayTrace 原版 kernel」**逐位元
相同**（``tests/test_settling_kernels.py`` 以合成網格驗證）。這是本模組正確性
的核心依據：逐粒子沉降速度沒有引入任何近似，只是把「加在場上」延後到「內插
時才加」，因此不需要為每個速度值複製場。

來源與漂移防護
--------------
複製來源是 BayTrace v4.7.2 的 ``baytrace/batch_interp.py``
（``batch_interpolate_velocity``、``_interp_velocity_at``、``batch_move_rk4``）。
``build_settling_kernels`` 在建立 kernel 前，驗證已受控載入的
``baytrace.batch_interp`` 原始檔 SHA-256 等於 ``FORKED_BATCH_INTERP_SHA256``；
BayTrace 日後若更新這三個函式，雜湊不符時 fail-closed 拒絕，提醒維護者重新
核對複製內容，而不是靜默沿用過期的複本。

均勻沉降速度（``representative`` 模式）完全不經過本模組，繼續走
``apply_settling_velocity`` ＋ BayTrace 原版 kernel，確保既有成果可重現。
"""

from __future__ import annotations

import sys
from collections import namedtuple
from pathlib import Path
from typing import Any

import numpy as np
from numba import njit

__all__ = [
    "FORKED_BATCH_INTERP_SHA256",
    "SettlingKernelError",
    "SettlingKernels",
    "build_settling_kernels",
]


FORKED_BATCH_INTERP_SHA256 = "61ef2e3b2200c6e1fa3dff07d6d1bd39a9c825e5113bbb07319ab16585389712"
"""複製來源 BayTrace v4.7.2 ``baytrace/batch_interp.py`` 的 SHA-256。"""


class SettlingKernelError(RuntimeError):
    """逐粒子沉降速度核心無法建立（BayTrace 來源與複製基準不符等）。"""


SettlingKernels = namedtuple(
    "SettlingKernels", ["batch_interpolate_velocity", "batch_move_rk4", "interp_velocity_at"]
)

_KERNEL_CACHE: dict[int, SettlingKernels] = {}


def _controlled_module(handles: Any, name: str) -> Any:
    """取回 ``load_baytrace_modules`` 已受控載入的 BayTrace 子模組。"""

    module = sys.modules.get(f"baytrace.{name}")
    expected = Path(handles.package_dir) / f"{name}.py"
    if module is None or getattr(module, "__lbt_controlled_source__", None) != str(expected):
        raise SettlingKernelError(f"baytrace.{name} 不是由 load_baytrace_modules 受控載入的來源")
    return module


def build_settling_kernels(handles: Any) -> SettlingKernels:
    """建立（或取回快取的）逐粒子沉降速度 njit 核心，見模組文件。"""

    cache_key = id(handles)
    cached = _KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    actual_sha = handles.dependency_sha256.get("batch_interp")
    if actual_sha != FORKED_BATCH_INTERP_SHA256:
        raise SettlingKernelError(
            "baytrace/batch_interp.py 的 SHA-256 與本模組複製基準不符："
            f"預期 {FORKED_BATCH_INTERP_SHA256}，實際 {actual_sha}；"
            "請重新核對 settling_kernels.py 內三個複製函式與新版 BayTrace 是否仍一致"
        )

    batch_interp = _controlled_module(handles, "batch_interp")
    vertical = _controlled_module(handles, "vertical")
    geometry = _controlled_module(handles, "geometry")
    _find_element_at = batch_interp._find_element_at
    _pt_in_poly3 = geometry._pt_in_poly3
    compute_z_levels_lsc = vertical.compute_z_levels_lsc
    find_vlevel = vertical.find_vlevel

    @njit(cache=False)
    def _w_level(w_field, eff_mask, k, node_id, p_settle):
        """讀取 (k, node) 的 w，在 effective_mask 內加上這顆粒子的沉降速度。"""

        value = w_field[k, node_id]
        if eff_mask[k, node_id]:
            value = value + p_settle
        return value

    @njit(cache=False)
    def batch_interpolate_velocity(
        px, py, pz, p_elem, p_status,
        i34, elnode, grid_x, grid_y,
        nvrt, kbp, kbe, sigma_lcl, dp,
        u_field1, v_field1, w_field1, eta_field1,
        u_field2, v_field2, w_field2,
        trat, h0,
        u_out, v_out, w_out,
        eff_mask, p_settle,
    ):
        """``baytrace.batch_interp.batch_interpolate_velocity`` 的逐粒子沉降版本。

        與原版逐行相同，唯一差異：``w`` 在節點層級改由 ``_w_level`` 讀取
        （``eff_mask`` 內加上 ``p_settle[ip]``），見模組文件。``h0`` 保留以維持
        與原版相同的簽章。
        """

        n_particles = len(px)

        _vx4 = np.empty(4, dtype=np.float64)
        _vy4 = np.empty(4, dtype=np.float64)
        _anodes4 = np.empty(4, dtype=np.int64)
        _nodes3 = np.empty(3, dtype=np.int64)
        _zlev = np.empty(nvrt, dtype=np.float64)

        for ip in range(n_particles):
            if p_status[ip] != 1:
                u_out[ip] = 0.0
                v_out[ip] = 0.0
                w_out[ip] = 0.0
                continue

            elem = p_elem[ip]
            if elem < 0:
                u_out[ip] = 0.0
                v_out[ip] = 0.0
                w_out[ip] = 0.0
                continue

            n_nodes = i34[elem]
            for j in range(n_nodes):
                nid = elnode[elem, j]
                _anodes4[j] = nid
                _vx4[j] = grid_x[nid]
                _vy4[j] = grid_y[nid]

            arco, nodel = _pt_in_poly3(n_nodes, _vx4[:n_nodes], _vy4[:n_nodes], px[ip], py[ip])

            for j in range(3):
                _nodes3[j] = _anodes4[nodel[j]]

            for k in range(nvrt):
                _zlev[k] = 0.0
            for i in range(3):
                node_id = _nodes3[i]
                node_z = compute_z_levels_lsc(
                    nvrt, kbp[node_id], sigma_lcl[:, node_id], dp[node_id], eta_field1[node_id]
                )
                for k in range(nvrt):
                    _zlev[k] += arco[i] * node_z[k]

            k_level, z_weight = find_vlevel(pz[ip], _zlev, nvrt)

            kbe_e = kbe[elem]
            if k_level < kbe_e:
                k_level = kbe_e
                z_weight = 0.0

            settle = p_settle[ip]
            u_val = 0.0
            v_val = 0.0
            w_val = 0.0

            for i in range(3):
                node_id = _nodes3[i]

                if k_level < nvrt - 1:
                    u_n1 = (u_field1[k_level, node_id] * (1.0 - z_weight) +
                            u_field1[k_level + 1, node_id] * z_weight)
                    v_n1 = (v_field1[k_level, node_id] * (1.0 - z_weight) +
                            v_field1[k_level + 1, node_id] * z_weight)
                    w_n1 = (_w_level(w_field1, eff_mask, k_level, node_id, settle) * (1.0 - z_weight) +
                            _w_level(w_field1, eff_mask, k_level + 1, node_id, settle) * z_weight)
                else:
                    u_n1 = u_field1[k_level, node_id]
                    v_n1 = v_field1[k_level, node_id]
                    w_n1 = _w_level(w_field1, eff_mask, k_level, node_id, settle)

                if k_level < nvrt - 1:
                    u_n2 = (u_field2[k_level, node_id] * (1.0 - z_weight) +
                            u_field2[k_level + 1, node_id] * z_weight)
                    v_n2 = (v_field2[k_level, node_id] * (1.0 - z_weight) +
                            v_field2[k_level + 1, node_id] * z_weight)
                    w_n2 = (_w_level(w_field2, eff_mask, k_level, node_id, settle) * (1.0 - z_weight) +
                            _w_level(w_field2, eff_mask, k_level + 1, node_id, settle) * z_weight)
                else:
                    u_n2 = u_field2[k_level, node_id]
                    v_n2 = v_field2[k_level, node_id]
                    w_n2 = _w_level(w_field2, eff_mask, k_level, node_id, settle)

                u_node = u_n1 * (1.0 - trat) + u_n2 * trat
                v_node = v_n1 * (1.0 - trat) + v_n2 * trat
                w_node = w_n1 * (1.0 - trat) + w_n2 * trat

                u_val += arco[i] * u_node
                v_val += arco[i] * v_node
                w_val += arco[i] * w_node

            u_out[ip] = u_val
            v_out[ip] = v_val
            w_out[ip] = w_val

    @njit(cache=False)
    def interp_velocity_at(
        px, py, pz, elem,
        i34, elnode, grid_x, grid_y,
        nvrt, kbp, kbe, sigma_lcl, dp,
        u_field1, v_field1, w_field1, eta_field1,
        u_field2, v_field2, w_field2, trat,
        eff_mask, settle,
    ):
        """``baytrace.batch_interp._interp_velocity_at`` 的逐粒子沉降版本（單點）。"""

        _vx4 = np.empty(4, dtype=np.float64)
        _vy4 = np.empty(4, dtype=np.float64)
        _anodes4 = np.empty(4, dtype=np.int64)
        _nodes3 = np.empty(3, dtype=np.int64)
        _zlev = np.empty(nvrt, dtype=np.float64)

        n_nodes = i34[elem]
        for j in range(n_nodes):
            nid = elnode[elem, j]
            _anodes4[j] = nid
            _vx4[j] = grid_x[nid]
            _vy4[j] = grid_y[nid]

        arco, nodel = _pt_in_poly3(n_nodes, _vx4[:n_nodes], _vy4[:n_nodes], px, py)

        for j in range(3):
            _nodes3[j] = _anodes4[nodel[j]]

        for k in range(nvrt):
            _zlev[k] = 0.0
        for i in range(3):
            node_id = _nodes3[i]
            node_z = compute_z_levels_lsc(
                nvrt, kbp[node_id], sigma_lcl[:, node_id], dp[node_id], eta_field1[node_id]
            )
            for k in range(nvrt):
                _zlev[k] += arco[i] * node_z[k]

        k_level, z_weight = find_vlevel(pz, _zlev, nvrt)
        kbe_e = kbe[elem]
        if k_level < kbe_e:
            k_level = kbe_e
            z_weight = 0.0

        u_val = 0.0
        v_val = 0.0
        w_val = 0.0
        for i in range(3):
            node_id = _nodes3[i]
            if k_level < nvrt - 1:
                u_n1 = u_field1[k_level, node_id] * (1.0 - z_weight) + u_field1[k_level + 1, node_id] * z_weight
                v_n1 = v_field1[k_level, node_id] * (1.0 - z_weight) + v_field1[k_level + 1, node_id] * z_weight
                w_n1 = (_w_level(w_field1, eff_mask, k_level, node_id, settle) * (1.0 - z_weight) +
                        _w_level(w_field1, eff_mask, k_level + 1, node_id, settle) * z_weight)
                u_n2 = u_field2[k_level, node_id] * (1.0 - z_weight) + u_field2[k_level + 1, node_id] * z_weight
                v_n2 = v_field2[k_level, node_id] * (1.0 - z_weight) + v_field2[k_level + 1, node_id] * z_weight
                w_n2 = (_w_level(w_field2, eff_mask, k_level, node_id, settle) * (1.0 - z_weight) +
                        _w_level(w_field2, eff_mask, k_level + 1, node_id, settle) * z_weight)
            else:
                u_n1 = u_field1[k_level, node_id]
                v_n1 = v_field1[k_level, node_id]
                w_n1 = _w_level(w_field1, eff_mask, k_level, node_id, settle)
                u_n2 = u_field2[k_level, node_id]
                v_n2 = v_field2[k_level, node_id]
                w_n2 = _w_level(w_field2, eff_mask, k_level, node_id, settle)

            u_val += arco[i] * (u_n1 * (1.0 - trat) + u_n2 * trat)
            v_val += arco[i] * (v_n1 * (1.0 - trat) + v_n2 * trat)
            w_val += arco[i] * (w_n1 * (1.0 - trat) + w_n2 * trat)

        return u_val, v_val, w_val

    @njit(cache=False)
    def batch_move_rk4(
        px, py, pz, p_u, p_v, p_w,
        p_x_prev, p_y_prev, p_elem, p_status,
        x0, y0, z0,
        i34, elnode, grid_x, grid_y, ic3,
        nvrt, kbp, kbe, sigma_lcl, dp,
        u_field1, v_field1, w_field1, eta_field1,
        u_field2, v_field2, w_field2,
        ibf, dtb, trat,
        eff_mask, p_settle,
    ):
        """``baytrace.batch_interp.batch_move_rk4`` 的逐粒子沉降版本。

        與原版逐行相同；四個階段的內插改呼叫 ``interp_velocity_at``，並帶入該
        粒子的 ``p_settle[ip]``。第一階段 ``k1`` 讀 ``p_u/p_v/p_w``（上一次
        ``batch_interpolate_velocity`` 已含沉降速度的結果），與原版相同。
        """

        n = len(px)
        for ip in range(n):
            if p_status[ip] != 1:
                continue

            p_x_prev[ip] = x0[ip]
            p_y_prev[ip] = y0[ip]
            elem = p_elem[ip]
            settle = p_settle[ip]

            k1u = ibf * p_u[ip]
            k1v = ibf * p_v[ip]
            k1w = ibf * p_w[ip]

            xh = x0[ip] + 0.5 * dtb * k1u
            yh = y0[ip] + 0.5 * dtb * k1v
            zh = z0[ip] + 0.5 * dtb * k1w
            eh = _find_element_at(xh, yh, elem, i34, elnode, grid_x, grid_y, ic3)
            u2, v2, w2 = interp_velocity_at(
                xh, yh, zh, eh, i34, elnode, grid_x, grid_y,
                nvrt, kbp, kbe, sigma_lcl, dp,
                u_field1, v_field1, w_field1, eta_field1,
                u_field2, v_field2, w_field2, trat, eff_mask, settle)
            k2u = ibf * u2
            k2v = ibf * v2
            k2w = ibf * w2

            xh = x0[ip] + 0.5 * dtb * k2u
            yh = y0[ip] + 0.5 * dtb * k2v
            zh = z0[ip] + 0.5 * dtb * k2w
            eh = _find_element_at(xh, yh, elem, i34, elnode, grid_x, grid_y, ic3)
            u3, v3, w3 = interp_velocity_at(
                xh, yh, zh, eh, i34, elnode, grid_x, grid_y,
                nvrt, kbp, kbe, sigma_lcl, dp,
                u_field1, v_field1, w_field1, eta_field1,
                u_field2, v_field2, w_field2, trat, eff_mask, settle)
            k3u = ibf * u3
            k3v = ibf * v3
            k3w = ibf * w3

            xe = x0[ip] + dtb * k3u
            ye = y0[ip] + dtb * k3v
            ze = z0[ip] + dtb * k3w
            eh = _find_element_at(xe, ye, elem, i34, elnode, grid_x, grid_y, ic3)
            u4, v4, w4 = interp_velocity_at(
                xe, ye, ze, eh, i34, elnode, grid_x, grid_y,
                nvrt, kbp, kbe, sigma_lcl, dp,
                u_field1, v_field1, w_field1, eta_field1,
                u_field2, v_field2, w_field2, trat, eff_mask, settle)
            k4u = ibf * u4
            k4v = ibf * v4
            k4w = ibf * w4

            px[ip] = x0[ip] + dtb / 6.0 * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
            py[ip] = y0[ip] + dtb / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
            pz[ip] = z0[ip] + dtb / 6.0 * (k1w + 2.0 * k2w + 2.0 * k3w + k4w)

    kernels = SettlingKernels(
        batch_interpolate_velocity=batch_interpolate_velocity,
        batch_move_rk4=batch_move_rk4,
        interp_velocity_at=interp_velocity_at,
    )
    _KERNEL_CACHE[cache_key] = kernels
    return kernels
