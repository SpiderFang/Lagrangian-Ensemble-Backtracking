"""以既有有限水深公式準備PI原版RK4可讀的節點／水層波浪速度。

政策finite_depth_node_fields_v1：在OCM兩端UTC，先沿既有manager選NWW全域時間列，
再於OCM節點取Hs（m）、fp（Hz）及來波方向，最後於各zcor（m，向上正）呼叫原有
有限水深Stokes核心。這不是舊逐粒子／stage時刻取樣的同一離散方案，不宣稱軌跡等價。
不使用平均週期、不替換深水公式、不讀PI tracker的k1-only波浪路徑。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numba import njit

from .accelerated import finite_depth_stokes_numba_kernel
from .forcing_window import _month_from_utc_ns
from .models import SampleQC
from .stokes import GRAVITY_MPS2, SURFACE_BOUNDARY_TOLERANCE_M

FINITE_DEPTH_NODE_FIELDS_POLICY = "finite_depth_node_fields_v1"
"""版本化離散取樣契約；物理公式不變，但不保證與舊stage-time取樣相同。"""


@dataclass(frozen=True, slots=True)
class WaveNodeFields:
    """兩端(node,layer)公尺／秒波浪場與逐節點QC及時間來源。

    缺資料節點保持NaN，node_qc保留原始拒絕原因；呼叫端只能排除該支援，不能補0。
    這些短期陣列只在準備frame時使用，合成總速度後不常駐LRU。
    """

    east: tuple[np.ndarray, np.ndarray]
    north: tuple[np.ndarray, np.ndarray]
    node_qc: np.ndarray
    metadata: dict[str, Any]


@njit(cache=False, fastmath=False)
def _stokes_layers(hs, fp, direction, z, eta, depth, node_qc):
    """逐節點／有效水層呼叫既有公式，不重寫色散、漂移或RK4。

    底部未使用的NaN zcor保持NaN，交給原版kbp語意；其他非有限與超水柱值拒絕。
    海面容差與既有finite_depth_stokes一致，只有容差內殘差可夾到海面。
    """
    east = np.full(z.shape, np.nan)
    north = np.full(z.shape, np.nan)
    qc = node_qc.copy()
    for node in range(z.shape[0]):
        if qc[node] != 0:
            continue
        for layer in range(z.shape[1]):
            if not np.isfinite(z[node, layer]):
                continue
            relative = z[node, layer] - eta[node]
            if (
                not np.isfinite(depth[node])
                or depth[node] <= 0
                or relative > SURFACE_BOUNDARY_TOLERANCE_M
                or relative < -depth[node] - 1e-9
            ):
                qc[node] |= 16  # 原SampleQC.VERTICAL_UNSUPPORTED，不混成數值失敗。
                break
            values = finite_depth_stokes_numba_kernel(
                hs[node], fp[node], direction[node], min(relative, 0.0), depth[node], GRAVITY_MPS2
            )
            if values[-1] != 0 or not np.isfinite(values[0]) or not np.isfinite(values[1]):
                qc[node] |= 64  # 原SampleQC.INVALID_PHYSICS，保留公式拒絕類別。
                break
            east[node, layer], north[node, layer] = values[0], values[1]
    return east, north, qc


def _node_wave_samples(manager, node_xy, utc_ns, *, required_month_ids=()):
    """沿既有manager全域NWW選列與四角遮罩取樣，不自訂時間補值。

    各端點先內插Hs/fp及未正規化方向圓向量，最後才轉為來向角度，保持359/1度語意。
    回傳node向量與QC、精確UTC／月份／補位來源；不取海流、不執行forcing物理請求。
    required_month_ids沿用OCM兩端來源月份，允許已知前月halo含下月UTC而下月尚無檔；
    仍由manager挑canonical列，不自行向未登錄月份外插。
    """
    count = node_xy.shape[0]
    values = np.full((count, 3), np.nan)
    qc = np.zeros(count, dtype=np.int64)
    month = _month_from_utc_ns(utc_ns)
    with manager._pin_months((month, *required_month_ids)):
        entries = manager._prepare_time_entries(
            month,
            utc_ns,
            source="nww",
            adjacent_if_missing=False,
            required_month_ids=(month, *required_month_ids),
        )
        bracket = manager._find_time_bracket(entries, utc_ns, source="nww")
        metadata = {
            "ocm_endpoint_utc_ns": int(utc_ns),
            "source": "nww_analysis",
            "time_fallback": bool(bracket.fallback_used),
            "time_fallback_policy": bracket.fallback_policy,
            **manager._time_fallback_diagnostics(bracket, utc_ns, source="nww"),
        }
        if bracket.qc != SampleQC.OK or bracket.before is None or bracket.after is None:
            qc[:] = int(bracket.qc if bracket.qc != SampleQC.OK else SampleQC.WAVE_UNSUPPORTED)
            return values, qc, metadata
        left_entry, left_index, left_ns = bracket.before
        right_entry, right_index, right_ns = bracket.after
        left, right = manager._ensure_nww(left_entry), manager._ensure_nww(right_entry)
        metadata.update(
            before_utc_ns=int(left_ns),
            after_utc_ns=int(right_ns),
            before_month_id=left_entry.month_id,
            after_month_id=right_entry.month_id,
            before_row_index=int(left_index),
            after_row_index=int(right_index),
            alpha=float(bracket.alpha),
        )
        if left is None or right is None:
            qc[:] = int(SampleQC.WAVE_UNSUPPORTED)
            return values, qc, metadata
        lon, lat = manager.projection.unproject(node_xy[:, 0], node_xy[:, 1])
        source_flags = 0
        for node in range(count):
            a = left.sample_at_time_index(float(lon[node]), float(lat[node]), left_index)
            b = (
                a
                if left is right and left_index == right_index
                else right.sample_at_time_index(float(lon[node]), float(lat[node]), right_index)
            )
            source_flags |= int(a.qc_flags | b.qc_flags)
            qc[node] = int(a.qc | b.qc)
            if qc[node] != 0:
                continue
            if any(v is None for v in (a.direction_x, a.direction_y, b.direction_x, b.direction_y)):
                qc[node] = int(SampleQC.WAVE_UNSUPPORTED)
                continue
            alpha = bracket.alpha
            hs = a.significant_wave_height_m + alpha * (
                b.significant_wave_height_m - a.significant_wave_height_m
            )
            fp = a.peak_frequency_hz + alpha * (b.peak_frequency_hz - a.peak_frequency_hz)
            dx = a.direction_x + alpha * (b.direction_x - a.direction_x)
            dy = a.direction_y + alpha * (b.direction_y - a.direction_y)
            if not np.all(np.isfinite([hs, fp, dx, dy])) or hs < 0 or fp <= 0 or abs(dx) + abs(dy) <= 1e-15:
                qc[node] = int(SampleQC.INVALID_PHYSICS)
                continue
            values[node] = hs, fp, np.degrees(np.arctan2(dx, dy)) % 360
        metadata["nww_qc_flags_union"] = source_flags
        return values, qc, metadata


def prepare_finite_depth_node_fields(
    manager,
    mesh,
    rows,
    before_utc_ns: int,
    after_utc_ns: int,
    *,
    required_month_ids=(),
) -> WaveNodeFields:
    """準備兩端波浪速度，保留既有有限水深公式與有效遮罩，不建立新時間政策。

    mesh是已縮減的OCM濕支援，rows為兩端node×layer的zcor與海面；水深為dp+eta（m）。
    回傳兩端u/v（m/s）與聯集QC；有效Hs=0才可得到零漂移，缺波浪一律NaN+原QC。
    所有RK4 stage稍後內插兩端總速度，不會只在k1加波浪或重複加入沉降。
    required_month_ids是OCM已選兩端的來源月，不一定等於UTC曆月；用於既有跨月halo
    選列。回傳陣列的暫時工作記憶體與節點×水層線性，不會保留於粒子或歷史物件。
    """
    east, north, diagnostics = [], [], []
    combined_qc = np.zeros(mesh.node_xy.shape[0], dtype=np.int64)
    for suffix, utc_ns in (("before", before_utc_ns), ("after", after_utc_ns)):
        waves, qc, metadata = _node_wave_samples(
            manager, mesh.node_xy, utc_ns, required_month_ids=required_month_ids
        )
        u, v, qc = _stokes_layers(
            waves[:, 0],
            waves[:, 1],
            waves[:, 2],
            rows[f"z_{suffix}"],
            rows[f"eta_{suffix}"],
            mesh.source_depth_m + rows[f"eta_{suffix}"],
            qc,
        )
        combined_qc |= qc
        east.append(u)
        north.append(v)
        diagnostics.append(metadata)
    return WaveNodeFields(
        tuple(east),
        tuple(north),
        combined_qc,
        {
            "field_policy": FINITE_DEPTH_NODE_FIELDS_POLICY,
            "include_stokes": True,
            "wave_parameters": "Hs_m_fp_Hz_coming_from_degrees",
            "wave_endpoint_sources": diagnostics,
            "discretization": "OCM_endpoint_node_layer_total_velocity_not_legacy_stage_sampling",
        },
    )
