"""OCM 缺時與重播小時的「潮汐調和＋有界殘差橋接」重建（方法 v2）。

為什麼需要新方法
----------------
``ocm_reconstruction.py`` 的方法 v1（多變量經驗正交函數＋二階自迴歸狀態空間）把缺口當成
「從兩端向缺口中央預測」的外推問題。二階自迴歸的係數只要略大於穩定邊界，預測值就會
隨預測步數指數放大；實測東北區 2025-11-19 至 11-21 的 49 小時缺口放大到 1e18 公尺／秒。
本方法 v2 改成「缺口在已知資料的內部，只做內插」：

1. **潮汐調和擬合**：以缺口前後各 ``context_hours`` 小時的真實逐時資料，對每個特徵
   （節點×層×分量）個別擬合常數、線性趨勢與一組主要分潮（``constituents``）的正弦與餘弦
   係數（普通最小平方法，附極小的嶺迴歸穩定項）。分潮頻率是天文常數，不是擬合出來的，
   所以擬合曲線在缺口內是有界的週期函數，不會發散。
2. **非潮汐殘差的雙向橋接**：擬合殘差（觀測減潮汐）包含風驅流、黑潮擺動等慢變化。缺口兩端
   各取最靠近缺口的 ``edge_hours`` 筆殘差、以距離遞減權重平均得到兩端殘差水準，缺口內的殘差
   是「兩端水準以時間線性加權的凸組合」，因此絕對值不會超過兩端殘差的較大者。
3. **有界限制**：最後把每個特徵限制在「訓練視窗內該特徵的最小值與最大值，各向外放寬
   ``envelope_margin_fraction`` 倍全距」之內；水平流速另限制每個節點層的流速大小不超過
   訓練視窗內該節點層最大流速的 ``speed_bound_factor`` 倍（方向保留）。

此方法完全沒有自迴歸遞推，不存在指數發散；輸出值必然落在訓練視窗的包絡內。

與既有契約的關係
----------------
* 輸出仍是 ``ocm_reconstruction_patch_v1`` 的稀疏 patch（九個固定陣列、manifest、sha256），
  可通過 ``validate_reconstruction_patch``；方法識別碼是 ``RECONSTRUCTION_METHOD_ID_V2``。
* ``origin_code`` 沿用 2（多來源預測），品質旗標新增本方法專屬位元
  ``QUALITY_HARMONIC_BRIDGE``（512）與 ``QUALITY_SUPERSEDES_OBSERVED``（1024）；
  既有讀取端只檢查位元 64，所以不受影響。
* **重播小時（供應者日檔是前一日整檔複本，內容不是該小時的真實流場）**：其 UTC 在觀測月份中
  仍有一列，但內容是錯的。patch 的月份 metadata 以 ``supersedes_observed.time_utc_ns``
  明列這些 UTC，讀取端（``forcing_bridge``）遇到這些時刻改用 patch 列、忽略觀測列；
  不在清單內的重疊仍一律拒絕。如此不必複製數百 GiB 的觀測月檔就能把重播列從觀測中移除。
* ``diffusivity`` 在本機資料根沒有觀測（逆向驅動層以 ``load_diffusivity=False`` 讀取），
  所以 patch 的 diffusivity 陣列是與格網同形的全 ``NaN`` 佔位檔，不是重建值。
* NaN 結構：OCM 的 hvel／zcor／vertical_velocity 在海床以下為固定的 NaN 遮罩（所有小時相同），
  重建結果沿用同一遮罩；訓練列中若有「部分小時有限、部分小時為 NaN」的特徵，輸出 NaN 並計數。

單位：流速公尺／秒，海面與垂向座標公尺；時間為世界協調時間（UTC）奈秒整數。
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .ocm_reconstruction import (
    HOURLY_NS,
    ORIGIN_RECONSTRUCTED_STATE_SPACE,
    QUALITY_NONFINITE_CELL,
    QUALITY_STATE_SPACE,
    RECONSTRUCTION_ARRAY_NAMES,
    RECONSTRUCTION_SCHEMA_VERSION,
    SUPERSEDES_OBSERVED_KEY,
    ReconstructionError,
    _array_metadata,
    _canonical_json_bytes,
    _month_for_ns,
    sha256_file,
    superseded_observed_times_from_metadata,
)

RECONSTRUCTION_METHOD_ID_V2 = "ocm_harmonic_residual_bridge_bounded_v2"
QUALITY_HARMONIC_BRIDGE = np.uint16(512)
QUALITY_SUPERSEDES_OBSERVED = np.uint16(1024)

#: 分潮角速度（度／小時）。數值來自標準天文引數表，與資料無關。
TIDAL_SPEED_DEG_PER_HOUR: dict[str, float] = {
    "M2": 28.9841042,
    "S2": 30.0,
    "N2": 28.4397295,
    "K2": 30.0821373,
    "NU2": 28.5125831,
    "MU2": 27.9682084,
    "2N2": 27.8953548,
    "L2": 29.5284789,
    "T2": 29.9589333,
    "K1": 15.0410686,
    "O1": 13.9430356,
    "P1": 14.9589314,
    "Q1": 13.3986609,
    "M4": 57.9682084,
    "MS4": 58.9841042,
    "MN4": 57.4238337,
    "S4": 60.0,
    "M6": 86.9523127,
    "MK3": 44.0251729,
    "M3": 43.4761563,
}

DEFAULT_CONSTITUENTS: tuple[str, ...] = (
    "M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1", "M4", "MS4", "MN4", "M6",
)


@dataclass(frozen=True, slots=True)
class HarmonicBridgeConfig:
    """方法 v2 的固定設定；全部進入 patch metadata 與 domain manifest。

    ``context_hours``：缺口每一側最多取用的鄰近真實資料時數（日曆視窗，視窗內若有其他
    缺口則只用實際存在的小時）。``min_context_hours`` 是每側至少需要的真實小時數，不足就
    拒絕重建（不以單側資料冒充）。``ridge_relative`` 是嶺迴歸穩定項相對於正規方程對角線平均的
    比例，只為避免分潮近乎共線時病態，數值極小。``edge_hours``／``edge_decay_hours`` 決定兩端
    殘差水準的取法。``residual_bridge=False`` 僅供驗證比較（純潮汐調和，無殘差）。
    """

    context_hours: int = 96
    min_context_hours: int = 24
    constituents: tuple[str, ...] = DEFAULT_CONSTITUENTS
    ridge_relative: float = 1.0e-8
    edge_hours: int = 3
    edge_decay_hours: float = 1.5
    residual_bridge: bool = True
    envelope_margin_fraction: float = 0.05
    speed_bound_factor: float = 1.0
    feature_block_size: int = 131_072
    wetdry_threshold: float = 0.5
    source_fingerprint: str = "unspecified"
    method_id: str = RECONSTRUCTION_METHOD_ID_V2

    def __post_init__(self) -> None:
        """在讀取大型資料前拒絕不合理的設定。"""

        if self.context_hours < 4 or self.min_context_hours < 2:
            raise ValueError("context_hours/min_context_hours 太小")
        if self.min_context_hours > self.context_hours:
            raise ValueError("min_context_hours 不得大於 context_hours")
        unknown = [name for name in self.constituents if name not in TIDAL_SPEED_DEG_PER_HOUR]
        if unknown:
            raise ValueError(f"未知分潮：{unknown}")
        if self.feature_block_size < 2 or self.feature_block_size % 2:
            raise ValueError("feature_block_size 必須是正偶數（hvel 兩分量成對）")
        if self.ridge_relative <= 0 or self.edge_hours < 1 or self.edge_decay_hours <= 0:
            raise ValueError("ridge_relative/edge_hours/edge_decay_hours 必須為正")
        if self.envelope_margin_fraction < 0 or self.speed_bound_factor <= 0:
            raise ValueError("envelope_margin_fraction 不得為負，speed_bound_factor 必須為正")


def harmonic_design(hours: np.ndarray, scale_hours: float, constituents: Sequence[str]) -> np.ndarray:
    """建立設計矩陣：常數、線性趨勢（以 ``scale_hours`` 正規化）與各分潮的餘弦／正弦欄。

    ``hours`` 是相對於缺口中點的小時數（浮點）。回傳 shape ``(len(hours), 2 + 2*K)``。
    """

    hours = np.asarray(hours, dtype=np.float64)
    columns = [np.ones_like(hours), hours / float(scale_hours)]
    for name in constituents:
        omega = np.deg2rad(TIDAL_SPEED_DEG_PER_HOUR[name])
        columns.append(np.cos(omega * hours))
        columns.append(np.sin(omega * hours))
    return np.column_stack(columns)


def bridge_predict(
    train_hours: np.ndarray,
    values: np.ndarray,
    gap_hours: np.ndarray,
    config: HarmonicBridgeConfig,
) -> np.ndarray:
    """對一個特徵區塊做「潮汐調和＋殘差橋接」，回傳缺口內的預測值（未套有界限制）。

    參數
    ----
    train_hours:
        訓練列的相對小時（嚴格遞增、全部在缺口之外；缺口之前與之後都必須有列）。
    values:
        ``(len(train_hours), F)`` 浮點，必須全部有限（呼叫端先處理 NaN 遮罩）。
    gap_hours:
        缺口內各小時的相對小時（遞增）。
    回傳
    ----
    ``(len(gap_hours), F)`` float64。
    """

    train_hours = np.asarray(train_hours, dtype=np.float64)
    gap_hours = np.asarray(gap_hours, dtype=np.float64)
    n_left = int(np.count_nonzero(train_hours < gap_hours[0]))
    n_right = int(np.count_nonzero(train_hours > gap_hours[-1]))
    if n_left == 0 or n_right == 0 or n_left + n_right != train_hours.size:
        raise ReconstructionError("訓練列必須同時位於缺口兩側且不得落在缺口內")
    scale = float(max(np.abs(train_hours).max(), 1.0))
    design = harmonic_design(train_hours, scale, config.constituents)
    design_gap = harmonic_design(gap_hours, scale, config.constituents)
    gram = design.T @ design
    gram[np.diag_indices_from(gram)] += config.ridge_relative * float(np.trace(gram)) / gram.shape[0]
    beta = np.linalg.solve(gram, design.T @ values)
    prediction = design_gap @ beta
    if config.residual_bridge:
        residual = values - design @ beta
        k = int(config.edge_hours)
        left_rows = np.arange(n_left)[-k:]
        right_rows = n_left + np.arange(n_right)[:k]
        left_dist = train_hours[left_rows[-1]] - train_hours[left_rows]
        right_dist = train_hours[right_rows] - train_hours[right_rows[0]]
        left_w = np.exp(-left_dist / config.edge_decay_hours)
        right_w = np.exp(-right_dist / config.edge_decay_hours)
        level_left = (left_w @ residual[left_rows]) / left_w.sum()
        level_right = (right_w @ residual[right_rows]) / right_w.sum()
        t_left = train_hours[left_rows[-1]]
        t_right = train_hours[right_rows[0]]
        s = np.clip((gap_hours - t_left) / (t_right - t_left), 0.0, 1.0)
        prediction = prediction + (1.0 - s)[:, None] * level_left[None, :] + s[:, None] * level_right[None, :]
    return prediction


def apply_envelope(prediction: np.ndarray, train: np.ndarray, config: HarmonicBridgeConfig) -> np.ndarray:
    """把每個特徵限制在訓練視窗的 [最小值, 最大值]（外加比例邊際）之內。"""

    low = train.min(axis=0)
    high = train.max(axis=0)
    margin = config.envelope_margin_fraction * (high - low)
    return np.clip(prediction, low - margin, high + margin)


def apply_speed_bound(prediction: np.ndarray, train: np.ndarray, config: HarmonicBridgeConfig) -> np.ndarray:
    """水平流速：每個節點層的流速大小不超過訓練視窗該節點層最大流速（方向保留）。

    特徵排列是 ``(node, layer, component)`` 展平，分量最快變化，所以偶數欄是 u、奇數欄是 v。
    """

    u_pred = prediction[:, 0::2]
    v_pred = prediction[:, 1::2]
    speed_train = np.hypot(train[:, 0::2], train[:, 1::2]).max(axis=0) * config.speed_bound_factor
    speed_pred = np.hypot(u_pred, v_pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(speed_pred > speed_train[None, :], speed_train[None, :] / speed_pred, 1.0)
    out = prediction.copy()
    out[:, 0::2] = u_pred * ratio
    out[:, 1::2] = v_pred * ratio
    return out


def reconstruct_feature_block(
    train_hours: np.ndarray,
    train_values: np.ndarray,
    gap_hours: np.ndarray,
    config: HarmonicBridgeConfig,
    *,
    kind: str,
) -> tuple[np.ndarray, int]:
    """重建一個特徵區塊，回傳 ``(預測, 不一致 NaN 特徵數)``。

    ``kind``：``"scalar"``（elev、zcor、vertical_velocity）、``"hvel"``（兩分量成對）、
    ``"wetdry"``（0／1，輸出前以 ``wetdry_threshold`` 二值化）。訓練列中任何一列為 NaN 的
    特徵一律輸出 NaN；其中「並非全部列都是 NaN」的特徵（局部有效）另行計數回傳。
    """

    finite = np.isfinite(train_values)
    all_finite = finite.all(axis=0)
    any_finite = finite.any(axis=0)
    inconsistent = int(np.count_nonzero(any_finite & ~all_finite))
    filled = np.where(finite, train_values, 0.0)
    prediction = bridge_predict(train_hours, filled, gap_hours, config)
    prediction = apply_envelope(prediction, filled, config)
    if kind == "hvel":
        prediction = apply_speed_bound(prediction, filled, config)
    elif kind == "wetdry":
        prediction = np.where(prediction >= config.wetdry_threshold, 1.0, 0.0)
    prediction[:, ~all_finite] = np.nan
    return prediction, inconsistent


def repair_zcor_order(zcor_rows: np.ndarray) -> int:
    """就地修正 zcor 的垂向順序，回傳被修正的格點數。

    zcor 沿層軸應嚴格遞增（海床到海面）。各層分別擬合與限制後，極少數節點可能出現
    相鄰層顛倒；以「由下往上取累積最大值並至少加 1e-6 公尺」修正，只動顛倒的格點。
    ``zcor_rows`` shape ``(time, node, layer)``，海床以下的 NaN 保持不動。
    """

    changed = 0
    for layer in range(1, zcor_rows.shape[2]):
        lower = zcor_rows[:, :, layer - 1]
        upper = zcor_rows[:, :, layer]
        bad = np.isfinite(lower) & np.isfinite(upper) & (upper <= lower)
        if np.any(bad):
            upper[bad] = lower[bad] + 1.0e-6
            changed += int(np.count_nonzero(bad))
    return changed


# ── 資料來源：本機月份目錄，可排除指定 UTC（重播小時） ─────────────────────


class CleanMonthsSource:
    """唯讀的 ``ocm_native`` 月份來源，建構時排除指定 UTC（視為不存在）。

    語意與 ``ocm_reconstruction.NpyDomainSource`` 相同（月界重疊 prefer-last），但不計算
    指紋，初始化便宜，適合多程序各自開啟。``exclude_utc_ns`` 是被判定為重播的小時，
    排除後這些 UTC 不在 ``times_utc_ns``，也無法被 ``read_times`` 讀取。
    """

    FIELDS = ("hvel", "vertical_velocity", "zcor", "elev", "wetdry_elem")

    def __init__(self, domain_root: str | Path, exclude_utc_ns: Sequence[int] | np.ndarray = ()) -> None:
        root = Path(domain_root)
        months_root = root / "months"
        if not months_root.is_dir():
            raise ReconstructionError(f"找不到 months：{months_root}")
        exclude = {int(v) for v in np.asarray(exclude_utc_ns, dtype=np.int64)}
        self.domain_root = root
        self.flow_id = root.name
        self._arrays: dict[str, dict[str, np.ndarray]] = {}
        self._location: dict[int, tuple[str, int]] = {}
        for directory in sorted(p for p in months_root.iterdir() if p.is_dir()):
            time_path = directory / "time_utc_ns.npy"
            if not time_path.is_file():
                continue
            times = np.asarray(np.load(time_path, mmap_mode="r"), dtype=np.int64)
            self._arrays[directory.name] = {
                name: np.load(directory / f"{name}.npy", mmap_mode="r") for name in self.FIELDS
            }
            for row, time in enumerate(times):
                self._location[int(time)] = (directory.name, row)
        for value in exclude:
            self._location.pop(value, None)
        self._times = np.asarray(sorted(self._location), dtype=np.int64)
        first = next(iter(self._arrays.values()))
        self._shapes = {name: tuple(array.shape[1:]) for name, array in first.items()}
        self._dtypes = {name: array.dtype for name, array in first.items()}

    @property
    def times_utc_ns(self) -> np.ndarray:
        return self._times.copy()

    @property
    def field_shapes(self) -> Mapping[str, tuple[int, ...]]:
        return dict(self._shapes)

    @property
    def field_dtypes(self) -> Mapping[str, np.dtype[Any]]:
        return dict(self._dtypes)

    def read_times(
        self,
        times_utc_ns: Sequence[int] | np.ndarray,
        field_name: str,
        *,
        flat_start: int | None = None,
        flat_stop: int | None = None,
    ) -> np.ndarray:
        """依 UTC 列表讀取指定欄位的展平特徵區塊（float32 原始 dtype）。"""

        total = int(np.prod(self._shapes[field_name], dtype=np.int64))
        start = 0 if flat_start is None else int(flat_start)
        stop = total if flat_stop is None else int(flat_stop)
        requested = np.asarray(times_utc_ns, dtype=np.int64)
        output = np.empty((requested.size, stop - start), dtype=self._dtypes[field_name])
        for index, time in enumerate(requested):
            try:
                month, row = self._location[int(time)]
            except KeyError as exc:
                raise ReconstructionError(f"來源沒有 UTC row：{int(time)}") from exc
            output[index] = np.asarray(self._arrays[month][field_name][row]).reshape(-1)[start:stop]
        return output


@dataclass(frozen=True, slots=True)
class GapCluster:
    """一段連續缺時（含被排除的重播小時）。

    ``missing_times_ns`` 嚴格逐時遞增；``left_time_ns``／``right_time_ns`` 是緊鄰缺口的
    兩個真實小時。``superseded_ns`` 是這一段中屬於「重播列」的 UTC（其餘為原本就缺的列）。
    """

    gap_id: str
    missing_times_ns: np.ndarray
    left_time_ns: int
    right_time_ns: int
    superseded_ns: np.ndarray


def find_gap_clusters(
    clean_times_ns: np.ndarray,
    superseded_ns: np.ndarray,
    *,
    flow_id: str,
) -> tuple[GapCluster, ...]:
    """從「已排除重播列」的真實時間軸找出所有內部缺口，每段一個 ``GapCluster``。"""

    times = np.asarray(clean_times_ns, dtype=np.int64)
    superseded = {int(v) for v in np.asarray(superseded_ns, dtype=np.int64)}
    clusters: list[GapCluster] = []
    for before, after in zip(times[:-1], times[1:], strict=True):
        difference = int(after) - int(before)
        if difference <= int(HOURLY_NS):
            continue
        if difference % int(HOURLY_NS):
            raise ReconstructionError(f"缺口不是整數小時：{int(before)}→{int(after)}")
        missing = np.arange(int(before) + int(HOURLY_NS), int(after), int(HOURLY_NS), dtype=np.int64)
        clusters.append(
            GapCluster(
                gap_id=f"{flow_id}_{int(missing[0])}_{int(missing[-1])}",
                missing_times_ns=missing,
                left_time_ns=int(before),
                right_time_ns=int(after),
                superseded_ns=np.asarray([t for t in missing if int(t) in superseded], dtype=np.int64),
            )
        )
    return tuple(clusters)


def context_times_for_cluster(
    clean_times_ns: np.ndarray, cluster: GapCluster, config: HarmonicBridgeConfig
) -> tuple[np.ndarray, np.ndarray]:
    """取缺口兩側日曆視窗內實際存在的真實小時；每側不足 ``min_context_hours`` 就拒絕。"""

    first = int(cluster.missing_times_ns[0])
    last = int(cluster.missing_times_ns[-1])
    window = int(config.context_hours) * int(HOURLY_NS)
    times = np.asarray(clean_times_ns, dtype=np.int64)
    left = times[(times >= first - window) & (times < first)]
    right = times[(times > last) & (times <= last + window)]
    if left.size < config.min_context_hours or right.size < config.min_context_hours:
        raise ReconstructionError(
            f"gap={cluster.gap_id} 兩側真實資料不足（左 {left.size}、右 {right.size}、"
            f"至少各 {config.min_context_hours}）"
        )
    return left, right


def reconstruct_cluster_field(
    source: Any,
    cluster: GapCluster,
    field_name: str,
    config: HarmonicBridgeConfig,
    *,
    sink: Any,
    flat_range: tuple[int, int] | None = None,
) -> dict[str, int]:
    """重建單一缺口的單一欄位，逐特徵區塊寫入 ``sink``。

    ``sink(start, stop, block)``：``block`` 為 ``(缺口小時數, stop-start)`` 的 float64，
    由呼叫端寫入目的陣列（可以是 memmap）。回傳統計：區塊數、局部有效特徵數。
    """

    left, right = context_times_for_cluster(source.times_utc_ns, cluster, config)
    train_times = np.concatenate((left, right))
    centre = 0.5 * (float(cluster.missing_times_ns[0]) + float(cluster.missing_times_ns[-1]))
    train_hours = (train_times.astype(np.float64) - centre) / float(HOURLY_NS)
    gap_hours = (cluster.missing_times_ns.astype(np.float64) - centre) / float(HOURLY_NS)
    shape = source.field_shapes[field_name]
    total = int(np.prod(shape, dtype=np.int64))
    kind = "hvel" if field_name == "hvel" else ("wetdry" if field_name == "wetdry_elem" else "scalar")
    start0, stop0 = (0, total) if flat_range is None else flat_range
    blocks = 0
    inconsistent = 0
    for start in range(start0, stop0, config.feature_block_size):
        stop = min(stop0, start + config.feature_block_size)
        train = source.read_times(train_times, field_name, flat_start=start, flat_stop=stop).astype(np.float64)
        block, bad = reconstruct_feature_block(train_hours, train, gap_hours, config, kind=kind)
        sink(start, stop, block)
        blocks += 1
        inconsistent += bad
    return {"blocks": blocks, "inconsistent_features": inconsistent}


# ── patch 輸出 ───────────────────────────────────────────────────────────────


def patch_month_layout(clusters: Sequence[GapCluster]) -> dict[str, dict[str, Any]]:
    """把所有缺口列依 UTC 月份分組，回傳每月的列時間與每個缺口列的 (月, 月內列號)。"""

    rows: dict[str, list[int]] = {}
    for cluster in clusters:
        for time in cluster.missing_times_ns:
            rows.setdefault(_month_for_ns(int(time)), []).append(int(time))
    layout: dict[str, dict[str, Any]] = {}
    for month, values in rows.items():
        ordered = sorted(values)
        layout[month] = {"times": np.asarray(ordered, dtype=np.int64), "index": {t: i for i, t in enumerate(ordered)}}
    return layout


def allocate_partial_month(
    partial: Path,
    times: np.ndarray,
    shapes: Mapping[str, tuple[int, ...]],
    dtypes: Mapping[str, np.dtype[Any]],
) -> None:
    """在 ``partial`` 目錄建立九個固定陣列檔（memmap，可由多程序依列範圍填入）。"""

    partial.mkdir(parents=True, exist_ok=False)
    count = int(times.size)
    np.save(partial / "time_utc_ns.npy", np.asarray(times, dtype=np.int64))
    for name in ("hvel", "vertical_velocity", "zcor", "elev"):
        array = np.lib.format.open_memmap(
            partial / f"{name}.npy", mode="w+", dtype=np.float32, shape=(count, *shapes[name])
        )
        del array
    wet = np.lib.format.open_memmap(
        partial / "wetdry_elem.npy", mode="w+", dtype=np.float32, shape=(count, *shapes["wetdry_elem"])
    )
    del wet
    diffusivity = np.lib.format.open_memmap(
        partial / "diffusivity.npy", mode="w+", dtype=np.float32, shape=(count, *shapes["vertical_velocity"])
    )
    diffusivity[:] = np.nan
    del diffusivity
    np.save(partial / "origin_code.npy", np.full(count, ORIGIN_RECONSTRUCTED_STATE_SPACE, dtype=np.uint8))


def finalize_month(
    partial: Path,
    destination: Path,
    *,
    flow_id: str,
    month: str,
    gap_ids: Sequence[str],
    superseded_ns: np.ndarray,
    quality_flags: np.ndarray,
    config: HarmonicBridgeConfig,
    source_fingerprint: str,
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    """寫入 quality_flags 與 metadata（含取代觀測清單），原子地發布月份目錄。"""

    if destination.exists():
        raise ReconstructionError(f"immutable patch 目的地已存在，拒絕覆寫：{destination}")
    np.save(partial / "quality_flags.npy", np.asarray(quality_flags, dtype=np.uint16))
    metadata: dict[str, Any] = {
        "schema_version": RECONSTRUCTION_SCHEMA_VERSION,
        "flow_id": flow_id,
        "month": month,
        "method": config.method_id,
        "gap_ids": list(gap_ids),
        "source_fingerprint": source_fingerprint,
        "configuration": asdict(config),
        "constraints": {
            "persistent_wet_only": False,
            "wetdry_semantics": "0=wet,1=dry（調和擬合後以門檻二值化）",
            "zcor_strict_increasing": True,
            "diffusivity_nonnegative": False,
            "diffusivity_note": "本機資料根無 diffusivity 觀測；陣列為全 NaN 佔位，逆向驅動層不讀取",
            "observed_arrays_untouched": True,
            "bounded_by_training_envelope": True,
        },
        SUPERSEDES_OBSERVED_KEY: {
            "policy": "supplier_daily_file_replayed_previous_day",
            "time_utc_ns": [int(v) for v in np.asarray(superseded_ns, dtype=np.int64)],
        },
        "arrays": {},
        "quality_counts": {
            str(int(flag)): int(np.count_nonzero(np.asarray(quality_flags) & flag))
            for flag in (
                np.uint16(2),
                np.uint16(8),
                QUALITY_HARMONIC_BRIDGE,
                QUALITY_SUPERSEDES_OBSERVED,
            )
        },
        "v2_details": dict(extra),
    }
    for name in RECONSTRUCTION_ARRAY_NAMES:
        path = partial / f"{name}.npy"
        metadata["arrays"][name] = _array_metadata(path, np.load(path, mmap_mode="r"))
    data = _canonical_json_bytes(metadata)
    (partial / "metadata.json").write_bytes(data)
    (partial / "metadata.json.sha256").write_text(hashlib.sha256(data).hexdigest() + "\n", encoding="ascii")
    with (partial / "metadata.json").open("rb") as stream:
        os.fsync(stream.fileno())
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial, destination)
    return metadata


def write_root_index(output_root: Path, domain_manifests: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """寫整個 patch 根的 ``reconstruction-index.json``（同 ``build_ocm_reconstruction.py``）。"""

    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "reconstruction-index.json"
    if path.exists():
        raise ReconstructionError(f"重建 index 已存在，拒絕覆寫：{path}")
    document = {
        "schema_version": "ocm_reconstruction_index_v1",
        "domains": [
            {
                "flow_id": item["flow_id"],
                "manifest_path": f"{item['flow_id']}/reconstruction-manifest.json",
                "manifest_sha256": item["manifest_sha256"],
            }
            for item in sorted(domain_manifests, key=lambda item: str(item["flow_id"]))
        ],
    }
    data = _canonical_json_bytes(document)
    partial = output_root / f".reconstruction-index.partial-{uuid.uuid4().hex}"
    partial.write_bytes(data)
    with partial.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(partial, path)
    (output_root / "reconstruction-index.json.sha256").write_text(
        hashlib.sha256(data).hexdigest() + "\n", encoding="ascii"
    )
    return document


__all__ = [
    "DEFAULT_CONSTITUENTS",
    "CleanMonthsSource",
    "GapCluster",
    "HarmonicBridgeConfig",
    "QUALITY_HARMONIC_BRIDGE",
    "QUALITY_SUPERSEDES_OBSERVED",
    "RECONSTRUCTION_METHOD_ID_V2",
    "SUPERSEDES_OBSERVED_KEY",
    "TIDAL_SPEED_DEG_PER_HOUR",
    "allocate_partial_month",
    "apply_envelope",
    "apply_speed_bound",
    "bridge_predict",
    "context_times_for_cluster",
    "find_gap_clusters",
    "finalize_month",
    "harmonic_design",
    "patch_month_layout",
    "reconstruct_cluster_field",
    "reconstruct_feature_block",
    "repair_zcor_order",
    "sha256_file",
    "superseded_observed_times_from_metadata",
    "write_root_index",
]
