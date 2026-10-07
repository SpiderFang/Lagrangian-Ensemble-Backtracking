"""WP-51 B：以「沉底時刻」重新標註季節、大小潮、潮相與事件旗標。

背景（缺陷 2）
--------------
舊流程的 ``tide_class``／``phase_or_event`` 是 48+2 觀測錨點在**觀測時刻**選出時附上的標籤
（見 ``arrival_times.select_arrival_times``），粒子實際起點卻是「觀測時刻往前推沉底年齡」的
**沉底時刻**；兩者相差最多 90 天，觀測時刻的潮況與粒子軌跡沒有任何物理關聯。``season`` 早已
改以沉底時刻計，但潮況、潮相與事件仍是觀測標籤。本模組提供以沉底 UTC 為準的標註：

- ``season``：沉底 UTC 的月份對應 ``arrival_times.SEASON_BY_MONTH``（與舊 selector 同一份對照）。
- ``tide_class``：沿用 v2 selector 的定義——對該站 OCM 潮位序列做 25 小時時窗的 rolling RMS
  （``arrival_times._rolling_rms``，去局地平均後 RMS）得到「潮汐強度」，再以**同一個
  （日曆年、季節）格內全部有限潮汐強度的中位數**切成 ``spring_proxy``（≥ 中位數）與
  ``neap_proxy``（< 中位數）。名稱與舊標籤相同，方便對照；它仍是 proxy，不宣稱現場大小潮。
- 潮相（``tide_phase``）：舊 selector 的三個相位只是在每個分層裡**挑極值時刻**
  （最大正導數＝fastest_rising、最大負導數＝fastest_falling、最小絕對導數＝slack_proxy），
  無法對任意時刻分類。本模組提出最簡單且一致的規則：以潮位的時間導數 ``d = dη/dt``
  （與 selector 相同的 ``np.gradient``）與**同一個 25 小時時窗內 |d| 的最大值** ``S``：
  ``|d| < slack_fraction × S``（預設 0.5）為 ``slack``，否則 ``d > 0`` 為 ``rising``、
  ``d < 0`` 為 ``falling``。對純弦波潮 ``|cos φ| < 1/2`` 恰佔三分之一週期，三類各約三分之一；
  三個名稱刻意不用 ``fastest_*``，因為這裡是區間分類而不是極值挑選。
- 事件旗標：``high_wave``＝NWW metric location 的 Hs **大於**該站整個序列有限 Hs 的
  ``event_quantile`` 分位數（預設 0.95）；``strong_current``＝OCM 表層流速大於同一分位數。
  舊 selector 只挑全期單一最大值，沒有門檻；分位數是本模組的提案值（見 reports/WP-51.md），
  門檻與實際數值都寫進 provenance。NWW 該時刻無有效值時 ``high_wave=False`` 並以
  ``wave_available=False`` 明示，不把缺值當成「沒有大浪」。

只用 numpy／scipy，不依賴 forcing 讀取；序列由呼叫端（``input_derivation``）以
``_SiteArrivalSelectionContext`` 的 OCM／NWW 字典提供。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any

import numpy as np
from scipy.ndimage import maximum_filter1d

from .arrival_times import DEFAULT_TIDAL_STRENGTH_WINDOW_HOURS_V2, SEASON_BY_MONTH, _rolling_rms

__all__ = [
    "DEPOSITION_LABEL_POLICY_ID",
    "DEFAULT_EVENT_QUANTILE",
    "DEFAULT_SLACK_FRACTION",
    "TIDE_CLASS_VALUES",
    "TIDE_PHASE_VALUES",
    "DEPOSITION_LABEL_METADATA_KEYS",
    "DepositionLabels",
    "DepositionLabelModel",
    "build_deposition_label_model",
    "season_for_utc_ns",
    "labels_digest",
]

DEPOSITION_LABEL_POLICY_ID = "deposition_time_labels_v1"
"""本模組標註規則的版本識別碼；規則或預設參數改變時必須遞增。"""

DEFAULT_SLACK_FRACTION = 0.5
"""``|dη/dt|`` 低於時窗內最大值的這個比例視為 ``slack``（純弦波潮時三類各佔 1/3）。"""

DEFAULT_EVENT_QUANTILE = 0.95
"""高波／強流事件旗標的門檻分位數（提案值，見模組文件）。"""

TIDE_CLASS_VALUES = ("spring_proxy", "neap_proxy", "unknown")
TIDE_PHASE_VALUES = ("rising", "falling", "slack", "unknown")

DEPOSITION_LABEL_METADATA_KEYS = (
    "deposition_tide_class",
    "deposition_tide_phase",
    "deposition_high_wave",
    "deposition_strong_current",
    "deposition_wave_available",
    "deposition_label_policy",
)
"""每筆 per_member 沉底 arrival 的 metadata 必須帶有的標籤鍵（``deposition_season`` 由 UTC 重算，
已經是 ``ArrivalTime.season``，另外也會寫入身分表標籤欄）。"""

_NS_PER_SECOND = 1_000_000_000


def season_for_utc_ns(time_utc_ns: int) -> str:
    """UTC 奈秒 → 季節字串（``SEASON_BY_MONTH``，與舊 selector 同一份對照）。"""

    return SEASON_BY_MONTH[datetime.fromtimestamp(int(time_utc_ns) / _NS_PER_SECOND, tz=UTC).month]


@dataclass(frozen=True, slots=True)
class DepositionLabels:
    """單一沉底時刻的標籤。"""

    season: str
    tide_class: str
    tide_phase: str
    high_wave: bool
    strong_current: bool
    wave_available: bool


class DepositionLabelModel:
    """單站的沉底時刻標註模型：序列預先換成規則小時軸，``labels_at`` 為 O(1) 查表。

    參數與規則見模組文件。``time_utc_ns`` 必須嚴格遞增；缺漏的小時以 NaN 填入規則軸
    （不補值、不外插），NaN 附近的潮汐強度與導數也是 NaN，對應標籤為 ``unknown``。
    """

    def __init__(
        self,
        *,
        study_site_id: str,
        time_utc_ns: np.ndarray,
        elevation_m: np.ndarray,
        significant_wave_height_m: np.ndarray,
        current_speed_mps: np.ndarray,
        tidal_strength_window_hours: float = DEFAULT_TIDAL_STRENGTH_WINDOW_HOURS_V2,
        slack_fraction: float = DEFAULT_SLACK_FRACTION,
        event_quantile: float = DEFAULT_EVENT_QUANTILE,
    ) -> None:
        times = np.asarray(time_utc_ns, dtype=np.int64)
        eta = np.asarray(elevation_m, dtype=np.float64)
        hs = np.asarray(significant_wave_height_m, dtype=np.float64)
        speed = np.asarray(current_speed_mps, dtype=np.float64)
        if times.ndim != 1 or times.size < 3 or np.any(np.diff(times) <= 0):
            raise ValueError("沉底標籤序列的 time_utc_ns 必須一維、嚴格遞增且至少 3 點")
        if not (eta.shape == hs.shape == speed.shape == times.shape):
            raise ValueError("沉底標籤序列 shape 必須相同")
        if not tidal_strength_window_hours > 0.0:
            raise ValueError("tidal_strength_window_hours 必須為正")
        if not 0.0 < slack_fraction < 1.0:
            raise ValueError("slack_fraction 必須在 (0, 1)")
        if not 0.0 < event_quantile < 1.0:
            raise ValueError("event_quantile 必須在 (0, 1)")
        self.study_site_id = str(study_site_id)
        self.tidal_strength_window_hours = float(tidal_strength_window_hours)
        self.slack_fraction = float(slack_fraction)
        self.event_quantile = float(event_quantile)

        step_ns = int(np.median(np.diff(times)))
        if step_ns <= 0:
            raise ValueError("沉底標籤序列的名義時間間隔無效")
        self._step_ns = step_ns
        start = int(times[0])
        n_axis = (int(times[-1]) - start) // step_ns + 1
        position = (times - start) // step_ns
        on_grid = (times - start) % step_ns == 0
        if not np.all(on_grid):
            raise ValueError("沉底標籤序列的時刻必須落在規則時間軸上")

        def regularise(values: np.ndarray) -> np.ndarray:
            axis = np.full(n_axis, np.nan, dtype=np.float64)
            axis[position] = values
            return axis

        self._start_ns = start
        self._n_axis = n_axis
        eta_axis = regularise(eta)
        self._hs = regularise(hs)
        self._speed = regularise(speed)
        step_seconds = step_ns / _NS_PER_SECOND
        seconds = np.arange(n_axis, dtype=np.float64) * step_seconds
        window = max(int(round(self.tidal_strength_window_hours * 3600.0 / step_seconds)), 3)
        self._window = window
        self._tidal_strength = _rolling_rms(eta_axis, window)
        derivative = np.gradient(eta_axis, seconds)
        self._derivative = derivative
        abs_deriv = np.where(np.isfinite(derivative), np.abs(derivative), 0.0)
        self._derivative_scale = maximum_filter1d(abs_deriv, size=window, mode="nearest")

        # (日曆年, 季節) 格的潮汐強度中位數（只用有限值）
        axis_ns = start + np.arange(n_axis, dtype=np.int64) * step_ns
        months = axis_ns.astype("datetime64[ns]").astype("datetime64[M]")
        years = months.astype("datetime64[Y]").astype(np.int64) + 1970
        month_of_year = (months.astype(np.int64) % 12) + 1
        self._years = years
        self._seasons = np.array([SEASON_BY_MONTH[int(m)] for m in range(1, 13)])[month_of_year - 1]
        self._cell_median: dict[tuple[int, str], float] = {}
        finite_strength = np.isfinite(self._tidal_strength)
        for year in np.unique(years):
            for season in ("DJF", "MAM", "JJA", "SON"):
                mask = finite_strength & (years == year) & (self._seasons == season)
                if np.any(mask):
                    self._cell_median[(int(year), season)] = float(np.median(self._tidal_strength[mask]))

        finite_hs = self._hs[np.isfinite(self._hs)]
        finite_speed = self._speed[np.isfinite(self._speed)]
        if finite_hs.size == 0 or finite_speed.size == 0:
            raise ValueError(f"{study_site_id} 沒有任何有限的 Hs 或流速，無法決定事件門檻")
        self.high_wave_threshold_m = float(np.quantile(finite_hs, self.event_quantile))
        self.strong_current_threshold_mps = float(np.quantile(finite_speed, self.event_quantile))

    # ── 查詢 ──────────────────────────────────────────────────────────

    def _index(self, time_utc_ns: int) -> int:
        offset = int(time_utc_ns) - self._start_ns
        if offset < 0 or offset % self._step_ns != 0:
            raise ValueError(f"{self.study_site_id} 沉底時刻 {time_utc_ns} 不在標籤序列的規則時間軸上")
        index = offset // self._step_ns
        if index >= self._n_axis:
            raise ValueError(f"{self.study_site_id} 沉底時刻 {time_utc_ns} 超出標籤序列範圍")
        return int(index)

    def labels_at(self, time_utc_ns: int) -> DepositionLabels:
        """回傳單一沉底 UTC 的標籤；時刻不在序列軸上時 fail-closed。"""

        index = self._index(time_utc_ns)
        season = str(self._seasons[index])
        strength = self._tidal_strength[index]
        median = self._cell_median.get((int(self._years[index]), season))
        if not np.isfinite(strength) or median is None:
            tide_class = "unknown"
        else:
            tide_class = "spring_proxy" if strength >= median else "neap_proxy"
        derivative = self._derivative[index]
        scale = self._derivative_scale[index]
        if not np.isfinite(derivative) or not scale > 0.0:
            tide_phase = "unknown"
        elif abs(derivative) < self.slack_fraction * scale:
            tide_phase = "slack"
        else:
            tide_phase = "rising" if derivative > 0.0 else "falling"
        hs = self._hs[index]
        speed = self._speed[index]
        wave_available = bool(np.isfinite(hs))
        return DepositionLabels(
            season=season,
            tide_class=tide_class,
            tide_phase=tide_phase,
            # 嚴格大於門檻：序列出現大量相同值（例如退化的常數場）時，不會把「等於分位數」的
            # 全部時刻誤標成事件。
            high_wave=bool(wave_available and hs > self.high_wave_threshold_m),
            strong_current=bool(np.isfinite(speed) and speed > self.strong_current_threshold_mps),
            wave_available=wave_available,
        )

    # ── provenance ────────────────────────────────────────────────────

    def provenance(self) -> dict[str, Any]:
        """可寫進 manifest 的單站模型描述（參數、門檻、各格中位數）。"""

        return {
            "study_site_id": self.study_site_id,
            "policy_id": DEPOSITION_LABEL_POLICY_ID,
            "tidal_strength_window_hours": self.tidal_strength_window_hours,
            "tidal_strength_window_samples": self._window,
            "slack_fraction": self.slack_fraction,
            "event_quantile": self.event_quantile,
            "high_wave_threshold_m": self.high_wave_threshold_m,
            "strong_current_threshold_mps": self.strong_current_threshold_mps,
            "series_start_utc_ns": int(self._start_ns),
            "series_hours": int(self._n_axis),
            "nominal_step_seconds": self._step_ns / _NS_PER_SECOND,
            "tidal_strength_cell_median_m": [
                {"year": year, "season": season, "median_m": value}
                for (year, season), value in sorted(self._cell_median.items())
            ],
        }


def build_deposition_label_model(
    *,
    study_site_id: str,
    elevation: Mapping[int, float],
    significant_wave_height: Mapping[int, float],
    current_speed: Mapping[int, float],
    wave_valid: Mapping[int, bool] | None = None,
    tidal_strength_window_hours: float = DEFAULT_TIDAL_STRENGTH_WINDOW_HOURS_V2,
    slack_fraction: float = DEFAULT_SLACK_FRACTION,
    event_quantile: float = DEFAULT_EVENT_QUANTILE,
) -> DepositionLabelModel:
    """由 ``{UTC ns: 值}`` 字典（``_SiteArrivalSelectionContext`` 的欄位）建立標註模型。

    序列時間軸取 ``elevation`` 的全部鍵（OCM 潮位的完整時間軸）；Hs／流速缺鍵視為 NaN，
    ``wave_valid`` 為 False 的時刻 Hs 視為缺值。
    """

    times = np.asarray(sorted(int(key) for key in elevation), dtype=np.int64)
    eta = np.asarray([float(elevation[int(t)]) for t in times], dtype=np.float64)
    hs = np.asarray(
        [float(significant_wave_height.get(int(t), float("nan"))) for t in times], dtype=np.float64
    )
    if wave_valid is not None:
        valid = np.asarray([bool(wave_valid.get(int(t), False)) for t in times], dtype=bool)
        hs = np.where(valid, hs, np.nan)
    speed = np.asarray([float(current_speed.get(int(t), float("nan"))) for t in times], dtype=np.float64)
    return DepositionLabelModel(
        study_site_id=study_site_id,
        time_utc_ns=times,
        elevation_m=eta,
        significant_wave_height_m=hs,
        current_speed_mps=speed,
        tidal_strength_window_hours=tidal_strength_window_hours,
        slack_fraction=slack_fraction,
        event_quantile=event_quantile,
    )


def labels_digest(models: Sequence[DepositionLabelModel]) -> str:
    """多站模型 provenance 的 SHA-256（排序後的 JSON），供 root provenance 一致性檢查。"""

    payload = json.dumps(
        [model.provenance() for model in sorted(models, key=lambda m: m.study_site_id)],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return sha256(payload.encode("utf-8")).hexdigest()
