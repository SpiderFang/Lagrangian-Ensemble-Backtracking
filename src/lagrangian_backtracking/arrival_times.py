"""由 2024–2025 forcing 衍生每站 48+2 到達時次的決定性 selector。

核心 48 時次按年、季節、spring/neap proxy 與三個潮內相位完整分層。spring/neap 使用
長窗潮位離差 RMS 的季節中位數分組；三相位分別最大正潮位導數、最小負導數與最小
絕對導數。它們是可重現的潮位 proxy，不宣稱為現場三維最大漲／退潮流。另選全期
local-domain 高波與強流各一時次，並要求與核心時次互異及具完整 backward window。
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from .scenarios import ArrivalTime, stable_identifier

SEASON_BY_MONTH = {
    12: "DJF",
    1: "DJF",
    2: "DJF",
    3: "MAM",
    4: "MAM",
    5: "MAM",
    6: "JJA",
    7: "JJA",
    8: "JJA",
    9: "SON",
    10: "SON",
    11: "SON",
}


def _rolling_rms(values: np.ndarray, window: int) -> np.ndarray:
    """以 finite convolution 計算去局地平均後 RMS，不以 0 填補缺值。

    selector 進入本函式前已由 valid mask 篩除不支援時次，但長窗仍可能跨資料缺口；
    finite count 不足一半時輸出 NaN，使該時次不能參與 spring/neap 分類。
    """

    finite = np.isfinite(values)
    numeric = np.where(finite, values, 0.0)
    kernel = np.ones(window, dtype=np.float64)
    count = np.convolve(finite.astype(np.float64), kernel, mode="same")
    mean = np.divide(
        np.convolve(numeric, kernel, mode="same"), count, out=np.full(values.shape, np.nan), where=count > 0
    )
    squared = np.where(finite, (values - mean) ** 2, 0.0)
    rms = np.sqrt(
        np.divide(
            np.convolve(squared, kernel, mode="same"),
            count,
            out=np.full(values.shape, np.nan),
            where=count >= max(window // 2, 1),
        )
    )
    return rms


def select_arrival_times(
    *,
    study_site_id: str,
    time_utc_ns: np.ndarray,
    elevation_m: np.ndarray,
    significant_wave_height_m: np.ndarray,
    current_speed_mps: np.ndarray,
    valid_forcing: np.ndarray,
    backward_window_available: np.ndarray,
    design_version: str,
    rolling_window_hours: float = 24.0 * 14.0,
) -> list[ArrivalTime]:
    """產生恰好 50 個 arrival records，資料不足立即失敗。

    所有輸入均為同一一維 UTC 軸的站點代表統計；Hs/current 可由 local-domain 有效格點
    的穩健高分位數產生，實際 aggregation 定義必須寫入 metadata。選取順序完全由數值、
    UTC 與固定 tie-break 決定，不使用亂數。
    """

    time_values = np.asarray(time_utc_ns, dtype=np.int64)
    elevation = np.asarray(elevation_m, dtype=np.float64)
    wave_height = np.asarray(significant_wave_height_m, dtype=np.float64)
    current_speed = np.asarray(current_speed_mps, dtype=np.float64)
    valid = np.asarray(valid_forcing, dtype=bool) & np.asarray(backward_window_available, dtype=bool)
    if time_values.ndim != 1 or time_values.size < 50 or np.any(np.diff(time_values) <= 0):
        raise ValueError("arrival selector time_utc_ns 必須一維、嚴格遞增且至少 50 點")
    if any(item.shape != time_values.shape for item in (elevation, wave_height, current_speed, valid)):
        raise ValueError("arrival selector 所有序列 shape 必須相同")
    finite = np.isfinite(elevation) & np.isfinite(wave_height) & np.isfinite(current_speed)
    valid &= finite
    seconds = time_values.astype(np.float64) / 1_000_000_000
    nominal_seconds = float(np.median(np.diff(seconds)))
    if nominal_seconds <= 0:
        raise ValueError("arrival selector nominal interval 無效")
    derivative = np.gradient(elevation, seconds)
    window = max(int(round(rolling_window_hours * 3_600.0 / nominal_seconds)), 3)
    tidal_strength = _rolling_rms(elevation, window)
    datetimes = [datetime.fromtimestamp(int(value) / 1_000_000_000, tz=UTC) for value in time_values]
    years = np.array([item.year for item in datetimes])
    seasons = np.array([SEASON_BY_MONTH[item.month] for item in datetimes])
    selected: set[int] = set()
    records: list[ArrivalTime] = []

    def choose(candidates: np.ndarray, score: np.ndarray, *, maximize: bool) -> int:
        """依 score 與較早 UTC tie-break 選尚未使用的有限候選。"""

        available = [
            int(index) for index in candidates if int(index) not in selected and np.isfinite(score[index])
        ]
        if not available:
            raise ValueError("arrival 分層沒有足夠且具完整 backward window 的候選")
        return min(
            available,
            key=lambda index: ((-score[index] if maximize else score[index]), int(time_values[index])),
        )

    unique_years = sorted(set(int(value) for value in years))
    if len(unique_years) != 2:
        raise ValueError(f"baseline arrival selector 預期恰好兩個年份，實際={unique_years}")
    for year in unique_years:
        for season in ("DJF", "MAM", "JJA", "SON"):
            cell = valid & (years == year) & (seasons == season) & np.isfinite(tidal_strength)
            indices = np.flatnonzero(cell)
            if indices.size < 6:
                raise ValueError(f"{study_site_id} {year}/{season} 有效候選不足 6")
            threshold = float(np.median(tidal_strength[indices]))
            class_masks = {
                "spring_proxy": indices[tidal_strength[indices] >= threshold],
                "neap_proxy": indices[tidal_strength[indices] < threshold],
            }
            for tide_class, class_indices in class_masks.items():
                phase_specs = (
                    ("fastest_rising", derivative, True),
                    ("fastest_falling", derivative, False),
                    ("slack_proxy", np.abs(derivative), False),
                )
                for phase, score, maximize in phase_specs:
                    index = choose(class_indices, score, maximize=maximize)
                    selected.add(index)
                    fields = [study_site_id, str(int(time_values[index])), tide_class, phase, design_version]
                    records.append(
                        ArrivalTime(
                            arrival_time_id=stable_identifier("arr", fields),
                            study_site_id=study_site_id,
                            time_utc_ns=int(time_values[index]),
                            year=year,
                            season=season,
                            tide_class=tide_class,
                            phase_or_event=phase,
                            metadata={
                                "elevation_m": float(elevation[index]),
                                "elevation_derivative_mps": float(derivative[index]),
                                "tidal_strength_proxy_m": float(tidal_strength[index]),
                            },
                        )
                    )

    remaining = np.flatnonzero(valid & ~np.isin(np.arange(time_values.size), list(selected)))
    for event_name, score in (("high_wave_event", wave_height), ("strong_current_event", current_speed)):
        index = choose(remaining, score, maximize=True)
        selected.add(index)
        remaining = remaining[remaining != index]
        fields = [study_site_id, str(int(time_values[index])), event_name, design_version]
        records.append(
            ArrivalTime(
                arrival_time_id=stable_identifier("arr", fields),
                study_site_id=study_site_id,
                time_utc_ns=int(time_values[index]),
                year=int(years[index]),
                season=str(seasons[index]),
                tide_class="event",
                phase_or_event=event_name,
                metadata={
                    "significant_wave_height_m": float(wave_height[index]),
                    "current_speed_mps": float(current_speed[index]),
                },
            )
        )
    if len(records) != 50 or len({item.time_utc_ns for item in records}) != 50:
        raise RuntimeError("arrival selector 未產生 50 個唯一 UTC")
    return records
