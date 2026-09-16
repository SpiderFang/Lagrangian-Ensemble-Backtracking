"""新版 2025 observation 到達時次分層 selector 的純函式測試。

測試序列只用規則合成潮位／波浪／流速，目的是驗證年份範圍、48 個核心分層、兩個
replicate 的 metadata 與事件年份契約；它不代表任何 SERVER forcing 的科學結果。
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from lagrangian_backtracking.arrival_times import (
    ARRIVAL_SELECTION_POLICY_OBSERVATION_YEAR_V1,
    select_arrival_times,
)


def _two_year_three_hour_series() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """建立完整 2024–2025 forcing 軸，讓 2024 可作 lookback 而不作 anchor。"""

    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2026, 1, 1, tzinfo=UTC)
    step = timedelta(hours=3)
    count = int((end - start) / step)
    time_ns = np.asarray(
        [int((start + index * step).timestamp() * 1_000_000_000) for index in range(count)],
        dtype=np.int64,
    )
    hours = np.arange(count, dtype=np.float64) * 3.0
    elevation = (1.0 + 0.5 * np.sin(2.0 * np.pi * hours / (24.0 * 14.0))) * np.sin(
        2.0 * np.pi * hours / 12.42
    )
    wave_height = np.ones(count, dtype=np.float64)
    current_speed = np.ones(count, dtype=np.float64)
    wave_height[1000] = 8.0
    current_speed[2000] = 3.0
    return time_ns, elevation, wave_height, current_speed


def test_observation_year_selector_emits_48_plus_two_with_replicate_metadata() -> None:
    """新版 selector 只用 2025，且每個季節／潮況／相位有兩個不同 replicate UTC。"""

    time_ns, elevation, wave_height, current_speed = _two_year_three_hour_series()
    records = select_arrival_times(
        study_site_id="gongliao",
        time_utc_ns=time_ns,
        elevation_m=elevation,
        significant_wave_height_m=wave_height,
        current_speed_mps=current_speed,
        valid_forcing=np.ones(time_ns.size, dtype=bool),
        backward_window_available=np.ones(time_ns.size, dtype=bool),
        design_version="design_baseline_v3_non_rising_a_v3_local20_observation_2025_20260916",
        observation_years=[2025],
        replicates=2,
        selection_policy=ARRIVAL_SELECTION_POLICY_OBSERVATION_YEAR_V1,
    )

    assert len(records) == 50
    assert len({item.time_utc_ns for item in records}) == 50
    assert {item.year for item in records} == {2025}
    core = [item for item in records if item.tide_class != "event"]
    events = [item for item in records if item.tide_class == "event"]
    assert len(core) == 48
    assert {item.year for item in events} == {2025}
    strata = Counter(
        (item.year, item.season, item.tide_class, item.phase_or_event) for item in core
    )
    assert set(strata.values()) == {2}
    for key in strata:
        members = [
            item
            for item in core
            if (item.year, item.season, item.tide_class, item.phase_or_event) == key
        ]
        assert {item.metadata["replicate_rank"] for item in members} == {0, 1}
        assert len({item.time_utc_ns for item in members}) == 2
        assert all(item.metadata["observation_year"] == 2025 for item in members)


def test_observation_year_selector_rejects_year_outside_forcing_axis() -> None:
    """observation 年份不在 forcing 軸時必須拒絕，不能由 selector 猜測資料期。"""

    time_ns, elevation, wave_height, current_speed = _two_year_three_hour_series()
    with pytest.raises(ValueError, match="observation_years 必須是 forcing time axis 年份的子集"):
        select_arrival_times(
            study_site_id="gongliao",
            time_utc_ns=time_ns,
            elevation_m=elevation,
            significant_wave_height_m=wave_height,
            current_speed_mps=current_speed,
            valid_forcing=np.ones(time_ns.size, dtype=bool),
            backward_window_available=np.ones(time_ns.size, dtype=bool),
            design_version="observation-test",
            observation_years=[2023],
            replicates=2,
            selection_policy=ARRIVAL_SELECTION_POLICY_OBSERVATION_YEAR_V1,
        )


def test_observation_arrival_identity_uses_utc_not_axis_position() -> None:
    """前置 forcing 擴充但選中 UTC 不變時，arrival identity 不得隨陣列 index 漂移。"""

    time_ns, elevation, wave_height, current_speed = _two_year_three_hour_series()
    kwargs = {
        "study_site_id": "gongliao",
        "elevation_m": elevation,
        "significant_wave_height_m": wave_height,
        "current_speed_mps": current_speed,
        "design_version": "observation-test",
        "observation_years": [2025],
        "replicates": 2,
        "selection_policy": ARRIVAL_SELECTION_POLICY_OBSERVATION_YEAR_V1,
    }
    baseline = select_arrival_times(
        time_utc_ns=time_ns,
        valid_forcing=np.ones(time_ns.size, dtype=bool),
        backward_window_available=np.ones(time_ns.size, dtype=bool),
        **kwargs,
    )

    # 只在 forcing 軸前面增加 2023-12 的完整資料；2025 observation 選取結果應維持
    # 相同 UTC，但每個資料列在 NumPy 陣列中的 index 會整體向後平移。
    extra_count = 31 * 8  # 31 日、每 3 小時一筆
    extra_hours = np.arange(-extra_count, 0, dtype=np.float64) * 3.0
    extra_time_ns = time_ns[0] + np.arange(-extra_count, 0, dtype=np.int64) * 3_600_000_000_000
    extra_elevation = (1.0 + 0.5 * np.sin(2.0 * np.pi * extra_hours / (24.0 * 14.0))) * np.sin(
        2.0 * np.pi * extra_hours / 12.42
    )
    extended = select_arrival_times(
        time_utc_ns=np.concatenate((extra_time_ns, time_ns)),
        elevation_m=np.concatenate((extra_elevation, elevation)),
        significant_wave_height_m=np.concatenate((np.ones(extra_count), wave_height)),
        current_speed_mps=np.concatenate((np.ones(extra_count), current_speed)),
        valid_forcing=np.ones(extra_count + time_ns.size, dtype=bool),
        backward_window_available=np.ones(extra_count + time_ns.size, dtype=bool),
        study_site_id=kwargs["study_site_id"],
        design_version=kwargs["design_version"],
        observation_years=kwargs["observation_years"],
        replicates=kwargs["replicates"],
        selection_policy=kwargs["selection_policy"],
    )

    baseline_by_utc = {item.time_utc_ns: item.arrival_time_id for item in baseline}
    extended_by_utc = {item.time_utc_ns: item.arrival_time_id for item in extended}
    assert set(baseline_by_utc) == set(extended_by_utc)
    assert baseline_by_utc == extended_by_utc
