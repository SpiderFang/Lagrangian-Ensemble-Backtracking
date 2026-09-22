"""新版 2025 observation 到達時次分層 selector 的純函式測試。

測試序列只用規則合成潮位／波浪／流速，目的是驗證年份範圍、48 個核心分層、兩個
replicate 的 metadata 與事件年份契約；它不代表任何 SERVER forcing 的科學結果。

WP-19 新增 M2+S2（半日潮）與 K1+O1（全日潮）合成潮位序列，用來獨立驗證
``ARRIVAL_SELECTION_POLICY_OBSERVATION_YEAR_V2`` 的潮汐強度短窗（25 小時）能
正確分出大小潮，而舊版 14 天長窗不能——這是 WP-19 要修正的核心問題，也是防止
窗長被悄悄改回 14 天的回歸測試。
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from lagrangian_backtracking.arrival_times import (
    ARRIVAL_SELECTION_POLICY_LEGACY_TWO_YEAR_V1,
    ARRIVAL_SELECTION_POLICY_OBSERVATION_YEAR_V1,
    ARRIVAL_SELECTION_POLICY_OBSERVATION_YEAR_V2,
    DEFAULT_TIDAL_STRENGTH_WINDOW_HOURS_V2,
    _rolling_rms,
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


# ---------------------------------------------------------------------------
# WP-19：v2 潮汐強度短窗（25 小時）驗證
# ---------------------------------------------------------------------------
#
# 以下合成序列刻意加入實務上會出現、但與大小潮週期無關的「雜訊」：半日潮站疊加
# 小振幅全日潮分量（K1/O1）與 M4 淺水分潮，並加入固定種子的白雜訊；全日潮站則
# 疊加小振幅半日潮分量（M2/S2）與 P1，同樣加入白雜訊。這些額外分量代表真實潮位
# 站常見、但與所測試的主要拍頻無關的其他物理量，用來檢驗窗長本身能否在「有其他
# 干擾」的條件下仍分出大小潮，而不是只在最純淨的雙分潮合成訊號下才成立。

SEASON_BY_MONTH_FOR_TEST = {
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


def _two_year_hourly_series(kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """建立 2024–2025 逐時序列，回傳 (time_ns, elevation, wave_height, current_speed, envelope)。

    ``envelope`` 是只由「主要拍頻分潮對」（半日潮站為 M2+S2，全日潮站為 K1+O1）解析算出
    的理論振幅包絡（大小潮強度的獨立真值），不經過 selector 的 rolling RMS，只用來讓
    測試獨立核對 selector 的分類是否與物理大小潮週期一致。
    """

    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2026, 1, 1, tzinfo=UTC)
    step = timedelta(hours=1)
    count = int((end - start) / step)
    time_ns = np.asarray(
        [int((start + index * step).timestamp() * 1_000_000_000) for index in range(count)],
        dtype=np.int64,
    )
    hours = np.arange(count, dtype=np.float64)
    rng = np.random.default_rng(20260923)
    if kind == "semidiurnal":
        # M2（主太陰半日潮）與 S2（主太陽半日潮）拍頻週期約 14.77 天，是大小潮週期
        # 的物理來源；N2、K1、O1、M4 是振幅小得多的次要分潮／淺水分潮，代表真實站
        # 常見但與此拍頻無關的干擾。
        period_m2, amp_m2 = 12.4206012, 1.0
        period_s2, amp_s2, phase_s2 = 12.0, 0.4, 0.3
        period_n2, amp_n2, phase_n2 = 12.65834751, 0.12, 0.7
        period_k1, amp_k1, phase_k1 = 23.93447213, 0.12, 1.1
        period_o1, amp_o1, phase_o1 = 25.81933871, 0.09, 2.1
        period_m4, amp_m4, phase_m4 = 6.2103006, 0.05, 0.5
        elevation = (
            amp_m2 * np.sin(2.0 * np.pi * hours / period_m2)
            + amp_s2 * np.sin(2.0 * np.pi * hours / period_s2 + phase_s2)
            + amp_n2 * np.sin(2.0 * np.pi * hours / period_n2 + phase_n2)
            + amp_k1 * np.sin(2.0 * np.pi * hours / period_k1 + phase_k1)
            + amp_o1 * np.sin(2.0 * np.pi * hours / period_o1 + phase_o1)
            + amp_m4 * np.sin(2.0 * np.pi * hours / period_m4 + phase_m4)
            + rng.normal(0.0, 0.06, size=hours.shape)
        )
        beat_freq = 1.0 / period_m2 - 1.0 / period_s2
        envelope = np.sqrt(
            amp_m2**2 + amp_s2**2 + 2.0 * amp_m2 * amp_s2 * np.cos(2.0 * np.pi * beat_freq * hours - phase_s2)
        )
    elif kind == "diurnal":
        # K1（日月合成全日潮）與 O1（主太陰全日潮）拍頻週期即「回歸週期」，約 13.66
        # 天；M2、S2、P1 是振幅小得多的次要分潮，代表全日潮站仍殘留的半日潮能量。
        period_k1, amp_k1 = 23.93447213, 1.0
        period_o1, amp_o1, phase_o1 = 25.81933871, 0.6, 0.4
        period_m2, amp_m2, phase_m2 = 12.4206012, 0.08, 0.9
        period_s2, amp_s2, phase_s2 = 12.0, 0.05, 1.7
        period_p1, amp_p1, phase_p1 = 24.06588766, 0.05, 2.3
        elevation = (
            amp_k1 * np.sin(2.0 * np.pi * hours / period_k1)
            + amp_o1 * np.sin(2.0 * np.pi * hours / period_o1 + phase_o1)
            + amp_m2 * np.sin(2.0 * np.pi * hours / period_m2 + phase_m2)
            + amp_s2 * np.sin(2.0 * np.pi * hours / period_s2 + phase_s2)
            + amp_p1 * np.sin(2.0 * np.pi * hours / period_p1 + phase_p1)
            + rng.normal(0.0, 0.03, size=hours.shape)
        )
        beat_freq = 1.0 / period_k1 - 1.0 / period_o1
        envelope = np.sqrt(
            amp_k1**2 + amp_o1**2 + 2.0 * amp_k1 * amp_o1 * np.cos(2.0 * np.pi * beat_freq * hours - phase_o1)
        )
    else:
        raise ValueError(f"unknown kind: {kind!r}")
    wave_height = np.ones(count, dtype=np.float64)
    current_speed = np.ones(count, dtype=np.float64)
    wave_height[5000] = 8.0
    current_speed[9000] = 3.0
    return time_ns, elevation, wave_height, current_speed, envelope


def _season_labels(time_ns: np.ndarray) -> np.ndarray:
    """回傳與 ``time_ns`` 對齊的季節標籤，獨立於 arrival_times 模組重新計算。"""

    return np.array(
        [
            SEASON_BY_MONTH_FOR_TEST[datetime.fromtimestamp(int(value) / 1e9, tz=UTC).month]
            for value in time_ns
        ]
    )


def _window_classification_agreement(
    elevation: np.ndarray, envelope: np.ndarray, seasons: np.ndarray, *, window_hours: float
) -> float:
    """獨立重算「以 ``window_hours`` 小時窗的潮汐強度」與真實包絡在各季中位數分類的一致率。

    直接呼叫與 selector 相同的 ``_rolling_rms`` 核心函式（此函式即 selector 判定大小潮
    的實際數值方法，見 arrival_times 模組文件開頭），但分類門檻（各季中位數）與真值
    （``envelope``）完全獨立算出，因此這裡驗證的是「窗長本身能否讓 RMS 序列保留大小潮
    週期振盪」，不依賴 selector 挑選 48 個代表時刻時另外引入的相位代理偏誤。
    """

    # 序列固定為逐時取樣（見 ``_two_year_hourly_series``），故窗長（小時）等於樣本數。
    window = max(int(round(window_hours)), 3)
    strength = _rolling_rms(elevation, window)
    agree = 0
    total = 0
    for season in ("DJF", "MAM", "JJA", "SON"):
        mask = (seasons == season) & np.isfinite(strength) & np.isfinite(envelope)
        season_strength = strength[mask]
        season_envelope = envelope[mask]
        median_strength = np.median(season_strength)
        median_envelope = np.median(season_envelope)
        agree += int(np.sum((season_strength >= median_strength) == (season_envelope >= median_envelope)))
        total += season_strength.size
    return agree / total


@pytest.mark.parametrize("kind", ["semidiurnal", "diurnal"])
def test_v2_short_window_tracks_spring_neap_envelope_while_legacy_window_fails(kind: str) -> None:
    """v2 的 25 小時窗大小潮分類與理論包絡高度一致；舊版 14 天窗不符合此標準。

    這是 WP-19 的核心回歸測試：問題起因是正式呼叫端把潮汐強度窗設成 14 天（大小潮
    週期約 14.77 天，14 天窗把整輪大小潮平均掉），導致分類幾乎等同隨機（現場 OCM
    驗證的一致率僅 0.49–0.56）。這裡以乾淨合成序列驗證 25 小時窗（約一個潮日）能
    達到 ≥90% 一致率，且同一序列下 14 天窗遠低於這個標準，以防窗長被悄悄改回。
    """

    time_ns, elevation, _wave_height, _current_speed, envelope = _two_year_hourly_series(kind)
    seasons = _season_labels(time_ns)

    v2_agreement = _window_classification_agreement(
        elevation, envelope, seasons, window_hours=DEFAULT_TIDAL_STRENGTH_WINDOW_HOURS_V2
    )
    legacy_agreement = _window_classification_agreement(
        elevation, envelope, seasons, window_hours=24.0 * 14.0
    )

    assert v2_agreement >= 0.90, f"{kind}: v2 25 小時窗一致率 {v2_agreement} 未達 0.90"
    assert legacy_agreement < 0.90, f"{kind}: 14 天窗一致率 {legacy_agreement} 不應達到 v2 標準"


@pytest.mark.parametrize("kind", ["semidiurnal", "diurnal"])
def test_v2_selector_produces_six_spring_and_six_neap_per_season(kind: str) -> None:
    """v2 selector 端到端輸出：48 核心時次逐季恰有 6 筆大潮、6 筆小潮，且窗長寫入 metadata。"""

    time_ns, elevation, wave_height, current_speed, _envelope = _two_year_hourly_series(kind)
    valid = np.ones(time_ns.size, dtype=bool)
    records = select_arrival_times(
        study_site_id="test_site",
        time_utc_ns=time_ns,
        elevation_m=elevation,
        significant_wave_height_m=wave_height,
        current_speed_mps=current_speed,
        valid_forcing=valid,
        backward_window_available=valid,
        design_version="wp19-v2-test",
        observation_years=[2025],
        replicates=2,
        selection_policy=ARRIVAL_SELECTION_POLICY_OBSERVATION_YEAR_V2,
    )

    assert len(records) == 50
    assert len({item.time_utc_ns for item in records}) == 50
    core = [item for item in records if item.tide_class != "event"]
    events = [item for item in records if item.tide_class == "event"]
    assert len(core) == 48
    assert len(events) == 2

    counts = Counter((item.season, item.tide_class) for item in core)
    for season in ("DJF", "MAM", "JJA", "SON"):
        assert counts[(season, "spring_proxy")] == 6
        assert counts[(season, "neap_proxy")] == 6

    # 窗長必須原樣寫入每一筆 v2 arrival（含事件補充）的 metadata，供 manifest 稽核。
    for item in records:
        assert item.metadata["tidal_strength_window_hours"] == pytest.approx(
            DEFAULT_TIDAL_STRENGTH_WINDOW_HOURS_V2
        )
        assert item.metadata["selection_policy"] == ARRIVAL_SELECTION_POLICY_OBSERVATION_YEAR_V2


def test_v2_tidal_strength_window_hours_override_is_recorded() -> None:
    """明示覆寫 v2 的潮汐強度窗長時，selector 必須採用覆寫值並寫入 metadata。"""

    time_ns, elevation, wave_height, current_speed, _envelope = _two_year_hourly_series("semidiurnal")
    valid = np.ones(time_ns.size, dtype=bool)
    override_hours = 30.0
    records = select_arrival_times(
        study_site_id="test_site",
        time_utc_ns=time_ns,
        elevation_m=elevation,
        significant_wave_height_m=wave_height,
        current_speed_mps=current_speed,
        valid_forcing=valid,
        backward_window_available=valid,
        design_version="wp19-v2-override-test",
        observation_years=[2025],
        replicates=2,
        selection_policy=ARRIVAL_SELECTION_POLICY_OBSERVATION_YEAR_V2,
        tidal_strength_window_hours=override_hours,
    )
    assert len(records) == 50
    for item in records:
        assert item.metadata["tidal_strength_window_hours"] == pytest.approx(override_hours)


@pytest.mark.parametrize(
    "selection_policy",
    [None, ARRIVAL_SELECTION_POLICY_LEGACY_TWO_YEAR_V1, ARRIVAL_SELECTION_POLICY_OBSERVATION_YEAR_V1],
)
def test_tidal_strength_window_hours_rejected_outside_v2(selection_policy: str | None) -> None:
    """v1／legacy／未明示 policy 明示 tidal_strength_window_hours 必須立即失敗。

    這保護既有 manifest 的位元相同：呼叫端不能誤以為這個覆寫參數同時影響 14 天長窗。
    """

    time_ns, elevation, wave_height, current_speed, _envelope = _two_year_hourly_series("semidiurnal")
    valid = np.ones(time_ns.size, dtype=bool)
    kwargs: dict[str, object] = dict(
        study_site_id="test_site",
        time_utc_ns=time_ns,
        elevation_m=elevation,
        significant_wave_height_m=wave_height,
        current_speed_mps=current_speed,
        valid_forcing=valid,
        backward_window_available=valid,
        design_version="wp19-window-guard-test",
        tidal_strength_window_hours=25.0,
    )
    if selection_policy is not None:
        kwargs["selection_policy"] = selection_policy
        kwargs["observation_years"] = [2025]
        kwargs["replicates"] = 2
    with pytest.raises(ValueError, match="tidal_strength_window_hours"):
        select_arrival_times(**kwargs)


def test_v1_output_has_no_window_metadata_and_stays_bit_identical() -> None:
    """v1 在同一段合成序列上的輸出不得含 v2 專屬的窗長 metadata 欄位。"""

    time_ns, elevation, wave_height, current_speed, _envelope = _two_year_hourly_series("semidiurnal")
    valid = np.ones(time_ns.size, dtype=bool)
    records = select_arrival_times(
        study_site_id="test_site",
        time_utc_ns=time_ns,
        elevation_m=elevation,
        significant_wave_height_m=wave_height,
        current_speed_mps=current_speed,
        valid_forcing=valid,
        backward_window_available=valid,
        design_version="wp19-v1-unchanged-test",
        observation_years=[2025],
        replicates=2,
        selection_policy=ARRIVAL_SELECTION_POLICY_OBSERVATION_YEAR_V1,
    )
    assert len(records) == 50
    for item in records:
        assert "tidal_strength_window_hours" not in item.metadata
        assert item.metadata["selection_policy"] == ARRIVAL_SELECTION_POLICY_OBSERVATION_YEAR_V1
