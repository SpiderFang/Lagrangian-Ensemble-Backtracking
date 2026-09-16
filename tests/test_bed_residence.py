"""沉底年齡分層抽樣、觀測錨點轉換與兩種回溯時間模式測試。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from lagrangian_backtracking.bed_residence import (
    BED_RESIDENCE_MODE_FIXED_CALENDAR_WINDOW,
    BED_RESIDENCE_MODE_FULL_HORIZON_FROM_DEPOSITION,
    BED_RESIDENCE_POLICY_ID,
    BED_RESIDENCE_SAMPLING_METHOD_ID,
    BED_RESIDENCE_SAMPLING_POLICY_ID,
    apply_bed_residence_sampling,
    resolve_bed_residence_timing,
    sample_bed_residence_age_hours,
)
from lagrangian_backtracking.scenarios import ArrivalTime

_HOUR_NS = 3_600_000_000_000
_DAY_NS = 86_400_000_000_000
_MAX_AGE_DAYS = 90
_SAMPLE_COUNT = 50
_SEED = 20260916


def _iso_utc_ns(value: int) -> str:
    """為整秒測試時間建立與正式 metadata 相同的 UTC Z 文字。"""

    seconds, nanoseconds = divmod(value, 1_000_000_000)
    timestamp = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(seconds=seconds)
    suffix = f".{nanoseconds:09d}" if nanoseconds else ""
    return f"{timestamp:%Y-%m-%dT%H:%M:%S}{suffix}Z"


def _stratum(age_hours: int) -> int:
    """依正式整數邊界找出 90 日、50 分層中年齡小時的位置。"""

    total_hours = _MAX_AGE_DAYS * 24 + 1
    for index in range(_SAMPLE_COUNT):
        if index * total_hours // _SAMPLE_COUNT <= age_hours < (
            (index + 1) * total_hours // _SAMPLE_COUNT
        ):
            return index
    raise AssertionError("測試年齡超出 90 日範圍")


def _anchor_arrivals(site_id: str, *, start_ns: int) -> tuple[ArrivalTime, ...]:
    """建立一站 50 筆排序無歧義的觀測錨點。"""

    return tuple(
        ArrivalTime(
            arrival_time_id=f"{site_id}-observation-{index:02d}",
            study_site_id=site_id,
            time_utc_ns=start_ns + index * 100 * _DAY_NS,
            year=2025,
            season="SON",
            tide_class="spring_proxy" if index % 2 == 0 else "neap_proxy",
            phase_or_event=f"event-{index % 5}",
            metadata={"anchor_source": "observation"},
        )
        for index in range(_SAMPLE_COUNT)
    )


def _runtime_arrival(age_hours: int) -> ArrivalTime:
    """建立 resolver 可核對的沉底紀錄，模擬 10/31 觀測錨點。"""

    observation_dt = datetime(2025, 10, 31, tzinfo=UTC)
    observation_ns = int(observation_dt.timestamp()) * 1_000_000_000
    deposition_ns = observation_ns - age_hours * _HOUR_NS
    return ArrivalTime(
        arrival_time_id=f"arrival-age-{age_hours}",
        study_site_id="gongliao",
        time_utc_ns=deposition_ns,
        year=2025,
        season="SON",
        tide_class="spring_proxy",
        phase_or_event="strong_current_event",
        metadata={
            "observation_time_utc_ns": observation_ns,
            "observation_time_utc": _iso_utc_ns(observation_ns),
            "deposition_time_utc_ns": deposition_ns,
            "deposition_time_utc": _iso_utc_ns(deposition_ns),
            "bed_residence_age_hours": age_hours,
            "bed_residence_stratum_index": _stratum(age_hours),
            "bed_residence_policy_id": BED_RESIDENCE_POLICY_ID,
            "bed_residence_sampling_policy_id": BED_RESIDENCE_SAMPLING_POLICY_ID,
            "bed_residence_sampling_method_id": BED_RESIDENCE_SAMPLING_METHOD_ID,
            "bed_residence_sampling_seed": _SEED,
            "bed_residence_maximum_age_days": _MAX_AGE_DAYS,
            "bed_residence_design_hash": "a" * 64,
            "observation_arrival_time_id": f"observation-{age_hours}",
        },
    )


def test_age_sampler_is_reproducible_unique_and_stratified_for_all_sites() -> None:
    """同一 seed 必須產生相同 50 年齡，每個連續分層恰有一個唯一整點小時。"""

    ages = sample_bed_residence_age_hours(
        maximum_age_days=_MAX_AGE_DAYS,
        sample_count=_SAMPLE_COUNT,
        seed=_SEED,
    )
    assert ages == sample_bed_residence_age_hours(
        maximum_age_days=_MAX_AGE_DAYS,
        sample_count=_SAMPLE_COUNT,
        seed=_SEED,
    )
    assert len(ages) == len(set(ages)) == _SAMPLE_COUNT
    assert min(ages) >= 0
    assert max(ages) <= _MAX_AGE_DAYS * 24
    assert {_stratum(age) for age in ages} == set(range(_SAMPLE_COUNT))

    anchors_a = _anchor_arrivals(
        "gongliao",
        start_ns=int(datetime(2025, 10, 31, tzinfo=UTC).timestamp()) * 1_000_000_000,
    )
    anchors_b = _anchor_arrivals(
        "guishan",
        start_ns=int(datetime(2025, 10, 31, tzinfo=UTC).timestamp()) * 1_000_000_000,
    )
    converted_a = apply_bed_residence_sampling(
        anchors_a,
        age_hours=ages,
        sampling_seed=_SEED,
        maximum_age_days=_MAX_AGE_DAYS,
        design_version="design-test-v1",
    )
    converted_b = apply_bed_residence_sampling(
        tuple(reversed(anchors_b)),
        age_hours=ages,
        sampling_seed=_SEED,
        maximum_age_days=_MAX_AGE_DAYS,
        design_version="design-test-v1",
    )
    age_by_anchor_a = {
        item.metadata["observation_arrival_time_id"]: item.metadata["bed_residence_age_hours"]
        for item in converted_a
    }
    age_by_anchor_b = {
        item.metadata["observation_arrival_time_id"]: item.metadata["bed_residence_age_hours"]
        for item in converted_b
    }
    assert [
        age_by_anchor_a[f"gongliao-observation-{index:02d}"]
        for index in range(_SAMPLE_COUNT)
    ] == list(ages)
    assert [
        age_by_anchor_b[f"guishan-observation-{index:02d}"]
        for index in range(_SAMPLE_COUNT)
    ] == list(ages)
    assert age_by_anchor_a == {
        key.replace("guishan", "gongliao"): value
        for key, value in age_by_anchor_b.items()
    }


def test_conversion_preserves_observation_provenance_and_recomputes_deposition_season() -> None:
    """沉底時刻需與年齡精確相扣，season/year 依新 UTC 重算且原分層可追溯。"""

    start_ns = int(datetime(2025, 3, 1, tzinfo=UTC).timestamp()) * 1_000_000_000
    anchors = _anchor_arrivals("site-a", start_ns=start_ns)
    ages = sample_bed_residence_age_hours(
        maximum_age_days=_MAX_AGE_DAYS,
        sample_count=_SAMPLE_COUNT,
        seed=_SEED,
    )
    converted = apply_bed_residence_sampling(
        anchors,
        age_hours=ages,
        sampling_seed=_SEED,
        maximum_age_days=_MAX_AGE_DAYS,
        design_version="design-test-v1",
    )
    by_original_id = {item.metadata["observation_arrival_time_id"]: item for item in converted}
    for index, source in enumerate(anchors):
        item = by_original_id[source.arrival_time_id]
        age = ages[index]
        assert item.time_utc_ns == source.time_utc_ns - age * _HOUR_NS
        assert item.metadata["observation_time_utc_ns"] == source.time_utc_ns
        assert item.metadata["deposition_time_utc_ns"] == item.time_utc_ns
        assert item.metadata["bed_residence_age_hours"] == age
        assert item.metadata["bed_residence_stratum_index"] == _stratum(age)
        assert item.metadata["bed_residence_sampling_seed"] == _SEED
        assert item.metadata["observation_season"] == source.season
        assert item.tide_class == source.tide_class
        assert item.phase_or_event == source.phase_or_event
        assert item.year == datetime.fromtimestamp(
            item.time_utc_ns / 1_000_000_000, tz=UTC
        ).year
        assert item.season in {"DJF", "MAM", "JJA", "SON"}
        assert item.arrival_time_id != source.arrival_time_id


def test_modes_resolve_october_31_observation_and_october_17_deposition() -> None:
    """觀測 10/31、沉底 10/17、H30 時兩種模式分別得到 16 日與 30 日。"""

    arrival = _runtime_arrival(14 * 24)
    fixed = resolve_bed_residence_timing(
        arrival,
        backtrack_mode=BED_RESIDENCE_MODE_FIXED_CALENDAR_WINDOW,
        requested_horizon_days=30,
        maximum_age_days=_MAX_AGE_DAYS,
        sampling_seed=_SEED,
    )
    full = resolve_bed_residence_timing(
        arrival,
        backtrack_mode=BED_RESIDENCE_MODE_FULL_HORIZON_FROM_DEPOSITION,
        requested_horizon_days=30,
        maximum_age_days=_MAX_AGE_DAYS,
        sampling_seed=_SEED,
    )
    assert fixed.effective_horizon_ns == 16 * _DAY_NS
    assert fixed.effective_horizon_seconds == 16 * 86_400.0
    assert fixed.earliest_forcing_time_utc_ns == arrival.time_utc_ns - 16 * _DAY_NS
    assert fixed.pre_window_deposition is False
    assert full.effective_horizon_ns == 30 * _DAY_NS
    assert full.effective_horizon_seconds == 30 * 86_400.0
    assert full.earliest_forcing_time_utc_ns == arrival.time_utc_ns - 30 * _DAY_NS
    assert full.pre_window_deposition is False


@pytest.mark.parametrize("age_hours", [30 * 24, 45 * 24])
def test_fixed_window_at_or_before_horizon_is_pre_window(age_hours: int) -> None:
    """沉底年齡到達或超過固定研究窗時回傳零期間及明確 pre-window 標記。"""

    timing = resolve_bed_residence_timing(
        _runtime_arrival(age_hours),
        backtrack_mode=BED_RESIDENCE_MODE_FIXED_CALENDAR_WINDOW,
        requested_horizon_days=30,
        maximum_age_days=_MAX_AGE_DAYS,
        sampling_seed=_SEED,
    )
    assert timing.pre_window_deposition is True
    assert timing.effective_horizon_ns == 0
    assert timing.effective_horizon_seconds == 0.0
    assert timing.earliest_forcing_time_utc_ns == _runtime_arrival(age_hours).time_utc_ns


def test_runtime_resolver_fails_closed_on_age_metadata_mismatch() -> None:
    """沉底／觀測時間差與 age metadata 不符時不可猜測或自動校正。"""

    arrival = _runtime_arrival(14 * 24)
    bad_metadata = dict(arrival.metadata)
    bad_metadata["bed_residence_age_hours"] += 1
    with pytest.raises(ValueError, match="必須等於 bed residence age"):
        resolve_bed_residence_timing(
            ArrivalTime(
                arrival_time_id=arrival.arrival_time_id,
                study_site_id=arrival.study_site_id,
                time_utc_ns=arrival.time_utc_ns,
                year=arrival.year,
                season=arrival.season,
                tide_class=arrival.tide_class,
                phase_or_event=arrival.phase_or_event,
                metadata=bad_metadata,
            ),
            backtrack_mode=BED_RESIDENCE_MODE_FIXED_CALENDAR_WINDOW,
            requested_horizon_days=30,
            maximum_age_days=_MAX_AGE_DAYS,
            sampling_seed=_SEED,
        )


def test_conversion_rejects_pilot_metadata_and_duplicate_deposition_times() -> None:
    """正式沉底轉換不能混入 pilot 欄位，也不能輸出重複沉底 UTC。"""

    start_ns = int(datetime(2025, 10, 31, tzinfo=UTC).timestamp()) * 1_000_000_000
    ages = tuple(sorted(sample_bed_residence_age_hours(
        maximum_age_days=_MAX_AGE_DAYS,
        sample_count=_SAMPLE_COUNT,
        seed=_SEED,
    )))
    anchors = list(_anchor_arrivals("site-a", start_ns=start_ns))
    anchors[0] = ArrivalTime(
        arrival_time_id=anchors[0].arrival_time_id,
        study_site_id=anchors[0].study_site_id,
        time_utc_ns=anchors[0].time_utc_ns,
        year=anchors[0].year,
        season=anchors[0].season,
        tide_class=anchors[0].tide_class,
        phase_or_event=anchors[0].phase_or_event,
        metadata={"pilot_selection_scope": "demo"},
    )
    with pytest.raises(ValueError, match="pilot metadata"):
        apply_bed_residence_sampling(
            anchors,
            age_hours=ages,
            sampling_seed=_SEED,
            maximum_age_days=_MAX_AGE_DAYS,
            design_version="design-test-v1",
        )

    # 以按分層排序的 age vector 配上同步遞增觀測 UTC，讓每筆都落在同一沉底 UTC。
    duplicate_anchors = tuple(
        ArrivalTime(
            arrival_time_id=f"duplicate-{index:02d}",
            study_site_id="site-a",
            time_utc_ns=start_ns + age * _HOUR_NS,
            year=2025,
            season="SON",
            tide_class="spring_proxy",
            phase_or_event="event",
            metadata={},
        )
        for index, age in enumerate(ages)
    )
    with pytest.raises(ValueError, match="重複 UTC"):
        apply_bed_residence_sampling(
            duplicate_anchors,
            age_hours=ages,
            sampling_seed=_SEED,
            maximum_age_days=_MAX_AGE_DAYS,
            design_version="design-test-v1",
        )
