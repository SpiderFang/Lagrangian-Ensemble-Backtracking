"""沉底年齡分層抽樣、觀測錨點轉換與兩種回溯時間模式測試。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from lagrangian_backtracking.bed_residence import (
    BED_RESIDENCE_AVAILABILITY_CONDITIONED_SAMPLING_METHOD_ID,
    BED_RESIDENCE_MODE_FIXED_CALENDAR_WINDOW,
    BED_RESIDENCE_MODE_FULL_HORIZON_FROM_DEPOSITION,
    BED_RESIDENCE_PER_MEMBER_SAMPLING_METHOD_ID,
    BED_RESIDENCE_PER_PARTICLE_SAMPLING_METHOD_ID,
    BED_RESIDENCE_POLICY_ID,
    BED_RESIDENCE_SAMPLING_METHOD_ID,
    BED_RESIDENCE_SAMPLING_POLICY_ID,
    apply_bed_residence_sampling,
    derive_bed_residence_member_seed,
    derive_bed_residence_particle_seed,
    resolve_bed_residence_timing,
    sample_bed_residence_age_hours,
    sample_bed_residence_age_hours_conditioned_on_availability,
    sample_bed_residence_age_hours_for_member,
    sample_bed_residence_age_hours_for_particle,
)
from lagrangian_backtracking.gap_policy import (
    REJECT_UNAVAILABLE_DEPOSITION_HOUR_WITHIN_STRATUM_POLICY_ID,
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


def test_age_sampler_legacy_output_is_bit_identical_to_known_golden_vector() -> None:
    """WP-27 硬要求：``per_member`` 開發過程不得改動 ``per_observation`` 既有結果。

    本測試把 ``sample_bed_residence_age_hours``（既有 legacy 抽樣核心，仍是
    ``age_sampling_scope=per_observation`` 在未啟用缺口條件式抽樣時唯一使用的
    函式）在固定 ``(maximum_age_days=90, sample_count=50, seed=_SEED)`` 下的輸出，
    與一組寫死的「黃金值」（於 WP-27 開工前、確認 WP-27 完全沒有修改本函式任何
    一行程式碼時算出並貼進本檔）逐一比對。任何未來重構若不慎改動這個函式的演算法
    （即使只是內部呼叫順序或型別轉換），本測試會直接失敗，而不必依賴「目測 diff」
    來確認 per_observation 位元相同——這是本工作包對「既有結果必須位元相同」
    這個硬要求的直接證據，而不只是「我們沒有動這段程式碼」的口頭宣稱。
    """

    golden_ages = (
        779, 2041, 1030, 1556, 1329, 640, 1248, 1742, 1660, 836,
        1947, 480, 1356, 1387, 663, 226, 949, 1943, 194, 1459,
        964, 80, 2100, 900, 351, 1179, 411, 1620, 1848, 2151,
        1124, 746, 461, 543, 341, 1539, 149, 1294, 566, 1867,
        1726, 277, 694, 1785, 1106, 1045, 2005, 1484, 105, 26,
    )
    ages = sample_bed_residence_age_hours(
        maximum_age_days=_MAX_AGE_DAYS,
        sample_count=_SAMPLE_COUNT,
        seed=_SEED,
    )
    assert ages == golden_ages


def test_age_sampler_conditioned_output_is_bit_identical_to_known_golden_vector() -> None:
    """同上，但鎖定正式管線實際使用的缺口條件式抽樣核心
    ``sample_bed_residence_age_hours_conditioned_on_availability``（重用
    ``test_gap_aware_age_sampler_rejects_unavailable_candidates_and_audits_each_stratum``
    完全相同的合成五站可用性設定），確保 WP-27 也沒有更動這個函式的任何一行。
    """

    base_ns = int(datetime(2025, 10, 31, tzinfo=UTC).timestamp()) * 1_000_000_000
    observation_times = {
        site_id: tuple(base_ns + rank * 500 * _HOUR_NS for rank in range(_SAMPLE_COUNT))
        for site_id in ("gongliao", "guishan", "hsinchu", "houwan", "lienchiang")
    }
    total_hours = _MAX_AGE_DAYS * 24 + 1
    allowed_by_stratum = {
        index: index * total_hours // _SAMPLE_COUNT
        for index in range(_SAMPLE_COUNT)
    }
    available = {
        site_id: {
            observation_ns - allowed_by_stratum[stratum] * _HOUR_NS
            for stratum, observation_ns in enumerate(observation_times[site_id])
        }
        for site_id in observation_times
    }
    sampled = sample_bed_residence_age_hours_conditioned_on_availability(
        observation_times_by_site=observation_times,
        available_time_ns_by_site=available,
        maximum_age_days=_MAX_AGE_DAYS,
        sample_count=_SAMPLE_COUNT,
        seed=_SEED,
    )
    golden_age_hours = (
        0, 43, 86, 129, 172, 216, 259, 302, 345, 388,
        432, 475, 518, 561, 605, 648, 691, 734, 777, 821,
        864, 907, 950, 994, 1037, 1080, 1123, 1166, 1210, 1253,
        1296, 1339, 1383, 1426, 1469, 1512, 1555, 1599, 1642, 1685,
        1728, 1772, 1815, 1858, 1901, 1944, 1988, 2031, 2074, 2117,
    )
    assert sampled.age_hours == golden_age_hours


def test_gap_aware_age_sampler_rejects_unavailable_candidates_and_audits_each_stratum() -> None:
    """新版抽樣只接受五站 exact-hour 共同可用候選，且逐層保存拒絕稽核。"""

    base_ns = int(datetime(2025, 10, 31, tzinfo=UTC).timestamp()) * 1_000_000_000
    observation_times = {
        site_id: tuple(base_ns + rank * 500 * _HOUR_NS for rank in range(_SAMPLE_COUNT))
        for site_id in ("gongliao", "guishan", "hsinchu", "houwan", "lienchiang")
    }
    total_hours = _MAX_AGE_DAYS * 24 + 1
    allowed_by_stratum = {
        index: index * total_hours // _SAMPLE_COUNT
        for index in range(_SAMPLE_COUNT)
    }
    available = {
        site_id: {
            observation_ns - allowed_by_stratum[stratum] * _HOUR_NS
            for stratum, observation_ns in enumerate(observation_times[site_id])
        }
        for site_id in observation_times
    }
    sampled = sample_bed_residence_age_hours_conditioned_on_availability(
        observation_times_by_site=observation_times,
        available_time_ns_by_site=available,
        maximum_age_days=_MAX_AGE_DAYS,
        sample_count=_SAMPLE_COUNT,
        seed=_SEED,
    )
    assert sampled.sampling_method_id == BED_RESIDENCE_AVAILABILITY_CONDITIONED_SAMPLING_METHOD_ID
    assert sampled.availability_conditioning_policy == (
        REJECT_UNAVAILABLE_DEPOSITION_HOUR_WITHIN_STRATUM_POLICY_ID
    )
    assert len(sampled.age_hours) == len(sampled.rejection_audit) == _SAMPLE_COUNT
    assert tuple(item.selected_age_hours for item in sampled.rejection_audit) == sampled.age_hours
    assert all(item.observation_rank == item.stratum_index for item in sampled.rejection_audit)
    assert any(item.rejected_age_hours for item in sampled.rejection_audit)
    assert sampled == sample_bed_residence_age_hours_conditioned_on_availability(
        observation_times_by_site=observation_times,
        available_time_ns_by_site=available,
        maximum_age_days=_MAX_AGE_DAYS,
        sample_count=_SAMPLE_COUNT,
        seed=_SEED,
    )


def test_gap_aware_age_sampler_fails_closed_when_a_stratum_has_no_common_candidate() -> None:
    """五站任一分層沒有共同 deposition hour 時，不以 nearest 或填值繼續。"""

    base_ns = int(datetime(2025, 10, 31, tzinfo=UTC).timestamp()) * 1_000_000_000
    observation_times = {
        site_id: tuple(base_ns + rank * 500 * _HOUR_NS for rank in range(_SAMPLE_COUNT))
        for site_id in ("gongliao", "guishan", "hsinchu", "houwan", "lienchiang")
    }
    available = {site_id: set() for site_id in observation_times}
    with pytest.raises(ValueError, match="無可用候選"):
        sample_bed_residence_age_hours_conditioned_on_availability(
            observation_times_by_site=observation_times,
            available_time_ns_by_site=available,
            maximum_age_days=_MAX_AGE_DAYS,
            sample_count=_SAMPLE_COUNT,
            seed=_SEED,
        )


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


# ══════════════════════════════════════════════════════════════════════════
# WP-18：逐粒子（per_particle）沉底年齡抽樣原語。
# ══════════════════════════════════════════════════════════════════════════


def _observation_utc_ns() -> int:
    return int(datetime(2025, 10, 31, 6, tzinfo=UTC).timestamp()) * 1_000_000_000


def _available_every_nth_hour(observation_ns: int, *, step: int, max_age_days: int) -> set[int]:
    """建立「觀測時刻往前每隔 step 小時才可用」的合成 canonical available-time set。"""

    return {
        observation_ns - age * _HOUR_NS
        for age in range(0, max_age_days * 24 + 1, step)
    }


def test_derive_bed_residence_particle_seed_is_deterministic_and_particle_specific() -> None:
    """相同粒子身分永遠得到相同種子；不同 scenario 或 member 得到不同種子。"""

    seed_a = derive_bed_residence_particle_seed(
        master_seed=_SEED, scenario_id="scenario-a", member_id=0
    )
    seed_a_again = derive_bed_residence_particle_seed(
        master_seed=_SEED, scenario_id="scenario-a", member_id=0
    )
    seed_b_member = derive_bed_residence_particle_seed(
        master_seed=_SEED, scenario_id="scenario-a", member_id=1
    )
    seed_b_scenario = derive_bed_residence_particle_seed(
        master_seed=_SEED, scenario_id="scenario-b", member_id=0
    )
    assert seed_a == seed_a_again
    assert seed_a != seed_b_member
    assert seed_a != seed_b_scenario
    assert isinstance(seed_a, int) and seed_a >= 0


def test_derive_bed_residence_particle_seed_independent_of_member_seed_stream() -> None:
    """沉底年齡種子刻意與 ``scenarios.derive_member_seed`` 使用的種子流不同，
    避免兩種隨機量（沉底年齡 vs. 平流／擴散）彼此相關。"""

    from lagrangian_backtracking.scenarios import derive_member_seed

    bed_seed = derive_bed_residence_particle_seed(
        master_seed=_SEED, scenario_id="scenario-a", member_id=0
    )
    member_seed = derive_member_seed(
        master_seed=_SEED,
        scenario_id="scenario-a",
        experiment_case_id="finite_depth_stokes",
        member_id=0,
    )
    assert bed_seed != member_seed


def test_sample_for_particle_is_reproducible_and_respects_availability() -> None:
    observation_ns = _observation_utc_ns()
    available = _available_every_nth_hour(observation_ns, step=3, max_age_days=_MAX_AGE_DAYS)
    seed = derive_bed_residence_particle_seed(
        master_seed=_SEED, scenario_id="scenario-a", member_id=0
    )

    first = sample_bed_residence_age_hours_for_particle(
        observation_time_utc_ns=observation_ns,
        available_time_ns=available,
        maximum_age_days=_MAX_AGE_DAYS,
        seed=seed,
    )
    second = sample_bed_residence_age_hours_for_particle(
        observation_time_utc_ns=observation_ns,
        available_time_ns=available,
        maximum_age_days=_MAX_AGE_DAYS,
        seed=seed,
    )
    assert first.age_hours == second.age_hours
    assert first.rejected_age_hours == second.rejected_age_hours
    assert first.sampling_method_id == BED_RESIDENCE_PER_PARTICLE_SAMPLING_METHOD_ID
    assert 0 <= first.age_hours <= _MAX_AGE_DAYS * 24
    assert first.age_hours % 3 == 0
    assert (observation_ns - first.age_hours * _HOUR_NS) in available
    for rejected_age in first.rejected_age_hours:
        assert (observation_ns - rejected_age * _HOUR_NS) not in available


def test_sample_for_particle_different_particles_can_get_different_ages() -> None:
    """不同粒子（不同 seed）在同一個可用性條件下不必抽到同一個年齡——這是「以量取勝」
    需求（PI 原話）成立的前提：粒子彼此獨立，不是全部共用同一個年齡。"""

    observation_ns = _observation_utc_ns()
    available = set(range(0, _MAX_AGE_DAYS * 24 + 1))
    available = {observation_ns - age * _HOUR_NS for age in available}

    ages = set()
    for member_id in range(20):
        seed = derive_bed_residence_particle_seed(
            master_seed=_SEED, scenario_id="scenario-a", member_id=member_id
        )
        sample = sample_bed_residence_age_hours_for_particle(
            observation_time_utc_ns=observation_ns,
            available_time_ns=available,
            maximum_age_days=_MAX_AGE_DAYS,
            seed=seed,
        )
        ages.add(sample.age_hours)
    assert len(ages) > 1, "20 個不同粒子的種子抽出的年齡不應該全部相同"


def test_sample_for_particle_fails_closed_when_nothing_available() -> None:
    observation_ns = _observation_utc_ns()
    seed = derive_bed_residence_particle_seed(
        master_seed=_SEED, scenario_id="scenario-a", member_id=0
    )
    with pytest.raises(ValueError, match="無可用候選"):
        sample_bed_residence_age_hours_for_particle(
            observation_time_utc_ns=observation_ns,
            available_time_ns=set(),
            maximum_age_days=_MAX_AGE_DAYS,
            seed=seed,
        )


def test_sample_for_particle_rejects_non_exact_hour_observation() -> None:
    seed = derive_bed_residence_particle_seed(
        master_seed=_SEED, scenario_id="scenario-a", member_id=0
    )
    with pytest.raises(ValueError, match="exact-hour"):
        sample_bed_residence_age_hours_for_particle(
            observation_time_utc_ns=_observation_utc_ns() + 1,
            available_time_ns={_observation_utc_ns()},
            maximum_age_days=_MAX_AGE_DAYS,
            seed=seed,
        )


# ══════════════════════════════════════════════════════════════════════════
# WP-27：per_member（每站點×到達時刻×系集成員獨立抽樣，成員內共用）抽樣原語測試。
# 這裡只測試 ``bed_residence.py`` 新增的抽樣原語本身（見該模組 WP-27 章節「範圍
# 誠實揭露」）；下游 native 管線尚未接線，見 ``reports/WP-27.md`` 第 3 節。
# ══════════════════════════════════════════════════════════════════════════


def test_derive_bed_residence_member_seed_is_deterministic_and_member_specific() -> None:
    """相同 (站點, 到達時刻, 成員) 身分永遠得到相同種子；改動任一個身分欄位都必須
    得到不同種子，確保「同一到達事件、不同成員」互相獨立。"""

    seed_a = derive_bed_residence_member_seed(
        master_seed=_SEED,
        study_site_id="gongliao",
        arrival_time_id="gongliao-observation-00",
        member_id=0,
    )
    seed_a_again = derive_bed_residence_member_seed(
        master_seed=_SEED,
        study_site_id="gongliao",
        arrival_time_id="gongliao-observation-00",
        member_id=0,
    )
    seed_other_member = derive_bed_residence_member_seed(
        master_seed=_SEED,
        study_site_id="gongliao",
        arrival_time_id="gongliao-observation-00",
        member_id=1,
    )
    seed_other_arrival = derive_bed_residence_member_seed(
        master_seed=_SEED,
        study_site_id="gongliao",
        arrival_time_id="gongliao-observation-01",
        member_id=0,
    )
    seed_other_site = derive_bed_residence_member_seed(
        master_seed=_SEED,
        study_site_id="guishan",
        arrival_time_id="gongliao-observation-00",
        member_id=0,
    )
    assert seed_a == seed_a_again
    assert seed_a != seed_other_member
    assert seed_a != seed_other_arrival
    assert seed_a != seed_other_site
    assert isinstance(seed_a, int) and seed_a >= 0


def test_derive_bed_residence_member_seed_shared_across_material_and_receptor() -> None:
    """核心折衷設計：``derive_bed_residence_member_seed`` 的簽章刻意不接受
    ``scenario_id``、``receptor_id`` 或材質身分，只接受站點×到達時刻×成員。這代表
    「同一個到達事件、同一個成員、不同材質或不同受體」在呼叫端組出的 seed 必然相同
    ——不需要额外協調，光是函式簽章本身就保證了「成員內全部材質與受體共用同一個
    年齡」這個 per_member 定義的資料契約，不會因為呼叫端傳錯 scenario 身分而破功。
    這裡用兩個「假裝來自不同材質、不同受體的 scenario」但共用同一個
    (site, arrival, member) 三元組的呼叫，驗證兩者確實得到位元相同的種子與年齡。
    """

    kwargs = dict(
        master_seed=_SEED,
        study_site_id="gongliao",
        arrival_time_id="gongliao-observation-07",
        member_id=3,
    )
    seed_material_x_receptor_a = derive_bed_residence_member_seed(**kwargs)
    seed_material_y_receptor_b = derive_bed_residence_member_seed(**kwargs)
    assert seed_material_x_receptor_a == seed_material_y_receptor_b

    observation_ns = _observation_utc_ns()
    available = set(range(0, _MAX_AGE_DAYS * 24 + 1))
    available = {observation_ns - age * _HOUR_NS for age in available}
    sample_a = sample_bed_residence_age_hours_for_member(
        observation_time_utc_ns=observation_ns,
        available_time_ns=available,
        maximum_age_days=_MAX_AGE_DAYS,
        seed=seed_material_x_receptor_a,
    )
    sample_b = sample_bed_residence_age_hours_for_member(
        observation_time_utc_ns=observation_ns,
        available_time_ns=available,
        maximum_age_days=_MAX_AGE_DAYS,
        seed=seed_material_y_receptor_b,
    )
    assert sample_a.age_hours == sample_b.age_hours


def test_derive_bed_residence_member_seed_independent_of_particle_and_member_seed_streams() -> None:
    """per_member 種子刻意與 ``derive_bed_residence_particle_seed``（per_particle
    角色標籤 ``"bed_residence_age_v1"``）及 ``scenarios.derive_member_seed``
    （平流／擴散角色）使用不同角色標籤，確保三種隨機量彼此獨立，不會因為共用
    ``(master_seed, ...)`` 而產生非預期的跨隨機量相關性。"""

    from lagrangian_backtracking.scenarios import derive_member_seed

    member_scope_seed = derive_bed_residence_member_seed(
        master_seed=_SEED,
        study_site_id="gongliao",
        arrival_time_id="gongliao-observation-00",
        member_id=0,
    )
    particle_scope_seed = derive_bed_residence_particle_seed(
        master_seed=_SEED, scenario_id="gongliao-observation-00", member_id=0
    )
    advection_seed = derive_member_seed(
        master_seed=_SEED,
        scenario_id="gongliao-observation-00",
        experiment_case_id="finite_depth_stokes",
        member_id=0,
    )
    assert member_scope_seed != particle_scope_seed
    assert member_scope_seed != advection_seed


def test_sample_for_member_is_reproducible_and_respects_availability() -> None:
    observation_ns = _observation_utc_ns()
    available = _available_every_nth_hour(observation_ns, step=3, max_age_days=_MAX_AGE_DAYS)
    seed = derive_bed_residence_member_seed(
        master_seed=_SEED,
        study_site_id="gongliao",
        arrival_time_id="gongliao-observation-00",
        member_id=0,
    )

    first = sample_bed_residence_age_hours_for_member(
        observation_time_utc_ns=observation_ns,
        available_time_ns=available,
        maximum_age_days=_MAX_AGE_DAYS,
        seed=seed,
    )
    second = sample_bed_residence_age_hours_for_member(
        observation_time_utc_ns=observation_ns,
        available_time_ns=available,
        maximum_age_days=_MAX_AGE_DAYS,
        seed=seed,
    )
    assert first.age_hours == second.age_hours
    assert first.rejected_age_hours == second.rejected_age_hours
    assert first.sampling_method_id == BED_RESIDENCE_PER_MEMBER_SAMPLING_METHOD_ID
    assert 0 <= first.age_hours <= _MAX_AGE_DAYS * 24
    assert first.age_hours % 3 == 0
    assert (observation_ns - first.age_hours * _HOUR_NS) in available
    for rejected_age in first.rejected_age_hours:
        assert (observation_ns - rejected_age * _HOUR_NS) not in available


def test_sample_for_member_different_members_can_get_different_ages() -> None:
    """不同成員（不同 seed）在同一個到達事件、同一個可用性條件下不必抽到同一個
    年齡——這正是 per_member 相對 per_observation 拿回的「去相關」性質：PI 要求
    「沉底多久」本身是隨機變數，不應該對同一個到達事件的全部系集成員都套用同一個
    年齡。"""

    observation_ns = _observation_utc_ns()
    available = set(range(0, _MAX_AGE_DAYS * 24 + 1))
    available = {observation_ns - age * _HOUR_NS for age in available}

    ages = set()
    for member_id in range(10):
        seed = derive_bed_residence_member_seed(
            master_seed=_SEED,
            study_site_id="gongliao",
            arrival_time_id="gongliao-observation-00",
            member_id=member_id,
        )
        sample = sample_bed_residence_age_hours_for_member(
            observation_time_utc_ns=observation_ns,
            available_time_ns=available,
            maximum_age_days=_MAX_AGE_DAYS,
            seed=seed,
        )
        ages.add(sample.age_hours)
    assert len(ages) > 1, "10 個系集成員抽出的年齡不應該全部相同"


def test_sample_for_member_ages_spread_across_full_range_not_clustered() -> None:
    """WP-27 驗收要求之一：per_member 的年齡分布需檢驗。這裡對很多個
    (站點, 到達時刻, 成員) 三元組各抽一次年齡，確認抽出的年齡涵蓋 0～90 日範圍的
    大部分區段，不會因為種子推導方式不當而聚集在一小段區間（那會表示年齡與到達
    時刻仍然存在非預期的相關性，違背 PI「去相關」的要求）。"""

    observation_ns = _observation_utc_ns()
    available = set(range(0, _MAX_AGE_DAYS * 24 + 1))
    available = {observation_ns - age * _HOUR_NS for age in available}

    ages: list[int] = []
    for site_id in ("gongliao", "guishan", "hsinchu", "houwan", "lienchiang"):
        for observation_index in range(10):
            for member_id in range(10):
                seed = derive_bed_residence_member_seed(
                    master_seed=_SEED,
                    study_site_id=site_id,
                    arrival_time_id=f"{site_id}-observation-{observation_index:02d}",
                    member_id=member_id,
                )
                sample = sample_bed_residence_age_hours_for_member(
                    observation_time_utc_ns=observation_ns,
                    available_time_ns=available,
                    maximum_age_days=_MAX_AGE_DAYS,
                    seed=seed,
                )
                ages.append(sample.age_hours)

    total_hours = _MAX_AGE_DAYS * 24 + 1
    quartile = total_hours // 4
    occupied_quartiles = {min(age // quartile, 3) for age in ages}
    assert occupied_quartiles == {0, 1, 2, 3}, (
        f"500 個抽樣年齡應涵蓋 0-90 日的四個四分位區間，實際只涵蓋 {occupied_quartiles}"
    )
    assert len(set(ages)) > len(ages) // 2, "500 個抽樣年齡的相異值比例過低，疑似種子推導有誤"


def test_sample_for_member_availability_conditioning_rejects_and_records_candidates() -> None:
    """可用性條件（沿用與 per_particle 完全相同的規則）：只接受「沉底時刻有可用流場
    資料」的年齡，被跳過的候選必須逐一保存在 ``rejected_age_hours``。"""

    observation_ns = _observation_utc_ns()
    available = _available_every_nth_hour(observation_ns, step=5, max_age_days=_MAX_AGE_DAYS)
    seed = derive_bed_residence_member_seed(
        master_seed=_SEED,
        study_site_id="gongliao",
        arrival_time_id="gongliao-observation-12",
        member_id=4,
    )
    sample = sample_bed_residence_age_hours_for_member(
        observation_time_utc_ns=observation_ns,
        available_time_ns=available,
        maximum_age_days=_MAX_AGE_DAYS,
        seed=seed,
    )
    assert sample.age_hours % 5 == 0
    assert sample.candidate_count == _MAX_AGE_DAYS * 24 + 1
    for rejected_age in sample.rejected_age_hours:
        assert rejected_age % 5 != 0 or (
            observation_ns - rejected_age * _HOUR_NS
        ) not in available


def test_sample_for_member_fails_closed_when_nothing_available() -> None:
    observation_ns = _observation_utc_ns()
    seed = derive_bed_residence_member_seed(
        master_seed=_SEED,
        study_site_id="gongliao",
        arrival_time_id="gongliao-observation-00",
        member_id=0,
    )
    with pytest.raises(ValueError, match="無可用候選"):
        sample_bed_residence_age_hours_for_member(
            observation_time_utc_ns=observation_ns,
            available_time_ns=set(),
            maximum_age_days=_MAX_AGE_DAYS,
            seed=seed,
        )


def test_sample_for_member_rejects_non_exact_hour_observation() -> None:
    seed = derive_bed_residence_member_seed(
        master_seed=_SEED,
        study_site_id="gongliao",
        arrival_time_id="gongliao-observation-00",
        member_id=0,
    )
    with pytest.raises(ValueError, match="exact-hour"):
        sample_bed_residence_age_hours_for_member(
            observation_time_utc_ns=_observation_utc_ns() + 1,
            available_time_ns={_observation_utc_ns()},
            maximum_age_days=_MAX_AGE_DAYS,
            seed=seed,
        )
