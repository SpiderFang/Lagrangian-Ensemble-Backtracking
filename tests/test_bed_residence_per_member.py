"""WP-51：逐成員（per_member）分層抽樣、隨機配對與缺口條件測試。

測的是 ``bed_residence.sample_bed_residence_member_ages_conditioned_on_availability``
與 ``apply_bed_residence_member_sampling``：每站 50 個觀測 × M 個成員的年齡必須
涵蓋整段 90 天（不再與觀測時刻單調相關）、五站共用同一槽位的年齡、缺口條件把
不可用整點拒絕、且輸出對輸入列順序不敏感。
"""

from __future__ import annotations

from datetime import UTC, datetime

import json
from pathlib import Path

import numpy as np
import pytest

from lagrangian_backtracking import bed_residence as bed_residence_module
from lagrangian_backtracking.bed_residence import (
    BED_RESIDENCE_PER_MEMBER_PERMUTATION_REPAIR_ALGORITHM_ID,
    BED_RESIDENCE_PER_MEMBER_STRATIFIED_SAMPLING_METHOD_ID,
    apply_bed_residence_member_sampling,
    recompute_member_initial_slot_permutation,
    sample_bed_residence_member_ages_conditioned_on_availability,
    summarize_member_permutation_repair,
)
from lagrangian_backtracking.gap_policy import (
    REJECT_UNAVAILABLE_DEPOSITION_HOUR_WITHIN_STRATUM_POLICY_ID,
)
from lagrangian_backtracking.scenarios import ArrivalTime

_HOUR_NS = 3_600_000_000_000
_DAY_NS = 24 * _HOUR_NS
_SITES = ("gongliao", "guishan", "hsinchu", "houwan", "lienchiang")
_MAX_AGE_DAYS = 90
_SEED = 20260916
_M = 10


def _ns(year: int, month: int, day: int) -> int:
    """UTC 午夜的奈秒。"""

    return int(datetime(year, month, day, tzinfo=UTC).timestamp()) * 1_000_000_000


def _observations() -> tuple[int, ...]:
    """50 個觀測時刻：2025-01-20 起每 5 天一個，涵蓋到 2025-09-22 之前。"""

    start = _ns(2025, 1, 20)
    return tuple(start + rank * 5 * _DAY_NS + (rank % 7) * _HOUR_NS for rank in range(50))


def _observation_times_by_site() -> dict[str, tuple[int, ...]]:
    return {site: _observations() for site in _SITES}


def _full_axis(*, drop: set[int] | None = None) -> dict[str, set[int]]:
    """2024-09-01 到 2025-12-31 逐小時 canonical 軸；``drop`` 為要拿掉的整點。"""

    start = _ns(2024, 9, 1)
    stop = _ns(2025, 12, 31)
    axis = set(range(start, stop, _HOUR_NS))
    if drop:
        axis -= drop
    return {site: set(axis) for site in _SITES}


def _sample(*, available=None, seed: int = _SEED, member_count: int = _M):
    return sample_bed_residence_member_ages_conditioned_on_availability(
        observation_times_by_site=_observation_times_by_site(),
        available_time_ns_by_site=available if available is not None else _full_axis(),
        maximum_age_days=_MAX_AGE_DAYS,
        sample_count=50,
        member_count=member_count,
        seed=seed,
    )


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman 等級相關（以平均名次處理同值）。"""

    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="stable")
        out = np.empty(len(values), dtype=float)
        out[order] = np.arange(len(values), dtype=float)
        for value in np.unique(values):
            mask = values == value
            out[mask] = out[mask].mean()
        return out

    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def test_member_ages_cover_full_range_and_are_not_monotonic_in_rank() -> None:
    """500 個槽位各佔一個分層：年齡涵蓋 0..2160 小時，且與觀測 rank 幾乎無相關。"""

    sampling = _sample()
    assert sampling.sampling_method_id == BED_RESIDENCE_PER_MEMBER_STRATIFIED_SAMPLING_METHOD_ID
    assert sampling.availability_conditioning_policy == (
        REJECT_UNAVAILABLE_DEPOSITION_HOUR_WITHIN_STRATUM_POLICY_ID
    )
    assert sampling.sample_count == 50 and sampling.member_count == _M
    assert sampling.slot_count == 500
    ages = np.array(sampling.age_hours)
    assert ages.shape == (50, _M)
    assert len(set(ages.ravel().tolist())) == 500, "500 個年齡必須互異（每層一個）"
    assert ages.min() <= 5 and ages.max() >= _MAX_AGE_DAYS * 24 - 5
    # 每個分層恰好被用一次：槽位 → 分層是 0..499 的排列
    assert sorted(sampling.slot_permutation) == list(range(500))
    assert sorted(np.array(sampling.stratum_index).ravel().tolist()) == list(range(500))
    ranks = np.repeat(np.arange(50), _M)
    rho = _spearman(ranks.astype(float), ages.ravel().astype(float))
    # 舊的 per_observation 是 rho == 1.0；隨機配對的 rho 期望為 0，標準差約 0.045。
    assert abs(rho) < 0.2, f"年齡與觀測 rank 的 Spearman 仍過高：{rho}"
    # 前 10 個與後 10 個觀測的平均年齡不可系統性不同
    assert abs(ages[:10].mean() - ages[-10:].mean()) < 0.35 * ages.max()


def test_member_ages_spread_deposition_times_across_more_than_the_observation_season() -> None:
    """沉底時刻（觀測 - 年齡）必須涵蓋跨年度範圍，而非集中在 1-9 月。"""

    sampling = _sample()
    obs = np.array(_observations(), dtype=np.int64)[:, None]
    deposition = obs - np.array(sampling.age_hours, dtype=np.int64) * _HOUR_NS
    months = {
        datetime.fromtimestamp(int(value) / 1e9, tz=UTC).month
        for value in deposition.ravel()
    }
    # 觀測在 1-9 月、年齡到 90 天，沉底時刻應涵蓋 10-12 月（舊 per_observation 沒有）
    assert {10, 11, 12}.issubset(months)
    # 不同（rank, 年齡）組合偶爾會得到相同整點，因此只要求絕大多數互異
    assert len(set(deposition.ravel().tolist())) >= 450


def test_member_sampling_is_reproducible_seed_dependent_and_input_order_free() -> None:
    first = _sample()
    assert first == _sample()
    assert first.age_matrix_sha256() == _sample().age_matrix_sha256()
    assert first.age_hours != _sample(seed=_SEED + 1).age_hours
    # 站點輸入的觀測列順序不影響（內部先排序）
    shuffled = {
        site: tuple(reversed(times)) for site, times in _observation_times_by_site().items()
    }
    again = sample_bed_residence_member_ages_conditioned_on_availability(
        observation_times_by_site=shuffled,
        available_time_ns_by_site=_full_axis(),
        maximum_age_days=_MAX_AGE_DAYS,
        sample_count=50,
        member_count=_M,
        seed=_SEED,
    )
    assert again == first


def test_member_sampling_golden_vector_locks_algorithm() -> None:
    """固定 seed 的前 8 個槽位年齡：任何演算法或亂數流順序改動都會讓此測試失敗。"""

    first = _sample()
    flat = [age for row in first.age_hours for age in row]
    # 黃金值：首次實作時在 .203 以 numpy 2.x 取得；若 numpy 的 PCG64DXSM permutation
    # 行為改版，請與主代理確認後再更新。
    assert flat[:8] == list(_GOLDEN_FIRST_EIGHT)


_GOLDEN_FIRST_EIGHT = (1298, 498, 1946, 1459, 797, 744, 170, 916)


def test_member_gap_conditioning_rejects_unavailable_hours_with_audit() -> None:
    """拿掉某些整點後，被選中的沉底整點必須五站皆可用；拒絕過程逐槽位記錄。"""

    base = _sample()
    obs = _observations()
    # 把第一個基準抽樣的前 20 個槽位的沉底整點在「第 2 站」拿掉
    drop = set()
    for slot in range(20):
        rank, member = divmod(slot, _M)
        drop.add(obs[rank] - base.age_hours[rank][member] * _HOUR_NS)
    available = _full_axis()
    available["guishan"] -= drop
    sampling = _sample(available=available)
    assert sampling != base
    for rank in range(50):
        for member in range(_M):
            age = sampling.age_hours[rank][member]
            for site in _SITES:
                assert obs[rank] - age * _HOUR_NS in available[site], (rank, member, site)
    audits = {a.slot_index: a for a in sampling.rejection_audit}
    assert sorted(audits) == list(range(500))
    rejected_slots = [slot for slot, a in audits.items() if a.rejected_age_hours]
    assert rejected_slots, "拿掉候選後至少有一個槽位被拒絕"
    for slot in rejected_slots:
        audit = audits[slot]
        assert audit.rejection_count if hasattr(audit, "rejection_count") else True
        assert all(
            sites == ("guishan",) for _age, sites in audit.rejected_by_site
        ), "只有 guishan 缺資料，被拒絕理由只應列 guishan"
        assert audit.member_id == slot % _M and audit.observation_rank == slot // _M
        meta = audit.as_metadata()
        assert meta["member_id"] == slot % _M and meta["slot_index"] == slot
    # 沒被拿掉整點的槽位與基準相同
    unchanged = [
        slot for slot in range(20, 500)
        if sampling.age_hours[slot // _M][slot % _M] == base.age_hours[slot // _M][slot % _M]
    ]
    assert len(unchanged) >= 400


def test_member_gap_conditioning_fails_closed_when_whole_stratum_unavailable() -> None:
    # 整個可用時間軸為空：可行性矩陣全為假，修補（單次交換與二分圖匹配）都找不到解，
    # 必須 fail closed，不可補零、取最近值或外插。
    with pytest.raises(ValueError, match="無法完全配對"):
        _sample(available={site: set() for site in _SITES})


def test_member_sampling_rejects_bad_inputs() -> None:
    times = _observation_times_by_site()
    with pytest.raises(ValueError, match="每站必須使用 50"):
        sample_bed_residence_member_ages_conditioned_on_availability(
            observation_times_by_site=times,
            available_time_ns_by_site=_full_axis(),
            maximum_age_days=_MAX_AGE_DAYS,
            sample_count=49,
            member_count=_M,
            seed=_SEED,
        )
    with pytest.raises(ValueError, match="槽位數"):
        sample_bed_residence_member_ages_conditioned_on_availability(
            observation_times_by_site=times,
            available_time_ns_by_site=_full_axis(),
            maximum_age_days=1,
            sample_count=50,
            member_count=_M,
            seed=_SEED,
        )
    bad = dict(times)
    bad["guishan"] = tuple(t + 1 for t in times["guishan"])
    with pytest.raises(ValueError, match="exact-hour"):
        sample_bed_residence_member_ages_conditioned_on_availability(
            observation_times_by_site=bad,
            available_time_ns_by_site=_full_axis(),
            maximum_age_days=_MAX_AGE_DAYS,
            sample_count=50,
            member_count=_M,
            seed=_SEED,
        )


def _arrivals(site: str) -> tuple[ArrivalTime, ...]:
    return tuple(
        ArrivalTime(
            arrival_time_id=f"{site}-obs-{rank:02d}",
            study_site_id=site,
            time_utc_ns=obs,
            year=2025,
            season="DJF",
            tide_class="spring" if rank % 2 else "neap",
            phase_or_event="rising_fastest",
            metadata={"anchor_source": "observation"},
        )
        for rank, obs in enumerate(_observations())
    )


def test_five_sites_share_age_per_slot_and_conversion_outputs_member_records() -> None:
    sampling = _sample()
    by_site = {
        site: apply_bed_residence_member_sampling(
            _arrivals(site),
            sampling=sampling,
            sampling_seed=_SEED,
            maximum_age_days=_MAX_AGE_DAYS,
            design_version="wp51-test",
        )
        for site in _SITES
    }
    for site, records in by_site.items():
        assert len(records) == 50 * _M
        assert len({r.arrival_time_id for r in records}) == 500
        assert len({r.time_utc_ns for r in records}) >= 450
    reference = by_site["gongliao"]
    for site in _SITES[1:]:
        ages = [r.metadata["bed_residence_age_hours"] for r in by_site[site]]
        assert ages == [r.metadata["bed_residence_age_hours"] for r in reference]
        members = [r.metadata["bed_residence_member_id"] for r in by_site[site]]
        assert members == [r.metadata["bed_residence_member_id"] for r in reference]
    for index, record in enumerate(reference):
        meta = record.metadata
        rank, member = divmod(index, _M)  # 輸入是依時間排序的，rank 與輸入序一致
        assert meta["bed_residence_member_id"] == member
        assert meta["bed_residence_observation_rank"] == rank
        assert meta["bed_residence_age_sampling_scope"] == "per_member"
        age = sampling.age_hours[rank][member]
        assert record.time_utc_ns == _observations()[rank] - age * _HOUR_NS
        assert meta["deposition_time_utc_ns"] == record.time_utc_ns
        assert meta["observation_time_utc_ns"] == _observations()[rank]
        assert meta["observation_arrival_time_id"] == f"gongliao-obs-{rank:02d}"
        dep = datetime.fromtimestamp(record.time_utc_ns / 1e9, tz=UTC)
        assert record.year == dep.year
        assert meta["observation_season"] == "DJF"
        assert meta["bed_residence_availability_rejection_audit"]["member_id"] == member


def test_conversion_is_independent_of_input_row_order_and_rejects_double_conversion() -> None:
    sampling = _sample()
    arrivals = _arrivals("hsinchu")
    forward = apply_bed_residence_member_sampling(
        arrivals, sampling=sampling, sampling_seed=_SEED,
        maximum_age_days=_MAX_AGE_DAYS, design_version="wp51-test",
    )
    backward = apply_bed_residence_member_sampling(
        tuple(reversed(arrivals)), sampling=sampling, sampling_seed=_SEED,
        maximum_age_days=_MAX_AGE_DAYS, design_version="wp51-test",
    )
    by_obs_f = {(r.metadata["observation_arrival_time_id"], r.metadata["bed_residence_member_id"]): r for r in forward}
    by_obs_b = {(r.metadata["observation_arrival_time_id"], r.metadata["bed_residence_member_id"]): r for r in backward}
    assert by_obs_f == by_obs_b
    with pytest.raises(ValueError, match="禁止再次轉換"):
        apply_bed_residence_member_sampling(
            tuple(forward[i] for i in range(0, 500, 10)),
            sampling=sampling, sampling_seed=_SEED,
            maximum_age_days=_MAX_AGE_DAYS, design_version="wp51-test",
        )


# ══════════════════════════════════════════════════════════════════════════
# WP-51 整合：排列後可行性修補（真實 OCM 整日缺口形狀的回歸測試）
# ══════════════════════════════════════════════════════════════════════════

_GAP_FIXTURE = Path(__file__).parent / "fixtures" / "wp51_real_ocm_gap_shape.json"


def _parse_utc_ns(text: str) -> int:
    return int(datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC).timestamp()) * 10**9


@pytest.fixture(scope="module")
def real_gap_case() -> dict:
    """真實 OCM 可用時間軸的缺口形狀（四區相同：48 段、缺 419 h，其中 15 段約 24 h 整日缺口）
    與 v3 正式母體五站各 50 個觀測時刻；不含流場資料。"""

    payload = json.loads(_GAP_FIXTURE.read_text(encoding="utf-8"))
    assert payload["四區缺口相同"] is True
    start = _parse_utc_ns(payload["time_start_utc"])
    end = _parse_utc_ns(payload["time_end_utc"])
    axis = set(range(start, end + _HOUR_NS, _HOUR_NS))
    for before, after in payload["gaps_before_after_utc"]:
        lo, hi = _parse_utc_ns(before), _parse_utc_ns(after)
        axis -= set(range(lo + _HOUR_NS, hi, _HOUR_NS))
    assert len(axis) == 17124, "重建的可用時間軸應與 forcing_inventory 的 canonical_time_count 相符"
    observations = {
        site: tuple(values) for site, values in payload["observation_time_utc_ns_by_site"].items()
    }
    return {"axis": axis, "observations": observations}


def _sample_real(case: dict, *, seed: int = _SEED, member_count: int = _M):
    sites = sorted(case["observations"])
    return sample_bed_residence_member_ages_conditioned_on_availability(
        observation_times_by_site={site: case["observations"][site] for site in sites},
        available_time_ns_by_site={site: set(case["axis"]) for site in sites},
        maximum_age_days=_MAX_AGE_DAYS,
        sample_count=50,
        member_count=member_count,
        seed=seed,
    )


def _random_permutation_infeasible_slots(case: dict, *, seed: int, member_count: int) -> list[int]:
    """純隨機排列（不修補）之下，整層沒有五站皆可用整點的槽位。"""

    slot_count = 50 * member_count
    permutation = recompute_member_initial_slot_permutation(seed, slot_count)
    sites = sorted(case["observations"])
    times = {site: sorted(case["observations"][site]) for site in sites}
    total_hours = _MAX_AGE_DAYS * 24 + 1
    bad = []
    for slot in range(slot_count):
        rank = slot // member_count
        start = (permutation[slot] * total_hours) // slot_count
        stop = ((permutation[slot] + 1) * total_hours) // slot_count
        if not any(
            all(times[site][rank] - age * _HOUR_NS in case["axis"] for site in sites)
            for age in range(start, stop)
        ):
            bad.append(slot)
    return bad


def test_real_gap_shape_makes_pure_random_permutation_infeasible(real_gap_case) -> None:
    """回歸測試的前提：沒有修補時，設定檔種子與其他種子的純隨機排列都有整層無候選的槽位
    （真實資料上舊版抽樣器必定 fail closed，見 reports/WP-51-V.md §3）。"""

    for seed in (_SEED, 1, 2, 3, 20261007):
        bad = _random_permutation_infeasible_slots(real_gap_case, seed=seed, member_count=_M)
        assert len(bad) >= 10, (seed, len(bad))


@pytest.mark.parametrize("seed", [_SEED, 1, 2, 3, 7, 20261007])
def test_real_gap_shape_sampling_succeeds_with_repair_and_stays_random(real_gap_case, seed) -> None:
    """含整日缺口的真實形狀下，設定檔種子與多個種子都能成功，且修補不破壞隨機配對性質。"""

    initially_bad = _random_permutation_infeasible_slots(real_gap_case, seed=seed, member_count=_M)
    sampling = _sample_real(real_gap_case, seed=seed)
    sites = sorted(real_gap_case["observations"])
    times = {site: sorted(real_gap_case["observations"][site]) for site in sites}
    ages = np.array(sampling.age_hours)
    assert ages.shape == (50, _M)
    assert len(set(ages.ravel().tolist())) == 500
    assert sorted(sampling.slot_permutation) == list(range(500))
    assert sorted(sampling.initial_slot_permutation) == list(range(500))
    # 每個槽位的五站沉底整點都在各自可用軸內（不外插、不補零）
    for rank in range(50):
        for member in range(_M):
            for site in sites:
                deposition = times[site][rank] - int(ages[rank, member]) * _HOUR_NS
                assert deposition in real_gap_case["axis"], (seed, rank, member, site)
    # 修補只動必要的槽位；稽核與純隨機排列的不可行槽位完全對得上
    audits = {audit.slot_index: audit for audit in sampling.rejection_audit}
    flagged = sorted(slot for slot, audit in audits.items() if audit.initially_infeasible)
    assert flagged == initially_bad
    assert flagged, "此形狀下必須真的發生過修補，測試才有意義"
    changed = [
        slot
        for slot in range(500)
        if sampling.slot_permutation[slot] != sampling.initial_slot_permutation[slot]
    ]
    assert set(flagged) <= set(changed) and len(changed) <= 2 * len(flagged)
    for slot in range(500):
        audit = audits[slot]
        assert audit.initial_stratum_index == sampling.initial_slot_permutation[slot]
        assert (slot in changed) == bool(audit.repair_steps)
    # 隨機性：年齡與觀測 rank 幾乎無相關（舊缺陷為 1.0；500 點理論標準差約 0.045）
    rho = _spearman(np.repeat(np.arange(50), _M).astype(float), ages.ravel().astype(float))
    assert abs(rho) < 0.2, (seed, rho)
    # 沉底月份涵蓋 10–12 月（舊 per_observation 沒有；觀測在 2025-01..12，年齡到 90 日）
    obs_ns = np.array(times[sites[0]], dtype=np.int64)[:, None]
    months = {
        datetime.fromtimestamp(int(value) / 1e9, tz=UTC).month
        for value in (obs_ns - ages.astype(np.int64) * _HOUR_NS).ravel()
    }
    assert {10, 11, 12}.issubset(months), (seed, sorted(months))
    # 年齡仍涵蓋整段 0..90 日（層層各用一次）
    assert ages.min() <= 5 and ages.max() >= _MAX_AGE_DAYS * 24 - 5


@pytest.mark.parametrize("member_count", [1, 2, 5])
def test_real_gap_shape_sampling_succeeds_for_other_member_counts(real_gap_case, member_count) -> None:
    """V 報告：M=1 為 15/20、M=2 為 2/20、M=5 為 0/20 個種子成功；修補後都應成功。"""

    for seed in (_SEED, 11, 12):
        sampling = _sample_real(real_gap_case, seed=seed, member_count=member_count)
        assert sampling.slot_count == 50 * member_count
        assert len(set(age for row in sampling.age_hours for age in row)) == 50 * member_count


def test_repair_is_reproducible_and_matches_audit_summary(real_gap_case) -> None:
    first = _sample_real(real_gap_case)
    assert first == _sample_real(real_gap_case)
    audits = [audit.as_metadata() for audit in first.rejection_audit]
    summary = summarize_member_permutation_repair(audits, slot_permutation=first.slot_permutation)
    assert summary["algorithm_id"] == BED_RESIDENCE_PER_MEMBER_PERMUTATION_REPAIR_ALGORITHM_ID
    assert summary["initially_infeasible_slot_count"] == len(
        _random_permutation_infeasible_slots(real_gap_case, seed=_SEED, member_count=_M)
    )
    assert summary["repair_step_count"] == len(first.repair_steps)
    assert summary["swap_step_count"] + summary["matching_step_count"] == len(first.repair_steps)
    assert summary["changed_slot_count"] >= summary["initially_infeasible_slot_count"]
    # method id 已升版；排列修補不應動到不需修補的槽位亂數流
    assert "feasibility_repair" in BED_RESIDENCE_PER_MEMBER_STRATIFIED_SAMPLING_METHOD_ID
    # 修補前排列由種子重算，載入器據此核對
    assert first.initial_slot_permutation == recompute_member_initial_slot_permutation(_SEED, 500)
    # 無缺口時修補是空操作：與沒有任何不可行槽位的結果一致
    clean = _sample()
    assert clean.repair_steps == () and clean.slot_permutation == clean.initial_slot_permutation
    assert all(audit.initially_infeasible is False for audit in clean.rejection_audit)


def test_summary_rejects_malformed_repair_audit() -> None:
    clean = _sample()
    audits = [audit.as_metadata() for audit in clean.rejection_audit]
    summarize_member_permutation_repair(audits, slot_permutation=clean.slot_permutation)
    broken = [dict(item) for item in audits]
    del broken[3]["repair_steps"]
    with pytest.raises(ValueError, match="修補稽核欄位"):
        summarize_member_permutation_repair(broken, slot_permutation=clean.slot_permutation)
    broken = [dict(item) for item in audits]
    broken[0]["initial_stratum_index"] = broken[1]["initial_stratum_index"]
    with pytest.raises(ValueError, match="排列"):
        summarize_member_permutation_repair(broken, slot_permutation=clean.slot_permutation)


def _feasible_from_sets(allowed: list[set[int]]) -> list[list[bool]]:
    size = len(allowed)
    return [[stratum in allowed[rank] for stratum in range(size)] for rank in range(size)]


def test_repair_falls_back_to_augmenting_path_when_no_single_swap_exists() -> None:
    """單次交換無解、但完全配對存在的小例子：必須改用增廣路徑而且結果可行。

    三個槽位（member_count=1，rank=槽位）：槽位 0 可用分層 {0,1}、槽位 1 只能用 {1}、
    槽位 2 可用 {1,2}。完全配對只有 (0→0, 1→1, 2→2)。修補前排列 [1, 2, 0]：槽位 1 與
    槽位 2 都不可行；槽位 1 沒有任何單次交換夥伴，必須用增廣路徑讓槽位 2 讓出分層 1。
    """

    feasible = _feasible_from_sets([{0, 1}, {1}, {1, 2}])
    permutation, steps = bed_residence_module._repair_member_slot_permutation(
        initial_permutation=(1, 2, 0),
        feasible=feasible,
        member_count=1,
        generator=bed_residence_module._member_repair_generator(5),
    )
    assert permutation == (0, 1, 2)
    kinds = {kind for kind, _ in steps}
    assert "matching_path" in kinds


def test_repair_raises_when_no_perfect_matching_exists() -> None:
    """兩個槽位都只能用同一分層：資料本身不允許，必須丟例外而不是輸出不完整排列。"""

    feasible = _feasible_from_sets([{0}, {0}])
    with pytest.raises(ValueError, match="無法完全配對"):
        bed_residence_module._repair_member_slot_permutation(
            initial_permutation=(0, 1),
            feasible=feasible,
            member_count=1,
            generator=bed_residence_module._member_repair_generator(5),
        )


def test_repair_leaves_feasible_permutation_untouched() -> None:
    feasible = _feasible_from_sets([{0, 1}, {0, 1}, {2}])
    permutation, steps = bed_residence_module._repair_member_slot_permutation(
        initial_permutation=(1, 0, 2),
        feasible=feasible,
        member_count=1,
        generator=bed_residence_module._member_repair_generator(1),
    )
    assert permutation == (1, 0, 2) and steps == ()
