"""WP-51-V 任務 C：以 .203 正式母體（v3-newsettling）的真實觀測時刻與 OCM 可用時間軸，
實際呼叫 per_member 抽樣器，回報 Spearman、沉底時刻月份分布、拒絕次數。

用法：python c_real_sampling.py <母體目錄（含 common-input/）> [seed] [M]
只讀輸入，不寫任何輸入檔。
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from lagrangian_backtracking.bed_residence import (
    sample_bed_residence_member_ages_conditioned_on_availability as sample_member,
)

HOUR = 3_600 * 10**9
SITE_REGION = {"gongliao": "A", "guishan": "A", "hsinchu": "B", "nanwan": "C", "lienchiang": "D"}


def parse_utc(text: str) -> int:
    dt = datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return int(dt.timestamp()) * 10**9


def spearman(a, b) -> float:
    ra = np.argsort(np.argsort(a, kind="stable"), kind="stable").astype(float)
    rb = np.argsort(np.argsort(b, kind="stable"), kind="stable").astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def available_axis(product: dict) -> set[int]:
    ct = product["canonical_time"]
    start = parse_utc(product["canonical_time"]["time_start_utc"])
    end = parse_utc(ct["time_end_utc"])
    hours = set(range(start, end + HOUR, HOUR))
    for gap in ct["gaps"]:
        lo, hi = parse_utc(gap["before_utc"]), parse_utc(gap["after_utc"])
        for t in range(lo + HOUR, hi, HOUR):
            hours.discard(t)
    if len(hours) != ct["canonical_time_count"]:
        raise SystemExit(f"可用時間軸重建與 canonical_time_count 不符：{len(hours)} != {ct['canonical_time_count']}")
    return hours


def main() -> None:
    root = Path(sys.argv[1])
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 20260916
    members = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    arrival = json.load(open(root / "common-input" / "arrival.json"))
    inv = json.load(open(root / "common-input" / "forcing_inventory.json"))
    avail_by_region = {
        p["analysis_region_id"]: available_axis(p) for p in inv["products"] if p["product"] == "ocm_native"
    }
    obs: dict[str, list[int]] = {s: [] for s in SITE_REGION}
    for rec in arrival["records"]:
        obs[rec["study_site_id"]].append(int(rec["metadata"]["observation_time_utc_ns"]))
    for s in obs:
        obs[s] = sorted(obs[s])
        assert len(obs[s]) == 50 and len(set(obs[s])) == 50, (s, len(obs[s]))
    avail = {s: avail_by_region[r] for s, r in SITE_REGION.items()}
    print("觀測時刻範圍：")
    for s in obs:
        print(f"  {s}: {datetime.fromtimestamp(obs[s][0]/1e9, timezone.utc):%Y-%m-%d} .. {datetime.fromtimestamp(obs[s][-1]/1e9, timezone.utc):%Y-%m-%d}")
    res = sample_member(
        observation_times_by_site=obs,
        available_time_ns_by_site=avail,
        maximum_age_days=90,
        sample_count=50,
        member_count=members,
        seed=seed,
    )
    age = np.array(res.age_hours)  # [rank][member]
    slots = age.size
    rank = np.repeat(np.arange(50), members)
    flat = age.reshape(-1)
    print(f"\n槽位數 {slots}；年齡範圍 {flat.min()}..{flat.max()} h；互異 {len(set(flat.tolist()))}")
    print(f"Spearman(年齡, 觀測 rank)（500 槽位全體）= {spearman(flat, rank):+.4f}")
    per_member = [spearman(age[:, m], np.arange(50)) for m in range(members)]
    print("逐成員 Spearman（50 點）：" + ", ".join(f"{v:+.2f}" for v in per_member))
    print(f"逐成員 Spearman 平均 {np.mean(per_member):+.3f}、絕對值最大 {np.max(np.abs(per_member)):.3f}")
    # 以第 rank 個觀測的日期做 Spearman：對每站
    for s in obs:
        obs_rank_time = np.repeat(np.array(obs[s]), members)
        print(f"  {s}: Spearman(年齡, 觀測 UTC) = {spearman(flat, obs_rank_time):+.4f}")
    # 排列品質
    perm = np.array(res.slot_permutation)
    print(f"slot_permutation 為排列：{sorted(perm.tolist()) == list(range(slots))}；Spearman(slot, stratum)={spearman(np.arange(slots), perm):+.4f}")
    print(f"固定點數（slot==stratum）={(perm == np.arange(slots)).sum()}（期望約 1）")
    # 各分層是否各用一次
    strata = np.array(res.stratum_index).reshape(-1)
    print(f"stratum 覆蓋 {len(set(strata.tolist()))}/{slots}")
    # 年齡分布均勻性：每 10 天一格
    hist, _ = np.histogram(flat, bins=np.arange(0, 91 * 24 + 1, 240))
    print("年齡直方圖（每 10 日一格，共 9 格）：", hist.tolist())
    # 沉底時刻月份分布，逐站
    print("\n沉底時刻月份分布（每站 500 槽位）：")
    allc = Counter()
    for s in obs:
        t = np.repeat(np.array(obs[s]), members) - flat * HOUR
        months = Counter(
            (datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=int(v // 10**9))).strftime("%Y-%m") for v in t
        )
        allc.update(months)
        print(f"  {s}: " + " ".join(f"{k}:{months[k]}" for k in sorted(months)))
        # 可用性核對
        bad = sum(1 for v in t.tolist() if v not in avail[s])
        mn = datetime.fromtimestamp(t.min() / 1e9, timezone.utc)
        mx = datetime.fromtimestamp(t.max() / 1e9, timezone.utc)
        print(f"    沉底最早 {mn:%Y-%m-%d %H} 最晚 {mx:%Y-%m-%d %H}；不在可用軸的槽位 {bad}")
    print("  五站合計: " + " ".join(f"{k}:{allc[k]}" for k in sorted(allc)))
    q4 = sum(v for k, v in allc.items() if k[5:7] in ("10", "11", "12"))
    print(f"  10–12 月（任一年）比例 {q4}/{sum(allc.values())} = {q4/sum(allc.values()):.3f}")
    # 拒絕統計
    rej = [len(a.rejected_age_hours) for a in res.rejection_audit]
    print(f"\n拒絕總次數 {sum(rej)}；有拒絕的槽位 {sum(1 for r in rej if r)}；單槽位最大 {max(rej)}；平均候選數 {np.mean([a.candidate_count for a in res.rejection_audit]):.2f}")
    by_site = Counter()
    for a in res.rejection_audit:
        for _age, sites in a.rejected_by_site:
            for s in sites:
                by_site[s] += 1
    print("拒絕時缺資料的站點計次：", dict(by_site))
    # 多個種子的穩健性
    print("\n多種子（Spearman 全體、拒絕次數）：")
    for sd in (1, 2, 3, 20260916, 20261007):
        r2 = sample_member(observation_times_by_site=obs, available_time_ns_by_site=avail, maximum_age_days=90,
                           sample_count=50, member_count=members, seed=sd)
        a2 = np.array(r2.age_hours).reshape(-1)
        print(f"  seed={sd}: rho={spearman(a2, rank):+.3f}, 拒絕={sum(len(a.rejected_age_hours) for a in r2.rejection_audit)}")


if __name__ == "__main__":
    main()
