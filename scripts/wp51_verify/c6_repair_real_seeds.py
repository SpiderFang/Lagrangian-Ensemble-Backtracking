"""WP-51 整合：以 .203 正式母體（v3-newsettling）的真實觀測時刻與 OCM 可用時間軸，
對「排列後可行性修補」版抽樣器做多種子與多成員數的成功率、修補次數、Spearman、月份涵蓋統計。

用法：python c6_repair_real_seeds.py <母體目錄（含 common-input/）> [種子數=200]
只讀輸入，不寫任何輸入檔。需先 ``ulimit -n`` 夠大不是必要（不開大量檔案）。"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c_real_sampling as c  # noqa: E402
from lagrangian_backtracking.bed_residence import (  # noqa: E402
    sample_bed_residence_member_ages_conditioned_on_availability as sample_member,
)


def main() -> None:
    root = Path(sys.argv[1])
    seed_count = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    arrival = json.load(open(root / "common-input" / "arrival.json"))
    inv = json.load(open(root / "common-input" / "forcing_inventory.json"))
    region_axis = {p["analysis_region_id"]: c.available_axis(p) for p in inv["products"] if p["product"] == "ocm_native"}
    obs = {s: [] for s in c.SITE_REGION}
    for rec in arrival["records"]:
        obs[rec["study_site_id"]].append(int(rec["metadata"]["observation_time_utc_ns"]))
    obs = {s: sorted(v) for s, v in obs.items()}
    avail = {s: region_axis[r] for s, r in c.SITE_REGION.items()}
    seeds = list(range(1, seed_count + 1)) + [20260916, 20261007]
    for members in (10, 5, 2, 1):
        use = seeds if members == 10 else seeds[:20] + [20260916, 20261007]
        ok = fail = 0
        bad0, swaps, matches, changed, rho, rej, q4, months_ok = [], [], [], [], [], [], [], 0
        for sd in use:
            try:
                r = sample_member(observation_times_by_site=obs, available_time_ns_by_site=avail, maximum_age_days=90,
                                  sample_count=50, member_count=members, seed=sd)
            except ValueError as err:
                fail += 1
                print(f"  M={members} seed={sd} 失敗：{err}")
                continue
            ok += 1
            age = np.array(r.age_hours)
            flat = age.reshape(-1)
            rank = np.repeat(np.arange(50), members)
            bad0.append(sum(1 for a in r.rejection_audit if a.initially_infeasible))
            swaps.append(sum(1 for k, _ in r.repair_steps if k == "swap"))
            matches.append(sum(1 for k, _ in r.repair_steps if k == "matching_path"))
            changed.append(sum(1 for i, (a, b) in enumerate(zip(r.initial_slot_permutation, r.slot_permutation)) if a != b))
            rho.append(c.spearman(flat, rank))
            rej.append(sum(len(a.rejected_age_hours) for a in r.rejection_audit))
            # 可用性與月份
            allc = Counter()
            for s in obs:
                t = np.repeat(np.array(obs[s]), members) - flat * c.HOUR
                assert all(int(v) in avail[s] for v in t.tolist()), "沉底整點不在可用軸內"
                allc.update((datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=int(v // 10**9))).strftime("%m") for v in t)
            q4.append(sum(allc[m] for m in ("10", "11", "12")) / sum(allc.values()))
            months_ok += int(all(allc[m] > 0 for m in ("10", "11", "12")))
        def stat(x): return f"min {min(x)} / 中位 {np.median(x):.0f} / max {max(x)}" if x else "-"
        print(f"\nM={members}：種子 {len(use)} 個，成功 {ok}、失敗 {fail}")
        if ok:
            print(f"  修補前不可行槽位數：{stat(bad0)}")
            print(f"  單次交換步數：{stat(swaps)}；增廣路徑步數：{stat(matches)}（>0 的種子 {sum(1 for m in matches if m)} 個）")
            print(f"  被動到的槽位數：{stat(changed)}；層內拒絕次數：{stat(rej)}")
            print(f"  Spearman(年齡, 觀測 rank)：平均 {np.mean(rho):+.3f}、標準差 {np.std(rho):.3f}、最大絕對值 {max(abs(x) for x in rho):.3f}")
            print(f"  沉底 10–12 月（任一年）比例：平均 {np.mean(q4):.3f}、最小 {min(q4):.3f}；10、11、12 月皆有沉底的種子 {months_ok}/{ok}")


if __name__ == "__main__":
    main()
