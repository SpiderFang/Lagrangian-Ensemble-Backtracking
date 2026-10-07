"""WP-51-V 任務 C 補充：真實 OCM 缺口下，(觀測 rank, 分層) 組合的可行性，
以及隨機排列後「整層無可用候選」的槽位數（即 per_member 抽樣器 fail-closed 的原因）。
用法：python c2_feasibility.py <母體目錄> [M]"""
import json, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c_real_sampling as c
from lagrangian_backtracking import bed_residence as br

root = Path(sys.argv[1]); M = int(sys.argv[2]) if len(sys.argv) > 2 else 10
arrival = json.load(open(root / "common-input" / "arrival.json"))
inv = json.load(open(root / "common-input" / "forcing_inventory.json"))
avail_r = {p["analysis_region_id"]: c.available_axis(p) for p in inv["products"] if p["product"] == "ocm_native"}
obs = {s: [] for s in c.SITE_REGION}
for rec in arrival["records"]:
    obs[rec["study_site_id"]].append(int(rec["metadata"]["observation_time_utc_ns"]))
obs = {s: sorted(v) for s, v in obs.items()}
avail = {s: avail_r[r] for s, r in c.SITE_REGION.items()}
sites = sorted(obs)
S = 50 * M
H = c.HOUR
# feasible[rank, stratum] = 該層存在某年齡使五站沉底整點皆可用
feas = np.zeros((50, S), dtype=bool)
ncand_ok = np.zeros((50, S), dtype=int)
for st in range(S):
    lo, hi = br._stratum_bounds(maximum_age_days=90, sample_count=S, stratum_index=st)
    for rk in range(50):
        ok = 0
        for age in range(lo, hi):
            if all(obs[s][rk] - age * H in avail[s] for s in sites):
                ok += 1
        ncand_ok[rk, st] = ok
        feas[rk, st] = ok > 0
sizes = [br._stratum_bounds(maximum_age_days=90, sample_count=S, stratum_index=st) for st in range(S)]
w = np.array([b - a for a, b in sizes])
print(f"M={M} 分層數 {S}；每層候選小時數：最小 {w.min()} 最大 {w.max()} 平均 {w.mean():.2f}")
print(f"(rank, 層) 共 {50*S}；無可用候選（infeasible）{int((~feas).sum())} = {(~feas).mean()*100:.2f}%")
print(f"  每個 rank 的 infeasible 層數：最小 {(~feas).sum(1).min()} 最大 {(~feas).sum(1).max()}")
print(f"  每一層的 infeasible rank 數：最小 {(~feas).sum(0).min()} 最大 {(~feas).sum(0).max()}")
# 在隨機排列下的失敗槽位數
fails = []
for seed in list(range(1, 201)) + [20260916, 20261007]:
    perm = np.array([int(v) for v in br._member_stratified_generator(seed).permutation(S)])
    bad = 0
    for slot in range(S):
        rk = slot // M
        if not feas[rk, perm[slot]]:
            bad += 1
    fails.append((seed, bad))
b = np.array([x[1] for x in fails])
print(f"202 個種子下，隨機排列後整層無候選的槽位數：最小 {b.min()} 中位 {int(np.median(b))} 最大 {b.max()}；失敗數為 0 的種子數 {(b == 0).sum()}")
print("config 種子 20260916 →", dict(fails)[20260916], "；20261007 →", dict(fails)[20261007])
# 是否存在完全匹配（可行）？以匈牙利／二分圖匹配檢查（每 rank 恰有 M 個槽位）
try:
    from scipy.optimize import linear_sum_assignment
    cost = np.where(np.repeat(feas, M, axis=0), 0, 1)  # (slots, strata)
    r, cc = linear_sum_assignment(cost)
    print(f"最小代價完全匹配的不可行配對數 = {int(cost[r, cc].sum())}（0 表示存在全可行的配對）")
except Exception as e:
    print("scipy 不可用：", e)
