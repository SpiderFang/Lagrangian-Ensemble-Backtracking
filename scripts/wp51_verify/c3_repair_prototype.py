"""WP-51-V 任務 C（原型）：分支內抽樣器在真實缺口下 fail-closed，所以這裡用「隨機排列＋
不可行槽位交換修補」的驗證用原型（非分支程式）產生 500 槽位，量測若修正後的統計性質。
修補規則：隨機排列後，對每個不可行槽位 i（rank_i, 層 a），依種子固定的順序找另一槽位 j
（rank_j, 層 b），使 (rank_i, b) 與 (rank_j, a) 都可行，交換兩者的層。層仍然一對一使用。
然後在層內以隨機順序挑第一個五站皆可用的整點（與分支抽樣器相同）。
用法：python c3_repair_prototype.py <母體目錄> [seed] [M]"""
import json, sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c_real_sampling as c
from lagrangian_backtracking import bed_residence as br

root = Path(sys.argv[1]); seed = int(sys.argv[2]) if len(sys.argv) > 2 else 20260916
M = int(sys.argv[3]) if len(sys.argv) > 3 else 10
arrival = json.load(open(root / "common-input" / "arrival.json"))
inv = json.load(open(root / "common-input" / "forcing_inventory.json"))
avail_r = {p["analysis_region_id"]: c.available_axis(p) for p in inv["products"] if p["product"] == "ocm_native"}
obs = {s: [] for s in c.SITE_REGION}
for rec in arrival["records"]:
    obs[rec["study_site_id"]].append(int(rec["metadata"]["observation_time_utc_ns"]))
obs = {s: sorted(v) for s, v in obs.items()}
avail = {s: avail_r[r] for s, r in c.SITE_REGION.items()}
sites = sorted(obs); S = 50 * M; H = c.HOUR
cand = {}  # (rank, stratum) -> 可用年齡清單
for st in range(S):
    lo, hi = br._stratum_bounds(maximum_age_days=90, sample_count=S, stratum_index=st)
    for rk in range(50):
        cand[(rk, st)] = [a for a in range(lo, hi) if all(obs[s][rk] - a * H in avail[s] for s in sites)]
feas = lambda rk, st: len(cand[(rk, st)]) > 0

g = br._member_stratified_generator(seed)
perm = [int(v) for v in g.permutation(S)]
bad0 = [i for i in range(S) if not feas(i // M, perm[i])]
rng = np.random.default_rng([seed, 51])
repaired = 0
for i in bad0:
    order = rng.permutation(S)
    for j in order:
        j = int(j)
        if j == i: continue
        if feas(i // M, perm[j]) and feas(j // M, perm[i]):
            perm[i], perm[j] = perm[j], perm[i]; repaired += 1; break
    else:
        raise SystemExit(f"槽位 {i} 無法以單次交換修補")
assert sorted(perm) == list(range(S)) and all(feas(i // M, perm[i]) for i in range(S))
print(f"seed={seed} M={M}：隨機排列後不可行槽位 {len(bad0)}；單次交換修補 {repaired}（受影響槽位約 {2*repaired}/{S}）")
age = np.zeros((50, M), dtype=int); rej_total = 0
for i in range(S):
    rk, m = divmod(i, M)
    lo, hi = br._stratum_bounds(maximum_age_days=90, sample_count=S, stratum_index=perm[i])
    order = [int(v) for v in g.permutation(np.arange(lo, hi))]
    for a in order:
        if a in cand[(rk, perm[i])]:
            age[rk, m] = a; break
        rej_total += 1
flat = age.reshape(-1); rank = np.repeat(np.arange(50), M)
print(f"原型：年齡互異 {len(set(flat.tolist()))}/{S}；Spearman(年齡, 觀測 rank) = {c.spearman(flat, rank):+.4f}；層內拒絕 {rej_total}")
print("逐成員 Spearman 絕對值最大 {:.3f}".format(max(abs(c.spearman(age[:, m], np.arange(50))) for m in range(M))))
hist, _ = np.histogram(flat, bins=np.arange(0, 91 * 24 + 1, 240)); print("年齡直方圖（10 日一格）", hist.tolist())
allc = Counter()
for s in sites:
    t = np.repeat(np.array(obs[s]), M) - flat * H
    mo = Counter((datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=int(v // 10**9))).strftime("%Y-%m") for v in t)
    allc.update(mo)
    print(f"  {s}: " + " ".join(f"{k}:{mo[k]}" for k in sorted(mo)), "| 不可用整點", sum(1 for v in t.tolist() if v not in avail[s]))
print("  五站合計 " + " ".join(f"{k}:{allc[k]}" for k in sorted(allc)))
q4 = sum(v for k, v in allc.items() if k[5:7] in ("10", "11", "12"))
print(f"  10–12 月（任一年）{q4}/{sum(allc.values())} = {q4/sum(allc.values()):.3f}")
by_year = Counter(); 
for k, v in allc.items(): by_year[k[:4]] += v
print("  依年份", dict(by_year))
if len(sys.argv) > 4:
    np.save(sys.argv[4], age)
    print("已存年齡矩陣", sys.argv[4])
