"""WP-51-V 項目 A：以「無缺口」可用軸檢查 per_member 抽樣器本身的性質（不受真實缺口影響）：
排列是否打亂、五站同槽位同年齡、種子可重現、輸入列順序不影響、拒絕針對配對後的實際觀測。
用法：python a1_sampler_properties.py <母體目錄>"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c_real_sampling as c
from lagrangian_backtracking.bed_residence import sample_bed_residence_member_ages_conditioned_on_availability as S

root = Path(sys.argv[1]); H = c.HOUR; M = 10
arrival = json.load(open(root / "common-input" / "arrival.json"))
obs = {s: [] for s in c.SITE_REGION}
for rec in arrival["records"]:
    obs[rec["study_site_id"]].append(int(rec["metadata"]["observation_time_utc_ns"]))
obs = {s: sorted(v) for s, v in obs.items()}
lo = min(min(v) for v in obs.values()) - 91 * 24 * H; hi = max(max(v) for v in obs.values()) + H
full = {s: range(lo, hi, H) for s in obs}
rhos = []
for seed in range(1, 31):
    r = S(observation_times_by_site=obs, available_time_ns_by_site=full, maximum_age_days=90, sample_count=50, member_count=M, seed=seed)
    a = np.array(r.age_hours).reshape(-1)
    rhos.append(c.spearman(a, np.repeat(np.arange(50), M)))
    assert sum(len(x.rejected_age_hours) for x in r.rejection_audit) == 0
print(f"無缺口、30 個種子：Spearman(年齡, 觀測 rank) 平均 {np.mean(rhos):+.3f}、標準差 {np.std(rhos):.3f}、絕對值最大 {np.max(np.abs(rhos)):.3f}（隨機 500 點的理論標準差 ≈ {1/np.sqrt(499):.3f}）")
r1 = S(observation_times_by_site=obs, available_time_ns_by_site=full, maximum_age_days=90, sample_count=50, member_count=M, seed=20260916)
r2 = S(observation_times_by_site={k: list(reversed(v)) for k, v in obs.items()}, available_time_ns_by_site=full, maximum_age_days=90, sample_count=50, member_count=M, seed=20260916)
print("同種子重現、輸入列順序反轉後結果相同：", r1.age_hours == r2.age_hours and r1.slot_permutation == r2.slot_permutation)
# 單一站缺口：把第 3 個觀測在某站的某些整點拿掉，確認拒絕只依該 rank 的該站觀測
avail = {s: set(range(lo, hi, H)) for s in obs}
a_ref = np.array(r1.age_hours)
tgt_rank, tgt_member = 7, 3
age0 = a_ref[tgt_rank, tgt_member]
avail["nanwan"].discard(obs["nanwan"][tgt_rank] - int(age0) * H)
r3 = S(observation_times_by_site=obs, available_time_ns_by_site=avail, maximum_age_days=90, sample_count=50, member_count=M, seed=20260916)
a3 = np.array(r3.age_hours)
aud = [x for x in r3.rejection_audit if x.observation_rank == tgt_rank and x.member_id == tgt_member][0]
print(f"nanwan 在 rank {tgt_rank} 缺 age={age0}h 的沉底整點 → 拒絕稽核 rejected={aud.rejected_age_hours} by_site={aud.rejected_by_site}；新年齡={a3[tgt_rank, tgt_member]}")
others_changed = int((a3 != a_ref).sum())
print(f"年齡有變動的槽位數 = {others_changed}（每個槽位固定消耗一次層內 permutation，拒絕不會位移後續亂數流，故只有該槽位改變）")
