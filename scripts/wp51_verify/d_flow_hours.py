"""WP-51-V 任務 D：以 .203 正式母體的真實到達時刻，估計 per_member 相對現行 per_observation 的
小時主軸「流場小時數」。per_member 的年齡取自 c3_repair_prototype.py 輸出的年齡矩陣（.npy）。
用法：python d_flow_hours.py <母體目錄> <年齡矩陣.npy> [K=42]"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c_real_sampling as c
from lagrangian_backtracking.native_runner.orchestrator import NativeWorkUnit, plan_native_hour_major_segments

root = Path(sys.argv[1]); age = np.load(sys.argv[2]); K = int(sys.argv[3]) if len(sys.argv) > 3 else 42
M = age.shape[1]; H = c.HOUR
arrival = json.load(open(root / "common-input" / "arrival.json"))
obs = {s: [] for s in c.SITE_REGION}; dep_old = {s: [] for s in c.SITE_REGION}
for rec in arrival["records"]:
    s = rec["study_site_id"]
    obs[s].append(int(rec["metadata"]["observation_time_utc_ns"])); dep_old[s].append(int(rec["time_utc_ns"]))
FD = {"A": "northeast", "B": "hsinchu", "C": "houwan", "D": "lienchiang"}

def units_for(times_by_site, per_arrival_particles):
    units = []
    for s, ts in times_by_site.items():
        fd = FD[c.SITE_REGION[s]]
        agg = {}
        for t in ts: agg[t] = agg.get(t, 0) + per_arrival_particles
        for t, n in agg.items():
            for k in range(10):
                units.append(NativeWorkUnit(unit_id=f"{s}_{t}_{k}", analysis_region_id=c.SITE_REGION[s], flow_domain_id=fd,
                    study_site_id=s, arrival_time_utc_ns=t, settling_velocity_mps=float(k + 1) * 1e-4,
                    particle_row_indices=np.arange(n, dtype=np.int64), input_hash=f"{s}{t}{k}"))
    return units

old_units = units_for(dep_old, 20 * M)  # 每 (site, arrival, settling) 單元：20 受體 × M 成員
new_ts = {s: [int(o - age[r, m] * H) for r, o in enumerate(sorted(obs[s])) for m in range(M)] for s in obs}
new_units = units_for(new_ts, 20)
for name, units in (("per_observation（現行，真實 v3 到達時刻）", old_units), ("per_member（原型年齡，真實觀測）", new_units)):
    arrs = sorted({u.arrival_time_utc_ns for u in units})
    print(f"== {name}：單元數 {len(units)}；相異到達時刻 {len(arrs)}；跨距 {(arrs[-1]-arrs[0])/H/24:.1f} 日")
    for Hd in (30, 60, 90):
        _, _, reps = plan_native_hour_major_segments(units, segment_count=K, max_backtrack_seconds=Hd * 86400.0)
        tot = sum(sum(r.segment_hours_to_synthesize) for r in reps); mx = max(max(r.segment_hours_to_synthesize) for r in reps)
        per = {r.flow_domain_id: sum(r.segment_hours_to_synthesize) for r in reps}
        print(f"  H={Hd}d K={K}/區：總流場小時 {tot}；最長單段 {mx}h；各區 {per}")
res = {}
for label, units in (("old", old_units), ("new", new_units)):
    _, _, reps = plan_native_hour_major_segments(units, segment_count=K, max_backtrack_seconds=90 * 86400.0)
    res[label] = sum(sum(r.segment_hours_to_synthesize) for r in reps)
print(f"H90 K={K}：per_member / per_observation = {res['new']/res['old']:.2f}x")
