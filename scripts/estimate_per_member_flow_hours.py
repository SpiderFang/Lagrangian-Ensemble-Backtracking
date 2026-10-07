"""WP-51 A.5：per_observation 與 per_member 的小時主軸工作單元／切段／流場小時數合成估計。
只用合成觀測時刻（9 個月、每站 50 個）與真實的 per_member 抽樣器（五站共用 canonical 全小時軸）。"""
import sys, time
import numpy as np
from lagrangian_backtracking.bed_residence import sample_bed_residence_member_ages_conditioned_on_availability
from lagrangian_backtracking.native_runner.orchestrator import NativeWorkUnit, plan_native_hour_major_segments

HOUR = 3600 * 10**9
sites = {"gongliao": "ne", "guishan": "ne", "hsinchu": "hs", "nanwan": "hw", "lienchiang": "lj"}
t0 = int(np.datetime64("2025-01-15T00", "ns").astype(np.int64)); t1 = int(np.datetime64("2025-10-15T00", "ns").astype(np.int64))
rng = np.random.default_rng(7)
obs = {s: sorted(int(t0 + rng.integers(0, (t1 - t0) // HOUR) * HOUR) for _ in range(50)) for s in sites}
# 取 50 個相異
for s in sites:
    while len(set(obs[s])) < 50:
        obs[s] = sorted(set(obs[s]) | {int(t0 + rng.integers(0, (t1 - t0) // HOUR) * HOUR)})[:50]
lo = t0 - 91 * 24 * HOUR
avail = {s: range(lo, t1 + HOUR, HOUR) for s in sites}
M = 10
samp = sample_bed_residence_member_ages_conditioned_on_availability(
    observation_times_by_site=obs, available_time_ns_by_site=avail, maximum_age_days=90, sample_count=50, member_count=M, seed=20261007)
age = np.array(samp.age_hours)  # [rank][member]
# per_observation：舊配對（分層 n 配第 n 筆）——以均勻分層中點近似；依 rank 單調
old_age = np.round(np.linspace(0, 90 * 24, 50)).astype(int)

def build_units(mode):
    units = []
    for s, dom in sites.items():
        times = []
        if mode == "obs":
            times = [(obs[s][r] - int(old_age[r]) * HOUR, 2000) for r in range(50)]
        else:
            times = [(obs[s][r] - int(age[r, m]) * HOUR, 200) for r in range(50) for m in range(M)]
        # 同 (arrival, settling, site) 合併粒子數
        agg = {}
        for a, n in times:
            agg[a] = agg.get(a, 0) + n
        for a, n in agg.items():
            for k in range(10):
                units.append(NativeWorkUnit(unit_id=f"{s}_{a}_{k}", analysis_region_id=s, flow_domain_id=dom, study_site_id=s,
                    arrival_time_utc_ns=a, settling_velocity_mps=float(k + 1) * 1e-4,
                    particle_row_indices=np.arange(0, n, dtype=np.int64), input_hash=f"{s}{a}{k}"))
    return units

for mode in ("obs", "member"):
    units = build_units(mode)
    arr = np.array([u.arrival_time_utc_ns for u in units])
    print(f"== {mode}: units={len(units)} distinct_arrivals_total={len(set(arr.tolist()))} "
          f"span_days={(arr.max()-arr.min())/HOUR/24:.1f} range={np.datetime64(int(arr.min()),'ns')}..{np.datetime64(int(arr.max()),'ns')}")
    for H in (30, 60, 90):
        row = []
        for K in (1, 6, 12, 24, 36):
            _, _, reps = plan_native_hour_major_segments(units, segment_count=K, max_backtrack_seconds=H * 86400.0)
            tot = sum(sum(r.segment_hours_to_synthesize) for r in reps)
            mx = max(max(r.segment_hours_to_synthesize) for r in reps)
            row.append(f"K={K}: 總{tot}h 最長段{mx}h")
        print(f"  H={H}d  " + " | ".join(row))
# 每小時單元迴圈與 _record_departures 線性搜尋開銷（合成）
for nunit, per in ((250, 2000), (2500, 200)):
    n = nunit * per
    starts = np.arange(nunit) * per
    mask = np.zeros(n, dtype=bool)
    t = time.perf_counter()
    for _ in range(20):
        for s in starts:
            np.any(mask[s:s + per])
    hourly = (time.perf_counter() - t) / 20
    rows = rng.integers(0, n, 2000)
    t = time.perf_counter()
    for r in rows:
        for s in starts:
            if s <= r < s + per:
                break
    dep = (time.perf_counter() - t) / 2000
    print(f"單元數={nunit}：每組每小時 np.any 迴圈 {hourly*1e3:.2f} ms；單列離場線性搜尋平均 {dep*1e6:.1f} us")
