import sys
sys.path.insert(0, "/home/mustlab/Workspace/lbt-wp51/verify")
import json, glob
import c_real_sampling as c
from lagrangian_backtracking.bed_residence import sample_bed_residence_member_ages_conditioned_on_availability as S
root = glob.glob("/home/mustlab/Workspace/lbt-inputs/formal/v3-newsettling/2026*_m10_v3")[0]
arrival = json.load(open(root + "/common-input/arrival.json")); inv = json.load(open(root + "/common-input/forcing_inventory.json"))
ar = {p["analysis_region_id"]: c.available_axis(p) for p in inv["products"] if p["product"] == "ocm_native"}
obs = {s: [] for s in c.SITE_REGION}
for a in arrival["records"]: obs[a["study_site_id"]].append(int(a["metadata"]["observation_time_utc_ns"]))
obs = {s: sorted(v) for s, v in obs.items()}; av = {s: ar[r] for s, r in c.SITE_REGION.items()}
ok = 0
for sd in list(range(1, 21)) + [20260916, 20261007]:
    try:
        S(observation_times_by_site=obs, available_time_ns_by_site=av, maximum_age_days=90, sample_count=50, member_count=10, seed=sd); ok += 1; print(sd, "OK")
    except ValueError as e:
        print(sd, "ValueError", str(e)[-70:])
print("成功種子數", ok, "/ 22")
for M in (1, 2, 5):
    okm = 0
    for sd in range(1, 21):
        try:
            S(observation_times_by_site=obs, available_time_ns_by_site=av, maximum_age_days=90, sample_count=50, member_count=M, seed=sd); okm += 1
        except ValueError: pass
    print(f"M={M}：20 個種子中成功 {okm}")
