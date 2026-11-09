import sys, os
sys.path.insert(0, os.path.expanduser("~/Workspace/lbt-repo-wp63/scripts/wp61")); sys.path.insert(0, os.path.expanduser("~/Workspace/lbt-repo-wp63/src"))
import numpy as np, arrival_check as ac, arrival_figs as af
recs = ac.load_manifest_records(); S = {s: ac.load_series(s) for s in ac.SITES}
R = {s: ac.analyze_site(s, recs[s], S[s], None) for s in ac.SITES}
stats = {s: af.site_stats(S[s], R[s]["rows"], R[s]["idx_of"], None) for s in ac.SITES}
for s in ac.SITES:
    st = S[s].strength[np.isfinite(S[s].strength)]
    q = np.nanpercentile(st, [1, 99]); rng = q[1] - q[0]
    out = []
    for v in ("U", "S"):
        lo, up, _ = af.axis_plan(stats, v, s)["strength"]
        out.append("%s 軸 %.2g–%.2g 曲線 1–99 百分位範圍佔軸高 %.0f%%（全距 %.0f%%）" % (v, lo, up, 100 * rng / (up - lo), 100 * (st.max() - st.min()) / (up - lo)))
    print(s, " | ".join(out))
