# 以圖面幾何驗算：50 個標記是否各自可見（位置換算成英寸，與 arrival_figs.render_overview 之版面一致）
import sys, os, itertools, json
sys.path.insert(0, os.path.expanduser("~/Workspace/lbt-repo-wp63/scripts/wp61"))
sys.path.insert(0, os.path.expanduser("~/Workspace/lbt-repo-wp63/src"))
import numpy as np
import arrival_check as ac
import arrival_figs as af
recs = ac.load_manifest_records()
S = {s: ac.load_series(s) for s in ac.SITES}
R = {s: ac.analyze_site(s, recs[s], S[s], None) for s in ac.SITES}
stats = {s: af.site_stats(S[s], R[s]["rows"], R[s]["idx_of"], None) for s in ac.SITES}
W = af.OV_W; left, right = af.OV_LEFT, 0.42; AXW = W - left - right; AXH = af.OV_HEIGHTS[0]
DIAM = 7.5 / 72.0  # 類別標記直徑（英寸）
out = {}
for site in ["gongliao", "guishan", "hsinchu", "nanwan", "lienchiang"]:
    s = S[site]; rows = R[site]["rows"]
    lo, up, _ = af.axis_plan(stats, "S", site)["strength"]
    days = (s.times.astype(np.int64) - af.T0) / af.DAY_NS
    pts = []; utc_of = []
    for r in rows:
        i = R[site]["idx_of"][int(np.datetime64(r["utc"].replace("Z", ""), "ns").astype("int64"))]
        x = AXW * days[i] / 365.0; y = AXH * (s.strength[i] - lo) / (up - lo)
        pts.append((x, y, r["tide_class"], r["phase_or_event"])); utc_of.append(r["utc"])
    d = []
    for a, b in itertools.combinations(range(len(pts)), 2):
        dist = float(np.hypot(pts[a][0] - pts[b][0], pts[a][1] - pts[b][1])); d.append((dist, a, b))
    d.sort()
    ov = [t for t in d if t[0] < DIAM]
    same = [t for t in d if t[0] < DIAM * 0.25 and pts[t[1]][2:] == pts[t[2]][2:]]      # 同類別同潮相、距離小於標記直徑 1/4：形狀完全相同，只能靠顏色加深辨識
    shape = [t for t in d if t[0] < DIAM * 0.25 and pts[t[1]][2:] != pts[t[2]][2:]]     # 距離很近但形狀不同（▲▼疊成六角形等）
    out[site] = dict(n=len(pts), overlapping_pairs=len(ov), identical_shape_near_coincident=len(same), different_shape_near_coincident=len(shape), min_dist_in=round(d[0][0], 4))
    print(site, out[site])
    for dist, a, b in same:
        print("   形狀相同且幾乎重合：距離 %.4f in；" % dist, pts[a][2:], utc_of[a], utc_of[b])
json.dump(out, open("/tmp/count_markers.json", "w"))
