"""WP-38：從 npz 直接串流重算（不一次載入全部）：各站 status==0 總樣本數、離開流場粒子數、快照時刻仍在漂流粒子數。

用法：recount_stream.py <results 目錄> <站 id[,站 id...]> [輸出 json]
只讀 trajectory_status、final_status 兩個陣列。
"""

import collections
import json
import sys
from pathlib import Path

import numpy as np

HOURS = (72, 168, 336, 720)
results_dir = Path(sys.argv[1])
sites = set(sys.argv[2].split(","))
tot = collections.Counter()
cross = collections.Counter()
parts = collections.Counter()
active = collections.defaultdict(collections.Counter)
files = collections.Counter()
for f in sorted(results_dir.glob("*.driver_result.npz")):
    site = f.name.split("__")[1]
    if site not in sites:
        continue
    with np.load(f) as z:
        ts = z["trajectory_status"]
        fs = z["final_status"]
        tot[site] += int(np.count_nonzero(ts == 0))
        cross[site] += int(np.count_nonzero(fs == 1))
        parts[site] += int(fs.size)
        for h in HOURS:
            if ts.shape[0] > h:
                active[site][h] += int(np.count_nonzero(ts[h] == 0))
    files[site] += 1
out = {
    s: {
        "files": files[s],
        "particles": parts[s],
        "total_particle_hours": tot[s],
        "crossed": cross[s],
        "active": {str(h): active[s][h] for h in HOURS},
    }
    for s in sorted(sites)
}
text = json.dumps(out, ensure_ascii=False, indent=1)
if len(sys.argv) > 3:
    Path(sys.argv[3]).write_text(text, encoding="utf-8")
print(text)
