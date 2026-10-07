"""WP-51-V 補充：per_member 的沉底時刻是否會讓受體所在 face 在該時刻為乾（dynamic 初始條件會 fail-closed）。
用 c3 原型年齡矩陣（.npy）＋真實受體 face＋真實 OCM wetdry_elem。只讀。
用法：python e_dry_face.py <母體目錄> <年齡矩陣.npy> <ocm_native 根目錄>"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c_real_sampling as c

root = Path(sys.argv[1]); age = np.load(sys.argv[2]); ocm = Path(sys.argv[3]); M = age.shape[1]; H = c.HOUR
FD = {"A": "northeast_taiwan_common_cache_v3", "B": "hsinchu_cache_v3", "C": "houwan_nmmba_cache_v3", "D": "lienchiang_common_cache_v3"}
arrival = json.load(open(root / "common-input" / "arrival.json"))
rec = json.load(open(root / "common-input" / "receptor.json"))["records"]
obs = {s: [] for s in c.SITE_REGION}
for a in arrival["records"]:
    obs[a["study_site_id"]].append(int(a["metadata"]["observation_time_utc_ns"]))
obs = {s: sorted(v) for s, v in obs.items()}
faces = {}
for r in rec:
    faces.setdefault(r["study_site_id"], set()).add(int(r["metadata"]["source_face_local_index"]))
cache = {}
def load(region):
    if region in cache: return cache[region]
    months = sorted((ocm / FD[region] / "months").glob("20*"))
    ts, mi, ix = [], [], []
    arrs = {}
    for k, m in enumerate(months):
        t = np.load(m / "time_utc_ns.npy", mmap_mode="r")
        arrs[k] = np.load(m / "wetdry_elem.npy", mmap_mode="r")
        ts.append(np.asarray(t)); mi.append(np.full(len(t), k)); ix.append(np.arange(len(t)))
    ts = np.concatenate(ts); mi = np.concatenate(mi); ix = np.concatenate(ix)
    idx = {}
    for t, k, i in zip(ts.tolist(), mi.tolist(), ix.tolist()): idx[t] = (k, i)   # prefer-last
    cache[region] = (idx, arrs, [m.name for m in months]); return cache[region]
tot_dry = 0
for s, reg in c.SITE_REGION.items():
    idx, arrs, names = load(reg)
    dep = [int(o - age[r, m] * H) for r, o in enumerate(obs[s]) for m in range(M)]
    dry_pairs = 0; missing = 0; dry_times = set()
    for t in dep:
        if t not in idx: missing += 1; continue
        k, i = idx[t]
        row = np.asarray(arrs[k][i])
        for f in faces[s]:
            if int(round(float(row[f]))) != 0:
                dry_pairs += 1; dry_times.add(t)
    old = [int(a["time_utc_ns"]) for a in arrival["records"] if a["study_site_id"] == s]
    dry_old = sum(1 for t in old if t in idx for f in faces[s] if int(round(float(np.asarray(arrs[idx[t][0]][idx[t][1]])[f]))) != 0)
    print(f"{s}: 月份資料 {names[0]}..{names[-1]}；受體 face {len(faces[s])} 個；per_member 沉底時刻 {len(dep)}（OCM 無此時刻 {missing}）"
          f"；乾 face 配對 {dry_pairs}（涉及 {len(dry_times)} 個時刻）；現行 per_observation 沉底 {len(old)} 個時刻的乾 face 配對 {dry_old}")
    tot_dry += dry_pairs
print("per_member 乾 face 配對總數：", tot_dry)
