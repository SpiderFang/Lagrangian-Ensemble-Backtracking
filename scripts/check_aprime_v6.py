#!/usr/bin/env python3
"""WP-53：檢查 A 版 ocm_native v6（母體建置用）：時間軸完整、重建列與 patch 一致、其餘列與 v5 一致。只讀。"""
import json, sys
from pathlib import Path
import numpy as np

H = 3600 * 10**9
dom, v5r, ap6, patch_root, bad_dir = sys.argv[1:6]
v5, ap, pt = Path(v5r) / dom / "months", Path(ap6) / dom / "months", Path(patch_root) / dom / "months"
bad = {int(x) for x in json.load(open(Path(bad_dir) / f"bad_replay_{dom}.json"))["bad"]}
axes = {m.name: np.load(m / "time_utc_ns.npy") for m in sorted(ap.iterdir())}
allt = np.concatenate(list(axes.values()))
u = np.unique(allt)
full = np.arange(u.min(), u.max() + 1, H)
miss = np.setdiff1d(full, u)
bad_axes = [m for m, a in axes.items() if np.any(np.diff(a) <= 0)]
print(dom, "範圍", np.datetime64(int(u.min()), "ns"), "→", np.datetime64(int(u.max()), "ns"), "獨立小時", u.size, "缺小時", miss.size, "月內非遞增月份", bad_axes)
ploc = {}
for pm in sorted(p.name for p in pt.iterdir() if p.is_dir()):
    for i, t in enumerate(np.load(pt / pm / "time_utc_ns.npy")):
        ploc[int(t)] = (pm, i)
rng = np.random.default_rng(53)
ok = True
checked = 0
for m, a in axes.items():
    if m < "202501" or not (v5 / m).is_dir():
        continue
    t5 = np.load(v5 / m / "time_utc_ns.npy")
    s5 = {int(t): i for i, t in enumerate(t5)}
    for name in ("zcor", "elev", "wetdry_elem"):
        arr = np.load(ap / m / f"{name}.npy", mmap_mode="r")
        src = np.load(v5 / m / f"{name}.npy", mmap_mode="r")
        if arr.shape[0] != a.size or arr.shape[1:] != src.shape[1:]:
            ok = False; print("形狀不符", m, name, arr.shape, src.shape); continue
        obs_rows = [j for j, t in enumerate(a) if int(t) in s5 and int(t) not in bad]
        pick = rng.choice(obs_rows, size=min(12, len(obs_rows)), replace=False)
        for j in pick:
            if not np.array_equal(np.asarray(arr[j]), np.asarray(src[s5[int(a[j])]]), equal_nan=True):
                ok = False; print("與 v5 不同", m, name, j)
            checked += 1
        for j, t in enumerate(a):
            if int(t) in ploc and (int(t) in bad or int(t) not in s5):
                pm, pi = ploc[int(t)]
                p_arr = np.load(pt / pm / f"{name}.npy", mmap_mode="r")
                if not np.array_equal(np.asarray(arr[j]), np.asarray(p_arr[pi]).astype(arr.dtype), equal_nan=True):
                    ok = False; print("與 patch 不同", m, name, j)
                checked += 1
print(dom, "抽查列數", checked, "全部一致" if ok else "有不一致")
sys.exit(0 if ok and miss.size == 0 else 1)
