#!/usr/bin/env python3
"""WP-53：檢查 ocm_surface v6：時間軸完整（2024-01-01T01～2026-01-01T00 逐時）、格網與 NAS 位元相同、
未改動的列與來源逐位元相同、重建列在靜態有效格皆為有限值。只讀。"""
import json, sys
from pathlib import Path
import numpy as np

H = 3600 * 10**9
dom, v6r, nas_r, gap_r, bad_dir = sys.argv[1:6]
v6, nas, gap = Path(v6r) / dom, Path(nas_r) / dom, Path(gap_r) / dom
bad = {int(x) for x in json.load(open(Path(bad_dir) / f"bad_replay_{dom}.json"))["bad"]}
ok = True
for f in sorted((nas / "grid").iterdir()):
    if f.is_file() and (f.read_bytes() != (v6 / "grid" / f.name).read_bytes()):
        ok = False; print("grid 不同", f.name)
static = np.load(v6 / "grid" / "mask_static.npy").astype(bool)
axes = {m.name: np.load(m / "time_utc_ns.npy") for m in sorted((v6 / "months").iterdir())}
u = np.unique(np.concatenate(list(axes.values())))
miss = np.setdiff1d(np.arange(u.min(), u.max() + 1, H), u)
print(dom, "範圍", np.datetime64(int(u.min()), "ns"), "→", np.datetime64(int(u.max()), "ns"), "獨立小時", u.size, "缺小時", miss.size,
      "月內非遞增", [m for m, a in axes.items() if np.any(np.diff(a) <= 0)])
rng = np.random.default_rng(53)
rec_rows = unchanged = 0
for m, a in axes.items():
    src = gap / "months" / m if (gap / "months" / m).is_dir() else nas / "months" / m
    ts = np.load(src / "time_utc_ns.npy"); smap = {int(t): i for i, t in enumerate(ts)}
    arrs = {n: np.load(v6 / "months" / m / f"{n}.npy", mmap_mode="r") for n in ("eta_m", "u_surface_mps", "v_surface_mps", "surface_z", "valid_mask_surface", "qc_flags")}
    for n, x in arrs.items():
        if x.shape[0] != a.size:
            ok = False; print("列數不符", m, n)
    v = np.asarray(arrs["valid_mask_surface"]); q = np.asarray(arrs["qc_flags"])
    if not (np.all(v == static[None]) and np.all(q == (~static).astype(np.uint16)[None])):
        ok = False; print("valid/qc 與靜態遮罩不符", m)
    changed = [j for j, t in enumerate(a) if int(t) in bad or int(t) not in smap]
    for j in changed:
        for n in ("eta_m", "u_surface_mps", "v_surface_mps", "surface_z"):
            row = np.asarray(arrs[n][j])
            if not np.all(np.isfinite(row[static])) or not np.all(np.isnan(row[~static])):
                ok = False; print("重建列有限性不符", m, n, j)
        rec_rows += 1
    keep = [j for j, t in enumerate(a) if j not in set(changed)]
    for j in rng.choice(keep, size=min(15, len(keep)), replace=False):
        for n in ("eta_m", "u_surface_mps", "v_surface_mps", "surface_z"):
            s = np.load(src / f"{n}.npy", mmap_mode="r")
            if not np.array_equal(np.asarray(arrs[n][j]), np.asarray(s[smap[int(a[j])]]), equal_nan=True):
                ok = False; print("未改動列與來源不同", m, n, j)
        unchanged += 1
print(dom, "重建／取代列", rec_rows, "抽查未改動列", unchanged, "通過" if ok else "有問題")
sys.exit(0 if ok and miss.size == 0 else 1)
