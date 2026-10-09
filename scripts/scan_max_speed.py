#!/usr/bin/env python3
"""WP-53：逐小時掃描 hvel 的最大流速大小（m/s），供「重建最大流速不超過真實域最大值」檢查。

輸出 JSON：{domain: {month: {"time": [...ns], "max_speed": [...]}}}。以月份為單位多程序平行。
可掃描 ocm_native 月份目錄或 patch 月份目錄（兩者的 hvel.npy/time_utc_ns.npy 版面相同）。
"""
import argparse, json, os, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np


def scan(args):
    d = Path(args)
    t = np.load(d / "time_utc_ns.npy")
    hv = np.load(d / "hvel.npy", mmap_mode="r")
    out = np.empty(t.size, dtype=np.float64)
    for i in range(t.size):
        a = np.asarray(hv[i])
        out[i] = float(np.nanmax(np.hypot(a[..., 0], a[..., 1])))
    return str(d), t.tolist(), out.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="含 <domain>/months/<YYYYMM> 的根")
    ap.add_argument("--first-month", default="000000")
    ap.add_argument("--output", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    root = Path(a.root)
    jobs = []
    for dom in sorted(p.name for p in root.iterdir() if (p / "months").is_dir()):
        for m in sorted(p.name for p in (root / dom / "months").iterdir() if p.is_dir() and p.name >= a.first_month and (p / "hvel.npy").is_file()):
            jobs.append(str(root / dom / "months" / m))
    res = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for path, t, s in ex.map(scan, jobs):
            p = Path(path)
            res.setdefault(p.parent.parent.name, {})[p.name] = {"time": t, "max_speed": s}
            print("done", p.parent.parent.name, p.name, flush=True)
    Path(a.output).write_text(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
