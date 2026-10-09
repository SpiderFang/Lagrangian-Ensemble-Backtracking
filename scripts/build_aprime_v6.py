#!/usr/bin/env python3
"""WP-53：v6 母體建置用 ocm_native 根（A' v6，時間軸完整）。

只供 ``horizon-suite-create`` 的結構指紋與到達時刻選取使用，執行期不使用（執行期用 lbt-local-data-v6）。

* 「沒有缺口也沒有重播小時」的月份：硬連結 A' v5 的同名月份（真實陣列為 v5 硬連結、
  其餘為稀疏佔位檔；不複製資料）。
* 受影響月份（含 2024-01～06 的原缺整點，其重建列來自舊 patch 的時間軸；含重播小時或原缺整點；真實值只寫 --real-from 之後的月份，到達時刻只在 2025 年）：時間軸補成完整（原有列＋插入缺的整點，月份規則同
  ``build_surface_v6.py``），``zcor``／``elev``／``wetdry_elem`` 寫入真實值（原有列取自 v5、
  重播列與缺的整點取自新 patch 的重建值；母體建置只在到達時刻讀這三個陣列），
  ``hvel``／``vertical_velocity``／``diffusivity`` 為稀疏佔位檔（母體建置不讀）。
不修改 v5 任何檔案。
"""
from __future__ import annotations

import argparse, json, os, shutil, sys, time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

H = 3600 * 10**9
REAL = ("zcor", "elev", "wetdry_elem")
SHELL = ("hvel", "vertical_velocity", "diffusivity")
FIRST = "202407"


def month_of_hour_end(t: int) -> str:
    return datetime.fromtimestamp((int(t) - 1) / 1e9, tz=UTC).strftime("%Y%m")


def hardlink_tree(a: Path, b: Path) -> None:
    b.mkdir(parents=True, exist_ok=True)
    for p in a.iterdir():
        if p.is_dir():
            hardlink_tree(p, b / p.name)
        else:
            os.link(p, b / p.name)


def build(args, dom: str, log) -> dict:
    v5 = Path(args.v5_native) / dom / "months"
    ap5 = Path(args.aprime_v5) / dom
    patch = Path(args.patch_root) / dom / "months"
    out = Path(args.output_root) / dom
    if out.exists():
        raise SystemExit(f"輸出已存在，拒絕覆寫：{out}")
    bad = {int(v) for v in json.load(open(Path(args.replay_dir) / f"bad_replay_{dom}.json"))["bad"]}
    (out / "months").mkdir(parents=True)
    hardlink_tree(ap5 / "grid", out / "grid")
    months = sorted(p.name for p in (ap5 / "months").iterdir() if p.is_dir())
    axes = {m: np.load(ap5 / "months" / m / "time_utc_ns.npy") for m in months}
    present = {int(t) for a in axes.values() for t in a}
    ploc: dict[int, tuple[str, int]] = {}
    for pm in sorted(p.name for p in patch.iterdir() if p.is_dir()):
        for i, t in enumerate(np.load(patch / pm / "time_utc_ns.npy")):
            ploc[int(t)] = (pm, i)
    inserted: dict[str, list[int]] = {}
    for t in ploc:
        if t not in present and min(int(a[0]) for a in axes.values()) <= t <= max(int(a[-1]) for a in axes.values()):
            inserted.setdefault(month_of_hour_end(t), []).append(t)
    pmm: dict[str, dict[str, np.ndarray]] = {}
    report = {}
    for m in months:
        dest = out / "months" / m
        affected = m in inserted or (m in axes and any(int(t) in bad for t in axes[m]))
        if not affected:
            hardlink_tree(ap5 / "months" / m, dest)
            continue
        t0 = axes[m]
        add = np.asarray(sorted(inserted.get(m, [])), dtype=np.int64)
        tn = np.sort(np.concatenate([t0, add]), kind="stable")
        T = int(tn.size)
        dest.mkdir()
        np.save(dest / "time_utc_ns.npy", tn.astype(np.int64))
        md = json.loads((ap5 / "months" / m / "metadata.json").read_text(encoding="utf-8"))
        for n, ar in md["arrays"].items():
            if "shape" in ar and ar["shape"] and ar["shape"][0] == t0.size:
                ar["shape"] = [T] + list(ar["shape"][1:])
        md["wp53_axis_completion"] = {"inserted_rows": int(add.size), "replaced_replay_rows": int(sum(int(t) in bad for t in t0)),
                                      "note": "時間軸補齊；重播列與缺的整點的 zcor/elev/wetdry_elem 取自重建 patch"}
        (dest / "metadata.json").write_text(json.dumps(md, ensure_ascii=False, indent=2), encoding="utf-8")
        shutil.copy2(ap5 / "months" / m / "quality_report.json", dest / "quality_report.json")
        src_map = {int(t): i for i, t in enumerate(t0)}
        real_names = REAL if m >= args.real_from else ()
        for name in real_names:
            src = np.load(v5 / m / f"{name}.npy", mmap_mode="r")
            dst = np.lib.format.open_memmap(dest / f"{name}.npy", mode="w+", dtype=src.dtype, shape=(T,) + src.shape[1:])
            use_patch = [j for j, t in enumerate(tn) if int(t) in ploc and (int(t) in bad or int(t) not in src_map)]
            use_src = [(j, src_map[int(t)]) for j, t in enumerate(tn) if int(t) in src_map and int(t) not in bad]
            # 以連續區段複製（來源列號與目的列號同時遞增的連續段）
            runs = []
            for j, i in use_src:
                if runs and runs[-1][0] + runs[-1][2] == j and runs[-1][1] + runs[-1][2] == i:
                    runs[-1][2] += 1
                else:
                    runs.append([j, i, 1])
            for j, i, n in runs:
                step = 24
                for s in range(0, n, step):
                    k = min(step, n - s)
                    dst[j + s : j + s + k] = src[i + s : i + s + k]
            for j in use_patch:
                pm, pi = ploc[int(tn[j])]
                key = f"{pm}/{name}"
                if key not in pmm:
                    pmm[key] = np.load(patch / pm / f"{name}.npy", mmap_mode="r")
                dst[j] = pmm[key][pi].astype(src.dtype)
            covered = len(use_patch) + len(use_src)
            assert covered == T, (dom, m, name, covered, T)
            dst.flush()
            del dst
        for name in tuple(SHELL) + tuple(n for n in REAL if n not in real_names):
            ar = md["arrays"][f"{name}.npy"]
            sh = tuple(ar["shape"])
            sh0 = sh
            mm = np.lib.format.open_memmap(dest / f"{name}.npy", mode="w+", dtype=np.dtype(ar["dtype"]), shape=sh0)
            mm.flush()
            del mm
        report[m] = {"rows": T, "inserted": int(add.size), "replay_rows": int(sum(int(t) in bad for t in t0))}
        log(f"[{dom}] 月份 {m}：{T} 列（插入 {add.size}）")
    return {"domain": dom, "months": report}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v5-native", required=True)
    ap.add_argument("--aprime-v5", required=True)
    ap.add_argument("--patch-root", required=True)
    ap.add_argument("--replay-dir", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--real-from", default="202501", help="此月份（含）以後的受影響月份寫入真實 zcor/elev/wetdry；更早者為稀疏佔位檔（到達時刻只在 2025 年，母體建置不讀）")
    ap.add_argument("--report", required=True)
    a = ap.parse_args()

    def log(s):
        print(time.strftime("%F %T"), s, flush=True)

    r = build(a, a.domain, log)
    Path(a.report).write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    log("完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
