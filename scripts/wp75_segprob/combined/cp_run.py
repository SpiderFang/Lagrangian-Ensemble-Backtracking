# -*- coding: utf-8 -*-
"""合併圖執行：python cp_run.py [--legacy] <輸出根目錄> <site> [A|B] [figs...]   figs：all season tide class（預設全部）

預設（v37）：多面板用括號旁／引線旁 xx.xx%（無圈號、無數值框），用語「域外來源」；``--legacy`` 重現 v36（圈號＋數值框，用語「邊界穿越」）。
"""
from __future__ import annotations
import csv, json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import cp_lib as C  # noqa: E402
import numpy as np  # noqa: E402
from shapely.geometry import LineString  # noqa: E402
import fp_run as FR  # noqa: E402
import wp75_make as M  # noqa: E402

L0, L4, SF, SA, sp = C.L0, C.L4, C.SF, C.SA, C.sp
H_FULL = FR.H_FULL
LAYOUT = {"season": (2, 2, "vrow", False, H_FULL), "tide": (1, 2, "vrow", False, H_FULL), "class": (4, 3, "slotLR", True, 6.5)}
LADDER = [(0.2, "list", False, False), (0.12, "list", False, False), (0.06, "list", False, False), (0.04, "list", False, False),
          (0.2, "list", False, True), (0.12, "list", False, True), (0.06, "list", False, True), (0.2, "list", True, True)]
INSET_RIGHT = {"lienchiang": 0.915}
INSET_WIN = {"lienchiang": (119.906, 120.016, 26.13, 26.25)}


def touch_report(S, name, w_in, h_in, key):
    """上色（>0）海岸分段與上色（>0）邊界分段之最近距離（英吋，依該圖面板尺寸）；< 0.03 in 者列出。"""
    sd, dr = S.sd, S.dr
    win = SF.WINDOWS[sd.site_id]
    sx, sy = w_in / (win[1] - win[0]), h_in / (win[3] - win[2])
    bc, cc, n = S.bcount[key], S.ccount[key], S.gn[key]
    tr = lambda a: np.column_stack([(np.asarray(a)[:, 0] - win[0]) * sx, (np.asarray(a)[:, 1] - win[2]) * sy])
    bl = [(i, LineString(tr(a))) for i, parts in enumerate(dr.b_sea) if bc[i] > 0 for a in parts if len(a) >= 2]
    out = []
    bcls, ccls = sp.bin_index(bc / n * 100.0, SF.EDGES), sp.bin_index(cc / n * 100.0, SF.EDGES)
    for j, parts in enumerate(dr.c_lines):
        if cc[j] <= 0:
            continue
        for a in parts:
            if len(a) < 2:
                continue
            g = LineString(tr(a))
            for i, bg in bl:
                d = g.distance(bg)
                if d < 0.03:
                    out.append((sd.cseg[j].seg_id, int(ccls[j]), sd.bseg[i].seg_id, int(bcls[i]), round(d, 4)))
    uniq = sorted(set((c, cb, b, bb) for c, cb, b, bb, d in out))
    return uniq


def fit_panels(S, name, path):
    nobox = name.endswith("_nobox")
    C.BONLY = name.endswith("_bonly")
    name = name.replace("_nobox", "").replace("_bonly", "").replace("_bonly", "")
    C.NOBOX = nobox
    SA.STRETCH_BOUNDARY_MAX_NUMBERED = 0 if nobox else 3
    L4.BOX_FC, L4.BOX_EC = ("none", "none") if nobox else ((1.0, 1.0, 1.0, 0.85), "#9e9e9e")
    nr, nc, mode, sh, H = LAYOUT[name]
    SA.MARKER_FS = 7.0 if name == "class" else 8.0
    last = None
    import os
    for pad, bm, land, pp in LADDER[int(os.environ.get("LADDER_START", 0)): int(os.environ.get("LADDER_N", 8))]:
        L4.BOX_PAD, C.BOXMODE, C.ALLOW_LAND = pad, bm, land
        C.set_per_panel(pp)
        try:
            rec = L4.render_fit3(S, "combined", name, path, nrow=nr, ncol=nc, H=H, mode=mode, corner=None,
                                 class_brackets=(name == "class"), shared_titles=sh)
            rec["ladder"] = dict(pad=pad, boxmode=bm, allow_land=land, per_panel=pp)
            L4.BOX_PAD, C.BOXMODE, C.ALLOW_LAND = 0.2, "list", False
            return rec
        except RuntimeError as e:
            last = e
            print("  retry", pad, bm, land, pp, e, flush=True)
            if Path(path).exists():
                Path(path).unlink()
    L4.BOX_PAD, C.BOXMODE, C.ALLOW_LAND = 0.2, "list", False
    raise RuntimeError(f"all ladder failed: {last}")


def collisions_of(rec):
    n = rec.get("collision_count", 0)
    ann = rec.get("annotations")
    items = []
    if isinstance(ann, dict):
        for k in ("boundary", "coast", "coast_inset"):
            a = ann.get(k)
            if a:
                items.append(a)
    elif isinstance(ann, list):
        items = ann
    viol = []
    for a in items:
        for c in a.get("collisions", []) or []:
            viol.append(c)
        for t in a.get("stretches", []):
            if t.get("violations"):
                viol.append(t["violations"])
    return n, viol


def main(out_root, site, figs, arr="A"):
    if site == "lienchiang" and os.environ.get("ALIAS") and not C.LABELMODE:
        figs = [{"tide": "tide_bonly", "class": "class_nobox"}.get(f, f) for f in figs]
    C.ARR = arr
    L4.DEBUG = bool(os.environ.get("CP_DEBUG"))  # 印出版面收斂迭代
    out = Path(out_root) / site
    out.mkdir(parents=True, exist_ok=True)
    S = L0.prep_site(site)
    FR.FONT = S.font.get_file()
    recs, rows = {}, []
    for name in figs:
        t0 = time.time()
        p = out / f"combined_{name}_{arr}.png" if name != "all" else out / "combined_all.png"
        if name == "all":
            iw = INSET_WIN.get(site) if site == "lienchiang" else None
            rec = C.render_single_c(S, p, inset_window=iw, inset_min_mag=4.0, inset_right_max=INSET_RIGHT.get(site, 0.995))
            fh = rec["canvas_in"][1]
            n, viol = collisions_of(rec)
            n_touch = touch_report(S, name, 4.082, 2.98, "all")
            rec["touch_pairs"] = n_touch
            rec["collision_count"] = n
            rec["violations"] = viol
        else:
            rec = fit_panels(S, name, p)
            fh = rec["canvas_in"][1]
            n, viol = collisions_of(rec)
            keys = L4._keys(S, name.replace("_nobox", "").replace("_bonly", ""))
            w_, h_ = rec["panel_w_h_in"]
            rec["touch_pairs"] = {k: touch_report(S, name, w_, h_, k) for k in keys}
            rec["violations"] = viol
        mk = FR.mockup(p, out / f"mockup_{name}_{arr}.png", fh, f"{site} combined_{name}")
        rec["mockup"] = mk
        recs[name] = {k: v for k, v in rec.items() if k != "text_audit"}
        tp = rec["touch_pairs"]
        ntouch = len(tp) if isinstance(tp, list) else sum(len(v) for v in tp.values())
        rows.append([site, name, f"{rec['canvas_in'][0]:.2f}x{fh:.2f}", n, len(viol), ntouch, f"{mk['fill']*100:.1f}",
                     json.dumps(rec.get("ladder", {}), ensure_ascii=False), json.dumps(rec.get("box_lines", ""), ensure_ascii=False)[:300]])
        print(site, name, rec["canvas_in"], "coll", n, "viol", len(viol), "touch", ntouch, f"{time.time()-t0:.0f}s", flush=True)
    (out / f"manifest_combined_{arr}_{'_'.join(figs)}.json").write_text(json.dumps(recs, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    with (out / f"summary_combined_{arr}_{'_'.join(figs)}.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["site", "fig", "canvas_in", "collisions", "violations", "touch_pairs", "page_fill_pct", "ladder", "box_lines"])
        w.writerows(rows)


if __name__ == "__main__":
    arr = "A"
    if "--legacy" in sys.argv:
        sys.argv.remove("--legacy")
        C.LABELMODE = False
    args = [a for a in sys.argv[3:]]
    if args and args[0] in ("A", "B"):
        arr, args = args[0], args[1:]
    figs = args or ["all", "season", "tide", "class"]
    raise SystemExit(main(sys.argv[1], sys.argv[2], figs, arr))
