# -*- coding: utf-8 -*-
"""v9b：十類（邊界、海岸）改高度上限（預設 6.45 in，面板等比縮小、維持 4×3），其餘 24 張圖與角落選擇沿用 final/ 之結果。
用法：python fp_final2.py <來源根目錄 final> <輸出根目錄> <H_class> <site> [site ...]"""
from __future__ import annotations
import csv, json, shutil, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import fp_final as F  # noqa: E402
L0, L3, FR = F.L0, F.L3, F.FR


def main(src_root: Path, out_root: Path, h_class: float, sites) -> int:
    for site in sites:
        t0 = time.time()
        out = out_root / site
        shutil.copytree(src_root / site, out)
        prev = json.load(open(out / "result.json", encoding="utf-8"))
        corner = prev["corner"]
        S = L0.prep_site(site)
        FR.FONT = S.font.get_file()
        recs = json.load(open(out / "manifest.json", encoding="utf-8"))
        rows = list(csv.reader(open(out / "checks.csv", encoding="utf-8-sig")))
        head, rows = rows[0], [r for r in rows[1:] if not r[1].endswith("_class")]
        fails = [f for f in prev["fails"] if not f.startswith(("boundary_class:", "coast_class:"))]
        var = dict(prev["variants"])
        nr, nc, mode, sh = F.LAYOUT["class"]
        for kind in ("boundary", "coast"):
            name = "class"
            p = out / f"{kind}_{name}.png"
            rec = None
            for pad, comp, ringok in [(0.2, False, False)] + F.LADDER:
                L3.BOX_PAD, L3.COMPACT, L3.RING_OK = pad, comp, ringok
                try:
                    rec = L3.render_fit3(S, kind, name, p, nrow=nr, ncol=nc, H=h_class, mode=mode, corner=corner[kind],
                                         class_brackets=(kind == "boundary"), shared_titles=sh)
                    break
                except RuntimeError as e:
                    print("  retry", f"pad={pad} compact={comp} ring_ok={ringok}", e, flush=True)
            L3.BOX_PAD, L3.COMPACT, L3.RING_OK = 0.2, False, False
            assert rec is not None, (site, kind)
            rec["corner_site"] = corner[kind]
            fl, rw, nb, nl = F.check_fig(site, kind, name, rec)
            rows += [[str(x) for x in r] for r in rw]
            fails += [f"{kind}_{name}: {f}" for f in fl]
            recs[f"{kind}_{name}"] = {k: v for k, v in rec.items() if k != "text_audit"}
            recs[f"{kind}_{name}"]["text_summary"] = rec["text_summary"]
            var[f"{kind}_{name}"] = dict(corner=rec["box_corner"], pad=rec["box_pad"], compact=rec["box_compact"], ring_ok=rec["box_ring_ok"], tier=rec["box_land_tier"],
                                         margin=rec["box_margin_in"], islets=rec["box_islets_covered"], panel_w=round(rec["panel_w_h_in"][0], 3), canvas_h=round(rec["canvas_in"][1], 3))
            print(site, f"{kind}_{name}", [round(v, 3) for v in rec["panel_w_h_in"]], [round(v, 3) for v in rec["canvas_in"]],
                  f"corner={rec['box_corner']} m={rec['box_margin_in']} pad={rec['box_pad']} cmp={rec['box_compact']} coll={rec['collision_count']} fails={len(fl)}", f"{time.time() - t0:.0f}s", flush=True)
        (out / "manifest.json").write_text(json.dumps(recs, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        with (out / "checks.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(head)
            w.writerows(rows)
        nbr = sum(1 for r in rows if r[4] == "括號範圍")
        (out / "result.json").write_text(json.dumps(dict(site=site, corner=corner, variants=var, fails=fails, n_checks=len(rows), n_brackets=nbr, class_H=h_class),
                                                    ensure_ascii=False, indent=1), encoding="utf-8")
        F.contact_sheet(out, site, recs)
        print(site, "DONE fails=", len(fails), fails[:6], f"{time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3]), sys.argv[4:]))
