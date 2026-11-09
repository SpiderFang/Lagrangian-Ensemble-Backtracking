# -*- coding: utf-8 -*-
"""第三輪：單行標題＋圖內數值框＋傳統共用色階。用法：python fp_run4.py <輸出目錄（全新）> [site] [only] [corner_only]"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import fp_lib as L0  # noqa: E402
import fp_lib3 as L3  # noqa: E402
import fp_run as FR  # noqa: E402

H = FR.H_FULL
# (檔名, kind, name, nrow, ncol, mode, class_brackets, shared_titles, 備註)
SPECS = [
    ("boundary_class_4x3_slotLR_brackets", "boundary", "class", 4, 3, "slotLR", True, True, "十類邊界 4×3；括號與圈號；色階左備用格、圖例右備用格"),
    ("coast_class_4x3_slotLR", "coast", "class", 4, 3, "slotLR", False, True, "十類海岸 4×3；色階左備用格、圖例右備用格"),
]
for kind in ("boundary", "coast"):
    SPECS += [
        (f"{kind}_season_2x2_hbar", kind, "season", 2, 2, "hbar", False, False, "季節 2×2；共用橫式色階置中於格網下方"),
        (f"{kind}_season_2x2_vbar", kind, "season", 2, 2, "vbar", False, False, "季節 2×2；共用直式色階（一個面板高）在右側"),
        (f"{kind}_tide_2x1_hbar", kind, "tide", 2, 1, "hbar", False, False, "大小潮 2×1；共用橫式色階置中於格網下方"),
        (f"{kind}_tide_2x1_vbar", kind, "tide", 2, 1, "vbar", False, False, "大小潮 2×1；共用直式色階（一個面板高）在右側"),
    ]


def main(out_root: Path, site: str, only: str = "", corner_only: bool = False) -> int:
    S = L0.prep_site(site)
    FR.FONT = S.font.get_file()
    corners = {}
    for kind, cb in (("boundary", True), ("coast", False)):
        pl, sz = L3.choose_corner(S, kind, cb)
        corners[kind] = pl["corner"] if pl else "UR"  # 最嚴格尺度（聯集）無解時，各圖各自就近放，統一用右上角
        print(f"corner[{kind}] = {pl} box_max_in={sz}", flush=True)
    if corner_only:
        return 0
    out = out_root / site
    out.mkdir(parents=True, exist_ok=False)
    rows, recs = [], {}
    for fn, kind, name, nr, nc, mode, cb, sh, note in SPECS:
        if only and only not in fn:
            continue
        t0 = time.time()
        p = out / f"{fn}.png"
        rec = L3.render_fit3(S, kind, name, p, nrow=nr, ncol=nc, H=H, mode=mode, corner=corners[kind], class_brackets=cb, shared_titles=sh)
        mk = FR.mockup(p, out / f"mockup_{fn}.png", rec["canvas_in"][1], note)
        recs[fn] = {k: v for k, v in rec.items() if k not in ("text_audit",)}
        recs[fn]["mockup"] = mk
        recs[fn]["note"] = note
        bf = rec["bar_frac_used"]
        rows.append([fn, note, f"{nr}x{nc}", mode, f"{rec['panel_w_h_in'][0]:.2f}", f"{rec['panel_w_h_in'][1]:.2f}", rec["limit"],
                     f"{rec['canvas_in'][0]:.2f}", f"{rec['canvas_in'][1]:.2f}", f"{mk['fill'] * 100:.1f}",
                     rec["box_corner"], f"{rec['box_margin_in']:.2f}", "" if bf is None else f"{bf:.2f}",
                     rec["collision_count"], f"{rec['min_text_sep_pt']:.2f}", rec["text_summary"]])
        print(fn, [round(v, 3) for v in rec["panel_w_h_in"]], [round(v, 2) for v in rec["canvas_in"]], f"fill={mk['fill']:.3f}",
              rec["limit"], f"corner={rec['box_corner']} m={rec['box_margin_in']} coll={rec['collision_count']} "
              f"sep={rec['min_text_sep_pt']:.2f}pt bar={bf}", f"{time.time() - t0:.0f}s", flush=True)
        for b in rec["collisions"][:8]:
            print("   !", b, flush=True)
    with (out / "summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "layout", "rows_x_cols", "colorbar_mode", "panel_w_in", "panel_h_in", "limited_by", "canvas_w_in",
                    "canvas_h_in", "page_fill_pct", "box_corner", "box_margin_in", "hbar_width_frac", "collisions",
                    "min_text_sep_pt", "text_audit"])
        w.writerows(rows)
    (out / "manifest.json").write_text(json.dumps(recs, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("done")
    return 0


if __name__ == "__main__":
    a = sys.argv
    raise SystemExit(main(Path(a[1]), a[2] if len(a) > 2 else "gongliao", a[3] if len(a) > 3 else "", len(a) > 4 and a[4] == "corner_only"))
