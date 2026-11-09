# -*- coding: utf-8 -*-
"""第二輪：緊湊邊界＋底部橫式色階。用法：python fp_run2.py <輸出目錄（全新）> [site] [only_substring]"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import fp_lib as L0  # noqa: E402
import fp_lib2 as L2  # noqa: E402
import fp_run as FR  # noqa: E402

H = FR.H_FULL
# (檔名, kind, name, nrow, ncol, mode, legend_in_slots, class_brackets, shared_titles, 備註)
SPECS = []
for kind in ("boundary", "coast"):
    SPECS += [
        (f"{kind}_season_2x2_bottom", kind, "season", 2, 2, "bottom", False, False, False, "季節 2×2；底部橫式色階＋圖例（緊湊邊界）"),
        (f"{kind}_season_2x2_rightbar", kind, "season", 2, 2, "right", False, False, False, "季節 2×2；右側色階（僅緊湊邊界，對照用）"),
        (f"{kind}_tide_2x1_rightbar", kind, "tide", 2, 1, "right", False, False, False, "大小潮 2×1；右側色階（緊湊邊界）"),
        (f"{kind}_tide_2x1_bottom", kind, "tide", 2, 1, "bottom", False, False, False, "大小潮 2×1；底部橫式色階"),
        (f"{kind}_tide_1x2_bottom", kind, "tide", 1, 2, "bottom", False, False, False, "大小潮 1×2；底部橫式色階"),
        (f"{kind}_class_4x3_slot", kind, "class", 4, 3, "slot", True, False, True, "十類 4×3；色階與圖例放備用格（緊湊邊界）"),
        (f"{kind}_class_4x3_bottom", kind, "class", 4, 3, "bottom", True, False, True, "十類 4×3；底部橫式色階，圖例放備用格"),
    ]
SPECS.append(("boundary_class_4x3_slot_brackets", "boundary", "class", 4, 3, "slot", True, True, True, "十類邊界 4×3；括號與圈號，備用格（緊湊邊界）"))
SPECS.append(("boundary_class_4x3_bottom_brackets", "boundary", "class", 4, 3, "bottom", True, True, True, "十類邊界 4×3；括號與圈號，底部色階"))


def main(out_root: Path, site: str, only: str = "") -> int:
    out = out_root / site
    out.mkdir(parents=True, exist_ok=False)
    S = L0.prep_site(site)
    FR.FONT = S.font.get_file()
    rows, recs = [], {}
    for fn, kind, name, nr, nc, mode, lis, cb, sh, note in SPECS:
        if only and only not in fn:
            continue
        t0 = time.time()
        p = out / f"{fn}.png"
        rec = L2.render_fit(S, kind, name, p, nrow=nr, ncol=nc, H=H, mode=mode, legend_in_slots=lis, class_brackets=cb, shared_titles=sh)
        mk = FR.mockup(p, out / f"mockup_{fn}.png", rec["canvas_in"][1], note)
        recs[fn] = {k: v for k, v in rec.items() if k not in ("text_audit",)}
        recs[fn]["mockup"] = mk
        recs[fn]["note"] = note
        ta = rec["text_summary"]
        rows.append([fn, note, f"{nr}x{nc}", mode, f"{rec['panel_w_h_in'][0]:.2f}", f"{rec['panel_w_h_in'][1]:.2f}", rec["limit"],
                     f"{rec['canvas_in'][0]:.2f}", f"{rec['canvas_in'][1]:.2f}", f"{mk['fill'] * 100:.1f}", rec["title_max_lines"],
                     f"{rec['gx_in']:.2f}", f"{rec['gy_in']:.2f}", f"{rec['L_in']:.2f}", rec["collision_count"],
                     f"{rec['min_text_sep_pt']:.2f}", ta])
        print(fn, [round(v, 3) for v in rec["panel_w_h_in"]], [round(v, 2) for v in rec["canvas_in"]], f"fill={mk['fill']:.3f}",
              rec["limit"], f"gx={rec['gx_in']} gy={rec['gy_in']} coll={rec['collision_count']} sep={rec['min_text_sep_pt']:.2f}pt",
              f"{time.time() - t0:.0f}s", flush=True)
        for b in rec["collisions"][:6]:
            print("   !", b, flush=True)
    with (out / "summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "layout", "rows_x_cols", "colorbar_mode", "panel_w_in", "panel_h_in", "limited_by", "canvas_w_in",
                    "canvas_h_in", "page_fill_pct", "title_max_lines", "gap_x_in", "gap_y_in", "left_margin_in", "collisions",
                    "min_text_sep_pt", "text_audit"])
        w.writerows(rows)
    (out / "manifest.json").write_text(json.dumps(recs, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "gongliao", sys.argv[3] if len(sys.argv) > 3 else ""))
