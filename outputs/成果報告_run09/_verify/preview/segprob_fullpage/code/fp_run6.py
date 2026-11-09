# -*- coding: utf-8 -*-
"""第五輪：季節／大小潮每列各一條全年樣式直式色階（高度＝該列面板高）。用法：python fp_run5.py <輸出目錄> [site] [only]"""
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

H_FULL = FR.H_FULL
H_75 = 0.75 * FR.TEXT_H - (FR.BEFORE_PT + FR.CAP3_PT) / 72.0  # 圖＋段前＋3 行圖說合計占文字區高 75%
SPECS = []
for kind in ("boundary", "coast"):
    SPECS += [
        (f"{kind}_season_2x2_vrow", kind, "season", 2, 2, H_FULL, "季節 2×2；每列一條直式色階（高＝該列面板高）"),
        (f"{kind}_tide_2x1_vrow", kind, "tide", 2, 1, H_FULL, "大小潮 2×1；每列一條直式色階（高＝該列面板高），整頁"),
        (f"{kind}_tide_2x1_vrow_p75", kind, "tide", 2, 1, H_75, "大小潮 2×1；每列一條直式色階（高＝該列面板高），約 75% 頁高"),
    ]


def main(out_root: Path, site: str, only: str = "") -> int:
    S = L0.prep_site(site)
    FR.FONT = S.font.get_file()
    out = out_root / site
    out.mkdir(parents=True, exist_ok=False)
    rows, recs = [], {}
    for fn, kind, name, nr, nc, H, note in SPECS:
        if only and only not in fn:
            continue
        t0 = time.time()
        p = out / f"{fn}.png"
        rec = L3.render_fit3(S, kind, name, p, nrow=nr, ncol=nc, H=H, mode="vrow", corner="UR")
        mk = FR.mockup(p, out / f"mockup_{fn}.png", rec["canvas_in"][1], note)
        recs[fn] = {k: v for k, v in rec.items() if k not in ("text_audit",)}
        recs[fn]["mockup"] = mk
        recs[fn]["note"] = note
        rows.append([fn, note, f"{nr}x{nc}", f"{rec['panel_w_h_in'][0]:.2f}", f"{rec['panel_w_h_in'][1]:.2f}", rec["limit"],
                     f"{rec['canvas_in'][0]:.2f}", f"{rec['canvas_in'][1]:.2f}", f"{mk['fill'] * 100:.1f}",
                     rec["box_corner"], f"{rec['box_margin_in']:.2f}", rec["collision_count"], f"{rec['min_text_sep_pt']:.2f}",
                     rec["text_summary"]])
        print(fn, [round(v, 3) for v in rec["panel_w_h_in"]], [round(v, 2) for v in rec["canvas_in"]], f"fill={mk['fill']:.3f}",
              rec["limit"], f"corner={rec['box_corner']} m={rec['box_margin_in']} coll={rec['collision_count']} "
              f"sep={rec['min_text_sep_pt']:.2f}pt R={rec['R_in']:.2f}", f"{time.time() - t0:.0f}s", flush=True)
        for b in rec["collisions"][:8]:
            print("   !", b, flush=True)
    with (out / "summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "layout", "rows_x_cols", "panel_w_in", "panel_h_in", "limited_by", "canvas_w_in", "canvas_h_in",
                    "page_fill_pct", "box_corner", "box_margin_in", "collisions", "min_text_sep_pt", "text_audit"])
        w.writerows(rows)
    (out / "manifest.json").write_text(json.dumps(recs, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("done")
    return 0


if __name__ == "__main__":
    a = sys.argv
    raise SystemExit(main(Path(a[1]), a[2] if len(a) > 2 else "gongliao", a[3] if len(a) > 3 else ""))
