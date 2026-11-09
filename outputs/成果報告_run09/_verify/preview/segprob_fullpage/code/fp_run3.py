# -*- coding: utf-8 -*-
"""第三輪：以較寬鬆之面板間距（gx 0.10、gy 0.08 英吋）重出季節與十類，供比較。用法：python fp_run3.py <輸出目錄> [site]"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import fp_lib as L0
import fp_lib2 as L2
import fp_run as FR
import fp_run2 as R2

KEEP = ("season_2x2_bottom", "class_4x3_slot", "class_4x3_slot_brackets")
out = Path(sys.argv[1]) / (sys.argv[2] if len(sys.argv) > 2 else "gongliao")
out.mkdir(parents=True, exist_ok=False)
S = L0.prep_site(out.name)
FR.FONT = S.font.get_file()
for fn, kind, name, nr, nc, mode, lis, cb, sh, note in R2.SPECS:
    if not any(fn.endswith(k) for k in KEEP):
        continue
    fn2 = fn + "_gap10"
    p = out / f"{fn2}.png"
    rec = L2.render_fit(S, kind, name, p, nrow=nr, ncol=nc, H=R2.H, mode=mode, legend_in_slots=lis, class_brackets=cb,
                        shared_titles=sh, gaps=((0.10, 0.08),))
    mk = FR.mockup(p, out / f"mockup_{fn2}.png", rec["canvas_in"][1], note + "（間距 0.10 in）")
    print(fn2, [round(v, 3) for v in rec["panel_w_h_in"]], [round(v, 2) for v in rec["canvas_in"]], f"fill={mk['fill']:.3f}",
          rec["limit"], f"coll={rec['collision_count']} sep={rec['min_text_sep_pt']:.2f}pt", flush=True)
print("done")
