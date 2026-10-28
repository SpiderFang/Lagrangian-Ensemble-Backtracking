# ruff: noqa: E501
"""開發用：只畫指定地點與指定圖。用法：dev_v2.py <輸出目錄> <site> <kind:name,kind:name,...>"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import run_v1 as R
import wp75_common as C
import wp75_make as M

from lagrangian_backtracking.native_runner import figures as F
from lagrangian_backtracking.native_runner import segprob as sp
from lagrangian_backtracking.native_runner import segprob_figures as SF

out = Path(sys.argv[1])
sid = sys.argv[2]
only = {tuple(x.split(":")) for x in sys.argv[3].split(",")} if len(sys.argv) > 3 else None
regions, sites = C.load_config()
anchors = F._load_receptor_anchors(C.RELEASE)
land = C.load_land()
font = F._resolve_font(None).font_properties
sd = C.build_site(sid, land, regions, sites, anchors)
sd.cols = C.load_site_columns(sid)
C.assign_site(sd)
grp, label, mats = R.build_groups(sd.cols)
gn = {k: int(m.sum()) for k, m in grp.items()}
nb, nc = len(sd.bseg), len(sd.cseg)
bcount = {k: sp.counts_by_segment(sd.b_idx[m & (sd.b_idx >= 0)], nb) for k, m in grp.items()}
ccount = {k: sp.counts_by_segment(sd.c_idx[m & (sd.c_idx >= 0)], nc) for k, m in grp.items()}
dr = SF.prepare_drawing(sd, land)
recs, rows, _tl, _ex = M.render_site(sd, dr, grp, label, mats, bcount, ccount, gn, out / sid, font, only=only)
for k, v in recs.items():
    a = v.get("annotations")
    col = []

    def collect(x):
        if isinstance(x, dict):
            if x.get("collisions"):
                col.extend(x["collisions"])  # noqa: B023
            for y in x.values():
                collect(y)
        elif isinstance(x, list):
            for y in x:
                collect(y)

    collect(v)
    print(
        k,
        "collisions:",
        col,
        "| title_overlaps:",
        v.get("title_overlaps"),
        "variant:",
        v.get("title_variant_index"),
        "xgap:",
        v.get("xtick_min_gap_pt"),
        "inset:",
        (v.get("inset") or {}).get("magnification"),
        flush=True,
    )
json.dump(recs, open(out / f"{sid}_recs.json", "w"), ensure_ascii=False, indent=1, default=str)  # noqa: SIM115
