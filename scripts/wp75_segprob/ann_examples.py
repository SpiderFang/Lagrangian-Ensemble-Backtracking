# ruff: noqa: E501
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import run_v1 as R
import wp75_common as C
import wp75_make as M

from lagrangian_backtracking.native_runner import figures as F
from lagrangian_backtracking.native_runner import segprob as sp
from lagrangian_backtracking.native_runner import segprob_figures as SF

sid = sys.argv[2]
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
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
for tag, mode in (
    (
        "A",
        {
            "boundary": {"season": "annot", "tide": "annot", "class": "title"},
            "coast": {"season": "annot", "tide": "annot", "class": "none"},
        },
    ),
    (
        "B",
        {
            "boundary": {"season": "title", "tide": "title", "class": "title"},
            "coast": {"season": "none", "tide": "none", "class": "none"},
        },
    ),
):
    recs, rows = M.render_site(sd, dr, grp, label, mats, bcount, ccount, gn, out / tag, font, panel_mode=mode)
    print(
        tag,
        {k: (v.get("annotations") if "annotations" in v else None) and "ann" for k, v in recs.items()},
        flush=True,
    )
    import json

    json.dump(
        {k: v.get("annotations") for k, v in recs.items()},
        open(out / f"ann_{tag}.json", "w"),  # noqa: SIM115
        ensure_ascii=False,
        indent=1,
        default=str,
    )
