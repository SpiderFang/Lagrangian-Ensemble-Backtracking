# ruff: noqa: E501, E402, SIM115
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
sys.path.insert(0, __import__("os").environ.get("LBT_CODE_SRC", "/home/mustlab/Workspace/lbt-repo-wp61/src"))
from pathlib import Path

import arrival_check as ac
import arrival_figs as af

sites = sys.argv[1].split(",") if len(sys.argv) > 1 else list(ac.SITES)
outroot = Path(sys.argv[2] if len(sys.argv) > 2 else "/home/mustlab/Workspace/lbt-wp61/figures")
# run09：貢寮與龜山島各依自己的潮位選時刻，不再以貢寮序列作為龜山島的挑選依據參考（LBT_ARRIVAL_PAIRED=0）。
PAIRED = __import__("os").environ.get("LBT_ARRIVAL_PAIRED", "1") != "0"
fonts = af.Fonts()
print(fonts.record)
recs = ac.load_manifest_records()
S = {s: ac.load_series(s) for s in ac.SITES}
R = {}
for site in ac.SITES:
    ref = S["gongliao"] if (PAIRED and site == "guishan") else None
    R[site] = ac.analyze_site(site, recs[site], S[site], ref)
stats = {
    site: af.site_stats(
        S[site], R[site]["rows"], R[site]["idx_of"], S["gongliao"] if (PAIRED and site == "guishan") else None
    )
    for site in ac.SITES
}
for site in sites:
    d = outroot / site
    d.mkdir(parents=True, exist_ok=True)
    for variant in ("U", "S"):
        plan = af.axis_plan(stats, variant, site)
        print(site, variant, plan["eta"], plan["strength"], plan["hs"], plan["speed"])
        for text in (True, False):
            tag = "text" if text else "notext"
            t = time.time()
            for kind in ("overview", "zoom"):
                out = d / f"wp61_{site}_{kind}_{variant}_{tag}.png"
                for p in (out, out.with_suffix(".svg")):
                    if p.exists():
                        p.unlink()
                if kind == "overview":
                    af.render_overview(site, S[site], R[site], fonts, plan, variant, text, out)
                else:
                    af.render_zoom(
                        site,
                        S[site],
                        R[site],
                        fonts,
                        plan,
                        stats[site],
                        variant,
                        text,
                        out,
                        ref=S["gongliao"] if (PAIRED and site == "guishan") else None,
                    )
            print("  ", variant, tag, round(time.time() - t, 1), "s")
