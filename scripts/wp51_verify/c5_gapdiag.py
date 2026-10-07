import json, sys
from datetime import datetime, timezone
from collections import Counter
sys.path.insert(0, "/home/mustlab/Workspace/lbt-wp51/verify")
import c_real_sampling as c
root = sys.argv[1]
inv = json.load(open(root + "/common-input/forcing_inventory.json"))
for p in inv["products"]:
    if p["product"] != "ocm_native": continue
    ct = p["canonical_time"]
    gaps = ct["gaps"]
    sizes = Counter(int(g["missing_step_count"]) for g in gaps)
    print(p["analysis_region_id"], "gaps", len(gaps), "missing_total", sum(int(g["missing_step_count"]) for g in gaps), "size_hist", dict(sorted(sizes.items())))
    big = [g for g in gaps if int(g["missing_step_count"]) >= 4]
    for g in big: print("   big:", g["before_utc"], "->", g["after_utc"], g["missing_step_count"])
