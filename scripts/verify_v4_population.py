#!/usr/bin/env python3
"""WP-51：v4（per_member）母體建好後的核對（唯讀）。

核對項目：十個沉降速度與 v3 相同；每站 500 個沉底時刻、Spearman（年齡 vs 觀測時刻）、
沉底月份涵蓋（特別是 10–12 月）、拒絕／交換次數；沉底整點未落在「受體不可支撐」排除集合；
龜山島錨點與 5 個受體在海上且在核心內、其他站受體與 v3 相同；標籤分組（unknown 比例、各季
是否為 0）；每案粒子數。輸出人讀的文字到 stdout，另寫 JSON。

用法：verify_v4_population.py <v4 母體目錄> <v3 母體目錄> <B 輸出目錄> <JSON 輸出> [coastline.geojson]
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

V4 = Path(sys.argv[1])
V3 = Path(sys.argv[2])
B = Path(sys.argv[3])
OUT = Path(sys.argv[4])
COAST = Path(sys.argv[5]) if len(sys.argv) > 5 else None
report: dict = {}


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def utc(ns: int) -> datetime:
    return datetime.fromtimestamp(ns / 1e9, tz=UTC)


def spearman(a, b) -> float:
    ra = np.argsort(np.argsort(a, kind="stable"), kind="stable").astype(float)
    rb = np.argsort(np.argsort(b, kind="stable"), kind="stable").astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra * rb).sum() / math.sqrt((ra**2).sum() * (rb**2).sum()))


def hav_m(lon1, lat1, lon2, lat2):
    r = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# 1. 沉降速度
m4, m3 = load(V4 / "common-input" / "material.json"), load(V3 / "common-input" / "material.json")
same_records = m4["records"] == m3["records"]
vel = {r.get("material_id", r.get("id", i)): r.get("settling_velocity_mps", r.get("velocity_mps")) for i, r in enumerate(m4["records"])}
report["settling"] = {"records_equal_v3": same_records, "count": len(m4["records"]), "velocities": vel}
print(f"[沉降速度] 十檔 records 與 v3 完全相同：{same_records}（{len(m4['records'])} 檔）")
for k, v in vel.items():
    print(f"    {k}: {v}")

# 2. arrival
a4 = load(V4 / "common-input" / "arrival.json")
prov = a4["provenance"]
bed = prov["bed_residence_sampling"]
by_site = defaultdict(list)
for rec in a4["records"]:
    by_site[rec["study_site_id"]].append(rec)
support = bed.get("deposition_support_conditioning")
excluded = {s: set(v["excluded_hours_utc_ns"]) for s, v in (support or {}).get("by_site", {}).items()}
report["arrival"] = {}
print(f"\n[沉底時刻] bed_residence_sampling keys：{sorted(bed)}")
print(f"    method_id={a4['provenance'].get('method_id')} support policy={(support or {}).get('policy_id')}")
all_months = Counter()
for sid in sorted(by_site):
    recs = by_site[sid]
    dep = np.array([int(r["time_utc_ns"]) for r in recs])
    obs = np.array([int(r["metadata"]["observation_time_utc_ns"]) for r in recs])
    age = np.array([int(r["metadata"]["bed_residence_age_hours"]) for r in recs])
    months = Counter(f"{utc(int(t)).year}-{utc(int(t)).month:02d}" for t in dep)
    all_months.update(months)
    q4 = sum(c for m, c in months.items() if int(m[-2:]) >= 10 and m.startswith("2025"))
    rho = spearman(age, obs)
    bad = int(sum(1 for t in dep if int(t) in excluded.get(sid, set())))
    rej = [r["metadata"]["bed_residence_availability_rejection_audit"] for r in recs]
    rej_total = sum(int(x.get("rejection_count", len(x.get("rejected_age_hours", [])))) for x in rej)
    report["arrival"][sid] = {
        "records": len(recs), "unique_deposition_times": int(len(set(dep.tolist()))),
        "age_min_h": int(age.min()), "age_max_h": int(age.max()), "spearman_age_vs_observation": rho,
        "months": dict(sorted(months.items())), "deposition_oct_to_dec_2025": int(q4),
        "deposition_in_excluded_set": bad, "rejections_total": int(rej_total),
        "deposition_min": utc(int(dep.min())).isoformat(), "deposition_max": utc(int(dep.max())).isoformat(),
    }
    print(f"  {sid}: 筆數 {len(recs)}，相異沉底時刻 {len(set(dep.tolist()))}，年齡 {age.min()}–{age.max()} h，"
          f"Spearman {rho:+.3f}，10–12月沉底 {q4}，落在排除集合 {bad}，逐槽位拒絕總數 {rej_total}")
    print(f"      月份分布：{dict(sorted(months.items()))}")
for key in sorted(bed):
    if key in ("deposition_support_conditioning",):
        continue
    val = bed[key]
    text = json.dumps(val, ensure_ascii=False)
    print(f"    bed_residence_sampling.{key} = {text[:300]}{'…' if len(text) > 300 else ''}")
if support:
    print("    deposition_support_conditioning：")
    for sid, v in support["by_site"].items():
        print(f"      {sid}: canonical {v['canonical_available_hour_count']}，可支撐 {v['supported_available_hour_count']}，"
              f"排除 {len(v['excluded_hours_utc_ns'])}")
    report["support"] = {s: {k: v[k] for k in ("canonical_available_hour_count", "supported_available_hour_count")}
                         | {"excluded": len(v["excluded_hours_utc_ns"])} for s, v in support["by_site"].items()}

# 3. 受體
r4, r3 = load(V4 / "common-input" / "receptor.json"), load(V3 / "common-input" / "receptor.json")


def per_site(doc):
    d = defaultdict(list)
    for r in doc["records"]:
        d[r["study_site_id"]].append((r["lon"], r["lat"], r["vertical_id"], r["z_m_positive_up"]))
    return d


p4, p3 = per_site(r4), per_site(r3)
report["receptors"] = {}
print("\n[受體]")
for sid in sorted(p4):
    eq = sorted(p4[sid]) == sorted(p3[sid])
    horiz = sorted({(round(x[0], 9), round(x[1], 9)) for x in p4[sid]})
    report["receptors"][sid] = {"count": len(p4[sid]), "horizontal_faces": len(horiz), "equal_v3": eq}
    print(f"  {sid}: 受體 {len(p4[sid])}，水平位置 {len(horiz)}，與 v3 相同：{eq}")
rp = r4["provenance"]
for key in ("random_horizontal_receptors_by_site",):
    for sid, blk in rp.get(key, {}).items():
        print(f"    {sid} selection_scope={blk.get('selection_scope')} face_support_failure_summary="
              f"{json.dumps(blk.get('face_support_failure_summary'), ensure_ascii=False)[:400]}")

# 龜山島：在海上、在核心內
try:
    import yaml

    cfg = yaml.safe_load((V4 / "common-config.yaml").read_text(encoding="utf-8"))
    site_cfg = {s["study_site_id"]: s for s in cfg["study_sites"]}
except Exception as exc:  # noqa: BLE001
    site_cfg = {}
    print("  讀 common-config.yaml 失敗：", exc)
gs = site_cfg.get("guishan")
if gs:
    alon, alat = gs["anchor_lonlat"]
    radius = float(gs["receptor_core_radius_m"])
    horiz = sorted({(x[0], x[1]) for x in p4["guishan"]})
    dists = [hav_m(alon, alat, lo, la) for lo, la in horiz]
    land = None
    if COAST is not None:
        from shapely.geometry import Point, shape
        from shapely.strtree import STRtree

        feats = load(COAST)["features"]
        polys = [shape(f["geometry"]) for f in feats]
        tree = STRtree(polys)
        deg_m = 111_000.0

        def land_info(lon, lat):
            pt = Point(lon, lat)
            inside = any(polys[i].contains(pt) for i in tree.query(pt))
            near = tree.query(pt.buffer(0.05))
            dmin = min((polys[i].exterior.distance(pt) for i in near), default=float("inf")) * deg_m * math.cos(math.radians(lat))
            return inside, dmin

        land = {"anchor": land_info(alon, alat), "receptors": [land_info(lo, la) for lo, la in horiz]}
    report["guishan"] = {"anchor": [alon, alat], "core_radius_m": radius, "receptor_dist_to_anchor_m": dists,
                         "land": land}
    print(f"  龜山島 anchor=({alon:.6f},{alat:.6f}) 核心半徑 {radius:.0f} m；5 個受體離錨點 {[round(d) for d in dists]} m，"
          f"全在核心內：{all(d <= radius for d in dists)}")
    if land:
        print(f"    錨點：在陸地內={land['anchor'][0]}，近似離岸 {land['anchor'][1]:.0f} m；"
              f"受體在陸地內={[x[0] for x in land['receptors']]}，近似離岸 m={[round(x[1]) for x in land['receptors']]}")

# 4. 身分表與標籤
try:
    from lagrangian_backtracking.native_runner.particle_input import load_particle_identity_table_npz
    from lagrangian_backtracking.native_runner.postprocess import _label_groupings

    table = load_particle_identity_table_npz(B / "inputs" / "identity_table.npz")
    n = len(table.study_site_id)
    print(f"\n[身分表] 粒子列數 {n}")
    report["identity_rows"] = int(n)
    labels = table.labels
    if labels is None:
        print("  身分表沒有標籤欄")
    else:
        summary = {}
        for name, values in _label_groupings(labels):
            c = Counter(str(v) for v in np.asarray(values).tolist())
            summary[name] = {k: {"count": v, "fraction": v / n} for k, v in sorted(c.items())}
            print(f"  {name}: " + "，".join(f"{k}={v}（{v / n:.1%}）" for k, v in sorted(c.items())))
        report["label_group_summary"] = summary
        unk = {name: sum(v["fraction"] for k, v in grp.items() if "unknown" in k.lower()) for name, grp in summary.items()}
        print("  unknown 比例：", {k: f"{v:.2%}" for k, v in unk.items()})
        season = summary.get("deposition_season", {})
        print("  沉底季節：", {k: v["count"] for k, v in season.items()}, "—— 任一季為 0：",
              any(season.get(s, {"count": 0})["count"] == 0 for s in ("DJF", "MAM", "JJA", "SON")))
except Exception as exc:  # noqa: BLE001
    print("  身分表／標籤核對失敗：", type(exc).__name__, exc)
    report["identity_error"] = f"{type(exc).__name__}: {exc}"

OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
print(f"\n已寫入 {OUT}")
