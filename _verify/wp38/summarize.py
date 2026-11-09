"""WP-38：把兩案例的 footprint_summary_by_site.csv 整理成 markdown 表（貼進報告用）。"""

import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])  # run04 根目錄
cases = [("no_stokes", "no_stokes"), ("finite_depth_stokes", "finite_depth_stokes")]
SITES = ["gongliao", "guishan", "hsinchu", "lienchiang", "nanwan"]


def rows(case):
    with (root / case / "postprocess.wp38" / "footprint_summary_by_site.csv").open(encoding="utf-8") as f:
        return {r["study_site_id"]: r for r in csv.DictReader(f)}


def f(v, nd=0):
    if v in ("", None):
        return "—"
    return f"{float(v):,.{nd}f}"


def pct(v):
    return "—" if v in ("", None) else f"{float(v) * 100:.1f}%"


tables = {c: rows(c) for c, _ in cases}
for case, _ in cases:
    t = tables[case]
    print(f"\n### {case}\n")
    print("| 站 | 粒子小時 | HDR50 (km²) | HDR75 | HDR90 | 質心距離 (km) | 方位 (°) | 水深<200 m | 缺水深 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for s in SITES:
        r = t[s]
        print(
            f"| {s} | {f(r['total_particle_hours'])} | {f(r['hdr50_area_km2'])} | {f(r['hdr75_area_km2'])} | "
            f"{f(r['hdr90_area_km2'])} | {f(r['centroid_distance_from_receptor_km'], 1)} | "
            f"{f(r['centroid_bearing_from_receptor_deg'])} | {pct(r['hours_fraction_depth_below_threshold'])} | "
            f"{pct(r['hours_fraction_depth_missing'])} |"
        )
    print("\n仍在漂流比例（快照粒子數）：\n")
    print("| 站 | 72 h | 168 h | 336 h | 720 h |")
    print("|---|---|---|---|---|")
    for s in SITES:
        r = t[s]
        cells = [f"{pct(r[f'active_fraction_{h}h'])} ({int(float(r[f'active_particle_count_{h}h'])):,})" for h in (72, 168, 336, 720)]
        print(f"| {s} | " + " | ".join(cells) + " |")
    print("\n收斂：成員分半 TVD 對雜訊底（成員隨機分半平均／p95；到達單元分半平均／p95）\n")
    print("| 站 | 成員分半 TVD | 成員分半 HDR50 面積相對差 | 雜訊底(成員) 平均／p95 | 雜訊底(單元) 平均／p95 |")
    print("|---|---|---|---|---|")
    for s in SITES:
        r = t[s]
        print(
            f"| {s} | {f(r['member_half_tvd'], 4)} | {pct(r['member_half_hdr50_area_relative_difference'])} | "
            f"{f(r['noise_tvd_member_partition_mean'], 4)}／{f(r['noise_tvd_member_partition_p95'], 4)} | "
            f"{f(r['noise_tvd_unit_split_mean'], 4)}／{f(r['noise_tvd_unit_split_p95'], 4)} |"
        )
    print("\n網格敏感度（HDR50／HDR90 面積 km²，0.01°／0.02°／0.04°）：\n")
    print("| 站 | 0.01° | 0.02° | 0.04° |")
    print("|---|---|---|---|")
    for s in SITES:
        r = t[s]
        cells = [f"{f(r[f'grid_sensitivity_hdr50_area_km2_{k}'])}／{f(r[f'grid_sensitivity_hdr90_area_km2_{k}'])}" for k in ("0p01", "0p02", "0p04")]
        print(f"| {s} | " + " | ".join(cells) + " |")
