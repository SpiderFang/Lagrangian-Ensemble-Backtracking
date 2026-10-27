# ruff: noqa: E501
"""WP-75 C：以「邊界分段＋海岸分段」的互斥類別分布重做相近／差異明顯判定（方法同 polar_source：TVD 對 p99 對半切雜訊）。

用法：python scripts/wp75_segprob/run_diff.py <統計表輸出目錄（已存在）>
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import wp75_common as C  # noqa: E402

from lagrangian_backtracking.native_runner import figures as F  # noqa: E402
from lagrangian_backtracking.native_runner import polar_source as PS  # noqa: E402
from lagrangian_backtracking.native_runner.display_names import material_display_name_zh  # noqa: E402

CASE = "finite_depth_stokes"
OLD_SUMMARY = C.RUN / "finite_depth_stokes/figures-polar-v3-ax2-coastB/tables/distinguishability_summary.csv"


def groupset(site, grouping, key, unit, cat, ncat, order_ids, labels):
    code_map = {g: i for i, g in enumerate(order_ids)}
    code = np.fromiter((code_map[k] for k in key.tolist()), dtype=np.int64, count=key.size)
    g_n, u_n = len(order_ids), int(unit.max()) + 1
    flat = (code * u_n + unit) * ncat + cat
    counts = np.bincount(flat, minlength=g_n * u_n * ncat).reshape(g_n, u_n, ncat)
    n_total = counts.sum(axis=(1, 2)).astype(np.int64)
    ns = SimpleNamespace(
        site_id=site,
        grouping=grouping,
        group_ids=tuple(order_ids),
        labels_zh=tuple(labels),
        n_total=n_total,
        counts=counts,
    )
    ns.proportions = lambda: counts.sum(axis=1) / n_total[:, None]
    return ns


def main(tab_dir: Path) -> int:
    regions, sites = C.load_config()
    anchors = F._load_receptor_anchors(C.RELEASE)  # noqa: SLF001
    land = C.load_land()
    params = PS.PolarParams()
    old = {}
    with OLD_SUMMARY.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            old[(r["site"], r["comparison_id"])] = r
    rows, check_rows, pair_rows = [], [], []
    old_pairs = {}
    with (OLD_SUMMARY.parent / "distinguishability_pairs.csv").open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            old_pairs[(r["site"], r["comparison_id"], r["group_a"], r["group_b"])] = r
    anchors2 = {k: v[:2] for k, v in anchors.items()}
    old_samples = PS.load_site_polar_samples(C.RUN / "finite_depth_stokes/postprocess-coastB", anchors2)
    for sid in C.SITES:
        # 舊方法重算（驗證：方法與資料與先前一致）
        an_old = PS.analyze_site(old_samples[sid], CASE, params)
        for c in an_old.comparisons:
            o = old[(sid, c.comparison_id)]
            check_rows.append(
                [
                    sid,
                    c.comparison_id,
                    o["verdict"],
                    c.verdict,
                    "OK"
                    if o["verdict"] == c.verdict and abs(float(o["max_tvd"]) - c.max_tvd) < 1e-5
                    else "MISMATCH",
                ]
            )
        # 新：互斥類別（邊界分段、海岸分段、觸底）
        sd = C.build_site(sid, land, regions, sites, anchors)
        sd.cols = C.load_site_columns(sid)
        C.assign_site(sd)
        nb, nc = len(sd.bseg), len(sd.cseg)
        cat = np.where(sd.b_idx >= 0, sd.b_idx, np.where(sd.c_idx >= 0, nb + sd.c_idx, nb + nc)).astype(
            np.int64
        )
        ncat = nb + nc + 1
        cols = sd.cols
        _, unit = np.unique(cols["arrival_time_id"].astype(str), return_inverse=True)
        unit = unit.astype(np.int64)
        mat = cols["material_id"].astype(str)
        idx = cols["settling_candidate_index"].astype(np.int64)
        # 類別順序同舊版（代表速度由慢到快）
        cls_speed = {
            m: float(np.median(np.abs(cols["settling_velocity_mps"][mat == m])))
            for m in sorted(set(mat.tolist()))
        }
        mats = sorted(cls_speed, key=lambda m: (cls_speed[m], m))
        key_cv = np.char.add(np.char.add(mat, "|"), idx.astype(str))
        cv_ids = [f"{m}|{i}" for m in mats for i in range(10)]
        sets = {
            "class_velocity": groupset(sid, "class_velocity", key_cv, unit, cat, ncat, cv_ids, cv_ids),
            "class": groupset(
                sid, "class", mat, unit, cat, ncat, mats, [material_display_name_zh(m) for m in mats]
            ),
            "season": groupset(
                sid,
                "season",
                cols["deposition_season"].astype(str),
                unit,
                cat,
                ncat,
                ["DJF", "MAM", "JJA", "SON"],
                ["DJF", "MAM", "JJA", "SON"],
            ),
            "tide": groupset(
                sid,
                "tide",
                cols["deposition_tide_class"].astype(str),
                unit,
                cat,
                ncat,
                ["spring_proxy", "neap_proxy"],
                ["大潮", "小潮"],
            ),
        }
        comps = []
        cv = sets["class_velocity"]
        for m in mats:
            members = [i for i, g in enumerate(cv.group_ids) if g.split("|")[0] == m]
            comps.append(
                PS._compare_groups(
                    cv, members, f"velocity_within_class:{m}", "velocity_within_class", m, CASE, params
                )
            )  # noqa: SLF001
        for kind in ("class", "season", "tide"):
            gs = sets[kind]
            comps.append(PS._compare_groups(gs, list(range(len(gs.group_ids))), kind, kind, "", CASE, params))  # noqa: SLF001
        for c in comps:
            for p in c.pairs:
                op = old_pairs[(sid, c.comparison_id, p["group_a"], p["group_b"])]
                old_over = op["over_p99"] == "True"
                pair_rows.append([sid, c.comparison_id, p["group_a"], p["group_b"], p["label_a"], p["label_b"], f"{float(op['tvd']):.4f}", f"{float(op['noise_tvd_p99']):.4f}",
                                  "差異明顯" if old_over else "相近", f"{p['tvd']:.4f}", f"{c.noise_tvd['p99']:.4f}", "差異明顯" if p["over_p99"] else "相近",
                                  "FLIP" if old_over != p["over_p99"] else ""])  # fmt: skip
        for c in comps:
            o = old[(sid, c.comparison_id)]
            rows.append([
                sid, c.comparison_id, c.kind, c.scope_material, material_display_name_zh(c.scope_material) if c.scope_material else "",
                o["verdict"], c.verdict, "FLIP" if o["verdict"] != c.verdict else "",
                o["n_pairs"], f"{float(o['noise_tvd_p99']):.4f}", f"{float(o['max_tvd']):.4f}", o["pairs_over_p99"],
                c.n_pairs, f"{c.noise_tvd['p99']:.4f}", f"{c.max_tvd:.4f}", c.pairs_over_p99, f"{c.mean_tvd:.4f}", f"{c.max_pair[0]} vs {c.max_pair[1]}",
            ])  # fmt: skip
        print(sid, "done", flush=True)
    hdr = [
        "site",
        "comparison_id",
        "kind",
        "material_id",
        "class_zh",
        "old_verdict",
        "new_verdict",
        "flip",
        "old_n_pairs",
        "old_noise_p99",
        "old_max_tvd",
        "old_pairs_over_p99",
        "new_n_pairs",
        "new_noise_p99",
        "new_max_tvd",
        "new_pairs_over_p99",
        "new_mean_tvd",
        "new_max_pair",
    ]
    ph = [
        "site",
        "comparison_id",
        "group_a",
        "group_b",
        "label_a",
        "label_b",
        "old_tvd",
        "old_noise_p99",
        "old_judgement",
        "new_tvd",
        "new_noise_p99",
        "new_judgement",
        "flip",
    ]
    for name, h, r in (
        ("diff_test_pairs_old_vs_new.csv", ph, pair_rows),
        ("diff_test_old_vs_new.csv", hdr, rows),
        (
            "diff_test_old_recompute_check.csv",
            ["site", "comparison_id", "old_csv_verdict", "recomputed_verdict", "status"],
            check_rows,
        ),
    ):
        with (tab_dir / name).open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(h)
            w.writerows(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
