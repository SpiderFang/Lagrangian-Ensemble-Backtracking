# ruff: noqa: E501
"""WP-75 試作：幾何期望表、新舊主要來源方向規則比較表。

用法：python run_pe_tables.py <表輸出目錄（全新）>
"""

from __future__ import annotations

import csv
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pe_data as D  # noqa: E402

from lagrangian_backtracking.native_runner import polar_expected as PE  # noqa: E402

N = PE.N
MIN_E_PCT = 0.1  # 敏感度分析：期望比例低於此值的扇區（朝陸、幾乎沒有邊界與海岸）不列入新規則


def d2(v: float) -> str:
    return str(Decimal(repr(float(v))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def pct2_counts(c: int, n: int) -> str:
    return str((Decimal(int(c)) * 100 / Decimal(int(n))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def runs_text(runs: list[list[int]], obs: np.ndarray, exp: np.ndarray) -> str:
    parts = []
    for r in runs:
        sec, bearing = PE.run_label(r)
        parts.append(f"{sec}（{bearing}；觀測{d2(obs[r].sum())}%，期望{d2(exp[r].sum())}%）")
    return "；".join(parts)


def main(out: Path, step_m: float) -> int:
    out.mkdir(parents=True)
    data = D.build_all(step_m=step_m)
    geo_rows, long_rows, tot_rows, main_rows, chg_rows, sanity = [], [], [], [], [], []
    for sid, sd in data.items():
        geo = sd["geo"]
        for k in range(N):
            geo_rows.append([sid, k, 10 * k, d2(geo.boundary_sea_m[k] / 1e3), d2(geo.coast_m[k] / 1e3),
                             f"{100 * geo.frac_boundary[k]:.3f}", f"{100 * geo.frac_coast[k]:.3f}"])  # fmt: skip
        sanity.append(
            [
                sid,
                "海上邊界總長 km",
                f"{geo.boundary_sea_total_m / 1e3:.3f}",
                "邊界全長含陸上部分 km",
                f"{geo.boundary_total_m / 1e3:.3f}",
            ]
        )
        sanity.append(
            [
                sid,
                "範圍內海岸總長 km",
                f"{geo.coast_total_m / 1e3:.3f}",
                "步行累加與總長差 m",
                f"{abs(geo.boundary_sea_m.sum() - geo.boundary_sea_total_m) + abs(geo.coast_m.sum() - geo.coast_total_m):.6f}",
            ]
        )
        zero_sec = [k for k in range(N) if geo.boundary_sea_m[k] == 0 and geo.coast_m[k] == 0]
        small = [k for k in range(N) if geo.boundary_sea_m[k] == 0 and 0 < geo.coast_m[k] < 1000]
        sanity.append(
            [
                sid,
                "邊界與海岸皆為 0 的扇區（e=0）",
                " ".join(map(str, zero_sec)) or "無",
                "無邊界、海岸長度小於 1 km 的扇區（e 接近 0）",
                " ".join(map(str, small)) or "無",
            ]
        )
        for gid, g in sd["groups"].items():
            n = g["n"]
            tot_rows.append([sid, gid, g["label"], n, g["n_cross"], d2(g["p_cross"]), g["n_coast"], d2(g["p_coast"]), g["n_dep"],
                             f"{g['exp'].sum():.6f}", f"{g['p_cross'] + g['p_coast']:.6f}", f"{abs(g['exp'].sum() - g['p_cross'] - g['p_coast']):.2e}"])  # fmt: skip
            assert abs(g["exp"].sum() - (g["p_cross"] + g["p_coast"])) < 1e-9, (sid, gid)
            for k in range(N):
                e = g["exp"][k]
                ratio = "" if e <= 0 else f"{g['obs'][k] / e:.2f}"
                long_rows.append([sid, gid, g["label"], k, 10 * k, n, int(g["counts"][k]), pct2_counts(g["counts"][k], n), d2(e), ratio,
                                  int(g["old"][k]), int(g["new"][k])])  # fmt: skip
            old_r = PE.ranked_runs(g["old"], g["obs"])
            new_r = PE.ranked_runs(g["new"], g["obs"])
            var_flags = g["new"] & (g["exp"] >= MIN_E_PCT)
            var_r = PE.ranked_runs(var_flags, g["obs"])
            for rule, runs in (
                ("舊（>1/36）", old_r),
                ("新（觀測>期望）", new_r),
                (f"新・敏感度（觀測>期望且期望≥{MIN_E_PCT:g}%）", var_r),
            ):
                for rank, r in enumerate(runs, start=1):
                    sec, bearing = PE.run_label(r)
                    main_rows.append([sid, gid, g["label"], rule, rank, sec, bearing, len(r), d2(g["obs"][r].sum()), d2(g["exp"][r].sum()),
                                      d2(g["obs"][r].sum() - g["exp"][r].sum()), "是" if rank <= 2 else ""])  # fmt: skip
            var_set = {i for r in var_r[:2] for i in r}
            old_set = {i for r in old_r[:2] for i in r}
            new_set = {i for r in new_r[:2] for i in r}
            all_old = set(np.flatnonzero(g["old"]).tolist())
            all_new = set(np.flatnonzero(g["new"]).tolist())
            fmt = lambda s: " ".join(map(str, sorted(s))) or "無"  # noqa: E731
            chg_rows.append([sid, gid, g["label"], "是" if old_set != new_set else "否", "是" if all_old != all_new else "否",
                             runs_text(old_r[:2], g["obs"], g["exp"]) or "無", runs_text(new_r[:2], g["obs"], g["exp"]) or "無",
                             fmt(old_set - new_set), fmt(new_set - old_set), runs_text(var_r[:2], g["obs"], g["exp"]) or "無", "是" if old_set != var_set else "否", len(old_r), len(new_r), fmt(all_old - all_new), fmt(all_new - all_old)])  # fmt: skip
    # 角落與正對各邊的扇區：邊界長度與全年期望（回答「朝角落的扇區涵蓋多少邊界」）
    corner_rows = []
    for sid, sd in data.items():
        lon0, lon1, lat0, lat1 = sd["bbox"]
        geo, g = sd["geo"], sd["groups"]["all"]
        anchor = sd["anchor"]
        items = [("NE 角", lon1, lat1), ("SE 角", lon1, lat0), ("SW 角", lon0, lat0), ("NW 角", lon0, lat1),
                 ("正北邊（扇區 0）", None, None), ("正東邊（扇區 9）", None, None), ("正南邊（扇區 18）", None, None), ("正西邊（扇區 27）", None, None)]  # fmt: skip
        for name, lo, la in items:
            if lo is None:
                k = {"正北": 0, "正東": 9, "正南": 18, "正西": 27}[name[:2]]
                az = 10.0 * k
            else:
                az = float(PE.ps.bearing_deg(anchor[0], anchor[1], np.array([lo]), np.array([la]))[0])
                k = int(PE.ps.sector_index(np.array([az]))[0])
            corner_rows.append(
                [
                    sid,
                    name,
                    f"{az:.2f}",
                    k,
                    d2(geo.boundary_sea_m[k] / 1e3),
                    d2(geo.coast_m[k] / 1e3),
                    d2(g["exp"][k]),
                    d2(g["obs"][k]),
                ]
            )
    write_csv(
        out / "corner_and_side_sectors.csv",
        [
            "site",
            "item",
            "bearing_deg",
            "sector",
            "sea_boundary_km_in_sector",
            "coast_km_in_sector",
            "exp_all_pct",
            "obs_all_pct",
        ],
        corner_rows,
    )
    write_csv(
        out / "sector_geometry.csv",
        [
            "site",
            "sector",
            "center_deg",
            "sea_boundary_km",
            "coast_km",
            "pct_of_sea_boundary",
            "pct_of_coast",
        ],
        geo_rows,
    )
    write_csv(
        out / "group_totals_and_check.csv",
        [
            "site",
            "group",
            "label",
            "n",
            "n_crossed",
            "P_cross_pct",
            "n_coast",
            "P_coast_pct",
            "n_deposited",
            "sum_expected_pct",
            "P_cross_plus_P_coast_pct",
            "abs_diff",
        ],
        tot_rows,
    )
    write_csv(
        out / "sector_observed_expected.csv",
        [
            "site",
            "group",
            "label",
            "sector",
            "center_deg",
            "n",
            "count_obs",
            "obs_pct",
            "exp_pct",
            "obs_over_exp",
            "old_rule_flag",
            "new_rule_flag",
        ],
        long_rows,
    )
    write_csv(
        out / "main_direction_ranges.csv",
        [
            "site",
            "group",
            "label",
            "rule",
            "rank",
            "sectors",
            "bearing_span",
            "n_sectors",
            "obs_sum_pct",
            "exp_sum_pct",
            "obs_minus_exp_pct",
            "top2",
        ],
        main_rows,
    )
    write_csv(
        out / "main_direction_changes.csv",
        [
            "site",
            "group",
            "label",
            "top2_set_changed",
            "flagged_set_changed",
            "old_top2",
            "new_top2",
            "sectors_only_in_old_top2",
            "sectors_only_in_new_top2",
            "sensitivity_top2",
            "sensitivity_top2_set_changed_vs_old",
            "n_old_ranges",
            "n_new_ranges",
            "flagged_only_old",
            "flagged_only_new",
        ],
        chg_rows,
    )
    write_csv(out / "geometry_sanity.csv", ["site", "item1", "value1", "item2", "value2"], sanity)
    print("OK", len(long_rows), len(chg_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]), float(sys.argv[2]) if len(sys.argv) > 2 else 100.0))
