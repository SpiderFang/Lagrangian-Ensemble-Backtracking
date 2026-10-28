# ruff: noqa: E501, B023
"""WP-75 v3：五地點全部分段機率圖（全年、十類 2×5、季節 2×2、大小潮 1×2；邊界與海岸各一張）與統計表。

v3：邊界圖移除邊小計、括號標出全部集中段（單張寫 %、多面板編圈號並把數值放進標題）；海岸圖加流場範圍細線。

用法：python scripts/wp75_segprob/run_v3.py <圖輸出根目錄（全新）> <統計表輸出目錄（全新）>
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import wp75_common as C  # noqa: E402
import wp75_make as M  # noqa: E402

from lagrangian_backtracking.native_runner import figures as F  # noqa: E402
from lagrangian_backtracking.native_runner import segprob as sp  # noqa: E402
from lagrangian_backtracking.native_runner import segprob_annot as SA  # noqa: E402
from lagrangian_backtracking.native_runner import segprob_figures as SF  # noqa: E402
from lagrangian_backtracking.native_runner.arc_by_group import group_label_zh  # noqa: E402
from lagrangian_backtracking.native_runner.display_names import material_display_name_zh  # noqa: E402

SEASON_ORDER = ("DJF", "MAM", "JJA", "SON")  # 同舊版圓形圖的順序
TIDE_ORDER = ("spring_proxy", "neap_proxy")
KIND_ZH = {"boundary": "穿越", "coast": "海岸來源"}


def build_groups(cols: dict) -> tuple[dict[str, np.ndarray], dict[str, str], list[str]]:
    """分組遮罩、顯示名稱與類別順序（依類別代表沉降速度絕對值由慢到快，同舊版）。"""

    n = len(cols["final_status_code"])
    g: dict[str, np.ndarray] = {"all": np.ones(n, bool)}
    label = {"all": "全年"}
    for k in SEASON_ORDER:
        g[f"season_{k}"] = cols["deposition_season"] == k
        label[f"season_{k}"] = group_label_zh("season", k)
    for k in TIDE_ORDER:
        g[f"tide_{k}"] = cols["deposition_tide_class"] == k
        label[f"tide_{k}"] = group_label_zh("tide", k)
    mats = sorted(set(cols["material_id"].tolist()))
    speed = {
        m: float(np.median(np.abs(cols["settling_velocity_mps"][cols["material_id"] == m]))) for m in mats
    }
    mats = sorted(mats, key=lambda m: (speed[m], m))
    for m in mats:
        g[f"class_{m}"] = cols["material_id"] == m
        label[f"class_{m}"] = material_display_name_zh(m)
    return g, label, mats


def lonlat_ends(sd, seg):
    c = np.asarray(seg.geom.coords)
    lon, lat = sd.proj.unproject(np.array([c[0, 0], c[-1, 0]]), np.array([c[0, 1], c[-1, 1]]))
    return float(lon[0]), float(lat[0]), float(lon[1]), float(lat[1])


LABEL_MAX_PT = 15.0  # 標籤中心到括號（直線取中點、繞角取括號線）的容許距離


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main(fig_root: Path, tab_dir: Path) -> int:
    for d in (fig_root, tab_dir):
        if d.exists():
            raise SystemExit(f"輸出目錄必須是全新的：{d}")
    fig_root.mkdir(parents=True)
    tab_dir.mkdir(parents=True)
    regions, sites = C.load_config()
    anchors = F._load_receptor_anchors(C.RELEASE)  # noqa: SLF001
    land = C.load_land()
    font = F._resolve_font(None).font_properties  # noqa: SLF001
    manifest: dict = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "edges_pct": list(SF.EDGES),
        "colors": [SF.COLOR_ZERO, *SF.COLORS],
        "bin_labels": sp.bin_labels(SF.EDGES),
        "percent_rule": "整數顆數 ÷ 分母 × 100，Decimal ROUND_HALF_UP 到小數 2 位",
        "figures": {},
    }
    group_rows, verification, stretch_all, title_all, cov_rows = [], [], [], [], []
    collision_log: list = []
    bracket_log: list = []
    label_log: list = []
    pct_diff_all: list = []
    tiny_all: list = []
    bracket_panel: dict = {}
    for sid in C.SITES:
        sd = C.build_site(sid, land, regions, sites, anchors)
        sd.cols = C.load_site_columns(sid)
        C.assign_site(sd)
        cols = sd.cols
        st = cols["final_status_code"].astype(int)
        grp, label, mats = build_groups(cols)
        nb, nc = len(sd.bseg), len(sd.cseg)
        gn = {k: int(m.sum()) for k, m in grp.items()}
        bcount = {k: sp.counts_by_segment(sd.b_idx[m & (sd.b_idx >= 0)], nb) for k, m in grp.items()}
        ccount = {k: sp.counts_by_segment(sd.c_idx[m & (sd.c_idx >= 0)], nc) for k, m in grp.items()}
        # 驗證與群組合計
        for k, m in grp.items():
            n_cr = int((m & (st == C.STATUS_CROSSED)).sum())
            n_co = int((m & (st == C.STATUS_COAST)).sum())
            n_dp = int((m & (st == C.STATUS_DEPOSITED)).sum())
            assert (
                int(bcount[k].sum()) == n_cr and int(ccount[k].sum()) == n_co and n_cr + n_co + n_dp == gn[k]
            ), (sid, k)
            group_rows.append(
                [
                    sid,
                    k,
                    label[k],
                    gn[k],
                    n_cr,
                    SF.pct2(n_cr, gn[k]),
                    n_co,
                    SF.pct2(n_co, gn[k]),
                    n_dp,
                    SF.pct2(n_dp, gn[k]),
                ]
            )
        verification.append([sid, gn["all"], "OK"])
        for k in grp:
            stb = SA.boundary_stretches(sd, bcount[k], gn[k])
            cov_c, cr_c = sum(x.count for x in stb), int(bcount[k].sum())
            cov_rows.append([sid, k, label[k], len(stb), SF.pct2(cr_c, gn[k]), SF.pct2(cov_c, gn[k]),
                             (f"{Decimal(cov_c) * 100 / Decimal(cr_c):.2f}" if cr_c else ""), " ".join(SF.pct2(x.count, gn[k]) for x in stb)])  # fmt: skip
        # 逐段表（全年、季節、大小潮、十類）
        gkeys = list(grp)
        hdr_b = [
            "segment_id",
            "side",
            "start_lon",
            "start_lat",
            "end_lon",
            "end_lat",
            "length_m",
            "land_length_m",
        ]
        rows = []
        for i, seg in enumerate(sd.bseg):
            a = lonlat_ends(sd, seg)
            rows.append([seg.seg_id, seg.group, f"{a[0]:.5f}", f"{a[1]:.5f}", f"{a[2]:.5f}", f"{a[3]:.5f}", f"{seg.length_m:.1f}", f"{sd.b_land_len[i]:.1f}"]
                        + [int(bcount[k][i]) for k in gkeys] + [SF.pct2(int(bcount[k][i]), gn[k]) for k in gkeys])  # fmt: skip
        write_csv(
            tab_dir / f"boundary_segments_{sid}.csv",
            hdr_b + [f"count_{k}" for k in gkeys] + [f"pct_{k}" for k in gkeys],
            rows,
        )
        hdr_c = [
            "segment_id",
            "piece",
            "class",
            "island_name",
            "start_lon",
            "start_lat",
            "end_lon",
            "end_lat",
            "length_m",
        ]
        rows = []
        for i, seg in enumerate(sd.cseg):
            a = lonlat_ends(sd, seg)
            rows.append([seg.seg_id, seg.piece, seg.group, seg.note, f"{a[0]:.5f}", f"{a[1]:.5f}", f"{a[2]:.5f}", f"{a[3]:.5f}", f"{seg.length_m:.1f}"]
                        + [int(ccount[k][i]) for k in gkeys] + [SF.pct2(int(ccount[k][i]), gn[k]) for k in gkeys])  # fmt: skip
        write_csv(
            tab_dir / f"coast_segments_{sid}.csv",
            hdr_c + [f"count_{k}" for k in gkeys] + [f"pct_{k}" for k in gkeys],
            rows,
        )

        # 圖
        dr = SF.prepare_drawing(sd, land)
        recs, srows, tlog, extra = M.render_site(
            sd, dr, grp, label, mats, bcount, ccount, gn, fig_root / sid, font
        )
        title_all.extend(tlog)
        pct_diff_all.extend(extra["pct_diff"])
        tiny_all.extend(extra["tiny"])
        manifest["figures"][sid] = recs
        stretch_all.extend(srows)

        def _collect(x, path):
            if isinstance(x, dict):
                for c in x.get("collisions", []) or []:
                    collision_log.append([sid, path, c["label"], ",".join(c["violations"])])  # noqa: B023
                if path.startswith("boundary") and any("bracket_side" in t for t in x.get("stretches", [])):
                    pi = bracket_panel.get((sid, path), 0)  # noqa: B023
                    bracket_panel[(sid, path)] = pi + 1  # noqa: B023
                    for rk, t in enumerate(x["stretches"], start=1):
                        e_start = max(
                            abs(a - b)
                            for a, b in zip(t["bracket_foot_start_lonlat"], t["start_lonlat"], strict=True)
                        )
                        e_end = max(
                            abs(a - b)
                            for a, b in zip(t["bracket_foot_end_lonlat"], t["end_lonlat"], strict=True)
                        )
                        ok = max(e_start, e_end) <= 1e-4 and t["bracket_span_err_m"] <= 1.0
                        lp, la, cn = t["label_perp_pt"], t["label_along_pt"], t["bracket_has_corner"]
                        has_lab = lp is not None  # 未編號的括號（多面板第 4 個以後）沒有標籤
                        lab_ok = has_lab and lp <= LABEL_MAX_PT and (cn or la <= LABEL_MAX_PT)
                        label_log.append(
                            [
                                sid,
                                path,
                                pi + 1,
                                rk,
                                t["marker"] or f"{t['shown_pct']}%",
                                t["marker_where"],
                                "繞角" if cn else "直線",
                                lp,
                                la,
                                "ok" if lab_ok else ("無標籤" if not has_lab else "FAIL"),
                            ]
                        )  # noqa: B023
                        bracket_log.append(
                            [
                                sid,
                                path,
                                pi + 1,
                                rk,
                                t["pct"],
                                t["bracket_side"],
                                f"{t['start_lonlat'][0]:.4f}",
                                f"{t['start_lonlat'][1]:.4f}",
                                f"{t['end_lonlat'][0]:.4f}",
                                f"{t['end_lonlat'][1]:.4f}",
                                f"{max(e_start, e_end):.2e}",
                                t["bracket_span_err_m"],
                                t["sea_clip_m"],
                                t["bracket_over_islet"],
                                t["marker"],
                                t["marker_where"],
                                "ok" if ok else "FAIL",
                            ]
                        )  # noqa: B023,E501
                for kk, y in x.items():
                    if kk != "collisions":
                        _collect(y, path)
            elif isinstance(x, list):
                for y in x:
                    _collect(y, path)

        for fname, rec_ in recs.items():
            _collect(rec_, fname)
        print(sid, "figures done", flush=True)
        # 所有上色分段必須在可見範圍內（窗口）
        w = SF.WINDOWS[sid]
        allc = [a for i, parts in enumerate(dr.c_lines) if ccount["all"][i] > 0 for a in parts]
        xy = np.vstack(allc)
        assert (
            xy[:, 0].min() >= w[0]
            and xy[:, 0].max() <= w[1]
            and xy[:, 1].min() >= w[2]
            and xy[:, 1].max() <= w[3]
        ), sid
        manifest.setdefault("coloured_extent_lon_lat", {})[sid] = [
            float(xy[:, 0].min()),
            float(xy[:, 0].max()),
            float(xy[:, 1].min()),
            float(xy[:, 1].max()),
        ]
    write_csv(
        tab_dir / "group_totals.csv",
        [
            "site",
            "group",
            "label",
            "n",
            "crossed_n",
            "crossed_pct",
            "coast_n",
            "coast_pct",
            "deposited_n",
            "deposited_pct",
        ],
        group_rows,
    )
    write_csv(
        tab_dir / "annotated_stretches.csv",
        [
            "site",
            "kind",
            "group",
            "rank",
            "pct",
            "count",
            "n_segments",
            "start_lon",
            "start_lat",
            "end_lon",
            "end_lat",
            "first_segment",
            "last_segment",
            "marker",
            "where",
        ],
        stretch_all,
    )
    write_csv(
        tab_dir / "boundary_stretch_coverage.csv",
        [
            "site",
            "group",
            "label",
            "n_stretches",
            "crossing_pct",
            "bracketed_pct",
            "coverage_pct_of_crossing",
            "stretch_pcts",
        ],
        cov_rows,
    )
    write_csv(
        tab_dir / "panel_title_values.csv",
        [
            "site",
            "figure",
            "group",
            "label",
            "n_stretches",
            "n_numbered",
            "n_unnumbered",
            "variant_index",
            "title_values_line",
        ],
        title_all,
    )
    write_csv(
        tab_dir / "pct_lr_vs_rounded.csv",
        ["site", "figure", "group", "item", "pct2_each_rounded", "figure_value_largest_remainder", "differs"],
        [[*r_, "是" if r_[4] != r_[5] else ""] for r_ in pct_diff_all],
    )
    write_csv(
        tab_dir / "unlabelled_small_gaps.csv",
        ["site", "segment", "n_seg", "reason", "pct2", "lon", "lat", "prev_neighbour", "next_neighbour"],
        tiny_all,
    )
    rem: dict = {}
    for r in tiny_all:
        k = rem.setdefault(r[0], [0, 0])
        k[0] += 1
        k[1] += int(round(float(r[4]) * 100))
    write_csv(
        tab_dir / "unbracketed_remainder.csv",
        ["site", "figure", "n_pieces", "sum_of_piece_pct2"],
        [
            [sid, "boundary_all", rem.get(sid, [0, 0])[0], f"{rem.get(sid, [0, 0])[1] / 100:.2f}"]
            for sid in C.SITES
        ],
    )
    write_csv(
        tab_dir / "label_distance_check.csv",
        [
            "site",
            "figure",
            "panel",
            "rank",
            "label",
            "label_where",
            "bracket_shape",
            "label_to_bracket_pt",
            "label_from_mid_pt",
            "label_check",
        ],
        label_log,
    )
    print("label distance FAIL:", sum(r_[-1] == "FAIL" for r_ in label_log), "of", len(label_log), flush=True)
    assert not any(r_[-1] == "FAIL" for r_ in label_log), "標籤離括號太遠"
    write_csv(
        tab_dir / "bracket_check.csv",
        [
            "site",
            "figure",
            "panel",
            "rank",
            "pct",
            "bracket_side",
            "stretch_start_lon",
            "stretch_start_lat",
            "stretch_end_lon",
            "stretch_end_lat",
            "max_foot_diff_deg",
            "span_err_m",
            "sea_clip_m",
            "over_islet",
            "marker",
            "marker_where",
            "span_check",
        ],
        bracket_log,
    )
    assert all(r_[-1] == "ok" for r_ in bracket_log), "括號與集中段起訖不一致"
    write_csv(tab_dir / "label_collisions.csv", ["site", "figure", "label", "violations"], collision_log)
    manifest["stretch_thresholds"] = {
        "boundary_min_segment_pct": str(SA.STRETCH_BOUNDARY_MIN_SEG_PCT),
        "boundary_min_total_pct": str(SA.STRETCH_BOUNDARY_MIN_TOTAL_PCT),
        "boundary_max_numbered_per_panel": SA.STRETCH_BOUNDARY_MAX_NUMBERED,
        "coast_min_segment_pct": str(SA.STRETCH_COAST_MIN_SEG_PCT),
        "coast_min_total_pct": str(SA.STRETCH_COAST_MIN_TOTAL_PCT),
        "coast_max_per_panel": SA.STRETCH_COAST_MAX_PER_PANEL,
    }
    manifest["font"] = F._resolve_font(None).manifest_record  # noqa: SLF001
    with (fig_root / "manifest.json").open("x", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
