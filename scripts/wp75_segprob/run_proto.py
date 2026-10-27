"""WP-75 第一階段試作：邊界／海岸分段機率圖（貢寮全年、季節 2×2；五地點全年縮圖）與統計表。

用法：python scripts/wp75_segprob/run_proto.py <圖輸出目錄（全新）> <統計表輸出目錄（全新）>
讀 run09 的 postprocess-coastB/departures.parquet，不改動任何既有輸出。
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import wp75_common as C  # noqa: E402

from lagrangian_backtracking.native_runner import figures as F  # noqa: E402
from lagrangian_backtracking.native_runner import segprob as sp  # noqa: E402
from lagrangian_backtracking.native_runner import segprob_figures as SF  # noqa: E402

SITE_ZH = {
    "gongliao": "貢寮",
    "guishan": "龜山島西側",
    "hsinchu": "新竹外海",
    "nanwan": "南灣",
    "lienchiang": "連江",
}
SEASONS = (("MAM", "春季"), ("JJA", "夏季"), ("SON", "秋季"), ("DJF", "冬季"))
TIDES = (("spring_proxy", "大潮"), ("neap_proxy", "小潮"))


def groups_of(cols: dict) -> dict[str, np.ndarray]:
    """所有分組的逐顆遮罩：全年、季節、大小潮、十類、速度序位（類內由慢到快的序位 1–10，十類合併）。"""

    g: dict[str, np.ndarray] = {"all": np.ones(len(cols["final_status_code"]), bool)}
    for k, _ in SEASONS:
        g[f"season_{k}"] = cols["deposition_season"] == k
    for k, _ in TIDES:
        g[f"tide_{k}"] = cols["deposition_tide_class"] == k
    for m in sorted(set(cols["material_id"].tolist())):
        g[f"class_{m}"] = cols["material_id"] == m
    for r in range(10):
        g[f"speedrank_{r + 1:02d}"] = cols["settling_candidate_index"] == r
    return g


def lonlat_ends(sd, seg):
    c = np.asarray(seg.geom.coords)
    lon, lat = sd.proj.unproject(np.array([c[0, 0], c[-1, 0]]), np.array([c[0, 1], c[-1, 1]]))
    return float(lon[0]), float(lat[0]), float(lon[1]), float(lat[1])


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main(fig_dir: Path, tab_dir: Path) -> int:
    for d in (fig_dir, tab_dir):
        if d.exists():
            raise SystemExit(f"輸出目錄必須是全新的：{d}")
    fig_dir.mkdir(parents=True)
    tab_dir.mkdir(parents=True)
    regions, sites = C.load_config()
    anchors = F._load_receptor_anchors(C.RELEASE)  # noqa: SLF001
    land = C.load_land()
    font = F._resolve_font(None).font_properties  # noqa: SLF001
    S: dict[str, dict] = {}
    verification, side_rows, class_rows, dist_rows, bin_rows, arc_rows = [], [], [], [], [], []
    for sid in C.SITES:
        sd = C.build_site(sid, land, regions, sites, anchors)
        sd.cols = C.load_site_columns(sid)
        C.assign_site(sd)
        cols = sd.cols
        n = len(cols["final_status_code"])
        st = cols["final_status_code"].astype(int)
        grp = groups_of(cols)
        nb, nc = len(sd.bseg), len(sd.cseg)
        bcount = {k: sp.counts_by_segment(sd.b_idx[m & (sd.b_idx >= 0)], nb) for k, m in grp.items()}
        ccount = {k: sp.counts_by_segment(sd.c_idx[m & (sd.c_idx >= 0)], nc) for k, m in grp.items()}
        gn = {k: int(m.sum()) for k, m in grp.items()}
        S[sid] = dict(sd=sd, grp=grp, bcount=bcount, ccount=ccount, gn=gn)
        # 統計表：邊界
        gkeys = list(grp)
        header = (
            [
                "segment_id",
                "side",
                "start_lon",
                "start_lat",
                "end_lon",
                "end_lat",
                "length_m",
                "land_length_m",
            ]
            + [f"count_{k}" for k in gkeys]
            + [f"pct_{k}" for k in gkeys]
        )
        rows = []
        for i, seg in enumerate(sd.bseg):
            a = lonlat_ends(sd, seg)
            rows.append(
                [
                    seg.seg_id,
                    seg.group,
                    f"{a[0]:.5f}",
                    f"{a[1]:.5f}",
                    f"{a[2]:.5f}",
                    f"{a[3]:.5f}",
                    f"{seg.length_m:.1f}",
                    f"{sd.b_land_len[i]:.1f}",
                ]
                + [int(bcount[k][i]) for k in gkeys]
                + [f"{bcount[k][i] / gn[k] * 100:.5f}" for k in gkeys]
            )
        write_csv(tab_dir / f"boundary_segments_{sid}.csv", header, rows)
        # 統計表：海岸
        header = (
            [
                "segment_id",
                "piece",
                "class",
                "island_name",
                "start_lon",
                "start_lat",
                "end_lon",
                "end_lat",
                "length_m",
                "max_assign_dist_m",
            ]
            + [f"count_{k}" for k in gkeys]
            + [f"pct_{k}" for k in gkeys]
        )
        rows = []
        maxd = np.zeros(nc)
        ok = sd.c_idx >= 0
        np.maximum.at(maxd, sd.c_idx[ok], sd.c_dist[ok])
        for i, seg in enumerate(sd.cseg):
            a = lonlat_ends(sd, seg)
            rows.append(
                [
                    seg.seg_id,
                    seg.piece,
                    seg.group,
                    seg.note,
                    f"{a[0]:.5f}",
                    f"{a[1]:.5f}",
                    f"{a[2]:.5f}",
                    f"{a[3]:.5f}",
                    f"{seg.length_m:.1f}",
                    f"{maxd[i]:.1f}",
                ]
                + [int(ccount[k][i]) for k in gkeys]
                + [f"{ccount[k][i] / gn[k] * 100:.5f}" for k in gkeys]
            )
        write_csv(tab_dir / f"coast_segments_{sid}.csv", header, rows)
        # 四邊合計（逐顆弧長，與分段無關）
        cm = sd.b_idx >= 0
        sides = sd.ring.side_of(sd.b_s[cm])
        for sdn in "NESW":
            side_rows.append([sid, sdn, int((sides == sdn).sum()), f"{(sides == sdn).sum() / n * 100:.4f}"])
        # 海岸分類合計
        for cl in sp.CLASS_ORDER:
            idxs = [i for i, s in enumerate(sd.cseg) if s.group == cl]
            if not idxs:
                continue
            cnt = int(ccount["all"][idxs].sum())
            class_rows.append(
                [
                    sid,
                    cl,
                    len(idxs),
                    f"{sum(sd.cseg[i].length_m for i in idxs) / 1e3:.2f}",
                    cnt,
                    f"{cnt / n * 100:.4f}",
                ]
            )
        named = {}
        for i, s in enumerate(sd.cseg):
            if s.group == sp.CLASS_NAMED:
                named[s.note] = named.get(s.note, 0) + int(ccount["all"][i])
        for nm, cnt in sorted(named.items()):
            class_rows.append([sid, f"{sp.CLASS_NAMED}：{nm}", "", "", cnt, f"{cnt / n * 100:.4f}"])
        ds = sp.distance_stats(sd.c_dist[ok])
        dist_rows.append(
            [
                sid,
                ds["n"],
                f"{ds['mean_m']:.1f}",
                f"{ds['p50_m']:.1f}",
                f"{ds['p90_m']:.1f}",
                f"{ds['p99_m']:.1f}",
                f"{ds['max_m']:.1f}",
                ds["n_gt_1km"],
            ]
        )
        # 驗證
        n_cr, n_co, n_dp = (
            int((st == C.STATUS_CROSSED).sum()),
            int((st == C.STATUS_COAST).sum()),
            int((st == C.STATUS_DEPOSITED).sum()),
        )
        pb, pc = bcount["all"].sum() / n * 100, ccount["all"].sum() / n * 100
        verification.append(
            [
                sid,
                n,
                n_cr,
                n_co,
                n_dp,
                f"{n_cr / n * 100:.4f}",
                f"{pb:.4f}",
                f"{n_co / n * 100:.4f}",
                f"{pc:.4f}",
                f"{n_dp / n * 100:.4f}",
                f"{pb + pc + n_dp / n * 100:.6f}",
                "OK"
                if (bcount["all"].sum() == n_cr and ccount["all"].sum() == n_co and n_cr + n_co + n_dp == n)
                else "FAIL",
            ]
        )
        # 各群組總和核對（每組：邊界分段合計＝該組穿越數、海岸分段合計＝該組到達海岸數）
        for k, m in grp.items():
            assert int(bcount[k].sum()) == int((m & (st == C.STATUS_CROSSED)).sum()), (sid, k, "boundary")
            assert int(ccount[k].sum()) == int((m & (st == C.STATUS_COAST)).sum()), (sid, k, "coast")
        # 色階各階分段數
        for kind, p in (("boundary", bcount["all"] / n * 100), ("coast", ccount["all"] / n * 100)):
            b = sp.bin_index(p, SF.EDGES)
            bin_rows.append([sid, kind] + [int((b == i).sum()) for i in range(len(SF.EDGES) + 2)])
        arc_rows.append(
            [
                sid,
                sd.arc_check["n_crossed"],
                f"{sd.ring.length_m:.1f}",
                f"{sd.bseg[-1].length_m:.1f}",
                f"{sd.arc_check['max_dist_to_ring_m']:.1f}",
                f"{sd.arc_check['max_abs_arc_diff_m_vs_old']:.3e}",
                nb,
                nc,
                len(sd.pieces),
            ]
        )
        print(sid, "OK", flush=True)
    write_csv(
        tab_dir / "verification_sums.csv",
        [
            "site",
            "n_particles",
            "n_crossed",
            "n_reached_coast",
            "n_deposited",
            "crossed_pct",
            "boundary_segment_sum_pct",
            "coast_pct",
            "coast_segment_sum_pct",
            "deposited_pct",
            "sum_three_pct",
            "status",
        ],
        verification,
    )
    write_csv(
        tab_dir / "boundary_side_totals.csv",
        ["site", "side(N北/E東/S南/W西)", "count", "pct_of_all"],
        side_rows,
    )
    write_csv(
        tab_dir / "coast_class_totals.csv",
        ["site", "class", "n_segments", "length_km", "count", "pct_of_all"],
        class_rows,
    )
    write_csv(
        tab_dir / "coast_assign_distance.csv",
        ["site", "n_points", "mean_m", "p50_m", "p90_m", "p99_m", "max_m", "n_gt_1km"],
        dist_rows,
    )
    write_csv(tab_dir / "colour_bin_segment_counts.csv", ["site", "kind"] + sp.bin_labels(SF.EDGES), bin_rows)
    write_csv(
        tab_dir / "boundary_arc_check.csv",
        [
            "site",
            "n_crossed",
            "ring_length_m",
            "last_segment_length_m",
            "max_dist_to_ring_m",
            "max_abs_arc_diff_vs_old_m",
            "n_boundary_segments",
            "n_coast_segments",
            "n_coast_pieces",
        ],
        arc_rows,
    )

    # 圖
    manifest: dict = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "edges_pct": list(SF.EDGES),
        "colors": [SF.COLOR_ZERO, *SF.COLORS],
        "bin_labels": sp.bin_labels(SF.EDGES),
        "segment_length_m": {"boundary": sp.BOUNDARY_SEGMENT_M, "coast": sp.COAST_SEGMENT_M},
        "figures": {},
    }
    drawings = {sid: SF.prepare_drawing(S[sid]["sd"], land) for sid in C.SITES}
    sid = "gongliao"
    sd, dr, gn = S[sid]["sd"], drawings[sid], S[sid]["gn"]
    n = gn["all"]
    for kind, cnt in (("boundary", S[sid]["bcount"]), ("coast", S[sid]["ccount"])):
        pct = cnt["all"] / n * 100
        tot = float(pct.sum())
        manifest["figures"][f"{sid}_{kind}_all"] = SF.render_single(
            sd, dr, kind, pct, tot, fig_dir / f"{kind}_all_{sid}.png", font
        )
        panels = []
        for k, zh in SEASONS:
            g = f"season_{k}"
            p = cnt[g] / gn[g] * 100
            panels.append(
                SF.Panel(
                    f"{zh}（{k}）｜{'穿越' if kind == 'boundary' else '海岸來源'} {p.sum():.2f}%",
                    p,
                    float(p.sum()),
                )
            )
        manifest["figures"][f"{sid}_{kind}_season"] = SF.render_panels(
            sd, dr, kind, panels, 2, 2, fig_dir / f"{kind}_season_{sid}.png", font
        )
    items = []
    for s2 in C.SITES:
        x = S[s2]
        nn = x["gn"]["all"]
        pb, pc = x["bcount"]["all"] / nn * 100, x["ccount"]["all"] / nn * 100
        items.append((x["sd"], drawings[s2], pb, pc, SITE_ZH[s2], float(pb.sum()), float(pc.sum())))
    manifest["figures"]["thumbs_five_sites"] = SF.render_thumbs(
        items, fig_dir / "thumbs_five_sites_all.png", font
    )
    manifest["font"] = F._resolve_font(None).manifest_record  # noqa: SLF001
    with (fig_dir / "manifest.json").open("x", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)
    print("done", fig_dir, tab_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
