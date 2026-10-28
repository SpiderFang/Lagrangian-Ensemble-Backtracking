# ruff: noqa: E501
"""WP-75 v3：邊界集中段門檻比較表（分段門檻 0.5%／1%／2%，集中段合計門檻 3%）。

用法：python run_stretch_thr.py <輸出目錄（全新）>
"""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import run_v1 as R1  # noqa: E402
import wp75_common as C  # noqa: E402

from lagrangian_backtracking.native_runner import figures as F  # noqa: E402
from lagrangian_backtracking.native_runner import segprob as sp  # noqa: E402
from lagrangian_backtracking.native_runner import segprob_annot as SA  # noqa: E402
from lagrangian_backtracking.native_runner import segprob_figures as SF  # noqa: E402

THRS = (Decimal("0.5"), Decimal("1"), Decimal("2"))
TOTAL_MIN = Decimal("3")


def ll(sd, xy):
    lon, lat = sd.proj.unproject(np.array([xy[0]]), np.array([xy[1]]))
    return float(lon[0]), float(lat[0])


def main(out: Path) -> int:
    out.mkdir(parents=True)
    regions, sites = C.load_config()
    anchors = F._load_receptor_anchors(C.RELEASE)  # noqa: SLF001
    land = C.load_land()
    long_rows, sum_rows = [], []
    for sid in C.SITES:
        sd = C.build_site(sid, land, regions, sites, anchors)
        sd.cols = C.load_site_columns(sid)
        C.assign_site(sd)
        grp, label, _mats = R1.build_groups(sd.cols)
        nb = len(sd.bseg)
        for k, m in grp.items():
            n = int(m.sum())
            cnt = sp.counts_by_segment(sd.b_idx[m & (sd.b_idx >= 0)], nb)
            cross = int(cnt.sum())
            for thr in THRS:
                st = SA.merge_stretches(
                    cnt,
                    n,
                    SA.boundary_chains(sd),
                    min_seg_pct=thr,
                    min_total_pct=TOTAL_MIN,
                    max_n=None,
                    kind="boundary",
                )
                exact = int(
                    sum(1 for c in cnt if c > 0 and Decimal(int(c)) * 100 == thr * n)
                )  # 恰等於門檻的分段數
                cov = sum(s.count for s in st)
                desc = []
                for rank, s in enumerate(st, start=1):
                    a = sd.bseg[s.indices[0]].geom.coords[0]
                    b = sd.bseg[s.indices[-1]].geom.coords[-1]
                    (a_lon, a_lat), (b_lon, b_lat) = ll(sd, a), ll(sd, b)
                    long_rows.append([sid, k, label[k], str(thr), rank, SF.pct2(s.count, n), s.count, len(s.indices), sd.bseg[s.indices[0]].seg_id, sd.bseg[s.indices[-1]].seg_id,
                                      f"{a_lon:.3f}", f"{a_lat:.3f}", f"{b_lon:.3f}", f"{b_lat:.3f}"])  # fmt: skip
                    desc.append(
                        f"{SF.pct2(s.count, n)}%({a_lon:.2f}E,{a_lat:.2f}N→{b_lon:.2f}E,{b_lat:.2f}N)"
                    )
                sum_rows.append([sid, k, label[k], str(thr), len(st), SF.pct2(cross, n), SF.pct2(cov, n),
                                 (f"{Decimal(cov) * 100 / Decimal(cross):.2f}" if cross else ""), "；".join(desc), exact])  # fmt: skip
        print(sid, "done", flush=True)
    R1.write_csv(out / "stretch_summary_by_threshold.csv",
                 ["site", "group", "label", "segment_threshold_pct", "n_stretches", "crossing_pct", "bracketed_pct", "coverage_pct_of_crossing", "stretches_pct_and_range", "segments_exactly_at_threshold"], sum_rows)  # fmt: skip
    R1.write_csv(out / "stretch_list_by_threshold.csv",
                 ["site", "group", "label", "segment_threshold_pct", "rank_by_pct", "pct", "count", "n_segments", "first_segment", "last_segment", "start_lon", "start_lat", "end_lon", "end_lat"], long_rows)  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
