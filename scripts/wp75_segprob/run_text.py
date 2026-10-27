# ruff: noqa: E501
"""WP-75 D：文稿素材——各地點各分組的邊界四邊小計、海岸集中段（相鄰 5 km 分段合併）與地名建議（需人工核對）。

用法：python scripts/wp75_segprob/run_text.py <統計表目錄（run_v1 已產生）>
輸出：coast_stretches_top.csv（每站每組前 5 名集中段，含端點經緯度、比例、地名建議與把握度）。
地名由下列「記憶點表」依集中段中點的最近點提出，**只是建議**：每個點的座標憑記憶、精度約 0.01°，
把握度＝點本身的把握度（高／中／低），再依距離降級（>3 km 降一級、>8 km 降為低並標「鄰近」）。
"""

from __future__ import annotations

import csv
import math
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from lagrangian_backtracking.native_runner import segprob_annot as SA  # noqa: E402
from lagrangian_backtracking.native_runner import segprob_figures as SF  # noqa: E402

SITES = ("gongliao", "guishan", "hsinchu", "nanwan", "lienchiang")
TOP_MIN_SEG_PCT = Decimal("0.5")
TOP_MIN_TOTAL_PCT = Decimal("1")
TOP_N = 5

# (名稱, 經度, 緯度, 點的把握度)；高＝座標與名稱確定，中＝大致位置，低＝僅供參考
GAZETTEER = [
    ("富貴角", 121.538, 25.298, "高"), ("石門", 121.568, 25.287, "中"), ("白沙灣", 121.51, 25.285, "中"), ("金山", 121.638, 25.231, "中"),
    ("野柳", 121.691, 25.206, "高"), ("淡水河口", 121.42, 25.18, "高"), ("林口", 121.30, 25.12, "中"),
    ("基隆港", 121.74, 25.14, "中"), ("和平島", 121.766, 25.158, "高"), ("基隆嶼", 121.789, 25.192, "高"), ("八斗子", 121.80, 25.147, "中"),
    ("深澳", 121.83, 25.13, "中"), ("鼻頭角", 121.921, 25.130, "高"), ("龍洞", 121.920, 25.108, "高"),
    ("澳底", 121.937, 25.043, "低"), ("福隆", 121.945, 25.017, "中"), ("三貂角", 122.003, 25.003, "高"),
    ("石城", 121.95, 24.97, "低"), ("大里", 121.93, 24.93, "中"), ("龜山島", 121.952, 24.842, "高"), ("頭城（烏石港）", 121.90, 24.87, "中"),
    ("蘇澳", 121.87, 24.60, "高"),
    ("竹南", 120.88, 24.69, "中"), ("新竹漁港", 120.92, 24.85, "中"), ("香山", 120.90, 24.77, "中"), ("後龍", 120.78, 24.62, "中"),
    ("通霄", 120.68, 24.49, "中"), ("苑裡", 120.65, 24.43, "中"), ("大安", 120.58, 24.35, "中"), ("竹圍", 120.97, 24.88, "低"),
    ("鵝鑾鼻", 120.85, 21.90, "高"), ("貓鼻頭", 120.74, 21.91, "高"), ("墾丁", 120.80, 21.95, "中"), ("後壁湖", 120.745, 21.945, "中"),
    ("佳樂水", 120.86, 21.99, "中"), ("恆春", 120.75, 22.00, "中"), ("小琉球", 120.371, 22.340, "高"), ("蘭嶼", 121.549, 22.048, "高"),
    ("小蘭嶼", 121.61, 21.95, "高"),
    ("南竿", 119.94, 26.15, "高"), ("北竿", 119.99, 26.22, "高"), ("東莒", 120.00, 25.96, "高"), ("西莒", 119.95, 25.97, "高"),
    ("東引", 120.49, 26.37, "高"), ("亮島", 119.94, 26.37, "中"),
    ("琯頭（福建）", 119.64, 26.13, "中"), ("閩江口（福建）", 119.62, 26.04, "中"), ("長樂（福建）", 119.5, 25.96, "中"),
    ("梅花（福建）", 119.65, 25.93, "中"), ("連江縣（福建）", 119.54, 26.20, "中"), ("黃岐（福建）", 119.80, 26.38, "低"),
    ("羅源灣（福建）", 119.7, 26.4, "低"),
]  # fmt: skip
CONF_ORDER = ["低", "中", "高"]


def km(lon1, lat1, lon2, lat2):
    return math.hypot(
        (lon1 - lon2) * 111.32 * math.cos(math.radians((lat1 + lat2) / 2)), (lat1 - lat2) * 110.57
    )


def propose(lon, lat):
    best = min(GAZETTEER, key=lambda g: km(lon, lat, g[1], g[2]))
    d = km(lon, lat, best[1], best[2])
    conf = best[3]
    name = best[0]
    if d > 8:
        conf, name = "低", f"鄰近{name}"
    elif d > 3:
        conf = CONF_ORDER[max(CONF_ORDER.index(conf) - 1, 0)]
    return name, conf, d


def main(tab: Path) -> int:
    rows = []
    for sid in SITES:
        with (tab / f"coast_segments_{sid}.csv").open(encoding="utf-8-sig") as f:
            data = list(csv.DictReader(f))
        groups = [k[len("count_") :] for k in data[0] if k.startswith("count_")]
        # 鏈：同一 piece 依序；閉合＝首段起點等於末段終點
        by: dict[str, list[int]] = {}
        for i, r in enumerate(data):
            by.setdefault(r["piece"], []).append(i)
        chains = []
        for k, v in by.items():
            a, b = data[v[0]], data[v[-1]]
            closed = (
                len(v) > 1
                and abs(float(a["start_lon"]) - float(b["end_lon"])) < 1e-4
                and abs(float(a["start_lat"]) - float(b["end_lat"])) < 1e-4
            )
            chains.append((k, v, closed))
        # 每組分母：讀 group_totals
        with (tab / "group_totals.csv").open(encoding="utf-8-sig") as f:
            gt = {(r["site"], r["group"]): int(r["n"]) for r in csv.DictReader(f)}
        for g in groups:
            cnt = np.array([int(r[f"count_{g}"]) for r in data], dtype=np.int64)
            n = gt[(sid, g)]
            st = SA.merge_stretches(
                cnt,
                n,
                chains,
                min_seg_pct=TOP_MIN_SEG_PCT,
                min_total_pct=TOP_MIN_TOTAL_PCT,
                max_n=TOP_N,
                kind="coast",
            )
            for rank, s in enumerate(st, start=1):
                segs = [data[i] for i in s.indices]
                lon0, lat0 = float(segs[0]["start_lon"]), float(segs[0]["start_lat"])
                lon1, lat1 = float(segs[-1]["end_lon"]), float(segs[-1]["end_lat"])
                mids = [
                    (
                        (float(r["start_lon"]) + float(r["end_lon"])) / 2,
                        (float(r["start_lat"]) + float(r["end_lat"])) / 2,
                    )
                    for r in segs
                ]
                mlon, mlat = float(np.mean([m[0] for m in mids])), float(np.mean([m[1] for m in mids]))
                name, conf, d = propose(mlon, mlat)
                cls = "/".join(sorted({r["class"] for r in segs}))
                isl = "/".join(sorted({r["island_name"] for r in segs if r["island_name"]}))
                rows.append([sid, g, rank, SF.pct2(s.count, n), s.count, len(segs), f"{sum(float(r['length_m']) for r in segs) / 1000:.1f}",
                             f"{lon0:.4f}", f"{lat0:.4f}", f"{lon1:.4f}", f"{lat1:.4f}", f"{mlon:.4f}", f"{mlat:.4f}", cls, isl, name, conf, f"{d:.1f}"])  # fmt: skip
    hdr = [
        "site",
        "group",
        "rank",
        "pct",
        "count",
        "n_segments",
        "length_km",
        "start_lon",
        "start_lat",
        "end_lon",
        "end_lat",
        "mid_lon",
        "mid_lat",
        "coast_class",
        "island",
        "proposed_name",
        "name_confidence",
        "dist_to_name_point_km",
    ]
    with (tab / "coast_stretches_top.csv").open("x", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(rows)
    print(len(rows), "stretches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
