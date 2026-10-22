"""WP-72 方法 B：由 postprocess-coastB/departures.parquet 彙總結果表（結果、海岸來源地理分布）。

用法：python coastB_tables.py <postprocess-coastB 目錄> <輸出目錄（須為全新）>
輸出：outcome_by_site.csv、coast_geography_by_site.csv、coast_geography_polygons.csv、facts_coastB.json
"""
from __future__ import annotations
import csv, json, math, os, sys
import numpy as np
import pyarrow.parquet as pq
from shapely.geometry import shape, Point
from shapely.strtree import STRtree

SITES = ["gongliao", "guishan", "hsinchu", "nanwan", "lienchiang"]
COAST = os.path.expanduser("~/Workspace/lbt-inputs/coastline-taiwan.geojson")
MATSU_BOX = (119.88, 120.55, 25.93, 26.42)  # 馬祖各島（連江縣）範圍：lon_lo, lon_hi, lat_lo, lat_hi
MATSU_MIN_DIST_TO_MAINLAND_KM = 5.0  # 馬祖各島：多邊形重心在上述範圍內，且與大陸多邊形距離 >= 此值（排除緊貼福建海岸之小島）


def main(pp: str, out: str) -> None:
    os.makedirs(out, exist_ok=False)
    cols = ["study_site_id", "material_id", "final_status_code", "backtrack_duration_seconds", "particle_id",
            "coast_arrival", "coast_arrival_hour", "coast_arrival_lon_deg", "coast_arrival_lat_deg",
            "departure_lon_deg", "departure_lat_deg", "particle_settling_velocity_mps"]
    t = pq.read_table(os.path.join(pp, "departures.parquet"), columns=cols)
    site = np.asarray(t["study_site_id"].to_pylist(), dtype=object)
    st = np.asarray(t["final_status_code"])
    dur = np.asarray(t["backtrack_duration_seconds"], dtype=np.float64)
    ca = np.asarray(t["coast_arrival"]).astype(bool)
    clon = np.asarray(t["coast_arrival_lon_deg"], dtype=np.float64)
    clat = np.asarray(t["coast_arrival_lat_deg"], dtype=np.float64)
    assert np.array_equal(ca, st == 8)
    facts = {"n_total": int(len(site)), "status": {}, "sites": {}}
    rows = []
    for s in SITES:
        m = site == s
        n = int(m.sum())
        cnt = {int(k): int(((st == k) & m).sum()) for k in np.unique(st[m])}
        facts["status"][s] = cnt
        hrs = dur[m & ca] / 3600.0
        rows.append({
            "site": s, "total": n,
            "crossed": cnt.get(1, 0), "coast": cnt.get(8, 0), "max_backtrack": cnt.get(2, 0), "touch_bottom": cnt.get(4, 0),
            "other": n - cnt.get(1, 0) - cnt.get(8, 0) - cnt.get(2, 0) - cnt.get(4, 0),
            "crossed_pct": 100 * cnt.get(1, 0) / n, "coast_pct": 100 * cnt.get(8, 0) / n,
            "max_backtrack_pct": 100 * cnt.get(2, 0) / n, "touch_bottom_pct": 100 * cnt.get(4, 0) / n,
            "coast_hour_p50": float(np.percentile(hrs, 50)) if hrs.size else "",
            "coast_hour_p90": float(np.percentile(hrs, 90)) if hrs.size else "",
            "coast_hour_max": float(hrs.max()) if hrs.size else "",
            "coast_le_1h": int((hrs <= 1.0 + 1e-9).sum()) if hrs.size else 0,
            "max_duration_days": float(dur[m].max() / 86400.0),
        })
    with open(os.path.join(out, "outcome_by_site.csv"), "x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    facts["outcome"] = rows

    # 海岸來源地理：以底圖海岸線多邊形的邊界最近距離歸屬
    g = json.load(open(COAST))
    geoms, props, polys = [], [], []
    for i, ft in enumerate(g["features"]):
        sh = shape(ft["geometry"])
        parts = list(sh.geoms) if hasattr(sh, "geoms") else [sh]
        for j, p in enumerate(parts):
            polys.append(p)
            geoms.append(p.exterior if hasattr(p, "exterior") else p.boundary)
            props.append((i, j, p.area, p.bounds))
    # 面積、邊界框
    tree = STRtree(geoms)
    # 福建大陸海岸多邊形：底圖要素 1904（範圍涵蓋連江網域內之整段福建海岸；由 bbox 與面積確認）
    main_i = next(k for k in range(len(props)) if props[k][0] == 1904)
    assert props[main_i][3][0] < 119.1 and props[main_i][2] > 1.0
    mainland = polys[main_i]
    # 多邊形分類：mainland（福建大陸）、matsu（馬祖各島）、fj_island（其他福建近岸島嶼）、taiwan 等（本島級以外之區域外多邊形不會出現在連江）
    pclass = {}
    def classify(j):
        if j in pclass:
            return pclass[j]
        c = polys[j].centroid
        if j == main_i:
            r = "mainland"
        elif props[j][2] > 0.5:
            r = "large_other"
        else:
            d = polys[j].distance(mainland) * 111.0 * math.cos(math.radians(c.y))
            inb = MATSU_BOX[0] <= c.x <= MATSU_BOX[1] and MATSU_BOX[2] <= c.y <= MATSU_BOX[3]
            r = "matsu" if (inb and d >= MATSU_MIN_DIST_TO_MAINLAND_KM) else "fj_island"
        pclass[j] = r
        return r
    geo_rows, poly_rows = [], []
    for s in SITES:
        m = ca & (site == s)
        lon, lat = clon[m], clat[m]
        n_all = int((site == s).sum())
        if lon.size == 0:
            continue
        idx = np.empty(lon.size, dtype=int); dist = np.empty(lon.size)
        for k in range(lon.size):
            pt = Point(lon[k], lat[k])
            j = int(tree.nearest(pt))
            idx[k] = j
            dist[k] = geoms[j].distance(pt) * 111.0 * math.cos(math.radians(lat[k]))
        uniq, cts = np.unique(idx, return_counts=True)
        order = np.argsort(-cts)
        for u, c in zip(uniq[order][:12], cts[order][:12]):
            b = props[u][3]
            sub_lon, sub_lat = lon[idx == u], lat[idx == u]
            poly_rows.append({"site": s, "polygon": f"{props[u][0]}:{props[u][1]}", "class": classify(int(u)), "area_deg2": props[u][2],
                              "lon_min": b[0], "lat_min": b[1], "lon_max": b[2], "lat_max": b[3],
                              "count": int(c), "pct_of_coast": 100 * c / lon.size, "pct_of_all": 100 * c / n_all,
                              "lon_p05": float(np.percentile(sub_lon, 5)), "lon_p95": float(np.percentile(sub_lon, 95)),
                              "lat_p05": float(np.percentile(sub_lat, 5)), "lat_p95": float(np.percentile(sub_lat, 95))})
        # 本島（面積最大者）與其他
        main_area = max(props[u][2] for u in uniq)
        inmatsu = (lon >= MATSU_BOX[0]) & (lon <= MATSU_BOX[1]) & (lat >= MATSU_BOX[2]) & (lat <= MATSU_BOX[3])
        big = np.array([props[j][2] > 0.5 for j in idx])  # 大陸或本島級多邊形（面積 > 0.5 平方度）
        cls = np.array([classify(int(j)) for j in idx])
        n_main = int((cls == "mainland").sum()); n_matsu = int((cls == "matsu").sum()); n_fji = int((cls == "fj_island").sum())
        geo_rows.append({
            "site": s, "n_coast": int(lon.size), "pct_of_all": 100 * lon.size / n_all,
            "lon_p05": float(np.percentile(lon, 5)), "lon_p50": float(np.percentile(lon, 50)), "lon_p95": float(np.percentile(lon, 95)),
            "lat_p05": float(np.percentile(lat, 5)), "lat_p50": float(np.percentile(lat, 50)), "lat_p95": float(np.percentile(lat, 95)),
            "dist_to_basemap_coast_km_p50": float(np.percentile(dist, 50)), "dist_to_basemap_coast_km_p90": float(np.percentile(dist, 90)),
            "dist_to_basemap_coast_km_max": float(dist.max()),
            "n_on_large_polygon": int(big.sum()), "pct_all_on_large_polygon": 100 * big.sum() / n_all,
            "n_in_matsu_box": int(inmatsu.sum()), "pct_all_in_matsu_box": 100 * inmatsu.sum() / n_all,
            "n_in_matsu_box_and_not_large": int((inmatsu & ~big).sum()),
            "n_fujian_mainland": n_main, "n_matsu_islands": n_matsu, "n_fujian_coastal_islands": n_fji,
            "pct_all_fujian_mainland": 100 * n_main / n_all, "pct_all_matsu": 100 * n_matsu / n_all, "pct_all_fujian_coastal_islands": 100 * n_fji / n_all,
            "pct_all_fujian_coast_incl_islands": 100 * (n_main + n_fji) / n_all,
            "pct_all_matsu_islands": 100 * int((inmatsu & ~big).sum()) / n_all,
            "n_other": int((~big & ~inmatsu).sum()), "pct_all_other": 100 * int((~big & ~inmatsu).sum()) / n_all,
        })
        facts["sites"][s] = geo_rows[-1]
    for name, rr in (("coast_geography_by_site.csv", geo_rows), ("coast_geography_polygons.csv", poly_rows)):
        with open(os.path.join(out, name), "x", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rr[0])); w.writeheader(); w.writerows(rr)
    with open(os.path.join(out, "facts_coastB.json"), "x", encoding="utf-8") as f:
        json.dump(facts, f, ensure_ascii=False, indent=1)
    print(json.dumps(facts["outcome"], ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
