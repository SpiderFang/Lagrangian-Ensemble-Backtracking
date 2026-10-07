#!/usr/bin/env python3
"""WP-51 C：由真實 OCM 網格推導龜山島西側站的海上錨點（唯讀分析，不寫入正式母體）。

規則：在設定檔 ``receptor_priority_polygon``（島西側走廊）內，取離「原錨點」最近、
且符合下列全部條件的原生網格面中心（face center）：

1. 面中心落在靜態海域（static ocean，即原生網格所有面的聯集）與 A 區 flow domain 內；
2. 50 個到達時刻（arrival）的 OCM wetdry 全為濕（persistent-wet，與 input-build 同一個判定）；
3. 距離候選區邊界（走廊 ∩ 靜態海域 ∩ flow domain）至少 ``margin`` 公尺，
   ``margin`` = ``minimum_flow_domain_margin_local_grid_scales`` × OCM 三角形邊長中位數
   （與 ``prepare_horizontal_receptor_candidates`` 的 boundary margin 完全相同的定義）；
4. 該點的 NWW 逐時四角支撐（exact-hour）在 50 個到達時刻皆有效；
5. （WP-51 整合新增）距離校閱海岸線的陸地至少 ``--min-land-distance-m`` 公尺（預設 500 m，
   且不在陸地內）。先前的版本只要求「不在陸地內」，第一名候選離島岸只有 146 m，星號
   貼著龜山島西岸；要求至少 500 m 後第一個符合者是離原錨點約 1.49 km、距陸地約 579 m
   的面中心。距離以公尺制運算座標（以 flow domain 中心為圓心的等距方位投影）計算，
   陸地取校閱海岸 GeoJSON 在原錨點周圍 0.5 度視窗內的多邊形。

重用 ``receptors.rank_anchor_snap_candidates``（與正式受體候選池同一個 persistent-wet、
覆蓋與邊界距離判定）排序，再逐一套用 NWW、OCM surface 與校閱海岸線檢核，取第一個全部
通過者。輸出 JSON 到 stdout，列出前 N 名候選與第一名的檢核明細；進度寫到 stderr。

用法（伺服器）：以 ``PYTHONPATH=<repo>/src`` 執行，設定檔與 arrival.json 取自正式母體
（唯讀），``--ocm-native-root``、``--ocm-surface-root``、``--nww-root`` 指向預處理產品根目錄，
``--coastline`` 指向校閱海岸 GeoJSON（CRS84）。本腳本不寫入任何正式母體或成果目錄。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import Point, Polygon

from lagrangian_backtracking import input_derivation as idv
from lagrangian_backtracking.config import load_config
from lagrangian_backtracking.geometry import DomainProjection, densified_bbox_polygon
from lagrangian_backtracking.manifests import load_arrival_manifest
from lagrangian_backtracking.receptors import rank_anchor_snap_candidates


def haversine_km(lon1, lat1, lon2, lat2):
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(np.asarray(lon2) - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--arrival", required=True)
    ap.add_argument("--ocm-native-root", required=True)
    ap.add_argument("--ocm-surface-root", required=True)
    ap.add_argument("--nww-root", required=True)
    ap.add_argument("--original-anchor", type=float, nargs=2, default=(121.951606, 24.843127),
                    metavar=("LON", "LAT"), help="搬移前的原錨點（預設為龜山島舊錨點，落在島上）")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--check-surface", action="store_true")
    ap.add_argument("--coastline", default=None, help="校閱海岸 GeoJSON（CRS84）；提供時要求錨點不在陸地內")
    ap.add_argument("--min-land-distance-m", type=float, default=500.0,
                    help="錨點距校閱海岸線陸地的最小距離（公尺，預設 500；0 代表只要求不在陸地內）。"
                         "大於 0 時必須提供 --coastline")
    ap.add_argument("--max-examine", type=int, default=3000,
                    help="最多逐一檢核幾個排序後的候選（預設 3000，涵蓋全部通過幾何篩選的候選）")
    args = ap.parse_args()
    if args.min_land_distance_m < 0:
        ap.error("--min-land-distance-m 不可為負")
    if args.min_land_distance_m > 0 and not args.coastline:
        ap.error("--min-land-distance-m 大於 0 時必須提供 --coastline")

    config = load_config(args.config)
    site = next(s for s in config.study_sites if s.study_site_id == "guishan")
    domain = next(d for d in config.domains if d.analysis_region_id == site.analysis_region_id)
    region = domain.analysis_region_id
    months = idv._months_for_config(config)
    native_root = Path(args.ocm_native_root)
    nww_root = Path(args.nww_root)
    flow_id = idv._resolve_flow_product_id(config, region, native_root, formal=True)
    ocm = idv._load_product(
        product="ocm_native", root=native_root, root_token=config.inputs.ocm_native_root_env,
        flow_domain_id=flow_id, months=months, config=config)
    nww = idv._load_product(
        product="nww3_analysis", root=nww_root, root_token=config.inputs.nww_analysis_root_env,
        flow_domain_id=flow_id, months=months, config=config)
    arrivals = [a for a in load_arrival_manifest(args.arrival, config, formal=True)
                if a.study_site_id == "guishan"]
    print(f"[info] guishan arrivals={len(arrivals)} flow={flow_id}", file=sys.stderr)

    projection = DomainProjection(*domain.center_lonlat)
    mesh = idv._load_native_mesh(ocm, projection)
    nfaces = mesh.face_nodes_local.shape[0]
    pair_cache = idv._load_ocm_pair_cache(ocm)
    wetdry = idv._face_times(ocm, arrivals, nfaces, pair_cache=pair_cache)
    persistent_wet = np.all(np.isfinite(wetdry) & np.isclose(wetdry, 0.0, atol=0.1), axis=0)
    print(f"[info] faces={nfaces} persistent_wet={int(persistent_wet.sum())}", file=sys.stderr)

    static_ocean = idv._face_polygon_union(mesh)
    flow_lonlat = densified_bbox_polygon(
        idv._authoritative_flow_domain_bbox_lon_lat(domain, flow_id))
    flow_metric = projection.project_geometry(flow_lonlat)
    prio_lonlat = Polygon([tuple(map(float, c)) for c in site.receptor_priority_polygon])
    prio_metric = projection.project_geometry(prio_lonlat)
    region_poly = prio_metric.intersection(static_ocean).intersection(flow_metric)

    margin_scales = float(site.model_extra.get("minimum_flow_domain_margin_local_grid_scales", 0.0))
    tri = mesh.node_xy[mesh.triangle_nodes]
    edges = np.linalg.norm(tri - np.roll(tri, -1, axis=1), axis=2)
    median_edge = float(np.median(edges[edges > 0]))
    margin_m = margin_scales * median_edge
    print(f"[info] median_edge={median_edge:.1f} m margin={margin_m:.1f} m", file=sys.stderr)

    orig = tuple(args.original_anchor)
    ox, oy = projection.project(*orig)
    # 重用正式受體候選池的 persistent-wet／覆蓋／邊界距離判定（receptors 模組），
    # 錨點因此一定是正式候選池的成員；這裡只負責排序後逐一套用外部資料檢核。
    ranked = rank_anchor_snap_candidates(
        study_site_id="guishan",
        mesh=mesh,
        region_polygon_metric=region_poly,
        original_anchor_xy=(float(ox), float(oy)),
        wetdry_at_arrivals=wetdry,
        boundary_margin_m=margin_m,
    )
    cands = [
        {
            "face_local": c.source_face_local_index, "face_global": c.source_face_global_index,
            "lon": c.lon, "lat": c.lat, "x": c.x_m, "y": c.y_m,
            "dist_to_orig_anchor_m": c.distance_to_original_anchor_m,
            "dist_to_region_boundary_m": float(Point(c.x_m, c.y_m).distance(region_poly.boundary)),
        }
        for c in ranked[: args.max_examine]
    ]
    print(f"[info] 通過 persistent-wet+靜態海域+走廊+邊界距離 的候選面：{len(ranked)}", file=sys.stderr)
    n_candidates = len(ranked)

    land_metric = None
    if args.coastline:
        from shapely.geometry import box
        from shapely.ops import unary_union
        from lagrangian_backtracking.native_runner.figures import _load_coastline
        window = box(orig[0] - 0.5, orig[1] - 0.5, orig[0] + 0.5, orig[1] + 0.5)
        lands = [g.intersection(window) for g in _load_coastline(Path(args.coastline)) if g.intersects(window)]
        land_lonlat = unary_union(lands)
        land_metric = projection.project_geometry(land_lonlat)
        print(f"[info] 原錨點在陸地內（校閱海岸）：{land_lonlat.contains(Point(*orig))}", file=sys.stderr)

    nww_cache = idv._load_nww_runtime_cache(nww)
    times = tuple(int(a.time_utc_ns) for a in arrivals)
    out = []
    first_pass = None
    for rank, c in enumerate(cands):
        if first_pass is not None and rank >= args.top:
            break  # 已找到錨點，且已列出前 top 名供審查
        _hs, valid = idv._nww_exact_hour_samples(nww_cache, lon=c["lon"], lat=c["lat"], requested_times=times)
        c["nww_all_valid"] = bool(all(valid.get(t, False) for t in times))
        c["rank"] = rank
        if land_metric is not None:
            p = Point(c["x"], c["y"])
            c["on_land_per_coastline"] = bool(land_metric.contains(p))
            c["distance_to_land_m"] = float(land_metric.distance(p))
            c["land_distance_ok"] = bool(
                (not c["on_land_per_coastline"]) and c["distance_to_land_m"] >= args.min_land_distance_m
            )
        land_ok = c.get("land_distance_ok", True)
        if (args.check_surface and c["nww_all_valid"] and first_pass is None
                and land_ok and not c.get("on_land_per_coastline", False)):
            surface_root = Path(args.ocm_surface_root)
            sid = idv._resolve_flow_product_id(config, region, surface_root, formal=True)
            surf = idv._load_product(
                product="ocm_surface", root=surface_root, root_token=config.inputs.ocm_surface_root_env,
                flow_domain_id=sid, months=months, config=config)
            elev, speed = idv._surface_series_for_location(surf, lon=c["lon"], lat=c["lat"])
            ok = all(np.isfinite(elev.get(t, float("nan"))) and np.isfinite(speed.get(t, float("nan"))) for t in times)
            c["surface_all_finite_at_arrivals"] = bool(ok)
        if (c["nww_all_valid"] and first_pass is None and c.get("surface_all_finite_at_arrivals", True)
                and land_ok):
            first_pass = c
        out.append(c)
    # 輸出
    res = {
        "original_anchor_lonlat": list(orig),
        "median_triangle_edge_m": median_edge,
        "margin_m": margin_m,
        "min_land_distance_m": args.min_land_distance_m,
        "candidate_count": n_candidates,
        "top": out,
        "selected": first_pass,
    }
    if first_pass:
        a_lon, a_lat = first_pass["lon"], first_pass["lat"]
        res["selected_distance_km_haversine"] = float(haversine_km(orig[0], orig[1], a_lon, a_lat))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
