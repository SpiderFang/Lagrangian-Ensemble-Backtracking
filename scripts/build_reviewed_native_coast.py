"""由已審兩個原生邊分量建立版本化模型岸界；只讀靜態資料，不載入流場或推進粒子。

輸入為C1原始公尺節點、來源索引、既有domain/local/open及原land feature佐證。
南灣只取已審主缺支援區，連江只取112邊內孔；交集比例不是自動分類門檻。
輸出必須為全新目錄，不能掛入現行run；設定掛載與隔離重算另待C3核准。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from pyproj import Transformer
from shapely.geometry import LineString, Point, mapping, shape
from shapely.ops import polygonize, transform

from lagrangian_backtracking.mesh import triangulate_faces
from lagrangian_backtracking.native_coast import NativeCoast, canonical_sha, file_sha

REVIEWED_COMPONENT_SHA = "edf6e71bd1b04ab7ec5435fe1529389591a283f56feefcc9cfde4453d1da9012"
REVIEWED_POINTS_SHA = "a99b6b4dc504f7414978906433623d378eab1f2c130b0f072e8a57022b87c14a"
"""主審C1原始證據的位元組身分；不是任意單owner邊自動升格成岸界的入口。"""


def reviewed_region(component, domain, points):
    """從原metric node_xy polygonize；南灣差集只保留含全5點的主區，絕不補封口。

    domain差集形成的人工封口只屬region描述，不會自動變成停止線；停止線另與
    原生邊作精確交集。連江必須為已審112邊、全core端點的閉環。
    """
    site = component["site"]
    expected = {"nanwan": ("nanwan-boundary-0", 1917, 5), "lienchiang": ("lienchiang-boundary-35", 112, 2)}
    identity, edge_count, point_count = expected[site]
    if (
        component["component_id"] != identity
        or len(component["edges"]) != edge_count
        or len(points) != point_count
        or not component["is_closed_degree2"]
        or any(n["degree"] != 2 for n in component["nodes"])
    ):
        raise ValueError("原生分量身分／節點度數／已知點數不符主審C1")
    nodes = {n["local_index"]: n for n in component["nodes"]}
    lines = [LineString([nodes[i]["xy_m"] for i in e["node_local_indices"]]) for e in component["edges"]]
    loops = list(polygonize(lines))
    if len(loops) != 1 or not loops[0].is_valid:
        raise ValueError("原始邊無法形成單一有效閉環，不修形或補邊")
    if site == "nanwan":
        difference = domain.difference(loops[0])
        regions = list(difference.geoms) if difference.geom_type == "MultiPolygon" else [difference]
        selected = [r for r in regions if all(r.contains(p) for p in points)]
        if len(regions) != 2 or len(selected) != 1 or selected[0].area != max(r.area for r in regions):
            raise ValueError("南灣必須是已審兩區中含全5點之主區")
        region = selected[0]
    else:
        if not all(e["has_core_endpoint"] for e in component["edges"]):
            raise ValueError("連江指定內孔不得含裁切外圍邊")
        region = loops[0]
        if not all(region.contains(p) for p in points):
            raise ValueError("連江指定內孔未包含原2點")
    return region, nodes, lines


def extract_edges(component, region, nodes, lines, domain, opened):
    """僅取原生單owner且有core端點的實際交界，排除外框／open線，不使用buffer。

    保留原邊全座標與來源索引；若domain精確裁切原邊，停止段可為原邊的子段，
    但不可把region人工封口納入。線段順序由原boundary edge index及子段座標決定。
    """
    edges = []
    for raw, line in sorted(
        zip(component["edges"], lines, strict=True), key=lambda item: item[0]["boundary_edge_index"]
    ):
        if not raw["has_core_endpoint"]:
            continue
        contact = line.intersection(region.boundary).difference(domain.boundary).difference(opened)
        parts = [contact] if contact.geom_type == "LineString" else list(getattr(contact, "geoms", ()))
        segments = sorted(
            (p for p in parts if p.geom_type == "LineString" and p.length > 0), key=lambda p: list(p.coords)
        )
        for part_id, part in enumerate(segments):
            if len(part.coords) != 2 or raw["source_edge_global_index"] is None:
                raise ValueError("原生來源邊索引或二點線段不完整")
            edges.append(
                dict(
                    raw,
                    edge_id=f"{component['component_id']}:{raw['boundary_edge_index']}:{part_id}",
                    owner_count=1,
                    coordinates_m=list(map(list, part.coords)),
                    source_coordinates_m=list(map(list, line.coords)),
                    node_global_indices=[nodes[i]["global_index"] for i in raw["node_local_indices"]],
                )
            )
    if not edges or (component["site"] == "lienchiang" and len(edges) != 112):
        raise ValueError("指定component停止線不完整")
    return edges


def build(components_path, c1_root, output):
    """核對原C1及grid SHA，以原node陣列重建精確公尺幾何並保存七點定位／效能證據。

    所有計算限兩個静態domain；三角形只供來源owner核對與已知點的相鄰水側中心定位。
    七條中心→失敗點線段是幾何回歸，不是速度積分、真資料軌跡重算或科學驗收。
    """
    started = time.monotonic()
    components_path, c1_root, output = map(Path, (components_path, c1_root, output))
    if (
        file_sha(components_path) != REVIEWED_COMPONENT_SHA
        or file_sha(c1_root / "result.json") != REVIEWED_POINTS_SHA
    ):
        raise ValueError("C1來源SHA不是此次主審指定證據")
    components_doc = json.loads(components_path.read_text())
    prior = json.loads((c1_root / "result.json").read_text())
    hashes = json.loads((c1_root / "sha256.json").read_text())
    docs = {}
    sources = {str(components_path.resolve()): file_sha(components_path)}
    for name in ("result.json", "domain.json", "local.json", "open.json", "original-land-features.json"):
        path = c1_root / name
        if file_sha(path) != hashes[name]:
            raise ValueError(f"C1來源SHA不符：{name}")
        docs[name] = json.loads(path.read_text())
        sources[str(path.resolve())] = file_sha(path)
    records, regressions = [], []
    features = docs["original-land-features.json"]
    if file_sha(features["source_path"]) != features["source_sha256"]:
        raise ValueError("原底圖SHA不符，不能替換來源佐證")
    sources[features["source_path"]] = features["source_sha256"]
    for component in sorted(components_doc["components"], key=lambda c: c["site"]):
        site, domain_id = component["site"], component["flow_domain_id"]
        inventory = prior["grid_inventory"][site]
        metadata_path = next(
            Path(p) for p in prior["source_sha256"] if p.endswith("/" + domain_id + "/grid/metadata.json")
        )
        metadata_sha = prior["source_sha256"][str(metadata_path)]
        if (
            file_sha(metadata_path) != metadata_sha
            or json.loads(metadata_path.read_text()) != inventory["metadata"]
        ):
            raise ValueError("原grid metadata SHA或內文與C1不一致")
        sources[str(metadata_path)] = metadata_sha
        forward = Transformer.from_crs("EPSG:4326", component["projection_wkt"], always_xy=True)
        domain = transform(
            forward.transform,
            shape(
                next(
                    r["geometry"] for r in docs["domain.json"]["records"] if r["flow_domain_id"] == domain_id
                )
            ),
        )
        from shapely.ops import unary_union

        opened = unary_union(
            [
                transform(forward.transform, shape(r["geometry"]))
                for r in docs["open.json"]["records"]
                if r["owner_kind"] == "flow_domain" and r["owner_id"] == domain_id
            ]
        )

        # 只開啟已由C1登錄的靜態npy；不用raw NetCDF，也不載入任何速度月份。
        def load(name, domain_id=domain_id, site=site):
            """將當前domain明確綁定於來源查詢，避免跨站索引或來源被後續迴圈替換。"""
            paths = [p for p in prior["source_sha256"] if p.endswith("/" + domain_id + "/grid/" + name)]
            if len(paths) != 1 or file_sha(paths[0]) != prior["source_sha256"][paths[0]]:
                raise ValueError(f"grid來源錯綁：{site}/{name}")
            sources[paths[0]] = prior["source_sha256"][paths[0]]
            return np.load(paths[0], mmap_mode="r", allow_pickle=False)

        xy = np.column_stack(forward.transform(load("source_lon.npy"), load("source_lat.npy")))
        triangles, face_ids = triangulate_faces(
            load("source_face_nodes_local.npy"), load("source_face_node_count.npy"), xy
        )
        node_global = load("source_node_global_index.npy")
        face_global = load("source_face_global_index.npy")
        native_edges = load("source_edge_nodes_local.npy")
        global_edges = load("source_edge_global_index.npy")
        points = [p for p in prior["points"] if p["site"] == site]
        region, nodes, lines = reviewed_region(
            component, domain, [Point(p["point"]["x_m"], p["point"]["y_m"]) for p in points]
        )
        for index, node in nodes.items():
            if not np.array_equal(xy[index], node["xy_m"]) or int(node_global[index]) != node["global_index"]:
                raise ValueError("C1座標不是原生metric node_xy或node ID已變")
        for edge in component["edges"]:
            i = edge["source_edge_local_index"]
            owner = edge["owner_triangle_id"]
            if (
                i is None
                or set(map(int, native_edges[i])) != set(edge["node_local_indices"])
                or int(global_edges[i]) != edge["source_edge_global_index"]
                or int(face_global[face_ids[owner]]) != edge["source_face_global_index"]
            ):
                raise ValueError("原native edge/owner/source face身分不符")
        edges = extract_edges(component, region, nodes, lines, domain, opened)
        region_payload = mapping(region)
        fid_index = "62" if site == "nanwan" else "597"
        feature = features["features_by_original_index"][fid_index]
        land = transform(forward.transform, shape(feature["geometry"]))
        record = dict(
            study_site_id=site,
            flow_domain_id=domain_id,
            component_id=component["component_id"],
            projection_wkt=component["projection_wkt"],
            projection_sha256=component["projection_sha256"],
            region_geometry_m=region_payload,
            region_sha256=canonical_sha(region_payload),
            edges=edges,
            edges_sha256=canonical_sha(edges),
            core_policy=component["core_policy"],
            core_bbox_lonlat=component["core_bbox_lonlat"],
            grid_metadata=inventory["metadata"],
            grid_metadata_sha256=metadata_sha,
            component_source_sha256=REVIEWED_COMPONENT_SHA,
            supporting_land=dict(
                source_sha256=features["source_sha256"],
                original_feature_index=int(fid_index),
                properties=feature["properties"],
                intersection_fraction_evidence_only=region.intersection(land).area / region.area,
            ),
            region_area_m2=region.area,
        )
        coast = NativeCoast(region, edges, manifest_sha256="pending", region_sha256=record["region_sha256"])
        for point in points:
            hint = next(e["triangle_id"] for e in point["edges"] if e["is_saved_hint"])
            start = np.mean(xy[triangles[hint]], axis=0).tolist()
            end = [point["point"]["x_m"], point["point"]["y_m"]]
            contact = coast.first_contact(tuple(start), tuple(end))
            if contact is None or region.covers(Point(start)):
                raise ValueError("已知點相鄰水側三角形未得到指定岸界首交點")
            t = time.monotonic()
            for _ in range(1000):
                coast.first_contact(tuple(start), tuple(end))
            regressions.append(
                dict(
                    site=site,
                    particle_id=point["particle_id"],
                    hint_triangle_id=hint,
                    start_xy_m=start,
                    end_xy_m=end,
                    contact=vars(contact),
                    query_1000_seconds=time.monotonic() - t,
                )
            )
        records.append(record)
    payload = dict(
        schema_version="1.0.0",
        manifest_kind="native_coast_manifest",
        status="approved",
        method="stop_at_first_contact",
        coverage_policy="reviewed_two_components_only_v1",
        source_files=[dict(path=p, sha256=h) for p, h in sorted(sources.items())],
        geometry_file_sha256=dict(
            domain=hashes["domain.json"], local=hashes["local.json"], open_boundary=hashes["open.json"]
        ),
        records=records,
    )
    output.mkdir(parents=False, exist_ok=False)
    manifest = output / "native_coast_manifest.json"
    manifest.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False))
    (output / "geometry-regression.json").write_text(
        json.dumps(
            dict(
                formal_result=False,
                advance_called=False,
                script_sha256=file_sha(__file__),
                manifest_sha256=file_sha(manifest),
                elapsed_seconds=time.monotonic() - started,
                points=regressions,
            ),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components", required=True)
    parser.add_argument("--c1-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    build(args.components, args.c1_root, args.output)
