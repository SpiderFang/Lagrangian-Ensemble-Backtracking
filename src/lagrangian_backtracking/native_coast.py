"""版本化模型岸界的嚴格載入與步內首接觸查詢；不是底圖岸線或全域空洞分類器。

只接受已審南灣主缺支援區及連江內孔；公尺線段來自原生單owner邊，底圖僅作來源
佐證。索引在靜態載入時建立，每步只查線段包圍盒，不聯集整個網格、不外插速度。
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, Point, shape
from shapely.strtree import STRtree

APPROVED_COMPONENTS = {"nanwan": "nanwan-boundary-0", "lienchiang": "lienchiang-boundary-35"}
"""本期人工審核的兩個component身分，不以owner數或land比例自動擴大覆蓋範圍。"""


def canonical_sha(value):
    """以有限數canonical JSON建立語意指紋，保留原公尺座標浮點數。"""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def file_sha(path):
    """讀取明示靜態來源位元組SHA，不掃軌跡或checkpoint歷史。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _points(geometry):
    """展開精確交集；重疊線取端點，保留首次接觸而不把共線資料buffer成面。"""
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Point":
        return [geometry]
    if geometry.geom_type == "LineString":
        return [Point(geometry.coords[0]), Point(geometry.coords[-1])]
    return [point for part in geometry.geoms for point in _points(part)]


@dataclass(frozen=True)
class CoastContact:
    """步內比例及原線交點，來源欄位交既有事件流程同步UTC、深度與年齡。"""

    fraction: float
    x_m: float
    y_m: float
    attributes: dict
    boundary_s_m: float
    # 自事件boundary_segment_id所指的公尺子段首端沿線量測，不冒稱全島累積岸距。


class NativeCoast:
    """單站已驗模型岸界；只終止從水側碰到指定線的步，不分類非法步首或未知空洞。"""

    def __init__(self, region, edges, *, manifest_sha256, region_sha256):
        """一次建立STRtree；線段與區域已由loader綁定，測試可用公尺合成資料建構。"""
        if region.geom_type != "Polygon" or not region.is_valid or region.is_empty:
            raise ValueError("模型岸界region必須為有效非空Polygon")
        self.region = region
        self.edges = tuple(dict(edge) for edge in edges)
        self.lines = tuple(LineString(edge["coordinates_m"]) for edge in self.edges)
        if not self.lines or any(not line.is_valid or line.length == 0 for line in self.lines):
            raise ValueError("模型岸界必須有有效非零長線段")
        self.tree = STRtree(self.lines)
        self.manifest_sha256 = manifest_sha256
        self.region_sha256 = region_sha256

    def first_contact(self, start_xy, end_xy):
        """返回步內首次精確線交點，水→水穿島、切線與共點均可判定。

        步首在region內是非法既有狀態，不改為COAST。步首在線上且立即朝水側離開
        不重報；朝region內或沿岸線移動在fraction=0停止。方向用相鄰精確交點間的
        線段中點判定，沒有貼岸buffer、snap或任意短距離外推。
        """
        start = Point(start_xy)
        if start_xy == end_xy or self.region.contains(start):
            return None
        path = LineString([start_xy, end_xy])
        hits = []
        for index in sorted(map(int, self.tree.query(path))):
            intersection = path.intersection(self.lines[index])
            for point in _points(intersection):
                hits.append((float(path.project(point, normalized=True)), index, point))
        for fraction, index, point in sorted(hits, key=lambda h: (h[0], h[1])):
            if fraction == 0.0:
                following = min((h[0] for h in hits if h[0] > 0), default=1.0)
                probe = path.interpolate(following / 2, normalized=True)
                if not self.region.covers(probe) and not self.lines[index].covers(probe):
                    continue
            edge = self.edges[index]
            return CoastContact(
                fraction,
                float(point.x),
                float(point.y),
                dict(
                    baseline_policy="stop_at_first_contact",
                    boundary_source="reviewed_native_model_coast_v1",
                    native_coast_manifest_sha256=self.manifest_sha256,
                    native_coast_region_sha256=self.region_sha256,
                    native_coast_edge_id=edge["edge_id"],
                    source_edge_global_index=edge["source_edge_global_index"],
                    source_face_global_index=edge["source_face_global_index"],
                ),
                float(self.lines[index].project(point)),
            )
        return None


def load_native_coasts(reference, *, config_path, site_flow, projections, geometry_sha256):
    """載入可選manifest並核對設定SHA、站點、投影、原幾何／grid來源及線區域指紋。

    設定reference必須明示path及sha256；因此只換檔案內容也不能沿用舊config/checkpoint。
    來源檔只在靜態載入時雜湊一次，任何缺檔／錯綁均拒絕，未覆蓋站不新增岸界。
    """
    if not isinstance(reference, Mapping) or set(reference) != {"path", "sha256"}:
        raise ValueError("geometry.native_coast_manifest必須含path與sha256")
    if (
        not isinstance(reference["path"], str)
        or not reference["path"]
        or reference["path"] != reference["path"].strip()
    ):
        raise ValueError("native coast path必須非空")
    if not isinstance(reference["sha256"], str) or not re.fullmatch("[0-9a-f]{64}", reference["sha256"]):
        raise ValueError("native coast SHA格式錯誤")
    path = Path(reference["path"])
    if not path.is_absolute():
        if config_path is None:
            raise ValueError("相對native coast path必須提供config YAML路徑")
        path = Path(config_path).resolve().parent / path
    actual = file_sha(path)
    if actual != reference["sha256"]:
        raise ValueError("native coast manifest SHA不符")
    payload = json.loads(path.read_text())
    required = {
        "schema_version",
        "manifest_kind",
        "status",
        "method",
        "coverage_policy",
        "source_files",
        "geometry_file_sha256",
        "records",
    }
    if set(payload) != required or payload["schema_version"] != "1.0.0":
        raise ValueError("native coast schema錯誤")
    if (
        payload["manifest_kind"] != "native_coast_manifest"
        or payload["status"] != "approved"
        or payload["method"] != "stop_at_first_contact"
        or payload["coverage_policy"] != "reviewed_two_components_only_v1"
    ):
        raise ValueError("native coast方法或核准範圍錯誤")
    if payload["geometry_file_sha256"] != dict(geometry_sha256):
        raise ValueError("native coast原geometry SHA綁定不符")
    if not payload["source_files"]:
        raise ValueError("native coast缺少靜態來源")
    for source in payload["source_files"]:
        source_path = Path(source["path"])
        if not source_path.is_absolute():
            source_path = path.parent / source_path
        if file_sha(source_path) != source["sha256"]:
            raise ValueError("native coast靜態來源SHA不符")
    coasts = {}
    for record in payload["records"]:
        site = record["study_site_id"]
        if site in coasts or site not in site_flow or site not in APPROVED_COMPONENTS:
            raise ValueError("native coast站點重複或非核准站點")
        if record["component_id"] != APPROVED_COMPONENTS[site] or record["flow_domain_id"] != site_flow[site]:
            raise ValueError("native coast component/flow錯綁")
        wkt = projections[site]._forward.target_crs.to_wkt()
        if record["projection_wkt"] != wkt:
            raise ValueError("native coast投影不符")
        region = record["region_geometry_m"]
        edges = record["edges"]
        if canonical_sha(region) != record["region_sha256"] or canonical_sha(edges) != record["edges_sha256"]:
            raise ValueError("native coast區域或邊指紋不符")
        if len({edge["edge_id"] for edge in edges}) != len(edges):
            raise ValueError("native coast邊ID重複")
        region_shape = shape(region)
        for edge in edges:
            coords = np.asarray(edge["coordinates_m"], dtype=float)
            if coords.shape != (2, 2) or not np.all(np.isfinite(coords)):
                raise ValueError("native coast線必須為兩點有限公尺座標")
            if (
                type(edge["owner_count"]) is not int
                or edge["owner_count"] != 1
                or edge["has_core_endpoint"] is not True
            ):
                raise ValueError("native coast不接受非單owner或裁切外圍邊")
            # 模型岸線必須確實在指定region原邊上；有來源全邊時另核對子段未外移。
            # 精確差集不容許任意海中線、人工封口或snap後座標混入manifest。
            line = LineString(coords)
            if not line.difference(region_shape.boundary).is_empty:
                raise ValueError("native coast停止線不在指定region邊界")
            if "source_coordinates_m" in edge:
                # 裁切交點是GEOS計算值；再次差集會因浮點共線測試拒絕其自身產物。
                # 重做原邊∩region邊界，按零容差正反座標精確比對，不加buffer／snap。
                # open／外框排除仍由已綁定建置來源負責，不接受任意截短或移動交集。
                source_contact = LineString(edge["source_coordinates_m"]).intersection(region_shape.boundary)
                parts = (
                    [source_contact]
                    if source_contact.geom_type == "LineString"
                    else list(getattr(source_contact, "geoms", ()))
                )
                if not any(
                    part.geom_type == "LineString"
                    and (
                        list(part.coords) == list(line.coords) or list(part.coords)[::-1] == list(line.coords)
                    )
                    for part in parts
                ):
                    raise ValueError("native coast停止線偏離原生來源邊的精確交集")
        coasts[site] = NativeCoast(
            region_shape, edges, manifest_sha256=actual, region_sha256=record["region_sha256"]
        )
    if not coasts:
        raise ValueError("native coast不能為空")
    return coasts, actual, canonical_sha(payload)
