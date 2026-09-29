"""WP-38：SCHISM 網格水深讀取、水深內插與等深線匯出。

背景
----
足跡圖需要底圖等深線（50／100／200／1000 公尺），停留時數統計也需要知道每個
網格中心的水深（計算「停留時數落在水深小於 200 公尺的比例」）。水深來源是正式
運算用的 OCM 標準化子網格（``<flow domain 快取目錄>/grid``），其中：

- ``source_lon.npy``／``source_lat.npy``：節點經緯度；
- ``source_depth_m.npy``：節點水深（公尺，向下為正，SCHISM 網格已套用最小水深）；
- ``source_face_nodes_local.npy``／``source_face_node_count.npy``：網格單元的節點
  索引（``-1`` 表示不存在的頂點），本專案網格全部是三角形。

本模組只讀這些陣列，不改寫任何輸入；等深線以三角網格線性內插後等值化，輸出
CRS84（經度、緯度）的 GeoJSON。缺水深（沒有提供網格）時由呼叫端略過等深線與
水深統計，本模組不假造任何水深。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "BathymetryError",
    "BathymetryField",
    "DEFAULT_ISOBATH_LEVELS_M",
    "load_ocm_grid_bathymetry",
    "resolve_site_grid_dirs",
    "build_isobath_geojson",
]

DEFAULT_ISOBATH_LEVELS_M: tuple[float, ...] = (50.0, 100.0, 200.0, 1000.0)
"""底圖預設的等深線水深（公尺）。"""

_CRS84_URN = "urn:ogc:def:crs:OGC:1.3:CRS84"


class BathymetryError(RuntimeError):
    """水深資料缺漏或格式不符時的錯誤（fail-closed，不以任何預設水深頂替）。"""


class BathymetryField:
    """由三角網格節點水深構成的分片線性水深場。

    ``depth_at(lon, lat)`` 對落在網格三角形內的點回傳線性內插水深，網格範圍外
    （含陸地、網格空洞）回傳 ``NaN``，不外插。
    """

    __slots__ = ("node_lon", "node_lat", "node_depth_m", "triangles", "_interpolator")

    def __init__(
        self,
        node_lon: np.ndarray,
        node_lat: np.ndarray,
        node_depth_m: np.ndarray,
        triangles: np.ndarray,
    ) -> None:
        lon = np.asarray(node_lon, dtype=np.float64)
        lat = np.asarray(node_lat, dtype=np.float64)
        depth = np.asarray(node_depth_m, dtype=np.float64)
        tri = np.asarray(triangles, dtype=np.int64)
        if lon.ndim != 1 or lon.shape != lat.shape or lon.shape != depth.shape:
            raise BathymetryError("節點經度、緯度、水深必須是相同長度的一維陣列")
        if tri.ndim != 2 or tri.shape[1] != 3 or tri.shape[0] == 0:
            raise BathymetryError("triangles 必須是 (n, 3) 且不得為空")
        if tri.min() < 0 or tri.max() >= lon.size:
            raise BathymetryError("triangles 含超出節點範圍的索引")
        if not (np.all(np.isfinite(lon)) and np.all(np.isfinite(lat)) and np.all(np.isfinite(depth))):
            raise BathymetryError("節點經緯度與水深必須全部為有限值")
        self.node_lon = lon
        self.node_lat = lat
        self.node_depth_m = depth
        self.triangles = tri
        self._interpolator = None

    def _triangulation(self):
        from matplotlib.tri import Triangulation

        return Triangulation(self.node_lon, self.node_lat, self.triangles)

    def depth_at(self, lon: np.ndarray | float, lat: np.ndarray | float) -> np.ndarray:
        """回傳指定經緯度的內插水深（公尺，向下為正）；網格外為 ``NaN``。"""

        from matplotlib.tri import LinearTriInterpolator

        if self._interpolator is None:
            tri = self._triangulation()
            self._interpolator = LinearTriInterpolator(tri, self.node_depth_m, trifinder=tri.get_trifinder())
        values = self._interpolator(np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64))
        return np.ma.filled(np.ma.asarray(values, dtype=np.float64), np.nan)


def load_ocm_grid_bathymetry(grid_dir: str | Path) -> BathymetryField:
    """唯讀讀取 OCM 標準化子網格的節點水深與三角形拓撲。

    網格單元若有四個頂點（四邊形），以對角線拆成兩個三角形；``-1`` 表示不存在
    的頂點。缺任何必要陣列一律 fail-closed。
    """

    root = Path(grid_dir)
    arrays: dict[str, np.ndarray] = {}
    for name in (
        "source_lon.npy",
        "source_lat.npy",
        "source_depth_m.npy",
        "source_face_nodes_local.npy",
        "source_face_node_count.npy",
    ):
        path = root / name
        if not path.is_file():
            raise BathymetryError(f"缺少網格陣列：{path}")
        arrays[name] = np.load(path, allow_pickle=False)
    faces = np.asarray(arrays["source_face_nodes_local.npy"], dtype=np.int64)
    counts = np.asarray(arrays["source_face_node_count.npy"], dtype=np.int64)
    if faces.ndim != 2 or faces.shape[0] != counts.shape[0]:
        raise BathymetryError("網格單元節點表與節點數表長度不一致")
    triangles = [faces[counts == 3, :3]]
    quads = faces[counts == 4, :4]
    if quads.size:
        triangles.append(quads[:, [0, 1, 2]])
        triangles.append(quads[:, [0, 2, 3]])
    if np.any((counts != 3) & (counts != 4)):
        raise BathymetryError("網格單元節點數只接受 3 或 4")
    return BathymetryField(
        arrays["source_lon.npy"],
        arrays["source_lat.npy"],
        arrays["source_depth_m.npy"],
        np.concatenate(triangles, axis=0),
    )


def resolve_site_grid_dirs(
    site_to_region: Mapping[str, str],
    region_to_flow_domain: Mapping[str, str],
    grid_root: str | Path,
) -> dict[str, Path]:
    """由「站 → 分析區 → flow domain 快取目錄名稱」對照，求出每站的網格目錄。

    ``grid_root`` 是各 flow domain 快取目錄所在的上層目錄，網格陣列位於
    ``<grid_root>/<flow domain 目錄>/grid``。找不到目錄的站直接 fail-closed。
    """

    root = Path(grid_root)
    result: dict[str, Path] = {}
    for site, region in sorted(site_to_region.items()):
        domain_name = region_to_flow_domain.get(region)
        if domain_name is None:
            raise BathymetryError(f"分析區 {region!r}（站 {site!r}）沒有對應的 flow domain 目錄")
        grid_dir = root / domain_name / "grid"
        if not grid_dir.is_dir():
            raise BathymetryError(f"站 {site!r} 的網格目錄不存在：{grid_dir}")
        result[site] = grid_dir
    return result


def build_isobath_geojson(
    fields_by_site: Mapping[str, BathymetryField],
    levels_m: Sequence[float] = DEFAULT_ISOBATH_LEVELS_M,
) -> dict[str, Any]:
    """對每站水深場求等深線，輸出 CRS84 的 GeoJSON ``FeatureCollection``。

    每個 feature 是某站某水深的 ``MultiLineString``，屬性為 ``site_id`` 與
    ``depth_m``。該水深未出現在該站網格範圍內（例如新竹外海最深只有約 105 公尺，
    沒有 200 公尺等深線）時不輸出 feature，不以空幾何頂替。
    """

    from matplotlib.figure import Figure

    features: list[dict[str, Any]] = []
    for site, field in sorted(fields_by_site.items()):
        depth = field.node_depth_m
        for level in levels_m:
            if not (float(depth.min()) < float(level) < float(depth.max())):
                continue
            fig = Figure()
            ax = fig.add_subplot(1, 1, 1)
            contour = ax.tricontour(field._triangulation(), depth, levels=[float(level)])
            segments = [
                [[float(lon), float(lat)] for lon, lat in seg] for seg in contour.allsegs[0] if len(seg) >= 2
            ]
            if not segments:
                continue
            features.append(
                {
                    "type": "Feature",
                    "properties": {"site_id": site, "depth_m": float(level)},
                    "geometry": {"type": "MultiLineString", "coordinates": segments},
                }
            )
    return {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": _CRS84_URN}},
        "features": features,
    }


def dump_geojson(payload: Mapping[str, Any], path: str | Path) -> None:
    """把 GeoJSON 寫成緊湊的 UTF-8 JSON（座標保留 6 位小數約 0.1 公尺）。"""

    def _round(obj: Any) -> Any:
        if isinstance(obj, float):
            return round(obj, 6)
        if isinstance(obj, list):
            return [_round(item) for item in obj]
        if isinstance(obj, dict):
            return {key: _round(value) for key, value in obj.items()}
        return obj

    Path(path).write_text(
        json.dumps(_round(dict(payload)), ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
