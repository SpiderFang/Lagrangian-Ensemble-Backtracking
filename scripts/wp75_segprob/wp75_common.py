# ruff: noqa: E501
"""WP-75：分段機率圖的資料準備（讀釋放設定、海岸線、departures，算每站分段與逐顆歸屬）。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml
from shapely.geometry import shape

from lagrangian_backtracking.geometry import DomainProjection
from lagrangian_backtracking.native_runner import segprob as sp

HOME = Path.home()
RUN = HOME / "Workspace/lbt-runs/run09-h180-m10-v8"
RELEASE = (
    HOME
    / "Workspace/lbt-inputs/formal/v8-run09/20261019_final-sites_h180_common_m10_v8"
    / "release-configs/release-180d-full_horizon_from_deposition.yaml"
)
DEPARTURES = RUN / "finite_depth_stokes/postprocess-coastB/departures.parquet"
COASTLINE = HOME / "Workspace/lbt-inputs/coastline-taiwan.geojson"
SITES = ("gongliao", "guishan", "hsinchu", "nanwan", "lienchiang")
STATUS_CROSSED, STATUS_COAST, STATUS_DEPOSITED = 1, 8, 4


@dataclass
class SiteData:
    site_id: str
    region_id: str
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    anchor: tuple[float, float]
    proj: DomainProjection
    ring: sp.BoundaryRing
    bseg: list
    cseg: list
    pieces: list
    land_proj: object
    b_land_len: np.ndarray
    # 逐顆（本站全部 100000 顆，列序同 departures 篩選後）
    cols: dict = field(default_factory=dict)
    b_idx: np.ndarray | None = None  # 穿越者的邊界分段序號（其餘 -1）
    b_s: np.ndarray | None = None
    b_dist: np.ndarray | None = None
    c_idx: np.ndarray | None = None  # 海岸到達者的海岸分段序號（其餘 -1）
    c_dist: np.ndarray | None = None
    arc_check: dict = field(default_factory=dict)


def load_config():
    cfg = yaml.safe_load(RELEASE.read_text(encoding="utf-8"))
    regions = {d["analysis_region_id"]: d for d in cfg["domains"]}
    sites = {s["study_site_id"]: s for s in cfg["study_sites"]}
    return regions, sites


def load_land():
    feats = json.loads(COASTLINE.read_text(encoding="utf-8"))["features"]
    return [shape(f["geometry"]) for f in feats]


def build_site(site_id: str, land, regions, sites, anchors) -> SiteData:
    s = sites[site_id]
    reg = regions[s["analysis_region_id"]]
    bbox = tuple(float(v) for v in reg["bbox_lon_lat"])
    center = tuple(float(v) for v in reg["center_lonlat"])
    proj = DomainProjection(*center)
    ring = sp.boundary_ring_nw_clockwise(bbox, proj)
    pieces, land_proj = sp.extract_coast_pieces(land, bbox, proj)
    bseg = sp.boundary_segments(ring, land_proj)
    cseg = sp.coast_segments(pieces)
    b_land = np.array([sp.land_length_m(g.geom, land_proj) for g in bseg])
    return SiteData(
        site_id,
        s["analysis_region_id"],
        bbox,
        center,
        anchors[site_id][:2],
        proj,
        ring,
        bseg,
        cseg,
        pieces,
        land_proj,
        b_land,
    )


def load_site_columns(site_id: str) -> dict:
    t = pq.read_table(
        DEPARTURES,
        columns=[
            "study_site_id",
            "particle_id",
            "material_id",
            "settling_velocity_mps",
            "settling_candidate_index",
            "deposition_season",
            "deposition_tide_class",
            "final_status_code",
            "departure_x_m",
            "departure_y_m",
            "boundary_arc_length_position_m",
            "coast_arrival_x_m",
            "coast_arrival_y_m",
            "arrival_time_id",
            "member_id",
        ],
    )
    sid = t["study_site_id"].to_numpy(zero_copy_only=False)
    m = sid == site_id
    out = {}
    for name in t.schema.names:
        a = t[name].to_numpy(zero_copy_only=False)
        out[name] = a[m]
    return out


def assign_site(sd: SiteData) -> None:
    c = sd.cols
    st = c["final_status_code"].astype(int)
    n = len(st)
    crossed = st == STATUS_CROSSED
    coast = st == STATUS_COAST
    # 邊界
    s, d = sp.arc_positions(sd.ring, c["departure_x_m"][crossed], c["departure_y_m"][crossed])
    old = c["boundary_arc_length_position_m"][crossed].astype(float)
    s_from_old = sp.old_arc_to_new(old, sd.ring)
    diff = np.abs(s - s_from_old)
    diff = np.minimum(diff, sd.ring.length_m - diff)
    sd.arc_check = {
        "n_crossed": int(crossed.sum()),
        "max_dist_to_ring_m": float(d.max()) if d.size else 0.0,
        "max_abs_arc_diff_m_vs_old": float(diff.max()) if diff.size else 0.0,
        "p99_abs_arc_diff_m_vs_old": float(np.percentile(diff, 99)) if diff.size else 0.0,
    }
    bi = np.full(n, -1, dtype=np.int64)
    bi[crossed] = sp.segment_index_boundary(sd.ring, s)
    sd.b_idx = bi
    bs = np.full(n, np.nan)
    bs[crossed] = s
    sd.b_s = bs
    # 海岸
    ci = np.full(n, -1, dtype=np.int64)
    cd = np.full(n, np.nan)
    if coast.any():
        idx, dist = sp.assign_nearest(sd.cseg, c["coast_arrival_x_m"][coast], c["coast_arrival_y_m"][coast])
        ci[coast] = idx
        cd[coast] = dist
    sd.c_idx = ci
    sd.c_dist = cd
