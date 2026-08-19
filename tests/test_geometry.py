"""公尺投影、local-domain 重疊、步內 crossing 與 maximin 測試。"""

from __future__ import annotations

import numpy as np
from shapely.geometry import Point, box

from lagrangian_backtracking.geometry import (
    DomainProjection,
    build_anchor_local_domain,
    deterministic_maximin,
    first_polygon_crossing,
)


def test_projection_round_trip_is_sub_centimetre() -> None:
    """A 區角落與中心的 lon/lat round-trip 應遠低於 0.05 m 契約。"""

    projection = DomainProjection(122.05, 25.05)
    lon = np.array([121.306315, 122.05, 122.793685])
    lat = np.array([24.600844, 25.05, 25.499156])
    x_m, y_m = projection.project(lon, lat)
    restored_lon, restored_lat = projection.unproject(x_m, y_m)
    assert np.max(np.abs(restored_lon - lon)) < 1e-10
    assert np.max(np.abs(restored_lat - lat)) < 1e-10


def test_anchor_domains_may_overlap_without_merging_identity() -> None:
    """相距約 30 km 的兩個 25 km local circles 應自然重疊。"""

    projection = DomainProjection(122.05, 25.05)
    ocean = box(-200_000, -200_000, 200_000, 200_000)
    gongliao = build_anchor_local_domain(
        projection=projection,
        anchor_lonlat=(121.92807, 25.11245),
        radius_m=25_000,
        static_ocean_polygon_metric=ocean,
    )
    guishan = build_anchor_local_domain(
        projection=projection,
        anchor_lonlat=(121.951606, 24.843127),
        radius_m=25_000,
        static_ocean_polygon_metric=ocean,
    )
    assert gongliao.intersection(guishan).area > 0
    assert gongliao is not guishan


def test_first_crossing_returns_fraction_and_boundary_arclength() -> None:
    """線段由正方形中心向東離域時，交點與 fraction 必須解析一致。"""

    polygon = box(-10.0, -10.0, 10.0, 10.0)
    crossing = first_polygon_crossing((0.0, 0.0), (20.0, 0.0), polygon)
    assert crossing is not None
    assert np.isclose(crossing.x_m, 10.0)
    assert np.isclose(crossing.y_m, 0.0)
    assert np.isclose(crossing.fraction, 0.5)
    assert 0.0 <= crossing.boundary_s_m <= polygon.exterior.length


def test_deterministic_maximin_starts_near_anchor_and_spreads() -> None:
    """選點先靠近 anchor，後續應選最遠角且重跑一致。"""

    candidates = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0], [5.0, 5.0]])
    selected = deterministic_maximin(candidates, count=3, anchor_xy=(0.1, 0.1))
    repeated = deterministic_maximin(candidates, count=3, anchor_xy=(0.1, 0.1))
    assert selected[0] == 0
    assert selected[1] == 3
    assert np.array_equal(selected, repeated)
    assert Point(*candidates[selected[2]]).distance(Point(*candidates[selected[0]])) > 0
