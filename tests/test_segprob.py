"""WP-75：邊界／海岸分段機率圖計算核心（segprob）的單元測試。"""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import LineString, box

from lagrangian_backtracking.geometry import DomainProjection
from lagrangian_backtracking.native_runner import segprob as sp

BBOX = (121.3, 122.8, 24.6, 25.5)
CENTER = (122.05, 25.05)


@pytest.fixture(scope="module")
def ring() -> sp.BoundaryRing:
    return sp.boundary_ring_nw_clockwise(BBOX, DomainProjection(*CENTER))


def test_cut_edges_keep_partial_last_segment_and_short_line_is_one_segment() -> None:
    e = sp.cut_edges(25_000.0, 10_000.0)
    assert e.tolist() == [0.0, 10_000.0, 20_000.0, 25_000.0]
    assert sp.cut_edges(20_000.0, 10_000.0).tolist() == [0.0, 10_000.0, 20_000.0]  # 恰整除：不多出零長度分段
    assert sp.cut_edges(3_200.0, 5_000.0).tolist() == [0.0, 3_200.0]  # 不足一段的小島整條算一段


def test_boundary_ring_starts_at_nw_corner_and_runs_clockwise(ring: sp.BoundaryRing) -> None:
    proj = DomainProjection(*CENTER)
    x0, y0 = ring.xy[0]
    lon, lat = proj.unproject(x0, y0)
    assert (round(float(lon), 6), round(float(lat), 6)) == (BBOX[0], BBOX[3])
    from shapely.geometry import LinearRing

    assert not LinearRing(ring.xy[:-1]).is_ccw  # 順時針
    c = ring.corner_s_m
    assert 0.0 == c["NW"] < c["NE"] < c["SE"] < c["SW"] < ring.length_m
    # 第二個角點是東北角：北界向東
    xs = np.interp(
        c["NE"], np.concatenate([[0], np.cumsum(np.hypot(*np.diff(ring.xy, axis=0).T))]), ring.xy[:, 0]
    )
    assert xs > x0
    assert 495_000.0 < ring.length_m < 505_000.0


def test_boundary_segments_cover_ring_with_partial_last(ring: sp.BoundaryRing) -> None:
    segs = sp.boundary_segments(ring, None)
    assert sum(s.length_m for s in segs) == pytest.approx(ring.length_m, rel=1e-9)
    assert all(s.length_m == pytest.approx(10_000.0, rel=1e-9) for s in segs[:-1])
    assert 0.0 < segs[-1].length_m <= 10_000.0 + 1e-6
    assert {s.group for s in segs} == {"N", "E", "S", "W"}


def test_old_arc_to_new_roundtrip_on_known_points(ring: sp.BoundaryRing) -> None:
    proj = DomainProjection(*CENTER)
    # 舊弧長自西南角逆時針：西南角＝0，西北角＝L − 西界長度
    west = ring.length_m - ring.corner_s_m["SW"]
    assert sp.old_arc_to_new(np.array([0.0]), ring)[0] == pytest.approx(ring.corner_s_m["SW"])  # 西南角
    assert sp.old_arc_to_new(np.array([ring.length_m - west]), ring)[0] == pytest.approx(
        0.0, abs=1e-6
    )  # 西北角
    # 南界上一點：舊弧長 5 km（自西南角向東）→ 新弧長 = SW 弧長 − 5 km
    new = sp.old_arc_to_new(np.array([5_000.0]), ring)[0]
    assert new == pytest.approx(ring.corner_s_m["SW"] - 5_000.0)
    x, y = proj.project(121.3 + 0.02, 24.6)
    s, d = sp.arc_positions(ring, np.array([x]), np.array([y]))
    assert d[0] < 1.0 and ring.corner_s_m["SE"] < s[0] < ring.corner_s_m["SW"]


def test_side_of_and_segment_index(ring: sp.BoundaryRing) -> None:
    c = ring.corner_s_m
    assert ring.side_of(np.array([1.0, c["NE"] + 1, c["SE"] + 1, c["SW"] + 1])).tolist() == [
        "N",
        "E",
        "S",
        "W",
    ]
    idx = sp.segment_index_boundary(ring, np.array([0.0, 9_999.0, 10_000.0, ring.length_m]))
    assert idx.tolist() == [0, 0, 1, len(sp.cut_edges(ring.length_m, 10_000.0)) - 2]


def test_orient_piece_closed_clockwise_from_northwest_and_open_from_northwest_end() -> None:
    ccw = LineString([(0, 0), (1000, 0), (1000, 1000), (0, 1000), (0, 0)])
    out, closed = sp.orient_piece(ccw)
    assert closed and out.coords[0] == (0.0, 1000.0)  # y−x 最大＝最西北頂點
    from shapely.geometry import LinearRing

    assert not LinearRing(out.coords[:-1]).is_ccw
    open_line = LineString([(0, 0), (500, 500), (2000, 3000)])  # 東北端 y−x=1000；西南端 0
    out2, closed2 = sp.orient_piece(open_line)
    assert not closed2 and out2.coords[0] == (2000.0, 3000.0)


def test_extract_coast_pieces_island_and_mainland_classification() -> None:
    proj = DomainProjection(*CENTER)
    big = box(121.0, 24.0, 121.6, 26.0)  # 面積 1.2 deg²（>1、<3）→ 福建沿岸（大陸）
    island = box(122.0, 25.0, 122.01, 25.01)  # 小島（周長約 4 km，未命名）
    pieces, land = sp.extract_coast_pieces([big, island], BBOX, proj)
    # 域內有「大陸」時，未命名且不在馬祖範圍的小島歸福建沿岸的近岸小島
    small = [p for p in pieces if p.name == "近岸小島"][0]
    assert small.cls == sp.CLASS_FUJIAN and small.closed and small.line.length < 5_000.0
    assert len(sp.coast_segments([small])) == 1  # 周長 < 5 km 的小島整條一段
    big_piece = [p for p in pieces if p.cls == sp.CLASS_FUJIAN and p.name == ""][0]
    assert not big_piece.closed  # 西邊落在域外，海岸線只剩域內的東緣
    xs = np.asarray(big_piece.line.coords)[:, 0]
    x_edge, _ = proj.project(121.6, 25.0)
    assert np.all(np.abs(xs - x_edge) < 2_000.0)  # 沒有域外裁切邊混進來
    assert land is not None
    # 沒有大陸時，未命名小島歸「其他小島」；命名比對用重心座標
    pieces2, _ = sp.extract_coast_pieces(
        [island, box(121.942, 24.832, 121.962, 24.852)], (121.3, 122.8, 24.6, 25.5), proj
    )
    assert sorted(p.cls for p in pieces2) == sorted([sp.CLASS_OTHER, sp.CLASS_NAMED])
    assert [p.name for p in pieces2 if p.cls == sp.CLASS_NAMED] == ["龜山島"]


def test_assign_nearest_and_distance_stats() -> None:
    a = sp.Segment("C1", "coast", LineString([(0, 0), (1000, 0)]), 1000.0, sp.CLASS_OTHER, "L1")
    b = sp.Segment("C2", "coast", LineString([(5000, 0), (6000, 0)]), 1000.0, sp.CLASS_OTHER, "L2")
    idx, d = sp.assign_nearest([a, b], np.array([500.0, 5500.0, 4000.0]), np.array([30.0, -40.0, 0.0]))
    assert idx.tolist() == [0, 1, 1]
    assert d.tolist() == pytest.approx([30.0, 40.0, 1000.0])
    st = sp.distance_stats(d)
    assert st["n"] == 3 and st["max_m"] == pytest.approx(1000.0) and st["n_gt_1km"] == 0


def test_counts_and_color_bins() -> None:
    idx = np.array([0, 0, 2, 2, 2, -1])
    assert sp.counts_by_segment(idx[idx >= 0], 4).tolist() == [2, 0, 3, 0]
    edges = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
    pct = np.array([0.0, 0.05, 0.1, 0.1001, 0.5, 1.0, 2.0, 5.0, 10.0, 10.01, 35.0])
    assert sp.bin_index(pct, edges).tolist() == [0, 1, 1, 2, 2, 3, 4, 5, 6, 7, 7]
    assert sp.bin_labels(edges) == ["0", "0–0.1", "0.1–0.5", "0.5–1", "1–2", "2–5", "5–10", ">10"]
