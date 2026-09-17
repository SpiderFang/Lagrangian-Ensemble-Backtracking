"""persistent-wet maximin 與四垂向 receptor 測試。"""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import box

from lagrangian_backtracking.mesh import NativeMesh
from lagrangian_backtracking.receptors import (
    build_vertical_targets,
    prepare_horizontal_receptor_candidates,
    select_horizontal_receptors,
    select_horizontal_receptors_from_coordinates,
    select_horizontal_receptors_from_pool,
)


def _five_face_mesh(face_count: int = 5) -> NativeMesh:
    """建立指定數量的互不重疊小 triangle，供 deterministic selector 測試。"""

    coordinates = []
    faces = []
    for face in range(face_count):
        x = float(face * 20)
        base = len(coordinates)
        coordinates.extend([[x, 0.0], [x + 5.0, 0.0], [x, 5.0]])
        faces.append([base, base + 1, base + 2, -1])
    xy = np.asarray(coordinates)
    node_count = xy.shape[0]
    return NativeMesh(
        node_lon=120.0 + xy[:, 0] * 1e-5,
        node_lat=24.0 + xy[:, 1] * 1e-5,
        node_xy=xy,
        source_depth_m=np.full(node_count, 20.0),
        source_node_bottom_index=np.zeros(node_count, dtype=np.int64),
        face_nodes_local=np.asarray(faces),
        face_node_count=np.full(face_count, 3),
        source_face_global_index=np.arange(100, 100 + face_count),
        bin_size_m=10.0,
    )


def test_horizontal_selector_requires_all_arrivals_wet() -> None:
    """任一 arrival 乾掉的 face 應被剔除，剩餘五面可重現選出。"""

    mesh = _five_face_mesh()
    wetdry = np.zeros((50, 5))
    wetdry[0, 2] = 1.0
    try:
        select_horizontal_receptors(
            study_site_id="test",
            mesh=mesh,
            candidate_polygon_metric=box(-10.0, -10.0, 100.0, 10.0),
            anchor_xy=(0.0, 0.0),
            wetdry_at_arrivals=wetdry,
            count=5,
        )
    except ValueError as error:
        assert "候選不足" in str(error)
    else:
        raise AssertionError("乾 face 被錯誤納入 persistent-wet candidates")
    wetdry[:, 2] = 0.0
    receptors = select_horizontal_receptors(
        study_site_id="test",
        mesh=mesh,
        candidate_polygon_metric=box(-10.0, -10.0, 100.0, 10.0),
        anchor_xy=(0.0, 0.0),
        wetdry_at_arrivals=wetdry,
        count=5,
    )
    assert len(receptors) == 5
    assert receptors[0].source_face_global_index == 100


def test_wrapper_matches_prepare_and_pool_selection_without_exclusion() -> None:
    """無 exclusion 時，新兩階段 API 必須與既有 wrapper 完全同結果。"""

    mesh = _five_face_mesh()
    wetdry = np.zeros((50, 5))
    polygon = box(-10.0, -10.0, 100.0, 10.0)
    wrapper_result = select_horizontal_receptors(
        study_site_id="test",
        mesh=mesh,
        candidate_polygon_metric=polygon,
        anchor_xy=(0.0, 0.0),
        wetdry_at_arrivals=wetdry,
        count=5,
    )
    pool = prepare_horizontal_receptor_candidates(
        study_site_id="test",
        mesh=mesh,
        candidate_polygon_metric=polygon,
        anchor_xy=(0.0, 0.0),
        wetdry_at_arrivals=wetdry,
    )
    pool_result = select_horizontal_receptors_from_pool(pool, count=5)
    assert wrapper_result == pool_result


def test_pool_exclusion_does_not_mutate_candidates_and_reselects_deterministically() -> None:
    """排除 face 後不得回傳該 face，且 pool 的唯讀內容與結果順序保持穩定。"""

    mesh = _five_face_mesh(face_count=7)
    wetdry = np.zeros((50, 7))
    pool = prepare_horizontal_receptor_candidates(
        study_site_id="test",
        mesh=mesh,
        candidate_polygon_metric=box(-10.0, -10.0, 200.0, 10.0),
        anchor_xy=(0.0, 0.0),
        wetdry_at_arrivals=wetdry,
    )
    before = {
        "local": np.array(pool.candidate_face_local_indices, copy=True),
        "global": np.array(pool.candidate_face_global_indices, copy=True),
        "xy": np.array(pool.candidate_xy_m, copy=True),
        "lon": np.array(pool.candidate_lon, copy=True),
        "lat": np.array(pool.candidate_lat, copy=True),
    }
    selected = select_horizontal_receptors_from_pool(
        pool,
        count=5,
        excluded_face_indices=(0,),
    )
    assert all(item.source_face_local_index != 0 for item in selected)
    assert tuple(item.source_face_local_index for item in selected) == tuple(
        item.source_face_local_index
        for item in select_horizontal_receptors_from_pool(
            pool,
            count=5,
            excluded_face_indices=(0,),
        )
    )
    assert np.array_equal(pool.candidate_face_local_indices, before["local"])
    assert np.array_equal(pool.candidate_face_global_indices, before["global"])
    assert np.array_equal(pool.candidate_xy_m, before["xy"])
    assert np.array_equal(pool.candidate_lon, before["lon"])
    assert np.array_equal(pool.candidate_lat, before["lat"])
    assert not pool.candidate_face_local_indices.flags.writeable
    assert not pool.candidate_face_global_indices.flags.writeable
    assert not pool.candidate_xy_m.flags.writeable
    assert not pool.candidate_lon.flags.writeable
    assert not pool.candidate_lat.flags.writeable


def test_pool_selection_fails_closed_when_exclusion_leaves_fewer_than_five() -> None:
    """排除後共同有效候選不足五面時，pool selector 必須明確 ValueError。"""

    mesh = _five_face_mesh()
    pool = prepare_horizontal_receptor_candidates(
        study_site_id="test",
        mesh=mesh,
        candidate_polygon_metric=box(-10.0, -10.0, 100.0, 10.0),
        anchor_xy=(0.0, 0.0),
        wetdry_at_arrivals=np.zeros((50, 5)),
    )
    with pytest.raises(ValueError, match="候選不足"):
        select_horizontal_receptors_from_pool(pool, count=5, excluded_face_indices=(0,))


def test_fixed_coordinates_preserve_declared_order_and_mesh_faces() -> None:
    """固定點 selector 應逐點映射同一 mesh，且不以 maximin 改變宣告順序。"""

    mesh = _five_face_mesh()
    pool = prepare_horizontal_receptor_candidates(
        study_site_id="hsinchu",
        mesh=mesh,
        candidate_polygon_metric=box(-10.0, -10.0, 200.0, 10.0),
        anchor_xy=(0.0, 0.0),
        wetdry_at_arrivals=np.zeros((50, 5)),
    )
    coordinates_lonlat = [
        (float(pool.candidate_lon[index]), float(pool.candidate_lat[index]))
        for index in (3, 0, 4, 1, 2)
    ]
    coordinates_xy = [
        tuple(float(value) for value in pool.candidate_xy_m[index])
        for index in (3, 0, 4, 1, 2)
    ]
    selected = select_horizontal_receptors_from_coordinates(
        pool,
        coordinates_lonlat=coordinates_lonlat,
        coordinates_xy=coordinates_xy,
    )
    assert [item.source_face_local_index for item in selected] == [3, 0, 4, 1, 2]


@pytest.mark.parametrize(
    "coordinates_factory",
    [
        lambda pool: [
            tuple(float(value) for value in pool.candidate_xy_m[index])
            for index in (0, 0, 2, 3, 4)
        ],
        lambda pool: [
            tuple(float(value) for value in pool.candidate_xy_m[index])
            for index in (0, 1, 2, 3, 4)
        ],
    ],
    ids=["duplicate-face", "tampered-distance"],
)
def test_fixed_coordinates_fail_closed_on_duplicate_or_tamper(coordinates_factory) -> None:
    """固定點重複映射或座標偏離候選 face 時必須停止，不得 fallback。"""

    mesh = _five_face_mesh()
    pool = prepare_horizontal_receptor_candidates(
        study_site_id="hsinchu",
        mesh=mesh,
        candidate_polygon_metric=box(-10.0, -10.0, 200.0, 10.0),
        anchor_xy=(0.0, 0.0),
        wetdry_at_arrivals=np.zeros((50, 5)),
    )
    coordinates_lonlat = [
        (float(pool.candidate_lon[index]), float(pool.candidate_lat[index]))
        for index in range(5)
    ]
    coordinates_xy = coordinates_factory(pool)
    if coordinates_xy == [tuple(float(value) for value in pool.candidate_xy_m[index]) for index in range(5)]:
        coordinates_xy[1] = (coordinates_xy[1][0] + 2.0, coordinates_xy[1][1])
    with pytest.raises(ValueError, match="固定受體"):
        select_horizontal_receptors_from_coordinates(
            pool,
            coordinates_lonlat=coordinates_lonlat,
            coordinates_xy=coordinates_xy,
        )


def test_vertical_targets_are_positive_up_and_distinct() -> None:
    """10/40/70% 與 near-bed 應落在水柱內並形成四個不同 z。"""

    targets = build_vertical_targets(
        surface_z_m=1.0,
        bed_z_m=-19.0,
        valid_layer_z_m=np.array([-19.0, -15.0, -10.0, -5.0, 0.0, 1.0]),
    )
    assert [item.vertical_id for item in targets] == [
        "upper_water_column",
        "mid_upper_water_column",
        "mid_lower_water_column",
        "near_bed",
    ]
    assert [item.z_m_positive_up for item in targets[:3]] == [-1.0, -7.0, -13.0]
    assert len({item.z_m_positive_up for item in targets}) == 4
