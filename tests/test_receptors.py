"""persistent-wet maximin 與四垂向 receptor 測試。"""

from __future__ import annotations

import numpy as np
from shapely.geometry import box

from lagrangian_backtracking.mesh import NativeMesh
from lagrangian_backtracking.receptors import build_vertical_targets, select_horizontal_receptors


def _five_face_mesh() -> NativeMesh:
    """建立五個互不重疊的小 triangle，供 deterministic selector 測試。"""

    coordinates = []
    faces = []
    for face in range(5):
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
        face_node_count=np.full(5, 3),
        source_face_global_index=np.arange(100, 105),
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
