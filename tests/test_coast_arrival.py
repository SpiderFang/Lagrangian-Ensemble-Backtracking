"""WP-72 方法 B（首次抵達海岸）的單元測試：合成網格、合成結果，不讀任何正式資料。"""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import box

from lagrangian_backtracking.native_runner.backward_driver import (
    NATIVE_STATUS_LEFT_FLOW_DOMAIN,
    NATIVE_STATUS_MAX_BACKTRACK,
    NATIVE_STATUS_REACHED_COAST,
    BackwardDriverResult,
    DepartureRecord,
    HourlyOverheadDiagnostics,
    LocalDomainFirstExitRecord,
    SubstepDiagnostics,
)
from lagrangian_backtracking.native_runner.coast_arrival import (
    CoastArrivalError,
    CoastArrivalOverride,
    apply_coast_arrival,
    coast_geometry_from_mesh,
    find_first_coast_row,
)


def _grid(n: int = 6, h: float = 1000.0):
    """n×n 個正方形格，每格切兩個三角形；節點座標 0..n*h（公尺）。"""
    xs, ys = np.meshgrid(np.arange(n + 1) * h, np.arange(n + 1) * h)
    x, y = xs.ravel(), ys.ravel()
    tri = []
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            b, c, d = a + 1, a + (n + 1), a + (n + 1) + 1
            tri += [[a, b, d], [a, d, c]]
    return x, y, np.array(tri)


def _result(tx, ty, ts, deps, local=()):
    t, n = ts.shape
    return BackwardDriverResult(
        row_particle_index=np.arange(100, 100 + n),
        final_status=np.array([d.status for d in sorted(deps, key=lambda r: r.row_particle_index)], dtype=np.int32),
        final_x_m=np.array([d.x_m for d in sorted(deps, key=lambda r: r.row_particle_index)]),
        final_y_m=np.array([d.y_m for d in sorted(deps, key=lambda r: r.row_particle_index)]),
        final_z_m=np.zeros(n),
        final_utc_ns=np.array([d.utc_ns for d in sorted(deps, key=lambda r: r.row_particle_index)], dtype=np.int64),
        trajectory_utc_ns=(10**18 - np.arange(t, dtype=np.int64) * 3_600_000_000_000),
        trajectory_x_m=tx,
        trajectory_y_m=ty,
        trajectory_z_m=np.zeros((t, n)),
        trajectory_status=ts,
        departures=tuple(deps),
        diagnostics=SubstepDiagnostics(),
        overhead_diagnostics=HourlyOverheadDiagnostics(),
        hours_run=t - 1,
        local_domain_departures=tuple(local),
    )


def test_coast_edges_only_inside_flow_domain():
    x, y, tri = _grid()
    # 流場範圍只含 x<=5000：右緣（x=6000）的網格外緣不算海岸；其餘三面算
    g = coast_geometry_from_mesh(x, y, tri, box(-10, -10, 5600, 6010))
    assert g.n_boundary_edges == 24
    assert g.n_coast_edges == 18
    # 內部元素（離邊界一格以上）不是海岸元素；最外一圈靠左、上、下的是；右緣最外圈不是
    el_at = lambda px, py: int(g.finder(np.array([px]), np.array([py]))[0])  # noqa: E731
    assert g.coast_element[el_at(500.0, 500.0)]
    assert not g.coast_element[el_at(2500.0, 2500.0)]
    assert not g.coast_element[el_at(5500.0, 2500.0)]


def test_first_row_and_override_semantics():
    x, y, tri = _grid()
    g = coast_geometry_from_mesh(x, y, tri, box(-10, -10, 6010, 6010))
    t, n = 8, 3
    tx = np.full((t, n), 3200.0)
    ty = np.full((t, n), 3200.0)
    ts = np.zeros((t, n), dtype=np.int32)
    # 粒子 0：第 3 列抵達左緣；之後本來還會穿越（狀態 1 自第 6 列）
    tx[3, 0], ty[3, 0] = 100.0, 3800.0
    tx[4:, 0], ty[4:, 0] = 2000.0, 2000.0
    ts[6:, 0] = NATIVE_STATUS_LEFT_FLOW_DOMAIN
    # 粒子 1：全程在內部，回溯到上限
    # 粒子 2：第 0 列就在海岸元素內（貼岸受體）照實計入
    tx[0, 2], ty[0, 2] = 100.0, 300.0
    deps = [
        DepartureRecord(0, NATIVE_STATUS_LEFT_FLOW_DOMAIN, 9.0, 9.0, 0.0, 111, also_local_domain_first_exit=True),
        DepartureRecord(1, NATIVE_STATUS_MAX_BACKTRACK, 3200.0, 3200.0, 0.0, 222),
        DepartureRecord(2, NATIVE_STATUS_MAX_BACKTRACK, 3200.0, 3200.0, 0.0, 333),
    ]
    local = [LocalDomainFirstExitRecord(0, 1.0, 1.0, 0.0, 10**18 - 5 * 3_600_000_000_000),
             LocalDomainFirstExitRecord(1, 1.0, 1.0, 0.0, 10**18 - 5 * 3_600_000_000_000)]
    res = _result(tx, ty, ts, deps, local)
    first, outside = find_first_coast_row(tx, ty, ts, g)
    assert outside == 0
    assert first.tolist() == [3, -1, 0]

    new = apply_coast_arrival(res, first)
    assert new.final_status.tolist() == [NATIVE_STATUS_REACHED_COAST, NATIVE_STATUS_MAX_BACKTRACK, NATIVE_STATUS_REACHED_COAST]
    # 抵達列起凍結並標狀態 8，不計停留（status != 0）
    assert (new.trajectory_status[3:, 0] == NATIVE_STATUS_REACHED_COAST).all()
    assert (new.trajectory_status[:3, 0] == 0).all()
    assert (new.trajectory_x_m[3:, 0] == 100.0).all()
    assert new.final_x_m[0] == 100.0 and new.final_utc_ns[0] == int(res.trajectory_utc_ns[3])
    # 離開紀錄被覆寫，穿越旗標清掉；未抵達者不變
    d0 = [d for d in new.departures if d.row_particle_index == 0][0]
    assert d0.status == NATIVE_STATUS_REACHED_COAST and not d0.also_local_domain_first_exit and d0.x_m == 100.0
    assert [d for d in new.departures if d.row_particle_index == 1][0].status == NATIVE_STATUS_MAX_BACKTRACK
    # 貼岸受體：第 0 列即抵達，整條軌跡都是非追蹤狀態
    assert (new.trajectory_status[:, 2] == NATIVE_STATUS_REACHED_COAST).all()
    # 抵達（第 3 列）之後才發生的 local 首次離開（第 5 列）刪除；沒抵達者的保留
    assert [r.row_particle_index for r in new.local_domain_departures] == [1]
    # 原結果不被修改
    assert res.final_status[0] == NATIVE_STATUS_LEFT_FLOW_DOMAIN and res.trajectory_x_m[4, 0] == 2000.0


def test_no_hits_returns_same_object_and_override_summary():
    x, y, tri = _grid()
    g = coast_geometry_from_mesh(x, y, tri, box(-10, -10, 6010, 6010))
    t, n = 4, 2
    tx = np.full((t, n), 3000.0)
    ty = np.full((t, n), 3000.0)
    ts = np.zeros((t, n), dtype=np.int32)
    deps = [DepartureRecord(i, NATIVE_STATUS_MAX_BACKTRACK, 3000.0, 3000.0, 0.0, 5) for i in range(n)]
    res = _result(tx, ty, ts, deps)
    ov = CoastArrivalOverride({"s": g})
    out, hit = ov.apply("s", res)
    assert out is res and (hit == -1).all()
    assert ov.summary()["s"]["reached_coast"] == 0 and ov.summary()["s"]["particles"] == 2
    with pytest.raises(CoastArrivalError):
        ov.apply("missing", res)


def test_non_triangle_mesh_rejected():
    with pytest.raises(CoastArrivalError):
        coast_geometry_from_mesh(np.zeros(4), np.zeros(4), np.zeros((1, 4), dtype=int), box(0, 0, 1, 1))
