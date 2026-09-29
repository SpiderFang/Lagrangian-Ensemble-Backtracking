"""WP-38：停留時數密度、HDR、回溯快照、邊界弧長分布與等深線的單元測試。

全部使用合成資料（``tests/wp38_fixture.py``），不讀任何正式運算輸出。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest
import wp38_fixture as fx
from shapely.geometry import LineString, box

from lagrangian_backtracking.native_runner import bathymetry, footprint
from lagrangian_backtracking.native_runner import postprocess as pp
from lagrangian_backtracking.native_runner.footprint import (
    FootprintAccumulator,
    FootprintConfig,
    FootprintError,
    FootprintSiteSpec,
)

# ── 既有產品的回歸基準（由 WP-38 之前的程式碼 03f9226 產生）──────────────────

# fmt: off
GOLDEN_LEGACY_DIGESTS = {
    "boundary_arc_segments_geometry_by_site.parquet": (
        "9f3d5fd986d13e6297046507d6fe07ba0eb2a6753a9cd1efa0a32a40552b653f"
    ),
    "boundary_entry_by_site.parquet": (
        "050055ce51f97db93445c8062288aec58b6819ef87ad5ac46f523152ec065f57"
    ),
    "boundary_entry_by_site_arc_segment.parquet": (
        "e97ccfa4b01946bbcc7bd7f46a22a4ea03971bd93c5e2747d06ee515c8b52477"
    ),
    "boundary_entry_by_site_azimuth_sector.parquet": (
        "18072f1f121785ec16c9022dfd84398745e2f777c2f65d968869004b027c301b"
    ),
    "boundary_entry_by_site_settling_velocity.parquet": (
        "142b8d30bef01a108e2c40d288131492818be8d5d49304688f1f1998ad065790"
    ),
    "boundary_entry_by_site_settling_velocity_arc_segment.parquet": (
        "54a58181494fadd501f93c977a6167a7516ea74b6922859572f373132c7d6a17"
    ),
    "boundary_entry_by_site_settling_velocity_azimuth_sector.parquet": (
        "1d4ea7d307faa2683084c8f4bc0a760734fb8a8712f19c64070ca15da20e24d7"
    ),
    "boundary_flux_by_depth_quartile.parquet": (
        "464c92bcf1af2179a53a10577a2e03cace72607cdae6103e1d2a84a65c021795"
    ),
    "boundary_flux_by_season.parquet": (
        "db9c70716ff5f7cd717c4f3c8f5d71d1a5ee40b371388824291136113295c8fc"
    ),
    "boundary_flux_by_settling_velocity.parquet": (
        "fa54c3b015d0a16e59e4fb753be3b47033b20a1e29a0ca7b5822c4260c0e2617"
    ),
    "boundary_flux_by_site.parquet": (
        "b52a29a6ed7de9039b509f2e75912daf931d264fb272835d17b7721649c3e106"
    ),
    "boundary_sectors_geometry_by_site.parquet": (
        "372a46fdd3efd351adfe393d6d1f91c30d96ad799f5bc0741bbf75e0a7be1c7a"
    ),
    "boundary_summary_by_site.parquet": (
        "b21fdb64823215a13b079f918013873944f44075bb00a8273d91a27cb574f844"
    ),
    "departures.parquet": (
        "de64fb15824d105d48bd6ac39f02d8a4a6a6b0eb8c238d0080170f5d3d2c754e"
    ),
    "kde_by_site.parquet": (
        "709e61880b0295f4a03f43d4d9e23a64f7943e58db969621d6f35a79a86a0532"
    ),
    "kde_by_site_settling_velocity.parquet": (
        "1cdf67ebc206d031d974ef721ed1f2a25cf97d94461e849339e53133f68b683d"
    ),
    "local_domain_departures.parquet": (
        "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
    ),
    "local_domain_first_exit_summary_by_site.parquet": (
        "1d29c242917f5e2b55eebc774e377cf622dc15520cca5f5d53f26ce92702d457"
    ),
    "trajectory_samples_by_site.parquet": (
        "0a51cd8a67bff18bd1f4558132e428d18a724db2ba9e382ff209b1462ae18dbb"
    ),
    "trajectory_samples_by_site_settling_velocity.parquet": (
        "35f2521dbd31a50c5b8fb89d33d7aec609fe9c7678baba1807b90c20d6d72baa"
    ),
}
# fmt: on

# ── 共用輔助 ─────────────────────────────────────────────────────────────

SITE = "siteA"
# 100 km（東西）× 80 km（南北）的矩形 flow domain，開放邊界依序是南邊、東邊、北邊。
DOMAIN = box(-50e3, -40e3, 50e3, 40e3)
OPEN_BOUNDARY = LineString([(-50e3, -40e3), (50e3, -40e3), (50e3, 40e3), (-50e3, 40e3)])
SPEC = FootprintSiteSpec(flow_domain=DOMAIN, flow_open_boundary=OPEN_BOUNDARY)
RECEPTORS = ((500.0, 500.0), (1000.0, 700.0))  # 同一個 0.02 度網格內（離格線超過查表格距）


def make_accumulator(
    identity, *, config: FootprintConfig | None = None, bathymetry_by_site=None, land_lonlat=None
):
    return FootprintAccumulator(
        identity,
        config=config or FootprintConfig(),
        specs={SITE: SPEC},
        lonlat_projector=fx.linear_projector,
        bathymetry_by_site=bathymetry_by_site,
        land_lonlat=land_lonlat,
        land_source={"file_name": "synthetic"} if land_lonlat is not None else None,
    )


def finalize(acc: FootprintAccumulator):
    return acc.finalize(receptor_centers_xy={SITE: RECEPTORS[0]}, physical_case_id="case-test")


def table_rows(products, name: str, **where: Any) -> list[dict[str, Any]]:
    rows = products.tables[name].to_pylist()
    return [r for r in rows if all(r.get(k) == v for k, v in where.items())]


def static_result(identity, rows, *, n_time: int, active_until: int | None = None, far_xy=(30e3, 20e3)):
    """粒子停在受體位置；``active_until`` 之後改為終止（status=3）並凍結在遠處位置。"""

    n = rows.size
    x = np.repeat(identity.x0_m[rows][None, :], n_time, axis=0).astype(np.float64)
    y = np.repeat(identity.y0_m[rows][None, :], n_time, axis=0).astype(np.float64)
    status = np.zeros((n_time, n), dtype=np.int32)
    if active_until is not None:
        x[active_until:] = far_xy[0]
        y[active_until:] = far_xy[1]
        status[active_until:] = 3
    return fx.make_result(identity, rows, x, y, status)


def walk_world(*, members: int = 8, arrivals: int = 6):
    return fx.build_world(
        site=SITE,
        velocities=(-0.001, -0.002),
        members=members,
        receptors=2,
        arrivals=arrivals,
        receptor_xy=RECEPTORS,
    )


def walk_results(
    seed: int = 7, *, members: int = 8, arrivals: int = 6, n_time: int = 40, step_m: float = 3000.0
):
    identity, units = walk_world(members=members, arrivals=arrivals)
    rng = np.random.default_rng(seed)
    results = [
        (key, rows, fx.random_walk_unit(rng, identity, rows, n_time=n_time, step_m=step_m, east_edge_m=50e3))
        for key, rows in units
    ]
    return identity, results


def run_accumulator(
    identity, batches, *, config: FootprintConfig | None = None, bathymetry_by_site=None, land_lonlat=None
):
    acc = make_accumulator(
        identity, config=config, bathymetry_by_site=bathymetry_by_site, land_lonlat=land_lonlat
    )
    for key, rows, result in batches:
        acc.add_batch(key, rows, result)
    return finalize(acc)


def concat_results(identity, batches):
    rows = np.concatenate([r for _, r, _ in batches])
    x = np.concatenate([res.trajectory_x_m for _, _, res in batches], axis=1)
    y = np.concatenate([res.trajectory_y_m for _, _, res in batches], axis=1)
    s = np.concatenate([res.trajectory_status for _, _, res in batches], axis=1)
    return rows, fx.make_result(identity, rows, x, y, s)


# ── 純函式 ───────────────────────────────────────────────────────────────


def test_cell_area_weights_follow_cos_latitude():
    area = footprint.cell_area_km2_by_row(np.array([0.0, 60.0, 25.0]), 0.02, 0.02)
    expected0 = footprint.EARTH_RADIUS_KM**2 * math.radians(0.02) ** 2
    assert area[0] == pytest.approx(expected0, rel=1e-12)
    assert area[1] == pytest.approx(0.5 * expected0, rel=1e-12)
    assert area[2] == pytest.approx(expected0 * math.cos(math.radians(25.0)), rel=1e-12)
    # 0.02 度在赤道約 2.22 km，面積約 4.9 km²。
    assert 4.8 < area[0] < 5.1


def test_make_grid_is_nested_and_covers_range():
    coarse = footprint.make_grid(120.13, 121.47, 24.61, 25.29, resolution_deg=0.04, align_deg=0.04)
    fine = footprint.make_grid(120.13, 121.47, 24.61, 25.29, resolution_deg=0.01, align_deg=0.04)
    assert coarse.lon0 == fine.lon0 and coarse.lat0 == fine.lat0
    for grid in (coarse, fine):
        assert grid.lon0 <= 120.13 and grid.lat0 <= 24.61
        assert grid.lon0 + grid.nx * grid.resolution_deg >= 121.47 - 1e-9
        assert grid.lat0 + grid.ny * grid.resolution_deg >= 25.29 - 1e-9


def test_flat_index_marks_outside_as_minus_one():
    grid = footprint.FootprintGrid(lon0=120.0, lat0=24.0, resolution_deg=0.5, nx=4, ny=2)
    idx = grid.flat_index(
        np.array([120.1, 121.6, 119.9, 122.1, 120.1]), np.array([24.1, 24.7, 24.1, 24.1, 25.1])
    )
    assert idx.tolist() == [0, 4 + 3, -1, -1, -1]


def test_compute_hdr_orders_by_density_and_uses_integer_counts():
    hours = np.array([50, 30, 15, 5, 0])
    area = np.ones(5)
    hdr = footprint.compute_hdr(hours, area, [0.5, 0.75, 0.9])
    assert [h["cell_count"] for h in hdr] == [1, 2, 3]
    assert [h["area_km2"] for h in hdr] == [1.0, 2.0, 3.0]
    assert hdr[0]["achieved_fraction"] == pytest.approx(0.5)
    # 密度而非時數決定順序：時數相同、面積小的格子密度較高。
    hdr2 = footprint.compute_hdr(np.array([10, 10]), np.array([1.0, 10.0]), [0.4])
    assert hdr2[0]["cell_indices"].tolist() == [0]
    assert hdr2[0]["density_threshold_fraction_per_km2"] == pytest.approx(0.5)
    assert footprint.compute_hdr(np.zeros(3, dtype=int), np.ones(3), [0.5]) == []


def test_total_variation_distance_basic_cases():
    assert footprint.total_variation_distance(np.array([1, 1]), np.array([2, 2])) == 0.0
    assert footprint.total_variation_distance(np.array([1, 0]), np.array([0, 5])) == 1.0
    assert footprint.total_variation_distance(np.array([0, 0]), np.array([1, 1])) is None


def test_haversine_and_bearing():
    d, b = footprint.haversine_km_and_bearing_deg(121.0, 25.0, 121.0, 26.0)
    assert d == pytest.approx(111.195, abs=0.05)
    assert b == pytest.approx(0.0, abs=1e-9)
    d, b = footprint.haversine_km_and_bearing_deg(121.0, 0.0, 122.0, 0.0)
    assert b == pytest.approx(90.0, abs=1e-9)
    _, b = footprint.haversine_km_and_bearing_deg(121.0, 25.0, 120.0, 25.0)
    assert 265.0 < b < 275.0


def test_config_validation_fail_closed():
    with pytest.raises(FootprintError):
        FootprintConfig(grid_resolution_deg=0.0)
    with pytest.raises(FootprintError):
        FootprintConfig(hdr_levels=(0.9, 0.5))
    with pytest.raises(FootprintError):
        FootprintConfig(arc_bin_km=-1.0)
    assert FootprintConfig().hdr_levels == (0.5, 0.75, 0.9)


# ── 停留時數：正規化、面積、凍結位置 ───────────────────────────────────────


def test_static_particles_normalisation_and_area_weight():
    identity, units = walk_world(members=4, arrivals=2)
    n_time = 5
    batches = [(k, rows, static_result(identity, rows, n_time=n_time)) for k, rows in units]
    products = run_accumulator(identity, batches)
    rows = table_rows(products, "residence_density_by_site", group_kind="all")
    assert len(rows) == 1  # 全部停在同一個網格
    total = len(identity) * n_time
    assert rows[0]["particle_hours"] == total
    assert rows[0]["fraction_of_total"] == 1.0
    lat_center = rows[0]["lat_center_deg"]
    expected_area = (
        footprint.EARTH_RADIUS_KM**2 * math.radians(0.02) ** 2 * math.cos(math.radians(lat_center))
    )
    assert rows[0]["cell_area_km2"] == pytest.approx(expected_area, rel=1e-12)
    assert rows[0]["density_fraction_per_km2"] * rows[0]["cell_area_km2"] == pytest.approx(1.0, abs=1e-12)
    site_manifest = products.manifest["sites"][SITE]
    assert site_manifest["total_particle_hours"] == total
    assert abs(site_manifest["normalized_sum"] - 1.0) <= 1e-9


def test_normalised_sum_is_one_for_random_walk_and_every_group():
    identity, results = walk_results()
    products = run_accumulator(identity, results)
    for name in (
        "residence_density_by_site",
        "residence_density_by_site_settling_velocity",
        "residence_density_by_site_member_half",
    ):
        table = products.tables[name].to_pylist()
        keys = {(r["group_kind"], r["group_value"]) for r in table}
        assert keys
        for kind, value in keys:
            sel = [r for r in table if (r["group_kind"], r["group_value"]) == (kind, value)]
            assert sum(r["fraction_of_total"] for r in sel) == pytest.approx(1.0, abs=1e-9)
            assert sum(r["density_fraction_per_km2"] * r["cell_area_km2"] for r in sel) == pytest.approx(
                1.0, abs=1e-9
            )
    assert (
        len(
            {
                r["group_value"]
                for r in products.tables["residence_density_by_site_settling_velocity"].to_pylist()
            }
        )
        == 2
    )
    halves = {r["group_value"] for r in products.tables["residence_density_by_site_member_half"].to_pylist()}
    assert halves == {"first", "second"}


def test_positions_frozen_after_termination_are_not_counted():
    identity, units = walk_world(members=4, arrivals=2)
    n_time = 12
    # 全部粒子只在 t=0、1 仍在漂流，之後終止並凍結在 30 km 外。
    batches = [(k, rows, static_result(identity, rows, n_time=n_time, active_until=2)) for k, rows in units]
    products = run_accumulator(identity, batches)
    rows = table_rows(products, "residence_density_by_site", group_kind="all")
    assert sum(r["particle_hours"] for r in rows) == len(identity) * 2
    lon_far, lat_far = fx.linear_projector(np.array([SITE]), np.array([30e3]), np.array([20e3]))
    for r in rows:
        assert abs(r["lon_center_deg"] - lon_far[0]) > 0.02 or abs(r["lat_center_deg"] - lat_far[0]) > 0.02


def test_arrival_sample_switch():
    identity, units = walk_world(members=4, arrivals=2)
    batches = [(k, rows, static_result(identity, rows, n_time=6, active_until=3)) for k, rows in units]
    with_arrival = run_accumulator(identity, batches)
    without = run_accumulator(identity, batches, config=FootprintConfig(count_arrival_sample=False))
    n = len(identity)
    assert with_arrival.manifest["sites"][SITE]["total_particle_hours"] == 3 * n
    assert without.manifest["sites"][SITE]["total_particle_hours"] == 2 * n
    assert with_arrival.manifest["arrival_sample_included"] is True
    assert without.manifest["arrival_sample_included"] is False


def test_no_valid_hours_fails_closed():
    identity, units = walk_world(members=4, arrivals=1)
    batches = [(k, rows, static_result(identity, rows, n_time=3, active_until=1)) for k, rows in units]
    with pytest.raises(FootprintError):
        run_accumulator(identity, batches, config=FootprintConfig(count_arrival_sample=False))


def test_normalisation_failure_is_fail_closed(monkeypatch):
    identity, units = walk_world(members=4, arrivals=1)
    batches = [(k, rows, static_result(identity, rows, n_time=3)) for k, rows in units]
    monkeypatch.setattr(footprint, "NORMALIZED_SUM_TOLERANCE", -1.0)
    with pytest.raises(FootprintError, match="正規化"):
        run_accumulator(identity, batches)


def test_missing_site_geometry_fails_closed():
    identity, _ = walk_world(members=4, arrivals=1)
    with pytest.raises(FootprintError):
        FootprintAccumulator(
            identity, config=FootprintConfig(), specs={}, lonlat_projector=fx.linear_projector
        )


# ── HDR、成員分半、雜訊底 ───────────────────────────────────────────────────


def test_hdr_areas_increase_monotonically():
    identity, results = walk_results()
    products = run_accumulator(identity, results)
    hdr = sorted(table_rows(products, "residence_hdr_by_site", group_kind="all"), key=lambda r: r["level"])
    assert [r["level"] for r in hdr] == [0.5, 0.75, 0.9]
    assert hdr[0]["area_km2"] < hdr[1]["area_km2"] < hdr[2]["area_km2"]
    assert hdr[0]["density_threshold_fraction_per_km2"] >= hdr[1]["density_threshold_fraction_per_km2"]
    assert hdr[1]["density_threshold_fraction_per_km2"] >= hdr[2]["density_threshold_fraction_per_km2"]
    for r in hdr:
        assert r["achieved_fraction"] >= r["level"] - 1e-12
    summary = products.summary_rows[0]
    assert summary["hdr50_area_km2"] < summary["hdr75_area_km2"] < summary["hdr90_area_km2"]
    assert summary["hdr50_area_km2"] == pytest.approx(hdr[0]["area_km2"])


def _iid_batches(seed: int, *, shift_second_half_m: float, members: int = 8, arrivals: int = 30):
    """所有成員同分布；可選擇讓後半成員整體平移，製造真實差異。"""

    identity, units = fx.build_world(
        site=SITE,
        velocities=(-0.001,),
        members=members,
        receptors=2,
        arrivals=arrivals,
        receptor_xy=((0.0, 0.0), (0.0, 0.0)),
    )
    rng = np.random.default_rng(seed)
    batches = []
    for key, rows in units:
        res = fx.random_walk_unit(rng, identity, rows, n_time=30, step_m=1000.0, terminate=False)
        x = res.trajectory_x_m.copy()
        if shift_second_half_m:
            x[:, identity.member_id[rows] >= members // 2] += shift_second_half_m
        batches.append(
            (key, rows, fx.make_result(identity, rows, x, res.trajectory_y_m, res.trajectory_status))
        )
    return identity, batches


def test_member_half_tvd_is_near_noise_floor_for_same_distribution():
    identity, batches = _iid_batches(11, shift_second_half_m=0.0)
    s = run_accumulator(identity, batches).summary_rows[0]
    assert s["member_half_tvd"] is not None
    assert s["noise_tvd_member_partition_mean"] > 0.0
    # 同分布時，前後半的 TVD 與「隨機均分成員」的雜訊底同量級。
    assert s["member_half_tvd"] <= 1.5 * s["noise_tvd_member_partition_p95"]
    assert s["member_half_tvd"] < 3.0 * s["noise_tvd_member_partition_mean"]


def test_member_half_tvd_detects_real_difference():
    identity, batches = _iid_batches(11, shift_second_half_m=30e3)
    s = run_accumulator(identity, batches).summary_rows[0]
    assert s["member_half_tvd"] > s["noise_tvd_member_partition_p95"]
    assert s["member_half_tvd"] > 0.9
    assert s["member_half_hdr50_area_relative_difference"] is not None


def test_summary_columns_and_grid_sensitivity():
    identity, results = walk_results()
    s = run_accumulator(identity, results).summary_rows[0]
    for col in (
        "hdr50_area_km2",
        "hdr75_area_km2",
        "hdr90_area_km2",
        "centroid_lon_deg",
        "centroid_lat_deg",
        "centroid_distance_from_receptor_km",
        "centroid_bearing_from_receptor_deg",
        "hours_fraction_depth_below_threshold",
        "active_fraction_72h",
        "active_fraction_168h",
        "active_fraction_336h",
        "active_fraction_720h",
        "member_half_tvd",
        "member_half_hdr50_area_relative_difference",
        "noise_tvd_member_partition_mean",
        "noise_tvd_unit_split_mean",
        "grid_sensitivity_hdr50_area_km2_0p01",
        "grid_sensitivity_hdr50_area_km2_0p02",
        "grid_sensitivity_hdr50_area_km2_0p04",
        "grid_sensitivity_hdr90_area_km2_0p01",
        "grid_sensitivity_hdr90_area_km2_0p02",
        "grid_sensitivity_hdr90_area_km2_0p04",
    ):
        assert col in s, col
    # 主解析度的敏感度欄位與主表 HDR 一致。
    assert s["grid_sensitivity_hdr50_area_km2_0p02"] == s["hdr50_area_km2"]
    assert s["grid_sensitivity_hdr90_area_km2_0p02"] == s["hdr90_area_km2"]
    # 細網格的 HDR 面積不大於粗網格太多（面積量級相同）。
    ratio = s["grid_sensitivity_hdr90_area_km2_0p01"] / s["grid_sensitivity_hdr90_area_km2_0p04"]
    assert 0.3 < ratio < 1.5
    assert 0.0 < s["centroid_distance_from_receptor_km"] < 200.0
    assert 0.0 <= s["centroid_bearing_from_receptor_deg"] < 360.0


# ── 分批累加與順序不變 ───────────────────────────────────────────────────────


def _assert_products_identical(a, b):
    assert a.tables.keys() == b.tables.keys()
    for name in a.tables:
        assert a.tables[name].equals(b.tables[name]), name
    assert a.summary_rows == b.summary_rows
    assert a.manifest == b.manifest


def test_chunked_accumulation_is_bit_identical_to_one_shot():
    identity, results = walk_results(n_time=90)
    cfg = FootprintConfig(lookback_snapshot_hours=(24, 72), snapshot_minimum_particle_count=10)
    per_unit = run_accumulator(identity, results, config=cfg)

    rows_all, big = concat_results(identity, results)
    one_shot = run_accumulator(identity, [("all", rows_all, big)], config=cfg)
    _assert_products_identical(per_unit, one_shot)

    reversed_order = run_accumulator(identity, list(reversed(results)), config=cfg)
    _assert_products_identical(per_unit, reversed_order)

    # 同一個單元拆成兩個批次（依欄位切開）。
    split = []
    for key, rows, res in results:
        half = rows.size // 2
        for part, sl in enumerate((slice(0, half), slice(half, None))):
            split.append(
                (
                    f"{key}-{part}",
                    rows[sl],
                    fx.make_result(
                        identity,
                        rows[sl],
                        res.trajectory_x_m[:, sl],
                        res.trajectory_y_m[:, sl],
                        res.trajectory_status[:, sl],
                    ),
                )
            )
    _assert_products_identical(per_unit, run_accumulator(identity, split, config=cfg))


# ── 邊界弧長分布 ─────────────────────────────────────────────────────────────


def crossing_results(identity, rows, points):
    """每顆粒子在最後一個樣本穿越邊界於 ``points[i]``。"""

    n = rows.size
    n_time = 3
    x = np.zeros((n_time, n))
    y = np.zeros((n_time, n))
    status = np.zeros((n_time, n), dtype=np.int32)
    for i, (px, py) in enumerate(points):
        x[:, i] = np.linspace(0.0, px, n_time)
        y[:, i] = np.linspace(0.0, py, n_time)
    x[0] = 0.0
    y[0] = 0.0
    status[-1] = 1
    return fx.make_result(identity, rows, x, y, status)


def test_arc_histogram_on_known_geometry():
    identity, units = walk_world(members=4, arrivals=1)
    key, rows = units[0]  # 一個單元 8 顆粒子（2 個受體 × 4 個系集成員）
    assert rows.size == 8
    points = [(-25e3, -40e3)] * 4 + [(50e3, 0.0)] * 2 + [(0.0, 40e3)] * 2
    acc = make_accumulator(identity)
    acc.add_batch(key, rows, crossing_results(identity, rows, points))
    for k2, r2 in units[1:]:
        acc.add_batch(k2, r2, static_result(identity, r2, n_time=3))
    products = finalize(acc)
    arc = sorted(
        products.tables["boundary_departure_along_arc_by_site"].to_pylist(), key=lambda r: r["bin_index"]
    )
    assert len(arc) == 28  # 280 km / 10 km
    assert arc[0]["s_start_km"] == 0.0 and arc[-1]["s_end_km"] == pytest.approx(280.0)
    counts = {r["bin_index"]: r["crossed_particle_count"] for r in arc}
    # S 邊 x=-25 km -> s=25 km；E 邊 y=0 -> s=140 km；N 邊 x=0 -> s=230 km。
    assert counts[2] == 4 and counts[14] == 2 and counts[23] == 2
    assert sum(counts.values()) == 8
    assert arc[-1]["cumulative_fraction"] == pytest.approx(1.0)
    assert arc[2]["cumulative_fraction"] == pytest.approx(0.5)
    assert arc[14]["cumulative_fraction"] == pytest.approx(6 / 8)
    assert sum(r["fraction"] for r in arc) == pytest.approx(1.0)
    assert arc[0]["boundary_length_km"] == pytest.approx(280.0)


def test_edge_intervals_on_known_geometry():
    identity, units = walk_world(members=4, arrivals=1)
    products = run_accumulator(identity, [(k, r, static_result(identity, r, n_time=3)) for k, r in units])
    intervals = products.tables["boundary_edge_intervals_by_site"].to_pylist()
    edges = [r for r in intervals if r["kind"] == "edge"]
    corners = [r for r in intervals if r["kind"] == "corner"]
    assert [r["name"] for r in edges] == ["south", "east", "north"]
    assert edges[0]["s_start_km"] == 0.0
    assert edges[-1]["s_end_km"] == pytest.approx(280.0)
    assert edges[0]["s_end_km"] == pytest.approx(100.0, abs=1.0)
    assert edges[1]["s_end_km"] == pytest.approx(180.0, abs=1.0)
    # 各邊首尾相接。
    assert edges[0]["s_end_km"] == edges[1]["s_start_km"] and edges[1]["s_end_km"] == edges[2]["s_start_km"]
    assert [r["name"] for r in corners] == ["south-east", "east-north"]
    assert corners[0]["s_start_km"] == pytest.approx(100.0, abs=1.0)
    assert corners[1]["s_start_km"] == pytest.approx(180.0, abs=1.0)


def test_arc_without_open_boundary_is_marked_unavailable():
    identity, units = walk_world(members=4, arrivals=1)
    acc = FootprintAccumulator(
        identity,
        config=FootprintConfig(),
        specs={SITE: FootprintSiteSpec(flow_domain=DOMAIN)},
        lonlat_projector=fx.linear_projector,
    )
    for k, r in units:
        acc.add_batch(k, r, static_result(identity, r, n_time=3))
    products = finalize(acc)
    assert products.tables["boundary_departure_along_arc_by_site"].num_rows == 0
    assert products.manifest["sites"][SITE]["boundary_departure"] == {"available": False}


# ── 回溯快照 ─────────────────────────────────────────────────────────────────


def _snap_status(products) -> dict[int, dict[str, Any]]:
    return {r["snapshot_hours"]: r for r in products.tables["lookback_snapshot_by_site"].to_pylist()}


def _snap_cells(products, hours: int) -> list[dict[str, Any]]:
    return table_rows(products, "lookback_snapshot_density_by_site", snapshot_hours=hours)


def test_snapshot_active_fraction_including_short_units():
    identity, units = walk_world(members=4, arrivals=2)
    assert len(units) == 4  # 2 到達 × 2 沉降速度
    cfg = FootprintConfig(lookback_snapshot_hours=(24, 72), snapshot_minimum_particle_count=3)
    rng = np.random.default_rng(3)
    batches = []
    for i, (key, rows) in enumerate(units):
        n_time = 100 if i < 2 else 50  # 後兩個單元的 T 不足 72 小時
        res = fx.random_walk_unit(rng, identity, rows, n_time=n_time, step_m=1500.0, terminate=False)
        status = res.trajectory_status.copy()
        if i == 0:  # 第一個單元的一半粒子在第 10 小時終止
            status[10:, :4] = 3
        batches.append(
            (key, rows, fx.make_result(identity, rows, res.trajectory_x_m, res.trajectory_y_m, status))
        )
    products = run_accumulator(identity, batches, config=cfg)
    total = len(identity)
    snaps = _snap_status(products)
    # 24 小時：單元 0 剩一半（4）、其餘三個單元各 8 顆全在漂流。
    assert total == 32
    assert snaps[24]["active_particle_count"] == 4 + 8 * 3
    assert snaps[24]["site_particle_count"] == total
    assert snaps[24]["active_fraction"] == pytest.approx(28 / total)
    # 72 小時：T=50 的兩個單元視為全部終止；單元 0 的一半也已終止。
    assert snaps[72]["active_particle_count"] == 4 + 8
    assert snaps[72]["active_fraction"] == pytest.approx(12 / total)
    s = products.summary_rows[0]
    assert s["active_fraction_72h"] == pytest.approx(12 / total)
    assert snaps[24]["status"] == "available"
    # 分母是該時刻仍在漂流的粒子數：質量守恆，總和為 1；原始計數總和等於仍在漂流粒子數。
    cells = _snap_cells(products, 24)
    assert sum(r["fraction_of_active"] for r in cells) == pytest.approx(1.0, abs=1e-9)
    assert sum(r["particle_count"] for r in cells) <= 28  # 平滑後才有計數為 0 的格
    assert sum(r["particle_count"] for r in cells) == 28


def test_snapshot_below_particle_threshold_draws_no_density():
    identity, results = walk_results(n_time=90)
    cfg = FootprintConfig(lookback_snapshot_hours=(24, 720), snapshot_minimum_particle_count=30)
    products = run_accumulator(identity, results, config=cfg)
    snaps = _snap_status(products)
    assert snaps[720]["active_particle_count"] == 0
    assert snaps[720]["status"] == "insufficient_active_particles"
    assert snaps[720]["active_fraction"] == 0.0
    assert _snap_cells(products, 720) == []
    assert snaps[720]["hdr50_area_km2"] is None
    assert products.summary_rows[0]["snapshot_hdr50_area_km2_720h"] is None
    # 門檻本身寫進表與 manifest。
    assert snaps[24]["minimum_active_particle_count"] == 30
    assert products.manifest["config"]["snapshot_minimum_particle_count"] == 30


def test_snapshot_threshold_is_inclusive_and_configurable():
    identity, results = walk_results(n_time=90)
    base = FootprintConfig(lookback_snapshot_hours=(24,), snapshot_minimum_particle_count=1)
    n_active = _snap_status(run_accumulator(identity, results, config=base))[24]["active_particle_count"]
    assert n_active > 10
    at = FootprintConfig(lookback_snapshot_hours=(24,), snapshot_minimum_particle_count=n_active)
    above = FootprintConfig(lookback_snapshot_hours=(24,), snapshot_minimum_particle_count=n_active + 1)
    p_at = run_accumulator(identity, results, config=at)
    p_above = run_accumulator(identity, results, config=above)
    assert _snap_status(p_at)[24]["status"] == "available" and _snap_cells(p_at, 24)
    assert _snap_status(p_above)[24]["status"] == "insufficient_active_particles"
    assert _snap_cells(p_above, 24) == []
    # 預設門檻是 200。
    assert FootprintConfig().snapshot_minimum_particle_count == 200


def test_snapshot_shares_grid_and_area_weights_with_residence():
    identity, results = walk_results(n_time=90, step_m=800.0)
    cfg = FootprintConfig(
        lookback_snapshot_hours=(24,), snapshot_minimum_particle_count=10, snapshot_smoothing_km=0.0
    )
    products = run_accumulator(identity, results, config=cfg)
    grid = products.manifest["sites"][SITE]["grid"]
    residence = {
        (r["cell_ix"], r["cell_iy"]): r
        for r in table_rows(products, "residence_density_by_site", group_kind="all")
    }
    cells = _snap_cells(products, 24)
    assert cells
    for r in cells:
        assert 0 <= r["cell_ix"] < grid["nx"] and 0 <= r["cell_iy"] < grid["ny"]
        # 快照的粒子位置也是停留時數的一部分，因此每個快照格都在停留時數表裡，面積與座標相同。
        ref = residence[(r["cell_ix"], r["cell_iy"])]
        assert r["cell_area_km2"] == ref["cell_area_km2"]
        assert r["lon_center_deg"] == ref["lon_center_deg"] and r["lat_center_deg"] == ref["lat_center_deg"]
        # 密度＝比例 ÷ 面積（面積加權）。
        assert r["density_fraction_per_km2"] == pytest.approx(r["fraction_of_active"] / r["cell_area_km2"])


def _land_box(lon_min, lat_min, lon_max, lat_max):
    return box(lon_min, lat_min, lon_max, lat_max)


def test_snapshot_density_is_zero_on_land_and_outside_domain_even_when_smoothed():
    identity, results = walk_results(n_time=90, step_m=500.0)
    east_x = 25e3  # 陸地位於受體東側 25 km 以外，粒子走不到
    lon_w, lat_s = fx.linear_projector(np.array(["x"]), np.array([east_x]), np.array([-40e3]))
    lon_e, lat_n = fx.linear_projector(np.array(["x"]), np.array([60e3]), np.array([40e3]))
    land = _land_box(float(lon_w[0]), float(lat_s[0]), float(lon_e[0]), float(lat_n[0]))
    for smoothing in (0.0, 3.0, 8.0):
        cfg = FootprintConfig(
            lookback_snapshot_hours=(24,), snapshot_minimum_particle_count=10, snapshot_smoothing_km=smoothing
        )
        products = run_accumulator(identity, results, config=cfg, land_lonlat=land)
        mask = {
            (r["cell_ix"], r["cell_iy"]): r for r in products.tables["footprint_sea_mask_by_site"].to_pylist()
        }
        cells = _snap_cells(products, 24)
        assert cells
        assert sum(r["fraction_of_active"] for r in cells) == pytest.approx(1.0, abs=1e-9)
        for r in cells:
            m = mask[(r["cell_ix"], r["cell_iy"])]
            assert m["is_sea_cell"], (smoothing, r)
            assert r["lon_center_deg"] < float(lon_w[0]) + 0.02
        # 陸地與 flow domain 外的格全部標成非海域，且沒有任何密度列。
        land_cells = [k for k, m in mask.items() if not m["is_sea_cell"]]
        assert land_cells
        with_density = {(r["cell_ix"], r["cell_iy"]) for r in cells}
        assert not (set(land_cells) & with_density)
        assert products.manifest["sites"][SITE]["sea_mask"]["land_or_outside_domain_cell_count"] == len(
            land_cells
        )


def test_snapshot_smoothing_conserves_total_mass_and_spreads_only_over_sea():
    identity, results = walk_results(n_time=90, step_m=500.0)
    raw = FootprintConfig(
        lookback_snapshot_hours=(24,), snapshot_minimum_particle_count=10, snapshot_smoothing_km=0.0
    )
    smooth = FootprintConfig(
        lookback_snapshot_hours=(24,), snapshot_minimum_particle_count=10, snapshot_smoothing_km=6.0
    )
    p_raw = run_accumulator(identity, results, config=raw)
    p_smooth = run_accumulator(identity, results, config=smooth)
    c_raw, c_smooth = _snap_cells(p_raw, 24), _snap_cells(p_smooth, 24)
    assert sum(r["fraction_of_active"] for r in c_raw) == pytest.approx(1.0, abs=1e-9)
    assert sum(r["fraction_of_active"] for r in c_smooth) == pytest.approx(1.0, abs=1e-9)
    assert len(c_smooth) > len(c_raw)  # 平滑把質量攤到相鄰格
    # 不平滑時，比例就是原始計數 ÷ 仍在漂流粒子數。
    n_active = _snap_status(p_raw)[24]["active_particle_count"]
    for r in c_raw:
        assert r["fraction_of_active"] == pytest.approx(r["particle_count"] / n_active, abs=1e-12)
    assert _snap_status(p_raw)[24]["mass_rescale_factor"] == pytest.approx(1.0, abs=1e-9)
    assert p_smooth.manifest["sites"][SITE]["sea_mask"]["smoothing_sigma_km"] == 6.0


def test_smooth_masked_density_is_normalized_convolution():
    ny, nx = 12, 15
    mask = np.ones((ny, nx), dtype=bool)
    mask[:, 9:] = False  # 東側是陸地
    mask[3:6, 2:4] = False  # 一座小島
    # 常數場經過正規化卷積仍是常數（陸地不會把海域的值拉低）。
    const = np.full((ny, nx), 2.0)
    out = footprint.smooth_masked_density(const, mask, (1.0, 1.0))
    assert np.allclose(out[mask], 2.0)
    assert np.all(out[~mask] == 0.0)
    # 質量只來自海域，陸地格恆為 0；集中在海岸邊的密度不會外洩到陸地。
    spike = np.zeros((ny, nx))
    spike[6, 8] = 5.0
    out2 = footprint.smooth_masked_density(spike, mask, (1.5, 1.5))
    assert np.all(out2[~mask] == 0.0)
    assert out2[mask].sum() > 0
    assert out2[6, 7] > 0 and out2[6, 8] > 0
    # 遮罩外的輸入值被忽略。
    dirty = spike.copy()
    dirty[:, 12] = 100.0
    assert np.array_equal(footprint.smooth_masked_density(dirty, mask, (1.5, 1.5)), out2)
    # sigma 為 0 時不平滑。
    assert np.array_equal(footprint.smooth_masked_density(spike, mask, (0.0, 0.0)), spike * mask)
    with pytest.raises(FootprintError):
        footprint.smooth_masked_density(spike, mask[:-1], (1.0, 1.0))


def test_sea_mask_marks_full_land_cells_and_keeps_partly_land_small_islands():
    identity, results = walk_results(n_time=90, step_m=500.0)
    res = 0.02
    # 大島：對齊 0.04 度網格、覆蓋整整兩個格（西側 20 km 外，粒子走不到）。
    lon_a = math.floor((fx.LON0 - 0.25) / 0.04) * 0.04
    lat_a = math.floor((fx.LAT0 + 0.10) / 0.04) * 0.04
    big = _land_box(lon_a, lat_a, lon_a + 2 * res, lat_a + res)
    # 小島：面積遠小於半個格，位在另一個沒有粒子的格內。
    lon_b = math.floor((fx.LON0 - 0.25) / 0.04) * 0.04 + 0.08
    small = _land_box(lon_b + 0.006, lat_a + 0.006, lon_b + 0.010, lat_a + 0.010)
    cfg = FootprintConfig(lookback_snapshot_hours=(24,), snapshot_minimum_particle_count=10)
    products = run_accumulator(identity, results, config=cfg, land_lonlat=big.union(small))
    mask = {
        (r["lon_center_deg"], r["lat_center_deg"]): r
        for r in products.tables["footprint_sea_mask_by_site"].to_pylist()
    }

    def cell_at(lon, lat):
        return min(
            mask.values(), key=lambda r: abs(r["lon_center_deg"] - lon) + abs(r["lat_center_deg"] - lat)
        )

    for dlon in (0.5 * res, 1.5 * res):
        c = cell_at(lon_a + dlon, lat_a + 0.5 * res)
        assert c["sea_area_fraction"] == pytest.approx(0.0, abs=1e-9)
        assert c["is_sea_cell"] is False
    c_small = cell_at(lon_b + 0.5 * res, lat_a + 0.5 * res)
    assert 0.9 < c_small["sea_area_fraction"] < 1.0  # 有陸地，但不到半個格
    assert c_small["is_sea_cell"] is True
    stats = products.manifest["sites"][SITE]["sea_mask"]
    assert stats["cells_partly_land_but_sea"] >= 1


def test_sea_cells_with_observed_particles_are_never_masked():
    # 陸地多邊形蓋住整個粒子活動範圍：有粒子出現的格仍必須是海域，質量不得被遮罩吃掉。
    identity, results = walk_results(n_time=90, step_m=500.0)
    land = _land_box(fx.LON0 - 0.3, fx.LAT0 - 0.3, fx.LON0 + 0.3, fx.LAT0 + 0.3)
    cfg = FootprintConfig(lookback_snapshot_hours=(24,), snapshot_minimum_particle_count=10)
    products = run_accumulator(identity, results, config=cfg, land_lonlat=land)
    stats = products.manifest["sites"][SITE]["sea_mask"]
    assert stats["cells_sea_by_observation_only"] > 0
    cells = _snap_cells(products, 24)
    assert sum(r["fraction_of_active"] for r in cells) == pytest.approx(1.0, abs=1e-9)


def test_snapshot_hdr_uses_area_weighted_density_like_residence():
    identity, results = walk_results(n_time=90, step_m=800.0)
    cfg = FootprintConfig(
        lookback_snapshot_hours=(24,), snapshot_minimum_particle_count=10, snapshot_smoothing_km=0.0
    )
    products = run_accumulator(identity, results, config=cfg)
    cells = _snap_cells(products, 24)
    weights = np.asarray([r["fraction_of_active"] for r in cells])
    area = np.asarray([r["cell_area_km2"] for r in cells])
    expected = footprint.compute_hdr(weights, area, cfg.hdr_levels)
    rows = table_rows(products, "residence_hdr_by_site", group_kind="snapshot", group_value="24")
    assert [r["level"] for r in rows] == [0.5, 0.75, 0.9]
    for row, hdr in zip(rows, expected, strict=True):
        assert row["area_km2"] == pytest.approx(hdr["area_km2"])
        assert row["achieved_fraction"] >= row["level"] - 1e-9
    status = _snap_status(products)[24]
    assert status["hdr50_area_km2"] < status["hdr75_area_km2"] <= status["hdr90_area_km2"]


def test_snapshot_config_validation():
    with pytest.raises(FootprintError):
        FootprintConfig(snapshot_smoothing_km=-1.0)
    with pytest.raises(FootprintError):
        FootprintConfig(snapshot_minimum_particle_count=0)
    with pytest.raises(FootprintError):
        FootprintConfig(land_mask_min_sea_fraction=0.0)
    assert FootprintConfig().snapshot_smoothing_km == 2.0


# ── 水深 ─────────────────────────────────────────────────────────────────────


class _ConstantDepth:
    def __init__(self, value: float) -> None:
        self.value = value

    def depth_at(self, lon, lat):
        return np.full(np.shape(lon), self.value, dtype=np.float64)


def test_depth_fraction_with_and_without_bathymetry():
    identity, units = walk_world(members=4, arrivals=1)
    batches = [(k, r, static_result(identity, r, n_time=4)) for k, r in units]
    absent = run_accumulator(identity, batches)
    s = absent.summary_rows[0]
    assert s["hours_fraction_depth_below_threshold"] is None
    assert s["hours_fraction_depth_missing"] is None
    assert absent.manifest["bathymetry"]["missing_sites"] == [SITE]
    assert absent.manifest["sites"][SITE]["bathymetry_available"] is False

    shallow = run_accumulator(identity, batches, bathymetry_by_site={SITE: _ConstantDepth(120.0)})
    assert shallow.summary_rows[0]["hours_fraction_depth_below_threshold"] == 1.0
    assert shallow.summary_rows[0]["hours_fraction_depth_missing"] == 0.0
    deep = run_accumulator(identity, batches, bathymetry_by_site={SITE: _ConstantDepth(500.0)})
    assert deep.summary_rows[0]["hours_fraction_depth_below_threshold"] == 0.0
    assert deep.manifest["bathymetry"]["available_sites"] == [SITE]
    nan = run_accumulator(identity, batches, bathymetry_by_site={SITE: _ConstantDepth(float("nan"))})
    assert nan.summary_rows[0]["hours_fraction_depth_below_threshold"] is None
    assert nan.summary_rows[0]["hours_fraction_depth_missing"] == 1.0


def _tri_field():
    # 兩個三角形組成 [120,122]×[24,26] 方形，水深 = 500 × (lon - 120)（0 到 1000 m）。
    lon = np.array([120.0, 122.0, 122.0, 120.0])
    lat = np.array([24.0, 24.0, 26.0, 26.0])
    depth = 500.0 * (lon - 120.0)
    tri = np.array([[0, 1, 2], [0, 2, 3]])
    return bathymetry.BathymetryField(lon, lat, depth, tri)


def test_bathymetry_field_interpolation_and_outside_nan():
    field = _tri_field()
    d = field.depth_at(np.array([121.0, 120.5, 119.0, 121.0]), np.array([25.0, 24.5, 25.0, 27.0]))
    assert d[0] == pytest.approx(500.0)
    assert d[1] == pytest.approx(250.0)
    assert np.isnan(d[2]) and np.isnan(d[3])


def test_bathymetry_validation_fail_closed():
    with pytest.raises(bathymetry.BathymetryError):
        bathymetry.BathymetryField(np.zeros(3), np.zeros(3), np.zeros(2), np.array([[0, 1, 2]]))
    with pytest.raises(bathymetry.BathymetryError):
        bathymetry.BathymetryField(np.zeros(3), np.zeros(3), np.zeros(3), np.array([[0, 1, 5]]))
    with pytest.raises(bathymetry.BathymetryError):
        bathymetry.BathymetryField(
            np.array([0.0, 1.0, np.nan]), np.zeros(3), np.zeros(3), np.array([[0, 1, 2]])
        )


def test_isobath_geojson_skips_levels_outside_depth_range():
    payload = bathymetry.build_isobath_geojson({SITE: _tri_field()}, (50, 100, 200, 1000, 2000))
    depths = sorted(f["properties"]["depth_m"] for f in payload["features"])
    assert depths == [50.0, 100.0, 200.0]  # 最深 1000 m 與更深的等深線不存在
    for feature in payload["features"]:
        assert feature["geometry"]["type"] == "MultiLineString"
        lons = [pt[0] for seg in feature["geometry"]["coordinates"] for pt in seg]
        assert all(abs(lon - (120.0 + feature["properties"]["depth_m"] / 500.0)) < 1e-6 for lon in lons)


def _write_grid_dir(root: Path) -> Path:
    grid = root / "domain-a" / "grid"
    grid.mkdir(parents=True)
    field = _tri_field()
    np.save(grid / "source_lon.npy", field.node_lon)
    np.save(grid / "source_lat.npy", field.node_lat)
    np.save(grid / "source_depth_m.npy", field.node_depth_m)
    np.save(grid / "source_face_nodes_local.npy", np.array([[0, 1, 2, -1], [0, 2, 3, -1]]))
    np.save(grid / "source_face_node_count.npy", np.array([3, 3]))
    return grid


def test_load_ocm_grid_bathymetry_and_missing_arrays(tmp_path):
    grid = _write_grid_dir(tmp_path)
    field = bathymetry.load_ocm_grid_bathymetry(grid)
    assert field.depth_at(np.array([121.0]), np.array([25.0]))[0] == pytest.approx(500.0)
    (grid / "source_depth_m.npy").unlink()
    with pytest.raises(bathymetry.BathymetryError, match="缺少"):
        bathymetry.load_ocm_grid_bathymetry(grid)


def test_resolve_site_grid_dirs(tmp_path):
    grid = _write_grid_dir(tmp_path)
    got = bathymetry.resolve_site_grid_dirs({SITE: "region-a"}, {"region-a": "domain-a"}, tmp_path)
    assert got == {SITE: grid}
    with pytest.raises(bathymetry.BathymetryError):
        bathymetry.resolve_site_grid_dirs({SITE: "region-b"}, {"region-a": "domain-a"}, tmp_path)
    with pytest.raises(bathymetry.BathymetryError):
        bathymetry.resolve_site_grid_dirs({SITE: "region-a"}, {"region-a": "nope"}, tmp_path)


def test_export_isobaths_script(tmp_path, capsys):
    import importlib.util

    _write_grid_dir(tmp_path)
    script = Path(__file__).resolve().parents[1] / "scripts" / "export_isobaths.py"
    spec = importlib.util.spec_from_file_location("export_isobaths", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "out" / "isobaths.geojson"
    rc = module.main(["--site-grid", f"{SITE}={tmp_path / 'domain-a' / 'grid'}", "--output", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert {f["properties"]["depth_m"] for f in payload["features"]} == {50.0, 100.0, 200.0}
    assert json.loads(capsys.readouterr().out)["features"] == 3
    with pytest.raises(SystemExit):  # 拒絕覆寫
        module.main(["--site-grid", f"{SITE}={tmp_path / 'domain-a' / 'grid'}", "--output", str(out)])


# ── 與 PostprocessAccumulator 的整合與既有產品回歸 ────────────────────────────


def test_existing_products_are_bit_identical_to_pre_wp38_baseline(tmp_path):
    out = fx.run_legacy_pipeline(tmp_path / "off")
    assert fx.legacy_digests(out) == GOLDEN_LEGACY_DIGESTS


def test_enabling_footprint_does_not_change_existing_products(tmp_path):
    off = fx.run_legacy_pipeline(tmp_path / "off")
    on = fx.run_legacy_pipeline(
        tmp_path / "on",
        config=pp.PostprocessConfig(footprint=FootprintConfig(lookback_snapshot_hours=(3, 6))),
    )
    assert fx.legacy_digests(on) == fx.legacy_digests(off) == GOLDEN_LEGACY_DIGESTS
    for stem in fx.FOOTPRINT_STEMS:
        assert (on / f"{stem}.parquet").is_file(), stem
        assert not (off / f"{stem}.parquet").exists(), stem
    assert (on / "footprint_summary_by_site.csv").is_file()
    assert not (off / "footprint_summary_by_site.csv").exists()


def test_postprocess_manifest_records_footprint_block(tmp_path):
    out = fx.run_legacy_pipeline(
        tmp_path / "on",
        config=pp.PostprocessConfig(footprint=FootprintConfig(lookback_snapshot_hours=(3, 6))),
    )
    manifests = sorted(out.glob("*manifest*.json"))
    assert manifests
    payload = json.loads(manifests[0].read_text(encoding="utf-8"))
    block = payload["footprint"]
    site = block["sites"]["siteA"]
    assert abs(site["normalized_sum"] - 1.0) <= 1e-9
    assert site["total_particle_hours"] > 0
    assert site["grid"]["resolution_deg"] == 0.02
    assert "一維" in block["kde_by_site_note_zh"] or "1D" in block["kde_by_site_note_zh"]
    assert payload["config"]["kde_by_site_note_zh"]
    assert payload["config"]["footprint_enabled"] is True
    assert "residence_density_by_site" in block["output_tables"]
    summary = pq.read_table(out / "footprint_summary_by_site.parquet").to_pylist()
    assert len(summary) == 1 and summary[0]["study_site_id"] == "siteA"
    csv_lines = (out / "footprint_summary_by_site.csv").read_text(encoding="utf-8").splitlines()
    assert len(csv_lines) == 2


def test_footprint_requires_projector_and_geometry(tmp_path):
    identity, _ = fx.regression_units()
    cfg = pp.PostprocessConfig(footprint=FootprintConfig())
    with pytest.raises(pp.PostprocessError):
        pp.PostprocessAccumulator(identity, output_dir=tmp_path / "a", config=cfg)
    with pytest.raises(pp.PostprocessError):
        pp.PostprocessAccumulator(
            identity, output_dir=tmp_path / "b", config=cfg, lonlat_projector=fx.linear_projector
        )


def test_trial_subset_identity_table():
    from lagrangian_backtracking.cli import _subset_identity_table

    identity, units = walk_world(members=4, arrivals=2)
    keep = np.concatenate([units[0][1], units[3][1]])
    sub = _subset_identity_table(identity, keep)
    assert len(sub) == keep.size
    assert sub.particle_id.tolist() == identity.particle_id[keep].tolist()
    assert sub.experiment_case_id == identity.experiment_case_id
