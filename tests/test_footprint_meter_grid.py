"""WP-70：停留時數密度的規則公尺網格（``FootprintConfig.grid_cell_km``）測試。

全部使用合成資料（``tests/wp38_fixture.py``），不讀任何正式運算輸出。重點：每格面積正好 1 km²、
正規化總和為 1、HDR 單調且是最小面積集合、與經緯度網格的總停留時數一致、預設模式不受影響。
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import test_footprint as tf
import wp38_fixture as fx
from shapely.geometry import box
from test_footprint_figures import (
    DOMAIN,
    OPEN_BOUNDARY,
    RECEPTORS,
    SITES,
    _build,
    _two_site_world,
    _write_isobath_geojson,
)
from test_native_figures import _write_synthetic_coastline, _write_synthetic_postprocess_dir

from lagrangian_backtracking.native_runner import figures as native_figures
from lagrangian_backtracking.native_runner import footprint
from lagrangian_backtracking.native_runner.footprint import (
    FootprintAccumulator,
    FootprintConfig,
    FootprintError,
    FootprintMeterGrid,
    FootprintSiteSpec,
    summary_table,
)

METER_CFG = FootprintConfig(grid_cell_km=1.0)


def _identity_projection(x, y):
    return np.asarray(x, dtype=np.float64) / 1000.0, np.asarray(y, dtype=np.float64) / 1000.0


# ── 網格本身 ─────────────────────────────────────────────────────────────


def test_meter_grid_cell_area_is_exactly_one_km2_and_edges_are_1000_m():
    grid = footprint.make_meter_grid(
        -52_345.0, 53_210.0, -41_000.0, 43_999.0, cell_m=1000.0, align_m=2000.0, project=_identity_projection
    )
    area = grid.cell_area_km2()
    assert area.shape == (grid.cell_count,)
    assert np.all(area == 1.0)
    xx, yy = grid.corners_xy()
    assert xx.shape == yy.shape == (grid.ny + 1, grid.nx + 1)
    assert np.all(np.diff(xx, axis=1) == 1000.0)
    assert np.all(np.diff(yy, axis=0) == 1000.0)
    # 西南角是對齊值（2 km 的整數倍，因此也是 1 km 的整數倍），範圍涵蓋輸入。
    assert grid.x0_m % 2000.0 == 0.0 and grid.y0_m % 2000.0 == 0.0
    assert grid.x0_m <= -52_345.0 and grid.y0_m <= -41_000.0
    assert grid.x0_m + grid.nx * 1000.0 >= 53_210.0 and grid.y0_m + grid.ny * 1000.0 >= 43_999.0


def test_meter_grid_flat_index_floors_and_marks_outside():
    grid = FootprintMeterGrid(
        x0_m=-2000.0, y0_m=-1000.0, cell_m=1000.0, nx=4, ny=2, project=_identity_projection
    )
    x = np.array([-1999.0, -0.001, 0.0, 999.999, 1000.0, 1999.0, -2000.1, 2000.0, 100.0, 100.0])
    y = np.array([-999.0, 0.0, 0.0, 999.0, 0.5, 0.5, 0.0, 0.0, -1000.1, 1000.0])
    idx = grid.flat_index_xy(x, y)
    # 格 (ix, iy)：x∈[-2000,-1000)→0、[-1000,0)→1、[0,1000)→2、[1000,2000)→3；y∈[-1000,0)→0、[0,1000)→1
    assert idx.tolist() == [0, 4 + 1, 4 + 2, 4 + 2, 4 + 3, 4 + 3, -1, -1, -1, -1]
    cx, cy = grid.centers_xy()
    assert grid.flat_index_xy(cx, cy).tolist() == list(range(grid.cell_count))


def test_meter_grids_of_different_sizes_are_nested():
    kwargs = {"align_m": 2000.0, "project": _identity_projection}
    grids = [
        footprint.make_meter_grid(-61_234.0, 71_234.0, -43_210.0, 49_876.0, cell_m=c, **kwargs)
        for c in (500.0, 1000.0, 2000.0)
    ]
    assert len({(g.x0_m, g.y0_m) for g in grids}) == 1
    for g in grids:
        assert g.x0_m + g.nx * g.cell_m >= 71_234.0
        assert g.y0_m + g.ny * g.cell_m >= 49_876.0
    # 同一個點在三種格寬下，細格落在粗格之內。
    x = np.array([123.4, -33_333.3, 70_000.0])
    y = np.array([-20_000.0, 456.7, 48_000.0])
    fine, mid, coarse = (g.flat_index_xy(x, y) for g in grids)
    for k in range(x.size):
        ix_f, iy_f = fine[k] % grids[0].nx, fine[k] // grids[0].nx
        ix_m, iy_m = mid[k] % grids[1].nx, mid[k] // grids[1].nx
        ix_c, iy_c = coarse[k] % grids[2].nx, coarse[k] // grids[2].nx
        assert (ix_f // 2, iy_f // 2) == (ix_m, iy_m)
        assert (ix_m // 2, iy_m // 2) == (ix_c, iy_c)


def test_config_validation_for_meter_grid():
    assert FootprintConfig().grid_cell_km is None
    with pytest.raises(FootprintError):
        FootprintConfig(grid_cell_km=0.0)
    with pytest.raises(FootprintError):
        FootprintConfig(grid_cell_km=1.0, sensitivity_cells_km=(0.0,))
    with pytest.raises(FootprintError):
        FootprintConfig(grid_cell_km=1.0, padding_km=-1.0)
    manifest = METER_CFG.to_manifest_dict()
    assert manifest["grid_cell_km"] == 1.0 and manifest["sensitivity_cells_km"] == [0.5, 2.0]
    # 預設模式的設定 manifest 不出現公尺網格欄位（與修改前完全相同）。
    assert not {"grid_cell_km", "sensitivity_cells_km", "padding_km"} & set(
        FootprintConfig().to_manifest_dict()
    )


# ── 累加、正規化、面積 ─────────────────────────────────────────────────────


def test_default_mode_products_do_not_contain_meter_grid_outputs():
    identity, results = tf.walk_results(members=4, arrivals=2)
    products = tf.run_accumulator(identity, results)
    assert "footprint_grid_corners_by_site" not in products.tables
    grid_row = products.tables["footprint_grid_by_site"].to_pylist()[0]
    assert "cell_m" not in grid_row and grid_row["resolution_deg"] == 0.02
    assert products.summary_rows[0]["grid_resolution_deg"] == 0.02
    assert "grid_cell_km" not in products.summary_rows[0]
    assert "grid_definition_zh" not in products.manifest


def test_meter_residence_area_is_one_km2_and_normalised_sum_is_one_for_every_group():
    identity, results = tf.walk_results()
    products = tf.run_accumulator(identity, results, config=METER_CFG)
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
            assert all(r["cell_area_km2"] == 1.0 for r in sel)
            assert sum(r["fraction_of_total"] for r in sel) == pytest.approx(1.0, abs=1e-12)
            assert sum(r["density_fraction_per_km2"] * r["cell_area_km2"] for r in sel) == pytest.approx(
                1.0, abs=1e-12
            )
            # 面積恆為 1，密度就等於正規化比例。
            assert all(r["density_fraction_per_km2"] == r["fraction_of_total"] for r in sel)
    site = products.manifest["sites"][tf.SITE]
    assert abs(site["normalized_sum"] - 1.0) <= 1e-12
    assert site["grid"]["kind"] == footprint.METER_GRID_KIND and site["grid"]["cell_m"] == 1000.0
    assert site["grid"]["cell_area_km2"] == 1.0
    assert products.manifest["config"]["grid_cell_km"] == 1.0
    assert "grid_definition_zh" in products.manifest


def test_meter_grid_tables_describe_grid_and_corners():
    identity, results = tf.walk_results(members=4, arrivals=2)
    products = tf.run_accumulator(identity, results, config=METER_CFG)
    grid_rows = products.tables["footprint_grid_by_site"].to_pylist()
    assert len(grid_rows) == 1
    g = grid_rows[0]
    assert g["cell_m"] == 1000.0 and g["x0_m"] % 2000.0 == 0.0 and g["y0_m"] % 2000.0 == 0.0
    assert "resolution_deg" not in g
    assert g["hours_outside_grid"] == 0 and g["hours_outside_lookup"] == 0
    # 外包範圍涵蓋 flow domain（±50 km × ±40 km）外推 5 km。
    assert g["x0_m"] <= -55_000.0 and g["y0_m"] <= -45_000.0
    assert g["x0_m"] + g["nx"] * 1000.0 >= 55_000.0 and g["y0_m"] + g["ny"] * 1000.0 >= 45_000.0
    corners = products.tables["footprint_grid_corners_by_site"].to_pylist()
    assert len(corners) == (g["nx"] + 1) * (g["ny"] + 1)
    # 格角經緯度回推到平面公尺後，相鄰格角相距正好 1000 m。
    lon = np.zeros((g["ny"] + 1, g["nx"] + 1))
    lat = np.zeros_like(lon)
    for c in corners:
        lon[c["corner_iy"], c["corner_ix"]] = c["lon_deg"]
        lat[c["corner_iy"], c["corner_ix"]] = c["lat_deg"]
    x, y = fx.lonlat_to_xy(lon, lat)
    assert np.allclose(np.diff(x, axis=1), 1000.0, atol=1e-6)
    assert np.allclose(np.diff(y, axis=0), 1000.0, atol=1e-6)
    # 格心經緯度位於四個格角的中間。
    rows = tf.table_rows(products, "residence_density_by_site", group_kind="all")
    for r in rows[:50]:
        ix, iy = r["cell_ix"], r["cell_iy"]
        cx = 0.25 * (lon[iy, ix] + lon[iy, ix + 1] + lon[iy + 1, ix] + lon[iy + 1, ix + 1])
        cy = 0.25 * (lat[iy, ix] + lat[iy, ix + 1] + lat[iy + 1, ix] + lat[iy + 1, ix + 1])
        assert r["lon_center_deg"] == pytest.approx(cx, abs=1e-9)
        assert r["lat_center_deg"] == pytest.approx(cy, abs=1e-9)


def test_meter_total_particle_hours_equal_lonlat_grid_total():
    # 粒子留在網格內時，兩種網格的總停留時數相同。
    identity, results = tf.walk_results(n_time=60, step_m=500.0)
    meter = tf.run_accumulator(identity, results, config=METER_CFG)
    legacy = tf.run_accumulator(identity, results)
    m = meter.manifest["sites"][tf.SITE]["total_particle_hours"]
    assert m == legacy.manifest["sites"][tf.SITE]["total_particle_hours"]
    assert meter.summary_rows[0]["total_particle_hours"] == legacy.summary_rows[0]["total_particle_hours"]
    rows = tf.table_rows(meter, "residence_density_by_site", group_kind="all")
    assert sum(r["particle_hours"] for r in rows) == m
    assert not meter.manifest["warnings"] and not legacy.manifest["warnings"]


def test_meter_total_plus_outside_hours_equals_valid_samples():
    # 大步長會讓少數有效樣本落在 flow domain 外推範圍之外；網格內加網格外的總和與網格種類無關。
    identity, results = tf.walk_results(n_time=60)
    meter = tf.run_accumulator(identity, results, config=METER_CFG)
    legacy = tf.run_accumulator(identity, results)

    def total(products):
        site = products.manifest["sites"][tf.SITE]
        return site["total_particle_hours"] + site["hours_outside_grid"] + site["hours_outside_lookup"]

    assert total(meter) == total(legacy)


def test_meter_hdr_is_monotone_and_is_the_minimum_area_set():
    identity, results = tf.walk_results()
    products = tf.run_accumulator(identity, results, config=METER_CFG)
    hdr = sorted(tf.table_rows(products, "residence_hdr_by_site", group_kind="all"), key=lambda r: r["level"])
    assert [r["level"] for r in hdr] == [0.5, 0.75, 0.9]
    assert hdr[0]["area_km2"] < hdr[1]["area_km2"] < hdr[2]["area_km2"]
    assert (
        hdr[0]["density_threshold_fraction_per_km2"]
        >= hdr[1]["density_threshold_fraction_per_km2"]
        >= hdr[2]["density_threshold_fraction_per_km2"]
    )
    # 面積＝格數 × 1 km²。
    for r in hdr:
        assert r["area_km2"] == float(r["cell_count"])
        assert r["achieved_fraction"] >= r["level"] - 1e-12
    # 最小面積：依停留時數由高到低排序累加，剛好達到比例所需的格數。
    rows = tf.table_rows(products, "residence_density_by_site", group_kind="all")
    hours = np.array(sorted((r["particle_hours"] for r in rows), reverse=True), dtype=np.int64)
    cum = np.cumsum(hours)
    for r in hdr:
        need = int(np.searchsorted(cum, r["level"] * hours.sum(), side="left")) + 1
        assert r["cell_count"] == need
        # 少一格就不夠。
        assert cum[need - 2] < r["level"] * hours.sum() if need > 1 else True
    summary = products.summary_rows[0]
    assert summary["hdr50_area_km2"] == hdr[0]["area_km2"]
    assert summary["hdr75_area_km2"] == hdr[1]["area_km2"]
    assert summary["hdr90_area_km2"] == hdr[2]["area_km2"]


def test_meter_grid_sensitivity_columns_and_nesting():
    identity, results = tf.walk_results()
    products = tf.run_accumulator(identity, results, config=METER_CFG)
    s = products.summary_rows[0]
    for col in (
        "grid_sensitivity_hdr50_area_km2_0p50km",
        "grid_sensitivity_hdr50_area_km2_1p00km",
        "grid_sensitivity_hdr50_area_km2_2p00km",
        "grid_sensitivity_hdr90_area_km2_0p50km",
        "grid_sensitivity_hdr90_area_km2_1p00km",
        "grid_sensitivity_hdr90_area_km2_2p00km",
    ):
        assert col in s, col
    assert s["grid_sensitivity_hdr50_area_km2_1p00km"] == s["hdr50_area_km2"]
    assert s["grid_sensitivity_hdr90_area_km2_1p00km"] == s["hdr90_area_km2"]
    assert "grid_sensitivity_hdr50_area_km2_0p02" not in s
    # 巢狀網格：細格的 HDR 面積不會比粗格大很多（相同量級）。
    ratio = s["grid_sensitivity_hdr90_area_km2_0p50km"] / s["grid_sensitivity_hdr90_area_km2_2p00km"]
    assert 0.3 < ratio < 1.5
    # 粒子都留在網格內時，三種格寬的總停留時數相同，不出警告。
    identity2, results2 = tf.walk_results(step_m=500.0)
    assert not tf.run_accumulator(identity2, results2, config=METER_CFG).manifest["warnings"]


def test_meter_chunked_accumulation_is_bit_identical_to_one_shot():
    identity, results = tf.walk_results(n_time=90)
    cfg = FootprintConfig(
        grid_cell_km=1.0, lookback_snapshot_hours=(24, 72), snapshot_minimum_particle_count=10
    )
    per_unit = tf.run_accumulator(identity, results, config=cfg)
    rows_all, big = tf.concat_results(identity, results)
    one_shot = tf.run_accumulator(identity, [("all", rows_all, big)], config=cfg)
    tf._assert_products_identical(per_unit, one_shot)
    tf._assert_products_identical(per_unit, tf.run_accumulator(identity, list(reversed(results)), config=cfg))


def test_meter_snapshots_and_sea_mask_use_the_same_grid_and_keep_land_empty():
    identity, results = tf.walk_results(n_time=90, step_m=500.0)
    east_x = 25e3
    lon_w, lat_s = fx.linear_projector(np.array(["x"]), np.array([east_x]), np.array([-40e3]))
    lon_e, lat_n = fx.linear_projector(np.array(["x"]), np.array([60e3]), np.array([40e3]))
    land = box(float(lon_w[0]), float(lat_s[0]), float(lon_e[0]), float(lat_n[0]))
    cfg = FootprintConfig(
        grid_cell_km=1.0,
        lookback_snapshot_hours=(24,),
        snapshot_minimum_particle_count=10,
        snapshot_smoothing_km=3.0,
    )
    products = tf.run_accumulator(identity, results, config=cfg, land_lonlat=land)
    mask = {
        (r["cell_ix"], r["cell_iy"]): r for r in products.tables["footprint_sea_mask_by_site"].to_pylist()
    }
    grid = products.tables["footprint_grid_by_site"].to_pylist()[0]
    assert len(mask) == grid["nx"] * grid["ny"]
    cells = [
        r
        for r in products.tables["lookback_snapshot_density_by_site"].to_pylist()
        if r["snapshot_hours"] == 24
    ]
    assert cells
    assert sum(r["fraction_of_active"] for r in cells) == pytest.approx(1.0, abs=1e-9)
    assert all(r["cell_area_km2"] == 1.0 for r in cells)
    for r in cells:
        assert mask[(r["cell_ix"], r["cell_iy"])]["is_sea_cell"]
    land_cells = [k for k, m in mask.items() if not m["is_sea_cell"]]
    assert land_cells
    assert not (set(land_cells) & {(r["cell_ix"], r["cell_iy"]) for r in cells})
    sigma = products.manifest["sites"][tf.SITE]["sea_mask"]["smoothing_sigma_cells_lat_lon"]
    assert sigma == [3.0, 3.0]


def test_meter_depth_fraction_is_evaluated_at_cell_centres():
    from lagrangian_backtracking.native_runner import bathymetry

    identity, results = tf.walk_results(members=4, arrivals=2)
    # 水深場：全部淺於 200 m。
    lon = np.array([119.0, 123.0, 119.0, 123.0])
    lat = np.array([23.0, 23.0, 27.0, 27.0])
    field = bathymetry.BathymetryField(lon, lat, np.full(4, 50.0), np.array([[0, 1, 2], [1, 3, 2]]))
    products = tf.run_accumulator(identity, results, config=METER_CFG, bathymetry_by_site={tf.SITE: field})
    assert products.summary_rows[0]["hours_fraction_depth_below_threshold"] == 1.0
    assert products.summary_rows[0]["hours_fraction_depth_missing"] == 0.0


# ── 繪圖：公尺網格的後處理目錄 ───────────────────────────────────────────────


def _write_meter_footprint_tables(postprocess_dir: Path) -> dict:
    identity, units = _two_site_world()
    cfg = FootprintConfig(
        grid_cell_km=1.0, lookback_snapshot_hours=(24, 72, 168, 336), snapshot_minimum_particle_count=20
    )
    specs = {s: FootprintSiteSpec(flow_domain=DOMAIN, flow_open_boundary=OPEN_BOUNDARY) for s in SITES}
    acc = FootprintAccumulator(identity, config=cfg, specs=specs, lonlat_projector=fx.linear_projector)
    rng = np.random.default_rng(4)
    for key, rows in units:
        site_scale = 2500.0 if key.startswith("site_a") else 5000.0
        result = fx.random_walk_unit(rng, identity, rows, n_time=200, step_m=site_scale, east_edge_m=50e3)
        acc.add_batch(key, rows, result)
    products = acc.finalize(
        receptor_centers_xy={s: RECEPTORS[0] for s in SITES}, physical_case_id="case-test"
    )
    for stem, table in products.tables.items():
        pq.write_table(table, postprocess_dir / f"{stem}.parquet")
    pq.write_table(
        summary_table(products.summary_rows), postprocess_dir / "footprint_summary_by_site.parquet"
    )
    manifest_path = postprocess_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["footprint"] = products.manifest
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return products.manifest


@pytest.fixture
def meter_inputs(tmp_path: Path):
    postprocess_dir = _write_synthetic_postprocess_dir(tmp_path, sites=SITES, ensemble_member_count=4)
    _write_meter_footprint_tables(postprocess_dir)
    coastline = _write_synthetic_coastline(tmp_path)
    isobaths = _write_isobath_geojson(tmp_path / "isobaths.geojson")
    return postprocess_dir, coastline, isobaths


def test_residence_figure_from_meter_grid_tables(tmp_path, meter_inputs):
    postprocess_dir, coastline, isobaths = meter_inputs
    out = _build(tmp_path, postprocess_dir, coastline, isobaths)
    for site in SITES:
        for stem in native_figures.FOOTPRINT_FIGURE_STEMS:
            assert (out / site / f"{stem}.png").stat().st_size > 0, stem
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    scales = manifest["footprint_figures"]["residence_color_scale"]
    shared = scales["variants"]["source_footprint_residence"]
    assert shared["upper_limit"] >= shared["raw_max"] > shared["upper_limit"] / 10.0
    assert shared["lower_limit"] == pytest.approx(shared["upper_limit"] * 1e-4)
    caption = manifest["output_png_sha256"]["site_a/source_footprint_residence.png"]["caption"]
    assert "1 km" in caption and "經緯度規則網格" not in caption
    # 色階越界統計：面積恆為 1 km²，所以越界質量比例 ＝ 越界格的密度和 ÷ 全部密度和。
    figs = manifest["footprint_figures"]["figures"]["site_a/source_footprint_residence.png"]
    stats = figs["limit_stats"]
    assert 0.0 <= stats["mass_fraction_below_lower_limit"] <= 1.0
    assert stats["cells_above_upper_limit"] == 0


def test_figure_data_reports_meter_grid_geometry(tmp_path, meter_inputs):
    postprocess_dir, _coastline, _isobaths = meter_inputs
    data, _paths = native_figures._load_footprint_figure_inputs(postprocess_dir, list(SITES))
    fp = data["site_a"]
    assert fp.is_meter_grid and fp.cell_km == 1.0
    ny, nx = fp.density_pct_per_km2.shape
    assert fp.grid_corners_lonlat.shape == (ny + 1, nx + 1, 2)
    lon_c, lat_c = fp.center_coords()
    assert lon_c.shape == lat_c.shape == (ny, nx)
    assert np.all(fp.cell_area_km2_rows() == 1.0) and fp.cell_area_km2_rows().shape == (ny, 1)
    # 密度對 1 km² 面積的積分為 100%。
    assert float(fp.density_pct_per_km2.sum()) == pytest.approx(100.0, abs=1e-9)
    assert dataclasses.is_dataclass(fp)
