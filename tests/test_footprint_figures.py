"""WP-38：停留時數密度、回溯快照、邊界弧長分布三種新圖的測試（合成資料）。"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import wp38_fixture as fx
from shapely.geometry import LineString, box
from test_native_figures import _write_synthetic_coastline, _write_synthetic_postprocess_dir

from lagrangian_backtracking.native_runner import bathymetry
from lagrangian_backtracking.native_runner import figures as native_figures
from lagrangian_backtracking.native_runner.footprint import (
    FootprintAccumulator,
    FootprintConfig,
    FootprintSiteSpec,
    summary_table,
)
from lagrangian_backtracking.native_runner.particle_input import ParticleIdentityTable

SITES = ("site_a", "site_b")
DOMAIN = box(-50e3, -40e3, 50e3, 40e3)
OPEN_BOUNDARY = LineString([(-50e3, -40e3), (50e3, -40e3), (50e3, 40e3), (-50e3, 40e3)])
RECEPTORS = ((500.0, 500.0), (1000.0, 700.0))


def _two_site_world():
    worlds = [
        fx.build_world(
            site=site, velocities=(-0.001, -0.002), members=4, receptors=2, arrivals=4, receptor_xy=RECEPTORS
        )
        for site in SITES
    ]
    names = [f.name for f in dataclasses.fields(ParticleIdentityTable)]
    arrays = {
        name: np.concatenate([getattr(w[0], name) for w in worlds])
        for name in names
        if isinstance(getattr(worlds[0][0], name), np.ndarray)
    }
    identity = ParticleIdentityTable(
        **arrays,
        experiment_case_id=worlds[0][0].experiment_case_id,
        master_seed=worlds[0][0].master_seed,
        random_stream_id=worlds[0][0].random_stream_id,
    )
    offset = len(worlds[0][0])
    units = list(worlds[0][1]) + [(key, rows + offset) for key, rows in worlds[1][1]]
    return identity, units


def _write_footprint_tables(postprocess_dir: Path, *, snapshot_hours=(24, 72, 168, 336)) -> dict:
    identity, units = _two_site_world()
    cfg = FootprintConfig(lookback_snapshot_hours=snapshot_hours, snapshot_minimum_particle_count=20)
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


def _write_isobath_geojson(path: Path) -> Path:
    # 兩排節點（南北各一排）、六個經度；水深由西向東增加，50／100／200／1000 公尺等深線都落在流場範圍內。
    lons = np.array([120.3, 120.55, 120.7, 121.0, 121.3, 121.7])
    depths = np.array([0.0, 40.0, 90.0, 180.0, 900.0, 1500.0])
    lon = np.concatenate([lons, lons])
    lat = np.concatenate([np.full(6, 24.4), np.full(6, 25.6)])
    depth = np.concatenate([depths, depths])
    triangles = []
    for i in range(5):
        triangles.append([i, i + 1, 6 + i + 1])
        triangles.append([i, 6 + i + 1, 6 + i])
    field = bathymetry.BathymetryField(lon, lat, depth, np.array(triangles))
    payload = bathymetry.build_isobath_geojson({s: field for s in SITES}, (50, 100, 200, 1000))
    bathymetry.dump_geojson(payload, path)
    return path


@pytest.fixture
def footprint_inputs(tmp_path: Path):
    postprocess_dir = _write_synthetic_postprocess_dir(tmp_path, sites=SITES, ensemble_member_count=4)
    manifest = _write_footprint_tables(postprocess_dir)
    coastline = _write_synthetic_coastline(tmp_path)
    isobaths = _write_isobath_geojson(tmp_path / "isobaths.geojson")
    return postprocess_dir, coastline, isobaths, manifest


def _build(
    tmp_path: Path, postprocess_dir: Path, coastline: Path, isobaths: Path | None, *, text="full", name="out"
):
    config = native_figures.NativeFiguresConfig(
        postprocess_dir=postprocess_dir,
        coastline_path=coastline,
        output_dir=tmp_path / name,
        dpi=60,
        figure_text=text,
        bathymetry_path=isobaths,
    )
    native_figures.build_native_figures(config, command="lbt native-figures --test")
    return tmp_path / name


def test_footprint_figures_are_produced_for_each_site(tmp_path, footprint_inputs):
    postprocess_dir, coastline, isobaths, _ = footprint_inputs
    out = _build(tmp_path, postprocess_dir, coastline, isobaths)
    for site in SITES:
        for stem in (*native_figures.FIGURE_STEMS, *native_figures.FOOTPRINT_FIGURE_STEMS):
            assert (out / site / f"{stem}.png").stat().st_size > 0, stem
            assert (out / site / f"{stem}.svg").stat().st_size > 0, stem
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["figure_stems"] == list(native_figures.FIGURE_STEMS)
    assert manifest["footprint_figure_stems"] == list(native_figures.FOOTPRINT_FIGURE_STEMS)
    block = manifest["footprint_figures"]
    scales = block["residence_color_scale"]
    assert scales["unit"] == "% 停留時數／km²"
    assert scales["norm"] == "log"
    assert scales["lower_limit_fraction_of_upper_limit"] == 1e-4
    shared = scales["variants"]["source_footprint_residence"]
    per_site = scales["variants"]["source_footprint_residence_site_scale"]
    assert shared["scale"] == "shared_across_sites" and per_site["scale"] == "per_site"
    assert shared["upper_limit"] == pytest.approx(max(per_site["upper_limit_by_site"].values()))
    assert per_site["upper_limit_by_site"]["site_a"] != per_site["upper_limit_by_site"]["site_b"]
    for key in ("site_a/source_footprint_residence.png", "site_b/lookback_snapshot.png"):
        assert key in manifest["output_png_sha256"]
        assert key in block["figures"]
    assert manifest["footprint_figures"]["bathymetry"]["status"] == "loaded"
    for site in SITES:
        rec = block["bathymetry"]["by_site"][site]
        assert rec["status"] == "drawn" and rec["levels_m"] == [50.0, 100.0, 200.0, 1000.0]


def test_missing_bathymetry_still_draws_and_records_reason(tmp_path, footprint_inputs):
    postprocess_dir, coastline, _isobaths, _ = footprint_inputs
    out = _build(tmp_path, postprocess_dir, coastline, None)
    assert (out / "site_a" / "source_footprint_residence.png").is_file()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    bathy = manifest["footprint_figures"]["bathymetry"]
    assert bathy["status"] == "not_provided"
    assert bathy["by_site"]["site_a"]["status"] == "not_drawn"
    assert "未提供" in bathy["by_site"]["site_a"]["reason"]
    caption = manifest["output_png_sha256"]["site_a/source_footprint_residence.png"]["caption"]
    assert "未繪等深線" in caption

    out2 = _build(tmp_path, postprocess_dir, coastline, tmp_path / "does-not-exist.geojson", name="out2")
    manifest2 = json.loads((out2 / "manifest.json").read_text(encoding="utf-8"))
    assert manifest2["footprint_figures"]["bathymetry"]["status"] == "file_not_found"


def test_malformed_bathymetry_fails_closed(tmp_path, footprint_inputs):
    postprocess_dir, coastline, _isobaths, _ = footprint_inputs
    bad = tmp_path / "bad.geojson"
    bad.write_text(json.dumps({"type": "FeatureCollection", "features": []}), encoding="utf-8")
    with pytest.raises(native_figures.FiguresError, match="CRS84"):
        _build(tmp_path, postprocess_dir, coastline, bad)


def test_legacy_directory_without_footprint_tables_is_unchanged(tmp_path):
    postprocess_dir = _write_synthetic_postprocess_dir(tmp_path, sites=SITES)
    coastline = _write_synthetic_coastline(tmp_path)
    out = _build(tmp_path, postprocess_dir, coastline, None)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert "footprint_figures" not in manifest and "footprint_figure_stems" not in manifest
    assert not (out / "site_a" / "source_footprint_residence.png").exists()
    assert "WP-38" not in (out / "README.md").read_text(encoding="utf-8")


def test_captions_cover_definition_normalisation_measured_m_and_wording(tmp_path, footprint_inputs):
    postprocess_dir, coastline, isobaths, _ = footprint_inputs
    out = _build(tmp_path, postprocess_dir, coastline, isobaths)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    for site in SITES:
        residence = manifest["output_png_sha256"][f"{site}/source_footprint_residence.png"]["caption"]
        site_scale = manifest["output_png_sha256"][f"{site}/source_footprint_residence_site_scale.png"][
            "caption"
        ]
        snapshot = manifest["output_png_sha256"][f"{site}/lookback_snapshot.png"]["caption"]
        arc = manifest["output_png_sha256"][f"{site}/boundary_departure_along_arc.png"]["caption"]
        for text in (residence, site_scale, snapshot, arc):
            assert "M=4" in text  # 實測系集成員數，不是寫死的常數
            assert "物理案例" in text
            assert "BayTrace" not in text and "baytrace" not in text.lower()
            assert ".parquet" not in text and "_by_site" not in text
        assert "逐小時" in residence and "仍在漂流" in residence and "不再計入" in residence
        assert "% 停留時數／km²" in residence and "積分為 100%" in residence
        assert "0.02°" in residence and "高密度區間（Highest Density Region, HDR）" in residence
        assert "一維" in residence and "二維核密度不適用" in residence
        assert "跨站共用色階" in residence and "本站自訂色階" in site_scale
        assert "回溯" in snapshot and "仍在漂流" in snapshot
        assert "沿流場範圍開放邊界折線" in arc
    captions_md = (out / "captions.md").read_text(encoding="utf-8")
    for stem in native_figures.FOOTPRINT_FIGURE_STEMS:
        assert f"### `{stem}.png`" in captions_md


def test_figure_text_none_omits_title_but_keeps_legend_and_panel_titles(
    tmp_path, footprint_inputs, monkeypatch
):
    postprocess_dir, coastline, isobaths, _ = footprint_inputs
    import matplotlib.figure
    import matplotlib.text

    texts: dict[str, list[str]] = {}
    original = matplotlib.figure.Figure.savefig

    def spy(self, *args, **kwargs):
        texts.setdefault("all", []).extend(t.get_text() for t in self.findobj(matplotlib.text.Text))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", spy)
    _build(tmp_path, postprocess_dir, coastline, isobaths, text="none", name="none")
    joined = "\n".join(texts["all"])
    assert "停留時數密度｜" not in joined and "回溯快照的條件式來源足跡｜" not in joined
    assert "穿越位置沿開放邊界的分布｜" not in joined
    assert "高密度區間 50%" in joined and "流場範圍開放邊界（終止邊界）" in joined
    assert "回溯 72 小時" in joined  # 面板標題保留
    texts.clear()
    _build(tmp_path, postprocess_dir, coastline, isobaths, text="full", name="full")
    joined_full = "\n".join(texts["all"])
    assert "停留時數密度｜" in joined_full and "回溯快照的條件式來源足跡｜" in joined_full


def test_input_files_unchanged_and_bathymetry_hashed(tmp_path, footprint_inputs):
    postprocess_dir, coastline, isobaths, _ = footprint_inputs
    out = _build(tmp_path, postprocess_dir, coastline, isobaths)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["inputs_sha256_before"] == manifest["inputs_sha256_after"]
    assert "bathymetry" in manifest["inputs_sha256_before"]
    assert "residence_density_by_site.parquet" in manifest["inputs_sha256_before"]


def test_missing_companion_table_fails_closed(tmp_path, footprint_inputs):
    postprocess_dir, coastline, isobaths, _ = footprint_inputs
    (postprocess_dir / "boundary_edge_intervals_by_site.parquet").unlink()
    with pytest.raises(native_figures.FiguresError, match="配套輸入表"):
        _build(tmp_path, postprocess_dir, coastline, isobaths)


def test_corner_name_uses_chinese_order():
    assert native_figures._corner_name_zh("south-east") == "東南角"
    assert native_figures._corner_name_zh("east-north") == "東北角"
    assert native_figures._corner_name_zh("north-west") == "西北角"


def test_snapshot_panel_with_too_few_particles_is_annotated_not_estimated(
    tmp_path, footprint_inputs, monkeypatch
):
    postprocess_dir, coastline, isobaths, _ = footprint_inputs
    import matplotlib.figure
    import matplotlib.text

    seen: list[str] = []
    original = matplotlib.figure.Figure.savefig

    def spy(self, *args, **kwargs):
        seen.extend(t.get_text() for t in self.findobj(matplotlib.text.Text))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", spy)
    _build(tmp_path, postprocess_dir, coastline, isobaths)
    # 合成資料的隨機終止使 336 小時面板只剩少數粒子或沒有粒子。
    manifest_text = "\n".join(seen)
    assert "仍在漂流粒子數不足（n=" in manifest_text or "回溯 336 小時" in manifest_text
    # 快照不再有 KDE 的「範圍外未繪」註記。
    assert "範圍外" not in manifest_text and "未繪" not in manifest_text


def _base_map_pixels(land_geometries, *, domain):
    """只畫底圖（陸地、邊界），回傳渲染後的 RGBA 陣列與 (x0, x1, y0, y1) 範圍。"""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fp = native_figures.FootprintSiteFigureData(
        site_id="t",
        grid={"lon0_deg": 0.0, "lat0_deg": 0.0, "nx": 1, "ny": 1, "resolution_deg": 1.0},
        density_pct_per_km2=np.zeros((1, 1)),
        hdr_threshold_pct_per_km2={},
        hdr_area_km2={},
        snapshots=(),
        arc_bins=(),
        edge_intervals=(),
        domain_lonlat=np.asarray(domain, dtype=float),
        open_boundary_parts=(),
        receptor_center=None,
        summary={},
    )
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    try:
        extent, _ = native_figures._draw_footprint_base_map(
            ax,
            fp,
            land_geometries,
            {},
            native_figures._Labeler(False),
            None,
            site_label="t",
        )
        fig.canvas.draw()
        pixels = np.asarray(fig.canvas.buffer_rgba()).copy()
        bbox = ax.get_window_extent()
    finally:
        plt.close(fig)
    return pixels, extent, bbox


def _pixel_at(pixels, extent, bbox, lon, lat):
    h = pixels.shape[0]
    fx = (lon - extent[0]) / (extent[1] - extent[0])
    fy = (lat - extent[2]) / (extent[3] - extent[2])
    col = int(bbox.x0 + fx * bbox.width)
    row = int(h - (bbox.y0 + fy * bbox.height))
    return tuple(int(v) for v in pixels[row, col, :3])


def test_land_crossing_the_frame_is_still_filled_inside_the_frame():
    """陸地多邊形跨出圖框時，框內部分仍須填色；內環（湖泊）與獨立小島也要正確。"""

    from shapely.geometry import MultiPolygon, Polygon

    land_rgb = tuple(int(native_figures._LAND_COLOR[i : i + 2], 16) for i in (1, 3, 5))
    domain = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    # 大陸塊：從框外（x<0、y>10）延伸進框內；內環是框內的湖，另有框內小島與整個在框外的島
    mainland = Polygon(
        [(-20.0, -20.0), (4.0, -20.0), (4.0, 30.0), (-20.0, 30.0)],
        holes=[[(1.0, 4.0), (3.0, 4.0), (3.0, 6.0), (1.0, 6.0)]],
    )
    islet = Polygon([(7.0, 7.0), (8.0, 7.0), (8.0, 8.0), (7.0, 8.0)])
    far_away = Polygon([(100.0, 100.0), (101.0, 100.0), (101.0, 101.0), (100.0, 101.0)])
    pixels, extent, bbox = _base_map_pixels(
        [MultiPolygon([mainland, islet]), far_away], domain=domain
    )
    assert _pixel_at(pixels, extent, bbox, 2.0, 8.0) == land_rgb  # 框內、跨界大陸塊
    assert _pixel_at(pixels, extent, bbox, 0.5, 9.5) == land_rgb  # 靠近跨出圖框的角落
    assert _pixel_at(pixels, extent, bbox, 7.5, 7.5) == land_rgb  # 框內小島
    assert _pixel_at(pixels, extent, bbox, 2.0, 5.0) != land_rgb  # 湖（內環）不填
    assert _pixel_at(pixels, extent, bbox, 6.0, 3.0) != land_rgb  # 海
