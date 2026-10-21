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
    tmp_path: Path,
    postprocess_dir: Path,
    coastline: Path,
    isobaths: Path | None,
    *,
    text="full",
    name="out",
    isobath_levels=native_figures.ISOBATH_LEVELS_DEFAULT,
    receptor_config: Path | None = None,
    allow_unanchored: bool = True,
):
    config = native_figures.NativeFiguresConfig(
        postprocess_dir=postprocess_dir,
        coastline_path=coastline,
        output_dir=tmp_path / name,
        dpi=60,
        figure_text=text,
        bathymetry_path=isobaths,
        isobath_levels_m=tuple(isobath_levels),
        receptor_config_path=receptor_config,
        allow_unanchored_receptor=receptor_config is None and allow_unanchored,
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
    assert scales["unit"] == "% 停留時數/km²"
    assert scales["norm"] == "log"
    assert scales["lower_limit_fraction_of_upper_limit"] == 1e-4
    shared = scales["variants"]["source_footprint_residence"]
    per_site = scales["variants"]["source_footprint_residence_site_scale"]
    assert shared["scale"] == "shared_across_sites" and per_site["scale"] == "per_site"
    assert shared["raw_max"] == pytest.approx(max(per_site["raw_max_by_site"].values()))
    assert per_site["raw_max_by_site"]["site_a"] != per_site["raw_max_by_site"]["site_b"]
    # 上限是最大值向上取整到 10 的整數次方，下限是上限的 1/10000
    for upper, lower in (
        (shared["upper_limit"], shared["lower_limit"]),
        *((per_site["upper_limit_by_site"][s], per_site["lower_limit_by_site"][s]) for s in SITES),
    ):
        assert np.log10(upper) == pytest.approx(round(np.log10(upper)))
        assert lower == pytest.approx(upper * 1e-4)
    assert shared["upper_limit"] >= shared["raw_max"] > shared["upper_limit"] / 10.0
    for key in ("site_a/source_footprint_residence.png", "site_b/lookback_snapshot.png"):
        assert key in manifest["output_png_sha256"]
        assert key in block["figures"]
    assert manifest["footprint_figures"]["bathymetry"]["status"] == "loaded"
    for site in SITES:
        rec = block["bathymetry"]["by_site"][site]
        # 預設只畫 200 m（陸棚邊緣）與 1000 m；其他水深不畫，免得干擾高密度區間
        assert rec["status"] == "drawn" and rec["levels_m"] == [200.0, 1000.0]
    assert block["bathymetry"]["requested_levels_m"] == [200.0, 1000.0]


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
        assert "跨站共用色階" not in residence and "本站自訂色階" in site_scale
        assert "跨站共用色階" not in site_scale
        assert "跨站" not in residence and "跨站" not in site_scale
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
    assert "高密度區間 50%" in joined and "流場範圍邊界" in joined
    assert "回溯 72 h" in joined  # 面板標題保留
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
    assert "仍在漂流虛擬質點數不足（n=" in manifest_text or "回溯 336 h" in manifest_text
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
        extent, _ = native_figures._draw_footprint_base_map(ax, fp, land_geometries, {})
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
    pixels, extent, bbox = _base_map_pixels([MultiPolygon([mainland, islet]), far_away], domain=domain)
    assert _pixel_at(pixels, extent, bbox, 2.0, 8.0) == land_rgb  # 框內、跨界大陸塊
    assert _pixel_at(pixels, extent, bbox, 0.5, 9.5) == land_rgb  # 靠近跨出圖框的角落
    assert _pixel_at(pixels, extent, bbox, 7.5, 7.5) == land_rgb  # 框內小島
    assert _pixel_at(pixels, extent, bbox, 2.0, 5.0) != land_rgb  # 湖（內環）不填
    assert _pixel_at(pixels, extent, bbox, 6.0, 3.0) != land_rgb  # 海


# ═══════════════════════════════════════════════════════════════════════════
# 改版（c）：色條與地圖同高、座標軸對齊刻度、等深線、受體權威來源、不寫站名
# ═══════════════════════════════════════════════════════════════════════════


def _count_tick_label_overlaps(fig, axes) -> int:
    """畫完後，所有座標軸可見刻度文字的外框兩兩相交的對數（含相鄰面板的首末刻度）。"""

    boxes = []
    renderer = fig.canvas.get_renderer()
    for ax in axes:
        for tick_label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
            if tick_label.get_visible() and tick_label.get_text():
                boxes.append(tick_label.get_window_extent(renderer))
    return sum(1 for i in range(len(boxes)) for j in range(i + 1, len(boxes)) if boxes[i].overlaps(boxes[j]))


def _describe_figure(fig) -> dict:
    """在存檔前量測實際版面：地圖軸、色條軸的顯示座標、座標軸範圍與刻度、星號與等深線。"""

    import matplotlib.colors
    import matplotlib.text

    fig.canvas.draw()
    visible = [a for a in fig.axes if a.get_visible()]
    cbar_axes = [a for a in visible if a.get_position().width < 0.05 and a.get_position().height > 0.2]
    maps = [a for a in visible if a not in cbar_axes]
    info: dict = {
        "n_maps": len(maps),
        "cbar": None,
        "twin": False,
        "stars": [
            tuple(float(v) for v in ln.get_xydata()[0])
            for a in maps
            for ln in a.lines
            if ln.get_marker() == "*"
        ],
        "texts": [t.get_text() for t in fig.findobj(matplotlib.text.Text)],
        "isobaths": [
            (ln.get_linewidth(), ln.get_linestyle(), ln.get_zorder())
            for a in maps
            for ln in a.lines
            if ln.get_color() == native_figures._ISOBATH_COLOR
        ],
        "collections": [(type(c).__name__, c.get_zorder()) for a in maps for c in a.collections],
        "tick_label_overlaps": _count_tick_label_overlaps(fig, maps),
        "axes_texts": [t.get_text() for a in maps for t in a.texts],
        "legend_texts": [
            t.get_text()
            for legend in [*fig.legends, *(a.get_legend() for a in maps)]
            if legend is not None
            for t in legend.get_texts()
        ],
        "sampling_circles": [
            np.asarray(ln.get_xydata())
            for a in maps
            for ln in a.lines
            if ln.get_color() == native_figures._SAMPLING_RANGE_STYLE["color"] and len(ln.get_xydata()) > 5
        ],
        "release_polygons": [
            p for a in maps for p in a.patches if type(p).__name__ == "Polygon" and p.get_zorder() == 7
        ],
    }
    ref = min(maps, key=lambda a: (round(a.get_window_extent().y0), round(a.get_window_extent().x0)))
    info.update(
        xlim=tuple(ref.get_xlim()),
        ylim=tuple(ref.get_ylim()),
        xticks=[float(v) for v in ref.get_xticks()],
        yticks=[float(v) for v in ref.get_yticks()],
        xlabels=[t.get_text() for t in ref.get_xticklabels()],
        ylabels=[t.get_text() for t in ref.get_yticklabels()],
    )
    if len(maps) == 2 and maps[0].get_position().bounds == maps[1].get_position().bounds:
        right = maps[1]
        info.update(
            twin=True,
            arc_label_boxes=[t.get_window_extent() for t in maps[0].texts],
            arc_axes_top_px=float(maps[0].get_window_extent().y1),
            right_ylim=tuple(right.get_ylim()),
            right_yticks=[float(v) for v in right.get_yticks()],
            right_ylabels=[t.get_text() for t in right.get_yticklabels()],
            xlim=tuple(maps[0].get_xlim()),
            ylim=tuple(maps[0].get_ylim()),
            xticks=[float(v) for v in maps[0].get_xticks()],
            yticks=[float(v) for v in maps[0].get_yticks()],
            xlabels=[t.get_text() for t in maps[0].get_xticklabels()],
            ylabels=[t.get_text() for t in maps[0].get_yticklabels()],
        )
    if cbar_axes:
        cax = cbar_axes[0]
        top = max(a.get_window_extent().y1 for a in maps)
        bottom = min(a.get_window_extent().y0 for a in maps)
        box = cax.get_window_extent()
        info["cbar"] = {
            "top_diff": float(box.y1 - top),
            "bottom_diff": float(box.y0 - bottom),
            "ylim": tuple(cax.get_ylim()),
            "ticks": [float(v) for v in cax.get_yticks()],
            "labels": [t.get_text() for t in cax.get_yticklabels()],
        }
    return info


def _capture_figures(monkeypatch) -> list[dict]:
    import matplotlib.figure

    described: list[dict] = []
    original = matplotlib.figure.Figure.savefig

    def spy(self, *args, **kwargs):
        if not getattr(self, "_wp38_described", False):
            self._wp38_described = True
            described.append(_describe_figure(self))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", spy)
    return described


def _wp38_colorbar_figs(described: list[dict]) -> list[dict]:
    """WP-38 的有色條圖（停留時數密度、快照）：色條刻度是 10 的次方；舊圖的色條不是。"""

    return [
        d
        for d in described
        if d["cbar"] is not None and d["cbar"]["labels"] and d["cbar"]["labels"][0].startswith("$10^{")
    ]


def _footprint_figs(described: list[dict]) -> list[dict]:
    """只留 WP-38 的圖（有色條的停留時數／快照圖，以及雙縱軸的弧長分布圖）。"""

    return [*_wp38_colorbar_figs(described), *(d for d in described if d["twin"])]


def test_colorbar_height_equals_map_axes_height_including_shared_snapshot_bar(
    tmp_path, footprint_inputs, monkeypatch
):
    postprocess_dir, coastline, isobaths, _ = footprint_inputs
    described = _capture_figures(monkeypatch)
    out = _build(tmp_path, postprocess_dir, coastline, isobaths)
    with_cbar = _wp38_colorbar_figs(described)
    assert len(with_cbar) == 3 * len(SITES)  # 每站：兩張停留時數密度＋一張快照
    assert sorted(d["n_maps"] for d in with_cbar) == [1] * (2 * len(SITES)) + [4] * len(SITES)
    tol = native_figures.COLORBAR_ALIGNMENT_TOLERANCE_PX
    for d in with_cbar:
        assert abs(d["cbar"]["top_diff"]) <= tol
        assert abs(d["cbar"]["bottom_diff"]) <= tol
    # manifest 記錄的量測值與獨立量到的一致，且同樣在容差內
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    records = manifest["footprint_figures"]["figures"]
    measured = [r["layout"] for k, r in records.items() if "boundary_departure" not in k]
    assert len(measured) == 3 * len(SITES)
    for layout in measured:
        assert abs(layout["top_diff_px"]) <= tol and abs(layout["bottom_diff_px"]) <= tol
        assert layout["colorbar_top_px"] > layout["colorbar_bottom_px"]
    assert manifest["footprint_figures"]["colorbar_alignment_tolerance_px"] == tol


def test_axes_limits_equal_first_and_last_ticks_with_all_ends_labelled(
    tmp_path, footprint_inputs, monkeypatch
):
    postprocess_dir, coastline, isobaths, _ = footprint_inputs
    described = _capture_figures(monkeypatch)
    _build(tmp_path, postprocess_dir, coastline, isobaths)
    figs = _footprint_figs(described)
    assert len(figs) == 4 * len(SITES)
    for d in figs:
        assert d["xlim"] == (d["xticks"][0], d["xticks"][-1])
        assert d["ylim"] == (d["yticks"][0], d["yticks"][-1])
        assert len(d["xticks"]) >= 2 and len(d["yticks"]) >= 2
        assert d["xlabels"][0] and d["xlabels"][-1] and d["ylabels"][0] and d["ylabels"][-1]
        if d["twin"]:  # 弧長分布圖：三條軸都從 0 起算，右軸 0–100%
            assert d["xticks"][0] == 0.0 and d["yticks"][0] == 0.0
            assert d["right_ylim"] == (0.0, 100.0)
            assert d["right_yticks"][0] == 0.0 and d["right_yticks"][-1] == 100.0
            assert d["right_ylabels"][0] and d["right_ylabels"][-1]
        else:  # 地圖：等間距、間距是 0.1／0.2／0.5／1 度，刻度格式如 121.60°E
            for ticks in (d["xticks"], d["yticks"]):
                step = np.diff(ticks)
                assert np.allclose(step, step[0])
                assert any(np.isclose(step[0], c) for c in native_figures._TICK_STEP_CHOICES_DEG)
            assert d["xlabels"][0].endswith("°E") and d["ylabels"][0].endswith("°N")


def test_map_extent_is_rounded_outward_and_contains_flow_domain(tmp_path, footprint_inputs):
    postprocess_dir, coastline, isobaths, _ = footprint_inputs
    config = native_figures.NativeFiguresConfig(
        postprocess_dir=postprocess_dir,
        coastline_path=coastline,
        output_dir=tmp_path / "unused",
        allow_unanchored_receptor=True,
    )
    fp = native_figures.load_native_figures_inputs(config).footprint["site_a"]
    for max_intervals in (native_figures._RESIDENCE_MAX_INTERVALS, native_figures._SNAPSHOT_MAX_INTERVALS):
        xt, yt = native_figures._site_axis_ticks(fp, max_intervals=max_intervals)
        lon, lat = fp.domain_lonlat[:, 0], fp.domain_lonlat[:, 1]
        assert xt[0] <= lon.min() and xt[-1] >= lon.max() and yt[0] <= lat.min() and yt[-1] >= lat.max()
        assert 4 <= len(xt) <= 7 and 4 <= len(yt) <= 7  # 每軸 4～7 個刻度
        assert len(xt) - 1 <= max_intervals or native_figures._TICK_STEP_CHOICES_DEG[-1] == round(
            xt[1] - xt[0], 10
        )
        assert native_figures._site_extent(fp, max_intervals=max_intervals) == (xt[0], xt[-1], yt[0], yt[-1])


def test_axis_ticks_helper_rounds_outward_to_step_multiples():
    ticks = native_figures._axis_ticks(121.63, 122.18, max_intervals=7)
    assert ticks[0] == pytest.approx(121.6) and ticks[-1] == pytest.approx(122.2)
    assert np.allclose(np.diff(ticks), 0.1)
    wide = native_figures._axis_ticks(119.4, 122.3, max_intervals=7)
    assert np.allclose(np.diff(wide), 0.5) and wide[0] == 119.0 and wide[-1] == 122.5
    quarter = native_figures._axis_ticks(120.0, 121.4, max_intervals=6)
    assert np.allclose(np.diff(quarter), 0.25) and quarter[0] == 120.0 and quarter[-1] == 121.5  # 0.25 度間距
    tiny = native_figures._axis_ticks(120.51, 120.58, max_intervals=6)  # 範圍太小：至少 4 個刻度
    assert len(tiny) == 4 and tiny[0] <= 120.51 and tiny[-1] >= 120.58
    exact = native_figures._axis_ticks(120.0, 121.0, max_intervals=10)
    assert exact[0] == 120.0 and exact[-1] == 121.0  # 剛好在刻度上就不多擴一格
    top, ticks_arc = native_figures._nice_ceiling_axis(412.0, max_intervals=10)
    assert top >= 412.0 and ticks_arc[0] == 0.0 and ticks_arc[-1] == top
    assert native_figures._nice_ceiling_axis(0.0, max_intervals=5)[0] == 1.0


def test_colorbar_end_ticks_equal_limits_are_decades_and_labelled(tmp_path, footprint_inputs, monkeypatch):
    postprocess_dir, coastline, isobaths, _ = footprint_inputs
    described = _capture_figures(monkeypatch)
    _build(tmp_path, postprocess_dir, coastline, isobaths)
    bars = [d["cbar"] for d in _wp38_colorbar_figs(described)]
    assert len(bars) == 3 * len(SITES)
    for bar in bars:
        vmin, vmax = bar["ylim"]
        assert bar["ticks"][0] == pytest.approx(vmin) and bar["ticks"][-1] == pytest.approx(vmax)
        assert vmax / vmin == pytest.approx(1e4) or vmax / vmin == pytest.approx(1e5)  # 停留時數密度 1e4；回溯快照 1e5
        exps = np.log10(bar["ticks"])
        assert np.allclose(exps, np.round(exps)) and np.allclose(np.diff(exps), 1.0)  # 每個 10 的次方一個刻度
        assert len(bar["ticks"]) == round(np.log10(vmax / vmin)) + 1
        assert all(label.startswith("$10^{") for label in bar["labels"])  # 兩端與中間都有顯示文字
        assert bar["labels"][0] and bar["labels"][-1]


def test_decade_limits_and_tick_text():
    assert native_figures._decade_limits(0.37) == (1e-4, 1.0)
    assert native_figures._decade_limits(1.0) == (1e-4, 1.0)  # 剛好是 10 的次方不再多進一位
    assert native_figures._decade_limits(1.0001) == (1e-3, 10.0)
    for raw in (1e-7, 3.3e-5, 0.999999999, 1.0 + 1e-12, 7.7, 1234.5):
        lo, hi = native_figures._decade_limits(raw)
        assert hi >= raw and lo == pytest.approx(hi * 1e-4)  # 上限一定 >= 資料最大值
    assert native_figures._decade_limits(0.0) == (1e-4, 1.0)
    assert native_figures._decade_formatter(1e-4) == "$10^{-4}$"
    assert native_figures._decade_formatter(100.0) == "$10^{2}$"


def test_limit_stats_count_cells_outside_the_limits_without_clipping():
    fp = native_figures.FootprintSiteFigureData(
        site_id="t",
        grid={
            "lon0_deg": 120.0,
            "lat0_deg": 24.0,
            "nx": 4,
            "ny": 1,
            "resolution_deg": 0.1,
            "total_particle_hours": 10,
        },
        density_pct_per_km2=np.zeros((1, 4)),
        hdr_threshold_pct_per_km2={},
        hdr_area_km2={},
        snapshots=(),
        arc_bins=(),
        edge_intervals=(),
        domain_lonlat=np.asarray([[120, 24], [121, 24], [121, 25]], dtype=float),
        open_boundary_parts=(),
        receptor_center=None,
        summary={},
    )
    density = np.array([[0.0, 1e-6, 1e-3, 1.0]])
    stats = native_figures._limit_stats(fp, density, 1e-4, 1.0)
    assert stats["cells_with_density"] == 3
    assert stats["cells_below_lower_limit"] == 1 and stats["cells_above_upper_limit"] == 0
    assert 0.0 < stats["mass_fraction_below_lower_limit"] < 1e-5
    over = native_figures._limit_stats(fp, np.array([[0.0, 0.0, 0.0, 5.0]]), 1e-4, 1.0)
    assert over["cells_above_upper_limit"] == 1 and over["mass_fraction_above_upper_limit"] == pytest.approx(
        1.0
    )


def test_manifest_records_cells_below_lower_limit_per_figure(tmp_path, footprint_inputs):
    postprocess_dir, coastline, isobaths, _ = footprint_inputs
    out = _build(tmp_path, postprocess_dir, coastline, isobaths)
    records = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["footprint_figures"]["figures"]
    for site in SITES:
        stats = records[f"{site}/source_footprint_residence.png"]["limit_stats"]
        assert {
            "cells_below_lower_limit",
            "mass_fraction_below_lower_limit",
            "cells_above_upper_limit",
        } <= set(stats)
        assert stats["cells_above_upper_limit"] == 0  # 上限取的是最大值向上取整
        for panel in records[f"{site}/lookback_snapshot.png"]["limit_stats_by_panel"]:
            assert panel["cells_above_upper_limit"] == 0


def test_isobaths_are_thin_gray_between_density_and_hdr_and_default_to_200_and_1000(
    tmp_path, footprint_inputs, monkeypatch
):
    postprocess_dir, coastline, isobaths, _ = footprint_inputs
    described = _capture_figures(monkeypatch)
    _build(tmp_path, postprocess_dir, coastline, isobaths)
    residence = [d for d in _wp38_colorbar_figs(described) if d["n_maps"] == 1]
    assert residence
    hdr_widths = [s["linewidth"] for s in native_figures._HDR_LINE_STYLES]
    assert min(hdr_widths) > native_figures._ISOBATH_LINEWIDTH * 2
    for d in residence:
        assert d["isobaths"], "合成水深應在範圍內畫出等深線"
        widths = {w for w, _style, _z in d["isobaths"]}
        assert widths == {native_figures._ISOBATH_LINEWIDTH}
        assert {style for _w, style, _z in d["isobaths"]} == {"-", "--"}  # 200 m 實線、1000 m 虛線
        iso_z = {z for _w, _s, z in d["isobaths"]}
        mesh_z = {z for name, z in d["collections"] if name == "QuadMesh"}
        contour_z = {z for name, z in d["collections"] if "Contour" in name or name == "LineCollection"}
        assert mesh_z and max(mesh_z) < min(iso_z)
        assert contour_z and max(iso_z) < min(contour_z)
    # 圖例交代水深與線型
    legend_texts = "\n".join("\n".join(d["texts"]) for d in residence)
    assert "等深線 200 m（實線）" in legend_texts and "等深線 1000 m（虛線）" in legend_texts


def test_isobath_levels_are_a_parameter(tmp_path, footprint_inputs, monkeypatch):
    postprocess_dir, coastline, isobaths, _ = footprint_inputs
    out = _build(tmp_path, postprocess_dir, coastline, isobaths, isobath_levels=(100.0,))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    for site in SITES:
        assert manifest["footprint_figures"]["bathymetry"]["by_site"][site]["levels_m"] == [100.0]
    assert manifest["config"]["isobath_levels_m"] == [100.0]
    with pytest.raises(native_figures.FiguresError, match="isobath_levels_m"):
        native_figures.NativeFiguresConfig(
            postprocess_dir=postprocess_dir,
            coastline_path=coastline,
            output_dir=tmp_path / "x",
            isobath_levels_m=(-5.0,),
        )


def _write_release_config(path: Path, anchors: dict[str, tuple[float, float]]) -> Path:
    lines = ["study_sites:"]
    for site, (lon, lat) in anchors.items():
        lines += [
            f"- study_site_id: {site}",
            "  anchor_lonlat:",
            f"  - {lon}",
            f"  - {lat}",
            "  receptor_core_radius_m: 12500.0",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_receptor_star_comes_from_release_config_anchor_not_release_point_mean(
    tmp_path, footprint_inputs, monkeypatch
):
    postprocess_dir, coastline, isobaths, _ = footprint_inputs
    anchors = {"site_a": (120.9, 25.1), "site_b": (121.2, 24.9)}
    config_path = _write_release_config(tmp_path / "release.yaml", anchors)
    described = _capture_figures(monkeypatch)
    out = _build(tmp_path, postprocess_dir, coastline, isobaths, receptor_config=config_path)
    stars = [d["stars"] for d in _wp38_colorbar_figs(described)]
    assert stars and all(len(s) >= 1 for s in stars)
    drawn_positions = {pos for s in stars for pos in s}
    assert drawn_positions == set(anchors.values())
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    by_site = manifest["footprint_figures"]["receptors"]["by_site"]
    for site, lonlat in anchors.items():
        rec = by_site[site]
        assert rec["source"] == "release_config_anchor_lonlat"
        assert rec["anchor_lonlat"] == list(lonlat) and rec["star_lonlat"] == list(lonlat)
        assert rec["receptor_core_radius_m"] == 12500.0
        # 後處理由釋放點反推的平均位置仍記錄著，但不是畫出來的位置
        assert rec["postprocess_release_point_mean_lonlat"] != list(lonlat)
        caption = manifest["output_png_sha256"][f"{site}/lookback_snapshot.png"]["caption"]
        assert f"{lonlat[0]:.4f}°E, {lonlat[1]:.4f}°N" in caption and "12.5 km" in caption
    assert "receptor_config" in manifest["inputs_sha256_before"]
    assert manifest["config"]["receptor_config_basename"] == "release.yaml"


def test_receptor_falls_back_to_release_point_mean_and_says_so(tmp_path, footprint_inputs):
    postprocess_dir, coastline, isobaths, _ = footprint_inputs
    out = _build(tmp_path, postprocess_dir, coastline, isobaths)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    rec = manifest["footprint_figures"]["receptors"]["by_site"]["site_a"]
    assert rec["source"] == "postprocess_release_point_mean" and rec["anchor_lonlat"] is None
    assert rec["star_lonlat"] == rec["postprocess_release_point_mean_lonlat"]
    caption = manifest["output_png_sha256"]["site_a/source_footprint_residence.png"]["caption"]
    assert "平均位置" in caption and "不是研究點錨點" in caption


def test_release_config_missing_a_site_fails_closed(tmp_path, footprint_inputs):
    postprocess_dir, coastline, isobaths, _ = footprint_inputs
    config_path = _write_release_config(tmp_path / "release.yaml", {"site_a": (120.9, 25.1)})
    with pytest.raises(native_figures.FiguresError, match="anchor_lonlat"):
        _build(tmp_path, postprocess_dir, coastline, isobaths, receptor_config=config_path)
    (tmp_path / "bad.yaml").write_text("study_sites: []\n", encoding="utf-8")
    with pytest.raises(native_figures.FiguresError, match="study_sites"):
        _build(
            tmp_path, postprocess_dir, coastline, isobaths, receptor_config=tmp_path / "bad.yaml", name="o2"
        )


def test_site_name_is_absent_from_none_mode_figures_but_kept_in_captions(
    tmp_path, footprint_inputs, monkeypatch
):
    postprocess_dir, coastline, isobaths, _ = footprint_inputs
    names = {"site_a": "甲灣", "site_b": "乙岸"}
    monkeypatch.setitem(native_figures.SITE_LABELS_ZH, "site_a", names["site_a"])
    monkeypatch.setitem(native_figures.SITE_LABELS_ZH, "site_b", names["site_b"])
    config_path = _write_release_config(
        tmp_path / "release.yaml", {"site_a": (120.9, 25.1), "site_b": (121.2, 24.9)}
    )
    full = _build(tmp_path, postprocess_dir, coastline, isobaths, name="full", receptor_config=config_path)
    none = _build(
        tmp_path, postprocess_dir, coastline, isobaths, name="none", text="none", receptor_config=config_path
    )
    forbidden = [*names.values(), "site_a", "site_b"]
    for site in SITES:
        for stem in native_figures.FOOTPRINT_FIGURE_STEMS:
            svg_none = (none / site / f"{stem}.svg").read_text(encoding="utf-8")
            svg_full = (full / site / f"{stem}.svg").read_text(encoding="utf-8")
            for word in forbidden:
                assert word not in svg_none, (site, stem, word)
            # 對照：full 模式的標題確實有站名，所以上面的檢查不是空轉
            assert names[site] in svg_full, (site, stem)
        caps = (none / "captions.md").read_text(encoding="utf-8")
        assert names[site] in caps
    # 星號旁沒有站名標籤，圖例只寫「受體」
    for stem in ("source_footprint_residence", "lookback_snapshot"):
        svg_full = (full / "site_a" / f"{stem}.svg").read_text(encoding="utf-8")
        assert "研究地點錨點" in svg_full and "抽樣範圍" in svg_full
    # 每張新圖的圖說都有站名與受體經緯度
    manifest = json.loads((none / "manifest.json").read_text(encoding="utf-8"))
    for site, (lon, lat) in {"site_a": (120.9, 25.1), "site_b": (121.2, 24.9)}.items():
        for stem in native_figures.FOOTPRINT_FIGURE_STEMS:
            caption = manifest["output_png_sha256"][f"{site}/{stem}.png"]["caption"]
            assert names[site] in caption and f"{lon:.4f}°E, {lat:.4f}°N" in caption, (site, stem)


def test_map_has_no_isobath_number_labels_and_no_site_names(tmp_path, footprint_inputs, monkeypatch):
    """地圖 axes 內沒有等深線數字標籤，地圖與圖例都沒有站名（只有 full 模式的圖標題可以有）。"""

    import re

    postprocess_dir, coastline, isobaths, _ = footprint_inputs
    names = {"site_a": "甲灣", "site_b": "乙岸"}
    for site, name in names.items():
        monkeypatch.setitem(native_figures.SITE_LABELS_ZH, site, name)
    config_path = _write_release_config(
        tmp_path / "release.yaml", {"site_a": (120.9, 25.1), "site_b": (121.2, 24.9)}
    )
    described = _capture_figures(monkeypatch)
    _build(tmp_path, postprocess_dir, coastline, isobaths, receptor_config=config_path)
    figs = _wp38_colorbar_figs(described)
    assert len(figs) == 3 * len(SITES)
    depth_label = re.compile(r"^\s*\d+(\.\d+)?\s*(m|公尺)\s*$")
    for d in figs:
        assert d["isobaths"], "等深線有畫，只是沒有數字標籤"
        for text in d["axes_texts"]:
            assert not depth_label.match(text), text
            assert not any(word in text for word in (*names.values(), *SITES)), text
        for text in d["legend_texts"]:
            assert not any(word in text for word in (*names.values(), *SITES)), text
        assert "研究地點錨點" in d["legend_texts"] and "抽樣範圍" in d["legend_texts"]
        assert "等深線 200 m（實線）" in d["legend_texts"] and "等深線 1000 m（虛線）" in d["legend_texts"]


def test_legend_omits_isobath_levels_that_are_absent_from_the_site(tmp_path, footprint_inputs, monkeypatch):
    postprocess_dir, coastline, isobaths, _ = footprint_inputs
    described = _capture_figures(monkeypatch)
    _build(tmp_path, postprocess_dir, coastline, isobaths, isobath_levels=(200.0,))
    for d in _wp38_colorbar_figs(described):
        assert "等深線 200 m（實線）" in d["legend_texts"]
        assert not any("1000" in text for text in d["legend_texts"])


def _footprint_data_for_circle(site_id, anchor, points):
    domain = np.array([[119.0, 21.0], [122.5, 21.0], [122.5, 27.0], [119.0, 27.0]])
    return native_figures.FootprintSiteFigureData(
        site_id=site_id,
        grid={
            "lon0_deg": 119.0,
            "lat0_deg": 21.0,
            "nx": 35,
            "ny": 60,
            "resolution_deg": 0.1,
            "total_particle_hours": 1,
        },
        density_pct_per_km2=np.zeros((60, 35)),
        hdr_threshold_pct_per_km2={},
        hdr_area_km2={},
        snapshots=(),
        arc_bins=(),
        edge_intervals=(),
        domain_lonlat=domain,
        open_boundary_parts=(),
        receptor_center=anchor,
        summary={},
        receptor_anchor=anchor,
        release_points=points,
    )


_RUN08_RELEASE_ANCHORS = {
    "gongliao": (121.9223889, 25.0964444),
    "guishan": (121.939148, 24.850326),
    "hsinchu": (120.45, 24.75),
    "nanwan": (120.763161, 21.946577),
    "lienchiang": (119.95, 26.2),
}


def test_sampling_range_anchor_table_matches_run08_release_config():
    assert dict(native_figures.SAMPLING_RANGE_ANCHORS_LONLAT) == _RUN08_RELEASE_ANCHORS
    assert native_figures.SAMPLING_RANGE_RADIUS_M == 12500.0
    for site, anchor in _RUN08_RELEASE_ANCHORS.items():
        circle = native_figures._sampling_range_circle(site, anchor, 12500.0)
        assert circle.shape == (361, 2)
        assert tuple(circle[0]) == tuple(circle[-1])  # 閉合
        # 半徑缺省時取常數 12.5 km
        assert native_figures._sampling_range_circle(site, anchor, None) == pytest.approx(circle)
    assert native_figures._sampling_range_circle("any_site", None, 12500.0) is None
    # 表外的站只依錨點與半徑計算
    other = native_figures._sampling_range_circle("other", (120.123, 23.456), 5000.0)
    d = native_figures.geodesic_distance_km(120.123, 23.456, other[:, 0], other[:, 1])
    assert d == pytest.approx(5.0, abs=1e-6)
    # 錨點移動或半徑不是 12500 m 時必須報錯，而不是默默畫出另一個圓
    with pytest.raises(native_figures.FiguresError, match="錨點"):
        native_figures._sampling_range_circle("guishan", (121.951606, 24.843127), 12500.0)
    with pytest.raises(native_figures.FiguresError, match="receptor_core_radius_m"):
        native_figures._sampling_range_circle("guishan", _RUN08_RELEASE_ANCHORS["guishan"], 10000.0)


def test_sampling_range_circle_is_geodesic_not_a_lonlat_circle():
    for site, (lon0, lat0) in _RUN08_RELEASE_ANCHORS.items():
        circle = native_figures._sampling_range_circle(site, (lon0, lat0), 12500.0)
        dist = native_figures.geodesic_distance_km(lon0, lat0, circle[:, 0], circle[:, 1])
        assert dist == pytest.approx(12.5, abs=1e-6)  # 每個頂點離錨點都是 12.5 km
        half_lon = (circle[:, 0].max() - circle[:, 0].min()) / 2
        half_lat = (circle[:, 1].max() - circle[:, 1].min()) / 2
        # 經度半寬應約為緯度半寬除以 cos(緯度)，也就是畫在經緯度平面上不是正圓
        assert half_lon / half_lat == pytest.approx(1.0 / np.cos(np.radians(lat0)), rel=0.01)
        # 方位涵蓋一圈
        dx = (circle[:, 0] - lon0) * np.cos(np.radians(lat0))
        az = np.degrees(np.arctan2(dx, circle[:, 1] - lat0)) % 360
        assert np.max(np.diff(np.sort(az[:-1]))) < 1.5


def test_sampling_range_circle_is_drawn_and_replaces_box_and_hull():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 南灣：5 個水平位置各 4 層深度，全部落在 12.5 km 內
    points = np.array(
        [[120.722, 21.867], [120.8807, 21.9229], [120.80, 21.90], [120.76, 21.88], [120.84, 21.92]]
    )
    points = np.repeat(points, 4, axis=0)
    fp = _footprint_data_for_circle("nanwan", (120.763161, 21.946577), points)
    fp = dataclasses.replace(fp, receptor_core_radius_m=12500.0)
    fig, ax = plt.subplots()
    try:
        native_figures._draw_footprint_base_map(ax, fp, [], {})
        circles = [ln for ln in ax.lines if len(ln.get_xydata()) == 361]
        assert len(circles) == 1
        circle = circles[0]
        assert circle.get_color() == "#1F4E79" and circle.get_linestyle() == "--"
        assert circle.get_zorder() == 7
        assert circle.get_linewidth() == pytest.approx(2.4)
        xy = np.asarray(circle.get_xydata())
        dist = native_figures.geodesic_distance_km(120.763161, 21.946577, xy[:, 0], xy[:, 1])
        assert dist == pytest.approx(12.5, abs=1e-6)
        # 不填色、沒有矩形框（5 點折線）、沒有淡紫色凸包多邊形
        assert not [p for p in ax.patches if type(p).__name__ == "Polygon"]
        assert not [ln for ln in ax.lines if len(ln.get_xydata()) == 5]
        # 全部釋放點在圓內
        pd_km = native_figures.geodesic_distance_km(120.763161, 21.946577, points[:, 0], points[:, 1])
        assert np.all(pd_km <= 12.5)
        star = [ln for ln in ax.lines if ln.get_marker() == "*"]
        assert len(star) == 1 and tuple(star[0].get_xydata()[0]) == (120.763161, 21.946577)
    finally:
        plt.close(fig)
    # 沒有錨點（診斷模式）：不畫圓，圖例也沒有抽樣範圍
    fp2 = dataclasses.replace(fp, receptor_anchor=None)
    fig, ax = plt.subplots()
    try:
        native_figures._draw_footprint_base_map(ax, fp2, [], {})
        assert not [ln for ln in ax.lines if len(ln.get_xydata()) == 361]
    finally:
        plt.close(fig)


def test_manifest_records_release_geometry_and_anchor_checks(tmp_path, footprint_inputs, monkeypatch):
    postprocess_dir, coastline, isobaths, _ = footprint_inputs
    anchors = {"site_a": (120.9, 25.1), "site_b": (121.1, 24.6)}  # site_b 的錨點落在合成陸地方塊內
    config_path = _write_release_config(tmp_path / "release.yaml", anchors)
    described = _capture_figures(monkeypatch)
    out = _build(tmp_path, postprocess_dir, coastline, isobaths, receptor_config=config_path)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    block = manifest["footprint_figures"]["receptors"]
    import hashlib

    assert block["config_sha256"] == hashlib.sha256(config_path.read_bytes()).hexdigest()
    assert block["config_path"] == str(config_path)
    for site, (lon, lat) in anchors.items():
        geo = block["by_site"][site]["release_geometry"]
        assert geo["release_point_count"] == len(RECEPTORS)
        pts = np.asarray(geo["release_points_lonlat"])
        centroid = pts.mean(axis=0)
        assert geo["release_centroid_lonlat"] == pytest.approx(list(centroid))
        # 獨立估算：小範圍內用等距圓柱近似（1 度緯度約 111.195 km）
        dx = (centroid[0] - lon) * np.cos(np.radians(lat)) * 111.195
        dy = (centroid[1] - lat) * 111.195
        assert geo["anchor_to_release_centroid_km"] == pytest.approx(float(np.hypot(dx, dy)), rel=0.02)
        assert geo["release_max_radius_km"] >= 0.0
        assert geo["release_max_distance_from_anchor_km"] >= geo["anchor_to_release_centroid_km"] - 1e-9
    assert block["by_site"]["site_a"]["release_geometry"]["anchor_on_sea"] is True
    assert block["by_site"]["site_b"]["release_geometry"]["anchor_on_sea"] is False
    # 圖上畫的是抽樣範圍虛線圓（以錨點為圓心、半徑＝設定檔 receptor_core_radius_m），不再畫釋放點凸包；
    # 圓與 manifest 記錄一致，且記錄「全部釋放點是否在圓內」與最遠釋放點距離
    for site, (lon, lat) in anchors.items():
        rec = block["by_site"][site]["release_geometry"]["sampling_range"]
        assert rec["shape"] == "geodesic_circle" and rec["radius_m"] == 12500.0
        assert rec["anchor_lonlat"] == [lon, lat] and rec["vertex_count"] == 361
        pts_geo = np.asarray(block["by_site"][site]["release_geometry"]["release_points_lonlat"])
        dist = native_figures.geodesic_distance_km(lon, lat, pts_geo[:, 0], pts_geo[:, 1])
        assert rec["release_point_count"] == len(RECEPTORS)
        # 合成釋放點不保證在圓內；真實資料另行核對 25/25
        assert rec["release_points_inside_circle"] == int(np.sum(dist <= 12.5))
        assert rec["max_release_point_distance_km"] == pytest.approx(float(dist.max()))
        assert rec["style"]["color"] == "#1F4E79"
    for d in _wp38_colorbar_figs(described):
        assert not d["release_polygons"]
        assert d["sampling_circles"]


def test_missing_config_is_rejected_unless_explicitly_allowed(tmp_path, footprint_inputs):
    postprocess_dir, coastline, isobaths, _ = footprint_inputs
    with pytest.raises(native_figures.FiguresError, match="--config"):
        _build(tmp_path, postprocess_dir, coastline, isobaths, allow_unanchored=False)
    out = _build(tmp_path, postprocess_dir, coastline, isobaths, name="diag")  # 診斷模式：明示允許
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"]["allow_unanchored_receptor"] is True
    rec = manifest["footprint_figures"]["receptors"]
    assert rec["config_sha256"] is None and rec["by_site"]["site_a"]["formal_anchor_used"] is False


def test_cli_requires_config_flag_for_formal_figures(tmp_path, footprint_inputs, monkeypatch):
    from lagrangian_backtracking import cli

    postprocess_dir, coastline, isobaths, _ = footprint_inputs
    parser = cli._native_figures_parser()
    args = parser.parse_args(
        [
            "--postprocess-dir",
            str(postprocess_dir),
            "--coastline",
            str(coastline),
            "--output-dir",
            str(tmp_path / "o"),
        ]
    )
    assert args.config is None and args.allow_unanchored_receptor is False
    assert args.isobath_levels == [200.0, 1000.0]
    with pytest.raises(native_figures.FiguresError, match="--config"):
        cli.run_native_figures_command(
            [
                "--postprocess-dir",
                str(postprocess_dir),
                "--coastline",
                str(coastline),
                "--output-dir",
                str(tmp_path / "o"),
            ]
        )


def test_json_policy_wp38_constants_match_figures_py():
    """JSON 契約 wp38_footprint_figures 區塊的常數與 figures.py 執行期常數逐項一致。"""

    policy_path = (
        Path(__file__).resolve().parents[1] / "docs" / "data" / "native_figures_formal_output_policy_v1.json"
    )
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    assert policy["schema_version"] == "1.1.0" and policy["rules_document_version"] == "1.1.0"
    block = policy["wp38_footprint_figures"]
    assert tuple(block["figure_stems"]["value"]) == native_figures.FOOTPRINT_FIGURE_STEMS
    iso = block["isobath"]
    assert tuple(iso["levels_m_default"]["value"]) == native_figures.ISOBATH_LEVELS_DEFAULT
    assert tuple(iso["linestyles_by_level_order"]["value"]) == native_figures._ISOBATH_LINESTYLES
    assert iso["color"]["value"] == native_figures._ISOBATH_COLOR
    assert iso["linewidth_pt"]["value"] == native_figures._ISOBATH_LINEWIDTH
    assert iso["number_labels_on_map"] is False
    sampling = block["sampling_range"]
    assert sampling["shape"] == "geodesic_circle"
    assert sampling["radius_m"]["value"] == native_figures.SAMPLING_RANGE_RADIUS_M
    assert sampling["azimuth_step_deg"]["value"] == native_figures.SAMPLING_RANGE_AZIMUTH_STEP_DEG
    assert {k: tuple(v) for k, v in sampling["anchors_lonlat"]["value"].items()} == dict(
        native_figures.SAMPLING_RANGE_ANCHORS_LONLAT
    )
    style = native_figures._SAMPLING_RANGE_STYLE
    assert sampling["color"]["value"] == style["color"]
    assert sampling["linewidth_pt"]["value"] == style["linewidth"]
    assert tuple(sampling["dash_in_linewidth_units"]["value"]) == tuple(style["linestyle"][1])
    assert sampling["legend_text_zh"] == native_figures._SAMPLING_RANGE_LABEL_ZH
    assert sampling["legend_text_en"] == native_figures._SAMPLING_RANGE_LABEL_EN
    assert block["receptor"]["legend_text_zh"] == "研究地點錨點"
    axes = block["map_axes"]
    assert tuple(axes["tick_step_choices_deg"]["value"]) == native_figures._TICK_STEP_CHOICES_DEG
    assert axes["min_intervals"]["value"] == native_figures._MIN_INTERVALS
    assert axes["residence_max_intervals"]["value"] == native_figures._RESIDENCE_MAX_INTERVALS
    assert axes["snapshot_max_intervals"]["value"] == native_figures._SNAPSHOT_MAX_INTERVALS
    bar = block["colorbar"]
    assert bar["extend"] == "neither"
    assert (
        bar["lower_limit_fraction_of_upper_limit"]["value"] == native_figures.RESIDENCE_COLOR_FLOOR_FRACTION
    )
    assert bar["alignment_tolerance_px"]["value"] == native_figures.COLORBAR_ALIGNMENT_TOLERANCE_PX
    lo, hi = native_figures._decade_limits(0.37)
    assert lo / hi == pytest.approx(bar["lower_limit_fraction_of_upper_limit"]["value"])
    assert block["receptor"]["config_required"] is True


def test_tick_labels_do_not_overlap_including_neighbouring_snapshot_panels(
    tmp_path, footprint_inputs, monkeypatch
):
    postprocess_dir, coastline, isobaths, _ = footprint_inputs
    described = _capture_figures(monkeypatch)
    _build(tmp_path, postprocess_dir, coastline, isobaths)
    figs = _wp38_colorbar_figs(described)
    assert len(figs) == 3 * len(SITES) and any(d["n_maps"] == 4 for d in figs)
    assert all(d["tick_label_overlaps"] == 0 for d in figs)


def test_arc_edge_and_corner_names_sit_above_the_frame_without_overlap(
    tmp_path, footprint_inputs, monkeypatch
):
    postprocess_dir, coastline, isobaths, _ = footprint_inputs
    described = _capture_figures(monkeypatch)
    _build(tmp_path, postprocess_dir, coastline, isobaths)
    arcs = [d for d in described if d["twin"]]
    assert len(arcs) == len(SITES)
    for d in arcs:
        boxes = d["arc_label_boxes"]
        assert len(boxes) >= 4  # 至少有各邊名稱
        assert all(box.y0 >= d["arc_axes_top_px"] - 0.5 for box in boxes)  # 不壓在直條與曲線上
        assert not any(
            boxes[i].overlaps(boxes[j]) for i in range(len(boxes)) for j in range(i + 1, len(boxes))
        )


def test_arc_distribution_and_snapshot_use_fixed_limits_for_every_site(tmp_path, footprint_inputs, monkeypatch):
    """沿邊界弧長分布圖左縱軸固定 0～50%；回溯快照色階固定 10^-4～10^1；各站相同。"""

    seen: dict[str, dict[str, tuple]] = {}
    original_save = native_figures._save_exclusive

    def _checking_save(fig, path, *, dpi, **kw):
        if path.stem == "boundary_departure_along_arc":
            ax = fig.axes[0]
            seen.setdefault("arc", {})[path.parent.name] = (ax.get_ylim(), [float(v) for v in ax.get_yticks()])
        return original_save(fig, path, dpi=dpi, **kw)

    monkeypatch.setattr(native_figures, "_save_exclusive", _checking_save)
    postprocess_dir, coastline, isobaths, _ = footprint_inputs
    out = _build(tmp_path, postprocess_dir, coastline, isobaths)
    values = list(seen["arc"].values())
    assert len(values) == len(SITES) and all(v == values[0] for v in values)
    assert values[0][0] == pytest.approx((0.0, 50.0)) and values[0][1][-1] == 50.0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    limits = manifest["footprint_figures"]["snapshot_color_scale"]["limits_by_site"]
    assert len({(v["lower_limit"], v["upper_limit"]) for v in limits.values()}) == 1
    for v in limits.values():
        assert v["lower_limit"] == pytest.approx(1e-4) and v["upper_limit"] == pytest.approx(10.0)
