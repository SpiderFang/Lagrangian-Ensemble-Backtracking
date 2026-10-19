"""圓形來源方位圖（WP-60）的測試：扇區邊界、方位角方向、擱淺點計入、比例加總、無文字版不含地點名稱（合成資料）。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lagrangian_backtracking.native_runner import figures as native_figures
from lagrangian_backtracking.native_runner import polar_source as ps
from lagrangian_backtracking.native_runner import polar_source_figures as psf

ANCHOR = (121.92, 25.10)
SITE = "gongliao"


# ── 扇區邊界與方位角 ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("azimuth", "expected"),
    [
        (355.0, 0),  # 正北扇區涵蓋 [355°, 5°)：355° 屬編號 0
        (359.99, 0),
        (0.0, 0),
        (4.99, 0),  # 4.99° 仍屬正北扇區
        (5.0, 1),  # 5° 起屬中心 10° 的扇區
        (14.99, 1),
        (15.0, 2),
        (354.99, 35),  # 354.99° 屬中心 350° 的扇區
        (350.0, 35),
        (90.0, 9),
        (180.0, 18),
        (270.0, 27),
    ],
)
def test_sector_boundaries(azimuth, expected):
    assert int(ps.sector_index(np.array([azimuth]))[0]) == expected


def test_sector_centers_cover_all_36_sectors_once():
    centers = np.arange(36) * 10.0
    assert np.array_equal(ps.sector_index(centers), np.arange(36))


def test_bearing_directions_on_equator():
    lon = np.array([1.0, 0.0, -1.0, 0.0])
    lat = np.array([0.0, 1.0, 0.0, -1.0])
    az = ps.bearing_deg(0.0, 0.0, lon, lat)
    assert az[0] == pytest.approx(90.0, abs=1e-9)  # 正東 = 90°
    assert az[1] == pytest.approx(0.0, abs=1e-9)  # 正北 = 0°
    assert az[2] == pytest.approx(270.0, abs=1e-9)  # 正西 = 270°
    assert az[3] == pytest.approx(180.0, abs=1e-9)  # 正南 = 180°


def test_bearing_east_at_study_latitude_is_close_to_90_and_in_range():
    az = ps.bearing_deg(ANCHOR[0], ANCHOR[1], np.array([ANCHOR[0] + 0.3, ANCHOR[0] - 0.3]), np.array([ANCHOR[1]] * 2))
    assert 89.0 < az[0] < 90.0  # 大圓初始方位角在同緯度的東邊略偏向赤道側，與 90° 差不到 1°
    assert 270.0 < az[1] < 271.0
    assert np.all((az >= 0.0) & (az < 360.0))
    assert int(ps.sector_index(az)[0]) == 9


# ── 擱淺點與其他終止類別 ─────────────────────────────────────────────────


def test_stranded_particles_are_counted_by_bearing_to_stranding_point():
    status = np.array([1, 3, 3, 2, 4])
    lon = np.array([ANCHOR[0] + 0.5, ANCHOR[0] - 0.05, ANCHOR[0] - 0.05, ANCHOR[0], np.nan])
    lat = np.array([ANCHOR[1], ANCHOR[1], ANCHOR[1] + 0.001, ANCHOR[1], np.nan])
    kind, az, bins = ps.classify_sources(status, lon, lat, ANCHOR)
    assert list(kind) == [ps.KIND_CROSSED, ps.KIND_STRANDED, ps.KIND_STRANDED, ps.KIND_OTHER, ps.KIND_OTHER]
    assert bins[0] == 9  # 東
    assert bins[1] == 27 and bins[2] == 27  # 擱淺點在西方（陸地一側），計入 270° 扇區
    assert bins[3] == ps.UNCOUNTED_BIN and bins[4] == ps.UNCOUNTED_BIN  # 其他終止類別不計入扇區
    assert np.isnan(az[3]) and np.isnan(az[4])


def test_missing_position_of_counted_particle_fails_closed():
    with pytest.raises(ps.PolarSourceError):
        ps.classify_sources(np.array([1]), np.array([np.nan]), np.array([25.0]), ANCHOR)


# ── 合成後處理資料 ───────────────────────────────────────────────────────

MATERIALS = ("oca_nonrecyclable_flexible_sheet", "oca_steel_rigid_cylinder")


def _departures(n_per: int = 60) -> pa.Table:
    rng = np.random.default_rng(7)
    rows: dict[str, list] = {c: [] for c in ps._REQUIRED_COLUMNS}
    k = 0
    for m_i, material in enumerate(MATERIALS):
        for cand in range(2):
            for j in range(n_per):
                az = rng.uniform(0.0, 360.0)
                r_km = 60.0
                lon = ANCHOR[0] + r_km * np.sin(np.radians(az)) / (111.195 * np.cos(np.radians(ANCHOR[1])))
                lat = ANCHOR[1] + r_km * np.cos(np.radians(az)) / 111.195
                status = 1
                if j % 20 == 0:
                    status = 3  # 擱淺：位置在西側海岸
                    lon, lat = ANCHOR[0] - 0.05, ANCHOR[1]
                elif j % 20 == 1:
                    status = 2  # 達回溯上限
                elif j % 20 == 2 and cand == 1:
                    status = 4  # 沉底
                rows["particle_id"].append(f"p{k}")
                rows["study_site_id"].append(SITE)
                rows["material_id"].append(material)
                rows["settling_velocity_mps"].append(-0.01 * (m_i + 1))
                rows["particle_settling_velocity_mps"].append(-0.01 * (m_i + 1) * (1 + cand))
                rows["settling_candidate_index"].append(cand)
                rows["arrival_time_id"].append(f"arr_{j % 6}")
                rows["final_status_code"].append(status)
                rows["crossed_interest_region_boundary"].append(status == 1)
                rows["departure_lon_deg"].append(float(lon))
                rows["departure_lat_deg"].append(float(lat))
                rows["deposition_season"].append(("DJF", "MAM", "JJA", "SON")[j % 4])
                rows["deposition_tide_class"].append(("spring_proxy", "neap_proxy")[j % 2])
                k += 1
    schema = pa.schema(
        [
            ("particle_id", pa.string()),
            ("study_site_id", pa.string()),
            ("material_id", pa.string()),
            ("settling_velocity_mps", pa.float64()),
            ("particle_settling_velocity_mps", pa.float64()),
            ("settling_candidate_index", pa.int16()),
            ("arrival_time_id", pa.string()),
            ("final_status_code", pa.int32()),
            ("crossed_interest_region_boundary", pa.bool_()),
            ("departure_lon_deg", pa.float64()),
            ("departure_lat_deg", pa.float64()),
            ("deposition_season", pa.string()),
            ("deposition_tide_class", pa.string()),
        ]
    )
    return pa.table(rows, schema=schema)


@pytest.fixture()
def synthetic(tmp_path: Path):
    pp = tmp_path / "postprocess"
    pp.mkdir()
    pq.write_table(_departures(), pp / "departures.parquet")
    cfg = tmp_path / "release.yaml"
    cfg.write_text(
        "study_sites:\n"
        f"  - study_site_id: {SITE}\n    anchor_lonlat: [{ANCHOR[0]}, {ANCHOR[1]}]\n    receptor_core_radius_m: 12500.0\n",
        encoding="utf-8",
    )
    coast = tmp_path / "coast.geojson"
    coast.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
                "features": [
                    {
                        "type": "Feature",
                        "properties": {},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[121.2, 24.5], [121.85, 24.5], [121.85, 25.6], [121.2, 25.6], [121.2, 24.5]]],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return pp, cfg, coast


def _samples(synthetic):
    pp, _cfg, _coast = synthetic
    return ps.load_site_polar_samples(pp, {SITE: ANCHOR})


def test_sector_shares_sum_to_counted_share_and_all_bins_sum_to_one(synthetic):
    sample = _samples(synthetic)[SITE]
    an = ps.analyze_site(sample, "case", ps.PolarParams(n_resamples=50))
    for grouping, gs in an.group_sets.items():
        pct = gs.pct_sectors()
        assert pct.shape == (len(gs.group_ids), 36)
        assert np.allclose(gs.proportions37().sum(axis=1), 1.0)
        for i in range(len(gs.group_ids)):
            sel = gs.group_code == i
            counted = np.count_nonzero(sample.kind[sel] != ps.KIND_OTHER)
            assert pct[i].sum() == pytest.approx(100.0 * counted / sel.sum(), rel=1e-12)
        assert int(gs.n_total.sum()) == sample.n_rows
    # 每組粒子數：類別×速度 4 組各 60、類別 2 組各 120。
    assert list(an.group_sets["class_velocity"].n_total) == [60] * 4
    assert list(an.group_sets["class"].n_total) == [120, 120]


def test_stranded_particles_land_in_the_west_sector_of_the_counts(synthetic):
    sample = _samples(synthetic)[SITE]
    an = ps.analyze_site(sample, "case", ps.PolarParams(n_resamples=50))
    gs = an.group_sets["class"]
    stranded_per_class = 3  # j % 20 == 0 的粒子，每個 (類別, 速度) 60 顆中有 3 顆，每類 2 個速度共 6 顆
    sector_counts = gs.counts.sum(axis=1)
    assert np.all(sector_counts[:, 27] >= 2 * stranded_per_class)
    assert int(np.count_nonzero(sample.kind == ps.KIND_STRANDED)) == 4 * stranded_per_class
    rows = ps.termination_rows(an, "case")
    site_row = next(r for r in rows if r["grouping"] == "site_all")
    assert site_row["n_stranded"] == 12
    assert site_row["n_crossed"] + site_row["n_stranded"] + site_row["n_uncounted"] == sample.n_rows
    assert site_row["n_status_2"] > 0 and site_row["n_status_4"] > 0


def test_half_split_noise_is_reproducible_and_bounded():
    rng_counts = np.random.default_rng(3).integers(0, 20, size=(12, ps.N_BINS))
    a = ps.half_split_noise(rng_counts, 200, np.random.default_rng(1))
    b = ps.half_split_noise(rng_counts, 200, np.random.default_rng(1))
    assert a is not None and b is not None
    assert np.array_equal(a[0], b[0])
    assert a[0].shape == (200,) and np.all((a[0] >= 0) & (a[0] <= 1))
    assert ps.half_split_noise(np.zeros((1, ps.N_BINS)), 10, np.random.default_rng(1)) is None


def test_tvd_identical_is_zero_and_disjoint_is_one():
    p = np.zeros(ps.N_BINS)
    p[3] = 1.0
    q = np.zeros(ps.N_BINS)
    q[10] = 1.0
    assert ps.tvd_and_max_sector(p, p) == (0.0, 0.0)
    tvd, mx = ps.tvd_and_max_sector(p, q)
    assert tvd == pytest.approx(1.0) and mx == pytest.approx(100.0)


def test_radius_scales_are_shared_and_nice(synthetic):
    sample = _samples(synthetic)[SITE]
    an = ps.analyze_site(sample, "case", ps.PolarParams(n_resamples=20))
    scales = psf.plan_radius_scales({"a": [an], "b": [an]})
    assert set(scales) == set("ABCD")
    assert scales["A"] == scales["B"] == scales["D"]
    top, ticks = scales["A"]
    assert top == ticks[-1] and top >= float(an.group_sets["class_velocity"].pct_sectors().max())
    forced = psf.plan_radius_scales({"a": [an]}, override_max_pct=40.0)
    assert all(v[0] == 40.0 for v in forced.values())


# ── 圖面：無文字版不含地點名稱、字型物件、軸刻度 ─────────────────────────


def _render_capture(monkeypatch, synthetic, tmp_path, mode):
    """以合成資料出一張圖，攔截 Figure 物件並回傳（含所有文字）。"""

    pp, cfg, coast = synthetic
    sample = _samples(synthetic)[SITE]
    an = ps.analyze_site(sample, "case", ps.PolarParams(n_resamples=20))
    scales = psf.plan_radius_scales({"c": [an]})
    plans = psf.build_figure_plans(an, "測試案例", scales)
    land = native_figures._load_coastline(coast)
    site_map = psf._build_site_map(ANCHOR, 25.0, land, {})
    font_res = native_figures._resolve_font(None)
    t = native_figures._Labeler(font_res.cjk_available)
    captured = {}

    def fake_save(fig, path, *, dpi):
        captured["fig"] = fig

    monkeypatch.setattr(psf, "_save_exclusive", fake_save)
    plan = next(p for p in plans if p.kind == "C")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "close", lambda fig=None: None)
    psf.render_figure(plan, site_map, t, font_res.font_properties, tmp_path / "x.png", 50, mode)
    return captured["fig"], font_res, plans


def _all_texts(fig) -> list[str]:
    out = [t.get_text() for t in fig.texts]
    for ax in fig.axes:
        out += [t.get_text() for t in ax.texts]
        out += [lab.get_text() for lab in ax.get_xticklabels() + ax.get_yticklabels()]
        for leg in [ax.get_legend(), *[a for a in ax.artists if hasattr(a, "get_texts")]]:
            if leg is not None:
                out += [t.get_text() for t in leg.get_texts()] + [leg.get_title().get_text()]
    return out


@pytest.mark.parametrize("mode", ["full", "none"])
def test_figure_text_has_no_site_name_or_code(monkeypatch, synthetic, tmp_path, mode):
    fig, _font, plans = _render_capture(monkeypatch, synthetic, tmp_path, mode)
    texts = _all_texts(fig)
    joined = "\n".join(texts)
    assert SITE not in joined
    assert "貢寮" not in joined
    for plan in plans:
        for s in (plan.title_zh, plan.subtitle_zh, plan.legend_title_zh, *plan.labels):
            assert SITE not in s and "貢寮" not in s
    if mode == "none":
        assert fig.texts == []  # 無文字版沒有標題與副標
        assert all("比例" not in s for s in texts)
    else:
        assert any("來源方位的比例" in s for s in texts)
    assert "機率" not in joined
    # 半徑刻度仍在：地圖上的文字含「%」。
    assert any(s.endswith("%") for s in texts)


def _same_font(a, b) -> bool:
    """比對字型物件的字型檔與字型族（字級各處不同，不納入比對）。"""

    return (a.get_file(), tuple(a.get_family())) == (b.get_file(), tuple(b.get_family()))


def test_tick_and_legend_text_use_font_object_not_default(monkeypatch, synthetic, tmp_path):
    fig, font_res, _ = _render_capture(monkeypatch, synthetic, tmp_path, "none")
    font = font_res.font_properties
    ax = fig.axes[0]
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        assert lab.get_text() != ""  # 上下界與中間刻度都有數字
        assert _same_font(lab.get_fontproperties(), font)
    for txt in ax.texts:
        assert _same_font(txt.get_fontproperties(), font)
    legends = [ax.get_legend(), *[a for a in ax.artists if hasattr(a, "get_texts")]]
    assert any(leg is not None for leg in legends)
    for leg in legends:
        if leg is None:
            continue
        if leg.get_title().get_text():
            assert _same_font(leg.get_title().get_fontproperties(), font)
        for t in leg.get_texts():
            assert _same_font(t.get_fontproperties(), font)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    assert float(ax.get_xticks()[0]) == pytest.approx(x0) and float(ax.get_xticks()[-1]) == pytest.approx(x1)
    assert float(ax.get_yticks()[0]) == pytest.approx(y0) and float(ax.get_yticks()[-1]) == pytest.approx(y1)


def test_circle_is_round_on_the_map():
    lo, la = psf.polar_xy_to_lonlat(ANCHOR, 25.0, np.linspace(0, 360, 361))
    dx_km = (lo - ANCHOR[0]) * 111.195 * np.cos(np.radians(ANCHOR[1]))
    dy_km = (la - ANCHOR[1]) * 111.195
    assert np.allclose(np.hypot(dx_km, dy_km), 25.0)


# ── 端到端 ───────────────────────────────────────────────────────────────


def test_end_to_end_outputs_and_stranded_marker(synthetic, tmp_path):
    pp, cfg, coast = synthetic
    case = psf.PolarCase("no_stokes", pp, tmp_path / "out" / "figures-polar", tmp_path / "out" / "figures-polar-notext")
    config = psf.PolarConfig(
        cases=(case,),
        release_config=cfg,
        coastline=coast,
        bathymetry=None,
        params=ps.PolarParams(n_resamples=30),
        dpi=40,
        workers=1,
    )
    outputs = psf.build_polar_source(config, command="lbt native-polar-source test")
    assert outputs == [case.text_dir]
    for base in (case.text_dir, case.notext_dir):
        pngs = sorted(p.name for p in (base / SITE).glob("*.png"))
        assert len(pngs) == 13 - 8  # 兩個類別的 A 圖 + B + C + D = 5 張
        assert (base / SITE / "polar_source_by_class.svg").is_file()
    tables = case.text_dir / "tables"
    for name in (
        "sector_shares_long.csv", "termination_by_group.csv", "distinguishability_pairs.csv",
        "distinguishability_summary.csv", "stranded_points.csv",
    ):  # fmt: skip
        assert (tables / name).is_file()
    stranded = (tables / "stranded_points.csv").read_text(encoding="utf-8-sig").strip().splitlines()
    assert len(stranded) == 1 + 12  # 標頭 + 12 顆擱淺
    summary = (case.text_dir / "summary.md").read_text(encoding="utf-8")
    assert "每地點結論" in summary and "機率" not in summary
    manifest = json.loads((case.text_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status_counts_by_site"][SITE]["3"] == 12
    with pytest.raises(psf.PolarSourceError):
        psf.build_polar_source(config, command="again")  # 輸出目錄已存在，拒絕覆寫
