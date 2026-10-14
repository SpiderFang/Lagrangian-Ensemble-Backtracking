"""沿開放邊界之相對來源權重曲線（依組別分線）的測試（合成資料）。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lagrangian_backtracking import cli
from lagrangian_backtracking.native_runner import arc_by_group as abg
from lagrangian_backtracking.native_runner import arc_by_group_figures as abgf
from lagrangian_backtracking.native_runner import figures as native_figures

SIGMA = 5.0
LENGTH = 500.0
N_GRID = 1000


# ── 平滑與積分 ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("closed", [True, False])
def test_smoothed_curve_integrates_to_100_percent(closed):
    rng = np.random.default_rng(1)
    s = np.concatenate([rng.uniform(0.0, LENGTH, 500), np.array([0.0, 0.2, LENGTH - 0.1, LENGTH])])
    x, density = abg.smooth_arc_density(s, LENGTH, sigma_km=SIGMA, n_grid=N_GRID, closed=closed)
    dx = LENGTH / N_GRID
    assert x.shape == density.shape == (N_GRID,)
    assert float(density.sum() * dx) == pytest.approx(100.0, rel=1e-12)
    assert np.all(density >= 0.0)


def test_closed_boundary_wraps_mass_across_the_seam():
    # 一顆粒子在弧長 1 km：閉合時核函數繞過端點，弧長末端要有質量；鏡射時末端沒有質量。
    s = np.array([1.0])
    _, closed_density = abg.smooth_arc_density(s, LENGTH, sigma_km=SIGMA, n_grid=N_GRID, closed=True)
    _, open_density = abg.smooth_arc_density(s, LENGTH, sigma_km=SIGMA, n_grid=N_GRID, closed=False)
    assert closed_density[-1] > 1.0e-3
    assert open_density[-1] == pytest.approx(0.0, abs=1.0e-12)


def test_open_boundary_reflects_mass_back_inside():
    # 同一顆粒子：鏡射把端點外的質量折回，所以端點附近比環狀（質量跑到另一端）更高，總量仍是 100%。
    s = np.array([1.0])
    _, closed_density = abg.smooth_arc_density(s, LENGTH, sigma_km=SIGMA, n_grid=N_GRID, closed=True)
    _, open_density = abg.smooth_arc_density(s, LENGTH, sigma_km=SIGMA, n_grid=N_GRID, closed=False)
    assert open_density[:20].sum() > closed_density[:20].sum()
    dx = LENGTH / N_GRID
    assert open_density.sum() * dx == pytest.approx(100.0, rel=1e-12)


def test_arc_length_equal_to_full_length_wraps_to_start_when_closed():
    idx_closed = abg.bin_index_for_arc(np.array([LENGTH]), LENGTH, N_GRID, closed=True)
    idx_open = abg.bin_index_for_arc(np.array([LENGTH]), LENGTH, N_GRID, closed=False)
    assert int(idx_closed[0]) == 0
    assert int(idx_open[0]) == N_GRID - 1


# ── 邊界閉合檢查 ─────────────────────────────────────────────────────────


def test_boundary_closure_detection():
    ring = np.array([[121.0, 24.0], [122.0, 24.0], [122.0, 25.0], [121.0, 25.0], [121.0, 24.0]])
    closed, gap = abg.boundary_is_closed(ring)
    assert closed and gap == pytest.approx(0.0, abs=1e-6)

    line = ring[:-1]
    closed, gap = abg.boundary_is_closed(line)
    assert not closed and gap > 100_000.0

    nearly = ring.copy()
    nearly[-1] = [121.0, 24.0 + 0.0005]  # 約 55 m
    assert abg.boundary_is_closed(nearly, tolerance_m=100.0)[0]
    assert not abg.boundary_is_closed(nearly, tolerance_m=10.0)[0]

    assert abg.boundary_is_closed(None) == (False, pytest.approx(float("nan"), nan_ok=True))
    assert not abg.boundary_is_closed(ring[:2])[0]


# ── 總變異距離 ───────────────────────────────────────────────────────────


def test_total_variation_distance_values():
    p = np.array([1.0, 0.0, 0.0])
    q = np.array([0.0, 0.0, 1.0])
    assert abg.total_variation_distance(p, q) == pytest.approx(1.0)
    assert abg.total_variation_distance(p, p) == 0.0
    a = np.array([0.5, 0.5, 0.0])
    b = np.array([0.0, 0.5, 0.5])
    assert abg.total_variation_distance(a, b) == pytest.approx(0.5)
    assert abg.total_variation_distance(a, b) == pytest.approx(abg.total_variation_distance(b, a))
    with pytest.raises(abg.ArcByGroupError):
        abg.total_variation_distance(a, np.array([1.0]))


def test_edge_share_percent_sums_to_100_and_includes_last_endpoint():
    edges = [(0.0, 100.0), (100.0, 200.0), (200.0, 300.0)]
    s = np.array([0.0, 99.9, 100.0, 250.0, 300.0])
    share = abg.edge_share_percent(s, edges)
    assert share.tolist() == pytest.approx([40.0, 20.0, 40.0])
    assert share.sum() == pytest.approx(100.0)


# ── 分組分析（合成樣本）──────────────────────────────────────────────────


def _sample(*, same: bool, closed: bool = True) -> abg.SiteArcSample:
    """兩個類別共用同一批釋放點；``same=True`` 時兩類穿越位置完全相同，否則整體偏移。"""

    rng = np.random.default_rng(7)
    n = 1500
    base = np.clip(rng.normal(120.0, 15.0, n), 1.0, LENGTH - 1.0)
    other = base if same else np.clip(base + 25.0, 1.0, LENGTH - 1.0)
    s = np.concatenate([base, other])
    material = np.array(["oca_a"] * n + ["oca_b"] * n)
    release = np.concatenate([np.arange(n), np.arange(n)])
    event = np.concatenate([np.arange(n) // 10, np.arange(n) // 10])
    season = np.array((["DJF"] * (n // 2) + ["JJA"] * (n // 2)) * 2)
    return abg.SiteArcSample(
        site_id="site_x",
        length_km=LENGTH,
        closed=closed,
        closure_gap_m=0.0 if closed else float("nan"),
        edges=(("south", 0.0, 200.0), ("east", 200.0, 300.0), ("north", 300.0, 400.0), ("west", 400.0, LENGTH)),
        corners=(("south-east", 200.0), ("east-north", 300.0), ("north-west", 400.0)),
        s_km=s,
        material_id=material,
        velocity_mps=np.where(material == "oca_a", -0.01, -0.02),
        season=season,
        tide=None,
        release_code=release,
        event_code=event,
        n_departure_rows=2 * n,
        n_crossed=2 * n,
    )


_PARAMS = abg.ArcByGroupParams(sigma_km=SIGMA, n_grid=500, n_resamples=40, seed=3)


def test_identical_classes_do_not_exceed_paired_swap_noise():
    res = abg.analyze_grouping(_sample(same=True), "class", _PARAMS)
    assert res is not None
    assert res.tvd[0, 1] == pytest.approx(0.0, abs=1e-12)
    row = res.pair_rows[0]
    assert row["null_kind"] == "paired_release_swap"
    assert not row["exceeds_null_p95"]
    # 兩類位置完全相同，交換標籤不改變任何分布，置換基準也是 0。
    assert row["null_p95"] == pytest.approx(0.0, abs=1e-12)


def test_shifted_classes_exceed_paired_swap_noise_and_curves_integrate_to_100():
    res = abg.analyze_grouping(_sample(same=False), "class", _PARAMS)
    assert res is not None
    row = res.pair_rows[0]
    assert row["exceeds_null_p95"] and row["exceeds_null_p99"]
    assert row["tvd"] > row["null_p95"] > 0.0
    dx = LENGTH / _PARAMS.n_grid
    assert res.density_pct_per_km.sum(axis=1) * dx == pytest.approx([100.0, 100.0])
    assert res.edge_share_pct.sum(axis=1) == pytest.approx([100.0, 100.0])
    assert res.labels_zh  # 未登錄的類別以識別碼為名
    assert res.noise["half_split_particle"]["pooled"]["mean"] < row["tvd"]


def test_open_boundary_is_analyzed_with_reflection():
    res = abg.analyze_grouping(_sample(same=False, closed=False), "class", _PARAMS)
    assert res is not None and res.end_treatment == "reflect" and not res.closed
    table = abg.grouping_table_dict(res, case_id="c")
    assert table["boundary"]["end_treatment"] == "reflect"


def test_season_grouping_uses_event_label_permutation():
    res = abg.analyze_grouping(_sample(same=True), "season", _PARAMS)
    assert res is not None
    assert res.group_ids == ("DJF", "JJA")
    assert res.pair_rows[0]["null_kind"] == "event_label_permutation"
    assert res.labels_zh == ("冬季（12–2 月）", "夏季（6–8 月）")


def test_missing_label_column_returns_none():
    assert abg.analyze_grouping(_sample(same=True), "tide", _PARAMS) is None


# ── 圖檔名登記與端到端 ───────────────────────────────────────────────────


def test_figure_stems_are_registered_and_distinct_from_existing_stems():
    assert abgf.ARC_BY_GROUP_FIGURE_STEMS == (
        "boundary_arc_weight_by_class_overlay",
        "boundary_arc_weight_by_class_panels",
        "boundary_arc_weight_by_season_overlay",
        "boundary_arc_weight_by_season_panels",
        "boundary_arc_weight_by_tide_overlay",
        "boundary_arc_weight_by_tide_panels",
    )
    existing = set(native_figures.FIGURE_STEMS) | set(native_figures.FOOTPRINT_FIGURE_STEMS)
    assert existing.isdisjoint(abgf.ARC_BY_GROUP_FIGURE_STEMS)
    assert set(abgf.ARC_BY_GROUP_TEXT_DIRS) == set(native_figures.FIGURE_TEXT_MODES)


def test_cli_parser_accepts_multiple_cases():
    args = cli._native_arc_by_group_parser().parse_args(
        ["--case", "a", "/pp/a", "/out/a", "--case", "b", "/pp/b", "/out/b", "--sigma-km", "5"]
    )
    assert args.case == [["a", "/pp/a", "/out/a"], ["b", "/pp/b", "/out/b"]]
    assert args.sigma_km == 5.0


def _write_postprocess(root: Path, *, closed: bool, per_particle: bool = False) -> None:
    """寫一個最小的合成後處理目錄：兩站、四個類別、四季、兩種潮況。"""

    rng = np.random.default_rng(11)
    rows: dict[str, list] = {
        k: []
        for k in (
            "study_site_id",
            "material_id",
            "settling_velocity_mps",
            "crossed_interest_region_boundary",
            "boundary_arc_length_position_m",
            "receptor_id",
            "arrival_time_id",
            "member_id",
            "deposition_season",
            "deposition_tide_class",
        )
    }
    arc_rows, edge_rows, geom_rows = [], [], []
    seasons = ["DJF", "MAM", "JJA", "SON"]
    for site in ("site_a", "site_b"):
        counts = np.zeros(50)
        for arrival in range(8):
            for member in range(4):
                season = seasons[(arrival + member) % 4]
                tide = "spring_proxy" if (arrival + member) % 2 else "neap_proxy"
                for receptor in range(5):
                    centre = 100.0 + 60.0 * seasons.index(season)
                    for m_i, material in enumerate(("m1", "m2", "m3", "m4")):
                        s_km = float(np.clip(rng.normal(centre, 10.0), 0.5, 499.0))
                        rows["study_site_id"].append(site)
                        rows["material_id"].append(material)
                        rows["settling_velocity_mps"].append(-0.01 * (m_i + 1))
                        if per_particle:
                            # 每類 10 個候選速度（等比排列），每顆粒子抽一個；範圍 = 代表速度 × [0.1, 3]。
                            cand = -0.01 * (m_i + 1) * np.geomspace(0.1, 3.0, 10)
                            rows.setdefault("particle_settling_velocity_mps", []).append(float(cand[rng.integers(0, 10)]))
                        rows["crossed_interest_region_boundary"].append(True)
                        rows["boundary_arc_length_position_m"].append(s_km * 1000.0)
                        rows["receptor_id"].append(f"r{receptor}")
                        rows["arrival_time_id"].append(f"a{arrival}")
                        rows["member_id"].append(member)
                        rows["deposition_season"].append(season)
                        rows["deposition_tide_class"].append(tide)
                        counts[int(s_km // 10)] += 1
        for i in range(50):
            arc_rows.append(
                {
                    "study_site_id": site,
                    "bin_index": i,
                    "s_start_km": 10.0 * i,
                    "s_end_km": 10.0 * (i + 1),
                    "crossed_particle_count": int(counts[i]),
                    "fraction": float(counts[i] / counts.sum()),
                    "cumulative_fraction": float(counts[: i + 1].sum() / counts.sum()),
                    "boundary_length_km": 500.0,
                    "bin_km": 10.0,
                }
            )
        for kind, name, a, b in (
            ("edge", "south", 0.0, 200.0),
            ("edge", "east", 200.0, 300.0),
            ("edge", "north", 300.0, 400.0),
            ("edge", "west", 400.0, 500.0),
            ("corner", "south-east", 200.0, 200.0),
        ):
            edge_rows.append({"study_site_id": site, "kind": kind, "name": name, "s_start_km": a, "s_end_km": b})
        ring = [(121.0, 24.0), (122.0, 24.0), (122.0, 25.0), (121.0, 25.0)]
        if closed:
            ring.append(ring[0])
        for v, (lon, lat) in enumerate(ring):
            geom_rows.append(
                {
                    "study_site_id": site,
                    "element": "open_boundary",
                    "part_index": 0,
                    "vertex_index": v,
                    "lon_deg": lon,
                    "lat_deg": lat,
                    "s_km": 125.0 * v,
                    "label": "",
                }
            )
    root.mkdir(parents=True)
    pq.write_table(pa.table({k: pa.array(v) for k, v in rows.items()}), root / "departures.parquet")
    pq.write_table(pa.Table.from_pylist(arc_rows), root / "boundary_departure_along_arc_by_site.parquet")
    pq.write_table(pa.Table.from_pylist(edge_rows), root / "boundary_edge_intervals_by_site.parquet")
    pq.write_table(pa.Table.from_pylist(geom_rows), root / "footprint_geometry_by_site.parquet")
    (root / "manifest.json").write_text(json.dumps({"experiment_case_id": "case-test"}), encoding="utf-8")


def test_load_site_arc_samples_detects_closure(tmp_path):
    _write_postprocess(tmp_path / "closed", closed=True)
    _write_postprocess(tmp_path / "open", closed=False)
    closed = abg.load_site_arc_samples(tmp_path / "closed")
    opened = abg.load_site_arc_samples(tmp_path / "open")
    assert set(closed) == {"site_a", "site_b"}
    assert closed["site_a"].closed and closed["site_a"].end_treatment == "circular"
    assert not opened["site_a"].closed and opened["site_a"].end_treatment == "reflect"
    assert closed["site_a"].n_crossed == 8 * 4 * 5 * 4
    with pytest.raises(abg.ArcByGroupError):
        abg.load_site_arc_samples(tmp_path / "closed", ["no_such_site"])


def test_build_end_to_end_writes_registered_figures_tables_and_manifest(tmp_path):
    _write_postprocess(tmp_path / "pp_a", closed=True)
    _write_postprocess(tmp_path / "pp_b", closed=False)
    config = abgf.ArcByGroupConfig(
        cases=(
            abgf.ArcByGroupCase("case_a", tmp_path / "pp_a", tmp_path / "out_a"),
            abgf.ArcByGroupCase("case_b", tmp_path / "pp_b", tmp_path / "out_b"),
        ),
        params=abg.ArcByGroupParams(sigma_km=5.0, n_grid=200, n_resamples=10, seed=1),
        dpi=40,
    )
    outputs = abgf.build_arc_by_group(config, command="test")
    assert [p.name for p in outputs] == ["out_a", "out_b"]
    for out in outputs:
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["figure_stems"] == list(abgf.ARC_BY_GROUP_FIGURE_STEMS)
        for site in ("site_a", "site_b"):
            for sub in ("figures", "figures-notext"):
                for stem in abgf.ARC_BY_GROUP_FIGURE_STEMS:
                    for ext in ("png", "svg"):
                        assert (out / sub / site / f"{stem}.{ext}").is_file()
                        assert f"{sub}/{site}/{stem}.{ext}" in manifest["output_figures_sha256"]
            for grouping in abg.ARC_GROUPINGS:
                assert (out / "tables" / site / f"{grouping}_summary.json").is_file()
                assert (out / "tables" / site / f"{grouping}_curves.csv").is_file()
                entry = manifest["tables"][f"{site}/{grouping}"]
                assert entry["summary"]["pair_count"] >= 1
                assert set(entry["groups"][0]["edge_share_pct"]) == {"south", "east", "north", "west"}
        captions = (out / "captions.md").read_text(encoding="utf-8")
        readme = (out / "README.md").read_text(encoding="utf-8")
        for text in (captions, readme):
            assert "機率" not in text
            assert "））" not in text
            assert "跨站" not in text and "五站一致" not in text
        # 圖說可以有站名（站名只出現在檔名與圖說）。
        assert "site_a" in captions
    # 兩案的邊界閉合狀態不同，需分別記錄於表。
    ta = json.loads((outputs[0] / "tables" / "site_a" / "class_summary.json").read_text(encoding="utf-8"))
    tb = json.loads((outputs[1] / "tables" / "site_a" / "class_summary.json").read_text(encoding="utf-8"))
    assert ta["boundary"]["end_treatment"] == "circular"
    assert tb["boundary"]["end_treatment"] == "reflect"
    # 輸出目錄必須是全新目錄。
    with pytest.raises(abg.ArcByGroupError):
        abgf.build_arc_by_group(config, command="test")


def test_axes_are_shared_across_sites_and_cases(tmp_path):
    _write_postprocess(tmp_path / "pp_a", closed=True)
    config = abgf.ArcByGroupConfig(
        cases=(abgf.ArcByGroupCase("case_a", tmp_path / "pp_a", tmp_path / "out_a"),),
        params=abg.ArcByGroupParams(sigma_km=5.0, n_grid=200, n_resamples=5, seed=1),
        groupings=("season",),
        dpi=40,
    )
    (out,) = abgf.build_arc_by_group(config, command="test")
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    ticks = manifest["axes"]["y_ticks_pct_per_km"]["season"]
    assert ticks[0] == 0.0 and ticks[-1] == manifest["axes"]["y_max_pct_per_km"]["season"]
    assert manifest["axes"]["x_max_km"] == 500.0 and manifest["axes"]["x_ticks_km"][-1] == 500.0


# ── 速度標示：有、無逐粒子速度欄位 ───────────────────────────────────────


def _sample_with_speeds(per_particle: bool) -> abg.SiteArcSample:
    """兩類：甲類代表速度較慢、實際速度範圍 0.000897–0.107722；乙類代表速度較快、範圍 0.0342–0.345。"""

    base = _sample(same=False)
    n = base.s_km.size // 2
    rng = np.random.default_rng(5)
    cand_a = -np.geomspace(0.000897, 0.107722, 10)
    cand_b = -np.geomspace(0.034223, 0.344737, 10)
    actual = np.concatenate([cand_a[rng.integers(0, 10, n)], cand_b[rng.integers(0, 10, n)]])
    base.velocity_mps = np.where(base.material_id == "oca_a", -0.0114, -0.0959)
    base.particle_velocity_mps = actual if per_particle else None
    return base


def test_speed_text_is_a_range_when_particle_speeds_exist():
    res = abg.analyze_grouping(_sample_with_speeds(True), "class", _PARAMS)
    assert res is not None and res.velocity_summary is not None
    assert [abgf._speed_text(v) for v in res.velocity_summary] == ["0.000897–0.108 m/s", "0.0342–0.345 m/s"]
    assert abgf._has_speed_range(res)
    assert all(v["per_particle_speeds"] for v in res.velocity_summary)


def test_speed_text_is_a_single_value_without_particle_speeds():
    res = abg.analyze_grouping(_sample_with_speeds(False), "class", _PARAMS)
    assert res is not None and res.velocity_summary is not None
    assert [abgf._speed_text(v) for v in res.velocity_summary] == ["0.0114 m/s", "0.0959 m/s"]
    assert not abgf._has_speed_range(res)
    assert not any(v["per_particle_speeds"] for v in res.velocity_summary)


def test_classes_are_ordered_by_representative_speed_not_by_actual_minimum():
    sample = _sample_with_speeds(True)
    # 讓甲類代表速度變成較快，但實際範圍不變：排序應跟著代表速度改變。
    sample.velocity_mps = np.where(sample.material_id == "oca_a", -0.5, -0.0959)
    res = abg.analyze_grouping(sample, "class", _PARAMS)
    assert res is not None
    assert res.group_ids == ("oca_b", "oca_a")


def test_summary_table_records_min_max_and_distinct_speeds():
    res = abg.analyze_grouping(_sample_with_speeds(True), "class", _PARAMS)
    assert res is not None
    table = abg.grouping_table_dict(res, case_id="c")
    rec = table["groups"][0]["settling_speed_mps"]
    assert rec["abs_min_mps"] == pytest.approx(0.000897) and rec["abs_max_mps"] == pytest.approx(0.107722)
    assert rec["distinct_values"] == 10 and rec["source"] == "particle_actual"
    assert rec["representative_abs_mps"] == pytest.approx(0.0114)
    plain = abg.grouping_table_dict(abg.analyze_grouping(_sample_with_speeds(False), "class", _PARAMS), case_id="c")
    rec2 = plain["groups"][0]["settling_speed_mps"]
    assert rec2["distinct_values"] == 1 and rec2["source"] == "class_representative"


def test_loader_reads_particle_speed_column_only_when_present(tmp_path):
    _write_postprocess(tmp_path / "plain", closed=True)
    _write_postprocess(tmp_path / "per", closed=True, per_particle=True)
    plain = abg.load_site_arc_samples(tmp_path / "plain")["site_a"]
    per = abg.load_site_arc_samples(tmp_path / "per")["site_a"]
    assert plain.particle_velocity_mps is None
    assert per.particle_velocity_mps is not None and per.particle_velocity_mps.shape == per.s_km.shape


def test_captions_describe_ten_speeds_only_for_per_particle_data(tmp_path):
    for name, flag in (("with", True), ("without", False)):
        _write_postprocess(tmp_path / f"pp_{name}", closed=True, per_particle=flag)
        config = abgf.ArcByGroupConfig(
            cases=(abgf.ArcByGroupCase("c", tmp_path / f"pp_{name}", tmp_path / f"out_{name}"),),
            params=abg.ArcByGroupParams(sigma_km=5.0, n_grid=200, n_resamples=5, seed=1),
            groupings=("class",),
            dpi=40,
        )
        (out,) = abgf.build_arc_by_group(config, command="test")
        captions = (out / "captions.md").read_text(encoding="utf-8")
        table = json.loads((out / "tables" / "site_a" / "class_summary.json").read_text(encoding="utf-8"))
        assert ("每類有 10 個沉降速度，各占該類粒子的十分之一" in captions) is flag
        assert "））" not in captions and "（（" not in captions
        assert ("每類只有一個沉降速度" in captions) is (not flag)
        assert all(
            ("particle_actual" if flag else "class_representative") == g["settling_speed_mps"]["source"]
            for g in table["groups"]
        )


def test_three_significant_digits_keep_trailing_zero():
    assert abgf._three_sig(0.920397) == "0.920"
    assert abgf._three_sig(0.000897) == "0.000897"
    assert abgf._three_sig(0.107722) == "0.108"
    assert abgf._three_sig(0.0142) == "0.0142"
    assert abgf._three_sig(0.1) == "0.100"


def test_season_captions_have_no_nested_parentheses(tmp_path):
    _write_postprocess(tmp_path / "pp", closed=True)
    config = abgf.ArcByGroupConfig(
        cases=(abgf.ArcByGroupCase("c", tmp_path / "pp", tmp_path / "out"),),
        params=abg.ArcByGroupParams(sigma_km=5.0, n_grid=200, n_resamples=5, seed=1),
        groupings=("season",),
        dpi=40,
    )
    (out,) = abgf.build_arc_by_group(config, command="test")
    captions = (out / "captions.md").read_text(encoding="utf-8")
    assert "總變異距離（兩分布差的絕對值總和之半）最大為" in captions
    assert "））" not in captions
    assert "冬季與" in captions or "春季與" in captions or "夏季與" in captions
