"""WP-73：報告圖字級與版面與 TRAPs 一致（report_style，fs8）的單元測試。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lagrangian_backtracking.native_runner import domain_fraction as df
from lagrangian_backtracking.native_runner import report_style as rs

pytest.importorskip("matplotlib")


def test_canvas_and_frame_match_traps_concise_map() -> None:
    # TRAPs 簡要報告大圖：畫布 6.3 × 4.13（TRAPs 文字區 6.3 吋；LBT 為 6.27 吋），主圖框 4.080 × 2.98 吋、左緣 0.91、下緣 0.95
    assert rs.canvas_in(2) == (6.27, 4.13)
    assert rs.canvas_in(3) == (6.27, 4.29)
    ratio = 1.8 * np.cos(np.radians(24.1)) / 1.2
    left, bottom, w, h = rs.frame_box_in(ratio, 2)
    assert (left, bottom, h) == (0.91, 0.95, 2.98)
    assert abs(w - 4.080) < 0.001  # 2.98 × 1.8 × cos(24.1°) / 1.2（TRAPs CONCISE_MAP_WIDTH_INCH 的程式值）
    assert (rs.CBAR_W_IN, rs.CBAR_PAD_IN) == (0.12, 0.093)
    assert (rs.FRAME_LW_PT, rs.FRAME_COLOUR, rs.TICK_LEN_PT, rs.TICK_W_PT) == (1.1, "#222222", 3.5, 0.8)
    assert rs.TEXT_PT == 8.0


def test_print_scale_factors_match_old_canvas_widths() -> None:
    assert abs(rs.PRINT_SCALE_TRAJ - 6.27 / 9.48) < 1e-9
    assert abs(rs.PRINT_SCALE_CURVE - 6.27 / 8.0) < 1e-9
    assert 0.70 < rs.PRINT_SCALE_FOOTPRINT < 0.71


def test_scale_print_effect_scales_lines_markers_and_patches_once() -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots()
    (ln,) = ax.plot([0, 1], [0, 1], linewidth=2.0, marker="o", markersize=10.0, markeredgewidth=1.0)
    ax.add_patch(Rectangle((0, 0), 1, 1, linewidth=1.0))
    n = rs.scale_print_effect(ax.get_children(), 0.5)
    assert n >= 2
    assert ln.get_linewidth() == pytest.approx(1.0)
    assert ln.get_markersize() == pytest.approx(5.0)
    assert ln.get_markeredgewidth() == pytest.approx(0.5)
    assert ax.patches[0].get_linewidth() == pytest.approx(0.5)
    assert all(sp.get_linewidth() == pytest.approx(0.8) for sp in ax.spines.values())  # 框線不被縮放
    plt.close(fig)


def test_ncol_for_rows_limits_rows() -> None:
    assert rs.ncol_for_rows(8, 2) == 4
    assert rs.ncol_for_rows(11, 3) == 4
    assert rs.ncol_for_rows(1, 2) == 1


def _font_name() -> str:
    from lagrangian_backtracking.native_runner.figures import _resolve_font

    return _resolve_font(None).font_properties.get_name()


def _write_departures(path: Path, durations_days: list[float]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    n = len(durations_days)
    pq.write_table(
        pa.table(
            {
                "study_site_id": ["siteA"] * n,
                "material_id": ["oca_steel_rigid_cylinder", "oca_glass_bottle_or_fragment"] * (n // 2),
                "final_status_code": pa.array([1] * (n - 1) + [2], type=pa.int32()),
                "backtrack_duration_seconds": [d * 86400.0 for d in durations_days],
            }
        ),
        path,
    )


def test_curve_report_fs8_canvas_fonts_and_no_grid(tmp_path: Path) -> None:
    post = tmp_path / "post"
    post.mkdir()
    _write_departures(post / "departures.parquet", [1.0, 20.0, 95.0, 120.0, 179.0, 180.0])
    out = df.build_domain_fraction(post, tmp_path / "f", tmp_path / "n", max_days=180.0, dpi=60, report_fs8=True)
    assert len(out) == 2 and all(p.parent == tmp_path / "n" / "siteA" for p in out)
    import json

    man = json.loads((tmp_path / "f" / "manifest.json").read_text(encoding="utf-8"))
    for rec in man["report_fs8"]["records"].values():
        assert rec["canvas_in"] == [6.27, 4.29]
        assert rec["frame_box_in_left_bottom_w_h"][:2] == pytest.approx([0.91, 1.11])
        assert rec["grid"] is False
        assert rec["text_summary"]["sizes_pt"] == [8.0] and rec["text_summary"]["min_size_pt"] == 8.0
        assert rec["text_summary"]["font_families"] == [_font_name()]
        assert rec["legend"]["margin_left_in"] > 0 and rec["legend"]["margin_right_in"] > 0 and rec["legend"]["margin_bottom_in"] > 0
        assert rec["legend"]["rows"] <= 3
    from PIL import Image

    with Image.open(out[0]) as im:
        assert im.size == (round(6.27 * 60), round(4.29 * 60))  # 畫布即輸出尺寸（不裁切、不縮放）


def test_report_fs8_requires_report_rc_and_none_mode(tmp_path: Path) -> None:
    curves = df.SiteCurves("siteA", np.arange(0, 181, 1.0), 4, np.ones(181), ("oca_steel_rigid_cylinder",), (4,), (np.ones(181),), 180.0)
    from lagrangian_backtracking.native_runner.figures import _Labeler, _resolve_font

    font = _resolve_font(None).font_properties
    with pytest.raises(df.DomainFractionError):
        df.render_site_figure(curves, tmp_path / "x.png", mode="full", font=font, t=_Labeler(True), report_fs8=True)
    with pytest.raises(RuntimeError):
        df.render_site_figure(curves, tmp_path / "y.png", mode="none", font=font, t=_Labeler(True), report_fs8=True)


def test_curve_report_fs8_plots_identical_data_as_legacy_layout(tmp_path: Path) -> None:
    """舊版面與報告版（fs8）畫出的線條資料（x、y 座標位元組）逐位元相同：版面改變不動數值。"""

    from lagrangian_backtracking.native_runner.figures import _Labeler, _resolve_font
    from lagrangian_backtracking.native_runner import report_style as _rs

    font = _resolve_font(None).font_properties
    grid = np.arange(0, 181, 0.25)
    cur = np.clip(1.0 - grid / 40.0, 0.0, 1.0)
    curves = df.SiteCurves("siteA", grid, 1000, cur, ("oca_steel_rigid_cylinder", "oca_glass_bottle_or_fragment"), (500, 500),
                           (np.clip(1.0 - grid / 30.0, 0, 1), np.clip(1.0 - grid / 50.0, 0, 1)), 40.0)
    for ys in ("linear", "log"):
        old = df.render_site_figure(curves, tmp_path / f"o_{ys}.png", mode="none", font=font, t=_Labeler(True), dpi=40, yscale=ys)
        with _rs.report_rc(font.get_name()):
            new = df.render_site_figure(curves, tmp_path / f"n_{ys}.png", mode="none", font=font, t=_Labeler(True), dpi=40, yscale=ys, report_fs8=True)
        assert old["data_fingerprint"] == new["data_fingerprint"]
        assert old["data_fingerprint"]["lines"] == 2 + 1 + 2  # 兩類別線、全部線、兩條 90／180 天標線
