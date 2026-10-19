"""WP-63：回溯時間內仍在流場區域內比例曲線的單元測試。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lagrangian_backtracking.native_runner import domain_fraction as df


def test_inside_fraction_is_exact_step_function() -> None:
    day = 86400.0
    dur = np.array([1.0, 2.0, 2.0, 5.0, 10.0, 10.0]) * day
    status = np.array([1, 1, 1, 1, 2, 4])  # 前四顆離開開放邊界；後兩顆沒有
    grid = np.array([0.0, 1.0, 1.5, 2.0, 4.99, 5.0, 180.0])
    got = df.inside_fraction(dur, status, grid)
    # 離開時間 <= τ 的顆數：0、1、1、3、3、4、4；總顆數 6
    assert got == pytest.approx(1.0 - np.array([0, 1, 1, 3, 3, 4, 4]) / 6.0)
    assert got[-1] == pytest.approx(2.0 / 6.0)  # 未離開者永遠算在域內


def test_inside_fraction_rejects_bad_input() -> None:
    with pytest.raises(df.DomainFractionError):
        df.inside_fraction(np.array([1.0]), np.array([1, 1]), np.array([0.0]))
    with pytest.raises(df.DomainFractionError):
        df.inside_fraction(np.array([-1.0]), np.array([1]), np.array([0.0]))
    with pytest.raises(df.DomainFractionError):
        df.inside_fraction(np.array([]), np.array([]), np.array([0.0]))


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


def test_build_writes_figures_table_manifest_without_site_names(tmp_path: Path) -> None:
    post = tmp_path / "post"
    post.mkdir()
    _write_departures(post / "departures.parquet", [1.0, 20.0, 95.0, 120.0, 179.0, 180.0])
    out = df.build_domain_fraction(post, tmp_path / "f", tmp_path / "n", max_days=180.0, dpi=40)
    names = sorted(p.name for p in out)
    assert names.count("domain_fraction_vs_backtrack_time.png") == 2
    assert names.count("domain_fraction_vs_backtrack_time_logy.png") == 2
    assert (tmp_path / "f" / "siteA" / "domain_fraction_vs_backtrack_time.svg").is_file()
    rows = (tmp_path / "f" / "inside_fraction_summary.csv").read_text(encoding="utf-8").splitlines()
    assert rows[0].endswith("inside_pct_at_90d,inside_pct_at_180d")
    allrow = next(r for r in rows if r.startswith("siteA,all,"))
    # 90 天時：離開者為 1、20 天（2 顆）；其餘 4 顆仍在域內 -> 66.6667%；180 天時：status 2 的最後一顆仍在 -> 1/6
    assert allrow.endswith("66.6667,16.6667")
    svg = (tmp_path / "f" / "siteA" / "domain_fraction_vs_backtrack_time.svg").read_text(encoding="utf-8")
    assert "siteA" not in svg
    # 不覆寫
    with pytest.raises(df.DomainFractionError):
        df.build_domain_fraction(post, tmp_path / "f", tmp_path / "n2", max_days=180.0, dpi=40)


def test_longer_than_axis_fails_closed(tmp_path: Path) -> None:
    post = tmp_path / "post"
    post.mkdir()
    _write_departures(post / "departures.parquet", [1.0, 2.0, 3.0, 181.0])
    with pytest.raises(df.DomainFractionError, match="超過圖的上限"):
        df.build_domain_fraction(post, tmp_path / "f", tmp_path / "n", max_days=180.0, dpi=40)
