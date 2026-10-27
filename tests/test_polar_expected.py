"""WP-75 試作：幾何期望與新主要來源方向規則的單元測試。"""

from __future__ import annotations

import math

import numpy as np
from shapely.geometry import LineString

from lagrangian_backtracking.native_runner import polar_expected as pe

KM = 111_194.9266  # 赤道上每度（公尺，與球面方位角換算用）


class _EquatorProj:
    """測試用投影：錨點在 (0°, 0°)，x、y 公尺直接換成經緯度（赤道附近誤差可忽略）。"""

    def unproject(self, x, y):
        return np.asarray(x) / KM, np.asarray(y) / KM


def test_straight_north_sector_length() -> None:
    # 錨點正北 50 km 處的一條東西向直線：正北扇區（±5°）涵蓋 2 × 50 × tan 5° ≈ 8.75 km
    line = LineString([(-100_000.0, 50_000.0), (100_000.0, 50_000.0)])
    out = pe.walk_length_by_sector(line, _EquatorProj(), (0.0, 0.0), step_m=50.0)
    assert math.isclose(out.sum(), 200_000.0, rel_tol=1e-9)
    assert math.isclose(out[0], 2 * 50_000.0 * math.tan(math.radians(5.0)), abs_tol=60.0)
    assert out[18] == 0.0 and out[27] == 0.0


def test_step_size_does_not_change_total() -> None:
    line = LineString([(-3_000.0, 20_000.0), (40_000.0, 20_000.0), (40_000.0, -30_000.0)])
    a = pe.walk_length_by_sector(line, _EquatorProj(), (0.0, 0.0), step_m=100.0)
    b = pe.walk_length_by_sector(line, _EquatorProj(), (0.0, 0.0), step_m=20.0)
    assert math.isclose(a.sum(), line.length, rel_tol=1e-9)
    assert np.abs(a - b).max() < 100.0  # 步長 100 m：每個扇區交界最多差一步


def test_expected_sums_to_cross_plus_coast() -> None:
    geo = pe.SiteGeometry(
        np.arange(36.0) + 1.0, np.r_[np.zeros(10), np.arange(26.0) + 1.0], 1.0, 666.0, 351.0
    )  # noqa: E501
    e = pe.expected_pct(geo, 70.0, 25.0)
    assert math.isclose(e.sum(), 95.0, rel_tol=1e-12)
    assert e[:10].min() > 0  # 只有邊界也大於 0


def test_zero_length_sector_has_zero_expected() -> None:
    b = np.ones(36)
    b[20:26] = 0.0
    c = np.ones(36)
    c[20:26] = 0.0
    geo = pe.SiteGeometry(b, c, 1.0, b.sum(), c.sum())
    assert (pe.expected_pct(geo, 60.0, 30.0)[20:26] == 0).all()


def test_circular_runs_wrap_and_rank() -> None:
    f = np.zeros(36, bool)
    f[[34, 35, 0, 1]] = True
    f[[10, 11]] = True
    runs = pe.circular_runs(f)
    assert sorted(map(tuple, runs)) == [(10, 11), (34, 35, 0, 1)]
    obs = np.zeros(36)
    obs[[10, 11]] = 5.0
    obs[[34, 35, 0, 1]] = 1.0
    assert pe.ranked_runs(f, obs)[0] == [10, 11]
    assert pe.run_label([34, 35, 0, 1]) == ("34–1", "335°–15°")
    assert pe.circular_runs(np.zeros(36, bool)) == []
    assert pe.circular_runs(np.ones(36, bool)) == [list(range(36))]


def test_old_rule_uses_integer_comparison() -> None:
    counts = np.zeros(36, np.int64)
    counts[0] = 2778  # 2778 / 100000 = 2.778% > 2.7777…%
    counts[1] = 2777  # 2.777% < 2.7777…%
    flags = pe.old_flags(counts, 100_000)
    assert flags[0] and not flags[1]


def test_new_rule_strict_greater() -> None:
    obs = np.array([1.0, 2.0, 0.0])
    exp = np.array([1.0, 1.0, 0.0])
    assert pe.new_flags(obs, exp).tolist() == [False, True, False]
