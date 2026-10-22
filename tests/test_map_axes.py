"""座標軸統一規則（map_axes，WP-71）的單元測試：一位小數標籤、跨度與間隔相同、上下限在刻度上、包住需求範圍。"""

from __future__ import annotations

import math

import pytest

from lagrangian_backtracking.native_runner import map_axes as ma

# 五地點流場範圍外框（run09，經度下、上，緯度下、上）加 0.04 度餘量
_NEEDS = {
    "gongliao": (121.3063 - 0.04, 122.7937 + 0.04, 24.6008 - 0.04, 25.4992 + 0.04),
    "guishan": (121.3063 - 0.04, 122.7937 + 0.04, 24.6008 - 0.04, 25.4992 + 0.04),
    "hsinchu": (119.7074 - 0.04, 121.1919 + 0.04, 24.3006 - 0.04, 25.1994 + 0.04),
    "lienchiang": (119.1991 - 0.04, 120.7010 + 0.04, 25.7507 - 0.04, 26.6494 + 0.04),
    "nanwan": (120.1664 - 0.04, 121.6201 + 0.04, 21.5508 - 0.04, 22.4492 + 0.04),
}


def test_format_one_decimal_and_rejects_non_tenths() -> None:
    assert ma.format_lat(25.0) == "25.0°N"
    assert ma.format_lon(121.4) == "121.4°E"
    assert ma.format_lon(119.6 + 1e-12) == "119.6°E"
    with pytest.raises(ValueError):
        ma.format_lat(24.79)  # 不可把非整刻度四捨五入成 24.8 標示
    with pytest.raises(ValueError):
        ma.format_lon(120.25)


def test_spec_and_windows_are_identical_in_span_and_step() -> None:
    spec = ma.plan_axis_spec(_NEEDS, step_deg=0.2)
    assert spec.lon_span_deg == pytest.approx(1.8)
    assert spec.lat_span_deg == pytest.approx(1.2)
    windows = {k: ma.window_for_content(spec, v) for k, v in _NEEDS.items()}
    for site, w in windows.items():
        lon_lo, lon_hi, lat_lo, lat_hi = _NEEDS[site]
        # 包住需求範圍
        assert w.lon_lo <= lon_lo and w.lon_hi >= lon_hi and w.lat_lo <= lat_lo and w.lat_hi >= lat_hi
        # 跨度相同
        assert w.lon_hi - w.lon_lo == pytest.approx(spec.lon_span_deg)
        assert w.lat_hi - w.lat_lo == pytest.approx(spec.lat_span_deg)
        # 上下限在刻度（間隔整數倍）上，刻度含兩端、等間距
        for v in (w.lon_lo, w.lon_hi, w.lat_lo, w.lat_hi):
            assert abs(v / 0.2 - round(v / 0.2)) < 1e-6
        for ticks in (w.lon_ticks, w.lat_ticks):
            assert ma.sequence_is_regular(ticks, 0.2)
        assert w.lon_ticks[0] == w.lon_lo and w.lon_ticks[-1] == w.lon_hi
        assert w.lat_ticks[0] == w.lat_lo and w.lat_ticks[-1] == w.lat_hi
        # 每個刻度標籤都可一位小數表示
        assert all(ma.format_lon(v).endswith("°E") for v in w.lon_ticks)


def test_window_is_centered_when_slack_is_even() -> None:
    spec = ma.AxisSpec(0.2, 1.8, 1.2, 24.0)
    w = ma.window_for_content(spec, (120.1, 121.7, 24.5, 25.5))
    # 需求中心 120.9，窗口 [120.0, 121.8] 中心 120.9
    assert (w.lon_lo, w.lon_hi) == (120.0, 121.8)


def test_aspect_error_is_small_for_run09_sites() -> None:
    spec = ma.plan_axis_spec(_NEEDS, step_deg=0.2)
    for site, (_, _, lat_lo, lat_hi) in _NEEDS.items():
        err = ma.aspect_error_fraction(spec, (lat_lo + lat_hi) / 2.0)
        assert abs(err) < 0.02, (site, err)
    assert spec.aspect == pytest.approx(1.0 / math.cos(math.radians(spec.ref_lat_deg)))
    assert spec.frame_width_over_height > 1.0


def test_too_small_spec_fails_closed() -> None:
    spec = ma.AxisSpec(0.2, 0.4, 0.4, 24.0)
    with pytest.raises(ValueError):
        ma.window_for_content(spec, (120.0, 121.0, 24.0, 24.2))
