"""WP-65：圓形來源方位圖報告版面（畫布寬＝報告插入寬度）的設定驗證與幾何不變量。"""

from pathlib import Path

import numpy as np
import pytest

from lagrangian_backtracking.native_runner import polar_source_figures as F
from lagrangian_backtracking.native_runner.polar_source import PolarSourceError


def _cfg(**kw):
    case = F.PolarCase("c", Path("a"), Path("b"), Path("d"))
    return F.PolarConfig(cases=(case,), release_config=Path("r"), coastline=Path("c"), bathymetry=None, **kw)


def test_report_layout_only_notext_default_envelope():
    _cfg(report_layout=True, modes=("none",))
    with pytest.raises(PolarSourceError):
        _cfg(report_layout=True)  # 預設 modes 含 full
    with pytest.raises(PolarSourceError):
        _cfg(report_layout=True, modes=("none",), envelope_auto=True)
    with pytest.raises(PolarSourceError):
        _cfg(report_layout=True, modes=("none",), radius_km=20.0)


def test_report_geometry_fits_canvas_and_page():
    assert F._RO_LEFT + F._RO_FRAME + F._RO_RIGHT_MARGIN == pytest.approx(F.REPORT_W_IN)
    assert F._RG_LEFT + 3 * F._RG_P + 2 * F._RG_GAP + F._RG_RIGHT_MARGIN == pytest.approx(F.REPORT_W_IN)
    assert F.REPORT_PANELS_H <= 9.0  # 版心高約 9.7 英寸，圖加圖說須在一頁內
    assert max(F.report_overview_height(k) for k in 'BCD') <= 9.0
    assert F.report_overview_height('D') < F.report_overview_height('C') < F.report_overview_height('B')
    assert set(F.REPORT_STEMS) == set(F.FIGURE_KINDS)


def test_report_ticks_regular_and_inner():
    t = F._report_ticks(121.6, 122.2, (0.1, 0.2), 5, inner=(0.055, 0.945))
    assert np.allclose(np.diff(t), 0.1) and t[0] > 121.6 and t[-1] < 122.2
    with pytest.raises(PolarSourceError):
        F._report_ticks(121.6, 121.7, (0.5,), 2)


def test_one_decimal_style_config_and_geometry():
    """WP-71：one_decimal 樣式只能搭配報告版面；畫布寬仍為報告插入寬度；刻度一位小數且不把非整刻度四捨五入。"""
    _cfg(report_layout=True, modes=("none",), report_axes_style="one_decimal")
    with pytest.raises(PolarSourceError):
        _cfg(report_axes_style="one_decimal")
    with pytest.raises(PolarSourceError):
        _cfg(report_layout=True, modes=("none",), report_axes_style="two_decimal")
    assert F._RO_LEFT_1D + F._RO_FRAME_1D + F._RO_RIGHT_MARGIN == pytest.approx(F.REPORT_W_IN)
    assert F._RG_LEFT_1D + 3 * F._RG_P_1D + 2 * F._RG_GAP + F._RG_RIGHT_MARGIN == pytest.approx(F.REPORT_W_IN)
    assert F.report_panels_height(True) <= 9.0 and max(F.report_overview_height(k, True) for k in "BCD") <= 9.0
    # 預設樣式之尺寸不變（舊版圖可逐位元重現）
    assert F.report_panels_height(False) == F.REPORT_PANELS_H
    # 刻度：0.1 度步距、只含框內整 0.1 度倍數；格式化函式拒絕非整刻度
    from lagrangian_backtracking.native_runner import map_axes as ma

    t = F._report_ticks(24.4194, 25.0806, (0.1,), 5, inner=(0.03, 0.97))
    assert "°" not in ma.format_lat(t[0]) and len(ma.format_lat(t[0]).split(".")[1]) == 1 and all(abs(v * 10 - round(v * 10)) < 1e-6 for v in t)
    with pytest.raises(ValueError):
        ma.format_lat(24.79)
