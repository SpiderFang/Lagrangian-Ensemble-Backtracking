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
