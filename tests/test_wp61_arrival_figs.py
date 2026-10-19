"""WP-61 腳本（scripts/wp61）的純函式測試：整齊座標軸、刻度格式、季節範圍。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "wp61"))

import arrival_figs as af  # noqa: E402


def test_nice_axis_covers_data_with_tidy_bounds() -> None:
    lo, up, step = af.nice_axis(0, 7.93, zero_based=True, max_int=5)
    assert (lo, up) == (0.0, 10.0) and step == 2.0
    lo, up, step = af.nice_axis(-0.45, 1.3, max_int=4)
    assert lo <= -0.45 and up >= 1.3
    n = round((up - lo) / step)
    assert 3 <= n <= 4
    assert np.isclose(lo / step, round(lo / step))


def test_axis_ticks_and_format() -> None:
    t = af.axis_ticks(0.0, 1.25, 0.25)
    assert t.tolist() == [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
    assert af.fmt_tick(1.5) == "1.5" and af.fmt_tick(2.0) == "2" and af.fmt_tick(-0.5) == "−0.5"
    assert af.fmt_tick(-0.0) == "0"


def test_season_spans_cover_year_exactly_once() -> None:
    spans = af.season_spans()
    assert spans[0][1] == 0 and spans[-1][2] == 365
    assert all(a[2] == b[1] for a, b in zip(spans, spans[1:], strict=False))
    assert [s[0] for s in spans] == ["DJF", "MAM", "JJA", "SON", "DJF"]
    ticks, labels = af.month_starts_days()
    assert ticks[0] == 0 and ticks[-1] == 365 and ticks[3] == 90 and len(labels) == 13
