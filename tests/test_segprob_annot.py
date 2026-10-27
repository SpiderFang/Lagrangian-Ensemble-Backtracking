"""WP-75：分段機率圖註記（集中段合併、邊小計文字、百分比四捨五入）的單元測試。"""

from __future__ import annotations

from decimal import Decimal

import numpy as np

from lagrangian_backtracking.native_runner import segprob_annot as sa
from lagrangian_backtracking.native_runner import segprob_figures as sf


def test_pct2_half_up_from_integer_counts() -> None:
    assert sf.pct2(31085, 100000) == "31.09"  # 浮點 31.08499… 會被誤判成 31.08
    assert sf.pct2(1, 100000) == "0.00"
    assert sf.pct2(5, 100000) == "0.01"  # 0.005 → 0.01（half-up）
    assert sf.pct2(73402, 100000) == "73.40"
    assert sf.pct2(100000, 100000) == "100.00"


def test_side_label_rules() -> None:
    assert sa.side_label(0, 100000) == "0.00"
    assert sa.side_label(4, 100000) == "<0.01"  # 有顆數但四捨五入為 0.00
    assert sa.side_label(5, 100000) == "0.01"
    assert sa.side_label(23158, 100000) == "23.16"
    assert sa.side_label(23158, 100000, decimals=0) == "23"
    assert sa.side_label(500, 100000, decimals=0) == "1"  # 0.5 → 1（half-up）


def test_merge_stretches_threshold_cap_and_wraparound() -> None:
    n = 1000
    counts = np.array([50, 30, 0, 25, 25, 25, 0, 0, 40, 60])  # 占比 5、3、0、2.5、2.5、2.5、0、0、4、6 %
    chains = [("c", list(range(10)), True)]  # 閉合：第 9 段與第 0 段相鄰
    st = sa.merge_stretches(
        counts, n, chains, min_seg_pct=Decimal("2"), min_total_pct=Decimal("10"), max_n=2, kind="boundary"
    )
    # 環繞合併：8、9、0、1 → 40+60+50+30=180（18%）；3–5 → 75（7.5%，不足 10% 捨去）
    assert [s.indices for s in st] == [[8, 9, 0, 1]]
    assert st[0].count == 180
    st2 = sa.merge_stretches(
        counts, n, chains, min_seg_pct=Decimal("2"), min_total_pct=Decimal("5"), max_n=1, kind="boundary"
    )
    assert len(st2) == 1 and st2[0].count == 180  # 只留合計最大者
    # 單段門檻是 ≥：恰 2.0% 的分段算
    c2 = np.array([20, 20, 19])
    s3 = sa.merge_stretches(
        c2,
        1000,
        [("c", [0, 1, 2], False)],
        min_seg_pct=Decimal("2"),
        min_total_pct=Decimal("1"),
        max_n=None,
        kind="coast",
    )
    assert s3[0].indices == [0, 1]


def test_thresholds_are_constants_in_one_place() -> None:
    assert (sa.STRETCH_BOUNDARY_MIN_SEG_PCT, sa.STRETCH_BOUNDARY_MIN_TOTAL_PCT) == (  # noqa: SIM300
        Decimal("2"),
        Decimal("10"),
    )
    assert sa.STRETCH_BOUNDARY_MAX_PER_PANEL == 2
    assert (sa.STRETCH_COAST_MIN_SEG_PCT, sa.STRETCH_COAST_MIN_TOTAL_PCT) == (Decimal("0.5"), Decimal("3"))  # noqa: SIM300
    assert sa.STRETCH_COAST_MAX_PER_PANEL == 3
