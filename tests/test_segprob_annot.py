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


def test_side_text_rules() -> None:
    assert sa.side_text(0, Decimal("0.00")) == "0.00"
    assert sa.side_text(4, Decimal("0.00")) == "<0.01"  # 有顆數但顯示值為 0.00
    assert sa.side_text(5, Decimal("0.01")) == "0.01"
    assert sa.side_text(23158, Decimal("23.16")) == "23.16"
    assert sa.side_text(23158, Decimal("23"), decimals=0) == "23"


def test_largest_remainder_sums_exactly() -> None:
    # 南灣全年四邊：逐邊四捨五入（half-up）合計 68.93，但穿越比例為 68.91；最大餘數法要恰為 68.91
    c = {"N": 10675, "E": 5105, "S": 9479, "W": 43655}
    v = sa.largest_remainder(c, sum(c.values()), 100000, 2)
    assert v == {"N": Decimal("10.67"), "E": Decimal("5.10"), "S": Decimal("9.48"), "W": Decimal("43.66")}
    assert sum(v.values()) == Decimal("68.91")
    assert sum(Decimal(sf.pct2(x, 100000)) for x in c.values()) == Decimal("68.93")
    # 0 位小數：目標為 round(總比例)
    v0 = sa.largest_remainder(
        {"N": 2500, "E": 100, "S": 4500, "W": 400}, 7500, 10000, 0
    )  # 25.00、1.00、45.00、4.00
    assert sum(v0.values()) == 75
    v1 = sa.largest_remainder(
        {"N": 2349, "E": 51, "S": 4551, "W": 349}, 7300, 10000, 0
    )  # 23.49、0.51、45.51、3.49
    assert sum(v1.values()) == 73 and v1["S"] == 46 and v1["E"] == 1 and v1["N"] == 23 and v1["W"] == 3
    # 單一值仍是 half-up
    assert sf.pct2(10675, 100000) == "10.68"


def test_circled_numbers() -> None:
    assert sa.circled(1) == "①" and sa.circled(3) == "③"


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
        Decimal("1"),
        Decimal("3"),
    )
    assert sa.STRETCH_BOUNDARY_MAX_NUMBERED == 3
    assert (sa.STRETCH_COAST_MIN_SEG_PCT, sa.STRETCH_COAST_MIN_TOTAL_PCT) == (Decimal("0.5"), Decimal("3"))  # noqa: SIM300
    assert sa.STRETCH_COAST_MAX_PER_PANEL == 3


def test_boundary_value_variants_prefer_two_decimals_then_integers() -> None:
    st = [
        sa.Stretch("boundary", "邊界", [0], 42150),
        sa.Stretch("boundary", "邊界", [5], 17780),
        sa.Stretch("boundary", "邊界", [9], 5120),
        sa.Stretch("boundary", "邊界", [12], 3010),
    ]  # noqa: E501
    v = sa.boundary_value_variants(st, 100_000)
    assert v[0] == "① 42.15%　② 17.78%　③ 5.12%"  # 只編前 3 段
    assert v[1] == "① 42.15%　② 17.78%\n③ 5.12%"
    assert v[2] == "① 42% ② 18% ③ 5%"
    assert v[3] == "①42% ②18% ③5%"
    assert v[-1] == "①42% ②18%\n③5%"
    assert sa.boundary_value_variants([], 100) == [""]
    # 整數用 ROUND_HALF_UP：2.5 → 3、0.5 → 1
    h = sa.boundary_value_variants(
        [sa.Stretch("boundary", "邊界", [0], 25), sa.Stretch("boundary", "邊界", [3], 5)], 1000
    )
    assert "① 3% ② 1%" in h


def test_title_values_add_rest_and_sum_to_crossing() -> None:
    # 各自四捨五入會加成 5.01，最大餘數法補回 5.00
    st = [sa.Stretch("boundary", "邊界", [0], 4285)]
    v = sa.boundary_value_variants(st, 100_000, crossing=5000)
    assert v[0] == "① 4.29%　其餘 0.71%"
    tbl = sa.title_value_table(st, 100_000, 5000)
    assert tbl == [("①", "4.29", "4.29"), ("其餘", "0.72", "0.71")]
    st3 = [sa.Stretch("boundary", "邊界", [0], 42150), sa.Stretch("boundary", "邊界", [5], 17780)]
    assert sa.boundary_value_variants(st3, 100_000, crossing=73402)[0] == "① 42.15%　② 17.78%　其餘 13.47%"


def test_bracket_must_span_exactly_the_stretch() -> None:
    from shapely.geometry import LineString
    from shapely.ops import substring

    line = LineString([(0, 0), (100, 0), (100, 100)])  # 含轉角的集中段
    for side in (5.0, -5.0):
        off = line.offset_curve(side)
        assert sa._span_error_m(line, off) < 1e-6  # 平行線兩端垂足恰在集中段兩端
    # 截短（例如為了閃避陸地）會被抓出來
    short = substring(line.offset_curve(5.0), 10.0, 150.0)
    assert sa._span_error_m(line, short) > 9.0
