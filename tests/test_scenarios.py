"""完整情境交叉、ID 與 member seed 測試。"""

from __future__ import annotations

import csv
import math
from dataclasses import replace
from pathlib import Path

import pytest

from lagrangian_backtracking.scenarios import (
    BASELINE_BEHAVIORS,
    SETTLING_CANDIDATE_COUNT,
    ArrivalTime,
    Receptor,
    build_scenarios,
    derive_member_seed,
    validate_baseline_coverage,
    validate_non_rising_behaviors,
)

ROOT = Path(__file__).resolve().parents[1]

SITES = {
    "gongliao": "A",
    "guishan": "A",
    "hsinchu": "B",
    "houwan": "C",
    "lienchiang": "D",
}


def _full_manifests() -> tuple[list[Receptor], list[ArrivalTime]]:
    """建立只含必要欄位的五站 20 receptor／50 arrival 合成 manifests。"""

    receptors: list[Receptor] = []
    arrivals: list[ArrivalTime] = []
    for site, region in SITES.items():
        for index in range(20):
            receptors.append(
                Receptor(
                    receptor_id=f"{site}_r{index:02d}",
                    study_site_id=site,
                    analysis_region_id=region,
                    lon=121.0,
                    lat=24.0,
                    z_m_positive_up=-5.0,
                    vertical_id=f"z{index % 4}",
                    metadata={},
                )
            )
        for index in range(50):
            arrivals.append(
                ArrivalTime(
                    arrival_time_id=f"{site}_t{index:02d}",
                    study_site_id=site,
                    time_utc_ns=1_704_067_200_000_000_000 + index * 3_600_000_000_000,
                    year=2024,
                    season="winter",
                    tide_class="synthetic",
                    phase_or_event="synthetic",
                    metadata={},
                )
            )
    return receptors, arrivals


def test_full_cross_produces_50000_unique_scenarios() -> None:
    """五站各自完整交叉後必須恰為 50,000，不能在 A 區先合併 receptor。"""

    receptors, arrivals = _full_manifests()
    scenarios = build_scenarios(
        behaviors=BASELINE_BEHAVIORS,
        receptors=receptors,
        arrival_times=arrivals,
        design_version="design_baseline_v2_non_rising_oca_proxy",
    )
    counts = validate_baseline_coverage(scenarios)
    assert len(scenarios) == 50_000
    assert len({item.scenario_id for item in scenarios}) == 50_000
    assert counts["gongliao"] == counts["guishan"] == 10_000


def test_baseline_behaviors_are_named_strictly_sinking_proxies() -> None:
    """十類須一對一保存 iOcean 名稱、材質、形狀與限制，且不得含零速或上浮。"""

    validate_non_rising_behaviors(BASELINE_BEHAVIORS)
    assert len(BASELINE_BEHAVIORS) == 10
    assert len({item.oca_category_zh for item in BASELINE_BEHAVIORS}) == 10
    assert all(item.settling_velocity_mps < 0.0 for item in BASELINE_BEHAVIORS)
    assert {item.behavior_class for item in BASELINE_BEHAVIORS} == {"sinking"}
    assert all(item.material_family_zh for item in BASELINE_BEHAVIORS)
    assert all(item.representative_shape_zh for item in BASELINE_BEHAVIORS)
    assert all(item.applicability_condition_zh for item in BASELINE_BEHAVIORS)


@pytest.mark.parametrize("invalid_velocity", [0.0, 0.001])
def test_rejects_zero_or_rising_baseline_velocity(invalid_velocity: float) -> None:
    """PI 已取消中性與上浮情境，建表閘門必須對零值及正值失敗。"""

    invalid = (replace(BASELINE_BEHAVIORS[0], settling_velocity_mps=invalid_velocity),)
    with pytest.raises(ValueError, match="嚴格小於 0"):
        validate_non_rising_behaviors(invalid)


def _round_to_3_significant_figures(value: float) -> float:
    """把正值四捨五入到三位有效數字，對應 WP-35 定案值「取三位有效數字」的規則。

    有效數字從第一個非零位數開始算，因此對 ``0.005274`` 這種前導零很多的小數，
    三位有效數字是 ``0.00527``，不是固定小數位數的四捨五入。
    """

    if value == 0.0:
        return 0.0
    exponent = math.floor(math.log10(abs(value)))
    factor = 10.0 ** (2 - exponent)
    return round(value * factor) / factor


def test_baseline_behaviors_match_settling_velocity_derivation_source() -> None:
    """v7 來源一致性測試：每類 10 個候選沉降速度必須與推導來源逐值一致。

    直接讀 ``data/settling_velocity_derivation/results.csv``（50 列＝10 類 × 5 個
    代表物件，每列有 ``w_lo_ms`` 與 ``w_hi_ms`` 兩個解），逐 ``class_id`` 收集該類
    全部 10 個沉降速度解，檢查：

    1. 每類恰有 10 個解，``BASELINE_BEHAVIORS`` 的 ``settling_velocity_candidates_mps``
       由小到大（以量值計）逐值等於這 10 個解（取負號，六位小數）。
    2. 代表值為 10 個解的幾何平均取三位有效數字（只作摘要與圖表標示用）。
    3. 慢／快界為 10 個解的最小／最大值。

    類別對應刻意用 CSV 的 ``class_id``（1–10，對應 ``BASELINE_BEHAVIORS`` 索引
    0–9），不用中文類別名稱：CSV 使用全形斜線的「其他／不可回收」，若用中文名稱比對
    很容易因全形／半形斜線不一致而誤判。
    """

    csv_path = ROOT / "data" / "settling_velocity_derivation" / "results.csv"
    # 檔案帶 UTF-8 BOM，必須用 utf-8-sig 開啟。
    solutions_by_class_id: dict[int, list[float]] = {}
    row_count = 0
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            row_count += 1
            class_id = int(row["class_id"])
            solutions_by_class_id.setdefault(class_id, []).extend(
                (float(row["w_lo_ms"]), float(row["w_hi_ms"]))
            )

    assert row_count == 50
    assert len(solutions_by_class_id) == 10
    assert set(solutions_by_class_id) == set(range(1, 11))

    for class_id, solutions in solutions_by_class_id.items():
        assert len(solutions) == SETTLING_CANDIDATE_COUNT, (
            f"class_id={class_id} 應有 5 個代表物件 x 2 界＝10 個解"
        )
        behavior = BASELINE_BEHAVIORS[class_id - 1]

        expected_candidates = tuple(-round(value, 6) for value in sorted(solutions))
        assert tuple(behavior.settling_velocity_candidates_mps) == pytest.approx(
            expected_candidates, abs=1e-9
        ), f"class_id={class_id} 候選速度與 results.csv 不符"

        geometric_mean = math.prod(solutions) ** (1.0 / len(solutions))
        expected_representative = _round_to_3_significant_figures(geometric_mean)
        assert expected_representative == pytest.approx(
            abs(behavior.settling_velocity_mps), rel=1e-9
        ), f"class_id={class_id} 代表值與 results.csv 重算不符"

        assert round(min(solutions), 6) == pytest.approx(
            abs(behavior.settling_velocity_slow_bound_mps), abs=1e-9
        ), f"class_id={class_id} slow_bound 與 results.csv min 不符"
        assert round(max(solutions), 6) == pytest.approx(
            abs(behavior.settling_velocity_fast_bound_mps), abs=1e-9
        ), f"class_id={class_id} fast_bound 與 results.csv max 不符"

        source_path = ROOT / behavior.derivation_source
        assert source_path.is_file(), f"derivation_source 指向的檔案不存在：{source_path}"


def test_baseline_candidates_are_sorted_negative_and_cover_bounds() -> None:
    """每類恰 10 個候選：嚴格為負、量值非遞減，端點等於慢／快界，代表值落在其間。"""

    for behavior in BASELINE_BEHAVIORS:
        candidates = behavior.settling_velocity_candidates_mps
        assert len(candidates) == SETTLING_CANDIDATE_COUNT
        assert all(value < 0.0 for value in candidates)
        magnitudes = [abs(value) for value in candidates]
        assert magnitudes == sorted(magnitudes)
        assert candidates[0] == behavior.settling_velocity_slow_bound_mps
        assert candidates[-1] == behavior.settling_velocity_fast_bound_mps
        assert magnitudes[0] <= abs(behavior.settling_velocity_mps) <= magnitudes[-1]


def test_seed_is_stable_and_member_specific() -> None:
    """同一四元組重跑 seed 相同，member 或 experiment case 改變則不同。"""

    kwargs = {"master_seed": 42, "scenario_id": "scn_a", "experiment_case_id": "baseline"}
    first = derive_member_seed(**kwargs, member_id=0)
    assert first == derive_member_seed(**kwargs, member_id=0)
    assert first != derive_member_seed(**kwargs, member_id=1)
    assert first != derive_member_seed(
        master_seed=42, scenario_id="scn_a", experiment_case_id="no_stokes", member_id=0
    )
