"""完整情境交叉、ID 與 member seed 測試。"""

from __future__ import annotations

from dataclasses import replace

import pytest

from lagrangian_backtracking.scenarios import (
    BASELINE_BEHAVIORS,
    ArrivalTime,
    Receptor,
    build_scenarios,
    derive_member_seed,
    validate_baseline_coverage,
    validate_non_rising_behaviors,
)

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


def test_seed_is_stable_and_member_specific() -> None:
    """同一四元組重跑 seed 相同，member 或 experiment case 改變則不同。"""

    kwargs = {"master_seed": 42, "scenario_id": "scn_a", "experiment_case_id": "baseline"}
    first = derive_member_seed(**kwargs, member_id=0)
    assert first == derive_member_seed(**kwargs, member_id=0)
    assert first != derive_member_seed(**kwargs, member_id=1)
    assert first != derive_member_seed(
        master_seed=42, scenario_id="scn_a", experiment_case_id="no_stokes", member_id=0
    )
