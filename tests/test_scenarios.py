"""完整情境交叉、ID 與 member seed 測試。"""

from __future__ import annotations

from lagrangian_backtracking.scenarios import (
    BASELINE_BEHAVIORS,
    ArrivalTime,
    Receptor,
    build_scenarios,
    derive_member_seed,
    validate_baseline_coverage,
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
        design_version="design_baseline_v1",
    )
    counts = validate_baseline_coverage(scenarios)
    assert len(scenarios) == 50_000
    assert len({item.scenario_id for item in scenarios}) == 50_000
    assert counts["gongliao"] == counts["guishan"] == 10_000


def test_seed_is_stable_and_member_specific() -> None:
    """同一四元組重跑 seed 相同，member 或 experiment case 改變則不同。"""

    kwargs = {"master_seed": 42, "scenario_id": "scn_a", "experiment_case_id": "baseline"}
    first = derive_member_seed(**kwargs, member_id=0)
    assert first == derive_member_seed(**kwargs, member_id=0)
    assert first != derive_member_seed(**kwargs, member_id=1)
    assert first != derive_member_seed(
        master_seed=42, scenario_id="scn_a", experiment_case_id="no_stokes", member_id=0
    )
