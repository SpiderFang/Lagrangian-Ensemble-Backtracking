"""pilot scenario selector 的 deterministic identity、分層與 fail-closed 契約測試。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from lagrangian_backtracking.pilot_selection import (
    PILOT_SCENARIO_SELECTION_RANKING_POLICY,
    PILOT_SCENARIO_SELECTION_STRATUM_FIELDS,
    apply_scenario_selection,
    build_full_scenario_selection,
    scenario_ids_sha256,
    select_pilot_scenarios,
    validate_scenario_selection_binding_shape,
)
from lagrangian_backtracking.scenarios import Receptor, Scenario, stable_identifier


def _records() -> tuple[tuple[Scenario, ...], tuple[Receptor, ...]]:
    """建立四個 site×vertical strata、每層三筆 scenario 的純 dataclass fixture。"""

    receptors: list[Receptor] = []
    scenarios: list[Scenario] = []
    for site_index, site_id in enumerate(("site-a", "site-b")):
        for vertical_index, vertical_id in enumerate(("surface", "deep")):
            receptor_id = f"{site_id}-{vertical_id}"
            receptors.append(
                Receptor(
                    receptor_id=receptor_id,
                    study_site_id=site_id,
                    analysis_region_id="region-a" if site_index == 0 else "region-b",
                    lon=120.0 + site_index,
                    lat=22.0 + vertical_index,
                    z_m_positive_up=-float(vertical_index + 1),
                    vertical_id=vertical_id,
                    metadata={},
                )
            )
            for sample_index in range(3):
                scenario_id = stable_identifier(
                    "scn",
                    [site_id, vertical_id, receptor_id, f"arrival-{sample_index}", "test-v1"],
                )
                scenarios.append(
                    Scenario(
                        scenario_id=scenario_id,
                        study_site_id=site_id,
                        analysis_region_id="region-a" if site_index == 0 else "region-b",
                        material_id=f"material-{sample_index}",
                        receptor_id=receptor_id,
                        arrival_time_id=f"arrival-{sample_index}",
                        settling_velocity_mps=-0.001,
                        arrival_time_utc_ns=1_700_000_000_000_000_000
                        + site_index * 100
                        + vertical_index * 10
                        + sample_index,
                        design_version="test-v1",
                    )
                )
    return tuple(scenarios), tuple(receptors)


def test_selection_is_order_independent_and_n1_is_n2_prefix_set() -> None:
    """輸入列順序不影響結果，N=1 選中集合必須是 N=2 的子集。"""

    scenarios, receptors = _records()
    selected_one, binding_one = select_pilot_scenarios(scenarios, receptors, 1, len(scenarios))
    selected_two, binding_two = select_pilot_scenarios(
        tuple(reversed(scenarios)),
        tuple(reversed(receptors)),
        2,
        len(scenarios),
    )

    selected_two_again, binding_two_again = select_pilot_scenarios(
        scenarios,
        receptors,
        2,
        len(scenarios),
    )
    assert {item.scenario_id for item in selected_one} <= {
        item.scenario_id for item in selected_two
    }
    assert tuple(item.scenario_id for item in selected_two) == tuple(
        item.scenario_id for item in selected_two_again
    )
    assert binding_two == binding_two_again
    assert binding_one["selected_scenario_count"] == 4
    assert binding_two["selected_scenario_count"] == 8
    assert binding_two["ranking_policy"] == PILOT_SCENARIO_SELECTION_RANKING_POLICY


def test_full_binding_and_apply_are_exact_and_order_independent() -> None:
    """full mode 保存 source=selected 的 hash，重排 current source 後仍可重算通過。"""

    scenarios, receptors = _records()
    assert isinstance(PILOT_SCENARIO_SELECTION_STRATUM_FIELDS, tuple)
    assert PILOT_SCENARIO_SELECTION_STRATUM_FIELDS == (
        "study_site_id",
        "receptor.vertical_id",
    )
    binding = build_full_scenario_selection(scenarios)
    assert binding["mode"] == "full"
    assert binding["samples_per_stratum"] is None
    assert binding["stratum_fields"] == []
    assert binding["strata"] == []
    assert binding["source_scenario_ids_sha256"] == binding["selected_scenario_ids_sha256"]
    selected = apply_scenario_selection(
        binding,
        tuple(reversed(scenarios)),
        receptors,
        len(scenarios),
        "pilot",
    )
    assert {item.scenario_id for item in selected} == {item.scenario_id for item in scenarios}
    validate_scenario_selection_binding_shape(binding, "formal", len(scenarios))


@pytest.mark.parametrize(
    "case",
    ("duplicate_receptor", "missing_receptor", "unknown_scenario_receptor", "bad_n", "bad_count"),
)
def test_selector_rejects_invalid_inputs(case: str) -> None:
    """duplicate/missing receptor、未知參照、N 與 source count 錯誤都必須拒絕。"""

    scenarios, receptors = _records()
    if case == "duplicate_receptor":
        invalid_receptors = receptors + (receptors[0],)
        with pytest.raises(ValueError, match="receptor_id"):
            select_pilot_scenarios(scenarios, invalid_receptors, 1, len(scenarios))
    elif case == "missing_receptor":
        invalid_receptors = receptors[1:]
        with pytest.raises(ValueError, match="找不到|strata"):
            select_pilot_scenarios(scenarios, invalid_receptors, 1, len(scenarios))
    elif case == "unknown_scenario_receptor":
        invalid_scenario = replace(scenarios[0], receptor_id="missing-receptor")
        with pytest.raises(ValueError, match="找不到"):
            select_pilot_scenarios((invalid_scenario,) + scenarios[1:], receptors, 1, len(scenarios))
    elif case == "bad_n":
        with pytest.raises(ValueError, match="正整數"):
            select_pilot_scenarios(scenarios, receptors, 0, len(scenarios))
    else:
        with pytest.raises(ValueError, match="expected_source"):
            select_pilot_scenarios(scenarios, receptors, 1, len(scenarios) + 1)


@pytest.mark.parametrize(
    "mutation",
    ("unknown", "missing", "type", "source_hash", "selected_hash", "count", "stratum"),
)
def test_binding_tamper_fails_closed(mutation: str) -> None:
    """binding 任一 root、型別、hash、count 或 strata 竄改都不得通過 apply。"""

    scenarios, receptors = _records()
    selected, binding = select_pilot_scenarios(scenarios, receptors, 1, len(scenarios))
    del selected
    tampered = deepcopy(binding)
    if mutation == "unknown":
        tampered["unexpected"] = None
    elif mutation == "missing":
        del tampered["strata"]
    elif mutation == "type":
        tampered["samples_per_stratum"] = True
    elif mutation == "source_hash":
        tampered["source_scenario_ids_sha256"] = "0" * 64
    elif mutation == "selected_hash":
        tampered["selected_scenario_ids_sha256"] = "0" * 64
    elif mutation == "count":
        tampered["selected_scenario_count"] = 2
    else:
        tampered["strata"][0]["selected_count"] = 2
    with pytest.raises(ValueError):
        apply_scenario_selection(tampered, scenarios, receptors, len(scenarios), "pilot")


def test_formal_rejects_stratified_binding_and_vertical_tamper_is_detected() -> None:
    """formal 不可使用 stratified；current receptor vertical 改變會使 pilot binding 失配。"""

    scenarios, receptors = _records()
    _selected, binding = select_pilot_scenarios(scenarios, receptors, 1, len(scenarios))
    with pytest.raises(ValueError, match="formal"):
        apply_scenario_selection(binding, scenarios, receptors, len(scenarios), "formal")
    changed_receptors = (
        replace(receptors[0], vertical_id="changed-vertical"),
        *receptors[1:],
    )
    with pytest.raises(ValueError, match="strata|binding"):
        apply_scenario_selection(binding, scenarios, changed_receptors, len(scenarios), "pilot")


def test_full_binding_selected_hash_tamper_is_rejected_by_shape_validator() -> None:
    """full binding 的 selected hash 被改寫時，shape gate 必須直接拒絕。"""

    scenarios, _receptors = _records()
    binding = build_full_scenario_selection(scenarios)
    tampered = deepcopy(binding)
    tampered["selected_scenario_ids_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        validate_scenario_selection_binding_shape(tampered, "pilot", len(scenarios))


def test_pilot_strata_source_count_sum_tamper_is_rejected_by_shape_validator() -> None:
    """任一 strata source count 被改寫而 root 總量不符時，shape gate 必須拒絕。"""

    scenarios, receptors = _records()
    _selected, binding = select_pilot_scenarios(scenarios, receptors, 1, len(scenarios))
    tampered = deepcopy(binding)
    tampered["strata"][0]["source_count"] += 1
    with pytest.raises(ValueError, match="source count"):
        validate_scenario_selection_binding_shape(tampered, "pilot", len(_selected))


def test_scenario_id_hash_rejects_duplicate_and_is_order_independent() -> None:
    """canonical scenario ID hash 不依賴輸入順序且拒絕重複 identity。"""

    assert scenario_ids_sha256(("a", "b")) == scenario_ids_sha256(("b", "a"))
    with pytest.raises(ValueError, match="重複"):
        scenario_ids_sha256(("a", "a"))
