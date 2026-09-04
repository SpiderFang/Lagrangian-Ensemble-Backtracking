"""pilot scenario selector 的 deterministic identity、分層與 fail-closed 契約測試。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from functools import lru_cache

import pytest

import lagrangian_backtracking.pilot_selection as selection_module
from lagrangian_backtracking.pilot_selection import (
    PILOT_EXACT_SELECTION_SCHEMA_VERSION,
    PILOT_SCENARIO_SELECTION_RANKING_POLICY,
    PILOT_SCENARIO_SELECTION_STRATUM_FIELDS,
    apply_scenario_selection,
    build_full_scenario_selection,
    scenario_ids_sha256,
    select_exact_pilot_scenarios,
    select_pilot_scenarios,
    validate_scenario_selection_binding_shape,
)
from lagrangian_backtracking.runner import iter_run_units, plan_scenario_shards
from lagrangian_backtracking.scenarios import (
    BASELINE_BEHAVIORS,
    ArrivalTime,
    Receptor,
    Scenario,
    build_scenarios,
    stable_identifier,
    validate_baseline_coverage,
)


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


@lru_cache(maxsize=1)
def _complete_exact_records() -> tuple[tuple[Scenario, ...], tuple[Receptor, ...], tuple[ArrivalTime, ...]]:
    """建立完整五萬情境的合成資料，驗證集合契約而非真實海洋成果。

    五站各有 5 水平×4 垂向受體、50 個唯一到達 ID，沿用十個負沉降代理與正式情境
    建立函式，再通過既有完整覆蓋驗證。固定 UTC 奈秒與公尺深度只供測試使用；呼叫端
    不得修改共用記錄的 metadata，變異案例應以 replace 建立新物件。
    """

    receptors = []
    arrivals = []
    for site, region in (
        ("gongliao", "A"), ("guishan", "A"), ("hsinchu", "B"), ("houwan", "C"), ("lienchiang", "D"),
    ):
        for horizontal in range(5):
            for vertical in range(4):
                receptors.append(Receptor(
                    receptor_id=f"{site}-h{horizontal}-v{vertical}", study_site_id=site,
                    analysis_region_id=region, lon=120.0 + horizontal * 0.001, lat=24.0,
                    z_m_positive_up=-float(vertical + 1), vertical_id=f"v{vertical}",
                    metadata={"horizontal_id": f"h{horizontal}"},
                ))
        for index in range(50):
            arrivals.append(ArrivalTime(
                arrival_time_id=f"{site}-arrival-{index}", study_site_id=site,
                time_utc_ns=1_700_000_000_000_000_000 + index * 3_600_000_000_000,
                year=2023, season="autumn", tide_class="spring", phase_or_event="test",
                metadata={},
            ))
    scenarios = tuple(build_scenarios(
        behaviors=BASELINE_BEHAVIORS, receptors=receptors, arrival_times=arrivals,
        design_version="synthetic-exact-test-v1",
    ))
    validate_baseline_coverage(scenarios)
    return scenarios, tuple(receptors), tuple(arrivals)


def _select_exact(
    scenarios: tuple[Scenario, ...], receptors: tuple[Receptor, ...], **overrides: object,
) -> tuple[tuple[Scenario, ...], dict]:
    """以小型合成來源的已知組合呼叫公開入口，僅在拒絕測試覆寫指定參數。"""

    kwargs = {
        "study_site_id": "site-a", "arrival_id": "arrival-0", "material_id": "material-0",
        "run_kind": "pilot", "expected_source_scenario_count": len(scenarios),
    }
    kwargs.update(overrides)
    return select_exact_pilot_scenarios(scenarios, receptors, **kwargs)


def test_exact_full_50000_preserves_all_receptors_and_nested_member_seeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """完整來源先驗證，選中 20 個原受體且不同 M 共用前綴成員 seed；不宣稱軌跡收斂。"""

    source, receptors, _ = _complete_exact_records()
    kwargs = {"study_site_id": "hsinchu", "arrival_id": "hsinchu-arrival-7",
              "material_id": BASELINE_BEHAVIORS[4].material_id, "run_kind": "pilot"}
    chosen, binding = select_exact_pilot_scenarios(source, receptors, 50_000, **kwargs)
    again, reordered = select_exact_pilot_scenarios(
        tuple(reversed(source)), tuple(reversed(receptors)), 50_000, **kwargs,
    )
    assert chosen == again and binding == reordered
    assert binding["schema_version"] == PILOT_EXACT_SELECTION_SCHEMA_VERSION
    assert binding["source_scenario_count"] == 50_000
    assert len(chosen) == binding["source_site_receptor_count"] == 20
    expected = {item.receptor_id: item.vertical_id for item in receptors if item.study_site_id == "hsinchu"}
    assert {item.receptor_id for item in chosen} == set(expected)
    assert {vertical: list(expected.values()).count(vertical) for vertical in set(expected.values())} == {
        f"v{index}": 5 for index in range(4)
    }
    originals = {item.scenario_id: item for item in source}
    assert all(item is originals[item.scenario_id] and item.settling_velocity_mps < 0 for item in chosen)
    seeds = []
    for members in (2, 4):
        shards = plan_scenario_shards(chosen, members_per_scenario=members,
                                     shard_scenario_count=7, experiment_case_id="no_stokes")
        seeds.append({unit.particle_id: unit.seed for shard in shards
                      for unit in iter_run_units(shard, master_seed=123)})
    assert len(seeds[0]) == 40 and len(seeds[1]) == 80
    assert all(seeds[1][key] == value for key, value in seeds[0].items())

    def cannot_select(*args: object, **kwargs: object) -> None:
        """若來源數量未驗證就解析受體，表示裁剪來源能進入選擇流程。"""
        raise AssertionError("來源數量驗證必須先於受體篩選")

    monkeypatch.setattr(selection_module, "_scenario_context", cannot_select)
    with pytest.raises(ValueError, match="expected_source_scenario_count"):
        select_exact_pilot_scenarios(chosen, receptors, 50_000, **kwargs)


@pytest.mark.parametrize("overrides", (
    {"run_kind": "formal"}, {"run_kind": "synthetic"}, {"study_site_id": "missing"},
    {"study_site_id": ""}, {"arrival_id": "unknown"}, {"material_id": "unknown"},
    {"arrival_id": "arrival-1"}, {"material_id": " material-0"}, {"arrival_id": None},
))
def test_exact_rejects_unknown_empty_cross_combination_and_nonpilot(overrides: dict) -> None:
    """未知、空白、到達／材質不配對及非 pilot 模式不能產生部分選擇。"""

    source, receptors = _records()
    with pytest.raises(ValueError):
        _select_exact(source, receptors, **overrides)


@pytest.mark.parametrize("mutation", ("missing", "extra", "duplicate", "non_sinking"))
def test_exact_requires_each_source_receptor_once(mutation: str) -> None:
    """來源受體集合不能缺漏、增補或重複，沉降代理不能變為零或正值。"""

    source, receptors = _records()
    if mutation == "missing":
        source = source[1:]
    elif mutation == "extra":
        receptors += (replace(receptors[0], receptor_id="extra-source-receptor"),)
    elif mutation == "duplicate":
        source += (replace(source[0], scenario_id="duplicate-pair-different-id"),)
    else:
        source = (replace(source[0], settling_velocity_mps=0.0), *source[1:])
    with pytest.raises(ValueError):
        _select_exact(source, receptors)


@pytest.mark.parametrize("field,value", (
    ("schema_version", "1.0.0"), ("mode", "full"), ("selection_policy", "unknown"),
    ("study_site_id", "site-b"), ("arrival_time_id", "arrival-1"), ("material_id", "material-1"),
    ("source_scenario_count", 2), ("selected_scenario_count", True),
    ("source_site_receptor_count", 1), ("source_site_receptor_ids_sha256", "0" * 64),
    ("source_scenario_ids_sha256", "0" * 64), ("selected_scenario_ids_sha256", "0" * 64),
    ("source_records_sha256", "0" * 64), ("samples_per_stratum", 1),
))
def test_exact_binding_tamper_is_rejected(field: str, value: object) -> None:
    """繫結的版本、識別碼、計數、雜湊及形式混用皆須由重算或形狀驗證拒絕。"""

    source, receptors = _records()
    chosen, binding = _select_exact(source, receptors)
    assert apply_scenario_selection(binding, source, receptors, len(source), "pilot") == chosen
    binding[field] = value
    with pytest.raises(ValueError):
        apply_scenario_selection(binding, source, receptors, len(source), "pilot")


@pytest.mark.parametrize(
    "mutation", ("unselected_value", "vertical", "receptor_position", "source_id", "missing_field"),
)
def test_exact_reopen_rejects_stale_source_even_with_unchanged_ids(mutation: str) -> None:
    """不只選中列：未選中情境物性、受體垂向／位置或來源識別改變也必須拒絕。"""

    source, receptors = _records()
    _, binding = _select_exact(source, receptors)
    if mutation == "unselected_value":
        source = (*source[:-1], replace(source[-1], settling_velocity_mps=-0.2))
    elif mutation == "vertical":
        receptors = (replace(receptors[0], vertical_id="changed"), *receptors[1:])
    elif mutation == "receptor_position":
        receptors = (replace(receptors[0], lon=121.0), *receptors[1:])
    elif mutation == "source_id":
        source = (*source[:-1], replace(source[-1], scenario_id="changed-source-id"))
    else:
        del binding["source_records_sha256"]
    with pytest.raises(ValueError):
        apply_scenario_selection(binding, source, receptors, len(source), "pilot")


@pytest.mark.parametrize("run_kind", ("formal", "synthetic"))
def test_exact_binding_cannot_be_relabelled_as_formal_or_synthetic(run_kind: str) -> None:
    """執行計畫 validator 使用的公開形狀驗證也必須拒絕精確模式改標籤。"""

    source, receptors = _records()
    chosen, binding = _select_exact(source, receptors)
    with pytest.raises(ValueError, match="pilot"):
        validate_scenario_selection_binding_shape(binding, run_kind, len(chosen))
