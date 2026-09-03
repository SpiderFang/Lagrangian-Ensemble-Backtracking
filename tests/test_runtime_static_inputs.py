"""runtime static input binding 的工程契約測試。

本檔只驗證「已發布 run 的靜態輸入繫結」：run plan、設定、scenario／dynamic initial
condition manifest、幾何 bundle、inventory gate、scenario execution ordering，以及
controller 建構前後的責任邊界。測試重用 ``test_runtime.py`` 的小型 pilot／formal fixture，
不複製正式五站 inventory，也不建立任何 OCM schema 3、NWW3 schema 1 或真實 SERVER 資料。

所有數值、material 與 geometry 都只是 synthetic 工程驗證資料；通過測試不代表 OCM/NWW
科學產品、海流場、來源足跡、來源機率或觀測結果已完成驗收。static helper 的目的，是
讓 runtime 與 aggregate pipeline 共用同一套 read-only binding，而不是在 helper 中開啟
forcing 陣列或執行粒子平流。
"""

from __future__ import annotations

import inspect
import json
import shutil
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import pytest
from test_runtime import (
    EXAMPLE_CONFIG,
    _configured_project_config,
    _formal_test_data,
    _geometry_bundle,
    _initialize_formal_test_run,
    _initialize_pilot_test_run,
    _patch_pilot_orchestration,
    _pilot_inventory,
    _pilot_provenance,
    _scenario_inputs,
    _scenario_records,
    _write_inventory,
)

import lagrangian_backtracking.runtime as runtime
from lagrangian_backtracking.config import ProjectConfig
from lagrangian_backtracking.manifests import BoundaryGeometryBundle, ScenarioInputs
from lagrangian_backtracking.run_control import (
    RunWorkspace,
    _scenario_hash,
    initialize_run_workspace,
    load_run_plan,
)
from lagrangian_backtracking.runner import scenario_execution_sort_key
from lagrangian_backtracking.scenarios import stable_identifier


class _UninspectablePath:
    """若 runtime 提前把 forcing sentinel 轉成 Path，立即暴露責任邊界回歸。"""

    def __fspath__(self) -> str:
        """禁止測試中的 static／controller 前置流程探查 forcing 路徑。"""

        raise AssertionError("static binding 不得探查 forcing path")


def _runtime_data() -> dict[str, Any]:
    """建立一組與既有 runtime 測試相同的小型 synthetic pilot records。

    這裡只呼叫 ``test_runtime.py`` 已使用的 component／geometry 建構 helper；資料仍是
    一個 scenario、兩個 ensemble members 的工程 fixture，不代表實際 OCM/NWW 產品。
    """

    config = _configured_project_config()
    behavior, receptor, arrival, pair, scenario = _scenario_records(config)
    inputs = _scenario_inputs(behavior, receptor, arrival, pair, (scenario,))
    geometries = _geometry_bundle(receptor)
    inputs = replace(
        inputs,
        canonical_component_hashes={
            "material": "1" * 64,
            "receptor": "2" * 64,
            "arrival": "3" * 64,
            "receptor_arrival_initial_condition": "4" * 64,
        },
    )
    geometries = replace(
        geometries,
        canonical_component_hashes={
            "domain": "5" * 64,
            "local": "6" * 64,
            "open_boundary": "7" * 64,
        },
    )
    return {
        "config": config,
        "behavior": behavior,
        "receptor": receptor,
        "arrival": arrival,
        "pair": pair,
        "scenario": scenario,
        "inputs": inputs,
        "geometries": geometries,
    }


@pytest.fixture(scope="module")
def pilot_template(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """發布一份可重複複製的 pilot workspace template，避免各案例重建完整 fixture。

    initializer 使用既有 test_runtime 的 fake loader，故只產生 plan、progress、scenario
    table、seed table 與小型 inventory；static helper 測試之後再把此目錄複製到各案例的
    ``tmp_path``，任何 mutation 都不會污染 module template。
    """

    data = _runtime_data()
    base = tmp_path_factory.mktemp("runtime-static-pilot")
    with pytest.MonkeyPatch.context() as patch:
        workspace, _ = _initialize_pilot_test_run(
            data,
            base,
            patch,
            run_id="static-pilot-template",
        )
    return {"data": data, "root": workspace.path}


@pytest.fixture(scope="module")
def two_scenario_template(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """建立兩個 scenario 的小型 plan，專門驗證 reverse loader order 與 shard hash。

    第二個 arrival 只改變 synthetic fixture 的 arrival ID／UTC 奈秒；所有 component hash
    與 config 仍沿用既有 fixture。run-control initializer 會先依版本化 execution key
    排序，因此此 template 可作為「loader 原始順序反向、published scenario table 仍固定」
    的最小 regression。
    """

    data = _runtime_data()
    arrival = data["arrival"]
    pair = data["pair"]
    scenario = data["scenario"]
    second_arrival_id = "arrival-2"
    second_time_ns = arrival.time_utc_ns - 3_600_000_000_000
    second_arrival = replace(
        arrival,
        arrival_time_id=second_arrival_id,
        time_utc_ns=second_time_ns,
    )
    second_pair = replace(
        pair,
        arrival_time_id=second_arrival_id,
        time_utc_ns=second_time_ns,
    )
    second_scenario = replace(
        scenario,
        scenario_id=stable_identifier(
            "scn",
            [
                scenario.study_site_id,
                scenario.material_id,
                scenario.receptor_id,
                second_arrival_id,
                scenario.design_version,
            ],
        ),
        arrival_time_id=second_arrival_id,
        arrival_time_utc_ns=second_time_ns,
    )
    data["inputs"] = replace(
        data["inputs"],
        arrival_times=(arrival, second_arrival),
        scenarios=(scenario, second_scenario),
        initial_conditions=(pair, second_pair),
        initial_conditions_by_pair={
            (pair.receptor_id, pair.arrival_time_id): pair,
            (second_pair.receptor_id, second_pair.arrival_time_id): second_pair,
        },
    )
    base = tmp_path_factory.mktemp("runtime-static-two-scenario")
    with pytest.MonkeyPatch.context() as patch:
        workspace, _ = _initialize_pilot_test_run(
            data,
            base,
            patch,
            run_id="static-two-scenario-template",
        )
    return {"data": data, "root": workspace.path}


@pytest.fixture(scope="module")
def formal_template(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """發布一份只有一筆 scenario、但具有完整 formal inventory topology 的 template。

    ``_formal_test_data`` 與 ``_formal_inventory`` 來自既有 runtime 測試；它們只建立
    metadata 摘要，不建立四域 forcing array。這足以驗證 static helper 的 formal=True
    config／scenario／geometry／inventory gate，而不把正式五站資料複製進本檔。
    """

    data = _formal_test_data(_runtime_data())
    base = tmp_path_factory.mktemp("runtime-static-formal")
    with pytest.MonkeyPatch.context() as patch:
        workspace, _ = _initialize_formal_test_run(
            data,
            base,
            patch,
            run_id="static-formal-template",
        )
    return {"data": data, "root": workspace.path}


def _copy_workspace(template: dict[str, Any], destination: Path) -> Path:
    """把 module template 複製成案例專用 workspace，保留普通檔案 bytes 與 topology。"""

    root = destination / Path(template["root"]).name
    shutil.copytree(template["root"], root, symlinks=True)
    return root


def _snapshot_tree(root: Path) -> dict[str, bytes]:
    """以相對路徑與普通檔案 bytes 保存 workspace snapshot，不保存絕對路徑。"""

    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def _static_loader() -> Any:
    """取得 production static loader；缺少時保留明確的 production discrepancy。"""

    candidate = getattr(runtime, "load_validated_run_static_inputs", None)
    if not callable(candidate):
        pytest.fail(
            "runtime 尚未提供公開 load_validated_run_static_inputs API；"
            "static input binding production 尚未落檔"
        )
    return candidate


def _static_class() -> type[Any]:
    """取得 production immutable result class，不以測試 fake 掩蓋缺少的 public API。"""

    candidate = getattr(runtime, "ValidatedRunStaticInputs", None)
    if not isinstance(candidate, type):
        pytest.fail(
            "runtime 尚未提供公開 ValidatedRunStaticInputs；"
            "static input binding production 尚未落檔"
        )
    return candidate


def _patch_static_loaders(
    monkeypatch: pytest.MonkeyPatch,
    data: dict[str, Any],
    *,
    scenario_inputs: ScenarioInputs | None = None,
) -> dict[str, Any]:
    """以既有小型 manifest records 取代磁碟 loader，並記錄 static gate 的模式參數。

    真實 loader 的責任已在 ``test_runtime.py`` 覆蓋；本檔只關心 static helper 是否以
    plan run kind 傳入 formal／pilot、是否保存同一份 validated records，以及是否在
    forcing 前完成 binding。fake loader 不讀 OCM/NWW，也不產生任何科學結果。
    """

    calls: dict[str, Any] = {}
    selected_inputs = scenario_inputs or data["inputs"]

    def fake_load_config(path: str | Path, *, formal_release: bool) -> ProjectConfig:
        """回傳已由既有 fixture 建立的 config，保存 formal_release 呼叫證據。"""

        calls["config"] = {"path": path, "formal_release": formal_release}
        return data["config"]

    def fake_load_scenario_inputs(config: ProjectConfig, **kwargs: Any) -> ScenarioInputs:
        """回傳 synthetic dynamic pair records，記錄不含 forcing 的 loader 參數。"""

        calls["scenario"] = {"config": config, "kwargs": kwargs}
        return selected_inputs

    def fake_load_boundary_geometries(
        config: ProjectConfig,
        **kwargs: Any,
    ) -> BoundaryGeometryBundle:
        """回傳公尺制幾何 bundle，記錄 formal flag 而不開啟外部資料。"""

        calls["geometry"] = {"config": config, "kwargs": kwargs}
        return data["geometries"]

    monkeypatch.setattr(runtime, "load_config", fake_load_config)
    monkeypatch.setattr(runtime, "load_scenario_inputs", fake_load_scenario_inputs)
    monkeypatch.setattr(runtime, "load_boundary_geometries", fake_load_boundary_geometries)
    return calls


def _assert_no_path_leak(error: BaseException, tmp_path: Path) -> None:
    """固定 static/open 失敗的安全邊界：例外與 repr 不得洩漏暫存絕對路徑。"""

    assert str(tmp_path) not in str(error)
    assert str(tmp_path) not in repr(error)


def _assert_snapshot_has_no_path_state(snapshot: Any, tmp_path: Path) -> None:
    """遞迴檢查 result 沒有 workspace/config/checkpoint Path 狀態或暫存絕對路徑。

    ``ProjectConfig`` 允許保存 manifest 的相對 token，因此本 helper 不把所有含 ``/`` 的
    字串當成路徑；它只拒絕 Path 物件、top-level path-bearing 欄位，以及本案例 workspace
    的絕對文字。plan 內固定的 ``shards``／``checkpoints``／``failures`` 相對 token 仍合法。
    """

    forbidden_names = {
        "workspace",
        "workspace_path",
        "source_run_root",
        "config_path",
        "checkpoint_root",
        "ocm_native_root",
        "nww_analysis_root",
    }
    if is_dataclass(snapshot):
        assert not forbidden_names.intersection(field.name for field in fields(snapshot))

    def walk(value: Any, *, field_name: str | None = None) -> None:
        """遞迴走訪 mapping、tuple、dataclass 與 Pydantic model 的可保存狀態。"""

        assert not isinstance(value, Path)
        if isinstance(value, str):
            assert str(tmp_path) not in value
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                walk(item, field_name=key if isinstance(key, str) else None)
            return
        if isinstance(value, (tuple, list, set, frozenset)):
            for item in value:
                walk(item)
            return
        if is_dataclass(value):
            for field in fields(value):
                walk(getattr(value, field.name), field_name=field.name)
            return
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            walk(model_dump(mode="python"))

    walk(snapshot)


def _json_plan(plan: Any) -> dict[str, Any]:
    """把 immutable plan 轉成比較用 JSON object，不改動來源 mapping。"""

    def to_plain(value: Any) -> Any:
        """把 mapping proxy 與 tuple 遞迴轉成 JSON encoder 可接受的 plain value。"""

        if isinstance(value, Mapping):
            return {key: to_plain(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [to_plain(item) for item in value]
        return value

    return json.loads(json.dumps(to_plain(plan), ensure_ascii=False, sort_keys=True))


def _changed_scenario_inputs(data: dict[str, Any]) -> ScenarioInputs:
    """建立 count 與 component hash 都相同、但 plan-order scenario hash 不同的 loader 結果。"""

    scenario = data["scenario"]
    changed_design = "synthetic-design-v2"
    changed = replace(
        scenario,
        design_version=changed_design,
        scenario_id=stable_identifier(
            "scn",
            [
                scenario.study_site_id,
                scenario.material_id,
                scenario.receptor_id,
                scenario.arrival_time_id,
                changed_design,
            ],
        ),
    )
    return replace(
        data["inputs"],
        scenarios=(changed,),
        design_version=changed_design,
    )


def _make_synthetic_workspace(data: dict[str, Any], destination: Path) -> RunWorkspace:
    """用既有 run-control writer 建立 synthetic workspace，驗證 runtime 不接受此模式。"""

    return initialize_run_workspace(
        destination,
        run_id="static-synthetic",
        scenarios=data["inputs"].scenarios,
        normalized_config=data["config"].normalized_payload(),
        config_hash=data["config"].config_hash(),
        input_inventory_file=_pilot_inventory(data["config"].config_hash()),
        component_canonical_hashes=data["inputs"].canonical_component_hashes,
        geometry_canonical_hashes=data["geometries"].canonical_component_hashes,
        provenance=_pilot_provenance(),
        experiment_case_id="no_stokes",
        master_seed=20260828,
        seed_policy="sha256_v1_pcg64dxsm",
        members_per_scenario=2,
        shard_scenario_count=1,
        checkpoint_interval_sweeps=1,
        active_chunk_size=2,
        run_kind="synthetic",
    )


def test_load_static_inputs_returns_exact_typed_pilot_binding_and_keeps_workspace_read_only(
    pilot_template: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pilot helper 應回傳 exact class、typed records、execution order，且不改 workspace。"""

    data = pilot_template["data"]
    root = _copy_workspace(pilot_template, tmp_path)
    calls = _patch_static_loaders(monkeypatch, data)
    before = _snapshot_tree(root)

    result = _static_loader()(
        root,
        config_path=EXAMPLE_CONFIG,
        require_complete=False,
        expected_run_kind="pilot",
    )

    assert type(result) is _static_class()
    plan = result.plan
    assert plan["run_id"] == "static-pilot-template"
    assert plan["run_kind"] == "pilot"
    assert plan["experiment_case_id"] == "no_stokes"
    assert plan["scenario_count"] == 1
    assert plan["particle_count"] == 2
    assert plan["shard_count"] == 1
    assert isinstance(result.config, ProjectConfig)
    assert isinstance(result.scenario_inputs, ScenarioInputs)
    assert isinstance(result.geometries, BoundaryGeometryBundle)
    assert tuple(result.scenario_inputs.scenarios) == tuple(
        sorted(result.scenario_inputs.scenarios, key=scenario_execution_sort_key)
    )
    assert calls["config"] == {"path": EXAMPLE_CONFIG, "formal_release": False}
    assert calls["scenario"]["config"] is data["config"]
    assert calls["scenario"]["kwargs"] == {
        "config_path": EXAMPLE_CONFIG,
        "require_dynamic_initial_conditions": True,
        "formal": False,
    }
    assert calls["geometry"] == {
        "config": data["config"],
        "kwargs": {"config_path": EXAMPLE_CONFIG, "formal": False},
    }
    assert _snapshot_tree(root) == before


def test_pilot_selection_is_published_and_static_loader_recomputes_full_source(
    two_scenario_template: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pilot N=1 只發布一筆 selected，static loader 仍以完整 current source 重算。"""

    base = two_scenario_template["data"]
    scenario_config = base["config"].scenarios.model_copy(update={"scenario_count": 2})
    data = {**base, "config": base["config"].model_copy(update={"scenarios": scenario_config})}
    inventory_path = tmp_path / "pilot-selection-inventory.json"
    _write_inventory(inventory_path, _pilot_inventory(data["config"].config_hash()))
    _patch_pilot_orchestration(monkeypatch, data)

    workspace = runtime.initialize_pilot_run(
        config_path=EXAMPLE_CONFIG,
        input_inventory_path=inventory_path,
        destination=tmp_path / "runs",
        run_id="pilot-selection",
        experiment_case_id="no_stokes",
        project_root=tmp_path / "project-root",
        declared_git_commit="0123456789abcdef0123456789abcdef01234567",
        pilot_scenarios_per_stratum=1,
    )
    plan = load_run_plan(workspace)
    selection = plan["scenario_selection"]
    assert selection["mode"] == "pilot_stratified"
    assert selection["source_scenario_count"] == 2
    assert selection["selected_scenario_count"] == 1
    assert plan["scenario_count"] == 1

    _patch_static_loaders(monkeypatch, data)
    result = _static_loader()(workspace.path, config_path=EXAMPLE_CONFIG, expected_run_kind="pilot")
    assert len(result.scenario_inputs.scenarios) == 1
    assert len(result.scenario_inputs.materials) == len(data["inputs"].materials)
    assert len(result.scenario_inputs.receptors) == len(data["inputs"].receptors)
    assert len(result.scenario_inputs.arrival_times) == len(data["inputs"].arrival_times)
    assert len(result.scenario_inputs.initial_conditions) == len(data["inputs"].initial_conditions)


@pytest.mark.parametrize("tamper", ("source", "vertical"))
def test_static_loader_rejects_current_pilot_source_or_vertical_tamper(
    two_scenario_template: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    """current full source 的 ID 或 receptor vertical 改變都不得繞過 pilot binding。"""

    base = two_scenario_template["data"]
    scenario_config = base["config"].scenarios.model_copy(update={"scenario_count": 2})
    data = {**base, "config": base["config"].model_copy(update={"scenarios": scenario_config})}
    inventory_path = tmp_path / f"tamper-{tamper}.json"
    _write_inventory(inventory_path, _pilot_inventory(data["config"].config_hash()))
    _patch_pilot_orchestration(monkeypatch, data)
    workspace = runtime.initialize_pilot_run(
        config_path=EXAMPLE_CONFIG,
        input_inventory_path=inventory_path,
        destination=tmp_path / "runs",
        run_id=f"pilot-tamper-{tamper}",
        experiment_case_id="no_stokes",
        project_root=tmp_path / "project-root",
        declared_git_commit="0123456789abcdef0123456789abcdef01234567",
        pilot_scenarios_per_stratum=1,
    )
    if tamper == "source":
        changed = replace(
            data["inputs"].scenarios[1],
            scenario_id=stable_identifier(
                "scn",
                ["changed", data["inputs"].scenarios[1].arrival_time_id],
            ),
        )
        changed_inputs = replace(data["inputs"], scenarios=(data["inputs"].scenarios[0], changed))
    else:
        changed_receptor = replace(data["inputs"].receptors[0], vertical_id="changed-vertical")
        changed_inputs = replace(data["inputs"], receptors=(changed_receptor,))
    _patch_static_loaders(monkeypatch, data, scenario_inputs=changed_inputs)
    with pytest.raises(ValueError):
        _static_loader()(workspace.path, config_path=EXAMPLE_CONFIG, expected_run_kind="pilot")


def test_static_helper_has_no_forcing_factory_or_manager_side_effect(
    pilot_template: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """helper 沒有 forcing 參數，且不應建立 factory、manager 或探查 OCM/NWW。"""

    data = pilot_template["data"]
    root = _copy_workspace(pilot_template, tmp_path)
    _patch_static_loaders(monkeypatch, data)
    calls = {"factory": 0, "manager": 0}

    def blocked_factory(**_: Any) -> None:
        """static binding 階段若建立 factory，立即暴露 runtime／aggregate 邊界錯誤。"""

        calls["factory"] += 1
        raise AssertionError("static helper 不得建立 RuntimeRequestFactory")

    def blocked_manager(**_: Any) -> None:
        """static binding 階段若建立 forcing manager，立即暴露 OCM/NWW eager load。"""

        calls["manager"] += 1
        raise AssertionError("static helper 不得建立 ForcingWindowManager")

    monkeypatch.setattr(runtime, "RuntimeRequestFactory", blocked_factory)
    monkeypatch.setattr(runtime.ForcingWindowManager, "from_roots", staticmethod(blocked_manager))
    loader = _static_loader()
    parameters = inspect.signature(loader).parameters
    assert "ocm_native_root" not in parameters
    assert "nww_analysis_root" not in parameters

    # 不可 inspect 的 forcing sentinel 僅能交給後續 controller/factory；static API 明確
    # 沒有 forcing 參數，所以這裡刻意不把它塞入 checkpoint_root 偽裝成另一種路徑。
    forcing_sentinel = _UninspectablePath()
    assert forcing_sentinel is not None
    result = loader(root, config_path=EXAMPLE_CONFIG, expected_run_kind="pilot")

    assert type(result) is _static_class()
    assert calls == {"factory": 0, "manager": 0}


def test_static_plan_snapshot_is_recursive_immutable_defensive_and_path_free(
    pilot_template: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """plan snapshot 的 top-level、shard row 與 code provenance 都必須不可改且不存路徑。"""

    data = pilot_template["data"]
    root = _copy_workspace(pilot_template, tmp_path)
    _patch_static_loaders(monkeypatch, data)
    loader = _static_loader()
    result = loader(root, config_path=EXAMPLE_CONFIG, expected_run_kind="pilot")
    second = loader(root, config_path=EXAMPLE_CONFIG, expected_run_kind="pilot")
    expected = _json_plan(result.plan)
    disk_plan = load_run_plan(root)

    with pytest.raises(TypeError):
        result.plan["run_id"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        result.plan["shards"][0]["shard_id"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        result.plan["code_provenance"]["git_commit"] = "f" * 40  # type: ignore[index]

    assert _json_plan(result.plan) == expected
    assert _json_plan(second.plan) == expected
    assert disk_plan == expected
    _assert_snapshot_has_no_path_state(result, tmp_path)


def test_static_helper_reorders_reverse_loader_two_scenario_and_rechecks_plan_shard_hashes(
    two_scenario_template: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """兩情境 loader 即使反向回傳，helper 仍須符合 immutable table 與 plan-order hash。"""

    data = two_scenario_template["data"]
    root = _copy_workspace(two_scenario_template, tmp_path)
    ordered = tuple(sorted(data["inputs"].scenarios, key=scenario_execution_sort_key))
    reversed_inputs = replace(data["inputs"], scenarios=tuple(reversed(ordered)))
    _patch_static_loaders(monkeypatch, data, scenario_inputs=reversed_inputs)

    result = _static_loader()(root, config_path=EXAMPLE_CONFIG, expected_run_kind="pilot")
    scenarios = tuple(result.scenario_inputs.scenarios)
    assert scenarios == ordered
    assert len(scenarios) == 2
    assert all(
        scenario.scenario_id
        == stable_identifier(
            "scn",
            [
                scenario.study_site_id,
                scenario.material_id,
                scenario.receptor_id,
                scenario.arrival_time_id,
                scenario.design_version,
            ],
        )
        for scenario in scenarios
    )

    for row in result.plan["shards"]:
        start = row["scenario_start_index"]
        stop = row["scenario_stop_index"]
        subset = scenarios[start:stop]
        assert row["scenario_count"] == len(subset)
        assert row["scenario_hash"] == _scenario_hash(subset)
        assert [item.scenario_id for item in subset]
        assert row["analysis_region_id"] == subset[0].analysis_region_id
        assert row["arrival_time_utc_ns"] == subset[0].arrival_time_utc_ns


def test_static_plan_order_hash_gate_rejects_changed_identity_with_same_count_and_component_hashes(
    pilot_template: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """current loader 改 scenario identity 時，不能只靠 count/hash mapping 放行。"""

    data = pilot_template["data"]
    root = _copy_workspace(pilot_template, tmp_path)
    changed_inputs = _changed_scenario_inputs(data)
    assert len(changed_inputs.scenarios) == len(data["inputs"].scenarios)
    assert changed_inputs.canonical_component_hashes == data["inputs"].canonical_component_hashes
    assert _scenario_hash(changed_inputs.scenarios) != load_run_plan(root)["shards"][0]["scenario_hash"]
    _patch_static_loaders(monkeypatch, data, scenario_inputs=changed_inputs)

    with pytest.raises((TypeError, ValueError)) as error_info:
        _static_loader()(root, config_path=EXAMPLE_CONFIG, expected_run_kind="pilot")
    _assert_no_path_leak(error_info.value, tmp_path)


def test_formal_static_helper_revalidates_formal_loaders_and_inventory(
    formal_template: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """formal=True 必須重跑 config、scenario、geometry 與完整 inventory semantic gate。"""

    data = formal_template["data"]
    root = _copy_workspace(formal_template, tmp_path)
    calls = _patch_static_loaders(monkeypatch, data)
    real_gate = runtime._validated_formal_inventory
    gate_calls: list[dict[str, Any]] = []

    def recording_gate(
        inventory_path: str | Path,
        *,
        expected_config_hash: str,
        config: ProjectConfig,
        scenario_inputs: ScenarioInputs,
    ) -> dict[str, Any]:
        """保留真實 formal inventory gate，只記錄 helper 是否再次套用它。"""

        gate_calls.append(
            {
                "path": Path(inventory_path),
                "expected_config_hash": expected_config_hash,
                "config": config,
                "scenario_inputs": scenario_inputs,
            }
        )
        return real_gate(
            inventory_path,
            expected_config_hash=expected_config_hash,
            config=config,
            scenario_inputs=scenario_inputs,
        )

    monkeypatch.setattr(runtime, "_validated_formal_inventory", recording_gate)
    result = _static_loader()(
        root,
        config_path=EXAMPLE_CONFIG,
        expected_run_kind="formal",
    )

    assert type(result) is _static_class()
    assert result.plan["run_kind"] == "formal"
    assert isinstance(result.config, ProjectConfig)
    assert isinstance(result.scenario_inputs, ScenarioInputs)
    assert isinstance(result.geometries, BoundaryGeometryBundle)
    assert calls["config"] == {"path": EXAMPLE_CONFIG, "formal_release": True}
    assert calls["scenario"]["config"] is data["config"]
    assert calls["scenario"]["kwargs"] == {
        "config_path": EXAMPLE_CONFIG,
        "require_dynamic_initial_conditions": True,
        "formal": True,
    }
    assert calls["geometry"] == {
        "config": data["config"],
        "kwargs": {"config_path": EXAMPLE_CONFIG, "formal": True},
    }
    assert gate_calls == [
        {
            "path": root / "input_inventory.json",
            "expected_config_hash": data["config"].config_hash(),
            "config": data["config"],
            "scenario_inputs": data["inputs"],
        }
    ]


@pytest.mark.parametrize(
    ("template_name", "expected_kind"),
    [("pilot", "formal"), ("formal", "pilot")],
)
def test_static_helper_requires_exact_expected_run_kind(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    template_name: str,
    expected_kind: str,
) -> None:
    """pilot/formal wrapper mismatch 必須拒絕，不能把 caller 的模式當第二個真相來源。"""

    template = request.getfixturevalue(f"{template_name}_template")
    data = template["data"]
    root = _copy_workspace(template, tmp_path)
    _patch_static_loaders(monkeypatch, data)

    with pytest.raises((TypeError, ValueError)) as error_info:
        _static_loader()(
            root,
            config_path=EXAMPLE_CONFIG,
            expected_run_kind=expected_kind,
        )
    _assert_no_path_leak(error_info.value, tmp_path)


@pytest.mark.parametrize("bad_value", [None, 0, 1, "false", []])
def test_static_helper_requires_native_bool_require_complete(
    pilot_template: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_value: Any,
) -> None:
    """require_complete 只接受原生 bool，避免字串或整數靜默改變完整性 gate。"""

    data = pilot_template["data"]
    root = _copy_workspace(pilot_template, tmp_path)
    _patch_static_loaders(monkeypatch, data)

    with pytest.raises((TypeError, ValueError)) as error_info:
        _static_loader()(
            root,
            config_path=EXAMPLE_CONFIG,
            require_complete=bad_value,
            expected_run_kind="pilot",
        )
    _assert_no_path_leak(error_info.value, tmp_path)


def test_static_helper_planned_run_accepts_false_but_rejects_true_require_complete(
    pilot_template: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PLANNED workspace 在 default/static binding 可讀，但 require_complete 必須拒絕。"""

    data = pilot_template["data"]
    root = _copy_workspace(pilot_template, tmp_path)
    _patch_static_loaders(monkeypatch, data)
    validation_calls: list[bool] = []
    real_validate_run = runtime.validate_run

    def recording_validate_run(
        path: str | Path,
        *,
        require_complete: bool,
        checkpoint_root: str | Path | None = None,
    ) -> dict[str, Any]:
        """記錄 helper 轉交 validator 的完整性需求，再執行真實 read-only validator。"""

        validation_calls.append(require_complete)
        return real_validate_run(
            path,
            require_complete=require_complete,
            checkpoint_root=checkpoint_root,
        )

    monkeypatch.setattr(runtime, "validate_run", recording_validate_run)
    accepted = _static_loader()(
        root,
        config_path=EXAMPLE_CONFIG,
        require_complete=False,
        expected_run_kind="pilot",
    )
    assert type(accepted) is _static_class()
    assert validation_calls == [False]

    with pytest.raises((TypeError, ValueError)) as error_info:
        _static_loader()(
            root,
            config_path=EXAMPLE_CONFIG,
            require_complete=True,
            expected_run_kind="pilot",
        )
    _assert_no_path_leak(error_info.value, tmp_path)
    assert validation_calls == [False, True]


def test_static_helper_rejects_synthetic_run_and_unknown_expected_kind(
    pilot_template: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """synthetic 不得進入 physical runtime；未知 expected kind 也不得 fallback。"""

    data = pilot_template["data"]
    synthetic_root = _make_synthetic_workspace(data, tmp_path / "synthetic-runs")
    calls = {"config": 0}

    def blocked_config(*_: Any, **__: Any) -> ProjectConfig:
        """若 synthetic 在 mode gate 前載入 config，立即暴露 static 邊界失效。"""

        calls["config"] += 1
        raise AssertionError("synthetic run 不得先載入 runtime config")

    monkeypatch.setattr(runtime, "load_config", blocked_config)
    with pytest.raises((TypeError, ValueError)) as error_info:
        _static_loader()(
            synthetic_root.path,
            config_path=EXAMPLE_CONFIG,
        )
    _assert_no_path_leak(error_info.value, tmp_path)
    assert calls == {"config": 0}

    pilot_root = _copy_workspace(pilot_template, tmp_path / "pilot")
    with pytest.raises((TypeError, ValueError)) as error_info:
        _static_loader()(
            pilot_root,
            config_path=EXAMPLE_CONFIG,
            expected_run_kind="synthetic",
        )
    _assert_no_path_leak(error_info.value, tmp_path)


def test_open_run_controller_calls_static_helper_once_before_factory_and_forwards_runtime_state(
    pilot_template: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """open controller 應先完成一次 static binding，再建立 factory 並原樣轉送 controller 參數。"""

    data = pilot_template["data"]
    root = _copy_workspace(pilot_template, tmp_path)
    _patch_static_loaders(monkeypatch, data)
    real_static_loader = _static_loader()
    checkpoint_root = tmp_path / "external-checkpoints"
    checkpoint_root.mkdir()
    static_calls: list[dict[str, Any]] = []
    static_results: list[Any] = []
    factory_calls: list[Any] = []
    controller_calls: list[dict[str, Any]] = []
    events: list[str] = []

    def recording_static_loader(
        workspace: str | Path | RunWorkspace,
        *,
        config_path: str | Path,
        checkpoint_root: str | Path | None = None,
        require_complete: bool = False,
        expected_run_kind: str | None = None,
    ) -> Any:
        """記錄 static helper 參數，並證明 factory 尚未建立便開始 binding。"""

        static_calls.append(
            {
                "workspace": workspace,
                "config_path": config_path,
                "checkpoint_root": checkpoint_root,
                "require_complete": require_complete,
                "expected_run_kind": expected_run_kind,
            }
        )
        assert factory_calls == []
        events.append("static-start")
        result = real_static_loader(
            workspace,
            config_path=config_path,
            checkpoint_root=checkpoint_root,
            require_complete=require_complete,
            expected_run_kind=expected_run_kind,
        )
        events.append("static-end")
        static_results.append(result)
        return result

    class RecordingFactory:
        """不觸碰 forcing 的 factory double，只驗證 static result 後的建構參數。"""

        def __init__(self, **kwargs: Any) -> None:
            """保存 runtime 傳入的 static records 與 forcing sentinel，不轉換任何 path。"""

            assert events[-1] == "static-end"
            events.append("factory")
            self.kwargs = kwargs
            factory_calls.append(self)

        def resource_stats(self) -> dict[str, int]:
            """提供 controller 所需的無副作用資源 callback。"""

            return {
                "manager_count": 0,
                "loads": 0,
                "hits": 0,
                "misses": 0,
                "evictions": 0,
                "resident_bytes": 0,
            }

    class RecordingController:
        """保存 open 對 RunController 的既有 resume/checkpoint/resource reporter 轉送。"""

        def __init__(
            self,
            workspace: str | Path | RunWorkspace,
            *,
            request_factory: Any,
            resume: bool,
            checkpoint_root: str | Path | None,
            resource_reporter: Any,
        ) -> None:
            """只保存參數，不執行 shard、reconcile 或 forcing request。"""

            assert events[-1] == "factory"
            events.append("controller")
            controller_calls.append(
                {
                    "workspace": workspace,
                    "request_factory": request_factory,
                    "resume": resume,
                    "checkpoint_root": checkpoint_root,
                    "resource_reporter": resource_reporter,
                }
            )

    def blocked_manager(**_: Any) -> None:
        """open 流程若提前建立 forcing manager，立即失敗。"""

        raise AssertionError("open controller 不得在 static helper 前建立 forcing manager")

    monkeypatch.setattr(runtime, "load_validated_run_static_inputs", recording_static_loader)
    monkeypatch.setattr(runtime, "RuntimeRequestFactory", RecordingFactory)
    monkeypatch.setattr(runtime, "RunController", RecordingController)
    monkeypatch.setattr(runtime.ForcingWindowManager, "from_roots", staticmethod(blocked_manager))
    ocm_sentinel = _UninspectablePath()
    nww_sentinel = _UninspectablePath()

    controller = runtime.open_pilot_run_controller(
        root,
        config_path=EXAMPLE_CONFIG,
        ocm_native_root=ocm_sentinel,
        nww_analysis_root=nww_sentinel,
        resume=True,
        checkpoint_root=checkpoint_root,
    )

    assert isinstance(controller, RecordingController)
    assert static_calls == [
        {
            "workspace": root,
            "config_path": EXAMPLE_CONFIG,
            "checkpoint_root": checkpoint_root,
            "require_complete": False,
            "expected_run_kind": "pilot",
        }
    ]
    assert len(factory_calls) == 1
    assert len(static_results) == 1
    factory = factory_calls[0]
    static_result = static_results[0]
    assert set(factory.kwargs) == {
        "config",
        "scenario_inputs",
        "geometries",
        "ocm_native_root",
        "nww_analysis_root",
        "experiment_case_id",
        "run_kind",
    }
    assert factory.kwargs["config"] is static_result.config
    assert factory.kwargs["scenario_inputs"] is static_result.scenario_inputs
    assert factory.kwargs["geometries"] is static_result.geometries
    assert factory.kwargs["ocm_native_root"] is ocm_sentinel
    assert factory.kwargs["nww_analysis_root"] is nww_sentinel
    assert factory.kwargs["experiment_case_id"] == "no_stokes"
    assert factory.kwargs["run_kind"] == "pilot"
    assert isinstance(factory.kwargs["scenario_inputs"], ScenarioInputs)
    assert controller_calls[0]["workspace"] == root
    assert controller_calls[0]["request_factory"] is factory
    assert controller_calls[0]["resume"] is True
    assert controller_calls[0]["checkpoint_root"] == checkpoint_root
    reporter = controller_calls[0]["resource_reporter"]
    assert getattr(reporter, "__self__", None) is factory
    assert getattr(reporter, "__func__", None) is RecordingFactory.resource_stats
    assert events == ["static-start", "static-end", "factory", "controller"]


def test_static_and_open_failures_do_not_expose_workspace_absolute_path(
    pilot_template: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """新增 static/open 失敗斷言固定檢查不洩漏 tmp absolute path，不改既有 API 錯誤文字。"""

    data = pilot_template["data"]
    _patch_static_loaders(monkeypatch, data)
    missing_root = tmp_path / "missing-static-workspace"

    with pytest.raises((TypeError, ValueError)) as static_error:
        _static_loader()(missing_root, config_path=EXAMPLE_CONFIG, expected_run_kind="pilot")
    _assert_no_path_leak(static_error.value, tmp_path)

    with pytest.raises((TypeError, ValueError)) as open_error:
        runtime.open_pilot_run_controller(
            missing_root,
            config_path=EXAMPLE_CONFIG,
            ocm_native_root=tmp_path / "ocm-native",
        )
    _assert_no_path_leak(open_error.value, tmp_path)
