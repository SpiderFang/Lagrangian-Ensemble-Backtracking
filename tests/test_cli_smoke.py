"""整合 CLI 的 constant-flow shard 垂直切片測試。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from test_pilot_matrix_validation import _write_run as _write_pilot_matrix_run
from test_run_control import _first_shard, _request, _workspace

import lagrangian_backtracking.cli as cli
from lagrangian_backtracking.cli import main
from lagrangian_backtracking.config import load_config
from lagrangian_backtracking.manifests import load_material_manifest
from lagrangian_backtracking.run_control import RunController, RunExecutionSummary

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = ROOT / "configs" / "lagrangian_backtracking.example.yaml"


@dataclass(frozen=True, slots=True)
class _FakeWorkspace:
    """只提供 CLI 顯示所需的 workspace 路徑，避免測試建立真實 run 目錄。"""

    path: Path
    plan: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class _FakeInputs:
    """保存 run-shard 解析 forcing root 所需的環境變數名稱。"""

    ocm_native_root_env: str
    nww_analysis_root_env: str


@dataclass(frozen=True, slots=True)
class _FakeConfig:
    """只模擬 CLI 會讀取的 ``config.inputs``，不承擔真實設定驗證責任。"""

    inputs: _FakeInputs


@dataclass(slots=True)
class _FakeShardController:
    """記錄單一 shard 執行呼叫，隔離 CLI 測試與實際 forcing 或 checkpoint I/O。"""

    summary: RunExecutionSummary
    calls: list[tuple[str, int | None]] = field(default_factory=list)

    def run_shard(self, shard_id: str, *, sweep_budget: int | None = None) -> RunExecutionSummary:
        """保存 CLI 傳入的 shard 與 sweep 預算，並回傳預先定義的真實摘要型別。"""

        self.calls.append((shard_id, sweep_budget))
        return self.summary


def _block_reconcile_runtime_calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """封鎖 reconcile 不應觸發的設定、runtime controller 與 initializer 副作用。"""

    calls = {
        "load_config": 0,
        "open_run_controller": 0,
        "initialize_pilot_run": 0,
    }

    def blocked_load_config(*args: object, **kwargs: object) -> object:
        """記錄任何設定載入嘗試，因 reconcile 應只處理 run-control 狀態。"""

        del args, kwargs
        calls["load_config"] += 1
        raise AssertionError("run-reconcile 不得呼叫 load_config")

    def blocked_open_run_controller(*args: object, **kwargs: object) -> object:
        """記錄任何 runtime factory 開啟嘗試，避免 reconcile 進入 forcing 執行路徑。"""

        del args, kwargs
        calls["open_run_controller"] += 1
        raise AssertionError("run-reconcile 不得呼叫 open_run_controller")

    def blocked_initialize_pilot_run(*args: object, **kwargs: object) -> object:
        """記錄任何 workspace initializer 嘗試，避免 reconcile 重建 run。"""

        del args, kwargs
        calls["initialize_pilot_run"] += 1
        raise AssertionError("run-reconcile 不得呼叫 initialize_pilot_run")

    monkeypatch.setattr(cli, "load_config", blocked_load_config)
    monkeypatch.setattr(cli, "open_run_controller", blocked_open_run_controller)
    monkeypatch.setattr(cli, "initialize_pilot_run", blocked_initialize_pilot_run)
    return calls


def test_integrated_synthetic_smoke_and_validator(tmp_path: Path) -> None:
    """同一 shard 應可由 synthetic-smoke 產生，再由 validate-shard 獨立驗證。"""

    output = tmp_path / "synthetic-shard"
    assert main(["synthetic-smoke", "--output", str(output)]) == 0
    assert main(["validate-shard", str(output)]) == 0
    events = pq.read_table(output / "events.parquet", columns=["event_type"]).column(0).to_pylist()
    assert events == [
        "local_domain_first_exit",
        "other_site_local_domain_enter",
        "other_site_local_domain_exit",
        "flow_domain_open_exit",
    ]


def test_behavior_manifest_has_ten_records(tmp_path: Path) -> None:
    """CLI 產生的十類代理須具名、可追溯、版本同步且能被 config loader 接受。"""

    output = tmp_path / "behaviors.json"
    assert main(["behavior-manifest", "--output", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "2.0.0"
    config = load_config(EXAMPLE_CONFIG, formal_release=False)
    assert payload["design_version"] == config.design_version
    assert payload["positive_or_zero_velocity_policy"] == "reject_config"
    assert len(payload["records"]) == 10
    assert len({item["oca_category_zh"] for item in payload["records"]}) == 10
    assert all(item["settling_velocity_mps"] < 0 for item in payload["records"])
    assert {item["behavior_class"] for item in payload["records"]} == {"sinking"}
    assert all(item["material_family_zh"] for item in payload["records"])
    assert all(item["representative_shape_zh"] for item in payload["records"])
    assert len(load_material_manifest(output, config, formal=False)) == 10


def test_pilot_matrix_validate_cli_outputs_canonical_json_and_exit_codes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """整合 CLI 應比較至少兩個 root，輸出 canonical JSON 並以 0／2 回報。"""

    first = _write_pilot_matrix_run(tmp_path / "matrix-a")
    second = _write_pilot_matrix_run(tmp_path / "matrix-b")
    assert main(["pilot-matrix-validate", str(first), str(second)]) == 0
    rendered = capsys.readouterr().out
    assert rendered == rendered.strip() + "\n"
    payload = json.loads(rendered)
    assert payload["valid"] is True
    assert payload["run_count"] == 2
    assert payload["errors"] == []
    assert main(["pilot-matrix-validate", str(first)]) == 2
    assert json.loads(capsys.readouterr().out)["valid"] is False


def test_inputs_build_cli_always_forwards_strict_and_preserves_formal_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """inputs-build CLI 不論 pilot 或 formal 都必須關閉 synthetic fallback。

    測試以 fake builder 隔離實際 accepted-product I/O，只驗證 CLI 的資料契約：
    ``strict`` 永遠是 ``True``，而 ``--formal-release`` 仍只改變傳入 builder 的
    ``formal`` 旗標。這保留了內部 Python API 以 ``strict=False`` 建立小型 fixture 的
    能力，同時避免命令列 production pipeline 意外產生合成輸入。
    """

    calls: list[dict[str, object]] = []

    class _FakeResult:
        """提供 CLI 輸出所需的最小結果介面，不建立任何 input artifact。"""

        def to_dict(self) -> dict[str, str]:
            """回傳可序列化摘要，讓測試專注於參數轉送而非檔案發布。"""

            return {"config_status": "generated"}

    def fake_build_input_derivatives(**kwargs: object) -> _FakeResult:
        """記錄 builder 關鍵字參數，隔離測試與真實 OCM／NWW3 產品讀取。"""

        calls.append(kwargs)
        return _FakeResult()

    monkeypatch.setattr(cli, "build_input_derivatives", fake_build_input_derivatives)
    common_args = [
        "--config",
        str(tmp_path / "config.yaml"),
        "--destination",
        str(tmp_path / "input-release"),
        "--ocm-native-root",
        str(tmp_path / "ocm-native"),
        "--ocm-surface-root",
        str(tmp_path / "ocm-surface"),
        "--nww-analysis-root",
        str(tmp_path / "nww-analysis"),
    ]

    assert main(["inputs-build", *common_args]) == 0
    capsys.readouterr()
    assert main(["inputs-build", *common_args, "--formal-release"]) == 0
    capsys.readouterr()
    assert (
        main(
            [
                "inputs-build",
                *common_args,
                "--pilot-arrival-utc",
                "hsinchu=2024-01-02T01:00:00Z",
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert [call["strict"] for call in calls] == [True, True, True]
    assert [call["formal"] for call in calls] == [False, True, False]
    assert calls[2]["pilot_arrival_utc"] == {"hsinchu": "2024-01-02T01:00:00Z"}


def test_inputs_build_cli_rejects_formal_pilot_before_builder_or_destination_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """formal CLI 搭配 pilot 選項應在 builder 前拒絕，並保持 destination 不存在。"""

    def unexpected_builder(**kwargs: object) -> object:
        """若 CLI 錯誤地進入 builder，立即使測試失敗。"""

        del kwargs
        raise AssertionError("formal pilot 不得呼叫 input builder")

    monkeypatch.setattr(cli, "build_input_derivatives", unexpected_builder)
    destination = tmp_path / "formal-pilot-inputs"
    with pytest.raises(ValueError, match="不得搭配 --pilot-arrival-utc"):
        main(
            [
                "inputs-build",
                "--config",
                str(tmp_path / "config.yaml"),
                "--destination",
                str(destination),
                "--formal-release",
                "--pilot-arrival-utc",
                "hsinchu=2024-01-02T01:00:00Z",
            ]
        )
    assert not destination.exists()


def test_horizon_suite_create_cli_forwards_all_arguments_and_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """horizon-suite-create 應保留日數順序、轉送三個 roots，並輸出核心 JSON。"""

    calls: list[dict[str, object]] = []

    def fake_build_horizon_suite(**kwargs: object) -> dict[str, object]:
        """記錄 suite builder 的完整參數，隔離測試與實際 accepted-product I/O。"""

        calls.append(kwargs)
        return {
            "destination": str(kwargs["destination"]),
            "formal": kwargs["formal"],
            "horizons_days": list(kwargs["backtrack_days"]),
            "valid": True,
        }

    monkeypatch.setattr(cli, "build_horizon_suite", fake_build_horizon_suite)
    template = tmp_path / "template.yaml"
    destination = tmp_path / "suite-formal"
    ocm_native_root = tmp_path / "ocm-native"
    ocm_surface_root = tmp_path / "ocm-surface"
    nww_analysis_root = tmp_path / "nww-analysis"
    common = [
        "horizon-suite-create",
        "--config-template",
        str(template),
        "--backtrack-days",
        "90",
        "30",
        "60",
        "--destination",
        str(destination),
        "--ocm-native-root",
        str(ocm_native_root),
        "--ocm-surface-root",
        str(ocm_surface_root),
        "--nww-analysis-root",
        str(nww_analysis_root),
    ]

    assert main([*common, "--formal"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "destination": str(destination),
        "formal": True,
        "horizons_days": [90, 30, 60],
        "valid": True,
    }
    assert calls == [
        {
            "config_template_path": template,
            "backtrack_days": [90, 30, 60],
            "destination": destination,
            "ocm_native_root": ocm_native_root,
            "ocm_surface_root": ocm_surface_root,
            "nww_analysis_root": nww_analysis_root,
            "formal": True,
        }
    ]

    pilot_destination = tmp_path / "suite-pilot"
    assert (
        main(
            [
                *common[:7],
                "--destination",
                str(pilot_destination),
                *common[9:],
                "--pilot",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == {
        "destination": str(pilot_destination),
        "formal": False,
        "horizons_days": [90, 30, 60],
        "valid": True,
    }
    assert calls[1]["destination"] == pilot_destination
    assert calls[1]["formal"] is False


def test_horizon_suite_validate_cli_forwards_suite_and_optional_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """horizon-suite-validate 應固定只傳 suite path，並以 valid 映射 0／2。"""

    calls: list[tuple[Path, dict[str, object]]] = []
    reports = iter(
        (
            {"valid": True, "errors": [], "warnings": []},
            {"valid": False, "errors": ["tampered"], "warnings": []},
        )
    )

    def fake_validate_horizon_suite(path: Path, **kwargs: object) -> dict[str, object]:
        """記錄 validator 呼叫，確保沒有外部 common-input override 被轉送。"""

        calls.append((path, kwargs))
        return next(reports)

    monkeypatch.setattr(cli, "validate_horizon_suite", fake_validate_horizon_suite)
    suite = tmp_path / "suite"
    ocm_native_root = tmp_path / "ocm-native"
    ocm_surface_root = tmp_path / "ocm-surface"
    nww_analysis_root = tmp_path / "nww-analysis"
    roots = [
        "--ocm-native-root",
        str(ocm_native_root),
        "--ocm-surface-root",
        str(ocm_surface_root),
        "--nww-analysis-root",
        str(nww_analysis_root),
    ]

    assert main(["horizon-suite-validate", str(suite), *roots, "--formal-release"]) == 0
    assert json.loads(capsys.readouterr().out) == {"errors": [], "valid": True, "warnings": []}
    assert calls[0] == (
        suite,
        {
            "formal": True,
            "ocm_native_root": ocm_native_root,
            "ocm_surface_root": ocm_surface_root,
            "nww_analysis_root": nww_analysis_root,
        },
    )

    assert main(["horizon-suite-validate", str(suite), "--pilot"]) == 2
    assert json.loads(capsys.readouterr().out) == {
        "errors": ["tampered"],
        "valid": False,
        "warnings": [],
    }
    assert calls[1] == (
        suite,
        {
            "formal": False,
            "ocm_native_root": None,
            "ocm_surface_root": None,
            "nww_analysis_root": None,
        },
    )


def test_horizon_suite_cli_registers_commands_and_rejects_conflicting_modes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """整合 help 應列出兩個新命令，formal 與 pilot 也必須互斥。"""

    with pytest.raises(SystemExit) as help_exit:
        main(["--help"])
    assert help_exit.value.code == 0
    help_text = capsys.readouterr().out
    assert "horizon-suite-create" in help_text
    assert "horizon-suite-validate" in help_text

    create_parser = cli._horizon_suite_create_parser()
    with pytest.raises(SystemExit) as conflict_exit:
        create_parser.parse_args(["--formal", "--pilot"])
    assert conflict_exit.value.code == 2

    validate_parser = cli._horizon_suite_validate_parser()
    with pytest.raises(SystemExit) as input_override_exit:
        validate_parser.parse_args(["suite", "--input-directory", "outside-common-input"])
    assert input_override_exit.value.code == 2


def test_run_validation_cli_exit_codes_and_external_checkpoint_root(tmp_path: Path) -> None:
    """整合 CLI 必須把 external root 傳入 validator/report，並以 0/2 回報結果。"""

    workspace = _workspace(tmp_path / "workspace", "cli-external")
    checkpoint_root = tmp_path / "external-checkpoints"
    shard_id = _first_shard(workspace)
    RunController(
        workspace,
        request_factory=_request,
        checkpoint_root=checkpoint_root,
    ).run_shard(shard_id, sweep_budget=1)
    assert (
        main(
            [
                "validate-run",
                str(workspace),
                "--checkpoint-root",
                str(checkpoint_root),
            ]
        )
        == 0
    )
    assert main(["validate-run", str(workspace)]) == 2
    assert (
        main(
            [
                "benchmark-report",
                str(workspace),
                "--checkpoint-root",
                str(checkpoint_root),
            ]
        )
        == 0
    )
    assert main(["validate-run", str(tmp_path / "missing")]) == 2


def test_run_create_pilot_integrated_parser_forwards_all_initializer_kwargs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """pilot run-create 應由整合 parser 解析，並在唯讀驗證成功後輸出固定摘要。

    initializer 的七個關鍵字參數與 validator 的 ``require_complete=False`` 是 CLI
    與 runtime 之間的資料契約；fake 只記錄呼叫，不建立目錄或讀取輸入檔，故本測試
    的副作用邊界限於 parser、參數轉送與 stdout JSON。所有路徑均由本測試獨立建立，
    不依賴其他測試先前留下的 workspace。
    """

    config_path = tmp_path / "config.yaml"
    inventory_path = tmp_path / "inventory.json"
    destination = tmp_path / "runs"
    project_root = tmp_path / "project"
    workspace = _FakeWorkspace(tmp_path / "created-pilot")
    initializer_calls: list[dict[str, object]] = []
    validator_calls: list[tuple[object, bool]] = []

    def fake_initialize_pilot_run(**kwargs: object) -> _FakeWorkspace:
        """記錄建立器的完整輸入，避免測試觸發真實 manifest 與檔案發布。"""

        initializer_calls.append(kwargs)
        return workspace

    def fake_validate_run(path: object, *, require_complete: bool) -> dict[str, object]:
        """記錄建立後驗證的完成度要求，並回傳可序列化的成功摘要。"""

        validator_calls.append((path, require_complete))
        return {
            "valid": True,
            "summary": {"scenario_count": 7, "particle_count": 14, "shard_count": 3},
        }

    monkeypatch.setattr(cli, "initialize_pilot_run", fake_initialize_pilot_run)
    monkeypatch.setattr(cli, "validate_run", fake_validate_run)

    assert (
        main(
            [
                "run-create",
                "--config",
                str(config_path),
                "--input-inventory",
                str(inventory_path),
                "--destination",
                str(destination),
                "--run-id",
                "pilot-cli-test",
                "--run-kind",
                "pilot",
                "--experiment-case",
                "finite_depth_stokes",
                "--project-root",
                str(project_root),
                "--declared-git-commit",
                "abc123",
            ]
        )
        == 0
    )

    assert initializer_calls == [
        {
            "config_path": config_path,
            "input_inventory_path": inventory_path,
            "destination": destination,
            "run_id": "pilot-cli-test",
            "experiment_case_id": "finite_depth_stokes",
            "project_root": project_root,
            "declared_git_commit": "abc123",
        }
    ]
    assert validator_calls == [(workspace, False)]
    assert json.loads(capsys.readouterr().out) == {
        "workspace": str(workspace.path),
        "run_id": "pilot-cli-test",
        "run_kind": "pilot",
        "experiment_case_id": "finite_depth_stokes",
        "scenario_count": 7,
        "particle_count": 14,
        "shard_count": 3,
        "valid": True,
    }


def test_run_create_pilot_forwards_selector_and_reports_selection_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """pilot N 參數只傳給 pilot initializer，成功 JSON 僅回報 mode/count。"""

    workspace = _FakeWorkspace(
        tmp_path / "created-selected-pilot",
        plan={
            "scenario_selection": {
                "mode": "pilot_stratified",
                "source_scenario_count": 50_000,
                "selected_scenario_count": 20,
            }
        },
    )
    initializer_calls: list[dict[str, object]] = []

    def fake_initialize_pilot_run(**kwargs: object) -> _FakeWorkspace:
        """記錄 selector 參數，隔離測試與真實 manifest I/O。"""

        initializer_calls.append(kwargs)
        return workspace

    monkeypatch.setattr(cli, "initialize_pilot_run", fake_initialize_pilot_run)
    monkeypatch.setattr(
        cli,
        "validate_run",
        lambda _path, *, require_complete: {
            "valid": True,
            "summary": {"scenario_count": 20, "particle_count": 20, "shard_count": 1},
        },
    )
    assert (
        main(
            [
                "run-create",
                "--config",
                str(tmp_path / "config.yaml"),
                "--input-inventory",
                str(tmp_path / "inventory.json"),
                "--destination",
                str(tmp_path / "runs"),
                "--run-id",
                "pilot-selected",
                "--run-kind",
                "pilot",
                "--experiment-case",
                "no_stokes",
                "--pilot-scenarios-per-stratum",
                "1",
            ]
        )
        == 0
    )
    assert initializer_calls[0]["pilot_scenarios_per_stratum"] == 1
    output = json.loads(capsys.readouterr().out)
    assert output["selection_mode"] == "pilot_stratified"
    assert output["source_scenario_count"] == 50_000
    assert output["selected_scenario_count"] == 20
    assert "scenario_ids" not in json.dumps(output)


def _exact_create_arguments(tmp_path: Path) -> list[str]:
    """建立僅供解析測試的明示三 ID 命令，不指向真實輸入或 SERVER。"""

    return [
        "run-create", "--config", str(tmp_path / "config.yaml"),
        "--input-inventory", str(tmp_path / "inventory.json"),
        "--destination", str(tmp_path / "runs"), "--run-id", "exact-pilot",
        "--run-kind", "pilot", "--experiment-case", "no_stokes",
        "--pilot-study-site-id", "hsinchu", "--pilot-arrival-id", "synthetic-arrival-id",
        "--pilot-material-id", "synthetic-material-id",
    ]


def test_run_create_forwards_exact_identifiers_and_keeps_scenario_and_member_counts_distinct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI 原樣傳遞識別碼；來源五萬、選中二十情境與八十粒子不得混為同一分母。"""

    calls = []

    def create(**kwargs):
        """只記錄參數，返回模擬計畫；本測試不執行來源讀取或粒子積分。"""
        calls.append(kwargs)
        return _FakeWorkspace(tmp_path / "exact-pilot", {"scenario_selection": {
            "mode": "pilot_exact", "source_scenario_count": 50_000, "selected_scenario_count": 20,
        }})

    monkeypatch.setattr(cli, "initialize_pilot_run", create)
    monkeypatch.setattr(cli, "validate_run", lambda *args, **kwargs: {
        "valid": True, "summary": {"scenario_count": 20, "particle_count": 80, "shard_count": 2},
    })
    assert main(_exact_create_arguments(tmp_path)) == 0
    assert calls[0]["pilot_study_site_id"] == "hsinchu"
    assert calls[0]["pilot_arrival_id"] == "synthetic-arrival-id"
    assert calls[0]["pilot_material_id"] == "synthetic-material-id"
    assert "pilot_scenarios_per_stratum" not in calls[0]
    result = json.loads(capsys.readouterr().out)
    assert result["selection_mode"] == "pilot_exact"
    assert result["source_scenario_count"] == 50_000
    assert result["scenario_count"] == 20
    assert result["particle_count"] == 80


@pytest.mark.parametrize(
    "case", ("missing", "duplicate", "empty", "whitespace", "mixed", "formal", "synthetic"),
)
def test_run_create_rejects_malformed_exact_requests_before_initializer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str,
) -> None:
    """三 ID 缺項、重複、空白、混用分層或非 pilot 都須在建立器之前拒絕。"""

    def blocked(**kwargs):
        """非法 CLI 請求不應觸發來源讀取或目錄發布。"""
        raise AssertionError("invalid request reached initializer")

    monkeypatch.setattr(cli, "initialize_pilot_run", blocked)
    monkeypatch.setattr(cli, "initialize_formal_run", blocked)
    args = _exact_create_arguments(tmp_path)
    if case == "missing":
        args = args[:-2]
    elif case == "duplicate":
        args += ["--pilot-arrival-id", "another-id"]
    elif case in {"empty", "whitespace"}:
        args[-1] = "" if case == "empty" else " material-id "
    elif case == "mixed":
        args += ["--pilot-scenarios-per-stratum", "1"]
    else:
        args[args.index("--run-kind") + 1] = case
    with pytest.raises((ValueError, SystemExit)):
        main(args)
    assert not (tmp_path / "runs").exists()


def test_run_create_formal_rejects_pilot_selector_before_initializer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """formal 搭配 pilot selector 必須在 initializer 前 fail closed。"""

    def blocked_initializer(**_: object) -> None:
        """若 formal initializer 被呼叫，表示 CLI 順序錯誤。"""

        raise AssertionError("formal selector rejection must precede initializer")

    monkeypatch.setattr(cli, "initialize_formal_run", blocked_initializer)
    with pytest.raises(ValueError, match="formal run 禁止"):
        main(
            [
                "run-create",
                "--config",
                str(tmp_path / "config.yaml"),
                "--input-inventory",
                str(tmp_path / "inventory.json"),
                "--destination",
                str(tmp_path / "runs"),
                "--run-id",
                "formal-selected",
                "--run-kind",
                "formal",
                "--experiment-case",
                "no_stokes",
                "--pilot-scenarios-per-stratum",
                "1",
            ]
        )


def test_run_create_formal_dispatches_formal_initializer_and_validates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """formal run-create 應呼叫 formal initializer，不得降級成 pilot，並立即驗證。"""

    initializer_calls: list[dict[str, object]] = []
    validator_calls: list[tuple[object, bool]] = []
    workspace = _FakeWorkspace(tmp_path / "created-formal")

    def fake_initialize_formal_run(**kwargs: object) -> _FakeWorkspace:
        """記錄 formal initializer 的完整輸入，避免測試觸發真實 manifest I/O。"""

        initializer_calls.append(kwargs)
        return workspace

    def fake_validate_run(path: object, *, require_complete: bool) -> dict[str, object]:
        """回傳成功摘要，鎖定 formal 建立後的 validator 呼叫。"""

        validator_calls.append((path, require_complete))
        return {
            "valid": True,
            "summary": {"scenario_count": 50_000, "particle_count": 100_000, "shard_count": 4},
        }

    monkeypatch.setattr(cli, "initialize_formal_run", fake_initialize_formal_run)
    monkeypatch.setattr(cli, "validate_run", fake_validate_run)

    assert (
        main(
            [
                "run-create",
                "--config",
                str(tmp_path / "config.yaml"),
                "--input-inventory",
                str(tmp_path / "inventory.json"),
                "--destination",
                str(tmp_path / "runs"),
                "--run-id",
                "formal-cli-test",
                "--run-kind",
                "formal",
                "--experiment-case",
                "no_stokes",
            ]
        )
        == 0
    )
    assert initializer_calls == [
        {
            "config_path": tmp_path / "config.yaml",
            "input_inventory_path": tmp_path / "inventory.json",
            "destination": tmp_path / "runs",
            "run_id": "formal-cli-test",
            "experiment_case_id": "no_stokes",
            "project_root": Path("."),
            "declared_git_commit": None,
        }
    ]
    assert validator_calls == [(workspace, False)]
    assert json.loads(capsys.readouterr().out) == {
        "workspace": str(workspace.path),
        "run_id": "formal-cli-test",
        "run_kind": "formal",
        "experiment_case_id": "no_stokes",
        "scenario_count": 50_000,
        "particle_count": 100_000,
        "shard_count": 4,
        "valid": True,
    }


def test_run_create_invalid_post_validation_hides_absolute_path_from_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run-create 建立後驗證失敗時應保留錯誤碼，但不得把 workspace 絕對路徑洩漏到例外。

    validator fake 刻意回傳含絕對路徑的詳細錯誤，模擬檔案檢查需要的操作環境資訊；
    測試只允許 initializer 執行一次，並驗證 CLI 的錯誤轉換只取錯誤碼。這個案例不
    依賴真實目錄內容，故不會修改 tmp_path 以外的狀態。
    """

    absolute_workspace = (tmp_path / "created-pilot").resolve()
    initializer_calls: list[dict[str, object]] = []
    validator_calls: list[dict[str, object]] = []
    validation_result = {
        "valid": False,
        "errors": [f"workspace_invalid: {absolute_workspace} detail"],
    }

    def fake_initialize_pilot_run(**kwargs: object) -> Path:
        """記錄唯一一次建立呼叫，並回傳明示的絕對 workspace 路徑。"""

        initializer_calls.append(kwargs)
        return absolute_workspace

    def fake_validate_run(path: object, *, require_complete: bool) -> dict[str, object]:
        """回傳含路徑詳細資訊的 invalid 結果，以驗證例外訊息的脫敏邊界。"""

        validator_calls.append({"path": path, "require_complete": require_complete})
        return validation_result

    monkeypatch.setattr(cli, "initialize_pilot_run", fake_initialize_pilot_run)
    monkeypatch.setattr(cli, "validate_run", fake_validate_run)

    with pytest.raises(ValueError) as exc_info:
        main(
            [
                "run-create",
                "--config",
                str(tmp_path / "config.yaml"),
                "--input-inventory",
                str(tmp_path / "inventory.json"),
                "--destination",
                str(tmp_path / "runs"),
                "--run-id",
                "invalid-post-validation",
                "--run-kind",
                "pilot",
                "--experiment-case",
                "no_stokes",
            ]
        )

    assert len(initializer_calls) == 1
    assert validator_calls == [{"path": absolute_workspace, "require_complete": False}]
    assert str(absolute_workspace) in validation_result["errors"][0]
    assert "workspace_invalid" in str(exc_info.value)
    assert str(absolute_workspace) not in str(exc_info.value)


def test_run_shard_invalid_prevalidation_returns_validator_json_without_loading_or_opening(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """run-shard 在 pre-validation invalid 時應回傳 2 並原樣輸出 validator JSON。

    checkpoint root 必須在唯一一次 validator 呼叫中精確保留；設定載入與 controller
    開啟 fake 只計數，確保 invalid gate 前沒有讀取設定或建立執行狀態。測試使用獨立
    路徑與固定 mapping，不依賴真實 run 或測試順序。
    """

    workspace = tmp_path / "workspace"
    config_path = tmp_path / "config.yaml"
    checkpoint_root = tmp_path / "checkpoint-root"
    validation_result = {"valid": False, "errors": ["workspace_invalid", "missing_manifest"]}
    validation_calls: list[dict[str, object]] = []
    plan_calls: list[int] = []
    load_calls: list[int] = []
    open_calls: list[int] = []

    def fake_validate_run(
        path: Path, *, require_complete: bool, checkpoint_root: Path | None
    ) -> dict[str, object]:
        """記錄 pre-validation 的 workspace、完成度與 checkpoint root 後回傳 invalid。"""

        validation_calls.append(
            {"path": path, "require_complete": require_complete, "checkpoint_root": checkpoint_root}
        )
        return validation_result

    def fake_load_config(*args: object, **kwargs: object) -> _FakeConfig:
        """若 invalid gate 失效則留下設定載入紀錄，並不讀取任何檔案。"""

        del args, kwargs
        load_calls.append(1)
        return _FakeConfig(_FakeInputs("UNUSED_OCM_ENV", "UNUSED_NWW_ENV"))

    def blocked_load_run_plan(*args: object, **kwargs: object) -> dict[str, str]:
        """若 validator 失敗後仍讀 plan，留下可檢查的順序回歸訊號。"""

        del args, kwargs
        plan_calls.append(1)
        raise AssertionError("invalid workspace 前不得讀取 run plan")

    def fake_open_pilot_run_controller(*args: object, **kwargs: object) -> object:
        """若 invalid gate 失效則留下 controller 建立紀錄。"""

        del args, kwargs
        open_calls.append(1)
        return object()

    monkeypatch.setattr(cli, "validate_run", fake_validate_run)
    monkeypatch.setattr(cli, "load_run_plan", blocked_load_run_plan)
    monkeypatch.setattr(cli, "load_config", fake_load_config)
    monkeypatch.setattr(cli, "open_run_controller", fake_open_pilot_run_controller)

    assert (
        main(
            [
                "run-shard",
                str(workspace),
                "--config",
                str(config_path),
                "--shard-id",
                "shard-0",
                "--checkpoint-root",
                str(checkpoint_root),
            ]
        )
        == 2
    )

    assert validation_calls == [
        {"path": workspace, "require_complete": False, "checkpoint_root": checkpoint_root}
    ]
    assert plan_calls == []
    assert load_calls == []
    assert open_calls == []
    assert json.loads(capsys.readouterr().out) == validation_result


def test_run_shard_forwards_explicit_roots_resume_and_sweep_to_controller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """run-shard 應優先採用 CLI roots，並只執行一次 controller.run_shard。

    config.inputs 仍提供環境變數名稱，且環境值刻意與 CLI root 不同，以鎖定優先序；
    fake controller 回傳真正的 ``RunExecutionSummary``，因此 stdout 契約直接以
    ``asdict`` 比對。validator、loader、open 與 controller 都在本測試內記錄，沒有
    forcing、checkpoint 或其他跨測試狀態副作用。
    """

    workspace = tmp_path / "workspace"
    config_path = tmp_path / "config.yaml"
    explicit_ocm_root = tmp_path / "cli-ocm"
    explicit_nww_root = tmp_path / "cli-nww"
    checkpoint_root = tmp_path / "checkpoint-root"
    ocm_env_name = "LBT_TEST_B5_OCM_ROOT"
    nww_env_name = "LBT_TEST_B5_NWW_ROOT"
    environment_ocm_root = tmp_path / "environment-ocm"
    environment_nww_root = tmp_path / "environment-nww"
    monkeypatch.setenv(ocm_env_name, str(environment_ocm_root))
    monkeypatch.setenv(nww_env_name, str(environment_nww_root))
    config = _FakeConfig(_FakeInputs(ocm_env_name, nww_env_name))
    summary = RunExecutionSummary(
        run_id="run-shard-cli",
        shard_id="target-shard",
        lifecycle="PAUSED",
        scenario_count=2,
        particle_count=4,
        sweeps_completed=3,
        particle_steps=12,
        output_relative_path="outputs/target-shard.parquet",
        checkpoint_relative_path="checkpoints/target-shard/generation-000001",
    )
    controller = _FakeShardController(summary)
    validation_calls: list[dict[str, object]] = []
    load_calls: list[dict[str, object]] = []
    open_calls: list[tuple[Path, dict[str, object]]] = []

    def fake_validate_run(
        path: Path, *, require_complete: bool, checkpoint_root: Path | None
    ) -> dict[str, object]:
        """記錄先行 validator 的輸入，並讓測試進入 root 解析與 controller 階段。"""

        validation_calls.append(
            {"path": path, "require_complete": require_complete, "checkpoint_root": checkpoint_root}
        )
        return {"valid": True, "summary": {}}

    def fake_load_config(path: Path, *, formal_release: bool) -> _FakeConfig:
        """回傳僅含 forcing 環境變數名稱的最小設定，並記錄正式模式旗標。"""

        load_calls.append({"path": path, "formal_release": formal_release})
        return config

    def fake_open_pilot_run_controller(
        workspace_arg: Path, **kwargs: object
    ) -> _FakeShardController:
        """記錄 open controller 的全部關鍵字參數，但不建立 runtime resource。"""

        open_calls.append((workspace_arg, kwargs))
        return controller

    monkeypatch.setattr(cli, "validate_run", fake_validate_run)
    monkeypatch.setattr(cli, "load_run_plan", lambda path: {"run_kind": "pilot"})
    monkeypatch.setattr(cli, "load_config", fake_load_config)
    monkeypatch.setattr(cli, "open_run_controller", fake_open_pilot_run_controller)

    assert (
        main(
            [
                "run-shard",
                str(workspace),
                "--config",
                str(config_path),
                "--shard-id",
                "target-shard",
                "--ocm-native-root",
                str(explicit_ocm_root),
                "--nww-analysis-root",
                str(explicit_nww_root),
                "--checkpoint-root",
                str(checkpoint_root),
                "--resume",
                "--sweep-budget",
                "3",
            ]
        )
        == 0
    )

    assert validation_calls == [
        {"path": workspace, "require_complete": False, "checkpoint_root": checkpoint_root}
    ]
    assert load_calls == [{"path": config_path, "formal_release": False}]
    assert open_calls == [
        (
            workspace,
            {
                "config_path": config_path,
                "ocm_native_root": explicit_ocm_root,
                "nww_analysis_root": explicit_nww_root,
                "resume": True,
                "checkpoint_root": checkpoint_root,
            },
        )
    ]
    assert explicit_ocm_root != environment_ocm_root
    assert explicit_nww_root != environment_nww_root
    assert controller.calls == [("target-shard", 3)]
    assert json.loads(capsys.readouterr().out) == asdict(summary)


def test_run_shard_selects_formal_config_and_generic_controller_from_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """run-shard 通過 pre-validation 後應從 plan 選 formal config 與 generic controller。"""

    workspace = tmp_path / "formal-workspace"
    config_path = tmp_path / "formal-config.yaml"
    ocm_root = tmp_path / "formal-ocm"
    summary = RunExecutionSummary(
        run_id="formal-cli-run",
        shard_id="formal-shard",
        lifecycle="PAUSED",
        scenario_count=1,
        particle_count=2,
        sweeps_completed=1,
        particle_steps=2,
        output_relative_path=None,
        checkpoint_relative_path="checkpoints/formal-shard/generation-000001",
    )
    config = _FakeConfig(_FakeInputs("FORMAL_OCM_ENV", "FORMAL_NWW_ENV"))
    controller = _FakeShardController(summary)
    validation_calls: list[dict[str, object]] = []
    load_plan_calls: list[Path] = []
    load_config_calls: list[dict[str, object]] = []
    open_calls: list[tuple[Path, dict[str, object]]] = []

    def fake_validate_run(
        path: Path, *, require_complete: bool, checkpoint_root: Path | None
    ) -> dict[str, object]:
        """讓測試進入 plan/config/runtime dispatch，並記錄先行驗證。"""

        validation_calls.append(
            {"path": path, "require_complete": require_complete, "checkpoint_root": checkpoint_root}
        )
        return {"valid": True}

    def fake_load_run_plan(path: Path) -> dict[str, str]:
        """回傳 formal immutable plan snapshot。"""

        load_plan_calls.append(path)
        return {"run_kind": "formal"}

    def fake_load_config(path: Path, *, formal_release: bool) -> _FakeConfig:
        """記錄 formal_release=True，並回傳 root 環境變數契約。"""

        load_config_calls.append({"path": path, "formal_release": formal_release})
        return config

    def fake_open_run_controller(
        workspace_arg: Path, **kwargs: object
    ) -> _FakeShardController:
        """記錄 generic controller 的正式執行參數，不建立 forcing。"""

        open_calls.append((workspace_arg, kwargs))
        return controller

    monkeypatch.setattr(cli, "validate_run", fake_validate_run)
    monkeypatch.setattr(cli, "load_run_plan", fake_load_run_plan)
    monkeypatch.setattr(cli, "load_config", fake_load_config)
    monkeypatch.setattr(cli, "open_run_controller", fake_open_run_controller)

    assert (
        main(
            [
                "run-shard",
                str(workspace),
                "--config",
                str(config_path),
                "--shard-id",
                "formal-shard",
                "--ocm-native-root",
                str(ocm_root),
                "--sweep-budget",
                "2",
            ]
        )
        == 0
    )
    assert validation_calls == [
        {"path": workspace, "require_complete": False, "checkpoint_root": None}
    ]
    assert load_plan_calls == [workspace]
    assert load_config_calls == [{"path": config_path, "formal_release": True}]
    assert open_calls == [
        (
            workspace,
            {
                "config_path": config_path,
                "ocm_native_root": ocm_root,
                "nww_analysis_root": None,
                "resume": False,
                "checkpoint_root": None,
            },
        )
    ]
    assert controller.calls == [("formal-shard", 2)]
    assert json.loads(capsys.readouterr().out) == asdict(summary)


def test_run_shard_resolves_environment_roots_and_uses_non_formal_config_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """未提供 CLI roots 時，run-shard 應由 config.inputs 指定的環境變數解析兩個 Path。

    fake config 只暴露 root 環境變數名稱，fake loader 另行記錄 ``formal_release=False``；
    controller fake 收到解析後的 Path 即完成本案例。所有呼叫都由本測試建立與檢查，
    不讀取 YAML、forcing 產品或既有 run 狀態。
    """

    workspace = tmp_path / "workspace"
    config_path = tmp_path / "config.yaml"
    ocm_env_name = "LBT_TEST_B6_OCM_ROOT"
    nww_env_name = "LBT_TEST_B6_NWW_ROOT"
    ocm_root = tmp_path / "env-ocm"
    nww_root = tmp_path / "env-nww"
    monkeypatch.setenv(ocm_env_name, str(ocm_root))
    monkeypatch.setenv(nww_env_name, str(nww_root))
    config = _FakeConfig(_FakeInputs(ocm_env_name, nww_env_name))
    summary = RunExecutionSummary(
        run_id="env-roots",
        shard_id="env-shard",
        lifecycle="COMPLETE",
        scenario_count=1,
        particle_count=2,
        sweeps_completed=1,
        particle_steps=2,
        output_relative_path=None,
        checkpoint_relative_path=None,
    )
    controller = _FakeShardController(summary)
    load_calls: list[dict[str, object]] = []
    open_calls: list[tuple[Path, dict[str, object]]] = []

    def fake_validate_run(
        path: Path, *, require_complete: bool, checkpoint_root: Path | None
    ) -> dict[str, object]:
        """讓測試專注在通過 pre-validation 後的環境 root 解析。"""

        del path, require_complete, checkpoint_root
        return {"valid": True}

    def fake_load_config(path: Path, *, formal_release: bool) -> _FakeConfig:
        """記錄 loader 模式並回傳 root 環境變數契約。"""

        load_calls.append({"path": path, "formal_release": formal_release})
        return config

    def fake_open_pilot_run_controller(
        workspace_arg: Path, **kwargs: object
    ) -> _FakeShardController:
        """記錄由環境變數轉成的 Path，隔離實際 controller 開啟副作用。"""

        open_calls.append((workspace_arg, kwargs))
        return controller

    monkeypatch.setattr(cli, "validate_run", fake_validate_run)
    monkeypatch.setattr(cli, "load_run_plan", lambda path: {"run_kind": "pilot"})
    monkeypatch.setattr(cli, "load_config", fake_load_config)
    monkeypatch.setattr(cli, "open_run_controller", fake_open_pilot_run_controller)

    assert (
        main(
            [
                "run-shard",
                str(workspace),
                "--config",
                str(config_path),
                "--shard-id",
                "env-shard",
            ]
        )
        == 0
    )

    assert load_calls == [{"path": config_path, "formal_release": False}]
    assert open_calls == [
        (
            workspace,
            {
                "config_path": config_path,
                "ocm_native_root": ocm_root,
                "nww_analysis_root": nww_root,
                "resume": False,
                "checkpoint_root": None,
            },
        )
    ]
    assert controller.calls == [("env-shard", None)]


def test_run_shard_passes_none_when_optional_nww_environment_root_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NWW3 環境變數缺失且無 CLI root 時，run-shard 應把 ``None`` 傳給 controller。

    OCM root 仍明確提供，讓測試越過必要 root gate；NWW3 是可選值，故只驗證其
    ``None`` 語意而不觸發任何 forcing 讀取。每個 fake 均在本測試內建立，結果不依賴
    其他案例是否先設定過環境變數。
    """

    workspace = tmp_path / "workspace"
    config_path = tmp_path / "config.yaml"
    ocm_env_name = "LBT_TEST_B7_OCM_ROOT"
    nww_env_name = "LBT_TEST_B7_NWW_ROOT"
    ocm_root = tmp_path / "env-ocm"
    monkeypatch.setenv(ocm_env_name, str(ocm_root))
    monkeypatch.delenv(nww_env_name, raising=False)
    config = _FakeConfig(_FakeInputs(ocm_env_name, nww_env_name))
    summary = RunExecutionSummary(
        run_id="optional-nww",
        shard_id="nww-missing",
        lifecycle="PAUSED",
        scenario_count=1,
        particle_count=1,
        sweeps_completed=0,
        particle_steps=0,
        output_relative_path=None,
        checkpoint_relative_path="checkpoints/nww-missing",
    )
    controller = _FakeShardController(summary)
    open_calls: list[tuple[Path, dict[str, object]]] = []

    def fake_validate_run(
        path: Path, *, require_complete: bool, checkpoint_root: Path | None
    ) -> dict[str, object]:
        """讓 optional NWW root 案例只檢查通過 validator 後的解析行為。"""

        del path, require_complete, checkpoint_root
        return {"valid": True}

    def fake_load_config(path: Path, *, formal_release: bool) -> _FakeConfig:
        """回傳 OCM 必要與 NWW 可選的環境變數名稱。"""

        del path, formal_release
        return config

    def fake_open_pilot_run_controller(
        workspace_arg: Path, **kwargs: object
    ) -> _FakeShardController:
        """記錄可選 root 的實際傳值，避免建立真實 runtime controller。"""

        open_calls.append((workspace_arg, kwargs))
        return controller

    monkeypatch.setattr(cli, "validate_run", fake_validate_run)
    monkeypatch.setattr(cli, "load_run_plan", lambda path: {"run_kind": "pilot"})
    monkeypatch.setattr(cli, "load_config", fake_load_config)
    monkeypatch.setattr(cli, "open_run_controller", fake_open_pilot_run_controller)

    assert (
        main(
            [
                "run-shard",
                str(workspace),
                "--config",
                str(config_path),
                "--shard-id",
                "nww-missing",
            ]
        )
        == 0
    )

    assert open_calls == [
        (
            workspace,
            {
                "config_path": config_path,
                "ocm_native_root": ocm_root,
                "nww_analysis_root": None,
                "resume": False,
                "checkpoint_root": None,
            },
        )
    ]
    assert controller.calls == [("nww-missing", None)]


def test_run_shard_missing_ocm_root_raises_before_opening_controller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OCM CLI 與環境變數 root 都缺失時，run-shard 應 ValueError 且不建立 controller。

    fake validator 與 config loader 只讓流程抵達必要 root gate；open fake 的零呼叫是
    本測試的副作用邊界，確保缺少必要資料不會進入 runtime。環境變數由 monkeypatch
    明確刪除，故不受主機環境或測試順序影響。
    """

    workspace = tmp_path / "workspace"
    config_path = tmp_path / "config.yaml"
    ocm_env_name = "LBT_TEST_B8_OCM_ROOT"
    nww_env_name = "LBT_TEST_B8_NWW_ROOT"
    monkeypatch.delenv(ocm_env_name, raising=False)
    config = _FakeConfig(_FakeInputs(ocm_env_name, nww_env_name))
    open_calls: list[int] = []

    def fake_validate_run(
        path: Path, *, require_complete: bool, checkpoint_root: Path | None
    ) -> dict[str, object]:
        """讓測試通過 pre-validation，聚焦必要 OCM root 的 fail-closed 行為。"""

        del path, require_complete, checkpoint_root
        return {"valid": True}

    def fake_load_config(path: Path, *, formal_release: bool) -> _FakeConfig:
        """回傳缺少 OCM root 值的 inputs 契約，不讀取實際設定檔。"""

        del path, formal_release
        return config

    def fake_open_pilot_run_controller(*args: object, **kwargs: object) -> object:
        """若必要 root gate 失效則留下 controller 呼叫紀錄。"""

        del args, kwargs
        open_calls.append(1)
        return object()

    monkeypatch.setattr(cli, "validate_run", fake_validate_run)
    monkeypatch.setattr(cli, "load_run_plan", lambda path: {"run_kind": "pilot"})
    monkeypatch.setattr(cli, "load_config", fake_load_config)
    monkeypatch.setattr(cli, "open_run_controller", fake_open_pilot_run_controller)

    with pytest.raises(ValueError, match=ocm_env_name):
        main(
            [
                "run-shard",
                str(workspace),
                "--config",
                str(config_path),
                "--shard-id",
                "missing-ocm",
            ]
        )

    assert open_calls == []


@pytest.mark.parametrize("sweep_budget", [0, -1, True, False])
def test_run_shard_rejects_invalid_sweep_budget_in_argparse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sweep_budget: int | bool,
) -> None:
    """零、負數與 True／False sweep budget 應由 argparse 以狀態 2 拒絕。

    參數型別檢查發生在 run-shard handler 之前，因此 validator、config loader 與
    controller open 均必須維持零呼叫；這鎖定 CLI 的輸入邊界，而不是測試 controller
    內部的防禦性驗證。每個參數化案例都有獨立 monkeypatch 狀態，不依賴案例順序。
    """

    validator_calls: list[int] = []
    load_calls: list[int] = []
    open_calls: list[int] = []

    def blocked_validate_run(*args: object, **kwargs: object) -> dict[str, object]:
        """若 argparse 未先拒絕 budget，留下 validator 呼叫紀錄。"""

        del args, kwargs
        validator_calls.append(1)
        return {"valid": True}

    def blocked_load_config(*args: object, **kwargs: object) -> _FakeConfig:
        """若流程越過 parser gate，留下 config loader 呼叫紀錄。"""

        del args, kwargs
        load_calls.append(1)
        return _FakeConfig(_FakeInputs("UNUSED_OCM_ENV", "UNUSED_NWW_ENV"))

    def blocked_open_pilot_run_controller(*args: object, **kwargs: object) -> object:
        """若流程越過 parser gate，留下 controller open 呼叫紀錄。"""

        del args, kwargs
        open_calls.append(1)
        return object()

    monkeypatch.setattr(cli, "validate_run", blocked_validate_run)
    monkeypatch.setattr(cli, "load_config", blocked_load_config)
    monkeypatch.setattr(cli, "open_run_controller", blocked_open_pilot_run_controller)

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "run-shard",
                str(tmp_path / "workspace"),
                "--config",
                str(tmp_path / "config.yaml"),
                "--shard-id",
                "invalid-budget",
                "--sweep-budget",
                str(sweep_budget),
            ]
        )

    assert exc_info.value.code == 2
    assert validator_calls == []
    assert load_calls == []
    assert open_calls == []


def test_run_reconcile_planned_workspace_emits_exact_read_only_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """planned 真實 workspace 應可 reconcile，並輸出完整且不涉及 runtime 的狀態摘要。

    ``_workspace`` 建立的 plan/progress 提供真實 run-control 拓撲；三個 CLI runtime
    入口被封鎖，用來證明 reconcile 不讀 config、不開 forcing factory，也不重建 workspace。
    預期摘要以 workspace 自身的 plan shard_count 與初始 revision 組成，因此測試不依賴
    其他測試的狀態或執行順序。
    """

    workspace = _workspace(tmp_path / "planned", "cli-reconcile-planned")
    plan = json.loads((workspace / "run_plan.json").read_text(encoding="utf-8"))
    progress = json.loads((workspace / "run_progress.json").read_text(encoding="utf-8"))
    calls = _block_reconcile_runtime_calls(monkeypatch)

    assert main(["run-reconcile", str(workspace)]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "run_id": plan["run_id"],
        "run_lifecycle": "PLANNED",
        "revision": progress["revision"],
        "shard_lifecycle_counts": {"PLANNED": plan["shard_count"]},
        "valid": True,
        "errors": [],
    }
    assert calls == {
        "load_config": 0,
        "open_run_controller": 0,
        "initialize_pilot_run": 0,
    }


def test_run_reconcile_external_paused_workspace_preserves_progress_and_hides_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """external checkpoint 的真實 PAUSED shard 應只被 reconcile 採認，不得推進執行狀態。

    先以既有 synthetic request 建立一代 external checkpoint，再保存 progress bytes；
    reconcile 後比較 lifecycle、attempt、sweeps、particle steps 與 output，允許合法的
    metadata 修復但不允許重新執行粒子。validator wrapper 仍呼叫真實 validator 並核對
    external root，三個 runtime/config 入口則以零呼叫封鎖，故不會引入 forcing 副作用。
    """

    workspace = _workspace(tmp_path / "paused", "cli-reconcile-paused")
    shard_id = _first_shard(workspace)
    external = tmp_path / "external-checkpoints"
    paused = RunController(
        workspace,
        request_factory=_request,
        checkpoint_root=external,
    ).run_shard(shard_id, sweep_budget=1)
    assert paused.lifecycle == "PAUSED"
    assert paused.output_relative_path is None

    before_progress_bytes = (workspace / "run_progress.json").read_bytes()
    before_progress = json.loads(before_progress_bytes.decode("utf-8"))
    before_row = before_progress["shards"][shard_id]
    plan = json.loads((workspace / "run_plan.json").read_text(encoding="utf-8"))
    real_validate_run = cli.validate_run
    validation_calls: list[dict[str, object]] = []

    def recording_validate_run(
        path: Path, *, require_complete: bool, checkpoint_root: Path | None
    ) -> dict[str, object]:
        """記錄 CLI validator 參數後仍執行真實驗證，避免測試繞過 external root gate。"""

        validation_calls.append(
            {"path": path, "require_complete": require_complete, "checkpoint_root": checkpoint_root}
        )
        return real_validate_run(
            path,
            require_complete=require_complete,
            checkpoint_root=checkpoint_root,
        )

    calls = _block_reconcile_runtime_calls(monkeypatch)
    monkeypatch.setattr(cli, "validate_run", recording_validate_run)

    assert main(["run-reconcile", str(workspace), "--checkpoint-root", str(external)]) == 0

    stdout = capsys.readouterr().out
    payload = json.loads(stdout)
    assert payload == {
        "run_id": plan["run_id"],
        "run_lifecycle": "PAUSED",
        "revision": before_progress["revision"],
        "shard_lifecycle_counts": {"PAUSED": 1, "PLANNED": plan["shard_count"] - 1},
        "valid": True,
        "errors": [],
    }
    assert str(external) not in stdout
    assert validation_calls == [
        {"path": workspace, "require_complete": False, "checkpoint_root": external}
    ]

    after_progress = json.loads((workspace / "run_progress.json").read_text(encoding="utf-8"))
    after_row = after_progress["shards"][shard_id]
    for field_name in (
        "lifecycle",
        "attempt_count",
        "sweeps_completed",
        "particle_steps",
        "output_relative_path",
    ):
        assert after_row[field_name] == before_row[field_name]
    assert after_row["lifecycle"] == "PAUSED"
    assert after_row["output_relative_path"] is None
    assert calls == {
        "load_config": 0,
        "open_run_controller": 0,
        "initialize_pilot_run": 0,
    }
