"""aggregate CLI 的 synthetic 整合測試。

本檔只驗證 ``lbt`` 整合入口如何串接 AggregateSpec 建立、aggregate payload／release
建立與 release 驗證。測試沿用 ``test_aggregate_pipeline`` 的小型 pilot fixture，並由
既有 reference ``RunController`` 完成一個兩成員 shard；不建立 OCM schema 3、NWW3
schema 1 或任何真實 forcing array。所有通過結果只代表 parser、路徑安全、lock、
source immutable binding、release checksum 與 JSON 錯誤界面的工程契約，不是真實
OCM／NWW 科學成果、來源足跡、絕對來源機率或觀測驗證。

CLI 輸出的成功摘要不保存 run、config、spec、checkpoint 或 destination 的絕對路徑。
正式 SERVER 執行仍須由已驗收 OCM／NWW 產品提供動態輸入；本檔的 local synthetic
success 不能取代正式資料驗收。
"""

from __future__ import annotations

import json
import os
import shutil
import stat
from collections.abc import Callable
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest
import test_aggregate_pipeline as aggregate_pipeline_fixture
from test_runtime import EXAMPLE_CONFIG

import lagrangian_backtracking as lbt
import lagrangian_backtracking.cli as cli
import lagrangian_backtracking.runtime as runtime
from lagrangian_backtracking.aggregate_release import (
    validate_aggregate_release,
    write_aggregate_release,
)
from lagrangian_backtracking.aggregate_spec import load_aggregate_spec
from lagrangian_backtracking.run_control import load_run_plan
from lagrangian_backtracking.run_locking import RunLockBusyError, acquire_run_lock

_SPEC_CREATE_FAILURE = "aggregate spec 建立失敗"
_BUILD_FAILURE = "aggregate build 失敗"
_DURABILITY_FAILURE = "aggregate release 已發布但 parent durability 未確認"
_SUMMARY_KEYS = {
    "schema_version",
    "run_id",
    "run_kind",
    "experiment_case_id",
    "members_per_scenario",
    "input_particle_count",
    "shard_row_count",
    "scenario_row_count",
    "site_row_count",
    "boundary_row_count",
    "source_receptor_row_count",
    "payload_file_count",
}


@pytest.fixture(scope="module")
def cli_aggregate_template(
    tmp_path_factory: pytest.TempPathFactory,
) -> aggregate_pipeline_fixture._AggregatePipelineFixture:
    """建立一份可供全部 CLI 案例重用的 completed synthetic pilot。

    fixture 只包含一個 scenario 與兩個 ensemble members，幾何是 200 m 公尺制方形、
    底邊是合法 open-water LineString；其 purpose 是固定 CLI 的格線與檔案 topology，
    不是實際海岸線。workspace 由既有 initializer 加上 ``_request`` reference runner
    發布，故不 fake CLI handler，也不碰 OCM/NWW forcing。
    """

    data, site_id = aggregate_pipeline_fixture._aggregate_runtime_data()
    base = tmp_path_factory.mktemp("cli-aggregate")
    workspace = aggregate_pipeline_fixture._initialize_run(
        data,
        base,
        run_id="cli-aggregate-pilot",
    )
    spec_path, spec, centers = aggregate_pipeline_fixture._write_test_spec(
        data,
        base / "aggregate_spec.json",
        run_id="cli-aggregate-pilot",
    )
    return aggregate_pipeline_fixture._AggregatePipelineFixture(
        data=data,
        workspace=workspace,
        spec_path=spec_path,
        spec=spec,
        site_id=site_id,
        centers=centers,
        plan=load_run_plan(workspace),
    )


def _spec_create_args(
    source_root: Path,
    destination: Path,
    *,
    config_path: Path = EXAMPLE_CONFIG,
) -> list[str]:
    """建立所有研究數值都明示的 aggregate-spec-create 命令列。

    公尺制格網／邊界 bin／KDE bandwidth、秒制 age 邊界與 bootstrap 參數都由 caller
    明示，測試不依賴 CLI 預設值。這些數值只服務 synthetic contract，不代表正式 OCM/NWW
    分析設定或科學結論。
    """

    return [
        "aggregate-spec-create",
        "--run",
        str(source_root),
        "--config",
        str(config_path),
        "--destination",
        str(destination),
        "--grid-cell-size-m",
        "100",
        "--boundary-bin-size-m",
        "100",
        "--kde-bandwidths-m",
        "100",
        "200",
        "300",
        "--age-bin-edges-seconds",
        "0",
        "10",
        "--bootstrap-replicates",
        "10",
        "--bootstrap-confidence-level",
        "0.95",
        "--bootstrap-seed",
        "123",
    ]


def _replace_option(args: list[str], option: str, values: list[str]) -> list[str]:
    """替換一個 option 的所有 token，供 parser 邊界案例保持命令列可讀。"""

    index = args.index(option)
    end = index + 1
    while end < len(args) and not args[end].startswith("--"):
        end += 1
    return [*args[: index + 1], *values, *args[end:]]


def _without_option(args: list[str], option: str) -> list[str]:
    """移除一個必填 option 及其值，驗證 argparse 會在任何 I/O 前回傳狀態 2。"""

    return _replace_option(args, option, [])[:]


def _build_args(
    source_root: Path,
    spec_path: Path,
    *,
    destination: Path | None = None,
    checkpoint_root: Path | None = None,
) -> list[str]:
    """建立 aggregate-build keyword/path 參數；不替 CLI 猜測任何路徑。"""

    args = [
        "aggregate-build",
        "--run",
        str(source_root),
        "--config",
        str(EXAMPLE_CONFIG),
        "--spec",
        str(spec_path),
    ]
    if destination is not None:
        args.extend(("--destination", str(destination)))
    if checkpoint_root is not None:
        args.extend(("--checkpoint-root", str(checkpoint_root)))
    return args


def _copy_case(
    fixture: aggregate_pipeline_fixture._AggregatePipelineFixture,
    tmp_path: Path,
) -> aggregate_pipeline_fixture._AggregatePipelineFixture:
    """複製 source/spec 成單一 CLI 案例，避免 writer 或 tamper 污染 module fixture。"""

    source_parent = tmp_path / "runs"
    source_parent.mkdir()
    source_root = source_parent / fixture.workspace.name
    shutil.copytree(fixture.workspace, source_root, symlinks=True)
    spec_path = tmp_path / "aggregate_spec.json"
    spec_path.write_bytes(fixture.spec_path.read_bytes())
    return replace(
        fixture,
        workspace=source_root,
        spec_path=spec_path,
    )


def _block_forcing(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """封鎖 static CLI handler 意外建立 request factory 或 forcing manager。"""

    calls = {"factory": 0, "manager": 0}

    def blocked_factory(*args: object, **kwargs: object) -> None:
        """若 spec static binding 觸碰 factory，立即揭露責任邊界回歸。"""

        del args, kwargs
        calls["factory"] += 1
        raise AssertionError("aggregate CLI static binding 不得建立 RuntimeRequestFactory")

    def blocked_manager(*args: object, **kwargs: object) -> None:
        """若 static binding 觸碰 OCM/NWW manager，立即揭露 eager load 回歸。"""

        del args, kwargs
        calls["manager"] += 1
        raise AssertionError("aggregate CLI static binding 不得建立 ForcingWindowManager")

    monkeypatch.setattr(runtime, "RuntimeRequestFactory", blocked_factory)
    monkeypatch.setattr(
        runtime.ForcingWindowManager,
        "from_roots",
        staticmethod(blocked_manager),
    )
    return calls


def _assert_fixed_value_error(
    operation: Callable[[], object],
    *,
    message: str,
    path_markers: tuple[Path, ...],
) -> None:
    """確認 CLI 一般失敗的固定訊息、無 cause 且不洩漏暫存／SERVER 路徑。"""

    with pytest.raises(ValueError) as error_info:
        operation()
    error = error_info.value
    assert type(error) is ValueError
    assert str(error) == message
    assert error.__cause__ is None
    for marker in path_markers:
        assert str(marker) not in str(error)
        assert str(marker) not in repr(error)


def _assert_no_path_in_json(value: object, markers: tuple[Path, ...]) -> None:
    """遞迴檢查 CLI JSON-safe 結果沒有 caller 的絕對路徑文字。"""

    encoded = repr(value)
    for marker in markers:
        assert str(marker) not in encoded


def _spec_semantic_dict(spec: object) -> dict[str, object]:
    """移除由原始 JSON bytes 導出的兩個 digest，只比較 AggregateSpec 的設定語意。"""

    document = dict(spec.to_dict())  # type: ignore[union-attr]
    document.pop("source_sha256", None)
    document.pop("canonical_sha256", None)
    return document


def test_main_registers_aggregate_commands_and_numeric_arguments_are_required(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """整合 ``lbt`` 必須登錄三個 aggregate 子命令，研究數值不可偷偷使用預設值。"""

    with pytest.raises(SystemExit) as error_info:
        cli.main(["--help"])
    assert error_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "aggregate-spec-create" in help_text
    assert "aggregate-build" in help_text
    assert "aggregate-validate" in help_text

    parser = cli._aggregate_spec_create_parser()
    numeric_destinations = {
        "grid_cell_size_m",
        "boundary_bin_size_m",
        "kde_bandwidths_m",
        "age_bin_edges_seconds",
        "bootstrap_replicates",
        "bootstrap_confidence_level",
        "bootstrap_seed",
    }
    numeric_actions = {
        action.dest: action
        for action in parser._actions
        if action.dest in numeric_destinations
    }
    assert set(numeric_actions) == numeric_destinations
    assert all(action.required for action in numeric_actions.values())


@pytest.mark.parametrize(
    "invalid_case",
    (
        "missing-required",
        "single-age-edge",
        "zero-grid",
        "negative-boundary-bin",
        "negative-kde-bandwidth",
        "nan-grid",
        "inf-age",
        "confidence-zero",
        "confidence-one",
        "negative-seed",
    ),
)
def test_aggregate_spec_parser_rejects_invalid_research_values_before_writing(
    tmp_path: Path,
    invalid_case: str,
) -> None:
    """缺參數、單一 age edge、零／負值與非有限數值都應由 argparse 回傳 2。"""

    destination = tmp_path / "spec-output" / "aggregate_spec.json"
    destination.parent.mkdir()
    args = _spec_create_args(tmp_path / "not-a-run", destination)
    if invalid_case == "missing-required":
        args = _without_option(args, "--grid-cell-size-m")
    elif invalid_case == "single-age-edge":
        args = _replace_option(args, "--age-bin-edges-seconds", ["0"])
    elif invalid_case == "zero-grid":
        args = _replace_option(args, "--grid-cell-size-m", ["0"])
    elif invalid_case == "negative-boundary-bin":
        args = _replace_option(args, "--boundary-bin-size-m", ["-1"])
    elif invalid_case == "negative-kde-bandwidth":
        args = _replace_option(args, "--kde-bandwidths-m", ["100", "-1", "300"])
    elif invalid_case == "nan-grid":
        args = _replace_option(args, "--grid-cell-size-m", ["nan"])
    elif invalid_case == "inf-age":
        args = _replace_option(args, "--age-bin-edges-seconds", ["0", "inf"])
    elif invalid_case == "confidence-zero":
        args = _replace_option(args, "--bootstrap-confidence-level", ["0"])
    elif invalid_case == "confidence-one":
        args = _replace_option(args, "--bootstrap-confidence-level", ["1"])
    else:
        args = _replace_option(args, "--bootstrap-seed", ["-1"])

    with pytest.raises(SystemExit) as error_info:
        cli.main(args)
    assert error_info.value.code == 2
    assert not destination.exists()


def test_aggregate_spec_create_success_has_exact_json_and_static_read_only_binding(
    cli_aggregate_template: aggregate_pipeline_fixture._AggregatePipelineFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """spec-create 應產生 exact digest JSON、只讀 source 並且 static loader 僅呼叫一次。"""

    fixture = cli_aggregate_template
    destination = tmp_path / "outside-spec" / "aggregate_spec.json"
    destination.parent.mkdir()
    source_before = aggregate_pipeline_fixture._snapshot_tree(fixture.workspace)
    runtime_static_calls = aggregate_pipeline_fixture.runtime_static_fixture._patch_static_loaders(
        monkeypatch,
        fixture.data,
    )
    static_call_count = 0
    real_static_loader = cli.load_validated_run_static_inputs

    def counted_static_loader(*args: object, **kwargs: object) -> object:
        """記錄 CLI 對 shared static helper 的呼叫次數，仍委派 production helper。"""

        nonlocal static_call_count
        static_call_count += 1
        return real_static_loader(*args, **kwargs)

    forcing_calls = _block_forcing(monkeypatch)
    monkeypatch.setattr(cli, "load_validated_run_static_inputs", counted_static_loader)

    assert cli.main(_spec_create_args(fixture.workspace, destination)) == 0
    output = json.loads(capsys.readouterr().out)
    loaded_spec = load_aggregate_spec(destination)
    expected = {
        "aggregate_spec_name": destination.name,
        "schema_version": loaded_spec.schema_version,
        "run_id": loaded_spec.run_id,
        "site_count": len(loaded_spec.site_grids),
        "source_sha256": loaded_spec.source_sha256,
        "canonical_sha256": loaded_spec.canonical_sha256,
    }
    assert output == expected
    assert set(output) == {
        "aggregate_spec_name",
        "schema_version",
        "run_id",
        "site_count",
        "source_sha256",
        "canonical_sha256",
    }
    # CLI parser 將數值 canonical 成 float，而 module fixture 直接傳入 Python int；兩份
    # raw JSON 因此可能有 100 與 100.0 的格式差異，source/canonical digest 不應被拿來
    # 取代設定語意比較。load spec 的 typed fields 必須 exact 相同，digest 則各自對應
    # 自己的實際 bytes。
    assert _spec_semantic_dict(loaded_spec) == _spec_semantic_dict(fixture.spec)
    assert loaded_spec.source_sha256 == sha256(destination.read_bytes()).hexdigest()
    assert static_call_count == 1
    assert runtime_static_calls["config"]["formal_release"] is False
    assert forcing_calls == {"factory": 0, "manager": 0}
    assert aggregate_pipeline_fixture._snapshot_tree(fixture.workspace) == source_before
    assert str(tmp_path) not in repr(output)


@pytest.mark.parametrize(
    "destination_case",
    ("source-root", "source-child", "symlink-parent", "missing-parent"),
)
def test_aggregate_spec_create_rejects_unsafe_destination_without_writing(
    cli_aggregate_template: aggregate_pipeline_fixture._AggregatePipelineFixture,
    tmp_path: Path,
    destination_case: str,
) -> None:
    """spec 不得寫入 source root／子層、symlink parent 或不存在的 parent。"""

    fixture = cli_aggregate_template
    local_fixture = _copy_case(fixture, tmp_path)
    if destination_case == "source-root":
        destination = local_fixture.workspace / "aggregate_spec.json"
    elif destination_case == "source-child":
        child = local_fixture.workspace / "spec-child"
        child.mkdir()
        destination = child / "aggregate_spec.json"
    elif destination_case == "symlink-parent":
        target = tmp_path / "outside-target"
        target.mkdir()
        parent = tmp_path / "symlink-parent"
        os.symlink(target, parent)
        destination = parent / "aggregate_spec.json"
    else:
        destination = tmp_path / "missing-parent" / "aggregate_spec.json"

    source_before = aggregate_pipeline_fixture._snapshot_tree(local_fixture.workspace)
    _assert_fixed_value_error(
        lambda: cli.main(_spec_create_args(local_fixture.workspace, destination)),
        message=_SPEC_CREATE_FAILURE,
        path_markers=(tmp_path,),
    )
    assert not destination.exists()
    assert aggregate_pipeline_fixture._snapshot_tree(local_fixture.workspace) == source_before


def test_aggregate_spec_create_preserves_outer_run_lock_busy_error(
    cli_aggregate_template: aggregate_pipeline_fixture._AggregatePipelineFixture,
    tmp_path: Path,
) -> None:
    """spec-create 被外層 exclusive gate 擋住時，必須保留 RunLockBusyError。"""

    fixture = cli_aggregate_template
    destination = tmp_path / "outside-spec" / "aggregate_spec.json"
    destination.parent.mkdir()
    lock_path = fixture.workspace / "locks" / "run_gate.lock"
    with acquire_run_lock(lock_path, mode="exclusive", blocking=False), pytest.raises(
        RunLockBusyError
    ):
        cli.main(_spec_create_args(fixture.workspace, destination))
    assert not destination.exists()


def test_aggregate_build_success_writes_release_and_returns_portable_summary(
    cli_aggregate_template: aggregate_pipeline_fixture._AggregatePipelineFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """aggregate-build 應真實串接 payload、writer、validator，且輸出只含 portable 摘要。"""

    fixture = cli_aggregate_template
    source_before = aggregate_pipeline_fixture._snapshot_tree(fixture.workspace)
    aggregate_pipeline_fixture.runtime_static_fixture._patch_static_loaders(
        monkeypatch,
        fixture.data,
    )
    _block_forcing(monkeypatch)
    assert cli.main(_build_args(fixture.workspace, fixture.spec_path)) == 0
    result = json.loads(capsys.readouterr().out)
    final_path = fixture.workspace.parent / f"{fixture.workspace.name}.aggregate-v1"
    validator_report = validate_aggregate_release(final_path)
    assert result == {
        "release_name": final_path.name,
        "valid": True,
        "summary": validator_report["summary"],
    }
    assert set(result) == {"release_name", "valid", "summary"}
    assert result["valid"] is True
    assert set(result["summary"]) == _SUMMARY_KEYS
    assert len(result["summary"]) == 12
    assert validator_report["valid"] is True
    entries = list(final_path.iterdir())
    assert len(entries) == 33
    assert all(stat.S_ISREG(entry.lstat().st_mode) for entry in entries)
    assert all(not stat.S_ISLNK(entry.lstat().st_mode) for entry in entries)
    assert aggregate_pipeline_fixture._snapshot_tree(fixture.workspace) == source_before
    for marker in (fixture.workspace, EXAMPLE_CONFIG, fixture.spec_path, tmp_path):
        assert str(marker) not in repr(result)


def test_aggregate_build_rejects_malformed_and_mismatched_spec_without_path_leak(
    cli_aggregate_template: aggregate_pipeline_fixture._AggregatePipelineFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """malformed 或 run_id 不符的 spec 應使用固定 aggregate build 失敗訊息。"""

    fixture = cli_aggregate_template
    source_before = aggregate_pipeline_fixture._snapshot_tree(fixture.workspace)
    aggregate_pipeline_fixture.runtime_static_fixture._patch_static_loaders(
        monkeypatch,
        fixture.data,
    )
    malformed = tmp_path / "malformed-spec.json"
    malformed.write_bytes(b"{}\n")
    _assert_fixed_value_error(
        lambda: cli.main(_build_args(fixture.workspace, malformed)),
        message=_BUILD_FAILURE,
        path_markers=(tmp_path, fixture.workspace, EXAMPLE_CONFIG),
    )

    _, mismatched_spec, _ = aggregate_pipeline_fixture._write_test_spec(
        fixture.data,
        tmp_path / "mismatched-spec.json",
        run_id="another-cli-pilot",
    )
    del mismatched_spec
    mismatched_path = tmp_path / "mismatched-spec.json"
    _assert_fixed_value_error(
        lambda: cli.main(_build_args(fixture.workspace, mismatched_path)),
        message=_BUILD_FAILURE,
        path_markers=(tmp_path, fixture.workspace, EXAMPLE_CONFIG),
    )
    assert aggregate_pipeline_fixture._snapshot_tree(fixture.workspace) == source_before


def test_aggregate_build_preserves_lock_busy_error(
    cli_aggregate_template: aggregate_pipeline_fixture._AggregatePipelineFixture,
    tmp_path: Path,
) -> None:
    """aggregate-build 取得不到 source run gate 時，必須保留 contention 型別。"""

    fixture = _copy_case(cli_aggregate_template, tmp_path)
    lock_path = fixture.workspace / "locks" / "run_gate.lock"
    with acquire_run_lock(lock_path, mode="exclusive", blocking=False), pytest.raises(
        RunLockBusyError
    ):
        cli.main(_build_args(fixture.workspace, fixture.spec_path))
    assert not (fixture.workspace.parent / f"{fixture.workspace.name}.aggregate-v1").exists()


def test_aggregate_build_preserves_durability_runtimeerror_but_collapses_other_runtimeerror(
    cli_aggregate_template: aggregate_pipeline_fixture._AggregatePipelineFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """writer 的固定 durability uncertainty 原樣傳出，其他 RuntimeError 收斂為 build error。"""

    fixture = cli_aggregate_template
    aggregate_pipeline_fixture.runtime_static_fixture._patch_static_loaders(
        monkeypatch,
        fixture.data,
    )

    def fake_durability_writer(**kwargs: object) -> None:
        """模擬已 rename 但 parent fsync 未確認；不建立任何檔案。"""

        del kwargs
        raise RuntimeError(_DURABILITY_FAILURE)

    monkeypatch.setattr(cli, "write_aggregate_release", fake_durability_writer)
    with pytest.raises(RuntimeError) as durability_error:
        cli.main(_build_args(fixture.workspace, fixture.spec_path))
    assert type(durability_error.value) is RuntimeError
    assert str(durability_error.value) == _DURABILITY_FAILURE

    def fake_other_runtime_error(**kwargs: object) -> None:
        """模擬 writer 其他非 durability RuntimeError。"""

        del kwargs
        raise RuntimeError("synthetic writer failure")

    monkeypatch.setattr(cli, "write_aggregate_release", fake_other_runtime_error)
    _assert_fixed_value_error(
        lambda: cli.main(_build_args(fixture.workspace, fixture.spec_path)),
        message=_BUILD_FAILURE,
        path_markers=(fixture.workspace,),
    )


def test_aggregate_validate_success_prints_pretty_sorted_report(
    cli_aggregate_template: aggregate_pipeline_fixture._AggregatePipelineFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """aggregate-validate 合法 release 應回傳 0 與 validator exact pretty sorted JSON。"""

    local_fixture = _copy_case(cli_aggregate_template, tmp_path)
    payload = aggregate_pipeline_fixture._build_payload(local_fixture, monkeypatch)
    final_path = write_aggregate_release(
        source_run_root=local_fixture.workspace,
        aggregate_spec_path=local_fixture.spec_path,
        payload=payload,
    )
    expected = validate_aggregate_release(final_path)
    assert cli.main(["aggregate-validate", str(final_path)]) == 0
    raw_output = capsys.readouterr().out
    assert raw_output == json.dumps(expected, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    output = json.loads(raw_output)
    assert output == expected
    assert set(output) == {"valid", "errors", "summary"}
    assert output["valid"] is True
    assert set(output["summary"]) == _SUMMARY_KEYS
    _assert_no_path_in_json(output, (tmp_path, local_fixture.workspace, local_fixture.spec_path))


def test_aggregate_validate_missing_and_tampered_release_return_two_without_path(
    cli_aggregate_template: aggregate_pipeline_fixture._AggregatePipelineFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """missing 或 checksum tamper release 應輸出固定 JSON-safe report 並回傳 2。"""

    local_fixture = _copy_case(cli_aggregate_template, tmp_path)
    payload = aggregate_pipeline_fixture._build_payload(local_fixture, monkeypatch)
    final_path = write_aggregate_release(
        source_run_root=local_fixture.workspace,
        aggregate_spec_path=local_fixture.spec_path,
        payload=payload,
    )

    missing = tmp_path / "missing.aggregate-v1"
    assert cli.main(["aggregate-validate", str(missing)]) == 2
    missing_output = json.loads(capsys.readouterr().out)
    assert set(missing_output) == {"valid", "errors", "summary"}
    assert missing_output["valid"] is False
    _assert_no_path_in_json(missing_output, (tmp_path, missing))

    tampered = tmp_path / "tampered.aggregate-v1"
    shutil.copytree(final_path, tampered)
    product = tampered / "age_bin_edges_seconds.npy"
    product.write_bytes(product.read_bytes() + b"tamper")
    assert cli.main(["aggregate-validate", str(tampered)]) == 2
    tampered_output = json.loads(capsys.readouterr().out)
    assert set(tampered_output) == {"valid", "errors", "summary"}
    assert tampered_output["valid"] is False
    _assert_no_path_in_json(tampered_output, (tmp_path, tampered))


def test_package_root_exports_all_planned_aggregate_symbols_without_private_geometry_helper() -> None:
    """package root 應提供規劃的十二個 aggregate API，且不外露 private center helper。"""

    expected_exports = {
        "AggregateSpec",
        "AggregateReleasePayload",
        "ValidatedRunStaticInputs",
        "AGGREGATE_RELEASE_SCHEMA_VERSION",
        "build_aggregate_release_payload",
        "load_aggregate_spec",
        "load_validated_run_static_inputs",
        "read_aggregate_release",
        "validate_aggregate_release",
        "validate_aggregate_spec_against_boundaries",
        "write_aggregate_release",
        "write_aggregate_spec_from_boundaries",
    }
    assert expected_exports <= set(lbt.__all__)
    assert all(hasattr(lbt, name) for name in expected_exports)
    assert not hasattr(lbt, "_build_site_metric_centers")
