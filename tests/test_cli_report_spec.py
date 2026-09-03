"""``report-spec-create`` 命令列的 bounded synthetic contract tests。

本檔只在 pytest 暫存目錄建立一份一站一邊界段的最小 ``AggregateSpec`` JSON，
再透過真正的 CLI handler、report spec writer、loader 與 aggregate binding 驗證
資料流。測試不建立 OCM schema 3、NWW3 schema 1、forcing array、軌跡或任何報告
圖表；所有數值、雜湊與路徑都是 synthetic engineering fixture。測試成功只表示
命令列與 immutable 檔案契約成立，不代表真實 OCM／NWW 科學成果、條件式來源足跡、
相對來源權重、絕對來源機率或觀測驗證已經成立。
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

import pytest

import lagrangian_backtracking as lbt
import lagrangian_backtracking.cli as cli
import lagrangian_backtracking.report_spec as report_spec_module
import lagrangian_backtracking.runtime as runtime
from lagrangian_backtracking.aggregate_spec import (
    AGGREGATE_SPEC_SCHEMA_VERSION,
    AggregateSpec,
    SiteBoundarySegments,
    SiteGridSpec,
    SiteMetricCRSSpec,
    load_aggregate_spec,
)
from lagrangian_backtracking.report_spec import (
    REPORT_SPEC_SCHEMA_VERSION,
    load_report_spec,
    validate_report_spec_against_aggregate_spec,
)

_HASH = "a" * 64
_OTHER_HASH = "b" * 64
_REPORT_CREATE_FAILURE = "report spec 建立失敗"
_DURABILITY_FAILURE = "report spec durability confirmation failed"

# 這組 key 是測試對 report spec input JSON 的獨立描述。兩個 provenance hash
# 不在 input root，因為它們必須由 writer／loader 對實際 bytes 推導，不能由 CLI
# caller 代填。垂向邊界是 positive-down 公尺（m），後續 reducer 才負責 edge 外
# observation 的 overflow；本 CLI 測試不介入該統計責任。
_REPORT_INPUT_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "aggregate_spec_canonical_sha256",
        "primary_kde_bandwidth_m",
        "minimum_kde_raw_count",
        "low_sample_min_member_count",
        "vertical_depth_bin_edges_m",
        "representative_trajectory_count_per_site",
        "representative_selection_policy",
        "representative_selection_seed",
        "travel_age_quantiles",
        "pathway_first_passage_quantiles",
        "figure_formats",
        "raster_dpi",
        "renderer_style_version",
        "language",
    }
)


@pytest.fixture(scope="module")
def synthetic_aggregate_spec() -> AggregateSpec:
    """建立一站一 segment 的最小 AggregateSpec，完全不依賴上游科學產品。

    x/y 範圍是 0 至 1000 m、單元是 100 m；一條 1000 m 的 synthetic boundary
    同時掛在 local 與 outer 角色。三個 KDE 帶寬為 100／200／300 m，age 軸為
    0／10 s。兩個 hash 只用來滿足 typed fixture 的格式，不能被解讀為任何真實
    OCM／NWW provenance。
    """

    return AggregateSpec(
        schema_version=AGGREGATE_SPEC_SCHEMA_VERSION,
        run_id="synthetic-report-cli-run",
        grid_cell_size_m=100,
        site_grids={
            "synthetic-site": SiteGridSpec(
                x_min_m=0,
                x_max_m=1000,
                y_min_m=0,
                y_max_m=1000,
            )
        },
        site_metric_crs={
            "synthetic-site": SiteMetricCRSSpec(
                projection_method="azimuthal_equidistant_wgs84",
                center_lon_deg=121.0,
                center_lat_deg=24.0,
                linear_unit="m",
                axis_order="x_east_y_north",
            )
        },
        boundary_bin_size_m=100,
        boundary_segment_lengths_m={"synthetic-segment": 1000},
        site_boundary_segment_ids={
            "synthetic-site": SiteBoundarySegments(
                local_segment_ids=("synthetic-segment",),
                outer_segment_ids=("synthetic-segment",),
            )
        },
        kde_bandwidths_m=(100, 200, 300),
        hdr_levels=(0.5, 0.75, 0.9),
        age_bin_edges_seconds=(0, 10),
        bootstrap_replicates=10,
        bootstrap_confidence_level=0.95,
        bootstrap_seed=0,
        denominator_policy="exclude_data_gap_and_numerical_failure_v1",
        source_sha256=_HASH,
        canonical_sha256=_OTHER_HASH,
    )


def _aggregate_input_document(spec: AggregateSpec) -> dict[str, object]:
    """將 typed fixture 轉成不含衍生 hash 的 aggregate input JSON document。"""

    document = spec.to_dict()
    del document["source_sha256"]
    del document["canonical_sha256"]
    return document


def _write_aggregate_input(path: Path, spec: AggregateSpec) -> AggregateSpec:
    """只在 pytest tmp_path 寫入 compact aggregate JSON，並回讀實際 hash snapshot。"""

    document = _aggregate_input_document(spec)
    raw = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    path.write_bytes(raw)
    return load_aggregate_spec(path)


def _replace_option(args: list[str], option: str, values: list[str]) -> list[str]:
    """替換一個 option 的全部 token，支援 vertical 多值參數。"""

    index = args.index(option)
    end = index + 1
    while end < len(args) and not args[end].startswith("--"):
        end += 1
    return [*args[: index + 1], *values, *args[end:]]


def _without_option(args: list[str], option: str) -> list[str]:
    """移除一個必填 option 及其所有值，驗證 parser 尚未進入檔案 I/O。"""

    index = args.index(option)
    end = index + 1
    while end < len(args) and not args[end].startswith("--"):
        end += 1
    return [*args[:index], *args[end:]]


def _report_spec_args(
    aggregate_path: Path,
    destination: Path,
) -> list[str]:
    """建立所有研究數值均明示的 report-spec-create 命令列。

    KDE 帶寬與深度邊界使用公尺（m），代表軌跡數是 4 season × 2 spring/neap
    strata 的一次串流取樣總額；這些只是 synthetic contract 數值，不是正式研究
    設定或科學結果。
    """

    return [
        "report-spec-create",
        "--aggregate-spec",
        str(aggregate_path),
        "--destination",
        str(destination),
        "--primary-kde-bandwidth-m",
        "100",
        "--minimum-kde-raw-count",
        "1",
        "--low-sample-min-member-count",
        "1",
        "--vertical-depth-bin-edges-m",
        "0",
        "5",
        "10",
        "--representative-trajectory-count-per-site",
        "8",
        "--representative-selection-seed",
        "0",
    ]


def _assert_fixed_value_error(
    operation: Callable[[], object],
    *,
    path_markers: tuple[Path, ...],
) -> None:
    """確認 handler 一般失敗固定為無 cause 且不洩漏暫存／SERVER 路徑。"""

    with pytest.raises(ValueError) as error_info:
        operation()
    error = error_info.value
    assert type(error) is ValueError
    assert str(error) == _REPORT_CREATE_FAILURE
    assert error.__cause__ is None
    assert all(str(marker) not in str(error) for marker in path_markers)
    assert all(str(marker) not in repr(error) for marker in path_markers)


def _assert_no_partial(destination: Path) -> None:
    """確認 writer 命名規則下沒有留下本次 partial 檔。"""

    if destination.parent.is_dir() and not destination.parent.is_symlink():
        assert list(destination.parent.glob(f".{destination.name}.partial-*")) == []


def _block_forcing(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """以 sentinel 封鎖 RuntimeRequestFactory 與 forcing manager 的意外建立。"""

    calls = {"factory": 0, "manager": 0}

    def blocked_factory(*args: object, **kwargs: object) -> None:
        """report spec handler 若觸碰 dynamic factory，立即讓測試失敗。"""

        del args, kwargs
        calls["factory"] += 1
        raise AssertionError("report-spec-create 不得建立 RuntimeRequestFactory")

    def blocked_manager(*args: object, **kwargs: object) -> None:
        """report spec handler 若觸碰 OCM/NWW forcing，立即讓測試失敗。"""

        del args, kwargs
        calls["manager"] += 1
        raise AssertionError("report-spec-create 不得建立 ForcingWindowManager")

    monkeypatch.setattr(runtime, "RuntimeRequestFactory", blocked_factory)
    monkeypatch.setattr(
        runtime.ForcingWindowManager,
        "from_roots",
        staticmethod(blocked_manager),
    )
    return calls


def test_main_help_registers_report_spec_create_and_parser_has_exact_required_options(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """整合 main 必須登錄命令，且八個 CLI option 都不得有研究數值預設值。"""

    with pytest.raises(SystemExit) as error_info:
        cli.main(["--help"])
    assert error_info.value.code == 0
    assert "report-spec-create" in capsys.readouterr().out

    parser = cli._report_spec_create_parser()
    expected_destinations = {
        "aggregate_spec",
        "destination",
        "primary_kde_bandwidth_m",
        "minimum_kde_raw_count",
        "low_sample_min_member_count",
        "vertical_depth_bin_edges_m",
        "representative_trajectory_count_per_site",
        "representative_selection_seed",
    }
    actions = {
        action.dest: action
        for action in parser._actions
        if action.dest in expected_destinations
    }
    assert set(actions) == expected_destinations
    assert all(action.required for action in actions.values())


@pytest.mark.parametrize(
    "missing_option",
    (
        "--aggregate-spec",
        "--destination",
        "--primary-kde-bandwidth-m",
        "--minimum-kde-raw-count",
        "--low-sample-min-member-count",
        "--vertical-depth-bin-edges-m",
        "--representative-trajectory-count-per-site",
        "--representative-selection-seed",
    ),
)
def test_report_spec_create_missing_required_option_fails_in_argparse_before_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_option: str,
) -> None:
    """每個必填 option 缺失都應在 parser 階段以 status 2 結束，不讀寫任何檔案。"""

    aggregate_path = tmp_path / "aggregate_spec.json"
    destination = tmp_path / "output" / "report_spec.json"
    destination.parent.mkdir()
    args = _without_option(_report_spec_args(aggregate_path, destination), missing_option)
    calls = {"count": 0}

    def forbidden_io(*args: object, **kwargs: object) -> object:
        """若 parser 失敗前仍進入 production I/O，測試必須立即揭露。"""

        del args, kwargs
        calls["count"] += 1
        raise AssertionError("argparse 失敗前不得進入 report spec I/O")

    for module in (cli, report_spec_module):
        monkeypatch.setattr(module, "load_aggregate_spec", forbidden_io, raising=False)
        monkeypatch.setattr(module, "load_report_spec", forbidden_io, raising=False)
        monkeypatch.setattr(module, "write_report_spec", forbidden_io, raising=False)

    with pytest.raises(SystemExit) as error_info:
        cli.main(args)
    assert error_info.value.code == 2
    assert calls["count"] == 0
    assert not destination.exists()


@pytest.mark.parametrize(
    "option, values",
    (
        ("--primary-kde-bandwidth-m", ["0"]),
        ("--primary-kde-bandwidth-m", ["-1"]),
        ("--primary-kde-bandwidth-m", ["nan"]),
        ("--primary-kde-bandwidth-m", ["inf"]),
        ("--minimum-kde-raw-count", ["0"]),
        ("--minimum-kde-raw-count", ["-1"]),
        ("--minimum-kde-raw-count", ["1.0"]),
        ("--low-sample-min-member-count", ["0"]),
        ("--low-sample-min-member-count", ["-1"]),
        ("--low-sample-min-member-count", ["1.0"]),
        ("--vertical-depth-bin-edges-m", ["0"]),
        ("--vertical-depth-bin-edges-m", ["0", "nan"]),
        ("--vertical-depth-bin-edges-m", ["0", "inf"]),
        ("--representative-trajectory-count-per-site", ["0"]),
        ("--representative-trajectory-count-per-site", ["-1"]),
        ("--representative-trajectory-count-per-site", ["8.0"]),
        (
            "--representative-selection-seed",
            ["-1"],
        ),
    ),
)
def test_report_spec_create_parser_rejects_invalid_research_values_before_io(
    tmp_path: Path,
    option: str,
    values: list[str],
) -> None:
    """正值、finite、深度排序與 8 倍數／128-bit seed 限制都應回傳 argparse status 2。"""

    aggregate_path = tmp_path / "aggregate_spec.json"
    destination = tmp_path / "output" / "report_spec.json"
    destination.parent.mkdir()
    args = _replace_option(
        _report_spec_args(aggregate_path, destination),
        option,
        values,
    )

    with pytest.raises(SystemExit) as error_info:
        cli.main(args)
    assert error_info.value.code == 2
    assert not destination.exists()


@pytest.mark.parametrize(
    "option, values",
    (
        ("--vertical-depth-bin-edges-m", ["1", "5"]),
        ("--vertical-depth-bin-edges-m", ["0", "5", "5"]),
        ("--vertical-depth-bin-edges-m", ["0", "10", "5"]),
        ("--representative-trajectory-count-per-site", ["1"]),
        ("--representative-trajectory-count-per-site", ["7"]),
        ("--representative-trajectory-count-per-site", ["9"]),
        ("--representative-trajectory-count-per-site", ["12"]),
        (
            "--representative-selection-seed",
            ["340282366920938463463374607431768211456"],
        ),
    ),
)
def test_report_spec_create_writer_domain_rejects_semantic_values_with_fixed_error(
    tmp_path: Path,
    synthetic_aggregate_spec: AggregateSpec,
    option: str,
    values: list[str],
) -> None:
    """深度排序、代表數分層倍數與 seed 上限屬 ReportSpec domain，應固定收斂 ValueError。"""

    aggregate_path = tmp_path / "aggregate_spec.json"
    _write_aggregate_input(aggregate_path, synthetic_aggregate_spec)
    destination_parent = tmp_path / "output"
    destination_parent.mkdir()
    destination = destination_parent / "report_spec.json"
    args = _replace_option(
        _report_spec_args(aggregate_path, destination),
        option,
        values,
    )

    _assert_fixed_value_error(
        lambda: cli.main(args),
        path_markers=(tmp_path, aggregate_path, destination),
    )
    assert not destination.exists()
    _assert_no_partial(destination)


def test_report_spec_create_success_round_trips_real_writer_loader_and_binding_without_forcing(
    tmp_path: Path,
    synthetic_aggregate_spec: AggregateSpec,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """成功命令應發布 portable spec，並由實際 loader／binding 回讀驗證。"""

    aggregate_path = tmp_path / "aggregate_spec.json"
    aggregate_loaded = _write_aggregate_input(aggregate_path, synthetic_aggregate_spec)
    destination_parent = tmp_path / "report-output"
    destination_parent.mkdir()
    destination = destination_parent / "report_spec.json"
    aggregate_before = aggregate_path.read_bytes()
    forcing_calls = _block_forcing(monkeypatch)

    assert cli.main(_report_spec_args(aggregate_path, destination)) == 0
    result = json.loads(capsys.readouterr().out)
    loaded = load_report_spec(destination)
    assert validate_report_spec_against_aggregate_spec(loaded, aggregate_loaded) is None

    expected = {
        "report_spec_name": destination.name,
        "schema_version": REPORT_SPEC_SCHEMA_VERSION,
        "run_id": aggregate_loaded.run_id,
        "aggregate_spec_canonical_sha256": aggregate_loaded.canonical_sha256,
        "source_sha256": loaded.source_sha256,
        "canonical_sha256": loaded.canonical_sha256,
    }
    assert result == expected
    assert set(result) == {
        "report_spec_name",
        "schema_version",
        "run_id",
        "aggregate_spec_canonical_sha256",
        "source_sha256",
        "canonical_sha256",
    }
    assert set(json.loads(destination.read_bytes())) == _REPORT_INPUT_KEYS
    assert "source_sha256" not in json.loads(destination.read_bytes())
    assert "canonical_sha256" not in json.loads(destination.read_bytes())
    assert loaded.vertical_depth_bin_edges_m == (0.0, 5.0, 10.0)
    assert loaded.representative_trajectory_count_per_site == 8
    assert loaded.representative_selection_policy == "stable_hash_core_season_tide_v1"
    assert aggregate_path.read_bytes() == aggregate_before
    assert forcing_calls == {"factory": 0, "manager": 0}
    assert all(str(marker) not in repr(result) for marker in (tmp_path, aggregate_path))
    _assert_no_partial(destination)


@pytest.mark.parametrize("parent_kind", ("missing", "symlink"))
def test_report_spec_create_rejects_missing_or_symlink_destination_parent(
    tmp_path: Path,
    synthetic_aggregate_spec: AggregateSpec,
    parent_kind: str,
) -> None:
    """destination parent 不得由 CLI 自動 mkdir，也不得透過 symlink 發布。"""

    aggregate_path = tmp_path / "aggregate_spec.json"
    _write_aggregate_input(aggregate_path, synthetic_aggregate_spec)
    if parent_kind == "missing":
        parent = tmp_path / "not-created"
    else:
        referent = tmp_path / "outside-parent"
        referent.mkdir()
        parent = tmp_path / "parent-link"
        os.symlink(referent, parent)
    destination = parent / "report_spec.json"

    _assert_fixed_value_error(
        lambda: cli.main(_report_spec_args(aggregate_path, destination)),
        path_markers=(tmp_path, aggregate_path, destination),
    )
    assert not destination.exists()
    if parent_kind == "missing":
        assert not parent.exists()
    else:
        assert parent.is_symlink()
        assert list((tmp_path / "outside-parent").iterdir()) == []


@pytest.mark.parametrize("target_kind", ("file", "directory", "symlink", "broken-symlink"))
def test_report_spec_create_rejects_existing_target_without_overwrite(
    tmp_path: Path,
    synthetic_aggregate_spec: AggregateSpec,
    target_kind: str,
) -> None:
    """既有普通檔、目錄、symlink 與 broken symlink 都不得被 CLI 覆寫或刪除。"""

    aggregate_path = tmp_path / "aggregate_spec.json"
    _write_aggregate_input(aggregate_path, synthetic_aggregate_spec)
    parent = tmp_path / f"existing-{target_kind}"
    parent.mkdir()
    destination = parent / "report_spec.json"
    if target_kind == "file":
        protected_bytes = b"protected-report-spec"
        destination.write_bytes(protected_bytes)
    elif target_kind == "directory":
        destination.mkdir()
        (destination / "marker").write_bytes(b"protected-directory")
    elif target_kind == "symlink":
        referent = parent / "referent.json"
        referent.write_bytes(b"protected-referent")
        os.symlink(referent, destination)
    else:
        os.symlink(parent / "does-not-exist.json", destination)
    before_mode = destination.lstat().st_mode
    before_link = os.readlink(destination) if destination.is_symlink() else None

    _assert_fixed_value_error(
        lambda: cli.main(_report_spec_args(aggregate_path, destination)),
        path_markers=(tmp_path, destination),
    )
    assert destination.lstat().st_mode == before_mode
    if target_kind == "file":
        assert destination.read_bytes() == protected_bytes
    elif target_kind == "directory":
        assert (destination / "marker").read_bytes() == b"protected-directory"
    else:
        assert destination.is_symlink()
        assert os.readlink(destination) == before_link
    _assert_no_partial(destination)


@pytest.mark.parametrize("input_kind", ("missing", "symlink", "tampered"))
def test_report_spec_create_rejects_missing_symlink_or_tampered_aggregate(
    tmp_path: Path,
    synthetic_aggregate_spec: AggregateSpec,
    input_kind: str,
) -> None:
    """aggregate input 不是既有合法普通檔時，CLI 應固定失敗且不產生 partial。"""

    valid_path = tmp_path / "valid-aggregate.json"
    _write_aggregate_input(valid_path, synthetic_aggregate_spec)
    if input_kind == "missing":
        aggregate_path = tmp_path / "missing-aggregate.json"
    elif input_kind == "symlink":
        aggregate_path = tmp_path / "aggregate-link.json"
        os.symlink(valid_path, aggregate_path)
    else:
        aggregate_path = tmp_path / "tampered-aggregate.json"
        tampered = _aggregate_input_document(synthetic_aggregate_spec)
        tampered["kde_bandwidths_m"] = [100, 100, 300]
        aggregate_path.write_bytes(
            json.dumps(
                tampered,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
    destination_parent = tmp_path / "output"
    destination_parent.mkdir()
    destination = destination_parent / "report_spec.json"

    _assert_fixed_value_error(
        lambda: cli.main(_report_spec_args(aggregate_path, destination)),
        path_markers=(tmp_path, aggregate_path, destination),
    )
    assert not destination.exists()
    _assert_no_partial(destination)


def test_report_spec_create_rejects_primary_bandwidth_not_registered_in_aggregate(
    tmp_path: Path,
    synthetic_aggregate_spec: AggregateSpec,
) -> None:
    """primary KDE 帶寬只能取自 aggregate 已登錄的三個公尺制值。"""

    aggregate_path = tmp_path / "aggregate_spec.json"
    aggregate_loaded = _write_aggregate_input(aggregate_path, synthetic_aggregate_spec)
    destination_parent = tmp_path / "output"
    destination_parent.mkdir()
    destination = destination_parent / "report_spec.json"
    args = _replace_option(
        _report_spec_args(aggregate_path, destination),
        "--primary-kde-bandwidth-m",
        ["150"],
    )

    _assert_fixed_value_error(
        lambda: cli.main(args),
        path_markers=(tmp_path, aggregate_path, destination),
    )
    assert aggregate_loaded.run_id == synthetic_aggregate_spec.run_id
    assert not destination.exists()
    _assert_no_partial(destination)


def test_report_spec_create_preserves_post_replace_durability_runtime_error(
    tmp_path: Path,
    synthetic_aggregate_spec: AggregateSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """writer 已發布但 parent durability 不確定時，CLI 必須原樣保留 RuntimeError。"""

    aggregate_path = tmp_path / "aggregate_spec.json"
    _write_aggregate_input(aggregate_path, synthetic_aggregate_spec)
    destination_parent = tmp_path / "output"
    destination_parent.mkdir()
    destination = destination_parent / "report_spec.json"

    def fail_durability(*args: object, **kwargs: object) -> Path:
        """模擬 writer post-replace durability gate 的固定 RuntimeError。"""

        del args, kwargs
        raise RuntimeError(_DURABILITY_FAILURE)

    monkeypatch.setattr(cli, "write_report_spec", fail_durability, raising=False)
    monkeypatch.setattr(report_spec_module, "write_report_spec", fail_durability)

    with pytest.raises(RuntimeError) as error_info:
        cli.main(_report_spec_args(aggregate_path, destination))
    assert type(error_info.value) is RuntimeError
    assert str(error_info.value) == _DURABILITY_FAILURE
    assert error_info.value.__cause__ is None
    assert str(tmp_path) not in str(error_info.value)
    assert str(tmp_path) not in repr(error_info.value)
    assert not destination.exists()
    _assert_no_partial(destination)


def test_package_root_exposes_report_spec_public_exports() -> None:
    """套件 root 應提供 report spec public API，而不要求 caller 匯入 private helper。"""

    expected = {
        "REPORT_SPEC_SCHEMA_VERSION",
        "ReportSpec",
        "load_report_spec",
        "validate_report_spec_against_aggregate_spec",
        "write_report_spec",
    }
    assert expected <= set(lbt.__all__)
    assert all(hasattr(lbt, name) for name in expected)
    assert lbt.REPORT_SPEC_SCHEMA_VERSION == REPORT_SPEC_SCHEMA_VERSION
