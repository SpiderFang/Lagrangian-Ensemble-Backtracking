"""Aggregate release 公開 writer 的原子發布與失敗清理契約測試。

本模組重用 ``test_aggregate_release_source_validation`` 的完整 synthetic engineering
run 與 payload fixture，但每個案例都把 source workspace 複製到自己的暫存
``runs/<run_id>``，並把 ``aggregate_spec.json`` 複製成獨立檔案。測試會逐檔保存 source
run 的普通檔案 bytes 與 SHA-256，確認 writer 的 lock、驗證、partial 寫入、原子改名與
durability 失敗都不會改動 source run 或 caller 提供的 spec。

這些測試驗證的是檔案拓撲、來源 provenance、產品 checksum 與工程錯誤界面；fixture
數值不是正式 OCM schema 3／NWW3 schema 1 forcing、軌跡或科學成果，不得解讀為來源
機率、因果歸因或觀測驗證結果。
"""

from __future__ import annotations

import json
import os
import shutil
import stat
from dataclasses import dataclass
from functools import partial
from hashlib import sha256
from pathlib import Path

import pytest
import test_aggregate_release_public_validation as public_validation
from test_aggregate_release_source_validation import (
    _release_json_bytes,
    _SourceReleaseFixture,
)

import lagrangian_backtracking.aggregate_release as release_module
from lagrangian_backtracking.aggregate_release import (
    read_aggregate_release,
    validate_aggregate_release,
)
from lagrangian_backtracking.aggregate_release_codec import encode_aggregate_release_payload
from lagrangian_backtracking.run_locking import RunLockBusyError, acquire_run_lock

pytest_plugins = ("test_aggregate_release_source_validation",)

_FINAL_SUFFIX = ".aggregate-v1"
_WRITER_FAILURE_MESSAGE = "aggregate release 寫入失敗"
_PARENT_DURABILITY_MESSAGE = "aggregate release 已發布但 parent durability 未確認"


@dataclass(frozen=True, slots=True)
class _FreshWriterInputs:
    """保存單一測試專用的 source run 與獨立 AggregateSpec 路徑。

    ``source_run_root`` 絕不指向 module-scoped fixture；``aggregate_spec_path`` 也不直接
    使用 fixture release 內的檔案，避免 writer 的任何意外寫入污染其他案例。路徑只供
    測試呼叫 API，不能被寫入 release manifest，也不代表正式 SERVER 位置。
    """

    source_run_root: Path
    aggregate_spec_path: Path


def _writer() -> object:
    """取得待測公開 writer，缺少時以測試失敗揭露 production discrepancy。

    測試不能自行補一個假的 writer，否則會掩蓋 production 尚未提供固定 API 的事實；
    因此只回傳 production module 的 callable，讓各案例保留真實失敗證據。
    """

    candidate = getattr(release_module, "write_aggregate_release", None)
    if not callable(candidate):
        pytest.fail("aggregate_release 尚未提供公開 write_aggregate_release API")
    return candidate


def _fresh_writer_inputs(
    fixture: _SourceReleaseFixture,
    tmp_path: Path,
) -> _FreshWriterInputs:
    """把 fixture source 與 spec 複製成單一案例的隔離輸入。

    source workspace 固定放在 ``tmp_path/runs/<run_id>``，保留 run controller 建立的
    locks、plan、progress、checkpoint 與 trajectory shard 普通檔案；spec 則由 fixture
    release 的 exact bytes 複製到 source 外。這裡只做工程測試資料搬移，不建立或宣稱
    任何真實 OCM／NWW 科學結果。
    """

    source_template = Path(fixture.workspace)
    source_parent = tmp_path / "runs"
    source_parent.mkdir()
    source_root = source_parent / source_template.name
    shutil.copytree(source_template, source_root, symlinks=True)
    assert source_root.name == fixture.metadata.run_id

    spec_path = tmp_path / "aggregate_spec.json"
    spec_path.write_bytes((fixture.root / "aggregate_spec.json").read_bytes())
    return _FreshWriterInputs(
        source_run_root=source_root,
        aggregate_spec_path=spec_path,
    )


def _snapshot_source_tree(root: Path) -> dict[str, tuple[str, object, str]]:
    """遞迴保存 source tree 每個節點，普通檔案同時保存 exact bytes 與 SHA-256。

    ordinary file 的 bytes 是 writer 不得改寫的直接證據，SHA-256 則提供不依賴檔案
    mtime 的完整性摘要；目錄與 symbolic link 也記錄節點型別，避免 writer 以刪除後重建
    的方式掩蓋 source tree 拓撲變化。snapshot 只針對 synthetic source run，不讀取 raw
    OCM／NWW 檔案，也不把資料內容當成科學驗證。
    """

    snapshot: dict[str, tuple[str, object, str]] = {}
    paths = [root, *sorted(root.rglob("*"), key=lambda path: str(path))]
    for path in paths:
        relative_name = "." if path == root else str(path.relative_to(root))
        node_status = path.lstat()
        if stat.S_ISREG(node_status.st_mode):
            raw_bytes = path.read_bytes()
            snapshot[relative_name] = (
                "file",
                raw_bytes,
                sha256(raw_bytes).hexdigest(),
            )
        elif stat.S_ISDIR(node_status.st_mode):
            snapshot[relative_name] = ("directory", "", "")
        elif stat.S_ISLNK(node_status.st_mode):
            snapshot[relative_name] = ("symlink", os.readlink(path), "")
        else:
            snapshot[relative_name] = ("other", node_status.st_mode, "")
    return snapshot


def _snapshot_file(path: Path) -> tuple[bytes, str]:
    """保存獨立 spec 的 exact bytes 與 SHA-256，供 writer 後 immutable 比對。"""

    raw_bytes = path.read_bytes()
    return raw_bytes, sha256(raw_bytes).hexdigest()


def _snapshot_node(path: Path) -> tuple[str, object, str]:
    """保存 final conflict 節點的型別、內容與摘要，確認 writer 未覆寫既有目標。"""

    node_status = path.lstat()
    if stat.S_ISREG(node_status.st_mode):
        raw_bytes = path.read_bytes()
        return "file", raw_bytes, sha256(raw_bytes).hexdigest()
    if stat.S_ISDIR(node_status.st_mode):
        return "directory", _snapshot_source_tree(path), ""
    if stat.S_ISLNK(node_status.st_mode):
        return "symlink", os.readlink(path), ""
    return "other", node_status.st_mode, ""


def _directory_names(path: Path) -> frozenset[str]:
    """取得既有 parent 的直接節點名稱，驗證 owned partial 是否被完整清理。"""

    return frozenset(entry.name for entry in path.iterdir())


def _assert_source_and_spec_unchanged(
    inputs: _FreshWriterInputs,
    source_snapshot: dict[str, tuple[str, object, str]],
    spec_snapshot: tuple[bytes, str],
) -> None:
    """確認 writer 後 source tree 與獨立 spec 的每份 bytes/hash 完全不變。"""

    assert _snapshot_source_tree(inputs.source_run_root) == source_snapshot
    assert _snapshot_file(inputs.aggregate_spec_path) == spec_snapshot


def _assert_fixed_writer_value_error(
    operation: object,
    *,
    path_marker: Path,
) -> None:
    """確認 rename 前所有一般失敗都使用固定 ValueError 且不洩漏暫存路徑。"""

    assert callable(operation)
    with pytest.raises(ValueError) as error_info:
        operation()
    error = error_info.value
    assert type(error) is ValueError
    assert str(error) == _WRITER_FAILURE_MESSAGE
    assert str(path_marker) not in str(error)
    assert str(path_marker) not in repr(error)


def _assert_final_release_topology(final_path: Path) -> None:
    """確認 final release 恰有 33 個普通非 symlink 檔案且沒有子目錄。"""

    root_status = final_path.lstat()
    assert stat.S_ISDIR(root_status.st_mode)
    assert not stat.S_ISLNK(root_status.st_mode)
    entries = list(final_path.iterdir())
    assert len(entries) == 33
    for entry in entries:
        entry_status = entry.lstat()
        assert stat.S_ISREG(entry_status.st_mode)
        assert not stat.S_ISLNK(entry_status.st_mode)


def _assert_successful_release(
    final_path: Path,
    fixture: _SourceReleaseFixture,
) -> None:
    """驗證成功發布的 validator 與 reader／encoder products exact round-trip。"""

    assert isinstance(final_path, Path)
    assert final_path.exists()
    _assert_final_release_topology(final_path)
    report = validate_aggregate_release(final_path)
    assert report["valid"] is True
    assert report["errors"] == []

    read_payload = read_aggregate_release(final_path)
    expected_products = encode_aggregate_release_payload(fixture.payload)
    actual_products = encode_aggregate_release_payload(read_payload)
    public_validation._assert_encoded_products_exact(expected_products, actual_products)


def _invoke_writer(
    inputs: _FreshWriterInputs,
    fixture: _SourceReleaseFixture,
    *,
    destination: Path | None = None,
) -> object:
    """以固定 keyword-only API 呼叫 production writer，供 failure helper 延後執行。"""

    arguments: dict[str, object] = {
        "source_run_root": inputs.source_run_root,
        "aggregate_spec_path": inputs.aggregate_spec_path,
        "payload": fixture.payload,
    }
    if destination is not None:
        arguments["destination"] = destination
    return _writer()(**arguments)


def test_public_writer_is_exported_with_stable_failure_contract() -> None:
    """公開模組必須輸出 writer 名稱；一般錯誤文字由其他案例固定驗證。"""

    assert "write_aggregate_release" in release_module.__all__


def test_writer_default_sibling_publishes_exact_release_and_preserves_source(
    valid_source_release: _SourceReleaseFixture,
    tmp_path: Path,
) -> None:
    """省略 destination 時，writer 應在 source 同層原子發布完整 33 檔 release。"""

    inputs = _fresh_writer_inputs(valid_source_release, tmp_path)
    source_parent = inputs.source_run_root.parent
    expected_final = source_parent / f"{valid_source_release.metadata.run_id}{_FINAL_SUFFIX}"
    parent_before = _directory_names(source_parent)
    source_snapshot = _snapshot_source_tree(inputs.source_run_root)
    spec_snapshot = _snapshot_file(inputs.aggregate_spec_path)

    writer = _writer()
    result = writer(
        source_run_root=inputs.source_run_root,
        aggregate_spec_path=inputs.aggregate_spec_path,
        payload=valid_source_release.payload,
    )

    assert result == expected_final
    _assert_successful_release(expected_final, valid_source_release)
    assert _directory_names(source_parent) == parent_before | {expected_final.name}
    _assert_source_and_spec_unchanged(inputs, source_snapshot, spec_snapshot)


def test_writer_explicit_exact_destination_publishes_fresh_release(
    valid_source_release: _SourceReleaseFixture,
    tmp_path: Path,
) -> None:
    """明示 source 同層且 basename 精確的 destination 也應成功發布。"""

    inputs = _fresh_writer_inputs(valid_source_release, tmp_path)
    source_parent = inputs.source_run_root.parent
    destination = source_parent / f"{valid_source_release.metadata.run_id}{_FINAL_SUFFIX}"
    parent_before = _directory_names(source_parent)
    source_snapshot = _snapshot_source_tree(inputs.source_run_root)
    spec_snapshot = _snapshot_file(inputs.aggregate_spec_path)

    result = _writer()(
        source_run_root=inputs.source_run_root,
        aggregate_spec_path=inputs.aggregate_spec_path,
        payload=valid_source_release.payload,
        destination=destination,
    )

    assert result == destination
    _assert_successful_release(destination, valid_source_release)
    assert _directory_names(source_parent) == parent_before | {destination.name}
    _assert_source_and_spec_unchanged(inputs, source_snapshot, spec_snapshot)


def test_writer_accepts_os_lexical_alias_for_source_and_destination(
    valid_source_release: _SourceReleaseFixture,
    tmp_path: Path,
) -> None:
    """source 與 destination 共用合法 OS lexical alias 時，writer 應仍可原子發布。

    macOS 常以 ``/var`` symbolic link 指向 ``/private/var``；這是作業系統提供的 lexical
    alias，不是 source run root 或其 parent final node 本身的 symbolic link。測試只在
    此平台且 pytest 暫存路徑確實位於 ``/private/var`` 時執行，並要求 source 與 destination
    使用同一個 alias sibling。writer 應比較最終普通節點與 parent identity，而不能把
    ancestor alias 誤判成 unsafe path。內容仍是 synthetic engineering fixture，不是
    OCM／NWW 科學成果。
    """

    private_var = Path("/private/var")
    var_alias = Path("/var")
    if not var_alias.is_symlink():
        pytest.skip("此平台沒有 /var lexical alias")
    try:
        relative_tmp_path = Path(tmp_path).relative_to(private_var)
    except ValueError:
        pytest.skip("pytest 暫存路徑不在 /private/var，無法建立 OS lexical alias")

    inputs = _fresh_writer_inputs(valid_source_release, tmp_path)
    physical_source_root = inputs.source_run_root
    alias_tmp_path = var_alias / relative_tmp_path
    alias_source_root = alias_tmp_path / "runs" / physical_source_root.name
    alias_source_parent = alias_source_root.parent
    final_name = f"{valid_source_release.metadata.run_id}{_FINAL_SUFFIX}"
    alias_destination = alias_source_parent / final_name
    physical_destination = physical_source_root.parent / final_name
    assert alias_source_root.resolve(strict=True) == physical_source_root.resolve(strict=True)
    assert not physical_source_root.is_symlink()
    assert not physical_source_root.parent.is_symlink()
    assert not alias_source_root.is_symlink()
    assert not alias_source_parent.is_symlink()
    assert not alias_destination.exists()

    parent_before = _directory_names(physical_source_root.parent)
    source_snapshot = _snapshot_source_tree(physical_source_root)
    spec_snapshot = _snapshot_file(inputs.aggregate_spec_path)
    result = _writer()(
        source_run_root=alias_source_root,
        aggregate_spec_path=inputs.aggregate_spec_path,
        payload=valid_source_release.payload,
        destination=alias_destination,
    )

    assert result == alias_destination
    assert physical_destination.exists()
    _assert_successful_release(alias_destination, valid_source_release)
    assert _directory_names(physical_source_root.parent) == parent_before | {final_name}
    _assert_source_and_spec_unchanged(inputs, source_snapshot, spec_snapshot)


@pytest.mark.parametrize(
    "destination_case",
    ["wrong-basename", "different-parent", "lexical-parent-traversal", "symlink-parent"],
)
def test_writer_rejects_invalid_destination_without_partial_or_source_mutation(
    valid_source_release: _SourceReleaseFixture,
    tmp_path: Path,
    destination_case: str,
) -> None:
    """錯誤 destination 必須在建立 owned partial 前固定拒絕。"""

    inputs = _fresh_writer_inputs(valid_source_release, tmp_path)
    source_parent = inputs.source_run_root.parent
    final_name = f"{valid_source_release.metadata.run_id}{_FINAL_SUFFIX}"
    parents_before: dict[Path, frozenset[str]] = {
        source_parent: _directory_names(source_parent),
    }

    if destination_case == "wrong-basename":
        destination = source_parent / "wrong-release-name.aggregate-v1"
    elif destination_case == "different-parent":
        different_parent = tmp_path / "different-parent"
        different_parent.mkdir()
        parents_before[different_parent] = _directory_names(different_parent)
        destination = different_parent / final_name
    elif destination_case == "lexical-parent-traversal":
        lexical_parent = source_parent / "lexical-parent"
        lexical_parent.mkdir()
        parents_before[lexical_parent] = _directory_names(lexical_parent)
        destination = lexical_parent / ".." / final_name
    else:
        symlink_parent = tmp_path / "symlink-parent"
        symlink_parent.symlink_to(source_parent, target_is_directory=True)
        destination = symlink_parent / final_name

    # lexical-parent case 的 helper 目錄是測試預先建立的合法 baseline，不屬於 writer owned
    # partial；重新擷取 source parent 名稱後，後續只檢查 writer 是否新增或刪除節點。
    parents_before[source_parent] = _directory_names(source_parent)
    assert not destination.exists()
    source_snapshot = _snapshot_source_tree(inputs.source_run_root)
    spec_snapshot = _snapshot_file(inputs.aggregate_spec_path)
    operation = partial(
        _invoke_writer,
        inputs,
        valid_source_release,
        destination=destination,
    )

    _assert_fixed_writer_value_error(operation, path_marker=tmp_path)
    for parent, names_before in parents_before.items():
        assert _directory_names(parent) == names_before
    _assert_source_and_spec_unchanged(inputs, source_snapshot, spec_snapshot)


def test_writer_rejects_source_basename_mismatch_before_publish(
    valid_source_release: _SourceReleaseFixture,
    tmp_path: Path,
) -> None:
    """source root basename 與 immutable plan run_id 不一致時不得建立 release。"""

    inputs = _fresh_writer_inputs(valid_source_release, tmp_path)
    original_root = inputs.source_run_root
    wrong_root = original_root.parent / "wrong-source-basename"
    original_root.rename(wrong_root)
    inputs = _FreshWriterInputs(
        source_run_root=wrong_root,
        aggregate_spec_path=inputs.aggregate_spec_path,
    )
    source_parent = wrong_root.parent
    parent_before = _directory_names(source_parent)
    source_snapshot = _snapshot_source_tree(wrong_root)
    spec_snapshot = _snapshot_file(inputs.aggregate_spec_path)

    operation = partial(_invoke_writer, inputs, valid_source_release)
    _assert_fixed_writer_value_error(operation, path_marker=tmp_path)
    assert _directory_names(source_parent) == parent_before
    _assert_source_and_spec_unchanged(inputs, source_snapshot, spec_snapshot)


@pytest.mark.parametrize(
    "conflict_case",
    ["ordinary-file", "ordinary-directory", "symlink", "broken-symlink"],
)
def test_writer_preserves_preexisting_final_target(
    valid_source_release: _SourceReleaseFixture,
    tmp_path: Path,
    conflict_case: str,
) -> None:
    """final 已被任何節點占用時，writer 不得覆寫、刪除或替換既有目標。"""

    inputs = _fresh_writer_inputs(valid_source_release, tmp_path)
    source_parent = inputs.source_run_root.parent
    final_path = source_parent / f"{valid_source_release.metadata.run_id}{_FINAL_SUFFIX}"
    if conflict_case == "ordinary-file":
        final_path.write_bytes(b"preexisting-final-file")
    elif conflict_case == "ordinary-directory":
        final_path.mkdir()
        (final_path / "marker.txt").write_bytes(b"preserve-directory")
    elif conflict_case == "symlink":
        link_target = tmp_path / "ordinary-link-target"
        link_target.write_bytes(b"preserve-link-target")
        final_path.symlink_to(link_target)
    else:
        final_path.symlink_to(tmp_path / "missing-final-target")

    target_snapshot = _snapshot_node(final_path)
    parent_before = _directory_names(source_parent)
    source_snapshot = _snapshot_source_tree(inputs.source_run_root)
    spec_snapshot = _snapshot_file(inputs.aggregate_spec_path)
    operation = partial(_invoke_writer, inputs, valid_source_release)

    _assert_fixed_writer_value_error(operation, path_marker=tmp_path)
    assert _directory_names(source_parent) == parent_before
    assert _snapshot_node(final_path) == target_snapshot
    _assert_source_and_spec_unchanged(inputs, source_snapshot, spec_snapshot)


def test_writer_returns_original_lock_contention_without_partial(
    valid_source_release: _SourceReleaseFixture,
    tmp_path: Path,
) -> None:
    """run gate 被占用時必須立即保留原始 RunLockBusyError，且不建立任何 output。"""

    inputs = _fresh_writer_inputs(valid_source_release, tmp_path)
    source_parent = inputs.source_run_root.parent
    parent_before = _directory_names(source_parent)
    source_snapshot = _snapshot_source_tree(inputs.source_run_root)
    spec_snapshot = _snapshot_file(inputs.aggregate_spec_path)
    lock_path = inputs.source_run_root / "locks" / "run_gate.lock"

    with acquire_run_lock(lock_path, mode="exclusive", blocking=False):
        with pytest.raises(RunLockBusyError) as error_info:
            _writer()(
                source_run_root=inputs.source_run_root,
                aggregate_spec_path=inputs.aggregate_spec_path,
                payload=valid_source_release.payload,
            )
        assert type(error_info.value) is RunLockBusyError
        assert str(tmp_path) not in repr(error_info.value)

    assert _directory_names(source_parent) == parent_before
    _assert_source_and_spec_unchanged(inputs, source_snapshot, spec_snapshot)


@pytest.mark.parametrize("binding_case", ["tampered-spec", "running-progress"])
def test_writer_rejects_source_or_spec_binding_before_owned_partial(
    valid_source_release: _SourceReleaseFixture,
    tmp_path: Path,
    binding_case: str,
) -> None:
    """source/spec binding 失敗必須發生在 partial 建立前，且不修改 caller 輸入。"""

    inputs = _fresh_writer_inputs(valid_source_release, tmp_path)
    if binding_case == "tampered-spec":
        spec_document = json.loads(inputs.aggregate_spec_path.read_bytes().decode("utf-8"))
        assert type(spec_document) is dict
        spec_document["run_id"] = "tampered-synthetic-run"
        inputs.aggregate_spec_path.write_bytes(_release_json_bytes(spec_document))
    else:
        progress_path = inputs.source_run_root / "run_progress.json"
        progress_document = json.loads(progress_path.read_bytes().decode("utf-8"))
        assert type(progress_document) is dict
        shards = progress_document["shards"]
        assert type(shards) is dict and len(shards) == 1
        shard = next(iter(shards.values()))
        assert type(shard) is dict
        shard["lifecycle"] = "RUNNING"
        shard["output_relative_path"] = None
        shard["failure_relative_path"] = None
        shard["error_code"] = None
        shard["attempt_count"] = max(1, int(shard["attempt_count"]))
        progress_document["run_lifecycle"] = "RUNNING"
        progress_path.write_bytes(_release_json_bytes(progress_document))

    source_parent = inputs.source_run_root.parent
    parent_before = _directory_names(source_parent)
    source_snapshot = _snapshot_source_tree(inputs.source_run_root)
    spec_snapshot = _snapshot_file(inputs.aggregate_spec_path)
    operation = partial(_invoke_writer, inputs, valid_source_release)

    _assert_fixed_writer_value_error(operation, path_marker=tmp_path)
    assert _directory_names(source_parent) == parent_before
    _assert_source_and_spec_unchanged(inputs, source_snapshot, spec_snapshot)


@pytest.mark.parametrize(
    "failure_point",
    [
        "write-prepared",
        "inspect",
        "second-trajectory-validation",
        "replace",
    ],
)
def test_writer_cleans_only_owned_partial_on_publish_failures(
    valid_source_release: _SourceReleaseFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    """partial 失敗只可清理本次 owned 目錄，不得碰 unrelated manual partial。"""

    inputs = _fresh_writer_inputs(valid_source_release, tmp_path)
    source_parent = inputs.source_run_root.parent
    unrelated_partial = source_parent / ".keep.partial-manual"
    unrelated_partial.mkdir()
    marker = unrelated_partial / "marker.txt"
    marker.write_bytes(b"must-preserve")
    parent_before = _directory_names(source_parent)
    source_snapshot = _snapshot_source_tree(inputs.source_run_root)
    spec_snapshot = _snapshot_file(inputs.aggregate_spec_path)

    if failure_point == "write-prepared":
        def fail_write(*_args: object, **_kwargs: object) -> object:
            """注入 partial writer 失敗，不模擬任何其他 stage。"""

            raise RuntimeError("injected partial write failure")

        monkeypatch.setattr(release_module, "_write_prepared_release_directory", fail_write)
    elif failure_point == "inspect":
        def fail_inspect(*_args: object, **_kwargs: object) -> object:
            """注入 partial 完成後 inspection 失敗。"""

            raise RuntimeError("injected inspection failure")

        monkeypatch.setattr(release_module, "_inspect_aggregate_release", fail_inspect)
    elif failure_point == "second-trajectory-validation":
        real_validate = release_module._validate_current_trajectory_manifests
        validation_count = 0

        def validate_twice(*args: object, **kwargs: object) -> object:
            """第一次使用真實 output gate，第二次注入 publish 前失敗。"""

            nonlocal validation_count
            validation_count += 1
            if validation_count == 1:
                return real_validate(*args, **kwargs)
            raise RuntimeError("injected second trajectory validation failure")

        monkeypatch.setattr(
            release_module,
            "_validate_current_trajectory_manifests",
            validate_twice,
        )
    else:
        def fail_replace(*_args: object, **_kwargs: object) -> object:
            """注入 final rename 前的 os.replace failure。"""

            raise OSError("injected replace failure")

        monkeypatch.setattr(release_module.os, "replace", fail_replace)

    operation = partial(_invoke_writer, inputs, valid_source_release)
    _assert_fixed_writer_value_error(operation, path_marker=tmp_path)
    assert _directory_names(source_parent) == parent_before
    assert marker.read_bytes() == b"must-preserve"
    assert unrelated_partial.is_dir()
    _assert_source_and_spec_unchanged(inputs, source_snapshot, spec_snapshot)
    if failure_point == "second-trajectory-validation":
        assert validation_count == 2


def test_writer_keeps_final_after_parent_fsync_uncertainty(
    valid_source_release: _SourceReleaseFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """os.replace 成功後 parent fsync 失敗時，final 保留並改拋固定 RuntimeError。"""

    inputs = _fresh_writer_inputs(valid_source_release, tmp_path)
    source_parent = inputs.source_run_root.parent
    final_path = source_parent / f"{valid_source_release.metadata.run_id}{_FINAL_SUFFIX}"
    parent_before = _directory_names(source_parent)
    source_snapshot = _snapshot_source_tree(inputs.source_run_root)
    spec_snapshot = _snapshot_file(inputs.aggregate_spec_path)
    parent_status = os.stat(source_parent)
    real_fsync = release_module.os.fsync
    parent_fsync_calls = 0

    def fail_published_parent_fsync(descriptor: int) -> None:
        """只在 final 已存在且 descriptor 是 final parent 時注入 OSError。"""

        nonlocal parent_fsync_calls
        descriptor_status = os.fstat(descriptor)
        if (
            final_path.exists()
            and descriptor_status.st_dev == parent_status.st_dev
            and descriptor_status.st_ino == parent_status.st_ino
        ):
            parent_fsync_calls += 1
            raise OSError("injected parent durability uncertainty")
        real_fsync(descriptor)

    monkeypatch.setattr(release_module.os, "fsync", fail_published_parent_fsync)
    with pytest.raises(RuntimeError) as error_info:
        _writer()(
            source_run_root=inputs.source_run_root,
            aggregate_spec_path=inputs.aggregate_spec_path,
            payload=valid_source_release.payload,
        )
    error = error_info.value
    assert type(error) is RuntimeError
    assert str(error) == _PARENT_DURABILITY_MESSAGE
    assert str(tmp_path) not in str(error)
    assert str(tmp_path) not in repr(error)
    assert parent_fsync_calls == 1
    assert final_path.exists()

    # 後續 validator 不需要 fsync；先還原注入，避免測試驗證本身受到 monkeypatch 影響。
    monkeypatch.undo()
    _assert_successful_release(final_path, valid_source_release)
    assert _directory_names(source_parent) == parent_before | {final_path.name}
    _assert_source_and_spec_unchanged(inputs, source_snapshot, spec_snapshot)
