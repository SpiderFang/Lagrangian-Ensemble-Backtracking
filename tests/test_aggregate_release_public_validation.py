"""Aggregate release 公開 validator／reader 的最小 full-stack 測試。

本模組重用 ``test_aggregate_release_source_validation`` 已建立的完整 synthetic
engineering release fixture，不再建立第二套 payload、run plan 或 33 檔 release。
測試只確認公開 API 的成功回傳形狀、產品 exact round-trip 與最小錯誤隱私契約；fixture
中的數值不是實際 OCM／NWW forcing、軌跡或科學成果，不能被解讀為來源機率、因果歸因
或觀測驗證結果。
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_aggregate_release_source_validation import (
    _copy_release,
    _read_manifest,
    _release_json_bytes,
    _SourceReleaseFixture,
    _write_manifest,
)

from lagrangian_backtracking.aggregate_release import (
    TABLE_SCHEMAS,
    _validate_release_topology_and_checksums,
    read_aggregate_release,
    validate_aggregate_release,
)
from lagrangian_backtracking.aggregate_release_codec import encode_aggregate_release_payload
from lagrangian_backtracking.aggregate_release_payload import AggregateReleasePayload

pytest_plugins = ("test_aggregate_release_source_validation",)


def _assert_public_json_safe(value: object) -> None:
    """遞迴確認公開 validator 結果只含 JSON 原生值。

    ``validate_aggregate_release`` 是供 CLI、SERVER job 與報告工具使用的邊界，因此
    結果不得攜帶暫存路徑、NumPy／Arrow 物件或完整 payload reference。只接受 JSON
    object、array、字串、數值、布林與 null；不使用 ``default=str`` 掩蓋不可序列化值。
    """

    assert not isinstance(value, (Path, np.ndarray, AggregateReleasePayload))
    if type(value) in {str, int, float, bool} or value is None:
        return
    if type(value) is dict:
        for key, item in value.items():
            assert type(key) is str
            _assert_public_json_safe(item)
        return
    if type(value) is list:
        for item in value:
            _assert_public_json_safe(item)
        return
    raise AssertionError(f"公開結果含未登錄型別：{type(value)!r}")


def _assert_validation_failure(
    report: dict[str, object],
    *,
    expected_stage: str,
    path_marker: Path,
) -> None:
    """確認失敗報告只有固定三欄，且不洩漏 caller 的暫存路徑。

    stage code 是公開 validator 唯一允許的診斷資訊；空 summary 代表尚未建立任何
    可供下游解讀的 release metadata，避免錯誤案例混入部分結果。
    """

    assert set(report) == {"valid", "errors", "summary"}
    assert report == {
        "valid": False,
        "errors": [expected_stage],
        "summary": {},
    }
    _assert_public_json_safe(report)
    json.dumps(report, ensure_ascii=False, sort_keys=True)
    assert str(path_marker) not in repr(report)


def _assert_encoded_products_exact(expected: object, actual: object) -> None:
    """逐列、逐陣列比對兩份 encoded products 的 exact storage 語意。

    九張表的欄位順序與每列 scalar 型別不可被 reader 重新排序或轉型；十八個陣列的
    dtype、shape 與元素則必須完全一致。此 helper 只驗證公開 read 後重新 encode 的
    產品，不重新計算事件、路徑、分母或任何科學統計量。
    """

    assert set(expected.tables) == set(TABLE_SCHEMAS)
    assert set(actual.tables) == set(TABLE_SCHEMAS)
    assert set(actual.arrays) == set(expected.arrays)

    for file_name in TABLE_SCHEMAS:
        expected_rows = expected.tables[file_name]
        actual_rows = actual.tables[file_name]
        assert len(actual_rows) == len(expected_rows)
        for expected_row, actual_row in zip(expected_rows, actual_rows, strict=True):
            assert tuple(actual_row) == tuple(expected_row)
            for field_name in expected_row:
                expected_value = expected_row[field_name]
                actual_value = actual_row[field_name]
                assert type(actual_value) is type(expected_value)
                if type(expected_value) is float:
                    assert np.float64(actual_value).tobytes() == np.float64(
                        expected_value
                    ).tobytes()
                else:
                    assert actual_value == expected_value

    for file_name, expected_array in expected.arrays.items():
        actual_array = actual.arrays[file_name]
        assert actual_array.dtype == expected_array.dtype
        assert actual_array.shape == expected_array.shape
        np.testing.assert_array_equal(actual_array, expected_array)


def test_validate_success_returns_exact_json_safe_summary(
    valid_source_release: _SourceReleaseFixture,
) -> None:
    """合法 release 的公開 validator 應回傳固定 root 與 metadata 摘要。"""

    report = validate_aggregate_release(valid_source_release.root)
    expected_summary = {
        "schema_version": valid_source_release.metadata.schema_version,
        "run_id": valid_source_release.metadata.run_id,
        "run_kind": valid_source_release.metadata.run_kind,
        "experiment_case_id": valid_source_release.metadata.experiment_case_id,
        "members_per_scenario": valid_source_release.metadata.members_per_scenario,
        "input_particle_count": valid_source_release.metadata.input_particle_count,
        "shard_row_count": valid_source_release.metadata.shard_row_count,
        "scenario_row_count": valid_source_release.metadata.scenario_row_count,
        "site_row_count": valid_source_release.metadata.site_row_count,
        "boundary_row_count": valid_source_release.metadata.boundary_row_count,
        "source_receptor_row_count": valid_source_release.metadata.source_receptor_row_count,
        "payload_file_count": 32,
    }

    assert set(report) == {"valid", "errors", "summary"}
    assert report["valid"] is True
    assert report["errors"] == []
    assert report["summary"] == expected_summary
    assert set(report["summary"]) == set(expected_summary)
    _assert_public_json_safe(report)
    json.dumps(report, ensure_ascii=False, sort_keys=True)


def test_read_success_reencodes_all_products_exactly(
    valid_source_release: _SourceReleaseFixture,
) -> None:
    """合法 release 的公開 reader 應還原 payload 並保留九表／十八陣列 exact 值。"""

    read_payload = read_aggregate_release(valid_source_release.root)
    assert type(read_payload) is AggregateReleasePayload

    expected_products = encode_aggregate_release_payload(valid_source_release.payload)
    actual_products = encode_aggregate_release_payload(read_payload)
    _assert_encoded_products_exact(expected_products, actual_products)

    # identity/provenance 不可因磁碟 round-trip 被重新推導、遺失或換成另一份 source。
    for field_name in (
        "run_id",
        "run_kind",
        "experiment_case_id",
        "members_per_scenario",
        "config_hash",
        "checkpoint_input_binding_hash",
        "source_run_plan_sha256",
        "source_run_progress_sha256",
        "source_normalized_config_sha256",
        "source_input_inventory_sha256",
    ):
        assert getattr(read_payload, field_name) == getattr(
            valid_source_release.payload, field_name
        )
    assert (
        read_payload.aggregate_spec.source_sha256
        == valid_source_release.payload.aggregate_spec.source_sha256
    )
    assert (
        read_payload.aggregate_spec.canonical_sha256
        == valid_source_release.payload.aggregate_spec.canonical_sha256
    )


def test_validate_missing_path_returns_topology_failure(
    tmp_path: Path,
) -> None:
    """不存在的 final release path 應只回傳 topology stage。"""

    missing_path = tmp_path / "missing.aggregate-v1"
    report = validate_aggregate_release(missing_path)
    _assert_validation_failure(report, expected_stage="topology", path_marker=tmp_path)


def test_validate_wrong_basename_returns_name_failure(
    valid_source_release: _SourceReleaseFixture,
    tmp_path: Path,
) -> None:
    """內容合法但 final basename 不符時，公開 validator 應回傳 name stage。"""

    copied_root = _copy_release(valid_source_release, tmp_path)
    wrong_root = tmp_path / "wrong-release-name"
    copied_root.rename(wrong_root)

    report = validate_aggregate_release(wrong_root)
    _assert_validation_failure(report, expected_stage="name", path_marker=tmp_path)


def test_read_missing_path_has_fixed_error_message(
    tmp_path: Path,
) -> None:
    """公開 reader 的不存在路徑錯誤應固定且不暴露 path 或 stage。"""

    missing_path = tmp_path / "missing.aggregate-v1"
    with pytest.raises(ValueError) as error_info:
        read_aggregate_release(missing_path)

    message = str(error_info.value)
    assert message == "aggregate release 驗證失敗"
    assert str(tmp_path) not in message
    assert "topology" not in message
    assert "manifest" not in message
    assert "name" not in message


def _write_canonical_manifest_for_test(
    root: Path,
    document: dict[str, object],
) -> None:
    """以 production 相同的 canonical bytes 寫回 tamper 測試 manifest。

    測試只修改複製出的 synthetic engineering release；manifest 必須仍使用排序鍵、
    compact separators、UTF-8 與單一尾端換行，才能讓 products tamper 通過前置的
    manifest 與所有檔案 checksum gate，確實測到產品 contract stage。這些資料不是實際
    OCM schema 3／NWW3 schema 1 forcing 或科學成果。
    """

    _write_manifest(root, document)
    assert (root / "aggregate_manifest.json").read_bytes() == _release_json_bytes(document)


def _assert_read_has_private_fixed_failure(root: Path, *, path_marker: Path) -> None:
    """確認公開 reader 對任何驗證階段都只回傳固定且無路徑的 ValueError。

    caller 不應從例外判斷失敗發生於 source 或 decoder，也不可取得本機／SERVER 暫存
    路徑。這個隱私契約只描述 synthetic engineering release 的讀取失敗，不代表已產生
    或驗證任何正式 OCM／NWW 科學成果。
    """

    with pytest.raises(ValueError) as error_info:
        read_aggregate_release(root)

    error = error_info.value
    assert type(error) is ValueError
    assert str(error) == "aggregate release 驗證失敗"
    representation = repr(error)
    assert str(path_marker) not in representation
    for stage in (
        "topology",
        "manifest",
        "name",
        "checksum",
        "source",
        "products",
        "decoder",
    ):
        assert stage not in representation


def test_validate_parquet_row_count_contract_mismatch_is_products_failure(
    valid_source_release: _SourceReleaseFixture,
    tmp_path: Path,
) -> None:
    """只把 Parquet row_count 宣告加一時，公開 validator 應回報 products。"""

    root = _copy_release(valid_source_release, tmp_path)
    document = _read_manifest(root)
    files = document["files"]
    assert type(files) is dict
    contract = files["shard_bindings.parquet"]
    assert type(contract) is dict
    original_row_count = valid_source_release.contracts["shard_bindings.parquet"][
        "row_count"
    ]
    assert type(original_row_count) is int
    assert contract["row_count"] == original_row_count
    contract["row_count"] = original_row_count + 1
    # Parquet bytes 與 manifest 既有 SHA-256 都不變；只有合法的 row_count contract 被竄改。
    assert contract["sha256"] == valid_source_release.contracts["shard_bindings.parquet"][
        "sha256"
    ]
    _write_canonical_manifest_for_test(root, document)

    # foundation 必須先成功，證明 canonical manifest、contract 型別與全部 checksum 均通過。
    _validate_release_topology_and_checksums(root, require_final_name=True)
    report = validate_aggregate_release(root)
    _assert_validation_failure(report, expected_stage="products", path_marker=tmp_path)


def test_validate_npy_shape_and_element_contract_mismatch_is_products_failure(
    valid_source_release: _SourceReleaseFixture,
    tmp_path: Path,
) -> None:
    """只把 NPY shape[0] 與 element_count 同加一時，公開 validator 應回報 products。"""

    root = _copy_release(valid_source_release, tmp_path)
    document = _read_manifest(root)
    files = document["files"]
    assert type(files) is dict
    contract = files["age_bin_edges_seconds.npy"]
    assert type(contract) is dict
    shape = contract["shape"]
    assert type(shape) is list and len(shape) == 1
    original_length = shape[0]
    original_element_count = contract["element_count"]
    assert type(original_length) is int
    assert type(original_element_count) is int
    shape[0] = original_length + 1
    contract["element_count"] = original_element_count + 1
    # NPY bytes、dtype 與 SHA-256 都不變；manifest 的兩個數量欄位仍各自是合法整數。
    assert contract["sha256"] == valid_source_release.contracts["age_bin_edges_seconds.npy"][
        "sha256"
    ]
    _write_canonical_manifest_for_test(root, document)

    # foundation 必須先成功，確保失敗不是 manifest 格式或 checksum stage 提前攔截。
    _validate_release_topology_and_checksums(root, require_final_name=True)
    report = validate_aggregate_release(root)
    _assert_validation_failure(report, expected_stage="products", path_marker=tmp_path)


def test_validate_shard_output_path_progress_mismatch_is_decoder_failure(
    valid_source_release: _SourceReleaseFixture,
    tmp_path: Path,
) -> None:
    """只改合法 shard output path 時，validator 應在 decoder 綁定拒絕。

    測試使用固定 Arrow schema 與 production 相同的 Parquet writer 參數重寫複製檔，
    並同步 release manifest 的實體 contract。如此 topology、checksum 與 products 都仍
    合法，真正待驗證的是 decoded ``output_relative_path`` 能否綁回內嵌 source progress。

    Detached release 不保存原始 source run 的可開啟路徑，因此 reader 只能保存
    ``trajectory_manifest_sha256``，不能重新判定該摘要所指外部 manifest 的真實性；摘要
    與原始 shard output 的核對屬於 writer publish-time gate。這項責任界線只驗證工程
    provenance，不代表正式 OCM／NWW 軌跡或科學結果。
    """

    root = _copy_release(valid_source_release, tmp_path)
    file_name = "shard_bindings.parquet"
    table_path = root / file_name
    schema = TABLE_SCHEMAS[file_name]
    original_table = pq.read_table(table_path)
    assert original_table.schema.equals(schema, check_metadata=True)
    assert original_table.num_rows == 1

    original_rows = original_table.to_pylist()
    original_path = original_rows[0]["output_relative_path"]
    replacement_path = "shards/other-shard"
    assert type(original_path) is str
    assert replacement_path != original_path

    path_field = schema.field("output_relative_path")
    path_column = pa.array([replacement_path], type=path_field.type)
    rewritten_table = original_table.set_column(
        schema.get_field_index(path_field.name),
        path_field,
        path_column,
    )
    assert rewritten_table.schema.equals(schema, check_metadata=True)
    rewritten_rows = rewritten_table.to_pylist()
    for field_name in schema.names:
        if field_name == "output_relative_path":
            assert rewritten_rows[0][field_name] == replacement_path
        else:
            assert rewritten_rows[0][field_name] == original_rows[0][field_name]

    rewritten_path = tmp_path / "rewritten-shard-bindings.parquet"
    with rewritten_path.open("xb") as stream:
        pq.write_table(
            rewritten_table,
            stream,
            compression="zstd",
            use_dictionary=False,
            write_statistics=True,
        )
    rewritten_path.replace(table_path)

    document = _read_manifest(root)
    files = document["files"]
    assert type(files) is dict
    contract = files[file_name]
    assert type(contract) is dict
    contract["size_bytes"] = table_path.stat().st_size
    contract["sha256"] = sha256(table_path.read_bytes()).hexdigest()
    contract["row_count"] = rewritten_table.num_rows
    assert contract["fields"] == valid_source_release.contracts[file_name]["fields"]
    _write_canonical_manifest_for_test(root, document)

    report = validate_aggregate_release(root)
    _assert_validation_failure(report, expected_stage="decoder", path_marker=tmp_path)
    _assert_read_has_private_fixed_failure(root, path_marker=tmp_path)


def test_validate_running_source_progress_is_source_failure(
    valid_source_release: _SourceReleaseFixture,
    tmp_path: Path,
) -> None:
    """checksum 與 metadata 同步的合法 RUNNING progress 仍不可發布 aggregate。

    測試只把完成 run 的生命週期改成 schema 允許的執行中狀態，並同步 source file
    contract 與 release metadata digest，確保失敗來自 COMPLETE 發布條件，而不是提早
    停在 checksum。此 synthetic 狀態轉換僅驗證工程管線，不是正式 OCM／NWW 成果。
    """

    root = _copy_release(valid_source_release, tmp_path)
    file_name = "source_run_progress.json"
    progress_path = root / file_name
    progress = json.loads(progress_path.read_bytes().decode("utf-8"))
    assert type(progress) is dict
    shards = progress["shards"]
    assert type(shards) is dict and len(shards) == 1
    shard = next(iter(shards.values()))
    assert type(shard) is dict
    shard["lifecycle"] = "RUNNING"
    shard["output_relative_path"] = None
    shard["failure_relative_path"] = None
    shard["error_code"] = None
    shard["attempt_count"] = max(1, int(shard["attempt_count"]))
    progress["run_lifecycle"] = "RUNNING"
    changed_bytes = _release_json_bytes(progress)
    progress_path.write_bytes(changed_bytes)
    changed_digest = sha256(changed_bytes).hexdigest()

    document = _read_manifest(root)
    files = document["files"]
    metadata = document["metadata"]
    assert type(files) is dict
    assert type(metadata) is dict
    contract = files[file_name]
    assert type(contract) is dict
    contract["size_bytes"] = len(changed_bytes)
    contract["sha256"] = changed_digest
    metadata["source_run_progress_sha256"] = changed_digest
    _write_canonical_manifest_for_test(root, document)

    # 先證明全部檔案 checksum 已通過，公開結果才可被解讀為 source semantic gate。
    verified_metadata, verified_contracts, verified_sources = (
        _validate_release_topology_and_checksums(root, require_final_name=True)
    )
    assert verified_metadata.source_run_progress_sha256 == changed_digest
    assert verified_contracts[file_name]["sha256"] == changed_digest
    assert verified_sources[file_name] == changed_bytes

    report = validate_aggregate_release(root)
    _assert_validation_failure(report, expected_stage="source", path_marker=tmp_path)
    _assert_read_has_private_fixed_failure(root, path_marker=tmp_path)
