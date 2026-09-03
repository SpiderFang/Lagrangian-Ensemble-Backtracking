"""Aggregate release manifest 基礎層的合成拓撲與 checksum 契約測試。

本模組所有資料均為小型合成工程 fixture：九張 Parquet、十八個 NumPy 陣列來自既有
payload encoder，五份來源 JSON 只保存任意 strict object。它們不讀取原始 OCM／NWW，
不執行粒子平流或來源歸因，也不是任何科學結果；測試只驗證固定檔案拓撲、canonical
manifest、實體契約、checksum 順序與不洩漏暫存路徑的 fail-closed 邊界。
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import pytest
import test_aggregate_release_payload as payload_fixture

import lagrangian_backtracking.aggregate_release as release_module
from lagrangian_backtracking.aggregate_release import (
    _ReleaseValidationError,
    _validate_release_topology_and_checksums,
    _write_encoded_products,
)
from lagrangian_backtracking.aggregate_release_codec import (
    AggregateReleaseMetadata,
    encode_aggregate_release_payload,
    metadata_from_payload,
)
from lagrangian_backtracking.aggregate_release_payload import AggregateReleasePayload

_SOURCE_FILE_ORDER = (
    "aggregate_spec.json",
    "source_run_plan.json",
    "source_run_progress.json",
    "source_normalized_config.json",
    "source_input_inventory.json",
)
"""合成來源文件的固定名稱；內容只用於工程 checksum，不代表真實 source run。"""


@dataclass(frozen=True, slots=True)
class _SyntheticRelease:
    """保存一份合法合成 release 及其預期驗證結果。

    ``contracts`` 與 ``source_bytes`` 是建立 fixture 時的獨立普通容器，供 success 測試
    exact 比對；後續 tamper 測試一律深拷貝 manifest 或修改自己的 ``tmp_path``，不會
    把合成數值誤當成 OCM／NWW 科學產品。
    """

    root: Path
    metadata: AggregateReleaseMetadata
    contracts: dict[str, dict[str, object]]
    source_bytes: dict[str, bytes]
    manifest_document: dict[str, object]


def _canonical_json_bytes(document: object) -> bytes:
    """以 release schema 的 compact、排序鍵名與單一換行建立 canonical JSON bytes。"""

    return (
        json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _source_documents() -> dict[str, dict[str, object]]:
    """建立五份彼此可辨識的 strict JSON object 合成來源快照。

    欄位只用來證明 validator 回傳 exact bytes；內容刻意不模仿 run plan、progress、
    AggregateSpec 或 forcing manifest schema，避免本 foundation slice 重複上層語意驗證。
    """

    return {
        "aggregate_spec.json": {"fixture": "aggregate-spec", "revision": 1},
        "source_run_plan.json": {"fixture": "source-plan", "shards": ["synthetic"]},
        "source_run_progress.json": {"fixture": "source-progress", "complete": False},
        "source_normalized_config.json": {"fixture": "normalized-config", "dt_seconds": 1.0},
        "source_input_inventory.json": {"fixture": "input-inventory", "products": []},
    }


@pytest.fixture
def valid_release(tmp_path: Path) -> _SyntheticRelease:
    """以既有 payload encoder 與實體 writer 建立合法 33 檔 release。

    payload 是單站 synthetic fixture，包含空 cross-site 表、公尺制格網與邊界、秒制
    age 軸及非負計數；這些值只用來取得 production writer 的真實 Parquet／NPY contracts，
    不表示絕對來源機率、因果歸因或觀測驗證。
    """

    payload = AggregateReleasePayload(**payload_fixture._valid_payload_kwargs())
    metadata = metadata_from_payload(payload)
    products = encode_aggregate_release_payload(payload)
    root = tmp_path / f"{metadata.run_id}.aggregate-v1"
    root.mkdir()

    product_contracts = _write_encoded_products(root, products)
    source_bytes: dict[str, bytes] = {}
    source_contracts: dict[str, dict[str, object]] = {}
    documents = _source_documents()
    for file_name in _SOURCE_FILE_ORDER:
        raw_bytes = _canonical_json_bytes(documents[file_name])
        (root / file_name).write_bytes(raw_bytes)
        source_bytes[file_name] = raw_bytes
        source_contracts[file_name] = {
            "kind": "json",
            "size_bytes": len(raw_bytes),
            "sha256": sha256(raw_bytes).hexdigest(),
        }

    contracts = {
        **source_contracts,
        **{file_name: dict(contract) for file_name, contract in product_contracts.items()},
    }
    manifest_document: dict[str, object] = {
        "schema_version": metadata.schema_version,
        "metadata": metadata.to_dict(),
        "files": deepcopy(contracts),
    }
    (root / "aggregate_manifest.json").write_bytes(
        _canonical_json_bytes(manifest_document)
    )
    return _SyntheticRelease(
        root=root,
        metadata=metadata,
        contracts=contracts,
        source_bytes=source_bytes,
        manifest_document=manifest_document,
    )


def _write_manifest(release: _SyntheticRelease, document: object) -> None:
    """以 canonical bytes 改寫單一測試的 manifest，不觸碰其他固定產品。"""

    (release.root / "aggregate_manifest.json").write_bytes(
        _canonical_json_bytes(document)
    )


def _assert_rejected(
    directory: Path,
    tmp_path: Path,
    *,
    stage: str,
    require_final_name: bool = True,
) -> _ReleaseValidationError:
    """斷言 foundation 以固定 stage 拒絕，且頂層文字不洩漏暫存絕對路徑。"""

    with pytest.raises(_ReleaseValidationError) as error_info:
        _validate_release_topology_and_checksums(
            directory,
            require_final_name=require_final_name,
        )
    error = error_info.value
    assert error.stage == stage
    assert str(error) == stage
    assert str(tmp_path) not in str(error)
    return error


def test_valid_release_returns_exact_metadata_contracts_and_source_bytes(
    valid_release: _SyntheticRelease,
) -> None:
    """合法 33 檔 release 應回傳 exact metadata、contracts 與五份來源 bytes。"""

    metadata, contracts, source_bytes = _validate_release_topology_and_checksums(
        valid_release.root,
        require_final_name=True,
    )

    assert metadata == valid_release.metadata
    assert contracts == valid_release.contracts
    assert source_bytes == valid_release.source_bytes
    assert tuple(source_bytes) == _SOURCE_FILE_ORDER
    assert all(type(raw_bytes) is bytes for raw_bytes in source_bytes.values())


def test_topology_rejects_missing_root(tmp_path: Path) -> None:
    """不存在的 release root 必須在讀取 manifest 前以 topology 拒絕。"""

    _assert_rejected(tmp_path / "missing.aggregate-v1", tmp_path, stage="topology")


def test_topology_rejects_regular_file_root(tmp_path: Path) -> None:
    """release root 若是普通檔而非目錄，不得嘗試解讀其中內容。"""

    root = tmp_path / "file.aggregate-v1"
    root.write_bytes(b"synthetic fixture\n")
    _assert_rejected(root, tmp_path, stage="topology")


def test_topology_rejects_root_symlink_and_broken_symlink(
    valid_release: _SyntheticRelease,
    tmp_path: Path,
) -> None:
    """指向合法目錄或不存在目標的 root symbolic link 都必須 fail closed。"""

    linked_root = tmp_path / "linked.aggregate-v1"
    linked_root.symlink_to(valid_release.root, target_is_directory=True)
    _assert_rejected(linked_root, tmp_path, stage="topology")

    broken_root = tmp_path / "broken.aggregate-v1"
    broken_root.symlink_to(tmp_path / "missing-target", target_is_directory=True)
    _assert_rejected(broken_root, tmp_path, stage="topology")


@pytest.mark.parametrize("broken", [False, True], ids=["symlink", "broken-symlink"])
def test_topology_rejects_target_symlink(
    valid_release: _SyntheticRelease,
    tmp_path: Path,
    broken: bool,
) -> None:
    """固定 source 節點不可用 symbolic link 轉向其他檔案或不存在位置。"""

    target = valid_release.root / "aggregate_spec.json"
    target.unlink()
    if broken:
        target.symlink_to(tmp_path / "missing-source.json")
    else:
        alternate = tmp_path / "alternate-source.json"
        alternate.write_bytes(b"{}\n")
        target.symlink_to(alternate)
    _assert_rejected(valid_release.root, tmp_path, stage="topology")


def test_topology_rejects_missing_and_extra_file(
    valid_release: _SyntheticRelease,
    tmp_path: Path,
) -> None:
    """固定 33 檔集合少一檔或多一檔都不得進入 manifest parser。"""

    missing_target = valid_release.root / "aggregate_spec.json"
    missing_bytes = missing_target.read_bytes()
    missing_target.unlink()
    _assert_rejected(valid_release.root, tmp_path, stage="topology")

    missing_target.write_bytes(missing_bytes)
    (valid_release.root / "unexpected.bin").write_bytes(b"extra")
    _assert_rejected(valid_release.root, tmp_path, stage="topology")


def test_topology_rejects_fixed_target_directory(
    valid_release: _SyntheticRelease,
    tmp_path: Path,
) -> None:
    """名稱正確但節點型別為目錄時，仍須以 topology 拒絕。"""

    target = valid_release.root / "aggregate_spec.json"
    target.unlink()
    target.mkdir()
    _assert_rejected(valid_release.root, tmp_path, stage="topology")


def test_manifest_rejects_duplicate_key(
    valid_release: _SyntheticRelease,
    tmp_path: Path,
) -> None:
    """manifest 任一 duplicate key 不得由 JSON parser 採最後值覆蓋。"""

    (valid_release.root / "aggregate_manifest.json").write_bytes(
        b'{"schema_version":"1.0.0","schema_version":"1.0.0"}\n'
    )
    _assert_rejected(valid_release.root, tmp_path, stage="manifest")


def test_manifest_rejects_pretty_noncanonical_json(
    valid_release: _SyntheticRelease,
    tmp_path: Path,
) -> None:
    """語意相同但含縮排與空白的 manifest 仍不是唯一 canonical bytes。"""

    pretty = (
        json.dumps(
            valid_release.manifest_document,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    (valid_release.root / "aggregate_manifest.json").write_bytes(pretty)
    _assert_rejected(valid_release.root, tmp_path, stage="manifest")


@pytest.mark.parametrize("token", [b"NaN", b"Infinity", b"1e999"])
def test_manifest_rejects_nonfinite_json_numbers(
    valid_release: _SyntheticRelease,
    tmp_path: Path,
    token: bytes,
) -> None:
    """非標準常數與解析後溢位的 JSON number 一律在 manifest stage 拒絕。"""

    raw_bytes = b'{"probe":' + token + b"}\n"
    (valid_release.root / "aggregate_manifest.json").write_bytes(raw_bytes)
    _assert_rejected(valid_release.root, tmp_path, stage="manifest")


@pytest.mark.parametrize(
    ("section", "operation"),
    [
        ("root", "unknown"),
        ("root", "missing"),
        ("metadata", "unknown"),
        ("metadata", "missing"),
        ("files", "unknown"),
        ("files", "missing"),
    ],
)
def test_manifest_requires_exact_root_metadata_and_file_keys(
    valid_release: _SyntheticRelease,
    tmp_path: Path,
    section: str,
    operation: str,
) -> None:
    """root、metadata 與固定 files 集合皆拒絕未知或缺少的 key。"""

    document = deepcopy(valid_release.manifest_document)
    if section == "root":
        if operation == "unknown":
            document["unknown"] = None
        else:
            del document["schema_version"]
    elif section == "metadata":
        metadata = document["metadata"]
        assert type(metadata) is dict
        if operation == "unknown":
            metadata["unknown"] = None
        else:
            del metadata["run_id"]
    else:
        files = document["files"]
        assert type(files) is dict
        if operation == "unknown":
            files["unexpected.json"] = {
                "kind": "json",
                "size_bytes": 0,
                "sha256": "0" * 64,
            }
        else:
            del files["aggregate_spec.json"]
    _write_manifest(valid_release, document)
    _assert_rejected(valid_release.root, tmp_path, stage="manifest")


def test_manifest_rejects_wrong_file_kind(
    valid_release: _SyntheticRelease,
    tmp_path: Path,
) -> None:
    """固定 JSON 檔不可在 contract 中冒充 Parquet 或其他 kind。"""

    document = deepcopy(valid_release.manifest_document)
    files = document["files"]
    assert type(files) is dict
    contract = files["aggregate_spec.json"]
    assert type(contract) is dict
    contract["kind"] = "parquet"
    _write_manifest(valid_release, document)
    _assert_rejected(valid_release.root, tmp_path, stage="manifest")


@pytest.mark.parametrize(
    "count_location",
    ["metadata", "size", "row", "shape", "element"],
)
def test_manifest_rejects_bool_as_count(
    valid_release: _SyntheticRelease,
    tmp_path: Path,
    count_location: str,
) -> None:
    """metadata、大小、列數、shape 與元素數都不可讓 bool 假冒原生整數。"""

    document = deepcopy(valid_release.manifest_document)
    metadata = document["metadata"]
    files = document["files"]
    assert type(metadata) is dict
    assert type(files) is dict
    if count_location == "metadata":
        metadata["input_particle_count"] = True
    elif count_location == "size":
        files["aggregate_spec.json"]["size_bytes"] = True
    elif count_location == "row":
        files["shard_bindings.parquet"]["row_count"] = True
    elif count_location == "shape":
        files["age_bin_edges_seconds.npy"]["shape"] = [True]
    else:
        files["age_bin_edges_seconds.npy"]["element_count"] = True
    _write_manifest(valid_release, document)
    _assert_rejected(valid_release.root, tmp_path, stage="manifest")


@pytest.mark.parametrize("tamper", ["order", "type", "nullability"])
def test_manifest_rejects_parquet_field_contract_tamper(
    valid_release: _SyntheticRelease,
    tmp_path: Path,
    tamper: str,
) -> None:
    """Parquet 欄位順序、Arrow 型別與 nullability 均是 exact manifest 契約。"""

    document = deepcopy(valid_release.manifest_document)
    files = document["files"]
    assert type(files) is dict
    fields = files["shard_bindings.parquet"]["fields"]
    assert type(fields) is list
    if tamper == "order":
        fields[0], fields[1] = fields[1], fields[0]
    elif tamper == "type":
        fields[0]["type"] = "string"
    else:
        fields[0]["nullable"] = True
    _write_manifest(valid_release, document)
    _assert_rejected(valid_release.root, tmp_path, stage="manifest")


@pytest.mark.parametrize("tamper", ["dtype", "shape", "element"])
def test_manifest_rejects_npy_contract_tamper(
    valid_release: _SyntheticRelease,
    tmp_path: Path,
    tamper: str,
) -> None:
    """NPY 的 little-endian dtype、一維 shape 與 element count 必須完全一致。"""

    document = deepcopy(valid_release.manifest_document)
    files = document["files"]
    assert type(files) is dict
    contract = files["age_bin_edges_seconds.npy"]
    assert type(contract) is dict
    if tamper == "dtype":
        contract["dtype"] = "<i8"
    elif tamper == "shape":
        shape = contract["shape"]
        assert type(shape) is list and type(shape[0]) is int
        contract["shape"] = [shape[0] + 1]
    else:
        element_count = contract["element_count"]
        assert type(element_count) is int
        contract["element_count"] = element_count + 1
    _write_manifest(valid_release, document)
    _assert_rejected(valid_release.root, tmp_path, stage="manifest")


def test_final_basename_mismatch_can_only_bypass_name_gate(
    valid_release: _SyntheticRelease,
    tmp_path: Path,
) -> None:
    """錯誤 basename 在 final 模式失敗，但 partial 模式仍須通過其餘完整契約。"""

    renamed = tmp_path / "writer-partial-directory"
    valid_release.root.rename(renamed)
    _assert_rejected(renamed, tmp_path, stage="name", require_final_name=True)

    metadata, contracts, source_bytes = _validate_release_topology_and_checksums(
        renamed,
        require_final_name=False,
    )
    assert metadata == valid_release.metadata
    assert contracts == valid_release.contracts
    assert source_bytes == valid_release.source_bytes


@pytest.mark.parametrize("tamper", ["size", "sha256"])
def test_checksum_rejects_declared_size_or_hash_tamper(
    valid_release: _SyntheticRelease,
    tmp_path: Path,
    tamper: str,
) -> None:
    """語法合法但與實體檔不一致的 size 或 SHA-256 必須落在 checksum stage。"""

    document = deepcopy(valid_release.manifest_document)
    files = document["files"]
    assert type(files) is dict
    contract = files["aggregate_spec.json"]
    assert type(contract) is dict
    if tamper == "size":
        size_bytes = contract["size_bytes"]
        assert type(size_bytes) is int
        contract["size_bytes"] = size_bytes + 1
    else:
        digest = contract["sha256"]
        assert type(digest) is str
        contract["sha256"] = ("0" if digest[0] != "0" else "1") + digest[1:]
    _write_manifest(valid_release, document)
    _assert_rejected(valid_release.root, tmp_path, stage="checksum")


def test_source_json_is_not_read_until_every_checksum_passes(
    valid_release: _SyntheticRelease,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """較後 NPY 失敗時只能讀 manifest，不得提前讀取五份 source JSON。

    測試破壞固定陣列順序最後一份 NPY，使前面五 JSON、九 Parquet 與十七份 NPY 的
    checksum 都已通過後才失敗。byte reader 的紀錄可區分 manifest 解析與 checksum
    串流；五份來源文件若在全體 checksum 完成前被開啟，測試會直接失敗。
    """

    late_array = valid_release.root / "source_receptor_travel_age_histogram.npy"
    late_array.write_bytes(late_array.read_bytes() + b"tamper")
    real_reader = release_module._read_regular_file_bytes
    byte_reader_calls: list[str] = []

    def recording_reader(path: Path) -> bytes:
        """記錄安全 byte reader 的固定 basename，再委派 production 實作。"""

        byte_reader_calls.append(path.name)
        return real_reader(path)

    monkeypatch.setattr(release_module, "_read_regular_file_bytes", recording_reader)
    _assert_rejected(valid_release.root, tmp_path, stage="checksum")

    assert byte_reader_calls == ["aggregate_manifest.json"]
    assert not set(_SOURCE_FILE_ORDER).intersection(byte_reader_calls)
