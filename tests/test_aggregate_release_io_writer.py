"""Aggregate release writer 的 27 檔實體輸出與封閉式失敗契約測試。

本檔沿用既有單站與兩站 synthetic payload fixture，驗證內部
``_write_encoded_products`` 只把 encoder 已建立的九張 Parquet 表與十八個 NumPy
陣列安全落檔。公尺制邊界、秒制 age／停留時間、非負計數、缺值與零列拓撲均不得在
writer 階段被轉型、補零或重排；輸出仍只是條件式來源足跡或相對來源權重的資料載體，
不代表絕對來源機率、因果歸因或觀測驗證結果。
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pyarrow.parquet as pq
import pytest
import test_aggregate_release_payload as payload_fixture

from lagrangian_backtracking.aggregate_release import (
    TABLE_SCHEMAS,
    _write_encoded_products,
)
from lagrangian_backtracking.aggregate_release_codec import encode_aggregate_release_payload
from lagrangian_backtracking.aggregate_release_layout import (
    AGGREGATE_RELEASE_ARRAY_FILES,
    AGGREGATE_RELEASE_TABLE_FILES,
    EncodedAggregateProducts,
)
from lagrangian_backtracking.aggregate_release_payload import AggregateReleasePayload

# writer 的實體範圍固定為九表加十八陣列；測試不引用 production 私有檔序常數，避免
# 只複製實作順序而漏掉固定集合本身的缺檔或額外檔案問題。
_PRODUCT_FILE_NAMES = AGGREGATE_RELEASE_TABLE_FILES | AGGREGATE_RELEASE_ARRAY_FILES


def _encoded_fixture(fixture_name: str) -> EncodedAggregateProducts:
    """由既有 payload fixture 取得單站或兩站的 canonical encoder 產品。

    單站包含零列 cross-site Parquet 與三條公尺制 boundary；兩站另含 A→B、B→A
    ordered zero topology 與多站串接陣列。helper 只呼叫公開 payload constructor 與
    encoder，不在測試端重做 row 排序、offset 或科學計數。
    """

    if fixture_name == "single":
        payload = AggregateReleasePayload(**payload_fixture._valid_payload_kwargs())
    elif fixture_name == "two":
        payload = AggregateReleasePayload(**payload_fixture._two_site_payload_kwargs())
    else:
        raise AssertionError(f"未知 fixture：{fixture_name}")
    return encode_aggregate_release_payload(payload)


def _rebuild_tampered_products(
    products: EncodedAggregateProducts,
    *,
    table_file: str | None = None,
    table_rows: tuple[dict[str, object], ...] | None = None,
    array_file: str | None = None,
    array_value: np.ndarray | None = None,
) -> EncodedAggregateProducts:
    """以公開容器重建產品，再保留 writer 必須自行拒絕的指定單點竄改。

    table row 可直接交給公開 ``EncodedAggregateProducts`` 建立新的唯讀 snapshot；陣列
    的錯誤 dtype、負值或非有限值會被該容器 canonical 或提前拒絕，因此先以公開
    constructor 建立完整合法副本，再用 ``object.__setattr__`` 模擬 frozen products
    遭低階竄改。writer 宣告會重新驗證 caller products，故仍必須在任何寫檔前拒絕。
    """

    if (table_file is None) != (table_rows is None):
        raise AssertionError("table_file 與 table_rows 必須成對提供")
    if (array_file is None) != (array_value is None):
        raise AssertionError("array_file 與 array_value 必須成對提供")

    tables = {
        file_name: tuple(dict(row) for row in rows)
        for file_name, rows in products.tables.items()
    }
    arrays = {
        file_name: np.array(values, dtype=values.dtype, order="C", copy=True)
        for file_name, values in products.arrays.items()
    }
    if table_file is not None and table_rows is not None:
        tables[table_file] = tuple(dict(row) for row in table_rows)

    rebuilt = EncodedAggregateProducts(tables=tables, arrays=arrays)
    if array_file is not None and array_value is not None:
        tampered_arrays = dict(rebuilt.arrays)
        tampered_arrays[array_file] = np.array(
            array_value,
            dtype=array_value.dtype,
            order="C",
            copy=True,
        )
        object.__setattr__(rebuilt, "arrays", MappingProxyType(tampered_arrays))
    return rebuilt


def _file_sha256(path: Path) -> str:
    """獨立計算小型測試輸出檔 SHA-256，核對 writer contract 而不重用其 helper。"""

    return sha256(path.read_bytes()).hexdigest()


def _expected_field_contracts(file_name: str) -> list[dict[str, object]]:
    """由公開 Arrow schema 建立 contract 應保存的有序普通 dict 清單。"""

    return [
        {
            "name": field.name,
            "type": str(field.type),
            "nullable": field.nullable,
        }
        for field in TABLE_SCHEMAS[file_name]
    ]


@pytest.mark.parametrize("fixture_name", ["single", "two"])
def test_writer_emits_27_files_and_exact_contracts(
    tmp_path: Path,
    fixture_name: str,
) -> None:
    """單站與兩站皆須建立 27 檔及 27 份含 size、hash、kind 的 exact contracts。

    Parquet contract 的 row count 必須等於 encoder 列數，``fields`` 必須是依 schema
    排序的普通 ``list[dict]``；NumPy contract 則保存 little-endian dtype、shape list
    與 element count。公尺、秒與計數內容不由 contract 推論，只描述實體檔案契約。
    """

    products = _encoded_fixture(fixture_name)
    output_directory = tmp_path / f"writer-{fixture_name}"
    output_directory.mkdir()

    contracts = _write_encoded_products(output_directory, products)

    assert type(contracts) is dict
    assert len(contracts) == 27
    assert frozenset(contracts) == _PRODUCT_FILE_NAMES
    assert {path.name for path in output_directory.iterdir()} == set(_PRODUCT_FILE_NAMES)

    for file_name in sorted(_PRODUCT_FILE_NAMES):
        target = output_directory / file_name
        contract = contracts[file_name]
        assert type(contract) is dict
        assert type(contract["size_bytes"]) is int
        assert contract["size_bytes"] == target.stat().st_size
        assert contract["size_bytes"] > 0
        assert contract["sha256"] == _file_sha256(target)

        if file_name in AGGREGATE_RELEASE_TABLE_FILES:
            assert set(contract) == {
                "kind",
                "size_bytes",
                "sha256",
                "row_count",
                "fields",
            }
            assert contract["kind"] == "parquet"
            assert contract["row_count"] == len(products.tables[file_name])
            assert type(contract["fields"]) is list
            assert all(type(field) is dict for field in contract["fields"])
            assert contract["fields"] == _expected_field_contracts(file_name)
        else:
            values = products.arrays[file_name]
            expected_dtype = "<f8" if values.dtype.kind == "f" else "<i8"
            assert set(contract) == {
                "kind",
                "size_bytes",
                "sha256",
                "dtype",
                "shape",
                "element_count",
            }
            assert contract["kind"] == "npy"
            assert contract["dtype"] == expected_dtype
            assert type(contract["shape"]) is list
            assert contract["shape"] == list(values.shape)
            assert contract["element_count"] == values.size


def test_writer_empty_cross_parquet_retains_three_columns(tmp_path: Path) -> None:
    """單站 cross-site Parquet 必須是零列，但保留三個 non-nullable 欄位。

    零列表示沒有 ordered site pair，並非 schema 或資料遺失；writer 不得輸出無欄表，
    也不能捏造 A→B 列。磁碟 schema 與 contract fields 都必須精確描述來源站、目標站
    及不重複成員計數三欄。
    """

    products = _encoded_fixture("single")
    output_directory = tmp_path / "empty-cross"
    output_directory.mkdir()
    contracts = _write_encoded_products(output_directory, products)
    file_name = "cross_site_counts.parquet"
    table = pq.read_table(output_directory / file_name)

    assert table.num_rows == 0
    assert table.num_columns == 3
    assert table.to_pylist() == []
    assert tuple(table.column_names) == (
        "source_study_site_id",
        "target_study_site_id",
        "unique_member_count",
    )
    assert table.schema.equals(TABLE_SCHEMAS[file_name], check_metadata=True)
    assert contracts[file_name]["row_count"] == 0
    assert contracts[file_name]["fields"] == _expected_field_contracts(file_name)


@pytest.mark.parametrize("fixture_name", ["single", "two"])
def test_writer_disk_rows_and_arrays_equal_encoder(
    tmp_path: Path,
    fixture_name: str,
) -> None:
    """寫檔後讀回每張 Parquet 與 NPY，逐欄、逐值核對 encoder snapshot。

    Parquet ``to_pylist`` 必須回傳普通 ``list[dict]``，欄序、原生 scalar 型別、None
    與值均不可改變；NPY 必須維持一維 C-order 的公尺、秒或計數序列，不得 reshape、
    cast、clip 或補值。
    """

    products = _encoded_fixture(fixture_name)
    output_directory = tmp_path / f"round-trip-{fixture_name}"
    output_directory.mkdir()
    _write_encoded_products(output_directory, products)

    for file_name in sorted(AGGREGATE_RELEASE_TABLE_FILES):
        actual_rows = pq.read_table(output_directory / file_name).to_pylist()
        expected_rows = [dict(row) for row in products.tables[file_name]]
        assert type(actual_rows) is list
        assert all(type(row) is dict for row in actual_rows)
        assert len(actual_rows) == len(expected_rows)
        for actual_row, expected_row in zip(actual_rows, expected_rows, strict=True):
            assert tuple(actual_row) == tuple(expected_row)
            for field_name, expected_value in expected_row.items():
                actual_value = actual_row[field_name]
                assert type(actual_value) is type(expected_value)
                if type(expected_value) is float:
                    assert np.float64(actual_value).tobytes() == np.float64(expected_value).tobytes()
                else:
                    assert actual_value == expected_value

    for file_name in sorted(AGGREGATE_RELEASE_ARRAY_FILES):
        expected = products.arrays[file_name]
        actual = np.load(output_directory / file_name, allow_pickle=False)
        assert actual.ndim == 1
        assert actual.flags.c_contiguous
        assert actual.shape == expected.shape
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("directory_kind", ["symlink", "regular-file"])
def test_writer_rejects_symlink_or_non_directory(
    tmp_path: Path,
    directory_kind: str,
) -> None:
    """writer 入口若是目錄 symlink 或普通檔案，必須在建立產品前拒絕。

    release 目標必須是 caller 已建立的普通目錄；拒絕 symlink 可避免路徑指向核准範圍
    之外，拒絕非目錄則避免把錯誤路徑當作可建立的 release。兩種情況都不得留下任何
    Parquet 或 NPY 檔案。
    """

    products = _encoded_fixture("single")
    if directory_kind == "symlink":
        real_directory = tmp_path / "real-directory"
        real_directory.mkdir()
        output_path = tmp_path / "directory-link"
        output_path.symlink_to(real_directory, target_is_directory=True)
    else:
        output_path = tmp_path / "not-a-directory"
        output_path.write_bytes(b"preserve")

    with pytest.raises(ValueError, match="directory"):
        _write_encoded_products(output_path, products)

    if directory_kind == "symlink":
        assert list(real_directory.iterdir()) == []
    else:
        assert output_path.read_bytes() == b"preserve"


@pytest.mark.parametrize("conflict_kind", ["existing", "broken-symlink"])
def test_writer_prescans_all_targets_without_partial_writes(
    tmp_path: Path,
    conflict_kind: str,
) -> None:
    """末一個固定目標已存在或為 broken symlink 時，其他 26 檔皆不得建立。

    衝突刻意放在 source-receptor travel-age 陣列，亦即固定寫入順序的末端；如此可證明
    writer 在開始建立任何 Parquet／NPY 前已掃描全部目標。broken symlink 的
    ``exists()`` 為 false，但仍是不可覆寫的路徑占用，必須和普通既有檔案同樣拒絕。
    """

    products = _encoded_fixture("single")
    output_directory = tmp_path / f"conflict-{conflict_kind}"
    output_directory.mkdir()
    conflict_name = "source_receptor_travel_age_histogram.npy"
    conflict_path = output_directory / conflict_name
    if conflict_kind == "existing":
        conflict_path.write_bytes(b"preserve-existing-target")
    else:
        conflict_path.symlink_to(tmp_path / "missing-target")
        assert not conflict_path.exists()
        assert conflict_path.is_symlink()

    with pytest.raises(FileExistsError, match=conflict_name):
        _write_encoded_products(output_directory, products)

    for file_name in _PRODUCT_FILE_NAMES - {conflict_name}:
        target = output_directory / file_name
        assert not target.exists()
        assert not target.is_symlink()
    if conflict_kind == "existing":
        assert conflict_path.read_bytes() == b"preserve-existing-target"
    else:
        assert conflict_path.is_symlink()
        assert not conflict_path.exists()


@pytest.mark.parametrize(
    ("tamper_case", "expected_message"),
    [
        pytest.param("numpy-int64", "site_index.*原生 int", id="numpy-int64"),
        pytest.param("numpy-float64", "x_min_m.*有限原生 float", id="numpy-float64"),
        pytest.param("bool", "x_cell_count.*原生 int", id="bool"),
        pytest.param("none", "study_site_id.*non-nullable", id="none-nonnullable"),
        pytest.param("column-order", "欄位名稱與順序", id="column-order"),
    ],
)
def test_writer_rejects_table_scalar_and_column_order_tamper(
    tmp_path: Path,
    tamper_case: str,
    expected_message: str,
) -> None:
    """表格拒絕 NumPy scalar、bool、non-nullable None 與欄位順序竄改。

    Arrow 本身可能把這些值寬鬆 cast 成目標 schema；writer 必須先要求原生 Python
    scalar 與 exact iteration order，避免不同 adapter 對相同 release 產生不同 bytes。
    Scenario 的 nullable ``initial_*`` 政策不會放寬 site index 的必填欄位。
    """

    products = _encoded_fixture("single")
    file_name = "site_index.parquet"
    rows = [dict(row) for row in products.tables[file_name]]
    row = rows[0]
    if tamper_case == "numpy-int64":
        row["site_index"] = np.int64(row["site_index"])
    elif tamper_case == "numpy-float64":
        row["x_min_m"] = np.float64(row["x_min_m"])
    elif tamper_case == "bool":
        row["x_cell_count"] = True
    elif tamper_case == "none":
        row["study_site_id"] = None
    else:
        first_value = row.pop("site_index")
        row["site_index"] = first_value
    tampered = _rebuild_tampered_products(
        products,
        table_file=file_name,
        table_rows=tuple(rows),
    )
    output_directory = tmp_path / tamper_case
    output_directory.mkdir()

    with pytest.raises(ValueError, match=expected_message):
        _write_encoded_products(output_directory, tampered)
    assert list(output_directory.iterdir()) == []


@pytest.mark.parametrize(
    ("tamper_case", "expected_message"),
    [
        pytest.param("two-dimensional", "一維", id="two-dimensional"),
        pytest.param("wrong-native-dtype", "exact native int64", id="wrong-native-dtype"),
        pytest.param("negative-int", "負", id="negative-int"),
        pytest.param("nonfinite-float", "有限", id="nonfinite-float"),
        pytest.param("negative-float", "負", id="negative-float"),
    ],
)
def test_writer_rejects_invalid_array_contract(
    tmp_path: Path,
    tamper_case: str,
    expected_message: str,
) -> None:
    """陣列拒絕二維、錯誤 native dtype、負整數、非有限值與負浮點值。

    NPY 只保存一維 C-order 的 native int64／float64，再寫成 little-endian bytes；計數
    與 offset 不得為負，公尺與秒的連續量必須有限且非負。writer 不可 reshape、cast、
    clip 或以零替代錯誤值，且所有驗證須在建立第一個檔案前完成。
    """

    products = _encoded_fixture("single")
    if tamper_case == "two-dimensional":
        file_name = "site_cell_offsets.npy"
        bad_value = np.array([[0, 2]], dtype=np.int64)
    elif tamper_case == "wrong-native-dtype":
        file_name = "site_cell_offsets.npy"
        bad_value = np.array([0.0, 2.0], dtype=np.float64)
    elif tamper_case == "negative-int":
        file_name = "local_first_exit_count.npy"
        bad_value = np.array([-1, 0], dtype=np.int64)
    elif tamper_case == "nonfinite-float":
        file_name = "boundary_bin_edges_m.npy"
        bad_value = np.array(products.arrays[file_name], copy=True)
        bad_value[1] = np.inf
    else:
        file_name = "pathway_residence_time_seconds.npy"
        bad_value = np.array(products.arrays[file_name], copy=True)
        bad_value[0] = -1.0
    tampered = _rebuild_tampered_products(
        products,
        array_file=file_name,
        array_value=bad_value,
    )
    output_directory = tmp_path / tamper_case
    output_directory.mkdir()

    with pytest.raises(ValueError, match=expected_message):
        _write_encoded_products(output_directory, tampered)
    assert list(output_directory.iterdir()) == []


def test_writer_does_not_mutate_caller_products_or_arrays(tmp_path: Path) -> None:
    """成功寫檔不得替換 caller products mapping、array 物件、dtype、旗標或內容。

    writer 的 little-endian 儲存轉換只能作用於防禦性副本；原始 encoder arrays 仍是
    native dtype、C-contiguous 且唯讀，公尺、秒與計數 bytes 必須逐一不變。表格 row
    的欄序與 scalar 也不得因 Arrow materialization 被改寫。
    """

    products = _encoded_fixture("two")
    original_tables = products.tables
    original_arrays = products.arrays
    table_snapshots = {
        file_name: tuple(dict(row) for row in rows)
        for file_name, rows in products.tables.items()
    }
    array_snapshots = {
        file_name: (
            id(values),
            values.dtype,
            values.shape,
            values.flags.c_contiguous,
            values.flags.writeable,
            values.tobytes(order="C"),
        )
        for file_name, values in products.arrays.items()
    }
    output_directory = tmp_path / "caller-isolation"
    output_directory.mkdir()

    _write_encoded_products(output_directory, products)

    assert products.tables is original_tables
    assert products.arrays is original_arrays
    for file_name, expected_rows in table_snapshots.items():
        assert tuple(dict(row) for row in products.tables[file_name]) == expected_rows
    for file_name, snapshot in array_snapshots.items():
        values = products.arrays[file_name]
        identity, dtype, shape, c_contiguous, writeable, content = snapshot
        assert id(values) == identity
        assert values.dtype == dtype
        assert values.shape == shape
        assert values.flags.c_contiguous is c_contiguous
        assert values.flags.writeable is writeable
        assert values.tobytes(order="C") == content
