"""Aggregate release reader 的實體 round-trip 與封閉式失敗契約測試。

本檔沿用既有單站與兩站 synthetic payload／encoder fixture，先由
``_write_encoded_products`` 建立合法九張 Parquet 表與十八個 NumPy 陣列，再測試
``_read_encoded_products`` 的 storage-level 邊界。公尺制邊界與座標、秒制 age／停留
時間、非負計數、可整組缺值的動態初始條件及零列拓撲都不得在 reader 階段被轉型、
補零、reshape 或猜測。synthetic 數值只供工程契約驗證，最多代表條件式來源足跡或
相對來源權重的載體，不是絕對來源機率、因果歸因或觀測驗證結果。

本測試刻意不竄改跨表 offset、join 或事件守恆；那些跨產品語意由 codec decoder
負責。本檔只驗證目錄與檔案節點安全、Arrow schema／row scalar、NPY dtype／shape／
數值、磁碟唯讀性，以及對外錯誤不洩漏暫存目錄絕對路徑。
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import test_aggregate_release_io_writer as writer_fixture

from lagrangian_backtracking.aggregate_release import (
    TABLE_SCHEMAS,
    _read_encoded_products,
    _write_encoded_products,
)
from lagrangian_backtracking.aggregate_release_layout import (
    AGGREGATE_RELEASE_ARRAY_FILES,
    AGGREGATE_RELEASE_TABLE_FILES,
    EncodedAggregateProducts,
)

# reader 固定處理九表與十八陣列；此集合只檢查 storage topology，不把五個來源 JSON、
# manifest 或跨產品科學語意混入本測試切片。
_PRODUCT_FILE_NAMES = AGGREGATE_RELEASE_TABLE_FILES | AGGREGATE_RELEASE_ARRAY_FILES


def _write_valid_fixture(
    tmp_path: Path,
    fixture_name: str,
    *,
    suffix: str = "valid",
) -> tuple[Path, EncodedAggregateProducts]:
    """沿用 writer 測試 fixture 寫出一份合法 27 檔 reader 輸入。

    單站產品包含精確空的 cross-site 表；兩站產品包含 ordered 零列與多站串接陣列。
    helper 只組裝測試目錄並呼叫既有 encoder／writer，不在測試端重作公尺、秒、計數
    或欄位排序邏輯。
    """

    products = writer_fixture._encoded_fixture(fixture_name)
    directory = tmp_path / f"{fixture_name}-{suffix}"
    directory.mkdir()
    _write_encoded_products(directory, products)
    return directory, products


def _file_sha256(path: Path) -> str:
    """獨立計算小型 fixture 檔案 SHA-256，確認 reader 沒有改寫磁碟 bytes。"""

    return sha256(path.read_bytes()).hexdigest()


def _assert_reader_rejects(
    directory: Path,
    tmp_path: Path,
    *,
    expected_fragment: str | None = None,
) -> None:
    """確認 reader 以 ValueError 拒絕，且頂層訊息不洩漏 tmp 絕對路徑。

    production 允許錯誤訊息指出固定產品檔名與違反的 dtype／schema 契約，但不得回傳
    caller 工作目錄或伺服器路徑。只檢查 user-visible 頂層例外；底層 cause 保留給本機
    除錯，不屬於 reader 對外訊息。
    """

    with pytest.raises(ValueError) as error_info:
        _read_encoded_products(directory)
    message = str(error_info.value)
    assert str(tmp_path) not in message
    if expected_fragment is not None:
        assert expected_fragment in message


def _assert_products_exact_and_read_only(
    expected: EncodedAggregateProducts,
    actual: EncodedAggregateProducts,
) -> None:
    """逐表逐陣列核對 exact 值，並驗證 reader 回傳防禦性唯讀 snapshot。

    表格欄位 iteration order、Python scalar exact type 與 float64 bytes 都是實體契約；
    陣列則核對 dtype、shape、C-order、元素與獨立記憶體。這裡不檢查跨表 join 或 offset
    守恆，以免 reader 測試越過 codec decoder 的責任邊界。
    """

    assert type(actual.tables) is MappingProxyType
    assert type(actual.arrays) is MappingProxyType
    assert frozenset(actual.tables) == AGGREGATE_RELEASE_TABLE_FILES
    assert frozenset(actual.arrays) == AGGREGATE_RELEASE_ARRAY_FILES

    for file_name in sorted(AGGREGATE_RELEASE_TABLE_FILES):
        expected_rows = expected.tables[file_name]
        actual_rows = actual.tables[file_name]
        assert type(actual_rows) is tuple
        assert len(actual_rows) == len(expected_rows)
        for expected_row, actual_row in zip(expected_rows, actual_rows, strict=True):
            assert type(actual_row) is MappingProxyType
            assert tuple(actual_row) == tuple(expected_row)
            for field_name, expected_value in expected_row.items():
                actual_value = actual_row[field_name]
                assert type(actual_value) is type(expected_value)
                if type(expected_value) is float:
                    assert np.float64(actual_value).tobytes() == np.float64(expected_value).tobytes()
                else:
                    assert actual_value == expected_value

    for file_name in sorted(AGGREGATE_RELEASE_ARRAY_FILES):
        expected_array = expected.arrays[file_name]
        actual_array = actual.arrays[file_name]
        assert actual_array is not expected_array
        assert not np.shares_memory(actual_array, expected_array)
        assert actual_array.dtype == expected_array.dtype
        assert actual_array.shape == expected_array.shape
        assert actual_array.flags.c_contiguous
        assert not actual_array.flags.writeable
        np.testing.assert_array_equal(actual_array, expected_array)

    with pytest.raises(TypeError):
        actual.tables["unexpected.parquet"] = ()
    with pytest.raises(TypeError):
        actual.tables["site_index.parquet"][0]["study_site_id"] = "mutated"
    with pytest.raises(ValueError):
        actual.arrays["site_cell_offsets.npy"][0] = 1


def _write_parquet_table(path: Path, table: pa.Table) -> None:
    """以 binary stream 覆寫單一 Parquet fixture，避免測試端產生額外副檔名。"""

    with path.open("wb") as stream:
        pq.write_table(
            table,
            stream,
            compression="zstd",
            use_dictionary=False,
            write_statistics=True,
        )


def _schema_with_replaced_field(
    schema: pa.Schema,
    field_index: int,
    field: pa.Field,
) -> pa.Schema:
    """建立只替換一欄的 Arrow schema，保留其他欄位順序與 metadata 空值。"""

    fields = list(schema)
    fields[field_index] = field
    return pa.schema(fields)


def _tampered_site_index_table(table: pa.Table, tamper_case: str) -> pa.Table:
    """建立指定的 site-index Parquet schema 或 row scalar 竄改。

    每個案例只改一個 storage-level 條件；nullable row null 因 Parquet required column
    無法合法承載 null，使用明示 nullable schema 寫入惡意列，reader 應在交給 codec 前
    即因 nullability／null 契約 fail closed。
    """

    schema = TABLE_SCHEMAS["site_index.parquet"]
    if tamper_case == "column-name":
        original = schema[0]
        changed = pa.field(
            "renamed_site_index",
            original.type,
            nullable=original.nullable,
        )
        bad_schema = _schema_with_replaced_field(schema, 0, changed)
        return pa.Table.from_arrays(table.columns, schema=bad_schema)
    if tamper_case == "column-order":
        order = (1, 0, *range(2, len(schema)))
        bad_schema = pa.schema([schema[index] for index in order])
        return pa.Table.from_arrays(
            [table.column(index) for index in order],
            schema=bad_schema,
        )
    if tamper_case == "logical-type":
        original = schema[0]
        changed = pa.field(original.name, pa.float64(), nullable=original.nullable)
        bad_schema = _schema_with_replaced_field(schema, 0, changed)
        columns = list(table.columns)
        columns[0] = pa.array(
            [float(value.as_py()) for value in table.column(0)],
            type=pa.float64(),
        )
        return pa.Table.from_arrays(columns, schema=bad_schema)
    if tamper_case in {"nullability", "nonnull-row-null"}:
        field_index = schema.get_field_index("study_site_id")
        original = schema[field_index]
        changed = pa.field(original.name, original.type, nullable=True)
        bad_schema = _schema_with_replaced_field(schema, field_index, changed)
        rows = table.to_pylist()
        if tamper_case == "nonnull-row-null":
            rows[0]["study_site_id"] = None
        return pa.Table.from_pylist(rows, schema=bad_schema)
    if tamper_case == "schema-metadata":
        return table.replace_schema_metadata({b"tamper": b"schema"})
    if tamper_case == "field-metadata":
        original = schema[0]
        changed = pa.field(
            original.name,
            original.type,
            nullable=original.nullable,
            metadata={b"tamper": b"field"},
        )
        bad_schema = _schema_with_replaced_field(schema, 0, changed)
        return pa.Table.from_arrays(table.columns, schema=bad_schema)

    rows = table.to_pylist()
    if tamper_case == "nan-float":
        rows[0]["x_min_m"] = float("nan")
    elif tamper_case == "infinite-float":
        rows[0]["x_min_m"] = float("inf")
    elif tamper_case == "blank-string":
        rows[0]["study_site_id"] = " "
    elif tamper_case == "negative-index":
        rows[0]["site_index"] = -1
    else:
        raise AssertionError(f"未知 Parquet tamper：{tamper_case}")
    return pa.Table.from_pylist(rows, schema=schema)


def _write_npy_or_archive(
    path: Path,
    values: np.ndarray,
    *,
    archive: bool = False,
) -> None:
    """以 binary stream 覆寫固定 ``.npy`` 名稱，不讓 NumPy 自動附加副檔名。

    archive 案例刻意把 NPZ container 寫入既有 NPY 路徑，驗證 reader 依載入物件型別
    拒絕 archive，而不是因測試誤產生 ``.npy.npz`` 導致只命中缺檔分支。
    """

    with path.open("wb") as stream:
        if archive:
            np.savez(stream, payload=values)
        else:
            np.save(stream, values, allow_pickle=False)


@pytest.mark.parametrize("fixture_name", ["single", "two"])
def test_reader_writer_exact_round_trip_and_read_only_products(
    tmp_path: Path,
    fixture_name: str,
) -> None:
    """單站與兩站 writer→reader 必須逐表、逐陣列 exact round-trip。

    九表保留 tuple row sequence、欄序、原生 scalar 型別與值；十八陣列保留固定
    int64／float64 dtype、一維 shape 與公尺／秒／計數元素。reader 回傳新的唯讀產品，
    caller 不可透過 mapping、row 或 ndarray 改寫其 snapshot。
    """

    directory, expected = _write_valid_fixture(tmp_path, fixture_name)

    actual = _read_encoded_products(directory)

    assert actual is not expected
    _assert_products_exact_and_read_only(expected, actual)


def test_reader_accepts_single_site_empty_cross_with_exact_schema(tmp_path: Path) -> None:
    """單站零列 cross-site Parquet 仍須以精確三欄 schema 讀回空 tuple。

    零列表示沒有 ordered site pair，而不是 schema 缺失或 count=0 的替代寫法；reader
    必須接受 writer 的合法空表，但不得捏造來源站、目標站或不重複成員計數列。
    """

    directory, _products = _write_valid_fixture(tmp_path, "single", suffix="empty-cross")
    file_name = "cross_site_counts.parquet"
    disk_table = pq.read_table(directory / file_name)

    loaded = _read_encoded_products(directory)

    assert disk_table.num_rows == 0
    assert tuple(disk_table.column_names) == (
        "source_study_site_id",
        "target_study_site_id",
        "unique_member_count",
    )
    assert disk_table.schema.equals(TABLE_SCHEMAS[file_name], check_metadata=True)
    assert loaded.tables[file_name] == ()


@pytest.mark.parametrize("directory_kind", ["missing", "regular-file", "directory-symlink"])
def test_reader_rejects_unsafe_input_directory_without_path_leak(
    tmp_path: Path,
    directory_kind: str,
) -> None:
    """不存在、普通檔案或目錄 symlink 均不可作為固定產品 reader 入口。"""

    if directory_kind == "missing":
        directory = tmp_path / "missing-release-directory"
    elif directory_kind == "regular-file":
        directory = tmp_path / "release-is-file"
        directory.write_bytes(b"not-a-directory")
    else:
        real_directory, _products = _write_valid_fixture(
            tmp_path,
            "single",
            suffix="real-directory",
        )
        directory = tmp_path / "release-directory-link"
        directory.symlink_to(real_directory, target_is_directory=True)

    _assert_reader_rejects(directory, tmp_path, expected_fragment="directory")


@pytest.mark.parametrize("product_kind", ["table", "npy"])
@pytest.mark.parametrize(
    "node_tamper",
    ["missing", "directory", "symlink", "broken-symlink"],
)
def test_reader_rejects_unsafe_fixed_target_nodes_without_path_leak(
    tmp_path: Path,
    product_kind: str,
    node_tamper: str,
) -> None:
    """固定 table 與 NPY 代表目標缺失或非普通檔案時皆須拒絕。

    每個案例先由 writer 建立完整 27 檔，再只替換一個固定節點；symlink 即使指向
    普通檔案也不接受，broken symlink 即使 ``exists()`` 為 false 仍須辨識為連結。
    錯誤只能指出固定檔名，不得洩漏 tmp 絕對路徑。
    """

    directory, _products = _write_valid_fixture(
        tmp_path,
        "single",
        suffix=f"{product_kind}-{node_tamper}",
    )
    file_name = (
        "site_index.parquet"
        if product_kind == "table"
        else "age_bin_edges_seconds.npy"
    )
    target = directory / file_name
    target.unlink()
    if node_tamper == "directory":
        target.mkdir()
    elif node_tamper == "symlink":
        replacement = tmp_path / f"{product_kind}-replacement-file"
        replacement.write_bytes(b"replacement")
        target.symlink_to(replacement)
    elif node_tamper == "broken-symlink":
        target.symlink_to(tmp_path / f"{product_kind}-missing-target")
        assert target.is_symlink()
        assert not target.exists()

    _assert_reader_rejects(directory, tmp_path, expected_fragment=file_name)


@pytest.mark.parametrize(
    "tamper_case",
    [
        "column-name",
        "column-order",
        "logical-type",
        "nullability",
        "schema-metadata",
        "field-metadata",
        "nonnull-row-null",
        "nan-float",
        "infinite-float",
        "blank-string",
        "negative-index",
    ],
)
def test_reader_rejects_parquet_schema_or_scalar_tamper_before_codec(
    tmp_path: Path,
    tamper_case: str,
) -> None:
    """Parquet schema、metadata、null 與錯誤 scalar 必須由 reader fail closed。

    案例只改 site-index 的 storage-level schema 或單一 row 值，不建立任何跨產品 join
    矛盾。reader 必須在回傳 ``EncodedAggregateProducts`` 前拒絕欄名／順序、logical
    type、nullability、schema／field metadata、null、非有限浮點、空白字串與負 index。
    """

    directory, _products = _write_valid_fixture(
        tmp_path,
        "single",
        suffix=f"parquet-{tamper_case}",
    )
    file_name = "site_index.parquet"
    target = directory / file_name
    valid_table = pq.read_table(target)
    tampered_table = _tampered_site_index_table(valid_table, tamper_case)
    _write_parquet_table(target, tampered_table)

    _assert_reader_rejects(directory, tmp_path, expected_fragment=file_name)


@pytest.mark.parametrize(
    "tamper_case",
    [
        "float32",
        "int32",
        "big-endian-int64",
        "two-dimensional",
        "negative-int",
        "nan-float",
        "infinite-float",
        "negative-float",
        "npz-archive",
    ],
)
def test_reader_rejects_npy_dtype_shape_value_or_archive_tamper(
    tmp_path: Path,
    tamper_case: str,
) -> None:
    """NPY 拒絕錯 dtype／byte order、二維、負值、非有限值與 NPZ archive。

    整數產品保存非負 count／index／offset，浮點產品保存有限非負的公尺或秒；reader
    只接受逐檔固定的 little-endian ``<i8``／``<f8`` 一維陣列，不可 astype、reshape、
    clip 或將 archive 誤認為單一 ndarray。
    """

    directory, products = _write_valid_fixture(
        tmp_path,
        "single",
        suffix=f"npy-{tamper_case}",
    )
    archive = tamper_case == "npz-archive"
    if tamper_case == "float32":
        file_name = "age_bin_edges_seconds.npy"
        values = np.array(products.arrays[file_name], dtype=np.float32)
    elif tamper_case == "int32":
        file_name = "site_cell_offsets.npy"
        values = np.array(products.arrays[file_name], dtype=np.int32)
    elif tamper_case == "big-endian-int64":
        file_name = "site_cell_offsets.npy"
        values = np.array(products.arrays[file_name], dtype=">i8")
    elif tamper_case == "two-dimensional":
        file_name = "site_cell_offsets.npy"
        values = np.array([[0, 2]], dtype="<i8")
    elif tamper_case == "negative-int":
        file_name = "local_first_exit_count.npy"
        values = np.array([-1, 0], dtype="<i8")
    elif tamper_case == "nan-float":
        file_name = "boundary_bin_edges_m.npy"
        values = np.array(products.arrays[file_name], dtype="<f8", copy=True)
        values[1] = np.nan
    elif tamper_case == "infinite-float":
        file_name = "boundary_bin_edges_m.npy"
        values = np.array(products.arrays[file_name], dtype="<f8", copy=True)
        values[1] = np.inf
    elif tamper_case == "negative-float":
        file_name = "pathway_residence_time_seconds.npy"
        values = np.array(products.arrays[file_name], dtype="<f8", copy=True)
        values[0] = -1.0
    else:
        file_name = "site_cell_offsets.npy"
        values = np.array(products.arrays[file_name], dtype="<i8", copy=True)
    _write_npy_or_archive(directory / file_name, values, archive=archive)

    _assert_reader_rejects(directory, tmp_path, expected_fragment=file_name)


def test_reader_does_not_modify_input_file_bytes(tmp_path: Path) -> None:
    """成功讀取前後全部 27 檔 SHA-256 必須一致，reader 不得改寫磁碟輸入。

    Parquet materialization 與 NPY defensive snapshot 都只能讀取既有 bytes；即使回傳
    ndarray 需要建立唯讀副本，也不得改變原檔的公尺、秒、計數、schema metadata 或
    filesystem 內容。
    """

    directory, _products = _write_valid_fixture(tmp_path, "two", suffix="disk-read-only")
    before = {
        file_name: _file_sha256(directory / file_name)
        for file_name in _PRODUCT_FILE_NAMES
    }

    loaded = _read_encoded_products(directory)

    assert isinstance(loaded, EncodedAggregateProducts)
    after = {
        file_name: _file_sha256(directory / file_name)
        for file_name in _PRODUCT_FILE_NAMES
    }
    assert after == before
