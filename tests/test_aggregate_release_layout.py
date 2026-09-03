"""測試 aggregate release 固定拓撲與安全數值容器。

本檔只測試 ``EncodedAggregateProducts`` 的容器邊界，不呼叫尚在實作中的
encoder，也不讀取 raw NetCDF、transfer archive 或任何外部產品。測試 fixture
以普通 ``dict``、``list`` 與可寫 NumPy 陣列模擬呼叫端輸入；表格列只保留可讀的
識別欄位，陣列則使用非負索引／計數與有限時間／距離數值。這些值只用來驗證
拓撲、型別、唯讀狀態及防禦性複製，不代表實際科學結果、絕對來源機率或因果歸因。

陣列 fixture 的浮點值代表秒制時間或公尺制距離的合法有限數值，整數值代表
非負計數或 offset；本容器並不在這一層驗證 shape、軸長或跨檔案守恆關係。每個
測試均重新建立輸入，避免某一個負向案例修改共用 fixture 後影響其他案例。
"""

from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest

from lagrangian_backtracking.aggregate_release_layout import (
    AGGREGATE_RELEASE_ARRAY_FILES,
    AGGREGATE_RELEASE_TABLE_FILES,
    EncodedAggregateProducts,
)

_CROSS_TABLE = "cross_site_counts.parquet"
_TABLE_WITH_ROW = "site_index.parquet"
_INTEGER_ARRAY = "site_cell_offsets.npy"

# 這三個檔案保存秒制或公尺制連續量，fixture 用 float32 輸入以確認容器會統一
# canonical 成 float64；其餘十五個檔案用整數計數／offset，並混用 signed 與
# unsigned 的合法小值，確認整數最後都會安全收斂到 int64。
_FLOAT_ARRAY_FILES = frozenset(
    {
        "age_bin_edges_seconds.npy",
        "boundary_bin_edges_m.npy",
        "pathway_residence_time_seconds.npy",
    }
)

# 某些平台的 NumPy 沒有獨立的 ``float128`` 名稱，``longdouble`` 可能退化為
# float64；只有在確實能保存超出 float64 上限的有限值時，才執行溢位案例。
_FLOAT128_DTYPE = np.dtype(getattr(np, "float128", np.longdouble))
_HAS_EXTENDED_FLOAT = np.finfo(_FLOAT128_DTYPE).max > np.finfo(np.float64).max


def _valid_inputs() -> tuple[dict[str, list[dict[str, object]]], dict[str, np.ndarray]]:
    """建立一組完整且可變的呼叫端輸入 mapping。

    ``tables`` 精確包含九個固定 Parquet 檔名；跨站計數表以空 ``list`` 表示沒有
    跨站事件，其餘八張表各有一列。列中的 ``file_name`` 與 ``row_index`` 只是
    測試用識別欄位，不能取代正式資料列 schema。``arrays`` 精確包含十八個固定
    NumPy 檔名；三個連續量使用有限 float32，其餘使用非負 int32／uint32，所有
    輸入都刻意保持可寫，供後續測試驗證容器的防禦性複製。

    Returns:
        一對普通 ``dict``。第一項的 value 是可變列 ``list``，第二項的 value 是
        可寫 NumPy 陣列；兩者都尚未經過容器 canonical 化。
    """

    tables: dict[str, list[dict[str, object]]] = {}
    for row_index, file_name in enumerate(sorted(AGGREGATE_RELEASE_TABLE_FILES)):
        if file_name == _CROSS_TABLE:
            # 單站或沒有跨站事件時仍要保留固定檔名，但允許其資料列為零筆。
            tables[file_name] = []
        else:
            tables[file_name] = [{"file_name": file_name, "row_index": row_index}]

    arrays: dict[str, np.ndarray] = {}
    for array_index, file_name in enumerate(sorted(AGGREGATE_RELEASE_ARRAY_FILES)):
        if file_name in _FLOAT_ARRAY_FILES:
            # 0.5 與 1.5 代表合法的連續秒／公尺數值，且不含 NaN 或 Infinity。
            arrays[file_name] = np.array(
                [array_index + 0.5, array_index + 1.5],
                dtype=np.float32,
            )
        elif array_index % 2 == 0:
            # 偶數索引用 signed integer，保留非負值以符合計數／offset 契約。
            arrays[file_name] = np.array(
                [array_index, array_index + 1],
                dtype=np.int32,
            )
        else:
            # 奇數索引用 unsigned integer，測試其在 int64 範圍內的正常 canonical 化。
            arrays[file_name] = np.array(
                [array_index, array_index + 1],
                dtype=np.uint32,
            )
    return tables, arrays


def test_valid_products_are_canonical_read_only_snapshots() -> None:
    """驗證合法輸入的固定 key、canonical dtype、唯讀容器與 caller 隔離。

    建構前先保存一列與一個陣列的原始值，建構後再同時修改 caller 的外層 mapping、
    表格 row list、row mapping 與 NumPy buffer。若產品內容仍維持原值，表示容器已
    將外部可變物件複製成自己的 snapshot；另外直接嘗試修改輸出，確認 tuple、
    ``MappingProxyType`` 與 ``write=False`` 三層防線都實際生效。
    """

    tables, arrays = _valid_inputs()
    original_row = dict(tables[_TABLE_WITH_ROW][0])
    original_array = arrays[_INTEGER_ARRAY].copy()
    products = EncodedAggregateProducts(tables=tables, arrays=arrays)

    # 外層 mapping 必須封存為唯讀代理，且固定集合的數量與 key 必須完整保留。
    assert len(products.tables) == 9
    assert len(products.arrays) == 18
    assert set(products.tables) == set(AGGREGATE_RELEASE_TABLE_FILES)
    assert set(products.arrays) == set(AGGREGATE_RELEASE_ARRAY_FILES)
    assert type(products.tables) is MappingProxyType
    assert type(products.arrays) is MappingProxyType

    for file_name in sorted(AGGREGATE_RELEASE_TABLE_FILES):
        rows = products.tables[file_name]
        assert type(rows) is tuple
        if file_name == _CROSS_TABLE:
            assert rows == ()
        else:
            assert len(rows) == 1
            assert type(rows[0]) is MappingProxyType
            assert rows[0]["file_name"] == file_name

    # 所有整數輸入都應 canonical 為 int64，浮點輸入都應 canonical 為 float64，
    # 並且 canonical buffer 不得再接受元素寫入。
    for file_name in sorted(AGGREGATE_RELEASE_ARRAY_FILES):
        array = products.arrays[file_name]
        expected_dtype = np.float64 if file_name in _FLOAT_ARRAY_FILES else np.int64
        assert type(array) is np.ndarray
        assert array.dtype == np.dtype(expected_dtype)
        assert array.flags.writeable is False

    # 修改 caller 的外層 mapping、表格列與列內容，不應污染已建立的產品 snapshot。
    tables[_TABLE_WITH_ROW][0]["file_name"] = "caller-mutated"
    tables[_TABLE_WITH_ROW].append({"file_name": "caller-appended", "row_index": 999})
    tables[_TABLE_WITH_ROW] = []
    assert dict(products.tables[_TABLE_WITH_ROW][0]) == original_row
    assert len(products.tables[_TABLE_WITH_ROW]) == 1

    # 修改 caller 的原始 array buffer 與外層 key 指向，也不應改變 canonical 複本。
    arrays[_INTEGER_ARRAY][0] = 999
    arrays[_INTEGER_ARRAY] = np.array([12345], dtype=np.int32)
    np.testing.assert_array_equal(products.arrays[_INTEGER_ARRAY], original_array)

    # 輸出本身的三種可變入口都必須被封住；例外型別也可作為公開契約的一部分檢查。
    with pytest.raises(TypeError):
        products.tables["new.parquet"] = ()
    with pytest.raises(TypeError):
        products.tables[_TABLE_WITH_ROW][0]["file_name"] = "output-mutated"
    with pytest.raises(AttributeError):
        products.tables[_TABLE_WITH_ROW].append({})
    with pytest.raises(ValueError):
        products.arrays[_INTEGER_ARRAY][0] = 0


@pytest.mark.parametrize("container_name", ["tables", "arrays"])
@pytest.mark.parametrize("key_change", ["missing", "extra"])
def test_outer_mappings_require_exact_fixed_key_sets(container_name: str, key_change: str) -> None:
    """確認 tables 與 arrays 都拒絕遺漏或額外的 release key。

    這裡只改變外層檔名 mapping，不改變任何資料列或陣列內容；因此失敗原因必須
    來自固定拓撲本身，而不是後續的 row 或 dtype 驗證。遺漏案例刪除排序後第一個
    合法 key，額外案例加入不屬於產品 schema 的新檔名。
    """

    tables, arrays = _valid_inputs()
    values: dict[str, object] = tables if container_name == "tables" else arrays
    expected_keys = (
        AGGREGATE_RELEASE_TABLE_FILES
        if container_name == "tables"
        else AGGREGATE_RELEASE_ARRAY_FILES
    )
    changed_key = sorted(expected_keys)[0]

    if key_change == "missing":
        del values[changed_key]
    elif container_name == "tables":
        values["unexpected.parquet"] = [{"file_name": "unexpected"}]
    else:
        values["unexpected.npy"] = np.array([0], dtype=np.int32)

    with pytest.raises(ValueError, match=container_name):
        EncodedAggregateProducts(tables=tables, arrays=arrays)


@pytest.mark.parametrize(
    "bad_container",
    [
        pytest.param(None, id="none"),
        pytest.param({}, id="mapping-not-sequence"),
        pytest.param({"row_index": 0}, id="single-row-mapping"),
        pytest.param("rows", id="string"),
        pytest.param(b"rows", id="bytes"),
        pytest.param({("row", 0)}, id="set"),
        pytest.param(object(), id="object"),
    ],
)
def test_table_values_must_be_sequences_of_rows(bad_container: object) -> None:
    """拒絕把非列序列物件當成表格內容。

    表格的外層檔名 mapping 已完整，只有 ``site_index.parquet`` 的 value 被替換；
    ``None``、mapping、字串、bytes、set 與一般物件都不是可重現的 row sequence，
    不應被容器自動轉換或猜測其資料意義。
    """

    tables, arrays = _valid_inputs()
    tables[_TABLE_WITH_ROW] = bad_container  # type: ignore[assignment]

    with pytest.raises(ValueError, match="tables"):
        EncodedAggregateProducts(tables=tables, arrays=arrays)


@pytest.mark.parametrize(
    "bad_row",
    [
        pytest.param(None, id="none"),
        pytest.param(1, id="integer"),
        pytest.param([], id="list"),
        pytest.param(("not", "mapping"), id="tuple"),
        pytest.param("row", id="string"),
    ],
)
def test_table_rows_must_be_mappings(bad_row: object) -> None:
    """拒絕不是 mapping 的單列資料，避免欄位結構被隱式猜測。

    row mapping 是後續 Parquet 欄位封存所需的最小結構；本容器不要求特定欄位名稱，
    但必須先確保每一列能以 key/value 方式被防禦性複製，否則無法提供 row proxy。
    """

    tables, arrays = _valid_inputs()
    tables[_TABLE_WITH_ROW] = [bad_row]  # type: ignore[list-item]

    with pytest.raises(ValueError, match="tables"):
        EncodedAggregateProducts(tables=tables, arrays=arrays)


@pytest.mark.parametrize(
    "empty_table",
    sorted(AGGREGATE_RELEASE_TABLE_FILES - {_CROSS_TABLE}),
)
def test_only_cross_site_counts_may_be_empty(empty_table: str) -> None:
    """確認只有 cross-site table 可以用零列表示沒有跨站事件。

    其他拓撲、分層、結果與分母表即使所有計數為零，也必須保留至少一列來表達
    固定資料拓撲；逐一參數化可避免只驗證到某一張表的空值規則。
    """

    tables, arrays = _valid_inputs()
    tables[empty_table] = []

    with pytest.raises(ValueError, match="不得為空"):
        EncodedAggregateProducts(tables=tables, arrays=arrays)


@pytest.mark.parametrize(
    "bad_value",
    [
        pytest.param(None, id="none"),
        pytest.param([0], id="list"),
        pytest.param((0,), id="tuple"),
        pytest.param(0, id="scalar"),
        pytest.param({"value": 0}, id="mapping"),
    ],
)
def test_array_values_must_be_numpy_arrays(bad_value: object) -> None:
    """拒絕 list、tuple、scalar 或 mapping，陣列輸入必須明確是 np.ndarray。

    不接受可由 NumPy 便利轉換的 Python 容器，是為了避免 dtype、缺值與記憶體
    layout 在資料契約外被隱式推導；encoder 應在進入本容器前完成明確的陣列建立。
    """

    tables, arrays = _valid_inputs()
    arrays[_INTEGER_ARRAY] = bad_value  # type: ignore[assignment]

    with pytest.raises(ValueError, match="np.ndarray"):
        EncodedAggregateProducts(tables=tables, arrays=arrays)


@pytest.mark.parametrize(
    "bad_array",
    [
        pytest.param(np.array([True], dtype=np.bool_), id="bool"),
        pytest.param(np.array([1], dtype=object), id="object"),
        pytest.param(np.array(["1"], dtype=np.str_), id="string"),
        pytest.param(np.array([1 + 0j], dtype=np.complex128), id="complex"),
        pytest.param(
            np.array(
                [(1, 2.0)],
                dtype=np.dtype([("count", np.int64), ("fraction", np.float64)]),
            ),
            id="structured",
        ),
    ],
)
def test_array_rejects_bool_object_string_complex_and_structured_dtypes(
    bad_array: np.ndarray,
) -> None:
    """拒絕不能直接代表計數、offset、時間或距離的 NumPy dtype。

    ``bool``、object、字串、複數與 structured dtype 即使某些元素表面上可轉成
    數值，也可能隱含欄位語意或轉換損失；容器應 fail-fast 拒絕，而不是替 caller
    猜測資料欄位、丟棄虛部或執行字串／object 轉數字。
    """

    tables, arrays = _valid_inputs()
    arrays[_INTEGER_ARRAY] = bad_array

    with pytest.raises(ValueError, match="不允許"):
        EncodedAggregateProducts(tables=tables, arrays=arrays)


@pytest.mark.parametrize(
    ("bad_array", "error_pattern"),
    [
        pytest.param(
            np.array([-1], dtype=np.int64),
            "不可為負",
            id="signed-negative",
        ),
        pytest.param(
            np.array([2**63], dtype=np.uint64),
            "超過 int64 上限",
            id="unsigned-over-int64",
        ),
        pytest.param(
            np.array([np.nan], dtype=np.float64),
            "有限",
            id="nan",
        ),
        pytest.param(
            np.array([np.inf], dtype=np.float64),
            "有限",
            id="positive-infinity",
        ),
        pytest.param(
            np.array([-np.inf], dtype=np.float64),
            "有限",
            id="negative-infinity",
        ),
    ],
)
def test_array_rejects_negative_or_nonfinite_numeric_values(
    bad_array: np.ndarray,
    error_pattern: str,
) -> None:
    """拒絕負整數、超出 int64 的 unsigned 整數與所有非有限浮點值。

    計數與 offset 不可用負值或 unsigned 溢位後的負值表示；時間、距離及其他
    連續量則不得含 NaN 或正負 Infinity。每個案例都只替換一個合法 array key，
    以確認錯誤由數值政策觸發，而不是由外層 key set 造成。
    """

    tables, arrays = _valid_inputs()
    arrays[_INTEGER_ARRAY] = bad_array

    with pytest.raises(ValueError, match=error_pattern):
        EncodedAggregateProducts(tables=tables, arrays=arrays)


@pytest.mark.skipif(
    not _HAS_EXTENDED_FLOAT,
    reason="此平台的 longdouble 沒有比 float64 更大的有限表示範圍",
)
def test_array_rejects_float128_value_that_overflows_float64() -> None:
    """拒絕可由 float128 表示、但 canonical 成 float64 會溢位的有限值。

    先建立仍屬有限值的 extended floating-point 最大值；若容器直接轉 float64，
    該值會變成 Infinity，因此測試要求容器在轉型後再次檢查有限性並回報錯誤。
    這個案例只在平台確實提供 extended ``longdouble``／``float128`` 時執行，避免
    在 float128 退化為 float64 的平台上製造不存在的測試資料。
    """

    tables, arrays = _valid_inputs()
    arrays[_INTEGER_ARRAY] = np.array(
        [np.finfo(_FLOAT128_DTYPE).max],
        dtype=_FLOAT128_DTYPE,
    )

    with pytest.raises(ValueError, match="float64"):
        EncodedAggregateProducts(tables=tables, arrays=arrays)
