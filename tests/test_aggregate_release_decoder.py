"""Aggregate release 公開 decoder 的 exact round-trip 與 fail-closed 契約測試。

本檔沿用 payload 測試的單站與兩站 synthetic fixtures，先由公開 encoder 產生固定九表、
十八陣列及 metadata，再交給公開 decoder。round-trip 會逐表核對欄位順序、scalar 型別
與值，並逐陣列核對 dtype、shape、C-order bytes 與元素；竄改案例則一律重新呼叫公開
``EncodedAggregateProducts`` 建構產品，模擬 Parquet／NumPy adapter 已載入但內容不符
release 契約的輸入。所有計數只代表條件式來源足跡載體，不是絕對來源機率或因果歸因。
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import test_aggregate_release_payload as payload_fixture

from lagrangian_backtracking.aggregate_release_codec import (
    decode_aggregate_release_payload,
    encode_aggregate_release_payload,
    metadata_from_payload,
)
from lagrangian_backtracking.aggregate_release_layout import EncodedAggregateProducts
from lagrangian_backtracking.aggregate_release_payload import AggregateReleasePayload


def _single_site_payload() -> AggregateReleasePayload:
    """由既有 helper 建立一站、兩格網、兩個 age bins 的合法 synthetic payload。"""

    return AggregateReleasePayload(**payload_fixture._valid_payload_kwargs())


def _two_site_payload() -> AggregateReleasePayload:
    """由既有 helper 建立兩站固定拓撲，涵蓋排序、跨站零列與串接 offset。"""

    return AggregateReleasePayload(**payload_fixture._two_site_payload_kwargs())


def _assert_products_exact(
    expected: EncodedAggregateProducts,
    actual: EncodedAggregateProducts,
) -> None:
    """逐表與逐陣列執行 exact comparison，不以 shape 或 key set 取代內容驗證。

    表格欄位 iteration order 是 release schema 的一部分；每個 scalar 也要求 exact
    Python 型別，float 另比對 float64 bytes，以免負零等位元差異被一般相等運算掩蓋。
    陣列除了 dtype、shape 與元素外，也比較 C-order bytes，確保多站串接與 histogram
    最後一軸沒有被轉置或重排。
    """

    assert tuple(sorted(actual.tables)) == tuple(sorted(expected.tables))
    for file_name in sorted(expected.tables):
        expected_rows = expected.tables[file_name]
        actual_rows = actual.tables[file_name]
        assert len(actual_rows) == len(expected_rows)
        for expected_row, actual_row in zip(expected_rows, actual_rows, strict=True):
            assert tuple(actual_row) == tuple(expected_row)
            for column_name in expected_row:
                expected_value = expected_row[column_name]
                actual_value = actual_row[column_name]
                assert type(actual_value) is type(expected_value)
                if type(expected_value) is float:
                    expected_bytes = np.asarray(expected_value, dtype=np.float64).tobytes()
                    actual_bytes = np.asarray(actual_value, dtype=np.float64).tobytes()
                    assert actual_bytes == expected_bytes
                else:
                    assert actual_value == expected_value

    assert tuple(sorted(actual.arrays)) == tuple(sorted(expected.arrays))
    for file_name in sorted(expected.arrays):
        expected_array = expected.arrays[file_name]
        actual_array = actual.arrays[file_name]
        assert actual_array.dtype == expected_array.dtype
        assert actual_array.shape == expected_array.shape
        assert actual_array.flags.c_contiguous
        np.testing.assert_array_equal(actual_array, expected_array)
        assert actual_array.tobytes(order="C") == expected_array.tobytes(order="C")


def _rebuild_products(
    products: EncodedAggregateProducts,
    *,
    table_file: str | None = None,
    table_rows: tuple[dict[str, object], ...] | None = None,
    array_file: str | None = None,
    array_value: np.ndarray | None = None,
) -> EncodedAggregateProducts:
    """以公開容器重建一份產品，僅替換測試指定的單一 table 或 array。

    所有未竄改表列都複製成新 dict，陣列也複製成原 dtype 的 C-contiguous buffer；因此
    decoder 的失敗可歸因於指定欄位或 offset，而不是 caller alias、缺少固定檔名，或
    繞過 ``EncodedAggregateProducts`` constructor 的低階屬性寫入。
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
        tables[table_file] = table_rows
    if array_file is not None and array_value is not None:
        arrays[array_file] = array_value
    return EncodedAggregateProducts(tables=tables, arrays=arrays)


def _assert_exact_round_trip(payload: AggregateReleasePayload) -> AggregateReleasePayload:
    """執行公開 encode／metadata／decode，再核對完整 products 與 metadata 快照。"""

    metadata = metadata_from_payload(payload)
    encoded = encode_aggregate_release_payload(payload)
    decoded = decode_aggregate_release_payload(metadata, payload.aggregate_spec, encoded)

    assert decoded is not payload
    assert decoded.aggregate_spec is not payload.aggregate_spec
    assert decoded.aggregate_spec == payload.aggregate_spec
    assert metadata_from_payload(decoded) == metadata
    _assert_products_exact(encoded, encode_aggregate_release_payload(decoded))
    return decoded


def test_decoder_single_site_exact_round_trip() -> None:
    """單站九表與十八陣列必須經公開 decoder 完整還原且逐值 exact re-encode。

    單站 fixture 包含一個空 cross-site 表、三條 boundary／source 拓撲列與兩個公尺制
    cell。此組合可同時驗證空表保留、零事件列、C-order 格網及 age histogram，不容許
    decoder 以補零、轉置或省略零列取得表面上相同的 shape。
    """

    decoded = _assert_exact_round_trip(_single_site_payload())

    assert tuple(decoded.pathway_by_site) == ("site-a",)
    assert len(decoded.shard_bindings) == 1
    assert len(decoded.scenario_strata) == 1


def test_decoder_two_site_exact_round_trip() -> None:
    """兩站產品必須保留 site 字典序、跨站 ordered pairs 與所有串接 offset 值。

    兩站 fixture 的站點、shard、scenario、boundary 與 source-receptor 都有多列，且包含
    A→B、B→A 兩筆零計數跨站拓撲。逐表／陣列 exact re-encode 可避免只驗證單站時漏掉
    的第二站切片錯位、列排序改變或跨站零列被省略。
    """

    decoded = _assert_exact_round_trip(_two_site_payload())

    assert tuple(decoded.pathway_by_site) == ("site-a", "site-b")
    assert len(decoded.shard_bindings) == 2
    assert len(decoded.scenario_strata) == 2


@pytest.mark.parametrize(
    ("count_field", "table_file"),
    (
        pytest.param("shard_row_count", "shard_bindings.parquet", id="shard"),
        pytest.param("scenario_row_count", "scenario_strata.parquet", id="scenario"),
        pytest.param("site_row_count", "site_index.parquet", id="site"),
        pytest.param("boundary_row_count", "boundary_index.parquet", id="boundary"),
        pytest.param(
            "source_receptor_row_count",
            "source_receptor_index.parquet",
            id="source-receptor",
        ),
    ),
)
def test_decoder_rejects_each_metadata_row_count_mismatch(
    count_field: str,
    table_file: str,
) -> None:
    """五個 metadata row count 任一與實際核心索引表不同時都必須拒絕。

    row count 是 manifest 對固定產品拓撲的工程宣告，不是事件數或科學分母。每個案例
    只把合法正整數加一，確保錯誤來自 metadata 與表格實際列數不一致，而不是 metadata
    constructor 的型別或正值檢查。
    """

    payload = _single_site_payload()
    metadata = metadata_from_payload(payload)
    products = encode_aggregate_release_payload(payload)
    mismatched = replace(
        metadata,
        **{count_field: getattr(metadata, count_field) + 1},
    )

    with pytest.raises(ValueError, match=table_file.split(".")[0]):
        decode_aggregate_release_payload(mismatched, payload.aggregate_spec, products)


@pytest.mark.parametrize(
    ("metadata_field", "replacement_value"),
    (
        pytest.param("run_id", "different-run", id="run-id"),
        pytest.param("aggregate_spec_source_sha256", "3" * 64, id="source-sha256"),
        pytest.param("aggregate_spec_canonical_sha256", "4" * 64, id="canonical-sha256"),
    ),
)
def test_decoder_rejects_metadata_and_spec_identity_mismatch(
    metadata_field: str,
    replacement_value: str,
) -> None:
    """metadata 與 spec 的 run、source 或 canonical 身分不一致時必須 fail closed。

    三個替代值本身都符合公開 metadata constructor 的 slug 或小寫 SHA-256 格式；因此
    decoder 必須在 typed inputs 的交叉核對階段拒絕，不能把另一個 run 或另一份規格的
    provenance 與現有產品拼接。摘要只用於來源追蹤，不代表科學有效性。
    """

    payload = _single_site_payload()
    metadata = metadata_from_payload(payload)
    products = encode_aggregate_release_payload(payload)
    mismatched = replace(metadata, **{metadata_field: replacement_value})

    with pytest.raises(ValueError, match=metadata_field):
        decode_aggregate_release_payload(mismatched, payload.aggregate_spec, products)


@pytest.mark.parametrize(
    ("table_file", "integer_column", "tamper_kind"),
    (
        pytest.param("shard_bindings.parquet", "shard_index", "column-order", id="shard-order"),
        pytest.param("shard_bindings.parquet", "shard_index", "numpy-int", id="shard-np-int"),
        pytest.param(
            "scenario_strata.parquet",
            "scenario_index",
            "column-order",
            id="scenario-order",
        ),
        pytest.param(
            "scenario_strata.parquet",
            "scenario_index",
            "numpy-int",
            id="scenario-np-int",
        ),
        pytest.param("site_index.parquet", "site_index", "column-order", id="site-order"),
        pytest.param("site_index.parquet", "site_index", "numpy-int", id="site-np-int"),
    ),
)
def test_decoder_rejects_core_row_order_or_numpy_integer_scalar(
    table_file: str,
    integer_column: str,
    tamper_kind: str,
) -> None:
    """shard、scenario、site row 的欄序與原生整數型別任一被改寫都必須拒絕。

    欄位集合相同但 iteration order 相反時，Parquet schema 契約仍已改變；``np.int64``
    雖可表示相同數值，也不是表格邊界要求的原生 Python ``int``。兩類 tamper 都透過
    公開產品容器重建，確認 decoder 不依賴 mapping equality 或隱式 ``int()`` 轉型。
    """

    payload = _single_site_payload()
    metadata = metadata_from_payload(payload)
    products = encode_aggregate_release_payload(payload)
    original_rows = products.tables[table_file]
    first_row = dict(original_rows[0])
    if tamper_kind == "column-order":
        tampered_row = dict(reversed(tuple(first_row.items())))
        expected_message = "固定欄位與順序"
    else:
        tampered_row = first_row
        tampered_row[integer_column] = np.int64(tampered_row[integer_column])
        expected_message = "原生 int"
    tampered_rows = (tampered_row,) + tuple(dict(row) for row in original_rows[1:])
    tampered_products = _rebuild_products(
        products,
        table_file=table_file,
        table_rows=tampered_rows,
    )

    with pytest.raises(ValueError, match=expected_message):
        decode_aggregate_release_payload(metadata, payload.aggregate_spec, tampered_products)


@pytest.mark.parametrize(
    "tamper_kind",
    (
        pytest.param("offset-start-nonzero", id="offset-start-nonzero"),
        pytest.param("offset-not-increasing", id="offset-not-increasing"),
        pytest.param("offset-tail-wrong", id="offset-tail-wrong"),
        pytest.param("row-start-wrong", id="row-start-wrong"),
        pytest.param("row-stop-wrong", id="row-stop-wrong"),
    ),
)
def test_decoder_rejects_invalid_site_cell_offsets_or_row_bounds(tamper_kind: str) -> None:
    """site cell offset 的起點、單調性、尾端或 row 邊界錯誤都必須拒絕。

    ``site_cell_offsets.npy`` 定義各站 C-order cell 片段，site row 的 start／stop 必須
    逐值對應同一陣列。測試不修補 event/pathway 陣列，故錯誤尾端也必須因宣告長度與
    實際資料不符而失敗，不能 clip、padding 或以 row 值覆蓋 canonical offset。
    """

    payload = _single_site_payload()
    metadata = metadata_from_payload(payload)
    products = encode_aggregate_release_payload(payload)

    if tamper_kind == "offset-start-nonzero":
        tampered_products = _rebuild_products(
            products,
            array_file="site_cell_offsets.npy",
            array_value=np.array([1, 2], dtype=np.int64),
        )
        expected_message = "從 0 開始"
    elif tamper_kind == "offset-not-increasing":
        tampered_products = _rebuild_products(
            products,
            array_file="site_cell_offsets.npy",
            array_value=np.array([0, 0], dtype=np.int64),
        )
        expected_message = "嚴格遞增"
    elif tamper_kind == "offset-tail-wrong":
        tampered_products = _rebuild_products(
            products,
            array_file="site_cell_offsets.npy",
            array_value=np.array([0, 3], dtype=np.int64),
        )
        expected_message = "shape"
    else:
        site_rows = products.tables["site_index.parquet"]
        tampered_row = dict(site_rows[0])
        if tamper_kind == "row-start-wrong":
            tampered_row["cell_start_offset"] = 1
        else:
            tampered_row["cell_stop_offset"] = 1
        tampered_products = _rebuild_products(
            products,
            table_file="site_index.parquet",
            table_rows=(tampered_row,),
        )
        expected_message = "row cell offset"

    with pytest.raises(ValueError, match=expected_message):
        decode_aggregate_release_payload(metadata, payload.aggregate_spec, tampered_products)
