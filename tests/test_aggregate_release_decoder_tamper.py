"""Aggregate release decoder 對四類固定拓撲竄改的封閉式失敗測試。

本檔沿用既有單站與兩站 synthetic payload fixture，先由公開 encoder 取得合法產品，
再透過公開 ``EncodedAggregateProducts`` constructor 建立單點竄改版本。邊界 edge 的
距離單位是公尺，source-receptor travel histogram 的 age-bin 軸是秒；跨站與 outcome
中計數為零的列仍是必要拓撲，不能把缺列解讀為零事件。這些資料只承載條件式來源
足跡或相對來源權重，不代表絕對來源機率或因果歸因。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pytest
import test_aggregate_release_codec as codec_fixture
import test_aggregate_release_payload as payload_fixture

from lagrangian_backtracking.aggregate_release_codec import (
    AggregateReleaseMetadata,
    decode_aggregate_release_payload,
    encode_aggregate_release_payload,
    metadata_from_payload,
)
from lagrangian_backtracking.aggregate_release_layout import EncodedAggregateProducts
from lagrangian_backtracking.aggregate_release_payload import AggregateReleasePayload
from lagrangian_backtracking.aggregate_spec import AggregateSpec


def _encoded_inputs(
    *,
    two_site: bool = False,
) -> tuple[AggregateReleaseMetadata, AggregateSpec, EncodedAggregateProducts]:
    """由既有 fixture 建立 decoder 的合法 metadata、規格與產品輸入。

    單站 fixture 用於公尺制 boundary edge、秒制 travel-age 與 outcome 拓撲；兩站
    fixture 額外提供 A→B、B→A 兩個 ordered cross-site 零列。helper 只呼叫公開 codec
    API，不複製 production 的排序、offset 或科學計算邏輯。
    """

    if two_site:
        payload = AggregateReleasePayload(**payload_fixture._two_site_payload_kwargs())
    else:
        payload = codec_fixture._single_site_payload()
    return (
        metadata_from_payload(payload),
        payload.aggregate_spec,
        encode_aggregate_release_payload(payload),
    )


def _tampered_products(
    products: EncodedAggregateProducts,
    *,
    table_overrides: Mapping[str, Sequence[Mapping[str, object]]] | None = None,
    array_overrides: Mapping[str, np.ndarray] | None = None,
) -> EncodedAggregateProducts:
    """防禦性複製合法產品並以公開 constructor 封存指定單點竄改。

    每次呼叫都複製全部 row mapping 與 NumPy buffer，只覆寫測試明示的 table 或 array，
    確保原始 encoder 產品不受影響。``EncodedAggregateProducts`` 僅驗證固定檔名與安全
    scalar／array 容器；跨表拓撲、公尺 edge、秒制 histogram 長度與零列完整性仍應由
    公開 decoder fail closed，這正是下列測試的責任邊界。
    """

    tables = {
        file_name: tuple(dict(row) for row in rows)
        for file_name, rows in products.tables.items()
    }
    arrays = {
        file_name: np.array(values, order="C", copy=True)
        for file_name, values in products.arrays.items()
    }
    for file_name, rows in (table_overrides or {}).items():
        tables[file_name] = tuple(dict(row) for row in rows)
    for file_name, values in (array_overrides or {}).items():
        arrays[file_name] = np.array(values, order="C", copy=True)
    return EncodedAggregateProducts(tables=tables, arrays=arrays)


def test_decoder_rejects_noncanonical_boundary_edge() -> None:
    """竄改一個公尺制 boundary edge 後，decoder 必須拒絕非 canonical 格線。

    測試保留 edge 數、offset 與嚴格遞增關係，只把第一段內部 edge 從 1.0 m 改成
    0.5 m；因此錯誤責任落在產品 edge 必須逐值等於 AggregateSpec canonical edge，
    不能由 decoder 近似、重分箱或依 segment 尾端猜測修補。
    """

    metadata, aggregate_spec, products = _encoded_inputs()
    edges = np.array(products.arrays["boundary_bin_edges_m.npy"], copy=True)
    edges[1] = 0.5
    tampered = _tampered_products(
        products,
        array_overrides={"boundary_bin_edges_m.npy": edges},
    )

    with pytest.raises(ValueError, match="canonical spec edges"):
        decode_aggregate_release_payload(metadata, aggregate_spec, tampered)


def test_decoder_rejects_short_source_travel_age_array() -> None:
    """source-receptor 秒制 travel-age 陣列少一值時，decoder 必須拒絕長度錯位。

    每一筆 source-receptor row 都必須對應完整 age-bin 片段；截掉最後一個 int64 值會
    破壞 ``row_count × age_bin_count`` 長度，但不改變固定檔名與非負型別，因此應由
    decoder 的 shape 契約拒絕，而不能 padding 零值或縮短最後一列的秒制時間軸。
    """

    metadata, aggregate_spec, products = _encoded_inputs()
    file_name = "source_receptor_travel_age_histogram.npy"
    shortened = np.array(products.arrays[file_name][:-1], copy=True)
    tampered = _tampered_products(
        products,
        array_overrides={file_name: shortened},
    )

    with pytest.raises(ValueError, match=file_name):
        decode_aggregate_release_payload(metadata, aggregate_spec, tampered)


def test_decoder_rejects_missing_two_site_a_to_b_zero_row() -> None:
    """兩站 cross-site 表缺少 A→B 零列時，decoder 必須拒絕不完整 ordered topology。

    合法兩站 fixture 同時保存 A→B 與 B→A，且目前兩列計數皆為零。移除 A→B 不代表
    觀測到零事件，而是遺失一個必要方向的拓撲；decoder 不得用反方向列、零值或站點
    集合推導補回缺列。
    """

    metadata, aggregate_spec, products = _encoded_inputs(two_site=True)
    file_name = "cross_site_counts.parquet"
    rows = [dict(row) for row in products.tables[file_name]]
    removed = [
        row
        for row in rows
        if row["source_study_site_id"] == "site-a"
        and row["target_study_site_id"] == "site-b"
    ]
    assert len(removed) == 1
    assert removed[0]["unique_member_count"] == 0
    remaining = [row for row in rows if row not in removed]
    tampered = _tampered_products(
        products,
        table_overrides={file_name: remaining},
    )

    with pytest.raises(ValueError, match="cross_site_counts"):
        decode_aggregate_release_payload(metadata, aggregate_spec, tampered)


def test_decoder_rejects_missing_zero_outcome_status() -> None:
    """outcome 表缺少一個零計數非 ACTIVE 狀態時，decoder 必須拒絕缺列。

    零計數狀態用來區分「已登錄但未發生」與「產品缺少狀態欄位」；本測試只刪除一筆
    count=0 的合法列，保留其他事件與分母不變。decoder 必須依完整狀態拓撲 fail closed，
    不可因總計數未改變就把缺列視為零。
    """

    metadata, aggregate_spec, products = _encoded_inputs()
    file_name = "outcome_counts.parquet"
    rows = [dict(row) for row in products.tables[file_name]]
    zero_row_index = next(index for index, row in enumerate(rows) if row["count"] == 0)
    removed = rows.pop(zero_row_index)
    assert removed["count"] == 0
    tampered = _tampered_products(
        products,
        table_overrides={file_name: rows},
    )

    with pytest.raises(ValueError, match="outcome_counts"):
        decode_aggregate_release_payload(metadata, aggregate_spec, tampered)
