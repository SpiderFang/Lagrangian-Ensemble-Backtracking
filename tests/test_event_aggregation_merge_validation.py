"""事件聚合 chunk merge 的拓撲、重建驗證與整數溢位測試。

本檔只透過公開的 ``initialize_event_aggregate``、
``EventAggregateChunk`` 與 ``merge_event_aggregate_chunks`` 驗證 merge 邊界。
零值拓撲沿用核心事件聚合測試的兩站 fixture；測試不呼叫產品模組的私有
helper。所有空間陣列的軸順序固定為 ``(y_cell, x_cell)``，距離以公尺、旅行
年齡以秒表示。每個 fail-closed 案例都代表一個無法安全合併的資料契約違反，
不能以缺少的 key、不同格線或錯誤 shape 靜默補零。

最後的 overflow 案例刻意建立兩個各自合法的 chunk：同一個 local 網格 cell、
對應的邊界弧長 bin、旅行年齡 histogram、source-receptor histogram、各層
分母與 outcome 都同步設為 ``int64`` 最大值。單一 chunk 因而能成功封存，
但兩個 chunk 合併後必定超過固定寬度計數範圍；測試要求公開 merge API 回傳
``RuntimeError``，不可讓 NumPy 整數繞回成負值或其他錯誤計數。
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from test_event_aggregation_core import (
    _AGE_BIN_EDGES_SECONDS,
    _aggregate_spec,
)

from lagrangian_backtracking.event_aggregation import (
    BoundaryAggregateKey,
    CrossSiteAggregateKey,
    EventAggregateChunk,
    ReceptorAggregateKey,
    SiteEventGridCounts,
    SourceReceptorAggregateKey,
    initialize_event_aggregate,
    merge_event_aggregate_chunks,
)
from lagrangian_backtracking.models import ParticleStatus

_INT64_MAX = int(np.iinfo(np.int64).max)
_GRID_FIELDS = (
    "local_first_exit_count",
    "outer_first_exit_count",
    "bed_first_contact_count",
    "bed_repeated_contact_count",
    "data_gap_failure_count",
    "numerical_failure_count",
)
_SHARED_SEGMENT_ID = "shared-segment"


def _zero_chunk() -> EventAggregateChunk:
    """以核心兩站規格建立公開 API 可接受的完整零值聚合 chunk。

    ``initialize_event_aggregate`` 會建立兩站 2×2 公尺網格、local/outer 邊界
    key 與 ``0/10/20`` 秒年齡軸。零值只表示此測試 chunk 尚未收到粒子，並非把
    資料缺口、陸地、乾點或數值失敗當成物理零值；這個固定拓撲讓各個 mismatch
    案例只改動一個指定契約。
    """

    return initialize_event_aggregate(
        _aggregate_spec(),
        age_bin_edges_seconds=np.array(_AGE_BIN_EDGES_SECONDS, copy=True),
    )


def _replace_boundary_key(
    mapping: dict[object, object],
    old_key: BoundaryAggregateKey,
    new_key: BoundaryAggregateKey,
) -> dict[object, object]:
    """複製 mapping 並將一個公開邊界 key 改名，保留其陣列資料與 shape。"""

    return {
        new_key if key == old_key else key: value
        for key, value in mapping.items()
    }


@pytest.mark.parametrize(
    "chunks, error_pattern",
    (
        pytest.param((), r"chunks", id="empty-chunks"),
        pytest.param((object(),), r"chunks\[0\]", id="wrong-element-type"),
    ),
)
def test_merge_rejects_empty_or_non_chunk_input(
    chunks: tuple[object, ...],
    error_pattern: str,
) -> None:
    """空序列與非 ``EventAggregateChunk`` 元素都必須在 merge 前拒絕。

    merge 的輸入是已完成 shard 的資料容器；沒有任何 chunk 沒有可追溯的
    分母，而把任意物件當成 chunk 則可能繞過資料契約。兩種情況都必須以
    ``ValueError`` fail-closed，且非 chunk 錯誤需保留輸入位置。
    """

    with pytest.raises(ValueError, match=error_pattern):
        merge_event_aggregate_chunks(chunks)


def test_merge_rejects_age_axis_mismatch() -> None:
    """兩個合法容器若使用不同秒制 age 格線，merge 必須拒絕固定軸不一致。

    第二個 chunk 的旅行 histogram shape 仍合法，只把同樣數量的 age 邊界改成
    ``0/10/30`` 秒；因此這不是容器本身的 shape 錯誤，而是跨 chunk 的共同
    age 軸契約錯誤，錯誤訊息必須指出 ``chunks[1]``。
    """

    first = _zero_chunk()
    second = replace(
        first,
        age_bin_edges_seconds=np.array([0.0, 10.0, 30.0], dtype=np.float64),
    )

    with pytest.raises(ValueError, match=r"chunks\[1\].*age_bin_edges_seconds"):
        merge_event_aggregate_chunks((first, second))


def test_merge_rejects_site_grid_shape_mismatch() -> None:
    """同一 site 的六類網格 shape 不同時，merge 必須指出第二個 chunk。

    替換後的 ``SiteEventGridCounts`` 六個陣列彼此仍同形且都是合法二維
    ``int64``；只有它與第一個 chunk 的固定 2×2 網格不同，藉此隔離 merge
    層級的 topology validation。
    """

    first = _zero_chunk()
    bad_site_counts = SiteEventGridCounts(
        **{
            field: np.zeros((1, 2), dtype=np.int64)
            for field in _GRID_FIELDS
        }
    )
    site_grid_counts = dict(first.site_grid_counts)
    site_grid_counts["site-a"] = bad_site_counts
    second = replace(first, site_grid_counts=site_grid_counts)

    with pytest.raises(ValueError, match=r"chunks\[1\].*shape"):
        merge_event_aggregate_chunks((first, second))


@pytest.mark.parametrize(
    "mismatch_kind",
    (
        pytest.param("boundary-key", id="boundary-key"),
        pytest.param("boundary-edges", id="boundary-edges"),
        pytest.param("cross-site-key", id="cross-site-key"),
        pytest.param("outcome-status-key", id="outcome-status-key"),
    ),
)
def test_merge_rejects_fixed_topology_mismatch(mismatch_kind: str) -> None:
    """邊界、跨站與 outcome 內層 key 不一致時，merge 必須 fail-closed。

    ``boundary-key`` 與 ``boundary-edges`` 使用合法但與第一個 chunk 不同的
    固定空間拓撲；``cross-site-key`` 與 ``outcome-status-key`` 則加入值為零的
    合法 key，避免把錯誤歸因於非負性或守恆驗證。所有案例都只改動第二個
    chunk，故應在錯誤訊息中保留 ``chunks[1]`` 的定位資訊。
    """

    first = _zero_chunk()
    if mismatch_kind == "boundary-key":
        old_key = next(iter(first.boundary_bin_edges_m))
        new_key = BoundaryAggregateKey(
            old_key.study_site_id,
            old_key.boundary_kind,
            f"{old_key.boundary_segment_id}-different",
        )
        second = replace(
            first,
            boundary_bin_edges_m=_replace_boundary_key(
                dict(first.boundary_bin_edges_m),
                old_key,
                new_key,
            ),
            boundary_arclength_raw_count=_replace_boundary_key(
                dict(first.boundary_arclength_raw_count),
                old_key,
                new_key,
            ),
            boundary_travel_age_histogram=_replace_boundary_key(
                dict(first.boundary_travel_age_histogram),
                old_key,
                new_key,
            ),
        )
        expected_pattern = r"chunks\[1\].*boundary key"
    elif mismatch_kind == "boundary-edges":
        key = next(iter(first.boundary_bin_edges_m))
        boundary_edges = dict(first.boundary_bin_edges_m)
        boundary_edges[key] = np.array([0.0, 1.25, 2.5], dtype=np.float64)
        boundary_raw = dict(first.boundary_arclength_raw_count)
        boundary_raw[key] = np.zeros(2, dtype=np.int64)
        boundary_travel = dict(first.boundary_travel_age_histogram)
        boundary_travel[key] = np.zeros((2, 2), dtype=np.int64)
        second = replace(
            first,
            boundary_bin_edges_m=boundary_edges,
            boundary_arclength_raw_count=boundary_raw,
            boundary_travel_age_histogram=boundary_travel,
        )
        expected_pattern = r"chunks\[1\].*公尺格線"
    elif mismatch_kind == "cross-site-key":
        cross_site_counts = dict(first.cross_site_unique_member_count)
        cross_site_counts[CrossSiteAggregateKey("site-a", "site-b")] = 0
        second = replace(
            first,
            cross_site_unique_member_count=cross_site_counts,
        )
        expected_pattern = r"chunks\[1\].*cross-site key"
    else:
        outcome_count_by_site = {
            site_id: dict(outcomes)
            for site_id, outcomes in first.outcome_count_by_site.items()
        }
        outcome_count_by_site["site-a"][ParticleStatus.MAX_AGE.value] = 0
        second = replace(
            first,
            outcome_count_by_site=outcome_count_by_site,
        )
        expected_pattern = r"chunks\[1\].*outcome status key"

    with pytest.raises(ValueError, match=expected_pattern):
        merge_event_aggregate_chunks((first, second))


@pytest.mark.parametrize(
    "field_name",
    (
        pytest.param("age_bin_edges_seconds", id="age-axis"),
        pytest.param("site_grid_counts", id="site-grid-mapping"),
        pytest.param("boundary_arclength_raw_count", id="boundary-raw"),
    ),
)
def test_merge_revalidates_corrupted_frozen_chunk_and_reports_index(
    field_name: str,
) -> None:
    """即使 frozen chunk 被 ``object.__setattr__`` 污染，merge 仍須重新驗證。

    ``EventAggregateChunk`` 的 frozen 只保護一般屬性指派，並不能防止測試、
    外部 reader 或不可信反序列化程式刻意竄改公開欄位。本測試直接污染第二個
    chunk：age 軸改為非遞增、site mapping 放入錯誤型別，或 boundary raw 放入
    負計數。merge 必須從公開欄位重建容器而非信任既有物件，並在錯誤中指出
    ``chunks[1]``；任何悄悄使用污染資料都會讓正式 release 無法定位壞 shard。
    """

    first = _zero_chunk()
    corrupted = _zero_chunk()
    if field_name == "age_bin_edges_seconds":
        object.__setattr__(
            corrupted,
            field_name,
            np.array([0.0, 10.0, 5.0], dtype=np.float64),
        )
    elif field_name == "site_grid_counts":
        object.__setattr__(
            corrupted,
            field_name,
            {"site-a": "not-a-site-grid", "site-b": corrupted.site_grid_counts["site-b"]},
        )
    else:
        boundary_raw = {
            key: np.array(value, copy=True)
            for key, value in corrupted.boundary_arclength_raw_count.items()
        }
        first_key = next(iter(boundary_raw))
        boundary_raw[first_key][0] = -1
        object.__setattr__(corrupted, field_name, boundary_raw)

    with pytest.raises(ValueError, match=r"chunks\[1\]"):
        merge_event_aggregate_chunks((first, corrupted))


def _legal_int64_max_chunk() -> EventAggregateChunk:
    """建立同一 local cell 為 int64 上限且所有守恆欄位同步的合法 chunk。

    這個 chunk 只有 site-a 的 local 首次離開資料：網格、local 邊界 raw、
    ``(s, age)`` histogram、source-receptor raw/age histogram、site/receptor
    valid 分母、site total、MAX_AGE outcome 及 input particle count 全部代表
    同一個 ``_INT64_MAX`` 成員數。site-b 與 outer/bed/failure/cross-site 欄位
    保留合法零值，因此容器本身必須成功建構；只有跨 chunk 加總才會溢位。
    """

    base = _zero_chunk()
    local_key = BoundaryAggregateKey("site-a", "local", _SHARED_SEGMENT_ID)
    source_key = SourceReceptorAggregateKey(
        "site-a",
        "receptor-overflow",
        "local",
        _SHARED_SEGMENT_ID,
    )
    receptor_key = ReceptorAggregateKey("site-a", "receptor-overflow")

    site_grid_counts: dict[str, SiteEventGridCounts] = {}
    for site_id, counts in base.site_grid_counts.items():
        arrays = {
            field: np.array(getattr(counts, field), copy=True)
            for field in _GRID_FIELDS
        }
        if site_id == "site-a":
            arrays["local_first_exit_count"][0, 0] = _INT64_MAX
        site_grid_counts[site_id] = SiteEventGridCounts(**arrays)

    boundary_raw = {
        key: np.array(value, copy=True)
        for key, value in base.boundary_arclength_raw_count.items()
    }
    boundary_raw[local_key][0] = _INT64_MAX
    boundary_travel = {
        key: np.array(value, copy=True)
        for key, value in base.boundary_travel_age_histogram.items()
    }
    boundary_travel[local_key][0, 0] = _INT64_MAX

    outcome_count_by_site = {
        "site-a": {ParticleStatus.MAX_AGE.value: _INT64_MAX},
        "site-b": {},
    }
    return EventAggregateChunk(
        site_grid_counts=site_grid_counts,
        boundary_bin_edges_m=dict(base.boundary_bin_edges_m),
        boundary_arclength_raw_count=boundary_raw,
        boundary_travel_age_histogram=boundary_travel,
        age_bin_edges_seconds=np.array(base.age_bin_edges_seconds, copy=True),
        source_receptor_raw_count={source_key: _INT64_MAX},
        source_receptor_travel_age_histogram={
            source_key: np.array([_INT64_MAX, 0], dtype=np.int64),
        },
        cross_site_unique_member_count={},
        outcome_count_by_site=outcome_count_by_site,
        valid_member_denominator_by_site={"site-a": _INT64_MAX, "site-b": 0},
        total_member_count_by_site={"site-a": _INT64_MAX, "site-b": 0},
        valid_member_denominator_by_receptor={receptor_key: _INT64_MAX},
        input_particle_count=_INT64_MAX,
    )


def test_merge_rejects_elementwise_int64_overflow_without_wraparound() -> None:
    """兩個合法上限 chunk 相加時必須以 RuntimeError 拒絕固定寬度溢位。

    測試故意讓兩個 chunk 的同一個 ``(y=0, x=0)`` local cell 都等於
    ``int64.max``，並同步所有 boundary/source/receptor/outcome 守恆欄位。
    若 merge 直接使用 NumPy 固定寬度加法，結果可能繞回負值；公開 API 必須在
    寫回 int64 前逐元素檢查，回傳可稽核的 ``RuntimeError``。
    """

    first = _legal_int64_max_chunk()
    second = _legal_int64_max_chunk()

    with pytest.raises(RuntimeError, match="int64"):
        merge_event_aggregate_chunks((first, second))
