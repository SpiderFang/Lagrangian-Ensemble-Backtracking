"""跨月份 UTC canonicalization、來源映射與缺口長度測試。"""

from __future__ import annotations

import numpy as np
import pytest

from lagrangian_backtracking.time_axis import TimeChunk, canonicalize_time_chunks

HOUR_NS = 3_600_000_000_000


def test_prefer_last_stably_sorts_and_maps_duplicate_halo() -> None:
    """倒序月份 halo 的同 UTC 必須保留後輸入 chunk，並保存正確 local index。"""

    chunks = [
        TimeChunk("202401", np.array([0, HOUR_NS, 2 * HOUR_NS], dtype=np.int64)),
        TimeChunk("202402", np.array([HOUR_NS, 2 * HOUR_NS, 3 * HOUR_NS], dtype=np.int64)),
    ]
    result = canonicalize_time_chunks(
        chunks,
        policy="sort_and_deduplicate_prefer_last",
        expected_timestep_hours=1.0,
    )
    assert np.array_equal(result.time_utc_ns, np.arange(4, dtype=np.int64) * HOUR_NS)
    assert np.array_equal(result.source_chunk_index, [0, 1, 1, 1])
    assert np.array_equal(result.source_local_index, [0, 0, 1, 2])
    assert result.input_time_count == 6
    assert result.dropped_duplicate_time_step_count == 2
    assert not result.gaps


def test_gap_count_means_missing_slots_not_endpoint_distance() -> None:
    """23:00→01:00 是缺一筆逐時樣本，避免報表把 2 小時誤稱為缺兩筆。"""

    result = canonicalize_time_chunks(
        [TimeChunk("202401", np.array([0, HOUR_NS, 3 * HOUR_NS, 28 * HOUR_NS], dtype=np.int64))],
        policy="sort_and_deduplicate_prefer_last",
        expected_timestep_hours=1.0,
    )
    assert [item.gap_hours for item in result.gaps] == [2.0, 25.0]
    assert [item.missing_step_count for item in result.gaps] == [1, 24]
    assert result.missing_step_count == 25
    assert result.maximum_gap_hours == 25.0


def test_reject_policy_does_not_silently_accept_overlap() -> None:
    """嚴格科學 run 遇到跨 chunk 重複 UTC 時必須拒絕，不可暗中套用 available policy。"""

    with pytest.raises(ValueError, match="倒序或重複"):
        canonicalize_time_chunks(
            [
                TimeChunk("a", np.array([0, HOUR_NS], dtype=np.int64)),
                TimeChunk("b", np.array([HOUR_NS, 2 * HOUR_NS], dtype=np.int64)),
            ],
            policy="reject",
            expected_timestep_hours=1.0,
        )


def test_single_chunk_must_be_strictly_increasing() -> None:
    """月檔內部重複不是合法 halo，必須由上游重新產製或明示修復。"""

    with pytest.raises(ValueError, match="嚴格遞增"):
        canonicalize_time_chunks(
            [TimeChunk("bad", np.array([0, HOUR_NS, HOUR_NS], dtype=np.int64))],
            policy="sort_and_deduplicate_prefer_last",
            expected_timestep_hours=1.0,
        )
