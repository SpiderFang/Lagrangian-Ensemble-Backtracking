"""ParticleBatch SoA 資料契約、壓縮與安全回寫測試。"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from lagrangian_backtracking.batch_state import (
    PARTICLE_CODE_TO_STATUS,
    PARTICLE_STATUS_TO_CODE,
    ParticleBatch,
)
from lagrangian_backtracking.models import ParticleState, ParticleStatus


def _state(index: int, status: ParticleStatus = ParticleStatus.ACTIVE) -> ParticleState:
    """建立含完整身分欄位的測試粒子，避免測試只覆蓋位置欄位而漏掉資料契約。"""

    return ParticleState(
        particle_id=f"particle-{index}",
        scenario_id=f"scenario-{index // 2}",
        member_id=100 + index,
        study_site_id=f"site-{index % 2}",
        analysis_region_id=f"region-{index % 3}",
        receptor_id=f"receptor-{index % 4}",
        x_m=10.0 + index,
        y_m=-20.0 - index,
        z_m=-3.5 - index / 10.0,
        time_utc_ns=1_700_000_000_000_000_000 + index,
        age_seconds=60.0 + index,
        status=status,
        own_local_exit_recorded=bool(index % 2),
    )


def _batch() -> ParticleBatch:
    """建立含 active 與終止粒子的批次，供 compaction/scatter 測試重複使用。"""

    statuses = (
        ParticleStatus.COAST_CONTACT,
        ParticleStatus.ACTIVE,
        ParticleStatus.MAX_AGE,
        ParticleStatus.ACTIVE,
    )
    return ParticleBatch.from_particle_states(_state(i, statuses[i]) for i in range(4))


def test_status_codebook_is_explicit_and_round_trips_all_statuses() -> None:
    """所有 ParticleStatus 都使用固定整數並可無損還原，不依賴 enum/hash 順序。"""

    statuses = tuple(ParticleStatus)
    states = [_state(index, status) for index, status in enumerate(statuses)]
    batch = ParticleBatch.from_particle_states(states, triangle_hints=np.arange(len(states)))

    assert {status: index for index, status in enumerate(statuses)} == PARTICLE_STATUS_TO_CODE
    assert {index: status for index, status in enumerate(statuses)} == PARTICLE_CODE_TO_STATUS
    assert np.array_equal(batch.status_code, np.arange(len(statuses), dtype=np.int64))
    assert np.array_equal(batch.triangle_hint, np.arange(len(statuses), dtype=np.int64))
    assert batch.to_particle_states() == states


def test_particle_batch_round_trip_preserves_all_particle_state_fields() -> None:
    """一般 batch 往返後，既有 ParticleState 的每一個欄位都保持原值。"""

    states = [_state(0), _state(1, ParticleStatus.DATA_GAP)]
    batch = ParticleBatch.from_particle_states(states, triangle_hints=[-1, 17])

    assert len(batch) == 2
    assert batch.to_particle_states() == states
    assert batch.triangle_hint.tolist() == [-1, 17]
    assert batch.particle_id == ("particle-0", "particle-1")
    assert batch.member_id.tolist() == [100, 101]


def test_empty_particle_batch_is_valid_and_round_trips() -> None:
    """零粒子與零 active compaction 都是合法狀態，且不需虛構一筆資料。"""

    batch = ParticleBatch.from_particle_states([])
    assert len(batch) == 0
    assert batch.to_particle_states() == []
    compacted, source_indices = batch.compact_active()
    assert len(compacted) == 0
    assert source_indices.dtype == np.int64
    assert source_indices.size == 0
    batch.scatter_dynamic_from(compacted, source_indices)


def test_particle_batch_has_required_shapes_and_dtypes() -> None:
    """每個欄位都是一維等長陣列，並使用固定的資料型別。"""

    batch = _batch()
    assert len(batch) == 4
    assert all(getattr(batch, name).shape == (4,) for name in batch._ARRAY_DTYPES)
    assert batch.member_id.dtype == np.int64
    assert batch.x_m.dtype == np.float64
    assert batch.y_m.dtype == np.float64
    assert batch.z_m.dtype == np.float64
    assert batch.age_seconds.dtype == np.float64
    assert batch.time_utc_ns.dtype == np.int64
    assert batch.status_code.dtype == np.int64
    assert batch.own_local_exit_recorded.dtype == np.bool_
    assert batch.triangle_hint.dtype == np.int64
    assert all(getattr(batch, name).flags.c_contiguous for name in batch._ARRAY_DTYPES)

    with pytest.raises(TypeError, match="status_code dtype"):
        replace(batch, status_code=batch.status_code.astype(np.int32))
    with pytest.raises(ValueError, match="triangle_hint"):
        replace(batch, triangle_hint=np.full(len(batch), -2, dtype=np.int64))


def test_slice_view_shares_contiguous_numeric_storage() -> None:
    """slice_view 的數值欄位共享底層記憶體，修改 view 會反映原批次。"""

    batch = _batch()
    view = batch.slice_view(1, 3)

    assert view.particle_id == batch.particle_id[1:3]
    for field_name in batch._ARRAY_DTYPES:
        assert np.shares_memory(getattr(batch, field_name), getattr(view, field_name))
        assert getattr(view, field_name).flags.c_contiguous
    view.x_m[0] = 999.0
    view.status_code[1] = PARTICLE_STATUS_TO_CODE[ParticleStatus.NUMERICAL_FAILURE]
    assert batch.x_m[1] == 999.0
    assert batch.status_code[2] == PARTICLE_STATUS_TO_CODE[ParticleStatus.NUMERICAL_FAILURE]


def test_active_indices_and_compaction_keep_source_order_and_identity() -> None:
    """active 索引與壓縮結果都維持來源順序，且壓縮陣列是獨立連續複本。"""

    batch = _batch()
    active_indices = batch.active_indices()
    compacted, source_indices = batch.compact_active()

    assert np.array_equal(active_indices, [1, 3])
    assert np.array_equal(source_indices, [1, 3])
    assert compacted.particle_id == ("particle-1", "particle-3")
    assert compacted.member_id.tolist() == [101, 103]
    assert all(
        not np.shares_memory(getattr(batch, field_name), getattr(compacted, field_name))
        for field_name in batch._ARRAY_DTYPES
    )


def test_scatter_dynamic_fields_writes_only_matching_source_particles() -> None:
    """scatter 只回寫動態欄位，且不改變來源身分與非 active 粒子。"""

    batch = _batch()
    original_identity = tuple(
        tuple(getattr(batch, field_name)) for field_name in batch._IDENTITY_FIELDS
    )
    original_member_id = batch.member_id.copy()
    compacted, source_indices = batch.compact_active()
    compacted.x_m += 100.0
    compacted.y_m -= 100.0
    compacted.z_m = np.array([-8.0, -9.0], dtype=np.float64)
    compacted.age_seconds += 5.0
    compacted.time_utc_ns += 10
    compacted.status_code[:] = PARTICLE_STATUS_TO_CODE[ParticleStatus.MAX_AGE]
    compacted.own_local_exit_recorded[:] = True
    compacted.triangle_hint[:] = [7, 8]

    batch.scatter_dynamic_from(compacted, source_indices)

    assert batch.x_m.tolist() == [10.0, 111.0, 12.0, 113.0]
    assert batch.y_m.tolist() == [-20.0, -121.0, -22.0, -123.0]
    assert batch.z_m.tolist() == [-3.5, -8.0, -3.7, -9.0]
    assert batch.status_code.tolist() == [2, 7, 7, 7]
    assert batch.own_local_exit_recorded.tolist() == [False, True, False, True]
    assert batch.triangle_hint.tolist() == [-1, 7, -1, 8]
    assert (
        tuple(tuple(getattr(batch, field_name)) for field_name in batch._IDENTITY_FIELDS)
        == original_identity
    )
    assert np.array_equal(batch.member_id, original_member_id)


def test_scatter_rejects_identity_mismatch_duplicate_and_out_of_range_indices() -> None:
    """錯誤身分、重複索引與越界索引都在任何動態回寫前被拒絕。"""

    batch = _batch()
    compacted, source_indices = batch.compact_active()

    wrong_identity = replace(compacted, particle_id=("wrong", "particle-3"))
    with pytest.raises(ValueError, match="particle_id"):
        batch.scatter_dynamic_from(wrong_identity, source_indices)

    wrong_member = replace(compacted, member_id=np.array([999, 103], dtype=np.int64))
    with pytest.raises(ValueError, match="member_id"):
        batch.scatter_dynamic_from(wrong_member, source_indices)

    with pytest.raises(ValueError, match="唯一"):
        batch.scatter_dynamic_from(compacted, [1, 1])
    with pytest.raises(IndexError, match="範圍"):
        batch.scatter_dynamic_from(compacted, [1, len(batch)])
