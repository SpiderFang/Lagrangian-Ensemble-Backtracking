"""checkpoint binding 與 48+2 arrival selector 測試。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from lagrangian_backtracking.arrival_times import select_arrival_times
from lagrangian_backtracking.checkpoint import CheckpointBinding, load_checkpoint, write_checkpoint
from lagrangian_backtracking.models import ParticleState


def _particle(particle_id: str) -> ParticleState:
    """建立 checkpoint 使用的 active particle。"""

    return ParticleState(
        particle_id=particle_id,
        scenario_id="s0",
        member_id=0,
        study_site_id="gongliao",
        analysis_region_id="A",
        receptor_id="r0",
        x_m=1.0,
        y_m=2.0,
        z_m=-3.0,
        time_utc_ns=1_704_067_200_000_000_000,
    )


def test_checkpoint_round_trip_and_binding_rejection(tmp_path: Path) -> None:
    """相同 binding 可恢復；config hash 改變必須拒絕續跑。"""

    binding = CheckpointBinding("config-a", "input-a", "baseline", "shard-1", "sha256-v1", "abc123")
    path = write_checkpoint(
        tmp_path / "checkpoint-0001", binding=binding, states=[_particle("p0")], sequence=1
    )
    states, sequence = load_checkpoint(path, expected_binding=binding)
    assert sequence == 1 and states == [_particle("p0")]
    incompatible = CheckpointBinding("config-b", "input-a", "baseline", "shard-1", "sha256-v1", "abc123")
    with pytest.raises(ValueError, match="binding 不相容"):
        load_checkpoint(path, expected_binding=incompatible)


def test_arrival_selector_produces_48_plus_2_unique_times() -> None:
    """兩年逐三小時合成潮位應完整覆蓋 2×4×2×3 核心與兩個事件。"""

    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2026, 1, 1, tzinfo=UTC)
    count = int((end - start) / timedelta(hours=3))
    time_ns = np.array(
        [int((start + timedelta(hours=3 * index)).timestamp() * 1_000_000_000) for index in range(count)],
        dtype=np.int64,
    )
    hours = np.arange(count, dtype=np.float64) * 3.0
    spring_neap = 1.0 + 0.5 * np.sin(2.0 * np.pi * hours / (24.0 * 14.0))
    elevation = spring_neap * np.sin(2.0 * np.pi * hours / 12.42)
    wave = 1.0 + 0.2 * np.sin(2.0 * np.pi * hours / (24.0 * 7.0))
    current = 0.5 + 0.1 * np.cos(2.0 * np.pi * hours / 12.42)
    wave[1000] = 8.0
    current[2000] = 3.0
    records = select_arrival_times(
        study_site_id="gongliao",
        time_utc_ns=time_ns,
        elevation_m=elevation,
        significant_wave_height_m=wave,
        current_speed_mps=current,
        valid_forcing=np.ones(count, dtype=bool),
        backward_window_available=np.ones(count, dtype=bool),
        design_version="design_baseline_v1",
    )
    assert len(records) == 50
    assert len({item.time_utc_ns for item in records}) == 50
    assert sum(item.tide_class != "event" for item in records) == 48
    assert {item.phase_or_event for item in records if item.tide_class == "event"} == {
        "high_wave_event",
        "strong_current_event",
    }
