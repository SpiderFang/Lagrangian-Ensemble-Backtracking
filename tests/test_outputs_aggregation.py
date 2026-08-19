"""ragged shard、checksum、KDE/HDR 與跨站分母測試。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from lagrangian_backtracking.aggregation import (
    boundary_arclength_histogram,
    conditional_kde_2d,
    cross_site_connectivity,
    outcome_summary,
    pathway_residence_grid,
)
from lagrangian_backtracking.engine import Observation, ParticleResult
from lagrangian_backtracking.models import BoundaryEvent, EventType, ParticleState, ParticleStatus
from lagrangian_backtracking.outputs import validate_trajectory_shard, write_trajectory_shard


def _result(particle_id: str, status: ParticleStatus) -> ParticleResult:
    """建立兩 observation 與一 terminal state 的小型輸出。"""

    state = ParticleState(
        particle_id=particle_id,
        scenario_id="s0",
        member_id=0,
        study_site_id="gongliao",
        analysis_region_id="A",
        receptor_id="r0",
        x_m=1.0,
        y_m=2.0,
        z_m=-3.0,
        time_utc_ns=10,
        age_seconds=10.0,
        status=status,
    )
    observations = [
        Observation(particle_id, 20, 0.0, 0.0, 0.0, -3.0, ParticleStatus.ACTIVE),
        Observation(particle_id, 10, 10.0, 1.0, 2.0, -3.0, status),
    ]
    return ParticleResult(state, observations, [], 1, 0)


def test_trajectory_shard_round_trip_and_checksum(tmp_path: Path) -> None:
    """原子發布後 CSR、Parquet row count 與所有 checksum 必須一致。"""

    destination = tmp_path / "shard-0001"
    write_trajectory_shard(
        destination,
        [_result("p0", ParticleStatus.MAX_AGE), _result("p1", ParticleStatus.FLOW_DOMAIN_EXIT)],
        run_metadata={"config_hash": "abc", "input_hash": "def", "seed_policy": "sha256_v1"},
    )
    validation = validate_trajectory_shard(destination)
    assert validation["valid"]
    assert validation["manifest"]["particle_count"] == 2
    assert validation["manifest"]["observation_count"] == 4
    formal = validate_trajectory_shard(destination, require_formal_metadata=True)
    assert not formal["valid"]
    assert any("missing_formal_fields" in error for error in formal["errors"])


def test_kde_probability_and_hdr_are_normalized() -> None:
    """KDE cell probability 總和為 1，較高 HDR 應包含較低 HDR。"""

    rng = np.random.default_rng(20260819)
    points = rng.normal(size=(500, 2)) * 1_000.0
    grid = conditional_kde_2d(
        points,
        x_edges_m=np.linspace(-5_000.0, 5_000.0, 101),
        y_edges_m=np.linspace(-5_000.0, 5_000.0, 101),
    )
    assert np.isclose(grid.cell_probability.sum(), 1.0)
    assert np.all(grid.hdr_masks[0.50] <= grid.hdr_masks[0.75])
    assert np.all(grid.hdr_masks[0.75] <= grid.hdr_masks[0.90])


def test_cross_site_fraction_uses_original_site_denominator_and_deduplicates() -> None:
    """同一 member 重複進入龜山島只計一次，分母仍是貢寮有效 members。"""

    state = _result("p0", ParticleStatus.MAX_AGE).final_state
    template = BoundaryEvent(
        particle_id=state.particle_id,
        scenario_id=state.scenario_id,
        member_id=state.member_id,
        study_site_id=state.study_site_id,
        analysis_region_id=state.analysis_region_id,
        receptor_id=state.receptor_id,
        event_type=EventType.OTHER_SITE_LOCAL_DOMAIN_ENTER,
        time_utc_ns=state.time_utc_ns,
        x_m=state.x_m,
        y_m=state.y_m,
        z_m=state.z_m,
        fraction=0.5,
        related_study_site_id="guishan",
    )
    rows = cross_site_connectivity(
        [template, replace(template, time_utc_ns=9)], valid_member_denominator_by_site={"gongliao": 10}
    )
    assert rows[0]["raw_unique_member_count"] == 1
    assert rows[0]["conditional_crossing_fraction"] == 0.1


def test_outcome_summary_keeps_raw_count_and_denominator() -> None:
    """停止比例必須與 raw count、共同 denominator 同列。"""

    rows = outcome_summary([ParticleStatus.MAX_AGE, ParticleStatus.MAX_AGE, ParticleStatus.DATA_GAP])
    max_age = next(row for row in rows if row["status"] == "max_age")
    assert max_age == {"status": "max_age", "raw_count": 2, "denominator": 3, "fraction": 2 / 3}


def test_pathway_grid_conserves_segment_time_and_unique_members() -> None:
    """跨兩格的直線段按格線切分秒數，unique count 不隨停留列數重複。"""

    result = _result("p-grid", ParticleStatus.MAX_AGE)
    result.observations = [
        Observation("p-grid", 100, 0.0, 0.25, 0.5, -1.0, ParticleStatus.ACTIVE),
        Observation("p-grid", 90, 10.0, 1.75, 0.5, -1.0, ParticleStatus.MAX_AGE),
    ]
    grid = pathway_residence_grid(
        [result],
        x_edges_m=np.array([0.0, 1.0, 2.0]),
        y_edges_m=np.array([0.0, 1.0]),
    )
    assert np.allclose(grid.residence_time_seconds, [[5.0, 5.0]])
    assert np.array_equal(grid.unique_particle_count, [[1, 1]])
    assert np.isclose(grid.input_interval_seconds, 10.0)
    assert np.isclose(grid.allocated_interval_seconds, 10.0)


def test_boundary_arclength_histogram_keeps_raw_count_and_denominator() -> None:
    """開放邊界一維產品同時保留 raw count、長度密度與 member 分母。"""

    state = _result("p0", ParticleStatus.MAX_AGE).final_state
    template = BoundaryEvent(
        particle_id=state.particle_id,
        scenario_id=state.scenario_id,
        member_id=state.member_id,
        study_site_id=state.study_site_id,
        analysis_region_id=state.analysis_region_id,
        receptor_id=state.receptor_id,
        event_type=EventType.LOCAL_DOMAIN_FIRST_EXIT,
        time_utc_ns=state.time_utc_ns,
        x_m=state.x_m,
        y_m=state.y_m,
        z_m=state.z_m,
        fraction=0.5,
        boundary_segment_id="gongliao-open",
    )
    events = [
        replace(template, particle_id=particle_id, boundary_s_m=value)
        for particle_id, value in (("p1", 25.0), ("p2", 75.0))
    ]
    histogram = boundary_arclength_histogram(
        events,
        boundary_segment_id="gongliao-open",
        s_edges_m=np.array([0.0, 50.0, 100.0]),
        valid_member_denominator=10,
    )
    assert np.array_equal(histogram.raw_count, [1, 1])
    assert np.allclose(histogram.count_density_per_m, [0.02, 0.02])
    assert np.isclose(np.sum(histogram.conditional_fraction_per_m * 50.0), 0.2)
