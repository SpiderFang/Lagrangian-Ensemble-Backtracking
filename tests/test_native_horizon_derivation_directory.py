"""WP-17 任務 2：``native_runner.horizon_derivation.derive_result_directory`` 與
``lbt native-derive-horizon`` CLI 的目錄層級測試。

只用合成 ``BackwardDriverResult``／``.npz``（不需要 ``LBT_BAYTRACE_ROOT``）：
``derive_result_directory`` 只做陣列切片與檔案 I/O，不呼叫任何 BayTrace 核心。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from lagrangian_backtracking import cli
from lagrangian_backtracking.native_runner import backward_driver as bd
from lagrangian_backtracking.native_runner import horizon_derivation as hd
from lagrangian_backtracking.native_runner.hour_major_provider import (
    load_backward_driver_result_npz,
    save_backward_driver_result_npz,
)

_HOUR_NS = 3_600_000_000_000


def _synthetic_result(*, arrival_utc_ns: int, n_hours: int, n_particles: int = 2) -> bd.BackwardDriverResult:
    """一個「全部粒子都撐到 n_hours 小時回溯上限」的合成結果。"""

    trajectory_utc = np.array(
        [arrival_utc_ns - h * _HOUR_NS for h in range(n_hours + 1)], dtype=np.int64
    )
    trajectory_x = np.stack(
        [np.full(n_particles, float(h)) for h in range(n_hours + 1)]
    )
    trajectory_y = -trajectory_x.copy()
    trajectory_z = np.full((n_hours + 1, n_particles), -5.0)
    trajectory_status = np.zeros((n_hours + 1, n_particles), dtype=np.int32)
    departures = tuple(
        bd.DepartureRecord(
            row_particle_index=row, status=bd.NATIVE_STATUS_MAX_BACKTRACK,
            x_m=float(n_hours), y_m=-float(n_hours), z_m=-5.0,
            utc_ns=arrival_utc_ns - n_hours * _HOUR_NS,
        )
        for row in range(n_particles)
    )
    return bd.BackwardDriverResult(
        row_particle_index=np.arange(n_particles, dtype=np.int64),
        final_status=np.full(n_particles, bd.NATIVE_STATUS_MAX_BACKTRACK, dtype=np.int32),
        final_x_m=np.full(n_particles, float(n_hours)),
        final_y_m=np.full(n_particles, -float(n_hours)),
        final_z_m=np.full(n_particles, -5.0),
        final_utc_ns=np.full(n_particles, arrival_utc_ns - n_hours * _HOUR_NS, dtype=np.int64),
        trajectory_utc_ns=trajectory_utc,
        trajectory_x_m=trajectory_x,
        trajectory_y_m=trajectory_y,
        trajectory_z_m=trajectory_z,
        trajectory_status=trajectory_status,
        departures=departures,
        diagnostics=bd.SubstepDiagnostics(),
        overhead_diagnostics=bd.HourlyOverheadDiagnostics(),
        hours_run=n_hours,
    )


def test_derive_result_directory_writes_derived_npz_and_manifest(tmp_path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    result = _synthetic_result(arrival_utc_ns=100_000 * _HOUR_NS, n_hours=6)
    save_backward_driver_result_npz(source_dir / "unit-a.driver_result.npz", result)

    output_dir = tmp_path / "derived"
    manifest = hd.derive_result_directory(
        source_results_dir=source_dir,
        output_dir=output_dir,
        source_backtrack_mode="full_horizon_from_deposition",
        source_requested_horizon_days=6 / 24,
        target_backtrack_mode="full_horizon_from_deposition",
        target_requested_horizon_days=3 / 24,
    )
    assert manifest["unit_count"] == 1
    assert manifest["units"]["unit-a"]["newly_capped_particle_count"] == 2
    assert manifest["units"]["unit-a"]["hours_run"] == 3
    assert manifest["derivation_method"] == "truncate_existing_full_horizon_from_deposition_result_v1"

    derived = load_backward_driver_result_npz(output_dir / "unit-a.driver_result.npz")
    assert derived.hours_run == 3
    assert np.all(derived.final_status == bd.NATIVE_STATUS_MAX_BACKTRACK)
    assert np.all(derived.final_x_m == 3.0)

    on_disk_manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk_manifest == manifest

    # 來源檔完全不受影響。
    untouched = load_backward_driver_result_npz(source_dir / "unit-a.driver_result.npz")
    assert untouched.hours_run == 6


def test_derive_result_directory_rejects_fixed_calendar_window_source(tmp_path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    result = _synthetic_result(arrival_utc_ns=100_000 * _HOUR_NS, n_hours=6)
    save_backward_driver_result_npz(source_dir / "unit-a.driver_result.npz", result)

    with pytest.raises(hd.HorizonDerivationError):
        hd.derive_result_directory(
            source_results_dir=source_dir,
            output_dir=tmp_path / "derived",
            source_backtrack_mode="fixed_calendar_window",
            source_requested_horizon_days=6 / 24,
            target_backtrack_mode="full_horizon_from_deposition",
            target_requested_horizon_days=3 / 24,
        )


def test_derive_result_directory_rejects_empty_source(tmp_path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    with pytest.raises(hd.HorizonDerivationError):
        hd.derive_result_directory(
            source_results_dir=source_dir,
            output_dir=tmp_path / "derived",
            source_backtrack_mode="full_horizon_from_deposition",
            source_requested_horizon_days=6 / 24,
            target_backtrack_mode="full_horizon_from_deposition",
            target_requested_horizon_days=3 / 24,
        )


def test_cli_native_derive_horizon_end_to_end(tmp_path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    result = _synthetic_result(arrival_utc_ns=100_000 * _HOUR_NS, n_hours=6)
    save_backward_driver_result_npz(source_dir / "unit-a.driver_result.npz", result)

    output_dir = tmp_path / "derived"
    rc = cli.run_native_derive_horizon_command(
        [
            "--source-results-dir", str(source_dir),
            "--source-backtrack-mode", "full_horizon_from_deposition",
            "--source-horizon-days", str(6 / 24),
            "--target-backtrack-mode", "full_horizon_from_deposition",
            "--target-horizon-days", str(3 / 24),
            "--output", str(output_dir),
        ]
    )
    assert rc == 0
    assert (output_dir / "unit-a.driver_result.npz").is_file()
    assert (output_dir / "manifest.json").is_file()


def test_cli_native_derive_horizon_requires_arrival_manifest_for_fixed_calendar_window(tmp_path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    result = _synthetic_result(arrival_utc_ns=100_000 * _HOUR_NS, n_hours=6)
    save_backward_driver_result_npz(source_dir / "unit-a.driver_result.npz", result)

    with pytest.raises(ValueError):
        cli.run_native_derive_horizon_command(
            [
                "--source-results-dir", str(source_dir),
                "--source-backtrack-mode", "full_horizon_from_deposition",
                "--source-horizon-days", str(6 / 24),
                "--target-backtrack-mode", "fixed_calendar_window",
                "--target-horizon-days", str(3 / 24),
                "--output", str(tmp_path / "derived"),
            ]
        )
