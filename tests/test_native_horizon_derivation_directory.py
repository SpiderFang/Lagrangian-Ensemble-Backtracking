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
from lagrangian_backtracking.native_runner.backtrack_horizon import ResolvedUnitHorizon
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


# ══════════════════════════════════════════════════════════════════════════
# WP-28 任務 A：local domain 相關欄位（``local_domain_departures``／
# ``also_local_domain_first_exit``）的截斷語意
#
# 見 ``horizon_derivation.py`` 模組文件「local domain 相關欄位的截斷語意」一節
# 的完整科學論證；這裡直接測 ``derive_backward_driver_result``（純陣列函式，
# 不需要 ``LBT_BAYTRACE_ROOT``），用手造欄位精確控制「事件發生在截止時刻前／
# 後」與「終止事件是否改變」，比透過真實子步 kernel 更能精確覆蓋邊界情況。
# ══════════════════════════════════════════════════════════════════════════


def _resolved_full_horizon(*, requested_horizon_days: float, effective_horizon_seconds: float) -> ResolvedUnitHorizon:
    return ResolvedUnitHorizon(
        backtrack_mode="full_horizon_from_deposition",
        requested_horizon_days=requested_horizon_days,
        effective_horizon_seconds=effective_horizon_seconds,
        effective_horizon_ns=int(round(effective_horizon_seconds * 1_000_000_000)),
        pre_window_deposition=False,
        bed_residence_age_hours=None,
    )


def _synthetic_result_with_local_domain_fields(
    *,
    arrival_utc_ns: int,
    n_hours: int,
    n_particles: int = 2,
    local_domain_departures: tuple[bd.LocalDomainFirstExitRecord, ...] = (),
    departures: tuple[bd.DepartureRecord, ...] | None = None,
    final_status: np.ndarray | None = None,
    final_utc_ns: np.ndarray | None = None,
) -> bd.BackwardDriverResult:
    """比 ``_synthetic_result`` 更靈活的合成結果建構器。

    可自訂 ``local_domain_departures``（非終止的 local 首次離開事件列表）與
    逐顆粒子的 ``departures``／``final_status``／``final_utc_ns``，用來精確
    佈置「事件發生在截止時刻之前／之後」「終止原因是否因截斷而改變」等情境，
    不受真實子步物理的隨機性影響。省略時的預設值與 ``_synthetic_result`` 相同
    （全部粒子撐到 ``n_hours`` 小時回溯上限）。
    """

    trajectory_utc = np.array(
        [arrival_utc_ns - h * _HOUR_NS for h in range(n_hours + 1)], dtype=np.int64
    )
    trajectory_x = np.stack([np.full(n_particles, float(h)) for h in range(n_hours + 1)])
    trajectory_y = -trajectory_x.copy()
    trajectory_z = np.full((n_hours + 1, n_particles), -5.0)
    trajectory_status = np.zeros((n_hours + 1, n_particles), dtype=np.int32)
    if final_status is None:
        final_status = np.full(n_particles, bd.NATIVE_STATUS_MAX_BACKTRACK, dtype=np.int32)
    if final_utc_ns is None:
        final_utc_ns = np.full(n_particles, arrival_utc_ns - n_hours * _HOUR_NS, dtype=np.int64)
    if departures is None:
        departures = tuple(
            bd.DepartureRecord(
                row_particle_index=row, status=int(final_status[row]),
                x_m=float(n_hours), y_m=-float(n_hours), z_m=-5.0,
                utc_ns=int(final_utc_ns[row]),
            )
            for row in range(n_particles)
        )
    return bd.BackwardDriverResult(
        row_particle_index=np.arange(n_particles, dtype=np.int64),
        final_status=final_status,
        final_x_m=np.full(n_particles, float(n_hours)),
        final_y_m=np.full(n_particles, -float(n_hours)),
        final_z_m=np.full(n_particles, -5.0),
        final_utc_ns=final_utc_ns,
        trajectory_utc_ns=trajectory_utc,
        trajectory_x_m=trajectory_x,
        trajectory_y_m=trajectory_y,
        trajectory_z_m=trajectory_z,
        trajectory_status=trajectory_status,
        departures=departures,
        diagnostics=bd.SubstepDiagnostics(),
        overhead_diagnostics=bd.HourlyOverheadDiagnostics(),
        hours_run=n_hours,
        local_domain_departures=local_domain_departures,
    )


def test_local_domain_departures_retained_only_at_or_before_cutoff() -> None:
    """``local_equals_flow=False`` 測站（貢寮／龜山島情境）：截止時刻或更早
    （短 H 範圍內確實已發生）的首次離開 local domain 事件必須保留；截止時刻
    之後（短 H 尚未走到）的事件必須捨棄，恰好落在截止時刻上的事件視為保留
    （屬於短 H 可達範圍的邊界情況）。"""

    t0 = 100_000 * _HOUR_NS
    kept_before_cutoff = bd.LocalDomainFirstExitRecord(
        row_particle_index=0, x_m=1.0, y_m=-1.0, z_m=-5.0, utc_ns=t0 - 2 * _HOUR_NS,
    )
    kept_exactly_at_cutoff = bd.LocalDomainFirstExitRecord(
        row_particle_index=1, x_m=2.0, y_m=-2.0, z_m=-5.0, utc_ns=t0 - 3 * _HOUR_NS,
    )
    dropped_after_cutoff = bd.LocalDomainFirstExitRecord(
        row_particle_index=2, x_m=3.0, y_m=-3.0, z_m=-5.0, utc_ns=t0 - 4 * _HOUR_NS,
    )
    result = _synthetic_result_with_local_domain_fields(
        arrival_utc_ns=t0, n_hours=6, n_particles=3,
        local_domain_departures=(kept_before_cutoff, kept_exactly_at_cutoff, dropped_after_cutoff),
    )
    resolved = _resolved_full_horizon(requested_horizon_days=3 / 24, effective_horizon_seconds=3 * 3600.0)

    derived, info = hd.derive_backward_driver_result(
        result, resolved=resolved, source_requested_horizon_days=6 / 24,
    )

    retained_by_row = {r.row_particle_index: r for r in derived.local_domain_departures}
    assert set(retained_by_row) == {0, 1}, "只有截止時刻或更早的事件應該保留"
    assert retained_by_row[0] == kept_before_cutoff
    assert retained_by_row[1] == kept_exactly_at_cutoff
    assert info.local_domain_departure_count == 2


def test_local_domain_departures_all_dropped_when_all_after_cutoff() -> None:
    """全部 local 首次離開事件都發生在截止時刻之後時，推導結果必須是空 tuple
    （不是保留任何一筆），避免「短 H 範圍內根本沒發生過的事件」出現在結果裡。"""

    t0 = 100_000 * _HOUR_NS
    result = _synthetic_result_with_local_domain_fields(
        arrival_utc_ns=t0, n_hours=6, n_particles=2,
        local_domain_departures=(
            bd.LocalDomainFirstExitRecord(row_particle_index=0, x_m=1.0, y_m=-1.0, z_m=-5.0, utc_ns=t0 - 5 * _HOUR_NS),
            bd.LocalDomainFirstExitRecord(row_particle_index=1, x_m=2.0, y_m=-2.0, z_m=-5.0, utc_ns=t0 - 6 * _HOUR_NS),
        ),
    )
    resolved = _resolved_full_horizon(requested_horizon_days=3 / 24, effective_horizon_seconds=3 * 3600.0)

    derived, info = hd.derive_backward_driver_result(
        result, resolved=resolved, source_requested_horizon_days=6 / 24,
    )

    assert derived.local_domain_departures == ()
    assert info.local_domain_departure_count == 0


def test_also_local_domain_first_exit_preserved_before_cutoff_and_redetermined_after() -> None:
    """``local_equals_flow=True`` 測站（新竹／南灣／連江情境）：

    - 粒子的「離開 flow domain」終止事件發生在截止時刻之前（不需要重新截斷）：
      ``also_local_domain_first_exit`` 原樣保留。
    - 粒子的終止事件發生在截止時刻之後（短 H 範圍內根本還沒發生，需要重新
      截斷成「回溯達上限」）：即使來源記錄的 ``also_local_domain_first_exit``
      是 ``True``，也必須重新判定為 ``False``——這正是任務要求「旗標語意要
      跟著重新判定、不能原樣複製」的直接迴歸測試：若程式碼退化成用
      ``dataclasses.replace`` 只改 ``status`` 而保留其餘欄位，這裡會抓到
      ``also_local_domain_first_exit`` 錯誤地維持 ``True``。
    """

    t0 = 100_000 * _HOUR_NS
    # row 0：3 小時前（第 2 小時）就已經離開 flow domain＝local domain，截止
    # 時刻是第 3 小時，這個事件在截止時刻之前，不受截斷影響。
    kept_utc_ns = t0 - 2 * _HOUR_NS
    # row 1：來源結果要到第 5 小時才離開，比截止時刻（第 3 小時）更深，短 H
    # 範圍內這件事根本沒有發生過，必須重新標記成回溯達上限。
    recapped_source_utc_ns = t0 - 5 * _HOUR_NS

    final_status = np.array(
        [bd.NATIVE_STATUS_LEFT_FLOW_DOMAIN, bd.NATIVE_STATUS_LEFT_FLOW_DOMAIN], dtype=np.int32,
    )
    final_utc_ns = np.array([kept_utc_ns, recapped_source_utc_ns], dtype=np.int64)
    departures = (
        bd.DepartureRecord(
            row_particle_index=0, status=bd.NATIVE_STATUS_LEFT_FLOW_DOMAIN,
            x_m=1.0, y_m=-1.0, z_m=-5.0, utc_ns=kept_utc_ns,
            also_local_domain_first_exit=True,
        ),
        bd.DepartureRecord(
            row_particle_index=1, status=bd.NATIVE_STATUS_LEFT_FLOW_DOMAIN,
            x_m=5.0, y_m=-5.0, z_m=-5.0, utc_ns=recapped_source_utc_ns,
            also_local_domain_first_exit=True,
        ),
    )
    result = _synthetic_result_with_local_domain_fields(
        arrival_utc_ns=t0, n_hours=6, n_particles=2,
        departures=departures, final_status=final_status, final_utc_ns=final_utc_ns,
    )
    resolved = _resolved_full_horizon(requested_horizon_days=3 / 24, effective_horizon_seconds=3 * 3600.0)

    derived, info = hd.derive_backward_driver_result(
        result, resolved=resolved, source_requested_horizon_days=6 / 24,
    )

    derived_by_row = {d.row_particle_index: d for d in derived.departures}
    # row 0：完全不受截斷影響，原樣保留（含旗標）。
    assert derived_by_row[0].status == bd.NATIVE_STATUS_LEFT_FLOW_DOMAIN
    assert derived_by_row[0].also_local_domain_first_exit is True
    assert derived_by_row[0].utc_ns == kept_utc_ns
    # row 1：終止原因改變成回溯達上限，旗標必須重新判定為 False，位置與時刻
    # 落在截止時刻（第 3 小時取樣點）。
    assert derived_by_row[1].status == bd.NATIVE_STATUS_MAX_BACKTRACK
    assert derived_by_row[1].also_local_domain_first_exit is False
    assert derived_by_row[1].utc_ns == t0 - 3 * _HOUR_NS
    assert info.newly_capped_particle_count == 1
    assert info.unchanged_particle_count == 1


def test_derive_result_directory_preserves_local_domain_departures_end_to_end(tmp_path) -> None:
    """完整走「合成結果 → 寫 npz → 目錄層級推導 → 讀回 npz」，證明 WP-23 的
    npz 讀寫欄位（另見 ``reports/WP-23-fix.md``）與本次任務 A 的截斷過濾邏輯
    串接起來後，local domain 欄位在完整鏈路上都正確——不是只在記憶體物件上
    測過就假設 npz 往返也對。"""

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    t0 = 100_000 * _HOUR_NS
    kept = bd.LocalDomainFirstExitRecord(row_particle_index=0, x_m=1.0, y_m=-1.0, z_m=-5.0, utc_ns=t0 - 2 * _HOUR_NS)
    dropped = bd.LocalDomainFirstExitRecord(row_particle_index=1, x_m=2.0, y_m=-2.0, z_m=-5.0, utc_ns=t0 - 5 * _HOUR_NS)
    result = _synthetic_result_with_local_domain_fields(
        arrival_utc_ns=t0, n_hours=6, n_particles=2, local_domain_departures=(kept, dropped),
    )
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
    assert manifest["units"]["unit-a"]["local_domain_departure_count"] == 1

    derived = load_backward_driver_result_npz(output_dir / "unit-a.driver_result.npz")
    assert len(derived.local_domain_departures) == 1
    assert derived.local_domain_departures[0].row_particle_index == 0
    assert derived.local_domain_departures[0].utc_ns == t0 - 2 * _HOUR_NS
