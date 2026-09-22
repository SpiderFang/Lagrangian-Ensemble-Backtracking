"""WP-17：native 回溯模式明確化測試（``native_runner.backtrack_horizon`` 與
``orchestrator.resolve_native_work_unit_backtrack_horizons``）。

涵蓋：

1. CLI／環境變數解析模式：預設值、只給一種來源、兩種來源一致／不一致。
2. 設定檔宣稱模式與實際模式不一致時的 fail-closed 與明確承認旗標。
3. ``resolve_unit_backtrack_horizon``：``full_horizon_from_deposition`` 不需要
   ``arrival`` 就能算出 H 天；``fixed_calendar_window`` 正確轉呼叫
   ``bed_residence.resolve_bed_residence_timing``（不重寫其邏輯，只驗證轉接）。
4. ``resolve_native_work_unit_backtrack_horizons``：
   - ``full_horizon_from_deposition`` 模式刻意不覆寫
     ``NativeWorkUnit.resolved_max_backtrack_seconds``（維持 ``None``，讓
     run-time 設定的值繼續生效，這是 WP-17 修正過的關鍵行為——最初實作曾經
     不小心用 native-plan 自己的 ``--max-backtrack-days`` 覆蓋掉 run-time
     config 的值，導致小型測試設定被靜默忽略，見本檔案
     ``test_full_horizon_mode_does_not_override_resolved_max_backtrack_seconds``）。
   - ``fixed_calendar_window`` 模式正確依到達時刻分別解析、快取重複到達時刻、
     且缺少 metadata 時 fail closed。
5. ``backward_driver.build_pre_window_deposition_result``：組出的結果形狀、
   狀態碼與位置正確。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from lagrangian_backtracking.bed_residence import (
    BED_RESIDENCE_POLICY_ID,
    BED_RESIDENCE_SAMPLING_METHOD_ID,
    BED_RESIDENCE_SAMPLING_POLICY_ID,
)
from lagrangian_backtracking.native_runner import backtrack_horizon as bh
from lagrangian_backtracking.native_runner import orchestrator as orch
from lagrangian_backtracking.native_runner.backward_driver import (
    NATIVE_STATUS_PRE_WINDOW_DEPOSITION,
    build_pre_window_deposition_result,
)
from lagrangian_backtracking.scenarios import ArrivalTime

_HOUR_NS = 3_600_000_000_000
_DAY_NS = 86_400_000_000_000
_MAX_AGE_DAYS = 90
_SEED = 20260916


def _iso_utc_ns(value: int) -> str:
    seconds, nanoseconds = divmod(value, 1_000_000_000)
    timestamp = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(seconds=seconds)
    suffix = f".{nanoseconds:09d}" if nanoseconds else ""
    return f"{timestamp:%Y-%m-%dT%H:%M:%S}{suffix}Z"


def _stratum(age_hours: int) -> int:
    total_hours = _MAX_AGE_DAYS * 24 + 1
    for index in range(50):
        if index * total_hours // 50 <= age_hours < (index + 1) * total_hours // 50:
            return index
    raise AssertionError("age_hours 超出範圍")


def _runtime_arrival(age_hours: int, *, site_id: str = "gongliao") -> ArrivalTime:
    """建立可通過 ``resolve_bed_residence_timing`` 驗證的沉底 ArrivalTime。"""

    observation_dt = datetime(2025, 10, 31, tzinfo=UTC)
    observation_ns = int(observation_dt.timestamp()) * 1_000_000_000
    deposition_ns = observation_ns - age_hours * _HOUR_NS
    return ArrivalTime(
        arrival_time_id=f"arrival-{site_id}-{age_hours}",
        study_site_id=site_id,
        time_utc_ns=deposition_ns,
        year=2025,
        season="SON",
        tide_class="spring_proxy",
        phase_or_event="strong_current_event",
        metadata={
            "observation_time_utc_ns": observation_ns,
            "observation_time_utc": _iso_utc_ns(observation_ns),
            "deposition_time_utc_ns": deposition_ns,
            "deposition_time_utc": _iso_utc_ns(deposition_ns),
            "bed_residence_age_hours": age_hours,
            "bed_residence_stratum_index": _stratum(age_hours),
            "bed_residence_policy_id": BED_RESIDENCE_POLICY_ID,
            "bed_residence_sampling_policy_id": BED_RESIDENCE_SAMPLING_POLICY_ID,
            "bed_residence_sampling_method_id": BED_RESIDENCE_SAMPLING_METHOD_ID,
            "bed_residence_sampling_seed": _SEED,
            "bed_residence_maximum_age_days": _MAX_AGE_DAYS,
            "bed_residence_design_hash": "a" * 64,
            "observation_arrival_time_id": f"observation-{age_hours}",
        },
    )


# ══════════════════════════════════════════════════════════════════════════
# 1-2. 模式解析與設定檔不一致的 fail-closed
# ══════════════════════════════════════════════════════════════════════════


def test_resolve_backtrack_mode_defaults_to_full_horizon_from_deposition() -> None:
    assert bh.resolve_backtrack_mode(cli_backtrack_mode=None, env={}) == (
        "full_horizon_from_deposition"
    )


def test_resolve_backtrack_mode_prefers_cli_and_accepts_matching_env() -> None:
    assert bh.resolve_backtrack_mode(
        cli_backtrack_mode="fixed_calendar_window",
        env={bh.BACKTRACK_MODE_ENV_VAR: "fixed_calendar_window"},
    ) == "fixed_calendar_window"


def test_resolve_backtrack_mode_rejects_cli_env_mismatch() -> None:
    with pytest.raises(bh.BacktrackHorizonError):
        bh.resolve_backtrack_mode(
            cli_backtrack_mode="full_horizon_from_deposition",
            env={bh.BACKTRACK_MODE_ENV_VAR: "fixed_calendar_window"},
        )


def test_resolve_backtrack_mode_rejects_unsupported_value() -> None:
    with pytest.raises(bh.BacktrackHorizonError):
        bh.resolve_backtrack_mode(cli_backtrack_mode="not_a_real_mode", env={})


def test_config_mode_mismatch_is_fail_closed_without_acknowledgement() -> None:
    with pytest.raises(bh.BacktrackHorizonError):
        bh.require_mode_acknowledged_if_mismatched(
            effective_backtrack_mode="full_horizon_from_deposition",
            config_declared_backtrack_mode="fixed_calendar_window",
            acknowledge_config_mode_mismatch=False,
        )


def test_config_mode_mismatch_passes_with_explicit_acknowledgement() -> None:
    bh.require_mode_acknowledged_if_mismatched(
        effective_backtrack_mode="full_horizon_from_deposition",
        config_declared_backtrack_mode="fixed_calendar_window",
        acknowledge_config_mode_mismatch=True,
    )


def test_no_config_declared_mode_is_never_a_mismatch() -> None:
    """legacy 非沉底設定沒有 backtrack_mode 可比較，不視為不一致。"""

    bh.require_mode_acknowledged_if_mismatched(
        effective_backtrack_mode="fixed_calendar_window",
        config_declared_backtrack_mode=None,
        acknowledge_config_mode_mismatch=False,
    )


# ══════════════════════════════════════════════════════════════════════════
# 3. resolve_unit_backtrack_horizon
# ══════════════════════════════════════════════════════════════════════════


def test_full_horizon_mode_does_not_require_arrival_metadata() -> None:
    resolved = bh.resolve_unit_backtrack_horizon(
        backtrack_mode="full_horizon_from_deposition",
        requested_horizon_days=30,
    )
    assert resolved.effective_horizon_seconds == 30 * 86_400.0
    assert resolved.effective_horizon_ns == 30 * _DAY_NS
    assert resolved.pre_window_deposition is False
    assert resolved.bed_residence_age_hours is None


def test_fixed_calendar_window_delegates_to_resolve_bed_residence_timing() -> None:
    """10/31 觀測、10/17 沉底（年齡 14 天）、H30 應得到 16 天有效期間。"""

    arrival = _runtime_arrival(14 * 24)
    resolved = bh.resolve_unit_backtrack_horizon(
        backtrack_mode="fixed_calendar_window",
        requested_horizon_days=30,
        arrival=arrival,
        maximum_age_days=_MAX_AGE_DAYS,
        sampling_seed=_SEED,
    )
    assert resolved.effective_horizon_ns == 16 * _DAY_NS
    assert resolved.pre_window_deposition is False
    assert resolved.bed_residence_age_hours == 14 * 24


def test_fixed_calendar_window_marks_pre_window_when_age_at_or_past_horizon() -> None:
    arrival = _runtime_arrival(30 * 24)
    resolved = bh.resolve_unit_backtrack_horizon(
        backtrack_mode="fixed_calendar_window",
        requested_horizon_days=30,
        arrival=arrival,
        maximum_age_days=_MAX_AGE_DAYS,
        sampling_seed=_SEED,
    )
    assert resolved.pre_window_deposition is True
    assert resolved.effective_horizon_ns == 0


def test_fixed_calendar_window_requires_arrival_and_config_params() -> None:
    with pytest.raises(bh.BacktrackHorizonError):
        bh.resolve_unit_backtrack_horizon(
            backtrack_mode="fixed_calendar_window", requested_horizon_days=30
        )


# ══════════════════════════════════════════════════════════════════════════
# 4. orchestrator.resolve_native_work_unit_backtrack_horizons
# ══════════════════════════════════════════════════════════════════════════


def _synthetic_unit(*, unit_id: str, arrival_time_utc_ns: int) -> orch.NativeWorkUnit:
    return orch.NativeWorkUnit(
        unit_id=unit_id,
        analysis_region_id="A",
        flow_domain_id="flow-a",
        study_site_id="gongliao",
        arrival_time_utc_ns=arrival_time_utc_ns,
        settling_velocity_mps=-0.001,
        particle_row_indices=np.array([0], dtype=np.int64),
        input_hash=f"hash-{unit_id}",
    )


def test_full_horizon_mode_does_not_override_resolved_max_backtrack_seconds() -> None:
    """關鍵回歸測試：full_horizon 模式必須保留 ``None``，讓 run-time 設定繼續生效。

    WP-17 最初實作曾經把 native-plan 自己的 ``--max-backtrack-days``（歷來只是
    缺口檢查窗口估計值，預設 30 天）烤進 ``resolved_max_backtrack_seconds``，
    結果覆蓋掉 ``LBT_NATIVE_CONFIG`` 的真正值，導致端對端測試用的 3 小時小型
    設定被靜默換成 30 天，讓 forcing_resolver 讀出範圍外的 UTC
    （見 ``tests/test_native_hour_major_provider.py`` 除錯過程）。這個測試把
    那次迴歸鎖死。
    """

    unit = _synthetic_unit(unit_id="u1", arrival_time_utc_ns=100 * _DAY_NS)
    resolved = orch.resolve_native_work_unit_backtrack_horizons(
        [unit],
        backtrack_mode="full_horizon_from_deposition",
        requested_horizon_days=30.0,
    )
    assert len(resolved) == 1
    assert resolved[0].resolved_max_backtrack_seconds is None
    assert resolved[0].pre_window_deposition is False


def test_fixed_calendar_window_resolves_per_arrival_and_caches_duplicates() -> None:
    arrival_a = _runtime_arrival(14 * 24, site_id="site-a")
    arrival_b = _runtime_arrival(45 * 24, site_id="site-b")
    unit_1 = _synthetic_unit(unit_id="u1", arrival_time_utc_ns=arrival_a.time_utc_ns)
    unit_2 = _synthetic_unit(unit_id="u2", arrival_time_utc_ns=arrival_a.time_utc_ns)
    unit_3 = _synthetic_unit(unit_id="u3", arrival_time_utc_ns=arrival_b.time_utc_ns)

    resolved = orch.resolve_native_work_unit_backtrack_horizons(
        [unit_1, unit_2, unit_3],
        backtrack_mode="fixed_calendar_window",
        requested_horizon_days=30.0,
        arrivals_by_time_utc_ns={
            arrival_a.time_utc_ns: arrival_a,
            arrival_b.time_utc_ns: arrival_b,
        },
        maximum_age_days=_MAX_AGE_DAYS,
        sampling_seed=_SEED,
    )
    by_id = {u.unit_id: u for u in resolved}
    assert by_id["u1"].resolved_max_backtrack_seconds == 16 * 86_400.0
    assert by_id["u2"].resolved_max_backtrack_seconds == 16 * 86_400.0
    assert by_id["u1"].bed_residence_age_hours == 14 * 24
    assert by_id["u3"].pre_window_deposition is True
    assert by_id["u3"].resolved_max_backtrack_seconds == 0.0


def test_fixed_calendar_window_fails_closed_without_matching_arrival() -> None:
    unit = _synthetic_unit(unit_id="u1", arrival_time_utc_ns=100 * _DAY_NS)
    with pytest.raises(orch.NativeOrchestratorError):
        orch.resolve_native_work_unit_backtrack_horizons(
            [unit],
            backtrack_mode="fixed_calendar_window",
            requested_horizon_days=30.0,
            arrivals_by_time_utc_ns={},
            maximum_age_days=_MAX_AGE_DAYS,
            sampling_seed=_SEED,
        )


# ══════════════════════════════════════════════════════════════════════════
# 5. build_pre_window_deposition_result
# ══════════════════════════════════════════════════════════════════════════


def test_build_pre_window_deposition_result_shape_and_status() -> None:
    result = build_pre_window_deposition_result(
        row_particle_index=np.array([0, 1], dtype=np.int64),
        x0_m=np.array([10.0, 20.0]),
        y0_m=np.array([-5.0, -6.0]),
        z0_m=np.array([-3.0, -4.0]),
        arrival_time_utc_ns=123_000_000_000,
    )
    assert np.all(result.final_status == NATIVE_STATUS_PRE_WINDOW_DEPOSITION)
    assert np.array_equal(result.final_x_m, [10.0, 20.0])
    assert np.array_equal(result.final_y_m, [-5.0, -6.0])
    assert np.array_equal(result.final_z_m, [-3.0, -4.0])
    assert np.all(result.final_utc_ns == 123_000_000_000)
    assert result.hours_run == 0
    assert result.trajectory_x_m.shape == (1, 2)
    assert len(result.departures) == 2
    assert all(d.status == NATIVE_STATUS_PRE_WINDOW_DEPOSITION for d in result.departures)
