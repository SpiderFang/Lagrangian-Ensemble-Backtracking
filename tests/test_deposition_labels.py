"""WP-51 B：沉底時刻標籤（``deposition_labels``）的單元測試。

用已知解的合成潮位（M2 週期 12.42 h 乘上 14.77 日大小潮包絡）驗證：大小潮分類跟隨包絡、
潮相三類各約三分之一且符號正確、事件旗標依分位數門檻、缺值不被當成「沒有大浪」、
季節與 (年, 季節) 格、時刻不在序列軸上時 fail-closed、provenance 可序列化且決定性。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from lagrangian_backtracking.deposition_labels import (
    DEFAULT_EVENT_QUANTILE,
    DEPOSITION_LABEL_POLICY_ID,
    DepositionLabelModel,
    build_deposition_label_model,
    labels_digest,
    season_for_utc_ns,
)

_HOUR_NS = 3_600_000_000_000
_START = int(np.datetime64("2024-09-01T00", "ns").astype(np.int64))
_M2_HOURS = 12.42
_SPRING_NEAP_DAYS = 14.77


def _series(days: int = 200):
    n = days * 24
    t_h = np.arange(n, dtype=np.float64)
    envelope = 1.0 + 0.5 * np.cos(2.0 * np.pi * t_h / (_SPRING_NEAP_DAYS * 24.0))
    eta = envelope * np.cos(2.0 * np.pi * t_h / _M2_HOURS)
    times = _START + np.arange(n, dtype=np.int64) * _HOUR_NS
    return times, t_h, envelope, eta


def _model(**overrides) -> tuple[DepositionLabelModel, np.ndarray, np.ndarray, np.ndarray]:
    times, t_h, envelope, eta = _series()
    rng = np.random.default_rng(3)
    hs = 1.0 + 0.2 * rng.random(times.size)
    speed = 0.3 + 0.1 * rng.random(times.size)
    kwargs = dict(
        study_site_id="site_x", time_utc_ns=times, elevation_m=eta,
        significant_wave_height_m=hs, current_speed_mps=speed,
    )
    kwargs.update(overrides)
    return DepositionLabelModel(**kwargs), times, envelope, eta


def test_tide_class_follows_spring_neap_envelope() -> None:
    model, times, envelope, _eta = _model()
    labels = [model.labels_at(int(t)) for t in times[200:-200]]
    spring = np.array([item.tide_class == "spring_proxy" for item in labels])
    assert all(item.tide_class in {"spring_proxy", "neap_proxy"} for item in labels)
    env = envelope[200:-200]
    # 包絡高於 1.25（大潮半週期）時幾乎都是 spring_proxy，低於 0.75 時幾乎都是 neap_proxy
    assert spring[env > 1.25].mean() > 0.9
    assert spring[env < 0.75].mean() < 0.1
    # 同一 (年, 季節) 格內大約一半一半
    assert 0.35 < spring.mean() < 0.65


def test_tide_phase_signs_and_roughly_equal_thirds() -> None:
    model, times, _env, eta = _model()
    derivative = np.gradient(eta, 3600.0)
    sl = slice(100, -100)
    phases = np.array([model.labels_at(int(t)).tide_phase for t in times[sl]])
    assert set(phases) == {"rising", "falling", "slack"}
    d = derivative[sl]
    assert np.all(d[phases == "rising"] > 0)
    assert np.all(d[phases == "falling"] < 0)
    for name in ("rising", "falling", "slack"):
        assert 0.27 < np.mean(phases == name) < 0.40, name
    # 高潮與低潮附近（導數趨近 0）是 slack
    high_water = np.flatnonzero((np.diff(np.sign(derivative)) < 0))[10]
    assert model.labels_at(int(times[high_water])).tide_phase == "slack"


def test_phase_rule_threshold_parameter() -> None:
    narrow, times, _e, _eta = _model(slack_fraction=0.1)
    wide, _t, _e2, _eta2 = _model(slack_fraction=0.9)
    sl = slice(100, -100)
    narrow_slack = np.mean([narrow.labels_at(int(t)).tide_phase == "slack" for t in times[sl]])
    wide_slack = np.mean([wide.labels_at(int(t)).tide_phase == "slack" for t in times[sl]])
    assert narrow_slack < 0.15 < 0.6 < wide_slack


def test_event_flags_use_quantile_thresholds_and_missing_wave_is_explicit() -> None:
    times, _t, _env, eta = _series()
    hs = np.full(times.size, 1.0)
    speed = np.full(times.size, 0.3)
    hs[1000] = 6.0
    speed[2000] = 2.0
    hs[1500] = np.nan  # NWW 缺值
    model = DepositionLabelModel(
        study_site_id="s", time_utc_ns=times, elevation_m=eta,
        significant_wave_height_m=hs, current_speed_mps=speed, event_quantile=0.999,
    )
    peak = model.labels_at(int(times[1000]))
    assert peak.high_wave and peak.wave_available
    assert not model.labels_at(int(times[1001])).high_wave
    assert model.labels_at(int(times[2000])).strong_current
    assert not model.labels_at(int(times[2001])).strong_current
    missing = model.labels_at(int(times[1500]))
    assert not missing.wave_available and not missing.high_wave
    assert model.high_wave_threshold_m >= 1.0
    # 預設分位數
    assert DEFAULT_EVENT_QUANTILE == 0.95


def test_event_share_matches_quantile_for_default_threshold() -> None:
    model, times, _env, _eta = _model()
    flags = np.array([model.labels_at(int(t)).high_wave for t in times])
    assert 0.03 < flags.mean() < 0.07


def test_season_and_year_season_cell() -> None:
    assert season_for_utc_ns(int(np.datetime64("2024-12-15T00", "ns").astype(np.int64))) == "DJF"
    assert season_for_utc_ns(int(np.datetime64("2025-01-15T00", "ns").astype(np.int64))) == "DJF"
    assert season_for_utc_ns(int(np.datetime64("2025-04-01T00", "ns").astype(np.int64))) == "MAM"
    assert season_for_utc_ns(int(np.datetime64("2025-07-31T23", "ns").astype(np.int64))) == "JJA"
    assert season_for_utc_ns(int(np.datetime64("2025-10-01T00", "ns").astype(np.int64))) == "SON"
    model, times, _env, _eta = _model()
    medians = {(item["year"], item["season"]) for item in model.provenance()["tidal_strength_cell_median_m"]}
    # 日曆年分格：2024 年 12 月與 2025 年 1–2 月屬於不同的 DJF 格
    assert (2024, "DJF") in medians and (2025, "DJF") in medians and (2024, "SON") in medians
    dec = int(np.datetime64("2024-12-10T00", "ns").astype(np.int64))
    assert model.labels_at(dec).season == "DJF"


def test_time_off_axis_fails_closed_and_gap_is_unknown() -> None:
    times, _t, _env, eta = _series()
    keep = np.ones(times.size, dtype=bool)
    keep[1000:1040] = False  # 40 小時缺口
    rng = np.random.default_rng(1)
    model = DepositionLabelModel(
        study_site_id="s", time_utc_ns=times[keep], elevation_m=eta[keep],
        significant_wave_height_m=1.0 + rng.random(keep.sum()), current_speed_mps=0.3 + rng.random(keep.sum()),
    )
    with pytest.raises(ValueError, match="規則時間軸"):
        model.labels_at(int(times[500]) + 1)
    with pytest.raises(ValueError, match="規則時間軸|超出"):
        model.labels_at(int(times[-1]) + 10 * _HOUR_NS)
    with pytest.raises(ValueError, match="規則時間軸"):
        model.labels_at(int(times[0]) - _HOUR_NS)
    inside_gap = model.labels_at(int(times[1020]))
    assert inside_gap.tide_class == "unknown" or inside_gap.tide_phase == "unknown"
    assert inside_gap.tide_phase == "unknown" and not inside_gap.wave_available
    # 遠離缺口的時刻仍有正常標籤
    assert model.labels_at(int(times[3000])).tide_phase in {"rising", "falling", "slack"}


def test_invalid_arguments_fail_closed() -> None:
    times, _t, _env, eta = _series(20)
    ones = np.ones(times.size)
    with pytest.raises(ValueError, match="嚴格遞增"):
        DepositionLabelModel(study_site_id="s", time_utc_ns=times[::-1], elevation_m=eta,
                             significant_wave_height_m=ones, current_speed_mps=ones)
    with pytest.raises(ValueError, match="shape"):
        DepositionLabelModel(study_site_id="s", time_utc_ns=times, elevation_m=eta[:-1],
                             significant_wave_height_m=ones, current_speed_mps=ones)
    for key, value in (("slack_fraction", 0.0), ("slack_fraction", 1.0), ("event_quantile", 1.0),
                       ("tidal_strength_window_hours", 0.0)):
        with pytest.raises(ValueError):
            DepositionLabelModel(study_site_id="s", time_utc_ns=times, elevation_m=eta,
                                 significant_wave_height_m=ones, current_speed_mps=ones, **{key: value})
    with pytest.raises(ValueError, match="有限"):
        DepositionLabelModel(study_site_id="s", time_utc_ns=times, elevation_m=eta,
                             significant_wave_height_m=np.full(times.size, np.nan), current_speed_mps=ones)


def test_build_from_mappings_matches_array_constructor_and_wave_valid_mask() -> None:
    times, _t, _env, eta = _series(60)
    rng = np.random.default_rng(5)
    hs = 1.0 + rng.random(times.size)
    speed = 0.3 + rng.random(times.size)
    mapping = lambda values: {int(t): float(v) for t, v in zip(times, values, strict=True)}  # noqa: E731
    valid = {int(t): bool(i % 7) for i, t in enumerate(times)}
    from_map = build_deposition_label_model(
        study_site_id="s", elevation=mapping(eta), significant_wave_height=mapping(hs),
        current_speed=mapping(speed), wave_valid=valid,
    )
    masked_hs = np.where([valid[int(t)] for t in times], hs, np.nan)
    direct = DepositionLabelModel(
        study_site_id="s", time_utc_ns=times, elevation_m=eta,
        significant_wave_height_m=masked_hs, current_speed_mps=speed,
    )
    for index in (100, 400, 700, 1000):
        assert from_map.labels_at(int(times[index])) == direct.labels_at(int(times[index]))
    assert not from_map.labels_at(int(times[7])).wave_available  # i % 7 == 0 → 無效


def test_provenance_is_json_serialisable_and_deterministic() -> None:
    model_a, *_ = _model()
    model_b, *_ = _model()
    payload = model_a.provenance()
    json.dumps(payload, allow_nan=False)
    assert payload["policy_id"] == DEPOSITION_LABEL_POLICY_ID
    assert payload["tidal_strength_window_hours"] == 25.0
    assert payload["tidal_strength_window_samples"] == 25
    assert labels_digest([model_a]) == labels_digest([model_b])
    other, *_ = _model(event_quantile=0.9)
    assert labels_digest([other]) != labels_digest([model_a])
