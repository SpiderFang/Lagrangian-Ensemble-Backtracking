"""WP-51 B2：沉底時刻標籤寫進 per_member arrival（input-build）、manifest 驗證與身分表標籤的測試。

全部合成資料，不讀 OCM／NWW。分三段：

1. ``attach_deposition_labels``：用解析潮位（12 小時週期弦波乘 15 天大小潮包絡）驗證潮相、
   大小潮、事件旗標與 ``wave_available``，並驗證缺站／時刻不在序列軸／季節不一致 fail closed。
2. ``validate_deposition_label_metadata`` 與 ``manifests._metadata`` 的 bool 例外。
3. 以 ``test_per_member_pipeline`` 的合成 per_member fixture 驗證端到端：帶標籤的 arrival manifest 可
   由正式 loader 載入並拒絕各種竄改，``build_formal_identity_table`` 的逐粒子標籤逐筆對得回
   該成員自己的沉底標籤。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_manifests import (
    SITES,
    _initial_condition_payload,
    _material_payload,
    _receptor_payload,
    _write_json,
)
from test_per_member_pipeline import _M, _all_hours, _member_arrival_payload, _member_config

from lagrangian_backtracking import input_derivation as input_derivation_module
from lagrangian_backtracking.deposition_label_arrivals import (
    DEPOSITION_LABEL_BOOL_KEYS,
    DEPOSITION_LABEL_PROVENANCE_KEY,
    attach_deposition_labels,
    validate_deposition_label_metadata,
)
from lagrangian_backtracking.deposition_labels import (
    DEPOSITION_LABEL_METADATA_KEYS,
    DEPOSITION_LABEL_POLICY_ID,
    build_deposition_label_model,
    labels_digest,
)
from lagrangian_backtracking.manifests import _load_arrival_document, _metadata
from lagrangian_backtracking.native_runner.production_wiring import build_formal_identity_table
from lagrangian_backtracking.scenarios import ArrivalTime

_HOUR_NS = 3_600_000_000_000
_DAY_NS = 24 * _HOUR_NS
_T0 = int(datetime(2024, 6, 1, tzinfo=UTC).timestamp()) * 1_000_000_000  # 2024-06-01 00:00Z（JJA）


def _context(*, hours: int = 24 * 90, hs_values=None, wave_valid=None, speed_values=None) -> SimpleNamespace:
    """解析潮位：eta = (1 + 0.7 cos(2π t/15d)) · sin(2π t/12h)，每小時一點。

    - 包絡最大（t = 0, 15 d, 30 d ...）→ 大潮；最小（7.5 d, 22.5 d ...）→ 小潮。
    - 12 小時週期恰為整數小時：小時 ≡ 0 (mod 12) 導數最大（rising）、≡ 6 為 falling、
      ≡ 3 或 9 為高低潮（slack）。
    """

    index = np.arange(hours)
    times = _T0 + index * _HOUR_NS
    t_hours = index.astype(float)
    eta = (1.0 + 0.7 * np.cos(2 * np.pi * t_hours / (15 * 24))) * np.sin(2 * np.pi * t_hours / 12.0)
    hs = np.full(hours, 1.0) if hs_values is None else np.asarray(hs_values, dtype=float)
    speed = np.full(hours, 0.2) if speed_values is None else np.asarray(speed_values, dtype=float)
    valid = np.ones(hours, dtype=bool) if wave_valid is None else np.asarray(wave_valid, dtype=bool)
    return SimpleNamespace(
        elevation={int(t): float(v) for t, v in zip(times, eta, strict=True)},
        speed={int(t): float(v) for t, v in zip(times, speed, strict=True)},
        nww_series={int(t): float(v) for t, v in zip(times, hs, strict=True)},
        nww_valid={int(t): bool(v) for t, v in zip(times, valid, strict=True)},
    )


def _arrival(site: str, hour: int, *, season: str = "JJA", name: str | None = None) -> ArrivalTime:
    return ArrivalTime(
        arrival_time_id=name or f"{site}-h{hour}",
        study_site_id=site,
        time_utc_ns=_T0 + hour * _HOUR_NS,
        year=2024,
        season=season,
        tide_class="spring_proxy",  # 觀測標籤，必須不影響沉底標籤
        phase_or_event="fastest_rising",
        metadata={"observation_season": "SON"},
    )


# ── 一、attach_deposition_labels ─────────────────────────────────────────


def test_attach_labels_tide_class_and_phase_follow_deposition_time() -> None:
    context = _context()
    arrivals = [
        _arrival("S", 15 * 24),  # 包絡最大、小時 ≡ 0 (mod 12) → 大潮、rising
        _arrival("S", 15 * 24 // 2),  # 7.5 d：包絡最小、≡ 0 → 小潮、rising
        _arrival("S", 15 * 24 + 6),  # 大潮、falling
        _arrival("S", 15 * 24 + 3),  # 大潮、高潮 slack
        _arrival("S", 15 * 24 // 2 + 9),  # 小潮、slack
    ]
    provenance = attach_deposition_labels({"S": arrivals}, {"S": context})
    got = [
        (a.metadata["deposition_tide_class"], a.metadata["deposition_tide_phase"]) for a in arrivals
    ]
    assert got == [
        ("spring_proxy", "rising"),
        ("neap_proxy", "rising"),
        ("spring_proxy", "falling"),
        ("spring_proxy", "slack"),
        ("neap_proxy", "slack"),
    ]
    # 觀測時刻標籤保持原樣（沒有被標註函式改寫）
    assert all(a.tide_class == "spring_proxy" and a.phase_or_event == "fastest_rising" for a in arrivals)
    for item in arrivals:
        assert set(DEPOSITION_LABEL_METADATA_KEYS) <= set(item.metadata)
        assert item.metadata["deposition_label_policy"] == DEPOSITION_LABEL_POLICY_ID
        assert validate_deposition_label_metadata(item.metadata, label="x") is True
    assert provenance["policy_id"] == DEPOSITION_LABEL_POLICY_ID
    assert [site["study_site_id"] for site in provenance["sites"]] == ["S"]


def test_attach_labels_event_flags_use_site_series_quantiles_and_wave_availability() -> None:
    hours = 24 * 90
    rng = np.random.default_rng(3)
    hs = rng.uniform(0.5, 2.0, size=hours)
    speed = rng.uniform(0.05, 0.6, size=hours)
    valid = np.ones(hours, dtype=bool)
    valid[100] = False  # NWW 在第 100 小時無有效值
    big_hs_hour, big_speed_hour = 700, 900
    hs[big_hs_hour], speed[big_speed_hour] = 9.0, 5.0
    hs[100] = 99.0  # 無效時刻的數值不得被當成高浪
    context = _context(hours=hours, hs_values=hs, wave_valid=valid, speed_values=speed)
    arrivals = [_arrival("S", h) for h in (big_hs_hour, big_speed_hour, 100, 1200)]
    attach_deposition_labels({"S": arrivals}, {"S": context})
    by_hour = {h: a.metadata for h, a in zip((big_hs_hour, big_speed_hour, 100, 1200), arrivals, strict=True)}
    assert by_hour[big_hs_hour]["deposition_high_wave"] is True
    assert by_hour[big_hs_hour]["deposition_wave_available"] is True
    assert by_hour[big_speed_hour]["deposition_strong_current"] is True
    assert by_hour[100]["deposition_wave_available"] is False
    assert by_hour[100]["deposition_high_wave"] is False
    # 獨立期望：門檻是整段序列有限值的 0.95 分位數，嚴格大於才算事件
    finite_hs = hs[valid]
    assert by_hour[1200]["deposition_high_wave"] == bool(hs[1200] > np.quantile(finite_hs, 0.95))
    assert by_hour[1200]["deposition_strong_current"] == bool(speed[1200] > np.quantile(speed, 0.95))


def test_attach_labels_matches_label_model_for_every_arrival_and_digest_is_stable() -> None:
    context = _context()
    arrivals = [_arrival("S", h) for h in range(0, 24 * 80, 17)]
    provenance = attach_deposition_labels({"S": arrivals}, {"S": context})
    model = build_deposition_label_model(
        study_site_id="S",
        elevation=context.elevation,
        significant_wave_height=context.nww_series,
        current_speed=context.speed,
        wave_valid=context.nww_valid,
    )
    for item in arrivals:
        labels = model.labels_at(item.time_utc_ns)
        assert item.metadata["deposition_tide_class"] == labels.tide_class
        assert item.metadata["deposition_tide_phase"] == labels.tide_phase
        assert item.metadata["deposition_high_wave"] is labels.high_wave
        assert item.metadata["deposition_strong_current"] is labels.strong_current
    assert provenance["labels_sha256"] == labels_digest([model])
    again = attach_deposition_labels({"S": [_arrival("S", 5)]}, {"S": context})
    assert again == provenance, "同一組序列、不同 arrival 集合，模型描述（與雜湊）必須相同"


def test_attach_labels_fails_closed() -> None:
    context = _context()
    with pytest.raises(ValueError, match="缺少站點"):
        attach_deposition_labels({"S": [_arrival("S", 5)], "T": [_arrival("T", 5)]}, {"S": context})
    with pytest.raises(ValueError, match="不在標籤序列|超出標籤序列"):
        attach_deposition_labels({"S": [_arrival("S", 24 * 90 + 10)]}, {"S": context})
    with pytest.raises(ValueError, match="不在標籤序列"):  # 不在整點時間軸上
        late = _arrival("S", 5)
        shifted = ArrivalTime(**{**asdict(late), "time_utc_ns": late.time_utc_ns + 1})
        attach_deposition_labels({"S": [shifted]}, {"S": context})
    with pytest.raises(ValueError, match="季節"):
        attach_deposition_labels({"S": [_arrival("S", 5, season="DJF")]}, {"S": context})
    with pytest.raises(ValueError, match="站點"):
        attach_deposition_labels({"S": [_arrival("T", 5)]}, {"S": context})


# ── 二、metadata 驗證 ─────────────────────────────────────────────────────


def _full_metadata() -> dict[str, object]:
    return {
        "deposition_tide_class": "neap_proxy",
        "deposition_tide_phase": "slack",
        "deposition_high_wave": False,
        "deposition_strong_current": True,
        "deposition_wave_available": True,
        "deposition_label_policy": DEPOSITION_LABEL_POLICY_ID,
    }


def test_validate_label_metadata_all_or_nothing() -> None:
    assert validate_deposition_label_metadata({"x": 1}, label="r") is False
    assert validate_deposition_label_metadata(_full_metadata(), label="r") is True
    partial = _full_metadata()
    del partial["deposition_tide_phase"]
    with pytest.raises(ValueError, match="不完整"):
        validate_deposition_label_metadata(partial, label="r")


@pytest.mark.parametrize(
    ("key", "value", "pattern"),
    [
        ("deposition_tide_class", "unknown_class", "tide_class"),
        ("deposition_tide_phase", "fastest_rising", "tide_phase"),
        ("deposition_high_wave", 1, "bool"),
        ("deposition_strong_current", "true", "bool"),
        ("deposition_label_policy", "deposition_time_labels_v0", "政策|policy|版本"),
    ],
)
def test_validate_label_metadata_rejects_bad_values(key: str, value: object, pattern: str) -> None:
    metadata = _full_metadata()
    metadata[key] = value
    with pytest.raises(ValueError, match=pattern):
        validate_deposition_label_metadata(metadata, label="r")


def test_validate_label_metadata_rejects_high_wave_without_wave() -> None:
    metadata = _full_metadata()
    metadata.update({"deposition_high_wave": True, "deposition_wave_available": False})
    with pytest.raises(ValueError, match="high_wave"):
        validate_deposition_label_metadata(metadata, label="r")


def test_manifest_metadata_allows_bool_only_for_label_keys() -> None:
    cleaned = _metadata({**_full_metadata(), "season_note": "x"}, "m")
    assert cleaned["deposition_high_wave"] is False and cleaned["deposition_strong_current"] is True
    assert {
        "deposition_high_wave",
        "deposition_strong_current",
        "deposition_wave_available",
    } == DEPOSITION_LABEL_BOOL_KEYS
    with pytest.raises(ValueError):
        _metadata({"some_other_flag": True}, "m")
    with pytest.raises(ValueError):
        _metadata({"deposition_tide_class": None}, "m")


# ── 三、端到端：per_member arrival manifest → loader → 身分表標籤 ───────────


def _context_for_site(site_index: int) -> SimpleNamespace:
    """涵蓋 ``_all_hours()`` 整段（2023-06 ～ 2026-01）的逐站合成序列；各站相位／強度略有不同。"""

    hours = _all_hours()
    index = np.arange(len(hours), dtype=float)
    rng = np.random.default_rng(100 + site_index)
    eta = (1.0 + 0.6 * np.cos(2 * np.pi * (index + 11 * site_index) / 354.0)) * np.sin(
        2 * np.pi * index / 12.4 + site_index
    )
    hs = rng.gamma(2.0, 0.5, size=len(hours))
    speed = rng.gamma(2.0, 0.1, size=len(hours))
    valid = rng.uniform(size=len(hours)) > 0.03  # 約 3% NWW 缺值
    return SimpleNamespace(
        elevation={int(t): float(v) for t, v in zip(hours, eta, strict=True)},
        speed={int(t): float(v) for t, v in zip(hours, speed, strict=True)},
        nww_series={int(t): float(v) for t, v in zip(hours, hs, strict=True)},
        nww_valid={int(t): bool(v) for t, v in zip(hours, valid, strict=True)},
    )


@pytest.fixture(scope="module")
def labelled_member_dir(tmp_path_factory: pytest.TempPathFactory):
    """per_member 合成 manifest 組＋以各站合成序列算出的沉底標籤（metadata 與 root provenance）。"""

    tmp_path = tmp_path_factory.mktemp("labelled_member")
    config = _member_config()
    payload = _member_arrival_payload(config)
    records = [ArrivalTime(**record) for record in payload["records"]]
    by_site: dict[str, list[ArrivalTime]] = {site: [] for site in SITES}
    for item in records:
        by_site[item.study_site_id].append(item)
    contexts = {site: _context_for_site(index) for index, site in enumerate(sorted(SITES))}
    provenance = attach_deposition_labels(by_site, contexts)
    payload["records"] = [asdict(item) for item in records]
    payload["provenance"][DEPOSITION_LABEL_PROVENANCE_KEY] = provenance
    receptor = _receptor_payload()
    material = _material_payload()
    material["design_version"] = config.design_version
    receptor["design_version"] = config.design_version
    (tmp_path / "config.yaml").write_text("placeholder: true\n", encoding="utf-8")
    _write_json(tmp_path / "material.json", material)
    _write_json(tmp_path / "receptor.json", receptor)
    _write_json(tmp_path / "arrival.json", payload)
    _write_json(
        tmp_path / "initial_conditions.json",
        _initial_condition_payload(config, receptors=receptor, arrivals=payload),
    )
    return tmp_path, config, payload, contexts


def _mutated(tmp_path: Path, payload: dict[str, object], mutate) -> Path:
    clone = deepcopy(payload)
    mutate(clone)
    path = tmp_path / "mutated-labelled-arrival.json"
    _write_json(path, clone)
    return path


def test_labelled_arrival_manifest_loads_and_labels_survive_json(labelled_member_dir) -> None:
    tmp_path, config, _payload, _contexts = labelled_member_dir
    arrivals, _ = _load_arrival_document(tmp_path / "arrival.json", config, formal=True)
    assert len(arrivals) == 250 * _M
    for item in arrivals:
        assert type(item.metadata["deposition_high_wave"]) is bool
        assert item.metadata["deposition_label_policy"] == DEPOSITION_LABEL_POLICY_ID
    # 沉底時刻不同於觀測時刻，所以兩套標籤通常不相同（至少有一部分粒子不同）
    differing = sum(
        item.metadata["deposition_tide_class"] != item.tide_class for item in arrivals
    )
    assert differing > 0


@pytest.mark.parametrize(
    "name",
    [
        "drop_one_key",
        "bad_tide_value",
        "mixed_presence",
        "missing_provenance",
        "wrong_policy_in_provenance",
        "provenance_missing_site",
        "int_instead_of_bool",
    ],
)
def test_labelled_arrival_manifest_rejects_inconsistent_labels(labelled_member_dir, name: str) -> None:
    tmp_path, config, payload, _contexts = labelled_member_dir

    def mutate(clone: dict[str, object]) -> None:
        records = clone["records"]
        if name == "drop_one_key":
            del records[7]["metadata"]["deposition_tide_phase"]
        elif name == "bad_tide_value":
            records[7]["metadata"]["deposition_tide_class"] = "not_a_class"
        elif name == "mixed_presence":
            for key in DEPOSITION_LABEL_METADATA_KEYS:
                del records[0]["metadata"][key]
        elif name == "missing_provenance":
            del clone["provenance"][DEPOSITION_LABEL_PROVENANCE_KEY]
        elif name == "wrong_policy_in_provenance":
            clone["provenance"][DEPOSITION_LABEL_PROVENANCE_KEY]["policy_id"] = "other"
        elif name == "provenance_missing_site":
            clone["provenance"][DEPOSITION_LABEL_PROVENANCE_KEY]["sites"].pop()
        elif name == "int_instead_of_bool":
            records[7]["metadata"]["deposition_high_wave"] = 1

    with pytest.raises(ValueError):
        _load_arrival_document(_mutated(tmp_path, payload, mutate), config, formal=True)


def test_arrival_manifest_without_labels_still_loads(labelled_member_dir) -> None:
    """標籤是選用的：整批省略（舊 per_member 產物）時 loader 照常接受。"""

    tmp_path, config, payload, _contexts = labelled_member_dir

    def mutate(clone: dict[str, object]) -> None:
        for record in clone["records"]:
            for key in DEPOSITION_LABEL_METADATA_KEYS:
                record["metadata"].pop(key, None)
        clone["provenance"].pop(DEPOSITION_LABEL_PROVENANCE_KEY, None)

    arrivals, _ = _load_arrival_document(_mutated(tmp_path, payload, mutate), config, formal=True)
    assert len(arrivals) == 250 * _M


def test_identity_table_labels_equal_each_members_own_deposition_labels(labelled_member_dir) -> None:
    tmp_path, config, payload, contexts = labelled_member_dir
    table = build_formal_identity_table(
        config, members_per_scenario=_M, config_path=tmp_path / "config.yaml", formal=False
    )
    labels = table.labels
    assert labels is not None
    for name in (
        "deposition_tide_class",
        "deposition_tide_phase",
        "deposition_high_wave",
        "deposition_strong_current",
        "deposition_wave_available",
    ):
        assert name in labels and labels[name].shape == (len(table),)
    assert labels["deposition_high_wave"].dtype == np.bool_
    assert labels["deposition_tide_class"].dtype.kind == "U"
    # 逐粒子抽查：標籤等於以該粒子「自己的沉底時刻」直接查標註模型的結果
    models = {
        site: build_deposition_label_model(
            study_site_id=site,
            elevation=ctx.elevation,
            significant_wave_height=ctx.nww_series,
            current_speed=ctx.speed,
            wave_valid=ctx.nww_valid,
        )
        for site, ctx in contexts.items()
    }
    rng = np.random.default_rng(11)
    for row in rng.choice(len(table), size=300, replace=False):
        expected = models[str(table.study_site_id[row])].labels_at(int(table.arrival_time_utc_ns[row]))
        assert str(labels["deposition_tide_class"][row]) == expected.tide_class
        assert str(labels["deposition_tide_phase"][row]) == expected.tide_phase
        assert bool(labels["deposition_high_wave"][row]) is expected.high_wave
        assert bool(labels["deposition_strong_current"][row]) is expected.strong_current
        assert bool(labels["deposition_wave_available"][row]) is expected.wave_available
        assert str(labels["deposition_season"][row]) == expected.season
    # 同一情境的 M 個成員沉底時刻不同，標籤分佈在全表上不是單一值
    assert len(set(labels["deposition_tide_class"].tolist())) >= 2
    assert len(set(labels["deposition_tide_phase"].tolist())) >= 2


def test_arrival_payload_injects_label_provenance(labelled_member_dir) -> None:
    """``input_derivation._arrival_payload`` 把標註模型描述寫進 root provenance（只在 per_member）。"""

    tmp_path, config, payload, _contexts = labelled_member_dir
    arrivals, _ = _load_arrival_document(tmp_path / "arrival.json", config, formal=True)
    label_provenance = payload["provenance"][DEPOSITION_LABEL_PROVENANCE_KEY]
    built = input_derivation_module._arrival_payload(
        config,
        arrivals,
        source_hashes={"x": "0" * 64},
        strict=True,
        horizon_settings=SimpleNamespace(
            is_generic=False,
            bed_residence_enabled=True,
            selection_support_days=180,
            runtime_support_days=90,
        ),
        deposition_label_provenance=label_provenance,
    )
    assert built["provenance"][DEPOSITION_LABEL_PROVENANCE_KEY] == label_provenance
    without = input_derivation_module._arrival_payload(
        config,
        arrivals,
        source_hashes={"x": "0" * 64},
        strict=True,
        horizon_settings=SimpleNamespace(
            is_generic=False,
            bed_residence_enabled=True,
            selection_support_days=180,
            runtime_support_days=90,
        ),
    )
    assert DEPOSITION_LABEL_PROVENANCE_KEY not in without["provenance"]
